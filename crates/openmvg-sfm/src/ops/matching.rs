// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matching stage: run a [`PairMatcher`] over the selected pairs.

use crate::backends::PairMatcherPreparation;
use crate::backends::{PairContext, PairMatcher};
use crate::determinism::PAIR_MATCHING_SEED_DOMAIN;
use crate::ops::resolve_pairs;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::two_view::PairMatches;
use crate::workspace::selector::PairSelector;
use crate::{PairId, Result, SfmError, SfmWorkspace};

/// Match features for the selected pairs.
///
/// Each pair is matched independently, so under the `parallel` feature the pairs are
/// fanned out across a rayon pool (matching is the second front-end bottleneck after
/// detection). Every match only reads shared feature/pair-graph state, so the fan-out
/// borrows the workspace immutably; results are gathered owned and written into the
/// (per-pair) store slots sequentially afterwards, keeping the output deterministic.
pub fn match_pairs(
    workspace: &mut SfmWorkspace,
    matcher: &dyn PairMatcher,
    selector: PairSelector,
) -> Result<()> {
    match_pairs_with_progress(workspace, matcher, selector, &())
}

/// [`match_pairs`], reporting [`SfmProgressStage::Matching`] in pairs. Matchers that build
/// per-image state first report [`SfmProgressStage::MatcherPreparation`] in images.
pub fn match_pairs_with_progress(
    workspace: &mut SfmWorkspace,
    matcher: &dyn PairMatcher,
    selector: PairSelector,
    progress: &dyn SfmProgressSink,
) -> Result<()> {
    if workspace.is_matching_data_released() {
        return Err(SfmError::MatchingDataReleased);
    }
    let pair_ids = resolve_pairs(workspace, selector)?;
    cancellation_point(progress)?;
    if pair_ids.is_empty() {
        progress.begin(SfmProgressStage::Matching, 0);
        progress.finish(SfmProgressStage::Matching);
        return Ok(());
    }

    let mut pairs = Vec::with_capacity(pair_ids.len());
    for &pair_id in &pair_ids {
        let pair = workspace
            .pair_graph()
            .pair(pair_id)
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
        pairs.push(pair);
    }

    // Preparation builds per-image indexes, so it runs inside the same bounded
    // pool as the pair fan-out.
    let ws: &SfmWorkspace = workspace;
    let run = || -> Result<Vec<(usize, PairMatches)>> {
        let prepared = matcher.prepare(PairMatcherPreparation::new(
            ws.features_view(),
            &pairs,
            ws.root_seed(),
            progress,
        ))?;
        cancellation_point(progress)?;
        progress.begin(SfmProgressStage::Matching, pair_ids.len());

        let match_one = |pair_id: PairId| -> Result<(usize, PairMatches)> {
            cancellation_point(progress)?;
            let pair = ws.pair_graph().pair(pair_id).ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
            let features_i = ws.feature_store().get(pair.first().index()).ok_or(
                SfmError::MissingPrerequisite {
                    prerequisite: "features",
                    method: "extract_features",
                },
            )?;
            let features_j = ws.feature_store().get(pair.second().index()).ok_or(
                SfmError::MissingPrerequisite {
                    prerequisite: "features",
                    method: "extract_features",
                },
            )?;
            let ctx = PairContext {
                pair,
                view_i: pair.first(),
                view_j: pair.second(),
                root_seed: ws.root_seed(),
                seed: ws.derive_seed(
                    PAIR_MATCHING_SEED_DOMAIN,
                    &[pair.first().value(), pair.second().value()],
                ),
            };
            let matches = if let Some(prepared) = prepared.as_deref() {
                prepared.match_pair(&ctx)?
            } else {
                matcher.match_features(&ctx, features_i, features_j)?
            };
            matches.validate_for_pair(pair, features_i.len(), features_j.len())?;
            progress.advance(SfmProgressStage::Matching, 1);
            Ok((pair_id.index(), matches))
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            pair_ids
                .par_iter()
                .map(|&pair_id| match_one(pair_id))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            pair_ids.iter().map(|&pair_id| match_one(pair_id)).collect()
        }
    };

    #[cfg(feature = "parallel")]
    let results = {
        // Matching is memory-light (descriptors are already resident; per-pair buffers
        // are small), so it can use all cores — the 0.5 GiB/worker budget only kicks in
        // to throttle if the machine is genuinely low on RAM.
        let workers = crate::ops::bounded_workers(
            pair_ids.len(),
            0.5,
            usize::MAX,
            ws.max_parallel_workers().map(|workers| workers.get()),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| SfmError::InvalidConfig(format!("rayon pool: {e}")))?;
        pool.install(run)?
    };
    #[cfg(not(feature = "parallel"))]
    let results = run()?;

    for (index, matches) in results {
        workspace.matches.set(index, matches)?;
    }
    progress.finish(SfmProgressStage::Matching);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::backends::PreparedPairMatcher;
    use crate::model::ImageMeta;
    use crate::progress::testing::{ProgressRecord, RecordingProgress};
    use crate::{Camera, FeatureId, IntrinsicId};
    use openmvg_features::model::{DescriptorSet, ExtractedFeatures, FeatureMatch, Keypoint};

    #[cfg(feature = "parallel")]
    use crate::adapters::matching::brute_force::RootSiftU8Matcher;
    #[cfg(feature = "parallel")]
    use crate::adapters::matching::{MatchAcceptance, MatcherType, PairMatcherAdapter};

    struct PreparedCountingMatcher {
        prepare_calls: AtomicUsize,
        prepared_calls: AtomicUsize,
        direct_calls: AtomicUsize,
        prepared_pairs: Mutex<Vec<(usize, usize)>>,
    }

    struct PreparedCounter<'a> {
        calls: &'a AtomicUsize,
        features: crate::workspace::views::FeaturesView<'a>,
    }

    impl PreparedPairMatcher for PreparedCounter<'_> {
        fn match_pair(&self, ctx: &PairContext) -> Result<PairMatches> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.features.get(ctx.view_i).is_none() || self.features.get(ctx.view_j).is_none() {
                return Err(SfmError::MissingPrerequisite {
                    prerequisite: "features",
                    method: "extract_features",
                });
            }
            PairMatches::try_new(ctx.pair, Vec::new())
        }
    }

    impl PairMatcher for PreparedCountingMatcher {
        fn prepare<'a>(
            &'a self,
            ctx: PairMatcherPreparation<'a>,
        ) -> Result<Option<Box<dyn PreparedPairMatcher + 'a>>> {
            self.prepare_calls.fetch_add(1, Ordering::Relaxed);
            *self.prepared_pairs.lock().unwrap() = ctx
                .pairs()
                .iter()
                .map(|pair| (pair.first().index(), pair.second().index()))
                .collect();
            Ok(Some(Box::new(PreparedCounter {
                calls: &self.prepared_calls,
                features: ctx.features(),
            })))
        }

        fn match_features(
            &self,
            ctx: &PairContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
        ) -> Result<PairMatches> {
            self.direct_calls.fetch_add(1, Ordering::Relaxed);
            PairMatches::try_new(ctx.pair, Vec::new())
        }
    }

    struct DirectCountingMatcher(AtomicUsize);

    impl PairMatcher for DirectCountingMatcher {
        fn match_features(
            &self,
            ctx: &PairContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
        ) -> Result<PairMatches> {
            self.0.fetch_add(1, Ordering::Relaxed);
            PairMatches::try_new(ctx.pair, Vec::new())
        }
    }

    struct FailingPreparation;

    impl PairMatcher for FailingPreparation {
        fn prepare<'a>(
            &'a self,
            _ctx: PairMatcherPreparation<'a>,
        ) -> Result<Option<Box<dyn PreparedPairMatcher + 'a>>> {
            Err(SfmError::InvalidInput("preparation failed".to_owned()))
        }

        fn match_features(
            &self,
            _ctx: &PairContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
        ) -> Result<PairMatches> {
            unreachable!("failed preparation must stop before pair matching")
        }
    }

    struct WrongPairMatcher;

    impl PairMatcher for WrongPairMatcher {
        fn match_features(
            &self,
            ctx: &PairContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
        ) -> Result<PairMatches> {
            PairMatches::try_new(
                crate::ViewPair::new(ctx.view_i, crate::ViewId::new(2)),
                Vec::new(),
            )
        }
    }

    struct OutOfRangeMatcher;

    impl PairMatcher for OutOfRangeMatcher {
        fn match_features(
            &self,
            ctx: &PairContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
        ) -> Result<PairMatches> {
            PairMatches::try_new(
                ctx.pair,
                vec![FeatureMatch::try_new(
                    FeatureId::new(99),
                    FeatureId::new(0),
                    None,
                )?],
            )
        }
    }

    fn workspace() -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..3).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap();
        for index in 0..3 {
            let keypoints = vec![Keypoint::try_new(index as f32, 0.0, 1.0, 0.0).unwrap()];
            let descriptors = DescriptorSet::try_u8(vec![index as u8], 1).unwrap();
            workspace
                .set_features(
                    index,
                    ExtractedFeatures::try_new(keypoints, descriptors).unwrap(),
                )
                .unwrap();
        }
        workspace
    }

    #[cfg(feature = "parallel")]
    fn workspace_with_descriptors(descriptors: impl Fn(usize) -> DescriptorSet) -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..4).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap();
        for image in 0..4 {
            let keypoints = (0..5)
                .map(|feature| Keypoint::try_new(feature as f32, image as f32, 1.0, 0.0).unwrap())
                .collect();
            workspace
                .set_features(
                    image,
                    ExtractedFeatures::try_new(keypoints, descriptors(image)).unwrap(),
                )
                .unwrap();
        }
        workspace
    }

    #[cfg(feature = "parallel")]
    fn u8_workspace() -> SfmWorkspace {
        workspace_with_descriptors(|image| {
            let mut data = Vec::with_capacity(5 * 4);
            for base in [0_u8, 40, 80, 120, 160] {
                data.extend([base.saturating_add(image as u8); 4]);
            }
            DescriptorSet::try_u8(data, 4).unwrap()
        })
    }

    #[cfg(feature = "parallel")]
    fn f32_workspace() -> SfmWorkspace {
        workspace_with_descriptors(|image| {
            let mut data = Vec::with_capacity(5 * 4);
            for base in [0.0_f32, 40.0, 80.0, 120.0, 160.0] {
                data.extend([base + image as f32; 4]);
            }
            DescriptorSet::try_f32(data, 4).unwrap()
        })
    }

    #[cfg(feature = "parallel")]
    fn rootsift_workspace() -> SfmWorkspace {
        workspace_with_descriptors(|image| {
            let mut data = Vec::with_capacity(5 * 128);
            for base in [0_u8, 40, 80, 120, 160] {
                data.extend([base.saturating_add(image as u8); 128]);
            }
            DescriptorSet::try_u8(data, 128).unwrap()
        })
    }

    #[cfg(feature = "parallel")]
    fn one_query_workspace() -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([
                ImageMeta::external_id("query", 64, 48),
                ImageMeta::external_id("database", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap();
        workspace
            .set_features(
                0,
                ExtractedFeatures::try_new(
                    vec![Keypoint::try_new(0.0, 0.0, 1.0, 0.0).unwrap()],
                    DescriptorSet::try_u8(vec![1, 0], 2).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        workspace
            .set_features(
                1,
                ExtractedFeatures::try_new(
                    vec![
                        Keypoint::try_new(0.0, 0.0, 1.0, 0.0).unwrap(),
                        Keypoint::try_new(1.0, 0.0, 1.0, 0.0).unwrap(),
                    ],
                    DescriptorSet::try_u8(vec![0, 0, 10, 10], 2).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        workspace
    }

    #[cfg(feature = "parallel")]
    fn match_store_with_workers(
        workspace: SfmWorkspace,
        matcher: &dyn PairMatcher,
        workers: usize,
    ) -> Vec<PairMatches> {
        match_store_with_progress(workspace, matcher, workers, &())
    }

    #[cfg(feature = "parallel")]
    fn match_store_with_progress(
        mut workspace: SfmWorkspace,
        matcher: &dyn PairMatcher,
        workers: usize,
        progress: &dyn SfmProgressSink,
    ) -> Vec<PairMatches> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                match_pairs_with_progress(&mut workspace, matcher, PairSelector::All, progress)
            })
            .unwrap();
        workspace
            .match_store()
            .iter_populated()
            .map(|(_, matches)| matches.clone())
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn direct_matches(workspace: &SfmWorkspace, matcher: &dyn PairMatcher) -> Vec<PairMatches> {
        (0..workspace.pair_graph().len())
            .map(|index| {
                let pair = workspace.pair_graph().pair(PairId::new(index)).unwrap();
                matcher
                    .match_features(
                        &PairContext {
                            pair,
                            view_i: pair.first(),
                            view_j: pair.second(),
                            root_seed: workspace.root_seed(),
                            seed: (index as u64).into(),
                        },
                        workspace.feature_store().get(pair.first().index()).unwrap(),
                        workspace
                            .feature_store()
                            .get(pair.second().index())
                            .unwrap(),
                    )
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn prepared_matcher_is_built_once_and_shared_by_all_pairs() {
        let mut workspace = workspace();
        let matcher = PreparedCountingMatcher {
            prepare_calls: AtomicUsize::new(0),
            prepared_calls: AtomicUsize::new(0),
            direct_calls: AtomicUsize::new(0),
            prepared_pairs: Mutex::new(Vec::new()),
        };

        match_pairs(&mut workspace, &matcher, PairSelector::All).unwrap();

        assert_eq!(matcher.prepare_calls.load(Ordering::Relaxed), 1);
        assert_eq!(matcher.prepared_calls.load(Ordering::Relaxed), 3);
        assert_eq!(matcher.direct_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            *matcher.prepared_pairs.lock().unwrap(),
            vec![(0, 1), (0, 2), (1, 2)]
        );
        assert_eq!(workspace.match_store().populated_count(), 3);
    }

    #[test]
    fn default_preparation_keeps_the_direct_path() {
        let mut workspace = workspace();
        let matcher = DirectCountingMatcher(AtomicUsize::new(0));

        match_pairs(&mut workspace, &matcher, PairSelector::All).unwrap();

        assert_eq!(matcher.0.load(Ordering::Relaxed), 3);
        assert_eq!(workspace.match_store().populated_count(), 3);
    }

    #[test]
    fn preparation_failure_does_not_mutate_match_store() {
        let mut workspace = workspace();
        let progress = RecordingProgress::default();
        let error = match_pairs_with_progress(
            &mut workspace,
            &FailingPreparation,
            PairSelector::All,
            &progress,
        )
        .unwrap_err();

        assert!(matches!(error, SfmError::InvalidInput(_)));
        assert_eq!(workspace.match_store().populated_count(), 0);
        // Pair matching never began.
        assert!(progress.records().is_empty());
    }

    #[test]
    fn direct_matching_reports_every_pair_and_empty_reruns() {
        const STAGE: SfmProgressStage = SfmProgressStage::Matching;
        let mut workspace = workspace();
        let matcher = DirectCountingMatcher(AtomicUsize::new(0));
        let progress = RecordingProgress::default();
        match_pairs_with_progress(&mut workspace, &matcher, PairSelector::All, &progress).unwrap();
        assert_eq!(
            progress.records(),
            [
                ProgressRecord::Begin(STAGE, 3),
                ProgressRecord::Advanced(STAGE, 3),
                ProgressRecord::Finish(STAGE),
            ]
        );

        let progress = RecordingProgress::default();
        match_pairs_with_progress(&mut workspace, &matcher, PairSelector::Unmatched, &progress)
            .unwrap();
        assert_eq!(
            progress.records(),
            [
                ProgressRecord::Begin(STAGE, 0),
                ProgressRecord::Finish(STAGE)
            ]
        );
    }

    #[test]
    fn matcher_contract_violation_abandons_the_matching_stage() {
        let mut workspace = workspace();
        let progress = RecordingProgress::default();
        match_pairs_with_progress(
            &mut workspace,
            &WrongPairMatcher,
            PairSelector::All,
            &progress,
        )
        .unwrap_err();

        let records = progress.records();
        assert_eq!(
            records.first(),
            Some(&ProgressRecord::Begin(SfmProgressStage::Matching, 3))
        );
        assert!(!records.contains(&ProgressRecord::Finish(SfmProgressStage::Matching)));
    }

    #[test]
    fn matcher_contract_violations_do_not_mutate_match_store() {
        for matcher in [
            &WrongPairMatcher as &dyn PairMatcher,
            &OutOfRangeMatcher as &dyn PairMatcher,
        ] {
            let mut workspace = workspace();
            let error = match_pairs(&mut workspace, matcher, PairSelector::All).unwrap_err();

            assert!(matches!(
                error,
                SfmError::InvalidInput(_) | SfmError::InvalidId { .. }
            ));
            assert_eq!(workspace.match_store().populated_count(), 0);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn prepared_hnsw_progress_is_deterministic_across_thread_counts() {
        let matcher = PairMatcherAdapter::new(
            MatcherType::hnsw_default(),
            0.8,
            MatchAcceptance::Reciprocal,
        );
        let unobserved = match_store_with_workers(u8_workspace(), &matcher, 1);
        for workers in [1, 4] {
            let progress = RecordingProgress::default();
            let observed = match_store_with_progress(u8_workspace(), &matcher, workers, &progress);

            assert_eq!(observed, unobserved);
            assert_eq!(
                progress.records(),
                [
                    ProgressRecord::Begin(SfmProgressStage::MatcherPreparation, 4),
                    ProgressRecord::Advanced(SfmProgressStage::MatcherPreparation, 4),
                    ProgressRecord::Finish(SfmProgressStage::MatcherPreparation),
                    ProgressRecord::Begin(SfmProgressStage::Matching, 6),
                    ProgressRecord::Advanced(SfmProgressStage::Matching, 6),
                    ProgressRecord::Finish(SfmProgressStage::Matching),
                ]
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn prepared_hnsw_is_deterministic_across_thread_counts_and_matches_direct_use() {
        let matcher = PairMatcherAdapter::new(
            MatcherType::hnsw_default(),
            0.8,
            MatchAcceptance::Reciprocal,
        );
        let workspace = u8_workspace();
        let pair = workspace.pair_graph().pair(PairId::new(0)).unwrap();
        let direct = matcher
            .match_features(
                &PairContext {
                    pair,
                    view_i: pair.first(),
                    view_j: pair.second(),
                    root_seed: workspace.root_seed(),
                    seed: 99.into(),
                },
                workspace.feature_store().get(pair.first().index()).unwrap(),
                workspace
                    .feature_store()
                    .get(pair.second().index())
                    .unwrap(),
            )
            .unwrap();

        let single_thread = match_store_with_workers(workspace, &matcher, 1);
        let four_threads = match_store_with_workers(u8_workspace(), &matcher, 4);
        let repeated = match_store_with_workers(u8_workspace(), &matcher, 4);

        assert_eq!(single_thread, four_threads);
        assert_eq!(four_threads, repeated);
        assert_eq!(single_thread[0], direct);
        assert_eq!(single_thread[0].len(), 5);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn prepared_hnsw_f32_matches_direct_use() {
        let matcher = PairMatcherAdapter::new(
            MatcherType::hnsw_default(),
            0.8,
            MatchAcceptance::Reciprocal,
        );
        let workspace = f32_workspace();
        let pair = workspace.pair_graph().pair(PairId::new(0)).unwrap();
        let direct = matcher
            .match_features(
                &PairContext {
                    pair,
                    view_i: pair.first(),
                    view_j: pair.second(),
                    root_seed: workspace.root_seed(),
                    seed: 123.into(),
                },
                workspace.feature_store().get(pair.first().index()).unwrap(),
                workspace
                    .feature_store()
                    .get(pair.second().index())
                    .unwrap(),
            )
            .unwrap();

        let prepared = match_store_with_workers(workspace, &matcher, 4);
        assert_eq!(prepared[0], direct);
        assert_eq!(prepared[0].len(), 5);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn prepared_one_way_hnsw_is_deterministic_across_thread_counts_and_matches_direct_use() {
        let matcher =
            PairMatcherAdapter::new(MatcherType::hnsw_default(), 0.8, MatchAcceptance::OneWay);
        for make_workspace in [u8_workspace, f32_workspace] {
            let workspace = make_workspace();
            let direct = direct_matches(&workspace, &matcher);

            let single_thread = match_store_with_workers(workspace, &matcher, 1);
            let four_threads = match_store_with_workers(make_workspace(), &matcher, 4);
            let repeated = match_store_with_workers(make_workspace(), &matcher, 4);

            assert_eq!(single_thread, four_threads);
            assert_eq!(four_threads, repeated);
            assert_eq!(single_thread, direct);
            assert!(direct.iter().all(|matches| matches.len() == 5));
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn acceptance_switches_hnsw_between_reciprocal_and_one_way() {
        // One query descriptor cannot pass the reverse ratio test, so only
        // one-way acceptance keeps its match.
        let one_way =
            PairMatcherAdapter::new(MatcherType::hnsw_default(), 0.8, MatchAcceptance::OneWay);
        let matches = match_store_with_workers(one_query_workspace(), &one_way, 4);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].len(), 1);
        assert_eq!(matches[0].as_slice()[0].first().index(), 0);
        assert_eq!(matches[0].as_slice()[0].second().index(), 0);

        let reciprocal = PairMatcherAdapter::new(
            MatcherType::hnsw_default(),
            0.8,
            MatchAcceptance::Reciprocal,
        );
        let matches = match_store_with_workers(one_query_workspace(), &reciprocal, 4);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].is_empty());
    }

    /// SfM matching thread-count replay: one-worker and four-worker matching produce identical
    /// scientific state; per-pair work is independent of Rayon scheduling.
    #[cfg(feature = "parallel")]
    #[test]
    fn rootsift_matching_is_deterministic_across_thread_counts() {
        let matcher = RootSiftU8Matcher::default();
        let single_thread = match_store_with_workers(rootsift_workspace(), &matcher, 1);
        let four_threads = match_store_with_workers(rootsift_workspace(), &matcher, 4);
        let repeated = match_store_with_workers(rootsift_workspace(), &matcher, 4);

        assert_eq!(single_thread, four_threads);
        assert_eq!(four_threads, repeated);
        assert_eq!(single_thread.len(), 6);
        assert!(single_thread.iter().all(|matches| matches.len() == 5));
        assert!(single_thread.iter().all(|matches| {
            matches
                .as_slice()
                .iter()
                .all(|matched| matched.first().index() == matched.second().index())
        }));
    }
}
