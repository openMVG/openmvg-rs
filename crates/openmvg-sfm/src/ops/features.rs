// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature extraction stage: run a [`FeatureExtractor`] over the
//! selected views.

use crate::backends::{FeatureContext, FeatureExtractor};
use crate::determinism::FEATURE_EXTRACTION_SEED_DOMAIN;
use crate::model::ImageMeta;
use crate::ops::resolve_views;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::workspace::selector::ViewSelector;
use crate::{Result, SfmError, SfmWorkspace, ViewId};
use openmvg_features::model::ExtractedFeatures;

/// Detect features for the selected images.
///
/// Detection is embarrassingly parallel (each image is independent) and dominates
/// front-end wall time, so under the `parallel` feature the per-image work is fanned
/// out across a rayon pool. Results are gathered owned and then written into the
/// workspace sequentially: each image owns its own store slot, so the write order is
/// irrelevant and the reconstruction stays byte-for-byte deterministic.
pub fn extract_features(
    workspace: &mut SfmWorkspace,
    extractor: &dyn FeatureExtractor,
    selector: ViewSelector,
) -> Result<()> {
    extract_features_with_progress(workspace, extractor, selector, &())
}

/// [`extract_features`], reporting [`SfmProgressStage::FeatureExtraction`] in images.
pub fn extract_features_with_progress(
    workspace: &mut SfmWorkspace,
    extractor: &dyn FeatureExtractor,
    selector: ViewSelector,
    progress: &dyn SfmProgressSink,
) -> Result<()> {
    if workspace.is_matching_data_released() {
        return Err(crate::SfmError::MatchingDataReleased);
    }
    let view_ids = resolve_views(workspace, selector)?;
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::FeatureExtraction, view_ids.len());

    let extract_one = |view_id: ViewId, meta: &ImageMeta| -> Result<(usize, ExtractedFeatures)> {
        cancellation_point(progress)?;
        let features = extractor.extract(
            &FeatureContext {
                view: view_id,
                seed: workspace.derive_seed(FEATURE_EXTRACTION_SEED_DOMAIN, &[view_id.value()]),
            },
            meta,
        )?;
        features.validate()?;
        progress.advance(SfmProgressStage::FeatureExtraction, 1);
        Ok((view_id.index(), features))
    };

    #[cfg(feature = "parallel")]
    let results: Vec<(usize, ExtractedFeatures)> = {
        use rayon::prelude::*;
        // Each worker holds one image's detection working memory, so fan-out is bounded by
        // the extractor's per-worker memory against the available memory, and by a fixed
        // total budget in case the available memory cannot be read. A dedicated pool keeps
        // this bound off the global pool (matching uses that at full width).
        let per_worker_gib = extractor.worker_memory_gib();
        let workers = crate::ops::bounded_workers(
            view_ids.len(),
            per_worker_gib,
            extraction_worker_ceiling(per_worker_gib),
            workspace
                .max_parallel_workers()
                .map(|workers| workers.get()),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| crate::SfmError::InvalidConfig(format!("rayon pool: {e}")))?;
        let images = workspace.views();
        pool.install(|| {
            view_ids
                .par_iter()
                .map(|&view_id| extract_one(view_id, &images[view_id.index()].meta))
                .collect::<Result<Vec<_>>>()
        })?
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(usize, ExtractedFeatures)> = {
        let images = workspace.views();
        view_ids
            .iter()
            .map(|&view_id| extract_one(view_id, &images[view_id.index()].meta))
            .collect::<Result<Vec<_>>>()?
    };

    for (index, features) in &results {
        match workspace.feature_sets.get(*index) {
            Some(feature_set) if feature_set.len() != features.len() => {
                return Err(SfmError::InvalidInput(format!(
                    "extracted feature count {} disagrees with imported feature geometry count {} for view {}",
                    features.len(),
                    feature_set.len(),
                    ViewId::new(*index)
                )));
            }
            Some(feature_set)
                if feature_set.pixels().iter().zip(features.keypoints()).any(
                    |(pixel, keypoint)| {
                        pixel.x as f32 != keypoint.x() || pixel.y as f32 != keypoint.y()
                    },
                ) =>
            {
                return Err(SfmError::InvalidInput(format!(
                    "extracted keypoints disagree with imported feature geometry after f32 rounding for view {}",
                    ViewId::new(*index)
                )));
            }
            _ => {}
        }
    }
    for (index, features) in results {
        if workspace.feature_sets.get(index).is_none() {
            workspace.feature_sets.set(index, features.feature_set())?;
        }
        workspace.set_features(index, features)?;
    }
    progress.finish(SfmProgressStage::FeatureExtraction);
    Ok(())
}

/// Total extraction working memory allowed when the available memory cannot be read: sixteen
/// workers of the default 4 GiB.
#[cfg(feature = "parallel")]
const EXTRACTION_MEMORY_CEILING_GIB: f64 = 64.0;

/// Most concurrent extractions within [`EXTRACTION_MEMORY_CEILING_GIB`]; an unusable
/// per-worker estimate keeps the default sixteen.
#[cfg(feature = "parallel")]
fn extraction_worker_ceiling(per_worker_gib: f64) -> usize {
    if per_worker_gib.is_finite() && per_worker_gib > 0.0 {
        ((EXTRACTION_MEMORY_CEILING_GIB / per_worker_gib).floor() as usize).max(1)
    } else {
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::testing::{ProgressRecord, RecordingProgress};
    use crate::{Camera, IntrinsicId, SfmError};
    use openmvg_features::model::{DescriptorSet, Keypoint};

    const STAGE: SfmProgressStage = SfmProgressStage::FeatureExtraction;

    struct SingleKeypointExtractor;

    impl FeatureExtractor for SingleKeypointExtractor {
        fn extract(&self, _ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
            Ok(ExtractedFeatures::try_new(
                vec![Keypoint::try_new(1.0, 2.0, 1.0, 0.0)?],
                DescriptorSet::try_u8(vec![1], 1)?,
            )?)
        }
    }

    struct MalformedSecondImageExtractor;

    impl FeatureExtractor for MalformedSecondImageExtractor {
        fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
            if ctx.view.index() == 0 {
                return Ok(ExtractedFeatures::try_new(
                    vec![Keypoint::try_new(1.0, 2.0, 1.0, 0.0)?],
                    DescriptorSet::try_u8(vec![1], 1)?,
                )?);
            }
            // Feature records validate on construction and deserialization, so the
            // malformed keypoint surfaces as the extractor's own error.
            Ok(ExtractedFeatures::try_new(
                vec![Keypoint::try_new(1.0, 2.0, -1.0, 0.0)?],
                DescriptorSet::try_u8(vec![1], 1)?,
            )?)
        }
    }

    fn two_image_workspace() -> SfmWorkspace {
        SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap()
    }

    #[test]
    fn extraction_reports_every_image_and_empty_reruns() {
        let mut workspace = two_image_workspace();
        let progress = RecordingProgress::default();
        extract_features_with_progress(
            &mut workspace,
            &SingleKeypointExtractor,
            ViewSelector::All,
            &progress,
        )
        .unwrap();
        assert_eq!(
            progress.records(),
            [
                ProgressRecord::Begin(STAGE, 2),
                ProgressRecord::Advanced(STAGE, 2),
                ProgressRecord::Finish(STAGE),
            ]
        );

        let progress = RecordingProgress::default();
        extract_features_with_progress(
            &mut workspace,
            &SingleKeypointExtractor,
            ViewSelector::Missing,
            &progress,
        )
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
    fn extractor_contract_violation_does_not_mutate_feature_store() {
        let mut workspace = two_image_workspace();
        let progress = RecordingProgress::default();

        let error = extract_features_with_progress(
            &mut workspace,
            &MalformedSecondImageExtractor,
            ViewSelector::All,
            &progress,
        )
        .unwrap_err();

        assert!(matches!(error, SfmError::InvalidFeatureData(_)));
        assert_eq!(workspace.feature_store().populated_count(), 0);
        // The failed stage is abandoned: begun, never finished.
        let records = progress.records();
        assert_eq!(records.first(), Some(&ProgressRecord::Begin(STAGE, 2)));
        assert!(!records.contains(&ProgressRecord::Finish(STAGE)));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn extraction_workers_follow_the_extractor_memory_budget() {
        assert_eq!(extraction_worker_ceiling(4.0), 16);
        assert_eq!(extraction_worker_ceiling(0.75), 85);
        assert_eq!(extraction_worker_ceiling(0.0), 16);
        assert_eq!(extraction_worker_ceiling(f64::NAN), 16);
        // On 64 threads with 100 GiB free, a pyramid detector keeps sixteen workers while a
        // streaming detector uses every thread.
        let workers = |per_worker_gib: f64| {
            crate::ops::bounded_workers_from_limits(
                500,
                per_worker_gib,
                extraction_worker_ceiling(per_worker_gib),
                64,
                Some(100.0),
                None,
            )
        };
        assert_eq!(workers(4.0), 16);
        assert_eq!(workers(0.75), 64);
        // Scarce memory bounds both.
        let scarce = |per_worker_gib: f64| {
            crate::ops::bounded_workers_from_limits(
                500,
                per_worker_gib,
                extraction_worker_ceiling(per_worker_gib),
                64,
                Some(6.0),
                None,
            )
        };
        assert_eq!(scarce(4.0), 1);
        assert_eq!(scarce(0.75), 8);
    }
}
