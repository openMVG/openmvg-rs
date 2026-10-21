// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature matching adapters: [`PairMatcherAdapter`] selects a brute-force or HNSW search backend; [`RootSiftU8Matcher`] is the exact matcher for RootSIFT byte descriptors.

use crate::backends::{PairContext, PairMatcher as PipelinePairMatcher};
use crate::backends::{PairMatcherPreparation, PreparedPairMatcher};
use crate::progress::{SfmProgressStage, cancellation_point};
use crate::two_view::PairMatches;
use crate::{FeatureId, Result, SfmError};
use openmvg_features::matching::FilteredMatch;
pub use openmvg_features::matching::MatchAcceptance;
use openmvg_features::model::{DescriptorSet, ExtractedFeatures, FeatureMatch};

pub(crate) mod brute_force;
mod hnsw;

use brute_force::match_brute_force_features;
#[cfg(feature = "unstable-internals")]
pub use brute_force::{ROOT_SIFT_U8_RATIO, RootSiftU8Matcher};
use hnsw::{HnswOptions, PreparedHnswMatcher, match_hnsw_features};

/// Nearest-neighbor search backend used by [`PairMatcherAdapter`].
#[derive(Debug, Clone, PartialEq)]
pub enum MatcherType {
    /// Exhaustive L2 search over every descriptor of the other image.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    BruteForce,
    /// Compact deterministic HNSW for `u8` or `f32` descriptors of any fixed
    /// runtime dimension. The graph uses M=12, M0=24 and packed `u32` links.
    Hnsw {
        /// Candidates explored during index construction.
        ef_construction: usize,
        /// Search beam width at query time (larger = higher recall, slower).
        ef_search: usize,
    },
}

impl Default for MatcherType {
    fn default() -> Self {
        Self::hnsw_default()
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl MatcherType {
    /// Stable identifier encoding the backend and its parameters, such as `hnsw_efc24_efs24`.
    pub fn name(&self) -> String {
        match self {
            Self::BruteForce => "brute_force".to_owned(),
            Self::Hnsw {
                ef_construction,
                ef_search,
            } => {
                format!("hnsw_efc{}_efs{}", ef_construction, ef_search)
            }
        }
    }

    /// HNSW matcher tuned for pairwise 2-NN ratio matching. The balanced
    /// `24/24` policy was selected on 65 VGG image pairs: 99.50% mean raw
    /// top-2 recall and 99.86% weighted accepted-target agreement.
    pub fn hnsw_default() -> Self {
        Self::Hnsw {
            ef_construction: 24,
            ef_search: 24,
        }
    }
}

/// Pair matcher backed by a [`MatcherType`] search, Lowe's ratio test, and a
/// [`MatchAcceptance`] policy.
///
/// The default is [`MatcherType::hnsw_default`] with ratio 0.8 and the default acceptance.
#[derive(Debug, Clone)]
pub struct PairMatcherAdapter {
    matcher_type: MatcherType,
    ratio_threshold: f32,
    acceptance: MatchAcceptance,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl PairMatcherAdapter {
    /// Match image `i`'s descriptors against image `j`'s with Lowe's ratio
    /// test, then keep the correspondences `acceptance` allows.
    pub fn new(
        matcher_type: MatcherType,
        ratio_threshold: f32,
        acceptance: MatchAcceptance,
    ) -> Self {
        Self {
            matcher_type,
            ratio_threshold,
            acceptance,
        }
    }

    /// Search backend.
    pub fn matcher_type(&self) -> &MatcherType {
        &self.matcher_type
    }

    /// Which ratio-test survivors are kept (one-way or reciprocal).
    pub const fn acceptance(&self) -> MatchAcceptance {
        self.acceptance
    }

    fn hnsw_options(&self, ef_construction: usize, ef_search: usize) -> HnswOptions {
        HnswOptions {
            ef_construction,
            ef_search,
            ratio_threshold: self.ratio_threshold,
            acceptance: self.acceptance,
        }
    }
}

impl Default for PairMatcherAdapter {
    fn default() -> Self {
        Self::new(MatcherType::default(), 0.8, MatchAcceptance::default())
    }
}

impl PipelinePairMatcher for PairMatcherAdapter {
    fn prepare<'a>(
        &'a self,
        ctx: PairMatcherPreparation<'a>,
    ) -> Result<Option<Box<dyn PreparedPairMatcher + 'a>>> {
        match self.matcher_type {
            MatcherType::BruteForce => Ok(None),
            MatcherType::Hnsw {
                ef_construction,
                ef_search,
            } => Ok(Some(Box::new(PreparedHnswMatcher::new(
                ctx,
                self.hnsw_options(ef_construction, ef_search),
            )?))),
        }
    }

    fn match_features(
        &self,
        ctx: &PairContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        let filtered = match self.matcher_type {
            MatcherType::BruteForce => {
                match_brute_force_features(fi, fj, self.ratio_threshold, self.acceptance)?
            }
            MatcherType::Hnsw {
                ef_construction,
                ef_search,
            } => match_hnsw_features(
                fi,
                fj,
                [ctx.view_i, ctx.view_j],
                ctx.root_seed,
                self.hnsw_options(ef_construction, ef_search),
            )?,
        };

        filtered_matches_to_pipeline(ctx.pair, filtered)
    }
}

/// Prepare every image referenced by `ctx.pairs()` once, in parallel under the
/// `parallel` feature, reporting `MatcherPreparation` progress per image. Unreferenced
/// slots stay `None`.
fn prepare_pair_views<'a, T, F>(
    ctx: PairMatcherPreparation<'a>,
    prepare: F,
) -> Result<Vec<Option<T>>>
where
    T: Send,
    F: Fn(crate::ViewId, &'a ExtractedFeatures) -> Result<T> + Sync,
{
    let features = ctx.features();
    let progress = ctx.progress();
    let mut selected = vec![false; features.slot_count()];
    for pair in ctx.pairs() {
        selected[pair.first().index()] = true;
        selected[pair.second().index()] = true;
    }
    cancellation_point(progress)?;
    progress.begin(
        SfmProgressStage::MatcherPreparation,
        selected.iter().filter(|&&is_selected| is_selected).count(),
    );
    let prepare_slot = |(index, is_selected): (usize, bool)| -> Result<Option<T>> {
        if !is_selected {
            return Ok(None);
        }
        cancellation_point(progress)?;
        let view = crate::ViewId::new(index);
        let view_features = features.get(view).ok_or(SfmError::MissingPrerequisite {
            prerequisite: "features",
            method: "extract_features",
        })?;
        let prepared = prepare(view, view_features)?;
        progress.advance(SfmProgressStage::MatcherPreparation, 1);
        Ok(Some(prepared))
    };

    #[cfg(feature = "parallel")]
    let prepared: Vec<_> = {
        use rayon::prelude::*;
        selected
            .into_par_iter()
            .enumerate()
            .map(prepare_slot)
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let prepared: Vec<_> = selected.into_iter().enumerate().map(prepare_slot).collect();
    // Collect sequentially so the reported error is always the first failing image.
    let prepared = prepared.into_iter().collect::<Result<_>>()?;
    progress.finish(SfmProgressStage::MatcherPreparation);
    Ok(prepared)
}

fn prepared_view<'p, T>(
    images: &'p [Option<T>],
    view: crate::ViewId,
    backend: &str,
) -> Result<&'p T> {
    images
        .get(view.index())
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            SfmError::InvalidInput(format!(
                "prepared {backend} matcher received unselected view {}",
                view.index()
            ))
        })
}
fn filtered_matches_to_pipeline(
    pair: crate::ViewPair,
    filtered: Vec<FilteredMatch>,
) -> Result<PairMatches> {
    let matches: Result<Vec<_>> = filtered
        .into_iter()
        .map(|matched| {
            Ok(FeatureMatch::try_new(
                FeatureId::new(matched.query_index()),
                FeatureId::new(matched.target_index()),
                Some(matched.distance()),
            )?)
        })
        .collect();
    PairMatches::try_new(pair, matches?)
}

fn descriptors_to_f32_vecs(descriptors: &DescriptorSet) -> Result<Vec<Vec<f32>>> {
    if let Some((data, dimension)) = descriptors.as_f32() {
        return dense_chunks(data, dimension, "f32 descriptor");
    }
    if let Some((data, dimension)) = descriptors.as_u8() {
        let chunks = dense_chunks(data, dimension, "u8 descriptor")?;
        return Ok(chunks
            .into_iter()
            .map(|descriptor| descriptor.into_iter().map(f32::from).collect())
            .collect());
    }
    let (data, dimension_bits) = descriptors.as_binary().ok_or_else(|| {
        SfmError::InvariantViolation("unknown descriptor storage kind".to_owned())
    })?;
    let words_per_descriptor = dimension_bits.div_ceil(64);
    dense_chunks(data, words_per_descriptor, "binary descriptor").map(|chunks| {
        chunks
            .into_iter()
            .map(|descriptor| descriptor.into_iter().map(|word| word as f32).collect())
            .collect()
    })
}

fn descriptor_slices_u8(data: &[u8], dim: usize) -> Result<Vec<&[u8]>> {
    if dim == 0 || !data.len().is_multiple_of(dim) {
        return Err(SfmError::InvalidInput(format!(
            "u8 descriptor data length {} is not divisible by dimension {}",
            data.len(),
            dim
        )));
    }
    Ok(data.chunks_exact(dim).collect())
}

fn descriptor_slices_f32(data: &[f32], dim: usize) -> Result<Vec<&[f32]>> {
    if dim == 0 || !data.len().is_multiple_of(dim) {
        return Err(SfmError::InvalidInput(format!(
            "f32 descriptor data length {} is not divisible by dimension {}",
            data.len(),
            dim
        )));
    }
    Ok(data.chunks_exact(dim).collect())
}

fn dense_chunks<T: Copy>(data: &[T], dim: usize, label: &str) -> Result<Vec<Vec<T>>> {
    if dim == 0 || !data.len().is_multiple_of(dim) {
        return Err(SfmError::InvalidInput(format!(
            "{} data length {} is not divisible by dimension {}",
            label,
            data.len(),
            dim
        )));
    }
    Ok(data.chunks_exact(dim).map(|chunk| chunk.to_vec()).collect())
}

#[cfg(test)]
mod adapter_tests {
    use super::*;

    #[test]
    fn reciprocal_matching_is_the_default() {
        assert_eq!(
            PairMatcherAdapter::default().acceptance(),
            MatchAcceptance::Reciprocal
        );
    }

    #[test]
    fn direct_and_prepared_hnsw_reject_unsupported_degenerate_storage() {
        let feature = ExtractedFeatures::try_new(
            vec![openmvg_features::model::Keypoint::try_new(0.0, 0.0, 1.0, 0.0).unwrap()],
            DescriptorSet::try_binary(vec![0; 4], 256, 1).unwrap(),
        )
        .unwrap();
        let adapter = PairMatcherAdapter::new(
            MatcherType::hnsw_default(),
            0.8,
            MatchAcceptance::Reciprocal,
        );
        let pair = crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1));
        let ctx = PairContext {
            pair,
            view_i: pair.first(),
            view_j: pair.second(),
            root_seed: crate::RunSeed::ZERO,
            seed: 0.into(),
        };
        let direct_error = adapter
            .match_features(&ctx, &feature, &feature)
            .unwrap_err();

        let mut store = crate::workspace::store::SparseStore::new(2);
        store.set(0, feature.clone()).unwrap();
        store.set(1, feature).unwrap();
        let pairs = [pair];
        let progress = crate::progress::testing::RecordingProgress::default();
        let preparation = PairMatcherPreparation::new(
            crate::workspace::views::FeaturesView::new(&store),
            &pairs,
            crate::RunSeed::ZERO,
            &progress,
        );
        let Err(prepared_error) = adapter.prepare(preparation) else {
            panic!("unsupported descriptors must fail during HNSW preparation");
        };

        assert_eq!(direct_error.to_string(), prepared_error.to_string());
        // Both images fail, so preparation is begun but never advanced or finished.
        assert_eq!(
            progress.records(),
            [crate::progress::testing::ProgressRecord::Begin(
                SfmProgressStage::MatcherPreparation,
                2
            )]
        );
    }
}
