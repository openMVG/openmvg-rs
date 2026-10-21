// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exhaustive (exact) nearest-neighbor search: the adapter's brute-force backend and the
//! RootSIFT matcher, fixed to its byte descriptor encoding.

use openmvg_features::matching::BruteForceMatcher;
use openmvg_features::matching::{DistanceMetric, FilteredMatch, MatchAcceptance};
use openmvg_features::matching::{SquaredL2Distance, SquaredL2DistanceU8};

use super::{descriptor_slices_u8, descriptors_to_f32_vecs, filtered_matches_to_pipeline};
use crate::backends::{PairContext, PairMatcher};
use crate::two_view::PairMatches;
use crate::{Result, SfmError};
use openmvg_features::model::ExtractedFeatures;

pub(super) fn match_brute_force_features(
    fi: &ExtractedFeatures,
    fj: &ExtractedFeatures,
    ratio_threshold: f32,
    acceptance: MatchAcceptance,
) -> Result<Vec<FilteredMatch>> {
    if !acceptance.can_match(fi.len(), fj.len()) {
        return Ok(Vec::new());
    }
    match (fi.descriptors().as_u8(), fj.descriptors().as_u8()) {
        (Some((data_i, 128)), Some((data_j, 128))) => Ok(BruteForceMatcher::match_exact_u8(
            data_i.as_chunks::<128>().0,
            data_j.as_chunks::<128>().0,
            ratio_threshold,
            acceptance,
        )),
        (Some((data_i, dim_i)), Some((data_j, dim_j))) if dim_i == dim_j => match_slices(
            SquaredL2DistanceU8::new(),
            &descriptor_slices_u8(data_i, dim_i)?,
            &descriptor_slices_u8(data_j, dim_j)?,
            ratio_threshold,
            acceptance,
        ),
        _ => {
            let desc_i = descriptors_to_f32_vecs(fi.descriptors())?;
            let desc_j = descriptors_to_f32_vecs(fj.descriptors())?;
            match_slices(
                SquaredL2Distance::new(),
                &desc_i.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &desc_j.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                ratio_threshold,
                acceptance,
            )
        }
    }
}

/// Exhaustive squared-L2 matching of image i against image j.
fn match_slices<T, D>(
    metric: D,
    slices_i: &[&[T]],
    slices_j: &[&[T]],
    ratio_threshold: f32,
    acceptance: MatchAcceptance,
) -> Result<Vec<FilteredMatch>>
where
    T: Clone + Send + Sync,
    D: DistanceMetric<T> + Clone + Send + Sync,
{
    let nearest_two = |queries: &[&[T]], database: &[&[T]]| {
        BruteForceMatcher::match_direct_slices(metric.clone(), queries, database, 2)
            .map_err(|err| SfmError::InvalidInput(format!("brute-force matching failed: {err}")))
    };
    let forward = nearest_two(slices_i, slices_j)?;
    acceptance.filter_squared(&forward, ratio_threshold, |targets| {
        let rows: Vec<_> = targets.iter().map(|&target| slices_j[target]).collect();
        nearest_two(&rows, slices_i)
    })
}

/// Lowe's conventional ratio threshold, expressed in ordinary L2 units.
pub const ROOT_SIFT_U8_RATIO: f32 = 0.8;

/// Exact squared-L2 RootSIFT matcher.
#[derive(Debug, Clone, Copy)]
pub struct RootSiftU8Matcher {
    ratio: f32,
    acceptance: MatchAcceptance,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl RootSiftU8Matcher {
    /// Creates a matcher with Lowe ratio `ratio` (in L2 units) and the default acceptance.
    ///
    /// # Errors
    ///
    /// Returns [`SfmError::InvalidInput`] unless `ratio` is finite and in `(0, 1)`.
    pub fn try_new(ratio: f32) -> Result<Self> {
        validate_ratio("RootSIFT", ratio)?;
        Ok(Self {
            ratio,
            acceptance: MatchAcceptance::default(),
        })
    }

    /// Select the correspondence acceptance rule.
    pub const fn with_acceptance(mut self, acceptance: MatchAcceptance) -> Self {
        self.acceptance = acceptance;
        self
    }

    /// Correspondence acceptance rule.
    pub const fn acceptance(&self) -> MatchAcceptance {
        self.acceptance
    }
}

impl Default for RootSiftU8Matcher {
    fn default() -> Self {
        Self {
            ratio: ROOT_SIFT_U8_RATIO,
            acceptance: MatchAcceptance::default(),
        }
    }
}

impl PairMatcher for RootSiftU8Matcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        let (left, dim_i) = u8_descriptors(fi)?;
        let (right, dim_j) = u8_descriptors(fj)?;
        if dim_i != 128 || dim_j != 128 {
            return Err(SfmError::InvalidInput(format!(
                "RootSiftU8Matcher requires 128-D descriptors, got {dim_i} and {dim_j}"
            )));
        }
        let filtered = BruteForceMatcher::match_exact_u8(
            left.as_chunks::<128>().0,
            right.as_chunks::<128>().0,
            self.ratio,
            self.acceptance,
        );
        filtered_matches_to_pipeline(ctx.pair, filtered)
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn validate_ratio(label: &str, ratio: f32) -> Result<()> {
    if !ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0 {
        return Err(SfmError::InvalidInput(format!(
            "{label} ratio must be finite and in (0, 1)"
        )));
    }
    Ok(())
}

fn u8_descriptors(features: &ExtractedFeatures) -> Result<(&[u8], usize)> {
    features.descriptors().as_u8().ok_or_else(|| {
        SfmError::InvalidInput("RootSiftU8Matcher requires u8 descriptors".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureId, ViewId, ViewPair};
    use openmvg_features::model::{DescriptorSet, Keypoint};

    #[test]
    fn matchers_reject_zero_and_unit_ratios() {
        assert!(RootSiftU8Matcher::try_new(0.0).is_err());
        assert!(RootSiftU8Matcher::try_new(1.0).is_err());
    }

    #[test]
    fn exact_matchers_default_to_reciprocal_and_keep_one_way_explicit() {
        assert_eq!(
            RootSiftU8Matcher::default().acceptance(),
            MatchAcceptance::Reciprocal
        );
        assert_eq!(
            RootSiftU8Matcher::default()
                .with_acceptance(MatchAcceptance::OneWay)
                .acceptance(),
            MatchAcceptance::OneWay
        );
    }

    #[test]
    fn rootsift_reciprocal_matching_rejects_targets_with_tied_queries() {
        let keypoints = vec![
            Keypoint::try_new(0.0, 0.0, 1.0, 0.0).unwrap(),
            Keypoint::try_new(1.0, 0.0, 1.0, 0.0).unwrap(),
        ];
        // Both left descriptors are identical, so right descriptor 0 cannot
        // tell them apart: one-way matching keeps both, reciprocal neither.
        let left = ExtractedFeatures::try_new(
            keypoints.clone(),
            DescriptorSet::try_u8(vec![0; 2 * 128], 128).unwrap(),
        )
        .unwrap();
        let mut right_data = vec![0; 2 * 128];
        right_data[128..].fill(255);
        let right =
            ExtractedFeatures::try_new(keypoints, DescriptorSet::try_u8(right_data, 128).unwrap())
                .unwrap();
        let context = PairContext {
            pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
            view_i: ViewId::new(0),
            view_j: ViewId::new(1),
            root_seed: 0.into(),
            seed: 0.into(),
        };

        let reciprocal = RootSiftU8Matcher::default()
            .match_features(&context, &left, &right)
            .unwrap();
        let one_way = RootSiftU8Matcher::default()
            .with_acceptance(MatchAcceptance::OneWay)
            .match_features(&context, &left, &right)
            .unwrap();

        assert!(reciprocal.is_empty());
        let pairs: Vec<_> = one_way
            .as_slice()
            .iter()
            .map(|matched| (matched.first(), matched.second()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                (FeatureId::new(0), FeatureId::new(0)),
                (FeatureId::new(1), FeatureId::new(0)),
            ]
        );
    }
}
