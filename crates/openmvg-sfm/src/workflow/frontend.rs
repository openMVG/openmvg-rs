// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Standard image front end that turns a dataset into a verified view graph.

use super::{Dataset, ExtractionMethod, ViewGraph};
use crate::{Result, SfmWorkspace};
use openmvg_features::extraction::FeatureBudgetPolicy;

/// Front-end JSON written before the field existed always extracted SIFT.
fn legacy_extraction_method() -> ExtractionMethod {
    ExtractionMethod::Sift
}

fn is_unlimited_budget(budget: &FeatureBudgetPolicy) -> bool {
    matches!(budget, FeatureBudgetPolicy::Unlimited)
}

/// Standard deterministic feature, matching, verification, and track-building policy.
///
/// The front end matches every image pair and prepares a reusable [`ViewGraph`]; it does not
/// choose an SfM engine or run bundle adjustment. Building fails when no pair passes geometric
/// verification or the verified pairs yield no tracks.
///
/// [`Frontend::default`] is the recommended profile. It detects ALP keypoints (Analytic
/// Laplacian Polynomial, the default detector of `openmvg-features`) at
/// [`DetectorDensity::High`](crate::DetectorDensity::High) on at most 1600 pixels per image edge,
/// describes them with RootSIFT (SIFT descriptors with a square-root normalization, stored as
/// bytes), matches them with [`MatchingMethod::Hnsw`], an approximate nearest-neighbor search,
/// keeps the matches that pass Lowe's ratio test in both directions, and pre-calibrates radial
/// distortion. The `with_*` knobs adjust it, and [`Frontend::with_extraction_method`] selects
/// [`ExtractionMethod::Sift`] at the same density instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Frontend {
    pub(super) sift: crate::adapters::features::SiftFrontendConfig,
    #[serde(default = "legacy_extraction_method")]
    pub(super) extraction_method: ExtractionMethod,
    #[serde(default, skip_serializing_if = "is_unlimited_budget")]
    pub(super) feature_budget: FeatureBudgetPolicy,
    #[serde(default)]
    pub(super) matching_method: MatchingMethod,
    #[serde(default)]
    pub(super) radial_calibration: bool,
    #[serde(default)]
    pub(super) max_pairs_per_view: Option<usize>,
}

/// How a [`Frontend`] finds descriptor correspondences between two images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum MatchingMethod {
    /// Approximate nearest-neighbor search over a graph built once per image (HNSW), keeping
    /// a match when the nearest descriptor is clearly closer than the second (Lowe's ratio test)
    /// in both directions. Deterministic for a given seed, and fast; the default.
    #[default]
    Hnsw,
    /// Exact search over every descriptor pair, with the same acceptance. Choose it to check
    /// the HNSW results, or when matching time does not matter.
    BruteForce,
}

impl Default for Frontend {
    /// ALP keypoints at [`DetectorDensity::High`](crate::DetectorDensity::High), RootSIFT
    /// descriptors stored as bytes, and HNSW matching with Lowe's ratio 0.8 in both directions.
    fn default() -> Self {
        Self::profile(
            crate::adapters::features::SiftFrontendPreset::Quality,
            crate::DetectorDensity::High,
        )
    }
}

impl Frontend {
    const fn profile(
        preset: crate::adapters::features::SiftFrontendPreset,
        density: crate::DetectorDensity,
    ) -> Self {
        Self {
            sift: crate::adapters::features::SiftFrontendConfig::new(preset)
                .with_density(density)
                .with_max_dimension(1600),
            extraction_method: ExtractionMethod::Alp,
            feature_budget: FeatureBudgetPolicy::Unlimited,
            matching_method: MatchingMethod::Hnsw,
            radial_calibration: true,
            max_pairs_per_view: None,
        }
    }

    /// Feature extraction method; both emit the descriptors the profile's matching expects.
    #[must_use]
    pub const fn with_extraction_method(mut self, extraction_method: ExtractionMethod) -> Self {
        self.extraction_method = extraction_method;
        self
    }

    /// The feature extraction method.
    #[must_use]
    pub const fn extraction_method(&self) -> ExtractionMethod {
        self.extraction_method
    }

    /// Limit the number of features per image, before description, with a
    /// [`FeatureBudgetPolicy`](crate::FeatureBudgetPolicy).
    /// [`FeatureBudgetPolicy::strongest`](crate::FeatureBudgetPolicy::strongest) is the
    /// recommended cap; the default keeps every feature.
    ///
    /// ALP ranks keypoints before orientation, and SIFT ranks oriented keypoints. The count is an
    /// upper bound, because a keypoint without a valid descriptor is dropped.
    ///
    /// ```
    /// use openmvg_sfm::{FeatureBudgetPolicy, Frontend};
    ///
    /// let frontend = Frontend::default().with_feature_budget(FeatureBudgetPolicy::strongest(4000));
    /// assert_eq!(frontend.feature_budget(), FeatureBudgetPolicy::strongest(4000));
    /// ```
    #[must_use]
    pub const fn with_feature_budget(mut self, budget: FeatureBudgetPolicy) -> Self {
        self.feature_budget = budget;
        self
    }

    /// Budget applied to either native extraction method.
    #[must_use]
    pub const fn feature_budget(&self) -> FeatureBudgetPolicy {
        self.feature_budget
    }

    /// Descriptor matching method.
    #[must_use]
    pub const fn with_matching_method(mut self, matching_method: MatchingMethod) -> Self {
        self.matching_method = matching_method;
        self
    }

    /// The descriptor matching method.
    #[must_use]
    pub const fn matching_method(&self) -> MatchingMethod {
        self.matching_method
    }

    /// Downscale images so their longest edge is at most `max_dimension` pixels before feature
    /// extraction; `None` keeps native resolution.
    #[must_use]
    pub const fn with_max_dimension(mut self, max_dimension: Option<u32>) -> Self {
        self.sift = self.sift.with_optional_max_dimension(max_dimension);
        self
    }

    /// Detector density for either extraction method: each level selects the SIFT or ALP
    /// preset of the same name. It does not change descriptor encoding or matching.
    /// [`DetectorDensity::Ultra`](crate::DetectorDensity::Ultra) is SIFT only; building an ALP
    /// front end with it fails.
    #[must_use]
    pub const fn with_density(mut self, density: crate::DetectorDensity) -> Self {
        self.sift = self.sift.with_density(density);
        self
    }

    /// The detector density.
    #[must_use]
    pub const fn density(&self) -> crate::DetectorDensity {
        self.sift.density()
    }

    /// Estimate `k1` after verification for camera groups that declare a radial model with
    /// unknown (all-zero) distortion. Pinhole groups, groups with known distortion, and datasets
    /// with [calibrated intrinsics](Dataset::has_calibrated_intrinsics) are never changed.
    #[must_use]
    pub const fn with_radial_calibration(mut self, radial_calibration: bool) -> Self {
        self.radial_calibration = radial_calibration;
        self
    }

    /// Keep, for every image, its `max_pairs_per_view` verified pairs with the most inliers, and
    /// deactivate the other pairs before tracking. A pair stays when it is among the strongest at
    /// either of its two images, so an image with at most that many pairs keeps all of them;
    /// `None` (the default) keeps every verified pair. The budget must be positive; building
    /// fails otherwise.
    ///
    /// Dense captures of repeated structure (two similar facades, for instance) verify many weak
    /// pairs that chain unrelated parts of the scene together; dropping them lets the engines
    /// reconstruct each part in place. Choose a budget near the number of images that truly
    /// overlap each image in the capture.
    ///
    /// It is a trade-off, not a free improvement: on most datasets it changes little, and on
    /// some it costs a few cameras or some accuracy, so it is opt-in.
    #[must_use]
    pub const fn with_max_pairs_per_view(mut self, max_pairs_per_view: Option<usize>) -> Self {
        self.max_pairs_per_view = max_pairs_per_view;
        self
    }

    /// The per-image pair budget of the strongest-pairs filter, when enabled.
    #[must_use]
    pub const fn max_pairs_per_view(&self) -> Option<usize> {
        self.max_pairs_per_view
    }

    /// SIFT descriptor encoding, detector density and resolution settings.
    ///
    /// The shared budget is available separately through [`Self::feature_budget`].
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn sift(&self) -> crate::adapters::features::SiftFrontendConfig {
        self.sift
    }

    /// Prepare an immutable [`ViewGraph`] without changing the input dataset; the same as
    /// [`ViewGraph::build`].
    pub fn build(&self, dataset: &Dataset) -> Result<ViewGraph> {
        self.build_with_report(dataset, &()).map(|(graph, _)| graph)
    }

    /// [`Self::build`], reporting progress for every front-end stage to `progress` and returning
    /// what each stage produced. The progress sink does not change the result.
    pub fn build_with_report(
        &self,
        dataset: &Dataset,
        progress: &dyn crate::progress::SfmProgressSink,
    ) -> Result<(ViewGraph, crate::reports::FrontendReport)> {
        let mut workspace = dataset.workspace.clone();
        // A dataset carries no pairs, so the front end matches every image pair.
        workspace.install_pair_graph_before_matching(
            crate::workspace::pair_graph::PairGraph::exhaustive(workspace.views().len())?,
        )?;
        // A calibrated dataset keeps its distortion, even when it is zero.
        let radial_calibration = self.radial_calibration && !dataset.calibrated_intrinsics;
        let report = self.run_stages(&mut workspace, None, None, radial_calibration, progress)?;
        // Descriptors serve matching only; the graph keeps the keypoints its matches index.
        workspace.release_descriptors()?;
        let graph = ViewGraph::try_from_workspace(workspace)?
            .with_calibrated_intrinsics(dataset.calibrated_intrinsics)
            .with_extraction_method(report.extraction_method);
        Ok((graph, report))
    }

    /// Run the front end on a workspace whose pair graph is already installed, completing only
    /// the images and pairs that still lack features, matches, or a verification verdict.
    ///
    /// `extractor` replaces the extractor of [`Self::extraction_method`] and `matcher` the matcher that
    /// [`Self::matching_method`] selects; `None` keeps the profile's own. A replacement extractor must
    /// emit the descriptors the matcher expects, and the report then records no extraction method.
    /// Verification, radial calibration, and the empty-result checks are the same either way.
    #[cfg(feature = "unstable-internals")]
    pub fn stage_with(
        &self,
        workspace: &mut SfmWorkspace,
        extractor: Option<&dyn crate::backends::FeatureExtractor>,
        matcher: Option<&dyn crate::backends::PairMatcher>,
        progress: &dyn crate::progress::SfmProgressSink,
    ) -> Result<crate::reports::FrontendReport> {
        self.run_stages(
            workspace,
            extractor,
            matcher,
            self.radial_calibration,
            progress,
        )
    }

    fn run_stages(
        &self,
        workspace: &mut SfmWorkspace,
        extractor: Option<&dyn crate::backends::FeatureExtractor>,
        matcher: Option<&dyn crate::backends::PairMatcher>,
        radial_calibration: bool,
        progress: &dyn crate::progress::SfmProgressSink,
    ) -> Result<crate::reports::FrontendReport> {
        use crate::adapters::features::SiftFrontendPreset;
        use crate::adapters::matching::{MatchAcceptance, MatcherType, PairMatcherAdapter};
        let preset = self.sift.preset();
        let hnsw;
        let matcher: &dyn crate::backends::PairMatcher =
            match (matcher, self.matching_method, preset) {
                (Some(matcher), _, _) => matcher,
                (None, MatchingMethod::BruteForce, _) => &preset,
                (None, MatchingMethod::Hnsw, SiftFrontendPreset::Quality) => {
                    hnsw = PairMatcherAdapter::new(
                        MatcherType::hnsw_default(),
                        preset.ratio(),
                        MatchAcceptance::Reciprocal,
                    );
                    &hnsw
                }
            };
        let (sift, alp);
        let (extractor, extraction_method): (&dyn crate::backends::FeatureExtractor, _) =
            match (extractor, self.extraction_method) {
                (Some(extractor), _) => (extractor, None),
                (None, ExtractionMethod::Sift) => {
                    sift = self
                        .sift
                        .extractor()
                        .with_feature_budget(self.feature_budget);
                    (&sift, Some(ExtractionMethod::Sift))
                }
                (None, ExtractionMethod::Alp) => {
                    let alp_preset = self.sift.density().alp_preset().ok_or_else(|| {
                        crate::SfmError::InvalidConfig(
                            "ALP has no ultra detector density; raise the maximum image \
                             dimension for finer scales"
                                .to_owned(),
                        )
                    })?;
                    alp = crate::adapters::features::alp::AlpFeatureExtractor::new(preset)
                        .with_alp_preset(alp_preset)
                        .with_feature_budget(self.feature_budget)
                        .with_optional_max_dimension(self.sift.max_dimension());
                    (&alp, Some(ExtractionMethod::Alp))
                }
            };
        let mut report = crate::ops::frontend::run_frontend(
            workspace,
            extractor,
            matcher,
            &crate::adapters::verification::CalibratedVerifier::default(),
            radial_calibration,
            self.max_pairs_per_view,
            progress,
        )?;
        report.extraction_method = extraction_method;
        crate::progress::cancellation_point(progress)?;
        if report.verified_pair_count == 0 {
            return Err(crate::SfmError::InvalidInput(format!(
                "none of the {} image pairs passed geometric verification",
                workspace.pair_graph().len()
            )));
        }
        if report.track_count == 0 {
            return Err(crate::SfmError::InvalidInput(
                "the verified image pairs produced no feature tracks".to_owned(),
            ));
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_frontend_is_rejected_and_quality_keeps_its_wire_shape() {
        let wire = serde_json::json!({
            "sift": {"preset": "compact", "max_dimension": 1600},
            "matching_method": "brute_force",
        });
        let error = serde_json::from_value::<Frontend>(wire).unwrap_err();
        assert!(error.to_string().contains("unknown variant `compact`"));

        let standard = Frontend::default();
        let wire = serde_json::to_value(standard).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "sift": {"preset": "quality", "density": "high", "max_dimension": 1600},
                "extraction_method": "alp",
                "matching_method": "hnsw",
                "radial_calibration": true,
                "max_pairs_per_view": null,
            })
        );
        let restored: Frontend = serde_json::from_value(wire).unwrap();
        assert_eq!(restored, standard);
        assert_eq!(restored.density(), crate::DetectorDensity::High);
        assert_eq!(restored.feature_budget(), FeatureBudgetPolicy::Unlimited);
    }

    #[test]
    fn shared_density_and_budget_round_trip_for_both_methods() {
        let budget = FeatureBudgetPolicy::spatial_four_by_four(32);
        let frontend = Frontend::default()
            .with_density(crate::DetectorDensity::Normal)
            .with_feature_budget(budget);
        assert_eq!(frontend.extraction_method(), ExtractionMethod::Alp);
        assert_eq!(frontend.density(), crate::DetectorDensity::Normal);
        assert_eq!(frontend.feature_budget(), budget);
        let wire = serde_json::to_value(frontend).unwrap();
        assert_eq!(wire["sift"]["density"], "normal");
        assert_eq!(
            wire["feature_budget"],
            serde_json::to_value(budget).unwrap()
        );
        assert_eq!(serde_json::from_value::<Frontend>(wire).unwrap(), frontend);
        let sift = frontend.with_extraction_method(ExtractionMethod::Sift);
        assert_eq!(sift.density(), crate::DetectorDensity::Normal);
        assert_eq!(sift.feature_budget(), budget);
        assert_eq!(sift.extraction_method(), ExtractionMethod::Sift);
    }

    #[test]
    fn legacy_serde_without_method_keeps_sift_and_unlimited() {
        let legacy = serde_json::json!({
            "sift": {"preset": "quality", "max_dimension": 1600},
        });
        let restored: Frontend = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.extraction_method(), ExtractionMethod::Sift);
        assert_eq!(restored.feature_budget(), FeatureBudgetPolicy::Unlimited);
        assert_eq!(restored.density(), crate::DetectorDensity::Normal);
    }
}
