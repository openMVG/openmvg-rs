// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration of the Global engine's candidate cascade and its track validation.

use crate::config::GlobalAveragingConfig;
use crate::pipelines::global::{
    GlobalBaIterationLimits, GlobalPostFixedRotationCleanupConfig,
    PretrackedGlobalPositioningConfig,
};
use crate::{Result, SfmError};

use super::policy::GlobalPolicy;

/// Fast, bounded validation of an admitted core, as defined on [`crate::SfmEngine::Global`].
///
/// Core rejection uses camera-balanced median holdout error. Aggregate p90 and triangulation
/// rejection remain telemetry because both are dominated by a minority of bad or low-parallax
/// tracks even when the recovered camera field is accurate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GlobalTrackValidationConfig {
    validation: crate::ops::track_scale::TrackScaleValidationConfig,
    minimum_evaluated_holdouts: usize,
    max_camera_median_deg: f64,
    max_inconsistent_camera_fraction: f64,
    max_peeled_camera_fraction: f64,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl GlobalTrackValidationConfig {
    /// Held-out track sampling and re-triangulation limits.
    #[must_use]
    pub const fn validation(&self) -> crate::ops::track_scale::TrackScaleValidationConfig {
        self.validation
    }

    /// Minimum evaluated held-outs before the core can be rejected.
    #[must_use]
    pub const fn minimum_evaluated_holdouts(&self) -> usize {
        self.minimum_evaluated_holdouts
    }

    /// Per-camera median held-out angular error limit, in degrees.
    #[must_use]
    pub const fn max_camera_median_deg(&self) -> f64 {
        self.max_camera_median_deg
    }

    /// Maximum admitted fraction of supported cameras above the median limit.
    #[must_use]
    pub const fn max_inconsistent_camera_fraction(&self) -> f64 {
        self.max_inconsistent_camera_fraction
    }

    /// Largest fraction of supported cameras that may be peeled off a rejected core.
    ///
    /// A core rejected for inconsistent cameras is not discarded outright: when the inconsistent
    /// cameras are a tail no larger than this fraction, they are removed and the rest is
    /// validated again. A larger share means the core itself is wrong, not just its tail.
    #[must_use]
    pub const fn max_peeled_camera_fraction(&self) -> f64 {
        self.max_peeled_camera_fraction
    }
}

impl Default for GlobalTrackValidationConfig {
    fn default() -> Self {
        Self {
            validation: crate::ops::track_scale::TrackScaleValidationConfig::default(),
            minimum_evaluated_holdouts: 64,
            max_camera_median_deg: 0.5,
            max_inconsistent_camera_fraction: 0.05,
            max_peeled_camera_fraction: 0.25,
        }
    }
}

impl GlobalTrackValidationConfig {
    /// Check the limits.
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidConfig`] when the nested validation config is invalid, the held-out
    /// minimum is zero, the camera-median limit is not finite and positive, or the camera
    /// fraction is outside `[0, 1]`.
    pub fn validate(self) -> Result<()> {
        self.validation.validate()?;
        if self.minimum_evaluated_holdouts == 0 {
            return Err(SfmError::InvalidConfig(
                "global track validation requires evaluated holdouts".to_owned(),
            ));
        }
        if !self.max_camera_median_deg.is_finite() || self.max_camera_median_deg <= 0.0 {
            return Err(SfmError::InvalidConfig(
                "global track validation camera-median limit must be finite and positive"
                    .to_owned(),
            ));
        }
        if !self.max_peeled_camera_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_peeled_camera_fraction)
        {
            return Err(SfmError::InvalidConfig(
                "global track validation peeled camera fraction must be in [0, 1]".to_owned(),
            ));
        }
        if !self.max_inconsistent_camera_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_inconsistent_camera_fraction)
        {
            return Err(SfmError::InvalidConfig(
                "global track validation camera fraction must be in [0, 1]".to_owned(),
            ));
        }
        Ok(())
    }

    /// Whether `report` rejects the core: at least `minimum_evaluated_holdouts` held-outs were
    /// evaluated and more than `max_inconsistent_camera_fraction` of the supported cameras have a
    /// median held-out angular error above `max_camera_median_deg`.
    ///
    /// Cameras below the validation's minimum camera support or without a residual are not
    /// counted; a report with too few held-outs or no supported camera never rejects.
    #[must_use]
    pub fn core_is_rejected(
        self,
        report: &crate::ops::track_scale::TrackScaleValidationReport,
    ) -> bool {
        let (supported, inconsistent) = self.inconsistent_views(report);
        report.evaluated_holdout_count >= self.minimum_evaluated_holdouts
            && supported > 0
            && inconsistent.len() as f64 / supported as f64 > self.max_inconsistent_camera_fraction
    }

    /// The cameras to peel off a core that `rejected` rejects, or `None` when the core must be
    /// discarded whole.
    ///
    /// The inconsistent cameras are peeled only if they are a tail in both independent
    /// measures: at most `max_peeled_camera_fraction` of the supported cameras disagree with
    /// held-out tracks, and at most that fraction of the `graph_camera_count` cameras evaluated
    /// against the verified relative poses (`graph_inconsistent_count` of them) disagree with
    /// those. A core wrong as a whole fails one of the two, and dropping the cameras that expose
    /// it would leave sub-reconstructions that validate separately but are not aligned.
    #[must_use]
    pub fn peelable_views(
        self,
        rejected: &crate::ops::track_scale::TrackScaleValidationReport,
        graph_camera_count: usize,
        graph_inconsistent_count: usize,
    ) -> Option<Vec<crate::ViewId>> {
        let (supported, tail) = self.inconsistent_views(rejected);
        let tail_in_tracks =
            tail.len() as f64 <= self.max_peeled_camera_fraction * supported as f64;
        let tail_in_graph = graph_inconsistent_count as f64
            <= self.max_peeled_camera_fraction * graph_camera_count as f64;
        (!tail.is_empty() && tail_in_tracks && tail_in_graph).then_some(tail)
    }

    /// The number of supported cameras in `report` and the ones whose median held-out angular
    /// error exceeds `max_camera_median_deg`.
    #[must_use]
    pub fn inconsistent_views(
        self,
        report: &crate::ops::track_scale::TrackScaleValidationReport,
    ) -> (usize, Vec<crate::ViewId>) {
        let mut supported = 0usize;
        let mut inconsistent = Vec::new();
        for camera in &report.cameras {
            if camera.evaluated_holdout_count < self.validation.minimum_camera_support() {
                continue;
            }
            let Some(residual) = camera.angular_residual_deg.as_ref() else {
                continue;
            };
            supported += 1;
            if residual.median_deg > self.max_camera_median_deg {
                inconsistent.push(camera.view);
            }
        }
        (supported, inconsistent)
    }
}

/// Configuration of the Global engine's cascade, which tries joint global positioning first and
/// direction-only translation averaging second.
///
/// The engine prepares the view graph and the tracks once, before the cascade, so both
/// candidates see the same input, and the first candidate that passes the admission checks
/// described on [`Global`](crate::Global#how-it-works) is returned as the core.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlobalSfmConfig {
    pub(crate) policy: GlobalPolicy,
    averaging: GlobalAveragingConfig,
    pub(crate) global_positioning: PretrackedGlobalPositioningConfig,
    pub(crate) independent_track_validation: Option<GlobalTrackValidationConfig>,
    release_matching_data: bool,
    pub(crate) post_fixed_rotation_cleanup: GlobalPostFixedRotationCleanupConfig,
    pub(crate) ba_iteration_limits: GlobalBaIterationLimits,
    self_calibrate_intrinsics: bool,
    max_reprojection_error: openmvg_core::PixelThreshold,
    max_robust_rounds: usize,
}

config_accessors!(GlobalSfmConfig {
    /// Rotation and translation averaging policy.
    copy averaging, with_averaging: GlobalAveragingConfig;
    /// Candidate admission thresholds: rotation consensus, core and post-fit trust.
    #[cfg(feature = "unstable-internals")]
    copy policy, with_policy: GlobalPolicy;
    /// Optional independent track-domain validation after global refinement.
    #[cfg(feature = "unstable-internals")]
    copy independent_track_validation, with_independent_track_validation: Option<GlobalTrackValidationConfig>;
    /// Release feature and match payloads after the last structural check that reads them, to
    /// lower the peak memory of large runs. Stable tracks and verified models remain intact,
    /// and the result is the same either way.
    copy release_matching_data, with_release_matching_data: bool;
    /// Conservative camera deferral between fixed-rotation BA and released-rotation BA.
    /// Enabled by the recommended default; only the adaptive global driver supplies it to the
    /// global finish.
    #[cfg(feature = "unstable-internals")]
    copy post_fixed_rotation_cleanup, with_post_fixed_rotation_cleanup: GlobalPostFixedRotationCleanupConfig;
    /// Iteration budgets for released-rotation Cauchy BA stages.
    #[cfg(feature = "unstable-internals")]
    copy ba_iteration_limits, with_ba_iteration_limits: GlobalBaIterationLimits;
    /// Refine the admitted candidate with a bundle adjustment that frees each intrinsic group's
    /// declared camera model (focal, plus k1 or k1 and k2 for radial models; the principal point
    /// stays fixed), then retriangulate and clean up. Groups whose observations do not constrain
    /// those parameters keep their input intrinsics. The refinement is rolled back when it
    /// degrades the verified relative-pose graph.
    copy self_calibrate_intrinsics, with_self_calibrate_intrinsics: bool;
    /// Reprojection-error ceiling used by core selection and the global finish.
    copy max_reprojection_error, with_max_reprojection_error: openmvg_core::PixelThreshold;
    /// Robust retriangulation and refit rounds after the full bundle adjustment (at most three).
    copy max_robust_rounds, with_max_robust_rounds: usize;
});

impl GlobalSfmConfig {
    /// Data budget of the pretracked global positioning candidate.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn global_positioning(&self) -> PretrackedGlobalPositioningConfig {
        self.global_positioning
    }
}

impl Default for GlobalSfmConfig {
    /// The recommended policy used by [`crate::Global`]: joint global positioning first, then
    /// direction-only translation averaging.
    ///
    /// Because the struct is `#[serde(default)]`, fields omitted from a serialized config also
    /// take these values.
    fn default() -> Self {
        let averaging = GlobalAveragingConfig::default().with_min_edge_isotropy(0.0);

        Self {
            policy: GlobalPolicy::default(),
            averaging,
            global_positioning: PretrackedGlobalPositioningConfig::default()
                .with_max_observations(50_000)
                .with_min_track_views(3)
                .with_direction_reweight(1.0),
            independent_track_validation: Some(GlobalTrackValidationConfig::default()),
            release_matching_data: true,
            post_fixed_rotation_cleanup: GlobalPostFixedRotationCleanupConfig::default()
                .with_enabled(true),
            ba_iteration_limits: GlobalBaIterationLimits::default(),
            self_calibrate_intrinsics: true,
            max_reprojection_error: crate::pipelines::global::GLOBAL_MAX_REPROJECTION_ERROR_PX,
            max_robust_rounds: crate::pipelines::global::GLOBAL_ROBUST_ROUNDS,
        }
    }
}

impl GlobalSfmConfig {
    /// Check the averaging, policy, positioning, core, cleanup, and BA settings together.
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidConfig`] (or the nested config's error) when a nested config is invalid,
    /// when the global positioning settings are invalid (tracks of fewer than two views, or a
    /// direction weight that is not finite and positive), or when more than three robust rounds
    /// are requested.
    pub fn validate(&self) -> Result<()> {
        self.averaging.validate()?;
        self.policy.validate()?;

        if self.global_positioning.min_track_views() < 2 {
            return Err(SfmError::InvalidConfig(
                "global positioning minimum track views must be at least two".to_owned(),
            ));
        }
        if !self.global_positioning.direction_reweight().is_finite()
            || self.global_positioning.direction_reweight() <= 0.0
        {
            return Err(SfmError::InvalidConfig(
                "global positioning direction reweight must be finite and positive".to_owned(),
            ));
        }
        if let Some(config) = self.independent_track_validation {
            config.validate()?;
        }
        self.post_fixed_rotation_cleanup.validate()?;
        self.ba_iteration_limits.validate()?;
        if self.max_robust_rounds > 3 {
            return Err(SfmError::InvalidConfig(
                "global robust rounds are bounded to at most three".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track_validation_report(
        camera_medians_deg: &[f64],
    ) -> crate::ops::track_scale::TrackScaleValidationReport {
        let evaluated_holdout_count = camera_medians_deg.len() * 20;
        crate::ops::track_scale::TrackScaleValidationReport {
            schema_version: 1,
            config: crate::ops::track_scale::TrackScaleValidationConfig::default(),
            posed_camera_count: camera_medians_deg.len(),
            input_track_count: 100,
            eligible_track_count: 100,
            selected_track_count: 100,
            attempted_holdout_count: evaluated_holdout_count,
            evaluated_holdout_count,
            triangulation_rejection_count: 0,
            invalid_prediction_count: 0,
            evaluated_holdout_fraction: Some(1.0),
            triangulation_rejection_fraction: Some(0.0),
            invalid_prediction_fraction: Some(0.0),
            positive_depth_count: evaluated_holdout_count,
            positive_depth_fraction: Some(1.0),
            angular_residual_deg: Some(crate::ops::averaging::AngularResidualDistribution {
                count: evaluated_holdout_count,
                minimum_deg: 0.0,
                median_deg: 0.1,
                p90_deg: 20.0,
                maximum_deg: 80.0,
            }),
            within_one_degree_fraction: Some(1.0),
            within_two_degree_fraction: Some(1.0),
            within_five_degree_fraction: Some(1.0),
            camera_tail_count: 0,
            cameras: camera_medians_deg
                .iter()
                .copied()
                .enumerate()
                .map(
                    |(view, median_deg)| crate::ops::track_scale::TrackScaleCameraReport {
                        view: crate::ViewId::new(view),
                        eligible_track_count: 20,
                        assigned_track_count: 20,
                        attempted_holdout_count: 20,
                        evaluated_holdout_count: 20,
                        triangulation_rejection_count: 0,
                        invalid_prediction_count: 0,
                        evaluated_holdout_fraction: Some(1.0),
                        triangulation_rejection_fraction: Some(0.0),
                        invalid_prediction_fraction: Some(0.0),
                        positive_depth_count: 20,
                        angular_residual_deg: Some(
                            crate::ops::averaging::AngularResidualDistribution {
                                count: 20,
                                minimum_deg: 0.0,
                                median_deg,
                                p90_deg: 20.0,
                                maximum_deg: 80.0,
                            },
                        ),
                        is_camera_tail: true,
                    },
                )
                .collect(),
            triangulation_rejections: Vec::new(),
        }
    }

    #[test]
    fn defaults_are_mixed_first_then_direction() {
        let config = GlobalSfmConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(
            config.policy.ordered_candidates(),
            [
                super::super::policy::GlobalCandidate::GlobalPositioning,
                super::super::policy::GlobalCandidate::TranslationAveraging,
            ]
        );
        assert_eq!(config.global_positioning.min_track_views(), 3);
        assert!(config.post_fixed_rotation_cleanup.enabled());
        assert_eq!(
            config.ba_iteration_limits,
            GlobalBaIterationLimits::default()
        );
        assert_eq!(config.max_robust_rounds, 3);
        assert_eq!(
            config.independent_track_validation,
            Some(GlobalTrackValidationConfig::default())
        );
        assert!(config.release_matching_data);
        assert!(config.self_calibrate_intrinsics);
    }

    #[test]
    fn independent_track_gate_uses_camera_balanced_medians() {
        let policy = GlobalTrackValidationConfig::default();
        assert!(!policy.core_is_rejected(&track_validation_report(&[0.1; 20])));

        let mut boundary = [0.1; 20];
        boundary[0] = 2.0;
        assert!(!policy.core_is_rejected(&track_validation_report(&boundary)));

        let mut rejected = boundary;
        rejected[1] = 1.0;
        assert!(policy.core_is_rejected(&track_validation_report(&rejected)));
    }

    #[test]
    fn only_a_tail_in_both_measures_is_peeled() {
        let policy = GlobalTrackValidationConfig::default();
        // Three of twenty cameras (15%) are inconsistent: rejected, and within the 25% tail bound.
        let mut medians = [0.1; 20];
        medians[..3].fill(2.0);
        let report = track_validation_report(&medians);
        assert!(policy.core_is_rejected(&report));

        let peeled = policy.peelable_views(&report, 100, 3).unwrap();
        assert_eq!(peeled, (0..3).map(crate::ViewId::new).collect::<Vec<_>>());

        // The verified pair graph calls half of the cameras inconsistent: the core is wrong whole.
        assert_eq!(policy.peelable_views(&report, 100, 48), None);

        // Half of the cameras disagree with held-out tracks: not a tail.
        let report = track_validation_report(&[2.0; 20]);
        assert_eq!(policy.peelable_views(&report, 100, 3), None);

        // Nothing to peel from a core that passes.
        let report = track_validation_report(&[0.1; 20]);
        assert_eq!(policy.peelable_views(&report, 100, 3), None);
    }

    #[test]
    fn peeled_camera_fraction_must_be_a_fraction() {
        let default = GlobalTrackValidationConfig::default();
        assert!(default.validate().is_ok());
        let mut invalid = default;
        invalid.max_peeled_camera_fraction = 1.5;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn released_rotation_iteration_limits_must_be_positive() {
        let config = GlobalSfmConfig {
            ba_iteration_limits: GlobalBaIterationLimits::default()
                .with_released_rotation(0)
                .with_post_retriangulation(15),
            ..GlobalSfmConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
