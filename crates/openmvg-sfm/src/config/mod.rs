// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{ViewId, ViewPair};
use openmvg_core::{AngularResidual, AngularThreshold, PixelThreshold};

pub use crate::pipelines::global::GlobalSfmConfig;

pub(crate) mod internal;

pub(crate) use internal::{GlobalTranslationScope, WorkspaceBuildConfig};

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
/// How the similarity scale of a reconstruction component is fixed.
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum GaugeScalePolicy {
    /// Scale is left free; bundle adjustment uses its free-scale gauge.
    #[default]
    Free,
    /// The camera-center distance between two registered views is held fixed.
    FixedBaseline {
        /// First view, which anchors the gauge.
        first: ViewId,
        /// Second view, whose center distance to `first` is held; distinct from `first`.
        second: ViewId,
        /// Center-to-center distance in `unit`; finite and positive.
        distance: f64,
        /// Unit of `distance`, recorded with the policy.
        unit: LengthUnit,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Unit attached to a [`GaugeScalePolicy::FixedBaseline`] distance.
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum LengthUnit {
    /// Meters.
    Meters,
    /// Centimeters.
    Centimeters,
    /// Millimeters.
    Millimeters,
    /// Unitless reconstruction scale (for example the unit baseline of the initial pair).
    Arbitrary,
}

/// Global SfM averaging policy: gauge scale, translation-core selection, and parallax gate.
///
/// The translation core is the largest parallel-rigid, biconnected block of the translation
/// graph. Recovery candidates restrict translation averaging to it; it is distinct from the
/// final core described on [`crate::SfmEngine::Global`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlobalAveragingConfig {
    gauge_scale_policy: GaugeScalePolicy,
    #[serde(default)]
    min_edge_isotropy: f64,
    #[serde(
        default,
        rename = "min_translation_parallax_deg",
        with = "optional_angular_residual_degrees"
    )]
    min_translation_parallax: Option<AngularResidual>,
}

config_accessors!(GlobalAveragingConfig {
    /// How the similarity gauge scale of the reconstruction is fixed.
    copy gauge_scale_policy, with_gauge_scale_policy: GaugeScalePolicy;
    /// Optional edge-loss directional-isotropy floor used while selecting the
    /// translation core. Zero disables stability
    /// pruning. The valid scale-invariant range is `[0, 1/3]`.
    copy min_edge_isotropy, with_min_edge_isotropy: f64;
    /// Minimum parallax for an edge to enter translation averaging.
    ///
    /// When `Some(d)`, edges whose median per-inlier parallax angle (the angle between bearing
    /// `b_i` and `R_ijᵀ·b_j`) falls below `d` are excluded from the translation solve only.
    /// Such edges are planar or near-pure-rotation, with no metric baseline: their
    /// relative-translation direction is degenerate, and the direction-only L1-IRLS solve
    /// collapses the scale if it trusts them. Rotation averaging still uses every edge, since
    /// low-parallax pairs still give good rotations. Dropped edges are re-added, highest
    /// parallax first, wherever the filter would otherwise leave a camera without translation
    /// edges. `None` (the default) disables the gate.
    copy min_translation_parallax, with_min_translation_parallax: Option<AngularResidual>;
});

impl GlobalAveragingConfig {
    /// Check the policy's numeric ranges.
    ///
    /// # Errors
    ///
    /// [`crate::SfmError::InvalidConfig`] if a fixed-baseline distance is not finite and positive or
    /// the edge-loss isotropy floor is outside `[0, 1/3]`.
    pub fn validate(&self) -> crate::Result<()> {
        if let GaugeScalePolicy::FixedBaseline { distance, .. } = self.gauge_scale_policy
            && (!distance.is_finite() || distance <= 0.0)
        {
            return Err(crate::SfmError::InvalidConfig(
                "fixed-baseline gauge distance must be finite and positive".to_owned(),
            ));
        }
        if !self.min_edge_isotropy.is_finite()
            || !(0.0..=1.0 / 3.0).contains(&self.min_edge_isotropy)
        {
            return Err(crate::SfmError::InvalidConfig(
                "min_edge_isotropy must be finite and in [0, 1/3]".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for GlobalAveragingConfig {
    fn default() -> Self {
        Self {
            gauge_scale_policy: GaugeScalePolicy::Free,
            min_edge_isotropy: 0.0,
            min_translation_parallax: None,
        }
    }
}

/// Relative landmark-support band considered before spatially ranking the growth frontier.
pub(crate) const INCREMENTAL_FRONTIER_SUPPORT_RATIO: f64 = 0.76;
/// Registration-support floor used to screen the frontiers of initial-pair candidates. The
/// default growth threshold is the stricter [`IncrementalSfmConfig::min_registration_support`].
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub(crate) const DEFAULT_INCREMENTAL_REGISTRATION_SUPPORT: usize = 6;

/// Incremental SfM policy: initial-pair selection, registration thresholds, triangulation, and
/// bundle adjustment (BA) cadence.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
// Removed experimental controls must fail loudly instead of becoming accepted no-ops.
#[serde(deny_unknown_fields)]
pub struct IncrementalSfmConfig {
    initial_pair: InitialPairPolicy,
    min_registration_support: usize,
    registration_min_inliers: usize,
    registration_min_inlier_ratio: f64,
    #[serde(default = "default_registration_min_verified_neighbors")]
    registration_min_verified_neighbors: usize,
    triangulation_min_observations: usize,
    retriangulate: bool,
    max_reprojection_error: Option<PixelThreshold>,
    #[serde(default)]
    bundle_adjust_initial_pair: bool,
    bundle_adjust_every: usize,
    #[serde(default)]
    pub(crate) periodic_ba_max_iterations: usize,
    #[serde(default)]
    bundle_adjust_growth_ratio: f64,
    self_calibrate_intrinsics: bool,
    max_iterations: usize,
    #[serde(
        rename = "registration_max_rotation_edge_error_deg",
        with = "optional_angular_threshold_degrees"
    )]
    registration_max_rotation_edge_error: Option<AngularThreshold>,
    #[serde(
        rename = "registration_max_translation_edge_error_deg",
        with = "optional_angular_threshold_degrees"
    )]
    registration_max_translation_edge_error: Option<AngularThreshold>,
}

config_accessors!(IncrementalSfmConfig {
    /// Initial-pair selection strategy (default: best verified pair).
    ref initial_pair, with_initial_pair: InitialPairPolicy;
    /// Minimum number of observations of already triangulated points a view needs before
    /// Incremental tries to register it.
    copy min_registration_support, with_min_registration_support: usize;
    /// Minimum RANSAC inliers required to accept a registered pose.
    copy registration_min_inliers, with_registration_min_inliers: usize;
    /// Minimum inlier ratio of the absolute pose (inliers / attempted 2D-3D observations) to
    /// accept a pose estimated with P3P, or with P4Pf for the first view of a camera whose focal
    /// length is a guess. It rejects well-connected but wrong poses that fit only a few of their
    /// many observations and would corrupt retriangulation. `0.0` disables it.
    copy registration_min_inlier_ratio, with_registration_min_inlier_ratio: f64;
    /// Minimum verified-pair neighbors that must already be registered before a
    /// growth candidate may be resected. `0` lets a view join through shared tracks alone; `1`
    /// requires a verified pair with a registered view, which keeps weakly attached chains of
    /// views out.
    copy registration_min_verified_neighbors, with_registration_min_verified_neighbors: usize;
    /// Minimum posed observations required to triangulate a new track.
    copy triangulation_min_observations, with_triangulation_min_observations: usize;
    /// After each successful registration, re-triangulate touched landmarks from their retained
    /// support plus the new view, and grow new tracks. This sharpens 3D points without restoring
    /// observations rejected by an earlier cleanup pass.
    copy retriangulate, with_retriangulate: bool;
    /// Reprojection-error ceiling, in pixels, for keeping a (re-)triangulated landmark. `None`
    /// disables the drop step.
    copy max_reprojection_error, with_max_reprojection_error: Option<PixelThreshold>;
    /// Refine the initial pair immediately after robust triangulation, before growth. The first
    /// pose of the pair and the unit baseline fix the frame and the scale; intrinsics remain
    /// fixed while the second pose and the initial points are optimized.
    copy bundle_adjust_initial_pair, with_bundle_adjust_initial_pair: bool;
    /// Run a full bundle adjustment every N successful registrations. `0` disables the fixed cadence, except when self-calibration requests an
    /// automatic cadence; adaptive growth BA remains independently controlled below.
    copy bundle_adjust_every, with_bundle_adjust_every: usize;
    /// Run a full bundle adjustment when either registered views or active landmarks have
    /// grown by this fraction since the previous growth-stage attempt. `0.0` disables adaptive
    /// bundle adjustment. For example, `0.1` is a COLMAP-style 10% growth policy.
    copy bundle_adjust_growth_ratio, with_bundle_adjust_growth_ratio: f64;
    /// Refine each shared intrinsic group's declared camera model (focal, plus k1 or k1 and k2
    /// for radial models) during periodic BA and before the final refinement. Principal point
    /// and fx/fy aspect ratio stay fixed; groups whose observations do not constrain those
    /// parameters keep their input intrinsics. When enabled with `bundle_adjust_every == 0`,
    /// the engine bundle-adjusts after every 8 registered views, so calibration improves during
    /// growth rather than after distorted geometry has already been absorbed.
    copy self_calibrate_intrinsics, with_self_calibrate_intrinsics: bool;
    /// Hard cap on registration-candidate evaluations. `0` selects a safety budget of the
    /// squared image count plus one.
    copy max_iterations, with_max_iterations: usize;
    /// Rotation check: reject a resected pose whose rotation disagrees (median over posed
    /// neighbors) with its verified relative rotations by more than this many degrees.
    /// `None` = off. It catches a wrong pose that the 2D-3D matches alone accept.
    copy registration_max_rotation_edge_error, with_registration_max_rotation_edge_error: Option<AngularThreshold>;
    /// Optional translation check: reject a resected pose whose camera center disagrees
    /// (median world-frame direction over posed neighbors) with its relative-translation
    /// edges by more than this many degrees. Enable this only when those edges have been
    /// conditioned: raw pairwise directions can be noisier than a well-supported absolute pose
    /// and otherwise fragment the reconstruction. `None` = off (the default).
    copy registration_max_translation_edge_error, with_registration_max_translation_edge_error: Option<AngularThreshold>;
});

impl IncrementalSfmConfig {
    /// Maximum optimizer iterations for each periodic growth-stage BA. `0` selects the default:
    /// 50, or 25 when the camera models are self-calibrated. The final refinement keeps its full
    /// budget.
    #[must_use]
    pub const fn periodic_ba_max_iterations(&self) -> usize {
        self.periodic_ba_max_iterations
    }

    /// Set [`Self::periodic_ba_max_iterations`]; `0` selects the default.
    #[must_use]
    pub const fn with_periodic_ba_max_iterations(
        mut self,
        periodic_ba_max_iterations: usize,
    ) -> Self {
        self.periodic_ba_max_iterations = periodic_ba_max_iterations;
        self
    }
}

impl Default for IncrementalSfmConfig {
    /// The recommended incremental policy used by [`crate::Incremental`].
    fn default() -> Self {
        Self {
            initial_pair: InitialPairPolicy::BestPair(InitialPairConfig::default()),
            // Conservative thresholds, accuracy first. Where the frontier stalls, coverage is
            // limited by structure rather than by these thresholds: unregistered views never see
            // enough triangulated structure to become candidates, so relaxing the thresholds adds
            // no coverage there and can trade accuracy for marginal coverage on sparse scenes.
            // Start from twelve independent 2D-3D observations. Once that ordinary frontier is
            // exhausted, growth may consider an eight-observation bridge while retaining the
            // inlier-ratio, pose-edge, BA, and cleanup safeguards.
            min_registration_support: 12,
            registration_min_inliers: 8,
            registration_min_inlier_ratio: 0.25,
            // A 2D-3D intersection assembled transitively through tracks is not, by
            // itself, a view-graph anchor. Requiring one active verified edge prevents
            // locally plausible but globally floating tails. Callers can set zero for a
            // coverage-first policy.
            registration_min_verified_neighbors: default_registration_min_verified_neighbors(),
            triangulation_min_observations: 2,
            retriangulate: true,
            max_reprojection_error: Some(PixelThreshold::FOUR_PIXELS),
            bundle_adjust_initial_pair: true,
            bundle_adjust_every: 0,
            periodic_ba_max_iterations: 0,
            // Keep growth refinement conservative: less frequent adjustment can admit coherent
            // but globally wrong pose drift on dense scenes.
            bundle_adjust_growth_ratio: 0.1,
            // Images arrive with a declared camera model whose focal and distortion are estimates.
            self_calibrate_intrinsics: true,
            max_iterations: 0,
            // Fold check: reject a resected pose whose rotation disagrees (median over
            // posed neighbors) with its relative-rotation edges by more than this. 20° leaves a
            // clear margin above the edge noise: it catches gross folds and removes inconsistent
            // views that degrade accuracy, at a coverage cost on views that cannot be placed
            // consistently.
            registration_max_rotation_edge_error: Some(AngularThreshold::TWENTY_DEGREES),
            // Raw translation directions are not a reliable registration veto: a tight
            // translation check rejects P3P poses that many active landmarks support, which
            // shrinks the reconstruction and worsens its accuracy. Keep the option for callers
            // that condition the relative-pose graph; the rotation fold check above protects
            // sparse tails.
            registration_max_translation_edge_error: None,
        }
    }
}

impl IncrementalSfmConfig {
    /// Check the policy's numeric ranges.
    ///
    /// # Errors
    ///
    /// [`crate::SfmError::InvalidConfig`] if `registration_min_inlier_ratio` is not finite and in
    /// `[0, 1]`, or `bundle_adjust_growth_ratio` is not finite and non-negative.
    pub fn validate(&self) -> crate::Result<()> {
        if !self.registration_min_inlier_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.registration_min_inlier_ratio)
        {
            return Err(crate::SfmError::InvalidConfig(
                "registration_min_inlier_ratio must be finite and in [0, 1]".to_owned(),
            ));
        }
        if !self.bundle_adjust_growth_ratio.is_finite() || self.bundle_adjust_growth_ratio < 0.0 {
            return Err(crate::SfmError::InvalidConfig(
                "bundle_adjust_growth_ratio must be finite and non-negative".to_owned(),
            ));
        }
        Ok(())
    }

    /// The per-registration config derived from the driver config.
    #[cfg(feature = "unstable-internals")]
    pub fn registration_config(&self) -> crate::pipelines::incremental::RegistrationConfig {
        crate::pipelines::incremental::RegistrationConfig::default()
            .with_min_observations(self.min_registration_support)
            .with_min_inliers(self.registration_min_inliers)
            .with_min_inlier_ratio(self.registration_min_inlier_ratio)
            .with_max_rotation_edge_error(self.registration_max_rotation_edge_error)
            .with_max_translation_edge_error(self.registration_max_translation_edge_error)
    }

    #[cfg(not(feature = "unstable-internals"))]
    pub(crate) fn registration_config(&self) -> crate::pipelines::incremental::RegistrationConfig {
        crate::pipelines::incremental::RegistrationConfig::default()
            .with_min_observations(self.min_registration_support)
            .with_min_inliers(self.registration_min_inliers)
            .with_min_inlier_ratio(self.registration_min_inlier_ratio)
            .with_max_rotation_edge_error(self.registration_max_rotation_edge_error)
            .with_max_translation_edge_error(self.registration_max_translation_edge_error)
    }

    /// The growth-candidate config derived from the driver config.
    #[cfg(feature = "unstable-internals")]
    pub fn growth_config(&self) -> crate::pipelines::incremental::GrowthConfig {
        crate::pipelines::incremental::GrowthConfig::default()
            .with_min_visible_landmarks(self.min_registration_support)
            .with_min_verified_registered_neighbors(self.registration_min_verified_neighbors)
    }

    #[cfg(not(feature = "unstable-internals"))]
    pub(crate) fn growth_config(&self) -> crate::pipelines::incremental::GrowthConfig {
        crate::pipelines::incremental::GrowthConfig::default()
            .with_min_visible_landmarks(self.min_registration_support)
            .with_min_verified_registered_neighbors(self.registration_min_verified_neighbors)
    }
}

const fn default_registration_min_verified_neighbors() -> usize {
    1
}

mod optional_angular_threshold_degrees {
    use openmvg_core::AngularThreshold;

    pub fn serialize<S>(value: &Option<AngularThreshold>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde::Serialize::serialize(
            &value.map(|threshold| threshold.value().to_degrees()),
            serializer,
        )
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<AngularThreshold>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = <Option<f64> as serde::Deserialize>::deserialize(deserializer)?;
        value
            .map(AngularThreshold::from_degrees)
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

mod optional_angular_residual_degrees {
    use openmvg_core::AngularResidual;

    pub fn serialize<S>(value: &Option<AngularResidual>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde::Serialize::serialize(
            &value.map(|residual| residual.value().to_degrees()),
            serializer,
        )
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<AngularResidual>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = <Option<f64> as serde::Deserialize>::deserialize(deserializer)?;
        value
            .map(AngularResidual::from_degrees)
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// How the Incremental engine chooses its initial pair.
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum InitialPairPolicy {
    /// Use one caller-selected, active, verified pair.
    ///
    /// This is mainly useful for reproducible experiments: it keeps the initial
    /// pair fixed while registration, triangulation, or BA policies are
    /// changed independently. The pair must still be an active verified pair of the graph.
    ExplicitPair(ViewPair),
    /// Rank every active verified pair passing [`InitialPairConfig`] and start from the best one.
    BestPair(InitialPairConfig),
}

/// Eligibility filter for automatically ranked initial pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InitialPairConfig {
    min_inliers: usize,
}

config_accessors!(InitialPairConfig {
    /// Minimum verified inliers a pair needs to be considered as the initial pair.
    copy min_inliers, with_min_inliers: usize;
});

impl Default for InitialPairConfig {
    fn default() -> Self {
        Self { min_inliers: 8 }
    }
}
