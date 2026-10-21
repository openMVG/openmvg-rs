// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consensus, acceptance and policy configuration for track triangulation.

use crate::triangulation::TwoViewMethod;
use openmvg_core::{AngularResidual, AngularThreshold, PixelThreshold, WorldDistanceResidual};

use super::TriangulationError;

/// Named configurations: a robust consensus mode and a clean-data mode.
///
/// Reached publicly through [`TrackTriangulator::robust`],
/// [`TrackTriangulator::all_observations`] and
/// [`TrackTriangulationConfig::default`].
///
/// [`TrackTriangulator::robust`]: super::TrackTriangulator::robust
/// [`TrackTriangulator::all_observations`]: super::TrackTriangulator::all_observations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum TriangulationPreset {
    /// Deterministic pair consensus using angular residuals. Recommended.
    #[default]
    Robust,
    /// Fit every observation once and accept or reject the complete track.
    AllObservations,
}

/// Typed residual threshold used to classify observations during consensus.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ConsensusThreshold {
    /// Angular residual threshold in radians.
    Angular(AngularThreshold),
    /// Camera-model reprojection threshold in pixels.
    Pixels(PixelThreshold),
}

/// Deterministic consensus controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsensusConfig {
    pub(super) threshold: ConsensusThreshold,
    pub(super) min_inliers: usize,
    pub(super) min_inlier_ratio: f64,
    pub(super) pair_budget: PairHypothesisBudget,
}

config_accessors!(ConsensusConfig {
    /// Residual metric and its inlier threshold.
    copy threshold, with_threshold: ConsensusThreshold;
    /// Minimum observations supporting an accepted point.
    copy min_inliers, with_min_inliers: usize;
    /// Minimum fraction of the input track supporting an accepted point.
    copy min_inlier_ratio, with_min_inlier_ratio: f64;
    /// Work budget for deterministic two-view hypothesis generation.
    copy pair_budget, with_pair_budget: PairHypothesisBudget;
});

impl Default for ConsensusConfig {
    /// The consensus controls of the recommended robust preset.
    fn default() -> Self {
        Self {
            threshold: ConsensusThreshold::Angular(AngularThreshold::QUARTER_DEGREE),
            min_inliers: 2,
            min_inlier_ratio: 0.5,
            pair_budget: PairHypothesisBudget::CONSERVATIVE,
        }
    }
}

impl ConsensusConfig {
    fn validate(self) -> Result<(), TriangulationError> {
        if self.min_inliers < 2 {
            return Err(TriangulationError::InvalidConfig(
                "consensus requires at least two inliers",
            ));
        }
        if !self.min_inlier_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.min_inlier_ratio)
            || self.min_inlier_ratio == 0.0
        {
            return Err(TriangulationError::InvalidConfig(
                "minimum inlier ratio must be in (0, 1]",
            ));
        }
        self.pair_budget.validate()?;
        Ok(())
    }
}

/// Bounded work policy for deterministic two-view hypotheses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairHypothesisBudget {
    pub(super) per_observation: usize,
    pub(super) min: usize,
    pub(super) max: usize,
}

config_accessors!(PairHypothesisBudget {
    /// Hypotheses requested per input observation.
    copy per_observation, with_per_observation: usize;
    /// Lower work bound when enough unique pairs exist.
    copy min, with_min: usize;
    /// Hard upper work bound for long tracks.
    copy max, with_max: usize;
});

impl PairHypothesisBudget {
    /// Conservative default for already-filtered feature tracks.
    ///
    /// The requested count is `clamp(2 * observations, 6, 16)`, capped by
    /// the number of unique pairs. Consequently, two-view tracks are handled
    /// directly, lengths three through five are exhaustive, and longer tracks
    /// never score more than sixteen pair hypotheses.
    ///
    /// Lowe's descriptor ratio is not an outlier probability, so this budget
    /// should be checked against the pipeline's observed track quality.
    pub const CONSERVATIVE: Self = Self {
        per_observation: 2,
        min: 6,
        max: 16,
    };

    /// Evaluate every unique observation pair.
    ///
    /// This policy is mainly useful as a reference for comparison. Other
    /// callers should normally use [`Self::CONSERVATIVE`] to keep work bounded
    /// on long tracks.
    pub const EXHAUSTIVE: Self = Self {
        per_observation: usize::MAX,
        min: 1,
        max: usize::MAX,
    };

    /// Number of unique pair hypotheses requested for a track length.
    pub fn for_observation_count(self, observations: usize) -> usize {
        let unique_pairs = observations.saturating_mul(observations.saturating_sub(1)) / 2;
        self.per_observation
            .saturating_mul(observations)
            .max(self.min)
            .min(self.max)
            .min(unique_pairs)
    }

    fn validate(self) -> Result<(), TriangulationError> {
        if self.per_observation == 0 {
            return Err(TriangulationError::InvalidConfig(
                "pair hypotheses per observation must be positive",
            ));
        }
        if self.min == 0 || self.max == 0 || self.min > self.max {
            return Err(TriangulationError::InvalidConfig(
                "pair hypothesis bounds must satisfy 0 < min <= max",
            ));
        }
        Ok(())
    }
}

impl Default for PairHypothesisBudget {
    fn default() -> Self {
        Self::CONSERVATIVE
    }
}

/// Final geometric rejection thresholds.
///
/// They check the final point, apart from `min_depth`, which also decides which observations
/// count as inliers during the consensus search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceConfig {
    pub(super) min_parallax: AngularResidual,
    pub(super) min_depth: WorldDistanceResidual,
    pub(super) max_angular_residual: AngularThreshold,
    pub(super) max_pixel_residual: Option<PixelThreshold>,
}

config_accessors!(AcceptanceConfig {
    /// Minimum conditioning angle attained by at least one supporting pair.
    copy min_parallax, with_min_parallax: AngularResidual;
    /// Minimum depth along every supporting ray, in the input scene's world-coordinate unit.
    /// An observation with a smaller depth is not an inlier during the consensus search either.
    copy min_depth, with_min_depth: WorldDistanceResidual;
    /// Maximum angular residual for every supporting observation.
    copy max_angular_residual, with_max_angular_residual: AngularThreshold;
    /// Optional maximum pixel residual for every supporting observation.
    copy max_pixel_residual, with_max_pixel_residual: Option<PixelThreshold>;
});

impl Default for AcceptanceConfig {
    fn default() -> Self {
        Self {
            min_parallax: AngularResidual::ONE_DEGREE,
            min_depth: WorldDistanceResidual::ZERO,
            max_angular_residual: AngularThreshold::HALF_DEGREE,
            max_pixel_residual: None,
        }
    }
}

impl AcceptanceConfig {
    fn validate(self) -> Result<(), TriangulationError> {
        if self.min_parallax.value() >= std::f64::consts::FRAC_PI_2 {
            return Err(TriangulationError::InvalidConfig(
                "minimum parallax must be below pi/2",
            ));
        }
        Ok(())
    }
}

/// Track estimation policy.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum TriangulationPolicy {
    /// Fit the complete track without observation filtering.
    AllObservations,
    /// Select observations using deterministic pair consensus.
    Consensus(ConsensusConfig),
}

/// Complete track-level triangulation configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackTriangulationConfig {
    pub(super) policy: TriangulationPolicy,
    pub(super) two_view_method: TwoViewMethod,
    pub(super) hypothesis_min_parallax: AngularResidual,
    pub(super) acceptance: AcceptanceConfig,
}

config_accessors!(TrackTriangulationConfig {
    /// Observation-selection policy.
    copy policy, with_policy: TriangulationPolicy;
    /// Numerical method used for two-ray hypotheses and two-view refits.
    copy two_view_method, with_two_view_method: TwoViewMethod;
    /// Pair geometry required before generating a hypothesis.
    copy hypothesis_min_parallax, with_hypothesis_min_parallax: AngularResidual;
    /// Final rejection policy.
    copy acceptance, with_acceptance: AcceptanceConfig;
});

impl TrackTriangulationConfig {
    /// Resolve a documented preset into an ordinary, fully editable configuration.
    pub(crate) fn from_preset(preset: TriangulationPreset) -> Self {
        let (policy, acceptance) = match preset {
            TriangulationPreset::Robust => (
                TriangulationPolicy::Consensus(ConsensusConfig::default()),
                AcceptanceConfig::default(),
            ),
            TriangulationPreset::AllObservations => (
                TriangulationPolicy::AllObservations,
                AcceptanceConfig {
                    max_pixel_residual: Some(PixelThreshold::FOUR_PIXELS),
                    ..AcceptanceConfig::default()
                },
            ),
        };
        Self {
            policy,
            two_view_method: TwoViewMethod::InverseDepthWeightedMidpoint,
            hypothesis_min_parallax: AngularResidual::QUARTER_DEGREE,
            acceptance,
        }
    }

    /// Validate all thresholds before processing tracks.
    pub fn validate(self) -> Result<(), TriangulationError> {
        if self.hypothesis_min_parallax.value() >= std::f64::consts::FRAC_PI_2 {
            return Err(TriangulationError::InvalidConfig(
                "hypothesis parallax must be below pi/2",
            ));
        }
        if let TriangulationPolicy::Consensus(config) = self.policy {
            config.validate()?;
        }
        self.acceptance.validate()
    }
}

impl Default for TrackTriangulationConfig {
    fn default() -> Self {
        Self::from_preset(TriangulationPreset::Robust)
    }
}
