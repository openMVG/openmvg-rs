// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated scheduling and advanced residual thresholds for GenRANSAC.

use crate::robust::error::RobustError;
use crate::robust::threshold::{
    AngularThreshold, NormalizedImageThreshold, PixelThreshold, WorldDistanceThreshold,
};
use openmvg_core::RunSeed;

/// Validated, unsquared threshold for an advanced custom solver's residual.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ResidualThreshold(f64);

impl ResidualThreshold {
    /// Validate and construct a threshold in the custom solver's residual units.
    #[cfg(test)]
    pub fn new(value: f64) -> Result<Self, RobustError> {
        validate_threshold(value, "solver residual units")?;
        Ok(Self(value))
    }

    /// Return the unsquared threshold value.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl From<NormalizedImageThreshold> for ResidualThreshold {
    fn from(threshold: NormalizedImageThreshold) -> Self {
        Self(threshold.value())
    }
}

impl From<AngularThreshold> for ResidualThreshold {
    fn from(threshold: AngularThreshold) -> Self {
        Self(threshold.value())
    }
}

impl From<PixelThreshold> for ResidualThreshold {
    fn from(threshold: PixelThreshold) -> Self {
        Self(threshold.value())
    }
}

impl From<WorldDistanceThreshold> for ResidualThreshold {
    fn from(threshold: WorldDistanceThreshold) -> Self {
        Self(threshold.value())
    }
}

#[cfg(test)]
fn validate_threshold(value: f64, unit: &str) -> Result<(), RobustError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RobustError::InvalidConfig {
            reason: format!("GenRANSAC threshold in {unit} must be finite and positive"),
        });
    }
    if value * value == f64::INFINITY {
        return Err(RobustError::InvalidConfig {
            reason: format!("squared GenRANSAC threshold in {unit} must be finite"),
        });
    }
    Ok(())
}

/// Options shared by every robust estimator of this crate.
///
/// Each estimator chooses its own number of samples from the confidence target and the inlier
/// ratio it finds; `max_draws` caps that number, and `None` leaves it to the estimator. The
/// default uses the fixed seed `42`; a pipeline that owns a root seed passes a seed derived
/// from it instead.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RobustOptions {
    /// Confidence target used by the adaptive draw-budget heuristic.
    ///
    /// Guided and multi-round sampling is not IID uniform sampling, so this is
    /// a scheduling heuristic rather than an exact probability guarantee.
    confidence: f64,
    /// Optional hard total draw cap across every population round.
    max_draws: Option<usize>,
    /// Deterministic random seed.
    seed: RunSeed,
    /// Whether to collect residual and score diagnostics.
    diagnostics: bool,
}

impl Default for RobustOptions {
    fn default() -> Self {
        Self {
            confidence: 0.9999,
            max_draws: None,
            seed: RunSeed::from_u64(42),
            diagnostics: false,
        }
    }
}

impl RobustOptions {
    /// Set the target confidence used by the adaptive draw-budget heuristic.
    ///
    /// Guided and multi-round sampling is not IID uniform sampling, so this is
    /// not an exact probability guarantee.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Cap the total number of minimal samples across every round.
    #[must_use]
    pub fn with_max_draws(mut self, max_draws: usize) -> Self {
        self.max_draws = Some(max_draws);
        self
    }

    /// Set the deterministic random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: impl Into<RunSeed>) -> Self {
        self.seed = seed.into();
        self
    }

    /// Enable residual and score diagnostics.
    #[must_use]
    pub fn with_diagnostics(mut self, enabled: bool) -> Self {
        self.diagnostics = enabled;
        self
    }

    /// Return the requested confidence.
    #[must_use]
    pub fn confidence(self) -> f64 {
        self.confidence
    }

    /// Return the optional hard draw cap.
    #[must_use]
    pub fn max_draws(self) -> Option<usize> {
        self.max_draws
    }

    /// Return the deterministic random seed.
    #[must_use]
    pub fn seed(self) -> RunSeed {
        self.seed
    }

    /// Return whether detailed diagnostics are enabled.
    #[must_use]
    pub fn diagnostics(self) -> bool {
        self.diagnostics
    }
}

/// Configuration for GenRANSAC's multi-round hypothesis search.
///
/// `max_hypotheses` is a per-round minimal-draw budget. `rounds` is the hard
/// round cap; confidence and patience may stop the search earlier.
#[derive(Debug, Clone, PartialEq)]
pub struct GenRansacConfig {
    pub(crate) max_draws: usize,
    pub(crate) max_hypotheses: usize,
    pub(crate) subsample_size: usize,
    pub(crate) prescreen_survivors: usize,
    pub(crate) rounds: usize,
    pub(crate) lo_candidates: usize,
    pub(crate) lo_iterations: usize,
    pub(crate) pool_init: usize,
    pub(crate) pool_growth_denom: usize,
    pub(crate) clamp_guided_pool: bool,
    pub(crate) patience: usize,
    pub(crate) patience_min_inlier_ratio: f64,
    pub(crate) confidence: f64,
    pub(crate) seed: RunSeed,
    pub(crate) compute_diagnostics: bool,
}

impl Default for GenRansacConfig {
    fn default() -> Self {
        Self {
            max_draws: 128 * 39,
            max_hypotheses: 128,
            subsample_size: 200,
            prescreen_survivors: 128,
            rounds: 39,
            lo_candidates: 3,
            lo_iterations: 3,
            pool_init: 20,
            pool_growth_denom: 20,
            clamp_guided_pool: false,
            patience: 0,
            patience_min_inlier_ratio: 0.20,
            confidence: 0.9999,
            seed: RunSeed::from_u64(42),
            compute_diagnostics: false,
        }
    }
}

impl GenRansacConfig {
    /// Start a generic search-scheduling configuration.
    pub fn builder() -> GenRansacBuilder {
        GenRansacBuilder::new(Self::default())
    }

    /// Tuned search shape from the archived Fundamental-matrix experiment.
    ///
    /// This selects only estimator scheduling. Pixel normalization and the
    /// non-minimal Fundamental refiner remain the caller's responsibility.
    pub fn fundamental() -> GenRansacBuilder {
        let config = Self {
            rounds: 20,
            prescreen_survivors: 64,
            lo_candidates: 1,
            lo_iterations: 1,
            pool_init: 10,
            pool_growth_denom: 200,
            clamp_guided_pool: true,
            patience: 5,
            ..Self::default()
        };
        GenRansacBuilder::new(config)
    }

    /// Search shape tuned for essential-matrix estimation.
    ///
    /// Residual units are not part of this scheduling preset;
    /// camera-aware facades accept their own typed thresholds.
    pub fn essential() -> GenRansacBuilder {
        let config = Self {
            rounds: 10,
            pool_init: 6,
            pool_growth_denom: 200,
            ..Self::default()
        };
        GenRansacBuilder::new(config)
    }

    /// Maximum scheduled minimal draws over all rounds.
    pub fn max_scheduled_draws(&self) -> usize {
        self.max_draws
            .min(self.max_hypotheses.saturating_mul(self.rounds))
    }

    pub(crate) fn from_options(builder: GenRansacBuilder, options: &RobustOptions) -> Self {
        let mut builder = builder
            .confidence(options.confidence())
            .seed(options.seed())
            .compute_diagnostics(options.diagnostics());
        if let Some(max_draws) = options.max_draws() {
            builder = builder.max_draws(max_draws);
        }
        builder.build()
    }

    pub(crate) fn validate(&self) -> Result<(), RobustError> {
        if !self.confidence.is_finite() || self.confidence <= 0.0 || self.confidence > 1.0 {
            return Err(RobustError::InvalidConfig {
                reason: "GenRANSAC confidence must be in (0, 1]".to_string(),
            });
        }
        if self.max_draws == 0 || self.max_hypotheses == 0 || self.rounds == 0 {
            return Err(RobustError::InvalidConfig {
                reason: "GenRANSAC draw and round budgets must be non-zero".to_string(),
            });
        }
        if self.prescreen_survivors == 0 || self.lo_candidates == 0 {
            return Err(RobustError::InvalidConfig {
                reason: "GenRANSAC survivor and LO candidate counts must be non-zero".to_string(),
            });
        }
        if self.lo_iterations == 0 || self.pool_growth_denom == 0 {
            return Err(RobustError::InvalidConfig {
                reason: "GenRANSAC LO iterations and pool growth denominator must be non-zero"
                    .to_string(),
            });
        }
        if !self.patience_min_inlier_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.patience_min_inlier_ratio)
        {
            return Err(RobustError::InvalidConfig {
                reason: "GenRANSAC patience inlier ratio must be in [0, 1]".to_string(),
            });
        }
        Ok(())
    }
}

/// Builder for GenRANSAC search scheduling.
#[derive(Debug, Clone)]
pub struct GenRansacBuilder {
    config: GenRansacConfig,
    explicit_max_draws: bool,
}

impl GenRansacBuilder {
    fn new(config: GenRansacConfig) -> Self {
        let mut config = config;
        config.max_draws = config.max_hypotheses.saturating_mul(config.rounds);
        Self {
            config,
            explicit_max_draws: false,
        }
    }

    /// Finish the scheduling configuration.
    #[must_use]
    pub fn build(self) -> GenRansacConfig {
        self.config
    }

    /// Set the number of minimal hypotheses generated per round.
    #[cfg(test)]
    #[must_use]
    pub fn max_hypotheses(mut self, value: usize) -> Self {
        self.config.max_hypotheses = value;
        self.update_implicit_draw_budget();
        self
    }

    /// Set a hard total minimal-sample budget across all rounds.
    #[must_use]
    pub fn max_draws(mut self, value: usize) -> Self {
        self.config.max_draws = value;
        self.explicit_max_draws = true;
        self
    }

    /// Set the number of prescreened hypotheses retained for full scoring.
    #[cfg(test)]
    #[must_use]
    pub fn prescreen_survivors(mut self, value: usize) -> Self {
        self.config.prescreen_survivors = value;
        self
    }

    /// Set the maximum number of population-guided rounds.
    #[cfg(test)]
    #[must_use]
    pub fn rounds(mut self, value: usize) -> Self {
        self.config.rounds = value;
        self.update_implicit_draw_budget();
        self
    }

    /// Set confidence. `1.0` disables the heuristic confidence budget.
    #[must_use]
    pub fn confidence(mut self, value: f64) -> Self {
        self.config.confidence = value;
        self
    }

    /// Set the deterministic random seed.
    #[must_use]
    pub fn seed(mut self, value: impl Into<RunSeed>) -> Self {
        self.config.seed = value.into();
        self
    }

    /// Enable or disable detailed diagnostics.
    #[must_use]
    pub fn compute_diagnostics(mut self, enabled: bool) -> Self {
        self.config.compute_diagnostics = enabled;
        self
    }

    #[cfg(test)]
    fn update_implicit_draw_budget(&mut self) {
        if !self.explicit_max_draws {
            self.config.max_draws = self
                .config
                .max_hypotheses
                .saturating_mul(self.config.rounds);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fundamental_preset_has_expected_budget() {
        let config = GenRansacConfig::fundamental().build();
        assert_eq!(config.max_scheduled_draws(), 2_560);
        assert_eq!(config.prescreen_survivors, 64);
    }

    #[test]
    fn public_options_override_only_stable_controls() {
        let options = RobustOptions::default()
            .with_confidence(0.99)
            .with_max_draws(321)
            .with_seed(7)
            .with_diagnostics(true);
        let config = GenRansacConfig::from_options(GenRansacConfig::essential(), &options);
        assert_eq!(config.max_scheduled_draws(), 321);
        assert_eq!(config.confidence, 0.99);
        assert_eq!(config.seed, 7.into());
        assert!(config.compute_diagnostics);
        assert_eq!(config.rounds, 10);
    }

    #[test]
    fn scheduled_draws_do_not_exceed_the_reachable_round_budget() {
        let config = GenRansacConfig::builder()
            .rounds(2)
            .max_hypotheses(10)
            .max_draws(1_000)
            .build();
        assert_eq!(config.max_scheduled_draws(), 20);
    }

    #[test]
    fn hostile_floats_are_rejected() {
        for threshold in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ResidualThreshold::new(threshold).is_err());
            assert!(NormalizedImageThreshold::try_new(threshold).is_err());
            assert!(AngularThreshold::try_new(threshold).is_err());
        }
    }
}
