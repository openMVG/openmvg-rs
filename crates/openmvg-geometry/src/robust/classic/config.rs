// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated configuration for the compact classic estimator.

use crate::robust::error::RobustError;
use openmvg_core::{RunSeed, Threshold};

/// Default confidence for uniform random-sample estimation.
pub const DEFAULT_CONFIDENCE: f64 = 0.99;

/// Default maximum number of random samples.
pub const DEFAULT_MAX_ITERATIONS: usize = 1_000;

#[cfg(test)]
const DEFAULT_COMBINATION_LIMIT: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RandomMethod {
    Ransac,
    Msac,
}

#[derive(Debug, Clone)]
pub(crate) enum ClassicMethod<Q> {
    Ransac(RandomParams<Q>),
    Msac(RandomParams<Q>),
    /// Exhaustive oracle for tests; the `max_consensus` builder constructs it.
    #[cfg(test)]
    MaxConsensus(MaxConsensusParams<Q>),
}

#[derive(Debug, Clone)]
pub(crate) struct RandomParams<Q> {
    pub(crate) threshold: Threshold<Q>,
    pub(crate) confidence: f64,
    pub(crate) max_iterations: usize,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct MaxConsensusParams<Q> {
    pub(crate) threshold: Threshold<Q>,
    pub(crate) combination_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SamplingStrategy {
    Uniform,
    Prosac,
}

#[derive(Debug, Clone, Default)]
struct CommonOptions {
    compute_diagnostics: bool,
    min_inliers: Option<usize>,
}

/// Residual unit of the 2D image-plane frame the classic estimators score in.
///
/// The homography and fundamental-matrix residuals are distances in the frame of
/// the input points, so the threshold unit must match that frame: [`Pixels`] for
/// raw image coordinates or [`NormalizedImage`] for normalized ones. This trait
/// is sealed.
///
/// [`Pixels`]: openmvg_core::Pixels
/// [`NormalizedImage`]: openmvg_core::NormalizedImage
pub trait ImagePlaneUnit: sealed::Sealed + Copy {}

impl ImagePlaneUnit for openmvg_core::Pixels {}
impl ImagePlaneUnit for openmvg_core::NormalizedImage {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for openmvg_core::Pixels {}
    impl Sealed for openmvg_core::NormalizedImage {}
}

/// Configuration for the compact classic robust-estimation loop.
///
/// This API is intended for teaching and controlled family-level baselines.
/// Library geometry estimation should prefer the typed [`crate::robust`]
/// facades.
#[derive(Debug, Clone)]
pub struct ClassicConfig<Q> {
    method: ClassicMethod<Q>,
    sampling: SamplingStrategy,
    compute_diagnostics: bool,
    min_inliers: Option<usize>,
    seed: Option<RunSeed>,
}

impl<Q: Copy> ClassicConfig<Q> {
    /// Configure binary-consensus RANSAC with an unsquared residual threshold.
    #[must_use]
    pub fn ransac(threshold: Threshold<Q>) -> RansacBuilder<Q> {
        RansacBuilder::new(RandomMethod::Ransac, threshold)
    }

    /// Configure textbook MSAC with an unsquared residual threshold.
    #[must_use]
    pub fn msac(threshold: Threshold<Q>) -> RansacBuilder<Q> {
        RansacBuilder::new(RandomMethod::Msac, threshold)
    }

    /// Configure the exhaustive maximum-consensus oracle (test-only).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn max_consensus(threshold: Threshold<Q>) -> MaxConsensusBuilder<Q> {
        MaxConsensusBuilder::new(threshold)
    }

    /// Return the unsquared residual threshold.
    #[must_use]
    pub fn threshold(&self) -> Threshold<Q> {
        match &self.method {
            ClassicMethod::Ransac(params) | ClassicMethod::Msac(params) => params.threshold,
            #[cfg(test)]
            ClassicMethod::MaxConsensus(params) => params.threshold,
        }
    }

    /// Return the random-draw cap.
    #[must_use]
    pub fn max_iterations(&self) -> usize {
        match &self.method {
            ClassicMethod::Ransac(params) | ClassicMethod::Msac(params) => params.max_iterations,
            #[cfg(test)]
            ClassicMethod::MaxConsensus(params) => params.combination_limit,
        }
    }

    /// Return the configured minimum consensus size, if any.
    #[must_use]
    pub fn min_inliers(&self) -> Option<usize> {
        self.min_inliers
    }

    /// Return whether detailed diagnostics are enabled.
    #[must_use]
    pub fn diagnostics(&self) -> bool {
        self.compute_diagnostics
    }

    /// Validate all method-independent and method-specific parameters.
    pub fn validate(&self) -> Result<(), RobustError> {
        if self.min_inliers == Some(0) {
            return invalid_config("min_inliers must be greater than zero");
        }

        match &self.method {
            ClassicMethod::Ransac(params) | ClassicMethod::Msac(params) => {
                validate_confidence(params.confidence)?;
                validate_positive_limit("max_iterations", params.max_iterations)
            }
            #[cfg(test)]
            ClassicMethod::MaxConsensus(params) => {
                validate_positive_limit("combination_limit", params.combination_limit)
            }
        }
    }

    pub(crate) fn method(&self) -> &ClassicMethod<Q> {
        &self.method
    }

    pub(crate) fn sampling(&self) -> SamplingStrategy {
        self.sampling
    }

    pub(crate) fn seed(&self) -> Option<RunSeed> {
        self.seed
    }
}

/// Shared builder for classic RANSAC and MSAC.
#[derive(Debug, Clone)]
pub struct RansacBuilder<Q> {
    method: RandomMethod,
    params: RandomParams<Q>,
    options: CommonOptions,
    seed: Option<RunSeed>,
    sampling: SamplingStrategy,
}

impl<Q> RansacBuilder<Q> {
    fn new(method: RandomMethod, threshold: Threshold<Q>) -> Self {
        Self {
            method,
            params: RandomParams {
                threshold,
                confidence: DEFAULT_CONFIDENCE,
                max_iterations: DEFAULT_MAX_ITERATIONS,
            },
            options: CommonOptions::default(),
            seed: None,
            sampling: SamplingStrategy::Uniform,
        }
    }

    /// Set the confidence level in `(0, 1)` for uniform RANSAC stopping.
    ///
    /// PROSAC uses the full configured draw budget instead.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.params.confidence = confidence;
        self
    }

    /// Set the maximum number of random samples.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.params.max_iterations = max_iterations;
        self
    }

    /// Enable or disable detailed diagnostics.
    #[must_use]
    pub fn with_diagnostics(mut self, enabled: bool) -> Self {
        self.options.compute_diagnostics = enabled;
        self
    }

    /// Require at least this many final inliers.
    #[must_use]
    pub fn with_min_inliers(mut self, minimum: usize) -> Self {
        self.options.min_inliers = Some(minimum);
        self
    }

    /// Set a reproducible random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: impl Into<RunSeed>) -> Self {
        self.seed = Some(seed.into());
        self
    }

    /// Use PROSAC with data sorted best-first by match quality.
    ///
    /// PROSAC runs the full `max_iterations` schedule because the standard
    /// uniform-RANSAC confidence bound is not valid for ranked-prefix draws.
    #[must_use]
    pub fn prosac(mut self) -> Self {
        self.sampling = SamplingStrategy::Prosac;
        self
    }

    /// Build the configuration. Numeric validation occurs before estimation.
    #[must_use]
    pub fn build(self) -> ClassicConfig<Q> {
        let method = match self.method {
            RandomMethod::Ransac => ClassicMethod::Ransac(self.params),
            RandomMethod::Msac => ClassicMethod::Msac(self.params),
        };
        ClassicConfig {
            method,
            sampling: self.sampling,
            compute_diagnostics: self.options.compute_diagnostics,
            min_inliers: self.options.min_inliers,
            seed: self.seed,
        }
    }
}

/// Builder for the test-only exhaustive maximum-consensus oracle.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct MaxConsensusBuilder<Q> {
    params: MaxConsensusParams<Q>,
    options: CommonOptions,
}

#[cfg(test)]
impl<Q> MaxConsensusBuilder<Q> {
    fn new(threshold: Threshold<Q>) -> Self {
        Self {
            params: MaxConsensusParams {
                threshold,
                combination_limit: DEFAULT_COMBINATION_LIMIT,
            },
            options: CommonOptions::default(),
        }
    }

    /// Refuse exhaustive estimation above this many combinations.
    ///
    /// The estimator never truncates enumeration: exceeding this limit is a
    /// typed configuration error.
    #[must_use]
    pub fn combination_limit(mut self, maximum: usize) -> Self {
        self.params.combination_limit = maximum;
        self
    }

    /// Build the configuration. Numeric validation occurs before estimation.
    #[must_use]
    pub fn build(self) -> ClassicConfig<Q> {
        ClassicConfig {
            method: ClassicMethod::MaxConsensus(self.params),
            sampling: SamplingStrategy::Uniform,
            compute_diagnostics: self.options.compute_diagnostics,
            min_inliers: self.options.min_inliers,
            seed: None,
        }
    }
}

fn validate_confidence(confidence: f64) -> Result<(), RobustError> {
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return invalid_config("confidence must be finite and in (0, 1)");
    }
    Ok(())
}

fn validate_positive_limit(name: &str, value: usize) -> Result<(), RobustError> {
    if value == 0 {
        return invalid_config(format!("{name} must be greater than zero"));
    }
    Ok(())
}

pub(crate) fn invalid_config<T>(reason: impl Into<String>) -> Result<T, RobustError> {
    Err(RobustError::InvalidConfig {
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmvg_core::ScalarThreshold;

    fn threshold(value: f64) -> ScalarThreshold {
        ScalarThreshold::try_new(value).unwrap()
    }

    #[test]
    fn builders_preserve_method_and_sampling() {
        let ransac = ClassicConfig::ransac(threshold(3.0))
            .with_confidence(0.999)
            .with_max_iterations(500)
            .prosac()
            .build();
        assert!(matches!(ransac.method(), ClassicMethod::Ransac(_)));
        assert_eq!(ransac.sampling(), SamplingStrategy::Prosac);
        assert_eq!(ransac.threshold().value(), 3.0);
        assert_eq!(ransac.max_iterations(), 500);

        let msac = ClassicConfig::msac(threshold(2.0)).build();
        assert!(matches!(msac.method(), ClassicMethod::Msac(_)));

        let exhaustive = ClassicConfig::max_consensus(threshold(1.5))
            .combination_limit(50_000)
            .build();
        assert!(matches!(
            exhaustive.method(),
            ClassicMethod::MaxConsensus(_)
        ));
    }

    #[test]
    fn invalid_numeric_values_are_rejected() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(ScalarThreshold::try_new(value).is_err());
        }
        for confidence in [0.0, 1.0, f64::NAN, f64::INFINITY] {
            assert!(
                ClassicConfig::ransac(threshold(1.0))
                    .with_confidence(confidence)
                    .build()
                    .validate()
                    .is_err()
            );
        }
    }

    #[test]
    fn zero_limits_and_consensus_are_rejected() {
        assert!(
            ClassicConfig::ransac(threshold(1.0))
                .with_max_iterations(0)
                .build()
                .validate()
                .is_err()
        );
        assert!(
            ClassicConfig::max_consensus(threshold(1.0))
                .combination_limit(0)
                .build()
                .validate()
                .is_err()
        );
        assert!(
            ClassicConfig::ransac(threshold(1.0))
                .with_min_inliers(0)
                .build()
                .validate()
                .is_err()
        );
    }
}
