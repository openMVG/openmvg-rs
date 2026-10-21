// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public refinement configuration, outcomes, and input validation.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use crate::numerics::robust;

use super::covariance::InformationDiagnostics;

/// Invalid user input or a numerical failure before refinement can finish.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementError {
    /// A correspondence model declared no scalar residual components.
    EmptyResidualBlock,
    /// Correspondence arrays have different lengths.
    PointCountMismatch {
        /// First-array length.
        points1: usize,
        /// Second-array length.
        points2: usize,
    },
    /// A homogeneous point coordinate is NaN or infinite.
    NonFinitePoint {
        /// One-based point-set identifier.
        set: usize,
        /// Observation index.
        index: usize,
        /// Coordinate index.
        coordinate: usize,
    },
    /// A bearing has zero or numerically unusable Euclidean norm.
    ZeroNormBearing {
        /// One-based bearing-set identifier.
        set: usize,
        /// Observation index.
        index: usize,
    },
    /// A homogeneous point cannot be converted to finite image coordinates.
    PointAtInfinity {
        /// One-based point-set identifier.
        set: usize,
        /// Observation index.
        index: usize,
    },
    /// A division-model distortion denominator is numerically zero.
    UndefinedDistortionProjection {
        /// One-based point-set identifier.
        set: usize,
        /// Observation index.
        index: usize,
    },
    /// The initial model contains NaN or infinity.
    NonFiniteInitialModel,
    /// The initial model does not define the requested geometric manifold.
    DegenerateInitialModel,
    /// A Sampson denominator is numerically zero.
    UndefinedSampsonResidual {
        /// Correspondence index.
        index: usize,
    },
    /// Per-observation noise does not match the observation count.
    NoiseCountMismatch {
        /// Required number of standard deviations.
        expected: usize,
        /// Supplied number of standard deviations.
        actual: usize,
    },
    /// A per-observation noise scale is missing at the requested index.
    NoiseIndexOutOfBounds {
        /// Requested zero-based observation index.
        index: usize,
        /// Number of supplied standard deviations.
        count: usize,
    },
    /// An outlier mask does not match the observation count.
    MaskCountMismatch {
        /// Required mask length.
        expected: usize,
        /// Supplied mask length.
        actual: usize,
    },
    /// A noise standard deviation is non-positive, non-finite, or has no finite reciprocal.
    InvalidNoiseStandardDeviation {
        /// Invalid scale index, or zero for a shared scale.
        index: usize,
    },
    /// A robust-loss parameter is invalid.
    InvalidLoss(&'static str),
    /// At least one stopping tolerance is non-positive or non-finite.
    InvalidConvergenceCriteria,
    /// An error threshold is negative or non-finite.
    InvalidThreshold,
    /// Too few observations were supplied for the model dimension.
    InsufficientObservations {
        /// Minimum required observation count.
        required: usize,
        /// Supplied observation count.
        actual: usize,
    },
}

impl fmt::Display for RefinementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyResidualBlock => {
                formatter.write_str("a correspondence residual block must not be empty")
            }
            Self::PointCountMismatch { points1, points2 } => write!(
                formatter,
                "point arrays have different lengths: {points1} and {points2}"
            ),
            Self::NonFinitePoint {
                set,
                index,
                coordinate,
            } => write!(
                formatter,
                "point set {set}, observation {index}, coordinate {coordinate} is not finite"
            ),
            Self::ZeroNormBearing { set, index } => write!(
                formatter,
                "bearing set {set}, observation {index} has zero or unusable norm"
            ),
            Self::PointAtInfinity { set, index } => write!(
                formatter,
                "point set {set}, observation {index} cannot be normalized to finite image coordinates"
            ),
            Self::UndefinedDistortionProjection { set, index } => write!(
                formatter,
                "point set {set}, observation {index} lies on the division-model distortion pole"
            ),
            Self::NonFiniteInitialModel => {
                formatter.write_str("initial model matrix must contain only finite values")
            }
            Self::DegenerateInitialModel => {
                formatter.write_str("initial model is degenerate for the requested manifold")
            }
            Self::UndefinedSampsonResidual { index } => write!(
                formatter,
                "Sampson residual {index} has an undefined or numerically singular denominator"
            ),
            Self::NoiseCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} noise standard deviations, received {actual}"
            ),
            Self::NoiseIndexOutOfBounds { index, count } => write!(
                formatter,
                "noise index {index} is outside {count} supplied standard deviations"
            ),
            Self::MaskCountMismatch { expected, actual } => write!(
                formatter,
                "expected an outlier mask of length {expected}, received {actual}"
            ),
            Self::InvalidNoiseStandardDeviation { index } => write!(
                formatter,
                "noise standard deviation at index {index} must be positive and finite with a finite reciprocal"
            ),
            Self::InvalidLoss(message) => formatter.write_str(message),
            Self::InvalidConvergenceCriteria => formatter
                .write_str("convergence tolerances must contain only positive finite values"),
            Self::InvalidThreshold => {
                formatter.write_str("the error threshold must be non-negative and finite")
            }
            Self::InsufficientObservations { required, actual } => write!(
                formatter,
                "at least {required} correspondences are required, received {actual}"
            ),
        }
    }
}

impl Error for RefinementError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

/// Validate paired finite homogeneous observations for a refinement kernel.
#[doc(hidden)]
pub fn validate_points(points1: &[[f64; 3]], points2: &[[f64; 3]]) -> Result<(), RefinementError> {
    if points1.len() != points2.len() {
        return Err(RefinementError::PointCountMismatch {
            points1: points1.len(),
            points2: points2.len(),
        });
    }
    for (set, points) in [points1, points2].into_iter().enumerate() {
        for (index, point) in points.iter().enumerate() {
            if let Some(coordinate) = point.iter().position(|value| !value.is_finite()) {
                return Err(RefinementError::NonFinitePoint {
                    set: set + 1,
                    index,
                    coordinate,
                });
            }
            let scale = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
            if point[2].abs() <= 64.0 * f64::EPSILON * scale {
                return Err(RefinementError::PointAtInfinity {
                    set: set + 1,
                    index,
                });
            }
        }
    }
    Ok(())
}

/// Validate that every coefficient of a model matrix is finite.
#[doc(hidden)]
pub fn validate_matrix(matrix: &[[f64; 3]; 3]) -> Result<(), RefinementError> {
    if matrix.iter().flatten().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(RefinementError::NonFiniteInitialModel)
    }
}

/// Validate a model's minimum correspondence count.
#[doc(hidden)]
pub fn validate_observation_count(actual: usize, required: usize) -> Result<(), RefinementError> {
    if actual < required {
        Err(RefinementError::InsufficientObservations { required, actual })
    } else {
        Ok(())
    }
}

/// Known standard deviations for correspondence residuals.
///
/// Whitening divides residual `i` and its Jacobian by `sigma_i` before the
/// robust loss is evaluated. Robust-loss scales are therefore dimensionless
/// (for example, a Huber threshold in standard deviations).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseModel<'a> {
    /// Unit standard deviation for every observation.
    Unit,
    /// One standard deviation shared by every observation.
    Isotropic {
        /// Positive finite residual standard deviation.
        standard_deviation: f64,
    },
    /// One scalar standard deviation per correspondence.
    PerObservation {
        /// Positive finite residual standard deviations.
        standard_deviations: &'a [f64],
    },
}

impl NoiseModel<'_> {
    /// Construct a shared residual-noise model.
    pub const fn isotropic(standard_deviation: f64) -> Self {
        Self::Isotropic { standard_deviation }
    }

    /// Construct a per-correspondence residual-noise model.
    pub const fn per_observation(standard_deviations: &[f64]) -> NoiseModel<'_> {
        NoiseModel::PerObservation {
            standard_deviations,
        }
    }

    /// Validate the observation count and positive finite scales with representable reciprocals.
    pub fn try_validate(self, observation_count: usize) -> Result<(), RefinementError> {
        if let Self::PerObservation {
            standard_deviations,
        } = self
            && standard_deviations.len() != observation_count
        {
            return Err(RefinementError::NoiseCountMismatch {
                expected: observation_count,
                actual: standard_deviations.len(),
            });
        }
        let invalid = match self {
            Self::Unit => None,
            Self::Isotropic { standard_deviation } => {
                (!valid_noise_standard_deviation(standard_deviation)).then_some(0)
            }
            Self::PerObservation {
                standard_deviations,
            } => standard_deviations
                .iter()
                .position(|value| !valid_noise_standard_deviation(*value)),
        };
        if let Some(index) = invalid {
            return Err(RefinementError::InvalidNoiseStandardDeviation { index });
        }
        Ok(())
    }

    /// Return a checked positive finite inverse standard deviation.
    ///
    /// Per-observation models reject a missing index. Shared models apply at
    /// every index; [`Self::try_validate`] checks their complete solve domain.
    #[inline]
    pub fn try_inverse_standard_deviation(
        self,
        observation: usize,
    ) -> Result<f64, RefinementError> {
        let (standard_deviation, index) = match self {
            Self::Unit => return Ok(1.0),
            Self::Isotropic { standard_deviation } => (standard_deviation, 0),
            Self::PerObservation {
                standard_deviations,
            } => (
                *standard_deviations.get(observation).ok_or(
                    RefinementError::NoiseIndexOutOfBounds {
                        index: observation,
                        count: standard_deviations.len(),
                    },
                )?,
                observation,
            ),
        };
        let inverse = standard_deviation.recip();
        if !(standard_deviation.is_finite()
            && standard_deviation > 0.0
            && inverse.is_finite()
            && inverse > 0.0)
        {
            return Err(RefinementError::InvalidNoiseStandardDeviation { index });
        }
        Ok(inverse)
    }

    /// Numeric access after complete entry validation; invalid access is non-finite, never a panic.
    #[inline(always)]
    #[cfg(any(test, feature = "faer", feature = "unstable-internals"))]
    pub(crate) fn inverse_standard_deviation(self, observation: usize) -> f64 {
        self.try_inverse_standard_deviation(observation)
            .unwrap_or(f64::NAN)
    }
}

fn valid_noise_standard_deviation(value: f64) -> bool {
    if !value.is_finite() || value <= 0.0 {
        return false;
    }
    let inverse = value.recip();
    inverse.is_finite() && inverse > 0.0
}

/// Loss used to form the iteratively reweighted least-squares system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// Ordinary least squares: `rho(r) = r^2 / 2`.
    L2,
    /// Convex quadratic-to-linear loss with transition at `delta`.
    Huber {
        /// Positive finite transition magnitude.
        delta: f64,
    },
    /// Cauchy/Lorentzian loss with residual scale `scale`.
    Cauchy {
        /// Positive finite residual scale.
        scale: f64,
    },
}

impl Loss {
    #[inline(always)]
    /// Return the IRLS square-root weight for a residual magnitude.
    pub fn sqrt_weight(self, residual: f64) -> f64 {
        match self {
            Self::L2 => 1.0,
            Self::Huber { delta } => robust::huber_irls_sqrt_weight(residual, delta),
            Self::Cauchy { scale } => robust::cauchy_irls_sqrt_weight(residual, scale),
        }
    }

    #[inline(always)]
    /// Evaluate the robust cost for a residual magnitude.
    pub fn cost(self, residual: f64) -> f64 {
        match self {
            Self::L2 => 0.5 * residual * residual,
            Self::Huber { delta } => robust::huber_cost(residual, delta),
            Self::Cauchy { scale } => robust::cauchy_cost(residual, scale),
        }
    }

    /// IRLS weight normalized to one at zero residual. The normalization
    /// removes arbitrary global loss scaling from covariance units.
    #[inline(always)]
    pub fn relative_sqrt_weight(self, residual: f64) -> f64 {
        self.sqrt_weight(residual) / self.sqrt_weight(0.0)
    }

    /// Validate the robust loss and convergence thresholds without running a solver.
    pub fn try_validate(self) -> Result<(), RefinementError> {
        match self {
            Self::L2 => Ok(()),
            Self::Huber { delta } => {
                if delta.is_finite() && delta > 0.0 {
                    Ok(())
                } else {
                    Err(RefinementError::InvalidLoss(
                        "Huber delta must be positive and finite",
                    ))
                }
            }
            Self::Cauchy { scale } => {
                if scale.is_finite() && scale > 0.0 {
                    Ok(())
                } else {
                    Err(RefinementError::InvalidLoss(
                        "Cauchy scale must be positive and finite",
                    ))
                }
            }
        }
    }
}

/// Configuration shared by geometric-model and bundle-adjustment refiners.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Maximum number of nonlinear iterations.
    pub max_iterations: usize,
    /// Loss applied to whitened residuals.
    pub loss: Loss,
    /// Scale-aware stopping thresholds.
    pub convergence: ConvergenceCriteria,
}

/// Scale-aware stopping tolerances for the local manifold solve.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvergenceCriteria {
    /// Stop when the model's stationarity norm is at most
    /// `gradient * (1 + cost)`. Every solver divides local gradient
    /// coordinates by square-root diagonal curvature scales so equivalent
    /// tangent-coordinate scalings do not change this test.
    pub gradient: f64,
    /// Stop when the Euclidean norm of a local tangent step is this small.
    pub step: f64,
    /// Stop when accepted relative cost reduction is this small.
    pub relative_cost: f64,
}

impl ConvergenceCriteria {
    /// Default stopping thresholds used by [`Options`].
    pub const DEFAULT: Self = Self {
        gradient: 1e-10,
        step: 1e-10,
        relative_cost: 1e-12,
    };

    /// Construct explicit gradient, step, and relative-cost tolerances.
    pub const fn new(gradient: f64, step: f64, relative_cost: f64) -> Self {
        Self {
            gradient,
            step,
            relative_cost,
        }
    }

    fn is_valid(self) -> bool {
        [self.gradient, self.step, self.relative_cost]
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0)
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Which [`ConvergenceCriteria`] tolerance a converged refinement satisfied.
///
/// Distinct from the stable [`crate::ConvergenceReason`] reported by
/// [`crate::BundleAdjuster`] implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvergenceCriterion {
    /// The scaled stationarity norm reached its tolerance.
    Gradient,
    /// The local tangent step reached its tolerance.
    Step,
    /// Relative accepted cost reduction reached its tolerance.
    Cost,
}

/// Explicit quality/status of the returned estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementStatus {
    /// A declared stopping tolerance was satisfied.
    Converged(ConvergenceCriterion),
    /// The iteration budget was exhausted.
    MaxIterations,
    /// LM could not find an acceptable finite step.
    Stalled,
    /// Final local information is rank deficient.
    Degenerate,
}

#[inline]
#[doc(hidden)]
pub fn is_numerically_usable(
    status: RefinementStatus,
    accepted_step_count: usize,
    initial_objective: f64,
    final_objective: f64,
) -> bool {
    let usable_status = match status {
        RefinementStatus::Converged(_) => true,
        RefinementStatus::MaxIterations | RefinementStatus::Stalled => accepted_step_count > 0,
        RefinementStatus::Degenerate => false,
    };
    usable_status
        && initial_objective.is_finite()
        && final_objective.is_finite()
        && final_objective <= initial_objective
}

/// Estimate plus enough evidence to judge whether refinement succeeded.
#[must_use = "inspect status and diagnostics before accepting a refined model"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefinementReport<State> {
    /// Returned model state.
    pub state: State,
    /// Loss used for refinement.
    pub loss: Loss,
    /// Explicit solution-quality status.
    pub status: RefinementStatus,
    /// Completed LM iterations.
    pub iteration_count: usize,
    /// Accepted trial steps.
    pub accepted_step_count: usize,
    /// Rejected trial steps.
    pub rejected_step_count: usize,
    /// Robust objective at the initial state.
    pub initial_objective: f64,
    /// Robust objective at the returned state.
    pub final_objective: f64,
    /// Final scale-aware stationarity norm.
    pub gradient_inf_norm: f64,
    /// Final local step norm.
    pub step_norm: f64,
    /// Final dimensionless LM damping value.
    pub final_damping: f64,
    /// Gain ratio from the last evaluated trial step, if one was available.
    pub last_gain_ratio: Option<f64>,
    /// Rank and condition evidence for the final undamped system.
    pub diagnostics: InformationDiagnostics,
}

impl<State> RefinementReport<State> {
    /// Construct a report from a completed checked numerical solve.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        state: State,
        loss: Loss,
        status: RefinementStatus,
        iteration_count: usize,
        accepted_step_count: usize,
        rejected_step_count: usize,
        initial_objective: f64,
        final_objective: f64,
        gradient_inf_norm: f64,
        step_norm: f64,
        final_damping: f64,
        last_gain_ratio: Option<f64>,
        diagnostics: InformationDiagnostics,
    ) -> Self {
        Self {
            state,
            loss,
            status,
            iteration_count,
            accepted_step_count,
            rejected_step_count,
            initial_objective,
            final_objective,
            gradient_inf_norm,
            step_norm,
            final_damping,
            last_gain_ratio,
            diagnostics,
        }
    }

    /// Whether solver status and objective evidence permit using this result.
    ///
    /// This checks a finite non-increasing objective, rejects degeneracy, and requires
    /// accepted progress when the solve stalls or exhausts its budget. It does
    /// not impose application-specific physical or pipeline commit policy;
    /// those checks remain the caller's responsibility. Built-in OpenMVG BA
    /// retractions preserve their declared model manifold.
    #[inline]
    pub fn is_numerically_usable(&self) -> bool {
        is_numerically_usable(
            self.status,
            self.accepted_step_count,
            self.initial_objective,
            self.final_objective,
        )
    }
}

impl Options {
    /// Construct ordinary least-squares options.
    pub const fn l2(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            loss: Loss::L2,
            convergence: ConvergenceCriteria::DEFAULT,
        }
    }

    /// Construct Huber-loss options.
    pub const fn huber(delta: f64, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            loss: Loss::Huber { delta },
            convergence: ConvergenceCriteria::DEFAULT,
        }
    }

    /// Construct Cauchy-loss options.
    pub const fn cauchy(scale: f64, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            loss: Loss::Cauchy { scale },
            convergence: ConvergenceCriteria::DEFAULT,
        }
    }

    /// Replace the default convergence thresholds.
    pub const fn with_convergence(mut self, convergence: ConvergenceCriteria) -> Self {
        self.convergence = convergence;
        self
    }

    /// Validate the loss and convergence thresholds without running a solver.
    pub fn try_validate(self) -> Result<(), RefinementError> {
        self.loss.try_validate()?;
        if self.convergence.is_valid() {
            Ok(())
        } else {
            Err(RefinementError::InvalidConvergenceCriteria)
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::l2(30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_usability_policy_is_fail_closed() {
        let converged = RefinementStatus::Converged(ConvergenceCriterion::Gradient);
        assert!(is_numerically_usable(converged, 0, 2.0, 2.0));
        assert!(is_numerically_usable(
            RefinementStatus::MaxIterations,
            1,
            2.0,
            1.0
        ));
        assert!(is_numerically_usable(
            RefinementStatus::Stalled,
            1,
            2.0,
            1.0
        ));

        for (status, accepted_steps, initial_cost, final_cost) in [
            (RefinementStatus::MaxIterations, 0, 2.0, 1.0),
            (RefinementStatus::Stalled, 0, 2.0, 1.0),
            (RefinementStatus::Degenerate, 4, 2.0, 1.0),
            (converged, 1, 1.0, 2.0),
            (converged, 1, f64::NAN, 1.0),
            (converged, 1, 1.0, f64::INFINITY),
        ] {
            assert!(!is_numerically_usable(
                status,
                accepted_steps,
                initial_cost,
                final_cost
            ));
        }
    }
}

#[cfg(test)]
mod checked_noise_tests {
    use super::*;

    #[test]
    fn checked_noise_access_validates_shape_indices_and_reciprocal_range() {
        let noise = NoiseModel::per_observation(&[2.0, 4.0]);
        assert_eq!(noise.try_validate(2), Ok(()));
        assert_eq!(noise.try_inverse_standard_deviation(0), Ok(0.5));
        assert_eq!(noise.try_inverse_standard_deviation(1), Ok(0.25));
        assert_eq!(
            noise.try_validate(1),
            Err(RefinementError::NoiseCountMismatch {
                expected: 1,
                actual: 2
            })
        );
        assert_eq!(
            noise.try_inverse_standard_deviation(usize::MAX),
            Err(RefinementError::NoiseIndexOutOfBounds {
                index: usize::MAX,
                count: 2
            })
        );
        assert!(noise.inverse_standard_deviation(usize::MAX).is_nan());
        assert_eq!(
            NoiseModel::per_observation(&[]).try_inverse_standard_deviation(0),
            Err(RefinementError::NoiseIndexOutOfBounds { index: 0, count: 0 })
        );
        assert_eq!(
            NoiseModel::Unit.try_inverse_standard_deviation(usize::MAX),
            Ok(1.0)
        );
        assert_eq!(
            NoiseModel::isotropic(2.0).try_inverse_standard_deviation(usize::MAX),
            Ok(0.5)
        );
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-310] {
            let invalid = NoiseModel::isotropic(sigma);
            assert_eq!(
                invalid.try_validate(1),
                Err(RefinementError::InvalidNoiseStandardDeviation { index: 0 })
            );
            assert_eq!(
                invalid.try_inverse_standard_deviation(0),
                Err(RefinementError::InvalidNoiseStandardDeviation { index: 0 })
            );
            assert!(invalid.inverse_standard_deviation(0).is_nan());
        }
        for sigma in [f64::MIN_POSITIVE, 0.5, f64::MAX] {
            let valid = NoiseModel::isotropic(sigma);
            assert_eq!(valid.try_validate(1), Ok(()));
            let inverse = valid.try_inverse_standard_deviation(0).unwrap();
            assert!(inverse.is_finite() && inverse > 0.0);
        }
        let mixed = NoiseModel::per_observation(&[1.0, f64::NAN]);
        assert_eq!(mixed.try_inverse_standard_deviation(0), Ok(1.0));
        assert_eq!(
            mixed.try_validate(2),
            Err(RefinementError::InvalidNoiseStandardDeviation { index: 1 })
        );
    }
}
