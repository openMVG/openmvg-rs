// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal PnP results and typed failures.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use super::RefinementPose;
use openmvg_ba::numerics::refine::is_numerically_usable;
use openmvg_ba::numerics::refine::{
    InformationDiagnostics, Loss, RefinementError, RefinementStatus,
};

/// Invalid input or numerical-domain failure in initialized PnP refinement.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpError {
    /// The world-point and observation arrays have different lengths.
    MeasurementCountMismatch {
        /// Number of fixed world points.
        points: usize,
        /// Number of observations.
        observations: usize,
    },
    /// Fewer than three correspondences were supplied.
    InsufficientPoints {
        /// Minimum accepted point count.
        required: usize,
        /// Supplied point count.
        actual: usize,
    },
    /// A world-point coordinate is NaN or infinite.
    NonFinitePoint {
        /// Point index.
        point: usize,
        /// Coordinate within the point.
        coordinate: usize,
    },
    /// An observation coordinate is NaN or infinite.
    NonFiniteObservation {
        /// Observation index.
        observation: usize,
        /// Coordinate within the observation.
        coordinate: usize,
    },
    /// A bearing observation has zero or numerically unusable norm.
    InvalidBearing {
        /// Observation index.
        observation: usize,
    },
    /// The initial rotation or translation is not a finite rigid pose.
    InvalidPose,
    /// A point lies outside the selected observation model's projection domain.
    PointNotProjectable {
        /// Invalid point index.
        point: usize,
    },
    /// Shared loss, noise, or convergence configuration is invalid.
    Configuration(RefinementError),
}

impl fmt::Display for PnpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeasurementCountMismatch {
                points,
                observations,
            } => write!(
                formatter,
                "expected one observation per world point: points={points}, observations={observations}"
            ),
            Self::InsufficientPoints { required, actual } => write!(
                formatter,
                "PnP refinement requires at least {required} points, received {actual}"
            ),
            Self::NonFinitePoint { point, coordinate } => write!(
                formatter,
                "world point {point}, coordinate {coordinate} is not finite"
            ),
            Self::NonFiniteObservation {
                observation,
                coordinate,
            } => write!(
                formatter,
                "observation {observation}, coordinate {coordinate} is not finite"
            ),
            Self::InvalidBearing { observation } => write!(
                formatter,
                "bearing observation {observation} must have a finite non-zero norm"
            ),
            Self::InvalidPose => formatter.write_str(
                "initial PnP pose must contain a finite SO(3) rotation and finite translation",
            ),
            Self::PointNotProjectable { point } => write!(
                formatter,
                "world point {point} is outside the observation model's projection domain"
            ),
            Self::Configuration(error) => error.fmt(formatter),
        }
    }
}

/// Coordinate units used by PnP residuals and reported RMSE values.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpResidualUnit {
    /// Pixel reprojection coordinates.
    Pixels,
    /// Tangent-space angular coordinates in radians.
    Radians,
}

impl Error for PnpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RefinementError> for PnpError {
    fn from(error: RefinementError) -> Self {
        Self::Configuration(error)
    }
}

/// Refined world-to-camera pose and explicit numerical quality evidence.
#[must_use = "inspect status and diagnostics before accepting the PnP pose"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnpReport {
    /// Refined world-to-camera pose.
    pub pose: RefinementPose,
    /// Loss used by the solve.
    pub loss: Loss,
    /// Units of the two-dimensional residual and RMSE values.
    pub residual_units: PnpResidualUnit,
    /// Explicit numerical solution status.
    pub status: RefinementStatus,
    /// Number of fixed 3D-to-observation correspondences.
    pub point_count: usize,
    /// Scale used to make translation increments dimensionless.
    pub translation_scale: f64,
    /// LM iterations attempted.
    pub iteration_count: usize,
    /// Trial steps accepted.
    pub accepted_step_count: usize,
    /// Trial steps rejected.
    pub rejected_step_count: usize,
    /// Initial robust objective.
    pub initial_objective: f64,
    /// Final robust objective.
    pub final_objective: f64,
    /// Initial unweighted per-coordinate residual root-mean-square.
    pub initial_rmse: f64,
    /// Final unweighted per-coordinate residual root-mean-square.
    pub final_rmse: f64,
    /// Maximum gradient divided by square-root diagonal curvature.
    pub gradient_inf_norm: f64,
    /// Dimensionless local pose-step norm.
    pub step_norm: f64,
    /// Final dimensionless Nielsen damping value.
    pub final_damping: f64,
    /// Gain ratio from the last evaluated finite trial step, if available.
    pub last_gain_ratio: Option<f64>,
    /// Final undamped rank and conditioning evidence.
    pub diagnostics: InformationDiagnostics,
}

impl PnpReport {
    /// Whether solver status and objective evidence permit using this pose.
    ///
    /// This does not replace application-specific cheirality, visibility, or
    /// registration-consistency checks.
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
