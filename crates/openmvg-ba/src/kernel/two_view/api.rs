// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public data model and typed failures for two-view bundle adjustment.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use crate::numerics::refine::{Loss, RefinementError, RefinementStatus, StepRejections};

use super::{PinholeCamera, RelativePose};

/// Two-view measurements and their reprojection geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TwoViewMeasurements<'a> {
    /// Exact S2 logarithm in the two-dimensional tangent plane of each
    /// observed bearing. Residual units are radians.
    Angular {
        /// Unit or nonzero bearings observed by the fixed first camera.
        bearings1: &'a [[f64; 3]],
        /// Unit or nonzero bearings observed by the refined second camera.
        bearings2: &'a [[f64; 3]],
    },
    /// Standard pinhole reprojection. Residual units are pixels.
    Pixel {
        /// Pixel observations from the fixed first camera.
        observations1: &'a [[f64; 2]],
        /// Pixel observations from the refined second camera.
        observations2: &'a [[f64; 2]],
        /// Fixed intrinsics for cameras one and two, respectively.
        cameras: [PinholeCamera; 2],
    },
}

/// Failure detected before two-view bundle adjustment can safely start.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleAdjustmentError {
    /// Points and the two measurement arrays have different lengths.
    MeasurementCountMismatch {
        /// Number of 3D points.
        points: usize,
        /// Number of measurements in view one.
        view1: usize,
        /// Number of measurements in view two.
        view2: usize,
    },
    /// Too few points were supplied for a constrained solve.
    InsufficientPoints {
        /// Minimum accepted point count.
        required: usize,
        /// Supplied point count.
        actual: usize,
    },
    /// A pixel or bearing component is NaN or infinite.
    NonFiniteMeasurement {
        /// One-based view index.
        view: usize,
        /// Measurement index.
        index: usize,
    },
    /// A bearing is zero length or otherwise invalid.
    InvalidBearing {
        /// One-based view index.
        view: usize,
        /// Bearing index.
        index: usize,
    },
    /// Pinhole focal lengths are non-positive or an intrinsic is non-finite.
    InvalidIntrinsics {
        /// One-based view index.
        view: usize,
    },
    /// A 3D point component is NaN or infinite.
    NonFinitePoint {
        /// Point index.
        index: usize,
        /// Coordinate index within the point.
        coordinate: usize,
    },
    /// The initial rotation is not a finite member of `SO(3)`.
    InvalidRotation,
    /// The baseline translation is zero length or non-finite.
    InvalidTranslation,
    /// A point lies outside the selected residual's projection domain.
    PointNotProjectable {
        /// One-based view index.
        view: usize,
        /// Point index.
        index: usize,
    },
    /// Shared refinement options or noise configuration are invalid.
    Configuration(RefinementError),
}

impl fmt::Display for BundleAdjustmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeasurementCountMismatch {
                points,
                view1,
                view2,
            } => write!(
                formatter,
                "expected one measurement per 3D point: points={points}, view1={view1}, view2={view2}"
            ),
            Self::InsufficientPoints { required, actual } => write!(
                formatter,
                "two-view bundle adjustment requires at least {required} points, received {actual}"
            ),
            Self::NonFiniteMeasurement { view, index } => {
                write!(formatter, "view {view} measurement {index} is not finite")
            }
            Self::InvalidBearing { view, index } => write!(
                formatter,
                "view {view} bearing {index} must be finite and nonzero"
            ),
            Self::InvalidIntrinsics { view } => write!(
                formatter,
                "view {view} intrinsics must be finite with positive focal lengths"
            ),
            Self::NonFinitePoint { index, coordinate } => write!(
                formatter,
                "3D point {index}, coordinate {coordinate} is not finite"
            ),
            Self::InvalidRotation => formatter
                .write_str("relative rotation must be a finite right-handed orthonormal matrix"),
            Self::InvalidTranslation => formatter.write_str(
                "relative translation must be finite and nonzero; its norm fixes the scale gauge",
            ),
            Self::PointNotProjectable { view, index } => write!(
                formatter,
                "3D point {index} is outside the valid projection domain of view {view}"
            ),
            Self::Configuration(error) => error.fmt(formatter),
        }
    }
}

impl Error for BundleAdjustmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RefinementError> for BundleAdjustmentError {
    fn from(error: RefinementError) -> Self {
        Self::Configuration(error)
    }
}

/// Pose result and explicit numerical quality of the in-place point update.
#[must_use = "inspect status and diagnostics before accepting the BA solution"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleAdjustmentReport {
    /// Refined second-camera world-to-camera pose.
    pub pose: RelativePose,
    /// Loss used by the solve.
    pub loss: Loss,
    /// Explicit numerical solution status.
    pub status: RefinementStatus,
    /// Number of refined 3D points.
    pub point_count: usize,
    /// Preserved translation norm that fixes monocular scale.
    pub fixed_baseline: f64,
    /// LM iterations attempted.
    pub iteration_count: usize,
    /// Trial steps accepted.
    pub accepted_step_count: usize,
    /// Trial steps rejected.
    pub rejected_step_count: usize,
    /// The rejected steps by reason; their total is `rejected_step_count`.
    pub step_rejections: StepRejections,
    /// Initial robust objective.
    pub initial_cost: f64,
    /// Final robust objective.
    pub final_cost: f64,
    /// Initial unweighted residual root-mean-square.
    pub initial_rmse: f64,
    /// Final unweighted residual root-mean-square.
    pub final_rmse: f64,
    /// Maximum full pose-or-point gradient divided by the square root of its
    /// corresponding diagonal Gauss--Newton curvature.
    pub gradient_inf_norm: f64,
    /// Root-sum-square of pose and dimensionless point-step RMS values.
    pub step_norm: f64,
    /// Final dimensionless Nielsen damping value.
    pub final_damping: f64,
    /// Gain ratio from the last evaluated finite trial step, if available.
    pub last_gain_ratio: Option<f64>,
    /// Final undamped rank and conditioning evidence.
    pub diagnostics: BundleAdjustmentDiagnostics,
}

/// Rank and conditioning evidence for the final undamped Schur system.
///
/// The reciprocal condition numbers refer to the Gauss--Newton Hessian blocks,
/// not their Jacobians. The camera value is spectral, while the point value is
/// the inexpensive infinity-norm estimate obtained from each scaled 3x3
/// Cholesky factor. A value near zero means that the estimate is sensitive to small
/// measurement perturbations. The two-view solver fixes the scale gauge by
/// preserving the initial baseline, so its expected camera nullity is zero.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleAdjustmentDiagnostics {
    /// Dimension of the reduced camera system; five for calibrated two-view BA.
    pub pose_dimension: usize,
    /// Numerical rank of the final undamped reduced camera Hessian.
    pub pose_rank: usize,
    /// Smallest/largest eigenvalue ratio of the reduced camera Hessian.
    pub pose_hessian_reciprocal_condition_number: f64,
    /// Number of 3x3 point Hessians rejected as numerically singular.
    pub weak_point_count: usize,
    /// Smallest infinity-norm reciprocal condition estimate among point blocks.
    pub minimum_point_hessian_reciprocal_condition_number: f64,
}

impl BundleAdjustmentDiagnostics {
    /// Whether the reduced camera system has all observable directions.
    pub const fn camera_is_full_rank(self) -> bool {
        self.pose_rank == self.pose_dimension
    }

    /// Whether both the camera system and every eliminated point block have
    /// full numerical rank.
    pub const fn is_well_constrained(self) -> bool {
        self.camera_is_full_rank() && self.weak_point_count == 0
    }
}
