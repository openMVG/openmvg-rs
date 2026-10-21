// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in covariance for fixed-baseline two-view bundle adjustment.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use nalgebra::{SMatrix, SVector};

use crate::numerics::manifold::{self, Mat3};
use crate::numerics::refine::covariance::uncertainty_model_for_loss;
use crate::numerics::refine::{Loss, NoiseModel, UncertaintyModel};

use super::optimizer;
use super::residual::{AngularModel, MeasurementModel, PixelModel};
use super::schur::{self, POSE_DIMENSION, ReducedSystem};
use super::{
    BundleAdjustmentDiagnostics, BundleAdjustmentError, RelativePose, TwoViewMeasurements,
};

const POINT_DIMENSION: usize = 3;

/// Camera and point marginal uncertainty for fixed-baseline two-view BA.
///
/// Pose coordinates are three left-multiplicative rotation radians followed
/// by two `S2` translation-direction coordinates. Point coordinates are
/// divided by [`normalization_scale`](Self::normalization_scale), which is the
/// preserved baseline length.
#[must_use = "covariance diagnostics should be checked before use"]
#[derive(Clone, Debug, PartialEq)]
pub struct TwoViewBundleAdjustmentCovariance {
    /// Marginal pose covariance in row-major local coordinates.
    pub pose_covariance: [[f64; POSE_DIMENSION]; POSE_DIMENSION],
    /// One marginal covariance per point in dimensionless local coordinates.
    pub point_covariances: Vec<Mat3>,
    /// Statistical construction used for the returned marginals.
    pub uncertainty_model: UncertaintyModel,
    /// True because the supplied observation noise is included in information.
    pub observation_noise_whitened: bool,
    /// Final undamped rank and conditioning evidence.
    pub diagnostics: BundleAdjustmentDiagnostics,
    normalization_scale: f64,
    pose_information: [[f64; POSE_DIMENSION]; POSE_DIMENSION],
    point_information: Vec<Mat3>,
}

impl TwoViewBundleAdjustmentCovariance {
    /// Baseline used to normalize every point increment.
    pub const fn normalization_scale(&self) -> f64 {
        self.normalization_scale
    }

    /// Pose-marginal normalized estimation error squared.
    pub fn pose_nees(&self, local_error: &[f64; POSE_DIMENSION]) -> f64 {
        quadratic_form(&self.pose_information, local_error)
    }

    /// Point-marginal normalized estimation error squared.
    pub fn point_nees(&self, point: usize, local_error: &[f64; 3]) -> Option<f64> {
        self.point_information
            .get(point)
            .map(|information| quadratic_form(information, local_error))
    }
}

/// Failure to construct two-view BA covariance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoViewBundleAdjustmentCovarianceError {
    /// The underlying BA state or measurements are invalid.
    Problem(BundleAdjustmentError),
    /// The reduced five-dimensional pose system is singular.
    PoseRankDeficient {
        /// Observed pose rank.
        rank: usize,
    },
    /// A point cannot be marginalized because its information is singular.
    PointRankDeficient {
        /// Singular point index.
        point: usize,
    },
    /// At least one covariance entry is NaN or infinite.
    NonFiniteResult,
}

impl fmt::Display for TwoViewBundleAdjustmentCovarianceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Problem(error) => error.fmt(formatter),
            Self::PoseRankDeficient { rank } => write!(
                formatter,
                "two-view pose information has rank {rank}, but covariance requires rank {POSE_DIMENSION}"
            ),
            Self::PointRankDeficient { point } => {
                write!(formatter, "point {point} has singular marginal information")
            }
            Self::NonFiniteResult => {
                formatter.write_str("two-view BA covariance contains a non-finite value")
            }
        }
    }
}

impl Error for TwoViewBundleAdjustmentCovarianceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Problem(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BundleAdjustmentError> for TwoViewBundleAdjustmentCovarianceError {
    fn from(error: BundleAdjustmentError) -> Self {
        Self::Problem(error)
    }
}

/// Compute opt-in pose and point marginal covariance after two-view BA.
///
/// The refinement hot path remains allocation-free; this separate function
/// allocates only its returned point marginals. It uses inverse final IRLS
/// information, whitened by `noise`. Robust weights are normalized to one at
/// zero residual so the loss scale cannot arbitrarily change covariance units.
/// With a robust loss the result is a fixed-weight local approximation: it is
/// not calibrated under outlier contamination and then understates the error.
pub fn covariance_ba_2view(
    pose: &RelativePose,
    points: &[[f64; 3]],
    measurements: TwoViewMeasurements<'_>,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<TwoViewBundleAdjustmentCovariance, TwoViewBundleAdjustmentCovarianceError> {
    loss.try_validate()
        .map_err(BundleAdjustmentError::Configuration)?;
    noise
        .try_validate(points.len())
        .map_err(BundleAdjustmentError::Configuration)?;
    optimizer::validate_pose(pose)?;
    optimizer::validate_points(points)?;
    if points.len() < 5 {
        return Err(BundleAdjustmentError::InsufficientPoints {
            required: 5,
            actual: points.len(),
        }
        .into());
    }
    match measurements {
        TwoViewMeasurements::Angular {
            bearings1,
            bearings2,
        } => covariance_impl(
            pose,
            points,
            AngularModel {
                bearings1,
                bearings2,
            },
            loss,
            noise,
        ),
        TwoViewMeasurements::Pixel {
            observations1,
            observations2,
            cameras,
        } => covariance_impl(
            pose,
            points,
            PixelModel {
                observations1,
                observations2,
                cameras,
            },
            loss,
            noise,
        ),
    }
}

fn covariance_impl<M: MeasurementModel>(
    pose: &RelativePose,
    points: &[[f64; 3]],
    model: M,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<TwoViewBundleAdjustmentCovariance, TwoViewBundleAdjustmentCovarianceError> {
    model.validate(points.len())?;
    optimizer::validate_projection_domain(model, pose, points)?;
    let baseline = norm(&pose.translation);
    let translation_direction = pose.translation.map(|value| value / baseline);
    let translation_basis = manifold::s2_tangent_basis(&translation_direction);
    let mut system = ReducedSystem::new();
    let mut local_terms = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let local = schur::linearize_point(
            model,
            pose,
            &translation_basis,
            baseline,
            point,
            index,
            loss,
            noise,
        )
        .ok_or(BundleAdjustmentError::PointNotProjectable { view: 1, index })?;
        system.add(&local, 0.0);
        local_terms.push(local);
    }
    system.finish_damping(0.0);
    let diagnostics = system.diagnostics();
    if !system.is_valid() || diagnostics.pose_rank != POSE_DIMENSION {
        return Err(TwoViewBundleAdjustmentCovarianceError::PoseRankDeficient {
            rank: diagnostics.pose_rank,
        });
    }

    let pose_information_raw =
        SMatrix::<f64, POSE_DIMENSION, POSE_DIMENSION>::from_fn(|row, column| {
            system.hessian[row][column]
        });
    let Some(pose_inverse_raw) = pose_information_raw
        .cholesky()
        .map(|factor| factor.inverse())
    else {
        return Err(TwoViewBundleAdjustmentCovarianceError::PoseRankDeficient {
            rank: diagnostics.pose_rank,
        });
    };
    let zero_weight = loss.sqrt_weight(0.0);
    let weight_normalization = zero_weight * zero_weight;
    let pose_covariance = std::array::from_fn(|row| {
        std::array::from_fn(|column| pose_inverse_raw[(row, column)] * weight_normalization)
    });
    let pose_information = std::array::from_fn(|row| {
        std::array::from_fn(|column| pose_information_raw[(row, column)] / weight_normalization)
    });

    let mut point_covariances = Vec::with_capacity(points.len());
    let mut point_information = Vec::with_capacity(points.len());
    for (point, local) in local_terms.iter().enumerate() {
        let point_block =
            SMatrix::<f64, 3, 3>::from_fn(|row, column| local.point_hessian[row][column]);
        let Some(point_inverse_raw) = point_block.cholesky().map(|factor| factor.inverse()) else {
            return Err(TwoViewBundleAdjustmentCovarianceError::PointRankDeficient { point });
        };
        let cross = SMatrix::<f64, POSE_DIMENSION, POINT_DIMENSION>::from_fn(|row, column| {
            local.cross_hessian[row][column]
        });
        let marginal = (point_inverse_raw
            + point_inverse_raw * cross.transpose() * pose_inverse_raw * cross * point_inverse_raw)
            * weight_normalization;
        let Some(information) = marginal.cholesky().map(|factor| factor.inverse()) else {
            return Err(TwoViewBundleAdjustmentCovarianceError::PointRankDeficient { point });
        };
        point_covariances.push(to_mat3(&marginal));
        point_information.push(to_mat3(&information));
    }
    if !pose_covariance
        .iter()
        .chain(&pose_information)
        .flatten()
        .chain(point_covariances.iter().flatten().flatten())
        .chain(point_information.iter().flatten().flatten())
        .all(|value| value.is_finite())
    {
        return Err(TwoViewBundleAdjustmentCovarianceError::NonFiniteResult);
    }

    Ok(TwoViewBundleAdjustmentCovariance {
        pose_covariance,
        point_covariances,
        uncertainty_model: uncertainty_model_for_loss(loss),
        observation_noise_whitened: true,
        diagnostics,
        normalization_scale: baseline,
        pose_information,
        point_information,
    })
}

fn quadratic_form<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> f64 {
    (0..N)
        .map(|row| {
            vector[row]
                * (0..N)
                    .map(|column| matrix[row][column] * vector[column])
                    .sum::<f64>()
        })
        .sum()
}

fn to_mat3(matrix: &SMatrix<f64, 3, 3>) -> Mat3 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}

fn norm(vector: &[f64; 3]) -> f64 {
    SVector::<f64, 3>::from_row_slice(vector).norm()
}
