// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in fixed-gauge covariance from the final Schur information matrix.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;
use std::ops::Range;

use nalgebra::DMatrix;

use crate::numerics::manifold::Mat3;
use crate::numerics::refine::covariance::uncertainty_model_for_loss;
use crate::numerics::refine::{Loss, NoiseModel, UncertaintyModel};

use super::super::RelativePose;
use super::model::ObservationModel;
use super::schur::{self, GaugeState};
use super::validation;
use super::{
    BundleAdjustmentPriors, FixedVariableBlocks, MultiViewBundleAdjustmentDiagnostics,
    MultiViewBundleAdjustmentError, MultiViewMeasurements, ScaleGauge,
    dense::DenseBundleAdjustmentWorkspace,
};

/// Marginal uncertainty for a full-rank multi-view BA solution.
///
/// Camera ranges start with left-multiplicative rotation radians. The
/// fixed-baseline camera then has two `S2` translation coordinates; other
/// cameras have three translations divided by
/// [`normalization_scale`](Self::normalization_scale). Camera 0 has no pose
/// coordinates. Per-view self-calibration appends `(log_focal, k1, k2)` to
/// every camera range, including camera 0. Grouped self-calibration stores
/// those coordinates in separate ranges returned by
/// [`intrinsic_group_parameter_range`](Self::intrinsic_group_parameter_range).
/// Points are also divided by the normalization scale.
#[must_use = "covariance diagnostics should be checked before use"]
#[derive(Clone, Debug, PartialEq)]
pub struct MultiViewBundleAdjustmentCovariance {
    /// Joint dimension of all free camera-local coordinates.
    pub camera_dimension: usize,
    /// Row-major joint camera covariance.
    pub camera_covariance: Vec<f64>,
    /// One 3x3 marginal covariance per point.
    pub point_covariances: Vec<Mat3>,
    /// Statistical construction used for the returned marginals.
    pub uncertainty_model: UncertaintyModel,
    /// True because the supplied observation noise is included in information.
    pub observation_noise_whitened: bool,
    /// Final undamped rank and conditioning evidence.
    pub diagnostics: MultiViewBundleAdjustmentDiagnostics,
    normalization_scale: f64,
    camera_offsets: Vec<usize>,
    camera_dimensions: Vec<usize>,
    intrinsic_group_offsets: Vec<usize>,
    intrinsic_group_dimensions: Vec<usize>,
    fixed_points: Vec<bool>,
    camera_information: Vec<f64>,
    point_information: Vec<Mat3>,
}

impl MultiViewBundleAdjustmentCovariance {
    /// Row-major range occupied by a camera in the joint camera covariance.
    /// Camera 0 is empty for calibrated and grouped BA, and contains its three
    /// intrinsic coordinates for per-view self-calibration.
    pub fn camera_parameter_range(&self, camera: usize) -> Option<Range<usize>> {
        let (&offset, &dimension) = self
            .camera_offsets
            .get(camera)
            .zip(self.camera_dimensions.get(camera))?;
        Some(offset..offset + dimension)
    }

    /// Row-major range occupied by a radial intrinsic group.
    ///
    /// Returns `None` for calibrated BA or an invalid group. For the per-view
    /// self-calibration API, group `i` is camera `i`'s `(log_focal, k1, k2)`
    /// tail. For grouped self-calibration, ranges are independent blocks after
    /// the pose coordinates.
    #[cfg(feature = "unstable-internals")]
    pub fn intrinsic_group_parameter_range(&self, group: usize) -> Option<Range<usize>> {
        let (&offset, &dimension) = self
            .intrinsic_group_offsets
            .get(group)
            .zip(self.intrinsic_group_dimensions.get(group))?;
        Some(offset..offset + dimension)
    }

    /// Physical scene scale used to normalize translations and points.
    pub const fn normalization_scale(&self) -> f64 {
        self.normalization_scale
    }

    /// Camera-marginal normalized estimation error squared.
    #[cfg(feature = "unstable-internals")]
    pub fn camera_nees(&self, local_error: &[f64]) -> Option<f64> {
        if local_error.len() != self.camera_dimension {
            return None;
        }
        Some(
            (0..self.camera_dimension)
                .map(|row| {
                    local_error[row]
                        * (0..self.camera_dimension)
                            .map(|column| {
                                self.camera_information[row * self.camera_dimension + column]
                                    * local_error[column]
                            })
                            .sum::<f64>()
                })
                .sum(),
        )
    }

    /// Point-marginal normalized estimation error squared.
    #[cfg(feature = "unstable-internals")]
    pub fn point_nees(&self, point: usize, local_error: &[f64; 3]) -> Option<f64> {
        if self.fixed_points.get(point).copied()? {
            return None;
        }
        let information = self.point_information.get(point)?;
        Some(
            (0..3)
                .map(|row| {
                    local_error[row]
                        * (0..3)
                            .map(|column| information[row][column] * local_error[column])
                            .sum::<f64>()
                })
                .sum(),
        )
    }
}

/// Failure to construct fixed-gauge BA covariance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiViewBundleAdjustmentCovarianceError {
    /// The underlying BA state, measurements, or workspace are invalid.
    Problem(MultiViewBundleAdjustmentError),
    /// Ordinary covariance is undefined until the scale gauge is fixed.
    FreeScaleGauge,
    /// The reduced camera system remains singular after gauge fixing.
    CameraRankDeficient {
        /// Observed camera rank.
        rank: usize,
        /// Required camera dimension.
        dimension: usize,
    },
    /// A point cannot be marginalized because its information is singular.
    PointRankDeficient {
        /// Singular point index.
        point: usize,
    },
    /// At least one covariance entry is NaN or infinite.
    NonFiniteResult,
}

impl fmt::Display for MultiViewBundleAdjustmentCovarianceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Problem(error) => error.fmt(formatter),
            Self::FreeScaleGauge => formatter.write_str(
                "ordinary covariance is undefined with a free scale gauge; fix one baseline first",
            ),
            Self::CameraRankDeficient { rank, dimension } => write!(
                formatter,
                "camera information has rank {rank}, but fixed-gauge covariance requires rank {dimension}"
            ),
            Self::PointRankDeficient { point } => {
                write!(formatter, "point {point} has singular marginal information")
            }
            Self::NonFiniteResult => {
                formatter.write_str("BA covariance contains a non-finite value")
            }
        }
    }
}

impl Error for MultiViewBundleAdjustmentCovarianceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Problem(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MultiViewBundleAdjustmentError> for MultiViewBundleAdjustmentCovarianceError {
    fn from(error: MultiViewBundleAdjustmentError) -> Self {
        Self::Problem(error)
    }
}

/// Compute camera and point marginal covariance after fixed-baseline BA.
///
/// This is opt-in because it allocates the returned marginals and costs
/// `O(camera_dimension^3 + points * camera_dimension^2)`. The normal solve
/// remains allocation-free. The result is inverse final IRLS information,
/// whitened by `noise`; robust weights are normalized to one at zero residual
/// so a loss scale cannot arbitrarily change covariance units.
#[allow(clippy::too_many_arguments)]
pub fn covariance_ba_multi_view_dense(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    covariance_for_model(
        poses,
        points,
        ObservationModel::from(measurements),
        gauge,
        workspace,
        loss,
        noise,
        0,
    )
}

/// Compute posterior covariance including exact block fixes and unary priors.
///
/// Exactly fixed point blocks have zero conditional covariance and
/// [`point_nees`](MultiViewBundleAdjustmentCovariance::point_nees) returns
/// `None` for them. A free scale gauge is accepted only when the supplied
/// fixed blocks or priors make the final information full rank.
#[allow(clippy::too_many_arguments)]
pub fn covariance_ba_multi_view_dense_with_factors(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    covariance_for_model_with_factors(
        poses,
        points,
        ObservationModel::from(measurements),
        gauge,
        fixed,
        priors,
        workspace,
        loss,
        noise,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn covariance_for_model(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
    intrinsic_dimension: usize,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    if matches!(gauge, ScaleGauge::Free) {
        return Err(MultiViewBundleAdjustmentCovarianceError::FreeScaleGauge);
    }
    covariance_for_model_with_factors(
        poses,
        points,
        model,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        workspace,
        loss,
        noise,
        intrinsic_dimension,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn covariance_for_model_with_factors(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
    intrinsic_dimension: usize,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    loss.try_validate()
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    noise
        .try_validate(model.observation_count())
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    validation::problem(poses, points, model, gauge, workspace)?;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    validation::projection_domain(model, poses, points)?;
    let dimension = schur::configure_layout_with_fixed(
        workspace,
        poses.len(),
        gauge,
        intrinsic_dimension,
        fixed,
    );
    covariance_with_configured_model(
        poses, points, model, gauge, fixed, priors, workspace, loss, noise, dimension,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub(super) fn covariance_for_grouped_model(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    observations: &[super::PixelObservation],
    point_offsets: &[usize],
    intrinsic_groups: &[super::self_calibration::RadialCamera],
    camera_to_group: &[usize],
    gauge: ScaleGauge,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    if matches!(gauge, ScaleGauge::Free) {
        return Err(MultiViewBundleAdjustmentCovarianceError::FreeScaleGauge);
    }
    covariance_for_grouped_model_with_factors(
        poses,
        points,
        observations,
        point_offsets,
        intrinsic_groups,
        camera_to_group,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        workspace,
        loss,
        noise,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub(super) fn covariance_for_grouped_model_with_factors(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    observations: &[super::PixelObservation],
    point_offsets: &[usize],
    intrinsic_groups: &[super::self_calibration::RadialCamera],
    camera_to_group: &[usize],
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    loss.try_validate()
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    let validation_model = ObservationModel::radial_grouped(
        observations,
        point_offsets,
        intrinsic_groups,
        camera_to_group,
    );
    noise
        .try_validate(validation_model.observation_count())
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    validation::problem(poses, points, validation_model, gauge, workspace)?;
    validation::factors(poses, points, validation_model, gauge, fixed, priors)?;
    let dimension = schur::configure_grouped_layout_with_fixed(
        workspace,
        poses.len(),
        gauge,
        intrinsic_groups.len(),
        fixed,
    );
    let model = ObservationModel::radial_grouped(
        observations,
        point_offsets,
        intrinsic_groups,
        camera_to_group,
    );
    validation::projection_domain(model, poses, points)?;
    covariance_with_configured_model(
        poses, points, model, gauge, fixed, priors, workspace, loss, noise, dimension,
    )
}

#[allow(clippy::too_many_arguments)]
fn covariance_with_configured_model(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
    dimension: usize,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    let normalization_scale = super::optimizer::scene_scale(poses, points, gauge);
    let gauge_state = GaugeState::new(gauge, poses, normalization_scale);
    let zero_weight = loss.sqrt_weight(0.0);
    let assembly = schur::assemble_problem(
        schur::BaProblemView {
            model,
            poses,
            points,
            fixed,
            priors,
        },
        schur::LinearizationPolicy {
            gauge: gauge_state,
            loss,
            noise,
            damping: 0.0,
            prior_sqrt_scale: zero_weight,
        },
        dimension,
        workspace,
    );
    let diagnostics = schur::diagnostics_with_expected_nullity(workspace, dimension, 0, &assembly);
    if !assembly.valid || diagnostics.pose_rank != dimension {
        return Err(
            MultiViewBundleAdjustmentCovarianceError::CameraRankDeficient {
                rank: diagnostics.pose_rank,
                dimension,
            },
        );
    }

    let schur_information = DMatrix::from_fn(dimension, dimension, |row, column| {
        workspace.hessian[row * dimension + column]
    });
    let Some(camera_inverse) = schur_information
        .clone()
        .cholesky()
        .map(|factor| factor.inverse())
    else {
        return Err(
            MultiViewBundleAdjustmentCovarianceError::CameraRankDeficient {
                rank: diagnostics.pose_rank,
                dimension,
            },
        );
    };
    let weight_normalization = zero_weight * zero_weight;
    let camera_covariance: Vec<_> = (0..dimension)
        .flat_map(|row| {
            let camera_inverse = &camera_inverse;
            (0..dimension).map(move |column| camera_inverse[(row, column)] * weight_normalization)
        })
        .collect();
    let camera_information: Vec<_> = (0..dimension)
        .flat_map(|row| {
            let schur_information = &schur_information;
            (0..dimension)
                .map(move |column| schur_information[(row, column)] / weight_normalization)
        })
        .collect();

    let mut point_covariances = Vec::with_capacity(points.len());
    let mut point_information = Vec::with_capacity(points.len());
    let mut point_prior_index = 0;
    for (point_index, point) in points.iter().enumerate() {
        if !fixed.points.is_empty() && fixed.points[point_index] {
            point_covariances.push([[0.0; 3]; 3]);
            point_information.push([[0.0; 3]; 3]);
            continue;
        }
        workspace.cross[..dimension * 3].fill(0.0);
        let layout = schur::VariableLayout {
            offsets: &workspace.camera_offsets,
            pose_dimensions: &workspace.camera_pose_dimensions,
            intrinsic_offsets: &workspace.intrinsic_offsets,
            intrinsic_dimensions: &workspace.intrinsic_dimensions,
            pose_masks: &[],
            #[cfg(feature = "faer")]
            pose_variables: super::CameraPoseVariables::RotationAndTranslation,
        };
        let mut local = schur::linearize_point(
            model,
            poses,
            point,
            point_index,
            gauge_state,
            loss,
            noise,
            layout,
            &mut workspace.cross,
        )
        .ok_or(
            MultiViewBundleAdjustmentCovarianceError::PointRankDeficient { point: point_index },
        )?;
        let point_prior = priors
            .points
            .get(point_prior_index)
            .filter(|prior| prior.point == point_index);
        if let Some(prior) = point_prior {
            schur::accumulate_point_prior(
                &mut local,
                point,
                prior,
                normalization_scale,
                zero_weight,
            );
            point_prior_index += 1;
        }
        let point_block = DMatrix::from_fn(3, 3, |row, column| local.hessian[row][column]);
        let Some(point_inverse) = point_block.cholesky().map(|factor| factor.inverse()) else {
            return Err(
                MultiViewBundleAdjustmentCovarianceError::PointRankDeficient { point: point_index },
            );
        };
        let cross = DMatrix::from_fn(dimension, 3, |row, column| {
            workspace.cross[row * 3 + column]
        });
        let marginal = (&point_inverse
            + &point_inverse * cross.transpose() * &camera_inverse * cross * &point_inverse)
            * weight_normalization;
        let Some(information) = marginal.clone().cholesky().map(|factor| factor.inverse()) else {
            return Err(
                MultiViewBundleAdjustmentCovarianceError::PointRankDeficient { point: point_index },
            );
        };
        point_covariances.push(to_mat3(&marginal));
        point_information.push(to_mat3(&information));
    }
    if !camera_covariance
        .iter()
        .chain(&camera_information)
        .chain(point_covariances.iter().flatten().flatten())
        .chain(point_information.iter().flatten().flatten())
        .all(|value| value.is_finite())
    {
        return Err(MultiViewBundleAdjustmentCovarianceError::NonFiniteResult);
    }

    let intrinsic_group_offsets =
        workspace.intrinsic_offsets[..model.intrinsic_group_count()].to_vec();
    let intrinsic_group_dimensions =
        workspace.intrinsic_dimensions[..model.intrinsic_group_count()].to_vec();
    Ok(MultiViewBundleAdjustmentCovariance {
        camera_dimension: dimension,
        camera_covariance,
        point_covariances,
        uncertainty_model: uncertainty_model_for_loss(loss),
        observation_noise_whitened: true,
        diagnostics,
        normalization_scale,
        camera_offsets: workspace.camera_offsets[..poses.len()].to_vec(),
        camera_dimensions: workspace.camera_dimensions[..poses.len()].to_vec(),
        intrinsic_group_offsets,
        intrinsic_group_dimensions,
        fixed_points: if fixed.points.is_empty() {
            vec![false; points.len()]
        } else {
            fixed.points.to_vec()
        },
        camera_information,
        point_information,
    })
}

fn to_mat3(matrix: &DMatrix<f64>) -> Mat3 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}
