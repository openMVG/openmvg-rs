// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in selected marginal covariance from exact sparse Schur information.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use crate::numerics::manifold::Mat3;
#[cfg(any(test, feature = "unstable-internals"))]
use crate::numerics::refine::UncertaintyModel;

use super::super::MultiViewBundleAdjustmentError;
use super::SparseBundleAdjustmentError;
use super::direct::solve_factored_right_hand_side;
use super::workspace::SparseWorkspaceCore;
#[cfg(any(test, feature = "unstable-internals"))]
use super::{
    super::{
        BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks, MultiViewMeasurements,
        ScaleGauge,
        model::ObservationModel,
        optimizer,
        schur::{self, GaugeState},
        validation,
    },
    SparseBundleAdjustmentWorkspace,
    calibrated::assemble_sparse_with_prior_scale,
    direct::factor_camera_system,
};
#[cfg(any(test, feature = "unstable-internals"))]
use crate::kernel::{RelativePose, point_block::PointBlockFactor};
#[cfg(any(test, feature = "unstable-internals"))]
use crate::numerics::refine::{Loss, NoiseModel, covariance::uncertainty_model_for_loss};

/// Selected full six-coordinate camera-pose marginal.
///
/// Coordinates are three left-multiplicative rotation radians followed by
/// three translations divided by the result's normalization scale.
#[must_use = "camera marginal covariance should be inspected"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(any(test, feature = "unstable-internals"))]
pub struct SparseCameraMarginalCovariance {
    /// Camera index in the caller's pose slice.
    pub camera: usize,
    /// Row-major 6x6 marginal covariance.
    pub matrix: [[f64; 6]; 6],
}

/// Selected three-coordinate Euclidean-point marginal.
///
/// Point coordinates are divided by the result's normalization scale.
#[must_use = "point marginal covariance should be inspected"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(any(test, feature = "unstable-internals"))]
pub struct SparsePointMarginalCovariance {
    /// Point index in the caller's point slice.
    pub point: usize,
    /// Row-major 3x3 marginal covariance.
    pub matrix: Mat3,
}

/// Radial-intrinsic marginal of one free group of a self-calibrating sparse BA.
///
/// Coordinates are the group's free leading `(log focal, k1, k2)` parameters,
/// so focal uncertainty is relative. Only the leading `dimension` rows and
/// columns of `matrix` are populated; the rest are zero.
#[must_use = "intrinsic marginal covariance should be inspected"]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SparseIntrinsicMarginalCovariance {
    /// Intrinsic group index.
    pub group: usize,
    /// Number of free leading `(log focal, k1, k2)` parameters.
    pub dimension: usize,
    /// Row-major 3x3 marginal covariance, zero-padded beyond `dimension`.
    pub matrix: Mat3,
}

/// Opt-in selected marginals from exact final sparse BA information.
#[must_use = "marginal covariance and its statistical model should be inspected"]
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg(any(test, feature = "unstable-internals"))]
pub struct SparseBundleAdjustmentMarginalCovariances {
    /// Camera marginals in caller selection order.
    pub cameras: Vec<SparseCameraMarginalCovariance>,
    /// Point marginals in caller selection order.
    pub points: Vec<SparsePointMarginalCovariance>,
    /// Statistical construction represented by the returned matrices.
    pub uncertainty_model: UncertaintyModel,
    /// True because the supplied observation noise is included in information.
    pub observation_noise_whitened: bool,
    normalization_scale: f64,
}

#[cfg(any(test, feature = "unstable-internals"))]
impl SparseBundleAdjustmentMarginalCovariances {
    /// Physical scene scale used to normalize translation and point tangents.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn normalization_scale(&self) -> f64 {
        self.normalization_scale
    }
}

/// Failure to compute selected sparse BA marginal covariance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseBundleAdjustmentCovarianceError {
    /// The supplied state, measurements, factors, loss, or noise are invalid.
    Problem(MultiViewBundleAdjustmentError),
    /// Measurements, gauge, or fixed blocks differ from workspace topology.
    TopologyMismatch,
    /// Ordinary covariance is undefined for a free scale gauge.
    FreeScaleGauge,
    /// Selected covariance supports only the legacy uniform pose layouts.
    /// Explicit per-camera masks require a covariance API with their actual chart.
    #[cfg(any(test, feature = "unstable-internals"))]
    UnsupportedPoseMasks,
    /// Translation-only workspaces require a dedicated three-coordinate
    /// camera-marginal API.
    #[cfg(any(test, feature = "unstable-internals"))]
    TranslationOnlyUnsupported,
    /// A camera selection is outside the pose slice.
    #[cfg(any(test, feature = "unstable-internals"))]
    SelectedCameraOutOfBounds {
        /// Position in the camera-selection slice.
        selection: usize,
        /// Invalid camera index.
        camera: usize,
        /// Available camera count.
        camera_count: usize,
    },
    /// A camera appears more than once in the selection.
    #[cfg(any(test, feature = "unstable-internals"))]
    DuplicateSelectedCamera {
        /// Repeated camera index.
        camera: usize,
    },
    /// A selected camera is exactly fixed and has no covariance coordinates.
    #[cfg(any(test, feature = "unstable-internals"))]
    SelectedCameraFixed {
        /// Fixed camera index.
        camera: usize,
    },
    /// A selected camera is the five-coordinate scale-gauge block.
    #[cfg(any(test, feature = "unstable-internals"))]
    SelectedCameraGaugeConstrained {
        /// Gauge-constrained camera index.
        camera: usize,
    },
    /// A point selection is outside the point slice.
    #[cfg(any(test, feature = "unstable-internals"))]
    SelectedPointOutOfBounds {
        /// Position in the point-selection slice.
        selection: usize,
        /// Invalid point index.
        point: usize,
        /// Available point count.
        point_count: usize,
    },
    /// A point appears more than once in the selection.
    #[cfg(any(test, feature = "unstable-internals"))]
    DuplicateSelectedPoint {
        /// Repeated point index.
        point: usize,
    },
    /// A selected point is exactly fixed and has no estimated covariance.
    #[cfg(any(test, feature = "unstable-internals"))]
    SelectedPointFixed {
        /// Fixed point index.
        point: usize,
    },
    /// Exact sparse pattern, factor, or numerical storage could not be prepared.
    SparseStorage(SparseBundleAdjustmentError),
    /// Final undamped information contains NaN or infinity.
    NonFiniteInformation,
    /// Final undamped information is rank deficient or indefinite.
    RankDeficientOrIndefinite,
    /// A selected point's three-coordinate information is singular.
    #[cfg(any(test, feature = "unstable-internals"))]
    PointRankDeficient {
        /// Singular selected point.
        point: usize,
    },
    /// A selected covariance solve produced NaN or infinity.
    NonFiniteResult,
    /// Returned selection storage could not be allocated.
    #[cfg(any(test, feature = "unstable-internals"))]
    OutputAllocation,
}

impl fmt::Display for SparseBundleAdjustmentCovarianceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Problem(error) => error.fmt(formatter),
            Self::TopologyMismatch => formatter.write_str(
                "measurements, gauge, or fixed blocks do not match the sparse covariance workspace",
            ),
            Self::FreeScaleGauge => formatter.write_str(
                "ordinary covariance is undefined with a free scale gauge; fix scale first",
            ),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::UnsupportedPoseMasks => formatter.write_str(
                "selected sparse BA covariance is unsupported for explicit pose masks; use a legacy uniform workspace",
            ),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::TranslationOnlyUnsupported => formatter.write_str(
                "translation-only sparse BA covariance is not supported; use a full-pose workspace",
            ),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::SelectedCameraOutOfBounds {
                selection,
                camera,
                camera_count,
            } => write!(
                formatter,
                "camera covariance selection {selection} refers to camera {camera}, but only {camera_count} cameras exist"
            ),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::DuplicateSelectedCamera { camera } => {
                write!(
                    formatter,
                    "camera covariance selects camera {camera} more than once"
                )
            }
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::SelectedCameraFixed { camera } => {
                write!(
                    formatter,
                    "camera {camera} is exactly fixed and has no covariance block"
                )
            }
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::SelectedCameraGaugeConstrained { camera } => write!(
                formatter,
                "camera {camera} carries the five-coordinate scale gauge; only full 6-DOF camera marginals are supported"
            ),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::SelectedPointOutOfBounds {
                selection,
                point,
                point_count,
            } => write!(
                formatter,
                "point covariance selection {selection} refers to point {point}, but only {point_count} points exist"
            ),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::DuplicateSelectedPoint { point } => {
                write!(
                    formatter,
                    "point covariance selects point {point} more than once"
                )
            }
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::SelectedPointFixed { point } => {
                write!(
                    formatter,
                    "point {point} is exactly fixed and has no estimated covariance"
                )
            }
            Self::SparseStorage(error) => error.fmt(formatter),
            Self::NonFiniteInformation => {
                formatter.write_str("sparse BA covariance information is non-finite")
            }
            Self::RankDeficientOrIndefinite => formatter
                .write_str("sparse BA covariance requires full-rank fixed-gauge information"),
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::PointRankDeficient { point } => {
                write!(formatter, "point {point} has singular marginal information")
            }
            Self::NonFiniteResult => {
                formatter.write_str("sparse BA covariance contains a non-finite value")
            }
            #[cfg(any(test, feature = "unstable-internals"))]
            Self::OutputAllocation => {
                formatter.write_str("sparse BA covariance output allocation failed")
            }
        }
    }
}

impl Error for SparseBundleAdjustmentCovarianceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Problem(error) => Some(error),
            Self::SparseStorage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MultiViewBundleAdjustmentError> for SparseBundleAdjustmentCovarianceError {
    fn from(error: MultiViewBundleAdjustmentError) -> Self {
        Self::Problem(error)
    }
}

/// Compute selected calibrated sparse-BA marginal covariances.
///
/// The final undamped IRLS information is rebuilt at `poses` and `points`,
/// factorized by the workspace's exact sparse LLT storage, and solved only for
/// the requested six-coordinate camera and three-coordinate point blocks. No
/// full inverse is formed. Results preserve selection order and duplicate or
/// out-of-range selections are rejected.
///
/// The fixed scale gauge must not be carried by a selected camera because that
/// camera has only five local coordinates. Camera 0 and explicitly fixed
/// blocks are also rejected. Robust losses report fixed-final-weight
/// uncertainty with weights normalized to one at zero residual, preserving
/// the ordinary information units. This opt-in query may allocate its
/// returned vectors and lazy direct storage; ordinary sparse refinement
/// remains unchanged.
/// A free scale gauge is rejected by this no-factor entry point. Explicit
/// per-camera-mask workspaces are rejected with
/// [`SparseBundleAdjustmentCovarianceError::UnsupportedPoseMasks`]; selected
/// covariance currently supports only legacy uniform pose layouts.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "unstable-internals"))]
pub fn selected_marginal_covariances_ba_multi_view_sparse(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    selected_cameras: &[usize],
    selected_points: &[usize],
    workspace: &mut SparseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentMarginalCovariances, SparseBundleAdjustmentCovarianceError> {
    if workspace.core.signature.explicit_pose_masks() {
        return Err(SparseBundleAdjustmentCovarianceError::UnsupportedPoseMasks);
    }
    if gauge == ScaleGauge::Free {
        return Err(SparseBundleAdjustmentCovarianceError::FreeScaleGauge);
    }
    selected_marginal_covariances_ba_multi_view_sparse_with_factors(
        poses,
        points,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        selected_cameras,
        selected_points,
        workspace,
        loss,
        noise,
    )
}

/// Compute selected calibrated sparse-BA marginals with exact fixes and priors.
///
/// Priors contribute ordinary information. Exactly fixed blocks cannot be
/// selected. A free scale gauge is accepted when the supplied exact fixes or
/// priors make the final information full rank. The workspace must have been
/// compiled with the same fixed masks. Explicit per-camera-mask workspaces are
/// rejected with [`SparseBundleAdjustmentCovarianceError::UnsupportedPoseMasks`]
/// before selecting or rebuilding information; this query supports only the
/// legacy uniform pose charts.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "unstable-internals"))]
pub fn selected_marginal_covariances_ba_multi_view_sparse_with_factors(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    selected_cameras: &[usize],
    selected_points: &[usize],
    workspace: &mut SparseBundleAdjustmentWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentMarginalCovariances, SparseBundleAdjustmentCovarianceError> {
    if workspace.core.signature.explicit_pose_masks() {
        return Err(SparseBundleAdjustmentCovarianceError::UnsupportedPoseMasks);
    }
    loss.try_validate()
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    if matches!(
        workspace.core.signature.pose_variables(),
        CameraPoseVariables::TranslationOnly
    ) {
        return Err(SparseBundleAdjustmentCovarianceError::TranslationOnlyUnsupported);
    }
    if !workspace.matches_with_fixed(poses.len(), measurements, gauge, fixed)
        || points.len() != workspace.point_count()
    {
        return Err(SparseBundleAdjustmentCovarianceError::TopologyMismatch);
    }
    let model = ObservationModel::from(measurements);
    noise
        .try_validate(model.observation_count())
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    validation::problem_with_capacity(
        poses,
        points,
        model,
        gauge,
        workspace.camera_count(),
        workspace.point_count(),
    )?;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    validation::projection_domain(model, poses, points)?;
    validate_selection(
        selected_cameras,
        selected_points,
        poses.len(),
        points.len(),
        workspace,
    )?;

    let mut camera_marginals = Vec::new();
    camera_marginals
        .try_reserve_exact(selected_cameras.len())
        .map_err(|_| SparseBundleAdjustmentCovarianceError::OutputAllocation)?;
    let mut point_marginals = Vec::new();
    point_marginals
        .try_reserve_exact(selected_points.len())
        .map_err(|_| SparseBundleAdjustmentCovarianceError::OutputAllocation)?;

    let normalization_scale = optimizer::scene_scale(poses, points, gauge);
    let gauge_state = GaugeState::new(gauge, poses, normalization_scale);
    let zero_weight = loss.sqrt_weight(0.0);
    if !(zero_weight.is_finite() && zero_weight > 0.0) {
        return Err(SparseBundleAdjustmentCovarianceError::NonFiniteInformation);
    }
    workspace
        .ensure_direct_storage()
        .map_err(SparseBundleAdjustmentCovarianceError::SparseStorage)?;
    let assembly = assemble_sparse_with_prior_scale(
        model,
        poses,
        points,
        gauge_state,
        loss,
        noise,
        0.0,
        false,
        fixed,
        priors,
        zero_weight,
        workspace,
    );
    if !information_is_finite(&workspace.core) {
        return Err(SparseBundleAdjustmentCovarianceError::NonFiniteInformation);
    }
    if !assembly.valid {
        return Err(SparseBundleAdjustmentCovarianceError::RankDeficientOrIndefinite);
    }
    let dimension = workspace.dimension();
    if dimension > 0 && !factor_camera_system(&mut workspace.core) {
        return Err(SparseBundleAdjustmentCovarianceError::RankDeficientOrIndefinite);
    }

    let weight_normalization = zero_weight * zero_weight;
    for &camera in selected_cameras {
        let offset = workspace.camera_offsets()[camera];
        let matrix = block_marginal(offset, 6, weight_normalization, &mut workspace.core)?;
        camera_marginals.push(SparseCameraMarginalCovariance { camera, matrix });
    }
    for &point in selected_points {
        let matrix = point_marginal(
            point,
            model,
            poses,
            points,
            gauge_state,
            loss,
            noise,
            priors,
            zero_weight,
            weight_normalization,
            &mut workspace.core,
        )?;
        point_marginals.push(SparsePointMarginalCovariance { point, matrix });
    }

    Ok(SparseBundleAdjustmentMarginalCovariances {
        cameras: camera_marginals,
        points: point_marginals,
        uncertainty_model: uncertainty_model_for_loss(loss),
        observation_noise_whitened: true,
        normalization_scale,
    })
}

#[cfg(any(test, feature = "unstable-internals"))]
fn validate_selection(
    selected_cameras: &[usize],
    selected_points: &[usize],
    camera_count: usize,
    point_count: usize,
    workspace: &SparseBundleAdjustmentWorkspace,
) -> Result<(), SparseBundleAdjustmentCovarianceError> {
    for (selection, &camera) in selected_cameras.iter().enumerate() {
        if camera >= camera_count {
            return Err(
                SparseBundleAdjustmentCovarianceError::SelectedCameraOutOfBounds {
                    selection,
                    camera,
                    camera_count,
                },
            );
        }
        if selected_cameras[..selection].contains(&camera) {
            return Err(SparseBundleAdjustmentCovarianceError::DuplicateSelectedCamera { camera });
        }
        match workspace.camera_dimensions()[camera] {
            0 => return Err(SparseBundleAdjustmentCovarianceError::SelectedCameraFixed { camera }),
            5 => {
                return Err(
                    SparseBundleAdjustmentCovarianceError::SelectedCameraGaugeConstrained {
                        camera,
                    },
                );
            }
            6 => {}
            _ => return Err(SparseBundleAdjustmentCovarianceError::UnsupportedPoseMasks),
        }
    }
    for (selection, &point) in selected_points.iter().enumerate() {
        if point >= point_count {
            return Err(
                SparseBundleAdjustmentCovarianceError::SelectedPointOutOfBounds {
                    selection,
                    point,
                    point_count,
                },
            );
        }
        if selected_points[..selection].contains(&point) {
            return Err(SparseBundleAdjustmentCovarianceError::DuplicateSelectedPoint { point });
        }
        if workspace
            .fixed_points()
            .get(point)
            .copied()
            .unwrap_or(false)
        {
            return Err(SparseBundleAdjustmentCovarianceError::SelectedPointFixed { point });
        }
    }
    Ok(())
}

pub(super) fn information_is_finite(workspace: &SparseWorkspaceCore) -> bool {
    workspace.values.iter().all(|value| value.is_finite())
        && workspace.gradient.iter().all(|value| value.is_finite())
        && workspace
            .full_gradient
            .iter()
            .all(|value| value.is_finite())
        && workspace.diagonal.iter().all(|value| value.is_finite())
        && workspace.cross.iter().all(|value| value.is_finite())
}

/// Weight-normalized marginal of the `dimension` coordinates at `offset` of the factored camera
/// system, zero-padded beyond `dimension`.
pub(super) fn block_marginal<const N: usize>(
    offset: usize,
    dimension: usize,
    weight_normalization: f64,
    workspace: &mut SparseWorkspaceCore,
) -> Result<[[f64; N]; N], SparseBundleAdjustmentCovarianceError> {
    let mut matrix = [[0.0; N]; N];
    for column in 0..dimension {
        workspace.step.fill(0.0);
        workspace.step[offset + column] = 1.0;
        if !solve_factored_right_hand_side(workspace) {
            return Err(SparseBundleAdjustmentCovarianceError::NonFiniteResult);
        }
        for (row, values) in matrix.iter_mut().enumerate().take(dimension) {
            values[column] = workspace.step[offset + row] * weight_normalization;
        }
    }
    symmetrize(&mut matrix);
    if matrix.iter().flatten().all(|value| value.is_finite()) {
        Ok(matrix)
    } else {
        Err(SparseBundleAdjustmentCovarianceError::NonFiniteResult)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "unstable-internals"))]
fn point_marginal(
    point_index: usize,
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    priors: BundleAdjustmentPriors<'_>,
    prior_sqrt_scale: f64,
    weight_normalization: f64,
    workspace: &mut SparseWorkspaceCore,
) -> Result<Mat3, SparseBundleAdjustmentCovarianceError> {
    let dimension = workspace.dimension();
    workspace.cross[..dimension * 3].fill(0.0);
    let layout = &workspace.layout;
    let mut local = schur::linearize_point(
        model,
        poses,
        &points[point_index],
        point_index,
        gauge,
        loss,
        noise,
        schur::VariableLayout::calibrated(layout.camera_offsets(), layout.camera_dimensions()),
        &mut workspace.cross,
    )
    .ok_or(SparseBundleAdjustmentCovarianceError::NonFiniteInformation)?;
    if let Ok(prior_index) = priors
        .points
        .binary_search_by_key(&point_index, |prior| prior.point)
    {
        schur::accumulate_point_prior(
            &mut local,
            &points[point_index],
            &priors.points[prior_index],
            gauge.scene_scale,
            prior_sqrt_scale,
        );
    }
    let factor = PointBlockFactor::new(&local.hessian, 0.0)
        .ok_or(SparseBundleAdjustmentCovarianceError::PointRankDeficient { point: point_index })?;
    let inverse_columns: Mat3 = std::array::from_fn(|column| {
        let mut basis = [0.0; 3];
        basis[column] = 1.0;
        factor.solve(&basis)
    });
    let inverse: Mat3 =
        std::array::from_fn(|row| std::array::from_fn(|column| inverse_columns[column][row]));
    let mut matrix = inverse;
    if dimension > 0 {
        for column in 0..3 {
            workspace.step.fill(0.0);
            for row in 0..dimension {
                workspace.step[row] = (0..3)
                    .map(|coordinate| {
                        workspace.cross[row * 3 + coordinate] * inverse[coordinate][column]
                    })
                    .sum();
            }
            if !solve_factored_right_hand_side(workspace) {
                return Err(SparseBundleAdjustmentCovarianceError::NonFiniteResult);
            }
            let projected = std::array::from_fn(|coordinate| {
                (0..dimension)
                    .map(|row| workspace.cross[row * 3 + coordinate] * workspace.step[row])
                    .sum()
            });
            let correction = factor.solve(&projected);
            for row in 0..3 {
                matrix[row][column] += correction[row];
            }
        }
    }
    for row in &mut matrix {
        for value in row {
            *value *= weight_normalization;
        }
    }
    symmetrize(&mut matrix);
    if matrix.iter().flatten().all(|value| value.is_finite()) {
        Ok(matrix)
    } else {
        Err(SparseBundleAdjustmentCovarianceError::NonFiniteResult)
    }
}

fn symmetrize<const N: usize>(matrix: &mut [[f64; N]; N]) {
    for row in 0..N {
        let (previous_rows, current_and_after) = matrix.split_at_mut(row);
        let current_row = &mut current_and_after[0];
        for (column, previous_row) in previous_rows.iter_mut().enumerate() {
            let symmetric = 0.5 * (current_row[column] + previous_row[row]);
            current_row[column] = symmetric;
            previous_row[row] = symmetric;
        }
    }
}
