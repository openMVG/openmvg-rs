// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable numerical storage for a prepared BATA problem shape.

#[cfg(test)]
use std::mem::size_of;

use nalgebra::{Matrix3, Vector3};

use super::api::GlobalPositioningError;
use super::plan::GlobalPositioningPlan;

pub(super) const DENSE_FALLBACK_MAX_DIMENSION: usize = 1024;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CameraTerm {
    pub(super) diagonal: f64,
    pub(super) right_hand_side: Vector3<f64>,
    pub(super) weighted_degree: f64,
}

/// All mutable storage used by repeated solves of one prepared topology.
///
/// A workspace may be reused with any plan requiring the same buffer shapes.
/// Topology and measurements live entirely in the plan; the workspace owns only
/// reset-on-solve numerical scratch.
pub struct GlobalPositioningWorkspace {
    pub(super) cameras: Vec<Vector3<f64>>,
    pub(super) trial_cameras: Vec<Vector3<f64>>,
    pub(super) points: Vec<Vector3<f64>>,
    pub(super) trial_points: Vec<Vector3<f64>>,
    pub(super) bearing_scale: Vec<f64>,
    pub(super) trial_bearing_scale: Vec<f64>,
    pub(super) bearing_weight: Vec<f64>,
    pub(super) bearing_effective_weight: Vec<f64>,
    pub(super) bearing_dot: Vec<f64>,
    pub(super) point_weight: Vec<f64>,
    pub(super) point_mean_offset: Vec<Vector3<f64>>,
    pub(super) camera_terms: Vec<CameraTerm>,
    pub(super) direction_scale: Vec<f64>,
    pub(super) trial_direction_scale: Vec<f64>,
    pub(super) direction_weight: Vec<f64>,
    pub(super) direction_effective_weight: Vec<f64>,
    pub(super) direction_dot: Vec<f64>,
    pub(super) direction_init_weight: Vec<f64>,
    pub(super) depths: Vec<f64>,
    pub(super) pcg_rhs: Vec<Vector3<f64>>,
    pub(super) pcg_residual: Vec<Vector3<f64>>,
    pub(super) pcg_preconditioned: Vec<Vector3<f64>>,
    pub(super) pcg_search: Vec<Vector3<f64>>,
    pub(super) pcg_applied: Vec<Vector3<f64>>,
    pub(super) direction_init_diagonal: Vec<Vector3<f64>>,
    pub(super) direction_init_degree: Vec<f64>,
    pub(super) coupling_values: Vec<f64>,
    pub(super) reduced_index: Vec<usize>,
    /// Union-find parents over cameras then points for the active-connectivity audit.
    pub(super) connectivity_parent: Vec<usize>,
    pub(super) dense_matrix: Vec<f64>,
    pub(super) dense_rhs: Vec<f64>,
    /// Polish: `w J^T J` of each bearing then each direction.
    pub(super) polish_hessian: Vec<Matrix3<f64>>,
    /// Polish: undamped and damped-inverse 3x3 point blocks, and point gradients.
    pub(super) polish_point_hessian: Vec<Matrix3<f64>>,
    pub(super) polish_point_inverse: Vec<Matrix3<f64>>,
    pub(super) polish_point_gradient: Vec<Vector3<f64>>,
    /// Polish: camera 3x3 diagonal blocks (undamped), their damped Schur preconditioner inverses,
    /// the camera gradient, and the camera step.
    pub(super) polish_camera_block: Vec<Matrix3<f64>>,
    pub(super) polish_preconditioner: Vec<Matrix3<f64>>,
    pub(super) polish_camera_gradient: Vec<Vector3<f64>>,
    pub(super) polish_step: Vec<Vector3<f64>>,
    /// Sparse Cholesky of the polish camera system, analyzed at the first polish.
    #[cfg(feature = "faer")]
    pub(super) polish_direct: Option<Box<super::direct::BlockCholesky>>,
    #[cfg(feature = "faer")]
    pub(super) polish_direct_unavailable: bool,
    /// Sparse Cholesky of the alternation's scalar camera system, analyzed at the first
    /// alternation iteration.
    #[cfg(feature = "faer")]
    pub(super) alternation_direct: Option<Box<super::direct::ScalarCholesky>>,
    #[cfg(feature = "faer")]
    pub(super) alternation_direct_unavailable: bool,
}

impl std::fmt::Debug for GlobalPositioningWorkspace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GlobalPositioningWorkspace")
            .field("camera_count", &self.cameras.len())
            .field("point_count", &self.points.len())
            .field("bearing_count", &self.bearing_weight.len())
            .field("direction_count", &self.direction_weight.len())
            .finish_non_exhaustive()
    }
}

impl GlobalPositioningWorkspace {
    pub(super) fn try_new(plan: &GlobalPositioningPlan) -> Result<Self, GlobalPositioningError> {
        let cameras = plan.camera_count;
        let points = plan.point_count;
        let bearings = plan.bearings.len();
        let directions = plan.directions.len();
        let reduced = cameras.saturating_sub(1);
        let dense_len = if reduced <= DENSE_FALLBACK_MAX_DIMENSION {
            reduced
                .checked_mul(reduced)
                .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?
        } else {
            0
        };
        let depth_capacity = bearings
            .checked_add(directions)
            .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?;
        let mut depths = reserved(depth_capacity)?;
        depths.clear();
        Ok(Self {
            cameras: filled(cameras, Vector3::zeros())?,
            trial_cameras: filled(cameras, Vector3::zeros())?,
            points: filled(points, Vector3::zeros())?,
            trial_points: filled(points, Vector3::zeros())?,
            bearing_scale: filled(bearings, 1.0)?,
            trial_bearing_scale: filled(bearings, 1.0)?,
            bearing_weight: filled(bearings, 1.0)?,
            bearing_effective_weight: filled(bearings, 0.0)?,
            bearing_dot: filled(bearings, 0.0)?,
            point_weight: filled(points, 0.0)?,
            point_mean_offset: filled(points, Vector3::zeros())?,
            camera_terms: filled(cameras, CameraTerm::default())?,
            direction_scale: filled(directions, 1.0)?,
            trial_direction_scale: filled(directions, 1.0)?,
            direction_weight: filled(directions, 1.0)?,
            direction_effective_weight: filled(directions, 0.0)?,
            direction_dot: filled(directions, 0.0)?,
            direction_init_weight: filled(directions, 0.0)?,
            depths,
            pcg_rhs: filled(cameras, Vector3::zeros())?,
            pcg_residual: filled(cameras, Vector3::zeros())?,
            pcg_preconditioned: filled(cameras, Vector3::zeros())?,
            pcg_search: filled(cameras, Vector3::zeros())?,
            pcg_applied: filled(cameras, Vector3::zeros())?,
            direction_init_diagonal: filled(cameras, Vector3::zeros())?,
            direction_init_degree: filled(cameras, 0.0)?,
            coupling_values: filled(plan.coupling_endpoints.len(), 0.0)?,
            reduced_index: filled(cameras, usize::MAX)?,
            connectivity_parent: filled(
                cameras
                    .checked_add(points)
                    .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?,
                0,
            )?,
            dense_matrix: filled(dense_len, 0.0)?,
            dense_rhs: filled(
                reduced
                    .checked_mul(3)
                    .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?,
                0.0,
            )?,
            polish_hessian: filled(depth_capacity, Matrix3::zeros())?,
            polish_point_hessian: filled(points, Matrix3::zeros())?,
            polish_point_inverse: filled(points, Matrix3::zeros())?,
            polish_point_gradient: filled(points, Vector3::zeros())?,
            polish_camera_block: filled(cameras, Matrix3::zeros())?,
            polish_preconditioner: filled(cameras, Matrix3::zeros())?,
            polish_camera_gradient: filled(cameras, Vector3::zeros())?,
            polish_step: filled(cameras, Vector3::zeros())?,
            #[cfg(feature = "faer")]
            polish_direct: None,
            #[cfg(feature = "faer")]
            polish_direct_unavailable: false,
            #[cfg(feature = "faer")]
            alternation_direct: None,
            #[cfg(feature = "faer")]
            alternation_direct_unavailable: false,
        })
    }

    /// Directly measurable bytes retained by owned numerical buffers.
    #[must_use]
    #[cfg(test)]
    pub fn retained_capacity_bytes(&self) -> usize {
        let vectors = self.cameras.capacity()
            + self.trial_cameras.capacity()
            + self.points.capacity()
            + self.trial_points.capacity()
            + self.point_mean_offset.capacity()
            + self.pcg_rhs.capacity()
            + self.pcg_residual.capacity()
            + self.pcg_preconditioned.capacity()
            + self.pcg_search.capacity()
            + self.pcg_applied.capacity()
            + self.direction_init_diagonal.capacity();
        let scalars = self.bearing_scale.capacity()
            + self.trial_bearing_scale.capacity()
            + self.bearing_weight.capacity()
            + self.bearing_effective_weight.capacity()
            + self.bearing_dot.capacity()
            + self.point_weight.capacity()
            + self.direction_scale.capacity()
            + self.trial_direction_scale.capacity()
            + self.direction_weight.capacity()
            + self.direction_effective_weight.capacity()
            + self.direction_dot.capacity()
            + self.direction_init_weight.capacity()
            + self.depths.capacity()
            + self.dense_matrix.capacity()
            + self.dense_rhs.capacity();
        let polish_matrices = self.polish_hessian.capacity()
            + self.polish_point_hessian.capacity()
            + self.polish_point_inverse.capacity()
            + self.polish_camera_block.capacity()
            + self.polish_preconditioner.capacity();
        let polish_vectors = self.polish_point_gradient.capacity()
            + self.polish_camera_gradient.capacity()
            + self.polish_step.capacity();
        polish_matrices * size_of::<Matrix3<f64>>()
            + polish_vectors * size_of::<Vector3<f64>>()
            + vectors * size_of::<Vector3<f64>>()
            + scalars * size_of::<f64>()
            + self.camera_terms.capacity() * size_of::<CameraTerm>()
            + self.reduced_index.capacity() * size_of::<usize>()
            + self.connectivity_parent.capacity() * size_of::<usize>()
            + self.direction_init_degree.capacity() * size_of::<f64>()
            + self.coupling_values.capacity() * size_of::<f64>()
    }

    pub(super) fn is_compatible(&self, plan: &GlobalPositioningPlan) -> bool {
        self.cameras.len() == plan.camera_count
            && self.points.len() == plan.point_count
            && self.bearing_scale.len() == plan.bearings.len()
            && self.direction_scale.len() == plan.directions.len()
            && self.coupling_values.len() == plan.coupling_endpoints.len()
    }
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, GlobalPositioningError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| GlobalPositioningError::WorkspaceAllocationFailed)?;
    Ok(values)
}

fn filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, GlobalPositioningError> {
    let mut values = reserved(length)?;
    values.resize(length, value);
    Ok(values)
}
