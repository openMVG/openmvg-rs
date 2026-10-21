// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cost metrics, gauge nullity, and observability diagnostics.

#[cfg(any(test, feature = "unstable-internals"))]
use super::assembly::Assembly;
use super::{multiply_matrix_vector, norm, squared_vector_norm};
use crate::kernel::RelativePose;
#[cfg(any(test, feature = "unstable-internals"))]
use crate::kernel::multi_view::MultiViewBundleAdjustmentDiagnostics;
#[cfg(any(test, feature = "unstable-internals"))]
use crate::kernel::multi_view::dense::{self, DenseBundleAdjustmentWorkspace};
use crate::kernel::multi_view::model::{ObservationModel, transform};
use crate::kernel::multi_view::{BundleAdjustmentPriors, FixedVariableBlocks, ScaleGauge};
use crate::numerics::refine::{Loss, NoiseModel};

pub(in crate::kernel::multi_view) fn metrics(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    loss: Loss,
    noise: NoiseModel<'_>,
) -> (f64, f64) {
    let mut total = 0.0;
    let mut squared = 0.0;
    for (point_index, point) in points.iter().enumerate() {
        let track = model.offsets()[point_index]..model.offsets()[point_index + 1];
        for observation in track {
            let camera = model.camera(observation);
            let camera_point = transform(&poses[camera], point);
            let Some(projection) = model.project(observation, &camera_point) else {
                return (f64::INFINITY, f64::INFINITY);
            };
            let residual_norm = norm(&projection.residual);
            total += loss.cost(residual_norm * noise.inverse_standard_deviation(observation));
            squared += residual_norm * residual_norm;
        }
    }
    (
        total,
        (squared / (2 * model.observation_count()) as f64).sqrt(),
    )
}

#[cfg(any(test, feature = "unstable-internals"))]
pub(in crate::kernel::multi_view) fn diagnostics_with_expected_nullity(
    workspace: &mut DenseBundleAdjustmentWorkspace,
    dimension: usize,
    expected_gauge_nullity: usize,
    assembly: &Assembly,
) -> MultiViewBundleAdjustmentDiagnostics {
    let assessed = workspace.exact_camera_rank_diagnostics && assembly.valid;
    let (rank, rcond) = if assessed {
        dense::spectral_diagnostics(workspace, dimension).unwrap_or((0, 0.0))
    } else {
        (0, 0.0)
    };
    MultiViewBundleAdjustmentDiagnostics {
        pose_dimension: dimension,
        pose_rank_assessed: assessed,
        pose_rank: rank,
        expected_gauge_nullity,
        observable_camera_hessian_reciprocal_condition_number: rcond,
        // Point-block conditioning is accumulated even when the assembled
        // camera system is invalid; those weak blocks are often the reason it
        // failed. Keep that evidence assessed instead of masking degeneracy.
        point_rank_assessed: true,
        weak_point_count: assembly.weak_point_count,
        minimum_point_hessian_reciprocal_condition_number: assembly.minimum_point_rcond,
    }
}

#[cfg(any(test, feature = "unstable-internals"))]
pub(in crate::kernel::multi_view) const fn unassessed_diagnostics(
    camera_dimension: usize,
    expected_gauge_nullity: usize,
) -> MultiViewBundleAdjustmentDiagnostics {
    MultiViewBundleAdjustmentDiagnostics {
        pose_dimension: camera_dimension,
        pose_rank_assessed: false,
        pose_rank: 0,
        expected_gauge_nullity,
        observable_camera_hessian_reciprocal_condition_number: 0.0,
        point_rank_assessed: false,
        weak_point_count: 0,
        minimum_point_hessian_reciprocal_condition_number: 0.0,
    }
}

pub(in crate::kernel::multi_view) fn expected_gauge_nullity(
    gauge: ScaleGauge,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
) -> usize {
    if !matches!(gauge, ScaleGauge::Free) {
        return 0;
    }
    let fixed_pose_constrains_scale = fixed
        .camera_poses
        .iter()
        .enumerate()
        .skip(1)
        .any(|(camera, &fixed)| fixed && squared_vector_norm(&poses[camera].translation) > 0.0);
    let fixed_point_constrains_scale = fixed
        .points
        .iter()
        .enumerate()
        .any(|(point, &fixed)| fixed && squared_vector_norm(&points[point]) > 0.0);
    let pose_prior_constrains_scale = priors.poses.iter().any(|prior| {
        let Some(pose) = poses.get(prior.camera) else {
            return false;
        };
        let direction = [
            0.0,
            0.0,
            0.0,
            pose.translation[0],
            pose.translation[1],
            pose.translation[2],
        ];
        squared_vector_norm(&multiply_matrix_vector(&prior.sqrt_information, &direction)) > 0.0
    });
    let point_prior_constrains_scale = priors.points.iter().any(|prior| {
        points.get(prior.point).is_some_and(|point| {
            squared_vector_norm(&multiply_matrix_vector(&prior.sqrt_information, point)) > 0.0
        })
    });
    usize::from(
        !(fixed_pose_constrains_scale
            || fixed_point_constrains_scale
            || pose_prior_constrains_scale
            || point_prior_constrains_scale),
    )
}
