// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Point and camera prior terms of the normal equations.

use super::assembly::{Assembly, HessianAccumulator, PointLinearization};
use super::layout::{GaugeState, VariableLayout};
use super::{dot3, multiply_matrix_vector};
use crate::kernel::RelativePose;
use crate::kernel::multi_view::BundleAdjustmentPriors;
use crate::kernel::multi_view::PointPrior;
use crate::kernel::multi_view::model::ObservationModel;
use crate::numerics::manifold;

pub(in crate::kernel::multi_view) fn accumulate_point_prior(
    local: &mut PointLinearization,
    point: &[f64; 3],
    prior: &PointPrior,
    scene_scale: f64,
    prior_sqrt_scale: f64,
) {
    let residual = std::array::from_fn(|index| point[index] - prior.mean[index]);
    let weighted_residual = multiply_matrix_vector(&prior.sqrt_information, &residual)
        .map(|value| value * prior_sqrt_scale);
    let weighted_jacobian: [[f64; 3]; 3] = prior
        .sqrt_information
        .map(|row| row.map(|value| value * scene_scale * prior_sqrt_scale));
    local.cost += 0.5 * dot3(&weighted_residual, &weighted_residual);
    for row in 0..3 {
        local.gradient[row] += (0..3)
            .map(|residual| weighted_jacobian[residual][row] * weighted_residual[residual])
            .sum::<f64>();
        for column in 0..3 {
            local.hessian[row][column] += (0..3)
                .map(|residual| {
                    weighted_jacobian[residual][row] * weighted_jacobian[residual][column]
                })
                .sum::<f64>();
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn accumulate_camera_priors<H: HessianAccumulator>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    gauge: GaugeState,
    layout: VariableLayout<'_>,
    priors: BundleAdjustmentPriors<'_>,
    prior_sqrt_scale: f64,
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    assembly: &mut Assembly,
) {
    accumulate_camera_priors_impl::<false, H>(
        model,
        poses,
        gauge,
        layout,
        priors,
        prior_sqrt_scale,
        hessian,
        gradient,
        full_gradient,
        diagonal,
        assembly,
    );
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn accumulate_camera_priors_translation_only<
    H: HessianAccumulator,
>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    gauge: GaugeState,
    layout: VariableLayout<'_>,
    priors: BundleAdjustmentPriors<'_>,
    prior_sqrt_scale: f64,
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    assembly: &mut Assembly,
) {
    accumulate_camera_priors_impl::<true, H>(
        model,
        poses,
        gauge,
        layout,
        priors,
        prior_sqrt_scale,
        hessian,
        gradient,
        full_gradient,
        diagonal,
        assembly,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_camera_priors_impl<const TRANSLATION_ONLY: bool, H: HessianAccumulator>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    gauge: GaugeState,
    layout: VariableLayout<'_>,
    priors: BundleAdjustmentPriors<'_>,
    prior_sqrt_scale: f64,
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    assembly: &mut Assembly,
) {
    for prior in priors.poses {
        let camera = prior.camera;
        let dimension = layout.pose_dimensions[camera];
        let residual = pose_prior_residual(&poses[camera], &prior.mean);
        let jacobian = if TRANSLATION_ONLY {
            pose_prior_translation_jacobian(camera, gauge, dimension)
        } else {
            pose_prior_jacobian(&residual, camera, gauge, dimension)
        };
        let weighted_residual = multiply_matrix_vector(&prior.sqrt_information, &residual)
            .map(|value| value * prior_sqrt_scale);
        let weighted_jacobian: [[f64; 6]; 6] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..6)
                    .map(|inner| prior.sqrt_information[row][inner] * jacobian[inner][column])
                    .sum::<f64>()
                    * prior_sqrt_scale
            })
        });
        assembly.cost += 0.5
            * weighted_residual
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
        accumulate_prior_block(
            layout.offsets[camera],
            dimension,
            &weighted_residual,
            &weighted_jacobian,
            hessian,
            gradient,
            full_gradient,
            diagonal,
        );
    }
    for prior in priors.intrinsics {
        let Some(camera) = model.intrinsic_value(prior.intrinsic) else {
            assembly.valid = false;
            continue;
        };
        let Some((offset, dimension)) = layout.intrinsic(prior.intrinsic) else {
            assembly.valid = false;
            continue;
        };
        let residual = [
            (camera.focal / prior.focal).ln(),
            camera.k1 - prior.k1,
            camera.k2 - prior.k2,
        ];
        let weighted_residual = multiply_matrix_vector(&prior.sqrt_information, &residual)
            .map(|value| value * prior_sqrt_scale);
        let weighted_jacobian = prior
            .sqrt_information
            .map(|row| row.map(|value| value * prior_sqrt_scale));
        assembly.cost += 0.5 * dot3(&weighted_residual, &weighted_residual);
        accumulate_prior_block(
            offset,
            dimension,
            &weighted_residual,
            &weighted_jacobian,
            hessian,
            gradient,
            full_gradient,
            diagonal,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_prior_block<const R: usize, const C: usize, H: HessianAccumulator>(
    offset: usize,
    dimension: usize,
    residual: &[f64; R],
    jacobian: &[[f64; C]; R],
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
) {
    for local_row in 0..dimension {
        let row = offset + local_row;
        let value = (0..R)
            .map(|residual_index| jacobian[residual_index][local_row] * residual[residual_index])
            .sum::<f64>();
        gradient[row] += value;
        full_gradient[row] += value;
        for local_column in 0..=local_row {
            let curvature = (0..R)
                .map(|residual_index| {
                    jacobian[residual_index][local_row] * jacobian[residual_index][local_column]
                })
                .sum::<f64>();
            hessian.add_symmetric(row, offset + local_column, curvature);
            if local_row == local_column {
                diagonal[row] += curvature;
            }
        }
    }
}

pub(super) fn pose_prior_residual(pose: &RelativePose, mean: &RelativePose) -> [f64; 6] {
    let mean_transpose: [[f64; 3]; 3] =
        std::array::from_fn(|row| std::array::from_fn(|column| mean.rotation[column][row]));
    let rotation_error = manifold::mat_mul(&pose.rotation, &mean_transpose);
    let rotation = manifold::log_so3(&rotation_error);
    [
        rotation[0],
        rotation[1],
        rotation[2],
        pose.translation[0] - mean.translation[0],
        pose.translation[1] - mean.translation[1],
        pose.translation[2] - mean.translation[2],
    ]
}

pub(super) fn pose_prior_jacobian(
    residual: &[f64; 6],
    camera: usize,
    gauge: GaugeState,
    dimension: usize,
) -> [[f64; 6]; 6] {
    let rotation = [residual[0], residual[1], residual[2]];
    let rotation_jacobian = manifold::so3_left_jacobian_inverse(&rotation);
    let mut jacobian = [[0.0; 6]; 6];
    for row in 0..3 {
        jacobian[row][..3].copy_from_slice(&rotation_jacobian[row]);
    }
    if dimension == 5 && gauge.fixed_component.is_none() {
        for coordinate in 0..3 {
            jacobian[3 + coordinate][3] = gauge.baseline * gauge.basis[0][coordinate];
            jacobian[3 + coordinate][4] = gauge.baseline * gauge.basis[1][coordinate];
        }
    } else if dimension == 5 && gauge.fixed_camera == Some(camera) {
        if let Some(fixed) = gauge.fixed_component {
            let mut local = 3;
            for coordinate in 0..3 {
                if coordinate != fixed {
                    jacobian[3 + coordinate][local] = gauge.scene_scale;
                    local += 1;
                }
            }
        }
    } else {
        for coordinate in 0..3 {
            jacobian[3 + coordinate][3 + coordinate] = gauge.scene_scale;
        }
    }
    jacobian
}

fn pose_prior_translation_jacobian(
    camera: usize,
    gauge: GaugeState,
    dimension: usize,
) -> [[f64; 6]; 6] {
    let mut jacobian = [[0.0; 6]; 6];
    if dimension == 0 {
        return jacobian;
    }
    if dimension == 2 && gauge.fixed_camera == Some(camera) {
        if let Some(fixed) = gauge.fixed_component {
            let mut local = 0;
            for coordinate in 0..3 {
                if coordinate != fixed {
                    jacobian[3 + coordinate][local] = gauge.scene_scale;
                    local += 1;
                }
            }
        } else {
            for coordinate in 0..3 {
                jacobian[3 + coordinate][0] = gauge.baseline * gauge.basis[0][coordinate];
                jacobian[3 + coordinate][1] = gauge.baseline * gauge.basis[1][coordinate];
            }
        }
    } else {
        debug_assert_eq!(dimension, 3);
        for coordinate in 0..3 {
            jacobian[3 + coordinate][coordinate] = gauge.scene_scale;
        }
    }
    jacobian
}

pub(in crate::kernel::multi_view) fn prior_cost(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    priors: BundleAdjustmentPriors<'_>,
) -> Option<f64> {
    let mut cost = 0.0;
    for prior in priors.poses {
        let residual = pose_prior_residual(poses.get(prior.camera)?, &prior.mean);
        let weighted = multiply_matrix_vector(&prior.sqrt_information, &residual);
        cost += 0.5 * weighted.iter().map(|value| value * value).sum::<f64>();
    }
    for prior in priors.points {
        let point = points.get(prior.point)?;
        let residual = std::array::from_fn(|index| point[index] - prior.mean[index]);
        let weighted = multiply_matrix_vector(&prior.sqrt_information, &residual);
        cost += 0.5 * dot3(&weighted, &weighted);
    }
    for prior in priors.intrinsics {
        let camera = model.intrinsic_value(prior.intrinsic)?;
        let residual = [
            (camera.focal / prior.focal).ln(),
            camera.k1 - prior.k1,
            camera.k2 - prior.k2,
        ];
        let weighted = multiply_matrix_vector(&prior.sqrt_information, &residual);
        cost += 0.5 * dot3(&weighted, &weighted);
    }
    cost.is_finite().then_some(cost)
}
