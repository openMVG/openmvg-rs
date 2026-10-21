// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free solve over a reusable dynamic dense-Schur workspace.

use crate::numerics::manifold;
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::nonlinear::{LmRunState, StepRejection, final_stage_index, stages};
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::{NoiseModel, Options};

use super::super::RelativePose;
use super::ScaleGauge;
use super::schur::GaugeState;
#[cfg(feature = "unstable-internals")]
use super::{
    BundleAdjustmentPriors, FixedVariableBlocks, MultiViewBundleAdjustmentError,
    MultiViewBundleAdjustmentReport, MultiViewMeasurements,
    dense::{self, DenseBundleAdjustmentWorkspace},
    model::ObservationModel,
    schur, validation,
};

/// Refine fixed camera 0, all remaining camera poses, and all 3D points.
///
/// Measurements are point-major CSR tracks. `noise` contains one standard
/// deviation per observation (or one shared value). The workspace must be
/// allocated before the call; the solve itself allocates no memory.
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_dense(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    refine_ba_multi_view_dense_with_factors(
        poses,
        points,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        workspace,
        options,
        noise,
    )
}

/// Refine a multi-view problem with exact block fixes and unary priors.
///
/// Fixed blocks are removed from the normal equations. Priors are finite
/// Gaussian factors and therefore remain part of the objective and posterior
/// information. The solve allocates no memory.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_dense_with_factors(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentReport, MultiViewBundleAdjustmentError> {
    options.try_validate()?;
    if !(workspace.initial_damping.is_finite() && workspace.initial_damping > 0.0) {
        return Err(MultiViewBundleAdjustmentError::InitialDampingInvalid);
    }
    let model = ObservationModel::from(measurements);
    noise.try_validate(model.observation_count())?;
    validation::problem(poses, points, model, gauge, workspace)?;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    let dimension = schur::configure_layout_with_fixed(
        workspace,
        poses.len(),
        gauge,
        model.intrinsic_dimension(),
        fixed,
    );
    validation::projection_domain(model, poses, points)?;
    let scene_scale = scene_scale(poses, points, gauge);

    let (observation_cost, initial_rmse) =
        schur::metrics(model, poses, points, options.loss, noise);
    let initial_cost =
        observation_cost + schur::prior_cost(model, poses, points, priors).unwrap_or(f64::INFINITY);
    let fixed_baseline = match gauge {
        ScaleGauge::Free => None,
        ScaleGauge::FixedBaseline { camera } => Some(norm(&poses[camera].translation)),
        ScaleGauge::FixedTranslationComponent { .. } => None,
    };
    let fixed_translation_component = match gauge {
        ScaleGauge::FixedTranslationComponent { camera, component } => {
            Some(poses[camera].translation[component.index()])
        }
        ScaleGauge::Free | ScaleGauge::FixedBaseline { .. } => None,
    };
    let schedule = stages(options);
    let target_stage = final_stage_index(&schedule);
    let mut run = LmRunState::with_initial_damping(workspace.initial_damping);

    'optimization: for (stage_index, stage) in schedule.into_iter().enumerate() {
        for _ in 0..stage.max_iterations {
            run.begin_iteration();
            let gauge_state = GaugeState::new(gauge, poses, scene_scale);
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
                    loss: stage.loss,
                    noise,
                    damping: run.damping(),
                    prior_sqrt_scale: 1.0,
                },
                dimension,
                workspace,
            );
            if !assembly.valid {
                run.reject_failure(StepRejection::Linearization);
                continue;
            }
            let gradient_norm = assembly.scaled_gradient_inf_norm;
            let current_cost = assembly.cost;
            if stage_index == target_stage
                && run.gradient_converged(gradient_norm, current_cost, options.convergence)
            {
                break 'optimization;
            }
            if !dense::solve_camera_step(workspace, dimension) {
                run.reject_failure(StepRejection::StepSolve);
                continue;
            }
            limit_camera_step(
                &mut workspace.step[..dimension],
                &workspace.camera_offsets,
                &workspace.camera_dimensions,
                poses.len(),
                0.5,
            );
            retract_poses(
                poses,
                &mut workspace.candidate_poses[..poses.len()],
                gauge_state,
                &workspace.camera_offsets,
                &workspace.camera_dimensions,
                &workspace.step[..dimension],
                0,
            );
            let Some((point_step_rms, point_damping_norm, candidate_cost)) = candidate_points(
                model,
                poses,
                points,
                gauge_state,
                stage.loss,
                noise,
                run.damping(),
                dimension,
                workspace,
                fixed,
                priors,
            ) else {
                run.reject_failure(StepRejection::Candidate);
                continue;
            };
            let damped_reduction = dense_damped_model_reduction(
                assembly.point_model_reduction,
                &workspace.gradient[..dimension],
                &workspace.hessian[..dimension * dimension],
                &workspace.step[..dimension],
                dimension,
            );
            let predicted_reduction = undamped_model_reduction(
                damped_reduction,
                &workspace.diagonal[..dimension],
                &workspace.step[..dimension],
                run.damping(),
                point_damping_norm,
            );
            let reduction = current_cost - candidate_cost;
            if run.accept_gain(reduction, predicted_reduction) {
                poses.copy_from_slice(&workspace.candidate_poses[..poses.len()]);
                points.copy_from_slice(&workspace.candidate_points[..points.len()]);
                let camera_step_rms =
                    (squared_norm(&workspace.step[..dimension]) / dimension.max(1) as f64).sqrt();
                let step_norm =
                    (camera_step_rms * camera_step_rms + point_step_rms * point_step_rms).sqrt();
                let is_target = stage_index == target_stage;
                let candidate_convergence = run.record_accepted_step(
                    step_norm,
                    reduction,
                    current_cost,
                    is_target,
                    options.convergence,
                );
                if is_target && run.certify_convergence(candidate_convergence) {
                    break 'optimization;
                }
            }
        }
    }

    let expected_gauge_nullity = schur::expected_gauge_nullity(gauge, poses, points, fixed, priors);
    let (gradient_inf_norm, diagnostics, final_system_valid) = if workspace.final_diagnostics() {
        let final_assembly = schur::assemble_problem(
            schur::BaProblemView {
                model,
                poses,
                points,
                fixed,
                priors,
            },
            schur::LinearizationPolicy {
                gauge: GaugeState::new(gauge, poses, scene_scale),
                loss: options.loss,
                noise,
                damping: 0.0,
                prior_sqrt_scale: 1.0,
            },
            dimension,
            workspace,
        );
        let gradient_inf_norm = if final_assembly.valid {
            final_assembly.scaled_gradient_inf_norm
        } else {
            f64::INFINITY
        };
        let diagnostics = schur::diagnostics_with_expected_nullity(
            workspace,
            dimension,
            expected_gauge_nullity,
            &final_assembly,
        );
        (gradient_inf_norm, diagnostics, final_assembly.valid)
    } else {
        (
            f64::NAN,
            schur::unassessed_diagnostics(dimension, expected_gauge_nullity),
            true,
        )
    };
    let status = run.status(final_system_valid && diagnostics.is_well_constrained());

    // Objective metrics remain meaningful when an undamped Hessian is
    // singular. Keep numerical observability in `status` and `diagnostics`
    // without discarding the last accepted finite objective.
    let (observation_cost, final_rmse) = schur::metrics(model, poses, points, options.loss, noise);
    let final_cost =
        observation_cost + schur::prior_cost(model, poses, points, priors).unwrap_or(f64::INFINITY);

    Ok(MultiViewBundleAdjustmentReport {
        loss: options.loss,
        status,
        scale_gauge: gauge,
        normalization_scale: scene_scale,
        fixed_baseline,
        fixed_translation_component,
        pose_count: poses.len(),
        point_count: points.len(),
        observation_count: model.observation_count(),
        noninvertible_measurement_count: model.noninvertible_measurement_count(),
        iteration_count: run.iterations(),
        accepted_step_count: run.accepted_steps(),
        rejected_step_count: run.rejected_steps(),
        step_rejections: run.rejections(),
        initial_cost,
        final_cost,
        initial_rmse,
        final_rmse,
        gradient_inf_norm,
        step_norm: run.step_norm(),
        initial_damping: workspace.initial_damping,
        final_damping: run.damping(),
        last_gain_ratio: run.last_gain_ratio(),
        diagnostics,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
fn candidate_points(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: crate::numerics::refine::Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    dimension: usize,
    workspace: &mut DenseBundleAdjustmentWorkspace,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
) -> Option<(f64, f64, f64)> {
    let mut squared = 0.0;
    let mut point_damping_norm = 0.0;
    let mut candidate_cost = 0.0;
    let layout = schur::VariableLayout {
        offsets: &workspace.camera_offsets,
        pose_dimensions: &workspace.camera_pose_dimensions,
        intrinsic_offsets: &workspace.intrinsic_offsets,
        intrinsic_dimensions: &workspace.intrinsic_dimensions,
        pose_masks: &[],
        #[cfg(feature = "faer")]
        pose_variables: super::CameraPoseVariables::RotationAndTranslation,
    };
    let mut point_prior_index = 0;
    for (point_index, point) in points.iter().enumerate() {
        let point_prior = priors
            .points
            .get(point_prior_index)
            .filter(|prior| prior.point == point_index);
        if point_prior.is_some() {
            point_prior_index += 1;
        }
        if !fixed.points.is_empty() && fixed.points[point_index] {
            workspace.candidate_points[point_index] = *point;
            candidate_cost += schur::fixed_point_cost(
                model,
                &workspace.candidate_poses[..poses.len()],
                point,
                point_index,
                loss,
                noise,
            )?;
            continue;
        }
        let (step, damping_norm) = schur::point_step_with_prior_and_damping_norm(
            model,
            poses,
            point,
            point_index,
            gauge,
            loss,
            noise,
            damping,
            &workspace.step[..dimension],
            layout,
            &mut workspace.cross,
            &mut workspace.active_parameters,
            &mut workspace.parameter_marks,
            point_prior,
        )?;
        let evaluated = schur::candidate_point(
            model,
            poses,
            &workspace.candidate_poses[..poses.len()],
            point,
            std::array::from_fn(|coordinate| {
                point[coordinate] + gauge.scene_scale * step[coordinate]
            }),
            point_index,
            loss,
            noise,
            gauge.scene_scale,
        )?;
        workspace.candidate_points[point_index] = evaluated.position;
        candidate_cost += evaluated.cost;
        if evaluated.moved {
            point_damping_norm += damping_norm;
            squared += squared_norm(&step);
        }
    }
    candidate_cost += schur::prior_cost(
        model,
        &workspace.candidate_poses[..poses.len()],
        &workspace.candidate_points[..points.len()],
        priors,
    )?;
    Some((
        (squared / points.len() as f64).sqrt(),
        point_damping_norm,
        candidate_cost,
    ))
}

pub(super) fn retract_poses(
    poses: &[RelativePose],
    candidates: &mut [RelativePose],
    gauge: GaugeState,
    offsets: &[usize],
    dimensions: &[usize],
    step: &[f64],
    intrinsic_dimension: usize,
) {
    for camera in 0..poses.len() {
        let offset = offsets[camera];
        let pose_dimension = dimensions[camera] - intrinsic_dimension;
        if pose_dimension == 0 {
            candidates[camera] = poses[camera];
            continue;
        }
        let rotation_update =
            manifold::exp_so3(&[step[offset], step[offset + 1], step[offset + 2]]);
        let translation = if gauge.fixed_camera == Some(camera) && gauge.fixed_component.is_none() {
            let direction = manifold::retract_s2(
                &gauge.direction,
                &gauge.basis,
                &[step[offset + 3], step[offset + 4]],
            );
            direction.map(|value| gauge.baseline * value)
        } else if let Some(fixed) = gauge
            .fixed_component
            .filter(|_| gauge.fixed_camera == Some(camera))
        {
            let mut translation = poses[camera].translation;
            let mut local = 3;
            for (coordinate, value) in translation.iter_mut().enumerate() {
                if coordinate != fixed {
                    *value += gauge.scene_scale * step[offset + local];
                    local += 1;
                }
            }
            translation[fixed] = gauge.fixed_component_value;
            translation
        } else {
            debug_assert_eq!(pose_dimension, 6);
            std::array::from_fn(|coordinate| {
                poses[camera].translation[coordinate]
                    + gauge.scene_scale * step[offset + 3 + coordinate]
            })
        };
        candidates[camera] = RelativePose::new(
            manifold::mat_mul(&rotation_update, &poses[camera].rotation),
            translation,
        );
    }
}

/// Retract the requested camera coordinates without an implicit camera-zero lock.
#[cfg(feature = "faer")]
#[allow(clippy::too_many_arguments)]
pub(super) fn retract_poses_with_masks(
    poses: &[RelativePose],
    candidates: &mut [RelativePose],
    gauge: GaugeState,
    offsets: &[usize],
    dimensions: &[usize],
    masks: &[crate::PoseVariables],
    step: &[f64],
) {
    for (camera, &mask) in masks.iter().enumerate() {
        let dimension = dimensions[camera];
        if dimension == 0 {
            candidates[camera] = poses[camera];
            continue;
        }
        let offset = offsets[camera];
        let (rotation, rotation_update, translation_start) = if mask.rotation() {
            let update = manifold::exp_so3(&[step[offset], step[offset + 1], step[offset + 2]]);
            (
                manifold::mat_mul(&update, &poses[camera].rotation),
                Some(update),
                3,
            )
        } else {
            (poses[camera].rotation, None, 0)
        };
        let translation = if !mask.translation() {
            // t' = Exp(omega)t keeps C=-R^T t fixed under the same left update.
            match rotation_update {
                Some(update) => std::array::from_fn(|row| {
                    (0..3)
                        .map(|column| update[row][column] * poses[camera].translation[column])
                        .sum()
                }),
                None => poses[camera].translation,
            }
        } else if gauge.fixed_camera == Some(camera) && gauge.fixed_component.is_none() {
            let direction = manifold::retract_s2(
                &gauge.direction,
                &gauge.basis,
                &[
                    step[offset + translation_start],
                    step[offset + translation_start + 1],
                ],
            );
            direction.map(|value| gauge.baseline * value)
        } else if let Some(fixed) = gauge
            .fixed_component
            .filter(|_| gauge.fixed_camera == Some(camera))
        {
            let mut translation = poses[camera].translation;
            let mut local = translation_start;
            for (coordinate, value) in translation.iter_mut().enumerate() {
                if coordinate != fixed {
                    *value += gauge.scene_scale * step[offset + local];
                    local += 1;
                }
            }
            translation[fixed] = gauge.fixed_component_value;
            translation
        } else {
            std::array::from_fn(|coordinate| {
                poses[camera].translation[coordinate]
                    + gauge.scene_scale * step[offset + translation_start + coordinate]
            })
        };
        candidates[camera] = RelativePose::new(rotation, translation);
    }
}

#[cfg(feature = "faer")]
pub(super) fn retract_translations(
    poses: &[RelativePose],
    candidates: &mut [RelativePose],
    gauge: GaugeState,
    offsets: &[usize],
    dimensions: &[usize],
    step: &[f64],
) {
    for camera in 0..poses.len() {
        let offset = offsets[camera];
        let dimension = dimensions[camera];
        if dimension == 0 {
            candidates[camera] = poses[camera];
            continue;
        }
        let translation = if gauge.fixed_camera == Some(camera) && gauge.fixed_component.is_none() {
            debug_assert_eq!(dimension, 2);
            let direction = manifold::retract_s2(
                &gauge.direction,
                &gauge.basis,
                &[step[offset], step[offset + 1]],
            );
            direction.map(|value| gauge.baseline * value)
        } else if let Some(fixed) = gauge
            .fixed_component
            .filter(|_| gauge.fixed_camera == Some(camera))
        {
            debug_assert_eq!(dimension, 2);
            let mut translation = poses[camera].translation;
            let mut local = 0;
            for (coordinate, value) in translation.iter_mut().enumerate() {
                if coordinate != fixed {
                    *value += gauge.scene_scale * step[offset + local];
                    local += 1;
                }
            }
            translation[fixed] = gauge.fixed_component_value;
            translation
        } else {
            debug_assert_eq!(dimension, 3);
            std::array::from_fn(|coordinate| {
                poses[camera].translation[coordinate]
                    + gauge.scene_scale * step[offset + coordinate]
            })
        };
        candidates[camera] = RelativePose::new(poses[camera].rotation, translation);
    }
}

pub(super) fn norm(vector: &[f64]) -> f64 {
    squared_norm(vector).sqrt()
}

pub(super) fn squared_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

#[cfg(any(test, feature = "unstable-internals"))]
fn dense_damped_model_reduction(
    point_model_reduction: f64,
    gradient: &[f64],
    hessian: &[f64],
    step: &[f64],
    dimension: usize,
) -> f64 {
    let mut quadratic = 0.0;
    for row in 0..dimension {
        let product = hessian[row * dimension..(row + 1) * dimension]
            .iter()
            .zip(step)
            .map(|(value, step)| value * step)
            .sum::<f64>();
        quadratic += step[row] * product;
    }
    reduced_model_reduction(point_model_reduction, gradient, step, quadratic)
}

pub(super) fn reduced_model_reduction(
    point_model_reduction: f64,
    gradient: &[f64],
    step: &[f64],
    hessian_quadratic: f64,
) -> f64 {
    point_model_reduction
        - gradient
            .iter()
            .zip(step)
            .map(|(gradient, step)| gradient * step)
            .sum::<f64>()
        - 0.5 * hessian_quadratic
}

pub(super) fn undamped_model_reduction(
    damped_reduction: f64,
    diagonal: &[f64],
    step: &[f64],
    damping: f64,
    point_damping_norm: f64,
) -> f64 {
    let camera_damping_norm = diagonal
        .iter()
        .zip(step)
        .map(|(curvature, step)| curvature * step * step)
        .sum::<f64>();
    damped_reduction + 0.5 * damping * (camera_damping_norm + point_damping_norm)
}

pub(super) fn scene_scale(poses: &[RelativePose], points: &[[f64; 3]], gauge: ScaleGauge) -> f64 {
    if let ScaleGauge::FixedBaseline { camera } = gauge {
        return norm(&poses[camera].translation);
    }
    let point_rms =
        (points.iter().map(|point| squared_norm(point)).sum::<f64>() / points.len() as f64).sqrt();
    point_rms.max(f64::MIN_POSITIVE)
}

/// Conditioning length from observations only; inactive inventory is not geometry.
#[cfg(feature = "faer")]
pub(super) fn observed_scene_scale(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: crate::kernel::multi_view::model::ObservationModel<'_>,
    gauge: ScaleGauge,
) -> f64 {
    if let ScaleGauge::FixedBaseline { camera } = gauge {
        return norm(&poses[camera].translation);
    }
    let mut scale = 0.0_f64;
    let mut squared = 1.0_f64;
    let mut count = 0usize;
    for (point, track) in points.iter().zip(model.offsets().windows(2)) {
        if track[0] == track[1] {
            continue;
        }
        count += 1;
        for &coordinate in point {
            let magnitude = coordinate.abs();
            if magnitude == 0.0 {
                continue;
            }
            if scale < magnitude {
                let ratio = scale / magnitude;
                squared = 1.0 + squared * ratio * ratio;
                scale = magnitude;
            } else {
                let ratio = magnitude / scale;
                squared += ratio * ratio;
            }
        }
    }
    if scale == 0.0 || count == 0 {
        return f64::MIN_POSITIVE;
    }
    let rms = scale * (squared / count as f64).sqrt();
    // This is a coordinate-conditioning length, not a physical constraint.
    // If the Euclidean RMS exceeds f64 range, the largest finite coordinate
    // still gives a usable scaling without changing the requested objective.
    if rms.is_finite() {
        rms.max(f64::MIN_POSITIVE)
    } else {
        scale
    }
}

pub(super) fn limit_camera_step(
    step: &mut [f64],
    offsets: &[usize],
    dimensions: &[usize],
    camera_count: usize,
    maximum: f64,
) {
    let mut largest_block = 0.0f64;
    for camera in 0..camera_count {
        let offset = offsets[camera];
        if dimensions[camera] == 0 {
            continue;
        }
        let first_dimension = dimensions[camera].min(3);
        largest_block = largest_block.max(norm(&step[offset..offset + first_dimension]));
        largest_block = largest_block.max(norm(
            &step[offset + first_dimension..offset + dimensions[camera]],
        ));
    }
    if largest_block > maximum {
        let scale = maximum / largest_block;
        for value in step {
            *value *= scale;
        }
    }
}

#[cfg(feature = "faer")]
pub(super) fn limit_translation_step(
    step: &mut [f64],
    offsets: &[usize],
    dimensions: &[usize],
    camera_count: usize,
    maximum: f64,
) {
    let mut largest_block = 0.0f64;
    for camera in 0..camera_count {
        let offset = offsets[camera];
        let dimension = dimensions[camera];
        if dimension == 0 {
            continue;
        }
        largest_block = largest_block.max(norm(&step[offset..offset + dimension]));
    }
    if largest_block > maximum {
        let scale = maximum / largest_block;
        for value in step {
            *value *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_prediction_evaluates_the_step_that_is_actually_applied() {
        let hessian = [4.0, 1.0, 1.0, 3.0];
        let gradient = [2.0, -1.0];
        let step = [-0.2, 0.3];
        let point_reduction = 0.7;
        let quadratic = step[0] * (4.0 * step[0] + step[1]) + step[1] * (step[0] + 3.0 * step[1]);
        let expected =
            point_reduction - gradient[0] * step[0] - gradient[1] * step[1] - 0.5 * quadratic;
        let predicted =
            dense_damped_model_reduction(point_reduction, &gradient, &hessian, &step, 2);
        assert!((predicted - expected).abs() < 1e-15);
    }
    #[cfg(feature = "faer")]
    #[test]
    fn mixed_retraction_moves_camera_zero_and_preserves_held_pose_coordinates() {
        let rotation = manifold::exp_so3(&[0.12, -0.08, 0.03]);
        let poses = [RelativePose::new(rotation, [0.6, -0.2, 0.4]); 4];
        let mut candidates = poses;
        let masks = [
            crate::PoseVariables::FREE,
            crate::PoseVariables::FREE.with_rotation(false),
            crate::PoseVariables::FREE.with_translation(false),
            crate::PoseVariables::FIXED,
        ];
        let offsets = [0, 6, 9, 12];
        let dimensions = [6, 3, 3, 0];
        let step = [
            0.02, -0.03, 0.01, 0.04, 0.01, -0.02, 0.04, -0.03, 0.02, 0.03, 0.01, -0.02,
        ];
        retract_poses_with_masks(
            &poses,
            &mut candidates,
            GaugeState::new(ScaleGauge::Free, &poses, 1.0),
            &offsets,
            &dimensions,
            &masks,
            &step,
        );
        assert_ne!(candidates[0].rotation, poses[0].rotation);
        assert_ne!(candidates[0].translation, poses[0].translation);
        assert_eq!(candidates[1].rotation, poses[1].rotation);
        assert_ne!(candidates[1].translation, poses[1].translation);
        let center = |pose: RelativePose| -> [f64; 3] {
            std::array::from_fn(|axis| {
                -(0..3)
                    .map(|row| pose.rotation[row][axis] * pose.translation[row])
                    .sum::<f64>()
            })
        };
        for (before, after) in center(poses[2]).into_iter().zip(center(candidates[2])) {
            assert!((before - after).abs() < 8.0e-16);
        }
        assert_ne!(candidates[2].rotation, poses[2].rotation);
        assert_eq!(candidates[3], poses[3]);

        let mut camera_zero_step = [2.0, 0.0, 0.0];
        limit_camera_step(&mut camera_zero_step, &[0], &[3], 1, 0.5);
        assert_eq!(camera_zero_step, [0.5, 0.0, 0.0]);
        let mut sphere_step = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        limit_camera_step(&mut sphere_step, &[0, 6], &[6, 2], 2, 0.5);
        assert_eq!(&sphere_step[6..], &[0.5, 0.0]);

        let observations = [crate::kernel::multi_view::AngularObservation::new(
            0,
            [0.0, 0.0, 1.0],
        )];
        let measurements = crate::kernel::multi_view::MultiViewMeasurements::Angular {
            observations: &observations,
            point_offsets: &[0, 1, 1],
        };
        let model = crate::kernel::multi_view::model::ObservationModel::from(measurements);
        let inventory = [[3.0, 4.0, 0.0], [1.0e200, -1.0e200, 1.0e200]];
        assert_eq!(
            observed_scene_scale(&poses, &inventory, model, ScaleGauge::Free),
            5.0
        );
    }
}
