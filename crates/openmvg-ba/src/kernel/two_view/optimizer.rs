// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free LM control for the dense-Schur two-view problem.

use crate::numerics::manifold;

use super::residual::{AngularModel, MeasurementModel, PixelModel, transform};
use super::schur::{
    self, POSE_DIMENSION, ReducedSystem, linearize_point, point_step, point_step_with_damping_norm,
};
use super::{BundleAdjustmentError, BundleAdjustmentReport, RelativePose, TwoViewMeasurements};
use crate::numerics::refine::nonlinear::{LmRunState, StepRejection, final_stage_index, stages};
use crate::numerics::refine::solver::solve_positive_definite;
use crate::numerics::refine::{Loss, NoiseModel, Options};

const MINIMUM_POINTS: usize = 5;

/// Refine one calibrated relative pose and its corresponding 3D points.
///
/// The points are updated in place only after accepted LM steps. `noise`
/// contains one standard deviation per correspondence (or one shared value),
/// expressed in radians for angular measurements and pixels for pixel
/// measurements. The function itself performs no allocation.
pub fn refine_ba_2view(
    initial_pose: &RelativePose,
    points: &mut [[f64; 3]],
    measurements: TwoViewMeasurements<'_>,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<BundleAdjustmentReport, BundleAdjustmentError> {
    refine_dispatch(initial_pose, points, measurements, options, noise)
}

fn refine_dispatch(
    initial_pose: &RelativePose,
    points: &mut [[f64; 3]],
    measurements: TwoViewMeasurements<'_>,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<BundleAdjustmentReport, BundleAdjustmentError> {
    options.try_validate()?;
    noise.try_validate(points.len())?;
    validate_pose(initial_pose)?;
    validate_points(points)?;
    if points.len() < MINIMUM_POINTS {
        return Err(BundleAdjustmentError::InsufficientPoints {
            required: MINIMUM_POINTS,
            actual: points.len(),
        });
    }

    match measurements {
        TwoViewMeasurements::Angular {
            bearings1,
            bearings2,
        } => refine_impl(
            *initial_pose,
            points,
            AngularModel {
                bearings1,
                bearings2,
            },
            options,
            noise,
        ),
        TwoViewMeasurements::Pixel {
            observations1,
            observations2,
            cameras,
        } => refine_impl(
            *initial_pose,
            points,
            PixelModel {
                observations1,
                observations2,
                cameras,
            },
            options,
            noise,
        ),
    }
}

fn refine_impl<M: MeasurementModel>(
    mut pose: RelativePose,
    points: &mut [[f64; 3]],
    model: M,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<BundleAdjustmentReport, BundleAdjustmentError> {
    model.validate(points.len())?;
    validate_projection_domain(model, &pose, points)?;
    let baseline = norm(&pose.translation);
    let (initial_cost, initial_rmse) = schur::metrics(model, &pose, points, options.loss, noise);
    let schedule = stages(options);
    let target_stage = final_stage_index(&schedule);
    let mut run = LmRunState::new();

    'optimization: for (stage_index, stage) in schedule.into_iter().enumerate() {
        for _ in 0..stage.max_iterations {
            run.begin_iteration();
            let trial_damping = run.damping();
            let translation_direction = pose.translation.map(|value| value / baseline);
            let translation_basis = manifold::s2_tangent_basis(&translation_direction);
            let Some(system) = reduced_system(
                model,
                &pose,
                &translation_basis,
                baseline,
                points,
                stage.loss,
                noise,
                trial_damping,
            )
            .filter(ReducedSystem::is_valid) else {
                run.reject_failure(StepRejection::Linearization);
                continue;
            };
            let gradient_norm = system.scaled_gradient_inf_norm();
            let current_cost = system.cost;
            if stage_index == target_stage
                && run.gradient_converged(gradient_norm, current_cost, options.convergence)
            {
                break 'optimization;
            }
            let Some(raw_step) = solve_positive_definite(&system.hessian, &system.gradient) else {
                run.reject_failure(StepRejection::StepSolve);
                continue;
            };
            let mut pose_step = raw_step.map(|value| -value);
            limit_pose_step(&mut pose_step, 0.5);
            let candidate_pose = retract_pose(
                &pose,
                &translation_direction,
                &translation_basis,
                baseline,
                &pose_step,
            );
            let Some((candidate_cost, point_step_rms, point_damping_norm)) = candidate_cost(
                model,
                &pose,
                &candidate_pose,
                &translation_basis,
                baseline,
                points,
                stage.loss,
                noise,
                trial_damping,
                &pose_step,
            ) else {
                run.reject_failure(StepRejection::Candidate);
                continue;
            };

            let reduction = current_cost - candidate_cost;
            let predicted_reduction =
                system.predicted_reduction(&pose_step, trial_damping, point_damping_norm);
            if run.accept_gain(reduction, predicted_reduction) {
                commit_points(
                    model,
                    &pose,
                    &translation_basis,
                    baseline,
                    points,
                    stage.loss,
                    noise,
                    trial_damping,
                    &pose_step,
                );
                pose = candidate_pose;
                let pose_step_rms = (squared_norm(&pose_step) / POSE_DIMENSION as f64).sqrt();
                let step_norm =
                    (pose_step_rms * pose_step_rms + point_step_rms * point_step_rms).sqrt();
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

    let translation_direction = pose.translation.map(|value| value / baseline);
    let translation_basis = manifold::s2_tangent_basis(&translation_direction);
    let final_system = reduced_system(
        model,
        &pose,
        &translation_basis,
        baseline,
        points,
        options.loss,
        noise,
        0.0,
    );
    let diagnostics = final_system.as_ref().map_or(
        super::BundleAdjustmentDiagnostics {
            pose_dimension: POSE_DIMENSION,
            pose_rank: 0,
            pose_hessian_reciprocal_condition_number: 0.0,
            weak_point_count: points.len(),
            minimum_point_hessian_reciprocal_condition_number: 0.0,
        },
        ReducedSystem::diagnostics,
    );
    let status = run.status(diagnostics.is_well_constrained());
    let gradient_inf_norm = final_system
        .as_ref()
        .filter(|system| system.is_valid())
        .map_or(f64::INFINITY, ReducedSystem::scaled_gradient_inf_norm);

    let (final_cost, final_rmse) = final_system.as_ref().map_or_else(
        || schur::metrics(model, &pose, points, options.loss, noise),
        |system| {
            (
                system.cost,
                (system.squared_residual / (4 * points.len()) as f64).sqrt(),
            )
        },
    );

    Ok(BundleAdjustmentReport {
        pose,
        loss: options.loss,
        status,
        point_count: points.len(),
        fixed_baseline: baseline,
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
        final_damping: run.damping(),
        last_gain_ratio: run.last_gain_ratio(),
        diagnostics,
    })
}

#[allow(clippy::too_many_arguments)]
fn reduced_system<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    translation_basis: &[[f64; 3]; 2],
    baseline: f64,
    points: &[[f64; 3]],
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
) -> Option<ReducedSystem> {
    let mut system = ReducedSystem::new();
    for (index, point) in points.iter().enumerate() {
        let local = linearize_point(
            model,
            pose,
            translation_basis,
            baseline,
            point,
            index,
            loss,
            noise,
        )?;
        system.add(&local, damping);
    }
    system.finish_damping(damping);
    Some(system)
}

#[allow(clippy::too_many_arguments)]
fn candidate_cost<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    candidate_pose: &RelativePose,
    translation_basis: &[[f64; 3]; 2],
    baseline: f64,
    points: &[[f64; 3]],
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    pose_step: &[f64; POSE_DIMENSION],
) -> Option<(f64, f64, f64)> {
    let mut total_cost = 0.0;
    let mut point_step_squared = 0.0;
    let mut point_damping_norm = 0.0;
    for (index, point) in points.iter().enumerate() {
        let local = linearize_point(
            model,
            pose,
            translation_basis,
            baseline,
            point,
            index,
            loss,
            noise,
        )?;
        let (step, damping_norm) = point_step_with_damping_norm(&local, damping, pose_step)?;
        let candidate =
            std::array::from_fn(|coordinate| point[coordinate] + baseline * step[coordinate]);
        total_cost += point_cost(model, candidate_pose, &candidate, index, loss, noise)?;
        point_step_squared += squared_norm(&step);
        point_damping_norm += damping_norm;
    }
    Some((
        total_cost,
        (point_step_squared / points.len() as f64).sqrt(),
        point_damping_norm,
    ))
}

#[allow(clippy::too_many_arguments)]
fn commit_points<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    translation_basis: &[[f64; 3]; 2],
    baseline: f64,
    points: &mut [[f64; 3]],
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    pose_step: &[f64; POSE_DIMENSION],
) {
    for (index, point) in points.iter_mut().enumerate() {
        let local = linearize_point(
            model,
            pose,
            translation_basis,
            baseline,
            point,
            index,
            loss,
            noise,
        );
        debug_assert!(local.is_some());
        let step = local.and_then(|local| point_step(&local, damping, pose_step));
        debug_assert!(step.is_some());
        if let Some(step) = step {
            for coordinate in 0..3 {
                point[coordinate] += baseline * step[coordinate];
            }
        }
    }
}

fn point_cost<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    point: &[f64; 3],
    index: usize,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Option<f64> {
    let transformed = transform(&pose.rotation, &pose.translation, point);
    let mut total = 0.0;
    for (view, camera_point) in [(0, point), (1, &transformed)] {
        let projection = model.project(view, index, camera_point)?;
        let norm = projection
            .residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        total += loss.cost(norm * noise.inverse_standard_deviation(index));
    }
    Some(total)
}

fn retract_pose(
    pose: &RelativePose,
    translation_direction: &[f64; 3],
    translation_basis: &[[f64; 3]; 2],
    baseline: f64,
    step: &[f64; POSE_DIMENSION],
) -> RelativePose {
    let rotation_update = manifold::exp_so3(&[step[0], step[1], step[2]]);
    let direction = manifold::retract_s2(
        translation_direction,
        translation_basis,
        &[step[3], step[4]],
    );
    RelativePose {
        rotation: manifold::mat_mul(&rotation_update, &pose.rotation),
        translation: direction.map(|value| baseline * value),
    }
}

pub(super) fn validate_pose(pose: &RelativePose) -> Result<(), BundleAdjustmentError> {
    if manifold::Rotation3::try_from_matrix(pose.rotation).is_err() {
        return Err(BundleAdjustmentError::InvalidRotation);
    }
    let baseline = norm(&pose.translation);
    if pose.translation.iter().any(|value| !value.is_finite())
        || !(baseline.is_finite() && baseline > 64.0 * f64::EPSILON)
    {
        return Err(BundleAdjustmentError::InvalidTranslation);
    }
    Ok(())
}

pub(super) fn validate_points(points: &[[f64; 3]]) -> Result<(), BundleAdjustmentError> {
    for (index, point) in points.iter().enumerate() {
        if let Some(coordinate) = point.iter().position(|value| !value.is_finite()) {
            return Err(BundleAdjustmentError::NonFinitePoint { index, coordinate });
        }
    }
    Ok(())
}

pub(super) fn validate_projection_domain<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    points: &[[f64; 3]],
) -> Result<(), BundleAdjustmentError> {
    for (index, point) in points.iter().enumerate() {
        let transformed = transform(&pose.rotation, &pose.translation, point);
        for (view, camera_point) in [(0, point), (1, &transformed)] {
            if model.project(view, index, camera_point).is_none() {
                return Err(BundleAdjustmentError::PointNotProjectable {
                    view: view + 1,
                    index,
                });
            }
        }
    }
    Ok(())
}

fn norm<const N: usize>(vector: &[f64; N]) -> f64 {
    squared_norm(vector).sqrt()
}

fn squared_norm<const N: usize>(vector: &[f64; N]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

fn limit_pose_step(vector: &mut [f64; POSE_DIMENSION], maximum: f64) {
    let rotation = norm(&[vector[0], vector[1], vector[2]]);
    let translation_direction = norm(&[vector[3], vector[4]]);
    let length = rotation.max(translation_direction);
    if length > maximum {
        for value in vector {
            *value *= maximum / length;
        }
    }
}
