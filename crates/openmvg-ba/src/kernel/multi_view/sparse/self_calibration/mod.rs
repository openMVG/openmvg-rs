// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact sparse radial-camera self-calibration with LLT or retained Schur-PCG.

mod assembly;
mod camera_step;
mod candidate;

use crate::kernel::RelativePose;
use crate::kernel::multi_view::optimizer::{norm, retract_poses, scene_scale, squared_norm};
use crate::kernel::multi_view::schur::{self, GaugeState};
use crate::kernel::multi_view::self_calibration::{
    RadialCamera, SelfCalibratingMeasurements, radial_model, retract_cameras,
};
use crate::kernel::multi_view::sparse::MAXIMUM_ITERATIVE_CONVERGENCE_RELATIVE_TOLERANCE;
use crate::kernel::multi_view::sparse::api::{
    SparseBundleAdjustmentDiagnostics, SparseBundleAdjustmentError, SparseBundleAdjustmentReport,
    SparseCameraRankAssessment, SparseCameraSolver, SparseNonlinearDiagnostics,
};
use crate::kernel::multi_view::sparse::covariance::{
    SparseBundleAdjustmentCovarianceError, SparseIntrinsicMarginalCovariance, block_marginal,
    information_is_finite,
};
use crate::kernel::multi_view::sparse::direct::{
    assess_camera_rank_with_expected_nullity, factor_camera_system,
};
use crate::kernel::multi_view::sparse::retained::RecomputedSchurContext;
use crate::kernel::multi_view::sparse::workspace::SparseSelfCalibrationWorkspace;
use crate::kernel::multi_view::validation;
use crate::kernel::multi_view::{
    BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks,
    MultiViewBundleAdjustmentError, ScaleGauge,
};
use crate::numerics::refine::nonlinear::{LmRunState, StepRejection, final_stage_index, stages};
use crate::numerics::refine::{Loss, NoiseModel, Options};
use assembly::assemble_sparse;
use camera_step::{
    RetainedStatistics, SelfCalibrationCameraStep, SelfCalibrationCameraStepOutcome,
    self_calibration_model_reduction, solve_self_calibration_camera_step,
};
use candidate::candidate_points;

/// Jointly refine per-view radial intrinsics with the workspace's sparse Schur solver.
///
/// The workspace must have been constructed from the same observation
/// topology, gauge, and all-free fixed-block layout. An error leaves all
/// caller-owned state unchanged.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "unstable-internals")]
pub fn refine_ba_multi_view_self_calibrating_sparse(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    cameras: &mut [RadialCamera],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut SparseSelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
    refine(
        poses,
        points,
        cameras,
        None,
        measurements,
        gauge,
        FixedVariableBlocks::NONE,
        BundleAdjustmentPriors::NONE,
        workspace,
        options,
        noise,
    )
}

/// Jointly refine shared radial intrinsic groups with fixed blocks and priors.
///
/// Observation topology, group assignment, gauge, and fixed masks must match
/// workspace construction; coordinates, parameters, and priors may change. An
/// error leaves all caller-owned state unchanged.
#[allow(clippy::too_many_arguments)]
pub fn refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    intrinsic_groups: &mut [RadialCamera],
    camera_to_group: &[usize],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseSelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
    refine(
        poses,
        points,
        intrinsic_groups,
        Some(camera_to_group),
        measurements,
        gauge,
        fixed,
        priors,
        workspace,
        options,
        noise,
    )
}

/// Marginal covariance of every free radial intrinsic group at the given state.
///
/// Uses the exact undamped information of the problem
/// [`refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors`]
/// solves without priors, typically evaluated at its solution. Robust losses
/// report fixed-final-weight uncertainty normalized to one at zero residual.
/// Intrinsic coordinates are gauge invariant, so the result does not depend on
/// which pose blocks anchor the gauge, but the gauge must leave no null space.
/// Groups with no free parameter are omitted. The workspace must match the
/// solve's topology, gauge, and fixed masks; its direct storage is prepared on
/// demand.
#[allow(clippy::too_many_arguments)]
pub(crate) fn intrinsic_marginal_covariances_self_calibrating_grouped_sparse(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    intrinsic_groups: &[RadialCamera],
    camera_to_group: &[usize],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    workspace: &mut SparseSelfCalibrationWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<Vec<SparseIntrinsicMarginalCovariance>, SparseBundleAdjustmentCovarianceError> {
    loss.try_validate()
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    if !workspace.matches(
        poses.len(),
        intrinsic_groups.len(),
        Some(camera_to_group),
        measurements,
        gauge,
        fixed,
    ) || points.len() != workspace.sparse.signature.point_count()
    {
        return Err(SparseBundleAdjustmentCovarianceError::TopologyMismatch);
    }
    let model = radial_model(measurements, intrinsic_groups, Some(camera_to_group));
    noise
        .try_validate(model.observation_count())
        .map_err(MultiViewBundleAdjustmentError::Configuration)?;
    validation::problem_with_capacity(
        poses,
        points,
        model,
        gauge,
        workspace.sparse.signature.camera_count(),
        workspace.sparse.signature.point_count(),
    )?;
    let priors = BundleAdjustmentPriors::NONE;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    let fixed = fixed.canonical();
    validation::projection_domain(model, poses, points)?;
    if schur::expected_gauge_nullity(gauge, poses, points, fixed, priors) != 0 {
        return Err(SparseBundleAdjustmentCovarianceError::FreeScaleGauge);
    }
    let zero_weight = loss.sqrt_weight(0.0);
    if !(zero_weight.is_finite() && zero_weight > 0.0) {
        return Err(SparseBundleAdjustmentCovarianceError::NonFiniteInformation);
    }
    workspace
        .ensure_direct_storage()
        .map_err(SparseBundleAdjustmentCovarianceError::SparseStorage)?;
    let normalization_scale = scene_scale(poses, points, gauge);
    let assembly = assemble_sparse(
        model,
        poses,
        points,
        GaugeState::new(gauge, poses, normalization_scale),
        loss,
        noise,
        0.0,
        fixed,
        priors,
        false,
        workspace,
    );
    let core = &mut workspace.sparse;
    if !information_is_finite(core) {
        return Err(SparseBundleAdjustmentCovarianceError::NonFiniteInformation);
    }
    if !assembly.valid || (core.layout.dimension() > 0 && !factor_camera_system(core)) {
        return Err(SparseBundleAdjustmentCovarianceError::RankDeficientOrIndefinite);
    }
    let weight_normalization = zero_weight * zero_weight;
    let mut marginals = Vec::new();
    for group in 0..intrinsic_groups.len() {
        let dimension = fixed.intrinsic_dimension(group);
        if dimension == 0 {
            continue;
        }
        let offset = core.layout.intrinsic_offsets()[group];
        marginals.push(SparseIntrinsicMarginalCovariance {
            group,
            dimension,
            matrix: block_marginal(offset, dimension, weight_normalization, core)?,
        });
    }
    Ok(marginals)
}

#[allow(clippy::too_many_arguments)]
fn refine(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    cameras: &mut [RadialCamera],
    camera_to_group: Option<&[usize]>,
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseSelfCalibrationWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
    options
        .try_validate()
        .map_err(MultiViewBundleAdjustmentError::from)?;
    if !(workspace.sparse.initial_damping.is_finite() && workspace.sparse.initial_damping > 0.0) {
        return Err(SparseBundleAdjustmentError::InitialDampingInvalid);
    }
    validation::fixed_topology(poses.len(), points.len(), cameras.len(), gauge, fixed)?;
    if !workspace.matches(
        poses.len(),
        cameras.len(),
        camera_to_group,
        measurements,
        gauge,
        fixed,
    ) || points.len() != workspace.sparse.signature.point_count()
    {
        return Err(SparseBundleAdjustmentError::TopologyMismatch);
    }
    let model = radial_model(measurements, cameras, camera_to_group);
    noise
        .try_validate(model.observation_count())
        .map_err(MultiViewBundleAdjustmentError::from)?;
    validation::problem_with_capacity(
        poses,
        points,
        model,
        gauge,
        workspace.sparse.signature.camera_count(),
        workspace.sparse.signature.point_count(),
    )?;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    let fixed = fixed.canonical();
    validation::projection_domain(model, poses, points)?;
    // Exact final rank assessment is part of the configured solve contract.
    // Prepare every fallible direct-storage dependency before an accepted LM
    // step can mutate caller-owned poses, points, or intrinsic groups.
    if workspace.sparse.exact_camera_rank_diagnostics {
        workspace.ensure_direct_storage()?;
    }

    let dimension = workspace.sparse.layout.dimension();
    let normalization_scale = scene_scale(poses, points, gauge);
    let schedule = stages(options);
    let reuse_initial_assembly = schedule
        .iter()
        .find(|stage| stage.max_iterations > 0)
        .is_some_and(|stage| stage.loss == options.loss);
    let mut initial_metrics = None;
    if !reuse_initial_assembly {
        let (observation_cost, initial_rmse) =
            schur::metrics(model, poses, points, options.loss, noise);
        let initial_cost = observation_cost
            + schur::prior_cost(model, poses, points, priors).unwrap_or(f64::INFINITY);
        initial_metrics = Some((initial_cost, initial_rmse));
    }
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
    let target_stage = final_stage_index(&schedule);
    let iterative_convergence_tolerance = options
        .convergence
        .relative_cost
        .sqrt()
        .min(MAXIMUM_ITERATIVE_CONVERGENCE_RELATIVE_TOLERANCE);
    let mut run = LmRunState::with_initial_damping(workspace.sparse.initial_damping);
    let mut convergence_confirmation_relative_residual = None;
    let mut retained_statistics = RetainedStatistics::default();
    let mut nonlinear_diagnostics = SparseNonlinearDiagnostics::default();

    'optimization: for (stage_index, stage) in schedule.into_iter().enumerate() {
        let mut failed_retained_lambda: Option<f64> = None;
        let mut confirm_iterative_convergence = false;
        for _ in 0..stage.max_iterations {
            run.begin_iteration();
            let gauge_state = GaugeState::new(gauge, poses, normalization_scale);
            let current_model = radial_model(measurements, cameras, camera_to_group);
            let linearization_context = RecomputedSchurContext {
                model: current_model,
                poses,
                points,
                gauge: gauge_state,
                loss: stage.loss,
                noise,
            };
            let attempt_retained = workspace.camera_solver() == SparseCameraSolver::IterativeSchur
                && failed_retained_lambda.is_none_or(|failed| run.damping() > failed);
            let assembly = assemble_sparse(
                current_model,
                poses,
                points,
                gauge_state,
                stage.loss,
                noise,
                run.damping(),
                fixed,
                priors,
                attempt_retained,
                workspace,
            );
            if initial_metrics.is_none() {
                initial_metrics = assembly.objective_metrics(current_model.observation_count());
            }
            if !assembly.valid {
                nonlinear_diagnostics.linearization_failures += 1;
                run.reject_failure(StepRejection::Linearization);
                continue;
            }
            let current_cost = assembly.cost;
            if stage_index == target_stage
                && run.gradient_converged(
                    assembly.scaled_gradient_inf_norm,
                    current_cost,
                    options.convergence,
                )
            {
                break 'optimization;
            }
            let confirmation_tolerance = (confirm_iterative_convergence
                && workspace.camera_solver() == SparseCameraSolver::IterativeSchur)
                .then(|| {
                    workspace
                        .iterative_schur_relative_tolerance()
                        .unwrap_or(iterative_convergence_tolerance)
                        .min(iterative_convergence_tolerance)
                });
            let camera_step = match solve_self_calibration_camera_step(
                linearization_context,
                workspace,
                attempt_retained,
                confirmation_tolerance,
            ) {
                SelfCalibrationCameraStepOutcome::Step(step) => step,
                SelfCalibrationCameraStepOutcome::Rejected => {
                    nonlinear_diagnostics.linear_solve_failures += 1;
                    run.reject_failure(StepRejection::StepSolve);
                    continue;
                }
                SelfCalibrationCameraStepOutcome::IterativeRetry(evidence) => {
                    retained_statistics.retry(evidence);
                    nonlinear_diagnostics.linear_solve_failures += 1;
                    run.reject_failure(StepRejection::StepSolve);
                    continue;
                }
                SelfCalibrationCameraStepOutcome::DirectFallbackUnavailable(evidence) => {
                    retained_statistics.unavailable_fallback(evidence);
                    nonlinear_diagnostics.linear_solve_failures += 1;
                    run.reject_failure(StepRejection::StepSolve);
                    continue;
                }
            };
            match camera_step {
                SelfCalibrationCameraStep::Direct => {}
                SelfCalibrationCameraStep::Iterative(result) => {
                    retained_statistics.accept(result);
                }
                SelfCalibrationCameraStep::DirectFallback(evidence) => {
                    failed_retained_lambda = Some(run.damping());
                    retained_statistics.fallback(evidence);
                }
            }
            nonlinear_diagnostics.record_reduced_step(
                &workspace.sparse.step,
                workspace.sparse.layout.camera_offsets(),
                workspace.sparse.layout.camera_dimensions(),
                workspace.sparse.layout.intrinsic_offsets(),
                workspace.sparse.layout.intrinsic_dimensions(),
            );
            retract_poses(
                poses,
                &mut workspace.sparse.candidate_poses,
                gauge_state,
                workspace.sparse.layout.camera_offsets(),
                workspace.sparse.layout.camera_dimensions(),
                &workspace.sparse.step,
                0,
            );
            retract_cameras(
                cameras,
                &mut workspace.candidate_cameras,
                workspace.sparse.layout.intrinsic_offsets(),
                workspace.sparse.layout.intrinsic_dimensions(),
                &workspace.sparse.step,
            );
            let candidate = candidate_points(
                measurements,
                cameras,
                camera_to_group,
                poses,
                points,
                gauge_state,
                stage.loss,
                noise,
                run.damping(),
                fixed,
                priors,
                workspace,
            );
            let Some((point_step_rms, point_damping_norm, candidate_cost)) = candidate else {
                nonlinear_diagnostics.candidate_failures += 1;
                run.reject_failure(StepRejection::Candidate);
                continue;
            };
            let predicted_reduction = self_calibration_model_reduction(
                workspace,
                assembly.point_model_reduction,
                camera_step,
                run.damping(),
                point_damping_norm,
            )
            .unwrap_or(f64::NAN);
            let reduction = current_cost - candidate_cost;
            nonlinear_diagnostics.record_candidate(point_step_rms, reduction, predicted_reduction);
            if run.accept_gain(reduction, predicted_reduction) {
                poses.copy_from_slice(&workspace.sparse.candidate_poses);
                points.copy_from_slice(&workspace.sparse.candidate_points);
                cameras.copy_from_slice(&workspace.candidate_cameras);
                if matches!(camera_step, SelfCalibrationCameraStep::DirectFallback(_)) {
                    // A successful fallback changes the state and therefore
                    // invalidates the retained-failure marker. Retry PCG on
                    // the next linearization instead of remaining on direct
                    // LLT for the rest of the stage.
                    failed_retained_lambda = None;
                }
                let reduced_rms =
                    (squared_norm(&workspace.sparse.step) / dimension.max(1) as f64).sqrt();
                let step_norm =
                    (reduced_rms * reduced_rms + point_step_rms * point_step_rms).sqrt();
                let is_target = stage_index == target_stage;
                let candidate_convergence = run.record_accepted_step(
                    step_norm,
                    reduction,
                    current_cost,
                    is_target,
                    options.convergence,
                );
                if is_target {
                    let converged = match camera_step {
                        SelfCalibrationCameraStep::Iterative(result)
                            if candidate_convergence.is_some()
                                && result.relative_residual > iterative_convergence_tolerance =>
                        {
                            confirm_iterative_convergence = true;
                            run.certify_convergence(None)
                        }
                        SelfCalibrationCameraStep::Iterative(_)
                        | SelfCalibrationCameraStep::Direct
                        | SelfCalibrationCameraStep::DirectFallback(_) => {
                            confirm_iterative_convergence = false;
                            run.certify_convergence(candidate_convergence)
                        }
                    };
                    if converged {
                        if let SelfCalibrationCameraStep::Iterative(result) = camera_step {
                            convergence_confirmation_relative_residual =
                                Some(result.relative_residual);
                        }
                        break 'optimization;
                    }
                }
            } else {
                nonlinear_diagnostics.gain_rejections += 1;
            }
        }
    }

    let final_model = radial_model(measurements, cameras, camera_to_group);
    let final_context = RecomputedSchurContext {
        model: final_model,
        poses,
        points,
        gauge: GaugeState::new(gauge, poses, normalization_scale),
        loss: options.loss,
        noise,
    };
    let final_assembly = assemble_sparse(
        final_model,
        poses,
        points,
        final_context.gauge,
        options.loss,
        noise,
        0.0,
        fixed,
        priors,
        workspace.camera_solver() == SparseCameraSolver::IterativeSchur
            && !workspace.sparse.exact_camera_rank_diagnostics,
        workspace,
    );
    let expected_nullity = schur::expected_gauge_nullity(gauge, poses, points, fixed, priors);
    // Exact diagnostics use the fresh f64 Schur assembly above, regardless of
    // the precision selected for nonlinear retained steps. A rounded cross-row
    // cache must not add curvature to a null direction and certify its rank.
    let camera_rank = if !workspace.sparse.exact_camera_rank_diagnostics {
        SparseCameraRankAssessment::NotAssessed
    } else if final_assembly.valid {
        assess_camera_rank_with_expected_nullity(
            &mut workspace.sparse,
            None,
            poses,
            expected_nullity,
        )
    } else {
        SparseCameraRankAssessment::RankDeficientOrIndefinite
    };
    let diagnostics = SparseBundleAdjustmentDiagnostics {
        pose_dimension: dimension,
        reduced_nonzero_count: workspace.reduced_nonzero_count(),
        pose_rank: camera_rank,
        weak_point_count: final_assembly.weak_point_count,
        minimum_point_hessian_reciprocal_condition_number: final_assembly.minimum_point_rcond,
        minimum_reduced_diagonal: final_assembly.minimum_reduced_diagonal,
        maximum_reduced_diagonal: final_assembly.maximum_reduced_diagonal,
        damping_floor_clamp_count: final_assembly.damping_floor_clamp_count,
    };
    let status = run.status(diagnostics.is_well_constrained());
    let (final_cost, final_rmse) = if final_assembly.valid {
        final_assembly
            .objective_metrics(final_model.observation_count())
            .unwrap_or((f64::INFINITY, f64::INFINITY))
    } else {
        let (observation_cost, final_rmse) =
            schur::metrics(final_model, poses, points, options.loss, noise);
        let final_cost = observation_cost
            + schur::prior_cost(final_model, poses, points, priors).unwrap_or(f64::INFINITY);
        (final_cost, final_rmse)
    };

    let (initial_cost, initial_rmse) = initial_metrics.unwrap_or((f64::INFINITY, f64::INFINITY));
    #[cfg(feature = "parallel")]
    let noninvertible_measurement_count = workspace
        .backend
        .install_parallel(|| final_model.noninvertible_measurement_count_parallel())
        .unwrap_or_else(|| final_model.noninvertible_measurement_count());
    #[cfg(not(feature = "parallel"))]
    let noninvertible_measurement_count = final_model.noninvertible_measurement_count();

    Ok(SparseBundleAdjustmentReport {
        loss: options.loss,
        status,
        scale_gauge: gauge,
        normalization_scale,
        camera_pose_variables: CameraPoseVariables::RotationAndTranslation,
        fixed_baseline,
        fixed_translation_component,
        pose_count: poses.len(),
        point_count: points.len(),
        observation_count: measurements.observation_count(),
        noninvertible_measurement_count,
        iteration_count: run.iterations(),
        accepted_step_count: run.accepted_steps(),
        rejected_step_count: run.rejected_steps(),
        step_rejections: run.rejections(),
        initial_cost,
        final_cost,
        initial_rmse,
        final_rmse,
        gradient_inf_norm: if final_assembly.valid {
            final_assembly.scaled_gradient_inf_norm
        } else {
            let layout = &workspace.sparse.layout;
            schur::unreduced_scaled_gradient_inf_norm(
                schur::BaProblemView {
                    model: final_model,
                    poses,
                    points,
                    fixed,
                    priors,
                },
                schur::LinearizationPolicy {
                    gauge: final_context.gauge,
                    loss: options.loss,
                    noise,
                    damping: 0.0,
                    prior_sqrt_scale: 1.0,
                },
                layout.dimension(),
                schur::VariableLayout {
                    offsets: layout.camera_offsets(),
                    pose_dimensions: layout.camera_dimensions(),
                    intrinsic_offsets: layout.intrinsic_offsets(),
                    intrinsic_dimensions: layout.intrinsic_dimensions(),
                    pose_masks: &[],
                    pose_variables: CameraPoseVariables::RotationAndTranslation,
                },
            )
        },
        step_norm: run.step_norm(),
        initial_damping: workspace.sparse.initial_damping,
        final_damping: run.damping(),
        last_gain_ratio: run.last_gain_ratio(),
        nonlinear_diagnostics,
        camera_solve_quality: retained_statistics.quality(workspace.camera_solver()),
        retained_camera_steps: retained_statistics.accepted_steps,
        direct_fallback_steps: retained_statistics.fallback_steps,
        iterative_retry_steps: retained_statistics.retry_steps,
        linear_guard_fallback_steps: retained_statistics.fallback_steps,
        unavailable_direct_fallback_steps: retained_statistics.unavailable_direct_fallback_steps,
        retained_inner_iterations: retained_statistics.iterations,
        maximum_retained_relative_residual: retained_statistics.maximum_relative_residual,
        convergence_confirmation_relative_residual,
        maximum_rejected_iterative_relative_residual: retained_statistics
            .maximum_rejected_relative_residual,
        diagnostics,
        execution_decision: workspace.execution_decision(),
        solver_working_set_bytes: workspace.solver_specific_working_set_bytes(),
    })
}

#[cfg(test)]
mod exact_final_information_tests {
    use super::*;
    use crate::kernel::multi_view::PixelObservation;
    use crate::kernel::multi_view::sparse::IterativeSchurOptions;
    use crate::kernel::multi_view::sparse::api::RetainedCrossPrecision;

    #[test]
    fn exact_final_information_does_not_use_f32_retained_cross_rows() {
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut poses = [
            RelativePose::new(rotation, [0.0; 3]),
            RelativePose::new(rotation, [-1.0, 0.0, 0.0]),
        ];
        let mut points = [
            [-0.4, -0.3, 4.0],
            [0.6, -0.2, 4.7],
            [0.2, 0.7, 5.2],
            [-0.7, 0.5, 5.8],
        ];
        let mut cameras = [RadialCamera::new(401.3, 0.03, -0.01)];
        let observations = (0..points.len())
            .flat_map(|_| {
                [
                    PixelObservation::new(0, [0.0; 2]),
                    PixelObservation::new(1, [0.0; 2]),
                ]
            })
            .collect::<Vec<_>>();
        let offsets = [0, 2, 4, 6, 8];
        let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
        let mapping = [0, 0];
        let gauge = ScaleGauge::FixedBaseline { camera: 1 };
        let fixed = FixedVariableBlocks {
            camera_poses: &[true, false],
            points: &[],
            intrinsics: &[false],
            intrinsic_dimensions: &[1],
        };
        let mut direct = SparseSelfCalibrationWorkspace::try_new_grouped(
            2,
            1,
            &mapping,
            measurements,
            gauge,
            fixed,
        )
        .unwrap();
        let mut retained =
            SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
                2,
                1,
                &mapping,
                measurements,
                gauge,
                fixed,
                IterativeSchurOptions::new().with_cross_precision(RetainedCrossPrecision::F32),
            )
            .unwrap();
        let mut direct_poses = poses;
        let mut direct_points = points;
        let mut direct_cameras = cameras;
        for (poses, points, cameras, workspace) in [
            (
                &mut direct_poses,
                &mut direct_points,
                &mut direct_cameras,
                &mut direct,
            ),
            (&mut poses, &mut points, &mut cameras, &mut retained),
        ] {
            // Zero iterations isolates the mandatory final information pass.
            let _ = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
                poses,
                points,
                cameras,
                &mapping,
                measurements,
                gauge,
                fixed,
                BundleAdjustmentPriors::NONE,
                workspace,
                Options::l2(0),
                NoiseModel::Unit,
            )
            .unwrap();
        }
        assert_eq!(direct.sparse.values.len(), retained.sparse.values.len());
        for (&expected, &actual) in direct.sparse.values.iter().zip(&retained.sparse.values) {
            let tolerance = 512.0 * f64::EPSILON * (1.0 + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "exact final information used a rounded cache: expected {expected:e}, actual {actual:e}"
            );
        }
    }
}
