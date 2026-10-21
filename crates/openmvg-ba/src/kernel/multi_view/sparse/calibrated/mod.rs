// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated sparse nonlinear driver.

mod assembly;
mod camera_step;
mod candidate;
#[cfg(all(test, feature = "parallel"))]
mod mixed_controls_and_free_points_tests;
#[cfg(all(test, feature = "parallel"))]
mod shared_pose_tests;

use crate::kernel::RelativePose;
#[cfg(feature = "unstable-internals")]
use crate::kernel::multi_view::CalibratedBundleAdjustmentProblem;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::schur::{self, GaugeState};
#[cfg(feature = "unstable-internals")]
use crate::kernel::multi_view::sparse::SparseWorkspaceOptions;
use crate::kernel::multi_view::sparse::api::{
    SparseBundleAdjustmentDiagnostics, SparseBundleAdjustmentError, SparseBundleAdjustmentReport,
    SparseCameraRankAssessment, SparseCameraSolver, SparseNonlinearDiagnostics,
};
use crate::kernel::multi_view::sparse::direct::assess_camera_rank_with_expected_nullity;
use crate::kernel::multi_view::sparse::retained::RecomputedSchurContext;
use crate::kernel::multi_view::sparse::workspace::SparseBundleAdjustmentWorkspace;
use crate::kernel::multi_view::{
    BundleAdjustmentPriors, FixedVariableBlocks, MultiViewBundleAdjustmentError,
    MultiViewMeasurements, ScaleGauge,
};
use crate::kernel::multi_view::{optimizer, validation};
use crate::numerics::refine::nonlinear::{LmRunState, StepRejection, final_stage_index, stages};
use crate::numerics::refine::{NoiseModel, Options};
use assembly::calibrated_layout;
use camera_step::{
    CameraStep, CameraStepOutcome, CameraStepPolicy, SchurStatistics, calibrated_model_reduction,
    solve_camera_step,
};
use candidate::evaluate_candidate;

pub(super) use assembly::assemble_sparse;
#[cfg(feature = "parallel")]
pub(super) use assembly::assemble_sparse_direct;
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) use assembly::assemble_sparse_with_prior_scale;
#[cfg(test)]
pub(super) use camera_step::materialize_retained_schur;

/// Upper bound on the true relative residual used to confirm iterative outer
/// convergence. The solve tightens further when the cost tolerance requires
/// it because quadratic-model error is second order in the linear error.
pub(super) const MAXIMUM_ITERATIVE_CONVERGENCE_RELATIVE_TOLERANCE: f64 = 1e-6;

#[cfg(feature = "unstable-internals")]
impl CalibratedBundleAdjustmentProblem<'_> {
    /// Analyze this visibility graph and allocate sparse solver scratch.
    pub fn try_sparse_workspace(
        self,
        camera_count: usize,
        options: SparseWorkspaceOptions,
    ) -> Result<SparseBundleAdjustmentWorkspace, SparseBundleAdjustmentError> {
        SparseBundleAdjustmentWorkspace::try_new_with_options(
            camera_count,
            self.measurements(),
            self.gauge(),
            options,
        )
    }

    /// Analyze this graph with exact fixed pose and point blocks.
    pub fn try_sparse_workspace_with_fixed_blocks(
        self,
        camera_count: usize,
        fixed: FixedVariableBlocks<'_>,
        options: SparseWorkspaceOptions,
    ) -> Result<SparseBundleAdjustmentWorkspace, SparseBundleAdjustmentError> {
        SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
            camera_count,
            self.measurements(),
            self.gauge(),
            fixed,
            options,
        )
    }

    /// Refine calibrated poses and points with the sparse Schur backend.
    ///
    /// Reusing the topology-specific `workspace` keeps the solve free of
    /// routine heap allocation.
    pub fn refine_sparse(
        self,
        poses: &mut [RelativePose],
        points: &mut [[f64; 3]],
        workspace: &mut SparseBundleAdjustmentWorkspace,
        options: Options,
    ) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
        refine_ba_multi_view_sparse(
            poses,
            points,
            self.measurements(),
            self.gauge(),
            workspace,
            options,
            self.noise(),
        )
    }

    /// Refine with exact fixed blocks and unary Gaussian factors.
    #[allow(clippy::too_many_arguments)]
    pub fn refine_sparse_with_factors(
        self,
        poses: &mut [RelativePose],
        points: &mut [[f64; 3]],
        fixed: FixedVariableBlocks<'_>,
        priors: BundleAdjustmentPriors<'_>,
        workspace: &mut SparseBundleAdjustmentWorkspace,
        options: Options,
    ) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
        refine_ba_multi_view_sparse_with_factors(
            poses,
            points,
            self.measurements(),
            self.gauge(),
            fixed,
            priors,
            workspace,
            options,
            self.noise(),
        )
    }
}

/// Refine calibrated cameras and points with the workspace's sparse Schur solver.
///
/// The workspace must have been constructed from the same camera count,
/// visibility graph, and gauge. Intrinsic values and measured coordinates may
/// change without rebuilding it. An error leaves `poses` and `points`
/// unchanged.
#[cfg(any(test, feature = "unstable-internals"))]
pub fn refine_ba_multi_view_sparse(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut SparseBundleAdjustmentWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
    refine_ba_multi_view_sparse_with_factors(
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

/// Refine a calibrated sparse problem with fixed blocks and unary priors.
///
/// The workspace must have been built with the same fixed-block masks. Prior
/// values may change between solves without rebuilding the workspace. An
/// error leaves `poses` and `points` unchanged.
#[allow(clippy::too_many_arguments)]
pub fn refine_ba_multi_view_sparse_with_factors(
    poses: &mut [RelativePose],
    points: &mut [[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
    options: Options,
    noise: NoiseModel<'_>,
) -> Result<SparseBundleAdjustmentReport, SparseBundleAdjustmentError> {
    options
        .try_validate()
        .map_err(MultiViewBundleAdjustmentError::from)?;
    if !(workspace.core.initial_damping.is_finite() && workspace.core.initial_damping > 0.0) {
        return Err(SparseBundleAdjustmentError::InitialDampingInvalid);
    }
    if !workspace.matches_with_fixed(poses.len(), measurements, gauge, fixed)
        || points.len() != workspace.core.signature.point_count()
    {
        return Err(SparseBundleAdjustmentError::TopologyMismatch);
    }
    let model = ObservationModel::from(measurements);
    noise
        .try_validate(model.observation_count())
        .map_err(MultiViewBundleAdjustmentError::from)?;
    validation::problem_with_capacity_and_fixed(
        poses,
        points,
        model,
        gauge,
        workspace.core.signature.camera_count(),
        workspace.core.signature.point_count(),
        fixed.points,
        !workspace.core.signature.explicit_pose_masks(),
    )?;
    validation::factors(poses, points, model, gauge, fixed, priors)?;
    if workspace.core.signature.explicit_pose_masks()
        && let Some(prior) = priors.poses.iter().find(|prior| {
            workspace.core.signature.pose_masks()[prior.camera] != crate::PoseVariables::FREE
        })
    {
        return Err(MultiViewBundleAdjustmentError::InvalidPriorTarget {
            block: crate::kernel::multi_view::BundleAdjustmentBlock::WorldToCameraPose,
            index: prior.camera,
        }
        .into());
    }
    let fixed = if workspace.core.signature.explicit_pose_masks() {
        fixed.canonical_with_camera_zero()
    } else {
        fixed.canonical()
    };
    validation::projection_domain(model, poses, points)?;
    // Exact final rank assessment is a configured part of this solve. Prepare
    // its fallible symbolic/numeric storage before an accepted LM step can
    // mutate caller-owned state. Retained-only workspaces keep this storage
    // lazy when exact rank diagnostics are explicitly disabled.
    if workspace.core.exact_camera_rank_diagnostics {
        workspace.ensure_direct_storage()?;
    }
    let dimension = workspace.core.layout.dimension();
    let scene_scale = if workspace.core.signature.explicit_pose_masks() {
        optimizer::observed_scene_scale(poses, points, model, gauge)
    } else {
        optimizer::scene_scale(poses, points, gauge)
    };
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
        ScaleGauge::FixedBaseline { camera } => Some(optimizer::norm(&poses[camera].translation)),
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
    // Workspace construction resolves Auto before selecting the route-aware
    // initial damping. An explicit workspace override remains authoritative.
    let mut run = LmRunState::with_initial_damping(workspace.core.initial_damping);
    let mut convergence_confirmation_relative_residual = None;
    let mut schur_statistics = SchurStatistics::default();
    let mut nonlinear_diagnostics = SparseNonlinearDiagnostics::default();

    'optimization: for (stage_index, stage) in schedule.into_iter().enumerate() {
        let mut failed_retained_lambda: Option<f64> = None;
        let mut confirm_iterative_convergence = false;
        for _ in 0..stage.max_iterations {
            run.begin_iteration();
            let gauge_state = GaugeState::new(gauge, poses, scene_scale);
            let attempt_retained = match workspace.camera_solver() {
                SparseCameraSolver::DirectLlt => false,
                SparseCameraSolver::IterativeSchur => {
                    failed_retained_lambda.is_none_or(|failed| run.damping() > failed)
                }
            };
            let assembly = assemble_sparse(
                model,
                poses,
                points,
                gauge_state,
                stage.loss,
                noise,
                run.damping(),
                attempt_retained,
                fixed,
                priors,
                workspace,
            );
            if initial_metrics.is_none() {
                initial_metrics = assembly.objective_metrics(model.observation_count());
            }
            if !assembly.valid {
                nonlinear_diagnostics.linearization_failures += 1;
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
            let confirmation_tolerance = (confirm_iterative_convergence
                && workspace.camera_solver() == SparseCameraSolver::IterativeSchur)
                .then(|| {
                    workspace
                        .iterative_schur_relative_tolerance()
                        .unwrap_or(iterative_convergence_tolerance)
                        .min(iterative_convergence_tolerance)
                });
            let linearization_context = RecomputedSchurContext {
                model,
                poses,
                points,
                gauge: gauge_state,
                loss: stage.loss,
                noise,
            };
            let camera_step = match solve_camera_step(
                linearization_context,
                fixed,
                priors,
                workspace,
                CameraStepPolicy {
                    damping: run.damping(),
                    attempt_retained,
                    relative_tolerance: confirmation_tolerance,
                },
            ) {
                CameraStepOutcome::Step(step) => step,
                CameraStepOutcome::Rejected => {
                    nonlinear_diagnostics.linear_solve_failures += 1;
                    run.reject_failure(StepRejection::StepSolve);
                    continue;
                }
                CameraStepOutcome::IterativeRetry(evidence) => {
                    schur_statistics.retry(evidence);
                    nonlinear_diagnostics.linear_solve_failures += 1;
                    run.reject_failure(StepRejection::StepSolve);
                    continue;
                }
                CameraStepOutcome::DirectFallbackUnavailable(evidence) => {
                    schur_statistics.unavailable_fallback(evidence);
                    nonlinear_diagnostics.linear_solve_failures += 1;
                    run.reject_failure(StepRejection::StepSolve);
                    continue;
                }
            };
            match camera_step {
                CameraStep::Direct => {}
                CameraStep::Iterative(result) => {
                    schur_statistics.accept(result);
                }
                CameraStep::DirectFallback(evidence) => {
                    failed_retained_lambda = Some(run.damping());
                    schur_statistics.linear_fallback(evidence);
                }
            }
            nonlinear_diagnostics.record_reduced_step(
                &workspace.core.step,
                workspace.core.layout.camera_offsets(),
                workspace.core.layout.camera_dimensions(),
                workspace.core.layout.intrinsic_offsets(),
                workspace.core.layout.intrinsic_dimensions(),
            );
            let candidate = evaluate_candidate(
                model,
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
            let damped_reduction = calibrated_model_reduction(
                linearization_context,
                assembly.point_model_reduction,
                camera_step,
                workspace,
            );
            let predicted_reduction = optimizer::undamped_model_reduction(
                damped_reduction,
                &workspace.core.diagonal,
                &workspace.core.step,
                run.damping(),
                point_damping_norm,
            );
            let reduction = current_cost - candidate_cost;
            nonlinear_diagnostics.record_candidate(point_step_rms, reduction, predicted_reduction);
            if run.accept_gain(reduction, predicted_reduction) {
                poses.copy_from_slice(&workspace.core.candidate_poses);
                points.copy_from_slice(&workspace.core.candidate_points);
                if matches!(camera_step, CameraStep::DirectFallback(_)) {
                    // The failed retained solve described the previous state.
                    // Once its exact fallback step is accepted, retry retained
                    // Schur on the newly linearized state instead of pinning
                    // all later iterations to the expensive direct path.
                    failed_retained_lambda = None;
                }
                let camera_step_rms = (optimizer::squared_norm(&workspace.core.step)
                    / dimension.max(1) as f64)
                    .sqrt();
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
                if is_target {
                    let converged = match camera_step {
                        CameraStep::Iterative(result)
                            if candidate_convergence.is_some()
                                && result.relative_residual > iterative_convergence_tolerance =>
                        {
                            confirm_iterative_convergence = true;
                            run.certify_convergence(None)
                        }
                        CameraStep::Iterative(_)
                        | CameraStep::Direct
                        | CameraStep::DirectFallback(_) => {
                            confirm_iterative_convergence = false;
                            run.certify_convergence(candidate_convergence)
                        }
                    };
                    if converged {
                        if let CameraStep::Iterative(result) = camera_step {
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

    let final_assembly = assemble_sparse(
        model,
        poses,
        points,
        GaugeState::new(gauge, poses, scene_scale),
        options.loss,
        noise,
        0.0,
        !workspace.core.exact_camera_rank_diagnostics,
        fixed,
        priors,
        workspace,
    );
    let expected_gauge_nullity = if workspace.core.signature.explicit_pose_masks() {
        // Native preparation removes only proved similarity symmetries.
        // Any remaining null direction belongs to requested coordinates.
        0
    } else {
        schur::expected_gauge_nullity(gauge, poses, points, fixed, priors)
    };
    let camera_rank = if !workspace.core.exact_camera_rank_diagnostics {
        SparseCameraRankAssessment::NotAssessed
    } else if final_assembly.valid {
        assess_camera_rank_with_expected_nullity(
            &mut workspace.core,
            workspace.backend.direct_executor(),
            poses,
            expected_gauge_nullity,
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
    // A singular undamped point or camera block invalidates observability,
    // not necessarily the objective at the accepted state.
    let (final_cost, final_rmse) = if final_assembly.valid {
        final_assembly
            .objective_metrics(model.observation_count())
            .unwrap_or((f64::INFINITY, f64::INFINITY))
    } else {
        let (observation_cost, final_rmse) =
            schur::metrics(model, poses, points, options.loss, noise);
        let final_cost = observation_cost
            + schur::prior_cost(model, poses, points, priors).unwrap_or(f64::INFINITY);
        (final_cost, final_rmse)
    };
    let gradient_inf_norm = if final_assembly.valid {
        final_assembly.scaled_gradient_inf_norm
    } else {
        schur::unreduced_scaled_gradient_inf_norm(
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
            calibrated_layout(
                workspace.core.layout.camera_offsets(),
                workspace.core.layout.camera_dimensions(),
                workspace.core.signature.pose_variables(),
                workspace.core.signature.pose_masks(),
            ),
        )
    };
    let camera_solve_quality = schur_statistics.quality(workspace.camera_solver());

    let (initial_cost, initial_rmse) = initial_metrics.unwrap_or((f64::INFINITY, f64::INFINITY));

    Ok(SparseBundleAdjustmentReport {
        loss: options.loss,
        status,
        scale_gauge: gauge,
        normalization_scale: scene_scale,
        camera_pose_variables: workspace.camera_pose_variables(),
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
        initial_damping: workspace.core.initial_damping,
        final_damping: run.damping(),
        last_gain_ratio: run.last_gain_ratio(),
        nonlinear_diagnostics,
        camera_solve_quality,
        retained_camera_steps: schur_statistics.accepted_steps,
        direct_fallback_steps: schur_statistics.fallback_steps,
        iterative_retry_steps: schur_statistics.retry_steps,
        linear_guard_fallback_steps: schur_statistics.linear_guard_fallback_steps,
        unavailable_direct_fallback_steps: schur_statistics.unavailable_direct_fallback_steps,
        retained_inner_iterations: schur_statistics.terms,
        maximum_retained_relative_residual: schur_statistics.maximum_relative_residual,
        convergence_confirmation_relative_residual,
        maximum_rejected_iterative_relative_residual: schur_statistics
            .maximum_rejected_relative_residual,
        diagnostics,
        execution_decision: workspace.execution_decision(),
        solver_working_set_bytes: workspace.solver_specific_working_set_bytes(),
    })
}

#[cfg(all(test, feature = "parallel"))]
mod calibrated_exact_final_information_tests {
    use super::*;
    use crate::kernel::PinholeCamera;
    use crate::kernel::multi_view::PixelObservation;
    use crate::kernel::multi_view::sparse::api::{RetainedCrossPrecision, RetainedSchurMatvec};
    use crate::kernel::multi_view::sparse::{IterativeSchurOptions, SparseWorkspaceOptions};

    #[test]
    fn calibrated_exact_final_information_avoids_parallel_f32_cross_rows() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let initial_poses = [
            RelativePose::new(identity, [0.0; 3]),
            RelativePose::new(identity, [-1.0, 0.0, 0.0]),
            RelativePose::new(identity, [0.15, -0.8, 0.07]),
        ];
        let initial_points = [
            [-0.4, -0.3, 4.0],
            [0.6, -0.2, 4.7],
            [0.2, 0.7, 5.2],
            [-0.7, 0.5, 5.8],
            [0.3, -0.6, 6.3],
            [0.8, 0.4, 4.5],
        ];
        let cameras = [PinholeCamera::new(401.3, 397.1, 0.0, 0.0); 3];
        let observations = (0..initial_points.len())
            .flat_map(|_| {
                [
                    PixelObservation::new(0, [0.0; 2]),
                    PixelObservation::new(1, [0.0; 2]),
                    PixelObservation::new(2, [0.0; 2]),
                ]
            })
            .collect::<Vec<_>>();
        let offsets = [0, 3, 6, 9, 12, 15, 18];
        let measurements = MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        };
        let gauge = ScaleGauge::FixedBaseline { camera: 1 };
        let fixed = FixedVariableBlocks::NONE;
        let mut direct = SparseBundleAdjustmentWorkspace::try_new_with_options(
            3,
            measurements,
            gauge,
            SparseWorkspaceOptions::direct(),
        )
        .unwrap();
        let mut retained = SparseBundleAdjustmentWorkspace::try_new_with_options(
            3,
            measurements,
            gauge,
            SparseWorkspaceOptions::iterative_schur(
                IterativeSchurOptions::new()
                    .with_cross_precision(RetainedCrossPrecision::F32)
                    .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 2 })
                    .with_exact_camera_rank_diagnostics(true),
            ),
        )
        .unwrap();
        assert!(retained.core.exact_camera_rank_diagnostics);
        assert_eq!(retained.retained_schur_parallel_threads(), Some(2));
        for workspace in [&mut direct, &mut retained] {
            let mut poses = initial_poses;
            let mut points = initial_points;
            // Zero nonlinear iterations isolates the mandatory exact final information pass.
            let _ = refine_ba_multi_view_sparse_with_factors(
                &mut poses,
                &mut points,
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
        assert_eq!(direct.core.values.len(), retained.core.values.len());
        let matrix_scale = direct
            .core
            .values
            .iter()
            .map(|value| value.abs())
            .fold(1.0, f64::max);
        // f64 reductions can differ in summation order; compare against the information scale.
        let tolerance = 512.0 * f64::EPSILON * matrix_scale;
        for (&expected, &actual) in direct.core.values.iter().zip(&retained.core.values) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "calibrated exact information used rounded cross rows: expected {expected:e}, actual {actual:e}, tolerance {tolerance:e}"
            );
        }
    }
}

#[cfg(all(test, feature = "parallel"))]
mod zero_dimension_point_major_tests {
    use super::*;
    use crate::kernel::PinholeCamera;
    use crate::kernel::multi_view::PixelObservation;
    use crate::kernel::multi_view::sparse::api::RetainedSchurMatvec;
    use crate::kernel::multi_view::sparse::{IterativeSchurOptions, SparseWorkspaceOptions};

    #[test]
    fn point_major_all_fixed_cameras_can_still_refine_free_points() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let poses = [
            RelativePose::new(identity, [0.0; 3]),
            RelativePose::new(identity, [-1.0, 0.0, 0.0]),
            RelativePose::new(identity, [0.15, -0.8, 0.07]),
        ];
        let truth = [
            [-0.4, -0.3, 4.0],
            [0.6, -0.2, 4.7],
            [0.2, 0.7, 5.2],
            [-0.7, 0.5, 5.8],
            [0.3, -0.6, 6.3],
            [0.8, 0.4, 4.5],
        ];
        let points = truth.map(|point| [point[0] + 0.02, point[1], point[2]]);
        let cameras = [PinholeCamera::new(401.3, 397.1, 0.0, 0.0); 3];
        let observations = truth
            .iter()
            .flat_map(|point| {
                poses.iter().enumerate().map(move |(camera, pose)| {
                    let q: [f64; 3] =
                        std::array::from_fn(|axis| point[axis] + pose.translation[axis]);
                    PixelObservation::new(
                        camera,
                        [
                            cameras[camera].fx * q[0] / q[2],
                            cameras[camera].fy * q[1] / q[2],
                        ],
                    )
                })
            })
            .collect::<Vec<_>>();
        let offsets = [0, 3, 6, 9, 12, 15, 18];
        let measurements = MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        };
        let fixed = FixedVariableBlocks {
            camera_poses: &[true; 3],
            points: &[],
            intrinsics: &[],
            intrinsic_dimensions: &[],
        };
        let gauge = ScaleGauge::Free;
        for matvec in [
            RetainedSchurMatvec::PointMajorParallel { threads: 2 },
            RetainedSchurMatvec::RecomputedParallel { threads: 2 },
        ] {
            let mut run_poses = poses;
            let mut run_points = points;
            // The constructor derives recomputed point storage from the matvec policy
            // and binds the same all-fixed camera mask to both variants.
            let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks(
                3,
                measurements,
                gauge,
                fixed,
                SparseWorkspaceOptions::iterative_schur(
                    IterativeSchurOptions::new().with_matvec(matvec),
                ),
            )
            .unwrap();
            assert_eq!(workspace.core.layout.dimension(), 0);
            let report = refine_ba_multi_view_sparse_with_factors(
                &mut run_poses,
                &mut run_points,
                measurements,
                gauge,
                fixed,
                BundleAdjustmentPriors::NONE,
                &mut workspace,
                Options::l2(1),
                NoiseModel::Unit,
            )
            .unwrap();
            assert!(report.accepted_step_count > 0);
            assert!(report.final_cost < report.initial_cost);
        }
    }
}
