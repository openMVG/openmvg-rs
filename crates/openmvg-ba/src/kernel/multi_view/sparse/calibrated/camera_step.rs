// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera-step solves over the direct or retained Schur system.

use super::assembly::assemble_sparse;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::optimizer;
use crate::kernel::multi_view::sparse::api::{
    SparseBundleAdjustmentError, SparseCameraSolveQuality, SparseCameraSolver,
};
use crate::kernel::multi_view::sparse::direct::{
    factor_camera_system, hessian_quadratic_form, solve_factored_camera_system,
};
use crate::kernel::multi_view::sparse::retained::{
    RecomputedSchurContext, RejectedSchurSolve, SchurAttempt, SchurSolve,
};
use crate::kernel::multi_view::sparse::workspace::{
    CalibratedBackend, SparseBundleAdjustmentWorkspace, SparseWorkspaceCore,
};
use crate::kernel::multi_view::{BundleAdjustmentPriors, FixedVariableBlocks};

#[derive(Clone, Copy)]
pub(super) enum CameraStep {
    Direct,
    Iterative(SchurSolve),
    DirectFallback(RejectedSchurSolve),
}

pub(super) enum CameraStepOutcome {
    Step(CameraStep),
    Rejected,
    IterativeRetry(RejectedSchurSolve),
    DirectFallbackUnavailable(RejectedSchurSolve),
}

#[derive(Clone, Copy)]
pub(super) struct CameraStepPolicy {
    pub(super) damping: f64,
    pub(super) attempt_retained: bool,
    pub(super) relative_tolerance: Option<f64>,
}

pub(super) fn calibrated_model_reduction(
    context: RecomputedSchurContext<'_>,
    point_model_reduction: f64,
    camera_step: CameraStep,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> f64 {
    let retained_step = match camera_step {
        CameraStep::Iterative(_) => true,
        CameraStep::Direct | CameraStep::DirectFallback(_) => false,
    };
    if retained_step {
        return retained_predicted_reduction(context, point_model_reduction, workspace)
            .unwrap_or(f64::NAN);
    }
    let quadratic = hessian_quadratic_form(&workspace.core, &workspace.core.step);
    optimizer::reduced_model_reduction(
        point_model_reduction,
        &workspace.core.gradient,
        &workspace.core.step,
        quadratic,
    )
}

fn retained_predicted_reduction(
    context: RecomputedSchurContext<'_>,
    point_model_reduction: f64,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> Option<f64> {
    let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
    let CalibratedBackend::Retained(retained) = backend else {
        return None;
    };
    if retained.uses_recomputed_cross_rows() {
        retained.predicted_reduction_recomputed(context, core, &core.step, point_model_reduction)
    } else {
        retained.predicted_reduction(context.model, core, &core.step)
    }
}

#[derive(Default)]
pub(super) struct SchurStatistics {
    pub(super) accepted_steps: usize,
    pub(super) fallback_steps: usize,
    pub(super) retry_steps: usize,
    pub(super) linear_guard_fallback_steps: usize,
    pub(super) unavailable_direct_fallback_steps: usize,
    pub(super) terms: usize,
    pub(super) maximum_relative_residual: Option<f64>,
    pub(super) maximum_rejected_relative_residual: Option<f64>,
}

impl SchurStatistics {
    pub(super) fn accept(&mut self, result: SchurSolve) {
        self.accepted_steps += 1;
        self.terms += result.terms;
        self.record_residual(result.relative_residual);
    }

    pub(super) fn linear_fallback(&mut self, evidence: RejectedSchurSolve) {
        self.fallback_steps += 1;
        self.linear_guard_fallback_steps += 1;
        self.record_rejection(evidence);
    }

    pub(super) fn retry(&mut self, evidence: RejectedSchurSolve) {
        self.retry_steps += 1;
        self.record_rejection(evidence);
    }

    pub(super) fn unavailable_fallback(&mut self, evidence: RejectedSchurSolve) {
        self.unavailable_direct_fallback_steps += 1;
        self.record_rejection(evidence);
    }

    fn record_rejection(&mut self, evidence: RejectedSchurSolve) {
        self.terms += evidence.terms;
        if let Some(residual) = evidence.relative_residual {
            self.record_residual(residual);
            self.maximum_rejected_relative_residual = Some(
                self.maximum_rejected_relative_residual
                    .map_or(residual, |maximum| maximum.max(residual)),
            );
        }
    }

    fn record_residual(&mut self, residual: f64) {
        self.maximum_relative_residual = Some(
            self.maximum_relative_residual
                .map_or(residual, |maximum| maximum.max(residual)),
        );
    }

    pub(super) fn quality(&self, solver: SparseCameraSolver) -> SparseCameraSolveQuality {
        match solver {
            SparseCameraSolver::DirectLlt => SparseCameraSolveQuality::Direct,
            SparseCameraSolver::IterativeSchur => {
                if self.accepted_steps == 0 && self.fallback_steps == 0 {
                    SparseCameraSolveQuality::Direct
                } else if self.fallback_steps > 0 {
                    SparseCameraSolveQuality::IterativeSchurWithDirectFallback
                } else {
                    SparseCameraSolveQuality::ResidualBoundedIterativeSchur
                }
            }
        }
    }
}

pub(super) fn solve_camera_step(
    context: RecomputedSchurContext<'_>,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
    policy: CameraStepPolicy,
) -> CameraStepOutcome {
    let CameraStepPolicy {
        damping,
        attempt_retained,
        relative_tolerance,
    } = policy;
    let RecomputedSchurContext {
        model,
        poses,
        points,
        gauge,
        loss,
        noise,
    } = context;
    match workspace.camera_solver() {
        SparseCameraSolver::DirectLlt => {
            return if solve_direct_camera_step(&mut workspace.core) {
                CameraStepOutcome::Step(CameraStep::Direct)
            } else {
                CameraStepOutcome::Rejected
            };
        }
        SparseCameraSolver::IterativeSchur if !attempt_retained => {
            return if solve_direct_camera_step(&mut workspace.core) {
                CameraStepOutcome::Step(CameraStep::Direct)
            } else {
                CameraStepOutcome::Rejected
            };
        }
        SparseCameraSolver::IterativeSchur => {}
    }
    let solver = workspace.camera_solver();
    let (attempt, recomputed) = {
        let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
        let CalibratedBackend::Retained(retained) = backend else {
            return CameraStepOutcome::Rejected;
        };
        let recomputed = retained.uses_recomputed_cross_rows();
        let attempt = if recomputed {
            retained.solve_recomputed(context, core, relative_tolerance)
        } else {
            retained.solve(model, core, relative_tolerance)
        };
        (attempt, recomputed)
    };
    match attempt {
        SchurAttempt::Accepted(result) => {
            let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
            let CalibratedBackend::Retained(retained) = backend else {
                return CameraStepOutcome::Rejected;
            };
            retained.copy_step_to(&mut core.step);
            CameraStepOutcome::Step(match solver {
                SparseCameraSolver::IterativeSchur => CameraStep::Iterative(result),
                SparseCameraSolver::DirectLlt => return CameraStepOutcome::Rejected,
            })
        }
        SchurAttempt::Rejected(evidence) => {
            if !workspace.core.iterative_direct_fallback {
                return CameraStepOutcome::IterativeRetry(evidence);
            }
            match solver {
                SparseCameraSolver::IterativeSchur => {
                    let materialized = if recomputed {
                        if workspace.ensure_direct_storage().is_err() {
                            return CameraStepOutcome::DirectFallbackUnavailable(evidence);
                        }
                        assemble_sparse(
                            model, poses, points, gauge, loss, noise, damping, false, fixed,
                            priors, workspace,
                        )
                        .valid
                    } else {
                        // A retained-only workspace keeps exact storage
                        // lazy. If the direct fallback storage cannot be
                        // prepared after an earlier accepted step, reject this
                        // linear attempt and report it explicitly instead of
                        // returning a non-atomic error after mutating caller
                        // state.
                        match materialize_retained_schur(model, workspace) {
                            Ok(materialized) => materialized,
                            Err(_) => {
                                return CameraStepOutcome::DirectFallbackUnavailable(evidence);
                            }
                        }
                    };
                    if materialized && solve_direct_camera_step(&mut workspace.core) {
                        CameraStepOutcome::Step(CameraStep::DirectFallback(evidence))
                    } else {
                        CameraStepOutcome::Rejected
                    }
                }
                SparseCameraSolver::DirectLlt => CameraStepOutcome::Rejected,
            }
        }
    }
}

fn solve_direct_camera_step(core: &mut SparseWorkspaceCore) -> bool {
    let factored = factor_camera_system(core);
    if !factored {
        return false;
    }
    solve_factored_camera_system(core)
}

pub(in crate::kernel::multi_view::sparse) fn materialize_retained_schur(
    model: ObservationModel<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> Result<bool, SparseBundleAdjustmentError> {
    if workspace.camera_solver() != SparseCameraSolver::IterativeSchur {
        return Ok(true);
    }
    workspace.ensure_direct_storage()?;
    let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
    let CalibratedBackend::Retained(retained) = backend else {
        return Ok(false);
    };
    Ok(retained.materialize_schur(model, core))
}
