// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera-step solves over the direct or retained Schur system.

use crate::kernel::multi_view::optimizer::undamped_model_reduction;
use crate::kernel::multi_view::self_calibration::exact_model_reduction;
use crate::kernel::multi_view::sparse::api::{SparseCameraSolveQuality, SparseCameraSolver};
use crate::kernel::multi_view::sparse::direct::{
    factor_camera_system, solve_factored_camera_system,
};
use crate::kernel::multi_view::sparse::retained::{
    RecomputedSchurContext, RejectedSchurSolve, SchurAttempt, SchurSolve,
};
use crate::kernel::multi_view::sparse::workspace::{
    SelfCalibrationBackend, SparseSelfCalibrationWorkspace, SparseWorkspaceCore,
};

#[derive(Clone, Copy)]
pub(super) enum SelfCalibrationCameraStep {
    Direct,
    Iterative(SchurSolve),
    DirectFallback(RejectedSchurSolve),
}

pub(super) enum SelfCalibrationCameraStepOutcome {
    Step(SelfCalibrationCameraStep),
    Rejected,
    IterativeRetry(RejectedSchurSolve),
    DirectFallbackUnavailable(RejectedSchurSolve),
}

pub(super) fn self_calibration_model_reduction(
    workspace: &mut SparseSelfCalibrationWorkspace,
    point_model_reduction: f64,
    camera_step: SelfCalibrationCameraStep,
    damping: f64,
    point_damping_norm: f64,
) -> Option<f64> {
    if matches!(camera_step, SelfCalibrationCameraStep::Iterative(_)) {
        let SparseSelfCalibrationWorkspace {
            sparse, backend, ..
        } = workspace;
        let SelfCalibrationBackend::Iterative(retained) = backend else {
            return None;
        };
        let damped_reduction =
            retained.predicted_reduction(&sparse.gradient, point_model_reduction, &sparse.step);
        return damped_reduction.map(|damped_reduction| {
            undamped_model_reduction(
                damped_reduction,
                &sparse.diagonal,
                &sparse.step,
                damping,
                point_damping_norm,
            )
        });
    }
    Some(exact_model_reduction(
        point_model_reduction,
        &workspace.sparse.gradient,
        &workspace.sparse.diagonal,
        &workspace.sparse.step,
        damping,
        point_damping_norm,
    ))
}

pub(super) fn solve_self_calibration_camera_step(
    context: RecomputedSchurContext<'_>,
    workspace: &mut SparseSelfCalibrationWorkspace,
    attempt_retained: bool,
    relative_tolerance: Option<f64>,
) -> SelfCalibrationCameraStepOutcome {
    if workspace.camera_solver() == SparseCameraSolver::DirectLlt || !attempt_retained {
        return if solve_direct_camera_step(&mut workspace.sparse) {
            SelfCalibrationCameraStepOutcome::Step(SelfCalibrationCameraStep::Direct)
        } else {
            SelfCalibrationCameraStepOutcome::Rejected
        };
    }
    let attempt = {
        let SelfCalibrationBackend::Iterative(retained) = &mut workspace.backend else {
            return SelfCalibrationCameraStepOutcome::Rejected;
        };
        retained.solve(context, &workspace.sparse, relative_tolerance)
    };
    match attempt {
        SchurAttempt::Accepted(result) => {
            let SelfCalibrationBackend::Iterative(retained) = &workspace.backend else {
                return SelfCalibrationCameraStepOutcome::Rejected;
            };
            retained.copy_step_to(&mut workspace.sparse.step);
            SelfCalibrationCameraStepOutcome::Step(SelfCalibrationCameraStep::Iterative(result))
        }
        SchurAttempt::Rejected(evidence) => {
            if !workspace.sparse.iterative_direct_fallback {
                return SelfCalibrationCameraStepOutcome::IterativeRetry(evidence);
            }
            if workspace.ensure_direct_storage().is_err() {
                // Preserve both the retained-only memory policy and atomic
                // `Err` semantics: an unavailable lazy fallback is a rejected
                // linear attempt reported by the caller, not a late error
                // after prior accepted steps.
                return SelfCalibrationCameraStepOutcome::DirectFallbackUnavailable(evidence);
            }
            let SelfCalibrationBackend::Iterative(retained) = &mut workspace.backend else {
                return SelfCalibrationCameraStepOutcome::Rejected;
            };
            let materialized = retained.materialize_schur(context, &mut workspace.sparse);
            if materialized && solve_direct_camera_step(&mut workspace.sparse) {
                SelfCalibrationCameraStepOutcome::Step(SelfCalibrationCameraStep::DirectFallback(
                    evidence,
                ))
            } else {
                SelfCalibrationCameraStepOutcome::Rejected
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

#[derive(Default)]
pub(super) struct RetainedStatistics {
    pub(super) accepted_steps: usize,
    pub(super) fallback_steps: usize,
    pub(super) retry_steps: usize,
    pub(super) unavailable_direct_fallback_steps: usize,
    pub(super) iterations: usize,
    pub(super) maximum_relative_residual: Option<f64>,
    pub(super) maximum_rejected_relative_residual: Option<f64>,
}

impl RetainedStatistics {
    pub(super) fn accept(&mut self, result: SchurSolve) {
        self.accepted_steps += 1;
        self.iterations += result.terms;
        self.record_residual(result.relative_residual);
    }

    pub(super) fn fallback(&mut self, result: RejectedSchurSolve) {
        self.fallback_steps += 1;
        self.record_rejection(result);
    }

    pub(super) fn retry(&mut self, result: RejectedSchurSolve) {
        self.retry_steps += 1;
        self.record_rejection(result);
    }

    pub(super) fn unavailable_fallback(&mut self, result: RejectedSchurSolve) {
        self.unavailable_direct_fallback_steps += 1;
        self.record_rejection(result);
    }

    fn record_rejection(&mut self, result: RejectedSchurSolve) {
        self.iterations += result.terms;
        if let Some(residual) = result.relative_residual {
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
        if solver == SparseCameraSolver::DirectLlt
            || (self.accepted_steps == 0 && self.fallback_steps == 0)
        {
            SparseCameraSolveQuality::Direct
        } else if self.fallback_steps > 0 {
            SparseCameraSolveQuality::IterativeSchurWithDirectFallback
        } else {
            SparseCameraSolveQuality::ResidualBoundedIterativeSchur
        }
    }
}
