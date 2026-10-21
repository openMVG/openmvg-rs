// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared convergence and iteration bookkeeping.

use crate::numerics::refine::covariance::InformationDiagnostics;
use crate::numerics::refine::solver::{NormalEquations, gradient_inf_norm};
use crate::numerics::refine::{ConvergenceCriteria, ConvergenceCriterion, RefinementStatus};

/// Why a nonlinear solver rejected a trial step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepRejection {
    /// The normal equations could not be assembled at the current state.
    Linearization,
    /// The damped linear system had no usable solution.
    StepSolve,
    /// The trial state could not be formed or scored: its back-substitution, retraction or
    /// cost was not finite.
    Candidate,
    /// The trial state was valid but failed the gain-ratio test.
    InsufficientReduction,
}

/// Rejected trial steps, counted by [`StepRejection`] reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepRejections {
    /// Steps rejected because the normal equations could not be assembled.
    pub linearization: usize,
    /// Steps rejected because the damped linear system had no usable solution.
    pub step_solve: usize,
    /// Steps rejected because the trial state could not be formed or scored.
    pub candidate: usize,
    /// Steps rejected by the gain-ratio test.
    pub insufficient_reduction: usize,
}

impl StepRejections {
    /// Count one rejection for `reason`.
    pub fn record(&mut self, reason: StepRejection) {
        let count = match reason {
            StepRejection::Linearization => &mut self.linearization,
            StepRejection::StepSolve => &mut self.step_solve,
            StepRejection::Candidate => &mut self.candidate,
            StepRejection::InsufficientReduction => &mut self.insufficient_reduction,
        };
        *count += 1;
    }

    /// Rejections of every reason.
    #[must_use]
    pub const fn total(self) -> usize {
        self.linearization + self.step_solve + self.candidate + self.insufficient_reduction
    }

    /// Reason-wise sum with the rejections of another run.
    #[must_use]
    pub const fn plus(self, other: Self) -> Self {
        Self {
            linearization: self.linearization + other.linearization,
            step_solve: self.step_solve + other.step_solve,
            candidate: self.candidate + other.candidate,
            insufficient_reduction: self.insufficient_reduction + other.insufficient_reduction,
        }
    }
}

/// Iteration counts and convergence state of a staged Levenberg-Marquardt run.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefinementProgress {
    /// Iterations started, across all stages.
    pub iteration_count: usize,
    /// Steps accepted, across all stages.
    pub accepted_step_count: usize,
    /// Steps rejected, including failed linearizations, solves, and retractions.
    pub rejected_step_count: usize,
    /// The rejected steps by reason; their total is `rejected_step_count`.
    pub rejections: StepRejections,
    /// Steps accepted during the final stage.
    pub target_accepted_step_count: usize,
    /// Norm of the last accepted step.
    pub step_norm: f64,
    /// Criterion that stopped the run, if any.
    pub convergence: Option<ConvergenceCriterion>,
}

impl RefinementProgress {
    /// Whether the scaled gradient meets `criteria.gradient`; records the criterion if so.
    pub fn gradient_converged<const N: usize>(
        &mut self,
        equations: &NormalEquations<N>,
        cost: f64,
        criteria: ConvergenceCriteria,
    ) -> bool {
        if gradient_inf_norm(equations) <= criteria.gradient * (1.0 + cost.abs()) {
            self.convergence = Some(ConvergenceCriterion::Gradient);
            true
        } else {
            false
        }
    }

    /// Record an accepted step and, in the final stage, test the step and cost criteria.
    ///
    /// Returns `true` when a criterion is met.
    pub fn accept(
        &mut self,
        step_norm: f64,
        cost_reduction: f64,
        previous_cost: f64,
        target_stage: bool,
        criteria: ConvergenceCriteria,
    ) -> bool {
        self.accepted_step_count += 1;
        self.step_norm = step_norm;
        if !target_stage {
            return false;
        }
        self.target_accepted_step_count += 1;
        let reason = if step_norm <= criteria.step {
            Some(ConvergenceCriterion::Step)
        } else if cost_reduction <= criteria.relative_cost * (1.0 + previous_cost.abs()) {
            Some(ConvergenceCriterion::Cost)
        } else {
            None
        };
        self.convergence = reason;
        reason.is_some()
    }

    /// Record a step rejected for `reason`.
    pub fn reject(&mut self, reason: StepRejection) {
        self.rejected_step_count += 1;
        self.rejections.record(reason);
    }

    /// Final status: degenerate if `diagnostics` show rank loss, then converged, stalled
    /// (no accepted final-stage step), or out of iterations.
    pub fn status(self, diagnostics: InformationDiagnostics) -> RefinementStatus {
        if !diagnostics.is_full_rank() {
            RefinementStatus::Degenerate
        } else if let Some(reason) = self.convergence {
            RefinementStatus::Converged(reason)
        } else if self.iteration_count > 0 && self.target_accepted_step_count == 0 {
            RefinementStatus::Stalled
        } else {
            RefinementStatus::MaxIterations
        }
    }
}
