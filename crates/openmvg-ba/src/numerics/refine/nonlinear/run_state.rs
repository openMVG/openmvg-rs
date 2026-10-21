// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free LM trajectory bookkeeping for bundle adjustment.

use super::{NielsenController, StepRejection, StepRejections};
use crate::numerics::refine::{ConvergenceCriteria, ConvergenceCriterion, RefinementStatus};

/// Scalar state shared by bundle-adjustment nonlinear drivers.
///
/// Linearization, step computation, retraction, and backend-specific
/// convergence confirmation remain in each driver. This type
/// owns only the LM trajectory and the evidence copied into public reports.
#[derive(Clone, Copy, Debug)]
pub struct LmRunState {
    damping: NielsenController,
    iterations: usize,
    accepted_steps: usize,
    rejections: StepRejections,
    target_accepted_steps: usize,
    step_norm: f64,
    last_relative_cost_reduction: Option<f64>,
    convergence: Option<ConvergenceCriterion>,
}

impl Default for LmRunState {
    fn default() -> Self {
        Self::new()
    }
}

impl LmRunState {
    pub const fn new() -> Self {
        Self::with_controller(NielsenController::new())
    }

    pub const fn with_initial_damping(damping: f64) -> Self {
        Self::with_controller(NielsenController::with_initial(damping))
    }

    const fn with_controller(damping: NielsenController) -> Self {
        Self {
            damping,
            iterations: 0,
            accepted_steps: 0,
            rejections: StepRejections {
                linearization: 0,
                step_solve: 0,
                candidate: 0,
                insufficient_reduction: 0,
            },
            target_accepted_steps: 0,
            step_norm: 0.0,
            last_relative_cost_reduction: None,
            convergence: None,
        }
    }

    pub fn begin_iteration(&mut self) {
        self.iterations += 1;
    }

    pub const fn damping(self) -> f64 {
        self.damping.value()
    }

    pub const fn iterations(self) -> usize {
        self.iterations
    }

    pub const fn accepted_steps(self) -> usize {
        self.accepted_steps
    }

    pub const fn rejected_steps(self) -> usize {
        self.rejections.total()
    }

    /// The rejected steps by reason.
    pub const fn rejections(self) -> StepRejections {
        self.rejections
    }

    pub const fn step_norm(self) -> f64 {
        self.step_norm
    }

    #[cfg(test)]
    pub const fn last_relative_cost_reduction(self) -> Option<f64> {
        self.last_relative_cost_reduction
    }

    pub const fn last_gain_ratio(self) -> Option<f64> {
        self.damping.last_gain_ratio()
    }

    /// Record a numerical failure, for `reason`, before a gain ratio can be evaluated.
    pub fn reject_failure(&mut self, reason: StepRejection) {
        self.rejections.record(reason);
        self.damping.reject();
    }

    /// Apply the gain-ratio decision and count a nonlinear rejection.
    ///
    /// A successful decision must be followed by the driver's already
    /// checked, infallible state commit and [`Self::record_accepted_step`].
    pub fn accept_gain(&mut self, actual_reduction: f64, predicted_reduction: f64) -> bool {
        if self.damping.accept(actual_reduction, predicted_reduction) {
            true
        } else {
            self.rejections.record(StepRejection::InsufficientReduction);
            false
        }
    }

    pub fn gradient_converged(
        &mut self,
        scaled_gradient_inf_norm: f64,
        cost: f64,
        criteria: ConvergenceCriteria,
    ) -> bool {
        if self.convergence.is_none()
            && scaled_gradient_inf_norm.is_finite()
            && scaled_gradient_inf_norm >= 0.0
            && cost.is_finite()
            && scaled_gradient_inf_norm <= criteria.gradient * (1.0 + cost.abs())
        {
            self.convergence = Some(ConvergenceCriterion::Gradient);
            true
        } else {
            false
        }
    }

    /// Record a committed step and return its target-stage stop candidate.
    ///
    /// Sparse drivers decide whether their linear solve is accurate enough to
    /// certify the returned reason. Dense drivers certify it immediately.
    pub fn record_accepted_step(
        &mut self,
        step_norm: f64,
        cost_reduction: f64,
        previous_cost: f64,
        target_stage: bool,
        criteria: ConvergenceCriteria,
    ) -> Option<ConvergenceCriterion> {
        self.accepted_steps += 1;
        self.step_norm = step_norm;
        if !target_stage {
            return None;
        }
        self.target_accepted_steps += 1;
        let relative_cost_reduction = cost_reduction / (1.0 + previous_cost.abs());
        self.last_relative_cost_reduction = Some(relative_cost_reduction);
        if step_norm <= criteria.step {
            Some(ConvergenceCriterion::Step)
        } else if relative_cost_reduction <= criteria.relative_cost {
            Some(ConvergenceCriterion::Cost)
        } else {
            None
        }
    }

    /// Store a target-stage convergence decision confirmed by the backend.
    pub fn certify_convergence(&mut self, convergence: Option<ConvergenceCriterion>) -> bool {
        self.convergence = convergence;
        convergence.is_some()
    }

    pub const fn status(self, well_constrained: bool) -> RefinementStatus {
        if !well_constrained {
            RefinementStatus::Degenerate
        } else if let Some(reason) = self.convergence {
            RefinementStatus::Converged(reason)
        } else if self.iterations > 0 && self.target_accepted_steps == 0 {
            RefinementStatus::Stalled
        } else {
            RefinementStatus::MaxIterations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejections_are_counted_by_reason() {
        let mut run = LmRunState::new();
        run.reject_failure(StepRejection::Linearization);
        run.reject_failure(StepRejection::StepSolve);
        run.reject_failure(StepRejection::StepSolve);
        run.reject_failure(StepRejection::Candidate);
        assert!(!run.accept_gain(-1.0, 1.0));
        let rejections = run.rejections();
        assert_eq!(
            rejections,
            StepRejections {
                linearization: 1,
                step_solve: 2,
                candidate: 1,
                insufficient_reduction: 1,
            }
        );
        assert_eq!(run.rejected_steps(), 5);
        assert_eq!(rejections.plus(rejections).total(), 10);
    }

    const CRITERIA: ConvergenceCriteria = ConvergenceCriteria::new(1e-3, 1e-2, 1e-4);

    fn assert_near(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs().max(1.0));
    }

    #[test]
    fn rejection_table_preserves_nielsen_history_and_gain_evidence() {
        let mut run = LmRunState::new();
        assert_eq!(run.damping(), 1e-4);
        assert_eq!(run.last_gain_ratio(), None);

        run.begin_iteration();
        run.reject_failure(StepRejection::StepSolve);
        assert_near(run.damping(), 2e-4);
        assert_eq!(run.rejected_steps(), 1);
        assert_eq!(run.last_gain_ratio(), None);

        run.begin_iteration();
        run.reject_failure(StepRejection::StepSolve);
        assert_near(run.damping(), 8e-4);
        assert_eq!(run.rejected_steps(), 2);

        run.begin_iteration();
        assert!(!run.accept_gain(-1.0, 1.0));
        assert_near(run.damping(), 6.4e-3);
        assert_eq!(run.rejected_steps(), 3);
        assert_eq!(run.last_gain_ratio(), Some(-1.0));

        run.begin_iteration();
        assert!(!run.accept_gain(1.0, -1.0));
        assert_near(run.damping(), 0.1024);
        assert_eq!(run.rejected_steps(), 4);
        assert_eq!(run.last_gain_ratio(), None);
        assert_eq!(run.iterations(), 4);
    }

    #[test]
    fn accepted_gain_resets_rejection_history_and_stage_bookkeeping_does_not() {
        let mut run = LmRunState::new();
        run.reject_failure(StepRejection::StepSolve);
        assert!(run.accept_gain(1.0, 1.0));
        let accepted_damping = 2e-4 / 3.0;
        assert_near(run.damping(), accepted_damping);

        run.record_accepted_step(1.0, 1.0, 2.0, false, CRITERIA);
        assert_near(run.damping(), accepted_damping);
        run.reject_failure(StepRejection::StepSolve);
        assert_near(run.damping(), 2.0 * accepted_damping);

        assert!(!run.accept_gain(-1.0, 1.0));
        assert_eq!(run.last_gain_ratio(), Some(-1.0));
        run.reject_failure(StepRejection::StepSolve);
        assert_eq!(run.last_gain_ratio(), None);
    }

    #[test]
    fn accepted_step_table_keeps_commit_and_certification_separate() {
        let mut run = LmRunState::with_initial_damping(1e-1);
        run.begin_iteration();
        assert!(run.accept_gain(2.0, 2.0));
        assert_near(run.damping(), 1e-1 / 3.0);
        assert_eq!(run.last_gain_ratio(), Some(1.0));
        assert_eq!(
            run.record_accepted_step(4.0, 2.0, 10.0, false, CRITERIA),
            None
        );
        assert_eq!(run.accepted_steps(), 1);
        assert_eq!(run.step_norm(), 4.0);
        assert_eq!(run.last_relative_cost_reduction(), None);
        assert_eq!(run.status(true), RefinementStatus::Stalled);

        run.begin_iteration();
        assert!(run.accept_gain(1.0, 2.0));
        let candidate = run.record_accepted_step(1e-3, 0.0, 8.0, true, CRITERIA);
        assert_eq!(run.last_relative_cost_reduction(), Some(0.0));
        assert_eq!(candidate, Some(ConvergenceCriterion::Step));
        // An unconfirmed sparse candidate does not terminate the run.
        assert_eq!(run.status(true), RefinementStatus::MaxIterations);
        assert!(run.certify_convergence(candidate));
        assert_eq!(
            run.status(true),
            RefinementStatus::Converged(ConvergenceCriterion::Step)
        );
        assert_eq!(run.status(false), RefinementStatus::Degenerate);
    }

    #[test]
    fn target_stop_table_prefers_step_then_cost() {
        let mut step = LmRunState::new();
        assert_eq!(
            step.record_accepted_step(1e-3, 0.0, 10.0, true, CRITERIA),
            Some(ConvergenceCriterion::Step)
        );

        let mut cost = LmRunState::new();
        assert_eq!(
            cost.record_accepted_step(1.0, 1e-3, 10.0, true, CRITERIA),
            Some(ConvergenceCriterion::Cost)
        );

        let mut running = LmRunState::new();
        assert_eq!(
            running.record_accepted_step(1.0, 2e-3, 10.0, true, CRITERIA),
            None
        );
    }

    #[test]
    fn gradient_stop_changes_no_iteration_outcome_counters_or_damping() {
        let mut run = LmRunState::new();
        for (gradient, cost) in [
            (0.0, f64::INFINITY),
            (f64::INFINITY, 0.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::INFINITY, f64::INFINITY),
            (-1.0, 0.0),
        ] {
            assert!(!run.gradient_converged(gradient, cost, CRITERIA));
        }
        run.begin_iteration();
        assert!(run.gradient_converged(1e-3, 0.0, CRITERIA));
        assert_eq!(run.accepted_steps(), 0);
        assert_eq!(run.rejected_steps(), 0);
        assert_eq!(run.damping(), 1e-4);
        assert_eq!(
            run.status(true),
            RefinementStatus::Converged(ConvergenceCriterion::Gradient)
        );
    }

    #[test]
    fn final_gradient_check_preserves_an_existing_step_or_cost_reason() {
        let mut run = LmRunState::new();
        run.certify_convergence(Some(ConvergenceCriterion::Cost));
        assert!(!run.gradient_converged(0.0, 0.0, CRITERIA));
        assert_eq!(
            run.status(true),
            RefinementStatus::Converged(ConvergenceCriterion::Cost)
        );
    }
}
