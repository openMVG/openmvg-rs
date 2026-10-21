// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Staged Levenberg-Marquardt loop for small fixed-size problems.
//!
//! Each problem supplies its linearization, cost and manifold retraction; the
//! loop owns the staged loss schedule, Nielsen damping and convergence tests.
//! The geometry homography and pose refiners and the multi-view point
//! triangulation share it.

use super::{NielsenController, RefinementProgress, StepRejection, final_stage_index, stages};
use crate::numerics::refine::solver::{NormalEquations, damped_step_system};
use crate::numerics::refine::{ConvergenceCriterion, Loss, Options};

/// Largest default step norm; longer steps are scaled down to it.
const DEFAULT_MAX_STEP_NORM: f64 = 0.5;

/// A nonlinear least-squares problem with `N` tangent parameters.
pub trait LmProblem<const N: usize> {
    /// Point on the model manifold.
    type State: Copy;

    /// Normal equations and robust cost at `state`, or `None` when the state
    /// cannot be linearized.
    fn linearize(&self, state: &Self::State, loss: Loss) -> Option<(NormalEquations<N>, f64)>;

    /// Scaled gradient infinity norm used by the gradient convergence test.
    fn gradient_inf_norm(&self, equations: &NormalEquations<N>) -> f64;

    /// Apply a tangent step, or `None` when the retraction fails.
    fn retract(&self, state: &Self::State, step: &[f64; N]) -> Option<Self::State>;

    /// Robust cost at `state`, or `None` when it cannot be evaluated.
    fn cost(&self, state: &Self::State, loss: Loss) -> Option<f64>;

    /// Damped Gauss-Newton step, or `None` when the damped system is singular.
    ///
    /// The default damps the diagonal and caps the step norm at 0.5.
    fn solve_step(&self, equations: &NormalEquations<N>, damping: f64) -> Option<[f64; N]> {
        damped_step_system(equations, damping, DEFAULT_MAX_STEP_NORM)
    }

    /// Step norm used by the step convergence test; Euclidean by default.
    fn step_norm(&self, _state: &Self::State, step: &[f64; N]) -> f64 {
        step.iter().map(|value| value * value).sum::<f64>().sqrt()
    }
}

/// Final state of [`minimize`] with its iteration and damping history.
pub struct LmSolution<State> {
    /// State after the last accepted step.
    pub state: State,
    /// Iteration counts, last step norm and convergence criterion.
    pub progress: RefinementProgress,
    /// Damping controller at exit.
    pub damping: NielsenController,
}

/// Run the staged Levenberg-Marquardt loop from `initial`.
pub fn minimize<const N: usize, P: LmProblem<N>>(
    problem: &P,
    initial: P::State,
    options: Options,
) -> LmSolution<P::State> {
    let mut state = initial;
    let mut damping = NielsenController::new();
    let schedule = stages(options);
    let target_stage = final_stage_index(&schedule);
    let mut progress = RefinementProgress::default();

    'optimization: for (stage_index, stage) in schedule.into_iter().enumerate() {
        for _ in 0..stage.max_iterations {
            progress.iteration_count += 1;
            let Some((equations, current_cost)) = problem.linearize(&state, stage.loss) else {
                progress.reject(StepRejection::Linearization);
                damping.reject();
                continue;
            };
            let is_target = stage_index == target_stage;
            if is_target
                && problem.gradient_inf_norm(&equations)
                    <= options.convergence.gradient * (1.0 + current_cost.abs())
            {
                progress.convergence = Some(ConvergenceCriterion::Gradient);
                break 'optimization;
            }
            let Some(step) = problem.solve_step(&equations, damping.value()) else {
                progress.reject(StepRejection::StepSolve);
                damping.reject();
                continue;
            };
            let Some(candidate) = problem.retract(&state, &step) else {
                progress.reject(StepRejection::Candidate);
                damping.reject();
                continue;
            };
            let Some(candidate_cost) = problem.cost(&candidate, stage.loss) else {
                progress.reject(StepRejection::Candidate);
                damping.reject();
                continue;
            };
            let reduction = current_cost - candidate_cost;
            let predicted_reduction = equations.quadratic_model_reduction(&step);
            if damping.accept(reduction, predicted_reduction) {
                let step_norm = problem.step_norm(&state, &step);
                state = candidate;
                if progress.accept(
                    step_norm,
                    reduction,
                    current_cost,
                    is_target,
                    options.convergence,
                ) {
                    break 'optimization;
                }
            } else {
                progress.reject(StepRejection::InsufficientReduction);
            }
        }
    }

    LmSolution {
        state,
        progress,
        damping,
    }
}
