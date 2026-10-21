// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-allocation fixed-size Levenberg-Marquardt refinement.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Staged Levenberg-Marquardt: minimize                                 │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: initial state with N tangent parameters, an LmProblem<N> (linearize,
//!        cost, retract), Options (loss, max_iterations, convergence tolerances)
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Schedule the loss                                            │
//! └──────────────────────────────────────────────────────────────────────┘
//! stages(options) returns at most three stages. L2 and Huber run one stage.
//! Cauchy runs L2 for min(max_iterations / 4, 5) iterations, then Cauchy.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Linearize and test the gradient                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! NormalEquations H = Σ w Jᵀ J and g = Σ w Jᵀ r from whitened residuals, plus
//! the robust cost. In the last stage, stop if the problem's scaled ‖g‖∞ is
//! at most gradient · (1 + |cost|).
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Solve the damped step                                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! (H + λ diag H) δ = −g by Cholesky; the default solve_step caps ‖δ‖ at 0.5.
//! A system that is not positive definite or not finite rejects the iteration.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Retract and compare                                          │
//! └──────────────────────────────────────────────────────────────────────┘
//! candidate = retract(state, δ). Gain ratio ρ = (cost − candidate cost) /
//! predicted reduction of the undamped quadratic model.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 5: Update the damping (Nielsen) and decide                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! ρ > 0: accept and λ ← λ · max(1/3, 1 − (2ρ − 1)³). Otherwise keep the
//! state and λ ← λ · ν, with ν doubling per rejection. λ stays in [1e-12, 1e12].
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 6: Test stopping in the last stage                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! After an accepted step: ‖δ‖ ≤ step, or cost reduction ≤ relative_cost ·
//! (1 + |previous cost|), or the iteration budget is spent.
//!   │
//!   ▼
//! Output: LmSolution (state, RefinementProgress, final damping)
//! ```
//!
//! # Problem and method
//!
//! The module minimizes `Σᵢ ρ(‖rᵢ‖ / σᵢ)` by Gauss-Newton on a manifold with iteratively
//! reweighted least squares. A problem implements [`LmProblem`]: it forms the [`NormalEquations`]
//! from residuals and Jacobians, evaluates the cost, and retracts a tangent step onto its
//! manifold. [`NoiseModel`] states the standard deviations `σᵢ` that a problem divides out
//! before the [`Loss`] is evaluated, so loss scales are dimensionless. The losses are
//! `ρ(r) = r²/2` (L2), `r²/2` up to `δ` and `δ(|r| − δ/2)` beyond (Huber), and
//! `ln(1 + (r/s)²/2)` (Cauchy). Cauchy is not convex, which is why its schedule starts with L2.
//! (`RobustLoss::Cauchy` of a bundle problem writes the same curve `ln(1 + (r/scale)²)`, with
//! `scale = √2 · s`.)
//! [`LinearizedResidual`] is the neutral residual and Jacobian row, and [`NormalEquations`] can be
//! accumulated per worker and merged. [`ConvergenceCriteria`] documents that the gradient test
//! divides each tangent coordinate by the square root of its curvature, so equivalent
//! rescalings of a coordinate leave it unchanged.
//! [`ConvergenceCriteria::DEFAULT`] sets `1e-10` for gradient and step and `1e-12` for relative
//! cost; [`Options::default`] allows 30 iterations. [`NielsenController`] starts at `λ = 1e-4`.
//!
//! [`normal_equation_diagnostics`] reports the rank and reciprocal condition number of the
//! Gauss-Newton Hessian from a Jacobi eigenvalue sweep: an eigenvalue counts when it exceeds
//! `64 ε N λmax`, and the condition number is `√(λmin / λmax)` at full rank, zero otherwise.
//!
//! # Failure modes
//!
//! [`RefinementError`] covers invalid options, loss parameters, noise, observation counts and
//! non-finite or degenerate initial models, all checked before the loop starts; call
//! [`Options::try_validate`] and [`NoiseModel::try_validate`] to check them yourself. Inside the
//! loop, a failed
//! linearization, solve, retraction or cost evaluation, or a non-positive or non-finite
//! predicted reduction, rejects the step and raises `λ`. [`RefinementProgress::status`] returns
//! `Degenerate` first when the diagnostics are rank deficient, then `Converged` with the
//! criterion, `Stalled` when no step was accepted in the last stage, else `MaxIterations`.
//! [`RefinementReport::is_numerically_usable`] additionally requires a finite, non-increasing
//! objective and, for `Stalled` or `MaxIterations`, at least one accepted step.
//!
//! # Alternatives
//!
//! [`minimize`] suits small dense problems, such as refining one homography, one pose or one
//! point; the homography and pose refiners of `openmvg-geometry` use it. For bundle adjustment,
//! whose many points make the dense system large, use the crate's `NativeBundleAdjuster`, which
//! eliminates the points and solves the smaller camera system.

mod api;
pub(crate) mod covariance;
pub(crate) mod nonlinear;
pub(crate) mod solver;
pub use api::{
    ConvergenceCriteria, ConvergenceCriterion, Loss, NoiseModel, Options, RefinementError,
    RefinementReport, RefinementStatus, is_numerically_usable, validate_matrix,
    validate_observation_count, validate_points,
};
pub use covariance::InformationDiagnostics;
// Only the unstable covariance kernels take it; not part of the documented API.
#[doc(hidden)]
pub use covariance::UncertaintyModel;
pub use nonlinear::{
    LmProblem, LmSolution, NielsenController, RefinementProgress, StepRejection, StepRejections,
    minimize,
};
pub use solver::{
    LinearizedResidual, NormalEquations, damped_step_system, normal_equation_diagnostics,
};
