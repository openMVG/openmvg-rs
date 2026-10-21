// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Projected Gauss-Newton / Levenberg-Marquardt polish of a BATA solution.
//!
//! The alternation of `solve.rs` converges linearly and slowly. Every constraint is a unit
//! direction `u` along a finite, nonzero forward `b = to - from`. On this halfline,
//! eliminating its nonnegative latent scale analytically gives the projection
//! of `u` off `b`: `f = u - (u.b / b.b) b`, so `|f| = sin(theta)`.
//! This stage uses Huber IRLS on the forward sine residual. Each LM attempt
//! minimizes the frozen surrogate `0.5 sum w |f|^2`; its loss curve differs
//! from chord-derived alternation weights and from the reported angular RMS.
//! A projected descending step can increase that separately reported RMS.
//! Points are eliminated by damped 3x3 Schur blocks, and the camera system is factored by the
//! sequential sparse Cholesky of [`super::direct`]. Without the `faer` feature, or if that
//! factorization is unavailable for a plan, conjugate gradients solve it matrix-free instead,
//! preconditioned by the scalar camera Laplacian of the alternation (factored densely when it fits)
//! or by 3x3 block-Jacobi.
//!
//! Camera 0 fixes the translation gauge and Nielsen damping regularizes the scale mode.
//!
//! Each linearization recomputes the weights, cut included: this is iteratively reweighted least
//! squares. The frozen surrogate gives the step and the gain ratio that drives the damping, and a
//! trial is accepted only if it does not raise the truncated robust objective
//! `sum beta rho(min(r, r_cut))` that the reweighting minimizes: `r` is the halfline residual,
//! `sin(theta)` in front of the camera and 1 behind it, `rho` the Huber loss, and `r_cut` the
//! residual at the hard cut. The surrogate lets an edge that turns backward leave the trial cost,
//! which sheds outlier edges inside the cut; this check keeps that freedom from paying for a step
//! that makes the fit worse. Without it, a polish that kept iterating after convergence accepted
//! a near-Gauss-Newton step that reversed four fifths of its edges and collapsed the layout. The
//! polish stops when an accepted step lowers the surrogate by less than
//! [`RELATIVE_COST_TOLERANCE`] of it, the function tolerance Ceres uses by default.
//!
//! ```text
//! linearize edges (weights refreshed) -> eliminate points -> sparse Cholesky -> back-substitute
//!        ^                                                                          |
//!        |-- gain-ratio accept if the robust objective does not rise; relative-cost stop <--'
//! ```

use nalgebra::{Matrix3, Vector3};
use openmvg_ba::numerics::refine::NielsenController;

use super::api::BataOptions;
use super::dense::{dense_fits, factor_reduced_dense, solve_with_factor};
#[cfg(feature = "faer")]
use super::direct::BlockCholesky;
use super::plan::GlobalPositioningPlan;
use super::robust::{cosine_cutoff, projected_huber_width};
use super::solve::assemble_camera_system;
use super::workspace::GlobalPositioningWorkspace;

const CG_RELATIVE_TOLERANCE: f64 = 1e-8;
const ABSOLUTE_DAMPING: f64 = 1e-12;
const ATTEMPTS_PER_STEP: usize = 6;
/// An accepted step that lowers the cost by less than this fraction of it ends the polish.
const RELATIVE_COST_TOLERANCE: f64 = 1e-6;

/// Linearized projection residual of one edge.
struct EdgeLinearization {
    residual: Vector3<f64>,
    jacobian: Matrix3<f64>,
    cosine: f64,
}

fn linearize(unit: &Vector3<f64>, baseline: Vector3<f64>) -> Option<EdgeLinearization> {
    let length = baseline.x.hypot(baseline.y).hypot(baseline.z);
    if !(length.is_finite() && length > 0.0) {
        return None;
    }
    let direction = baseline / length;
    let cosine = direction.dot(unit).clamp(-1.0, 1.0);
    let projector = Matrix3::identity() - direction * direction.transpose();
    let residual = projector * unit;
    let jacobian =
        -(Matrix3::identity() * cosine + direction * unit.transpose()) * projector / length;
    (cosine.is_finite()
        && residual
            .iter()
            .chain(jacobian.iter())
            .all(|value| value.is_finite()))
    .then_some(EdgeLinearization {
        residual,
        jacobian,
        cosine,
    })
}

/// Camera, point, and edge accessors shared by the passes below.
struct Problem<'a> {
    plan: &'a GlobalPositioningPlan,
    huber_delta: f64,
    cosine_cut: Option<f64>,
}

impl<'a> Problem<'a> {
    fn new(
        plan: &'a GlobalPositioningPlan,
        options: &BataOptions,
        allow_angular_cut: bool,
    ) -> Self {
        Self {
            plan,
            huber_delta: projected_huber_width(options.final_huber_delta_radians),
            cosine_cut: (allow_angular_cut && options.hard_cut_enabled())
                .then(|| cosine_cutoff(options.hard_cut_deg)),
        }
    }

    fn bearing_count(&self) -> usize {
        self.plan.bearings.len()
    }

    /// Robust weight of an edge from its residual: Huber on `sin(theta)`, zero past the hard cut
    /// or behind the camera.
    fn weight(&self, base_weight: f64, linearization: &EdgeLinearization) -> f64 {
        let sine = linearization.residual.norm();
        if linearization.cosine < 0.0
            || self
                .cosine_cut
                .is_some_and(|cutoff| linearization.cosine < cutoff)
        {
            return 0.0;
        }
        base_weight
            * if sine <= self.huber_delta {
                1.0
            } else {
                self.huber_delta / sine
            }
    }

    /// Truncated Huber objective of the halfline residual: `sin(theta)` in front of the camera
    /// and 1 behind it, capped at the residual of the hard cut. A collapsed baseline counts as
    /// cut.
    fn objective(&self, cameras: &[Vector3<f64>], points: &[Vector3<f64>]) -> f64 {
        let cut = self
            .cosine_cut
            .map_or(1.0, |cosine| (1.0 - cosine * cosine).max(0.0).sqrt());
        let loss = |unit: &Vector3<f64>, baseline: Vector3<f64>| {
            let residual = linearize(unit, baseline).map_or(cut, |lin| {
                if lin.cosine < 0.0 || self.cosine_cut.is_some_and(|cutoff| lin.cosine < cutoff) {
                    cut
                } else {
                    lin.residual.norm().min(cut)
                }
            });
            if residual <= self.huber_delta {
                0.5 * residual * residual
            } else {
                self.huber_delta * (residual - 0.5 * self.huber_delta)
            }
        };
        let bearings = self
            .plan
            .bearings
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                observation.base_weight
                    * loss(
                        &observation.ray,
                        self.bearing_baseline(index, cameras, points),
                    )
            })
            .sum::<f64>();
        let directions = self
            .plan
            .directions
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                edge.base_weight * loss(&edge.direction, self.direction_baseline(index, cameras))
            })
            .sum::<f64>();
        bearings + directions
    }

    fn bearing_baseline(
        &self,
        index: usize,
        cameras: &[Vector3<f64>],
        points: &[Vector3<f64>],
    ) -> Vector3<f64> {
        let observation = &self.plan.bearings[index];
        points[observation.point] - cameras[observation.camera]
    }

    fn direction_baseline(&self, index: usize, cameras: &[Vector3<f64>]) -> Vector3<f64> {
        let edge = &self.plan.directions[index];
        cameras[edge.target] - cameras[edge.source]
    }

    /// Frozen half-squared surrogate. A positive-weight edge whose baseline
    /// collapses or is not finite makes the trial infeasible. One whose
    /// baseline turns backward is cut, as [`Self::weight`] cuts it at the next
    /// linearization, instead of vetoing the step: a ray that left the forward
    /// halfline is an outlier, and an infinite cost would force a heavily
    /// damped first step that settles in a worse minimum; [`Self::objective`]
    /// keeps such a step from making the fit worse. Zero weights stay
    /// excluded, and angular cuts are not re-evaluated to lower a trial's
    /// objective.
    fn cost(
        &self,
        cameras: &[Vector3<f64>],
        points: &[Vector3<f64>],
        weights: (&[f64], &[f64]),
    ) -> f64 {
        let mut total = 0.0;
        for (index, observation) in self.plan.bearings.iter().enumerate() {
            let weight = weights.0[index];
            if weight == 0.0 {
                continue;
            }
            let Some(lin) = linearize(
                &observation.ray,
                self.bearing_baseline(index, cameras, points),
            ) else {
                return f64::INFINITY;
            };
            if lin.cosine < 0.0 {
                continue;
            }
            total += 0.5 * weight * lin.residual.norm_squared();
        }
        for (index, edge) in self.plan.directions.iter().enumerate() {
            let weight = weights.1[index];
            if weight == 0.0 {
                continue;
            }
            let Some(lin) = linearize(&edge.direction, self.direction_baseline(index, cameras))
            else {
                return f64::INFINITY;
            };
            if lin.cosine < 0.0 {
                continue;
            }
            total += 0.5 * weight * lin.residual.norm_squared();
        }
        total
    }
}

/// Outcome of a polish.
pub(super) struct PolishReport {
    pub(super) accepted_steps: usize,
}

pub(super) fn polish(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    options: &BataOptions,
    steps: usize,
    allow_angular_cut: bool,
) -> PolishReport {
    let problem = Problem::new(plan, options, allow_angular_cut);
    let mut damping = NielsenController::new();
    let mut accepted_steps = 0usize;
    let mut objective = problem.objective(&workspace.cameras, &workspace.points);
    for _ in 0..steps {
        let Some(cost) = linearize_system(&problem, workspace) else {
            break;
        };
        if !(cost.is_finite() && cost > 0.0) {
            break;
        }
        let mut accepted = false;
        let mut converged = false;
        for _ in 0..ATTEMPTS_PER_STEP {
            let Some(predicted) = solve_damped_step(&problem, workspace, damping.value()) else {
                damping.reject();
                continue;
            };
            let trial_objective =
                problem.objective(&workspace.trial_cameras, &workspace.trial_points);
            if trial_objective > objective {
                damping.reject();
                continue;
            }
            let trial_cost = problem.cost(
                &workspace.trial_cameras,
                &workspace.trial_points,
                (&workspace.bearing_weight, &workspace.direction_weight),
            );
            if damping.accept(cost - trial_cost, predicted) {
                std::mem::swap(&mut workspace.cameras, &mut workspace.trial_cameras);
                std::mem::swap(&mut workspace.points, &mut workspace.trial_points);
                objective = trial_objective;
                accepted = true;
                converged = cost - trial_cost <= RELATIVE_COST_TOLERANCE * cost;
                break;
            }
        }
        if !accepted {
            break;
        }
        accepted_steps += 1;
        if converged {
            break;
        }
    }
    PolishReport { accepted_steps }
}

/// Weights, per-edge Hessians, gradients, and the undamped camera diagonal blocks at the current
/// state. Returns the weighted cost.
fn linearize_system(
    problem: &Problem<'_>,
    workspace: &mut GlobalPositioningWorkspace,
) -> Option<f64> {
    let plan = problem.plan;
    workspace.polish_hessian.fill(Matrix3::zeros());
    workspace.polish_camera_block.fill(Matrix3::zeros());
    workspace.polish_camera_gradient.fill(Vector3::zeros());
    workspace.polish_point_hessian.fill(Matrix3::zeros());
    workspace.polish_point_gradient.fill(Vector3::zeros());
    workspace.point_weight.fill(0.0);
    let mut cost = 0.0;
    for (index, observation) in plan.bearings.iter().enumerate() {
        let baseline = problem.bearing_baseline(index, &workspace.cameras, &workspace.points);
        let Some(lin) = linearize(&observation.ray, baseline) else {
            workspace.bearing_weight[index] = 0.0;
            workspace.bearing_effective_weight[index] = 0.0;
            continue;
        };
        let weight = problem.weight(observation.base_weight, &lin);
        workspace.bearing_weight[index] = weight;
        // The scalar Laplacian that preconditions the camera system: `|J|^2 ~ 1 / |b|^2`.
        let effective = weight / baseline.norm_squared().max(1e-24);
        workspace.bearing_effective_weight[index] = effective;
        workspace.point_weight[observation.point] += effective;
        if weight == 0.0 {
            continue;
        }
        cost += 0.5 * weight * lin.residual.norm_squared();
        let hessian = lin.jacobian.transpose() * lin.jacobian * weight;
        let gradient = lin.jacobian.transpose() * lin.residual * weight;
        workspace.polish_hessian[index] = hessian;
        workspace.polish_camera_block[observation.camera] += hessian;
        workspace.polish_camera_gradient[observation.camera] -= gradient;
        workspace.polish_point_hessian[observation.point] += hessian;
        workspace.polish_point_gradient[observation.point] += gradient;
    }
    let offset = plan.bearings.len();
    for (index, edge) in plan.directions.iter().enumerate() {
        let baseline = problem.direction_baseline(index, &workspace.cameras);
        let Some(lin) = linearize(&edge.direction, baseline) else {
            workspace.direction_weight[index] = 0.0;
            workspace.direction_effective_weight[index] = 0.0;
            continue;
        };
        let weight = problem.weight(edge.base_weight, &lin);
        workspace.direction_weight[index] = weight;
        workspace.direction_effective_weight[index] = weight / baseline.norm_squared().max(1e-24);
        if weight == 0.0 {
            continue;
        }
        cost += 0.5 * weight * lin.residual.norm_squared();
        let hessian = lin.jacobian.transpose() * lin.jacobian * weight;
        let gradient = lin.jacobian.transpose() * lin.residual * weight;
        workspace.polish_hessian[offset + index] = hessian;
        workspace.polish_camera_block[edge.source] += hessian;
        workspace.polish_camera_block[edge.target] += hessian;
        workspace.polish_camera_gradient[edge.source] -= gradient;
        workspace.polish_camera_gradient[edge.target] += gradient;
    }
    cost.is_finite().then_some(cost)
}

fn damped(block: &Matrix3<f64>, damping: f64) -> Matrix3<f64> {
    let mut result = *block;
    for axis in 0..3 {
        result[(axis, axis)] += damping * block[(axis, axis)] + ABSOLUTE_DAMPING;
    }
    result
}

/// Solve one damped step into `trial_cameras` / `trial_points`; returns the predicted cost
/// reduction of the quadratic model, or `None` if the system could not be solved.
fn solve_damped_step(
    problem: &Problem<'_>,
    workspace: &mut GlobalPositioningWorkspace,
    damping: f64,
) -> Option<f64> {
    let plan = problem.plan;
    let cameras = plan.camera_count;
    // Damped point inverses.
    for point in 0..plan.point_count {
        workspace.polish_point_inverse[point] =
            damped(&workspace.polish_point_hessian[point], damping).try_inverse()?;
    }
    // Reduced right-hand side and block-Jacobi preconditioner (the Schur diagonal blocks).
    for camera in 0..cameras {
        workspace.pcg_rhs[camera] = -workspace.polish_camera_gradient[camera];
        workspace.polish_preconditioner[camera] =
            damped(&workspace.polish_camera_block[camera], damping);
    }
    for point in 0..plan.point_count {
        let inverse = workspace.polish_point_inverse[point];
        let gradient = workspace.polish_point_gradient[point];
        for &index in plan.point_bearing_range(point) {
            let camera = plan.bearings[index].camera;
            let hessian = workspace.polish_hessian[index];
            workspace.pcg_rhs[camera] -= hessian * inverse * gradient;
            workspace.polish_preconditioner[camera] -= hessian * inverse * hessian;
        }
    }
    workspace.pcg_rhs[0] = Vector3::zeros();
    #[cfg(feature = "faer")]
    if let Some(solved) = solve_direct(problem, workspace, damping) {
        return solved.then(|| apply_step(plan, workspace)).flatten();
    }
    for camera in 0..cameras {
        workspace.polish_preconditioner[camera] = if camera == 0 {
            Matrix3::zeros()
        } else {
            let block = workspace.polish_preconditioner[camera];
            block.try_inverse().or_else(|| diagonal_inverse(&block))?
        };
    }
    // The scalar Laplacian, factored densely, preconditions far better than the 3x3 blocks when
    // it fits (chain-like graphs need hundreds of block-Jacobi iterations).
    let dense = dense_fits(plan, workspace) && {
        assemble_camera_system(plan, workspace);
        factor_reduced_dense(plan, workspace, 0, damping).is_ok()
    };
    if !conjugate_gradients(problem, workspace, damping, dense) {
        return None;
    }
    apply_step(plan, workspace)
}

/// Move the trial cameras by `polish_step`, back-substitute the points, and return the predicted
/// cost reduction of the quadratic model.
fn apply_step(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
) -> Option<f64> {
    let cameras = plan.camera_count;
    for camera in 0..cameras {
        workspace.trial_cameras[camera] = workspace.cameras[camera] + workspace.polish_step[camera];
    }
    for point in 0..plan.point_count {
        let mut coupled = workspace.polish_point_gradient[point];
        for &index in plan.point_bearing_range(point) {
            let camera = plan.bearings[index].camera;
            coupled -= workspace.polish_hessian[index] * workspace.polish_step[camera];
        }
        let step = -(workspace.polish_point_inverse[point] * coupled);
        workspace.trial_points[point] = workspace.points[point] + step;
    }
    applied_model_reduction(plan, workspace)
}

/// Solve the damped camera system with the sparse Cholesky of [`BlockCholesky`], analyzed at the
/// first polish of the workspace. `None` when the factorization is unavailable for this plan, so
/// the caller falls back to conjugate gradients; otherwise whether the solve succeeded.
#[cfg(feature = "faer")]
fn solve_direct(
    problem: &Problem<'_>,
    workspace: &mut GlobalPositioningWorkspace,
    damping: f64,
) -> Option<bool> {
    let plan = problem.plan;
    if workspace
        .polish_direct
        .as_ref()
        .is_some_and(|cholesky| !cholesky.matches(plan))
    {
        workspace.polish_direct = None;
    }
    if workspace.polish_direct.is_none() {
        if workspace.polish_direct_unavailable {
            return None;
        }
        match BlockCholesky::analyze(plan) {
            Some(cholesky) => workspace.polish_direct = Some(Box::new(cholesky)),
            None => {
                workspace.polish_direct_unavailable = true;
                return None;
            }
        }
    }
    let mut direct = workspace.polish_direct.take()?;
    direct.clear();
    for camera in 1..plan.camera_count {
        direct.add_diagonal(
            camera,
            &damped(&workspace.polish_camera_block[camera], damping),
        );
    }
    let offset = problem.bearing_count();
    for index in 0..plan.directions.len() {
        direct.add_direction(index, &-workspace.polish_hessian[offset + index]);
    }
    let mut pair = 0usize;
    for point in 0..plan.point_count {
        let inverse = workspace.polish_point_inverse[point];
        let indices = plan.point_bearing_range(point);
        for (position, &left) in indices.iter().enumerate() {
            let projected = workspace.polish_hessian[left] * inverse;
            for &right in &indices[position..] {
                direct.add_point_pair(pair, &-(projected * workspace.polish_hessian[right]));
                pair += 1;
            }
        }
    }
    let solved = direct.solve(&workspace.pcg_rhs, &mut workspace.polish_step);
    workspace.polish_direct = Some(direct);
    Some(solved)
}

/// Undamped quadratic reduction for the camera/point differences actually
/// applied to trial storage. This remains valid for partial PCG solves and
/// roundoff, without using an exact damped-solve identity.
fn applied_model_reduction(
    plan: &GlobalPositioningPlan,
    workspace: &GlobalPositioningWorkspace,
) -> Option<f64> {
    let mut linear = 0.0;
    for camera in 0..plan.camera_count {
        let step = workspace.trial_cameras[camera] - workspace.cameras[camera];
        linear += workspace.polish_camera_gradient[camera].dot(&step);
    }
    for point in 0..plan.point_count {
        let step = workspace.trial_points[point] - workspace.points[point];
        linear += workspace.polish_point_gradient[point].dot(&step);
    }
    let mut quadratic = 0.0;
    for (index, observation) in plan.bearings.iter().enumerate() {
        let step = (workspace.trial_points[observation.point]
            - workspace.points[observation.point])
            - (workspace.trial_cameras[observation.camera] - workspace.cameras[observation.camera]);
        quadratic += step.dot(&(workspace.polish_hessian[index] * step));
    }
    let offset = plan.bearings.len();
    for (index, edge) in plan.directions.iter().enumerate() {
        let step = (workspace.trial_cameras[edge.target] - workspace.cameras[edge.target])
            - (workspace.trial_cameras[edge.source] - workspace.cameras[edge.source]);
        quadratic += step.dot(&(workspace.polish_hessian[offset + index] * step));
    }
    let predicted = -linear - 0.5 * quadratic;
    (predicted.is_finite() && predicted > 0.0).then_some(predicted)
}

fn diagonal_inverse(block: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let mut inverse = Matrix3::zeros();
    for axis in 0..3 {
        let entry = block[(axis, axis)];
        if !(entry.is_finite() && entry > 0.0) {
            return None;
        }
        inverse[(axis, axis)] = 1.0 / entry;
    }
    Some(inverse)
}

/// `out = S v` for the damped reduced camera system with camera 0 held fixed.
fn apply(
    problem: &Problem<'_>,
    workspace: &GlobalPositioningWorkspace,
    damping: f64,
    input: &[Vector3<f64>],
    output: &mut [Vector3<f64>],
) {
    let plan = problem.plan;
    for camera in 0..plan.camera_count {
        output[camera] = if camera == 0 {
            Vector3::zeros()
        } else {
            damped(&workspace.polish_camera_block[camera], damping) * input[camera]
        };
    }
    let offset = problem.bearing_count();
    for (index, edge) in plan.directions.iter().enumerate() {
        let hessian = workspace.polish_hessian[offset + index];
        if edge.source != 0 {
            output[edge.source] -= hessian * input[edge.target];
        }
        if edge.target != 0 {
            output[edge.target] -= hessian * input[edge.source];
        }
    }
    for point in 0..plan.point_count {
        let mut coupled = Vector3::zeros();
        for &index in plan.point_bearing_range(point) {
            coupled += workspace.polish_hessian[index] * input[plan.bearings[index].camera];
        }
        let projected = workspace.polish_point_inverse[point] * coupled;
        for &index in plan.point_bearing_range(point) {
            let camera = plan.bearings[index].camera;
            if camera != 0 {
                output[camera] -= workspace.polish_hessian[index] * projected;
            }
        }
    }
}

fn dot(left: &[Vector3<f64>], right: &[Vector3<f64>]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a.dot(b)).sum()
}

/// Preconditioned conjugate gradients on the reduced system; the step lands in `polish_step`.
/// Stops once the recursively updated residual falls below [`CG_RELATIVE_TOLERANCE`] of the
/// initial one. A step left at the iteration cap is returned as is, and the caller's gain-ratio
/// test on its applied model reduction decides whether it is accepted.
fn conjugate_gradients(
    problem: &Problem<'_>,
    workspace: &mut GlobalPositioningWorkspace,
    damping: f64,
    dense: bool,
) -> bool {
    let cameras = problem.plan.camera_count;
    workspace.polish_step.fill(Vector3::zeros());
    workspace.pcg_residual.copy_from_slice(&workspace.pcg_rhs);
    let initial = dot(&workspace.pcg_residual, &workspace.pcg_residual).sqrt();
    if !initial.is_finite() {
        return false;
    }
    if initial == 0.0 {
        return true;
    }
    if !precondition(problem, workspace, dense) {
        return false;
    }
    workspace
        .pcg_search
        .copy_from_slice(&workspace.pcg_preconditioned);
    let mut rho = dot(&workspace.pcg_residual, &workspace.pcg_preconditioned);
    if !(rho.is_finite() && rho > 0.0) {
        return false;
    }
    let maximum_iterations = (3 * cameras + 50).min(1000);
    for _ in 0..maximum_iterations {
        let search = std::mem::take(&mut workspace.pcg_search);
        let mut applied = std::mem::take(&mut workspace.pcg_applied);
        apply(problem, workspace, damping, &search, &mut applied);
        let curvature = dot(&search, &applied);
        if !(curvature.is_finite() && curvature > 0.0) {
            workspace.pcg_search = search;
            workspace.pcg_applied = applied;
            return false;
        }
        let alpha = rho / curvature;
        for camera in 0..cameras {
            workspace.polish_step[camera] += search[camera] * alpha;
            workspace.pcg_residual[camera] -= applied[camera] * alpha;
        }
        let residual_norm = dot(&workspace.pcg_residual, &workspace.pcg_residual).sqrt();
        workspace.pcg_search = search;
        workspace.pcg_applied = applied;
        if residual_norm <= CG_RELATIVE_TOLERANCE * initial {
            return true;
        }
        if !precondition(problem, workspace, dense) {
            return false;
        }
        let next = dot(&workspace.pcg_residual, &workspace.pcg_preconditioned);
        if !(next.is_finite() && next > 0.0) {
            return false;
        }
        let beta = next / rho;
        rho = next;
        for camera in 0..cameras {
            workspace.pcg_search[camera] =
                workspace.pcg_preconditioned[camera] + workspace.pcg_search[camera] * beta;
        }
    }
    // Iteration cap: the caller's gain-ratio test judges the partial step.
    true
}

/// `pcg_preconditioned = M^-1 pcg_residual`: the dense scalar Laplacian solve per coordinate, or
/// the 3x3 block-Jacobi inverse when the dense system does not fit.
fn precondition(
    problem: &Problem<'_>,
    workspace: &mut GlobalPositioningWorkspace,
    dense: bool,
) -> bool {
    if dense {
        let residual = std::mem::take(&mut workspace.pcg_residual);
        let mut output = std::mem::take(&mut workspace.pcg_preconditioned);
        let solved = solve_with_factor(problem.plan, workspace, &residual, &mut output).is_ok();
        workspace.pcg_residual = residual;
        workspace.pcg_preconditioned = output;
        return solved;
    }
    for camera in 0..problem.plan.camera_count {
        workspace.pcg_preconditioned[camera] =
            workspace.polish_preconditioner[camera] * workspace.pcg_residual[camera];
    }
    true
}

#[cfg(test)]
mod positioning_polish_contract_tests {
    use super::super::api::{GlobalPositioningProblem, PositioningObservation};
    use super::*;

    fn direction_plan(options: BataOptions) -> GlobalPositioningPlan {
        let observations = [PositioningObservation::relative_direction(
            0,
            1,
            Vector3::x(),
            1.0,
        )];
        GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: 2,
                point_count: 0,
                observations: &observations,
            },
            options,
        )
        .expect("one directed two-camera constraint")
    }

    fn configured_problem(plan: &GlobalPositioningPlan) -> Problem<'_> {
        Problem::new(plan, &plan.options, true)
    }

    #[test]
    fn positioning_polish_cutoff_is_monotone_and_disabled_beyond_budget() {
        let theta = 20.0_f64.to_radians();
        let baseline = Vector3::new(theta.cos(), theta.sin(), 0.0);
        let mut previously_admitted = false;
        for cutoff in [10.0, 30.0, 60.0, 90.0, 120.0, 180.0] {
            let plan = direction_plan(BataOptions {
                hard_cut_deg: cutoff,
                ..BataOptions::default()
            });
            let problem = configured_problem(&plan);
            let edge = linearize(&Vector3::x(), baseline).expect("finite nonzero baseline");
            let admitted = problem.weight(1.0, &edge) > 0.0;
            assert!(
                !previously_admitted || admitted,
                "increasing cutoff to {cutoff} removed a forward ray"
            );
            previously_admitted = admitted;
        }
        assert!(
            previously_admitted,
            "180 degrees must admit every forward ray"
        );
        let plan = direction_plan(BataOptions {
            max_outer_iterations: 2,
            hard_cut_start_iteration: 2,
            hard_cut_deg: 10.0,
            ..BataOptions::default()
        });
        let problem = configured_problem(&plan);
        let edge = linearize(&Vector3::x(), baseline).expect("finite nonzero baseline");
        assert!(
            problem.weight(1.0, &edge) > 0.0,
            "a cut delayed beyond the budget must remain disabled in polish"
        );
        let backward = linearize(&Vector3::x(), -Vector3::x()).expect("finite nonzero baseline");
        assert_eq!(
            problem.weight(1.0, &backward),
            0.0,
            "no angular cut still requires a forward halfline"
        );
    }

    #[test]
    fn positioning_polish_cuts_a_flipped_ray_and_rejects_a_collapsed_one() {
        let plan = direction_plan(BataOptions {
            hard_cut_deg: 180.0,
            ..BataOptions::default()
        });
        let problem = configured_problem(&plan);
        let flipped = [Vector3::zeros(), -Vector3::x()];
        assert_eq!(
            problem.cost(&flipped, &[], (&[], &[1.0])),
            0.0,
            "a ray that turned backward is cut, not counted as a perfect fit"
        );
        let collapsed = [Vector3::zeros(), Vector3::zeros()];
        assert!(
            problem.cost(&collapsed, &[], (&[], &[1.0])).is_infinite(),
            "an active constraint cannot collapse to a zero-length baseline"
        );
        for cameras in [flipped, collapsed] {
            assert_eq!(
                problem.cost(&cameras, &[], (&[], &[0.0])),
                0.0,
                "zero-weight constraints remain excluded before domain validation"
            );
        }
    }

    #[test]
    fn positioning_polish_exact_linear_surrogate_has_unit_gain_ratio() {
        let plan = direction_plan(BataOptions {
            hard_cut_deg: 90.0,
            initial_huber_delta_radians: 1.0,
            final_huber_delta_radians: 1.0,
            ..BataOptions::default()
        });
        let problem = configured_problem(&plan);
        let theta = 30.0_f64.to_radians();
        let probe_cameras = [
            Vector3::zeros(),
            Vector3::new(theta.cos(), theta.sin(), 0.0),
        ];
        // Read the production frozen squared-cost coefficient through an exact
        // projected residual norm sin(theta), without assuming its factor.
        let coefficient = problem.cost(&probe_cameras, &[], (&[], &[1.0])) / theta.sin().powi(2);
        let mut workspace = plan.try_new_workspace().expect("tiny workspace");
        workspace.cameras[0] = Vector3::zeros();
        workspace.cameras[1] = Vector3::x();
        workspace.polish_camera_block.fill(Matrix3::identity());
        workspace.polish_camera_gradient[0] = -Vector3::x();
        workspace.polish_camera_gradient[1] = Vector3::x();
        workspace.polish_hessian[0] = Matrix3::identity();
        workspace.direction_weight[0] = 1.0;
        workspace.direction_effective_weight[0] = 1.0;
        // This manually supplied r=x, J=I system is exactly LINEAR. The test
        // makes no claim that a nonlinear projected residual has gain ratio1.
        let predicted =
            solve_damped_step(&problem, &mut workspace, 0.5).expect("linear damped step");
        let delta = workspace.trial_cameras[1].x - workspace.cameras[1].x;
        let actual_linear_reduction = coefficient * (1.0 - (1.0 + delta).powi(2));
        let ratio = actual_linear_reduction / predicted;
        assert!(
            (ratio - 1.0).abs() <= 128.0 * f64::EPSILON,
            "linear gain ratio {ratio}"
        );
    }

    #[test]
    fn positioning_polish_full_applied_quadratic_matches_camera_point_edges() {
        let rays = [
            Vector3::new(0.35, 0.10, 2.8).normalize(),
            Vector3::new(-0.4, 0.25, 3.1).normalize(),
        ];
        let observations = [
            PositioningObservation::bearing(0, 0, rays[0], 0.7),
            PositioningObservation::bearing(1, 0, rays[1], 1.3),
            PositioningObservation::relative_direction(
                0,
                1,
                Vector3::new(1.0, 0.15, 0.05).normalize(),
                0.9,
            ),
        ];
        let options = BataOptions {
            hard_cut_deg: 180.0,
            initial_huber_delta_radians: 2.0,
            final_huber_delta_radians: 2.0,
            ..BataOptions::default()
        };
        let plan = GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: 2,
                point_count: 1,
                observations: &observations,
            },
            options,
        )
        .expect("two bearing edges and one relative direction");
        let problem = configured_problem(&plan);
        let mut workspace = plan
            .try_new_workspace()
            .expect("tiny camera-point workspace");
        workspace.cameras[0] = Vector3::zeros();
        workspace.cameras[1] = Vector3::new(1.0, 0.2, 0.0);
        workspace.points[0] = Vector3::new(0.2, -0.35, 3.4);
        linearize_system(&problem, &mut workspace).expect("finite frozen linearization");

        // An arbitrary inexact step, independent of PCG or its damped-solve identity.
        // Camera zero stays anchored; raw polish_step storage is deliberately unused.
        workspace.trial_cameras[0] = workspace.cameras[0];
        workspace.trial_cameras[1] = workspace.cameras[1] + Vector3::new(-0.008, -0.011, 0.004);
        workspace.trial_points[0] = workspace.points[0] + Vector3::new(0.006, 0.014, 0.005);
        let camera_step = workspace.trial_cameras[1] - workspace.cameras[1];
        let point_step = workspace.trial_points[0] - workspace.points[0];
        let projection = |unit: Vector3<f64>, baseline: Vector3<f64>| {
            unit - baseline * (unit.dot(&baseline) / baseline.dot(&baseline))
        };
        let mut expected = 0.0;
        for (index, observation) in plan.bearings.iter().enumerate() {
            let baseline = workspace.points[0] - workspace.cameras[observation.camera];
            let residual = projection(observation.ray, baseline);
            let h = 1.0e-6;
            let jacobian = Matrix3::from_fn(|row, column| {
                let mut shift = Vector3::zeros();
                shift[column] = h;
                (projection(observation.ray, baseline + shift)[row]
                    - projection(observation.ray, baseline - shift)[row])
                    / (2.0 * h)
            });
            let edge_step = point_step
                - if observation.camera == 0 {
                    Vector3::zeros()
                } else {
                    camera_step
                };
            let linear_trial = residual + jacobian * edge_step;
            expected += 0.5
                * workspace.bearing_weight[index]
                * (residual.norm_squared() - linear_trial.norm_squared());
        }
        let edge = plan.directions[0];
        let baseline = workspace.cameras[1] - workspace.cameras[0];
        let residual = projection(edge.direction, baseline);
        let jacobian = Matrix3::from_fn(|row, column| {
            let mut shift = Vector3::zeros();
            shift[column] = 1.0e-6;
            (projection(edge.direction, baseline + shift)[row]
                - projection(edge.direction, baseline - shift)[row])
                / 2.0e-6
        });
        let linear_trial = residual + jacobian * camera_step;
        expected += 0.5
            * workspace.direction_weight[0]
            * (residual.norm_squared() - linear_trial.norm_squared());
        assert!(
            expected > 0.0,
            "the prescribed inexact step descends its quadratic"
        );
        let actual =
            applied_model_reduction(&plan, &workspace).expect("finite positive model reduction");
        assert!(
            (actual - expected).abs() < 2.0e-10,
            "full camera-point reduction {actual}, independent edge quadratic {expected}"
        );
    }
}
