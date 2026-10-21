// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Matrix2x3, Matrix3, Point3};
use openmvg_core::camera::{Camera, CameraModel, PinholeCamera};
use openmvg_core::model::{CameraIntrinsics, WorldToCameraPose};

use crate::numerics::refine::StepRejections;
use crate::{BundleObservation, BundleProblem, IntrinsicVariables, ResidualScale, ResidualSpace};

/// Native linear-solver policy requested by the caller.
///
/// Each Levenberg-Marquardt step solves a linear system over every pose, intrinsic and point.
/// The native adjuster first eliminates the points: each point's 3 × 3 block is independent, so
/// the points drop out and leave a much smaller system over the camera parameters alone, the
/// reduced camera system (the Schur complement). The policy chooses how to solve that system;
/// the points then follow by back-substitution.
///
/// # When to change it
///
/// Keep [`Auto`](Self::Auto): it factors the camera system exactly on small and medium
/// problems, and switches to the iterative solver on large ones or when the factorization does
/// not fit the workspace budget. Choose [`Direct`](Self::Direct) to force the exact
/// factorization, raising [`SolveConfig::with_workspace_budget_bytes`] if it does not fit.
/// Choose [`Iterative`](Self::Iterative) to keep memory low even when the factorization would
/// fit; its steps are approximate, so it usually needs more iterations. Each choice is
/// deterministic, but the two take different steps, so their results can differ within the
/// convergence tolerances. [`SolveDiagnostics::linear_solver`] reports the one that ran.
///
/// Every solve ends with an exact rank check of the camera system, which
/// [`Solution::audit_for_commit`] needs, even when the iterative solver took the steps. Only
/// under an explicit [`SolveConfig::with_workspace_budget_bytes`] can `Auto` skip it, when the
/// check does not fit beside the iterative solver's storage: the report's observability is then
/// [`Observability::NotAssessed`](crate::Observability::NotAssessed) and the audit fails. Raise
/// the budget, or choose `Iterative`, which always keeps the check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearSolver {
    /// Choose the solver from the problem size and the workspace budget
    /// ([`SolveConfig::workspace_budget_bytes`]). A large problem, with hundreds of thousands of
    /// observations and hundreds of camera parameters, gets the iterative solver, which is
    /// faster there. Otherwise the exact factorization runs when it fits comfortably within the
    /// budget, and the iterative solver when it does not. The choice is made once, when the
    /// workspace is built, and kept for every iteration.
    #[default]
    Auto,
    /// Exact sparse factorization of the reduced camera system.
    Direct,
    /// Conjugate gradients on the reduced camera system, preconditioned with its per-camera
    /// diagonal blocks (block Jacobi), which needs less memory than the factorization on large
    /// problems.
    Iterative,
}

/// Execution policy that does not alter the problem's objective.
///
/// Start from [`SolveConfig::default`] and override fields with the `with_*` methods.
/// Values are checked by [`SolveConfig::validate`] when a solve starts. The defaults are 50
/// iterations, a function tolerance of `1e-6`, a gradient tolerance of `1e-10`, a parameter
/// tolerance of `1e-8`, [`LinearSolver::Auto`], and an automatic workspace budget
/// (`NativeBundleAdjuster::AUTO_WORKSPACE_BYTES`, 1 GiB).
///
/// It is plain data and `Copy`: the same `config` can go to a solve and then to
/// [`Solution::audit_for_commit`].
///
/// ```
/// use openmvg_ba::SolveConfig;
///
/// let config = SolveConfig::default().with_max_iterations(100);
/// let (for_solve, for_audit) = (config, config); // a copy each, no clone needed
/// assert_eq!(for_solve.max_iterations(), for_audit.max_iterations());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    max_iterations: usize,
    function_tolerance: f64,
    gradient_tolerance: f64,
    parameter_tolerance: f64,
    linear_solver: LinearSolver,
    workspace_budget_bytes: Option<u128>,
}

config_accessors!(SolveConfig {
    /// Maximum number of nonlinear iterations; must be positive.
    copy max_iterations, with_max_iterations: usize;
    /// Stop when the accepted relative objective reduction is this small; must be finite
    /// and positive.
    copy function_tolerance, with_function_tolerance: f64;
    /// Stop when the scaled gradient norm is at most `gradient_tolerance * (1 + objective)`;
    /// must be finite and positive.
    copy gradient_tolerance, with_gradient_tolerance: f64;
    /// Stop when the parameter step norm is this small; must be finite and positive.
    copy parameter_tolerance, with_parameter_tolerance: f64;
    /// Linear-solver policy of the native adjuster.
    copy linear_solver, with_linear_solver: LinearSolver;
});

impl SolveConfig {
    /// Bound the linear-solver workspace, in bytes; must be positive.
    ///
    /// The native adjuster selects its linear-solver strategy against this budget and
    /// refuses, before iterating, any solve whose workspace exceeds it.
    #[must_use]
    pub const fn with_workspace_budget_bytes(mut self, workspace_budget_bytes: u128) -> Self {
        self.workspace_budget_bytes = Some(workspace_budget_bytes);
        self
    }

    /// Linear-solver workspace budget in bytes.
    ///
    /// `None`, the default, selects the budget automatically: the native adjuster plans its
    /// linear-solver strategy against `NativeBundleAdjuster::AUTO_WORKSPACE_BYTES` and still
    /// runs a problem whose smallest strategy exceeds it.
    #[must_use]
    pub const fn workspace_budget_bytes(&self) -> Option<u128> {
        self.workspace_budget_bytes
    }

    /// Check the iteration limit, tolerances, and workspace budget.
    ///
    /// Bundle adjusters call this when a solve starts.
    pub fn validate(&self) -> Result<(), SolveError> {
        if self.max_iterations == 0 {
            return Err(SolveError::InvalidConfig("max_iterations must be positive"));
        }
        for (name, value) in [
            ("function_tolerance", self.function_tolerance),
            ("gradient_tolerance", self.gradient_tolerance),
            ("parameter_tolerance", self.parameter_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SolveError::InvalidConfig(name));
            }
        }
        if self.workspace_budget_bytes == Some(0) {
            return Err(SolveError::InvalidConfig(
                "workspace_budget_bytes must be positive",
            ));
        }
        Ok(())
    }
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            function_tolerance: 1.0e-6,
            gradient_tolerance: 1.0e-10,
            parameter_tolerance: 1.0e-8,
            linear_solver: LinearSolver::Auto,
            workspace_budget_bytes: None,
        }
    }
}

/// Reason a converged solve stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvergenceReason {
    /// The accepted relative objective reduction reached [`SolveConfig::function_tolerance`].
    Objective,
    /// The scaled gradient norm reached [`SolveConfig::gradient_tolerance`].
    Gradient,
    /// The parameter step norm reached [`SolveConfig::parameter_tolerance`].
    Step,
}

/// Numerical termination state returned by a bundle adjustment (BA) implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveStatus {
    /// A [`SolveConfig`] tolerance was satisfied.
    Converged(ConvergenceReason),
    /// [`SolveConfig::max_iterations`] ran out before any tolerance was satisfied.
    MaxIterations,
    /// No acceptable finite step could be found.
    Stalled,
    /// The final local information is rank deficient.
    Degenerate,
    /// The input state is returned unchanged, for example when a converged solve only
    /// raised the objective by round-off. Iteration and step counts retain work attempted
    /// before that return.
    Skipped,
}

/// Rank evidence from the final undamped normal equations: whether the observations determine
/// every free camera coordinate once the gauge is fixed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Observability {
    /// The gauge-reduced camera system has full rank. This is the usual report. Under
    /// [`Gauge::Free`](crate::Gauge::Free) with nothing that fixes the scale, the native adjuster
    /// holds the scale at its starting value, so a full rank there means the scale came from the
    /// start, not from the observations.
    FullRank,
    /// The only rank deficiency is the scale direction left free by
    /// [`Gauge::Free`](crate::Gauge::Free). The native adjuster reports it in one narrow case:
    /// when it could not assess the full camera system, it confirms the rank on a subset of the
    /// tracks instead, and under a free gauge that check leaves the scale free. Read it as
    /// [`Self::FullRank`] under a free gauge.
    OnlyExpectedGauge,
    /// The camera system is rank deficient beyond the expected gauge, or indefinite.
    RankDeficient,
    /// Rank was not assessed; [`Solution::audit_for_commit`] rejects this.
    #[default]
    NotAssessed,
}

/// Dimensions and weak-point evidence from the final undamped reduced system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservabilityDiagnostics {
    pose_rank: usize,
    pose_dimension: usize,
    weak_point_count: usize,
}

impl ObservabilityDiagnostics {
    /// For adjuster implementers: record the rank evidence of the reduced camera system.
    ///
    /// Fails with [`SolutionError::InvalidReport`] when `pose_dimension` is zero or `pose_rank`
    /// exceeds it.
    pub fn try_new(
        pose_rank: usize,
        pose_dimension: usize,
        weak_point_count: usize,
    ) -> Result<Self, SolutionError> {
        if pose_dimension == 0 || pose_rank > pose_dimension {
            return Err(SolutionError::InvalidReport);
        }
        Ok(Self {
            pose_rank,
            pose_dimension,
            weak_point_count,
        })
    }

    /// Rank of the undamped gauge-reduced pose system.
    ///
    /// The native adjuster reports [`Self::pose_dimension`] when the system has full rank and 0
    /// when it finds any deficiency; it does not count the missing directions.
    #[must_use]
    pub const fn pose_rank(self) -> usize {
        self.pose_rank
    }

    /// Free pose coordinates after gauge removal; positive. Four free poses under a fixed
    /// baseline, for example, have 4 × 6 = 24 coordinates, of which the gauge fixes 7, leaving 17.
    #[must_use]
    pub const fn pose_dimension(self) -> usize {
        self.pose_dimension
    }

    /// Weak points left in the solved system; the crate documentation's
    /// [weak and quarantined points](crate#weak-and-quarantined-points) section defines them.
    #[must_use]
    pub const fn weak_point_count(self) -> usize {
        self.weak_point_count
    }
}

/// Stationarity measured from a fresh linearization with zero LM damping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationarityAudit {
    scaled_gradient_inf_norm: f64,
}

impl StationarityAudit {
    /// For adjuster implementers: record the scaled gradient computed at the returned state with
    /// zero damping; it must be finite and non-negative.
    pub fn try_from_fresh_zero_damping(
        scaled_gradient_inf_norm: f64,
    ) -> Result<Self, SolutionError> {
        if !scaled_gradient_inf_norm.is_finite() || scaled_gradient_inf_norm < 0.0 {
            return Err(SolutionError::InvalidStationarityAudit);
        }
        Ok(Self {
            scaled_gradient_inf_norm,
        })
    }

    /// Largest gradient coordinate divided by the square root of its diagonal
    /// Gauss-Newton curvature; finite and non-negative.
    ///
    /// Read it against [`SolveConfig::gradient_tolerance`]: a solve that stopped on its gradient
    /// passes [`Solution::audit_for_commit`] only when this is at most the gradient tolerance
    /// times one plus the final objective.
    #[must_use]
    pub const fn scaled_gradient_inf_norm(self) -> f64 {
        self.scaled_gradient_inf_norm
    }
}

/// Deterministic diagnostics of the final state: step counts, rank, conditioning and
/// stationarity, which [`Solution::audit_for_commit`] reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveDiagnostics {
    accepted_step_count: usize,
    rejected_step_count: usize,
    step_rejections: StepRejections,
    stationarity: Option<StationarityAudit>,
    observability: Observability,
    weak_point_count: Option<usize>,
    quarantined_point_count: usize,
    noninvertible_observation_count: Option<usize>,
    observability_details: Option<ObservabilityDiagnostics>,
    linear_solver: Option<LinearSolver>,
}

impl SolveDiagnostics {
    /// For adjuster implementers: record final-state diagnostics; each argument is read back by
    /// the getter of the same name. Observability details stay unset until
    /// [`Self::with_observability_details`].
    #[must_use]
    pub const fn new(
        accepted_step_count: usize,
        rejected_step_count: usize,
        stationarity: StationarityAudit,
        observability: Observability,
        weak_point_count: usize,
        noninvertible_observation_count: usize,
    ) -> Self {
        Self {
            accepted_step_count,
            rejected_step_count,
            step_rejections: StepRejections {
                linearization: 0,
                step_solve: 0,
                candidate: 0,
                insufficient_reduction: 0,
            },
            stationarity: Some(stationarity),
            observability,
            weak_point_count: Some(weak_point_count),
            quarantined_point_count: 0,
            noninvertible_observation_count: Some(noninvertible_observation_count),
            observability_details: None,
            linear_solver: None,
        }
    }

    /// Record why the solve attempts rejected their trial steps.
    #[must_use]
    pub const fn with_step_rejections(mut self, step_rejections: StepRejections) -> Self {
        self.step_rejections = step_rejections;
        self
    }

    /// For adjuster implementers: record how many points the solver held out (quarantined).
    #[must_use]
    pub const fn with_quarantined_point_count(mut self, quarantined_point_count: usize) -> Self {
        self.quarantined_point_count = quarantined_point_count;
        self
    }

    /// Attach the reduced-system rank evidence.
    #[must_use]
    pub const fn with_observability_details(mut self, details: ObservabilityDiagnostics) -> Self {
        self.observability_details = Some(details);
        self
    }

    /// For adjuster implementers: record the linear solver that solved the camera system,
    /// [`LinearSolver::Direct`] or [`LinearSolver::Iterative`].
    #[must_use]
    pub const fn with_linear_solver(mut self, linear_solver: LinearSolver) -> Self {
        self.linear_solver = Some(linear_solver);
        self
    }

    /// The linear solver that solved the camera system: [`LinearSolver::Direct`] or
    /// [`LinearSolver::Iterative`], so with [`LinearSolver::Auto`] this is the one it chose
    /// from the problem size and the memory budget. The native adjuster reports `Direct` for
    /// two-view problems, which it solves exactly. `None` when the adjuster does not report it.
    #[must_use]
    pub const fn linear_solver(self) -> Option<LinearSolver> {
        self.linear_solver
    }

    /// Nonlinear steps accepted by the solve attempts, including steps discarded when the
    /// adjuster returned the input state instead ([`SolveStatus::Skipped`]).
    #[must_use]
    pub const fn accepted_step_count(self) -> usize {
        self.accepted_step_count
    }

    /// Trial steps rejected.
    #[must_use]
    pub const fn rejected_step_count(self) -> usize {
        self.rejected_step_count
    }

    /// The rejected trial steps by reason, as recorded by [`Self::with_step_rejections`];
    /// all zero when the solver did not record them.
    #[must_use]
    pub const fn step_rejections(self) -> StepRejections {
        self.step_rejections
    }

    /// Stationarity of the returned state; `None` when the adjuster returned the input state
    /// instead of its result and did not assess it again.
    #[must_use]
    pub const fn stationarity(self) -> Option<StationarityAudit> {
        self.stationarity
    }

    /// Rank classification of the final undamped system.
    #[must_use]
    pub const fn observability(self) -> Observability {
        self.observability
    }

    /// Weak points left in the final solved system, when assessed:
    /// [`Solution::audit_for_commit`] rejects a missing or nonzero count. The crate
    /// documentation's [weak and quarantined points](crate#weak-and-quarantined-points) section
    /// defines them.
    #[must_use]
    pub const fn weak_point_count(self) -> Option<usize> {
        self.weak_point_count
    }

    /// Weak points the solver quarantined: held out of the solve and left at their input
    /// position, as the crate documentation's
    /// [weak and quarantined points](crate#weak-and-quarantined-points) section explains.
    /// [`Solution::quarantined_points`] lists them; do not trust their landmarks.
    #[must_use]
    pub const fn quarantined_point_count(self) -> usize {
        self.quarantined_point_count
    }

    /// Final observations outside the camera model's unambiguous inverse domain.
    ///
    /// `None` means the returned state was not reassessed. A nonzero count does not
    /// invalidate the objective: robust outliers may remain.
    #[must_use]
    pub const fn noninvertible_observation_count(self) -> Option<usize> {
        self.noninvertible_observation_count
    }

    /// Reduced-system rank evidence, when the implementation measured it.
    #[must_use]
    pub const fn observability_details(self) -> Option<ObservabilityDiagnostics> {
        self.observability_details
    }

    /// Discard certificates whose evaluated geometry is no longer the returned state.
    #[cfg(feature = "faer")]
    fn invalidate_returned_state(&mut self) {
        self.stationarity = None;
        self.observability = Observability::NotAssessed;
        self.weak_point_count = None;
        self.noninvertible_observation_count = None;
        self.observability_details = None;
    }
}

/// What one solve did: its status, its objectives and errors before and after, and the
/// diagnostics of the returned state. [Accepting a result](crate#accepting-a-result) lists what
/// to check before using the solution.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveReport {
    status: SolveStatus,
    iteration_count: usize,
    initial_objective: f64,
    final_objective: f64,
    initial_rmse: f64,
    final_rmse: f64,
    diagnostics: SolveDiagnostics,
    unconstrained_intrinsics: Vec<usize>,
    observation_count: Option<usize>,
    inlier_count: Option<usize>,
    final_inlier_rmse: Option<f64>,
}

impl SolveReport {
    /// For adjuster implementers: record one solve's evidence; each argument is read back by the
    /// getter of the same name.
    ///
    /// Fails with [`SolutionError::InvalidReport`] when an objective or RMSE is negative or
    /// not finite, or when more steps were accepted than iterations ran.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        status: SolveStatus,
        iteration_count: usize,
        initial_objective: f64,
        final_objective: f64,
        initial_rmse: f64,
        final_rmse: f64,
        diagnostics: SolveDiagnostics,
    ) -> Result<Self, SolutionError> {
        if [initial_objective, final_objective, initial_rmse, final_rmse]
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0)
        {
            return Err(SolutionError::InvalidReport);
        }
        if diagnostics.accepted_step_count > iteration_count {
            return Err(SolutionError::InvalidReport);
        }
        Ok(Self {
            status,
            iteration_count,
            initial_objective,
            final_objective,
            initial_rmse,
            final_rmse,
            diagnostics,
            unconstrained_intrinsics: Vec::new(),
            observation_count: None,
            inlier_count: None,
            final_inlier_rmse: None,
        })
    }

    /// Record the free intrinsic groups the solve held at their input values because their
    /// observations did not constrain them. The list describes the solve attempt, so it stays
    /// when the adjuster returns the input state instead of its result.
    #[must_use]
    pub fn with_unconstrained_intrinsics(mut self, intrinsics: Vec<usize>) -> Self {
        self.unconstrained_intrinsics = intrinsics;
        self
    }

    /// Accumulate the work of an earlier solve. If point domains differ, final lifting
    /// recomputes both metrics on the final retained domain before returning the report.
    #[cfg(feature = "faer")]
    pub(crate) fn continued_from(mut self, first: &Self) -> Self {
        self.initial_objective = first.initial_objective;
        self.initial_rmse = first.initial_rmse;
        self.iteration_count += first.iteration_count;
        self.diagnostics.accepted_step_count += first.diagnostics.accepted_step_count;
        self.diagnostics.rejected_step_count += first.diagnostics.rejected_step_count;
        self.diagnostics.step_rejections = self
            .diagnostics
            .step_rejections
            .plus(first.diagnostics.step_rejections);
        self
    }

    /// Termination state.
    #[must_use]
    pub const fn status(&self) -> SolveStatus {
        self.status
    }

    /// Nonlinear iterations run.
    #[must_use]
    pub const fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    /// Robust objective at the original input state, summed over the same observations
    /// used for [`Self::final_objective`]. Native quarantined points are excluded from both;
    /// [`Solution::quarantined_points`] and [`Self::observation_count`] identify this scope.
    #[must_use]
    pub const fn initial_objective(&self) -> f64 {
        self.initial_objective
    }

    /// Robust objective at the returned state on the same domain as the initial objective.
    #[must_use]
    pub const fn final_objective(&self) -> f64 {
        self.final_objective
    }

    /// Root-mean-square unweighted observation residual norm at the original input state,
    /// in pixels or radians, on the same retained domain as [`Self::final_rmse`].
    #[must_use]
    pub const fn initial_rmse(&self) -> f64 {
        self.initial_rmse
    }

    /// Root-mean-square observation residual norm at the returned state, without robust
    /// weighting, in the problem's residual unit.
    #[must_use]
    pub const fn final_rmse(&self) -> f64 {
        self.final_rmse
    }

    /// Observations contributing to both initial and final metrics.
    ///
    /// A standalone report is unbound (`None`). [`Solution`] binds the count to its
    /// problem and excludes observations of explicitly quarantined points.
    #[must_use]
    pub const fn observation_count(&self) -> Option<usize> {
        self.observation_count
    }

    /// Retained observations whose final residual is within the robust-loss scale: every
    /// retained observation under [`RobustLoss::Squared`](crate::RobustLoss::Squared). `None`
    /// until a [`Solution`] is built against its problem.
    #[must_use]
    pub const fn inlier_count(&self) -> Option<usize> {
        self.inlier_count
    }

    /// Root mean square of the final residuals within the robust-loss scale, in pixels or
    /// radians; `None` before a [`Solution`] is built or when no observation is within the scale.
    ///
    /// [`Self::final_rmse`] counts every retained observation, so a few wrong matches dominate it
    /// even when a robust loss kept them from pulling the solution. This figure keeps only the
    /// observations within the loss scale, the ones [`Self::inlier_count`] counts, and equals
    /// [`Self::final_rmse`] under [`RobustLoss::Squared`](crate::RobustLoss::Squared). Under a
    /// robust loss it cannot exceed the scale, by construction, so judge the fit from
    /// [`Self::inlier_count`] against [`Self::observation_count`] first.
    #[must_use]
    pub const fn final_inlier_rmse(&self) -> Option<f64> {
        self.final_inlier_rmse
    }

    /// Final-state diagnostics.
    #[must_use]
    pub const fn diagnostics(&self) -> SolveDiagnostics {
        self.diagnostics
    }

    /// Free intrinsic groups held by the solve attempt because their observations did not
    /// constrain them, in ascending order. The list stays when the adjuster returns the input
    /// state instead of its result.
    #[must_use]
    pub fn unconstrained_intrinsics(&self) -> &[usize] {
        &self.unconstrained_intrinsics
    }
}

fn validate_solution_blocks(
    problem: &BundleProblem,
    poses: &[WorldToCameraPose],
    intrinsics: &[CameraIntrinsics],
    points: &[Point3<f64>],
) -> Result<(), SolutionError> {
    if poses.len() != problem.poses().len()
        || intrinsics.len() != problem.intrinsics().len()
        || points.len() != problem.points().len()
    {
        return Err(SolutionError::ShapeMismatch);
    }
    for (index, (pose, initial)) in poses.iter().zip(problem.poses()).enumerate() {
        pose.validate()
            .map_err(|_| SolutionError::InvalidPose { pose: index })?;
        if (!initial.variables().rotation() && pose.rotation() != initial.pose().rotation())
            || (!initial.variables().translation() && pose.center() != initial.pose().center())
        {
            return Err(SolutionError::FixedPoseChanged { pose: index });
        }
    }
    for (intrinsic, (data, initial)) in intrinsics.iter().zip(problem.intrinsics()).enumerate() {
        data.validate_camera_domain()
            .map_err(|_| SolutionError::InvalidIntrinsic { intrinsic })?;
        if data.dimensions() != initial.intrinsics().dimensions()
            || data.is_fixed() != initial.intrinsics().is_fixed()
            || !intrinsic_variables_respected(
                initial.intrinsics().camera(),
                data.camera(),
                initial.variables(),
            )
        {
            return Err(SolutionError::FixedIntrinsicChanged { intrinsic });
        }
    }
    for (point, (position, initial)) in points.iter().zip(problem.points()).enumerate() {
        if !position.coords.iter().all(|value| value.is_finite()) {
            return Err(SolutionError::InvalidPoint { point });
        }
        if initial.is_fixed() && position != initial.position() {
            return Err(SolutionError::FixedPointChanged { point });
        }
    }
    if let crate::Gauge::FixedBaseline {
        first,
        second,
        distance,
    } = problem.gauge()
    {
        if &poses[first] != problem.poses()[first].pose() {
            return Err(SolutionError::GaugeChanged);
        }
        let actual = center_distance(poses[first].center(), poses[second].center());
        let baseline_scale = actual.max(distance);
        let relative_tolerance = 512.0 * f64::EPSILON * baseline_scale;
        let rounding_bound = center_rounding_bound(poses[first].center(), poses[second].center());
        let error = (actual - distance).abs();
        if !actual.is_finite() || !error.is_finite() {
            return Err(SolutionError::GaugeChanged);
        }
        if actual <= 0.0 || error > relative_tolerance {
            // A few coordinate ulps can explain an unattainable requested metric
            // baseline, but they do not authorize a less precise gauge certificate.
            return Err(if error <= relative_tolerance + rounding_bound {
                SolutionError::GaugePrecisionLimited
            } else {
                SolutionError::GaugeChanged
            });
        }
    }
    Ok(())
}

fn center_distance(first: &Point3<f64>, second: &Point3<f64>) -> f64 {
    let offset = second - first;
    let scale = offset.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale > 0.0 {
        scale
            * offset
                .iter()
                .map(|value| (value / scale).powi(2))
                .sum::<f64>()
                .sqrt()
    } else {
        0.0
    }
}

fn center_rounding_bound(first: &Point3<f64>, second: &Point3<f64>) -> f64 {
    let mut spacing = [0.0; 3];
    for axis in 0..3 {
        spacing[axis] = coordinate_spacing(first[axis]) + coordinate_spacing(second[axis]);
    }
    let scale = spacing.iter().copied().fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    // Two center roundings per axis and a small margin for inverse frame conversion.
    2.0 * scale
        * spacing
            .iter()
            .map(|value| (value / scale).powi(2))
            .sum::<f64>()
            .sqrt()
}

fn coordinate_spacing(value: f64) -> f64 {
    let value = value.abs();
    let upper = value.next_up();
    if upper.is_finite() {
        upper - value
    } else {
        value - value.next_down()
    }
}

fn intrinsic_variables_respected(
    initial: &Camera,
    candidate: &Camera,
    variables: IntrinsicVariables,
) -> bool {
    match (initial, candidate) {
        (Camera::Pinhole(initial), Camera::Pinhole(candidate)) => {
            pinhole_variables_respected(initial, candidate, variables)
        }
        (Camera::BrownConrady(initial), Camera::BrownConrady(candidate)) => {
            pinhole_variables_respected(initial.pinhole(), candidate.pinhole(), variables)
                && std::mem::discriminant(initial.radial_distortion())
                    == std::mem::discriminant(candidate.radial_distortion())
                && std::mem::discriminant(initial.tangential_distortion())
                    == std::mem::discriminant(candidate.tangential_distortion())
                && (variables.radial()
                    || initial.radial_distortion() == candidate.radial_distortion())
                && (variables.tangential()
                    || initial.tangential_distortion() == candidate.tangential_distortion())
        }
        (Camera::Fisheye(initial), Camera::Fisheye(candidate)) => {
            pinhole_variables_respected(initial.pinhole(), candidate.pinhole(), variables)
                && std::mem::discriminant(initial.fisheye_distortion())
                    == std::mem::discriminant(candidate.fisheye_distortion())
                && (variables.radial()
                    || initial.fisheye_distortion() == candidate.fisheye_distortion())
        }
        (Camera::Spherical(initial), Camera::Spherical(candidate)) => initial == candidate,
        _ => false,
    }
}

fn pinhole_variables_respected(
    initial: &PinholeCamera,
    candidate: &PinholeCamera,
    variables: IntrinsicVariables,
) -> bool {
    (variables.focal()
        || (initial.focal_x() == candidate.focal_x() && initial.focal_y() == candidate.focal_y()))
        && (variables.principal_point()
            || (initial.principal_x() == candidate.principal_x()
                && initial.principal_y() == candidate.principal_y()))
}

/// Owned proposal returned by a bundle-adjustment implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    poses: Vec<WorldToCameraPose>,
    intrinsics: Vec<CameraIntrinsics>,
    points: Vec<Point3<f64>>,
    residual_space: ResidualSpace,
    report: SolveReport,
    quarantined_points: Vec<usize>,
}

impl Solution {
    /// For adjuster implementers: check the returned state against the problem's shape and its
    /// fixed blocks, and build the solution.
    ///
    /// A report claiming quarantined points must use
    /// [`Self::try_new_with_quarantined_points`] to supply their identities.
    pub fn try_new(
        problem: &BundleProblem,
        poses: Vec<WorldToCameraPose>,
        intrinsics: Vec<CameraIntrinsics>,
        points: Vec<Point3<f64>>,
        report: SolveReport,
    ) -> Result<Self, SolutionError> {
        Self::try_new_with_quarantined_points(
            problem,
            poses,
            intrinsics,
            points,
            report,
            Vec::new(),
        )
    }

    /// For adjuster implementers: build the solution of a solve that held some points out
    /// (quarantined them).
    ///
    /// The IDs are indices in `problem.points()`; they must be distinct, in bounds, and as many
    /// as the report's quarantine count, and are stored in ascending order. A quarantined point
    /// keeps its exact input position. With quarantined points, the initial and final objective
    /// and RMSE are recomputed on the remaining observations, so the improvement compares the
    /// same observations; without any, the adjuster's own metrics are kept. The final rank and
    /// stationarity must describe the system without the quarantined points.
    pub fn try_new_with_quarantined_points(
        problem: &BundleProblem,
        poses: Vec<WorldToCameraPose>,
        intrinsics: Vec<CameraIntrinsics>,
        points: Vec<Point3<f64>>,
        mut report: SolveReport,
        mut quarantined_points: Vec<usize>,
    ) -> Result<Self, SolutionError> {
        validate_solution_blocks(problem, &poses, &intrinsics, &points)?;
        quarantined_points.sort_unstable();
        if quarantined_points.len() != report.diagnostics.quarantined_point_count
            || quarantined_points.windows(2).any(|pair| pair[0] == pair[1])
            || quarantined_points.iter().any(|&point| {
                point >= points.len() || points[point] != *problem.points()[point].position()
            })
        {
            return Err(SolutionError::InvalidQuarantine);
        }
        let mut active = vec![true; points.len()];
        for &point in &quarantined_points {
            active[point] = false;
        }
        report.observation_count = Some(
            problem
                .observations()
                .iter()
                .filter(|observation| active[observation.point()])
                .count(),
        );
        let mut solution = Self {
            poses,
            intrinsics,
            points,
            residual_space: problem.residual_space(),
            report,
            quarantined_points,
        };
        if !solution.quarantined_points.is_empty() {
            let initial = active_objective_metrics(problem, &active, None)?;
            let final_metrics = active_objective_metrics(problem, &active, Some(&solution))?;
            solution.report.initial_objective = initial.0;
            solution.report.initial_rmse = initial.1;
            solution.report.final_objective = final_metrics.0;
            solution.report.final_rmse = final_metrics.1;
        }
        solution.refresh_inlier_metrics(problem, &active);
        Ok(solution)
    }

    /// Recompute [`SolveReport::inlier_count`] and [`SolveReport::final_inlier_rmse`] at this
    /// state, over the observations of `active` points. A state that cannot be evaluated leaves
    /// both unset; the other report checks reject it.
    fn refresh_inlier_metrics(&mut self, problem: &BundleProblem, active: &[bool]) {
        let (count, rmse) = inlier_metrics(problem, active, self).unwrap_or((None, None));
        self.report.inlier_count = count;
        self.report.final_inlier_rmse = rmse;
    }

    /// Lift this solution of a restricted problem back to `original`: `kept[i]` is the index in
    /// `original` of this solution's point `i`, and every other point keeps its input position and
    /// counts as quarantined. `origin`, when given, is the report of the earlier solves this one continued.
    #[cfg(feature = "faer")]
    pub(crate) fn lifted(
        self,
        original: &BundleProblem,
        kept: &[usize],
        origin: Option<&SolveReport>,
        quarantined: usize,
    ) -> Result<Self, SolutionError> {
        let mut points: Vec<Point3<f64>> = original
            .points()
            .iter()
            .map(|point| *point.position())
            .collect();
        if kept.len() != self.points.len() || !self.quarantined_points.is_empty() {
            return Err(SolutionError::InvalidQuarantine);
        }
        let mut retained = vec![false; points.len()];
        for (restricted, &index) in kept.iter().enumerate() {
            if index >= points.len() || retained[index] {
                return Err(SolutionError::InvalidQuarantine);
            }
            retained[index] = true;
            points[index] = self.points[restricted];
        }
        let quarantined_points: Vec<_> = retained
            .iter()
            .enumerate()
            .filter_map(|(point, &retained)| (!retained).then_some(point))
            .collect();
        if quarantined_points.len() != quarantined {
            return Err(SolutionError::InvalidQuarantine);
        }
        let mut report = match origin {
            Some(origin) => self.report.continued_from(origin),
            None => self.report,
        };
        report.diagnostics = report.diagnostics.with_quarantined_point_count(quarantined);
        Self::try_new_with_quarantined_points(
            original,
            self.poses,
            self.intrinsics,
            points,
            report,
            quarantined_points,
        )
    }

    /// Returned world-to-camera poses, one per problem camera, in problem order.
    #[must_use]
    pub fn poses(&self) -> &[WorldToCameraPose] {
        &self.poses
    }

    /// Returned intrinsics, one per problem intrinsic block, in problem order.
    #[must_use]
    pub fn intrinsics(&self) -> &[CameraIntrinsics] {
        &self.intrinsics
    }

    /// Returned world-frame points, one per problem point, in problem order.
    #[must_use]
    pub fn points(&self) -> &[Point3<f64>] {
        &self.points
    }

    /// Indices of points held out of the solved and certified system, in ascending order.
    ///
    /// They retain their input coordinates and remain untrusted even if the returned cameras
    /// make their geometric conditioning appear healthy. Their observations are excluded from
    /// both initial and final metrics in [`Self::report`].
    #[must_use]
    pub fn quarantined_points(&self) -> &[usize] {
        &self.quarantined_points
    }

    /// Residual space of the problem this proposal solves.
    #[must_use]
    pub const fn residual_space(&self) -> ResidualSpace {
        self.residual_space
    }

    /// Numerical evidence for the solve.
    #[must_use]
    pub const fn report(&self) -> &SolveReport {
        &self.report
    }

    /// The residual norm of every observation at this state, in [`BundleProblem::observations`]
    /// order and in the problem's residual space: pixels, or radians for angular residuals.
    ///
    /// Compare each with the robust loss scale to find the observations the loss treated as
    /// outliers: a wrong match that the loss rejected keeps a large residual, while the report's
    /// [`SolveReport::final_inlier_rmse`] cannot show it. An entry is `None` when its point is
    /// quarantined ([`Self::quarantined_points`]), since the point kept its input position, or
    /// when the observation cannot be evaluated at this state, for example a point behind its
    /// camera.
    ///
    /// # Errors
    ///
    /// [`SolutionError::ResidualSpaceMismatch`] if `problem` uses another residual space, and
    /// [`SolutionError::ShapeMismatch`] if its pose, intrinsic or point count differs from this
    /// solution's: `problem` must be the problem this solution answers.
    pub fn observation_residuals(
        &self,
        problem: &BundleProblem,
    ) -> Result<Vec<Option<f64>>, SolutionError> {
        if self.residual_space != problem.residual_space() {
            return Err(SolutionError::ResidualSpaceMismatch);
        }
        if self.poses.len() != problem.poses().len()
            || self.intrinsics.len() != problem.intrinsics().len()
            || self.points.len() != problem.points().len()
        {
            return Err(SolutionError::ShapeMismatch);
        }
        let mut quarantined = vec![false; self.points.len()];
        for &point in &self.quarantined_points {
            if let Some(flag) = quarantined.get_mut(point) {
                *flag = true;
            }
        }
        Ok(problem
            .observations()
            .iter()
            .map(|observation| {
                if quarantined[observation.point()] {
                    return None;
                }
                residual_norm(
                    self.residual_space,
                    observation,
                    &self.poses[observation.pose()],
                    self.intrinsics[observation.intrinsic()].camera(),
                    &self.points[observation.point()],
                )
            })
            .collect())
    }

    #[cfg(feature = "faer")]
    pub(crate) fn canonicalize_roundoff_non_descent(mut self, problem: &BundleProblem) -> Self {
        if let crate::Gauge::FixedBaseline {
            first,
            second,
            distance,
        } = problem.gauge()
        {
            let initial = center_distance(
                problem.poses()[first].pose().center(),
                problem.poses()[second].pose().center(),
            );
            if !initial.is_finite()
                || (initial - distance).abs() > 512.0 * f64::EPSILON * initial.max(distance)
            {
                // Restoring the original physical scale would violate the requested gauge.
                return self;
            }
        }
        let increase = self.report.final_objective - self.report.initial_objective;
        let tolerance = 2.0 * f64::EPSILON.sqrt() * self.report.initial_objective.abs().max(1.0);
        if !(increase > 0.0
            && increase <= tolerance
            && matches!(self.report.status, SolveStatus::Converged(_))
            && !matches!(
                self.report.diagnostics.observability,
                Observability::RankDeficient
            ))
        {
            return self;
        }
        let mut active = vec![true; problem.points().len()];
        for &point in &self.quarantined_points {
            let Some(retained) = active.get_mut(point) else {
                return self;
            };
            *retained = false;
        }
        let Ok((input_objective, input_rmse)) = active_objective_metrics(problem, &active, None)
        else {
            // A finite proposal must not be replaced by unprojectable input geometry.
            return self;
        };
        if !self.has_input_geometry(problem) {
            self.poses = problem
                .poses()
                .iter()
                .map(|camera| camera.pose().clone())
                .collect();
            self.intrinsics = problem
                .intrinsics()
                .iter()
                .map(|intrinsic| intrinsic.intrinsics().clone())
                .collect();
            self.points = problem
                .points()
                .iter()
                .map(|point| *point.position())
                .collect();
            // The input's rank and gradient need not match the accepted proposal.
            // Quarantine scope and held-intrinsic attempt decisions remain unchanged.
            self.report.diagnostics.invalidate_returned_state();
        }
        // Exact same-geometry proposals keep their fresh certificates. Both paths
        // report the returned input metrics while retaining actual attempted work.
        self.report.initial_objective = input_objective;
        self.report.final_objective = input_objective;
        self.report.initial_rmse = input_rmse;
        self.report.final_rmse = input_rmse;
        self.report.status = SolveStatus::Skipped;
        self.refresh_inlier_metrics(problem, &active);
        self
    }

    /// Conservative equality for reusing a certificate without reassessment.
    #[cfg(feature = "faer")]
    fn has_input_geometry(&self, problem: &BundleProblem) -> bool {
        self.poses.len() == problem.poses().len()
            && self.intrinsics.len() == problem.intrinsics().len()
            && self.points.len() == problem.points().len()
            && self.poses.iter().zip(problem.poses()).all(|(pose, input)| {
                pose.rotation()
                    .matrix()
                    .iter()
                    .zip(input.pose().rotation().matrix().iter())
                    .all(|(value, initial)| value.to_bits() == initial.to_bits())
                    && pose
                        .center()
                        .coords
                        .iter()
                        .zip(input.pose().center().coords.iter())
                        .all(|(value, initial)| value.to_bits() == initial.to_bits())
            })
            && self
                .intrinsics
                .iter()
                .zip(problem.intrinsics())
                .all(|(intrinsic, input)| intrinsic == input.intrinsics())
            && self
                .points
                .iter()
                .zip(problem.points())
                .all(|(point, input)| {
                    point
                        .coords
                        .iter()
                        .zip(input.position().coords.iter())
                        .all(|(value, initial)| value.to_bits() == initial.to_bits())
                })
    }

    /// Weak points of this solution: points seen by cameras that cannot project them, sitting on
    /// an observing camera center, seen along a single ray, or so poorly constrained that
    /// eliminating them would cost the camera system its precision. The crate documentation's
    /// [weak and quarantined points](crate#weak-and-quarantined-points) section explains them.
    ///
    /// Such a point carries no usable depth, so landmarks built from it must not be trusted. The
    /// native adjuster already holds them out of the system it solves
    /// ([`SolveDiagnostics::quarantined_point_count`]); this lists them for any solution.
    #[must_use]
    pub fn weak_points(&self, problem: &BundleProblem) -> Vec<usize> {
        let loss = problem.robust_loss().kernel_loss();
        let mut blocks = vec![Some(Matrix3::zeros()); self.points.len()];
        for observation in problem.observations() {
            let Some(block) = blocks[observation.point()].as_mut() else {
                continue;
            };
            let pose = &self.poses[observation.pose()];
            let camera = self.intrinsics[observation.intrinsic()].camera();
            let camera_point = pose
                .project_to_camera(&self.points[observation.point()])
                .coords;
            let linearized = match self.residual_space {
                ResidualSpace::Pixels => camera
                    .project_with_jacobian(&camera_point)
                    .ok()
                    .map(|(pixel, jacobian)| ((pixel - observation.pixel()).norm(), jacobian)),
                ResidualSpace::Angular => {
                    camera
                        .bearing(observation.pixel())
                        .ok()
                        .and_then(|bearing| {
                            crate::kernel::angular_projection(
                                &[bearing.x, bearing.y, bearing.z],
                                &[camera_point.x, camera_point.y, camera_point.z],
                            )
                            .map(|projection| {
                                let [row_x, row_y] = projection.jacobian;
                                (
                                    projection.residual[0].hypot(projection.residual[1]),
                                    Matrix2x3::from_rows(&[row_x.into(), row_y.into()]),
                                )
                            })
                        })
                }
            };
            match linearized {
                Some((residual, jacobian)) if residual.is_finite() => {
                    let jacobian = jacobian * pose.rotation().matrix();
                    *block += loss.sqrt_weight(residual).powi(2) * jacobian.transpose() * jacobian;
                }
                _ => blocks[observation.point()] = None,
            }
        }
        let mut degenerate = vec![false; self.points.len()];
        for &point in &self.quarantined_points {
            degenerate[point] = true;
        }
        for point in crate::quarantine::degenerate_points(problem, &self.poses, &self.points) {
            if !problem.points()[point].is_fixed() {
                degenerate[point] = true;
            }
        }
        blocks
            .iter()
            .enumerate()
            .filter(|(point, block)| {
                degenerate[*point]
                    || (!problem.points()[*point].is_fixed()
                        && block.is_none_or(|block| {
                            let eigenvalues = block.symmetric_eigenvalues();
                            let threshold =
                                crate::kernel::POINT_BLOCK_QUARANTINE_RCOND * eigenvalues.max();
                            // A NaN spectrum carries no more information than a singular one.
                            eigenvalues.min().partial_cmp(&threshold)
                                != Some(std::cmp::Ordering::Greater)
                        }))
            })
            .map(|(point, _)| point)
            .collect()
    }

    /// Apply the backend-independent numerical and gauge checks required before commit.
    ///
    /// The audit checks that the result is safe to use, not that it is good: the blocks match
    /// the problem, the objective did not increase, every point and camera is observable, the
    /// gauge matches, and the solve neither stalled nor degenerated. A solve that hit
    /// [`SolveConfig::max_iterations`] after lowering the objective passes. Judge quality from
    /// [`SolveReport::status`], [`SolveReport::final_inlier_rmse`] and the poses themselves.
    pub fn audit_for_commit(
        &self,
        problem: &BundleProblem,
        config: &SolveConfig,
    ) -> Result<(), AuditError> {
        if self.residual_space != problem.residual_space() {
            return Err(AuditError::InvalidSolution(
                SolutionError::ResidualSpaceMismatch,
            ));
        }
        validate_solution_blocks(problem, &self.poses, &self.intrinsics, &self.points)
            .map_err(AuditError::InvalidSolution)?;
        if self.quarantined_points.iter().any(|&point| {
            point >= problem.points().len()
                || self.points[point] != *problem.points()[point].position()
        }) || self.quarantined_points.len() != self.report.diagnostics.quarantined_point_count
        {
            return Err(AuditError::InvalidSolution(
                SolutionError::InvalidQuarantine,
            ));
        }
        let expected_observations = problem
            .observations()
            .iter()
            .filter(|observation| {
                self.quarantined_points
                    .binary_search(&observation.point())
                    .is_err()
            })
            .count();
        if self.report.observation_count != Some(expected_observations) {
            return Err(AuditError::InvalidSolution(
                SolutionError::InvalidQuarantine,
            ));
        }
        let report = &self.report;
        let diagnostics = report.diagnostics;
        if report.final_objective
            > report.initial_objective + 64.0 * f64::EPSILON * report.initial_objective.abs()
        {
            return Err(AuditError::ObjectiveIncreased);
        }
        let weak_point_count = diagnostics
            .weak_point_count
            .ok_or(AuditError::ObservabilityNotAssessed)?;
        if weak_point_count > 0 {
            return Err(AuditError::WeakPoints {
                count: weak_point_count,
            });
        }
        if diagnostics.noninvertible_observation_count.is_none() {
            return Err(AuditError::ObservabilityNotAssessed);
        }
        match diagnostics.observability {
            Observability::RankDeficient => return Err(AuditError::RankDeficient),
            Observability::NotAssessed => return Err(AuditError::ObservabilityNotAssessed),
            Observability::OnlyExpectedGauge if !matches!(problem.gauge(), crate::Gauge::Free) => {
                return Err(AuditError::GaugeMismatch);
            }
            Observability::FullRank | Observability::OnlyExpectedGauge => {}
        }
        let stationarity = diagnostics
            .stationarity
            .ok_or(AuditError::StationarityNotCertified)?;
        match report.status {
            // A config that no adjuster would accept cannot certify gradient convergence.
            SolveStatus::Converged(ConvergenceReason::Gradient)
                if config.validate().is_err()
                    || stationarity.scaled_gradient_inf_norm
                        > config.gradient_tolerance * (1.0 + report.final_objective) =>
            {
                Err(AuditError::StationarityNotCertified)
            }
            SolveStatus::Converged(_) | SolveStatus::Skipped => Ok(()),
            SolveStatus::MaxIterations
                if diagnostics.accepted_step_count > 0
                    && report.final_objective < report.initial_objective =>
            {
                Ok(())
            }
            SolveStatus::MaxIterations => Err(AuditError::NoAcceptedProgress),
            SolveStatus::Stalled | SolveStatus::Degenerate => {
                Err(AuditError::Status(report.status))
            }
        }
    }
}

/// Evaluate one retained observation domain in the public problem's exact residual units.
fn active_objective_metrics(
    problem: &BundleProblem,
    active: &[bool],
    solution: Option<&Solution>,
) -> Result<(f64, f64), SolutionError> {
    let loss = problem.robust_loss().kernel_loss();
    let mut cost = 0.0;
    let mut residual_scale = 0.0;
    let mut scaled_squared_residual = 1.0;
    let mut count = 0usize;
    visit_residual_norms(problem, active, solution, |norm| {
        // BundleProblem declares no observation-noise model; native whitening is Unit.
        cost += loss.cost(norm);
        if norm > 0.0 {
            if residual_scale < norm {
                scaled_squared_residual =
                    1.0 + scaled_squared_residual * (residual_scale / norm).powi(2);
                residual_scale = norm;
            } else {
                scaled_squared_residual += (norm / residual_scale).powi(2);
            }
        }
        count += 1;
    })?;
    if count == 0 || !cost.is_finite() || cost < 0.0 {
        return Err(SolutionError::InvalidReport);
    }
    let rmse = residual_scale * (scaled_squared_residual / count as f64).sqrt();
    if !rmse.is_finite() {
        return Err(SolutionError::InvalidReport);
    }
    Ok((cost, rmse))
}

/// The count and root mean square of the residuals at `solution` within the robust-loss scale,
/// over the observations of `active` points.
fn inlier_metrics(
    problem: &BundleProblem,
    active: &[bool],
    solution: &Solution,
) -> Result<(Option<usize>, Option<f64>), SolutionError> {
    let limit = problem
        .robust_loss()
        .scale()
        .map_or(f64::INFINITY, ResidualScale::value);
    let (mut count, mut squared_sum) = (0usize, 0.0);
    visit_residual_norms(problem, active, Some(solution), |norm| {
        if norm <= limit {
            count += 1;
            squared_sum += norm * norm;
        }
    })?;
    let rmse = (count > 0).then(|| (squared_sum / count as f64).sqrt());
    Ok((Some(count), rmse.filter(|rmse| rmse.is_finite())))
}

/// Call `visit` with the residual norm, in the problem's residual space, of every observation of
/// an `active` point, at the input state or at `solution`'s state.
fn visit_residual_norms(
    problem: &BundleProblem,
    active: &[bool],
    solution: Option<&Solution>,
    mut visit: impl FnMut(f64),
) -> Result<(), SolutionError> {
    for observation in problem.observations() {
        if !active[observation.point()] {
            continue;
        }
        let pose = solution.map_or(problem.poses()[observation.pose()].pose(), |solution| {
            &solution.poses[observation.pose()]
        });
        let intrinsic = observation.intrinsic();
        let camera = solution.map_or(
            problem.intrinsics()[intrinsic].intrinsics().camera(),
            |solution| solution.intrinsics[intrinsic].camera(),
        );
        let point = solution.map_or(
            problem.points()[observation.point()].position(),
            |solution| &solution.points[observation.point()],
        );
        let norm = residual_norm(problem.residual_space(), observation, pose, camera, point)
            .ok_or(SolutionError::InvalidReport)?;
        visit(norm);
    }
    Ok(())
}

/// Residual norm of `observation` in `space` for a camera at `pose` with `camera`, seeing `point`,
/// or `None` when the observation cannot be evaluated there.
fn residual_norm(
    space: ResidualSpace,
    observation: &BundleObservation,
    pose: &WorldToCameraPose,
    camera: &Camera,
    point: &Point3<f64>,
) -> Option<f64> {
    let camera_point = pose.project_to_camera(point).coords;
    let norm = match space {
        ResidualSpace::Pixels => camera.project(&camera_point).ok().map(|pixel| {
            let residual = pixel - observation.pixel();
            residual.x.hypot(residual.y)
        }),
        ResidualSpace::Angular => camera
            .bearing(observation.pixel())
            .ok()
            .and_then(|bearing| {
                crate::kernel::angular_projection(
                    &[bearing.x, bearing.y, bearing.z],
                    &[camera_point.x, camera_point.y, camera_point.z],
                )
                .map(|projection| projection.residual[0].hypot(projection.residual[1]))
            }),
    }?;
    norm.is_finite().then_some(norm)
}

/// Reason an owned proposal cannot be committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AuditError {
    /// Returned blocks or residual units violate the problem this audit received.
    #[error("bundle-adjustment proposal violates the problem: {0}")]
    InvalidSolution(#[source] SolutionError),
    /// The final objective exceeds the initial one by more than round-off.
    #[error("bundle-adjustment objective increased")]
    ObjectiveIncreased,
    /// The report is [`Observability::RankDeficient`].
    #[error("bundle-adjustment result is rank deficient")]
    RankDeficient,
    /// The report is [`Observability::NotAssessed`].
    #[error("bundle-adjustment observability was not assessed")]
    ObservabilityNotAssessed,
    /// The report is [`Observability::OnlyExpectedGauge`] but the problem's gauge fixes
    /// the scale.
    #[error("bundle-adjustment observability is inconsistent with its gauge")]
    GaugeMismatch,
    /// The final undamped system has numerically singular point blocks.
    #[error("bundle-adjustment result contains {count} weak point blocks")]
    WeakPoints {
        /// Number of weak point blocks.
        count: usize,
    },
    /// Gradient convergence is claimed, but the config is invalid or the zero-damping
    /// stationarity exceeds [`SolveConfig::gradient_tolerance`].
    #[error("zero-damping stationarity does not satisfy the convergence tolerance")]
    StationarityNotCertified,
    /// The iteration budget ran out without an accepted step that lowered the objective.
    #[error("bundle adjustment made no accepted progress")]
    NoAcceptedProgress,
    /// The solve ended [`SolveStatus::Stalled`] or [`SolveStatus::Degenerate`].
    #[error("bundle-adjustment status {0:?} does not permit commit")]
    Status(SolveStatus),
}

/// Error returned by a bundle adjustment (BA) implementation.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SolveError {
    /// The [`SolveConfig`] failed [`SolveConfig::validate`]; names the rejected setting.
    #[error("invalid solve config: {0}")]
    InvalidConfig(&'static str),
    /// The implementation does not support this problem, for example its camera model,
    /// loss or variable layout; carries the reason.
    #[error("unsupported bundle-adjustment problem: {0}")]
    Unsupported(String),
    /// The solver workspace exceeds [`SolveConfig::workspace_budget_bytes`] or could not be
    /// allocated; retrying with less memory pressure may succeed.
    #[error("bundle-adjustment workspace is unavailable: {0}")]
    ResourceExhausted(String),
    /// The solve failed numerically, for example on a non-finite value or a failed
    /// factorization; carries the details.
    #[error("bundle adjustment failed numerically: {0}")]
    Numerical(String),
    /// The implementation broke one of its own invariants; this is a bug.
    #[error("bundle-adjustment implementation failed: {0}")]
    Internal(String),
}

/// Invalid owned solver output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SolutionError {
    /// The returned state changed the anchored pose or requested center-to-center baseline.
    #[error("solution changed its fixed-baseline gauge")]
    GaugeChanged,
    /// The represented coordinates cannot certify the requested baseline to the relative tolerance.
    #[error("solution coordinates cannot represent its fixed-baseline gauge precisely enough")]
    GaugePrecisionLimited,
    /// The proposal's residual domain differs from the problem being audited.
    #[error("solution residual units do not match the problem")]
    ResidualSpaceMismatch,
    /// Held-out point identities are duplicated, out of bounds, changed, or disagree with the report.
    #[error("solution quarantine identities do not match the problem or report")]
    InvalidQuarantine,
    /// The pose, intrinsic or point count differs from the problem's.
    #[error("solution shape does not match the problem")]
    ShapeMismatch,
    /// A returned pose is not a valid rigid transform.
    #[error("solution pose {pose} is invalid")]
    InvalidPose {
        /// Index of the pose in [`BundleProblem::poses`].
        pose: usize,
    },
    /// A returned intrinsic is outside its camera model's domain.
    #[error("solution intrinsic {intrinsic} is invalid")]
    InvalidIntrinsic {
        /// Index of the intrinsic block in the problem.
        intrinsic: usize,
    },
    /// A returned point has a non-finite coordinate.
    #[error("solution point {point} is non-finite")]
    InvalidPoint {
        /// Index of the point in the problem.
        point: usize,
    },
    /// A rotation or translation held fixed by its [`PoseVariables`](crate::PoseVariables)
    /// changed.
    #[error("solution changed fixed pose {pose}")]
    FixedPoseChanged {
        /// Index of the pose in [`BundleProblem::poses`].
        pose: usize,
    },
    /// An intrinsic changed its model, image metadata, or coordinates outside its variable mask.
    #[error("solution changed intrinsic {intrinsic} outside its variables or image domain")]
    FixedIntrinsicChanged {
        /// Index of the intrinsic block in the problem.
        intrinsic: usize,
    },
    /// A fixed point moved.
    #[error("solution changed fixed point {point}")]
    FixedPointChanged {
        /// Index of the point in the problem.
        point: usize,
    },
    /// Report values are negative, not finite or mutually inconsistent.
    #[error("solution report contains invalid numerical evidence")]
    InvalidReport,
    /// A stationarity norm is negative or not finite.
    #[error("stationarity audit must be finite and non-negative")]
    InvalidStationarityAudit,
}

/// Stateful bundle-adjustment implementation.
///
/// Implementations reject a config that fails [`SolveConfig::validate`] with that error.
pub trait BundleAdjuster: Send {
    /// Stable implementation identifier, recorded in SfM reports.
    fn name(&self) -> &'static str;

    /// Refine `problem` from its initial state and return the proposal without mutating
    /// the problem.
    fn solve(
        &mut self,
        problem: &BundleProblem,
        config: SolveConfig,
    ) -> Result<Solution, SolveError>;
}

#[cfg(all(test, feature = "faer"))]
#[path = "solver/rollback_tests.rs"]
mod rollback_tests;
