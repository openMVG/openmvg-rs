// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lowering of the checked [`BundleProblem`] contract to this crate's native BA kernels.

mod calibrated;
mod fixed_rotation;
mod gauge;
mod reporting;
mod self_calibration;
#[cfg(test)]
mod tests;
mod two_view;
mod world_rebase;

use std::collections::BTreeSet;

use crate::kernel::PinholeCamera;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::RetainedSchurMatvec;
use crate::kernel::multi_view::sparse::{
    IterativeSchurOptions, SparseBundleAdjustmentWorkspace, SparseWorkspaceMemoryBudget,
    SparseWorkspaceOptions, refine_ba_multi_view_sparse_with_factors,
};
use crate::kernel::multi_view::{
    BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks, MultiViewMeasurements,
};
use crate::numerics::refine::{ConvergenceCriteria, Options};
use crate::{
    BundleAdjuster, BundleProblem, LinearSolver, Observability, ResidualScale, ResidualSpace,
    Solution, SolveConfig, SolveError, SolveStatus, quarantine,
};
use calibrated::{PreparedCalibratedProblem, raise_solution, solve_angular};
use fixed_rotation::probe_prepared;
use gauge::common_pose_variables;
use openmvg_core::camera::Camera;
use reporting::map_native_error;
use self_calibration::solve_self_calibrating;
use two_view::{can_use_two_view, solve_two_view};

/// The native bundle adjuster: Levenberg-Marquardt on the sparse camera system of a checked
/// [`BundleProblem`].
///
/// This is the default implementation of the checked [`BundleProblem`] contract. With fixed
/// intrinsics it keeps the complete pinhole or Brown-Conrady calibration, and it honors each
/// pose's rotation and center masks. Fixed world points that cameras observe (control points)
/// can set the reference frame without a fixed pose. What nothing fixes, it holds at its
/// starting value, as [`Gauge`](crate::Gauge) explains, so the camera system it reports on is
/// fully determined: a well-observed scene reports
/// [`Observability::FullRank`](crate::Observability::FullRank) even with every pose free.
///
/// The solver *quarantines* the points it cannot place and leaves them at their input
/// position; the crate documentation's [weak and quarantined points](crate#weak-and-quarantined-points)
/// section explains which, and [`Solution::quarantined_points`](crate::Solution::quarantined_points)
/// lists them.
///
/// [`Gauge::FixedBaseline`](crate::Gauge::FixedBaseline) explains when a requested baseline
/// distance is unsupported or too imprecise to reach.
///
/// # Algorithm
///
/// ```text
/// Input: BundleProblem (poses, intrinsics, points, observations, loss, gauge),
///        SolveConfig (iteration limit, tolerances, linear solver, workspace budget)
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Fix the gauge and set weak points aside                      │
/// └──────────────────────────────────────────────────────────────────────┘
/// Hold an anchor pose, and a baseline when nothing else fixes the scale
/// (Gauge); quarantine the points that the input already cannot place.
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Iterate Levenberg-Marquardt                                  │
/// └──────────────────────────────────────────────────────────────────────┘
/// Linearize the residuals, eliminate each point's 3 × 3 block, solve the
/// reduced camera system (LinearSolver), back-substitute the points; keep
/// the step when the cost drops and adapt the damping. Stop at a tolerance
/// or at the iteration limit.
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Certify, and re-solve without newly weak points              │
/// └──────────────────────────────────────────────────────────────────────┘
/// Assess the rank of the camera system, the weak points left and the
/// stationarity. When weak points remain or the rank falls short, quarantine
/// the weak points and warm-start a short re-solve, for a few rounds at most.
///   │
///   ▼
/// Output: Solution (refined state, quarantined points) and its SolveReport
/// ```
///
/// The solver runs Levenberg-Marquardt with Nielsen's gain-ratio damping: each step solves the
/// damped normal equations, keeps the step when the objective drops, and adapts the damping
/// from how well the linear model predicted that drop. The points are eliminated first, so each
/// step solves the reduced camera system chosen by [`LinearSolver`], with the deterministic
/// sparse Cholesky factorization or with block-Jacobi preconditioned conjugate gradients. The workspace is
/// built once per solve; the iterations after it do not allocate.
///
/// # Supported problems
///
/// A combination outside this table returns [`SolveError::Unsupported`](crate::SolveError),
/// and the problem can go to another [`BundleAdjuster`](crate::BundleAdjuster) unchanged.
///
/// | Problem | Native support |
/// |---|---|
/// | Bundle adjustment in pixels with fixed intrinsics | Pinhole and complete Brown-Conrady calibration, including unequal focal lengths, `k3` and tangential distortion |
/// | Shared physical poses with fixed intrinsics | Observations that share one pose may select different supported lens groups |
/// | Intrinsic refinement | Pixel residuals; full-pose or fixed cameras; pinhole or Brown-Conrady cameras with one focal length (`f_x = f_y`), no tangential distortion and no `k3` term (a `K1K2K3` model with free radial coefficients is rejected even when `k3` is zero); the focal length free, alone or with `k1` and `k2`; the principal point fixed |
/// | Angular bundle adjustment | Fixed intrinsics; measured pixels must define unique bearings |
/// | Pose masks with fixed intrinsics | Per-pose full, translation-only, rotation-only and fixed blocks, in any mixture; a held translation holds the world camera center |
/// | Fixed world points with fixed intrinsics | Known control points constrain the poses without triangulation; every pose can stay free |
///
/// Self-calibration needs each observed pose to use one intrinsic group; several variable lens
/// groups on one pose are unsupported.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBundleAdjuster;

/// Outcome of [`NativeBundleAdjuster::preflight`], a quick check that `openmvg-sfm` runs before a
/// bundle adjustment of camera positions with fixed rotations; most callers never need it.
///
/// The check solves the camera system of a limited subset of the tracks. If that subset already
/// determines every camera, so does the whole problem, since more tracks only add information.
/// Every status other than [`NativePreflightStatus::Proceed`] means the check could not show
/// that; it is a reason not to run the problem, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativePreflightStatus {
    /// The track subset determines every camera, or the problem is not one the check covers and
    /// the solve's own assessment will decide.
    Proceed,
    /// The track subset did not see every free camera.
    InsufficientCameraCoverage,
    /// The check needed more memory or work than it allows itself.
    ResourceLimit,
    /// At least one selected point is too weakly observed to locate.
    WeakPointGeometry,
    /// The track subset does not determine every camera.
    ObservabilityNotCertified,
}

/// What [`NativeBundleAdjuster::preflight`] found, used by `openmvg-sfm` to decide whether to run a
/// bundle adjustment; most callers never need it.
///
/// Problems the check does not cover report [`NativePreflightStatus::Proceed`] with every count
/// unset, and rely on the assessment the solve makes at the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NativePreflight {
    /// Outcome of the probe.
    pub status: NativePreflightStatus,
    /// Free camera coordinates the check tested, once the gauge is fixed; `None` when it did not
    /// run.
    pub pose_dimension: Option<usize>,
    /// Cameras with a free position that the selected tracks see; `None` when the check did not
    /// run.
    pub covered_free_pose_count: Option<usize>,
    /// Selected points too weakly observed to locate; `None` when the check did not run.
    pub weak_point_count: Option<usize>,
    /// The check let the problem proceed without showing that the subset determines every
    /// camera, so the cameras count as determined only if the solve's own assessment confirms
    /// it.
    pub dense_observability_escalation: bool,
}

impl NativeBundleAdjuster {
    /// Linear-solver workspace budget, 1 GiB, that the solver plans its strategy against when
    /// [`SolveConfig::workspace_budget_bytes`] is `None`.
    pub const AUTO_WORKSPACE_BYTES: usize = 1 << 30;

    /// Construct the stateless native adjuster.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check, on a subset of the tracks, that the observations determine every camera of a
    /// problem whose poses only move their centers, before solving it.
    ///
    /// `openmvg-sfm` runs it before such bundle adjustments; other problems report
    /// [`NativePreflightStatus::Proceed`] without a check. Not part of the documented API.
    #[doc(hidden)]
    pub fn preflight(
        &self,
        problem: &BundleProblem,
        config: &SolveConfig,
    ) -> Result<NativePreflight, SolveError> {
        config.validate()?;
        if common_pose_variables(problem).ok() != Some(CameraPoseVariables::TranslationOnly)
            || matches!(problem.residual_space(), ResidualSpace::Angular)
            || problem.points().iter().any(|point| point.is_fixed())
        {
            return Ok(NativePreflight {
                status: NativePreflightStatus::Proceed,
                pose_dimension: None,
                covered_free_pose_count: None,
                weak_point_count: None,
                dense_observability_escalation: false,
            });
        }
        preflight_calibrated(problem)
    }
}

// The documented automatic budget is the one the sparse workspace plans against.
const _: () =
    assert!(NativeBundleAdjuster::AUTO_WORKSPACE_BYTES == SparseWorkspaceMemoryBudget::AUTO_BYTES);

/// The observability probe of a translation-only problem on the sparse path's prepared frame.
/// It needs a scale coordinate and one camera model per pose; without them it lets the solve
/// proceed uncertified.
fn preflight_calibrated(problem: &BundleProblem) -> Result<NativePreflight, SolveError> {
    let prepared = PreparedCalibratedProblem::try_new(problem)?;
    if matches!(prepared.gauge, crate::kernel::multi_view::ScaleGauge::Free)
        || !prepared.observation_intrinsics.is_empty()
        || !can_probe_scale_representative(problem)
    {
        return Ok(NativePreflight {
            status: NativePreflightStatus::Proceed,
            pose_dimension: None,
            covered_free_pose_count: None,
            weak_point_count: None,
            dense_observability_escalation: false,
        });
    }
    probe_prepared(problem, &prepared)
}

impl BundleAdjuster for NativeBundleAdjuster {
    fn name(&self) -> &'static str {
        "openmvg-ba-native"
    }

    fn solve(
        &mut self,
        problem: &BundleProblem,
        config: SolveConfig,
    ) -> Result<Solution, SolveError> {
        solve_with_quarantine(problem, config, &solve_once)
    }
}

/// Largest relative change of a fixed baseline that still counts as keeping the physical scale
/// while points are held out. A held-out point keeps its input position while the solve rescales
/// the scene to the requested baseline, so a real scale change would leave it behind. The round-off
/// a previous solve leaves in the baseline (about 1e-13 relative) moves it far less than any
/// measurement, and refusing it would make every later solve of the same scene fail too, since a
/// refused solve never restores the baseline.
const HELD_OUT_SCALE_TOLERANCE: f64 = 1.0e-9;

/// Solve `problem`, holding out the landmarks it cannot adjust.
///
/// Free points seen fewer than twice, on a camera center, or without parallax are removed before
/// the solve; when that leaves no point, the solve returns [`SolveError::Unsupported`]. Weak
/// points found afterwards trigger bounded warm-started re-solves. Held-out points retain
/// their input positions and explicit identities in [`Solution::quarantined_points`]. Initial
/// and final metrics both use the final retained observations; rank and stationarity certify
/// that same system.
fn solve_with_quarantine(
    problem: &BundleProblem,
    config: SolveConfig,
    solve_once: &dyn Fn(&BundleProblem, SolveConfig) -> Result<Solution, SolveError>,
) -> Result<Solution, SolveError> {
    config.validate()?;
    let input_degenerate: BTreeSet<usize> = quarantine::degenerate_input_points(problem)
        .into_iter()
        .collect();
    if input_degenerate.len() == problem.points().len() {
        return Err(SolveError::Unsupported(
            "no point can be adjusted: every point is seen fewer than twice or without parallax"
                .to_owned(),
        ));
    }
    let restricted_input;
    let (working, mut kept) = if input_degenerate.is_empty() {
        (problem, (0..problem.points().len()).collect())
    } else {
        let (restricted, kept) = problem
            .without_points(None, &input_degenerate)
            .map_err(|error| SolveError::Internal(error.to_string()))?;
        restricted_input = restricted;
        (&restricted_input, kept)
    };
    let mut quarantined = input_degenerate.len();
    let changes_physical_scale = matches!(problem.gauge(), crate::Gauge::FixedBaseline { .. })
        && (gauge::gauge_plan(problem)?.scale - 1.0).abs() > HELD_OUT_SCALE_TOLERANCE;
    if changes_physical_scale && quarantined > 0 {
        return Err(SolveError::Unsupported(
            "native BA cannot change its physical baseline while holding out points".to_owned(),
        ));
    }
    let mut solution = solve_once(working, config)?;
    // Each re-solve starts from the previous solution without the points that made it fail its
    // certificate, so the work shrinks with every round.
    let retry =
        config.with_max_iterations(config.max_iterations().min(quarantine::RESOLVE_ITERATIONS));
    let mut restricted: Option<BundleProblem> = None;
    let mut origin = None;
    for _ in 0..quarantine::MAX_RESOLVES {
        if !needs_resolve(&solution) {
            break;
        }
        let current = restricted.as_ref().unwrap_or(working);
        let weak: BTreeSet<usize> = solution
            .weak_points(current)
            .into_iter()
            .filter(|&point| !current.points()[point].is_fixed())
            .collect();
        if weak.is_empty() {
            break;
        }
        let Ok((next, next_kept)) = current.without_points(Some(&solution), &weak) else {
            break;
        };
        let Ok(resolved) = solve_once(&next, retry) else {
            break;
        };
        kept = next_kept.iter().map(|&point| kept[point]).collect();
        quarantined += weak.len();
        let previous = std::mem::replace(&mut solution, resolved);
        origin = Some(match origin {
            Some(origin) => previous.report().clone().continued_from(&origin),
            None => previous.report().clone(),
        });
        restricted = Some(next);
    }
    if changes_physical_scale && quarantined > 0 {
        return Err(SolveError::Unsupported(
            "native BA cannot change its physical baseline while holding out points".to_owned(),
        ));
    }
    if quarantined == 0 {
        // Nothing was held out: the solution already belongs to `problem`.
        return Ok(solution);
    }
    solution
        .lifted(problem, &kept, origin.as_ref(), quarantined)
        .map_err(reporting::map_solution_error)
}

/// Whether a solve failed its own certificate in a way held-out points can repair.
fn needs_resolve(solution: &Solution) -> bool {
    let diagnostics = solution.report().diagnostics();
    diagnostics
        .weak_point_count()
        .is_some_and(|count| count > 0)
        || matches!(diagnostics.observability(), Observability::RankDeficient)
        || matches!(solution.report().status(), SolveStatus::Degenerate)
}

fn solve_once(problem: &BundleProblem, config: SolveConfig) -> Result<Solution, SolveError> {
    config.validate()?;
    if matches!(problem.residual_space(), ResidualSpace::Angular) {
        return solve_angular(problem, &config)
            .map(|solution| solution.canonicalize_roundoff_non_descent(problem));
    }
    if has_free_intrinsics(problem) {
        return solve_self_calibrating(problem, &config)
            .map(|solution| solution.canonicalize_roundoff_non_descent(problem));
    }
    if can_use_two_view(problem) {
        return solve_two_view(problem, &config)
            .map(|solution| solution.canonicalize_roundoff_non_descent(problem));
    }
    solve_calibrated(problem, &config)
}

/// The sparse multi-view solve of a problem with fixed intrinsics.
fn solve_calibrated(problem: &BundleProblem, config: &SolveConfig) -> Result<Solution, SolveError> {
    let mut prepared = PreparedCalibratedProblem::try_new(problem)?;
    let measurements = if !prepared.observation_intrinsics.is_empty() {
        MultiViewMeasurements::FixedPerspectiveIndexed {
            observations: &prepared.observations,
            point_offsets: &prepared.point_offsets,
            cameras: &prepared.fixed_perspective_cameras,
            observation_intrinsics: &prepared.observation_intrinsics,
        }
    } else if prepared.fixed_perspective_cameras.is_empty() {
        MultiViewMeasurements::Pixel {
            observations: &prepared.observations,
            point_offsets: &prepared.point_offsets,
            cameras: &prepared.cameras,
        }
    } else {
        MultiViewMeasurements::FixedPerspective {
            observations: &prepared.observations,
            point_offsets: &prepared.point_offsets,
            cameras: &prepared.fixed_perspective_cameras,
        }
    };
    let fixed = FixedVariableBlocks {
        camera_poses: &prepared.fixed_poses,
        points: &prepared.fixed_points,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    // A translation-only solve serves as the metric-core rank check, and its reduced system
    // has only three unknowns per camera: resolve Auto to the direct solver so the audit gets
    // the exact camera rank that the diagnostic-free iterative speed path never computes.
    let workspace_options = if prepared.pose_variables == CameraPoseVariables::TranslationOnly
        && matches!(config.linear_solver(), LinearSolver::Auto)
    {
        direct_workspace_options()
    } else {
        workspace_options(config)
    };
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_masks(
        prepared.poses.len(),
        measurements,
        prepared.gauge,
        fixed,
        &prepared.pose_masks,
        workspace_options,
    )
    .map_err(map_native_error)?;
    check_workspace_budget(
        "native BA",
        workspace.execution_decision().selected_workspace_bytes(),
        config,
    )?;
    let options = solver_options(problem, config)?;
    let report = refine_ba_multi_view_sparse_with_factors(
        &mut prepared.poses,
        &mut prepared.points,
        measurements,
        prepared.gauge,
        fixed,
        BundleAdjustmentPriors::NONE,
        &mut workspace,
        options,
        crate::numerics::refine::NoiseModel::Unit,
    )
    .map_err(map_native_error)?;
    let observability = if prepared.pose_variables == CameraPoseVariables::TranslationOnly
        && !prepared.fixed_points.iter().any(|fixed| *fixed)
        && prepared.fixed_perspective_cameras.is_empty()
        && can_probe_scale_representative(problem)
        && !matches!(prepared.gauge, crate::kernel::multi_view::ScaleGauge::Free)
    {
        Some(probe_prepared(problem, &prepared)?)
    } else {
        None
    };
    raise_solution(
        &prepared.rebase,
        problem,
        &prepared.poses,
        &prepared.points,
        &prepared.camera_source_order,
        &report,
        observability,
    )
    .map(|solution| solution.canonicalize_roundoff_non_descent(problem))
}

pub(super) fn has_free_intrinsics(problem: &BundleProblem) -> bool {
    problem.observations().iter().any(|observation| {
        !problem.intrinsics()[observation.intrinsic()]
            .variables()
            .is_fixed()
    })
}

pub(super) fn unobserved_free_intrinsics(problem: &BundleProblem) -> Vec<usize> {
    let mut observed = vec![false; problem.intrinsics().len()];
    for observation in problem.observations() {
        observed[observation.intrinsic()] = true;
    }
    problem
        .intrinsics()
        .iter()
        .enumerate()
        .filter_map(|(intrinsic, block)| {
            (!observed[intrinsic] && !block.variables().is_fixed()).then_some(intrinsic)
        })
        .collect()
}

/// Per-pose calibration indices for solvers that require one model per pose.
/// Unobserved poses use a placeholder that is never projected.
pub(super) fn single_pose_intrinsics(problem: &BundleProblem) -> Option<Vec<usize>> {
    let mut indices = vec![usize::MAX; problem.poses().len()];
    for observation in problem.observations() {
        let slot = &mut indices[observation.pose()];
        if *slot != usize::MAX && *slot != observation.intrinsic() {
            return None;
        }
        *slot = observation.intrinsic();
    }
    for index in &mut indices {
        if *index == usize::MAX {
            *index = 0;
        }
    }
    Some(indices)
}

/// Specialized track layouts require one exposure per pose and point.
pub(super) fn has_repeated_pose_points(problem: &BundleProblem) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    problem
        .observations()
        .iter()
        .any(|row| !seen.insert((row.pose(), row.point())))
}

/// The Cartesian fixed-rotation probe is equivalent to the norm chart only when
/// scene scale is still a symmetry before the gauge constraint is imposed.
fn can_probe_scale_representative(problem: &BundleProblem) -> bool {
    let crate::Gauge::FixedBaseline { first, .. } = problem.gauge() else {
        return true;
    };
    let observed = gauge::observed_cameras(problem);
    !problem.poses().iter().enumerate().any(|(camera, pose)| {
        camera != first
            && observed[camera]
            && !pose.variables().translation()
            && pose.pose().center() != problem.poses()[first].pose().center()
    })
}

fn pinhole(camera: &Camera) -> Result<PinholeCamera, SolveError> {
    match camera {
        Camera::Pinhole(camera) => Ok(PinholeCamera::new(
            camera.focal_x(),
            camera.focal_y(),
            camera.principal_x(),
            camera.principal_y(),
        )),
        Camera::BrownConrady(camera) if !camera.has_distortion() => Ok(PinholeCamera::new(
            camera.focal_x(),
            camera.focal_y(),
            camera.principal_x(),
            camera.principal_y(),
        )),
        Camera::BrownConrady(_) | Camera::Fisheye(_) | Camera::Spherical(_) => {
            Err(SolveError::Unsupported(
                "direct calibrated lowering requires an undistorted perspective camera".to_owned(),
            ))
        }
    }
}

/// Sparse planning budget of `config`: automatic when unset, otherwise its byte count
/// saturated to the address space. [`SolveConfig::validate`] has already rejected zero.
fn sparse_memory_budget(config: &SolveConfig) -> SparseWorkspaceMemoryBudget {
    config
        .workspace_budget_bytes()
        .map_or(SparseWorkspaceMemoryBudget::Auto, |bytes| {
            SparseWorkspaceMemoryBudget::Bytes(usize::try_from(bytes).unwrap_or(usize::MAX))
        })
}

/// Refuse, before the first iteration, a constructed workspace that exceeds an explicit
/// [`SolveConfig::workspace_budget_bytes`]. An automatic budget only guides strategy selection.
fn check_workspace_budget(
    label: &str,
    workspace_bytes: usize,
    config: &SolveConfig,
) -> Result<(), SolveError> {
    match config.workspace_budget_bytes() {
        Some(budget) if workspace_bytes as u128 > budget => {
            Err(SolveError::ResourceExhausted(format!(
                "{label} workspace {workspace_bytes} bytes exceeds the workspace budget {budget}"
            )))
        }
        _ => Ok(()),
    }
}

fn workspace_options(config: &SolveConfig) -> SparseWorkspaceOptions {
    match config.linear_solver() {
        LinearSolver::Auto => SparseWorkspaceOptions::auto_with_iterative_relative_tolerance(
            sparse_memory_budget(config),
            iterative_schur_options().relative_tolerance(),
        ),
        LinearSolver::Direct => direct_workspace_options(),
        LinearSolver::Iterative => {
            SparseWorkspaceOptions::iterative_schur(iterative_schur_options())
        }
    }
}

fn direct_workspace_options() -> SparseWorkspaceOptions {
    #[cfg(feature = "parallel")]
    {
        SparseWorkspaceOptions::direct_auto()
    }
    #[cfg(not(feature = "parallel"))]
    {
        SparseWorkspaceOptions::direct()
    }
}

fn iterative_schur_options() -> IterativeSchurOptions {
    let options = IterativeSchurOptions::new()
        .with_relative_tolerance(1e-1)
        .with_direct_fallback(false);
    #[cfg(feature = "parallel")]
    let options = options.with_matvec(RetainedSchurMatvec::Auto);
    options
}

fn solver_options(problem: &BundleProblem, config: &SolveConfig) -> Result<Options, SolveError> {
    Ok(Options {
        max_iterations: config.max_iterations(),
        loss: problem.robust_loss().kernel_loss(),
        convergence: ConvergenceCriteria::new(
            config.gradient_tolerance(),
            config.parameter_tolerance(),
            config.function_tolerance(),
        ),
    })
}

fn native_loss(problem: &BundleProblem) -> Result<crate::numerics::refine::Loss, SolveError> {
    let loss = problem.robust_loss();
    if matches!(loss.scale(), Some(ResidualScale::Angular(_))) {
        return Err(SolveError::Unsupported(
            "fixed-rotation observability supports squared, pixel Huber, and pixel Cauchy losses"
                .to_owned(),
        ));
    }
    Ok(loss.kernel_loss())
}
