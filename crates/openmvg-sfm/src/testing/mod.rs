// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synthetic fixtures and ground-truth-initialized validation runners for tests and benchmarks.
//!
//! The `unstable-test-support` feature exposes this module for OpenMVG's in-tree validation: the
//! synthetic scene builders, the [`oracle`] fixture and runners (and data types such as
//! [`oracle::OracleObservation`]), the stable hashing and seed
//! derivation in [`determinism`], and the bundle-adjustment, registration, and engine-backend
//! fault-injection [`seams`]. None of it is covered by semver; any item may change or disappear
//! in any release, including patch releases. Applications should not enable
//! `unstable-test-support`.

#[cfg(feature = "unstable-test-support")]
pub mod oracle;

/// Stable hashing and hierarchical seed derivation utilities.
#[cfg(feature = "unstable-test-support")]
pub mod determinism {
    pub use crate::determinism::*;
}

/// A minimal incremental policy, pinned for tests and reference benchmarks.
///
/// Unlike the default [`crate::config::IncrementalSfmConfig::default`], it registers from six
/// 2D-3D observations and six inliers and disables the reprojection drop, initial-pair bundle
/// adjustment, adaptive growth bundle adjustment, and intrinsic self-calibration.
#[cfg(feature = "unstable-test-support")]
#[must_use]
pub fn minimal_incremental_config() -> crate::config::IncrementalSfmConfig {
    crate::config::IncrementalSfmConfig::default()
        .with_min_registration_support(6)
        .with_registration_min_inliers(6)
        .with_max_reprojection_error(None)
        .with_bundle_adjust_initial_pair(false)
        .with_bundle_adjust_growth_ratio(0.0)
        .with_self_calibrate_intrinsics(false)
}

/// A minimal global policy, pinned for tests and reference benchmarks.
///
/// Unlike the default [`crate::config::GlobalSfmConfig::default`], it keeps matching data,
/// skips independent track validation and post-fixed-rotation camera deferral, and keeps the
/// input intrinsics fixed.
#[cfg(feature = "unstable-test-support")]
#[must_use]
pub fn minimal_global_config() -> crate::config::GlobalSfmConfig {
    let mut config = crate::config::GlobalSfmConfig::default()
        .with_release_matching_data(false)
        .with_self_calibrate_intrinsics(false);
    config.independent_track_validation = None;
    config.post_fixed_rotation_cleanup =
        crate::pipelines::global::GlobalPostFixedRotationCleanupConfig::default();
    config
}

/// SfM's internal engine boundaries and the entry points that accept them.
///
/// Tests and benchmarks script proposals, preflight deferrals, and typed failures through
/// [`seams::BundleAdjustmentBackend`]; a [`crate::ba::BundleAdjuster`] cannot express those. They
/// replace pose registration through [`seams::PoseRegistrar`], and an engine's registration,
/// averaging, and triangulation algorithms through [`seams::IncrementalBackends`] and
/// [`seams::GlobalBackends`] passed to [`crate::Runner::incremental_backends`] and
/// [`crate::Runner::global_backends`]. These are fault-injection seams, not extension points:
/// custom solvers plug in through [`crate::ba::BundleAdjuster`].
#[cfg(feature = "unstable-test-support")]
pub mod seams {
    pub use crate::ba::backend::{
        BaBackendResult, BaBackendStatus, BaPreflightResult, BundleAdjustmentBackend,
    };
    pub use crate::backends::{
        PoseRegistrar, PoseRegistrationContext, PoseRegistrationEstimate,
        PoseRegistrationObservation,
    };
    pub use crate::ops::reconstruction::bundle::{bundle_adjust, bundle_adjust_and_clean};
    pub use crate::pipelines::incremental::incremental_sfm_with;
    pub use crate::workflow::run::{GlobalBackends, IncrementalBackends};

    /// The adapter the engines put around `adjuster`: their config-to-control mapping,
    /// workspace limit, and failure attribution.
    pub fn bundle_adjustment_backend(
        adjuster: &mut dyn crate::ba::BundleAdjuster,
    ) -> impl BundleAdjustmentBackend + '_ {
        crate::ba::native::DirectBundleAdjustmentBackend::new(adjuster)
    }
}

/// Reconstruction state used only to assemble downstream adapter fixtures.
///
/// This keeps tests from depending on persistence DTOs or exposing mutable workflow state in the
/// public API.
#[derive(Debug, Default)]
pub struct WorkspaceFixtureState {
    /// Graph pairs to deactivate.
    pub inactive_pairs: Vec<crate::ViewPair>,
    /// Verified two-view models to store, keyed by a pair already present in the graph.
    pub geometric_models: Vec<(crate::ViewPair, crate::two_view::GeometricModel)>,
    /// Pose records to set.
    pub poses: Vec<crate::model::PoseRecord>,
    /// Landmarks that replace the workspace landmark set.
    pub landmarks: Vec<crate::model::Landmark>,
}

/// Applies `state` to `workspace` and revalidates it.
///
/// # Errors
///
/// Returns an `SfmError` when a pair is not in the graph, a pose or landmark is rejected, or the
/// resulting workspace fails validation.
pub fn apply_workspace_fixture_state(
    mut workspace: crate::SfmWorkspace,
    state: WorkspaceFixtureState,
) -> crate::Result<crate::SfmWorkspace> {
    for pair in state.inactive_pairs {
        workspace.pair_graph_mut().deactivate_pair(pair)?;
    }
    for (pair, model) in state.geometric_models {
        let pair_id = workspace
            .pair_graph()
            .pair_id(pair.first(), pair.second())
            .ok_or_else(|| {
                crate::SfmError::InvalidInput(format!(
                    "fixture model pair {pair} is not present in the graph"
                ))
            })?;
        workspace.geometric_models.set(pair_id.index(), model)?;
    }
    for pose in state.poses {
        workspace.set_pose(pose)?;
    }
    workspace.landmarks = crate::model::Landmarks::try_new(state.landmarks)?;
    workspace.validate()?;
    Ok(workspace)
}
#[cfg(feature = "unstable-test-support")]
mod synthetic;

#[cfg(feature = "unstable-test-support")]
pub use synthetic::{
    CameraArrangement, SceneType, SyntheticScene, SyntheticSceneBuilder, SyntheticSceneConfig,
    SyntheticSceneGenerator,
};

/// Build a checked BA proposal for orchestration contract tests.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::panic)]
pub fn bundle_solution(
    problem: &openmvg_ba::BundleProblem,
    poses: Vec<openmvg_core::model::WorldToCameraPose>,
    intrinsics: Vec<openmvg_core::model::CameraIntrinsics>,
    points: Vec<nalgebra::Point3<f64>>,
    status: openmvg_ba::SolveStatus,
    initial_objective: f64,
    final_objective: f64,
    accepted_steps: usize,
    observability: openmvg_ba::Observability,
) -> openmvg_ba::Solution {
    let diagnostics = openmvg_ba::SolveDiagnostics::new(
        accepted_steps,
        0,
        match openmvg_ba::StationarityAudit::try_from_fresh_zero_damping(0.0) {
            Ok(audit) => audit,
            Err(error) => panic!("zero stationarity must be valid: {error}"),
        },
        observability,
        0,
        0,
    );
    bundle_solution_with_diagnostics(
        problem,
        poses,
        intrinsics,
        points,
        status,
        initial_objective,
        final_objective,
        accepted_steps,
        diagnostics,
    )
}

/// Build a checked BA proposal with explicit final diagnostics.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::panic)]
pub fn bundle_solution_with_diagnostics(
    problem: &openmvg_ba::BundleProblem,
    poses: Vec<openmvg_core::model::WorldToCameraPose>,
    intrinsics: Vec<openmvg_core::model::CameraIntrinsics>,
    points: Vec<nalgebra::Point3<f64>>,
    status: openmvg_ba::SolveStatus,
    initial_objective: f64,
    final_objective: f64,
    iterations: usize,
    diagnostics: openmvg_ba::SolveDiagnostics,
) -> openmvg_ba::Solution {
    let report = match openmvg_ba::SolveReport::try_new(
        status,
        iterations,
        initial_objective,
        final_objective,
        initial_objective.sqrt(),
        final_objective.sqrt(),
        diagnostics,
    ) {
        Ok(report) => report,
        Err(error) => panic!("test report must be valid: {error}"),
    };
    match openmvg_ba::Solution::try_new(problem, poses, intrinsics, points, report) {
        Ok(solution) => solution,
        Err(error) => panic!("test solution must be valid: {error}"),
    }
}

/// Build an unchanged, committable proposal for a checked problem.
pub fn identity_bundle_solution(problem: &openmvg_ba::BundleProblem) -> openmvg_ba::Solution {
    bundle_solution(
        problem,
        problem
            .poses()
            .iter()
            .map(|camera| camera.pose().clone())
            .collect(),
        problem
            .intrinsics()
            .iter()
            .map(|intrinsic| intrinsic.intrinsics().clone())
            .collect(),
        problem
            .points()
            .iter()
            .map(|point| *point.position())
            .collect(),
        openmvg_ba::SolveStatus::Skipped,
        1.0,
        1.0,
        0,
        match problem.gauge() {
            openmvg_ba::Gauge::Free => openmvg_ba::Observability::OnlyExpectedGauge,
            _ => openmvg_ba::Observability::FullRank,
        },
    )
}
