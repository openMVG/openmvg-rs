// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transactional refinement for global-SfM candidates.
//!
//! Every fallible released-rotation or robust-cleanup proposal is accepted only
//! after numerical and verified-pose-graph checks; otherwise it is rolled back.
//!
//! This module owns the finish inputs and the staged schedule; `fixed_rotation` prepares and
//! configures the fixed-rotation solve, `self_calibration` runs the final intrinsic refinement,
//! `pose_trust` measures agreement with the verified relative-pose graph and gates proposals on
//! it, and `report` holds the serialized receipts.

use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
use crate::backends::Triangulator;
use crate::events::{EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventTransition};
use crate::ops::rollback::GeometryRollback;
use crate::progress::{SfmProgressStage, cancellation_point};
use crate::{Result, SfmError, SfmWorkspace};
use openmvg_ba::{ResidualScale, RobustLoss};
use openmvg_core::PixelThreshold;

use super::cleanup::run_post_fixed_rotation_cleanup;
#[cfg(test)]
use super::cleanup::{
    allowed_post_cleanup_views, certify_post_cleanup_topology_from_graph,
    prepared_domain_signature, prepared_domain_signature_with_landmark_prefix,
    reject_under_supported_landmarks_on_deferred_tracks,
};

mod fixed_rotation;
mod pose_trust;
mod report;
mod self_calibration;

pub(crate) use fixed_rotation::{
    FixedRotationPreparationReport, fixed_rotation_bundle_config,
    prepare_fixed_rotation_core_structure, prepare_fixed_rotation_structure,
};
pub(super) use pose_trust::{
    GraphConsistencyStats, graph_candidate_is_acceptable, graph_consistency_stats,
};
use report::global_pose_trust_report_version;
pub(super) use report::global_post_cleanup_report_version;
pub use report::{
    GlobalFinishReport, GlobalPoseTrustReport, GlobalPostFixedRotationCleanupNotAttempted,
    GlobalPostFixedRotationCleanupRejection, GlobalPostFixedRotationCleanupReport,
    GlobalPostFixedRotationCleanupStatus, GlobalRetriangulationRoundReport,
};
#[cfg(feature = "unstable-internals")]
pub use report::{
    GlobalSelfCalibrationRejection, GlobalSelfCalibrationReport, GlobalSelfCalibrationStatus,
    GlobalTrappedViewDiagnostic,
};
pub(super) use self_calibration::run_self_calibration;

/// Default pixel outlier threshold used by the global finish.
pub const GLOBAL_MAX_REPROJECTION_ERROR_PX: openmvg_core::PixelThreshold =
    openmvg_core::PixelThreshold::FOUR_PIXELS;
/// Default bound on robust retriangulation/full-BA cleanup rounds. The loop stops as soon as a
/// round leaves landmark membership unchanged, so the extra rounds cost only when they still
/// remove outliers (on MegaDepth: +2 points of cameras within 1% of the scene, +9% time).
pub const GLOBAL_ROBUST_ROUNDS: usize = 3;
/// Posed views a track needs to be (re)triangulated once the poses are admitted, as in incremental.
/// Two-view tracks carry no evidence for averaging, but each is an ordinary reprojection
/// constraint on the final bundle adjustment.
const FINISH_RETRIANGULATION_MIN_OBSERVATIONS: usize = 2;

/// Conservative adaptive-only policy for deferring a tiny set of newly trapped views.
///
/// The Global driver may opt in through its crate-private prepared-finish seam.
/// This stage acts only after a successful fixed-rotation solve and only on cameras identified
/// by the existing verified-pose-graph residual gate. It is not a general preflight cleanup for
/// cameras that the BA backend could not observe or optimize.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GlobalPostFixedRotationCleanupConfig {
    enabled: bool,
    max_views: usize,
    max_fraction: f64,
}

config_accessors!(GlobalPostFixedRotationCleanupConfig {
    /// Whether the cleanup may defer cameras.
    copy enabled, with_enabled: bool;
});

impl GlobalPostFixedRotationCleanupConfig {
    /// Maximum number of deferred cameras.
    #[must_use]
    pub const fn max_views(&self) -> usize {
        self.max_views
    }

    /// Maximum deferred fraction of the posed cameras.
    #[must_use]
    pub const fn max_fraction(&self) -> f64 {
        self.max_fraction
    }
}

impl Default for GlobalPostFixedRotationCleanupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_views: 4,
            max_fraction: 0.005,
        }
    }
}

impl GlobalPostFixedRotationCleanupConfig {
    /// Check the limits.
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidConfig`] when `max_views` is zero or `max_fraction` is not finite and in
    /// `(0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.max_views == 0 {
            return Err(SfmError::InvalidConfig(
                "post-fixed-rotation cleanup max_views must be positive".to_owned(),
            ));
        }
        if !(self.max_fraction.is_finite() && 0.0 < self.max_fraction && self.max_fraction <= 1.0) {
            return Err(SfmError::InvalidConfig(
                "post-fixed-rotation cleanup max_fraction must be finite and in (0, 1]".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Iteration budgets for the released-rotation stages of the global finish.
///
/// The fixed-rotation translation/point stage retains its dedicated policy. These limits control
/// the initial Cauchy pose/point solve and each post-retriangulation Cauchy solve, including the
/// self-calibration refit; the first self-calibrating solve keeps the default budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GlobalBaIterationLimits {
    released_rotation: usize,
    post_retriangulation: usize,
}

config_accessors!(GlobalBaIterationLimits {
    /// Iteration budget of the initial released-rotation Cauchy solve.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy released_rotation, with_released_rotation: usize;
    /// Iteration budget of each post-retriangulation Cauchy solve.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy post_retriangulation, with_post_retriangulation: usize;
});

impl Default for GlobalBaIterationLimits {
    fn default() -> Self {
        Self {
            released_rotation: 25,
            post_retriangulation: 15,
        }
    }
}

impl GlobalBaIterationLimits {
    /// Check the budgets.
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidConfig`] when either budget is zero.
    pub fn validate(self) -> Result<()> {
        if self.released_rotation == 0 || self.post_retriangulation == 0 {
            return Err(SfmError::InvalidConfig(
                "global BA iteration limits must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

struct GlobalFinishTransaction {
    geometry: GeometryRollback,
}

impl GlobalFinishTransaction {
    fn capture(workspace: &SfmWorkspace) -> Self {
        Self {
            geometry: GeometryRollback::capture(workspace),
        }
    }

    fn rollback(self, workspace: &mut SfmWorkspace) {
        self.geometry.rollback(workspace);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_global_ba_retriangulate_prepared_with_events(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    robust_triangulator: &dyn Triangulator,
    max_reproj: f64,
    max_robust_rounds: usize,
    iteration_limits: GlobalBaIterationLimits,
    preparation: FixedRotationPreparationReport,
    post_fixed_rotation_cleanup: Option<&GlobalPostFixedRotationCleanupConfig>,
    branch: SfmEventBranch,
    events: &mut EventEmitter<'_>,
) -> Result<Option<GlobalFinishReport>> {
    if workspace.registered_view_count() < 2 {
        return Ok(None);
    }
    if !max_reproj.is_finite() || max_reproj <= 0.0 {
        return Err(SfmError::InvalidInput(
            "global maximum reprojection error must be finite and positive".to_owned(),
        ));
    }
    finish_global_ba_retriangulate_inner(
        workspace,
        ba,
        robust_triangulator,
        max_reproj,
        max_robust_rounds,
        iteration_limits,
        preparation,
        post_fixed_rotation_cleanup,
        branch,
        events,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn finish_global_ba_retriangulate_inner(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    robust_triangulator: &dyn Triangulator,
    max_reproj: f64,
    max_robust_rounds: usize,
    iteration_limits: GlobalBaIterationLimits,
    preparation: FixedRotationPreparationReport,
    post_fixed_rotation_cleanup: Option<&GlobalPostFixedRotationCleanupConfig>,
    branch: SfmEventBranch,
    events: &mut EventEmitter<'_>,
) -> Result<GlobalFinishReport> {
    iteration_limits.validate()?;
    let progress = events.progress();
    cancellation_point(progress)?;
    let planned_robust_rounds = max_robust_rounds.min(3);
    progress.begin(
        SfmProgressStage::BundleAdjustment,
        2 + planned_robust_rounds,
    );
    let graph_before = graph_consistency_stats(workspace);

    let translation_config = crate::ops::reconstruction::pipeline_ba_config_for_workspace(
        workspace,
        fixed_rotation_bundle_config(),
    );
    let mut translation_and_points = workspace.bundle_adjust(ba, &translation_config)?;
    events.emit(
        branch,
        SfmEventPhase::GlobalFixedRotation,
        SfmEventTransition::Tentative,
        workspace,
    );
    let post_fixed_rotation_graph = graph_consistency_stats(workspace);
    let mut retained_fixed_rotation_graph = post_fixed_rotation_graph.clone();
    let post_fixed_rotation_cleanup = run_post_fixed_rotation_cleanup(
        workspace,
        ba,
        robust_triangulator,
        max_reproj,
        &translation_config,
        &graph_before,
        &post_fixed_rotation_graph,
        &mut translation_and_points,
        post_fixed_rotation_cleanup,
    );
    if matches!(
        &post_fixed_rotation_cleanup.outcome,
        GlobalPostFixedRotationCleanupStatus::Applied { .. }
    ) {
        retained_fixed_rotation_graph = graph_consistency_stats(workspace);
    }
    let cleanup_transition = match &post_fixed_rotation_cleanup.outcome {
        GlobalPostFixedRotationCleanupStatus::Applied { .. } => SfmEventTransition::Committed,
        GlobalPostFixedRotationCleanupStatus::Rejected { .. } => SfmEventTransition::RolledBack,
        GlobalPostFixedRotationCleanupStatus::NotAttempted { .. } => SfmEventTransition::Checkpoint,
    };
    events.emit(
        branch,
        SfmEventPhase::GlobalPostFixedRotationCleanup,
        cleanup_transition,
        workspace,
    );
    events.emit(
        branch,
        SfmEventPhase::GlobalFixedRotation,
        SfmEventTransition::Committed,
        workspace,
    );
    progress.advance(SfmProgressStage::BundleAdjustment, 1);

    let fixed_rotation_state = GlobalFinishTransaction::capture(workspace);
    // Releasing rotations adds three degrees of freedom per pose, so a camera whose translation
    // was observable can still have a free pose and would veto every other camera. Peel it with
    // the same structural passes inside the transaction: a rejected refinement restores the
    // fixed-rotation camera domain.
    let released_rotation_peel =
        crate::ops::reconstruction::remove_underdetermined_poses(workspace, 6)?;
    let released_rotation_component = if released_rotation_peel.removed_views.is_empty() {
        crate::ops::reconstruction::PoseSupportCleanupReport::default()
    } else {
        crate::ops::reconstruction::retain_primary_bundle_component(workspace)?
    };
    let mut removed_views_before_full_bundle_adjustment = released_rotation_peel.removed_views;
    removed_views_before_full_bundle_adjustment.extend(released_rotation_component.removed_views);
    removed_views_before_full_bundle_adjustment.sort_unstable();
    let full_config = crate::ops::reconstruction::pipeline_ba_config_for_workspace(
        workspace,
        BundleAdjustConfig::default()
            .with_scope(OptimizationScope::PosesAndPoints)
            .with_robust_loss(RobustLoss::Cauchy(ResidualScale::Pixels(
                PixelThreshold::FOUR_PIXELS,
            )))
            .with_max_iterations(iteration_limits.released_rotation),
    );
    let mut initial_full_ba_candidate_graph = None;
    cancellation_point(progress)?;
    let (full_bundle_adjustment, rejected_full_bundle_adjustment) =
        match workspace.bundle_adjust(ba, &full_config) {
            Ok(report) => {
                let candidate_graph = graph_consistency_stats(workspace);
                events.emit(
                    branch,
                    SfmEventPhase::GlobalBundleAdjustment,
                    SfmEventTransition::Tentative,
                    workspace,
                );
                let accepted =
                    graph_candidate_is_acceptable(&retained_fixed_rotation_graph, &candidate_graph);
                initial_full_ba_candidate_graph = Some(candidate_graph);
                if accepted {
                    events.emit(
                        branch,
                        SfmEventPhase::GlobalBundleAdjustment,
                        SfmEventTransition::Committed,
                        workspace,
                    );
                    (Some(report), false)
                } else {
                    fixed_rotation_state.rollback(workspace);
                    events.emit(
                        branch,
                        SfmEventPhase::GlobalBundleAdjustment,
                        SfmEventTransition::RolledBack,
                        workspace,
                    );
                    (None, true)
                }
            }
            Err(error) if recoverable_refinement_round_error(&error) => {
                fixed_rotation_state.rollback(workspace);
                events.emit(
                    branch,
                    SfmEventPhase::GlobalBundleAdjustment,
                    SfmEventTransition::RolledBack,
                    workspace,
                );
                (None, true)
            }
            Err(error) => return Err(error),
        };
    progress.advance(SfmProgressStage::BundleAdjustment, 1);
    if rejected_full_bundle_adjustment {
        removed_views_before_full_bundle_adjustment.clear();
    }
    let cleanup_after_full_bundle_adjustment =
        crate::ops::reconstruction::clean_outlier_landmark_support(workspace, max_reproj, 2.0)?;
    let removed_after_full_bundle_adjustment =
        cleanup_after_full_bundle_adjustment.rejected_landmarks;
    let removed_observations_after_full_bundle_adjustment =
        cleanup_after_full_bundle_adjustment.removed_observations;

    let mut robust_rounds = Vec::new();
    let mut rejected_robust_rounds = 0usize;
    let mut rejected_robust_candidate_graph = None;
    for round in 0..planned_robust_rounds {
        cancellation_point(progress)?;
        let round_graph_before = graph_consistency_stats(workspace);
        let round_state = GlobalFinishTransaction::capture(workspace);
        let retriangulation = crate::ops::reconstruction::retriangulate_incremental(
            workspace,
            robust_triangulator,
            FINISH_RETRIANGULATION_MIN_OBSERVATIONS,
            max_reproj,
            None,
        )?;
        let bundle_adjustment = match workspace.bundle_adjust(
            ba,
            &full_config
                .clone()
                .with_max_iterations(iteration_limits.post_retriangulation),
        ) {
            Ok(report) => report,
            Err(error) if recoverable_refinement_round_error(&error) => {
                round_state.rollback(workspace);
                rejected_robust_rounds += 1;
                events.emit(
                    branch,
                    SfmEventPhase::GlobalRobustRound { round },
                    SfmEventTransition::RolledBack,
                    workspace,
                );
                break;
            }
            Err(error) => return Err(error),
        };
        let cleanup =
            crate::ops::reconstruction::clean_outlier_landmark_support(workspace, max_reproj, 2.0)?;
        let round_graph_after = graph_consistency_stats(workspace);
        events.emit(
            branch,
            SfmEventPhase::GlobalRobustRound { round },
            SfmEventTransition::Tentative,
            workspace,
        );
        if !graph_candidate_is_acceptable(&round_graph_before, &round_graph_after) {
            rejected_robust_candidate_graph = Some(round_graph_after);
            round_state.rollback(workspace);
            rejected_robust_rounds += 1;
            events.emit(
                branch,
                SfmEventPhase::GlobalRobustRound { round },
                SfmEventTransition::RolledBack,
                workspace,
            );
            break;
        }
        let membership_changed = retriangulation.new_landmarks
            + retriangulation.revived_landmarks
            + retriangulation.dropped_landmarks
            + bundle_adjustment.rejected_landmarks
            + bundle_adjustment.removed_observations
            + cleanup.rejected_landmarks
            + cleanup.removed_observations
            > 0;
        robust_rounds.push(GlobalRetriangulationRoundReport {
            retriangulation,
            bundle_adjustment,
            removed_landmark_count: cleanup.rejected_landmarks,
            removed_observation_count: cleanup.removed_observations,
        });
        events.emit(
            branch,
            SfmEventPhase::GlobalRobustRound { round },
            SfmEventTransition::Committed,
            workspace,
        );
        progress.advance(SfmProgressStage::BundleAdjustment, 1);
        if !membership_changed {
            break;
        }
    }
    // Rounds that were rolled back or skipped after convergence complete the planned total.
    progress.advance(
        SfmProgressStage::BundleAdjustment,
        planned_robust_rounds - robust_rounds.len(),
    );
    progress.finish(SfmProgressStage::BundleAdjustment);

    let graph_after = graph_consistency_stats(workspace);
    events.emit(
        branch,
        SfmEventPhase::GlobalCandidateAdmission,
        SfmEventTransition::Tentative,
        workspace,
    );
    let accepted = graph_candidate_is_acceptable(&graph_before, &graph_after);
    let final_bundle_adjustment = robust_rounds
        .last()
        .map(|round| &round.bundle_adjustment)
        .or(full_bundle_adjustment.as_ref())
        .unwrap_or(&translation_and_points);
    let final_reprojection_rmse_px = (final_bundle_adjustment.ba.residual_units
        == crate::two_view::ResidualUnit::Pixels)
        .then_some(final_bundle_adjustment.ba.final_rmse);

    Ok(GlobalFinishReport {
        accepted,
        removed_landmark_count_before_translation_and_points: preparation.rejected_landmarks,
        removed_views_before_translation_and_points: preparation.removed_views,
        translation_and_points,
        full_bundle_adjustment,
        removed_views_before_full_bundle_adjustment,
        rejected_full_bundle_adjustment,
        removed_landmark_count_after_full_bundle_adjustment: removed_after_full_bundle_adjustment,
        removed_observation_count_after_full_bundle_adjustment:
            removed_observations_after_full_bundle_adjustment,
        robust_rounds,
        rejected_robust_round_count: rejected_robust_rounds,
        self_calibration: None,
        translation_graph_p90_before_deg: graph_before.translation_p90_deg,
        translation_graph_p90_after_deg: graph_after.translation_p90_deg,
        final_reprojection_rmse_px,
        positive_scale_fraction_before: graph_before.positive_scale_fraction,
        positive_scale_fraction_after: graph_after.positive_scale_fraction,
        rotation_graph_p90_before_deg: graph_before.rotation_p90_deg,
        rotation_graph_p90_after_deg: graph_after.rotation_p90_deg,
        locally_trapped_view_count_before: graph_before.locally_trapped_cameras,
        locally_trapped_view_count_after: graph_after.locally_trapped_cameras,
        post_fixed_rotation_cleanup,
        pose_trust: GlobalPoseTrustReport {
            schema_version: global_pose_trust_report_version(),
            before_trapped_views: graph_before.trapped_views,
            post_fixed_rotation_trapped_views: post_fixed_rotation_graph.trapped_views,
            initial_full_ba_candidate_trapped_views: initial_full_ba_candidate_graph
                .map(|graph| graph.trapped_views),
            rejected_robust_candidate_trapped_views: rejected_robust_candidate_graph
                .map(|graph| graph.trapped_views),
            final_candidate_trapped_views: graph_after.trapped_views,
        },
    })
}

fn recoverable_refinement_round_error(error: &SfmError) -> bool {
    match error {
        SfmError::BundleAdjustmentBackendFailed(failure) => matches!(
            failure.kind(),
            crate::BundleAdjustmentFailureKind::Numerical
                | crate::BundleAdjustmentFailureKind::ResourceExhausted
        ),
        SfmError::BundleAdjustmentNotApplicable(_)
        | SfmError::BundleAdjustmentProposalRejected { .. } => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
