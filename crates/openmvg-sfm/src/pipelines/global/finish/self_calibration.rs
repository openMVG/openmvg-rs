// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The final transactional self-calibration of an admitted global candidate.

use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
use crate::backends::Triangulator;
use crate::events::{EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventTransition};
use crate::{Result, SfmWorkspace};
use openmvg_ba::{ResidualScale, RobustLoss};
use openmvg_core::PixelThreshold;

use super::{
    FINISH_RETRIANGULATION_MIN_OBSERVATIONS, GlobalBaIterationLimits, GlobalFinishTransaction,
    pose_trust::{graph_candidate_is_acceptable, graph_consistency_stats},
    recoverable_refinement_round_error,
    report::{
        GlobalRetriangulationRoundReport, GlobalSelfCalibrationRejection,
        GlobalSelfCalibrationReport, GlobalSelfCalibrationStatus,
    },
};

/// Free each intrinsic group's declared camera model in a final BA, then retriangulate and
/// re-adjust under the refined intrinsics. Runs once on an admitted candidate: admission judged
/// the core against pairwise geometry verified under the input intrinsics, which cannot judge
/// recalibrated poses. Every pose and point stays free, so the stage is transactional like a
/// robust round: a numerical failure or a candidate that degrades the verified relative-pose graph
/// restores the admitted reconstruction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_self_calibration(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    robust_triangulator: &dyn Triangulator,
    max_reproj: f64,
    iteration_limits: GlobalBaIterationLimits,
    branch: SfmEventBranch,
    events: &mut EventEmitter<'_>,
) -> Result<GlobalSelfCalibrationStatus> {
    let graph_before = graph_consistency_stats(workspace);
    let state = GlobalFinishTransaction::capture(workspace);
    let config = BundleAdjustConfig::default()
        .with_scope(OptimizationScope::SelfCalibration)
        .with_robust_loss(RobustLoss::Cauchy(ResidualScale::Pixels(
            PixelThreshold::FOUR_PIXELS,
        )));
    let refit_config = config
        .clone()
        .with_max_iterations(iteration_limits.post_retriangulation);
    let stage = (|| -> Result<_> {
        let bundle_adjustment = workspace.bundle_adjust(ba, &config)?;
        let cleanup =
            crate::ops::reconstruction::clean_outlier_landmark_support(workspace, max_reproj, 2.0)?;
        let retriangulation = crate::ops::reconstruction::retriangulate_incremental(
            workspace,
            robust_triangulator,
            FINISH_RETRIANGULATION_MIN_OBSERVATIONS,
            max_reproj,
            None,
        )?;
        let refit = workspace.bundle_adjust(ba, &refit_config)?;
        let refit_cleanup =
            crate::ops::reconstruction::clean_outlier_landmark_support(workspace, max_reproj, 2.0)?;
        Ok((
            bundle_adjustment,
            cleanup,
            GlobalRetriangulationRoundReport {
                retriangulation,
                bundle_adjustment: refit,
                removed_landmark_count: refit_cleanup.rejected_landmarks,
                removed_observation_count: refit_cleanup.removed_observations,
            },
        ))
    })();
    let roll_back = |workspace: &mut SfmWorkspace, events: &mut EventEmitter<'_>| {
        state.rollback(workspace);
        events.emit(
            branch,
            SfmEventPhase::GlobalSelfCalibration,
            SfmEventTransition::RolledBack,
            workspace,
        );
    };
    let (bundle_adjustment, cleanup, retriangulation) = match stage {
        Ok(stage) => stage,
        Err(error) if recoverable_refinement_round_error(&error) => {
            roll_back(workspace, events);
            return Ok(GlobalSelfCalibrationStatus::RolledBack(
                GlobalSelfCalibrationRejection::Refinement {
                    error: error.to_string(),
                },
            ));
        }
        Err(error) => return Err(error),
    };
    events.emit(
        branch,
        SfmEventPhase::GlobalSelfCalibration,
        SfmEventTransition::Tentative,
        workspace,
    );
    let graph_after = graph_consistency_stats(workspace);
    if !graph_candidate_is_acceptable(&graph_before, &graph_after) {
        let rejection = GlobalSelfCalibrationRejection::GraphDegraded {
            translation_graph_p90_before_deg: graph_before.translation_p90_deg,
            translation_graph_p90_after_deg: graph_after.translation_p90_deg,
            rotation_graph_p90_before_deg: graph_before.rotation_p90_deg,
            rotation_graph_p90_after_deg: graph_after.rotation_p90_deg,
            locally_trapped_view_count_before: graph_before.locally_trapped_cameras,
            locally_trapped_view_count_after: graph_after.locally_trapped_cameras,
            trapped_views_after: graph_after.trapped_views.clone(),
        };
        roll_back(workspace, events);
        return Ok(GlobalSelfCalibrationStatus::RolledBack(rejection));
    }
    events.emit(
        branch,
        SfmEventPhase::GlobalSelfCalibration,
        SfmEventTransition::Committed,
        workspace,
    );
    Ok(GlobalSelfCalibrationStatus::Committed(Box::new(
        GlobalSelfCalibrationReport {
            bundle_adjustment,
            removed_landmark_count: cleanup.rejected_landmarks,
            removed_observation_count: cleanup.removed_observations,
            retriangulation,
            translation_graph_p90_after_deg: graph_after.translation_p90_deg,
            rotation_graph_p90_after_deg: graph_after.rotation_p90_deg,
        },
    )))
}
