// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural preparation and the bundle-adjustment policy of the fixed-rotation solve.

use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
use crate::{Result, SfmWorkspace, ViewId};
use openmvg_ba::{ResidualScale, RobustLoss};
use openmvg_core::PixelThreshold;

use super::pose_trust::graph_consistency_stats;
use crate::pipelines::global::cleanup::{
    certify_post_cleanup_topology, reject_under_supported_landmarks_on_deferred_tracks,
};

const MAX_INCIDENCE_CORE_PREFIT_DEFERRED_VIEWS: usize = 1;

#[derive(Debug, Default)]
pub(crate) struct FixedRotationPreparationReport {
    pub(crate) rejected_landmarks: usize,
    pub(crate) removed_views: Vec<ViewId>,
}

pub(crate) fn prepare_fixed_rotation_structure(
    workspace: &mut SfmWorkspace,
) -> Result<FixedRotationPreparationReport> {
    let support =
        crate::ops::reconstruction::clean_outlier_landmark_support(workspace, f64::INFINITY, 0.0)?;
    // A camera translation that stays free with every other pose fixed, and a camera cluster
    // attached to the rest through one pose or one landmark, which keeps an independent scale,
    // cannot pass exact observability, so keeping them only lets them veto every other camera.
    // Both passes are necessary conditions only; exact numerical observability remains a
    // mandatory native-BA check.
    let peel = crate::ops::reconstruction::remove_underdetermined_poses(workspace, 3)?;
    let component = crate::ops::reconstruction::retain_primary_bundle_component(workspace)?;
    let mut removed_views = peel.removed_views;
    removed_views.extend(component.removed_views);
    removed_views.sort_unstable();
    removed_views.dedup();
    Ok(FixedRotationPreparationReport {
        rejected_landmarks: support
            .rejected_landmarks
            .saturating_add(peel.rejected_landmarks)
            .saturating_add(component.rejected_landmarks),
        removed_views,
    })
}

pub(crate) fn prepare_fixed_rotation_core_structure(
    workspace: &mut SfmWorkspace,
) -> Result<FixedRotationPreparationReport> {
    // The shared structural pass isolates the incidence block that can own the problem's single
    // gauge; this is what localizes the final fallback's trapped camera. If that camera is
    // deferred, run the same structural pass once more so the graph presented to BA is still a
    // valid block.
    let first = prepare_fixed_rotation_structure(workspace)?;
    let deferred = defer_prefit_trapped_views(workspace)?;
    let (second_peel, second_component) = if deferred.is_empty() {
        (
            crate::ops::reconstruction::PoseSupportCleanupReport::default(),
            crate::ops::reconstruction::PoseSupportCleanupReport::default(),
        )
    } else {
        (
            crate::ops::reconstruction::remove_underdetermined_poses(workspace, 3)?,
            crate::ops::reconstruction::retain_primary_bundle_component(workspace)?,
        )
    };
    let mut removed_views = deferred.removed_views;
    removed_views.extend(first.removed_views);
    removed_views.extend(second_peel.removed_views);
    removed_views.extend(second_component.removed_views);
    removed_views.sort_unstable();
    removed_views.dedup();
    Ok(FixedRotationPreparationReport {
        rejected_landmarks: first
            .rejected_landmarks
            .saturating_add(deferred.rejected_landmarks)
            .saturating_add(second_peel.rejected_landmarks)
            .saturating_add(second_component.rejected_landmarks),
        removed_views,
    })
}

/// Defer an already-inconsistent camera only for the final incidence-core recovery.
///
/// Ordinary candidates keep their full camera domain. This private fallback is limited to one
/// camera and uses the same articulation/connectivity guard as post-fixed-rotation cleanup. Exact
/// BA observability and the ordinary five-degree admission gate remain mandatory afterward.
pub(super) fn defer_prefit_trapped_views(
    workspace: &mut SfmWorkspace,
) -> Result<crate::ops::reconstruction::PoseSupportCleanupReport> {
    let candidates = graph_consistency_stats(workspace)
        .trapped_views
        .into_iter()
        .map(|diagnostic| diagnostic.view)
        .collect::<Vec<_>>();
    if candidates.is_empty()
        || candidates.len() > MAX_INCIDENCE_CORE_PREFIT_DEFERRED_VIEWS
        || !workspace.components().first().is_some_and(|component| {
            matches!(
                component.gauge_scale_policy,
                crate::config::GaugeScalePolicy::Free
            )
        })
        || certify_post_cleanup_topology(workspace, &candidates).is_err()
    {
        return Ok(crate::ops::reconstruction::PoseSupportCleanupReport::default());
    }

    workspace.clear_poses_for_views(&candidates)?;
    let rejected = reject_under_supported_landmarks_on_deferred_tracks(workspace, &candidates)?;
    crate::ops::reconstruction::refresh_primary_component(workspace);
    Ok(crate::ops::reconstruction::PoseSupportCleanupReport {
        min_observations: 2,
        rounds: 1,
        removed_views: candidates,
        rejected_landmarks: rejected,
    })
}

pub(crate) fn fixed_rotation_bundle_config() -> BundleAdjustConfig {
    BundleAdjustConfig::default()
        .with_scope(OptimizationScope::TranslationsAndPoints)
        .with_robust_loss(RobustLoss::Huber(ResidualScale::Pixels(
            PixelThreshold::FOUR_PIXELS,
        )))
        // The graph gates judge this solve, so it must reach its own convergence test: averaged
        // translations can leave a slow relative-scale mode between weakly linked camera blocks.
        .with_max_iterations(100)
}
