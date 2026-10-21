// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bundle-adjustment policy and quality gates shared by growth, and the final
//! refinement.

use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::config::{BundleAdjustConfig, OptimizationScope};
use crate::backends::Triangulator;
use crate::config::IncrementalSfmConfig;
use crate::events::{SfmEventPhase, SfmEventTransition};
use crate::ops::reconstruction::{
    angular_regression_limit, bundle_adjust_and_clean_report_skipping_negligible_verification,
    camera_bearing_error_rad, recover_missing_tracks, remove_unsupported_poses,
};
use crate::progress::{SfmProgressStage, cancellation_point};
use crate::{Result, SfmWorkspace, ViewId};
use openmvg_ba::{ResidualScale, RobustLoss};
use openmvg_core::PixelThreshold;
use openmvg_core::model::WorldToCameraPose;

use super::{
    driver::IncrementalEvents,
    registration::{
        RegistrationFailureReason, rotation_edge_consistency_deg, translation_edge_consistency_deg,
    },
    report::{IncrementalFinalRefinementReport, IncrementalFinalRefinementStatus},
    transaction::BundleAdjustmentRollback,
};

/// Share of active landmarks a cleanup round may reject and still count as negligible: after the
/// required rounds, a negligible round ends the final refinement without another verification BA.
const FINAL_REFINEMENT_NEGLIGIBLE_REMOVAL_FRACTION: f64 = 1e-4;
/// Smallest triangulation angle, in degrees, a landmark keeps through cleanup; flatter points
/// constrain depth too weakly to support the poses that observe them.
const CLEANUP_MIN_TRIANGULATION_ANGLE_DEG: f64 = 2.0;
/// Reprojection-valid observations a non-protected pose needs after BA. Six is the minimal DLT
/// resection; with fewer, the pose is no longer constrained on its own and is removed.
pub(super) const MIN_POST_BA_POSE_SUPPORT: usize = 6;
/// Reprojection limit for the cleanup after BA, the same 4 px as the robust scale of
/// [`incremental_ba_config`] and the default triangulation limit.
pub(super) const POST_BA_MAX_PIXEL_RESIDUAL: PixelThreshold = PixelThreshold::FOUR_PIXELS;

/// Robust BA config for the incremental loop: Cauchy (redescending) loss.
///
/// A redescending loss suits poses that are already well initialized, as they are after P3P;
/// the convex Huber loss is the default for less certain starting points.
pub(super) fn incremental_ba_config() -> BundleAdjustConfig {
    BundleAdjustConfig::default()
        // Cauchy at 4 px: a robust loss loose enough not to over-penalize valid
        // full-resolution observations (the OpenMVG-style clean loop removes the gross
        // outliers this only down-weights). A tighter 1 px scale over-penalizes inliers.
        .with_robust_loss(RobustLoss::Cauchy(ResidualScale::Pixels(
            PixelThreshold::FOUR_PIXELS,
        )))
}

/// Run the single default cleanup policy.
///
/// A final verification BA is skipped only after the scene has completed the required
/// cleanup rounds and the latest removal is negligible. Outliers are never pre-removed
/// after retriangulation: corpus ablation showed that doing so can change accepted poses.
fn incremental_adjust_and_clean_report(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    config: &BundleAdjustConfig,
    max_rounds: usize,
    minimum_rounds_before_skip: usize,
) -> Result<crate::ops::reconstruction::BundleAdjustAndCleanReport> {
    bundle_adjust_and_clean_report_skipping_negligible_verification(
        workspace,
        ba,
        config,
        POST_BA_MAX_PIXEL_RESIDUAL.value(),
        CLEANUP_MIN_TRIANGULATION_ANGLE_DEG,
        max_rounds,
        minimum_rounds_before_skip,
        FINAL_REFINEMENT_NEGLIGIBLE_REMOVAL_FRACTION,
    )
}

/// Observable, conservative intrinsic refinement used during growth. Huber keeps a
/// non-zero gradient for the initially large residuals caused by an unknown lens;
/// the selected bundle-adjustment backend controls the exact intrinsic scope.
pub(super) fn incremental_self_calibration_ba_config() -> BundleAdjustConfig {
    BundleAdjustConfig::default()
        .with_scope(OptimizationScope::SelfCalibration)
        .with_robust_loss(RobustLoss::Huber(ResidualScale::Pixels(
            PixelThreshold::FOUR_PIXELS,
        )))
}

/// Fresh angular bearing RMSE over current poses, active landmarks, and supporting observations.
/// Every calibrated camera model is evaluated in radians so scores remain comparable across
/// perspective, fisheye, spherical, and mixed-camera reconstructions.
/// Recomputed from the live state (not the cached per-landmark value), so it reflects the
/// solution currently in the workspace. Used by the BA reconstruction-quality regression gate.
#[derive(Debug, Clone, Copy)]
struct BearingQuality {
    rmse_rad: f64,
    invalid_observations: usize,
}

fn fresh_bearing_quality(workspace: &SfmWorkspace) -> BearingQuality {
    let mut sse = 0.0;
    let mut n = 0usize;
    let mut invalid_observations = 0usize;
    for lm in workspace.landmarks().iter_active() {
        let x = lm.point;
        let Some(track) = workspace.tracks().get(lm.source_track) else {
            invalid_observations += 1;
            continue;
        };
        let mut supporting_observations = 0usize;
        for (_, obs) in lm.supporting_observations(track) {
            supporting_observations += 1;
            let Some(rec) = workspace.pose(obs.view) else {
                invalid_observations += 1;
                continue;
            };
            let Some(img) = workspace.view(obs.view) else {
                invalid_observations += 1;
                continue;
            };
            let Some(camera) = workspace.camera_model(img.intrinsic_id) else {
                invalid_observations += 1;
                continue;
            };
            let pc = rec.pose.rotation() * (x - rec.pose.center());
            if !pc.iter().all(|value| value.is_finite()) {
                invalid_observations += 1;
                continue;
            }
            let Ok(pixel) = workspace.track_observation_pixel(obs) else {
                invalid_observations += 1;
                continue;
            };
            let Some(residual) = camera_bearing_error_rad(camera, &pc, pixel) else {
                invalid_observations += 1;
                continue;
            };
            let squared_error = residual * residual;
            if !squared_error.is_finite() {
                invalid_observations += 1;
                continue;
            }
            sse += squared_error;
            n += 1;
        }
        if supporting_observations == 0 {
            invalid_observations += 1;
        }
    }
    let rmse = if n == 0 { 0.0 } else { (sse / n as f64).sqrt() };
    BearingQuality {
        rmse_rad: rmse,
        invalid_observations,
    }
}

fn ba_edge_violation_count(
    workspace: &SfmWorkspace,
    reg_config: &crate::pipelines::incremental::RegistrationConfig,
) -> usize {
    workspace
        .registered_views()
        .filter(|(view, record)| {
            registration_pose_violates_edges(workspace, *view, &record.pose, reg_config)
        })
        .count()
}

fn registration_pose_violates_edges(
    workspace: &SfmWorkspace,
    view: ViewId,
    pose: &WorldToCameraPose,
    reg_config: &crate::pipelines::incremental::RegistrationConfig,
) -> bool {
    reg_config
        .max_rotation_edge_error()
        .is_some_and(|threshold| {
            rotation_edge_consistency_deg(workspace, view, pose.rotation())
                .is_some_and(|error| error > threshold.value().to_degrees())
        })
        || reg_config
            .max_translation_edge_error()
            .is_some_and(|threshold| {
                translation_edge_consistency_deg(workspace, view, pose.center(), pose.rotation())
                    .is_some_and(|error| error > threshold.value().to_degrees())
            })
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ReconstructionQuality {
    landmarks: usize,
    bearing: BearingQuality,
    edge_violations: usize,
}

impl ReconstructionQuality {
    pub(super) fn capture(
        workspace: &SfmWorkspace,
        reg_config: &crate::pipelines::incremental::RegistrationConfig,
    ) -> Self {
        Self {
            landmarks: workspace.landmarks().active_count(),
            bearing: fresh_bearing_quality(workspace),
            edge_violations: ba_edge_violation_count(workspace, reg_config),
        }
    }

    fn regressed_from(self, before: Self) -> bool {
        let bearing_limit = angular_regression_limit(before.bearing.rmse_rad);
        self.landmarks.saturating_mul(2) < before.landmarks
            || !self.bearing.rmse_rad.is_finite()
            || self.bearing.rmse_rad > bearing_limit
            || self.bearing.invalid_observations > before.bearing.invalid_observations
            || self.edge_violations > before.edge_violations
    }
}

pub(super) fn global_ba_quality_regressed(
    workspace: &SfmWorkspace,
    reg_config: &crate::pipelines::incremental::RegistrationConfig,
    before: ReconstructionQuality,
) -> bool {
    ReconstructionQuality::capture(workspace, reg_config).regressed_from(before)
}

#[derive(Debug, Default)]
pub(super) struct FinalRefinementOutcome {
    pub(super) cleaned_landmarks: usize,
    pub(super) rejected_count: usize,
    pub(super) retriangulation: Option<crate::ops::reconstruction::RetriangulationReport>,
    pub(super) report: Option<Box<IncrementalFinalRefinementReport>>,
}

pub(super) fn run_final_refinement(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    registration: &crate::pipelines::incremental::RegistrationConfig,
    triangulator: &dyn Triangulator,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
    protected_views: &[ViewId],
    events: &mut IncrementalEvents<'_, '_>,
) -> Result<FinalRefinementOutcome> {
    let Some(ba) = ba_backend else {
        return Ok(FinalRefinementOutcome::default());
    };
    if workspace.registered_view_count() < 2 {
        return Ok(FinalRefinementOutcome::default());
    }
    cancellation_point(events.progress())?;
    events
        .progress()
        .begin(SfmProgressStage::BundleAdjustment, 1);

    let quality_before = ReconstructionQuality::capture(workspace, registration);
    let snapshot = BundleAdjustmentRollback::capture(workspace);
    let mut self_calibration = None;
    let mut before_retriangulation = None;
    let mut pose_support_cleanups = Vec::new();
    let mut post_pose_cleanup_adjustments = Vec::new();
    let result: Result<(
        usize,
        Option<crate::ops::reconstruction::RetriangulationReport>,
    )> = (|| {
        if config.self_calibrate_intrinsics() {
            self_calibration =
                Some(workspace.bundle_adjust(ba, &incremental_self_calibration_ba_config())?);
        }
        let ba_config = crate::ops::reconstruction::pipeline_ba_config_for_workspace(
            workspace,
            incremental_ba_config(),
        );
        let before = incremental_adjust_and_clean_report(workspace, ba, &ba_config, 4, 2)?;
        let mut cleaned = before.total_removed_landmarks;
        before_retriangulation = Some(before);
        loop {
            let cleanup = remove_unsupported_poses(
                workspace,
                protected_views,
                MIN_POST_BA_POSE_SUPPORT,
                POST_BA_MAX_PIXEL_RESIDUAL.value(),
            )?;
            if cleanup.is_empty() {
                break;
            }
            cleaned += cleanup.rejected_landmarks;
            for &view in &cleanup.removed_views {
                workspace.record_registration_failure(
                    crate::pipelines::incremental::RegistrationFailure {
                        view,
                        reason: RegistrationFailureReason::PostBundleInsufficientSupport,
                    },
                );
            }
            pose_support_cleanups.push(cleanup);
            let adjustment = incremental_adjust_and_clean_report(workspace, ba, &ba_config, 4, 2)?;
            cleaned += adjustment.total_removed_landmarks;
            post_pose_cleanup_adjustments.push(adjustment);
        }
        let mut retriangulation = None;
        if config.retriangulate() {
            let max_reproj = config
                .max_reprojection_error()
                .map_or(4.0, openmvg_core::PixelThreshold::value);
            // The preceding BA jointly polished every active camera and landmark. Recover only
            // still-unmaterialized tracks: robust triangulation validates their independent point
            // estimates, so adding them cannot invalidate the polished geometry and does not
            // justify another whole-scene solve.
            let report = recover_missing_tracks(
                workspace,
                triangulator,
                config.triangulation_min_observations(),
                max_reproj,
            )?;
            retriangulation = Some(report);
        }
        Ok((cleaned, retriangulation))
    })();

    let accepted =
        result.is_ok() && !global_ba_quality_regressed(workspace, registration, quality_before);
    let final_report = |outcome| {
        Some(Box::new(IncrementalFinalRefinementReport {
            schema_version: 1,
            outcome,
            self_calibration,
            before_retriangulation,
            pose_support_cleanups,
            post_pose_cleanup_adjustments,
        }))
    };

    if result.is_ok() {
        events.emit(
            SfmEventPhase::IncrementalFinalBundleAdjustment,
            SfmEventTransition::Tentative,
            workspace,
        );
    }

    let outcome = match result {
        Ok((cleaned_landmarks, retriangulation)) if accepted => {
            events.emit(
                SfmEventPhase::IncrementalFinalBundleAdjustment,
                SfmEventTransition::Committed,
                workspace,
            );
            Ok(FinalRefinementOutcome {
                cleaned_landmarks,
                rejected_count: 0,
                retriangulation,
                report: final_report(IncrementalFinalRefinementStatus::Applied),
            })
        }
        Ok(_) => {
            snapshot.rollback(workspace)?;
            events.emit(
                SfmEventPhase::IncrementalFinalBundleAdjustment,
                SfmEventTransition::RolledBack,
                workspace,
            );
            Ok(FinalRefinementOutcome {
                rejected_count: 1,
                report: final_report(IncrementalFinalRefinementStatus::RejectedRolledBack),
                ..FinalRefinementOutcome::default()
            })
        }
        Err(error) => {
            snapshot.rollback(workspace)?;
            events.emit(
                SfmEventPhase::IncrementalFinalBundleAdjustment,
                SfmEventTransition::RolledBack,
                workspace,
            );
            if error.is_recoverable_bundle_adjustment_failure() {
                Ok(FinalRefinementOutcome {
                    rejected_count: 1,
                    report: final_report(IncrementalFinalRefinementStatus::RejectedRolledBack),
                    ..FinalRefinementOutcome::default()
                })
            } else {
                Err(error)
            }
        }
    };
    if outcome.is_ok() {
        events
            .progress()
            .advance(SfmProgressStage::BundleAdjustment, 1);
        events.progress().finish(SfmProgressStage::BundleAdjustment);
    }
    outcome
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::ComponentId;
    use crate::model::{ImageMeta, Landmark, Landmarks, ObservationSupport, PoseRecord};
    use crate::{
        Camera, FeatureId, FeatureSet, IntrinsicId, LandmarkId, Track, TrackId, TrackObservation,
        TrackSet,
    };
    use nalgebra::{Point2, Point3, Rotation3};

    pub(crate) fn workspace_with_filtered_outlier() -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
                ImageMeta::external_id("c", 64, 48),
            ])
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .expect("valid workspace");
        for (index, center_x) in [0.0, 1.0, 2.0].into_iter().enumerate() {
            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(index),
                    pose: WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::new(center_x, 0.0, 0.0),
                    )
                    .unwrap(),
                    component: Some(ComponentId::new(0)),
                })
                .expect("pose index is within the workspace view domain");
        }
        workspace.tracks = TrackSet::try_new(vec![
            Track::try_new(
                TrackId::new(0),
                vec![
                    TrackObservation {
                        view: ViewId::new(0),
                        feature: FeatureId::new(0),
                    },
                    TrackObservation {
                        view: ViewId::new(1),
                        feature: FeatureId::new(0),
                    },
                    TrackObservation {
                        view: ViewId::new(2),
                        feature: FeatureId::new(0),
                    },
                ],
            )
            .expect("valid track"),
        ])
        .expect("valid tracks");
        for (view, pixel) in [
            Point2::new(32.0, 24.0),
            Point2::new(12.0, 24.0),
            Point2::new(500.0, -300.0),
        ]
        .into_iter()
        .enumerate()
        {
            workspace
                .feature_sets
                .set(view, FeatureSet::try_new(vec![pixel]).unwrap())
                .unwrap();
        }
        workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
            LandmarkId::new(0),
            TrackId::new(0),
            Point3::new(0.0, 0.0, 5.0),
            openmvg_core::PixelResidual::ZERO,
            ObservationSupport::try_from_indices([0, 1], 3).expect("valid support"),
        )])
        .expect("valid landmarks");
        workspace
    }

    #[test]
    fn bearing_quality_scores_only_supporting_observations() {
        let workspace = workspace_with_filtered_outlier();
        let quality = fresh_bearing_quality(&workspace);
        assert!(quality.rmse_rad < 1e-12);
        assert_eq!(quality.invalid_observations, 0);
    }
}
