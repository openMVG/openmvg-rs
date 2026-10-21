// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The growth loop: register views one at a time, update tracks, and run periodic bundle
//! adjustment.

use std::collections::BTreeSet;

use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::config::BundleAdjustConfig;
use crate::backends::{PoseRegistrar, Triangulator};
use crate::config::IncrementalSfmConfig;
use crate::events::{SfmEventPhase, SfmEventTransition};
use crate::ops::reconstruction::{
    TriangulationRejectionSummary, remove_unsupported_poses, retriangulate_incremental_for_views,
    triangulate_incremental_for_views,
};
use crate::pipelines::incremental::RegistrationVerdict;
use crate::progress::cancellation_point;
use crate::{Result, SfmError, SfmWorkspace, ViewId};

use super::{
    driver::IncrementalEvents,
    next_view::{GrowthConfig, select_registration_candidate_indexed},
    refinement::{
        MIN_POST_BA_POSE_SUPPORT, POST_BA_MAX_PIXEL_RESIDUAL, ReconstructionQuality,
        global_ba_quality_regressed, incremental_ba_config, incremental_self_calibration_ba_config,
    },
    registration::{
        RegistrationFailure, RegistrationFailureReason, RegistrationPreparation,
        commit_registration_outcome, evaluate_prepared_registration,
        prepare_view_registration_indexed, rotation_edge_consistency_deg,
        translation_edge_consistency_deg,
    },
    registration_index::RegistrationObservationIndex,
    report::{
        GrowthRegistrationDiagnostic, IncrementalGrowthReport, PostRegistrationSupportDiagnostic,
    },
    transaction::{BundleAdjustmentRollback, GrowthCommitTransaction},
};

/// Bundle adjustments a periodic refinement runs while its pose-support cleanup keeps removing
/// poses. Poses still unsupported after the last round are left to the next periodic or the final
/// refinement, which re-run the same cleanup; on the validation corpus one round always sufficed.
const MAX_PERIODIC_BA_ROUNDS: usize = 3;

/// Run `round` until it reports nothing more to clean or `max_rounds` rounds have run, returning
/// what each cleaning round removed.
fn repeat_while_cleaning<T>(
    max_rounds: usize,
    mut round: impl FnMut() -> Result<Option<T>>,
) -> Result<Vec<T>> {
    let mut cleanups = Vec::new();
    for _ in 0..max_rounds {
        match round()? {
            Some(cleanup) => cleanups.push(cleanup),
            None => break,
        }
    }
    Ok(cleanups)
}

fn incremental_periodic_ba_config(config: &IncrementalSfmConfig) -> BundleAdjustConfig {
    let automatic_budget = if config.self_calibrate_intrinsics() {
        25
    } else {
        BundleAdjustConfig::default().max_iterations()
    };
    let max_iterations = if config.periodic_ba_max_iterations() == 0 {
        automatic_budget
    } else {
        config.periodic_ba_max_iterations()
    };
    if config.self_calibrate_intrinsics() {
        incremental_self_calibration_ba_config()
            // Intrinsic convergence is slower than pose/point stabilization, but a full
            // 50-iteration solve at every cadence dominates runtime. Periodic solves warm-
            // start one another; reserve the full budget for the final calibration pass.
            .with_max_iterations(max_iterations)
    } else {
        incremental_ba_config()
            // A truncated growth solve can leave the camera/point system in a worse basin and
            // change the downstream registration path. Real-scene ablations showed materially
            // lower coverage and accuracy at 15/25 iterations, so fixed-intrinsic periodic BA
            // keeps the full convergence budget unless the caller explicitly overrides it.
            .with_max_iterations(max_iterations)
    }
}

fn adaptive_growth_reached(current: usize, previous: usize, ratio: f64) -> bool {
    if ratio <= 0.0 || current <= previous {
        return false;
    }
    let required_growth = ((previous as f64 * ratio).ceil() as usize).max(1);
    current >= previous.saturating_add(required_growth)
}

fn incremental_ba_cadence(config: &IncrementalSfmConfig) -> usize {
    if config.self_calibrate_intrinsics() && config.bundle_adjust_every() == 0 {
        8
    } else {
        config.bundle_adjust_every()
    }
}

#[derive(Debug, Default)]
struct GrowthBaStats {
    periodic_ba_count: usize,
    periodic_ba_attempt_count: usize,
    periodic_ba_rejected_count: usize,
    periodic_ba_iterations: usize,
    periodic_ba_snapshot_payload_peak_bytes: u128,
    adaptive_ba_attempt_count: usize,
    adaptive_ba_applied_count: usize,
    adaptive_ba_rejected_count: usize,
    post_ba_pose_support_cleanups: Vec<crate::ops::reconstruction::PoseSupportCleanupReport>,
}

#[derive(Debug, Default)]
pub(super) struct GrowthLoopReport {
    iterations: usize,
    registrations: Vec<GrowthRegistrationDiagnostic>,
    pub(super) triangulation_rejections: TriangulationRejectionSummary,
    ba: GrowthBaStats,
    frontier_selection_count: usize,
    frontier_eligible_candidate_evaluations: usize,
    frontier_support_band_candidate_evaluations: usize,
    pub(super) hit_max_iterations: bool,
}

impl GrowthLoopReport {
    pub(super) fn into_public(self, grown_view_count: usize) -> IncrementalGrowthReport {
        let GrowthBaStats {
            periodic_ba_count,
            periodic_ba_attempt_count,
            periodic_ba_rejected_count,
            periodic_ba_iterations,
            periodic_ba_snapshot_payload_peak_bytes,
            adaptive_ba_attempt_count,
            adaptive_ba_applied_count,
            adaptive_ba_rejected_count,
            post_ba_pose_support_cleanups,
        } = self.ba;
        IncrementalGrowthReport {
            grown_view_count,
            growth_registrations: self.registrations,
            iteration_count: self.iterations,
            frontier_selection_count: self.frontier_selection_count,
            frontier_eligible_candidate_evaluation_count: self
                .frontier_eligible_candidate_evaluations,
            frontier_support_band_candidate_evaluation_count: self
                .frontier_support_band_candidate_evaluations,
            triangulation_rejections: self.triangulation_rejections,
            periodic_ba_count,
            periodic_ba_attempt_count,
            periodic_ba_rejected_count,
            post_ba_pose_support_cleanups,
            periodic_ba_iteration_count: periodic_ba_iterations,
            periodic_ba_snapshot_payload_peak_bytes,
            adaptive_ba_attempt_count,
            adaptive_ba_applied_count,
            adaptive_ba_rejected_count,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct GrowthBaOutcome {
    applied: bool,
    removed_views: Vec<ViewId>,
}

#[allow(clippy::too_many_arguments)]
fn try_growth_global_ba(
    workspace: &mut SfmWorkspace,
    config: &IncrementalSfmConfig,
    reg_config: &crate::pipelines::incremental::RegistrationConfig,
    ba: &dyn BundleAdjustmentBackend,
    protected_views: &[ViewId],
    adaptive: bool,
    stats: &mut GrowthBaStats,
    events: &mut IncrementalEvents<'_, '_>,
) -> Result<GrowthBaOutcome> {
    stats.periodic_ba_attempt_count += 1;
    if adaptive {
        stats.adaptive_ba_attempt_count += 1;
    }
    let quality_before = ReconstructionQuality::capture(workspace, reg_config);
    let snapshot = BundleAdjustmentRollback::capture(workspace);
    stats.periodic_ba_snapshot_payload_peak_bytes = stats
        .periodic_ba_snapshot_payload_peak_bytes
        .max(snapshot.payload_bytes());
    let round = events.next_global_refinement();
    let progress = events.progress();
    let ba_result = (|| {
        let mut iterations = 0usize;
        let cleanups = repeat_while_cleaning(MAX_PERIODIC_BA_ROUNDS, || {
            cancellation_point(progress)?;
            let ba_config = crate::ops::reconstruction::pipeline_ba_config_for_workspace(
                workspace,
                incremental_periodic_ba_config(config),
            );
            let report = workspace.bundle_adjust(ba, &ba_config)?;
            iterations += report.ba.iteration_count;

            let cleanup = remove_unsupported_poses(
                workspace,
                protected_views,
                MIN_POST_BA_POSE_SUPPORT,
                POST_BA_MAX_PIXEL_RESIDUAL.value(),
            )?;
            Ok((!cleanup.is_empty()).then_some(cleanup))
        })?;
        Ok::<_, SfmError>((iterations, cleanups))
    })();
    match ba_result {
        Ok((iterations, cleanups)) => {
            events.emit(
                SfmEventPhase::IncrementalBundleAdjustment { round },
                SfmEventTransition::Tentative,
                workspace,
            );
            let regressed = global_ba_quality_regressed(workspace, reg_config, quality_before);
            if regressed {
                snapshot.rollback(workspace)?;
                events.emit(
                    SfmEventPhase::IncrementalBundleAdjustment { round },
                    SfmEventTransition::RolledBack,
                    workspace,
                );
                stats.periodic_ba_rejected_count += 1;
                if adaptive {
                    stats.adaptive_ba_rejected_count += 1;
                }
                Ok(GrowthBaOutcome::default())
            } else {
                stats.periodic_ba_count += 1;
                stats.periodic_ba_iterations += iterations;
                if adaptive {
                    stats.adaptive_ba_applied_count += 1;
                }
                let removed_views = cleanups
                    .iter()
                    .flat_map(|cleanup| cleanup.removed_views.iter().copied())
                    .collect::<Vec<_>>();
                for &view in &removed_views {
                    workspace.record_registration_failure(
                        crate::pipelines::incremental::RegistrationFailure {
                            view,
                            reason: RegistrationFailureReason::PostBundleInsufficientSupport,
                        },
                    );
                }
                stats.post_ba_pose_support_cleanups.extend(cleanups);
                events.emit(
                    SfmEventPhase::IncrementalBundleAdjustment { round },
                    SfmEventTransition::Committed,
                    workspace,
                );
                Ok(GrowthBaOutcome {
                    applied: true,
                    removed_views,
                })
            }
        }
        Err(error) => {
            snapshot.rollback(workspace)?;
            events.emit(
                SfmEventPhase::IncrementalBundleAdjustment { round },
                SfmEventTransition::RolledBack,
                workspace,
            );
            if error.is_recoverable_bundle_adjustment_failure() {
                stats.periodic_ba_rejected_count += 1;
                if adaptive {
                    stats.adaptive_ba_rejected_count += 1;
                }
                Ok(GrowthBaOutcome::default())
            } else {
                Err(error)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GrowthLimits {
    pub(super) max_iterations: usize,
}

/// One growth loop: repeatedly pick the highest-scoring registration candidate not on
/// the local `failed` set, register it with `registrar`/`reg_config`, and on success
/// (re-)triangulate before continuing the same sweep. Failed candidates are retried only
/// after the frontier is exhausted with net pose growth, or after a committed BA changes
/// the geometry. Returns driver iterations and accumulated triangulation rejections.
pub(super) struct GrowthPass<'a> {
    pub(super) config: &'a IncrementalSfmConfig,
    pub(super) registrar: &'a dyn PoseRegistrar,
    pub(super) registration: &'a crate::pipelines::incremental::RegistrationConfig,
    pub(super) growth: &'a GrowthConfig,
    pub(super) triangulator: &'a dyn Triangulator,
    pub(super) ba_backend: Option<&'a dyn BundleAdjustmentBackend>,
    pub(super) protected_views: &'a [ViewId],
    pub(super) limits: GrowthLimits,
}

fn rebuild_registration_index(
    index: &mut RegistrationObservationIndex,
    workspace: &SfmWorkspace,
) -> Result<()> {
    index.rebuild(workspace)?;
    Ok(())
}

/// Begin another registration sweep only after the current one both exhausted its
/// candidates and added structure. A failed view is therefore retried against a
/// materially larger map, rather than after every unrelated camera registration.
fn restart_completed_growth_sweep(
    failed: &mut BTreeSet<ViewId>,
    sweep_start_pose_count: &mut usize,
    current_pose_count: usize,
) -> bool {
    if current_pose_count <= *sweep_start_pose_count || failed.is_empty() {
        return false;
    }
    failed.clear();
    *sweep_start_pose_count = current_pose_count;
    true
}

impl GrowthPass<'_> {
    pub(super) fn run(
        self,
        workspace: &mut SfmWorkspace,
        events: &mut IncrementalEvents<'_, '_>,
    ) -> Result<GrowthLoopReport> {
        let Self {
            config,
            registrar,
            registration: reg_config,
            growth: growth_config,
            triangulator,
            ba_backend,
            protected_views,
            limits,
        } = self;
        let GrowthLimits { max_iterations } = limits;
        let max_reproj = config
            .max_reprojection_error()
            .map_or(f64::INFINITY, openmvg_core::PixelThreshold::value);
        let mut failed: BTreeSet<ViewId> = BTreeSet::new();
        let mut sweep_start_pose_count = workspace.registered_view_count();
        let mut iterations = 0usize;
        let mut registered_since_ba = 0usize;
        let mut ba_reference_registered = workspace.registered_view_count();
        let mut ba_reference_landmarks = workspace.landmarks().active_count();
        let bundle_adjust_every = incremental_ba_cadence(config);
        let mut triangulation_rejections = TriangulationRejectionSummary::default();
        let mut ba_stats = GrowthBaStats::default();
        let mut registrations = Vec::new();
        let mut frontier_selection_count = 0usize;
        let mut frontier_eligible_candidate_evaluations = 0usize;
        let mut frontier_support_band_candidate_evaluations = 0usize;
        let mut registration_index = RegistrationObservationIndex::default();
        rebuild_registration_index(&mut registration_index, workspace)?;
        let sparse_registration = reg_config.with_min_observations(reg_config.min_inliers());
        let sparse_growth = growth_config.with_min_visible_landmarks(reg_config.min_inliers());
        loop {
            cancellation_point(events.progress())?;
            if iterations >= max_iterations {
                break;
            }
            let primary_selection = select_registration_candidate_indexed(
                workspace,
                growth_config,
                &registration_index,
                &failed,
                crate::config::INCREMENTAL_FRONTIER_SUPPORT_RATIO,
            );
            let use_sparse_frontier = primary_selection.candidate.is_none()
                && sparse_growth.min_visible_landmarks() < growth_config.min_visible_landmarks();
            let selection = if use_sparse_frontier {
                let sparse_selection = select_registration_candidate_indexed(
                    workspace,
                    &sparse_growth,
                    &registration_index,
                    &failed,
                    crate::config::INCREMENTAL_FRONTIER_SUPPORT_RATIO,
                );
                frontier_eligible_candidate_evaluations += primary_selection.eligible_candidates;
                frontier_support_band_candidate_evaluations +=
                    primary_selection.support_band_candidates;
                sparse_selection
            } else {
                primary_selection
            };
            let active_growth = if use_sparse_frontier {
                &sparse_growth
            } else {
                growth_config
            };
            let active_registration = if use_sparse_frontier {
                &sparse_registration
            } else {
                reg_config
            };
            frontier_selection_count += 1;
            frontier_eligible_candidate_evaluations += selection.eligible_candidates;
            frontier_support_band_candidate_evaluations += selection.support_band_candidates;
            let Some(candidate) = selection.candidate else {
                if restart_completed_growth_sweep(
                    &mut failed,
                    &mut sweep_start_pose_count,
                    workspace.registered_view_count(),
                ) {
                    continue;
                }
                break;
            };

            iterations += 1;
            let preparation = prepare_view_registration_indexed(
                workspace,
                &registration_index,
                candidate.view,
                active_registration,
            )?;
            let outcome = match preparation {
                RegistrationPreparation::Ready(prepared) => evaluate_prepared_registration(
                    workspace,
                    &prepared,
                    registrar,
                    active_registration,
                )?,
                RegistrationPreparation::Finished(outcome) => outcome,
            };

            let (registration, transaction) = match outcome {
                RegistrationVerdict::Registered(report) => {
                    let transaction = GrowthCommitTransaction::capture(workspace, report.view)?;
                    match commit_registration_outcome(
                        workspace,
                        RegistrationVerdict::Registered(report),
                    ) {
                        Ok(RegistrationVerdict::Registered(report)) => (report, transaction),
                        Ok(_) => {
                            transaction.rollback(workspace)?;
                            return Err(SfmError::InvariantViolation(
                                "registration commit changed a registered outcome".to_owned(),
                            ));
                        }
                        Err(error) => {
                            transaction.rollback(workspace)?;
                            return Err(error);
                        }
                    }
                }
                outcome @ (RegistrationVerdict::Rejected(_) | RegistrationVerdict::Deferred(_)) => {
                    let view = match &outcome {
                        RegistrationVerdict::Rejected(failure)
                        | RegistrationVerdict::Deferred(failure) => failure.view,
                        RegistrationVerdict::Registered(_) => {
                            return Err(SfmError::InvariantViolation(
                                "registered outcome reached the rejection path".to_owned(),
                            ));
                        }
                    };
                    commit_registration_outcome(workspace, outcome)?;
                    registration_index.record_failure(view)?;
                    failed.insert(view);
                    continue;
                }
            };
            let rotation_edge_error_deg = rotation_edge_consistency_deg(
                workspace,
                registration.view,
                registration.pose.rotation(),
            );
            let translation_edge_error_deg = translation_edge_consistency_deg(
                workspace,
                registration.view,
                registration.pose.center(),
                registration.pose.rotation(),
            );

            let changed_views = [registration.view];
            let track_update = if config.retriangulate() {
                retriangulate_incremental_for_views(
                    workspace,
                    triangulator,
                    config.triangulation_min_observations(),
                    max_reproj,
                    &changed_views,
                )
                .map(|report| {
                    triangulation_rejections.merge(report.refit_rejections);
                    triangulation_rejections.merge(report.growth_rejections);
                    report.attempted_refits > 0
                })
            } else {
                triangulate_incremental_for_views(
                    workspace,
                    triangulator,
                    config.triangulation_min_observations(),
                    max_reproj,
                    &changed_views,
                )
                .map(|report| {
                    triangulation_rejections.merge(report.rejections);
                    false
                })
            };
            let refit_performed = match track_update {
                Ok(refit_performed) => refit_performed,
                Err(error) => {
                    transaction.rollback(workspace)?;
                    return Err(error);
                }
            };
            if let Err(error) = registration_index.sync_view(workspace, registration.view) {
                transaction.rollback(workspace)?;
                return Err(error);
            }
            let support_counts = refit_performed
                .then(|| registration_index.support_counts(workspace, registration.view));
            let required_post_refit_support = active_growth.min_visible_landmarks();
            if !post_refit_support_is_sufficient(support_counts, required_post_refit_support) {
                let view = registration.view;
                transaction.rollback(workspace)?;
                registration_index.sync_view(workspace, view)?;
                commit_registration_outcome(
                    workspace,
                    RegistrationVerdict::Rejected(RegistrationFailure {
                        view,
                        reason: RegistrationFailureReason::PostRefitInsufficientSupport,
                    }),
                )?;
                registration_index.record_failure(view)?;
                failed.insert(view);
                continue;
            }

            // Only a transaction that passed the support check may affect BA cadence or emit a committed
            // registration event.
            registered_since_ba += 1;

            drop(transaction);
            events.emit_registration(workspace);

            let mut periodic_ba_after = false;
            let mut adaptive_ba_after = false;
            if let Some(ba) = ba_backend {
                let cadence_reached =
                    bundle_adjust_every > 0 && registered_since_ba >= bundle_adjust_every;
                let adaptive_reached = adaptive_growth_reached(
                    workspace.registered_view_count(),
                    ba_reference_registered,
                    config.bundle_adjust_growth_ratio(),
                ) || adaptive_growth_reached(
                    workspace.landmarks().active_count(),
                    ba_reference_landmarks,
                    config.bundle_adjust_growth_ratio(),
                );
                if cadence_reached || adaptive_reached {
                    let adaptive_only = adaptive_reached && !cadence_reached;
                    let outcome = try_growth_global_ba(
                        workspace,
                        config,
                        reg_config,
                        ba,
                        protected_views,
                        adaptive_only,
                        &mut ba_stats,
                        events,
                    )?;
                    periodic_ba_after = outcome.applied;
                    adaptive_ba_after = adaptive_only && outcome.applied;
                    registered_since_ba = 0;
                    ba_reference_registered = workspace.registered_view_count();
                    ba_reference_landmarks = workspace.landmarks().active_count();
                    if outcome.applied {
                        rebuild_registration_index(&mut registration_index, workspace)?;
                        failed.clear();
                        sweep_start_pose_count = workspace.registered_view_count();
                        failed.extend(outcome.removed_views);
                    }
                }
            }

            registrations.push(GrowthRegistrationDiagnostic {
                view: registration.view,
                method: registration.method,
                visible_landmark_count: candidate.visible_landmarks,
                visible_image_cell_count: candidate.occupied_image_cells,
                verified_registered_neighbor_count: candidate.verified_registered_neighbors,
                track_observation_count: candidate.track_observations,
                previous_failure_count: candidate.previous_failures,
                attempted_observation_count: registration.attempted_observations,
                inlier_count: registration.inliers,
                residual_median: registration.residual_median,
                rotation_edge_error_deg,
                translation_edge_error_deg,
                pre_ba_post_refit_support: support_counts.map(|support| {
                    PostRegistrationSupportDiagnostic {
                        active_landmark_count: support.active_landmarks,
                        supporting_landmark_count: support.supporting_landmarks,
                    }
                }),
                adaptive_ba_after,
                periodic_ba_after,
            });
        }
        Ok(GrowthLoopReport {
            iterations,
            registrations,
            triangulation_rejections,
            ba: ba_stats,
            frontier_selection_count,
            frontier_eligible_candidate_evaluations,
            frontier_support_band_candidate_evaluations,
            hit_max_iterations: iterations >= max_iterations,
        })
    }
}

fn post_refit_support_is_sufficient(
    support: Option<super::registration_index::ImageSupportCounts>,
    required: usize,
) -> bool {
    support.is_none_or(|support| support.supporting_landmarks >= required)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ba::config::OptimizationScope;
    use crate::pipelines::incremental::registration_index::ImageSupportCounts;

    #[test]
    fn periodic_cleanup_stops_when_clean_or_after_the_round_cap() {
        let mut calls = 0;
        let removals = repeat_while_cleaning(MAX_PERIODIC_BA_ROUNDS, || {
            calls += 1;
            Ok(Some(calls))
        })
        .unwrap();
        assert_eq!(removals, vec![1, 2, 3]);
        assert_eq!(calls, MAX_PERIODIC_BA_ROUNDS);

        let mut calls = 0;
        let removals = repeat_while_cleaning(MAX_PERIODIC_BA_ROUNDS, || {
            calls += 1;
            Ok((calls < 2).then_some(calls))
        })
        .unwrap();
        assert_eq!(removals, vec![1]);
        assert_eq!(calls, 2);

        let failed: Result<Vec<usize>> = repeat_while_cleaning(MAX_PERIODIC_BA_ROUNDS, || {
            Err(SfmError::InvalidInput("round failed".to_owned()))
        });
        assert!(failed.is_err());
    }

    #[test]
    fn post_refit_support_uses_the_active_frontier_floor() {
        let ten_supporting = Some(ImageSupportCounts {
            active_landmarks: 17,
            supporting_landmarks: 10,
        });
        assert!(!post_refit_support_is_sufficient(ten_supporting, 12));
        assert!(post_refit_support_is_sufficient(ten_supporting, 8));
        assert!(post_refit_support_is_sufficient(None, 12));
    }

    #[test]
    fn adaptive_ba_growth_uses_relative_camera_or_landmark_increase() {
        assert!(!adaptive_growth_reached(10, 10, 0.1));
        assert!(!adaptive_growth_reached(10, 10, 0.0));
        assert!(!adaptive_growth_reached(10, 10, -0.1));
        assert!(adaptive_growth_reached(11, 10, 0.1));
        assert!(!adaptive_growth_reached(109, 100, 0.1));
        assert!(adaptive_growth_reached(110, 100, 0.1));
        // Reconstructions the size of the initial pair trigger on the first added view,
        // matching COLMAP's relative-growth policy.
        assert!(adaptive_growth_reached(3, 2, 0.1));
    }

    #[test]
    fn completed_growth_sweep_retries_failures_only_after_progress() {
        let failed_view = ViewId::new(3);
        let mut failed = BTreeSet::from([failed_view]);
        let mut sweep_start_pose_count = 2;

        assert!(!restart_completed_growth_sweep(
            &mut failed,
            &mut sweep_start_pose_count,
            2,
        ));
        assert!(failed.contains(&failed_view));

        assert!(restart_completed_growth_sweep(
            &mut failed,
            &mut sweep_start_pose_count,
            4,
        ));
        assert!(failed.is_empty());
        assert_eq!(sweep_start_pose_count, 4);

        failed.insert(failed_view);
        assert!(!restart_completed_growth_sweep(
            &mut failed,
            &mut sweep_start_pose_count,
            4,
        ));
    }

    #[test]
    fn unknown_intrinsics_enable_periodic_calibration_before_final_ba() {
        let fixed_intrinsics =
            IncrementalSfmConfig::default().with_self_calibrate_intrinsics(false);
        assert_eq!(
            incremental_periodic_ba_config(&fixed_intrinsics).max_iterations(),
            BundleAdjustConfig::default().max_iterations()
        );

        let mut config = IncrementalSfmConfig::default().with_self_calibrate_intrinsics(true);
        assert_eq!(incremental_ba_cadence(&config), 8);
        assert_eq!(
            incremental_periodic_ba_config(&config).scope(),
            OptimizationScope::SelfCalibration
        );
        assert_eq!(incremental_periodic_ba_config(&config).max_iterations(), 25);
        assert_eq!(
            incremental_self_calibration_ba_config().max_iterations(),
            BundleAdjustConfig::default().max_iterations()
        );

        config = config.with_bundle_adjust_every(12);
        assert_eq!(incremental_ba_cadence(&config), 12);
        config.periodic_ba_max_iterations = 31;
        assert_eq!(incremental_periodic_ba_config(&config).max_iterations(), 31);
    }
}
