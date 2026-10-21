// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Conservative camera deferral after fixed-rotation refinement.

use std::collections::BTreeSet;

use openmvg_core::graph::analyze_undirected_topology;

use crate::ba::backend::{
    BaPreflightDecision, BaPreflightReport, BaRankAssessment, BundleAdjustmentBackend,
};
use crate::ba::config::BundleAdjustConfig;
use crate::backends::Triangulator;
use crate::model::LandmarkStatus;
use crate::ops::rollback::GeometryRollback;
use crate::{Result, SfmError, SfmWorkspace, ViewId};

use super::{
    bounded_deferred_view_count,
    finish::{
        GlobalPostFixedRotationCleanupConfig, GlobalPostFixedRotationCleanupNotAttempted,
        GlobalPostFixedRotationCleanupRejection, GlobalPostFixedRotationCleanupReport,
        GlobalPostFixedRotationCleanupStatus, GraphConsistencyStats,
        global_post_cleanup_report_version, graph_candidate_is_acceptable, graph_consistency_stats,
    },
};

struct AppliedPostFixedRotationCleanup {
    rejected_under_supported_landmarks: usize,
    retriangulation: crate::ops::reconstruction::RetriangulationReport,
    rerun_preflight: BaPreflightReport,
    translation_and_points: crate::ops::reconstruction::BundleReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedDomainSignature {
    pub(super) pair_count: usize,
    pub(super) pair_digest: [u64; 2],
    pub(super) track_count: usize,
    pub(super) track_observation_count: usize,
    pub(super) track_digest: [u64; 2],
    pub(super) landmark_prefix_count: usize,
    pub(super) landmark_prefix_digest: [u64; 2],
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_post_fixed_rotation_cleanup(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    triangulator: &dyn Triangulator,
    max_reproj: f64,
    translation_config: &BundleAdjustConfig,
    graph_before_finish: &GraphConsistencyStats,
    graph_after_fixed_rotation: &GraphConsistencyStats,
    translation_and_points: &mut crate::ops::reconstruction::BundleReport,
    config: Option<&GlobalPostFixedRotationCleanupConfig>,
) -> GlobalPostFixedRotationCleanupReport {
    run_post_fixed_rotation_cleanup_inner(
        workspace,
        ba,
        triangulator,
        max_reproj,
        translation_config,
        graph_before_finish,
        graph_after_fixed_rotation,
        translation_and_points,
        config,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_post_fixed_rotation_cleanup_inner(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    triangulator: &dyn Triangulator,
    max_reproj: f64,
    translation_config: &BundleAdjustConfig,
    graph_before_finish: &GraphConsistencyStats,
    graph_after_fixed_rotation: &GraphConsistencyStats,
    translation_and_points: &mut crate::ops::reconstruction::BundleReport,
    config: Option<&GlobalPostFixedRotationCleanupConfig>,
    explicit_candidates: Option<&[ViewId]>,
) -> GlobalPostFixedRotationCleanupReport {
    let Some(config) = config.filter(|config| config.enabled()) else {
        return post_cleanup_not_attempted(
            GlobalPostFixedRotationCleanupNotAttempted::NotConfigured,
        );
    };
    if let Err(error) = config.validate() {
        return post_cleanup_rejected(
            Vec::new(),
            0,
            GlobalPostFixedRotationCleanupRejection::MutationFailed {
                error: error.to_string(),
            },
        );
    }

    let candidates = explicit_candidates.map_or_else(
        || {
            let before_trapped = graph_before_finish
                .trapped_views
                .iter()
                .map(|view| view.view)
                .collect::<BTreeSet<_>>();
            graph_after_fixed_rotation
                .trapped_views
                .iter()
                .map(|view| view.view)
                .filter(|image| !before_trapped.contains(image))
                .collect::<Vec<_>>()
        },
        <[ViewId]>::to_vec,
    );
    if candidates.is_empty() {
        return post_cleanup_not_attempted(
            GlobalPostFixedRotationCleanupNotAttempted::NoNewTrappedViews,
        );
    }

    let original_pose_count = workspace.registered_view_count();
    let allowed_views = allowed_post_cleanup_views(config, original_pose_count);
    if candidates.len() > allowed_views {
        return post_cleanup_rejected(
            candidates,
            allowed_views,
            GlobalPostFixedRotationCleanupRejection::CandidateLimitExceeded,
        );
    }

    if !workspace.components().first().is_some_and(|component| {
        matches!(
            component.gauge_scale_policy,
            crate::config::GaugeScalePolicy::Free
        )
    }) {
        return post_cleanup_rejected(
            candidates,
            allowed_views,
            GlobalPostFixedRotationCleanupRejection::GaugeNotFree,
        );
    }
    if let Err(reason) = certify_post_cleanup_topology(workspace, &candidates) {
        return post_cleanup_rejected(candidates, allowed_views, reason);
    }

    let prepared_domains = prepared_domain_signature(workspace);
    let transaction = GeometryRollback::capture(workspace);
    let attempt = (|| {
        workspace
            .clear_poses_for_views(&candidates)
            .map_err(
                |error| GlobalPostFixedRotationCleanupRejection::MutationFailed {
                    error: error.to_string(),
                },
            )?;
        let rejected_under_supported_landmarks =
            reject_under_supported_landmarks_on_deferred_tracks(workspace, &candidates).map_err(
                |error| GlobalPostFixedRotationCleanupRejection::MutationFailed {
                    error: error.to_string(),
                },
            )?;
        let retriangulation = crate::ops::reconstruction::refit_incremental_for_views(
            workspace,
            triangulator,
            2,
            max_reproj,
            &candidates,
        )
        .map_err(
            |error| GlobalPostFixedRotationCleanupRejection::MutationFailed {
                error: error.to_string(),
            },
        )?;
        crate::ops::reconstruction::refresh_primary_component(workspace);
        let retained_graph_before_rerun = graph_consistency_stats(workspace);

        let problem =
            crate::ops::reconstruction::bundle_adjustment_problem(workspace, translation_config)
                .map_err(
                    |error| GlobalPostFixedRotationCleanupRejection::MutationFailed {
                        error: error.to_string(),
                    },
                )?;
        if !matches!(problem.gauge(), openmvg_ba::Gauge::Free) {
            return Err(GlobalPostFixedRotationCleanupRejection::GaugeNotFree);
        }
        let rerun_preflight = ba
            .preflight(&problem, translation_config)
            .map_err(|failure| {
                GlobalPostFixedRotationCleanupRejection::FixedRotationPreflightFailed {
                    error: failure.to_string(),
                }
            })?;
        if !matches!(rerun_preflight.decision, BaPreflightDecision::Proceed) {
            return Err(
                GlobalPostFixedRotationCleanupRejection::FixedRotationPreflightRejected {
                    report: Box::new(rerun_preflight),
                },
            );
        }
        let rerun = workspace
            .bundle_adjust(ba, translation_config)
            .map_err(
                |error| GlobalPostFixedRotationCleanupRejection::FixedRotationRerunFailed {
                    error: error.to_string(),
                },
            )?;
        let retained_pose_count = original_pose_count - candidates.len();
        // The bounded preflight probe may conservatively omit a camera while requesting an
        // exact-direct escalation. Once that exact solve completes, its full camera count and
        // gauge-reduced rank are the authoritative coverage certificate.
        if rerun.adjusted_cameras != retained_pose_count
            || workspace.registered_view_count() != retained_pose_count
        {
            return Err(
                GlobalPostFixedRotationCleanupRejection::IncompleteCameraCoverage {
                    expected: retained_pose_count,
                    preflight_covered: rerun_preflight.covered_free_pose_count,
                    adjusted: rerun.adjusted_cameras,
                },
            );
        }
        if !post_cleanup_observability_is_certified(&rerun_preflight, &rerun, retained_pose_count) {
            return Err(GlobalPostFixedRotationCleanupRejection::ObservabilityNotCertified);
        }
        let retained_graph_after_rerun = graph_consistency_stats(workspace);
        if !graph_candidate_is_acceptable(&retained_graph_before_rerun, &retained_graph_after_rerun)
        {
            return Err(GlobalPostFixedRotationCleanupRejection::GraphRegression);
        }
        let baseline_trapped = graph_before_finish
            .trapped_views
            .iter()
            .map(|view| view.view)
            .collect::<BTreeSet<_>>();
        let unexpected_trapped = retained_graph_after_rerun
            .trapped_views
            .iter()
            .map(|view| view.view)
            .filter(|image| !baseline_trapped.contains(image))
            .collect::<Vec<_>>();
        if retained_graph_after_rerun.locally_trapped_cameras
            > graph_before_finish.locally_trapped_cameras
            || !unexpected_trapped.is_empty()
        {
            return Err(
                GlobalPostFixedRotationCleanupRejection::RetainedTrappedViews {
                    images: unexpected_trapped,
                },
            );
        }
        workspace.validate().map_err(|error| {
            GlobalPostFixedRotationCleanupRejection::InvalidWorkspace {
                error: error.to_string(),
            }
        })?;
        let updated_domains = prepared_domain_signature_with_landmark_prefix(
            workspace,
            prepared_domains.landmark_prefix_count,
        );
        if updated_domains.pair_count != prepared_domains.pair_count
            || updated_domains.pair_digest != prepared_domains.pair_digest
            || updated_domains.track_count != prepared_domains.track_count
            || updated_domains.track_observation_count != prepared_domains.track_observation_count
            || updated_domains.track_digest != prepared_domains.track_digest
        {
            return Err(GlobalPostFixedRotationCleanupRejection::PreparedDomainsChanged);
        }
        if updated_domains.landmark_prefix_count != prepared_domains.landmark_prefix_count
            || updated_domains.landmark_prefix_digest != prepared_domains.landmark_prefix_digest
        {
            return Err(GlobalPostFixedRotationCleanupRejection::LandmarkIdentityChanged);
        }
        Ok(AppliedPostFixedRotationCleanup {
            rejected_under_supported_landmarks,
            retriangulation,
            rerun_preflight,
            translation_and_points: rerun,
        })
    })();

    match attempt {
        Ok(applied) => {
            let rejected_candidates = candidates.clone();
            let report = GlobalPostFixedRotationCleanupReport {
                schema_version: global_post_cleanup_report_version(),
                outcome: GlobalPostFixedRotationCleanupStatus::Applied {
                    deferred_views: candidates,
                    allowed_views,
                    rejected_under_supported_landmark_count: applied
                        .rejected_under_supported_landmarks,
                    retriangulation: applied.retriangulation,
                    rerun_preflight: Box::new(applied.rerun_preflight),
                    original_translation_and_points: Box::new(translation_and_points.clone()),
                },
            };
            if let Err(error) = report.validate() {
                transaction.rollback(workspace);
                return post_cleanup_rejected(
                    rejected_candidates,
                    allowed_views,
                    GlobalPostFixedRotationCleanupRejection::InvalidWorkspace {
                        error: error.to_string(),
                    },
                );
            }
            *translation_and_points = applied.translation_and_points;
            drop(transaction);
            report
        }
        Err(reason) => {
            transaction.rollback(workspace);
            post_cleanup_rejected(candidates, allowed_views, reason)
        }
    }
}

pub(super) fn allowed_post_cleanup_views(
    config: &GlobalPostFixedRotationCleanupConfig,
    posed_views: usize,
) -> usize {
    bounded_deferred_view_count(posed_views, config.max_views(), config.max_fraction())
}

fn post_cleanup_not_attempted(
    reason: GlobalPostFixedRotationCleanupNotAttempted,
) -> GlobalPostFixedRotationCleanupReport {
    GlobalPostFixedRotationCleanupReport {
        schema_version: global_post_cleanup_report_version(),
        outcome: GlobalPostFixedRotationCleanupStatus::NotAttempted { reason },
    }
}

fn post_cleanup_rejected(
    candidates: Vec<ViewId>,
    allowed_views: usize,
    reason: GlobalPostFixedRotationCleanupRejection,
) -> GlobalPostFixedRotationCleanupReport {
    let report = GlobalPostFixedRotationCleanupReport {
        schema_version: global_post_cleanup_report_version(),
        outcome: GlobalPostFixedRotationCleanupStatus::Rejected {
            candidates,
            allowed_views,
            reason,
        },
    };
    debug_assert!(report.validate().is_ok());
    report
}

pub(super) fn certify_post_cleanup_topology(
    workspace: &SfmWorkspace,
    candidates: &[ViewId],
) -> std::result::Result<(), GlobalPostFixedRotationCleanupRejection> {
    let mut dense_by_view = vec![None; workspace.views().len()];
    let posed_views = workspace
        .registered_views()
        .enumerate()
        .map(|(dense, (view, _))| {
            dense_by_view[view.index()] = Some(dense);
            view
        })
        .collect::<Vec<_>>();
    let candidate_set = candidates.iter().copied().collect::<BTreeSet<_>>();
    let endpoints = posed_verified_endpoints(workspace, &dense_by_view);
    certify_post_cleanup_topology_from_graph(&posed_views, &endpoints, &candidate_set)
}

pub(super) fn certify_post_cleanup_topology_from_graph(
    posed_views: &[ViewId],
    endpoints: &[(usize, usize)],
    candidate_set: &BTreeSet<ViewId>,
) -> std::result::Result<(), GlobalPostFixedRotationCleanupRejection> {
    let topology = analyze_undirected_topology(posed_views.len(), endpoints).map_err(|error| {
        GlobalPostFixedRotationCleanupRejection::TopologyAnalysisFailed {
            error: error.to_string(),
        }
    })?;
    let articulation_images = topology
        .articulation_nodes()
        .iter()
        .map(|&dense| posed_views[dense])
        .filter(|image| candidate_set.contains(image))
        .collect::<Vec<_>>();
    if !articulation_images.is_empty() {
        return Err(
            GlobalPostFixedRotationCleanupRejection::ContainsArticulationViews {
                images: articulation_images,
            },
        );
    }

    let mut retained_dense_by_original = vec![None; posed_views.len()];
    let mut retained_views =
        Vec::with_capacity(posed_views.len().saturating_sub(candidate_set.len()));
    for (original_dense, image) in posed_views.iter().copied().enumerate() {
        if candidate_set.contains(&image) {
            continue;
        }
        retained_dense_by_original[original_dense] = Some(retained_views.len());
        retained_views.push(image);
    }
    let retained_endpoints = endpoints
        .iter()
        .filter_map(|&(first, second)| {
            Some((
                retained_dense_by_original.get(first).copied().flatten()?,
                retained_dense_by_original.get(second).copied().flatten()?,
            ))
        })
        .collect::<Vec<_>>();
    let retained_topology = analyze_undirected_topology(retained_views.len(), &retained_endpoints)
        .map_err(
            |error| GlobalPostFixedRotationCleanupRejection::TopologyAnalysisFailed {
                error: error.to_string(),
            },
        )?;
    if retained_views.len() < 2 || retained_topology.component_count() != 1 {
        return Err(
            GlobalPostFixedRotationCleanupRejection::RetainedGraphDisconnected {
                component_count: retained_topology.component_count(),
            },
        );
    }
    Ok(())
}

fn posed_verified_endpoints(
    workspace: &SfmWorkspace,
    dense_by_view: &[Option<usize>],
) -> Vec<(usize, usize)> {
    workspace
        .pair_graph()
        .active_pair_ids()
        .filter_map(|pair_id| {
            workspace
                .geometric_model_store()
                .get(pair_id.index())
                .and_then(|model| model.relative_pose())?;
            let pair = workspace.pair_graph().pair(pair_id)?;
            Some((
                dense_by_view.get(pair.first().index()).copied().flatten()?,
                dense_by_view
                    .get(pair.second().index())
                    .copied()
                    .flatten()?,
            ))
        })
        .collect()
}

pub(super) fn reject_under_supported_landmarks_on_deferred_tracks(
    workspace: &mut SfmWorkspace,
    deferred: &[ViewId],
) -> Result<usize> {
    let affected_tracks = deferred
        .iter()
        .flat_map(|view| workspace.tracks().observing_view(*view).iter().copied())
        .collect::<BTreeSet<_>>();
    let to_reject = affected_tracks
        .into_iter()
        .filter_map(|track_id| {
            let landmark_id = workspace.landmarks().for_track(track_id)?;
            let landmark = workspace.landmarks().get(landmark_id)?;
            if landmark.status != LandmarkStatus::Active {
                return None;
            }
            let track = workspace.tracks().get(track_id)?;
            let posed_support = landmark
                .supporting_observations(track)
                .filter(|(_, observation)| workspace.has_pose(observation.view))
                .count();
            (posed_support < 2).then_some(landmark_id)
        })
        .collect::<Vec<_>>();
    for landmark_id in &to_reject {
        let landmark = workspace
            .landmarks
            .get_mut(*landmark_id)
            .ok_or(SfmError::InvalidId {
                kind: crate::LandmarkId::kind(),
                index: landmark_id.index(),
            })?;
        landmark.status = LandmarkStatus::Rejected;
    }
    Ok(to_reject.len())
}

fn post_cleanup_observability_is_certified(
    preflight: &BaPreflightReport,
    rerun: &crate::ops::reconstruction::BundleReport,
    retained_pose_count: usize,
) -> bool {
    let expected_dimension = retained_pose_count.saturating_mul(3).saturating_sub(4);
    let diagnostics = &rerun.ba.diagnostics;
    let Some(observability) = diagnostics.observability.as_ref() else {
        return false;
    };
    // The rerun was committed by the same audit as the original fixed-rotation solve, so its
    // assessed rank is the certificate. On the gauge-reduced camera layout `FullRank` and
    // `OnlyExpectedGauge` both mean no null direction beyond the gauge; an unassessed rank never
    // reaches here as either. The dimensions additionally require it to cover every retained
    // camera.
    matches!(preflight.decision, BaPreflightDecision::Proceed)
        && preflight.pose_dimension == Some(expected_dimension)
        && preflight.weak_point_count == Some(0)
        && matches!(
            diagnostics.rank,
            BaRankAssessment::FullRank | BaRankAssessment::OnlyExpectedGauge
        )
        && diagnostics.weak_point_count == Some(0)
        && observability.pose_dimension == expected_dimension
        && observability.pose_rank == expected_dimension
        && observability.weak_point_count == Some(0)
}

pub(super) fn prepared_domain_signature(workspace: &SfmWorkspace) -> PreparedDomainSignature {
    prepared_domain_signature_with_landmark_prefix(workspace, workspace.landmarks().len())
}

pub(super) fn prepared_domain_signature_with_landmark_prefix(
    workspace: &SfmWorkspace,
    landmark_prefix_count: usize,
) -> PreparedDomainSignature {
    let mut pair_digest = domain_digest_seed();
    for (index, pair) in workspace.pair_graph().pairs().iter().enumerate() {
        domain_digest_mix(&mut pair_digest, index as u64);
        domain_digest_mix(&mut pair_digest, pair.first().index() as u64);
        domain_digest_mix(&mut pair_digest, pair.second().index() as u64);
        domain_digest_mix(
            &mut pair_digest,
            u64::from(workspace.pair_graph().is_active(crate::PairId::new(index))),
        );
        domain_digest_mix(
            &mut pair_digest,
            workspace
                .pair_graph()
                .confidence(crate::PairId::new(index))
                .unwrap_or(f64::NAN)
                .to_bits(),
        );
    }
    let mut track_digest = domain_digest_seed();
    let mut track_observation_count = 0usize;
    for track in workspace.tracks().as_slice() {
        domain_digest_mix(&mut track_digest, track.id().index() as u64);
        domain_digest_mix(&mut track_digest, track.observations().len() as u64);
        track_observation_count += track.observations().len();
        for observation in track.observations() {
            domain_digest_mix(&mut track_digest, observation.view.index() as u64);
            domain_digest_mix(&mut track_digest, observation.feature.index() as u64);
            if let Ok(pixel) = workspace.track_observation_pixel(observation) {
                domain_digest_mix(&mut track_digest, pixel.x.to_bits());
                domain_digest_mix(&mut track_digest, pixel.y.to_bits());
            }
        }
    }
    let mut landmark_prefix_digest = domain_digest_seed();
    for landmark in workspace
        .landmarks()
        .as_slice()
        .iter()
        .take(landmark_prefix_count)
    {
        domain_digest_mix(&mut landmark_prefix_digest, landmark.id.index() as u64);
        domain_digest_mix(
            &mut landmark_prefix_digest,
            landmark.source_track.index() as u64,
        );
    }
    PreparedDomainSignature {
        pair_count: workspace.pair_graph().len(),
        pair_digest,
        track_count: workspace.tracks().len(),
        track_observation_count,
        track_digest,
        landmark_prefix_count,
        landmark_prefix_digest,
    }
}

const fn domain_digest_seed() -> [u64; 2] {
    [0xcbf2_9ce4_8422_2325, 0x9e37_79b9_7f4a_7c15]
}

fn domain_digest_mix(state: &mut [u64; 2], value: u64) {
    state[0] = (state[0] ^ value).wrapping_mul(0x0000_0100_0000_01b3);
    state[1] = state[1]
        .wrapping_add(value ^ state[0].rotate_left(17))
        .rotate_left(27)
        .wrapping_mul(0x94d0_49bb_1331_11eb);
}
