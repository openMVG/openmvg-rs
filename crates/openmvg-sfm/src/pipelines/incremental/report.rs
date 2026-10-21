// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reports of the Incremental engine: growth, final refinement, and unregistered-view
//! diagnostics.

use std::collections::{BTreeMap, BTreeSet};

use crate::ops::reconstruction::TriangulationRejectionSummary;
use crate::pipelines::incremental::RegistrationFailureReason;
use crate::{SfmWorkspace, ViewId};

#[cfg(feature = "unstable-test-support")]
use super::next_view::RegistrationCandidate;
use super::scene_initializer::IncrementalInitializationReport;
#[cfg(feature = "unstable-test-support")]
use crate::evaluation::EvaluationReport;

/// Growth-phase telemetry of one Incremental SfM run: registrations, track updates, and growth
/// bundle adjustments.
///
/// [`IncrementalSfmReport`] flattens it, so its fields serialize at the top level of that report.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncrementalGrowthReport {
    /// Views registered by this growth pass.
    pub grown_view_count: usize,
    /// Successful registrations in acceptance order.
    #[serde(default)]
    pub growth_registrations: Vec<GrowthRegistrationDiagnostic>,
    /// Registration-candidate evaluations performed before termination.
    pub iteration_count: usize,
    /// Frontier scans, including terminal scans that found no candidate.
    #[serde(default)]
    pub frontier_selection_count: usize,
    /// Sum of candidates eligible before the relative support-band policy.
    #[serde(default)]
    pub frontier_eligible_candidate_evaluation_count: usize,
    /// Sum of candidates considered after the relative support-band policy.
    #[serde(default)]
    pub frontier_support_band_candidate_evaluation_count: usize,
    /// Aggregate initial-pair, growth, and refit rejection attempts supplied by the caller.
    pub triangulation_rejections: TriangulationRejectionSummary,
    /// Growth-stage global BA proposals retained by their transaction.
    #[serde(default)]
    pub periodic_ba_count: usize,
    /// Every growth-stage global BA execution, including rejected attempts.
    #[serde(default)]
    pub periodic_ba_attempt_count: usize,
    /// Growth-stage global BA proposals rejected without aborting reconstruction.
    #[serde(default)]
    pub periodic_ba_rejected_count: usize,
    /// Post-BA support audits that removed at least one camera outside the initial pair during
    /// growth.
    #[serde(default)]
    pub post_ba_pose_support_cleanups: Vec<crate::ops::reconstruction::PoseSupportCleanupReport>,
    /// Aggregate optimizer iterations spent in growth-stage global BA.
    #[serde(default)]
    pub periodic_ba_iteration_count: usize,
    /// Largest compact rollback payload retained beside a periodic BA attempt.
    #[serde(default)]
    pub periodic_ba_snapshot_payload_peak_bytes: u128,
    /// Adaptive global BA attempts caused by camera or landmark growth.
    #[serde(default)]
    pub adaptive_ba_attempt_count: usize,
    /// Adaptive global BA proposals that were committed.
    #[serde(default)]
    pub adaptive_ba_applied_count: usize,
    /// Adaptive global BA proposals rejected without aborting reconstruction.
    #[serde(default)]
    pub adaptive_ba_rejected_count: usize,
}

/// Diagnostics of one Incremental SfM run.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncrementalSfmReport {
    /// Initial-pair selection and initial two-view structure.
    pub initialization: IncrementalInitializationReport,
    /// Views holding a pose at the end of the run.
    pub registered_view_count: usize,
    /// Active landmarks at the end of the run.
    pub landmark_count: usize,
    /// Why incremental growth stopped.
    pub termination: IncrementalTermination,
    /// Registration and growth-stage BA telemetry, serialized inline.
    #[serde(flatten)]
    pub growth: IncrementalGrowthReport,
    /// Landmarks rejected by the final BA/clean loop.
    #[serde(default)]
    pub final_cleaned_landmark_count: usize,
    /// Final BA/clean proposals rejected transactionally while retaining the valid
    /// growth-stage reconstruction.
    #[serde(default)]
    pub final_ba_rejected_count: usize,
    /// Missing-track recovery after the final robust clean. Existing active and rejected
    /// landmarks are preserved. `None` when no BA backend is supplied or recovery is disabled.
    #[serde(default)]
    pub final_retriangulation: Option<crate::ops::reconstruction::RetriangulationReport>,
    /// Typed receipt for the final robust clean and optional missing-track recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_refinement: Option<Box<IncrementalFinalRefinementReport>>,
    /// Every rejected/deferred registration receipt, including frontier retries and cameras
    /// removed by the post-BA support audit.
    #[serde(default)]
    pub registration_failure_attempt_count: usize,
    /// Last failure reason for each image still unregistered at termination.
    #[serde(default)]
    pub unregistered_failure_counts: BTreeMap<RegistrationFailureReason, usize>,
    /// Unregistered images that never reached the registration-candidate frontier.
    #[serde(default)]
    pub unregistered_without_attempt_count: usize,
    /// Final frontier state for every view left unregistered. This makes
    /// graph disconnection, missing 2D-3D intersections, and geometric rejection
    /// distinguishable without enabling temporary debug logging.
    #[serde(default)]
    pub unregistered_views: Vec<UnregisteredViewDiagnostic>,
}

/// Why an Incremental SfM run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncrementalTermination {
    /// Every view in the workspace is registered.
    Complete,
    /// Some unregistered views never had a registration attempt: the frontier ran dry.
    FrontierExhausted,
    /// Every remaining unregistered view was attempted and rejected or deferred.
    AllCandidatesRejected,
    /// The candidate-evaluation budget (`max_iterations`) was exhausted first.
    MaxIterations,
}

impl IncrementalTermination {
    pub(super) fn classify(
        registered_view_count: usize,
        total_view_count: usize,
        hit_max_iterations: bool,
        unregistered_without_attempt_count: usize,
    ) -> Self {
        if registered_view_count == total_view_count {
            Self::Complete
        } else if hit_max_iterations {
            Self::MaxIterations
        } else if unregistered_without_attempt_count > 0 {
            Self::FrontierExhausted
        } else {
            Self::AllCandidatesRejected
        }
    }
}

/// Whether the final incremental refinement was kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncrementalFinalRefinementStatus {
    /// The refinement succeeded without regressing reconstruction quality and was committed.
    Applied,
    /// The refinement failed recoverably or regressed quality; the workspace was restored to
    /// its state before refinement.
    RejectedRolledBack,
}

/// Final incremental polish diagnostics retained instead of discarding its BA reports.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncrementalFinalRefinementReport {
    /// Layout version of this report's serialized form (currently 1).
    pub schema_version: u32,
    /// Whether the refinement was committed or rolled back.
    pub outcome: IncrementalFinalRefinementStatus,
    /// Intrinsics self-calibration BA run first; `None` when self-calibration is disabled or
    /// the refinement failed before it completed.
    pub self_calibration: Option<crate::ops::reconstruction::BundleReport>,
    /// Robust adjust-and-clean phase preceding missing-track recovery.
    pub before_retriangulation: Option<crate::ops::reconstruction::BundleAdjustAndCleanReport>,
    /// Fixed-point support audits applied after full-scene refinement.
    #[serde(default)]
    pub pose_support_cleanups: Vec<crate::ops::reconstruction::PoseSupportCleanupReport>,
    /// Additional robust polish passes required only when an audit removed cameras.
    #[serde(default)]
    pub post_pose_cleanup_adjustments: Vec<crate::ops::reconstruction::BundleAdjustAndCleanReport>,
}

/// Evidence available for a view the Incremental engine left unregistered.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct UnregisteredViewDiagnostic {
    /// The unregistered view.
    pub view: ViewId,
    /// Whether the view is connected to the initial-pair views through active verified pairs.
    pub in_initial_pair_component: bool,
    /// Active verified pairs incident to the view.
    pub verified_neighbor_count: usize,
    /// Those verified neighbors that are registered.
    pub verified_registered_neighbor_count: usize,
    /// Tracks observing the view.
    pub observed_track_count: usize,
    /// Observing tracks that back an active landmark (usable 2D-3D matches).
    pub visible_landmark_count: usize,
    /// Observing tracks with at least one registered observer.
    pub tracks_with_posed_observer_count: usize,
    /// Observing tracks with at least two registered observers (triangulatable).
    pub tracks_with_two_posed_observers_count: usize,
    /// Last recorded registration failure; `None` if the view was never attempted.
    pub last_failure: Option<RegistrationFailureReason>,
}

/// One successful growth registration and the candidate evidence that ranked it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GrowthRegistrationDiagnostic {
    /// The registered view.
    pub view: ViewId,
    /// Estimator name reported by the pose registrar.
    #[serde(default)]
    pub method: String,
    /// Active landmarks the view observed when it was selected.
    pub visible_landmark_count: usize,
    /// Occupied cells in the 4x4 grid used to rank this candidate before P3P.
    #[serde(default)]
    pub visible_image_cell_count: usize,
    /// Active verified pairs to registered views when the view was selected.
    pub verified_registered_neighbor_count: usize,
    /// Tracks observing the view.
    pub track_observation_count: usize,
    /// Registration failures recorded for the view before this attempt.
    pub previous_failure_count: usize,
    /// 2D-3D observations passed to the registrar.
    pub attempted_observation_count: usize,
    /// Registrar inliers among `attempted_observations`.
    pub inlier_count: usize,
    /// Median unsquared angular bearing residual, in radians.
    pub residual_median: openmvg_core::AngularResidual,
    /// Median disagreement in degrees between the accepted rotation and the relative-rotation
    /// edges to posed neighbors; `None` without such an edge.
    pub rotation_edge_error_deg: Option<f64>,
    /// Median disagreement in degrees between the accepted camera center and the
    /// relative-translation edge directions to posed neighbors; `None` without such an edge.
    pub translation_edge_error_deg: Option<f64>,
    /// Fitted map support after a scoped refit. Present only when that registration actually
    /// refit existing landmarks and retained the active frontier's support floor. Captured before
    /// any optional periodic BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_ba_post_refit_support: Option<PostRegistrationSupportDiagnostic>,
    /// The growth-stage BA applied after this registration was triggered only by the adaptive
    /// growth-ratio policy.
    #[serde(default)]
    pub adaptive_ba_after: bool,
    /// A growth-stage global BA (periodic or adaptive) was applied after this registration.
    pub periodic_ba_after: bool,
}

/// Landmark support of a newly registered view after its scoped refit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PostRegistrationSupportDiagnostic {
    /// Active landmarks whose immutable source tracks observe the registered image.
    pub active_landmark_count: usize,
    /// Those landmarks whose current fitted support still includes the registered observation.
    pub supporting_landmark_count: usize,
}

impl IncrementalSfmReport {
    /// Assemble the report from the final workspace state and the initial-pair report.
    /// The registered, landmark, and grown counts are all derived here so the default and
    /// oracle entry points report them consistently.
    pub(super) fn from_workspace(
        workspace: &SfmWorkspace,
        initialization: IncrementalInitializationReport,
        growth: IncrementalGrowthReport,
        final_cleaned_landmark_count: usize,
        hit_max_iterations: bool,
    ) -> Self {
        let registered_view_count = workspace.registered_view_count();
        let mut last_failure = BTreeMap::new();
        for failure in workspace.registration_failures() {
            if !workspace.has_pose(failure.view) {
                last_failure.insert(failure.view, failure.reason);
            }
        }
        let mut unregistered_failure_counts = BTreeMap::new();
        for reason in last_failure.values() {
            *unregistered_failure_counts.entry(*reason).or_default() += 1;
        }
        let unregistered_without_attempt_count = workspace
            .views()
            .len()
            .saturating_sub(registered_view_count)
            .saturating_sub(last_failure.len());
        let unregistered_views =
            collect_unregistered_image_diagnostics(workspace, &initialization, &last_failure);
        let termination = IncrementalTermination::classify(
            registered_view_count,
            workspace.views().len(),
            hit_max_iterations,
            unregistered_without_attempt_count,
        );
        Self {
            registered_view_count,
            landmark_count: workspace.landmarks().active_count(),
            termination,
            growth,
            final_cleaned_landmark_count,
            final_ba_rejected_count: 0,
            final_retriangulation: None,
            final_refinement: None,
            registration_failure_attempt_count: workspace.registration_failures().len(),
            unregistered_failure_counts,
            unregistered_without_attempt_count,
            unregistered_views,
            initialization,
        }
    }
}

fn collect_unregistered_image_diagnostics(
    workspace: &SfmWorkspace,
    initialization: &IncrementalInitializationReport,
    last_failure: &BTreeMap<ViewId, RegistrationFailureReason>,
) -> Vec<UnregisteredViewDiagnostic> {
    collect_unregistered_image_diagnostics_from_roots(
        workspace,
        &initialization.seed.views,
        last_failure,
    )
}

fn collect_unregistered_image_diagnostics_from_roots(
    workspace: &SfmWorkspace,
    component_roots: &[ViewId],
    last_failure: &BTreeMap<ViewId, RegistrationFailureReason>,
) -> Vec<UnregisteredViewDiagnostic> {
    let view_count = workspace.views().len();
    let mut observed_track_count = vec![0usize; view_count];
    let mut visible_landmark_count = vec![0usize; view_count];
    let mut tracks_with_posed_observer_count = vec![0usize; view_count];
    let mut tracks_with_two_posed_observers_count = vec![0usize; view_count];
    let active_tracks: BTreeSet<_> = workspace
        .landmarks()
        .iter_active()
        .map(|landmark| landmark.source_track)
        .collect();

    for track in workspace.tracks().as_slice() {
        let posed_observers = track
            .observations()
            .iter()
            .filter(|observation| workspace.has_pose(observation.view))
            .count();
        let is_active = active_tracks.contains(&track.id());
        for observation in track.observations() {
            let image_index = observation.view.index();
            observed_track_count[image_index] += 1;
            if workspace.has_pose(observation.view) {
                continue;
            }
            visible_landmark_count[image_index] += usize::from(is_active);
            tracks_with_posed_observer_count[image_index] += usize::from(posed_observers >= 1);
            tracks_with_two_posed_observers_count[image_index] += usize::from(posed_observers >= 2);
        }
    }

    let mut initial_pair_component = BTreeSet::new();
    let mut stack = component_roots.to_vec();
    while let Some(image) = stack.pop() {
        if !initial_pair_component.insert(image) {
            continue;
        }
        if let Ok(neighbors) = workspace.pair_graph().neighbors(image) {
            stack.extend(neighbors.iter().filter_map(|neighbor| {
                workspace
                    .pair_graph()
                    .is_active(neighbor.pair_id)
                    .then_some(neighbor.view)
            }));
        }
    }

    workspace
        .views()
        .iter()
        .filter(|image| !workspace.has_pose(image.id))
        .map(|image| {
            let (verified_neighbor_count, verified_registered_neighbor_count) = workspace
                .pair_graph()
                .neighbors(image.id)
                .map(|neighbors| {
                    neighbors
                        .iter()
                        .filter(|neighbor| workspace.pair_graph().is_active(neighbor.pair_id))
                        .fold((0usize, 0usize), |(verified, registered), neighbor| {
                            (
                                verified + 1,
                                registered + usize::from(workspace.has_pose(neighbor.view)),
                            )
                        })
                })
                .unwrap_or_default();
            let index = image.id.index();
            UnregisteredViewDiagnostic {
                view: image.id,
                in_initial_pair_component: initial_pair_component.contains(&image.id),
                verified_neighbor_count,
                verified_registered_neighbor_count,
                observed_track_count: observed_track_count[index],
                visible_landmark_count: visible_landmark_count[index],
                tracks_with_posed_observer_count: tracks_with_posed_observer_count[index],
                tracks_with_two_posed_observers_count: tracks_with_two_posed_observers_count[index],
                last_failure: last_failure.get(&image.id).copied(),
            }
        })
        .collect()
}

/// Incremental run initialized from ground truth, with its evaluation against that ground truth.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg(feature = "unstable-test-support")]
pub struct IncrementalOracleSfmReport {
    /// The Incremental reconstruction diagnostics.
    pub reconstruction: IncrementalSfmReport,
    /// Candidates still reachable after the default growth tail terminates.
    /// A complete oracle reconstruction leaves this empty.
    pub growth_candidates: Vec<RegistrationCandidate>,
    /// Pose and trajectory metrics against the ground truth, without alignment (the ground-truth
    /// initial pair fixes the frame).
    pub evaluation: EvaluationReport,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_termination_distinguishes_frontier_exhaustion_from_rejection() {
        assert_eq!(
            IncrementalTermination::classify(3, 3, true, 0),
            IncrementalTermination::Complete
        );
        assert_eq!(
            IncrementalTermination::classify(2, 3, true, 1),
            IncrementalTermination::MaxIterations
        );
        assert_eq!(
            IncrementalTermination::classify(2, 3, false, 1),
            IncrementalTermination::FrontierExhausted
        );
        assert_eq!(
            IncrementalTermination::classify(2, 3, false, 0),
            IncrementalTermination::AllCandidatesRejected
        );
    }
}
