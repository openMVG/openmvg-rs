// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retriangulation and deterministic track growth.

use super::triangulation::{
    PosedObservationPolicy, gather_posed, triangulate_posed_with_limit, validate_reprojection_limit,
};
use super::*;

/// Why the pipeline is revisiting tracked geometry.
///
/// The intent owns both the affected track domain and the admissible landmark transitions. This
/// prevents a caller from accidentally combining full-map refitting with rejected-landmark
/// recovery and immediately undoing a cleanup pass.
#[derive(Debug, Clone, Copy)]
pub(super) enum TrackUpdateIntent<'a> {
    /// New camera poses affect only tracks observing those images. Existing landmarks on those
    /// tracks are refit from their retained support plus observations in the new images, including
    /// rejected landmarks that may now have enough support to revive.
    NewPoses(&'a [ViewId]),
    /// Existing camera poses changed, but no observation is newly eligible. Refit touched
    /// landmarks from their retained support and retry only tracks that have no landmark yet.
    RefitTouched(&'a [ViewId]),
    /// New camera poses affect only tracks observing those images. Materialize missing tracks and
    /// retry rejected landmarks, but do not refit active geometry.
    GrowOrReviveForNewPoses(&'a [ViewId]),
    /// Refit every stored landmark from its retained support.
    RefitAll,
    /// Restore missing and rejected tracks without replacing optimized active landmarks.
    RestoreArchived,
    /// Recover only tracks that have never produced a landmark.
    RecoverMissing,
}

/// Report for a re-triangulation (track-continuation) pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RetriangulationReport {
    /// Existing landmarks passed to the triangulator in this pass.
    pub attempted_refits: usize,
    /// Existing landmark slots successfully re-fit against now-posed observations.
    pub refined_landmarks: usize,
    /// Previously rejected landmark slots restored to active status.
    pub revived_landmarks: usize,
    /// Active landmarks marked `Rejected` because their refit failed or exceeded
    /// `max_reprojection_error`.
    pub dropped_landmarks: usize,
    /// Expected reasons attempted refits did not produce accepted geometry.
    pub refit_rejections: TriangulationRejectionSummary,
    /// Source tracks without an active landmark considered by the grow step.
    pub attempted_new_tracks: usize,
    /// New landmarks appended from tracks that became triangulable this pass.
    pub new_landmarks: usize,
    /// Expected reasons considered new tracks did not produce landmarks.
    pub growth_rejections: TriangulationRejectionSummary,
    /// Total active landmarks after the pass.
    pub total_landmarks: usize,
}

/// Re-triangulate the reconstruction against the *current* pose set and grow it:
///
/// 1. **Re-fit** every affected landmark from its retained support plus observations in newly
///    posed cameras. As more cameras register, a landmark first triangulated from two views can
///    gain support without silently restoring observations rejected by an earlier robust pass.
///    Rejected landmarks can revive when new support makes them valid. A landmark whose refit
///    reprojection error exceeds `max_reprojection_error` is marked `Rejected`.
/// 2. **Grow** by triangulating tracks that still have no landmark but now have
///    `>= min_observations` posed views ([`triangulate_incremental`] semantics).
///
/// `max_reprojection_error` is in the triangulator's own residual units; pass
/// `f64::INFINITY` to disable the drop step. Landmark ids are preserved for existing
/// landmarks (BA problems reference them by id), and new ids continue the numbering.
///
/// `changed_view`: when `Some(view)` (the just-registered view), only tracks that
/// observe `img` are re-fit / grown — every other track's posed-observation set is
/// unchanged, so its result is identical. This turns the per-registration cost from
/// O(all landmarks) into O(landmarks touching the new image). Pass `None` to process
/// the whole map (e.g. after a batch of pose changes).
pub(crate) fn retriangulate_incremental(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    changed_view: Option<ViewId>,
) -> Result<RetriangulationReport> {
    match changed_view {
        Some(view) => retriangulate_incremental_inner(
            workspace,
            triangulator,
            min_observations,
            max_reprojection_error,
            TrackUpdateIntent::NewPoses(std::slice::from_ref(&view)),
        ),
        None => retriangulate_incremental_inner(
            workspace,
            triangulator,
            min_observations,
            max_reprojection_error,
            TrackUpdateIntent::RefitAll,
        ),
    }
}

pub(crate) fn retriangulate_incremental_for_views(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    changed_views: &[ViewId],
) -> Result<RetriangulationReport> {
    retriangulate_incremental_inner(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        TrackUpdateIntent::NewPoses(changed_views),
    )
}

/// Refit tracks affected by changed poses without extending existing landmark support.
///
/// This is the post-optimization counterpart to [`retriangulate_incremental_for_views`]: the
/// images are already part of the reconstruction, so observations rejected by the preceding
/// robust refit must remain excluded.
pub(crate) fn refit_incremental_for_views(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    changed_views: &[ViewId],
) -> Result<RetriangulationReport> {
    retriangulate_incremental_inner(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        TrackUpdateIntent::RefitTouched(changed_views),
    )
}

/// Recover tracks that have never produced a landmark while preserving existing geometry.
///
/// A cleanup rejection remains authoritative until new observations arrive or an explicit archive
/// restoration is requested, so this operation cannot immediately revive the outliers removed by
/// a preceding final cleanup.
pub(crate) fn recover_missing_tracks(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
) -> Result<RetriangulationReport> {
    retriangulate_incremental_inner(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        TrackUpdateIntent::RecoverMissing,
    )
}

fn retriangulate_incremental_inner(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    intent: TrackUpdateIntent<'_>,
) -> Result<RetriangulationReport> {
    validate_reprojection_limit(max_reprojection_error)?;
    let min_observations = min_observations.max(2);
    let touched_tracks = match intent {
        TrackUpdateIntent::NewPoses(images)
        | TrackUpdateIntent::RefitTouched(images)
        | TrackUpdateIntent::GrowOrReviveForNewPoses(images) => {
            Some(tracks_observing_views(workspace, images))
        }
        TrackUpdateIntent::RefitAll
        | TrackUpdateIntent::RestoreArchived
        | TrackUpdateIntent::RecoverMissing => None,
    };
    let mut refined_landmarks = 0usize;
    let mut revived_landmarks = 0usize;
    let mut dropped_landmarks = 0usize;
    let mut refit_rejections = TriangulationRejectionSummary::default();

    // 1. Re-fit existing landmarks IN PLACE (no clone, no re-sort — ids are stable and
    //    the vec stays sorted). We collect refit outcomes first (immutable borrow to
    //    gather), then apply them, to satisfy the borrow checker without cloning tracks.
    //    Each outcome is a pure function of its own landmark + read-only `workspace`,
    //    so the gather is parallelized (see `refit_one`); `?` semantics are preserved
    //    via `collect::<Result<_>>`.
    let mut refit_candidates = Vec::new();
    match intent {
        TrackUpdateIntent::NewPoses(_) | TrackUpdateIntent::RefitTouched(_) => {
            let tracks = touched_tracks.as_deref().ok_or_else(|| {
                SfmError::InvariantViolation(
                    "scoped refit intent is missing its touched-track domain".to_owned(),
                )
            })?;
            for &track_id in tracks {
                let Some(landmark_id) = workspace.landmarks().for_track(track_id) else {
                    continue;
                };
                let landmark =
                    workspace
                        .landmarks()
                        .get(landmark_id)
                        .ok_or(SfmError::InvalidId {
                            kind: LandmarkId::kind(),
                            index: landmark_id.index(),
                        })?;
                let track = workspace
                    .tracks()
                    .get(track_id)
                    .ok_or(SfmError::InvalidId {
                        kind: TrackId::kind(),
                        index: track_id.index(),
                    })?;
                refit_candidates.push((landmark, track));
            }
        }
        TrackUpdateIntent::RefitAll => {
            for landmark in workspace.landmarks().as_slice() {
                let track =
                    workspace
                        .tracks()
                        .get(landmark.source_track)
                        .ok_or(SfmError::InvalidId {
                            kind: TrackId::kind(),
                            index: landmark.source_track.index(),
                        })?;
                refit_candidates.push((landmark, track));
            }
        }
        TrackUpdateIntent::GrowOrReviveForNewPoses(_)
        | TrackUpdateIntent::RestoreArchived
        | TrackUpdateIntent::RecoverMissing => {}
    }

    let refit_one = |candidate: &(&Landmark, &crate::Track)| -> Result<Option<RefitOutcome>> {
        let (landmark, track) = *candidate;
        let policy = match intent {
            TrackUpdateIntent::NewPoses(images) => {
                PosedObservationPolicy::CurrentSupportPlusImages {
                    support: &landmark.observation_support,
                    images,
                }
            }
            TrackUpdateIntent::RefitTouched(_) => {
                PosedObservationPolicy::CurrentSupport(&landmark.observation_support)
            }
            TrackUpdateIntent::RefitAll => {
                PosedObservationPolicy::CurrentSupport(&landmark.observation_support)
            }
            TrackUpdateIntent::GrowOrReviveForNewPoses(_)
            | TrackUpdateIntent::RestoreArchived
            | TrackUpdateIntent::RecoverMissing => {
                return Err(SfmError::InvariantViolation(
                    "non-refitting intent produced a refit candidate".to_owned(),
                ));
            }
        };
        let outcome = match gather_posed(workspace, track, policy) {
            Ok(Some(posed)) if posed.len() >= min_observations => triangulate_posed_with_limit(
                triangulator,
                &posed,
                track.observations().len(),
                max_reprojection_error,
            )?,
            // fewer than min_observations posed views now (or none): nothing to do.
            Ok(_) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some((landmark.id, outcome)))
    };
    let mut refit_results = map_ordered(&refit_candidates, refit_one);
    if let Some(error) = take_first_error(&mut refit_results) {
        return Err(error);
    }
    // Evaluate growth before applying refits so a fatal growth error cannot leave
    // half of the retriangulation pass committed.
    let evaluated_growth = evaluate_new_tracks(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        intent,
        touched_tracks.as_deref(),
    )?;

    let mut attempted_refits = 0;
    for result in refit_results {
        let Some((id, outcome)) = result? else {
            continue;
        };
        attempted_refits += 1;
        let landmark = workspace
            .landmarks
            .get_mut(id)
            .ok_or(SfmError::InvariantViolation(format!(
                "landmark {id} disappeared before refit commit"
            )))?;
        match outcome {
            PosedTriangulationOutcome::Accepted(accepted) => {
                if landmark.status == LandmarkStatus::Rejected {
                    revived_landmarks += 1;
                }
                landmark.point = accepted.point;
                landmark.reprojection_error = accepted.reprojection_error;
                landmark.observation_support = accepted.support;
                landmark.status = LandmarkStatus::Active; // revive if it had been Rejected
                refined_landmarks += 1;
            }
            PosedTriangulationOutcome::Rejected(reason) => {
                refit_rejections.record(reason);
                // Only a genuine Active→Rejected transition is a "drop"; an already-
                // Rejected landmark that still can't fit stays put.
                if landmark.status == LandmarkStatus::Active {
                    landmark.status = LandmarkStatus::Rejected;
                    dropped_landmarks += 1;
                }
            }
        }
    }
    // 2. Commit tracks that had no landmark and became triangulable.
    let growth = commit_growth(workspace, evaluated_growth)?;
    revived_landmarks += growth.revived_landmarks;
    let added_poses = matches!(intent, TrackUpdateIntent::NewPoses(images) if !images.is_empty());
    if added_poses || dropped_landmarks > 0 || revived_landmarks > 0 || growth.new_landmarks > 0 {
        refresh_primary_component(workspace);
    }
    let total_landmarks = workspace.landmarks().active_count();

    Ok(RetriangulationReport {
        attempted_refits,
        refined_landmarks,
        revived_landmarks,
        dropped_landmarks,
        refit_rejections,
        attempted_new_tracks: growth.attempted_tracks,
        new_landmarks: growth.new_landmarks,
        growth_rejections: growth.rejections,
        total_landmarks,
    })
}

/// Shared grow step for [`triangulate_incremental`] and [`retriangulate_incremental`].
/// Scoped jobs are selected serially, estimated in deterministic parallel order, and
/// committed only after every job has returned without a fatal error.
pub(super) struct GrowTracksReport {
    pub(super) attempted_tracks: usize,
    pub(super) new_landmarks: usize,
    pub(super) revived_landmarks: usize,
    pub(super) rejections: TriangulationRejectionSummary,
}

type GrowthOutcome = (TrackId, Option<LandmarkId>, PosedTriangulationOutcome);

// Construction verifies that every entry is `Ok`; retaining the Result wrapper
// avoids copying a potentially large outcome vector before the commit phase.
pub(super) struct EvaluatedGrowth {
    outcomes: Vec<Result<GrowthOutcome>>,
}

pub(super) fn evaluate_new_tracks(
    workspace: &SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    intent: TrackUpdateIntent<'_>,
    preselected_tracks: Option<&[TrackId]>,
) -> Result<EvaluatedGrowth> {
    // Scoped callers can share one already-computed touched-track domain; `None` retains the
    // full-map selection used by archive restoration and missing-track recovery.
    let mut candidates = Vec::new();
    let mut consider_track = |track_id: TrackId| -> Result<()> {
        let existing = workspace.landmarks().for_track(track_id);
        if let Some(landmark_id) = existing {
            let landmark = workspace
                .landmarks()
                .as_slice()
                .get(landmark_id.index())
                .ok_or(SfmError::InvalidId {
                    kind: LandmarkId::kind(),
                    index: landmark_id.index(),
                })?;
            let retry_rejected = matches!(
                intent,
                TrackUpdateIntent::GrowOrReviveForNewPoses(_) | TrackUpdateIntent::RestoreArchived
            );
            if !retry_rejected || landmark.status == LandmarkStatus::Active {
                return Ok(());
            }
        }
        candidates.push((track_id, existing));
        Ok(())
    };
    if let Some(tracks) = preselected_tracks {
        for &track_id in tracks {
            consider_track(track_id)?;
        }
    } else {
        match intent {
            TrackUpdateIntent::NewPoses(images)
            | TrackUpdateIntent::RefitTouched(images)
            | TrackUpdateIntent::GrowOrReviveForNewPoses(images) => {
                for track_id in tracks_observing_views(workspace, images) {
                    consider_track(track_id)?;
                }
            }
            TrackUpdateIntent::RefitAll
            | TrackUpdateIntent::RestoreArchived
            | TrackUpdateIntent::RecoverMissing => {
                for track in workspace.tracks().as_slice() {
                    consider_track(track.id())?;
                }
            }
        }
    }

    let grow_one = |candidate: &(TrackId, Option<LandmarkId>)| {
        let (track_id, existing) = *candidate;
        let track = workspace
            .tracks()
            .get(track_id)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: track_id.index(),
            })?;
        let policy = match (existing, intent) {
            (Some(landmark_id), TrackUpdateIntent::GrowOrReviveForNewPoses(images)) => {
                let landmark =
                    workspace
                        .landmarks()
                        .get(landmark_id)
                        .ok_or(SfmError::InvalidId {
                            kind: LandmarkId::kind(),
                            index: landmark_id.index(),
                        })?;
                PosedObservationPolicy::CurrentSupportPlusImages {
                    support: &landmark.observation_support,
                    images,
                }
            }
            _ => PosedObservationPolicy::All,
        };
        let posed = match gather_posed(workspace, track, policy) {
            Ok(Some(posed)) => posed,
            Ok(None) => {
                return Ok((
                    track.id(),
                    existing,
                    PosedTriangulationOutcome::Rejected(
                        TriangulationRejectReason::InsufficientPosedObservations,
                    ),
                ));
            }
            Err(error) => return Err(error),
        };
        if posed.len() < min_observations {
            return Ok((
                track.id(),
                existing,
                PosedTriangulationOutcome::Rejected(
                    TriangulationRejectReason::InsufficientPosedObservations,
                ),
            ));
        }
        let outcome = triangulate_posed_with_limit(
            triangulator,
            &posed,
            track.observations().len(),
            max_reprojection_error,
        )?;
        Ok((track.id(), existing, outcome))
    };
    let mut outcomes = map_ordered(&candidates, grow_one);
    if let Some(error) = take_first_error(&mut outcomes) {
        return Err(error);
    }

    Ok(EvaluatedGrowth { outcomes })
}

pub(super) fn tracks_observing_views(workspace: &SfmWorkspace, views: &[ViewId]) -> Vec<TrackId> {
    if let [view] = views {
        // The immutable track index already guarantees one stable, unique entry per source track.
        return workspace.tracks().observing_view(*view).to_vec();
    }
    let mut tracks = Vec::new();
    for &view in views {
        tracks.extend_from_slice(workspace.tracks().observing_view(view));
    }
    tracks.sort_unstable();
    tracks.dedup();
    tracks
}

pub(super) fn commit_growth(
    workspace: &mut SfmWorkspace,
    evaluated: EvaluatedGrowth,
) -> Result<GrowTracksReport> {
    let outcomes = evaluated.outcomes;
    let attempted_tracks = outcomes.len();
    let first_new_landmark = workspace.landmarks().len();
    let mut existing_updates = Vec::new();
    let mut appended_landmarks = Vec::new();
    let mut revived_landmarks = 0;
    let mut rejections = TriangulationRejectionSummary::default();
    for outcome in outcomes {
        let (track_id, existing, outcome) = outcome?;
        match outcome {
            PosedTriangulationOutcome::Accepted(accepted) => {
                if let Some(id) = existing {
                    let landmark =
                        workspace
                            .landmarks
                            .get(id)
                            .ok_or(SfmError::InvariantViolation(format!(
                                "growth landmark {id} disappeared before commit"
                            )))?;
                    if landmark.status == LandmarkStatus::Rejected {
                        revived_landmarks += 1;
                    }
                    existing_updates.push((id, accepted));
                } else {
                    let id = LandmarkId::new(first_new_landmark + appended_landmarks.len());
                    appended_landmarks.push(Landmark::active(
                        id,
                        track_id,
                        accepted.point,
                        accepted.reprojection_error,
                        accepted.support,
                    ));
                }
            }
            PosedTriangulationOutcome::Rejected(reason) => rejections.record(reason),
        }
    }
    let new_landmarks = appended_landmarks.len();
    workspace.landmarks.append_batch(appended_landmarks)?;
    for (id, accepted) in existing_updates {
        let landmark = workspace
            .landmarks
            .get_mut(id)
            .ok_or(SfmError::InvariantViolation(format!(
                "growth landmark {id} disappeared during commit"
            )))?;
        landmark.point = accepted.point;
        landmark.reprojection_error = accepted.reprojection_error;
        landmark.observation_support = accepted.support;
        landmark.status = LandmarkStatus::Active;
    }
    Ok(GrowTracksReport {
        attempted_tracks,
        new_landmarks,
        revived_landmarks,
        rejections,
    })
}
