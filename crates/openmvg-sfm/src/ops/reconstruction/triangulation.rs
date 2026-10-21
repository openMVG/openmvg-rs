// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Initial and incremental landmark triangulation.

use super::*;
use super::{
    residual::{CameraResidualPolicy, is_generalized_camera},
    track_growth::{TrackUpdateIntent, commit_growth, evaluate_new_tracks, tracks_observing_views},
};
/// Triangulate every track that has at least two posed observations, replacing all landmarks.
///
/// Rebuilds component 0 from the registered views and the new landmarks with `gauge_scale_policy`.
/// Fails with `SfmError::MissingPrerequisite` when no tracks have been built.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn triangulate(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    gauge_scale_policy: GaugeScalePolicy,
) -> Result<TriangulationReport> {
    triangulate_with_min_observations(workspace, triangulator, gauge_scale_policy, 2)
}

/// Global triangulation with an explicit minimum number of posed observations per track.
///
/// The global-SfM bootstrap uses three-view tracks: two-view tracks are useful at
/// an incremental frontier, but are a weak scale/depth constraint for the first global BA.
pub(crate) fn triangulate_with_min_observations(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    gauge_scale_policy: GaugeScalePolicy,
    min_observations: usize,
) -> Result<TriangulationReport> {
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "tracks",
            method: "build_tracks",
        });
    }

    let min_observations = min_observations.max(2);
    let triangulate_one = |track: &crate::Track| {
        // Global averaging may leave some views unposed, so use the posed subset.
        let Some(posed) = gather_posed(workspace, track, PosedObservationPolicy::All)? else {
            return Ok((
                track.id(),
                PosedTriangulationOutcome::Rejected(
                    TriangulationRejectReason::InsufficientPosedObservations,
                ),
            ));
        };
        if posed.len() < min_observations {
            return Ok((
                track.id(),
                PosedTriangulationOutcome::Rejected(
                    TriangulationRejectReason::InsufficientPosedObservations,
                ),
            ));
        }

        Ok((
            track.id(),
            triangulate_posed(triangulator, &posed, track.observations().len())?,
        ))
    };

    // Track estimates are independent. Ordered collection plus sequential commit
    // preserves dense landmark IDs across worker counts; small batches stay serial.
    let mut outcomes = map_ordered(workspace.tracks().as_slice(), triangulate_one);
    if let Some(error) = take_first_error(&mut outcomes) {
        return Err(error);
    }

    let mut landmarks = Vec::new();
    let mut rejections = TriangulationRejectionSummary::default();
    for outcome in outcomes {
        let (track_id, outcome) = outcome?;
        let accepted = match outcome {
            PosedTriangulationOutcome::Accepted(accepted) => accepted,
            PosedTriangulationOutcome::Rejected(reason) => {
                rejections.record(reason);
                continue;
            }
        };
        let landmark_id = LandmarkId::new(landmarks.len());
        landmarks.push(Landmark::active(
            landmark_id,
            track_id,
            accepted.point,
            accepted.reprojection_error,
            accepted.support,
        ));
    }

    let triangulated_landmarks = landmarks.len();
    workspace.landmarks = Landmarks::try_new(landmarks)?;
    workspace.components = vec![ReconstructionComponent::new(
        ComponentId::new(0),
        workspace.registered_views().map(|(view, _)| view).collect(),
        workspace
            .landmarks()
            .as_slice()
            .iter()
            .map(|landmark| landmark.id)
            .collect(),
        gauge_scale_policy,
    )];

    Ok(TriangulationReport {
        attempted_tracks: workspace.tracks().len(),
        triangulated_landmarks,
        rejections,
        gauge_scale_policy,
        min_observations,
    })
}

/// Global translation stage using the ordinary three-view landmark policy.
///
/// Two-view recovery is an explicit candidate-level decision: keeping it out of
/// this primitive prevents an apparently ordinary attempt from silently changing
/// the camera-point rigidity of its BA problem.
pub(crate) fn triangulate_global_translation(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    gauge_scale_policy: GaugeScalePolicy,
) -> Result<TriangulationReport> {
    triangulate_with_min_observations(workspace, triangulator, gauge_scale_policy, 3)
}

/// Report for an incremental triangulation pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncrementalTriangulationReport {
    /// Tracks that had no landmark yet at the start of this pass.
    pub attempted_tracks: usize,
    /// New landmark slots appended by this pass.
    pub new_landmarks: usize,
    /// Previously rejected landmark slots restored to active status.
    #[serde(default)]
    pub revived_landmarks: usize,
    /// Expected reasons attempted source tracks did not produce landmarks.
    pub rejections: TriangulationRejectionSummary,
    /// Total active landmarks after the pass.
    pub total_landmarks: usize,
}

/// Triangulate only tracks that do **not** yet have a landmark, using only their
/// currently-posed observations. Appends `Landmark::active(...)` (continuing the
/// existing landmark-id numbering) and refreshes the primary component. Unlike
/// [`triangulate`], this never wipes existing landmarks and does not require every
/// observation of a track to be posed — it is the incremental driver's grow step.
pub fn triangulate_incremental(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
) -> Result<IncrementalTriangulationReport> {
    triangulate_incremental_scoped_inner(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        None,
    )
}

pub(crate) fn triangulate_incremental_for_views(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    changed_views: &[ViewId],
) -> Result<IncrementalTriangulationReport> {
    triangulate_incremental_scoped_inner(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        Some(changed_views),
    )
}

fn triangulate_incremental_scoped_inner(
    workspace: &mut SfmWorkspace,
    triangulator: &dyn Triangulator,
    min_observations: usize,
    max_reprojection_error: f64,
    changed_views: Option<&[ViewId]>,
) -> Result<IncrementalTriangulationReport> {
    validate_reprojection_limit(max_reprojection_error)?;
    let min_observations = min_observations.max(2);

    let intent = changed_views.map_or(
        TrackUpdateIntent::RestoreArchived,
        TrackUpdateIntent::GrowOrReviveForNewPoses,
    );
    let touched_tracks = changed_views.map(|images| tracks_observing_views(workspace, images));
    let evaluated = evaluate_new_tracks(
        workspace,
        triangulator,
        min_observations,
        max_reprojection_error,
        intent,
        touched_tracks.as_deref(),
    )?;
    let growth = commit_growth(workspace, evaluated)?;

    let added_poses = changed_views.is_some_and(|images| !images.is_empty());
    if added_poses || growth.new_landmarks + growth.revived_landmarks > 0 {
        refresh_primary_component(workspace);
    }

    Ok(IncrementalTriangulationReport {
        attempted_tracks: growth.attempted_tracks,
        new_landmarks: growth.new_landmarks,
        revived_landmarks: growth.revived_landmarks,
        rejections: growth.rejections,
        total_landmarks: workspace.landmarks().active_count(),
    })
}

/// Which immutable source-track observations may participate in one geometric update.
///
/// Keeping this policy typed prevents callers from accidentally interpreting `None` as either
/// "all observations" or "no support restriction". Incremental refits extend the landmark's
/// current support only with observations from newly posed images, so an observation rejected by
/// cleanup cannot silently return during later growth.
#[derive(Debug, Clone, Copy)]
pub(super) enum PosedObservationPolicy<'a> {
    /// Use every currently posed source-track observation.
    All,
    /// Use only observations already supporting the landmark's current geometry.
    CurrentSupport(&'a ObservationSupport),
    /// Preserve current support and admit observations belonging to newly posed images.
    CurrentSupportPlusImages {
        support: &'a ObservationSupport,
        images: &'a [ViewId],
    },
}

impl PosedObservationPolicy<'_> {
    fn includes(self, source_index: usize, view: ViewId) -> bool {
        match self {
            Self::All => true,
            Self::CurrentSupport(support) => support.contains(source_index),
            Self::CurrentSupportPlusImages { support, images } => {
                support.contains(source_index) || images.contains(&view)
            }
        }
    }
}

/// Gather a track's currently posed observations, poses, and intrinsics under one explicit
/// observation-selection policy. Returns `None` if fewer than two selected observations are posed.
pub(super) fn gather_posed<'a>(
    workspace: &'a SfmWorkspace,
    track: &crate::Track,
    policy: PosedObservationPolicy<'_>,
) -> Result<Option<PosedTrack<'a>>> {
    let capacity = track.observations().len();
    let mut posed = PosedTrack {
        observations: Vec::with_capacity(capacity),
        source_indices: Vec::with_capacity(capacity),
    };
    for (source_index, observation) in track.observations().iter().enumerate() {
        if !policy.includes(source_index, observation.view) {
            continue;
        }
        let Some(pose) = workspace.pose(observation.view) else {
            continue;
        };
        let view = workspace
            .view(observation.view)
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: observation.view.index(),
            })?;
        let camera = workspace
            .camera_model(view.intrinsic_id)
            .ok_or(SfmError::InvalidId {
                kind: IntrinsicId::kind(),
                index: view.intrinsic_id.index(),
            })?;
        posed.observations.push(PosedObservation {
            view: observation.view,
            xy: workspace.track_observation_pixel(observation)?,
            pose: &pose.pose,
            camera,
        });
        posed.source_indices.push(source_index);
    }
    Ok((posed.observations.len() >= 2).then_some(posed))
}

pub(super) fn triangulate_posed(
    triangulator: &dyn Triangulator,
    posed: &PosedTrack<'_>,
    track_length: usize,
) -> Result<PosedTriangulationOutcome> {
    let observations = posed
        .observations
        .iter()
        .map(|observation| PosedPixelObservation {
            view: observation.view,
            xy: observation.xy,
            pose: observation.pose,
            camera: observation.camera,
        })
        .collect::<Vec<_>>();
    let triangulated = match triangulator.triangulate(&observations)? {
        TriangulationVerdict::Accepted(triangulated) => triangulated,
        TriangulationVerdict::Rejected(reason) => {
            return Ok(PosedTriangulationOutcome::Rejected(reason));
        }
    };
    triangulated.validate(posed.len())?;
    let TriangulatedLandmark {
        point,
        reprojection_error,
        supporting_observation_indices,
    } = triangulated;
    let all_input_observations = supporting_observation_indices.len() == posed.len()
        && supporting_observation_indices
            .iter()
            .enumerate()
            .all(|(index, &supported)| index == supported);
    let all_source_observations = posed.source_indices.len() == track_length
        && posed
            .source_indices
            .iter()
            .enumerate()
            .all(|(index, &source)| index == source);
    let support = if all_input_observations && all_source_observations {
        ObservationSupport::default()
    } else {
        ObservationSupport::try_from_indices(
            supporting_observation_indices
                .into_vec()
                .into_iter()
                .map(|index| posed.source_indices[index]),
            track_length,
        )?
    };
    Ok(PosedTriangulationOutcome::Accepted(AcceptedTriangulation {
        point,
        reprojection_error,
        support,
    }))
}

pub(super) fn triangulate_posed_with_limit(
    triangulator: &dyn Triangulator,
    posed: &PosedTrack<'_>,
    track_length: usize,
    max_reprojection_error: f64,
) -> Result<PosedTriangulationOutcome> {
    match triangulate_posed(triangulator, posed, track_length)? {
        PosedTriangulationOutcome::Accepted(accepted) => {
            let has_generalized_support = posed.observations.iter().zip(&posed.source_indices).any(
                |(observation, &source_index)| {
                    accepted.support.contains(source_index)
                        && is_generalized_camera(observation.camera)
                },
            );
            if !has_generalized_support {
                return if accepted.reprojection_error.value() > max_reprojection_error {
                    Ok(PosedTriangulationOutcome::Rejected(
                        TriangulationRejectReason::ReprojectionLimit,
                    ))
                } else {
                    Ok(PosedTriangulationOutcome::Accepted(accepted))
                };
            }

            for (observation, &source_index) in posed.observations.iter().zip(&posed.source_indices)
            {
                if !accepted.support.contains(source_index) {
                    continue;
                }
                let camera_point = observation.pose.project_to_camera(&accepted.point);
                let policy = CameraResidualPolicy::from_pixel_limit(
                    observation.camera,
                    max_reprojection_error,
                )?;
                if !policy.accepts(observation.camera, &camera_point.coords, observation.xy) {
                    return Ok(PosedTriangulationOutcome::Rejected(
                        TriangulationRejectReason::ReprojectionLimit,
                    ));
                }
            }
            Ok(PosedTriangulationOutcome::Accepted(accepted))
        }
        outcome => Ok(outcome),
    }
}

pub(super) fn validate_reprojection_limit(limit: f64) -> Result<()> {
    if limit.is_nan() || limit < 0.0 {
        return Err(SfmError::InvalidConfig(
            "max_reprojection_error must be non-negative or +infinity".to_owned(),
        ));
    }
    Ok(())
}
