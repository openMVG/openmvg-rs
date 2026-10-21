// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Held-out track re-triangulation that checks the cross-camera scale consistency of metric
//! poses.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Track-scale validation (validate_track_scale_consistency)            │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: posed workspace with tracks, a Triangulator, the config
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Eligible tracks and camera-balanced selection                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Tracks with ≥ 3 posed observations. Each camera first receives up to
//! min(max_tracks / cameras, 32) tracks as its preferred held-out view;
//! the rest of the 20 000-track budget is filled by an even stride.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Hold out and re-triangulate (per track, ordered parallel)    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Hold out ≤ 2 observations (the preferred camera, then the ones farthest
//! from the track's camera centroid). Triangulate the point from the ≤ 6
//! remaining cameras farthest from the held-out one.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Score the held-out camera                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Angle (degrees) between the observed and the predicted bearing; a
//! rejected triangulation or an unusable bearing is counted, not scored.
//! The angle is invariant to the similarity gauge, so it exposes cameras
//! whose scale disagrees with the rest.
//!   │
//!   ▼
//! Output: TrackScaleValidationReport — residual distribution, fractions
//!         within 1°/2°/5°, per-camera statistics and p90 tails (≥ 4
//!         held-outs, p90 > 5°); the Global policy decides from it
//! ```

use std::collections::BTreeMap;

use nalgebra::Vector3;
use openmvg_core::camera::{Camera, CameraModel};

use crate::backends::{
    PosedPixelObservation, TriangulatedLandmark, TriangulationRejectReason, TriangulationVerdict,
    Triangulator,
};
use crate::ops::averaging::AngularResidualDistribution;
use crate::{Result, SfmError, SfmWorkspace, TrackId, ViewId};

#[cfg(feature = "parallel")]
const MIN_PARALLEL_JOBS: usize = 128;

/// Bounded post-fit check of metric poses against stable multi-view tracks.
///
/// For each selected track, one or more observations are withheld. The point is re-triangulated
/// from the other posed observations and projected as a bearing into the withheld camera. This is
/// invariant to the reconstruction's global similarity gauge and directly exposes cross-camera
/// scale disagreement. It is still correlated with bundle adjustment because the candidate poses
/// may have been fitted to the same tracks. This check only produces a report; the Global
/// engine's track-validation policy decides from that report whether to reject a core.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TrackScaleValidationConfig {
    max_tracks: usize,
    max_holdouts_per_track: usize,
    max_anchor_views: usize,
    minimum_track_views: usize,
    max_assigned_tracks_per_camera: usize,
    minimum_camera_support: usize,
    camera_p90_tail_deg: f64,
}

config_accessors!(TrackScaleValidationConfig {
    /// Maximum number of stable tracks selected before held-out expansion.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_tracks, with_max_tracks: usize;
    /// Maximum number of held-out observations evaluated per selected track.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_holdouts_per_track, with_max_holdouts_per_track: usize;
    /// Maximum number of remaining observations used to triangulate one held-out prediction.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_anchor_views, with_max_anchor_views: usize;
    /// Minimum posed observations required before a track is eligible.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy minimum_track_views, with_minimum_track_views: usize;
    /// Maximum number of tracks assigned specifically to one camera during balanced sampling.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_assigned_tracks_per_camera, with_max_assigned_tracks_per_camera: usize;
    /// Minimum evaluated held-outs before a camera can be classified as a diagnostic tail.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy minimum_camera_support, with_minimum_camera_support: usize;
    /// Camera-tail threshold used only for the report's tail statistics; it rejects nothing.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy camera_p90_tail_deg, with_camera_p90_tail_deg: f64;
});

impl Default for TrackScaleValidationConfig {
    fn default() -> Self {
        Self {
            max_tracks: 20_000,
            max_holdouts_per_track: 2,
            max_anchor_views: 6,
            minimum_track_views: 3,
            max_assigned_tracks_per_camera: 32,
            minimum_camera_support: 4,
            camera_p90_tail_deg: 5.0,
        }
    }
}

impl TrackScaleValidationConfig {
    /// Check every bound; fails with `SfmError::InvalidConfig` naming the first invalid setting.
    pub fn validate(self) -> Result<()> {
        if self.max_tracks == 0 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation requires at least one selected track".to_owned(),
            ));
        }
        if self.max_holdouts_per_track == 0 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation requires at least one held-out observation".to_owned(),
            ));
        }
        if self.max_anchor_views < 2 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation requires at least two anchor views".to_owned(),
            ));
        }
        if self.minimum_track_views < 3 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation requires tracks with at least three posed views".to_owned(),
            ));
        }
        if self.max_assigned_tracks_per_camera == 0 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation camera assignment cap must be positive".to_owned(),
            ));
        }
        if self.minimum_camera_support == 0 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation camera support must be positive".to_owned(),
            ));
        }
        if !self.camera_p90_tail_deg.is_finite() || self.camera_p90_tail_deg <= 0.0 {
            return Err(SfmError::InvalidConfig(
                "track-scale validation camera-tail threshold must be finite and positive"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Held-out evidence attributed to one camera.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrackScaleCameraReport {
    /// Camera the evidence belongs to.
    pub view: ViewId,
    /// Eligible tracks (enough posed views) observed by this camera.
    pub eligible_track_count: usize,
    /// Tracks selected with this camera as their preferred held-out view.
    pub assigned_track_count: usize,
    /// Held-out predictions attempted in this camera.
    pub attempted_holdout_count: usize,
    /// Held-out predictions that produced a finite angular residual.
    pub evaluated_holdout_count: usize,
    /// Held-outs whose anchor triangulation was rejected.
    pub triangulation_rejection_count: usize,
    /// Held-outs whose observed or predicted bearing was not finite or degenerate.
    pub invalid_prediction_count: usize,
    /// `evaluated_holdout_count` over `attempted_holdout_count`; absent when nothing was attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_holdout_fraction: Option<f64>,
    /// `triangulation_rejection_count` over `attempted_holdout_count`; absent when nothing was
    /// attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triangulation_rejection_fraction: Option<f64>,
    /// `invalid_prediction_count` over `attempted_holdout_count`; absent when nothing was
    /// attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid_prediction_fraction: Option<f64>,
    /// Evaluated held-outs whose re-triangulated point lies in front of this camera.
    pub positive_depth_count: usize,
    /// Angle between observed and predicted bearings over evaluated held-outs; absent without any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_residual_deg: Option<AngularResidualDistribution>,
    /// Diagnostic only. The adaptive admission policy does not consume this field.
    pub is_camera_tail: bool,
}

/// Typed accounting of triangulator rejections encountered by the held-out pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TrackScaleTriangulationRejection {
    /// Triangulator rejection reason.
    pub reason: TriangulationRejectReason,
    /// Held-outs rejected for this reason.
    pub count: usize,
}

/// Solver-independent post-fit track consistency report.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrackScaleValidationReport {
    /// Version of this report layout; currently `1`.
    pub schema_version: u32,
    /// Configuration the validation ran with.
    pub config: TrackScaleValidationConfig,
    /// Cameras with a pose when the validation ran.
    pub posed_camera_count: usize,
    /// Tracks in the workspace.
    pub input_track_count: usize,
    /// Tracks with at least `minimum_track_views` posed observations.
    pub eligible_track_count: usize,
    /// Eligible tracks selected for held-out evaluation (at most `max_tracks`).
    pub selected_track_count: usize,
    /// Held-out predictions attempted across all selected tracks.
    pub attempted_holdout_count: usize,
    /// Held-out predictions that produced a finite angular residual.
    pub evaluated_holdout_count: usize,
    /// Held-outs whose anchor triangulation was rejected.
    pub triangulation_rejection_count: usize,
    /// Held-outs whose observed or predicted bearing was not finite or degenerate.
    pub invalid_prediction_count: usize,
    /// `evaluated_holdout_count` over `attempted_holdout_count`; absent when nothing was attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_holdout_fraction: Option<f64>,
    /// `triangulation_rejection_count` over `attempted_holdout_count`; absent when nothing was
    /// attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triangulation_rejection_fraction: Option<f64>,
    /// `invalid_prediction_count` over `attempted_holdout_count`; absent when nothing was
    /// attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid_prediction_fraction: Option<f64>,
    /// Evaluated held-outs whose re-triangulated point lies in front of the held-out camera.
    pub positive_depth_count: usize,
    /// `positive_depth_count` over `evaluated_holdout_count`; absent when nothing was evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive_depth_fraction: Option<f64>,
    /// Angle between observed and predicted bearings over all evaluated held-outs; absent without
    /// any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_residual_deg: Option<AngularResidualDistribution>,
    /// Fraction of evaluated held-outs with a residual of at most 1 degree; absent without any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_one_degree_fraction: Option<f64>,
    /// Fraction of evaluated held-outs with a residual of at most 2 degrees; absent without any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_two_degree_fraction: Option<f64>,
    /// Fraction of evaluated held-outs with a residual of at most 5 degrees; absent without any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_five_degree_fraction: Option<f64>,
    /// Cameras flagged as `is_camera_tail`.
    pub camera_tail_count: usize,
    /// Per-camera evidence for every camera with an eligible track or an attempted held-out, in
    /// `ViewId` order.
    pub cameras: Vec<TrackScaleCameraReport>,
    /// Triangulator rejections by reason, in stable reason order.
    pub triangulation_rejections: Vec<TrackScaleTriangulationRejection>,
}

#[derive(Debug, Clone, Copy)]
struct SelectedTrack {
    track: TrackId,
    preferred_holdout: Option<ViewId>,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedObservation<'a> {
    view: ViewId,
    xy: nalgebra::Point2<f64>,
    pose: &'a openmvg_core::model::WorldToCameraPose,
    camera: &'a Camera,
}

#[derive(Debug)]
enum HoldoutOutcome {
    Evaluated {
        view: ViewId,
        angle_deg: f64,
        positive_depth: bool,
    },
    TriangulationRejected {
        view: ViewId,
        reason: TriangulationRejectReason,
    },
    InvalidPrediction {
        view: ViewId,
    },
}

#[derive(Default)]
struct CameraAccumulator {
    attempted: usize,
    evaluated: usize,
    triangulation_rejections: usize,
    invalid_predictions: usize,
    positive_depth: usize,
    angular_residuals: Vec<f64>,
}

/// Re-triangulate bounded stable tracks from anchor cameras and score withheld cameras.
///
/// Selection is deterministic and camera-balanced. Work is parallelized with ordered collection,
/// so serialized results do not depend on Rayon scheduling.
pub fn validate_track_scale_consistency(
    workspace: &SfmWorkspace,
    triangulator: &dyn Triangulator,
    config: TrackScaleValidationConfig,
) -> Result<TrackScaleValidationReport> {
    config.validate()?;
    let view_count = workspace.views().len();
    let posed_camera_count = workspace.registered_view_count();
    let mut eligible_by_camera = vec![0usize; view_count];
    let mut eligible_tracks = Vec::new();

    for track in workspace.tracks().as_slice() {
        let posed_observation_count = track
            .observations()
            .iter()
            .filter(|observation| workspace.has_pose(observation.view))
            .count();
        if posed_observation_count < config.minimum_track_views {
            continue;
        }
        eligible_tracks.push(track.id());
        for observation in track.observations() {
            if workspace.has_pose(observation.view) {
                eligible_by_camera[observation.view.index()] += 1;
            }
        }
    }

    let (selected, assigned_by_camera) =
        select_tracks(workspace, &eligible_tracks, posed_camera_count, config);
    let results = map_ordered(&selected, |selected| {
        evaluate_selected_track(workspace, triangulator, *selected, config)
    });

    let mut aggregate_residuals = Vec::new();
    let mut camera_accumulators: Vec<CameraAccumulator> = (0..view_count)
        .map(|_| CameraAccumulator::default())
        .collect();
    let mut rejection_counts = BTreeMap::<TriangulationRejectReason, usize>::new();
    let mut attempted_holdout_count = 0usize;
    let mut evaluated_holdout_count = 0usize;
    let mut triangulation_rejection_count = 0usize;
    let mut invalid_prediction_count = 0usize;
    let mut positive_depth_count = 0usize;

    for result in results {
        for outcome in result? {
            attempted_holdout_count += 1;
            match outcome {
                HoldoutOutcome::Evaluated {
                    view,
                    angle_deg,
                    positive_depth,
                } => {
                    evaluated_holdout_count += 1;
                    positive_depth_count += usize::from(positive_depth);
                    aggregate_residuals.push(angle_deg);
                    let camera = &mut camera_accumulators[view.index()];
                    camera.attempted += 1;
                    camera.evaluated += 1;
                    camera.positive_depth += usize::from(positive_depth);
                    camera.angular_residuals.push(angle_deg);
                }
                HoldoutOutcome::TriangulationRejected { view, reason } => {
                    triangulation_rejection_count += 1;
                    *rejection_counts.entry(reason).or_default() += 1;
                    let camera = &mut camera_accumulators[view.index()];
                    camera.attempted += 1;
                    camera.triangulation_rejections += 1;
                }
                HoldoutOutcome::InvalidPrediction { view } => {
                    invalid_prediction_count += 1;
                    let camera = &mut camera_accumulators[view.index()];
                    camera.attempted += 1;
                    camera.invalid_predictions += 1;
                }
            }
        }
    }

    let within_fraction = |threshold: f64| {
        (!aggregate_residuals.is_empty()).then(|| {
            aggregate_residuals
                .iter()
                .filter(|residual| **residual <= threshold)
                .count() as f64
                / aggregate_residuals.len() as f64
        })
    };
    let within_one_degree_fraction = within_fraction(1.0);
    let within_two_degree_fraction = within_fraction(2.0);
    let within_five_degree_fraction = within_fraction(5.0);
    let angular_residual_deg =
        crate::ops::averaging::angular_distribution(&mut aggregate_residuals);

    let mut cameras = Vec::new();
    for (view_index, mut accumulator) in camera_accumulators.into_iter().enumerate() {
        if accumulator.attempted == 0 && eligible_by_camera[view_index] == 0 {
            continue;
        }
        let angular_residual_deg =
            crate::ops::averaging::angular_distribution(&mut accumulator.angular_residuals);
        let is_camera_tail = accumulator.evaluated >= config.minimum_camera_support
            && angular_residual_deg
                .as_ref()
                .is_some_and(|distribution| distribution.p90_deg > config.camera_p90_tail_deg);
        cameras.push(TrackScaleCameraReport {
            view: ViewId::new(view_index),
            eligible_track_count: eligible_by_camera[view_index],
            assigned_track_count: assigned_by_camera[view_index],
            attempted_holdout_count: accumulator.attempted,
            evaluated_holdout_count: accumulator.evaluated,
            triangulation_rejection_count: accumulator.triangulation_rejections,
            invalid_prediction_count: accumulator.invalid_predictions,
            evaluated_holdout_fraction: fraction(accumulator.evaluated, accumulator.attempted),
            triangulation_rejection_fraction: fraction(
                accumulator.triangulation_rejections,
                accumulator.attempted,
            ),
            invalid_prediction_fraction: fraction(
                accumulator.invalid_predictions,
                accumulator.attempted,
            ),
            positive_depth_count: accumulator.positive_depth,
            angular_residual_deg,
            is_camera_tail,
        });
    }
    let camera_tail_count = cameras
        .iter()
        .filter(|camera| camera.is_camera_tail)
        .count();

    Ok(TrackScaleValidationReport {
        schema_version: 1,
        config,
        posed_camera_count,
        input_track_count: workspace.tracks().len(),
        eligible_track_count: eligible_tracks.len(),
        selected_track_count: selected.len(),
        attempted_holdout_count,
        evaluated_holdout_count,
        triangulation_rejection_count,
        invalid_prediction_count,
        evaluated_holdout_fraction: fraction(evaluated_holdout_count, attempted_holdout_count),
        triangulation_rejection_fraction: fraction(
            triangulation_rejection_count,
            attempted_holdout_count,
        ),
        invalid_prediction_fraction: fraction(invalid_prediction_count, attempted_holdout_count),
        positive_depth_count,
        positive_depth_fraction: (evaluated_holdout_count > 0)
            .then(|| positive_depth_count as f64 / evaluated_holdout_count as f64),
        angular_residual_deg,
        within_one_degree_fraction,
        within_two_degree_fraction,
        within_five_degree_fraction,
        camera_tail_count,
        cameras,
        triangulation_rejections: rejection_counts
            .into_iter()
            .map(|(reason, count)| TrackScaleTriangulationRejection { reason, count })
            .collect(),
    })
}

fn select_tracks(
    workspace: &SfmWorkspace,
    eligible_tracks: &[TrackId],
    posed_camera_count: usize,
    config: TrackScaleValidationConfig,
) -> (Vec<SelectedTrack>, Vec<usize>) {
    let mut selected_mask = vec![false; workspace.tracks().len()];
    let mut assigned_by_camera = vec![0usize; workspace.views().len()];
    let mut selected = Vec::new();
    let quota = config
        .max_tracks
        .checked_div(posed_camera_count)
        .unwrap_or(0)
        .min(config.max_assigned_tracks_per_camera);

    if quota > 0 {
        for &track_id in eligible_tracks {
            let Some(track) = workspace.tracks().get(track_id) else {
                continue;
            };
            let target = track
                .observations()
                .iter()
                .filter(|observation| {
                    workspace.has_pose(observation.view)
                        && assigned_by_camera[observation.view.index()] < quota
                })
                .min_by_key(|observation| {
                    (
                        assigned_by_camera[observation.view.index()],
                        observation.view,
                    )
                })
                .map(|observation| observation.view);
            let Some(target) = target else {
                continue;
            };
            selected_mask[track_id.index()] = true;
            assigned_by_camera[target.index()] += 1;
            selected.push(SelectedTrack {
                track: track_id,
                preferred_holdout: Some(target),
            });
        }
    }

    let remaining_capacity = config.max_tracks.saturating_sub(selected.len());
    if remaining_capacity > 0 {
        let remaining: Vec<_> = eligible_tracks
            .iter()
            .copied()
            .filter(|track| !selected_mask[track.index()])
            .collect();
        let take = remaining_capacity.min(remaining.len());
        for slot in 0..take {
            let index = slot * remaining.len() / take;
            let track = remaining[index];
            selected.push(SelectedTrack {
                track,
                preferred_holdout: None,
            });
        }
    }
    selected.sort_by_key(|selected| selected.track);
    (selected, assigned_by_camera)
}

fn evaluate_selected_track(
    workspace: &SfmWorkspace,
    triangulator: &dyn Triangulator,
    selected: SelectedTrack,
    config: TrackScaleValidationConfig,
) -> Result<Vec<HoldoutOutcome>> {
    let track = workspace
        .tracks()
        .get(selected.track)
        .ok_or(SfmError::InvalidId {
            kind: TrackId::kind(),
            index: selected.track.index(),
        })?;
    let mut observations = Vec::with_capacity(track.observations().len());
    for observation in track.observations() {
        let Some(pose) = workspace.pose(observation.view) else {
            continue;
        };
        let image = workspace
            .view(observation.view)
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: observation.view.index(),
            })?;
        let camera = workspace
            .camera_model(image.intrinsic_id)
            .ok_or(SfmError::InvalidId {
                kind: crate::IntrinsicId::kind(),
                index: image.intrinsic_id.index(),
            })?;
        observations.push(ResolvedObservation {
            view: observation.view,
            xy: workspace.track_observation_pixel(observation)?,
            pose: &pose.pose,
            camera,
        });
    }
    if observations.len() < config.minimum_track_views {
        return Ok(Vec::new());
    }

    let centroid = observations
        .iter()
        .fold(Vector3::zeros(), |sum, observation| {
            sum + observation.pose.center().coords
        })
        / observations.len() as f64;
    let mut holdout_indices: Vec<_> = (0..observations.len()).collect();
    holdout_indices.sort_by(|&left, &right| {
        let left_distance = (observations[left].pose.center().coords - centroid).norm_squared();
        let right_distance = (observations[right].pose.center().coords - centroid).norm_squared();
        right_distance
            .total_cmp(&left_distance)
            .then_with(|| observations[left].view.cmp(&observations[right].view))
    });
    if let Some(preferred) = selected.preferred_holdout
        && let Some(position) = holdout_indices
            .iter()
            .position(|&index| observations[index].view == preferred)
    {
        holdout_indices.swap(0, position);
    }
    holdout_indices.truncate(config.max_holdouts_per_track.min(observations.len()));

    holdout_indices
        .into_iter()
        .map(|holdout_index| {
            evaluate_holdout(
                triangulator,
                &observations,
                holdout_index,
                config.max_anchor_views,
            )
        })
        .collect()
}

fn evaluate_holdout(
    triangulator: &dyn Triangulator,
    observations: &[ResolvedObservation<'_>],
    holdout_index: usize,
    max_anchor_views: usize,
) -> Result<HoldoutOutcome> {
    let heldout = observations[holdout_index];
    let mut anchor_indices: Vec<_> = (0..observations.len())
        .filter(|&index| index != holdout_index)
        .collect();
    anchor_indices.sort_by(|&left, &right| {
        let left_distance =
            (observations[left].pose.center() - heldout.pose.center()).norm_squared();
        let right_distance =
            (observations[right].pose.center() - heldout.pose.center()).norm_squared();
        right_distance
            .total_cmp(&left_distance)
            .then_with(|| observations[left].view.cmp(&observations[right].view))
    });
    anchor_indices.truncate(max_anchor_views);
    let anchors: Vec<_> = anchor_indices
        .iter()
        .map(|&index| {
            let observation = observations[index];
            PosedPixelObservation {
                view: observation.view,
                xy: observation.xy,
                pose: observation.pose,
                camera: observation.camera,
            }
        })
        .collect();
    let point = match triangulator.triangulate(&anchors)? {
        TriangulationVerdict::Accepted(point) => {
            point.validate(anchors.len())?;
            point
        }
        TriangulationVerdict::Rejected(reason) => {
            return Ok(HoldoutOutcome::TriangulationRejected {
                view: heldout.view,
                reason,
            });
        }
    };
    heldout_prediction(heldout, &point)
}

fn heldout_prediction(
    heldout: ResolvedObservation<'_>,
    point: &TriangulatedLandmark,
) -> Result<HoldoutOutcome> {
    let Ok(observed) = heldout.camera.unproject(&heldout.xy) else {
        return Ok(HoldoutOutcome::InvalidPrediction { view: heldout.view });
    };
    let predicted = heldout.pose.project_to_camera(&point.point).coords;
    if !observed.iter().all(|value| value.is_finite())
        || !predicted.iter().all(|value| value.is_finite())
        || observed.norm_squared() <= f64::EPSILON
        || predicted.norm_squared() <= f64::EPSILON
    {
        return Ok(HoldoutOutcome::InvalidPrediction { view: heldout.view });
    }
    let observed = observed.normalize();
    let predicted_direction = predicted.normalize();
    let angle_deg = observed
        .cross(&predicted_direction)
        .norm()
        .atan2(observed.dot(&predicted_direction))
        .to_degrees();
    if !angle_deg.is_finite() {
        return Ok(HoldoutOutcome::InvalidPrediction { view: heldout.view });
    }
    Ok(HoldoutOutcome::Evaluated {
        view: heldout.view,
        angle_deg,
        // In front along the observed ray: a spherical view sees points behind its optical axis.
        positive_depth: observed.dot(&predicted_direction) > 0.0,
    })
}

fn fraction(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(feature = "parallel")]
fn map_ordered<T, U, F>(items: &[T], map: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    if items.len() < MIN_PARALLEL_JOBS {
        items.iter().map(&map).collect()
    } else {
        use rayon::prelude::*;
        items.par_iter().map(map).collect()
    }
}

#[cfg(not(feature = "parallel"))]
fn map_ordered<T, U, F>(items: &[T], map: F) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    items.iter().map(map).collect()
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Rotation3};

    use crate::adapters::triangulation::DltTriangulator;
    use crate::model::{ImageMeta, PoseRecord};
    use crate::{
        Camera, ComponentId, FeatureId, FeatureSet, IntrinsicId, Track, TrackObservation, TrackSet,
    };

    use super::*;

    fn workspace(camera_two_x: f64) -> SfmWorkspace {
        let points = (0..24)
            .map(|track_index| {
                Point3::new(
                    -0.8 + 0.07 * track_index as f64,
                    -0.3 + 0.12 * (track_index % 6) as f64,
                    4.0 + 0.15 * (track_index % 5) as f64,
                )
            })
            .collect::<Vec<_>>();
        let tracks = TrackSet::try_new(
            points
                .iter()
                .enumerate()
                .map(|(track_index, _point)| {
                    let observations = (0..3)
                        .map(|camera| TrackObservation {
                            view: ViewId::new(camera),
                            feature: FeatureId::new(track_index),
                        })
                        .collect();
                    Track::try_new(TrackId::new(track_index), observations)
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        )
        .unwrap();
        let feature_sets = (0..3)
            .map(|camera| {
                FeatureSet::try_new(
                    points
                        .iter()
                        .map(|point| {
                            let camera_point = point.coords - Vector3::new(camera as f64, 0.0, 0.0);
                            nalgebra::Point2::new(
                                100.0 * camera_point.x / camera_point.z + 32.0,
                                100.0 * camera_point.y / camera_point.z + 24.0,
                            )
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..3).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .tracks(tracks, feature_sets)
            .build()
            .unwrap();
        for (image, center_x) in [0.0, 1.0, camera_two_x].into_iter().enumerate() {
            workspace
                .set_pose(PoseRecord {
                    id: crate::PoseId::new(image),
                    pose: openmvg_core::model::WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::new(center_x, 0.0, 0.0),
                    )
                    .unwrap(),
                    component: Some(ComponentId::new(0)),
                })
                .unwrap();
        }
        workspace
    }

    fn test_config() -> TrackScaleValidationConfig {
        TrackScaleValidationConfig {
            max_tracks: 24,
            max_holdouts_per_track: 2,
            max_anchor_views: 2,
            minimum_track_views: 3,
            max_assigned_tracks_per_camera: 8,
            minimum_camera_support: 4,
            camera_p90_tail_deg: 1.0,
        }
    }

    #[test]
    fn clean_geometry_has_small_heldout_residuals() {
        let report = validate_track_scale_consistency(
            &workspace(2.0),
            &DltTriangulator::default(),
            test_config(),
        )
        .unwrap();
        assert_eq!(report.selected_track_count, 24);
        assert_eq!(report.evaluated_holdout_count, 48);
        assert!(report.angular_residual_deg.unwrap().p90_deg < 1.0e-6);
        assert_eq!(report.camera_tail_count, 0);
    }

    #[test]
    fn a_spherical_holdout_behind_its_optical_axis_has_positive_depth() {
        let camera =
            Camera::Spherical(openmvg_core::camera::SphericalCamera::try_new(512).unwrap());
        let pose = openmvg_core::model::WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let behind = Point3::new(0.5, -0.2, -4.0);
        let landmark = |point| TriangulatedLandmark {
            point,
            reprojection_error: openmvg_core::PixelResidual::try_new(0.0).unwrap(),
            supporting_observation_indices: Box::new([0, 1]),
        };
        let heldout = |point: &Point3<f64>| ResolvedObservation {
            view: ViewId::new(0),
            xy: camera.project(&point.coords).unwrap(),
            pose: &pose,
            camera: &camera,
        };

        let in_front = heldout_prediction(heldout(&behind), &landmark(behind)).unwrap();
        let opposite =
            heldout_prediction(heldout(&behind), &landmark(Point3::new(-0.5, 0.2, 4.0))).unwrap();

        let HoldoutOutcome::Evaluated {
            positive_depth: true,
            angle_deg,
            ..
        } = in_front
        else {
            panic!("expected a positive-depth holdout, got {in_front:?}");
        };
        assert!(angle_deg < 1e-9);
        assert!(matches!(
            opposite,
            HoldoutOutcome::Evaluated {
                positive_depth: false,
                ..
            }
        ));
    }

    #[test]
    fn independent_camera_scale_error_is_localized() {
        let report = validate_track_scale_consistency(
            &workspace(4.0),
            &DltTriangulator::default(),
            test_config(),
        )
        .unwrap();
        let camera_two = report
            .cameras
            .iter()
            .find(|camera| camera.view == ViewId::new(2))
            .unwrap();
        assert!(camera_two.assigned_track_count >= 4);
        assert!(camera_two.is_camera_tail);
        assert!(camera_two.angular_residual_deg.as_ref().unwrap().p90_deg > 5.0);
    }

    #[test]
    fn invalid_configuration_fails_before_execution() {
        let config = TrackScaleValidationConfig {
            max_anchor_views: 1,
            ..TrackScaleValidationConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
