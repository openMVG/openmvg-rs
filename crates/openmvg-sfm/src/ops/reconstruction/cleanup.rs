// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reconstruction cleanup and component maintenance.

use super::triangulation::{PosedObservationPolicy, gather_posed};
use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Result of peeling views that no longer have enough valid landmark observations
/// after bundle adjustment. The caller may protect initial-pair or imported poses.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PoseSupportCleanupReport {
    /// Minimum valid observations a pose needed to be kept; `0` when the cleanup was not
    /// count-based.
    pub min_observations: usize,
    /// Removal rounds that removed at least one pose.
    pub rounds: usize,
    /// Views whose pose was removed, sorted and deduplicated.
    pub removed_views: Vec<ViewId>,
    /// Landmarks rejected because the removed poses left them without enough posed support.
    pub rejected_landmarks: usize,
}

impl PoseSupportCleanupReport {
    /// Whether no image was removed; landmark rejections alone do not count.
    pub fn is_empty(&self) -> bool {
        self.removed_views.is_empty()
    }
}

pub fn release_matching_data(workspace: &mut SfmWorkspace) -> Result<()> {
    if workspace.matching_data_released {
        return Ok(());
    }
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "tracks with retained feature geometry",
            method: "build_tracks",
        });
    }
    for track in workspace.tracks().as_slice() {
        for observation in track.observations() {
            workspace.track_observation_pixel(observation)?;
        }
    }
    workspace.release_descriptors()?;
    for index in 0..workspace.keypoints.slot_count() {
        workspace.keypoints.clear(index)?;
    }
    for index in 0..workspace.matches.slot_count() {
        workspace.matches.clear(index)?;
    }
    workspace.matching_data_released = true;
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LandmarkSupportCleanupReport {
    pub rejected_landmarks: usize,
    pub removed_observations: usize,
}

impl LandmarkSupportCleanupReport {
    /// Whether the cleanup left both landmark membership and observation support unchanged.
    pub const fn is_empty(&self) -> bool {
        self.rejected_landmarks == 0 && self.removed_observations == 0
    }

    /// Accumulate another cleanup receipt from the same refinement transaction.
    pub fn merge(&mut self, other: Self) {
        self.rejected_landmarks += other.rejected_landmarks;
        self.removed_observations += other.removed_observations;
    }
}

/// Prune gross observation outliers and reject landmarks that no longer have
/// valid multi-view support after bundle adjustment.
///
/// Perspective observations are removed individually when projection fails or
/// their residual exceeds the pixel limit. A landmark survives while at least two
/// valid observations with sufficient apex angle remain. Perspective observations use the
/// supplied pixel limit; fisheye and spherical observations use its camera-scale-equivalent
/// bearing-angle limit.
///
pub(super) fn validate_landmark_cleanup_config(
    max_pixel_residual: f64,
    min_angle_deg: f64,
) -> Result<()> {
    if max_pixel_residual.is_nan() || max_pixel_residual < 0.0 {
        return Err(SfmError::InvalidConfig(
            "maximum pixel residual must be non-negative or +infinity".to_owned(),
        ));
    }
    if !min_angle_deg.is_finite() || !(0.0..=180.0).contains(&min_angle_deg) {
        return Err(SfmError::InvalidConfig(
            "minimum triangulation angle must be finite and in [0, 180] degrees".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn clean_outlier_landmark_support(
    workspace: &mut SfmWorkspace,
    max_pixel_residual: f64,
    min_angle_deg: f64,
) -> Result<LandmarkSupportCleanupReport> {
    validate_landmark_cleanup_config(max_pixel_residual, min_angle_deg)?;
    let min_cos = min_angle_deg.to_radians().cos();
    let mut candidates = Vec::new();
    for landmark in workspace.landmarks().iter_active() {
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        candidates.push((landmark, track));
    }

    enum CleanupOutcome {
        Reject(LandmarkId),
        RetainUnchanged {
            landmark: LandmarkId,
            reprojection_error: openmvg_core::PixelResidual,
        },
        RetainPruned {
            landmark: LandmarkId,
            support: ObservationSupport,
            reprojection_error: openmvg_core::PixelResidual,
            removed_observations: usize,
        },
    }

    let clean_one = |candidate: &(&Landmark, &crate::Track)| -> Result<CleanupOutcome> {
        let (landmark, track) = *candidate;
        let posed = match gather_posed(
            workspace,
            track,
            PosedObservationPolicy::CurrentSupport(&landmark.observation_support),
        ) {
            Ok(Some(posed)) => posed,
            Ok(None) => return Ok(CleanupOutcome::Reject(landmark.id)),
            Err(error) => return Err(error),
        };
        let x = landmark.point;
        let mut sum_res = 0.0;
        let mut retained = Vec::with_capacity(posed.observations.len());
        for (obs, &source_index) in posed.observations.iter().zip(&posed.source_indices) {
            // world → camera: p = R·(X − C).
            let pc = obs.pose.rotation() * (x - obs.pose.center());
            let policy = CameraResidualPolicy::from_pixel_limit(obs.camera, max_pixel_residual)?;
            let Some(evaluation) = policy.evaluate(obs.camera, &pc, obs.xy) else {
                continue;
            };
            if !evaluation.accepted {
                continue;
            }
            retained.push((source_index, obs.pose.center().coords));
            sum_res += evaluation.pixel_equivalent;
        }
        if retained.len() < 2 {
            return Ok(CleanupOutcome::Reject(landmark.id));
        }
        // Largest apex angle between any two viewing rays = smallest cosine. `best_cos` only
        // decreases; once it drops to/below `min_cos` the angle can never trigger a reject, so we
        // stop early — the final `best_cos > min_cos` boolean is identical to a full scan.
        let mut best_cos = 1.0_f64;
        'scan: for i in 0..retained.len() {
            for j in (i + 1)..retained.len() {
                let a = (x.coords - retained[i].1).normalize();
                let b = (x.coords - retained[j].1).normalize();
                best_cos = best_cos.min(a.dot(&b).clamp(-1.0, 1.0));
                if best_cos <= min_cos {
                    break 'scan;
                }
            }
        }
        if best_cos > min_cos {
            return Ok(CleanupOutcome::Reject(landmark.id));
        }
        let removed_observations = landmark
            .observation_support
            .len(track.observations().len())
            .saturating_sub(retained.len());
        let reprojection_error =
            openmvg_core::PixelResidual::try_new(sum_res / retained.len() as f64).map_err(
                |_| SfmError::InvalidInput("cleanup produced an invalid pixel residual".to_owned()),
            )?;
        if removed_observations == 0 {
            Ok(CleanupOutcome::RetainUnchanged {
                landmark: landmark.id,
                reprojection_error,
            })
        } else {
            Ok(CleanupOutcome::RetainPruned {
                landmark: landmark.id,
                support: ObservationSupport::try_from_indices(
                    retained.iter().map(|(source_index, _)| *source_index),
                    track.observations().len(),
                )?,
                reprojection_error,
                removed_observations,
            })
        }
    };
    let mut results = map_ordered(&candidates, clean_one);
    if let Some(error) = take_first_error(&mut results) {
        return Err(error);
    }
    let mut report = LandmarkSupportCleanupReport::default();
    for result in results {
        match result? {
            CleanupOutcome::Reject(id) => {
                workspace
                    .landmarks
                    .get_mut(id)
                    .ok_or(SfmError::InvariantViolation(format!(
                        "active landmark {id} disappeared before cleanup"
                    )))?
                    .status = LandmarkStatus::Rejected;
                report.rejected_landmarks += 1;
            }
            CleanupOutcome::RetainUnchanged {
                landmark,
                reprojection_error,
            } => {
                workspace
                    .landmarks
                    .get_mut(landmark)
                    .ok_or(SfmError::InvariantViolation(format!(
                        "active landmark {landmark} disappeared before cleanup"
                    )))?
                    .reprojection_error = reprojection_error;
            }
            CleanupOutcome::RetainPruned {
                landmark,
                support,
                reprojection_error,
                removed_observations,
            } => {
                let landmark =
                    workspace
                        .landmarks
                        .get_mut(landmark)
                        .ok_or(SfmError::InvariantViolation(format!(
                            "active landmark {landmark} disappeared before cleanup"
                        )))?;
                landmark.observation_support = support;
                landmark.reprojection_error = reprojection_error;
                report.removed_observations += removed_observations;
            }
        }
    }
    if report.rejected_landmarks > 0 {
        refresh_primary_component(workspace);
    }
    Ok(report)
}

/// Remove non-protected poses whose post-BA reprojection-valid support is too small.
///
/// The camera/landmark bipartite graph is peeled to a fixed point: removing one pose can leave
/// a landmark with fewer than two posed observations, which can in turn make another pose
/// unsupported. Source tracks remain immutable, so a removed image can be registered again after
/// later growth supplies better structure.
pub(crate) fn remove_unsupported_poses(
    workspace: &mut SfmWorkspace,
    protected_views: &[ViewId],
    min_observations: usize,
    max_pixel_residual: f64,
) -> Result<PoseSupportCleanupReport> {
    if min_observations == 0 {
        return Ok(PoseSupportCleanupReport::default());
    }
    if max_pixel_residual.is_nan() || max_pixel_residual < 0.0 {
        return Err(SfmError::InvalidConfig(
            "maximum pixel residual must be non-negative or +infinity".to_owned(),
        ));
    }

    let view_count = workspace.views().len();
    let mut protected = vec![false; view_count];
    for &image in protected_views {
        let slot = protected
            .get_mut(image.index())
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: image.index(),
            })?;
        *slot = true;
    }

    let mut report = PoseSupportCleanupReport {
        min_observations,
        ..PoseSupportCleanupReport::default()
    };
    loop {
        let support = reprojection_valid_pose_support(workspace, max_pixel_residual)?;
        let mut removed_pose_ids = Vec::new();
        let mut removed_this_round = Vec::new();
        for (pose_index, _) in workspace.pose_store().iter_populated() {
            let pose_id = crate::PoseId::new(pose_index);
            let views = workspace
                .views()
                .iter()
                .filter(|view| view.pose_id == pose_id)
                .map(|view| view.id)
                .collect::<Vec<_>>();
            // The optimized variable is the pose, not an individual view. Multiple views may
            // share it, so their observations jointly constrain the translation.
            let pose_support = support.get(pose_id.index()).copied().unwrap_or(0);
            if !views.is_empty()
                && views.iter().all(|view| !protected[view.index()])
                && pose_support < min_observations
            {
                removed_pose_ids.push(pose_id);
                removed_this_round.extend(views);
            }
        }
        if removed_this_round.is_empty() {
            break;
        }
        removed_this_round.sort_unstable();
        for pose in removed_pose_ids {
            workspace.poses.take(pose.index())?;
        }
        report.rounds += 1;
        report.removed_views.extend(removed_this_round);
        report.rejected_landmarks += retain_posed_landmark_support(workspace)?;
    }

    report.removed_views.sort_unstable();
    report.removed_views.dedup();
    if !report.is_empty() || report.rejected_landmarks > 0 {
        refresh_primary_component(workspace);
    }
    Ok(report)
}

/// Remove poses that remain unobservable even when every other pose is held fixed.
///
/// With the other poses fixed, a landmark they observe at least twice is determined and gives a
/// pose two equations, while a landmark shared with one other pose keeps its depth along that
/// pose's ray and gives one. Landmarks shared only with the same neighbor fix at most the
/// relative pose up to scale, `pose_dof - 1` equations. A pose whose bound stays below
/// `pose_dof` has a null space no global solve can remove, so it is peeled to a fixed point
/// together with the landmarks it leaves under-supported. The peel never leaves fewer than three
/// poses, whose relative scale could otherwise be the gauge itself, and never removes a pose
/// shared by several views, whose rigid internal baseline the bound ignores; both cases are left
/// to the exact solver certificate.
pub(crate) fn remove_underdetermined_poses(
    workspace: &mut SfmWorkspace,
    pose_dof: usize,
) -> Result<PoseSupportCleanupReport> {
    let mut report = PoseSupportCleanupReport {
        min_observations: pose_dof,
        ..PoseSupportCleanupReport::default()
    };
    loop {
        let mut views_by_pose = BTreeMap::<crate::PoseId, Vec<ViewId>>::new();
        for (view, _) in workspace.registered_views() {
            if let Some(pose) = workspace.pose_id(view) {
                views_by_pose.entry(pose).or_default().push(view);
            }
        }
        let mut bound = BTreeMap::<crate::PoseId, usize>::new();
        let mut shared_with_one_neighbour =
            BTreeMap::<(crate::PoseId, crate::PoseId), usize>::new();
        for landmark in workspace.landmarks().iter_active() {
            let track =
                workspace
                    .tracks()
                    .get(landmark.source_track)
                    .ok_or(SfmError::InvalidId {
                        kind: TrackId::kind(),
                        index: landmark.source_track.index(),
                    })?;
            let mut poses = landmark
                .supporting_observations(track)
                .filter_map(|(_, observation)| {
                    workspace
                        .has_pose(observation.view)
                        .then(|| workspace.pose_id(observation.view))
                        .flatten()
                })
                .collect::<Vec<_>>();
            poses.sort_unstable();
            poses.dedup();
            match poses.as_slice() {
                [first, second] => {
                    *shared_with_one_neighbour
                        .entry((*first, *second))
                        .or_default() += 1;
                    *shared_with_one_neighbour
                        .entry((*second, *first))
                        .or_default() += 1;
                }
                [_, _, _, ..] => {
                    for pose in poses {
                        *bound.entry(pose).or_default() += 2;
                    }
                }
                _ => {}
            }
        }
        for ((pose, _), count) in shared_with_one_neighbour {
            *bound.entry(pose).or_default() += count.min(pose_dof.saturating_sub(1));
        }
        let mut removed_pose_ids = Vec::new();
        let mut removed_this_round = Vec::new();
        for (pose, views) in &views_by_pose {
            if views.len() == 1 && bound.get(pose).copied().unwrap_or(0) < pose_dof {
                removed_pose_ids.push(*pose);
                removed_this_round.extend(views.iter().copied());
            }
        }
        if removed_this_round.is_empty() || views_by_pose.len() - removed_pose_ids.len() < 3 {
            break;
        }
        for pose in removed_pose_ids {
            workspace.poses.take(pose.index())?;
        }
        report.rounds += 1;
        report.removed_views.extend(removed_this_round);
        report.rejected_landmarks += retain_posed_landmark_support(workspace)?;
    }

    report.removed_views.sort_unstable();
    if !report.is_empty() || report.rejected_landmarks > 0 {
        refresh_primary_component(workspace);
    }
    Ok(report)
}

/// Keep one vertex-biconnected block of the active pose--landmark incidence graph.
///
/// Connectivity alone is insufficient for fixed-rotation BA: two otherwise rigid incidence
/// subgraphs joined at one pose or one landmark retain an independent scale. Keeping one block
/// removes that structural ambiguity before numerical preflight. Views sharing a pose are modeled
/// as one variable, and repeated observations of one landmark from that pose form one incidence
/// edge. This pass makes no geometric rank claim; the native solver remains
/// responsible for exact observability.
///
/// Free-scale problems retain the block with the most pose variables, then the most landmarks,
/// with stable IDs breaking ties. A fixed-baseline problem retains the block containing both
/// anchor poses and rejects split or missing anchors instead of silently changing the gauge.
pub(crate) fn retain_primary_bundle_component(
    workspace: &mut SfmWorkspace,
) -> Result<PoseSupportCleanupReport> {
    #[derive(Debug, Default)]
    struct IncidenceBlock {
        poses: BTreeSet<crate::PoseId>,
        landmarks: BTreeSet<LandmarkId>,
    }

    #[derive(Debug)]
    struct BlockCandidate {
        poses: Vec<crate::PoseId>,
        images: Vec<ViewId>,
        landmarks: Vec<LandmarkId>,
    }

    let mut views_by_pose = BTreeMap::<crate::PoseId, Vec<ViewId>>::new();
    for (view, _) in workspace.registered_views() {
        let pose = workspace.pose_id(view).ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: view.index(),
        })?;
        views_by_pose.entry(pose).or_default().push(view);
    }
    let pose_ids = views_by_pose.keys().copied().collect::<Vec<_>>();
    let pose_dense = pose_ids
        .iter()
        .enumerate()
        .map(|(dense, &pose)| (pose, dense))
        .collect::<BTreeMap<_, _>>();
    let landmark_ids = workspace
        .landmarks()
        .iter_active()
        .map(|landmark| landmark.id)
        .collect::<Vec<_>>();
    let landmark_dense = landmark_ids
        .iter()
        .enumerate()
        .map(|(dense, &landmark)| (landmark, dense))
        .collect::<BTreeMap<_, _>>();
    let mut endpoints = Vec::new();
    for landmark in workspace.landmarks().iter_active() {
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        let mut supporting_poses = landmark
            .supporting_observations(track)
            .filter_map(|(_, observation)| {
                workspace
                    .has_pose(observation.view)
                    .then(|| workspace.pose_id(observation.view))
                    .flatten()
            })
            .collect::<Vec<_>>();
        supporting_poses.sort_unstable();
        supporting_poses.dedup();
        for pose in supporting_poses {
            endpoints.push((
                pose_dense[&pose],
                pose_ids.len() + landmark_dense[&landmark.id],
            ));
        }
    }

    let topology = openmvg_core::graph::analyze_undirected_topology(
        pose_ids.len() + landmark_ids.len(),
        &endpoints,
    )
    .map_err(|error| {
        SfmError::InvalidInput(format!(
            "active bundle-incidence topology analysis failed: {error}"
        ))
    })?;
    let mut blocks = (0..topology.biconnected_block_count())
        .map(|_| IncidenceBlock::default())
        .collect::<Vec<_>>();
    for (&(pose_node, landmark_node), &block) in
        endpoints.iter().zip(topology.biconnected_block_by_edge())
    {
        blocks[block].poses.insert(pose_ids[pose_node]);
        blocks[block]
            .landmarks
            .insert(landmark_ids[landmark_node - pose_ids.len()]);
    }
    let candidates = blocks
        .into_iter()
        .map(|block| {
            let poses = block.poses.into_iter().collect::<Vec<_>>();
            let mut images = Vec::new();
            for pose in &poses {
                if let Some(views) = views_by_pose.get(pose) {
                    images.extend(views.iter().copied());
                }
            }
            images.sort_unstable();
            BlockCandidate {
                poses,
                images,
                landmarks: block.landmarks.into_iter().collect(),
            }
        })
        .collect::<Vec<_>>();

    let gauge = workspace
        .components()
        .first()
        .map(|component| component.gauge_scale_policy)
        .unwrap_or(GaugeScalePolicy::Free);
    let selected = match gauge {
        GaugeScalePolicy::FixedBaseline {
            first: view_a,
            second: view_b,
            ..
        } => {
            let pose_a = workspace
                .has_pose(view_a)
                .then(|| workspace.pose_id(view_a))
                .flatten();
            let pose_b = workspace
                .has_pose(view_b)
                .then(|| workspace.pose_id(view_b))
                .flatten();
            candidates
                .iter()
                .position(|component| {
                    pose_a.is_some_and(|pose| component.poses.binary_search(&pose).is_ok())
                        && pose_b.is_some_and(|pose| component.poses.binary_search(&pose).is_ok())
                })
                .ok_or_else(|| {
                    SfmError::InvalidInput(
                        "fixed-baseline anchors are missing or split across active bundle-incidence blocks"
                            .to_owned(),
                    )
                })?
        }
        _ => candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.poses
                    .len()
                    .cmp(&right.poses.len())
                    .then_with(|| left.landmarks.len().cmp(&right.landmarks.len()))
                    // `max_by` selects the latter equal element. Reverse stable-ID comparisons so
                    // the lexicographically lowest block wins a tie.
                    .then_with(|| right.images.cmp(&left.images))
                    .then_with(|| right.landmarks.cmp(&left.landmarks))
            })
            .map(|(index, _)| index)
            .ok_or_else(|| {
                SfmError::InvalidInput(
                    "fixed-rotation bundle adjustment has no active pose-landmark incidence block"
                        .to_owned(),
                )
            })?,
    };

    let retained_views = candidates[selected]
        .images
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let retained_landmarks = candidates[selected]
        .landmarks
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut removed_views = Vec::new();
    let mut removed_pose_ids = BTreeSet::new();
    for view in workspace.views().iter() {
        if workspace.has_pose(view.id) && !retained_views.contains(&view.id) {
            removed_views.push(view.id);
            removed_pose_ids.insert(view.pose_id);
        }
    }
    for pose in removed_pose_ids {
        workspace.poses.take(pose.index())?;
    }
    let mut rejected_landmarks = 0;
    for landmark in workspace.landmarks.as_mut_slice() {
        if landmark.status == LandmarkStatus::Active && !retained_landmarks.contains(&landmark.id) {
            landmark.status = LandmarkStatus::Rejected;
            rejected_landmarks += 1;
        }
    }
    removed_views.sort_unstable();
    if !removed_views.is_empty() || rejected_landmarks > 0 {
        refresh_primary_component(workspace);
    }
    Ok(PoseSupportCleanupReport {
        min_observations: 0,
        rounds: usize::from(!removed_views.is_empty()),
        removed_views,
        rejected_landmarks,
    })
}

fn reprojection_valid_pose_support(
    workspace: &SfmWorkspace,
    max_pixel_residual: f64,
) -> Result<Vec<usize>> {
    let mut support = vec![0; workspace.pose_store().slot_count()];
    for landmark in workspace.landmarks().iter_active() {
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        let mut supporting_poses = Vec::new();
        for (_, observation) in landmark.supporting_observations(track) {
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
                    kind: IntrinsicId::kind(),
                    index: image.intrinsic_id.index(),
                })?;
            let camera_point = pose.pose.project_to_camera(&landmark.point);
            let policy = CameraResidualPolicy::from_pixel_limit(camera, max_pixel_residual)?;
            let pixel = workspace.track_observation_pixel(observation)?;
            if policy.accepts(camera, &camera_point.coords, pixel) {
                let pose_id = workspace
                    .pose_id(observation.view)
                    .ok_or(SfmError::InvalidId {
                        kind: ViewId::kind(),
                        index: observation.view.index(),
                    })?;
                supporting_poses.push(pose_id);
            }
        }
        supporting_poses.sort_unstable();
        supporting_poses.dedup();
        for pose_id in supporting_poses {
            support[pose_id.index()] += 1;
        }
    }
    Ok(support)
}

pub(crate) fn retain_posed_landmark_support(workspace: &mut SfmWorkspace) -> Result<usize> {
    let mut updates = Vec::new();
    for landmark in workspace.landmarks().iter_active() {
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        let surviving = landmark
            .supporting_observations(track)
            .filter_map(|(index, observation)| {
                workspace
                    .has_pose(observation.view)
                    .then_some(index.index())
            })
            .collect::<Vec<_>>();
        updates.push((landmark.id, track.observations().len(), surviving));
    }

    let mut rejected = 0;
    for (landmark_id, track_length, surviving) in updates {
        let landmark =
            workspace
                .landmarks
                .get_mut(landmark_id)
                .ok_or(SfmError::InvariantViolation(format!(
                    "active landmark {landmark_id} disappeared during pose audit"
                )))?;
        if surviving.len() < 2 {
            landmark.status = LandmarkStatus::Rejected;
            rejected += 1;
        } else {
            landmark.observation_support =
                ObservationSupport::try_from_indices(surviving, track_length)?;
        }
    }
    Ok(rejected)
}

/// Rebuild the primary component from the currently populated poses and active landmarks.
pub(crate) fn refresh_primary_component(workspace: &mut SfmWorkspace) {
    let gauge = workspace
        .components()
        .first()
        .map(|component| component.gauge_scale_policy)
        .unwrap_or(GaugeScalePolicy::Free);
    let images = workspace.registered_views().map(|(view, _)| view).collect();
    let landmark_ids = workspace
        .landmarks()
        .iter_active()
        .map(|landmark| landmark.id)
        .collect();
    workspace.components = vec![ReconstructionComponent::new(
        ComponentId::new(0),
        images,
        landmark_ids,
        gauge,
    )];
}
