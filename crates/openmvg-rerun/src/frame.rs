// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_core::camera::CameraModel as _;
use openmvg_sfm::events::{
    SfmEvent, SfmEventBranch, SfmEventKey, SfmEventPhase, SfmEventTransition, SfmSnapshot,
};
use openmvg_sfm::{Camera, ViewId, ViewPair};

use crate::{GeometryCapture, RerunRecorderConfig};

/// Nested views of the verified pair graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphLayer {
    Verified,
    Active,
    RegisteredInduced,
    Frontier,
}

impl GraphLayer {
    pub(crate) const fn entity_name(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Active => "active",
            Self::RegisteredInduced => "registered_induced",
            Self::Frontier => "frontier",
        }
    }

    /// Whether this layer should display a camera before it has a reconstructed pose.
    pub(crate) const fn includes_node(self, registered: bool) -> bool {
        match self {
            Self::RegisteredInduced => registered,
            Self::Verified | Self::Active | Self::Frontier => true,
        }
    }
}

/// One graph node. Rerun owns its 2D auto-layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VisualNode {
    pub(crate) view: ViewId,
    pub(crate) registered: bool,
    /// Active reconstructed landmarks observed by this image.
    pub(crate) active_landmark_count: usize,
}

/// One stable pair-graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VisualEdge {
    pub(crate) first: ViewId,
    pub(crate) second: ViewId,
}

/// One deterministically sampled graph layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisualGraphLayer {
    pub(crate) layer: GraphLayer,
    pub(crate) edges: Vec<VisualEdge>,
}

/// Graph nodes and configured, deterministically sampled edge layers for one event.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisualGraph {
    pub(crate) nodes: Vec<VisualNode>,
    pub(crate) layers: Vec<VisualGraphLayer>,
    /// Largest source-track incidence count for any image.
    pub(crate) max_tracks_per_image: usize,
}

impl VisualGraph {
    pub(crate) fn layer(&self, layer: GraphLayer) -> Option<&VisualGraphLayer> {
        self.layers
            .iter()
            .find(|candidate| candidate.layer == layer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VisualCameraKind {
    Pinhole,
    BrownConrady,
    Fisheye,
    Spherical,
}

impl VisualCameraKind {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Pinhole => "pinhole",
            Self::BrownConrady => "brown_conrady",
            Self::Fisheye => "fisheye",
            Self::Spherical => "spherical",
        }
    }

    pub(crate) const fn color(self) -> u32 {
        match self {
            Self::Pinhole => 0x43A047FF,
            Self::BrownConrady => 0xFB8C00FF,
            Self::Fisheye => 0x8E24AAFF,
            Self::Spherical => 0x00ACC1FF,
        }
    }
}

/// A reconstructed camera center and its three camera-frame axes in recentered world space.
///
/// Camera-frame axes avoid implying a perspective frustum for fisheye or spherical models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VisualCamera {
    pub(crate) view: ViewId,
    pub(crate) kind: VisualCameraKind,
    pub(crate) center: [f32; 3],
    pub(crate) axis_endpoints: [[f32; 3]; 3],
}

/// One active registered view-graph edge embedded in reconstruction space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VisualRegisteredEdge {
    pub(crate) first: ViewId,
    pub(crate) second: ViewId,
    pub(crate) endpoints: [[f32; 3]; 2],
    /// Active reconstructed landmarks observed by both endpoint cameras.
    pub(crate) shared_active_landmark_count: usize,
}

/// Bounded 3D reconstruction data.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisualGeometry {
    /// Add this f64 offset to logged f32 positions to recover reconstruction coordinates.
    pub(crate) world_origin: [f64; 3],
    pub(crate) cameras: Vec<VisualCamera>,
    /// Sampled active landmarks. `None` preserves the previous Rerun point-cloud sample.
    pub(crate) landmarks: Option<Vec<[f32; 3]>>,
    pub(crate) registered_edges: Vec<VisualRegisteredEdge>,
    /// Largest source-track co-observation count for any verified pair.
    pub(crate) max_shared_tracks_per_verified_pair: usize,
    pub(crate) skipped_camera_count: usize,
    pub(crate) skipped_non_finite_landmarks: usize,
}

/// Cheap counts emitted for every observed event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VisualMetrics {
    pub(crate) view_count: usize,
    pub(crate) pair_count: usize,
    pub(crate) source_track_count: usize,
    pub(crate) verified_edge_count: usize,
    pub(crate) active_edge_count: usize,
    pub(crate) registered_induced_edge_count: usize,
    pub(crate) frontier_edge_count: usize,
    pub(crate) registered_view_count: usize,
    pub(crate) active_landmark_count: usize,
}

/// Camera-model-aware quality for one sampled geometry frame.
///
/// When there are no valid observations, all residual summaries are zero. Consumers must use
/// `valid_observation_count` to distinguish that empty state from a perfect reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VisualQuality {
    pub(crate) bearing_rmse_rad: f64,
    pub(crate) bearing_p90_rad: f64,
    pub(crate) reprojection_rmse_px: f64,
    pub(crate) valid_observation_count: usize,
    pub(crate) invalid_observation_count: usize,
    pub(crate) unevaluable_landmark_count: usize,
}

/// Pose agreement on verified edges.
///
/// Rotation includes rotation-only constraints; translation requires a full relative pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VisualPoseGraphQuality {
    pub(crate) active_rotation_p90_deg: Option<f64>,
    pub(crate) active_translation_p90_deg: Option<f64>,
    pub(crate) all_verified_rotation_p90_deg: Option<f64>,
}

/// Owned, deterministic payload consumed by the Rerun adapter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisualFrame {
    pub(crate) sequence: u64,
    pub(crate) branch: SfmEventBranch,
    pub(crate) phase: SfmEventPhase,
    pub(crate) transition: SfmEventTransition,
    pub(crate) metrics: VisualMetrics,
    pub(crate) quality: Option<VisualQuality>,
    pub(crate) pose_graph_quality: Option<VisualPoseGraphQuality>,
    pub(crate) graph: Option<VisualGraph>,
    pub(crate) geometry: Option<VisualGeometry>,
}

impl VisualFrame {
    pub(crate) fn extract(
        event: SfmEvent<'_>,
        config: &RerunRecorderConfig,
        world_origin: [f64; 3],
        incidence_ceilings: SourceTrackIncidenceCeilings,
    ) -> Self {
        let snapshot = event.snapshot;
        let metrics = metrics(snapshot);
        let capture_topology = should_capture_topology(&event, config);
        let capture_detail = capture_topology && should_capture_detail(&event, config);
        let (quality, pose_graph_quality, graph, geometry) = if capture_topology {
            let incidence = extract_active_landmark_incidence(snapshot);
            (
                capture_detail.then(|| extract_quality(snapshot)),
                Some(extract_pose_graph_quality(snapshot)),
                Some(extract_graph(
                    snapshot,
                    config,
                    metrics,
                    &incidence,
                    incidence_ceilings,
                )),
                Some(extract_geometry(
                    snapshot,
                    config,
                    world_origin,
                    &incidence,
                    incidence_ceilings,
                    capture_detail,
                )),
            )
        } else {
            (None, None, None, None)
        };
        Self {
            sequence: event.sequence,
            branch: event.branch,
            phase: event.phase,
            transition: event.transition,
            metrics,
            quality,
            pose_graph_quality,
            graph,
            geometry,
        }
    }

    #[must_use]
    pub(crate) const fn has_geometry(&self) -> bool {
        self.geometry.is_some()
    }

    #[must_use]
    pub(crate) const fn has_detail(&self) -> bool {
        self.quality.is_some()
    }
}

fn capture_stride(event: &SfmEvent<'_>, config: &RerunRecorderConfig) -> Option<u64> {
    let GeometryCapture::Every(stride) = config.geometry else {
        return None;
    };
    if event.transition == SfmEventTransition::Tentative && !config.include_tentative_geometry {
        return None;
    }
    Some(stride.get())
}

fn should_capture_topology(event: &SfmEvent<'_>, config: &RerunRecorderConfig) -> bool {
    let Some(stride) = capture_stride(event, config) else {
        return false;
    };
    is_detail_boundary(event)
        || matches!(event.phase, SfmEventPhase::IncrementalRegistration { .. })
        || event.sequence.is_multiple_of(stride)
}

fn should_capture_detail(event: &SfmEvent<'_>, config: &RerunRecorderConfig) -> bool {
    let Some(stride) = capture_stride(event, config) else {
        return false;
    };
    is_detail_boundary(event) || event.sequence.is_multiple_of(stride)
}

fn is_detail_boundary(event: &SfmEvent<'_>) -> bool {
    matches!(
        event.transition,
        SfmEventTransition::Started
            | SfmEventTransition::RolledBack
            | SfmEventTransition::Discarded
            | SfmEventTransition::Selected
            | SfmEventTransition::Completed
    ) || matches!(
        event.phase.key(),
        SfmEventKey::Graph | SfmEventKey::GlobalCandidate | SfmEventKey::BundleAdjustment
    ) || matches!(
        event.phase,
        SfmEventPhase::IncrementalInitialPair | SfmEventPhase::IncrementalFinal
    )
}

fn metrics(workspace: SfmSnapshot<'_>) -> VisualMetrics {
    let mut verified_edge_count = 0;
    let mut active_edge_count = 0;
    let mut registered_induced_edge_count = 0;
    let mut frontier_edge_count = 0;
    for pair in workspace.pairs() {
        if !workspace.is_pair_verified(*pair) {
            continue;
        }
        verified_edge_count += 1;
        if !workspace.is_pair_active(*pair) {
            continue;
        }
        active_edge_count += 1;
        let posed_a = workspace.has_pose(pair.first());
        let posed_b = workspace.has_pose(pair.second());
        if posed_a && posed_b {
            registered_induced_edge_count += 1;
        } else if posed_a != posed_b {
            frontier_edge_count += 1;
        }
    }
    VisualMetrics {
        view_count: workspace.views().len(),
        pair_count: workspace.pairs().len(),
        source_track_count: workspace.tracks().len(),
        verified_edge_count,
        active_edge_count,
        registered_induced_edge_count,
        frontier_edge_count,
        registered_view_count: workspace.registered_view_count(),
        active_landmark_count: workspace.active_landmark_count(),
    }
}

fn extract_quality(workspace: SfmSnapshot<'_>) -> VisualQuality {
    let mut residuals = Vec::new();
    let mut pixel_squared_error_sum = 0.0;
    let mut invalid_observation_count = 0usize;
    let mut unevaluable_landmark_count = 0usize;

    for landmark in workspace.active_landmarks() {
        let Some(track) = workspace.tracks().get(landmark.source_track()) else {
            unevaluable_landmark_count += 1;
            continue;
        };
        let mut support_count = 0usize;
        for (_, observation) in landmark.supporting_observations(track) {
            support_count += 1;
            let Some(pose) = workspace.pose(observation.view) else {
                invalid_observation_count += 1;
                continue;
            };
            let Some(view) = workspace.view(observation.view) else {
                invalid_observation_count += 1;
                continue;
            };
            let Some(camera) = workspace.camera_model(view.intrinsic_id) else {
                invalid_observation_count += 1;
                continue;
            };
            let camera_point = pose.project_to_camera(landmark.point());
            let Ok(projected) = camera.project(&camera_point.coords) else {
                invalid_observation_count += 1;
                continue;
            };
            let Ok(pixel) = workspace.track_observation_pixel(observation) else {
                invalid_observation_count += 1;
                continue;
            };
            let Ok(residual) = camera.bearing_error_rad(&camera_point.coords, &pixel) else {
                invalid_observation_count += 1;
                continue;
            };
            let pixel_residual =
                camera_pixel_residual(camera, [projected.x, projected.y], [pixel.x, pixel.y]);
            let pixel_squared_error = pixel_residual * pixel_residual;
            if !residual.is_finite()
                || !pixel_residual.is_finite()
                || !pixel_squared_error.is_finite()
            {
                invalid_observation_count += 1;
                continue;
            }
            residuals.push(residual);
            pixel_squared_error_sum += pixel_squared_error;
        }
        if support_count == 0 {
            unevaluable_landmark_count += 1;
        }
    }

    residuals.sort_by(f64::total_cmp);
    let valid_observation_count = residuals.len();
    let bearing_rmse_rad = if residuals.is_empty() {
        0.0
    } else {
        (residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>()
            / residuals.len() as f64)
            .sqrt()
    };
    VisualQuality {
        bearing_rmse_rad,
        bearing_p90_rad: sorted_upper_quantile(&residuals, 0.9),
        reprojection_rmse_px: if valid_observation_count == 0 {
            0.0
        } else {
            (pixel_squared_error_sum / valid_observation_count as f64).sqrt()
        },
        valid_observation_count,
        invalid_observation_count,
        unevaluable_landmark_count,
    }
}

/// Image-space Euclidean residual; spherical longitude wraps at the seam.
fn camera_pixel_residual(camera: &Camera, projected: [f64; 2], observed: [f64; 2]) -> f64 {
    if !projected.iter().all(|value| value.is_finite())
        || !observed.iter().all(|value| value.is_finite())
    {
        return f64::INFINITY;
    }
    let mut dx = (projected[0] - observed[0]).abs();
    if let Camera::Spherical(spherical) = camera {
        let width = spherical.width() as f64;
        dx %= width;
        dx = dx.min(width - dx);
    }
    dx.hypot(projected[1] - observed[1])
}

fn extract_pose_graph_quality(workspace: SfmSnapshot<'_>) -> VisualPoseGraphQuality {
    let mut active_rotation = Vec::new();
    let mut active_translation = Vec::new();
    let mut all_verified_rotation = Vec::new();
    for pair in workspace.pairs() {
        let Some(relative_rotation) = workspace.pair_relative_rotation(*pair) else {
            continue;
        };
        let (Some(pose_i), Some(pose_j)) =
            (workspace.pose(pair.first()), workspace.pose(pair.second()))
        else {
            continue;
        };
        let active = workspace.is_pair_active(*pair);
        let predicted = pose_j.rotation() * pose_i.rotation().inverse();
        let error = (relative_rotation.inverse() * predicted)
            .angle()
            .to_degrees();
        if error.is_finite() {
            all_verified_rotation.push(error);
            if active {
                active_rotation.push(error);
            }
        }
        if !active {
            continue;
        }
        let Some(relative_pose) = workspace.pair_relative_pose(*pair) else {
            continue;
        };
        let predicted = pose_j.center() - pose_i.center();
        let measured =
            pose_i.rotation().inverse() * relative_pose.baseline_direction_in_source().as_ref();
        if predicted.norm_squared() <= 1e-20 || measured.norm_squared() <= 1e-20 {
            continue;
        }
        let error = predicted
            .normalize()
            .dot(&measured.normalize())
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if error.is_finite() {
            active_translation.push(error);
        }
    }
    VisualPoseGraphQuality {
        active_rotation_p90_deg: interpolated_p90(&mut active_rotation),
        active_translation_p90_deg: interpolated_p90(&mut active_translation),
        all_verified_rotation_p90_deg: interpolated_p90(&mut all_verified_rotation),
    }
}

/// Deterministic upper order statistic used for the bearing-error tail.
fn sorted_upper_quantile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let index =
        (quantile.clamp(0.0, 1.0) * sorted_values.len().saturating_sub(1) as f64).ceil() as usize;
    sorted_values[index]
}

fn interpolated_p90(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let position = 0.9 * values.len().saturating_sub(1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(values[lower] * (1.0 - fraction) + values[upper] * fraction)
}

fn extract_graph(
    workspace: SfmSnapshot<'_>,
    config: &RerunRecorderConfig,
    counts: VisualMetrics,
    incidence: &ActiveLandmarkIncidence,
    incidence_ceilings: SourceTrackIncidenceCeilings,
) -> VisualGraph {
    let nodes = extract_nodes(workspace, incidence);
    let requested = [
        (
            GraphLayer::Verified,
            config.graph_layers.verified,
            counts.verified_edge_count,
        ),
        (
            GraphLayer::Active,
            config.graph_layers.active,
            counts.active_edge_count,
        ),
        (
            GraphLayer::RegisteredInduced,
            config.graph_layers.registered_induced,
            counts.registered_induced_edge_count,
        ),
        (
            GraphLayer::Frontier,
            config.graph_layers.frontier,
            counts.frontier_edge_count,
        ),
    ];
    let mut layers = requested
        .into_iter()
        .filter(|(_, enabled, _)| *enabled)
        .map(|(layer, _, total_edges)| VisualGraphLayer {
            layer,
            edges: Vec::with_capacity(total_edges.min(config.max_edges_per_layer)),
        })
        .collect::<Vec<_>>();
    let mut ordinals = [0usize; 4];
    for pair in workspace.pairs() {
        let verified = workspace.is_pair_verified(*pair);
        if !verified {
            continue;
        }
        let active = workspace.is_pair_active(*pair);
        let posed_a = workspace.has_pose(pair.first());
        let posed_b = workspace.has_pose(pair.second());
        let membership = [
            verified,
            active,
            active && posed_a && posed_b,
            active && (posed_a != posed_b),
        ];
        let totals = [
            counts.verified_edge_count,
            counts.active_edge_count,
            counts.registered_induced_edge_count,
            counts.frontier_edge_count,
        ];
        let layer_kinds = [
            GraphLayer::Verified,
            GraphLayer::Active,
            GraphLayer::RegisteredInduced,
            GraphLayer::Frontier,
        ];
        for (layer_index, ((layer_kind, belongs), total_edges)) in layer_kinds
            .into_iter()
            .zip(membership)
            .zip(totals)
            .enumerate()
        {
            if !belongs {
                continue;
            }
            let ordinal = ordinals[layer_index];
            ordinals[layer_index] += 1;
            if selected_sample(ordinal, total_edges, config.max_edges_per_layer)
                && let Some(layer) = layers.iter_mut().find(|layer| layer.layer == layer_kind)
            {
                layer.edges.push(VisualEdge {
                    first: pair.first(),
                    second: pair.second(),
                });
            }
        }
    }
    VisualGraph {
        nodes,
        layers,
        max_tracks_per_image: incidence_ceilings.max_tracks_per_image,
    }
}

fn extract_nodes(
    workspace: SfmSnapshot<'_>,
    incidence: &ActiveLandmarkIncidence,
) -> Vec<VisualNode> {
    workspace
        .views()
        .iter()
        .map(|view| VisualNode {
            view: view.id,
            registered: workspace.has_pose(view.id),
            active_landmark_count: incidence.images[view.id.index()],
        })
        .collect()
}

struct ActiveLandmarkIncidence {
    images: Vec<usize>,
    pairs: Vec<usize>,
}

/// Immutable color-scale ceilings derived from source tracks, images, and verified pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceTrackIncidenceCeilings {
    view_count: usize,
    pair_count: usize,
    track_count: usize,
    verified_pair_count: usize,
    max_tracks_per_image: usize,
    max_shared_tracks_per_verified_pair: usize,
}

impl SourceTrackIncidenceCeilings {
    pub(crate) fn extract(workspace: SfmSnapshot<'_>) -> Self {
        let mut camera_counts = vec![0usize; workspace.views().len()];
        let mut pair_counts = vec![0usize; workspace.pairs().len()];
        let mut supporting_images = Vec::new();
        for track in workspace.tracks().as_slice() {
            supporting_images.clear();
            supporting_images.extend(
                track
                    .observations()
                    .iter()
                    .map(|observation| observation.view),
            );
            accumulate_camera_pair_support(
                workspace,
                &supporting_images,
                &mut camera_counts,
                &mut pair_counts,
            );
        }
        Self {
            view_count: workspace.views().len(),
            pair_count: workspace.pairs().len(),
            track_count: workspace.tracks().len(),
            verified_pair_count: workspace.verified_pair_count(),
            max_tracks_per_image: camera_counts.into_iter().max().unwrap_or(0),
            max_shared_tracks_per_verified_pair: pair_counts
                .into_iter()
                .enumerate()
                .filter(|(index, _)| workspace.is_pair_verified(workspace.pairs()[*index]))
                .map(|(_, count)| count)
                .max()
                .unwrap_or(0),
        }
    }

    pub(crate) fn matches_shape(self, workspace: SfmSnapshot<'_>) -> bool {
        self.view_count == workspace.views().len()
            && self.pair_count == workspace.pairs().len()
            && self.track_count == workspace.tracks().len()
            && self.verified_pair_count == workspace.verified_pair_count()
    }
}

/// Count active reconstructed landmarks once per observing image and image pair.
///
/// This is covisibility evidence, not a confidence score: a highly supported edge can still be
/// geometrically inconsistent. The exact landmark support mask is used so rejected observations
/// do not inflate the visualization.
fn extract_active_landmark_incidence(workspace: SfmSnapshot<'_>) -> ActiveLandmarkIncidence {
    let mut incidence = ActiveLandmarkIncidence {
        images: vec![0; workspace.views().len()],
        pairs: vec![0; workspace.pairs().len()],
    };
    let mut supporting_images = Vec::new();
    for landmark in workspace.active_landmarks() {
        let Some(track) = workspace.tracks().get(landmark.source_track()) else {
            continue;
        };
        supporting_images.clear();
        supporting_images.extend(
            landmark
                .supporting_observations(track)
                .map(|(_, observation)| observation.view),
        );
        accumulate_camera_pair_support(
            workspace,
            &supporting_images,
            &mut incidence.images,
            &mut incidence.pairs,
        );
    }
    incidence
}

fn accumulate_camera_pair_support(
    workspace: SfmSnapshot<'_>,
    ordered_images: &[ViewId],
    camera_counts: &mut [usize],
    pair_counts: &mut [usize],
) {
    debug_assert!(
        ordered_images
            .windows(2)
            .all(|images| images[0] < images[1])
    );
    for view in ordered_images {
        if let Some(count) = camera_counts.get_mut(view.index()) {
            *count = count.saturating_add(1);
        }
    }
    for (offset, view_a) in ordered_images.iter().enumerate() {
        for view_b in &ordered_images[offset + 1..] {
            let pair = ViewPair::new(*view_a, *view_b);
            if let Ok(index) = workspace.pairs().binary_search(&pair) {
                pair_counts[index] = pair_counts[index].saturating_add(1);
            }
        }
    }
}

fn extract_geometry(
    workspace: SfmSnapshot<'_>,
    config: &RerunRecorderConfig,
    world_origin: [f64; 3],
    incidence: &ActiveLandmarkIncidence,
    incidence_ceilings: SourceTrackIncidenceCeilings,
    capture_detail: bool,
) -> VisualGeometry {
    let mut centers = vec![None; workspace.views().len()];
    let mut cameras = Vec::with_capacity(workspace.registered_view_count());
    let mut skipped_camera_count = 0;
    for (view, pose) in workspace.registered_views() {
        let center = pose.center();
        let matrix = pose.rotation().matrix();
        let Some(center) = finite_recentered_point3(center.x, center.y, center.z, world_origin)
        else {
            skipped_camera_count += 1;
            continue;
        };
        let image_record = &workspace.views()[view.index()];
        let kind = match workspace.camera_model(image_record.intrinsic_id) {
            Some(Camera::Pinhole(_)) => VisualCameraKind::Pinhole,
            Some(Camera::BrownConrady(_)) => VisualCameraKind::BrownConrady,
            Some(Camera::Fisheye(_)) => VisualCameraKind::Fisheye,
            Some(Camera::Spherical(_)) => VisualCameraKind::Spherical,
            None => {
                skipped_camera_count += 1;
                continue;
            }
        };
        let endpoints = [
            finite_axis_endpoint(
                center,
                [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
                config.camera_axis_length,
            ),
            finite_axis_endpoint(
                center,
                [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
                config.camera_axis_length,
            ),
            finite_axis_endpoint(
                center,
                [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
                config.camera_axis_length,
            ),
        ];
        let [Some(x), Some(y), Some(z)] = endpoints else {
            skipped_camera_count += 1;
            continue;
        };
        centers[view.index()] = Some(center);
        cameras.push(VisualCamera {
            view,
            kind,
            center,
            axis_endpoints: [x, y, z],
        });
    }

    let active_landmark_count = workspace.active_landmark_count();
    let mut landmarks =
        capture_detail.then(|| Vec::with_capacity(active_landmark_count.min(config.max_landmarks)));
    let mut skipped_non_finite_landmarks = 0;
    if let Some(landmarks) = &mut landmarks {
        for (ordinal, landmark) in workspace.active_landmarks().enumerate() {
            let point = landmark.point();
            let Some(position) = finite_recentered_point3(point.x, point.y, point.z, world_origin)
            else {
                skipped_non_finite_landmarks += 1;
                continue;
            };
            if selected_sample(ordinal, active_landmark_count, config.max_landmarks) {
                landmarks.push(position);
            }
        }
    }

    let finite_registered_edge_count = workspace
        .pairs()
        .iter()
        .filter(|pair| {
            workspace.is_pair_verified(**pair)
                && workspace.is_pair_active(**pair)
                && centers[pair.first().index()].is_some()
                && centers[pair.second().index()].is_some()
        })
        .count();
    let mut registered_edges =
        Vec::with_capacity(finite_registered_edge_count.min(config.max_edges_per_layer));
    let mut ordinal = 0usize;
    for (index, pair) in workspace.pairs().iter().enumerate() {
        if !workspace.is_pair_verified(*pair) || !workspace.is_pair_active(*pair) {
            continue;
        }
        let (Some(center_a), Some(center_b)) = (
            centers[pair.first().index()],
            centers[pair.second().index()],
        ) else {
            continue;
        };
        if selected_sample(
            ordinal,
            finite_registered_edge_count,
            config.max_edges_per_layer,
        ) {
            registered_edges.push(VisualRegisteredEdge {
                first: pair.first(),
                second: pair.second(),
                endpoints: [center_a, center_b],
                shared_active_landmark_count: incidence.pairs[index],
            });
        }
        ordinal += 1;
    }

    VisualGeometry {
        world_origin,
        cameras,
        landmarks,
        registered_edges,
        max_shared_tracks_per_verified_pair: incidence_ceilings.max_shared_tracks_per_verified_pair,
        skipped_camera_count,
        skipped_non_finite_landmarks,
    }
}

fn finite_axis_endpoint(center: [f32; 3], axis: [f64; 3], length: f64) -> Option<[f32; 3]> {
    finite_point3(
        f64::from(center[0]) + axis[0] * length,
        f64::from(center[1]) + axis[1] * length,
        f64::from(center[2]) + axis[2] * length,
    )
}

fn finite_point3(x: f64, y: f64, z: f64) -> Option<[f32; 3]> {
    let point = [x as f32, y as f32, z as f32];
    (x.is_finite()
        && y.is_finite()
        && z.is_finite()
        && point.iter().all(|component| component.is_finite()))
    .then_some(point)
}

fn finite_recentered_point3(x: f64, y: f64, z: f64, origin: [f64; 3]) -> Option<[f32; 3]> {
    finite_point3(x - origin[0], y - origin[1], z - origin[2])
}

pub(crate) fn first_finite_camera_center(workspace: SfmSnapshot<'_>) -> Option<[f64; 3]> {
    workspace.registered_views().find_map(|(_, pose)| {
        let center = pose.center();
        [center.x, center.y, center.z]
            .iter()
            .all(|value| value.is_finite())
            .then_some([center.x, center.y, center.z])
    })
}

fn selected_sample(ordinal: usize, total: usize, limit: usize) -> bool {
    if limit == 0 || total == 0 {
        return false;
    }
    if total <= limit {
        return true;
    }
    let before = (ordinal as u128 * limit as u128) / total as u128;
    let after = ((ordinal as u128 + 1) * limit as u128) / total as u128;
    after > before
}

#[cfg(test)]
mod tests;
