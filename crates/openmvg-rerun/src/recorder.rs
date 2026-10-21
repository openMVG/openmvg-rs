// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use openmvg_sfm::events::{GlobalCandidate, SfmEvent, SfmEventBranch, SfmEventSink};
use re_sdk::log::ChunkBatcherConfig;
use re_sdk::{AsComponents, RecordingStream, RecordingStreamBuilder};
use re_sdk_types::archetypes::{
    GraphEdges, GraphNodes, LineStrips3D, Points3D, Scalars, TextLog, ViewCoordinates,
};
use re_sdk_types::components::TextLogLevel;

use crate::{
    RerunRecorderConfig,
    blueprint::default_blueprint,
    frame::{
        GraphLayer, SourceTrackIncidenceCeilings, VisualEdge, VisualFrame, VisualGeometry,
        VisualGraph, first_finite_camera_center,
    },
};

const CURRENT_ROOT: &str = "/sfm/current";
pub(crate) const INITIAL_GRAPH_ROOT: &str = "/sfm/initial_view_graph";
pub(crate) const CURRENT_GRAPH_ROOT: &str = "/sfm/current/graph";
pub(crate) const CURRENT_WORLD_ROOT: &str = "/sfm/current/world";
const CAMERA_CENTERS_PATH: &str = "/sfm/current/world/cameras/centers";
const CAMERA_AXES_PATH: &str = "/sfm/current/world/cameras/axes";
const ACTIVE_LANDMARKS_PATH: &str = "/sfm/current/world/landmarks/active";
const REGISTERED_EDGES_PATH: &str = "/sfm/current/world/view_edges/registered_induced";
pub(crate) const LANDMARK_GROWTH_ROOT: &str = "/sfm/landmark_growth";
pub(crate) const COVERAGE_ROOT: &str = "/sfm/coverage";
pub(crate) const BEARING_QUALITY_ROOT: &str = "/sfm/bearing_error";
pub(crate) const PIXEL_QUALITY_ROOT: &str = "/sfm/image_reprojection";
pub(crate) const POSE_GRAPH_QUALITY_ROOT: &str = "/sfm/pose_graph_quality";
pub(crate) const EVENT_PATH: &str = "/sfm/events";
const DIAGNOSTICS_ROOT: &str = "/sfm/diagnostics";
const RECENTER_ORIGIN_ROOT: &str = "/sfm/metadata/recenter_origin";

/// Failure while configuring or starting a Rerun recording.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum RerunRecorderError {
    /// The requested recorder policy violates its documented invariants.
    #[error("invalid Rerun recorder configuration: {0}")]
    InvalidConfig(&'static str),
    /// The recording stream or its static metadata could not be initialized.
    #[error("failed to initialize Rerun recording: {0}")]
    Initialization(String),
}

/// Final accounting for an observer whose errors never affect SfM routing.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RerunFinishReport {
    /// Events delivered before the sink disabled itself after its first logging error.
    pub events_seen: u64,
    /// Events accepted by the Rerun stream before final flushing.
    pub events_written: u64,
    /// Written events that included graph and 3D geometry.
    pub geometry_frames_written: u64,
    /// Geometry frames that also refreshed landmarks and residual summaries.
    pub detail_frames_written: u64,
    /// Written events containing only metrics and event telemetry.
    pub metrics_only_events_written: u64,
    /// First observer-only failure, after which recording was disabled.
    pub first_error: Option<String>,
}

/// Synchronous `.rrd` sink with lossless, backpressured transport for [`SfmEvent`] values.
///
/// Metrics and event messages are emitted for every delivered event; geometry is sampled by the
/// configured [`crate::GeometryCapture`] policy.
pub struct RerunRecorder {
    stream: RecordingStream,
    config: RerunRecorderConfig,
    report: RerunFinishReport,
    world_origin: Option<[f64; 3]>,
    track_incidence_ceilings: Option<SourceTrackIncidenceCeilings>,
    initial_graph_logged: bool,
    disabled: bool,
}

impl RerunRecorder {
    /// Creates an `.rrd` recorder using backpressured SDK transport.
    ///
    /// # Errors
    ///
    /// Returns [`RerunRecorderError`] when the policy is invalid, the output cannot be created, or
    /// static recording metadata cannot be written.
    pub fn save(
        path: impl AsRef<Path>,
        config: RerunRecorderConfig,
    ) -> Result<Self, RerunRecorderError> {
        config
            .validate()
            .map_err(RerunRecorderError::InvalidConfig)?;
        let mut batcher = ChunkBatcherConfig::DEFAULT;
        batcher.max_bytes_in_flight = config.max_bytes_in_flight;
        let stream = RecordingStreamBuilder::new("openmvg_sfm")
            .recording_name("OpenMVG SfM")
            .with_blueprint(default_blueprint())
            .batcher_config(batcher)
            .save(path.as_ref())
            .map_err(|error| RerunRecorderError::Initialization(error.to_string()))?;
        stream
            .log_static(CURRENT_ROOT, &ViewCoordinates::RDF())
            .map_err(|error| RerunRecorderError::Initialization(error.to_string()))?;
        Ok(Self {
            stream,
            config,
            report: RerunFinishReport::default(),
            world_origin: None,
            track_incidence_ceilings: None,
            initial_graph_logged: false,
            disabled: false,
        })
    }

    /// Flush and finalize the recording, returning observer-only failures as telemetry.
    ///
    /// Call this method rather than relying on drop when the output must be complete before the
    /// process exits.
    #[must_use]
    pub fn finish(mut self) -> RerunFinishReport {
        if let Err(error) = self.stream.flush_with_timeout(self.config.flush_timeout) {
            self.fail(error.to_string());
        }
        self.stream.finalize_deferred_sinks();
        self.report
    }

    /// Returns the live accounting state without flushing the recording.
    #[must_use]
    pub const fn report(&self) -> &RerunFinishReport {
        &self.report
    }

    fn write_frame(&mut self, frame: &VisualFrame) -> re_sdk::RecordingStreamResult<()> {
        self.stream
            .set_time_sequence("sfm_step", frame.sequence as i64);
        let metric_lane = metric_lane(frame.branch);
        log_count(
            &self.stream,
            LANDMARK_GROWTH_ROOT,
            &metric_lane,
            "active_landmarks",
            frame.metrics.active_landmark_count,
        )?;
        log_count(
            &self.stream,
            LANDMARK_GROWTH_ROOT,
            &metric_lane,
            "source_tracks_upper_bound",
            frame.metrics.source_track_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/images",
            frame.metrics.view_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/pairs",
            frame.metrics.pair_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/edges/verified",
            frame.metrics.verified_edge_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/edges/active",
            frame.metrics.active_edge_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/edges/registered_induced",
            frame.metrics.registered_induced_edge_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/edges/frontier",
            frame.metrics.frontier_edge_count,
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "counts/registered_cameras",
            frame.metrics.registered_view_count,
        )?;
        if frame.metrics.view_count == 0 {
            clear_scalar(
                &self.stream,
                COVERAGE_ROOT,
                &metric_lane,
                "registered_camera_fraction",
            )?;
        } else {
            log_scalar(
                &self.stream,
                COVERAGE_ROOT,
                &metric_lane,
                "registered_camera_fraction",
                frame.metrics.registered_view_count as f64 / frame.metrics.view_count as f64,
            )?;
        }
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "geometry_captured",
            usize::from(frame.has_geometry()),
        )?;
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            &metric_lane,
            "detail_captured",
            usize::from(frame.has_detail()),
        )?;
        if let Some(quality) = &frame.quality {
            log_count(
                &self.stream,
                DIAGNOSTICS_ROOT,
                &metric_lane,
                "quality_evidence/observations/valid",
                quality.valid_observation_count,
            )?;
            log_count(
                &self.stream,
                DIAGNOSTICS_ROOT,
                &metric_lane,
                "quality_evidence/observations/invalid",
                quality.invalid_observation_count,
            )?;
            log_count(
                &self.stream,
                DIAGNOSTICS_ROOT,
                &metric_lane,
                "quality_evidence/landmarks/unevaluable",
                quality.unevaluable_landmark_count,
            )?;
            if quality.valid_observation_count == 0 {
                clear_scalar(
                    &self.stream,
                    BEARING_QUALITY_ROOT,
                    &metric_lane,
                    "bearing_rmse_deg",
                )?;
                clear_scalar(
                    &self.stream,
                    BEARING_QUALITY_ROOT,
                    &metric_lane,
                    "bearing_p90_deg",
                )?;
                clear_scalar(
                    &self.stream,
                    PIXEL_QUALITY_ROOT,
                    &metric_lane,
                    "reprojection_rmse_px",
                )?;
            } else {
                log_scalar(
                    &self.stream,
                    BEARING_QUALITY_ROOT,
                    &metric_lane,
                    "bearing_rmse_deg",
                    quality.bearing_rmse_rad.to_degrees(),
                )?;
                log_scalar(
                    &self.stream,
                    BEARING_QUALITY_ROOT,
                    &metric_lane,
                    "bearing_p90_deg",
                    quality.bearing_p90_rad.to_degrees(),
                )?;
                log_scalar(
                    &self.stream,
                    PIXEL_QUALITY_ROOT,
                    &metric_lane,
                    "reprojection_rmse_px",
                    quality.reprojection_rmse_px,
                )?;
            }
        }
        if let Some(quality) = &frame.pose_graph_quality {
            log_optional_scalar(
                &self.stream,
                POSE_GRAPH_QUALITY_ROOT,
                &metric_lane,
                "active_rotation_p90_deg",
                quality.active_rotation_p90_deg,
            )?;
            log_optional_scalar(
                &self.stream,
                POSE_GRAPH_QUALITY_ROOT,
                &metric_lane,
                "active_translation_p90_deg",
                quality.active_translation_p90_deg,
            )?;
            log_optional_scalar(
                &self.stream,
                POSE_GRAPH_QUALITY_ROOT,
                &metric_lane,
                "all_verified_rotation_p90_deg",
                quality.all_verified_rotation_p90_deg,
            )?;
        }
        self.stream.log(
            EVENT_PATH,
            &TextLog::new(format!(
                "branch={:?} phase={:?} transition={:?}",
                frame.branch, frame.phase, frame.transition
            ))
            .with_level(TextLogLevel::INFO),
        )?;
        if let (Some(graph), Some(geometry)) = (&frame.graph, &frame.geometry) {
            log_world_origin_component(&self.stream, "x", geometry.world_origin[0])?;
            log_world_origin_component(&self.stream, "y", geometry.world_origin[1])?;
            log_world_origin_component(&self.stream, "z", geometry.world_origin[2])?;
            log_count(
                &self.stream,
                DIAGNOSTICS_ROOT,
                &metric_lane,
                "source_track_incidence/max_tracks_per_image",
                graph.max_tracks_per_image,
            )?;
            log_count(
                &self.stream,
                DIAGNOSTICS_ROOT,
                &metric_lane,
                "source_track_incidence/max_shared_tracks_per_verified_pair",
                geometry.max_shared_tracks_per_verified_pair,
            )?;
            self.log_graph(frame.branch, frame.transition, graph)?;
            self.log_geometry(&metric_lane, geometry)?;
        }
        Ok(())
    }

    fn log_graph(
        &mut self,
        branch: SfmEventBranch,
        transition: openmvg_sfm::events::SfmEventTransition,
        graph: &VisualGraph,
    ) -> re_sdk::RecordingStreamResult<()> {
        let ids = graph
            .nodes
            .iter()
            .map(|node| image_name(node.view))
            .collect::<Vec<_>>();
        if let Some(verified) = take_initial_topology(&mut self.initial_graph_logged, graph) {
            let nodes = GraphNodes::new(ids.iter().map(String::as_str)).with_show_labels(false);
            let edges = graph_edges(&verified.edges, &ids);
            self.stream.log_static(
                INITIAL_GRAPH_ROOT,
                &[&nodes as &dyn AsComponents, &edges as &dyn AsComponents],
            )?;
        }

        if self.initial_graph_logged {
            let nodes =
                GraphNodes::update_fields()
                    .with_colors(graph.nodes.iter().map(|node| {
                        visual_node_status(branch, transition, node.registered).color()
                    }))
                    .with_labels(graph.nodes.iter().map(|node| {
                        let status = visual_node_status(branch, transition, node.registered);
                        format!(
                            "{} · {} · {} active landmarks",
                            ids[node.view.index()],
                            status.label(),
                            node.active_landmark_count
                        )
                    }));
            self.stream.log(INITIAL_GRAPH_ROOT, &nodes)?;
        }

        for layer in &graph.layers {
            if layer.layer == GraphLayer::Verified {
                continue;
            }
            let visible_nodes = || {
                graph
                    .nodes
                    .iter()
                    .filter(|node| layer.layer.includes_node(node.registered))
            };
            let path = format!("{CURRENT_GRAPH_ROOT}/{}", layer.layer.entity_name());
            if visible_nodes().next().is_none() {
                self.stream.log(
                    path,
                    &[
                        &GraphNodes::clear_fields() as &dyn AsComponents,
                        &GraphEdges::clear_fields() as &dyn AsComponents,
                    ],
                )?;
                continue;
            }
            let nodes =
                GraphNodes::new(visible_nodes().map(|node| ids[node.view.index()].as_str()))
                    .with_colors(visible_nodes().map(|node| {
                        incidence_color(node.active_landmark_count, graph.max_tracks_per_image)
                    }))
                    .with_labels(visible_nodes().map(|node| {
                        let state = if node.registered {
                            "registered"
                        } else {
                            "unregistered"
                        };
                        format!(
                            "{} · {} active landmarks · {state}",
                            ids[node.view.index()],
                            node.active_landmark_count
                        )
                    }))
                    .with_show_labels(false);
            let edges = graph_edges(&layer.edges, &ids);
            self.stream.log(
                path,
                &[&nodes as &dyn AsComponents, &edges as &dyn AsComponents],
            )?;
        }
        Ok(())
    }

    fn log_geometry(
        &self,
        metric_lane: &str,
        geometry: &VisualGeometry,
    ) -> re_sdk::RecordingStreamResult<()> {
        let centers = geometry.cameras.iter().map(|camera| camera.center);
        let labels = geometry
            .cameras
            .iter()
            .map(|camera| format!("{} ({})", camera.view, camera.kind.label()));
        let colors = geometry.cameras.iter().map(|camera| camera.kind.color());
        if geometry.cameras.is_empty() {
            self.stream
                .log(CAMERA_CENTERS_PATH, &Points3D::clear_fields())?;
            self.stream
                .log(CAMERA_AXES_PATH, &LineStrips3D::clear_fields())?;
        } else {
            self.stream.log(
                CAMERA_CENTERS_PATH,
                &Points3D::new(centers)
                    .with_colors(colors)
                    .with_labels(labels)
                    .with_show_labels(false),
            )?;
            let axes = geometry.cameras.iter().flat_map(|camera| {
                camera
                    .axis_endpoints
                    .map(|endpoint| [camera.center, endpoint])
            });
            let axis_colors =
                (0..geometry.cameras.len()).flat_map(|_| [0xE53935FF, 0x43A047FF, 0x1E88E5FF]);
            self.stream.log(
                CAMERA_AXES_PATH,
                &LineStrips3D::new(axes).with_colors(axis_colors),
            )?;
        }

        if let Some(landmarks) = &geometry.landmarks {
            if landmarks.is_empty() {
                self.stream
                    .log(ACTIVE_LANDMARKS_PATH, &Points3D::clear_fields())?;
            } else {
                self.stream.log(
                    ACTIVE_LANDMARKS_PATH,
                    &Points3D::new(landmarks.iter().copied()).with_colors([0xECEFF1FF]),
                )?;
            }
        }

        if geometry.registered_edges.is_empty() {
            self.stream
                .log(REGISTERED_EDGES_PATH, &LineStrips3D::clear_fields())?;
        } else {
            self.stream.log(
                REGISTERED_EDGES_PATH,
                &LineStrips3D::new(geometry.registered_edges.iter().map(|edge| edge.endpoints))
                    .with_colors(geometry.registered_edges.iter().map(|edge| {
                        incidence_color(
                            edge.shared_active_landmark_count,
                            geometry.max_shared_tracks_per_verified_pair,
                        )
                    }))
                    .with_labels(geometry.registered_edges.iter().map(|edge| {
                        format!(
                            "{} ↔ {} · {} shared active landmarks",
                            edge.first, edge.second, edge.shared_active_landmark_count
                        )
                    }))
                    .with_show_labels(false),
            )?;
        }
        log_count(
            &self.stream,
            DIAGNOSTICS_ROOT,
            metric_lane,
            "quality_evidence/skipped/cameras",
            geometry.skipped_camera_count,
        )?;
        if geometry.landmarks.is_some() {
            log_count(
                &self.stream,
                DIAGNOSTICS_ROOT,
                metric_lane,
                "quality_evidence/skipped_non_finite/landmarks",
                geometry.skipped_non_finite_landmarks,
            )?;
        }
        Ok(())
    }

    fn fail(&mut self, error: String) {
        if self.report.first_error.is_none() {
            self.report.first_error = Some(error);
        }
        self.disabled = true;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualNodeStatus {
    Unused,
    Tentative,
    Committed,
}

impl VisualNodeStatus {
    const fn color(self) -> u32 {
        match self {
            Self::Unused => 0x6B7280FF,
            Self::Tentative => 0xF59E0BFF,
            Self::Committed => 0x22C55EFF,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Unused => "unused",
            Self::Tentative => "tentative",
            Self::Committed => "committed",
        }
    }
}

fn visual_node_status(
    branch: SfmEventBranch,
    transition: openmvg_sfm::events::SfmEventTransition,
    registered: bool,
) -> VisualNodeStatus {
    if !registered {
        return VisualNodeStatus::Unused;
    }
    let unselected_candidate = matches!(branch, SfmEventBranch::GlobalCandidate { .. })
        && transition != openmvg_sfm::events::SfmEventTransition::Selected;
    if matches!(
        transition,
        openmvg_sfm::events::SfmEventTransition::Tentative
            | openmvg_sfm::events::SfmEventTransition::Discarded
    ) || unselected_candidate
    {
        VisualNodeStatus::Tentative
    } else {
        VisualNodeStatus::Committed
    }
}

fn take_initial_topology<'a>(
    initial_graph_logged: &mut bool,
    graph: &'a VisualGraph,
) -> Option<&'a crate::frame::VisualGraphLayer> {
    if *initial_graph_logged {
        return None;
    }
    let verified = graph.layer(GraphLayer::Verified)?;
    *initial_graph_logged = true;
    Some(verified)
}

fn graph_edges<'a>(edges: &'a [VisualEdge], ids: &'a [String]) -> GraphEdges {
    GraphEdges::new(edges.iter().map(|edge| {
        (
            ids[edge.first.index()].as_str(),
            ids[edge.second.index()].as_str(),
        )
    }))
    .with_undirected_edges()
}

impl SfmEventSink for RerunRecorder {
    fn enabled(&self, _key: openmvg_sfm::events::SfmEventKey) -> bool {
        !self.disabled
    }

    fn record(&mut self, event: SfmEvent<'_>) {
        if self.disabled {
            return;
        }
        self.report.events_seen = self.report.events_seen.saturating_add(1);
        if self.world_origin.is_none() {
            self.world_origin = first_finite_camera_center(event.snapshot);
        }
        // Production SfM keeps image, pair, and source-track topology immutable. The shape check
        // also makes the sink robust when an integration emits an early pre-topology event.
        let incidence_ceilings = match self.track_incidence_ceilings {
            Some(cached) if cached.matches_shape(event.snapshot) => cached,
            _ => {
                let extracted = SourceTrackIncidenceCeilings::extract(event.snapshot);
                self.track_incidence_ceilings = Some(extracted);
                extracted
            }
        };
        let mut frame_config = self.config.clone();
        if self.initial_graph_logged {
            frame_config.graph_layers.verified = false;
        }
        let frame = VisualFrame::extract(
            event,
            &frame_config,
            self.world_origin.unwrap_or([0.0; 3]),
            incidence_ceilings,
        );
        let has_geometry = frame.has_geometry();
        let has_detail = frame.has_detail();
        match self.write_frame(&frame) {
            Ok(()) => {
                self.report.events_written = self.report.events_written.saturating_add(1);
                if has_geometry {
                    self.report.geometry_frames_written =
                        self.report.geometry_frames_written.saturating_add(1);
                } else {
                    self.report.metrics_only_events_written =
                        self.report.metrics_only_events_written.saturating_add(1);
                }
                if has_detail {
                    self.report.detail_frames_written =
                        self.report.detail_frames_written.saturating_add(1);
                }
            }
            Err(error) => self.fail(error.to_string()),
        }
    }
}

fn log_count(
    stream: &RecordingStream,
    root: &str,
    metric_lane: &str,
    name: &str,
    value: usize,
) -> re_sdk::RecordingStreamResult<()> {
    log_scalar(stream, root, metric_lane, name, value as f64)
}

fn log_scalar(
    stream: &RecordingStream,
    root: &str,
    metric_lane: &str,
    name: &str,
    value: f64,
) -> re_sdk::RecordingStreamResult<()> {
    stream.log(
        format!("{root}/{metric_lane}/{name}"),
        &Scalars::new([value]),
    )
}

fn clear_scalar(
    stream: &RecordingStream,
    root: &str,
    metric_lane: &str,
    name: &str,
) -> re_sdk::RecordingStreamResult<()> {
    stream.log(
        format!("{root}/{metric_lane}/{name}"),
        &Scalars::clear_fields(),
    )
}

fn log_optional_scalar(
    stream: &RecordingStream,
    root: &str,
    metric_lane: &str,
    name: &str,
    value: Option<f64>,
) -> re_sdk::RecordingStreamResult<()> {
    if let Some(value) = value {
        log_scalar(stream, root, metric_lane, name, value)
    } else {
        clear_scalar(stream, root, metric_lane, name)
    }
}

fn metric_lane(branch: SfmEventBranch) -> String {
    let branch = match branch {
        SfmEventBranch::Main => "main".to_owned(),
        SfmEventBranch::GlobalPreparation => "global_preparation".to_owned(),
        SfmEventBranch::GlobalCandidate { attempt, candidate } => format!(
            "global_candidate_{attempt:04}_{}",
            candidate_entity_name(candidate)
        ),
        _ => "unknown".to_owned(),
    };
    format!("_branches/{branch}")
}

const fn candidate_entity_name(candidate: GlobalCandidate) -> &'static str {
    match candidate {
        GlobalCandidate::TranslationAveraging => "translation_averaging",
        GlobalCandidate::GlobalPositioning => "global_positioning",
        _ => "unknown",
    }
}

fn log_world_origin_component(
    stream: &RecordingStream,
    name: &str,
    value: f64,
) -> re_sdk::RecordingStreamResult<()> {
    stream.log(
        format!("{RECENTER_ORIGIN_ROOT}/{name}"),
        &Scalars::new([value]),
    )
}

fn image_name(view: openmvg_sfm::ViewId) -> String {
    format!("image:{:08}", view.index())
}

/// Stable colorblind-safe Viridis mapping for incidence counts.
///
/// Zero incidence is neutral gray. Positive incidence uses a logarithmic scale from one to the
/// immutable source-track ceiling, keeping colors stable as the reconstruction grows.
fn incidence_color(count: usize, maximum_available_count: usize) -> u32 {
    if count == 0 {
        return 0x6B7280FF;
    }
    const STOPS: [[u8; 3]; 5] = [
        [0x44, 0x01, 0x54],
        [0x3B, 0x52, 0x8B],
        [0x21, 0x91, 0x8C],
        [0x5E, 0xC9, 0x62],
        [0xFD, 0xE7, 0x25],
    ];
    let ceiling = maximum_available_count.max(count).max(2);
    let normalized = ((count as f64).log2() / (ceiling as f64).log2()).clamp(0.0, 1.0);
    let position = normalized * (STOPS.len() - 1) as f64;
    let lower = (position.floor() as usize).min(STOPS.len() - 2);
    let fraction = position - lower as f64;
    let channel = |index| {
        let start = f64::from(STOPS[lower][index]);
        let end = f64::from(STOPS[lower + 1][index]);
        (start + (end - start) * fraction).round() as u32
    };
    (channel(0) << 24) | (channel(1) << 16) | (channel(2) << 8) | 0xFF
}

#[cfg(test)]
mod tests;
