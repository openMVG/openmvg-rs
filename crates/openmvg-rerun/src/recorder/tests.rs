// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::frame::{VisualGraphLayer, VisualNode};
use openmvg_sfm::events::GlobalCandidate;
use openmvg_sfm::events::{SfmEventBranch, SfmEventPhase, SfmEventTransition, SfmSnapshot};
use openmvg_sfm::{SfmEngine, ViewId, unstable::SfmWorkspace};

fn node(index: usize, registered: bool) -> VisualNode {
    VisualNode {
        view: ViewId::new(index),
        registered,
        active_landmark_count: 0,
    }
}

fn graph(nodes: Vec<VisualNode>, verified_edges: &[(usize, usize)]) -> VisualGraph {
    VisualGraph {
        nodes,
        layers: vec![VisualGraphLayer {
            layer: GraphLayer::Verified,
            edges: verified_edges
                .iter()
                .map(|&(first, second)| VisualEdge {
                    first: ViewId::new(first),
                    second: ViewId::new(second),
                })
                .collect(),
        }],
        max_tracks_per_image: 0,
    }
}

#[test]
fn metric_lanes_are_stable_per_branch() {
    assert_eq!(metric_lane(SfmEventBranch::Main), "_branches/main");
    assert_eq!(
        metric_lane(SfmEventBranch::GlobalPreparation),
        "_branches/global_preparation"
    );
    assert_eq!(
        metric_lane(SfmEventBranch::GlobalCandidate {
            attempt: 7,
            candidate: GlobalCandidate::GlobalPositioning,
        }),
        "_branches/global_candidate_0007_global_positioning"
    );
}

#[test]
fn incidence_colormap_has_stable_endpoints_and_distinguishes_dense_counts() {
    assert_eq!(incidence_color(0, 5_123), 0x6B7280FF);
    assert_eq!(incidence_color(1, 5_123), 0x440154FF);
    assert_eq!(incidence_color(5_123, 5_123), 0xFDE725FF);
    assert_eq!(incidence_color(usize::MAX, 5_123), 0xFDE725FF);
    assert_ne!(incidence_color(500, 5_123), incidence_color(1_700, 5_123));
}

#[test]
fn initial_graph_topology_is_captured_once() {
    let mut logged = false;
    let without_verified = VisualGraph {
        nodes: vec![node(0, false), node(1, false)],
        layers: Vec::new(),
        max_tracks_per_image: 0,
    };
    assert!(take_initial_topology(&mut logged, &without_verified).is_none());
    assert!(!logged);

    let first = graph(
        vec![node(0, false), node(1, false), node(2, false)],
        &[(0, 1), (1, 2)],
    );
    assert_eq!(
        take_initial_topology(&mut logged, &first).unwrap().edges,
        first.layers[0].edges
    );
    assert!(logged);

    let changed = graph(vec![node(0, true), node(1, true), node(2, true)], &[(0, 2)]);
    assert!(take_initial_topology(&mut logged, &changed).is_none());
}

#[test]
fn node_statuses_follow_event_and_branch_semantics() {
    assert_eq!(
        visual_node_status(SfmEventBranch::Main, SfmEventTransition::Committed, false,),
        VisualNodeStatus::Unused
    );
    assert_eq!(
        visual_node_status(SfmEventBranch::Main, SfmEventTransition::Tentative, true,),
        VisualNodeStatus::Tentative
    );
    let candidate = SfmEventBranch::GlobalCandidate {
        attempt: 0,
        candidate: GlobalCandidate::GlobalPositioning,
    };
    assert_eq!(
        visual_node_status(candidate, SfmEventTransition::Checkpoint, true,),
        VisualNodeStatus::Tentative
    );
    assert_eq!(
        visual_node_status(candidate, SfmEventTransition::Discarded, true),
        VisualNodeStatus::Tentative
    );
    assert_eq!(
        visual_node_status(candidate, SfmEventTransition::Selected, true),
        VisualNodeStatus::Committed
    );
    assert_eq!(
        visual_node_status(SfmEventBranch::Main, SfmEventTransition::Committed, true,),
        VisualNodeStatus::Committed
    );
}

#[test]
fn status_colors_are_distinct_and_stable() {
    assert_eq!(VisualNodeStatus::Unused.color(), 0x6B7280FF);
    assert_eq!(VisualNodeStatus::Tentative.color(), 0xF59E0BFF);
    assert_eq!(VisualNodeStatus::Committed.color(), 0x22C55EFF);
}

#[test]
fn empty_geometry_logging_is_supported() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("empty.rrd");
    let stream = RecordingStreamBuilder::new("openmvg_rerun_test")
        .save(&path)
        .unwrap();
    stream
        .log("/empty/points", &Points3D::clear_fields())
        .unwrap();
    stream
        .log("/empty/lines", &LineStrips3D::clear_fields())
        .unwrap();
    stream.flush_blocking().unwrap();
    stream.finalize_deferred_sinks();
    assert!(std::fs::metadata(path).unwrap().len() > 0);
}

#[test]
fn recorder_writes_and_finalizes_one_pipeline_event() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("pipeline.rrd");
    let workspace = SfmWorkspace::builder().build().unwrap();
    let mut recorder = RerunRecorder::save(&path, RerunRecorderConfig::default()).unwrap();
    recorder.record(SfmEvent::new(
        0,
        SfmEventBranch::Main,
        SfmEventPhase::Run {
            engine: SfmEngine::Global,
        },
        SfmEventTransition::Completed,
        SfmSnapshot::from_workspace(&workspace),
    ));
    let report = recorder.finish();
    assert_eq!(report.events_seen, 1);
    assert_eq!(report.events_written, 1);
    assert_eq!(report.geometry_frames_written, 1);
    assert_eq!(report.detail_frames_written, 1);
    assert_eq!(report.metrics_only_events_written, 0);
    assert_eq!(report.first_error, None);
    assert!(std::fs::metadata(path).unwrap().len() > 0);
}

#[test]
fn disabled_recorder_ignores_direct_record_calls() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("disabled.rrd");
    let workspace = SfmWorkspace::builder().build().unwrap();
    let mut recorder = RerunRecorder::save(&path, RerunRecorderConfig::default()).unwrap();
    recorder.disabled = true;

    recorder.record(SfmEvent::new(
        0,
        SfmEventBranch::Main,
        SfmEventPhase::Run {
            engine: SfmEngine::Incremental,
        },
        SfmEventTransition::Started,
        SfmSnapshot::from_workspace(&workspace),
    ));

    assert_eq!(recorder.report().events_seen, 0);
    assert_eq!(recorder.report().events_written, 0);
}

#[test]
fn recorder_counts_lightweight_topology_frames_separately() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("topology_only.rrd");
    let workspace = SfmWorkspace::builder().build().unwrap();
    let config = RerunRecorderConfig {
        geometry: crate::GeometryCapture::Every(std::num::NonZeroU64::new(8).unwrap()),
        ..RerunRecorderConfig::default()
    };
    let mut recorder = RerunRecorder::save(&path, config).unwrap();

    recorder.record(SfmEvent::new(
        1,
        SfmEventBranch::Main,
        SfmEventPhase::IncrementalRegistration { index: 0 },
        SfmEventTransition::Committed,
        SfmSnapshot::from_workspace(&workspace),
    ));

    assert_eq!(recorder.report().geometry_frames_written, 1);
    assert_eq!(recorder.report().detail_frames_written, 0);
    assert_eq!(recorder.report().metrics_only_events_written, 0);
}
