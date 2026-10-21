// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Rotation3, Vector3};
use openmvg_core::model::RelativePose;

use super::*;
use crate::mapping::columns::Rows;

const VIEWS: usize = 6;
const FEATURES: usize = 16;

fn retained_inliers(first: usize, second: usize) -> Option<usize> {
    match (first, second) {
        (0, 1) => Some(16),
        (2, 3) => Some(15),
        (4, 5) => Some(14),
        _ => None,
    }
}

fn identity_matches() -> Vec<FeatureMatch> {
    (0..FEATURES)
        .map(|feature| {
            FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), Some(0.25))
                .unwrap()
        })
        .collect()
}

// The expected public shape after strongest-pairs rejection: all putatives survive, and
// only the three accepted pairs carry geometry. No SfM private operation or IO cycle needed.
fn filtered_graph() -> ViewGraph {
    let views = (0..VIEWS)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                ImageMeta::external_id(view.to_string(), 640, 480),
                IntrinsicId::new(0),
                PoseId::new(view),
            )
        })
        .collect();
    let keypoints = (0..VIEWS)
        .map(|view| {
            (0..FEATURES)
                .map(|feature| {
                    let x = -0.6 + 0.4 * (feature % 4) as f64;
                    let y = -0.6 + 0.4 * (feature / 4) as f64;
                    let z = 5.0 + 0.1 * (feature % 3) as f64;
                    Keypoint::try_new(
                        (100.0 * (x - 0.1 * view as f64) / z + 320.0) as f32,
                        (100.0 * y / z + 240.0) as f32,
                        2.0,
                        0.0,
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect();
    let mut parts = ViewGraphParts::new(
        11,
        vec![Camera::try_pinhole(100.0, 100.0, 320.0, 240.0).unwrap()],
        views,
        keypoints,
    )
    .with_calibrated_intrinsics(true);
    for first in 0..VIEWS {
        for second in first + 1..VIEWS {
            let verified = retained_inliers(first, second).map(|count| {
                let pose =
                    RelativePose::from_baseline_direction(Rotation3::identity(), Vector3::x_axis())
                        .unwrap();
                (
                    PairGeometry::Essential {
                        matrix: pose.translation_source_to_target().cross_matrix(),
                        relative_pose: Some(pose),
                    },
                    (0..FEATURES).map(|feature| feature < count).collect(),
                )
            });
            parts = parts.with_pair(pair(first, second), identity_matches(), verified);
        }
    }
    ViewGraph::try_from_parts(parts).unwrap()
}

#[test]
fn filtered_pairs_round_trip_without_reaccepting_putative_matches() {
    let graph = filtered_graph();
    assert_eq!(graph.pair_count(), 15);
    assert_eq!(graph.verified_pair_count(), 3);
    assert_eq!(graph.tracks().len(), 45);
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("filtered.omvg");
    let write_options = WriteOptions::new();
    let read_options = ReadOptions::new().with_root_seed(graph.root_seed());
    write_view_graph(&graph, &path, &write_options).unwrap();
    assert!(check(&path).is_empty());

    let loaded = load(&path);
    let pairs = Rows::new(&loaded.tables["pairs"]);
    let raw_matches = Rows::new(&loaded.tables["matches"]);
    assert_eq!(pairs.len(), 15);
    assert_eq!(raw_matches.len(), 240);
    for row in 0..pairs.len() {
        let first = usize::try_from(pairs.int("view_first", row).unwrap()).unwrap();
        let second = usize::try_from(pairs.int("view_second", row).unwrap()).unwrap();
        assert_eq!(
            pairs.string("model", row),
            retained_inliers(first, second).map(|_| "essential")
        );
    }
    for row in 0..raw_matches.len() {
        let first = usize::try_from(raw_matches.int("view_first", row).unwrap()).unwrap();
        let second = usize::try_from(raw_matches.int("view_second", row).unwrap()).unwrap();
        let feature = usize::try_from(raw_matches.int("feature_first", row).unwrap()).unwrap();
        let accepted = retained_inliers(first, second).is_some_and(|count| feature < count);
        assert_eq!(raw_matches.boolean("inlier", row), Some(accepted));
        assert_eq!(raw_matches.float("distance", row), Some(0.25));
    }

    let restored = read_view_graph(&path, &read_options).unwrap();
    assert_eq!(graph_pairs(&restored), graph_pairs(&graph));
    assert_eq!(all_keypoints(&restored), all_keypoints(&graph));
    assert_eq!(restored.root_seed(), graph.root_seed());
    assert_eq!(
        restored.has_calibrated_intrinsics(),
        graph.has_calibrated_intrinsics()
    );
    assert_eq!(
        restored
            .pairs()
            .map(|pair| pair.matches().len())
            .sum::<usize>(),
        240
    );
    assert_eq!(restored.verified_pair_count(), 3);
    assert_eq!(restored.tracks(), graph.tracks());
    let again = scratch.path().join("again.omvg");
    write_view_graph(&restored, &again, &write_options).unwrap();
    assert_eq!(
        fs::read(again.join("manifest.json")).unwrap(),
        fs::read(path.join("manifest.json")).unwrap()
    );
}
