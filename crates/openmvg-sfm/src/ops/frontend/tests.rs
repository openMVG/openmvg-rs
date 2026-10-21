// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Rotation3, Vector3};
use openmvg_core::model::RelativePose;

use super::deactivate_weak_pairs;
use crate::model::{ImageMeta, PairGeometry, ViewRecord};
use crate::{
    Camera, FeatureId, FeatureMatch, IntrinsicId, Keypoint, PoseId, ViewGraph, ViewGraphParts,
    ViewId, ViewPair,
};

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

fn matches() -> Vec<FeatureMatch> {
    (0..FEATURES)
        .map(|feature| {
            FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), Some(0.25))
                .unwrap()
        })
        .collect()
}

// This supplies already accepted verification evidence, not an extraction or solver oracle.
// All cameras have identity rotation and increasing horizontal centers, so every pair has
// the same true baseline direction. Unique keypoints prevent orientation aliases in tracks.
fn accepted_graph() -> ViewGraph {
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
    );
    for first in 0..VIEWS {
        for second in first + 1..VIEWS {
            let pose =
                RelativePose::from_baseline_direction(Rotation3::identity(), Vector3::x_axis())
                    .unwrap();
            let count = retained_inliers(first, second).unwrap_or(8);
            parts = parts.with_pair(
                ViewPair::new(ViewId::new(first), ViewId::new(second)),
                matches(),
                Some((
                    PairGeometry::Essential {
                        matrix: pose.translation_source_to_target().cross_matrix(),
                        relative_pose: Some(pose),
                    },
                    (0..FEATURES).map(|feature| feature < count).collect(),
                )),
            );
        }
    }
    ViewGraph::try_from_parts(parts).unwrap()
}

fn assert_filtered_graph(graph: &ViewGraph) {
    graph.validate().unwrap();
    assert_eq!(graph.root_seed(), 11.into());
    assert_eq!(graph.pair_count(), 15);
    assert_eq!(graph.verified_pair_count(), 3);
    assert_eq!(
        graph
            .pairs()
            .map(|pair| pair.matches().len())
            .sum::<usize>(),
        240
    );
    for pair in graph.pairs() {
        assert_eq!(pair.matches(), matches().as_slice());
        let expected =
            retained_inliers(pair.views().first().index(), pair.views().second().index());
        match (expected, pair.verified()) {
            (Some(count), Some((_, inliers))) => {
                assert_eq!(
                    inliers,
                    (0..FEATURES)
                        .map(|feature| feature < count)
                        .collect::<Vec<_>>()
                        .as_slice()
                );
            }
            (None, None) => {}
            state => panic!("filter acceptance mismatch: {state:?}"),
        }
    }
    assert_eq!(graph.tracks().len(), 45);
    for track in graph.tracks().as_slice() {
        let observations = track.observations();
        assert_eq!(observations.len(), 2);
        let count = retained_inliers(observations[0].view.index(), observations[1].view.index())
            .expect("tracks use only the three independently expected strong pairs");
        assert_eq!(observations[0].feature, observations[1].feature);
        assert!(observations[0].feature.index() < count);
    }
}

#[test]
fn strongest_pairs_filter_preserves_putatives_and_prepared_checkpoint() {
    let graph = accepted_graph();
    let mut workspace = graph.workspace().clone();
    assert_eq!(deactivate_weak_pairs(&mut workspace, 1).unwrap(), 12);
    crate::ops::tracks::build_tracks(&mut workspace).unwrap();
    let filtered = ViewGraph::try_from_workspace(workspace)
        .expect("policy-rejected pairs cannot retain accepted verification evidence");
    assert_filtered_graph(&filtered);

    let bytes = filtered.to_bytes().unwrap();
    let restored = ViewGraph::from_bytes(&bytes).unwrap();
    assert_filtered_graph(&restored);
    assert_eq!(restored.tracks(), filtered.tracks());
    assert_eq!(restored.to_bytes().unwrap(), bytes);
}
