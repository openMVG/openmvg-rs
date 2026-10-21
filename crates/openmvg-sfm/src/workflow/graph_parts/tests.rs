// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Point3, Rotation3, Unit, Vector3};

use super::*;
use crate::model::ImageMeta;
use crate::{FeatureId, IntrinsicId, PoseId, ViewId};

const VIEW_COUNT: usize = 3;
const POINT_COUNT: usize = 24;
const FOCAL: f64 = 500.0;

/// A pair as [`ViewGraphParts::with_pair`] takes it.
type StoredPair = (
    ViewPair,
    Vec<FeatureMatch>,
    Option<(PairGeometry, Vec<bool>)>,
);

fn camera() -> Camera {
    Camera::try_pinhole(FOCAL, FOCAL, 320.0, 240.0).unwrap()
}

/// World-to-camera rotation and camera center of `view`.
fn view_pose(view: usize) -> (Rotation3<f64>, Vector3<f64>) {
    let step = view as f64;
    (
        Rotation3::from_euler_angles(0.02 * step, -0.06 * step, 0.01 * step),
        Vector3::new(0.5 * step, 0.05 * step, -0.1 * step),
    )
}

fn points() -> Vec<Point3<f64>> {
    (0..POINT_COUNT)
        .map(|index| {
            Point3::new(
                -1.0 + 0.3 * (index % 6) as f64,
                -0.6 + 0.4 * (index / 6) as f64,
                5.0 + 0.5 * (index % 5) as f64,
            )
        })
        .collect()
}

fn keypoints(view: usize) -> Vec<Keypoint> {
    let (rotation, center) = view_pose(view);
    points()
        .iter()
        .map(|point| {
            let local = rotation * (point.coords - center);
            Keypoint::try_new(
                (FOCAL * local.x / local.z + 320.0) as f32,
                (FOCAL * local.y / local.z + 240.0) as f32,
                2.0,
                0.0,
            )
            .unwrap()
        })
        .collect()
}

/// True pose of `second` relative to `first`.
fn relative_pose(first: usize, second: usize) -> RelativePose {
    let (rotation_i, center_i) = view_pose(first);
    let (rotation_j, center_j) = view_pose(second);
    RelativePose::from_projection_translation(
        rotation_j * rotation_i.inverse(),
        Unit::new_normalize(rotation_j * (center_i - center_j)),
    )
    .unwrap()
}

fn essential(pose: &RelativePose) -> Matrix3<f64> {
    pose.translation_source_to_target().cross_matrix() * pose.rotation_source_to_target().matrix()
}

fn identity_matches() -> Vec<FeatureMatch> {
    (0..POINT_COUNT)
        .map(|feature| {
            FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), Some(0.25))
                .unwrap()
        })
        .collect()
}

fn pair(first: usize, second: usize) -> ViewPair {
    ViewPair::new(ViewId::new(first), ViewId::new(second))
}

fn view_records() -> Vec<ViewRecord> {
    (0..VIEW_COUNT)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                ImageMeta::external_id(format!("view-{view}"), 640, 480),
                IntrinsicId::new(0),
                PoseId::new(view),
            )
        })
        .collect()
}

fn base_parts() -> ViewGraphParts {
    ViewGraphParts::new(
        11,
        vec![camera()],
        view_records(),
        (0..VIEW_COUNT).map(keypoints).collect(),
    )
}

fn accepted(first: usize, second: usize, with_pose: bool) -> Option<(PairGeometry, Vec<bool>)> {
    let pose = relative_pose(first, second);
    Some((
        PairGeometry::Essential {
            matrix: essential(&pose),
            relative_pose: with_pose.then_some(pose),
        },
        vec![true; POINT_COUNT],
    ))
}

/// Pair (0, 1) stores its pose, pair (0, 2) leaves it to recovery, and pair (1, 2) is rejected.
fn parts() -> ViewGraphParts {
    base_parts()
        .with_pair(pair(0, 1), identity_matches(), accepted(0, 1, true))
        .with_pair(pair(0, 2), identity_matches(), accepted(0, 2, false))
        .with_pair(pair(1, 2), identity_matches(), None)
}

fn error_message(parts: ViewGraphParts) -> String {
    match ViewGraph::try_from_parts(parts) {
        Err(SfmError::InvalidInput(message)) => message,
        other => panic!("expected invalid input, got {other:?}"),
    }
}

fn graph_pairs(graph: &ViewGraph) -> Vec<StoredPair> {
    graph
        .pairs()
        .map(|pair| {
            (
                pair.views(),
                pair.matches().to_vec(),
                pair.verified()
                    .map(|(geometry, inliers)| (geometry, inliers.to_vec())),
            )
        })
        .collect()
}

#[test]
fn parts_rebuild_keypoints_pairs_and_tracks() {
    let graph = ViewGraph::try_from_parts(parts().with_calibrated_intrinsics(true)).unwrap();

    assert_eq!(graph.root_seed(), RunSeed::from(11));
    assert!(graph.has_calibrated_intrinsics());
    assert_eq!(graph.views(), view_records().as_slice());
    for view in 0..VIEW_COUNT {
        let expected = keypoints(view);
        assert_eq!(
            graph.keypoints(ViewId::new(view)),
            Some(expected.as_slice())
        );
        let pixels = graph.features(ViewId::new(view)).unwrap().pixels();
        assert_eq!(pixels.len(), POINT_COUNT);
        assert_eq!(pixels[3].x, f64::from(expected[3].x()));
    }
    assert_eq!(graph.keypoints(ViewId::new(VIEW_COUNT)), None);

    let pairs = graph.pairs();
    assert_eq!(pairs.len(), 3);
    assert_eq!(graph.verified_pair_count(), 2);
    let rejected = graph.pairs().nth(2).unwrap();
    assert_eq!(rejected.views(), pair(1, 2));
    assert_eq!(rejected.matches(), identity_matches().as_slice());
    assert!(rejected.verified().is_none());

    // Tracks join the accepted pairs only: every point is seen by all three views.
    assert_eq!(graph.tracks().len(), POINT_COUNT);
    assert!(
        graph
            .tracks()
            .as_slice()
            .iter()
            .all(|track| track.observations().len() == VIEW_COUNT)
    );
}

#[test]
fn missing_essential_pose_is_recovered_by_cheirality() {
    let graph = ViewGraph::try_from_parts(parts()).unwrap();
    let (geometry, inliers) = graph.pairs().nth(1).unwrap().verified().unwrap();
    let PairGeometry::Essential {
        relative_pose: Some(recovered),
        ..
    } = geometry
    else {
        panic!("expected a recovered essential pose, got {geometry:?}");
    };
    let truth = relative_pose(0, 2);

    assert_eq!(inliers, vec![true; POINT_COUNT].as_slice());
    let rotation_error = (recovered.rotation_source_to_target().matrix()
        - truth.rotation_source_to_target().matrix())
    .norm();
    assert!(rotation_error < 1e-4, "rotation error {rotation_error}");
    let direction = recovered
        .translation_source_to_target()
        .dot(truth.translation_source_to_target());
    assert!(direction > 1.0 - 1e-6, "translation cosine {direction}");
}

#[test]
fn graph_round_trips_through_its_parts() {
    let graph = ViewGraph::try_from_parts(parts()).unwrap();
    let rebuilt_parts = graph_pairs(&graph).into_iter().fold(
        ViewGraphParts::new(
            graph.root_seed(),
            graph.camera_models().to_vec(),
            graph.views().to_vec(),
            (0..VIEW_COUNT)
                .map(|view| graph.keypoints(ViewId::new(view)).unwrap().to_vec())
                .collect(),
        ),
        |parts, (views, matches, verified)| parts.with_pair(views, matches, verified),
    );
    let rebuilt = ViewGraph::try_from_parts(rebuilt_parts).unwrap();

    assert_eq!(graph_pairs(&rebuilt), graph_pairs(&graph));
    assert_eq!(rebuilt.tracks(), graph.tracks());
    assert_eq!(rebuilt.camera_models(), graph.camera_models());
    assert_eq!(rebuilt.root_seed(), graph.root_seed());
}

#[test]
fn unrecoverable_essential_pose_names_the_pair() {
    let pose = relative_pose(0, 2);
    let no_inliers = base_parts().with_pair(
        pair(0, 2),
        identity_matches(),
        Some((
            PairGeometry::Essential {
                matrix: essential(&pose),
                relative_pose: None,
            },
            vec![false; POINT_COUNT],
        )),
    );
    let message = error_message(no_inliers);
    assert!(message.starts_with("pair (0, 2): "), "{message}");
    assert!(message.contains("in front of both cameras"), "{message}");

    let not_essential = base_parts().with_pair(
        pair(0, 2),
        identity_matches(),
        Some((
            PairGeometry::Essential {
                matrix: Matrix3::identity(),
                relative_pose: None,
            },
            vec![true; POINT_COUNT],
        )),
    );
    let message = error_message(not_essential);
    assert!(
        message.starts_with("pair (0, 2): the essential matrix is invalid"),
        "{message}"
    );
}

#[test]
fn pairs_must_be_strictly_increasing_distinct_known_views() {
    let repeated = base_parts()
        .with_pair(pair(0, 1), identity_matches(), None)
        .with_pair(pair(0, 1), identity_matches(), None);
    assert_eq!(
        error_message(repeated),
        "pair (0, 1) is repeated or out of order; pairs must be strictly increasing"
    );

    let unsorted = base_parts()
        .with_pair(pair(0, 2), identity_matches(), None)
        .with_pair(pair(0, 1), identity_matches(), None);
    assert_eq!(
        error_message(unsorted),
        "pair (0, 1) is repeated or out of order; pairs must be strictly increasing"
    );

    let self_pair = base_parts().with_pair(pair(1, 1), Vec::new(), None);
    assert_eq!(
        error_message(self_pair),
        "pair (1, 1) pairs a view with itself"
    );

    let dangling = base_parts().with_pair(pair(0, VIEW_COUNT), Vec::new(), None);
    assert_eq!(
        error_message(dangling),
        format!("pair (0, {VIEW_COUNT}) references an unknown view")
    );
}

#[test]
fn matches_and_inlier_masks_are_checked_per_pair() {
    let mut reversed = identity_matches();
    reversed.swap(0, 1);
    let unsorted = base_parts().with_pair(pair(0, 1), reversed, None);
    assert_eq!(
        error_message(unsorted),
        "pair (0, 1) has matches that are not strictly increasing by feature pair"
    );

    let mut duplicated = identity_matches();
    duplicated[1] = duplicated[0];
    let repeated = base_parts().with_pair(pair(0, 1), duplicated, None);
    assert_eq!(
        error_message(repeated),
        "pair (0, 1) has matches that are not strictly increasing by feature pair"
    );

    let short_mask = base_parts().with_pair(
        pair(0, 1),
        identity_matches(),
        accepted(0, 1, true).map(|(geometry, _)| (geometry, vec![true; 2])),
    );
    assert_eq!(
        error_message(short_mask),
        format!("pair (0, 1) has 2 inlier flags for {POINT_COUNT} matches")
    );

    let unknown_feature = base_parts().with_pair(
        pair(0, 1),
        vec![FeatureMatch::try_new(FeatureId::new(POINT_COUNT), FeatureId::new(0), None).unwrap()],
        None,
    );
    assert_eq!(
        error_message(unknown_feature),
        format!("pair (0, 1): invalid feature id {POINT_COUNT}")
    );
}

#[test]
fn zero_projective_matrices_are_rejected_without_an_absolute_scale_floor() {
    let pose = relative_pose(0, 1);
    for geometry in [
        PairGeometry::Essential {
            matrix: Matrix3::zeros(),
            relative_pose: Some(pose),
        },
        PairGeometry::Fundamental {
            matrix: Matrix3::zeros(),
            relative_pose: Some(pose),
        },
        PairGeometry::Homography {
            matrix: Matrix3::zeros(),
            relative_pose: Some(pose),
        },
    ] {
        let parts = base_parts().with_pair(
            pair(0, 1),
            identity_matches(),
            Some((geometry, vec![true; POINT_COUNT])),
        );
        assert_eq!(
            error_message(parts),
            "pair (0, 1): geometric model contains an all-zero matrix"
        );
    }

    // Projective validity has no absolute magnitude floor when a pose is supplied.
    let parts = base_parts().with_pair(
        pair(0, 1),
        identity_matches(),
        Some((
            PairGeometry::Essential {
                matrix: essential(&pose) * 1e-200,
                relative_pose: Some(pose),
            },
            vec![true; POINT_COUNT],
        )),
    );
    assert!(ViewGraph::try_from_parts(parts).is_ok());
}

#[test]
fn non_finite_geometry_names_the_pair() {
    let mut matrix = Matrix3::identity();
    matrix[(0, 2)] = f64::NAN;
    let parts = base_parts().with_pair(
        pair(0, 1),
        identity_matches(),
        Some((
            PairGeometry::Fundamental {
                matrix,
                relative_pose: None,
            },
            vec![true; POINT_COUNT],
        )),
    );
    assert_eq!(
        error_message(parts),
        "pair (0, 1): geometric model contains non-finite values"
    );
}

#[test]
fn views_must_be_dense_and_carry_keypoints() {
    let mut views = view_records();
    views.swap(0, 1);
    let shuffled = ViewGraphParts::new(
        11,
        vec![camera()],
        views,
        (0..VIEW_COUNT).map(keypoints).collect(),
    );
    assert_eq!(
        error_message(shuffled),
        "view 1 is stored at position 0; view IDs must be dense and in order"
    );

    let missing = ViewGraphParts::new(11, vec![camera()], view_records(), vec![keypoints(0)]);
    assert_eq!(
        error_message(missing),
        format!("got 1 keypoint lists for {VIEW_COUNT} views")
    );
}

#[test]
fn every_pair_keeps_its_matches_and_a_model_exactly_when_active() {
    let graph = ViewGraph::try_from_parts(parts()).unwrap();
    let validate = |workspace: SfmWorkspace| match ViewGraph::try_from_workspace(workspace) {
        Err(SfmError::InvalidInput(message)) => message,
        other => panic!("expected invalid input, got {other:?}"),
    };

    let mut unmatched = graph.workspace().clone();
    unmatched.matches.clear(2).unwrap();
    assert_eq!(validate(unmatched), "pair (1, 2) has no match record");

    let mut unverified = graph.workspace().clone();
    unverified
        .pair_graph_mut()
        .activate_pair_and(PairId::new(2))
        .unwrap();
    assert_eq!(
        validate(unverified),
        "pair (1, 2) is active without a verified model"
    );

    let mut stale = graph.workspace().clone();
    stale
        .pair_graph_mut()
        .deactivate_pair_and(PairId::new(0))
        .unwrap();
    assert_eq!(
        validate(stale),
        "pair (0, 1) keeps a verified model but is inactive"
    );
}
