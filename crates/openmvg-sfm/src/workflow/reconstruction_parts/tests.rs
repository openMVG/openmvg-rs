// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Point2, Point3, Rotation3};

use super::*;
use crate::model::{ImageMeta, Observation};
use crate::{IntrinsicId, PoseId, ViewId};

const FOCAL: f64 = 500.0;

fn camera() -> Camera {
    Camera::try_pinhole(FOCAL, FOCAL, 320.0, 240.0).unwrap()
}

fn pose(view: usize) -> WorldToCameraPose {
    WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.0, -0.05 * view as f64, 0.0),
        Point3::new(0.5 * view as f64, 0.0, 0.0),
    )
    .unwrap()
}

/// The pixel at which `view` observes `point`.
fn project(view: usize, point: &Point3<f64>) -> Point2<f64> {
    camera()
        .project(&pose(view).project_to_camera(point).coords)
        .unwrap()
}

/// Three views on their own pose blocks, the last unregistered.
fn parts() -> ReconstructionParts {
    let views = (0..3)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                ImageMeta::external_id(format!("{view}"), 640, 480),
                IntrinsicId::new(0),
                PoseId::new(view),
            )
        })
        .collect();
    ReconstructionParts::new(7_u64, SfmEngine::Global, vec![camera()], views)
        .with_pose(PoseId::new(0), pose(0))
        .with_pose(PoseId::new(1), pose(1))
}

fn observed(point: Point3<f64>, views: &[usize]) -> Vec<(ViewId, Point2<f64>)> {
    views
        .iter()
        .map(|&view| (ViewId::new(view), project(view, &point)))
        .collect()
}

fn error_of(parts: ReconstructionParts) -> String {
    Reconstruction::try_from_parts(parts)
        .unwrap_err()
        .to_string()
}

#[test]
fn parts_rebuild_the_points_their_observations_and_the_run_metadata() {
    let near = Point3::new(0.2, -0.1, 4.0);
    let far = Point3::new(-0.3, 0.2, 6.0);
    // Listed out of view order: the track sorts the observations, the feature numbering keeps
    // the order points were added in.
    let far_observations = observed(far, &[1, 0]);
    let reconstruction = Reconstruction::try_from_parts(
        parts()
            .with_point(near, observed(near, &[0, 1]))
            .with_point(far, far_observations)
            .with_calibrated_intrinsics(true),
    )
    .unwrap();

    assert_eq!(reconstruction.engine(), SfmEngine::Global);
    assert_eq!(reconstruction.root_seed(), RunSeed::from(7_u64));
    assert!(reconstruction.has_calibrated_intrinsics());
    assert_eq!(reconstruction.registered_view_count(), 2);
    assert_eq!(reconstruction.pose(ViewId::new(1)).unwrap().pose, pose(1));
    assert!(reconstruction.pose(ViewId::new(2)).is_none());
    assert!(reconstruction.components().is_empty());
    assert_eq!(reconstruction.point_count(), 2);
    let landmarks = reconstruction.landmarks().as_slice();
    assert_eq!(landmarks[1].point, far);
    assert!(landmarks[0].reprojection_error.value() < 1e-9);

    let observation =
        |landmark: usize, view: usize, feature: usize, point: &Point3<f64>| Observation {
            landmark: LandmarkId::new(landmark),
            view: ViewId::new(view),
            feature: FeatureId::new(feature),
            pixel: project(view, point),
        };
    assert_eq!(
        reconstruction.observations().collect::<Vec<_>>(),
        [
            observation(0, 0, 0, &near),
            observation(0, 1, 0, &near),
            observation(1, 0, 1, &far),
            observation(1, 1, 1, &far),
        ]
    );
}

#[test]
fn reprojection_error_is_the_mean_pixel_distance() {
    let point = Point3::new(0.0, 0.0, 5.0);
    let mut observations = observed(point, &[0, 1]);
    observations[1].1 += nalgebra::Vector2::new(3.0, 4.0);
    let reconstruction =
        Reconstruction::try_from_parts(parts().with_point(point, observations)).unwrap();
    let error = reconstruction.landmarks().as_slice()[0]
        .reprojection_error
        .value();
    assert!((error - 2.5).abs() < 1e-9, "{error}");
}

#[test]
fn invalid_points_are_rejected_by_index() {
    let point = Point3::new(0.0, 0.0, 5.0);
    let valid = observed(point, &[0, 1]);
    let with = |observations| {
        parts()
            .with_point(point, valid.clone())
            .with_point(point, observations)
    };

    let cases = [
        (
            with(observed(point, &[0])),
            "point 1 has fewer than two observations",
        ),
        (
            with(observed(point, &[1, 1])),
            "point 1 observes a view twice",
        ),
        (
            with(vec![valid[0], (ViewId::new(3), Point2::origin())]),
            "point 1 observes the unknown view",
        ),
        (
            with(vec![valid[0], (ViewId::new(1), Point2::new(f64::NAN, 0.0))]),
            "point 1 has a non-finite pixel in view",
        ),
        (with(observed(point, &[0, 2])), "unregistered view"),
        (
            parts().with_point(Point3::new(f64::INFINITY, 0.0, 1.0), valid.clone()),
            "point 0 has a non-finite position",
        ),
    ];
    for (parts, expected) in cases {
        let error = error_of(parts);
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn invalid_pose_blocks_and_view_ids_are_rejected() {
    let error = error_of(parts().with_pose(PoseId::new(3), pose(0)));
    assert!(error.contains("pose block 3 does not exist"), "{error}");
    let error = error_of(parts().with_pose(PoseId::new(1), pose(0)));
    assert!(error.contains("pose block 1 is given twice"), "{error}");

    let misplaced = ViewRecord::new(
        ViewId::new(1),
        ImageMeta::external_id("0", 640, 480),
        IntrinsicId::new(0),
        PoseId::new(0),
    );
    let parts = ReconstructionParts::new(7_u64, SfmEngine::Global, vec![camera()], vec![misplaced]);
    let error = error_of(parts);
    assert!(error.contains("view IDs must be dense"), "{error}");
}
