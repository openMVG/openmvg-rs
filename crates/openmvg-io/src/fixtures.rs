// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A synthetic scene shared by the tests: pinhole views of a point grid, with the exact
//! projections as keypoints, and the reconstructions both engines make of it.

use std::path::Path;

use nalgebra::{Matrix3, Point3, Rotation3, Unit, Vector3};
use openmvg_sfm::model::{ImageMeta, PairGeometry, ViewRecord};
use openmvg_sfm::{
    Camera, FeatureId, FeatureMatch, Global, Incremental, IntrinsicId, Keypoint, PoseId,
    Reconstruction, RelativePose, ViewGraph, ViewGraphParts, ViewId, ViewPair,
};

pub(crate) const IMAGE_ROOT: &str = "/data/scene/images";
pub(crate) const ENGINE_POINTS: usize = 96;
pub(crate) const ENGINE_VIEWS: usize = 6;
pub(crate) const FOCAL: f64 = 500.0;

/// An image file below `IMAGE_ROOT`, named by its path relative to it, as a dataset reads back.
pub(crate) fn image(view: usize, width: u32, height: u32) -> ImageMeta {
    let relative = format!("rig/view-{view}.jpg");
    let mut meta = ImageMeta::path(Path::new(IMAGE_ROOT).join(&relative), width, height);
    meta.name = relative;
    meta
}

/// World-to-camera rotation and camera center of `view`.
pub(crate) fn view_pose(view: usize) -> (Rotation3<f64>, Vector3<f64>) {
    let step = view as f64;
    (
        Rotation3::from_euler_angles(0.02 * step, -0.06 * step, 0.01 * step),
        Vector3::new(0.5 * step, 0.05 * step, -0.1 * step),
    )
}

pub(crate) fn points(count: usize) -> Vec<Point3<f64>> {
    (0..count)
        .map(|index| {
            Point3::new(
                -1.0 + 0.25 * (index % 9) as f64,
                -0.8 + 0.2 * ((index / 9) % 9) as f64,
                5.0 + 0.5 * (index % 7) as f64,
            )
        })
        .collect()
}

/// The pinhole projections of every point in `view`; every fourth has no scale.
pub(crate) fn keypoints(view: usize, count: usize) -> Vec<Keypoint> {
    let (rotation, center) = view_pose(view);
    points(count)
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let local = rotation * (point.coords - center);
            Keypoint::try_new(
                (FOCAL * local.x / local.z + 320.0) as f32,
                (FOCAL * local.y / local.z + 240.0) as f32,
                if index % 4 == 0 {
                    0.0
                } else {
                    1.5 + index as f32
                },
                0.1 * index as f32 - 1.0,
            )
            .unwrap()
        })
        .collect()
}

pub(crate) fn relative_pose(first: usize, second: usize) -> RelativePose {
    let (rotation_i, center_i) = view_pose(first);
    let (rotation_j, center_j) = view_pose(second);
    RelativePose::from_projection_translation(
        rotation_j * rotation_i.inverse(),
        Unit::new_normalize(rotation_j * (center_i - center_j)),
    )
    .unwrap()
}

pub(crate) fn essential(pose: &RelativePose) -> Matrix3<f64> {
    pose.translation_source_to_target().cross_matrix() * pose.rotation_source_to_target().matrix()
}

/// Identity matches; every third has no distance.
pub(crate) fn matches(count: usize) -> Vec<FeatureMatch> {
    (0..count)
        .map(|feature| {
            let distance = (feature % 3 != 0).then_some(0.125 * feature as f32);
            FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), distance)
                .unwrap()
        })
        .collect()
}

pub(crate) fn pair(first: usize, second: usize) -> ViewPair {
    ViewPair::new(ViewId::new(first), ViewId::new(second))
}

/// Pinhole views of the points, every pair accepted with an essential geometry, so both engines
/// reconstruct the scene.
///
/// Each relative rotation is off by about a milliradian: Global conditions its rotations on
/// their spread, and exactly consistent rotations leave no spread to measure.
pub(crate) fn engine_graph() -> ViewGraph {
    let camera = Camera::try_pinhole(FOCAL, FOCAL, 320.0, 240.0).unwrap();
    let views = (0..ENGINE_VIEWS)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                image(view, 640, 480),
                IntrinsicId::new(0),
                PoseId::new(view),
            )
        })
        .collect();
    let keypoints = (0..ENGINE_VIEWS)
        .map(|view| {
            keypoints(view, ENGINE_POINTS)
                .into_iter()
                .map(|keypoint| Keypoint::try_new(keypoint.x(), keypoint.y(), 2.0, 0.0).unwrap())
                .collect()
        })
        .collect();
    let mut parts = ViewGraphParts::new(7, vec![camera], views, keypoints);
    for first in 0..ENGINE_VIEWS {
        for second in first + 1..ENGINE_VIEWS {
            let exact = relative_pose(first, second);
            let noise = Rotation3::from_euler_angles(
                1e-3 * (first as f64 - 2.0),
                -7e-4 * second as f64,
                5e-4,
            );
            let pose = RelativePose::from_projection_translation(
                noise * exact.rotation_source_to_target(),
                *exact.translation_source_to_target(),
            )
            .unwrap();
            parts = parts.with_pair(
                pair(first, second),
                matches(ENGINE_POINTS),
                Some((
                    PairGeometry::Essential {
                        matrix: essential(&pose),
                        relative_pose: Some(pose),
                    },
                    vec![true; ENGINE_POINTS],
                )),
            );
        }
    }
    ViewGraph::try_from_parts(parts).unwrap()
}

pub(crate) fn reconstruct(graph: &ViewGraph, engine: &str) -> Reconstruction {
    let output = match engine {
        "incremental" => Incremental::default().run_prepared(graph),
        _ => Global::default().run_prepared(graph),
    };
    output.unwrap().reconstruction().clone()
}
