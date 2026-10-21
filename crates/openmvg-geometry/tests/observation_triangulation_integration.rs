// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-solver triangulation contract over one typed observation set.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Point2, Point3, Rotation3};
use openmvg_core::model::{ViewId, WorldToCameraPose};
use openmvg_geometry::triangulation::{TriangulationConfig, TriangulationMethod, triangulate};

/// Every retained pure solver consumes the same typed view/pixel observations
/// and recovers a well-conditioned point without an extraction adapter.
#[test]
fn test_all_solvers_with_extracted_observations() {
    let point = Point3::new(0.25, -0.1, 5.0);
    let intrinsics = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let centers = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];

    let mut poses = BTreeMap::new();
    let mut camera_matrices = BTreeMap::new();
    let mut observations = Vec::new();
    for (index, center) in centers.into_iter().enumerate() {
        let view = ViewId::new(index);
        let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)
            .expect("valid pose");
        let camera_point = pose.project_to_camera(&point);
        let pixel = Point2::new(
            intrinsics[(0, 0)] * camera_point.x / camera_point.z + intrinsics[(0, 2)],
            intrinsics[(1, 1)] * camera_point.y / camera_point.z + intrinsics[(1, 2)],
        );
        poses.insert(view, pose);
        camera_matrices.insert(view, intrinsics);
        observations.push((view, pixel));
    }

    let methods = [
        ("dlt", TriangulationMethod::Dlt),
        ("midpoint", TriangulationMethod::Midpoint),
        ("l1-angular", TriangulationMethod::L1Angular),
        ("linfinity-angular", TriangulationMethod::LInfinityAngular),
        (
            "inverse-depth-weighted-midpoint",
            TriangulationMethod::InverseDepthWeightedMidpoint,
        ),
    ];
    let config = TriangulationConfig::default();

    for (name, method) in methods {
        let result = triangulate(method, &observations, &poses, &camera_matrices, &config)
            .unwrap_or_else(|error| panic!("{name} solver failed: {error}"));
        let error = (result.point - point).norm();
        assert!(error < 0.1, "{name} error {error} exceeded 0.1");
    }
}
