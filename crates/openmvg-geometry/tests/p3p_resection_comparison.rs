// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comparison tests: P3P vs ResectionSolver
//!
//! Verifies that P3P produces poses consistent with the 6-point DLT solver.

use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, UnitQuaternion, UnitVector3, Vector3};
use std::f64::consts::PI;

use openmvg_geometry::pose::p3p::{self, P3pMethod, P3pSolutions};
use openmvg_geometry::pose::{BearingToPoint, ResectionConfig, ResectionSolver};

/// Solve LambdaTwist P3P on an exact three-correspondence slice.
fn lambda_twist(correspondences: &[BearingToPoint]) -> P3pSolutions {
    let sample: &[BearingToPoint; 3] = correspondences.try_into().unwrap();
    p3p::solve(
        sample.map(|correspondence| *correspondence.world_point()),
        sample.map(|correspondence| *correspondence.bearing()),
        P3pMethod::LambdaTwist,
    )
}

/// Create a rotation matrix from axis-angle representation.
fn rotation_from_axis_angle(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;

    let x = axis.x;
    let y = axis.y;
    let z = axis.z;

    Matrix3::new(
        t * x * x + c,
        t * x * y - z * s,
        t * x * z + y * s,
        t * x * y + z * s,
        t * y * y + c,
        t * y * z - x * s,
        t * x * z - y * s,
        t * y * z + x * s,
        t * z * z + c,
    )
}

/// Create pose from rotation matrix and translation.
fn pose_from_rt(r: &Matrix3<f64>, t: Vector3<f64>) -> Isometry3<f64> {
    let rotation = Rotation3::from_matrix(r);
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
    Isometry3::from_parts(t.into(), quaternion)
}

/// Generate correspondences from a known pose.
fn generate_correspondences(
    pose: &Isometry3<f64>,
    world_points: &[Point3<f64>],
) -> Vec<BearingToPoint> {
    world_points
        .iter()
        .filter_map(|wp| {
            let camera = pose.transform_point(wp);
            (camera.z > 1e-10).then(|| {
                BearingToPoint::try_new(*wp, UnitVector3::new_normalize(camera.coords)).unwrap()
            })
        })
        .collect()
}

#[test]
fn test_p3p_vs_resection_identity() {
    // Ground truth pose
    let true_pose = Isometry3::<f64>::identity();

    // Generate 8 world points (enough for both P3P and resection)
    let world_points = vec![
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 0.0, 5.0),
        Point3::new(0.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.0),
        Point3::new(-1.0, 0.0, 5.0),
        Point3::new(0.0, -1.0, 5.0),
        Point3::new(0.5, 0.5, 6.0),
        Point3::new(-0.5, -0.5, 4.0),
    ];

    let correspondences = generate_correspondences(&true_pose, &world_points);
    assert!(correspondences.len() >= 6);

    // Test P3P with first 3 points
    let p3p_correspondences = &correspondences[0..3];
    let p3p_poses = lambda_twist(p3p_correspondences);

    assert!(!p3p_poses.is_empty(), "P3P returned no poses");

    // Test resection with all 6+ points
    let resection_config = ResectionConfig::default();
    let resection_poses = ResectionSolver::solve(&correspondences, &resection_config).unwrap();

    assert_eq!(resection_poses.len(), 1, "Resection should return 1 pose");

    // Find the P3P pose with lowest error against ALL points
    let best_p3p_pose = p3p_poses
        .iter()
        .map(|pose| {
            let error: f64 = correspondences
                .iter()
                .map(|c| ResectionSolver::residual(pose, c))
                .sum();
            (pose, error)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(pose, _)| pose)
        .unwrap();

    // Compare P3P best pose with resection pose
    let resection_pose = &resection_poses[0];

    // Both poses should produce similar reprojection errors
    let p3p_total_error: f64 = correspondences
        .iter()
        .map(|c| ResectionSolver::residual(best_p3p_pose, c))
        .sum();

    let resection_total_error: f64 = correspondences
        .iter()
        .map(|c| ResectionSolver::residual(resection_pose, c))
        .sum();

    // Both should have very low error
    assert!(
        p3p_total_error < 1e-4,
        "P3P error too high: {}",
        p3p_total_error
    );
    assert!(
        resection_total_error < 1e-4,
        "Resection error too high: {}",
        resection_total_error
    );
}

#[test]
fn test_p3p_vs_resection_complex_pose() {
    // More complex pose: rotation + translation
    let angle = PI / 5.0; // 36 degrees
    let r = rotation_from_axis_angle(Vector3::new(1.0, 0.5, 0.2).normalize(), angle);
    let true_pose = pose_from_rt(&r, Vector3::new(2.0, -1.0, 4.0));

    // Generate world points at various depths
    let world_points = vec![
        Point3::new(0.0, 0.0, 12.0),
        Point3::new(2.0, 0.0, 12.0),
        Point3::new(0.0, 2.0, 12.0),
        Point3::new(2.0, 2.0, 14.0),
        Point3::new(-2.0, 0.0, 12.0),
        Point3::new(0.0, -2.0, 12.0),
        Point3::new(1.0, 1.0, 13.0),
        Point3::new(-1.0, -1.0, 11.0),
    ];

    let correspondences = generate_correspondences(&true_pose, &world_points);
    assert!(correspondences.len() >= 6);

    // Test P3P
    let p3p_correspondences = &correspondences[0..3];
    let p3p_poses = lambda_twist(p3p_correspondences);

    assert!(!p3p_poses.is_empty(), "P3P returned no poses");

    // Validate P3P poses against all points
    for pose in &p3p_poses {
        let max_error: f64 = correspondences
            .iter()
            .map(|c| ResectionSolver::residual(pose, c))
            .fold(0.0, f64::max);

        // At least one pose should have low error
        if max_error < 1e-5 {
            // Found a good pose!
            return;
        }
    }

    // If we get here, check that at least one pose is reasonably good
    let best_error = p3p_poses
        .iter()
        .map(|pose| {
            correspondences
                .iter()
                .map(|c| ResectionSolver::residual(pose, c))
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min);

    assert!(
        best_error < 1e-4,
        "P3P best pose has too high error: {}",
        best_error
    );
}

#[test]
fn test_p3p_validates_against_extra_points() {
    // This test checks that P3P poses are consistent when validated
    // against more than the minimal 3 points used for estimation.

    let true_pose = Isometry3::<f64>::identity();

    // Generate many points
    let world_points: Vec<Point3<f64>> = vec![
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 0.0, 5.0),
        Point3::new(0.0, 1.0, 5.0),
        // Additional validation points
        Point3::new(0.5, 0.5, 5.0),
        Point3::new(-0.5, 0.5, 5.0),
        Point3::new(0.5, -0.5, 5.0),
        Point3::new(-0.5, -0.5, 5.0),
        Point3::new(0.0, 0.0, 7.0),
        Point3::new(1.0, 1.0, 6.0),
        Point3::new(-1.0, -1.0, 4.0),
    ];

    let correspondences = generate_correspondences(&true_pose, &world_points);

    // Use P3P on first 3 points
    let p3p_correspondences = &correspondences[0..3];
    let poses = lambda_twist(p3p_correspondences);

    // Find best pose using ALL points
    let best_pose = poses
        .iter()
        .map(|pose| {
            let error: f64 = correspondences
                .iter()
                .map(|c| ResectionSolver::residual(pose, c))
                .sum();
            (pose.clone(), error)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // Best pose should have very low error on ALL points
    let max_individual_error: f64 = correspondences
        .iter()
        .map(|c| ResectionSolver::residual(&best_pose.0, c))
        .fold(0.0, f64::max);

    assert!(
        max_individual_error < 1e-6,
        "P3P pose has high error on validation points: {}",
        max_individual_error
    );
}
