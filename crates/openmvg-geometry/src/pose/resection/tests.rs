// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use nalgebra::{Matrix3, Point2, Point3, Vector3};
use std::f64::consts::PI;

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

/// Generate synthetic correspondences from a known pose.
fn generate_correspondences(
    pose: &SolverPose<f64>,
    world_points: &[Point3<f64>],
) -> Vec<BearingToPoint> {
    world_points
        .iter()
        .filter_map(|wp| {
            let camera = pose.transform_point(wp);
            (camera.z > 1e-10).then(|| {
                BearingToPoint::from_normalized(
                    *wp,
                    Point2::new(camera.x / camera.z, camera.y / camera.z),
                )
            })
        })
        .collect()
}

#[test]
fn test_resection_residual_is_unsquared_angle() {
    let angle: f64 = 0.2;
    let corr = BearingToPoint::from_direction(
        Point3::new(0.0, 0.0, 5.0),
        Vector3::new(angle.sin(), 0.0, angle.cos()),
    );

    let residual = ResectionSolver::residual(&WorldToCameraPose::identity(), &corr);
    assert!((residual - angle).abs() < 1e-12);
}

#[test]
fn bearing_refinement_never_increases_angular_cost() {
    let truth_rotation = *nalgebra::Rotation3::from_euler_angles(0.08, -0.13, 0.05).matrix();
    let truth_translation = Vector3::new(0.2, -0.1, 0.3);
    let correspondences: Vec<_> = (0..16)
        .map(|index| {
            let value = index as f64;
            let point = Point3::new(
                1.2 * (0.7 * value).sin(),
                0.9 * (1.1 * value).cos(),
                4.0 + 0.15 * value,
            );
            BearingToPoint::from_direction(point, truth_rotation * point.coords + truth_translation)
        })
        .collect();
    let initial_rotation =
        *nalgebra::Rotation3::from_scaled_axis(Vector3::new(0.015, -0.01, 0.008)).matrix()
            * truth_rotation;
    let initial = world_to_camera_from_matrix(
        &initial_rotation,
        truth_translation + Vector3::new(0.03, -0.02, 0.015),
    );
    let before = correspondences
        .iter()
        .map(|correspondence| bearing_pose_angular_residual(&initial, correspondence).powi(2))
        .sum::<f64>();
    let refined = refine_bearing_solver_pose(&initial, &correspondences)
        .expect("well-conditioned resection should refine");
    let after = correspondences
        .iter()
        .map(|correspondence| bearing_pose_angular_residual(&refined, correspondence).powi(2))
        .sum::<f64>();

    assert!(
        after <= before,
        "angular cost worsened: {before} -> {after}"
    );
}

#[test]
fn bearing_refinement_supports_rear_hemisphere_rays() {
    let truth_rotation = *nalgebra::Rotation3::from_euler_angles(0.06, -0.04, 0.03).matrix();
    let truth_translation = Vector3::new(0.15, -0.08, 0.2);
    let world_points = [
        Point3::new(-2.0, -0.5, -4.0),
        Point3::new(1.4, -1.1, -3.2),
        Point3::new(-1.2, 1.5, -2.7),
        Point3::new(2.2, 0.8, -4.8),
        Point3::new(-1.6, -1.3, 3.1),
        Point3::new(1.8, -0.7, 4.2),
        Point3::new(-0.9, 1.7, 5.0),
        Point3::new(2.4, 1.2, 3.6),
    ];
    let correspondences: Vec<_> = world_points
        .into_iter()
        .map(|point| {
            BearingToPoint::from_direction(point, truth_rotation * point.coords + truth_translation)
        })
        .collect();
    assert!(
        correspondences
            .iter()
            .any(|correspondence| correspondence.bearing.z < 0.0)
    );
    let initial_rotation = *nalgebra::Rotation3::from_scaled_axis(Vector3::new(0.02, -0.015, 0.01))
        .matrix()
        * truth_rotation;
    let initial = world_to_camera_from_matrix(
        &initial_rotation,
        truth_translation + Vector3::new(0.04, -0.03, 0.025),
    );
    let before = correspondences
        .iter()
        .map(|correspondence| bearing_pose_angular_residual(&initial, correspondence).powi(2))
        .sum::<f64>();

    let refined = refine_bearing_solver_pose(&initial, &correspondences)
        .expect("full-sphere bearing pose should refine");
    let after = correspondences
        .iter()
        .map(|correspondence| bearing_pose_angular_residual(&refined, correspondence).powi(2))
        .sum::<f64>();

    assert!(after < before * 1e-8, "angular cost: {before} -> {after}");
}

#[test]
fn test_resection_identity_pose() {
    // Camera at origin, looking along +Z
    let true_pose = SolverPose::<f64>::identity();

    // 3D points in front of camera
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

    let config = ResectionConfig::default();
    let result = ResectionSolver::solve(&correspondences, &config);

    assert!(result.is_ok(), "Resection failed: {:?}", result.err());
    let poses = result.unwrap();
    assert_eq!(poses.len(), 1);

    let pose = &poses[0];

    // Check reprojection errors (angular error should be tiny)
    for corr in &correspondences {
        let error = ResectionSolver::residual(pose, corr);
        assert!(error < 1e-6, "Angular error too large: {}", error);
    }
}

#[test]
fn test_resection_with_translation() {
    // Camera translated back
    let true_pose = world_to_camera_from_matrix(&Matrix3::identity(), Vector3::new(0.0, 0.0, 2.0));

    let world_points = vec![
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 0.0, 5.0),
        Point3::new(0.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 6.0),
        Point3::new(-1.0, -1.0, 5.0),
        Point3::new(2.0, 0.0, 7.0),
        Point3::new(0.0, 2.0, 4.0),
        Point3::new(-1.0, 1.0, 5.5),
    ];

    let correspondences = generate_correspondences(&true_pose, &world_points);

    let config = ResectionConfig::default();
    let result = ResectionSolver::solve(&correspondences, &config);

    assert!(result.is_ok(), "Resection failed: {:?}", result.err());
    let poses = result.unwrap();
    let pose = &poses[0];

    // Check angular errors
    for corr in &correspondences {
        let error = ResectionSolver::residual(pose, corr);
        assert!(error < 1e-6, "Angular error too large: {}", error);
    }
}

/// Calibrated resection pose recovery: calibrated resection recovers a non-identity world-to-camera
/// rotation; the recovered pose preserves the angular residual convention.
#[test]
fn test_resection_with_rotation() {
    // Camera rotated 30 degrees around Y axis, translated back
    let angle = PI / 6.0; // 30 degrees
    let r = rotation_from_axis_angle(Vector3::new(0.0, 1.0, 0.0).normalize(), angle);
    let true_pose = world_to_camera_from_matrix(&r, Vector3::new(1.0, 0.0, 3.0));

    let world_points = vec![
        Point3::new(0.0, 0.0, 8.0),
        Point3::new(2.0, 0.0, 8.0),
        Point3::new(0.0, 2.0, 8.0),
        Point3::new(2.0, 2.0, 10.0),
        Point3::new(-2.0, 0.0, 8.0),
        Point3::new(0.0, -2.0, 8.0),
        Point3::new(1.0, 1.0, 9.0),
        Point3::new(-1.0, -1.0, 7.0),
    ];

    let correspondences = generate_correspondences(&true_pose, &world_points);

    let config = ResectionConfig::default();
    let result = ResectionSolver::solve(&correspondences, &config);

    assert!(result.is_ok(), "Resection failed: {:?}", result.err());
    let poses = result.unwrap();
    let pose = &poses[0];

    // Check angular errors
    let mut max_error = 0.0;
    for corr in &correspondences {
        let error = ResectionSolver::residual(pose, corr);
        max_error = max_error.max(error);
    }
    assert!(
        max_error < 1e-6,
        "Max angular error too large: {}",
        max_error
    );
}

#[test]
fn test_resection_constraint_is_finite_at_bearing_horizon() {
    let correspondences: Vec<_> = (0..6)
        .map(|i| {
            BearingToPoint::from_direction(
                Point3::new(i as f64, (i * i) as f64, 4.0 + i as f64),
                Vector3::new(1.0, 0.2 + 0.1 * i as f64, 0.0),
            )
        })
        .collect();
    let constraints = ResectionSolver::build_constraint_matrix_6point(&correspondences);
    assert!(constraints.iter().all(|value| value.is_finite()));
}

#[test]
fn test_resection_supports_full_sphere_bearings() {
    let world_points = [
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(-1.0, 1.2, 0.2),
        Point3::new(0.2, -2.0, -1.0),
        Point3::new(1.0, 1.0, -2.0),
        Point3::new(-1.0, -0.5, -3.0),
        Point3::new(0.5, -1.0, 2.0),
        Point3::new(1.5, 0.3, 1.0),
        Point3::new(-0.3, 1.4, -1.2),
    ];
    let correspondences = world_points
        .into_iter()
        .map(|point| BearingToPoint::from_direction(point, point.coords))
        .collect::<Vec<_>>();

    let poses = ResectionSolver::solve(&correspondences, &ResectionConfig::default())
        .expect("full-sphere resection should succeed");
    let pose = &poses[0];
    let maximum_residual = correspondences
        .iter()
        .map(|correspondence| ResectionSolver::residual(pose, correspondence))
        .fold(0.0_f64, f64::max);

    assert!(
        maximum_residual < 1.0e-6,
        "maximum residual: {maximum_residual}"
    );
    assert!(correspondences.iter().any(|correspondence| {
        correspondence.bearing.z <= 0.0
            && ResectionSolver::residual(pose, correspondence).is_finite()
    }));
}

#[test]
fn test_resection_rejects_repeated_world_points() {
    let correspondence =
        BearingToPoint::from_direction(Point3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, 1.0));
    let result = ResectionSolver::solve(&vec![correspondence; 6], &ResectionConfig::default());
    assert!(matches!(
        result,
        Err(SolverError::DegenerateConfiguration { .. })
    ));
}

#[test]
fn test_resection_large_world_offset_and_noisy_bearings() {
    let offset = Vector3::new(1.0e6, -2.0e6, 3.0e6);
    let true_pose = world_to_camera_from_matrix(&Matrix3::identity(), -offset);
    let local_points = [
        Point3::new(-1.0, -0.5, 4.0),
        Point3::new(0.5, -1.0, 5.0),
        Point3::new(1.2, 0.3, 6.0),
        Point3::new(-0.7, 1.1, 5.5),
        Point3::new(0.2, 0.8, 7.0),
        Point3::new(1.5, -0.6, 6.5),
        Point3::new(-1.4, 0.2, 7.5),
        Point3::new(0.8, 1.4, 8.0),
    ];
    let world_points: Vec<_> = local_points
        .iter()
        .map(|point| Point3::from(point.coords + offset))
        .collect();
    let mut correspondences = generate_correspondences(&true_pose, &world_points);
    for (i, correspondence) in correspondences.iter_mut().enumerate() {
        let mut direction = *correspondence.bearing.as_ref();
        direction.x += 1e-7 * (i as f64 + 0.2).sin();
        direction.y += 1e-7 * (i as f64 + 0.4).cos();
        *correspondence = BearingToPoint::from_direction(correspondence.world_point, direction);
    }

    let poses = ResectionSolver::solve(&correspondences, &ResectionConfig::default()).unwrap();
    let max_error = correspondences
        .iter()
        .map(|corr| ResectionSolver::residual(&poses[0], corr))
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1e-5, "max angular error: {max_error}");
}

#[test]
fn indexed_minimal_sample_matches_the_gathered_slice() {
    let angle = PI / 6.0;
    let r = rotation_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), angle);
    let true_pose = world_to_camera_from_matrix(&r, Vector3::new(1.0, 0.0, 3.0));
    // Generic positions: no four coplanar, no three collinear.
    let world_points = [
        Point3::new(0.3, -0.2, 8.1),
        Point3::new(2.1, 0.4, 7.3),
        Point3::new(-0.7, 2.2, 9.4),
        Point3::new(1.6, 1.9, 10.2),
        Point3::new(-2.3, -0.6, 8.7),
        Point3::new(0.2, -2.4, 6.9),
        Point3::new(1.1, 0.8, 11.5),
        Point3::new(-1.4, -1.7, 7.6),
    ];
    let data = generate_correspondences(&true_pose, &world_points);
    let config = ResectionConfig::default();
    let indices = [7, 1, 2, 4, 5, 0];
    let gathered: Vec<_> = indices.iter().map(|&index| data[index]).collect();

    let expected = ResectionSolver::solve_single(&gathered, &config).unwrap();
    let indexed = ResectionSolver::solve_single_from_indices(&indices, &data, &config).unwrap();
    assert_eq!(indexed, expected);
    assert!(expected.is_some());
    assert!(matches!(
        ResectionSolver::solve_single_from_indices(&[0, 1, 2, 3, 4, 99], &data, &config),
        Err(SolverError::InvalidDataIndex { index: 99, .. })
    ));
}

#[test]
fn test_resection_insufficient_points() {
    let correspondences: Vec<BearingToPoint> = (0..5)
        .map(|i| {
            BearingToPoint::from_normalized(
                Point3::new(i as f64, 0.0, 5.0),
                Point2::new((i as f64) / 5.0, 0.0),
            )
        })
        .collect();

    let config = ResectionConfig::default();
    let result = ResectionSolver::solve(&correspondences, &config);

    assert!(result.is_err());
    if let Err(SolverError::InsufficientData { required, provided }) = result {
        assert_eq!(required, 6);
        assert_eq!(provided, 5);
    } else {
        panic!("Expected InsufficientData error");
    }
}
