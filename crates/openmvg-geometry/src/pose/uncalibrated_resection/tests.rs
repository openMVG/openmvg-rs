// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use nalgebra::{Matrix3, Point3, Vector3};

/// Generate synthetic correspondences from known K and pose.
fn generate_correspondences(
    intrinsics: &PinholeIntrinsics,
    pose: &SolverPose<f64>,
    world_points: &[Point3<f64>],
) -> Vec<PixelToPoint> {
    world_points
        .iter()
        .filter_map(|wp| {
            let camera = pose.transform_point(wp);
            intrinsics
                .project(&camera)
                .map(|pixel| PixelToPoint::new(*wp, pixel))
        })
        .collect()
}

#[test]
fn test_rq_decomposition_identity() {
    let m = Matrix3::<f64>::identity();
    let (k, r) = UncalibratedResectionSolver::rq_decomposition(&m).unwrap();

    // K should be identity, R should be identity
    assert!((k - Matrix3::identity()).norm() < 1e-10);
    assert!((r - Matrix3::identity()).norm() < 1e-10);
}

#[test]
fn test_rq_decomposition_with_k() {
    let k_true = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r_true = Matrix3::identity();
    let m = k_true * r_true;

    let (k, r) = UncalibratedResectionSolver::rq_decomposition(&m).unwrap();

    assert!((k - k_true).norm() < 1e-8, "K mismatch: {:?}", k);
    assert!((r - r_true).norm() < 1e-8, "R mismatch: {:?}", r);
}

#[test]
fn test_uncalibrated_resection_identity() {
    let intrinsics = PinholeIntrinsics::try_from_focal_and_principal(500.0, 320.0, 240.0).unwrap();
    let pose = SolverPose::<f64>::identity();

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

    let correspondences = generate_correspondences(&intrinsics, &pose, &world_points);
    assert!(correspondences.len() >= 6);

    let config = UncalibratedResectionConfig::default();
    let result = UncalibratedResectionSolver::solve(&correspondences, &config);

    assert!(
        result.is_ok(),
        "Uncalibrated resection failed: {:?}",
        result.err()
    );
    let models = result.unwrap();
    assert_eq!(models.len(), 1);

    let model = &models[0];

    // Check reprojection errors
    for corr in &correspondences {
        let error = UncalibratedResectionSolver::residual(model, corr);
        assert!(error < 1.0, "Reprojection error too large: {}", error);
    }
}

#[test]
fn test_uncalibrated_resection_with_translation() {
    let intrinsics = PinholeIntrinsics::try_from_focal_and_principal(600.0, 400.0, 300.0).unwrap();
    let pose = world_to_camera_from_matrix(&Matrix3::identity(), Vector3::new(0.5, 0.3, 1.0));

    let world_points = vec![
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(2.0, 0.0, 5.0),
        Point3::new(0.0, 2.0, 5.0),
        Point3::new(2.0, 2.0, 6.0),
        Point3::new(-2.0, -2.0, 5.0),
        Point3::new(3.0, 0.0, 7.0),
        Point3::new(0.0, 3.0, 4.0),
        Point3::new(-2.0, 2.0, 5.5),
    ];

    let correspondences = generate_correspondences(&intrinsics, &pose, &world_points);

    let config = UncalibratedResectionConfig::default();
    let result = UncalibratedResectionSolver::solve(&correspondences, &config);

    assert!(
        result.is_ok(),
        "Uncalibrated resection failed: {:?}",
        result.err()
    );
    let model = &result.unwrap()[0];

    // Check reprojection errors
    for corr in &correspondences {
        let error = UncalibratedResectionSolver::residual(model, corr);
        assert!(error < 1.0, "Reprojection error too large: {}", error);
    }
}

#[test]
fn test_uncalibrated_resection_with_rotation() {
    use std::f64::consts::PI;

    let intrinsics = PinholeIntrinsics::try_from_focal_and_principal(500.0, 320.0, 240.0).unwrap();

    // 30 degree rotation around Y axis
    let angle = PI / 6.0;
    let c = angle.cos();
    let s = angle.sin();
    let r = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);

    let pose = world_to_camera_from_matrix(&r, Vector3::new(1.0, 0.5, 2.0));

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

    let correspondences = generate_correspondences(&intrinsics, &pose, &world_points);

    let config = UncalibratedResectionConfig::default();
    let result = UncalibratedResectionSolver::solve(&correspondences, &config);

    assert!(
        result.is_ok(),
        "Uncalibrated resection with rotation failed: {:?}",
        result.err()
    );
    let model = &result.unwrap()[0];

    // Check reprojection errors
    let mut max_error: f64 = 0.0;
    for corr in &correspondences {
        let error = UncalibratedResectionSolver::residual(model, corr);
        max_error = max_error.max(error);
    }
    assert!(
        max_error < 1.0,
        "Max reprojection error too large: {}",
        max_error
    );
}

#[test]
fn test_uncalibrated_resection_rejects_repeated_correspondences() {
    let correspondence = PixelToPoint::new(
        Point3::new(0.0, 0.0, 5.0),
        nalgebra::Point2::new(320.0, 240.0),
    );
    let result = UncalibratedResectionSolver::solve(
        &vec![correspondence; 6],
        &UncalibratedResectionConfig::default(),
    );
    assert!(matches!(
        result,
        Err(SolverError::DegenerateConfiguration { .. })
    ));
}

#[test]
fn test_uncalibrated_resection_accepts_noisy_overdetermined_data() {
    let intrinsics = PinholeIntrinsics::try_new(700.0, 680.0, 320.0, 240.0, 0.2).unwrap();
    let pose = world_to_camera_from_matrix(&Matrix3::identity(), Vector3::new(0.3, -0.2, 1.0));
    let world_points = vec![
        Point3::new(-1.0, -0.5, 5.0),
        Point3::new(0.5, -1.0, 6.0),
        Point3::new(1.2, 0.3, 7.0),
        Point3::new(-0.7, 1.1, 5.5),
        Point3::new(0.2, 0.8, 8.0),
        Point3::new(1.5, -0.6, 6.5),
        Point3::new(-1.4, 0.2, 7.5),
        Point3::new(0.8, 1.4, 9.0),
        Point3::new(-0.3, -1.3, 8.5),
        Point3::new(1.7, 0.9, 10.0),
    ];
    let mut correspondences = generate_correspondences(&intrinsics, &pose, &world_points);
    for (i, correspondence) in correspondences.iter_mut().enumerate() {
        correspondence.pixel.x += 0.02 * (i as f64 + 0.2).sin();
        correspondence.pixel.y += 0.02 * (i as f64 + 0.7).cos();
    }

    let models = UncalibratedResectionSolver::solve(
        &correspondences,
        &UncalibratedResectionConfig::default(),
    )
    .unwrap();
    let average = correspondences
        .iter()
        .map(|corr| UncalibratedResectionSolver::residual(&models[0], corr))
        .sum::<f64>()
        / correspondences.len() as f64;
    assert!(average < 0.1, "average reprojection error: {average}");
}

#[test]
fn fallible_intrinsics_constructors_reject_invalid_inputs() {
    assert!(PinholeIntrinsics::try_new(0.0, 500.0, 320.0, 240.0, 0.0).is_err());
    assert!(PinholeIntrinsics::try_from_focal_and_principal(f64::NAN, 320.0, 240.0,).is_err());
    let singular = Matrix3::<f64>::zeros();
    assert!(PinholeIntrinsics::try_from_matrix(&singular).is_err());

    let valid = PinholeIntrinsics::try_new(500.0, 510.0, 320.0, 240.0, 0.1).unwrap();
    assert!(valid.is_valid());
    assert_eq!(
        PinholeIntrinsics::try_from_matrix(&valid.to_matrix()).unwrap(),
        valid
    );
}

#[test]
fn test_intrinsics_to_matrix() {
    let k = PinholeIntrinsics::try_new(500.0, 600.0, 320.0, 240.0, 0.5).unwrap();
    let m = k.to_matrix();

    assert_eq!(m[(0, 0)], 500.0); // fx
    assert_eq!(m[(1, 1)], 600.0); // fy
    assert_eq!(m[(0, 2)], 320.0); // cx
    assert_eq!(m[(1, 2)], 240.0); // cy
    assert_eq!(m[(0, 1)], 0.5); // skew
    assert_eq!(m[(2, 2)], 1.0);
}
