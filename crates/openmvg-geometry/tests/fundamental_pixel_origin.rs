// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Exact nonplanar scene, P1=K1[I|0], P2=K2[I|ex]. Only image2's y origin changes.
use nalgebra::{Matrix3, Point2, Vector3};
use openmvg_geometry::{
    correspondence::PixelCorrespondence,
    fundamental::{EightPointSolver, FundamentalMatrix, estimate_fundamental},
    normalization::NormalizedCorrespondences,
    robust::{RobustOptions, threshold::PixelThreshold},
};

fn scene(
    shift: f64,
) -> (
    Vec<PixelCorrespondence>,
    Matrix3<f64>,
    Matrix3<f64>,
    Matrix3<f64>,
) {
    let world = [
        [-1.0, -1.0, 4.0],
        [1.0, -1.0, 5.0],
        [-1.0, 1.0, 6.0],
        [1.0, 1.0, 4.0],
        [-0.5, 0.25, 7.0],
        [0.75, -0.5, 4.5],
        [0.25, 0.8, 5.5],
        [-0.8, -0.2, 6.5],
        [0.45, 0.3, 4.2],
        [-0.35, -0.65, 5.7],
        [0.9, 0.6, 6.2],
        [-0.6, 0.7, 4.8],
    ];
    let k1 = Matrix3::new(500.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 1.0);
    let k2 = Matrix3::new(500.0, 0.0, 0.0, 0.0, 500.0, shift, 0.0, 0.0, 1.0);
    // Independent [ex]_x * I camera-frame oracle.
    let essential = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
    let pixels = world
        .into_iter()
        .map(|p| {
            let point = Vector3::from(p);
            let first = k1 * point;
            let second = k2 * (point + Vector3::new(1.0, 0.0, 0.0));
            PixelCorrespondence::new(
                Point2::new(first.x / first.z, first.y / first.z),
                Point2::new(second.x / second.z, second.y / second.z),
            )
        })
        .collect();
    (pixels, k1, k2, essential)
}

fn projective_distance(a: Matrix3<f64>, b: Matrix3<f64>) -> f64 {
    let a = a / a.norm();
    let b = b / b.norm();
    (a - b).norm().min((a + b).norm())
}

fn assert_geometry(
    model: &FundamentalMatrix,
    data: &[PixelCorrespondence],
    k1: Matrix3<f64>,
    k2: Matrix3<f64>,
    essential: Matrix3<f64>,
) {
    // Pull back pixels to calibrated rays. This avoids comparing ill-conditioned raw F entries.
    assert!(projective_distance(k2.transpose() * model.matrix() * k1, essential) < 1e-7);
    for datum in data {
        assert!(EightPointSolver::residual(model, datum) < 1e-6);
    }
}

#[test]
fn physical_pixel_fundamental_rank_does_not_depend_on_image_origin() {
    for shift in [0.0, 500.0, 1000.0, 2000.0] {
        let (data, k1, k2, essential) = scene(shift);
        let expected =
            k2.try_inverse().unwrap().transpose() * essential * k1.try_inverse().unwrap();
        let model = FundamentalMatrix::try_new(expected)
            .unwrap_or_else(|e| panic!("physical rank-two F rejected for y-origin {shift}: {e}"));
        assert!(model.is_valid());
        assert_eq!(model.matrix().map(f64::to_bits), expected.map(f64::to_bits));
        assert_geometry(&model, &data, k1, k2, essential);
    }
}

#[test]
fn eight_point_pixel_fundamental_survives_image_origin_shift() {
    for shift in [0.0, 500.0, 1000.0, 2000.0] {
        let (data, k1, k2, essential) = scene(shift);
        let models =
            EightPointSolver::solve(&NormalizedCorrespondences::from_pixels(&data)).unwrap();
        assert_eq!(models.len(), 1, "y-origin {shift}");
        assert!(
            models[0].is_valid(),
            "pixel model invalid for y-origin {shift}"
        );
        assert_geometry(&models[0], &data, k1, k2, essential);
    }
}

#[test]
fn robust_pixel_fundamental_survives_image_origin_shift() {
    let options = RobustOptions::default().with_seed(31).with_max_draws(128);
    for shift in [0.0, 500.0, 1000.0, 2000.0] {
        let (data, k1, k2, essential) = scene(shift);
        let result = estimate_fundamental(&data, PixelThreshold::try_new(1e-5).unwrap(), &options)
            .unwrap_or_else(|e| panic!("physical pair rejected for y-origin {shift}: {e}"));
        assert!(result.model().is_valid());
        assert_eq!(result.inliers().len(), data.len(), "y-origin {shift}");
        assert_geometry(result.model(), &data, k1, k2, essential);
    }
}
