// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression test: F8 solver must return F, NOT Fᵀ.
//!
//! This test exists because a bug in the constraint row vectorization order
//! (row-major vs column-major) caused the 8-point solver to return the
//! TRANSPOSE of the fundamental matrix. The bug was invisible to the original
//! tests because they used a skew-symmetric ground truth matrix where F = -Fᵀ ∝ Fᵀ.
//!
//! This test uses:
//!   - A realistic camera geometry producing an ASYMMETRIC F (F ≢ αFᵀ)
//!   - 3D points projected through both cameras (not synthesized from F)
//!   - Direct matrix comparison against ground truth (not just algebraic error)

use nalgebra::{Matrix3, Point2, Rotation3, Unit, Vector3};

use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::fundamental::EightPointSolver;
use openmvg_geometry::normalization::NormalizedCorrespondences;

/// Create ground-truth fundamental matrix: F = K⁻ᵀ · [t]× · R · K⁻¹
fn make_ground_truth_fundamental(
    focal: f64,
    cx: f64,
    cy: f64,
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
) -> Matrix3<f64> {
    let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
    let k_inv = k.try_inverse().unwrap();
    let t = translation;
    let t_cross = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    let r = rotation.matrix();
    k_inv.transpose() * t_cross * r * k_inv
}

/// Project 3D points through a camera pair to get pixel correspondences.
fn generate_correspondences(
    focal: f64,
    cx: f64,
    cy: f64,
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
    num_points: usize,
) -> Vec<PixelCorrespondence> {
    let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
    let r = rotation.matrix();

    (0..num_points)
        .map(|i| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_points as f64);
            let radius = 0.5 + (i as f64) * 0.1;
            let world_pt = Vector3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                4.0 + (i as f64) * 0.3,
            );

            let proj1 = k * world_pt;
            let p1 = Point2::new(proj1.x / proj1.z, proj1.y / proj1.z);

            let cam2_pt = r * world_pt + translation;
            let proj2 = k * cam2_pt;
            let p2 = Point2::new(proj2.x / proj2.z, proj2.y / proj2.z);

            PixelCorrespondence::new(p1, p2)
        })
        .collect()
}

/// Frobenius distance after normalizing both matrices to unit norm (up to sign).
fn matrix_distance_up_to_scale(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let a_n = a / a.norm();
    let b_n = b / b.norm();
    (a_n - b_n).norm().min((a_n + b_n).norm())
}

/// Verify the solver returns the correct F, NOT its transpose.
///
/// The key insight: we use a camera geometry that produces an asymmetric F
/// (F ≠ αFᵀ for any scalar α). Then we check that the estimated matrix is
/// closer to F_true than to F_true^T. If the solver has a row/column-major
/// mismatch, it returns Fᵀ and this test fails.
#[test]
fn test_f8_returns_f_not_transpose() {
    let focal = 500.0;
    let cx = 320.0;
    let cy = 240.0;
    let rotation =
        Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 15.0_f64.to_radians());
    let translation = Vector3::new(1.0, 0.2, 0.0);

    let f_true = make_ground_truth_fundamental(focal, cx, cy, &rotation, &translation);
    let f_true_t = f_true.transpose();

    // Precondition: F must be genuinely asymmetric for this test to be meaningful
    let asymmetry = matrix_distance_up_to_scale(&f_true, &f_true_t);
    assert!(
        asymmetry > 0.1,
        "Test precondition failed: F_true is nearly symmetric (distance={:.6}). \
         This test requires an asymmetric fundamental matrix to distinguish F from Fᵀ.",
        asymmetry,
    );

    let correspondences = generate_correspondences(focal, cx, cy, &rotation, &translation, 20);

    // Normalize → solve → denormalize (the standard RANSAC pipeline); the
    // public solve returns the pixel-space model.
    let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

    let f_est = EightPointSolver::solve(&normalized)
        .expect("Solver should not error on noiseless data")
        .pop()
        .expect("Solver should return a model");

    // --- Core assertion: F_est ≈ F_true, NOT F_trueᵀ ---
    let dist_to_f = matrix_distance_up_to_scale(f_est.matrix(), &f_true);
    let dist_to_ft = matrix_distance_up_to_scale(f_est.matrix(), &f_true_t);

    assert!(
        dist_to_f < 0.05,
        "Estimated F does not match ground truth F.\n\
         dist(F_est, F_true)  = {:.6}\n\
         dist(F_est, F_trueᵀ) = {:.6}",
        dist_to_f,
        dist_to_ft,
    );

    assert!(
        dist_to_f < dist_to_ft,
        "BUG DETECTED: Estimated F is closer to Fᵀ than to F!\n\
         dist(F_est, F_true)  = {:.6}\n\
         dist(F_est, F_trueᵀ) = {:.6}\n\
         This indicates a row-major/column-major vectorization mismatch \
         in the constraint row or matrix reshape.",
        dist_to_f,
        dist_to_ft,
    );

    // --- Also verify algebraic error is near zero ---
    for (i, corr) in correspondences.iter().enumerate() {
        let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
        let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);
        let error = (x2.transpose() * f_est.matrix() * x1)[(0, 0)].abs();
        assert!(
            error < 1e-6,
            "Algebraic error at correspondence {}: {:.2e}",
            i,
            error,
        );
    }
}
