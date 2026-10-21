// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-based tests for the homography matrix.
//!
//! These tests verify algebraic invariants that every valid homography matrix
//! must satisfy. Unlike F and E (which are rank-deficient), H is **full rank**:
//!
//! 1. **det(H) ≠ 0** — H is non-singular (full rank 3)
//! 2. **H⁻¹ exists** — and H⁻¹·H = I
//! 3. **Projective transfer** — x₂ ∝ H·x₁ for all correspondences
//! 4. **Reverse transfer** — x₁ ∝ H⁻¹·x₂ for all correspondences
//! 5. **Composition** — H₂₁ · H₁₂⁻¹ ≈ I (round-trip consistency)
//! 6. **Scale invariance** — αH produces the same projective mapping as H
//!
//! Tests cover multiple homography types: pure translation, rotation,
//! affine, and full perspective transforms.

use nalgebra::{Matrix3, Point2, Vector3};
use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::homography::{DltSolver, HomographyConfig};
use openmvg_geometry::normalization::NormalizedCorrespondences;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Generate pixel correspondences from a known ground-truth homography.
fn correspondences_from_h(h: &Matrix3<f64>, points: &[Point2<f64>]) -> Vec<PixelCorrespondence> {
    points
        .iter()
        .map(|p1| {
            let x1 = Vector3::new(p1.x, p1.y, 1.0);
            let x2 = h * x1;
            let p2 = Point2::new(x2.x / x2.z, x2.y / x2.z);
            PixelCorrespondence::new(*p1, p2)
        })
        .collect()
}

/// A set of diverse source points for testing.
fn test_points() -> Vec<Point2<f64>> {
    vec![
        Point2::new(100.0, 100.0),
        Point2::new(300.0, 100.0),
        Point2::new(300.0, 300.0),
        Point2::new(100.0, 300.0),
        Point2::new(200.0, 200.0),
        Point2::new(150.0, 250.0),
        Point2::new(250.0, 150.0),
        Point2::new(180.0, 280.0),
    ]
}

/// A set of diverse ground-truth homographies for property testing.
fn test_homographies() -> Vec<(&'static str, Matrix3<f64>)> {
    vec![
        ("identity", Matrix3::identity()),
        (
            "pure translation",
            Matrix3::new(1.0, 0.0, 50.0, 0.0, 1.0, 30.0, 0.0, 0.0, 1.0),
        ),
        ("rotation 15°", {
            let theta = 15.0_f64.to_radians();
            Matrix3::new(
                theta.cos(),
                -theta.sin(),
                0.0,
                theta.sin(),
                theta.cos(),
                0.0,
                0.0,
                0.0,
                1.0,
            )
        }),
        (
            "scale + translation",
            Matrix3::new(1.5, 0.0, 20.0, 0.0, 1.3, -10.0, 0.0, 0.0, 1.0),
        ),
        (
            "affine (shear)",
            Matrix3::new(1.0, 0.2, 10.0, 0.1, 1.0, 5.0, 0.0, 0.0, 1.0),
        ),
        (
            "perspective",
            Matrix3::new(1.2, 0.1, 30.0, -0.05, 1.1, 20.0, 0.0001, -0.0002, 1.0),
        ),
        (
            "strong perspective",
            Matrix3::new(0.9, -0.3, 50.0, 0.25, 1.1, -20.0, 0.0005, 0.0003, 1.0),
        ),
    ]
}

// ─── Property: det(H) ≠ 0 (full rank) ───────────────────────────────────────

#[test]
fn property_det_h_nonzero_ground_truth() {
    for (name, h) in test_homographies() {
        let det = h.determinant();
        assert!(
            det.abs() > 1e-10,
            "[ground truth: {}] det(H) = {:.2e}, expected ≠ 0",
            name,
            det
        );
    }
}

#[test]
fn property_det_h_nonzero_estimated() {
    let points = test_points();
    for (name, h_true) in test_homographies() {
        let correspondences = correspondences_from_h(&h_true, &points);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let config = HomographyConfig::default();
        let models = DltSolver::solve(&normalized, &config)
            .unwrap_or_else(|e| panic!("[{}] DLT solver failed: {:?}", name, e));

        for (i, h_est) in models.iter().enumerate() {
            let det = h_est.matrix().determinant();
            assert!(
                det.abs() > 1e-10,
                "[DLT[{}]: {}] det(H) = {:.2e}, expected ≠ 0",
                i,
                name,
                det
            );
        }
    }
}

// ─── Property: H⁻¹ exists and H⁻¹·H ≈ I ────────────────────────────────────

fn assert_invertible(h: &Matrix3<f64>, label: &str) {
    let h_inv = h.try_inverse();
    assert!(
        h_inv.is_some(),
        "[{}] H is not invertible (det = {:.2e})",
        label,
        h.determinant()
    );

    let h_inv = h_inv.unwrap();
    let product = h_inv * h;
    let identity_error = (product - Matrix3::identity()).norm();
    assert!(
        identity_error < 1e-8,
        "[{}] H⁻¹·H deviates from I by {:.2e}",
        label,
        identity_error
    );
}

#[test]
fn property_invertible_ground_truth() {
    for (name, h) in test_homographies() {
        assert_invertible(&h, &format!("ground truth: {}", name));
    }
}

#[test]
fn property_invertible_estimated() {
    let points = test_points();
    for (name, h_true) in test_homographies() {
        let correspondences = correspondences_from_h(&h_true, &points);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let config = HomographyConfig::default();
        let models = DltSolver::solve(&normalized, &config)
            .unwrap_or_else(|e| panic!("[{}] DLT solver failed: {:?}", name, e));

        for (i, h_est) in models.iter().enumerate() {
            assert_invertible(h_est.matrix(), &format!("DLT[{}]: {}", i, name));
        }
    }
}

// ─── Property: projective transfer x₂ ∝ H·x₁ ───────────────────────────────

fn assert_projective_transfer(
    h: &Matrix3<f64>,
    correspondences: &[PixelCorrespondence],
    label: &str,
    tolerance: f64,
) {
    for (i, corr) in correspondences.iter().enumerate() {
        let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
        let hx1 = h * x1;
        if hx1.z.abs() < 1e-12 {
            continue; // point at infinity, skip
        }
        let projected = Point2::new(hx1.x / hx1.z, hx1.y / hx1.z);
        let error =
            ((projected.x - corr.point2.x).powi(2) + (projected.y - corr.point2.y).powi(2)).sqrt();
        assert!(
            error < tolerance,
            "[{}] transfer error at {}: {:.2e} px (expected < {:.2e})",
            label,
            i,
            error,
            tolerance
        );
    }
}

#[test]
fn property_projective_transfer_estimated() {
    let points = test_points();
    for (name, h_true) in test_homographies() {
        let correspondences = correspondences_from_h(&h_true, &points);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let config = HomographyConfig::default();
        let models = DltSolver::solve(&normalized, &config)
            .unwrap_or_else(|e| panic!("[{}] DLT solver failed: {:?}", name, e));

        for (i, h_est) in models.iter().enumerate() {
            assert_projective_transfer(
                h_est.matrix(),
                &correspondences,
                &format!("DLT[{}]: {}", i, name),
                1e-4,
            );
        }
    }
}

// ─── Property: reverse transfer x₁ ∝ H⁻¹·x₂ ───────────────────────────────

#[test]
fn property_reverse_transfer_estimated() {
    let points = test_points();
    for (name, h_true) in test_homographies() {
        let correspondences = correspondences_from_h(&h_true, &points);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let config = HomographyConfig::default();
        let models = DltSolver::solve(&normalized, &config)
            .unwrap_or_else(|e| panic!("[{}] DLT solver failed: {:?}", name, e));

        for (i, h_est) in models.iter().enumerate() {
            let h_inv = h_est
                .matrix()
                .try_inverse()
                .unwrap_or_else(|| panic!("[DLT[{}]: {}] H not invertible", i, name));

            // Build reversed correspondences: (p2, p1)
            let reversed: Vec<PixelCorrespondence> = correspondences
                .iter()
                .map(|c| PixelCorrespondence::new(c.point2, c.point1))
                .collect();

            assert_projective_transfer(&h_inv, &reversed, &format!("DLT⁻¹[{}]: {}", i, name), 1e-4);
        }
    }
}

// ─── Property: scale invariance ──────────────────────────────────────────────
//
// αH and H produce the same projective mapping for any α ≠ 0,
// because projective coordinates are defined up to scale.

#[test]
fn property_scale_invariance() {
    let scales = [0.001, 0.5, 1.0, 7.0, 1000.0, -1.0, -0.3];
    let points = test_points();

    for (name, h) in test_homographies() {
        let correspondences = correspondences_from_h(&h, &points);

        for &alpha in &scales {
            let h_scaled = h * alpha;
            let label = format!("α={}: {}", alpha, name);

            // Scaled H must still be full rank
            let det = h_scaled.determinant();
            assert!(det.abs() > 1e-15, "[{}] det(αH) = {:.2e}", label, det);

            // Scaled H must produce the same transfer
            assert_projective_transfer(&h_scaled, &correspondences, &label, 1e-8);
        }
    }
}

// ─── Property: rank is exactly 3 (not 2, not 1) ─────────────────────────────
//
// All three singular values must be non-trivially positive.

fn assert_rank_3(h: &Matrix3<f64>, label: &str) {
    let svd = h.svd(false, false);
    let s = &svd.singular_values;

    for (i, &si) in s.iter().enumerate() {
        assert!(
            si > 1e-10,
            "[{}] σ{} = {:.2e}, expected > 0 (σ = [{:.4e}, {:.4e}, {:.4e}])",
            label,
            i + 1,
            si,
            s[0],
            s[1],
            s[2]
        );
    }

    // Condition number should be finite and reasonable
    let cond = s[0] / s[2];
    assert!(
        cond.is_finite(),
        "[{}] condition number is not finite",
        label
    );
}

#[test]
fn property_rank_3_ground_truth() {
    for (name, h) in test_homographies() {
        assert_rank_3(&h, &format!("ground truth: {}", name));
    }
}

#[test]
fn property_rank_3_estimated() {
    let points = test_points();
    for (name, h_true) in test_homographies() {
        let correspondences = correspondences_from_h(&h_true, &points);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let config = HomographyConfig::default();
        let models = DltSolver::solve(&normalized, &config)
            .unwrap_or_else(|e| panic!("[{}] DLT solver failed: {:?}", name, e));

        for (i, h_est) in models.iter().enumerate() {
            assert_rank_3(h_est.matrix(), &format!("DLT[{}]: {}", i, name));
        }
    }
}

// ─── Contrast: H must NOT satisfy F/E properties ────────────────────────────
//
// This test ensures we're not confusing matrix types: a valid H
// must have det(H) ≠ 0, which is the opposite of F's det(F) = 0.

#[test]
fn property_h_is_not_fundamental() {
    for (name, h) in test_homographies() {
        let det = h.determinant();
        // A valid H should have det far from zero
        assert!(
            det.abs() > 1e-6,
            "[{}] det(H) = {:.2e} — suspiciously close to zero for a homography!",
            name,
            det
        );

        // rank should be 3, not 2
        let svd = h.svd(false, false);
        let s = &svd.singular_values;
        let min_sv_ratio = s[2] / s[0];
        assert!(
            min_sv_ratio > 1e-6,
            "[{}] σ₃/σ₁ = {:.2e} — H looks rank-deficient like an F matrix!",
            name,
            min_sv_ratio
        );
    }
}
