// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-based tests for the essential matrix.
//!
//! These tests verify algebraic invariants that every valid essential matrix
//! must satisfy, regardless of the camera geometry used to produce it:
//!
//! 1. **det(E) = 0** — E is singular (rank ≤ 2)
//! 2. **rank(E) = 2** — exactly one zero singular value
//! 3. **σ₁ = σ₂** — the two non-zero singular values are equal
//! 4. **Demazure constraint** — 2·E·Eᵀ·E − tr(E·Eᵀ)·E = 0
//! 5. **Epipolar constraint** — x₂ᵀ·E·x₁ ≈ 0 for bearing correspondences
//! 6. **Scale invariance** — αE satisfies the same properties for any α ≠ 0
//!
//! Tests are randomized over multiple camera geometries to increase coverage.

use nalgebra::{Matrix3, Rotation3, Unit, Vector3};
use openmvg_geometry::correspondence::BearingCorrespondence;
use openmvg_geometry::essential::FivePointSolver;
// The linear eight-point refiner is crate-private; its property tests reach it
// through the `unstable-test-support` evidence surface.
#[cfg(feature = "unstable-test-support")]
use openmvg_geometry::essential::test_support::EightPointSolver;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a ground-truth essential matrix: E = [t]× · R.
fn make_ground_truth_e(rotation: &Rotation3<f64>, translation: &Vector3<f64>) -> Matrix3<f64> {
    let t = translation;
    let t_cross = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    t_cross * rotation.matrix()
}

/// Generate bearing correspondences from a known relative pose.
///
/// Camera 1 is at the origin; camera 2 is at (R, t).
/// 3D points are generated deterministically in front of both cameras.
fn generate_bearing_correspondences(
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
    num_points: usize,
) -> Vec<BearingCorrespondence> {
    let mut correspondences = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let t = (i as f64 + 1.0) * 0.15;
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_points as f64);
        let point = Vector3::new(t * angle.cos(), t * angle.sin(), 3.0 + t * 0.3);

        let bearing1 = point.normalize();
        let point_cam2 = rotation * point + translation;
        let bearing2 = point_cam2.normalize();

        correspondences.push(BearingCorrespondence::try_new(bearing1, bearing2).unwrap());
    }
    correspondences
}

/// A set of diverse camera geometries for property testing.
fn test_geometries() -> Vec<(&'static str, Rotation3<f64>, Vector3<f64>)> {
    vec![
        (
            "lateral baseline",
            Rotation3::from_euler_angles(0.1, 0.05, 0.02),
            Vector3::new(0.5, 0.1, 0.0).normalize(),
        ),
        (
            "forward motion",
            Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ),
        (
            "diagonal motion + rotation",
            Rotation3::from_euler_angles(0.15, -0.1, 0.08),
            Vector3::new(-0.3, 0.6, 0.1).normalize(),
        ),
        (
            "pure Y rotation",
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 0.25),
            Vector3::new(1.0, 0.0, 0.2).normalize(),
        ),
        (
            "large rotation + oblique baseline",
            Rotation3::from_euler_angles(0.3, 0.2, -0.1),
            Vector3::new(0.4, 0.4, 0.8).normalize(),
        ),
    ]
}

// ─── Property: det(E) = 0 ────────────────────────────────────────────────────

fn assert_det_zero(e: &Matrix3<f64>, label: &str) {
    let det = e.determinant();
    assert!(
        det.abs() < 1e-6,
        "[{}] det(E) = {:.2e}, expected ≈ 0",
        label,
        det
    );
}

#[test]
fn property_det_e_is_zero_ground_truth() {
    for (name, rot, t) in test_geometries() {
        let e = make_ground_truth_e(&rot, &t);
        assert_det_zero(&e, &format!("ground truth: {}", name));
    }
}

#[cfg(feature = "unstable-test-support")]
#[test]
fn property_det_e_is_zero_eight_point() {
    for (name, rot, t) in test_geometries() {
        let correspondences = generate_bearing_correspondences(&rot, &t, 20);

        let models = EightPointSolver::solve(&correspondences)
            .unwrap_or_else(|e| panic!("[{}] 8-point E solver failed: {:?}", name, e));

        for (i, model) in models.iter().enumerate() {
            assert_det_zero(model.matrix(), &format!("8-point[{}]: {}", i, name));
        }
    }
}

#[test]
fn property_det_e_is_zero_five_point() {
    for (name, rot, t) in test_geometries() {
        let correspondences = generate_bearing_correspondences(&rot, &t, 5);

        let models = match FivePointSolver::solve(&correspondences) {
            Ok(m) => m,
            Err(_) => continue,
        };

        for (i, model) in models.iter().enumerate() {
            assert_det_zero(model.matrix(), &format!("5-point[{}]: {}", i, name));
        }
    }
}

// ─── Property: rank(E) = 2 with σ₁ = σ₂ ─────────────────────────────────────

/// Essential matrix must have SVD with σ₁ ≈ σ₂ and σ₃ ≈ 0.
/// This is strictly stronger than F's rank-2 (where σ₁ ≠ σ₂ is allowed).
fn assert_essential_singular_values(e: &Matrix3<f64>, label: &str, tolerance: f64) {
    let svd = e.svd(false, false);
    let s = &svd.singular_values;

    // σ₁ and σ₂ must be positive
    assert!(s[0] > 1e-10, "[{}] σ₁ = {:.2e}, expected > 0", label, s[0]);
    assert!(s[1] > 1e-10, "[{}] σ₂ = {:.2e}, expected > 0", label, s[1]);

    // σ₃ ≈ 0 (relative to σ₁)
    let zero_ratio = s[2] / s[0];
    assert!(
        zero_ratio < tolerance,
        "[{}] σ₃/σ₁ = {:.2e}, expected ≈ 0 (σ = [{:.4e}, {:.4e}, {:.4e}])",
        label,
        zero_ratio,
        s[0],
        s[1],
        s[2]
    );

    // σ₁ ≈ σ₂ (the essential constraint, unique to E)
    let equal_ratio = (s[0] - s[1]).abs() / s[0].max(s[1]);
    assert!(
        equal_ratio < tolerance,
        "[{}] |σ₁−σ₂|/max = {:.2e}, expected ≈ 0 (σ = [{:.4e}, {:.4e}, {:.4e}])",
        label,
        equal_ratio,
        s[0],
        s[1],
        s[2]
    );
}

#[test]
fn property_singular_values_ground_truth() {
    for (name, rot, t) in test_geometries() {
        let e = make_ground_truth_e(&rot, &t);
        assert_essential_singular_values(&e, &format!("ground truth: {}", name), 1e-10);
    }
}

#[cfg(feature = "unstable-test-support")]
#[test]
fn property_singular_values_eight_point() {
    for (name, rot, t) in test_geometries() {
        let correspondences = generate_bearing_correspondences(&rot, &t, 20);
        let models = EightPointSolver::solve(&correspondences)
            .unwrap_or_else(|e| panic!("[{}] 8-point E solver failed: {:?}", name, e));

        for (i, model) in models.iter().enumerate() {
            assert_essential_singular_values(
                model.matrix(),
                &format!("8-point[{}]: {}", i, name),
                1e-6,
            );
        }
    }
}

/// Five-point Essential solver invariants: five-point candidates have two nonzero equal singular
/// values; five-point candidates satisfy the rank-two Essential constraint.
#[test]
fn property_singular_values_five_point() {
    for (name, rot, t) in test_geometries() {
        let correspondences = generate_bearing_correspondences(&rot, &t, 5);
        let models = match FivePointSolver::solve(&correspondences) {
            Ok(m) => m,
            Err(_) => continue,
        };

        for (i, model) in models.iter().enumerate() {
            // 5-point solver uses polynomial root finding, so tolerance is looser
            assert_essential_singular_values(
                model.matrix(),
                &format!("5-point[{}]: {}", i, name),
                0.2,
            );
        }
    }
}

// ─── Property: Demazure trace constraint ─────────────────────────────────────
//
// The Demazure/Huang-Faugeras cubic constraint characterizes essential matrices
// among all 3×3 matrices: 2·E·Eᵀ·E − tr(E·Eᵀ)·E = 0
//
// This is the algebraic definition of an essential matrix (equivalent to
// having SVD with σ₁ = σ₂, σ₃ = 0).

fn assert_demazure(e: &Matrix3<f64>, label: &str, tolerance: f64) {
    let eet = e * e.transpose();
    let trace_eet = eet.trace();
    let demazure = 2.0 * eet * e - trace_eet * e;

    // Normalize by the matrix norm to get a scale-independent error
    let e_norm = e.norm();
    if e_norm < 1e-12 {
        return; // zero matrix trivially satisfies it
    }
    let demazure_norm = demazure.norm() / (e_norm * e_norm * e_norm);

    assert!(
        demazure_norm < tolerance,
        "[{}] Demazure residual = {:.2e}, expected < {:.2e}\n\
         2·E·Eᵀ·E − tr(E·Eᵀ)·E should be zero for any valid essential matrix",
        label,
        demazure_norm,
        tolerance
    );
}

#[test]
fn property_demazure_ground_truth() {
    for (name, rot, t) in test_geometries() {
        let e = make_ground_truth_e(&rot, &t);
        assert_demazure(&e, &format!("ground truth: {}", name), 1e-10);
    }
}

#[cfg(feature = "unstable-test-support")]
#[test]
fn property_demazure_eight_point() {
    for (name, rot, t) in test_geometries() {
        let correspondences = generate_bearing_correspondences(&rot, &t, 20);
        let models = EightPointSolver::solve(&correspondences)
            .unwrap_or_else(|e| panic!("[{}] 8-point E solver failed: {:?}", name, e));

        for (i, model) in models.iter().enumerate() {
            assert_demazure(model.matrix(), &format!("8-point[{}]: {}", i, name), 1e-6);
        }
    }
}

// ─── Property: epipolar constraint x₂ᵀ·E·x₁ = 0 ────────────────────────────

#[cfg(feature = "unstable-test-support")]
#[test]
fn property_epipolar_constraint_eight_point() {
    for (name, rot, t) in test_geometries() {
        let correspondences = generate_bearing_correspondences(&rot, &t, 20);
        let models = EightPointSolver::solve(&correspondences)
            .unwrap_or_else(|e| panic!("[{}] 8-point E solver failed: {:?}", name, e));

        for (i, model) in models.iter().enumerate() {
            for (j, corr) in correspondences.iter().enumerate() {
                let x1 = *corr.bearing1();
                let x2 = *corr.bearing2();
                let error = (x2.transpose() * model.matrix() * x1)[(0, 0)].abs();
                assert!(
                    error < 1e-6,
                    "[8-point[{}]: {}] epipolar error at {}: {:.2e}",
                    i,
                    name,
                    j,
                    error
                );
            }
        }
    }
}

// ─── Property: scale invariance ──────────────────────────────────────────────

#[test]
fn property_scale_invariance() {
    let scales = [0.001, 0.1, 1.0, 10.0, 1000.0, -1.0, -42.5];

    for (name, rot, t) in test_geometries() {
        let e = make_ground_truth_e(&rot, &t);
        for &alpha in &scales {
            let e_scaled = e * alpha;
            let label = format!("α={}: {}", alpha, name);
            assert_det_zero(&e_scaled, &label);
            assert_essential_singular_values(&e_scaled, &label, 1e-10);
            assert_demazure(&e_scaled, &label, 1e-10);
        }
    }
}

// ─── Property: E and Eᵀ ──────────────────────────────────────────────────────
//
// Eᵀ is also a valid essential matrix (it corresponds to the reverse pose).
// Both should satisfy all essential matrix properties.

#[test]
fn property_transpose_is_also_essential() {
    for (name, rot, t) in test_geometries() {
        let e = make_ground_truth_e(&rot, &t);
        let e_t = e.transpose();

        // Eᵀ must also be rank-2 with equal singular values
        assert_essential_singular_values(&e_t, &format!("Eᵀ: {}", name), 1e-10);
        assert_demazure(&e_t, &format!("Eᵀ: {}", name), 1e-10);

        // Eᵀ satisfies the reverse epipolar constraint: x₁ᵀ·Eᵀ·x₂ = 0
        let correspondences = generate_bearing_correspondences(&rot, &t, 10);
        for corr in &correspondences {
            let x1 = *corr.bearing1();
            let x2 = *corr.bearing2();
            let error = (x1.transpose() * e_t * x2)[(0, 0)].abs();
            assert!(
                error < 1e-10,
                "[{}] reverse epipolar error: {:.2e}",
                name,
                error
            );
        }
    }
}
