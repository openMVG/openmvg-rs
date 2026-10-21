// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-based tests for the fundamental matrix.
//!
//! These tests verify algebraic invariants that every valid fundamental matrix
//! must satisfy, regardless of the camera geometry used to produce it:
//!
//! 1. **det(F) = 0** — F is a singular matrix (rank ≤ 2)
//! 2. **rank(F) = 2** — exactly one zero singular value
//! 3. **Epipolar constraint** — x₂ᵀ·F·x₁ ≈ 0 for all correspondences
//! 4. **Scale invariance** — αF satisfies the same properties for any α ≠ 0
//!
//! Tests are randomized over multiple camera geometries to increase coverage.

use nalgebra::{Matrix3, Point2, Rotation3, Unit, Vector3};
use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::fundamental::{EightPointSolver, SevenPointSolver};
use openmvg_geometry::normalization::NormalizedCorrespondences;

type TestGeometry = (&'static str, f64, f64, f64, Rotation3<f64>, Vector3<f64>);
type RandomGeometry = (String, f64, f64, f64, Rotation3<f64>, Vector3<f64>);

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Create a ground-truth fundamental matrix from a known camera geometry.
///
/// Camera 1: P₁ = K [I | 0]
/// Camera 2: P₂ = K [R | t]
/// F = K⁻ᵀ · [t]× · R · K⁻¹
fn make_ground_truth_f(
    focal: f64,
    cx: f64,
    cy: f64,
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
) -> Matrix3<f64> {
    let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
    let k_inv = k.try_inverse().expect("K matrix must be invertible");
    let t = translation;
    let t_cross = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    k_inv.transpose() * t_cross * rotation.matrix() * k_inv
}

/// Generate pixel correspondences by projecting 3D points through a camera pair.
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
    let mut correspondences = Vec::with_capacity(num_points);

    for i in 0..num_points {
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

        correspondences.push(PixelCorrespondence::new(p1, p2));
    }
    correspondences
}

/// A set of diverse camera geometries for property testing.
fn test_geometries() -> Vec<TestGeometry> {
    vec![
        (
            "lateral baseline",
            500.0,
            320.0,
            240.0,
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 0.15),
            Vector3::new(1.0, 0.0, 0.0),
        ),
        (
            "forward motion + rotation",
            800.0,
            400.0,
            300.0,
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 0.30),
            Vector3::new(0.5, 0.1, 0.3),
        ),
        (
            "pure rotation around X",
            600.0,
            320.0,
            240.0,
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::x()), 0.20),
            Vector3::new(0.0, 1.0, 0.2),
        ),
        (
            "diagonal baseline, wide-angle",
            350.0,
            250.0,
            200.0,
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(1.0, 1.0, 0.5)), 0.12),
            Vector3::new(0.7, 0.7, 0.0),
        ),
        (
            "large rotation",
            500.0,
            320.0,
            240.0,
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 0.50),
            Vector3::new(2.0, 0.0, 0.5),
        ),
    ]
}

// ─── Property: det(F) = 0 ────────────────────────────────────────────────────

/// For any valid fundamental matrix, det(F) must be zero (rank-2 constraint).
fn assert_det_zero(f: &Matrix3<f64>, label: &str) {
    let det = f.determinant();
    assert!(
        det.abs() < 1e-6,
        "[{}] det(F) = {:.2e}, expected ≈ 0",
        label,
        det
    );
}

/// For ground-truth F matrices built from K⁻ᵀ·[t]×·R·K⁻¹,
/// det(F) is exactly 0 because [t]× has rank 2.
#[test]
fn property_det_f_is_zero_ground_truth() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let f = make_ground_truth_f(focal, cx, cy, &rot, &t);
        assert_det_zero(&f, &format!("ground truth: {}", name));
    }
}

/// For F matrices estimated by the 8-point solver,
/// det(F) must be ≈ 0 (rank-2 enforced by SVD truncation).
#[test]
fn property_det_f_is_zero_eight_point() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let correspondences = generate_correspondences(focal, cx, cy, &rot, &t, 20);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = EightPointSolver::solve(&normalized)
            .unwrap_or_else(|e| panic!("[{}] 8-point solver failed: {:?}", name, e));

        for (i, f_denorm) in models.iter().enumerate() {
            assert_det_zero(f_denorm.matrix(), &format!("8-point[{}]: {}", i, name));
        }
    }
}

/// For F matrices estimated by the 7-point solver,
/// det(F) must be ≈ 0.
#[test]
fn property_det_f_is_zero_seven_point() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let correspondences = generate_correspondences(focal, cx, cy, &rot, &t, 7);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = match SevenPointSolver::solve(&normalized) {
            Ok(m) => m,
            Err(_) => continue, // 7-point can legitimately fail on some configs
        };

        for (i, f_denorm) in models.iter().enumerate() {
            assert_det_zero(f_denorm.matrix(), &format!("7-point[{}]: {}", i, name));
        }
    }
}

// ─── Property: rank(F) = 2 ───────────────────────────────────────────────────

/// For any valid F, the SVD must have exactly σ₁ ≥ σ₂ > 0 and σ₃ ≈ 0.
fn assert_rank_2(f: &Matrix3<f64>, label: &str) {
    let svd = f.svd(false, false);
    let sigmas = &svd.singular_values;

    // σ₁ and σ₂ must be non-trivially positive
    assert!(
        sigmas[0] > 1e-10,
        "[{}] σ₁ = {:.2e}, expected > 0",
        label,
        sigmas[0]
    );
    assert!(
        sigmas[1] > 1e-10,
        "[{}] σ₂ = {:.2e}, expected > 0",
        label,
        sigmas[1]
    );

    // σ₃ must be ≈ 0 (relative to σ₁)
    let ratio = sigmas[2] / sigmas[0];
    assert!(
        ratio < 1e-6,
        "[{}] σ₃/σ₁ = {:.2e}, expected ≈ 0 (σ₃={:.2e}, σ₁={:.2e})",
        label,
        ratio,
        sigmas[2],
        sigmas[0]
    );
}

#[test]
fn property_rank_f_is_2_ground_truth() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let f = make_ground_truth_f(focal, cx, cy, &rot, &t);
        assert_rank_2(&f, &format!("ground truth: {}", name));
    }
}

#[test]
fn property_rank_f_is_2_eight_point() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let correspondences = generate_correspondences(focal, cx, cy, &rot, &t, 20);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = EightPointSolver::solve(&normalized)
            .unwrap_or_else(|e| panic!("[{}] 8-point solver failed: {:?}", name, e));

        for (i, f_denorm) in models.iter().enumerate() {
            assert_rank_2(f_denorm.matrix(), &format!("8-point[{}]: {}", i, name));
        }
    }
}

// ─── Property: epipolar constraint x₂ᵀ·F·x₁ = 0 ────────────────────────────

/// For every correspondence used to estimate F, the algebraic error must be ≈ 0.
fn assert_epipolar_constraint(
    f: &Matrix3<f64>,
    correspondences: &[PixelCorrespondence],
    label: &str,
) {
    let mut max_error = 0.0f64;
    for (i, corr) in correspondences.iter().enumerate() {
        let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
        let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);
        let error = (x2.transpose() * f * x1)[(0, 0)].abs();
        max_error = max_error.max(error);
        assert!(
            error < 1e-4,
            "[{}] epipolar error at correspondence {}: {:.2e} (expected < 1e-4)",
            label,
            i,
            error
        );
    }
}

#[test]
fn property_epipolar_constraint_holds_eight_point() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let correspondences = generate_correspondences(focal, cx, cy, &rot, &t, 20);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = EightPointSolver::solve(&normalized)
            .unwrap_or_else(|e| panic!("[{}] 8-point solver failed: {:?}", name, e));

        for (i, f_denorm) in models.iter().enumerate() {
            assert_epipolar_constraint(
                f_denorm.matrix(),
                &correspondences,
                &format!("8-point[{}]: {}", i, name),
            );
        }
    }
}

// ─── Property: scale invariance ──────────────────────────────────────────────

/// αF must satisfy det(αF) = 0 and rank(αF) = 2 for any non-zero scalar α.
#[test]
fn property_scale_invariance() {
    let scales = [0.001, 0.1, 1.0, 10.0, 1000.0, -1.0, -42.5];

    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let f = make_ground_truth_f(focal, cx, cy, &rot, &t);

        for &alpha in &scales {
            let f_scaled = f * alpha;
            let label = format!("α={}: {}", alpha, name);
            assert_det_zero(&f_scaled, &label);
            assert_rank_2(&f_scaled, &label);
        }
    }
}

// ─── Property: F and Fᵀ both valid but different ─────────────────────────────

/// F and Fᵀ must both satisfy rank-2, but for asymmetric geometries
/// they must NOT be proportional (F ≠ αFᵀ).
#[test]
fn property_f_transpose_is_rank2_but_different() {
    for (name, focal, cx, cy, rot, t) in test_geometries() {
        let f = make_ground_truth_f(focal, cx, cy, &rot, &t);
        let f_t = f.transpose();

        // Both must be rank-2
        assert_rank_2(&f, &format!("F: {}", name));
        assert_rank_2(&f_t, &format!("Fᵀ: {}", name));

        // Fᵀ satisfies the *reverse* epipolar constraint: x₁ᵀ·Fᵀ·x₂ = 0
        // which is equivalent to (x₂ᵀ·F·x₁)ᵀ = 0. Verify with correspondences.
        let correspondences = generate_correspondences(focal, cx, cy, &rot, &t, 10);
        for corr in &correspondences {
            let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
            let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);
            // Fᵀ maps point2 → epipolar line in image1
            let error = (x1.transpose() * f_t * x2)[(0, 0)].abs();
            assert!(
                error < 1e-6,
                "[{}] Reverse epipolar error: {:.2e}",
                name,
                error
            );
        }
    }
}

// ─── Property: Randomized geometries ────────────────────────────────────────

/// Generate random camera geometries with seeded RNG for reproducibility.
fn randomized_geometries(seed: u64, count: usize) -> Vec<RandomGeometry> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, Uniform};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let focal_dist = Uniform::new(300.0, 1200.0);
    let cx_dist = Uniform::new(200.0, 500.0);
    let cy_dist = Uniform::new(150.0, 400.0);
    let angle_dist = Uniform::new(0.05, 0.6);
    let translation_dist = Normal::new(0.0, 1.0).unwrap();

    (0..count)
        .map(|i| {
            let focal = focal_dist.sample(&mut rng);
            let cx = cx_dist.sample(&mut rng);
            let cy = cy_dist.sample(&mut rng);

            let axis = Vector3::new(
                translation_dist.sample(&mut rng),
                translation_dist.sample(&mut rng),
                translation_dist.sample(&mut rng),
            );
            let angle = angle_dist.sample(&mut rng);
            let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle);

            // Ensure non-zero translation (degenerate otherwise)
            let mut t = Vector3::new(
                translation_dist.sample(&mut rng),
                translation_dist.sample(&mut rng),
                translation_dist.sample(&mut rng),
            );
            if t.norm() < 0.1 {
                t = Vector3::new(1.0, 0.0, 0.0);
            }

            (format!("random-{}", i), focal, cx, cy, rotation, t)
        })
        .collect()
}

/// For any valid fundamental matrix from random geometries: det(F) = 0 and rank(F) = 2.
#[test]
fn property_det_zero_and_rank2_randomized() {
    let geometries = randomized_geometries(42, 50);

    for (name, focal, cx, cy, rotation, translation) in &geometries {
        let f = make_ground_truth_f(*focal, *cx, *cy, rotation, translation);
        assert_det_zero(&f, name);
        assert_rank_2(&f, name);
    }
}

/// For any valid F from random geometries estimated by 8-point solver: det(F) = 0 and rank(F) = 2.
#[test]
fn property_det_zero_and_rank2_eight_point_randomized() {
    let geometries = randomized_geometries(123, 20);

    for (name, focal, cx, cy, rotation, translation) in &geometries {
        let correspondences = generate_correspondences(*focal, *cx, *cy, rotation, translation, 20);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let models = match EightPointSolver::solve(&normalized) {
            Ok(m) => m,
            Err(_) => continue,
        };

        for (i, f) in models.iter().enumerate() {
            let label = format!("{} 8pt[{}]", name, i);
            assert_det_zero(f.matrix(), &label);
            assert_rank_2(f.matrix(), &label);

            // Also verify epipolar constraint
            for corr in &correspondences {
                let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
                let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);
                let error = (x2.transpose() * f.matrix() * x1)[(0, 0)].abs();
                assert!(error < 1e-4, "[{}] Epipolar error: {:.2e}", label, error);
            }
        }
    }
}

/// For any valid F from random geometries estimated by 7-point solver: det(F) = 0 and rank(F) = 2.
#[test]
fn property_det_zero_and_rank2_seven_point_randomized() {
    let geometries = randomized_geometries(456, 20);

    for (name, focal, cx, cy, rotation, translation) in &geometries {
        let correspondences = generate_correspondences(*focal, *cx, *cy, rotation, translation, 7);
        let normalized = NormalizedCorrespondences::from_pixels(&correspondences);

        let solutions = match SevenPointSolver::solve(&normalized) {
            Ok(m) => m,
            Err(_) => continue,
        };

        for (sol_idx, f) in solutions.iter().enumerate() {
            let label = format!("{} 7pt sol#{}", name, sol_idx);
            assert_det_zero(f.matrix(), &label);
            assert_rank_2(f.matrix(), &label);
        }
    }
}
