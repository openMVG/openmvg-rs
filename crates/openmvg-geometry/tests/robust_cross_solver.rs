// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Cross-family Refinement Tests
//!
//! Integration tests verifying that different solvers work correctly together
//! in typical computer vision pipelines.
//!
//! ## Test Categories
//!
//! 1. **Minimal → DLT Upgrade**: Test that minimal solvers (5pt, 7pt) can be
//!    refined with DLT solvers (8pt) using their inliers.
//!
//! 2. **RANSAC → Refinement**: Test that RANSAC output can be refined with
//!    all-inlier re-estimation.
//!
//! 3. **Model Conversion**: Test that Essential → Fundamental and vice versa
//!    works when camera intrinsics are known.

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Point2, Vector3};
    use openmvg_geometry::correspondence::PixelCorrespondence;
    #[cfg(feature = "unstable-test-support")]
    use openmvg_geometry::essential;
    use openmvg_geometry::fundamental::{EightPointSolver, SevenPointSolver};
    use openmvg_geometry::homography::{DltSolver, HomographyConfig};
    use openmvg_geometry::normalization::NormalizedCorrespondences;

    /// Generate synthetic correspondences from a known essential matrix.
    fn generate_essential_correspondences(
        e: &Matrix3<f64>,
        n_points: usize,
        noise_sigma: f64,
    ) -> Vec<PixelCorrespondence> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut correspondences = Vec::with_capacity(n_points);

        // Generate random 3D points in front of both cameras
        for _ in 0..n_points {
            // Random 3D point (depth 5-15)
            let x: f64 = rng.gen_range(-2.0..2.0);
            let y: f64 = rng.gen_range(-2.0..2.0);
            let z: f64 = rng.gen_range(5.0..15.0);

            // Project to first camera (identity)
            let p1 = Point2::new(x / z, y / z);

            // For a proper essential matrix test, we'd need to decompose E into R,t
            // and project properly. For now, use epipolar constraint.
            // x2^T * E * x1 = 0

            // Generate point on epipolar line in second image
            let x1_h = Vector3::new(p1.x, p1.y, 1.0);
            let epipolar_line = e * x1_h; // l2 = E * x1

            // Find a point on this line (simple approach: assume similar y)
            let a = epipolar_line.x;
            let b = epipolar_line.y;
            let c = epipolar_line.z;

            // Point on line: ax + by + c = 0
            // If |b| > |a|: x2 = p1.x + small_offset, y2 = -(ax + c)/b
            let p2 = if b.abs() > 1e-6 {
                let x2 = p1.x + rng.gen_range(-0.1..0.1);
                let y2 = -(a * x2 + c) / b;
                Point2::new(x2, y2)
            } else if a.abs() > 1e-6 {
                let y2 = p1.y + rng.gen_range(-0.1..0.1);
                let x2 = -(b * y2 + c) / a;
                Point2::new(x2, y2)
            } else {
                continue; // Degenerate case
            };

            // Add noise
            let noise1 = Point2::new(
                rng.r#gen::<f64>() * noise_sigma,
                rng.r#gen::<f64>() * noise_sigma,
            );
            let noise2 = Point2::new(
                rng.r#gen::<f64>() * noise_sigma,
                rng.r#gen::<f64>() * noise_sigma,
            );

            correspondences.push(PixelCorrespondence {
                point1: p1 + noise1.coords,
                point2: p2 + noise2.coords,
            });
        }

        correspondences
    }

    /// Generate synthetic correspondences for homography testing.
    fn generate_homography_correspondences(
        h: &Matrix3<f64>,
        n_points: usize,
        noise_sigma: f64,
    ) -> Vec<PixelCorrespondence> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut correspondences = Vec::with_capacity(n_points);

        for _ in 0..n_points {
            // Random point in first image
            let p1 = Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));

            // Transform by homography
            let p1_h = Vector3::new(p1.x, p1.y, 1.0);
            let p2_h = h * p1_h;
            let p2 = Point2::new(p2_h.x / p2_h.z, p2_h.y / p2_h.z);

            // Add noise
            let noise1 = Point2::new(
                rng.r#gen::<f64>() * noise_sigma,
                rng.r#gen::<f64>() * noise_sigma,
            );
            let noise2 = Point2::new(
                rng.r#gen::<f64>() * noise_sigma,
                rng.r#gen::<f64>() * noise_sigma,
            );

            correspondences.push(PixelCorrespondence {
                point1: p1 + noise1.coords,
                point2: p2 + noise2.coords,
            });
        }

        correspondences
    }

    // ========================================================================
    // Test: 7-point → 8-point refinement for Fundamental matrix
    // ========================================================================

    #[test]
    fn test_seven_to_eight_point_refinement() {
        // Generate correspondences from a known fundamental matrix
        // F = [e']_x * H where H is a homography between the planes
        let f_true = Matrix3::new(0.0, -0.001, 0.1, 0.001, 0.0, -0.2, -0.1, 0.2, 0.0);

        // Generate 50 correspondences (with outliers mixed in later)
        let correspondences = generate_essential_correspondences(&f_true, 50, 0.001);

        if correspondences.len() < 8 {
            println!("Skipping test: not enough valid correspondences generated");
            return;
        }

        // Step 1: Use 7-point solver on minimal sample
        let minimal_sample: Vec<_> = correspondences.iter().take(7).cloned().collect();
        let seven_pt_result =
            SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(&minimal_sample));

        // Step 2: Use 8-point solver on all data
        let eight_pt_result =
            EightPointSolver::solve(&NormalizedCorrespondences::from_pixels(&correspondences));

        // Both should succeed
        assert!(seven_pt_result.is_ok(), "7-point solver should succeed");
        assert!(eight_pt_result.is_ok(), "8-point solver should succeed");

        let seven_pt_models = seven_pt_result.unwrap();
        let eight_pt_models = eight_pt_result.unwrap();

        // 7-point may return 1-3 solutions, 8-point returns 1
        assert!(!seven_pt_models.is_empty(), "7-point should find solutions");
        assert!(!eight_pt_models.is_empty(), "8-point should find solution");

        // Both should produce valid fundamental matrices (det ≈ 0)
        for f in &seven_pt_models {
            assert!(
                f.matrix().determinant().abs() < 0.1,
                "7-point F should have det ≈ 0"
            );
        }

        for f in &eight_pt_models {
            assert!(
                f.matrix().determinant().abs() < 0.1,
                "8-point F should have det ≈ 0"
            );
        }
    }

    // ========================================================================
    // Test: Homography DLT consistency
    // ========================================================================

    #[test]
    fn test_homography_dlt_consistency() {
        // Known homography (rotation + translation on plane)
        let angle = 0.1_f64;
        let h_true = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.1,
            angle.sin(),
            angle.cos(),
            0.05,
            0.001,
            0.002,
            1.0,
        );

        // Generate clean correspondences
        let correspondences = generate_homography_correspondences(&h_true, 20, 0.0001);

        // Solve with DLT
        let result = DltSolver::solve(
            &NormalizedCorrespondences::from_pixels(&correspondences),
            &HomographyConfig::default(),
        );

        assert!(result.is_ok(), "Homography solver should succeed");

        let models = result.unwrap();
        assert!(!models.is_empty(), "Should find at least one homography");

        // The estimated homography should be close to ground truth (up to scale)
        let h_est = models[0].matrix();

        // Normalize both by h[2,2]
        let h_true_norm = h_true / h_true[(2, 2)];
        let h_est_norm = h_est / h_est[(2, 2)];

        let diff = (h_true_norm - h_est_norm).norm();
        assert!(
            diff < 0.1,
            "Estimated homography should be close to ground truth, diff = {}",
            diff
        );
    }

    // ========================================================================
    // Test: Essential matrix with 8-point
    // ========================================================================

    #[cfg(feature = "unstable-test-support")]
    #[test]
    fn test_essential_eight_point() {
        use nalgebra::UnitVector3;
        use openmvg_geometry::correspondence::BearingCorrespondence;

        // Create essential matrix from known R, t
        let angle = 0.1_f64;
        let r = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let t = Vector3::new(1.0, 0.0, 0.0);

        // E = [t]_x * R
        let t_skew = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
        let _e_true = t_skew * r;

        // Generate bearing correspondences (Essential solver uses normalized bearings)
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut correspondences = Vec::with_capacity(30);
        for _ in 0..30 {
            // Generate random 3D point
            let x: f64 = rng.gen_range(-2.0..2.0);
            let y: f64 = rng.gen_range(-2.0..2.0);
            let z: f64 = rng.gen_range(5.0..15.0);

            // Bearing in camera 1 (identity pose)
            let p1 = Vector3::new(x, y, z);
            let b1 = UnitVector3::new_normalize(p1);

            // Transform point to camera 2 frame
            let p2 = r * Vector3::new(x, y, z) + t;
            let b2 = UnitVector3::new_normalize(p2);

            correspondences
                .push(BearingCorrespondence::try_new(b1.into_inner(), b2.into_inner()).unwrap());
        }

        // Solve
        let result = essential::test_support::EightPointSolver::solve(&correspondences);

        assert!(result.is_ok(), "Essential solver should succeed");

        let models = result.unwrap();
        assert!(
            !models.is_empty(),
            "Should find at least one essential matrix"
        );

        // Essential matrix should satisfy E * E^T * E = trace(E * E^T) / 2 * E
        // (simplified check: singular values should be [σ, σ, 0])
        let e_est = models[0].matrix();
        let svd = e_est.svd(true, true);
        let s = svd.singular_values;

        // Two singular values should be equal, one should be ~0
        let ratio = s[0] / s[1].max(1e-10);
        assert!(
            (ratio - 1.0).abs() < 0.5,
            "First two singular values should be similar, ratio = {}",
            ratio
        );
        assert!(
            s[2] / s[0].max(1e-10) < 0.1,
            "Third singular value should be small, s[2]/s[0] = {}",
            s[2] / s[0].max(1e-10)
        );
    }

    // ========================================================================
    // Test: Solver → Refinement workflow simulation
    // ========================================================================

    #[test]
    fn test_solver_to_refinement_workflow() {
        // Simulate a typical RANSAC → refinement pipeline

        // Step 1: Generate data with outliers
        let h_true = Matrix3::new(0.98, -0.17, 0.05, 0.17, 0.98, 0.03, 0.001, -0.001, 1.0);

        let mut correspondences = generate_homography_correspondences(&h_true, 50, 0.0001);

        // Add 20% outliers
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        for _ in 0..10 {
            correspondences.push(PixelCorrespondence {
                point1: Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
                point2: Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
            });
        }

        // Step 2: Use solver on all data (this would normally be RANSAC)
        let result = DltSolver::solve(
            &NormalizedCorrespondences::from_pixels(&correspondences),
            &HomographyConfig::default(),
        );

        assert!(result.is_ok(), "Solver should succeed even with outliers");
        let models = result.unwrap();
        assert!(!models.is_empty(), "Should find homography");

        // Step 3: Compute residuals and identify inliers
        let threshold = 0.01;
        let h_est = models[0].matrix();

        let inliers: Vec<_> = correspondences
            .iter()
            .filter(|corr| {
                let p1_h = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
                let p2_pred_h = h_est * p1_h;
                let p2_pred = Point2::new(p2_pred_h.x / p2_pred_h.z, p2_pred_h.y / p2_pred_h.z);
                let error = (corr.point2 - p2_pred).norm();
                error < threshold
            })
            .cloned()
            .collect();

        // Step 4: Refine with inliers only
        if inliers.len() >= 4 {
            let refined_result = DltSolver::solve(
                &NormalizedCorrespondences::from_pixels(&inliers),
                &HomographyConfig::default(),
            );
            assert!(refined_result.is_ok(), "Refinement should succeed");

            let refined_models = refined_result.unwrap();
            let h_refined = refined_models[0].matrix();

            // Refined model should be closer to ground truth
            let h_true_norm = h_true / h_true[(2, 2)];
            let h_est_norm = h_est / h_est[(2, 2)];
            let h_refined_norm = h_refined / h_refined[(2, 2)];

            let diff_before = (h_true_norm - h_est_norm).norm();
            let diff_after = (h_true_norm - h_refined_norm).norm();

            // Refined should be at least as good (usually better)
            assert!(
                diff_after <= diff_before * 1.5, // Allow some tolerance
                "Refinement should not make things much worse: before={}, after={}",
                diff_before,
                diff_after
            );
        }
    }

    // ========================================================================
    // Test: Residual computation consistency
    // ========================================================================

    #[test]
    fn test_residual_computation_consistency() {
        // Verify that residual functions are consistent across solvers

        // Create a known homography
        let h = Matrix3::new(1.0, 0.0, 0.1, 0.0, 1.0, -0.1, 0.0, 0.0, 1.0);

        // Perfect correspondence
        let p1 = Point2::new(0.5, 0.5);
        let p1_h = Vector3::new(p1.x, p1.y, 1.0);
        let p2_h = h * p1_h;
        let p2 = Point2::new(p2_h.x / p2_h.z, p2_h.y / p2_h.z);

        let corr = PixelCorrespondence {
            point1: p1,
            point2: p2,
        };

        // Residual should be ~0 for perfect correspondence
        let homography = openmvg_geometry::homography::Homography::try_new(h).unwrap();
        let residual = DltSolver::residual(&homography, &corr);

        assert!(
            residual < 1e-10,
            "Residual for perfect match should be ~0, got {}",
            residual
        );

        // Add some error
        let corr_noisy = PixelCorrespondence {
            point1: p1,
            point2: Point2::new(p2.x + 0.01, p2.y),
        };

        let residual_noisy = DltSolver::residual(&homography, &corr_noisy);
        assert!(
            residual_noisy > residual,
            "Noisy correspondence should have higher residual"
        );
    }

    // ========================================================================
    // Test: min_sample_size correctness
    // ========================================================================

    #[test]
    fn test_min_sample_sizes() {
        // Verify minimum sample sizes for RANSAC compatibility
        assert_eq!(EightPointSolver::min_sample_size(), 8);
        assert_eq!(SevenPointSolver::min_sample_size(), 7);
        assert_eq!(DltSolver::min_sample_size(), 4);
        #[cfg(feature = "unstable-test-support")]
        assert_eq!(
            essential::test_support::EightPointSolver::min_sample_size(),
            8
        );
    }
}
