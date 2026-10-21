// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Affine-Core-Affine (ACA) algorithm for homography estimation.
//!
//! A direct 4-point homography solver in closed form, without an SVD.
//!
//! # Algorithm
//!
//! The ACA algorithm decomposes the homography H into three factors:
//!
//! ```text
//!   H = A₂ · C · A₁⁻¹
//! ```
//!
//! where:
//! - A₁ is the source affine transformation, determined by the first 3
//!   source points, mapping them to a reference triangle.
//! - A₂ is the target affine transformation, determined by the first 3
//!   target points, mapping the reference triangle to them.
//! - C is the "core" matrix, a projective transformation that captures the
//!   non-affine part using the 4th point correspondence.
//!
//! The core matrix has the structure:
//!
//! ```text
//!       ⎡ c₁₁   0    0  ⎤
//!   C = ⎢  0   c₂₂   0  ⎥
//!       ⎣ c₃₁  c₃₂  c₃₃ ⎦
//! ```
//!
//! It preserves the first 3 correspondences (handled by the affine parts)
//! while adjusting the projective component via (c₃₁, c₃₂) to satisfy
//! the 4th correspondence.
//!
//! ## Algorithm steps
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │             Homography estimation (ACA 4-point)                     │
//! │                 Affine-Core-Affine decomposition                    │
//! └─────────────────────────────────────────────────────────────────────┘
//!
//!     Input: 4 pixel correspondences (xᵢ, x'ᵢ), i = 0..3
//!
//!     Step 1: Source affine frame
//!       n₁ = x₁ - x₀,  p₁ = x₂ - x₀,  q₁ = x₃ - x₀
//!       fₐ₁ = n₁ × p₁  (signed area of source triangle × 2)
//!       Check: |fₐ₁| > ε, else first 3 source points are collinear
//!       q₃ = barycentric coordinates of x₃ in triangle (x₀, x₁, x₂)
//!            (unnormalized; the normalization cancels in the final formula)
//!
//!     Step 2: Target affine frame
//!       n₂ = x'₁ - x'₀,  p₂ = x'₂ - x'₀,  q₂ = x'₃ - x'₀
//!       fₐ₂ = n₂ × p₂  (signed area of target triangle × 2)
//!       Check: |fₐ₂| > ε, else first 3 target points are collinear
//!       q₄ = barycentric coordinates of x'₃ in triangle (x'₀, x'₁, x'₂)
//!
//!     Step 3: Core matrix coefficients
//!       tt₁ = fₐ₁ - q₃ₓ - q₃ᵧ
//!       c₁₁ = q₃ᵧ · q₄ₓ · tt₁
//!       c₂₂ = q₃ₓ · q₄ᵧ · tt₁
//!       c₃₃ = q₃ₓ · q₃ᵧ · (fₐ₂ - q₄ₓ - q₄ᵧ)
//!       c₃₁ = c₁₁ - c₃₃,   c₃₂ = c₂₂ - c₃₃
//!
//!     Step 4: Assemble H (fused A₂ · C · A₁⁻¹)
//!       Directly compute the 9 entries of H from the factored form
//!
//!     Step 5: Normalize
//!       H ← H / H₃₃  (if H₃₃ ≠ 0)
//!
//!     Output: 3×3 homography matrix H
//!
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  Complexity: O(1), constant time, no iterative solver               │
//! │  Memory: Stack-allocated scalar intermediates only                  │
//! │  Exact: No least squares; algebraically exact for 4 point pairs     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Degenerate configurations
//!
//! The solver returns an error for these cases:
//! - Collinear source points: fₐ₁ ≈ 0 (first 3 source points on a line).
//! - Collinear target points: fₐ₂ ≈ 0 (first 3 target points on a line).
//! - 4th source point on a triangle edge: q₃ₓ ≈ 0 or q₃ᵧ ≈ 0.
//! - 4th target point on a triangle edge: q₄ₓ ≈ 0 or q₄ᵧ ≈ 0.
//! - 4th source point on the remaining triangle edge: tt₁ ≈ 0 (the remaining
//!   barycentric coordinate vanishes). Affine homographies are not degenerate.
//! - Singular result: det(H) ≈ 0.
//!
//! ## Notation
//!
//! | Symbol | Meaning |
//! |---|---|
//! | x₀ | First source point, the reference origin |
//! | n₁ = x₁ − x₀, p₁ = x₂ − x₀, q₁ = x₃ − x₀ | Source vectors from the first point to the others |
//! | fₐ₁ = n₁ × p₁ | Twice the signed area of the source triangle |
//! | q₃ | Unnormalized barycentric coordinates of the fourth source point |
//! | n₂, p₂, q₂ | The same vectors in the target image, from x'₀ |
//! | fₐ₂ = n₂ × p₂ | Twice the signed area of the target triangle |
//! | q₄ | Unnormalized barycentric coordinates of the fourth target point |
//!
//! # References
//!
//! - S. Cai, Z. Wu, L. Guo, J. Wang, S. Zhang, J. Yan, S. Shen, "Fast and Interpretable 2D
//!   Homography Decomposition: Similarity-Kernel-Similarity and Affine-Core-Affine
//!   Transformations", arXiv:2402.18008, 2024. <https://arxiv.org/abs/2402.18008>

use nalgebra::Matrix3;

use crate::normalization::CorrespondenceCoordinates;
use crate::{SolverError, SolverResult};

use super::Homography;

/// ACA homography solver.
///
/// Requires exactly 4 point correspondences whose first 3 points span a
/// triangle in each image, with the 4th point off that triangle's edges.
///
/// Reached through [`super::HomographyMethod::Aca`]; the struct itself is
/// crate-private so there is a single public ACA path.
pub(crate) struct AcaSolver;

impl AcaSolver {
    /// Solve for the homography with the ACA algorithm.
    ///
    /// Computes H = A₂ · C · A₁⁻¹ directly from the 4 correspondences
    /// using fused closed-form algebra (no intermediate matrix construction).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input does not hold exactly 4 correspondences
    /// - First 3 source or target points are collinear (fₐ₁ ≈ 0 or fₐ₂ ≈ 0)
    /// - 4th point lies on an edge of the triangle formed by the first 3 (q₃ or q₄ degenerate)
    /// - Near-affine configuration (tt₁ ≈ 0)
    /// - Resulting homography is singular
    pub(crate) fn solve<D: CorrespondenceCoordinates>(
        correspondences: &[D],
    ) -> SolverResult<Homography> {
        match correspondences.len() {
            0..=3 => {
                return Err(SolverError::InsufficientData {
                    required: 4,
                    provided: correspondences.len(),
                });
            }
            4 => {}
            provided => {
                return Err(SolverError::InvalidSampleSize {
                    expected: 4,
                    provided,
                });
            }
        }

        // Extract the 4 point correspondences
        let xa = [
            correspondences[0].coordinates().point1,
            correspondences[1].coordinates().point1,
            correspondences[2].coordinates().point1,
            correspondences[3].coordinates().point1,
        ];
        let xb = [
            correspondences[0].coordinates().point2,
            correspondences[1].coordinates().point2,
            correspondences[2].coordinates().point2,
            correspondences[3].coordinates().point2,
        ];
        if xa
            .iter()
            .chain(&xb)
            .flat_map(|point| point.coords.iter())
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(SolverError::InvalidInput {
                reason: "ACA correspondences must be finite",
            });
        }

        let relative_tolerance = f64::EPSILON * 1024.0;

        // ── Step 1: Source affine frame ──
        // Vectors from x₀ to x₁, x₂, x₃
        let m1n1_x = xa[1].x - xa[0].x; // n₁.x
        let m1p1_x = xa[2].x - xa[0].x; // p₁.x
        let m1q1_x = xa[3].x - xa[0].x; // q₁.x
        let m1n1_y = xa[1].y - xa[0].y; // n₁.y
        let m1p1_y = xa[2].y - xa[0].y; // p₁.y
        let m1q1_y = xa[3].y - xa[0].y; // q₁.y
        let source_scale = [
            m1n1_x.abs(),
            m1p1_x.abs(),
            m1q1_x.abs(),
            m1n1_y.abs(),
            m1p1_y.abs(),
            m1q1_y.abs(),
        ]
        .into_iter()
        .fold(0.0, f64::max);
        let source_area_tolerance = source_scale * source_scale * relative_tolerance;

        // fₐ₁ = n₁ × p₁ = signed area of source triangle × 2
        let fa1 = m1n1_x * m1p1_y - m1n1_y * m1p1_x;
        if source_scale <= 0.0 || fa1.abs() <= source_area_tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "first 3 source points are collinear (fa1 ≈ 0)".to_string(),
            });
        }

        // q₃ = barycentric coordinates of x₃ relative to triangle (x₀, x₁, x₂)
        // (unnormalized by fa1; the normalization cancels in the final formula)
        let q3_x = m1p1_y * m1q1_x - m1p1_x * m1q1_y;
        let q3_y = m1n1_x * m1q1_y - m1n1_y * m1q1_x;

        // ── Step 2: Target affine frame ──
        let m2n2_x = xb[1].x - xb[0].x; // n₂.x
        let m2p2_x = xb[2].x - xb[0].x; // p₂.x
        let m2q2_x = xb[3].x - xb[0].x; // q₂.x
        let m2n2_y = xb[1].y - xb[0].y; // n₂.y
        let m2p2_y = xb[2].y - xb[0].y; // p₂.y
        let m2q2_y = xb[3].y - xb[0].y; // q₂.y
        let target_scale = [
            m2n2_x.abs(),
            m2p2_x.abs(),
            m2q2_x.abs(),
            m2n2_y.abs(),
            m2p2_y.abs(),
            m2q2_y.abs(),
        ]
        .into_iter()
        .fold(0.0, f64::max);
        let target_area_tolerance = target_scale * target_scale * relative_tolerance;

        // fₐ₂ = n₂ × p₂ = signed area of target triangle × 2
        let fa2 = m2n2_x * m2p2_y - m2n2_y * m2p2_x;
        if target_scale <= 0.0 || fa2.abs() <= target_area_tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "first 3 target points are collinear (fa2 ≈ 0)".to_string(),
            });
        }

        // q₄ = barycentric coordinates of x'₃ relative to triangle (x'₀, x'₁, x'₂)
        let q4_x = m2p2_y * m2q2_x - m2p2_x * m2q2_y;
        let q4_y = m2n2_x * m2q2_y - m2n2_y * m2q2_x;
        if q3_x.abs() <= source_area_tolerance
            || q3_y.abs() <= source_area_tolerance
            || q4_x.abs() <= target_area_tolerance
            || q4_y.abs() <= target_area_tolerance
        {
            return Err(SolverError::DegenerateConfiguration {
                reason: "ACA fourth point lies on a reference-triangle edge".to_owned(),
            });
        }

        // ── Step 3: Core matrix coefficients ──
        // tt₁ is the third homogeneous barycentric coordinate of x₃.
        // Near zero, the fourth point lies on the remaining reference-triangle edge.
        let tt1 = fa1 - q3_x - q3_y;
        if tt1.abs() <= source_area_tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "ACA fourth source point lies on a reference-triangle edge".to_owned(),
            });
        }

        let c11 = q3_y * q4_x * tt1;
        let c22 = q3_x * q4_y * tt1;
        let c33 = q3_x * q3_y * (fa2 - q4_x - q4_y);
        let c31 = c11 - c33;
        let c32 = c22 - c33;

        // ── Step 4: Assemble H entries (fused A₂ · C · A₁⁻¹) ──
        let tt3 = xb[0].x * c33;
        let tt4 = xb[0].y * c33;
        let h1_11 = xb[1].x * c11 - tt3;
        let h1_12 = xb[2].x * c22 - tt3;
        let h1_21 = xb[1].y * c11 - tt4;
        let h1_22 = xb[2].y * c22 - tt4;

        let h0 = h1_11 * m1p1_y - h1_12 * m1n1_y;
        let h1 = h1_12 * m1n1_x - h1_11 * m1p1_x;
        let h3 = h1_21 * m1p1_y - h1_22 * m1n1_y;
        let h4 = h1_22 * m1n1_x - h1_21 * m1p1_x;
        let h6 = c31 * m1p1_y - c32 * m1n1_y;
        let h7 = c32 * m1n1_x - c31 * m1p1_x;
        let h2 = tt3 * fa1 - h0 * xa[0].x - h1 * xa[0].y;
        let h5 = tt4 * fa1 - h3 * xa[0].x - h4 * xa[0].y;
        let h8 = c33 * fa1 - h6 * xa[0].x - h7 * xa[0].y;

        // ── Step 5: Normalize by h₃₃ before constructing Matrix3 ──
        // This avoids constructing the matrix twice (once un-normalized, once divided).
        let h_scale = [
            h0.abs(),
            h1.abs(),
            h2.abs(),
            h3.abs(),
            h4.abs(),
            h5.abs(),
            h6.abs(),
            h7.abs(),
            h8.abs(),
        ]
        .into_iter()
        .fold(0.0, f64::max);
        let matrix_tolerance = h_scale * relative_tolerance;
        let (h0, h1, h2, h3, h4, h5, h6, h7, h8) = if h8.abs() > matrix_tolerance {
            let inv_h8 = 1.0 / h8;
            (
                h0 * inv_h8,
                h1 * inv_h8,
                h2 * inv_h8,
                h3 * inv_h8,
                h4 * inv_h8,
                h5 * inv_h8,
                h6 * inv_h8,
                h7 * inv_h8,
                1.0,
            )
        } else {
            (h0, h1, h2, h3, h4, h5, h6, h7, h8)
        };

        let h = Matrix3::new(h0, h1, h2, h3, h4, h5, h6, h7, h8);
        if !h.iter().all(|value| value.is_finite()) {
            return Err(SolverError::Numerical {
                reason: "ACA produced a non-finite homography".to_owned(),
            });
        }

        // Check for degenerate homography
        let det = h.determinant();
        let output_scale = h
            .iter()
            .copied()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let normalized_det_tolerance =
            relative_tolerance * output_scale * output_scale * output_scale;
        if !det.is_finite() || output_scale <= 0.0 || det.abs() <= normalized_det_tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "resulting homography is singular".to_string(),
            });
        }

        Ok(Homography::new(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correspondence::PixelCorrespondence;
    use nalgebra::Point2;

    fn create_correspondences(
        h: &Matrix3<f64>,
        points: &[Point2<f64>],
    ) -> Vec<PixelCorrespondence> {
        use nalgebra::Vector3;

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

    #[test]
    fn test_aca_identity() {
        let correspondences = vec![
            PixelCorrespondence::new(Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
            PixelCorrespondence::new(Point2::new(100.0, 0.0), Point2::new(100.0, 0.0)),
            PixelCorrespondence::new(Point2::new(100.0, 100.0), Point2::new(100.0, 100.0)),
            PixelCorrespondence::new(Point2::new(0.0, 100.0), Point2::new(0.0, 100.0)),
        ];

        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_ok(), "ACA solver failed: {:?}", result.err());

        let h = result.unwrap();

        // Should be close to identity
        let identity = Matrix3::<f64>::identity();
        let diff = (h.0 - identity).norm();
        assert!(diff < 1e-8, "Expected identity, diff = {}", diff);
    }

    #[test]
    fn test_aca_translation() {
        let h_true = Matrix3::new(1.0, 0.0, 50.0, 0.0, 1.0, 30.0, 0.0, 0.0, 1.0);

        let points = vec![
            Point2::new(10.0, 10.0),
            Point2::new(110.0, 10.0),
            Point2::new(110.0, 110.0),
            Point2::new(10.0, 110.0),
        ];

        let correspondences = create_correspondences(&h_true, &points);
        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_ok(), "ACA solver failed: {:?}", result.err());

        let h = result.unwrap();

        for corr in &correspondences {
            let x1 = nalgebra::Vector3::new(corr.point1.x, corr.point1.y, 1.0);
            let x2_pred = h.0 * x1;
            let p2_pred = Point2::new(x2_pred.x / x2_pred.z, x2_pred.y / x2_pred.z);
            let error = (p2_pred - corr.point2).norm();
            assert!(error < 1e-8, "Reprojection error too large: {}", error);
        }
    }

    #[test]
    fn test_aca_rotation() {
        use std::f64::consts::PI;
        let angle = PI / 6.0; // 30 degrees
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let h_true = Matrix3::new(cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0);

        let points = vec![
            Point2::new(50.0, 50.0),
            Point2::new(150.0, 50.0),
            Point2::new(150.0, 150.0),
            Point2::new(50.0, 150.0),
        ];

        let correspondences = create_correspondences(&h_true, &points);
        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_ok(), "ACA solver failed: {:?}", result.err());

        let h = result.unwrap();

        for corr in &correspondences {
            let x1 = nalgebra::Vector3::new(corr.point1.x, corr.point1.y, 1.0);
            let x2_pred = h.0 * x1;
            let p2_pred = Point2::new(x2_pred.x / x2_pred.z, x2_pred.y / x2_pred.z);
            let error = (p2_pred - corr.point2).norm();
            assert!(error < 1e-8, "Reprojection error too large: {}", error);
        }
    }

    /// Four-point ACA Homography solver: four exact correspondences recover a perspective
    /// homography; the returned projective transform maps every source point to its target.
    #[test]
    fn test_aca_perspective() {
        let h_true = Matrix3::new(1.2, 0.1, 20.0, 0.05, 1.1, 10.0, 0.0001, 0.0002, 1.0);

        let points = vec![
            Point2::new(100.0, 100.0),
            Point2::new(200.0, 100.0),
            Point2::new(200.0, 200.0),
            Point2::new(100.0, 200.0),
        ];

        let correspondences = create_correspondences(&h_true, &points);
        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_ok(), "ACA solver failed: {:?}", result.err());

        let h = result.unwrap();

        for corr in &correspondences {
            let x1 = nalgebra::Vector3::new(corr.point1.x, corr.point1.y, 1.0);
            let x2_pred = h.0 * x1;
            let p2_pred = Point2::new(x2_pred.x / x2_pred.z, x2_pred.y / x2_pred.z);
            let error = (p2_pred - corr.point2).norm();
            assert!(error < 1e-6, "Reprojection error too large: {}", error);
        }
    }

    #[test]
    fn test_aca_insufficient_points() {
        let correspondences = vec![
            PixelCorrespondence::new(Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
            PixelCorrespondence::new(Point2::new(1.0, 0.0), Point2::new(1.0, 0.0)),
        ];

        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(SolverError::InsufficientData {
                required: 4,
                provided: 2
            })
        ));
    }

    #[test]
    fn test_aca_too_many_points() {
        let correspondences = vec![
            PixelCorrespondence::new(Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
            PixelCorrespondence::new(Point2::new(1.0, 0.0), Point2::new(1.0, 0.0)),
            PixelCorrespondence::new(Point2::new(1.0, 1.0), Point2::new(1.0, 1.0)),
            PixelCorrespondence::new(Point2::new(0.0, 1.0), Point2::new(0.0, 1.0)),
            PixelCorrespondence::new(Point2::new(0.5, 0.5), Point2::new(0.5, 0.5)),
        ];

        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(SolverError::InvalidSampleSize {
                expected: 4,
                provided: 5
            })
        ));
    }

    #[test]
    fn rejects_non_finite_correspondences() {
        let correspondences = [
            PixelCorrespondence::new(Point2::new(f64::NAN, 0.0), Point2::new(0.0, 0.0)),
            PixelCorrespondence::new(Point2::new(1.0, 0.0), Point2::new(1.0, 0.0)),
            PixelCorrespondence::new(Point2::new(1.0, 1.0), Point2::new(1.0, 1.0)),
            PixelCorrespondence::new(Point2::new(0.0, 1.0), Point2::new(0.0, 1.0)),
        ];

        assert!(matches!(
            AcaSolver::solve(&correspondences),
            Err(SolverError::InvalidInput { .. })
        ));
    }

    #[test]
    fn degeneracy_decision_is_coordinate_scale_invariant() {
        for scale in [1e-9, 1.0, 1e9] {
            let correspondences = [
                PixelCorrespondence::new(Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
                PixelCorrespondence::new(Point2::new(scale, 0.0), Point2::new(scale, 0.0)),
                PixelCorrespondence::new(
                    Point2::new(2.0 * scale, 0.0),
                    Point2::new(2.0 * scale, 0.0),
                ),
                PixelCorrespondence::new(Point2::new(0.0, scale), Point2::new(0.0, scale)),
            ];
            assert!(matches!(
                AcaSolver::solve(&correspondences),
                Err(SolverError::DegenerateConfiguration { .. })
            ));
        }
    }

    // ====================================================================
    // Degeneracy tests: check the error reporting
    // ====================================================================

    #[test]
    fn test_aca_collinear_source_points() {
        // First 3 source points are collinear (all on y=0 line)
        let correspondences = vec![
            PixelCorrespondence::new(Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
            PixelCorrespondence::new(Point2::new(100.0, 0.0), Point2::new(100.0, 50.0)),
            PixelCorrespondence::new(Point2::new(200.0, 0.0), Point2::new(200.0, 100.0)),
            PixelCorrespondence::new(Point2::new(50.0, 50.0), Point2::new(50.0, 50.0)),
        ];

        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_err(), "Should fail for collinear source points");
        assert!(
            matches!(result, Err(SolverError::DegenerateConfiguration { .. })),
            "Should return DegenerateConfiguration"
        );
    }

    #[test]
    fn test_aca_collinear_target_points() {
        // First 3 target points are collinear (all on x=50 line)
        let correspondences = vec![
            PixelCorrespondence::new(Point2::new(0.0, 0.0), Point2::new(50.0, 0.0)),
            PixelCorrespondence::new(Point2::new(100.0, 0.0), Point2::new(50.0, 100.0)),
            PixelCorrespondence::new(Point2::new(100.0, 100.0), Point2::new(50.0, 200.0)),
            PixelCorrespondence::new(Point2::new(0.0, 100.0), Point2::new(0.0, 100.0)),
        ];

        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_err(), "Should fail for collinear target points");
        assert!(
            matches!(result, Err(SolverError::DegenerateConfiguration { .. })),
            "Should return DegenerateConfiguration"
        );
    }

    #[test]
    fn test_aca_coincident_source_points() {
        // x₀ = x₁ → n₁ = (0,0) → fa1 = 0
        let correspondences = vec![
            PixelCorrespondence::new(Point2::new(50.0, 50.0), Point2::new(0.0, 0.0)),
            PixelCorrespondence::new(Point2::new(50.0, 50.0), Point2::new(100.0, 0.0)),
            PixelCorrespondence::new(Point2::new(100.0, 100.0), Point2::new(100.0, 100.0)),
            PixelCorrespondence::new(Point2::new(0.0, 100.0), Point2::new(0.0, 100.0)),
        ];

        let result = AcaSolver::solve(&correspondences);
        assert!(result.is_err(), "Should fail for coincident source points");
    }
}
