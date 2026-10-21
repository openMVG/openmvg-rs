// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How far a data point lies from a model: the error measures that robust estimation scores and
//! compares with its threshold.

use nalgebra::{Matrix3, Vector3};

use crate::correspondence::{BearingCorrespondence, PixelCorrespondence};

/// Normalize the representation scale before projective residual arithmetic.
fn normalized_projective(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return None;
    }
    let scale = matrix
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    (scale > 0.0).then(|| matrix / scale)
}

/// Compute the Sampson error, in pixels and not squared, of a fundamental matrix correspondence.
///
/// The Sampson error is a first-order approximation to the geometric (reprojection) error.
/// It avoids the per-correspondence optimal triangulation that the geometric error requires.
/// Returns infinity when the denominator vanishes.
///
/// # Mathematical background
///
/// For the epipolar constraint `x₂ᵀ F x₁ = 0`, the Sampson error is:
///
/// ```text
/// e = |x₂ᵀ F x₁| / √((F x₁)₁² + (F x₁)₂² + (Fᵀ x₂)₁² + (Fᵀ x₂)₂²)
/// ```
///
/// This approximates the geometric distance from the points to their respective epipolar lines.
#[inline]
pub fn sampson_error_fundamental(f: &Matrix3<f64>, corr: &PixelCorrespondence) -> f64 {
    let Some((numerator, denominator)) = fundamental_sampson_terms(f, corr) else {
        return f64::INFINITY;
    };
    numerator / denominator.sqrt()
}

/// Squared Sampson error for a Fundamental matrix correspondence.
///
/// This is algebraically identical to [`sampson_error_fundamental`] squared,
/// but avoids a square root in scoring paths that consume squared distances.
#[inline]
pub(crate) fn sampson_error_fundamental_squared(
    f: &Matrix3<f64>,
    corr: &PixelCorrespondence,
) -> f64 {
    let Some((numerator, denominator)) = fundamental_sampson_terms(f, corr) else {
        return f64::INFINITY;
    };
    numerator * numerator / denominator
}

#[inline]
fn fundamental_sampson_terms(f: &Matrix3<f64>, corr: &PixelCorrespondence) -> Option<(f64, f64)> {
    let f = normalized_projective(f)?;
    let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
    let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);

    // Epipolar constraint: x2^T * F * x1
    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;
    let x2tfx1 = x2.dot(&fx1);

    // Denominator: sum of squared components of epipolar line normals
    let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;

    // Avoid division by zero
    if !denom.is_finite() || denom < 1e-20 {
        return None;
    }

    // Non-squared Sampson error (in pixels) is |x2^T * F * x1| / sqrt(denom)
    let numerator = x2tfx1.abs();
    numerator.is_finite().then_some((numerator, denom))
}

/// Compute the first-order angular Sampson error, in radians and not squared, of an essential matrix.
///
/// Bearing vectors live on the unit sphere, so the relevant perturbations are
/// tangent to that sphere.  For the epipolar constraint
/// `f = b₂ᵀ E b₁`, the tangent gradients are:
///
/// ```text
/// g₁ = Eᵀ b₂ − b₁ f
/// g₂ = E b₁  − b₂ f
/// ```
///
/// and this function returns `|f| / √(‖g₁‖² + ‖g₂‖²)`. This is
/// coordinate-frame invariant and, for small errors, has angular units.  The
/// [`BearingCorrespondence`] guarantees unit, finite input vectors.
#[inline]
pub fn sampson_error_essential(e: &Matrix3<f64>, corr: &BearingCorrespondence) -> f64 {
    let Some((numerator, denominator)) = essential_sampson_terms(e, corr) else {
        return f64::INFINITY;
    };
    numerator / denominator.sqrt()
}

/// Squared angular Sampson error for an Essential matrix correspondence.
///
/// This avoids the square root performed by [`sampson_error_essential`] when
/// the robust estimator compares against a squared angular threshold.
#[inline]
pub(crate) fn sampson_error_essential_squared(
    e: &Matrix3<f64>,
    corr: &BearingCorrespondence,
) -> f64 {
    let Some((numerator, denominator)) = essential_sampson_terms(e, corr) else {
        return f64::INFINITY;
    };
    numerator * numerator / denominator
}

/// Squared Sampson error in the normalized pinhole image plane.
///
/// This evaluates the usual image-plane Sampson denominator in homogeneous
/// form, avoiding explicit `x / z` and `y / z` divisions while remaining
/// independent of each input bearing's arbitrary scale. Bearings on or near
/// the camera equator (`z = 0`) are unsupported and return infinity; use the
/// angular [`sampson_error_essential`] for spherical or otherwise unrestricted
/// bearing cameras.
#[inline]
pub fn sampson_error_essential_pinhole_squared(
    e: &Matrix3<f64>,
    corr: &BearingCorrespondence,
) -> f64 {
    let Some(e) = normalized_projective(e) else {
        return f64::INFINITY;
    };
    let b1 = corr.bearing1();
    let b2 = corr.bearing2();
    if b1.z.abs() < 1e-15 || b2.z.abs() < 1e-15 {
        return f64::INFINITY;
    }

    // Spell out the fixed 3x3 products: this is a scoring hot path and the
    // scalar form gives LLVM a simple loop body to inline and vectorize.
    let ex1_0 = e[(0, 0)] * b1.x + e[(0, 1)] * b1.y + e[(0, 2)] * b1.z;
    let ex1_1 = e[(1, 0)] * b1.x + e[(1, 1)] * b1.y + e[(1, 2)] * b1.z;
    let ex1_2 = e[(2, 0)] * b1.x + e[(2, 1)] * b1.y + e[(2, 2)] * b1.z;
    let etx2_0 = e[(0, 0)] * b2.x + e[(1, 0)] * b2.y + e[(2, 0)] * b2.z;
    let etx2_1 = e[(0, 1)] * b2.x + e[(1, 1)] * b2.y + e[(2, 1)] * b2.z;

    let constraint = b2.x * ex1_0 + b2.y * ex1_1 + b2.z * ex1_2;
    let denominator = b2.z * b2.z * (ex1_0 * ex1_0 + ex1_1 * ex1_1)
        + b1.z * b1.z * (etx2_0 * etx2_0 + etx2_1 * etx2_1);
    if !constraint.is_finite() || !denominator.is_finite() || denominator < 1e-20 {
        return f64::INFINITY;
    }
    constraint * constraint / denominator
}

#[inline]
fn essential_sampson_terms(e: &Matrix3<f64>, corr: &BearingCorrespondence) -> Option<(f64, f64)> {
    let e = normalized_projective(e)?;
    let b1 = *corr.bearing1();
    let b2 = *corr.bearing2();

    // Epipolar constraint: x2^T * E * x1
    let eb1 = e * b1;
    let etb2 = e.transpose() * b2;
    let constraint = b2.dot(&eb1);

    // Squared norms of the two tangent-plane projections. For unit bearings,
    // ||g - b(b^T g)||^2 = ||g||^2 - (b^T g)^2. Use that identity directly
    // instead of constructing both projected-gradient vectors.
    let constraint_squared = constraint * constraint;
    let denom = eb1.norm_squared() + etb2.norm_squared() - 2.0 * constraint_squared;

    // Avoid division by zero
    if !denom.is_finite() || denom < 1e-20 {
        return None;
    }

    let numerator = constraint.abs();
    numerator.is_finite().then_some((numerator, denom))
}

/// Compute the symmetric transfer error, in squared pixels, of a homography correspondence.
///
/// The symmetric transfer error measures the sum of squared distances between:
/// - point2 and H point1 (forward transfer)
/// - point1 and H⁻¹ point2 (backward transfer)
///
/// Returns infinity if H is not invertible.
#[inline]
fn symmetric_transfer_error_homography(h: &Matrix3<f64>, corr: &PixelCorrespondence) -> f64 {
    let Some(h) = crate::homography::normalized_homography_matrix(h) else {
        return f64::INFINITY;
    };
    let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
    let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);

    // Forward transfer: H * x1
    let hx1 = h * x1;
    if hx1.z.abs() < 1e-10 {
        return f64::INFINITY;
    }
    let hx1_normalized = Vector3::new(hx1.x / hx1.z, hx1.y / hx1.z, 1.0);

    // Backward transfer: H^-1 * x2
    // Invert the normalized representation, so raw determinant scale cannot
    // overflow/underflow. The inverse has an independent projective gauge.
    let h_inv = match h
        .try_inverse()
        .and_then(|inverse| crate::homography::normalized_homography_matrix(&inverse))
    {
        Some(inv) => inv,
        None => return f64::INFINITY,
    };
    let hinvx2 = h_inv * x2;
    if hinvx2.z.abs() < 1e-10 {
        return f64::INFINITY;
    }
    let hinvx2_normalized = Vector3::new(hinvx2.x / hinvx2.z, hinvx2.y / hinvx2.z, 1.0);

    // Forward error: ||x2 - H*x1||^2
    let forward_err = (x2.x - hx1_normalized.x) * (x2.x - hx1_normalized.x)
        + (x2.y - hx1_normalized.y) * (x2.y - hx1_normalized.y);

    // Backward error: ||x1 - H^-1*x2||^2
    let backward_err = (x1.x - hinvx2_normalized.x) * (x1.x - hinvx2_normalized.x)
        + (x1.y - hinvx2_normalized.y) * (x1.y - hinvx2_normalized.y);

    let error = forward_err + backward_err;
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

/// Compute the unsquared symmetric transfer distance for a homography.
///
/// This is the Euclidean norm of the forward and backward 2D transfer-error
/// vectors combined.  Its units are pixels, so it is suitable for
/// robust estimation, whose scorers square residuals when evaluating costs.
#[inline]
pub(crate) fn symmetric_transfer_distance_homography(
    h: &Matrix3<f64>,
    corr: &PixelCorrespondence,
) -> f64 {
    symmetric_transfer_error_homography(h, corr).sqrt()
}

/// Symmetric transfer error for a homography with a pre-computed inverse.
///
/// Returns `‖x₂ − H x₁‖² + ‖x₁ − H⁻¹ x₂‖²`, the same squared-pixel value
/// as [`symmetric_transfer_error_homography`], but avoids computing `H⁻¹` on
/// every call: the caller passes it in.
#[inline]
pub(crate) fn symmetric_transfer_error_h_with_inv(
    h: &Matrix3<f64>,
    h_inv: &Matrix3<f64>,
    corr: &PixelCorrespondence,
) -> f64 {
    let Some(h) = crate::homography::normalized_homography_matrix(h) else {
        return f64::INFINITY;
    };
    let Some(h_inv) = crate::homography::normalized_homography_matrix(h_inv) else {
        return f64::INFINITY;
    };
    let (x1x, x1y) = (corr.point1.x, corr.point1.y);
    let (x2x, x2y) = (corr.point2.x, corr.point2.y);

    // Forward: H * [x1; 1]
    let hw = h[(2, 0)] * x1x + h[(2, 1)] * x1y + h[(2, 2)];
    if hw.abs() < 1e-10 {
        return f64::INFINITY;
    }
    let inv_hw = 1.0 / hw;
    let hx = (h[(0, 0)] * x1x + h[(0, 1)] * x1y + h[(0, 2)]) * inv_hw;
    let hy = (h[(1, 0)] * x1x + h[(1, 1)] * x1y + h[(1, 2)]) * inv_hw;
    let dx = x2x - hx;
    let dy = x2y - hy;
    let fwd = dx * dx + dy * dy;

    // Backward: H_inv * [x2; 1]
    let iw = h_inv[(2, 0)] * x2x + h_inv[(2, 1)] * x2y + h_inv[(2, 2)];
    if iw.abs() < 1e-10 {
        return f64::INFINITY;
    }
    let inv_iw = 1.0 / iw;
    let ix = (h_inv[(0, 0)] * x2x + h_inv[(0, 1)] * x2y + h_inv[(0, 2)]) * inv_iw;
    let iy = (h_inv[(1, 0)] * x2x + h_inv[(1, 1)] * x2y + h_inv[(1, 2)]) * inv_iw;
    let bx = x1x - ix;
    let by = x1y - iy;

    let error = fwd + bx * bx + by * by;
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Rotation3};

    #[test]
    fn homography_public_residual_and_gauge_preserve_projective_scale() {
        use crate::correspondence::PixelCorrespondence;
        use crate::homography::{DltSolver, Homography};
        let exact = PixelCorrespondence::from_coords(2.0, 3.0, 2.0, 3.0);
        let noisy = PixelCorrespondence::from_coords(2.0, 3.0, 5.0, 7.0);
        let reference = 50.0_f64.sqrt();
        for scale in [
            1e-150, -1e-150, 1e-12, -1e-12, 1.0, -1.0, 1e12, -1e12, 1e150, -1e150,
        ] {
            let original = Matrix3::identity() * scale;
            let model = Homography::try_new(original).unwrap();
            assert_eq!(DltSolver::residual(&model, &exact), 0.0, "scale {scale}");
            assert!(
                (DltSolver::residual(&model, &noisy) - reference).abs() < 1e-12,
                "scale {scale}"
            );
            assert!(
                (model.normalized().matrix() - Matrix3::identity()).norm() < 1e-12,
                "scale {scale}"
            );
            assert_eq!(*model.matrix(), original);
            let inverse = Matrix3::identity() / scale;
            assert!(symmetric_transfer_error_h_with_inv(&original, &inverse, &exact).abs() < 1e-24);
        }
        // Zero h22 selects the Frobenius gauge, independent of stored scale/sign.
        let base = Matrix3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        let expected = base / base.norm();
        for scale in [
            1e-200, -1e-200, 1e-160, -1e-160, 1e-150, -1e-150, 1.0, -1.0, 1e150, -1e150, 1e200,
            -1e200,
        ] {
            let original = base * scale;
            let model = Homography::try_new(original).unwrap();
            let normalized = model.normalized();
            assert!((normalized.matrix().norm() - 1.0).abs() < 1e-12);
            assert!(
                (normalized.matrix() - expected)
                    .norm()
                    .min((normalized.matrix() + expected).norm())
                    < 1e-12
            );
            assert_eq!(*model.matrix(), original);
        }
        let huge = PixelCorrespondence::from_coords(1e308, 1e308, 1e308, 1e308);
        let matrix = Matrix3::new(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0);
        let model = Homography::try_new(matrix).unwrap();
        assert_eq!(DltSolver::residual(&model, &huge), f64::INFINITY);
        assert_eq!(
            symmetric_transfer_error_h_with_inv(&matrix, &matrix.try_inverse().unwrap(), &huge),
            f64::INFINITY
        );
    }

    #[test]
    fn epipolar_residuals_preserve_extreme_projective_scale() {
        let matrix = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        let exact_pixel = PixelCorrespondence::from_coords(0.0, 0.0, 0.0, 0.0);
        let noisy_pixel = PixelCorrespondence::from_coords(0.0, 0.0, 0.0, 0.01);
        let exact_bearing = BearingCorrespondence::try_new(Vector3::z(), Vector3::z()).unwrap();
        let noisy_bearing = BearingCorrespondence::try_new(
            Vector3::z(),
            Vector3::new(0.0, 0.01_f64.sin(), 0.01_f64.cos()),
        )
        .unwrap();
        let angular_expected = 0.01_f64.tan() / 2.0_f64.sqrt();
        for scale in [1e-150, 1e-100, 1e-12, 1.0, 1e12, 1e100, 1e150] {
            for sign in [-1.0, 1.0] {
                let scaled = matrix * (sign * scale);
                assert_eq!(sampson_error_fundamental(&scaled, &exact_pixel), 0.0);
                assert_eq!(sampson_error_essential(&scaled, &exact_bearing), 0.0);
                assert_eq!(
                    sampson_error_essential_pinhole_squared(&scaled, &exact_bearing),
                    0.0
                );
                assert!(
                    (sampson_error_fundamental(&scaled, &noisy_pixel) - 0.01 / 2.0_f64.sqrt())
                        .abs()
                        < 1e-14
                );
                assert!(
                    (sampson_error_essential(&scaled, &noisy_bearing) - angular_expected).abs()
                        < 1e-14
                );
                assert!(
                    (sampson_error_essential_squared(&scaled, &noisy_bearing)
                        - angular_expected.powi(2))
                    .abs()
                        < 1e-14
                );
            }
        }
        assert!(sampson_error_essential(&Matrix3::zeros(), &exact_bearing).is_infinite());
    }

    #[test]
    fn test_sampson_error_perfect_correspondence() {
        // Create a fundamental matrix that is satisfied by the correspondence
        // For simplicity, use a trivial case where F = 0 (degenerate but illustrative)
        // In practice, we'd use a real F matrix

        let f = Matrix3::new(0.0, 0.0, 0.01, 0.0, 0.0, -0.01, -0.01, 0.01, 0.0);

        // A point that approximately satisfies the epipolar constraint
        let corr = PixelCorrespondence::from_coords(0.0, 0.0, 0.0, 0.0);

        let error = sampson_error_fundamental(&f, &corr);
        // Error should be small for points near origin with this F
        assert!(error < 1.0);
    }

    #[test]
    fn direct_squared_fundamental_sampson_matches_reference() {
        let f = Matrix3::new(0.0, -0.2, 0.4, 0.1, 0.0, -0.3, -0.5, 0.2, 0.01);
        let corr = PixelCorrespondence::from_coords(12.0, -4.0, 8.0, 3.0);
        let reference = sampson_error_fundamental(&f, &corr).powi(2);
        let direct = sampson_error_fundamental_squared(&f, &corr);
        assert!((reference - direct).abs() <= f64::EPSILON * direct.abs().max(1.0) * 4.0);
    }

    #[test]
    fn test_symmetric_transfer_identity() {
        // Identity homography should give zero error for matching points
        let h = Matrix3::identity();

        let corr = PixelCorrespondence::from_coords(100.0, 200.0, 100.0, 200.0);

        let error = symmetric_transfer_error_homography(&h, &corr);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_sampson_essential() {
        let e = Matrix3::new(0.0, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0);

        let corr = BearingCorrespondence::try_new(
            Vector3::new(0.0, 0.0, 1.0).normalize(),
            Vector3::new(0.0, 0.0, 1.0).normalize(),
        )
        .unwrap();

        let error = sampson_error_essential(&e, &corr);
        // Should be finite and non-negative
        assert!(error.is_finite());
        assert!(error >= 0.0);
    }

    #[test]
    fn direct_squared_essential_sampson_matches_reference() {
        let e = Matrix3::<f64>::new(0.0, -0.3, 0.2, 0.3, 0.0, -0.7, -0.2, 0.7, 0.0);
        let corr = BearingCorrespondence::try_new(
            Vector3::new(0.4, -0.2, 0.9) * 3.0,
            Vector3::new(-0.3, 0.6, 0.7) * 0.25,
        )
        .unwrap();
        let reference = sampson_error_essential(&e, &corr).powi(2);
        let direct = sampson_error_essential_squared(&e, &corr);
        assert!((reference - direct).abs() <= f64::EPSILON * direct.abs().max(1.0) * 4.0);
    }

    #[test]
    fn simplified_tangent_sampson_matches_projected_gradients() {
        let e = Matrix3::<f64>::new(0.0, -0.3, 0.2, 0.3, 0.0, -0.7, -0.2, 0.7, 0.0);
        let corr = BearingCorrespondence::try_new(
            Vector3::new(0.4, -0.2, 0.9) * 3.0,
            Vector3::new(-0.3, 0.6, 0.7) * 0.25,
        )
        .unwrap();
        let b1 = *corr.bearing1();
        let b2 = *corr.bearing2();
        let eb1 = e * b1;
        let etb2 = e.transpose() * b2;
        let constraint = b2.dot(&eb1);
        let grad1 = etb2 - b1 * constraint;
        let grad2 = eb1 - b2 * constraint;
        let reference = constraint * constraint / (grad1.norm_squared() + grad2.norm_squared());
        let simplified = sampson_error_essential_squared(&e, &corr);
        assert!((reference - simplified).abs() <= f64::EPSILON * reference.abs().max(1.0) * 16.0);
    }

    #[test]
    fn pinhole_essential_sampson_is_homogeneous_scale_invariant() {
        let e = Matrix3::new(0.0, -0.3, 0.2, 0.3, 0.0, -0.7, -0.2, 0.7, 0.0);
        let normalized = BearingCorrespondence::try_new(
            Vector3::new(0.4, -0.2, 1.0),
            Vector3::new(-0.3, 0.6, 1.0),
        )
        .unwrap();
        let scaled = BearingCorrespondence::try_new(
            *normalized.bearing1() * 3.7,
            *normalized.bearing2() * 0.23,
        )
        .unwrap();
        let reference = sampson_error_essential_pinhole_squared(&e, &normalized);
        let actual = sampson_error_essential_pinhole_squared(&e, &scaled);
        assert!((reference - actual).abs() <= f64::EPSILON * reference.abs().max(1.0) * 8.0);
    }

    #[test]
    fn homogeneous_pinhole_sampson_matches_dehomogenized_reference() {
        let e = Matrix3::new(0.0, -0.3, 0.2, 0.3, 0.0, -0.7, -0.2, 0.7, 0.0);
        let corr = BearingCorrespondence::try_new(
            Vector3::new(0.8, -0.4, 2.0),
            Vector3::new(-0.15, 0.3, 0.5),
        )
        .unwrap();
        let x1 = corr.bearing1() / corr.bearing1().z;
        let x2 = corr.bearing2() / corr.bearing2().z;
        let ex1 = e * x1;
        let etx2 = e.transpose() * x2;
        let constraint = x2.dot(&ex1);
        let reference = constraint * constraint
            / (ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y);
        let actual = sampson_error_essential_pinhole_squared(&e, &corr);
        assert!((reference - actual).abs() <= f64::EPSILON * reference.abs().max(1.0) * 8.0);
    }

    #[test]
    fn pinhole_essential_sampson_rejects_equatorial_bearing() {
        let e = Matrix3::identity();
        let corr = BearingCorrespondence::try_new(Vector3::x(), Vector3::z()).unwrap();
        assert!(sampson_error_essential_pinhole_squared(&e, &corr).is_infinite());
    }

    #[test]
    fn angular_essential_sampson_supports_equatorial_bearings() {
        let e = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let corr =
            BearingCorrespondence::try_new(Vector3::x(), Vector3::y() + Vector3::z()).unwrap();
        assert!(sampson_error_essential_squared(&e, &corr).is_finite());
    }

    #[test]
    fn test_essential_angular_sampson_is_frame_and_scale_invariant() {
        let e = Matrix3::new(0.0, -0.3, 0.2, 0.3, 0.0, -0.7, -0.2, 0.7, 0.0);
        let b1 = Vector3::new(0.4, -0.2, 0.9).normalize();
        let b2 = Vector3::new(-0.3, 0.6, 0.7).normalize();
        let corr = BearingCorrespondence::try_new(b1 * 3.0, b2 * 0.25).unwrap();
        let reference = sampson_error_essential(&e, &corr);

        let q1 = *Rotation3::from_euler_angles(0.2, -0.4, 0.1).matrix();
        let q2 = *Rotation3::from_euler_angles(-0.3, 0.1, 0.5).matrix();
        let transformed_e = q2 * e * q1.transpose();
        let transformed = BearingCorrespondence::try_new(q1 * b1, q2 * b2).unwrap();
        let rotated = sampson_error_essential(&transformed_e, &transformed);

        assert!(reference.is_finite());
        assert!((reference - rotated).abs() < 1e-12);
    }

    #[test]
    fn test_bearing_correspondence_rejects_zero_bearing() {
        assert!(BearingCorrespondence::try_new(Vector3::<f64>::zeros(), Vector3::z()).is_err());
    }

    #[test]
    fn test_essential_angular_sampson_preserves_small_angle_threshold_scale() {
        // E = [t]_x for t along +X. Perturbing the second bearing by theta
        // gives the two-view Sampson correction tan(theta) / sqrt(2).
        let e = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        let theta: f64 = 0.01;
        let corr = BearingCorrespondence::try_new(
            Vector3::z(),
            Vector3::new(0.0, theta.sin(), theta.cos()),
        )
        .unwrap();

        let residual = sampson_error_essential(&e, &corr);
        let expected = theta.tan() / 2.0_f64.sqrt();
        assert!((residual - expected).abs() < 1e-12);
    }
}
