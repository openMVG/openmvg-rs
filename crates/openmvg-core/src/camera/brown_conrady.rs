// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Brown-Conrady distortion camera model
//!
//! The Brown-Conrady camera model extends the pinhole model with radial and
//! tangential lens distortion. It reuses the `PinholeCamera` implementation and adds
//! configurable distortion parameters.
//!
//! ## Mathematical model
//!
//! The distortion is applied in normalized image coordinates before scaling by intrinsics:
//!
//! 1. Perspective division: `[X, Y, Z] → [X/Z, Y/Z]` (normalized coordinates)
//! 2. Distortion: apply radial and tangential distortion
//! 3. Intrinsic scaling: use `PinholeCamera` to convert to pixel coordinates
//!
//! ### Radial distortion
//!
//! ```text
//! r² = x² + y²
//! radial_factor = 1 + k1*r² + k2*r⁴ + k3*r⁶  (depending on configuration)
//! ```
//!
//! ### Tangential distortion
//!
//! ```text
//! dx_tangential = 2*p1*x*y + p2*(r² + 2*x²)
//! dy_tangential = p1*(r² + 2*y²) + 2*p2*x*y
//! ```
//!
//! ## Projection domain
//!
//! With k3=0 and zero tangential terms, the principal radial branch ends at
//! the first positive zero of `1 + 3*k1*r² + 5*k2*r⁴`. Projection and inverse
//! acceptance use this branch and local endpoint conditioning. Other models
//! check the axis-to-point path at 33 samples; this is a sampled domain test,
//! not a proof of global invertibility for arbitrary coefficients.
//!
//! ## Inversion
//!
//! Unprojection removes the distortion with a damped Newton solve in normalized image
//! coordinates, starting at the distorted point. Each iteration takes the Newton step of the
//! 2×2 distortion Jacobian and halves it, at most 16 times, until the residual decreases. The
//! solve accepts a squared residual of at most 1e-24 within 30 iterations, then checks the
//! domain above. It fails with `DistortionError::NonFinite` on a non-finite value,
//! `SingularJacobian` when the Jacobian determinant is at most 1e-15, `DidNotConverge` (with the
//! iteration it stopped at) when the line search stalls or the iterations run out, and
//! `OutsideInvertibleDomain` when the converged point lies off the principal branch.
//!
//! ## References
//!
//! - Brown, D.C. (1966). "Decentering Distortion of Lenses."
//!   Photogrammetric Engineering, 32(3), 444-462.

use crate::camera::camera_model::{CameraError, CameraModel, ProjectionError};
use crate::camera::distortion::{
    RadialDistortion, TangentialDistortion, apply_distortion, distortion_jacobian,
    first_radial_turn_squared, is_validated_distortion_invertible_at, remove_validated_distortion,
};
use crate::camera::pinhole::PinholeCamera;
use nalgebra::{Matrix2x3, Matrix3, Point2, Vector3};
use serde::{Deserialize, Deserializer};

/// Pinhole camera with Brown-Conrady radial and tangential lens distortion.
///
/// The perspective part is a [`PinholeCamera`] (single or double focal length).
/// Projection applies the distortion in normalized image coordinates; unprojection
/// removes it iteratively.
///
/// A camera-frame point `(X, Y, Z)` projects as follows; the tangential offsets use the
/// undistorted `x` and `y`:
///
/// ```text
/// x = X / Z,  y = Y / Z,  r² = x² + y²
/// radial = 1 + k1 r² + k2 r⁴ + k3 r⁶
/// x_d = x · radial + 2 p1 x y + p2 (r² + 2 x²)
/// y_d = y · radial + p1 (r² + 2 y²) + 2 p2 x y
/// u = fx · x_d + cx,  v = fy · y_d + cy
/// ```
///
/// Unprojection inverts the distortion with a damped Newton solve, which converges to a
/// squared residual of at most `1e-24` in normalized image units, about `1e-12` in `x_d` and
/// `y_d`. It returns [`ProjectionError::DistortionInversion`] when the solve does not converge
/// within 30 iterations or meets a singular Jacobian.
///
/// The distortion is one-to-one only out to a radius. With radial `k1`, `k2` only, that radius
/// is the first positive zero of `1 + 3 k1 r² + 5 k2 r⁴`; with `k3` or tangential terms the
/// path from the optical axis to the point is checked at 33 samples. Projection rejects points
/// past that radius with [`ProjectionError::OutsideModelDomain`]. A pixel beyond the image of
/// that radius has no valid preimage: unprojection usually fails with `DistortionInversion`, and
/// returns `OutsideModelDomain` when the solve converges to a point past the radius.
///
/// The distortion is OpenCV's five-coefficient model, `plumb_bob` in ROS: OpenCV's
/// `D = [k1, k2, p1, p2, k3]` is [`RadialDistortion::K1K2K3`]`(k1, k2, k3)` with
/// [`TangentialDistortion::P1P2`]`(p1, p2)`. Trailing zero coefficients may be left out:
/// `K1K2(a, b)` and `K1K2K3(a, b, 0.0)` project identically, and `openmvg_io::write_colmap`
/// exports both as the smallest COLMAP model that holds them.
///
/// # Examples
///
/// Barrel distortion of a wide-angle lens:
///
/// ```rust
/// use openmvg_core::camera::{BrownConradyCamera, PinholeCamera, RadialDistortion, TangentialDistortion, CameraModel};
/// use nalgebra::Vector3;
///
/// // Create underlying pinhole camera (single focal length)
/// let pinhole = PinholeCamera::try_with_focal(600.0, 320.0, 240.0)?;
///
/// // Add barrel distortion
/// let camera = BrownConradyCamera::try_new(
///     pinhole,
///     RadialDistortion::K1(-0.15),  // Negative k1 = barrel distortion
///     TangentialDistortion::None
/// )?;
///
/// // Project; the distortion is applied after perspective division
/// let point = Vector3::new(1.0, 0.5, 2.0);
/// let pixel = camera.project(&point)?;
/// // Barrel distortion pulls the pixel toward the center; without it the pixel is (620, 390).
/// assert!((pixel.x - 605.9375).abs() < 1e-9 && (pixel.y - 382.96875).abs() < 1e-9);
///
/// // Unprojection removes the distortion and gives back the direction of the point.
/// let ray = camera.unproject(&pixel)?;
/// assert!((ray / ray.z - point / point.z).norm() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrownConradyCamera {
    /// The underlying pinhole camera for basic perspective projection.
    pinhole: PinholeCamera,
    /// Radial distortion configuration.
    radial: RadialDistortion,
    /// Tangential distortion configuration.
    tangential: TangentialDistortion,
    /// Derived bound for the radial-only two-coefficient branch; never serialized.
    #[serde(skip)]
    radial_turn_squared: Option<f64>,
}

impl BrownConradyCamera {
    /// Creates a Brown-Conrady camera from a pinhole camera and distortion coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::InvalidDistortion`] if a coefficient is non-finite.
    pub fn try_new(
        pinhole: PinholeCamera,
        radial: RadialDistortion,
        tangential: TangentialDistortion,
    ) -> Result<Self, CameraError> {
        radial
            .validate()
            .and_then(|()| tangential.validate())
            .map_err(|_| CameraError::InvalidDistortion {
                reason: "Brown-Conrady coefficients must be finite",
            })?;
        let radial_turn_squared = first_radial_turn_squared(radial.k1(), radial.k2());
        Ok(Self {
            pinhole,
            radial,
            tangential,
            radial_turn_squared,
        })
    }

    /// Creates a Brown-Conrady camera with a single focal length `f`, in pixels.
    ///
    /// The principal point `(cx, cy)` is in pixels. For separate `fx` and `fy`, pass a
    /// [`PinholeCamera::try_new`] camera to [`Self::try_new`], or use
    /// [`Camera::try_brown_conrady`](super::Camera::try_brown_conrady).
    ///
    /// # Errors
    ///
    /// Returns [`CameraError`] when the pinhole part is invalid or a distortion
    /// coefficient is non-finite.
    pub fn try_with_focal(
        f: f64,
        cx: f64,
        cy: f64,
        radial: RadialDistortion,
        tangential: TangentialDistortion,
    ) -> Result<Self, CameraError> {
        let pinhole = PinholeCamera::try_with_focal(f, cx, cy)?;
        Self::try_new(pinhole, radial, tangential)
    }

    // Accessor methods for the underlying pinhole camera

    /// Returns a reference to the underlying pinhole camera.
    pub fn pinhole(&self) -> &PinholeCamera {
        &self.pinhole
    }

    /// Returns the horizontal focal length from the underlying pinhole camera.
    pub fn focal_x(&self) -> f64 {
        self.pinhole.focal_x()
    }

    /// Returns the vertical focal length from the underlying pinhole camera.
    pub fn focal_y(&self) -> f64 {
        self.pinhole.focal_y()
    }

    /// Returns the x-coordinate of the principal point.
    pub fn principal_x(&self) -> f64 {
        self.pinhole.principal_x()
    }

    /// Returns the y-coordinate of the principal point.
    pub fn principal_y(&self) -> f64 {
        self.pinhole.principal_y()
    }

    // Accessor methods for distortion parameters

    /// Returns a reference to the radial distortion configuration.
    pub fn radial_distortion(&self) -> &RadialDistortion {
        &self.radial
    }

    /// Returns a reference to the tangential distortion configuration.
    pub fn tangential_distortion(&self) -> &TangentialDistortion {
        &self.tangential
    }

    /// Returns whether this camera has any distortion (radial or tangential).
    pub fn has_distortion(&self) -> bool {
        !matches!(self.radial, RadialDistortion::None)
            || !matches!(self.tangential, TangentialDistortion::None)
    }

    /// Applies distortion to normalized image coordinates.
    fn apply_distortion(&self, x: f64, y: f64) -> (f64, f64) {
        apply_distortion(x, y, &self.radial, &self.tangential)
    }

    /// Check the principal branch before evaluating the forward polynomial.
    fn validate_forward_domain(&self, x: f64, y: f64) -> Result<(), ProjectionError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(ProjectionError::NonFinite);
        }
        if !self.has_distortion() {
            return Ok(());
        }
        if self.radial.k3() == 0.0 && self.tangential.p1() == 0.0 && self.tangential.p2() == 0.0 {
            let radius_squared = x.mul_add(x, y * y);
            if !radius_squared.is_finite() {
                return Err(ProjectionError::NonFinite);
            }
            if self
                .radial_turn_squared
                .is_some_and(|bound| radius_squared >= bound)
            {
                return Err(ProjectionError::OutsideModelDomain);
            }
            let (_, _, _, determinant) = distortion_jacobian(x, y, &self.radial, &self.tangential)
                .ok_or(ProjectionError::NonFinite)?;
            if determinant <= 1e-12 {
                return Err(ProjectionError::OutsideModelDomain);
            }
        } else if !is_validated_distortion_invertible_at(x, y, &self.radial, &self.tangential) {
            return Err(ProjectionError::OutsideModelDomain);
        }
        Ok(())
    }

    /// Removes distortion from normalized image coordinates.
    fn remove_distortion(
        &self,
        x_distorted: f64,
        y_distorted: f64,
    ) -> Result<(f64, f64), ProjectionError> {
        remove_validated_distortion(x_distorted, y_distorted, &self.radial, &self.tangential)
            .map_err(ProjectionError::from)
    }
}

impl<'de> Deserialize<'de> for BrownConradyCamera {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            pinhole: PinholeCamera,
            radial: RadialDistortion,
            tangential: TangentialDistortion,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::try_new(repr.pinhole, repr.radial, repr.tangential).map_err(serde::de::Error::custom)
    }
}

impl PartialEq for BrownConradyCamera {
    fn eq(&self, other: &Self) -> bool {
        self.pinhole == other.pinhole
            && self.radial == other.radial
            && self.tangential == other.tangential
    }
}

impl CameraModel for BrownConradyCamera {
    /// Projects a 3D point to 2D pixel coordinates, applying the lens distortion.
    ///
    /// The point is divided by its depth, distorted in normalized image coordinates, then
    /// scaled by the focal lengths and shifted by the principal point.
    fn project(&self, point: &Vector3<f64>) -> Result<Point2<f64>, ProjectionError> {
        // Validate input: check for non-finite coordinates
        if !point.iter().all(|v| v.is_finite()) {
            return Err(ProjectionError::NonFinite);
        }

        // Check if point is behind the camera
        if point[2] <= 0.0 {
            return Err(ProjectionError::BehindCamera);
        }
        // Step 1: Perspective division to normalized image coordinates
        let x_norm = point[0] / point[2];
        let y_norm = point[1] / point[2];
        self.validate_forward_domain(x_norm, y_norm)?;

        // Step 2: Apply distortion
        let (x_dist, y_dist) = self.apply_distortion(x_norm, y_norm);

        // Step 3: Convert to pixel coordinates using the underlying pinhole camera
        let distorted_point = Vector3::new(x_dist, y_dist, 1.0);

        self.pinhole.project(&distorted_point)
    }

    fn project_with_jacobian(
        &self,
        point: &Vector3<f64>,
    ) -> Result<(Point2<f64>, Matrix2x3<f64>), ProjectionError> {
        let pixel = self.project(point)?;
        let inverse_depth = point.z.recip();
        let x = point.x * inverse_depth;
        let y = point.y * inverse_depth;
        let (j11, j12, j22, _) = distortion_jacobian(x, y, &self.radial, &self.tangential)
            .ok_or(ProjectionError::NonFinite)?;
        let fx = self.focal_x();
        let fy = self.focal_y();
        let jacobian = Matrix2x3::new(
            fx * j11 * inverse_depth,
            fx * j12 * inverse_depth,
            -fx * (j11 * x + j12 * y) * inverse_depth,
            fy * j12 * inverse_depth,
            fy * j22 * inverse_depth,
            -fy * (j12 * x + j22 * y) * inverse_depth,
        );
        if jacobian.iter().all(|value| value.is_finite()) {
            Ok((pixel, jacobian))
        } else {
            Err(ProjectionError::NonFinite)
        }
    }

    /// Unprojects a 2D pixel coordinate to a 3D ray direction with distortion removal.
    ///
    /// The distortion is removed iteratively, and the ray is returned at unit depth:
    /// `(x, y, 1)` in normalized image coordinates.
    fn unproject(&self, pixel: &Point2<f64>) -> Result<Vector3<f64>, ProjectionError> {
        // Validate input
        if !pixel.coords.iter().all(|v| v.is_finite()) {
            return Err(ProjectionError::NonFinite);
        }

        // Step 1: Use pinhole camera to convert to normalized coordinates (distorted)
        let ray_distorted = self.pinhole.unproject(pixel)?;

        // Extract normalized coordinates (assuming ray_distorted.z is close to 1.0)
        let x_dist = ray_distorted.x / ray_distorted.z;
        let y_dist = ray_distorted.y / ray_distorted.z;

        // Step 2: Remove distortion
        let (x_norm, y_norm) = self.remove_distortion(x_dist, y_dist)?;

        // Step 3: Create 3D ray direction
        Ok(Vector3::new(x_norm, y_norm, 1.0))
    }

    /// Returns the principal point of the camera.
    fn principal_point(&self) -> Point2<f64> {
        self.pinhole.principal_point()
    }

    /// Returns the intrinsic matrix K (from the underlying pinhole camera).
    fn intrinsic_matrix(&self) -> Option<Matrix3<f64>> {
        self.pinhole.intrinsic_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinhole() -> PinholeCamera {
        PinholeCamera::try_with_focal(600.0, 320.0, 240.0).expect("valid pinhole")
    }

    #[test]
    fn constructor_rejects_non_finite_distortion() {
        for coefficient in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                BrownConradyCamera::try_new(
                    pinhole(),
                    RadialDistortion::K1(coefficient),
                    TangentialDistortion::None,
                )
                .is_err()
            );
            assert!(
                BrownConradyCamera::try_new(
                    pinhole(),
                    RadialDistortion::None,
                    TangentialDistortion::P1(coefficient),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn radial_principal_domain_rejects_fold_and_preserves_round_trip() {
        let camera = BrownConradyCamera::try_with_focal(
            500.0,
            320.0,
            240.0,
            RadialDistortion::K1(-0.1),
            TangentialDistortion::None,
        )
        .unwrap();
        let folded = Vector3::new(10.0_f64.sqrt(), 0.0, 1.0);
        assert_eq!(
            camera.project(&folded),
            Err(ProjectionError::OutsideModelDomain)
        );
        assert_eq!(
            camera.project_with_jacobian(&folded),
            Err(ProjectionError::OutsideModelDomain)
        );
        let point = Vector3::new(0.5, 0.2, 1.0);
        let pixel = camera.project(&point).unwrap();
        assert!((camera.unproject(&pixel).unwrap().normalize() - point.normalize()).norm() < 1e-12);
        let serialized = serde_json::to_value(&camera).unwrap();
        assert_eq!(serialized.as_object().unwrap().len(), 3);
        assert!(serialized.get("radial_turn_squared").is_none());
        let restored: BrownConradyCamera = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored, camera);
        assert_eq!(restored.radial_turn_squared, camera.radial_turn_squared);
    }

    #[test]
    fn radial_turn_bound_handles_extreme_finite_coefficients() {
        let bound = first_radial_turn_squared(-1e308, 0.0).unwrap();
        assert!(bound.is_finite() && bound > 0.0);
        assert!((bound / (1.0 / 3.0 / 1e308) - 1.0).abs() < 1e-14);
        let camera = BrownConradyCamera::try_with_focal(
            1.0,
            0.0,
            0.0,
            RadialDistortion::K1(-1e308),
            TangentialDistortion::None,
        )
        .unwrap();
        let regular = Vector3::new((bound * 0.25).sqrt(), 0.0, 1.0);
        assert!(camera.project_with_jacobian(&regular).is_ok());
        let folded = Vector3::new((bound * 2.0).sqrt(), 0.0, 1.0);
        assert_eq!(
            camera.project(&folded),
            Err(ProjectionError::OutsideModelDomain)
        );
        let bound_k2 = first_radial_turn_squared(0.0, -1e308).unwrap();
        assert!(bound_k2.is_finite() && bound_k2 > 0.0);
        let camera_k2 = BrownConradyCamera::try_with_focal(
            1.0,
            0.0,
            0.0,
            RadialDistortion::K1K2(0.0, -1e308),
            TangentialDistortion::None,
        )
        .unwrap();
        assert!(
            camera_k2
                .project_with_jacobian(&Vector3::new((bound_k2 * 0.25).sqrt(), 0.0, 1.0))
                .is_ok()
        );
    }

    #[test]
    fn serde_round_trip_preserves_valid_camera() {
        let camera = BrownConradyCamera::try_new(
            pinhole(),
            RadialDistortion::K1K2(-0.1, 0.01),
            TangentialDistortion::P1P2(0.001, -0.002),
        )
        .expect("valid camera");

        let json = serde_json::to_string(&camera).expect("serialize camera");
        let recovered = serde_json::from_str(&json).expect("deserialize camera");
        assert_eq!(camera, recovered);
    }

    #[test]
    fn serde_rejects_invalid_or_malformed_camera() {
        for json in [
            r#"{"pinhole":{"fx":0.0,"fy":0.0,"cx":320.0,"cy":240.0},"radial":"none","tangential":"none"}"#,
            r#"{"pinhole":{"fx":-1.0,"fy":-1.0,"cx":320.0,"cy":240.0},"radial":"none","tangential":"none"}"#,
            r#"{"pinhole":{"fx":600.0,"fy":600.0,"cx":320.0,"cy":240.0},"radial":"none"}"#,
            r#"{"pinhole":"invalid","radial":"none","tangential":"none"}"#,
        ] {
            assert!(
                serde_json::from_str::<BrownConradyCamera>(json).is_err(),
                "accepted invalid camera JSON: {json}"
            );
        }
    }
}
