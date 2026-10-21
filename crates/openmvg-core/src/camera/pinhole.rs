// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pinhole camera model
//!
//! The classic pinhole camera model with focal lengths `fx`, `fy` and principal point
//! `cx`, `cy`. The fields are private so every instance goes through the constructor checks.
//!
//! ## Projection equation
//!
//! ```text
//! ┌───┐   ┌             ┐ ┌───┐
//! │ u │   │ fx  0   cx  │ │ X │
//! │ v │ = │ 0   fy  cy  │ │ Y │ / Z
//! │ 1 │   │ 0   0   1   │ │ Z │
//! └───┘   └             ┘ └───┘
//! ```
//!
//! Where:
//! - `(u, v)` = pixel coordinates
//! - `(X, Y, Z)` = 3D point in camera frame
//! - `fx, fy` = focal lengths in pixels
//! - `(cx, cy)` = principal point, usually near the image center
//!
//! ## Parameters
//!
//! | Parameter | Type | Description | Typical range |
//! |-----------|------|-------------|---------------|
//! | `fx` | f64 | Horizontal focal length | 500-3000 pixels |
//! | `fy` | f64 | Vertical focal length | 500-3000 pixels |
//! | `cx` | f64 | Principal point x | ~width/2 |
//! | `cy` | f64 | Principal point y | ~height/2 |

use crate::camera::camera_model::{CameraError, CameraModel, ProjectionError};
use nalgebra::{Matrix2x3, Matrix3, Point2, Vector3};

/// A pinhole camera model with intrinsic parameters.
///
/// Its fields are private, so every instance has finite, positive focal lengths and a
/// finite principal point.
///
/// # Mathematical model and coordinate systems
///
/// The pinhole camera follows the standard perspective projection model:
///
/// ```text
/// K = [fx  0  cx]    fx, fy: focal lengths (pixels)
///     [ 0 fy  cy]    cx, cy: principal point (pixels)  
///     [ 0  0   1]    (homogeneous coordinates)
/// ```
///
/// ## Coordinate systems
///
/// 3D camera coordinates (right-handed system, OpenCV convention):
/// ```text
///       Y (down in image coords)
///       |
///       |
///       +----> X (right)
///      /
///     /
///    Z (into scene / depth)
/// ```
///
/// - Positive Z points into the scene (away from the camera)
/// - Points with Z > 0 are in front of the camera and can be projected
/// - Points with Z ≤ 0 are behind the camera and return [`ProjectionError::BehindCamera`]
///
/// 2D image coordinates (standard computer-vision convention):
/// ```text
///    (0,0) ------> u (image width)
///      |
///      |  (cx,cy) = principal point
///      |     +
///      |
///      v (image height)
/// ```
///
/// ## Focal lengths
///
/// [`PinholeCamera::try_new`] takes separate horizontal and vertical focal lengths and
/// [`PinholeCamera::try_with_focal`] one focal length for square pixels (`fx = fy`); both
/// store the same four parameters.
///
/// # Examples
///
/// ```rust
/// use openmvg_core::{PinholeCamera, CameraModel};
/// use nalgebra::Vector3;
///
/// let camera = PinholeCamera::try_with_focal(800.0, 320.0, 240.0)?;
/// assert_eq!(camera.focal_x(), 800.0);
/// assert_eq!(camera.focal_y(), 800.0);
///
/// let camera = PinholeCamera::try_new(800.0, 600.0, 320.0, 240.0)?;
/// assert_eq!(camera.focal_y(), 600.0);
///
/// // u = fx·X/Z + cx = 800·0.5 + 320, v = fy·Y/Z + cy = 600·0.25 + 240.
/// let point = Vector3::new(1.0, 0.5, 2.0);
/// let pixel = camera.project(&point)?;
/// assert!((pixel.x - 720.0).abs() < 1e-9 && (pixel.y - 390.0).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PinholeCamera {
    /// Horizontal focal length in pixels.
    fx: f64,
    /// Vertical focal length in pixels.
    fy: f64,
    /// Principal point x-coordinate in pixels.
    cx: f64,
    /// Principal point y-coordinate in pixels.
    cy: f64,
}

impl<'de> serde::Deserialize<'de> for PinholeCamera {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Repr {
            fx: f64,
            fy: f64,
            cx: f64,
            cy: f64,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::try_new(repr.fx, repr.fy, repr.cx, repr.cy).map_err(serde::de::Error::custom)
    }
}

impl PinholeCamera {
    /// Creates a pinhole camera with a single focal length (square pixels, `focal_x = focal_y = focal`).
    ///
    /// # Errors
    ///
    /// Returns [`CameraError`] when `focal` is not finite and positive or the principal point is
    /// not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::PinholeCamera;
    ///
    /// let camera = PinholeCamera::try_with_focal(800.0, 320.0, 240.0)?;
    /// assert_eq!(camera.focal_x(), 800.0);
    /// assert_eq!(camera.focal_y(), 800.0);
    /// # Ok::<(), openmvg_core::CameraError>(())
    /// ```
    pub fn try_with_focal(
        focal: f64,
        principal_x: f64,
        principal_y: f64,
    ) -> Result<Self, CameraError> {
        if focal <= 0.0 {
            return Err(CameraError::InvalidFocalLength {
                value: focal,
                reason: "focal length must be positive",
            });
        }
        if !focal.is_finite() {
            return Err(CameraError::InvalidFocalLength {
                value: focal,
                reason: "focal length must be finite (not NaN or infinity)",
            });
        }
        validate_principal_point(principal_x, principal_y)?;
        Ok(Self {
            fx: focal,
            fy: focal,
            cx: principal_x,
            cy: principal_y,
        })
    }

    /// Creates a pinhole camera with separate horizontal and vertical focal lengths.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError`] when a focal length is not finite and positive or the
    /// principal point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::PinholeCamera;
    ///
    /// let camera = PinholeCamera::try_new(800.0, 600.0, 320.0, 240.0)?;
    /// assert_eq!(camera.focal_x(), 800.0);
    /// assert_eq!(camera.focal_y(), 600.0);
    /// # Ok::<(), openmvg_core::CameraError>(())
    /// ```
    pub fn try_new(
        focal_x: f64,
        focal_y: f64,
        principal_x: f64,
        principal_y: f64,
    ) -> Result<Self, CameraError> {
        if focal_x <= 0.0 {
            return Err(CameraError::InvalidFocalLength {
                value: focal_x,
                reason: "horizontal focal length (focal_x) must be positive",
            });
        }
        if focal_y <= 0.0 {
            return Err(CameraError::InvalidFocalLength {
                value: focal_y,
                reason: "vertical focal length (focal_y) must be positive",
            });
        }
        if !focal_x.is_finite() {
            return Err(CameraError::InvalidFocalLength {
                value: focal_x,
                reason: "horizontal focal length (focal_x) must be finite",
            });
        }
        if !focal_y.is_finite() {
            return Err(CameraError::InvalidFocalLength {
                value: focal_y,
                reason: "vertical focal length (focal_y) must be finite",
            });
        }
        validate_principal_point(principal_x, principal_y)?;
        Ok(Self {
            fx: focal_x,
            fy: focal_y,
            cx: principal_x,
            cy: principal_y,
        })
    }

    /// Horizontal focal length in pixels.
    #[inline]
    #[must_use]
    pub const fn focal_x(&self) -> f64 {
        self.fx
    }

    /// Vertical focal length in pixels.
    #[inline]
    #[must_use]
    pub const fn focal_y(&self) -> f64 {
        self.fy
    }

    /// Principal point x-coordinate in pixels.
    #[inline]
    #[must_use]
    pub const fn principal_x(&self) -> f64 {
        self.cx
    }

    /// Principal point y-coordinate in pixels.
    #[inline]
    #[must_use]
    pub const fn principal_y(&self) -> f64 {
        self.cy
    }
}

fn validate_principal_point(principal_x: f64, principal_y: f64) -> Result<(), CameraError> {
    if !principal_x.is_finite() || !principal_y.is_finite() {
        return Err(CameraError::InvalidPrincipalPoint {
            principal_x,
            principal_y,
            reason: "principal point coordinates must be finite",
        });
    }
    Ok(())
}

impl CameraModel for PinholeCamera {
    /// Projects a 3D point to 2D pixel coordinates using the pinhole camera model.
    fn project(&self, point: &Vector3<f64>) -> Result<Point2<f64>, ProjectionError> {
        // Validate input: check for non-finite coordinates (NaN, infinity)
        if !point.iter().all(|v| v.is_finite()) {
            return Err(ProjectionError::NonFinite);
        }

        // Check if point is behind the camera (non-positive Z coordinate)
        if point[2] <= 0.0 {
            return Err(ProjectionError::BehindCamera);
        }

        let inverse_depth = point.z.recip();
        let pixel = Point2::new(
            self.focal_x()
                .mul_add(point.x * inverse_depth, self.principal_x()),
            self.focal_y()
                .mul_add(point.y * inverse_depth, self.principal_y()),
        );
        if pixel.coords.iter().all(|value| value.is_finite()) {
            Ok(pixel)
        } else {
            Err(ProjectionError::NonFinite)
        }
    }

    fn project_with_jacobian(
        &self,
        point: &Vector3<f64>,
    ) -> Result<(Point2<f64>, Matrix2x3<f64>), ProjectionError> {
        let pixel = self.project(point)?;
        let inverse_depth = point.z.recip();
        let inverse_depth_squared = inverse_depth * inverse_depth;
        let jacobian = Matrix2x3::new(
            self.focal_x() * inverse_depth,
            0.0,
            -self.focal_x() * point.x * inverse_depth_squared,
            0.0,
            self.focal_y() * inverse_depth,
            -self.focal_y() * point.y * inverse_depth_squared,
        );
        if jacobian.iter().all(|value| value.is_finite()) {
            Ok((pixel, jacobian))
        } else {
            Err(ProjectionError::NonFinite)
        }
    }

    /// Unprojects a 2D pixel coordinate to a 3D ray direction.
    fn unproject(&self, pixel: &Point2<f64>) -> Result<Vector3<f64>, ProjectionError> {
        // Validate input
        if !pixel.coords.iter().all(|v| v.is_finite()) {
            return Err(ProjectionError::NonFinite);
        }

        let bearing = Vector3::new(
            (pixel.x - self.principal_x()) / self.focal_x(),
            (pixel.y - self.principal_y()) / self.focal_y(),
            1.0,
        );
        if bearing.iter().all(|value| value.is_finite()) {
            Ok(bearing)
        } else {
            Err(ProjectionError::NonFinite)
        }
    }

    /// Returns the principal point of the camera.
    fn principal_point(&self) -> Point2<f64> {
        Point2::new(self.principal_x(), self.principal_y())
    }

    /// Returns the intrinsic matrix K.
    fn intrinsic_matrix(&self) -> Option<Matrix3<f64>> {
        Some(Matrix3::new(
            self.focal_x(),
            0.0,
            self.principal_x(),
            0.0,
            self.focal_y(),
            self.principal_y(),
            0.0,
            0.0,
            1.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_preserve_calibration() {
        let square = PinholeCamera::try_with_focal(800.0, 320.0, 240.0).unwrap();
        assert_eq!(square.focal_x(), 800.0);
        assert_eq!(square.focal_y(), 800.0);
        assert_eq!(square.principal_point(), Point2::new(320.0, 240.0));

        let separate = PinholeCamera::try_new(800.0, 600.0, 320.0, 240.0).unwrap();
        assert_eq!(
            separate.intrinsic_matrix().unwrap(),
            Matrix3::new(800.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn constructors_reject_invalid_intrinsics() {
        for focal in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(PinholeCamera::try_with_focal(focal, 0.0, 0.0).is_err());
            assert!(PinholeCamera::try_new(focal, 1.0, 0.0, 0.0).is_err());
            assert!(PinholeCamera::try_new(1.0, focal, 0.0, 0.0).is_err());
        }

        for principal in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(PinholeCamera::try_with_focal(1.0, principal, 0.0).is_err());
            assert!(PinholeCamera::try_with_focal(1.0, 0.0, principal).is_err());
        }
    }

    #[test]
    fn projection_rejects_finite_input_that_overflows() {
        let camera = PinholeCamera::try_with_focal(f64::MAX, 0.0, 0.0).unwrap();
        let point = Vector3::new(f64::MAX, 0.0, f64::MIN_POSITIVE);
        assert_eq!(camera.project(&point), Err(ProjectionError::NonFinite));

        let camera = PinholeCamera::try_with_focal(f64::MIN_POSITIVE, -f64::MAX, 0.0).unwrap();
        assert_eq!(
            camera.unproject(&Point2::new(f64::MAX, 0.0)),
            Err(ProjectionError::NonFinite)
        );
    }

    #[test]
    fn serialization_stores_only_canonical_parameters() {
        let camera = PinholeCamera::try_new(800.0, 600.0, 320.0, 240.0).unwrap();
        let json = serde_json::to_string(&camera).unwrap();

        assert!(!json.contains("k_inv"));
        assert!(!json.contains("\"k\""));
        assert_eq!(
            serde_json::from_str::<PinholeCamera>(&json).unwrap(),
            camera
        );
    }

    #[test]
    fn deserialization_rejects_invalid_intrinsics() {
        for json in [
            r#"{"fx":0.0,"fy":800.0,"cx":320.0,"cy":240.0}"#,
            r#"{"fx":-1.0,"fy":800.0,"cx":320.0,"cy":240.0}"#,
            r#"{"fx":800.0,"fy":0.0,"cx":320.0,"cy":240.0}"#,
            r#"{"fx":800.0,"fy":600.0,"cx":"invalid","cy":240.0}"#,
            r#"{"fx":800.0,"fy":800.0,"cx":320.0}"#,
        ] {
            assert!(
                serde_json::from_str::<PinholeCamera>(json).is_err(),
                "accepted invalid camera JSON: {json}"
            );
        }
    }
}
