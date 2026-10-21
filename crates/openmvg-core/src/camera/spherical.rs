// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spherical camera model
//!
//! A spherical camera model for 360-degree and omnidirectional imaging.
//! This model maps 3D points to spherical coordinates and then to equirectangular projection
//! on a 2D image plane with a fixed 2:1 aspect ratio.
//!
//! ## Mathematical model
//!
//! The spherical camera converts 3D points to spherical coordinates and then to image coordinates:
//!
//! 1. 3D to spherical: a 3D point P = [X, Y, Z]ᵀ is converted to spherical coordinates:
//!    - longitude = `atan2(X, Z)` in [-pi, pi]
//!    - latitude = `atan2(Y, hypot(X, Z))` in [-pi/2, pi/2]
//!
//! 2. Spherical to image: the spherical coordinates are mapped to pixel coordinates:
//!    - `u = (longitude + pi) * width / (2*pi) - 0.5`, wrapped into `[-0.5, width - 0.5)`
//!    - `v = (latitude + pi/2) * height / pi - 0.5` where `height = width / 2`
//!
//! Integer pixel coordinates are pixel centers, as in the pinhole model, so the image
//! edges sit at -0.5 and `width - 0.5`. The poles lie on the top and bottom image edges,
//! and the optical axis (+Z) projects to the image center `((width - 1) / 2, (height - 1) / 2)`.
//!
//! ## Coordinate systems
//!
//! 3D camera coordinates (same as the pinhole model, OpenCV convention):
//! ```text
//!       Y (down in image coords)
//!       |
//!       |
//!       +----> X (right)
//!      /
//!     /
//!    Z (into scene / depth)
//! ```
//!
//! Unlike pinhole cameras, spherical cameras are omnidirectional: they see in all
//! directions, including behind the camera (negative Z). The `project()` method
//! therefore does not check the sign of Z.

use crate::camera::camera_model::{
    CameraError, CameraModel, ProjectionError, try_normalize_bearing,
};
use nalgebra::{Matrix2x3, Matrix3, Point2, Vector3};
use std::f64::consts::PI;

/// Spherical camera model for 360-degree omnidirectional imaging.
///
/// It models panoramic and 360-degree cameras that capture the entire sphere around
/// the camera center. Ray directions are expressed as (longitude, latitude) and mapped
/// to an equirectangular image with a 2:1 aspect ratio (width:height).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SphericalCamera {
    /// Image width in pixels.
    width: u32,
}

impl<'de> serde::Deserialize<'de> for SphericalCamera {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct SerializedCamera {
            width: u32,
        }

        let serialized = <SerializedCamera as serde::Deserialize>::deserialize(deserializer)?;
        Self::try_new(serialized.width).map_err(serde::de::Error::custom)
    }
}

impl SphericalCamera {
    /// Creates a spherical camera from its image width in pixels.
    ///
    /// The width spans 360° of longitude. The height is `width / 2`, the 2:1
    /// aspect ratio of equirectangular images.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::InvalidWidth`] when:
    /// - Width is zero
    /// - Width is odd (would result in non-integer height)
    pub fn try_new(width: u32) -> Result<Self, CameraError> {
        // Validate width
        if width == 0 {
            return Err(CameraError::InvalidWidth {
                width,
                reason: "width must be non-zero",
            });
        }

        // Width must be even to ensure integer height
        if !width.is_multiple_of(2) {
            return Err(CameraError::InvalidWidth {
                width,
                reason: "width must be even for 2:1 aspect ratio (height = width/2)",
            });
        }

        Ok(Self { width })
    }

    /// Returns the image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    ///
    /// The height is always exactly half the width to maintain a 2:1 aspect ratio.
    pub fn height(&self) -> u32 {
        self.width / 2
    }

    /// Canonicalize an equirectangular pixel on the horizontal seam.
    ///
    /// Longitude is periodic, so horizontal coordinates are wrapped into
    /// `[-0.5, width - 0.5)`. Latitude is not periodic: vertical coordinates outside
    /// the closed spherical image domain `[-0.5, height - 0.5]` are rejected. The two
    /// vertical endpoints are the image edges and represent the poles.
    fn canonicalize_pixel(&self, pixel: &Point2<f64>) -> Result<Point2<f64>, ProjectionError> {
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return Err(ProjectionError::NonFinite);
        }
        let height = f64::from(self.height());
        if pixel.y < -0.5 || pixel.y > height - 0.5 {
            return Err(ProjectionError::OutsideModelDomain);
        }
        Ok(Point2::new(
            (pixel.x + 0.5).rem_euclid(f64::from(self.width)) - 0.5,
            pixel.y,
        ))
    }

    /// Converts a 3D camera-frame point to `(longitude, latitude)` in radians.
    ///
    /// Longitude rotates about the camera Y axis and positive latitude points down,
    /// matching image coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::NonFinite`] for non-finite input and
    /// [`ProjectionError::OutsideModelDomain`] for the zero vector.
    pub(crate) fn cartesian_to_spherical(
        &self,
        point: &Vector3<f64>,
    ) -> Result<(f64, f64), ProjectionError> {
        let direction = try_normalize_bearing(point)?;
        let longitude = direction.x.atan2(direction.z);
        let latitude = direction.y.atan2(direction.x.hypot(direction.z));
        Ok((longitude, latitude))
    }

    /// Converts spherical coordinates to a 3D unit vector in the camera frame.
    ///
    /// `longitude` is the horizontal angle in radians in [-pi, pi]; `latitude` is the
    /// downward vertical angle in radians in [-pi/2, pi/2].
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::NonFinite`] for non-finite angles and
    /// [`ProjectionError::OutsideModelDomain`] when latitude is outside
    /// `[-pi/2, pi/2]`.
    pub(crate) fn spherical_to_cartesian(
        &self,
        longitude: f64,
        latitude: f64,
    ) -> Result<Vector3<f64>, ProjectionError> {
        if !longitude.is_finite() || !latitude.is_finite() {
            return Err(ProjectionError::NonFinite);
        }
        if !(-PI / 2.0..=PI / 2.0).contains(&latitude) {
            return Err(ProjectionError::OutsideModelDomain);
        }
        let cos_latitude = latitude.cos();
        let x = cos_latitude * longitude.sin();
        let y = latitude.sin();
        let z = cos_latitude * longitude.cos();
        Ok(Vector3::new(x, y, z))
    }
}

impl CameraModel for SphericalCamera {
    /// Projects a 3D point to 2D pixel coordinates using spherical projection.
    ///
    /// This implementation converts 3D Cartesian coordinates to spherical coordinates
    /// and then maps them to equirectangular image coordinates.
    ///
    /// Unlike a pinhole camera, a spherical camera projects points in every
    /// direction, including behind the camera, so this method does not check
    /// `point.z > 0`.
    fn project(&self, point: &Vector3<f64>) -> Result<Point2<f64>, ProjectionError> {
        // Convert to spherical coordinates
        let (longitude, latitude) = self.cartesian_to_spherical(point)?;

        // Map to image coordinates whose integer values are pixel centers
        // Longitude: [-pi, pi] -> [-0.5, width - 0.5)
        let u = ((longitude + PI) * self.width as f64 / (2.0 * PI))
            .rem_euclid(f64::from(self.width))
            - 0.5;

        // Latitude: [-pi/2, pi/2] -> [-0.5, height - 0.5]
        let v = (latitude + PI / 2.0) * self.height() as f64 / PI - 0.5;

        Ok(Point2::new(u, v))
    }

    fn project_with_jacobian(
        &self,
        point: &Vector3<f64>,
    ) -> Result<(Point2<f64>, Matrix2x3<f64>), ProjectionError> {
        let pixel = self.project(point)?;
        let scale = point
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let scaled = point / scale;
        let horizontal_squared = scaled.x.mul_add(scaled.x, scaled.z * scaled.z);
        if horizontal_squared == 0.0 {
            return Err(ProjectionError::OutsideModelDomain);
        }
        let horizontal = horizontal_squared.sqrt();
        let norm_squared = horizontal_squared + scaled.y * scaled.y;
        let longitude_scale = f64::from(self.width) / (2.0 * PI);
        let latitude_scale = f64::from(self.height()) / PI;
        let inverse_scale = scale.recip();
        let jacobian = Matrix2x3::new(
            longitude_scale * scaled.z / horizontal_squared * inverse_scale,
            0.0,
            -longitude_scale * scaled.x / horizontal_squared * inverse_scale,
            -latitude_scale * scaled.y * scaled.x / (horizontal * norm_squared) * inverse_scale,
            latitude_scale * horizontal / norm_squared * inverse_scale,
            -latitude_scale * scaled.y * scaled.z / (horizontal * norm_squared) * inverse_scale,
        );
        if jacobian.iter().all(|value| value.is_finite()) {
            Ok((pixel, jacobian))
        } else {
            Err(ProjectionError::NonFinite)
        }
    }

    /// Unprojects a 2D pixel coordinate to a 3D normalized ray direction.
    ///
    /// This operation converts image coordinates back to spherical coordinates
    /// and then to a 3D unit vector representing the ray direction.
    fn unproject(&self, pixel: &Point2<f64>) -> Result<Vector3<f64>, ProjectionError> {
        let pixel = self.canonicalize_pixel(pixel)?;

        // Convert pixel-center coordinates to spherical coordinates
        let longitude = (pixel.x + 0.5) * (2.0 * PI) / self.width as f64 - PI;
        let latitude = (pixel.y + 0.5) * PI / self.height() as f64 - PI / 2.0;

        // Convert to 3D unit vector
        self.spherical_to_cartesian(longitude, latitude)
    }

    /// Returns the principal point (center) of the spherical camera.
    ///
    /// For spherical cameras, this is the center of the image,
    /// `((width - 1) / 2, (height - 1) / 2)` in pixel-center coordinates.
    fn principal_point(&self) -> Point2<f64> {
        Point2::new(
            (f64::from(self.width) - 1.0) / 2.0,
            (f64::from(self.height()) - 1.0) / 2.0,
        )
    }

    /// Returns None as spherical cameras don't have a traditional intrinsic matrix.
    ///
    /// The spherical projection model uses spherical coordinates rather than
    /// the linear pinhole model, so there's no meaningful intrinsic matrix K.
    fn intrinsic_matrix(&self) -> Option<Matrix3<f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_jacobian_rejects_coordinate_singularities_at_poles() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        for pole in [Vector3::y(), -Vector3::y()] {
            assert!(camera.project(&pole).is_ok());
            assert_eq!(
                camera.project_with_jacobian(&pole),
                Err(ProjectionError::OutsideModelDomain)
            );
        }
    }

    #[test]
    fn image_center_is_positive_camera_z() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let ray = camera.unproject(&camera.principal_point()).unwrap();
        assert!((ray - Vector3::z()).norm() < 1e-12);
    }

    #[test]
    fn axes_follow_image_coordinates() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let right = camera.project(&Vector3::x()).unwrap();
        let down = camera.project(&Vector3::y()).unwrap();
        assert_eq!(right, Point2::new(767.5, 255.5));
        assert_eq!(down, Point2::new(511.5, 511.5));
    }

    #[test]
    fn poles_lie_on_the_image_edges() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        assert_eq!(camera.project(&-Vector3::y()).unwrap().y, -0.5);
        assert_eq!(camera.project(&Vector3::y()).unwrap().y, 511.5);
        assert_eq!(camera.principal_point(), Point2::new(511.5, 255.5));
    }

    #[test]
    fn projection_uses_one_canonical_coordinate_for_the_longitude_seam() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let behind = camera.project(&-Vector3::z()).unwrap();
        assert_eq!(behind, Point2::new(-0.5, 255.5));

        let left = camera.unproject(&Point2::new(-0.5, 255.5)).unwrap();
        let right = camera.unproject(&Point2::new(1023.5, 255.5)).unwrap();
        assert!((left - right).norm() < 1e-12);
    }

    #[test]
    fn unprojection_wraps_longitude_but_rejects_invalid_latitude() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let wrapped = camera.unproject(&Point2::new(-1.0, 256.0)).unwrap();
        let canonical = camera.unproject(&Point2::new(1023.0, 256.0)).unwrap();
        assert!((wrapped - canonical).norm() < 1e-12);
        assert!(camera.unproject(&Point2::new(0.0, -0.5)).is_ok());
        assert!(camera.unproject(&Point2::new(0.0, 511.5)).is_ok());
        assert_eq!(
            camera.unproject(&Point2::new(0.0, -0.75)),
            Err(ProjectionError::OutsideModelDomain)
        );
        assert_eq!(
            camera.unproject(&Point2::new(0.0, 511.75)),
            Err(ProjectionError::OutsideModelDomain)
        );
    }

    #[test]
    fn projection_handles_large_finite_vectors() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let point = Vector3::new(f64::MAX / 2.0, f64::MAX / 2.0, 1.0);
        let pixel = camera.project(&point).unwrap();
        assert!(pixel.coords.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn projection_is_invariant_to_extreme_finite_direction_scales() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let expected = camera.project(&Vector3::new(1.0, 1.0, 1.0)).unwrap();

        for direction in [
            Vector3::new(1e-300, 1e-300, 1e-300),
            Vector3::new(f64::MAX, f64::MAX, f64::MAX),
        ] {
            let actual = camera.project(&direction).unwrap();
            assert!((actual - expected).norm() < 1e-10);
        }
    }

    #[test]
    fn coordinate_conversions_reject_invalid_values() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        assert_eq!(
            camera.cartesian_to_spherical(&Vector3::zeros()),
            Err(ProjectionError::OutsideModelDomain)
        );
        assert_eq!(
            camera.spherical_to_cartesian(f64::NAN, 0.0),
            Err(ProjectionError::NonFinite)
        );
        assert_eq!(
            camera.spherical_to_cartesian(0.0, PI),
            Err(ProjectionError::OutsideModelDomain)
        );
    }

    #[test]
    fn serialized_form_has_one_source_of_truth() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let value = serde_json::to_value(&camera).unwrap();
        assert_eq!(value, serde_json::json!({ "width": 1024 }));
    }

    #[test]
    fn deserialization_validates_width() {
        assert!(serde_json::from_str::<SphericalCamera>(r#"{"width":0}"#).is_err());
        assert!(serde_json::from_str::<SphericalCamera>(r#"{"width":1023}"#).is_err());
    }
}
