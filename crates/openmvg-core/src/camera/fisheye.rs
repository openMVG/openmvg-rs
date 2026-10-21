// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Fisheye camera model
//!
//! OpenCV-compatible equidistant fisheye camera model. The image radius is the
//! polynomially distorted incident angle, so valid calibrations can represent
//! rays beyond 90 degrees from the optical axis.
//!
//! Unlike the Brown-Conrady model which uses radial distance r², the fisheye model
//! uses the angle θ = atan(r) from the optical axis, making it more suitable for
//! extreme wide-angle lenses where the relationship between image coordinates and
//! incident angle becomes highly non-linear.
//!
//! ## Mathematical model
//!
//! ```text
//! r = √(X² + Y²)
//! θ = atan2(r, Z)                   // Angle from optical axis
//! θ_dist = θ × (1 + k1×θ² + k2×θ⁴ + k3×θ⁶ + k4×θ⁸)  // Apply distortion
//! x_dist = θ_dist × X/r, y_dist = θ_dist × Y/r
//! ```

use crate::camera::camera_model::{
    CameraError, CameraModel, ProjectionError, try_normalize_bearing,
};
use crate::camera::distortion::FisheyeDistortion;
use crate::camera::pinhole::PinholeCamera;
use nalgebra::{Matrix2x3, Matrix3, Point2, Vector3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MAX_INCIDENT_ANGLE: f64 = std::f64::consts::PI - 1e-10;

/// Pinhole camera with equidistant fisheye lens distortion.
///
/// The intrinsic part is a [`PinholeCamera`] (single or double focal length). The
/// distortion polynomial, with 0 to 4 coefficients, acts on the angle θ from the
/// optical axis rather than on the radial distance, which suits wide-angle lenses.
#[derive(Debug, Clone)]
pub struct FisheyeCamera {
    /// The underlying pinhole camera for basic perspective projection.
    pinhole: PinholeCamera,
    /// Fisheye distortion configuration.
    fisheye: FisheyeDistortion,
    /// Cached end of the polynomial's principal monotone branch.
    max_incident_angle: f64,
}

impl FisheyeCamera {
    /// Creates a fisheye camera from a pinhole camera and distortion coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::InvalidDistortion`] if a coefficient is non-finite.
    ///
    /// The polynomial is used only up to its first turn, where the distorted angle stops
    /// increasing; projection rejects points past that angle with
    /// [`ProjectionError::OutsideModelDomain`].
    pub fn try_new(
        pinhole: PinholeCamera,
        fisheye: FisheyeDistortion,
    ) -> Result<Self, CameraError> {
        fisheye
            .validate()
            .map_err(|_| CameraError::InvalidDistortion {
                reason: "fisheye coefficients must be finite",
            })?;
        let max_incident_angle =
            fisheye
                .max_invertible_angle(MAX_INCIDENT_ANGLE)
                .map_err(|_| CameraError::InvalidDistortion {
                    reason: "fisheye polynomial has no finite invertible domain",
                })?;
        Ok(Self {
            pinhole,
            fisheye,
            max_incident_angle,
        })
    }

    /// Creates a fisheye camera with a single focal length `f`, in pixels.
    ///
    /// The principal point `(cx, cy)` is in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError`] when the pinhole part is invalid, a coefficient is
    /// non-finite, or the polynomial has no usable monotone branch.
    pub fn try_with_focal(
        f: f64,
        cx: f64,
        cy: f64,
        fisheye: FisheyeDistortion,
    ) -> Result<Self, CameraError> {
        let pinhole = PinholeCamera::try_with_focal(f, cx, cy)?;
        Self::try_new(pinhole, fisheye)
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

    // Accessor methods for fisheye distortion parameters

    /// Returns a reference to the fisheye distortion configuration.
    pub fn fisheye_distortion(&self) -> &FisheyeDistortion {
        &self.fisheye
    }

    /// Returns whether this camera has any fisheye distortion.
    pub fn has_distortion(&self) -> bool {
        self.fisheye.has_distortion()
    }
}

impl<'de> Deserialize<'de> for FisheyeCamera {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            pinhole: PinholeCamera,
            fisheye: FisheyeDistortion,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::try_new(repr.pinhole, repr.fisheye).map_err(serde::de::Error::custom)
    }
}

impl Serialize for FisheyeCamera {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Repr<'a> {
            pinhole: &'a PinholeCamera,
            fisheye: &'a FisheyeDistortion,
        }

        Repr {
            pinhole: &self.pinhole,
            fisheye: &self.fisheye,
        }
        .serialize(serializer)
    }
}

impl PartialEq for FisheyeCamera {
    fn eq(&self, other: &Self) -> bool {
        self.pinhole == other.pinhole && self.fisheye == other.fisheye
    }
}

impl CameraModel for FisheyeCamera {
    /// Projects a 3D direction with the equidistant fisheye model.
    fn project(&self, point: &Vector3<f64>) -> Result<Point2<f64>, ProjectionError> {
        let direction = try_normalize_bearing(point)?;
        let radial = direction.x.hypot(direction.y);
        if radial == 0.0 {
            return if direction.z > 0.0 {
                Ok(self.principal_point())
            } else {
                Err(ProjectionError::OutsideModelDomain)
            };
        }

        let theta = radial.atan2(direction.z);
        if theta > self.max_incident_angle + 1e-12 {
            return Err(ProjectionError::OutsideModelDomain);
        }
        let distorted_theta = self.fisheye.apply(theta);
        let scale = distorted_theta / radial;
        let pixel = Point2::new(
            self.focal_x()
                .mul_add(direction.x * scale, self.principal_x()),
            self.focal_y()
                .mul_add(direction.y * scale, self.principal_y()),
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
        let scale = point
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let scaled = point / scale;
        let radial = scaled.x.hypot(scaled.y);
        if scaled.z > 0.0 && radial <= f64::EPSILON.sqrt() * scaled.z {
            let inverse_depth = scale.recip() / scaled.z;
            let jacobian = Matrix2x3::new(
                self.focal_x() * inverse_depth,
                0.0,
                -self.focal_x() * scaled.x / (scaled.z * scaled.z) * scale.recip(),
                0.0,
                self.focal_y() * inverse_depth,
                -self.focal_y() * scaled.y / (scaled.z * scaled.z) * scale.recip(),
            );
            return if jacobian.iter().all(|value| value.is_finite()) {
                Ok((pixel, jacobian))
            } else {
                Err(ProjectionError::NonFinite)
            };
        }

        let norm_squared = scaled.norm_squared();
        let theta = radial.atan2(scaled.z);
        let distorted_theta = self.fisheye.apply(theta);
        let distorted_derivative = self.fisheye.derivative(theta);
        let radial_squared = radial * radial;
        let radial_cubed = radial_squared * radial;
        let theta_x = scaled.z * scaled.x / (radial * norm_squared);
        let theta_y = scaled.z * scaled.y / (radial * norm_squared);
        let theta_z = -radial / norm_squared;
        let q = distorted_theta / radial;
        let q_x =
            distorted_derivative * theta_x / radial - distorted_theta * scaled.x / radial_cubed;
        let q_y =
            distorted_derivative * theta_y / radial - distorted_theta * scaled.y / radial_cubed;
        let q_z = distorted_derivative * theta_z / radial;
        let inverse_scale = scale.recip();
        let jacobian = Matrix2x3::new(
            self.focal_x() * (q + scaled.x * q_x) * inverse_scale,
            self.focal_x() * scaled.x * q_y * inverse_scale,
            self.focal_x() * scaled.x * q_z * inverse_scale,
            self.focal_y() * scaled.y * q_x * inverse_scale,
            self.focal_y() * (q + scaled.y * q_y) * inverse_scale,
            self.focal_y() * scaled.y * q_z * inverse_scale,
        );
        if jacobian.iter().all(|value| value.is_finite()) {
            Ok((pixel, jacobian))
        } else {
            Err(ProjectionError::NonFinite)
        }
    }

    /// Unprojects a pixel to a unit camera-frame bearing.
    fn unproject(&self, pixel: &Point2<f64>) -> Result<Vector3<f64>, ProjectionError> {
        if !pixel.coords.iter().all(|v| v.is_finite()) {
            return Err(ProjectionError::NonFinite);
        }
        let x_distorted = (pixel.x - self.principal_x()) / self.focal_x();
        let y_distorted = (pixel.y - self.principal_y()) / self.focal_y();
        if !x_distorted.is_finite() || !y_distorted.is_finite() {
            return Err(ProjectionError::NonFinite);
        }
        let distorted_theta = x_distorted.hypot(y_distorted);
        if distorted_theta <= f64::EPSILON {
            return Ok(Vector3::z());
        }
        let theta = self
            .fisheye
            .invert_on_validated_interval(distorted_theta, self.max_incident_angle)
            .map_err(ProjectionError::from)?;
        let azimuth_scale = theta.sin() / distorted_theta;
        let bearing = Vector3::new(
            x_distorted * azimuth_scale,
            y_distorted * azimuth_scale,
            theta.cos(),
        );
        if bearing.iter().all(|value| value.is_finite()) {
            Ok(bearing)
        } else {
            Err(ProjectionError::NonFinite)
        }
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

    #[test]
    fn projection_jacobian_has_a_finite_optical_axis_limit() {
        let camera = FisheyeCamera::try_new(
            PinholeCamera::try_new(400.0, 410.0, 320.0, 240.0).unwrap(),
            FisheyeDistortion::K1K2(-0.01, 0.001),
        )
        .unwrap();
        let (_, jacobian) = camera
            .project_with_jacobian(&Vector3::new(0.0, 0.0, 2.0))
            .unwrap();
        assert_eq!(jacobian, Matrix2x3::new(200.0, 0.0, 0.0, 0.0, 205.0, 0.0));
    }

    #[test]
    fn optical_axis_jacobian_rejects_non_finite_derivatives() {
        let camera =
            FisheyeCamera::try_with_focal(300.0, 320.0, 240.0, FisheyeDistortion::None).unwrap();
        let point = Vector3::new(0.0, 0.0, 1e-310);
        assert_eq!(camera.project(&point).unwrap(), camera.principal_point());
        assert_eq!(
            camera.project_with_jacobian(&point),
            Err(ProjectionError::NonFinite),
        );

        let point = Vector3::new(0.0, 0.0, 2.0);
        let (pixel, jacobian) = camera.project_with_jacobian(&point).unwrap();
        assert_eq!(pixel, camera.project(&point).unwrap());
        assert_eq!(jacobian, Matrix2x3::new(150.0, 0.0, 0.0, 0.0, 150.0, 0.0));
    }

    #[test]
    fn ideal_equidistant_maps_ninety_degrees_to_pi_over_two() {
        let camera = FisheyeCamera::try_with_focal(100.0, 320.0, 240.0, FisheyeDistortion::None)
            .expect("valid camera");

        let pixel = camera
            .project(&Vector3::x())
            .expect("90-degree ray is representable");

        assert!((pixel.x - (320.0 + 50.0 * std::f64::consts::PI)).abs() < 1e-12);
        assert!((pixel.y - 240.0).abs() < 1e-12);
    }

    #[test]
    fn rays_beyond_ninety_degrees_round_trip() {
        let camera = FisheyeCamera::try_with_focal(400.0, 320.0, 240.0, FisheyeDistortion::None)
            .expect("valid camera");
        let angle = 100.0_f64.to_radians();
        let ray = Vector3::new(angle.sin(), 0.0, angle.cos());

        let pixel = camera.project(&ray).expect("wide ray should project");
        let recovered = camera.unproject(&pixel).expect("pixel should unproject");

        assert!((ray.dot(&recovered) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn projection_is_invariant_to_extreme_finite_direction_scales() {
        let camera =
            FisheyeCamera::try_with_focal(400.0, 320.0, 240.0, FisheyeDistortion::K1(0.02))
                .expect("valid camera");
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
    fn rejects_non_finite_and_folded_calibrations() {
        let pinhole = PinholeCamera::try_with_focal(400.0, 320.0, 240.0).expect("valid pinhole");
        for coefficient in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                FisheyeCamera::try_new(pinhole.clone(), FisheyeDistortion::K1(coefficient))
                    .is_err()
            );
        }

        let folded = FisheyeCamera::try_new(pinhole, FisheyeDistortion::K1(-1.0))
            .expect("finite coefficients are structurally valid");
        assert_eq!(
            folded.project(&Vector3::new(1.0, 0.0, 1.0)),
            Err(ProjectionError::OutsideModelDomain)
        );
    }

    #[test]
    fn optical_axis_has_a_unique_forward_projection() {
        let camera = FisheyeCamera::try_with_focal(400.0, 320.0, 240.0, FisheyeDistortion::None)
            .expect("valid camera");

        assert_eq!(
            camera.project(&Vector3::z()).expect("forward axis"),
            Point2::new(320.0, 240.0)
        );
        assert_eq!(
            camera.project(&-Vector3::z()),
            Err(ProjectionError::OutsideModelDomain)
        );
    }

    #[test]
    fn serialization_omits_derived_domain_cache() {
        let camera =
            FisheyeCamera::try_with_focal(400.0, 320.0, 240.0, FisheyeDistortion::K1(-0.05))
                .expect("valid camera");
        let value = serde_json::to_value(&camera).expect("serialize camera");
        assert!(value.get("max_incident_angle").is_none());

        let recovered: FisheyeCamera = serde_json::from_value(value).expect("deserialize camera");
        assert_eq!(recovered, camera);
    }

    #[test]
    fn serde_rejects_invalid_or_malformed_camera() {
        for json in [
            r#"{"pinhole":{"fx":0.0,"fy":0.0,"cx":320.0,"cy":240.0},"fisheye":"none"}"#,
            r#"{"pinhole":{"fx":400.0,"fy":-1.0,"cx":320.0,"cy":240.0},"fisheye":"none"}"#,
            r#"{"pinhole":{"fx":400.0,"fy":400.0,"cx":320.0,"cy":240.0}}"#,
            r#"{"pinhole":"invalid","fisheye":"none"}"#,
        ] {
            assert!(
                serde_json::from_str::<FisheyeCamera>(json).is_err(),
                "accepted invalid camera JSON: {json}"
            );
        }
    }
}
