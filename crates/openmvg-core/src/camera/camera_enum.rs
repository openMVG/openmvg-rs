// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Polymorphic camera type
//!
//! This module provides an enum that holds any supported camera model behind a
//! single type, for code that chooses the model at runtime or stores several
//! models in one collection.

use crate::camera::brown_conrady::BrownConradyCamera;
use crate::camera::camera_model::{
    CameraError, CameraModel, ProjectionError, try_normalize_bearing,
};
use crate::camera::distortion::{RadialDistortion, TangentialDistortion};
use crate::camera::fisheye::FisheyeCamera;
use crate::camera::pinhole::PinholeCamera;
use crate::camera::spherical::SphericalCamera;
use nalgebra::{Matrix2x3, Matrix3, Point2, Unit, UnitVector3, Vector3};

/// Camera that can hold any supported camera model.
///
/// It implements the [`CameraModel`] trait by delegating to the wrapped model.
///
/// The enum is exhaustive: a new camera model would come with a new major version, so a
/// `match` over it can list every model without a fallback arm.
///
/// # Camera models
///
/// - [`Camera::Pinhole`]: perspective projection (single or double focal).
/// - [`Camera::BrownConrady`]: perspective projection with lens distortion.
/// - [`Camera::Fisheye`]: equidistant fisheye projection with angle-based distortion.
/// - [`Camera::Spherical`]: 360-degree equirectangular projection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
// Closed set for v0.1: adding a camera model is a breaking change.
#[serde(rename_all = "snake_case")]
pub enum Camera {
    /// Perspective projection without distortion; see [`PinholeCamera`].
    ///
    /// The focal length is single (square pixels) or double (separate fx/fy).
    Pinhole(PinholeCamera),

    /// Perspective projection with Brown-Conrady radial and tangential distortion; see
    /// [`BrownConradyCamera`].
    BrownConrady(BrownConradyCamera),

    /// Equidistant fisheye projection whose distortion acts on the angle θ from the optical
    /// axis, for fields of view near or beyond 180°; see [`FisheyeCamera`].
    Fisheye(FisheyeCamera),

    /// Spherical projection to an equirectangular image, for 360-degree cameras; see
    /// [`SphericalCamera`].
    Spherical(SphericalCamera),
}

impl CameraModel for Camera {
    /// Projects a 3D point using the appropriate camera model.
    fn project(&self, point: &Vector3<f64>) -> Result<Point2<f64>, ProjectionError> {
        match self {
            Camera::Pinhole(cam) => cam.project(point),
            Camera::BrownConrady(cam) => cam.project(point),
            Camera::Fisheye(cam) => cam.project(point),
            Camera::Spherical(cam) => cam.project(point),
        }
    }

    fn project_with_jacobian(
        &self,
        point: &Vector3<f64>,
    ) -> Result<(Point2<f64>, Matrix2x3<f64>), ProjectionError> {
        match self {
            Camera::Pinhole(camera) => camera.project_with_jacobian(point),
            Camera::BrownConrady(camera) => camera.project_with_jacobian(point),
            Camera::Fisheye(camera) => camera.project_with_jacobian(point),
            Camera::Spherical(camera) => camera.project_with_jacobian(point),
        }
    }

    /// Unprojects a 2D pixel using the appropriate camera model.
    fn unproject(&self, pixel: &Point2<f64>) -> Result<Vector3<f64>, ProjectionError> {
        match self {
            Camera::Pinhole(cam) => cam.unproject(pixel),
            Camera::BrownConrady(cam) => cam.unproject(pixel),
            Camera::Fisheye(cam) => cam.unproject(pixel),
            Camera::Spherical(cam) => cam.unproject(pixel),
        }
    }

    /// Returns the principal point using the appropriate camera model.
    fn principal_point(&self) -> Point2<f64> {
        match self {
            Camera::Pinhole(cam) => cam.principal_point(),
            Camera::BrownConrady(cam) => cam.principal_point(),
            Camera::Fisheye(cam) => cam.principal_point(),
            Camera::Spherical(cam) => cam.principal_point(),
        }
    }

    /// Returns the intrinsic matrix using the appropriate camera model.
    fn intrinsic_matrix(&self) -> Option<Matrix3<f64>> {
        match self {
            Camera::Pinhole(cam) => cam.intrinsic_matrix(),
            Camera::BrownConrady(cam) => cam.intrinsic_matrix(),
            Camera::Fisheye(cam) => cam.intrinsic_matrix(),
            Camera::Spherical(cam) => cam.intrinsic_matrix(),
        }
    }
}

impl PartialEq for Camera {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Camera::Pinhole(a), Camera::Pinhole(b)) => a == b,
            (Camera::BrownConrady(a), Camera::BrownConrady(b)) => a == b,
            (Camera::Fisheye(a), Camera::Fisheye(b)) => a == b,
            (Camera::Spherical(a), Camera::Spherical(b)) => a == b,
            _ => false,
        }
    }
}

impl Camera {
    /// Converts a pixel into the unit bearing in this camera's frame.
    ///
    /// # Errors
    ///
    /// Returns the model-specific [`ProjectionError`] when the pixel is non-finite, outside the
    /// camera domain, or cannot be inverted through its distortion model.
    pub fn bearing(&self, pixel: &Point2<f64>) -> Result<UnitVector3<f64>, ProjectionError> {
        self.unproject(pixel)
            .and_then(|direction| try_normalize_bearing(&direction))
            .map(Unit::new_unchecked)
    }

    /// Construct a validated perspective camera with independent focal lengths.
    pub fn try_pinhole(
        focal_x: f64,
        focal_y: f64,
        principal_x: f64,
        principal_y: f64,
    ) -> Result<Self, CameraError> {
        Ok(Self::Pinhole(PinholeCamera::try_new(
            focal_x,
            focal_y,
            principal_x,
            principal_y,
        )?))
    }

    /// Construct a validated Brown-Conrady perspective camera.
    ///
    /// The pinhole part is the same as [`Camera::try_pinhole`]: focal lengths and principal
    /// point in pixels. The distortion
    /// coefficients act in normalized image coordinates; the full five-coefficient model
    /// is `RadialDistortion::K1K2K3(k1, k2, k3)` with `TangentialDistortion::P1P2(p1, p2)`.
    ///
    /// # Errors
    ///
    /// Returns [`CameraError::InvalidFocalLength`] or [`CameraError::InvalidPrincipalPoint`]
    /// for an invalid pinhole part (checked first), then [`CameraError::InvalidDistortion`]
    /// when a coefficient is non-finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::{Camera, RadialDistortion, TangentialDistortion};
    ///
    /// let camera = Camera::try_brown_conrady(
    ///     800.0,
    ///     810.0,
    ///     320.0,
    ///     240.0,
    ///     RadialDistortion::K1K2K3(-0.1, 0.01, 0.0),
    ///     TangentialDistortion::P1P2(0.001, -0.001),
    /// )?;
    /// assert_eq!(camera.focal_lengths(), Some((800.0, 810.0)));
    /// # Ok::<(), openmvg_core::CameraError>(())
    /// ```
    pub fn try_brown_conrady(
        focal_x: f64,
        focal_y: f64,
        principal_x: f64,
        principal_y: f64,
        radial: RadialDistortion,
        tangential: TangentialDistortion,
    ) -> Result<Self, CameraError> {
        let pinhole = PinholeCamera::try_new(focal_x, focal_y, principal_x, principal_y)?;
        BrownConradyCamera::try_new(pinhole, radial, tangential).map(Self::BrownConrady)
    }

    /// Return `(focal_x, focal_y)` for camera models with a focal-length parameter.
    /// Spherical equirectangular cameras have no focal length.
    #[must_use]
    pub fn focal_lengths(&self) -> Option<(f64, f64)> {
        match self {
            Self::Pinhole(camera) => Some((camera.focal_x(), camera.focal_y())),
            Self::BrownConrady(camera) => Some((camera.focal_x(), camera.focal_y())),
            Self::Fisheye(camera) => Some((camera.focal_x(), camera.focal_y())),
            Self::Spherical(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::FisheyeDistortion;

    fn camera_models() -> [Camera; 4] {
        [
            Camera::try_pinhole(800.0, 810.0, 320.0, 240.0).unwrap(),
            Camera::try_brown_conrady(
                800.0,
                810.0,
                320.0,
                240.0,
                RadialDistortion::K1K2K3(-0.01, 0.001, 0.0),
                TangentialDistortion::P1P2(0.0001, -0.0002),
            )
            .unwrap(),
            Camera::Fisheye(
                FisheyeCamera::try_new(
                    PinholeCamera::try_new(400.0, 410.0, 320.0, 240.0).unwrap(),
                    FisheyeDistortion::K1K2(-0.01, 0.001),
                )
                .unwrap(),
            ),
            Camera::Spherical(SphericalCamera::try_new(1024).unwrap()),
        ]
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:e}, expected {expected:e}, tolerance {tolerance:e}"
        );
    }

    #[test]
    fn convenience_constructors_validate_calibration() {
        let Camera::Pinhole(single) = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0).unwrap()
        else {
            unreachable!();
        };
        assert_eq!(single.focal_x(), single.focal_y());

        let camera = Camera::try_brown_conrady(
            800.0,
            810.0,
            320.0,
            240.0,
            RadialDistortion::K1K2K3(-0.1, 0.01, 0.0),
            TangentialDistortion::P1P2(0.001, -0.001),
        )
        .unwrap();
        assert_eq!(camera.focal_lengths(), Some((800.0, 810.0)));
        assert!(Camera::try_pinhole(0.0, 800.0, 320.0, 240.0).is_err());

        let Camera::BrownConrady(single) = Camera::try_brown_conrady(
            800.0,
            800.0,
            320.0,
            240.0,
            RadialDistortion::K1(-0.1),
            TangentialDistortion::None,
        )
        .unwrap() else {
            unreachable!();
        };
        assert_eq!(single.focal_x(), single.focal_y());
        // The pinhole part is validated before the distortion coefficients.
        assert!(matches!(
            Camera::try_brown_conrady(
                0.0,
                800.0,
                320.0,
                240.0,
                RadialDistortion::K1(f64::NAN),
                TangentialDistortion::None,
            ),
            Err(CameraError::InvalidFocalLength { .. })
        ));
        assert!(matches!(
            Camera::try_brown_conrady(
                800.0,
                800.0,
                320.0,
                240.0,
                RadialDistortion::K1(f64::NAN),
                TangentialDistortion::None,
            ),
            Err(CameraError::InvalidDistortion { .. })
        ));
    }

    #[test]
    fn polymorphic_deserialization_preserves_camera_validation() {
        for json in [
            r#"{"pinhole":{"fx":0.0,"fy":0.0,"cx":320.0,"cy":240.0}}"#,
            r#"{"brown_conrady":{"pinhole":{"fx":-1.0,"fy":-1.0,"cx":320.0,"cy":240.0},"radial":"none","tangential":"none"}}"#,
            r#"{"fisheye":{"pinhole":{"fx":400.0,"fy":400.0,"cx":320.0,"cy":240.0}}}"#,
        ] {
            assert!(
                serde_json::from_str::<Camera>(json).is_err(),
                "accepted invalid camera JSON: {json}"
            );
        }
    }

    #[test]
    fn bearing_is_unit_length_for_every_camera_model() {
        let point = Vector3::new(0.2, -0.1, 1.0);
        let expected = point.normalize();
        for camera in camera_models() {
            let pixel = camera.project(&point).unwrap();
            let bearing = camera.bearing(&pixel).unwrap();
            assert!((bearing.norm() - 1.0).abs() < 1e-14);
            assert!(bearing.dot(&expected) > 1.0 - 1e-12);
        }
    }

    #[test]
    fn analytic_projection_jacobians_match_finite_differences() {
        let point = Vector3::new(0.23, -0.17, 1.31);
        let step = 1e-6;
        for camera in camera_models() {
            let (pixel, analytic) = camera.project_with_jacobian(&point).unwrap();
            assert_eq!(pixel, camera.project(&point).unwrap());

            for column in 0..3 {
                let mut plus = point;
                let mut minus = point;
                plus[column] += step;
                minus[column] -= step;
                let plus = camera.project(&plus).unwrap();
                let minus = camera.project(&minus).unwrap();
                let numerical = (plus - minus) / (2.0 * step);
                for row in 0..2 {
                    assert_close(analytic[(row, column)], numerical[row], 2e-5);
                }
            }
        }
    }

    #[test]
    fn core_projection_matches_existing_ba_kernel_equations() {
        let point = Vector3::new(0.71, -0.43, 2.4);
        let focal = 730.0;
        let k1 = 0.08;
        let k2 = -0.025;
        let camera = Camera::try_brown_conrady(
            focal,
            focal,
            0.0,
            0.0,
            RadialDistortion::K1K2K3(k1, k2, 0.0),
            TangentialDistortion::P1P2(0.0, 0.0),
        )
        .unwrap();
        let (pixel, jacobian) = camera.project_with_jacobian(&point).unwrap();

        // This is the centered radial projection used by the native BA kernel.
        let inverse_depth = point.z.recip();
        let x = point.x * inverse_depth;
        let y = point.y * inverse_depth;
        let radius_squared = x.mul_add(x, y * y);
        let distortion = 1.0 + k1 * radius_squared + k2 * radius_squared * radius_squared;
        let radial_derivative = k1 + 2.0 * k2 * radius_squared;
        let j11 = focal * (distortion + 2.0 * x * x * radial_derivative);
        let j12 = focal * 2.0 * x * y * radial_derivative;
        let j22 = focal * (distortion + 2.0 * y * y * radial_derivative);
        let expected_pixel = [focal * distortion * x, focal * distortion * y];
        let expected_jacobian = [
            [
                j11 * inverse_depth,
                j12 * inverse_depth,
                -(j11 * x + j12 * y) * inverse_depth,
            ],
            [
                j12 * inverse_depth,
                j22 * inverse_depth,
                -(j12 * x + j22 * y) * inverse_depth,
            ],
        ];

        for row in 0..2 {
            assert_close(pixel[row], expected_pixel[row], 1e-12);
            for column in 0..3 {
                assert_close(
                    jacobian[(row, column)],
                    expected_jacobian[row][column],
                    1e-12,
                );
            }
        }
    }
}
