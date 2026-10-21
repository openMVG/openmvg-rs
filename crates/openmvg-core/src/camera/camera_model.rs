// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Camera model contract
//!
//! The [`CameraModel`] trait and the projection math that every camera model implements. The
//! camera frame, pixel coordinates and pose convention are defined once, in the
//! [crate root conventions](crate#conventions).
//!
//! Each model has its own depth domain: pinhole and Brown-Conrady cameras require positive `Z`,
//! fisheye visibility is limited by the model's incident-angle domain, and spherical cameras
//! accept every finite nonzero direction, including negative `Z`.
//!
//! ## Projection models
//!
//! ### Pinhole projection
//!
//! The standard pinhole camera model follows these steps:
//!
//! 1. 3D point: P = [X, Y, Z]ᵀ in camera coordinates
//! 2. Perspective division: p_norm = [X/Z, Y/Z, 1]ᵀ (normalized image plane)
//! 3. Intrinsic transform: p_pixel = K × p_norm where K is the intrinsic matrix
//!
//! ```text
//! K = [fx  0  cx]    fx, fy: focal lengths in pixels
//!     [ 0 fy  cy]    cx, cy: principal point coordinates  
//!     [ 0  0   1]
//! ```
//!
//! ### Spherical projection
//!
//! Spherical cameras use spherical coordinates for omnidirectional imaging:
//!
//! 1. Cartesian to spherical:
//!    - θ (longitude) = atan2(X, Z) ∈ [-π, π]
//!    - φ (latitude, positive downward) = atan2(Y, √(X² + Z²)) ∈ [-π/2, π/2]
//! 2. Equirectangular mapping:
//!    - u = (θ + π) × width / (2π) - 0.5, wrapped into [-0.5, width - 0.5)
//!    - v = (φ + π/2) × height / π - 0.5
//!
//! ## Numerical considerations
//!
//! - Precision: all calculations use `f64`
//! - Stability: points very close to the camera center may be ill-conditioned
//! - Range checks: inputs are checked for finite values and the model's valid domain

use nalgebra::{Matrix2x3, Matrix3, Point2, Vector3};

use super::DistortionError;

/// Reason a 3D point cannot be projected to a pixel, or a pixel unprojected to a bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProjectionError {
    /// The 3D point is behind a forward-looking camera's visible half-space.
    ///
    /// Perspective models return this for non-positive depth. Omnidirectional
    /// models such as spherical cameras do not use this half-space restriction.
    #[error("3D point is behind the camera (Z ≤ 0)")]
    BehindCamera,

    /// An input or intermediate camera value is non-finite (NaN or infinity).
    ///
    /// Returned when an input coordinate is NaN or infinite, or when an
    /// intermediate computation produces such a value.
    #[error("camera operation produced non-finite data")]
    NonFinite,

    /// Lens distortion could not be inverted at the requested pixel.
    #[error("lens distortion could not be inverted")]
    DistortionInversion,

    /// The point or pixel lies outside the one-to-one domain of the model.
    #[error("coordinate lies outside the camera model domain")]
    OutsideModelDomain,
}

impl From<DistortionError> for ProjectionError {
    fn from(error: DistortionError) -> Self {
        match error {
            DistortionError::NonFinite => Self::NonFinite,
            DistortionError::OutsideInvertibleDomain => Self::OutsideModelDomain,
            DistortionError::SingularJacobian | DistortionError::DidNotConverge { .. } => {
                Self::DistortionInversion
            }
        }
    }
}

#[inline]
pub(crate) fn try_normalize_bearing(
    vector: &Vector3<f64>,
) -> Result<Vector3<f64>, ProjectionError> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(ProjectionError::NonFinite);
    }
    let scale = vector
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return Err(ProjectionError::OutsideModelDomain);
    }
    let scaled = vector / scale;
    let norm = scaled.norm();
    if !norm.is_finite() || norm == 0.0 {
        return Err(ProjectionError::OutsideModelDomain);
    }
    Ok(scaled / norm)
}

/// Reason a set of camera intrinsic parameters is rejected at construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CameraError {
    /// The focal length parameter is invalid.
    ///
    /// Returned when a focal length is zero, negative, NaN, or infinite.
    #[error("invalid focal length {value}: {reason}")]
    InvalidFocalLength {
        /// The invalid focal length value that was provided.
        value: f64,
        /// Human-readable explanation of why it's invalid.
        reason: &'static str,
    },

    /// The principal point coordinates are invalid.
    ///
    /// Returned when a principal point coordinate is NaN or infinite. Negative
    /// coordinates are accepted.
    #[error("invalid principal point ({principal_x}, {principal_y}): {reason}")]
    InvalidPrincipalPoint {
        /// The invalid horizontal coordinate.
        principal_x: f64,
        /// The invalid vertical coordinate.
        principal_y: f64,
        /// Human-readable explanation of why they're invalid.
        reason: &'static str,
    },

    /// The image width parameter is invalid.
    ///
    /// Returned when the width is zero, or odd where an even width is required
    /// (spherical camera).
    #[error("invalid width {width}: {reason}")]
    InvalidWidth {
        /// The invalid width value.
        width: u32,
        /// Human-readable explanation of why it's invalid.
        reason: &'static str,
    },

    /// The distortion parameters are invalid.
    ///
    /// Returned when a distortion coefficient is NaN or infinite.
    #[error("invalid distortion parameters: {reason}")]
    InvalidDistortion {
        /// Human-readable explanation of the problem.
        reason: &'static str,
    },
}

/// Common interface of the camera models: projection (3D point to pixel) and
/// unprojection (pixel to 3D ray).
///
/// Camera coordinates are right-handed with Z+ pointing into the scene. Image
/// coordinates have their origin at the top-left pixel center, u+ to the right
/// and v+ down, with pixel centers at integer coordinates.
///
/// Implementations must be `Send + Sync`, follow these conventions, and return a
/// [`ProjectionError`] for inputs outside their domain instead of panicking.
pub trait CameraModel: Send + Sync {
    /// Projects a 3D point in camera coordinates to 2D pixel coordinates.
    ///
    /// # Coordinate systems
    ///
    /// - Input: 3D point in camera coordinates [X, Y, Z]ᵀ
    ///   - Right-handed coordinate system
    ///   - Z+ is the conventional forward axis
    ///   - Visibility is model-specific: perspective models require positive Z,
    ///     fisheye models enforce their incident-angle domain, and spherical models
    ///     accept any finite nonzero direction
    ///
    /// - Output: 2D pixel coordinates (u, v)
    ///   - Image coordinates with `(0, 0)` at the center of the top-left pixel
    ///   - u+ increases to the right, v+ increases downward
    ///   - Fractional coordinates indicate sub-pixel locations
    ///
    /// # Errors
    ///
    /// * [`ProjectionError::BehindCamera`] - Point lies outside a forward-looking
    ///   model's visible half-space
    /// * [`ProjectionError::NonFinite`] - Point contains NaN or infinite values
    /// * [`ProjectionError::OutsideModelDomain`] - Point is outside the selected
    ///   camera model's one-to-one projection domain
    fn project(&self, point: &Vector3<f64>) -> Result<Point2<f64>, ProjectionError>;

    /// Projects a camera-frame point and returns the analytic pixel derivative `d(u,v)/d(X,Y,Z)`.
    ///
    /// The method has the same visibility contract as [`CameraModel::project`]. It additionally
    /// rejects points where the model is projectable but its pixel coordinates are not
    /// differentiable, such as a spherical camera's poles.
    fn project_with_jacobian(
        &self,
        point: &Vector3<f64>,
    ) -> Result<(Point2<f64>, Matrix2x3<f64>), ProjectionError>;

    /// Unprojects a 2D pixel coordinate to a 3D ray direction.
    ///
    /// Depth is lost during projection, so this returns a direction rather than
    /// a 3D point.
    ///
    /// # Coordinate systems
    ///
    /// - Input: 2D pixel coordinates (u, v)
    ///   - Image coordinates with `(0, 0)` at the center of the top-left pixel
    ///   - u+ increases to the right, v+ increases downward
    ///   - Fractional coordinates represent sub-pixel locations
    ///
    /// - Output: 3D ray direction [X, Y, Z]ᵀ
    ///   - Direction vector in camera coordinate system
    ///   - Points from camera center through the specified pixel
    ///   - Not necessarily of unit length; [`Camera::bearing`](super::Camera::bearing)
    ///     returns the unit bearing
    ///
    /// # Errors
    ///
    /// Returns a [`ProjectionError`] when the pixel is non-finite, lies outside
    /// the model's domain, or cannot be inverted through its distortion model.
    fn unproject(&self, pixel: &Point2<f64>) -> Result<Vector3<f64>, ProjectionError>;

    /// Squared chord distance between a predicted camera-space point and an observed pixel ray.
    ///
    /// The value lies in `[0, 4]` for the corresponding bearing angle in `[0, π]`. To test many
    /// observations against an angular threshold `θ`, compare it with `4 sin²(θ / 2)`: that
    /// avoids an inverse trigonometric function per observation. Both the predicted point and
    /// the observed pixel must lie in this camera model's projection domain.
    #[inline]
    fn bearing_chord_squared(
        &self,
        camera_point: &Vector3<f64>,
        observed_pixel: &Point2<f64>,
    ) -> Result<f64, ProjectionError> {
        // Projection is evaluated even though its pixel result is not needed: it
        // enforces model-specific visibility, cheirality, and one-to-one-domain constraints.
        self.project(camera_point)?;
        let observed = self.unproject(observed_pixel)?;

        let predicted = try_normalize_bearing(camera_point)?;
        let observed = try_normalize_bearing(&observed)?;
        let chord_squared = (predicted - observed).norm_squared();
        if !chord_squared.is_finite() {
            return Err(ProjectionError::NonFinite);
        }
        Ok(chord_squared.clamp(0.0, 4.0))
    }

    /// Angular bearing residual in radians between a predicted camera-space point and pixel.
    ///
    /// Use [`CameraModel::bearing_chord_squared`] for repeated threshold comparisons and this
    /// method when the actual angle is needed for reports, ranking, or robust statistics.
    #[inline]
    fn bearing_error_rad(
        &self,
        camera_point: &Vector3<f64>,
        observed_pixel: &Point2<f64>,
    ) -> Result<f64, ProjectionError> {
        let chord_squared = self.bearing_chord_squared(camera_point, observed_pixel)?;
        Ok(2.0 * (0.5 * chord_squared.sqrt()).clamp(0.0, 1.0).asin())
    }

    /// Returns the principal point (optical center) of the camera.
    ///
    /// The principal point is where the optical axis intersects the image plane,
    /// typically near the center of the image. The coordinates (cx, cy) are in pixels.
    ///
    /// - For pinhole cameras: Returns the (cx, cy) from the intrinsic matrix K
    /// - For spherical cameras: Returns the image center ((width - 1)/2, (height - 1)/2)
    /// - For fisheye cameras: Returns the (cx, cy) from the underlying pinhole
    #[must_use]
    fn principal_point(&self) -> Point2<f64>;

    /// Returns the 3×3 intrinsic camera matrix K, if applicable.
    ///
    /// The intrinsic matrix K encodes the camera's internal parameters:
    ///
    /// ```text
    /// K = [fx  0  cx]
    ///     [ 0 fy  cy]
    ///     [ 0  0   1]
    /// ```
    ///
    /// Where:
    /// - `fx, fy`: Focal lengths in pixels
    /// - `cx, cy`: Principal point coordinates in pixels
    ///
    /// - Pinhole cameras always return `Some(K)`
    /// - Distortion cameras (Brown-Conrady, Fisheye) return the underlying pinhole's K
    /// - Spherical cameras return `None` as they don't use an intrinsic matrix
    #[must_use]
    fn intrinsic_matrix(&self) -> Option<Matrix3<f64>>;
}

#[cfg(test)]
mod bearing_residual_tests {
    use super::*;
    use crate::camera::{FisheyeCamera, FisheyeDistortion, PinholeCamera, SphericalCamera};

    #[test]
    fn distortion_failures_keep_their_projection_semantics() {
        assert_eq!(
            ProjectionError::from(DistortionError::NonFinite),
            ProjectionError::NonFinite
        );
        assert_eq!(
            ProjectionError::from(DistortionError::OutsideInvertibleDomain),
            ProjectionError::OutsideModelDomain
        );
        assert_eq!(
            ProjectionError::from(DistortionError::SingularJacobian),
            ProjectionError::DistortionInversion
        );
        assert_eq!(
            ProjectionError::from(DistortionError::DidNotConverge { iterations: 1 }),
            ProjectionError::DistortionInversion
        );
    }

    #[test]
    fn bearing_error_matches_its_fast_chord_representation() {
        let camera = PinholeCamera::try_with_focal(800.0, 320.0, 240.0).unwrap();
        let point = Vector3::new(0.4, -0.2, 3.0);
        let exact = camera.project(&point).unwrap();
        assert!(camera.bearing_chord_squared(&point, &exact).unwrap() < 1e-28);
        assert!(camera.bearing_error_rad(&point, &exact).unwrap() < 1e-14);

        let shifted = Point2::new(exact.x + 2.0, exact.y - 1.0);
        let chord_squared = camera.bearing_chord_squared(&point, &shifted).unwrap();
        let angle = camera.bearing_error_rad(&point, &shifted).unwrap();
        let reconstructed = 4.0 * (0.5 * angle).sin().powi(2);
        assert!((chord_squared - reconstructed).abs() < 1e-14);

        let large_point = Vector3::new(f64::MAX, 0.5 * f64::MAX, f64::MAX);
        let large_pixel = camera.project(&large_point).unwrap();
        assert!(
            camera
                .bearing_error_rad(&large_point, &large_pixel)
                .unwrap()
                < 1e-14
        );
    }

    #[test]
    fn bearing_error_preserves_projection_domain_checks() {
        let camera = PinholeCamera::try_with_focal(800.0, 320.0, 240.0).unwrap();
        assert_eq!(
            camera
                .bearing_error_rad(&Vector3::new(0.0, 0.0, -1.0), &Point2::new(320.0, 240.0))
                .unwrap_err(),
            ProjectionError::BehindCamera
        );
        assert_eq!(
            camera
                .bearing_error_rad(&Vector3::zeros(), &Point2::new(320.0, 240.0))
                .unwrap_err(),
            ProjectionError::BehindCamera
        );
        assert_eq!(
            camera
                .bearing_error_rad(&Vector3::new(0.0, 0.0, 1.0), &Point2::new(f64::NAN, 240.0),)
                .unwrap_err(),
            ProjectionError::NonFinite
        );
    }

    #[test]
    fn spherical_bearing_error_is_seam_and_pole_invariant() {
        let camera = SphericalCamera::try_new(1024).unwrap();
        let seam_point = camera.unproject(&Point2::new(1023.75, 256.0)).unwrap();
        assert!(
            camera
                .bearing_error_rad(&seam_point, &Point2::new(-0.25, 256.0))
                .unwrap()
                < 1e-14
        );

        let pole_point = camera.unproject(&Point2::new(0.0, -0.5)).unwrap();
        assert!(
            camera
                .bearing_error_rad(&pole_point, &Point2::new(700.0, -0.5))
                .unwrap()
                < 1e-14
        );
    }

    #[test]
    fn fisheye_bearing_error_round_trips_and_rejects_outside_domain() {
        let camera =
            FisheyeCamera::try_with_focal(300.0, 320.0, 240.0, FisheyeDistortion::K1(0.01))
                .unwrap();
        let point = Vector3::new(0.8, -0.3, 0.4);
        let pixel = camera.project(&point).unwrap();
        assert!(camera.bearing_error_rad(&point, &pixel).unwrap() < 1e-14);
        assert_eq!(
            camera
                .bearing_error_rad(&-Vector3::z(), &Point2::new(320.0, 240.0))
                .unwrap_err(),
            ProjectionError::OutsideModelDomain
        );
    }
}
