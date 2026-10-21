// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Camera intrinsics
//!
//! A camera model paired with the image domain it calibrates.
//!
//! ## Overview
//!
//! [`CameraIntrinsics`] validates that a camera model and image dimensions are
//! mutually compatible.
//!
//! ## Usage
//!
//! ```rust
//! use openmvg_core::camera::{Camera, PinholeCamera};
//! use openmvg_core::model::CameraIntrinsics;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Create the intrinsics of a pinhole camera
//! let camera = Camera::Pinhole(PinholeCamera::try_with_focal(3500.0, 960.0, 540.0)?);
//! let intrinsic = CameraIntrinsics::try_new(camera.clone(), 1920, 1080)?;
//!
//! // Hold the intrinsics fixed during bundle adjustment
//! let fixed_intrinsic = CameraIntrinsics::try_new(camera.clone(), 1920, 1080)?.with_fixed(true);
//! assert!(fixed_intrinsic.is_fixed());
//! # Ok(())
//! # }
//! ```

use crate::camera::{Camera, CameraModel, ProjectionError};
use nalgebra::Point2;
use serde::{Deserialize, Deserializer, Serialize};

/// Invalid image metadata attached to a camera model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CameraIntrinsicsError {
    /// Images must have a non-empty pixel domain.
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    EmptyImage {
        /// Attached image width.
        width: u32,
        /// Attached image height.
        height: u32,
    },
    /// A spherical model owns an exact equirectangular pixel domain.
    #[error(
        "spherical camera domain is {camera_width}x{camera_height}, not the attached {image_width}x{image_height} image"
    )]
    SphericalDimensionMismatch {
        /// Width encoded by the spherical camera.
        camera_width: u32,
        /// Height encoded by the spherical camera.
        camera_height: u32,
        /// Attached image width.
        image_width: u32,
        /// Attached image height.
        image_height: u32,
    },
    /// At least one sampled sensor location cannot be unprojected uniquely.
    #[error("camera cannot unproject sensor-domain sample ({column}, {row}): {source}")]
    NonInvertibleSensorDomain {
        /// Horizontal sample index in the validation grid.
        column: usize,
        /// Vertical sample index in the validation grid.
        row: usize,
        /// Camera-model failure at the sampled pixel.
        source: ProjectionError,
    },
}

/// Camera intrinsics: a full camera model and the image domain it calibrates.
///
/// This struct wraps a [`Camera`] with the image dimensions and a flag that keeps bundle
/// adjustment from refining it.
///
/// It stores the full camera model, distortion included, together with the image domain, so
/// triangulation can remove the distortion when unprojecting and reprojection errors can apply
/// it. Bundle adjustment refines a variable intrinsic only as far as its solver supports: the
/// native `openmvg-ba` self-calibration refines one focal length and the `k1`, `k2` radial
/// coefficients of a perspective camera, and keeps the principal point fixed.
///
/// # Examples
///
/// ```rust
/// use openmvg_core::camera::{Camera, PinholeCamera, BrownConradyCamera, RadialDistortion, TangentialDistortion};
/// use openmvg_core::model::CameraIntrinsics;
///
/// // Simple pinhole
/// let pinhole = Camera::Pinhole(PinholeCamera::try_with_focal(3500.0, 960.0, 540.0)?);
/// let pinhole_intrinsics = CameraIntrinsics::try_new(pinhole, 1920, 1080)?;
/// assert_eq!(pinhole_intrinsics.dimensions(), (1920, 1080));
///
/// // With distortion
/// let distorted = Camera::BrownConrady(BrownConradyCamera::try_with_focal(
///     3500.0, 960.0, 540.0,
///     RadialDistortion::K1K2K3(-0.1, 0.01, 0.0),
///     TangentialDistortion::None,
/// )?);
/// let distorted_intrinsics = CameraIntrinsics::try_new(distorted, 1920, 1080)?;
/// assert!(matches!(distorted_intrinsics.camera(), Camera::BrownConrady(c) if c.has_distortion()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct CameraIntrinsics {
    /// Full camera model (Pinhole, BrownConrady, Fisheye, Spherical).
    camera: Camera,

    /// Image width in pixels.
    width: u32,

    /// Image height in pixels.
    height: u32,

    /// If true, do not refine this intrinsic during bundle adjustment.
    ///
    /// Use this when you have high-confidence calibration data that
    /// should not be modified.
    fixed: bool,
}

impl CameraIntrinsics {
    /// Create camera intrinsics from a camera model and image dimensions in pixels.
    ///
    /// The intrinsic is variable by default (bundle adjustment may refine it).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::camera::{Camera, PinholeCamera};
    /// use openmvg_core::model::CameraIntrinsics;
    ///
    /// let camera = Camera::Pinhole(PinholeCamera::try_with_focal(3500.0, 960.0, 540.0)?);
    /// let intrinsic = CameraIntrinsics::try_new(camera, 1920, 1080)?;
    /// assert!(!intrinsic.is_fixed());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CameraIntrinsicsError`] if the image dimensions are empty, a
    /// spherical camera has incompatible dimensions, or the camera cannot
    /// unproject the sampled sensor domain.
    #[inline]
    pub fn try_new(camera: Camera, width: u32, height: u32) -> Result<Self, CameraIntrinsicsError> {
        let data = Self {
            camera,
            width,
            height,
            fixed: false,
        };
        data.validate()?;
        data.validate_camera_domain()?;
        Ok(data)
    }

    /// Validates the image-domain metadata.
    pub fn validate(&self) -> Result<(), CameraIntrinsicsError> {
        if self.width == 0 || self.height == 0 {
            Err(CameraIntrinsicsError::EmptyImage {
                width: self.width,
                height: self.height,
            })
        } else {
            Ok(())
        }
    }

    /// Validate compatibility between the camera model and its complete sensor domain.
    ///
    /// [`Self::try_new`] and [`Self::try_with_camera`] run it. It samples the whole sensor, so
    /// it costs more than [`Self::validate`]: Brown-Conrady and fisheye inverses are checked on
    /// a fixed 9×9 grid, and a spherical model must match the image dimensions exactly.
    pub fn validate_camera_domain(&self) -> Result<(), CameraIntrinsicsError> {
        self.validate()?;
        if let Camera::Spherical(camera) = &self.camera
            && (camera.width() != self.width || camera.height() != self.height)
        {
            return Err(CameraIntrinsicsError::SphericalDimensionMismatch {
                camera_width: camera.width(),
                camera_height: camera.height(),
                image_width: self.width,
                image_height: self.height,
            });
        }

        const GRID_STEPS: usize = 8;
        let maximum_x = f64::from(self.width.saturating_sub(1));
        let maximum_y = f64::from(self.height.saturating_sub(1));
        for row in 0..=GRID_STEPS {
            let y = maximum_y * row as f64 / GRID_STEPS as f64;
            for column in 0..=GRID_STEPS {
                let x = maximum_x * column as f64 / GRID_STEPS as f64;
                self.camera
                    .unproject(&Point2::new(x, y))
                    .map_err(|source| CameraIntrinsicsError::NonInvertibleSensorDomain {
                        column,
                        row,
                        source,
                    })?;
            }
        }
        Ok(())
    }

    /// Returns this intrinsic with the given fixed flag.
    ///
    /// A fixed intrinsic is held constant during bundle adjustment.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::camera::{Camera, PinholeCamera};
    /// use openmvg_core::model::CameraIntrinsics;
    ///
    /// let camera = Camera::Pinhole(PinholeCamera::try_with_focal(3500.0, 960.0, 540.0)?);
    /// let intrinsic = CameraIntrinsics::try_new(camera, 1920, 1080)?.with_fixed(true);
    /// assert!(intrinsic.is_fixed());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn with_fixed(mut self, fixed: bool) -> Self {
        self.fixed = fixed;
        self
    }

    /// Returns whether bundle adjustment holds this intrinsic constant.
    #[inline]
    pub fn is_fixed(&self) -> bool {
        self.fixed
    }

    /// Returns the image dimensions as (width, height).
    #[inline]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the image width in pixels.
    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns a reference to the camera model.
    #[inline]
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Returns these intrinsics with the camera model replaced, validated over the attached
    /// image domain.
    ///
    /// # Errors
    ///
    /// Returns [`CameraIntrinsicsError`] if the replacement camera is incompatible
    /// with the attached image domain.
    pub fn try_with_camera(self, camera: Camera) -> Result<Self, CameraIntrinsicsError> {
        let candidate = Self { camera, ..self };
        candidate.validate_camera_domain()?;
        Ok(candidate)
    }
}

impl<'de> Deserialize<'de> for CameraIntrinsics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            camera: Camera,
            width: u32,
            height: u32,
            fixed: bool,
        }

        let repr = Repr::deserialize(deserializer)?;
        let mut data = Self::try_new(repr.camera, repr.width, repr.height)
            .map_err(serde::de::Error::custom)?;
        data.fixed = repr.fixed;
        Ok(data)
    }
}

impl PartialEq for CameraIntrinsics {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.fixed == other.fixed
            && self.camera == other.camera
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{
        BrownConradyCamera, FisheyeCamera, FisheyeDistortion, PinholeCamera, RadialDistortion,
        SphericalCamera, TangentialDistortion,
    };

    fn sample_pinhole() -> Camera {
        Camera::Pinhole(PinholeCamera::try_with_focal(3500.0, 960.0, 540.0).unwrap())
    }

    #[test]
    fn construction_exposes_camera_and_image_geometry() {
        let camera = sample_pinhole();
        let data = CameraIntrinsics::try_new(camera.clone(), 1920, 1080).unwrap();

        assert_eq!(data.width(), 1920);
        assert_eq!(data.height(), 1080);
        assert_eq!(data.dimensions(), (1920, 1080));
        assert!(!data.is_fixed());
        assert_eq!(data.camera(), &camera);
    }

    #[test]
    fn rejects_empty_image_domains() {
        assert!(CameraIntrinsics::try_new(sample_pinhole(), 0, 1080).is_err());

        let valid = CameraIntrinsics::try_new(sample_pinhole(), 1920, 1080).unwrap();
        let mut value = serde_json::to_value(valid).expect("serialize intrinsic");
        value["height"] = serde_json::json!(0);
        assert!(serde_json::from_value::<CameraIntrinsics>(value).is_err());
    }

    #[test]
    fn validates_spherical_camera_dimensions_exactly() {
        let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
        assert!(CameraIntrinsics::try_new(camera.clone(), 1024, 512).is_ok());
        assert_eq!(
            CameraIntrinsics::try_new(camera, 1024, 500),
            Err(CameraIntrinsicsError::SphericalDimensionMismatch {
                camera_width: 1024,
                camera_height: 512,
                image_width: 1024,
                image_height: 500,
            })
        );
    }

    #[test]
    fn validates_distorted_camera_over_the_attached_sensor() {
        let valid_fisheye = Camera::Fisheye(
            FisheyeCamera::try_with_focal(300.0, 320.0, 240.0, FisheyeDistortion::K1(-0.05))
                .unwrap(),
        );
        assert!(CameraIntrinsics::try_new(valid_fisheye, 640, 480).is_ok());

        let folded_fisheye = Camera::Fisheye(
            FisheyeCamera::try_with_focal(100.0, 320.0, 240.0, FisheyeDistortion::K1(-1.0))
                .unwrap(),
        );
        assert!(matches!(
            CameraIntrinsics::try_new(folded_fisheye, 640, 480),
            Err(CameraIntrinsicsError::NonInvertibleSensorDomain { .. })
        ));

        let folded_brown = Camera::BrownConrady(
            BrownConradyCamera::try_with_focal(
                100.0,
                320.0,
                240.0,
                RadialDistortion::K1(-1.0),
                TangentialDistortion::None,
            )
            .unwrap(),
        );
        assert!(matches!(
            CameraIntrinsics::try_new(folded_brown, 640, 480),
            Err(CameraIntrinsicsError::NonInvertibleSensorDomain { .. })
        ));
    }

    #[test]
    fn fixed_flag_is_set_by_the_builder() {
        let data = CameraIntrinsics::try_new(sample_pinhole(), 1920, 1080).unwrap();
        assert!(!data.is_fixed());

        let data = data.with_fixed(true);
        assert!(data.is_fixed());
        assert!(!data.with_fixed(false).is_fixed());
    }

    #[test]
    fn camera_replacement_is_validated_over_the_image_domain() {
        let camera = sample_pinhole();
        let data = CameraIntrinsics::try_new(camera.clone(), 1920, 1080).unwrap();

        assert_eq!(data.camera(), &camera);

        let incompatible = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
        assert!(matches!(
            data.clone().try_with_camera(incompatible),
            Err(CameraIntrinsicsError::SphericalDimensionMismatch { .. })
        ));

        let replacement = Camera::try_pinhole(3600.0, 3600.0, 960.0, 540.0).unwrap();
        let replaced = data.try_with_camera(replacement.clone()).unwrap();
        assert_eq!(replaced.camera(), &replacement);
        assert_eq!(replaced.dimensions(), (1920, 1080));
    }

    #[test]
    fn serde_roundtrip_preserves_state() {
        let camera = sample_pinhole();
        let data = CameraIntrinsics::try_new(camera, 1920, 1080)
            .unwrap()
            .with_fixed(true);

        let json = serde_json::to_string(&data).unwrap();
        let parsed: CameraIntrinsics = serde_json::from_str(&json).unwrap();

        assert_eq!(data, parsed);
    }
}
