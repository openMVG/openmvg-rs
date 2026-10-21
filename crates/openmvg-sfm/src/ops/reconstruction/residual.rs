// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera-aware residual measurement and acceptance.

use nalgebra::{Point2, Vector3};
use openmvg_core::camera::{Camera, CameraModel};

use crate::{Result, SfmError};

/// Residual gate whose unit is selected from the camera projection family.
///
/// Perspective cameras retain image-space pixel thresholds. Fisheye and spherical cameras use
/// bearing angles, avoiding a pixel metric whose geometric meaning changes across a wide field of
/// view. Constructors translate the caller's existing threshold at the principal ray, so callers
/// do not need parallel pixel and angular configuration knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CameraResidualPolicy {
    PerspectivePixels {
        maximum: f64,
    },
    Angular {
        maximum_radians: f64,
        pixels_per_radian: f64,
    },
}

/// One residual expressed both in its acceptance domain and in the pipeline's reporting unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CameraResidualEvaluation {
    pub(crate) accepted: bool,
    pub(crate) pixel_equivalent: f64,
}

impl CameraResidualPolicy {
    /// Select a camera-aware gate from an existing pixel threshold.
    pub(crate) fn from_pixel_limit(camera: &Camera, maximum: f64) -> Result<Self> {
        if maximum.is_nan() || maximum < 0.0 {
            return Err(SfmError::InvalidConfig(
                "maximum pixel residual must be non-negative or +infinity".to_owned(),
            ));
        }
        if !is_generalized_camera(camera) {
            return Ok(Self::PerspectivePixels { maximum });
        }
        let pixels_per_radian = nominal_pixels_per_radian(camera)?;
        let maximum_radians = if maximum.is_infinite() {
            std::f64::consts::PI
        } else {
            (maximum / pixels_per_radian).min(std::f64::consts::PI)
        };
        Ok(Self::Angular {
            maximum_radians,
            pixels_per_radian,
        })
    }

    /// Measure one observation and retain a pixel-equivalent value for common reporting.
    pub(crate) fn evaluate(
        self,
        camera: &Camera,
        camera_point: &Vector3<f64>,
        observed: Point2<f64>,
    ) -> Option<CameraResidualEvaluation> {
        let (accepted, pixel_equivalent) = match self {
            Self::PerspectivePixels { maximum } => {
                let projected = camera.project(camera_point).ok()?;
                let residual = camera_pixel_residual(camera, projected, observed);
                (residual <= maximum, residual)
            }
            Self::Angular {
                maximum_radians,
                pixels_per_radian,
            } => {
                let residual = camera_bearing_error_rad(camera, camera_point, observed)?;
                (residual <= maximum_radians, residual * pixels_per_radian)
            }
        };
        (pixel_equivalent.is_finite()).then_some(CameraResidualEvaluation {
            accepted,
            pixel_equivalent,
        })
    }

    /// Test one camera-frame point against its measured image observation.
    pub(crate) fn accepts(
        self,
        camera: &Camera,
        camera_point: &Vector3<f64>,
        observed: Point2<f64>,
    ) -> bool {
        self.evaluate(camera, camera_point, observed)
            .is_some_and(|evaluation| evaluation.accepted)
    }
}

fn nominal_pixels_per_radian(camera: &Camera) -> Result<f64> {
    let scale = match camera {
        Camera::Spherical(spherical) => f64::from(spherical.width()) / std::f64::consts::TAU,
        Camera::Pinhole(_) | Camera::BrownConrady(_) | Camera::Fisheye(_) => {
            let matrix = camera.intrinsic_matrix().ok_or_else(|| {
                SfmError::InvalidInput(
                    "camera residual policy requires a finite focal scale".to_owned(),
                )
            })?;
            matrix[(0, 0)].abs().min(matrix[(1, 1)].abs())
        }
    };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SfmError::InvalidInput(
            "camera residual policy requires a positive finite pixel-to-angle scale".to_owned(),
        ));
    }
    Ok(scale)
}

/// Camera-aware image-space residual in pixels.
///
/// Equirectangular longitude is periodic, so spherical horizontal differences take the
/// shortest wrapped path across the seam. Other camera models use ordinary Euclidean pixels.
pub(crate) fn camera_pixel_residual(
    camera: &Camera,
    projected: Point2<f64>,
    observed: Point2<f64>,
) -> f64 {
    if !projected.coords.iter().all(|value| value.is_finite())
        || !observed.coords.iter().all(|value| value.is_finite())
    {
        return f64::INFINITY;
    }
    let mut dx = (projected.x - observed.x).abs();
    if let Camera::Spherical(spherical) = camera {
        let width = spherical.width() as f64;
        dx %= width;
        dx = dx.min(width - dx);
    }
    dx.hypot(projected.y - observed.y)
}

/// Camera-model-aware angular residual in radians.
pub(crate) fn camera_bearing_error_rad(
    camera: &Camera,
    camera_point: &Vector3<f64>,
    observed: Point2<f64>,
) -> Option<f64> {
    camera.bearing_error_rad(camera_point, &observed).ok()
}

pub(super) fn is_generalized_camera(camera: &Camera) -> bool {
    matches!(camera, Camera::Fisheye(_) | Camera::Spherical(_))
}
