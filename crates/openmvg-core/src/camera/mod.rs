// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Camera models
//!
//! Camera models that map camera-frame points to pixels and pixels back to bearings.
//!
//! ## Overview
//!
//! The camera module is organized around the [`CameraModel`] trait, which defines a common interface
//! for all camera types. Code can be written once against the trait, or use the
//! [`Camera`] enum to hold any supported model; [`Camera::bearing`] gives the unit ray of a
//! pixel for every model.
//!
//! ## Available camera models
//!
//! - [`PinholeCamera`]: a perspective camera without distortion, with one focal length or
//!   separate `fx` and `fy`.
//! - [`BrownConradyCamera`]: a pinhole camera with Brown-Conrady lens distortion.
//! - [`FisheyeCamera`]: the OpenCV-compatible equidistant fisheye model.
//! - [`SphericalCamera`]: an equirectangular camera for 360-degree panoramas.
//!
//! ## Usage example
//!
//! ```rust
//! use openmvg_core::camera::{PinholeCamera, SphericalCamera, CameraModel};
//! use nalgebra::{Point2, Vector3};
//!
//! // Create different camera types
//! let pinhole_single = PinholeCamera::try_with_focal(800.0, 320.0, 240.0)?;
//! let pinhole_double = PinholeCamera::try_new(800.0, 600.0, 320.0, 240.0)?;
//! // An equirectangular panorama 2048 pixels wide (and 1024 high).
//! let spherical = SphericalCamera::try_new(2048)?;
//!
//! // Use polymorphically
//! let point = Vector3::new(0.0, 0.0, 1.0);
//! let pixel_single = pinhole_single.project(&point)?;
//! let pixel_double = pinhole_double.project(&point)?;
//! let pixel_spherical = spherical.project(&point)?;
//!
//! // A point on the optical axis lands on the principal point of a pinhole camera.
//! assert_eq!(pixel_single, Point2::new(320.0, 240.0));
//! assert_eq!(pixel_double, Point2::new(320.0, 240.0));
//! // Unprojecting the pixel gives back the direction of the point.
//! let ray = spherical.unproject(&pixel_spherical)?;
//! assert!((ray.normalize() - point).norm() < 1e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod brown_conrady;
mod camera_enum;
mod camera_model;
mod distortion;
mod fisheye;
mod model_type;
mod pinhole;
mod spherical;

pub use brown_conrady::BrownConradyCamera;
pub use camera_enum::Camera;
pub(crate) use camera_model::try_normalize_bearing;
pub use camera_model::{CameraError, CameraModel, ProjectionError};
pub use distortion::{DistortionError, FisheyeDistortion, RadialDistortion, TangentialDistortion};
pub use fisheye::FisheyeCamera;
pub use model_type::{CameraModelType, ParseCameraModelTypeError};
pub use pinhole::PinholeCamera;
pub use spherical::SphericalCamera;
