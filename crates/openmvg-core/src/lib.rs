// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera models, poses, identifiers, typed units and the seed protocol that every OpenMVG-rs
//! crate shares.
//!
//! This is the vocabulary of the library: `openmvg-geometry` solves for poses and points in
//! these types, and `openmvg-sfm` builds reconstructions out of them. The crate holds values and
//! their checks, with no solvers.
//!
//! ## Start here
//!
//! - [`Camera`] is one type for every camera model. It projects camera-frame points to pixels
//!   and turns pixels back into bearings.
//! - [`WorldToCameraPose`] is a camera's rotation and center, the pose convention of every crate.
//! - [`PixelThreshold`], [`AngularThreshold`] and [`RunSeed`] are the typed inputs that solvers
//!   and engines take.
//!
//! Points, vectors and rotations are [`nalgebra`] 0.35 types, which this crate does not
//! re-export. Depend on the same version so that your values are the types the API takes:
//!
//! ```toml
//! [dependencies]
//! nalgebra = "0.35"
//! openmvg-core = "0.1"
//! ```
//!
//! ## Example
//!
//! Project a world point into a camera, then recover the ray that sees it:
//!
//! ```rust
//! use nalgebra::{Point3, Rotation3};
//! use openmvg_core::camera::{Camera, CameraModel};
//! use openmvg_core::WorldToCameraPose;
//!
//! // A pinhole camera: focal length 800 px, principal point at pixel (320, 240).
//! let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
//! // A camera at the world origin that looks along +Z.
//! let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())?;
//!
//! let point = Point3::new(0.5, -0.25, 4.0);
//! let pixel = camera.project(&pose.project_to_camera(&point).coords)?;
//! assert!((pixel.x - 420.0).abs() < 1e-9 && (pixel.y - 190.0).abs() < 1e-9);
//!
//! // The bearing is the unit ray from the camera center through that pixel.
//! let bearing = camera.bearing(&pixel)?;
//! assert!((bearing.into_inner() - point.coords.normalize()).norm() < 1e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Learning path
//!
//! 1. [`camera`]: the pinhole, Brown-Conrady, fisheye and spherical models, projection and
//!    lens distortion.
//! 2. [`model`]: identifiers such as [`ViewId`] and [`PoseId`], poses, intrinsics, relative
//!    measurements, and the estimation error.
//! 3. [`units`]: thresholds and residuals that carry their unit in their type.
//! 4. [`determinism`]: how a [`RunSeed`] becomes reproducible random streams.
//! 5. [`graph`]: checked graph algorithms over views, such as connected components and spanning
//!    trees.
//!
//! From a clone of the repository, the `camera_projection` example walks through the frames,
//! the pixel convention, distortion and bearings, and checks each step:
//! `cargo run --release -p openmvg-core --example camera_projection`.
//!
//! The solvers that use these types are in `openmvg-geometry`.
//!
//! ## Common mistakes
//!
//! - Passing a translation where a camera center is expected.
//!   [`WorldToCameraPose::from_rotation_center`] takes the center `C`; the translation is
//!   `t = −R C`.
//! - Passing a camera-to-world rotation, the camera's orientation in the world as robotics code
//!   usually stores it. A pose takes the world-to-camera rotation, its inverse; [Converting
//!   poses](#converting-poses) shows both directions.
//! - Reading pixel coordinates whose origin is the top-left pixel corner, as in COLMAP files,
//!   without the half-pixel shift.
//! - Building a threshold in the wrong unit. Each solver takes the threshold type of its
//!   residual, so the compiler rejects a pixel threshold where radians are expected.
//!
//! ## Conventions
//!
//! Every crate in OpenMVG-rs uses the frames and units below. This section defines them once.
//!
//! - Pixel coordinates `(u, v)` start at the center of the top-left pixel: `u` grows to the
//!   right, `v` grows down, and every pixel center sits on integer coordinates. COLMAP files put
//!   that center at `(0.5, 0.5)`; `openmvg-io` adds the half-pixel shift when it writes them.
//! - The camera frame has `X` pointing right, `Y` down, and `Z` forward along the optical axis,
//!   into the scene. A pinhole camera maps a camera-frame point to the pixel
//!   `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`, where `(cx, cy)` is the principal point. OpenGL
//!   cameras point `Y` up and look along `−Z`; [Converting poses](#converting-poses) shows the
//!   conversion. In ROS this camera frame is the `*_optical_frame` convention; a ROS body or
//!   `camera_link` frame (`X` forward, `Z` up) differs from it by a fixed rotation.
//! - A pose maps world coordinates to camera coordinates. [`WorldToCameraPose`] stores the
//!   rotation `R` from world to camera and the camera center `C` in world coordinates, so
//!   `X_cam = R (X_world − C) = R X_world + t` with `t = −R C`. The world frame is any
//!   right-handed frame you choose.
//! - A bearing is the unit ray from the camera center toward a point, in the camera frame:
//!   `X_cam / ‖X_cam‖`. [`Camera::bearing`] returns it for a pixel. The calibrated solvers, such
//!   as the essential matrix and P3P, take bearings. Those that score an angular error, such as
//!   `openmvg_geometry::essential::estimate_essential_angular` and
//!   `openmvg_geometry::pose::estimate_absolute_pose`, work with every camera model;
//!   `estimate_essential` scores in normalized image units and suits perspective cameras.
//! - Thresholds and residuals carry their unit in their type: [`PixelThreshold`],
//!   [`AngularThreshold`] (radians), [`NormalizedImageThreshold`] and [`WorldDistanceThreshold`].
//!   [`units`] names the solvers that take each.
//! - Randomized algorithms take a [`RunSeed`] from the caller, and the same inputs and seed give
//!   the same result. [`determinism`] describes the protocol.
//!
#![doc = include_str!("../figures/conventions.svg")]
//!
//! ## Converting poses
//!
//! - COLMAP stores the same world-to-camera pose: the rotation `R` as a quaternion
//!   `QW QX QY QZ`, and the translation `t` as `TX TY TZ`. Its pixel coordinates, keypoints and
//!   principal point alike, are 0.5 larger than ours. `openmvg_io::write_colmap` does these
//!   conversions and maps each camera model to a COLMAP one, except spherical cameras, which
//!   COLMAP cannot model; its documentation lists the mapping.
//! - A camera-to-world pose holds the inverse rotation `Rᵀ` and the camera position, which is
//!   the center `C`.
//! - An OpenGL view matrix flips the camera's `Y` and `Z` axes: it is `diag(1, −1, −1, 1)` times
//!   the 4 × 4 matrix `[R | t]`. OpenGL window coordinates start at the bottom-left image
//!   corner, so our pixel `(u, v)` is at `(u + 0.5, height − v − 0.5)` there, with the image
//!   height of [`CameraIntrinsics::height`](model::CameraIntrinsics::height). An OpenGL
//!   projection matrix can hold the focal lengths and the principal point, but not lens
//!   distortion: undistort the image first.
//!
//! ```rust
//! use nalgebra::{Matrix4, Point3, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4};
//! use openmvg_core::WorldToCameraPose;
//!
//! // A camera 2 units behind the world origin, turned 30° about the Y axis.
//! let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), 30_f64.to_radians());
//! let pose = WorldToCameraPose::from_rotation_center(rotation, Point3::new(0.0, 0.0, -2.0))?;
//!
//! // To COLMAP, and back.
//! let q = UnitQuaternion::from_rotation_matrix(pose.rotation());
//! let (qvec, tvec) = ([q.w, q.i, q.j, q.k], pose.translation());
//! let quaternion = Quaternion::new(qvec[0], qvec[1], qvec[2], qvec[3]);
//! let rotation = UnitQuaternion::from_quaternion(quaternion).to_rotation_matrix();
//! let center = Point3::from(-(rotation.inverse() * tvec));
//! let from_colmap = WorldToCameraPose::from_rotation_center(rotation, center)?;
//! assert!((from_colmap.center() - pose.center()).norm() < 1e-12);
//!
//! // From a camera-to-world rotation and the camera position.
//! let (camera_to_world, position) = (pose.rotation().inverse(), *pose.center());
//! let world_to_camera = camera_to_world.inverse();
//! let from_robotics = WorldToCameraPose::from_rotation_center(world_to_camera, position)?;
//! assert!((from_robotics.translation() - pose.translation()).norm() < 1e-12);
//!
//! // To an OpenGL view matrix: a point in front of the camera gets a negative Z.
//! let flip = Matrix4::from_diagonal(&Vector4::new(1.0, -1.0, -1.0, 1.0));
//! let view = flip * pose.to_isometry().to_homogeneous();
//! assert!((view * Point3::new(0.0, 0.0, 5.0).to_homogeneous()).z < 0.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Features
//!
//! The crate has no optional features. It builds for native targets and for
//! `wasm32-unknown-unknown`, and it never reads host or browser entropy: randomized callers
//! supply their seeds.
//!

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![deny(rustdoc::broken_intra_doc_links)]
// Unit tests and fixed fixtures may use assertion-oriented conveniences. Library
// code must keep failure explicit at public data boundaries.
#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

pub mod camera;
pub mod determinism;
pub mod graph;
pub mod model;
pub mod units;

// Re-export camera types
pub use camera::{
    BrownConradyCamera, Camera, CameraError, CameraModel, DistortionError, FisheyeCamera,
    FisheyeDistortion, PinholeCamera, ProjectionError, RadialDistortion, SphericalCamera,
    TangentialDistortion,
};
pub use determinism::{AlgorithmRng, RunSeed};
pub use model::{
    ComponentId, FeatureId, IntrinsicId, LandmarkId, PoseId, TrackId, ViewId, ViewPair,
    WorldToCameraPose,
};

pub use units::{
    AngularResidual, AngularThreshold, Meters, NormalizedImage, NormalizedImageResidual,
    NormalizedImageThreshold, PixelResidual, PixelThreshold, Pixels, Radians, Residual,
    ResidualError, ScalarResidual, ScalarThreshold, Threshold, ThresholdError, ThresholdResult,
    Unitless, WorldDistanceResidual, WorldDistanceThreshold, WorldUnits,
};

// Compile the README example.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
