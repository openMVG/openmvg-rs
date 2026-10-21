// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Geometric solvers and robust estimation: two-view models, camera pose, triangulation,
//! alignment and averaging.
//!
//! Given matched points between images, this crate recovers how the cameras relate and where
//! the points are. `openmvg-features` produces the matches, and `openmvg-sfm` chains these
//! solvers into a full reconstruction. Every solver works on the types and conventions of
//! `openmvg-core`.
//!
//! ## Start here
//!
//! - The `estimate_*` functions find a model in correspondences that contain wrong matches:
//!   [`estimate_homography`](homography::estimate_homography),
//!   [`estimate_fundamental`](fundamental::estimate_fundamental),
//!   [`estimate_essential`](essential::estimate_essential),
//!   [`estimate_absolute_pose`](pose::estimate_absolute_pose) and
//!   [`estimate_rigid_alignment`](alignment::estimate_rigid_alignment).
//! - [`correspondence`] holds their inputs: pixel pairs, bearing pairs, or 3D point pairs.
//! - [`triangulation`] and [`averaging`] turn posed views and view graphs into points and
//!   global poses.
//!
//! ## Example
//!
//! Recover a homography from 50 matches, 5 of them wrong:
//!
//! ```rust
//! use nalgebra::{Matrix3, Point2};
//! use openmvg_core::PixelThreshold;
//! use openmvg_geometry::correspondence::PixelCorrespondence;
//! use openmvg_geometry::homography::estimate_homography;
//! use openmvg_geometry::robust::RobustOptions;
//!
//! // A known homography maps a grid in the first image to the second image.
//! let truth = Matrix3::new(1.1, 0.02, 15.0, -0.03, 0.95, -8.0, 1e-4, -2e-4, 1.0);
//! let mut matches: Vec<_> = (0..45)
//!     .map(|i| {
//!         let p = Point2::new(20.0 * f64::from(i % 9), 30.0 * f64::from(i / 9));
//!         let q = truth * p.to_homogeneous();
//!         PixelCorrespondence::new(p, Point2::new(q.x / q.z, q.y / q.z))
//!     })
//!     .collect();
//! // Five wrong matches, far from where the homography sends their points.
//! matches.extend((0..5).map(|i| {
//!     let p = Point2::new(15.0 + 30.0 * f64::from(i), 100.0);
//!     PixelCorrespondence::new(p, Point2::new(p.x + 80.0, p.y - 60.0))
//! }));
//!
//! let threshold = PixelThreshold::try_new(1.0)?;
//! let estimate = estimate_homography(&matches, threshold, &RobustOptions::default())?;
//! assert_eq!(estimate.inliers(), (0..45).collect::<Vec<_>>());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Getting good results
//!
//! - Use an `estimate_*` function on real matches. The minimal solvers behind it, such as
//!   [`FivePointSolver`](essential::FivePointSolver), assume a clean sample and return every
//!   candidate that passes their checks.
//! - Set the threshold in the residual's unit, a few times the measurement noise. A threshold
//!   much larger admits wrong matches; one below the noise discards good ones.
//! - Choose the two-view model from the scene and the calibration: a homography for a planar
//!   scene or a camera that only rotates, the essential matrix for calibrated cameras, and the
//!   fundamental matrix for uncalibrated ones. A planar scene does not determine the
//!   fundamental matrix, and when the camera only rotates there is no baseline, so neither the
//!   essential nor the fundamental matrix is defined; use a homography in those cases.
//! - With calibrated cameras, estimate the essential matrix from bearings rather than the
//!   fundamental matrix from pixels, since it uses the known intrinsics:
//!   [`estimate_essential`](essential::estimate_essential) for perspective cameras, and
//!   [`estimate_essential_angular`](essential::estimate_essential_angular) for fisheye and
//!   spherical ones.
//! - [`RobustOptions`](robust::RobustOptions) fixes the random seed, so a run repeats exactly;
//!   change the seed to see how stable an estimate is.
//!
//! ## Common mistakes
//!
//! - Giving pixel coordinates to a calibrated solver, which expects bearings. Convert pixels
//!   with `openmvg_core::camera::Camera::bearing`.
//! - Reading a returned pose as camera-to-world. Poses are world-to-camera, as in
//!   `openmvg-core`.
//! - Calling a minimal solver on noisy data with outliers instead of the matching
//!   `estimate_*` function.
//!
//! ## Learning path
//!
//! 1. [`correspondence`]: the inputs, as pixels, bearings, or 3D points.
//! 2. [`robust`]: how RANSAC and MSAC find a model among wrong matches, and how to run your own
//!    solver in the same loop through [`robust::Solver`].
//! 3. [`homography`], [`fundamental`], [`essential`]: models between two views.
//! 4. [`pose`]: a camera's pose from 3D points, with P3P and resection, and its refinement.
//! 5. [`triangulation`]: 3D points from posed views.
//! 6. [`alignment`]: the rigid or similarity transform between two point sets.
//! 7. [`averaging`]: global rotations and positions from the pairwise relations of a view
//!    graph.
//!
//! From a clone of the repository, three examples build a known scene, run the estimators and
//! check what they recover: `robust_homography` (wrong matches, thresholds, and why a
//! least-squares fit fails), `relative_pose` (bearings, the essential matrix and its
//! decomposition), and `triangulate_and_resect`, which continues from the relative pose to
//! triangulated points and the pose of a third camera. Run them with
//! `cargo run --release -p openmvg-geometry --example <name>`.
//!
//! [`residuals`] and [`normalization`] hold the error measures and the conditioning that the
//! solvers share. Each solver's page explains its method with a diagram, and most pages also give
//! its failure modes and references.
//!
//! | Problem | Minimal solver, for hypotheses | Non-minimal or global solver |
//! |---|---|---|
//! | Fundamental matrix | 7-point | 8-point |
//! | Essential matrix | 5-point | angular and pinhole 8-point |
//! | Homography | 4-point ACA (Affine-Core-Affine) | DLT |
//! | Absolute pose | P3P family | calibrated and uncalibrated resection |
//! | Alignment | 3-point Kabsch | weighted Kabsch or Umeyama fit |
//! | Triangulation | two-view angular or midpoint | N-view DLT and robust track triangulation |
//! | Global Structure from Motion (SfM) | | rotation and translation averaging, global positioning |
//!
//! ## Features
//!
//! All features are off by default; `openmvg-sfm` and the `openmvg` umbrella turn on `faer`
//! and `parallel`.
//!
//! | Feature | Effect |
//! |---|---|
//! | `faer` | Sparse rotation averaging (`averaging::average_rotations`), the sparse Cholesky of global positioning, and the parallel-rigidity audit of large view graphs; turns on `openmvg-ba/faer` |
//! | `parallel` | Rayon for global positioning; the serial fallbacks give identical results |
//! | `unstable-test-support` | Not for applications: hidden hooks for this crate's own tests, outside semver |
//!
//! Positions and poses follow the conventions of `openmvg-core`: pixel centers on integer
//! coordinates and world-to-camera poses.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Unit tests and fixed fixtures may use assertion-oriented conveniences.
// Library code must keep failure explicit at public data boundaries.
#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

#[macro_use]
mod macros;

pub mod alignment;
pub mod averaging;
pub mod correspondence;
pub mod essential;
pub mod fundamental;
pub mod homography;
pub mod normalization;
pub mod pose;
pub mod residuals;
pub mod robust;
pub(crate) mod shared;
mod solver_error;
mod solver_traits;
pub mod triangulation;

// The shared solver error type is the only root item; every other item lives
// at its module path.
pub use solver_error::{SolverError, SolverResult};

// Compile the README example.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
