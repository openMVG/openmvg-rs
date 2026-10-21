// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Averaging: every camera's rotation and position from the pairwise relations of a view graph.
//!
//! Each verified image pair gives a relative rotation and a translation direction. Averaging
//! solves for all cameras at once so that these pairwise measurements agree as well as possible:
//! first the rotations, then the positions. It is the core of global Structure from Motion
//! (SfM).
//!
//! ```text
//! relative measurements -> graph checks/filtering -> rotation averaging
//!                       -> translation averaging -> global positioning
//!                       -> camera and landmark state
//! ```
//!
//! Pairwise measurements fix the cameras only up to a *gauge*: one global rotation for the
//! rotations, and one translation and one scale for the positions. Every solver fixes the frame
//! by holding one camera, so the result is in an arbitrary frame and, for positions, at an
//! arbitrary scale.
//!
//! ## Module contents
//!
//! - Rotation averaging: [`average_rotations`] runs the default robust sparse solver
//!   (requires `faer`; [`average_rotations_with_robust_loss`] selects the loss), and
//!   [`average_rotations_dense`] runs the same robust loop over a dense solve with no
//!   optional dependency (Chatterjee & Govindu, ICCV 2013; Martinec & Pajdla, CVPR 2007).
//! - Translation averaging: [`average_translations`] runs L1-IRLS (iteratively reweighted least
//!   squares that minimizes the sum of absolute errors, so a few wrong directions weigh little).
//! - [`filtering`] and [`view_graph`]: pre-averaging topology, rigidity and
//!   cycle-consistency checks on the relative measurements.
//! - [`global_positioning`]: camera positions solved jointly with the 3D points from the track
//!   bearings, as in GLOMAP (Pan et al., ECCV 2024), with the bounded objective of BATA (Zhuang
//!   et al., CVPR 2018) that keeps long camera-to-point distances from dominating; an independent
//!   implementation, [`global_positioning::position_globally`].
//!
//! ### Usage example
//!
//! The example averages rotations with [`average_rotations`], which needs the `faer` feature:
//!
//! ```toml
//! [dependencies]
//! openmvg-geometry = { version = "0.1", features = ["faer"] }
//! ```
//!
//! Without `faer`, call [`average_rotations_dense`] with the same arguments; it suits small
//! graphs.
//!
//! Three cameras, their relative rotations `R_ij = R_j R_iᵀ` and their relative directions
//! `d_ij = R_i (C_j − C_i) / |C_j − C_i|`, seen from camera `i`, give back every rotation and
//! every center, up to a common rotation, position and scale:
//!
//! ```rust
//! # #[cfg(feature = "faer")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use nalgebra::{Point3, Rotation3, Vector3};
//! use openmvg_core::model::{RelativeRotation, RelativeTranslation, ViewId, rotation_angle_deg};
//! use openmvg_geometry::averaging::{
//!     RotationConfig, TranslationConfig, average_rotations, average_translations,
//! };
//!
//! let rotations = [0.0, 0.2, -0.3].map(|angle| Rotation3::from_euler_angles(0.0, angle, 0.1));
//! let centers = [Point3::origin(), Point3::new(1.0, 0.0, 0.2), Point3::new(0.4, 0.8, 0.0)];
//! let (mut relative_rotations, mut relative_directions) = (Vec::new(), Vec::new());
//! for (i, j) in [(0, 1), (0, 2), (1, 2)] {
//!     let (a, b) = (ViewId::new(i), ViewId::new(j));
//!     let rotation = rotations[j] * rotations[i].inverse();
//!     relative_rotations.push(RelativeRotation::try_new(a, b, rotation, 1.0)?);
//!     let direction = rotations[i] * (centers[j] - centers[i]);
//!     relative_directions.push(RelativeTranslation::try_new(a, b, direction, 1.0)?);
//! }
//!
//! // Rotations first, then the centers that agree with them.
//! let averaged = average_rotations(&relative_rotations, &RotationConfig::default())?;
//! let positioned = average_translations(
//!     &relative_directions,
//!     &averaged.rotations,
//!     &TranslationConfig::default(),
//! )?;
//!
//! // Compare in the first camera's frame, which removes the common rotation and position.
//! let (r0, c) = (averaged.rotations[&ViewId::new(0)], &positioned.translations);
//! let relative = averaged.rotations[&ViewId::new(2)] * r0.inverse();
//! let expected = rotations[2] * rotations[0].inverse();
//! assert!(rotation_angle_deg(&(relative * expected.inverse())) < 1e-6);
//! let seen = r0 * (c[&ViewId::new(2)] - c[&ViewId::new(0)]);
//! let truth: Vector3<f64> = rotations[0] * (centers[2] - centers[0]);
//! assert!(seen.normalize().angle(&truth.normalize()) < 1e-6);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "faer"))]
//! # fn main() {}
//! ```
//!
//! ## Available algorithms
//!
//! ### Rotation averaging
//!
//! | Solver | Method | Use case |
//! |--------|--------|----------|
//! | [`average_rotations`] | Robust sparse L2 | Default (requires `faer`) |
//! | [`average_rotations_dense`] | Robust dense L2 | Small graphs, builds without `faer` |
//!
//! ### Translation averaging (camera↔camera relative directions)
//!
//! | Solver | Method | Use case |
//! |--------|--------|----------|
//! | [`average_translations`] | L1-IRLS | General use; the L1 loss down-weights outliers |
//!
//! ### Global positioning (camera↔point track constraints, preferred over relative-direction averaging)
//!
//! | Solver | Method | Use case |
//! |--------|--------|----------|
//! | [`global_positioning::position_globally`] | BATA-style bounded objective (independent implementation) with point and positive-scale elimination | Camera↔point positioning; deterministic |
//!
//! ## View graph filtering
//!
//! [`filtering`] and [`view_graph`] audit and clean the view graph before averaging: they find
//! disconnected or non-rigid parts and drop pairs whose relative rotation disagrees with the
//! others, so that averaging works on consistent measurements.
//!
//! # References
//!
//! - A. Chatterjee, V. M. Govindu, "Efficient and Robust Large-Scale Rotation Averaging",
//!   ICCV 2013.
//! - D. Martinec, T. Pajdla, "Robust Rotation and Translation Estimation in Multiview
//!   Reconstruction", CVPR 2007.
//! - B. Zhuang, L.-F. Cheong, G. H. Lee, "Baseline Desensitizing in Translation Averaging",
//!   CVPR 2018.
//! - L. Pan, D. Baráth, M. Pollefeys, J. L. Schönberger, "Global Structure-from-Motion
//!   Revisited", ECCV 2024. arXiv:2407.20219.

// The solver links target `faer`-gated items; full-feature doc builds check them.
#![cfg_attr(not(feature = "faer"), allow(rustdoc::broken_intra_doc_links))]

mod contracts;
pub mod filtering;
pub mod global_positioning;

fn checked_angular_residual(
    value: f64,
) -> openmvg_core::model::EstimationResult<openmvg_core::AngularResidual> {
    openmvg_core::AngularResidual::try_new(value)
        .map_err(|error| openmvg_core::model::EstimationError::Numerical(error.to_string()))
}
mod rotation;
mod translation;
pub mod view_graph;
pub(crate) mod view_id_mapping;

pub use contracts::{RotationConfig, RotationEstimate, TranslationConfig, TranslationEstimate};
pub(crate) use contracts::{RotationSolver, TranslationSolver};

pub use rotation::average_rotations_dense;
#[cfg(feature = "faer")]
pub use rotation::{average_rotations, average_rotations_with_robust_loss};
pub use translation::average_translations;
