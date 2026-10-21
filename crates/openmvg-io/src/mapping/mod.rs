// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The typed reader and writer: OMVG tables to and from the SfM types.
//!
//! ```text
//! ViewGraph       <──>  dataset profile          cameras, views, keypoints, matches, pairs
//! Reconstruction  <──>  reconstruction profile   cameras, views, poses, points, observations
//! GroundTruth     <──   benchmark profile        cameras, views, gt_poses
//! Correspondences <──>  dataset [+ benchmark]    cameras, views, keypoints, matches, gt_poses
//! ```
//!
//! Both directions share the conventions of the SfM types (§4): pixel centers at integer
//! coordinates, world-to-camera poses, and relative poses from the first view to the second, so
//! no frame or pixel convention changes on the way. Numeric storage types and feature indices
//! can change. What the format does not record is not written: the
//! extraction method, the root seed, the kind of a non-file image source, and solver
//! diagnostics.

mod cameras;
mod columns;
mod correspondences;
mod export;
mod import;

#[cfg(test)]
mod tests;

pub use correspondences::{Correspondences, read_correspondences, write_correspondences};
pub use export::{write_reconstruction, write_view_graph};
pub use import::{
    GroundTruth, GroundTruthView, StoredReconstruction, read_ground_truth, read_reconstruction,
    read_view_graph,
};
