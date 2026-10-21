// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Structure from Motion (SfM) values: identifiers, poses, intrinsics,
//! relative measurements, and the estimation error.
//!
//! End-to-end workflow orchestration lives in the `openmvg-sfm` crate.

mod ids;
mod intrinsic;
mod relative;
mod state;

pub use ids::{ComponentId, FeatureId, IntrinsicId, LandmarkId, PoseId, TrackId, ViewId, ViewPair};
pub use intrinsic::{CameraIntrinsics, CameraIntrinsicsError};
pub use relative::{
    EstimationError, EstimationResult, RelativePose, RelativeRotation, RelativeTranslation,
    rotation_angle_deg,
};
pub use state::{StateValidationError, WorldToCameraPose};
