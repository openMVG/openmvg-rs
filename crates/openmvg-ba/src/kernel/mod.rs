// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structure-exploiting two-view and multi-view bundle adjustment.
//!
//! The two-view path uses a fixed-size 5x5 camera Schur solve. Multi-view paths
//! use reusable dense-camera workspaces for calibrated or opt-in centered
//! radial self-calibrating models. All paths eliminate independent 3x3 point
//! blocks and keep their iterative hot paths allocation-free.

//!
//! The solvers are compiled only for the native adjuster (`faer`) or the
//! validation surface (`unstable-internals`). Without either feature only the
//! projection and point-block helpers used by [`crate::Solution::weak_points`]
//! remain.

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
mod camera;
#[cfg(feature = "faer")]
mod fixed_rotation;
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub mod multi_view;
mod point_block;
mod point_geometry;
mod projection;
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub mod two_view;

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub use camera::{PinholeCamera, RelativePose};
#[cfg(all(feature = "faer", feature = "unstable-internals"))]
pub use fixed_rotation::FixedRotationError;
#[cfg(feature = "faer")]
pub use fixed_rotation::{
    FixedRotationGauge, FixedRotationIntrinsics, FixedRotationObservabilityProbe,
    FixedRotationObservabilityProbeOptions, FixedRotationObservation, FixedRotationProbeReason,
    FixedRotationProbeStatus, FixedRotationProblem, probe_fixed_rotation_observability,
};
pub(crate) use point_block::POINT_BLOCK_QUARANTINE_RCOND;
pub(crate) use point_geometry::TrackGeometry;
pub(crate) use projection::angular_projection;
