// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private camera representations used by the bundle-adjustment kernels.

use crate::numerics::manifold::{Mat3, Pose3, Pose3Error};

/// Calibrated world-to-camera pose: `x_camera = rotation * x_world + translation`.
///
/// In two-view BA, camera 1 is the world frame and this value is camera 2's
/// relative pose. Multi-view BA stores one value per camera and requires
/// camera 0 to be the identity world frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativePose {
    /// World-to-camera rotation in `SO(3)`.
    pub rotation: Mat3,
    /// World-to-camera translation.
    pub translation: [f64; 3],
}

impl RelativePose {
    /// Construct a world-to-camera pose without validating its components.
    ///
    /// Refinement entry points validate this compatibility representation
    /// before use. Prefer [`try_from_parts`](Self::try_from_parts), or use
    /// [`Pose3`] directly, when invalid states should be unrepresentable.
    pub const fn new(rotation: Mat3, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Validate the rotation and translation before constructing a pose.
    pub fn try_from_parts(rotation: Mat3, translation: [f64; 3]) -> Result<Self, Pose3Error> {
        Pose3::try_from_parts(rotation, translation).map(Into::into)
    }

    /// Convert this compatibility representation into a checked [`Pose3`].
    pub fn try_as_pose3(self) -> Result<Pose3, Pose3Error> {
        self.try_into()
    }
}

impl TryFrom<RelativePose> for Pose3 {
    type Error = Pose3Error;

    fn try_from(pose: RelativePose) -> Result<Self, Self::Error> {
        Self::try_from_parts(pose.rotation, pose.translation)
    }
}

impl From<Pose3> for RelativePose {
    fn from(pose: Pose3) -> Self {
        Self {
            rotation: pose.rotation().into_matrix(),
            translation: pose.translation(),
        }
    }
}

/// Pinhole intrinsics used by pixel-space bundle adjustment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeCamera {
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Horizontal principal point in pixels.
    pub cx: f64,
    /// Vertical principal point in pixels.
    pub cy: f64,
    /// Horizontal skew coefficient in pixels.
    pub skew: f64,
}

impl PinholeCamera {
    /// Construct zero-skew pinhole intrinsics.
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            skew: 0.0,
        }
    }

    /// Return these intrinsics with an explicit skew coefficient.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn with_skew(mut self, skew: f64) -> Self {
        self.skew = skew;
        self
    }

    /// Whether every coefficient is finite and both focal lengths are positive.
    pub fn is_valid(self) -> bool {
        [self.fx, self.fy, self.cx, self.cy, self.skew]
            .into_iter()
            .all(f64::is_finite)
            && self.fx > 0.0
            && self.fy > 0.0
    }
}
