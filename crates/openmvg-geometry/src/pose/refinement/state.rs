// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::numerics::manifold::{Mat3, Pose3, Pose3Error};

/// Private Euclidean storage used while refining an absolute camera pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefinementPose {
    /// World-to-camera rotation matrix `R` in `x_cam = R * x_world + t`.
    pub rotation: Mat3,
    /// World-to-camera translation `t` in `x_cam = R * x_world + t`.
    pub translation: [f64; 3],
}

impl RefinementPose {
    /// Wrap a world-to-camera rotation and translation without validation.
    pub const fn new(rotation: Mat3, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Convert to a validated [`Pose3`]; fails when `rotation` is not a finite
    /// member of `SO(3)` or `translation` is non-finite.
    pub fn try_as_pose3(self) -> Result<Pose3, Pose3Error> {
        self.try_into()
    }
}

impl TryFrom<RefinementPose> for Pose3 {
    type Error = Pose3Error;

    fn try_from(pose: RefinementPose) -> Result<Self, Self::Error> {
        Self::try_from_parts(pose.rotation, pose.translation)
    }
}

impl From<Pose3> for RefinementPose {
    fn from(pose: Pose3) -> Self {
        Self {
            rotation: pose.rotation().into_matrix(),
            translation: pose.translation(),
        }
    }
}
