// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact fixed blocks and unary Gaussian factors for multi-view BA.

#![deny(missing_docs)]

use super::super::RelativePose;

/// Exact variable-block fixes for bundle adjustment.
///
/// Each slice is either empty (no blocks fixed) or has exactly the number of
/// corresponding blocks in the problem. `true` removes the whole block from
/// the optimization. Camera pose 0 is always fixed, independently of this
/// mask. Intrinsic blocks mean cameras for per-view calibration and groups for
/// grouped calibration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedVariableBlocks<'a> {
    /// Fixed six-dimensional camera-pose blocks.
    pub camera_poses: &'a [bool],
    /// Fixed three-dimensional Euclidean point blocks.
    pub points: &'a [bool],
    /// Fixed three-dimensional `(log focal, k1, k2)` intrinsic blocks.
    pub intrinsics: &'a [bool],
    /// Number of leading `(log focal, k1, k2)` parameters that are free in
    /// each intrinsic block: 1 refines a pinhole focal length, 2 adds `k1`,
    /// and 3 adds `k2`. The trailing parameters stay at their input values.
    /// Empty frees all three; otherwise one entry in `1..=3` per intrinsic
    /// block. Blocks fixed by `intrinsics` ignore their entry.
    pub intrinsic_dimensions: &'a [usize],
}

impl FixedVariableBlocks<'_> {
    /// No additional fixed variables.
    #[cfg(any(test, feature = "unstable-internals", feature = "parallel"))]
    pub const NONE: Self = Self {
        camera_poses: &[],
        points: &[],
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };

    /// Free parameters of radial intrinsic block `block`.
    pub(crate) fn intrinsic_dimension(self, block: usize) -> usize {
        if !self.intrinsics.is_empty() && self.intrinsics[block] {
            0
        } else {
            self.intrinsic_dimensions.get(block).copied().unwrap_or(3)
        }
    }
}

impl<'a> FixedVariableBlocks<'a> {
    #[cfg(any(feature = "faer", test))]
    pub(crate) fn canonical_with_camera_zero(self) -> Self {
        let mut canonical = self.canonical();
        if self.camera_poses.iter().any(|&fixed| fixed) {
            canonical.camera_poses = self.camera_poses;
        }
        canonical
    }

    /// Replace masks that fix nothing with empty masks.
    ///
    /// Camera pose 0 is always fixed, so a pose mask fixing at most camera 0,
    /// an all-false point or intrinsic mask, and intrinsic dimensions that
    /// free all three parameters describe the same problem as an empty mask.
    /// Sparse solvers select their parallel paths from the empty form, so
    /// checked masks are canonicalized before solving.
    #[cfg(any(feature = "faer", test))]
    pub(crate) fn canonical(self) -> Self {
        fn any_fixed(mask: &[bool]) -> bool {
            mask.iter().any(|&fixed| fixed)
        }
        Self {
            camera_poses: if self.camera_poses.len() > 1 && any_fixed(&self.camera_poses[1..]) {
                self.camera_poses
            } else {
                &[]
            },
            points: if any_fixed(self.points) {
                self.points
            } else {
                &[]
            },
            intrinsics: if any_fixed(self.intrinsics) {
                self.intrinsics
            } else {
                &[]
            },
            intrinsic_dimensions: if self
                .intrinsic_dimensions
                .iter()
                .any(|&dimension| dimension != 3)
            {
                self.intrinsic_dimensions
            } else {
                &[]
            },
        }
    }
}

/// A unary Gaussian camera-pose factor.
///
/// The residual is `(Log(R R_mean^T), t - t_mean)`. `sqrt_information` is a
/// finite square-root information matrix `L`; the prior contributes
/// `0.5 * ||L r||^2` to the objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosePrior {
    /// Camera-pose block receiving the prior.
    pub camera: usize,
    /// Prior mean.
    pub mean: RelativePose,
    /// Square-root information in rotation-radian/translation units.
    pub sqrt_information: [[f64; 6]; 6],
}

/// A unary Gaussian 3D-point factor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointPrior {
    /// Point block receiving the prior.
    pub point: usize,
    /// Prior mean in world coordinates.
    pub mean: [f64; 3],
    /// Finite square-root information matrix.
    pub sqrt_information: [[f64; 3]; 3],
}

/// A unary Gaussian radial-intrinsic factor.
///
/// Its residual is `(ln(f/f_mean), k1-k1_mean, k2-k2_mean)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialIntrinsicPrior {
    /// Per-view camera index or grouped intrinsic-block index.
    pub intrinsic: usize,
    /// Positive focal-length mean.
    pub focal: f64,
    /// Quadratic radial-distortion mean.
    pub k1: f64,
    /// Quartic radial-distortion mean.
    pub k2: f64,
    /// Finite square-root information matrix.
    pub sqrt_information: [[f64; 3]; 3],
}

/// Optional unary Gaussian factors.
///
/// Priors are separate from exact fixed blocks: fixed blocks
/// have no uncertainty, while priors retain a finite covariance and contribute
/// to the posterior information matrix. Each slice must be strictly ordered by
/// its block index; this permits a linear, allocation-free merge with the BA
/// state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BundleAdjustmentPriors<'a> {
    /// Camera-pose priors.
    pub poses: &'a [PosePrior],
    /// 3D-point priors.
    pub points: &'a [PointPrior],
    /// Radial-intrinsic priors; calibrated entry points reject a nonempty
    /// slice.
    pub intrinsics: &'a [RadialIntrinsicPrior],
}

impl BundleAdjustmentPriors<'_> {
    /// No prior factors.
    pub const NONE: Self = Self {
        poses: &[],
        points: &[],
        intrinsics: &[],
    };
}

#[cfg(test)]
mod tests {
    use super::FixedVariableBlocks;

    #[test]
    fn masks_that_fix_nothing_are_canonically_empty() {
        let anchor_only = [true, false, false];
        let no_points = [false; 4];
        let fixed = FixedVariableBlocks {
            camera_poses: &anchor_only,
            points: &no_points,
            intrinsics: &[false],
            intrinsic_dimensions: &[],
        };
        assert_eq!(fixed.canonical(), FixedVariableBlocks::NONE);
        assert_eq!(
            FixedVariableBlocks::NONE.canonical(),
            FixedVariableBlocks::NONE
        );
    }

    #[test]
    fn masks_that_fix_blocks_are_preserved() {
        let poses = [true, false, true];
        let points = [false, true];
        let intrinsics = [true];
        let fixed = FixedVariableBlocks {
            camera_poses: &poses,
            points: &points,
            intrinsics: &intrinsics,
            intrinsic_dimensions: &[],
        };
        assert_eq!(fixed.canonical(), fixed);
    }
}
