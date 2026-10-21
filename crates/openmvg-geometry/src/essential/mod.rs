// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The essential matrix: the relative pose of two calibrated cameras, from matched bearings.
//!
//! When both cameras are calibrated, matched points determine how the second camera is rotated
//! and in which direction it moved, though not how far: the scale stays unknown. Start with
//! [`estimate_essential`], which finds the matrix among wrong matches, then
//! [`EssentialMatrix::decompose_with_cheirality`] to read the rotation and the translation
//! direction that place the points in front of both cameras.
//!
//! The two estimators measure their residuals differently, so their thresholds differ:
//! [`estimate_essential`] uses the Sampson error in normalized image units (pixels divided by
//! the focal length), for perspective cameras, and [`estimate_essential_angular`] a first-order
//! angular error in radians, for wide-angle and spherical cameras.
//!
//! The decomposed pose `(R, t)` maps the first camera's frame to the second's,
//! `X₂ = R X₁ + t`, with `|t| = 1`. With the first camera at the origin, the second camera's
//! center is `C₂ = −Rᵀ t`, one baseline away, and the points triangulated from these two poses
//! are measured in baselines. As the world-to-camera poses that triangulation takes:
//!
//! ```rust
//! use nalgebra::{Point3, Rotation3};
//! use openmvg_core::model::{RelativePose, WorldToCameraPose};
//!
//! # fn poses(pose: &RelativePose) -> Result<(), Box<dyn std::error::Error>> {
//! let first = WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())?;
//! let center = Point3::from(pose.baseline_direction_in_source().into_inner()); // −Rᵀ t
//! let rotation = *pose.rotation_source_to_target();
//! let second = WorldToCameraPose::from_rotation_center(rotation, center)?;
//! # Ok(())
//! # }
//! ```
//!
//! The `relative_pose` example recovers the pose from matches, and the `triangulate_and_resect`
//! example continues: it triangulates the points and finds the pose of a third camera.
//!
//! ## Available algorithms
//!
//! | Solver | Bearings | Role |
//! |---|---|---|
//! | [`FivePointSolver`] | arbitrary central/spherical | minimal solver inside both estimators |
//!
//! Both estimators refit the model on its inliers with a linear eight-point fit, and keep the
//! refit only when it scores better.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                    Essential matrix solver family                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: calibrated bearing correspondences
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: E5 minimal hypotheses or E8 non-minimal linear fit           │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Enforce or validate the essential singular values            │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Decompose into four poses; select by bearing cheirality      │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use openmvg_geometry::essential::{EssentialMatrix, FivePointSolver};
//! use openmvg_geometry::SolverResult;
//! use openmvg_geometry::correspondence::BearingCorrespondence;
//!
//! /// Cheirality-checked candidates from exactly five bearing correspondences.
//! fn estimate(correspondences: &[BearingCorrespondence; 5]) -> SolverResult<Vec<EssentialMatrix>> {
//!     FivePointSolver::solve(correspondences)
//! }
//! ```
//!
//! ## Pose decomposition
//!
//! Essential matrices can be decomposed into rotation and translation:
//!
//! ```rust
//! use openmvg_geometry::essential::EssentialMatrix;
//!
//! fn candidate_poses(essential: &EssentialMatrix) -> openmvg_geometry::SolverResult<()> {
//!     let poses = essential.decompose()?;
//!     assert_eq!(poses.len(), 4);
//!     Ok(())
//! }
//! ```

mod eight_point;
mod five_point;
mod five_point_kernel;
mod pose_decomposition;
mod robust;

use nalgebra::Matrix3;

use crate::{SolverError, SolverResult};

pub(crate) use eight_point::{EightPointSolver, PinholeEssentialSolver};
pub use five_point::FivePointSolver;
pub use robust::{
    estimate_essential, estimate_essential_angular, estimate_essential_angular_guided,
    estimate_essential_guided,
};

/// Internal evidence surface used only by integration tests (allocation
/// contract, five-point reference fixture and the linear eight-point property
/// tests).
#[cfg(feature = "unstable-test-support")]
#[doc(hidden)]
pub mod test_support {
    pub use super::eight_point::EightPointSolver;
    pub use super::five_point::FivePointSolutions;
    pub use super::five_point_kernel::{
        Pose as FivePointPose, Poses as FivePointPoses, path as five_point_path,
    };
    use crate::correspondence::BearingCorrespondence;

    /// Solve one exact five-point sample without allocating the result container.
    pub fn solve_five_point_minimal(data: &[BearingCorrespondence; 5]) -> FivePointSolutions {
        super::FivePointSolver::solve_minimal(data)
    }

    /// Run the five-point kernel on raw 3-vectors and return its code-path
    /// flags (see [`five_point_path`]).
    pub fn solve_five_point_kernel(
        bearings1: &[[f64; 3]; 5],
        bearings2: &[[f64; 3]; 5],
        out: &mut FivePointPoses,
    ) -> u32 {
        super::five_point_kernel::solve(bearings1, bearings2, out)
    }
}

/// Essential matrix `E` of the epipolar constraint `b₂ᵀ E b₁ = 0` on bearings.
///
/// `E = [t]_x * R`, where `[t]_x` is the skew-symmetric matrix of the
/// translation and `R` the rotation, so E has two equal singular values and one zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssentialMatrix(Matrix3<f64>);

impl EssentialMatrix {
    /// Validate and wrap an essential matrix.
    ///
    /// Accepts exactly the matrices for which [`is_valid`](Self::is_valid)
    /// holds: finite, non-zero, and singular values `(s, s, 0)` within that
    /// method's projective tolerance. The matrix is stored as given, without
    /// rescaling or projection onto the essential manifold.
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidInput`] when an entry is non-finite, the matrix
    /// is zero, or its singular values are not `(s, s, 0)`.
    pub fn try_new(matrix: Matrix3<f64>) -> SolverResult<Self> {
        let essential = Self(matrix);
        if essential.is_valid() {
            Ok(essential)
        } else {
            Err(SolverError::InvalidInput {
                reason: "essential matrix must be finite, non-zero, and have singular values (s, s, 0)",
            })
        }
    }

    /// Wrap a matrix without validation (internal solver outputs).
    pub(crate) fn new(matrix: Matrix3<f64>) -> Self {
        Self(matrix)
    }

    /// Return the underlying 3x3 matrix.
    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.0
    }

    /// Check whether the matrix satisfies the essential matrix constraints.
    ///
    /// An essential matrix must be finite and non-zero, with:
    /// - Rank 2 (determinant = 0)
    /// - Two equal singular values (σ1 = σ2)
    ///
    /// Both are tested together, projectively, through Demazure's cubic
    /// constraint `2 E Eᵀ E - trace(E Eᵀ) E = 0`: its Frobenius norm must be
    /// below `0.11 ‖E‖³`.
    pub fn is_valid(&self) -> bool {
        if !self.0.iter().all(|value| value.is_finite()) {
            return false;
        }

        let scale = self
            .0
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        if scale == 0.0 {
            return false;
        }
        let normalized = self.0 / scale;
        let scale_squared = normalized.norm_squared();

        // Demazure's cubic constraint is zero exactly for matrices with
        // singular values (s, s, 0). Compare its squared norm to ||E||^6 so
        // the check remains projective without paying for a 3x3 SVD.
        let gram = normalized * normalized.transpose();
        let two = 2.0;
        let residual = gram * normalized * two - normalized * gram.trace();
        let tolerance = 0.11;
        residual.norm_squared()
            < tolerance * tolerance * scale_squared * scale_squared * scale_squared
    }
}

impl AsRef<Matrix3<f64>> for EssentialMatrix {
    fn as_ref(&self) -> &Matrix3<f64> {
        &self.0
    }
}

#[cfg(test)]
mod config_tests {
    use super::EssentialMatrix;
    use nalgebra::{Matrix3, Rotation3, Vector3};

    #[test]
    fn essential_validity_is_scale_invariant() {
        let rotation = Rotation3::from_euler_angles(0.2, -0.1, 0.05);
        let translation = Vector3::new(1.0, 2.0, -0.5).normalize();
        let cross = Matrix3::new(
            0.0,
            -translation.z,
            translation.y,
            translation.z,
            0.0,
            -translation.x,
            -translation.y,
            translation.x,
            0.0,
        );
        let matrix = cross * rotation.matrix();

        assert!(EssentialMatrix::new(matrix).is_valid());
        assert!(EssentialMatrix::new(matrix * 1e-12).is_valid());
        assert!(EssentialMatrix::new(matrix * 1e12).is_valid());
    }

    #[test]
    fn essential_validation_preserves_extreme_projective_scales() {
        for scale in [1e-150, 1e-100, 1e-12, 1.0, 1e12, 1e100, 1e150] {
            for sign in [-1.0, 1.0] {
                let matrix = Matrix3::from_diagonal(&Vector3::new(sign * scale, sign * scale, 0.0));
                let model = EssentialMatrix::try_new(matrix).unwrap();
                assert_eq!(model.matrix(), &matrix);
                assert!(model.is_valid());
                for spectrum in [
                    Vector3::new(scale, 0.0, 0.0),
                    Vector3::new(scale, scale, scale),
                    Vector3::new(scale, 0.8 * scale, 0.0),
                ] {
                    assert!(EssentialMatrix::try_new(Matrix3::from_diagonal(&spectrum)).is_err());
                }
            }
        }
    }

    #[test]
    fn essential_validity_rejects_wrong_spectra() {
        assert!(!EssentialMatrix::new(Matrix3::<f64>::zeros()).is_valid());
        assert!(!EssentialMatrix::new(Matrix3::<f64>::identity()).is_valid());
        assert!(
            !EssentialMatrix::new(Matrix3::from_diagonal(&Vector3::new(1.0, 0.0, 0.0,))).is_valid()
        );
        assert!(
            !EssentialMatrix::new(Matrix3::from_diagonal(&Vector3::new(1.0, 0.8, 0.0,))).is_valid()
        );
    }

    #[test]
    fn try_new_accepts_exactly_the_valid_matrices() {
        let valid = Matrix3::from_diagonal(&Vector3::new(2.0, 2.0, 0.0));
        let model = EssentialMatrix::try_new(valid).unwrap();
        assert_eq!(model.as_ref(), &valid);

        let mut non_finite = valid;
        non_finite[(0, 1)] = f64::NAN;
        for invalid in [Matrix3::identity(), Matrix3::zeros(), non_finite] {
            assert!(!EssentialMatrix::new(invalid).is_valid());
            assert!(matches!(
                EssentialMatrix::try_new(invalid),
                Err(crate::SolverError::InvalidInput { .. })
            ));
        }
    }
}
