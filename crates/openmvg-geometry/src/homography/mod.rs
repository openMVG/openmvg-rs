// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Homographies: the projective map between two images of a plane, or of a camera that only
//! rotates.
//!
//! A homography `H` maps a pixel of one image to its match in the other, `x₂ ~ H x₁`. It holds
//! exactly for points on one plane, and for any scene when the camera turns without moving.
//! Start with [`estimate_homography`], which finds it among wrong matches.
//!
//! ## Available algorithms
//!
//! - DLT: Direct Linear Transform (4+ points).
//! - ACA: Affine-Core-Affine (exactly 4 points, closed form without an SVD),
//!   selected with [`HomographyMethod::Aca`].
//!
//! ## Normalized input
//!
//! The solvers fit on Hartley-normalized coordinates, which keeps the linear system well
//! conditioned: [`DltSolver::solve`] takes a
//! [`NormalizedCorrespondences`](crate::normalization::NormalizedCorrespondences) set built
//! from pixel correspondences, and returns the homography in pixel coordinates.
//! [`estimate_homography`] does this for you.
//!
//! ## Usage
//!
//! ```rust
//! use nalgebra::{Matrix3, Point2};
//! use openmvg_geometry::correspondence::PixelCorrespondence;
//! use openmvg_geometry::homography::{DltSolver, HomographyConfig};
//! use openmvg_geometry::normalization::NormalizedCorrespondences;
//!
//! // The corners of a square, and where a known homography sends them.
//! let truth = Matrix3::new(1.2, 0.1, 5.0, -0.05, 0.9, 3.0, 1e-3, 2e-3, 1.0);
//! let apply = |h: &Matrix3<f64>, p: Point2<f64>| {
//!     let q = h * p.to_homogeneous();
//!     Point2::new(q.x / q.z, q.y / q.z)
//! };
//! let pixels = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
//!     .map(|(x, y)| PixelCorrespondence::new(Point2::new(x, y), apply(&truth, Point2::new(x, y))));
//!
//! // Four exact matches determine the homography; DLT returns at most one.
//! let normalized = NormalizedCorrespondences::from_pixels(&pixels);
//! let estimate = &DltSolver::solve(&normalized, &HomographyConfig::default())?[0];
//! let check = Point2::new(50.0, 25.0);
//! assert!((apply(estimate.matrix(), check) - apply(&truth, check)).norm() < 1e-6);
//! # Ok::<(), openmvg_geometry::SolverError>(())
//! ```

pub mod aca;
mod dlt;
pub mod refinement;

use nalgebra::{Matrix3, Vector3};

use crate::normalization::CorrespondenceCoordinates;
use crate::{SolverError, SolverResult};

/// Reject opposite image orientations among the first four correspondences.
///
/// Compare signs of corresponding homogeneous triple determinants; zero areas
/// do not trigger this test. This is an
/// optional orientation-preservation policy; an orientation-reversing homography
/// can be finite and nonsingular. The test does not recover camera poses or prove
/// positive camera depth. Fewer than four correspondences pass.
fn passes_cheirality_test<D: CorrespondenceCoordinates>(correspondences: &[D]) -> bool {
    if correspondences.len() < 4 {
        return true; // Not enough points to test
    }

    // Convert points to homogeneous coordinates
    let x1: [Vector3<f64>; 4] = std::array::from_fn(|index| {
        let point = correspondences[index].coordinates().point1;
        Vector3::new(point.x, point.y, 1.0)
    });
    let x2: [Vector3<f64>; 4] = std::array::from_fn(|index| {
        let point = correspondences[index].coordinates().point2;
        Vector3::new(point.x, point.y, 1.0)
    });

    // Test 1: Cross product of points 0 and 1, check against points 2 and 3
    let p = x1[0].cross(&x1[1]);
    let q = x2[0].cross(&x2[1]);

    // Check point 2
    if (p.dot(&x1[2]) * q.dot(&x2[2])) < 0.0 {
        return false;
    }

    // Check point 3
    if (p.dot(&x1[3]) * q.dot(&x2[3])) < 0.0 {
        return false;
    }

    // Test 2: Cross product of points 2 and 3, check against points 0 and 1
    let p = x1[2].cross(&x1[3]);
    let q = x2[2].cross(&x2[3]);

    // Check point 0
    if (p.dot(&x1[0]) * q.dot(&x2[0])) < 0.0 {
        return false;
    }

    // Check point 1
    if (p.dot(&x1[1]) * q.dot(&x2[1])) < 0.0 {
        return false;
    }

    true
}

/// Homography `H` mapping `x2 ~ H * x1` (up to scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography(Matrix3<f64>);

impl Homography {
    /// Validate and wrap a pixel-space homography.
    ///
    /// Accepts exactly the matrices for which [`is_valid`](Self::is_valid)
    /// holds: finite entries and reciprocal condition number `σ3/σ1 > 1e-12`.
    /// The matrix is stored as given; use [`normalized`](Self::normalized)
    /// to fix its scale.
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidInput`] when an entry is non-finite or the
    /// matrix is numerically singular.
    pub fn try_new(matrix: Matrix3<f64>) -> SolverResult<Self> {
        let homography = Self(matrix);
        if homography.is_valid() {
            Ok(homography)
        } else {
            Err(SolverError::InvalidInput {
                reason: "homography must be finite and non-singular",
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

    /// Check whether the homography is finite and numerically non-singular.
    ///
    /// Homographies are projective, so the singularity test uses a
    /// singular-value ratio (`σ3/σ1 > 1e-12`) and is invariant to
    /// multiplication by any finite, non-zero scale.
    pub fn is_valid(&self) -> bool {
        if !self.0.iter().all(|value| value.is_finite()) {
            return false;
        }

        let matrix_scale = self.0.amax();
        if matrix_scale == 0.0 {
            return false;
        }
        let singular_values = (self.0 / matrix_scale).svd(false, false).singular_values;
        let scale = singular_values[0];
        if !scale.is_finite() || scale <= 0.0 {
            return false;
        }

        let reciprocal_condition = singular_values[2] / scale;
        reciprocal_condition.is_finite() && reciprocal_condition > 1e-12
    }

    /// Normalize the homography so that `H[2, 2] = 1` or `‖H‖ = 1`.
    pub fn normalized(&self) -> Self {
        normalized_homography_matrix(&self.0).map_or(*self, Homography)
    }
}

/// Choose a scale-safe projective gauge without changing the input matrix.
/// The h22 gauge retains the ordinary raw division when it is well scaled.
pub(crate) fn normalized_homography_matrix(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return None;
    }
    let scale = matrix.amax();
    if scale == 0.0 {
        return None;
    }
    let h22 = matrix[(2, 2)];
    if (h22 / scale).abs() > 1e-10 {
        return Some(matrix / h22);
    }
    // Retain the ordinary norm/division rounding whenever the raw norm is usable.
    let norm = matrix.norm();
    if norm.is_finite() && scale >= f64::MIN_POSITIVE.sqrt() {
        Some(matrix / norm)
    } else {
        let scaled = matrix / scale;
        Some(scaled / scaled.norm())
    }
}

impl AsRef<Matrix3<f64>> for Homography {
    fn as_ref(&self) -> &Matrix3<f64> {
        &self.0
    }
}

/// Algorithm selection for homography estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum HomographyMethod {
    /// Direct Linear Transform (4+ points), described on [`DltSolver`].
    #[default]
    Dlt,
    /// Affine-Core-Affine (exactly 4 points; larger samples fall back to DLT), described in
    /// [`aca`].
    Aca,
}

/// Configuration for homography estimation.
///
/// Start from [`HomographyConfig::default`] (DLT, no orientation check) and
/// adjust it with the by-value `with_*` setters, for example
/// `HomographyConfig::default().with_method(HomographyMethod::Aca)`.
///
/// Normalization is not configured here: the solver input
/// ([`NormalizedCorrespondences`](crate::normalization::NormalizedCorrespondences)) carries it.
#[derive(Debug, Clone)]
pub struct HomographyConfig {
    /// Algorithm to use.
    method: HomographyMethod,
    /// Whether opposite image orientations are rejected.
    check_cheirality: bool,
}

impl Default for HomographyConfig {
    fn default() -> Self {
        Self {
            method: HomographyMethod::Dlt,
            check_cheirality: false,
        }
    }
}

impl HomographyConfig {
    /// Select the homography algorithm (default: [`HomographyMethod::Dlt`]).
    #[must_use]
    pub fn with_method(mut self, method: HomographyMethod) -> Self {
        self.method = method;
        self
    }

    /// Set whether to reject opposite image orientations (default: `false`).
    ///
    /// The test looks at the first four correspondences only: it compares the orientation of
    /// four of their point triples in both images, ignores zero areas, and does not prove camera
    /// cheirality. Enable it only when the expected mapping preserves image orientation. Finite
    /// nonsingular homographies may reverse orientation; leave the test disabled for unrestricted
    /// homography estimation.
    #[must_use]
    pub fn with_check_cheirality(mut self, check: bool) -> Self {
        self.check_cheirality = check;
        self
    }

    /// Selected homography algorithm.
    #[must_use]
    pub fn method(&self) -> HomographyMethod {
        self.method
    }

    /// Whether opposite image orientations are rejected.
    #[must_use]
    pub fn check_cheirality(&self) -> bool {
        self.check_cheirality
    }
}

pub use dlt::DltSolver;
pub(crate) use dlt::UnnormalizedHomographySolver;

mod robust;

pub use robust::{estimate_homography, estimate_homography_guided};
