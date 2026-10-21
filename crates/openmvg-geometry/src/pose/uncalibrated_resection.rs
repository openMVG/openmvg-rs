// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # 6-point uncalibrated DLT resection solver
//!
//! Its teaching docs are on [`UncalibratedResectionSolver`], where docs.rs shows them.

use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix3x4, Point2, Point3, SMatrix, SVector, SymmetricEigen, Vector3,
};

use crate::normalization::{PointNormalization, WorldNormalization};
use crate::{SolverError, SolverResult};

use super::{
    PixelToPoint, SolverPose, SolverPoseExt, WorldToCameraPose, camera_pose_from_world_to_camera,
    world_to_camera_from_matrix,
};
use crate::shared::svd_utils::solve_homogeneous_system;
use crate::solver_traits::{SingleSolutionSolver, Solver};

/// Pinhole intrinsics matrix K.
///
/// Upper triangular 3x3 matrix:
/// ```text
/// K = | fx  s  cx |
///     | 0  fy  cy |
///     | 0   0   1 |
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeIntrinsics {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    skew: f64,
}

impl PinholeIntrinsics {
    /// Focal length in x direction (pixels).
    #[must_use]
    pub fn focal_x(&self) -> f64 {
        self.fx
    }

    /// Focal length in y direction (pixels).
    #[must_use]
    pub fn focal_y(&self) -> f64 {
        self.fy
    }

    /// Principal point x coordinate (pixels).
    #[must_use]
    pub fn principal_x(&self) -> f64 {
        self.cx
    }

    /// Principal point y coordinate (pixels).
    #[must_use]
    pub fn principal_y(&self) -> f64 {
        self.cy
    }

    /// Skew coefficient (usually 0).
    #[must_use]
    pub fn skew(&self) -> f64 {
        self.skew
    }

    /// Validate and construct camera intrinsics.
    pub fn try_new(
        focal_x: f64,
        focal_y: f64,
        principal_x: f64,
        principal_y: f64,
        skew: f64,
    ) -> SolverResult<Self> {
        let intrinsics = Self {
            fx: focal_x,
            fy: focal_y,
            cx: principal_x,
            cy: principal_y,
            skew,
        };
        if intrinsics.is_valid() {
            Ok(intrinsics)
        } else {
            Err(SolverError::InvalidInput {
                reason: "intrinsics require finite values and positive focal lengths",
            })
        }
    }

    /// Validate square-pixel intrinsics with zero skew.
    pub fn try_from_focal_and_principal(
        focal: f64,
        principal_x: f64,
        principal_y: f64,
    ) -> SolverResult<Self> {
        Self::try_new(focal, focal, principal_x, principal_y, 0.0)
    }

    /// Create identity intrinsics (normalized coordinates).
    pub fn identity() -> Self {
        Self {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
            skew: 0.0,
        }
    }

    /// Convert to 3x3 calibration matrix K.
    pub fn to_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, self.skew, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        )
    }

    /// Create from upper triangular 3x3 matrix without validation.
    ///
    /// Normalizes so `K[2, 2] = 1`.
    pub(crate) fn from_matrix(k: &Matrix3<f64>) -> Self {
        let scale = k[(2, 2)];
        Self {
            fx: k[(0, 0)] / scale,
            fy: k[(1, 1)] / scale,
            cx: k[(0, 2)] / scale,
            cy: k[(1, 2)] / scale,
            skew: k[(0, 1)] / scale,
        }
    }

    /// Validate and normalize an upper-triangular calibration matrix.
    pub fn try_from_matrix(k: &Matrix3<f64>) -> SolverResult<Self> {
        let matrix_scale = k
            .iter()
            .copied()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let tolerance = matrix_scale * f64::EPSILON * 64.0;
        if !k.iter().all(|value| value.is_finite())
            || !matrix_scale.is_finite()
            || matrix_scale <= 0.0
            || k[(2, 2)].abs() <= tolerance
            || k[(1, 0)].abs() > tolerance
            || k[(2, 0)].abs() > tolerance
            || k[(2, 1)].abs() > tolerance
        {
            return Err(SolverError::InvalidInput {
                reason: "calibration matrix must be finite, upper triangular, and non-singular",
            });
        }
        let scale = k[(2, 2)];
        Self::try_new(
            k[(0, 0)] / scale,
            k[(1, 1)] / scale,
            k[(0, 2)] / scale,
            k[(1, 2)] / scale,
            k[(0, 1)] / scale,
        )
    }

    /// Project a 3D point in camera coordinates to pixel coordinates.
    pub fn project(&self, point: &Point3<f64>) -> Option<Point2<f64>> {
        if point.z <= 0.0 || !point.coords.iter().all(|value| value.is_finite()) || !self.is_valid()
        {
            return None;
        }
        let x = point.x / point.z;
        let y = point.y / point.z;
        let projected = Point2::new(self.fx * x + self.skew * y + self.cx, self.fy * y + self.cy);
        projected
            .coords
            .iter()
            .all(|value| value.is_finite())
            .then_some(projected)
    }

    /// Check whether the intrinsics are valid (positive focal lengths).
    pub fn is_valid(&self) -> bool {
        self.fx > 0.0
            && self.fy > 0.0
            && [self.fx, self.fy, self.cx, self.cy, self.skew]
                .iter()
                .all(|value| value.is_finite())
    }
}

/// Configuration for the uncalibrated resection solver.
#[derive(Debug, Clone)]
pub struct UncalibratedResectionConfig {
    check_cheirality: bool,
    enforce_positive_focal: bool,
}

config_accessors!(UncalibratedResectionConfig {
    /// Whether to verify that all points are in front of the camera.
    copy check_cheirality, with_check_cheirality: bool;
    /// Whether to enforce positive focal lengths.
    copy enforce_positive_focal, with_enforce_positive_focal: bool;
});

impl Default for UncalibratedResectionConfig {
    fn default() -> Self {
        Self {
            check_cheirality: true,
            enforce_positive_focal: true,
        }
    }
}

/// Result of uncalibrated resection: intrinsics and pose.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CalibratedPose {
    /// Camera intrinsics K.
    pub intrinsics: PinholeIntrinsics,
    /// World-to-camera pose.
    pub pose: WorldToCameraPose,
}

impl CalibratedPose {
    /// Create a calibrated pose.
    pub fn new(intrinsics: PinholeIntrinsics, pose: WorldToCameraPose) -> Self {
        Self { intrinsics, pose }
    }

    /// Return the full 3x4 projection matrix P = K[R|t].
    pub fn projection_matrix(&self) -> Matrix3x4<f64> {
        let k = self.intrinsics.to_matrix();
        let r = self.pose.rotation();
        let t = self.pose.translation();

        // P = K * [R | t]
        let mut rt = Matrix3x4::zeros();
        for i in 0..3 {
            for j in 0..3 {
                rt[(i, j)] = r[(i, j)];
            }
            rt[(i, 3)] = t[i];
        }
        k * rt
    }

    /// Project a world point to pixel coordinates.
    pub fn project(&self, world_point: &Point3<f64>) -> Option<Point2<f64>> {
        let cam_point = self.pose.project_to_camera(world_point);
        self.intrinsics.project(&cam_point)
    }
}

/// The model [`UncalibratedResectionSolver`] produces through the generic
/// [`Solver`](crate::robust::Solver) traits, for example from
/// [`ClassicConfig::estimate`](crate::robust::classic::ClassicConfig::estimate).
///
/// It keeps the pose in the solver's internal form so that scoring stays cheap;
/// convert it with [`Self::into_calibrated_pose`]. The inherent
/// [`UncalibratedResectionSolver::solve`] returns [`CalibratedPose`] directly.
#[derive(Debug, Clone)]
pub struct SolverCalibratedPose {
    intrinsics: PinholeIntrinsics,
    pose: SolverPose<f64>,
}

impl SolverCalibratedPose {
    fn new(intrinsics: PinholeIntrinsics, pose: SolverPose<f64>) -> Self {
        Self { intrinsics, pose }
    }

    /// Convert to the public [`CalibratedPose`].
    ///
    /// # Errors
    ///
    /// [`SolverError::Numerical`] when the rotation is not a proper rotation.
    pub fn into_calibrated_pose(self) -> SolverResult<CalibratedPose> {
        let pose = camera_pose_from_world_to_camera(&self.pose).map_err(|error| {
            SolverError::Numerical {
                reason: error.to_string(),
            }
        })?;
        Ok(CalibratedPose::new(self.intrinsics, pose))
    }

    fn project(&self, world_point: &Point3<f64>) -> Option<Point2<f64>> {
        self.intrinsics
            .project(&self.pose.transform_point(world_point))
    }
}

/// Uncalibrated DLT resection solver.
///
/// Estimates both intrinsics K and pose [R|t] from pixel correspondences.
///
/// # How it works
///
/// Estimates both camera intrinsics K and pose [R|t] from 6+ 3D-2D pixel
/// correspondences using DLT followed by RQ decomposition.
///
/// ## Algorithm diagram
///
/// ```text
/// ╔═══════════════════════════════════════════════════════════════════════════╗
/// ║                 Uncalibrated DLT resection solver                         ║
/// ║                    (Unknown intrinsics K)                                 ║
/// ╠═══════════════════════════════════════════════════════════════════════════╣
/// ║                                                                           ║
/// ║  Input: n ≥ 6 correspondences                                             ║
/// ║         World points X_i = (X,Y,Z) ∈ ℝ³                                   ║
/// ║         Pixel observations p_i = (u,v) in pixels                          ║
/// ║                                                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 1: Hartley normalization                                       │  ║
/// ║  │                                                                     │  ║
/// ║  │  2D Points:                                                         │  ║
/// ║  │    μ_2D = mean(p_i), σ_2D = std(p_i)                                │  ║
/// ║  │    p̃_i = T_2D · p_i  where T_2D scales to √2                        │  ║
/// ║  │                                                                     │  ║
/// ║  │  3D Points:                                                         │  ║
/// ║  │    μ_3D = mean(X_i), σ_3D = std(X_i)                                │  ║
/// ║  │    X̃_i = T_3D · X_i  where T_3D scales to √3                        │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 2: Build constraint matrix (2n × 12)                           │  ║
/// ║  │                                                                     │  ║
/// ║  │  For each normalized correspondence (X̃_i, p̃_i):                     │  ║
/// ║  │                                                                     │  ║
/// ║  │    Row 1: [-X̃ -Ỹ -Z̃ -1 | 0 0 0 0 | ũ·X̃ ũ·Ỹ ũ·Z̃ ũ]                   │  ║
/// ║  │    Row 2: [0 0 0 0 | -X̃ -Ỹ -Z̃ -1 | ṽ·X̃ ṽ·Ỹ ṽ·Z̃ ṽ]                   │  ║
/// ║  │                                                                     │  ║
/// ║  │  A(2n×12) · p = 0                                                   │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 3: Solve for null space                                        │  ║
/// ║  │                                                                     │  ║
/// ║  │  SVD(A) = U·Σ·Vᵀ  or  Eigen(AᵀA)                                    │  ║
/// ║  │  p = smallest singular/eigenvector                                  │  ║
/// ║  │                                                                     │  ║
/// ║  │  Reshape p to P̃_norm (3×4)                                          │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 4: Denormalize projection matrix                               │  ║
/// ║  │                                                                     │  ║
/// ║  │  Undo Hartley normalization:                                        │  ║
/// ║  │                                                                     │  ║
/// ║  │    P = T_2D⁻¹ · P̃_norm · T_3D                                       │  ║
/// ║  │                                                                     │  ║
/// ║  │  P then projects world points to pixel coordinates                  │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 5: RQ decomposition                                            │  ║
/// ║  │                                                                     │  ║
/// ║  │  P = [M | p₄]  where M is 3×3                                       │  ║
/// ║  │                                                                     │  ║
/// ║  │  RQ(M) → M = K · R                                                  │  ║
/// ║  │    where K = upper triangular (intrinsics)                          │  ║
/// ║  │          R = orthogonal (rotation)                                  │  ║
/// ║  │                                                                     │  ║
/// ║  │  Method: QR on M^T, then transpose                                  │  ║
/// ║  │    QR(Mᵀ) = Q · Rᵀ  →  M = R · Qᵀ  but want K·R                     │  ║
/// ║  │    Rearrange with sign corrections for K upper triangular           │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 6: Normalize K and extract t                                   │  ║
/// ║  │                                                                     │  ║
/// ║  │  Ensure K[2,2] = 1 and positive diagonal:                           │  ║
/// ║  │    K = K / K[2,2]                                                   │  ║
/// ║  │    If K[0,0] < 0: flip sign of K[:,0] and R[0,:]                    │  ║
/// ║  │    If K[1,1] < 0: flip sign of K[:,1] and R[1,:]                    │  ║
/// ║  │                                                                     │  ║
/// ║  │  Extract translation:                                               │  ║
/// ║  │    t = K⁻¹ · p₄                                                     │  ║
/// ║  │                                                                     │  ║
/// ║  │  Extract intrinsics:                                                │  ║
/// ║  │    fx = K[0,0], fy = K[1,1]                                         │  ║
/// ║  │    cx = K[0,2], cy = K[1,2]                                         │  ║
/// ║  │    skew = K[0,1]                                                    │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 7: Cheirality check                                            │  ║
/// ║  │                                                                     │  ║
/// ║  │  For each point X_i:                                                │  ║
/// ║  │    depth = R · X_i + t  (z-component)                               │  ║
/// ║  │    If depth ≤ 0: Point is behind camera → reject                    │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                                                                           ║
/// ║  Output: Camera intrinsics K = (fx, fy, cx, cy, skew)                     ║
/// ║          Camera pose (R, t)                                               ║
/// ║                                                                           ║
/// ╠═══════════════════════════════════════════════════════════════════════════╣
/// ║  Decomposition diagram:                                                   ║
/// ║                                                                           ║
/// ║    P (3×4)  =  [ K (3×3) ]  ×  [ R (3×3) | t (3×1) ]                      ║
/// ║    ├──────┤    ├─────────┤     ├──────────────────┤                       ║
/// ║    Known       Intrinsics      Extrinsics (pose)                          ║
/// ║    (from DLT)  (fx,fy,cx,cy)   (rotation, translation)                    ║
/// ║                                                                           ║
/// ║    K = | fx  s  cx |    R ∈ SO(3)    t ∈ ℝ³                               ║
/// ║        | 0  fy  cy |                                                      ║
/// ║        | 0   0   1 |                                                      ║
/// ╚═══════════════════════════════════════════════════════════════════════════╝
/// ```
///
/// # References
///
/// R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*,
/// Cambridge University Press, 2nd ed., 2004, Chapter 7.
#[derive(Debug, Clone, Default)]
pub struct UncalibratedResectionSolver;

impl UncalibratedResectionSolver {
    /// Return the minimum correspondence count.
    #[must_use]
    pub fn min_sample_size() -> usize {
        <Self as Solver>::min_sample_size()
    }

    /// Evaluate the pixel residual for one correspondence.
    pub fn residual(model: &CalibratedPose, datum: &PixelToPoint) -> f64 {
        match model.project(&datum.world_point) {
            Some(projected) => (projected - datum.pixel).norm(),
            None => f64::INFINITY,
        }
    }

    /// Validate the estimated intrinsics and pose.
    #[must_use]
    pub fn is_valid(model: &CalibratedPose) -> bool {
        model.intrinsics.is_valid() && model.pose.validate().is_ok()
    }

    /// Solve from a checked correspondence slice.
    pub fn solve(
        data: &[PixelToPoint],
        config: &UncalibratedResectionConfig,
    ) -> SolverResult<Vec<CalibratedPose>> {
        Ok(<Self as SingleSolutionSolver>::solve_single(data, config)?
            .map(SolverCalibratedPose::into_calibrated_pose)
            .transpose()?
            .into_iter()
            .collect())
    }

    /// Build the constraint matrix for DLT from normalized coordinates.
    fn build_constraint_matrix(
        pixels: &[Point2<f64>],
        world_points: &[Point3<f64>],
    ) -> DMatrix<f64> {
        let n = pixels.len();
        let mut a = DMatrix::zeros(2 * n, 12);

        for i in 0..n {
            Self::fill_constraint_rows(
                &mut |row, col, val| a[(row, col)] = val,
                i,
                &pixels[i],
                &world_points[i],
            );
        }

        a
    }

    /// Build the 12x12 DLT constraint matrix on the stack for the 6-point
    /// minimal case used by RANSAC.
    fn build_constraint_matrix_6point(
        pixels: &[Point2<f64>],
        world_points: &[Point3<f64>],
    ) -> SMatrix<f64, 12, 12> {
        debug_assert_eq!(pixels.len(), 6);
        debug_assert_eq!(world_points.len(), 6);
        let mut a: SMatrix<f64, 12, 12> = SMatrix::zeros();

        for i in 0..6 {
            Self::fill_constraint_rows(
                &mut |row, col, val| a[(row, col)] = val,
                i,
                &pixels[i],
                &world_points[i],
            );
        }

        a
    }

    /// Fill constraint rows for a single correspondence.
    #[inline]
    fn fill_constraint_rows<F: FnMut(usize, usize, f64)>(
        setter: &mut F,
        i: usize,
        pixel: &Point2<f64>,
        world_point: &Point3<f64>,
    ) {
        let u = pixel.x;
        let v = pixel.y;

        let x = world_point.x;
        let y = world_point.y;
        let z = world_point.z;
        let w = 1.0;

        // First row: [-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u]
        let row1_idx = 2 * i;
        setter(row1_idx, 0, -x);
        setter(row1_idx, 1, -y);
        setter(row1_idx, 2, -z);
        setter(row1_idx, 3, -w);
        // columns 4-7 are already zero
        setter(row1_idx, 8, u * x);
        setter(row1_idx, 9, u * y);
        setter(row1_idx, 10, u * z);
        setter(row1_idx, 11, u * w);

        // Second row: [0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v]
        let row2_idx = 2 * i + 1;
        // columns 0-3 are already zero
        setter(row2_idx, 4, -x);
        setter(row2_idx, 5, -y);
        setter(row2_idx, 6, -z);
        setter(row2_idx, 7, -w);
        setter(row2_idx, 8, v * x);
        setter(row2_idx, 9, v * y);
        setter(row2_idx, 10, v * z);
        setter(row2_idx, 11, v * w);
    }

    /// Solve for nullspace via SVD.
    fn solve_nullspace(a: &DMatrix<f64>) -> SolverResult<DVector<f64>> {
        solve_homogeneous_system(a)
    }

    /// Solve for the nullspace via a stack-allocated 12x12 eigendecomposition,
    /// used for the RANSAC minimal case.
    fn solve_nullspace_6point(a: &SMatrix<f64, 12, 12>) -> SolverResult<SVector<f64, 12>> {
        let ata = a.transpose() * a;
        let eigen = SymmetricEigen::new(ata);
        let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        order.sort_by(|&left, &right| {
            let left = eigen.eigenvalues[left].abs();
            let right = eigen.eigenvalues[right].abs();
            left.total_cmp(&right)
        });
        let spectrum: [f64; 12] = std::array::from_fn(|i| eigen.eigenvalues[order[i]].abs());
        if spectrum.iter().any(|value| !value.is_finite()) || spectrum[11] <= 0.0 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "uncalibrated resection has zero or non-finite numerical rank".to_string(),
            });
        }
        let tolerance = spectrum[11] * f64::EPSILON * 12.0 * 64.0;
        if spectrum[1] <= tolerance || spectrum[1] - spectrum[0] <= tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "uncalibrated resection has an ambiguous null space".to_string(),
            });
        }
        let solution: SVector<f64, 12> = eigen.eigenvectors.column(order[0]).into();
        if !solution.iter().all(|value| value.is_finite()) {
            return Err(SolverError::SvdFailed);
        }
        Ok(solution)
    }

    /// Reshape 12-element vector to 3x4 projection matrix (row-major).
    fn reshape_to_projection(p: &DVector<f64>) -> Matrix3x4<f64> {
        Matrix3x4::new(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
        )
    }

    /// Reshape 12-element SVector to 3x4 projection matrix (row-major).
    #[inline]
    fn reshape_to_projection_fixed(p: &SVector<f64, 12>) -> Matrix3x4<f64> {
        Matrix3x4::new(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
        )
    }

    /// RQ decomposition of a 3x3 matrix.
    ///
    /// Decomposes M = K * R where K is upper triangular and R is orthogonal.
    /// Uses the approach: reverse rows/cols, do QR, reverse back.
    fn rq_decomposition(m: &Matrix3<f64>) -> SolverResult<(Matrix3<f64>, Matrix3<f64>)> {
        // Permutation matrix that reverses rows/columns
        let p = Matrix3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);

        // Reverse M: P * M * P
        let m_rev = p * m * p;

        // QR decomposition of reversed matrix transpose
        let qr = m_rev.transpose().qr();
        let q_rev = qr.q();
        let r_rev = qr.r();

        // Reverse back: K = P * R^T * P, R = P * Q^T * P
        let k = p * r_rev.transpose() * p;
        let r = p * q_rev.transpose() * p;

        // Ensure K has positive diagonal entries
        let mut d = Matrix3::identity();
        if k[(0, 0)] < 0.0 {
            d[(0, 0)] = -1.0;
        }
        if k[(1, 1)] < 0.0 {
            d[(1, 1)] = -1.0;
        }
        if k[(2, 2)] < 0.0 {
            d[(2, 2)] = -1.0;
        }

        let k = k * d;
        let r = d * r;

        // Normalize K so K[2,2] = 1
        let scale = k[(2, 2)];
        if scale.abs() < 1e-10 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "K[2,2] is near zero".to_string(),
            });
        }
        let k = k / scale;

        // Flipping R alone would break M = K R. The caller evaluates both
        // projective signs of P, so reject this branch and let the opposite
        // sign provide the proper-rotation decomposition.
        if r.determinant() < 0.0 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "RQ decomposition produced an improper rotation branch".to_string(),
            });
        }

        if !k.iter().all(|value| value.is_finite()) || !r.iter().all(|value| value.is_finite()) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "RQ decomposition produced non-finite coefficients".to_string(),
            });
        }
        Ok((k, r))
    }

    /// Decompose projection matrix P into K, R, t using RQ decomposition.
    fn decompose_projection(
        p: &Matrix3x4<f64>,
        config: &UncalibratedResectionConfig,
    ) -> SolverResult<SolverCalibratedPose> {
        // First normalize P so that ||P[2,:3]|| = 1 (the third row of M)
        // This ensures consistent scale for K and t
        let m_row3 = Vector3::new(p[(2, 0)], p[(2, 1)], p[(2, 2)]);
        let m_row3_norm = m_row3.norm();

        if m_row3_norm < 1e-10 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "P[2,:3] is near zero".to_string(),
            });
        }

        // Normalize the entire projection matrix
        let p_normalized = p / m_row3_norm;

        // Extract M = P[:,:3] and p4 = P[:,3]
        let m = p_normalized.fixed_view::<3, 3>(0, 0).into_owned();
        let p4 = Vector3::new(
            p_normalized[(0, 3)],
            p_normalized[(1, 3)],
            p_normalized[(2, 3)],
        );

        // RQ decomposition: M = K * R
        let (k, r) = Self::rq_decomposition(&m)?;

        // Compute translation: t = K^-1 * p4
        let k_inv = k
            .try_inverse()
            .ok_or(SolverError::DegenerateConfiguration {
                reason: "K is not invertible".to_string(),
            })?;
        let t = k_inv * p4;

        // Extract intrinsics
        let intrinsics = PinholeIntrinsics::from_matrix(&k);

        // Check for valid intrinsics
        if config.enforce_positive_focal && !intrinsics.is_valid() {
            return Err(SolverError::DegenerateConfiguration {
                reason: "invalid intrinsics: focal lengths must be positive".to_string(),
            });
        }

        // Create pose
        let pose = world_to_camera_from_matrix(&r, t);

        Ok(SolverCalibratedPose::new(intrinsics, pose))
    }

    /// Check that all points are in front of the camera (positive depth).
    fn check_cheirality(result: &SolverCalibratedPose, correspondences: &[PixelToPoint]) -> bool {
        correspondences
            .iter()
            .all(|corr| result.pose.depth(&corr.world_point) > 0.0)
    }
}

impl Solver for UncalibratedResectionSolver {
    type Datum = PixelToPoint;
    type Model = SolverCalibratedPose;
    type Config = UncalibratedResectionConfig;
    type Unit = openmvg_core::Pixels;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        match model.project(&datum.world_point) {
            Some(projected) => {
                let dx = projected.x - datum.pixel.x;
                let dy = projected.y - datum.pixel.y;
                (dx * dx + dy * dy).sqrt()
            }
            None => f64::INFINITY,
        }
    }

    fn min_sample_size() -> usize {
        6
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.intrinsics.is_valid()
    }
}

impl SingleSolutionSolver for UncalibratedResectionSolver {
    fn solve_single(
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        if data.len() < 6 {
            return Err(SolverError::InsufficientData {
                required: 6,
                provided: data.len(),
            });
        }
        if data.iter().any(|corr| {
            !corr
                .world_point
                .coords
                .iter()
                .all(|value| value.is_finite())
                || !corr.pixel.coords.iter().all(|value| value.is_finite())
        }) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "uncalibrated resection input contains non-finite coordinates".to_string(),
            });
        }

        // Step 1: Normalize points for numerical stability
        let pixels: Vec<Point2<f64>> = data.iter().map(|c| c.pixel).collect();
        let pixel_normalization = PointNormalization::from_points(&pixels);
        let world_normalization = WorldNormalization::from_points(
            data.iter().map(|c| c.world_point),
        )
        .ok_or_else(|| SolverError::DegenerateConfiguration {
            reason: "uncalibrated resection world points have no spatial extent".to_string(),
        })?;
        let pixels_norm: Vec<Point2<f64>> = pixels
            .iter()
            .map(|p| pixel_normalization.normalize_point(p))
            .collect();
        let world_norm: Vec<Point3<f64>> = data
            .iter()
            .map(|c| world_normalization.normalize_point(&c.world_point))
            .collect();
        let t2 = pixel_normalization.transform();
        let t3 = world_normalization.transform();

        // Step 2-4: Build constraint matrix and solve for projection
        // Use stack-allocated version for minimal 6-point case (RANSAC hot path)
        let p_norm = if data.len() == 6 {
            let a = Self::build_constraint_matrix_6point(&pixels_norm, &world_norm);
            let p_vec = Self::solve_nullspace_6point(&a)?;
            Self::reshape_to_projection_fixed(&p_vec)
        } else {
            let a = Self::build_constraint_matrix(&pixels_norm, &world_norm);
            let p_vec = Self::solve_nullspace(&a)?;
            Self::reshape_to_projection(&p_vec)
        };

        // Step 5: Denormalize: P = T2_inv * P_norm * T3
        let t2_inv = t2
            .try_inverse()
            .ok_or(SolverError::DegenerateConfiguration {
                reason: "2D normalization matrix is singular".to_string(),
            })?;

        let p_norm_full = p_norm;
        let p_times_t3 = p_norm_full * t3;
        let p = t2_inv * p_times_t3;

        // Step 6: Try both signs of P (SVD determines P only up to sign)
        let result_pos = Self::decompose_projection(&p, config);
        let result_neg = Self::decompose_projection(&(-p), config);

        // Check which result satisfies cheirality
        let result = match (result_pos, result_neg) {
            (Ok(pos), Ok(neg)) => {
                let pos_valid = Self::check_cheirality(&pos, data);
                let neg_valid = Self::check_cheirality(&neg, data);

                match (pos_valid, neg_valid) {
                    (true, false) => Ok(pos),
                    (false, true) => Ok(neg),
                    (true, true) => {
                        let pos_err: f64 = data
                            .iter()
                            .map(|d| <Self as Solver>::residual(&pos, d))
                            .sum();
                        let neg_err: f64 = data
                            .iter()
                            .map(|d| <Self as Solver>::residual(&neg, d))
                            .sum();
                        if pos_err <= neg_err { Ok(pos) } else { Ok(neg) }
                    }
                    (false, false) => {
                        if config.check_cheirality {
                            Err(SolverError::DegenerateConfiguration {
                                reason: "some points are behind the camera for both signs"
                                    .to_string(),
                            })
                        } else {
                            Ok(pos)
                        }
                    }
                }
            }
            (Ok(pos), Err(_)) => {
                if config.check_cheirality && !Self::check_cheirality(&pos, data) {
                    Err(SolverError::DegenerateConfiguration {
                        reason: "some points are behind the camera".to_string(),
                    })
                } else {
                    Ok(pos)
                }
            }
            (Err(_), Ok(neg)) => {
                if config.check_cheirality && !Self::check_cheirality(&neg, data) {
                    Err(SolverError::DegenerateConfiguration {
                        reason: "some points are behind the camera".to_string(),
                    })
                } else {
                    Ok(neg)
                }
            }
            (Err(e), Err(_)) => Err(e),
        }?;

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests;
