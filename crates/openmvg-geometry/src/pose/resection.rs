// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # 6-point DLT resection solver (calibrated camera)
//!
//! Its teaching docs are on [`ResectionSolver`], where docs.rs shows them.

use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix3x4, RealField, SMatrix, SVector, SymmetricEigen, Vector3,
};
use num_traits::ToPrimitive;

use crate::normalization::WorldNormalization;
use crate::{SolverError, SolverResult};
use openmvg_core::PinholeCamera;

use super::{
    BearingToPoint, SolverPose, WorldToCameraPose, bearing_pose_angular_residual,
    camera_pose_from_world_to_camera, world_to_camera_from_camera_pose,
    world_to_camera_from_matrix,
};
use crate::shared::svd_utils::solve_homogeneous_system;
use crate::solver_traits::{SingleSolutionSolver, Solver};

/// Configuration for the calibrated resection solver.
#[derive(Debug, Clone)]
pub struct ResectionConfig {
    check_cheirality: bool,
}

config_accessors!(ResectionConfig {
    /// Whether to verify every point lies along its directed observed ray.
    copy check_cheirality, with_check_cheirality: bool;
});

impl Default for ResectionConfig {
    fn default() -> Self {
        Self {
            check_cheirality: true,
        }
    }
}

/// Refine a calibrated pose directly from directed bearing correspondences.
///
/// The input pose uses the world-to-camera convention. Bearings may lie anywhere on the viewing
/// sphere. The function returns `None` when fewer than three correspondences are supplied, a
/// correspondence opposes its directed ray, the nonlinear solve is unusable, or refinement would
/// increase the full-set angular cost. This conservative acceptance rule makes the function
/// suitable for polishing a robust pose estimate without weakening it.
///
/// For a pinhole camera, prefer [`refine_perspective_pose`], whose image-plane error matches the
/// noise of pixel measurements; use this function for fisheye, spherical and other cameras
/// described by bearings.
///
/// # How it works
///
/// A local least-squares polish, not a solver: it needs a pose that is already close, and it
/// keeps the world points and the camera fixed.
///
/// ```text
/// Input: initial pose (R, t), n ≥ 3 world points Xᵢ and their bearings bᵢ
///   │
///   ▼
/// Step 1: Validate
///   every value is finite, R is a rotation, and each point lies along the direction of its
///   bearing at the initial pose
///   │
///   ▼
/// Step 2: Residuals
///   per point, the angle between bᵢ and the predicted direction to Xᵢ, as a 2-vector in the
///   plane tangent to the sphere at bᵢ, with its analytic Jacobian
///   │
///   ▼
/// Step 3: Levenberg-Marquardt, at most 30 iterations
///   solve the damped normal equations for a six-parameter pose step, keep the step when the
///   cost drops, and adapt the damping with Nielsen's rule
///   │
///   ▼
/// Output: the refined pose, or None when the angular cost did not improve
/// ```
///
/// The residual is `r = (θ / sin θ) (a, b)`, where `(a, b)` are the coordinates of the predicted
/// direction in a tangent basis at the observed bearing and `θ` is the angle between the two, so
/// `‖r‖ = θ` in radians. The objective is the sum of the squared residuals.
///
/// [`refine_perspective_pose`] runs the same loop on image-plane residuals, the projection
/// `(x/z, y/z)` minus the normalized observation, and needs at least four points in front of
/// the camera. Both keep a result only when the summed squared angular residual over all
/// correspondences does not grow, and [`estimate_absolute_pose`](super::estimate_absolute_pose)
/// applies one of them to its inliers when [`PoseRefinement`](super::PoseRefinement) is set.
///
/// # Alternatives
///
/// The P3P solvers and [`ResectionSolver`] produce a pose from correspondences alone; this
/// function needs one as its start.
#[must_use]
pub fn refine_bearing_pose(
    initial: &WorldToCameraPose,
    correspondences: &[BearingToPoint],
) -> Option<WorldToCameraPose> {
    let initial = world_to_camera_from_camera_pose(initial);
    refine_bearing_solver_pose(&initial, correspondences)
        .and_then(|pose| camera_pose_from_world_to_camera(&pose).ok())
}

pub(crate) fn refine_bearing_solver_pose(
    initial: &SolverPose<f64>,
    correspondences: &[BearingToPoint],
) -> Option<SolverPose<f64>> {
    use super::refinement::optimize_bearing_pnp;
    use openmvg_ba::numerics::refine::{ConvergenceCriteria, NoiseModel, Options};

    if correspondences.len() < 3 {
        return None;
    }

    let world_points = refinement_world_points(correspondences);
    let observations: Vec<[f64; 3]> = correspondences
        .iter()
        .map(|correspondence| {
            let bearing = correspondence.bearing.as_ref();
            [bearing.x, bearing.y, bearing.z]
        })
        .collect();
    let initial_pose = to_refinement_pose(initial);
    let options = Options::l2(30).with_convergence(ConvergenceCriteria::new(1e-8, 1e-8, 1e-10));
    let report = optimize_bearing_pnp(
        &initial_pose,
        &world_points,
        &observations,
        options,
        NoiseModel::Unit,
    )
    .ok()?;
    accept_refined_pose(initial, correspondences, report)
}

/// Refine a forward-facing calibrated pose with a perspective PnP objective.
///
/// PnP (Perspective-n-Point) recovers a pose from n known 3D points and their images; here the
/// objective is the reprojection error in normalized image coordinates, the image plane at unit
/// focal length. It preserves the approximately isotropic image-plane noise model expected by
/// pinhole and Brown-Conrady cameras. Use [`refine_bearing_pose`] for fisheye, spherical, or other
/// bearing-only camera models.
///
/// Pass the inlier correspondences only. It returns `None` for fewer than four
/// correspondences, for a bearing that does not point forward (`z ≤ 0`), or when refinement
/// would raise the summed squared angular error over all correspondences; keep the initial pose
/// then.
#[must_use]
pub fn refine_perspective_pose(
    initial: &WorldToCameraPose,
    correspondences: &[BearingToPoint],
) -> Option<WorldToCameraPose> {
    let initial = world_to_camera_from_camera_pose(initial);
    refine_perspective_solver_pose(&initial, correspondences)
        .and_then(|pose| camera_pose_from_world_to_camera(&pose).ok())
}

pub(crate) fn refine_perspective_solver_pose(
    initial: &SolverPose<f64>,
    correspondences: &[BearingToPoint],
) -> Option<SolverPose<f64>> {
    use super::refinement::optimize_pixel_pnp;
    use openmvg_ba::numerics::refine::{ConvergenceCriteria, NoiseModel, Options};

    if correspondences.len() < 4
        || correspondences
            .iter()
            .any(|correspondence| correspondence.bearing.z <= 1e-9)
    {
        return None;
    }

    let world_points = refinement_world_points(correspondences);
    let observations: Vec<[f64; 2]> = correspondences
        .iter()
        .map(|correspondence| {
            let bearing = correspondence.bearing.as_ref();
            [bearing.x / bearing.z, bearing.y / bearing.z]
        })
        .collect();
    let options = Options::l2(30).with_convergence(ConvergenceCriteria::new(1e-8, 1e-8, 1e-10));
    let camera = PinholeCamera::try_new(1.0, 1.0, 0.0, 0.0).ok()?;
    let report = optimize_pixel_pnp(
        &to_refinement_pose(initial),
        &world_points,
        &observations,
        &camera,
        options,
        NoiseModel::Unit,
    )
    .ok()?;
    accept_refined_pose(initial, correspondences, report)
}

fn refinement_world_points(correspondences: &[BearingToPoint]) -> Vec<[f64; 3]> {
    correspondences
        .iter()
        .map(|correspondence| {
            let point = correspondence.world_point;
            [point.x, point.y, point.z]
        })
        .collect()
}

fn to_refinement_pose(pose: &SolverPose<f64>) -> super::refinement::RefinementPose {
    let rotation = pose.rotation.to_rotation_matrix();
    let translation = pose.translation.vector;
    super::refinement::RefinementPose::new(
        std::array::from_fn(|row| std::array::from_fn(|column| rotation.matrix()[(row, column)])),
        [translation.x, translation.y, translation.z],
    )
}

fn accept_refined_pose(
    initial: &SolverPose<f64>,
    correspondences: &[BearingToPoint],
    report: super::refinement::PnpReport,
) -> Option<SolverPose<f64>> {
    if !report.is_numerically_usable() {
        return None;
    }
    let refined = world_to_camera_from_matrix(
        &Matrix3::from_fn(|row, column| report.pose.rotation[row][column]),
        Vector3::new(
            report.pose.translation[0],
            report.pose.translation[1],
            report.pose.translation[2],
        ),
    );
    let initial_angular_cost = correspondences
        .iter()
        .map(|correspondence| bearing_pose_angular_residual(initial, correspondence).powi(2))
        .sum::<f64>();
    let refined_angular_cost = correspondences
        .iter()
        .map(|correspondence| bearing_pose_angular_residual(&refined, correspondence).powi(2))
        .sum::<f64>();
    (refined_angular_cost.is_finite() && refined_angular_cost <= initial_angular_cost)
        .then_some(refined)
}

/// 6-point DLT resection solver (calibrated).
///
/// Estimates the camera pose from 6+ 3D-bearing correspondences and returns one pose.
/// A degenerate configuration, or a point behind the camera when the cheirality check is on,
/// returns [`SolverError::DegenerateConfiguration`]; fewer than six correspondences return
/// [`SolverError::InsufficientData`].
///
/// The [`pose`](crate::pose) module page shows it in use.
///
/// # How it works
///
/// Estimates camera pose (R, t) from 6+ 3D-2D bearing correspondences using
/// the Direct Linear Transform (DLT) algorithm. Assumes camera intrinsics K
/// are known (calibrated camera).
///
/// ## Algorithm diagram
///
/// ```text
/// ╔═══════════════════════════════════════════════════════════════════════════╗
/// ║                     6-point DLT resection solver                          ║
/// ║                        (Calibrated camera)                                ║
/// ╠═══════════════════════════════════════════════════════════════════════════╣
/// ║                                                                           ║
/// ║  Input: n ≥ 6 correspondences                                             ║
/// ║         World points X_i = (X,Y,Z) ∈ ℝ³                                   ║
/// ║         Bearing vectors b_i = (bx,by,bz) normalized                       ║
/// ║                                                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 1: Build constraint matrix (3n × 12)                           │  ║
/// ║  │                                                                     │  ║
/// ║  │  For each correspondence (X_i, b_i):                                │  ║
/// ║  │    Add all three rows of [b_i]x P X_i = 0.                          │  ║
/// ║  │    This avoids dividing by b_z and supports full-sphere rays.       │  ║
/// ║  │                                                                     │  ║
/// ║  │  A(3n×12) · p = 0  where p = vec(P) = [p₁₁...p₃₄]ᵀ                  │  ║
/// ║  │                                                                     │  ║
/// ║  │  n = 6 gives the minimal 18×12 system; n > 6 an overdetermined one.  │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 2: Solve for null space                                        │  ║
/// ║  │                                                                     │  ║
/// ║  │  ┌─────────────────────────────────────────────────────────────┐    │  ║
/// ║  │  │ n = 6 (18×12 constraints):                                  │    │  ║
/// ║  │  │   Compute AᵀA (12×12 symmetric)                             │    │  ║
/// ║  │  │   Eigendecomposition: find smallest eigenvalue              │    │  ║
/// ║  │  │   p = corresponding eigenvector                             │    │  ║
/// ║  │  └─────────────────────────────────────────────────────────────┘    │  ║
/// ║  │  ┌─────────────────────────────────────────────────────────────┐    │  ║
/// ║  │  │ n > 6 (overdetermined):                                     │    │  ║
/// ║  │  │   SVD(A) = U·Σ·Vᵀ                                           │    │  ║
/// ║  │  │   p = last column of V (smallest singular value)            │    │  ║
/// ║  │  └─────────────────────────────────────────────────────────────┘    │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 3: Reshape to projection matrix (3×4)                          │  ║
/// ║  │                                                                     │  ║
/// ║  │  P = [ p₁ p₂ p₃ p₄  ]    (row-major from 12-vector p)               │  ║
/// ║  │      [ p₅ p₆ p₇ p₈  ]                                               │  ║
/// ║  │      [ p₉ p₁₀ p₁₁ p₁₂]                                              │  ║
/// ║  │                                                                     │  ║
/// ║  │  P = [ R | t ]  where R ∈ SO(3), t ∈ ℝ³                             │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 4: Normalize scale                                             │  ║
/// ║  │                                                                     │  ║
/// ║  │  Extract R_raw = P[0:3, 0:3]                                        │  ║
/// ║  │  Estimate scale from the mean singular value of R_raw               │  ║
/// ║  │                                                                     │  ║
/// ║  │  Use det(R_raw) only to select the projective sign                  │  ║
/// ║  │  P_norm = P / signed_scale                                          │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 5: Extract R and t                                             │  ║
/// ║  │                                                                     │  ║
/// ║  │  R = P_norm[0:3, 0:3]   (enforce orthogonality via SVD)             │  ║
/// ║  │  t = P_norm[0:3, 3]                                                 │  ║
/// ║  │                                                                     │  ║
/// ║  │  Orthogonalize R: SVD(R) = U·Σ·Vᵀ → R_ortho = U·Vᵀ                  │  ║
/// ║  │  Return pose = Isometry3(R_ortho, t)                                │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                               │                                           ║
/// ║                               ▼                                           ║
/// ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
/// ║  │ Step 6: Cheirality check                                            │  ║
/// ║  │                                                                     │  ║
/// ║  │  For each point X_i:                                                │  ║
/// ║  │    ray_depth = b_i · (R · X_i + t)                                  │  ║
/// ║  │    If ray_depth ≤ 0: Point opposes its directed ray → reject        │  ║
/// ║  │                                                                     │  ║
/// ║  │  This admits valid equatorial and rear-hemisphere observations.     │  ║
/// ║  └─────────────────────────────────────────────────────────────────────┘  ║
/// ║                                                                           ║
/// ║  Output: one camera pose (R, t), or an error                              ║
/// ║                                                                           ║
/// ╠═══════════════════════════════════════════════════════════════════════════╣
/// ║  Geometry:                                                                ║
/// ║                                                                           ║
/// ║       World frame              Camera frame                               ║
/// ║          ↑ Z                      ↑ Z (optical axis)                      ║
/// ║          │                        │                                       ║
/// ║        X_i ●──────────────────────┼──→ b_i (bearing)                      ║
/// ║            \                      │                                       ║
/// ║             \    Pose (R,t)      ●─────→ X                                ║
/// ║              \  ──────────▶     Camera                                    ║
/// ║                                   │                                       ║
/// ║                                   ▼ Y                                     ║
/// ║                                                                           ║
/// ║  Relationship: b_i = normalize(R · X_i + t)                               ║
/// ╚═══════════════════════════════════════════════════════════════════════════╝
/// ```
///
/// ## Optional model-aware refinement
///
/// [`refine_perspective_pose`] and [`refine_bearing_pose`] provide conservative nonlinear
/// polishing after robust P3P:
///
/// ```text
/// Initial world-to-camera pose + inlier correspondences
///                               │
///                               ▼
/// Choose the observation model represented by the measurements
/// (perspective image points or full-sphere directed bearings)
///                               │
///                               ▼
/// Evaluate normalized-image reprojection or exact S² logarithm residuals
///                               │
///                               ▼
///             Solve one shared six-DOF LM problem
///                               │
///                               ▼
///              Require usable numerical diagnostics
///                               │
///                               ▼
///        Accept only finite, non-increasing full-set angular cost
/// ```
///
/// # References
///
/// R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*,
/// Cambridge University Press, 2nd ed., 2004, Chapter 7.
#[derive(Debug, Clone, Default)]
pub struct ResectionSolver;

crate::solver_traits::expose_pose_solver_api!(ResectionSolver, BearingToPoint, ResectionConfig);

impl ResectionSolver {
    /// Hartley-style 3D normalization of the world points.
    fn world_normalization(correspondences: &[BearingToPoint]) -> SolverResult<WorldNormalization> {
        if correspondences.iter().any(|corr| {
            !corr
                .world_point
                .coords
                .iter()
                .all(|value| value.is_finite())
                || !corr.bearing.as_ref().iter().all(|value| value.is_finite())
        }) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "resection input contains non-finite coordinates".to_string(),
            });
        }
        WorldNormalization::from_points(correspondences.iter().map(|corr| corr.world_point))
            .ok_or_else(|| SolverError::DegenerateConfiguration {
                reason: "resection world points have no spatial extent".to_string(),
            })
    }

    /// Map one correspondence into the normalized world frame.
    #[inline]
    fn normalize_point(
        corr: &BearingToPoint,
        normalization: &WorldNormalization,
    ) -> BearingToPoint {
        BearingToPoint::new(
            normalization.normalize_point(&corr.world_point),
            corr.bearing,
        )
    }

    /// Build the constraint matrix for DLT from bearing vectors (heap-allocated).
    ///
    /// For each homogeneous world point `X` and bearing `b`, use all three
    /// rows of `b × (P X) = 0`. The three rows have rank two, but retaining all
    /// of them avoids selecting a coordinate chart and, in particular, avoids
    /// dividing by `b.z` for bearings close to the image horizon.
    fn build_constraint_matrix(correspondences: &[BearingToPoint]) -> DMatrix<f64> {
        let n = correspondences.len();
        let mut a = DMatrix::zeros(3 * n, 12);

        for (i, corr) in correspondences.iter().enumerate() {
            Self::fill_constraint_rows(&mut |row, col, val| a[(row, col)] = val, i, corr);
        }

        a
    }

    /// Build the constraint matrix for DLT (stack-allocated, 6 points = 18x12).
    /// Optimized for RANSAC minimal case.
    fn build_constraint_matrix_6point(correspondences: &[BearingToPoint]) -> SMatrix<f64, 18, 12> {
        debug_assert_eq!(correspondences.len(), 6);
        let mut a: SMatrix<f64, 18, 12> = SMatrix::zeros();

        for (i, corr) in correspondences.iter().enumerate() {
            Self::fill_constraint_rows(&mut |row, col, val| a[(row, col)] = val, i, corr);
        }

        a
    }

    /// Fill constraint rows for a single correspondence.
    #[inline]
    fn fill_constraint_rows<F: FnMut(usize, usize, f64)>(
        setter: &mut F,
        i: usize,
        corr: &BearingToPoint,
    ) {
        let b = corr.bearing.as_ref();
        let x = [
            corr.world_point.x,
            corr.world_point.y,
            corr.world_point.z,
            1.0,
        ];

        // [b]_x P X = 0.
        let coefficients = [(0.0, -b.z, b.y), (b.z, 0.0, -b.x), (-b.y, b.x, 0.0)];
        for (equation, (p0, p1, p2)) in coefficients.into_iter().enumerate() {
            let row = 3 * i + equation;
            for (col, coordinate) in x.iter().copied().enumerate() {
                setter(row, col, p0 * coordinate);
                setter(row, 4 + col, p1 * coordinate);
                setter(row, 8 + col, p2 * coordinate);
            }
        }
    }

    /// Solve for nullspace via SVD (heap-allocated).
    fn solve_nullspace<T: RealField + Copy + ToPrimitive>(
        a: &DMatrix<T>,
    ) -> SolverResult<DVector<T>> {
        solve_homogeneous_system(a)
    }

    /// Solve for nullspace via eigendecomposition (stack-allocated, 12x12).
    /// Optimized for RANSAC minimal case.
    fn solve_nullspace_6point<T: RealField + Copy + ToPrimitive>(
        a: &SMatrix<T, 18, 12>,
    ) -> SolverResult<SVector<T, 12>> {
        let ata = a.transpose() * a;
        let eigen = SymmetricEigen::new(ata);
        let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        order.sort_by(|&left, &right| {
            let left = eigen.eigenvalues[left].to_f64().unwrap_or(f64::NAN).abs();
            let right = eigen.eigenvalues[right].to_f64().unwrap_or(f64::NAN).abs();
            left.total_cmp(&right)
        });
        let spectrum: [f64; 12] = std::array::from_fn(|i| {
            eigen.eigenvalues[order[i]]
                .to_f64()
                .unwrap_or(f64::NAN)
                .abs()
        });
        if spectrum.iter().any(|value| !value.is_finite()) || spectrum[11] <= 0.0 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "resection constraints have zero or non-finite numerical rank".to_string(),
            });
        }
        let tolerance = spectrum[11] * f64::EPSILON * 18.0 * 64.0;
        if spectrum[1] <= tolerance || spectrum[1] - spectrum[0] <= tolerance {
            return Err(SolverError::DegenerateConfiguration {
                reason: "resection constraints have an ambiguous null space".to_string(),
            });
        }
        let solution: SVector<T, 12> = eigen.eigenvectors.column(order[0]).into();
        if !solution.iter().all(|value| value.is_finite()) {
            return Err(SolverError::SvdFailed);
        }
        Ok(solution)
    }

    /// Reshape 12-element vector to 3x4 projection matrix (row-major).
    fn reshape_to_projection<T: RealField + Copy>(p: &DVector<T>) -> Matrix3x4<T> {
        // Row-major: row i, column j -> index i*4 + j
        Matrix3x4::new(
            p[0], p[1], p[2], p[3], // Row 0
            p[4], p[5], p[6], p[7], // Row 1
            p[8], p[9], p[10], p[11], // Row 2
        )
    }

    /// Reshape 12-element SVector to 3x4 projection matrix (row-major).
    #[inline]
    fn reshape_to_projection_fixed<T: RealField + Copy>(p: &SVector<T, 12>) -> Matrix3x4<T> {
        Matrix3x4::new(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
        )
    }

    /// Decompose projection matrix P = [R|t] into rotation and translation.
    fn decompose_projection(p: Matrix3x4<f64>) -> SolverResult<SolverPose<f64>> {
        if !p.iter().all(|value| value.is_finite()) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "projection matrix contains non-finite coefficients".to_string(),
            });
        }

        // Extract the left 3x3 block (R, up to scale)
        let r_raw = p.fixed_view::<3, 3>(0, 0).into_owned();
        let singular_values = r_raw.svd(false, false).singular_values;
        let largest = singular_values[0];
        let smallest = singular_values[2];
        if !largest.is_finite()
            || !smallest.is_finite()
            || largest <= 0.0
            || smallest / largest <= f64::EPSILON * 64.0
        {
            return Err(SolverError::DegenerateConfiguration {
                reason: "projection matrix has a singular rotation block".to_string(),
            });
        }

        // A calibrated projection has a scaled rotation block, hence three
        // equal singular values. Their mean is a more stable scale estimate
        // than cbrt(det), especially after world-point denormalization.
        let scale_magnitude = singular_values.sum() / 3.0;
        let signed_scale = if r_raw.determinant() >= 0.0 {
            scale_magnitude
        } else {
            -scale_magnitude
        };
        if !signed_scale.is_finite() || signed_scale == 0.0 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "projection matrix has invalid scale".to_string(),
            });
        }
        let p_normalized = p / signed_scale;

        // Extract normalized R and t
        let r_normalized = p_normalized.fixed_view::<3, 3>(0, 0).into_owned();
        let translation = Vector3::new(
            p_normalized[(0, 3)],
            p_normalized[(1, 3)],
            p_normalized[(2, 3)],
        );
        if !r_normalized.iter().all(|value| value.is_finite())
            || !translation.iter().all(|value| value.is_finite())
        {
            return Err(SolverError::DegenerateConfiguration {
                reason: "projection decomposition produced a non-finite pose".to_string(),
            });
        }

        // world_to_camera_from_matrix projects the noisy block onto SO(3).
        Ok(world_to_camera_from_matrix(&r_normalized, translation))
    }

    /// Check that every point lies in the positive direction of its observed ray.
    fn check_cheirality(pose: &SolverPose<f64>, correspondences: &[BearingToPoint]) -> bool {
        correspondences.iter().all(|corr| {
            let camera_point = pose.transform_point(&corr.world_point).coords;
            camera_point.dot(corr.bearing.as_ref()) > 0.0
        })
    }
}

impl Solver for ResectionSolver {
    type Datum = BearingToPoint;
    type Model = SolverPose<f64>;
    type Config = ResectionConfig;
    type Unit = openmvg_core::Radians;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        bearing_pose_angular_residual(model, datum)
    }

    fn min_sample_size() -> usize {
        6
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.to_homogeneous().iter().all(|value| value.is_finite())
    }
}

impl SingleSolutionSolver for ResectionSolver {
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

        let normalization = Self::world_normalization(data)?;
        let world_transform = normalization.transform();

        // Use stack-allocated version for minimal 6-point case (RANSAC hot path)
        let p_normalized = if let Ok(sample) = <&[BearingToPoint; 6]>::try_from(data) {
            let normalized: [BearingToPoint; 6] =
                std::array::from_fn(|i| Self::normalize_point(&sample[i], &normalization));
            let a = Self::build_constraint_matrix_6point(&normalized);
            let p_vec = Self::solve_nullspace_6point(&a)?;
            Self::reshape_to_projection_fixed(&p_vec)
        } else {
            // Fallback to heap-allocated version for overdetermined case
            let normalized: Vec<BearingToPoint> = data
                .iter()
                .map(|corr| Self::normalize_point(corr, &normalization))
                .collect();
            let a = Self::build_constraint_matrix(&normalized);
            let p_vec = Self::solve_nullspace(&a)?;
            Self::reshape_to_projection(&p_vec)
        };

        // Decompose while coordinates are centered. Orthogonalizing the left
        // block after expanding it back around a large world-space centroid
        // can turn a tiny rotation correction into a large translation error.
        let normalized_pose = Self::decompose_projection(p_normalized)?;
        let world_scale = world_transform[(0, 0)];
        if !world_scale.is_finite() || world_scale <= 0.0 {
            return Err(SolverError::DegenerateConfiguration {
                reason: "resection normalization has invalid scale".to_string(),
            });
        }
        let centroid = Vector3::new(
            -world_transform[(0, 3)] / world_scale,
            -world_transform[(1, 3)] / world_scale,
            -world_transform[(2, 3)] / world_scale,
        );
        let rotation = normalized_pose
            .rotation
            .to_rotation_matrix()
            .matrix()
            .clone_owned();
        // For X_n = s(X-c), t_n = s(t + R c).
        let translation = normalized_pose.translation.vector / world_scale - rotation * centroid;
        let pose = world_to_camera_from_matrix(&rotation, translation);

        // Check cheirality (all points in front of camera)
        if config.check_cheirality && !Self::check_cheirality(&pose, data) {
            return Err(SolverError::DegenerateConfiguration {
                reason: "some points oppose their directed bearing".to_string(),
            });
        }

        Ok(Some(pose))
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        // RANSAC hot path: gather a minimal sample on the stack.
        let Ok(indices) = <&[usize; 6]>::try_from(indices) else {
            let sample = indices
                .iter()
                .map(|&index| {
                    data.get(index)
                        .copied()
                        .ok_or(SolverError::InvalidDataIndex {
                            index,
                            data_len: data.len(),
                        })
                })
                .collect::<SolverResult<Vec<_>>>()?;
            return Self::solve_single(&sample, config);
        };
        let datum = |index: usize| {
            data.get(index)
                .copied()
                .ok_or(SolverError::InvalidDataIndex {
                    index,
                    data_len: data.len(),
                })
        };
        let mut sample = [datum(indices[0])?; 6];
        for (target, &source) in sample.iter_mut().zip(indices).skip(1) {
            *target = datum(source)?;
        }
        Self::solve_single(&sample, config)
    }
}

#[cfg(test)]
mod tests;
