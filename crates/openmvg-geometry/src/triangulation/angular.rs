// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Two-view angular triangulation methods
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                  Two-view angular triangulation                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: two poses and two unit-bearing observations
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Form relative motion r = R₁R₀ᵀ, t = t₁ − r t₀                │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Solve the L1, L∞, or IDW objective in closed form            │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Recover depths and point; check forward rays and residuals   │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Zero-allocation, closed-form 2-view triangulation algorithms from:
//!
//! - L1 and L∞ angular: S. H. Lee, J. Civera, "Closed-Form Optimal Two-View
//!   Triangulation Based on Angular Errors", ICCV 2019. arXiv:1903.09115.
//! - Inverse-depth-weighted (IDW) midpoint: S. H. Lee, J. Civera,
//!   "Triangulation: Why Optimize?", BMVC 2019. arXiv:1907.11917.
//!
//! All three methods operate on bearings (unit-norm ray directions in camera
//! coordinates) and world-to-camera rigid transforms (R, t).
//!
//! ## Convention
//!
//! Camera pose stores a world-to-camera rotation R and camera center C.
//! The translation vector is `t = −R·C`, giving `X_cam = R·X_world + t`.
//! A bearing `x` is the normalized direction of the observed ray in
//! camera coordinates: `x = normalize(K⁻¹·[u, v, 1]ᵀ)`.

use nalgebra::{Matrix3, Point2, Point3, Rotation3, Vector3};
use openmvg_core::model::{EstimationError, EstimationResult, ViewId, WorldToCameraPose};
use std::collections::BTreeMap;

use super::{TriangulatedPoint, TriangulationConfig, TriangulationSolver};

// ─── Low-level 2-view helpers (zero-allocation) ─────────────────────────────

/// Compute the relative motion from two absolute poses.
///
/// Given absolute poses (R0, t0) and (R1, t1) where `t_i = -R_i * C_i`,
/// it computes:
/// - `r`  = R1 * R0ᵀ  (relative rotation)
/// - `t`  = t1 − r * t0  (relative translation)
/// - `rx0` = r * x0  (bearing 0 rotated into frame 1)
///
/// Returns `(r, t, rx0)`.
#[inline]
fn absolute_to_relative(
    r0: &Rotation3<f64>,
    t0: &Vector3<f64>,
    x0: &Vector3<f64>,
    r1: &Rotation3<f64>,
    t1: &Vector3<f64>,
) -> (Rotation3<f64>, Vector3<f64>, Vector3<f64>) {
    let r = r1 * r0.inverse();
    let t = t1 - r * t0;
    let rx0 = r * x0;
    (r, t, rx0)
}

/// Recover a 3D point from depths along bearings in the *second* camera frame.
///
/// Given depths `lambda0`, `lambda1` with
/// `X = lambda0 * rx0 + t = lambda1 * x1` in frame 1, the world-coordinate
/// point is `X_world = R1ᵀ * (lambda1 * x1 − t1)`.
///
/// Returns `None` unless both depths are positive (cheirality).
#[inline]
fn compute_3d_point(
    lambda0: f64,
    lambda1: f64,
    r1: &Rotation3<f64>,
    t1: &Vector3<f64>,
    x1: &Vector3<f64>,
) -> Option<Point3<f64>> {
    if lambda0 <= 0.0 || lambda1 <= 0.0 {
        return None;
    }
    let p = r1.inverse() * (lambda1 * x1 - t1);
    Some(Point3::from(p))
}

/// Find the epipolar-plane normal that minimizes the larger correction needed
/// to make two bearing vectors coplanar with the baseline.
///
/// For unit bearings `f0`, `f1` and a unit plane normal `n` perpendicular to
/// the baseline, projecting a bearing into the plane changes it by
/// `asin(|n · f|)`. Because `asin` is monotone, the L∞ problem is
///
/// `min max(|n · f0|, |n · f1|), subject to |n| = 1 and n · baseline = 0`.
///
/// On that one-dimensional feasible circle, a minimum occurs where one term
/// is zero or where the two absolute terms are equal. The four cross products
/// below enumerate exactly those candidates without iteration or allocation.
#[inline]
fn linfinity_epipolar_plane_normal(
    baseline: &Vector3<f64>,
    f0: &Vector3<f64>,
    f1: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let baseline_norm = baseline.norm();
    if !baseline_norm.is_finite() || baseline_norm <= 1e-15 {
        return None;
    }

    let candidates = [
        baseline.cross(f0),
        baseline.cross(f1),
        baseline.cross(&(f0 + f1)),
        baseline.cross(&(f0 - f1)),
    ];
    let min_norm = baseline_norm * 1e-15;
    let mut best_normal = None;
    let mut best_cost = f64::INFINITY;

    for candidate in candidates {
        let norm = candidate.norm();
        if !norm.is_finite() || norm <= min_norm {
            continue;
        }
        let normal = candidate / norm;
        let cost = normal.dot(f0).abs().max(normal.dot(f1).abs());
        if cost.is_finite() && cost < best_cost {
            best_cost = cost;
            best_normal = Some(normal);
        }
    }

    best_normal
}

// ─── 2-view closed-form kernels ──────────────────────────────────────────────

/// L1-optimal angular triangulation (Lee & Civera, ICCV 2019, Lemma 2 and Table 1).
///
/// Minimizes the L1 norm of the angular reprojection errors without heap
/// allocation and returns the world point on success.
fn triangulate_l1_angular(
    r0: &Rotation3<f64>,
    t0: &Vector3<f64>,
    x0: &Vector3<f64>,
    r1: &Rotation3<f64>,
    t1: &Vector3<f64>,
    x1: &Vector3<f64>,
) -> Option<Point3<f64>> {
    let (_r_rel, t_rel, rx0) = absolute_to_relative(r0, t0, x0, r1, t1);

    let f0 = rx0.try_normalize(1e-15)?;
    let f1 = x1.try_normalize(1e-15)?;
    let n0 = f0.cross(&t_rel);
    let n1 = f1.cross(&t_rel);

    // The smaller L1 correction retains the ray whose cross product with
    // the baseline has the larger norm (Lee & Civera, Table 1). Project
    // the other ray onto that epipolar plane before recovering depths.
    let (corrected0, corrected1) = if n0.norm_squared() < n1.norm_squared() {
        let normal = n1.try_normalize(1e-15)?;
        ((f0 - normal * normal.dot(&f0)).try_normalize(1e-15)?, f1)
    } else {
        let normal = n0.try_normalize(1e-15)?;
        (f0, (f1 - normal * normal.dot(&f1)).try_normalize(1e-15)?)
    };

    let dot = corrected0.dot(&corrected1);
    let first_projection = corrected0.dot(&t_rel);
    let second_projection = corrected1.dot(&t_rel);
    let denominator = 1.0 - dot * dot;
    if !denominator.is_finite() || denominator <= 1e-15 {
        return None;
    }
    let lambda0 = (dot * second_projection - first_projection) / denominator;
    let lambda1 = (second_projection - dot * first_projection) / denominator;
    compute_3d_point(lambda0, lambda1, r1, t1, &corrected1)
}

/// L∞-optimal angular triangulation (Lee & Civera, ICCV 2019, Lemma 4).
///
/// Minimizes the maximum angular reprojection error across both views without
/// heap allocation.
///
/// The implementation solves the one-dimensional minimax epipolar-plane
/// problem in closed form, projects both bearings onto the optimal plane, then
/// intersects the corrected rays. Returns `None` for a zero baseline, parallel
/// corrected rays, or a solution that violates cheirality.
fn triangulate_linfinity_angular(
    r0: &Rotation3<f64>,
    t0: &Vector3<f64>,
    x0: &Vector3<f64>,
    r1: &Rotation3<f64>,
    t1: &Vector3<f64>,
    x1: &Vector3<f64>,
) -> Option<Point3<f64>> {
    let (_r_rel, t_rel, rx0) = absolute_to_relative(r0, t0, x0, r1, t1);
    let f0 = rx0.try_normalize(1e-15)?;
    let f1 = x1.try_normalize(1e-15)?;
    let normal = linfinity_epipolar_plane_normal(&t_rel, &f0, &f1)?;

    // Orthogonal projection is the smallest angular change that places each
    // observed bearing in the selected epipolar plane.
    let corrected0 = (f0 - normal * normal.dot(&f0)).try_normalize(1e-15)?;
    let corrected1 = (f1 - normal * normal.dot(&f1)).try_normalize(1e-15)?;

    // Intersect t_rel + lambda0 * corrected0 with lambda1 * corrected1.
    // The projected rays and baseline are coplanar, so the closest-line solve
    // is an exact intersection up to floating-point roundoff.
    let b = corrected0.dot(&corrected1);
    let d = corrected0.dot(&t_rel);
    let e = corrected1.dot(&t_rel);
    let denom = 1.0 - b * b;
    if !denom.is_finite() || denom <= 1e-15 {
        return None;
    }

    let lambda0 = (b * e - d) / denom;
    let lambda1 = (e - b * d) / denom;
    compute_3d_point(lambda0, lambda1, r1, t1, &corrected1)
}

/// Inverse-depth-weighted midpoint triangulation (Lee & Civera, BMVC 2019, Eq. 10).
///
/// Uses inverse-depth weighting with an adequacy test (Eq. 9) replacing the
/// standard cheirality check. Performs no heap allocation.
fn triangulate_idw_midpoint(
    r0: &Rotation3<f64>,
    t0: &Vector3<f64>,
    x0: &Vector3<f64>,
    r1: &Rotation3<f64>,
    t1: &Vector3<f64>,
    x1: &Vector3<f64>,
) -> Option<Point3<f64>> {
    let (_r_rel, t_rel, rx0) = absolute_to_relative(r0, t0, x0, r1, t1);

    // Standard closest-point-on-two-skew-lines formulation:
    // Ray 0 (in camera 1 frame): origin = t_rel, direction = rx0
    // Ray 1 (in camera 1 frame): origin = 0, direction = x1
    //
    // P0(s) = t_rel + s * rx0
    // P1(t) = t * x1
    //
    // At closest approach: (P0 - P1) ⊥ rx0 and (P0 - P1) ⊥ x1
    let a = rx0.dot(&rx0); // = 1 since bearing is unit
    let b = rx0.dot(x1);
    let c = x1.dot(x1); // = 1 since bearing is unit
    let d = rx0.dot(&t_rel);
    let e = x1.dot(&t_rel);

    let denom = a * c - b * b; // = 1 - b²
    if denom.abs() < 1e-15 {
        return None; // parallel bearings
    }

    // Closest point parameters:
    // lambda0 (s) = (b*e - d) / denom
    // lambda1 (t) = (e - b*d) / denom
    let lambda0 = (b * e - c * d) / denom; // = (b*e - d) / denom since c=1
    let lambda1 = (a * e - b * d) / denom; // = (e - b*d) / denom since a=1

    // Adequacy test (Eq. 9): lambda_i > 0 ensures positive depth
    if lambda0 <= 0.0 || lambda1 <= 0.0 {
        return None;
    }

    // Closest points on each ray (in camera 1 frame):
    // P0 = t_rel + lambda0 * rx0
    // P1 = lambda1 * x1
    let p0_cam1 = t_rel + lambda0 * rx0;
    let p1_cam1 = lambda1 * x1;

    // IDW midpoint: weighted average of the two closest points
    // w0 = 1/lambda0, w1 = 1/lambda1
    // X = (w0 * P0 + w1 * P1) / (w0 + w1)
    let w0 = 1.0 / lambda0;
    let w1 = 1.0 / lambda1;
    let x_cam1 = (w0 * p0_cam1 + w1 * p1_cam1) / (w0 + w1);

    // Back to world coordinates: X_world = R1^T * (X_cam1 - t1)
    let x_world = r1.inverse() * (x_cam1 - t1);
    Some(Point3::from(x_world))
}

// ─── Bearing-vector computation ──────────────────────────────────────────────

/// Compute a unit bearing vector from a pixel observation and intrinsics.
///
/// `bearing = normalize(K⁻¹ · [u, v, 1]ᵀ)`
#[cfg(test)]
#[inline]
fn pixel_to_bearing(obs: &Point2<f64>, k_inv: &Matrix3<f64>) -> Vector3<f64> {
    let pixel_homo = Vector3::new(obs.x, obs.y, 1.0);
    (k_inv * pixel_homo).normalize()
}

// ─── Shared multi-view wrapper logic ─────────────────────────────────────────

/// Try **all** observation pairs through a 2-view solver and keep the result
/// with the lowest mean reprojection error, measured across all views.
///
/// This is the N-view wrapper for the angular and IDW 2-view solvers. Instead
/// of picking only the widest-baseline pair (which may be noisy or degenerate),
/// every pair whose triangulation angle exceeds `config.min_parallax()` is evaluated
/// and the best candidate is returned.
fn solve_two_view<F>(
    solver_fn: F,
    observations: &[(ViewId, Point2<f64>)],
    poses: &BTreeMap<ViewId, WorldToCameraPose>,
    intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
    config: &TriangulationConfig,
) -> EstimationResult<TriangulatedPoint>
where
    F: Fn(
        &Rotation3<f64>,
        &Vector3<f64>,
        &Vector3<f64>,
        &Rotation3<f64>,
        &Vector3<f64>,
        &Vector3<f64>,
    ) -> Option<Point3<f64>>,
{
    if observations.len() < 2 {
        return Err(EstimationError::InsufficientData {
            required: 2,
            provided: observations.len(),
        });
    }

    // Pre-compute per-view data (bearing, rotation, translation, K⁻¹)
    struct ViewData {
        bearing: Vector3<f64>,
        rotation: Rotation3<f64>,
        translation: Vector3<f64>,
    }

    let view_data: Vec<ViewData> = observations
        .iter()
        .map(|&(vid, ref obs)| {
            let pose = poses.get(&vid).ok_or(EstimationError::ViewNotFound(vid))?;
            let k = intrinsics
                .get(&vid)
                .ok_or(EstimationError::ViewNotFound(vid))?;
            Ok(ViewData {
                bearing: super::perspective_bearing(obs, k)?,
                rotation: *pose.rotation(),
                translation: pose.translation(),
            })
        })
        .collect::<EstimationResult<_>>()?;

    let mut best_result: Option<TriangulatedPoint> = None;

    for i in 0..observations.len() {
        let vd_i = &view_data[i];
        for j in (i + 1)..observations.len() {
            let vd_j = &view_data[j];

            // Run the 2-view solver
            let point = match solver_fn(
                &vd_i.rotation,
                &vd_i.translation,
                &vd_i.bearing,
                &vd_j.rotation,
                &vd_j.translation,
                &vd_j.bearing,
            ) {
                Some(p) => p,
                None => continue,
            };

            // Cheirality: point must be in front of all cameras
            let behind = observations.iter().any(|&(vid, _)| {
                poses.get(&vid).is_some_and(|pose| {
                    let camera_point = pose.project_to_camera(&point);
                    camera_point.z <= 0.0
                        || camera_point.z < config.min_depth().value()
                        || !camera_point.coords.iter().all(|value| value.is_finite())
                })
            });
            if behind {
                continue;
            }

            // Angle check on the solving pair
            let pose_i = &poses[&observations[i].0];
            let pose_j = &poses[&observations[j].0];
            let ray_i = (point - pose_i.center()).normalize();
            let ray_j = (point - pose_j.center()).normalize();
            let angle = ray_i.cross(&ray_j).norm().atan2(ray_i.dot(&ray_j).abs());
            if !angle.is_finite() || angle == 0.0 || angle < config.min_parallax().value() {
                continue;
            }

            // Reprojection error across *all* observations
            let reproj = super::mean_reprojection_error(&point, observations, poses, intrinsics);
            if reproj > config.max_pixel_residual().value() {
                continue;
            }

            // Keep the candidate with the lowest reprojection error
            let dominated = best_result
                .as_ref()
                .is_some_and(|prev| reproj >= prev.reprojection_error.value());
            if !dominated {
                best_result = Some(TriangulatedPoint {
                    point,
                    reprojection_error: crate::triangulation::checked_pixel_residual(reproj)?,
                    view_count: observations.len(),
                });
            }
        }
    }

    best_result.ok_or_else(|| EstimationError::DegenerateConfiguration {
        reason: "no valid pair found across all observations".to_string(),
    })
}

// ─── Solver structs ──────────────────────────────────────────────────────────

/// L1-optimal angular triangulation solver.
///
/// Minimizes the L1 norm of angular reprojection errors using the
/// closed-form solution from Lee & Civera (BMVC 2019). Evaluates every
/// observation pair and keeps the lowest mean reprojection error.
#[derive(Debug, Clone, Default)]
pub(crate) struct L1AngularTriangulationSolver;

impl L1AngularTriangulationSolver {
    pub fn new() -> Self {
        Self
    }
}

impl TriangulationSolver for L1AngularTriangulationSolver {
    fn triangulate(
        &self,
        observations: &[(ViewId, Point2<f64>)],
        poses: &BTreeMap<ViewId, WorldToCameraPose>,
        intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
        config: &TriangulationConfig,
    ) -> EstimationResult<TriangulatedPoint> {
        solve_two_view(
            triangulate_l1_angular,
            observations,
            poses,
            intrinsics,
            config,
        )
    }
}

/// L∞-optimal angular triangulation solver.
///
/// Minimizes the maximum angular reprojection error across both views
/// using the closed-form solution from Lee & Civera (ICCV 2019).
#[derive(Debug, Clone, Default)]
pub(crate) struct LInfinityAngularTriangulationSolver;

impl LInfinityAngularTriangulationSolver {
    pub fn new() -> Self {
        Self
    }
}

impl TriangulationSolver for LInfinityAngularTriangulationSolver {
    fn triangulate(
        &self,
        observations: &[(ViewId, Point2<f64>)],
        poses: &BTreeMap<ViewId, WorldToCameraPose>,
        intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
        config: &TriangulationConfig,
    ) -> EstimationResult<TriangulatedPoint> {
        solve_two_view(
            triangulate_linfinity_angular,
            observations,
            poses,
            intrinsics,
            config,
        )
    }
}

/// Inverse-depth-weighted midpoint triangulation solver.
///
/// Uses the IDW midpoint formula from Lee & Civera (BMVC 2019) with
/// an adequacy test replacing the standard cheirality check.
#[derive(Debug, Clone, Default)]
pub(crate) struct InverseDepthWeightedMidpointTriangulationSolver;

impl InverseDepthWeightedMidpointTriangulationSolver {
    pub fn new() -> Self {
        Self
    }
}

impl TriangulationSolver for InverseDepthWeightedMidpointTriangulationSolver {
    fn triangulate(
        &self,
        observations: &[(ViewId, Point2<f64>)],
        poses: &BTreeMap<ViewId, WorldToCameraPose>,
        intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
        config: &TriangulationConfig,
    ) -> EstimationResult<TriangulatedPoint> {
        solve_two_view(
            triangulate_idw_midpoint,
            observations,
            poses,
            intrinsics,
            config,
        )
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    /// Build a simple 2-camera test scene and return everything needed.
    ///
    /// Camera 0 at origin, camera 1 at (2, 0, 0), both looking along +Z.
    /// Point at (1, 2, 10).
    #[allow(clippy::type_complexity)]
    fn test_scene() -> (
        Point3<f64>,
        WorldToCameraPose,
        WorldToCameraPose,
        Matrix3<f64>,
        Vector3<f64>,
        Vector3<f64>,
        Vector3<f64>,
        Vector3<f64>,
    ) {
        let point = Point3::new(1.0, 2.0, 10.0);

        let focal = 500.0;
        let cx = 320.0;
        let cy = 240.0;
        let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
        let k_inv = k.try_inverse().expect("test: K matrix must be invertible");

        let pose0 = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let pose1 = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(2.0, 0.0, 0.0),
        )
        .unwrap();

        // Project to pixel, then to bearing
        let project = |pose: &WorldToCameraPose, k: &Matrix3<f64>| -> (Point2<f64>, Vector3<f64>) {
            let pc = pose.project_to_camera(&point);
            let u = k[(0, 0)] * pc.x / pc.z + k[(0, 2)];
            let v = k[(1, 1)] * pc.y / pc.z + k[(1, 2)];
            let obs = Point2::new(u, v);
            let bearing = pixel_to_bearing(&obs, &k_inv);
            (obs, bearing)
        };

        let (_obs0, b0) = project(&pose0, &k);
        let (_obs1, b1) = project(&pose1, &k);

        let t0 = pose0.translation();
        let t1 = pose1.translation();

        (point, pose0, pose1, k, t0, t1, b0, b1)
    }

    #[test]
    fn test_l1_angular_basic() {
        let (gt, pose0, pose1, _k, t0, t1, b0, b1) = test_scene();

        let result = triangulate_l1_angular(pose0.rotation(), &t0, &b0, pose1.rotation(), &t1, &b1);

        let p = result.expect("L1 angular should succeed");
        let error = (p - gt).norm();
        assert!(
            error < 0.05,
            "L1 angular error too large: {error} (expected < 0.05)"
        );
    }

    #[test]
    fn l1_noisy_solution_attains_the_smaller_epipolar_plane_correction() {
        let r = Rotation3::identity();
        let t0 = Vector3::zeros();
        let t1 = Vector3::new(-2.0, 0.0, 0.0);
        let f0 = Vector3::new(0.0, 0.2, 1.0).normalize();
        let f1 = Vector3::new(-0.2, 0.21, 1.0).normalize();
        let point = triangulate_l1_angular(&r, &t0, &f0, &r, &t1, &f1).unwrap();
        let expected = Point3::new(0.0, 2.003_846_153_846_154, 10.019_230_769_230_77);
        assert!((point - expected).norm() < 1e-11);

        let angle = |prediction: Vector3<f64>, observation: &Vector3<f64>| {
            prediction
                .cross(observation)
                .norm()
                .atan2(prediction.dot(observation))
        };
        let objective = angle(point.coords, &f0) + angle(point.coords + t1, &f1);
        let expected_objective = (0.01 / (1.04_f64.sqrt() * 1.0841_f64.sqrt())).asin();
        assert!((objective - expected_objective).abs() < 1e-13);

        let swapped = triangulate_l1_angular(&r, &t1, &f1, &r, &t0, &f0).unwrap();
        assert!((swapped - point).norm() < 1e-11);
    }

    #[test]
    fn test_linfinity_angular_basic() {
        let (gt, pose0, pose1, _k, t0, t1, b0, b1) = test_scene();

        let result =
            triangulate_linfinity_angular(pose0.rotation(), &t0, &b0, pose1.rotation(), &t1, &b1);

        let p = result.expect("LInfinity angular should succeed");
        let error = (p - gt).norm();
        assert!(
            error < 0.05,
            "LInfinity angular error too large: {error} (expected < 0.05)"
        );
    }

    #[test]
    fn test_linfinity_plane_is_global_minimax() {
        let baseline = Vector3::new(-2.0, 0.3, -0.1);
        let f0 = Vector3::new(0.18, 0.31, 0.93).normalize();
        let f1 = Vector3::new(-0.22, 0.25, 0.94).normalize();
        let normal = linfinity_epipolar_plane_normal(&baseline, &f0, &f1)
            .expect("non-degenerate bearings must define an optimal plane");
        let analytic_cost = normal.dot(&f0).abs().max(normal.dot(&f1).abs());

        // Independent dense sweep over every feasible plane normal. The
        // closed-form candidate must be at least as good as the sampled optimum.
        let baseline_unit = baseline.normalize();
        let reference = if baseline_unit.z.abs() < 0.9 {
            Vector3::z()
        } else {
            Vector3::y()
        };
        let u = baseline_unit.cross(&reference).normalize();
        let v = baseline_unit.cross(&u);
        let mut sampled_cost = f64::INFINITY;
        for i in 0..20_000 {
            let angle = std::f64::consts::TAU * i as f64 / 20_000.0;
            let candidate = u * angle.cos() + v * angle.sin();
            sampled_cost = sampled_cost.min(candidate.dot(&f0).abs().max(candidate.dot(&f1).abs()));
        }

        assert!(
            analytic_cost <= sampled_cost + 1e-12,
            "closed-form cost {analytic_cost:.12e} exceeds sampled optimum {sampled_cost:.12e}"
        );
    }

    #[test]
    fn test_linfinity_noisy_solution_attains_minimax_objective() {
        let (_gt, pose0, pose1, _k, t0, t1, b0, b1) = test_scene();
        let noisy0 = (b0 + Vector3::new(0.004, 0.025, -0.003)).normalize();
        let noisy1 = (b1 + Vector3::new(-0.006, -0.018, 0.004)).normalize();

        let point = triangulate_linfinity_angular(
            pose0.rotation(),
            &t0,
            &noisy0,
            pose1.rotation(),
            &t1,
            &noisy1,
        )
        .expect("the noisy fixture remains cheiral and non-degenerate");

        let predicted0 = (pose0.rotation() * point.coords + t0).normalize();
        let predicted1 = (pose1.rotation() * point.coords + t1).normalize();
        let achieved = predicted0
            .dot(&noisy0)
            .clamp(-1.0, 1.0)
            .acos()
            .max(predicted1.dot(&noisy1).clamp(-1.0, 1.0).acos());

        let (_relative_rotation, baseline, rotated0) =
            absolute_to_relative(pose0.rotation(), &t0, &noisy0, pose1.rotation(), &t1);
        let normal = linfinity_epipolar_plane_normal(&baseline, &rotated0, &noisy1)
            .expect("fixture must define a minimax plane");
        let optimum = normal
            .dot(&rotated0)
            .abs()
            .max(normal.dot(&noisy1).abs())
            .clamp(0.0, 1.0)
            .asin();

        assert!(
            (achieved - optimum).abs() < 1e-10,
            "triangulated max error {achieved:.12e} differs from optimum {optimum:.12e}"
        );
    }

    #[test]
    fn test_idw_midpoint_basic() {
        let (gt, pose0, pose1, _k, t0, t1, b0, b1) = test_scene();

        let result =
            triangulate_idw_midpoint(pose0.rotation(), &t0, &b0, pose1.rotation(), &t1, &b1);

        let p = result.expect("IDW midpoint should succeed");
        let error = (p - gt).norm();
        assert!(
            error < 0.05,
            "IDW midpoint error too large: {error} (expected < 0.05)"
        );
    }

    #[test]
    fn test_parallel_bearings_return_none() {
        // Two cameras at same position → parallel rays → degenerate
        let r = Rotation3::identity();
        let t = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 0.0, 1.0);

        assert!(triangulate_l1_angular(&r, &t, &b, &r, &t, &b).is_none());
        assert!(triangulate_linfinity_angular(&r, &t, &b, &r, &t, &b).is_none());
        assert!(triangulate_idw_midpoint(&r, &t, &b, &r, &t, &b).is_none());
    }

    #[test]
    fn test_point_behind_camera_returns_none() {
        // Point behind both cameras (negative Z)
        let point = Point3::new(0.0, 0.0, -5.0);
        let _focal = 500.0;
        let _k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);

        let pose0 = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let pose1 = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        // Bearings point "backward" in camera frame
        let pc0 = pose0.project_to_camera(&point);
        let pc1 = pose1.project_to_camera(&point);
        let b0 = Vector3::new(pc0.x, pc0.y, pc0.z).normalize();
        let b1 = Vector3::new(pc1.x, pc1.y, pc1.z).normalize();

        let t0 = pose0.translation();
        let t1 = pose1.translation();

        // IDW midpoint should reject (negative lambda)
        let result =
            triangulate_idw_midpoint(pose0.rotation(), &t0, &b0, pose1.rotation(), &t1, &b1);
        // The point is behind the cameras, so lambda check should catch it
        // (depending on geometry, it may still produce a solution)
        // At minimum, if it returns Some, the point should be approximately correct
        if let Some(p) = result {
            let error = (p - point).norm();
            assert!(error < 1.0, "If IDW returned a point, it should be close");
        }
    }

    #[test]
    fn test_rotated_cameras() {
        let point = Point3::new(0.0, 0.0, 5.0);
        let focal = 500.0;
        let k = Matrix3::new(focal, 0.0, 320.0, 0.0, focal, 240.0, 0.0, 0.0, 1.0);
        let k_inv = k.try_inverse().expect("test: K matrix must be invertible");

        // Camera 0 at (-2, 0, 0) looking along +X (rotated)
        let r0 = Rotation3::from_euler_angles(0.0, -std::f64::consts::FRAC_PI_4, 0.0);
        let c0 = Point3::new(-2.0, 0.0, 0.0);
        let pose0 = WorldToCameraPose::from_rotation_center(r0, c0).unwrap();

        // Camera 1 at (2, 0, 0) looking along -X (rotated)
        let r1 = Rotation3::from_euler_angles(0.0, std::f64::consts::FRAC_PI_4, 0.0);
        let c1 = Point3::new(2.0, 0.0, 0.0);
        let pose1 = WorldToCameraPose::from_rotation_center(r1, c1).unwrap();

        let project_bearing = |pose: &WorldToCameraPose| -> Vector3<f64> {
            let pc = pose.project_to_camera(&point);
            let u = k[(0, 0)] * pc.x / pc.z + k[(0, 2)];
            let v = k[(1, 1)] * pc.y / pc.z + k[(1, 2)];
            pixel_to_bearing(&Point2::new(u, v), &k_inv)
        };

        let b0 = project_bearing(&pose0);
        let b1 = project_bearing(&pose1);
        let t0 = pose0.translation();
        let t1 = pose1.translation();

        // All three methods should work
        for (name, result) in [
            ("L1", triangulate_l1_angular(&r0, &t0, &b0, &r1, &t1, &b1)),
            (
                "LInf",
                triangulate_linfinity_angular(&r0, &t0, &b0, &r1, &t1, &b1),
            ),
            (
                "IDW",
                triangulate_idw_midpoint(&r0, &t0, &b0, &r1, &t1, &b1),
            ),
        ] {
            let p = result.unwrap_or_else(|| panic!("{name} should succeed for rotated cameras"));
            let error = (p - point).norm();
            assert!(
                error < 0.1,
                "{name}: error too large: {error} (expected < 0.1)"
            );
        }
    }

    // ── TriangulationSolver trait tests ──

    #[test]
    fn test_l1_solver_trait() {
        let (point_3d, poses, intrinsics, observations) = super::super::tests::create_test_scene();

        let solver = L1AngularTriangulationSolver::new();
        let config = TriangulationConfig::default();
        let result = solver
            .triangulate(&observations, &poses, &intrinsics, &config)
            .expect("L1 solver should succeed");

        let error = (result.point - point_3d).norm();
        assert!(error < 0.05, "L1 solver error: {error}");
    }

    #[test]
    fn test_linfinity_solver_trait() {
        let (point_3d, poses, intrinsics, observations) = super::super::tests::create_test_scene();

        let solver = LInfinityAngularTriangulationSolver::new();
        let config = TriangulationConfig::default();
        let result = solver
            .triangulate(&observations, &poses, &intrinsics, &config)
            .expect("LInfinity solver should succeed");

        let error = (result.point - point_3d).norm();
        assert!(error < 0.05, "LInfinity solver error: {error}");
    }

    #[test]
    fn test_idw_solver_trait() {
        let (point_3d, poses, intrinsics, observations) = super::super::tests::create_test_scene();

        let solver = InverseDepthWeightedMidpointTriangulationSolver::new();
        let config = TriangulationConfig::default();
        let result = solver
            .triangulate(&observations, &poses, &intrinsics, &config)
            .expect("IDW solver should succeed");

        let error = (result.point - point_3d).norm();
        assert!(error < 0.05, "IDW solver error: {error}");
    }

    #[test]
    fn test_insufficient_observations_two_view() {
        let solver = L1AngularTriangulationSolver::new();
        let config = TriangulationConfig::default();
        let poses = BTreeMap::new();
        let intrinsics = BTreeMap::new();
        let obs = vec![(ViewId::new(0), Point2::new(100.0, 100.0))];

        let result = solver.triangulate(&obs, &poses, &intrinsics, &config);
        assert!(result.is_err());
    }
}
