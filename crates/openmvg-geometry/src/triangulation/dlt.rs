// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # DLT triangulation
//!
//! Direct Linear Transform triangulation using SVD.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         DLT triangulation                            │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: observations + camera matrices
//!              │
//!              ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Build the homogeneous design matrix A                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! For each observation xᵢ, add the two rows xᵢ × (Pᵢ X) = 0 to A.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Solve A X = 0                                                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Compute A = UΣVᵀ and select the right singular vector for σₘᵢₙ.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Dehomogenize and check                                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! Divide by X₄, then check finite coordinates, depth, and reprojection error.
//! ```
//!
//! # Algorithm
//!
//! For each observation (u, v) in view i with projection matrix P_i = K_i [R_i | -R_i C_i]:
//!
//! u × (P_i * X) = 0
//!
//! This gives two linear equations per view. Stacking all equations and
//! solving via SVD gives the 3D point X that minimizes ||A * X|| with ||X|| = 1.
//!
//! # References
//!
//! R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*,
//! Cambridge University Press, 2nd ed., 2004, Section 12.2.

use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, Point3};
use openmvg_core::model::{EstimationError, EstimationResult, ViewId, WorldToCameraPose};
use std::collections::BTreeMap;

use super::{TriangulatedPoint, TriangulationConfig, TriangulationSolver};

/// Direct Linear Transform (DLT) triangulation solver.
///
/// Uses SVD to find the 3D point that minimizes algebraic error
/// across all observations.
#[derive(Debug, Clone, Default)]
pub(crate) struct DltTriangulationSolver;

impl DltTriangulationSolver {
    /// Create a DLT triangulation solver.
    pub fn new() -> Self {
        Self
    }

    /// Compute projection matrix `P = K [R | t]`.
    ///
    /// `WorldToCameraPose.rotation` is already world-to-camera and is used
    /// directly; `t = −R * C` is the translation in the camera frame.
    fn compute_projection_matrix(
        pose: &WorldToCameraPose,
        intrinsics: &Matrix3<f64>,
    ) -> Matrix3x4<f64> {
        let r = pose.rotation().matrix();
        let t = pose.translation();

        // [R | t]
        let mut rt = Matrix3x4::<f64>::zeros();
        rt.fixed_columns_mut::<3>(0).copy_from(r);
        rt.column_mut(3).copy_from(&t);

        intrinsics * rt
    }

    /// Check whether the point is in front of all cameras.
    fn check_cheirality(
        point: &Point3<f64>,
        observations: &[(ViewId, Point2<f64>)],
        poses: &BTreeMap<ViewId, WorldToCameraPose>,
        min_depth: f64,
    ) -> bool {
        for &(view_id, _) in observations {
            if let Some(pose) = poses.get(&view_id) {
                let point_cam = pose.project_to_camera(point);
                if point_cam.z <= 0.0
                    || point_cam.z < min_depth
                    || !point_cam.coords.iter().all(|value| value.is_finite())
                {
                    return false;
                }
            }
        }
        true
    }
}

impl TriangulationSolver for DltTriangulationSolver {
    fn triangulate(
        &self,
        observations: &[(ViewId, Point2<f64>)],
        poses: &BTreeMap<ViewId, WorldToCameraPose>,
        intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
        config: &TriangulationConfig,
    ) -> EstimationResult<TriangulatedPoint> {
        if observations.len() < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: observations.len(),
            });
        }

        // Build linear system: 2 equations per view
        let num_equations = observations.len() * 2;
        let mut a = DMatrix::<f64>::zeros(num_equations, 4);

        for (i, &(view_id, ref obs)) in observations.iter().enumerate() {
            let Some(pose) = poses.get(&view_id) else {
                return Err(EstimationError::ViewNotFound(view_id));
            };
            let Some(k) = intrinsics.get(&view_id) else {
                return Err(EstimationError::ViewNotFound(view_id));
            };

            let p = Self::compute_projection_matrix(pose, k);

            // Equation 1: u * p3 - p1
            // Equation 2: v * p3 - p2
            let row1 = i * 2;
            let row2 = i * 2 + 1;

            for col in 0..4 {
                a[(row1, col)] = obs.x * p[(2, col)] - p[(0, col)];
                a[(row2, col)] = obs.y * p[(2, col)] - p[(1, col)];
            }
        }

        if !a.iter().all(|value| value.is_finite()) {
            return Err(EstimationError::Numerical(
                "DLT triangulation constraints are not finite".to_owned(),
            ));
        }
        // Solve via SVD and require a distinguishable one-dimensional least-squares direction.
        let row_count = a.nrows();
        let svd = a.svd(true, true);
        let singular = svd.singular_values;
        if !singular.iter().all(|value| value.is_finite()) {
            return Err(EstimationError::Numerical(
                "DLT triangulation spectrum is not finite".to_owned(),
            ));
        }
        let tolerance = singular[0] * f64::EPSILON * row_count.max(4) as f64 * 64.0;
        if singular[0] <= 0.0 || singular[2] <= tolerance || singular[2] - singular[3] <= tolerance
        {
            return Err(EstimationError::DegenerateConfiguration {
                reason: "DLT constraints do not determine a unique point".to_owned(),
            });
        }
        let v_t = svd.v_t.ok_or_else(|| {
            EstimationError::Numerical("SVD failed to compute V matrix".to_string())
        })?;

        // Solution is the last column of V (last row of V^T)
        let solution = v_t.row(3);

        if solution[3].abs() < 1e-10 {
            return Err(EstimationError::DegenerateConfiguration {
                reason: "point is at infinity".to_string(),
            });
        }

        let point = Point3::new(
            solution[0] / solution[3],
            solution[1] / solution[3],
            solution[2] / solution[3],
        );

        // Validate result
        if !Self::check_cheirality(&point, observations, poses, config.min_depth().value()) {
            return Err(EstimationError::DegenerateConfiguration {
                reason: "point is behind a camera".to_string(),
            });
        }

        super::check_parallax(&point, observations, poses, config.min_parallax().value())?;

        let reprojection_error =
            super::mean_reprojection_error(&point, observations, poses, intrinsics);

        if reprojection_error > config.max_pixel_residual().value() {
            return Err(EstimationError::DegenerateConfiguration {
                reason: format!(
                    "reprojection error {:.2} px exceeds the limit {:.2} px",
                    reprojection_error,
                    config.max_pixel_residual().value()
                ),
            });
        }

        Ok(TriangulatedPoint {
            point,
            reprojection_error: crate::triangulation::checked_pixel_residual(reprojection_error)?,
            view_count: observations.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    #[test]
    fn test_projection_matrix() {
        let pose = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);

        let p = DltTriangulationSolver::compute_projection_matrix(&pose, &k);

        // For identity pose at origin, P = K [I | 0]
        assert!((p[(0, 0)] - 500.0).abs() < 1e-10);
        assert!((p[(0, 2)] - 320.0).abs() < 1e-10);
        assert!((p[(0, 3)]).abs() < 1e-10);
    }

    #[test]
    fn test_cheirality_check() {
        let mut poses = BTreeMap::new();
        poses.insert(
            ViewId::new(0),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0.0, 0.0, 0.0),
            )
            .unwrap(),
        );

        // Point in front of camera
        let point_front = Point3::new(0.0, 0.0, 10.0);
        let obs = vec![(ViewId::new(0), Point2::new(320.0, 240.0))];
        assert!(DltTriangulationSolver::check_cheirality(
            &point_front,
            &obs,
            &poses,
            0.0
        ));

        // Point behind camera
        let point_behind = Point3::new(0.0, 0.0, -10.0);
        assert!(!DltTriangulationSolver::check_cheirality(
            &point_behind,
            &obs,
            &poses,
            0.0
        ));
    }
}
