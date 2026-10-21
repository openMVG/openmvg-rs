// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Midpoint triangulation
//!
//! Triangulation using the midpoint of closest approach between rays.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                      Midpoint triangulation                          │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: camera poses, intrinsics, and pixel observations
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Unproject pixels into world-space camera rays                │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Compute closest points for each ray pair                     │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Weight midpoints by inverse ray gap; check depth and error   │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Algorithm
//!
//! For each pair of rays from camera centers through observed points,
//! find the point of closest approach. The midpoint of the closest
//! points on both rays is used as the triangulated point.
//!
//! For more than two views, the pairwise midpoints are averaged with weights
//! equal to the inverse distance between the two rays (closer rays get higher
//! weight).
//!
//! This method is simpler than DLT. For many views, the pairwise weighting is a
//! heuristic rather than a joint least-squares estimate.

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use openmvg_core::model::{EstimationError, EstimationResult, ViewId, WorldToCameraPose};
use std::collections::BTreeMap;

use super::{TriangulatedPoint, TriangulationConfig, TriangulationSolver};

/// Midpoint triangulation solver.
///
/// Finds the midpoint of closest approach between viewing rays.
#[derive(Debug, Clone, Default)]
pub(crate) struct MidpointTriangulationSolver;

impl MidpointTriangulationSolver {
    /// Create a midpoint triangulation solver.
    pub fn new() -> Self {
        Self
    }

    /// Compute the ray from the camera center through a pixel.
    fn compute_ray(
        obs: &Point2<f64>,
        pose: &WorldToCameraPose,
        intrinsics: &Matrix3<f64>,
    ) -> EstimationResult<(Point3<f64>, Vector3<f64>)> {
        let origin = *pose.center();
        let dir_cam = super::perspective_bearing(obs, intrinsics)?;

        // Transform direction to world coordinates (R is world-to-camera,
        // so its inverse maps camera to world)
        let dir_world = pose.rotation().inverse() * dir_cam;
        let norm = dir_world.norm();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(EstimationError::DegenerateConfiguration {
                reason: "unprojected camera ray is degenerate".to_string(),
            });
        }

        Ok((origin, dir_world / norm))
    }

    /// Find the closest points between two rays as
    /// (point on ray 1, point on ray 2, distance between them).
    fn closest_points_on_rays(
        p1: &Point3<f64>,
        d1: &Vector3<f64>,
        p2: &Point3<f64>,
        d2: &Vector3<f64>,
    ) -> Option<(Point3<f64>, Point3<f64>, f64)> {
        // Rays: R1(t) = p1 + t*d1, R2(s) = p2 + s*d2
        // Find t, s that minimize ||R1(t) - R2(s)||

        let w0 = p1 - p2;
        let a = d1.dot(d1);
        let b = d1.dot(d2);
        let c = d2.dot(d2);
        let d = d1.dot(&w0);
        let e = d2.dot(&w0);

        let denom = a * c - b * b;

        let scale = (a * c).abs();
        if !denom.is_finite() || scale <= 0.0 || denom.abs() <= scale * f64::EPSILON * 64.0 {
            return None;
        }
        let t = (b * e - c * d) / denom;
        let s = (a * e - b * d) / denom;
        if !t.is_finite() || !s.is_finite() || t <= 0.0 || s <= 0.0 {
            return None;
        }

        let point1 = Point3::from(p1.coords + t * d1);
        let point2 = Point3::from(p2.coords + s * d2);
        let distance = (point1 - point2).norm();
        if !distance.is_finite() {
            return None;
        }

        Some((point1, point2, distance))
    }

    /// Compute the weighted midpoint over all ray pairs.
    fn compute_weighted_midpoint(rays: &[(Point3<f64>, Vector3<f64>)]) -> Option<Point3<f64>> {
        if rays.len() < 2 {
            return None;
        }

        let mut weighted_sum = Vector3::zeros();
        let mut total_weight = 0.0;

        // Consider all pairs of rays
        for i in 0..rays.len() {
            for j in (i + 1)..rays.len() {
                let (p1, d1) = &rays[i];
                let (p2, d2) = &rays[j];

                let Some((closest1, closest2, dist)) = Self::closest_points_on_rays(p1, d1, p2, d2)
                else {
                    continue;
                };

                // Weight inversely by distance (closer rays are more reliable)
                let weight = if dist > 1e-10 { 1.0 / dist } else { 1e10 };

                let midpoint = (closest1.coords + closest2.coords) / 2.0;
                weighted_sum += weight * midpoint;
                total_weight += weight;
            }
        }

        if total_weight.is_finite()
            && total_weight > 0.0
            && weighted_sum.iter().all(|value| value.is_finite())
        {
            let point = Point3::from(weighted_sum / total_weight);
            point
                .coords
                .iter()
                .all(|value| value.is_finite())
                .then_some(point)
        } else {
            None
        }
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

impl TriangulationSolver for MidpointTriangulationSolver {
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

        // Compute rays for all observations
        let mut rays = Vec::new();
        for &(view_id, ref obs) in observations {
            let Some(pose) = poses.get(&view_id) else {
                return Err(EstimationError::ViewNotFound(view_id));
            };
            let Some(k) = intrinsics.get(&view_id) else {
                return Err(EstimationError::ViewNotFound(view_id));
            };

            let (origin, direction) = Self::compute_ray(obs, pose, k)?;
            rays.push((origin, direction));
        }

        // Compute weighted midpoint
        let point = Self::compute_weighted_midpoint(&rays).ok_or_else(|| {
            EstimationError::DegenerateConfiguration {
                reason: "failed to compute the midpoint".to_string(),
            }
        })?;

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
    fn test_closest_points_parallel_rays() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let d1 = Vector3::new(0.0, 0.0, 1.0);
        let p2 = Point3::new(1.0, 0.0, 0.0);
        let d2 = Vector3::new(0.0, 0.0, 1.0);

        assert!(MidpointTriangulationSolver::closest_points_on_rays(&p1, &d1, &p2, &d2).is_none());
    }

    #[test]
    fn test_closest_points_intersecting_rays() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let d1 = Vector3::new(1.0, 0.0, 1.0).normalize();
        let p2 = Point3::new(2.0, 0.0, 0.0);
        let d2 = Vector3::new(-1.0, 0.0, 1.0).normalize();

        let (c1, c2, dist) =
            MidpointTriangulationSolver::closest_points_on_rays(&p1, &d1, &p2, &d2).unwrap();

        // Rays should intersect at (1, 0, 1)
        assert!(dist < 0.1, "Rays should nearly intersect");

        let midpoint = (c1.coords + c2.coords) / 2.0;
        assert!((midpoint.x - 1.0).abs() < 0.1);
        assert!((midpoint.z - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_ray_computation() {
        let pose = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let obs = Point2::new(320.0, 240.0); // Principal point

        let (origin, direction) =
            MidpointTriangulationSolver::compute_ray(&obs, &pose, &k).unwrap();

        // Origin should be camera center
        assert!((origin - Point3::new(0.0, 0.0, 0.0)).norm() < 1e-10);

        // Direction through principal point should be along Z
        assert!((direction - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_ray_computation_rejects_singular_intrinsics() {
        let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())
            .unwrap();
        let result = MidpointTriangulationSolver::compute_ray(
            &Point2::new(10.0, 20.0),
            &pose,
            &Matrix3::zeros(),
        );
        assert!(result.is_err());
    }
}
