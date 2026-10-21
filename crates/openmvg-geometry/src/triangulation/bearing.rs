// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera-model-independent triangulation primitives on world-space rays.
//!
//! Its teaching docs are on [`TwoViewMethod`], where docs.rs shows them.

use nalgebra::{Matrix3, Point3, Vector3};
use openmvg_core::model::{EstimationError, EstimationResult, ViewId, WorldToCameraPose};

/// A calibrated observation resolved to a unit ray in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WorldRay {
    view_id: ViewId,
    origin: Point3<f64>,
    direction: Vector3<f64>,
}

impl WorldRay {
    /// Construct a validated world-space ray directly.
    pub(crate) fn try_new(
        view_id: ViewId,
        origin: Point3<f64>,
        direction: Vector3<f64>,
    ) -> EstimationResult<Self> {
        if !origin.coords.iter().all(|value| value.is_finite())
            || !direction.iter().all(|value| value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "ray for view {view_id} must be finite"
            )));
        }
        let norm = direction.norm();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(EstimationError::InvalidInput(format!(
                "ray for view {view_id} must have non-zero direction"
            )));
        }
        Ok(Self {
            view_id,
            origin,
            direction: direction / norm,
        })
    }

    /// Resolve a camera-frame bearing through a world-to-camera pose.
    pub(crate) fn try_from_camera(
        view_id: ViewId,
        pose: &WorldToCameraPose,
        bearing: Vector3<f64>,
    ) -> EstimationResult<Self> {
        pose.validate().map_err(|error| {
            EstimationError::InvalidInput(format!("invalid pose for view {view_id}: {error}"))
        })?;
        if !bearing.iter().all(|value| value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "bearing for view {view_id} must be finite"
            )));
        }
        let norm = bearing.norm();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(EstimationError::InvalidInput(format!(
                "bearing for view {view_id} must have non-zero norm"
            )));
        }
        let direction = pose.rotation().inverse() * (bearing / norm);
        Self::try_new(view_id, *pose.center(), direction)
    }

    /// Identifier of the source view.
    pub(crate) fn view_id(&self) -> ViewId {
        self.view_id
    }

    /// Signed distance from the camera center along the observed ray.
    pub(crate) fn oriented_distance(&self, point: &Point3<f64>) -> f64 {
        self.direction.dot(&(point - self.origin))
    }

    /// Angular residual in radians between this ray and a proposed point.
    pub(crate) fn angular_residual(&self, point: &Point3<f64>) -> f64 {
        let displacement = point - self.origin;
        let norm = displacement.norm();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return f64::INFINITY;
        }
        let predicted = displacement / norm;
        self.direction
            .cross(&predicted)
            .norm()
            .atan2(self.direction.dot(&predicted))
    }
}

/// How [`TrackTriangulator`](crate::triangulation::TrackTriangulator) estimates a point from two
/// rays, for its two-view hypotheses and refits; set it with
/// [`with_two_view_method`](crate::triangulation::TrackTriangulationConfig::with_two_view_method)
/// on a `TrackTriangulationConfig`.
///
/// # How it works
///
/// The method only computes the point from the rays; the track triangulator chooses the
/// observations, keeps the inliers, and accepts or rejects the result.
///
/// ```text
/// camera poses + bearings -> ray geometry -> two-ray / N-ray estimate
///                         -> degeneracy and direction checks
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TwoViewMethod {
    /// Inverse-depth-weighted midpoint of the closest points on both rays.
    #[default]
    InverseDepthWeightedMidpoint,
    /// Unweighted midpoint of the closest points on both rays.
    Midpoint,
}

/// Conditioning angle between two unoriented ray lines, in `[0, pi/2]`.
pub(crate) fn parallax_angle(first: &WorldRay, second: &WorldRay) -> f64 {
    first
        .direction
        .cross(&second.direction)
        .norm()
        .clamp(0.0, 1.0)
        .asin()
}

/// Estimate a point from two calibrated rays.
pub(crate) fn triangulate_two_view(
    first: &WorldRay,
    second: &WorldRay,
    method: TwoViewMethod,
) -> EstimationResult<Point3<f64>> {
    if first.view_id == second.view_id {
        return Err(EstimationError::InvalidInput(
            "two-view triangulation requires distinct views".to_owned(),
        ));
    }

    let baseline = first.origin - second.origin;
    let dot = first.direction.dot(&second.direction);
    let first_projection = first.direction.dot(&baseline);
    let second_projection = second.direction.dot(&baseline);
    let denominator = 1.0 - dot * dot;
    if !denominator.is_finite() || denominator <= 1e-15 {
        return Err(EstimationError::DegenerateConfiguration {
            reason: "viewing rays are parallel".to_owned(),
        });
    }

    let first_depth = (dot * second_projection - first_projection) / denominator;
    let second_depth = (second_projection - dot * first_projection) / denominator;
    if !first_depth.is_finite()
        || !second_depth.is_finite()
        || first_depth <= 0.0
        || second_depth <= 0.0
    {
        return Err(EstimationError::DegenerateConfiguration {
            reason: "triangulated point has non-positive oriented depth".to_owned(),
        });
    }

    let first_point = first.origin + first.direction * first_depth;
    let second_point = second.origin + second.direction * second_depth;
    let coordinates = match method {
        TwoViewMethod::InverseDepthWeightedMidpoint => {
            let first_weight = first_depth.recip();
            let second_weight = second_depth.recip();
            (first_point.coords * first_weight + second_point.coords * second_weight)
                / (first_weight + second_weight)
        }
        TwoViewMethod::Midpoint => (first_point.coords + second_point.coords) * 0.5,
    };
    if coordinates.iter().all(|value| value.is_finite()) {
        Ok(Point3::from(coordinates))
    } else {
        Err(EstimationError::Numerical(
            "two-view triangulation produced a non-finite point".to_owned(),
        ))
    }
}

/// Refine an initial point for angular consistency across all rays.
///
/// Each iteration solves the ray-distance system weighted by inverse squared
/// range, which locally minimizes squared sine angular error. An update is kept
/// only when the exact squared angular objective does not increase.
pub(crate) fn refine_angular_nview(
    rays: &[WorldRay],
    initial: Point3<f64>,
    max_iterations: usize,
) -> EstimationResult<Point3<f64>> {
    if rays.len() < 2 {
        return Err(EstimationError::InsufficientData {
            required: 2,
            provided: rays.len(),
        });
    }
    if !initial.coords.iter().all(|value| value.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "initial triangulated point must be finite".to_owned(),
        ));
    }

    let identity = Matrix3::identity();
    let mut point = initial;
    let mut objective = angular_objective(rays, &point);
    if !objective.is_finite() {
        return Err(EstimationError::DegenerateConfiguration {
            reason: "initial point has undefined angular residual".to_owned(),
        });
    }

    for _ in 0..max_iterations {
        let mut normal = Matrix3::zeros();
        let mut right_hand_side = Vector3::zeros();
        for ray in rays {
            let range_squared = (point - ray.origin).norm_squared();
            if !range_squared.is_finite() || range_squared <= f64::EPSILON {
                return Err(EstimationError::DegenerateConfiguration {
                    reason: "point coincides with a camera center".to_owned(),
                });
            }
            let weight = range_squared.recip();
            let projector = identity - ray.direction * ray.direction.transpose();
            normal += projector * weight;
            right_hand_side += projector * ray.origin.coords * weight;
        }

        let svd = normal.svd(true, true);
        let largest = svd.singular_values.max();
        let smallest = svd.singular_values.min();
        if !largest.is_finite() || !smallest.is_finite() || smallest <= largest * 1e-12 {
            return Err(EstimationError::DegenerateConfiguration {
                reason: "angular refit system is rank deficient".to_owned(),
            });
        }
        let coordinates = svd
            .solve(&right_hand_side, largest * 1e-12)
            .map_err(|_| EstimationError::Numerical("angular refit solve failed".to_owned()))?;
        let candidate = Point3::from(coordinates);
        let candidate_objective = angular_objective(rays, &candidate);
        if !candidate_objective.is_finite() || candidate_objective > objective {
            break;
        }
        let step_squared = (candidate - point).norm_squared();
        point = candidate;
        objective = candidate_objective;
        if step_squared <= 1e-24 * (1.0 + point.coords.norm_squared()) {
            break;
        }
    }
    Ok(point)
}

fn angular_objective(rays: &[WorldRay], point: &Point3<f64>) -> f64 {
    rays.iter()
        .map(|ray| {
            let residual = ray.angular_residual(point);
            residual * residual
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    /// Unweighted reference fit: minimize squared orthogonal ray distance.
    ///
    /// This solves
    /// `sum (I - d d^T) X = sum (I - d d^T) C`
    /// for unit world ray direction `d` and camera center `C`.
    fn triangulate_rays_least_squares(rays: &[WorldRay]) -> EstimationResult<Point3<f64>> {
        if rays.len() < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: rays.len(),
            });
        }

        let identity = Matrix3::identity();
        let mut normal = Matrix3::zeros();
        let mut right_hand_side = Vector3::zeros();
        for ray in rays {
            let projector = identity - ray.direction * ray.direction.transpose();
            normal += projector;
            right_hand_side += projector * ray.origin.coords;
        }

        let svd = normal.svd(true, true);
        let singular_values = svd.singular_values;
        let largest = singular_values.max();
        let smallest = singular_values.min();
        if !largest.is_finite() || !smallest.is_finite() || smallest <= largest * 1e-12 {
            return Err(EstimationError::DegenerateConfiguration {
                reason: "ray intersection system is rank deficient".to_owned(),
            });
        }
        let coordinates = svd
            .solve(&right_hand_side, largest * 1e-12)
            .map_err(|_| EstimationError::Numerical("ray intersection solve failed".to_owned()))?;
        if coordinates.iter().all(|value| value.is_finite()) {
            Ok(Point3::from(coordinates))
        } else {
            Err(EstimationError::Numerical(
                "N-view triangulation produced a non-finite point".to_owned(),
            ))
        }
    }

    fn ray(view_id: ViewId, center: Point3<f64>, point: Point3<f64>) -> WorldRay {
        WorldRay::try_from_camera(
            view_id,
            &WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap(),
            point - center,
        )
        .expect("fixture ray is valid")
    }

    #[test]
    fn two_view_and_nview_recover_exact_point() {
        let expected = Point3::new(0.4, -0.2, 5.0);
        let rays = [
            ray(ViewId::new(0), Point3::origin(), expected),
            ray(ViewId::new(1), Point3::new(1.0, 0.0, 0.0), expected),
            ray(ViewId::new(2), Point3::new(0.0, 1.0, 0.0), expected),
        ];

        let pair = triangulate_two_view(
            &rays[0],
            &rays[1],
            TwoViewMethod::InverseDepthWeightedMidpoint,
        )
        .expect("exact pair should triangulate");
        let nview = triangulate_rays_least_squares(&rays).expect("exact track should triangulate");

        assert!((pair - expected).norm() < 1e-10);
        assert!((nview - expected).norm() < 1e-10);
    }

    #[test]
    fn rejects_parallel_and_backward_rays() {
        let first =
            WorldRay::try_from_camera(ViewId::new(0), &WorldToCameraPose::identity(), Vector3::z())
                .expect("valid ray");
        let parallel = WorldRay::try_from_camera(
            ViewId::new(1),
            &WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
            Vector3::z(),
        )
        .expect("valid ray");
        assert!(triangulate_two_view(&first, &parallel, TwoViewMethod::Midpoint).is_err());

        let backward = WorldRay::try_from_camera(
            ViewId::new(1),
            &WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
            Vector3::new(1.0, 0.0, -1.0),
        )
        .expect("valid ray");
        assert!(triangulate_two_view(&first, &backward, TwoViewMethod::Midpoint).is_err());
    }

    #[test]
    fn angular_residual_is_camera_axis_independent() {
        let point = Point3::new(0.0, 0.0, -5.0);
        let ray = ray(ViewId::new(0), Point3::origin(), point);
        assert!(ray.oriented_distance(&point) > 0.0);
        assert!(ray.angular_residual(&point) < 1e-12);
    }

    #[test]
    fn angular_refit_is_not_dominated_by_a_distant_noisy_ray() {
        let expected = Point3::new(0.2, 0.0, 5.0);
        let first = ray(ViewId::new(0), Point3::origin(), expected);
        let second = ray(ViewId::new(1), Point3::new(1.0, 0.0, 0.0), expected);
        let far_center = Point3::new(0.0, 0.0, -995.0);
        let exact_far_direction = (expected - far_center).normalize();
        let perturbed_far_direction =
            Rotation3::from_axis_angle(&Vector3::y_axis(), 0.2_f64.to_radians())
                * exact_far_direction;
        let far = WorldRay::try_new(ViewId::new(2), far_center, perturbed_far_direction)
            .expect("valid ray");
        let rays = [first, second, far];
        let initial = triangulate_two_view(&first, &second, TwoViewMethod::Midpoint)
            .expect("clean pair should triangulate");

        let unweighted =
            triangulate_rays_least_squares(&rays).expect("ray-distance fit should exist");
        let refined = refine_angular_nview(&rays, initial, 8).expect("angular fit should exist");

        assert!((refined - expected).norm() < (unweighted - expected).norm());
        assert!(angular_objective(&rays, &refined) <= angular_objective(&rays, &initial));
    }
}
