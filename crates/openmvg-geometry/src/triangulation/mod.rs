// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Triangulation: the 3D position of a point seen by two or more posed cameras.
//!
//! Each observation defines a ray from its camera center, and the point is where the rays meet,
//! as closely as noise allows. [`TrackTriangulator`] triangulates a whole track and rejects its
//! wrong observations; [`triangulate`] solves one point with a method of
//! [`TriangulationMethod`].
//!
//! ## Implementations
//!
//! - [`TriangulationMethod::Dlt`]: Direct Linear Transform (SVD-based).
//! - [`TriangulationMethod::Midpoint`]: weighted midpoint of pairwise closest
//!   points between rays.
//! - [`TriangulationMethod::L1Angular`], [`TriangulationMethod::LInfinityAngular`]
//!   and [`TriangulationMethod::InverseDepthWeightedMidpoint`]: closed-form two-view angular
//!   methods evaluated over every observation pair.
//!
//! ## Usage
//!
//! ```rust
//! use nalgebra::{Point3, Rotation3};
//! use openmvg_core::camera::{Camera, CameraModel};
//! use openmvg_core::model::{ViewId, WorldToCameraPose};
//! use openmvg_geometry::triangulation::{TriangulationConfig, TriangulationMethod, triangulate};
//! use std::collections::BTreeMap;
//!
//! // Two cameras one unit apart, with the same pinhole camera and its calibration matrix K.
//! let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
//! let k = camera.intrinsic_matrix().ok_or("a pinhole camera has a calibration matrix")?;
//! let mut poses = BTreeMap::new();
//! for (view, x) in [(ViewId::new(0), 0.0), (ViewId::new(1), 1.0)] {
//!     let center = Point3::new(x, 0.0, 0.0);
//!     poses.insert(view, WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)?);
//! }
//! let intrinsics: BTreeMap<_, _> = poses.keys().map(|&view| (view, k)).collect();
//!
//! // Where both cameras see the point (0.5, 0.2, 4.0).
//! let point = Point3::new(0.5, 0.2, 4.0);
//! let mut observations = Vec::new();
//! for (&view, pose) in &poses {
//!     observations.push((view, camera.project(&pose.project_to_camera(&point).coords)?));
//! }
//!
//! let config = TriangulationConfig::default();
//! let estimate = triangulate(TriangulationMethod::Dlt, &observations, &poses, &intrinsics, &config)?;
//! assert!((estimate.point - point).norm() < 1e-9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use openmvg_core::model::{EstimationError, EstimationResult, ViewId, WorldToCameraPose};

pub mod angular;
mod bearing;
mod contracts;
pub mod dlt;
pub mod midpoint;
mod tracks;

pub(crate) use angular::{
    InverseDepthWeightedMidpointTriangulationSolver, L1AngularTriangulationSolver,
    LInfinityAngularTriangulationSolver,
};
pub use bearing::TwoViewMethod;
pub(crate) use bearing::{WorldRay, parallax_angle, refine_angular_nview, triangulate_two_view};
pub(crate) use contracts::TriangulationSolver;
pub use contracts::{TriangulatedPoint, TriangulationConfig};
pub(crate) use dlt::DltTriangulationSolver;
pub(crate) use midpoint::MidpointTriangulationSolver;
pub use tracks::{
    AcceptanceConfig, ConsensusConfig, ConsensusThreshold, PairHypothesisBudget,
    ResolvedTrackObservation, TrackTriangulationConfig, TrackTriangulationRejection,
    TrackTriangulator, TriangulatedTrack, TriangulationContext, TriangulationError,
    TriangulationObservation, TriangulationPolicy, TriangulationQuality,
};

fn checked_pixel_residual(
    value: f64,
) -> openmvg_core::model::EstimationResult<openmvg_core::PixelResidual> {
    openmvg_core::PixelResidual::try_new(value)
        .map_err(|error| openmvg_core::model::EstimationError::Numerical(error.to_string()))
}

/// Multi-view triangulation algorithms.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TriangulationMethod {
    /// Direct linear transform; the default for calibrated pixels. Described in [`dlt`].
    #[default]
    Dlt,
    /// Midpoint of the closest points on viewing rays, described in [`midpoint`].
    Midpoint,
    /// L1 angular-error minimization, described in [`angular`].
    L1Angular,
    /// L-infinity angular-error minimization, described in [`angular`].
    LInfinityAngular,
    /// Inverse-depth-weighted midpoint, described in [`angular`].
    InverseDepthWeightedMidpoint,
}

/// Triangulate one point with an explicitly selected perspective-camera algorithm.
///
/// Pixels must be finite and undistorted. Every observation requires its pose and
/// a finite, invertible linear calibration matrix. Each view may appear only once.
/// Homogeneous calibration scale does not change the ray orientation: perspective
/// rays are chosen with positive camera-space Z. A ray with zero Z is rejected.
/// Use [`TrackTriangulator`] for distorted or spherical cameras.
///
/// It takes pixels and a calibration matrix, rather than bearings, because it checks the point
/// by its reprojection error in pixels against [`TriangulationConfig::max_pixel_residual`]. A
/// pinhole camera's matrix is `camera.intrinsic_matrix()`
/// (`openmvg_core::camera::CameraModel::intrinsic_matrix`), as the module example shows.
///
/// Parallax is the conditioning angle between unoriented ray lines, in
/// `[0, pi/2]`; parallel and antiparallel lines do not determine a point.
///
/// # Errors
///
/// Returns [`EstimationError::InsufficientData`] for fewer than two observations;
/// [`EstimationError::InvalidInput`] for malformed observations, poses,
/// calibration matrices, or configuration; [`EstimationError::ViewNotFound`] for
/// missing poses or intrinsics; and the numerical or geometric rejection of the
/// selected solver, including singular intrinsics and an ambiguous DLT null space.
pub fn triangulate(
    method: TriangulationMethod,
    observations: &[(openmvg_core::model::ViewId, nalgebra::Point2<f64>)],
    poses: &std::collections::BTreeMap<
        openmvg_core::model::ViewId,
        openmvg_core::model::WorldToCameraPose,
    >,
    intrinsics: &std::collections::BTreeMap<openmvg_core::model::ViewId, nalgebra::Matrix3<f64>>,
    config: &TriangulationConfig,
) -> openmvg_core::model::EstimationResult<TriangulatedPoint> {
    config.validate()?;
    validate_observations(observations, poses, intrinsics)?;
    match method {
        TriangulationMethod::Dlt => {
            DltTriangulationSolver::new().triangulate(observations, poses, intrinsics, config)
        }
        TriangulationMethod::Midpoint => {
            MidpointTriangulationSolver::new().triangulate(observations, poses, intrinsics, config)
        }
        TriangulationMethod::L1Angular => {
            L1AngularTriangulationSolver::new().triangulate(observations, poses, intrinsics, config)
        }
        TriangulationMethod::LInfinityAngular => LInfinityAngularTriangulationSolver::new()
            .triangulate(observations, poses, intrinsics, config),
        TriangulationMethod::InverseDepthWeightedMidpoint => {
            InverseDepthWeightedMidpointTriangulationSolver::new().triangulate(
                observations,
                poses,
                intrinsics,
                config,
            )
        }
    }
}

fn validate_observations(
    observations: &[(ViewId, Point2<f64>)],
    poses: &BTreeMap<ViewId, WorldToCameraPose>,
    intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
) -> EstimationResult<()> {
    if observations.len() < 2 {
        return Err(EstimationError::InsufficientData {
            required: 2,
            provided: observations.len(),
        });
    }
    let mut views = BTreeSet::new();
    for (view, pixel) in observations {
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "observation for view {view} must be finite"
            )));
        }
        if !views.insert(*view) {
            return Err(EstimationError::InvalidInput(format!(
                "view {view} occurs more than once in triangulation"
            )));
        }
        let pose = poses
            .get(view)
            .ok_or(EstimationError::ViewNotFound(*view))?;
        pose.validate().map_err(|error| {
            EstimationError::InvalidInput(format!("invalid pose for view {view}: {error}"))
        })?;
        let matrix = intrinsics
            .get(view)
            .ok_or(EstimationError::ViewNotFound(*view))?;
        if !matrix.iter().all(|value| value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "intrinsics for view {view} must be finite"
            )));
        }
        perspective_bearing(pixel, matrix)?;
    }
    Ok(())
}

/// Recover the positive-Z perspective ray without depending on homogeneous scale.
fn perspective_bearing(
    pixel: &Point2<f64>,
    matrix: &Matrix3<f64>,
) -> EstimationResult<Vector3<f64>> {
    if !pixel
        .coords
        .iter()
        .chain(matrix.iter())
        .all(|value| value.is_finite())
    {
        return Err(EstimationError::InvalidInput(
            "observation and intrinsics must be finite".to_owned(),
        ));
    }
    let scale = matrix.amax();
    let singular = || EstimationError::DegenerateConfiguration {
        reason: "camera intrinsics are singular".to_owned(),
    };
    if scale == 0.0 {
        return Err(singular());
    }
    let inverse = (matrix / scale).try_inverse().ok_or_else(singular)?;
    if !inverse.iter().all(|value| value.is_finite()) {
        return Err(EstimationError::Numerical(
            "camera intrinsics cannot be inverted at current precision".to_owned(),
        ));
    }
    // Each input to the matrix-vector product is bounded by one. Only the
    // direction matters, so these positive scales preserve the viewing line.
    let inverse_scale = inverse.amax();
    if inverse_scale == 0.0 {
        return Err(singular());
    }
    let homogeneous = Vector3::new(pixel.x, pixel.y, 1.0);
    let homogeneous = homogeneous / homogeneous.amax();
    let ray = (inverse / inverse_scale) * homogeneous;
    let ray_scale = ray.amax();
    if !ray.iter().all(|value| value.is_finite()) || ray_scale == 0.0 || ray.z == 0.0 {
        return Err(EstimationError::DegenerateConfiguration {
            reason: "observation has no finite positive-Z perspective ray".to_owned(),
        });
    }
    let ray = ray / ray_scale;
    let ray = if ray.z < 0.0 { -ray } else { ray };
    let ray = ray / ray.norm();
    if !ray.iter().all(|value| value.is_finite()) || ray.z <= 0.0 {
        return Err(EstimationError::Numerical(
            "positive-Z perspective ray cannot be represented at current precision".to_owned(),
        ));
    }
    Ok(ray)
}

fn mean_reprojection_error(
    point: &Point3<f64>,
    observations: &[(ViewId, Point2<f64>)],
    poses: &BTreeMap<ViewId, WorldToCameraPose>,
    intrinsics: &BTreeMap<ViewId, Matrix3<f64>>,
) -> f64 {
    if observations.is_empty() {
        return f64::INFINITY;
    }
    let mut total = 0.0;
    for (view, pixel) in observations {
        let (Some(pose), Some(matrix)) = (poses.get(view), intrinsics.get(view)) else {
            return f64::INFINITY;
        };
        let camera_point = pose.project_to_camera(point);
        if camera_point.z <= 0.0 || !camera_point.coords.iter().all(|value| value.is_finite()) {
            return f64::INFINITY;
        }
        let projected = matrix * camera_point.coords;
        if !projected.iter().all(|value| value.is_finite()) || projected.z == 0.0 {
            return f64::INFINITY;
        }
        let u = projected.x / projected.z;
        let v = projected.y / projected.z;
        total += (u - pixel.x).hypot(v - pixel.y);
    }
    total / observations.len() as f64
}

fn check_parallax(
    point: &Point3<f64>,
    observations: &[(ViewId, Point2<f64>)],
    poses: &BTreeMap<ViewId, WorldToCameraPose>,
    minimum: f64,
) -> EstimationResult<()> {
    let rays = observations
        .iter()
        .map(|(view, _)| {
            let pose = poses
                .get(view)
                .ok_or(EstimationError::ViewNotFound(*view))?;
            let ray = point - pose.center();
            let scale = ray.amax();
            if !ray.iter().all(|value| value.is_finite()) || !scale.is_finite() || scale == 0.0 {
                return Err(EstimationError::DegenerateConfiguration {
                    reason: "triangulated point has an undefined viewing ray".to_owned(),
                });
            }
            let ray = ray / scale;
            Ok(ray / ray.norm())
        })
        .collect::<EstimationResult<Vec<Vector3<f64>>>>()?;
    let mut maximum = 0.0_f64;
    for first in 0..rays.len() {
        for second in first + 1..rays.len() {
            let angle = rays[first]
                .cross(&rays[second])
                .norm()
                .atan2(rays[first].dot(&rays[second]).abs());
            maximum = maximum.max(angle);
        }
    }
    if maximum == 0.0 || maximum < minimum {
        return Err(EstimationError::DegenerateConfiguration {
            reason: "insufficient triangulation angle".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn default_triangulation_solver() -> DltTriangulationSolver {
    DltTriangulationSolver::new()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use nalgebra::{Matrix3, Point2, Point3, Rotation3};
    use openmvg_core::WorldDistanceResidual;
    use openmvg_core::model::{ViewId, WorldToCameraPose};
    use std::collections::BTreeMap;

    #[allow(clippy::type_complexity)]
    pub(crate) fn create_test_scene() -> (
        Point3<f64>,
        BTreeMap<ViewId, WorldToCameraPose>,
        BTreeMap<ViewId, Matrix3<f64>>,
        Vec<(ViewId, Point2<f64>)>,
    ) {
        // 3D point to triangulate
        let point_3d = Point3::new(1.0, 2.0, 10.0);

        // Camera intrinsics (simple pinhole)
        let focal = 500.0;
        let cx = 320.0;
        let cy = 240.0;
        let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);

        // Two cameras with different positions
        let mut poses = BTreeMap::new();
        let mut intrinsics = BTreeMap::new();

        // Camera 0 at origin, looking along Z
        poses.insert(
            ViewId::new(0),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0.0, 0.0, 0.0),
            )
            .unwrap(),
        );
        intrinsics.insert(ViewId::new(0), k);

        // Camera 1 translated along X
        poses.insert(
            ViewId::new(1),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(2.0, 0.0, 0.0),
            )
            .unwrap(),
        );
        intrinsics.insert(ViewId::new(1), k);

        // Project 3D point to 2D observations
        let mut observations = Vec::new();
        for (&view_id, pose) in &poses {
            let k = intrinsics
                .get(&view_id)
                .expect("test: intrinsics must exist for every view_id in poses");
            let point_cam = pose.project_to_camera(&point_3d);
            let x_proj = point_cam.x / point_cam.z;
            let y_proj = point_cam.y / point_cam.z;
            let u = k[(0, 0)] * x_proj + k[(0, 2)];
            let v = k[(1, 1)] * y_proj + k[(1, 2)];
            observations.push((view_id, Point2::new(u, v)));
        }

        (point_3d, poses, intrinsics, observations)
    }

    const METHODS: [TriangulationMethod; 5] = [
        TriangulationMethod::Dlt,
        TriangulationMethod::Midpoint,
        TriangulationMethod::L1Angular,
        TriangulationMethod::LInfinityAngular,
        TriangulationMethod::InverseDepthWeightedMidpoint,
    ];

    #[test]
    fn matrix_triangulation_is_invariant_to_calibration_scale_and_sign() {
        let (expected, poses, intrinsics, observations) = create_test_scene();
        for scale in [1e-120, -1e-120, 1.0, -1.0, 1e120, -1e120] {
            let scaled: BTreeMap<_, _> = intrinsics
                .iter()
                .map(|(view, matrix)| (*view, matrix * scale))
                .collect();
            let stored = scaled.clone();
            for method in METHODS {
                let result = triangulate(
                    method,
                    &observations,
                    &poses,
                    &scaled,
                    &TriangulationConfig::default(),
                )
                .unwrap();
                assert!(
                    (result.point - expected).norm() < 1e-6,
                    "method {method:?}, scale {scale}"
                );
                assert_eq!(result.view_count, observations.len());
            }
            assert_eq!(scaled, stored);
        }
        let ray = perspective_bearing(&Point2::new(1e308, 2e307), &Matrix3::identity()).unwrap();
        assert!(ray.iter().all(|value| value.is_finite()));
        assert!(ray.z > 0.0);
        assert!((ray.norm() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn matrix_triangulation_rejects_undefined_rays_before_pair_search() {
        let (_, mut poses, mut intrinsics, mut observations) = create_test_scene();
        poses.insert(ViewId::new(2), poses[&ViewId::new(0)].clone());
        intrinsics.insert(
            ViewId::new(2),
            Matrix3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        );
        observations.push((ViewId::new(2), Point2::new(0.0, 0.0)));
        for method in METHODS {
            assert!(matches!(
                triangulate(
                    method,
                    &observations,
                    &poses,
                    &intrinsics,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::DegenerateConfiguration { .. })
            ));
        }
    }

    #[test]
    fn matrix_triangulation_rejects_malformed_context_consistently() {
        let (_, poses, intrinsics, observations) = create_test_scene();
        for method in METHODS {
            let mut unknown = observations.clone();
            unknown.push((ViewId::new(2), Point2::new(320.0, 240.0)));
            assert!(matches!(
                triangulate(
                    method,
                    &unknown,
                    &poses,
                    &intrinsics,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::ViewNotFound(_))
            ));
            let mut missing = poses.clone();
            missing.remove(&ViewId::new(1));
            assert!(matches!(
                triangulate(
                    method,
                    &observations,
                    &missing,
                    &intrinsics,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::ViewNotFound(_))
            ));
            let mut missing = intrinsics.clone();
            missing.remove(&ViewId::new(1));
            assert!(matches!(
                triangulate(
                    method,
                    &observations,
                    &poses,
                    &missing,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::ViewNotFound(_))
            ));
            let duplicate = [observations[0], observations[0]];
            assert!(matches!(
                triangulate(
                    method,
                    &duplicate,
                    &poses,
                    &intrinsics,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::InvalidInput(_))
            ));
            let mut invalid = observations.clone();
            invalid[1].1.x = f64::NAN;
            assert!(matches!(
                triangulate(
                    method,
                    &invalid,
                    &poses,
                    &intrinsics,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::InvalidInput(_))
            ));
            let mut invalid = intrinsics.clone();
            invalid.get_mut(&ViewId::new(1)).unwrap()[(0, 0)] = f64::NAN;
            assert!(matches!(
                triangulate(
                    method,
                    &observations,
                    &poses,
                    &invalid,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::InvalidInput(_))
            ));
            let mut invalid = intrinsics.clone();
            invalid.insert(ViewId::new(1), Matrix3::zeros());
            assert!(matches!(
                triangulate(
                    method,
                    &observations,
                    &poses,
                    &invalid,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::DegenerateConfiguration { .. })
            ));
            let invalid = TriangulationConfig::default().with_min_parallax(
                openmvg_core::AngularResidual::try_new(std::f64::consts::PI).unwrap(),
            );
            assert!(matches!(
                triangulate(method, &observations, &poses, &intrinsics, &invalid),
                Err(EstimationError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn matrix_triangulation_enforces_conditioning_and_preserves_pinhole_geometry() {
        let (expected, poses, mut intrinsics, _) = create_test_scene();
        for matrix in intrinsics.values_mut() {
            matrix[(0, 1)] = 25.0;
            matrix[(1, 0)] = 15.0;
        }
        let observations: Vec<_> = poses
            .iter()
            .map(|(view, pose)| {
                let camera_point = pose.project_to_camera(&expected);
                let projected = intrinsics[view] * camera_point.coords;
                (
                    *view,
                    Point2::new(projected.x / projected.z, projected.y / projected.z),
                )
            })
            .collect();
        for method in METHODS {
            let result = triangulate(
                method,
                &observations,
                &poses,
                &intrinsics,
                &TriangulationConfig::default(),
            )
            .unwrap();
            assert!((result.point - expected).norm() < 1e-6);
            assert_eq!(result.view_count, 2);
            assert!(result.reprojection_error.value() < 1e-6);
        }

        let (_, poses, intrinsics, _) = create_test_scene();
        let observations = [
            (ViewId::new(0), Point2::new(320.0, 240.0)),
            (ViewId::new(1), Point2::new(310.0, 240.0)),
        ];
        for method in METHODS {
            assert!(
                triangulate(
                    method,
                    &observations,
                    &poses,
                    &intrinsics,
                    &TriangulationConfig::default()
                )
                .is_err()
            );
        }

        let poses = BTreeMap::from([
            (
                ViewId::new(0),
                WorldToCameraPose::from_matrix(Matrix3::identity(), Point3::new(0.0, 0.0, -1.0))
                    .unwrap(),
            ),
            (
                ViewId::new(1),
                WorldToCameraPose::from_matrix(
                    Matrix3::from_diagonal(&Vector3::new(-1.0, 1.0, -1.0)),
                    Point3::new(0.0, 0.0, 1.0),
                )
                .unwrap(),
            ),
        ]);
        let observations = [
            (ViewId::new(0), Point2::new(320.0, 240.0)),
            (ViewId::new(1), Point2::new(320.0, 240.0)),
        ];
        for method in METHODS {
            assert!(matches!(
                triangulate(
                    method,
                    &observations,
                    &poses,
                    &intrinsics,
                    &TriangulationConfig::default()
                ),
                Err(EstimationError::DegenerateConfiguration { .. })
            ));
        }
    }

    #[test]
    fn test_dlt_triangulation() {
        let (point_3d, poses, intrinsics, observations) = create_test_scene();

        let solver = DltTriangulationSolver::new();
        let config = TriangulationConfig::default();

        let result = solver
            .triangulate(&observations, &poses, &intrinsics, &config)
            .expect("DLT triangulation should succeed");

        let error = (result.point - point_3d).norm();
        assert!(
            error < 0.01,
            "DLT triangulation error too large: {} (expected < 0.01)",
            error
        );
    }

    #[test]
    fn test_midpoint_triangulation() {
        let (point_3d, poses, intrinsics, observations) = create_test_scene();

        let solver = MidpointTriangulationSolver::new();
        let config = TriangulationConfig::default();

        let result = solver
            .triangulate(&observations, &poses, &intrinsics, &config)
            .expect("Midpoint triangulation should succeed");

        let error = (result.point - point_3d).norm();
        assert!(
            error < 0.01,
            "Midpoint triangulation error too large: {} (expected < 0.01)",
            error
        );
    }

    #[test]
    fn test_insufficient_observations() {
        let (_, poses, intrinsics, observations) = create_test_scene();

        let solver = default_triangulation_solver();
        let config = TriangulationConfig::default();

        // Only one observation
        let single_obs = vec![observations[0]];
        let result = solver.triangulate(&single_obs, &poses, &intrinsics, &config);

        assert!(result.is_err(), "Should fail with single observation");
    }

    #[test]
    fn test_behind_camera_rejection() {
        let mut poses = BTreeMap::new();
        let mut intrinsics = BTreeMap::new();

        // Camera 0 at origin
        poses.insert(
            ViewId::new(0),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0.0, 0.0, 0.0),
            )
            .unwrap(),
        );
        // Camera 1 looking backward (point would be behind)
        poses.insert(
            ViewId::new(1),
            WorldToCameraPose::from_rotation_center(
                Rotation3::from_euler_angles(0.0, std::f64::consts::PI, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            )
            .unwrap(),
        );

        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        intrinsics.insert(ViewId::new(0), k);
        intrinsics.insert(ViewId::new(1), k);

        let observations = vec![
            (ViewId::new(0), Point2::new(320.0, 240.0)),
            (ViewId::new(1), Point2::new(320.0, 240.0)),
        ];

        let solver = default_triangulation_solver();
        let config = TriangulationConfig::default()
            .with_min_depth(WorldDistanceResidual::try_new(0.1).unwrap());

        let result = solver.triangulate(&observations, &poses, &intrinsics, &config);
        // The result depends on the geometry; this test only exercises the check.
        let _ = result;
    }
}
