// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera pose from known 3D points: where a camera is, given what it sees.
//!
//! Given 3D points and the image positions where a camera sees them, these solvers recover the
//! camera's rotation and position. Start with [`estimate_absolute_pose`], which runs P3P among
//! wrong matches. To polish its pose, set [`AbsolutePoseOptions::with_refinement`] with
//! [`PoseRefinement::Perspective`] for a pinhole camera or [`PoseRefinement::Bearing`] for other
//! cameras. Poses are world-to-camera: `X_camera = R X_world + t`.
//!
//! ## Calibrated and uncalibrated
//!
//! - Calibrated: the camera intrinsics K are known. The input is unit
//!   bearings and the output is the pose (R, t). Use [`BearingToPoint`].
//!
//! - Uncalibrated: K is unknown. The input is raw pixel coordinates and the
//!   output is both the intrinsics K and the pose. Use [`PixelToPoint`].
//!
//! ## Available algorithms
//!
//! | Solver | Calibration | Min points | Output |
//! |--------|-------------|------------|--------|
//! | [`ResectionSolver`] | Calibrated | 6 | [`WorldToCameraPose`] |
//! | [`UncalibratedResectionSolver`] | Uncalibrated | 6 | [`CalibratedPose`] (intrinsics and pose) |
//! | [`p3p::solve`] with a [`p3p::P3pMethod`] | Calibrated | 3 | 0-4 poses |
//! | [`estimate_absolute_pose`] (P3P hypotheses) | Calibrated | 3 (robust) | [`AbsolutePoseEstimate`] |
//! | [`p4pf::estimate_absolute_pose_with_focal`] | Unknown focal length | 4 (robust) | [`p4pf::FocalPoseEstimate`] |
//!
//! ## Usage
//!
//! ```rust
//! use nalgebra::{Point3, Rotation3, UnitVector3};
//! use openmvg_core::model::WorldToCameraPose;
//! use openmvg_geometry::pose::p3p::{self, P3pMethod};
//! use openmvg_geometry::pose::{BearingToPoint, ResectionConfig, ResectionSolver};
//!
//! // A camera five units in front of six points, turned slightly, sees them as bearings.
//! let rotation = Rotation3::from_euler_angles(0.1, -0.2, 0.05);
//! let truth = WorldToCameraPose::from_rotation_center(rotation, Point3::new(0.3, -0.2, -5.0))?;
//! let world = [
//!     Point3::new(-1.0, -0.5, 0.2),
//!     Point3::new(1.0, -0.4, -0.3),
//!     Point3::new(0.1, 0.9, 0.5),
//!     Point3::new(-0.6, 0.6, -0.4),
//!     Point3::new(0.7, 0.3, 0.8),
//!     Point3::new(-0.2, -0.8, -0.6),
//! ];
//! let bearing = |x: &Point3<f64>| UnitVector3::new_normalize(truth.project_to_camera(x).coords);
//!
//! // P3P on three points returns up to four poses; one of them is the camera.
//! let three = [world[0], world[1], world[2]];
//! let candidates = p3p::solve(three, three.map(|x| bearing(&x)), P3pMethod::default());
//! assert!(candidates.iter().any(|pose| (pose.center() - truth.center()).norm() < 1e-6));
//!
//! // Resection on six or more points returns one pose.
//! let matches = world.iter().map(|x| BearingToPoint::try_new(*x, bearing(x)));
//! let matches = matches.collect::<Result<Vec<_>, _>>()?;
//! let poses = ResectionSolver::solve(&matches, &ResectionConfig::default())?;
//! assert!((poses[0].center() - truth.center()).norm() < 1e-6);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod p3p;
pub mod p4pf;
mod refinement;
mod resection;
mod uncalibrated_resection;

use crate::{SolverError, SolverResult};
use nalgebra::{
    Isometry3, Matrix3, Matrix4, Point2, Point3, Quaternion, RealField, Rotation3, UnitQuaternion,
    UnitVector3, Vector3,
};
pub(crate) use openmvg_core::model::WorldToCameraPose;

pub use p3p::robust::{
    AbsolutePoseEstimate, AbsolutePoseOptions, PoseRefinement, estimate_absolute_pose,
    estimate_absolute_pose_guided,
};
pub use resection::{
    ResectionConfig, ResectionSolver, refine_bearing_pose, refine_perspective_pose,
};
pub use uncalibrated_resection::{
    CalibratedPose, PinholeIntrinsics, SolverCalibratedPose, UncalibratedResectionConfig,
    UncalibratedResectionSolver,
};

/// Internal evidence surface for the native pose-refinement kernel.
#[cfg(feature = "unstable-test-support")]
#[doc(hidden)]
pub mod test_support {
    pub use openmvg_core::camera::PinholeCamera;

    pub use super::refinement::{
        PnpError, PnpReport, PnpResidualUnit, RefinementPose, optimize_bearing_pnp,
        optimize_pixel_pnp,
    };
}

pub(crate) type SolverPose<T> = Isometry3<T>;

pub(crate) trait SolverPoseExt<T: RealField + Copy> {
    /// Compute the depth of a world point (Z coordinate in the camera frame).
    fn depth(&self, world_point: &Point3<T>) -> T;

    /// Return the camera center in world coordinates.
    fn camera_center(&self) -> Point3<T>;
}

impl<T: RealField + Copy> SolverPoseExt<T> for Isometry3<T> {
    fn depth(&self, world_point: &Point3<T>) -> T {
        self.transform_point(world_point).z
    }

    fn camera_center(&self) -> Point3<T> {
        self.inverse_transform_point(&Point3::origin())
    }
}

// ============================================================================
// Correspondence types
// ============================================================================

/// 3D-2D correspondence for calibrated pose estimation.
///
/// Pairs a world point with the unit bearing, in the camera frame, that
/// observes it. This is the input of the P3P and calibrated resection solvers.
///
/// ## Checked construction
///
/// ```rust
/// use nalgebra::{Point3, UnitVector3, Vector3};
/// use openmvg_geometry::SolverResult;
/// use openmvg_geometry::pose::BearingToPoint;
///
/// fn correspondence() -> SolverResult<BearingToPoint> {
///     let direction = UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0));
///     BearingToPoint::try_new(Point3::new(1.0, 2.0, 5.0), direction)
/// }
/// ```
///
/// For image observations, get the unit direction with `Camera::bearing(pixel)`; it returns a
/// `UnitVector3` that this constructor takes as is. The two-view
/// [`BearingCorrespondence::try_new`](crate::correspondence::BearingCorrespondence::try_new)
/// takes plain vectors instead and normalizes them, so pass it `bearing.into_inner()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingToPoint {
    /// 3D point in world coordinates.
    pub(crate) world_point: Point3<f64>,
    /// Unit bearing in the camera frame.
    pub(crate) bearing: UnitVector3<f64>,
}

impl BearingToPoint {
    /// Validate a world point and an already-normalized bearing.
    pub fn try_new(world_point: Point3<f64>, bearing: UnitVector3<f64>) -> SolverResult<Self> {
        if !world_point.coords.iter().all(|value| value.is_finite())
            || !bearing.as_ref().iter().all(|value| value.is_finite())
        {
            return Err(SolverError::InvalidInput {
                reason: "pose correspondence must be finite",
            });
        }
        Ok(Self {
            world_point,
            bearing,
        })
    }

    pub(crate) fn new(world_point: Point3<f64>, bearing: UnitVector3<f64>) -> Self {
        Self {
            world_point,
            bearing,
        }
    }

    #[cfg(test)]
    pub(crate) fn try_from_direction(
        world_point: Point3<f64>,
        direction: Vector3<f64>,
    ) -> SolverResult<Self> {
        if !world_point.coords.iter().all(|value| value.is_finite())
            || !direction.iter().all(|value| value.is_finite())
        {
            return Err(SolverError::InvalidInput {
                reason: "pose correspondence must be finite",
            });
        }
        let bearing =
            UnitVector3::try_new(direction, f64::EPSILON).ok_or(SolverError::InvalidInput {
                reason: "bearing direction must be non-zero",
            })?;
        Self::try_new(world_point, bearing)
    }

    #[cfg(test)]
    pub(crate) fn from_direction(world_point: Point3<f64>, direction: Vector3<f64>) -> Self {
        Self::try_from_direction(world_point, direction)
            .expect("synthetic test bearing must be finite and non-zero")
    }

    #[cfg(test)]
    pub(crate) fn from_normalized(world_point: Point3<f64>, normalized: Point2<f64>) -> Self {
        Self::from_direction(world_point, Vector3::new(normalized.x, normalized.y, 1.0))
    }

    /// World point matched by this observation.
    #[must_use]
    pub const fn world_point(&self) -> &Point3<f64> {
        &self.world_point
    }

    /// Unit bearing in the camera frame.
    #[must_use]
    pub const fn bearing(&self) -> &UnitVector3<f64> {
        &self.bearing
    }
}

/// Unsquared angular residual, in radians, of a directed bearing.
///
/// Cheirality is measured along the observed bearing, rather than against the
/// camera Z axis, so the contract also covers equatorial and rear-hemisphere
/// rays from generalized or spherical cameras.
pub(crate) fn bearing_pose_angular_residual(
    pose: &SolverPose<f64>,
    correspondence: &BearingToPoint,
) -> f64 {
    let camera_point = pose.transform_point(&correspondence.world_point).coords;
    let bearing = correspondence.bearing.as_ref();
    let ray_depth = camera_point.dot(bearing);
    if !ray_depth.is_finite() || ray_depth <= 0.0 {
        return f64::INFINITY;
    }
    camera_point.cross(bearing).norm().atan2(ray_depth)
}

/// 3D-2D correspondence for uncalibrated pose estimation.
///
/// Uses raw pixel coordinates. The solver estimates both the camera
/// intrinsics K and the pose [R|t].
///
/// ```compile_fail
/// use openmvg_geometry::pose::PixelToPoint;
/// let _: PixelToPoint<f32>;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelToPoint {
    /// 3D point in world coordinates.
    pub(crate) world_point: Point3<f64>,
    /// 2D point in raw pixel coordinates.
    pub(crate) pixel: Point2<f64>,
}

impl PixelToPoint {
    /// Validate and construct a pixel pose correspondence.
    pub fn try_new(world_point: Point3<f64>, pixel: Point2<f64>) -> SolverResult<Self> {
        if !world_point.coords.iter().all(|value| value.is_finite())
            || !pixel.coords.iter().all(|value| value.is_finite())
        {
            return Err(SolverError::InvalidInput {
                reason: "pixel pose correspondence must be finite",
            });
        }
        Ok(Self { world_point, pixel })
    }

    #[cfg(test)]
    pub(crate) fn new(world_point: Point3<f64>, pixel: Point2<f64>) -> Self {
        Self { world_point, pixel }
    }

    /// World point matched by this observation.
    #[must_use]
    pub const fn world_point(&self) -> &Point3<f64> {
        &self.world_point
    }

    /// Raw image-space pixel matched by this observation.
    #[must_use]
    pub const fn pixel(&self) -> &Point2<f64> {
        &self.pixel
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Iteration cap for the 4 × 4 symmetric eigensolver in [`nearest_rotation`].
/// A 4 × 4 matrix converges in a few dozen implicit QR steps; the cap only
/// bounds pathological input.
const NEAREST_ROTATION_EIGEN_ITERATIONS: usize = 1000;

/// Nearest rotation to `matrix` in the Frobenius norm, as a unit quaternion.
///
/// A matrix that is already a rotation up to rounding is converted directly
/// (Shepperd's trace-branch formula). Any other matrix uses Bar-Itzhack's
/// method ("New Method for Extracting the Quaternion from a Rotation Matrix",
/// Journal of Guidance, Control, and Dynamics, 2000): the rotation maximizing
/// `tr(Rᵀ M)` is the dominant eigenvector
/// of a symmetric 4 × 4 matrix built from `M`, which yields a proper rotation
/// even when `det M ≤ 0`. Non-finite input yields a non-finite quaternion,
/// which kernel pose validation rejects.
pub(crate) fn nearest_rotation(matrix: &Matrix3<f64>) -> UnitQuaternion<f64> {
    let non_finite =
        UnitQuaternion::new_unchecked(Quaternion::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    if !matrix.iter().all(|value| value.is_finite()) {
        return non_finite;
    }
    let direct = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*matrix));
    if let Some(direct) = UnitQuaternion::try_new(direct.into_inner(), f64::EPSILON)
        && (matrix - direct.to_rotation_matrix().matrix()).norm() <= 64.0 * f64::EPSILON
    {
        return direct;
    }

    let m = matrix;
    let k = Matrix4::new(
        m[(0, 0)] - m[(1, 1)] - m[(2, 2)],
        m[(1, 0)] + m[(0, 1)],
        m[(2, 0)] + m[(0, 2)],
        m[(2, 1)] - m[(1, 2)],
        m[(1, 0)] + m[(0, 1)],
        m[(1, 1)] - m[(0, 0)] - m[(2, 2)],
        m[(2, 1)] + m[(1, 2)],
        m[(0, 2)] - m[(2, 0)],
        m[(2, 0)] + m[(0, 2)],
        m[(2, 1)] + m[(1, 2)],
        m[(2, 2)] - m[(0, 0)] - m[(1, 1)],
        m[(1, 0)] - m[(0, 1)],
        m[(2, 1)] - m[(1, 2)],
        m[(0, 2)] - m[(2, 0)],
        m[(1, 0)] - m[(0, 1)],
        m[(0, 0)] + m[(1, 1)] + m[(2, 2)],
    );
    k.try_symmetric_eigen(f64::EPSILON, NEAREST_ROTATION_EIGEN_ITERATIONS)
        .map_or(non_finite, |eigen| {
            // Eigenvector components are (x, y, z, w).
            let v = eigen.eigenvectors.column(eigen.eigenvalues.imax());
            UnitQuaternion::new_normalize(Quaternion::new(v[3], v[0], v[1], v[2]))
        })
}

/// Create a world-to-camera transform from `R` and `t`.
///
/// `translation` follows `X_camera = R * X_world + t`; it is not a camera
/// center. The rotation matrix is projected onto SO(3) by [`nearest_rotation`].
pub(crate) fn world_to_camera_from_matrix(
    rotation: &Matrix3<f64>,
    translation: Vector3<f64>,
) -> SolverPose<f64> {
    Isometry3::from_parts(translation.into(), nearest_rotation(rotation))
}

/// Convert the kernel's rotation-plus-center pose to a world-to-camera
/// isometry without changing its geometric convention.
pub(crate) fn world_to_camera_from_camera_pose(pose: &WorldToCameraPose) -> SolverPose<f64> {
    pose.to_isometry()
}

/// Convert a solver transform to the kernel's rotation-plus-center pose.
pub(crate) fn camera_pose_from_world_to_camera(
    transform: &SolverPose<f64>,
) -> Result<WorldToCameraPose, openmvg_core::model::StateValidationError> {
    let rotation =
        UnitQuaternion::new_normalize(transform.rotation.into_inner()).to_rotation_matrix();
    WorldToCameraPose::from_rotation_center(
        rotation,
        Point3::from(-(rotation.matrix().transpose() * transform.translation.vector)),
    )
}

// The pose boundary, checked as doc tests only: it takes `f64` unit bearings, exposes no
// normalized-image constructor, and keeps its fields private.
/// ```compile_fail
/// use openmvg_geometry::pose::BearingToPoint;
/// let _: BearingToPoint<f32>;
/// ```
///
/// ```compile_fail
/// use nalgebra::{Point2, Point3};
/// use openmvg_geometry::pose::BearingToPoint;
/// let _ = BearingToPoint::from_normalized(Point3::origin(), Point2::origin());
/// ```
///
/// ```compile_fail
/// use nalgebra::{Point3, UnitVector3, Vector3};
/// use openmvg_geometry::pose::BearingToPoint;
/// let correspondence = BearingToPoint::try_new(
///     Point3::origin(),
///     UnitVector3::new_normalize(Vector3::z()),
/// ).unwrap();
/// let _ = correspondence.world_point;
/// ```
#[cfg(doctest)]
#[derive(Debug)]
pub struct BearingToPointBoundary;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Point2, Point3, Vector3};

    #[test]
    fn bearing_to_point_accessors_preserve_checked_values() {
        let corr = BearingToPoint::try_from_direction(
            Point3::new(1.0, 2.0, 10.0),
            Vector3::new(0.1, 0.2, 1.0),
        )
        .unwrap();
        assert_eq!(corr.world_point(), &Point3::new(1.0, 2.0, 10.0));
        let expected = Vector3::new(0.1, 0.2, 1.0).normalize();
        assert!((corr.bearing().as_ref() - expected).norm() < 1e-10);
    }

    #[test]
    fn fallible_bearing_constructors_reject_invalid_inputs() {
        assert!(
            BearingToPoint::try_from_direction(Point3::origin(), Vector3::<f64>::zeros(),).is_err()
        );
        assert!(
            BearingToPoint::try_from_direction(Point3::origin(), Vector3::new(f64::NAN, 0.0, 1.0),)
                .is_err()
        );
    }

    #[test]
    fn pixel_to_point_accessors_preserve_checked_values() {
        let correspondence =
            PixelToPoint::try_new(Point3::new(0.0, 0.0, 10.0), Point2::new(320.0, 240.0)).unwrap();
        assert_eq!(correspondence.world_point(), &Point3::new(0.0, 0.0, 10.0));
        assert_eq!(correspondence.pixel(), &Point2::new(320.0, 240.0));
    }

    #[test]
    fn test_camera_center() {
        // Camera translated along Z (t = [0, 0, 5] means camera is at [0, 0, -5] in world)
        let pose = world_to_camera_from_matrix(&Matrix3::identity(), Vector3::new(0.0, 0.0, 5.0));

        let center = pose.camera_center();
        assert!((center - Point3::new(0.0, 0.0, -5.0)).norm() < 1e-10);
    }

    #[test]
    fn test_camera_pose_from_matrix() {
        use std::f64::consts::FRAC_PI_4;

        // 45 degree rotation around Z axis
        let c = FRAC_PI_4.cos();
        let s = FRAC_PI_4.sin();
        let r = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        let t = Vector3::new(1.0, 2.0, 3.0);

        let pose = world_to_camera_from_matrix(&r, t);

        // Verify rotation by transforming a point
        let p = Point3::new(1.0, 0.0, 0.0);
        let transformed = pose.transform_point(&p);

        // After rotation: (cos(45), sin(45), 0) + t = (c+1, s+2, 3)
        assert!((transformed.x - (c + 1.0)).abs() < 1e-10);
        assert!((transformed.y - (s + 2.0)).abs() < 1e-10);
        assert!((transformed.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn nearest_rotation_keeps_half_turns() {
        for diagonal in [
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
        ] {
            let matrix = Matrix3::from_diagonal(&diagonal);
            let rotation = nearest_rotation(&matrix).to_rotation_matrix();
            assert!((rotation.matrix() - matrix).norm() < 1e-15);
        }
    }

    #[test]
    fn nearest_rotation_satisfies_the_optimality_certificate() {
        let rotation = *Rotation3::from_euler_angles(0.7, -1.1, 2.9).matrix();
        let stretch = Matrix3::new(0.3, -0.2, 0.1, 0.05, -0.4, 0.2, -0.1, 0.25, 0.15);
        // The nearest proper rotation R makes R^T M symmetric with a
        // non-negative trace gap to any other rotation.
        for matrix in [rotation + stretch, -rotation + 0.01 * stretch] {
            let nearest = nearest_rotation(&matrix).to_rotation_matrix();
            let product = nearest.matrix().transpose() * matrix;
            assert!((product - product.transpose()).norm() < 1e-14);
            approx::assert_relative_eq!(nearest.matrix().determinant(), 1.0, epsilon = 1e-14);
            for candidate in [
                rotation,
                -rotation * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0)),
            ] {
                assert!(product.trace() >= (candidate.transpose() * matrix).trace() - 1e-14);
            }
        }
    }

    #[test]
    fn nearest_rotation_propagates_non_finite_input() {
        let mut matrix = Matrix3::identity();
        matrix[(1, 2)] = f64::NAN;
        assert!(
            nearest_rotation(&matrix)
                .coords
                .iter()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn kernel_pose_conversion_preserves_projection() {
        let kernel_pose = openmvg_core::model::WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.1, -0.2, 0.05),
            Point3::new(1.0, -2.0, 0.5),
        )
        .unwrap();
        let transform = world_to_camera_from_camera_pose(&kernel_pose);
        let point = Point3::new(0.4, -0.3, 4.0);
        approx::assert_relative_eq!(
            transform.transform_point(&point),
            kernel_pose.project_to_camera(&point),
            epsilon = 1e-12
        );

        let round_trip = camera_pose_from_world_to_camera(&transform).unwrap();
        approx::assert_relative_eq!(
            round_trip.rotation().matrix(),
            kernel_pose.rotation().matrix(),
            epsilon = 1e-12
        );
        approx::assert_relative_eq!(round_trip.center(), kernel_pose.center(), epsilon = 1e-12);
    }
}
