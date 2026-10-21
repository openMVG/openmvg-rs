// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated geometry values shared by SfM and export code.

use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, Vector3};
use serde::{Deserialize, Deserializer, Serialize};

/// Invalid numerical state at a shared geometry boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StateValidationError {
    /// A named field contains NaN or infinity.
    #[error("{field} must be finite")]
    NonFinite {
        /// Name of the field containing a non-finite value.
        field: &'static str,
    },
    /// A matrix is not a proper three-dimensional rotation.
    #[error("rotation must be orthonormal with determinant +1")]
    InvalidRotation,
}

pub(crate) fn validate_rotation(rotation: &Rotation3<f64>) -> Result<(), StateValidationError> {
    let matrix = rotation.matrix();
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(StateValidationError::NonFinite { field: "rotation" });
    }
    let orthogonality_error = (matrix.transpose() * matrix - Matrix3::identity()).norm();
    if !orthogonality_error.is_finite()
        || orthogonality_error > 1e-8
        || (matrix.determinant() - 1.0).abs() > 1e-8
    {
        return Err(StateValidationError::InvalidRotation);
    }
    Ok(())
}

/// World-to-camera rotation `R` and camera center `C` of a camera.
///
/// The center is in world coordinates, and the pose follows the standard SfM / COLMAP
/// convention:
///
///   X_cam = R * (X_world − C)
///
/// Equivalently the translation vector is `t = −R * C`, so the 3 × 4
/// projection matrix is `P = K [R | t]`. The crate's [Converting
/// poses](crate#converting-poses) section converts to and from COLMAP files, camera-to-world
/// poses and OpenGL view matrices.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WorldToCameraPose {
    /// World-to-camera rotation R.
    rotation: Rotation3<f64>,
    /// Camera center in world coordinates.
    center: Point3<f64>,
}

impl WorldToCameraPose {
    /// Creates a camera pose from a world-to-camera rotation and a camera center.
    ///
    /// `rotation` is the world-to-camera rotation R. `center` is the camera center C in
    /// world coordinates, not the translation `t = −R * C` (use [`Self::translation`] to
    /// read `t`).
    ///
    /// # Errors
    ///
    /// Returns [`StateValidationError::NonFinite`] when the center, the rotation or the
    /// derived translation is not finite, and [`StateValidationError::InvalidRotation`]
    /// when the rotation is not orthonormal with determinant +1, to within `1e-8`: both the
    /// Frobenius norm of `RᵀR − I` and `|det R − 1|` must stay below it. A matrix read from a
    /// file with a few decimals can fail; project it onto the closest rotation first, for
    /// example with nalgebra's `Rotation3::from_matrix`.
    #[inline]
    pub fn from_rotation_center(
        rotation: Rotation3<f64>,
        center: Point3<f64>,
    ) -> Result<Self, StateValidationError> {
        let pose = Self { rotation, center };
        pose.validate()?;
        Ok(pose)
    }

    /// Create a pose from a world-to-camera rotation matrix and a camera center.
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_rotation_center`]: the matrix must be a proper rotation to within
    /// `1e-8`.
    #[inline]
    pub fn from_matrix(
        rotation: Matrix3<f64>,
        center: Point3<f64>,
    ) -> Result<Self, StateValidationError> {
        Self::from_rotation_center(Rotation3::from_matrix_unchecked(rotation), center)
    }

    /// Validates that this pose contains finite Euclidean geometry.
    pub fn validate(&self) -> Result<(), StateValidationError> {
        if !self.center.coords.iter().all(|value| value.is_finite()) {
            return Err(StateValidationError::NonFinite {
                field: "camera center",
            });
        }
        validate_rotation(&self.rotation)?;
        if !self.translation().iter().all(|value| value.is_finite()) {
            return Err(StateValidationError::NonFinite {
                field: "camera translation",
            });
        }
        Ok(())
    }

    /// Create an identity pose (camera at the origin with world and camera axes aligned).
    #[inline]
    pub fn identity() -> Self {
        Self {
            rotation: Rotation3::identity(),
            center: Point3::origin(),
        }
    }

    /// World-to-camera rotation.
    #[inline]
    #[must_use]
    pub const fn rotation(&self) -> &Rotation3<f64> {
        &self.rotation
    }

    /// Camera center in world coordinates.
    #[inline]
    #[must_use]
    pub const fn center(&self) -> &Point3<f64> {
        &self.center
    }

    /// The world-to-camera rotation as a plain 3 × 3 matrix, the same rotation as
    /// [`Self::rotation`]. Use it where a `Matrix3` is expected, such as building a projection
    /// matrix `K [R | t]`; [`Self::rotation`] keeps the type that guarantees a rotation.
    #[inline]
    pub fn rotation_matrix(&self) -> &Matrix3<f64> {
        self.rotation.matrix()
    }

    /// Get the translation vector `t = −R * C`.
    #[inline]
    pub fn translation(&self) -> Vector3<f64> {
        -(self.rotation * self.center.coords)
    }

    /// Convert to world-to-camera isometry (rotation + translation).
    #[inline]
    pub fn to_isometry(&self) -> Isometry3<f64> {
        let rotation = nalgebra::UnitQuaternion::new_normalize(
            nalgebra::UnitQuaternion::from_rotation_matrix(&self.rotation).into_inner(),
        );
        // Derive translation from the quaternion used by the isometry so
        // accepted matrix roundoff cannot move the stored camera center.
        let translation = -(rotation * self.center.coords);
        Isometry3::from_parts(nalgebra::Translation3::from(translation), rotation)
    }

    /// Project a world point to camera coordinates: `X_cam = R * (X_world − C)`.
    #[inline]
    pub fn project_to_camera(&self, world_point: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.rotation * (world_point - self.center))
    }

    /// Transform a camera-frame point back to world coordinates:
    /// `X_world = R⁻¹ * X_cam + C`.
    #[inline]
    pub fn transform_to_world(&self, camera_point: &Point3<f64>) -> Point3<f64> {
        self.rotation.inverse() * camera_point + self.center.coords
    }
}

impl<'de> Deserialize<'de> for WorldToCameraPose {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            rotation: Rotation3<f64>,
            center: Point3<f64>,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::from_rotation_center(repr.rotation, repr.center).map_err(serde::de::Error::custom)
    }
}

/// The identity pose, [`WorldToCameraPose::identity`]: the camera at the world origin, with
/// its axes along the world axes.
impl Default for WorldToCameraPose {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_pose_rejects_invalid_geometry() {
        assert!(
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(f64::NAN, 0.0, 0.0)
            )
            .is_err()
        );

        let scaled = Matrix3::identity() * 2.0;
        assert!(WorldToCameraPose::from_matrix(scaled, Point3::origin()).is_err());

        let overflowing_center = Point3::new(f64::MAX, f64::MAX, 0.0);
        let mixing_rotation = Rotation3::from_euler_angles(0.0, 0.0, 0.25);
        assert!(
            WorldToCameraPose::from_rotation_center(mixing_rotation, overflowing_center).is_err()
        );
    }

    #[test]
    fn isometry_preserves_center_for_accepted_rotation_roundoff() {
        let center = Point3::new(1e12, 0.0, 0.0);
        let pose =
            WorldToCameraPose::from_matrix(Matrix3::identity() * (1.0 + 1e-10), center).unwrap();
        let isometry = pose.to_isometry();
        assert!(isometry.transform_point(&center).coords.norm() < 1e-6);
        assert!((isometry.rotation.norm() - 1.0).abs() < 1e-15);
        // The accepted source representation is preserved for serialization.
        assert_eq!(
            pose.rotation_matrix(),
            &(Matrix3::identity() * (1.0 + 1e-10))
        );
    }

    #[test]
    fn camera_pose_deserialization_revalidates_geometry() {
        let valid = WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.1, -0.2, 0.3),
            Point3::new(1.0, 2.0, 3.0),
        )
        .unwrap();
        let encoded = serde_json::to_value(&valid).unwrap();
        let decoded: WorldToCameraPose = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, valid);

        let mut invalid_rotation = encoded.clone();
        invalid_rotation["rotation"] =
            serde_json::json!([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        assert!(serde_json::from_value::<WorldToCameraPose>(invalid_rotation).is_err());

        let mut overflowing_translation = encoded;
        overflowing_translation["center"] = serde_json::json!([f64::MAX, f64::MAX, 0.0]);
        assert!(serde_json::from_value::<WorldToCameraPose>(overflowing_translation).is_err());

        let malformed = serde_json::json!({
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        });
        assert!(serde_json::from_value::<WorldToCameraPose>(malformed).is_err());
    }
}
