// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Shared SfM values
//!
//! Validated relative-pose measurements shared by geometry and SfM.
//!
//! ## Determinism
//!
//! Estimators built on these values are deterministic: identical inputs,
//! including RANSAC seeds, produce identical outputs.

use crate::camera::try_normalize_bearing;
use crate::model::ids::ViewId;
use crate::model::state::validate_rotation;
use nalgebra::{Rotation3, Vector3};
use serde::{Deserialize, Serialize};

/// Rotation angle in degrees, in `[0, 180]` and always finite for a finite rotation.
///
/// Prefer it to nalgebra's `Rotation3::angle` for errors: that can return NaN for a rotation
/// within rounding of the identity, which drops the smallest errors from a statistic that skips
/// non-finite values.
#[must_use]
pub fn rotation_angle_deg(rotation: &Rotation3<f64>) -> f64 {
    let m = rotation.matrix();
    ((m[(0, 0)] + m[(1, 1)] + m[(2, 2)] - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Error of the relative-measurement constructors and of the `openmvg-geometry` averaging and
/// triangulation functions.
///
/// The averaging functions (rotation, translation, global positioning, and view-graph
/// conditioning) and the triangulation functions return it, as do [`RelativePose`],
/// [`RelativeRotation`], and [`RelativeTranslation`] when their input is invalid. The minimal
/// solvers of `openmvg-geometry` return its `SolverError` instead, and its robust estimators
/// its `RobustError`.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum EstimationError {
    /// Insufficient input data.
    #[error("insufficient data: need {required}, got {provided}")]
    InsufficientData {
        /// Minimum number of inputs required by the estimator.
        required: usize,
        /// Number of inputs supplied by the caller.
        provided: usize,
    },

    /// Degenerate configuration.
    #[error("degenerate configuration: {reason}")]
    DegenerateConfiguration {
        /// Human-readable description of the detected degeneracy.
        reason: String,
    },

    /// Numerical issues (singular matrix, etc.).
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// View not found.
    #[error("view {0} not found")]
    ViewNotFound(ViewId),

    /// Invalid input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Result of a solver or estimator.
pub type EstimationResult<T> = Result<T, EstimationError>;

// ============================================================================
// Relative Pose Types
// ============================================================================

/// A directed relative camera pose from source camera `i` to target camera `j`.
///
/// The unit translation is the target-camera-frame projection direction:
/// `x_j = R_ji * x_i + scale * t_hat_ji` and `E_ji = [t_hat_ji]_x R_ji`.
///
/// The serialized form stores both fields as they are, so a pose reloads bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RelativePose {
    rotation_source_to_target: Rotation3<f64>,
    translation_source_to_target: nalgebra::UnitVector3<f64>,
}

impl RelativePose {
    /// Creates a checked source-to-target pose.
    pub fn from_projection_translation(
        rotation_source_to_target: Rotation3<f64>,
        translation_source_to_target: nalgebra::UnitVector3<f64>,
    ) -> EstimationResult<Self> {
        validate_relative_rotation(&rotation_source_to_target)?;
        validate_relative_direction(translation_source_to_target.as_ref())?;
        Ok(Self {
            rotation_source_to_target,
            translation_source_to_target,
        })
    }

    /// Creates a pose from the source-frame direction between camera centers.
    pub fn from_baseline_direction(
        rotation_source_to_target: Rotation3<f64>,
        baseline_direction_in_source: nalgebra::UnitVector3<f64>,
    ) -> EstimationResult<Self> {
        let translation_source_to_target = nalgebra::Unit::new_normalize(
            -(rotation_source_to_target * baseline_direction_in_source.as_ref()),
        );
        Self::from_projection_translation(rotation_source_to_target, translation_source_to_target)
    }

    /// Revalidates the rotation and unit translation invariants.
    pub fn validate(&self) -> EstimationResult<()> {
        validate_relative_rotation(&self.rotation_source_to_target)?;
        validate_relative_direction(self.translation_source_to_target.as_ref())?;
        let norm = self.translation_source_to_target.norm();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-8 {
            return Err(EstimationError::InvalidInput(
                "relative pose translation must be normalized".to_owned(),
            ));
        }
        Ok(())
    }

    /// Rotation `R_ji` from the source camera frame to the target camera frame.
    #[must_use]
    pub const fn rotation_source_to_target(&self) -> &Rotation3<f64> {
        &self.rotation_source_to_target
    }

    /// Unit projection translation `t_hat_ji` expressed in the target camera frame.
    #[must_use]
    pub const fn translation_source_to_target(&self) -> &nalgebra::UnitVector3<f64> {
        &self.translation_source_to_target
    }

    /// Unit camera-baseline direction from source to target, expressed in the source frame.
    #[must_use]
    pub fn baseline_direction_in_source(&self) -> nalgebra::UnitVector3<f64> {
        nalgebra::Unit::new_normalize(
            -(self.rotation_source_to_target.inverse()
                * self.translation_source_to_target.as_ref()),
        )
    }

    /// Constructs the unit-scale projection matrix `[R_ji | t_hat_ji]`.
    #[must_use]
    pub fn projection_matrix(&self) -> nalgebra::Matrix3x4<f64> {
        let mut projection = nalgebra::Matrix3x4::zeros();
        projection
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.rotation_source_to_target.matrix());
        projection
            .column_mut(3)
            .copy_from(self.translation_source_to_target.as_ref());
        projection
    }
}

impl<'de> Deserialize<'de> for RelativePose {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            rotation_source_to_target: Rotation3<f64>,
            translation_source_to_target: nalgebra::UnitVector3<f64>,
        }

        let repr = Repr::deserialize(deserializer)?;
        let pose = Self {
            rotation_source_to_target: repr.rotation_source_to_target,
            translation_source_to_target: repr.translation_source_to_target,
        };
        // A stored unit vector is not renormalized, so its norm is checked too.
        pose.validate().map_err(serde::de::Error::custom)?;
        Ok(pose)
    }
}

fn validate_relative_views(view_i: ViewId, view_j: ViewId) -> EstimationResult<()> {
    if view_i == view_j {
        return Err(EstimationError::InvalidInput(
            "relative measurement requires two distinct views".to_owned(),
        ));
    }
    Ok(())
}

fn validate_relative_rotation(rotation: &Rotation3<f64>) -> EstimationResult<()> {
    if validate_rotation(rotation).is_err() {
        return Err(EstimationError::InvalidInput(
            "relative rotation must be a finite proper rotation".to_owned(),
        ));
    }
    Ok(())
}

fn validate_relative_direction(direction: &Vector3<f64>) -> EstimationResult<()> {
    if !direction.iter().all(|value| value.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "relative direction must be finite".to_owned(),
        ));
    }
    if direction.iter().all(|value| *value == 0.0) {
        return Err(EstimationError::InvalidInput(
            "relative direction must have non-zero norm".to_owned(),
        ));
    }
    Ok(())
}

fn normalize_relative_direction(direction: Vector3<f64>) -> EstimationResult<Vector3<f64>> {
    validate_relative_direction(&direction)?;
    try_normalize_bearing(&direction).map_err(|_| {
        EstimationError::InvalidInput("relative direction must be finite and non-zero".to_owned())
    })
}

/// Relative rotation between two views (for rotation averaging).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RelativeRotation {
    /// Source view ID.
    view_i: ViewId,
    /// Target view ID.
    view_j: ViewId,
    /// Relative rotation R_ij such that R_j = R_ij * R_i.
    rotation: Rotation3<f64>,
    /// Information/confidence weight used linearly by averaging objectives.
    ///
    /// For example, an edge with weight `2` contributes twice the objective
    /// of an otherwise identical edge with weight `1`. This is not a
    /// square-root information value.
    weight: f64,
}

impl RelativeRotation {
    /// Create a validated relative rotation.
    pub fn try_new(
        view_i: ViewId,
        view_j: ViewId,
        rotation: Rotation3<f64>,
        weight: f64,
    ) -> EstimationResult<Self> {
        validate_relative_views(view_i, view_j)?;
        validate_relative_rotation(&rotation)?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EstimationError::InvalidInput(
                "relative rotation weight must be finite and positive".to_owned(),
            ));
        }
        Ok(Self {
            view_i,
            view_j,
            rotation,
            weight,
        })
    }

    /// Validates this weighted relative rotation.
    pub fn validate(&self) -> EstimationResult<()> {
        validate_relative_views(self.view_i, self.view_j)?;
        validate_relative_rotation(&self.rotation)?;
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(EstimationError::InvalidInput(
                "relative rotation weight must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }

    /// Source view identifier.
    #[inline]
    #[must_use]
    pub const fn view_i(&self) -> ViewId {
        self.view_i
    }

    /// Target view identifier.
    #[inline]
    #[must_use]
    pub const fn view_j(&self) -> ViewId {
        self.view_j
    }

    /// Relative rotation from the source view to the target view.
    #[inline]
    #[must_use]
    pub const fn rotation(&self) -> Rotation3<f64> {
        self.rotation
    }

    /// Information/confidence weight used linearly by averaging objectives.
    #[inline]
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Return this measurement with a replacement relative rotation.
    pub fn with_rotation(&self, rotation: Rotation3<f64>) -> EstimationResult<Self> {
        Self::try_new(self.view_i, self.view_j, rotation, self.weight)
    }

    /// Return this measurement with a replacement confidence weight.
    pub fn with_weight(&self, weight: f64) -> EstimationResult<Self> {
        Self::try_new(self.view_i, self.view_j, self.rotation, weight)
    }
}

impl<'de> Deserialize<'de> for RelativeRotation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            view_i: ViewId,
            view_j: ViewId,
            rotation: Rotation3<f64>,
            weight: f64,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::try_new(repr.view_i, repr.view_j, repr.rotation, repr.weight)
            .map_err(serde::de::Error::custom)
    }
}

/// Weighted relative translation direction between two views, the input of translation
/// averaging.
///
/// The direction points from the source camera center to the target camera center and is
/// expressed in the source camera frame: `R_i (c_j - c_i)`, normalized, where `R_i` is the
/// source camera's world-to-camera rotation and `c_i`, `c_j` are the camera centers. This is
/// what [`RelativePose::baseline_direction_in_source`] returns, not the projection translation
/// `t_hat_ji` of a [`RelativePose`], which is expressed in the target camera frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RelativeTranslation {
    /// Source view ID.
    view_i: ViewId,
    /// Target view ID.
    view_j: ViewId,
    /// Translation direction (unit vector).
    direction: Vector3<f64>,
    /// Weight for averaging.
    weight: f64,
}

impl RelativeTranslation {
    /// Create a validated relative translation.
    pub fn try_new(
        view_i: ViewId,
        view_j: ViewId,
        direction: Vector3<f64>,
        weight: f64,
    ) -> EstimationResult<Self> {
        validate_relative_views(view_i, view_j)?;
        let direction = normalize_relative_direction(direction)?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EstimationError::InvalidInput(
                "relative translation weight must be finite and positive".to_owned(),
            ));
        }
        Ok(Self {
            view_i,
            view_j,
            direction,
            weight,
        })
    }

    /// Validates this weighted unit direction.
    pub fn validate(&self) -> EstimationResult<()> {
        validate_relative_views(self.view_i, self.view_j)?;
        validate_relative_direction(&self.direction)?;
        let norm = self.direction.norm();
        if (norm - 1.0).abs() > 1e-8 {
            return Err(EstimationError::InvalidInput(
                "relative translation direction must be normalized".to_owned(),
            ));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(EstimationError::InvalidInput(
                "relative translation weight must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }

    /// Source view identifier.
    #[inline]
    #[must_use]
    pub const fn view_i(&self) -> ViewId {
        self.view_i
    }

    /// Target view identifier.
    #[inline]
    #[must_use]
    pub const fn view_j(&self) -> ViewId {
        self.view_j
    }

    /// Unit direction from the source camera center to the target camera center, in the source
    /// camera frame.
    #[inline]
    #[must_use]
    pub const fn direction(&self) -> Vector3<f64> {
        self.direction
    }

    /// Information/confidence weight used by the averaging objective.
    #[inline]
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

impl<'de> Deserialize<'de> for RelativeTranslation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            view_i: ViewId,
            view_j: ViewId,
            direction: Vector3<f64>,
            weight: f64,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::try_new(repr.view_i, repr.view_j, repr.direction, repr.weight)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rotation_angle_is_finite_where_the_unclamped_trace_rounds_above_three() {
        let almost_identity =
            Rotation3::from_matrix_unchecked(nalgebra::Matrix3::identity() * (1.0 + 1.0e-15));
        assert_eq!(rotation_angle_deg(&almost_identity), 0.0);
        let half_turn = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI);
        assert!((rotation_angle_deg(&half_turn) - 180.0).abs() < 1e-9);
        let quarter = Rotation3::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        assert!((rotation_angle_deg(&quarter) - 90.0).abs() < 1e-9);
    }

    use super::*;

    #[test]
    fn relative_direction_normalization_is_scale_invariant() {
        let base = Vector3::new(1.0, -0.5, 0.25);
        let expected = base.normalize();
        for scale in [f64::MIN_POSITIVE, 1.0, f64::MAX] {
            let translation =
                RelativeTranslation::try_new(ViewId::new(0), ViewId::new(1), base * scale, 1.0)
                    .unwrap();
            assert!((translation.direction() - expected).norm() < 1e-14);
        }
    }

    #[test]
    fn fallible_relative_measurements_reject_invalid_numerics() {
        assert!(
            RelativeTranslation::try_new(ViewId::new(0), ViewId::new(1), Vector3::zeros(), 1.0,)
                .is_err()
        );
        assert!(
            RelativeTranslation::try_new(ViewId::new(0), ViewId::new(1), Vector3::x(), f64::NAN,)
                .is_err()
        );
        assert!(
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(0), Rotation3::identity(), 1.0,)
                .is_err()
        );
        assert!(
            RelativeRotation::try_new(ViewId::new(0), ViewId::new(1), Rotation3::identity(), 0.0,)
                .is_err()
        );
    }

    #[test]
    fn relative_measurement_deserialization_revalidates_invariants() {
        let invalid_rotation = r#"{
            "view_i": 0,
            "view_j": 1,
            "rotation": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            "weight": 0.0
        }"#;
        assert!(serde_json::from_str::<RelativeRotation>(invalid_rotation).is_err());

        let invalid_translation = r#"{
            "view_i": 0,
            "view_j": 1,
            "direction": [0.0,0.0,0.0],
            "weight": 1.0
        }"#;
        assert!(serde_json::from_str::<RelativeTranslation>(invalid_translation).is_err());

        let pose =
            RelativePose::from_projection_translation(Rotation3::identity(), Vector3::x_axis())
                .unwrap();
        let mut value = serde_json::to_value(pose).unwrap();
        assert_eq!(
            serde_json::from_value::<RelativePose>(value.clone()).unwrap(),
            pose
        );
        value["translation_source_to_target"] = serde_json::json!([2.0, 0.0, 0.0]);
        assert!(serde_json::from_value::<RelativePose>(value).is_err());
    }
}
