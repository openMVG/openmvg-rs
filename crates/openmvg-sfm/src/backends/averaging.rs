// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated rotation and camera-center estimates returned by averaging, and the averaging
//! entry points a custom composition can substitute.

use nalgebra::{Rotation3, Vector3};
use openmvg_core::{
    AngularResidual,
    model::{EstimationResult, RelativeRotation, RelativeTranslation},
};
use openmvg_geometry::averaging::{
    RotationConfig, RotationEstimate, TranslationConfig, TranslationEstimate,
};

use crate::{Result, SfmError, ViewId};

/// Rotation-averaging entry point accepted by unstable custom SfM compositions.
pub type RotationAveragingFn =
    fn(&[RelativeRotation], &RotationConfig) -> EstimationResult<RotationEstimate>;

/// Translation-averaging entry point accepted by unstable custom SfM compositions.
pub type TranslationAveragingFn = fn(
    &[RelativeTranslation],
    &std::collections::BTreeMap<ViewId, Rotation3<f64>>,
    &TranslationConfig,
) -> EstimationResult<TranslationEstimate>;
/// Validated absolute world-to-camera rotations from rotation averaging, sorted by view.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RotationEstimates {
    rotations: Vec<(ViewId, Rotation3<f64>)>,
    /// Iterations the averaging solver performed.
    pub iterations: usize,
    /// Final mean angular residual reported by the averaging solver.
    pub residual: AngularResidual,
}

/// Validated world-frame camera centers from translation averaging, sorted by view.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CameraCenters {
    centers: Vec<(ViewId, Vector3<f64>)>,
    /// Iterations the averaging solver performed.
    pub iterations: usize,
    /// Final mean angular residual reported by the averaging solver.
    pub residual: AngularResidual,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl CameraCenters {
    /// Construct a checked camera-center estimate.
    ///
    /// Image identifiers must be unique, coordinates and residuals must be
    /// finite, and residuals must be non-negative. Coverage of a particular
    /// solve domain is checked separately at the pipeline boundary.
    pub fn try_new(
        mut centers: Vec<(ViewId, Vector3<f64>)>,
        iterations: usize,
        residual: AngularResidual,
    ) -> Result<Self> {
        centers.sort_by_key(|(view_id, _)| *view_id);
        let estimates = Self {
            centers,
            iterations,
            residual,
        };
        estimates.validate()?;
        Ok(estimates)
    }

    /// World-frame camera centers, in strictly increasing view order.
    pub fn centers(&self) -> &[(ViewId, Vector3<f64>)] {
        &self.centers
    }

    /// Center of `view_id`, or `None` if that view has no estimate.
    pub fn get(&self, view_id: ViewId) -> Option<&Vector3<f64>> {
        self.centers
            .binary_search_by_key(&view_id, |(id, _)| *id)
            .ok()
            .map(|index| &self.centers[index].1)
    }

    pub(crate) fn validate_dense_domain(&self, view_count: usize) -> Result<()> {
        self.validate()?;
        if self.centers.len() != view_count {
            return Err(SfmError::InvalidInput(format!(
                "translation averaging returned {} centers for {view_count} views",
                self.centers.len()
            )));
        }
        for (expected, (actual, _)) in self.centers.iter().enumerate() {
            if actual.index() != expected {
                return Err(SfmError::InvalidInput(format!(
                    "translation averaging returned unexpected view {actual}; expected dense view {expected}"
                )));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        for pair in self.centers.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(SfmError::InvalidInput(format!(
                    "camera-center estimates must contain strictly ordered unique view IDs near {}",
                    pair[0].0
                )));
            }
        }
        if let Some((image, _)) = self
            .centers
            .iter()
            .find(|(_, center)| center.iter().any(|value| !value.is_finite()))
        {
            return Err(SfmError::InvalidInput(format!(
                "camera-center estimate for view {image} is non-finite"
            )));
        }
        Ok(())
    }
}

impl RotationEstimates {
    /// Construct a checked global-rotation estimate.
    ///
    /// Image identifiers must be unique, every matrix must be a finite proper
    /// rotation, and the reported residual must be finite and non-negative.
    pub fn try_new(
        mut rotations: Vec<(ViewId, Rotation3<f64>)>,
        iterations: usize,
        residual: AngularResidual,
    ) -> Result<Self> {
        rotations.sort_by_key(|(view_id, _)| *view_id);
        let estimates = Self {
            rotations,
            iterations,
            residual,
        };
        estimates.validate()?;
        Ok(estimates)
    }

    /// World-to-camera rotations, in strictly increasing view order.
    pub fn rotations(&self) -> &[(ViewId, Rotation3<f64>)] {
        &self.rotations
    }

    /// Rotation of `view_id`, or `None` if that view has no estimate.
    pub fn get(&self, view_id: ViewId) -> Option<&Rotation3<f64>> {
        self.rotations
            .binary_search_by_key(&view_id, |(id, _)| *id)
            .ok()
            .map(|index| &self.rotations[index].1)
    }

    pub(crate) fn validate_dense_domain(&self, view_count: usize) -> Result<()> {
        self.validate()?;
        if self.rotations.len() != view_count {
            return Err(SfmError::InvalidInput(format!(
                "rotation averaging returned {} rotations for {view_count} views",
                self.rotations.len()
            )));
        }
        for (expected, (actual, _)) in self.rotations.iter().enumerate() {
            if actual.index() != expected {
                return Err(SfmError::InvalidInput(format!(
                    "rotation averaging returned unexpected view {actual}; expected dense view {expected}"
                )));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        for pair in self.rotations.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(SfmError::InvalidInput(format!(
                    "rotation estimates must contain strictly ordered unique view IDs near {}",
                    pair[0].0
                )));
            }
        }
        for (image, rotation) in &self.rotations {
            let matrix = rotation.matrix();
            let orthogonality_error =
                (matrix.transpose() * matrix - nalgebra::Matrix3::identity()).norm();
            if matrix.iter().any(|value| !value.is_finite())
                || !orthogonality_error.is_finite()
                || orthogonality_error > 1e-6
                || (matrix.determinant() - 1.0).abs() > 1e-6
            {
                return Err(SfmError::InvalidInput(format!(
                    "rotation estimate for view {image} is not a finite proper rotation"
                )));
            }
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RotationEstimatesWire {
    rotations: Vec<(ViewId, Rotation3<f64>)>,
    iterations: usize,
    residual: AngularResidual,
}

impl<'de> serde::Deserialize<'de> for RotationEstimates {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RotationEstimatesWire::deserialize(deserializer)?;
        Self::try_new(wire.rotations, wire.iterations, wire.residual)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CameraCentersWire {
    centers: Vec<(ViewId, Vector3<f64>)>,
    iterations: usize,
    residual: AngularResidual,
}

impl<'de> serde::Deserialize<'de> for CameraCenters {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CameraCentersWire::deserialize(deserializer)?;
        Self::try_new(wire.centers, wire.iterations, wire.residual)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averaging_estimates_reject_duplicate_ids_and_typed_residuals_reject_invalid_scalars() {
        let duplicate_rotations = vec![
            (ViewId::new(0), Rotation3::identity()),
            (ViewId::new(0), Rotation3::identity()),
        ];
        assert!(RotationEstimates::try_new(duplicate_rotations, 0, AngularResidual::ZERO).is_err());
        assert!(AngularResidual::try_new(f64::NAN).is_err());

        let duplicate_centers = vec![
            (ViewId::new(0), Vector3::zeros()),
            (ViewId::new(0), Vector3::x()),
        ];
        assert!(CameraCenters::try_new(duplicate_centers, 0, AngularResidual::ZERO).is_err());
        assert!(AngularResidual::try_new(-1.0).is_err());
    }

    #[test]
    fn averaging_estimates_reject_invalid_geometry() {
        let reflection = Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
        ));
        assert!(
            RotationEstimates::try_new(
                vec![(ViewId::new(0), reflection)],
                0,
                AngularResidual::ZERO,
            )
            .is_err()
        );
        assert!(
            CameraCenters::try_new(
                vec![(ViewId::new(0), Vector3::new(f64::NAN, 0.0, 0.0))],
                0,
                AngularResidual::ZERO,
            )
            .is_err()
        );
    }

    #[test]
    fn averaging_estimates_enforce_the_dense_solver_domain() {
        let rotations = RotationEstimates::try_new(
            vec![
                (ViewId::new(1), Rotation3::identity()),
                (ViewId::new(2), Rotation3::identity()),
            ],
            0,
            AngularResidual::ZERO,
        )
        .unwrap();
        let centers = CameraCenters::try_new(
            vec![
                (ViewId::new(0), Vector3::zeros()),
                (ViewId::new(2), Vector3::x()),
            ],
            0,
            AngularResidual::ZERO,
        )
        .unwrap();

        assert!(rotations.validate_dense_domain(2).is_err());
        assert!(centers.validate_dense_domain(2).is_err());
    }
}
