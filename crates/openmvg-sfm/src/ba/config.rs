// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral bundle-adjustment configuration.

use std::collections::BTreeSet;

use openmvg_ba::{LinearSolver, ResidualScale, RobustLoss};
use openmvg_core::PixelThreshold;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Result, SfmError},
    ids::{LandmarkId, ViewId},
};

/// Configuration shared by SfM bundle-adjustment backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleAdjustConfig {
    scope: OptimizationScope,
    max_iterations: usize,
    function_tolerance: f64,
    gradient_tolerance: f64,
    parameter_tolerance: f64,
    #[serde(default, with = "robust_loss_serde")]
    robust_loss: RobustLoss,
    #[serde(default, with = "linear_solver_serde")]
    linear_solver: LinearSolver,
    fixed_views: BTreeSet<ViewId>,
    fixed_points: BTreeSet<LandmarkId>,
}

config_accessors!(BundleAdjustConfig {
    /// What to optimize.
    copy scope, with_scope: OptimizationScope;
    /// Maximum nonlinear iterations.
    copy max_iterations, with_max_iterations: usize;
    /// Relative objective-reduction tolerance for convergence.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy function_tolerance, with_function_tolerance: f64;
    /// Gradient-norm tolerance for convergence.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy gradient_tolerance, with_gradient_tolerance: f64;
    /// Parameter-step tolerance for convergence.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy parameter_tolerance, with_parameter_tolerance: f64;
    /// Robust loss function; `None` means squared loss.
    copy robust_loss, with_robust_loss: RobustLoss;
    /// Linear solver for the reduced camera system.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy linear_solver, with_linear_solver: LinearSolver;
    /// Views whose assigned pose blocks must remain fixed.
    ///
    /// Selecting one view fixes every view that shares its [`crate::PoseId`].
    ref fixed_views, with_fixed_views: BTreeSet<ViewId>;
    /// Landmarks whose positions must remain fixed.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    ref fixed_points, with_fixed_points: BTreeSet<LandmarkId>;
});

impl Default for BundleAdjustConfig {
    fn default() -> Self {
        Self {
            scope: OptimizationScope::PosesAndPoints,
            max_iterations: 50,
            function_tolerance: 1e-6,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-8,
            // Huber is a safe general default. A pipeline stage may opt into a
            // redescending loss explicitly.
            robust_loss: RobustLoss::Huber(ResidualScale::Pixels(PixelThreshold::ONE_PIXEL)),
            linear_solver: LinearSolver::Auto,
            fixed_views: BTreeSet::new(),
            fixed_points: BTreeSet::new(),
        }
    }
}

impl BundleAdjustConfig {
    /// Validate optimizer limits.
    ///
    /// Robust-loss scales need no check here: [`ResidualScale`] only holds checked thresholds.
    pub fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(SfmError::InvalidInput(
                "bundle adjustment max_iterations must be positive".to_owned(),
            ));
        }
        for (name, value) in [
            ("function_tolerance", self.function_tolerance),
            ("gradient_tolerance", self.gradient_tolerance),
            ("parameter_tolerance", self.parameter_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SfmError::InvalidInput(format!(
                    "bundle adjustment {name} must be finite and positive"
                )));
            }
        }
        Ok(())
    }
}

/// Parameter blocks optimized by bundle adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationScope {
    /// Optimize camera poses and 3D points.
    PosesAndPoints,
    /// Optimize camera translations and points while keeping rotations fixed.
    TranslationsAndPoints,
    /// Optimize camera poses only.
    PosesOnly,
    /// Optimize 3D points only.
    PointsOnly,
    /// Optimize poses, points, and variable intrinsics.
    SelfCalibration,
}

/// Serialized form of [`BundleAdjustConfig::robust_loss`].
///
/// `openmvg-ba` is serde-free, so the loss keeps the externally tagged shape of the former
/// `RobustLossType` enum: `"none"` for squared loss, `{"huber": x}` and `{"cauchy": x}` for pixel
/// scales, and `{"huber_radians": x}` and `{"cauchy_radians": x}` for angular scales. A missing
/// field reads as squared loss and `null` is rejected. Deserialized scales are checked like the
/// thresholds they become.
mod robust_loss_serde {
    use openmvg_ba::{ResidualScale, RobustLoss};
    use openmvg_core::{AngularThreshold, PixelThreshold, ThresholdResult};
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser};

    /// Wire mirror of the former `RobustLossType`; variant names and order are part of the format.
    #[derive(Serialize, Deserialize)]
    #[serde(rename = "RobustLossType", rename_all = "snake_case")]
    enum Wire {
        None,
        Huber(f64),
        Cauchy(f64),
        HuberRadians(f64),
        CauchyRadians(f64),
    }

    impl Wire {
        fn from_loss(loss: RobustLoss) -> Option<Self> {
            Some(match loss {
                RobustLoss::Squared => Self::None,
                RobustLoss::Huber(ResidualScale::Pixels(scale)) => Self::Huber(scale.value()),
                RobustLoss::Cauchy(ResidualScale::Pixels(scale)) => Self::Cauchy(scale.value()),
                RobustLoss::Huber(ResidualScale::Angular(scale)) => {
                    Self::HuberRadians(scale.value())
                }
                RobustLoss::Cauchy(ResidualScale::Angular(scale)) => {
                    Self::CauchyRadians(scale.value())
                }
                _ => return None,
            })
        }

        fn into_loss(self) -> ThresholdResult<RobustLoss> {
            let pixels = |value| PixelThreshold::try_new(value).map(ResidualScale::Pixels);
            let radians = |value| AngularThreshold::try_new(value).map(ResidualScale::Angular);
            Ok(match self {
                Self::None => RobustLoss::Squared,
                Self::Huber(value) => RobustLoss::Huber(pixels(value)?),
                Self::Cauchy(value) => RobustLoss::Cauchy(pixels(value)?),
                Self::HuberRadians(value) => RobustLoss::Huber(radians(value)?),
                Self::CauchyRadians(value) => RobustLoss::Cauchy(radians(value)?),
            })
        }
    }

    pub(super) fn serialize<S: Serializer>(
        loss: &RobustLoss,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Wire::from_loss(*loss)
            .ok_or_else(|| ser::Error::custom("robust loss has no serialized form"))?
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<RobustLoss, D::Error> {
        Wire::deserialize(deserializer)?
            .into_loss()
            .map_err(de::Error::custom)
    }
}

/// Serialized form of [`BundleAdjustConfig::linear_solver`].
///
/// `openmvg-ba` is serde-free, so the solver is written as `"auto"`, `"direct"`, or
/// `"iterative"`.
mod linear_solver_serde {
    use openmvg_ba::LinearSolver;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, ser};

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Wire {
        Auto,
        Direct,
        Iterative,
    }

    pub(super) fn serialize<S: Serializer>(
        solver: &LinearSolver,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let wire = match solver {
            LinearSolver::Auto => Wire::Auto,
            LinearSolver::Direct => Wire::Direct,
            LinearSolver::Iterative => Wire::Iterative,
            _ => return Err(ser::Error::custom("linear solver has no serialized form")),
        };
        wire.serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<LinearSolver, D::Error> {
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Auto => LinearSolver::Auto,
            Wire::Direct => LinearSolver::Direct,
            Wire::Iterative => LinearSolver::Iterative,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configuration_is_valid() {
        BundleAdjustConfig::default().validate().unwrap();
    }

    #[test]
    fn invalid_limits_are_rejected() {
        let config = BundleAdjustConfig {
            max_iterations: 0,
            ..BundleAdjustConfig::default()
        };
        assert!(config.validate().is_err());
    }

    fn with_loss(robust_loss: RobustLoss) -> BundleAdjustConfig {
        BundleAdjustConfig {
            robust_loss,
            ..BundleAdjustConfig::default()
        }
    }

    fn golden(robust_loss: &str) -> String {
        format!(
            "{{\"scope\":\"poses_and_points\",\"max_iterations\":50,\"function_tolerance\":1e-6,\
             \"gradient_tolerance\":1e-10,\"parameter_tolerance\":1e-8,\
             \"robust_loss\":{robust_loss},\"linear_solver\":\"auto\",\"fixed_views\":[],\
             \"fixed_points\":[]}}"
        )
    }

    #[test]
    fn robust_loss_keeps_the_robust_loss_type_json_shape() {
        let pixels = |value| ResidualScale::Pixels(PixelThreshold::try_new(value).unwrap());
        let radians =
            |value| ResidualScale::Angular(openmvg_core::AngularThreshold::try_new(value).unwrap());
        let cases = [
            (RobustLoss::Squared, "\"none\""),
            (RobustLoss::Huber(pixels(1.0)), "{\"huber\":1.0}"),
            (RobustLoss::Huber(pixels(4.0)), "{\"huber\":4.0}"),
            (RobustLoss::Cauchy(pixels(4.0)), "{\"cauchy\":4.0}"),
            (RobustLoss::Huber(radians(0.01)), "{\"huber_radians\":0.01}"),
            (
                RobustLoss::Cauchy(radians(0.005)),
                "{\"cauchy_radians\":0.005}",
            ),
        ];
        assert_eq!(
            serde_json::to_string(&BundleAdjustConfig::default()).unwrap(),
            golden("{\"huber\":1.0}")
        );
        for (loss, json) in cases {
            let serialized = serde_json::to_string(&with_loss(loss)).unwrap();
            assert_eq!(serialized, golden(json));
            let restored: BundleAdjustConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(restored.robust_loss, loss);
        }
    }

    #[test]
    fn robust_loss_deserialization_validates_scales_and_defaults_to_squared() {
        let mut value = serde_json::to_value(BundleAdjustConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("robust_loss");
        let missing: BundleAdjustConfig = serde_json::from_value(value).unwrap();
        assert_eq!(missing.robust_loss, RobustLoss::Squared);

        for invalid in [
            "null",
            "{\"huber\":0.0}",
            "{\"cauchy\":-1.0}",
            "{\"huber_radians\":4.0}",
            "{\"cauchy_radians\":0.0}",
            "{\"huber\":1e200}",
        ] {
            assert!(
                serde_json::from_str::<BundleAdjustConfig>(&golden(invalid)).is_err(),
                "{invalid} must be rejected"
            );
        }
    }
}
