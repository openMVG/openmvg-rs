// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared values and crate-private implementation contracts for averaging.

use std::collections::BTreeMap;

use nalgebra::{Rotation3, Vector3};
use openmvg_core::{
    AngularResidual, AngularThreshold,
    model::{EstimationError, EstimationResult, RelativeRotation, RelativeTranslation, ViewId},
};

/// Configuration shared by rotation-averaging algorithms.
#[derive(Debug, Clone)]
pub struct RotationConfig {
    max_iterations: usize,
}

config_accessors!(RotationConfig {
    /// Maximum iterations of the selected rotation solver.
    ///
    /// The robust wrapper caps outer weighted solves at `min(max_iterations, 20)`.
    copy max_iterations, with_max_iterations: usize;
});

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
        }
    }
}

impl RotationConfig {
    /// Validate iteration limits.
    pub fn validate(&self) -> EstimationResult<()> {
        validate_iteration_limit(self.max_iterations, "rotation")
    }
}

/// Absolute rotations estimated by rotation averaging.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RotationEstimate {
    /// Absolute rotations in deterministic `ViewId` order.
    pub rotations: BTreeMap<ViewId, Rotation3<f64>>,
    /// Number of iterations performed by the selected rotation solver.
    ///
    /// Robust averaging reports outer weighted-solve calls, without accumulating
    /// the inner solver's iteration counts.
    pub iteration_count: usize,
    /// Final mean angular residual.
    pub residual: AngularResidual,
}

impl RotationEstimate {
    /// Assemble an estimate, for example from a custom rotation-averaging function.
    #[must_use]
    pub fn new(
        rotations: BTreeMap<ViewId, Rotation3<f64>>,
        iteration_count: usize,
        residual: AngularResidual,
    ) -> Self {
        Self {
            rotations,
            iteration_count,
            residual,
        }
    }
}

/// Configuration shared by translation-averaging algorithms.
#[derive(Debug, Clone)]
pub struct TranslationConfig {
    max_iterations: usize,
    tolerance: AngularThreshold,
}

config_accessors!(TranslationConfig {
    /// Maximum iterations.
    copy max_iterations, with_max_iterations: usize;
    /// Convergence threshold.
    copy tolerance, with_tolerance: AngularThreshold;
});

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: AngularThreshold::ONE_MICRORADIAN,
        }
    }
}

impl TranslationConfig {
    /// Validate iteration limits.
    pub fn validate(&self) -> EstimationResult<()> {
        validate_iteration_limit(self.max_iterations, "translation")
    }
}

/// Camera centers estimated by translation averaging.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TranslationEstimate {
    /// Absolute camera centers in deterministic `ViewId` order.
    pub translations: BTreeMap<ViewId, Vector3<f64>>,
    /// Number of iterations performed.
    pub iteration_count: usize,
    /// Final mean angular residual.
    pub residual: AngularResidual,
}

impl TranslationEstimate {
    /// Assemble an estimate, for example from a custom translation-averaging function.
    #[must_use]
    pub fn new(
        translations: BTreeMap<ViewId, Vector3<f64>>,
        iteration_count: usize,
        residual: AngularResidual,
    ) -> Self {
        Self {
            translations,
            iteration_count,
            residual,
        }
    }
}

fn validate_iteration_limit(max_iterations: usize, name: &str) -> EstimationResult<()> {
    if max_iterations == 0 {
        return Err(EstimationError::InvalidInput(format!(
            "{name} max_iterations must be positive"
        )));
    }
    Ok(())
}

/// Internal common shape implemented by the rotation algorithms.
pub(crate) trait RotationSolver {
    /// State kept between solves over one edge list whose weights change, as in IRLS.
    type Workspace: Default;

    fn solve(
        &self,
        relative_rotations: &[RelativeRotation],
        config: &RotationConfig,
    ) -> EstimationResult<RotationEstimate>;

    /// [`Self::solve`] with state reused from earlier calls. It must give the same bits as
    /// `solve`; a solver with no reusable state keeps this default.
    fn solve_in(
        &self,
        relative_rotations: &[RelativeRotation],
        config: &RotationConfig,
        _workspace: &mut Self::Workspace,
    ) -> EstimationResult<RotationEstimate> {
        self.solve(relative_rotations, config)
    }
}

/// Internal common shape implemented by the translation algorithms.
pub(crate) trait TranslationSolver {
    fn solve(
        &self,
        relative_translations: &[RelativeTranslation],
        absolute_rotations: &BTreeMap<ViewId, Rotation3<f64>>,
        config: &TranslationConfig,
    ) -> EstimationResult<TranslationEstimate>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averaging_configs_reject_empty_iteration_limits() {
        assert!(
            RotationConfig::default()
                .with_max_iterations(0)
                .validate()
                .is_err()
        );
        assert!(
            TranslationConfig::default()
                .with_max_iterations(0)
                .validate()
                .is_err()
        );
    }
}
