// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared information diagnostics for nonlinear least squares.

#![deny(missing_docs)]

use super::api::Loss;

/// What stochastic assumptions produced local uncertainty information.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UncertaintyModel {
    /// Inverse weighted residual information, `(J^T W J)^-1`.
    ResidualLinearized,
    /// Inverse final IRLS information with robust weights held fixed.
    ///
    /// This is a local conditional approximation. It does not include random
    /// robust-weight variation and must not be interpreted as calibrated
    /// coverage under unknown outlier contamination.
    RobustFixedWeightLinearized,
    /// First-order L2 sandwich under isotropic image-coordinate noise.
    ImageNoiseSandwich,
}

impl UncertaintyModel {
    /// Return true when robust weights were frozen at their final values.
    pub const fn is_fixed_weight_approximation(self) -> bool {
        matches!(self, Self::RobustFixedWeightLinearized)
    }
}

/// Classify the uncertainty approximation implied by a loss.
// Only the oracle-exposed covariance kernels call it.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub const fn uncertainty_model_for_loss(loss: Loss) -> UncertaintyModel {
    match loss {
        Loss::L2 => UncertaintyModel::ResidualLinearized,
        Loss::Huber { .. } | Loss::Cauchy { .. } => UncertaintyModel::RobustFixedWeightLinearized,
    }
}

/// Rank and conditioning of final weighted local information.
///
/// `reciprocal_condition_number` describes a square-root information operator,
/// so it is comparable to conditioning of the weighted Jacobian rather than
/// its squared normal matrix. The value is zero for a rank-deficient system
/// and approaches zero as the local problem becomes ill-conditioned.
#[must_use = "rank and conditioning diagnostics should be inspected"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InformationDiagnostics {
    /// Local parameter dimension.
    pub dimension: usize,
    /// Numerically observable parameter rank.
    pub rank: usize,
    /// Number of scalar residual components.
    pub residual_count: usize,
    /// Reciprocal square-root-information condition estimate, or zero when singular.
    pub reciprocal_condition_number: f64,
}

impl InformationDiagnostics {
    /// Construct rank and conditioning diagnostics.
    pub const fn new(
        dimension: usize,
        rank: usize,
        residual_count: usize,
        reciprocal_condition_number: f64,
    ) -> Self {
        Self {
            dimension,
            rank,
            residual_count,
            reciprocal_condition_number,
        }
    }

    /// Return true exactly when every local parameter is observable.
    pub fn is_full_rank(self) -> bool {
        self.rank == self.dimension
    }
}
