// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for robust geometry estimation.

use thiserror::Error;

/// Errors that can occur during robust estimation.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum RobustError {
    /// Not enough data points for the chosen algorithm.
    #[error("insufficient data: need at least {required} points, got {provided}")]
    InsufficientData {
        /// Minimum points required.
        required: usize,
        /// Points actually provided.
        provided: usize,
    },

    /// No valid model was found after all iterations.
    #[error("no valid model found after {iterations} iterations: {reason}")]
    NoValidModel {
        /// Number of iterations performed.
        iterations: usize,
        /// Description of the failure.
        reason: String,
    },

    /// All samples resulted in degenerate configurations.
    #[error("all {total_samples} samples resulted in degenerate configurations")]
    AllSamplesDegenerate {
        /// Total samples attempted.
        total_samples: usize,
    },

    /// A valid model was found, but its consensus was too small.
    #[error("insufficient consensus: need at least {required} inliers, found {found}")]
    InsufficientConsensus {
        /// Minimum accepted consensus size.
        required: usize,
        /// Consensus size of the selected model.
        found: usize,
    },

    /// A solver returned an error, most often because it rejected its input; `reason` holds the
    /// solver's message.
    #[error("solver failed: {reason}")]
    SolverFailed {
        /// Description of the solver failure.
        reason: String,
    },

    /// Invalid configuration.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Description of the configuration problem.
        reason: String,
    },

    /// Numerical error during estimation.
    #[error("numerical error: {reason}")]
    Numerical {
        /// Description of the numerical error.
        reason: String,
    },
}

/// Convert from solver errors.
impl From<crate::SolverError> for RobustError {
    fn from(e: crate::SolverError) -> Self {
        RobustError::SolverFailed {
            reason: e.to_string(),
        }
    }
}
