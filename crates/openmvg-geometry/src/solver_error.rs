// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for geometry solver operations.
//!
//! One error type shared by every solver, with one variant per failure mode.

use thiserror::Error;

/// Error returned by geometric model estimation.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum SolverError {
    /// Input values violate the solver contract.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Static description of the violated contract.
        reason: &'static str,
    },

    /// Too few data points for the algorithm.
    #[error("insufficient data: need {required} points, got {provided}")]
    InsufficientData {
        /// Minimum number of points required.
        required: usize,
        /// Number of points provided.
        provided: usize,
    },

    /// A minimal solver received more data than its exact sample size.
    ///
    /// Use an overdetermined solver type when fitting all available data.
    #[error("invalid sample size: expected exactly {expected} points, got {provided}")]
    InvalidSampleSize {
        /// Exact number of points consumed by the minimal solver.
        expected: usize,
        /// Number of points provided.
        provided: usize,
    },

    /// A sampled datum index lies outside the supplied data slice.
    #[error("invalid data index {index}: data length is {data_len}")]
    InvalidDataIndex {
        /// Invalid index requested by the sampler.
        index: usize,
        /// Number of available data points.
        data_len: usize,
    },

    /// Degenerate point configuration.
    ///
    /// Examples: collinear points for a homography, coplanar points for a fundamental matrix.
    #[error("degenerate configuration: {reason}")]
    DegenerateConfiguration {
        /// Description of the degeneracy.
        reason: String,
    },

    /// Numerical failure.
    ///
    /// Examples: a singular system or a non-finite intermediate result. A failed SVD is
    /// [`SolverError::SvdFailed`].
    #[error("numerical error: {reason}")]
    Numerical {
        /// Description of the failure.
        reason: String,
    },

    /// SVD failed.
    #[error("SVD decomposition failed")]
    SvdFailed,
}

/// Result type for solver operations.
pub type SolverResult<T> = Result<T, SolverError>;
