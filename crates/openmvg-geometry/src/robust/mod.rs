// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Robust estimation: find a model in data that contains wrong matches.
//!
//! RANSAC draws minimal samples, fits a model to each, and keeps the model that most data points
//! agree with, within a threshold. MSAC scores each point by its residual, capped at the
//! threshold, instead of counting inliers. The `estimate_*` functions run GenRANSAC, this
//! crate's loop: RANSAC with an adaptive sample budget, local refits of promising models, and a
//! final refit kept only when it raises the score. [`classic`] is a compact version of the same
//! idea, written to be read.
//!
//! You normally call an estimator of a problem module, such as
//! [`crate::fundamental`] or [`crate::pose::estimate_absolute_pose`]. This module holds what they
//! share: the options ([`RobustOptions`]), the result ([`RobustEstimate`]) and the error
//! ([`RobustError`]), plus the traits that plug your own solver into [`classic`].

pub mod classic;
pub(crate) mod genransac;
pub mod threshold;

pub(crate) mod error;
pub(crate) mod result;

pub use crate::solver_traits::{MultiSolutionSolver, SingleSolutionSolver, Solver};
pub use error::RobustError;
pub use genransac::{RobustGuidance, RobustOptions};
pub use result::{Diagnostics, EstimateStats, RobustEstimate};
