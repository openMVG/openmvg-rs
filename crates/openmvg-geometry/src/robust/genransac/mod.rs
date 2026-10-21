// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consensus-guided robust estimation for common geometric models.
//!
//! GenRANSAC runs batches of minimal hypotheses and feeds a population-level
//! consensus marginal into later sampling rounds. Most callers should use one
//! of the typed geometry facades in this module. The generic estimator contract
//! is crate-private so each public family API owns its complete numerical
//! policy.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         GenRANSAC pipeline                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! typed geometry facade
//!        │
//!        ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Guided sample                                                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Draw the minimal sample using the current population-guidance prior.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Minimal hypothesis                                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! Run the typed geometry kernel and emit zero or more candidate models.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Score, consensus, and termination                            │
//! └──────────────────────────────────────────────────────────────────────┘
//! Compute residuals in the declared units, retain the best consensus, and
//! update the stopping bound and next-round sampling guidance.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Refine and report                                            │
//! └──────────────────────────────────────────────────────────────────────┘
//! Refit with the non-minimal solver, re-score, and return the model, inliers,
//! and deterministic diagnostics.
//! ```
//!
//! The implementation reuses the crate's solver and result contracts.

pub(crate) mod config;
pub(crate) mod estimator;
pub(crate) mod refiner;

pub use config::RobustOptions;
pub use estimator::RobustGuidance;
