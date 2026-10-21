// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reproducible statistical acceptance tests: the self-calibrating BA covariance
//! must match the empirical error distribution of repeated noisy refinements.

#[path = "statistical_validation/ba_common.rs"]
mod ba_common;
#[path = "statistical_validation/ba_self_calibration.rs"]
mod ba_self_calibration;
#[path = "statistical_validation/common.rs"]
mod common;
