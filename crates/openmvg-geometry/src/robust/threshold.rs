// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed thresholds and residuals, re-exported from `openmvg-core` so that the estimators'
//! inputs can be named from this crate.
//!
//! Each estimator takes the threshold of its residual's unit; the `units` page of
//! `openmvg-core` lists which estimator takes which.

pub use openmvg_core::{
    AngularResidual, AngularThreshold, NormalizedImageResidual, NormalizedImageThreshold,
    PixelResidual, PixelThreshold, Residual, ResidualError, ScalarResidual, ScalarThreshold,
    Threshold, ThresholdError, ThresholdResult, WorldDistanceResidual, WorldDistanceThreshold,
};
