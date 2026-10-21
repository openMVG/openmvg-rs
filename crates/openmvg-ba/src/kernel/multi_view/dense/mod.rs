// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense-Schur calibrated bundle adjustment.

mod linear_solver;

pub use super::api::DenseBundleAdjustmentWorkspace;
#[cfg(feature = "unstable-internals")]
pub use super::api::{
    AngularObservation, CalibratedBundleAdjustmentProblem, MultiViewBundleAdjustmentDiagnostics,
    MultiViewBundleAdjustmentError, MultiViewBundleAdjustmentReport, MultiViewMeasurements,
    PixelObservation, ScaleGauge, TranslationComponent,
};
#[cfg(feature = "unstable-internals")]
pub use super::covariance::{
    MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError,
};
pub use super::covariance::{
    covariance_ba_multi_view_dense, covariance_ba_multi_view_dense_with_factors,
};
#[cfg(feature = "unstable-internals")]
pub use super::factors::{
    BundleAdjustmentPriors, FixedVariableBlocks, PointPrior, PosePrior, RadialIntrinsicPrior,
};
#[cfg(feature = "unstable-internals")]
pub use super::optimizer::{refine_ba_multi_view_dense, refine_ba_multi_view_dense_with_factors};

pub(super) use linear_solver::{solve_camera_step, spectral_diagnostics};
