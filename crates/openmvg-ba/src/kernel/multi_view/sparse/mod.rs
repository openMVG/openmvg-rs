// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in sparse Schur solvers with exact `faer` LLT and retained operators.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

mod api;
mod calibrated;
mod covariance;
mod direct;
mod direct_assembly;
mod direct_assembly_policy;
mod direct_materialization;
mod execution_policy;
mod parallel;
mod pcg;
mod retained;
mod self_calibration;
mod self_calibration_retained;
#[cfg(test)]
mod tests;
mod workspace;

#[cfg(any(test, feature = "unstable-internals", feature = "parallel"))]
pub use api::RetainedSchurMatvec;
#[cfg(any(test, feature = "unstable-internals"))]
pub use api::SparseCameraSolver;
pub use api::{
    DirectSchurAssembly, IterativeSchurOptions, SparseBundleAdjustmentError,
    SparseBundleAdjustmentReport, SparseCameraRankAssessment, SparseExecutionBackend,
    SparseExecutionDecision, SparseWorkspaceMemoryBudget, SparseWorkspaceOptions,
};
#[cfg(feature = "unstable-internals")]
pub use api::{
    RetainedCrossPrecision, SparseBundleAdjustmentDiagnostics, SparseCameraSolveQuality,
    SparseExecutionReason, SparseNonlinearDiagnostics, SparsePointBackSubstitution,
};
#[cfg(any(test, feature = "unstable-internals"))]
pub use calibrated::refine_ba_multi_view_sparse;
pub use calibrated::refine_ba_multi_view_sparse_with_factors;
#[cfg(any(test, feature = "unstable-internals"))]
pub use covariance::{
    SparseBundleAdjustmentCovarianceError, selected_marginal_covariances_ba_multi_view_sparse,
    selected_marginal_covariances_ba_multi_view_sparse_with_factors,
};
#[cfg(feature = "unstable-internals")]
pub use covariance::{
    SparseBundleAdjustmentMarginalCovariances, SparseCameraMarginalCovariance,
    SparsePointMarginalCovariance,
};
pub(crate) use self_calibration::intrinsic_marginal_covariances_self_calibrating_grouped_sparse;
pub use self_calibration::refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors;
#[cfg(feature = "unstable-internals")]
pub use self_calibration::refine_ba_multi_view_self_calibrating_sparse;
pub use workspace::{SparseBundleAdjustmentWorkspace, SparseSelfCalibrationWorkspace};

#[cfg(any(test, feature = "unstable-internals"))]
pub use super::super::RelativePose;
#[cfg(any(test, feature = "unstable-internals"))]
pub use super::{BundleAdjustmentPriors, FixedVariableBlocks, MultiViewMeasurements, ScaleGauge};
#[cfg(feature = "unstable-internals")]
pub use super::{CalibratedBundleAdjustmentProblem, MultiViewBundleAdjustmentError};

#[cfg(test)]
use super::{model::ObservationModel, optimizer, schur::GaugeState};
#[cfg(test)]
use crate::numerics::refine::{Loss, NoiseModel, Options};
use calibrated::MAXIMUM_ITERATIVE_CONVERGENCE_RELATIVE_TOLERANCE;
#[cfg(test)]
use calibrated::assemble_sparse;
#[cfg(test)]
use calibrated::materialize_retained_schur;
#[cfg(test)]
use direct::{
    assess_camera_rank_with_expected_nullity, factor_camera_system, solve_factored_camera_system,
};
