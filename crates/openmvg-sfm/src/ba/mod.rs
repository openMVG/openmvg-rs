// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) mod backend;
pub(crate) mod config;
pub(crate) mod native;

// Shown as links to openmvg-ba, whose pages and crate documentation explain them.
#[doc(no_inline)]
pub use openmvg_ba::{
    AuditError, BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose,
    BundleProblem, BundleProblemBuilder, ConvergenceReason, Gauge, IntrinsicCertification,
    IntrinsicVariables, LinearSolver, NativeBundleAdjuster, Observability,
    ObservabilityDiagnostics, PoseVariables, ProblemError, ResidualScale, ResidualSpace,
    RobustLoss, Solution, SolutionError, SolveConfig, SolveDiagnostics, SolveError, SolveReport,
    SolveStatus, StationarityAudit,
};
