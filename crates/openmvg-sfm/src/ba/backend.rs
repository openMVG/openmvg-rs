// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bundle-adjustment backend contract: the [`BundleAdjustmentBackend`] trait, its preflight
//! and solve results, and the diagnostics the pipelines audit before committing a solution.

use crate::IntrinsicId;
use crate::ba::config::BundleAdjustConfig;
use crate::two_view::ResidualUnit;

/// Tolerance that ended a converged BA solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaConvergenceReason {
    /// The relative objective reduction reached the function tolerance.
    Objective,
    /// The scaled gradient norm reached the gradient tolerance.
    Gradient,
    /// The parameter step norm reached the parameter tolerance.
    Step,
    /// The backend reported convergence without one of the reasons above.
    Unspecified,
}

/// Typed numerical status of a returned BA proposal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaSolveStatus {
    /// A convergence tolerance was satisfied.
    Converged(BaConvergenceReason),
    /// The iteration budget ran out before any tolerance was satisfied.
    MaxIterations,
    /// No acceptable finite step could be found.
    Stalled,
    /// The final local information is rank deficient.
    Degenerate,
    /// The input state was returned unchanged with no iteration counted.
    Skipped,
    /// The backend failed numerically.
    NumericalFailure,
    /// The backend reported no status this vocabulary represents; also the value of a
    /// deserialized report without a status.
    #[default]
    Unknown,
}

/// Rank evidence for the gauge-reduced camera system of the final undamped BA normal
/// equations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaRankAssessment {
    /// The gauge-reduced camera system has full rank.
    FullRank,
    /// The only rank deficiency is the scale direction a free-scale gauge leaves open.
    OnlyExpectedGauge,
    /// The camera system is rank deficient beyond the expected gauge, or indefinite.
    RankDeficient,
    /// Rank was not assessed; the commit audit rejects such a proposal.
    #[default]
    NotAssessed,
}

/// Backend-provided observability evidence for the undamped reduced BA system.
/// This is reporting-only; commit semantics remain governed by the flat
/// [`BaDiagnostics::rank`] and [`BaDiagnostics::weak_point_count`] fields.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BaObservabilityDiagnostics {
    /// Numerical rank of the undamped gauge-reduced pose system.
    pub pose_rank: usize,
    /// Free pose coordinates after gauge removal.
    pub pose_dimension: usize,
    /// Point blocks that could not be eliminated in the undamped assessment; `None` when the
    /// backend did not report it.
    pub weak_point_count: Option<usize>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl BaObservabilityDiagnostics {
    /// Whether the pose rank is below the gauge-reduced pose dimension.
    #[must_use]
    pub const fn pose_is_rank_deficient(&self) -> bool {
        self.pose_rank < self.pose_dimension
    }
}

/// Backend-neutral numerical evidence accompanying a BA proposal.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BaDiagnostics {
    /// Nonlinear steps accepted; `None` when the backend did not report it.
    pub accepted_step_count: Option<usize>,
    /// Trial steps rejected; `None` when the backend did not report it.
    pub rejected_step_count: Option<usize>,
    /// Largest final gradient coordinate divided by the square root of its Gauss-Newton
    /// curvature; `None` when the backend did not report it.
    pub gradient_inf_norm: Option<f64>,
    /// Rank evidence for the final undamped camera system.
    pub rank: BaRankAssessment,
    /// Numerically singular point blocks in the final undamped system; `None` when the
    /// backend did not report it.
    pub weak_point_count: Option<usize>,
    /// Final observations outside the camera model's unambiguous inverse domain; `None` when
    /// the backend did not report it.
    pub noninvertible_observation_count: Option<usize>,
    /// Detailed rank and conditioning evidence, for reporting only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observability: Option<BaObservabilityDiagnostics>,
    /// Intrinsic groups a self-calibrating solve held at their input values because their
    /// observations did not constrain the declared camera model's parameters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unconstrained_intrinsics: Vec<IntrinsicId>,
}

/// Evidence-based reason a numerically returned BA proposal cannot be committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaCommitRejection {
    /// A reported cost or residual metric is not finite.
    NonFiniteMetric,
    /// The final objective exceeds the initial one by more than round-off.
    CostIncrease,
    /// The camera system is rank deficient beyond the expected gauge.
    RankDeficient,
    /// The final undamped system has numerically singular point blocks.
    WeakPoints {
        /// Number of weak point blocks.
        count: usize,
    },
    /// The iteration budget ran out without an accepted step that lowered the objective.
    NoAcceptedProgress,
    /// The backend did not assess the camera-system rank.
    ObservabilityNotAssessed,
    /// Only the free-scale gauge deficiency was reported, but the problem's gauge fixes the
    /// scale.
    GaugeMismatch,
    /// Gradient convergence was claimed, but the zero-damping stationarity exceeds the
    /// gradient tolerance.
    StationarityNotCertified,
    /// The solve status does not permit commit.
    Status(BaSolveStatus),
}

/// Serialized receipt of one bundle-adjustment solve.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BundleAdjustmentReport {
    /// [`BundleAdjustmentBackend::name`] of the backend that produced the solution.
    pub backend_name: String,
    /// Robust objective at the input state: the robust loss summed over every observation's
    /// residual norm.
    pub initial_objective: f64,
    /// Robust objective at the returned state.
    pub final_objective: f64,
    /// Root-mean-square observation residual norm at the input state, without robust
    /// weighting, in [`Self::residual_units`].
    pub initial_rmse: f64,
    /// Root-mean-square observation residual norm at the returned state, without robust
    /// weighting, in [`Self::residual_units`].
    pub final_rmse: f64,
    /// Units of residual vectors underlying the objective and reported RMSE.
    pub residual_units: ResidualUnit,
    /// Nonlinear iterations run.
    pub iteration_count: usize,
    /// Observations in the BA problem.
    pub observation_count: usize,
    /// Camera-pose blocks in the BA problem.
    pub pose_count: usize,
    /// Point blocks in the BA problem.
    pub point_count: usize,
    /// Shared intrinsic blocks in the BA problem.
    pub intrinsic_group_count: usize,
    /// Typed termination status, the field control flow must use.
    #[serde(default)]
    pub status: BaSolveStatus,
    /// Numerical evidence accompanying the solution.
    #[serde(default)]
    pub diagnostics: BaDiagnostics,
    /// Human-readable backend detail for logs. Control flow must use
    /// [`Self::status`] instead.
    pub termination_reason: String,
}

/// Capability category that prevented an exact backend lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaUnsupportedCapability {
    /// A camera model in the problem.
    CameraModel,
    /// The requested intrinsic refinement.
    IntrinsicRefinement,
    /// The requested optimization scope.
    OptimizationScope,
    /// The requested robust loss.
    RobustLoss,
    /// The requested linear solver.
    LinearSolver,
    /// The problem's gauge.
    Gauge,
    /// The structure of the observation graph.
    Topology,
    /// The target platform or enabled features.
    Platform,
    /// Any other capability; the native adapter reports every unsupported solve as this.
    Other,
}

/// A typed, non-failing declaration that a backend cannot represent a problem.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct BaNotApplicable {
    /// [`BundleAdjustmentBackend::name`] of the declaring backend.
    pub backend_name: String,
    /// Capability category the backend lacks.
    pub capability: BaUnsupportedCapability,
    /// Human-readable detail for logs.
    pub reason: String,
}

impl BaNotApplicable {
    /// Declare that `backend_name` cannot represent a problem because it lacks `capability`.
    pub fn new(
        backend_name: impl Into<String>,
        capability: BaUnsupportedCapability,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            backend_name: backend_name.into(),
            capability,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for BaNotApplicable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} does not support {:?}: {}",
            self.backend_name, self.capability, self.reason
        )
    }
}

impl std::error::Error for BaNotApplicable {}

/// Why a backend-neutral BA preflight deferred an otherwise supported solve.
///
/// Deferral is distinct from both capability non-applicability and
/// execution failure: the caller may safely choose a different reconstruction
/// route without claiming that the numerical problem is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaPreflightDeferralReason {
    /// The solve workspace upper bound exceeds its configured limit.
    WorkspaceLimit,
    /// The bounded observability probe itself exceeded its work budget.
    ObservabilityProbeResourceLimit,
    /// The bounded track subset of the probe did not cover every free camera.
    InsufficientCameraCoverage,
    /// At least one point block selected by the probe was numerically weak.
    WeakPointGeometry,
    /// The probe produced no positive observability certificate; also used for probe
    /// refusals the adapter does not recognize.
    ObservabilityNotCertified,
    /// The bounded probe did not provide an aligned certificate, so admissibility requires exact
    /// rank evidence while the selected execution plan is iterative.
    ExactObservabilityRequiresDirect,
}

/// Typed decision returned before a potentially expensive BA solve.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum BaPreflightDecision {
    /// Run the solve.
    Proceed,
    /// Skip the solve; the caller may choose a different reconstruction route.
    Deferred(BaPreflightDeferralReason),
    /// The backend cannot represent the problem.
    NotApplicable(BaNotApplicable),
}

/// Backend-neutral resource and observability evidence gathered before BA.
///
/// Optional fields remain absent for backends that use the default capability-only
/// preflight. Concrete backend probe types do not cross this boundary.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BaPreflightReport {
    /// Whether, and why not, the solve should run.
    pub decision: BaPreflightDecision,
    /// Linear-solver workspace budget in bytes the backend applied; `None` when it planned
    /// against its automatic budget.
    pub workspace_budget_bytes: Option<u128>,
    /// Gauge-reduced pose dimension tested by the observability probe; `None` when the
    /// probe did not run.
    pub pose_dimension: Option<usize>,
    /// Poses with free translation covered by the probe's selected tracks; `None` when the
    /// probe did not run.
    pub covered_free_pose_count: Option<usize>,
    /// Point blocks selected by the probe that could not be eliminated; `None` when the
    /// probe did not run.
    pub weak_point_count: Option<usize>,
    /// The probe proceeded without a sparse certificate, so the pose rank is credited only
    /// if the solve's own observability assessment confirms it.
    pub dense_observability_escalation: bool,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl BaPreflightReport {
    /// A report admitting the solve, with no probe evidence.
    #[must_use]
    pub const fn proceed() -> Self {
        Self {
            decision: BaPreflightDecision::Proceed,
            workspace_budget_bytes: None,
            pose_dimension: None,
            covered_free_pose_count: None,
            weak_point_count: None,
            dense_observability_escalation: false,
        }
    }

    /// A report deferring the solve for `reason`, with no probe evidence.
    #[must_use]
    pub const fn deferred(reason: BaPreflightDeferralReason) -> Self {
        Self {
            decision: BaPreflightDecision::Deferred(reason),
            workspace_budget_bytes: None,
            pose_dimension: None,
            covered_free_pose_count: None,
            weak_point_count: None,
            dense_observability_escalation: false,
        }
    }

    /// Whether the decision is [`BaPreflightDecision::Proceed`].
    #[must_use]
    pub const fn should_proceed(&self) -> bool {
        matches!(&self.decision, BaPreflightDecision::Proceed)
    }
}

/// Preflight result: a report, or the backend failure that prevented the probe.
pub type BaPreflightResult = std::result::Result<BaPreflightReport, BaBackendFailure>;

/// Category of a [`BaBackendFailure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BaFailureKind {
    /// The backend rejected the problem as malformed.
    InvalidProblem,
    /// The backend rejected its configuration or solver options.
    InvalidConfig,
    /// The solve failed numerically.
    Numerical,
    /// The solve exceeded a bounded resource such as its workspace limit.
    ResourceExhausted,
    /// The backend hit an internal defect.
    Internal,
}

/// Backend execution failure, distinct from unsupported capability.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct BaBackendFailure {
    /// [`BundleAdjustmentBackend::name`] of the failing backend.
    pub backend_name: String,
    /// Failure category.
    pub kind: BaFailureKind,
    /// Human-readable detail for logs.
    pub message: String,
    /// Hint that the same problem may succeed with other resources; the native adapter sets
    /// it only for resource exhaustion.
    pub retryable: bool,
}

impl BaBackendFailure {
    /// A non-retryable failure of `backend_name`.
    pub fn new(
        backend_name: impl Into<String>,
        kind: BaFailureKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            backend_name: backend_name.into(),
            kind,
            message: message.into(),
            retryable: false,
        }
    }

    /// Set the [`Self::retryable`] hint.
    pub const fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
}

impl std::fmt::Display for BaBackendFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {:?} failure: {}",
            self.backend_name, self.kind, self.message
        )
    }
}

impl std::error::Error for BaBackendFailure {}

/// Result of backend dispatch. `Proposed` is not named `Committed` because only the
/// pipeline may check and commit it to a workspace.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BaBackendStatus {
    /// An uncommitted solution the pipeline audits before committing.
    Proposed(Box<openmvg_ba::Solution>),
    /// The backend cannot represent the problem.
    NotApplicable(BaNotApplicable),
}

/// Backend dispatch result: an outcome, or an execution failure.
pub type BaBackendResult = std::result::Result<BaBackendStatus, BaBackendFailure>;

/// Bundle-adjustment solver used by the pipelines.
pub trait BundleAdjustmentBackend: Send + Sync {
    /// Stable implementation name recorded in pipeline receipts.
    fn name(&self) -> &'static str {
        "custom-bundle-adjuster"
    }

    /// Bounded, non-mutating admission probe before a potentially expensive solve.
    ///
    /// The default admits the problem. Backends may report resource, capability, or
    /// observability deferrals without exposing implementation-specific probe types to the
    /// pipeline. [`Self::adjust`] remains authoritative when deeper preparation discovers an
    /// unsupported condition.
    fn preflight(
        &self,
        _problem: &openmvg_ba::BundleProblem,
        _cfg: &BundleAdjustConfig,
    ) -> BaPreflightResult {
        Ok(BaPreflightReport::proceed())
    }

    /// Solve `problem` under `cfg` and return a proposal for the pipeline to audit and
    /// commit, or [`BaBackendStatus::NotApplicable`] when the problem is unsupported.
    fn adjust(
        &self,
        problem: &openmvg_ba::BundleProblem,
        cfg: &BundleAdjustConfig,
    ) -> BaBackendResult;
}
