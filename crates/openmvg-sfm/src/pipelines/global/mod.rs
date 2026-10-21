// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Global SfM policy, reports, and orchestration.
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                    Global engine (Global::run_prepared)                    │
//! └────────────────────────────────────────────────────────────────────────────┘
//! Input: ViewGraph workspace with verified relative poses and tracks, no poses
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Preparation (prepare_global)                                       │
//! └────────────────────────────────────────────────────────────────────────────┘
//! retain_largest_view_graph_component → condition_view_graph (drop pairs that
//! disagree with the rotations; bridges stay) → prepare_global_rotation_solution
//! (rotation averaging, solved once) → peel the cameras that disagree with those
//! rotations when they are a tail the Step 2 check would reject (at most 2 rounds,
//! at most 25% of the cameras).
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Rotation consensus check                                           │
//! └────────────────────────────────────────────────────────────────────────────┘
//! Against every verified pair: more than 10% of cameras with a median rotation
//! residual above 5° ends the run with no reconstruction.
//!   │
//!   ▼
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Candidate cascade (run_candidate_cascade), first admitted core wins│
//! └────────────────────────────────────────────────────────────────────────────┘
//! 6 attempts from the prepared input, as (candidate, route), in this order:
//!   1. (GlobalPositioning,    FullDomainThreeView)
//!   2. (TranslationAveraging, FullDomainThreeView)
//!   3. (GlobalPositioning,    FullDomainTwoView)
//!   4. (TranslationAveraging, FullDomainTwoView)
//!   5. (GlobalPositioning,    FullDomainIncidenceCoreThreeView)
//!   6. (TranslationAveraging, RigidIncidenceCoreTwoView)
//! Each attempt (run_candidate):
//!   solve_direction_translation or solve_mixed_translation → triangulate tracks →
//!   prepare_fixed_rotation_structure → BA preflight → finish_global_ba_retriangulate_*
//!   (fixed-rotation BA, released-rotation BA, robust retriangulation rounds) →
//!   GlobalPolicy::admit (post-fit trust, metric core) →
//!   self-calibration → validate_admitted_candidate (independent track validation).
//!   │
//!   ▼
//! Output: GlobalSfmReport; the committed core's workspace, or the prepared input
//!         unchanged with GlobalCompletionReport::NoReconstruction
//! ```
//!
//! # Algorithm
//!
//! Rotation averaging recovers every view's rotation from the relative rotations of the verified
//! pairs. The first candidate, [`GlobalCandidate::GlobalPositioning`], then fixes those rotations
//! and estimates centers directly from track bearings and relative directions (tracks of at least
//! three views, at most 50,000 observations by default). The second,
//! [`GlobalCandidate::TranslationAveraging`], solves camera centers from the relative translation
//! directions alone with L1-IRLS. The three-view routes triangulate tracks seen by at least three
//! views and the two-view routes also accept two-view tracks. Both candidates are followed by the
//! same finish: a bundle adjustment (BA) with rotations fixed (translations and points, Huber at 4
//! px, up to 100 iterations), a BA with rotations released (Cauchy at 4 px, 25 iterations by
//! default), and up to `max_robust_rounds` rounds (3 by default, stopping once a round changes
//! nothing) of retriangulation and BA. The released BA and each robust round are kept only when the
//! p90 disagreement with the verified relative rotations and translations stays within its limit;
//! otherwise they are rolled back. The count of locally trapped cameras is reported but does not
//! decide: bundle adjustment legitimately moves a camera away from the wrong pair edges that
//! rotation averaging had fit.
//!
//! A candidate is admitted as the core only when two certificates pass: post-fit trust (the finish
//! was accepted, the rotation-graph p90 residual is at most 5°, and the final reprojection RMSE is
//! at most 4 px) and the fixed-rotation BA's observability (full pose rank, no weak point blocks).
//! Rotation consensus over all verified pairs gates the averaged rotations in Step 2 and is only
//! reported after bundle adjustment. An admitted core is then checked with
//! held-out track triangulations. A tail of inconsistent cameras (at most 25%) is peeled and
//! the rest is validated once more; otherwise the cascade continues with the next attempt.
//!
//! # Determinism
//!
//! The attempt order is fixed. Every attempt starts from the same prepared input: the first uses
//! the conditioned workspace, and each later one rebuilds it from the pristine workspace and the
//! stored pair state, after a check that both share identical views, pairs, tracks, matches, and
//! models. Global positioning takes its seed from the workspace seed
//! (`GLOBAL_MIXED_POSITIONING_SEED_DOMAIN`), and its track subsampling order is identical on
//! 32-bit and 64-bit targets.
//!
//! # Failure modes
//!
//! No admitted core is not an error: the run returns the prepared workspace and a
//! [`GlobalCompletionReport::NoReconstruction`] whose reason is one of
//! `RotationConsensusRejected`, `NoCandidateAdmitted`, `ResourceCapacity`,
//! `FixedRotationPreflight` or `IndependentTrackValidation`. Per-attempt failures (a stage error,
//! a deferred BA preflight, a rejected admission) are recorded in the report and the cascade moves
//! on. A resource-capacity failure, or a BA preflight that is not applicable or exceeds the
//! workspace limit, ends the cascade. The run returns an error for
//! missing tracks or verified models, an invalid configuration, a workspace that already has poses
//! or landmarks, or two views that share one `PoseId`. Views outside the core stay unregistered.
//!
//! # Alternatives
//!
//! The Global engine solves all views at once and has no initial pair or growth order, but it
//! commits a reconstruction only when a core passes the checks above, and views outside the core
//! stay unregistered. The Incremental engine attempts to register every view and records a reason
//! for each one it cannot place.
//!
//! # Module layout
//!
//! `preparation` conditions the graph and solves the rotations; `driver` runs the cascade;
//! `candidate` runs one attempt; `translation` holds the two translation candidates; `finish`
//! owns the BA schedule; `policy` and `admission` hold the certificates; `cleanup` defers a
//! few cameras after the fixed-rotation BA; `config` and `report` hold the settings and the
//! serialized receipts.

mod admission;
mod candidate;
mod cleanup;
mod config;
mod driver;
mod finish;
mod policy;
mod preparation;
pub(crate) mod report;
mod translation;

pub use config::GlobalSfmConfig;
#[cfg(feature = "unstable-internals")]
pub use config::GlobalTrackValidationConfig;
pub(crate) use driver::global_core_from_owned_prepared_rotations_with_events;
pub use finish::{
    GLOBAL_MAX_REPROJECTION_ERROR_PX, GLOBAL_ROBUST_ROUNDS, GlobalBaIterationLimits,
    GlobalFinishReport, GlobalPostFixedRotationCleanupConfig,
};
#[cfg(feature = "unstable-internals")]
pub use finish::{
    GlobalPoseTrustReport, GlobalPostFixedRotationCleanupNotAttempted,
    GlobalPostFixedRotationCleanupRejection, GlobalPostFixedRotationCleanupReport,
    GlobalPostFixedRotationCleanupStatus, GlobalRetriangulationRoundReport,
    GlobalSelfCalibrationRejection, GlobalSelfCalibrationReport, GlobalSelfCalibrationStatus,
    GlobalTrappedViewDiagnostic,
};
pub(crate) use preparation::{PreparedGlobal, apply_pair_conditioning_state, prepare_global};
pub use translation::{GlobalTranslationReport, PretrackedGlobalPositioningConfig};

#[cfg(feature = "unstable-internals")]
pub use policy::{
    CandidateAdmission, CandidateAdmissionEvidence, FixedRotationObservabilityEvidence,
    GlobalPolicy, MetricCoreCertificationEvidence, MetricCoreCertificationPolicy,
    PostFitTrustEvidence, PostFitTrustPolicy, RotationConsensusEvidence, RotationConsensusPolicy,
};
pub use policy::{CandidateRejectionReason, GlobalCandidate};
#[cfg(feature = "unstable-internals")]
pub use report::{
    GlobalCandidateAttemptReport, GlobalCandidateDeferral, GlobalCandidatePostAdmissionFailure,
    GlobalCandidateRoute, GlobalCandidateStage, GlobalCandidateStatus,
    GlobalNoReconstructionReason, GlobalTrackValidationReport,
};
pub use report::{GlobalCompletionReport, GlobalPreparationReport, GlobalSfmReport};

/// Candidate-local numerical dependencies used by global routing.
#[derive(Clone, Copy)]
pub(crate) struct GlobalRuntime<'a> {
    pub(crate) direction_translation: crate::backends::TranslationAveragingFn,
    pub(crate) blind_triangulator: &'a dyn crate::backends::Triangulator,
    pub(crate) robust_triangulator: &'a dyn crate::backends::Triangulator,
    pub(crate) bundle_adjustment: &'a dyn crate::ba::backend::BundleAdjustmentBackend,
}

fn bounded_deferred_view_count(total_views: usize, max_views: usize, max_fraction: f64) -> usize {
    let fractional = (max_fraction * total_views as f64).floor() as usize;
    max_views.min(fractional.max(1))
}
