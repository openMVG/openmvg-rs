// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reports of the Global engine: graph preparation, candidate attempts, and completion.

use crate::ba::backend::{BaBackendFailure, BaPreflightReport};
use crate::ops::view_graph::{ViewGraphComponentReport, ViewGraphConditioningReport};
use crate::pipelines::global::{GlobalFinishReport, GlobalTranslationReport};
use crate::{Result, SfmError, ViewId};

use super::{
    config::GlobalSfmConfig,
    policy::{CandidateAdmission, CandidateAdmissionEvidence, GlobalCandidate},
};

/// Schema of [`GlobalSfmReport`]. Increment when serialized semantics change.
pub(crate) const GLOBAL_SFM_REPORT_SCHEMA_VERSION: u32 = 1;

/// Independent track-domain evidence used before committing a global result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalTrackValidationReport {
    /// Serialized schema version of this report; currently 1.
    pub schema_version: u32,
    /// Held-out track validation of the admitted core, after any peeling.
    pub core: Box<crate::ops::track_scale::TrackScaleValidationReport>,
    /// Cameras removed from a rejected core because their held-out error exceeded the limit,
    /// before the remaining core passed validation; empty when nothing was peeled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peeled_views: Vec<crate::ViewId>,
}

impl GlobalTrackValidationReport {
    /// Report `core`, from which `peeled_views` were removed first.
    pub(crate) fn new(
        core: crate::ops::track_scale::TrackScaleValidationReport,
        peeled_views: Vec<crate::ViewId>,
    ) -> Self {
        Self {
            schema_version: 1,
            core: Box::new(core),
            peeled_views,
        }
    }
}

/// Candidate stage that returned an execution error before an admission decision was possible.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalCandidateStage {
    /// Rotation validation, translation estimation, or candidate triangulation.
    Translation,
    /// Fixed-rotation structure preparation or BA preflight.
    FixedRotationPreflight,
    /// The transactional global finish (bundle adjustment, retriangulation, cleanup).
    Finish,
}

/// Deterministic structural route used for one global candidate attempt.
#[allow(clippy::enum_variant_names)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalCandidateRoute {
    /// Ordinary full-domain translation followed by three-view triangulation.
    #[default]
    FullDomainThreeView,
    /// Full-domain translation followed by explicit two-view recovery triangulation.
    FullDomainTwoView,
    /// Rigid translation subgraph, two-view triangulation, and bundle-incidence preparation.
    RigidIncidenceCoreTwoView,
    /// Full-domain translation, three-view triangulation, and bundle-incidence preparation.
    FullDomainIncidenceCoreThreeView,
}

/// Typed reason a candidate was not attempted to completion.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum GlobalCandidateDeferral {
    /// The BA backend's fixed-rotation preflight declined to run the problem.
    FixedRotationPreflight {
        /// Preflight decision and problem measurements.
        report: Box<BaPreflightReport>,
    },
    /// The fixed-rotation preflight ran out of its workspace budget.
    ResourceCapacity {
        /// Backend failure with kind `ResourceExhausted`.
        failure: BaBackendFailure,
    },
}

/// Explicit state of one candidate attempt.
///
/// Accepted/rejected variants retain the complete reconstruction evidence. Execution failures
/// remain distinct from geometrically valid proposals that failed the admission policy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GlobalCandidateStatus {
    /// The candidate ran to completion and passed the admission policy.
    Accepted {
        /// Certificates the policy admitted.
        evidence: CandidateAdmissionEvidence,
        /// Solver-independent post-fit evidence over every verified relative-pose edge.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        all_verified_pose_graph:
            Option<Box<crate::ops::averaging::VerifiedPoseGraphConsistencyReport>>,
        /// Fixed-rotation BA preflight that let the candidate proceed.
        preflight: Box<BaPreflightReport>,
        /// Translation estimation and triangulation.
        translation: Box<GlobalTranslationReport>,
        /// Transactional global finish.
        finish: Box<GlobalFinishReport>,
    },
    /// The candidate ran to completion but failed the admission policy.
    Rejected {
        /// Certificates the policy evaluated.
        evidence: CandidateAdmissionEvidence,
        /// The policy decision; always `Rejected`, with its reasons.
        admission: CandidateAdmission,
        /// Retained before candidate rollback so removed-edge conflicts remain diagnosable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        all_verified_pose_graph:
            Option<Box<crate::ops::averaging::VerifiedPoseGraphConsistencyReport>>,
        /// Fixed-rotation BA preflight that let the candidate proceed.
        preflight: Box<BaPreflightReport>,
        /// Translation estimation and triangulation.
        translation: Box<GlobalTranslationReport>,
        /// Transactional global finish.
        finish: Box<GlobalFinishReport>,
    },
    /// The candidate stopped before the finish without an execution error.
    Deferred {
        /// Why the candidate was deferred.
        reason: GlobalCandidateDeferral,
        /// Translation stage, when it completed before the deferral.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        translation: Option<Box<GlobalTranslationReport>>,
    },
    /// A stage returned an error; no admission decision was made.
    Failed {
        /// Stage that failed.
        stage: GlobalCandidateStage,
        /// `Display` text of the stage error.
        error: String,
        /// Translation stage, when it completed before the failure.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        translation: Option<Box<GlobalTranslationReport>>,
    },
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl GlobalCandidateStatus {
    /// Whether the candidate passed the admission policy.
    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted { .. })
    }

    /// Whether the candidate was deferred before the finish.
    #[must_use]
    pub const fn is_deferred(&self) -> bool {
        matches!(self, Self::Deferred { .. })
    }
}

/// One transactional candidate execution from the same prepared workspace.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalCandidateAttemptReport {
    /// Positioning candidate that was attempted.
    pub candidate: GlobalCandidate,
    /// Structural route taken by this attempt.
    #[serde(default)]
    pub route: GlobalCandidateRoute,
    /// A core may pass the common global admission gates yet fail the independent track-domain
    /// check. Such a failure rejects only this candidate and lets the cascade try the next global
    /// backend from the pristine prepared input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_admission_failure: Option<GlobalCandidatePostAdmissionFailure>,
    /// Final state of the attempt.
    pub outcome: GlobalCandidateStatus,
}

/// Why an otherwise admitted core was not committed by the adaptive cascade.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum GlobalCandidatePostAdmissionFailure {
    /// Independent held-out track validation rejected the core or failed to run.
    IndependentTrackValidation {
        /// Validation report that rejected the core; `None` when validation returned an error.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        report: Option<Box<GlobalTrackValidationReport>>,
        /// `Display` text of the validation error; `None` when a report was produced.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// Why the cascade returned its original workspace without a core.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum GlobalNoReconstructionReason {
    /// Neither global positioning candidate passed the admission checks.
    NoCandidateAdmitted,
    /// Fixed-rotation refinement exceeded its configured workspace budget.
    ResourceCapacity {
        /// Candidate whose fixed-rotation preflight exhausted the budget.
        candidate: GlobalCandidate,
        /// Backend failure with kind `ResourceExhausted`.
        failure: BaBackendFailure,
    },
    /// The common fixed-rotation BA backend deferred every global route before execution.
    FixedRotationPreflight {
        /// Candidate whose preflight was deferred on the workspace limit.
        candidate: GlobalCandidate,
        /// Preflight decision and problem measurements.
        report: Box<BaPreflightReport>,
    },
    /// Preliminary rotations fitted the conditioned graph but conflicted with the complete
    /// verified relative-rotation evidence, so translation/triangulation were not attempted. The
    /// full typed evidence lives once in [`GlobalSfmReport::rotation_preflight`].
    RotationConsensusRejected {
        /// Failed rotation-consensus checks.
        reasons: Vec<crate::pipelines::global::CandidateRejectionReason>,
    },
    /// The core failed independent track-domain validation.
    IndependentTrackValidation {
        /// Index into [`GlobalSfmReport::attempts`] of the admitted attempt that failed validation.
        selected_attempt: usize,
        /// Validation report that rejected the core; `None` when validation returned an error.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        report: Option<Box<GlobalTrackValidationReport>>,
        /// `Display` text of the validation error; `None` when a report was produced.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// How the cascade completed after attempting the global positioning candidates.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GlobalCompletionReport {
    /// A core, as defined on [`crate::SfmEngine::Global`], passed the admission checks and was
    /// accepted.
    Reconstructed {
        /// Index into [`GlobalSfmReport::attempts`] of the committed attempt.
        selected_attempt: usize,
        /// Registered views in the committed core; at least one.
        registered_view_count: usize,
        /// Views in the input workspace; at least `registered_view_count`.
        input_view_count: usize,
        /// Independent track validation of the core; `None` when that validation is disabled.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        track_validation: Option<Box<GlobalTrackValidationReport>>,
    },
    /// No candidate passed the admission checks. The returned workspace is the pristine
    /// prepared input.
    NoReconstruction {
        /// Why no core was committed.
        reason: GlobalNoReconstructionReason,
        /// Views in the input workspace; at least one.
        input_view_count: usize,
    },
}

impl GlobalCompletionReport {
    /// Index of the committed attempt; `None` when no core was accepted.
    #[must_use]
    pub const fn accepted_global_attempt_index(&self) -> Option<usize> {
        match self {
            Self::Reconstructed {
                selected_attempt, ..
            } => Some(*selected_attempt),
            Self::NoReconstruction { .. } => None,
        }
    }
}

/// Versioned report for the Global SfM cascade.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "GlobalSfmReportWire")]
pub struct GlobalSfmReport {
    schema_version: u32,
    config: GlobalSfmConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rotation_preflight: Option<crate::ops::averaging::RotationAveragingConsistencyReport>,
    attempts: Vec<GlobalCandidateAttemptReport>,
    completion: GlobalCompletionReport,
}

#[derive(serde::Deserialize)]
#[serde(rename = "GlobalSfmReport")]
struct GlobalSfmReportWire {
    schema_version: u32,
    config: GlobalSfmConfig,
    #[serde(default)]
    rotation_preflight: Option<crate::ops::averaging::RotationAveragingConsistencyReport>,
    attempts: Vec<GlobalCandidateAttemptReport>,
    completion: GlobalCompletionReport,
}

impl TryFrom<GlobalSfmReportWire> for GlobalSfmReport {
    type Error = SfmError;

    fn try_from(wire: GlobalSfmReportWire) -> Result<Self> {
        if wire.schema_version != GLOBAL_SFM_REPORT_SCHEMA_VERSION {
            return Err(SfmError::InvalidInput(format!(
                "unsupported GlobalSfmReport schema_version {} (this build reads {})",
                wire.schema_version, GLOBAL_SFM_REPORT_SCHEMA_VERSION
            )));
        }
        Self::try_new(
            wire.config,
            wire.rotation_preflight,
            wire.attempts,
            wire.completion,
        )
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl GlobalSfmReport {
    pub(crate) fn try_new(
        config: GlobalSfmConfig,
        rotation_preflight: Option<crate::ops::averaging::RotationAveragingConsistencyReport>,
        attempts: Vec<GlobalCandidateAttemptReport>,
        completion: GlobalCompletionReport,
    ) -> Result<Self> {
        config.validate()?;
        for attempt in &attempts {
            match &attempt.outcome {
                GlobalCandidateStatus::Accepted { evidence, .. }
                    if !config.policy.admit(*evidence).is_accepted() =>
                {
                    return Err(SfmError::InvalidInput(
                        "global report marks a policy-rejected candidate as accepted".to_owned(),
                    ));
                }
                GlobalCandidateStatus::Rejected {
                    evidence,
                    admission,
                    ..
                } if config.policy.admit(*evidence) != *admission || admission.is_accepted() => {
                    return Err(SfmError::InvalidInput(
                        "global report rejection disagrees with its policy evidence".to_owned(),
                    ));
                }
                _ => {}
            }
            if !attempt.outcome.is_accepted() && attempt.post_admission_failure.is_some() {
                return Err(SfmError::InvalidInput(
                    "global post-admission failure requires an admitted core".to_owned(),
                ));
            }
        }
        let accepted_index = completion.accepted_global_attempt_index();
        if let Some(index) = accepted_index {
            if !attempts.get(index).is_some_and(|attempt| {
                attempt.outcome.is_accepted() && attempt.post_admission_failure.is_none()
            }) {
                return Err(SfmError::InvalidInput(
                    "global completion references a missing or unaccepted attempt".to_owned(),
                ));
            }
        } else if attempts.iter().any(|attempt| {
            attempt.outcome.is_accepted() && attempt.post_admission_failure.is_none()
        }) {
            return Err(SfmError::InvalidInput(
                "global completion cannot discard an accepted attempt without a validation failure"
                    .to_owned(),
            ));
        }
        if let GlobalCompletionReport::Reconstructed {
            registered_view_count,
            input_view_count,
            ..
        } = &completion
            && (*registered_view_count == 0
                || *input_view_count == 0
                || registered_view_count > input_view_count)
        {
            return Err(SfmError::InvalidInput(
                "global registered camera count must be non-zero and bounded by the input count"
                    .to_owned(),
            ));
        }
        if let GlobalCompletionReport::NoReconstruction {
            input_view_count, ..
        } = &completion
            && *input_view_count == 0
        {
            return Err(SfmError::InvalidInput(
                "global execution requires a non-empty input camera domain".to_owned(),
            ));
        }
        Ok(Self {
            schema_version: GLOBAL_SFM_REPORT_SCHEMA_VERSION,
            config,
            rotation_preflight,
            attempts,
            completion,
        })
    }

    /// Serialized schema version; incremented when serialized semantics change.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Validated configuration the cascade ran with.
    #[must_use]
    pub const fn config(&self) -> &GlobalSfmConfig {
        &self.config
    }

    /// All-verified rotation consistency of the preliminary rotations, checked before any
    /// candidate; the cascade always records it, so `None` only comes from a report without it.
    #[must_use]
    pub const fn rotation_preflight(
        &self,
    ) -> Option<&crate::ops::averaging::RotationAveragingConsistencyReport> {
        self.rotation_preflight.as_ref()
    }

    /// Candidate attempts in execution order.
    #[must_use]
    pub fn attempts(&self) -> &[GlobalCandidateAttemptReport] {
        &self.attempts
    }

    /// How the cascade completed.
    #[must_use]
    pub const fn completion(&self) -> &GlobalCompletionReport {
        &self.completion
    }

    /// The committed attempt; `None` when no core was accepted.
    #[must_use]
    pub fn accepted_global_attempt(&self) -> Option<&GlobalCandidateAttemptReport> {
        self.completion
            .accepted_global_attempt_index()
            .and_then(|index| self.attempts.get(index))
    }
}

/// Typed graph preparation for global execution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalPreparationReport {
    /// Retention of the largest active connected component of the input view graph.
    pub initial_largest_component: ViewGraphComponentReport,
    /// Pair edges deactivated by the adaptive cycle-reliability rotation conditioning, and the
    /// cameras peeled afterwards as a rotation tail (a `_peeled<N>` suffix of the method).
    pub rotation_conditioning: ViewGraphConditioningReport,
    /// Retention of the largest connected component after rotation conditioning.
    pub final_largest_component: ViewGraphComponentReport,
    /// Number of entries in `candidate_domain_camera_ids`.
    pub candidate_domain_camera_count: usize,
    /// Views incident to at least one active pair after conditioning, ascending.
    pub candidate_domain_camera_ids: Vec<ViewId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_is_typed_and_serialized_without_fake_admission() {
        let report = GlobalCandidateAttemptReport {
            candidate: GlobalCandidate::TranslationAveraging,
            route: GlobalCandidateRoute::FullDomainThreeView,
            post_admission_failure: None,
            outcome: GlobalCandidateStatus::Failed {
                stage: GlobalCandidateStage::Translation,
                error: "translation solve failed".to_owned(),
                translation: None,
            },
        };
        assert!(!report.outcome.is_accepted());
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["candidate"], "translation_averaging");
        assert_eq!(json["route"], "full_domain_three_view");
        assert_eq!(json["outcome"]["status"], "failed");
        assert_eq!(json["outcome"]["stage"], "translation");
        assert!(json["outcome"].get("admission").is_none());

        let mut legacy = json;
        legacy.as_object_mut().unwrap().remove("route");
        let restored: GlobalCandidateAttemptReport = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.route, GlobalCandidateRoute::FullDomainThreeView);
    }

    #[test]
    fn report_rejects_an_impossible_selected_attempt() {
        let attempt = GlobalCandidateAttemptReport {
            candidate: GlobalCandidate::TranslationAveraging,
            route: GlobalCandidateRoute::FullDomainThreeView,
            post_admission_failure: None,
            outcome: GlobalCandidateStatus::Failed {
                stage: GlobalCandidateStage::Finish,
                error: "test".to_owned(),
                translation: None,
            },
        };
        let error = GlobalSfmReport::try_new(
            GlobalSfmConfig::default(),
            None,
            vec![attempt],
            GlobalCompletionReport::Reconstructed {
                selected_attempt: 0,
                registered_view_count: 3,
                input_view_count: 3,
                track_validation: None,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("unaccepted attempt"));
    }

    #[test]
    fn completion_exposes_its_selected_global_attempt() {
        assert_eq!(
            GlobalCompletionReport::Reconstructed {
                selected_attempt: 3,
                registered_view_count: 8,
                input_view_count: 12,
                track_validation: None,
            }
            .accepted_global_attempt_index(),
            Some(3)
        );
    }

    #[test]
    fn core_only_no_core_requires_a_non_empty_input_domain() {
        let error = GlobalSfmReport::try_new(
            GlobalSfmConfig::default(),
            None,
            Vec::new(),
            GlobalCompletionReport::NoReconstruction {
                reason: GlobalNoReconstructionReason::NoCandidateAdmitted,
                input_view_count: 0,
            },
        )
        .expect_err("empty input count must be rejected");
        assert!(error.to_string().contains("non-empty input camera domain"));
    }

    #[test]
    fn resource_capacity_is_a_deferral_not_a_solver_failure() {
        let failure = BaBackendFailure::new(
            "test-ba",
            crate::ba::backend::BaFailureKind::ResourceExhausted,
            "workspace limit exceeded",
        );
        let outcome = GlobalCandidateStatus::Deferred {
            reason: GlobalCandidateDeferral::ResourceCapacity { failure },
            translation: None,
        };
        assert!(outcome.is_deferred());
        assert!(!outcome.is_accepted());
        let json = serde_json::to_value(&outcome).unwrap();
        assert_eq!(json["status"], "deferred");
        assert_eq!(json["reason"]["reason"], "resource_capacity");
        assert_eq!(json["reason"]["failure"]["kind"], "resource_exhausted");
    }
}
