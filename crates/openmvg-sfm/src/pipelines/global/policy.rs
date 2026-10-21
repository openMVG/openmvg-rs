// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Admission policies of the Global engine: the rotation-consensus, metric-core, and post-fit
//! trust certificates that a positioning candidate must pass.

use crate::ba::backend::BaObservabilityDiagnostics;
use crate::{Result, SfmError};

/// The two ways the Global engine estimates camera positions. The default policy tries
/// [`Self::GlobalPositioning`] first and [`Self::TranslationAveraging`] second; both start
/// from the same averaged rotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum GlobalCandidate {
    /// Joint positioning of cameras and points from the bearings of stable tracks and the
    /// relative directions.
    GlobalPositioning,
    /// Direction-only translation averaging of the pairwise directions, with an
    /// iteratively reweighted L1 solver (L1-IRLS).
    TranslationAveraging,
}

/// Backend-neutral fixed-rotation BA observability evidence.
///
/// This mirrors only the typed diagnostics exposed by the pipeline BA trait, so the adaptive
/// global policy does not depend on a concrete backend.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FixedRotationObservabilityEvidence {
    /// Numerical rank of the undamped gauge-reduced pose system.
    pub pose_rank: usize,
    /// Free pose coordinates after gauge removal; zero marks invalid evidence.
    pub pose_dimension: usize,
    /// Point blocks that could not be eliminated; `None` when the backend did not report it.
    pub weak_point_count: Option<usize>,
}

impl From<&BaObservabilityDiagnostics> for FixedRotationObservabilityEvidence {
    fn from(diagnostics: &BaObservabilityDiagnostics) -> Self {
        Self {
            pose_rank: diagnostics.pose_rank,
            pose_dimension: diagnostics.pose_dimension,
            weak_point_count: diagnostics.weak_point_count,
        }
    }
}

/// Rotation evidence that is independent of the conditioned candidate graph.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RotationConsensusEvidence {
    /// Cameras with incident all-verified rotation evidence, including conditioned-out edges.
    pub all_verified_camera_count: usize,
    /// Cameras whose incident median rotation residual exceeds the configured limit.
    pub all_verified_inconsistent_camera_count: usize,
}

/// Structural and numerical evidence that the candidate is a solvable core.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MetricCoreCertificationEvidence {
    /// Observability of the fixed-rotation translation/point refinement.
    pub fixed_rotation_observability: Option<FixedRotationObservabilityEvidence>,
}

/// Evidence owned by the common transactional global-finish schedule.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PostFitTrustEvidence {
    /// Whether the transactional refinement schedule accepted its final proposal.
    pub finish_accepted: bool,
    /// Post-fit p90 disagreement with verified relative translations, in degrees.
    pub translation_graph_p90_deg: Option<f64>,
    /// Root-mean-square reprojection error of the last accepted bundle adjustment, in pixels;
    /// `None` when that adjustment did not report pixel residuals.
    #[serde(default)]
    pub final_reprojection_rmse_px: Option<f64>,
    /// Post-fit p90 disagreement with verified relative rotations, in degrees.
    pub rotation_graph_p90_deg: Option<f64>,
    /// Number of posed cameras evaluated against every verified relative-pose edge.
    #[serde(default)]
    pub all_verified_camera_count: usize,
    /// Cameras whose incident median exceeds the policy's rotation or translation threshold.
    #[serde(default)]
    pub all_verified_inconsistent_camera_count: usize,
}

/// Three independent certificates used to admit or reject one positioning candidate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CandidateAdmissionEvidence {
    /// All-verified rotation consensus, measured independently of the conditioned graph.
    pub rotation_consensus: RotationConsensusEvidence,
    /// Structural and fixed-rotation numerical core evidence.
    pub metric_core: MetricCoreCertificationEvidence,
    /// Post-fit pose-graph agreement after the transactional finish.
    pub post_fit_trust: PostFitTrustEvidence,
}

/// Stable reason a numerically completed candidate was not admitted as a global core.
#[non_exhaustive]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejectionReason {
    /// The transactional finish schedule did not accept its final proposal.
    FinishRejected,
    /// No post-fit translation-graph p90 residual was measured.
    MissingTranslationGraphEvidence,
    /// No post-fit rotation-graph p90 residual was measured.
    MissingRotationGraphEvidence,
    /// The post-fit rotation-graph p90 residual is not finite or exceeds the p90 limit.
    RotationGraphResidual,
    /// The final reprojection RMSE is not finite or exceeds the RMSE limit.
    ReprojectionRmse,
    /// The camera-balanced check is enabled but no posed camera was evaluated.
    MissingCameraBalancedGraphEvidence,
    /// Too large a fraction of posed cameras disagree with their verified edges after the fit.
    CameraBalancedGraphTail,
    /// No camera has incident all-verified rotation evidence.
    MissingAllVerifiedRotationCameraEvidence,
    /// Too large a fraction of cameras exceed the all-verified rotation median limit.
    AllVerifiedRotationCameraTail,
    /// Fixed-rotation BA reported no observability diagnostics.
    MissingObservabilityEvidence,
    /// The observability diagnostics are inconsistent: zero camera dimension, rank above the
    /// dimension, or a negative or non-finite reciprocal condition number.
    InvalidObservabilityEvidence,
    /// The fixed-rotation camera system is rank deficient after gauge removal.
    CameraRankDeficient,
    /// Fixed-rotation BA did not report its weak point-block count.
    MissingWeakPointEvidence,
    /// Fixed-rotation BA found at least one point block it could not eliminate.
    WeakPointGeometry,
}

/// Typed admission result. An accepted candidate cannot carry rejection reasons.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CandidateAdmission {
    /// Every certificate passed; the candidate is admitted as the global core.
    Accepted,
    /// At least one certificate failed.
    Rejected {
        /// Failed checks, in policy evaluation order.
        reasons: Vec<CandidateRejectionReason>,
    },
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl CandidateAdmission {
    /// Whether the candidate was admitted.
    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Failed checks; empty when accepted.
    #[must_use]
    pub fn rejection_reasons(&self) -> &[CandidateRejectionReason] {
        match self {
            Self::Accepted => &[],
            Self::Rejected { reasons } => reasons,
        }
    }
}

/// Policy for all-verified rotation consensus.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RotationConsensusPolicy {
    max_all_verified_camera_median_deg: f64,
    max_all_verified_inconsistent_camera_fraction: f64,
}

config_accessors!(RotationConsensusPolicy {
    /// Maximum admitted fraction of cameras above the per-camera median threshold.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_all_verified_inconsistent_camera_fraction, with_max_all_verified_inconsistent_camera_fraction: f64;
});

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl RotationConsensusPolicy {
    /// Per-camera median rotation threshold over every verified incident edge.
    #[must_use]
    pub const fn max_all_verified_camera_median_deg(&self) -> f64 {
        self.max_all_verified_camera_median_deg
    }
}

impl Default for RotationConsensusPolicy {
    fn default() -> Self {
        Self {
            max_all_verified_camera_median_deg: 5.0,
            max_all_verified_inconsistent_camera_fraction: 0.1,
        }
    }
}

impl RotationConsensusPolicy {
    /// Check the thresholds.
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidConfig`] when the camera-median limit is not finite and positive or the
    /// camera fraction is outside `[0, 1]`.
    pub fn validate(self) -> Result<()> {
        if !self.max_all_verified_camera_median_deg.is_finite()
            || self.max_all_verified_camera_median_deg <= 0.0
        {
            return Err(SfmError::InvalidConfig(
                "global all-verified camera rotation limit must be finite and positive".to_owned(),
            ));
        }
        if !self
            .max_all_verified_inconsistent_camera_fraction
            .is_finite()
            || !(0.0..=1.0).contains(&self.max_all_verified_inconsistent_camera_fraction)
        {
            return Err(SfmError::InvalidConfig(
                "global all-verified rotation camera fraction must be in [0, 1]".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn evidence(
        self,
        camera_medians_deg: impl IntoIterator<Item = Option<f64>>,
    ) -> RotationConsensusEvidence {
        let mut all_verified_camera_count = 0usize;
        let mut all_verified_inconsistent_camera_count = 0usize;
        for median in camera_medians_deg.into_iter().flatten() {
            all_verified_camera_count += 1;
            all_verified_inconsistent_camera_count +=
                usize::from(median > self.max_all_verified_camera_median_deg);
        }
        RotationConsensusEvidence {
            all_verified_camera_count,
            all_verified_inconsistent_camera_count,
        }
    }

    /// Rotation-consensus failures for `evidence`; empty when the check passes.
    #[must_use]
    pub fn rejection_reasons(
        self,
        evidence: RotationConsensusEvidence,
    ) -> Vec<CandidateRejectionReason> {
        if evidence.all_verified_camera_count == 0 {
            return vec![CandidateRejectionReason::MissingAllVerifiedRotationCameraEvidence];
        }
        let inconsistent_fraction = evidence.all_verified_inconsistent_camera_count as f64
            / evidence.all_verified_camera_count as f64;
        (inconsistent_fraction > self.max_all_verified_inconsistent_camera_fraction)
            .then_some(CandidateRejectionReason::AllVerifiedRotationCameraTail)
            .into_iter()
            .collect()
    }
}

/// Policy for the structural and fixed-rotation numerical core certificate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MetricCoreCertificationPolicy;

impl MetricCoreCertificationPolicy {
    /// Observability failures for `evidence`; empty when the fixed-rotation camera system is
    /// full rank and has no weak point block.
    #[must_use]
    pub fn rejection_reasons(
        self,
        evidence: MetricCoreCertificationEvidence,
    ) -> Vec<CandidateRejectionReason> {
        let mut reasons = Vec::new();
        match evidence.fixed_rotation_observability {
            Some(observability)
                if observability.pose_dimension == 0
                    || observability.pose_rank > observability.pose_dimension =>
            {
                reasons.push(CandidateRejectionReason::InvalidObservabilityEvidence);
            }
            Some(observability) => {
                if observability.pose_rank < observability.pose_dimension {
                    reasons.push(CandidateRejectionReason::CameraRankDeficient);
                }
                match observability.weak_point_count {
                    Some(0) => {}
                    Some(_) => reasons.push(CandidateRejectionReason::WeakPointGeometry),
                    None => reasons.push(CandidateRejectionReason::MissingWeakPointEvidence),
                }
            }
            None => reasons.push(CandidateRejectionReason::MissingObservabilityEvidence),
        }
        reasons
    }
}

/// Policy for the common post-fit pose-graph trust certificate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PostFitTrustPolicy {
    max_rotation_graph_p90_deg: f64,
    max_final_reprojection_rmse_px: Option<f64>,
    max_all_verified_camera_rotation_median_deg: f64,
    max_all_verified_camera_translation_median_deg: f64,
    max_all_verified_inconsistent_camera_fraction: Option<f64>,
}

config_accessors!(PostFitTrustPolicy {
    /// Maximum admitted post-fit p90 relative-rotation residual over the active view graph,
    /// in degrees.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_rotation_graph_p90_deg, with_max_rotation_graph_p90_deg: f64;
    /// Optional maximum reprojection RMSE of the last accepted bundle adjustment, in pixels.
    /// `None` disables the check.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_final_reprojection_rmse_px, with_max_final_reprojection_rmse_px: Option<f64>;
    /// Per-camera median translation threshold used to count inconsistent cameras.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_all_verified_camera_translation_median_deg, with_max_all_verified_camera_translation_median_deg: f64;
    /// Optional maximum fraction of inconsistent posed cameras over all verified edges.
    /// `None` uses p90-only admission.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_all_verified_inconsistent_camera_fraction, with_max_all_verified_inconsistent_camera_fraction: Option<f64>;
});

impl PostFitTrustPolicy {
    /// Per-camera median rotation threshold used to count inconsistent cameras.
    #[must_use]
    pub const fn max_all_verified_camera_rotation_median_deg(&self) -> f64 {
        self.max_all_verified_camera_rotation_median_deg
    }
}

impl Default for PostFitTrustPolicy {
    fn default() -> Self {
        Self {
            max_rotation_graph_p90_deg: 5.0,
            max_final_reprojection_rmse_px: Some(4.0),
            max_all_verified_camera_rotation_median_deg: 5.0,
            max_all_verified_camera_translation_median_deg: 10.0,
            max_all_verified_inconsistent_camera_fraction: None,
        }
    }
}

impl PostFitTrustPolicy {
    /// Check the thresholds.
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidConfig`] when a degree or RMSE limit is not finite and positive or
    /// the inconsistent-camera fraction is outside `[0, 1]`.
    pub fn validate(self) -> Result<()> {
        if !self.max_rotation_graph_p90_deg.is_finite() || self.max_rotation_graph_p90_deg <= 0.0 {
            return Err(SfmError::InvalidConfig(
                "global rotation-graph residual limit must be finite and positive".to_owned(),
            ));
        }
        if self
            .max_final_reprojection_rmse_px
            .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
        {
            return Err(SfmError::InvalidConfig(
                "global reprojection RMSE limit must be finite and positive".to_owned(),
            ));
        }
        if !self.max_all_verified_camera_rotation_median_deg.is_finite()
            || self.max_all_verified_camera_rotation_median_deg <= 0.0
            || !self
                .max_all_verified_camera_translation_median_deg
                .is_finite()
            || self.max_all_verified_camera_translation_median_deg <= 0.0
        {
            return Err(SfmError::InvalidConfig(
                "global camera-balanced residual limits must be finite and positive".to_owned(),
            ));
        }
        if self
            .max_all_verified_inconsistent_camera_fraction
            .is_some_and(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
        {
            return Err(SfmError::InvalidConfig(
                "global inconsistent-camera fraction must be in [0, 1]".to_owned(),
            ));
        }
        Ok(())
    }

    /// Post-fit trust failures for `evidence`; empty when the finish was accepted, translation
    /// evidence exists, and the rotation p90 and the optional reprojection RMSE and
    /// camera-balanced fraction are within limits.
    #[must_use]
    pub fn rejection_reasons(
        self,
        evidence: PostFitTrustEvidence,
    ) -> Vec<CandidateRejectionReason> {
        let mut reasons = Vec::new();
        if !evidence.finish_accepted {
            reasons.push(CandidateRejectionReason::FinishRejected);
        }
        if evidence.translation_graph_p90_deg.is_none() {
            reasons.push(CandidateRejectionReason::MissingTranslationGraphEvidence);
        }
        check_graph_residual(
            evidence.rotation_graph_p90_deg,
            self.max_rotation_graph_p90_deg,
            CandidateRejectionReason::MissingRotationGraphEvidence,
            CandidateRejectionReason::RotationGraphResidual,
            &mut reasons,
        );
        if let (Some(limit), Some(rmse)) = (
            self.max_final_reprojection_rmse_px,
            evidence.final_reprojection_rmse_px,
        ) && !within(rmse, limit)
        {
            reasons.push(CandidateRejectionReason::ReprojectionRmse);
        }
        if let Some(max_fraction) = self.max_all_verified_inconsistent_camera_fraction {
            if evidence.all_verified_camera_count == 0 {
                reasons.push(CandidateRejectionReason::MissingCameraBalancedGraphEvidence);
            } else if evidence.all_verified_inconsistent_camera_count as f64
                / evidence.all_verified_camera_count as f64
                > max_fraction
            {
                reasons.push(CandidateRejectionReason::CameraBalancedGraphTail);
            }
        }
        reasons
    }
}

/// Admission policy for the adaptive Global cascade.
///
/// Candidate order is not configurable: the inexpensive direction-only L1-IRLS candidate is
/// always tried first and global positioning is the only escalation. The thresholds are serialized so a
/// report completely describes the decision that was applied.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GlobalPolicy {
    rotation_consensus: RotationConsensusPolicy,
    metric_core: MetricCoreCertificationPolicy,
    post_fit_trust: PostFitTrustPolicy,
}

config_accessors!(GlobalPolicy {
    /// Post-fit pose-graph trust thresholds.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy post_fit_trust, with_post_fit_trust: PostFitTrustPolicy;
});

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl GlobalPolicy {
    /// All-verified rotation consensus thresholds.
    #[must_use]
    pub const fn rotation_consensus(&self) -> RotationConsensusPolicy {
        self.rotation_consensus
    }

    /// Structural and fixed-rotation numerical core certificate.
    #[must_use]
    pub const fn metric_core(&self) -> MetricCoreCertificationPolicy {
        self.metric_core
    }
}

impl Default for GlobalPolicy {
    fn default() -> Self {
        Self {
            rotation_consensus: RotationConsensusPolicy::default(),
            metric_core: MetricCoreCertificationPolicy,
            post_fit_trust: PostFitTrustPolicy::default(),
        }
    }
}

impl GlobalPolicy {
    /// Escalation order: global positioning, then direction-only L1-IRLS. Positioning from track
    /// bearings usually starts bundle adjustment closer to the solution, and the first admitted
    /// candidate wins.
    pub const CANDIDATES: [GlobalCandidate; 2] = [
        GlobalCandidate::GlobalPositioning,
        GlobalCandidate::TranslationAveraging,
    ];

    /// Return the fixed escalation order.
    #[must_use]
    pub const fn ordered_candidates(self) -> [GlobalCandidate; 2] {
        Self::CANDIDATES
    }

    /// Validate user-controlled thresholds before changing a workspace.
    pub fn validate(self) -> Result<()> {
        self.rotation_consensus.validate()?;
        self.post_fit_trust.validate()?;
        Ok(())
    }

    /// Decide whether a completed candidate can be exposed as the core.
    ///
    /// The all-verified rotation evidence must exist, but its inconsistent-camera tail gates the
    /// averaged rotations before translation (see [`Self::rotation_rejection_reasons`]) and is only
    /// reported here: bundle adjustment legitimately moves a camera away from wrong pair edges, so
    /// after it that tail rejected candidates closer to ground truth than the ones it let through.
    #[must_use]
    pub fn admit(self, evidence: CandidateAdmissionEvidence) -> CandidateAdmission {
        let mut reasons = Vec::new();
        reasons.extend(
            self.post_fit_trust
                .rejection_reasons(evidence.post_fit_trust),
        );
        if evidence.rotation_consensus.all_verified_camera_count == 0 {
            reasons.push(CandidateRejectionReason::MissingAllVerifiedRotationCameraEvidence);
        }
        reasons.extend(self.metric_core.rejection_reasons(evidence.metric_core));

        if reasons.is_empty() {
            CandidateAdmission::Accepted
        } else {
            CandidateAdmission::Rejected { reasons }
        }
    }

    /// Evaluate rotation-only evidence that is available before translation averaging.
    #[must_use]
    pub fn rotation_rejection_reasons(
        self,
        evidence: RotationConsensusEvidence,
    ) -> Vec<CandidateRejectionReason> {
        self.rotation_consensus.rejection_reasons(evidence)
    }
}

fn check_graph_residual(
    residual: Option<f64>,
    limit: f64,
    missing: CandidateRejectionReason,
    excessive: CandidateRejectionReason,
    reasons: &mut Vec<CandidateRejectionReason>,
) {
    match residual {
        Some(value) if within(value, limit) => {}
        Some(_) => reasons.push(excessive),
        None => reasons.push(missing),
    }
}

fn within(value: f64, limit: f64) -> bool {
    value.is_finite() && value <= limit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepted_evidence() -> CandidateAdmissionEvidence {
        CandidateAdmissionEvidence {
            rotation_consensus: RotationConsensusEvidence {
                all_verified_camera_count: 10,
                all_verified_inconsistent_camera_count: 0,
            },
            metric_core: MetricCoreCertificationEvidence {
                fixed_rotation_observability: Some(FixedRotationObservabilityEvidence {
                    pose_rank: 8,
                    pose_dimension: 8,
                    weak_point_count: Some(0),
                }),
            },
            post_fit_trust: PostFitTrustEvidence {
                finish_accepted: true,
                translation_graph_p90_deg: Some(1.0),
                final_reprojection_rmse_px: Some(1.0),
                rotation_graph_p90_deg: Some(1.0),
                all_verified_camera_count: 10,
                all_verified_inconsistent_camera_count: 0,
            },
        }
    }

    #[test]
    fn candidate_order_is_fixed_and_positioning_first() {
        assert_eq!(
            GlobalPolicy::default().ordered_candidates(),
            [
                GlobalCandidate::GlobalPositioning,
                GlobalCandidate::TranslationAveraging,
            ]
        );
    }

    #[test]
    fn admission_requires_complete_safe_evidence() {
        let policy = GlobalPolicy::default();
        let accepted = policy.admit(accepted_evidence());
        assert!(accepted.is_accepted());
        assert_eq!(accepted, CandidateAdmission::Accepted);

        let rejected = CandidateAdmissionEvidence {
            post_fit_trust: PostFitTrustEvidence {
                final_reprojection_rmse_px: Some(20.0),
                ..accepted_evidence().post_fit_trust
            },
            metric_core: MetricCoreCertificationEvidence {
                fixed_rotation_observability: Some(FixedRotationObservabilityEvidence {
                    pose_rank: 7,
                    weak_point_count: Some(2),
                    ..accepted_evidence()
                        .metric_core
                        .fixed_rotation_observability
                        .unwrap()
                }),
            },
            ..accepted_evidence()
        };
        assert_eq!(
            policy.admit(rejected).rejection_reasons(),
            &[
                CandidateRejectionReason::ReprojectionRmse,
                CandidateRejectionReason::CameraRankDeficient,
                CandidateRejectionReason::WeakPointGeometry,
            ]
        );
    }

    #[test]
    fn missing_evidence_fails_closed() {
        let policy = GlobalPolicy::default();
        let admission = policy.admit(CandidateAdmissionEvidence {
            rotation_consensus: RotationConsensusEvidence {
                all_verified_camera_count: 0,
                all_verified_inconsistent_camera_count: 0,
            },
            metric_core: MetricCoreCertificationEvidence {
                fixed_rotation_observability: None,
            },
            post_fit_trust: PostFitTrustEvidence {
                finish_accepted: true,
                translation_graph_p90_deg: None,
                final_reprojection_rmse_px: None,
                rotation_graph_p90_deg: None,
                all_verified_camera_count: 0,
                all_verified_inconsistent_camera_count: 0,
            },
        });
        assert_eq!(
            admission.rejection_reasons(),
            &[
                CandidateRejectionReason::MissingTranslationGraphEvidence,
                CandidateRejectionReason::MissingRotationGraphEvidence,
                CandidateRejectionReason::MissingAllVerifiedRotationCameraEvidence,
                CandidateRejectionReason::MissingObservabilityEvidence,
            ]
        );
    }

    #[test]
    fn policy_validation_rejects_invalid_thresholds() {
        assert!(
            GlobalPolicy {
                post_fit_trust: PostFitTrustPolicy {
                    max_rotation_graph_p90_deg: 0.0,
                    ..PostFitTrustPolicy::default()
                },
                ..GlobalPolicy::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn only_the_final_rmse_vetoes_a_noisy_translation_graph() {
        let policy = GlobalPolicy::default();
        let with = |post_fit_trust| CandidateAdmissionEvidence {
            post_fit_trust,
            ..accepted_evidence()
        };
        let base = accepted_evidence().post_fit_trust;
        let noisy_weak_edges = PostFitTrustEvidence {
            translation_graph_p90_deg: Some(7.5),
            ..base
        };
        assert!(policy.admit(with(noisy_weak_edges)).is_accepted());
        let high_rmse = PostFitTrustEvidence {
            final_reprojection_rmse_px: Some(15.0),
            ..base
        };
        assert_eq!(
            policy.admit(with(high_rmse)).rejection_reasons(),
            &[CandidateRejectionReason::ReprojectionRmse]
        );
        let unchecked_rmse = GlobalPolicy::default().with_post_fit_trust(
            PostFitTrustPolicy::default().with_max_final_reprojection_rmse_px(None),
        );
        assert!(unchecked_rmse.admit(with(high_rmse)).is_accepted());
    }

    #[test]
    fn camera_balanced_tail_gate_is_optional_and_fail_closed_when_enabled() {
        let evidence = CandidateAdmissionEvidence {
            post_fit_trust: PostFitTrustEvidence {
                all_verified_camera_count: 100,
                all_verified_inconsistent_camera_count: 2,
                ..accepted_evidence().post_fit_trust
            },
            ..accepted_evidence()
        };
        assert!(GlobalPolicy::default().admit(evidence).is_accepted());

        let policy = GlobalPolicy {
            post_fit_trust: PostFitTrustPolicy {
                max_all_verified_inconsistent_camera_fraction: Some(0.01),
                ..PostFitTrustPolicy::default()
            },
            ..GlobalPolicy::default()
        };
        assert_eq!(
            policy.admit(evidence).rejection_reasons(),
            &[CandidateRejectionReason::CameraBalancedGraphTail]
        );
    }

    #[test]
    fn all_verified_rotation_consensus_is_camera_balanced() {
        let policy = GlobalPolicy::default();
        let consensus = |inconsistent| RotationConsensusEvidence {
            all_verified_camera_count: 20,
            all_verified_inconsistent_camera_count: inconsistent,
        };
        assert!(policy.rotation_rejection_reasons(consensus(2)).is_empty());
        assert_eq!(
            policy.rotation_rejection_reasons(consensus(3)),
            &[CandidateRejectionReason::AllVerifiedRotationCameraTail]
        );
    }

    #[test]
    fn rotation_consensus_after_bundle_adjustment_is_reported_not_admission() {
        let evidence = CandidateAdmissionEvidence {
            rotation_consensus: RotationConsensusEvidence {
                all_verified_camera_count: 20,
                all_verified_inconsistent_camera_count: 10,
            },
            ..accepted_evidence()
        };
        assert!(GlobalPolicy::default().admit(evidence).is_accepted());
    }
}
