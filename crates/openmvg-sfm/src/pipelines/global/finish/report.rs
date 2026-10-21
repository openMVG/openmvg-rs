// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serialized receipts of the global finish stages.

use crate::ba::backend::BaPreflightReport;
use crate::{Result, SfmError, ViewId};

/// One bounded robust-triangulation/full-BA cleanup round in the global schedule.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalRetriangulationRoundReport {
    /// Robust retriangulation of tracks seen by at least two posed views.
    pub retriangulation: crate::ops::reconstruction::RetriangulationReport,
    /// Cauchy pose/point bundle adjustment after the retriangulation.
    pub bundle_adjustment: crate::ops::reconstruction::BundleReport,
    /// Landmarks rejected by the reprojection cleanup after the BA.
    pub removed_landmark_count: usize,
    /// Observations removed by that cleanup from landmarks that stay active.
    #[serde(default)]
    pub removed_observation_count: usize,
}

/// Final global refinement that frees each intrinsic group's declared camera model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalSelfCalibrationReport {
    /// Self-calibrating BA from the robust rounds' result. Groups their observations do not
    /// constrain keep their input intrinsics and are listed in its diagnostics.
    pub bundle_adjustment: crate::ops::reconstruction::BundleReport,
    /// Landmarks rejected by the reprojection cleanup after that BA.
    pub removed_landmark_count: usize,
    /// Observations removed by that cleanup from landmarks that stay active.
    pub removed_observation_count: usize,
    /// Robust retriangulation under the refined intrinsics, followed by a second
    /// self-calibrating BA and cleanup.
    pub retriangulation: GlobalRetriangulationRoundReport,
    /// Verified relative-pose graph residuals of the committed reconstruction. The finish
    /// report's graph residuals describe the admitted, uncalibrated candidate.
    pub translation_graph_p90_after_deg: Option<f64>,
    /// p90 relative-rotation residual of the committed reconstruction over the verified graph,
    /// in degrees; `None` when no posed edge carries a relative rotation.
    pub rotation_graph_p90_after_deg: Option<f64>,
}

/// Outcome of self-calibrating an admitted global candidate.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalSelfCalibrationStatus {
    /// The refined intrinsics and reconstruction were kept.
    Committed(Box<GlobalSelfCalibrationReport>),
    /// The admitted, uncalibrated reconstruction was restored.
    RolledBack(GlobalSelfCalibrationRejection),
}

/// Why a self-calibration proposal was rolled back.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalSelfCalibrationRejection {
    /// A self-calibrating bundle adjustment failed numerically or was not applicable.
    Refinement {
        /// `Display` text of the bundle-adjustment error.
        error: String,
    },
    /// The refined reconstruction degraded the verified relative-pose graph.
    GraphDegraded {
        /// p90 translation-direction residual of the admitted reconstruction, in degrees.
        translation_graph_p90_before_deg: Option<f64>,
        /// p90 translation-direction residual of the refined proposal, in degrees.
        translation_graph_p90_after_deg: Option<f64>,
        /// p90 relative-rotation residual of the admitted reconstruction, in degrees.
        rotation_graph_p90_before_deg: Option<f64>,
        /// p90 relative-rotation residual of the refined proposal, in degrees.
        rotation_graph_p90_after_deg: Option<f64>,
        /// Locally trapped views in the admitted reconstruction.
        locally_trapped_view_count_before: usize,
        /// Locally trapped views in the refined proposal.
        locally_trapped_view_count_after: usize,
        /// Cameras the refined reconstruction leaves locally trapped.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        trapped_views_after: Vec<GlobalTrappedViewDiagnostic>,
    },
}

/// One camera whose incident verified relative poses identify a locally inconsistent pose.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalTrappedViewDiagnostic {
    /// The trapped camera's view.
    pub view: ViewId,
    /// Degree in the active verified pair graph, independent of residual availability.
    pub active_pair_degree: usize,
    /// Finite verified rotation residuals supporting the median below.
    pub rotation_residual_support: usize,
    /// Finite verified translation-direction residuals supporting the median below.
    pub translation_residual_support: usize,
    /// Lower median of the incident verified rotation residuals, in degrees; the camera is
    /// trapped above 5°. `None` with fewer than two residuals.
    pub median_rotation_residual_deg: Option<f64>,
    /// Lower median of the incident translation-direction residuals, in degrees; the camera is
    /// trapped above 10°. `None` with fewer than two residuals.
    pub median_translation_residual_deg: Option<f64>,
}

/// Typed pose-trust evidence from the verified relative-pose graph, not ground truth.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalPoseTrustReport {
    /// Serialized schema version of this report; currently 1.
    #[serde(default = "global_pose_trust_report_version")]
    pub schema_version: u32,
    /// Trapped cameras of the candidate entering the finish.
    #[serde(default)]
    pub before_trapped_views: Vec<GlobalTrappedViewDiagnostic>,
    /// Trapped cameras right after the fixed-rotation BA, before any camera deferral.
    #[serde(default)]
    pub post_fixed_rotation_trapped_views: Vec<GlobalTrappedViewDiagnostic>,
    /// Trapped cameras of the first released-rotation BA proposal; `None` when that BA
    /// returned an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_full_ba_candidate_trapped_views: Option<Vec<GlobalTrappedViewDiagnostic>>,
    /// Trapped cameras of the robust-round proposal rolled back for degrading the graph; `None`
    /// when no round was rejected that way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_robust_candidate_trapped_views: Option<Vec<GlobalTrappedViewDiagnostic>>,
    /// Final candidate before any outer whole-finish rollback.
    #[serde(default)]
    pub final_candidate_trapped_views: Vec<GlobalTrappedViewDiagnostic>,
}

/// Why the conservative post-fixed-rotation cleanup was not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalPostFixedRotationCleanupNotAttempted {
    /// The cleanup config is absent or disabled.
    NotConfigured,
    /// The fixed-rotation BA trapped no camera that was not already trapped before it.
    NoNewTrappedViews,
}

/// Typed reason a proposed camera deferral was rejected and rolled back.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum GlobalPostFixedRotationCleanupRejection {
    /// More newly trapped views than the `max_views` / `max_fraction` budget allows.
    CandidateLimitExceeded,
    /// The primary component or the rerun problem does not have a free scale gauge.
    GaugeNotFree,
    /// Topology analysis of the posed verified-pair graph failed.
    TopologyAnalysisFailed {
        /// `Display` text of the analysis error.
        error: String,
    },
    /// Some candidates are articulation points of the posed verified-pair graph.
    ContainsArticulationViews {
        /// Candidates whose removal would split the graph.
        images: Vec<ViewId>,
    },
    /// Without the candidates, fewer than two posed views remain or the graph is not connected.
    RetainedGraphDisconnected {
        /// Connected components of the retained graph.
        component_count: usize,
    },
    /// The config was invalid, or clearing poses, rejecting landmarks, retriangulating, or
    /// building the BA problem failed.
    MutationFailed {
        /// `Display` text of the error.
        error: String,
    },
    /// The BA backend's preflight for the fixed-rotation rerun returned a failure.
    FixedRotationPreflightFailed {
        /// `Display` text of the backend failure.
        error: String,
    },
    /// The BA backend's preflight declined to run the fixed-rotation rerun.
    FixedRotationPreflightRejected {
        /// Preflight decision and problem measurements.
        report: Box<BaPreflightReport>,
    },
    /// The fixed-rotation rerun bundle adjustment failed.
    FixedRotationRerunFailed {
        /// `Display` text of the error.
        error: String,
    },
    /// The rerun did not adjust exactly the retained posed cameras.
    IncompleteCameraCoverage {
        /// Retained posed cameras the rerun had to adjust.
        expected: usize,
        /// Free cameras the preflight covered; `None` when the backend did not report it.
        preflight_covered: Option<usize>,
        /// Cameras the rerun adjusted.
        adjusted: usize,
    },
    /// The rerun's preflight and solve do not certify full camera observability.
    ObservabilityNotCertified,
    /// The rerun degraded the verified relative-pose graph residuals.
    GraphRegression,
    /// The rerun left cameras trapped that were not trapped before the finish, or more of them.
    RetainedTrappedViews {
        /// Trapped cameras absent from the pre-finish trapped set.
        images: Vec<ViewId>,
    },
    /// The workspace or the cleanup receipt failed validation after the rerun.
    InvalidWorkspace {
        /// `Display` text of the validation error.
        error: String,
    },
    /// The rerun changed the prepared pair or track domains.
    PreparedDomainsChanged,
    /// The rerun changed the identity of landmarks that existed before the cleanup.
    LandmarkIdentityChanged,
}

/// Outcome of the optional conservative camera-deferral stage.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GlobalPostFixedRotationCleanupStatus {
    /// The stage did not run; the workspace is unchanged.
    NotAttempted {
        /// Why the stage did not run.
        reason: GlobalPostFixedRotationCleanupNotAttempted,
    },
    /// A deferral was proposed and rolled back.
    Rejected {
        /// Newly trapped views proposed for deferral, in ascending order.
        candidates: Vec<ViewId>,
        /// Deferral budget for this workspace; zero when the config was invalid.
        allowed_views: usize,
        /// Why the proposal was rolled back.
        reason: GlobalPostFixedRotationCleanupRejection,
    },
    /// The candidates were unposed and the fixed-rotation BA was rerun without them.
    Applied {
        /// Unposed cameras, in strictly ascending order; non-empty.
        deferred_views: Vec<ViewId>,
        /// Deferral budget for this workspace; at least `deferred_views.len()`.
        allowed_views: usize,
        /// Active landmarks rejected because fewer than two posed views still support them.
        rejected_under_supported_landmark_count: usize,
        /// Refit of the deferred cameras' tracks against the retained poses.
        retriangulation: crate::ops::reconstruction::RetriangulationReport,
        /// BA preflight of the rerun; its decision was to proceed.
        rerun_preflight: Box<BaPreflightReport>,
        /// Receipt for the first fixed-rotation solve. The enclosing finish report's
        /// `translation_and_points` field describes the retained rerun state.
        original_translation_and_points: Box<crate::ops::reconstruction::BundleReport>,
    },
}

/// Versioned receipt for the adaptive-only post-fixed-rotation cleanup stage.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalPostFixedRotationCleanupReport {
    /// Serialized schema version of this report; currently 1.
    #[serde(default = "global_post_cleanup_report_version")]
    pub schema_version: u32,
    /// What the stage did.
    pub outcome: GlobalPostFixedRotationCleanupStatus,
}

pub(crate) const fn global_post_cleanup_report_version() -> u32 {
    1
}

impl Default for GlobalPostFixedRotationCleanupReport {
    fn default() -> Self {
        Self {
            schema_version: global_post_cleanup_report_version(),
            outcome: GlobalPostFixedRotationCleanupStatus::NotAttempted {
                reason: GlobalPostFixedRotationCleanupNotAttempted::NotConfigured,
            },
        }
    }
}

impl GlobalPostFixedRotationCleanupReport {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version != global_post_cleanup_report_version() {
            return Err(SfmError::InvalidInput(
                "unsupported post-fixed-rotation cleanup report version".to_owned(),
            ));
        }
        match &self.outcome {
            GlobalPostFixedRotationCleanupStatus::NotAttempted { .. } => return Ok(()),
            GlobalPostFixedRotationCleanupStatus::Rejected { candidates, .. } => {
                if candidates.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return Err(SfmError::InvalidInput(
                        "invalid post-fixed-rotation cleanup receipt".to_owned(),
                    ));
                }
            }
            GlobalPostFixedRotationCleanupStatus::Applied {
                deferred_views,
                allowed_views,
                ..
            } => {
                if deferred_views.is_empty()
                    || deferred_views.windows(2).any(|pair| pair[0] >= pair[1])
                    || deferred_views.len() > *allowed_views
                {
                    return Err(SfmError::InvalidInput(
                        "invalid post-fixed-rotation cleanup receipt".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }
}

pub(super) const fn global_pose_trust_report_version() -> u32 {
    1
}

impl Default for GlobalPoseTrustReport {
    fn default() -> Self {
        Self {
            schema_version: global_pose_trust_report_version(),
            before_trapped_views: Vec::new(),
            post_fixed_rotation_trapped_views: Vec::new(),
            initial_full_ba_candidate_trapped_views: None,
            rejected_robust_candidate_trapped_views: None,
            final_candidate_trapped_views: Vec::new(),
        }
    }
}

/// Diagnostics for the staged global-SfM refinement.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalFinishReport {
    /// Whether the final candidate remained consistent with the independently verified
    /// relative-pose graph. A rejected candidate is rolled back transactionally.
    pub accepted: bool,
    /// Landmarks rejected by structural preparation before fixed-rotation optimization.
    pub removed_landmark_count_before_translation_and_points: usize,
    /// Posed views removed by structural preparation before fixed-rotation optimization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_views_before_translation_and_points: Vec<ViewId>,
    /// Fixed-rotation BA: camera translations and 3D points are free.
    pub translation_and_points: crate::ops::reconstruction::BundleReport,
    /// First full BA after rotations are released. Numerical, rank, or relative-pose graph
    /// rejection is transactional: the fixed-rotation result remains the valid candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_bundle_adjustment: Option<crate::ops::reconstruction::BundleReport>,
    /// Posed views whose support observes a translation but not a released-rotation pose,
    /// removed before the first full BA. Empty when that BA is rejected and rolled back.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_views_before_full_bundle_adjustment: Vec<ViewId>,
    /// Whether releasing rotations was attempted but rejected numerically or by graph sanity.
    #[serde(default)]
    pub rejected_full_bundle_adjustment: bool,
    /// Gross landmarks removed after the first full BA.
    pub removed_landmark_count_after_full_bundle_adjustment: usize,
    /// Gross observations removed after the first full BA while retaining their landmarks.
    #[serde(default)]
    pub removed_observation_count_after_full_bundle_adjustment: usize,
    /// Bounded robust retriangulation/full-BA cleanup rounds.
    pub robust_rounds: Vec<GlobalRetriangulationRoundReport>,
    /// Numerical or graph-inconsistent robust-round proposals rolled back while retaining the
    /// last valid stage.
    pub rejected_robust_round_count: usize,
    /// Self-calibration of the admitted candidate. `None` when it is disabled or the candidate
    /// was not admitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_calibration: Option<GlobalSelfCalibrationStatus>,
    /// p90 translation-direction residual over verified edges with at least 5° median parallax,
    /// for the candidate entering the finish, in degrees; `None` without such an edge.
    pub translation_graph_p90_before_deg: Option<f64>,
    /// The same p90 for the final candidate, in degrees; `None` without such an edge.
    pub translation_graph_p90_after_deg: Option<f64>,
    /// Reprojection RMSE of the last accepted bundle adjustment, in pixels; `None` when it
    /// reported other residual units.
    #[serde(default)]
    pub final_reprojection_rmse_px: Option<f64>,
    /// Fraction of finite, non-zero active view-graph baselines whose solved orientation agrees
    /// with the independently verified relative translation. This is an in-sample sanity check,
    /// not a correctness proof.
    #[serde(default)]
    pub positive_scale_fraction_before: Option<f64>,
    /// The same fraction for the final candidate; `None` without an oriented baseline.
    #[serde(default)]
    pub positive_scale_fraction_after: Option<f64>,
    /// p90 relative-rotation residual over verified edges for the candidate entering the
    /// finish, in degrees; `None` without a posed edge carrying a rotation.
    pub rotation_graph_p90_before_deg: Option<f64>,
    /// The same p90 for the final candidate, in degrees.
    pub rotation_graph_p90_after_deg: Option<f64>,
    /// Cameras whose incident verified edges mostly disagree with their pose (lower median above
    /// 5° in rotation or 10° in translation direction), entering the finish.
    pub locally_trapped_view_count_before: usize,
    /// The same count for the final candidate.
    pub locally_trapped_view_count_after: usize,
    /// Optional adaptive-only camera deferral between the two BA scopes.
    #[serde(default)]
    pub post_fixed_rotation_cleanup: GlobalPostFixedRotationCleanupReport,
    /// Per-camera localization of the existing aggregate pose-trust counts.
    #[serde(default)]
    pub pose_trust: GlobalPoseTrustReport,
}
