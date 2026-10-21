// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IntrinsicId;

pub use crate::workflow::run::{
    FinalRoute, GlobalEmptyReason, RUN_REPORT_SCHEMA_VERSION, RunReport, ViewOmissionReason,
    ViewVerdict, ViewVerdictStatus,
};
// Reached from `RunReport` through `ViewOmissionReason::Incremental`.
pub use crate::pipelines::incremental::{RegistrationFailureReason, UnregisteredViewDiagnostic};

/// What one front-end run produced. Deterministic for a given input; stage durations are left to
/// an [`SfmProgressSink`](crate::progress::SfmProgressSink).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct FrontendReport {
    /// Views considered by the front end.
    pub view_count: usize,
    /// Views with a populated feature record.
    pub extracted_view_count: usize,
    /// Total detected keypoints across all images.
    #[serde(default)]
    pub keypoint_count: usize,
    /// Image pairs for which matching produced a stored result, including empty results.
    pub match_record_count: usize,
    /// Image pairs containing at least one raw descriptor match.
    #[serde(default)]
    pub raw_match_pair_count: usize,
    /// Total descriptor matches before geometric verification.
    #[serde(default)]
    pub raw_match_count: usize,
    /// Image pairs accepted by geometric verification, before strongest-pairs filtering.
    pub verified_pair_count: usize,
    /// Total accepted geometric inliers across verified pairs, before strongest-pairs filtering.
    #[serde(default)]
    pub verified_inlier_count: usize,
    /// Verified pairs the strongest-pairs filter deactivated before tracking; zero when it is off.
    #[serde(default)]
    pub filtered_pair_count: usize,
    /// Multi-view tracks built from the verified matches.
    pub track_count: usize,
    /// Feature extraction method of the run; `None` when a replacement extractor supplied the
    /// features.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_method: Option<crate::ExtractionMethod>,
    /// Radial pre-calibration, when requested and some camera group has unknown distortion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radial_calibration: Option<RadialCalibrationReport>,
}

/// One camera group's radial-distortion estimate in one calibration round.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct RadialGroupEstimate {
    /// Camera group (shared intrinsic) the estimate belongs to.
    pub intrinsic_id: IntrinsicId,
    /// Verified pairs inside the group with enough inliers to profile.
    pub pair_count: usize,
    /// Pairs whose residual is lower at the selected distortion than without distortion.
    pub improved_pair_count: usize,
    /// Selected distortion at the farthest image corner, `k1 * r_max^2`.
    pub corner_distortion: f64,
    /// Selected radial coefficient in normalized image coordinates, `corner_distortion / r_max^2`.
    pub k1: f64,
    /// Median over pairs of the residual ratio against the undistorted model.
    pub cost_ratio: f64,
    /// The selected distortion is an interior grid minimum that either passes the sign test or
    /// refines the model the group was verified with.
    pub accepted: bool,
}

/// Per-round estimates of the radial-distortion calibration.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct RadialCalibrationReport {
    /// One entry per calibration round, each holding one estimate per profiled camera group.
    pub rounds: Vec<Vec<RadialGroupEstimate>>,
    /// The adopted models stopped changing before the round limit.
    pub settled: bool,
}
