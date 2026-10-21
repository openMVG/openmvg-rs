// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Final validation of a tentatively admitted global core.

use crate::{
    SfmWorkspace,
    events::{EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventTransition},
};

use crate::ops::reconstruction::{
    refresh_primary_component, retain_posed_landmark_support, retain_primary_bundle_component,
};
use crate::ops::track_scale::{TrackScaleValidationReport, validate_track_scale_consistency};

use super::{
    GlobalRuntime,
    config::{GlobalSfmConfig, GlobalTrackValidationConfig},
    report::{
        GlobalCandidateAttemptReport, GlobalCandidatePostAdmissionFailure, GlobalCandidateStatus,
        GlobalNoReconstructionReason, GlobalTrackValidationReport,
    },
};

pub(super) struct GlobalAcceptedResult {
    pub(super) selected_attempt: usize,
    pub(super) workspace: SfmWorkspace,
    pub(super) registered_view_count: usize,
    pub(super) input_view_count: usize,
    pub(super) track_validation: Option<Box<GlobalTrackValidationReport>>,
}

pub(super) enum PostAdmissionResult {
    Accepted {
        attempt: GlobalCandidateAttemptReport,
        result: Box<GlobalAcceptedResult>,
    },
    Retry {
        attempt: GlobalCandidateAttemptReport,
        reason: GlobalNoReconstructionReason,
    },
}

pub(super) fn validate_admitted_candidate(
    mut workspace: SfmWorkspace,
    attempt: GlobalCandidateAttemptReport,
    selected_attempt: usize,
    config: &GlobalSfmConfig,
    runtime: GlobalRuntime<'_>,
    branch: SfmEventBranch,
    events: &mut EventEmitter<'_>,
) -> PostAdmissionResult {
    let track_validation = match config.independent_track_validation {
        Some(validation_config) => match validate_track_scale_consistency(
            &workspace,
            runtime.robust_triangulator,
            validation_config.validation(),
        ) {
            Ok(report) if validation_config.core_is_rejected(&report) => {
                match peel_tail(
                    &mut workspace,
                    &attempt,
                    &report,
                    validation_config,
                    runtime,
                ) {
                    Some(peeled) => Some(Box::new(peeled)),
                    None => {
                        let report = GlobalTrackValidationReport::new(report, Vec::new());
                        discard_admitted_candidate(events, branch, &workspace);
                        return independent_validation_retry(
                            attempt,
                            selected_attempt,
                            Some(report),
                            None,
                        );
                    }
                }
            }
            Ok(report) => Some(Box::new(GlobalTrackValidationReport::new(
                report,
                Vec::new(),
            ))),
            Err(error) => {
                discard_admitted_candidate(events, branch, &workspace);
                return independent_validation_retry(
                    attempt,
                    selected_attempt,
                    None,
                    Some(error.to_string()),
                );
            }
        },
        None => None,
    };

    PostAdmissionResult::Accepted {
        attempt,
        result: Box::new(GlobalAcceptedResult {
            selected_attempt,
            registered_view_count: workspace.registered_view_count(),
            input_view_count: workspace.views().len(),
            workspace,
            track_validation,
        }),
    }
}

/// Remove the inconsistent cameras of a rejected core and validate what is left, once.
///
/// [`GlobalTrackValidationConfig::peelable_views`] decides whether the inconsistent cameras are a
/// tail of a sound core; a core that is wrong as a whole is left to the cascade. The tail is
/// dropped together with the cameras only it tied to the core, and landmarks left without two
/// posed observations are rejected. Returns the validation of the peeled core, or `None` if
/// nothing was peeled or the rest still fails.
fn peel_tail(
    workspace: &mut SfmWorkspace,
    attempt: &GlobalCandidateAttemptReport,
    rejected: &TrackScaleValidationReport,
    config: GlobalTrackValidationConfig,
    runtime: GlobalRuntime<'_>,
) -> Option<GlobalTrackValidationReport> {
    let GlobalCandidateStatus::Accepted { evidence, .. } = &attempt.outcome else {
        return None;
    };
    let graph = evidence.post_fit_trust;
    let mut peeled = config.peelable_views(
        rejected,
        graph.all_verified_camera_count,
        graph.all_verified_inconsistent_camera_count,
    )?;
    workspace.clear_poses_for_views(&peeled).ok()?;
    // Landmarks keep only their posed observations, so none lists a peeled view as support.
    retain_posed_landmark_support(workspace).ok()?;
    // Cameras that only the peeled ones tied to the core go with them.
    peeled.extend(
        retain_primary_bundle_component(workspace)
            .ok()?
            .removed_views,
    );
    // The stragglers removed above may have supported landmarks too.
    retain_posed_landmark_support(workspace).ok()?;
    peeled.sort_unstable();
    peeled.dedup();
    refresh_primary_component(workspace);
    workspace.validate().ok()?;
    let report = validate_track_scale_consistency(
        workspace,
        runtime.robust_triangulator,
        config.validation(),
    )
    .ok()?;
    (!config.core_is_rejected(&report)).then(|| GlobalTrackValidationReport::new(report, peeled))
}

fn discard_admitted_candidate(
    events: &mut EventEmitter<'_>,
    branch: SfmEventBranch,
    workspace: &SfmWorkspace,
) {
    events.emit(
        branch,
        SfmEventPhase::GlobalCandidateAdmission,
        SfmEventTransition::Discarded,
        workspace,
    );
}

fn independent_validation_retry(
    mut attempt: GlobalCandidateAttemptReport,
    selected_attempt: usize,
    report: Option<GlobalTrackValidationReport>,
    error: Option<String>,
) -> PostAdmissionResult {
    let report = report.map(Box::new);
    attempt.post_admission_failure = Some(
        GlobalCandidatePostAdmissionFailure::IndependentTrackValidation {
            report: report.clone(),
            error: error.clone(),
        },
    );
    PostAdmissionResult::Retry {
        attempt,
        reason: GlobalNoReconstructionReason::IndependentTrackValidation {
            selected_attempt,
            report,
            error,
        },
    }
}
