// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Global engine orchestration: try the positioning candidates in escalation order and commit
//! the first admitted core.

use crate::{
    Result, SfmError, SfmWorkspace,
    events::{EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventTransition},
};

use super::{
    GlobalRuntime,
    admission::{GlobalAcceptedResult, PostAdmissionResult, validate_admitted_candidate},
    candidate::{CandidateAttempt, CandidateRun, run_candidate},
    config::GlobalSfmConfig,
    policy::GlobalCandidate,
    report::{
        GlobalCandidateAttemptReport, GlobalCandidateRoute, GlobalCompletionReport,
        GlobalNoReconstructionReason, GlobalSfmReport,
    },
};

/// The pristine input is retained only to rebuild a later candidate and to return unchanged when
/// no candidate passes the admission checks.
pub(crate) fn global_core_from_owned_prepared_rotations_with_events<F>(
    pristine_workspace: SfmWorkspace,
    global_prepared: SfmWorkspace,
    config: &GlobalSfmConfig,
    rotation_preparation: &crate::ops::averaging::GlobalRotationPreparation,
    runtime: GlobalRuntime<'_>,
    mut rebuild_global_prepared: F,
    events: &mut EventEmitter<'_>,
) -> Result<(SfmWorkspace, GlobalSfmReport)>
where
    F: FnMut(&SfmWorkspace) -> Result<SfmWorkspace>,
{
    config.validate()?;
    validate_uninitialized_prepared_workspace(&pristine_workspace)?;
    validate_uninitialized_prepared_workspace(&global_prepared)?;
    validate_prepared_domains(&pristine_workspace, &global_prepared)?;

    let rotation_preflight =
        crate::ops::averaging::rotation_averaging_consistency_from_preparation(
            &global_prepared,
            rotation_preparation,
            crate::ops::averaging::VerifiedPairSelection::All,
        )?;
    let mut first_global_prepared = Some(global_prepared);
    let cascade = run_candidate_cascade(
        rotation_preflight,
        config,
        rotation_preparation,
        runtime,
        |_| {
            if let Some(prepared) = first_global_prepared.take() {
                Ok(prepared)
            } else {
                rebuild_global_prepared(&pristine_workspace)
            }
        },
        events,
    )?;
    complete_global_core_run(cascade, config, pristine_workspace, events)
}

struct GlobalCandidateCascade {
    rotation_preflight: crate::ops::averaging::RotationAveragingConsistencyReport,
    attempts: Vec<GlobalCandidateAttemptReport>,
    accepted: Option<GlobalAcceptedResult>,
    no_reconstruction_reason: GlobalNoReconstructionReason,
}

fn run_candidate_cascade<F>(
    rotation_preflight: crate::ops::averaging::RotationAveragingConsistencyReport,
    config: &GlobalSfmConfig,
    rotation_preparation: &crate::ops::averaging::GlobalRotationPreparation,
    runtime: GlobalRuntime<'_>,
    mut candidate_workspace: F,
    events: &mut EventEmitter<'_>,
) -> Result<GlobalCandidateCascade>
where
    F: FnMut(GlobalCandidate) -> Result<SfmWorkspace>,
{
    let rotation_preflight_rejections = config.policy.rotation_rejection_reasons(
        config.policy.rotation_consensus().evidence(
            rotation_preflight
                .cameras
                .iter()
                .map(|camera| camera.median_rotation_residual_deg),
        ),
    );
    let rotation_preflight_accepted = rotation_preflight_rejections.is_empty();

    let mut attempts = Vec::with_capacity(6);
    let mut accepted = None;
    let mut no_reconstruction_reason = if rotation_preflight_rejections.is_empty() {
        GlobalNoReconstructionReason::NoCandidateAdmitted
    } else {
        GlobalNoReconstructionReason::RotationConsensusRejected {
            reasons: rotation_preflight_rejections,
        }
    };

    // Preserve the stronger three-view fast path for every candidate. Only when no core passes
    // admission do we retry full-domain two-view tracks. The final bounded recovery combines
    // explicit two-view triangulation with a generically parallel-rigid, biconnected direction
    // core. One final mixed-positioning retry extracts the primary camera--landmark incidence
    // component. Exact BA observability and ordinary admission still decide whether either core
    // may commit.
    let candidates = candidate_attempt_order(config.policy.ordered_candidates());
    'candidate_cascade: for (candidate, route) in candidates
        .into_iter()
        .filter(|_| rotation_preflight_accepted)
    {
        // A candidate records a canceled stage as its failure, so the next attempt re-checks.
        crate::progress::cancellation_point(events.progress())?;
        let index = attempts.len();
        let branch = SfmEventBranch::GlobalCandidate {
            attempt: index,
            candidate,
        };
        let candidate_workspace = candidate_workspace(candidate)?;
        let run = run_candidate(
            candidate_workspace,
            config,
            CandidateAttempt {
                candidate,
                route,
                branch,
            },
            rotation_preparation,
            runtime,
            events,
        );
        match run {
            CandidateRun::Accepted { report, workspace } => {
                crate::progress::cancellation_point(events.progress())?;
                match validate_admitted_candidate(
                    *workspace, *report, index, config, runtime, branch, events,
                ) {
                    PostAdmissionResult::Accepted { attempt, result } => {
                        attempts.push(attempt);
                        accepted = Some(*result);
                        break 'candidate_cascade;
                    }
                    PostAdmissionResult::Retry { attempt, reason } => {
                        attempts.push(attempt);
                        no_reconstruction_reason = reason;
                    }
                }
            }
            CandidateRun::NotAccepted {
                report,
                abort_global,
            } => {
                attempts.push(*report);
                if let Some(reason) = abort_global {
                    no_reconstruction_reason = reason;
                    break 'candidate_cascade;
                }
            }
        }
    }

    Ok(GlobalCandidateCascade {
        rotation_preflight,
        attempts,
        accepted,
        no_reconstruction_reason,
    })
}

fn candidate_attempt_order(
    candidates: [GlobalCandidate; 2],
) -> [(GlobalCandidate, GlobalCandidateRoute); 6] {
    let [first, second] = candidates;
    [
        (first, GlobalCandidateRoute::FullDomainThreeView),
        (second, GlobalCandidateRoute::FullDomainThreeView),
        (first, GlobalCandidateRoute::FullDomainTwoView),
        (second, GlobalCandidateRoute::FullDomainTwoView),
        (first, recovery_route(first)),
        (second, recovery_route(second)),
    ]
}

/// Bounded recovery route of each candidate: the direction solver needs a parallel-rigid core,
/// and global positioning needs the full active edge set.
const fn recovery_route(candidate: GlobalCandidate) -> GlobalCandidateRoute {
    match candidate {
        GlobalCandidate::TranslationAveraging => GlobalCandidateRoute::RigidIncidenceCoreTwoView,
        GlobalCandidate::GlobalPositioning => {
            GlobalCandidateRoute::FullDomainIncidenceCoreThreeView
        }
    }
}

fn complete_global_core_run(
    cascade: GlobalCandidateCascade,
    config: &GlobalSfmConfig,
    pristine_workspace: SfmWorkspace,
    events: &mut EventEmitter<'_>,
) -> Result<(SfmWorkspace, GlobalSfmReport)> {
    let GlobalCandidateCascade {
        rotation_preflight,
        attempts,
        accepted,
        no_reconstruction_reason,
    } = cascade;

    match accepted {
        Some(GlobalAcceptedResult {
            selected_attempt,
            workspace,
            registered_view_count,
            input_view_count,
            track_validation,
        }) => {
            emit_selected_candidate(events, &attempts, selected_attempt, &workspace);
            build_adaptive_result(
                workspace,
                config,
                rotation_preflight,
                attempts,
                GlobalCompletionReport::Reconstructed {
                    selected_attempt,
                    registered_view_count,
                    input_view_count,
                    track_validation,
                },
            )
        }
        None => {
            let input_view_count = pristine_workspace.views().len();
            build_adaptive_result(
                pristine_workspace,
                config,
                rotation_preflight,
                attempts,
                GlobalCompletionReport::NoReconstruction {
                    reason: no_reconstruction_reason,
                    input_view_count,
                },
            )
        }
    }
}

fn emit_selected_candidate(
    events: &mut EventEmitter<'_>,
    attempts: &[GlobalCandidateAttemptReport],
    selected_attempt: usize,
    workspace: &SfmWorkspace,
) {
    let Some(attempt) = attempts.get(selected_attempt) else {
        return;
    };
    events.emit(
        SfmEventBranch::GlobalCandidate {
            attempt: selected_attempt,
            candidate: attempt.candidate,
        },
        SfmEventPhase::GlobalCandidateAdmission,
        SfmEventTransition::Selected,
        workspace,
    );
}

fn build_adaptive_result(
    completed: SfmWorkspace,
    config: &GlobalSfmConfig,
    rotation_preflight: crate::ops::averaging::RotationAveragingConsistencyReport,
    attempts: Vec<GlobalCandidateAttemptReport>,
    completion: GlobalCompletionReport,
) -> Result<(SfmWorkspace, GlobalSfmReport)> {
    completed.validate()?;
    let report = GlobalSfmReport::try_new(
        config.clone(),
        Some(rotation_preflight),
        attempts,
        completion,
    )?;
    Ok((completed, report))
}

fn validate_uninitialized_prepared_workspace(workspace: &SfmWorkspace) -> Result<()> {
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "precomputed tracks",
            method: "build_tracks",
        });
    }
    if workspace.geometric_model_store().populated_count() == 0 {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "verified geometric models",
            method: "verify_pairs",
        });
    }
    if workspace.registered_view_count() != 0 || workspace.landmarks().active_count() != 0 {
        return Err(SfmError::InvalidInput(
            "global SfM requires a prepared workspace without poses or landmarks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_prepared_domains(
    pristine: &SfmWorkspace,
    global_prepared: &SfmWorkspace,
) -> Result<()> {
    let same_models = pristine.geometric_model_store().populated_count()
        == global_prepared.geometric_model_store().populated_count()
        && (0..pristine.pair_graph().len()).all(|index| {
            pristine.geometric_model_store().get(index)
                == global_prepared.geometric_model_store().get(index)
        });
    let same_matches = pristine.is_matching_data_released()
        == global_prepared.is_matching_data_released()
        && pristine.match_store().populated_count()
            == global_prepared.match_store().populated_count()
        && (0..pristine.pair_graph().len()).all(|index| {
            pristine.match_store().get(index) == global_prepared.match_store().get(index)
        });
    if pristine.root_seed() != global_prepared.root_seed()
        || pristine.views() != global_prepared.views()
        || pristine.camera_models() != global_prepared.camera_models()
        || pristine.pair_graph().pairs() != global_prepared.pair_graph().pairs()
        || pristine.tracks() != global_prepared.tracks()
        || !same_matches
        || !same_models
    {
        return Err(SfmError::InvalidInput(
            "global candidate workspaces must share identical prepared input domains".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_domain_candidates_precede_their_own_core_recovery() {
        let direction = GlobalCandidate::TranslationAveraging;
        let mixed = GlobalCandidate::GlobalPositioning;
        let recovery = [
            (
                mixed,
                GlobalCandidateRoute::FullDomainIncidenceCoreThreeView,
            ),
            (direction, GlobalCandidateRoute::RigidIncidenceCoreTwoView),
        ];

        assert_eq!(
            candidate_attempt_order([mixed, direction]),
            [
                (mixed, GlobalCandidateRoute::FullDomainThreeView),
                (direction, GlobalCandidateRoute::FullDomainThreeView),
                (mixed, GlobalCandidateRoute::FullDomainTwoView),
                (direction, GlobalCandidateRoute::FullDomainTwoView),
                recovery[0],
                recovery[1],
            ]
        );
        // Each recovery route follows its candidate, whatever the escalation order.
        let swapped = candidate_attempt_order([direction, mixed]);
        assert_eq!(swapped[4], recovery[1]);
        assert_eq!(swapped[5], recovery[0]);
    }
}
