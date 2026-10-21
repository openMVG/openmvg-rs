// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One transactional positioning-candidate attempt: solve translations, run the global finish,
//! and record the admission decision.

use crate::ba::backend::{BaFailureKind, BaPreflightDecision, BaPreflightDeferralReason};
use crate::events::{EventEmitter, SfmEventBranch, SfmEventPhase, SfmEventTransition};
use crate::pipelines::global::GlobalTranslationReport;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::{Result, SfmError, SfmWorkspace};

use super::{
    GlobalRuntime,
    config::GlobalSfmConfig,
    finish::{
        finish_global_ba_retriangulate_prepared_with_events, prepare_fixed_rotation_core_structure,
        prepare_fixed_rotation_structure, run_self_calibration,
    },
    policy::{
        CandidateAdmissionEvidence, FixedRotationObservabilityEvidence, GlobalCandidate,
        MetricCoreCertificationEvidence, PostFitTrustEvidence,
    },
    report::{
        GlobalCandidateAttemptReport, GlobalCandidateDeferral, GlobalCandidateRoute,
        GlobalCandidateStage, GlobalCandidateStatus, GlobalNoReconstructionReason,
    },
    translation::{solve_direction_translation, solve_mixed_translation},
};

pub(super) enum CandidateRun {
    Accepted {
        report: Box<GlobalCandidateAttemptReport>,
        workspace: Box<SfmWorkspace>,
    },
    NotAccepted {
        report: Box<GlobalCandidateAttemptReport>,
        abort_global: Option<GlobalNoReconstructionReason>,
    },
}

#[derive(Clone, Copy)]
pub(super) struct CandidateAttempt {
    pub(super) candidate: GlobalCandidate,
    pub(super) route: GlobalCandidateRoute,
    pub(super) branch: SfmEventBranch,
}

impl GlobalCandidateRoute {
    const fn min_track_observations(self) -> usize {
        match self {
            Self::FullDomainThreeView | Self::FullDomainIncidenceCoreThreeView => 3,
            Self::FullDomainTwoView | Self::RigidIncidenceCoreTwoView => 2,
        }
    }

    const fn translation_scope(self) -> crate::config::GlobalTranslationScope {
        match self {
            Self::RigidIncidenceCoreTwoView => {
                crate::config::GlobalTranslationScope::LargestRigidBiconnectedCore
            }
            Self::FullDomainThreeView
            | Self::FullDomainTwoView
            | Self::FullDomainIncidenceCoreThreeView => {
                crate::config::GlobalTranslationScope::AllActiveEdges
            }
        }
    }

    const fn uses_bundle_incidence_core(self) -> bool {
        matches!(
            self,
            Self::RigidIncidenceCoreTwoView | Self::FullDomainIncidenceCoreThreeView
        )
    }
}

struct CandidateAttemptContext {
    candidate: GlobalCandidate,
    route: GlobalCandidateRoute,
    translation: GlobalTranslationReport,
}

impl CandidateAttemptContext {
    fn into_report(
        self,
        make_outcome: impl FnOnce(Box<GlobalTranslationReport>) -> GlobalCandidateStatus,
    ) -> GlobalCandidateAttemptReport {
        GlobalCandidateAttemptReport {
            candidate: self.candidate,
            route: self.route,
            post_admission_failure: None,
            outcome: make_outcome(Box::new(self.translation)),
        }
    }

    fn failed(self, stage: GlobalCandidateStage, error: SfmError) -> CandidateRun {
        CandidateRun::NotAccepted {
            report: Box::new(
                self.into_report(|translation| GlobalCandidateStatus::Failed {
                    stage,
                    error: error.to_string(),
                    translation: Some(translation),
                }),
            ),
            abort_global: None,
        }
    }

    fn resource_deferred(self, failure: crate::ba::backend::BaBackendFailure) -> CandidateRun {
        let candidate = self.candidate;
        let no_core_failure = failure.clone();
        CandidateRun::NotAccepted {
            report: Box::new(
                self.into_report(|translation| GlobalCandidateStatus::Deferred {
                    reason: GlobalCandidateDeferral::ResourceCapacity { failure },
                    translation: Some(translation),
                }),
            ),
            abort_global: Some(GlobalNoReconstructionReason::ResourceCapacity {
                candidate,
                failure: no_core_failure,
            }),
        }
    }

    fn preflight_deferred(
        self,
        preflight: crate::ba::backend::BaPreflightReport,
        abort: bool,
    ) -> CandidateRun {
        let candidate = self.candidate;
        let abort_reason = abort.then(|| GlobalNoReconstructionReason::FixedRotationPreflight {
            candidate,
            report: Box::new(preflight.clone()),
        });
        CandidateRun::NotAccepted {
            report: Box::new(
                self.into_report(|translation| GlobalCandidateStatus::Deferred {
                    reason: GlobalCandidateDeferral::FixedRotationPreflight {
                        report: Box::new(preflight),
                    },
                    translation: Some(translation),
                }),
            ),
            abort_global: abort_reason,
        }
    }
}

pub(super) fn run_candidate(
    candidate_workspace: SfmWorkspace,
    config: &GlobalSfmConfig,
    attempt: CandidateAttempt,
    rotation_preparation: &crate::ops::averaging::GlobalRotationPreparation,
    runtime: GlobalRuntime<'_>,
    events: &mut EventEmitter<'_>,
) -> CandidateRun {
    let CandidateAttempt {
        candidate,
        route,
        branch,
    } = attempt;
    let mut candidate_workspace = candidate_workspace;
    let averaging = config.averaging();
    let translation = solve_candidate_translation(
        &mut candidate_workspace,
        &averaging,
        &config.global_positioning,
        candidate,
        route.translation_scope(),
        rotation_preparation,
        runtime,
        events.progress(),
    );
    let translation =
        translation.and_then(|report| cancellation_point(events.progress()).map(|()| report));
    let mut translation = match translation {
        Ok(report) => report,
        Err(error) => {
            discard_candidate(events, branch, &candidate_workspace);
            return failed_candidate(
                candidate,
                route,
                GlobalCandidateStage::Translation,
                error,
                None,
            );
        }
    };
    if route.min_track_observations() == 2 && translation.triangulation.min_observations > 2 {
        let track_count = candidate_workspace.tracks().len();
        events
            .progress()
            .begin(SfmProgressStage::Triangulation, track_count);
        translation.triangulation =
            match crate::ops::reconstruction::triangulate_with_min_observations(
                &mut candidate_workspace,
                runtime.blind_triangulator,
                averaging.gauge_scale_policy(),
                2,
            ) {
                Ok(report) => report,
                Err(error) => {
                    discard_candidate(events, branch, &candidate_workspace);
                    return failed_candidate(
                        candidate,
                        route,
                        GlobalCandidateStage::Translation,
                        error,
                        Some(translation),
                    );
                }
            };
        events
            .progress()
            .advance(SfmProgressStage::Triangulation, track_count);
        events.progress().finish(SfmProgressStage::Triangulation);
        translation.final_active_landmarks = candidate_workspace.landmarks().active_count();
    }
    events.emit(
        branch,
        SfmEventPhase::GlobalTranslationAveraging,
        SfmEventTransition::Checkpoint,
        &candidate_workspace,
    );

    let mut context = CandidateAttemptContext {
        candidate,
        route,
        translation,
    };
    let preparation = if route.uses_bundle_incidence_core() {
        prepare_fixed_rotation_core_structure(&mut candidate_workspace)
    } else {
        prepare_fixed_rotation_structure(&mut candidate_workspace)
    };
    let preparation = match preparation {
        Ok(report) => report,
        Err(error) => {
            discard_candidate(events, branch, &candidate_workspace);
            return context.failed(GlobalCandidateStage::FixedRotationPreflight, error);
        }
    };
    events.emit(
        branch,
        SfmEventPhase::GlobalTranslationCore,
        SfmEventTransition::Checkpoint,
        &candidate_workspace,
    );
    let (problem, fixed_rotation_config) =
        match fixed_rotation_preflight_input(&candidate_workspace) {
            Ok(input) => input,
            Err(error) => {
                discard_candidate(events, branch, &candidate_workspace);
                return context.failed(GlobalCandidateStage::FixedRotationPreflight, error);
            }
        };
    let preflight = match runtime
        .bundle_adjustment
        .preflight(&problem, &fixed_rotation_config)
    {
        Ok(report) => report,
        Err(failure) if failure.kind == BaFailureKind::ResourceExhausted => {
            discard_candidate(events, branch, &candidate_workspace);
            return context.resource_deferred(failure);
        }
        Err(failure) => {
            discard_candidate(events, branch, &candidate_workspace);
            return context.failed(
                GlobalCandidateStage::FixedRotationPreflight,
                SfmError::bundle_adjustment_backend_failed(failure),
            );
        }
    };
    if !preflight.should_proceed() {
        let abort = matches!(
            &preflight.decision,
            BaPreflightDecision::Deferred(BaPreflightDeferralReason::WorkspaceLimit)
                | BaPreflightDecision::NotApplicable(_)
        );
        discard_candidate(events, branch, &candidate_workspace);
        return context.preflight_deferred(preflight, abort);
    }

    let finish_result = finish_global_ba_retriangulate_prepared_with_events(
        &mut candidate_workspace,
        runtime.bundle_adjustment,
        runtime.robust_triangulator,
        config.max_reprojection_error().value(),
        config.max_robust_rounds(),
        config.ba_iteration_limits,
        preparation,
        Some(&config.post_fixed_rotation_cleanup),
        branch,
        events,
    );
    let mut finish = match finish_result {
        Ok(Some(finish)) => finish,
        Ok(None) => {
            discard_candidate(events, branch, &candidate_workspace);
            return context.failed(
                GlobalCandidateStage::Finish,
                SfmError::InvalidInput(
                    "global refinement requires at least two registered cameras".to_owned(),
                ),
            );
        }
        Err(error) => {
            discard_candidate(events, branch, &candidate_workspace);
            return match error.into_resource_exhausted_failure() {
                Ok(failure) => context.resource_deferred(failure),
                Err(error) => context.failed(GlobalCandidateStage::Finish, error),
            };
        }
    };
    context.translation.final_active_landmarks = candidate_workspace.landmarks().active_count();

    let fixed_rotation_observability = finish
        .translation_and_points
        .ba
        .diagnostics
        .observability
        .as_ref()
        .map(FixedRotationObservabilityEvidence::from);
    let all_verified_pose_graph = crate::ops::averaging::verified_pose_graph_consistency(
        &candidate_workspace,
        crate::ops::averaging::VerifiedPairSelection::All,
    );
    let all_verified_inconsistent_camera_count = all_verified_pose_graph
        .cameras
        .iter()
        .filter(|camera| {
            camera.median_rotation_residual_deg.is_some_and(|residual| {
                residual
                    > config
                        .policy
                        .post_fit_trust()
                        .max_all_verified_camera_rotation_median_deg()
            }) || camera
                .median_translation_residual_deg
                .is_some_and(|residual| {
                    residual
                        > config
                            .policy
                            .post_fit_trust()
                            .max_all_verified_camera_translation_median_deg()
                })
        })
        .count();
    let evidence = CandidateAdmissionEvidence {
        rotation_consensus: config.policy.rotation_consensus().evidence(
            all_verified_pose_graph
                .cameras
                .iter()
                .map(|camera| camera.median_rotation_residual_deg),
        ),
        metric_core: MetricCoreCertificationEvidence {
            fixed_rotation_observability,
        },
        post_fit_trust: PostFitTrustEvidence {
            finish_accepted: finish.accepted,
            translation_graph_p90_deg: finish.translation_graph_p90_after_deg,
            final_reprojection_rmse_px: finish.final_reprojection_rmse_px,
            rotation_graph_p90_deg: finish.rotation_graph_p90_after_deg,
            all_verified_camera_count: all_verified_pose_graph.cameras.len(),
            all_verified_inconsistent_camera_count,
        },
    };
    let admission = config.policy.admit(evidence);
    let accepted = admission.is_accepted();
    if accepted && config.self_calibrate_intrinsics() {
        // Admission checked the core against pairwise geometry verified under the input
        // intrinsics; those edges cannot judge recalibrated poses, so only the admitted candidate
        // is refined.
        if let Err(error) = cancellation_point(events.progress()) {
            discard_candidate(events, branch, &candidate_workspace);
            return context.failed(GlobalCandidateStage::Finish, error);
        }
        events
            .progress()
            .begin(SfmProgressStage::SelfCalibration, 1);
        let self_calibration = run_self_calibration(
            &mut candidate_workspace,
            runtime.bundle_adjustment,
            runtime.robust_triangulator,
            config.max_reprojection_error().value(),
            config.ba_iteration_limits,
            branch,
            events,
        );
        match self_calibration {
            Ok(outcome) => {
                events
                    .progress()
                    .advance(SfmProgressStage::SelfCalibration, 1);
                events.progress().finish(SfmProgressStage::SelfCalibration);
                finish.self_calibration = Some(outcome);
                context.translation.final_active_landmarks =
                    candidate_workspace.landmarks().active_count();
            }
            Err(error) => {
                discard_candidate(events, branch, &candidate_workspace);
                return context.failed(GlobalCandidateStage::Finish, error);
            }
        }
    }
    if accepted
        && config.release_matching_data()
        && let Err(error) = candidate_workspace.release_matching_data()
    {
        discard_candidate(events, branch, &candidate_workspace);
        return context.failed(GlobalCandidateStage::Finish, error);
    }
    events.emit(
        branch,
        SfmEventPhase::GlobalCandidateAdmission,
        if accepted {
            SfmEventTransition::Committed
        } else {
            SfmEventTransition::Discarded
        },
        &candidate_workspace,
    );
    let report = if accepted {
        context.into_report(|translation| GlobalCandidateStatus::Accepted {
            evidence,
            all_verified_pose_graph: Some(Box::new(all_verified_pose_graph)),
            preflight: Box::new(preflight),
            translation,
            finish: Box::new(finish),
        })
    } else {
        context.into_report(|translation| GlobalCandidateStatus::Rejected {
            evidence,
            admission,
            all_verified_pose_graph: Some(Box::new(all_verified_pose_graph)),
            preflight: Box::new(preflight),
            translation,
            finish: Box::new(finish),
        })
    };
    if accepted {
        CandidateRun::Accepted {
            report: Box::new(report),
            workspace: Box::new(candidate_workspace),
        }
    } else {
        CandidateRun::NotAccepted {
            report: Box::new(report),
            abort_global: None,
        }
    }
}

/// Apply the camera residual policy before constructing the immutable preflight problem.
fn fixed_rotation_preflight_input(
    workspace: &SfmWorkspace,
) -> Result<(
    openmvg_ba::BundleProblem,
    crate::ba::config::BundleAdjustConfig,
)> {
    let config = crate::ops::reconstruction::pipeline_ba_config_for_workspace(
        workspace,
        super::finish::fixed_rotation_bundle_config(),
    );
    let problem = crate::ops::reconstruction::bundle_adjustment_problem(workspace, &config)?;
    Ok((problem, config))
}

fn discard_candidate(
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

#[allow(clippy::too_many_arguments)]
fn solve_candidate_translation(
    workspace: &mut SfmWorkspace,
    averaging: &crate::config::GlobalAveragingConfig,
    global_positioning: &super::translation::PretrackedGlobalPositioningConfig,
    candidate: GlobalCandidate,
    translation_scope: crate::config::GlobalTranslationScope,
    rotation_preparation: &crate::ops::averaging::GlobalRotationPreparation,
    runtime: GlobalRuntime<'_>,
    progress: &dyn SfmProgressSink,
) -> Result<GlobalTranslationReport> {
    match candidate {
        GlobalCandidate::TranslationAveraging => solve_direction_translation(
            workspace,
            averaging,
            translation_scope,
            rotation_preparation,
            runtime.direction_translation,
            runtime.blind_triangulator,
            progress,
        ),
        GlobalCandidate::GlobalPositioning => solve_mixed_translation(
            workspace,
            averaging,
            translation_scope,
            global_positioning,
            rotation_preparation,
            runtime.blind_triangulator,
            progress,
        ),
    }
}

fn failed_candidate(
    candidate: GlobalCandidate,
    route: GlobalCandidateRoute,
    stage: GlobalCandidateStage,
    error: SfmError,
    translation: Option<GlobalTranslationReport>,
) -> CandidateRun {
    CandidateRun::NotAccepted {
        report: Box::new(GlobalCandidateAttemptReport {
            candidate,
            route,
            post_admission_failure: None,
            outcome: GlobalCandidateStatus::Failed {
                stage,
                error: error.to_string(),
                translation: translation.map(Box::new),
            },
        }),
        abort_global: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageMeta, ViewRecord};
    use crate::{
        Camera, IntrinsicId, PoseId, Reconstruction, ReconstructionParts, SfmEngine, ViewId,
        WorldToCameraPose,
    };
    use nalgebra::{Matrix3, Point3, Vector3};
    use openmvg_ba::{ResidualSpace, RobustLoss};
    use openmvg_core::camera::{CameraModel, SphericalCamera};

    fn posed_workspace(camera: Camera) -> SfmWorkspace {
        let views = (0..2)
            .map(|view| {
                ViewRecord::new(
                    ViewId::new(view),
                    ImageMeta::external_id(format!("view-{view}"), 640, 320),
                    IntrinsicId::new(0),
                    PoseId::new(view),
                )
            })
            .collect();
        let observed = vec![
            (
                ViewId::new(0),
                camera.project(&Vector3::new(0.0, 0.0, 5.0)).unwrap(),
            ),
            (
                ViewId::new(1),
                camera.project(&Vector3::new(-1.0, 0.0, 5.0)).unwrap(),
            ),
        ];
        Reconstruction::try_from_parts(
            ReconstructionParts::new(7_u64, SfmEngine::Global, vec![camera], views)
                .with_pose(PoseId::new(0), WorldToCameraPose::identity())
                .with_pose(
                    PoseId::new(1),
                    WorldToCameraPose::from_matrix(Matrix3::identity(), Point3::new(1.0, 0.0, 0.0))
                        .unwrap(),
                )
                .with_point(Point3::new(0.0, 0.0, 5.0), observed),
        )
        .unwrap()
        .workspace
    }

    #[test]
    fn preflight_problem_and_config_share_the_camera_residual_domain() {
        let spherical = posed_workspace(Camera::Spherical(SphericalCamera::try_new(640).unwrap()));
        let (problem, config) = fixed_rotation_preflight_input(&spherical).unwrap();
        assert_eq!(problem.residual_space(), ResidualSpace::Angular);
        // The configured pixel scale is converted to radians when the problem is built.
        assert!(matches!(config.robust_loss(), RobustLoss::Huber(_)));
        assert_eq!(
            config.robust_loss().scale().unwrap().space(),
            ResidualSpace::Pixels
        );
        assert!(matches!(problem.robust_loss(), RobustLoss::Huber(_)));
        assert_eq!(
            problem.robust_loss().scale().unwrap().space(),
            ResidualSpace::Angular
        );

        let pinhole = posed_workspace(Camera::try_pinhole(500.0, 500.0, 320.0, 160.0).unwrap());
        let (problem, config) = fixed_rotation_preflight_input(&pinhole).unwrap();
        assert_eq!(problem.residual_space(), ResidualSpace::Pixels);
        assert_eq!(problem.robust_loss(), config.robust_loss());
        assert_eq!(
            problem.robust_loss().scale().unwrap().space(),
            ResidualSpace::Pixels
        );
    }
}
