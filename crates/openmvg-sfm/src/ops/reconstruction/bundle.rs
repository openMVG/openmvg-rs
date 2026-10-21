// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bundle-adjustment orchestration, validation, and problem construction.

use super::residual::is_generalized_camera;
use super::*;
use crate::ba::backend::{
    BaCommitRejection, BaConvergenceReason, BaDiagnostics, BaObservabilityDiagnostics,
    BaRankAssessment, BaSolveStatus,
};
use crate::model::PoseRecord;
use crate::two_view::ResidualUnit;
use openmvg_ba::{
    BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem, Gauge,
    IntrinsicCertification, IntrinsicVariables, PoseVariables, ResidualScale, ResidualSpace,
    RobustLoss, Solution,
};
use openmvg_core::model::CameraIntrinsics;
/// Robust BA + scene clean with a typed receipt for every executed adjustment.
#[cfg_attr(not(feature = "unstable-test-support"), allow(dead_code))]
pub fn bundle_adjust_and_clean(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    config: &BundleAdjustConfig,
    max_pixel_residual: f64,
    min_angle_deg: f64,
    max_rounds: usize,
) -> Result<BundleAdjustAndCleanReport> {
    bundle_adjust_and_clean_report_with_policy(
        workspace,
        ba,
        config,
        max_pixel_residual,
        min_angle_deg,
        max_rounds,
        FinalPolishPolicy::Required,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bundle_adjust_and_clean_report_skipping_negligible_verification(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    config: &BundleAdjustConfig,
    max_pixel_residual: f64,
    min_angle_deg: f64,
    max_rounds: usize,
    minimum_rounds: usize,
    maximum_removed_fraction: f64,
) -> Result<BundleAdjustAndCleanReport> {
    bundle_adjust_and_clean_report_with_policy(
        workspace,
        ba,
        config,
        max_pixel_residual,
        min_angle_deg,
        max_rounds,
        FinalPolishPolicy::SkipAfterNegligibleRemoval {
            minimum_rounds,
            maximum_removed_fraction,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn bundle_adjust_and_clean_report_with_policy(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    config: &BundleAdjustConfig,
    max_pixel_residual: f64,
    min_angle_deg: f64,
    max_rounds: usize,
    final_polish_policy: FinalPolishPolicy,
) -> Result<BundleAdjustAndCleanReport> {
    config
        .validate()
        .map_err(|error| SfmError::InvalidConfig(error.to_string()))?;
    super::cleanup::validate_landmark_cleanup_config(max_pixel_residual, min_angle_deg)?;
    if let FinalPolishPolicy::SkipAfterNegligibleRemoval {
        maximum_removed_fraction,
        ..
    } = final_polish_policy
        && (!maximum_removed_fraction.is_finite()
            || !(0.0..=1.0).contains(&maximum_removed_fraction))
    {
        return Err(SfmError::InvalidConfig(
            "maximum removed fraction must be finite and in [0, 1]".to_owned(),
        ));
    }
    let mut total_support_change = LandmarkSupportCleanupReport::default();
    let mut rounds = Vec::new();
    for round in 0..max_rounds.max(1) {
        let active_landmarks_before = workspace.landmarks().active_count();
        if active_landmarks_before == 0 && !rounds.is_empty() {
            return Ok(BundleAdjustAndCleanReport {
                schema_version: 1,
                rounds,
                final_bundle_adjustment: None,
                termination: BundleAdjustAndCleanTermination::Stable,
                total_removed_landmarks: total_support_change.rejected_landmarks,
                total_removed_observations: total_support_change.removed_observations,
            });
        }
        let bundle_adjustment = workspace.bundle_adjust(ba, config)?;
        let ba_support_change = LandmarkSupportCleanupReport {
            rejected_landmarks: bundle_adjustment.rejected_landmarks,
            removed_observations: bundle_adjustment.removed_observations,
        };
        let observation_count = bundle_adjustment.ba.observation_count;
        total_support_change.merge(ba_support_change);
        let cleanup = clean_outlier_landmark_support(workspace, max_pixel_residual, min_angle_deg)?;
        total_support_change.merge(cleanup);
        rounds.push(BundleAdjustAndCleanRoundReport {
            bundle_adjustment,
            removed_after_adjustment: cleanup.rejected_landmarks,
            removed_observations_after_adjustment: cleanup.removed_observations,
        });
        if ba_support_change.is_empty() && cleanup.is_empty() {
            return Ok(BundleAdjustAndCleanReport {
                schema_version: 1,
                rounds,
                final_bundle_adjustment: None,
                termination: BundleAdjustAndCleanTermination::Stable,
                total_removed_landmarks: total_support_change.rejected_landmarks,
                total_removed_observations: total_support_change.removed_observations,
            });
        }
        if let FinalPolishPolicy::SkipAfterNegligibleRemoval {
            minimum_rounds,
            maximum_removed_fraction,
        } = final_polish_policy
        {
            let mut round_support_change = ba_support_change;
            round_support_change.merge(cleanup);
            if round + 1 >= minimum_rounds
                && cleanup_removal_is_negligible(
                    round_support_change.rejected_landmarks,
                    active_landmarks_before,
                    maximum_removed_fraction,
                )
                && cleanup_removal_is_negligible(
                    round_support_change.removed_observations,
                    observation_count,
                    maximum_removed_fraction,
                )
            {
                return Ok(BundleAdjustAndCleanReport {
                    schema_version: 1,
                    rounds,
                    final_bundle_adjustment: None,
                    termination: BundleAdjustAndCleanTermination::NegligibleRemoval,
                    total_removed_landmarks: total_support_change.rejected_landmarks,
                    total_removed_observations: total_support_change.removed_observations,
                });
            }
        }
    }
    // Rounds exhausted while still removing: a final BA on the cleaned scene.
    let final_bundle_adjustment = workspace.bundle_adjust(ba, config)?;
    total_support_change.merge(LandmarkSupportCleanupReport {
        rejected_landmarks: final_bundle_adjustment.rejected_landmarks,
        removed_observations: final_bundle_adjustment.removed_observations,
    });
    Ok(BundleAdjustAndCleanReport {
        schema_version: 1,
        rounds,
        final_bundle_adjustment: Some(final_bundle_adjustment),
        termination: BundleAdjustAndCleanTermination::FinalPolishCompleted,
        total_removed_landmarks: total_support_change.rejected_landmarks,
        total_removed_observations: total_support_change.removed_observations,
    })
}

/// Rebuild component 0 from the currently posed images and active landmarks, keeping
/// the existing gauge policy (defaults to `Free`). This is the single component-rebuild
/// seam; registration and (re)triangulation both call it, so the image set is always
/// derived from `pose_store` and the landmark set from `iter_active()`.
pub fn bundle_adjust(
    workspace: &mut SfmWorkspace,
    ba: &dyn BundleAdjustmentBackend,
    cfg: &BundleAdjustConfig,
) -> Result<BundleReport> {
    if workspace.landmarks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "landmarks",
            method: "triangulate",
        });
    }

    let effective_cfg = config_with_complete_shared_pose_fixes(workspace, cfg)?;
    let cfg = &effective_cfg;
    let prepared =
        prepare_bundle_adjustment_problem(workspace, cfg, &std::collections::BTreeSet::new())?;
    let solution = propose_bundle_adjustment(ba, &prepared.problem, cfg)?;
    certify_bundle_adjustment(ba, &prepared.problem, cfg, &solution)?;
    // Landmarks the adjuster held out of the system it solved and certified are nuisance
    // parameters: the cameras stand without them, and their positions mean nothing.
    let unobservable: std::collections::BTreeSet<_> =
        if solution.report().diagnostics().quarantined_point_count() > 0 {
            solution
                .weak_points(&prepared.problem)
                .into_iter()
                .map(|point| prepared.point_ids[point])
                .collect()
        } else {
            std::collections::BTreeSet::new()
        };
    let mut report = bundle_adjustment_report(ba.name(), &prepared.problem, &solution)?;
    report.diagnostics.unconstrained_intrinsics = solution
        .report()
        .unconstrained_intrinsics()
        .iter()
        .filter_map(|&group| prepared.intrinsic_ids.get(group).copied())
        .collect();

    // Treat the backend as a numerical boundary: validate the complete proposal before
    // committing any part of it. A divergent BA used to write non-finite cameras first,
    // reject the resulting landmarks later, and leave a poisoned workspace that failed
    // only when exported or evaluated.
    for (pose_id, pose) in prepared.pose_ids.iter().zip(solution.poses()) {
        if workspace.pose_by_id(*pose_id).is_none() {
            return Err(SfmError::InvalidId {
                kind: crate::PoseId::kind(),
                index: pose_id.index(),
            });
        }
        pose.validate().map_err(|error| {
            SfmError::InvalidInput(format!(
                "bundle adjustment returned invalid pose {pose_id}: {error}"
            ))
        })?;
    }
    for (intrinsic_id, intrinsic) in prepared.intrinsic_ids.iter().zip(solution.intrinsics()) {
        if workspace.camera_model(*intrinsic_id).is_none() {
            return Err(SfmError::InvalidId {
                kind: IntrinsicId::kind(),
                index: intrinsic_id.index(),
            });
        }
        intrinsic.validate_camera_domain().map_err(|error| {
            SfmError::InvalidInput(format!(
                "bundle adjustment returned invalid intrinsic {}: {error}",
                intrinsic_id
            ))
        })?;
    }
    for (landmark_id, point) in prepared.point_ids.iter().zip(solution.points()) {
        if workspace.landmarks().get(*landmark_id).is_none() {
            return Err(SfmError::InvalidId {
                kind: LandmarkId::kind(),
                index: landmark_id.index(),
            });
        }
        if !point.coords.iter().all(|value| value.is_finite()) {
            return Err(SfmError::InvalidInput(format!(
                "bundle adjustment returned non-finite landmark {landmark_id}"
            )));
        }
    }
    workspace.replace_camera_model_groups(
        prepared
            .intrinsic_ids
            .iter()
            .zip(solution.intrinsics())
            .map(|(id, intrinsic)| (*id, intrinsic.camera().clone()))
            .collect(),
    )?;

    for (pose_id, pose) in prepared.pose_ids.iter().zip(solution.poses()) {
        let component = workspace
            .pose_by_id(*pose_id)
            .and_then(|record| record.component);
        workspace.set_pose(PoseRecord {
            id: *pose_id,
            pose: pose.clone(),
            component,
        })?;
    }
    for (landmark_id, point) in prepared.point_ids.iter().zip(solution.points()) {
        let landmark = workspace
            .landmarks
            .as_mut_slice()
            .get_mut(landmark_id.index())
            .ok_or(SfmError::InvariantViolation(format!(
                "bundle-adjustment landmark {landmark_id} disappeared before commit"
            )))?;
        landmark.point = *point;
    }
    for landmark_id in &unobservable {
        if let Some(landmark) = workspace
            .landmarks
            .as_mut_slice()
            .get_mut(landmark_id.index())
        {
            landmark.status = LandmarkStatus::Rejected;
        }
    }
    let refresh = refresh_landmark_reprojection_errors(workspace)?;
    if refresh.rejected_landmarks > 0 || !unobservable.is_empty() {
        refresh_primary_component(workspace);
    }

    Ok(BundleReport {
        adjusted_cameras: solution.poses().len(),
        adjusted_landmarks: solution.points().len(),
        rejected_landmarks: refresh.rejected_landmarks + unobservable.len(),
        removed_observations: refresh.removed_observations,
        ba: report,
    })
}

pub(super) fn config_with_complete_shared_pose_fixes(
    workspace: &SfmWorkspace,
    config: &BundleAdjustConfig,
) -> Result<BundleAdjustConfig> {
    let fixed_pose_ids = config
        .fixed_views()
        .iter()
        .map(|view| {
            workspace.pose_id(*view).ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: view.index(),
            })
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let mut fixed_views = config.fixed_views().clone();
    fixed_views.extend(
        workspace
            .views()
            .iter()
            .filter(|view| fixed_pose_ids.contains(&view.pose_id))
            .map(|view| view.id),
    );
    Ok(config.clone().with_fixed_views(fixed_views))
}

fn propose_bundle_adjustment(
    ba: &dyn BundleAdjustmentBackend,
    problem: &BundleProblem,
    cfg: &BundleAdjustConfig,
) -> Result<Solution> {
    match ba
        .adjust(problem, cfg)
        .map_err(SfmError::bundle_adjustment_backend_failed)?
    {
        BaBackendStatus::Proposed(solution) => Ok(*solution),
        BaBackendStatus::NotApplicable(reason) => {
            Err(SfmError::bundle_adjustment_not_applicable(reason))
        }
    }
}

fn certify_bundle_adjustment(
    ba: &dyn BundleAdjustmentBackend,
    problem: &BundleProblem,
    cfg: &BundleAdjustConfig,
    solution: &Solution,
) -> Result<()> {
    let solve_config = crate::ba::native::audit_solve_config(cfg, ba.name())
        .map_err(SfmError::bundle_adjustment_backend_failed)?;
    if let Err(error) = solution.audit_for_commit(problem, &solve_config) {
        return Err(SfmError::BundleAdjustmentProposalRejected {
            backend_name: ba.name().to_owned(),
            reason: crate::BundleAdjustmentFailure::proposal_rejected(audit_rejection(error)),
        });
    }
    Ok(())
}

fn audit_rejection(error: openmvg_ba::AuditError) -> BaCommitRejection {
    match error {
        openmvg_ba::AuditError::ObjectiveIncreased => BaCommitRejection::CostIncrease,
        openmvg_ba::AuditError::RankDeficient => BaCommitRejection::RankDeficient,
        openmvg_ba::AuditError::WeakPoints { count } => BaCommitRejection::WeakPoints { count },
        openmvg_ba::AuditError::NoAcceptedProgress => BaCommitRejection::NoAcceptedProgress,
        openmvg_ba::AuditError::ObservabilityNotAssessed => {
            BaCommitRejection::ObservabilityNotAssessed
        }
        openmvg_ba::AuditError::GaugeMismatch => BaCommitRejection::GaugeMismatch,
        openmvg_ba::AuditError::StationarityNotCertified => {
            BaCommitRejection::StationarityNotCertified
        }
        openmvg_ba::AuditError::Status(status) => {
            BaCommitRejection::Status(canonical_solve_status(status))
        }
        _ => BaCommitRejection::Status(BaSolveStatus::Unknown),
    }
}

fn canonical_solve_status(status: openmvg_ba::SolveStatus) -> BaSolveStatus {
    match status {
        openmvg_ba::SolveStatus::Converged(reason) => BaSolveStatus::Converged(match reason {
            openmvg_ba::ConvergenceReason::Objective => BaConvergenceReason::Objective,
            openmvg_ba::ConvergenceReason::Gradient => BaConvergenceReason::Gradient,
            openmvg_ba::ConvergenceReason::Step => BaConvergenceReason::Step,
            _ => BaConvergenceReason::Unspecified,
        }),
        openmvg_ba::SolveStatus::MaxIterations => BaSolveStatus::MaxIterations,
        openmvg_ba::SolveStatus::Stalled => BaSolveStatus::Stalled,
        openmvg_ba::SolveStatus::Degenerate => BaSolveStatus::Degenerate,
        openmvg_ba::SolveStatus::Skipped => BaSolveStatus::Skipped,
        _ => BaSolveStatus::Unknown,
    }
}

/// Convert the BA problem and solution into the serialized SfM run receipt.
///
/// # Errors
///
/// Returns [`SfmError::InvariantViolation`] when the solution reports a residual space this
/// receipt has no unit for.
pub fn bundle_adjustment_report(
    backend_name: &str,
    problem: &BundleProblem,
    solution: &Solution,
) -> Result<BundleAdjustmentReport> {
    let residual_units = match solution.residual_space() {
        ResidualSpace::Pixels => ResidualUnit::Pixels,
        ResidualSpace::Angular => ResidualUnit::AngularRadians,
        other => {
            return Err(SfmError::InvariantViolation(format!(
                "bundle-adjustment residual space {other:?} has no report unit"
            )));
        }
    };
    let native = solution.report();
    let diagnostics = native.diagnostics();
    let rank = match diagnostics.observability() {
        openmvg_ba::Observability::FullRank => BaRankAssessment::FullRank,
        openmvg_ba::Observability::OnlyExpectedGauge => BaRankAssessment::OnlyExpectedGauge,
        openmvg_ba::Observability::RankDeficient => BaRankAssessment::RankDeficient,
        openmvg_ba::Observability::NotAssessed => BaRankAssessment::NotAssessed,
        _ => BaRankAssessment::NotAssessed,
    };
    let observability =
        diagnostics
            .observability_details()
            .map(|details| BaObservabilityDiagnostics {
                pose_rank: details.pose_rank(),
                pose_dimension: details.pose_dimension(),
                weak_point_count: Some(details.weak_point_count()),
            });
    let status = canonical_solve_status(native.status());
    Ok(BundleAdjustmentReport {
        backend_name: backend_name.to_owned(),
        initial_objective: native.initial_objective(),
        final_objective: native.final_objective(),
        initial_rmse: native.initial_rmse(),
        final_rmse: native.final_rmse(),
        residual_units,
        iteration_count: native.iteration_count(),
        observation_count: problem.observations().len(),
        pose_count: problem.poses().len(),
        point_count: problem.points().len(),
        intrinsic_group_count: problem.intrinsics().len(),
        status,
        diagnostics: BaDiagnostics {
            accepted_step_count: Some(diagnostics.accepted_step_count()),
            rejected_step_count: Some(diagnostics.rejected_step_count()),
            gradient_inf_norm: diagnostics
                .stationarity()
                .map(|audit| audit.scaled_gradient_inf_norm()),
            rank,
            weak_point_count: diagnostics.weak_point_count(),
            noninvertible_observation_count: diagnostics.noninvertible_observation_count(),
            observability,
            ..BaDiagnostics::default()
        },
        termination_reason: format!("{status:?}"),
    })
}

/// Roundoff allowance for reconstructing a fixed baseline after iterative manifold updates.
///
/// The fixed term covers the final coordinate conversions and Euclidean norm. Each accepted
/// update adds a conservative budget for the S2 retraction and pose conversion. The report is an
/// untrusted backend value, so the iteration contribution is bounded by both the reported total
/// and the caller's configured maximum.
pub(crate) fn fixed_baseline_roundoff_tolerance(
    actual: f64,
    expected: f64,
    iterations: usize,
    accepted_steps: Option<usize>,
    max_iterations: usize,
) -> f64 {
    const FIXED_OPERATION_BUDGET: usize = 1024;
    const OPERATIONS_PER_ACCEPTED_UPDATE: usize = 64;

    let accepted_updates = accepted_steps
        .unwrap_or(iterations)
        .min(iterations)
        .min(max_iterations);
    let operation_budget = FIXED_OPERATION_BUDGET
        .saturating_add(OPERATIONS_PER_ACCEPTED_UPDATE.saturating_mul(accepted_updates));
    operation_budget as f64 * f64::EPSILON * actual.abs().max(expected.abs()).max(1.0)
}

fn refresh_landmark_reprojection_errors(
    workspace: &mut SfmWorkspace,
) -> Result<LandmarkSupportCleanupReport> {
    let landmark_ids = workspace
        .landmarks()
        .iter_active()
        .map(|landmark| landmark.id)
        .collect::<Vec<_>>();
    refresh_landmark_reprojection_errors_for(workspace, &landmark_ids)
}

pub(crate) fn refresh_landmark_reprojection_errors_for(
    workspace: &mut SfmWorkspace,
    landmark_ids: &[LandmarkId],
) -> Result<LandmarkSupportCleanupReport> {
    enum ReprojectionUpdate {
        Reject,
        RetainUnchanged(openmvg_core::PixelResidual),
        RetainPruned {
            support: ObservationSupport,
            reprojection_error: openmvg_core::PixelResidual,
            removed_observations: usize,
        },
    }

    let mut updates = Vec::with_capacity(landmark_ids.len());
    for &landmark_id in landmark_ids {
        let landmark = workspace
            .landmarks()
            .get(landmark_id)
            .ok_or(SfmError::InvalidId {
                kind: LandmarkId::kind(),
                index: landmark_id.index(),
            })?;
        if landmark.status != LandmarkStatus::Active {
            continue;
        }
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        let finite_residual = |observation: &crate::TrackObservation| -> Result<Option<f64>> {
            let Some(pose) = workspace.pose(observation.view) else {
                return Ok(None);
            };
            let image = workspace
                .view(observation.view)
                .ok_or(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: observation.view.index(),
                })?;
            let camera = workspace
                .camera_model(image.intrinsic_id)
                .ok_or(SfmError::InvalidId {
                    kind: IntrinsicId::kind(),
                    index: image.intrinsic_id.index(),
                })?;
            let camera_point = pose.pose.project_to_camera(&landmark.point);
            let policy = CameraResidualPolicy::from_pixel_limit(camera, f64::INFINITY)?;
            let pixel = workspace.track_observation_pixel(observation)?;
            Ok(policy
                .evaluate(camera, &camera_point.coords, pixel)
                .map(|evaluation| evaluation.pixel_equivalent))
        };

        let previous_support = landmark.observation_support.len(track.observations().len());
        let mut residual_sum = 0.0;
        let mut retained_count = 0usize;
        for (_, observation) in landmark.supporting_observations(track) {
            if let Some(residual) = finite_residual(observation)? {
                residual_sum += residual;
                retained_count += 1;
            }
        }

        let update = if retained_count < 2 {
            ReprojectionUpdate::Reject
        } else {
            let reprojection_error =
                openmvg_core::PixelResidual::try_new(residual_sum / retained_count as f64)
                    .map_err(|_| {
                        SfmError::InvalidInput(
                            "bundle refresh produced an invalid pixel residual".to_owned(),
                        )
                    })?;
            if retained_count == previous_support {
                ReprojectionUpdate::RetainUnchanged(reprojection_error)
            } else {
                let mut retained_indices = Vec::with_capacity(retained_count);
                for (source_index, observation) in landmark.supporting_observations(track) {
                    if finite_residual(observation)?.is_some() {
                        retained_indices.push(source_index.index());
                    }
                }
                ReprojectionUpdate::RetainPruned {
                    support: ObservationSupport::try_from_indices(
                        retained_indices,
                        track.observations().len(),
                    )?,
                    reprojection_error,
                    removed_observations: previous_support - retained_count,
                }
            }
        };
        updates.push((landmark.id, update));
    }

    let mut report = LandmarkSupportCleanupReport::default();
    for (landmark_id, update) in updates {
        let landmark =
            workspace
                .landmarks
                .get_mut(landmark_id)
                .ok_or(SfmError::InvariantViolation(format!(
                    "landmark {landmark_id} disappeared before residual commit"
                )))?;
        match update {
            ReprojectionUpdate::Reject => {
                landmark.status = LandmarkStatus::Rejected;
                report.rejected_landmarks += 1;
            }
            ReprojectionUpdate::RetainUnchanged(reprojection_error) => {
                landmark.reprojection_error = reprojection_error;
            }
            ReprojectionUpdate::RetainPruned {
                support,
                reprojection_error,
                removed_observations,
            } => {
                landmark.observation_support = support;
                landmark.reprojection_error = reprojection_error;
                report.removed_observations += removed_observations;
            }
        }
    }
    Ok(report)
}

/// Conservative angular-quality threshold for post-BA rollback decisions.
pub(crate) fn angular_regression_limit(rmse_rad: f64) -> f64 {
    (1.5 * rmse_rad).max(rmse_rad + 0.005)
}

/// Apply the standard pipeline's conservative policy before a precise refinement window exists.
///
/// A workspace whose posed views mix perspective and generalized cameras can later produce either
/// a pixel or an angular subproblem, and no single angle stands for one pixel across them. Preserve
/// explicitly angular losses; discard only losses whose pixel scale would otherwise be silently
/// reinterpreted. When every posed camera is generalized the problem is angular throughout, and
/// a pixel scale is converted to radians when the problem is built.
pub(crate) fn workspace_mixes_posed_perspective_and_generalized_cameras(
    workspace: &SfmWorkspace,
) -> bool {
    let mut generalized = false;
    let mut perspective = false;
    for camera in workspace
        .registered_views()
        .filter_map(|(view, _)| workspace.view(view))
        .filter_map(|image| workspace.camera_model(image.intrinsic_id))
    {
        if is_generalized_camera(camera) {
            generalized = true;
        } else {
            perspective = true;
        }
    }
    generalized && perspective
}

pub(crate) fn pipeline_ba_config_for_workspace(
    workspace: &SfmWorkspace,
    mut config: BundleAdjustConfig,
) -> BundleAdjustConfig {
    if matches!(config.scope(), OptimizationScope::SelfCalibration) {
        return config;
    }
    if workspace_mixes_posed_perspective_and_generalized_cameras(workspace)
        && config
            .robust_loss()
            .scale()
            .is_some_and(|scale| scale.space() == ResidualSpace::Pixels)
    {
        config = config.with_robust_loss(RobustLoss::Squared);
    }
    config
}

/// Build the immutable BA problem represented by the current
/// workspace and optimization configuration without invoking a solver or
/// mutating reconstruction state.
pub(crate) fn bundle_adjustment_problem(
    workspace: &SfmWorkspace,
    cfg: &BundleAdjustConfig,
) -> Result<BundleProblem> {
    prepare_bundle_adjustment_problem(workspace, cfg, &std::collections::BTreeSet::new())
        .map(|prepared| prepared.problem)
}

struct PreparedBundleProblem {
    problem: BundleProblem,
    pose_ids: Vec<crate::PoseId>,
    intrinsic_ids: Vec<IntrinsicId>,
    point_ids: Vec<LandmarkId>,
}

fn prepare_bundle_adjustment_problem(
    workspace: &SfmWorkspace,
    cfg: &BundleAdjustConfig,
    excluded: &std::collections::BTreeSet<LandmarkId>,
) -> Result<PreparedBundleProblem> {
    cfg.validate()?;
    // Views own measurements and calibration; PoseIds own the physical state. Never group
    // different identities merely because their current transforms happen to coincide.
    let camera_ids = workspace
        .registered_views()
        .map(|(view, _)| view)
        .collect::<Vec<_>>();
    let pose_ids = workspace
        .views()
        .iter()
        .filter(|view| workspace.has_pose(view.id))
        .map(|view| view.pose_id)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let pose_index = pose_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    // Preparation is also the preflight entry: resolve fixed aliases here rather than relying
    // on the execution wrapper to expand them before constructing the problem.
    let fixed_pose_ids = cfg
        .fixed_views()
        .iter()
        .map(|view| {
            workspace.pose_id(*view).ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: view.index(),
            })
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let image_intrinsics = workspace
        .views()
        .iter()
        .filter(|image| workspace.has_pose(image.id))
        .map(|image| (image.id, image.intrinsic_id))
        .collect::<BTreeMap<_, _>>();
    // Only groups seen by a posed camera are parameter blocks of this problem: an unregistered
    // group has no residual, and the commit writes back exactly the groups listed here.
    let intrinsic_ids = camera_ids
        .iter()
        .filter_map(|view| image_intrinsics.get(view).copied())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let intrinsic_index = intrinsic_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let pose_variables = |pose: crate::PoseId| {
        if fixed_pose_ids.contains(&pose) || matches!(cfg.scope(), OptimizationScope::PointsOnly) {
            PoseVariables::FIXED
        } else if matches!(cfg.scope(), OptimizationScope::TranslationsAndPoints) {
            PoseVariables::FIXED.with_translation(true)
        } else {
            PoseVariables::FREE
        }
    };
    let cameras = pose_ids
        .iter()
        .map(|id| {
            let record = workspace.pose_by_id(*id).ok_or(SfmError::InvalidId {
                kind: crate::PoseId::kind(),
                index: id.index(),
            })?;
            Ok(BundlePose::new(record.pose.clone(), pose_variables(*id)))
        })
        .collect::<Result<Vec<_>>>()?;
    let intrinsics = intrinsic_ids
        .iter()
        .map(|intrinsic_id| {
            let camera = workspace
                .camera_model(*intrinsic_id)
                .ok_or(SfmError::InvalidId {
                    kind: IntrinsicId::kind(),
                    index: intrinsic_id.index(),
                })?
                .clone();
            let mut dimensions = workspace
                .views()
                .iter()
                .filter(|image| image.intrinsic_id == *intrinsic_id)
                .map(|image| (image.meta.width, image.meta.height));
            let (width, height) = dimensions.next().ok_or_else(|| {
                SfmError::InvalidInput(format!(
                    "intrinsic {intrinsic_id} is not assigned to any view"
                ))
            })?;
            if dimensions.any(|candidate| candidate != (width, height)) {
                return Err(SfmError::InvalidInput(format!(
                    "intrinsic {intrinsic_id} is shared by images with different dimensions"
                )));
            }
            // A spherical or fisheye camera has no intrinsic the pipeline estimates, and angular
            // bundle adjustment solves only with fixed intrinsics.
            let fixed =
                cfg.scope() != OptimizationScope::SelfCalibration || is_generalized_camera(&camera);
            let data = CameraIntrinsics::try_new(camera, width, height)
                .map_err(|error| {
                    SfmError::InvalidInput(format!("invalid intrinsic {intrinsic_id}: {error}"))
                })?
                .with_fixed(fixed);
            let variables = if fixed {
                IntrinsicVariables::FIXED
            } else {
                IntrinsicVariables::default()
                    .with_focal(true)
                    .with_radial(true)
            };
            // A placeholder focal has nothing to be held to: let it move to what the
            // observations say, as a trusted one may not.
            let certification = if workspace.intrinsic_prior(*intrinsic_id)
                == Some(crate::model::IntrinsicPrior::Guess)
            {
                IntrinsicCertification::Waived
            } else {
                IntrinsicCertification::Required
            };
            BundleIntrinsic::try_new(data, variables)
                .map(|block| block.with_certification(certification))
                .map_err(|error| SfmError::InvalidInput(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let point_records = workspace
        .landmarks()
        .as_slice()
        .iter()
        .filter(|landmark| {
            landmark.status == LandmarkStatus::Active && !excluded.contains(&landmark.id)
        })
        .map(|landmark| (landmark.id, landmark.point))
        .collect::<Vec<_>>();
    let point_ids = point_records
        .iter()
        .map(|(landmark, _)| *landmark)
        .collect::<Vec<_>>();
    let point_index = point_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let points = point_records
        .iter()
        .map(|(id, point)| {
            BundlePoint::try_new(*point)
                .map(|point| {
                    point.with_fixed(
                        cfg.fixed_points().contains(id)
                            || matches!(cfg.scope(), OptimizationScope::PosesOnly),
                    )
                })
                .map_err(|error| SfmError::InvalidInput(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut observations = Vec::new();
    for landmark in workspace.landmarks().as_slice() {
        if landmark.status != LandmarkStatus::Active || excluded.contains(&landmark.id) {
            continue;
        }
        let track = workspace
            .tracks()
            .get(landmark.source_track)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: landmark.source_track.index(),
            })?;
        for (_, observation) in landmark.supporting_observations(track) {
            if workspace.has_pose(observation.view) {
                let view = workspace
                    .view(observation.view)
                    .ok_or(SfmError::InvalidId {
                        kind: ViewId::kind(),
                        index: observation.view.index(),
                    })?;
                let pose = pose_index
                    .get(&view.pose_id)
                    .copied()
                    .ok_or(SfmError::InvalidId {
                        kind: crate::PoseId::kind(),
                        index: view.pose_id.index(),
                    })?;
                let intrinsic = intrinsic_index.get(&view.intrinsic_id).copied().ok_or(
                    SfmError::InvalidId {
                        kind: IntrinsicId::kind(),
                        index: view.intrinsic_id.index(),
                    },
                )?;
                let point = point_index
                    .get(&landmark.id)
                    .copied()
                    .ok_or(SfmError::InvalidId {
                        kind: LandmarkId::kind(),
                        index: landmark.id.index(),
                    })?;
                observations.push(
                    BundleObservation::try_new(
                        pose,
                        intrinsic,
                        point,
                        workspace.track_observation_pixel(observation)?,
                    )
                    .map_err(|error| SfmError::InvalidInput(error.to_string()))?,
                );
            }
        }
    }
    let mut owning_components = workspace
        .components()
        .iter()
        .filter(|component| camera_ids.iter().any(|view| component.views.contains(view)));
    let owning_component = owning_components.next();
    if owning_components.next().is_some()
        || owning_component.is_some_and(|component| {
            camera_ids
                .iter()
                .any(|view| !component.views.contains(view))
        })
    {
        return Err(SfmError::InvalidInput(
            "bundle adjustment currently requires one reconstruction component per problem"
                .to_owned(),
        ));
    }
    let gauge = match owning_component.map(|component| component.gauge_scale_policy) {
        None | Some(GaugeScalePolicy::Free) => Gauge::Free,
        Some(GaugeScalePolicy::FixedBaseline {
            first: view_a,
            second: view_b,
            distance,
            ..
        }) => Gauge::FixedBaseline {
            first: workspace
                .pose_id(view_a)
                .and_then(|pose| pose_index.get(&pose).copied())
                .ok_or_else(|| {
                    SfmError::InvalidInput("fixed-baseline anchor is missing".to_owned())
                })?,
            second: workspace
                .pose_id(view_b)
                .and_then(|pose| pose_index.get(&pose).copied())
                .ok_or_else(|| {
                    SfmError::InvalidInput("fixed-baseline endpoint is missing".to_owned())
                })?,
            distance,
        },
    };
    let residual_space = if intrinsics.iter().any(|intrinsic| {
        matches!(
            intrinsic.intrinsics().camera(),
            openmvg_core::camera::Camera::Fisheye(_) | openmvg_core::camera::Camera::Spherical(_)
        )
    }) {
        ResidualSpace::Angular
    } else {
        ResidualSpace::Pixels
    };
    let radians_per_pixel = mean_radians_per_pixel(&intrinsics);
    let robust_loss = canonical_robust_loss(cfg.robust_loss(), residual_space, radians_per_pixel)?;
    let problem = BundleProblem::builder(cameras, intrinsics, points, observations)
        .with_residual_space(residual_space)
        .with_robust_loss(robust_loss)
        .with_gauge(gauge)
        .build()
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    Ok(PreparedBundleProblem {
        problem,
        pose_ids,
        intrinsic_ids,
        point_ids,
    })
}

/// Angle subtended by one pixel at the image center, averaged over the intrinsic groups: the
/// longitude step of a spherical image, or the reciprocal focal length of the others.
fn mean_radians_per_pixel(intrinsics: &[BundleIntrinsic]) -> f64 {
    let steps: Vec<f64> = intrinsics
        .iter()
        .filter_map(|intrinsic| match intrinsic.intrinsics().camera() {
            openmvg_core::camera::Camera::Spherical(camera) => {
                Some(std::f64::consts::TAU / f64::from(camera.width()))
            }
            camera => camera.focal_lengths().map(|(focal_x, _)| 1.0 / focal_x),
        })
        .collect();
    steps.iter().sum::<f64>() / steps.len().max(1) as f64
}

/// The loss in the units of the residual domain. A pixel scale over angular residuals, as when
/// a view is spherical or fisheye, is converted with the angle one pixel subtends.
fn canonical_robust_loss(
    loss: RobustLoss,
    residual_space: ResidualSpace,
    radians_per_pixel: f64,
) -> Result<RobustLoss> {
    let Some(scale) = loss.scale() else {
        return Ok(loss);
    };
    if scale.space() == residual_space {
        return Ok(loss);
    }
    let ResidualScale::Pixels(pixels) = scale else {
        return Err(SfmError::InvalidInput(
            "robust-loss units do not match the BA residual domain".to_owned(),
        ));
    };
    let angular = openmvg_core::AngularThreshold::try_new(pixels.value() * radians_per_pixel)
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    let scale = ResidualScale::Angular(angular);
    Ok(match loss {
        RobustLoss::Huber(_) => RobustLoss::Huber(scale),
        RobustLoss::Cauchy(_) => RobustLoss::Cauchy(scale),
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::{canonical_robust_loss, fixed_baseline_roundoff_tolerance};
    use openmvg_ba::{ResidualScale, ResidualSpace, RobustLoss};
    use openmvg_core::{AngularThreshold, PixelThreshold};

    #[test]
    fn a_pixel_loss_over_angular_residuals_is_converted_with_the_angle_of_a_pixel() {
        let pixels = ResidualScale::Pixels(PixelThreshold::try_new(2.0).unwrap());

        let converted =
            canonical_robust_loss(RobustLoss::Cauchy(pixels), ResidualSpace::Angular, 0.001)
                .unwrap();

        let Some(ResidualScale::Angular(angle)) = converted.scale() else {
            panic!("expected an angular scale, got {converted:?}");
        };
        assert!(matches!(converted, RobustLoss::Cauchy(_)));
        assert!((angle.value() - 0.002).abs() < 1e-15);
        assert_eq!(
            canonical_robust_loss(RobustLoss::Cauchy(pixels), ResidualSpace::Pixels, 0.001)
                .unwrap(),
            RobustLoss::Cauchy(pixels)
        );
        let angular = ResidualScale::Angular(AngularThreshold::try_new(0.01).unwrap());
        assert!(
            canonical_robust_loss(RobustLoss::Huber(angular), ResidualSpace::Pixels, 0.001)
                .is_err()
        );
    }

    #[test]
    fn fixed_baseline_tolerance_covers_zero_iteration_coordinate_roundoff() {
        let expected = 1.0;
        let actual = 0.999_999_999_999_845_1;
        let tolerance = fixed_baseline_roundoff_tolerance(actual, expected, 0, Some(0), 25);
        assert!((actual - expected).abs() <= tolerance);
        assert!(tolerance < 1.0e-12);
    }
}
