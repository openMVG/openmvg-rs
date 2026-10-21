// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Conversion of kernel reports and errors into the checked solve report.

use super::{NativePreflight, NativePreflightStatus};
use crate::kernel::multi_view::sparse::{
    SparseBundleAdjustmentReport, SparseCameraRankAssessment, SparseExecutionBackend,
};
use crate::kernel::two_view::BundleAdjustmentReport as TwoViewBundleAdjustmentReport;
use crate::numerics::refine::RefinementStatus;
use crate::{
    ConvergenceReason, Gauge, LinearSolver, Observability, ObservabilityDiagnostics,
    SolveDiagnostics, SolveError, SolveReport, SolveStatus, StationarityAudit,
};

pub(super) fn map_solution_error(error: crate::SolutionError) -> SolveError {
    match error {
        crate::SolutionError::GaugePrecisionLimited => SolveError::Numerical(error.to_string()),
        _ => SolveError::Internal(error.to_string()),
    }
}

pub(super) fn solve_report(
    report: &SparseBundleAdjustmentReport,
    preflight: Option<NativePreflight>,
    gauge: Gauge,
) -> Result<SolveReport, SolveError> {
    let status = map_refinement_status(report.status);
    let observability = certified_observability(report.diagnostics.pose_rank, preflight, gauge);
    let stationarity = StationarityAudit::try_from_fresh_zero_damping(report.gradient_inf_norm)
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let mut diagnostics = SolveDiagnostics::new(
        report.accepted_step_count,
        report.rejected_step_count,
        stationarity,
        observability,
        report.diagnostics.weak_point_count,
        report.noninvertible_measurement_count,
    )
    .with_step_rejections(report.step_rejections)
    .with_linear_solver(linear_solver(report.execution_decision.selected_backend()));
    if let Some(preflight) = preflight {
        let pose_dimension = preflight.pose_dimension.ok_or_else(|| {
            SolveError::Internal("fixed-rotation probe omitted pose dimension".to_owned())
        })?;
        let weak_point_count = preflight.weak_point_count.ok_or_else(|| {
            SolveError::Internal("fixed-rotation probe omitted weak-point count".to_owned())
        })?;
        let pose_rank = if matches!(preflight.status, NativePreflightStatus::Proceed)
            && (!preflight.dense_observability_escalation
                || matches!(
                    observability,
                    Observability::FullRank | Observability::OnlyExpectedGauge
                )) {
            pose_dimension
        } else {
            0
        };
        let details =
            ObservabilityDiagnostics::try_new(pose_rank, pose_dimension, weak_point_count)
                .map_err(|error| SolveError::Internal(error.to_string()))?;
        diagnostics = diagnostics.with_observability_details(details);
    } else if let Some(details) = assessed_observability_details(report, observability) {
        diagnostics = diagnostics.with_observability_details(details);
    }
    SolveReport::try_new(
        status,
        report.iteration_count,
        report.initial_cost,
        report.final_cost,
        report.initial_rmse * 2.0_f64.sqrt(),
        report.final_rmse * 2.0_f64.sqrt(),
        diagnostics,
    )
    .map_err(|error| SolveError::Internal(error.to_string()))
}

/// Observability details for a solve that ran no fixed-rotation probe, such as one over angular
/// residuals, from the exact rank assessment of its final reduced camera system.
///
/// The reduced dimension drops the independent scale null that a free gauge leaves. `None` when
/// the assessment was disabled or the system is empty.
fn assessed_observability_details(
    report: &SparseBundleAdjustmentReport,
    observability: Observability,
) -> Option<ObservabilityDiagnostics> {
    let reduced = report.diagnostics.pose_dimension;
    let (rank, dimension) = match observability {
        Observability::FullRank => (reduced, reduced),
        Observability::OnlyExpectedGauge => (reduced.checked_sub(1)?, reduced.checked_sub(1)?),
        Observability::RankDeficient => (0, reduced),
        Observability::NotAssessed => return None,
    };
    ObservabilityDiagnostics::try_new(rank, dimension, report.diagnostics.weak_point_count).ok()
}

pub(super) fn certified_observability(
    rank: SparseCameraRankAssessment,
    post_solve_probe: Option<NativePreflight>,
    gauge: Gauge,
) -> Observability {
    let primary = match rank {
        SparseCameraRankAssessment::FullRank => Observability::FullRank,
        SparseCameraRankAssessment::OnlyExpectedScaleGauge => Observability::OnlyExpectedGauge,
        SparseCameraRankAssessment::RankDeficientOrIndefinite => Observability::RankDeficient,
        SparseCameraRankAssessment::NotAssessed => Observability::NotAssessed,
    };
    if !matches!(primary, Observability::NotAssessed) {
        return primary;
    }
    let Some(probe) = post_solve_probe else {
        return primary;
    };
    // The probe runs on the accepted geometry. Its selected tracks form a
    // positive-definite gauge-reduced Schur system; adding the omitted
    // positive-semidefinite track contributions cannot destroy that rank.
    let subset_certifies_full_reduced_rank = matches!(probe.status, NativePreflightStatus::Proceed)
        && !probe.dense_observability_escalation
        && probe.weak_point_count == Some(0);
    if !subset_certifies_full_reduced_rank {
        return primary;
    }
    match gauge {
        Gauge::FixedBaseline { .. } => Observability::FullRank,
        Gauge::Free => Observability::OnlyExpectedGauge,
    }
}

pub(super) fn map_refinement_status(status: RefinementStatus) -> SolveStatus {
    match status {
        RefinementStatus::Converged(reason) => SolveStatus::Converged(match reason {
            crate::numerics::refine::ConvergenceCriterion::Cost => ConvergenceReason::Objective,
            crate::numerics::refine::ConvergenceCriterion::Gradient => ConvergenceReason::Gradient,
            crate::numerics::refine::ConvergenceCriterion::Step => ConvergenceReason::Step,
        }),
        RefinementStatus::MaxIterations => SolveStatus::MaxIterations,
        RefinementStatus::Stalled => SolveStatus::Stalled,
        RefinementStatus::Degenerate => SolveStatus::Degenerate,
    }
}

pub(super) fn two_view_solve_report(
    report: &TwoViewBundleAdjustmentReport,
) -> Result<SolveReport, SolveError> {
    let status = match report.status {
        RefinementStatus::Converged(reason) => SolveStatus::Converged(match reason {
            crate::numerics::refine::ConvergenceCriterion::Cost => ConvergenceReason::Objective,
            crate::numerics::refine::ConvergenceCriterion::Gradient => ConvergenceReason::Gradient,
            crate::numerics::refine::ConvergenceCriterion::Step => ConvergenceReason::Step,
        }),
        RefinementStatus::MaxIterations => SolveStatus::MaxIterations,
        RefinementStatus::Stalled => SolveStatus::Stalled,
        RefinementStatus::Degenerate => SolveStatus::Degenerate,
    };
    let observability = if report.diagnostics.camera_is_full_rank() {
        Observability::FullRank
    } else {
        Observability::RankDeficient
    };
    let stationarity = StationarityAudit::try_from_fresh_zero_damping(report.gradient_inf_norm)
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    SolveReport::try_new(
        status,
        report.iteration_count,
        report.initial_cost,
        report.final_cost,
        report.initial_rmse * 2.0_f64.sqrt(),
        report.final_rmse * 2.0_f64.sqrt(),
        SolveDiagnostics::new(
            report.accepted_step_count,
            report.rejected_step_count,
            stationarity,
            observability,
            report.diagnostics.weak_point_count,
            0,
        )
        .with_step_rejections(report.step_rejections)
        .with_linear_solver(LinearSolver::Direct),
    )
    .map_err(|error| SolveError::Internal(error.to_string()))
}

/// The public linear-solver choice that a sparse backend implements.
fn linear_solver(backend: SparseExecutionBackend) -> LinearSolver {
    match backend {
        SparseExecutionBackend::DirectLlt => LinearSolver::Direct,
        SparseExecutionBackend::CachedIterativeSchur
        | SparseExecutionBackend::RecomputedIterativeSchur => LinearSolver::Iterative,
    }
}

pub(super) fn map_native_error(
    error: crate::kernel::multi_view::sparse::SparseBundleAdjustmentError,
) -> SolveError {
    if error.is_resource_exhausted() {
        SolveError::ResourceExhausted(error.to_string())
    } else {
        SolveError::Numerical(error.to_string())
    }
}
