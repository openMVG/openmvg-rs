// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Construction-time sparse backend selection and its immutable receipt.

use super::api::{
    RetainedCrossPrecision, SparseBundleAdjustmentError, SparseExecutionBackend,
    SparseExecutionDecision, SparseExecutionReason, SparseWorkspaceMemoryBudget,
};
use super::workspace::dense_rank_diagnostic_len;

// Account conservatively for CSC values, symbolic-fill-dependent LLT values,
// and reusable factor/solve scratch without materializing a rejected pattern.
const DIRECT_NUMERIC_ESTIMATE_MULTIPLIER: usize = 4;
const DIRECT_BUDGET_UTILIZATION_NUMERATOR: usize = 4;
const DIRECT_BUDGET_UTILIZATION_DENOMINATOR: usize = 5;
// Direct-LLT cost rises sharply beyond this joint frontier on BlendedMVS
// reconstructions. Keep both bounds: observations alone would route dense
// two-view problems to PCG, while dimension alone would route small, sparse
// camera graphs without enough point work to amortize retained Schur.
const CACHED_SPEED_MIN_OBSERVATIONS: usize = 250_000;
const CACHED_SPEED_MIN_REDUCED_DIMENSION: usize = 768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExecutionEstimates {
    pub(super) direct: usize,
    pub(super) cached_retained: usize,
    pub(super) cached_retained_f32: Option<usize>,
    pub(super) recomputed_retained: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AutomaticExecutionPlan {
    pub(super) budget: SparseWorkspaceMemoryBudget,
    pub(super) selected: SparseExecutionBackend,
    pub(super) reason: SparseExecutionReason,
    pub(super) retained_cross_precision: RetainedCrossPrecision,
    pub(super) estimates: ExecutionEstimates,
}

impl AutomaticExecutionPlan {
    /// Whether the selected iterative strategy keeps the final exact camera-rank certificate,
    /// which factors the reduced camera system once after the solve.
    ///
    /// A commit needs that certificate, and the automatic budget only guides the choice of
    /// strategy, so under it the certificate always runs. An explicit byte budget keeps it when
    /// the certificate fits beside the selected strategy's own storage.
    pub(super) fn rank_certificate_fits(&self, certificate_bytes: usize) -> bool {
        let SparseWorkspaceMemoryBudget::Bytes(limit) = self.budget else {
            return true;
        };
        let backend = match (self.selected, self.retained_cross_precision) {
            (SparseExecutionBackend::DirectLlt, _) => return true,
            (SparseExecutionBackend::CachedIterativeSchur, RetainedCrossPrecision::F64) => {
                Some(self.estimates.cached_retained)
            }
            (SparseExecutionBackend::CachedIterativeSchur, RetainedCrossPrecision::F32) => {
                self.estimates.cached_retained_f32
            }
            (SparseExecutionBackend::RecomputedIterativeSchur, _) => {
                self.estimates.recomputed_retained
            }
        };
        backend
            .is_some_and(|bytes| direct_fits_budget(bytes.saturating_add(certificate_bytes), limit))
    }
}

pub(super) fn estimated_point_cache_bytes(
    point_count: usize,
    point_cross_count: usize,
    retain_point_steps: bool,
    retained_cross_precision: Option<RetainedCrossPrecision>,
) -> usize {
    let offsets = point_count
        .saturating_add(1)
        .saturating_mul(std::mem::size_of::<usize>());
    let point_steps = if retain_point_steps {
        point_count
            .saturating_mul(std::mem::size_of::<bool>() + 2 * std::mem::size_of::<[f64; 3]>())
    } else {
        0
    };
    let cross_rows = match retained_cross_precision {
        Some(RetainedCrossPrecision::F64) => {
            point_cross_count.saturating_mul(std::mem::size_of::<[f64; 3]>())
        }
        Some(RetainedCrossPrecision::F32) => {
            point_cross_count.saturating_mul(std::mem::size_of::<[f32; 3]>())
        }
        None => 0,
    };
    offsets
        .saturating_add(point_steps)
        .saturating_add(cross_rows)
}

/// Conservative fill-and-scratch estimate. Exact symbolic fill is not built,
/// since building it only to reject the direct backend under a memory budget
/// would cost the memory the budget protects.
pub(super) fn estimated_direct_bytes(
    reduced_nonzero_count: usize,
    reduced_dimension: usize,
    point_cache_bytes: usize,
    direct_assembly_bytes: usize,
    dense_rank_diagnostics: bool,
) -> usize {
    let numeric = reduced_nonzero_count
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_mul(DIRECT_NUMERIC_ESTIMATE_MULTIPLIER);
    let dense_rank = if dense_rank_diagnostics {
        dense_rank_diagnostic_len(reduced_dimension).saturating_mul(std::mem::size_of::<f64>())
    } else {
        0
    };
    numeric
        .saturating_add(dense_rank)
        .saturating_add(point_cache_bytes)
        .saturating_add(direct_assembly_bytes)
}

pub(super) fn select_calibrated(
    budget: SparseWorkspaceMemoryBudget,
    estimates: ExecutionEstimates,
    observation_count: usize,
    reduced_dimension: usize,
) -> Result<AutomaticExecutionPlan, SparseBundleAdjustmentError> {
    validate_budget(budget)?;
    let Some(limit) = budget.effective_bytes() else {
        return Ok(AutomaticExecutionPlan {
            budget,
            selected: SparseExecutionBackend::DirectLlt,
            reason: SparseExecutionReason::UnboundedPrefersDirect,
            retained_cross_precision: RetainedCrossPrecision::F64,
            estimates,
        });
    };
    let prefers_cached = observation_count >= CACHED_SPEED_MIN_OBSERVATIONS
        && reduced_dimension >= CACHED_SPEED_MIN_REDUCED_DIMENSION;
    let cached_f64_fits = estimates.cached_retained <= limit;
    let cached_f32_fits = estimates
        .cached_retained_f32
        .is_some_and(|estimate| estimate <= limit);
    let (selected, reason, retained_cross_precision) = if prefers_cached && cached_f64_fits {
        (
            SparseExecutionBackend::CachedIterativeSchur,
            SparseExecutionReason::WorkloadPrefersCachedRetained,
            RetainedCrossPrecision::F64,
        )
    } else if prefers_cached && cached_f32_fits {
        (
            SparseExecutionBackend::CachedIterativeSchur,
            SparseExecutionReason::CompressedCachedRetainedFitsBudget,
            RetainedCrossPrecision::F32,
        )
    } else if direct_fits_budget(estimates.direct, limit) {
        (
            SparseExecutionBackend::DirectLlt,
            SparseExecutionReason::DirectFitsBudget,
            RetainedCrossPrecision::F64,
        )
    } else if cached_f64_fits {
        (
            SparseExecutionBackend::CachedIterativeSchur,
            SparseExecutionReason::DirectExceedsBudget,
            RetainedCrossPrecision::F64,
        )
    } else if cached_f32_fits {
        (
            SparseExecutionBackend::CachedIterativeSchur,
            SparseExecutionReason::CompressedCachedRetainedFitsBudget,
            RetainedCrossPrecision::F32,
        )
    } else {
        (
            SparseExecutionBackend::RecomputedIterativeSchur,
            SparseExecutionReason::CachedRetainedExceedsBudget,
            RetainedCrossPrecision::F64,
        )
    };
    Ok(AutomaticExecutionPlan {
        budget,
        selected,
        reason,
        retained_cross_precision,
        estimates,
    })
}

pub(super) fn select_self_calibrating(
    budget: SparseWorkspaceMemoryBudget,
    estimates: ExecutionEstimates,
) -> Result<AutomaticExecutionPlan, SparseBundleAdjustmentError> {
    validate_budget(budget)?;
    let Some(limit) = budget.effective_bytes() else {
        return Ok(AutomaticExecutionPlan {
            budget,
            selected: SparseExecutionBackend::DirectLlt,
            reason: SparseExecutionReason::UnboundedPrefersDirect,
            retained_cross_precision: RetainedCrossPrecision::F64,
            estimates,
        });
    };
    let (selected, reason) = if direct_fits_budget(estimates.direct, limit) {
        (
            SparseExecutionBackend::DirectLlt,
            SparseExecutionReason::DirectFitsBudget,
        )
    } else if estimates.cached_retained <= limit {
        (
            SparseExecutionBackend::CachedIterativeSchur,
            SparseExecutionReason::DirectExceedsBudget,
        )
    } else {
        (
            SparseExecutionBackend::CachedIterativeSchur,
            SparseExecutionReason::LowestMemoryBackend,
        )
    };
    Ok(AutomaticExecutionPlan {
        budget,
        selected,
        reason,
        retained_cross_precision: RetainedCrossPrecision::F64,
        estimates,
    })
}

pub(super) fn finish_automatic_decision(
    plan: AutomaticExecutionPlan,
    selected_workspace_bytes: usize,
    direct_assembly_threads: Option<usize>,
    retained_threads: Option<usize>,
) -> SparseExecutionDecision {
    let estimated_cached_retained_bytes = match plan.retained_cross_precision {
        RetainedCrossPrecision::F64 => Some(plan.estimates.cached_retained),
        RetainedCrossPrecision::F32 => plan.estimates.cached_retained_f32,
    };
    SparseExecutionDecision {
        automatic: true,
        memory_budget: plan.budget,
        selected_backend: plan.selected,
        reason: plan.reason,
        estimated_direct_bytes: Some(plan.estimates.direct),
        estimated_cached_retained_bytes,
        estimated_recomputed_retained_bytes: plan.estimates.recomputed_retained,
        selected_workspace_bytes,
        direct_assembly_threads,
        retained_threads,
    }
}

pub(super) fn explicit_decision(
    selected_backend: SparseExecutionBackend,
    selected_workspace_bytes: usize,
    direct_assembly_threads: Option<usize>,
    retained_threads: Option<usize>,
) -> SparseExecutionDecision {
    SparseExecutionDecision {
        automatic: false,
        memory_budget: SparseWorkspaceMemoryBudget::Unbounded,
        selected_backend,
        reason: SparseExecutionReason::ExplicitBackend,
        estimated_direct_bytes: None,
        estimated_cached_retained_bytes: None,
        estimated_recomputed_retained_bytes: None,
        selected_workspace_bytes,
        direct_assembly_threads,
        retained_threads,
    }
}

fn validate_budget(budget: SparseWorkspaceMemoryBudget) -> Result<(), SparseBundleAdjustmentError> {
    if matches!(budget, SparseWorkspaceMemoryBudget::Bytes(0)) {
        return Err(SparseBundleAdjustmentError::WorkspaceMemoryBudgetMustBePositive);
    }
    Ok(())
}

pub(super) fn direct_fits_budget(estimate: usize, limit: usize) -> bool {
    let guarded_limit = limit / DIRECT_BUDGET_UTILIZATION_DENOMINATOR
        * DIRECT_BUDGET_UTILIZATION_NUMERATOR
        + limit % DIRECT_BUDGET_UTILIZATION_DENOMINATOR * DIRECT_BUDGET_UTILIZATION_NUMERATOR
            / DIRECT_BUDGET_UTILIZATION_DENOMINATOR;
    estimate <= guarded_limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_auto_degrades_monotonically_with_budget() {
        let estimates = ExecutionEstimates {
            direct: 300,
            cached_retained: 200,
            cached_retained_f32: None,
            recomputed_retained: Some(100),
        };
        assert_eq!(
            select_calibrated(SparseWorkspaceMemoryBudget::Bytes(375), estimates, 1, 1)
                .expect("positive budget")
                .selected,
            SparseExecutionBackend::DirectLlt
        );
        assert_eq!(
            select_calibrated(SparseWorkspaceMemoryBudget::Bytes(250), estimates, 1, 1)
                .expect("positive budget")
                .selected,
            SparseExecutionBackend::CachedIterativeSchur
        );
        assert_eq!(
            select_calibrated(SparseWorkspaceMemoryBudget::Bytes(150), estimates, 1, 1)
                .expect("positive budget")
                .selected,
            SparseExecutionBackend::RecomputedIterativeSchur
        );
        assert_eq!(
            select_calibrated(
                SparseWorkspaceMemoryBudget::Unbounded,
                estimates,
                usize::MAX,
                usize::MAX
            )
            .expect("unbounded policy")
            .reason,
            SparseExecutionReason::UnboundedPrefersDirect
        );
    }

    #[test]
    fn direct_selection_reserves_twenty_percent_of_the_budget() {
        let fits = ExecutionEstimates {
            direct: 800,
            cached_retained: 700,
            cached_retained_f32: Some(600),
            recomputed_retained: Some(600),
        };
        let exceeds = ExecutionEstimates {
            direct: 801,
            ..fits
        };
        assert_eq!(
            select_calibrated(SparseWorkspaceMemoryBudget::Bytes(1_000), fits, 1, 1)
                .expect("positive budget")
                .selected,
            SparseExecutionBackend::DirectLlt
        );
        assert_eq!(
            select_calibrated(SparseWorkspaceMemoryBudget::Bytes(1_000), exceeds, 1, 1)
                .expect("positive budget")
                .selected,
            SparseExecutionBackend::CachedIterativeSchur
        );
    }

    #[test]
    fn zero_budget_is_rejected() {
        let estimates = ExecutionEstimates {
            direct: 3,
            cached_retained: 2,
            cached_retained_f32: Some(1),
            recomputed_retained: Some(1),
        };
        assert_eq!(
            select_calibrated(SparseWorkspaceMemoryBudget::Bytes(0), estimates, 1, 1),
            Err(SparseBundleAdjustmentError::WorkspaceMemoryBudgetMustBePositive)
        );
    }

    #[test]
    fn calibrated_auto_uses_bounded_measured_speed_frontier() {
        let estimates = ExecutionEstimates {
            direct: 300,
            cached_retained: 200,
            cached_retained_f32: Some(150),
            recomputed_retained: Some(100),
        };
        let budget = SparseWorkspaceMemoryBudget::Bytes(1_000);
        let selected = |observations, dimension| {
            select_calibrated(budget, estimates, observations, dimension).expect("positive budget")
        };

        assert_eq!(
            selected(
                CACHED_SPEED_MIN_OBSERVATIONS,
                CACHED_SPEED_MIN_REDUCED_DIMENSION
            )
            .selected,
            SparseExecutionBackend::CachedIterativeSchur
        );
        assert_eq!(
            selected(
                CACHED_SPEED_MIN_OBSERVATIONS - 1,
                CACHED_SPEED_MIN_REDUCED_DIMENSION
            )
            .selected,
            SparseExecutionBackend::DirectLlt
        );
        assert_eq!(
            selected(
                CACHED_SPEED_MIN_OBSERVATIONS,
                CACHED_SPEED_MIN_REDUCED_DIMENSION - 1
            )
            .selected,
            SparseExecutionBackend::DirectLlt
        );
        assert_eq!(
            selected(
                CACHED_SPEED_MIN_OBSERVATIONS,
                CACHED_SPEED_MIN_REDUCED_DIMENSION
            )
            .reason,
            SparseExecutionReason::WorkloadPrefersCachedRetained
        );
    }

    #[test]
    fn speed_frontier_does_not_override_cached_workspace_budget() {
        let estimates = ExecutionEstimates {
            direct: 300,
            cached_retained: 1_001,
            cached_retained_f32: None,
            recomputed_retained: Some(100),
        };
        let plan = select_calibrated(
            SparseWorkspaceMemoryBudget::Bytes(1_000),
            estimates,
            CACHED_SPEED_MIN_OBSERVATIONS,
            CACHED_SPEED_MIN_REDUCED_DIMENSION,
        )
        .expect("positive budget");
        assert_eq!(plan.selected, SparseExecutionBackend::DirectLlt);
        assert_eq!(plan.reason, SparseExecutionReason::DirectFitsBudget);
    }

    #[test]
    fn cached_rank_certificate_must_fit_beside_the_retained_rows() {
        let estimates = ExecutionEstimates {
            direct: 300,
            cached_retained: 200,
            cached_retained_f32: Some(150),
            recomputed_retained: Some(100),
        };
        let plan = select_calibrated(
            SparseWorkspaceMemoryBudget::Bytes(1_000),
            estimates,
            CACHED_SPEED_MIN_OBSERVATIONS,
            CACHED_SPEED_MIN_REDUCED_DIMENSION,
        )
        .expect("positive budget");
        assert_eq!(plan.selected, SparseExecutionBackend::CachedIterativeSchur);
        assert!(plan.rank_certificate_fits(600));
        assert!(!plan.rank_certificate_fits(601));
    }

    #[test]
    fn recomputed_rows_keep_a_rank_certificate_that_fits_beside_them() {
        let estimates = ExecutionEstimates {
            direct: 300,
            cached_retained: 200,
            cached_retained_f32: None,
            recomputed_retained: Some(100),
        };
        let plan = select_calibrated(SparseWorkspaceMemoryBudget::Bytes(150), estimates, 1, 1)
            .expect("positive budget");
        assert_eq!(
            plan.selected,
            SparseExecutionBackend::RecomputedIterativeSchur
        );
        assert!(plan.rank_certificate_fits(20));
        assert!(!plan.rank_certificate_fits(21));
    }

    #[test]
    fn the_automatic_budget_always_keeps_the_rank_certificate() {
        // Rows too large for the automatic budget select the recomputed strategy; a commit still
        // needs the certificate, and the automatic budget only guides the strategy.
        let too_large = SparseWorkspaceMemoryBudget::AUTO_BYTES + 1;
        let estimates = ExecutionEstimates {
            direct: too_large,
            cached_retained: too_large,
            cached_retained_f32: Some(too_large),
            recomputed_retained: Some(too_large),
        };
        let plan = select_calibrated(SparseWorkspaceMemoryBudget::Auto, estimates, 1, 1)
            .expect("automatic budget");
        assert_eq!(
            plan.selected,
            SparseExecutionBackend::RecomputedIterativeSchur
        );
        assert!(plan.rank_certificate_fits(usize::MAX));
    }

    #[test]
    fn calibrated_auto_prefers_compact_cached_rows_before_recomputation() {
        let estimates = ExecutionEstimates {
            direct: 2_000,
            cached_retained: 1_100,
            cached_retained_f32: Some(700),
            recomputed_retained: Some(200),
        };
        let plan = select_calibrated(
            SparseWorkspaceMemoryBudget::Bytes(1_000),
            estimates,
            CACHED_SPEED_MIN_OBSERVATIONS,
            CACHED_SPEED_MIN_REDUCED_DIMENSION,
        )
        .expect("positive budget");

        assert_eq!(plan.selected, SparseExecutionBackend::CachedIterativeSchur);
        assert_eq!(
            plan.reason,
            SparseExecutionReason::CompressedCachedRetainedFitsBudget
        );
        assert_eq!(plan.retained_cross_precision, RetainedCrossPrecision::F32);
        let decision = finish_automatic_decision(plan, 650, None, Some(4));
        assert_eq!(decision.estimated_cached_retained_bytes(), Some(700));
        assert_eq!(decision.selected_workspace_bytes(), 650);
    }
}
