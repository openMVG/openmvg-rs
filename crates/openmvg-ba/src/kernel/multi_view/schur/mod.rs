// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Schur-complement assembly of the multi-view normal equations.

mod assembly;
mod diagnostics;
mod layout;
mod point;
mod priors;

#[cfg(any(test, feature = "faer"))]
pub(super) use assembly::Assembly;
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) use assembly::assemble_problem;
pub(super) use assembly::{BaProblemView, LinearizationPolicy};
#[cfg(feature = "faer")]
pub(super) use assembly::{
    HessianAccumulator, PointCacheBuffers, PointCrossCacheMut, PointStepCacheBuffers,
    SchurAssemblyMut, add_block_scalar, add_gram_scalar, assemble_with_hessian_problem,
    assemble_with_hessian_problem_translation_only, finalize_assembly, parameter_run_end,
    unreduced_scaled_gradient_inf_norm,
};
#[cfg(all(feature = "faer", feature = "parallel"))]
pub(super) use assembly::{assemble_retained_range, assemble_retained_range_translation_only};
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) use diagnostics::{diagnostics_with_expected_nullity, unassessed_diagnostics};
pub(super) use diagnostics::{expected_gauge_nullity, metrics};
#[cfg(feature = "unstable-internals")]
pub(super) use layout::configure_grouped_layout_with_fixed;
#[cfg(test)]
pub(super) use layout::configure_layout;
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) use layout::configure_layout_with_fixed;
pub(super) use layout::{GaugeState, VariableLayout};
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) use point::linearize_point;
#[cfg(feature = "unstable-internals")]
pub(super) use point::point_cost;
#[cfg(feature = "faer")]
pub(super) use point::{
    add_scaled_pair, cross_row, point_step_with_prior_and_damping_norm_translation_only,
    recompute_point_cross_rows, recompute_point_cross_rows_translation_only,
};
pub(super) use point::{candidate_point, fixed_point_cost, point_step_with_prior_and_damping_norm};
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) use priors::accumulate_point_prior;
pub(super) use priors::prior_cost;
#[cfg(feature = "faer")]
pub(super) use priors::{accumulate_camera_priors, accumulate_camera_priors_translation_only};

const POINT_DIMENSION: usize = 3;

const MAX_POSE_DIMENSION: usize = 6;

const MAX_CAMERA_BLOCK_DIMENSION: usize = 9;

/// `scale * row`, entry by entry.
#[inline(always)]
fn scaled<const N: usize>(row: &[f64; N], scale: f64) -> [f64; N] {
    let mut scaled = [0.0; N];
    for (target, value) in scaled.iter_mut().zip(row) {
        *target = scale * value;
    }
    scaled
}

#[inline(always)]
fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn multiply_matrix_vector<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|row| {
        (0..N)
            .map(|column| matrix[row][column] * vector[column])
            .sum()
    })
}

fn squared_vector_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

#[inline]
fn norm<const N: usize>(vector: &[f64; N]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests;
