// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hessian accumulators and the reduced camera system assembly.

use super::layout::{ActiveParameterTracker, GaugeState, VariableLayout, is_fixed};
use super::point::{add_scaled_pair, cross_row, linearize_point_with_hessian_and_active};
use super::priors::{accumulate_camera_priors_impl, accumulate_point_prior};
use super::{MAX_CAMERA_BLOCK_DIMENSION, POINT_DIMENSION, dot3};
use crate::kernel::RelativePose;
#[cfg(feature = "faer")]
use crate::kernel::multi_view::CameraPoseVariables;
#[cfg(all(feature = "faer", feature = "parallel"))]
use crate::kernel::multi_view::PointPrior;
#[cfg(any(test, feature = "unstable-internals"))]
use crate::kernel::multi_view::dense::DenseBundleAdjustmentWorkspace;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::{BundleAdjustmentPriors, FixedVariableBlocks};
#[cfg(feature = "faer")]
use crate::kernel::point_block::PackedPointBlock;
use crate::kernel::point_block::{POINT_BLOCK_RCOND_THRESHOLD, PointBlockFactor};
use crate::numerics::refine::{Loss, NoiseModel};

pub(in crate::kernel::multi_view) struct PointLinearization {
    pub hessian: [[f64; POINT_DIMENSION]; POINT_DIMENSION],
    pub gradient: [f64; POINT_DIMENSION],
    pub(super) cost: f64,
    pub(super) squared_residual: f64,
}

pub(in crate::kernel::multi_view) trait HessianAccumulator {
    fn clear(&mut self);
    fn add_symmetric(&mut self, row: usize, column: usize, value: f64);
    fn is_finite(&self) -> bool;

    /// End of the run of consecutive parameters starting at `start`.
    ///
    /// Block-structured accumulators override this so runs never straddle a
    /// storage block and [`Self::add_block`] can resolve each block once.
    #[inline]
    fn run_end(&self, parameters: &[usize], start: usize) -> usize {
        parameter_run_end(parameters, start, MAX_CAMERA_BLOCK_DIMENSION)
    }

    /// Add `value(i, j)` at `(rows[i], columns[j])` for one dense block.
    ///
    /// `rows` and `columns` are each runs returned by [`Self::run_end`]. With
    /// `lower`, they are the same run and only pairs `j <= i` are visited.
    /// Block-structured accumulators override this to resolve storage once
    /// per block instead of once per scalar.
    #[inline]
    fn add_block(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        lower: bool,
        value: impl FnMut(usize, usize) -> f64,
    ) where
        Self: Sized,
    {
        add_block_scalar(self, rows, columns, lower, value);
    }

    /// Add `left[0][i] * right[0][j]` and then `left[1][i] * right[1][j]` at
    /// `(rows[i], columns[j])`: one block of the Gram matrix of two weighted
    /// Jacobian rows.
    ///
    /// With `lower`, `rows` and `columns` are the same run, `right` is `left`,
    /// and only pairs `j <= i` are visited; otherwise the runs are disjoint.
    /// Block-structured accumulators override this to update contiguous
    /// storage lines when each run lies inside one variable block.
    #[inline]
    fn add_gram(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        lower: bool,
        left: [&[f64]; 2],
        right: [&[f64]; 2],
    ) {
        add_gram_scalar(self, rows, columns, lower, left, right);
    }
}

/// Scalar reference for [`HessianAccumulator::add_block`].
#[inline]
pub(in crate::kernel::multi_view) fn add_block_scalar<H: HessianAccumulator>(
    hessian: &mut H,
    rows: &[usize],
    columns: &[usize],
    lower: bool,
    mut value: impl FnMut(usize, usize) -> f64,
) {
    for (i, &row) in rows.iter().enumerate() {
        let count = if lower { i + 1 } else { columns.len() };
        for (j, &column) in columns.iter().take(count).enumerate() {
            hessian.add_symmetric(row, column, value(i, j));
        }
    }
}

/// Scalar reference for [`HessianAccumulator::add_gram`].
#[inline]
pub(in crate::kernel::multi_view) fn add_gram_scalar<H: HessianAccumulator + ?Sized>(
    hessian: &mut H,
    rows: &[usize],
    columns: &[usize],
    lower: bool,
    left: [&[f64]; 2],
    right: [&[f64]; 2],
) {
    for (i, ((&row, &first), &second)) in rows.iter().zip(left[0]).zip(left[1]).enumerate() {
        let count = if lower { i + 1 } else { columns.len() };
        for ((&column, &column_first), &column_second) in
            columns.iter().zip(right[0]).zip(right[1]).take(count)
        {
            hessian.add_symmetric(row, column, first * column_first);
            hessian.add_symmetric(row, column, second * column_second);
        }
    }
}

/// End of the consecutive-parameter run starting at `start`, capped at
/// `maximum_length` and at one camera block so run-local scratch stays
/// fixed-size.
#[inline]
pub(in crate::kernel::multi_view) fn parameter_run_end(
    parameters: &[usize],
    start: usize,
    maximum_length: usize,
) -> usize {
    let limit = parameters
        .len()
        .min(start + maximum_length.clamp(1, MAX_CAMERA_BLOCK_DIMENSION));
    let mut end = start + 1;
    while end < limit && parameters[end] == parameters[end - 1] + 1 {
        end += 1;
    }
    end
}

#[cfg(any(test, feature = "unstable-internals"))]
pub(super) struct DenseHessian<'a> {
    pub(super) values: &'a mut [f64],
    pub(super) dimension: usize,
}

#[cfg(any(test, feature = "unstable-internals"))]
impl HessianAccumulator for DenseHessian<'_> {
    fn clear(&mut self) {
        self.values[..self.dimension * self.dimension].fill(0.0);
    }

    #[inline(always)]
    fn add_symmetric(&mut self, row: usize, column: usize, value: f64) {
        self.values[row * self.dimension + column] += value;
        if row != column {
            self.values[column * self.dimension + row] += value;
        }
    }

    fn is_finite(&self) -> bool {
        self.values[..self.dimension * self.dimension]
            .iter()
            .all(|value| value.is_finite())
    }
}

pub(in crate::kernel::multi_view) struct ReducedBuffers<'a, H> {
    pub(super) hessian: &'a mut H,
    pub(super) gradient: &'a mut [f64],
    pub(super) full_gradient: &'a mut [f64],
    pub(super) diagonal: &'a mut [f64],
}

impl<H> ReducedBuffers<'_, H> {
    /// Add the gradient and diagonal terms of the parameters in `span`, whose
    /// two scaled Jacobian rows are `terms`.
    #[inline(always)]
    pub(super) fn add_run_terms(
        &mut self,
        span: std::ops::Range<usize>,
        [first, second]: [&[f64]; 2],
        [residual_first, residual_second]: [f64; 2],
    ) {
        for gradient in [&mut *self.gradient, &mut *self.full_gradient] {
            let gradient = &mut gradient[span.clone()];
            add_scaled_pair(gradient, first, residual_first, second, residual_second);
        }
        for ((diagonal, first), second) in self.diagonal[span].iter_mut().zip(first).zip(second) {
            *diagonal += first * first;
            *diagonal += second * second;
        }
    }

    /// [`Self::add_run_terms`] for a run of `D` parameters from `start`.
    #[inline(always)]
    pub(super) fn add_run<const D: usize>(
        &mut self,
        start: usize,
        terms: [&[f64]; 2],
        residual: [f64; 2],
    ) {
        // Local copies cannot alias the buffers, so each entry is loaded and
        // stored once.
        let (mut first, mut second) = ([0.0; D], [0.0; D]);
        first.copy_from_slice(&terms[0][..D]);
        second.copy_from_slice(&terms[1][..D]);
        self.add_run_terms(start..start + D, [&first, &second], residual);
    }
}

#[cfg_attr(not(feature = "faer"), allow(dead_code))]
pub(in crate::kernel::multi_view) enum PointCrossCacheMut<'a> {
    F64(&'a mut [[f64; POINT_DIMENSION]]),
    F32(&'a mut [[f32; POINT_DIMENSION]]),
    Recomputed { len: usize },
}

impl PointCrossCacheMut<'_> {
    #[cfg(all(feature = "faer", feature = "parallel"))]
    pub(in crate::kernel::multi_view) fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Recomputed { len } => *len,
        }
    }

    #[cfg(all(feature = "faer", feature = "parallel"))]
    pub(in crate::kernel::multi_view) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(in crate::kernel::multi_view) const fn retains_values(&self) -> bool {
        !matches!(self, Self::Recomputed { .. })
    }

    pub(in crate::kernel::multi_view) fn set(
        &mut self,
        index: usize,
        value: [f64; POINT_DIMENSION],
    ) {
        match self {
            Self::F64(values) => values[index] = value,
            Self::F32(values) => values[index] = value.map(|value| value as f32),
            Self::Recomputed { len } => debug_assert!(index < *len),
        }
    }

    #[cfg(all(feature = "faer", feature = "parallel"))]
    pub(in crate::kernel::multi_view) fn split_at_mut(self, middle: usize) -> (Self, Self) {
        match self {
            Self::F64(values) => {
                let (left, right) = values.split_at_mut(middle);
                (Self::F64(left), Self::F64(right))
            }
            Self::F32(values) => {
                let (left, right) = values.split_at_mut(middle);
                (Self::F32(left), Self::F32(right))
            }
            Self::Recomputed { len } => {
                debug_assert!(middle <= len);
                (
                    Self::Recomputed { len: middle },
                    Self::Recomputed { len: len - middle },
                )
            }
        }
    }
}

pub(in crate::kernel::multi_view) struct PointStepCacheBuffers<'a> {
    pub valid: &'a mut [bool],
    pub inverse_gradients: &'a mut [[f64; POINT_DIMENSION]],
    pub damping_diagonals: &'a mut [[f64; POINT_DIMENSION]],
}

pub(in crate::kernel::multi_view) struct PointCacheBuffers<'a> {
    pub point_steps: Option<PointStepCacheBuffers<'a>>,
    pub cross_offsets: &'a [usize],
    pub inverse_cross_values: PointCrossCacheMut<'a>,
    pub point_offset: usize,
    pub cross_offset: usize,
    #[cfg(feature = "faer")]
    pub damped_hessians: Option<&'a mut [PackedPointBlock]>,
}

pub(in crate::kernel::multi_view) struct Assembly {
    pub valid: bool,
    pub weak_point_count: usize,
    pub minimum_point_rcond: f64,
    pub scaled_gradient_inf_norm: f64,
    /// Reduction contributed by eliminating the damped point blocks.
    pub point_model_reduction: f64,
    pub cost: f64,
    pub squared_residual: f64,
    pub minimum_reduced_diagonal: f64,
    pub maximum_reduced_diagonal: f64,
    pub damping_floor_clamp_count: usize,
}

impl Assembly {
    #[cfg(feature = "faer")]
    pub(in crate::kernel::multi_view) fn objective_metrics(
        &self,
        observation_count: usize,
    ) -> Option<(f64, f64)> {
        if observation_count == 0 || !(self.cost.is_finite() && self.squared_residual.is_finite()) {
            return None;
        }
        Some((
            self.cost,
            (self.squared_residual / (2 * observation_count) as f64).sqrt(),
        ))
    }

    #[cfg(all(feature = "parallel", feature = "faer"))]
    pub(in crate::kernel::multi_view) const fn empty() -> Self {
        Self {
            valid: true,
            weak_point_count: 0,
            minimum_point_rcond: 1.0,
            scaled_gradient_inf_norm: 0.0,
            point_model_reduction: 0.0,
            cost: 0.0,
            squared_residual: 0.0,
            minimum_reduced_diagonal: 0.0,
            maximum_reduced_diagonal: 0.0,
            damping_floor_clamp_count: 0,
        }
    }

    #[cfg(feature = "faer")]
    pub(in crate::kernel::multi_view) const fn invalid() -> Self {
        Self {
            valid: false,
            weak_point_count: 0,
            minimum_point_rcond: 0.0,
            scaled_gradient_inf_norm: f64::INFINITY,
            point_model_reduction: 0.0,
            cost: f64::INFINITY,
            squared_residual: f64::INFINITY,
            minimum_reduced_diagonal: 0.0,
            maximum_reduced_diagonal: 0.0,
            damping_floor_clamp_count: 0,
        }
    }
}

/// Immutable state and factors consumed by one Schur assembly.
#[derive(Clone, Copy)]
pub(in crate::kernel::multi_view) struct BaProblemView<'a> {
    pub model: ObservationModel<'a>,
    pub poses: &'a [RelativePose],
    pub points: &'a [[f64; 3]],
    pub fixed: FixedVariableBlocks<'a>,
    pub priors: BundleAdjustmentPriors<'a>,
}

/// Numerical policy applied while forming one damped linearization.
#[derive(Clone, Copy)]
pub(in crate::kernel::multi_view) struct LinearizationPolicy<'a> {
    pub gauge: GaugeState,
    pub loss: Loss,
    pub noise: NoiseModel<'a>,
    pub damping: f64,
    pub prior_sqrt_scale: f64,
}

/// Exclusive output and scratch borrowed for one Schur assembly.
pub(in crate::kernel::multi_view) struct SchurAssemblyMut<'a, H> {
    pub dimension: usize,
    pub layout: VariableLayout<'a>,
    pub materialize_schur: bool,
    /// Apply camera priors, damping, and final finite checks in this call.
    ///
    /// Explicit parallel materialization defers this phase until every
    /// column-owned point-coupling contribution has been written.
    pub finalize: bool,
    pub hessian: &'a mut H,
    pub gradient: &'a mut [f64],
    pub full_gradient: &'a mut [f64],
    pub diagonal: &'a mut [f64],
    pub cross: &'a mut [f64],
    pub active_parameters: &'a mut [usize],
    pub parameter_marks: &'a mut [bool],
    pub point_cache: Option<PointCacheBuffers<'a>>,
}

#[cfg(any(test, feature = "unstable-internals"))]
pub(in crate::kernel::multi_view) fn assemble_problem(
    problem: BaProblemView<'_>,
    policy: LinearizationPolicy<'_>,
    dimension: usize,
    workspace: &mut DenseBundleAdjustmentWorkspace,
) -> Assembly {
    let layout = VariableLayout {
        offsets: &workspace.camera_offsets,
        pose_dimensions: &workspace.camera_pose_dimensions,
        intrinsic_offsets: &workspace.intrinsic_offsets,
        intrinsic_dimensions: &workspace.intrinsic_dimensions,
        pose_masks: &[],
        #[cfg(feature = "faer")]
        pose_variables: CameraPoseVariables::RotationAndTranslation,
    };
    let mut hessian = DenseHessian {
        values: &mut workspace.hessian,
        dimension,
    };
    assemble_with_hessian_problem(
        problem,
        policy,
        SchurAssemblyMut {
            dimension,
            layout,
            materialize_schur: true,
            finalize: true,
            hessian: &mut hessian,
            gradient: &mut workspace.gradient,
            full_gradient: &mut workspace.full_gradient,
            diagonal: &mut workspace.diagonal,
            cross: &mut workspace.cross,
            active_parameters: &mut workspace.active_parameters,
            parameter_marks: &mut workspace.parameter_marks,
            point_cache: None,
        },
    )
}

/// Assemble one contiguous point partition of a retained Schur system.
///
/// Camera blocks and vectors are worker-local. Point-cache slices correspond
/// exactly to `points`, while `point_offset` and `cross_offset` retain global
/// model indexing. Damping of the reduced camera diagonal and final finite
/// checks are deferred until the ordered worker reduction.
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "faer", feature = "parallel"))]
pub(in crate::kernel::multi_view) fn assemble_retained_range<H: HessianAccumulator>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    point_offset: usize,
    fixed_points: &[bool],
    point_priors: &[PointPrior],
    prior_sqrt_scale: f64,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    dimension: usize,
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
    layout: VariableLayout<'_>,
    point_cache: PointCacheBuffers<'_>,
) -> Assembly {
    assemble_retained_range_impl::<false, H>(
        model,
        poses,
        points,
        point_offset,
        fixed_points,
        point_priors,
        prior_sqrt_scale,
        gauge,
        loss,
        noise,
        damping,
        dimension,
        hessian,
        gradient,
        full_gradient,
        diagonal,
        cross,
        active_parameters,
        parameter_marks,
        layout,
        point_cache,
    )
}

/// Translation-only specialization of [`assemble_retained_range`].
#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "faer", feature = "parallel"))]
pub(in crate::kernel::multi_view) fn assemble_retained_range_translation_only<
    H: HessianAccumulator,
>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    point_offset: usize,
    fixed_points: &[bool],
    point_priors: &[PointPrior],
    prior_sqrt_scale: f64,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    dimension: usize,
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
    layout: VariableLayout<'_>,
    point_cache: PointCacheBuffers<'_>,
) -> Assembly {
    assemble_retained_range_impl::<true, H>(
        model,
        poses,
        points,
        point_offset,
        fixed_points,
        point_priors,
        prior_sqrt_scale,
        gauge,
        loss,
        noise,
        damping,
        dimension,
        hessian,
        gradient,
        full_gradient,
        diagonal,
        cross,
        active_parameters,
        parameter_marks,
        layout,
        point_cache,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "faer", feature = "parallel"))]
fn assemble_retained_range_impl<const TRANSLATION_ONLY: bool, H: HessianAccumulator>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    point_offset: usize,
    fixed_points: &[bool],
    point_priors: &[PointPrior],
    prior_sqrt_scale: f64,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    dimension: usize,
    hessian: &mut H,
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
    layout: VariableLayout<'_>,
    mut point_cache: PointCacheBuffers<'_>,
) -> Assembly {
    hessian.clear();
    gradient[..dimension].fill(0.0);
    full_gradient[..dimension].fill(0.0);
    diagonal[..dimension].fill(0.0);
    if let Some(point_steps) = point_cache.point_steps.as_mut() {
        point_steps.valid.fill(false);
    }
    let retain_cross_rows = point_cache.inverse_cross_values.retains_values();
    let mut assembly = Assembly {
        valid: true,
        weak_point_count: 0,
        minimum_point_rcond: 1.0,
        scaled_gradient_inf_norm: 0.0,
        point_model_reduction: 0.0,
        cost: 0.0,
        squared_residual: 0.0,
        minimum_reduced_diagonal: 0.0,
        maximum_reduced_diagonal: 0.0,
        damping_floor_clamp_count: 0,
    };

    for (local_point, value) in points.iter().enumerate() {
        let point = point_offset + local_point;
        let local = linearize_point_with_hessian_and_active::<true, TRANSLATION_ONLY, _>(
            model,
            poses,
            value,
            point,
            gauge,
            loss,
            noise,
            layout,
            cross,
            Some(ReducedBuffers {
                hessian,
                gradient,
                full_gradient,
                diagonal,
            }),
            Some(ActiveParameterTracker::new(
                active_parameters,
                parameter_marks,
            )),
        );
        let Some((local, active_count)) = local else {
            assembly.valid = false;
            continue;
        };
        let mut local = local;
        if let Ok(prior_index) = point_priors.binary_search_by_key(&point, |prior| prior.point) {
            accumulate_point_prior(
                &mut local,
                value,
                &point_priors[prior_index],
                gauge.scene_scale,
                prior_sqrt_scale,
            );
        }
        assembly.cost += local.cost;
        assembly.squared_residual += local.squared_residual;
        if is_fixed(fixed_points, point) {
            continue;
        }
        let point_diagonal: [f64; POINT_DIMENSION] =
            std::array::from_fn(|index| local.hessian[index][index]);
        assembly.scaled_gradient_inf_norm = assembly
            .scaled_gradient_inf_norm
            .max(scaled_gradient_inf_norm(&local.gradient, &point_diagonal));
        let Some(factor) = PointBlockFactor::new(&local.hessian, damping) else {
            assembly.valid = false;
            if damping == 0.0 {
                assembly.weak_point_count += 1;
                assembly.minimum_point_rcond = 0.0;
            }
            continue;
        };
        let Some(damping_diagonal) = PointBlockFactor::damping_diagonal(&local.hessian) else {
            assembly.valid = false;
            continue;
        };
        if let Some(point_steps) = point_cache.point_steps.as_mut() {
            point_steps.damping_diagonals[local_point] = damping_diagonal;
        }
        let Some(damped) = PointBlockFactor::damped_matrix(&local.hessian, damping) else {
            assembly.valid = false;
            continue;
        };
        let Some(point_hessians) = point_cache.damped_hessians.as_deref_mut() else {
            assembly.valid = false;
            continue;
        };
        point_hessians[local_point] = PackedPointBlock::from_matrix(&damped);
        if damping == 0.0 {
            let rcond = factor.reciprocal_condition_number(&local.hessian);
            assembly.minimum_point_rcond = assembly.minimum_point_rcond.min(rcond);
            if rcond <= POINT_BLOCK_RCOND_THRESHOLD {
                assembly.weak_point_count += 1;
            }
        }
        let inverse_gradient = factor.solve(&local.gradient);
        assembly.point_model_reduction += 0.5 * dot3(&local.gradient, &inverse_gradient);
        let global_range = point_cache.cross_offsets[point]..point_cache.cross_offsets[point + 1];
        let range = (global_range.start - point_cache.cross_offset)
            ..(global_range.end - point_cache.cross_offset);
        if range.len() != active_count {
            assembly.valid = false;
            continue;
        }
        if let Some(point_steps) = point_cache.point_steps.as_mut() {
            point_steps.valid[local_point] = true;
            point_steps.inverse_gradients[local_point] = inverse_gradient;
        }
        for (row_index, &row) in active_parameters[..active_count].iter().enumerate() {
            let row_cross = cross_row(cross, row);
            gradient[row] -= dot3(&row_cross, &inverse_gradient);
            if retain_cross_rows {
                point_cache
                    .inverse_cross_values
                    .set(range.start + row_index, factor.solve(&row_cross));
            }
        }
    }
    assembly
}

#[allow(clippy::too_many_arguments)]
pub(in crate::kernel::multi_view) fn finalize_assembly<H: HessianAccumulator>(
    hessian: &mut H,
    gradient: &[f64],
    full_gradient: &[f64],
    diagonal: &mut [f64],
    damping: f64,
    dimension: usize,
    layout: VariableLayout<'_>,
    mut assembly: Assembly,
) -> Assembly {
    let active_diagonal = &diagonal[..dimension];
    let scale = active_diagonal.iter().copied().fold(0.0, f64::max);
    assembly.minimum_reduced_diagonal = active_diagonal
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if dimension == 0 {
        assembly.minimum_reduced_diagonal = 0.0;
    }
    assembly.maximum_reduced_diagonal = scale;
    assembly.damping_floor_clamp_count =
        prepare_damping_diagonal(&mut diagonal[..dimension], layout);
    for (parameter, &curvature) in diagonal[..dimension].iter().enumerate() {
        hessian.add_symmetric(parameter, parameter, damping * curvature);
    }
    assembly.scaled_gradient_inf_norm =
        assembly
            .scaled_gradient_inf_norm
            .max(scaled_gradient_inf_norm_prepared(
                &full_gradient[..dimension],
                &diagonal[..dimension],
            ));
    assembly.valid = assembly.valid
        && hessian.is_finite()
        && gradient[..dimension].iter().all(|value| value.is_finite());
    assembly
}

pub(super) fn prepare_damping_diagonal(diagonal: &mut [f64], layout: VariableLayout<'_>) -> usize {
    let mut clamped = 0;
    for (&offset, &dimension) in layout.offsets.iter().zip(layout.pose_dimensions) {
        clamped += clamp_damping_block(&mut diagonal[offset..offset + dimension]);
    }
    for (&offset, &dimension) in layout
        .intrinsic_offsets
        .iter()
        .zip(layout.intrinsic_dimensions)
    {
        clamped += clamp_damping_block(&mut diagonal[offset..offset + dimension]);
    }
    clamped
}

fn clamp_damping_block(diagonal: &mut [f64]) -> usize {
    if diagonal.is_empty() {
        return 0;
    }
    let scale = diagonal.iter().copied().fold(0.0, f64::max);
    // A block without curvature (for example a camera whose observations were
    // all rejected) has no scale to be relative to. A floor derived from
    // `MIN_POSITIVE` would underflow to zero once multiplied by the damping,
    // leaving a zero pivot, so such a block is damped with unit curvature.
    let floor = if scale > 0.0 {
        64.0 * f64::EPSILON * scale
    } else {
        1.0
    };
    let mut clamped = 0;
    for curvature in diagonal {
        if *curvature < floor {
            *curvature = floor;
            clamped += 1;
        }
    }
    clamped
}

pub(in crate::kernel::multi_view) fn assemble_with_hessian_problem<H: HessianAccumulator>(
    problem: BaProblemView<'_>,
    policy: LinearizationPolicy<'_>,
    target: SchurAssemblyMut<'_, H>,
) -> Assembly {
    assemble_with_hessian_problem_impl::<false, H>(problem, policy, target)
}

#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn assemble_with_hessian_problem_translation_only<
    H: HessianAccumulator,
>(
    problem: BaProblemView<'_>,
    policy: LinearizationPolicy<'_>,
    target: SchurAssemblyMut<'_, H>,
) -> Assembly {
    assemble_with_hessian_problem_impl::<true, H>(problem, policy, target)
}

/// Scaled gradient infinity norm of a linearization whose Schur reduction is
/// invalid.
///
/// A singular undamped point block (a weak point) invalidates the reduced
/// system but not the gradient `J^T r` or its curvature scaling. This serial
/// pass forms both without accumulating any Hessian, so the final audit can
/// still measure stationarity at the returned state.
#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn unreduced_scaled_gradient_inf_norm(
    problem: BaProblemView<'_>,
    policy: LinearizationPolicy<'_>,
    dimension: usize,
    layout: VariableLayout<'_>,
) -> f64 {
    let mut gradient = vec![0.0; dimension];
    let mut full_gradient = vec![0.0; dimension];
    let mut diagonal = vec![0.0; dimension];
    let mut cross = vec![0.0; dimension.saturating_mul(3)];
    let mut active_parameters = vec![0; dimension];
    let mut parameter_marks = vec![false; dimension];
    let target = SchurAssemblyMut {
        dimension,
        layout,
        materialize_schur: false,
        finalize: true,
        hessian: &mut IgnoredHessian,
        gradient: &mut gradient,
        full_gradient: &mut full_gradient,
        diagonal: &mut diagonal,
        cross: &mut cross,
        active_parameters: &mut active_parameters,
        parameter_marks: &mut parameter_marks,
        point_cache: None,
    };
    let assembly = match layout.pose_variables {
        CameraPoseVariables::RotationAndTranslation => {
            assemble_with_hessian_problem_impl::<false, _>(problem, policy, target)
        }
        CameraPoseVariables::TranslationOnly => {
            assemble_with_hessian_problem_impl::<true, _>(problem, policy, target)
        }
    };
    assembly.scaled_gradient_inf_norm
}

fn assemble_with_hessian_problem_impl<const TRANSLATION_ONLY: bool, H: HessianAccumulator>(
    problem: BaProblemView<'_>,
    policy: LinearizationPolicy<'_>,
    target: SchurAssemblyMut<'_, H>,
) -> Assembly {
    let BaProblemView {
        model,
        poses,
        points,
        fixed,
        priors,
    } = problem;
    let LinearizationPolicy {
        gauge,
        loss,
        noise,
        damping,
        prior_sqrt_scale,
    } = policy;
    let SchurAssemblyMut {
        dimension,
        layout,
        materialize_schur,
        finalize,
        hessian,
        gradient,
        full_gradient,
        diagonal,
        cross,
        active_parameters,
        parameter_marks,
        mut point_cache,
    } = target;
    let fixed_points = fixed.points;
    hessian.clear();
    gradient[..dimension].fill(0.0);
    full_gradient[..dimension].fill(0.0);
    diagonal[..dimension].fill(0.0);
    if let Some(point_steps) = point_cache
        .as_mut()
        .and_then(|cache| cache.point_steps.as_mut())
    {
        point_steps.valid.fill(false);
    }
    let retain_cross_rows = point_cache
        .as_ref()
        .is_some_and(|cache| cache.inverse_cross_values.retains_values());
    let mut assembly = Assembly {
        valid: true,
        weak_point_count: 0,
        minimum_point_rcond: 1.0,
        scaled_gradient_inf_norm: 0.0,
        point_model_reduction: 0.0,
        cost: 0.0,
        squared_residual: 0.0,
        minimum_reduced_diagonal: 0.0,
        maximum_reduced_diagonal: 0.0,
        damping_floor_clamp_count: 0,
    };
    let mut point_prior_index = 0;

    for (point, value) in points.iter().enumerate() {
        let local = linearize_point_with_hessian_and_active::<true, TRANSLATION_ONLY, _>(
            model,
            poses,
            value,
            point,
            gauge,
            loss,
            noise,
            layout,
            cross,
            Some(ReducedBuffers {
                hessian,
                gradient,
                full_gradient,
                diagonal,
            }),
            Some(ActiveParameterTracker::new(
                active_parameters,
                parameter_marks,
            )),
        );
        let Some((local, active_count)) = local else {
            assembly.valid = false;
            continue;
        };
        let mut local = local;
        let point_prior = priors
            .points
            .get(point_prior_index)
            .filter(|prior| prior.point == point);
        if let Some(prior) = point_prior {
            accumulate_point_prior(
                &mut local,
                value,
                prior,
                gauge.scene_scale,
                prior_sqrt_scale,
            );
            point_prior_index += 1;
        }
        assembly.cost += local.cost;
        assembly.squared_residual += local.squared_residual;
        if is_fixed(fixed_points, point) {
            continue;
        }
        let point_diagonal: [f64; POINT_DIMENSION] =
            std::array::from_fn(|index| local.hessian[index][index]);
        assembly.scaled_gradient_inf_norm = assembly
            .scaled_gradient_inf_norm
            .max(scaled_gradient_inf_norm(&local.gradient, &point_diagonal));
        let Some(factor) = PointBlockFactor::new(&local.hessian, damping) else {
            assembly.valid = false;
            if damping == 0.0 {
                assembly.weak_point_count += 1;
                assembly.minimum_point_rcond = 0.0;
            }
            continue;
        };
        if let Some(cache) = point_cache.as_mut() {
            let cache_point = point - cache.point_offset;
            if let Some(point_steps) = cache.point_steps.as_mut() {
                let Some(damping_diagonal) = PointBlockFactor::damping_diagonal(&local.hessian)
                else {
                    assembly.valid = false;
                    continue;
                };
                point_steps.damping_diagonals[cache_point] = damping_diagonal;
            }
        }
        #[cfg(feature = "faer")]
        if let Some(cache) = point_cache.as_mut() {
            let cache_point = point - cache.point_offset;
            if let Some(hessians) = cache.damped_hessians.as_deref_mut() {
                let Some(damped) = PointBlockFactor::damped_matrix(&local.hessian, damping) else {
                    assembly.valid = false;
                    continue;
                };
                hessians[cache_point] = PackedPointBlock::from_matrix(&damped);
            }
        }
        if damping == 0.0 {
            let rcond = factor.reciprocal_condition_number(&local.hessian);
            assembly.minimum_point_rcond = assembly.minimum_point_rcond.min(rcond);
            if rcond <= POINT_BLOCK_RCOND_THRESHOLD {
                assembly.weak_point_count += 1;
            }
        }
        let inverse_gradient = factor.solve(&local.gradient);
        assembly.point_model_reduction += 0.5 * dot3(&local.gradient, &inverse_gradient);
        let cache_range = if let Some(cache) = point_cache.as_mut() {
            let range = (cache.cross_offsets[point] - cache.cross_offset)
                ..(cache.cross_offsets[point + 1] - cache.cross_offset);
            if range.len() != active_count {
                assembly.valid = false;
                continue;
            }
            if let Some(point_steps) = cache.point_steps.as_mut() {
                let cache_point = point - cache.point_offset;
                point_steps.valid[cache_point] = true;
                point_steps.inverse_gradients[cache_point] = inverse_gradient;
            }
            Some(range)
        } else {
            None
        };
        let active = &active_parameters[..active_count];
        let mut row_start = 0;
        while row_start < active.len() {
            let row_end = hessian.run_end(active, row_start);
            let rows = &active[row_start..row_end];
            let mut inverse_rows = [[0.0; POINT_DIMENSION]; MAX_CAMERA_BLOCK_DIMENSION];
            for (local, &row) in rows.iter().enumerate() {
                let row_cross = cross_row(cross, row);
                gradient[row] -= dot3(&row_cross, &inverse_gradient);
                if materialize_schur || retain_cross_rows {
                    inverse_rows[local] = factor.solve(&row_cross);
                }
                if retain_cross_rows {
                    let (Some(cache), Some(range)) = (point_cache.as_mut(), cache_range.as_ref())
                    else {
                        assembly.valid = false;
                        continue;
                    };
                    cache
                        .inverse_cross_values
                        .set(range.start + row_start + local, inverse_rows[local]);
                }
            }
            if materialize_schur {
                let cross = &*cross;
                let mut column_start = 0;
                while column_start < row_start {
                    let column_end = hessian.run_end(active, column_start);
                    let columns = &active[column_start..column_end];
                    hessian.add_block(rows, columns, false, |i, j| {
                        -dot3(&inverse_rows[i], &cross_row(cross, columns[j]))
                    });
                    column_start = column_end;
                }
                hessian.add_block(rows, rows, true, |i, j| {
                    -dot3(&inverse_rows[i], &cross_row(cross, rows[j]))
                });
            }
            row_start = row_end;
        }
    }

    if !finalize {
        return assembly;
    }

    accumulate_camera_priors_impl::<TRANSLATION_ONLY, H>(
        model,
        poses,
        gauge,
        layout,
        priors,
        prior_sqrt_scale,
        hessian,
        gradient,
        full_gradient,
        diagonal,
        &mut assembly,
    );

    finalize_assembly(
        hessian,
        gradient,
        full_gradient,
        diagonal,
        damping,
        dimension,
        layout,
        assembly,
    )
}

pub(in crate::kernel::multi_view) struct IgnoredHessian;

impl HessianAccumulator for IgnoredHessian {
    fn clear(&mut self) {}
    fn add_symmetric(&mut self, _row: usize, _column: usize, _value: f64) {}
    fn is_finite(&self) -> bool {
        true
    }
}

fn scaled_gradient_inf_norm(gradient: &[f64], diagonal: &[f64]) -> f64 {
    let scale = diagonal.iter().copied().fold(0.0, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return if gradient.iter().all(|value| *value == 0.0) {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let floor = 64.0 * f64::EPSILON * scale;
    gradient
        .iter()
        .zip(diagonal)
        .map(|(&value, &curvature)| value.abs() / curvature.max(floor).sqrt())
        .fold(0.0, f64::max)
}

fn scaled_gradient_inf_norm_prepared(gradient: &[f64], damping_diagonal: &[f64]) -> f64 {
    gradient
        .iter()
        .zip(damping_diagonal)
        .map(|(&value, &curvature)| {
            if curvature.is_finite() && curvature > 0.0 {
                value.abs() / curvature.sqrt()
            } else if value == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}
