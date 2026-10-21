// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic point-parallel linearization for retained self-calibration.

use crate::kernel::point_block::PackedPointBlock;

use super::super::super::schur::{
    self, BaProblemView, HessianAccumulator, LinearizationPolicy, PointCacheBuffers,
    PointCrossCacheMut, PointStepCacheBuffers, VariableLayout,
};
use super::super::api::SparseBundleAdjustmentError;
use super::super::retained::{
    PointCrossRows, PointCrossStorage, balanced_contiguous_ranges, deterministic_point_major_shards,
};
use super::operator::SelfCalibrationMatvec;
use super::{PackedBlock, PoseIntrinsicEdge, SelfCalibrationRetainedHessian};

/// Per-logical-shard reduced-system scratch.
///
/// A physical worker may process several logical shards over successive waves.
/// Keeping the logical ranges independent of physical worker count and merging
/// them in ascending order makes the floating-point reduction identical at
/// every executor width.
struct LinearizationScratch {
    blocks: Vec<PackedBlock>,
    edges: Vec<PoseIntrinsicEdge>,
    gradient: Vec<f64>,
    full_gradient: Vec<f64>,
    diagonal: Vec<f64>,
    cross: Vec<f64>,
    active_parameters: Vec<usize>,
    parameter_marks: Vec<bool>,
    assembly: Option<schur::Assembly>,
}

impl LinearizationScratch {
    fn try_new(
        block_count: usize,
        edge_topology: &[PoseIntrinsicEdge],
        dimension: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let cross_len = dimension
            .checked_mul(3)
            .ok_or(SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
        Ok(Self {
            blocks: vec![PackedBlock::default(); block_count],
            edges: edge_topology.to_vec(),
            gradient: vec![0.0; dimension],
            full_gradient: vec![0.0; dimension],
            diagonal: vec![0.0; dimension],
            cross: vec![0.0; cross_len],
            active_parameters: vec![0; dimension],
            parameter_marks: vec![false; dimension],
            assembly: None,
        })
    }

    fn retained_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<PackedBlock>()
            + self.edges.capacity() * std::mem::size_of::<PoseIntrinsicEdge>()
            + (self.gradient.capacity()
                + self.full_gradient.capacity()
                + self.diagonal.capacity()
                + self.cross.capacity())
                * std::mem::size_of::<f64>()
            + self.active_parameters.capacity() * std::mem::size_of::<usize>()
            + self.parameter_marks.capacity() * std::mem::size_of::<bool>()
    }
}

#[derive(Clone, Copy)]
struct LinearizationContext<'a> {
    problem: BaProblemView<'a>,
    policy: LinearizationPolicy<'a>,
    layout: VariableLayout<'a>,
    dimension: usize,
    pose_edges: &'a [usize],
    parameter_blocks: &'a [usize],
    block_offsets: &'a [usize],
    camera_count: usize,
    point_cross_offsets: &'a [usize],
}

pub(super) struct SelfCalibrationLinearizationWorkspace {
    point_ranges: Vec<usize>,
    scratch: Vec<LinearizationScratch>,
}

impl SelfCalibrationLinearizationWorkspace {
    pub(super) fn try_new(
        threads: usize,
        observation_count: usize,
        point_work_offsets: &[usize],
        block_count: usize,
        edge_topology: &[PoseIntrinsicEdge],
        dimension: usize,
    ) -> Result<Option<Self>, SparseBundleAdjustmentError> {
        let worker_bytes = estimated_worker_bytes(block_count, edge_topology.len(), dimension);
        let Some(logical_shards) =
            deterministic_point_major_shards(observation_count, worker_bytes)
        else {
            return Ok(None);
        };
        let point_ranges = balanced_contiguous_ranges(point_work_offsets, logical_shards);
        let active_shards = point_ranges.len().saturating_sub(1).min(threads);
        if active_shards == 0 {
            return Ok(None);
        }
        let scratch = (0..active_shards)
            .map(|_| LinearizationScratch::try_new(block_count, edge_topology, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            point_ranges,
            scratch,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn assemble_points(
        &mut self,
        matvec: &SelfCalibrationMatvec,
        problem: BaProblemView<'_>,
        policy: LinearizationPolicy<'_>,
        layout: VariableLayout<'_>,
        pose_edges: &[usize],
        parameter_blocks: &[usize],
        block_offsets: &[usize],
        camera_count: usize,
        point_cross_offsets: &[usize],
        mut point_steps: Option<PointStepCacheBuffers<'_>>,
        inverse_crosses: PointCrossCacheMut<'_>,
        point_hessians: &mut [PackedPointBlock],
        target_blocks: &mut [PackedBlock],
        target_edges: &mut [PoseIntrinsicEdge],
        gradient: &mut [f64],
        full_gradient: &mut [f64],
        diagonal: &mut [f64],
    ) -> Option<schur::Assembly> {
        let point_count = problem.points.len();
        let dimension = gradient.len();
        let valid_shape = problem.model.offsets().len() == point_count + 1
            && point_cross_offsets.len() == point_count + 1
            && point_cross_offsets.first().copied() == Some(0)
            && point_cross_offsets
                .windows(2)
                .all(|range| range[0] <= range[1])
            && point_hessians.len() == point_count
            && inverse_crosses.len() == point_cross_offsets.last().copied().unwrap_or(0)
            && target_blocks.len() == block_offsets.len()
            && target_edges.len() <= pose_edges.len()
            && full_gradient.len() == dimension
            && diagonal.len() == dimension
            && parameter_blocks.len() == dimension
            && self.point_ranges.first().copied() == Some(0)
            && self.point_ranges.last().copied() == Some(point_count)
            && point_steps.as_ref().is_none_or(|steps| {
                steps.valid.len() == point_count
                    && steps.inverse_gradients.len() == point_count
                    && steps.damping_diagonals.len() == point_count
            });
        if !valid_shape {
            return Some(schur::Assembly::invalid());
        }

        for block in &mut *target_blocks {
            block.values.fill(0.0);
        }
        for edge in &mut *target_edges {
            edge.values.fill(0.0);
        }
        gradient.fill(0.0);
        full_gradient.fill(0.0);
        diagonal.fill(0.0);
        if let Some(steps) = point_steps.as_mut() {
            steps.valid.fill(false);
        }

        let context = LinearizationContext {
            problem,
            policy,
            layout,
            dimension,
            pose_edges,
            parameter_blocks,
            block_offsets,
            camera_count,
            point_cross_offsets,
        };
        let shard_count = self.point_ranges.len().saturating_sub(1);
        let wave_width = self.scratch.len();
        if shard_count == 0 {
            return Some(schur::Assembly::empty());
        }
        if wave_width == 0 {
            return Some(schur::Assembly::invalid());
        }

        let mut merged = schur::Assembly::empty();
        let mut remaining_inverse_crosses = inverse_crosses;
        let mut remaining_cross_start = 0usize;
        for wave_start in (0..shard_count).step_by(wave_width) {
            let wave_end = (wave_start + wave_width).min(shard_count);
            let active_shards = wave_end - wave_start;
            let shard_ranges = &self.point_ranges[wave_start..=wave_end];
            let point_start = shard_ranges[0];
            let point_end = shard_ranges[active_shards];
            let cross_start = point_cross_offsets[point_start];
            let cross_end = point_cross_offsets[point_end];
            if cross_start != remaining_cross_start {
                return Some(schur::Assembly::invalid());
            }
            let wave_cross_count = cross_end - cross_start;
            let (wave_inverse_crosses, remaining) =
                remaining_inverse_crosses.split_at_mut(wave_cross_count);
            remaining_inverse_crosses = remaining;
            remaining_cross_start = cross_end;
            let wave_point_steps = match point_steps.as_mut() {
                Some(steps) => Some(PointStepCacheBuffers {
                    valid: &mut steps.valid[point_start..point_end],
                    inverse_gradients: &mut steps.inverse_gradients[point_start..point_end],
                    damping_diagonals: &mut steps.damping_diagonals[point_start..point_end],
                }),
                None => None,
            };
            let scratch = &mut self.scratch[..active_shards];
            let assemble_wave = || {
                assemble_shards(
                    context,
                    shard_ranges,
                    scratch,
                    &problem.points[point_start..point_end],
                    point_start,
                    wave_point_steps,
                    wave_inverse_crosses,
                    cross_start,
                    &mut point_hessians[point_start..point_end],
                )
            };
            // A one-worker executor must not enter Rayon once per logical
            // shard. Running the exact same shard kernel inline preserves the
            // same floating-point operation order while removing pool/task overhead.
            let wave_valid = if active_shards == 1 {
                assemble_wave()
            } else {
                matvec.install_parallel(assemble_wave)?
            };
            if !wave_valid {
                return Some(schur::Assembly::invalid());
            }

            // The order here is the numerical contract: it is independent of
            // how many shards the pool executed concurrently in this wave.
            for shard in scratch {
                let Some(assembly) = shard.assembly.take() else {
                    return Some(schur::Assembly::invalid());
                };
                merge_assembly(&mut merged, assembly);
                for (target, source) in target_blocks.iter_mut().zip(&shard.blocks) {
                    for (target, &source) in target.values.iter_mut().zip(&source.values) {
                        *target += source;
                    }
                }
                for (target, source) in target_edges.iter_mut().zip(&shard.edges) {
                    for (target, &source) in target.values.iter_mut().zip(&source.values) {
                        *target += source;
                    }
                }
                for (target, &source) in gradient.iter_mut().zip(&shard.gradient) {
                    *target += source;
                }
                for (target, &source) in full_gradient.iter_mut().zip(&shard.full_gradient) {
                    *target += source;
                }
                for (target, &source) in diagonal.iter_mut().zip(&shard.diagonal) {
                    *target += source;
                }
            }
        }

        merged.valid &= remaining_inverse_crosses.is_empty()
            && target_blocks
                .iter()
                .all(|block| block.values.iter().all(|value| value.is_finite()))
            && target_edges
                .iter()
                .all(|edge| edge.values.iter().all(|value| value.is_finite()))
            && gradient.iter().all(|value| value.is_finite())
            && full_gradient.iter().all(|value| value.is_finite())
            && diagonal.iter().all(|value| value.is_finite());
        Some(merged)
    }

    /// Build the point-elimination part of the block-Jacobi preconditioner on
    /// the same fixed logical shards used by linearization.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_point_coupling<C: PointCrossStorage + ?Sized>(
        &mut self,
        matvec: &SelfCalibrationMatvec,
        point_hessians: &[PackedPointBlock],
        point_cross_offsets: &[usize],
        point_parameters: &[u32],
        parameter_blocks: &[usize],
        block_offsets: &[usize],
        inverse_crosses: &C,
        target: &mut [PackedBlock],
    ) -> bool {
        let point_count = point_hessians.len();
        if point_cross_offsets.len() != point_count + 1
            || point_cross_offsets.first().copied() != Some(0)
            || point_cross_offsets.last().copied() != Some(point_parameters.len())
            || inverse_crosses.rows(0..point_parameters.len()).len() != point_parameters.len()
            || self.point_ranges.first().copied() != Some(0)
            || self.point_ranges.last().copied() != Some(point_count)
            || self
                .scratch
                .iter()
                .any(|scratch| scratch.blocks.len() != target.len())
        {
            return false;
        }

        target.iter_mut().for_each(|block| block.values.fill(0.0));
        let shard_count = self.point_ranges.len().saturating_sub(1);
        let wave_width = self.scratch.len();
        if shard_count == 0 {
            return true;
        }
        if wave_width == 0 {
            return false;
        }

        for wave_start in (0..shard_count).step_by(wave_width) {
            let wave_end = (wave_start + wave_width).min(shard_count);
            let active_shards = wave_end - wave_start;
            let shard_ranges = &self.point_ranges[wave_start..=wave_end];
            let scratch = &mut self.scratch[..active_shards];
            let mut build_wave = || {
                build_point_coupling_shards(
                    shard_ranges,
                    scratch,
                    point_hessians,
                    point_cross_offsets,
                    point_parameters,
                    parameter_blocks,
                    block_offsets,
                    inverse_crosses,
                )
            };
            let wave_valid = if active_shards == 1 {
                build_wave()
            } else {
                matvec.install_parallel(build_wave).unwrap_or(false)
            };
            if !wave_valid {
                return false;
            }
            for shard in scratch {
                for (target, source) in target.iter_mut().zip(&shard.blocks) {
                    for (target, &source) in target.values.iter_mut().zip(&source.values) {
                        *target += source;
                    }
                }
            }
        }
        target
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.point_ranges.capacity() * std::mem::size_of::<usize>()
            + self
                .scratch
                .iter()
                .map(LinearizationScratch::retained_bytes)
                .sum::<usize>()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_point_coupling_shards<C: PointCrossStorage + ?Sized>(
    shard_ranges: &[usize],
    scratch: &mut [LinearizationScratch],
    point_hessians: &[PackedPointBlock],
    point_cross_offsets: &[usize],
    point_parameters: &[u32],
    parameter_blocks: &[usize],
    block_offsets: &[usize],
    inverse_crosses: &C,
) -> bool {
    if shard_ranges.len() != scratch.len() + 1 || scratch.is_empty() {
        return false;
    }
    if scratch.len() == 1 {
        let blocks = &mut scratch[0].blocks;
        blocks.iter_mut().for_each(|block| block.values.fill(0.0));
        return super::accumulate_point_coupling_range(
            point_hessians,
            point_cross_offsets,
            point_parameters,
            parameter_blocks,
            block_offsets,
            inverse_crosses,
            blocks,
            shard_ranges[0],
            shard_ranges[1],
        );
    }

    let middle = scratch.len() / 2;
    let (left_scratch, right_scratch) = scratch.split_at_mut(middle);
    let (left, right) = rayon::join(
        || {
            build_point_coupling_shards(
                &shard_ranges[..=middle],
                left_scratch,
                point_hessians,
                point_cross_offsets,
                point_parameters,
                parameter_blocks,
                block_offsets,
                inverse_crosses,
            )
        },
        || {
            build_point_coupling_shards(
                &shard_ranges[middle..],
                right_scratch,
                point_hessians,
                point_cross_offsets,
                point_parameters,
                parameter_blocks,
                block_offsets,
                inverse_crosses,
            )
        },
    );
    left && right
}

pub(super) fn estimated_worker_bytes(
    block_count: usize,
    edge_count: usize,
    dimension: usize,
) -> usize {
    block_count
        .saturating_mul(std::mem::size_of::<PackedBlock>())
        .saturating_add(edge_count.saturating_mul(std::mem::size_of::<PoseIntrinsicEdge>()))
        .saturating_add(dimension.saturating_mul(
            6 * std::mem::size_of::<f64>()
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<bool>(),
        ))
}

#[allow(clippy::too_many_arguments)]
fn assemble_shards(
    context: LinearizationContext<'_>,
    shard_ranges: &[usize],
    scratch: &mut [LinearizationScratch],
    points: &[[f64; 3]],
    point_start: usize,
    point_steps: Option<PointStepCacheBuffers<'_>>,
    inverse_crosses: PointCrossCacheMut<'_>,
    cross_start: usize,
    point_hessians: &mut [PackedPointBlock],
) -> bool {
    if shard_ranges.len() != scratch.len() + 1 {
        return false;
    }
    if scratch.len() == 1 {
        if shard_ranges[0] != point_start
            || shard_ranges[1] - shard_ranges[0] != points.len()
            || point_steps.as_ref().is_some_and(|steps| {
                steps.valid.len() != points.len()
                    || steps.inverse_gradients.len() != points.len()
                    || steps.damping_diagonals.len() != points.len()
            })
            || point_hessians.len() != points.len()
        {
            return false;
        }
        let LinearizationScratch {
            blocks,
            edges,
            gradient,
            full_gradient,
            diagonal,
            cross,
            active_parameters,
            parameter_marks,
            assembly,
        } = &mut scratch[0];
        let mut hessian = SelfCalibrationRetainedHessian {
            blocks,
            edges,
            pose_edges: context.pose_edges,
            parameter_blocks: context.parameter_blocks,
            block_offsets: context.block_offsets,
            camera_count: context.camera_count,
            valid: true,
        };
        let point_cache = PointCacheBuffers {
            point_steps,
            cross_offsets: context.point_cross_offsets,
            inverse_cross_values: inverse_crosses,
            point_offset: point_start,
            cross_offset: cross_start,
            damped_hessians: Some(point_hessians),
        };
        let mut result = schur::assemble_retained_range(
            context.problem.model,
            context.problem.poses,
            points,
            point_start,
            context.problem.fixed.points,
            context.problem.priors.points,
            context.policy.prior_sqrt_scale,
            context.policy.gauge,
            context.policy.loss,
            context.policy.noise,
            context.policy.damping,
            context.dimension,
            &mut hessian,
            gradient,
            full_gradient,
            diagonal,
            cross,
            active_parameters,
            parameter_marks,
            context.layout,
            point_cache,
        );
        result.valid &= hessian.is_finite();
        *assembly = Some(result);
        return true;
    }
    if scratch.is_empty() {
        return points.is_empty()
            && point_steps.is_none_or(|steps| {
                steps.valid.is_empty()
                    && steps.inverse_gradients.is_empty()
                    && steps.damping_diagonals.is_empty()
            })
            && inverse_crosses.is_empty()
            && point_hessians.is_empty();
    }

    let middle = scratch.len() / 2;
    let point_split = shard_ranges[middle];
    let Some(local_point_split) = point_split.checked_sub(point_start) else {
        return false;
    };
    let Some(&cross_split) = context.point_cross_offsets.get(point_split) else {
        return false;
    };
    let Some(local_cross_split) = cross_split.checked_sub(cross_start) else {
        return false;
    };
    if local_point_split > points.len() || local_cross_split > inverse_crosses.len() {
        return false;
    }
    let (left_scratch, right_scratch) = scratch.split_at_mut(middle);
    let (left_points, right_points) = points.split_at(local_point_split);
    let (left_steps, right_steps) = split_point_steps(point_steps, local_point_split);
    let (left_hessians, right_hessians) = point_hessians.split_at_mut(local_point_split);
    let (left_crosses, right_crosses) = inverse_crosses.split_at_mut(local_cross_split);
    let (left, right) = rayon::join(
        || {
            assemble_shards(
                context,
                &shard_ranges[..=middle],
                left_scratch,
                left_points,
                point_start,
                left_steps,
                left_crosses,
                cross_start,
                left_hessians,
            )
        },
        || {
            assemble_shards(
                context,
                &shard_ranges[middle..],
                right_scratch,
                right_points,
                point_split,
                right_steps,
                right_crosses,
                cross_split,
                right_hessians,
            )
        },
    );
    left && right
}

fn split_point_steps(
    point_steps: Option<PointStepCacheBuffers<'_>>,
    middle: usize,
) -> (
    Option<PointStepCacheBuffers<'_>>,
    Option<PointStepCacheBuffers<'_>>,
) {
    match point_steps {
        Some(PointStepCacheBuffers {
            valid,
            inverse_gradients,
            damping_diagonals,
        }) => {
            let (left_valid, right_valid) = valid.split_at_mut(middle);
            let (left_gradients, right_gradients) = inverse_gradients.split_at_mut(middle);
            let (left_diagonals, right_diagonals) = damping_diagonals.split_at_mut(middle);
            (
                Some(PointStepCacheBuffers {
                    valid: left_valid,
                    inverse_gradients: left_gradients,
                    damping_diagonals: left_diagonals,
                }),
                Some(PointStepCacheBuffers {
                    valid: right_valid,
                    inverse_gradients: right_gradients,
                    damping_diagonals: right_diagonals,
                }),
            )
        }
        None => (None, None),
    }
}

fn merge_assembly(target: &mut schur::Assembly, source: schur::Assembly) {
    target.valid &= source.valid;
    target.weak_point_count += source.weak_point_count;
    target.minimum_point_rcond = target.minimum_point_rcond.min(source.minimum_point_rcond);
    target.scaled_gradient_inf_norm = target
        .scaled_gradient_inf_norm
        .max(source.scaled_gradient_inf_norm);
    target.point_model_reduction += source.point_model_reduction;
    target.cost += source.cost;
    target.squared_residual += source.squared_residual;
}
