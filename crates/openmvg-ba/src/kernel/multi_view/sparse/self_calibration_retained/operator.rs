// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced Schur operator and its fused, recomputed, and parallel execution policies.

#[cfg(test)]
mod tests;

use crate::kernel::multi_view::schur;
use crate::kernel::point_block::PackedPointBlock;

use super::super::api::{RetainedSchurMatvec, SparseBundleAdjustmentError};
use super::super::direct::add_sparse_symmetric;
#[cfg(feature = "parallel")]
use super::super::retained::try_build_retained_schur_pool;
use super::super::retained::{
    PointCrossRows, PointCrossStorage, RecomputedSchurContext, RetainedMatvecPlan,
    RetainedProblemKind, balanced_contiguous_ranges,
};
use super::super::workspace::{PointCrossCache, SparseWorkspaceCore};
use super::{PackedBlock, PoseIntrinsicEdge};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_reduced_system(
    matvec: &mut SelfCalibrationMatvec,
    context: RecomputedSchurContext<'_>,
    sparse: &SparseWorkspaceCore,
    blocks: &[PackedBlock],
    edges: &[PoseIntrinsicEdge],
    block_offsets: &[usize],
    block_dimensions: &[usize],
    point_hessians: &[PackedPointBlock],
    point_parameters: &[u32],
    input: &[f64],
    output: &mut [f64],
) {
    match &sparse.point_inverse_cross_values {
        PointCrossCache::F64(crosses) => matvec.apply(
            blocks,
            edges,
            block_offsets,
            block_dimensions,
            point_hessians,
            &sparse.point_cross_offsets,
            point_parameters,
            crosses.as_slice(),
            input,
            output,
        ),
        PointCrossCache::F32(crosses) => matvec.apply(
            blocks,
            edges,
            block_offsets,
            block_dimensions,
            point_hessians,
            &sparse.point_cross_offsets,
            point_parameters,
            crosses.as_slice(),
            input,
            output,
        ),
        PointCrossCache::Recomputed { .. } => matvec.apply_recomputed(
            context,
            sparse,
            blocks,
            edges,
            block_offsets,
            block_dimensions,
            point_hessians,
            input,
            output,
        ),
    }
}

pub(super) enum SelfCalibrationMatvec {
    Fused,
    Recomputed(RecomputedMatvecWorkspace),
    PointMajor(PointMajorWorkspace),
}

pub(super) struct RecomputedMatvecWorkspace {
    cross: Vec<f64>,
    active_parameters: Vec<usize>,
    parameter_marks: Vec<bool>,
}

impl RecomputedMatvecWorkspace {
    fn new(dimension: usize) -> Result<Self, SparseBundleAdjustmentError> {
        let cross_len = dimension
            .checked_mul(3)
            .ok_or(SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
        Ok(Self {
            cross: vec![0.0; cross_len],
            active_parameters: vec![0; dimension],
            parameter_marks: vec![false; dimension],
        })
    }

    fn retained_bytes(&self) -> usize {
        self.cross.capacity() * std::mem::size_of::<f64>()
            + self.active_parameters.capacity() * std::mem::size_of::<usize>()
            + self.parameter_marks.capacity() * std::mem::size_of::<bool>()
    }

    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        blocks: &[PackedBlock],
        edges: &[PoseIntrinsicEdge],
        block_offsets: &[usize],
        block_dimensions: &[usize],
        point_inverses: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
    ) -> bool {
        apply_base(
            blocks,
            edges,
            block_offsets,
            block_dimensions,
            input,
            output,
        );
        for (point, inverse) in point_inverses.iter().enumerate() {
            if !sparse.signature.fixed_points().is_empty() && sparse.signature.fixed_points()[point]
            {
                continue;
            }
            let Some(active_count) = self.recompute_point(context, sparse, point) else {
                output.fill(f64::NAN);
                return false;
            };
            let mut sum = [0.0; 3];
            for &parameter in &self.active_parameters[..active_count] {
                let row = schur::cross_row(&self.cross, parameter);
                for coordinate in 0..3 {
                    sum[coordinate] += row[coordinate] * input[parameter];
                }
            }
            let inverse_sum = inverse.multiply(&sum);
            for &parameter in &self.active_parameters[..active_count] {
                output[parameter] -= dot3(&schur::cross_row(&self.cross, parameter), &inverse_sum);
            }
        }
        output.iter().all(|value| value.is_finite())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_diagonal(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        parameter_blocks: &[usize],
        block_offsets: &[usize],
        point_inverses: &[PackedPointBlock],
        output: &mut [PackedBlock],
    ) -> bool {
        for (point, inverse) in point_inverses.iter().enumerate() {
            if !sparse.signature.fixed_points().is_empty() && sparse.signature.fixed_points()[point]
            {
                continue;
            }
            let Some(active_count) = self.recompute_point(context, sparse, point) else {
                return false;
            };
            for (row_index, &row_parameter) in
                self.active_parameters[..active_count].iter().enumerate()
            {
                let block = parameter_blocks[row_parameter];
                let row_local = row_parameter - block_offsets[block];
                let row = schur::cross_row(&self.cross, row_parameter);
                for &column_parameter in &self.active_parameters[..=row_index] {
                    if parameter_blocks[column_parameter] != block {
                        continue;
                    }
                    let column_local = column_parameter - block_offsets[block];
                    let column = schur::cross_row(&self.cross, column_parameter);
                    let value = inverse.bilinear(&row, &column);
                    output[block].add(row_local, column_local, value);
                }
            }
        }
        output
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
    }

    fn materialize(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &mut SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
    ) -> bool {
        for (point, inverse) in point_inverses.iter().enumerate() {
            if !sparse.signature.fixed_points().is_empty() && sparse.signature.fixed_points()[point]
            {
                continue;
            }
            let Some(active_count) = self.recompute_point(context, sparse, point) else {
                return false;
            };
            for (row_index, &row_parameter) in
                self.active_parameters[..active_count].iter().enumerate()
            {
                let row = schur::cross_row(&self.cross, row_parameter);
                for &column_parameter in &self.active_parameters[..=row_index] {
                    let column = schur::cross_row(&self.cross, column_parameter);
                    let value = -inverse.bilinear(&row, &column);
                    if !add_sparse_symmetric(sparse, row_parameter, column_parameter, value) {
                        return false;
                    }
                }
            }
        }
        sparse.values.iter().all(|value| value.is_finite())
    }

    fn recompute_point(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point: usize,
    ) -> Option<usize> {
        let layout = schur::VariableLayout {
            offsets: sparse.layout.camera_offsets(),
            pose_dimensions: sparse.layout.camera_dimensions(),
            intrinsic_offsets: sparse.layout.intrinsic_offsets(),
            intrinsic_dimensions: sparse.layout.intrinsic_dimensions(),
            pose_masks: &[],
            pose_variables: crate::kernel::multi_view::CameraPoseVariables::RotationAndTranslation,
        };
        schur::recompute_point_cross_rows(
            context.model,
            context.poses,
            &context.points[point],
            point,
            context.gauge,
            context.loss,
            context.noise,
            layout,
            &mut self.cross,
            &mut self.active_parameters,
            &mut self.parameter_marks,
        )
    }
}

impl SelfCalibrationMatvec {
    pub(super) fn try_new(
        policy: RetainedSchurMatvec,
        observation_count: usize,
        point_offsets: &[usize],
        dimension: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let worker_bytes = dimension.saturating_mul(std::mem::size_of::<f64>());
        match RetainedMatvecPlan::compile(
            policy,
            RetainedProblemKind::SelfCalibrating,
            observation_count,
            worker_bytes,
        )? {
            RetainedMatvecPlan::Fused => Ok(Self::Fused),
            RetainedMatvecPlan::Recomputed => {
                Ok(Self::Recomputed(RecomputedMatvecWorkspace::new(dimension)?))
            }
            RetainedMatvecPlan::PointMajorParallel {
                threads,
                logical_shards,
            } => Ok(Self::PointMajor(PointMajorWorkspace::try_new(
                threads,
                logical_shards,
                point_offsets,
                dimension,
            )?)),
            RetainedMatvecPlan::RecomputedParallel { .. } => {
                Err(SparseBundleAdjustmentError::RecomputedSchurUnsupportedForSelfCalibration)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply<C: PointCrossStorage + ?Sized>(
        &mut self,
        blocks: &[PackedBlock],
        edges: &[PoseIntrinsicEdge],
        block_offsets: &[usize],
        block_dimensions: &[usize],
        point_hessians: &[PackedPointBlock],
        point_offsets: &[usize],
        point_parameters: &[u32],
        inverse_crosses: &C,
        input: &[f64],
        output: &mut [f64],
    ) {
        apply_base(
            blocks,
            edges,
            block_offsets,
            block_dimensions,
            input,
            output,
        );
        match self {
            Self::Fused => accumulate_point_range::<-1, _>(
                point_hessians,
                point_offsets,
                point_parameters,
                inverse_crosses,
                input,
                output,
                0,
                point_hessians.len(),
            ),
            Self::PointMajor(workspace) => workspace.apply(
                point_hessians,
                point_offsets,
                point_parameters,
                inverse_crosses,
                input,
                output,
            ),
            // Dispatch routes recomputed storage through `apply_recomputed`.
            #[allow(clippy::unreachable)]
            Self::Recomputed(_) => {
                unreachable!("recomputed cross rows use the analytic recomputation path")
            }
        }
    }

    pub(super) const fn parallel_threads(&self) -> Option<usize> {
        match self {
            Self::Fused => None,
            Self::Recomputed(_) => None,
            Self::PointMajor(workspace) => Some(workspace.threads),
        }
    }

    #[cfg(feature = "parallel")]
    pub(super) fn install_parallel<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> Option<R> {
        match self {
            Self::PointMajor(workspace) => Some(workspace.pool.install(operation)),
            Self::Fused | Self::Recomputed(_) => None,
        }
    }

    pub(super) fn retained_bytes(&self) -> usize {
        match self {
            Self::Fused => 0,
            Self::Recomputed(workspace) => workspace.retained_bytes(),
            Self::PointMajor(workspace) => workspace.retained_bytes(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_recomputed(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        blocks: &[PackedBlock],
        edges: &[PoseIntrinsicEdge],
        block_offsets: &[usize],
        block_dimensions: &[usize],
        point_inverses: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
    ) {
        // Construction and dispatch keep the operator paired with its cache policy.
        #[allow(clippy::unreachable)]
        let Self::Recomputed(workspace) = self else {
            unreachable!("recomputed point cache requires the recomputed operator")
        };
        workspace.apply(
            context,
            sparse,
            blocks,
            edges,
            block_offsets,
            block_dimensions,
            point_inverses,
            input,
            output,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_recomputed_diagonal(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        parameter_blocks: &[usize],
        block_offsets: &[usize],
        point_inverses: &[PackedPointBlock],
        output: &mut [PackedBlock],
    ) -> bool {
        let Self::Recomputed(workspace) = self else {
            return false;
        };
        workspace.build_diagonal(
            context,
            sparse,
            parameter_blocks,
            block_offsets,
            point_inverses,
            output,
        )
    }

    pub(super) fn materialize_recomputed(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &mut SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
    ) -> bool {
        let Self::Recomputed(workspace) = self else {
            return false;
        };
        workspace.materialize(context, sparse, point_inverses)
    }
}

pub(super) struct PointMajorWorkspace {
    threads: usize,
    dimension: usize,
    point_ranges: Vec<usize>,
    partial_outputs: Vec<f64>,
    #[cfg(feature = "parallel")]
    pool: rayon::ThreadPool,
}

impl PointMajorWorkspace {
    fn try_new(
        threads: usize,
        logical_shards: usize,
        point_offsets: &[usize],
        dimension: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let point_ranges = balanced_contiguous_ranges(point_offsets, logical_shards);
        let active_shards = point_ranges.len().saturating_sub(1).min(threads);
        let partial_len = active_shards
            .checked_mul(dimension)
            .ok_or(SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
        Ok(Self {
            threads,
            dimension,
            point_ranges,
            partial_outputs: vec![0.0; partial_len],
            #[cfg(feature = "parallel")]
            pool: try_build_retained_schur_pool(threads)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn apply<C: PointCrossStorage + ?Sized>(
        &mut self,
        point_hessians: &[PackedPointBlock],
        point_offsets: &[usize],
        point_parameters: &[u32],
        inverse_crosses: &C,
        input: &[f64],
        output: &mut [f64],
    ) {
        let dimension = self.dimension;
        let shard_count = self.point_ranges.len().saturating_sub(1);
        let wave_width = shard_count.min(self.threads);
        if dimension == 0 || wave_width == 0 {
            return;
        }
        if self.threads == 1 {
            let partial = &mut self.partial_outputs[..dimension];
            for range in self.point_ranges.windows(2) {
                partial.fill(0.0);
                accumulate_point_range::<1, _>(
                    point_hessians,
                    point_offsets,
                    point_parameters,
                    inverse_crosses,
                    input,
                    partial,
                    range[0],
                    range[1],
                );
                for (target, &coupling) in output.iter_mut().zip(&*partial) {
                    *target -= coupling;
                }
            }
            return;
        }
        #[cfg(feature = "parallel")]
        self.pool.install(|| {
            for wave_start in (0..shard_count).step_by(wave_width) {
                let wave_end = (wave_start + wave_width).min(shard_count);
                let active_shards = wave_end - wave_start;
                let ranges = &self.point_ranges[wave_start..=wave_end];
                let partials = &mut self.partial_outputs[..active_shards * dimension];
                ranges
                    .par_windows(2)
                    .zip(partials.par_chunks_mut(dimension))
                    .for_each(|(range, partial)| {
                        partial.fill(0.0);
                        accumulate_point_range::<1, _>(
                            point_hessians,
                            point_offsets,
                            point_parameters,
                            inverse_crosses,
                            input,
                            partial,
                            range[0],
                            range[1],
                        );
                    });
                for partial in partials.chunks(dimension) {
                    for (target, &coupling) in output.iter_mut().zip(partial) {
                        *target -= coupling;
                    }
                }
            }
        });
        #[cfg(not(feature = "parallel"))]
        for wave_start in (0..shard_count).step_by(wave_width) {
            let wave_end = (wave_start + wave_width).min(shard_count);
            let active_shards = wave_end - wave_start;
            let ranges = &self.point_ranges[wave_start..=wave_end];
            let partials = &mut self.partial_outputs[..active_shards * dimension];
            std::thread::scope(|scope| {
                for (range, partial) in ranges.windows(2).zip(partials.chunks_mut(dimension)) {
                    scope.spawn(move || {
                        partial.fill(0.0);
                        accumulate_point_range::<1, _>(
                            point_hessians,
                            point_offsets,
                            point_parameters,
                            inverse_crosses,
                            input,
                            partial,
                            range[0],
                            range[1],
                        );
                    });
                }
            });
            for partial in partials.chunks(dimension) {
                for (target, &coupling) in output.iter_mut().zip(partial) {
                    *target -= coupling;
                }
            }
        }
    }

    fn retained_bytes(&self) -> usize {
        self.point_ranges.capacity() * std::mem::size_of::<usize>()
            + self.partial_outputs.capacity() * std::mem::size_of::<f64>()
    }
}

fn apply_base(
    blocks: &[PackedBlock],
    edges: &[PoseIntrinsicEdge],
    offsets: &[usize],
    dimensions: &[usize],
    input: &[f64],
    output: &mut [f64],
) {
    output.fill(0.0);
    for (block, (&offset, &dimension)) in blocks.iter().zip(offsets.iter().zip(dimensions)) {
        block.multiply_add(
            &input[offset..offset + dimension],
            &mut output[offset..offset + dimension],
            dimension,
        );
    }
    for edge in edges {
        let pose_offset = offsets[edge.pose_block];
        let intrinsic_offset = offsets[edge.intrinsic_block];
        let pose_dimension = dimensions[edge.pose_block];
        let intrinsic_dimension = dimensions[edge.intrinsic_block];
        for pose_local in 0..pose_dimension {
            for intrinsic_local in 0..intrinsic_dimension {
                let value = edge.values[pose_local * 3 + intrinsic_local];
                output[pose_offset + pose_local] +=
                    value * input[intrinsic_offset + intrinsic_local];
                output[intrinsic_offset + intrinsic_local] +=
                    value * input[pose_offset + pose_local];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_point_range<const SIGN: i8, C: PointCrossStorage + ?Sized>(
    point_hessians: &[PackedPointBlock],
    point_offsets: &[usize],
    point_parameters: &[u32],
    inverse_crosses: &C,
    input: &[f64],
    output: &mut [f64],
    point_start: usize,
    point_end: usize,
) {
    debug_assert!(SIGN == -1 || SIGN == 1);
    for point in point_start..point_end {
        let range = point_offsets[point]..point_offsets[point + 1];
        let crosses = inverse_crosses.rows(range.clone());
        let mut sum = [0.0; 3];
        for (local, cross) in range.clone().enumerate() {
            let parameter = point_parameters[cross] as usize;
            let value = input[parameter];
            let inverse_cross = crosses.row(local);
            sum[0] += inverse_cross[0] * value;
            sum[1] += inverse_cross[1] * value;
            sum[2] += inverse_cross[2] * value;
        }
        let weighted = point_hessians[point].multiply(&sum);
        for (local, cross) in range.enumerate() {
            let coupling = dot3(&crosses.row(local), &weighted);
            if SIGN < 0 {
                output[point_parameters[cross] as usize] -= coupling;
            } else {
                output[point_parameters[cross] as usize] += coupling;
            }
        }
    }
}

#[inline(always)]
fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
