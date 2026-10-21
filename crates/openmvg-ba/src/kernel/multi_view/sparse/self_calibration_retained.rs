// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact retained Schur solve storage for radial self-calibration.

#[cfg(feature = "parallel")]
mod assembly;
mod operator;

#[cfg(feature = "parallel")]
use assembly::{SelfCalibrationLinearizationWorkspace, estimated_worker_bytes};
use operator::{SelfCalibrationMatvec, apply_reduced_system};

use crate::kernel::point_block::PackedPointBlock;

use super::super::schur::{self, HessianAccumulator, add_scaled_pair};
use super::api::{RetainedSchurMatvec, SparseBundleAdjustmentError};
use super::direct::add_sparse_symmetric;
use super::retained::{
    PackedCameraBlockFactor, PackedSymmetricCameraBlock, PointCrossRows, PointCrossStorage,
    RecomputedSchurContext, RejectedSchurSolve, SchurAttempt,
};
use super::retained::{auto_self_calibration_threads, deterministic_point_major_shards};
use super::workspace::{PointCrossCache, SparseWorkspaceCore};

const MAXIMUM_BLOCK_DIMENSION: usize = 6;
const PACKED_BLOCK_STORAGE: usize = MAXIMUM_BLOCK_DIMENSION * (MAXIMUM_BLOCK_DIMENSION + 1) / 2;
const MAXIMUM_EDGE_STORAGE: usize = 6 * 3;
const NO_EDGE: usize = usize::MAX;

type PackedBlock = PackedSymmetricCameraBlock<MAXIMUM_BLOCK_DIMENSION, PACKED_BLOCK_STORAGE>;
type BlockFactor = PackedCameraBlockFactor<MAXIMUM_BLOCK_DIMENSION, PACKED_BLOCK_STORAGE>;
#[derive(Clone, Copy)]
struct PoseIntrinsicEdge {
    pose_block: usize,
    intrinsic_block: usize,
    values: [f64; MAXIMUM_EDGE_STORAGE],
}

impl PoseIntrinsicEdge {
    fn new(pose_block: usize, intrinsic_block: usize) -> Self {
        Self {
            pose_block,
            intrinsic_block,
            values: [0.0; MAXIMUM_EDGE_STORAGE],
        }
    }

    #[inline]
    fn add(&mut self, pose_local: usize, intrinsic_local: usize, value: f64) {
        self.values[pose_local * 3 + intrinsic_local] += value;
    }

    /// Add `pose[0][i] * intrinsic[0][j]` and then `pose[1][i] * intrinsic[1][j]`
    /// at `(pose_start + i, intrinsic_start + j)`.
    #[inline]
    fn add_gram(
        &mut self,
        pose_start: usize,
        intrinsic_start: usize,
        pose: [&[f64]; 2],
        intrinsic: [&[f64]; 2],
    ) {
        let (pose_rows, _) = self.values.as_chunks_mut::<3>();
        for ((row, &first), &second) in pose_rows[pose_start..].iter_mut().zip(pose[0]).zip(pose[1])
        {
            add_scaled_pair(
                &mut row[intrinsic_start..],
                intrinsic[0],
                first,
                intrinsic[1],
                second,
            );
        }
    }
}

pub(super) struct SelfCalibrationRetainedHessian<'a> {
    blocks: &'a mut [PackedBlock],
    edges: &'a mut [PoseIntrinsicEdge],
    pose_edges: &'a [usize],
    parameter_blocks: &'a [usize],
    block_offsets: &'a [usize],
    camera_count: usize,
    valid: bool,
}

impl HessianAccumulator for SelfCalibrationRetainedHessian<'_> {
    fn clear(&mut self) {
        for block in &mut *self.blocks {
            block.values.fill(0.0);
        }
        for edge in &mut *self.edges {
            edge.values.fill(0.0);
        }
        self.valid = true;
    }

    #[inline]
    fn add_symmetric(&mut self, row: usize, column: usize, value: f64) {
        let Some((&row_block, &column_block)) = self
            .parameter_blocks
            .get(row)
            .zip(self.parameter_blocks.get(column))
        else {
            self.valid = false;
            return;
        };
        if row_block == column_block {
            let offset = self.block_offsets[row_block];
            self.blocks[row_block].add(row - offset, column - offset, value);
            return;
        }

        let (pose_block, pose_parameter, intrinsic_block, intrinsic_parameter) =
            if row_block < self.camera_count && column_block >= self.camera_count {
                (row_block, row, column_block, column)
            } else if column_block < self.camera_count && row_block >= self.camera_count {
                (column_block, column, row_block, row)
            } else {
                self.valid = false;
                return;
            };
        let edge_index = self.pose_edges[pose_block];
        let Some(edge) = self.edges.get_mut(edge_index) else {
            self.valid = false;
            return;
        };
        if edge.intrinsic_block != intrinsic_block {
            self.valid = false;
            return;
        }
        edge.add(
            pose_parameter - self.block_offsets[pose_block],
            intrinsic_parameter - self.block_offsets[intrinsic_block],
            value,
        );
    }

    #[inline]
    fn add_gram(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        lower: bool,
        left: [&[f64]; 2],
        right: [&[f64]; 2],
    ) {
        // Runs are consecutive, so their end points decide the owning block.
        let owner = |parameters: &[usize]| {
            let first = *parameters.first()?;
            let block = *self.parameter_blocks.get(first)?;
            (*self.parameter_blocks.get(*parameters.last()?)? == block)
                .then(|| (block, first - self.block_offsets[block]))
        };
        // A lower run is also its own column run.
        let row_run = owner(rows);
        let column_run = if lower { row_run } else { owner(columns) };
        let (Some((row_block, row_local)), Some((column_block, column_local))) =
            (row_run, column_run)
        else {
            return schur::add_gram_scalar(self, rows, columns, lower, left, right);
        };
        if row_block == column_block {
            self.blocks[row_block].add_gram(row_local, column_local, lower, left, right);
            return;
        }

        let pose_rows = row_block < self.camera_count && column_block >= self.camera_count;
        let (pose_block, intrinsic_block) = if pose_rows {
            (row_block, column_block)
        } else if column_block < self.camera_count && row_block >= self.camera_count {
            (column_block, row_block)
        } else {
            self.valid = false;
            return;
        };
        let Some(edge) = self
            .pose_edges
            .get(pose_block)
            .and_then(|&edge_index| self.edges.get_mut(edge_index))
            .filter(|edge| edge.intrinsic_block == intrinsic_block)
        else {
            self.valid = false;
            return;
        };
        if pose_rows {
            edge.add_gram(row_local, column_local, left, right);
        } else {
            edge.add_gram(column_local, row_local, right, left);
        }
    }

    fn is_finite(&self) -> bool {
        self.valid
            && self
                .blocks
                .iter()
                .all(|block| block.values.iter().all(|value| value.is_finite()))
            && self
                .edges
                .iter()
                .all(|edge| edge.values.iter().all(|value| value.is_finite()))
    }
}

pub(super) struct SelfCalibrationRetainedWorkspace {
    blocks: Vec<PackedBlock>,
    diagonal_point_coupling: Vec<PackedBlock>,
    factors: Vec<BlockFactor>,
    edges: Vec<PoseIntrinsicEdge>,
    pose_edges: Vec<usize>,
    point_hessians: Vec<PackedPointBlock>,
    point_parameters: Vec<u32>,
    parameter_blocks: Vec<usize>,
    block_offsets: Vec<usize>,
    block_dimensions: Vec<usize>,
    camera_count: usize,
    solution: Vec<f64>,
    residual: Vec<f64>,
    preconditioned: Vec<f64>,
    direction: Vec<f64>,
    product: Vec<f64>,
    maximum_iterations: usize,
    relative_tolerance: f64,
    requested_matvec: RetainedSchurMatvec,
    matvec: SelfCalibrationMatvec,
    #[cfg(feature = "parallel")]
    linearization: Option<SelfCalibrationLinearizationWorkspace>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn estimated_self_calibrating_retained_bytes(
    block_count: usize,
    active_edge_count: usize,
    point_count: usize,
    point_parameter_count: usize,
    dimension: usize,
    observation_count: usize,
    point_cache_bytes: usize,
    candidate_point_metrics_eligible: bool,
) -> usize {
    let base = block_count
        .saturating_mul(2 * std::mem::size_of::<PackedBlock>() + std::mem::size_of::<BlockFactor>())
        .saturating_add(active_edge_count.saturating_mul(std::mem::size_of::<PoseIntrinsicEdge>()))
        .saturating_add(block_count.saturating_mul(3 * std::mem::size_of::<usize>()))
        .saturating_add(point_count.saturating_mul(std::mem::size_of::<PackedPointBlock>()))
        .saturating_add(point_parameter_count.saturating_mul(std::mem::size_of::<u32>()))
        .saturating_add(
            dimension.saturating_mul(5 * std::mem::size_of::<f64>() + std::mem::size_of::<usize>()),
        )
        .saturating_add(point_cache_bytes);
    let worker_bytes = dimension.saturating_mul(std::mem::size_of::<f64>());
    let parallel =
        auto_self_calibration_threads(observation_count, worker_bytes).map_or(0, |threads| {
            let shards = deterministic_point_major_shards(observation_count, worker_bytes)
                .unwrap_or(threads);
            let matvec = threads
                .saturating_mul(worker_bytes)
                .saturating_add((shards + 1) * std::mem::size_of::<usize>());
            #[cfg(feature = "parallel")]
            let linearization = {
                let worker_bytes =
                    estimated_worker_bytes(block_count, active_edge_count, dimension);
                deterministic_point_major_shards(observation_count, worker_bytes).map_or(
                    0,
                    |linearization_shards| {
                        threads
                            .min(linearization_shards)
                            .saturating_mul(worker_bytes)
                            .saturating_add(
                                (linearization_shards + 1)
                                    .saturating_mul(std::mem::size_of::<usize>()),
                            )
                    },
                )
            };
            #[cfg(not(feature = "parallel"))]
            let linearization = 0;
            let candidate_point_metrics = if candidate_point_metrics_eligible {
                point_count.saturating_mul(std::mem::size_of::<[f64; 3]>())
            } else {
                0
            };
            matvec
                .saturating_add(linearization)
                .saturating_add(candidate_point_metrics)
        });
    base.saturating_add(parallel)
}

impl SelfCalibrationRetainedWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_new(
        camera_count: usize,
        camera_to_group: &[usize],
        camera_offsets: &[usize],
        camera_dimensions: &[usize],
        intrinsic_offsets: &[usize],
        intrinsic_dimensions: &[usize],
        point_offsets: &[usize],
        point_parameters: Vec<u32>,
        observation_count: usize,
        matvec: RetainedSchurMatvec,
        maximum_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let mut block_offsets = Vec::with_capacity(camera_count + intrinsic_offsets.len());
        block_offsets.extend_from_slice(camera_offsets);
        block_offsets.extend_from_slice(intrinsic_offsets);
        let mut block_dimensions = Vec::with_capacity(camera_count + intrinsic_dimensions.len());
        block_dimensions.extend_from_slice(camera_dimensions);
        block_dimensions.extend_from_slice(intrinsic_dimensions);
        let dimension = block_offsets
            .iter()
            .zip(&block_dimensions)
            .map(|(&offset, &dimension)| offset + dimension)
            .max()
            .unwrap_or(0);
        let mut parameter_blocks = vec![0; dimension];
        for (block, (&offset, &block_dimension)) in
            block_offsets.iter().zip(&block_dimensions).enumerate()
        {
            parameter_blocks[offset..offset + block_dimension].fill(block);
        }

        let mut edges = Vec::with_capacity(camera_count);
        let mut pose_edges = vec![NO_EDGE; camera_count];
        for camera in 0..camera_count {
            let group = camera_to_group[camera];
            if camera_dimensions[camera] == 0 || intrinsic_dimensions[group] == 0 {
                continue;
            }
            pose_edges[camera] = edges.len();
            edges.push(PoseIntrinsicEdge::new(camera, camera_count + group));
        }
        let matvec_workspace =
            SelfCalibrationMatvec::try_new(matvec, observation_count, point_offsets, dimension)?;
        #[cfg(feature = "parallel")]
        let linearization = matvec_workspace
            .parallel_threads()
            .map(|threads| {
                SelfCalibrationLinearizationWorkspace::try_new(
                    threads,
                    observation_count,
                    point_offsets,
                    block_offsets.len(),
                    &edges,
                    dimension,
                )
            })
            .transpose()?
            .flatten();
        let stored_point_parameters = if matvec == RetainedSchurMatvec::Recomputed {
            Vec::new()
        } else {
            point_parameters
        };
        Ok(Self {
            blocks: vec![PackedBlock::default(); block_offsets.len()],
            diagonal_point_coupling: vec![PackedBlock::default(); block_offsets.len()],
            factors: vec![BlockFactor::default(); block_offsets.len()],
            edges,
            pose_edges,
            point_hessians: vec![PackedPointBlock::default(); point_offsets.len() - 1],
            point_parameters: stored_point_parameters,
            parameter_blocks,
            block_offsets,
            block_dimensions,
            camera_count,
            solution: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            preconditioned: vec![0.0; dimension],
            direction: vec![0.0; dimension],
            product: vec![0.0; dimension],
            maximum_iterations,
            relative_tolerance,
            requested_matvec: matvec,
            matvec: matvec_workspace,
            #[cfg(feature = "parallel")]
            linearization,
        })
    }

    pub(super) fn assembly_buffers(
        &mut self,
    ) -> (SelfCalibrationRetainedHessian<'_>, &mut [PackedPointBlock]) {
        let Self {
            blocks,
            edges,
            pose_edges,
            point_hessians,
            parameter_blocks,
            block_offsets,
            camera_count,
            ..
        } = self;
        (
            SelfCalibrationRetainedHessian {
                blocks,
                edges,
                pose_edges,
                parameter_blocks,
                block_offsets,
                camera_count: *camera_count,
                valid: true,
            },
            point_hessians,
        )
    }

    /// Assemble point contributions on fixed topology shards and merge them
    /// in a worker-count-independent order. `None` selects the serial path.
    #[cfg(feature = "parallel")]
    pub(super) fn assemble_parallel(
        &mut self,
        problem: schur::BaProblemView<'_>,
        policy: schur::LinearizationPolicy<'_>,
        sparse: &mut SparseWorkspaceCore,
    ) -> Option<schur::Assembly> {
        let Self {
            blocks,
            edges,
            pose_edges,
            point_hessians,
            parameter_blocks,
            block_offsets,
            camera_count,
            matvec,
            linearization,
            ..
        } = self;
        let linearization = linearization.as_mut()?;
        let SparseWorkspaceCore {
            layout: reduced_layout,
            gradient,
            full_gradient,
            diagonal,
            point_cache_valid,
            point_inverse_gradients,
            point_damping_diagonals,
            point_cross_offsets,
            point_inverse_cross_values,
            ..
        } = sparse;
        let layout = schur::VariableLayout {
            offsets: reduced_layout.camera_offsets(),
            pose_dimensions: reduced_layout.camera_dimensions(),
            intrinsic_offsets: reduced_layout.intrinsic_offsets(),
            intrinsic_dimensions: reduced_layout.intrinsic_dimensions(),
            pose_masks: &[],
            pose_variables: super::super::CameraPoseVariables::RotationAndTranslation,
        };
        let point_steps = (!point_cache_valid.is_empty()).then_some(schur::PointStepCacheBuffers {
            valid: point_cache_valid,
            inverse_gradients: point_inverse_gradients,
            damping_diagonals: point_damping_diagonals,
        });
        let mut assembly = linearization.assemble_points(
            matvec,
            problem,
            policy,
            layout,
            pose_edges,
            parameter_blocks,
            block_offsets,
            *camera_count,
            point_cross_offsets,
            point_steps,
            point_inverse_cross_values.as_mut(),
            point_hessians,
            blocks,
            edges,
            gradient,
            full_gradient,
            diagonal,
        )?;
        let mut hessian = SelfCalibrationRetainedHessian {
            blocks,
            edges,
            pose_edges,
            parameter_blocks,
            block_offsets,
            camera_count: *camera_count,
            valid: true,
        };
        schur::accumulate_camera_priors(
            problem.model,
            problem.poses,
            policy.gauge,
            layout,
            problem.priors,
            policy.prior_sqrt_scale,
            &mut hessian,
            gradient,
            full_gradient,
            diagonal,
            &mut assembly,
        );
        Some(schur::finalize_assembly(
            &mut hessian,
            gradient,
            full_gradient,
            diagonal,
            policy.damping,
            reduced_layout.dimension(),
            layout,
            assembly,
        ))
    }

    pub(super) fn solve(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        relative_tolerance: Option<f64>,
    ) -> SchurAttempt {
        let is_confirmation = relative_tolerance.is_some();
        let relative_tolerance = relative_tolerance.unwrap_or(self.relative_tolerance);
        let factored = self.factor_preconditioner(context, sparse);
        if !factored {
            return rejected(0, None);
        }
        // Preserve the dimension bound for ordinary solves. Strict
        // confirmation may need extra iterations after finite-precision loss
        // of conjugacy, but never exceeds the configured finite guard.
        let limit = if is_confirmation {
            self.maximum_iterations
        } else {
            self.maximum_iterations.min(self.solution.len())
        };
        let matvec = &mut self.matvec;
        let blocks = &self.blocks;
        let edges = &self.edges;
        let block_offsets = &self.block_offsets;
        let block_dimensions = &self.block_dimensions;
        let point_hessians = &self.point_hessians;
        let point_parameters = &self.point_parameters;
        let factors = &self.factors;
        super::pcg::solve(
            &sparse.gradient,
            super::pcg::PcgBuffers {
                solution: &mut self.solution,
                residual: &mut self.residual,
                preconditioned: &mut self.preconditioned,
                direction: &mut self.direction,
                product: &mut self.product,
            },
            super::pcg::PcgOptions {
                maximum_iterations: limit,
                relative_tolerance,
                require_bounded_residual: true,
            },
            |input, output| {
                apply_reduced_system(
                    matvec,
                    context,
                    sparse,
                    blocks,
                    edges,
                    block_offsets,
                    block_dimensions,
                    point_hessians,
                    point_parameters,
                    input,
                    output,
                );
            },
            |residual, preconditioned| {
                solve_blocks(
                    factors,
                    block_offsets,
                    block_dimensions,
                    residual,
                    preconditioned,
                );
            },
        )
    }

    pub(super) fn copy_step_to(&self, step: &mut [f64]) {
        step.copy_from_slice(&self.solution);
    }

    pub(super) fn predicted_reduction(
        &self,
        gradient: &[f64],
        point_model_reduction: f64,
        step: &[f64],
    ) -> Option<f64> {
        // Accepted PCG exits only after a confirming true-residual evaluation,
        // which leaves `product` equal to A*x for the returned solution. Reuse
        // that exact product instead of immediately applying the Schur operator
        // a second time for the quadratic model.
        let camera_reduction = -dot(gradient, step) - 0.5 * dot(step, &self.product);
        let predicted = point_model_reduction + camera_reduction;
        (predicted.is_finite() && predicted > 0.0).then_some(predicted)
    }

    pub(super) fn materialize_schur(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &mut SparseWorkspaceCore,
    ) -> bool {
        sparse.values.fill(0.0);
        for (block, (&offset, &dimension)) in self
            .blocks
            .iter()
            .zip(self.block_offsets.iter().zip(&self.block_dimensions))
        {
            for row in 0..dimension {
                for column in 0..=row {
                    if !add_sparse_symmetric(
                        sparse,
                        offset + row,
                        offset + column,
                        block.get(row, column),
                    ) {
                        return false;
                    }
                }
            }
        }
        for edge in &self.edges {
            let pose_offset = self.block_offsets[edge.pose_block];
            let intrinsic_offset = self.block_offsets[edge.intrinsic_block];
            let pose_dimension = self.block_dimensions[edge.pose_block];
            let intrinsic_dimension = self.block_dimensions[edge.intrinsic_block];
            for pose_local in 0..pose_dimension {
                for intrinsic_local in 0..intrinsic_dimension {
                    if !add_sparse_symmetric(
                        sparse,
                        pose_offset + pose_local,
                        intrinsic_offset + intrinsic_local,
                        edge.values[pose_local * 3 + intrinsic_local],
                    ) {
                        return false;
                    }
                }
            }
        }
        if matches!(
            sparse.point_inverse_cross_values,
            PointCrossCache::Recomputed { .. }
        ) {
            return self
                .matvec
                .materialize_recomputed(context, sparse, &self.point_hessians);
        }
        for point in 0..sparse.signature.point_count() {
            let range = sparse.point_cross_offsets[point]..sparse.point_cross_offsets[point + 1];
            for row in 0..range.len() {
                let row_cross = sparse.point_inverse_cross_values.get(range.start + row);
                let row_parameter = self.point_parameters[range.start + row] as usize;
                for column in 0..=row {
                    let column_cross = sparse.point_inverse_cross_values.get(range.start + column);
                    let column_parameter = self.point_parameters[range.start + column] as usize;
                    let value = -self.point_hessians[point].bilinear(&row_cross, &column_cross);
                    if !add_sparse_symmetric(sparse, row_parameter, column_parameter, value) {
                        return false;
                    }
                }
            }
        }
        sparse.values.iter().all(|value| value.is_finite())
    }

    #[cfg(feature = "unstable-internals")]
    pub(super) const fn requested_matvec(&self) -> RetainedSchurMatvec {
        self.requested_matvec
    }

    pub(super) const fn parallel_threads(&self) -> Option<usize> {
        self.matvec.parallel_threads()
    }

    pub(super) fn point_parameters(&self) -> &[u32] {
        &self.point_parameters
    }

    #[cfg(feature = "parallel")]
    pub(super) fn install_parallel<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> Option<R> {
        self.matvec.install_parallel(operation)
    }

    #[cfg(feature = "unstable-internals")]
    pub(super) const fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }

    pub(super) const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    pub(super) const fn uses_recomputed_cross_rows(&self) -> bool {
        matches!(self.requested_matvec, RetainedSchurMatvec::Recomputed)
    }

    pub(super) fn prepare_recomputed_point_blocks(&mut self, fixed_points: &[bool]) -> bool {
        if !self.uses_recomputed_cross_rows() {
            return true;
        }
        self.point_hessians
            .iter_mut()
            .enumerate()
            .all(|(point, hessian)| {
                (!fixed_points.is_empty() && fixed_points[point]) || hessian.invert_in_place()
            })
    }

    pub(super) fn retained_bytes(&self) -> usize {
        let bytes = (self.blocks.capacity() + self.diagonal_point_coupling.capacity())
            * std::mem::size_of::<PackedBlock>()
            + self.factors.capacity() * std::mem::size_of::<BlockFactor>()
            + self.edges.capacity() * std::mem::size_of::<PoseIntrinsicEdge>()
            + self.pose_edges.capacity() * std::mem::size_of::<usize>()
            + self.point_hessians.capacity() * std::mem::size_of::<PackedPointBlock>()
            + self.point_parameters.capacity() * std::mem::size_of::<u32>()
            + (self.parameter_blocks.capacity()
                + self.block_offsets.capacity()
                + self.block_dimensions.capacity())
                * std::mem::size_of::<usize>()
            + (self.solution.capacity()
                + self.residual.capacity()
                + self.preconditioned.capacity()
                + self.direction.capacity()
                + self.product.capacity())
                * std::mem::size_of::<f64>()
            + self.matvec.retained_bytes();
        #[cfg(feature = "parallel")]
        let bytes = bytes
            + self
                .linearization
                .as_ref()
                .map_or(0, SelfCalibrationLinearizationWorkspace::retained_bytes);
        bytes
    }

    fn factor_preconditioner(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
    ) -> bool {
        for block in &mut self.diagonal_point_coupling {
            block.values.fill(0.0);
        }
        match &sparse.point_inverse_cross_values {
            PointCrossCache::Recomputed { .. } => {
                if !self.matvec.build_recomputed_diagonal(
                    context,
                    sparse,
                    &self.parameter_blocks,
                    &self.block_offsets,
                    &self.point_hessians,
                    &mut self.diagonal_point_coupling,
                ) {
                    return false;
                }
            }
            PointCrossCache::F64(crosses) => {
                if !self.build_retained_point_coupling(sparse, crosses.as_slice()) {
                    return false;
                }
            }
            PointCrossCache::F32(crosses) => {
                if !self.build_retained_point_coupling(sparse, crosses.as_slice()) {
                    return false;
                }
            }
        }
        for (block, factor) in self.factors.iter_mut().enumerate() {
            let dimension = self.block_dimensions[block];
            if dimension == 0 {
                continue;
            }
            let mut preconditioner = self.blocks[block];
            for (value, coupling) in preconditioner
                .values
                .iter_mut()
                .zip(&self.diagonal_point_coupling[block].values)
            {
                *value -= coupling;
            }
            if !factor.factor(&preconditioner, dimension) {
                return false;
            }
        }
        true
    }

    fn build_retained_point_coupling<C: PointCrossStorage + ?Sized>(
        &mut self,
        sparse: &SparseWorkspaceCore,
        crosses: &C,
    ) -> bool {
        #[cfg(feature = "parallel")]
        if let Some(linearization) = self.linearization.as_mut() {
            return linearization.build_point_coupling(
                &self.matvec,
                &self.point_hessians,
                &sparse.point_cross_offsets,
                &self.point_parameters,
                &self.parameter_blocks,
                &self.block_offsets,
                crosses,
                &mut self.diagonal_point_coupling,
            );
        }
        accumulate_point_coupling_range(
            &self.point_hessians,
            &sparse.point_cross_offsets,
            &self.point_parameters,
            &self.parameter_blocks,
            &self.block_offsets,
            crosses,
            &mut self.diagonal_point_coupling,
            0,
            sparse.signature.point_count(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_point_coupling_range<C: PointCrossStorage + ?Sized>(
    point_hessians: &[PackedPointBlock],
    point_cross_offsets: &[usize],
    point_parameters: &[u32],
    parameter_blocks: &[usize],
    block_offsets: &[usize],
    inverse_crosses: &C,
    output: &mut [PackedBlock],
    point_start: usize,
    point_end: usize,
) -> bool {
    for point in point_start..point_end {
        let range = point_cross_offsets[point]..point_cross_offsets[point + 1];
        let crosses = inverse_crosses.rows(range.clone());
        let mut local_start = 0;
        while local_start < crosses.len() {
            let block = parameter_blocks[point_parameters[range.start + local_start] as usize];
            let mut local_end = local_start + 1;
            while local_end < crosses.len()
                && parameter_blocks[point_parameters[range.start + local_end] as usize] == block
            {
                local_end += 1;
            }
            for row in local_start..local_end {
                let row_parameter = point_parameters[range.start + row] as usize;
                let row_local = row_parameter - block_offsets[block];
                let row_cross = crosses.row(row);
                for column in local_start..=row {
                    let column_parameter = point_parameters[range.start + column] as usize;
                    let column_local = column_parameter - block_offsets[block];
                    let column_cross = crosses.row(column);
                    let value = point_hessians[point].bilinear(&row_cross, &column_cross);
                    output[block].add(row_local, column_local, value);
                }
            }
            local_start = local_end;
        }
    }
    output
        .iter()
        .all(|block| block.values.iter().all(|value| value.is_finite()))
}

fn solve_blocks(
    factors: &[BlockFactor],
    offsets: &[usize],
    dimensions: &[usize],
    right: &[f64],
    solution: &mut [f64],
) {
    solution.fill(0.0);
    for (block, (&offset, &dimension)) in offsets.iter().zip(dimensions).enumerate() {
        if dimension > 0 {
            factors[block].solve(
                &right[offset..offset + dimension],
                &mut solution[offset..offset + dimension],
                dimension,
            );
        }
    }
}

fn rejected(terms: usize, relative_residual: Option<f64>) -> SchurAttempt {
    SchurAttempt::Rejected(RejectedSchurSolve {
        terms,
        relative_residual,
    })
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    #[test]
    fn retained_estimate_counts_parallel_candidate_metrics_when_eligible() {
        const POINT_COUNT: usize = 4_096;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test pool is constructible");
        let estimate = |eligible| {
            pool.install(|| {
                estimated_self_calibrating_retained_bytes(
                    32,
                    16,
                    POINT_COUNT,
                    POINT_COUNT * 12,
                    138,
                    65_536,
                    POINT_COUNT * 64,
                    eligible,
                )
            })
        };
        let without_metrics = estimate(false);
        let with_metrics = estimate(true);
        assert_eq!(
            with_metrics,
            without_metrics.saturating_add(POINT_COUNT * std::mem::size_of::<[f64; 3]>())
        );
    }
}
