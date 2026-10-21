// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic direct-Schur point assembly and column-owned materialization.

#[cfg(feature = "parallel")]
use super::super::schur::{
    self, BaProblemView, HessianAccumulator, LinearizationPolicy, PointCacheBuffers,
    PointCrossCacheMut, PointStepCacheBuffers, VariableLayout, add_scaled_pair,
};
use super::api::SparseBundleAdjustmentError;
#[cfg(feature = "parallel")]
use super::direct_assembly_policy::{
    LOGICAL_POINT_SHARD_OBSERVATIONS, MAX_REDUCED_BLOCK_DIMENSION, MAXIMUM_LOGICAL_POINT_SHARDS,
};
pub(super) use super::direct_assembly_policy::{
    auto_direct_assembly_threads, estimated_calibrated_auxiliary_bytes,
    estimated_indexed_auxiliary_bytes,
};
use super::direct_materialization::CrossRow;
#[cfg(not(feature = "parallel"))]
use super::direct_materialization::materialize_block_range;
#[cfg(feature = "parallel")]
use super::direct_materialization::materialize_partitions;
#[cfg(feature = "parallel")]
use super::retained::PackedSymmetricCameraBlock;
#[cfg(feature = "parallel")]
use super::retained::balanced_contiguous_ranges;
#[cfg(feature = "parallel")]
use super::workspace::PointCrossCache;
use super::workspace::SparseScatter;
use crate::kernel::point_block::PackedPointBlock;

mod index;

#[cfg(feature = "parallel")]
const PACKED_REDUCED_BLOCK_STORAGE: usize =
    MAX_REDUCED_BLOCK_DIMENSION * (MAX_REDUCED_BLOCK_DIMENSION + 1) / 2;
#[cfg(feature = "parallel")]
const PACKED_REDUCED_EDGE_STORAGE: usize =
    MAX_REDUCED_BLOCK_DIMENSION * MAX_REDUCED_BLOCK_DIMENSION;

#[cfg(feature = "parallel")]
type PackedReducedBaseBlock =
    PackedSymmetricCameraBlock<MAX_REDUCED_BLOCK_DIMENSION, PACKED_REDUCED_BLOCK_STORAGE>;

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct PackedReducedBaseEdge {
    values: [f64; PACKED_REDUCED_EDGE_STORAGE],
}

#[cfg(feature = "parallel")]
impl Default for PackedReducedBaseEdge {
    fn default() -> Self {
        Self {
            values: [0.0; PACKED_REDUCED_EDGE_STORAGE],
        }
    }
}

#[cfg(feature = "parallel")]
impl PackedReducedBaseEdge {
    fn add(&mut self, row: usize, column: usize, value: f64) {
        self.values[row * MAX_REDUCED_BLOCK_DIMENSION + column] += value;
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * MAX_REDUCED_BLOCK_DIMENSION + column]
    }

    /// Add `left[0][i] * right[0][j]` and then `left[1][i] * right[1][j]` at
    /// `(row + i, column + j)`.
    fn add_gram(&mut self, row: usize, column: usize, left: [&[f64]; 2], right: [&[f64]; 2]) {
        let (rows, _) = self.values.as_chunks_mut::<MAX_REDUCED_BLOCK_DIMENSION>();
        for ((target, &first), &second) in rows[row..].iter_mut().zip(left[0]).zip(left[1]) {
            add_scaled_pair(&mut target[column..], right[0], first, right[1], second);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PointCrossRef {
    pub(super) point: u32,
    pub(super) cross_offset: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PointBlockRef {
    pub(super) block: u32,
    pub(super) cross_offset: u32,
}

#[cfg(feature = "parallel")]
struct DirectPointAssemblyScratch {
    reduced_blocks: Vec<PackedReducedBaseBlock>,
    reduced_block_marks: Vec<bool>,
    touched_reduced_blocks: Vec<usize>,
    base_edges: Vec<PackedReducedBaseEdge>,
    base_edge_marks: Vec<bool>,
    touched_base_edges: Vec<usize>,
    gradient: Vec<f64>,
    full_gradient: Vec<f64>,
    diagonal: Vec<f64>,
    cross: Vec<f64>,
    active_parameters: Vec<usize>,
    parameter_marks: Vec<bool>,
    assembly: Option<schur::Assembly>,
}

#[cfg(feature = "parallel")]
impl DirectPointAssemblyScratch {
    fn new(
        reduced_block_count: usize,
        base_edge_count: usize,
        dimension: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let cross_len = dimension
            .checked_mul(3)
            .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
        Ok(Self {
            reduced_blocks: vec![PackedReducedBaseBlock::default(); reduced_block_count],
            reduced_block_marks: vec![false; reduced_block_count],
            touched_reduced_blocks: Vec::with_capacity(reduced_block_count),
            base_edges: vec![PackedReducedBaseEdge::default(); base_edge_count],
            base_edge_marks: vec![false; base_edge_count],
            touched_base_edges: Vec::with_capacity(base_edge_count),
            gradient: vec![0.0; dimension],
            full_gradient: vec![0.0; dimension],
            diagonal: vec![0.0; dimension],
            cross: vec![0.0; cross_len],
            active_parameters: vec![0; dimension],
            parameter_marks: vec![false; dimension],
            assembly: None,
        })
    }

    fn capacity_bytes(&self) -> usize {
        self.reduced_blocks.capacity() * std::mem::size_of::<PackedReducedBaseBlock>()
            + self.reduced_block_marks.capacity() * std::mem::size_of::<bool>()
            + self.touched_reduced_blocks.capacity() * std::mem::size_of::<usize>()
            + self.base_edges.capacity() * std::mem::size_of::<PackedReducedBaseEdge>()
            + self.base_edge_marks.capacity() * std::mem::size_of::<bool>()
            + self.touched_base_edges.capacity() * std::mem::size_of::<usize>()
            + (self.gradient.capacity()
                + self.full_gradient.capacity()
                + self.diagonal.capacity()
                + self.cross.capacity())
                * std::mem::size_of::<f64>()
            + self.active_parameters.capacity() * std::mem::size_of::<usize>()
            + self.parameter_marks.capacity() * std::mem::size_of::<bool>()
    }
}

#[cfg(feature = "parallel")]
struct PackedReducedBaseHessian<'a> {
    reduced_blocks: &'a mut [PackedReducedBaseBlock],
    reduced_block_marks: &'a mut [bool],
    touched_reduced_blocks: &'a mut Vec<usize>,
    base_edge_blocks: &'a [(usize, usize)],
    base_edges: &'a mut [PackedReducedBaseEdge],
    base_edge_marks: &'a mut [bool],
    touched_base_edges: &'a mut Vec<usize>,
    scatter: &'a SparseScatter,
    valid: bool,
}

#[cfg(feature = "parallel")]
impl HessianAccumulator for PackedReducedBaseHessian<'_> {
    fn clear(&mut self) {
        for &block in self.touched_reduced_blocks.iter() {
            self.reduced_blocks[block].values.fill(0.0);
            self.reduced_block_marks[block] = false;
        }
        self.touched_reduced_blocks.clear();
        for &edge in self.touched_base_edges.iter() {
            self.base_edges[edge].values.fill(0.0);
            self.base_edge_marks[edge] = false;
        }
        self.touched_base_edges.clear();
        self.valid = true;
    }

    #[inline]
    fn add_symmetric(&mut self, row: usize, column: usize, value: f64) {
        let Some((row_block, row_local)) = self.scatter.parameter_location(row) else {
            self.valid = false;
            return;
        };
        let Some((column_block, column_local)) = self.scatter.parameter_location(column) else {
            self.valid = false;
            return;
        };
        if row_local >= MAX_REDUCED_BLOCK_DIMENSION || column_local >= MAX_REDUCED_BLOCK_DIMENSION {
            self.valid = false;
            return;
        }
        if row_block == column_block {
            if !self.reduced_block_marks[row_block] {
                self.reduced_block_marks[row_block] = true;
                self.touched_reduced_blocks.push(row_block);
            }
            self.reduced_blocks[row_block].add(row_local, column_local, value);
            return;
        }
        let (edge_blocks, edge_row, edge_column) = if row_block > column_block {
            ((row_block, column_block), row_local, column_local)
        } else {
            ((column_block, row_block), column_local, row_local)
        };
        let Ok(edge) = self.base_edge_blocks.binary_search(&edge_blocks) else {
            self.valid = false;
            return;
        };
        if !self.base_edge_marks[edge] {
            self.base_edge_marks[edge] = true;
            self.touched_base_edges.push(edge);
        }
        self.base_edges[edge].add(edge_row, edge_column, value);
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
        // A lower run is also its own column run.
        let row_run = self.scatter.parameter_run_location(rows);
        let column_run = if lower {
            row_run
        } else {
            self.scatter.parameter_run_location(columns)
        };
        let (Some((row_block, row_local)), Some((column_block, column_local))) =
            (row_run, column_run)
        else {
            return schur::add_gram_scalar(self, rows, columns, lower, left, right);
        };
        if row_local + rows.len() > MAX_REDUCED_BLOCK_DIMENSION
            || column_local + columns.len() > MAX_REDUCED_BLOCK_DIMENSION
        {
            self.valid = false;
            return;
        }
        if row_block == column_block {
            if !self.reduced_block_marks[row_block] {
                self.reduced_block_marks[row_block] = true;
                self.touched_reduced_blocks.push(row_block);
            }
            self.reduced_blocks[row_block].add_gram(row_local, column_local, lower, left, right);
            return;
        }
        // One edge lookup for the whole block; edges are stored with the
        // larger block as rows.
        let edge_blocks = (row_block.max(column_block), row_block.min(column_block));
        let Ok(edge) = self.base_edge_blocks.binary_search(&edge_blocks) else {
            self.valid = false;
            return;
        };
        if !self.base_edge_marks[edge] {
            self.base_edge_marks[edge] = true;
            self.touched_base_edges.push(edge);
        }
        if row_block > column_block {
            self.base_edges[edge].add_gram(row_local, column_local, left, right);
        } else {
            self.base_edges[edge].add_gram(column_local, row_local, right, left);
        }
    }

    fn is_finite(&self) -> bool {
        self.valid
            && self.touched_reduced_blocks.iter().all(|&block| {
                self.reduced_blocks[block]
                    .values
                    .iter()
                    .all(|value| value.is_finite())
            })
            && self.touched_base_edges.iter().all(|&edge| {
                self.base_edges[edge]
                    .values
                    .iter()
                    .all(|value| value.is_finite())
            })
    }
}

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct DirectPointAssemblyContext<'a> {
    problem: BaProblemView<'a>,
    policy: LinearizationPolicy<'a>,
    layout: VariableLayout<'a>,
    dimension: usize,
    scatter: &'a SparseScatter,
    base_edge_blocks: &'a [(usize, usize)],
    point_cross_offsets: &'a [usize],
}

#[cfg(feature = "parallel")]
pub(super) struct DirectPointCacheMut<'a> {
    pub(super) point_steps: Option<PointStepCacheBuffers<'a>>,
    pub(super) cross_offsets: &'a [usize],
    pub(super) inverse_crosses: PointCrossCacheMut<'a>,
    pub(super) point_hessians: &'a mut [PackedPointBlock],
}

#[cfg(feature = "parallel")]
pub(super) struct DirectReducedAssemblyMut<'a> {
    pub(super) gradient: &'a mut [f64],
    pub(super) full_gradient: &'a mut [f64],
    pub(super) diagonal: &'a mut [f64],
}

/// Reusable direct-Schur executor and reduced-block incidence index.
///
/// Construction is allowed to allocate. [`Self::materialize`] reads this
/// topology, point factors, and transformed cross rows, writes only its
/// preallocated edge slots, and gives every CSC block column to exactly one
/// deterministic partition.
pub(super) struct DirectSchurAssemblyWorkspace {
    block_point_offsets: Vec<usize>,
    block_points: Vec<PointCrossRef>,
    point_block_offsets: Vec<usize>,
    point_blocks: Vec<PointBlockRef>,
    #[cfg(feature = "parallel")]
    block_ranges: Vec<usize>,
    /// One edge slot per block for every materialization partition.
    edge_slots: Vec<u32>,
    block_value_offsets: Vec<usize>,
    point_count: usize,
    cross_count: usize,
    #[cfg(feature = "parallel")]
    dimension: usize,
    value_count: usize,
    #[cfg(feature = "parallel")]
    threads: usize,
    #[cfg(feature = "parallel")]
    logical_point_ranges: Vec<usize>,
    #[cfg(feature = "parallel")]
    reduced_base_blocks: Vec<PackedReducedBaseBlock>,
    #[cfg(feature = "parallel")]
    base_edge_blocks: Vec<(usize, usize)>,
    #[cfg(feature = "parallel")]
    base_edge_values: Vec<PackedReducedBaseEdge>,
    #[cfg(feature = "parallel")]
    point_assembly_scratch: Vec<DirectPointAssemblyScratch>,
    #[cfg(feature = "parallel")]
    pool: rayon::ThreadPool,
}

impl DirectSchurAssemblyWorkspace {
    #[cfg(feature = "parallel")]
    pub(super) fn install_parallel<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.pool.install(operation)
    }

    /// Assemble point linearizations into deterministic logical shards.
    ///
    /// Logical shard boundaries depend only on the point topology. Physical
    /// workers execute consecutive shards in waves, and the caller-visible
    /// camera blocks, vectors, and diagnostics are merged strictly in shard
    /// order. Point caches and packed point Hessians are written through
    /// disjoint point and cross-row slices.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn assemble_points(
        &mut self,
        problem: BaProblemView<'_>,
        policy: LinearizationPolicy<'_>,
        layout: VariableLayout<'_>,
        scatter: &SparseScatter,
        point_cache: DirectPointCacheMut<'_>,
        reduced: DirectReducedAssemblyMut<'_>,
    ) -> schur::Assembly {
        let DirectPointCacheMut {
            mut point_steps,
            cross_offsets: point_cross_offsets,
            inverse_crosses,
            point_hessians,
        } = point_cache;
        let DirectReducedAssemblyMut {
            gradient,
            full_gradient,
            diagonal,
        } = reduced;
        let points = problem.points;
        let fixed_points = problem.fixed.points;
        let point_priors = problem.priors.points;
        let valid_shape = points.len() == self.point_count
            && problem.model.offsets().len() == self.point_count + 1
            && layout.offsets.len() == problem.poses.len()
            && layout.pose_dimensions.len() == problem.poses.len()
            && layout.intrinsic_offsets.len() == layout.intrinsic_dimensions.len()
            && problem.poses.len() + layout.intrinsic_dimensions.len() == scatter.block_count()
            && (fixed_points.is_empty() || fixed_points.len() == self.point_count);
        let valid_shape = valid_shape
            && layout
                .pose_dimensions
                .iter()
                .chain(layout.intrinsic_dimensions)
                .all(|&dimension| dimension <= MAX_REDUCED_BLOCK_DIMENSION)
            && layout
                .offsets
                .iter()
                .zip(layout.pose_dimensions)
                .chain(
                    layout
                        .intrinsic_offsets
                        .iter()
                        .zip(layout.intrinsic_dimensions),
                )
                .all(|(&offset, &dimension)| offset.saturating_add(dimension) <= self.dimension)
            && point_cross_offsets.len() == self.point_count + 1
            && point_cross_offsets.last().copied() == Some(self.cross_count)
            && point_steps.as_ref().is_none_or(|steps| {
                steps.valid.len() == self.point_count
                    && steps.inverse_gradients.len() == self.point_count
                    && steps.damping_diagonals.len() == self.point_count
            })
            && point_hessians.len() == self.point_count
            && inverse_crosses.len() == self.cross_count
            && gradient.len() == self.dimension
            && full_gradient.len() == self.dimension
            && diagonal.len() == self.dimension
            && self.logical_point_ranges.last().copied().unwrap_or(0) == self.point_count
            && point_priors
                .windows(2)
                .all(|pair| pair[0].point < pair[1].point)
            && point_priors
                .last()
                .is_none_or(|prior| prior.point < self.point_count);
        if !valid_shape {
            return schur::Assembly::invalid();
        }

        for block in &mut self.reduced_base_blocks {
            block.values.fill(0.0);
        }
        for edge in &mut self.base_edge_values {
            edge.values.fill(0.0);
        }
        gradient.fill(0.0);
        full_gradient.fill(0.0);
        diagonal.fill(0.0);

        let context = DirectPointAssemblyContext {
            problem,
            policy,
            layout,
            dimension: self.dimension,
            scatter,
            base_edge_blocks: &self.base_edge_blocks,
            point_cross_offsets,
        };
        let shard_count = self.logical_point_ranges.len().saturating_sub(1);
        if shard_count == 0 {
            return schur::Assembly::empty();
        }
        let wave_width = self.point_assembly_scratch.len();
        if wave_width == 0 {
            return schur::Assembly::invalid();
        }

        let mut merged = schur::Assembly::empty();
        let Self {
            logical_point_ranges,
            reduced_base_blocks,
            base_edge_values,
            point_assembly_scratch,
            pool,
            ..
        } = self;
        let mut remaining_inverse_crosses = inverse_crosses;
        let mut remaining_cross_start = 0usize;
        for wave_start in (0..shard_count).step_by(wave_width) {
            let wave_end = (wave_start + wave_width).min(shard_count);
            let active_shards = wave_end - wave_start;
            let shard_ranges = &logical_point_ranges[wave_start..=wave_end];
            let point_start = shard_ranges[0];
            let point_end = shard_ranges[active_shards];
            let cross_start = point_cross_offsets[point_start];
            let cross_end = point_cross_offsets[point_end];
            if cross_start != remaining_cross_start {
                return schur::Assembly::invalid();
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
            let scratch = &mut point_assembly_scratch[..active_shards];
            let wave_valid = pool.install(|| {
                assemble_point_shards(
                    context,
                    shard_ranges,
                    scratch,
                    &points[point_start..point_end],
                    point_start,
                    wave_point_steps,
                    wave_inverse_crosses,
                    cross_start,
                    &mut point_hessians[point_start..point_end],
                )
            });
            if !wave_valid {
                return schur::Assembly::invalid();
            }

            for shard in scratch {
                let Some(assembly) = shard.assembly.take() else {
                    return schur::Assembly::invalid();
                };
                merge_assembly(&mut merged, assembly);
                for &block in &shard.touched_reduced_blocks {
                    let target = &mut reduced_base_blocks[block].values;
                    let source = &shard.reduced_blocks[block].values;
                    for (target, &source) in target.iter_mut().zip(source) {
                        *target += source;
                    }
                }
                for &edge in &shard.touched_base_edges {
                    let target = &mut base_edge_values[edge].values;
                    let source = &shard.base_edges[edge].values;
                    for (target, &source) in target.iter_mut().zip(source) {
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

        merged.valid &= reduced_base_blocks
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
            && base_edge_values
                .iter()
                .all(|edge| edge.values.iter().all(|value| value.is_finite()))
            && gradient.iter().all(|value| value.is_finite())
            && full_gradient.iter().all(|value| value.is_finite())
            && diagonal.iter().all(|value| value.is_finite());
        merged
    }

    /// Clear the explicit CSC values and seed the assembled camera base blocks.
    #[cfg(feature = "parallel")]
    pub(super) fn seed_base_values(&self, scatter: &SparseScatter, values: &mut [f64]) -> bool {
        if scatter.block_count() != self.reduced_base_blocks.len()
            || values.len() != self.value_count
        {
            return false;
        }
        values.fill(0.0);
        for (reduced_block, block) in self.reduced_base_blocks.iter().enumerate() {
            let Some(dimension) = scatter.block_dimension(reduced_block) else {
                return false;
            };
            if dimension == 0 {
                continue;
            }
            let Some(column_starts) =
                scatter.block_edge_column_starts(reduced_block, reduced_block)
            else {
                return false;
            };
            if column_starts.len() != dimension {
                return false;
            }
            for (column, &start) in column_starts.iter().enumerate() {
                for row in column..dimension {
                    let Some(target) = values.get_mut(start + row - column) else {
                        return false;
                    };
                    *target = block.get(row, column);
                }
            }
        }
        for (&(row_block, column_block), edge) in
            self.base_edge_blocks.iter().zip(&self.base_edge_values)
        {
            let Some(row_dimension) = scatter.block_dimension(row_block) else {
                return false;
            };
            let Some(column_dimension) = scatter.block_dimension(column_block) else {
                return false;
            };
            let Some(column_starts) = scatter.block_edge_column_starts(row_block, column_block)
            else {
                return false;
            };
            if column_starts.len() != column_dimension {
                return false;
            }
            for (column, &start) in column_starts.iter().enumerate() {
                for row in 0..row_dimension {
                    let Some(target) = values.get_mut(start + row) else {
                        return false;
                    };
                    *target = edge.get(row, column);
                }
            }
        }
        values.iter().all(|value| value.is_finite())
    }

    /// Add cached point-elimination coupling to an already assembled CSC base.
    ///
    /// Every stored entry has one owner. The owner visits contributing points
    /// in ascending point order, making the result independent of scheduling
    /// and of the number of worker ranges.
    pub(super) fn materialize<T: CrossRow>(
        &mut self,
        scatter: &SparseScatter,
        values: &mut [f64],
        point_hessians: &[PackedPointBlock],
        inverse_crosses: &[T],
    ) -> bool {
        if scatter.block_count() + 1 != self.block_value_offsets.len()
            || values.len() != self.value_count
            || point_hessians.len() != self.point_count
            || inverse_crosses.len() != self.cross_count
        {
            return false;
        }

        #[cfg(feature = "parallel")]
        {
            self.pool.install(|| {
                materialize_partitions(
                    &self.block_ranges,
                    &self.block_value_offsets,
                    &self.block_point_offsets,
                    &self.block_points,
                    &self.point_block_offsets,
                    &self.point_blocks,
                    scatter,
                    &mut self.edge_slots,
                    values,
                    0,
                    point_hessians,
                    inverse_crosses,
                )
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            materialize_block_range(
                0,
                scatter.block_count(),
                &self.block_point_offsets,
                &self.block_points,
                &self.point_block_offsets,
                &self.point_blocks,
                scatter,
                &mut self.edge_slots,
                values,
                0,
                point_hessians,
                inverse_crosses,
            )
        }
    }

    /// Materialize cached retained-Schur rows through the same deterministic
    /// column-owned executor used by the direct backend.
    #[cfg(feature = "parallel")]
    pub(super) fn materialize_cached(
        &mut self,
        scatter: &SparseScatter,
        values: &mut [f64],
        point_hessians: &[PackedPointBlock],
        inverse_crosses: &PointCrossCache,
    ) -> bool {
        match inverse_crosses {
            PointCrossCache::F64(rows) => self.materialize(scatter, values, point_hessians, rows),
            PointCrossCache::F32(rows) => self.materialize(scatter, values, point_hessians, rows),
            // Recomputed storage retains no rows, so only an empty cache can
            // materialize.
            PointCrossCache::Recomputed { len } => {
                *len == 0 && self.materialize::<[f64; 3]>(scatter, values, point_hessians, &[])
            }
        }
    }

    pub(super) const fn parallel_threads(&self) -> Option<usize> {
        #[cfg(feature = "parallel")]
        {
            Some(self.threads)
        }
        #[cfg(not(feature = "parallel"))]
        {
            None
        }
    }

    pub(super) fn auxiliary_bytes(&self) -> usize {
        let bytes = self.block_point_offsets.capacity() * std::mem::size_of::<usize>()
            + self.block_points.capacity() * std::mem::size_of::<PointCrossRef>()
            + self.point_block_offsets.capacity() * std::mem::size_of::<usize>()
            + self.point_blocks.capacity() * std::mem::size_of::<PointBlockRef>()
            + self.edge_slots.capacity() * std::mem::size_of::<u32>()
            + self.block_value_offsets.capacity() * std::mem::size_of::<usize>();
        #[cfg(feature = "parallel")]
        let bytes = bytes
            + self.block_ranges.capacity() * std::mem::size_of::<usize>()
            + self.logical_point_ranges.capacity() * std::mem::size_of::<usize>()
            + self.reduced_base_blocks.capacity() * std::mem::size_of::<PackedReducedBaseBlock>()
            + self.base_edge_blocks.capacity() * std::mem::size_of::<(usize, usize)>()
            + self.base_edge_values.capacity() * std::mem::size_of::<PackedReducedBaseEdge>()
            + self
                .point_assembly_scratch
                .iter()
                .map(DirectPointAssemblyScratch::capacity_bytes)
                .sum::<usize>();
        bytes
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn assemble_point_shards(
    context: DirectPointAssemblyContext<'_>,
    shard_ranges: &[usize],
    scratch: &mut [DirectPointAssemblyScratch],
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
        let DirectPointAssemblyScratch {
            reduced_blocks,
            reduced_block_marks,
            touched_reduced_blocks,
            base_edges,
            base_edge_marks,
            touched_base_edges,
            gradient,
            full_gradient,
            diagonal,
            cross,
            active_parameters,
            parameter_marks,
            assembly,
        } = &mut scratch[0];
        let mut hessian = PackedReducedBaseHessian {
            reduced_blocks,
            reduced_block_marks,
            touched_reduced_blocks,
            base_edge_blocks: context.base_edge_blocks,
            base_edges,
            base_edge_marks,
            touched_base_edges,
            scatter: context.scatter,
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
        let mut result = match context.layout.pose_variables {
            super::super::CameraPoseVariables::RotationAndTranslation => {
                schur::assemble_retained_range(
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
                )
            }
            super::super::CameraPoseVariables::TranslationOnly => {
                schur::assemble_retained_range_translation_only(
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
                )
            }
        };
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
    let (left_steps, right_steps) = match point_steps {
        Some(PointStepCacheBuffers {
            valid,
            inverse_gradients,
            damping_diagonals,
        }) => {
            let (left_valid, right_valid) = valid.split_at_mut(local_point_split);
            let (left_gradients, right_gradients) =
                inverse_gradients.split_at_mut(local_point_split);
            let (left_diagonals, right_diagonals) =
                damping_diagonals.split_at_mut(local_point_split);
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
    };
    let (left_hessians, right_hessians) = point_hessians.split_at_mut(local_point_split);
    let (left_crosses, right_crosses) = inverse_crosses.split_at_mut(local_cross_split);
    let (left, right) = rayon::join(
        || {
            assemble_point_shards(
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
            assemble_point_shards(
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
fn try_build_direct_schur_pool(
    threads: usize,
) -> Result<rayon::ThreadPool, SparseBundleAdjustmentError> {
    super::parallel::try_build_pool(threads, "openmvg-ba-direct-schur")
        .map_err(|_| SparseBundleAdjustmentError::DirectSchurThreadPoolBuild)
}

#[cfg(test)]
mod tests;
