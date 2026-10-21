// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced-system sparsity, scalar scatter, and its symbolic Cholesky analysis.

use std::collections::BTreeSet;

use crate::numerics::sparse::{SparseCholesky, SymmetricPattern};

use crate::kernel::multi_view::MultiViewBundleAdjustmentError;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::sparse::api::SparseBundleAdjustmentError;

pub(in crate::kernel::multi_view::sparse) struct SparseScatter {
    parameter_blocks: Vec<u32>,
    parameter_locals: Vec<u8>,
    block_dimensions: Vec<u8>,
    block_value_offsets: Vec<usize>,
    block_col_ptr: Vec<usize>,
    block_rows: Vec<u32>,
    edge_start_ptr: Vec<usize>,
    column_starts: Vec<usize>,
}

impl SparseScatter {
    pub(super) fn empty() -> Self {
        Self {
            parameter_blocks: Vec::new(),
            parameter_locals: Vec::new(),
            block_dimensions: Vec::new(),
            block_value_offsets: vec![0],
            block_col_ptr: Vec::new(),
            block_rows: Vec::new(),
            edge_start_ptr: Vec::new(),
            column_starts: Vec::new(),
        }
    }

    #[cfg(feature = "unstable-internals")]
    pub(super) fn capacity_bytes(&self) -> usize {
        self.parameter_blocks.capacity() * std::mem::size_of::<u32>()
            + self.parameter_locals.capacity() * std::mem::size_of::<u8>()
            + self.block_dimensions.capacity() * std::mem::size_of::<u8>()
            + (self.block_value_offsets.capacity()
                + self.block_col_ptr.capacity()
                + self.edge_start_ptr.capacity()
                + self.column_starts.capacity())
                * std::mem::size_of::<usize>()
            + self.block_rows.capacity() * std::mem::size_of::<u32>()
    }

    pub(in crate::kernel::multi_view::sparse) fn try_new(
        pattern: &SymmetricPattern,
        block_offsets: &[usize],
        block_dimensions: &[usize],
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let column_pointers = pattern.column_pointers();
        let row_indices = pattern.row_indices();
        let dimension = pattern.dimension();
        let mut parameter_blocks = vec![0u32; dimension];
        let mut parameter_locals = vec![0u8; dimension];
        let compact_block_dimensions = block_dimensions
            .iter()
            .map(|&dimension| {
                u8::try_from(dimension)
                    .map_err(|_| SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut block_value_offsets = Vec::with_capacity(block_offsets.len() + 1);
        for &offset in block_offsets {
            block_value_offsets.push(column_pointers[offset]);
        }
        block_value_offsets.push(row_indices.len());
        for (block, (&offset, &block_dimension)) in
            block_offsets.iter().zip(block_dimensions).enumerate()
        {
            let block = u32::try_from(block)
                .map_err(|_| SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
            for local in 0..block_dimension {
                parameter_blocks[offset + local] = block;
                parameter_locals[offset + local] = u8::try_from(local)
                    .map_err(|_| SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
            }
        }

        let mut block_col_ptr = Vec::with_capacity(block_offsets.len() + 1);
        let mut block_rows = Vec::new();
        let mut edge_start_ptr = vec![0usize];
        let mut column_starts = Vec::new();
        block_col_ptr.push(0);
        for (column_block, (&column_offset, &column_dimension)) in
            block_offsets.iter().zip(block_dimensions).enumerate()
        {
            if column_dimension > 0 {
                let scalar_range =
                    column_pointers[column_offset]..column_pointers[column_offset + 1];
                let mut previous = None;
                for &row in &row_indices[scalar_range] {
                    let row_block = parameter_blocks[row] as usize;
                    if previous == Some(row_block) {
                        continue;
                    }
                    previous = Some(row_block);
                    block_rows.push(u32::try_from(row_block).map_err(|_| {
                        SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow
                    })?);
                    let row_offset = block_offsets[row_block];
                    for column_local in 0..column_dimension {
                        let column = column_offset + column_local;
                        let row = if row_block == column_block {
                            column
                        } else {
                            row_offset
                        };
                        let range = column_pointers[column]..column_pointers[column + 1];
                        let local = row_indices[range.clone()]
                            .binary_search(&row)
                            .map_err(|_| SparseBundleAdjustmentError::SymbolicFactorization)?;
                        column_starts.push(range.start + local);
                    }
                    edge_start_ptr.push(column_starts.len());
                }
            }
            block_col_ptr.push(block_rows.len());
        }
        Ok(Self {
            parameter_blocks,
            parameter_locals,
            block_dimensions: compact_block_dimensions,
            block_value_offsets,
            block_col_ptr,
            block_rows,
            edge_start_ptr,
            column_starts,
        })
    }

    pub(in crate::kernel::multi_view::sparse) fn block_count(&self) -> usize {
        self.block_dimensions.len()
    }

    pub(in crate::kernel::multi_view::sparse) fn block_dimension(
        &self,
        block: usize,
    ) -> Option<usize> {
        self.block_dimensions.get(block).copied().map(usize::from)
    }

    pub(in crate::kernel::multi_view::sparse) fn parameter_location(
        &self,
        parameter: usize,
    ) -> Option<(usize, usize)> {
        Some((
            *self.parameter_blocks.get(parameter)? as usize,
            *self.parameter_locals.get(parameter)? as usize,
        ))
    }

    /// Block and first local index of a run of consecutive parameters, when
    /// the whole run lies inside one block.
    #[inline]
    pub(in crate::kernel::multi_view::sparse) fn parameter_run_location(
        &self,
        run: &[usize],
    ) -> Option<(usize, usize)> {
        let (&first, &last) = (run.first()?, run.last()?);
        debug_assert!(run.windows(2).all(|pair| pair[1] == pair[0] + 1));
        let (block, local) = self.parameter_location(first)?;
        (self.parameter_location(last)? == (block, local + run.len() - 1)).then_some((block, local))
    }

    pub(in crate::kernel::multi_view::sparse) fn block_column_value_span(
        &self,
        block: usize,
    ) -> Option<std::ops::Range<usize>> {
        Some(
            *self.block_value_offsets.get(block)?
                ..*self.block_value_offsets.get(block.checked_add(1)?)?,
        )
    }

    /// Scalar-column starts for one lower-triangular block edge.
    ///
    /// Each returned position addresses the first stored row of the edge in
    /// the corresponding scalar column. Diagonal edges therefore start at
    /// the scalar diagonal, while off-diagonal edges start at row-local zero.
    pub(in crate::kernel::multi_view::sparse) fn block_edge_column_starts(
        &self,
        row_block: usize,
        column_block: usize,
    ) -> Option<&[usize]> {
        if row_block < column_block {
            return None;
        }
        let range = *self.block_col_ptr.get(column_block)?
            ..*self.block_col_ptr.get(column_block.checked_add(1)?)?;
        let edge_local = self
            .block_rows
            .get(range.clone())?
            .binary_search(&u32::try_from(row_block).ok()?)
            .ok()?;
        let edge = range.start + edge_local;
        let starts =
            *self.edge_start_ptr.get(edge)?..*self.edge_start_ptr.get(edge.checked_add(1)?)?;
        self.column_starts.get(starts)
    }

    /// Row blocks of every stored edge of one block column, ascending from its
    /// diagonal, and the scalar-column starts of those edges in the same
    /// order: one start per scalar column of the block for each edge.
    pub(in crate::kernel::multi_view::sparse) fn block_column_edges(
        &self,
        column_block: usize,
    ) -> Option<(&[u32], &[usize])> {
        let edges = *self.block_col_ptr.get(column_block)?
            ..*self.block_col_ptr.get(column_block.checked_add(1)?)?;
        let starts = *self.edge_start_ptr.get(edges.start)?..*self.edge_start_ptr.get(edges.end)?;
        Some((self.block_rows.get(edges)?, self.column_starts.get(starts)?))
    }

    #[inline]
    pub(in crate::kernel::multi_view::sparse) fn position(
        &self,
        row: usize,
        column: usize,
    ) -> Option<usize> {
        let (row, column) = if row >= column {
            (row, column)
        } else {
            (column, row)
        };
        let row_block = *self.parameter_blocks.get(row)? as usize;
        let column_block = *self.parameter_blocks.get(column)? as usize;
        let range = self.block_col_ptr[column_block]..self.block_col_ptr[column_block + 1];
        let edge_local = self.block_rows[range.clone()]
            .binary_search(&(row_block as u32))
            .ok()?;
        let edge = range.start + edge_local;
        let column_local = self.parameter_locals[column] as usize;
        let row_local = self.parameter_locals[row] as usize;
        let start = self.column_starts[self.edge_start_ptr[edge] + column_local];
        Some(
            start
                + if row_block == column_block {
                    row_local.checked_sub(column_local)?
                } else {
                    row_local
                },
        )
    }
}

/// Analyze the reduced pattern `(column_pointers, row_indices)` of a `dimension`-wide system
/// whose parameter blocks start at `block_offsets`: its scalar scatter and its symbolic Cholesky
/// factor, without numeric storage. Nothing is kept when either fails.
pub(super) fn analyze_reduced_pattern(
    dimension: usize,
    (column_pointers, row_indices): (Vec<usize>, Vec<usize>),
    block_offsets: &[usize],
    block_dimensions: &[usize],
) -> Result<(SparseCholesky, SparseScatter), SparseBundleAdjustmentError> {
    let pattern = SymmetricPattern::new(dimension, column_pointers, row_indices)
        .ok_or(SparseBundleAdjustmentError::SymbolicFactorization)?;
    let scatter = SparseScatter::try_new(&pattern, block_offsets, block_dimensions)?;
    let cholesky = SparseCholesky::analyze(pattern)
        .ok_or(SparseBundleAdjustmentError::SymbolicFactorization)?;
    Ok((cholesky, scatter))
}

pub(super) fn point_cross_offsets(
    model: ObservationModel<'_>,
    point_count: usize,
    camera_dimensions: &[usize],
    fixed_points: &[bool],
) -> Result<Vec<usize>, MultiViewBundleAdjustmentError> {
    let mut offsets = Vec::with_capacity(point_count + 1);
    let mut count = 0usize;
    let mut marked = vec![false; camera_dimensions.len()];
    let mut touched = Vec::new();
    offsets.push(0);
    for point in 0..point_count {
        if !fixed_points.is_empty() && fixed_points[point] {
            offsets.push(count);
            continue;
        }
        touched.clear();
        for observation in model.offsets()[point]..model.offsets()[point + 1] {
            let camera = model.camera(observation);
            if !marked[camera] {
                marked[camera] = true;
                touched.push(camera);
                count = count
                    .checked_add(camera_dimensions[camera])
                    .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
            }
        }
        for &camera in &touched {
            marked[camera] = false;
        }
        offsets.push(count);
    }
    Ok(offsets)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn radial_point_parameters(
    model: ObservationModel<'_>,
    point_count: usize,
    dimension: usize,
    camera_offsets: &[usize],
    camera_dimensions: &[usize],
    intrinsic_offsets: &[usize],
    intrinsic_dimensions: &[usize],
    fixed_points: &[bool],
) -> Result<(Vec<usize>, Vec<u32>, usize), MultiViewBundleAdjustmentError> {
    let mut offsets = Vec::with_capacity(point_count + 1);
    let mut parameters = Vec::new();
    let mut active = Vec::with_capacity(dimension);
    let mut marks = vec![false; dimension];
    let mut active_block_references = 0usize;
    offsets.push(0);
    for point in 0..point_count {
        active.clear();
        if !fixed_points.is_empty() && fixed_points[point] {
            offsets.push(parameters.len());
            continue;
        }
        for observation in model.offsets()[point]..model.offsets()[point + 1] {
            let camera = model.camera(observation);
            let camera_offset = camera_offsets[camera];
            let camera_dimension = camera_dimensions[camera];
            if camera_dimension > 0 && !marks[camera_offset] {
                active_block_references = active_block_references
                    .checked_add(1)
                    .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
            }
            for (parameter, mark) in marks
                .iter_mut()
                .enumerate()
                .skip(camera_offset)
                .take(camera_dimension)
            {
                if !*mark {
                    *mark = true;
                    active.push(parameter);
                }
            }
            if let Some(group) = model.intrinsic_group(camera) {
                let intrinsic_offset = intrinsic_offsets[group];
                let intrinsic_dimension = intrinsic_dimensions[group];
                if intrinsic_dimension > 0 && !marks[intrinsic_offset] {
                    active_block_references = active_block_references
                        .checked_add(1)
                        .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
                }
                for (parameter, mark) in marks
                    .iter_mut()
                    .enumerate()
                    .skip(intrinsic_offset)
                    .take(intrinsic_dimension)
                {
                    if !*mark {
                        *mark = true;
                        active.push(parameter);
                    }
                }
            }
        }
        for &parameter in &active {
            parameters.push(
                u32::try_from(parameter)
                    .map_err(|_| MultiViewBundleAdjustmentError::CameraCapacityOverflow)?,
            );
            marks[parameter] = false;
        }
        offsets.push(parameters.len());
    }
    Ok((offsets, parameters, active_block_references))
}

pub(super) fn radial_reduced_block_columns(
    point_offsets: &[usize],
    observation_cameras: &[u32],
    camera_to_group: Option<&[usize]>,
    camera_dimensions: &[usize],
    intrinsic_dimensions: &[usize],
    fixed_points: &[bool],
) -> Vec<BTreeSet<usize>> {
    let camera_count = camera_dimensions.len();
    let block_count = camera_count + intrinsic_dimensions.len();
    let block_dimension = |block: usize| {
        if block < camera_count {
            camera_dimensions[block]
        } else {
            intrinsic_dimensions[block - camera_count]
        }
    };
    let mut block_columns: Vec<BTreeSet<usize>> = (0..block_count)
        .map(|block| {
            if block_dimension(block) == 0 {
                BTreeSet::new()
            } else {
                BTreeSet::from([block])
            }
        })
        .collect();
    let mut active_blocks = Vec::new();
    for point in 0..point_offsets.len().saturating_sub(1) {
        if !fixed_points.is_empty() && fixed_points[point] {
            for &camera in &observation_cameras[point_offsets[point]..point_offsets[point + 1]] {
                let camera = camera as usize;
                let group = camera_to_group.map_or(camera, |groups| groups[camera]);
                if camera_dimensions[camera] > 0 && intrinsic_dimensions[group] > 0 {
                    block_columns[camera].insert(camera_count + group);
                }
            }
            continue;
        }
        active_blocks.clear();
        for &camera in &observation_cameras[point_offsets[point]..point_offsets[point + 1]] {
            let camera = camera as usize;
            if camera_dimensions[camera] > 0 {
                active_blocks.push(camera);
            }
            let group = camera_to_group.map_or(camera, |groups| groups[camera]);
            if intrinsic_dimensions[group] > 0 {
                active_blocks.push(camera_count + group);
            }
        }
        active_blocks.sort_unstable();
        active_blocks.dedup();
        for (right_index, &row_block) in active_blocks.iter().enumerate() {
            for &column_block in &active_blocks[..=right_index] {
                block_columns[column_block].insert(row_block);
            }
        }
    }
    block_columns
}

pub(super) fn radial_reduced_pattern(
    block_columns: &[BTreeSet<usize>],
    dimension: usize,
    block_offsets: &[usize],
    block_dimensions: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut col_ptr = Vec::with_capacity(dimension + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for (column_block, column_rows) in block_columns.iter().enumerate() {
        for column_local in 0..block_dimensions[column_block] {
            for &row_block in column_rows {
                let row_local_start = if row_block == column_block {
                    column_local
                } else {
                    0
                };
                row_idx.extend(
                    block_offsets[row_block] + row_local_start
                        ..block_offsets[row_block] + block_dimensions[row_block],
                );
            }
            col_ptr.push(row_idx.len());
        }
    }
    debug_assert_eq!(col_ptr.len(), dimension + 1);
    (col_ptr, row_idx)
}

pub(super) fn radial_reduced_nonzero_count(
    block_columns: &[BTreeSet<usize>],
    dimensions: &[usize],
) -> usize {
    block_columns
        .iter()
        .enumerate()
        .map(|(column_block, row_blocks)| {
            (0..dimensions[column_block])
                .map(|column_local| {
                    row_blocks
                        .iter()
                        .map(|&row_block| {
                            dimensions[row_block]
                                - usize::from(row_block == column_block) * column_local
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
        })
        .sum()
}

pub(super) fn reduced_block_columns(
    point_offsets: &[usize],
    observation_cameras: &[u32],
    dimensions: &[usize],
    fixed_points: &[bool],
) -> Vec<Vec<usize>> {
    const MAX_BITMAP_BYTES: usize = 256 * 1024 * 1024;

    reduced_block_columns_with_bitmap_limit(
        point_offsets,
        observation_cameras,
        dimensions,
        fixed_points,
        MAX_BITMAP_BYTES,
    )
}

pub(super) fn reduced_block_columns_with_bitmap_limit(
    point_offsets: &[usize],
    observation_cameras: &[u32],
    dimensions: &[usize],
    fixed_points: &[bool],
    maximum_bitmap_bytes: usize,
) -> Vec<Vec<usize>> {
    let camera_count = dimensions.len();
    let words_per_column = camera_count.div_ceil(u64::BITS as usize);
    let maximum_words = maximum_bitmap_bytes / std::mem::size_of::<u64>();
    if let Some(bitmap_len) = camera_count
        .checked_mul(words_per_column)
        .filter(|&len| len <= maximum_words)
    {
        let mut bitmap = vec![0u64; bitmap_len];
        for (camera, &dimension) in dimensions.iter().enumerate() {
            if dimension > 0 {
                set_visibility_bit(&mut bitmap, words_per_column, camera, camera);
            }
        }
        let mut active_cameras = Vec::new();
        for point in 0..point_offsets.len().saturating_sub(1) {
            if !fixed_points.is_empty() && fixed_points[point] {
                continue;
            }
            active_cameras.clear();
            for &camera in &observation_cameras[point_offsets[point]..point_offsets[point + 1]] {
                let camera = camera as usize;
                if dimensions[camera] > 0 {
                    active_cameras.push(camera);
                }
            }
            active_cameras.sort_unstable();
            active_cameras.dedup();
            for (right_index, &row_camera) in active_cameras.iter().enumerate() {
                for &column_camera in &active_cameras[..=right_index] {
                    set_visibility_bit(&mut bitmap, words_per_column, column_camera, row_camera);
                }
            }
        }
        return (0..camera_count)
            .map(|camera| visibility_rows(&bitmap, words_per_column, camera, camera_count))
            .collect();
    }

    // Build the visibility graph once at camera-block granularity. Expanding
    // every observation pair directly to scalar parameters performs up to 36
    // identical ordered-set insertions for each 6x6 block and dominates setup
    // on multi-million-observation BAL scenes.
    let mut camera_columns: Vec<BTreeSet<usize>> = (0..camera_count)
        .map(|camera| {
            if dimensions[camera] == 0 {
                BTreeSet::new()
            } else {
                BTreeSet::from([camera])
            }
        })
        .collect();
    let mut active_cameras = Vec::new();
    for point in 0..point_offsets.len().saturating_sub(1) {
        if !fixed_points.is_empty() && fixed_points[point] {
            continue;
        }
        active_cameras.clear();
        for &camera in &observation_cameras[point_offsets[point]..point_offsets[point + 1]] {
            let camera = camera as usize;
            if dimensions[camera] > 0 {
                active_cameras.push(camera);
            }
        }
        active_cameras.sort_unstable();
        active_cameras.dedup();
        for (right_index, &row_camera) in active_cameras.iter().enumerate() {
            for &column_camera in &active_cameras[..=right_index] {
                camera_columns[column_camera].insert(row_camera);
            }
        }
    }
    camera_columns
        .into_iter()
        .map(|column| column.into_iter().collect())
        .collect()
}

fn set_visibility_bit(bitmap: &mut [u64], words_per_column: usize, column: usize, row: usize) {
    let word = column * words_per_column + row / u64::BITS as usize;
    bitmap[word] |= 1 << (row % u64::BITS as usize);
}

fn visibility_rows(
    bitmap: &[u64],
    words_per_column: usize,
    column: usize,
    camera_count: usize,
) -> Vec<usize> {
    let start = column * words_per_column;
    let mut rows = Vec::new();
    for (word_index, &source) in bitmap[start..start + words_per_column].iter().enumerate() {
        let mut word = source;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let row = word_index * u64::BITS as usize + bit;
            if row < camera_count {
                rows.push(row);
            }
            word &= word - 1;
        }
    }
    rows
}

pub(super) fn reduced_nonzero_count(camera_columns: &[Vec<usize>], dimensions: &[usize]) -> usize {
    camera_columns
        .iter()
        .enumerate()
        .map(|(column_camera, row_cameras)| {
            (0..dimensions[column_camera])
                .map(|column_local| {
                    row_cameras
                        .iter()
                        .map(|&row_camera| {
                            dimensions[row_camera]
                                - usize::from(row_camera == column_camera) * column_local
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
        })
        .sum()
}

pub(super) fn reduced_pattern(
    camera_columns: &[Vec<usize>],
    dimension: usize,
    offsets: &[usize],
    dimensions: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let camera_count = dimensions.len();
    let mut col_ptr = Vec::with_capacity(dimension + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for column_camera in 0..camera_count {
        for column_local in 0..dimensions[column_camera] {
            for &row_camera in &camera_columns[column_camera] {
                let row_local_start = if row_camera == column_camera {
                    column_local
                } else {
                    0
                };
                row_idx.extend(
                    offsets[row_camera] + row_local_start
                        ..offsets[row_camera] + dimensions[row_camera],
                );
            }
            col_ptr.push(row_idx.len());
        }
    }
    debug_assert_eq!(col_ptr.len(), dimension + 1);
    (col_ptr, row_idx)
}
