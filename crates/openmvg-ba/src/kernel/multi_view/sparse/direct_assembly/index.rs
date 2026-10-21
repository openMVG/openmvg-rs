// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct-Schur topology validation and compact incidence-index construction.

use super::*;
impl DirectSchurAssemblyWorkspace {
    /// Build the compact calibrated-camera index and deterministic work ranges.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::kernel::multi_view::sparse) fn try_new(
        threads: usize,
        point_offsets: &[usize],
        observation_cameras: &[u32],
        camera_dimensions: &[usize],
        point_cross_offsets: &[usize],
        reduced_dimension: usize,
        reduced_value_count: usize,
        scatter: &SparseScatter,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        if threads == 0 {
            return Err(SparseBundleAdjustmentError::DirectSchurThreadsMustBePositive);
        }
        let point_count = point_offsets.len().saturating_sub(1);
        if point_offsets.len() != point_cross_offsets.len()
            || point_offsets.last().copied().unwrap_or(0) != observation_cameras.len()
            || camera_dimensions.len() != scatter.block_count()
            || reduced_dimension != camera_dimensions.iter().copied().sum::<usize>()
        {
            return Err(SparseBundleAdjustmentError::TopologyMismatch);
        }

        let block_count = camera_dimensions.len();
        let mut point_block_offsets = Vec::with_capacity(point_count + 1);
        let mut point_blocks = Vec::new();
        let mut active_blocks = Vec::new();
        let mut block_marks = vec![false; block_count];
        point_block_offsets.push(0);

        for point in 0..point_count {
            active_blocks.clear();
            let cross_start = point_cross_offsets[point];
            let cross_end = point_cross_offsets[point + 1];
            if cross_start == cross_end {
                point_block_offsets.push(point_blocks.len());
                continue;
            }

            let mut cross_offset = cross_start;
            for &camera in &observation_cameras[point_offsets[point]..point_offsets[point + 1]] {
                let camera = usize::try_from(camera)
                    .map_err(|_| SparseBundleAdjustmentError::TopologyMismatch)?;
                let Some(&dimension) = camera_dimensions.get(camera) else {
                    return Err(SparseBundleAdjustmentError::TopologyMismatch);
                };
                if dimension > 0 {
                    if block_marks[camera] {
                        return Err(SparseBundleAdjustmentError::TopologyMismatch);
                    }
                    block_marks[camera] = true;
                    active_blocks.push(PointBlockRef {
                        block: u32::try_from(camera).map_err(|_| {
                            SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow
                        })?,
                        cross_offset: u32::try_from(cross_offset).map_err(|_| {
                            SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow
                        })?,
                    });
                }
                cross_offset = cross_offset
                    .checked_add(dimension)
                    .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
            }
            if cross_offset != cross_end {
                return Err(SparseBundleAdjustmentError::TopologyMismatch);
            }

            active_blocks.sort_unstable_by_key(|reference| reference.block);
            point_blocks.extend_from_slice(&active_blocks);
            for reference in &active_blocks {
                block_marks[reference.block as usize] = false;
            }
            point_block_offsets.push(point_blocks.len());
        }

        Self::try_new_from_point_blocks(
            threads,
            point_offsets,
            camera_dimensions,
            point_cross_offsets,
            point_block_offsets,
            point_blocks,
            Vec::new(),
            reduced_dimension,
            reduced_value_count,
            scatter,
        )
    }
}
impl DirectSchurAssemblyWorkspace {
    /// Build an index from each point's unique active scalar parameters.
    ///
    /// The parameter list must contain complete reduced blocks in local order.
    /// This is the representation produced by grouped radial self-calibration,
    /// where shared intrinsic blocks are deduplicated per point.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::kernel::multi_view::sparse) fn try_new_indexed(
        threads: usize,
        point_offsets: &[usize],
        point_cross_offsets: &[usize],
        point_parameters: &[u32],
        block_dimensions: &[usize],
        base_block_edges: &[(usize, usize)],
        reduced_dimension: usize,
        reduced_value_count: usize,
        scatter: &SparseScatter,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        if threads == 0 {
            return Err(SparseBundleAdjustmentError::DirectSchurThreadsMustBePositive);
        }
        let point_count = point_cross_offsets.len().saturating_sub(1);
        if point_offsets.len() != point_count + 1
            || point_offsets.first().copied() != Some(0)
            || point_offsets.windows(2).any(|range| range[0] > range[1])
            || point_cross_offsets.first().copied() != Some(0)
            || point_cross_offsets.last().copied() != Some(point_parameters.len())
            || block_dimensions.len() != scatter.block_count()
            || reduced_dimension != block_dimensions.iter().copied().sum::<usize>()
        {
            return Err(SparseBundleAdjustmentError::TopologyMismatch);
        }

        let mut point_block_offsets = Vec::with_capacity(point_count + 1);
        let mut point_blocks = Vec::new();
        let mut active_blocks = Vec::new();
        let mut block_marks = vec![false; block_dimensions.len()];
        point_block_offsets.push(0);
        for point in 0..point_count {
            active_blocks.clear();
            let mut cross_offset = point_cross_offsets[point];
            let cross_end = point_cross_offsets[point + 1];
            while cross_offset < cross_end {
                let parameter = usize::try_from(point_parameters[cross_offset])
                    .map_err(|_| SparseBundleAdjustmentError::TopologyMismatch)?;
                let Some((block, local)) = scatter.parameter_location(parameter) else {
                    return Err(SparseBundleAdjustmentError::TopologyMismatch);
                };
                let Some(&dimension) = block_dimensions.get(block) else {
                    return Err(SparseBundleAdjustmentError::TopologyMismatch);
                };
                if dimension == 0
                    || local != 0
                    || block_marks[block]
                    || cross_offset.saturating_add(dimension) > cross_end
                {
                    return Err(SparseBundleAdjustmentError::TopologyMismatch);
                }
                for block_local in 0..dimension {
                    let parameter = usize::try_from(point_parameters[cross_offset + block_local])
                        .map_err(|_| SparseBundleAdjustmentError::TopologyMismatch)?;
                    if scatter.parameter_location(parameter) != Some((block, block_local)) {
                        return Err(SparseBundleAdjustmentError::TopologyMismatch);
                    }
                }
                block_marks[block] = true;
                active_blocks.push(PointBlockRef {
                    block: u32::try_from(block).map_err(|_| {
                        SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow
                    })?,
                    cross_offset: u32::try_from(cross_offset).map_err(|_| {
                        SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow
                    })?,
                });
                cross_offset = cross_offset
                    .checked_add(dimension)
                    .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
            }
            active_blocks.sort_unstable_by_key(|reference| reference.block);
            point_blocks.extend_from_slice(&active_blocks);
            for reference in &active_blocks {
                block_marks[reference.block as usize] = false;
            }
            point_block_offsets.push(point_blocks.len());
        }

        Self::try_new_from_point_blocks(
            threads,
            point_offsets,
            block_dimensions,
            point_cross_offsets,
            point_block_offsets,
            point_blocks,
            base_block_edges.to_vec(),
            reduced_dimension,
            reduced_value_count,
            scatter,
        )
    }
}
impl DirectSchurAssemblyWorkspace {
    #[allow(clippy::too_many_arguments)]
    fn try_new_from_point_blocks(
        threads: usize,
        point_work_offsets: &[usize],
        block_dimensions: &[usize],
        point_cross_offsets: &[usize],
        point_block_offsets: Vec<usize>,
        point_blocks: Vec<PointBlockRef>,
        mut base_edge_blocks: Vec<(usize, usize)>,
        reduced_dimension: usize,
        reduced_value_count: usize,
        scatter: &SparseScatter,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        #[cfg(not(feature = "parallel"))]
        let _ = (threads, point_work_offsets, reduced_dimension);
        let point_count = point_cross_offsets.len().saturating_sub(1);
        let block_count = block_dimensions.len();
        if point_block_offsets.len() != point_count + 1
            || point_block_offsets.first().copied() != Some(0)
            || point_block_offsets.last().copied() != Some(point_blocks.len())
        {
            return Err(SparseBundleAdjustmentError::TopologyMismatch);
        }
        base_edge_blocks.sort_unstable();
        base_edge_blocks.dedup();
        if base_edge_blocks.iter().any(|&(row, column)| {
            row <= column
                || block_dimensions.get(row).copied().unwrap_or(0) == 0
                || block_dimensions.get(column).copied().unwrap_or(0) == 0
                || scatter.block_edge_column_starts(row, column).is_none()
        }) {
            return Err(SparseBundleAdjustmentError::TopologyMismatch);
        }

        let value_count = reduced_value_count;
        let mut block_value_offsets = Vec::with_capacity(block_count + 1);
        let mut expected_value_start = 0usize;
        for block in 0..block_count {
            let Some(span) = scatter.block_column_value_span(block) else {
                return Err(SparseBundleAdjustmentError::TopologyMismatch);
            };
            if span.start != expected_value_start {
                return Err(SparseBundleAdjustmentError::TopologyMismatch);
            }
            block_value_offsets.push(span.start);
            expected_value_start = span.end;
        }
        block_value_offsets.push(expected_value_start);
        if block_value_offsets.last().copied() != Some(value_count) {
            return Err(SparseBundleAdjustmentError::TopologyMismatch);
        }

        let mut block_counts = vec![0usize; block_count];
        let mut block_work = vec![0usize; block_count];
        for point in 0..point_count {
            let active_blocks =
                &point_blocks[point_block_offsets[point]..point_block_offsets[point + 1]];
            if active_blocks
                .windows(2)
                .any(|pair| pair[0].block >= pair[1].block)
            {
                return Err(SparseBundleAdjustmentError::TopologyMismatch);
            }
            for (column_index, reference) in active_blocks.iter().enumerate() {
                let column_block = reference.block as usize;
                let Some(&column_dimension) = block_dimensions.get(column_block) else {
                    return Err(SparseBundleAdjustmentError::TopologyMismatch);
                };
                block_counts[column_block] = block_counts[column_block]
                    .checked_add(1)
                    .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
                let diagonal_work = column_dimension
                    .checked_mul(column_dimension + 1)
                    .and_then(|work| work.checked_div(2))
                    .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
                block_work[column_block] = block_work[column_block]
                    .checked_add(diagonal_work)
                    .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
                for row in &active_blocks[column_index + 1..] {
                    let row_dimension = block_dimensions[row.block as usize];
                    let work = column_dimension
                        .checked_mul(row_dimension)
                        .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
                    block_work[column_block] = block_work[column_block]
                        .checked_add(work)
                        .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
                }
            }
        }

        let mut block_point_offsets = Vec::with_capacity(block_count + 1);
        block_point_offsets.push(0usize);
        for count in block_counts {
            let next = block_point_offsets
                .last()
                .copied()
                .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?
                .checked_add(count)
                .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
            block_point_offsets.push(next);
        }
        let reference_count = block_point_offsets.last().copied().unwrap_or(0);
        let mut block_points = vec![PointCrossRef::default(); reference_count];
        let mut cursors = block_point_offsets[..block_count].to_vec();
        for point in 0..point_count {
            for reference in
                &point_blocks[point_block_offsets[point]..point_block_offsets[point + 1]]
            {
                let block = reference.block as usize;
                let destination = cursors[block];
                block_points[destination] = PointCrossRef {
                    point: u32::try_from(point).map_err(|_| {
                        SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow
                    })?,
                    cross_offset: reference.cross_offset,
                };
                cursors[block] += 1;
            }
        }
        if cursors != block_point_offsets[1..] {
            return Err(SparseBundleAdjustmentError::TopologyMismatch);
        }

        let mut work_offsets = Vec::with_capacity(block_count + 1);
        work_offsets.push(0usize);
        for work in block_work {
            let next = work_offsets
                .last()
                .copied()
                .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?
                .checked_add(work)
                .ok_or(SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow)?;
            work_offsets.push(next);
        }
        #[cfg(feature = "parallel")]
        let block_ranges = balanced_contiguous_ranges(&work_offsets, threads);
        #[cfg(feature = "parallel")]
        let partitions = block_ranges.len().saturating_sub(1).max(1);
        #[cfg(not(feature = "parallel"))]
        let partitions = 1usize;
        let edge_slots = vec![
            0;
            partitions.checked_mul(block_count).ok_or(
                SparseBundleAdjustmentError::DirectSchurIndexCapacityOverflow
            )?
        ];

        #[cfg(feature = "parallel")]
        let logical_point_ranges = {
            let logical_shards = if point_count == 0 {
                0
            } else {
                point_work_offsets
                    .last()
                    .copied()
                    .unwrap_or(0)
                    .div_ceil(LOGICAL_POINT_SHARD_OBSERVATIONS)
                    .clamp(1, MAXIMUM_LOGICAL_POINT_SHARDS)
                    .min(point_count)
            };
            balanced_contiguous_ranges(point_work_offsets, logical_shards)
        };
        #[cfg(feature = "parallel")]
        let active_shards = logical_point_ranges.len().saturating_sub(1).min(threads);
        #[cfg(feature = "parallel")]
        let point_assembly_scratch = (0..active_shards)
            .map(|_| {
                DirectPointAssemblyScratch::new(
                    block_count,
                    base_edge_blocks.len(),
                    reduced_dimension,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "parallel")]
        let pool = try_build_direct_schur_pool(threads)?;

        Ok(Self {
            block_point_offsets,
            block_points,
            point_block_offsets,
            point_blocks,
            #[cfg(feature = "parallel")]
            block_ranges,
            edge_slots,
            block_value_offsets,
            point_count,
            cross_count: point_cross_offsets.last().copied().unwrap_or(0),
            #[cfg(feature = "parallel")]
            dimension: reduced_dimension,
            value_count,
            #[cfg(feature = "parallel")]
            threads,
            #[cfg(feature = "parallel")]
            logical_point_ranges,
            #[cfg(feature = "parallel")]
            reduced_base_blocks: vec![PackedReducedBaseBlock::default(); block_count],
            #[cfg(feature = "parallel")]
            base_edge_values: vec![PackedReducedBaseEdge::default(); base_edge_blocks.len()],
            #[cfg(feature = "parallel")]
            base_edge_blocks,
            #[cfg(feature = "parallel")]
            point_assembly_scratch,
            #[cfg(feature = "parallel")]
            pool,
        })
    }
}
