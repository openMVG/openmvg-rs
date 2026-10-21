// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Column-owned numeric materialization for cached point elimination.
//!
//! Reduced block dimensions are dispatched once per block into fixed-size
//! kernels, so every scalar row loop has a compile-time trip count and
//! vectorizes without runtime-length tails on any target.

use crate::kernel::point_block::PackedPointBlock;

use super::direct_assembly::{PointBlockRef, PointCrossRef};
use super::direct_assembly_policy::MAX_REDUCED_BLOCK_DIMENSION;
use super::workspace::SparseScatter;

// The fixed-size kernel dispatch below covers every reduced block dimension.
const _: () = assert!(MAX_REDUCED_BLOCK_DIMENSION == 6);

/// One retained `V^-1 W` row, widened to `f64` before every product.
pub(super) trait CrossRow: Copy + Sync {
    fn widen(self) -> [f64; 3];
}

impl CrossRow for [f64; 3] {
    #[inline(always)]
    fn widen(self) -> [f64; 3] {
        self
    }
}

impl CrossRow for [f32; 3] {
    #[inline(always)]
    fn widen(self) -> [f64; 3] {
        self.map(f64::from)
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_partitions<T: CrossRow>(
    block_ranges: &[usize],
    block_value_offsets: &[usize],
    block_point_offsets: &[usize],
    block_points: &[PointCrossRef],
    point_block_offsets: &[usize],
    point_blocks: &[PointBlockRef],
    scatter: &SparseScatter,
    edge_slots: &mut [u32],
    values: &mut [f64],
    value_start: usize,
    point_hessians: &[PackedPointBlock],
    inverse_crosses: &[T],
) -> bool {
    let partitions = block_ranges.len().saturating_sub(1);
    if partitions <= 1 {
        return materialize_block_range(
            block_ranges.first().copied().unwrap_or(0),
            block_ranges.last().copied().unwrap_or(0),
            block_point_offsets,
            block_points,
            point_block_offsets,
            point_blocks,
            scatter,
            edge_slots,
            values,
            value_start,
            point_hessians,
            inverse_crosses,
        );
    }

    let middle = partitions / 2;
    let split_block = block_ranges[middle];
    let split_value = block_value_offsets[split_block];
    let Some(local_split) = split_value.checked_sub(value_start) else {
        return false;
    };
    if local_split > values.len() {
        return false;
    }
    let (left_values, right_values) = values.split_at_mut(local_split);
    // Every partition owns one edge slot per block.
    let Some((left_slots, right_slots)) =
        edge_slots.split_at_mut_checked(middle * scatter.block_count())
    else {
        return false;
    };
    let (left, right) = rayon::join(
        || {
            materialize_partitions(
                &block_ranges[..=middle],
                block_value_offsets,
                block_point_offsets,
                block_points,
                point_block_offsets,
                point_blocks,
                scatter,
                left_slots,
                left_values,
                value_start,
                point_hessians,
                inverse_crosses,
            )
        },
        || {
            materialize_partitions(
                &block_ranges[middle..],
                block_value_offsets,
                block_point_offsets,
                block_points,
                point_block_offsets,
                point_blocks,
                scatter,
                right_slots,
                right_values,
                split_value,
                point_hessians,
                inverse_crosses,
            )
        },
    );
    left && right
}

/// Materialize the block columns `block_start..block_end`.
///
/// `edge_slots` holds one entry per block. Each column scatters the local
/// index of its edges there, so a coupled row block finds its edge in
/// constant time; a slot left by another column names a different row and is
/// rejected.
#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_block_range<T: CrossRow>(
    block_start: usize,
    block_end: usize,
    block_point_offsets: &[usize],
    block_points: &[PointCrossRef],
    point_block_offsets: &[usize],
    point_blocks: &[PointBlockRef],
    scatter: &SparseScatter,
    edge_slots: &mut [u32],
    values: &mut [f64],
    value_start: usize,
    point_hessians: &[PackedPointBlock],
    inverse_crosses: &[T],
) -> bool {
    for column_block in block_start..block_end {
        let Some(column_dimension) = scatter.block_dimension(column_block) else {
            return false;
        };
        if column_dimension == 0 {
            continue;
        }
        if column_dimension > MAX_REDUCED_BLOCK_DIMENSION {
            return false;
        }
        // The diagonal edge leads every stored column.
        let Some((edge_rows, edge_starts)) = scatter.block_column_edges(column_block) else {
            return false;
        };
        if edge_rows.first().map(|&row| row as usize) != Some(column_block)
            || edge_starts.len() != edge_rows.len() * column_dimension
        {
            return false;
        }
        let diagonal_starts = &edge_starts[..column_dimension];
        for (local, &row) in (0..).zip(edge_rows) {
            let Some(slot) = edge_slots.get_mut(row as usize) else {
                return false;
            };
            *slot = local;
        }
        let Some(&reference_start) = block_point_offsets.get(column_block) else {
            return false;
        };
        let Some(&reference_end) = block_point_offsets.get(column_block + 1) else {
            return false;
        };
        let Some(references) = block_points.get(reference_start..reference_end) else {
            return false;
        };
        for reference in references {
            let point = reference.point as usize;
            let Some(point_hessian) = point_hessians.get(point) else {
                return false;
            };
            let Some(&point_start) = point_block_offsets.get(point) else {
                return false;
            };
            let Some(&point_end) = point_block_offsets.get(point + 1) else {
                return false;
            };
            let Some(active_blocks) = point_blocks.get(point_start..point_end) else {
                return false;
            };
            let column_offset = reference.cross_offset as usize;
            let Some(column_crosses) =
                inverse_crosses.get(column_offset..column_offset + column_dimension)
            else {
                return false;
            };
            // Same products as `bilinear`, with each point-inverse product
            // shared by the diagonal block and every coupled row block.
            let mut column_products = [[0.0; 3]; MAX_REDUCED_BLOCK_DIMENSION];
            for (product, cross) in column_products.iter_mut().zip(column_crosses) {
                *product = point_hessian.multiply(&cross.widen());
            }
            let column_products = &column_products[..column_dimension];
            if !subtract_diagonal(
                values,
                value_start,
                diagonal_starts,
                column_crosses,
                column_products,
            ) {
                return false;
            }
            // Active blocks ascend strictly, so the row blocks below the
            // diagonal follow the column block itself.
            let first_row = active_blocks.partition_point(|row| row.block as usize <= column_block);
            for row_reference in &active_blocks[first_row..] {
                let row_block = row_reference.block as usize;
                let Some(row_dimension) = scatter.block_dimension(row_block) else {
                    return false;
                };
                let Some(&local) = edge_slots.get(row_block) else {
                    return false;
                };
                let local = local as usize;
                if edge_rows.get(local) != Some(&row_reference.block) {
                    return false;
                }
                let Some(row_starts) =
                    edge_starts.get(local * column_dimension..(local + 1) * column_dimension)
                else {
                    return false;
                };
                let row_offset = row_reference.cross_offset as usize;
                let Some(row_crosses) = inverse_crosses.get(row_offset..row_offset + row_dimension)
                else {
                    return false;
                };
                if !subtract_edge(
                    values,
                    value_start,
                    row_starts,
                    row_crosses,
                    column_products,
                ) {
                    return false;
                }
            }
        }
    }
    // Non-finite values persist, so the final scan catches them.
    values.iter().all(|value| value.is_finite())
}

/// Subtract one point's coupling from the stored lower triangle of a diagonal
/// block whose columns start at `starts`.
fn subtract_diagonal<T: CrossRow>(
    values: &mut [f64],
    value_start: usize,
    starts: &[usize],
    crosses: &[T],
    products: &[[f64; 3]],
) -> bool {
    match crosses.len() {
        1 => subtract_triangle::<1, T>(values, value_start, starts, crosses, products),
        2 => subtract_triangle::<2, T>(values, value_start, starts, crosses, products),
        3 => subtract_triangle::<3, T>(values, value_start, starts, crosses, products),
        4 => subtract_triangle::<4, T>(values, value_start, starts, crosses, products),
        5 => subtract_triangle::<5, T>(values, value_start, starts, crosses, products),
        6 => subtract_triangle::<6, T>(values, value_start, starts, crosses, products),
        _ => false,
    }
}

/// Subtract one point's coupling from an off-diagonal block with one row per
/// cross row and one column per point-inverse product.
fn subtract_edge<T: CrossRow>(
    values: &mut [f64],
    value_start: usize,
    starts: &[usize],
    rows: &[T],
    products: &[[f64; 3]],
) -> bool {
    match rows.len() {
        1 => subtract_rectangle::<1, T>(values, value_start, starts, rows, products),
        2 => subtract_rectangle::<2, T>(values, value_start, starts, rows, products),
        3 => subtract_rectangle::<3, T>(values, value_start, starts, rows, products),
        4 => subtract_rectangle::<4, T>(values, value_start, starts, rows, products),
        5 => subtract_rectangle::<5, T>(values, value_start, starts, rows, products),
        6 => subtract_rectangle::<6, T>(values, value_start, starts, rows, products),
        _ => false,
    }
}

#[inline(always)]
fn subtract_rectangle<const R: usize, T: CrossRow>(
    values: &mut [f64],
    value_start: usize,
    starts: &[usize],
    rows: &[T],
    products: &[[f64; 3]],
) -> bool {
    let Some(rows) = rows.first_chunk::<R>() else {
        return false;
    };
    let [x, y, z] = components(rows);
    for (&start, product) in starts.iter().zip(products) {
        let Some(targets) = start
            .checked_sub(value_start)
            .and_then(|start| values.get_mut(start..))
            .and_then(|targets| targets.first_chunk_mut::<R>())
        else {
            return false;
        };
        for (((target, x), y), z) in targets.iter_mut().zip(x).zip(y).zip(z) {
            *target -= x * product[0] + y * product[1] + z * product[2];
        }
    }
    true
}

#[inline(always)]
fn subtract_triangle<const D: usize, T: CrossRow>(
    values: &mut [f64],
    value_start: usize,
    starts: &[usize],
    crosses: &[T],
    products: &[[f64; 3]],
) -> bool {
    let (Some(crosses), Some(starts), Some(products)) = (
        crosses.first_chunk::<D>(),
        starts.first_chunk::<D>(),
        products.first_chunk::<D>(),
    ) else {
        return false;
    };
    let [x, y, z] = components(crosses);
    // Column `j` of a stored diagonal block holds rows `j..D`.
    for (column, (&start, product)) in starts.iter().zip(products).enumerate() {
        let Some(targets) = start
            .checked_sub(value_start)
            .and_then(|start| values.get_mut(start..))
            .and_then(|targets| targets.get_mut(..D - column))
        else {
            return false;
        };
        for (((target, x), y), z) in targets
            .iter_mut()
            .zip(&x[column..])
            .zip(&y[column..])
            .zip(&z[column..])
        {
            *target -= x * product[0] + y * product[1] + z * product[2];
        }
    }
    true
}

/// Transpose rows into coordinate lanes so each lane product is one SIMD
/// operation over the rows.
#[inline(always)]
fn components<const N: usize, T: CrossRow>(rows: &[T; N]) -> [[f64; N]; 3] {
    let mut lanes = [[0.0; N]; 3];
    for (local, row) in rows.iter().enumerate() {
        let [x, y, z] = row.widen();
        lanes[0][local] = x;
        lanes[1][local] = y;
        lanes[2][local] = z;
    }
    lanes
}
