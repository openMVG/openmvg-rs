// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact sparse-direct Schur factorization and rank checks.

use crate::kernel::RelativePose;
use crate::kernel::multi_view::CameraPoseVariables;
use crate::numerics::sparse::SymmetricPattern;

use super::super::schur::{
    HessianAccumulator, add_block_scalar, add_gram_scalar, add_scaled_pair, parameter_run_end,
};
use super::api::SparseCameraRankAssessment;
use super::direct_assembly::DirectSchurAssemblyWorkspace;
use super::workspace::{SparseScatter, SparseWorkspaceCore, dense_rank_diagnostic_len};

pub(super) struct SparseHessian<'a> {
    pub(super) scatter: &'a SparseScatter,
    pub(super) values: &'a mut [f64],
    pub(super) pattern_complete: bool,
}

impl HessianAccumulator for SparseHessian<'_> {
    fn clear(&mut self) {
        self.values.fill(0.0);
        self.pattern_complete = true;
    }

    #[inline]
    fn add_symmetric(&mut self, row: usize, column: usize, value: f64) {
        match self.scatter.position(row, column) {
            Some(position) => self.values[position] += value,
            None => self.pattern_complete = false,
        }
    }

    fn is_finite(&self) -> bool {
        self.pattern_complete && self.values.iter().all(|value| value.is_finite())
    }

    #[inline]
    fn run_end(&self, parameters: &[usize], start: usize) -> usize {
        let block_remaining = parameters
            .get(start)
            .and_then(|&parameter| self.scatter.parameter_location(parameter))
            .and_then(|(block, local)| self.scatter.block_dimension(block)?.checked_sub(local))
            .unwrap_or(1);
        parameter_run_end(parameters, start, block_remaining)
    }

    #[inline]
    fn add_block(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        lower: bool,
        value: impl FnMut(usize, usize) -> f64,
    ) {
        let (Some(row_run), Some(column_run)) = (
            self.scatter.parameter_run_location(rows),
            self.scatter.parameter_run_location(columns),
        ) else {
            return add_block_scalar(self, rows, columns, lower, value);
        };
        let square_lower = lower && row_run == column_run && rows.len() == columns.len();
        if row_run.0 == column_run.0 && !square_lower {
            return add_block_scalar(self, rows, columns, lower, value);
        }
        if self
            .add_edge_block(row_run, column_run, rows.len(), columns.len(), value)
            .is_none()
        {
            self.pattern_complete = false;
        }
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
        let (Some(row_run), Some(column_run)) = (row_run, column_run) else {
            return add_gram_scalar(self, rows, columns, lower, left, right);
        };
        let added = if row_run.0 != column_run.0 {
            self.add_edge_gram(row_run, column_run, left, right)
        } else if lower {
            // A fixed run length unrolls the triangle into straight-line code.
            match rows.len() {
                1 => self.add_diagonal_gram::<1>(row_run, left),
                2 => self.add_diagonal_gram::<2>(row_run, left),
                3 => self.add_diagonal_gram::<3>(row_run, left),
                4 => self.add_diagonal_gram::<4>(row_run, left),
                5 => self.add_diagonal_gram::<5>(row_run, left),
                6 => self.add_diagonal_gram::<6>(row_run, left),
                _ => return add_gram_scalar(self, rows, columns, lower, left, right),
            }
        } else {
            // A diagonal edge takes only the lower triangle of a whole run.
            return add_gram_scalar(self, rows, columns, lower, left, right);
        };
        if added.is_none() {
            self.pattern_complete = false;
        }
    }
}

impl SparseHessian<'_> {
    /// [`Self::add_edge_block`] for an off-diagonal Gram block of
    /// [`HessianAccumulator::add_gram`]: each stored scalar column segment is
    /// updated in one contiguous pass.
    #[inline]
    fn add_edge_gram(
        &mut self,
        row_run: (usize, usize),
        column_run: (usize, usize),
        left: [&[f64]; 2],
        right: [&[f64]; 2],
    ) -> Option<()> {
        // Stored edges are lower triangular in block order; an upper block is
        // written through its transpose.
        let ((row_block, row_local), (column_block, column_local), left, right) =
            if row_run.0 > column_run.0 {
                (row_run, column_run, left, right)
            } else {
                (column_run, row_run, right, left)
            };
        let starts = self
            .scatter
            .block_edge_column_starts(row_block, column_block)?;
        let starts = starts.get(column_local..column_local + right[0].len())?;
        for ((&start, &first), &second) in starts.iter().zip(right[0]).zip(right[1]) {
            let start = start + row_local;
            let column = self.values.get_mut(start..start + left[0].len())?;
            add_scaled_pair(column, left[0], first, left[1], second);
        }
        Some(())
    }

    /// The lower triangle of the Gram block of a run of `D` parameters from
    /// `local` in `block`, or `None` when the pattern does not hold it.
    #[inline(always)]
    fn add_diagonal_gram<const D: usize>(
        &mut self,
        (block, local): (usize, usize),
        terms: [&[f64]; 2],
    ) -> Option<()> {
        let starts = self.scatter.block_edge_column_starts(block, block)?;
        // Local copies cannot alias the values, so each entry is loaded and
        // stored once.
        let (starts, first, second) = (
            *starts.get(local..)?.first_chunk::<D>()?,
            *terms[0].first_chunk::<D>()?,
            *terms[1].first_chunk::<D>()?,
        );
        // Diagonal edges start each scalar column at its diagonal entry.
        for (j, &start) in starts.iter().enumerate() {
            let column = self.values.get_mut(start..start + D - j)?;
            add_scaled_pair(column, &first[j..], first[j], &second[j..], second[j]);
        }
        Some(())
    }

    /// Add a dense block through the scalar columns of its stored edge, or
    /// `None` when the pattern does not hold the block. Blocks on a diagonal
    /// edge must be square and lower triangular.
    #[inline]
    fn add_edge_block(
        &mut self,
        (row_block, row_local): (usize, usize),
        (column_block, column_local): (usize, usize),
        row_count: usize,
        column_count: usize,
        mut value: impl FnMut(usize, usize) -> f64,
    ) -> Option<()> {
        // Stored edges are lower triangular in block order; an upper block is
        // written through its transpose.
        if row_block < column_block {
            let starts = self
                .scatter
                .block_edge_column_starts(column_block, row_block)?;
            for (i, &start) in starts
                .get(row_local..row_local + row_count)?
                .iter()
                .enumerate()
            {
                let column = self
                    .values
                    .get_mut(start + column_local..start + column_local + column_count)?;
                for (j, target) in column.iter_mut().enumerate() {
                    *target += value(i, j);
                }
            }
            return Some(());
        }
        let starts = self
            .scatter
            .block_edge_column_starts(row_block, column_block)?;
        let starts = starts.get(column_local..column_local + column_count)?;
        if row_block > column_block {
            for (j, &start) in starts.iter().enumerate() {
                let column = self
                    .values
                    .get_mut(start + row_local..start + row_local + row_count)?;
                for (i, target) in column.iter_mut().enumerate() {
                    *target += value(i, j);
                }
            }
        } else {
            // Diagonal edges start each scalar column at its diagonal entry.
            for (j, &start) in starts.iter().enumerate() {
                let column = self.values.get_mut(start..start + row_count - j)?;
                for (offset, target) in column.iter_mut().enumerate() {
                    *target += value(j + offset, j);
                }
            }
        }
        Some(())
    }
}

/// `step' S step` for the assembled reduced system `S`; not finite before the direct pattern
/// is built.
pub(super) fn hessian_quadratic_form(workspace: &SparseWorkspaceCore, step: &[f64]) -> f64 {
    let Some(cholesky) = workspace.cholesky.as_ref() else {
        return f64::NAN;
    };
    let (column_pointers, row_indices) = (
        cholesky.pattern().column_pointers(),
        cholesky.pattern().row_indices(),
    );
    let mut quadratic = 0.0;
    for column in 0..step.len() {
        let range = column_pointers[column]..column_pointers[column + 1];
        for index in range {
            let row = row_indices[index];
            let contribution = workspace.values[index] * step[row] * step[column];
            quadratic += if row == column {
                contribution
            } else {
                2.0 * contribution
            };
        }
    }
    quadratic
}

pub(super) fn factor_camera_system(workspace: &mut SparseWorkspaceCore) -> bool {
    workspace
        .cholesky
        .as_mut()
        .is_some_and(|cholesky| cholesky.factor(&workspace.values))
}

pub(super) fn solve_factored_camera_system(workspace: &mut SparseWorkspaceCore) -> bool {
    for (step, gradient) in workspace.step.iter_mut().zip(&workspace.gradient) {
        *step = -*gradient;
    }
    solve_factored_right_hand_side(workspace)
}

/// Solve the already-factorized camera system for `workspace.step` in place.
///
/// This is shared by the ordinary Newton right-hand side and opt-in selected
/// covariance queries. It performs no symbolic analysis or allocation.
pub(super) fn solve_factored_right_hand_side(workspace: &mut SparseWorkspaceCore) -> bool {
    workspace
        .cholesky
        .as_mut()
        .is_some_and(|cholesky| cholesky.solve_in_place(&mut workspace.step))
}

/// Classify the reduced camera system against its expected gauge nullity.
///
/// `executor` lends its worker pool to the dense spectral certificate, whose
/// result does not depend on the pool.
pub(super) fn assess_camera_rank_with_expected_nullity(
    workspace: &mut SparseWorkspaceCore,
    executor: Option<&DirectSchurAssemblyWorkspace>,
    poses: &[RelativePose],
    expected_gauge_nullity: usize,
) -> SparseCameraRankAssessment {
    match dense_rank_assessment(workspace, executor, expected_gauge_nullity) {
        DenseRankOutcome::Assessed(assessment) => return assessment,
        DenseRankOutcome::Invalid => return SparseCameraRankAssessment::RankDeficientOrIndefinite,
        DenseRankOutcome::Unavailable => {}
    }
    if expected_gauge_nullity == 0 {
        return if factor_camera_system(workspace) {
            SparseCameraRankAssessment::FullRank
        } else {
            SparseCameraRankAssessment::RankDeficientOrIndefinite
        };
    }
    if expected_gauge_nullity != 1 {
        return SparseCameraRankAssessment::RankDeficientOrIndefinite;
    }
    let Some(anchor) = scale_anchor(workspace, poses) else {
        return SparseCameraRankAssessment::RankDeficientOrIndefinite;
    };
    let scale = workspace
        .diagonal
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let Some(slot) = workspace
        .cholesky
        .as_ref()
        .and_then(|cholesky| sparse_position(cholesky.pattern(), anchor, anchor))
    else {
        return SparseCameraRankAssessment::RankDeficientOrIndefinite;
    };
    let original = workspace.values[slot];
    workspace.values[slot] = original + scale;
    let valid = factor_camera_system(workspace);
    workspace.values[slot] = original;
    if valid {
        SparseCameraRankAssessment::OnlyExpectedScaleGauge
    } else {
        SparseCameraRankAssessment::RankDeficientOrIndefinite
    }
}

enum DenseRankOutcome {
    Assessed(SparseCameraRankAssessment),
    Unavailable,
    Invalid,
}

fn dense_rank_assessment(
    workspace: &mut SparseWorkspaceCore,
    executor: Option<&DirectSchurAssemblyWorkspace>,
    expected_gauge_nullity: usize,
) -> DenseRankOutcome {
    let dimension = workspace.layout.dimension();
    if dimension == 0 {
        return DenseRankOutcome::Assessed(SparseCameraRankAssessment::FullRank);
    }
    if workspace.dense_rank_diagnostic.len() != dense_rank_diagnostic_len(dimension) {
        return DenseRankOutcome::Unavailable;
    }
    let Some(cholesky) = workspace.cholesky.as_ref() else {
        return DenseRankOutcome::Invalid;
    };
    let (column_pointers, row_indices) = (
        cholesky.pattern().column_pointers(),
        cholesky.pattern().row_indices(),
    );
    let (matrix, products) = workspace
        .dense_rank_diagnostic
        .split_at_mut(dimension * dimension);
    matrix.fill(0.0);
    for column in 0..dimension {
        let range = column_pointers[column]..column_pointers[column + 1];
        for index in range {
            let row = row_indices[index];
            let value = workspace.values[index];
            matrix[row * dimension + column] = value;
            matrix[column * dimension + row] = value;
        }
    }
    let scale = &mut workspace.step;
    #[cfg(feature = "parallel")]
    let rank = match executor {
        Some(executor) if dimension > PARALLEL_ROW_PASS_MINIMUM_ROWS => {
            executor.install_parallel(|| spectral_rank(matrix, scale, products, dimension, true))
        }
        _ => spectral_rank(matrix, scale, products, dimension, false),
    };
    #[cfg(not(feature = "parallel"))]
    let rank = {
        let _ = executor;
        spectral_rank(matrix, scale, products, dimension, false)
    };
    let Ok(rank) = rank else {
        return DenseRankOutcome::Invalid;
    };
    if rank + expected_gauge_nullity != dimension {
        return DenseRankOutcome::Assessed(SparseCameraRankAssessment::RankDeficientOrIndefinite);
    }
    DenseRankOutcome::Assessed(if expected_gauge_nullity == 0 {
        SparseCameraRankAssessment::FullRank
    } else if expected_gauge_nullity == 1 {
        SparseCameraRankAssessment::OnlyExpectedScaleGauge
    } else {
        SparseCameraRankAssessment::RankDeficientOrIndefinite
    })
}

/// Rank of a symmetric PSD matrix after diagonal equilibration.
///
/// Tiny negative eigenvalues inside the scale-aware roundoff tolerance are
/// treated as zero; materially negative curvature fails closed. The eigenvalue
/// solve is deterministic and uses only the supplied storage: `scale` holds
/// `dimension` values and `products` holds `2 * dimension`. With `parallel`,
/// large Householder passes run on the current rayon pool; the rank does not
/// depend on the pool.
fn spectral_rank(
    matrix: &mut [f64],
    scale: &mut [f64],
    products: &mut [f64],
    dimension: usize,
    parallel: bool,
) -> Result<usize, ()> {
    let diagonal_scale = (0..dimension)
        .map(|index| matrix[index * dimension + index].abs())
        .fold(0.0, f64::max);
    if !(diagonal_scale.is_finite() && diagonal_scale > 0.0) {
        return Err(());
    }
    let diagonal_floor = 64.0 * f64::EPSILON * diagonal_scale;
    for index in 0..dimension {
        let diagonal = matrix[index * dimension + index];
        if !diagonal.is_finite() {
            return Err(());
        }
        scale[index] = diagonal.max(diagonal_floor).sqrt().recip();
    }
    for row in 0..dimension {
        for column in 0..dimension {
            let value = matrix[row * dimension + column] * scale[row] * scale[column];
            if !value.is_finite() {
                return Err(());
            }
            matrix[row * dimension + column] = value;
        }
    }
    symmetric_eigenvalues_in_place(matrix, scale, products, dimension, parallel)?;
    let eigenvalues = &matrix[..dimension];
    let largest = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return Err(());
    }
    let tolerance = 64.0 * f64::EPSILON * dimension as f64 * largest;
    let mut rank = 0;
    for &value in eigenvalues {
        if !value.is_finite() || value < -tolerance {
            return Err(());
        }
        rank += usize::from(value > tolerance);
    }
    Ok(rank)
}

/// Eigenvalues of the symmetric row-major `matrix`, written unsorted to `matrix[..dimension]`.
///
/// Householder reflections reduce the lower triangle to tridiagonal form, then implicit QL
/// iterations with Wilkinson shifts diagonalize it (the eigenvalue-only `tred2`/`tqli` pair).
/// The upper triangle is first overwritten with the lower one, after which each reflection
/// makes a single [`RowPass`] over the leading rows. `off_diagonal` holds `dimension` values
/// and `products` holds `2 * dimension`. With `parallel`, passes over at least
/// `PARALLEL_ROW_PASS_MINIMUM_ROWS` rows are split across the current rayon pool; the
/// eigenvalues are bitwise the same on every pool.
fn symmetric_eigenvalues_in_place(
    matrix: &mut [f64],
    off_diagonal: &mut [f64],
    products: &mut [f64],
    dimension: usize,
    parallel: bool,
) -> Result<(), ()> {
    let n = dimension;
    mirror_lower_triangle(matrix, n);
    // `off_diagonal[i]` couples rows i and i - 1. Each reflection's update
    // A <- A - u q^T - q u^T waits for the next pass, with q in `pending_q`.
    let (mut pending_q, mut next_q) = products.split_at_mut(n);
    let mut pending = false;
    for i in (1..n).rev() {
        let last = i - 1;
        let (block, tail) = matrix.split_at_mut(i * n);
        let (reflector_row, below) = tail.split_at_mut(n);
        // The pending u is row i + 1; row i takes its update before being read.
        let pending_u = if pending { &below[..=i] } else { &below[..0] };
        if pending {
            let (u_i, q_i) = (pending_u[i], pending_q[i]);
            for ((value, &q), &u) in reflector_row[..=i]
                .iter_mut()
                .zip(&pending_q[..=i])
                .zip(pending_u)
            {
                *value -= u_i * q + q_i * u;
            }
        }
        let reflector = &mut reflector_row[..i];
        let scale: f64 = reflector.iter().map(|value| value.abs()).sum();
        let trivial = last == 0 || scale == 0.0;
        let mut h = 0.0;
        if trivial {
            off_diagonal[i] = reflector[last];
        } else {
            let mut norm_squared = 0.0;
            for value in reflector.iter_mut() {
                *value /= scale;
                norm_squared += *value * *value;
            }
            let f = reflector[last];
            let g = if f >= 0.0 {
                -norm_squared.sqrt()
            } else {
                norm_squared.sqrt()
            };
            off_diagonal[i] = scale * g;
            h = norm_squared - f * g;
            reflector[last] = f - g;
        }
        let reflector: &[f64] = reflector;
        let pass = RowPass {
            stride: n,
            size: i,
            update: pending.then(|| (pending_u, &pending_q[..=i])),
            product: (!trivial).then_some(reflector),
        };
        if pass.update.is_some() || pass.product.is_some() {
            pass.run(block, &mut next_q[..i], parallel);
        }
        if trivial {
            pending = false;
            continue;
        }
        // p = A u / h, then q = p - (u^T p / 2h) u.
        let mut coupling = 0.0;
        for (q, &u) in next_q[..i].iter_mut().zip(reflector) {
            *q /= h;
            coupling += *q * u;
        }
        let half_coupling = coupling / (h + h);
        for (q, &u) in next_q[..i].iter_mut().zip(reflector) {
            *q -= half_coupling * u;
        }
        std::mem::swap(&mut pending_q, &mut next_q);
        pending = true;
    }
    tridiagonal_eigenvalues(matrix, off_diagonal, n)
}

/// Overwrite the strict upper triangle of the row-major `matrix` with its lower
/// triangle, in square tiles so the transposed reads stay in cache.
fn mirror_lower_triangle(matrix: &mut [f64], dimension: usize) {
    const TILE: usize = 64;
    for row_tile in (0..dimension).step_by(TILE) {
        for column_tile in (row_tile..dimension).step_by(TILE) {
            for row in row_tile..(row_tile + TILE).min(dimension) {
                for column in column_tile.max(row + 1)..(column_tile + TILE).min(dimension) {
                    matrix[row * dimension + column] = matrix[column * dimension + row];
                }
            }
        }
    }
}

/// Householder passes over at least this many rows are split across the rayon
/// pool; a smaller pass costs less than the fork-join.
#[cfg(feature = "parallel")]
const PARALLEL_ROW_PASS_MINIMUM_ROWS: usize = 128;

/// Upper bound on the row blocks of a parallel pass, so that their claim cells
/// fit in a stack array.
#[cfg(feature = "parallel")]
const MAX_CLAIMED_ROW_BLOCKS: usize = 64;

/// One Householder pass over the leading `size x size` block of a row-major
/// matrix with row stride `stride` whose two triangles are bitwise equal.
///
/// Row `r` first takes the previous reflection's rank-2 `update` on its whole
/// row, which keeps the triangles bitwise equal because `+` and `*` commute.
/// It then forms `(A u)[r]` for the `product` reflector in the summation order
/// of the classic lower-triangle loop: `A[r][r] u[r]`, the terms `k < r`, then
/// `0.0 +` that partial sum followed by the terms `k > r`. Rows are
/// independent, so the result does not depend on how rows are split across
/// threads.
#[derive(Clone, Copy)]
struct RowPass<'a> {
    stride: usize,
    size: usize,
    /// `(u, q)` of the pending update `A <- A - u q^T - q u^T`.
    update: Option<(&'a [f64], &'a [f64])>,
    /// Reflector `u` of the product `A u`.
    product: Option<&'a [f64]>,
}

impl RowPass<'_> {
    /// Run the pass over `block`, the leading `size` rows, writing `A u` to `products`.
    fn run(self, block: &mut [f64], products: &mut [f64], parallel: bool) {
        #[cfg(feature = "parallel")]
        if parallel && self.size >= PARALLEL_ROW_PASS_MINIMUM_ROWS {
            self.run_claimed(block, products);
            return;
        }
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        self.run_rows(block, 0, products);
    }

    /// Run the pass as one contiguous row block per worker of the current pool.
    ///
    /// Each of the `count` tasks claims the block of its own worker index if
    /// still free, else the next free one, so workers tend to keep their rows,
    /// and cache, from pass to pass. Blocks are never refilled and a task
    /// claims at most one, so every task claims exactly one and every row is
    /// processed once. The claim cells live on the stack: the pass does not
    /// allocate.
    #[cfg(feature = "parallel")]
    fn run_claimed(self, block: &mut [f64], products: &mut [f64]) {
        use crossbeam_utils::atomic::AtomicCell;
        use rayon::prelude::*;

        let workers = rayon::current_num_threads().clamp(1, MAX_CLAIMED_ROW_BLOCKS);
        // Whole four-row groups leave the single-row kernel to the last block.
        let block_rows = self.size.div_ceil(workers).next_multiple_of(4);
        let chunks = block
            .chunks_mut(block_rows * self.stride)
            .zip(products.chunks_mut(block_rows));
        // At most `workers` blocks, since `block_rows * workers >= size`.
        let count = chunks.len();
        let cells: [_; MAX_CLAIMED_ROW_BLOCKS] = std::array::from_fn(|_| AtomicCell::new(None));
        for (index, (cell, (rows, products))) in cells.iter().zip(chunks).enumerate() {
            cell.store(Some((index * block_rows, rows, products)));
        }
        let cells = &cells[..count];
        (0..count).into_par_iter().with_max_len(1).for_each(|_| {
            let own = rayon::current_thread_index().unwrap_or(0) % count;
            for offset in 0..count {
                if let Some((start, rows, products)) = cells[(own + offset) % count].take() {
                    self.run_rows(rows, start, products);
                    break;
                }
            }
        });
    }

    /// Run rows `start..start + products.len()` of the pass, stored in `rows`.
    /// Each four-row group is updated, then its four product sums are formed
    /// together as independent chains.
    fn run_rows(self, rows: &mut [f64], start: usize, products: &mut [f64]) {
        let (n, size) = (self.stride, self.size);
        for (group, (rows, products)) in rows
            .chunks_mut(4 * n)
            .zip(products.chunks_mut(4))
            .enumerate()
        {
            let first = start + 4 * group;
            if let Some((u, q)) = self.update {
                for (row, (&u_r, &q_r)) in
                    rows.chunks_mut(n).zip(u[first..].iter().zip(&q[first..]))
                {
                    for ((value, &q_c), &u_c) in
                        row[..size].iter_mut().zip(&q[..size]).zip(&u[..size])
                    {
                        *value -= u_r * q_c + q_r * u_c;
                    }
                }
            }
            if let Some(u) = self.product {
                let u = &u[..size];
                let rows: &[f64] = rows;
                if let [p0, p1, p2, p3] = products {
                    [*p0, *p1, *p2, *p3] = row_products4(
                        [
                            &rows[..size],
                            &rows[n..n + size],
                            &rows[2 * n..2 * n + size],
                            &rows[3 * n..3 * n + size],
                        ],
                        first,
                        u,
                    );
                } else {
                    for ((row, product), r) in rows.chunks(n).zip(products.iter_mut()).zip(first..)
                    {
                        *product = row_product(&row[..size], r, u);
                    }
                }
            }
        }
    }
}

/// `(A u)[r]` from row `r` alone, in [`RowPass`] summation order.
fn row_product(row: &[f64], r: usize, u: &[f64]) -> f64 {
    let mut dot = row[r] * u[r];
    for (&value, &u_k) in row[..r].iter().zip(&u[..r]) {
        dot += value * u_k;
    }
    // Like the classic accumulator started at +0.0, this maps -0.0 to +0.0.
    let mut sum = 0.0 + dot;
    for (&value, &u_k) in row[r + 1..].iter().zip(&u[r + 1..]) {
        sum += value * u_k;
    }
    sum
}

/// [`row_product`] for rows `r0..r0 + 4`, each with its own accumulator.
fn row_products4(rows: [&[f64]; 4], r0: usize, u: &[f64]) -> [f64; 4] {
    let [a0, a1, a2, a3] = rows;
    let (r1, r2, r3) = (r0 + 1, r0 + 2, r0 + 3);
    let mut d0 = a0[r0] * u[r0];
    let mut d1 = a1[r1] * u[r1];
    let mut d2 = a2[r2] * u[r2];
    let mut d3 = a3[r3] * u[r3];
    for ((((&x0, &x1), &x2), &x3), &u_k) in a0[..r0]
        .iter()
        .zip(&a1[..r0])
        .zip(&a2[..r0])
        .zip(&a3[..r0])
        .zip(&u[..r0])
    {
        d0 += x0 * u_k;
        d1 += x1 * u_k;
        d2 += x2 * u_k;
        d3 += x3 * u_k;
    }
    // The 4 x 4 diagonal tile splits each row between its two sums.
    let mut s0 = 0.0 + d0;
    s0 += a0[r1] * u[r1];
    s0 += a0[r2] * u[r2];
    s0 += a0[r3] * u[r3];
    d1 += a1[r0] * u[r0];
    let mut s1 = 0.0 + d1;
    s1 += a1[r2] * u[r2];
    s1 += a1[r3] * u[r3];
    d2 += a2[r0] * u[r0];
    d2 += a2[r1] * u[r1];
    let mut s2 = 0.0 + d2;
    s2 += a2[r3] * u[r3];
    d3 += a3[r0] * u[r0];
    d3 += a3[r1] * u[r1];
    d3 += a3[r2] * u[r2];
    let mut s3 = 0.0 + d3;
    let tail = r0 + 4;
    for ((((&x0, &x1), &x2), &x3), &u_k) in a0[tail..]
        .iter()
        .zip(&a1[tail..])
        .zip(&a2[tail..])
        .zip(&a3[tail..])
        .zip(&u[tail..])
    {
        s0 += x0 * u_k;
        s1 += x1 * u_k;
        s2 += x2 * u_k;
        s3 += x3 * u_k;
    }
    [s0, s1, s2, s3]
}

/// Eigenvalues of the tridiagonal matrix with diagonal `matrix[i * dimension + i]`
/// and `off_diagonal[i]` coupling rows `i` and `i - 1`, written unsorted to
/// `matrix[..dimension]`.
fn tridiagonal_eigenvalues(
    matrix: &mut [f64],
    off_diagonal: &mut [f64],
    dimension: usize,
) -> Result<(), ()> {
    const MAX_QL_ITERATIONS_PER_EIGENVALUE: usize = 64;
    let n = dimension;
    let off_diagonal = &mut off_diagonal[..n];
    for i in 1..n {
        matrix[i] = matrix[i * n + i];
    }
    let diagonal = &mut matrix[..n];
    for i in 1..n {
        off_diagonal[i - 1] = off_diagonal[i];
    }
    if let Some(last) = off_diagonal.last_mut() {
        *last = 0.0;
    }
    for start in 0..n {
        let mut iterations = 0;
        loop {
            let mut end = start;
            while end + 1 < n {
                let magnitude = diagonal[end].abs() + diagonal[end + 1].abs();
                if off_diagonal[end].abs() <= f64::EPSILON * magnitude {
                    break;
                }
                end += 1;
            }
            if end == start {
                break;
            }
            if iterations == MAX_QL_ITERATIONS_PER_EIGENVALUE {
                return Err(());
            }
            iterations += 1;
            let shift = (diagonal[start + 1] - diagonal[start]) / (2.0 * off_diagonal[start]);
            let radius = shift.hypot(1.0);
            let signed_radius = if shift >= 0.0 { radius } else { -radius };
            let mut g =
                diagonal[end] - diagonal[start] + off_diagonal[start] / (shift + signed_radius);
            let (mut sine, mut cosine, mut correction) = (1.0, 1.0, 0.0);
            let mut deflated = false;
            for row in (start..end).rev() {
                let f = sine * off_diagonal[row];
                let b = cosine * off_diagonal[row];
                let r = f.hypot(g);
                off_diagonal[row + 1] = r;
                if r == 0.0 {
                    diagonal[row + 1] -= correction;
                    off_diagonal[end] = 0.0;
                    deflated = true;
                    break;
                }
                sine = f / r;
                cosine = g / r;
                g = diagonal[row + 1] - correction;
                let rotated = (diagonal[row] - g) * sine + 2.0 * cosine * b;
                correction = sine * rotated;
                diagonal[row + 1] = g + correction;
                g = cosine * rotated - b;
            }
            if deflated {
                continue;
            }
            diagonal[start] -= correction;
            off_diagonal[start] = g;
            off_diagonal[end] = 0.0;
        }
    }
    Ok(())
}

fn scale_anchor(workspace: &SparseWorkspaceCore, poses: &[RelativePose]) -> Option<usize> {
    let mut best = None;
    let mut magnitude = 0.0;
    for (camera, pose) in poses.iter().enumerate().skip(1) {
        let dimension = workspace.layout.camera_dimensions()[camera];
        let translation_offset = match workspace.signature.pose_variables() {
            CameraPoseVariables::RotationAndTranslation if dimension >= 6 => 3,
            CameraPoseVariables::TranslationOnly if dimension >= 3 => 0,
            CameraPoseVariables::RotationAndTranslation | CameraPoseVariables::TranslationOnly => {
                continue;
            }
        };
        if dimension < translation_offset + 3 {
            continue;
        }
        let offset = workspace.layout.camera_offsets()[camera];
        for coordinate in 0..3 {
            let candidate = pose.translation[coordinate].abs();
            if candidate > magnitude {
                magnitude = candidate;
                best = Some(offset + translation_offset + coordinate);
            }
        }
    }
    best.filter(|_| magnitude > 0.0)
}

fn sparse_position(pattern: &SymmetricPattern, row: usize, column: usize) -> Option<usize> {
    let range = pattern.column_pointers()[column]..pattern.column_pointers()[column + 1];
    pattern.row_indices()[range.clone()]
        .binary_search(&row)
        .ok()
        .map(|local| range.start + local)
}

pub(super) fn add_sparse_symmetric(
    workspace: &mut SparseWorkspaceCore,
    row: usize,
    column: usize,
    value: f64,
) -> bool {
    let Some(position) = workspace.scatter.position(row, column) else {
        return false;
    };
    workspace.values[position] += value;
    true
}

#[cfg(test)]
mod tests {
    use super::{spectral_rank, symmetric_eigenvalues_in_place, tridiagonal_eigenvalues};

    /// Symmetric matrix `F S F^T` with `F` random `dimension x rank` and `S` a sign pattern.
    fn random_symmetric(
        dimension: usize,
        rank: usize,
        indefinite: bool,
        state: &mut u64,
    ) -> nalgebra::DMatrix<f64> {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        };
        let factor = nalgebra::DMatrix::from_fn(dimension, rank, |_, _| next());
        let signs = nalgebra::DVector::from_fn(rank, |index, _| {
            if indefinite && index % 3 == 2 {
                -1.0
            } else {
                1.0
            }
        });
        &factor * nalgebra::DMatrix::from_diagonal(&signs) * factor.transpose()
    }

    /// Eigenvalue bits from the solver under test.
    fn eigenvalue_bits(matrix: &[f64], dimension: usize, parallel: bool) -> Result<Vec<u64>, ()> {
        let mut storage = matrix.to_vec();
        let mut off_diagonal = vec![0.0; dimension];
        let mut products = vec![0.0; 2 * dimension];
        symmetric_eigenvalues_in_place(
            &mut storage,
            &mut off_diagonal,
            &mut products,
            dimension,
            parallel,
        )?;
        Ok(storage[..dimension]
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    /// Eigenvalue bits from the classic lower-triangle `tred2` loop that the
    /// row passes replace, kept as their bitwise oracle.
    fn lower_triangle_eigenvalue_bits(matrix: &[f64], n: usize) -> Result<Vec<u64>, ()> {
        let mut matrix = matrix.to_vec();
        let mut off_diagonal = vec![0.0; n];
        for i in (1..n).rev() {
            let last = i - 1;
            let (block, reflector_row) = matrix.split_at_mut(i * n);
            let reflector = &mut reflector_row[..i];
            let scale: f64 = reflector.iter().map(|value| value.abs()).sum();
            if last == 0 || scale == 0.0 {
                off_diagonal[i] = reflector[last];
                continue;
            }
            let mut norm_squared = 0.0;
            for value in reflector.iter_mut() {
                *value /= scale;
                norm_squared += *value * *value;
            }
            let f = reflector[last];
            let g = if f >= 0.0 {
                -norm_squared.sqrt()
            } else {
                norm_squared.sqrt()
            };
            off_diagonal[i] = scale * g;
            let h = norm_squared - f * g;
            reflector[last] = f - g;
            off_diagonal[..i].fill(0.0);
            for row in 0..i {
                let entries = &block[row * n..row * n + row + 1];
                let mut dot = entries[row] * reflector[row];
                for column in 0..row {
                    dot += entries[column] * reflector[column];
                    off_diagonal[column] += entries[column] * reflector[row];
                }
                off_diagonal[row] += dot;
            }
            let mut coupling = 0.0;
            for row in 0..i {
                off_diagonal[row] /= h;
                coupling += off_diagonal[row] * reflector[row];
            }
            let half_coupling = coupling / (h + h);
            for row in 0..i {
                off_diagonal[row] -= half_coupling * reflector[row];
            }
            for row in 0..i {
                let (u_row, q_row) = (reflector[row], off_diagonal[row]);
                let entries = &mut block[row * n..row * n + row + 1];
                for column in 0..=row {
                    entries[column] -= u_row * off_diagonal[column] + q_row * reflector[column];
                }
            }
        }
        tridiagonal_eigenvalues(&mut matrix, &mut off_diagonal, n)?;
        Ok(matrix[..n].iter().map(|value| value.to_bits()).collect())
    }

    /// Row-major matrices that reach every reflection path: dense indefinite,
    /// rank deficient, block diagonal and with a zero row and column (exactly
    /// zero reflectors between nontrivial ones), diagonal (no pass at all),
    /// sparse with signed zeros, and a `-0.0` product sum. Only the lower
    /// triangle is meaningful; the strict upper triangle is NaN, which no
    /// solver may read.
    fn oracle_matrices(n: usize, state: &mut u64) -> Vec<Vec<f64>> {
        let dense = random_symmetric(n, n, true, state);
        let deficient = random_symmetric(n, n.div_ceil(2), false, state);
        let (half, hole) = (n / 2, n / 3);
        let block = nalgebra::DMatrix::from_fn(n, n, |r, c| {
            if (r < half) == (c < half) {
                dense[(r, c)]
            } else {
                0.0
            }
        });
        let holed = nalgebra::DMatrix::from_fn(n, n, |r, c| {
            if r == hole || c == hole {
                0.0
            } else {
                dense[(r, c)]
            }
        });
        let diagonal = nalgebra::DMatrix::from_diagonal(&dense.diagonal());
        let sparse = nalgebra::DMatrix::from_fn(n, n, |r, c| match (r + c) % 5 {
            _ if r == c => dense[(r, c)],
            0 => dense[(r, c)],
            1 | 3 => -0.0,
            _ => 0.0,
        });
        // A zero row 0 whose first product sums only -0.0 terms: the classic
        // accumulator, started at +0.0, returns +0.0.
        let negative_zero_sum = nalgebra::DMatrix::from_fn(n, n, |r, c| match (r, c) {
            (0, 0) => -0.0,
            (0, k) | (k, 0) if k + 1 == n => 0.0,
            (0, k) | (k, 0) => 0.0_f64.copysign(-dense[(n - 1, k)]),
            _ => dense[(r, c)],
        });
        [
            dense,
            deficient,
            block,
            holed,
            diagonal,
            sparse,
            negative_zero_sum,
        ]
        .iter()
        .map(|matrix| {
            let mut storage: Vec<f64> = matrix.transpose().iter().copied().collect();
            for row in 0..n {
                storage[row * n + row + 1..(row + 1) * n].fill(f64::NAN);
            }
            storage
        })
        .collect()
    }

    #[test]
    fn row_passes_reproduce_the_lower_triangle_oracle_bitwise() {
        let mut state = 0x3c6e_f372_fe94_f82b;
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 17, 31, 64, 65, 130] {
            for (kind, matrix) in oracle_matrices(n, &mut state).iter().enumerate() {
                let expected = lower_triangle_eigenvalue_bits(matrix, n);
                assert!(expected.is_ok(), "dimension {n}, matrix {kind}");
                assert_eq!(
                    eigenvalue_bits(matrix, n, false),
                    expected,
                    "dimension {n}, matrix {kind}"
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_row_passes_are_bitwise_identical_on_every_pool() {
        // 65 workers exceed the block cap; 257 rows fill all 64 blocks.
        let pools: Vec<_> = [1, 2, 3, 7, 65]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        let mut state = 0xa54f_f53a_5f1d_36f1;
        for n in [129, 257] {
            let matrices = oracle_matrices(n, &mut state);
            // Dense, and block diagonal for a nontrivial pass after a trivial one.
            for (kind, matrix) in [(0, &matrices[0]), (2, &matrices[2])] {
                let expected = lower_triangle_eigenvalue_bits(matrix, n);
                assert!(expected.is_ok(), "dimension {n}, matrix {kind}");
                for pool in &pools {
                    let threads = pool.current_num_threads();
                    let actual = pool.install(|| eigenvalue_bits(matrix, n, true));
                    assert_eq!(
                        actual, expected,
                        "dimension {n}, matrix {kind}, {threads} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn in_place_eigenvalues_match_a_reference_decomposition() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        for (dimension, rank) in [(1, 1), (2, 1), (3, 3), (8, 5), (40, 33), (65, 65)] {
            for indefinite in [false, true] {
                let matrix = random_symmetric(dimension, rank, indefinite, &mut state);
                let mut expected: Vec<f64> =
                    matrix.symmetric_eigenvalues().iter().copied().collect();
                let mut storage: Vec<f64> = matrix.iter().copied().collect();
                let mut off_diagonal = vec![0.0; dimension];
                let mut products = vec![0.0; 2 * dimension];
                symmetric_eigenvalues_in_place(
                    &mut storage,
                    &mut off_diagonal,
                    &mut products,
                    dimension,
                    false,
                )
                .unwrap();
                let mut actual = storage[..dimension].to_vec();
                expected.sort_by(f64::total_cmp);
                actual.sort_by(f64::total_cmp);
                let magnitude = expected
                    .iter()
                    .fold(0.0_f64, |largest, value| largest.max(value.abs()));
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() <= 1e-12 * magnitude,
                        "dimension {dimension}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn dense_rank_diagnostic_recovers_rank_of_a_generic_semidefinite_matrix() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let matrix = random_symmetric(40, 33, false, &mut state);
        let mut storage: Vec<f64> = matrix.iter().copied().collect();
        let (mut scale, mut products) = (vec![0.0; 40], vec![0.0; 80]);
        assert_eq!(
            spectral_rank(&mut storage, &mut scale, &mut products, 40, false),
            Ok(33)
        );
    }

    #[test]
    fn dense_rank_diagnostic_tolerates_roundoff_but_rejects_real_indefiniteness() {
        let (mut scale, mut products) = (vec![0.0; 3], vec![0.0; 6]);
        let mut rank =
            |mut matrix: Vec<f64>| spectral_rank(&mut matrix, &mut scale, &mut products, 3, false);
        let roundoff = vec![1.0, 1.0, 0.0, 1.0, 1.0 - f64::EPSILON, 0.0, 0.0, 0.0, 0.5];
        assert_eq!(rank(roundoff), Ok(2));

        let deficient = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(rank(deficient), Ok(1));

        let indefinite = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1e-3];
        assert!(rank(indefinite).is_err());

        // 105 I - v v^T has the known null vector v = [1, 2, 10].
        // This checks that the eigen solver recovers rank even when the null is
        // not aligned with a coordinate axis.
        let rotated_null = vec![104.0, -2.0, -10.0, -2.0, 101.0, -20.0, -10.0, -20.0, 5.0];
        assert_eq!(rank(rotated_null), Ok(2));
    }
}
