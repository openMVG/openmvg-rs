// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic sparse Cholesky factorization of symmetric positive definite systems.
//!
//! Bundle adjustment, its observability probe, rotation averaging, and global positioning all
//! factor a sparse symmetric positive definite system in compressed columns. [`SymmetricPattern`]
//! holds the structure; [`SparseCholesky`] takes it, orders it once by approximate minimum degree
//! and analyzes its symbolic factor, and every later factorization and solve reuses that analysis
//! and its storage without allocating. Only entries on or below the diagonal are read.
//!
//! Every factorization and solve runs on one thread (`Par::Seq`): faer's Rayon factorization can
//! change the result bits with the worker count, and these solves must be bit-identical across
//! thread counts. A solve's result also depends on its number of right-hand sides, so a caller
//! keeps that number fixed rather than batching or splitting columns.
//!
//! ```text
//! pattern -> AMD ordering + symbolic factor (once) -> reserve factor and scratch (once)
//! per system: values -> numeric LLT -> triangular solves in place -> finite check
//! ```
//!
//! This module is shared by the workspace crates and carries no semver guarantee.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LltRef, SymbolicCholesky, SymmetricOrdering,
    factorize_symbolic_cholesky,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat};
use faer::{Conj, MatMut, Par, Side};

/// Structure of a square sparse matrix in compressed columns with sorted rows.
///
/// Only entries on or below the diagonal take part in a factorization, so a caller may store
/// the lower triangle alone or both triangles.
pub struct SymmetricPattern {
    inner: SymbolicSparseColMat<usize>,
}

impl SymmetricPattern {
    /// The `dimension x dimension` pattern whose column `c` holds the rows
    /// `row_indices[column_pointers[c]..column_pointers[c + 1]]`.
    ///
    /// `None` unless there are `dimension + 1` nondecreasing column pointers starting at zero
    /// and ending at the row count, and every column's rows are strictly increasing and below
    /// `dimension`.
    #[must_use]
    pub fn new(
        dimension: usize,
        column_pointers: Vec<usize>,
        row_indices: Vec<usize>,
    ) -> Option<Self> {
        let valid = column_pointers.len() == dimension.checked_add(1)?
            && column_pointers.first() == Some(&0)
            && column_pointers.last() == Some(&row_indices.len())
            && column_pointers
                .iter()
                .all(|&pointer| pointer <= row_indices.len())
            && column_pointers.windows(2).all(|bounds| {
                bounds[0] <= bounds[1]
                    && row_indices[bounds[0]..bounds[1]]
                        .windows(2)
                        .all(|rows| rows[0] < rows[1])
                    && row_indices[bounds[0]..bounds[1]]
                        .last()
                        .is_none_or(|&row| row < dimension)
            });
        valid.then(|| Self {
            inner: SymbolicSparseColMat::new_checked(
                dimension,
                dimension,
                column_pointers,
                None,
                row_indices,
            ),
        })
    }

    /// Number of rows and columns.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.inner.ncols()
    }

    /// Start of each column in [`Self::row_indices`], plus the total count.
    #[must_use]
    pub fn column_pointers(&self) -> &[usize] {
        self.inner.col_ptr()
    }

    /// Row of every stored entry, column by column.
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        self.inner.row_idx()
    }

    /// Number of stored entries, the length of a value array on this pattern.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.row_indices().len()
    }
}

/// A pattern with its symbolic Cholesky analysis and reusable numeric storage.
pub struct SparseCholesky {
    pattern: SymmetricPattern,
    symbolic: SymbolicCholesky<usize>,
    factor: Vec<f64>,
    scratch: Option<MemBuffer>,
    right_hand_sides: usize,
    factored: bool,
}

impl SparseCholesky {
    /// Order `pattern` by approximate minimum degree and analyze its symbolic factor; `None`
    /// when faer rejects the pattern.
    #[must_use]
    pub fn analyze(pattern: SymmetricPattern) -> Option<Self> {
        Self::analyze_with(pattern, CholeskySymbolicParams::default())
    }

    /// [`Self::analyze`] with a simplicial factor. A positive-definiteness certificate needs no
    /// supernodes, whose dense kernels fill faer's large thread-local packing cache.
    #[must_use]
    pub fn analyze_simplicial(pattern: SymmetricPattern) -> Option<Self> {
        Self::analyze_with(
            pattern,
            CholeskySymbolicParams {
                supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
                ..Default::default()
            },
        )
    }

    fn analyze_with(pattern: SymmetricPattern, params: CholeskySymbolicParams) -> Option<Self> {
        let symbolic = factorize_symbolic_cholesky(
            pattern.inner.as_ref(),
            Side::Lower,
            SymmetricOrdering::Amd,
            params,
        )
        .ok()?;
        Some(Self {
            pattern,
            symbolic,
            factor: Vec::new(),
            scratch: None,
            right_hand_sides: 0,
            factored: false,
        })
    }

    /// The analyzed pattern; [`Self::factor`] takes values in its order.
    #[must_use]
    pub fn pattern(&self) -> &SymmetricPattern {
        &self.pattern
    }

    /// Number of entries of the Cholesky factor, known before any storage is reserved.
    #[must_use]
    pub fn factor_value_count(&self) -> usize {
        self.symbolic.len_val()
    }

    /// Allocate the factor and the scratch for solves of up to `right_hand_sides` columns;
    /// `false` when the scratch cannot be allocated. Storage already large enough is kept.
    #[must_use]
    pub fn reserve(&mut self, right_hand_sides: usize) -> bool {
        if self.scratch.is_some() && right_hand_sides <= self.right_hand_sides {
            return true;
        }
        let Ok(scratch) = MemBuffer::try_new(StackReq::or(
            self.symbolic
                .factorize_numeric_llt_scratch::<f64>(Par::Seq, Default::default()),
            self.symbolic
                .solve_in_place_scratch::<f64>(right_hand_sides, Par::Seq),
        )) else {
            return false;
        };
        self.factor = vec![0.0; self.symbolic.len_val()];
        self.scratch = Some(scratch);
        self.right_hand_sides = right_hand_sides;
        self.factored = false;
        true
    }

    /// Bytes held by the factor and the scratch.
    #[must_use]
    pub fn reserved_bytes(&self) -> usize {
        self.factor.capacity() * std::mem::size_of::<f64>()
            + self.scratch.as_ref().map_or(0, |buffer| buffer.len())
    }

    /// Factor the matrix with `values` on [`Self::pattern`].
    ///
    /// `false` when the storage is not reserved, `values` does not match the pattern in length,
    /// or the matrix is not numerically positive definite.
    #[must_use]
    pub fn factor(&mut self, values: &[f64]) -> bool {
        self.factored = false;
        if values.len() != self.pattern.value_count() {
            return false;
        }
        let Some(scratch) = self.scratch.as_mut() else {
            return false;
        };
        self.factored = self
            .symbolic
            .factorize_numeric_llt(
                &mut self.factor,
                SparseColMatRef::new(self.pattern.inner.as_ref(), values),
                Side::Lower,
                Default::default(),
                Par::Seq,
                MemStack::new(scratch),
                Default::default(),
            )
            .is_ok();
        self.factored
    }

    /// Overwrite `right_hand_sides`, column-major columns of the pattern's dimension, with the
    /// solution of the last successful factorization.
    ///
    /// `false` without one, when the column count exceeds the reserved one, or when the
    /// solution is not finite.
    #[must_use]
    pub fn solve_in_place(&mut self, right_hand_sides: &mut [f64]) -> bool {
        let dimension = self.pattern.dimension();
        let Some(columns) = right_hand_sides.len().checked_div(dimension) else {
            return right_hand_sides.is_empty() && self.factored;
        };
        if !self.factored
            || columns * dimension != right_hand_sides.len()
            || columns > self.right_hand_sides
        {
            return false;
        }
        let Some(scratch) = self.scratch.as_mut() else {
            return false;
        };
        LltRef::new(&self.symbolic, &self.factor).solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(right_hand_sides, dimension, columns),
            Par::Seq,
            MemStack::new(scratch),
        );
        right_hand_sides.iter().all(|value| value.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::{SparseCholesky, SymmetricPattern};

    /// Lower triangle of the 3x3 matrix [[4, 1, 0], [1, 3, 1], [0, 1, 2]].
    fn lower() -> (SymmetricPattern, Vec<f64>) {
        let pattern = SymmetricPattern::new(3, vec![0, 2, 4, 5], vec![0, 1, 1, 2, 2]);
        (
            pattern.expect("valid pattern"),
            vec![4.0, 1.0, 3.0, 1.0, 2.0],
        )
    }

    fn residual(solution: &[f64], rhs: &[f64]) -> f64 {
        let product = [
            4.0 * solution[0] + solution[1],
            solution[0] + 3.0 * solution[1] + solution[2],
            solution[1] + 2.0 * solution[2],
        ];
        product
            .iter()
            .zip(rhs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn solves_every_reserved_right_hand_side() {
        let (pattern, values) = lower();
        let mut cholesky = SparseCholesky::analyze(pattern).expect("analysis");
        assert!(cholesky.reserve(2));
        assert!(cholesky.factor(&values));
        let mut rhs = vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
        assert!(cholesky.solve_in_place(&mut rhs));
        assert!(residual(&rhs[..3], &[1.0, 2.0, 3.0]) < 1e-14);
        assert!(residual(&rhs[3..], &[-1.0, 0.5, 4.0]) < 1e-14);
        let mut three = vec![0.0; 9];
        assert!(!cholesky.solve_in_place(&mut three));
    }

    #[test]
    fn entries_above_the_diagonal_are_ignored() {
        let (pattern, values) = lower();
        let full = SymmetricPattern::new(3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2])
            .expect("valid pattern");
        // Upper entries hold garbage the factorization must not read.
        let full_values = [4.0, 1.0, 99.0, 3.0, 1.0, -99.0, 2.0];
        let mut solutions = Vec::new();
        for (pattern, values) in [(pattern, &values[..]), (full, &full_values[..])] {
            let mut cholesky = SparseCholesky::analyze(pattern).expect("analysis");
            assert!(cholesky.reserve(1) && cholesky.factor(values));
            let mut rhs = vec![1.0, 2.0, 3.0];
            assert!(cholesky.solve_in_place(&mut rhs));
            solutions.push(rhs);
        }
        assert_eq!(solutions[0], solutions[1]);
    }

    #[test]
    fn simplicial_analysis_factors_and_solves() {
        let (pattern, values) = lower();
        let mut cholesky = SparseCholesky::analyze_simplicial(pattern).expect("analysis");
        assert!(cholesky.reserve(1) && cholesky.factor(&values));
        let mut rhs = vec![1.0, 2.0, 3.0];
        assert!(cholesky.solve_in_place(&mut rhs));
        assert!(residual(&rhs, &[1.0, 2.0, 3.0]) < 1e-14);
    }

    #[test]
    fn refuses_to_factor_or_solve_out_of_contract() {
        let (pattern, values) = lower();
        let mut cholesky = SparseCholesky::analyze(pattern).expect("analysis");
        // Nothing reserved yet.
        assert!(!cholesky.factor(&values));
        assert!(cholesky.reserve(1));
        assert!(!cholesky.solve_in_place(&mut [1.0, 2.0, 3.0]));
        assert!(!cholesky.factor(&values[..4]));
        // Indefinite: the first pivot is negative.
        assert!(!cholesky.factor(&[-4.0, 1.0, 3.0, 1.0, 2.0]));
        assert!(!cholesky.solve_in_place(&mut [1.0, 2.0, 3.0]));
        assert!(cholesky.factor(&values));
        assert!(cholesky.solve_in_place(&mut [1.0, 2.0, 3.0]));
    }

    #[test]
    fn rejects_malformed_patterns() {
        assert!(SymmetricPattern::new(2, vec![0, 1], vec![0]).is_none());
        assert!(SymmetricPattern::new(2, vec![1, 1, 1], vec![0]).is_none());
        assert!(SymmetricPattern::new(2, vec![0, 3, 1], vec![0]).is_none());
        assert!(SymmetricPattern::new(2, vec![0, 2, 1], vec![0, 1]).is_none());
        assert!(SymmetricPattern::new(2, vec![0, 2, 2], vec![1, 0]).is_none());
        assert!(SymmetricPattern::new(2, vec![0, 2, 2], vec![0, 0]).is_none());
        assert!(SymmetricPattern::new(2, vec![0, 1, 2], vec![0, 2]).is_none());
        assert!(SymmetricPattern::new(2, vec![0, 1, 2], vec![0, 1]).is_some());
    }
}
