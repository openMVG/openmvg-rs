// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed symmetric block algebra shared by retained Schur backends.

use super::super::super::schur::add_scaled_pair;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedSymmetricCameraBlock<const MAX_DIMENSION: usize, const STORAGE: usize> {
    pub(crate) values: [f64; STORAGE],
}

impl<const MAX_DIMENSION: usize, const STORAGE: usize> Default
    for PackedSymmetricCameraBlock<MAX_DIMENSION, STORAGE>
{
    fn default() -> Self {
        assert!(
            STORAGE >= MAX_DIMENSION * (MAX_DIMENSION + 1) / 2,
            "packed symmetric storage is too small for its maximum dimension"
        );
        Self {
            values: [0.0; STORAGE],
        }
    }
}

impl<const MAX_DIMENSION: usize, const STORAGE: usize>
    PackedSymmetricCameraBlock<MAX_DIMENSION, STORAGE>
{
    #[inline]
    pub(crate) const fn index(row: usize, column: usize) -> usize {
        let (row, column) = if row >= column {
            (row, column)
        } else {
            (column, row)
        };
        row * (row + 1) / 2 + column
    }

    #[inline]
    pub(crate) fn get(&self, row: usize, column: usize) -> f64 {
        self.values[Self::index(row, column)]
    }

    #[inline]
    pub(crate) fn add(&mut self, row: usize, column: usize, value: f64) {
        self.values[Self::index(row, column)] += value;
    }

    /// Add `left[0][i] * right[0][j]` and then `left[1][i] * right[1][j]` at
    /// `(row + i, column + j)`, as in `HessianAccumulator::add_gram`.
    ///
    /// With `lower`, the rows and columns are the same run and only pairs
    /// `j <= i` are visited; otherwise the two runs are disjoint.
    #[inline]
    pub(crate) fn add_gram(
        &mut self,
        row: usize,
        column: usize,
        lower: bool,
        left: [&[f64]; 2],
        right: [&[f64]; 2],
    ) {
        if lower {
            // A fixed run length unrolls the triangle into straight-line code.
            let unrolled = match left[0].len() {
                1 => self.add_lower::<1>(row, left),
                2 => self.add_lower::<2>(row, left),
                3 => self.add_lower::<3>(row, left),
                4 => self.add_lower::<4>(row, left),
                5 => self.add_lower::<5>(row, left),
                6 => self.add_lower::<6>(row, left),
                _ => None,
            };
            if unrolled.is_some() {
                return;
            }
        }
        // Only the lower triangle is stored, so a block above the diagonal is
        // added through its transpose.
        let (row, column, left, right) = if row >= column {
            (row, column, left, right)
        } else {
            (column, row, right, left)
        };
        // Packed row `row + i` stores its columns contiguously, and the next
        // row begins `row + i + 1` values later.
        let mut first = Self::index(row, column);
        for (i, (&row_first, &row_second)) in left[0].iter().zip(left[1]).enumerate() {
            let count = if lower { i + 1 } else { right[0].len() };
            add_scaled_pair(
                &mut self.values[first..first + count],
                right[0],
                row_first,
                right[1],
                row_second,
            );
            first += row + i + 1;
        }
    }

    /// [`Self::add_gram`] for a lower run of `D` parameters from `row`, or
    /// `None` when `terms` are shorter than the run.
    #[inline(always)]
    fn add_lower<const D: usize>(&mut self, row: usize, terms: [&[f64]; 2]) -> Option<()> {
        // Local copies cannot alias the block, so each entry is loaded and
        // stored once.
        let (first, second) = (*terms[0].first_chunk::<D>()?, *terms[1].first_chunk::<D>()?);
        let mut start = Self::index(row, row);
        for i in 0..D {
            add_scaled_pair(
                &mut self.values[start..=start + i],
                &first[..=i],
                first[i],
                &second[..=i],
                second[i],
            );
            start += row + i + 1;
        }
        Some(())
    }

    pub(crate) fn multiply_add(&self, input: &[f64], output: &mut [f64], dimension: usize) {
        self.multiply_add_scaled(input, output, dimension, 1.0);
    }

    pub(crate) fn multiply_add_scaled(
        &self,
        input: &[f64],
        output: &mut [f64],
        dimension: usize,
        scale: f64,
    ) {
        debug_assert!(dimension <= MAX_DIMENSION);
        for row in 0..dimension {
            for column in 0..=row {
                let value = scale * self.get(row, column);
                output[row] += value * input[column];
                if row != column {
                    output[column] += value * input[row];
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct PackedCameraBlockFactor<const MAX_DIMENSION: usize, const STORAGE: usize> {
    lower: [f64; STORAGE],
    scale: [f64; MAX_DIMENSION],
}

impl<const MAX_DIMENSION: usize, const STORAGE: usize> Default
    for PackedCameraBlockFactor<MAX_DIMENSION, STORAGE>
{
    fn default() -> Self {
        assert!(
            STORAGE >= MAX_DIMENSION * (MAX_DIMENSION + 1) / 2,
            "packed factor storage is too small for its maximum dimension"
        );
        Self {
            lower: [0.0; STORAGE],
            scale: [0.0; MAX_DIMENSION],
        }
    }
}

impl<const MAX_DIMENSION: usize, const STORAGE: usize>
    PackedCameraBlockFactor<MAX_DIMENSION, STORAGE>
{
    pub(crate) fn factor(
        &mut self,
        matrix: &PackedSymmetricCameraBlock<MAX_DIMENSION, STORAGE>,
        dimension: usize,
    ) -> bool {
        if dimension == 0 || dimension > MAX_DIMENSION {
            return false;
        }
        self.lower.fill(0.0);
        let diagonal_scale = (0..dimension)
            .map(|index| matrix.get(index, index))
            .fold(0.0, f64::max);
        if !(diagonal_scale.is_finite() && diagonal_scale > 0.0) {
            return false;
        }
        let floor = 64.0 * f64::EPSILON * diagonal_scale;
        for index in 0..dimension {
            self.scale[index] = matrix.get(index, index).max(floor).sqrt();
        }
        for row in 0..dimension {
            for column in 0..=row {
                let mut value = matrix.get(row, column) / (self.scale[row] * self.scale[column]);
                for inner in 0..column {
                    value -= self.lower
                        [PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(row, inner)]
                        * self.lower[PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(
                            column, inner,
                        )];
                }
                if row == column {
                    if !(value.is_finite() && value > 64.0 * f64::EPSILON) {
                        return false;
                    }
                    self.lower[PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(
                        row, column,
                    )] = value.sqrt();
                } else {
                    self.lower[PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(
                        row, column,
                    )] = value
                        / self.lower[PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(
                            column, column,
                        )];
                }
            }
        }
        true
    }

    pub(crate) fn solve(&self, right: &[f64], solution: &mut [f64], dimension: usize) {
        debug_assert!(dimension <= MAX_DIMENSION);
        for row in 0..dimension {
            let mut value = right[row] / self.scale[row];
            for (column, &known) in solution[..row].iter().enumerate() {
                value -= self.lower
                    [PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(row, column)]
                    * known;
            }
            solution[row] = value
                / self.lower[PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(row, row)];
        }
        for row in (0..dimension).rev() {
            let mut value = solution[row];
            for (column, &known) in solution.iter().enumerate().take(dimension).skip(row + 1) {
                value -= self.lower
                    [PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(column, row)]
                    * known;
            }
            solution[row] = value
                / self.lower[PackedSymmetricCameraBlock::<MAX_DIMENSION, STORAGE>::index(row, row)];
        }
        for (value, scale) in solution.iter_mut().zip(&self.scale).take(dimension) {
            *value /= scale;
        }
    }
}
