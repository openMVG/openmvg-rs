// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense reduced-camera Cholesky and spectral diagnostics (Householder tridiagonalization and
//! implicit QL, eigenvalues only).

use super::DenseBundleAdjustmentWorkspace;

pub(in crate::kernel::multi_view) fn solve_camera_step(
    workspace: &mut DenseBundleAdjustmentWorkspace,
    dimension: usize,
) -> bool {
    let matrix_len = dimension * dimension;
    workspace.factor[..matrix_len].copy_from_slice(&workspace.hessian[..matrix_len]);
    let scale = (0..dimension)
        .map(|index| workspace.factor[index * dimension + index].abs())
        .fold(0.0, f64::max);
    let tolerance = 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    for column in 0..dimension {
        for row in column..dimension {
            let (head, tail) = workspace.factor.split_at_mut(row * dimension);
            let row_start = if row == column {
                // The row being factored is also the pivot row: read it from `tail` below.
                None
            } else {
                Some(&head[column * dimension..column * dimension + column])
            };
            let row_prefix = &tail[..column];
            let dot = match row_start {
                Some(pivot_prefix) => dot_product(row_prefix, pivot_prefix),
                None => dot_product(row_prefix, row_prefix),
            };
            let value = tail[column] - dot;
            if row == column {
                if !(value.is_finite() && value > tolerance) {
                    return false;
                }
                workspace.factor[row * dimension + column] = value.sqrt();
            } else {
                workspace.factor[row * dimension + column] =
                    value / workspace.factor[column * dimension + column];
            }
        }
    }
    for row in 0..dimension {
        let mut value = -workspace.gradient[row];
        for column in 0..row {
            value -= workspace.factor[row * dimension + column] * workspace.step[column];
        }
        workspace.step[row] = value / workspace.factor[row * dimension + row];
    }
    for row in (0..dimension).rev() {
        let mut value = workspace.step[row];
        for column in row + 1..dimension {
            value -= workspace.factor[column * dimension + row] * workspace.step[column];
        }
        workspace.step[row] = value / workspace.factor[row * dimension + row];
    }
    workspace.step[..dimension]
        .iter()
        .all(|value| value.is_finite())
}

/// Dot product with four independent accumulators, so the loop vectorizes: a single running sum
/// serializes every addition and keeps the compiler from using SIMD.
#[inline]
fn dot_product(left: &[f64], right: &[f64]) -> f64 {
    let (left_blocks, left_tail) = left.as_chunks::<4>();
    let (right_blocks, right_tail) = right.as_chunks::<4>();
    let mut sums = [0.0f64; 4];
    for (a, b) in left_blocks.iter().zip(right_blocks) {
        for lane in 0..4 {
            sums[lane] += a[lane] * b[lane];
        }
    }
    let remainder: f64 = left_tail.iter().zip(right_tail).map(|(a, b)| a * b).sum();
    (sums[0] + sums[1]) + (sums[2] + sums[3]) + remainder
}

pub(in crate::kernel::multi_view) fn spectral_diagnostics(
    workspace: &mut DenseBundleAdjustmentWorkspace,
    dimension: usize,
) -> Option<(usize, f64)> {
    let matrix_len = dimension * dimension;
    let diagonal_scale = (0..dimension)
        .map(|index| workspace.hessian[index * dimension + index].abs())
        .fold(0.0, f64::max);
    if !(diagonal_scale.is_finite() && diagonal_scale > 0.0) {
        return None;
    }
    let diagonal_floor = 64.0 * f64::EPSILON * diagonal_scale;
    for index in 0..dimension {
        let diagonal = workspace.hessian[index * dimension + index];
        if !diagonal.is_finite() {
            return None;
        }
        workspace.step[index] = diagonal.max(diagonal_floor).sqrt().recip();
    }
    for row in 0..dimension {
        for column in 0..dimension {
            let value = 0.5
                * (workspace.hessian[row * dimension + column]
                    + workspace.hessian[column * dimension + row])
                * workspace.step[row]
                * workspace.step[column];
            if !value.is_finite() {
                return None;
            }
            workspace.factor[row * dimension + column] = value;
        }
    }
    // Eigenvalues only: Householder reduction to tridiagonal form, then implicit QL. The
    // tridiagonal diagonal lands in `step` and its off-diagonal in `diagonal`, both unused here.
    let eigenvalues = &mut workspace.step[..dimension];
    let off_diagonal = &mut workspace.diagonal[..dimension];
    if !symmetric_eigenvalues(
        &mut workspace.factor[..matrix_len],
        dimension,
        eigenvalues,
        off_diagonal,
    ) {
        return None;
    }
    let largest = workspace.step[..dimension]
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return None;
    }
    let tolerance = 64.0 * f64::EPSILON * dimension as f64 * largest;
    let mut rank = 0;
    let mut smallest = f64::INFINITY;
    for index in 0..dimension {
        let value = workspace.step[index];
        if !value.is_finite() || value < -tolerance {
            return None;
        }
        if value > tolerance {
            rank += 1;
            smallest = smallest.min(value);
        }
    }
    let reciprocal_condition_number = if rank == 0 { 0.0 } else { smallest / largest };
    Some((rank, reciprocal_condition_number))
}

/// Eigenvalues of a symmetric `dimension`-square matrix, stored row-major with its lower triangle
/// current. `matrix` is destroyed; `eigenvalues` and `off_diagonal` each hold `dimension`
/// scratch entries and the eigenvalues, unsorted, are left in `eigenvalues`. Returns `false` when
/// an eigenvalue does not converge within 30 QL iterations or a value is not finite.
///
/// Householder tridiagonalization costs `4/3 n³` flops and the QL iteration `O(n²)`: the
/// eigenvalues-only form of the EISPACK `tred1`/`tql1` pair.
fn symmetric_eigenvalues(
    matrix: &mut [f64],
    dimension: usize,
    eigenvalues: &mut [f64],
    off_diagonal: &mut [f64],
) -> bool {
    let n = dimension;
    if n == 0 {
        return true;
    }
    let at = |row: usize, column: usize| row * n + column;
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| matrix[at(i, k)].abs()).sum();
            if scale == 0.0 {
                off_diagonal[i] = matrix[at(i, l)];
            } else {
                for k in 0..=l {
                    matrix[at(i, k)] /= scale;
                    h += matrix[at(i, k)] * matrix[at(i, k)];
                }
                let f = matrix[at(i, l)];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                off_diagonal[i] = scale * g;
                h -= f * g;
                matrix[at(i, l)] = f - g;
                let mut f_sum = 0.0;
                for j in 0..=l {
                    let mut g_sum = 0.0;
                    for k in 0..=j {
                        g_sum += matrix[at(j, k)] * matrix[at(i, k)];
                    }
                    for k in j + 1..=l {
                        g_sum += matrix[at(k, j)] * matrix[at(i, k)];
                    }
                    off_diagonal[j] = g_sum / h;
                    f_sum += off_diagonal[j] * matrix[at(i, j)];
                }
                let hh = f_sum / (h + h);
                for j in 0..=l {
                    let f = matrix[at(i, j)];
                    let g = off_diagonal[j] - hh * f;
                    off_diagonal[j] = g;
                    for k in 0..=j {
                        matrix[at(j, k)] -= f * off_diagonal[k] + g * matrix[at(i, k)];
                    }
                }
            }
        } else {
            off_diagonal[i] = matrix[at(i, l)];
        }
    }
    off_diagonal[0] = 0.0;
    for (i, eigenvalue) in eigenvalues.iter_mut().enumerate().take(n) {
        *eigenvalue = matrix[at(i, i)];
    }
    if !eigenvalues
        .iter()
        .chain(off_diagonal.iter())
        .all(|v| v.is_finite())
    {
        return false;
    }
    // Shift the off-diagonal so that off_diagonal[i] couples i and i + 1.
    for i in 1..n {
        off_diagonal[i - 1] = off_diagonal[i];
    }
    off_diagonal[n - 1] = 0.0;
    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let scale = eigenvalues[m].abs() + eigenvalues[m + 1].abs();
                if off_diagonal[m].abs() + scale == scale {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iterations == 30 {
                return false;
            }
            iterations += 1;
            let mut g = (eigenvalues[l + 1] - eigenvalues[l]) / (2.0 * off_diagonal[l]);
            let mut r = g.hypot(1.0);
            g = eigenvalues[m] - eigenvalues[l] + off_diagonal[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut underflow = false;
            for i in (l..m).rev() {
                let f = s * off_diagonal[i];
                let b = c * off_diagonal[i];
                r = f.hypot(g);
                off_diagonal[i + 1] = r;
                if r == 0.0 {
                    eigenvalues[i + 1] -= p;
                    off_diagonal[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = eigenvalues[i + 1] - p;
                r = (eigenvalues[i] - g) * s + 2.0 * c * b;
                p = s * r;
                eigenvalues[i + 1] = g + p;
                g = c * r - b;
            }
            if underflow {
                continue;
            }
            eigenvalues[l] -= p;
            off_diagonal[l] = g;
            off_diagonal[m] = 0.0;
        }
    }
    eigenvalues.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, SymmetricEigen};

    use super::*;

    #[test]
    fn spectral_diagnostics_match_nalgebra_on_ba_sized_psd_systems() {
        for &(dimension, nullity) in &[(11, 0), (12, 1), (18, 0)] {
            let mut seed = DMatrix::from_fn(dimension, dimension, |row, column| {
                ((13 * row + 7 * column + 3) as f64).sin()
                    + 0.3 * ((5 * row + 11 * column + 1) as f64).cos()
            });
            for index in 0..dimension {
                seed[(index, index)] += 2.0;
            }
            let q = seed.qr().q();
            let eigenvalues = DVector::from_fn(dimension, |index, _| {
                if index + nullity >= dimension {
                    0.0
                } else {
                    10.0f64.powf(-5.0 * index as f64 / (dimension - nullity - 1) as f64)
                }
            });
            let base = &q * DMatrix::from_diagonal(&eigenvalues) * q.transpose();
            let coordinate_scale = DVector::from_fn(dimension, |index, _| {
                10.0f64.powf(-4.0 + 8.0 * index as f64 / (dimension - 1) as f64)
            });
            let matrix = DMatrix::from_fn(dimension, dimension, |row, column| {
                coordinate_scale[row] * base[(row, column)] * coordinate_scale[column]
            });
            let mut workspace =
                DenseBundleAdjustmentWorkspace::try_new(dimension.div_ceil(6) + 1, 1)
                    .expect("small diagnostic workspace");
            for row in 0..dimension {
                for column in 0..dimension {
                    workspace.hessian[row * dimension + column] = matrix[(row, column)];
                }
            }

            let diagonal_max = (0..dimension)
                .map(|index| matrix[(index, index)].abs())
                .fold(0.0, f64::max);
            let floor = 64.0 * f64::EPSILON * diagonal_max;
            let inverse_scale = DVector::from_fn(dimension, |index, _| {
                matrix[(index, index)].max(floor).sqrt().recip()
            });
            let equilibrated = DMatrix::from_fn(dimension, dimension, |row, column| {
                matrix[(row, column)] * inverse_scale[row] * inverse_scale[column]
            });
            let oracle = SymmetricEigen::new(equilibrated);
            let largest = oracle
                .eigenvalues
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            let tolerance = 64.0 * f64::EPSILON * dimension as f64 * largest;
            let positive: Vec<_> = oracle
                .eigenvalues
                .iter()
                .copied()
                .filter(|value| *value > tolerance)
                .collect();
            let oracle_rcond = positive.iter().copied().fold(f64::INFINITY, f64::min) / largest;
            let (rank, rcond) = spectral_diagnostics(&mut workspace, dimension)
                .expect("the eigenvalue iteration must converge for finite PSD input");
            assert_eq!(rank, dimension - nullity);
            assert_eq!(rank, positive.len());
            assert!((rcond - oracle_rcond).abs() <= 2e-9 * oracle_rcond.max(1.0));
        }
    }

    #[test]
    fn diagnostics_fail_closed_on_nonfinite_input() {
        let mut workspace =
            DenseBundleAdjustmentWorkspace::try_new(3, 1).expect("small diagnostic workspace");
        let dimension = 11;
        for index in 0..dimension {
            workspace.hessian[index * dimension + index] = 1.0;
        }
        workspace.hessian[3] = f64::NAN;
        assert_eq!(spectral_diagnostics(&mut workspace, dimension), None);
    }
}
