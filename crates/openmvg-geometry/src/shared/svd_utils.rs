// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SVD utilities for geometric solvers.
//!
//! Common SVD-related operations used across multiple solvers.

use nalgebra::{DMatrix, DVector, Matrix3, RealField, SVD, SVector};
use num_traits::ToPrimitive;

use crate::{SolverError, SolverResult};

/// Solve the homogeneous linear system Ax = 0 using SVD.
///
/// Returns the unit-norm null-space vector x of the constraint matrix A: the
/// right singular vector of the smallest singular value.
pub fn solve_homogeneous_system<T: RealField + Copy + ToPrimitive>(
    constraint_matrix: &DMatrix<T>,
) -> SolverResult<DVector<T>> {
    let rows = constraint_matrix.nrows();
    let cols = constraint_matrix.ncols();
    if rows < cols {
        return Err(SolverError::DegenerateConfiguration {
            reason: format!(
                "homogeneous SVD needs at least as many rows as columns ({rows} < {cols})"
            ),
        });
    }
    if cols < 2 || !constraint_matrix.iter().all(|value| value.is_finite()) {
        return Err(SolverError::DegenerateConfiguration {
            reason: "homogeneous system is empty or contains non-finite coefficients".to_string(),
        });
    }

    let svd = SVD::new(constraint_matrix.clone(), false, true);
    let v_t = svd.v_t.ok_or(SolverError::SvdFailed)?;

    let singular_values: Vec<f64> = svd
        .singular_values
        .iter()
        .map(|value| value.to_f64().unwrap_or(f64::NAN).abs())
        .collect();
    if singular_values.len() != cols || singular_values.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::SvdFailed);
    }

    // A geometric homogeneous system must have at least rank n-1, and its
    // least-squares direction must be distinguishable from the next one.
    // The tolerance is relative to the data scale and only rejects gaps at
    // the numerical resolution of the decomposition.
    let largest = singular_values[0];
    if largest <= 0.0 {
        return Err(SolverError::DegenerateConfiguration {
            reason: "homogeneous system has zero numerical rank".to_string(),
        });
    }
    let smallest = singular_values[cols - 1];
    let second_smallest = singular_values[cols - 2];
    let tolerance = largest * f64::EPSILON * rows.max(cols) as f64 * 64.0;
    if second_smallest <= tolerance || second_smallest - smallest <= tolerance {
        return Err(SolverError::DegenerateConfiguration {
            reason: "homogeneous system has an ambiguous or rank-deficient null space".to_string(),
        });
    }

    // Get last row of V^T (smallest singular value)
    let last_row_index = v_t.nrows() - 1;

    if v_t.ncols() != cols {
        return Err(SolverError::Numerical {
            reason: format!(
                "V^T column dimension mismatch: expected {}, got {}",
                cols,
                v_t.ncols()
            ),
        });
    }

    let solution = v_t.row(last_row_index).transpose();
    if !solution.iter().all(|value| value.is_finite()) {
        return Err(SolverError::SvdFailed);
    }
    Ok(solution)
}

/// Reshape a 9-element vector into a 3x3 matrix (row major).
pub fn vector_to_matrix3<T: RealField + Copy>(vec: &DVector<T>) -> SolverResult<Matrix3<T>> {
    if vec.len() != 9 {
        return Err(SolverError::Numerical {
            reason: format!("expected a 9-element vector, got {}", vec.len()),
        });
    }

    let mut matrix = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            matrix[(i, j)] = vec[i * 3 + j];
        }
    }

    Ok(matrix)
}

/// Reshape a fixed-size 9-element vector into a 3x3 matrix (row major).
///
/// This stack-allocated form avoids a heap allocation.
#[inline]
pub fn vector9_to_matrix3<T: RealField + Copy>(vec: &SVector<T, 9>) -> Matrix3<T> {
    Matrix3::new(
        vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8],
    )
}

/// Enforce a rank constraint on a 3x3 matrix.
///
/// Sets the singular values beyond `target_rank` (1, 2, or 3) to zero and
/// returns the reconstructed matrix.
#[cfg(test)]
pub fn enforce_rank_constraint<T: RealField + Copy>(
    matrix: Matrix3<T>,
    target_rank: usize,
) -> SolverResult<Matrix3<T>> {
    let svd = SVD::new(matrix, true, true);

    let u = svd.u.ok_or(SolverError::SvdFailed)?;
    let vt = svd.v_t.ok_or(SolverError::SvdFailed)?;
    let mut s = svd.singular_values;

    // Zero out singular values beyond target rank
    for i in target_rank..s.len() {
        s[i] = T::zero();
    }

    let s_mat = Matrix3::from_diagonal(&s);
    Ok(u * s_mat * vt)
}

/// Compute the condition number of a matrix.
#[cfg(test)]
pub fn condition_number<T: RealField + Copy + ToPrimitive>(matrix: &Matrix3<T>) -> f64 {
    let svd = SVD::new(*matrix, false, false);
    let s = &svd.singular_values;

    if s.is_empty() {
        return f64::INFINITY;
    }

    let max_sv = s[0];
    let min_sv = s[s.len() - 1];

    if min_sv > T::zero() {
        (max_sv / min_sv).to_f64().unwrap_or(f64::INFINITY)
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_solve_homogeneous_system() {
        // Rank-eight overdetermined system with nullspace span(e_9).
        let mut a = DMatrix::<f64>::zeros(10, 9);
        for i in 0..8 {
            a[(i, i)] = 1.0;
        }
        a[(8, 0)] = 1.0;
        a[(8, 3)] = -2.0;
        a[(9, 2)] = 0.5;
        a[(9, 7)] = 1.0;

        let result = solve_homogeneous_system(&a);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.len(), 9);
        assert!((&a * &solution).norm() < 1e-12);
        assert!(solution[8].abs() > 1.0 - 1e-12);
    }

    #[test]
    fn test_solve_homogeneous_system_rejects_ambiguous_nullspace() {
        let underdetermined = DMatrix::<f64>::zeros(8, 9);
        assert!(solve_homogeneous_system(&underdetermined).is_err());

        let mut rank_seven = DMatrix::<f64>::zeros(10, 9);
        for i in 0..7 {
            rank_seven[(i, i)] = 1.0;
        }
        assert!(solve_homogeneous_system(&rank_seven).is_err());
    }

    #[test]
    fn test_vector_to_matrix3() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = vector_to_matrix3(&vec);

        assert!(result.is_ok());
        let matrix = result.unwrap();
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(2, 2)], 9.0);
    }

    #[test]
    fn test_enforce_rank_constraint() {
        let matrix = Matrix3::<f64>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

        let result = enforce_rank_constraint(matrix, 2);
        assert!(result.is_ok());

        let constrained = result.unwrap();
        let svd = SVD::new(constrained, false, false);
        assert!(svd.singular_values[2].abs() < 1e-10);
    }

    #[test]
    fn test_condition_number() {
        // Well-conditioned matrix
        let good = Matrix3::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5);
        let cond = condition_number(&good);
        assert!(cond.is_finite());
        assert!(cond < 100.0);

        // Near-singular matrix
        let bad = Matrix3::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0);
        let cond_bad = condition_number(&bad);
        assert!(cond_bad > 1e10);
    }
}
