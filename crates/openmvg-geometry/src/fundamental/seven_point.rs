// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 7-point algorithm for fundamental matrix estimation.
//!
//! Its teaching docs are on [`SevenPointSolver`], where docs.rs shows them.

use arrayvec::ArrayVec;
use nalgebra::{Matrix3, RealField, SMatrix, SVector};
use num_traits::ToPrimitive;

use crate::correspondence::PixelCorrespondence;
use crate::normalization::{
    CorrespondenceCoordinates, NormalizedCorrespondence, NormalizedCorrespondences,
};
use crate::residuals::{sampson_error_fundamental, sampson_error_fundamental_squared};
use crate::shared::polynomial::cubic_real;
use crate::shared::svd_utils::vector9_to_matrix3;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use crate::{SolverError, SolverResult};

use super::{FundamentalMatrix, NORMALIZED_FUNDAMENTAL_CONDITIONING_RATIO, denormalize};

const ROOT_DEGREE_TOLERANCE: f64 = f64::EPSILON * 64.0;

/// Up to three projectively distinct models, stored inline so a RANSAC draw
/// does not allocate.
type SevenPointModels = ArrayVec<FundamentalMatrix, 3>;

/// Fundamental matrix solver using the 7-point algorithm.
///
/// This is the minimal solver for fundamental matrix estimation.
/// It requires exactly 7 point correspondences and returns up to 3 solutions, none when the
/// sample is degenerate.
///
/// # How it works
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                    7-point fundamental matrix solver                        │
/// │                         (minimal solver)                                    │
/// └─────────────────────────────────────────────────────────────────────────────┘
///
///     Input: exactly 7 pixel correspondences (p₁, p₂)
///
///                    ┌───────────────────────────────────────┐
///                    │  Step 1: Build constraint matrix      │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each correspondence (p₁ᵢ, p₂ᵢ):
///                    p₂ᵢᵀ · F · p₁ᵢ = 0  (epipolar constraint)
///
///                    Expand: [x₂x₁, x₂y₁, x₂, y₂x₁, y₂y₁, y₂, x₁, y₁, 1] · vec(F) = 0
///
///                    Build constraint matrix A (7×9)
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 2: Find 2D nullspace            │
///                    └───────────────────────────────────────┘
///                                      │
///                    With 7 equations in 9 unknowns:
///                    nullspace has dimension 2
///
///                    Pivoted Householder QR: AᵀP = Q·R
///                    F₁, F₂ = last two columns of complete Q
///                    (direct factorization; never forms AᵀA)
///
///                    Any F can be written as: F = x·F₁ + F₂
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 3: Apply rank-2 constraint      │
///                    └───────────────────────────────────────┘
///                                      │
///                    Fundamental matrix must have rank 2:
///                    det(F) = 0
///
///                    det(x·F₁ + F₂) = 0
///
///                    Expand determinant → cubic polynomial in x:
///                    c₃·x³ + c₂·x² + c₁·x + c₀ = 0
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 4: Solve cubic equation         │
///                    └───────────────────────────────────────┘
///                                      │
///                    Use cubic formula (Cardano's method) or
///                    numerical root finding
///
///                    Yields 1, 2, or 3 real projective roots
///                                      │
///                                      ▼
///                    ┌───────────────────────────────────────┐
///                    │  Step 5: Construct solutions          │
///                    └───────────────────────────────────────┘
///                                      │
///                    For each finite real root xᵢ:
///                    Fᵢ = xᵢ·F₁ + F₂
///
///                    Normalize: Fᵢ ← Fᵢ / ||Fᵢ||_F
///                                      │
///                                      ▼
///     Output: 0-3 fundamental matrices
///
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │  Complexity: O(1), fixed 9×7 pivoted QR + cubic solve                       │
/// │  Memory: Fixed-size stack intermediates; one result Vec allocation          │
/// │  Solutions: one per real root of the cubic that passes the checks, 0-3      │
/// │  Note: Fits on normalized input; the public solve denormalizes              │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Normalized input
///
/// [`SevenPointSolver::solve`] takes a [`NormalizedCorrespondences`] set, so the
/// Hartley-normalized input is part of the type and the fit runs on well-conditioned
/// coordinates. It returns the models in pixel coordinates, denormalized with that set's
/// transforms.
///
/// The two null-space basis vectors are the last two columns of `Q` in the
/// column-pivoted Householder QR of `Aᵀ`.
///
/// See the usage example in [`crate::fundamental`].
#[derive(Debug, Clone, Default)]
pub struct SevenPointSolver;

impl SevenPointSolver {
    /// Return the solver's minimum correspondence count.
    #[must_use]
    pub fn min_sample_size() -> usize {
        <Self as Solver>::min_sample_size()
    }

    /// Evaluate the Sampson distance of a pixel-space model for one pixel correspondence.
    pub fn residual(model: &FundamentalMatrix, datum: &PixelCorrespondence) -> f64 {
        sampson_error_fundamental(&model.0, datum)
    }

    /// Validate the model's geometric invariants.
    #[must_use]
    pub fn is_valid(model: &FundamentalMatrix) -> bool {
        <Self as Solver>::is_valid(model)
    }

    /// Fit on exactly seven normalized correspondences and return pixel-space models.
    ///
    /// Conditioning is checked before denormalization. A pixel representation
    /// whose rank is unresolved in f64 is omitted from the returned models.
    pub fn solve(data: &NormalizedCorrespondences) -> SolverResult<Vec<FundamentalMatrix>> {
        let models = <Self as MultiSolutionSolver>::solve(data.as_slice(), &())?;
        Ok(models
            .iter()
            .map(|model| denormalize(model, data.normalization()))
            .filter(FundamentalMatrix::is_valid)
            .collect())
    }
}

fn build_constraint_matrix<D: CorrespondenceCoordinates>(
    correspondences: &[D],
) -> SMatrix<f64, 7, 9> {
    let mut a = SMatrix::<f64, 7, 9>::zeros();
    for (i, corr) in correspondences.iter().map(D::coordinates).enumerate() {
        let x1 = corr.point1.x;
        let y1 = corr.point1.y;
        let x2 = corr.point2.x;
        let y2 = corr.point2.y;

        // Row-major vectorization: x2^T F x1 = 0.
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }
    a
}

/// Compute the complete two-dimensional right nullspace of a full-row-rank
/// 7x9 matrix without forming normal equations.
///
/// A column-pivoted Householder QR of A^T gives `A^T P = Q R`. Pivoting only
/// reorders the seven constraints, so columns 7 and 8 of the complete Q span
/// null(A). nalgebra exposes a thin `q()`, therefore the complete Q is obtained
/// by applying the stored reflectors to a 9x9 identity with `q_tr_mul`.
fn direct_nullspace<T>(a: &SMatrix<T, 7, 9>) -> SolverResult<(SVector<T, 9>, SVector<T, 9>)>
where
    T: RealField + Copy + ToPrimitive,
{
    let epsilon = T::default_epsilon().to_f64().unwrap_or(f64::EPSILON);
    // The pivoted R diagonal is a rank-revealing check on A itself. Unlike
    // eigenanalysis of A^T A, it does not square the condition number.
    let qr = a.transpose().col_piv_qr();
    let r = qr.r();
    let r_largest = (0..7)
        .map(|i| r[(i, i)].to_f64().unwrap_or(f64::NAN).abs())
        .fold(0.0_f64, f64::max);
    let r_smallest = (0..7)
        .map(|i| r[(i, i)].to_f64().unwrap_or(f64::NAN).abs())
        .fold(f64::INFINITY, f64::min);
    if !r_largest.is_finite()
        || !r_smallest.is_finite()
        || r_largest <= 0.0
        || r_smallest <= r_largest * epsilon * 9.0 * 64.0
    {
        return Err(SolverError::DegenerateConfiguration {
            reason: "seven-point pivoted QR has an insufficient null-space gap".to_string(),
        });
    }

    let mut q_transpose = SMatrix::<T, 9, 9>::identity();
    qr.q_tr_mul(&mut q_transpose);
    let first = q_transpose.row(7).transpose();
    let second = q_transpose.row(8).transpose();
    if !first.iter().all(|value| value.is_finite()) || !second.iter().all(|value| value.is_finite())
    {
        return Err(SolverError::SvdFailed);
    }

    // Validate the actual factors, not only decomposition metadata.
    let matrix_norm = a.norm().to_f64().unwrap_or(f64::INFINITY);
    let residual_tolerance = matrix_norm * epsilon * 9.0 * 256.0;
    let first_residual = (a * first).norm().to_f64().unwrap_or(f64::INFINITY);
    let second_residual = (a * second).norm().to_f64().unwrap_or(f64::INFINITY);
    let orthogonality = first.dot(&second).to_f64().unwrap_or(f64::INFINITY).abs();
    if first_residual > residual_tolerance
        || second_residual > residual_tolerance
        || orthogonality > epsilon * 256.0
    {
        return Err(SolverError::Numerical {
            reason: "seven-point complete QR failed the null-space check".to_string(),
        });
    }

    Ok((first, second))
}

impl SevenPointSolver {
    /// Solve for fundamental matrices: the core of the public [`Self::solve`].
    ///
    /// Works in the coordinates of `correspondences` (Hartley-normalized on
    /// every non-test path) and returns up to 3 fundamental matrices.
    fn solve_sample<D: CorrespondenceCoordinates>(
        correspondences: &[D],
    ) -> SolverResult<SevenPointModels> {
        if correspondences.len() < 7 {
            return Err(SolverError::InsufficientData {
                required: 7,
                provided: correspondences.len(),
            });
        }
        if correspondences.len() > 7 {
            return Err(SolverError::InvalidSampleSize {
                expected: 7,
                provided: correspondences.len(),
            });
        }

        // Build the 7x9 constraint matrix (stack-allocated for the hot path).
        let a = build_constraint_matrix(correspondences);
        if !a.iter().all(|value| value.is_finite()) {
            return Err(SolverError::InvalidInput {
                reason: "seven-point coordinates must be finite",
            });
        }

        let (f1_vec, f2_vec) = match direct_nullspace(&a) {
            Ok(nullspace) => nullspace,
            Err(SolverError::DegenerateConfiguration { .. }) => return Ok(ArrayVec::new()),
            Err(error) => return Err(error),
        };

        let f1 = vector9_to_matrix3(&f1_vec);
        let f2 = vector9_to_matrix3(&f2_vec);

        // Use the direct projective parameterization F(x) = x*F1 + F2.
        // It avoids the cancellation introduced by first forming F1-F2.
        let Some(coefficients_f64) = normalized_det_coefficients(&f1, &f2) else {
            return Ok(ArrayVec::new());
        };
        // If both endpoint determinants are tiny but a mixed coefficient is
        // large, neither original affine chart exposes the true cubic degree.
        // A conditional 45-degree orthonormal basis rotation moves those
        // finite projective roots away from infinity. The common path performs
        // only the endpoint comparison and keeps the original basis.
        let endpoint_strength = coefficients_f64[0].abs().max(coefficients_f64[3].abs());
        if endpoint_strength <= ROOT_DEGREE_TOLERANCE {
            return solve_with_rotated_root_chart(&f1, &f2, coefficients_f64);
        }
        solve_root_chart(&f1, &f2, coefficients_f64)
    }
}

fn normalized_det_coefficients<T>(f1: &Matrix3<T>, f2: &Matrix3<T>) -> Option<[f64; 4]>
where
    T: RealField + Copy + ToPrimitive,
{
    let coefficients = compute_det_cubic_coefficients(f1, f2);
    let mut coefficients: [f64; 4] =
        std::array::from_fn(|i| coefficients[i].to_f64().unwrap_or(f64::NAN));
    let scale = coefficients
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= f64::MIN_POSITIVE {
        return None;
    }
    for coefficient in &mut coefficients {
        *coefficient /= scale;
    }
    Some(coefficients)
}

#[cold]
#[inline(never)]
fn solve_with_rotated_root_chart(
    f1: &Matrix3<f64>,
    f2: &Matrix3<f64>,
    coefficients: [f64; 4],
) -> SolverResult<SevenPointModels> {
    let endpoint_strength = coefficients[0].abs().max(coefficients[3].abs());
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    let rotated_f1 = (*f1 + *f2) * scale;
    let rotated_f2 = (*f1 - *f2) * scale;
    let Some(rotated_coefficients) = normalized_det_coefficients(&rotated_f1, &rotated_f2) else {
        return solve_root_chart(f1, f2, coefficients);
    };
    let rotated_strength = rotated_coefficients[0]
        .abs()
        .max(rotated_coefficients[3].abs());
    if rotated_strength > endpoint_strength {
        solve_root_chart(&rotated_f1, &rotated_f2, rotated_coefficients)
    } else {
        solve_root_chart(f1, f2, coefficients)
    }
}

#[inline]
fn solve_root_chart(
    f1: &Matrix3<f64>,
    f2: &Matrix3<f64>,
    mut coefficients: [f64; 4],
) -> SolverResult<SevenPointModels> {
    // A nullspace basis defines two equivalent affine charts on the
    // projective root line. Prefer the chart whose cubic leading coefficient
    // is larger. Reversing coefficients is exactly F1/F2 swapping (x -> 1/x).
    let (root_direction, root_origin) = choose_projective_root_chart(f1, f2, &mut coefficients);
    let [c0, c1, c2, c3] = coefficients;
    let quadratic = c3.abs() <= ROOT_DEGREE_TOLERANCE;
    let mut roots = [0.0; 3];
    // Projective degree policy belongs here: the generic cubic backend keeps
    // every finite root of a nonzero leading coefficient.
    let leading = if quadratic { 0.0 } else { c3 };
    let root_count = cubic_real([leading, c2, c1, c0], &mut roots);

    let mut results = SevenPointModels::new();
    for root in roots.into_iter().take(root_count) {
        if let Some(model) = construct_fundamental_matrix(root_direction, root_origin, Some(root)) {
            push_projectively_unique(&mut results, model);
        }
    }
    if quadratic
        && let Some(model) = construct_fundamental_matrix(root_direction, root_origin, None)
    {
        push_projectively_unique(&mut results, model);
    }
    Ok(results)
}

fn choose_projective_root_chart<'a, T: RealField>(
    f1: &'a Matrix3<T>,
    f2: &'a Matrix3<T>,
    coefficients: &mut [f64; 4],
) -> (&'a Matrix3<T>, &'a Matrix3<T>) {
    if coefficients[3].abs() < coefficients[0].abs() {
        coefficients.reverse();
        (f2, f1)
    } else {
        (f1, f2)
    }
}

/// Compute coefficients `[c0, c1, c2, c3]` of `det(x*F1 + F2)`.
fn compute_det_cubic_coefficients<T: RealField + Copy>(f1: &Matrix3<T>, f2: &Matrix3<T>) -> [T; 4] {
    let c0 = f2.determinant();
    let c1 = determinant_linear_coefficient(f2, f1);
    let c2 = determinant_linear_coefficient(f1, f2);
    let c3 = f1.determinant();
    [c0, c1, c2, c3]
}

/// Coefficient of `x` in `det(base + x*direction)`, expanded directly.
fn determinant_linear_coefficient<T: RealField + Copy>(
    base: &Matrix3<T>,
    direction: &Matrix3<T>,
) -> T {
    direction[(0, 0)] * (base[(1, 1)] * base[(2, 2)] - base[(1, 2)] * base[(2, 1)])
        + base[(0, 0)]
            * (direction[(1, 1)] * base[(2, 2)] + base[(1, 1)] * direction[(2, 2)]
                - direction[(1, 2)] * base[(2, 1)]
                - base[(1, 2)] * direction[(2, 1)])
        - direction[(0, 1)] * (base[(1, 0)] * base[(2, 2)] - base[(1, 2)] * base[(2, 0)])
        - base[(0, 1)]
            * (direction[(1, 0)] * base[(2, 2)] + base[(1, 0)] * direction[(2, 2)]
                - direction[(1, 2)] * base[(2, 0)]
                - base[(1, 2)] * direction[(2, 0)])
        + direction[(0, 2)] * (base[(1, 0)] * base[(2, 1)] - base[(1, 1)] * base[(2, 0)])
        + base[(0, 2)]
            * (direction[(1, 0)] * base[(2, 1)] + base[(1, 0)] * direction[(2, 1)]
                - direction[(1, 1)] * base[(2, 0)]
                - base[(1, 1)] * direction[(2, 0)])
}

/// Compute the adjugate (classical adjoint) of a 3x3 matrix.
///
/// adj(M) = det(M) * M^(-1) for invertible M
/// For any matrix: adj(M)_ij = (-1)^(i+j) * M_ji where M_ji is the (j,i) minor
fn adjugate<T: RealField + Copy>(m: &Matrix3<T>) -> Matrix3<T> {
    // Cofactor matrix entries
    let c00 = m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)];
    let c01 = -(m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]);
    let c02 = m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)];

    let c10 = -(m[(0, 1)] * m[(2, 2)] - m[(0, 2)] * m[(2, 1)]);
    let c11 = m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)];
    let c12 = -(m[(0, 0)] * m[(2, 1)] - m[(0, 1)] * m[(2, 0)]);

    let c20 = m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)];
    let c21 = -(m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)]);
    let c22 = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];

    // Adjugate is transpose of cofactor matrix
    Matrix3::new(c00, c10, c20, c01, c11, c21, c02, c12, c22)
}

/// Construct a Frobenius-normalized member of the projective nullspace.
///
/// A finite root represents `x*F1 + F2`; `None` is the root at infinity and
/// represents `F1`. For large roots the equivalent `F1 + F2/x` form avoids
/// overflow and retains the lower-order term. No SVD projection is applied:
/// that would move the solution out of the exact linear nullspace.
fn construct_fundamental_matrix(
    f1: &Matrix3<f64>,
    f2: &Matrix3<f64>,
    root: Option<f64>,
) -> Option<FundamentalMatrix> {
    let f = match root {
        Some(root) if root.abs() > 1.0 => *f1 + *f2 / root,
        Some(root) => *f1 * root + *f2,
        None => *f1,
    };
    let norm = f.norm();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    let normalized = f / norm;
    if !normalized.iter().all(|value| value.is_finite()) {
        return None;
    }

    let determinant_tolerance = f64::EPSILON * 4096.0;
    let rank_ratio_tolerance = NORMALIZED_FUNDAMENTAL_CONDITIONING_RATIO;
    // For an exactly rank-2, unit-Frobenius matrix with singular-value ratio
    // r = sigma_2 / sigma_1, ||adj(F)||_F = r / (1 + r^2). This is the cheap,
    // conditioning bound for this normalized fit. It must not be applied to
    // denormalized pixel representations.
    let rank_two_lower_bound =
        rank_ratio_tolerance / (1.0 + rank_ratio_tolerance * rank_ratio_tolerance);
    if normalized.determinant().abs() > determinant_tolerance
        || adjugate(&normalized).norm() <= rank_two_lower_bound
    {
        return None;
    }
    let model = FundamentalMatrix::new(normalized);
    // A determinant residual alone does not resolve numerical rank. Keep the
    // exact linear nullspace member or refuse it; an SVD projection would move
    // it out of the sample's epipolar nullspace.
    model.is_valid().then_some(model)
}

fn push_projectively_unique(results: &mut SevenPointModels, model: FundamentalMatrix) {
    let tolerance = f64::EPSILON * 4096.0;
    let duplicate = results.iter().any(|existing| {
        (existing.0 - model.0)
            .norm()
            .min((existing.0 + model.0).norm())
            <= tolerance
    });
    if !duplicate && results.len() < 3 {
        results.push(model);
    }
}

impl Solver for SevenPointSolver {
    type Datum = NormalizedCorrespondence;
    type Model = FundamentalMatrix;
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental(&model.0, datum.coordinates())
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_fundamental_squared(&model.0, datum.coordinates())
    }

    fn min_sample_size() -> usize {
        7
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

impl MultiSolutionSolver for SevenPointSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        SevenPointSolver::solve_sample(data).map(|models| models.into_iter().collect())
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
        mut emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        // RANSAC hot path: gather the sample on the stack and emit the inline
        // models directly, with no heap allocation per draw.
        let indices: &[usize; 7] = indices.try_into().map_err(|_| {
            if indices.len() < 7 {
                SolverError::InsufficientData {
                    required: 7,
                    provided: indices.len(),
                }
            } else {
                SolverError::InvalidSampleSize {
                    expected: 7,
                    provided: indices.len(),
                }
            }
        })?;
        let datum = |index: usize| {
            data.get(index)
                .copied()
                .ok_or(SolverError::InvalidDataIndex {
                    index,
                    data_len: data.len(),
                })
        };
        let mut sample = [datum(indices[0])?; 7];
        for (target, &source) in sample.iter_mut().zip(indices).skip(1) {
            *target = datum(source)?;
        }

        let models = SevenPointSolver::solve_sample(&sample)?;
        let count = models.len();
        models.into_iter().for_each(&mut emit);
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
