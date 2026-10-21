// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Symmetric-transfer refinement of a homography, which
//! [`estimate_homography`](super::estimate_homography) runs on the inliers of its best
//! hypothesis.
//!
//! The homography arithmetic lives here; `openmvg-ba` supplies only the reusable manifold and
//! nonlinear primitives.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │              Symmetric-transfer homography refinement                │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: initial H (3×3), indexed correspondences x₁ ↔ x₂ (at least 4,
//!        w ≠ 0), loss and Levenberg-Marquardt options
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Validate and normalize to SL(3)                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Check indices, finiteness, w, options. Scale H to max |entry| = 1, then
//! divide by cbrt(det H). Check that H and H⁻¹ have a nonzero denominator at
//! every point.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Residuals and Jacobian on SL(3)                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Four residuals per correspondence: x₂ − π(H x₁) and x₁ − π(H⁻¹ x₂), each
//! in inhomogeneous coordinates. Eight tangent parameters δ, with forward-mode
//! dual numbers (`Jet<8>`) for the derivatives.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Levenberg-Marquardt loop                                     │
//! └──────────────────────────────────────────────────────────────────────┘
//! Solve the damped normal equations for δ, retract H ← exp(Σ δₖ Gₖ) H,
//! renormalize to SL(3), accept or reject by cost, adapt the damping.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Report                                                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! Output: refined H (det = 1) in a `RefinementReport` with initial and
//! final cost, status, and a rank and condition diagnostic of the
//! weighted Jacobian
//! ```
//!
//! # Problem and method
//!
//! Given matched points `x₁ᵢ`, `x₂ᵢ` (homogeneous, divided by `w` first), the cost is
//! `Σᵢ ρ(‖rᵢ‖)` with `rᵢ ∈ ℝ⁴` the forward and backward transfer residuals
//! above and `ρ` the loss in `Options` (`L2`: `r²/2`, or Huber or Cauchy). The
//! residuals have the unit of the input coordinates. The error uses `H` and
//! `H⁻¹` together, so neither image is privileged.
//!
//! A homography is defined up to scale, so the state is fixed to `det H = 1`
//! (`SL(3)`, 8 degrees of freedom). The eight generators `Gₖ` are an orthonormal
//! basis for the diagonal traceless block plus the six off-diagonal unit
//! matrices. A step is applied on the left: `H ← exp(Σ δₖ Gₖ) H`. The
//! matrix exponential is a scaled Taylor series with squaring. Left
//! multiplication gives the tangent of `H` as `Gₖ H` and the tangent of `H⁻¹`
//! as `−H⁻¹ Gₖ`.
//!
//! The Levenberg-Marquardt loop is `openmvg_ba::numerics::refine::minimize`
//! with Nielsen damping. A candidate is accepted when the damping controller
//! accepts the ratio of actual to predicted cost reduction. The gradient
//! convergence test divides each gradient entry by the square root of the
//! matching Hessian diagonal (floored at `64 ε` times the largest diagonal entry).
//! The rank and condition diagnostics of the report come from an incremental
//! Givens QR of the weighted Jacobian rows at the final state.
//!
//! # Use in the crate
//!
//! [`estimate_homography`](super::estimate_homography) runs this refinement in
//! the refit step of its robust search, both for local optimization of a hypothesis and
//! for the final refinement, on the current inlier set in pixel coordinates.
//! The call uses `Options::l2(14)`: at most 14 iterations, the `L2` loss, and
//! tolerances of `1e-7` (gradient), `1e-7` (step), and `1e-9` (relative cost). The
//! refit is kept only when `RefinementReport::is_numerically_usable` holds,
//! which requires a converged status (or a stalled or iteration-limited one
//! with at least one accepted step) and a final cost no larger than the
//! initial cost; the robust search then also compares the refit's score.
//!
//! # Failure modes
//!
//! The refinement gives no result, and the estimate keeps the unrefined hypothesis, when:
//!
//! - an index is out of range, fewer than four indices are given, a point or
//!   an entry of `H` is not finite, or a point has `|w| ≤ 64 ε max|coordinate|`
//!   (`ε` is `f64::EPSILON`),
//! - the options are invalid (non-positive Huber `delta` or Cauchy `scale`),
//! - `H` has `|det| ≤ 64 ε` after scaling, or
//! - the denominator `H₃ · x` (or the one of `H⁻¹`) is near zero at a point,
//!   meaning that the point maps to the line at infinity.
//!
//! Inside the loop, a step whose retraction is not a finite, nondegenerate
//! matrix, or whose cost is not finite, is rejected and the damping rises.
//!
//! # Alternatives
//!
//! [`DltSolver`](super::DltSolver) (and ACA for four points) produce the
//! hypotheses that [`estimate_homography`](super::estimate_homography) scores;
//! this module is the nonlinear polish applied to a consensus set.

use nalgebra::Matrix3;
use openmvg_ba::numerics::jet::Jet;
use openmvg_ba::numerics::manifold::{Mat3, from_na, mat_mul, to_na};
use openmvg_ba::numerics::refine::{
    InformationDiagnostics, LmProblem, LmSolution, Loss, NormalEquations, Options,
    RefinementReport, minimize,
};

type HMatrix = Mat3;

struct SquareRootInformation<const N: usize> {
    upper: [[f64; N]; N],
    rows: usize,
}

impl<const N: usize> SquareRootInformation<N> {
    const fn new() -> Self {
        Self {
            upper: [[0.0; N]; N],
            rows: 0,
        }
    }

    fn add_row(&mut self, mut row: [f64; N]) {
        for column in 0..N {
            let diagonal = self.upper[column][column];
            let below = row[column];
            if below == 0.0 {
                continue;
            }
            let norm = diagonal.hypot(below);
            let cosine = diagonal / norm;
            let sine = below / norm;
            self.upper[column][column] = norm;
            for (trailing, bottom) in row.iter_mut().enumerate().skip(column + 1) {
                let top = self.upper[column][trailing];
                self.upper[column][trailing] = cosine * top + sine * *bottom;
                *bottom = -sine * top + cosine * *bottom;
            }
        }
        self.rows += 1;
    }

    fn numerical_rank(&self) -> usize {
        let largest_diagonal = (0..N)
            .map(|index| self.upper[index][index].abs())
            .fold(0.0, f64::max);
        let tolerance = largest_diagonal * f64::EPSILON * self.rows.max(N).max(1) as f64;
        (0..N)
            .filter(|&index| self.upper[index][index].abs() > tolerance)
            .count()
    }

    fn inverse_upper(&self) -> Option<[[f64; N]; N]> {
        if self.numerical_rank() < N {
            return None;
        }
        let mut inverse = [[0.0; N]; N];
        let mut column = 0;
        while column < N {
            let mut row = N;
            while row > 0 {
                row -= 1;
                let mut value = if row == column { 1.0 } else { 0.0 };
                let mut trailing = row + 1;
                while trailing < N {
                    value -= self.upper[row][trailing] * inverse[trailing][column];
                    trailing += 1;
                }
                inverse[row][column] = value / self.upper[row][row];
            }
            column += 1;
        }
        Some(inverse)
    }

    fn diagnostics(&self) -> InformationDiagnostics {
        let rank = self.numerical_rank();
        let reciprocal_condition_number = self.inverse_upper().map_or(0.0, |inverse| {
            let norm = self
                .upper
                .iter()
                .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
                .fold(0.0, f64::max);
            let inverse_norm = inverse
                .iter()
                .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
                .fold(0.0, f64::max);
            1.0 / (norm * inverse_norm)
        });
        InformationDiagnostics::new(N, rank, self.rows, reciprocal_condition_number)
    }
}

fn normalized_point(point: &[f64; 3]) -> [f64; 3] {
    if point[2] == 1.0 {
        *point
    } else {
        [point[0] / point[2], point[1] / point[2], 1.0]
    }
}

fn projective_denominator(homography: &HMatrix, point: &[f64; 3]) -> (f64, f64) {
    let point = normalized_point(point);
    let denominator = homography[2][0] * point[0] + homography[2][1] * point[1] + homography[2][2];
    let row_norm = (homography[2][0] * homography[2][0]
        + homography[2][1] * homography[2][1]
        + homography[2][2] * homography[2][2])
        .sqrt();
    let point_norm = (point[0] * point[0] + point[1] * point[1] + 1.0).sqrt();
    (denominator, row_norm * point_norm)
}

fn homography_residual<S>(
    homography: &[[S; 3]; 3],
    point1: &[S; 3],
    point2: &[S; 3],
    component: usize,
) -> S
where
    S: Copy
        + std::ops::Add<S, Output = S>
        + std::ops::Sub<S, Output = S>
        + std::ops::Mul<S, Output = S>
        + std::ops::Div<S, Output = S>,
{
    let denominator =
        homography[2][0] * point1[0] + homography[2][1] * point1[1] + homography[2][2] * point1[2];
    let numerator = homography[component][0] * point1[0]
        + homography[component][1] * point1[1]
        + homography[component][2] * point1[2];
    numerator / denominator - point2[component]
}

fn homography_reproj_scalar(
    homography: &HMatrix,
    point1: &[f64; 3],
    point2: &[f64; 3],
) -> [f64; 2] {
    let point1 = normalized_point(point1);
    let point2 = normalized_point(point2);
    [
        homography_residual(homography, &point1, &point2, 0),
        homography_residual(homography, &point1, &point2, 1),
    ]
}

fn homography_residual_jet(
    homography: &[[Jet<8>; 3]; 3],
    point1: &[f64; 3],
    point2: &[f64; 3],
    component: usize,
) -> Jet<8> {
    let point1 = normalized_point(point1).map(Jet::constant);
    let point2 = normalized_point(point2).map(Jet::constant);
    homography_residual(homography, &point1, &point2, component)
}

fn sl3_basis() -> [Mat3; 8] {
    let mut basis = [[[0.0; 3]; 3]; 8];
    let sqrt_two = 2.0f64.sqrt();
    let sqrt_six = 6.0f64.sqrt();
    basis[0][0][0] = 1.0 / sqrt_two;
    basis[0][1][1] = -1.0 / sqrt_two;
    basis[1][0][0] = 1.0 / sqrt_six;
    basis[1][1][1] = 1.0 / sqrt_six;
    basis[1][2][2] = -2.0 / sqrt_six;
    basis[2][0][1] = 1.0;
    basis[3][0][2] = 1.0;
    basis[4][1][0] = 1.0;
    basis[5][1][2] = 1.0;
    basis[6][2][0] = 1.0;
    basis[7][2][1] = 1.0;
    basis
}

fn exp_sl3(delta: &[f64; 8]) -> Mat3 {
    let basis = sl3_basis();
    let mut matrix = Matrix3::zeros();
    for parameter in 0..8 {
        matrix += to_na(&basis[parameter]) * delta[parameter];
    }
    let norm = matrix.norm();
    let squarings = if norm <= 0.5 {
        0
    } else {
        (norm / 0.5).log2().ceil() as u32
    };
    matrix /= 2.0f64.powi(squarings as i32);
    let mut exponential = Matrix3::identity();
    let mut term = Matrix3::identity();
    for order in 1..=18 {
        term = term * matrix / order as f64;
        exponential += term;
    }
    for _ in 0..squarings {
        exponential = exponential * exponential;
    }
    from_na(&exponential)
}

fn try_normalize_sl3(homography: &Mat3) -> Option<Mat3> {
    let scale = homography
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let scaled =
        std::array::from_fn(|row| std::array::from_fn(|column| homography[row][column] / scale));
    let determinant = to_na(&scaled).determinant();
    if !(determinant.is_finite() && determinant.abs() > 64.0 * f64::EPSILON) {
        return None;
    }
    let cube_root = determinant.cbrt();
    let normalized =
        std::array::from_fn(|row| std::array::from_fn(|column| scaled[row][column] / cube_root));
    normalized
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

fn tangent_basis(homography: &HMatrix) -> [Mat3; 8] {
    let generators = sl3_basis();
    std::array::from_fn(|parameter| mat_mul(&generators[parameter], homography))
}

fn inverse(homography: &HMatrix) -> HMatrix {
    let [[a, b, c], [d, e, f], [g, h, i]] = *homography;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    debug_assert!(determinant.is_finite() && determinant.abs() > 1e-15);
    let inverse_determinant = 1.0 / determinant;
    [
        [
            (e * i - f * h) * inverse_determinant,
            (c * h - b * i) * inverse_determinant,
            (b * f - c * e) * inverse_determinant,
        ],
        [
            (f * g - d * i) * inverse_determinant,
            (a * i - c * g) * inverse_determinant,
            (c * d - a * f) * inverse_determinant,
        ],
        [
            (d * h - e * g) * inverse_determinant,
            (b * g - a * h) * inverse_determinant,
            (a * e - b * d) * inverse_determinant,
        ],
    ]
}

fn points<'a>(
    points1: &'a [[f64; 3]],
    points2: &'a [[f64; 3]],
    indices: &'a [usize],
    position: usize,
) -> (&'a [f64; 3], &'a [f64; 3]) {
    let index = indices[position];
    (&points1[index], &points2[index])
}

fn validate_inputs(
    initial: &HMatrix,
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
    indices: &[usize],
    options: Options,
) -> bool {
    for &index in indices {
        if index >= points1.len() || index >= points2.len() {
            return false;
        }
    }
    for set in 0..2 {
        for (position, _) in indices.iter().enumerate() {
            let point = if set == 0 {
                points1[indices[position]]
            } else {
                points2[indices[position]]
            };
            if point.iter().any(|value| !value.is_finite()) {
                return false;
            }
            let scale = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
            if point[2].abs() <= 64.0 * f64::EPSILON * scale {
                return false;
            }
        }
    }
    indices.len() >= 4
        && initial.iter().flatten().all(|value| value.is_finite())
        && options.try_validate().is_ok()
}

fn validate_residual_domain(
    homography: &HMatrix,
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
    indices: &[usize],
) -> bool {
    let inverse_h = inverse(homography);
    for position in 0..indices.len() {
        let (point1, point2) = points(points1, points2, indices, position);
        for (matrix, point) in [(homography, point1), (&inverse_h, point2)] {
            let (denominator, scale) = projective_denominator(matrix, point);
            if !denominator.is_finite() || denominator.abs() <= 64.0 * f64::EPSILON * scale {
                return false;
            }
        }
    }
    true
}

fn cost(
    homography: &HMatrix,
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
    indices: &[usize],
    loss: Loss,
) -> f64 {
    let inverse_h = inverse(homography);
    let mut total = 0.0;
    for position in 0..indices.len() {
        let (point1, point2) = points(points1, points2, indices, position);
        for (matrix, point) in [(homography, point1), (&inverse_h, point2)] {
            let (denominator, scale) = projective_denominator(matrix, point);
            if !denominator.is_finite() || denominator.abs() <= 64.0 * f64::EPSILON * scale {
                return f64::INFINITY;
            }
        }
        let forward = homography_reproj_scalar(homography, point1, point2);
        let backward = homography_reproj_scalar(&inverse_h, point2, point1);
        let norm =
            (forward[0].powi(2) + forward[1].powi(2) + backward[0].powi(2) + backward[1].powi(2))
                .sqrt();
        total += loss.cost(norm);
    }
    total
}

fn jet_matrix<const N: usize>(matrix: &Mat3, tangent: &[Mat3; N]) -> [[Jet<N>; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| Jet {
            value: matrix[row][column],
            deriv: std::array::from_fn(|parameter| tangent[parameter][row][column]),
        })
    })
}

fn residuals(
    jet_h: &[[Jet<8>; 3]; 3],
    jet_inverse: &[[Jet<8>; 3]; 3],
    point1: &[f64; 3],
    point2: &[f64; 3],
) -> [Jet<8>; 4] {
    [
        homography_residual_jet(jet_h, point1, point2, 0),
        homography_residual_jet(jet_h, point1, point2, 1),
        homography_residual_jet(jet_inverse, point2, point1, 0),
        homography_residual_jet(jet_inverse, point2, point1, 1),
    ]
}

fn linearization(
    homography: &HMatrix,
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
    indices: &[usize],
    loss: Loss,
) -> (NormalEquations<8>, SquareRootInformation<8>) {
    let tangent = tangent_basis(homography);
    let jet_h = jet_matrix(homography, &tangent);
    let inverse_h = inverse(homography);
    let generators = sl3_basis();
    let inverse_tangent: [Mat3; 8] = std::array::from_fn(|parameter| {
        let mut derivative = mat_mul(&inverse_h, &generators[parameter]);
        for row in &mut derivative {
            for value in row {
                *value = -*value;
            }
        }
        derivative
    });
    let jet_inverse = jet_matrix(&inverse_h, &inverse_tangent);
    let mut equations = NormalEquations::new();
    let mut information = SquareRootInformation::new();
    for position in 0..indices.len() {
        let (point1, point2) = points(points1, points2, indices, position);
        let residuals = residuals(&jet_h, &jet_inverse, point1, point2);
        let norm = residuals
            .iter()
            .map(|residual| residual.value * residual.value)
            .sum::<f64>()
            .sqrt();
        let weight = loss.sqrt_weight(norm);
        for residual in residuals {
            equations.accumulate(&residual, weight);
            information.add_row(residual.deriv.map(|derivative| derivative * weight));
        }
    }
    (equations, information)
}

fn gradient_inf_norm(equations: &NormalEquations<8>) -> f64 {
    let scale = equations
        .hessian()
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .fold(0.0, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return if equations.gradient().iter().all(|value| *value == 0.0) {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let floor = 64.0 * f64::EPSILON * scale;
    equations
        .gradient()
        .iter()
        .enumerate()
        .map(|(index, value)| value.abs() / equations.hessian()[index][index].max(floor).sqrt())
        .fold(0.0, f64::max)
}

/// Symmetric-transfer refinement of a homography on `SL(3)`.
struct SymmetricTransfer<'a> {
    points1: &'a [[f64; 3]],
    points2: &'a [[f64; 3]],
    indices: &'a [usize],
}

impl LmProblem<8> for SymmetricTransfer<'_> {
    type State = HMatrix;

    fn linearize(&self, homography: &HMatrix, loss: Loss) -> Option<(NormalEquations<8>, f64)> {
        let current_cost = cost(homography, self.points1, self.points2, self.indices, loss);
        let (equations, _) =
            linearization(homography, self.points1, self.points2, self.indices, loss);
        Some((equations, current_cost))
    }

    fn gradient_inf_norm(&self, equations: &NormalEquations<8>) -> f64 {
        gradient_inf_norm(equations)
    }

    fn retract(&self, homography: &HMatrix, step: &[f64; 8]) -> Option<HMatrix> {
        try_normalize_sl3(&mat_mul(&exp_sl3(step), homography))
    }

    fn cost(&self, homography: &HMatrix, loss: Loss) -> Option<f64> {
        Some(cost(
            homography,
            self.points1,
            self.points2,
            self.indices,
            loss,
        ))
    }
}

pub(super) fn try_optimize_report_indexed(
    initial: &HMatrix,
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
    indices: &[usize],
    options: Options,
) -> Option<RefinementReport<HMatrix>> {
    if !validate_inputs(initial, points1, points2, indices, options) {
        return None;
    }
    let homography = try_normalize_sl3(initial)?;
    if !validate_residual_domain(&homography, points1, points2, indices) {
        return None;
    }
    let initial_cost = cost(&homography, points1, points2, indices, options.loss);
    let problem = SymmetricTransfer {
        points1,
        points2,
        indices,
    };
    let LmSolution {
        state,
        progress,
        damping,
    } = minimize(&problem, homography, options);
    let state = try_normalize_sl3(&state)?;
    let (equations, information) = linearization(&state, points1, points2, indices, options.loss);
    let diagnostics = information.diagnostics();
    Some(RefinementReport::from_parts(
        state,
        options.loss,
        progress.status(diagnostics),
        progress.iteration_count,
        progress.accepted_step_count,
        progress.rejected_step_count,
        initial_cost,
        cost(&state, points1, points2, indices, options.loss),
        gradient_inf_norm(&equations),
        progress.step_norm,
        damping.value(),
        damping.last_gain_ratio(),
        diagnostics,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SVD};
    use rand::{SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, StandardNormal};

    fn svd_rank(rows: &[[f64; 3]]) -> usize {
        let matrix = DMatrix::from_fn(rows.len(), 3, |row, column| rows[row][column]);
        let singular_values = SVD::new(matrix, false, false).singular_values;
        let largest = singular_values.iter().copied().fold(0.0, f64::max);
        let tolerance = largest * f64::EPSILON * rows.len().max(3) as f64;
        singular_values
            .iter()
            .filter(|value| **value > tolerance)
            .count()
    }

    #[test]
    fn incremental_qr_rank_matches_svd_oracle_and_row_permutation() {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        let mut rows = [[0.0; 3]; 24];
        for row in &mut rows {
            row[0] = StandardNormal.sample(&mut rng);
            row[1] = StandardNormal.sample(&mut rng);
            let sample: f64 = StandardNormal.sample(&mut rng);
            row[2] = 1e-8 * sample;
        }
        for dependent in [false, true] {
            if dependent {
                for row in &mut rows {
                    row[2] = 2.0 * row[0] - 0.5 * row[1];
                }
            }
            let mut forward = SquareRootInformation::<3>::new();
            let mut reverse = SquareRootInformation::<3>::new();
            for row in rows {
                forward.add_row(row);
            }
            for row in rows.into_iter().rev() {
                reverse.add_row(row);
            }
            let expected = svd_rank(&rows);
            assert_eq!(forward.diagnostics().rank, expected);
            assert_eq!(reverse.diagnostics().rank, expected);
        }
    }

    #[test]
    fn sl3_retraction_preserves_unit_determinant() {
        let homography = exp_sl3(&[0.2, -0.1, 0.05, 0.0, -0.03, 0.1, 0.02, -0.04]);
        assert!((to_na(&homography).determinant() - 1.0).abs() < 1e-10);
        assert_eq!(try_normalize_sl3(&[[0.0; 3]; 3]), None);
    }

    #[test]
    fn indexed_refinement_rejects_invalid_input() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let points = [[0.0, 0.0, 1.0]; 4];
        assert!(
            try_optimize_report_indexed(&identity, &points, &points, &[0, 1, 2, 4], Options::l2(1))
                .is_none()
        );
    }
}
