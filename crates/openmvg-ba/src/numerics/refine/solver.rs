// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size scalar LM mechanics.

use super::covariance::InformationDiagnostics;
use crate::numerics::jet::Jet;

/// One raw residual value and its local Jacobian columns.
///
/// This is the neutral boundary between fixed-size correspondence models and
/// the LM driver. Analytic models can populate it directly, while automatic
/// differentiation models can convert from [`Jet`]. Whitening and robust
/// weighting are applied later by the driver.
#[derive(Clone, Copy)]
pub struct LinearizedResidual<const N: usize> {
    value: f64,
    jacobian: [f64; N],
}

impl<const N: usize> LinearizedResidual<N> {
    /// Construct a residual from a raw value and unweighted local derivatives.
    #[inline(always)]
    pub const fn new(value: f64, jacobian: [f64; N]) -> Self {
        Self { value, jacobian }
    }

    /// Raw residual value.
    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Unweighted local Jacobian columns.
    #[inline(always)]
    pub const fn jacobian(&self) -> &[f64; N] {
        &self.jacobian
    }

    /// Split the residual into its raw value and Jacobian columns.
    #[inline(always)]
    pub fn into_parts(self) -> (f64, [f64; N]) {
        (self.value, self.jacobian)
    }
}

impl<const N: usize> From<Jet<N>> for LinearizedResidual<N> {
    #[inline(always)]
    fn from(residual: Jet<N>) -> Self {
        Self::new(residual.value, residual.deriv)
    }
}

/// Fixed-size normal equations that can be accumulated and merged per worker.
///
/// `merge` enables deterministic allocation-free reduction of independently
/// evaluated correspondence chunks. Use one value per worker; do not share a
/// mutable accumulator between threads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalEquations<const N: usize> {
    pub(super) jtj: [[f64; N]; N],
    pub(super) jtr: [f64; N],
}

impl<const N: usize> Default for NormalEquations<N> {
    fn default() -> Self {
        Self {
            jtj: [[0.0; N]; N],
            jtr: [0.0; N],
        }
    }
}

impl<const N: usize> NormalEquations<N> {
    /// Construct an empty normal-equation accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all accumulated terms to zero.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.jtj = [[0.0; N]; N];
        self.jtr = [0.0; N];
    }

    /// Accumulate one scalar residual and its derivatives.
    #[inline]
    pub fn accumulate(&mut self, residual: &Jet<N>, sqrt_weight: f64) {
        self.accumulate_scalar(residual.value, &residual.deriv, sqrt_weight);
    }

    /// Accumulate one scalar residual with an explicit Jacobian row.
    pub fn accumulate_scalar(&mut self, residual: f64, jacobian: &[f64; N], sqrt_weight: f64) {
        let weighted_residual = sqrt_weight * residual;
        let weighted_jacobian = jacobian.map(|value| sqrt_weight * value);
        for row in 0..N {
            self.jtr[row] += weighted_jacobian[row] * weighted_residual;
            for column in row..N {
                let value = weighted_jacobian[row] * weighted_jacobian[column];
                self.jtj[row][column] += value;
                if row != column {
                    self.jtj[column][row] += value;
                }
            }
        }
    }

    /// Accumulate one neutral residual/Jacobian value.
    #[inline]
    pub fn accumulate_linearized(&mut self, residual: &LinearizedResidual<N>, sqrt_weight: f64) {
        self.accumulate_scalar(residual.value, &residual.jacobian, sqrt_weight);
    }

    /// Add another independently accumulated system into this one.
    pub fn merge(&mut self, other: &Self) {
        for row in 0..N {
            self.jtr[row] += other.jtr[row];
            for column in 0..N {
                self.jtj[row][column] += other.jtj[row][column];
            }
        }
    }

    /// Build a system from an accumulated approximate Hessian and gradient.
    pub fn from_parts(hessian: [[f64; N]; N], gradient: [f64; N]) -> Self {
        Self {
            jtj: hessian,
            jtr: gradient,
        }
    }

    /// Borrow the accumulated approximate Hessian.
    pub fn hessian(&self) -> &[[f64; N]; N] {
        &self.jtj
    }

    /// Borrow the accumulated residual gradient.
    pub fn gradient(&self) -> &[f64; N] {
        &self.jtr
    }

    /// Predicted decrease of the undamped local quadratic for `step`.
    ///
    /// If `M < N`, only the leading `M` variables are used. This supports the
    /// shared eight-column E/F/H workspace without copying its active prefix.
    /// A step with `M > N` is rejected at compile time.
    ///
    /// ```compile_fail
    /// use openmvg_ba::numerics::refine::NormalEquations;
    /// let equations = NormalEquations::<1>::new();
    /// equations.quadratic_model_reduction(&[0.0; 2]);
    /// ```
    pub fn quadratic_model_reduction<const M: usize>(&self, step: &[f64; M]) -> f64 {
        const { assert!(M <= N, "step dimension exceeds the normal equations") };
        let linear = (0..M).map(|row| self.jtr[row] * step[row]).sum::<f64>();
        let quadratic = (0..M)
            .map(|row| {
                step[row]
                    * (0..M)
                        .map(|column| self.jtj[row][column] * step[column])
                        .sum::<f64>()
            })
            .sum::<f64>();
        -linear - 0.5 * quadratic
    }
}

pub(super) fn gradient_inf_norm<const N: usize>(equations: &NormalEquations<N>) -> f64 {
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

/// Rank and reciprocal condition number of the Gauss-Newton Hessian, from a Jacobi
/// eigenvalue sweep over `residual_count` residuals.
#[allow(clippy::needless_range_loop)] // Symmetric Jacobi updates require paired row/column indices.
pub fn normal_equation_diagnostics<const N: usize>(
    equations: &NormalEquations<N>,
    residual_count: usize,
) -> InformationDiagnostics {
    if N == 0 {
        return InformationDiagnostics {
            dimension: 0,
            rank: 0,
            residual_count,
            reciprocal_condition_number: 1.0,
        };
    }
    let mut matrix = *equations.hessian();
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return InformationDiagnostics {
            dimension: N,
            rank: 0,
            residual_count,
            reciprocal_condition_number: 0.0,
        };
    }
    let sweep_limit = 32usize.saturating_mul(N).saturating_mul(N).max(1);
    for _ in 0..sweep_limit {
        let mut pivot = (0, 0);
        let mut largest_off_diagonal = 0.0;
        for row in 0..N {
            for column in row + 1..N {
                let magnitude = matrix[row][column].abs();
                if magnitude > largest_off_diagonal {
                    largest_off_diagonal = magnitude;
                    pivot = (row, column);
                }
            }
        }
        let scale = (0..N)
            .map(|index| matrix[index][index].abs())
            .fold(f64::MIN_POSITIVE, f64::max);
        if largest_off_diagonal <= 64.0 * f64::EPSILON * N as f64 * scale {
            break;
        }
        let (p, q) = pivot;
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        for index in 0..N {
            if index == p || index == q {
                continue;
            }
            let left = matrix[index][p];
            let right = matrix[index][q];
            matrix[index][p] = cosine * left - sine * right;
            matrix[p][index] = matrix[index][p];
            matrix[index][q] = sine * left + cosine * right;
            matrix[q][index] = matrix[index][q];
        }
        let pp = matrix[p][p];
        let qq = matrix[q][q];
        let pq = matrix[p][q];
        matrix[p][p] = cosine * cosine * pp - 2.0 * sine * cosine * pq + sine * sine * qq;
        matrix[q][q] = sine * sine * pp + 2.0 * sine * cosine * pq + cosine * cosine * qq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
    }
    let largest = (0..N)
        .map(|index| matrix[index][index].max(0.0))
        .fold(0.0, f64::max);
    let tolerance = 64.0 * f64::EPSILON * N as f64 * largest;
    let rank = (0..N)
        .filter(|&index| matrix[index][index] > tolerance)
        .count();
    let reciprocal_condition_number = if rank == N && largest > 0.0 {
        let smallest = (0..N)
            .map(|index| matrix[index][index].max(0.0))
            .fold(f64::INFINITY, f64::min);
        (smallest / largest).sqrt()
    } else {
        0.0
    };
    InformationDiagnostics {
        dimension: N,
        rank,
        residual_count,
        reciprocal_condition_number,
    }
}

/// Stack-allocated Cholesky solve for a fixed-size positive-definite system.
pub fn solve_positive_definite<const N: usize>(
    a: &[[f64; N]; N],
    b: &[f64; N],
) -> Option<[f64; N]> {
    if a.iter().flatten().chain(b).any(|value| !value.is_finite()) {
        return None;
    }
    let mut l = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = a[i][j];
            sum -= l[i][..j]
                .iter()
                .zip(&l[j][..j])
                .map(|(left, right)| left * right)
                .sum::<f64>();
            if !sum.is_finite() {
                return None;
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut y = [0.0; N];
    for i in 0..N {
        let mut sum = b[i];
        for (j, &value) in y.iter().take(i).enumerate() {
            sum -= l[i][j] * value;
        }
        y[i] = sum / l[i][i];
    }

    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let mut sum = y[i];
        for j in (i + 1)..N {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }
    x.iter().all(|value| value.is_finite()).then_some(x)
}

/// Solve one diagonally damped, norm-limited normal-equation step.
///
/// Returns `None` for non-finite inputs or solve output, negative damping or
/// step limit, or a damped system that is not positive definite.
pub fn damped_step_system<const N: usize>(
    equations: &NormalEquations<N>,
    lambda: f64,
    max_norm: f64,
) -> Option<[f64; N]> {
    if !lambda.is_finite() || lambda < 0.0 || !max_norm.is_finite() || max_norm < 0.0 {
        return None;
    }
    let mut a = equations.jtj;
    for (index, row) in a.iter_mut().enumerate() {
        row[index] += lambda * row[index].max(1e-10);
    }
    let raw = solve_positive_definite(&a, &equations.jtr)?;
    let mut step: [f64; N] = std::array::from_fn(|i| -raw[i]);
    let scale = step.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale > 0.0 {
        let scaled_norm = step
            .iter()
            .map(|value| (value / scale).powi(2))
            .sum::<f64>()
            .sqrt();
        if scale > max_norm || scaled_norm > max_norm / scale {
            let capped_scale = max_norm / scaled_norm;
            for value in &mut step {
                *value = (*value / scale) * capped_scale;
            }
        }
    }
    step.iter().all(|value| value.is_finite()).then_some(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damped_step_keeps_finite_large_steps_and_rejects_unrepresentable_solves() {
        let large = NormalEquations::from_parts([[1.0]], [1e200]);
        let step = damped_step_system(&large, 0.0, 1.0).unwrap();
        assert!((step[0] + 1.0).abs() <= 4.0 * f64::EPSILON);
        let two_dimensional =
            NormalEquations::from_parts([[1.0, 0.0], [0.0, 1.0]], [-3e200, -4e200]);
        let direction = damped_step_system(&two_dimensional, 0.0, 1.0).unwrap();
        assert!(
            (direction[0] - 0.6).abs() <= 4.0 * f64::EPSILON
                && (direction[1] - 0.8).abs() <= 4.0 * f64::EPSILON
        );
        let overflow = NormalEquations::from_parts([[1e-308]], [10.0]);
        assert!(damped_step_system(&overflow, 0.0, 1.0).is_none());
        let non_finite = NormalEquations::from_parts([[f64::NAN]], [0.0]);
        assert!(damped_step_system(&non_finite, 1.0, 1.0).is_none());
        assert!(damped_step_system(&large, -1.0, 1.0).is_none());
        assert!(damped_step_system(&large, 1.0, -1.0).is_none());
    }

    #[test]
    fn independently_accumulated_systems_merge_exactly() {
        let first = Jet::<2>::variable(0.7, 0);
        let second = Jet::<2>::variable(-0.4, 1);
        let weight = 1.0;

        let mut combined = NormalEquations::<2>::new();
        combined.accumulate(&first, weight);
        combined.accumulate(&second, weight);

        let mut left = NormalEquations::<2>::new();
        let mut right = NormalEquations::<2>::new();
        left.accumulate(&first, weight);
        right.accumulate(&second, weight);
        left.merge(&right);

        assert_eq!(left, combined);
    }

    #[test]
    fn quadratic_model_reduction_supports_an_active_prefix() {
        let mut equations = NormalEquations::<3>::new();
        equations.jtj = [[4.0, 1.0, 99.0], [1.0, 2.0, 88.0], [99.0, 88.0, 77.0]];
        equations.jtr = [2.0, -1.0, 66.0];
        let step = [-0.25, 0.5];
        let expected = -(2.0 * step[0] - step[1])
            - 0.5 * (4.0 * step[0] * step[0] + 2.0 * step[0] * step[1] + 2.0 * step[1] * step[1]);
        assert!((equations.quadratic_model_reduction(&step) - expected).abs() < 1e-15);
    }

    #[test]
    fn quadratic_model_reduction_handles_zero_dimension_and_known_prefix() {
        assert_eq!(
            NormalEquations::<0>::new().quadratic_model_reduction(&[]),
            0.0
        );
        let equations = NormalEquations::from_parts(
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 7.0]],
            [1.0, -2.0, 5.0],
        );
        // q(0)-q(step) for q(x)=x0-2*x1+5*x2+x0²+1.5*x1²+3.5*x2².
        assert_eq!(equations.quadratic_model_reduction(&[-0.5, 0.25]), 0.65625);
    }

    #[test]
    fn gradient_norm_is_invariant_to_equivalent_local_coordinate_scaling() {
        let mut unit_chart = NormalEquations::<1>::new();
        unit_chart.jtj = [[1.0]];
        unit_chart.jtr = [1.0e-7];

        let mut scaled_chart = NormalEquations::<1>::new();
        scaled_chart.jtj = [[1.0e-8]];
        scaled_chart.jtr = [1.0e-11];

        let unit = gradient_inf_norm(&unit_chart);
        let scaled = gradient_inf_norm(&scaled_chart);
        assert!((unit - 1.0e-7).abs() <= f64::EPSILON);
        assert!((scaled - unit).abs() <= f64::EPSILON);
    }
}
