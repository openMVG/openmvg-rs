// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free PCG recurrence shared by retained Schur backends.

use super::retained::{RejectedSchurSolve, SchurAttempt, SchurSolve};

pub(super) struct PcgBuffers<'a> {
    pub(super) solution: &'a mut [f64],
    pub(super) residual: &'a mut [f64],
    pub(super) preconditioned: &'a mut [f64],
    pub(super) direction: &'a mut [f64],
    pub(super) product: &'a mut [f64],
}

#[derive(Clone, Copy)]
pub(super) struct PcgOptions {
    pub(super) maximum_iterations: usize,
    pub(super) relative_tolerance: f64,
    pub(super) require_bounded_residual: bool,
}

pub(super) fn solve(
    gradient: &[f64],
    buffers: PcgBuffers<'_>,
    options: PcgOptions,
    mut apply_operator: impl FnMut(&[f64], &mut [f64]),
    mut apply_preconditioner: impl FnMut(&[f64], &mut [f64]),
) -> SchurAttempt {
    let PcgBuffers {
        solution,
        residual,
        preconditioned,
        direction,
        product,
    } = buffers;
    solution.fill(0.0);
    for (value, gradient) in residual.iter_mut().zip(gradient) {
        *value = -*gradient;
    }
    let initial_euclidean_norm = norm(residual);
    if !initial_euclidean_norm.is_finite() {
        return rejected(0, None);
    }
    if initial_euclidean_norm == 0.0 {
        product.fill(0.0);
        return accepted(0, 0.0);
    }

    apply_preconditioner(residual, preconditioned);
    direction.copy_from_slice(preconditioned);
    let mut rho = dot(residual, preconditioned);
    if !(rho.is_finite() && rho > 0.0) {
        return rejected(0, None);
    }

    for iteration in 1..=options.maximum_iterations {
        apply_operator(direction, product);
        let curvature = dot(direction, product);
        if !(curvature.is_finite() && curvature > 0.0) {
            return rejected(iteration - 1, None);
        }
        let alpha = rho / curvature;
        if !alpha.is_finite() {
            return rejected(iteration - 1, None);
        }
        for index in 0..solution.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }

        let recursive_ratio = norm(residual) / initial_euclidean_norm;
        if !recursive_ratio.is_finite() {
            return rejected(iteration, None);
        }
        if options.require_bounded_residual && recursive_ratio <= options.relative_tolerance {
            recompute_residual(gradient, solution, residual, product, &mut apply_operator);
            let true_ratio = norm(residual) / initial_euclidean_norm;
            if true_ratio.is_finite() && true_ratio <= options.relative_tolerance {
                return accepted(iteration, true_ratio);
            }
            if !true_ratio.is_finite() {
                return rejected(iteration, None);
            }
            apply_preconditioner(residual, preconditioned);
            direction.copy_from_slice(preconditioned);
            rho = dot(residual, preconditioned);
            if !(rho.is_finite() && rho > 0.0) {
                return rejected(iteration, Some(true_ratio));
            }
            continue;
        }

        apply_preconditioner(residual, preconditioned);
        let next_rho = dot(residual, preconditioned);
        if !(next_rho.is_finite() && next_rho > 0.0) {
            return rejected(iteration, None);
        }
        let beta = next_rho / rho;
        if !beta.is_finite() {
            return rejected(iteration, None);
        }
        for (direction, preconditioned) in direction.iter_mut().zip(&*preconditioned) {
            *direction = *preconditioned + beta * *direction;
        }
        rho = next_rho;
    }

    recompute_residual(gradient, solution, residual, product, &mut apply_operator);
    let relative_residual = norm(residual) / initial_euclidean_norm;
    if relative_residual.is_finite() && !options.require_bounded_residual {
        accepted(options.maximum_iterations, relative_residual)
    } else {
        rejected(
            options.maximum_iterations,
            relative_residual.is_finite().then_some(relative_residual),
        )
    }
}

fn recompute_residual(
    gradient: &[f64],
    solution: &[f64],
    residual: &mut [f64],
    product: &mut [f64],
    apply_operator: &mut impl FnMut(&[f64], &mut [f64]),
) {
    apply_operator(solution, product);
    for ((residual, gradient), product) in residual.iter_mut().zip(gradient).zip(product) {
        *residual = -*gradient - *product;
    }
}

fn accepted(terms: usize, relative_residual: f64) -> SchurAttempt {
    SchurAttempt::Accepted(SchurSolve {
        terms,
        relative_residual,
    })
}

fn rejected(terms: usize, relative_residual: Option<f64>) -> SchurAttempt {
    SchurAttempt::Rejected(RejectedSchurSolve {
        terms,
        relative_residual,
    })
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_right_hand_side_leaves_a_certified_zero_product() {
        let mut solution = [1.0; 2];
        let mut residual = [1.0; 2];
        let mut preconditioned = [1.0; 2];
        let mut direction = [1.0; 2];
        let mut product = [1.0; 2];
        let attempt = solve(
            &[0.0; 2],
            PcgBuffers {
                solution: &mut solution,
                residual: &mut residual,
                preconditioned: &mut preconditioned,
                direction: &mut direction,
                product: &mut product,
            },
            PcgOptions {
                maximum_iterations: 2,
                relative_tolerance: 1e-2,
                require_bounded_residual: true,
            },
            |_, output| output.fill(f64::NAN),
            |_, output| output.fill(f64::NAN),
        );
        let SchurAttempt::Accepted(evidence) = attempt else {
            panic!("zero right-hand side was rejected");
        };
        assert_eq!(evidence.terms, 0);
        assert_eq!(evidence.relative_residual, 0.0);
        assert_eq!(solution, [0.0; 2]);
        assert_eq!(product, [0.0; 2]);
    }
}
