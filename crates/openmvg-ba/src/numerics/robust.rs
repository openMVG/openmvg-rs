// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free robust losses for iteratively reweighted least squares.

/// Huber cost with transition point `delta`.
#[inline(always)]
pub fn huber_cost(r: f64, delta: f64) -> f64 {
    let magnitude = r.abs();
    if magnitude <= delta {
        0.5 * r * r
    } else {
        delta * (magnitude - 0.5 * delta)
    }
}

/// Huber IRLS sqrt-weight: multiply both residual and Jacobian by this value.
#[inline(always)]
pub fn huber_irls_sqrt_weight(r: f64, delta: f64) -> f64 {
    let magnitude = r.abs();
    if magnitude <= delta || magnitude == 0.0 {
        1.0
    } else {
        (delta / magnitude).sqrt()
    }
}

/// Cauchy cost `ln(1 + (r / scale)^2 / 2)`.
#[inline(always)]
pub fn cauchy_cost(r: f64, scale: f64) -> f64 {
    let x2 = (r / scale) * (r / scale);
    (0.5 * x2).ln_1p()
}

/// Cauchy IRLS sqrt-weight: multiply both residual and Jacobian by this value.
///
/// The loss is redescending and the weight may reach zero in the numerical far
/// tail. No positive floor is applied, so the weighted gradient remains the
/// derivative of [`cauchy_cost`], which is the objective used for LM acceptance.
#[inline(always)]
pub fn cauchy_irls_sqrt_weight(r: f64, scale: f64) -> f64 {
    let x2 = (r / scale) * (r / scale);
    (1.0 / (scale * scale * (0.5 * x2 + 1.0))).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cauchy_downweights_large_residuals() {
        let w_in = cauchy_irls_sqrt_weight(0.001, 0.01);
        let w_out = cauchy_irls_sqrt_weight(0.5, 0.01);
        assert!(w_out < w_in * 0.1);
    }

    #[test]
    fn cauchy_weight_matches_the_cost_derivative_across_residual_decades() {
        let scale = 0.01;
        for exponent in -4..=4 {
            let residual = scale * 10f64.powi(exponent);
            let epsilon = 1.0e-6 * residual;
            let numerical = (cauchy_cost(residual + epsilon, scale)
                - cauchy_cost(residual - epsilon, scale))
                / (2.0 * epsilon);
            let sqrt_weight = cauchy_irls_sqrt_weight(residual, scale);
            let analytic = sqrt_weight * sqrt_weight * residual;
            let rounding = 4.0 * f64::EPSILON * cauchy_cost(residual, scale) / epsilon;
            assert!(
                (numerical - analytic).abs() <= 1.0e-6 * analytic.abs() + rounding,
                "residual {residual}: {numerical} vs {analytic}"
            );
        }
    }

    #[test]
    fn cauchy_far_tail_weight_is_not_floored() {
        let sqrt_weight = cauchy_irls_sqrt_weight(1.0e18, 0.01);
        assert!(sqrt_weight * sqrt_weight < 1.0e-14);
    }

    #[test]
    fn huber_weight_matches_cost_derivative_on_both_branches() {
        let delta = 0.01;
        let epsilon = 1e-8;
        for residual in [0.004, 0.04, -0.04] {
            let numerical = (huber_cost(residual + epsilon, delta)
                - huber_cost(residual - epsilon, delta))
                / (2.0 * epsilon);
            let sqrt_weight = huber_irls_sqrt_weight(residual, delta);
            let analytic = sqrt_weight * sqrt_weight * residual;
            assert!((numerical - analytic).abs() < 1e-10);
        }
    }

    #[test]
    fn every_loss_is_zero_at_zero_even_and_nondecreasing() {
        let scale = 0.01;
        let costs: [&dyn Fn(f64) -> f64; 2] =
            [&|r| huber_cost(r, scale), &|r| cauchy_cost(r, scale)];
        for cost in costs {
            assert_eq!(cost(0.0), 0.0);
            let mut previous = 0.0;
            for exponent in -8..=8 {
                let residual = scale * 10f64.powi(exponent);
                assert_eq!(cost(residual), cost(-residual));
                assert!(cost(residual) >= previous);
                previous = cost(residual);
            }
        }
    }

    #[test]
    fn cauchy_cost_keeps_full_precision_for_tiny_residuals() {
        for residual in [1.0e-5, 1.0e-9, 1.0e-150] {
            let expected = 0.5 * residual * residual;
            let actual = cauchy_cost(residual, 1.0);
            assert!((actual - expected).abs() <= 1.0e-9 * expected);
        }
    }

    #[test]
    fn huber_is_continuously_differentiable_at_its_transition() {
        let delta = 0.01;
        let epsilon = 1.0e-12;
        let below = huber_cost(delta - epsilon, delta);
        let above = huber_cost(delta + epsilon, delta);
        assert!((above - below - 2.0 * epsilon * delta).abs() <= 1.0e-20);
        assert_eq!(huber_irls_sqrt_weight(delta, delta), 1.0);
        assert!((huber_irls_sqrt_weight(delta + epsilon, delta) - 1.0).abs() <= 1.0e-9);
    }
}
