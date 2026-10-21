// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small, allocation-free polynomial primitives shared by minimal solvers.
//!
//! These routines expose distinct policies on purpose: P3P sometimes needs
//! one well-conditioned real cubic root, whereas the seven-point solver needs
//! every real root. Conflating those contracts is numerically unsafe.

/// Stable real roots of `x^2 + b*x + c = 0`.
#[inline]
pub(crate) fn quadratic_real_monic(b: f64, c: f64) -> Option<[f64; 2]> {
    quadratic_real_monic_computed(b, c, c.abs())
}

/// [`quadratic_real_monic`] for a constant term formed by arithmetic whose
/// rounding error scales with `c_scale`, rather than with the rounded `|c|`.
#[inline]
fn quadratic_real_monic_computed(b: f64, c: f64, c_scale: f64) -> Option<[f64; 2]> {
    if !b.is_finite() || !c.is_finite() {
        return None;
    }
    let discriminant = b.mul_add(b, -4.0 * c);
    let discriminant_scale = b * b + 4.0 * c_scale;
    let tolerance = 32.0 * f64::EPSILON * discriminant_scale.max(f64::MIN_POSITIVE);
    if discriminant < -tolerance || !discriminant.is_finite() {
        return None;
    }
    let discriminant = discriminant.max(0.0);
    if discriminant == 0.0 {
        return Some([-0.5 * b; 2]);
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    if q == 0.0 {
        Some([0.0; 2])
    } else {
        Some([q, c / q])
    }
}

/// Fast full-precision cube root used by the Cardano branch.
#[inline]
pub(crate) fn fast_cbrt(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let magnitude = value.abs();
    if !magnitude.is_finite() || magnitude < f64::MIN_POSITIVE {
        return value.cbrt();
    }
    let mut root = f64::from_bits(magnitude.to_bits() / 3 + 0x2A9F_7625_3119_D328);
    for _ in 0..2 {
        let cube = root * root * root;
        if !cube.is_finite() || cube == 0.0 {
            return value.cbrt();
        }
        let ratio = magnitude / cube;
        root *= (1.0 + 2.0 * ratio) / (2.0 + ratio);
    }
    root.copysign(value)
}

/// Cancellation-free Cardano evaluation for a depressed cubic with one real
/// root. The sign keeps the cube-root argument away from zero.
#[inline]
fn cardano_single_real(p: f64, half_q: f64, discriminant: f64) -> f64 {
    let u = fast_cbrt(-half_q - discriminant.sqrt().copysign(half_q));
    u - p / (3.0 * u)
}

/// One real root of a monic cubic and whether it has a single real branch.
#[inline]
pub(crate) fn cubic_real_branch(b: f64, c: f64, d: f64) -> Option<(f64, bool)> {
    if !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return None;
    }
    if b == 0.0 && c == 0.0 && d == 0.0 {
        return Some((0.0, false));
    }
    let variable_scale = cubic_variable_scale(b, c, d);
    let (b, c, d) = if variable_scale == 1.0 {
        (b, c, d)
    } else {
        let inverse = variable_scale.recip();
        let inverse_squared = inverse * inverse;
        (
            b * inverse,
            c * inverse_squared,
            d * inverse_squared * inverse,
        )
    };
    let (root, single_real) = cubic_real_branch_normalized(b, c, d)?;
    let root = root * variable_scale;
    root.is_finite().then_some((root, single_real))
}

/// One branch of a finite, scale-conditioned monic cubic.
#[inline]
fn cubic_real_branch_normalized(b: f64, c: f64, d: f64) -> Option<(f64, bool)> {
    let b3 = b / 3.0;
    let p = c - b * b3;
    let q = d - b3 * (c - 2.0 * b3 * b3);
    let half_q = 0.5 * q;
    let discriminant = half_q * half_q + p * p * p / 27.0;
    let (mut root, single_real) = if p == 0.0 && q == 0.0 {
        (-b3, false)
    } else if discriminant > 0.0 {
        (cardano_single_real(p, half_q, discriminant) - b3, true)
    } else {
        let radius = (-p / 3.0).sqrt();
        let denominator = (-(p * p * p) / 27.0).sqrt();
        if denominator == 0.0 {
            return None;
        }
        let angle = (-half_q / denominator).clamp(-1.0, 1.0).acos();
        (2.0 * radius * (angle / 3.0).cos() - b3, false)
    };
    for _ in 0..2 {
        let value = ((root + b) * root + c) * root + d;
        let root_magnitude = root.abs();
        let value_scale = root_magnitude * root_magnitude * root_magnitude
            + b.abs() * root_magnitude * root_magnitude
            + c.abs() * root_magnitude
            + d.abs();
        if value.abs() <= 8.0 * f64::EPSILON * value_scale.max(f64::MIN_POSITIVE) {
            break;
        }
        let derivative = (3.0 * root + 2.0 * b) * root + c;
        if !value.is_finite() || !derivative.is_finite() || derivative == 0.0 {
            break;
        }
        let candidate = root - value / derivative;
        if !candidate.is_finite() {
            break;
        }
        root = candidate;
    }
    root.is_finite().then_some((root, single_real))
}

/// One polished real root of `x^3 + b*x^2 + c*x + d = 0`.
///
/// The single-real branch uses the fast Cardano cube root. The three-real
/// branch starts outside the nearest turning point, which is the branch needed
/// by the conic-pencil P3P reduction. Coefficient and variable scaling keeps
/// both paths finite before four bounded Newton steps.
#[inline]
pub(crate) fn polished_cubic_root(b: f64, c: f64, d: f64) -> Option<f64> {
    if !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return None;
    }
    if b == 0.0 && c == 0.0 && d == 0.0 {
        return Some(0.0);
    }
    let variable_scale = cubic_variable_scale(b, c, d);
    let (b, c, d) = if variable_scale == 1.0 {
        (b, c, d)
    } else {
        let inverse = variable_scale.recip();
        let inverse_squared = inverse * inverse;
        (
            b * inverse,
            c * inverse_squared,
            d * inverse_squared * inverse,
        )
    };
    let b3 = b / 3.0;
    let p = c - b * b3;
    let q = d + b3 * (2.0 * b3 * b3 - c);
    let half_q = 0.5 * q;
    let discriminant = half_q * half_q + p * p * p / 27.0;
    let mut root = if discriminant > 0.0 {
        cardano_single_real(p, half_q, discriminant) - b3
    } else if b * b >= 3.0 * c {
        let turning_span = (b * b - 3.0 * c).sqrt();
        let left = (-b - turning_span) / 3.0;
        let left_value = ((left + b) * left + c) * left + d;
        if left_value > 0.0 {
            let denominator = 3.0 * left + b;
            let radicand = -left_value / denominator;
            if denominator == 0.0 || radicand < 0.0 {
                return None;
            }
            left - radicand.sqrt()
        } else {
            let right = (-b + turning_span) / 3.0;
            let right_value = ((right + b) * right + c) * right + d;
            let denominator = 3.0 * right + b;
            let radicand = -right_value / denominator;
            if denominator == 0.0 || radicand < 0.0 {
                return None;
            }
            right + radicand.sqrt()
        }
    } else {
        -b3
    };
    for _ in 0..4 {
        let value = ((root + b) * root + c) * root + d;
        if value.abs() < 1e-14 {
            break;
        }
        let derivative = (3.0 * root + 2.0 * b) * root + c;
        if !value.is_finite() || !derivative.is_finite() || derivative == 0.0 {
            break;
        }
        let candidate = root - value / derivative;
        if !candidate.is_finite() {
            break;
        }
        let candidate_value = ((candidate + b) * candidate + c) * candidate + d;
        if candidate_value.abs() > value.abs() {
            break;
        }
        root = candidate;
    }
    let root = root * variable_scale;
    root.is_finite().then_some(root)
}

/// Real roots of `a*x^4 + b*x^3 + c*x^2 + d*x + e = 0`.
///
/// An exactly absent quartic term falls back to the all-real-roots cubic
/// backend. Numerical degree policy belongs to the calling geometric solver:
/// a tiny nonzero leading coefficient can still encode a large finite root.
#[inline]
pub(crate) fn quartic_real(coefficients: [f64; 5], roots: &mut [f64; 4]) -> usize {
    if coefficients.iter().any(|value| !value.is_finite()) {
        return 0;
    }
    let coefficient_scale = coefficients
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if coefficient_scale == 0.0 {
        return 0;
    }
    let [a, b, c, d, e] = coefficients;
    if a == 0.0 {
        return cubic_real_into([b, c, d, e], roots);
    }
    let mut count = quartic_real_monic(b / a, c / a, d / a, e / a, roots);

    // Ferrari can lose a small root when root magnitudes span many orders. A
    // reciprocal chart is only worthwhile when the constant term signals that
    // imbalance; two real roots is otherwise the normal quartic outcome.
    let reciprocal_needed =
        count < roots.len() && e != 0.0 && e.abs() <= 4.0 * f64::EPSILON.sqrt() * coefficient_scale;
    if reciprocal_needed {
        let mut reciprocal_roots = [0.0; 4];
        let reciprocal_count =
            quartic_real_monic(d / e, c / e, b / e, a / e, &mut reciprocal_roots);
        for reciprocal in reciprocal_roots.into_iter().take(reciprocal_count) {
            if reciprocal == 0.0 {
                continue;
            }
            let candidate = reciprocal.recip();
            let duplicate = roots[..count].iter().any(|root| {
                (candidate - root).abs()
                    <= 64.0 * f64::EPSILON * (1.0 + candidate.abs().max(root.abs()))
            });
            if !duplicate && count < roots.len() {
                roots[count] = candidate;
                count += 1;
            }
        }
    }
    if !reciprocal_needed {
        return count;
    }
    let normalized = coefficients.map(|value| value / coefficient_scale);
    count = polish_filter_sort_dedup(&normalized, roots, count);
    if count == 3 {
        let product = roots[..count].iter().product::<f64>();
        let candidate = if product != 0.0 && product.is_finite() {
            (e / a) / product
        } else {
            -b / a - roots[..count].iter().sum::<f64>()
        };
        if candidate.is_finite() {
            roots[count] = candidate;
            count += 1;
            count = polish_filter_sort_dedup(&[a, b, c, d, e], roots, count);
        }
    }
    count
}

/// Real roots of a quartic produced from already conditioned geometry.
///
/// P3P formulations using this path produce bounded, scale-free coefficients
/// from unit bearings and normalized frames or ratios, so the scale-general
/// reciprocal chart and normalized backward-error bookkeeping in
/// [`quartic_real`] are unnecessary here. The bounded Ferrari/Newton path is
/// used by formulations whose pose recovery supplies final geometric
/// validation.
#[inline]
pub(crate) fn quartic_real_conditioned(coefficients: [f64; 5], roots: &mut [f64; 4]) -> usize {
    if coefficients.iter().any(|value| !value.is_finite()) {
        return 0;
    }
    let [a, b, c, d, e] = coefficients;
    if a == 0.0 {
        return quartic_real(coefficients, roots);
    }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;
    if quartic_variable_scale(b, c, d, e) != 1.0 {
        return quartic_real(coefficients, roots);
    }

    let b4 = 0.25 * b;
    let b4_squared = b4 * b4;
    let p = c - 6.0 * b4_squared;
    let q = d - 2.0 * c * b4 + 8.0 * b4_squared * b4;
    let r = e - d * b4 + c * b4_squared - 3.0 * b4_squared * b4_squared;
    let mut count = 0;
    if q == 0.0 {
        let discriminant = p.mul_add(p, -4.0 * r);
        let p_scale = c.abs() + 6.0 * b4_squared;
        let r_scale =
            e.abs() + (d * b4).abs() + (c * b4_squared).abs() + 3.0 * b4_squared * b4_squared;
        let tolerance =
            32.0 * f64::EPSILON * (p_scale * p_scale + 4.0 * r_scale).max(f64::MIN_POSITIVE);
        if discriminant <= 0.0 {
            if discriminant < -tolerance {
                return 0;
            }
            // A rounded tangent has no exact root to polish: Newton's
            // derivative vanishes at the intended double roots and can take
            // an unbounded step. Return the two geometric candidates for the
            // caller's final residual/cheirality validation.
            let squared_root = -0.5 * p;
            if squared_root < 0.0 {
                return 0;
            }
            let root = squared_root.sqrt();
            roots[0] = root - b4;
            if root > 0.0 {
                roots[1] = -root - b4;
                return 2;
            }
            return 1;
        }
        let span = discriminant.sqrt();
        for squared_root in [0.5 * (-p + span), 0.5 * (-p - span)] {
            if squared_root < 0.0 {
                continue;
            }
            let root = squared_root.sqrt();
            roots[count] = root - b4;
            count += 1;
            if root > 0.0 {
                roots[count] = -root - b4;
                count += 1;
            }
        }
    } else {
        let Some((u, _)) = cubic_real_branch_normalized(2.0 * p, p * p - 4.0 * r, -q * q) else {
            return 0;
        };
        if u > 1e-300 {
            let k = u.sqrt();
            let q_over_k = q / k;
            for (linear, constant) in [
                (k, 0.5 * (p + u - q_over_k)),
                (-k, 0.5 * (p + u + q_over_k)),
            ] {
                let discriminant = linear.mul_add(linear, -4.0 * constant);
                if discriminant < 0.0 {
                    // A near-tangent quadratic can round slightly below zero.
                    // Use the robust path only on this degenerate boundary.
                    let inverse_k = k.recip();
                    let p_scale = c.abs() + 6.0 * b4_squared;
                    let q_scale = d.abs() + 2.0 * (c * b4).abs() + 8.0 * (b4_squared * b4).abs();
                    let constant_scale = 0.5 * (p_scale + u + q_scale * inverse_k);
                    let tolerance = 32.0
                        * f64::EPSILON
                        * (linear * linear + 4.0 * constant_scale).max(f64::MIN_POSITIVE);
                    if discriminant >= -tolerance {
                        return quartic_real(coefficients, roots);
                    }
                    continue;
                }
                let span = discriminant.sqrt();
                roots[count] = 0.5 * (-linear + span) - b4;
                roots[count + 1] = 0.5 * (-linear - span) - b4;
                count += 2;
            }
        }
    }

    let mut valid = 0;
    for index in 0..count {
        let mut root = roots[index];
        for _ in 0..2 {
            let value = (((root + b) * root + c) * root + d) * root + e;
            let derivative = ((4.0 * root + 3.0 * b) * root + 2.0 * c) * root + d;
            if derivative.abs() < 1e-300 {
                break;
            }
            let step = value / derivative;
            root -= step;
            if step.abs() < 1e-15 * (1.0 + root.abs()) {
                break;
            }
        }
        if !root.is_finite() {
            return quartic_real(coefficients, roots);
        }
        roots[valid] = root;
        valid += 1;
    }
    if valid == 4 {
        // Four finite Ferrari candidates can still contain a duplicated
        // simple root near tangency. Vieta gives a cheap hot-path audit.
        let root_sum = roots.iter().sum::<f64>();
        let sum_scale = 1.0 + b.abs() + roots.iter().map(|root| root.abs()).sum::<f64>();
        if (root_sum + b).abs() > 256.0 * f64::EPSILON * sum_scale {
            return quartic_real(coefficients, roots);
        }
    }
    valid
}

/// All distinct real roots of `a*x^3 + b*x^2 + c*x + d = 0`.
#[inline]
pub(crate) fn cubic_real(coefficients: [f64; 4], roots: &mut [f64; 3]) -> usize {
    cubic_real_into(coefficients, roots)
}

#[inline]
fn cubic_real_into<const N: usize>(coefficients: [f64; 4], roots: &mut [f64; N]) -> usize {
    debug_assert!(N >= 3);
    if coefficients.iter().any(|value| !value.is_finite()) {
        return 0;
    }
    let coefficient_scale = coefficients
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if coefficient_scale == 0.0 {
        return 0;
    }
    let [a, b, c, d] = coefficients.map(|value| value / coefficient_scale);
    if a == 0.0 {
        return quadratic_real([b, c, d], roots);
    }

    let b = b / a;
    let c = c / a;
    let d = d / a;
    let variable_scale = cubic_variable_scale(b, c, d);
    let (b, c, d) = if variable_scale == 1.0 {
        (b, c, d)
    } else {
        let inverse = variable_scale.recip();
        let inverse_squared = inverse * inverse;
        (
            b * inverse,
            c * inverse_squared,
            d * inverse_squared * inverse,
        )
    };
    let offset = b / 3.0;
    let p = c - b * offset;
    let q = d - offset * (c - 2.0 * offset * offset);
    let half_q = 0.5 * q;
    let p_cubed = p * p * p / 27.0;
    let discriminant = half_q * half_q + p_cubed;
    let tolerance = 64.0 * f64::EPSILON * (half_q * half_q + p_cubed.abs()).max(f64::MIN_POSITIVE);

    let mut count = if discriminant > tolerance {
        roots[0] = cardano_single_real(p, half_q, discriminant) - offset;
        1
    } else if discriminant < -tolerance {
        let radius = 2.0 * (-p / 3.0).sqrt();
        let denominator = (-(p * p * p) / 27.0).sqrt();
        if denominator == 0.0 {
            return 0;
        }
        let angle = (-half_q / denominator).clamp(-1.0, 1.0).acos() / 3.0;
        let (sine, cosine) = angle.sin_cos();
        let side = 0.866_025_403_784_438_6 * sine;
        let opposite = -0.5 * cosine;
        roots[0] = radius * cosine - offset;
        roots[1] = radius * (opposite + side) - offset;
        roots[2] = radius * (opposite - side) - offset;
        3
    } else {
        let u = fast_cbrt(-half_q);
        roots[0] = 2.0 * u - offset;
        roots[1] = -u - offset;
        2
    };
    count = polish_filter_sort_dedup(&[1.0, b, c, d], roots, count);
    for root in roots.iter_mut().take(count) {
        *root *= variable_scale;
    }
    count
}

fn quadratic_real<const N: usize>(coefficients: [f64; 3], roots: &mut [f64; N]) -> usize {
    debug_assert!(N >= 2);
    let [a, b, c] = coefficients;
    let scale = a.abs().max(b.abs()).max(c.abs());
    if a.abs() <= 64.0 * f64::EPSILON * scale {
        if b.abs() <= 64.0 * f64::EPSILON * scale {
            return 0;
        }
        roots[0] = -c / b;
        return 1;
    }
    let Some(pair) = quadratic_real_monic(b / a, c / a) else {
        return 0;
    };
    roots[..2].copy_from_slice(&pair);
    polish_filter_sort_dedup(&[a, b, c], roots, 2)
}

#[inline]
fn polish_filter_sort_dedup<const N: usize, const C: usize>(
    coefficients: &[f64; C],
    roots: &mut [f64; N],
    count: usize,
) -> usize {
    let mut valid = 0;
    for index in 0..count {
        let mut root = roots[index];
        if !root.is_finite() {
            continue;
        }
        let mut best_root = root;
        let mut best_residual = f64::INFINITY;
        let mut unevaluated_candidate = false;
        for _ in 0..2 {
            let (value, derivative, value_scale, derivative_scale) =
                polynomial_value_derivative(coefficients, root);
            unevaluated_candidate = false;
            let residual = value.abs() / value_scale.max(f64::MIN_POSITIVE);
            if residual < best_residual {
                best_root = root;
                best_residual = residual;
            }
            if best_residual <= 8.0 * f64::EPSILON {
                break;
            }
            if !value.is_finite() || !derivative.is_finite() {
                break;
            }
            if derivative.abs() <= 16.0 * f64::EPSILON * derivative_scale.max(f64::MIN_POSITIVE) {
                break;
            }
            let candidate = root - value / derivative;
            if !candidate.is_finite() {
                break;
            }
            root = candidate;
            unevaluated_candidate = true;
        }
        if unevaluated_candidate {
            let (value, _, value_scale, _) = polynomial_value_derivative(coefficients, root);
            let residual = value.abs() / value_scale.max(f64::MIN_POSITIVE);
            if residual < best_residual {
                best_root = root;
                best_residual = residual;
            }
        }
        if best_residual <= 64.0 * f64::EPSILON.sqrt() {
            roots[valid] = best_root;
            valid += 1;
        }
    }

    // At most four roots: insertion sort has lower fixed overhead than the
    // general slice sorter and preserves deterministic total ordering.
    for index in 1..valid {
        let root = roots[index];
        let mut destination = index;
        while destination > 0 && root.total_cmp(&roots[destination - 1]).is_lt() {
            roots[destination] = roots[destination - 1];
            destination -= 1;
        }
        roots[destination] = root;
    }
    let mut unique = 0;
    for index in 0..valid {
        let root = roots[index];
        let duplicate = unique > 0
            && (root - roots[unique - 1]).abs()
                <= 64.0 * f64::EPSILON * (1.0 + root.abs().max(roots[unique - 1].abs()));
        if !duplicate {
            roots[unique] = root;
            unique += 1;
        }
    }
    unique
}

#[inline]
fn polynomial_value_derivative<const C: usize>(
    coefficients: &[f64; C],
    root: f64,
) -> (f64, f64, f64, f64) {
    let mut value = coefficients[0];
    let mut derivative = 0.0_f64;
    let mut value_scale = coefficients[0].abs();
    let mut derivative_scale = 0.0_f64;
    let root_magnitude = root.abs();
    for &coefficient in &coefficients[1..] {
        derivative = derivative.mul_add(root, value);
        derivative_scale = derivative_scale.mul_add(root_magnitude, value_scale);
        value = value.mul_add(root, coefficient);
        value_scale = value_scale.mul_add(root_magnitude, coefficient.abs());
    }
    (value, derivative, value_scale, derivative_scale)
}

/// Real roots of `x^4 + b*x^3 + c*x^2 + d*x + e = 0`.
///
/// Uses Ferrari factorization followed by two Newton polishing iterations, as
/// specified by Ke and Roumeliotis. Returns the initialized prefix of `roots`.
pub(crate) fn quartic_real_monic(b: f64, c: f64, d: f64, e: f64, roots: &mut [f64; 4]) -> usize {
    if [b, c, d, e].iter().any(|value| !value.is_finite()) {
        return 0;
    }
    let variable_scale = quartic_variable_scale(b, c, d, e);
    let (b, c, d, e) = if variable_scale == 1.0 {
        (b, c, d, e)
    } else {
        let inverse = variable_scale.recip();
        let inverse_squared = inverse * inverse;
        (
            b * inverse,
            c * inverse_squared,
            d * inverse_squared * inverse,
            e * inverse_squared * inverse_squared,
        )
    };
    let b4 = 0.25 * b;
    let b4_squared = b4 * b4;
    let p = c - 6.0 * b4_squared;
    let q = d - 2.0 * c * b4 + 8.0 * b4_squared * b4;
    let r = e - d * b4 + c * b4_squared - 3.0 * b4_squared * b4_squared;

    let mut count = 0;
    // Keep the biquadratic shortcut exact. A scale-dependent tolerance can
    // erase a meaningful linear term when every root is uniformly small.
    if q == 0.0 {
        let discriminant = p.mul_add(p, -4.0 * r);
        let discriminant_tolerance =
            32.0 * f64::EPSILON * (p * p + 4.0 * r.abs()).max(f64::MIN_POSITIVE);
        if discriminant < -discriminant_tolerance {
            return 0;
        }
        let span = discriminant.max(0.0).sqrt();
        for squared_root in [0.5 * (-p + span), 0.5 * (-p - span)] {
            if squared_root < 0.0 {
                continue;
            }
            let root = squared_root.sqrt();
            roots[count] = root - b4;
            count += 1;
            if root > 0.0 {
                roots[count] = -root - b4;
                count += 1;
            }
        }
    } else {
        let Some((u, _)) = cubic_real_branch(2.0 * p, p * p - 4.0 * r, -q * q) else {
            return 0;
        };
        // The resolvent root u has units x^2.
        let u_tolerance =
            64.0 * f64::EPSILON * (u.abs() + p.abs() + r.abs().sqrt()).max(f64::MIN_POSITIVE);
        if u >= -u_tolerance {
            let k = u.max(0.0).sqrt();
            if k <= f64::MIN_POSITIVE {
                return 0;
            }
            let inverse_k = k.recip();
            let q_over_k = q * inverse_k;
            let p_scale = c.abs() + 6.0 * b4_squared;
            let q_scale = d.abs() + 2.0 * (c * b4).abs() + 8.0 * (b4_squared * b4).abs();
            let constant_scale = 0.5 * (p_scale + u + q_scale * inverse_k);
            for (linear, constant) in [
                (k, 0.5 * (p + u - q_over_k)),
                (-k, 0.5 * (p + u + q_over_k)),
            ] {
                if let Some(pair) = quadratic_real_monic_computed(linear, constant, constant_scale)
                {
                    for root in pair {
                        roots[count] = root - b4;
                        count += 1;
                    }
                }
            }
        }
    }

    let count = polish_filter_sort_dedup(&[1.0, b, c, d, e], roots, count);
    for root in roots.iter_mut().take(count) {
        *root *= variable_scale;
    }
    count
}

#[inline]
fn cubic_variable_scale(b: f64, c: f64, d: f64) -> f64 {
    if b == 0.0 && c == 0.0 && d == 0.0 {
        return 1.0;
    }
    let large = b.abs() > 1e50 || c.abs() > 1e100 || d.abs() > 1e150;
    let tiny = b.abs() < 1e-50 && c.abs() < 1e-100 && d.abs() < 1e-150;
    if large || tiny {
        b.abs().max(c.abs().sqrt()).max(d.abs().cbrt())
    } else {
        1.0
    }
}

#[inline]
fn quartic_variable_scale(b: f64, c: f64, d: f64, e: f64) -> f64 {
    if b == 0.0 && c == 0.0 && d == 0.0 && e == 0.0 {
        return 1.0;
    }
    let large = b.abs() > 1e37 || c.abs() > 1e74 || d.abs() > 1e111 || e.abs() > 1e148;
    let tiny = b.abs() < 1e-37 && c.abs() < 1e-74 && d.abs() < 1e-111 && e.abs() < 1e-148;
    if large || tiny {
        b.abs()
            .max(c.abs().sqrt())
            .max(d.abs().cbrt())
            .max(e.abs().sqrt().sqrt())
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests;
