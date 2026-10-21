// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn coefficients_from_roots(roots: &[f64]) -> Vec<f64> {
    let mut coefficients = vec![1.0];
    for &root in roots {
        let mut next = vec![0.0; coefficients.len() + 1];
        for (index, coefficient) in coefficients.iter().copied().enumerate() {
            next[index] += coefficient;
            next[index + 1] -= root * coefficient;
        }
        coefficients = next;
    }
    coefficients
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

fn assert_relative_eq(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn quadratic_handles_large_cancellation() {
    let roots = quadratic_real_monic(-1.0e8, 1.0).unwrap();
    assert!(roots.iter().any(|root| (*root - 1.0e8).abs() < 1e-7));
    assert!(roots.iter().any(|root| (*root - 1.0e-8).abs() < 1e-20));
}

#[test]
fn fast_cube_root_matches_stdlib_across_normal_exponents() {
    for exponent in (-1022..=1023).step_by(17) {
        for sign in [-1.0, 1.0] {
            let value = sign * 1.234_567_890_123_45 * 2.0_f64.powi(exponent);
            let expected = value.cbrt();
            let relative_error = ((fast_cbrt(value) - expected) / expected).abs();
            assert!(
                relative_error <= 4e-15,
                "value={value:e}, relative_error={relative_error:e}"
            );
        }
    }
    assert_eq!(fast_cbrt(0.0), 0.0);
    assert_eq!(fast_cbrt(f64::from_bits(1)), f64::from_bits(1).cbrt());
}

#[test]
fn cubic_returns_all_distinct_and_repeated_real_roots() {
    let mut roots = [0.0; 3];
    let count = cubic_real([1.0, -6.0, 11.0, -6.0], &mut roots);
    assert_eq!(count, 3);
    for (actual, expected) in roots.into_iter().take(count).zip([1.0, 2.0, 3.0]) {
        assert_relative_eq(actual, expected, 1e-12);
    }

    let count = cubic_real([1.0, -3.0, 0.0, 4.0], &mut roots);
    assert_eq!(count, 2);
    assert_relative_eq(roots[0], -1.0, 1e-12);
    assert_relative_eq(roots[1], 2.0, 1e-12);
}

#[test]
fn polished_cubic_root_is_scale_stable() {
    for variable_scale in [1e-100, 1.0, 1e100] {
        let d = variable_scale * variable_scale * variable_scale;
        let root = polished_cubic_root(0.0, 0.0, d).unwrap();
        assert_relative_eq(root / variable_scale, -1.0, 1e-12);
    }
}

#[test]
fn cubic_branch_polishes_cardano_cancellation() {
    // Ferrari's resolvent for this valid P3P quartic has a small root whose
    // Cardano terms nearly cancel. The residual-gated polish must recover
    // it accurately enough for quartic factorization.
    let [a, b, c, d, e]: [f64; 5] = [
        0.247_643_784_614_227_2,
        -1.025_804_021_169_005_6,
        1.529_019_286_688_232,
        -0.966_681_968_554_496_1,
        0.214_345_028_724_083_7,
    ];
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;
    let p = c - 3.0 * b * b / 8.0;
    let q = d - 0.5 * b * c + b * b * b / 8.0;
    let r = e - 0.25 * b * d + b * b * c / 16.0 - 3.0 * b.powi(4) / 256.0;
    let cubic_b = 2.0 * p;
    let cubic_c = p * p - 4.0 * r;
    let cubic_d = -q * q;

    let (root, single_real) = cubic_real_branch(cubic_b, cubic_c, cubic_d).unwrap();
    assert!(single_real);
    let residual = ((root + cubic_b) * root + cubic_c) * root + cubic_d;
    let magnitude = root.abs();
    let value_scale = magnitude.powi(3)
        + cubic_b.abs() * magnitude.powi(2)
        + cubic_c.abs() * magnitude
        + cubic_d.abs();
    assert!(residual.abs() <= 32.0 * f64::EPSILON * value_scale);
}

#[test]
fn one_root_cubics_recover_scaled_known_real_root() {
    let mut state = 0x5033_c0de_c4ad_a110;
    for exponent in -40..=40 {
        for _ in 0..64 {
            let scale = 2.0_f64.powi(exponent);
            let expected = (-1.0 + 2.0 * next_unit(&mut state)) * scale;
            let quadratic_b = (0.1 + 0.3 * next_unit(&mut state)) * scale;
            let quadratic_c = (1.0 + next_unit(&mut state)) * scale * scale;
            let b = quadratic_b - expected;
            let c = quadratic_c - expected * quadratic_b;
            let d = -expected * quadratic_c;

            let (branch_root, single_real) = cubic_real_branch(b, c, d).unwrap();
            assert!(single_real);
            assert_relative_eq(branch_root, expected, 2e-12);
            assert_relative_eq(polished_cubic_root(b, c, d).unwrap(), expected, 2e-12);
        }
    }
}

#[test]
fn quartic_recovers_four_real_roots() {
    // (x-1)(x-2)(x-3)(x-4)
    let mut roots = [0.0; 4];
    let count = quartic_real_monic(-10.0, 35.0, -50.0, 24.0, &mut roots);
    assert_eq!(count, 4);
    roots.sort_by(f64::total_cmp);
    for (actual, expected) in roots.into_iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn quartic_is_scale_invariant_and_handles_wide_root_magnitudes() {
    let expected = [1e-6, 2e-3, 3.0, 4e3];
    let coefficients = coefficients_from_roots(&expected);
    let coefficients: [f64; 5] = coefficients.try_into().unwrap();
    for coefficient_scale in [1e-200, 1.0, 1e200] {
        let mut roots = [0.0; 4];
        let count = quartic_real(
            coefficients.map(|coefficient| coefficient * coefficient_scale),
            &mut roots,
        );
        assert_eq!(count, 4);
        for (actual, expected) in roots.into_iter().take(count).zip(expected) {
            assert_relative_eq(actual, expected, 2e-9);
        }
    }
}

#[test]
fn quartic_degree_fallback_uses_all_real_cubic_roots() {
    let mut roots = [0.0; 4];
    let count = quartic_real([0.0, 1.0, -6.0, 11.0, -6.0], &mut roots);
    assert_eq!(count, 3);
    for (actual, expected) in roots.into_iter().take(count).zip([1.0, 2.0, 3.0]) {
        assert_relative_eq(actual, expected, 1e-12);
    }
}

#[test]
fn all_root_backends_recover_scaled_random_known_roots() {
    let mut state = 0x5033_c0de_7a11_0ca7;
    for index in 0..4_096 {
        let variable_scale = 2.0_f64.powi(index % 81 - 40);
        let coefficient_scale = 2.0_f64.powi(index % 1_001 - 500);
        let expected = [
            (-4.0 + 0.5 * next_unit(&mut state)) * variable_scale,
            (-1.5 + 0.5 * next_unit(&mut state)) * variable_scale,
            (0.2 + 0.5 * next_unit(&mut state)) * variable_scale,
            (2.0 + 2.0 * next_unit(&mut state)) * variable_scale,
        ];
        let cubic_coefficients = coefficients_from_roots(&expected[..3]);
        let coefficients = coefficients_from_roots(&expected);

        let mut cubic_roots = [0.0; 3];
        let cubic_count = cubic_real(
            std::array::from_fn(|coefficient| cubic_coefficients[coefficient] * coefficient_scale),
            &mut cubic_roots,
        );
        assert_eq!(cubic_count, 3);
        for (actual, expected) in cubic_roots.into_iter().zip(expected) {
            assert_relative_eq(actual, expected, 2e-10);
        }

        let mut quartic_roots = [0.0; 4];
        let quartic_count = quartic_real(
            std::array::from_fn(|coefficient| coefficients[coefficient] * coefficient_scale),
            &mut quartic_roots,
        );
        assert_eq!(
            quartic_count, 4,
            "index={index}, roots={expected:?}, coefficients={coefficients:?}, scale={coefficient_scale:e}"
        );
        for (actual, expected) in quartic_roots.into_iter().zip(expected) {
            assert_relative_eq(actual, expected, 2e-9);
        }
    }
}

#[test]
fn quartic_handles_repeated_and_complex_conjugate_roots() {
    let mut roots = [0.0; 4];
    let count = quartic_real([1.0, -2.0, -3.0, 4.0, 4.0], &mut roots);
    assert_eq!(count, 2);
    assert_relative_eq(roots[0], -1.0, 1e-10);
    assert_relative_eq(roots[1], 2.0, 1e-10);

    // (x - 1)(x + 2)(x^2 + 1): two real and one complex-conjugate pair.
    let count = quartic_real([1.0, 1.0, -1.0, 1.0, -2.0], &mut roots);
    assert_eq!(count, 2);
    assert_relative_eq(roots[0], -2.0, 1e-10);
    assert_relative_eq(roots[1], 1.0, 1e-10);
}

#[test]
fn quartic_preserves_large_and_uniformly_small_finite_roots() {
    for expected in [
        [1_000.0, 2_000.0, 3_000.0, 4_000.0],
        [1e-6, 2e-6, 4e-6, 8e-6],
    ] {
        let coefficients: [f64; 5] = coefficients_from_roots(&expected).try_into().unwrap();
        let mut roots = [0.0; 4];
        let count = quartic_real(coefficients, &mut roots);
        assert_eq!(count, 4);
        for (actual, expected) in roots.into_iter().zip(expected) {
            assert_relative_eq(actual, expected, 2e-9);
        }
    }
}

#[test]
fn conditioned_quartic_recovers_random_distinct_roots() {
    let mut state = 0x5033_c0de_f377_a111;
    for index in 0..4_096 {
        let scale = 2.0_f64.powi(index % 21 - 10);
        let expected = [
            (-4.0 + 0.5 * next_unit(&mut state)) * scale,
            (-1.5 + 0.5 * next_unit(&mut state)) * scale,
            (0.2 + 0.5 * next_unit(&mut state)) * scale,
            (2.0 + 2.0 * next_unit(&mut state)) * scale,
        ];
        let coefficients: [f64; 5] = coefficients_from_roots(&expected).try_into().unwrap();
        let mut actual = [0.0; 4];
        let count = quartic_real_conditioned(coefficients, &mut actual);
        assert_eq!(count, 4, "index={index}, coefficients={coefficients:?}");
        actual.sort_by(f64::total_cmp);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_relative_eq(actual, expected, 2e-9);
        }
    }
}

#[test]
fn conditioned_quartic_preserves_a_small_ferrari_resolvent_root() {
    let coefficients = [
        0.248_742_231_148_003_3,
        -1.030_254_955_874_838_2,
        1.535_432_123_873_65,
        -0.970_411_260_042_061_8,
        0.215_002_236_479_670_32,
    ];
    let mut roots = [0.0; 4];

    let count = quartic_real_conditioned(coefficients, &mut roots);

    assert_eq!(count, 2);
    roots[..count].sort_by(f64::total_cmp);
    assert_relative_eq(roots[0], 0.505_146_554_255_190_8, 1e-12);
    assert_relative_eq(roots[1], 1.564_333_705_911_879_1, 1e-12);
}

#[test]
fn conditioned_quartic_survives_cardano_cancellation_in_the_resolvent() {
    // Nakano P3P quartic of a benign scene. The resolvent's only real root
    // is ~5e-6; the canceling Cardano branch returned rounding noise and
    // both Ferrari quadratics lost their real roots.
    let coefficients = [
        0.250_115_895_589_928_5,
        -1.035_854_211_176_821_4,
        1.543_705_348_473_466_6,
        -0.975_585_561_818_199_9,
        0.216_122_132_607_010_8,
    ];
    let mut roots = [0.0; 4];

    let count = quartic_real_conditioned(coefficients, &mut roots);

    assert_eq!(count, 2);
    roots[..count].sort_by(f64::total_cmp);
    assert_relative_eq(roots[0], 0.504_958_210_127_287_1, 1e-12);
    assert_relative_eq(roots[1], 1.563_541_199_638_261_3, 1e-12);
}

#[test]
fn quartics_keep_tangent_double_roots() {
    // A double root is where two P3P solutions merge. Rounding can make
    // its Ferrari quadratic's discriminant slightly negative; it must
    // still be returned to the sqrt(epsilon) accuracy a double root allows.
    let mut state = 0x5033_c0de_d0b1_e000;
    for index in 0..4_096 {
        let double = -2.0 + 4.0 * next_unit(&mut state);
        let simple = [
            double + 0.5 + 2.0 * next_unit(&mut state),
            double - 0.5 - 2.0 * next_unit(&mut state),
        ];
        let coefficients: [f64; 5] =
            coefficients_from_roots(&[double, double, simple[0], simple[1]])
                .try_into()
                .unwrap();
        for (name, solver) in [
            (
                "general",
                quartic_real as fn([f64; 5], &mut [f64; 4]) -> usize,
            ),
            ("conditioned", quartic_real_conditioned),
        ] {
            let mut roots = [0.0; 4];
            let count = solver(coefficients, &mut roots);
            for expected in [double, simple[0], simple[1]] {
                assert!(
                    roots[..count]
                        .iter()
                        .any(|root| (root - expected).abs() <= 1e-6),
                    "{name} index={index}: {expected} not in {:?}",
                    &roots[..count]
                );
            }
        }
    }
}

#[test]
fn conditioned_biquadratic_keeps_a_rounded_tangent_root() {
    // This is (up to coefficient rounding)
    // ((x - h)^2 - a)^2. Depressing it produces q == 0 exactly, while
    // cancellation rounds p^2 - 4r to -2.3e-15.
    let coefficients = [
        1.0,
        -4.540_777_424_556_362,
        7.731_731_762_759_709,
        -5.850_943_505_006_703,
        1.660_318_397_546_811_9,
    ];
    let mut roots = [0.0; 4];

    let count = quartic_real_conditioned(coefficients, &mut roots);

    assert_eq!(count, 2);
    roots[..count].sort_by(f64::total_cmp);
    assert_relative_eq(roots[0], 1.123_670_587_212_285, 1e-6);
    assert_relative_eq(roots[1], 1.146_718_125_065_896, 1e-6);
}
