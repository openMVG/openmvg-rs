// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The P4Pf kernel against outputs exported from the project's Julia reference implementation:
//! bit-exact on its strata, and on the crafted inputs of the branches no stratum reaches.

use openmvg_geometry::pose::p4pf::test_support::{
    MAX_POSES, P4pfPose, house_nullspace, solve_p4pf_kernel,
};

/// The first 10 instances of every reference stratum (generic, image noise, world and image
/// scalings, the near-degenerate sets) plus every instance where the lane declines. One case per
/// line: `tag | x (4 x 3) uv (4 x 2) | status, number of poses, then R (row-major) t f reproj of
/// each pose`, as shortest round-trip decimals.
const FIXTURE: &str = include_str!("data/p4pf_reference.txt");

/// Crafted inputs of the null-space branches: `nullspace <name> | m k pivot tol W (m k,
/// row-major) | ratio | N (m (m - k), row-major)`, tol NaN = nothing.
const BRANCHES: &str = include_str!("data/p4pf_branches.txt");

fn nums(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}

fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same(status: i32, poses: &[P4pfPose], reference: &[f64]) -> bool {
    reference.len() == 2 + 14 * poses.len()
        && reference[0] as i32 == status
        && reference[1] as usize == poses.len()
        && poses.iter().enumerate().all(|(index, pose)| {
            pose.r
                .iter()
                .chain(pose.t.iter())
                .chain([pose.f, pose.reproj].iter())
                .zip(&reference[2 + 14 * index..])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        })
}

#[test]
fn the_kernel_is_bit_exact_against_the_reference_strata() {
    let (mut total, mut bad) = (0, Vec::new());
    for line in FIXTURE.lines() {
        let mut parts = line.split(" | ");
        let tag = parts.next().unwrap();
        let input = nums(parts.next().unwrap());
        let reference = nums(parts.next().unwrap());
        let x: [[f64; 3]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| input[3 * i + j]));
        let uv: [[f64; 2]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| input[12 + 2 * i + j]));
        let (mut out, mut status) = ([P4pfPose::default(); MAX_POSES], -1);
        let count = solve_p4pf_kernel(&x, &uv, &mut out, &mut status);
        total += 1;
        if !same(status, &out[..count], &reference) {
            bad.push(tag.to_string());
        }
    }
    assert!(total > 140, "fixture has {total} lines");
    assert!(
        bad.is_empty(),
        "{} / {total} mismatches, first: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}

fn nullspace<const M: usize, const K: usize, const L: usize>(
    input: &[f64],
    fields: &[&str],
) -> bool {
    if input.len() != 4 + M * K || fields.len() != 2 {
        return false;
    }
    let pivot = input[2] != 0.0;
    let tolerance = (!input[3].is_nan()).then_some(input[3]);
    let w: [[f64; K]; M] = std::array::from_fn(|i| std::array::from_fn(|j| input[4 + K * i + j]));
    let (n, ratio) = house_nullspace::<M, K, L>(&w, pivot, tolerance);
    let expected_ratio = nums(fields[0]);
    let flat: Vec<f64> = n.iter().flatten().copied().collect();
    let expected = nums(fields[1]);
    expected_ratio.len() == 1
        && same_bits(ratio, expected_ratio[0])
        && flat.len() == expected.len()
        && flat.iter().zip(&expected).all(|(a, b)| same_bits(*a, *b))
}

#[test]
fn the_null_space_branches_are_bit_exact_against_the_reference() {
    let (mut total, mut bad) = (0, Vec::new());
    for line in BRANCHES.lines().filter(|line| !line.starts_with('#')) {
        let parts: Vec<&str> = line.split('|').map(str::trim).collect();
        let input = nums(parts[1]);
        let fields = &parts[2..];
        let ok = match (input[0] as usize, input[1] as usize) {
            (4, 4) => nullspace::<4, 4, 0>(&input, fields),
            (12, 8) => nullspace::<12, 8, 4>(&input, fields),
            (9, 8) => nullspace::<9, 8, 1>(&input, fields),
            _ => false,
        };
        total += 1;
        if !ok {
            bad.push(parts[0].to_string());
        }
    }
    assert!(total >= 7, "branch fixture has {total} cases");
    assert!(
        bad.is_empty(),
        "{} / {total} mismatches: {bad:?}",
        bad.len()
    );
}
