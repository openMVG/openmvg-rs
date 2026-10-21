// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The five-point kernel against outputs exported from the project's Julia
//! reference implementation: bit-exact on cases that exercise every code path,
//! plus its output, panic-freedom and thread-independence contracts.

use nalgebra::{Matrix3, Vector3};
use openmvg_geometry::essential::test_support::{
    FivePointPose, FivePointPoses, five_point_path, solve_five_point_kernel,
};

/// One case per line: `tag index | b1 (5x3) b2 (5x3) | count, then e (9),
/// r (9) and t (3) of each pose`, as shortest round-trip decimals.
const FIXTURE: &str = include_str!("data/five_point_reference.txt");

type Bearings = [[f64; 3]; 5];

struct Case {
    tag: &'static str,
    bearings1: Bearings,
    bearings2: Bearings,
    reference: Vec<f64>,
}

fn values(field: &str) -> Vec<f64> {
    field
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect()
}

fn cases() -> Vec<Case> {
    FIXTURE
        .lines()
        .map(|line| {
            let [tag, input, reference]: [&'static str; 3] =
                line.split(" | ").collect::<Vec<_>>().try_into().unwrap();
            let input = values(input);
            assert_eq!(input.len(), 30, "{tag}");
            let bearings = |offset: usize| -> Bearings {
                std::array::from_fn(|i| std::array::from_fn(|j| input[offset + 3 * i + j]))
            };
            Case {
                tag,
                bearings1: bearings(0),
                bearings2: bearings(15),
                reference: values(reference),
            }
        })
        .collect()
}

fn matches_reference(poses: &[FivePointPose], reference: &[f64]) -> bool {
    reference.len() == 1 + 21 * poses.len()
        && reference[0] == poses.len() as f64
        && poses
            .iter()
            .zip(reference[1..].as_chunks::<21>().0)
            .all(|(pose, expected)| {
                pose.e
                    .iter()
                    .chain(&pose.r)
                    .chain(&pose.t)
                    .zip(expected)
                    .all(|(value, expected)| value.to_bits() == expected.to_bits())
            })
}

#[test]
fn matches_the_reference_bit_for_bit_on_every_code_path() {
    let cases = cases();
    assert_eq!(cases.len(), 39);
    let mut poses = FivePointPoses::new();
    let mut paths = 0;
    let mut mismatches = Vec::new();
    for case in &cases {
        paths |= solve_five_point_kernel(&case.bearings1, &case.bearings2, &mut poses);
        if !matches_reference(&poses, &case.reference) {
            mismatches.push(case.tag);
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} cases differ: {mismatches:?}",
        mismatches.len(),
        cases.len()
    );

    let every_path = [
        five_point_path::FRAME_RETRY,
        five_point_path::RESCUE,
        five_point_path::SMALL_EXTRA,
        five_point_path::SMALL_DD,
        five_point_path::GATE_NEWTON,
        five_point_path::NONFINITE_RETRY,
        five_point_path::DD_UNION,
        five_point_path::DD_ADDED,
        five_point_path::EXTRA_NEWTON,
        five_point_path::RESIGN_FLIP,
        five_point_path::DROPPED,
        five_point_path::RESCALE,
    ]
    .into_iter()
    .fold(0, |all, path| all | path);
    assert_eq!(
        paths, every_path,
        "the fixture must exercise every kernel path"
    );
}

fn assert_output_contract(pose: &FivePointPose, label: &str) {
    let rotation = Matrix3::from_row_slice(&pose.r);
    let translation = Vector3::from(pose.t);
    let essential = Matrix3::from_row_slice(&pose.e);
    assert!(
        (rotation.transpose() * rotation - Matrix3::identity()).amax() < 1e-9,
        "{label}: R is not orthonormal"
    );
    assert!(
        (rotation.determinant() - 1.0).abs() < 1e-9,
        "{label}: det R != 1"
    );
    assert!(
        (translation.norm() - 1.0).abs() < 1e-12,
        "{label}: |t| != 1"
    );
    let expected = translation.cross_matrix() * rotation * 0.5f64.sqrt();
    assert!(
        (essential - expected).amax() < 1e-12,
        "{label}: E != [t]x R / sqrt(2)"
    );
    assert!(
        (essential.norm() - 1.0).abs() < 1e-12,
        "{label}: |E|_F != 1"
    );
}

#[test]
fn every_pose_is_a_rotation_a_unit_translation_and_their_essential_matrix() {
    let mut poses = FivePointPoses::new();
    for case in cases() {
        solve_five_point_kernel(&case.bearings1, &case.bearings2, &mut poses);
        for pose in &poses {
            assert_output_contract(pose, case.tag);
        }
    }
}

/// Baseline 1/100 of the scene depth (parallax below 0.2 degrees): the elimination turns the true
/// root into a complex pair and the two remaining roots pass cheirality 3.3 degrees off, so only
/// the data-side rotation trigger reaches the small-parallax candidates.
#[test]
fn near_pure_rotation_recovers_the_pose_without_root_evidence() {
    let bearings1 = [
        [0.4063592354175939, 0.25699892130270785, 0.8768259383937459],
        [-0.3349227565823937, 0.12154205507583095, 0.9343737346325584],
        [
            0.41962615364640676,
            -0.17451653720311075,
            0.8907625213369486,
        ],
        [0.37738795115494517, -0.1748359178198572, 0.9094013064446086],
        [
            -0.20729030333716902,
            -0.35981539770261695,
            0.9097052323244559,
        ],
    ];
    let bearings2 = [
        [0.26834968157620326, 0.1340149831617983, 0.9539541040773887],
        [
            -0.43129302656328056,
            -0.14591247033579968,
            0.8903346989971748,
        ],
        [0.3696980361200079, -0.27967477171217153, 0.886061727057858],
        [
            0.32746498297839133,
            -0.29279241904255976,
            0.8983536521182335,
        ],
        [
            -0.20626154702675237,
            -0.5779995909102057,
            0.7895395158735032,
        ],
    ];
    let rotation = Matrix3::new(
        0.9745128428719297,
        -0.2080646482819838,
        -0.08386787950678146,
        0.18609765610611811,
        0.9585654516729576,
        -0.21568480996775777,
        0.1252692359259792,
        0.19458000152716917,
        0.9728546867524503,
    );
    let mut poses = FivePointPoses::new();
    let paths = solve_five_point_kernel(&bearings1, &bearings2, &mut poses);
    assert_ne!(paths & five_point_path::SMALL_DD, 0);
    let error = poses
        .iter()
        .map(|pose| (Matrix3::from_row_slice(&pose.r) - rotation).amax())
        .fold(f64::INFINITY, f64::min);
    assert!(error < 1e-9, "best rotation is {error:e} away");
    for pose in &poses {
        assert_output_contract(pose, "near pure rotation");
    }
}

#[test]
fn results_do_not_depend_on_the_calling_thread() {
    let cases = cases();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let cases = &cases;
                scope.spawn(move || {
                    let mut poses = FivePointPoses::new();
                    cases
                        .iter()
                        .cycle()
                        .skip(7 * worker)
                        .take(2 * cases.len())
                        .all(|case| {
                            solve_five_point_kernel(&case.bearings1, &case.bearings2, &mut poses);
                            matches_reference(&poses, &case.reference)
                        })
                })
            })
            .collect();
        for worker in workers {
            assert!(worker.join().unwrap());
        }
    });
}

#[test]
fn degenerate_inputs_never_panic_and_finite_ones_keep_the_output_contract() {
    let cases = cases();
    let generic = cases.iter().find(|case| case.tag == "pinhole 0").unwrap();
    let (b1, b2) = (generic.bearings1, generic.bearings2);
    let mut duplicated = (b1, b2);
    duplicated.0[1] = duplicated.0[0];
    duplicated.1[1] = duplicated.1[0];
    let collinear = std::array::from_fn(|i| [1.0 + i as f64, 0.0, 1.0]);
    let mut inputs: Vec<(Bearings, Bearings)> = vec![
        duplicated,
        ([b1[0]; 5], [b2[0]; 5]),
        (b1, b1),
        (collinear, b2),
        ([[0.0; 3]; 5], [[0.0; 3]; 5]),
        ([[f64::NAN; 3]; 5], b2),
        (b1, [[f64::INFINITY; 3]; 5]),
        ([[f64::MAX; 3]; 5], [[f64::MIN_POSITIVE; 3]; 5]),
        ([[1e-300; 3]; 5], b2),
    ];

    // Deterministic xorshift garbage with a large dynamic range and repeated points.
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for sample in 0..2000 {
        let scale = if sample % 3 == 0 { 1e6 } else { 1.0 };
        let mut bearings1 = [[0.0; 3]; 5];
        let mut bearings2 = [[0.0; 3]; 5];
        for (point1, point2) in bearings1.iter_mut().zip(&mut bearings2) {
            for (value1, value2) in point1.iter_mut().zip(point2) {
                *value1 = uniform() * scale;
                *value2 = uniform();
            }
        }
        if sample % 7 == 0 {
            bearings2[3] = bearings2[2];
        }
        inputs.push((bearings1, bearings2));
    }

    let mut poses = FivePointPoses::new();
    for (index, (bearings1, bearings2)) in inputs.iter().enumerate() {
        solve_five_point_kernel(bearings1, bearings2, &mut poses);
        let finite = bearings1
            .iter()
            .chain(bearings2)
            .flatten()
            .all(|v| v.is_finite());
        if finite {
            for pose in &poses {
                assert_output_contract(pose, &format!("input {index}"));
            }
        }
    }
}
