// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Isometry3, Point3, UnitQuaternion, UnitVector3, Vector3};
use openmvg_geometry::pose::p3p::{P3pMethod, P3pSolutions, solve};
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn relative_pose_error(
    candidates: &P3pSolutions,
    truth: &Isometry3<f64>,
    world: &[Point3<f64>; 3],
) -> f64 {
    let scene_scale = (world[0] - world[1])
        .norm()
        .max((world[0] - world[2]).norm())
        .max((world[1] - world[2]).norm());
    let truth_center = truth.inverse_transform_point(&Point3::origin());
    candidates
        .iter()
        .map(|candidate| {
            let rotation_error = UnitQuaternion::from_rotation_matrix(candidate.rotation())
                .angle_to(&truth.rotation);
            let center_error = (candidate.center() - truth_center).norm() / scene_scale;
            rotation_error + center_error
        })
        .fold(f64::INFINITY, f64::min)
}

fn assert_recovers_truth(
    world: [Point3<f64>; 3],
    bearings: [UnitVector3<f64>; 3],
    truth: &Isometry3<f64>,
) {
    for (name, candidates) in solve_all(world, bearings) {
        assert_valid_candidates(name, &candidates, &world, &bearings);
        let error = relative_pose_error(&candidates, truth, &world);
        assert!(
            error < 1e-6,
            "{name}: relative pose error {error:e}, {} candidates",
            candidates.len()
        );
    }
}

fn assert_valid_candidates(
    name: &str,
    candidates: &P3pSolutions,
    world: &[Point3<f64>; 3],
    bearings: &[UnitVector3<f64>; 3],
) {
    for candidate in candidates {
        assert!(
            (UnitQuaternion::from_rotation_matrix(candidate.rotation())
                .quaternion()
                .norm_squared()
                - 1.0)
                .abs()
                <= 1e-10,
            "{name}: rotation quaternion is not unit length"
        );
        assert!(
            candidate
                .translation()
                .iter()
                .all(|value| value.is_finite()),
            "{name}: non-finite translation"
        );
        for (point, bearing) in world.iter().zip(bearings) {
            let camera = candidate.project_to_camera(point).coords;
            assert!(
                camera.iter().all(|value| value.is_finite()) && camera.dot(bearing) > 0.0,
                "{name}: returned a non-cheiral candidate"
            );
            let angular_residual = camera.cross(bearing).norm().atan2(camera.dot(bearing));
            assert!(
                angular_residual <= 2e-6,
                "{name}: candidate angular residual {angular_residual:e}"
            );
        }
    }
}

fn solve_all(
    world: [Point3<f64>; 3],
    bearings: [UnitVector3<f64>; 3],
) -> [(&'static str, P3pSolutions); 7] {
    [
        (
            "LambdaTwist",
            solve(world, bearings, P3pMethod::LambdaTwist),
        ),
        ("Ding", solve(world, bearings, P3pMethod::Ding)),
        ("KeOrtho", solve(world, bearings, P3pMethod::KeOrtho)),
        ("Kneip", solve(world, bearings, P3pMethod::Kneip)),
        ("Fused", solve(world, bearings, P3pMethod::Fused)),
        ("Wu", solve(world, bearings, P3pMethod::Wu)),
        ("Nakano", solve(world, bearings, P3pMethod::Nakano)),
    ]
}

#[test]
fn fused_p3p_is_the_default_method() {
    assert_eq!(P3pMethod::default(), P3pMethod::Fused);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// Shared P3P solver family: all seven methods share one array-typed solve function and
    /// fixed-capacity result; every method recovers randomized well-conditioned world-to-camera
    /// poses; every candidate has a unit rotation, finite translation, positive cheirality, and
    /// bounded angular residual.
    #[test]
    fn all_p3p_solvers_recover_well_conditioned_scenes(
        rotation_vector in prop::array::uniform3(-1.0_f64..1.0),
        translation in prop::array::uniform3(-5.0_f64..5.0),
        scale_log2 in -30_i32..31,
        perturbation in prop::array::uniform3(-0.25_f64..0.25),
        permutation_index in 0_usize..6,
    ) {
        let rotation = UnitQuaternion::from_scaled_axis(Vector3::from(rotation_vector));
        let scale = 2.0_f64.powi(scale_log2);
        let pose = Isometry3::from_parts((Vector3::from(translation) * scale).into(), rotation);
        let camera_points = [
            Point3::new((-1.2 + perturbation[0]) * scale, -0.5 * scale, 4.0 * scale),
            Point3::new(1.1 * scale, (-0.8 + perturbation[1]) * scale, 5.5 * scale),
            Point3::new(0.2 * scale, (1.3 + perturbation[2]) * scale, 6.5 * scale),
        ];
        let world = camera_points.map(|point| pose.inverse_transform_point(&point));
        let bearings = camera_points.map(|point| UnitVector3::new_normalize(point.coords));
        let permutation = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0],
        ][permutation_index];
        let world = permutation.map(|index| world[index]);
        let bearings = permutation.map(|index| bearings[index]);
        assert_recovers_truth(world, bearings, &pose);
    }
}

#[test]
fn all_solvers_are_scale_invariant_at_extremes() {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let scenes = [
        [
            Point3::new(-1.290_876_283_094_080_6, -0.5, 4.0),
            Point3::new(1.1, -1.045_265_169_156_624_6, 5.5),
            Point3::new(0.2, 1.3, 6.728_130_915_886_78),
        ],
        [
            Point3::new(-1.317_456_026_700_417_4, -0.5, 4.0),
            Point3::new(1.1, -0.627_585_641_893_859_9, 5.5),
            Point3::new(0.2, 1.3, 6.670_492_730_828_727),
        ],
    ];
    let truth = Isometry3::identity();
    for scene in scenes {
        for scale_log2 in [-30, -28, -9, 0, 30] {
            let scale = 2.0_f64.powi(scale_log2);
            let camera = scene.map(|point| point * scale);
            let bearings = camera.map(|point| UnitVector3::new_normalize(point.coords));
            for permutation in permutations {
                assert_recovers_truth(
                    permutation.map(|index| camera[index]),
                    permutation.map(|index| bearings[index]),
                    &truth,
                );
            }
        }
    }
}

#[test]
fn cubic_polish_recovers_nakano_near_biquadratic_case() {
    let scale = 0.5;
    let truth = Isometry3::from_parts(
        (Vector3::new(0.0, -2.212_220_062_337_294_3, 4.030_728_176_765_445) * scale).into(),
        UnitQuaternion::identity(),
    );
    let camera = [
        Point3::new(-1.333_921_544_049_885_5, -0.5, 4.0) * scale,
        Point3::new(1.1, -0.722_830_053_520_761_9, 5.5) * scale,
        Point3::new(0.2, 1.3, 6.567_597_775_961_706) * scale,
    ];
    let world = camera.map(|point| truth.inverse_transform_point(&point));
    let bearings = camera.map(|point| UnitVector3::new_normalize(point.coords));
    let permutation = [0, 2, 1];
    assert_recovers_truth(
        permutation.map(|index| world[index]),
        permutation.map(|index| bearings[index]),
        &truth,
    );
}

#[test]
fn unanchored_world_coordinates_preserve_nakano_candidates() {
    let world = [
        Point3::new(-1.316_549_056_143_172_8, -0.5, 4.0),
        Point3::new(1.1, -0.587_085_479_019_721, 5.5),
        Point3::new(0.2, 1.3, 6.729_009_583_882_854),
    ];
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
    let candidates = solve(world, bearings, P3pMethod::Nakano);
    assert_valid_candidates("Nakano", &candidates, &world, &bearings);
    let error = relative_pose_error(&candidates, &Isometry3::identity(), &world);
    assert!(error < 1e-6, "relative pose error {error:e}");
}

#[test]
fn all_solvers_reject_collinear_world_points() {
    let world = [
        Point3::new(0.0, 0.0, 4.0),
        Point3::new(1.0, 0.0, 4.0),
        Point3::new(2.0, 0.0, 4.0),
    ];
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
    for (name, candidates) in solve_all(world, bearings) {
        assert!(
            candidates.is_empty(),
            "{name}: accepted collinear world points"
        );
    }
}

#[test]
fn all_solvers_handle_translated_world_coordinates() {
    let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.4, 0.3));
    let local_translation = Vector3::new(0.3, -0.2, 0.5);
    for magnitude in [1.0, 1.0e3, 1.0e6, 1.0e7] {
        let offset = Vector3::new(magnitude, -2.0 * magnitude, 3.0 * magnitude);
        let translation = local_translation - rotation * offset;
        let truth = Isometry3::from_parts(translation.into(), rotation);
        let world = [
            Point3::from(offset + Vector3::new(-1.2, -0.5, 4.0)),
            Point3::from(offset + Vector3::new(1.1, -0.8, 5.5)),
            Point3::from(offset + Vector3::new(0.2, 1.3, 6.5)),
        ];
        let bearings =
            world.map(|point| UnitVector3::new_normalize(truth.transform_point(&point).coords));
        assert_recovers_truth(world, bearings, &truth);
    }
}

#[test]
fn wu_handles_its_isosceles_singular_input_ordering() {
    let truth = Isometry3::identity();
    let world = [
        Point3::new(-1.0, 0.0, 4.0),
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(0.0, 1.0, 6.0),
    ];
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
    let candidates = solve(world, bearings, P3pMethod::Wu);
    let error = relative_pose_error(&candidates, &truth, &world);
    assert!(error < 1e-6, "relative pose error {error:e}");
}

#[test]
fn lambda_twist_retries_a_valid_lower_degree_cubic_chart() {
    let world = [
        Point3::new(-1.0, 0.0, 4.0),
        Point3::new(1.0, 0.0, 4.0),
        Point3::new(0.0, 1.0, 4.0),
    ];
    assert_solver_recovers_all_permutations("LambdaTwist", world, |world, bearings| {
        solve(world, bearings, P3pMethod::LambdaTwist)
    });
}

#[test]
fn kneip_retries_a_valid_zero_bearing_y_chart() {
    let world = [
        Point3::new(0.0, 0.0, 4.0),
        Point3::new(4.0, 0.0, 4.0),
        Point3::new(0.0, -3.0, 6.0),
    ];
    assert_solver_recovers_all_permutations("Kneip", world, |world, bearings| {
        solve(world, bearings, P3pMethod::Kneip)
    });
}

#[test]
fn wu_falls_back_on_its_equilateral_chart_singularity() {
    let height = 3.0_f64.sqrt() * 0.5;
    let world = [
        Point3::new(-0.5, 0.0, 4.0),
        Point3::new(0.5, 0.0, 4.0),
        Point3::new(0.0, height, 4.0),
    ];
    assert_solver_recovers_all_permutations("Wu", world, |world, bearings| {
        solve(world, bearings, P3pMethod::Wu)
    });
}

#[test]
fn fused_guard_returns_distinct_candidates_on_danger_cylinder() {
    let world = [
        Point3::new(0.0, 0.0, 4.0),
        Point3::new(1.0, 2.0, 4.0),
        Point3::new(2.0, 0.0, 4.0),
    ];
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let world = permutation.map(|index| world[index]);
        let bearings = permutation.map(|index| bearings[index]);
        let candidates = solve(world, bearings, P3pMethod::Fused);
        assert_valid_candidates("FusedP3P", &candidates, &world, &bearings);
        assert!(relative_pose_error(&candidates, &Isometry3::identity(), &world) < 1e-6);
        for left in 0..candidates.len() {
            for right in left + 1..candidates.len() {
                let rotation_distance = candidates[left]
                    .rotation()
                    .angle_to(candidates[right].rotation());
                let center_distance =
                    (candidates[left].center() - candidates[right].center()).norm();
                assert!(rotation_distance >= 1e-7 || center_distance > 1e-6);
            }
        }
    }
}

#[test]
fn fused_p3p_pose_handles_equilateral_permutations() {
    let height = 3.0_f64.sqrt() * 0.5;
    let world = [
        Point3::new(-0.5, 0.0, 4.0),
        Point3::new(0.5, 0.0, 4.0),
        Point3::new(0.0, height, 4.0),
    ];
    assert_solver_recovers_all_permutations("FusedP3P", world, |world, bearings| {
        solve(world, bearings, P3pMethod::Fused)
    });
}

fn assert_solver_recovers_all_permutations(
    name: &str,
    world: [Point3<f64>; 3],
    mut solve: impl FnMut([Point3<f64>; 3], [UnitVector3<f64>; 3]) -> P3pSolutions,
) {
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let permuted_world = permutation.map(|index| world[index]);
        let permuted_bearings = permutation.map(|index| bearings[index]);
        let candidates = solve(permuted_world, permuted_bearings);
        assert_valid_candidates(name, &candidates, &permuted_world, &permuted_bearings);
        let error = relative_pose_error(&candidates, &Isometry3::identity(), &permuted_world);
        assert!(
            error < 1e-6,
            "{name} permutation {permutation:?}: relative pose error {error:e}"
        );
    }
}

#[test]
fn all_solvers_remain_bounded_with_noisy_bearings() {
    let mut rng = StdRng::seed_from_u64(0x5033_0015e);
    let mut errors: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(512));
    let mut names = [""; 7];
    for _ in 0..512 {
        let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(
            rng.gen_range(-0.8..0.8),
            rng.gen_range(-0.8..0.8),
            rng.gen_range(-0.8..0.8),
        ));
        let truth = Isometry3::from_parts(
            Vector3::new(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
            )
            .into(),
            rotation,
        );
        let camera = [
            Point3::new(-1.2, -0.5, rng.gen_range(3.5..5.0)),
            Point3::new(1.1, -0.8, rng.gen_range(4.5..6.5)),
            Point3::new(0.2, 1.3, rng.gen_range(5.5..8.0)),
        ];
        let world = camera.map(|point| truth.inverse_transform_point(&point));
        let bearings = camera.map(|point| {
            UnitVector3::new_normalize(
                point.coords
                    + Vector3::new(
                        rng.gen_range(-1e-4..1e-4),
                        rng.gen_range(-1e-4..1e-4),
                        rng.gen_range(-1e-4..1e-4),
                    ),
            )
        });
        for (index, (name, candidates)) in solve_all(world, bearings).into_iter().enumerate() {
            assert_valid_candidates(name, &candidates, &world, &bearings);
            names[index] = name;
            errors[index].push(relative_pose_error(&candidates, &truth, &world));
        }
    }
    for (name, errors) in names.into_iter().zip(&mut errors) {
        errors.sort_by(f64::total_cmp);
        let median = errors[errors.len() / 2];
        let p95 = errors[errors.len() * 95 / 100];
        let p99 = errors[errors.len() * 99 / 100];
        eprintln!("{name}: median={median:e}, p95={p95:e}, p99={p99:e}");
        assert!(median < 2e-3, "{name}: noisy median pose error {median:e}");
        assert!(p95 < 0.05, "{name}: noisy p95 pose error {p95:e}");
        assert!(p99.is_finite(), "{name}: failed to return candidates");
    }
}

#[test]
fn all_solvers_preserve_output_invariants_near_degeneracy() {
    let mut rng = StdRng::seed_from_u64(0x5033_de6e);
    for _ in 0..512 {
        let depth = rng.gen_range(4.0..12.0);
        let spread = 10.0_f64.powf(rng.gen_range(-7.0..-1.0));
        let camera = [
            Point3::new(-spread * depth, 0.0, depth),
            Point3::new(spread * depth, spread * spread * depth, depth * 1.1),
            Point3::new(2.0 * spread * depth, -spread * spread * depth, depth * 1.2),
        ];
        let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(
            rng.gen_range(-0.8..0.8),
            rng.gen_range(-0.8..0.8),
            rng.gen_range(-0.8..0.8),
        ));
        let truth = Isometry3::from_parts(
            Vector3::new(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
            )
            .into(),
            rotation,
        );
        let world = camera.map(|point| truth.inverse_transform_point(&point));
        let bearings = camera.map(|point| UnitVector3::new_normalize(point.coords));
        for (name, candidates) in solve_all(world, bearings) {
            assert_valid_candidates(name, &candidates, &world, &bearings);
        }
    }
}
