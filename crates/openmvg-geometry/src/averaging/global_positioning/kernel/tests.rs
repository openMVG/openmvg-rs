// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BATA kernel regression tests.

use nalgebra::Vector3;

use super::{
    BataOptions, GlobalPositioningError, GlobalPositioningPlan, GlobalPositioningProblem,
    PositioningObservation,
};

fn normalized(value: Vector3<f64>) -> Vector3<f64> {
    value / value.norm()
}

fn exact_problem() -> (
    Vec<PositioningObservation>,
    Vec<Vector3<f64>>,
    Vec<Vector3<f64>>,
) {
    let cameras = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.1, 0.0),
        Vector3::new(0.2, 1.0, 0.1),
    ];
    let points = vec![
        Vector3::new(0.2, 0.1, 3.0),
        Vector3::new(1.1, 0.8, 4.0),
        Vector3::new(-0.4, 1.2, 3.5),
        Vector3::new(0.7, -0.4, 2.8),
    ];
    let mut bearings = Vec::new();
    for (point_index, point) in points.iter().enumerate() {
        for (camera_index, camera) in cameras.iter().enumerate() {
            bearings.push(PositioningObservation::bearing(
                camera_index,
                point_index,
                normalized(*point - *camera),
                1.0,
            ));
        }
    }
    (bearings, cameras, points)
}

#[test]
fn exact_bearing_scene_solves_deterministically() {
    let (bearings, _, _) = exact_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    let mut first_cameras = vec![Vector3::zeros(); 3];
    let mut first_points = vec![Vector3::zeros(); 4];
    let retained_before = workspace.retained_capacity_bytes();
    let first = plan
        .solve_into(&mut first_cameras, &mut first_points, &mut workspace)
        .unwrap();
    let mut second_cameras = vec![Vector3::zeros(); 3];
    let mut second_points = vec![Vector3::zeros(); 4];
    let second = plan
        .solve_into(&mut second_cameras, &mut second_points, &mut workspace)
        .unwrap();
    assert_eq!(first_cameras, second_cameras);
    assert_eq!(first_points, second_points);
    assert_eq!(first, second);
    assert_eq!(retained_before, workspace.retained_capacity_bytes());
    assert!(first.rms_angular_error < 0.02);
}

#[test]
fn workspace_reuses_safely_across_equal_shape_plans() {
    let (bearings, _, _) = exact_problem();
    let first_plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    let mut reordered = bearings.clone();
    reordered.reverse();
    let second_plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &reordered,
        },
        BataOptions::default(),
    )
    .unwrap();

    let mut reused_workspace = first_plan.try_new_workspace().unwrap();
    let mut discarded_cameras = vec![Vector3::zeros(); 3];
    let mut discarded_points = vec![Vector3::zeros(); 4];
    first_plan
        .solve_into(
            &mut discarded_cameras,
            &mut discarded_points,
            &mut reused_workspace,
        )
        .unwrap();
    let mut reused_cameras = vec![Vector3::zeros(); 3];
    let mut reused_points = vec![Vector3::zeros(); 4];
    let reused_report = second_plan
        .solve_into(
            &mut reused_cameras,
            &mut reused_points,
            &mut reused_workspace,
        )
        .unwrap();

    let mut fresh_workspace = second_plan.try_new_workspace().unwrap();
    let mut fresh_cameras = vec![Vector3::zeros(); 3];
    let mut fresh_points = vec![Vector3::zeros(); 4];
    let fresh_report = second_plan
        .solve_into(&mut fresh_cameras, &mut fresh_points, &mut fresh_workspace)
        .unwrap();

    assert_eq!(reused_report, fresh_report);
    assert_eq!(reused_cameras, fresh_cameras);
    assert_eq!(reused_points, fresh_points);
}

#[test]
fn large_scene_uses_matrix_free_pcg() {
    let camera_count = 257;
    let cameras: Vec<_> = (0..camera_count)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / camera_count as f64;
            Vector3::new(angle.cos(), angle.sin(), 0.1 * (3.0 * angle).sin())
        })
        .collect();
    let points = [
        Vector3::new(0.0, 0.0, 3.0),
        Vector3::new(0.8, 0.3, 3.5),
        Vector3::new(-0.6, 0.7, 4.0),
        Vector3::new(0.2, -0.9, 4.5),
        Vector3::new(1.1, -0.4, 5.0),
        Vector3::new(-0.9, -0.5, 5.5),
    ];
    let mut bearings = Vec::with_capacity(camera_count * points.len());
    for (point_index, point) in points.iter().enumerate() {
        for (camera_index, camera) in cameras.iter().enumerate() {
            bearings.push(PositioningObservation::bearing(
                camera_index,
                point_index,
                normalized(*point - *camera),
                1.0,
            ));
        }
    }
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count,
            point_count: points.len(),
            observations: &bearings,
        },
        BataOptions {
            max_outer_iterations: 8,
            ..BataOptions::default()
        },
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    // This test pins the conjugate-gradient path, which the sparse factorization would replace.
    #[cfg(feature = "faer")]
    {
        workspace.alternation_direct_unavailable = true;
    }
    let mut estimated_cameras = vec![Vector3::zeros(); camera_count];
    let mut estimated_points = vec![Vector3::zeros(); points.len()];
    let report = plan
        .solve_into(
            &mut estimated_cameras,
            &mut estimated_points,
            &mut workspace,
        )
        .unwrap();
    assert_eq!(
        report.reduced_backend,
        super::api::ReducedSystemBackend::MatrixFreePcg
    );
    assert!(report.pcg_iteration_count > 0);
    assert!(
        estimated_cameras
            .iter()
            .all(|camera| camera.iter().all(|value| value.is_finite()))
    );
}

#[test]
fn large_sparse_coupling_and_matrix_free_solves_agree() {
    let camera_count = 257;
    let cameras: Vec<_> = (0..camera_count)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / camera_count as f64;
            Vector3::new(angle.cos(), angle.sin(), 0.05 * (2.0 * angle).sin())
        })
        .collect();
    let points: Vec<_> = (0..camera_count)
        .map(|index| {
            let angle = std::f64::consts::TAU * (index as f64 + 0.35) / camera_count as f64;
            Vector3::new(
                0.35 * angle.cos(),
                0.35 * angle.sin(),
                3.0 + 0.001 * index as f64,
            )
        })
        .collect();
    let mut bearings = Vec::with_capacity(camera_count * 8);
    for (point, position) in points.iter().enumerate() {
        for offset in 0..8 {
            let camera = (point + offset) % camera_count;
            bearings.push(PositioningObservation::bearing(
                camera,
                point,
                normalized(*position - cameras[camera]),
                1.0,
            ));
        }
    }
    let problem = GlobalPositioningProblem {
        camera_count,
        point_count: points.len(),
        observations: &bearings,
    };
    // The polish is backend independent; this test pins the equivalence of the alternation's
    // reduced-system backends, which a nonlinear polish step would only amplify.
    let options = BataOptions {
        max_outer_iterations: 8,
        polish_steps: 0,
        ..BataOptions::default()
    };
    let sparse_plan = GlobalPositioningPlan::try_new(problem, options).unwrap();
    assert!(sparse_plan.uses_materialized_couplings());
    let mut matrix_free_plan = GlobalPositioningPlan::try_new(problem, options).unwrap();
    matrix_free_plan.materialized_couplings = false;
    let mut sparse_workspace = sparse_plan.try_new_workspace().unwrap();
    let mut matrix_free_workspace = matrix_free_plan.try_new_workspace().unwrap();
    // Both workspaces pin their conjugate-gradient path, which the sparse factorization would
    // replace.
    #[cfg(feature = "faer")]
    {
        sparse_workspace.alternation_direct_unavailable = true;
        matrix_free_workspace.alternation_direct_unavailable = true;
    }
    let mut sparse_cameras = vec![Vector3::zeros(); camera_count];
    let mut sparse_points = vec![Vector3::zeros(); points.len()];
    let mut matrix_free_cameras = vec![Vector3::zeros(); camera_count];
    let mut matrix_free_points = vec![Vector3::zeros(); points.len()];
    let sparse_report = sparse_plan
        .solve_into(
            &mut sparse_cameras,
            &mut sparse_points,
            &mut sparse_workspace,
        )
        .unwrap();
    let matrix_free_report = matrix_free_plan
        .solve_into(
            &mut matrix_free_cameras,
            &mut matrix_free_points,
            &mut matrix_free_workspace,
        )
        .unwrap();
    assert_eq!(
        sparse_report.reduced_backend,
        super::api::ReducedSystemBackend::SparseCouplingPcg
    );
    assert_eq!(
        matrix_free_report.reduced_backend,
        super::api::ReducedSystemBackend::MatrixFreePcg
    );
    assert_eq!(
        sparse_report.iteration_count,
        matrix_free_report.iteration_count
    );
    assert!((sparse_report.rms_angular_error - matrix_free_report.rms_angular_error).abs() < 1e-9);
    let maximum_camera_difference = sparse_cameras
        .iter()
        .zip(&matrix_free_cameras)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max);
    let maximum_point_difference = sparse_points
        .iter()
        .zip(&matrix_free_points)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max);
    assert!(
        maximum_camera_difference < 1e-7,
        "{maximum_camera_difference:e}"
    );
    assert!(
        maximum_point_difference < 1e-7,
        "{maximum_point_difference:e}"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn large_parallel_solve_is_bit_identical_across_thread_counts() {
    let camera_count = 300usize;
    let points_per_camera = 110usize;
    let cameras = (0..camera_count)
        .map(|camera| {
            let angle = std::f64::consts::TAU * camera as f64 / camera_count as f64;
            Vector3::new(20.0 * angle.cos(), 20.0 * angle.sin(), 0.01 * camera as f64)
        })
        .collect::<Vec<_>>();
    let mut bearings = Vec::with_capacity(camera_count * points_per_camera * 8);
    for base in 0..camera_count {
        for sample in 0..points_per_camera {
            let phase = base as f64 * 0.37 + sample as f64 * 1.91;
            let point = Vector3::new(
                4.0 * phase.sin(),
                3.0 * (0.7 * phase).cos(),
                2.0 * (1.3 * phase).sin(),
            );
            let point_index = points_per_camera * base + sample;
            for offset in 0..8 {
                let camera = (base + offset) % camera_count;
                bearings.push(PositioningObservation::bearing(
                    camera,
                    point_index,
                    normalized(point - cameras[camera]),
                    1.0,
                ));
            }
        }
    }
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count,
            point_count: camera_count * points_per_camera,
            observations: &bearings,
        },
        // The polish is serial; this test pins the deterministic parallel passes of the alternation.
        BataOptions {
            max_outer_iterations: 8,
            polish_steps: 0,
            ..BataOptions::default()
        },
    )
    .unwrap();
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut serial_workspace = plan.try_new_workspace().unwrap();
    let mut parallel_workspace = plan.try_new_workspace().unwrap();
    let mut serial_cameras = vec![Vector3::zeros(); camera_count];
    let mut parallel_cameras = vec![Vector3::zeros(); camera_count];
    let mut serial_points = vec![Vector3::zeros(); camera_count * points_per_camera];
    let mut parallel_points = vec![Vector3::zeros(); camera_count * points_per_camera];
    let serial_report = serial_pool
        .install(|| {
            plan.solve_into(
                &mut serial_cameras,
                &mut serial_points,
                &mut serial_workspace,
            )
        })
        .unwrap();
    let parallel_report = parallel_pool
        .install(|| {
            plan.solve_into(
                &mut parallel_cameras,
                &mut parallel_points,
                &mut parallel_workspace,
            )
        })
        .unwrap();

    assert!(!serial_report.parallel_execution);
    assert!(parallel_report.parallel_execution);
    assert_eq!(
        serial_report.iteration_count,
        parallel_report.iteration_count
    );
    assert_eq!(serial_report.anchor_camera, parallel_report.anchor_camera);
    assert_eq!(
        serial_report.reduced_backend,
        parallel_report.reduced_backend
    );
    assert_eq!(
        serial_report.pcg_iteration_count,
        parallel_report.pcg_iteration_count
    );
    assert_eq!(
        serial_report.rms_angular_error,
        parallel_report.rms_angular_error
    );
    assert_eq!(serial_cameras, parallel_cameras);
    assert_eq!(serial_points, parallel_points);
}

#[test]
fn direction_only_triangle_does_not_use_point_scratch() {
    let truth = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.2, 1.0, 0.1),
        Vector3::new(-0.5, 0.2, 0.8),
    ];
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let directions: Vec<_> = pairs
        .into_iter()
        .map(|(source, target)| {
            PositioningObservation::relative_direction(
                source,
                target,
                normalized(truth[target] - truth[source]),
                1.0,
            )
        })
        .collect();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: truth.len(),
            point_count: 0,
            observations: &directions,
        },
        BataOptions::default(),
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    let mut cameras = vec![Vector3::zeros(); truth.len()];
    let report = plan
        .solve_into(&mut cameras, &mut [], &mut workspace)
        .unwrap();
    assert!(report.rms_angular_error < 1e-3);
    assert!(
        cameras
            .iter()
            .all(|camera| camera.iter().all(|v| v.is_finite()))
    );
}

#[test]
fn duplicate_bearing_is_rejected() {
    let observations = [
        PositioningObservation::bearing(0, 0, Vector3::z(), 1.0),
        PositioningObservation::bearing(0, 0, Vector3::z(), 1.0),
        PositioningObservation::bearing(1, 0, Vector3::z(), 1.0),
    ];
    let error = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 2,
            point_count: 1,
            observations: &observations,
        },
        BataOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(
        error,
        GlobalPositioningError::DuplicateBearing { observation: 1 }
    );
}

#[test]
fn duplicate_direction_pair_is_rejected() {
    let directions = [
        PositioningObservation::relative_direction(0, 1, Vector3::x(), 1.0),
        PositioningObservation::relative_direction(1, 0, -Vector3::x(), 1.0),
    ];
    let error = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 2,
            point_count: 0,
            observations: &directions,
        },
        BataOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(
        error,
        GlobalPositioningError::DuplicateDirection { direction: 1 }
    );
}

#[test]
fn invalid_options_and_camera_count_are_reported_before_preparation() {
    assert_eq!(
        BataOptions {
            max_outer_iterations: 0,
            ..BataOptions::default()
        }
        .validate(),
        Err(GlobalPositioningError::InvalidOptions)
    );
    assert_eq!(
        GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: 1,
                point_count: 0,
                observations: &[],
            },
            BataOptions::default(),
        )
        .err(),
        Some(GlobalPositioningError::TooFewCameras { provided: 1 })
    );
}

#[test]
fn failed_solve_is_transactional() {
    let (bearings, _, _) = exact_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    workspace.cameras.pop();
    let mut cameras = vec![Vector3::repeat(7.0); 3];
    let mut points = vec![Vector3::repeat(9.0); 4];
    assert_eq!(
        plan.solve_into(&mut cameras, &mut points, &mut workspace),
        Err(GlobalPositioningError::WorkspaceMismatch)
    );
    assert!(cameras.iter().all(|camera| *camera == Vector3::repeat(7.0)));
    assert!(points.iter().all(|point| *point == Vector3::repeat(9.0)));
}

#[test]
fn convergence_waits_for_the_final_robust_stage() {
    let (bearings, _, _) = exact_problem();
    for hard_cut_start_iteration in [3, 15] {
        let options = BataOptions {
            hard_cut_start_iteration,
            ..BataOptions::default()
        };
        let plan = GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: 3,
                point_count: 4,
                observations: &bearings,
            },
            options,
        )
        .unwrap();
        let mut workspace = plan.try_new_workspace().unwrap();
        let mut cameras = vec![Vector3::zeros(); 3];
        let mut points = vec![Vector3::zeros(); 4];
        let report = plan
            .solve_into(&mut cameras, &mut points, &mut workspace)
            .unwrap();

        // An exact scene is stationary almost at once, but the graduated loss has not reached its
        // final width, nor the hard cut started, until the configured iteration has run.
        assert!(
            report.iteration_count > hard_cut_start_iteration,
            "stopped after {} iterations with the final stage at {hard_cut_start_iteration}",
            report.iteration_count
        );
        assert_eq!(report.connectivity_fallback_count, 0);
    }
}

#[test]
fn cuts_that_disconnect_the_active_graph_fall_back_to_huber_weights() {
    // A zero-degree cut with no warmup zeroes every weight of an inexact iterate, which leaves no
    // active edge at all. The ridge would still factor that system and return arbitrary cameras.
    let (bearings, _, _) = exact_problem();
    let options = BataOptions {
        warmup_iterations: 0,
        hard_cut_start_iteration: 0,
        hard_cut_deg: 0.0,
        ..BataOptions::default()
    };
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &bearings,
        },
        options,
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    let mut cameras = vec![Vector3::zeros(); 3];
    let mut points = vec![Vector3::zeros(); 4];
    let report = plan
        .solve_into(&mut cameras, &mut points, &mut workspace)
        .unwrap();

    assert!(report.connectivity_fallback_count > 0);
    assert!(
        report.polish_step_count > 0,
        "polish must retain the last connectivity fallback angular policy"
    );
    assert!(report.rms_angular_error < 0.02, "{report:?}");
}

#[test]
fn active_connectivity_follows_the_positive_weights() {
    use super::solve::active_graph_is_connected;

    let (bearings, _, _) = exact_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();

    workspace.bearing_weight.fill(1.0);
    assert!(active_graph_is_connected(&plan, &mut workspace));

    // Camera 2 keeps no active observation: it is stranded.
    for (index, observation) in plan.bearings.iter().enumerate() {
        if observation.camera == 2 {
            workspace.bearing_weight[index] = 0.0;
        }
    }
    assert!(!active_graph_is_connected(&plan, &mut workspace));
}

#[test]
fn polish_solves_an_exact_scene_to_machine_scale() {
    let (bearings, _, _) = exact_problem();
    let solve = |polish_steps| {
        let plan = GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: 3,
                point_count: 4,
                observations: &bearings,
            },
            BataOptions {
                polish_steps,
                ..BataOptions::default()
            },
        )
        .unwrap();
        let mut workspace = plan.try_new_workspace().unwrap();
        let mut cameras = vec![Vector3::zeros(); 3];
        let mut points = vec![Vector3::zeros(); 4];
        plan.solve_into(&mut cameras, &mut points, &mut workspace)
            .unwrap()
    };
    let alternation = solve(0);
    let polished = solve(5);

    // The alternation alone is still 0.8 degrees from the exact fit after its iterations.
    assert!(alternation.rms_angular_error > 1e-3, "{alternation:?}");
    assert!(polished.polish_step_count > 0);
    assert!(polished.rms_angular_error < 1e-4, "{polished:?}");
}

#[test]
fn polish_stops_on_its_relative_cost_tolerance() {
    let (bearings, _, _) = exact_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 3,
            point_count: 4,
            observations: &bearings,
        },
        BataOptions {
            polish_steps: 50,
            ..BataOptions::default()
        },
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    let mut cameras = vec![Vector3::zeros(); 3];
    let mut points = vec![Vector3::zeros(); 4];
    let report = plan
        .solve_into(&mut cameras, &mut points, &mut workspace)
        .unwrap();
    // Converged long before the budget: running on would only invite a cut cascade.
    assert!((1..50).contains(&report.polish_step_count), "{report:?}");
    assert!(report.rms_angular_error < 1e-4, "{report:?}");
}

/// Twelve cameras on a ring around thirty points, with deterministic bearing noise.
#[cfg(feature = "faer")]
fn noisy_ring_problem() -> Vec<PositioningObservation> {
    let cameras: Vec<_> = (0..12)
        .map(|index| {
            let angle = std::f64::consts::TAU * f64::from(index) / 12.0;
            Vector3::new(3.0 * angle.cos(), 3.0 * angle.sin(), 0.2 * angle.sin())
        })
        .collect();
    let points: Vec<_> = (0..30)
        .map(|index| {
            let value = f64::from(index);
            Vector3::new(
                (1.3 * value).sin(),
                (0.7 * value).cos(),
                0.5 * (0.9 * value).sin(),
            )
        })
        .collect();
    let mut bearings = Vec::new();
    for (point_index, point) in points.iter().enumerate() {
        for (camera_index, camera) in cameras.iter().enumerate() {
            let seed = (point_index * 31 + camera_index) as f64;
            let noise = Vector3::new(seed.sin(), (2.0 * seed).cos(), (3.0 * seed).sin()) * 2e-3;
            bearings.push(PositioningObservation::bearing(
                camera_index,
                point_index,
                normalized(normalized(*point - *camera) + noise),
                1.0,
            ));
        }
    }
    bearings
}

#[cfg(feature = "faer")]
#[test]
fn polish_sparse_cholesky_matches_conjugate_gradients() {
    let bearings = noisy_ring_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 12,
            point_count: 30,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    let solve = |force_conjugate_gradients: bool| {
        let mut workspace = plan.try_new_workspace().unwrap();
        workspace.polish_direct_unavailable = force_conjugate_gradients;
        let mut cameras = vec![Vector3::zeros(); 12];
        let mut points = vec![Vector3::zeros(); 30];
        let report = plan
            .solve_into(&mut cameras, &mut points, &mut workspace)
            .unwrap();
        (cameras, report)
    };
    let (direct, direct_report) = solve(false);
    let (iterative, iterative_report) = solve(true);
    assert!(direct_report.polish_step_count > 0, "{direct_report:?}");
    assert_eq!(
        direct_report.polish_step_count,
        iterative_report.polish_step_count
    );
    for (a, b) in direct.iter().zip(&iterative) {
        assert!((a - b).norm() < 1e-6, "{a} vs {b}");
    }
}

/// Solve `bearings` with the alternation only (no polish), forcing the reduced-system path:
/// the sparse factorization, or the path that runs without it.
#[cfg(feature = "faer")]
fn alternation_solve(
    camera_count: usize,
    point_count: usize,
    bearings: &[PositioningObservation],
    sparse: bool,
) -> (Vec<Vector3<f64>>, super::GlobalPositioningReport) {
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count,
            point_count,
            observations: bearings,
        },
        BataOptions {
            polish_steps: 0,
            ..BataOptions::default()
        },
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    workspace.alternation_direct_unavailable = !sparse;
    let mut cameras = vec![Vector3::zeros(); camera_count];
    let mut points = vec![Vector3::zeros(); point_count];
    let report = plan
        .solve_into(&mut cameras, &mut points, &mut workspace)
        .unwrap();
    (cameras, report)
}

#[cfg(feature = "faer")]
#[test]
fn scalar_sparse_cholesky_matches_the_dense_reduced_solve_for_every_anchor() {
    let bearings = noisy_ring_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 12,
            point_count: 30,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    let mut workspace = plan.try_new_workspace().unwrap();
    // Unequal weights so that every coupling and diagonal entry differs.
    for (index, weight) in workspace.bearing_effective_weight.iter_mut().enumerate() {
        *weight = 0.5 + 0.01 * (index % 37) as f64;
    }
    for point in 0..plan.point_count {
        workspace.point_weight[point] = plan
            .point_bearing_range(point)
            .iter()
            .map(|&index| workspace.bearing_effective_weight[index])
            .sum();
    }
    super::solve::assemble_camera_system(&plan, &mut workspace);
    let mut direct = super::direct::ScalarCholesky::analyze(&plan, usize::MAX).unwrap();
    for anchor in 0..plan.camera_count {
        super::dense::solve_reduced_dense(&plan, &mut workspace, anchor).unwrap();
        let dense = workspace.trial_cameras.clone();
        let mut sparse = vec![Vector3::repeat(f64::NAN); plan.camera_count];
        assert!(direct.solve(
            &plan,
            &super::direct::ScalarSystem {
                camera_terms: &workspace.camera_terms,
                point_weight: &workspace.point_weight,
                bearing_effective_weight: &workspace.bearing_effective_weight,
                direction_effective_weight: &workspace.direction_effective_weight,
                anchor,
            },
            &mut sparse,
        ));
        assert_eq!(sparse[anchor], Vector3::zeros(), "anchor {anchor}");
        let scale = dense.iter().map(Vector3::norm).fold(0.0, f64::max);
        for (camera, (left, right)) in sparse.iter().zip(&dense).enumerate() {
            assert!(
                (left - right).norm() <= 1e-10 * scale,
                "anchor {anchor} camera {camera}: {left} vs {right}"
            );
        }
    }
}

#[cfg(feature = "faer")]
#[test]
fn alternation_sparse_cholesky_matches_the_dense_path_on_a_small_scene() {
    let bearings = noisy_ring_problem();
    let (sparse, sparse_report) = alternation_solve(12, 30, &bearings, true);
    let (dense, dense_report) = alternation_solve(12, 30, &bearings, false);
    assert_eq!(
        sparse_report.reduced_backend,
        super::api::ReducedSystemBackend::SparseCholesky
    );
    assert_eq!(
        dense_report.reduced_backend,
        super::api::ReducedSystemBackend::DenseCholesky
    );
    assert_eq!(sparse_report.iteration_count, dense_report.iteration_count);
    for (left, right) in sparse.iter().zip(&dense) {
        assert!((left - right).norm() < 1e-9, "{left} vs {right}");
    }
}

#[cfg(feature = "faer")]
#[test]
fn alternation_sparse_cholesky_matches_conjugate_gradients_on_a_large_scene() {
    let camera_count = 257;
    let cameras: Vec<_> = (0..camera_count)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / camera_count as f64;
            Vector3::new(angle.cos(), angle.sin(), 0.05 * (2.0 * angle).sin())
        })
        .collect();
    let mut bearings = Vec::with_capacity(camera_count * 8);
    for point in 0..camera_count {
        let angle = std::f64::consts::TAU * (point as f64 + 0.35) / camera_count as f64;
        let position = Vector3::new(
            0.35 * angle.cos(),
            0.35 * angle.sin(),
            3.0 + 0.001 * point as f64,
        );
        for offset in 0..8 {
            let camera = (point + offset) % camera_count;
            bearings.push(PositioningObservation::bearing(
                camera,
                point,
                normalized(position - cameras[camera]),
                1.0,
            ));
        }
    }
    let (sparse, sparse_report) = alternation_solve(camera_count, camera_count, &bearings, true);
    let (iterative, iterative_report) =
        alternation_solve(camera_count, camera_count, &bearings, false);
    assert_eq!(
        sparse_report.reduced_backend,
        super::api::ReducedSystemBackend::SparseCholesky
    );
    assert_eq!(sparse_report.pcg_iteration_count, 0);
    assert_eq!(
        iterative_report.reduced_backend,
        super::api::ReducedSystemBackend::SparseCouplingPcg
    );
    assert_eq!(
        sparse_report.iteration_count,
        iterative_report.iteration_count
    );
    let difference = sparse
        .iter()
        .zip(&iterative)
        .map(|(left, right)| (left - right).norm())
        .fold(0.0, f64::max);
    assert!(difference < 1e-7, "{difference:e}");
}

#[cfg(feature = "faer")]
#[test]
fn alternation_sparse_cholesky_is_deterministic() {
    let bearings = noisy_ring_problem();
    let (first, first_report) = alternation_solve(12, 30, &bearings, true);
    let (second, second_report) = alternation_solve(12, 30, &bearings, true);
    assert_eq!(first, second);
    assert_eq!(first_report, second_report);
}

#[cfg(feature = "faer")]
#[test]
fn scalar_sparse_cholesky_refuses_an_analysis_over_its_budget() {
    let bearings = noisy_ring_problem();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: 12,
            point_count: 30,
            observations: &bearings,
        },
        BataOptions::default(),
    )
    .unwrap();
    assert!(super::direct::ScalarCholesky::analyze(&plan, usize::MAX).is_some());
    assert!(super::direct::ScalarCholesky::analyze(&plan, 1024).is_none());
}
