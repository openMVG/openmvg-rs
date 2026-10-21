// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::calibrated::PreparedCalibratedProblem;
use super::gauge::gauge_plan;
use super::reporting::certified_observability;
use super::*;
use crate::kernel::multi_view::ScaleGauge;
use crate::kernel::multi_view::sparse::SparseCameraRankAssessment;
use crate::{
    AuditError, BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose,
    BundleProblem, Gauge, IntrinsicVariables, Observability, PoseVariables, Solution, SolveConfig,
    SolveError,
};
use nalgebra::{Point2, Point3, Rotation3, Vector2, Vector3};
use openmvg_core::{
    camera::{BrownConradyCamera, Camera, CameraModel, RadialDistortion},
    model::{CameraIntrinsics, WorldToCameraPose},
};

fn certified_post_solve_probe(dense_observability_escalation: bool) -> NativePreflight {
    NativePreflight {
        status: NativePreflightStatus::Proceed,
        pose_dimension: Some(12),
        covered_free_pose_count: Some(4),
        weak_point_count: Some(0),
        dense_observability_escalation,
    }
}

#[test]
fn certified_fixed_rotation_subset_closes_unassessed_sparse_rank() {
    let fixed = Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance: 1.0,
    };
    assert_eq!(
        certified_observability(
            SparseCameraRankAssessment::NotAssessed,
            Some(certified_post_solve_probe(false)),
            fixed,
        ),
        Observability::FullRank
    );
    assert_eq!(
        certified_observability(
            SparseCameraRankAssessment::NotAssessed,
            Some(certified_post_solve_probe(false)),
            Gauge::Free,
        ),
        Observability::OnlyExpectedGauge
    );
}

#[test]
fn escalation_cannot_promote_unassessed_sparse_rank() {
    let fixed = Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance: 1.0,
    };
    assert_eq!(
        certified_observability(
            SparseCameraRankAssessment::NotAssessed,
            Some(certified_post_solve_probe(true)),
            fixed,
        ),
        Observability::NotAssessed
    );
}

#[test]
fn subset_certificate_never_overrides_negative_primary_evidence() {
    let fixed = Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance: 1.0,
    };
    assert_eq!(
        certified_observability(
            SparseCameraRankAssessment::RankDeficientOrIndefinite,
            Some(certified_post_solve_probe(false)),
            fixed,
        ),
        Observability::RankDeficient
    );
}

fn fixture(observation_order: [usize; 6]) -> BundleProblem {
    let camera = Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
    let poses = [
        WorldToCameraPose::identity(),
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(0.5, 0.0, 0.0))
            .unwrap(),
    ];
    let points = [
        Point3::new(-0.5, 0.0, 4.0),
        Point3::new(0.5, 0.0, 4.5),
        Point3::new(0.0, 0.5, 5.0),
    ];
    let canonical = poses
        .iter()
        .enumerate()
        .flat_map(|(camera_index, pose)| {
            let camera = camera.clone();
            points
                .into_iter()
                .enumerate()
                .map(move |(point_index, point)| {
                    let projected = pose.project_to_camera(&point);
                    BundleObservation::try_new(
                        camera_index,
                        0,
                        point_index,
                        openmvg_core::camera::CameraModel::project(&camera, &projected.coords)
                            .unwrap(),
                    )
                    .unwrap()
                })
        })
        .collect::<Vec<_>>();
    BundleProblem::builder(
        poses
            .into_iter()
            .enumerate()
            .map(|(index, pose)| {
                BundlePose::new(
                    pose,
                    if index == 0 {
                        PoseVariables::FIXED
                    } else {
                        PoseVariables::FREE
                    },
                )
            })
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(camera, 640, 480)
                    .unwrap()
                    .with_fixed(true),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        points
            .into_iter()
            .map(|point| BundlePoint::try_new(point).unwrap())
            .collect(),
        observation_order
            .into_iter()
            .map(|index| canonical[index])
            .collect(),
    )
    .with_gauge(Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance: 0.5,
    })
    .build()
    .unwrap()
}

fn translated_non_identity_fixture(
    offset: Vector3<f64>,
    observation_order: [usize; 12],
    anchor: usize,
    translation_only: bool,
) -> BundleProblem {
    let camera = Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap();
    let truth_poses = [
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.11, -0.07, 0.04),
            Point3::from(offset),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.08, 0.05, -0.03),
            Point3::from(offset + Vector3::new(0.8, 0.1, -0.05)),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(-0.06, 0.09, 0.07),
            Point3::from(offset + Vector3::new(0.15, 0.7, 0.1)),
        )
        .unwrap(),
    ];
    let truth_points = [
        Point3::from(offset + Vector3::new(-0.7, -0.4, 4.2)),
        Point3::from(offset + Vector3::new(0.6, -0.3, 4.8)),
        Point3::from(offset + Vector3::new(-0.2, 0.7, 5.4)),
        Point3::from(offset + Vector3::new(0.8, 0.5, 6.1)),
    ];
    let observations = truth_poses
        .iter()
        .enumerate()
        .flat_map(|(camera_index, pose)| {
            let camera = camera.clone();
            truth_points
                .iter()
                .enumerate()
                .map(move |(point_index, point)| {
                    let camera_point = pose.project_to_camera(point);
                    BundleObservation::try_new(
                        camera_index,
                        0,
                        point_index,
                        openmvg_core::camera::CameraModel::project(&camera, &camera_point.coords)
                            .unwrap(),
                    )
                    .unwrap()
                })
        })
        .collect::<Vec<_>>();
    let initial_poses = [
        truth_poses[0].clone(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.085, 0.045, -0.025),
            truth_poses[1].center() + Vector3::new(0.02, -0.015, 0.01),
        )
        .unwrap(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(-0.055, 0.085, 0.065),
            truth_poses[2].center() + Vector3::new(-0.015, 0.01, -0.012),
        )
        .unwrap(),
    ];
    let initial_points = truth_points.map(|point| point + Vector3::new(0.01, -0.008, 0.015));
    let scale_camera = if anchor == 0 { 1 } else { 0 };
    let baseline = (truth_poses[scale_camera].center() - truth_poses[anchor].center()).norm();

    BundleProblem::builder(
        initial_poses
            .into_iter()
            .enumerate()
            .map(|(index, pose)| {
                BundlePose::new(
                    pose,
                    if index == anchor {
                        PoseVariables::FIXED
                    } else if translation_only {
                        PoseVariables::FIXED.with_translation(true)
                    } else {
                        PoseVariables::FREE
                    },
                )
            })
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(camera, 640, 480)
                    .unwrap()
                    .with_fixed(true),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        initial_points
            .into_iter()
            .map(|point| BundlePoint::try_new(point).unwrap())
            .collect(),
        observation_order
            .into_iter()
            .map(|index| observations[index])
            .collect(),
    )
    .with_gauge(Gauge::FixedBaseline {
        first: anchor,
        second: scale_camera,
        distance: baseline,
    })
    .build()
    .unwrap()
}

fn solve_stress_fixture(problem: &BundleProblem) -> Solution {
    let config = SolveConfig::default();
    let solution = NativeBundleAdjuster::new()
        .solve(problem, config)
        .expect("well-conditioned stress fixture must solve");
    solution
        .audit_for_commit(problem, &config)
        .expect("stress solution must pass the independent commit audit");
    solution
}

#[test]
fn unobserved_camera_never_sets_the_scale_gauge_and_does_not_move() {
    let base = translated_non_identity_fixture(
        Vector3::zeros(),
        std::array::from_fn(|index| index),
        0,
        false,
    );
    // A far camera whose observations were all removed, as after outlier
    // pruning; it must not become the scale gauge.
    let unobserved = WorldToCameraPose::from_rotation_center(
        Rotation3::identity(),
        Point3::new(0.0, 0.0, 1.0e7),
    )
    .unwrap();
    let mut cameras = base.poses().to_vec();
    cameras.push(BundlePose::new(unobserved.clone(), PoseVariables::FREE));
    let problem = BundleProblem::builder(
        cameras,
        base.intrinsics().to_vec(),
        base.points().to_vec(),
        base.observations().to_vec(),
    )
    .build()
    .unwrap();

    let plan = gauge_plan(&problem).unwrap();
    assert_eq!(plan.anchor, 0);
    assert!(matches!(plan.native, ScaleGauge::FixedBaseline { camera } if camera != 3));
    let prepared = PreparedCalibratedProblem::try_new(&problem).unwrap();
    assert_eq!(prepared.fixed_poses, [true, false, false, false]);

    let config = SolveConfig::default();
    let solution = NativeBundleAdjuster::new()
        .solve(&problem, config)
        .expect("an unobserved camera must not break the solve");
    // Its pose is unconstrained, which the commit audit reports.
    assert!(matches!(
        solution.audit_for_commit(&problem, &config),
        Err(AuditError::RankDeficient)
    ));
    // Without curvature or gradient the pose takes no step; only the
    // native world-frame round trip may touch it.
    let held = &solution.poses()[3];
    assert!((held.center() - unobserved.center()).norm() <= 1.0e-15 * 1.0e7);
    assert!(held.rotation().angle_to(unobserved.rotation()) <= 1.0e-15);
}

#[test]
fn a_point_without_depth_information_is_quarantined_instead_of_vetoing_the_cameras() {
    let base = translated_non_identity_fixture(
        Vector3::zeros(),
        std::array::from_fn(|index| index),
        0,
        false,
    );
    // A point seen by two fixed cameras whose centers lie on its viewing
    // ray has no depth information: its undamped point block is singular.
    let camera = Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap();
    let lone = Point3::new(0.1, -0.2, 5.0);
    let behind = WorldToCameraPose::from_rotation_center(
        *base.poses()[0].pose().rotation(),
        base.poses()[0].pose().center() - 0.2 * (lone - base.poses()[0].pose().center()),
    )
    .unwrap();
    let mut cameras = base.poses().to_vec();
    cameras.push(BundlePose::new(behind, PoseVariables::FIXED));
    let mut points = base.points().to_vec();
    points.push(BundlePoint::try_new(lone).unwrap());
    let mut observations = base.observations().to_vec();
    for camera_index in [0, 3] {
        let pixel = openmvg_core::camera::CameraModel::project(
            &camera,
            &cameras[camera_index].pose().project_to_camera(&lone).coords,
        )
        .unwrap();
        observations
            .push(BundleObservation::try_new(camera_index, 0, points.len() - 1, pixel).unwrap());
    }
    let problem = BundleProblem::builder(cameras, base.intrinsics().to_vec(), points, observations)
        .build()
        .unwrap();

    let config = SolveConfig::default();
    let solution = NativeBundleAdjuster::new()
        .solve(&problem, config)
        .expect("a weak point must not discard the solve");
    let diagnostics = solution.report().diagnostics();
    let lone_index = problem.points().len() - 1;
    // The solver held the point out of the system it certified, so no weak block remains, and it
    // keeps its input position.
    assert_eq!(diagnostics.weak_point_count(), Some(0));
    assert_eq!(diagnostics.quarantined_point_count(), 1);
    assert_eq!(solution.points()[lone_index], lone);
    assert_eq!(solution.weak_points(&problem), [lone_index]);
    assert!(
        diagnostics
            .stationarity()
            .expect("the native solve measures final stationarity")
            .scaled_gradient_inf_norm()
            .is_finite()
    );
    assert_eq!(solution.audit_for_commit(&problem, &config), Ok(()));
}

#[test]
fn points_seen_once_or_never_are_quarantined_instead_of_failing_the_solve() {
    let base = translated_non_identity_fixture(
        Vector3::zeros(),
        std::array::from_fn(|index| index),
        0,
        false,
    );
    // One free point seen by a single camera has no depth, and another is seen by none.
    let camera = Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap();
    let once = Point3::new(0.2, 0.1, 5.0);
    let never = Point3::new(-0.3, 0.2, 6.0);
    let mut points = base.points().to_vec();
    points.push(BundlePoint::try_new(once).unwrap());
    points.push(BundlePoint::try_new(never).unwrap());
    let mut observations = base.observations().to_vec();
    let pixel = CameraModel::project(
        &camera,
        &base.poses()[1].pose().project_to_camera(&once).coords,
    )
    .unwrap();
    observations.push(BundleObservation::try_new(1, 0, points.len() - 2, pixel).unwrap());
    let problem = BundleProblem::builder(
        base.poses().to_vec(),
        base.intrinsics().to_vec(),
        points,
        observations,
    )
    .build()
    .unwrap();

    let config = SolveConfig::default();
    let solution = NativeBundleAdjuster::new()
        .solve(&problem, config)
        .expect("points the observations cannot place must not fail the solve");
    let (once_index, never_index) = (problem.points().len() - 2, problem.points().len() - 1);
    assert_eq!(solution.quarantined_points(), [once_index, never_index]);
    assert_eq!(solution.points()[once_index], once);
    assert_eq!(solution.points()[never_index], never);
    assert_eq!(solution.audit_for_commit(&problem, &config), Ok(()));

    // When no point is left to adjust, the solve says so instead of failing inside.
    let lone = BundleProblem::builder(
        problem.poses().to_vec(),
        problem.intrinsics().to_vec(),
        vec![BundlePoint::try_new(once).unwrap()],
        vec![BundleObservation::try_new(1, 0, 0, pixel).unwrap()],
    )
    .build()
    .unwrap();
    assert!(matches!(
        NativeBundleAdjuster::new().solve(&lone, config),
        Err(SolveError::Unsupported(_))
    ));
}

#[test]
fn a_failed_certificate_is_repaired_by_resolving_without_the_weak_points() {
    let base = translated_non_identity_fixture(
        Vector3::zeros(),
        std::array::from_fn(|index| index),
        0,
        false,
    );
    // A point 16 000 baselines away is seen at a parallax of about 5e-5 rad: above the geometric
    // floor, but its undamped block is far too ill conditioned to eliminate.
    let camera = Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap();
    let far = Point3::new(0.3, -0.1, 1.6e4);
    let mut points = base.points().to_vec();
    points.push(BundlePoint::try_new(far).unwrap());
    let mut observations = base.observations().to_vec();
    for camera_index in [0, 1] {
        let pixel = CameraModel::project(
            &camera,
            &base.poses()[camera_index]
                .pose()
                .project_to_camera(&far)
                .coords,
        )
        .unwrap();
        observations
            .push(BundleObservation::try_new(camera_index, 0, points.len() - 1, pixel).unwrap());
    }
    let problem = BundleProblem::builder(
        base.poses().to_vec(),
        base.intrinsics().to_vec(),
        points,
        observations,
    )
    // The stub returns its input state. Use a free gauge so that state is feasible;
    // the perturbed seed does not satisfy the truth's metric baseline.
    .with_gauge(Gauge::Free)
    .build()
    .unwrap();
    let far_index = problem.points().len() - 1;
    let config = SolveConfig::default();

    // The first solve claims a rank deficiency caused by weak points; the repair must then
    // resolve a problem that no longer contains the far point.
    let calls = std::cell::RefCell::new(Vec::new());
    let solve_stub = |candidate: &BundleProblem, config: SolveConfig| {
        calls.borrow_mut().push(candidate.points().len());
        if calls.borrow().len() > 1 {
            return solve_once(candidate, config);
        }
        let diagnostics = crate::SolveDiagnostics::new(
            2,
            1,
            crate::StationarityAudit::try_from_fresh_zero_damping(0.0).unwrap(),
            Observability::RankDeficient,
            1,
            0,
        );
        let report = crate::SolveReport::try_new(
            SolveStatus::Degenerate,
            3,
            10.0,
            5.0,
            1.0,
            0.5,
            diagnostics,
        )
        .unwrap();
        Solution::try_new(
            candidate,
            candidate
                .poses()
                .iter()
                .map(|pose| pose.pose().clone())
                .collect(),
            candidate
                .intrinsics()
                .iter()
                .map(|intrinsic| intrinsic.intrinsics().clone())
                .collect(),
            candidate
                .points()
                .iter()
                .map(|point| *point.position())
                .collect(),
            report,
        )
        .map_err(|error| SolveError::Internal(error.to_string()))
    };
    let solution = solve_with_quarantine(&problem, config, &solve_stub)
        .expect("the resolve repairs the failed certificate");

    assert_eq!(
        *calls.borrow(),
        [problem.points().len(), problem.points().len() - 1]
    );
    let diagnostics = solution.report().diagnostics();
    assert_eq!(diagnostics.quarantined_point_count(), 1);
    assert_eq!(diagnostics.weak_point_count(), Some(0));
    assert!(!matches!(
        solution.report().status(),
        SolveStatus::Degenerate
    ));
    assert_eq!(solution.points()[far_index], far);
    // Quarantine recomputes the input cost on the final retained observations;
    // the stub's arbitrary metric must not survive lifting.
    let initial_cost = problem
        .observations()
        .iter()
        .filter(|observation| observation.point() != far_index)
        .map(|observation| {
            let camera = problem.intrinsics()[observation.intrinsic()]
                .intrinsics()
                .camera();
            let camera_point = problem.poses()[observation.pose()]
                .pose()
                .project_to_camera(problem.points()[observation.point()].position());
            let residual = camera.project(&camera_point.coords).unwrap() - observation.pixel();
            0.5 * residual.norm_squared()
        })
        .sum::<f64>();
    assert!((solution.report().initial_objective() - initial_cost).abs() < 1.0e-10);
    assert_eq!(solution.quarantined_points(), &[far_index]);
    assert!(solution.report().iteration_count() > 3);
    assert_eq!(solution.audit_for_commit(&problem, &config), Ok(()));
}

#[test]
fn a_healthy_solve_is_not_resolved_or_quarantined() {
    let problem = translated_non_identity_fixture(
        Vector3::zeros(),
        std::array::from_fn(|index| index),
        0,
        false,
    );
    let calls = std::cell::Cell::new(0);
    let counting = |candidate: &BundleProblem, config: SolveConfig| {
        calls.set(calls.get() + 1);
        solve_once(candidate, config)
    };
    let solution = solve_with_quarantine(&problem, SolveConfig::default(), &counting).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(solution.report().diagnostics().quarantined_point_count(), 0);
}

#[test]
fn preparation_is_point_major_and_retains_source_order_within_tracks() {
    let problem = fixture([5, 1, 3, 0, 4, 2]);
    let prepared = PreparedCalibratedProblem::try_new(&problem).unwrap();
    assert_eq!(prepared.point_offsets, [0, 2, 4, 6]);
    assert_eq!(prepared.observation_source_order, [2, 3, 1, 4, 0, 5]);
    assert_eq!(
        prepared
            .observations
            .iter()
            .map(|observation| observation.camera)
            .collect::<Vec<_>>(),
        [1, 0, 0, 1, 1, 0]
    );
}

#[test]
fn native_failure_cannot_mutate_the_owned_problem() {
    let mut problem = fixture([0, 1, 2, 3, 4, 5]);
    let before = problem.clone();
    let mut adjuster = NativeBundleAdjuster::new();
    assert_eq!(
        adjuster.solve(&problem, SolveConfig::default().with_max_iterations(0)),
        Err(SolveError::InvalidConfig("max_iterations must be positive"))
    );
    assert_eq!(problem, before);

    // Keep the binding mutable to demonstrate that failure did not rely on
    // Rust forbidding a caller-side mutation after the call.
    problem = before;
    assert_eq!(problem.observations().len(), 6);
}

/// Bundle adjustment world-origin invariance: a non-identity camera configuration solves in both
/// near-origin and far-origin world frames; a translation of one thousand scene scales preserves
/// poses and points after undoing the shift; the shifted and unshifted solves require the same
/// iteration count and pass the fresh audit.
#[test]
fn world_origin_shift_preserves_non_identity_solution() {
    let order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11];
    let origin = translated_non_identity_fixture(Vector3::zeros(), order, 0, false);
    let scene_scale = 6.1;
    let offset = Vector3::new(1000.0, -700.0, 500.0) * scene_scale;
    let shifted = translated_non_identity_fixture(offset, order, 0, false);

    // At this origin, rounded world coordinates cannot certify the requested
    // sub-unit baseline at the public relative tolerance. The refusal is separate
    // from coordinate equivariance of the unconstrained-scale objective.
    assert!(matches!(
        NativeBundleAdjuster::new().solve(&shifted, SolveConfig::default()),
        Err(SolveError::Numerical(message)) if message.contains("fixed-baseline")
    ));
    let without_metric_baseline = |problem: &BundleProblem| {
        BundleProblem::builder(
            problem.poses().to_vec(),
            problem.intrinsics().to_vec(),
            problem.points().to_vec(),
            problem.observations().to_vec(),
        )
        .with_residual_space(problem.residual_space())
        .with_robust_loss(problem.robust_loss())
        .with_gauge(Gauge::Free)
        .build()
        .unwrap()
    };
    let origin = without_metric_baseline(&origin);
    let shifted = without_metric_baseline(&shifted);

    let origin_solution = solve_stress_fixture(&origin);
    let shifted_solution = solve_stress_fixture(&shifted);

    let max_rotation_error = origin_solution
        .poses()
        .iter()
        .zip(shifted_solution.poses())
        .map(|(left, right)| left.rotation().angle_to(right.rotation()))
        .fold(0.0_f64, f64::max);
    let max_center_error = origin_solution
        .poses()
        .iter()
        .zip(shifted_solution.poses())
        .map(|(left, right)| (left.center() - (right.center() - offset)).norm())
        .fold(0.0_f64, f64::max);
    let max_point_error = origin_solution
        .points()
        .iter()
        .zip(shifted_solution.points())
        .map(|(left, right)| (left - (right - offset)).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_rotation_error < 1.0e-9,
        "rotation changed by {max_rotation_error:e} radians"
    );
    assert!(
        max_center_error < 1.0e-8,
        "camera center changed by {max_center_error:e} scene units"
    );
    assert!(
        max_point_error < 1.0e-8,
        "point changed by {max_point_error:e} scene units"
    );
    assert_eq!(
        origin_solution.report().iteration_count(),
        shifted_solution.report().iteration_count()
    );
}

/// Bundle adjustment observation-order invariance: permuting the same checked observations
/// preserves optimized poses and points; both solutions pass the independent commit audit.
#[test]
fn observation_permutation_preserves_native_solution() {
    let canonical = translated_non_identity_fixture(
        Vector3::zeros(),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        0,
        false,
    );
    let permuted = translated_non_identity_fixture(
        Vector3::zeros(),
        [11, 2, 9, 4, 7, 0, 10, 3, 6, 1, 8, 5],
        0,
        false,
    );
    let canonical_solution = solve_stress_fixture(&canonical);
    let permuted_solution = solve_stress_fixture(&permuted);

    let max_rotation_error = canonical_solution
        .poses()
        .iter()
        .zip(permuted_solution.poses())
        .map(|(left, right)| left.rotation().angle_to(right.rotation()))
        .fold(0.0_f64, f64::max);
    let max_center_error = canonical_solution
        .poses()
        .iter()
        .zip(permuted_solution.poses())
        .map(|(left, right)| (left.center() - right.center()).norm())
        .fold(0.0_f64, f64::max);
    let max_point_error = canonical_solution
        .points()
        .iter()
        .zip(permuted_solution.points())
        .map(|(left, right)| (left - right).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_rotation_error < 1.0e-9,
        "rotation changed by {max_rotation_error:e} radians"
    );
    assert!(
        max_center_error < 1.0e-9,
        "camera center changed by {max_center_error:e} scene units"
    );
    assert!(
        max_point_error < 1.0e-9,
        "point changed by {max_point_error:e} scene units"
    );
}

#[test]
fn nonzero_gauge_anchor_is_permuted_into_the_native_world_frame() {
    let problem = translated_non_identity_fixture(
        Vector3::new(10.0, -20.0, 30.0),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        2,
        false,
    );
    let solution = solve_stress_fixture(&problem);
    assert_eq!(solution.poses()[2], *problem.poses()[2].pose());
}

#[test]
fn translation_only_solve_preserves_rotations_bit_exactly() {
    let problem = translated_non_identity_fixture(
        Vector3::new(10.0, -20.0, 30.0),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        2,
        true,
    );
    let solution = solve_stress_fixture(&problem);
    for (candidate, input) in solution.poses().iter().zip(problem.poses()) {
        assert_eq!(candidate.rotation(), input.pose().rotation());
    }
    let observability = solution
        .report()
        .diagnostics()
        .observability_details()
        .expect("translation-only solve must carry fixed-rotation rank evidence");
    assert_eq!(observability.pose_rank(), observability.pose_dimension());
    assert_eq!(observability.weak_point_count(), 0);
}

fn brown_conrady(focal: f64, radial: RadialDistortion) -> Camera {
    Camera::BrownConrady(
        BrownConradyCamera::try_with_focal(
            focal,
            320.0,
            240.0,
            radial,
            openmvg_core::camera::TangentialDistortion::None,
        )
        .unwrap(),
    )
}

/// Nine cameras on an arc around a deep point cloud, three per intrinsic group. Observations
/// are noise-free projections through `truth`; the solve starts from `initial` intrinsics,
/// perturbed poses, and perturbed points.
fn mixed_model_fixture(
    truth: &[Camera; 3],
    initial: &[Camera; 3],
    variables: [IntrinsicVariables; 3],
) -> BundleProblem {
    mixed_model_fixture_observing(truth, initial, variables, |_, _| true)
}

/// [`mixed_model_fixture`] keeping only the observations `observes(camera, pixel)` accepts.
fn mixed_model_fixture_observing(
    truth: &[Camera; 3],
    initial: &[Camera; 3],
    variables: [IntrinsicVariables; 3],
    observes: impl Fn(usize, &Point2<f64>) -> bool + Copy,
) -> BundleProblem {
    let pose = |camera: usize, perturbation: f64| {
        let angle = -0.48 + 0.12 * camera as f64;
        let rotation = Rotation3::from_euler_angles(
            0.03 * ((camera % 3) as f64 - 1.0) + perturbation,
            angle,
            0.05 * ((camera % 2) as f64 - 0.5) - perturbation,
        );
        let center = Point3::new(
            5.5 * angle.sin(),
            0.2 * ((camera % 3) as f64 - 1.0),
            5.5 - 5.5 * angle.cos(),
        ) + Vector3::repeat(perturbation);
        WorldToCameraPose::from_rotation_center(rotation, center).unwrap()
    };
    let points = (0..270)
        .map(|index| {
            let jitter = index as f64;
            Point3::new(
                -2.4 + 0.6 * (index % 9) as f64 + 0.07 * jitter.sin(),
                -1.75 + 0.7 * ((index / 9) % 6) as f64 + 0.05 * jitter.cos(),
                4.5 + 0.5 * (index / 54) as f64 + 0.1 * (0.7 * jitter).sin(),
            )
        })
        .collect::<Vec<_>>();
    let observations = (0..9)
        .flat_map(|camera| {
            let truth_pose = pose(camera, 0.0);
            let model = truth[camera / 3].clone();
            points
                .iter()
                .enumerate()
                .filter_map(move |(point, position)| {
                    let projected = truth_pose.project_to_camera(position);
                    let pixel =
                        openmvg_core::camera::CameraModel::project(&model, &projected.coords)
                            .unwrap();
                    observes(camera, &pixel).then(|| {
                        BundleObservation::try_new(camera, camera / 3, point, pixel).unwrap()
                    })
                })
        })
        .collect();
    let distance = (pose(1, 0.0).center() - pose(0, 0.0).center()).norm();
    BundleProblem::builder(
        (0..9)
            .map(|camera| {
                if camera == 0 {
                    BundlePose::new(pose(0, 0.0), PoseVariables::FIXED)
                } else {
                    BundlePose::new(pose(camera, 0.004), PoseVariables::FREE)
                }
            })
            .collect(),
        initial
            .iter()
            .zip(variables)
            .map(|(camera, variables)| {
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(camera.clone(), 640, 480).unwrap(),
                    variables,
                )
                .unwrap()
            })
            .collect(),
        points
            .iter()
            .map(|point| BundlePoint::try_new(point + Vector3::new(0.02, -0.015, 0.03)).unwrap())
            .collect(),
        observations,
    )
    .with_gauge(Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance,
    })
    .build()
    .unwrap()
}

const SELF_CALIBRATE: IntrinsicVariables =
    IntrinsicVariables::FIXED.with_focal(true).with_radial(true);

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn self_calibration_frees_exactly_each_groups_declared_model() {
    let truth = [
        Camera::try_pinhole(600.0, 600.0, 320.0, 240.0).unwrap(),
        brown_conrady(720.0, RadialDistortion::K1(-0.12)),
        brown_conrady(650.0, RadialDistortion::K1K2(0.08, -0.03)),
    ];
    let initial = [
        Camera::try_pinhole(624.0, 624.0, 320.0, 240.0).unwrap(),
        brown_conrady(749.0, RadialDistortion::K1(0.0)),
        brown_conrady(676.0, RadialDistortion::K1K2(0.0, 0.0)),
    ];
    let problem = mixed_model_fixture(&truth, &initial, [SELF_CALIBRATE; 3]);
    let solution = solve_stress_fixture(&problem);
    assert!(solution.report().final_rmse() < 1e-5);

    let Camera::Pinhole(pinhole) = solution.intrinsics()[0].camera() else {
        panic!("a pinhole group must stay pinhole");
    };
    assert_relative(pinhole.focal_x(), 600.0, 1e-6);
    let Camera::BrownConrady(k1) = solution.intrinsics()[1].camera() else {
        panic!("a K1 group must stay Brown-Conrady");
    };
    let RadialDistortion::K1(k1_value) = *k1.radial_distortion() else {
        panic!("a K1 group must stay K1, got {:?}", k1.radial_distortion());
    };
    assert_relative(k1.focal_x(), 720.0, 1e-6);
    assert_relative(k1_value, -0.12, 1e-6);
    let Camera::BrownConrady(k1k2) = solution.intrinsics()[2].camera() else {
        panic!("a K1K2 group must stay Brown-Conrady");
    };
    let RadialDistortion::K1K2(k1_value, k2_value) = *k1k2.radial_distortion() else {
        panic!("a K1K2 group must stay K1K2");
    };
    assert_relative(k1k2.focal_x(), 650.0, 1e-6);
    assert_relative(k1_value, 0.08, 1e-6);
    assert_relative(k2_value, -0.03, 1e-6);
    for camera in solution.intrinsics() {
        assert_eq!(camera.camera().principal_point(), Point2::new(320.0, 240.0));
    }
}

#[test]
fn thin_group_keeps_its_input_intrinsics_while_others_self_calibrate() {
    let truth = [
        Camera::try_pinhole(600.0, 600.0, 320.0, 240.0).unwrap(),
        brown_conrady(720.0, RadialDistortion::K1(-0.12)),
        brown_conrady(650.0, RadialDistortion::K1K2(0.08, -0.03)),
    ];
    let initial = [
        Camera::try_pinhole(624.0, 624.0, 320.0, 240.0).unwrap(),
        brown_conrady(749.0, RadialDistortion::K1(0.0)),
        truth[2].clone(),
    ];
    // The K1K2 group only sees the image center, where k1 and k2 barely move a pixel: the
    // reduced system stays full rank, but its distortion is not constrained.
    let problem =
        mixed_model_fixture_observing(&truth, &initial, [SELF_CALIBRATE; 3], |camera, pixel| {
            camera < 6 || (pixel - Point2::new(320.0, 240.0)).norm() < 100.0
        });
    let solution = solve_stress_fixture(&problem);
    assert_eq!(solution.report().unconstrained_intrinsics(), [2]);
    assert_eq!(
        solution.intrinsics()[2],
        *problem.intrinsics()[2].intrinsics()
    );
    assert!(solution.report().final_rmse() < 1e-5);
    assert_relative(
        solution.intrinsics()[0].camera().focal_lengths().unwrap().0,
        600.0,
        1e-6,
    );
    let Camera::BrownConrady(k1) = solution.intrinsics()[1].camera() else {
        panic!("a K1 group must stay Brown-Conrady");
    };
    assert_relative(k1.focal_x(), 720.0, 1e-6);
    assert_relative(k1.radial_distortion().k1(), -0.12, 1e-6);
}

#[test]
fn fixed_distorted_group_is_honored_while_others_self_calibrate() {
    let truth = [
        Camera::try_pinhole(600.0, 600.0, 320.0, 240.0).unwrap(),
        brown_conrady(720.0, RadialDistortion::K1K2(-0.12, 0.02)),
        brown_conrady(650.0, RadialDistortion::K1(0.08)),
    ];
    let initial = [
        Camera::try_pinhole(624.0, 624.0, 320.0, 240.0).unwrap(),
        truth[1].clone(),
        brown_conrady(676.0, RadialDistortion::K1(0.0)),
    ];
    let problem = mixed_model_fixture(
        &truth,
        &initial,
        [SELF_CALIBRATE, IntrinsicVariables::FIXED, SELF_CALIBRATE],
    );
    let solution = solve_stress_fixture(&problem);
    assert!(solution.report().final_rmse() < 1e-5);
    assert_eq!(
        solution.intrinsics()[1],
        *problem.intrinsics()[1].intrinsics()
    );
    assert_relative(
        solution.intrinsics()[0].camera().focal_lengths().unwrap().0,
        600.0,
        1e-6,
    );
    let Camera::BrownConrady(k1) = solution.intrinsics()[2].camera() else {
        panic!("a K1 group must stay Brown-Conrady");
    };
    assert_eq!(k1.radial_distortion().k2(), 0.0);
    assert_relative(k1.radial_distortion().k1(), 0.08, 1e-6);
}

/// The first two views of [`mixed_model_fixture`] with one radially distorted camera: an
/// incremental initial pair whose solve starts from `initial` intrinsics.
fn distorted_seed_pair(initial: Camera, variables: IntrinsicVariables) -> BundleProblem {
    let truth = brown_conrady(720.0, RadialDistortion::K1(-0.12));
    let nine_views = mixed_model_fixture(
        &[truth.clone(), truth.clone(), truth],
        &[initial.clone(), initial.clone(), initial],
        [variables; 3],
    );
    BundleProblem::builder(
        nine_views.poses()[..2].to_vec(),
        nine_views.intrinsics()[..1].to_vec(),
        nine_views.points().to_vec(),
        nine_views
            .observations()
            .iter()
            .filter(|observation| observation.pose() < 2)
            .cloned()
            .collect(),
    )
    .with_gauge(nine_views.gauge())
    .build()
    .unwrap()
}

#[test]
fn distorted_two_view_seed_is_solved_with_its_declared_distortion() {
    let problem = distorted_seed_pair(
        brown_conrady(720.0, RadialDistortion::K1(-0.12)),
        IntrinsicVariables::FIXED,
    );
    let solution = solve_stress_fixture(&problem);
    assert!(solution.report().final_rmse() < 1e-5);
    assert_eq!(
        solution.intrinsics()[0],
        *problem.intrinsics()[0].intrinsics()
    );
}

#[test]
fn distorted_two_view_seed_self_calibration_holds_the_unconstrained_group() {
    // An initial pair does not certify its group's focal and distortion: starting 5% off the
    // true focal, the group is held at its input intrinsics and only the poses and points are
    // re-solved, so the residual stays small but nonzero.
    let problem = distorted_seed_pair(
        brown_conrady(756.0, RadialDistortion::K1(-0.12)),
        SELF_CALIBRATE,
    );
    let solution = solve_stress_fixture(&problem);
    assert_eq!(solution.report().unconstrained_intrinsics(), [0]);
    assert_eq!(
        solution.intrinsics()[0],
        *problem.intrinsics()[0].intrinsics()
    );
    assert!(solution.report().final_rmse() < solution.report().initial_rmse());
}

#[test]
fn a_waived_two_view_seed_group_moves_without_certification() {
    // The same pair as the held-group test above, starting 5% off the true focal: with its
    // certification waived the group follows the observations toward the truth instead of keeping
    // its input values, and the residual falls further than the held solve reaches.
    let held_problem = distorted_seed_pair(
        brown_conrady(756.0, RadialDistortion::K1(-0.12)),
        SELF_CALIBRATE,
    );
    let held = solve_stress_fixture(&held_problem);
    let waived_problem = BundleProblem::builder(
        held_problem.poses().to_vec(),
        held_problem
            .intrinsics()
            .iter()
            .map(|block| {
                block
                    .clone()
                    .with_certification(crate::IntrinsicCertification::Waived)
            })
            .collect(),
        held_problem.points().to_vec(),
        held_problem.observations().to_vec(),
    )
    .with_gauge(held_problem.gauge())
    .build()
    .unwrap();

    let waived = solve_stress_fixture(&waived_problem);

    assert!(waived.report().unconstrained_intrinsics().is_empty());
    assert_ne!(
        waived.intrinsics()[0],
        *waived_problem.intrinsics()[0].intrinsics()
    );
    let Camera::BrownConrady(refined) = waived.intrinsics()[0].camera() else {
        panic!("the group must stay Brown-Conrady");
    };
    assert!(
        (refined.focal_x() - 720.0).abs() < (756.0_f64 - 720.0).abs(),
        "focal {} did not move toward 720",
        refined.focal_x()
    );
    assert!(waived.report().final_rmse() < held.report().final_rmse());
}

#[test]
fn focal_only_refinement_keeps_declared_distortion_bit_exactly() {
    let truth = [
        Camera::try_pinhole(600.0, 600.0, 320.0, 240.0).unwrap(),
        brown_conrady(720.0, RadialDistortion::K1(-0.12)),
        brown_conrady(650.0, RadialDistortion::K1K2(0.08, -0.03)),
    ];
    let initial = [
        truth[0].clone(),
        truth[1].clone(),
        brown_conrady(676.0, RadialDistortion::K1K2(0.08, -0.03)),
    ];
    let focal_only = IntrinsicVariables::default().with_focal(true);
    let problem = mixed_model_fixture(
        &truth,
        &initial,
        [
            IntrinsicVariables::FIXED,
            IntrinsicVariables::FIXED,
            focal_only,
        ],
    );
    let solution = solve_stress_fixture(&problem);
    assert!(solution.report().final_rmse() < 1e-5);
    let Camera::BrownConrady(refined) = solution.intrinsics()[2].camera() else {
        panic!("a K1K2 group must stay Brown-Conrady");
    };
    assert_eq!(
        *refined.radial_distortion(),
        RadialDistortion::K1K2(0.08, -0.03)
    );
    assert_relative(refined.focal_x(), 650.0, 1e-6);
}

#[test]
fn an_angular_solve_reports_observability_details_from_its_own_rank_assessment() {
    let camera = Camera::Spherical(openmvg_core::camera::SphericalCamera::try_new(1024).unwrap());
    let centers = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.1, 0.0),
        Point3::new(0.4, 0.9, 0.2),
    ];
    let truth_points = [
        Point3::new(-2.0, -0.5, 4.0),
        Point3::new(2.5, 0.3, -3.5),
        Point3::new(0.4, 2.2, 5.0),
        Point3::new(-1.1, -2.0, -4.2),
        Point3::new(3.0, 1.0, 1.5),
        Point3::new(-3.2, 0.8, -1.0),
    ];
    let poses: Vec<_> = centers
        .iter()
        .map(|center| {
            WorldToCameraPose::from_rotation_center(Rotation3::identity(), *center).unwrap()
        })
        .collect();
    let observations: Vec<_> = poses
        .iter()
        .enumerate()
        .flat_map(|(view, pose)| {
            let camera = camera.clone();
            truth_points.iter().enumerate().map(move |(point, world)| {
                let xy =
                    CameraModel::project(&camera, &pose.project_to_camera(world).coords).unwrap();
                BundleObservation::try_new(view, 0, point, xy).unwrap()
            })
        })
        .collect();
    let problem = BundleProblem::builder(
        poses
            .iter()
            .enumerate()
            .map(|(index, pose)| {
                BundlePose::new(
                    pose.clone(),
                    if index == 0 {
                        PoseVariables::FIXED
                    } else {
                        PoseVariables::FIXED.with_translation(true)
                    },
                )
            })
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(camera, 1024, 512)
                    .unwrap()
                    .with_fixed(true),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        truth_points
            .iter()
            .map(|point| BundlePoint::try_new(*point + Vector3::new(0.02, -0.01, 0.015)).unwrap())
            .collect(),
        observations,
    )
    .with_residual_space(crate::ResidualSpace::Angular)
    .with_gauge(Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance: 1.0,
    })
    .build()
    .unwrap();

    let solution = NativeBundleAdjuster::new()
        .solve(&problem, SolveConfig::default())
        .unwrap();

    let details = solution
        .report()
        .diagnostics()
        .observability_details()
        .expect("an angular solve has no probe, so its details come from the rank assessment");
    assert_eq!(details.pose_rank(), details.pose_dimension());
    assert_eq!(details.weak_point_count(), 0);
}

/// Eight cameras on an arc with fixed rotations, one fixed `camera` shared by all, and pixels
/// with seeded half-pixel noise, so the optimum has a nonzero cost; with `outliers`, every 23rd
/// observation is moved by about 40 px. `fixed_first` holds camera 0 entirely.
fn fixed_rotation_fixture(
    camera: &Camera,
    pixel_camera: &Camera,
    gauge: impl Fn(&[WorldToCameraPose]) -> Gauge,
    robust_loss: crate::RobustLoss,
    outliers: bool,
    fixed_first: bool,
) -> BundleProblem {
    fixed_rotation_fixture_observing(
        camera,
        pixel_camera,
        gauge,
        robust_loss,
        outliers,
        fixed_first,
        |_, _| true,
    )
}

/// [`fixed_rotation_fixture`] keeping only the observations `observes(pose, point)` accepts.
fn fixed_rotation_fixture_observing(
    camera: &Camera,
    pixel_camera: &Camera,
    gauge: impl Fn(&[WorldToCameraPose]) -> Gauge,
    robust_loss: crate::RobustLoss,
    outliers: bool,
    fixed_first: bool,
    observes: impl Fn(usize, usize) -> bool,
) -> BundleProblem {
    let truth = (0..8)
        .map(|index| {
            let angle = -0.42 + 0.12 * index as f64;
            let rotation = Rotation3::from_euler_angles(
                0.04 * ((index % 3) as f64 - 1.0),
                angle,
                0.03 * ((index % 2) as f64 - 0.5),
            );
            let center = Point3::new(
                5.0 * angle.sin(),
                0.15 * ((index % 3) as f64 - 1.0),
                5.0 - 5.0 * angle.cos(),
            );
            WorldToCameraPose::from_rotation_center(rotation, center).unwrap()
        })
        .collect::<Vec<_>>();
    let points = (0..64)
        .map(|index| {
            let jitter = index as f64;
            Point3::new(
                -2.1 + 0.6 * (index % 8) as f64 + 0.05 * jitter.sin(),
                -1.6 + 0.45 * (index / 8) as f64 + 0.04 * jitter.cos(),
                4.6 + 0.4 * ((index * 5) % 7) as f64,
            )
        })
        .collect::<Vec<_>>();
    let noise = |seed: usize| {
        let value = (seed as f64 * 12.9898 + 78.233).sin() * 43_758.545_3;
        value - value.floor() - 0.5
    };
    let mut observations = Vec::new();
    for (pose, truth_pose) in truth.iter().enumerate() {
        for (point, position) in points.iter().enumerate() {
            if !observes(pose, point) {
                continue;
            }
            let projected = truth_pose.project_to_camera(position);
            let mut pixel = CameraModel::project(pixel_camera, &projected.coords).unwrap();
            let key = pose * points.len() + point;
            pixel += Vector2::new(noise(2 * key), noise(2 * key + 1));
            if outliers && key % 23 == 0 {
                pixel += Vector2::new(30.0, -25.0);
            }
            observations.push(BundleObservation::try_new(pose, 0, point, pixel).unwrap());
        }
    }
    let initial = truth
        .iter()
        .enumerate()
        .map(|(index, pose)| {
            let offset = Vector3::new(0.03, -0.02, 0.025) * (index as f64 / 8.0);
            WorldToCameraPose::from_rotation_center(*pose.rotation(), pose.center() + offset)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let translation_only = PoseVariables::FIXED.with_translation(true);
    BundleProblem::builder(
        initial
            .iter()
            .enumerate()
            .map(|(index, pose)| {
                let variables = if fixed_first && index == 0 {
                    PoseVariables::FIXED
                } else {
                    translation_only
                };
                BundlePose::new(pose.clone(), variables)
            })
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(camera.clone(), 640, 480).unwrap(),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        points
            .iter()
            .map(|point| BundlePoint::try_new(point + Vector3::new(0.02, -0.015, 0.03)).unwrap())
            .collect(),
        observations,
    )
    .with_gauge(gauge(&initial))
    .with_robust_loss(robust_loss)
    .build()
    .unwrap()
}

/// The preflight certificate of distorted translation-only problems, which SfM's global cleanup
/// requires. These values are the ones the retired dense probe returned on the same problems.
#[test]
fn sparse_preflight_certifies_distorted_translation_only_problems() {
    let brown = Camera::BrownConrady(
        BrownConradyCamera::try_with_focal(
            700.0,
            320.0,
            240.0,
            RadialDistortion::K1K2K3(-0.055, 0.008, 0.018),
            openmvg_core::camera::TangentialDistortion::P1P2(0.0012, -0.0008),
        )
        .unwrap(),
    );
    // An EXIF camera group starts with a zero K1 coefficient, which still counts as distorted.
    let exif_start = brown_conrady(700.0, RadialDistortion::K1(0.0));
    let huber = crate::RobustLoss::Huber(crate::ResidualScale::Pixels(
        openmvg_core::PixelThreshold::FOUR_PIXELS,
    ));
    let free = |_: &[WorldToCameraPose]| Gauge::Free;
    let longer = |poses: &[WorldToCameraPose]| Gauge::FixedBaseline {
        first: 0,
        second: 1,
        distance: 1.2 * (poses[1].center() - poses[0].center()).norm(),
    };
    let squared = crate::RobustLoss::Squared;
    let mut problems = Vec::new();
    for camera in [&brown, &exif_start] {
        problems.push(fixed_rotation_fixture(
            camera, &brown, free, huber, true, false,
        ));
        problems.push(fixed_rotation_fixture(
            camera, &brown, free, squared, false, false,
        ));
        problems.push(fixed_rotation_fixture(
            camera, &brown, longer, squared, false, false,
        ));
        problems.push(fixed_rotation_fixture(
            camera, &brown, free, huber, false, true,
        ));
    }
    let observing = |observes: fn(usize, usize) -> bool,
                     gauge: &dyn Fn(&[WorldToCameraPose]) -> Gauge| {
        fixed_rotation_fixture_observing(&brown, &brown, gauge, huber, false, false, observes)
    };
    // Camera 7 sees five points, shares its points with camera 6 alone, sees one point, or
    // sees two; points 0..8 are seen by cameras 0 and 1 only.
    problems.push(observing(|pose, point| pose != 7 || point < 5, &free));
    problems.push(observing(two_view_link, &free));
    problems.push(observing(two_view_link, &longer));
    problems.push(observing(|pose, point| point >= 8 || pose <= 1, &free));
    problems.push(observing(|pose, point| pose != 7 || point == 20, &free));
    problems.push(observing(
        |pose, point| pose != 7 || point == 20 || point == 41,
        &free,
    ));
    problems.push(observing(
        |pose, point| pose != 7 || point == 20 || point == 41,
        &longer,
    ));
    let mut escalated = 0;
    for problem in &problems {
        let preflight = NativeBundleAdjuster::new()
            .preflight(problem, &SolveConfig::default())
            .unwrap();
        assert_eq!(preflight.status, NativePreflightStatus::Proceed);
        assert_eq!(preflight.pose_dimension, Some(20));
        assert_eq!(preflight.covered_free_pose_count, Some(7));
        assert_eq!(preflight.weak_point_count, Some(0));
        escalated += usize::from(preflight.dense_observability_escalation);
    }
    assert_eq!(escalated, 3, "the two-view and one-point links escalate");
}

fn two_view_link(pose: usize, point: usize) -> bool {
    if point < 16 {
        pose == 6 || pose == 7
    } else {
        pose != 7
    }
}

/// A camera whose tracks it shares with one neighbour only has a free distance to it: the
/// translation-only rank check must refuse it, and certify the same scene with full tracks.
#[test]
fn translation_only_rank_check_refuses_a_two_view_link() {
    let camera = brown_conrady(700.0, RadialDistortion::K1(-0.05));
    let free = |_: &[WorldToCameraPose]| Gauge::Free;
    let squared = crate::RobustLoss::Squared;
    let config = SolveConfig::default();
    let full = fixed_rotation_fixture(&camera, &camera, free, squared, false, false);
    let solution = NativeBundleAdjuster::new().solve(&full, config).unwrap();
    assert_eq!(
        solution.report().diagnostics().observability(),
        Observability::FullRank
    );
    solution.audit_for_commit(&full, &config).unwrap();

    let linked = fixed_rotation_fixture_observing(
        &camera,
        &camera,
        free,
        squared,
        false,
        false,
        two_view_link,
    );
    let solution = NativeBundleAdjuster::new().solve(&linked, config).unwrap();
    assert_eq!(
        solution.report().diagnostics().observability(),
        Observability::RankDeficient
    );
    assert_eq!(
        solution.audit_for_commit(&linked, &config),
        Err(AuditError::RankDeficient)
    );
}

#[test]
fn a_round_off_baseline_change_does_not_refuse_a_solve_that_holds_out_points() {
    let base = translated_non_identity_fixture(
        Vector3::zeros(),
        std::array::from_fn(|index| index),
        0,
        false,
    );
    // A far point without parallax is held out before the solve.
    let camera = Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap();
    let far = Point3::new(0.0, 0.0, 1.0e9);
    let mut points = base.points().to_vec();
    points.push(BundlePoint::try_new(far).unwrap());
    let mut observations = base.observations().to_vec();
    for camera_index in [0, 1] {
        let pixel = CameraModel::project(
            &camera,
            &base.poses()[camera_index]
                .pose()
                .project_to_camera(&far)
                .coords,
        )
        .unwrap();
        observations
            .push(BundleObservation::try_new(camera_index, 0, points.len() - 1, pixel).unwrap());
    }
    let current = (base.poses()[1].pose().center() - base.poses()[0].pose().center()).norm();
    let with_baseline = |distance: f64| {
        BundleProblem::builder(
            base.poses().to_vec(),
            base.intrinsics().to_vec(),
            points.clone(),
            observations.clone(),
        )
        .with_gauge(Gauge::FixedBaseline {
            first: 0,
            second: 1,
            distance,
        })
        .build()
        .unwrap()
    };

    // The round-off a previous solve leaves in the baseline is not a physical scale change.
    let rounded = with_baseline(current * (1.0 + 1.0e-13));
    assert_ne!(gauge_plan(&rounded).unwrap().scale, 1.0);
    let solution = NativeBundleAdjuster::new()
        .solve(&rounded, SolveConfig::default())
        .expect("a round-off baseline change must not refuse the solve");
    assert_eq!(solution.quarantined_points(), [points.len() - 1]);

    // A real change of the physical scale would leave the held-out point behind, so it still is.
    assert!(matches!(
        NativeBundleAdjuster::new().solve(&with_baseline(current * 1.01), SolveConfig::default()),
        Err(SolveError::Unsupported(_))
    ));
}
