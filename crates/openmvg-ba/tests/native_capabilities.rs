// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small, independently projected scenes for the supported native BA contract.
//!
//! Expected measurements use the Brown/pinhole equations below, rather than the
//! camera implementation used by the solver. Recovery checks use known world
//! geometry; a decreasing reported objective alone does not satisfy these tests.

use nalgebra::{Matrix2x3, Matrix3, Point2, Point3, Rotation3, Vector3};
use openmvg_ba::{
    BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
    Gauge, IntrinsicVariables, LinearSolver, NativeBundleAdjuster, Observability, PoseVariables,
    ResidualScale, ResidualSpace, RobustLoss, Solution, SolutionError, SolveConfig, SolveError,
};
use openmvg_core::{
    camera::{BrownConradyCamera, Camera, PinholeCamera, RadialDistortion, TangentialDistortion},
    model::{CameraIntrinsics, WorldToCameraPose},
};

const WORLD_TOLERANCE: f64 = 2.0e-5;
const ROTATION_TOLERANCE: f64 = 5.0e-6;

#[derive(Clone, Copy)]
struct Calibration {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    p1: f64,
    p2: f64,
}

impl Calibration {
    const PINHOLE: Self = Self {
        fx: 620.0,
        fy: 710.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
    };

    const BROWN: Self = Self {
        k1: -0.055,
        k2: 0.008,
        k3: 0.018,
        p1: 0.0012,
        p2: -0.0008,
        ..Self::PINHOLE
    };

    fn camera(self, brown: bool) -> Camera {
        let pinhole = PinholeCamera::try_new(self.fx, self.fy, self.cx, self.cy).unwrap();
        if brown {
            Camera::BrownConrady(
                BrownConradyCamera::try_new(
                    pinhole,
                    RadialDistortion::K1K2K3(self.k1, self.k2, self.k3),
                    TangentialDistortion::P1P2(self.p1, self.p2),
                )
                .unwrap(),
            )
        } else {
            Camera::Pinhole(pinhole)
        }
    }

    // Independent normalized-coordinate Brown projection in image pixels.
    fn project(self, pose: &WorldToCameraPose, point: &Point3<f64>) -> Point2<f64> {
        let camera_point = pose.rotation().matrix() * (point - pose.center());
        assert!(
            camera_point.z > 0.0,
            "fixture point must be in front of the camera"
        );
        let x = camera_point.x / camera_point.z;
        let y = camera_point.y / camera_point.z;
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let distorted_x = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let distorted_y = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        Point2::new(
            self.fx * distorted_x + self.cx,
            self.fy * distorted_y + self.cy,
        )
    }
}

fn truth_poses() -> Vec<WorldToCameraPose> {
    [
        (Point3::new(0.35, -0.20, 0.15), (0.03, -0.05, 0.02)),
        (Point3::new(1.45, 0.12, 0.32), (-0.02, 0.07, -0.03)),
        (Point3::new(-0.50, 0.90, -0.08), (0.04, 0.02, 0.05)),
    ]
    .into_iter()
    .map(|(center, (rx, ry, rz))| {
        WorldToCameraPose::from_rotation_center(Rotation3::from_euler_angles(rx, ry, rz), center)
            .unwrap()
    })
    .collect()
}

fn truth_points() -> Vec<Point3<f64>> {
    (0..12)
        .map(|index| {
            let i = index as f64;
            Point3::new(
                1.30 * (0.83 * i + 0.2).sin(),
                0.85 * (1.17 * i - 0.3).cos(),
                4.4 + 0.12 * i + 0.35 * (0.51 * i).sin(),
            )
        })
        .collect()
}

fn perturb_pose(
    truth: &WorldToCameraPose,
    variables: PoseVariables,
    index: usize,
) -> WorldToCameraPose {
    let i = index as f64 + 1.0;
    let rotation = if variables.rotation() {
        Rotation3::from_euler_angles(0.009 * i, -0.007 * i, 0.006 * i) * truth.rotation()
    } else {
        *truth.rotation()
    };
    let center = if variables.translation() {
        truth.center() + Vector3::new(0.035 * i, -0.022 * i, 0.018 * i)
    } else {
        *truth.center()
    };
    WorldToCameraPose::from_rotation_center(rotation, center).unwrap()
}

fn perturb_points(points: &[Point3<f64>]) -> Vec<Point3<f64>> {
    points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let i = index as f64;
            point + Vector3::new(0.028 * (i + 0.4).sin(), -0.022 * (i + 0.7).cos(), 0.035)
        })
        .collect()
}

fn observations(
    calibration: Calibration,
    poses: &[WorldToCameraPose],
    points: &[Point3<f64>],
) -> Vec<BundleObservation> {
    points
        .iter()
        .enumerate()
        .flat_map(|(point_id, point)| {
            poses.iter().enumerate().map(move |(pose_id, pose)| {
                BundleObservation::try_new(pose_id, 0, point_id, calibration.project(pose, point))
                    .unwrap()
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn problem(
    calibration: Calibration,
    brown: bool,
    poses: Vec<WorldToCameraPose>,
    masks: &[PoseVariables],
    points: Vec<Point3<f64>>,
    fixed_points: bool,
    observations: Vec<BundleObservation>,
    gauge: Gauge,
) -> BundleProblem {
    assert_eq!(poses.len(), masks.len());
    BundleProblem::builder(
        poses
            .into_iter()
            .zip(masks.iter().copied())
            .map(|(pose, mask)| BundlePose::new(pose, mask))
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(calibration.camera(brown), 640, 480)
                    .unwrap()
                    .with_fixed(true),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        points
            .into_iter()
            .map(|point| {
                BundlePoint::try_new(point)
                    .unwrap()
                    .with_fixed(fixed_points)
            })
            .collect(),
        observations,
    )
    .with_robust_loss(RobustLoss::Squared)
    .with_gauge(gauge)
    .build()
    .unwrap()
}

fn solve_config(linear_solver: LinearSolver) -> SolveConfig {
    SolveConfig::default()
        .with_max_iterations(100)
        .with_function_tolerance(1.0e-12)
        .with_gradient_tolerance(1.0e-11)
        .with_parameter_tolerance(1.0e-11)
        .with_linear_solver(linear_solver)
}

fn solve(problem: &BundleProblem, calibration: Calibration) -> Solution {
    solve_with_linear_solver(problem, calibration, LinearSolver::Direct)
}

fn solve_with_linear_solver(
    problem: &BundleProblem,
    calibration: Calibration,
    linear_solver: LinearSolver,
) -> Solution {
    let config = solve_config(linear_solver);
    let solution = NativeBundleAdjuster::new()
        .solve(problem, config)
        .unwrap_or_else(|error| panic!("valid native capability fixture was refused: {error}"));
    solution.audit_for_commit(problem, &config).unwrap();
    assert_eq!(
        solution.intrinsics()[0],
        *problem.intrinsics()[0].intrinsics()
    );
    assert!(solution.quarantined_points().is_empty());
    assert_eq!(
        solution.report().observation_count(),
        Some(problem.observations().len())
    );

    let squared: f64 = problem
        .observations()
        .iter()
        .map(|observation| {
            let pose = &solution.poses()[observation.pose()];
            let point = &solution.points()[observation.point()];
            match problem.residual_space() {
                ResidualSpace::Pixels => {
                    (calibration.project(pose, point) - observation.pixel()).norm_squared()
                }
                ResidualSpace::Angular => {
                    // Independent pinhole bearing equation for the angular fixture.
                    assert_eq!(
                        [
                            calibration.k1,
                            calibration.k2,
                            calibration.k3,
                            calibration.p1,
                            calibration.p2
                        ],
                        [0.0; 5],
                    );
                    let measured = Vector3::new(
                        (observation.pixel().x - calibration.cx) / calibration.fx,
                        (observation.pixel().y - calibration.cy) / calibration.fy,
                        1.0,
                    )
                    .normalize();
                    let predicted =
                        (pose.rotation().matrix() * (point - pose.center())).normalize();
                    let angle = measured
                        .cross(&predicted)
                        .norm()
                        .atan2(measured.dot(&predicted));
                    angle * angle
                }
                _ => panic!("this fixture needs an independently defined residual domain"),
            }
        })
        .sum();
    let rmse = (squared / problem.observations().len() as f64).sqrt();
    let tolerance = match problem.residual_space() {
        ResidualSpace::Angular => 2.0e-7,
        _ => 2.0e-5,
    };
    assert!(rmse < tolerance, "independent residual RMSE is {rmse}");
    assert!(
        (solution.report().final_rmse() - rmse).abs() < 1.0e-10,
        "reported RMSE does not describe the independent residuals"
    );
    assert!(
        (solution.report().final_objective() - 0.5 * squared).abs() < 1.0e-9,
        "reported objective does not describe the independent squared loss"
    );
    solution
}

fn assert_pose_recovered(actual: &WorldToCameraPose, expected: &WorldToCameraPose) {
    let center_error = (actual.center() - expected.center()).norm();
    let rotation_error = (actual.rotation().matrix() - expected.rotation().matrix()).norm();
    assert!(
        center_error < WORLD_TOLERANCE,
        "center error is {center_error}"
    );
    assert!(
        rotation_error < ROTATION_TOLERANCE,
        "rotation matrix error is {rotation_error}"
    );
}

fn assert_points_recovered(actual: &[Point3<f64>], expected: &[Point3<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).norm();
        assert!(error < WORLD_TOLERANCE, "point {index} error is {error}");
    }
}

fn assert_rotation_bits_equal(actual: &WorldToCameraPose, expected: &WorldToCameraPose) {
    for (actual, expected) in actual
        .rotation()
        .matrix()
        .iter()
        .zip(expected.rotation().matrix().iter())
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "a held rotation changed"
        );
    }
}

fn assert_center_bits_equal(actual: &WorldToCameraPose, expected: &WorldToCameraPose) {
    for (actual, expected) in actual
        .center()
        .coords
        .iter()
        .zip(expected.center().coords.iter())
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "a held center changed"
        );
    }
}

fn controls_problem(
    calibration: Calibration,
    brown: bool,
    masks: &[PoseVariables],
) -> BundleProblem {
    let mut poses = truth_poses();
    poses.truncate(masks.len());
    let points = truth_points()[..8].to_vec();
    let observations = observations(calibration, &poses, &points);
    let initial = poses
        .iter()
        .zip(masks)
        .enumerate()
        .map(|(index, (pose, &mask))| perturb_pose(pose, mask, index))
        .collect();
    problem(
        calibration,
        brown,
        initial,
        masks,
        points,
        true,
        observations,
        Gauge::Free,
    )
}

#[test]
fn fixed_brown_pixels_recover_metric_geometry_without_changing_calibration() {
    let truth = truth_poses();
    let points = truth_points();
    let masks = [
        PoseVariables::FIXED,
        PoseVariables::FREE,
        PoseVariables::FREE,
    ];
    let mut initial: Vec<_> = truth
        .iter()
        .enumerate()
        .map(|(index, pose)| perturb_pose(pose, masks[index], index))
        .collect();
    // Keep the already feasible metric baseline; this test requests no redesign
    // of baseline changes combined with fixed state.
    initial[1] =
        WorldToCameraPose::from_rotation_center(*initial[1].rotation(), *truth[1].center())
            .unwrap();
    let distance = (truth[1].center() - truth[0].center()).norm();
    let input = problem(
        Calibration::BROWN,
        true,
        initial,
        &masks,
        perturb_points(&points),
        false,
        observations(Calibration::BROWN, &truth, &points),
        Gauge::FixedBaseline {
            first: 0,
            second: 1,
            distance,
        },
    );
    let output = solve(&input, Calibration::BROWN);
    for (actual, expected) in output.poses().iter().zip(&truth) {
        assert_pose_recovered(actual, expected);
    }
    assert_points_recovered(output.points(), &points);
    assert_rotation_bits_equal(&output.poses()[0], input.poses()[0].pose());
    assert_center_bits_equal(&output.poses()[0], input.poses()[0].pose());
    let returned_distance = (output.poses()[1].center() - output.poses()[0].center()).norm();
    assert!((returned_distance - distance).abs() < 1.0e-10);
}

#[test]
fn fixed_world_controls_optimize_every_free_pose() {
    let masks = [PoseVariables::FREE; 3];
    let input = controls_problem(Calibration::PINHOLE, false, &masks);
    let output = solve(&input, Calibration::PINHOLE);
    for (actual, expected) in output.poses().iter().zip(truth_poses()) {
        assert_pose_recovered(actual, &expected);
    }
    assert_eq!(output.points(), &truth_points()[..8]);
    assert!((output.poses()[0].center() - input.poses()[0].pose().center()).norm() > 0.02);
    assert!(
        (output.poses()[0].rotation().matrix() - input.poses()[0].pose().rotation().matrix())
            .norm()
            > 0.005,
        "world controls must not make camera zero an implicit fixed rotation"
    );
}

#[test]
fn fixed_world_controls_allow_single_view_and_unused_inventory() {
    let mut truth = truth_poses();
    truth.truncate(2);
    let points = truth_points();
    let measured: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(point, position)| {
            let view = point / 6;
            BundleObservation::try_new(
                view,
                0,
                point,
                Calibration::PINHOLE.project(&truth[view], position),
            )
            .unwrap()
        })
        .collect();
    let masks = [PoseVariables::FREE; 2];
    let initial = truth
        .iter()
        .enumerate()
        .map(|(index, pose)| perturb_pose(pose, masks[index], index))
        .collect::<Vec<_>>();
    let plain = problem(
        Calibration::PINHOLE,
        false,
        initial.clone(),
        &masks,
        points.clone(),
        true,
        measured.clone(),
        Gauge::Free,
    );
    let inventory = Point3::new(3.0, -2.0, 8.0);
    let mut with_inventory = points;
    with_inventory.push(inventory);
    let input = problem(
        Calibration::PINHOLE,
        false,
        initial,
        &masks,
        with_inventory,
        true,
        measured,
        Gauge::Free,
    );
    let plain_output = solve(&plain, Calibration::PINHOLE);
    let output = solve(&input, Calibration::PINHOLE);
    for ((actual, without_inventory), expected) in
        output.poses().iter().zip(plain_output.poses()).zip(&truth)
    {
        assert_pose_recovered(actual, expected);
        assert!((actual.center() - without_inventory.center()).norm() < 1.0e-8);
        assert!(
            (actual.rotation().matrix() - without_inventory.rotation().matrix()).norm() < 1.0e-8
        );
    }
    assert_eq!(output.points().last(), Some(&inventory));
    assert_eq!(
        output.report().observation_count(),
        plain_output.report().observation_count()
    );
    assert_eq!(
        output.report().diagnostics().observability(),
        plain_output.report().diagnostics().observability()
    );
}

#[test]
fn mixed_pose_masks_preserve_rotations_and_camera_centers() {
    let masks = [
        PoseVariables::FREE,
        PoseVariables::FIXED.with_translation(true),
        PoseVariables::FIXED.with_rotation(true),
    ];
    let input = controls_problem(Calibration::PINHOLE, false, &masks);
    let output = solve(&input, Calibration::PINHOLE);
    for (actual, expected) in output.poses().iter().zip(truth_poses()) {
        assert_pose_recovered(actual, &expected);
    }
    assert_eq!(output.points(), &truth_points()[..8]);
    assert_rotation_bits_equal(&output.poses()[1], input.poses()[1].pose());
    assert_center_bits_equal(&output.poses()[2], input.poses()[2].pose());
    assert!(
        (output.poses()[2].translation() - input.poses()[2].pose().translation()).norm() > 0.005,
        "rotation-only refinement holds C, so t = -R*C must be allowed to change"
    );
}

#[test]
fn two_camera_fixed_controls_use_general_native_formulation() {
    let masks = [PoseVariables::FREE; 2];
    let input = controls_problem(Calibration::PINHOLE, false, &masks);
    let output = solve(&input, Calibration::PINHOLE);
    for (actual, expected) in output.poses().iter().zip(truth_poses().into_iter().take(2)) {
        assert_pose_recovered(actual, &expected);
    }
    assert_eq!(output.points(), &truth_points()[..8]);
}

#[test]
fn two_camera_points_only_recover_known_world_geometry() {
    let mut truth = truth_poses();
    truth.truncate(2);
    let points = truth_points();
    let masks = [PoseVariables::FIXED; 2];
    let input = problem(
        Calibration::PINHOLE,
        false,
        truth.clone(),
        &masks,
        perturb_points(&points),
        false,
        observations(Calibration::PINHOLE, &truth, &points),
        Gauge::Free,
    );
    let output = solve(&input, Calibration::PINHOLE);
    assert_points_recovered(output.points(), &points);
    for (actual, expected) in output.poses().iter().zip(&truth) {
        assert_rotation_bits_equal(actual, expected);
        assert_center_bits_equal(actual, expected);
    }
}

#[test]
fn fixed_brown_controls_with_known_rotations_keep_world_frame() {
    let masks = [PoseVariables::FIXED.with_translation(true); 3];
    let input = controls_problem(Calibration::BROWN, true, &masks);
    let output = solve(&input, Calibration::BROWN);
    for (index, (actual, expected)) in output.poses().iter().zip(truth_poses()).enumerate() {
        assert_pose_recovered(actual, &expected);
        assert_rotation_bits_equal(actual, input.poses()[index].pose());
    }
    assert_eq!(output.points(), &truth_points()[..8]);
}

#[test]
fn known_rotation_does_not_lock_another_free_camera_rotation() {
    let truth = truth_poses();
    let points = truth_points();
    let masks = [
        PoseVariables::FREE,
        PoseVariables::FIXED.with_translation(true),
        PoseVariables::FREE,
    ];
    let initial = truth
        .iter()
        .enumerate()
        .map(|(index, pose)| perturb_pose(pose, masks[index], index))
        .collect();
    let input = problem(
        Calibration::PINHOLE,
        false,
        initial,
        &masks,
        perturb_points(&points),
        false,
        observations(Calibration::PINHOLE, &truth, &points),
        Gauge::Free,
    );
    let output = solve(&input, Calibration::PINHOLE);
    assert_rotation_bits_equal(&output.poses()[1], input.poses()[1].pose());
    for (actual, expected) in output.poses().iter().zip(&truth) {
        let error = (actual.rotation().matrix() - expected.rotation().matrix()).norm();
        assert!(
            error < ROTATION_TOLERANCE,
            "observable world rotation error is {error}"
        );
    }
    // The known rotation removes rotational gauge. Only positive scale and
    // translation are removed when comparing the reconstructed world geometry.
    let estimated_mean = output
        .poses()
        .iter()
        .map(|pose| pose.center().coords)
        .sum::<Vector3<f64>>()
        / 3.0;
    let truth_mean = truth
        .iter()
        .map(|pose| pose.center().coords)
        .sum::<Vector3<f64>>()
        / 3.0;
    let numerator: f64 = output
        .poses()
        .iter()
        .zip(&truth)
        .map(|(actual, expected)| {
            (actual.center().coords - estimated_mean).dot(&(expected.center().coords - truth_mean))
        })
        .sum();
    let denominator: f64 = output
        .poses()
        .iter()
        .map(|pose| (pose.center().coords - estimated_mean).norm_squared())
        .sum();
    let scale = numerator / denominator;
    assert!(scale.is_finite() && scale > 0.0);
    let translation = truth_mean - scale * estimated_mean;
    for (actual, expected) in output.poses().iter().zip(&truth) {
        let error =
            (scale * actual.center().coords + translation - expected.center().coords).norm();
        assert!(
            error < WORLD_TOLERANCE,
            "center error after scale/translation is {error}"
        );
    }
    for (actual, expected) in output.points().iter().zip(&points) {
        let error = (scale * actual.coords + translation - expected.coords).norm();
        assert!(
            error < WORLD_TOLERANCE,
            "point error after scale/translation is {error}"
        );
    }
}

fn assert_scale_translation_geometry(
    output: &Solution,
    truth: &[WorldToCameraPose],
    points: &[Point3<f64>],
) {
    assert_eq!(output.poses().len(), truth.len());
    let count = truth.len() as f64;
    let estimated_mean = output
        .poses()
        .iter()
        .map(|pose| pose.center().coords)
        .sum::<Vector3<f64>>()
        / count;
    let truth_mean = truth
        .iter()
        .map(|pose| pose.center().coords)
        .sum::<Vector3<f64>>()
        / count;
    let numerator: f64 = output
        .poses()
        .iter()
        .zip(truth)
        .map(|(actual, expected)| {
            (actual.center().coords - estimated_mean).dot(&(expected.center().coords - truth_mean))
        })
        .sum();
    let denominator: f64 = output
        .poses()
        .iter()
        .map(|pose| (pose.center().coords - estimated_mean).norm_squared())
        .sum();
    let scale = numerator / denominator;
    assert!(scale.is_finite() && scale > 0.0);
    let translation = truth_mean - scale * estimated_mean;
    for (actual, expected) in output.poses().iter().zip(truth) {
        assert!(
            (scale * actual.center().coords + translation - expected.center().coords).norm()
                < WORLD_TOLERANCE
        );
    }
    for (actual, expected) in output.points().iter().zip(points) {
        assert!((scale * actual.coords + translation - expected.coords).norm() < WORLD_TOLERANCE);
    }
}

#[test]
fn angular_controls_support_mixed_pose_masks() {
    let masks = [
        PoseVariables::FREE,
        PoseVariables::FIXED.with_translation(true),
        PoseVariables::FIXED.with_rotation(true),
    ];
    let pixels = controls_problem(Calibration::PINHOLE, false, &masks);
    let input = BundleProblem::builder(
        pixels.poses().to_vec(),
        pixels.intrinsics().to_vec(),
        pixels.points().to_vec(),
        pixels.observations().to_vec(),
    )
    .with_residual_space(ResidualSpace::Angular)
    .with_robust_loss(RobustLoss::Squared)
    .with_gauge(Gauge::Free)
    .build()
    .unwrap();
    let output = solve(&input, Calibration::PINHOLE);
    for (actual, expected) in output.poses().iter().zip(truth_poses()) {
        assert_pose_recovered(actual, &expected);
    }
    assert_eq!(output.points(), &truth_points()[..8]);
    assert_rotation_bits_equal(&output.poses()[1], input.poses()[1].pose());
    assert_center_bits_equal(&output.poses()[2], input.poses()[2].pose());
}

#[test]
fn insufficient_world_controls_do_not_certify_hidden_anchor() {
    let truth = truth_poses();
    let points = truth_points();
    let masks = [PoseVariables::FREE; 3];
    let initial = truth
        .iter()
        .enumerate()
        .map(|(index, pose)| perturb_pose(pose, masks[index], index))
        .collect();
    let mut initial_points = perturb_points(&points);
    initial_points[0] = points[0];
    let unconstrained = problem(
        Calibration::PINHOLE,
        false,
        initial,
        &masks,
        initial_points,
        false,
        observations(Calibration::PINHOLE, &truth, &points),
        Gauge::Free,
    );
    let fixed = unconstrained
        .points()
        .iter()
        .enumerate()
        .map(|(index, point)| {
            BundlePoint::try_new(*point.position())
                .unwrap()
                .with_fixed(index == 0)
        })
        .collect();
    let input = BundleProblem::builder(
        unconstrained.poses().to_vec(),
        unconstrained.intrinsics().to_vec(),
        fixed,
        unconstrained.observations().to_vec(),
    )
    .with_robust_loss(RobustLoss::Squared)
    .with_gauge(Gauge::Free)
    .build()
    .unwrap();
    let config = solve_config(LinearSolver::Direct);
    // One fixed point leaves four continuous similarity directions. A hidden
    // full-pose anchor could instead remove observable axes and fake a certificate.
    match NativeBundleAdjuster::new().solve(&input, config) {
        Ok(output) => {
            assert_eq!(
                output.report().diagnostics().observability(),
                Observability::RankDeficient
            );
            assert!(output.audit_for_commit(&input, &config).is_err());
            assert_eq!(output.points()[0], points[0]);
        }
        Err(SolveError::Numerical(_)) => {}
        Err(error) => panic!("insufficient controls need a numerical/rank refusal, got {error}"),
    }
}

#[test]
fn inactive_survey_inventory_preserves_extreme_values_and_active_solution() {
    let masks = [PoseVariables::FREE; 3];
    let plain = controls_problem(Calibration::PINHOLE, false, &masks);
    let inventory_pose = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.12, -0.18, 0.09),
        Point3::new(1.0e300, -1.0e300, 1.0e300),
    )
    .unwrap();
    let mut poses = plain.poses().to_vec();
    // Inventory is explicitly fixed. Unobserved free pose axes must still
    // prevent a full-rank certificate and are covered by the library regression.
    poses.push(BundlePose::new(
        inventory_pose.clone(),
        PoseVariables::FIXED,
    ));
    let inventory_points = [
        Point3::new(1.0e300, -1.0e300, 1.0e300),
        Point3::new(-1.0e300, 1.0e300, -1.0e300),
    ];
    let mut points = plain.points().to_vec();
    points.extend(
        inventory_points
            .into_iter()
            .map(|point| BundlePoint::try_new(point).unwrap().with_fixed(true)),
    );
    let input = BundleProblem::builder(
        poses,
        plain.intrinsics().to_vec(),
        points,
        plain.observations().to_vec(),
    )
    .with_robust_loss(RobustLoss::Squared)
    .with_gauge(Gauge::Free)
    .build()
    .unwrap();
    let reference = solve(&plain, Calibration::PINHOLE);
    let output = solve(&input, Calibration::PINHOLE);
    for (actual, expected) in output.poses()[..3].iter().zip(reference.poses()) {
        assert!((actual.center() - expected.center()).norm() < 1.0e-8);
        assert!((actual.rotation().matrix() - expected.rotation().matrix()).norm() < 1.0e-8);
    }
    assert_rotation_bits_equal(&output.poses()[3], &inventory_pose);
    assert_center_bits_equal(&output.poses()[3], &inventory_pose);
    for (actual, expected) in output.points()[8..].iter().zip(inventory_points) {
        for (actual, expected) in actual.coords.iter().zip(expected.coords.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(
        output.report().observation_count(),
        reference.report().observation_count()
    );
    assert_eq!(
        output.report().diagnostics().observability(),
        reference.report().diagnostics().observability()
    );
}

#[test]
fn unused_variable_intrinsics_do_not_disable_two_view() {
    let mut truth = truth_poses();
    truth.truncate(2);
    let points = truth_points();
    let masks = [PoseVariables::FIXED, PoseVariables::FREE];
    let initial = truth
        .iter()
        .enumerate()
        .map(|(index, pose)| perturb_pose(pose, masks[index], index))
        .collect();
    let plain = problem(
        Calibration::PINHOLE,
        false,
        initial,
        &masks,
        perturb_points(&points),
        false,
        observations(Calibration::PINHOLE, &truth, &points),
        Gauge::Free,
    );
    let unused = CameraIntrinsics::try_new(
        Camera::try_pinhole(1120.0, 970.0, 111.0, 222.0).unwrap(),
        640,
        480,
    )
    .unwrap();
    let mut intrinsics = plain.intrinsics().to_vec();
    intrinsics.push(
        BundleIntrinsic::try_new(unused.clone(), IntrinsicVariables::FIXED.with_focal(true))
            .unwrap(),
    );
    let input = BundleProblem::builder(
        plain.poses().to_vec(),
        intrinsics,
        plain.points().to_vec(),
        plain.observations().to_vec(),
    )
    .with_robust_loss(RobustLoss::Squared)
    .with_gauge(Gauge::Free)
    .build()
    .unwrap();
    let reference = solve(&plain, Calibration::PINHOLE);
    let output = solve(&input, Calibration::PINHOLE);
    assert_eq!(output.intrinsics()[1], unused);
    assert_eq!(output.report().unconstrained_intrinsics(), &[1]);
    assert_rotation_bits_equal(&output.poses()[0], plain.poses()[0].pose());
    assert_center_bits_equal(&output.poses()[0], plain.poses()[0].pose());
    for (actual, expected) in output.poses().iter().zip(&truth) {
        assert!(
            (actual.rotation().matrix() - expected.rotation().matrix()).norm() < ROTATION_TOLERANCE
        );
    }
    assert_scale_translation_geometry(&output, &truth, &points);
    for (actual, expected) in output.poses().iter().zip(reference.poses()) {
        assert!((actual.center() - expected.center()).norm() < 1.0e-8);
        assert!((actual.rotation().matrix() - expected.rotation().matrix()).norm() < 1.0e-8);
    }
}

#[test]
fn mixed_controls_direct_and_iterative_agree_on_geometry() {
    let masks = [
        PoseVariables::FREE,
        PoseVariables::FIXED.with_translation(true),
        PoseVariables::FIXED.with_rotation(true),
    ];
    let input = controls_problem(Calibration::PINHOLE, false, &masks);
    let direct = solve_with_linear_solver(&input, Calibration::PINHOLE, LinearSolver::Direct);
    let iterative = solve_with_linear_solver(&input, Calibration::PINHOLE, LinearSolver::Iterative);
    for ((actual, reference), expected) in iterative
        .poses()
        .iter()
        .zip(direct.poses())
        .zip(truth_poses())
    {
        assert_pose_recovered(actual, &expected);
        assert!((actual.center() - reference.center()).norm() < 1.0e-6);
        assert!((actual.rotation().matrix() - reference.rotation().matrix()).norm() < 1.0e-6);
    }
    assert_eq!(iterative.points(), direct.points());
    assert_rotation_bits_equal(&iterative.poses()[1], input.poses()[1].pose());
    assert_center_bits_equal(&iterative.poses()[2], input.poses()[2].pose());
}

#[test]
fn single_camera_world_controls_recover_the_free_pose() {
    let truth = vec![truth_poses()[0].clone()];
    let points = truth_points()[..6].to_vec();
    let masks = [PoseVariables::FREE];
    let initial = vec![perturb_pose(&truth[0], masks[0], 0)];
    let input = problem(
        Calibration::PINHOLE,
        false,
        initial,
        &masks,
        points.clone(),
        true,
        observations(Calibration::PINHOLE, &truth, &points),
        Gauge::Free,
    );
    let output = solve(&input, Calibration::PINHOLE);
    assert_eq!(output.poses().len(), 1);
    assert_pose_recovered(&output.poses()[0], &truth[0]);
    assert_eq!(output.points(), points.as_slice());
    assert_eq!(
        output.report().diagnostics().observability(),
        Observability::FullRank
    );
    assert!((output.poses()[0].center() - input.poses()[0].pose().center()).norm() > 0.02);
    assert!(
        (output.poses()[0].rotation().matrix() - input.poses()[0].pose().rotation().matrix())
            .norm()
            > 0.005,
        "the sole free pose must not become an implicit fixed camera zero"
    );
}

#[test]
fn two_fixed_poses_triangulate_one_free_point_with_rank_three() {
    let mut truth = truth_poses();
    truth.truncate(2);
    let points = vec![truth_points()[4]];
    let masks = [PoseVariables::FIXED; 2];

    // Independent point information: d(pixel)/d(X_world) = d(pixel)/d(X_cam) * R.
    // Two separated fixed cameras give this point three well-observed coordinates.
    let mut information = Matrix3::zeros();
    for pose in &truth {
        let q = pose.rotation().matrix() * (points[0] - pose.center());
        let inverse_z = 1.0 / q.z;
        let camera_jacobian = Matrix2x3::new(
            Calibration::PINHOLE.fx * inverse_z,
            0.0,
            -Calibration::PINHOLE.fx * q.x * inverse_z * inverse_z,
            0.0,
            Calibration::PINHOLE.fy * inverse_z,
            -Calibration::PINHOLE.fy * q.y * inverse_z * inverse_z,
        );
        let world_jacobian = camera_jacobian * pose.rotation().matrix();
        information += world_jacobian.transpose() * world_jacobian;
    }
    let eigenvalues = information.symmetric_eigenvalues();
    assert!(
        eigenvalues.min() > 1.0e-4 * eigenvalues.max(),
        "fixture point must have rank-three information"
    );

    let input = problem(
        Calibration::PINHOLE,
        false,
        truth.clone(),
        &masks,
        perturb_points(&points),
        false,
        observations(Calibration::PINHOLE, &truth, &points),
        Gauge::Free,
    );
    let output = solve(&input, Calibration::PINHOLE);
    assert_eq!(output.points().len(), 1);
    assert_points_recovered(output.points(), &points);
    assert!((output.points()[0] - input.points()[0].position()).norm() > 0.01);
    for (actual, expected) in output.poses().iter().zip(&truth) {
        assert_rotation_bits_equal(actual, expected);
        assert_center_bits_equal(actual, expected);
    }
}

#[test]
fn shared_physical_poses_use_each_observations_fixed_lens_and_keep_distinct_pose_identity() {
    let calibrations = [
        Calibration::PINHOLE,
        Calibration {
            fx: 740.0,
            fy: 680.0,
            cx: 300.0,
            cy: 250.0,
            ..Calibration::BROWN
        },
    ];
    let truth = [
        Point3::origin(),
        Point3::new(1.0, 0.1, 0.2),
        Point3::new(-0.4, 0.6, -0.1),
    ]
    .map(|center| WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap());
    let same_initial =
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(0.3, 0.05, 0.1))
            .unwrap();
    let initial = [truth[0].clone(), same_initial.clone(), same_initial];
    assert_eq!(
        initial[1], initial[2],
        "equal values still name two physical pose blocks"
    );
    let points = truth_points()[..8].to_vec();
    let observations = points
        .iter()
        .enumerate()
        .flat_map(|(point, position)| {
            truth.iter().enumerate().flat_map(move |(pose, physical)| {
                calibrations
                    .into_iter()
                    .enumerate()
                    .map(move |(intrinsic, calibration)| {
                        BundleObservation::try_new(
                            pose,
                            intrinsic,
                            point,
                            calibration.project(physical, position),
                        )
                        .unwrap()
                    })
            })
        })
        .collect::<Vec<_>>();

    // The first lens alone resolves all three center coordinates of each
    // pose: J_C=-d(pixel)/d(X_cam) for these identity rotations and fixed controls.
    for physical in &truth {
        let mut information = Matrix3::zeros();
        for point in &points {
            let q = point - physical.center();
            let z_inverse = q.z.recip();
            let j = Matrix2x3::new(
                calibrations[0].fx * z_inverse,
                0.0,
                -calibrations[0].fx * q.x * z_inverse * z_inverse,
                0.0,
                calibrations[0].fy * z_inverse,
                -calibrations[0].fy * q.y * z_inverse * z_inverse,
            );
            information += j.transpose() * j;
        }
        let spectrum = information.symmetric_eigenvalues();
        assert!(
            spectrum.min() > 1e-4 * spectrum.max(),
            "fixed controls must resolve each center"
        );
    }
    let translation = PoseVariables::FIXED.with_translation(true);
    let input = BundleProblem::builder(
        initial
            .iter()
            .enumerate()
            .map(|(pose, physical)| {
                BundlePose::new(
                    physical.clone(),
                    if pose == 0 {
                        PoseVariables::FIXED
                    } else {
                        translation
                    },
                )
            })
            .collect(),
        calibrations
            .into_iter()
            .enumerate()
            .map(|(group, calibration)| {
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(calibration.camera(group == 1), 640, 480)
                        .unwrap()
                        .with_fixed(true),
                    IntrinsicVariables::FIXED,
                )
                .unwrap()
            })
            .collect(),
        points
            .iter()
            .map(|position| BundlePoint::try_new(*position).unwrap().with_fixed(true))
            .collect(),
        observations.clone(),
    )
    .build()
    .unwrap();
    assert_eq!(input.observations(), observations.as_slice());
    let config = solve_config(LinearSolver::Direct);
    let output = NativeBundleAdjuster::new().solve(&input, config).unwrap();
    output.audit_for_commit(&input, &config).unwrap();
    assert_eq!(output.poses().len(), 3);
    assert_eq!(output.intrinsics().len(), 2);
    assert_eq!(output.points(), points.as_slice());
    assert!(output.quarantined_points().is_empty());
    assert_eq!(
        output.report().diagnostics().observability(),
        Observability::FullRank
    );
    assert_eq!(
        output.report().observation_count(),
        Some(observations.len())
    );
    for (pose, expected) in truth.iter().enumerate() {
        assert_pose_recovered(&output.poses()[pose], expected);
        assert_rotation_bits_equal(&output.poses()[pose], input.poses()[pose].pose());
    }
    assert!((output.poses()[1].center() - output.poses()[2].center()).norm() > 1.0);
    for group in 0..2 {
        assert_eq!(
            output.intrinsics()[group],
            *input.intrinsics()[group].intrinsics()
        );
    }
    let squared = |poses: &[WorldToCameraPose]| {
        observations
            .iter()
            .map(|row| {
                (calibrations[row.intrinsic()].project(&poses[row.pose()], &points[row.point()])
                    - row.pixel())
                .norm_squared()
            })
            .sum::<f64>()
    };
    let final_squared = squared(output.poses());
    assert!((final_squared / observations.len() as f64).sqrt() < 2e-5);
    assert!((output.report().initial_objective() - 0.5 * squared(&initial)).abs() < 1e-8);
    assert!((output.report().final_objective() - 0.5 * final_squared).abs() < 1e-9);
    assert!(
        (output.report().final_rmse() - (final_squared / observations.len() as f64).sqrt()).abs()
            < 1e-10
    );
}

#[test]
fn repeated_measurements_contribute_to_the_native_least_squares_optimum() {
    let calibration = Calibration::PINHOLE;
    let points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.0),
        Point3::new(-1.0, 1.0, 4.0),
        Point3::new(1.0, 1.0, 4.0),
    ];
    let mut rows = points
        .iter()
        .enumerate()
        .map(|(point, position)| {
            BundleObservation::try_new(
                0,
                0,
                point,
                calibration.project(&WorldToCameraPose::identity(), position),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let repeated =
        BundleObservation::try_new(0, 0, 0, *rows[0].pixel() + nalgebra::Vector2::new(6.0, 0.0))
            .unwrap();
    rows.extend([repeated, repeated]);
    assert_eq!(rows[4], rows[5]);

    // For controls on Z=4 and fixed R=I, projection is linear in
    // alpha=fx/(4-Cz), beta=-alpha*Cx, gamma=-(fy/fx)*alpha*Cy.
    // The independently assembled 3x3 LS problem has a unique positive-alpha
    // optimum, and every repeated row must enter that sum independently.
    let oracle = |observations: &[BundleObservation]| {
        let mut normal = Matrix3::zeros();
        let mut right = Vector3::zeros();
        let aspect = calibration.fy / calibration.fx;
        for row in observations {
            let point = points[row.point()];
            let u = Vector3::new(point.x, 1.0, 0.0);
            let v = Vector3::new(aspect * point.y, 0.0, 1.0);
            normal += u * u.transpose() + v * v.transpose();
            right += u * (row.pixel().x - calibration.cx) + v * (row.pixel().y - calibration.cy);
        }
        let chart = normal.lu().solve(&right).unwrap();
        assert!(chart[0] > 0.0);
        Point3::new(
            -chart[1] / chart[0],
            -chart[2] / (aspect * chart[0]),
            4.0 - calibration.fx / chart[0],
        )
    };
    let expected = oracle(&rows);
    assert!(
        (expected - oracle(&rows[..5])).norm() > 1e-3,
        "one repeated factor must change the fitted center"
    );
    let initial = WorldToCameraPose::from_rotation_center(
        Rotation3::identity(),
        Point3::new(0.05, -0.03, 0.1),
    )
    .unwrap();
    let input = problem(
        calibration,
        false,
        vec![initial.clone()],
        &[PoseVariables::FIXED.with_translation(true)],
        points.to_vec(),
        true,
        rows.clone(),
        Gauge::Free,
    );
    assert_eq!(input.observations(), rows.as_slice());
    let config = solve_config(LinearSolver::Direct);
    let output = NativeBundleAdjuster::new().solve(&input, config).unwrap();
    output.audit_for_commit(&input, &config).unwrap();
    assert!((output.poses()[0].center() - expected).norm() < WORLD_TOLERANCE);
    assert_rotation_bits_equal(&output.poses()[0], &initial);
    assert_eq!(output.points(), points.as_slice());
    assert_eq!(output.report().observation_count(), Some(rows.len()));
    let squared = |pose: &WorldToCameraPose| {
        rows.iter()
            .map(|row| {
                (calibration.project(pose, &points[row.point()]) - row.pixel()).norm_squared()
            })
            .sum::<f64>()
    };
    assert!((output.report().initial_objective() - 0.5 * squared(&initial)).abs() < 1e-8);
    let final_squared = squared(&output.poses()[0]);
    assert!((output.report().final_objective() - 0.5 * final_squared).abs() < 1e-8);
    assert!(
        (output.report().final_rmse() - (final_squared / rows.len() as f64).sqrt()).abs() < 1e-10
    );
}

#[test]
fn inlier_error_leaves_out_observations_beyond_the_robust_scale() {
    let truth = truth_poses();
    let points = truth_points();
    let mut observed = observations(Calibration::PINHOLE, &truth, &points);
    // One wrong match, 40 px off its point; three views of that point cannot absorb it.
    let wrong = &observed[0];
    let shifted = wrong.pixel() + nalgebra::Vector2::new(0.0, 40.0);
    observed[0] =
        BundleObservation::try_new(wrong.pose(), wrong.intrinsic(), wrong.point(), shifted)
            .unwrap();
    let total = observed.len();
    let cauchy = RobustLoss::Cauchy(ResidualScale::Pixels(
        openmvg_core::PixelThreshold::try_new(1.0).unwrap(),
    ));
    let input = BundleProblem::builder(
        truth
            .iter()
            .map(|pose| BundlePose::new(pose.clone(), PoseVariables::FIXED))
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(Calibration::PINHOLE.camera(false), 640, 480)
                    .unwrap()
                    .with_fixed(true),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        perturb_points(&points)
            .into_iter()
            .map(|point| BundlePoint::try_new(point).unwrap())
            .collect(),
        observed,
    )
    .with_robust_loss(cauchy)
    .build()
    .unwrap();

    let output = NativeBundleAdjuster::new()
        .solve(&input, solve_config(LinearSolver::Direct))
        .unwrap();
    let report = output.report();
    assert_eq!(report.observation_count(), Some(total));
    assert_eq!(report.inlier_count(), Some(total - 1));
    let inlier_rmse = report.final_inlier_rmse().unwrap();
    assert!(
        inlier_rmse < 1.0e-2,
        "the good observations fit: {inlier_rmse} px"
    );
    assert!(
        report.final_rmse() > 1.0,
        "the plain RMSE still counts the wrong match"
    );

    // The per-observation residuals name the wrong match: it alone stays beyond the scale.
    let residuals = output.observation_residuals(&input).unwrap();
    assert_eq!(residuals.len(), total);
    let beyond: Vec<_> = residuals
        .iter()
        .enumerate()
        .filter(|(_, residual)| residual.unwrap() > 1.0)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(beyond, [0]);
    assert!(residuals[0].unwrap() > 30.0, "{:?}", residuals[0]);
}

#[test]
fn the_report_names_the_linear_solver_that_ran() {
    let truth = truth_poses();
    let points = truth_points();
    let problem = BundleProblem::builder(
        truth
            .iter()
            .map(|pose| BundlePose::new(pose.clone(), PoseVariables::FIXED))
            .collect(),
        vec![
            BundleIntrinsic::try_new(
                CameraIntrinsics::try_new(Calibration::PINHOLE.camera(false), 640, 480)
                    .unwrap()
                    .with_fixed(true),
                IntrinsicVariables::FIXED,
            )
            .unwrap(),
        ],
        perturb_points(&points)
            .into_iter()
            .map(|point| BundlePoint::try_new(point).unwrap())
            .collect(),
        observations(Calibration::PINHOLE, &truth, &points),
    )
    .build()
    .unwrap();
    // Auto resolves to the exact factorization while it fits the default budget.
    for (requested, ran) in [
        (LinearSolver::Direct, LinearSolver::Direct),
        (LinearSolver::Iterative, LinearSolver::Iterative),
        (LinearSolver::Auto, LinearSolver::Direct),
    ] {
        let output = NativeBundleAdjuster::new()
            .solve(&problem, solve_config(requested))
            .unwrap();
        assert_eq!(
            output.report().diagnostics().linear_solver(),
            Some(ran),
            "{requested:?}"
        );
    }
}

#[test]
fn observation_residuals_need_the_problem_the_solution_answers() {
    let truth = truth_poses();
    let points = truth_points();
    let problem = |space| {
        BundleProblem::builder(
            truth
                .iter()
                .map(|pose| BundlePose::new(pose.clone(), PoseVariables::FIXED))
                .collect(),
            vec![
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(Calibration::PINHOLE.camera(false), 640, 480)
                        .unwrap()
                        .with_fixed(true),
                    IntrinsicVariables::FIXED,
                )
                .unwrap(),
            ],
            perturb_points(&points)
                .into_iter()
                .map(|point| BundlePoint::try_new(point).unwrap())
                .collect(),
            observations(Calibration::PINHOLE, &truth, &points),
        )
        .with_residual_space(space)
        .build()
        .unwrap()
    };
    let pixels = problem(ResidualSpace::Pixels);
    let output = NativeBundleAdjuster::new()
        .solve(&pixels, solve_config(LinearSolver::Direct))
        .unwrap();
    let residuals = output.observation_residuals(&pixels).unwrap();
    assert!(residuals.iter().all(|residual| residual.unwrap() < 1.0e-3));
    assert_eq!(
        output.observation_residuals(&problem(ResidualSpace::Angular)),
        Err(SolutionError::ResidualSpaceMismatch)
    );
}
