// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Point2, Point3, Rotation3};
use openmvg_ba::{
    BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem, ConvergenceReason,
    Gauge, IntrinsicVariables, Observability, PoseVariables, ProblemError, ResidualScale,
    ResidualSpace, RobustLoss, Solution, SolutionError, SolveConfig, SolveDiagnostics, SolveError,
    SolveReport, SolveStatus, StationarityAudit,
};
use openmvg_core::{
    AngularThreshold, PixelThreshold,
    camera::{Camera, CameraModel},
    model::{CameraIntrinsics, WorldToCameraPose},
};

fn intrinsic(fixed: bool) -> CameraIntrinsics {
    let data = CameraIntrinsics::try_new(
        Camera::try_pinhole(800.0, 800.0, 320.0, 240.0).unwrap(),
        640,
        480,
    )
    .unwrap();
    data.with_fixed(fixed)
}

const BASELINE: Gauge = Gauge::FixedBaseline {
    first: 0,
    second: 1,
    distance: 1.0,
};

fn pixel_huber() -> RobustLoss {
    RobustLoss::Huber(ResidualScale::Pixels(PixelThreshold::try_new(2.0).unwrap()))
}

fn problem() -> BundleProblem {
    BundleProblem::builder(
        vec![
            BundlePose::new(WorldToCameraPose::identity(), PoseVariables::FIXED),
            BundlePose::new(
                WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
                PoseVariables::FREE,
            ),
        ],
        vec![BundleIntrinsic::try_new(intrinsic(true), IntrinsicVariables::FIXED).unwrap()],
        vec![
            BundlePoint::try_new(Point3::new(0.0, 0.0, 4.0))
                .unwrap()
                .with_fixed(true),
        ],
        vec![
            BundleObservation::try_new(0, 0, 0, Point2::new(320.0, 240.0)).unwrap(),
            BundleObservation::try_new(1, 0, 0, Point2::new(120.0, 240.0)).unwrap(),
        ],
    )
    .with_robust_loss(pixel_huber())
    .with_gauge(BASELINE)
    .build()
    .unwrap()
}

#[test]
fn problem_owns_one_checked_indexed_representation() {
    let problem = problem();
    assert_eq!(problem.poses().len(), 2);
    assert_eq!(problem.intrinsics().len(), 1);
    assert_eq!(problem.points().len(), 1);
    assert!(problem.points()[0].is_fixed());
    assert_eq!(problem.observations().len(), 2);
    assert_eq!(problem.residual_space(), ResidualSpace::Pixels);
    assert_eq!(problem.robust_loss(), pixel_huber());
    assert_eq!(problem.gauge(), BASELINE);
}

#[test]
fn builder_defaults_and_validates_in_build_regardless_of_order() {
    let problem = problem();
    let builder = || {
        BundleProblem::builder(
            problem.poses().to_vec(),
            problem.intrinsics().to_vec(),
            problem.points().to_vec(),
            problem.observations().to_vec(),
        )
    };
    let defaults = builder().build().unwrap();
    assert_eq!(defaults.residual_space(), ResidualSpace::Pixels);
    assert_eq!(defaults.robust_loss(), RobustLoss::Squared);
    assert_eq!(defaults.gauge(), Gauge::Free);

    assert_eq!(
        builder()
            .with_gauge(BASELINE)
            .with_robust_loss(pixel_huber())
            .build(),
        Ok(problem.clone())
    );
    assert_eq!(
        builder()
            .with_robust_loss(pixel_huber())
            .with_residual_space(ResidualSpace::Angular)
            .build(),
        Err(ProblemError::LossUnitMismatch)
    );
    assert_eq!(
        builder()
            .with_residual_space(ResidualSpace::Angular)
            .with_robust_loss(pixel_huber())
            .build(),
        Err(ProblemError::LossUnitMismatch)
    );
    let angular_huber = RobustLoss::Huber(ResidualScale::Angular(
        AngularThreshold::try_new(0.01).unwrap(),
    ));
    assert!(
        builder()
            .with_robust_loss(angular_huber)
            .with_residual_space(ResidualSpace::Angular)
            .build()
            .is_ok()
    );
    assert_eq!(
        builder()
            .with_gauge(Gauge::FixedBaseline {
                first: 1,
                second: 1,
                distance: 1.0,
            })
            .build(),
        Err(ProblemError::GaugeUsesOnePose)
    );
}

#[test]
fn variable_masks_start_from_presets() {
    assert!(PoseVariables::FREE.rotation() && PoseVariables::FREE.translation());
    assert!(PoseVariables::FIXED.is_fixed());
    let translation = PoseVariables::FIXED.with_translation(true);
    assert!(!translation.rotation() && translation.translation());
    assert_eq!(
        PoseVariables::FREE
            .with_rotation(false)
            .with_translation(false),
        PoseVariables::FIXED
    );

    assert_eq!(IntrinsicVariables::default(), IntrinsicVariables::FIXED);
    let self_calibrating = IntrinsicVariables::default()
        .with_focal(true)
        .with_radial(true);
    assert!(self_calibrating.focal() && self_calibrating.radial());
    assert!(!self_calibrating.principal_point() && !self_calibrating.tangential());
    assert!(
        IntrinsicVariables::default()
            .with_principal_point(true)
            .with_tangential(true)
            .with_principal_point(false)
            .tangential()
    );

    let point = BundlePoint::try_new(Point3::new(0.0, 0.0, 4.0)).unwrap();
    assert!(!point.is_fixed());
    assert!(point.with_fixed(true).is_fixed());
    assert_eq!(point.with_fixed(true).with_fixed(false), point);
    assert_eq!(
        BundlePoint::try_new(Point3::new(f64::NAN, 0.0, 4.0)),
        Err(ProblemError::NonFinitePoint)
    );
}

#[test]
fn problem_retains_repeated_rows_and_rejects_bad_indices_and_unit_mismatch() {
    let cameras = vec![BundlePose::new(
        WorldToCameraPose::identity(),
        PoseVariables::FREE,
    )];
    let intrinsics =
        vec![BundleIntrinsic::try_new(intrinsic(true), IntrinsicVariables::FIXED).unwrap()];
    let points = vec![BundlePoint::try_new(Point3::new(0.0, 0.0, 4.0)).unwrap()];
    let observation = BundleObservation::try_new(0, 0, 0, Point2::new(320.0, 240.0)).unwrap();

    let repeated = BundleProblem::builder(
        cameras.clone(),
        intrinsics.clone(),
        points.clone(),
        vec![observation, observation],
    )
    .build()
    .unwrap();
    assert_eq!(repeated.observations(), &[observation, observation]);
    for (pose, intrinsic, point, expected) in [
        (
            1,
            0,
            0,
            ProblemError::ObservationPoseOutOfBounds {
                observation: 0,
                pose: 1,
                pose_count: 1,
            },
        ),
        (
            0,
            1,
            0,
            ProblemError::IntrinsicIndexOutOfBounds {
                observation: 0,
                intrinsic: 1,
                intrinsic_count: 1,
            },
        ),
        (
            0,
            0,
            1,
            ProblemError::ObservationPointOutOfBounds {
                observation: 0,
                point: 1,
                point_count: 1,
            },
        ),
    ] {
        let row = BundleObservation::try_new(pose, intrinsic, point, *observation.pixel()).unwrap();
        assert_eq!(
            BundleProblem::builder(
                cameras.clone(),
                intrinsics.clone(),
                points.clone(),
                vec![row],
            )
            .build(),
            Err(expected)
        );
    }
    assert_eq!(
        BundleProblem::builder(cameras, intrinsics, points, vec![observation])
            .with_residual_space(ResidualSpace::Angular)
            .with_robust_loss(RobustLoss::Huber(ResidualScale::Pixels(
                PixelThreshold::try_new(1.0).unwrap()
            )))
            .build(),
        Err(ProblemError::LossUnitMismatch)
    );
}

#[test]
fn solution_is_owned_and_preserves_fixed_blocks() {
    let problem = problem();
    let report = SolveReport::try_new(
        SolveStatus::Converged(ConvergenceReason::Objective),
        2,
        4.0,
        1.0,
        2.0,
        1.0,
        SolveDiagnostics::new(
            1,
            1,
            StationarityAudit::try_from_fresh_zero_damping(1.0e-8).unwrap(),
            Observability::FullRank,
            0,
            0,
        ),
    )
    .unwrap();
    let solution = Solution::try_new(
        &problem,
        problem
            .poses()
            .iter()
            .map(|camera| camera.pose().clone())
            .collect(),
        problem
            .intrinsics()
            .iter()
            .map(|intrinsic| intrinsic.intrinsics().clone())
            .collect(),
        problem
            .points()
            .iter()
            .map(|point| *point.position())
            .collect(),
        report,
    )
    .unwrap();
    assert_eq!(solution.poses().len(), 2);
    assert_eq!(solution.residual_space(), ResidualSpace::Pixels);
    solution
        .audit_for_commit(&problem, &SolveConfig::default())
        .unwrap();

    let mut changed_points = solution.points().to_vec();
    changed_points[0].x += 1.0;
    assert_eq!(
        Solution::try_new(
            &problem,
            solution.poses().to_vec(),
            solution.intrinsics().to_vec(),
            changed_points,
            solution.report().clone(),
        ),
        Err(SolutionError::FixedPointChanged { point: 0 })
    );
}

#[test]
fn control_and_report_reject_invalid_numerical_policy() {
    assert_eq!(SolveConfig::default().validate(), Ok(()));
    assert_eq!(
        SolveConfig::default().with_max_iterations(0).validate(),
        Err(SolveError::InvalidConfig("max_iterations must be positive"))
    );
    for (config, name) in [
        (
            SolveConfig::default().with_function_tolerance(-1.0),
            "function_tolerance",
        ),
        (
            SolveConfig::default().with_gradient_tolerance(f64::NAN),
            "gradient_tolerance",
        ),
        (
            SolveConfig::default().with_parameter_tolerance(0.0),
            "parameter_tolerance",
        ),
    ] {
        assert_eq!(config.validate(), Err(SolveError::InvalidConfig(name)));
    }
    assert_eq!(
        SolveConfig::default()
            .with_workspace_budget_bytes(0)
            .validate(),
        Err(SolveError::InvalidConfig(
            "workspace_budget_bytes must be positive"
        ))
    );
    assert_eq!(SolveConfig::default().workspace_budget_bytes(), None);
    assert_eq!(
        SolveConfig::default()
            .with_workspace_budget_bytes(1)
            .validate(),
        Ok(())
    );
    assert_eq!(
        SolveReport::try_new(
            SolveStatus::Converged(ConvergenceReason::Objective),
            1,
            1.0,
            0.5,
            1.0,
            0.5,
            SolveDiagnostics::new(
                2,
                0,
                StationarityAudit::try_from_fresh_zero_damping(1.0e-8).unwrap(),
                Observability::FullRank,
                0,
                0,
            ),
        ),
        Err(SolutionError::InvalidReport)
    );
    assert!(StationarityAudit::try_from_fresh_zero_damping(f64::INFINITY).is_err());
}

#[test]
fn weak_points_flags_unobservable_and_unprojectable_points() {
    let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
    let poses = vec![
        WorldToCameraPose::identity(),
        WorldToCameraPose::from_rotation_center(
            nalgebra::Rotation3::identity(),
            Point3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
    ];
    // Well triangulated, effectively at infinity (no parallax), and on the first camera centre.
    let points = vec![
        Point3::new(0.5, 0.0, 4.0),
        Point3::new(0.0, 0.0, 1.0e9),
        Point3::new(0.0, 0.0, 0.0),
    ];
    let pixel = |pose: &WorldToCameraPose, point: &Point3<f64>| {
        camera
            .project(&pose.project_to_camera(point).coords)
            .unwrap_or(Point2::new(320.0, 240.0))
    };
    let observations = (0..points.len())
        .flat_map(|point| {
            poses
                .iter()
                .enumerate()
                .map(move |(camera, _)| (camera, point))
        })
        .map(|(camera, point)| {
            BundleObservation::try_new(camera, 0, point, pixel(&poses[camera], &points[point]))
                .unwrap()
        })
        .collect();
    let problem = BundleProblem::builder(
        poses
            .iter()
            .map(|pose| BundlePose::new(pose.clone(), PoseVariables::FIXED))
            .collect(),
        vec![BundleIntrinsic::try_new(intrinsic(true), IntrinsicVariables::FIXED).unwrap()],
        points
            .iter()
            .map(|point| BundlePoint::try_new(*point).unwrap())
            .collect(),
        observations,
    )
    .build()
    .unwrap();
    let report = SolveReport::try_new(
        SolveStatus::Converged(ConvergenceReason::Objective),
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        SolveDiagnostics::new(
            1,
            0,
            StationarityAudit::try_from_fresh_zero_damping(0.0).unwrap(),
            Observability::FullRank,
            0,
            0,
        ),
    )
    .unwrap();
    let solution =
        Solution::try_new(&problem, poses, vec![intrinsic(true)], points, report).unwrap();

    assert_eq!(solution.weak_points(&problem), vec![1, 2]);
}
