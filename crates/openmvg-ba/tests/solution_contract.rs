// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{Point2, Point3, Rotation3};
use openmvg_ba::{
    AuditError, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
    ConvergenceReason, IntrinsicVariables, Observability, PoseVariables, Solution, SolutionError,
    SolveConfig, SolveDiagnostics, SolveReport, SolveStatus, StationarityAudit,
};
use openmvg_core::{
    camera::Camera,
    model::{CameraIntrinsics, WorldToCameraPose},
};

fn problem() -> BundleProblem {
    let intrinsic = CameraIntrinsics::try_new(
        Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
        64,
        48,
    )
    .unwrap();
    BundleProblem::builder(
        vec![BundlePose::new(
            WorldToCameraPose::identity(),
            PoseVariables::FIXED.with_translation(true),
        )],
        vec![BundleIntrinsic::try_new(intrinsic, IntrinsicVariables::FIXED).unwrap()],
        vec![BundlePoint::try_new(Point3::new(0.0, 0.0, 4.0)).unwrap()],
        vec![BundleObservation::try_new(0, 0, 0, Point2::new(32.0, 24.0)).unwrap()],
    )
    .build()
    .unwrap()
}

fn report() -> SolveReport {
    report_with(SolveStatus::Converged(ConvergenceReason::Step))
}

fn report_with(status: SolveStatus) -> SolveReport {
    SolveReport::try_new(
        status,
        1,
        1.0,
        0.5,
        1.0,
        0.5,
        SolveDiagnostics::new(
            1,
            0,
            StationarityAudit::try_from_fresh_zero_damping(0.0).unwrap(),
            Observability::OnlyExpectedGauge,
            0,
            0,
        ),
    )
    .unwrap()
}

#[test]
fn translation_only_solution_rejects_rotation_changes() {
    let problem = problem();
    let changed = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.0, 0.01, 0.0),
        *problem.poses()[0].pose().center(),
    )
    .unwrap();

    assert_eq!(
        Solution::try_new(
            &problem,
            vec![changed],
            vec![problem.intrinsics()[0].intrinsics().clone()],
            vec![*problem.points()[0].position()],
            report(),
        ),
        Err(SolutionError::FixedPoseChanged { pose: 0 })
    );
}

#[test]
fn translation_only_solution_accepts_translation_changes() {
    let problem = problem();
    let changed = WorldToCameraPose::from_rotation_center(
        *problem.poses()[0].pose().rotation(),
        Point3::new(0.1, 0.0, 0.0),
    )
    .unwrap();

    Solution::try_new(
        &problem,
        vec![changed],
        vec![problem.intrinsics()[0].intrinsics().clone()],
        vec![*problem.points()[0].position()],
        report(),
    )
    .unwrap();
}

#[test]
fn audit_does_not_certify_gradient_convergence_against_an_invalid_control() {
    let problem = problem();
    let solution = Solution::try_new(
        &problem,
        vec![problem.poses()[0].pose().clone()],
        vec![problem.intrinsics()[0].intrinsics().clone()],
        vec![*problem.points()[0].position()],
        report_with(SolveStatus::Converged(ConvergenceReason::Gradient)),
    )
    .unwrap();

    assert_eq!(
        solution.audit_for_commit(&problem, &SolveConfig::default()),
        Ok(())
    );
    assert_eq!(
        solution.audit_for_commit(
            &problem,
            &SolveConfig::default().with_gradient_tolerance(f64::NAN)
        ),
        Err(AuditError::StationarityNotCertified)
    );
}

#[cfg(feature = "faer")]
mod fixed_contract_regressions {
    use nalgebra::{Point2, Point3, Rotation3, Vector3};
    use openmvg_ba::{
        BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
        ConvergenceReason, Gauge, IntrinsicVariables, NativeBundleAdjuster, Observability,
        PoseVariables, Solution, SolveConfig, SolveDiagnostics, SolveReport, SolveStatus,
        StationarityAudit,
    };
    use openmvg_core::{
        camera::{Camera, CameraModel},
        model::{CameraIntrinsics, WorldToCameraPose},
    };

    fn evidence_report() -> SolveReport {
        SolveReport::try_new(
            SolveStatus::Converged(ConvergenceReason::Step),
            1,
            1.,
            0.5,
            1.,
            0.5,
            SolveDiagnostics::new(
                1,
                0,
                StationarityAudit::try_from_fresh_zero_damping(0.).unwrap(),
                Observability::FullRank,
                0,
                0,
            ),
        )
        .unwrap()
    }

    #[test]
    fn focal_only_refinement_must_reject_changed_principal_point() {
        let initial =
            CameraIntrinsics::try_new(Camera::try_pinhole(100., 100., 32., 24.).unwrap(), 64, 48)
                .unwrap();
        let problem = BundleProblem::builder(
            vec![BundlePose::new(
                WorldToCameraPose::identity(),
                PoseVariables::FIXED,
            )],
            vec![
                BundleIntrinsic::try_new(initial, IntrinsicVariables::FIXED.with_focal(true))
                    .unwrap(),
            ],
            vec![BundlePoint::try_new(Point3::new(0., 0., 4.)).unwrap()],
            vec![BundleObservation::try_new(0, 0, 0, Point2::new(32., 24.)).unwrap()],
        )
        .build()
        .unwrap();
        let changed =
            CameraIntrinsics::try_new(Camera::try_pinhole(110., 110., 33., 24.).unwrap(), 64, 48)
                .unwrap();
        let candidate = Solution::try_new(
            &problem,
            vec![WorldToCameraPose::identity()],
            vec![changed],
            vec![Point3::new(0., 0., 4.)],
            evidence_report(),
        );
        assert!(
            candidate.is_err(),
            "partial variable mask accepted changed fixed principal point"
        );
    }

    fn exact_scene(
        gauge: Gauge,
        fixed_first_point: bool,
        offset: Vector3<f64>,
        orientation: Rotation3<f64>,
    ) -> BundleProblem {
        let camera = Camera::try_pinhole(400., 400., 320., 240.).unwrap();
        let poses: Vec<_> = [
            Point3::new(0., 0., 0.),
            Point3::new(1., 0., 0.),
            Point3::new(0., 1., 0.),
        ]
        .into_iter()
        .map(|center| {
            WorldToCameraPose::from_rotation_center(
                orientation,
                Point3::from(orientation.inverse() * center.coords + offset),
            )
            .unwrap()
        })
        .collect();
        let points: Vec<_> = (0..18)
            .map(|i| {
                let canonical = Vector3::new(
                    (i % 3) as f64 * 0.5 - 0.5,
                    (i / 3 % 3) as f64 * 0.4 - 0.4,
                    4. + (i / 9) as f64,
                );
                Point3::from(orientation.inverse() * canonical + offset)
            })
            .collect();
        let observations = points
            .iter()
            .enumerate()
            .flat_map(|(point_id, point)| {
                let camera = &camera;
                poses.iter().enumerate().map(move |(pose_id, pose)| {
                    BundleObservation::try_new(
                        pose_id,
                        0,
                        point_id,
                        camera
                            .project(&pose.project_to_camera(point).coords)
                            .unwrap(),
                    )
                    .unwrap()
                })
            })
            .collect();
        BundleProblem::builder(
            poses
                .into_iter()
                .enumerate()
                .map(|(i, pose)| {
                    BundlePose::new(
                        pose,
                        if i == 0 {
                            PoseVariables::FIXED
                        } else {
                            PoseVariables::FREE
                        },
                    )
                })
                .collect(),
            vec![
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(camera, 640, 480).unwrap(),
                    IntrinsicVariables::FIXED,
                )
                .unwrap(),
            ],
            points
                .into_iter()
                .enumerate()
                .map(|(i, point)| {
                    BundlePoint::try_new(point)
                        .unwrap()
                        .with_fixed(fixed_first_point && i == 0)
                })
                .collect(),
            observations,
        )
        .with_gauge(gauge)
        .build()
        .unwrap()
    }

    #[test]
    fn audit_rebinds_the_partial_intrinsic_mask_to_its_problem() {
        let initial =
            CameraIntrinsics::try_new(Camera::try_pinhole(100., 100., 32., 24.).unwrap(), 64, 48)
                .unwrap();
        let make = |variables| {
            BundleProblem::builder(
                vec![BundlePose::new(
                    WorldToCameraPose::identity(),
                    PoseVariables::FIXED,
                )],
                vec![BundleIntrinsic::try_new(initial.clone(), variables).unwrap()],
                vec![BundlePoint::try_new(Point3::new(0., 0., 4.)).unwrap()],
                vec![BundleObservation::try_new(0, 0, 0, Point2::new(32., 24.)).unwrap()],
            )
            .build()
            .unwrap()
        };
        let loose = make(
            IntrinsicVariables::FIXED
                .with_focal(true)
                .with_principal_point(true),
        );
        let strict = make(IntrinsicVariables::FIXED.with_focal(true));
        let changed =
            CameraIntrinsics::try_new(Camera::try_pinhole(110., 110., 33., 24.).unwrap(), 64, 48)
                .unwrap();
        let proposal = Solution::try_new(
            &loose,
            vec![WorldToCameraPose::identity()],
            vec![changed],
            vec![Point3::new(0., 0., 4.)],
            evidence_report(),
        )
        .unwrap();
        assert!(matches!(
            proposal.audit_for_commit(&strict, &SolveConfig::default()),
            Err(openmvg_ba::AuditError::InvalidSolution(
                openmvg_ba::SolutionError::FixedIntrinsicChanged { intrinsic: 0 }
            ))
        ));
    }

    #[test]
    fn a_large_world_origin_does_not_hide_a_wrong_baseline() {
        let problem = exact_scene(
            Gauge::FixedBaseline {
                first: 0,
                second: 1,
                distance: 2.,
            },
            false,
            Vector3::new(1e12, 0., 0.),
            Rotation3::identity(),
        );
        let mut poses: Vec<_> = problem
            .poses()
            .iter()
            .map(|block| block.pose().clone())
            .collect();
        poses[1] = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(1e12 + 1.99, 0., 0.),
        )
        .unwrap();
        let proposed = Solution::try_new(
            &problem,
            poses,
            problem
                .intrinsics()
                .iter()
                .map(|block| block.intrinsics().clone())
                .collect(),
            problem
                .points()
                .iter()
                .map(|block| *block.position())
                .collect(),
            evidence_report(),
        );
        assert!(
            proposed.is_err(),
            "large origin hid a one-percent baseline error"
        );
    }

    #[test]
    fn native_fixed_baseline_must_return_the_requested_distance() {
        let problem = exact_scene(
            Gauge::FixedBaseline {
                first: 0,
                second: 1,
                distance: 2.,
            },
            false,
            Vector3::zeros(),
            Rotation3::identity(),
        );
        let solution = NativeBundleAdjuster::new()
            .solve(&problem, SolveConfig::default())
            .unwrap();
        let distance = (solution.poses()[1].center() - solution.poses()[0].center()).norm();
        assert!(
            (distance - 2.).abs() < 1e-10,
            "requested 2, returned {distance}"
        );
    }

    #[test]
    fn native_fixed_point_must_survive_a_nonidentity_gauge_roundtrip() {
        let problem = exact_scene(
            Gauge::Free,
            true,
            Vector3::new(2.3, -1.7, 0.4),
            Rotation3::from_euler_angles(0.13, -0.17, 0.23),
        );
        let solution = NativeBundleAdjuster::new().solve(&problem, SolveConfig::default());
        assert!(solution.is_ok(), "fixed-point solve failed: {solution:?}");
        assert_eq!(
            solution.unwrap().points()[0],
            *problem.points()[0].position()
        );
    }

    #[test]
    fn checked_solution_must_certify_the_actual_fixed_baseline() {
        let problem = exact_scene(
            Gauge::FixedBaseline {
                first: 0,
                second: 1,
                distance: 2.,
            },
            false,
            Vector3::zeros(),
            Rotation3::identity(),
        );
        let candidate = Solution::try_new(
            &problem,
            problem.poses().iter().map(|b| b.pose().clone()).collect(),
            problem
                .intrinsics()
                .iter()
                .map(|b| b.intrinsics().clone())
                .collect(),
            problem.points().iter().map(|b| *b.position()).collect(),
            evidence_report(),
        );
        if let Ok(solution) = candidate {
            assert!(
                solution
                    .audit_for_commit(&problem, &SolveConfig::default())
                    .is_err(),
                "unchecked wrong baseline was certified"
            );
        }
    }

    #[test]
    fn native_fixed_world_points_must_not_freeze_a_declared_free_pose() {
        let exact = exact_scene(Gauge::Free, false, Vector3::zeros(), Rotation3::identity());
        let truth_anchor = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(0.1, 0., 0.),
        )
        .unwrap();
        let camera = exact.intrinsics()[0].intrinsics().camera();
        let observations = exact
            .observations()
            .iter()
            .map(|o| {
                if o.pose() == 0 {
                    BundleObservation::try_new(
                        0,
                        0,
                        o.point(),
                        camera
                            .project(
                                &truth_anchor
                                    .project_to_camera(exact.points()[o.point()].position())
                                    .coords,
                            )
                            .unwrap(),
                    )
                    .unwrap()
                } else {
                    *o
                }
            })
            .collect();
        let problem = BundleProblem::builder(
            exact
                .poses()
                .iter()
                .map(|b| BundlePose::new(b.pose().clone(), PoseVariables::FREE))
                .collect(),
            exact.intrinsics().to_vec(),
            exact
                .points()
                .iter()
                .map(|b| {
                    BundlePoint::try_new(*b.position())
                        .unwrap()
                        .with_fixed(true)
                })
                .collect(),
            observations,
        )
        .build()
        .unwrap();
        match NativeBundleAdjuster::new().solve(&problem, SolveConfig::default()) {
            Ok(solution) => assert!(
                (solution.poses()[0].center() - truth_anchor.center()).norm() < 1e-8,
                "free pose was silently held at {:?}; report {:?}",
                solution.poses()[0].center(),
                solution.report()
            ),
            Err(openmvg_ba::SolveError::Unsupported(message)) => assert!(
                message.contains("fixed world points") && message.contains("explicitly fixed"),
                "{message}"
            ),
            other => {
                panic!("expected optimized free poses or an explicit capability refusal: {other:?}")
            }
        }
    }

    // Optional localized public-contract addition; imports match numerics_public_probes.rs.
    #[test]
    fn fixed_rotation_gradient_receipt_must_include_scaled_final_point_gradient() {
        use openmvg_core::camera::{BrownConradyCamera, RadialDistortion, TangentialDistortion};
        let scale = 1e10;
        let camera = Camera::BrownConrady(
            BrownConradyCamera::try_with_focal(
                400.,
                320.,
                240.,
                RadialDistortion::K1(0.01),
                TangentialDistortion::None,
            )
            .unwrap(),
        );
        let poses: Vec<_> = [
            Point3::new(0., 0., 0.),
            Point3::new(scale, 0., 0.),
            Point3::new(0., scale, 0.),
        ]
        .into_iter()
        .map(|center| {
            WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap()
        })
        .collect();
        let truth: Vec<_> = (0..18)
            .map(|i| {
                Point3::new(
                    ((i % 3) as f64 * 0.5 - 0.5) * scale,
                    ((i / 3 % 3) as f64 * 0.4 - 0.4) * scale,
                    (4. + (i / 9) as f64) * scale,
                )
            })
            .collect();
        let observations = truth
            .iter()
            .enumerate()
            .flat_map(|(point_id, p)| {
                let camera = &camera;
                poses.iter().enumerate().map(move |(pose_id, pose)| {
                    BundleObservation::try_new(
                        pose_id,
                        0,
                        point_id,
                        camera.project(&pose.project_to_camera(p).coords).unwrap(),
                    )
                    .unwrap()
                })
            })
            .collect();
        let problem = BundleProblem::builder(
            poses
                .into_iter()
                .enumerate()
                .map(|(i, p)| {
                    BundlePose::new(
                        p,
                        if i == 0 {
                            PoseVariables::FIXED
                        } else {
                            PoseVariables::FIXED.with_translation(true)
                        },
                    )
                })
                .collect(),
            vec![
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(camera, 640, 480).unwrap(),
                    IntrinsicVariables::FIXED,
                )
                .unwrap(),
            ],
            truth
                .into_iter()
                .map(|p| BundlePoint::try_new(p + Vector3::new(0.01 * scale, 0., 0.)).unwrap())
                .collect(),
            observations,
        )
        .with_gauge(Gauge::FixedBaseline {
            first: 0,
            second: 1,
            distance: scale,
        })
        .build()
        .unwrap();
        let config = SolveConfig::default();
        let solution = NativeBundleAdjuster::new().solve(&problem, config).unwrap();
        let mut gradients = vec![Vector3::zeros(); solution.points().len()];
        let mut diagonals = gradients.clone();
        for o in problem.observations() {
            let pose = &solution.poses()[o.pose()];
            let camera = solution.intrinsics()[o.intrinsic()].camera();
            let (pixel, jacobian) = camera
                .project_with_jacobian(
                    &pose.project_to_camera(&solution.points()[o.point()]).coords,
                )
                .unwrap();
            let jacobian = jacobian * pose.rotation().matrix();
            gradients[o.point()] += jacobian.transpose() * (pixel - o.pixel());
            for (c, diagonal) in diagonals[o.point()].iter_mut().enumerate() {
                *diagonal += jacobian.column(c).norm_squared();
            }
        }
        let actual = gradients
            .iter()
            .zip(diagonals)
            .map(|(g, d)| (0..3).map(|c| g[c].abs() / d[c].sqrt()).fold(0., f64::max))
            .fold(0., f64::max);
        let reported = solution
            .report()
            .diagnostics()
            .stationarity()
            .expect("the native solve measures final stationarity")
            .scaled_gradient_inf_norm();
        assert!(
            reported + 1e-8 >= actual,
            "reported {reported}, final point gradient {actual}; {:?}",
            solution.report()
        );
        if matches!(
            solution.report().status(),
            SolveStatus::Converged(ConvergenceReason::Gradient)
        ) {
            assert!(
                actual <= config.gradient_tolerance() * (1. + solution.report().final_objective()),
                "gradient convergence did not certify the returned point state"
            );
        }
    }

    // Optional localized addition to numerics_public_probes.rs; shares exact_scene helper.
    // Root owns execution. The final cameras can make an input-held-out point look healthy.
    #[test]
    fn native_quarantined_point_identity_must_survive_improved_camera_geometry() {
        let base = exact_scene(Gauge::Free, false, Vector3::zeros(), Rotation3::identity());
        let mut poses = base.poses().to_vec();
        poses[1] = BundlePose::new(
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0., 0., -1.),
            )
            .unwrap(),
            PoseVariables::FREE,
        );
        poses[2] = BundlePose::new(base.poses()[2].pose().clone(), PoseVariables::FIXED);
        let mut points = base.points().to_vec();
        let index = points.len();
        points.push(BundlePoint::try_new(Point3::new(0., 0., 5.)).unwrap());
        let truth = Point3::new(0.01, 0., 5.);
        let camera = base.intrinsics()[0].intrinsics().camera();
        let mut observations = base.observations().to_vec();
        for pose in [0, 1] {
            let pixel = camera
                .project(&base.poses()[pose].pose().project_to_camera(&truth).coords)
                .unwrap();
            observations.push(BundleObservation::try_new(pose, 0, index, pixel).unwrap());
        }
        let problem =
            BundleProblem::builder(poses, base.intrinsics().to_vec(), points, observations)
                .build()
                .unwrap();
        let solution = NativeBundleAdjuster::new()
            .solve(&problem, SolveConfig::default())
            .unwrap();
        assert!(solution.report().diagnostics().quarantined_point_count() > 0);
        assert!(
            (solution.poses()[1].center() - base.poses()[1].pose().center()).norm() < 1e-3,
            "healthy base did not recover free camera: {:?}",
            solution.report()
        );
        assert!(
            solution.weak_points(&problem).contains(&index),
            "point held out of the solve lost its identity when cameras improved"
        );
    }

    // The two localized quarantine contract cases. Append inside the existing fixed_contract_regressions
    // module, which already has exact_scene and the imported public types. Root owns Cargo execution.
    fn scene_with_collinear_nuisance(fixed: bool) -> BundleProblem {
        let base = exact_scene(Gauge::Free, false, Vector3::zeros(), Rotation3::identity());
        let mut poses = base.poses().to_vec();
        poses.push(BundlePose::new(
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0., 0., -1.),
            )
            .unwrap(),
            PoseVariables::FIXED,
        ));
        let point = Point3::new(0., 0., 5.);
        let mut points = base.points().to_vec();
        points.push(BundlePoint::try_new(point).unwrap().with_fixed(fixed));
        let mut observations = base.observations().to_vec();
        for pose_id in [0, 3] {
            let pixel = base.intrinsics()[0]
                .intrinsics()
                .camera()
                .project(&poses[pose_id].pose().project_to_camera(&point).coords)
                .unwrap();
            observations.push(
                BundleObservation::try_new(
                    pose_id,
                    0,
                    points.len() - 1,
                    pixel + nalgebra::Vector2::new(30., 0.),
                )
                .unwrap(),
            );
        }
        BundleProblem::builder(poses, base.intrinsics().to_vec(), points, observations)
            .build()
            .unwrap()
    }

    #[test]
    fn native_quarantine_metrics_bind_one_final_retained_observation_domain() {
        let problem = scene_with_collinear_nuisance(false);
        let solution = NativeBundleAdjuster::new()
            .solve(&problem, SolveConfig::default())
            .unwrap();
        let held = problem.points().len() - 1;
        assert_eq!(solution.quarantined_points(), &[held]);
        assert_eq!(solution.points()[held], *problem.points()[held].position());
        assert_eq!(
            solution.report().observation_count(),
            Some(problem.observations().len() - 2)
        );
        // The retained exact scene has zero input residual; the two excluded observations have
        // residual 30 pixels each and an original all-observation L2 objective of 900.
        assert!(solution.report().initial_objective() < 1e-10);
        assert!(solution.report().initial_rmse() < 1e-10);
        assert!(solution.report().final_objective() < 1e-8);

        // A checked external backend can supply the same explicit domain. Apparent cross-domain
        // improvement must be replaced with metrics on that domain at the two actual states.
        let stale = SolveReport::try_new(
            solution.report().status(),
            solution.report().iteration_count(),
            9999.0,
            8888.0,
            20.0,
            10.0,
            solution.report().diagnostics(),
        )
        .unwrap();
        assert_eq!(stale.observation_count(), None);
        let poses = solution.poses().to_vec();
        let intrinsics = solution.intrinsics().to_vec();
        let points = solution.points().to_vec();
        assert!(
            Solution::try_new(
                &problem,
                poses.clone(),
                intrinsics.clone(),
                points.clone(),
                stale.clone()
            )
            .is_err()
        );
        let external = Solution::try_new_with_quarantined_points(
            &problem,
            poses,
            intrinsics,
            points,
            stale,
            vec![held],
        )
        .unwrap();
        assert_eq!(
            external.report().observation_count(),
            solution.report().observation_count()
        );
        assert!(external.report().initial_objective() < 1e-10);
        assert!(
            (external.report().final_objective() - solution.report().final_objective()).abs()
                < 1e-10
        );
    }

    // Exactly two localized public tests; append inside fixed_contract_regressions,
    // which already has exact_scene and the common imported public types. Root owns Cargo.
    fn metric_fixed_rotation_problem(gauge: Gauge, variable_focal: bool) -> BundleProblem {
        let base = exact_scene(Gauge::Free, false, Vector3::zeros(), Rotation3::identity());
        let camera = Camera::BrownConrady(
            openmvg_core::camera::BrownConradyCamera::try_with_focal(
                400.0,
                320.0,
                240.0,
                openmvg_core::camera::RadialDistortion::K1(0.01),
                openmvg_core::camera::TangentialDistortion::None,
            )
            .unwrap(),
        );
        let mut poses = base.poses().to_vec();
        poses[1] = BundlePose::new(base.poses()[1].pose().clone(), PoseVariables::FIXED);
        poses[2] = BundlePose::new(
            base.poses()[2].pose().clone(),
            PoseVariables::FREE.with_rotation(false),
        );
        let truth_third = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(0.0, 1.1, 0.0),
        )
        .unwrap();
        let observations = base
            .points()
            .iter()
            .enumerate()
            .flat_map(|(point_id, point)| {
                let camera = &camera;
                let poses = &poses;
                let truth_third = &truth_third;
                (0..3).map(move |pose_id| {
                    let pose = if pose_id == 2 {
                        truth_third
                    } else {
                        poses[pose_id].pose()
                    };
                    BundleObservation::try_new(
                        pose_id,
                        0,
                        point_id,
                        camera
                            .project(&pose.project_to_camera(point.position()).coords)
                            .unwrap(),
                    )
                    .unwrap()
                })
            })
            .collect();
        BundleProblem::builder(
            poses,
            vec![
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(camera, 640, 480).unwrap(),
                    IntrinsicVariables::FIXED.with_focal(variable_focal),
                )
                .unwrap(),
            ],
            base.points().to_vec(),
            observations,
        )
        .with_gauge(gauge)
        .build()
        .unwrap()
    }

    #[test]
    fn metric_fixed_rotation_keeps_all_free_center_coordinates_and_fixed_endpoints() {
        let config = SolveConfig::default();
        for gauge in [
            Gauge::Free,
            Gauge::FixedBaseline {
                first: 0,
                second: 1,
                distance: 1.0,
            },
        ] {
            let problem = metric_fixed_rotation_problem(gauge, false);
            let probe = NativeBundleAdjuster::new()
                .preflight(&problem, &config)
                .unwrap();
            assert_eq!(probe.status, openmvg_ba::NativePreflightStatus::Proceed);
            // Two held centers already fix the scale, so the preflight needs no free-scale
            // proof, as for a pinhole camera below; the solve still reports the three active
            // coordinates of the third center.
            assert_eq!(probe.pose_dimension, None);
            let solution = NativeBundleAdjuster::new().solve(&problem, config).unwrap();
            solution.audit_for_commit(&problem, &config).unwrap();
            for fixed in [0, 1] {
                assert_eq!(solution.poses()[fixed], *problem.poses()[fixed].pose());
            }
            assert!(
                (solution.poses()[2].center() - Point3::new(0.0, 1.1, 0.0)).norm() < 1e-4,
                "an observable free coordinate was frozen: {:?}",
                solution.report()
            );
            assert!(solution.report().final_objective() < 1e-5);
            assert_eq!(
                solution
                    .report()
                    .diagnostics()
                    .observability_details()
                    .unwrap()
                    .pose_dimension(),
                3
            );
        }
        // An already-feasible endpoint can move on its sphere while another center
        // stays fixed. This is a metric constraint, so the old scale-gauge probe
        // cannot certify it; the general solve supplies fresh final information.
        let mixed = metric_fixed_rotation_problem(
            Gauge::FixedBaseline {
                first: 0,
                second: 2,
                distance: 1.0,
            },
            false,
        );
        let probe = NativeBundleAdjuster::new()
            .preflight(&mixed, &config)
            .unwrap();
        assert_eq!(probe.status, openmvg_ba::NativePreflightStatus::Proceed);
        assert_eq!(probe.pose_dimension, None);
        let solution = NativeBundleAdjuster::new().solve(&mixed, config).unwrap();
        solution.audit_for_commit(&mixed, &config).unwrap();
        for fixed in [0, 1] {
            assert_eq!(solution.poses()[fixed], *mixed.poses()[fixed].pose());
        }
        assert!(
            ((solution.poses()[2].center() - solution.poses()[0].center()).norm() - 1.0).abs()
                < 1e-12
        );
        assert_eq!(
            solution.report().diagnostics().observability(),
            Observability::FullRank
        );
    }

    #[test]
    fn native_capability_masks_and_preflight_preserve_the_actual_residual_domain() {
        let config = SolveConfig::default().with_linear_solver(openmvg_ba::LinearSolver::Direct);
        let variable = metric_fixed_rotation_problem(Gauge::Free, true);
        assert!(matches!(
            NativeBundleAdjuster::new().preflight(&variable, &config),
            Err(openmvg_ba::SolveError::Unsupported(_))
        ));
        assert!(matches!(
            NativeBundleAdjuster::new().solve(&variable, config),
            Err(openmvg_ba::SolveError::Unsupported(_))
        ));

        let base = exact_scene(Gauge::Free, false, Vector3::zeros(), Rotation3::identity());
        let k3_camera = Camera::BrownConrady(
            openmvg_core::camera::BrownConradyCamera::try_with_focal(
                400.0,
                320.0,
                240.0,
                openmvg_core::camera::RadialDistortion::K1K2K3(0.0, 0.0, 0.0),
                openmvg_core::camera::TangentialDistortion::None,
            )
            .unwrap(),
        );
        let k3_problem = BundleProblem::builder(
            base.poses().to_vec(),
            vec![
                BundleIntrinsic::try_new(
                    CameraIntrinsics::try_new(k3_camera, 640, 480).unwrap(),
                    IntrinsicVariables::FIXED.with_focal(true).with_radial(true),
                )
                .unwrap(),
            ],
            base.points().to_vec(),
            base.observations().to_vec(),
        )
        .build()
        .unwrap();
        assert!(matches!(
            NativeBundleAdjuster::new().solve(&k3_problem, config),
            Err(openmvg_ba::SolveError::Unsupported(_))
        ));
        let translation_poses: Vec<_> = base
            .poses()
            .iter()
            .enumerate()
            .map(|(pose, block)| {
                BundlePose::new(
                    block.pose().clone(),
                    if pose == 0 {
                        PoseVariables::FIXED
                    } else {
                        PoseVariables::FREE.with_rotation(false)
                    },
                )
            })
            .collect();
        let angular = BundleProblem::builder(
            translation_poses.clone(),
            base.intrinsics().to_vec(),
            base.points().to_vec(),
            base.observations().to_vec(),
        )
        .with_residual_space(openmvg_ba::ResidualSpace::Angular)
        .build()
        .unwrap();
        let angular_probe = NativeBundleAdjuster::new()
            .preflight(&angular, &config)
            .unwrap();
        assert_eq!(
            angular_probe.status,
            openmvg_ba::NativePreflightStatus::Proceed
        );
        assert_eq!(
            angular_probe.pose_dimension, None,
            "pixel probe must not reinterpret angular residuals"
        );
        let angular_solution = NativeBundleAdjuster::new().solve(&angular, config).unwrap();
        angular_solution
            .audit_for_commit(&angular, &config)
            .unwrap();
        assert_eq!(
            angular_solution.report().diagnostics().observability(),
            Observability::FullRank
        );

        let mut metric_poses = translation_poses;
        metric_poses[1] = BundlePose::new(base.poses()[1].pose().clone(), PoseVariables::FIXED);
        let metric = BundleProblem::builder(
            metric_poses,
            base.intrinsics().to_vec(),
            base.points().to_vec(),
            base.observations().to_vec(),
        )
        .build()
        .unwrap();
        let metric_probe = NativeBundleAdjuster::new()
            .preflight(&metric, &config)
            .unwrap();
        assert_eq!(
            metric_probe.status,
            openmvg_ba::NativePreflightStatus::Proceed
        );
        assert_eq!(
            metric_probe.pose_dimension, None,
            "already metric camera masks require no free-scale pixel proof"
        );
        let metric_solution = NativeBundleAdjuster::new().solve(&metric, config).unwrap();
        metric_solution.audit_for_commit(&metric, &config).unwrap();
        assert_eq!(
            metric_solution.report().diagnostics().observability(),
            Observability::FullRank
        );
    }
}
