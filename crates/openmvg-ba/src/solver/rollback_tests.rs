// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Controlled backend proposals exercise the returned-state boundary. These
//! tests do not claim that a native solve produced the proposed non-descent.

use super::{
    AuditError, ConvergenceReason, Observability, Solution, SolveConfig, SolveDiagnostics,
    SolveReport, SolveStatus, StationarityAudit,
};
use crate::{
    BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem, IntrinsicVariables,
    PoseVariables,
};
use nalgebra::{Matrix2x3, Matrix3, Point2, Point3, Rotation3, Vector3};
use openmvg_core::{
    camera::Camera,
    model::{CameraIntrinsics, WorldToCameraPose},
};

fn poses() -> Vec<WorldToCameraPose> {
    vec![
        WorldToCameraPose::identity(),
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(0.0, 0.0, 1.0))
            .unwrap(),
    ]
}

fn project(point: Point3<f64>, center_z: f64, focal: f64) -> Point2<f64> {
    let z = point.z - center_z;
    Point2::new(focal * point.x / z + 32.0, focal * point.y / z + 24.0)
}

/// Independently derive the point's pixel information and scaled gradient.
fn point_evidence(
    point: Point3<f64>,
    measured: &[Point2<f64>; 2],
    focal: f64,
) -> (f64, f64, f64, f64) {
    let mut information = Matrix3::zeros();
    let mut gradient = Vector3::zeros();
    let mut squared = 0.0;
    for (camera, pixel) in measured.iter().enumerate() {
        let z = point.z - camera as f64;
        let jacobian = Matrix2x3::new(
            focal / z,
            0.0,
            -focal * point.x / (z * z),
            0.0,
            focal / z,
            -focal * point.y / (z * z),
        );
        let residual = project(point, camera as f64, focal) - pixel;
        information += jacobian.transpose() * jacobian;
        gradient += jacobian.transpose() * residual;
        squared += residual.norm_squared();
    }
    let eigenvalues = information.symmetric_eigenvalues();
    let scaled_gradient = (0..3)
        .map(|axis| gradient[axis].abs() / information[(axis, axis)].sqrt())
        .fold(0.0, f64::max);
    (
        0.5 * squared,
        (squared / 2.0).sqrt(),
        scaled_gradient,
        eigenvalues.min() / eigenvalues.max(),
    )
}

fn problem(
    point: Point3<f64>,
    measured: [Point2<f64>; 2],
    quarantine: bool,
    focal: f64,
) -> BundleProblem {
    let camera = CameraIntrinsics::try_new(
        Camera::try_pinhole(focal, focal, 32.0, 24.0).unwrap(),
        64,
        48,
    )
    .unwrap();
    let mut points = vec![BundlePoint::try_new(point).unwrap()];
    let mut observations = (0..2)
        .map(|camera| BundleObservation::try_new(camera, 0, 0, measured[camera]).unwrap())
        .collect::<Vec<_>>();
    if quarantine {
        let held = Point3::new(0.5, 0.2, 6.0);
        points.push(BundlePoint::try_new(held).unwrap());
        for camera in 0..2 {
            observations.push(
                BundleObservation::try_new(camera, 0, 1, project(held, camera as f64, focal))
                    .unwrap(),
            );
        }
    }
    BundleProblem::builder(
        poses()
            .into_iter()
            .map(|pose| BundlePose::new(pose, PoseVariables::FIXED))
            .collect(),
        vec![
            BundleIntrinsic::try_new(camera.clone(), IntrinsicVariables::FIXED).unwrap(),
            // An unused group is explicitly held by this backend proposal.
            BundleIntrinsic::try_new(camera, IntrinsicVariables::FIXED.with_focal(true)).unwrap(),
        ],
        points,
        observations,
    )
    .build()
    .unwrap()
}

fn candidate(
    problem: &BundleProblem,
    point: Point3<f64>,
    measured: &[Point2<f64>; 2],
    quarantine: bool,
    scalar_roundoff_only: bool,
    focal: f64,
) -> Solution {
    let input = point_evidence(*problem.points()[0].position(), measured, focal);
    let proposed = point_evidence(point, measured, focal);
    assert!(
        proposed.3 > 1.0e-5,
        "the proposed point must have healthy independent information"
    );
    let counters = if scalar_roundoff_only {
        (3, 0, 2)
    } else {
        (7, 2, 3)
    };
    let diagnostics = SolveDiagnostics::new(
        counters.1,
        counters.2,
        StationarityAudit::try_from_fresh_zero_damping(proposed.2).unwrap(),
        Observability::FullRank,
        0,
        0,
    )
    .with_quarantined_point_count(usize::from(quarantine));
    let report = SolveReport::try_new(
        SolveStatus::Converged(if scalar_roundoff_only {
            ConvergenceReason::Gradient
        } else {
            ConvergenceReason::Step
        }),
        counters.0,
        input.0,
        if scalar_roundoff_only {
            proposed.0.next_up()
        } else {
            proposed.0
        },
        input.1,
        proposed.1,
        diagnostics,
    )
    .unwrap()
    .with_unconstrained_intrinsics(vec![1]);
    let mut points = problem
        .points()
        .iter()
        .map(|point| *point.position())
        .collect::<Vec<_>>();
    points[0] = point;
    Solution::try_new_with_quarantined_points(
        problem,
        problem
            .poses()
            .iter()
            .map(|pose| pose.pose().clone())
            .collect(),
        problem
            .intrinsics()
            .iter()
            .map(|intrinsic| intrinsic.intrinsics().clone())
            .collect(),
        points,
        report,
        if quarantine { vec![1] } else { Vec::new() },
    )
    .unwrap()
}

#[test]
fn state_changing_roundoff_rollback_cannot_reuse_the_proposal_certificate() {
    // Exact collinear rays leave the input point's depth unconstrained. A
    // checked positive focal of 1e-5 makes the proposal's non-descent small in
    // pixel units without making its world-coordinate change merely roundoff.
    let focal = 1.0e-5;
    let input_point = Point3::new(0.0, 0.0, 4.0);
    let proposed_point = Point3::new(0.1, 0.0, 4.0);
    let measured = std::array::from_fn(|camera| project(input_point, camera as f64, focal));
    let input_evidence = point_evidence(input_point, &measured, focal);
    let proposal_evidence = point_evidence(proposed_point, &measured, focal);
    assert_eq!(input_evidence.3, 0.0);
    assert!(proposal_evidence.3 > 1.0e-5);
    assert!((input_evidence.2 - proposal_evidence.2).abs() > 1.0e-8);
    let problem = problem(input_point, measured, true, focal);
    let proposal = candidate(&problem, proposed_point, &measured, true, false, focal);
    let increase = proposal.report().final_objective() - proposal.report().initial_objective();
    assert!(increase > 0.0 && increase < 2.0 * f64::EPSILON.sqrt());
    assert!(!proposal.weak_points(&problem).contains(&0));
    assert_eq!(
        proposal.audit_for_commit(&problem, &SolveConfig::default()),
        Err(AuditError::ObjectiveIncreased)
    );

    let rolled_back = proposal.canonicalize_roundoff_non_descent(&problem);
    assert_eq!(rolled_back.points()[0], input_point);
    assert!(rolled_back.weak_points(&problem).contains(&0));
    assert_eq!(rolled_back.quarantined_points(), [1]);
    assert_eq!(rolled_back.report().observation_count(), Some(2));
    assert_eq!(rolled_back.report().unconstrained_intrinsics(), [1]);
    let diagnostics = rolled_back.report().diagnostics();
    assert_eq!(diagnostics.stationarity(), None);
    assert_eq!(diagnostics.weak_point_count(), None);
    assert_eq!(diagnostics.noninvertible_observation_count(), None);
    assert_eq!(diagnostics.observability_details(), None);
    assert_eq!(
        rolled_back.audit_for_commit(&problem, &SolveConfig::default()),
        Err(AuditError::ObservabilityNotAssessed),
        "restored weak input accepted the proposal's unrelated final certificate",
    );
    assert_eq!(rolled_back.report().iteration_count(), 7);
    assert_eq!(rolled_back.report().diagnostics().accepted_step_count(), 2);
    assert_eq!(rolled_back.report().diagnostics().rejected_step_count(), 3);
}

#[test]
fn same_geometry_scalar_roundoff_keeps_the_fresh_certificate() {
    let point = Point3::new(0.2, 0.0, 4.0);
    let focal = 100.0;
    // The opposing vertical residuals cancel their analytic gradient at y=0.
    let measured = [
        project(point, 0.0, focal) - nalgebra::Vector2::new(0.0, 0.02),
        project(point, 1.0, focal) + nalgebra::Vector2::new(0.0, 0.015),
    ];
    let problem = problem(point, measured, false, focal);
    let proposal = candidate(&problem, point, &measured, false, true, focal);
    let canonical = proposal.canonicalize_roundoff_non_descent(&problem);
    assert_eq!(canonical.points()[0], point);
    assert_eq!(
        canonical.audit_for_commit(&problem, &SolveConfig::default()),
        Ok(())
    );
    let diagnostics = canonical.report().diagnostics();
    assert!(diagnostics.stationarity().is_some());
    assert_eq!(diagnostics.weak_point_count(), Some(0));
    assert_eq!(diagnostics.noninvertible_observation_count(), Some(0));
    assert_eq!(
        canonical.report().final_objective(),
        canonical.report().initial_objective()
    );
    assert_eq!(canonical.report().iteration_count(), 3);
    assert_eq!(canonical.report().diagnostics().accepted_step_count(), 0);
    assert_eq!(canonical.report().diagnostics().rejected_step_count(), 2);
    assert_eq!(canonical.report().unconstrained_intrinsics(), [1]);
}
