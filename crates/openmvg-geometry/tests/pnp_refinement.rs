// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{NoiseModel, Options, RefinementStatus};
use openmvg_core::camera::CameraModel;
use openmvg_geometry::pose::test_support::{
    PinholeCamera, PnpError, PnpResidualUnit, RefinementPose, optimize_bearing_pnp,
    optimize_pixel_pnp,
};

const POINTS: [[f64; 3]; 8] = [
    [-0.8, -0.4, 3.2],
    [0.7, -0.5, 3.8],
    [-0.6, 0.8, 4.4],
    [0.9, 0.6, 5.0],
    [-1.1, 0.3, 5.7],
    [0.4, -0.9, 6.3],
    [1.2, 0.1, 7.0],
    [-0.2, 1.0, 7.8],
];

fn pose(rotation: [f64; 3], translation: [f64; 3]) -> RefinementPose {
    RefinementPose::new(manifold::exp_so3(&rotation), translation)
}

fn project(pose: RefinementPose, camera: &PinholeCamera, points: &[[f64; 3]]) -> Vec<[f64; 2]> {
    points
        .iter()
        .map(|point| {
            let camera_point = std::array::from_fn::<_, 3, _>(|row| {
                pose.rotation[row][0] * point[0]
                    + pose.rotation[row][1] * point[1]
                    + pose.rotation[row][2] * point[2]
                    + pose.translation[row]
            });
            let pixel = camera
                .project(&nalgebra::Vector3::from_row_slice(&camera_point))
                .unwrap();
            [pixel.x, pixel.y]
        })
        .collect()
}

fn bearings(pose: RefinementPose, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|point| {
            let camera_point = std::array::from_fn::<_, 3, _>(|row| {
                pose.rotation[row][0] * point[0]
                    + pose.rotation[row][1] * point[1]
                    + pose.rotation[row][2] * point[2]
                    + pose.translation[row]
            });
            let inverse_norm = camera_point
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                .recip();
            camera_point.map(|value| value * inverse_norm)
        })
        .collect()
}

fn rotation_error(actual: RefinementPose, expected: RefinementPose) -> f64 {
    let transpose =
        std::array::from_fn(|row| std::array::from_fn(|column| expected.rotation[column][row]));
    let relative = manifold::mat_mul(&actual.rotation, &transpose);
    manifold::log_so3(&relative)
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn translation_error(actual: RefinementPose, expected: RefinementPose) -> f64 {
    actual
        .translation
        .iter()
        .zip(expected.translation)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn initialized_pnp_recovers_pose_and_reports_full_rank() {
    let camera = PinholeCamera::try_new(720.0, 700.0, 320.0, 240.0).unwrap();
    let expected = pose([0.12, -0.08, 0.05], [0.25, -0.18, 0.45]);
    let observations = project(expected, &camera, &POINTS);
    let initial = pose([0.16, -0.11, 0.07], [0.36, -0.27, 0.58]);

    let report = optimize_pixel_pnp(
        &initial,
        &POINTS,
        &observations,
        &camera,
        Options::l2(40),
        NoiseModel::Unit,
    )
    .unwrap();

    assert!(report.final_objective < report.initial_objective * 1e-10);
    assert!(rotation_error(report.pose, expected) < 1e-9);
    assert!(translation_error(report.pose, expected) < 1e-9);
    assert_eq!(report.diagnostics.rank, 6);
    assert!(matches!(
        report.status,
        RefinementStatus::Converged(_) | RefinementStatus::MaxIterations
    ));
    assert_eq!(report.residual_units, PnpResidualUnit::Pixels);
}

#[test]
fn bearing_pnp_recovers_pose_across_the_full_viewing_sphere() {
    let expected = pose([0.09, -0.06, 0.04], [0.18, -0.12, 0.28]);
    let camera_points = [
        [-2.0, -0.5, -4.0],
        [1.4, -1.1, -3.2],
        [-1.2, 1.5, -2.7],
        [2.2, 0.8, -4.8],
        [-1.6, -1.3, 0.0],
        [1.8, -0.7, 4.2],
        [-0.9, 1.7, 5.0],
        [2.4, 1.2, 3.6],
    ];
    let points = camera_points.map(|camera_point| {
        let centered =
            std::array::from_fn::<_, 3, _>(|row| camera_point[row] - expected.translation[row]);
        std::array::from_fn(|column| {
            (0..3)
                .map(|row| expected.rotation[row][column] * centered[row])
                .sum()
        })
    });
    let observations = bearings(expected, &points);
    assert!(observations.iter().any(|bearing| bearing[2] < 0.0));
    assert!(observations.iter().any(|bearing| bearing[2].abs() < 1e-14));
    let initial = pose([0.11, -0.075, 0.05], [0.23, -0.16, 0.34]);

    let report = optimize_bearing_pnp(
        &initial,
        &points,
        &observations,
        Options::l2(50),
        NoiseModel::Unit,
    )
    .unwrap();

    assert_eq!(report.residual_units, PnpResidualUnit::Radians);
    assert!(
        report.final_objective < report.initial_objective * 1e-10,
        "{report:#?}"
    );
    assert!(rotation_error(report.pose, expected) < 1e-8, "{report:#?}");
    assert!(
        translation_error(report.pose, expected) < 1e-7,
        "{report:#?}"
    );
    assert_eq!(report.diagnostics.rank, 6);
}

#[test]
fn pixel_pnp_refines_an_initialized_three_point_hypothesis() {
    let camera = PinholeCamera::try_new(600.0, 610.0, 320.0, 240.0).unwrap();
    let points = [POINTS[0], POINTS[3], POINTS[6]];
    let expected = pose([0.06, -0.04, 0.03], [0.15, -0.1, 0.35]);
    let observations: [[f64; 2]; 3] = project(expected, &camera, &points).try_into().unwrap();
    let initial = pose([0.07, -0.045, 0.035], [0.17, -0.11, 0.37]);

    let report = optimize_pixel_pnp(
        &initial,
        &points,
        &observations,
        &camera,
        Options::l2(50),
        NoiseModel::Unit,
    )
    .unwrap();

    assert!(report.final_rmse < 1e-7);
    assert!(rotation_error(report.pose, expected) < 1e-7);
    assert!(translation_error(report.pose, expected) < 1e-6);
    assert_eq!(report.diagnostics.rank, 6);
}

#[test]
fn pnp_rejects_mismatched_and_nonprojectable_input() {
    let camera = PinholeCamera::try_with_focal(500.0, 320.0, 240.0).unwrap();
    let pose = RefinementPose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let mismatch = optimize_pixel_pnp(
        &pose,
        &POINTS[..3],
        &[[0.0; 2]; 2],
        &camera,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert!(matches!(
        mismatch,
        Err(PnpError::MeasurementCountMismatch { .. })
    ));

    let points = [[0.0, 0.0, -1.0]; 3];
    let invalid = optimize_pixel_pnp(
        &pose,
        &points,
        &[[0.0; 2]; 3],
        &camera,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert_eq!(invalid, Err(PnpError::PointNotProjectable { point: 0 }));
}

#[test]
fn bearing_pnp_rejects_zero_and_opposing_rays() {
    let identity = RefinementPose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let points = [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    let zero = [[0.0; 3], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    assert_eq!(
        optimize_bearing_pnp(&identity, &points, &zero, Options::l2(1), NoiseModel::Unit,),
        Err(PnpError::InvalidBearing { observation: 0 })
    );

    let opposing = [[0.0, 0.0, -1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    assert_eq!(
        optimize_bearing_pnp(
            &identity,
            &points,
            &opposing,
            Options::l2(1),
            NoiseModel::Unit,
        ),
        Err(PnpError::PointNotProjectable { point: 0 })
    );
}

#[test]
fn repeated_three_point_problem_is_reported_as_degenerate() {
    let camera = PinholeCamera::try_with_focal(500.0, 320.0, 240.0).unwrap();
    let pose = RefinementPose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let points = [[0.2, -0.1, 4.0]; 3];
    let observation = project(pose, &camera, &points)[0];
    let observations = [observation; 3];

    let report = optimize_pixel_pnp(
        &pose,
        &points,
        &observations,
        &camera,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .unwrap();
    assert_eq!(report.status, RefinementStatus::Degenerate);
    assert!(report.diagnostics.rank < 6);
}
