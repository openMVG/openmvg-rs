// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::{
    RelativePose, multi_view::BundleAdjustmentPriors, multi_view::FixedVariableBlocks,
    multi_view::MultiViewBundleAdjustmentError, multi_view::PixelObservation,
    multi_view::RadialIntrinsicPrior, multi_view::ScaleGauge, multi_view::TranslationComponent,
    multi_view::self_calibration::RadialCamera,
    multi_view::self_calibration::SelfCalibratingMeasurements,
    multi_view::self_calibration::SelfCalibrationWorkspace,
    multi_view::self_calibration::covariance_ba_multi_view_self_calibrating_dense,
    multi_view::self_calibration::covariance_ba_multi_view_self_calibrating_grouped_dense,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_dense,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_dense_with_factors,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_grouped_dense,
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{Loss, NoiseModel, Options};

fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

#[test]
fn self_calibration_contract_fixture_freezes_state_and_workspace_reuse() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(manifold::exp_so3(&[0.01, 0.05, -0.01]), [0.7, 0.03, 0.08]),
        RelativePose::new(manifold::exp_so3(&[-0.03, -0.04, 0.02]), [-0.6, 0.07, 0.12]),
    ];
    let truth_cameras = [
        RadialCamera::new(680.0, 0.03, -0.01),
        RadialCamera::new(720.0, -0.025, 0.008),
        RadialCamera::new(700.0, 0.015, -0.006),
    ];
    let truth_points: Vec<_> = (0..30)
        .map(|index| {
            [
                -1.2 + 0.42 * (index % 6) as f64,
                -0.8 + 0.38 * (index / 6) as f64,
                3.6 + 0.19 * (index % 5) as f64,
            ]
        })
        .collect();
    let mut observations = Vec::with_capacity(3 * truth_points.len());
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for camera in 0..3 {
            observations.push(PixelObservation::new(
                camera,
                project(
                    truth_cameras[camera],
                    transform(truth_poses[camera], *point),
                ),
            ));
        }
        offsets.push(observations.len());
    }
    assert_eq!(
        offsets,
        (0..=truth_points.len())
            .map(|point| 3 * point)
            .collect::<Vec<_>>()
    );
    let (tracks, remainder) = observations.as_chunks::<3>();
    assert!(remainder.is_empty());
    assert!(
        tracks
            .iter()
            .all(|track| track.iter().map(|observation| observation.camera).eq(0..3))
    );
    let mut poses = truth_poses;
    poses[1].rotation = manifold::mat_mul(
        &manifold::exp_so3(&[0.004, -0.003, 0.002]),
        &poses[1].rotation,
    );
    poses[2].translation[1] -= 0.012;
    let fixed_x = poses[1].translation[0];
    let mut estimated_points: Vec<_> = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| [point[0] + 0.01 * (index as f64).sin(), point[1], point[2]])
        .collect();
    let mut cameras = [
        truth_cameras[0],
        RadialCamera::new(735.0, -0.02, 0.006),
        RadialCamera::new(685.0, 0.01, -0.004),
    ];
    let initial_poses = poses;
    let initial_points = estimated_points.clone();
    let initial_cameras = cameras;
    let fixed_camera = cameras[0];
    let fixed_intrinsics = [true, false, false];
    let intrinsic_priors = [RadialIntrinsicPrior {
        intrinsic: 1,
        focal: truth_cameras[1].focal,
        k1: truth_cameras[1].k1,
        k2: truth_cameras[1].k2,
        sqrt_information: [[0.2, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    }];
    let mut workspace =
        SelfCalibrationWorkspace::try_new(3, truth_points.len()).expect("small workspace must fit");
    let report = refine_ba_multi_view_self_calibrating_dense_with_factors(
        &mut poses,
        &mut estimated_points,
        &mut cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &[],
            intrinsics: &fixed_intrinsics,
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &[],
            points: &[],
            intrinsics: &intrinsic_priors,
        },
        &mut workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("fixed and prior intrinsic blocks are valid");

    assert_eq!(cameras[0], fixed_camera);
    assert_eq!(poses[1].translation[0], fixed_x);
    assert!((report.initial_cost - 863.644_893_653_298_3).abs() <= 1e-8);
    assert!(report.final_cost <= 1e-14, "{report:#?}");
    const FINAL_STATE_TOLERANCE: f64 = 5e-7;
    for (actual, expected) in poses.iter().zip(&truth_poses) {
        for (actual, expected) in actual
            .rotation
            .iter()
            .flatten()
            .zip(expected.rotation.iter().flatten())
        {
            assert!((actual - expected).abs() <= FINAL_STATE_TOLERANCE);
        }
        for (actual, expected) in actual.translation.iter().zip(expected.translation) {
            assert!((actual - expected).abs() <= FINAL_STATE_TOLERANCE);
        }
    }
    for (actual, expected) in estimated_points.iter().zip(&truth_points) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= FINAL_STATE_TOLERANCE);
        }
    }
    for (actual, expected) in cameras.iter().zip(&truth_cameras) {
        assert!((actual.focal - expected.focal).abs() <= FINAL_STATE_TOLERANCE);
        assert!((actual.k1 - expected.k1).abs() <= FINAL_STATE_TOLERANCE);
        assert!((actual.k2 - expected.k2).abs() <= FINAL_STATE_TOLERANCE);
    }

    let mut warm_poses = initial_poses;
    let mut warm_points = initial_points.clone();
    let mut warm_cameras = initial_cameras;
    let warm_report = refine_ba_multi_view_self_calibrating_dense_with_factors(
        &mut warm_poses,
        &mut warm_points,
        &mut warm_cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &[],
            intrinsics: &fixed_intrinsics,
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &[],
            points: &[],
            intrinsics: &intrinsic_priors,
        },
        &mut workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("a reused workspace must solve the same fixture");
    let mut fresh_poses = initial_poses;
    let mut fresh_points = initial_points.clone();
    let mut fresh_cameras = initial_cameras;
    let mut fresh_workspace =
        SelfCalibrationWorkspace::try_new(3, truth_points.len()).expect("small workspace must fit");
    let fresh_report = refine_ba_multi_view_self_calibrating_dense_with_factors(
        &mut fresh_poses,
        &mut fresh_points,
        &mut fresh_cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &[],
            intrinsics: &fixed_intrinsics,
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &[],
            points: &[],
            intrinsics: &intrinsic_priors,
        },
        &mut fresh_workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("a fresh workspace must solve the same fixture");
    assert_eq!(warm_report, fresh_report);
    assert_eq!(warm_poses, fresh_poses);
    assert_eq!(warm_points, fresh_points);
    assert_eq!(warm_cameras, fresh_cameras);

    let mut invalid_cameras = initial_cameras;
    invalid_cameras[1].focal = 0.0;
    let invalid_cameras_before = invalid_cameras;
    let mut rollback_poses = initial_poses;
    let mut rollback_points = initial_points.clone();
    let rollback_error = refine_ba_multi_view_self_calibrating_dense_with_factors(
        &mut rollback_poses,
        &mut rollback_points,
        &mut invalid_cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        FixedVariableBlocks {
            camera_poses: &[],
            points: &[],
            intrinsics: &fixed_intrinsics,
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &[],
            points: &[],
            intrinsics: &intrinsic_priors,
        },
        &mut workspace,
        Options::l2(100),
        NoiseModel::Unit,
    );
    assert_eq!(
        rollback_error,
        Err(MultiViewBundleAdjustmentError::InvalidIntrinsics { camera: 1 })
    );
    assert_eq!(rollback_poses, initial_poses);
    assert_eq!(rollback_points, initial_points);
    assert_eq!(invalid_cameras, invalid_cameras_before);
}

#[test]
fn shared_intrinsic_groups_recover_one_block_for_multiple_views() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(manifold::exp_so3(&[0.02, 0.06, -0.01]), [0.7, 0.03, 0.08]),
        RelativePose::new(manifold::exp_so3(&[-0.04, -0.05, 0.02]), [-0.6, 0.07, 0.12]),
        RelativePose::new(manifold::exp_so3(&[0.05, -0.02, 0.03]), [0.2, -0.5, 0.1]),
    ];
    let camera_to_group = [0, 0, 1, 1];
    let truth_groups = [
        RadialCamera::new(680.0, 0.055, -0.022),
        RadialCamera::new(735.0, -0.042, 0.018),
    ];
    let truth_points: Vec<_> = (0..72)
        .map(|index| {
            [
                -1.4 + 0.34 * (index % 9) as f64,
                -1.0 + 0.3 * (index / 9) as f64,
                3.2 + 0.31 * (index % 7) as f64,
            ]
        })
        .collect();
    let mut observations = Vec::with_capacity(4 * truth_points.len());
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for camera in 0..truth_poses.len() {
            observations.push(PixelObservation::new(
                camera,
                project(
                    truth_groups[camera_to_group[camera]],
                    transform(truth_poses[camera], *point),
                ),
            ));
        }
        offsets.push(observations.len());
    }

    let mut poses = truth_poses;
    for (camera, pose) in poses.iter_mut().enumerate().skip(1) {
        pose.rotation = manifold::mat_mul(
            &manifold::exp_so3(&[0.004, -0.003 * camera as f64, 0.002]),
            &pose.rotation,
        );
        pose.translation[0] += 0.01;
        pose.translation[1] -= 0.008;
    }
    let baseline = norm(&truth_poses[1].translation);
    let baseline_scale = baseline / norm(&poses[1].translation);
    poses[1].translation = poses[1].translation.map(|value| value * baseline_scale);
    let mut points: Vec<_> = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.01 * phase.sin(),
                point[1] - 0.008 * phase.cos(),
                point[2] + 0.012 * (0.4 * phase).sin(),
            ]
        })
        .collect();
    let mut groups = [
        RadialCamera::new(700.0, 0.062, -0.026),
        RadialCamera::new(715.0, -0.035, 0.014),
    ];
    let initial_error: f64 = groups
        .iter()
        .zip(truth_groups)
        .map(|(estimate, truth)| {
            (estimate.focal / truth.focal - 1.0).abs()
                + (estimate.k1 - truth.k1).abs()
                + (estimate.k2 - truth.k2).abs()
        })
        .sum();
    let mut workspace = SelfCalibrationWorkspace::try_new(poses.len(), points.len())
        .expect("small grouped workspace must fit");
    let report = refine_ba_multi_view_self_calibrating_grouped_dense(
        &mut poses,
        &mut points,
        &mut groups,
        &camera_to_group,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("well-conditioned grouped self-calibration must be valid");
    let final_error: f64 = groups
        .iter()
        .zip(truth_groups)
        .map(|(estimate, truth)| {
            (estimate.focal / truth.focal - 1.0).abs()
                + (estimate.k1 - truth.k1).abs()
                + (estimate.k2 - truth.k2).abs()
        })
        .sum();

    assert!(report.final_rmse < 1e-7, "{report:#?}");
    assert!(final_error < initial_error * 1e-3);
    assert_eq!(report.diagnostics.pose_dimension, 23);
    assert_eq!(report.diagnostics.pose_rank, 23, "{report:#?}");
    assert!((norm(&poses[1].translation) - baseline).abs() < 2e-13);

    let covariance = covariance_ba_multi_view_self_calibrating_grouped_dense(
        &poses,
        &points,
        &groups,
        &camera_to_group,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Loss::L2,
        NoiseModel::Unit,
    )
    .expect("full-rank grouped self-calibration has finite covariance");
    assert_eq!(covariance.camera_parameter_range(0), Some(0..0));
    assert_eq!(covariance.camera_parameter_range(1), Some(0..5));
    assert_eq!(covariance.camera_parameter_range(2), Some(5..11));
    assert_eq!(covariance.camera_parameter_range(3), Some(11..17));
    assert_eq!(covariance.intrinsic_group_parameter_range(0), Some(17..20));
    assert_eq!(covariance.intrinsic_group_parameter_range(1), Some(20..23));
}

fn transform(pose: RelativePose, point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

fn project(camera: RadialCamera, point: [f64; 3]) -> [f64; 2] {
    let normalized = [point[0] / point[2], point[1] / point[2]];
    let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
    let distortion = 1.0 + camera.k1 * radius_squared + camera.k2 * radius_squared * radius_squared;
    normalized.map(|value| camera.focal * distortion * value)
}

fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn exact_report(
    truth_poses: &[RelativePose],
    truth_points: &[[f64; 3]],
    truth_cameras: &[RadialCamera],
    gauge: ScaleGauge,
) -> openmvg_ba::kernel::multi_view::MultiViewBundleAdjustmentReport {
    report_with_final_diagnostics(truth_poses, truth_points, truth_cameras, gauge, true)
}

fn report_with_final_diagnostics(
    truth_poses: &[RelativePose],
    truth_points: &[[f64; 3]],
    truth_cameras: &[RadialCamera],
    gauge: ScaleGauge,
    final_diagnostics: bool,
) -> openmvg_ba::kernel::multi_view::MultiViewBundleAdjustmentReport {
    let mut observations = Vec::with_capacity(truth_poses.len() * truth_points.len());
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in truth_points {
        for camera in 0..truth_poses.len() {
            observations.push(PixelObservation::new(
                camera,
                project(
                    truth_cameras[camera],
                    transform(truth_poses[camera], *point),
                ),
            ));
        }
        offsets.push(observations.len());
    }
    let mut poses = truth_poses.to_vec();
    let mut points = truth_points.to_vec();
    let mut cameras = truth_cameras.to_vec();
    let mut workspace = SelfCalibrationWorkspace::try_new(poses.len(), points.len())
        .expect("small diagnostic workspace must fit")
        .with_final_diagnostics(final_diagnostics);
    refine_ba_multi_view_self_calibrating_dense(
        &mut poses,
        &mut points,
        &mut cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        gauge,
        &mut workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .expect("finite diagnostic geometry must be accepted")
}

#[test]
fn disabling_final_diagnostics_preserves_the_nonlinear_result() {
    let poses = [
        identity_pose(),
        RelativePose::new(manifold::exp_so3(&[0.01, 0.04, -0.02]), [0.8, 0.03, 0.1]),
        RelativePose::new(manifold::exp_so3(&[-0.03, 0.02, 0.01]), [-0.6, 0.08, 0.15]),
    ];
    let points: Vec<_> = (0..20)
        .map(|index| {
            [
                -1.0 + 0.4 * (index % 5) as f64,
                -0.6 + 0.4 * (index / 5) as f64,
                3.5 + 0.2 * (index % 3) as f64,
            ]
        })
        .collect();
    let cameras = [
        RadialCamera::new(680.0, 0.02, -0.005),
        RadialCamera::new(700.0, -0.01, 0.004),
        RadialCamera::new(720.0, 0.015, -0.003),
    ];

    let assessed = report_with_final_diagnostics(
        &poses,
        &points,
        &cameras,
        ScaleGauge::FixedBaseline { camera: 1 },
        true,
    );
    let unassessed = report_with_final_diagnostics(
        &poses,
        &points,
        &cameras,
        ScaleGauge::FixedBaseline { camera: 1 },
        false,
    );

    assert_eq!(unassessed.status, assessed.status);
    assert_eq!(unassessed.iteration_count, assessed.iteration_count);
    assert_eq!(unassessed.final_cost, assessed.final_cost);
    assert_eq!(unassessed.final_rmse, assessed.final_rmse);
    assert!(assessed.diagnostics.pose_rank_assessed);
    assert!(assessed.diagnostics.point_rank_assessed);
    assert!(!unassessed.diagnostics.pose_rank_assessed);
    assert!(!unassessed.diagnostics.point_rank_assessed);
    assert!(unassessed.gradient_inf_norm.is_nan());
}

#[test]
fn four_view_self_calibration_recovers_focal_lengths_and_radial_distortion() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(
            manifold::exp_so3(&[0.025, 0.075, -0.018]),
            [0.72, 0.03, 0.08],
        ),
        RelativePose::new(
            manifold::exp_so3(&[-0.045, -0.065, 0.022]),
            [-0.61, 0.08, 0.13],
        ),
        RelativePose::new(
            manifold::exp_so3(&[0.06, -0.025, 0.035]),
            [0.18, -0.49, 0.1],
        ),
    ];
    let truth_cameras = [
        RadialCamera::new(690.0, 0.070, -0.030),
        RadialCamera::new(735.0, -0.045, 0.024),
        RadialCamera::new(665.0, 0.055, -0.018),
        RadialCamera::new(710.0, -0.065, 0.032),
    ];
    let truth_points: Vec<_> = (0..80)
        .map(|index| {
            let column = (index % 10) as f64;
            let row = (index / 10) as f64;
            [
                -1.65 + 0.37 * column,
                -1.15 + 0.33 * row,
                3.2 + 0.37 * (index % 7) as f64 + 0.08 * row,
            ]
        })
        .collect();
    let mut observations = Vec::with_capacity(4 * truth_points.len());
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for camera in 0..truth_poses.len() {
            observations.push(PixelObservation::new(
                camera,
                project(
                    truth_cameras[camera],
                    transform(truth_poses[camera], *point),
                ),
            ));
        }
        offsets.push(observations.len());
    }

    let mut poses = truth_poses;
    for (camera, pose) in poses.iter_mut().enumerate().skip(1) {
        pose.rotation = manifold::mat_mul(
            &manifold::exp_so3(&[0.004 * camera as f64, -0.006, 0.003]),
            &pose.rotation,
        );
        pose.translation[0] += 0.012 * camera as f64;
        pose.translation[1] -= 0.008;
        pose.translation[2] += 0.006;
    }
    let baseline = norm(&truth_poses[1].translation);
    let scale = baseline / norm(&poses[1].translation);
    poses[1].translation = poses[1].translation.map(|value| scale * value);
    let mut points: Vec<_> = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.012 * phase.sin(),
                point[1] - 0.009 * (0.7 * phase).cos(),
                point[2] + 0.018 * (0.3 * phase).sin(),
            ]
        })
        .collect();
    let focal_scales = [1.025, 0.978, 1.018, 0.972];
    let mut cameras: Vec<_> = truth_cameras
        .iter()
        .enumerate()
        .map(|(index, camera)| {
            RadialCamera::new(
                camera.focal * focal_scales[index],
                camera.k1 + if index % 2 == 0 { 0.008 } else { -0.007 },
                camera.k2 + if index % 2 == 0 { -0.004 } else { 0.003 },
            )
        })
        .collect();
    let initial_focal_error: f64 = cameras
        .iter()
        .zip(truth_cameras)
        .map(|(estimate, truth)| (estimate.focal / truth.focal - 1.0).abs())
        .sum();
    let initial_distortion_error: f64 = cameras
        .iter()
        .zip(truth_cameras)
        .map(|(estimate, truth)| (estimate.k1 - truth.k1).abs() + (estimate.k2 - truth.k2).abs())
        .sum();
    let mut workspace = SelfCalibrationWorkspace::try_new(poses.len(), points.len())
        .expect("small synthetic workspace must fit");
    let report = refine_ba_multi_view_self_calibrating_dense(
        &mut poses,
        &mut points,
        &mut cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("well-conditioned synthetic self-calibration problem must be valid");

    let final_focal_error: f64 = cameras
        .iter()
        .zip(truth_cameras)
        .map(|(estimate, truth)| (estimate.focal / truth.focal - 1.0).abs())
        .sum();
    let final_distortion_error: f64 = cameras
        .iter()
        .zip(truth_cameras)
        .map(|(estimate, truth)| (estimate.k1 - truth.k1).abs() + (estimate.k2 - truth.k2).abs())
        .sum();
    assert!(report.final_rmse < 1e-7, "{report:#?}");
    assert!(
        report.final_cost < report.initial_cost * 1e-12,
        "{report:#?}"
    );
    assert!(final_focal_error < initial_focal_error * 1e-4);
    assert!(final_distortion_error < initial_distortion_error * 1e-3);
    assert_eq!(report.diagnostics.pose_dimension, 29);
    assert_eq!(report.diagnostics.pose_rank, 29, "{report:#?}");
    assert!((norm(&poses[1].translation) - baseline).abs() < 2e-13);
    assert_eq!(poses[0], identity_pose());

    let covariance = covariance_ba_multi_view_self_calibrating_dense(
        &poses,
        &points,
        &cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Loss::L2,
        NoiseModel::Unit,
    )
    .expect("full-rank self-calibration has finite local covariance");
    assert_eq!(covariance.camera_dimension, 29);
    assert_eq!(covariance.camera_parameter_range(0), Some(0..3));
    assert_eq!(covariance.camera_parameter_range(1), Some(3..11));
    assert_eq!(covariance.camera_parameter_range(2), Some(11..20));
    assert_eq!(covariance.camera_parameter_range(3), Some(20..29));
    assert_eq!(covariance.camera_covariance.len(), 29 * 29);
    assert_eq!(covariance.point_covariances.len(), points.len());
    assert!(
        covariance
            .camera_covariance
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn self_calibration_rejects_nonpositive_focal_length() {
    let mut poses = [
        identity_pose(),
        RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        ),
        RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-1.0, 0.0, 0.0],
        ),
    ];
    let mut points = [
        [0.0, 0.0, 3.0],
        [0.2, 0.0, 3.2],
        [0.0, 0.2, 3.4],
        [-0.2, -0.1, 3.6],
    ];
    let mut cameras = [
        RadialCamera::new(500.0, 0.0, 0.0),
        RadialCamera::new(0.0, 0.0, 0.0),
        RadialCamera::new(500.0, 0.0, 0.0),
    ];
    let observations: [PixelObservation; 12] =
        std::array::from_fn(|index| PixelObservation::new(index % 3, [index as f64, 0.0]));
    let offsets = [0, 3, 6, 9, 12];
    let mut workspace = SelfCalibrationWorkspace::try_new(3, 4).expect("small workspace");
    let result = refine_ba_multi_view_self_calibrating_dense(
        &mut poses,
        &mut points,
        &mut cameras,
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert_eq!(
        result,
        Err(MultiViewBundleAdjustmentError::InvalidIntrinsics { camera: 1 })
    );

    let mut groups = [RadialCamera::new(500.0, 0.0, 0.0)];
    let result = refine_ba_multi_view_self_calibrating_grouped_dense(
        &mut poses,
        &mut points,
        &mut groups,
        &[0, 0],
        SelfCalibratingMeasurements::new(&observations, &offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert_eq!(
        result,
        Err(
            MultiViewBundleAdjustmentError::IntrinsicGroupMapCountMismatch {
                expected: 3,
                actual: 2,
            }
        )
    );
}

#[test]
fn self_calibration_diagnostics_expose_critical_motion_and_narrow_field_of_view() {
    let cameras = [RadialCamera::new(700.0, 0.04, -0.01); 4];
    let points: Vec<_> = (0..48)
        .map(|index| {
            let x = (index % 8) as f64;
            let y = (index / 8) as f64;
            [
                -1.4 + 0.4 * x,
                -1.0 + 0.4 * y,
                3.5 + 0.2 * (index % 5) as f64,
            ]
        })
        .collect();

    let pure_rotation_poses = [
        identity_pose(),
        RelativePose::new(manifold::exp_so3(&[0.02, 0.08, -0.01]), [0.0; 3]),
        RelativePose::new(manifold::exp_so3(&[-0.05, -0.06, 0.03]), [0.0; 3]),
        RelativePose::new(manifold::exp_so3(&[0.07, -0.02, 0.04]), [0.0; 3]),
    ];
    let pure_rotation = exact_report(&pure_rotation_poses, &points, &cameras, ScaleGauge::Free);
    assert_eq!(pure_rotation.diagnostics.weak_point_count, points.len());
    assert!(matches!(
        pure_rotation.status,
        openmvg_ba::numerics::refine::RefinementStatus::Degenerate
    ));
    assert!(pure_rotation.final_cost.is_finite());
    assert!(pure_rotation.final_rmse.is_finite());

    let healthy_poses = [
        identity_pose(),
        RelativePose::new(manifold::exp_so3(&[0.01, 0.05, -0.01]), [0.7, 0.02, 0.08]),
        RelativePose::new(manifold::exp_so3(&[-0.03, -0.04, 0.02]), [-0.6, 0.08, 0.12]),
        RelativePose::new(manifold::exp_so3(&[0.04, -0.01, 0.03]), [0.15, -0.5, 0.1]),
    ];
    let healthy = exact_report(
        &healthy_poses,
        &points,
        &cameras,
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    assert!(healthy.diagnostics.is_well_constrained(), "{healthy:#?}");

    let planar_points: Vec<_> = points
        .iter()
        .map(|point| [point[0], point[1], 4.0])
        .collect();
    let planar = exact_report(
        &healthy_poses,
        &planar_points,
        &cameras,
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    assert!(
        planar.diagnostics.additional_camera_nullity() > 0
            || planar
                .diagnostics
                .observable_camera_hessian_reciprocal_condition_number
                < healthy
                    .diagnostics
                    .observable_camera_hessian_reciprocal_condition_number
                    * 1e-4,
        "healthy={healthy:#?}\nplanar={planar:#?}"
    );

    let low_parallax_poses = healthy_poses.map(|mut pose| {
        pose.translation = pose.translation.map(|value| value * 0.01);
        pose
    });
    let low_parallax = exact_report(
        &low_parallax_poses,
        &points,
        &cameras,
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    assert!(
        low_parallax
            .diagnostics
            .minimum_point_hessian_reciprocal_condition_number
            < healthy
                .diagnostics
                .minimum_point_hessian_reciprocal_condition_number
                * 1e-3,
        "healthy={healthy:#?}\nlow_parallax={low_parallax:#?}"
    );

    let narrow_points: Vec<_> = points
        .iter()
        .map(|point| [point[0] * 1e-3, point[1] * 1e-3, point[2]])
        .collect();
    let narrow = exact_report(
        &healthy_poses,
        &narrow_points,
        &cameras,
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    assert!(
        narrow.diagnostics.additional_camera_nullity() > 0
            || narrow
                .diagnostics
                .observable_camera_hessian_reciprocal_condition_number
                < healthy
                    .diagnostics
                    .observable_camera_hessian_reciprocal_condition_number
                    * 1e-4,
        "healthy={healthy:#?}\nnarrow={narrow:#?}"
    );
}
