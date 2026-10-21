// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::AngularObservation,
    multi_view::BundleAdjustmentPriors, multi_view::CalibratedBundleAdjustmentProblem,
    multi_view::FixedVariableBlocks, multi_view::MultiViewBundleAdjustmentError,
    multi_view::MultiViewMeasurements, multi_view::PixelObservation, multi_view::PointPrior,
    multi_view::PosePrior, multi_view::ScaleGauge, multi_view::TranslationComponent,
    multi_view::dense::DenseBundleAdjustmentWorkspace,
    multi_view::dense::MultiViewBundleAdjustmentCovarianceError,
    multi_view::dense::covariance_ba_multi_view_dense,
    multi_view::dense::covariance_ba_multi_view_dense_with_factors,
    multi_view::dense::refine_ba_multi_view_dense,
    multi_view::dense::refine_ba_multi_view_dense_with_factors,
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{Loss, NoiseModel, Options, RefinementStatus, UncertaintyModel};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (sine, cosine) = angle.sin_cos();
    [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
}

fn mat_mul(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

fn points() -> Vec<[f64; 3]> {
    (0..24)
        .map(|index| {
            let column = (index % 6) as f64;
            let row = (index / 6) as f64;
            [
                -1.1 + 0.42 * column,
                -0.65 + 0.43 * row,
                4.0 + 0.17 * (index % 5) as f64 + 0.11 * row,
            ]
        })
        .collect()
}

fn perturb(poses: &[RelativePose], points: &[[f64; 3]]) -> (Vec<RelativePose>, Vec<[f64; 3]>) {
    let mut perturbed_poses = poses.to_vec();
    for (camera, pose) in perturbed_poses.iter_mut().enumerate().skip(1) {
        let delta = rotation_y(if camera % 2 == 0 { -0.025 } else { 0.03 });
        pose.rotation = mat_mul(&delta, &pose.rotation);
        pose.translation[0] += 0.035 * camera as f64;
        pose.translation[1] -= 0.018 * camera as f64;
        pose.translation[2] += 0.025;
    }
    let perturbed_points = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.025 * phase.sin(),
                point[1] - 0.018 * (0.7 * phase).cos(),
                point[2] + 0.04 * (0.4 * phase).sin(),
            ]
        })
        .collect();
    (perturbed_poses, perturbed_points)
}

#[test]
fn calibrated_contract_fixture_freezes_factors_state_and_workspace_reuse() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.045), [0.62, 0.015, 0.04]),
        RelativePose::new(rotation_y(-0.055), [-0.53, 0.045, 0.09]),
    ];
    let truth_points = points();
    let camera = PinholeCamera::new(720.0, 710.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut observations = Vec::new();
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for (camera_index, pose) in truth_poses.iter().enumerate() {
            let value = transform(pose, point);
            observations.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * value[0] / value[2] + camera.cx,
                    camera.fy * value[1] / value[2] + camera.cy,
                ],
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
    let (mut poses, mut estimated_points) = perturb(&truth_poses, &truth_points);
    poses[1].translation[0] = truth_poses[1].translation[0];
    poses[2] = truth_poses[2];
    estimated_points[0] = truth_points[0];
    let initial_poses = poses.clone();
    let initial_points = estimated_points.clone();
    let fixed_pose = poses[2];
    let fixed_point = estimated_points[0];
    let fixed_poses = [false, false, true];
    let mut fixed_points = vec![false; truth_points.len()];
    fixed_points[0] = true;
    let sqrt_pose = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { 2.0 } else { 0.0 })
    });
    let sqrt_point = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { 4.0 } else { 0.0 })
    });
    let pose_priors = [PosePrior {
        camera: 1,
        mean: truth_poses[1],
        sqrt_information: sqrt_pose,
    }];
    let point_priors = [PointPrior {
        point: 1,
        mean: truth_points[1],
        sqrt_information: sqrt_point,
    }];
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let gauge = ScaleGauge::FixedTranslationComponent {
        camera: 1,
        component: TranslationComponent::X,
    };
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let priors = BundleAdjustmentPriors {
        poses: &pose_priors,
        points: &point_priors,
        intrinsics: &[],
    };
    let noise = NoiseModel::isotropic(1.5);
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(3, truth_points.len())
        .expect("small workspace must fit");
    let problem = CalibratedBundleAdjustmentProblem::new(measurements, gauge).with_noise(noise);
    let report = refine_ba_multi_view_dense_with_factors(
        &mut poses,
        &mut estimated_points,
        problem.measurements(),
        problem.gauge(),
        fixed,
        priors,
        &mut workspace,
        Options::l2(80),
        problem.noise(),
    )
    .expect("typed factors form a valid borrowed problem");

    assert_eq!(poses[2], fixed_pose);
    assert_eq!(estimated_points[0], fixed_point);
    assert_eq!(poses[1].translation[0], truth_poses[1].translation[0]);
    assert_eq!(report.fixed_translation_component, Some(0.62));
    assert_eq!(report.diagnostics.pose_dimension, 5);
    assert!((report.initial_cost - 2_667.548_105_962_842).abs() <= 1e-8);
    assert!(report.final_cost <= 1e-16, "{report:#?}");
    const FINAL_STATE_TOLERANCE: f64 = 1e-9;
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

    let mut warm_poses = initial_poses.clone();
    let mut warm_points = initial_points.clone();
    let warm_report = refine_ba_multi_view_dense_with_factors(
        &mut warm_poses,
        &mut warm_points,
        problem.measurements(),
        problem.gauge(),
        fixed,
        priors,
        &mut workspace,
        Options::l2(80),
        problem.noise(),
    )
    .expect("a reused workspace must solve the same fixture");
    let mut fresh_poses = initial_poses.clone();
    let mut fresh_points = initial_points.clone();
    let mut fresh_workspace = DenseBundleAdjustmentWorkspace::try_new(3, truth_points.len())
        .expect("small workspace must fit");
    let fresh_report = refine_ba_multi_view_dense_with_factors(
        &mut fresh_poses,
        &mut fresh_points,
        problem.measurements(),
        problem.gauge(),
        fixed,
        priors,
        &mut fresh_workspace,
        Options::l2(80),
        problem.noise(),
    )
    .expect("a fresh workspace must solve the same fixture");
    assert_eq!(warm_report, fresh_report);
    assert_eq!(warm_poses, fresh_poses);
    assert_eq!(warm_points, fresh_points);

    let mut invalid_observations = observations.clone();
    invalid_observations[0].point[0] = f64::NAN;
    let mut rollback_poses = initial_poses.clone();
    let mut rollback_points = initial_points.clone();
    let rollback_error = refine_ba_multi_view_dense_with_factors(
        &mut rollback_poses,
        &mut rollback_points,
        MultiViewMeasurements::Pixel {
            observations: &invalid_observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        gauge,
        fixed,
        priors,
        &mut workspace,
        Options::l2(80),
        noise,
    );
    assert_eq!(
        rollback_error,
        Err(MultiViewBundleAdjustmentError::NonFiniteMeasurement { observation: 0 })
    );
    assert_eq!(rollback_poses, initial_poses);
    assert_eq!(rollback_points, initial_points);
    let covariance = covariance_ba_multi_view_dense_with_factors(
        &poses,
        &estimated_points,
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        FixedVariableBlocks {
            camera_poses: &fixed_poses,
            points: &fixed_points,
            intrinsics: &[],
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &pose_priors,
            points: &point_priors,
            intrinsics: &[],
        },
        &mut workspace,
        Loss::L2,
        NoiseModel::Unit,
    )
    .expect("factor-aware posterior covariance is full rank");
    assert_eq!(covariance.camera_dimension, 5);
    assert_eq!(covariance.point_covariances[0], [[0.0; 3]; 3]);
    assert_eq!(covariance.point_nees(0, &[0.0; 3]), None);
    let cauchy_covariance = covariance_ba_multi_view_dense_with_factors(
        &poses,
        &estimated_points,
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
        FixedVariableBlocks {
            camera_poses: &fixed_poses,
            points: &fixed_points,
            intrinsics: &[],
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &pose_priors,
            points: &point_priors,
            intrinsics: &[],
        },
        &mut workspace,
        Loss::Cauchy { scale: 2.0 },
        NoiseModel::Unit,
    )
    .expect("loss-normalized prior covariance is full rank");
    for (l2, cauchy) in covariance
        .camera_covariance
        .iter()
        .zip(&cauchy_covariance.camera_covariance)
    {
        assert!((l2 - cauchy).abs() <= 1e-8 * l2.abs().max(1.0));
    }
}

#[test]
fn three_view_pixel_ba_preserves_the_selected_baseline() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.045), [0.62, 0.015, 0.04]),
        RelativePose::new(rotation_y(-0.055), [-0.53, 0.045, 0.09]),
    ];
    let truth_points = points();
    let camera = PinholeCamera::new(720.0, 710.0, 320.0, 240.0).with_skew(0.2);
    let cameras = [camera; 3];
    let mut observations = Vec::new();
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for (camera_index, pose) in truth_poses.iter().enumerate() {
            let value = transform(pose, point);
            observations.push(PixelObservation::new(
                camera_index,
                [
                    (camera.fx * value[0] + camera.skew * value[1]) / value[2] + camera.cx,
                    camera.fy * value[1] / value[2] + camera.cy,
                ],
            ));
        }
        offsets.push(observations.len());
    }
    let (mut poses, mut estimated_points) = perturb(&truth_poses, &truth_points);
    let baseline = norm(&truth_poses[1].translation);
    let initial_norm = norm(&poses[1].translation);
    poses[1].translation = poses[1]
        .translation
        .map(|value| value * baseline / initial_norm);
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(3, truth_points.len())
        .expect("test workspace dimensions are valid");
    let report = refine_ba_multi_view_dense(
        &mut poses,
        &mut estimated_points,
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("synthetic problem is valid");

    assert!(report.final_rmse < 1e-7, "{report:#?}");
    assert!(report.final_cost < report.initial_cost * 1e-12);
    assert!(report.final_damping.is_finite() && report.final_damping > 0.0);
    assert!(report.last_gain_ratio.is_none_or(f64::is_finite));
    assert!((norm(&poses[1].translation) - baseline).abs() < 2e-13);
    assert_eq!(poses[0], identity_pose());
    assert_eq!(report.diagnostics.pose_dimension, 11);
    assert_eq!(report.diagnostics.expected_gauge_nullity, 0);
    assert!(report.diagnostics.is_well_constrained(), "{report:#?}");
}

#[test]
fn four_view_angular_ba_accepts_sparse_tracks_and_reports_free_scale() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.035), [0.58, 0.02, 0.06]),
        RelativePose::new(rotation_y(-0.05), [-0.48, 0.05, 0.1]),
        RelativePose::new(rotation_y(0.075), [0.16, -0.34, 0.08]),
    ];
    let truth_points = points();
    let mut observations = Vec::new();
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for (point_index, point) in truth_points.iter().enumerate() {
        for (camera, pose) in truth_poses.iter().enumerate() {
            if camera == 3 && point_index % 3 == 0 || camera == 2 && point_index % 4 == 0 {
                continue;
            }
            let value = transform(pose, point);
            let length = norm(&value);
            observations.push(AngularObservation::new(
                camera,
                value.map(|coordinate| coordinate / length),
            ));
        }
        offsets.push(observations.len());
    }
    let (mut poses, mut estimated_points) = perturb(&truth_poses, &truth_points);
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(4, truth_points.len())
        .expect("test workspace dimensions are valid");
    let report = refine_ba_multi_view_dense(
        &mut poses,
        &mut estimated_points,
        MultiViewMeasurements::Angular {
            observations: &observations,
            point_offsets: &offsets,
        },
        ScaleGauge::Free,
        &mut workspace,
        Options::l2(80),
        NoiseModel::Unit,
    )
    .expect("synthetic problem is valid");

    assert!(report.final_rmse < 1e-9, "{report:#?}");
    assert!(report.final_cost < report.initial_cost * 1e-12);
    assert_eq!(report.diagnostics.pose_dimension, 18);
    assert_eq!(report.diagnostics.expected_gauge_nullity, 1);
    assert_eq!(
        report.diagnostics.additional_camera_nullity(),
        0,
        "{report:#?}"
    );
    assert!(report.diagnostics.is_well_constrained(), "{report:#?}");
}

#[test]
fn fixed_gauge_solution_is_invariant_to_scene_units() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.045), [0.62, 0.015, 0.04]),
        RelativePose::new(rotation_y(-0.055), [-0.53, 0.045, 0.09]),
    ];
    let truth_points = points();
    let (initial_poses, initial_points) = perturb(&truth_poses, &truth_points);
    let camera = PinholeCamera::new(720.0, 710.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut observations = Vec::new();
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for (camera_index, pose) in truth_poses.iter().enumerate() {
            let value = transform(pose, point);
            observations.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * value[0] / value[2] + camera.cx,
                    camera.fy * value[1] / value[2] + camera.cy,
                ],
            ));
        }
        offsets.push(observations.len());
    }

    let mut reference: Option<(usize, f64)> = None;
    for scale in [1e-6, 1.0, 1e6] {
        let mut poses = initial_poses.clone();
        for pose in poses.iter_mut().skip(1) {
            pose.translation = pose.translation.map(|value| scale * value);
        }
        let mut scaled_points: Vec<_> = initial_points
            .iter()
            .map(|point| point.map(|value| scale * value))
            .collect();
        let mut workspace = DenseBundleAdjustmentWorkspace::try_new(3, scaled_points.len())
            .expect("test workspace dimensions are valid");
        let report = refine_ba_multi_view_dense(
            &mut poses,
            &mut scaled_points,
            MultiViewMeasurements::Pixel {
                observations: &observations,
                point_offsets: &offsets,
                cameras: &cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            &mut workspace,
            Options::l2(40),
            NoiseModel::Unit,
        )
        .expect("uniform scene scaling preserves the BA problem");
        assert!(report.final_rmse < 1e-7, "scale={scale}, {report:#?}");
        assert_eq!(report.diagnostics.pose_rank, 11, "scale={scale}");
        assert!(!matches!(report.status, RefinementStatus::Degenerate));
        if let Some((iterations, rcond)) = reference {
            assert!(
                report.iteration_count.abs_diff(iterations) <= 1,
                "scale={scale}, {report:#?}"
            );
            assert!(
                (report
                    .diagnostics
                    .observable_camera_hessian_reciprocal_condition_number
                    / rcond
                    - 1.0)
                    .abs()
                    < 1e-5,
                "scale={scale}, {report:#?}"
            );
        } else {
            reference = Some((
                report.iteration_count,
                report
                    .diagnostics
                    .observable_camera_hessian_reciprocal_condition_number,
            ));
        }
    }
}

#[test]
fn invalid_gauge_and_out_of_range_cameras_are_typed_errors() {
    let mut poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.02), [0.5, 0.0, 0.0]),
        RelativePose::new(rotation_y(-0.02), [-0.5, 0.0, 0.0]),
    ];
    let mut points = [[0.0, 0.0, 4.0]; 4];
    let observations = [AngularObservation::new(0, [0.0, 0.0, 1.0]); 8];
    let offsets = [0, 2, 4, 6, 8];
    let measurements = MultiViewMeasurements::Angular {
        observations: &observations,
        point_offsets: &offsets,
    };
    let mut workspace =
        DenseBundleAdjustmentWorkspace::try_new(3, 4).expect("test workspace dimensions are valid");

    let gauge_error = refine_ba_multi_view_dense(
        &mut poses,
        &mut points,
        measurements,
        ScaleGauge::FixedBaseline { camera: 0 },
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert_eq!(
        gauge_error,
        Err(MultiViewBundleAdjustmentError::InvalidGaugeCamera { camera: 0 })
    );

    let invalid_observations = [AngularObservation::new(3, [0.0, 0.0, 1.0]); 8];
    let invalid_measurements = MultiViewMeasurements::Angular {
        observations: &invalid_observations,
        point_offsets: &offsets,
    };
    let track_error = refine_ba_multi_view_dense(
        &mut poses,
        &mut points,
        invalid_measurements,
        ScaleGauge::Free,
        &mut workspace,
        Options::l2(1),
        NoiseModel::Unit,
    );
    assert_eq!(
        track_error,
        Err(MultiViewBundleAdjustmentError::InvalidObservationCamera {
            observation: 0,
            camera: 3,
        })
    );
}

#[test]
fn coincident_cameras_are_reported_as_degenerate() {
    let mut poses = [identity_pose(); 3];
    let mut points = [
        [-0.3, -0.2, 4.0],
        [0.4, -0.1, 4.4],
        [-0.2, 0.3, 4.8],
        [0.35, 0.25, 5.2],
    ];
    let mut observations = Vec::new();
    let mut offsets = Vec::with_capacity(points.len() + 1);
    offsets.push(0);
    for point in &points {
        let length = norm(point);
        let bearing = point.map(|coordinate| coordinate / length);
        for camera in 0..3 {
            observations.push(AngularObservation::new(camera, bearing));
        }
        offsets.push(observations.len());
    }
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(3, points.len())
        .expect("test workspace dimensions are valid");
    let report = refine_ba_multi_view_dense(
        &mut poses,
        &mut points,
        MultiViewMeasurements::Angular {
            observations: &observations,
            point_offsets: &offsets,
        },
        ScaleGauge::Free,
        &mut workspace,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect("coincident cameras are a numerical degeneracy, not invalid input");

    assert_eq!(report.status, RefinementStatus::Degenerate);
    assert_eq!(report.diagnostics.weak_point_count, points.len());
    assert!(!report.diagnostics.is_well_constrained());
    assert!(report.final_cost.is_finite());
    assert!(report.final_rmse.is_finite());
}

#[test]
fn fixed_gauge_covariance_is_opt_in_whitened_and_loss_normalized() {
    let poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.045), [0.62, 0.015, 0.04]),
        RelativePose::new(rotation_y(-0.055), [-0.53, 0.045, 0.09]),
    ];
    let points = points();
    let camera = PinholeCamera::new(720.0, 710.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut observations = Vec::new();
    let mut offsets = Vec::with_capacity(points.len() + 1);
    offsets.push(0);
    for point in &points {
        for (camera_index, pose) in poses.iter().enumerate() {
            let value = transform(pose, point);
            observations.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * value[0] / value[2] + camera.cx,
                    camera.fy * value[1] / value[2] + camera.cy,
                ],
            ));
        }
        offsets.push(observations.len());
    }
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(3, points.len())
        .expect("test workspace dimensions are valid");
    let unit = covariance_ba_multi_view_dense(
        &poses,
        &points,
        measurements,
        gauge,
        &mut workspace,
        Loss::L2,
        NoiseModel::Unit,
    )
    .expect("exact fixed-gauge geometry has covariance");
    let doubled = covariance_ba_multi_view_dense(
        &poses,
        &points,
        measurements,
        gauge,
        &mut workspace,
        Loss::L2,
        NoiseModel::isotropic(2.0),
    )
    .expect("noise scaling preserves rank");
    let cauchy = covariance_ba_multi_view_dense(
        &poses,
        &points,
        measurements,
        gauge,
        &mut workspace,
        Loss::Cauchy { scale: 2.0 },
        NoiseModel::Unit,
    )
    .expect("Cauchy information is normalized at zero residual");
    assert_eq!(
        cauchy.uncertainty_model,
        UncertaintyModel::RobustFixedWeightLinearized
    );

    assert_eq!(unit.camera_dimension, 11);
    assert_eq!(unit.camera_parameter_range(0), Some(0..0));
    assert_eq!(unit.camera_parameter_range(1), Some(0..5));
    assert_eq!(unit.camera_parameter_range(2), Some(5..11));
    assert!(unit.diagnostics.is_well_constrained());
    assert_eq!(unit.point_covariances.len(), points.len());
    for ((base, scaled), robust) in unit
        .camera_covariance
        .iter()
        .zip(&doubled.camera_covariance)
        .zip(&cauchy.camera_covariance)
    {
        assert!((scaled / base - 4.0).abs() < 2e-8);
        assert!((robust / base - 1.0).abs() < 2e-8);
    }
    assert_eq!(unit.camera_nees(&[0.0; 11]), Some(0.0));
    assert_eq!(unit.point_nees(0, &[0.0; 3]), Some(0.0));
    assert_eq!(unit.camera_nees(&[0.0; 10]), None);
    assert_eq!(
        covariance_ba_multi_view_dense(
            &poses,
            &points,
            measurements,
            ScaleGauge::Free,
            &mut workspace,
            Loss::L2,
            NoiseModel::Unit,
        ),
        Err(MultiViewBundleAdjustmentCovarianceError::FreeScaleGauge)
    );
}

#[test]
fn fixed_gauge_covariance_has_calibrated_nonlinear_nees() {
    const TRIALS: usize = 80;
    let truth_poses = [
        identity_pose(),
        RelativePose::new(rotation_y(0.045), [0.62, 0.015, 0.04]),
        RelativePose::new(rotation_y(-0.055), [-0.53, 0.045, 0.09]),
    ];
    let truth_points = points();
    let camera = PinholeCamera::new(720.0, 710.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut exact_pixels = Vec::with_capacity(3 * truth_points.len());
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for (camera_index, pose) in truth_poses.iter().enumerate() {
            let value = transform(pose, point);
            exact_pixels.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * value[0] / value[2] + camera.cx,
                    camera.fy * value[1] / value[2] + camera.cy,
                ],
            ));
        }
        offsets.push(exact_pixels.len());
    }
    let sigma = 0.2;
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(3, truth_points.len())
        .expect("test workspace dimensions are valid");
    let mut rng = StdRng::seed_from_u64(0xba5e_cafe);
    let mut camera_nees = 0.0;
    let mut point_nees = 0.0;
    for _ in 0..TRIALS {
        let mut observations = exact_pixels.clone();
        for observation in &mut observations {
            let x: f64 = StandardNormal.sample(&mut rng);
            let y: f64 = StandardNormal.sample(&mut rng);
            observation.point[0] += sigma * x;
            observation.point[1] += sigma * y;
        }
        let measurements = MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        };
        let mut estimated_poses = truth_poses;
        let mut estimated_points = truth_points.clone();
        let report = refine_ba_multi_view_dense(
            &mut estimated_poses,
            &mut estimated_points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(30),
            NoiseModel::isotropic(sigma),
        )
        .expect("noisy synthetic problem remains valid");
        assert!(report.diagnostics.is_well_constrained(), "{report:#?}");
        let covariance = covariance_ba_multi_view_dense(
            &estimated_poses,
            &estimated_points,
            measurements,
            gauge,
            &mut workspace,
            Loss::L2,
            NoiseModel::isotropic(sigma),
        )
        .expect("converged fixed-gauge problem has covariance");
        let camera_error = camera_local_error(
            &estimated_poses,
            &truth_poses,
            covariance.normalization_scale(),
        );
        camera_nees += covariance
            .camera_nees(&camera_error)
            .expect("camera error has the reported dimension");
        for (point_index, (estimate, truth)) in
            estimated_points.iter().zip(&truth_points).enumerate()
        {
            let error = std::array::from_fn(|coordinate| {
                (truth[coordinate] - estimate[coordinate]) / covariance.normalization_scale()
            });
            point_nees += covariance
                .point_nees(point_index, &error)
                .expect("each estimated point has a marginal");
        }
    }
    let mean_camera_nees = camera_nees / TRIALS as f64;
    let mean_point_nees = point_nees / (TRIALS * truth_points.len()) as f64;
    println!(
        "BA_NEES,trials={TRIALS},camera={mean_camera_nees:.6},camera_dof=11,point={mean_point_nees:.6},point_dof=3"
    );
    assert!(
        (mean_camera_nees - 11.0).abs() < 2.2,
        "camera mean NEES {mean_camera_nees} should be close to 11"
    );
    assert!(
        (mean_point_nees - 3.0).abs() < 0.45,
        "point mean NEES {mean_point_nees} should be close to 3"
    );
}

fn camera_local_error(
    estimate: &[RelativePose; 3],
    truth: &[RelativePose; 3],
    scene_scale: f64,
) -> [f64; 11] {
    let mut error = [0.0; 11];
    let fixed_rotation = manifold::mat_mul(&truth[1].rotation, &transpose(&estimate[1].rotation));
    error[..3].copy_from_slice(&manifold::log_so3(&fixed_rotation));
    let estimate_direction = normalize(estimate[1].translation);
    let truth_direction = normalize(truth[1].translation);
    let basis = manifold::s2_tangent_basis(&estimate_direction);
    let cosine = dot(&estimate_direction, &truth_direction).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let tangent = if angle < 1e-10 {
        std::array::from_fn(|coordinate| {
            truth_direction[coordinate] - estimate_direction[coordinate]
        })
    } else {
        let scale = angle / angle.sin();
        std::array::from_fn(|coordinate| {
            scale * (truth_direction[coordinate] - cosine * estimate_direction[coordinate])
        })
    };
    error[3] = dot(&basis[0], &tangent);
    error[4] = dot(&basis[1], &tangent);

    let free_rotation = manifold::mat_mul(&truth[2].rotation, &transpose(&estimate[2].rotation));
    error[5..8].copy_from_slice(&manifold::log_so3(&free_rotation));
    for coordinate in 0..3 {
        error[8 + coordinate] =
            (truth[2].translation[coordinate] - estimate[2].translation[coordinate]) / scene_scale;
    }
    error
}

fn transpose(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let length = norm(&vector);
    vector.map(|value| value / length)
}

fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}
