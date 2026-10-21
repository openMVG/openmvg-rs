// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::AngularObservation,
    multi_view::CalibratedBundleAdjustmentProblem, multi_view::MultiViewBundleAdjustmentError,
    multi_view::MultiViewMeasurements, multi_view::MultiViewObservationResidual,
    multi_view::MultiViewTrackClassification, multi_view::MultiViewTrackDiagnostics,
    multi_view::PixelObservation, multi_view::ScaleGauge,
    multi_view::dense::DenseBundleAdjustmentWorkspace,
    multi_view::dense::refine_ba_multi_view_dense,
};
use openmvg_ba::numerics::refine::{Loss, NoiseModel, Options};

fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        pose.rotation[row][0] * point[0]
            + pose.rotation[row][1] * point[1]
            + pose.rotation[row][2] * point[2]
            + pose.translation[row]
    })
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let inverse_norm = vector
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .recip();
    vector.map(|value| value * inverse_norm)
}

type EvaluationFixture = (
    [RelativePose; 3],
    Vec<[f64; 3]>,
    Vec<PixelObservation>,
    Vec<usize>,
    [PinholeCamera; 3],
);

fn fixture() -> EvaluationFixture {
    let poses = [
        identity_pose(),
        RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.7, 0.05, 0.02],
        ),
        RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-0.55, 0.25, 0.08],
        ),
    ];
    let points = (0..8)
        .map(|index| {
            [
                -0.7 + 0.25 * (index % 4) as f64,
                -0.3 + 0.35 * (index / 4) as f64,
                3.5 + 0.2 * index as f64,
            ]
        })
        .collect::<Vec<_>>();
    let camera = PinholeCamera::new(600.0, 610.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut observations = Vec::with_capacity(3 * points.len());
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
    (poses, points, observations, offsets, cameras)
}

#[test]
fn evaluation_reports_native_and_standardized_residuals_without_changing_topology() {
    let (poses, points, mut observations, offsets, cameras) = fixture();
    observations[1].point[0] += 3.0;
    observations[1].point[1] += 4.0;
    let standard_deviations = vec![2.0; observations.len()];
    let problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
    )
    .with_noise(NoiseModel::per_observation(&standard_deviations));
    let mut residuals = vec![MultiViewObservationResidual::default(); observations.len()];
    let mut tracks = vec![MultiViewTrackDiagnostics::default(); points.len()];

    let summary = problem
        .evaluate(
            &poses,
            &points,
            Loss::Huber { delta: 1.0 },
            &mut residuals,
            &mut tracks,
        )
        .expect("valid tracks must be evaluated");

    assert_eq!(residuals[1].residual, [-3.0, -4.0]);
    assert_eq!(residuals[1].standardized_norm, 2.5);
    assert!((summary.robust_cost - 2.0).abs() < 1e-12, "{summary:#?}");
    assert!((summary.rmse - 5.0 / ((2 * observations.len()) as f64).sqrt()).abs() < 1e-12);
    assert_eq!(summary.maximum_standardized_residual_norm, 2.5);
    assert_eq!(summary.weak_point_count, 0, "{tracks:#?}");
    assert!((tracks[0].rmse - 5.0 / 6.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(tracks[0].maximum_standardized_residual_norm, 2.5);
    assert!(tracks.iter().all(|track| !track.weak));

    let mut observation_inliers = vec![false; observations.len()];
    let mut track_classifications = vec![MultiViewTrackClassification::Stable; points.len()];
    let lifecycle_evaluation = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::Huber { delta: 1.0 },
            2.0,
            &mut residuals,
            &mut tracks,
            &mut observation_inliers,
            &mut track_classifications,
        )
        .expect("evaluated diagnostics must be classifiable");
    let lifecycle = lifecycle_evaluation.lifecycle;
    assert_eq!(lifecycle_evaluation.evaluation, summary);
    assert!(!observation_inliers[1]);
    assert!(
        observation_inliers
            .iter()
            .enumerate()
            .all(|(index, &inlier)| inlier == (index != 1))
    );
    assert_eq!(
        track_classifications[0],
        MultiViewTrackClassification::Retriangulate
    );
    assert!(
        track_classifications[1..]
            .iter()
            .all(|classification| *classification == MultiViewTrackClassification::Stable)
    );
    assert_eq!(lifecycle.retained_observation_count, observations.len() - 1);
    assert_eq!(lifecycle.rejected_observation_count, 1);
    assert_eq!(lifecycle.stable_track_count, points.len() - 1);
    assert_eq!(lifecycle.retriangulation_track_count, 1);
    assert_eq!(lifecycle.weak_geometry_track_count, 0);
    assert_eq!(lifecycle.insufficient_support_track_count, 0);

    let error = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::Huber { delta: 1.0 },
            2.0,
            &mut residuals,
            &mut tracks,
            &mut observation_inliers[..observations.len() - 1],
            &mut track_classifications,
        )
        .expect_err("short observation mask must be rejected");
    assert_eq!(
        error,
        MultiViewBundleAdjustmentError::ObservationInlierMaskCountMismatch {
            expected: observations.len(),
            actual: observations.len() - 1,
        }
    );
    let error = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::Huber { delta: 1.0 },
            2.0,
            &mut residuals,
            &mut tracks,
            &mut observation_inliers,
            &mut track_classifications[..points.len() - 1],
        )
        .expect_err("short track classification scratch must be rejected");
    assert_eq!(
        error,
        MultiViewBundleAdjustmentError::TrackClassificationCountMismatch {
            expected: points.len(),
            actual: points.len() - 1,
        }
    );

    let error = problem
        .evaluate(
            &poses,
            &points,
            Loss::L2,
            &mut residuals[..observations.len() - 1],
            &mut tracks,
        )
        .expect_err("short observation scratch must be rejected");
    assert_eq!(
        error,
        MultiViewBundleAdjustmentError::ObservationResidualCountMismatch {
            expected: observations.len(),
            actual: observations.len() - 1,
        }
    );
    let error = problem
        .evaluate(
            &poses,
            &points,
            Loss::L2,
            &mut residuals,
            &mut tracks[..points.len() - 1],
        )
        .expect_err("short track scratch must be rejected");
    assert_eq!(
        error,
        MultiViewBundleAdjustmentError::TrackDiagnosticCountMismatch {
            expected: points.len(),
            actual: points.len() - 1,
        }
    );
    let mut invalid_points = points.clone();
    invalid_points[0][2] = -1.0;
    let error = problem
        .evaluate(
            &poses,
            &invalid_points,
            Loss::L2,
            &mut residuals,
            &mut tracks,
        )
        .expect_err("invalid projection geometry must stay typed");
    assert_eq!(
        error,
        MultiViewBundleAdjustmentError::PointNotProjectable {
            point: 0,
            camera: 0,
        }
    );
    let error = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::L2,
            f64::NAN,
            &mut residuals,
            &mut tracks,
            &mut observation_inliers,
            &mut track_classifications,
        )
        .expect_err("a non-finite residual gate must be rejected");
    assert_eq!(
        error,
        MultiViewBundleAdjustmentError::Configuration(
            openmvg_ba::numerics::refine::RefinementError::InvalidThreshold
        )
    );
}

#[test]
fn evaluation_preserves_singular_tracks_as_explicit_quality_evidence() {
    let (_, points, _, offsets, cameras) = fixture();
    let poses = [identity_pose(); 3];
    let camera = cameras[0];
    let mut observations = Vec::with_capacity(3 * points.len());
    for point in &points {
        let value = transform(&poses[0], point);
        let pixel = [
            camera.fx * value[0] / value[2] + camera.cx,
            camera.fy * value[1] / value[2] + camera.cy,
        ];
        for camera_index in 0..3 {
            observations.push(PixelObservation::new(camera_index, pixel));
        }
    }
    let problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::Free,
    );
    let mut residuals = vec![MultiViewObservationResidual::default(); observations.len()];
    let mut tracks = vec![MultiViewTrackDiagnostics::default(); points.len()];

    let summary = problem
        .evaluate(&poses, &points, Loss::L2, &mut residuals, &mut tracks)
        .expect("singular geometry is valid diagnostic input");

    assert_eq!(summary.rmse, 0.0);
    assert_eq!(summary.weak_point_count, points.len());
    assert_eq!(
        summary.minimum_point_hessian_reciprocal_condition_number,
        0.0
    );
    assert!(tracks.iter().all(|track| track.weak));
    let mut observation_inliers = vec![false; observations.len()];
    let mut track_classifications = vec![MultiViewTrackClassification::Stable; points.len()];
    let lifecycle = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::L2,
            0.0,
            &mut residuals,
            &mut tracks,
            &mut observation_inliers,
            &mut track_classifications,
        )
        .expect("weak tracks remain classifiable")
        .lifecycle;
    assert_eq!(lifecycle.retriangulation_track_count, 0);
    assert_eq!(lifecycle.weak_geometry_track_count, points.len());
    assert!(
        track_classifications
            .iter()
            .all(|classification| *classification == MultiViewTrackClassification::WeakGeometry)
    );

    let mut rejected_observations = observations.clone();
    rejected_observations[0].point[0] += 1.0;
    rejected_observations[1].point[0] += 1.0;
    let rejected_problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &rejected_observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::Free,
    );
    let lifecycle = rejected_problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::L2,
            0.0,
            &mut residuals,
            &mut tracks,
            &mut observation_inliers,
            &mut track_classifications,
        )
        .expect("under-supported tracks remain classifiable")
        .lifecycle;
    assert_eq!(lifecycle.insufficient_support_track_count, 1);
    assert_eq!(
        track_classifications[0],
        MultiViewTrackClassification::InsufficientSupport
    );
}

#[test]
fn lifecycle_conditions_the_retained_subset_after_rejection() {
    let (mut poses, points, mut observations, offsets, cameras) = fixture();
    // Cameras zero and one provide the same ray geometry. Camera two is the
    // only source of parallax, but its measurement is deliberately rejected.
    poses[1] = poses[0];
    for (point_index, point) in points.iter().enumerate() {
        let value = transform(&poses[0], point);
        observations[3 * point_index] = PixelObservation::new(
            0,
            [
                cameras[0].fx * value[0] / value[2] + cameras[0].cx,
                cameras[0].fy * value[1] / value[2] + cameras[0].cy,
            ],
        );
        observations[3 * point_index + 1] = PixelObservation::new(
            1,
            [
                cameras[1].fx * value[0] / value[2] + cameras[1].cx,
                cameras[1].fy * value[1] / value[2] + cameras[1].cy,
            ],
        );
        observations[3 * point_index + 2].point[0] += 10.0;
    }
    let problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::Free,
    );
    let mut residuals = vec![MultiViewObservationResidual::default(); observations.len()];
    let mut all_tracks = vec![MultiViewTrackDiagnostics::default(); points.len()];
    let _ = problem
        .evaluate(&poses, &points, Loss::L2, &mut residuals, &mut all_tracks)
        .expect("the complete tracks are valid");
    assert!(all_tracks.iter().all(|track| !track.weak));

    let mut retained_tracks = vec![MultiViewTrackDiagnostics::default(); points.len()];
    let mut observation_inliers = vec![false; observations.len()];
    let mut classifications = vec![MultiViewTrackClassification::Stable; points.len()];
    let lifecycle = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::L2,
            1.0,
            &mut residuals,
            &mut retained_tracks,
            &mut observation_inliers,
            &mut classifications,
        )
        .expect("the retained subset is valid")
        .lifecycle;

    assert_eq!(lifecycle.weak_geometry_track_count, points.len());
    assert_eq!(lifecycle.retriangulation_track_count, 0);
    assert!(
        retained_tracks
            .iter()
            .all(|track| { track.observation_count == 2 && track.weak })
    );
    assert!(
        classifications
            .iter()
            .all(|classification| *classification == MultiViewTrackClassification::WeakGeometry)
    );
}

#[test]
fn angular_tracks_use_the_same_borrowed_evaluation_api() {
    let (poses, points, _, offsets, _) = fixture();
    let mut observations = Vec::with_capacity(3 * points.len());
    for point in &points {
        for (camera, pose) in poses.iter().enumerate() {
            observations.push(AngularObservation::new(
                camera,
                normalize(transform(pose, point)),
            ));
        }
    }
    let problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Angular {
            observations: &observations,
            point_offsets: &offsets,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    let mut residuals = vec![MultiViewObservationResidual::default(); observations.len()];
    let mut tracks = vec![MultiViewTrackDiagnostics::default(); points.len()];

    let summary = problem
        .evaluate(&poses, &points, Loss::L2, &mut residuals, &mut tracks)
        .expect("angular tracks must share the diagnostic path");

    assert!(summary.rmse < 1e-15, "{summary:#?}");
    assert_eq!(summary.weak_point_count, 0, "{tracks:#?}");
    assert!(tracks.iter().all(|track| !track.weak));
}

#[test]
fn evaluation_matches_the_dense_solvers_final_observation_evidence() {
    let (mut poses, mut points, observations, offsets, cameras) = fixture();
    let problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(poses.len(), points.len())
        .expect("small workspace must fit");
    let report = refine_ba_multi_view_dense(
        &mut poses,
        &mut points,
        problem.measurements(),
        problem.gauge(),
        &mut workspace,
        Options::l2(2),
        problem.noise(),
    )
    .expect("exact geometry must be accepted");
    let mut residuals = vec![MultiViewObservationResidual::default(); observations.len()];
    let mut tracks = vec![MultiViewTrackDiagnostics::default(); points.len()];
    let summary = problem
        .evaluate(&poses, &points, report.loss, &mut residuals, &mut tracks)
        .expect("the final state must be evaluable");

    assert!((summary.robust_cost - report.final_cost).abs() < 1e-20);
    assert!((summary.rmse - report.final_rmse).abs() < 1e-20);
    assert_eq!(
        summary.weak_point_count,
        report.diagnostics.weak_point_count
    );
    let expected = report
        .diagnostics
        .minimum_point_hessian_reciprocal_condition_number;
    assert!(
        (summary.minimum_point_hessian_reciprocal_condition_number - expected).abs()
            <= 1e-12 * expected,
        "summary={summary:#?}, report={report:#?}"
    );
}
