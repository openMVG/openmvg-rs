// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::multi_view::self_calibration::{
    RadialCamera, SelfCalibratingBundleAdjustmentProblem, SelfCalibratingMeasurements,
};
use openmvg_ba::kernel::multi_view::{
    MultiViewObservationResidual, MultiViewTrackClassification, MultiViewTrackDiagnostics,
};
use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::AngularObservation,
    multi_view::CalibratedBundleAdjustmentProblem, multi_view::MultiViewBundleAdjustmentError,
    multi_view::MultiViewMeasurements, multi_view::MultiViewTriangulationReport,
    multi_view::PixelObservation, multi_view::ScaleGauge,
};
use openmvg_ba::numerics::refine::{Loss, NoiseModel, Options, RefinementStatus};

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn poses() -> [RelativePose; 3] {
    [
        RelativePose::new(IDENTITY, [0.0; 3]),
        RelativePose::new(IDENTITY, [0.7, 0.0, 0.0]),
        RelativePose::new(IDENTITY, [0.0, -0.5, 0.1]),
    ]
}

fn points() -> [[f64; 3]; 4] {
    [
        [-0.4, -0.2, 3.8],
        [0.3, -0.1, 4.2],
        [-0.2, 0.5, 4.7],
        [0.6, 0.3, 5.1],
    ]
}

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

#[test]
fn calibrated_pixel_and_angular_tracks_triangulate_without_map_mutation() {
    let poses = poses();
    let points = points();
    let camera = PinholeCamera::new(700.0, 690.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut pixels = Vec::new();
    let mut angular = Vec::new();
    let mut offsets = vec![0];
    for point in &points {
        for (camera_index, pose) in poses.iter().enumerate() {
            let camera_point = transform(pose, point);
            pixels.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * camera_point[0] / camera_point[2] + camera.cx,
                    camera.fy * camera_point[1] / camera_point[2] + camera.cy,
                ],
            ));
            angular.push(AngularObservation::new(camera_index, camera_point));
        }
        offsets.push(pixels.len());
    }

    for measurements in [
        MultiViewMeasurements::Pixel {
            observations: &pixels,
            point_offsets: &offsets,
            cameras: &cameras,
        },
        MultiViewMeasurements::Angular {
            observations: &angular,
            point_offsets: &offsets,
        },
    ] {
        let problem = CalibratedBundleAdjustmentProblem::new(measurements, ScaleGauge::Free)
            .with_noise(NoiseModel::Unit);
        let mut reports = vec![MultiViewTriangulationReport::default(); points.len()];
        problem
            .triangulate_tracks(&poses, &[], &mut reports, Options::l2(15))
            .unwrap();
        for (report, truth) in reports.iter().zip(&points) {
            assert!(
                matches!(
                    report.status,
                    RefinementStatus::Converged(_) | RefinementStatus::MaxIterations
                ),
                "{report:#?}"
            );
            assert_eq!(report.observation_count, 3);
            assert!(report.final_cost <= 1e-18, "{report:#?}");
            assert!(report.point_hessian_reciprocal_condition_number > 0.0);
            assert!(report.is_usable_proposal());
            for (&actual, &expected) in report.point.iter().zip(truth) {
                assert!((actual - expected).abs() <= 1e-10);
            }
        }

        let selection = [true, false, true, false];
        problem
            .triangulate_selected_tracks(&poses, &[], &selection, &mut reports, Options::l2(15))
            .unwrap();
        for (index, report) in reports.iter().enumerate() {
            if selection[index] {
                assert!(report.point_hessian_reciprocal_condition_number > 0.0);
                assert!(report.is_usable_proposal());
            } else {
                assert_eq!(*report, MultiViewTriangulationReport::default());
                assert!(!report.is_usable_proposal());
            }
        }
        let error = problem
            .triangulate_selected_tracks(
                &poses,
                &[],
                &selection[..selection.len() - 1],
                &mut reports,
                Options::l2(1),
            )
            .expect_err("a short track-selection mask must be rejected");
        assert_eq!(
            error,
            MultiViewBundleAdjustmentError::TrackSelectionCountMismatch {
                expected: points.len(),
                actual: points.len() - 1,
            }
        );
        let error = problem
            .triangulate_selected_tracks(&poses, &[], &[], &mut reports, Options::l2(1))
            .expect_err("empty means all tracks only in triangulate_tracks");
        assert_eq!(
            error,
            MultiViewBundleAdjustmentError::TrackSelectionCountMismatch {
                expected: points.len(),
                actual: 0,
            }
        );
        problem
            .triangulate_selected_tracks(&poses, &[], &[false; 4], &mut reports, Options::l2(1))
            .unwrap();
        assert!(
            reports
                .iter()
                .all(|report| *report == MultiViewTriangulationReport::default())
        );
    }
}

#[test]
fn triangulation_rejects_a_retained_subset_without_parallax() {
    let poses = [
        RelativePose::new(IDENTITY, [0.0; 3]),
        RelativePose::new(IDENTITY, [0.0; 3]),
        RelativePose::new(IDENTITY, [0.8, 0.0, 0.0]),
    ];
    let points = points();
    let camera = PinholeCamera::new(700.0, 700.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut observations = Vec::new();
    let mut offsets = vec![0];
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
    let problem = CalibratedBundleAdjustmentProblem::new(measurements, ScaleGauge::Free);
    let mut inliers = vec![true; observations.len()];
    inliers[2] = false;
    let mut reports = vec![MultiViewTriangulationReport::default(); points.len()];
    problem
        .triangulate_tracks(&poses, &inliers, &mut reports, Options::l2(5))
        .unwrap();

    assert_eq!(reports[0].status, RefinementStatus::Degenerate);
    assert_eq!(reports[0].observation_count, 2);
    assert_eq!(reports[0].point_hessian_reciprocal_condition_number, 0.0);
    assert!(!reports[0].is_usable_proposal());
    assert!(
        reports[1..]
            .iter()
            .all(|report| report.status != RefinementStatus::Degenerate)
    );
}

#[test]
fn a_single_two_view_track_triangulates_in_an_arbitrary_world_frame() {
    let poses = [
        RelativePose::new(IDENTITY, [0.2, -0.1, 0.3]),
        RelativePose::new(IDENTITY, [0.9, -0.1, 0.3]),
    ];
    let truth = [0.4, 0.25, 4.5];
    let observations: Vec<_> = poses
        .iter()
        .enumerate()
        .map(|(camera, pose)| AngularObservation::new(camera, transform(pose, &truth)))
        .collect();
    let offsets = [0, observations.len()];
    let problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Angular {
            observations: &observations,
            point_offsets: &offsets,
        },
        ScaleGauge::Free,
    );
    let mut reports = [MultiViewTriangulationReport::default()];

    problem
        .triangulate_tracks(&poses, &[], &mut reports, Options::l2(10))
        .unwrap();

    assert_ne!(reports[0].status, RefinementStatus::Degenerate);
    for (&actual, &expected) in reports[0].point.iter().zip(&truth) {
        assert!((actual - expected).abs() <= 1e-10, "{reports:#?}");
    }
}

#[test]
fn grouped_radial_lifecycle_and_triangulation_share_the_native_model() {
    let poses = poses();
    let points = points();
    let groups = [
        RadialCamera::new(710.0, 0.045, -0.018),
        RadialCamera::new(760.0, -0.032, 0.011),
    ];
    let camera_to_group = [0, 0, 1];
    let mut observations = Vec::new();
    let mut offsets = vec![0];
    for point in &points {
        for (camera, pose) in poses.iter().enumerate() {
            let value = transform(pose, point);
            let normalized = [value[0] / value[2], value[1] / value[2]];
            let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
            let radial = groups[camera_to_group[camera]];
            let distortion =
                1.0 + radial.k1 * radius_squared + radial.k2 * radius_squared * radius_squared;
            observations.push(PixelObservation::new(
                camera,
                [
                    radial.focal * distortion * normalized[0],
                    radial.focal * distortion * normalized[1],
                ],
            ));
        }
        offsets.push(observations.len());
    }
    observations[2].point[0] += 30.0;
    let problem = SelfCalibratingBundleAdjustmentProblem::new_grouped(
        SelfCalibratingMeasurements::new(&observations, &offsets),
        &groups,
        &camera_to_group,
        ScaleGauge::Free,
    );
    let mut residuals = vec![MultiViewObservationResidual::default(); observations.len()];
    let mut diagnostics = vec![MultiViewTrackDiagnostics::default(); points.len()];
    let mut inliers = vec![false; observations.len()];
    let mut classifications = vec![MultiViewTrackClassification::Stable; points.len()];

    let lifecycle = problem
        .evaluate_track_lifecycle(
            &poses,
            &points,
            Loss::L2,
            5.0,
            &mut residuals,
            &mut diagnostics,
            &mut inliers,
            &mut classifications,
        )
        .unwrap();
    assert_eq!(lifecycle.lifecycle.rejected_observation_count, 1);
    assert_eq!(
        classifications[0],
        MultiViewTrackClassification::Retriangulate
    );
    assert_eq!(&inliers[..3], &[true, true, false]);

    let selection = [true, false, false, false];
    let mut reports = vec![MultiViewTriangulationReport::default(); points.len()];
    problem
        .triangulate_selected_tracks(&poses, &inliers, &selection, &mut reports, Options::l2(15))
        .unwrap();
    assert!(reports[0].is_usable_proposal(), "{:#?}", reports[0]);
    assert_eq!(reports[0].observation_count, 2);
    for (&actual, &expected) in reports[0].point.iter().zip(&points[0]) {
        assert!((actual - expected).abs() <= 1.0e-9, "{:#?}", reports[0]);
    }
    assert!(
        reports[1..]
            .iter()
            .all(|report| *report == MultiViewTriangulationReport::default())
    );
}
