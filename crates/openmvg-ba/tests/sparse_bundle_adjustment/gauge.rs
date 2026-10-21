// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gauge nullity, component gauges, and degenerate camera layouts.

use super::*;

#[test]
fn sparse_component_gauge_preserves_exact_coordinate_and_full_rank() {
    let Fixture {
        truth_poses,
        truth_points,
        observations,
        offsets,
        cameras,
    } = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let gauge = ScaleGauge::FixedTranslationComponent {
        camera: 1,
        component: TranslationComponent::X,
    };
    let (mut poses, mut points) = perturbed(&truth_poses, &truth_points);
    let fixed_x = poses[1].translation[0];
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new(4, measurements, gauge)
        .expect("component-gauge topology is valid");
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .expect("component-gauge fixture is valid");

    assert_eq!(poses[1].translation[0], fixed_x);
    assert_eq!(report.fixed_translation_component, Some(fixed_x));
    assert_eq!(
        report.diagnostics.pose_rank,
        SparseCameraRankAssessment::FullRank
    );
    assert!(report.final_damping.is_finite() && report.final_damping > 0.0);
    assert!(report.last_gain_ratio.is_none_or(f64::is_finite));
    assert!(report.final_rmse < 1e-7, "{report:#?}");
}

#[test]
fn sparse_free_gauge_reports_only_the_expected_scale_null() {
    let Fixture {
        truth_poses,
        truth_points,
        observations,
        offsets,
        cameras,
    } = fixture();
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let (mut poses, mut points) = perturbed(&truth_poses, &truth_points);
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new(4, measurements, ScaleGauge::Free)
        .expect("fixture topology is valid");
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        ScaleGauge::Free,
        &mut workspace,
        Options::l2(60),
        NoiseModel::Unit,
    )
    .expect("sparse fixture is valid");

    assert!(report.final_rmse < 1e-7, "{report:#?}");
    assert_eq!(
        report.diagnostics.pose_rank,
        SparseCameraRankAssessment::OnlyExpectedScaleGauge,
        "{report:#?}"
    );
    assert!(!matches!(report.status, RefinementStatus::Degenerate));
}

#[test]
fn sparse_solver_reports_coincident_cameras_as_degenerate() {
    let mut poses = vec![identity_pose(); 3];
    let mut points = vec![
        [-0.3, -0.2, 4.0],
        [0.4, -0.1, 4.4],
        [-0.2, 0.3, 4.8],
        [0.35, 0.25, 5.2],
    ];
    let camera = PinholeCamera::new(700.0, 700.0, 320.0, 240.0);
    let cameras = [camera; 3];
    let mut observations = Vec::new();
    let mut offsets = vec![0];
    for point in &points {
        for camera_index in 0..3 {
            observations.push(PixelObservation::new(
                camera_index,
                [
                    camera.fx * point[0] / point[2] + camera.cx,
                    camera.fy * point[1] / point[2] + camera.cy,
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
    let mut workspace = SparseBundleAdjustmentWorkspace::try_new(3, measurements, ScaleGauge::Free)
        .expect("degenerate geometry still has valid structure");
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        ScaleGauge::Free,
        &mut workspace,
        Options::l2(4),
        NoiseModel::Unit,
    )
    .expect("degeneracy is a report status, not an input error");

    assert_eq!(report.status, RefinementStatus::Degenerate);
    assert_eq!(
        report.diagnostics.pose_rank,
        SparseCameraRankAssessment::RankDeficientOrIndefinite
    );
    assert!(report.final_cost.is_finite());
    assert!(report.final_rmse.is_finite());
}
