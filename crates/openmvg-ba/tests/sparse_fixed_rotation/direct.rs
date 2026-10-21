// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct translation-only solves, dense references, gauges, and measurement models.

use super::*;

#[test]
fn explicit_full_se3_preserves_the_legacy_sparse_path() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let make_legacy = || {
        SparseBundleAdjustmentWorkspace::try_new_with_options(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            SparseWorkspaceOptions::direct(),
        )
        .expect("legacy full-pose workspace is valid")
    };
    let make_explicit = || {
        SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_variables(
            fixture.initial_poses.len(),
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            CameraPoseVariables::RotationAndTranslation,
            SparseWorkspaceOptions::direct(),
        )
        .expect("explicit full-pose workspace is valid")
    };
    let run = |mut workspace: SparseBundleAdjustmentWorkspace| {
        let mut poses = fixture.initial_poses.clone();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(20),
            NoiseModel::Unit,
        )
        .expect("full-pose sparse solve succeeds");
        (poses, points, report)
    };

    let legacy = run(make_legacy());
    let explicit = run(make_explicit());
    assert_eq!(legacy.0, explicit.0);
    assert_eq!(legacy.1, explicit.1);
    assert_eq!(legacy.2.status, explicit.2.status);
    assert_eq!(legacy.2.iteration_count, explicit.2.iteration_count);
    assert_eq!(
        legacy.2.initial_cost.to_bits(),
        explicit.2.initial_cost.to_bits()
    );
    assert_eq!(
        legacy.2.final_cost.to_bits(),
        explicit.2.final_cost.to_bits()
    );
}

#[test]
fn translation_only_sparse_ba_preserves_every_rotation_bit() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let mut workspace =
        translation_only_workspace(fixture.initial_poses.len(), measurements, gauge);
    let mut poses = fixture.initial_poses.clone();
    let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
    let mut points = fixture.initial_points.clone();
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .expect("translation-only solve succeeds");

    assert!(report.final_cost < report.initial_cost);
    for (pose, rotation) in poses.iter().zip(rotations) {
        assert_eq!(pose.rotation, rotation);
    }
    assert_eq!(poses[0], fixture.initial_poses[0]);
}

#[test]
fn translation_only_respects_both_supported_scale_gauges() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    for gauge in [
        ScaleGauge::FixedBaseline { camera: 1 },
        ScaleGauge::FixedTranslationComponent {
            camera: 2,
            component: TranslationComponent::Y,
        },
    ] {
        let mut poses = fixture.initial_poses.clone();
        let initial_baseline = norm(poses[1].translation);
        let initial_component = poses[2].translation[1];
        let mut points = fixture.initial_points.clone();
        let mut workspace = translation_only_workspace(poses.len(), measurements, gauge);
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(20),
            NoiseModel::Unit,
        )
        .expect("translation-only gauge solve succeeds");

        assert!(report.final_cost < report.initial_cost);
        assert_eq!(poses[0], fixture.initial_poses[0]);
        match gauge {
            ScaleGauge::FixedBaseline { .. } => {
                assert_close(norm(poses[1].translation), initial_baseline, 1e-12);
            }
            ScaleGauge::FixedTranslationComponent { .. } => {
                assert_eq!(
                    poses[2].translation[1].to_bits(),
                    initial_component.to_bits()
                );
            }
            ScaleGauge::Free => unreachable!(),
        }
    }
}

#[test]
fn translation_only_supports_pinhole_and_camera_agnostic_angular_measurements() {
    let fixture = fixture();
    for measurements in [pixel_measurements(&fixture), angular_measurements(&fixture)] {
        let gauge = ScaleGauge::FixedBaseline { camera: 1 };
        let mut workspace =
            translation_only_workspace(fixture.initial_poses.len(), measurements, gauge);
        let mut poses = fixture.initial_poses.clone();
        let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(30),
            NoiseModel::Unit,
        )
        .expect("translation-only calibrated solve succeeds");

        assert!(report.final_cost < report.initial_cost);
        for (pose, rotation) in poses.iter().zip(rotations) {
            assert_eq!(pose.rotation, rotation);
        }
    }
}

#[test]
fn translation_only_free_scale_reports_only_the_expected_gauge() {
    let fixture = fixture();
    let measurements = pixel_measurements(&fixture);
    let gauge = ScaleGauge::Free;
    let mut workspace =
        translation_only_workspace(fixture.initial_poses.len(), measurements, gauge);
    let mut poses = fixture.initial_poses.clone();
    let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
    let mut points = fixture.initial_points.clone();
    let report = refine_ba_multi_view_sparse(
        &mut poses,
        &mut points,
        measurements,
        gauge,
        &mut workspace,
        Options::l2(30),
        NoiseModel::Unit,
    )
    .expect("free-scale translation-only solve succeeds");

    assert!(report.final_cost < report.initial_cost);
    assert_eq!(
        report.diagnostics.pose_rank,
        SparseCameraRankAssessment::OnlyExpectedScaleGauge
    );
    for (pose, rotation) in poses.iter().zip(rotations) {
        assert_eq!(pose.rotation, rotation);
    }
}

#[test]
fn robust_translation_only_objectives_descend_without_changing_rotations() {
    let fixture = fixture();
    let mut outlier_pixels = fixture.pixels.clone();
    outlier_pixels[3].point[0] += 180.0;
    outlier_pixels[3].point[1] -= 120.0;
    let measurements = MultiViewMeasurements::Pixel {
        observations: &outlier_pixels,
        point_offsets: &fixture.offsets,
        cameras: &fixture.cameras,
    };
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    for options in [Options::huber(4.0, 30), Options::cauchy(4.0, 30)] {
        let mut workspace =
            translation_only_workspace(fixture.initial_poses.len(), measurements, gauge);
        let mut poses = fixture.initial_poses.clone();
        let rotations = poses.iter().map(|pose| pose.rotation).collect::<Vec<_>>();
        let mut points = fixture.initial_points.clone();
        let report = refine_ba_multi_view_sparse(
            &mut poses,
            &mut points,
            measurements,
            gauge,
            &mut workspace,
            options,
            NoiseModel::Unit,
        )
        .expect("robust translation-only solve succeeds");

        assert!(report.final_cost < report.initial_cost);
        for (pose, rotation) in poses.iter().zip(rotations) {
            assert_eq!(pose.rotation, rotation);
        }
    }
}
