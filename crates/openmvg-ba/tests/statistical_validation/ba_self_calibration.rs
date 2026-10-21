// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::{
    RelativePose, multi_view::PixelObservation, multi_view::ScaleGauge,
    multi_view::self_calibration::RadialCamera,
    multi_view::self_calibration::SelfCalibratingMeasurements,
    multi_view::self_calibration::SelfCalibrationWorkspace,
    multi_view::self_calibration::covariance_ba_multi_view_self_calibrating_dense,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_dense,
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{Loss, NoiseModel, Options, RefinementStatus};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

use super::ba_common::{
    ClusteredCoverage, depth_bins, dot, identity_pose, norm, normalize, transpose,
};
use super::common::{COVARIANCE_TRIALS, IndependentCoverage, transform};

fn project(camera: RadialCamera, point: &[f64; 3]) -> [f64; 2] {
    let normalized = [point[0] / point[2], point[1] / point[2]];
    let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
    let distortion = 1.0 + camera.k1 * radius_squared + camera.k2 * radius_squared * radius_squared;
    normalized.map(|value| camera.focal * distortion * value)
}

fn camera_error(
    estimate_poses: &[RelativePose; 3],
    truth_poses: &[RelativePose; 3],
    estimate_cameras: &[RadialCamera; 3],
    truth_cameras: &[RadialCamera; 3],
    scene_scale: f64,
) -> [f64; 20] {
    let mut error = [0.0; 20];
    intrinsic_error(&mut error[0..3], estimate_cameras[0], truth_cameras[0]);

    let fixed_rotation = manifold::mat_mul(
        &truth_poses[1].rotation,
        &transpose(&estimate_poses[1].rotation),
    );
    error[3..6].copy_from_slice(&manifold::log_so3(&fixed_rotation));
    let estimate_direction = normalize(estimate_poses[1].translation);
    let truth_direction = normalize(truth_poses[1].translation);
    let basis = manifold::s2_tangent_basis(&estimate_direction);
    let cosine = dot(&estimate_direction, &truth_direction).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let tangent: [f64; 3] = if angle < 1e-10 {
        std::array::from_fn(|coordinate| {
            truth_direction[coordinate] - estimate_direction[coordinate]
        })
    } else {
        let scale = angle / angle.sin();
        std::array::from_fn(|coordinate| {
            scale * (truth_direction[coordinate] - cosine * estimate_direction[coordinate])
        })
    };
    error[6] = dot(&basis[0], &tangent);
    error[7] = dot(&basis[1], &tangent);
    intrinsic_error(&mut error[8..11], estimate_cameras[1], truth_cameras[1]);

    let free_rotation = manifold::mat_mul(
        &truth_poses[2].rotation,
        &transpose(&estimate_poses[2].rotation),
    );
    error[11..14].copy_from_slice(&manifold::log_so3(&free_rotation));
    for coordinate in 0..3 {
        error[14 + coordinate] = (truth_poses[2].translation[coordinate]
            - estimate_poses[2].translation[coordinate])
            / scene_scale;
    }
    intrinsic_error(&mut error[17..20], estimate_cameras[2], truth_cameras[2]);
    error
}

fn intrinsic_error(output: &mut [f64], estimate: RadialCamera, truth: RadialCamera) {
    output[0] = (truth.focal / estimate.focal).ln();
    output[1] = truth.k1 - estimate.k1;
    output[2] = truth.k2 - estimate.k2;
}

#[test]
fn self_calibration_covariance_coverage_200_trials() {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(
            manifold::exp_so3(&[0.025, 0.07, -0.015]),
            [0.72, 0.03, 0.08],
        ),
        RelativePose::new(
            manifold::exp_so3(&[-0.04, -0.06, 0.025]),
            [-0.62, 0.09, 0.14],
        ),
    ];
    let truth_cameras = [
        RadialCamera::new(690.0, 0.07, -0.028),
        RadialCamera::new(735.0, -0.045, 0.022),
        RadialCamera::new(665.0, 0.052, -0.017),
    ];
    let truth_points: Vec<_> = (0..72)
        .map(|index| {
            let column = (index % 9) as f64;
            let row = (index / 9) as f64;
            [
                -1.55 + 0.39 * column,
                -1.12 + 0.32 * row,
                3.1 + 0.34 * (index % 7) as f64 + 0.07 * row,
            ]
        })
        .collect();
    let mut exact = Vec::with_capacity(3 * truth_points.len());
    let mut offsets = Vec::with_capacity(truth_points.len() + 1);
    offsets.push(0);
    for point in &truth_points {
        for camera in 0..3 {
            exact.push(PixelObservation::new(
                camera,
                project(
                    truth_cameras[camera],
                    &transform(&truth_poses[camera], point),
                ),
            ));
        }
        offsets.push(exact.len());
    }
    let sigma = 0.08;
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let bins = depth_bins(&truth_points);
    let mut workspace = SelfCalibrationWorkspace::try_new(3, truth_points.len())
        .expect("statistical self-calibration workspace must fit");
    let mut rng = StdRng::seed_from_u64(0x5e1f_ca11_ba03);
    let mut camera_coverage = IndependentCoverage::new("ba_self_cal_camera", 20);
    let mut all_points = ClusteredCoverage::new("ba_self_cal/all_points");
    let mut depths: [ClusteredCoverage; 3] = std::array::from_fn(|index| {
        ClusteredCoverage::new(format!(
            "ba_self_cal/depth={}",
            ["near", "middle", "far"][index]
        ))
    });

    for _ in 0..COVARIANCE_TRIALS {
        let mut observations = exact.clone();
        for observation in &mut observations {
            let x: f64 = StandardNormal.sample(&mut rng);
            let y: f64 = StandardNormal.sample(&mut rng);
            observation.point[0] += sigma * x;
            observation.point[1] += sigma * y;
        }
        let measurements = SelfCalibratingMeasurements::new(&observations, &offsets);
        let mut poses = truth_poses;
        for (camera, pose) in poses.iter_mut().enumerate().skip(1) {
            pose.rotation = manifold::mat_mul(
                &manifold::exp_so3(&[0.0004 * camera as f64, -0.0005, 0.0003]),
                &pose.rotation,
            );
            pose.translation[0] += 0.0008 * camera as f64;
            pose.translation[1] -= 0.0005;
            pose.translation[2] += 0.0004;
        }
        let baseline = norm(&truth_poses[1].translation);
        let initial_baseline = norm(&poses[1].translation);
        poses[1].translation = poses[1]
            .translation
            .map(|value| baseline * value / initial_baseline);
        let mut points: Vec<_> = truth_points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let phase = index as f64;
                [
                    point[0] + 0.0008 * phase.sin(),
                    point[1] - 0.0007 * (0.7 * phase).cos(),
                    point[2] + 0.001 * (0.4 * phase).sin(),
                ]
            })
            .collect();
        let mut cameras = truth_cameras.map(|camera| {
            RadialCamera::new(
                camera.focal * 1.0002,
                camera.k1 + 0.0002,
                camera.k2 - 0.0001,
            )
        });
        let report = refine_ba_multi_view_self_calibrating_dense(
            &mut poses,
            &mut points,
            &mut cameras,
            measurements,
            gauge,
            &mut workspace,
            Options::l2(60),
            NoiseModel::isotropic(sigma),
        )
        .expect("statistical self-calibration fixture must be valid");
        assert!(!matches!(report.status, RefinementStatus::Degenerate));
        let covariance = covariance_ba_multi_view_self_calibrating_dense(
            &poses,
            &points,
            &cameras,
            measurements,
            gauge,
            &mut workspace,
            Loss::L2,
            NoiseModel::isotropic(sigma),
        )
        .expect("well-conditioned self-calibration covariance must exist");
        camera_coverage.push(
            covariance
                .camera_nees(&camera_error(
                    &poses,
                    &truth_poses,
                    &cameras,
                    &truth_cameras,
                    covariance.normalization_scale(),
                ))
                .expect("self-calibration camera error has dimension 20"),
        );
        let point_nees: Vec<_> = points
            .iter()
            .zip(&truth_points)
            .enumerate()
            .map(|(index, (estimate, truth))| {
                let error = std::array::from_fn(|coordinate| {
                    (truth[coordinate] - estimate[coordinate]) / covariance.normalization_scale()
                });
                covariance
                    .point_nees(index, &error)
                    .expect("every self-calibration point has a marginal")
            })
            .collect();
        all_points.push_trial(point_nees.iter().copied());
        for (depth, cluster) in depths.iter_mut().enumerate() {
            cluster.push_trial(
                point_nees
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| (bins[index] == depth).then_some(*value)),
            );
        }
    }

    camera_coverage.report_and_check();
    all_points.report_and_check();
    for cluster in &depths {
        cluster.report_and_check();
    }
}
