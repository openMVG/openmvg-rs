// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "faer")]

use openmvg_ba::kernel::{
    RelativePose, multi_view::BundleAdjustmentPriors, multi_view::FixedVariableBlocks,
    multi_view::PixelObservation, multi_view::PointPrior, multi_view::PosePrior,
    multi_view::RadialIntrinsicPrior, multi_view::ScaleGauge,
    multi_view::self_calibration::RadialCamera,
    multi_view::self_calibration::SelfCalibratingMeasurements,
    multi_view::self_calibration::SelfCalibrationWorkspace,
    multi_view::self_calibration::refine_ba_multi_view_self_calibrating_grouped_dense_with_factors,
    multi_view::sparse::IterativeSchurOptions, multi_view::sparse::RetainedSchurMatvec,
    multi_view::sparse::SparseSelfCalibrationWorkspace,
    multi_view::sparse::refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors,
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{NoiseModel, Options};

const CAMERA_TO_GROUP: [usize; 4] = [0, 0, 1, 1];

struct Fixture {
    truth_poses: [RelativePose; 4],
    truth_points: Vec<[f64; 3]>,
    truth_groups: [RadialCamera; 2],
    observations: Vec<PixelObservation>,
    point_offsets: Vec<usize>,
}

fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

fn fixture() -> Fixture {
    let truth_poses = [
        identity_pose(),
        RelativePose::new(
            manifold::exp_so3(&[0.018, 0.055, -0.012]),
            [0.72, 0.03, 0.09],
        ),
        RelativePose::new(
            manifold::exp_so3(&[-0.035, -0.045, 0.024]),
            [-0.61, 0.08, 0.13],
        ),
        RelativePose::new(
            manifold::exp_so3(&[0.046, -0.018, 0.031]),
            [0.19, -0.48, 0.11],
        ),
    ];
    let truth_groups = [
        RadialCamera::new(690.0, 0.052, -0.021),
        RadialCamera::new(735.0, -0.039, 0.016),
    ];
    let truth_points: Vec<_> = (0..30)
        .map(|index| {
            [
                -1.35 + 0.48 * (index % 6) as f64,
                -0.92 + 0.43 * (index / 6) as f64,
                3.35 + 0.27 * (index % 7) as f64,
            ]
        })
        .collect();
    let mut observations = Vec::with_capacity(truth_poses.len() * truth_points.len());
    let mut point_offsets = Vec::with_capacity(truth_points.len() + 1);
    point_offsets.push(0);
    for point in &truth_points {
        for (camera, pose) in truth_poses.iter().enumerate() {
            observations.push(PixelObservation::new(
                camera,
                project(
                    truth_groups[CAMERA_TO_GROUP[camera]],
                    transform(*pose, *point),
                ),
            ));
        }
        point_offsets.push(observations.len());
    }
    Fixture {
        truth_poses,
        truth_points,
        truth_groups,
        observations,
        point_offsets,
    }
}

fn perturbed(fixture: &Fixture) -> (Vec<RelativePose>, Vec<[f64; 3]>, Vec<RadialCamera>) {
    let mut poses = fixture.truth_poses.to_vec();
    for (camera, pose) in poses.iter_mut().enumerate().skip(1) {
        pose.rotation = manifold::mat_mul(
            &manifold::exp_so3(&[0.003 * camera as f64, -0.004, 0.002 * camera as f64]),
            &pose.rotation,
        );
        pose.translation[0] += 0.012 * camera as f64;
        pose.translation[1] -= 0.008;
        pose.translation[2] += 0.006;
    }
    let baseline = norm(&fixture.truth_poses[1].translation);
    let scale = baseline / norm(&poses[1].translation);
    poses[1].translation = poses[1].translation.map(|value| scale * value);

    let points = fixture
        .truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.014 * phase.sin(),
                point[1] - 0.011 * (0.7 * phase).cos(),
                point[2] + 0.018 * (0.35 * phase).sin(),
            ]
        })
        .collect();
    let groups = vec![
        RadialCamera::new(706.0, 0.058, -0.024),
        RadialCamera::new(719.0, -0.033, 0.013),
    ];
    (poses, points, groups)
}

#[test]
fn grouped_sparse_direct_matches_dense_with_fixed_blocks_and_unary_priors() {
    let fixture = fixture();
    let measurements =
        SelfCalibratingMeasurements::new(&fixture.observations, &fixture.point_offsets);
    let gauge = ScaleGauge::FixedBaseline { camera: 1 };
    let (initial_poses, initial_points, initial_groups) = perturbed(&fixture);

    let fixed_poses = [false, false, false, true];
    let mut fixed_points = vec![false; fixture.truth_points.len()];
    fixed_points[0] = true;
    let fixed_intrinsics = [true, false];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &fixed_intrinsics,
        intrinsic_dimensions: &[],
    };

    let pose_priors = [PosePrior {
        camera: 2,
        mean: fixture.truth_poses[2],
        sqrt_information: diagonal6([0.8, 0.8, 0.8, 1.5, 1.5, 1.5]),
    }];
    let point_priors = [PointPrior {
        point: 1,
        mean: fixture.truth_points[1],
        sqrt_information: diagonal3([2.0, 2.0, 2.0]),
    }];
    let intrinsic_priors = [RadialIntrinsicPrior {
        intrinsic: 1,
        focal: fixture.truth_groups[1].focal,
        k1: fixture.truth_groups[1].k1,
        k2: fixture.truth_groups[1].k2,
        sqrt_information: diagonal3([0.7, 12.0, 12.0]),
    }];
    let priors = BundleAdjustmentPriors {
        poses: &pose_priors,
        points: &point_priors,
        intrinsics: &intrinsic_priors,
    };

    let mut dense_poses = initial_poses.clone();
    let mut dense_points = initial_points.clone();
    let mut dense_groups = initial_groups.clone();
    let mut sparse_poses = initial_poses.clone();
    let mut sparse_points = initial_points.clone();
    let mut sparse_groups = initial_groups.clone();
    let mut retained_poses = initial_poses.clone();
    let mut retained_points = initial_points.clone();
    let mut retained_groups = initial_groups.clone();
    let mut recomputed_poses = initial_poses.clone();
    let mut recomputed_points = initial_points.clone();
    let mut recomputed_groups = initial_groups.clone();
    let mut dense_workspace =
        SelfCalibrationWorkspace::try_new(initial_poses.len(), initial_points.len())
            .expect("small dense workspace must fit");
    let mut sparse_workspace = SparseSelfCalibrationWorkspace::try_new_grouped(
        initial_poses.len(),
        initial_groups.len(),
        &CAMERA_TO_GROUP,
        measurements,
        gauge,
        fixed,
    )
    .expect("grouped sparse topology and fixed masks are valid");
    let mut retained_workspace =
        SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
            initial_poses.len(),
            initial_groups.len(),
            &CAMERA_TO_GROUP,
            measurements,
            gauge,
            fixed,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Fused)
                .with_maximum_iterations(300)
                .with_exact_camera_rank_diagnostics(false),
        )
        .expect("grouped retained topology and fixed masks are valid");
    let mut recomputed_workspace =
        SparseSelfCalibrationWorkspace::try_new_grouped_with_iterative_schur_options(
            initial_poses.len(),
            initial_groups.len(),
            &CAMERA_TO_GROUP,
            measurements,
            gauge,
            fixed,
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Recomputed)
                .with_maximum_iterations(300)
                .with_exact_camera_rank_diagnostics(false),
        )
        .expect("grouped recomputed topology and fixed masks are valid");
    assert!(!recomputed_workspace.retains_cross_rows());

    let dense = refine_ba_multi_view_self_calibrating_grouped_dense_with_factors(
        &mut dense_poses,
        &mut dense_points,
        &mut dense_groups,
        &CAMERA_TO_GROUP,
        measurements,
        gauge,
        fixed,
        priors,
        &mut dense_workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("dense grouped fixture is valid");
    let sparse = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut sparse_poses,
        &mut sparse_points,
        &mut sparse_groups,
        &CAMERA_TO_GROUP,
        measurements,
        gauge,
        fixed,
        priors,
        &mut sparse_workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("sparse grouped fixture is valid");
    let retained = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut retained_poses,
        &mut retained_points,
        &mut retained_groups,
        &CAMERA_TO_GROUP,
        measurements,
        gauge,
        fixed,
        priors,
        &mut retained_workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("retained grouped fixture is valid");
    let recomputed = refine_ba_multi_view_self_calibrating_grouped_sparse_with_factors(
        &mut recomputed_poses,
        &mut recomputed_points,
        &mut recomputed_groups,
        &CAMERA_TO_GROUP,
        measurements,
        gauge,
        fixed,
        priors,
        &mut recomputed_workspace,
        Options::l2(100),
        NoiseModel::Unit,
    )
    .expect("recomputed grouped fixture is valid");

    assert!(dense.final_cost < dense.initial_cost, "{dense:#?}");
    assert!(sparse.final_cost < sparse.initial_cost, "{sparse:#?}");
    assert!(retained.final_cost < retained.initial_cost, "{retained:#?}");
    assert!(
        recomputed.final_cost < recomputed.initial_cost,
        "{recomputed:#?}"
    );
    assert_relative_close(dense.final_cost, sparse.final_cost, 2e-7);
    assert_relative_close(dense.final_rmse, sparse.final_rmse, 2e-7);
    assert_relative_close(sparse.final_cost, retained.final_cost, 2e-7);
    assert_relative_close(sparse.final_rmse, retained.final_rmse, 2e-7);
    assert_relative_close(sparse.final_cost, recomputed.final_cost, 2e-7);
    assert_relative_close(sparse.final_rmse, recomputed.final_rmse, 2e-7);

    assert_eq!(dense_poses[3], initial_poses[3]);
    assert_eq!(sparse_poses[3], initial_poses[3]);
    assert_eq!(retained_poses[3], initial_poses[3]);
    assert_eq!(recomputed_poses[3], initial_poses[3]);
    assert_eq!(dense_points[0], initial_points[0]);
    assert_eq!(sparse_points[0], initial_points[0]);
    assert_eq!(retained_points[0], initial_points[0]);
    assert_eq!(recomputed_points[0], initial_points[0]);
    assert_eq!(dense_groups[0], initial_groups[0]);
    assert_eq!(sparse_groups[0], initial_groups[0]);
    assert_eq!(retained_groups[0], initial_groups[0]);
    assert_eq!(recomputed_groups[0], initial_groups[0]);

    let baseline = norm(&initial_poses[1].translation);
    assert_relative_close(norm(&dense_poses[1].translation), baseline, 2e-13);
    assert_relative_close(norm(&sparse_poses[1].translation), baseline, 2e-13);
}

#[test]
fn per_view_sparse_constructor_accepts_the_same_fixed_layout_contract() {
    let fixture = fixture();
    let measurements =
        SelfCalibratingMeasurements::new(&fixture.observations, &fixture.point_offsets);
    let fixed_poses = [false, false, false, true];
    let fixed_points = vec![false; fixture.truth_points.len()];
    let fixed_intrinsics = [true, false, false, true];
    let fixed = FixedVariableBlocks {
        camera_poses: &fixed_poses,
        points: &fixed_points,
        intrinsics: &fixed_intrinsics,
        intrinsic_dimensions: &[],
    };
    SparseSelfCalibrationWorkspace::try_new(
        fixture.truth_poses.len(),
        measurements,
        ScaleGauge::FixedBaseline { camera: 1 },
        fixed,
    )
    .expect("per-view radial topology accepts pose, point, and intrinsic masks");
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
    let distortion = 1.0 + camera.k1 * radius_squared + camera.k2 * radius_squared.powi(2);
    normalized.map(|value| camera.focal * distortion * value)
}

fn diagonal3(values: [f64; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { values[row] } else { 0.0 })
    })
}

fn diagonal6(values: [f64; 6]) -> [[f64; 6]; 6] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { values[row] } else { 0.0 })
    })
}

fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn assert_relative_close(left: f64, right: f64, tolerance: f64) {
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= tolerance * scale,
        "left={left:e}, right={right:e}, relative error={:e}",
        (left - right).abs() / scale
    );
}
