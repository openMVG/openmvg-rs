// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One mixed-mask system with both surveyed controls and eliminated free points.

use super::*;
use crate::PoseVariables;
use crate::kernel::multi_view::sparse::api::RetainedCrossPrecision;
use crate::kernel::multi_view::sparse::{
    IterativeSchurOptions, RetainedSchurMatvec, SparseWorkspaceOptions,
};
use crate::kernel::multi_view::{AngularObservation, PixelObservation};
use crate::kernel::{PinholeCamera, RelativePose};
#[cfg(feature = "unstable-internals")]
use crate::numerics::refine::Loss;
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

const CAMERA_DIMENSION: usize = 12;
const FULL_DIMENSION: usize = 24;
const FIXED_CONTROLS: usize = 4;

fn rotation(pose: RelativePose) -> Matrix3<f64> {
    Matrix3::from_fn(|row, column| pose.rotation[row][column])
}

fn center(pose: RelativePose) -> Vector3<f64> {
    -(rotation(pose).transpose() * Vector3::from(pose.translation))
}

fn pose(rotation_vector: Vector3<f64>, center: Vector3<f64>) -> RelativePose {
    let r = Rotation3::new(rotation_vector).into_inner();
    let t = -(r * center);
    RelativePose::new(
        std::array::from_fn(|row| std::array::from_fn(|column| r[(row, column)])),
        t.into(),
    )
}

// Independent projection and chart perturbation. Full uses Exp(w)R X+t+dt;
// rotation-only uses Exp(w)R(X-C), with the initial C held. Free point columns
// use the same documented dimensionless coordinate scale as the solver.
fn residuals(
    poses: &[RelativePose; 4],
    points: &[[f64; 3]; 8],
    observations: &[PixelObservation],
    scale: f64,
    delta: &[f64; FULL_DIMENSION],
) -> DVector<f64> {
    DVector::from_fn(2 * observations.len(), |row, _| {
        let observation = observations[row / 2];
        let point_index = row / 8;
        let camera = observation.camera;
        let mut x = Vector3::from(points[point_index]);
        if point_index >= FIXED_CONTROLS {
            let offset = CAMERA_DIMENSION + 3 * (point_index - FIXED_CONTROLS);
            x += scale * Vector3::new(delta[offset], delta[offset + 1], delta[offset + 2]);
        }
        let r = rotation(poses[camera]);
        let t = Vector3::from(poses[camera].translation);
        let camera_point = match camera {
            0 => {
                Rotation3::new(Vector3::new(delta[0], delta[1], delta[2])) * (r * x)
                    + t
                    + scale * Vector3::new(delta[3], delta[4], delta[5])
            }
            1 => r * x + t + scale * Vector3::new(delta[6], delta[7], delta[8]),
            2 => {
                Rotation3::new(Vector3::new(delta[9], delta[10], delta[11]))
                    * (r * (x - center(poses[camera])))
            }
            _ => r * x + t,
        };
        let prediction = if row % 2 == 0 {
            430.0 * camera_point.x / camera_point.z + 13.0
        } else {
            415.0 * camera_point.y / camera_point.z - 8.0
        };
        prediction - observation.point[row % 2]
    })
}

fn normal_terms(
    poses: &[RelativePose; 4],
    points: &[[f64; 3]; 8],
    observations: &[PixelObservation],
    scale: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut j = DMatrix::zeros(2 * observations.len(), FULL_DIMENSION);
    let h = 1.0e-6;
    for column in 0..FULL_DIMENSION {
        let mut plus = [0.0; FULL_DIMENSION];
        let mut minus = [0.0; FULL_DIMENSION];
        plus[column] = h;
        minus[column] = -h;
        let derivative = (residuals(poses, points, observations, scale, &plus)
            - residuals(poses, points, observations, scale, &minus))
            / (2.0 * h);
        j.set_column(column, &derivative);
    }
    let r = residuals(poses, points, observations, scale, &[0.0; FULL_DIMENSION]);
    (j.transpose() * &j, j.transpose() * r)
}

fn scaled_gradient(h: &DMatrix<f64>, g: &DVector<f64>) -> f64 {
    let mut norm = 0.0_f64;
    for (start, end) in [
        (0, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        (15, 18),
        (18, 21),
        (21, 24),
    ] {
        let curvature = (start..end).map(|axis| h[(axis, axis)]).fold(0.0, f64::max);
        let floor = 64.0 * f64::EPSILON * curvature;
        for axis in start..end {
            norm = norm.max(g[axis].abs() / h[(axis, axis)].max(floor).sqrt());
        }
    }
    norm
}

#[test]
fn mixed_masks_with_controls_and_free_points_match_dense_schur_and_recover_truth() {
    let truth_poses = [
        pose(Vector3::zeros(), Vector3::zeros()),
        pose(Vector3::new(0.02, -0.05, 0.01), Vector3::new(1.0, 0.1, 0.0)),
        pose(
            Vector3::new(-0.03, 0.04, -0.01),
            Vector3::new(-0.5, 0.8, 0.1),
        ),
        pose(Vector3::new(0.02, 0.01, 0.03), Vector3::new(0.2, -0.8, 0.0)),
    ];
    let truth_points = [
        [-0.6, -0.4, 4.0],
        [0.6, -0.3, 4.4],
        [0.1, 0.7, 5.0],
        [-0.2, 0.1, 5.8],
        [0.8, 0.5, 5.2],
        [-0.8, 0.2, 5.5],
        [0.4, -0.7, 6.1],
        [-0.1, -0.4, 4.8],
    ];
    let mut observations = Vec::new();
    for point in truth_points {
        for (camera, &view) in truth_poses.iter().enumerate() {
            let x = rotation(view) * Vector3::from(point) + Vector3::from(view.translation);
            assert!(x.z > 0.0);
            observations.push(PixelObservation::new(
                camera,
                [430.0 * x.x / x.z + 13.0, 415.0 * x.y / x.z - 8.0],
            ));
        }
    }
    let offsets = [0, 4, 8, 12, 16, 20, 24, 28, 32];
    let cameras = [PinholeCamera::new(430.0, 415.0, 13.0, -8.0); 4];
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    };
    let masks = [
        PoseVariables::FREE,
        PoseVariables::FREE.with_rotation(false),
        PoseVariables::FREE.with_translation(false),
        PoseVariables::FIXED,
    ];
    let fixed = FixedVariableBlocks {
        camera_poses: &[false, false, false, true],
        points: &[true, true, true, true, false, false, false, false],
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let initial_poses = [
        pose(
            Vector3::new(0.02, -0.012, 0.015),
            Vector3::new(0.08, -0.03, 0.025),
        ),
        pose(
            Vector3::new(0.02, -0.05, 0.01),
            Vector3::new(0.96, 0.16, -0.02),
        ),
        pose(Vector3::new(-0.04, 0.06, -0.028), center(truth_poses[2])),
        truth_poses[3],
    ];
    let mut initial_points = truth_points;
    for (index, x) in initial_points.iter_mut().enumerate().skip(FIXED_CONTROLS) {
        x[0] += if index % 2 == 0 { 0.025 } else { -0.02 };
        x[1] += 0.018;
        x[2] -= 0.03;
    }
    let scale = (initial_points
        .iter()
        .flat_map(|x| x.iter())
        .map(|x| x * x)
        .sum::<f64>()
        / initial_points.len() as f64)
        .sqrt();
    let (h, g) = normal_terms(&initial_poses, &initial_points, &observations, scale);
    let a = h
        .view((0, 0), (CAMERA_DIMENSION, CAMERA_DIMENSION))
        .into_owned();
    let b = h
        .view((0, CAMERA_DIMENSION), (CAMERA_DIMENSION, CAMERA_DIMENSION))
        .into_owned();
    let c = h
        .view(
            (CAMERA_DIMENSION, CAMERA_DIMENSION),
            (CAMERA_DIMENSION, CAMERA_DIMENSION),
        )
        .into_owned();
    let point_factor = c.cholesky().expect("four free point blocks are observable");
    let expected_schur = a - &b * point_factor.solve(&b.transpose());
    let expected_gradient = g.rows(0, CAMERA_DIMENSION).into_owned()
        - &b * point_factor.solve(&g.rows(CAMERA_DIMENSION, CAMERA_DIMENSION).into_owned());
    let matrix_tolerance = 2.0e-7 * expected_schur.amax().max(1.0);
    let gradient_tolerance = 2.0e-6 * expected_gradient.amax().max(1.0);
    let iterative = IterativeSchurOptions::new()
        .with_relative_tolerance(1.0e-8)
        .with_direct_fallback(false)
        .with_exact_camera_rank_diagnostics(true);
    let policies = [
        ("direct", SparseWorkspaceOptions::direct()),
        (
            "cached-f64",
            SparseWorkspaceOptions::iterative_schur(
                iterative.with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 2 }),
            ),
        ),
        (
            "cached-f32",
            SparseWorkspaceOptions::iterative_schur(
                iterative
                    .with_cross_precision(RetainedCrossPrecision::F32)
                    .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 2 }),
            ),
        ),
        (
            "recomputed",
            SparseWorkspaceOptions::iterative_schur(
                iterative.with_matvec(RetainedSchurMatvec::RecomputedParallel { threads: 2 }),
            ),
        ),
    ];
    for (name, policy) in policies {
        let mut workspace =
            SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_masks(
                4,
                measurements,
                ScaleGauge::Free,
                fixed,
                &masks,
                policy,
            )
            .expect("mixed fixed-control/free-point topology");
        assert_eq!(workspace.camera_pose_masks(), &masks);
        let mut poses = initial_poses;
        let mut points = initial_points;
        // Zero nonlinear iterations isolates the exact final Schur/gradient pass,
        // even when an iterative workspace stores its nonlinear cross rows in f32.
        let initial_report = refine_ba_multi_view_sparse_with_factors(
            &mut poses,
            &mut points,
            measurements,
            ScaleGauge::Free,
            fixed,
            BundleAdjustmentPriors::NONE,
            &mut workspace,
            Options::l2(0),
            NoiseModel::Unit,
        )
        .expect("fresh initial information");
        assert_eq!(
            initial_report.diagnostics.pose_rank,
            SparseCameraRankAssessment::FullRank,
            "{name}"
        );
        assert_eq!(initial_report.diagnostics.weak_point_count, 0, "{name}");
        for row in 0..CAMERA_DIMENSION {
            assert!(
                (workspace.core.gradient[row] - expected_gradient[row]).abs() <= gradient_tolerance,
                "{name}: reduced gradient row {row}"
            );
            for column in 0..=row {
                let slot = workspace
                    .core
                    .scatter
                    .position(row, column)
                    .expect("dense Schur slot");
                assert!(
                    (workspace.core.values[slot] - expected_schur[(row, column)]).abs()
                        <= matrix_tolerance,
                    "{name}: Schur entry ({row}, {column})"
                );
            }
        }
        let expected_initial_gradient = scaled_gradient(&h, &g);
        assert!(
            (initial_report.gradient_inf_norm - expected_initial_gradient).abs()
                < 2.0e-6 * (1.0 + expected_initial_gradient),
            "{name}: fresh full initial gradient"
        );

        if name == "direct" {
            let angular = observations
                .iter()
                .map(|observation| AngularObservation::new(observation.camera, [0.0, 0.0, 1.0]))
                .collect::<Vec<_>>();
            let wrong_family = MultiViewMeasurements::Angular {
                observations: &angular,
                point_offsets: &offsets,
            };
            let result = refine_ba_multi_view_sparse_with_factors(
                &mut poses,
                &mut points,
                wrong_family,
                ScaleGauge::Free,
                fixed,
                BundleAdjustmentPriors::NONE,
                &mut workspace,
                Options::l2(1),
                NoiseModel::Unit,
            );
            assert_eq!(result, Err(SparseBundleAdjustmentError::TopologyMismatch));
            assert_eq!(poses, initial_poses);
            assert_eq!(points, initial_points);
        }
        #[cfg(feature = "unstable-internals")]
        if name == "direct" {
            use crate::kernel::multi_view::sparse::{
                SparseBundleAdjustmentCovarianceError,
                selected_marginal_covariances_ba_multi_view_sparse_with_factors,
            };
            let masks_before = workspace.camera_pose_masks().to_vec();
            let information_before = workspace.core.values.clone();
            // This valid truth state has camera zero at identity, so the old
            // legacy validation would reach the partial-camera selection.
            let covariance = selected_marginal_covariances_ba_multi_view_sparse_with_factors(
                &truth_poses,
                &truth_points,
                measurements,
                ScaleGauge::Free,
                fixed,
                BundleAdjustmentPriors::NONE,
                &[1],
                &[4],
                &mut workspace,
                Loss::L2,
                NoiseModel::Unit,
            );
            assert_eq!(
                covariance,
                Err(SparseBundleAdjustmentCovarianceError::UnsupportedPoseMasks)
            );
            assert_eq!(workspace.camera_pose_masks(), &masks_before);
            assert_eq!(workspace.core.values, information_before);
            assert_eq!(poses, initial_poses);
            assert_eq!(points, initial_points);
        }
        let report = refine_ba_multi_view_sparse_with_factors(
            &mut poses,
            &mut points,
            measurements,
            ScaleGauge::Free,
            fixed,
            BundleAdjustmentPriors::NONE,
            &mut workspace,
            Options::l2(80),
            NoiseModel::Unit,
        )
        .expect("coupled mixed-mask optimization");
        assert!(report.accepted_step_count > 0, "{name}");
        assert!(
            report.final_cost < 1.0e-8,
            "{name}: final cost {}",
            report.final_cost
        );
        assert_eq!(
            report.diagnostics.pose_rank,
            SparseCameraRankAssessment::FullRank,
            "{name}"
        );
        assert_eq!(report.diagnostics.weak_point_count, 0, "{name}");
        if name != "direct" {
            assert!(
                report.retained_camera_steps > 0,
                "{name}: selected operator was exercised"
            );
            assert_eq!(report.direct_fallback_steps, 0, "{name}");
        }
        assert_eq!(poses[1].rotation, initial_poses[1].rotation);
        assert!(
            (center(poses[2]) - center(initial_poses[2])).norm() < 1.0e-12,
            "{name}: held center"
        );
        assert_eq!(poses[3], initial_poses[3]);
        assert_eq!(&points[..FIXED_CONTROLS], &initial_points[..FIXED_CONTROLS]);
        for camera in 0..4 {
            assert!(
                (center(poses[camera]) - center(truth_poses[camera])).norm() < 2.0e-5,
                "{name}: center {camera}"
            );
            assert!(
                (rotation(poses[camera]) - rotation(truth_poses[camera])).norm() < 5.0e-6,
                "{name}: rotation {camera}"
            );
        }
        for point in FIXED_CONTROLS..points.len() {
            assert!(
                (Vector3::from(points[point]) - Vector3::from(truth_points[point])).norm() < 2.0e-5,
                "{name}: free point {point}"
            );
        }
        let (final_h, final_g) =
            normal_terms(&poses, &points, &observations, report.normalization_scale);
        let expected_final_gradient = scaled_gradient(&final_h, &final_g);
        assert!(
            (report.gradient_inf_norm - expected_final_gradient).abs()
                < 2.0e-8 * (1.0 + expected_final_gradient),
            "{name}: fresh full final gradient"
        );
    }
}
