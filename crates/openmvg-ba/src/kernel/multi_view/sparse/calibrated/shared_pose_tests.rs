// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One shared physical pose, two fixed poses, and independent exposure models.

use super::*;
use crate::PoseVariables;
use crate::kernel::multi_view::PixelObservation;
use crate::kernel::multi_view::sparse::api::RetainedCrossPrecision;
use crate::kernel::multi_view::sparse::{
    IterativeSchurOptions, RetainedSchurMatvec, SparseWorkspaceOptions,
};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};
use openmvg_core::camera::{
    BrownConradyCamera, Camera, PinholeCamera, RadialDistortion, TangentialDistortion,
};

#[derive(Clone, Copy)]
struct Calibration {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    p1: f64,
    p2: f64,
}
impl Calibration {
    fn camera(self) -> Camera {
        let pinhole = PinholeCamera::try_new(self.fx, self.fy, self.cx, self.cy).unwrap();
        if self.k1 == 0.0 && self.p1 == 0.0 {
            Camera::Pinhole(pinhole)
        } else {
            Camera::BrownConrady(
                BrownConradyCamera::try_new(
                    pinhole,
                    RadialDistortion::K1K2K3(self.k1, self.k2, self.k3),
                    TangentialDistortion::P1P2(self.p1, self.p2),
                )
                .unwrap(),
            )
        }
    }
    fn project(self, x: Vector3<f64>) -> [f64; 2] {
        let a = x.x / x.z;
        let b = x.y / x.z;
        let q = a * a + b * b;
        let radial = 1.0 + self.k1 * q + self.k2 * q * q + self.k3 * q * q * q;
        [
            self.fx * (a * radial + 2.0 * self.p1 * a * b + self.p2 * (q + 2.0 * a * a)) + self.cx,
            self.fy * (b * radial + self.p1 * (q + 2.0 * b * b) + 2.0 * self.p2 * a * b) + self.cy,
        ]
    }
}
fn rotation(pose: RelativePose) -> Matrix3<f64> {
    Matrix3::from_fn(|row, column| pose.rotation[row][column])
}
fn pose(r: Vector3<f64>, center: Vector3<f64>) -> RelativePose {
    let rotation = Rotation3::new(r).into_inner();
    RelativePose::new(
        std::array::from_fn(|row| std::array::from_fn(|column| rotation[(row, column)])),
        (-(rotation * center)).into(),
    )
}
fn residuals(
    poses: &[RelativePose; 3],
    points: &[[f64; 3]; 8],
    observations: &[PixelObservation],
    selectors: &[usize],
    calibrations: &[Calibration; 4],
    scale: f64,
    delta: &[f64; 18],
) -> DVector<f64> {
    DVector::from_fn(2 * observations.len(), |row, _| {
        let observation = observations[row / 2];
        let point_index = row / 8;
        let mut x = Vector3::from(points[point_index]);
        if point_index >= 4 {
            let offset = 6 + 3 * (point_index - 4);
            x += scale * Vector3::new(delta[offset], delta[offset + 1], delta[offset + 2]);
        }
        let mut r = rotation(poses[observation.camera]);
        let mut t = Vector3::from(poses[observation.camera].translation);
        if observation.camera == 1 {
            r = Rotation3::new(Vector3::new(delta[0], delta[1], delta[2])).into_inner() * r;
            t += scale * Vector3::new(delta[3], delta[4], delta[5]);
        }
        calibrations[selectors[row / 2]].project(r * x + t)[row % 2] - observation.point[row % 2]
    })
}
fn normal_terms(
    poses: &[RelativePose; 3],
    points: &[[f64; 3]; 8],
    observations: &[PixelObservation],
    selectors: &[usize],
    calibrations: &[Calibration; 4],
    scale: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut j = DMatrix::zeros(2 * observations.len(), 18);
    for column in 0..18 {
        let mut plus = [0.0; 18];
        let mut minus = [0.0; 18];
        plus[column] = 1.0e-6;
        minus[column] = -1.0e-6;
        let derivative = (residuals(
            poses,
            points,
            observations,
            selectors,
            calibrations,
            scale,
            &plus,
        ) - residuals(
            poses,
            points,
            observations,
            selectors,
            calibrations,
            scale,
            &minus,
        )) / (2.0e-6);
        j.set_column(column, &derivative);
    }
    let r = residuals(
        poses,
        points,
        observations,
        selectors,
        calibrations,
        scale,
        &[0.0; 18],
    );
    (j.transpose() * &j, j.transpose() * r)
}
fn scaled_gradient(h: &DMatrix<f64>, g: &DVector<f64>) -> f64 {
    let mut norm = 0.0_f64;
    for (start, end) in [(0, 6), (6, 9), (9, 12), (12, 15), (15, 18)] {
        let floor = 64.0 * f64::EPSILON * (start..end).map(|i| h[(i, i)]).fold(0.0, f64::max);
        for i in start..end {
            norm = norm.max(g[i].abs() / h[(i, i)].max(floor).sqrt());
        }
    }
    norm
}

#[test]
fn shared_pose_exposures_match_joint_information_schur_and_all_sparse_paths() {
    let calibrations = [
        Calibration {
            fx: 430.0,
            fy: 415.0,
            cx: 13.0,
            cy: -8.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
        Calibration {
            fx: 620.0,
            fy: 640.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.04,
            k2: 0.005,
            k3: 0.002,
            p1: 0.001,
            p2: -0.001,
        },
        Calibration {
            fx: 790.0,
            fy: 700.0,
            cx: 120.0,
            cy: 40.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
        Calibration {
            fx: 500.0,
            fy: 510.0,
            cx: 0.0,
            cy: 0.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
    ];
    let cameras = calibrations.map(Calibration::camera);
    let truth_poses = [
        pose(Vector3::zeros(), Vector3::zeros()),
        pose(
            Vector3::new(0.03, -0.02, 0.01),
            Vector3::new(1.0, 0.1, 0.05),
        ),
        pose(
            Vector3::new(-0.02, 0.03, 0.01),
            Vector3::new(-0.4, 0.8, 0.1),
        ),
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
    let mut selectors = Vec::new();
    for point in truth_points {
        // Aliases of pose1 are deliberately separated by a fixed-pose row.
        for (physical, intrinsic) in [(1, 1), (0, 0), (1, 2), (2, 3)] {
            let camera_point = rotation(truth_poses[physical]) * Vector3::from(point)
                + Vector3::from(truth_poses[physical].translation);
            observations.push(PixelObservation::new(
                physical,
                calibrations[intrinsic].project(camera_point),
            ));
            selectors.push(intrinsic);
        }
    }
    let offsets = [0, 4, 8, 12, 16, 20, 24, 28, 32];
    let measurements = MultiViewMeasurements::FixedPerspectiveIndexed {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
        observation_intrinsics: &selectors,
    };
    let masks = [
        PoseVariables::FIXED,
        PoseVariables::FREE,
        PoseVariables::FIXED,
    ];
    let fixed = FixedVariableBlocks {
        camera_poses: &[true, false, true],
        points: &[true, true, true, true, false, false, false, false],
        intrinsics: &[],
        intrinsic_dimensions: &[],
    };
    let initial_poses = [
        truth_poses[0],
        pose(
            Vector3::new(0.045, -0.01, 0.02),
            Vector3::new(1.08, 0.06, 0.075),
        ),
        truth_poses[2],
    ];
    let mut initial_points = truth_points;
    for (i, x) in initial_points.iter_mut().enumerate().skip(4) {
        x[0] += if i % 2 == 0 { 0.02 } else { -0.025 };
        x[1] += 0.018;
        x[2] -= 0.03;
    }
    let scale = (initial_points
        .iter()
        .flat_map(|p| p.iter())
        .map(|x| x * x)
        .sum::<f64>()
        / 8.0)
        .sqrt();
    let (h, g) = normal_terms(
        &initial_poses,
        &initial_points,
        &observations,
        &selectors,
        &calibrations,
        scale,
    );
    let a = h.view((0, 0), (6, 6)).into_owned();
    let b = h.view((0, 6), (6, 12)).into_owned();
    let factor = h
        .view((6, 6), (12, 12))
        .into_owned()
        .cholesky()
        .expect("observable free point blocks");
    let expected_schur = a - &b * factor.solve(&b.transpose());
    let expected_g = g.rows(0, 6).into_owned() - &b * factor.solve(&g.rows(6, 12).into_owned());
    let matrix_tolerance = 2.0e-7 * expected_schur.amax().max(1.0);
    let gradient_tolerance = 2.0e-6 * expected_g.amax().max(1.0);
    let iterative = IterativeSchurOptions::new()
        .with_relative_tolerance(1.0e-8)
        .with_direct_fallback(false)
        .with_exact_camera_rank_diagnostics(true);
    for (name, policy) in [
        ("direct", SparseWorkspaceOptions::direct()),
        (
            "direct-parallel",
            SparseWorkspaceOptions::direct_parallel(2),
        ),
        (
            "cached-f64",
            SparseWorkspaceOptions::iterative_schur(iterative),
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
    ] {
        let mut workspace =
            SparseBundleAdjustmentWorkspace::try_new_with_fixed_blocks_and_pose_masks(
                3,
                measurements,
                ScaleGauge::Free,
                fixed,
                &masks,
                policy,
            )
            .expect("shared-pose topology");
        assert_eq!(
            workspace.core.layout.dimension(),
            6,
            "{name}: one physical pose has6 free coordinates"
        );
        assert_eq!(
            workspace.core.point_cross_offsets.last(),
            Some(&24),
            "{name}: one summed cross block per freepoint/pose"
        );
        let mut poses = initial_poses;
        let mut points = initial_points;
        let zero = refine_ba_multi_view_sparse_with_factors(
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
        .unwrap();
        assert_eq!(
            zero.observation_count, 32,
            "{name}: every exposure row remains"
        );
        assert_eq!(
            zero.diagnostics.pose_rank,
            SparseCameraRankAssessment::FullRank,
            "{name}"
        );
        for row in 0..6 {
            assert!(
                (workspace.core.gradient[row] - expected_g[row]).abs() < gradient_tolerance,
                "{name}: reduced gradient"
            );
            for column in 0..=row {
                let slot = workspace.core.scatter.position(row, column).unwrap();
                assert!(
                    (workspace.core.values[slot] - expected_schur[(row, column)]).abs()
                        < matrix_tolerance,
                    "{name}: Schur({row},{column})"
                );
            }
        }
        let initial_gradient = scaled_gradient(&h, &g);
        assert!(
            (zero.gradient_inf_norm - initial_gradient).abs() < 2.0e-6 * (1.0 + initial_gradient),
            "{name}: full gradient"
        );
        if name == "direct" {
            let mut changed = selectors.clone();
            changed[0] = 2;
            let changed_measurements = MultiViewMeasurements::FixedPerspectiveIndexed {
                observations: &observations,
                point_offsets: &offsets,
                cameras: &cameras,
                observation_intrinsics: &changed,
            };
            assert_eq!(
                refine_ba_multi_view_sparse_with_factors(
                    &mut poses,
                    &mut points,
                    changed_measurements,
                    ScaleGauge::Free,
                    fixed,
                    BundleAdjustmentPriors::NONE,
                    &mut workspace,
                    Options::l2(1),
                    NoiseModel::Unit
                ),
                Err(SparseBundleAdjustmentError::TopologyMismatch)
            );
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
        .unwrap();
        assert!(report.accepted_step_count > 0, "{name}");
        assert!(
            report.final_cost < 1.0e-8,
            "{name}: cost {}",
            report.final_cost
        );
        assert_eq!(
            report.diagnostics.pose_rank,
            SparseCameraRankAssessment::FullRank,
            "{name}"
        );
        assert_eq!(report.diagnostics.weak_point_count, 0, "{name}");
        if name.starts_with("cached") || name == "recomputed" {
            assert!(report.retained_camera_steps > 0, "{name}");
            assert_eq!(report.direct_fallback_steps, 0);
        }
        assert_eq!(poses[0], initial_poses[0]);
        assert_eq!(poses[2], initial_poses[2]);
        assert_eq!(&points[..4], &initial_points[..4]);
        assert!(
            (rotation(poses[1]) - rotation(truth_poses[1])).norm() < 5.0e-6,
            "{name}"
        );
        assert!(
            (Vector3::from(poses[1].translation) - Vector3::from(truth_poses[1].translation))
                .norm()
                < 2.0e-5,
            "{name}"
        );
        for point in 4..8 {
            assert!(
                (Vector3::from(points[point]) - Vector3::from(truth_points[point])).norm() < 2.0e-5,
                "{name}: point{point}"
            );
        }
        let (final_h, final_g) = normal_terms(
            &poses,
            &points,
            &observations,
            &selectors,
            &calibrations,
            report.normalization_scale,
        );
        let final_gradient = scaled_gradient(&final_h, &final_g);
        assert!(
            (report.gradient_inf_norm - final_gradient).abs() < 2.0e-8 * (1.0 + final_gradient),
            "{name}: fresh final gradient"
        );
    }
}
