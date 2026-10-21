// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, two_view::BundleAdjustmentError, two_view::TwoViewMeasurements,
    two_view::covariance_ba_2view,
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{
    self, ConvergenceCriteria, Loss, NoiseModel, Options, RefinementStatus,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    [
        pose.rotation[0][0] * point[0]
            + pose.rotation[0][1] * point[1]
            + pose.rotation[0][2] * point[2]
            + pose.translation[0],
        pose.rotation[1][0] * point[0]
            + pose.rotation[1][1] * point[1]
            + pose.rotation[1][2] * point[2]
            + pose.translation[1],
        pose.rotation[2][0] * point[0]
            + pose.rotation[2][1] * point[1]
            + pose.rotation[2][2] * point[2]
            + pose.translation[2],
    ]
}

fn pixel(camera: PinholeCamera, point: &[f64; 3]) -> [f64; 2] {
    [
        (camera.fx * point[0] + camera.skew * point[1]) / point[2] + camera.cx,
        camera.fy * point[1] / point[2] + camera.cy,
    ]
}

fn bearing(point: &[f64; 3]) -> [f64; 3] {
    let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
    point.map(|value| value / norm)
}

fn fixture() -> (RelativePose, RelativePose, Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let truth = RelativePose::new(manifold::exp_so3(&[0.04, -0.08, 0.025]), [0.8, 0.1, 0.15]);
    let points: Vec<_> = (0..30)
        .map(|index| {
            let x = (index % 6) as f64 * 0.22 - 0.55;
            let y = (index / 6) as f64 * 0.18 - 0.36;
            let z = 3.0 + 0.07 * index as f64;
            [x, y, z]
        })
        .collect();
    let rotation_error = manifold::exp_so3(&[0.025, -0.018, 0.012]);
    let baseline = truth
        .translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let raw_translation = [0.79, 0.125, 0.12];
    let raw_norm = raw_translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let initial = RelativePose::new(
        manifold::mat_mul(&rotation_error, &truth.rotation),
        raw_translation.map(|value| baseline * value / raw_norm),
    );
    let perturbed = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.025 * phase.sin(),
                point[1] + 0.02 * (0.7 * phase).cos(),
                point[2] + 0.04 * (0.3 * phase).sin(),
            ]
        })
        .collect();
    (truth, initial, points, perturbed)
}

fn assert_report_improves(
    report: &openmvg_ba::kernel::two_view::BundleAdjustmentReport,
    baseline: f64,
) {
    assert!(report.final_cost < report.initial_cost);
    assert!(report.final_rmse < 0.05 * report.initial_rmse);
    assert!(report.final_damping.is_finite() && report.final_damping > 0.0);
    assert!(report.last_gain_ratio.is_none_or(f64::is_finite));
    assert!(!matches!(report.status, RefinementStatus::Degenerate));
    assert!(report.diagnostics.is_well_constrained());
    assert_eq!(report.diagnostics.pose_rank, 5);
    assert_eq!(report.diagnostics.weak_point_count, 0);
    assert!(report.diagnostics.pose_hessian_reciprocal_condition_number > 0.0);
    assert!(
        report
            .diagnostics
            .minimum_point_hessian_reciprocal_condition_number
            > 0.0
    );
    let final_baseline = report
        .pose
        .translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!((final_baseline - baseline).abs() < 1e-12);
}

fn pose_error(estimate: &RelativePose, truth: &RelativePose) -> f64 {
    let truth_transpose: [[f64; 3]; 3] =
        std::array::from_fn(|row| std::array::from_fn(|column| truth.rotation[column][row]));
    let rotation_error =
        manifold::log_so3(&manifold::mat_mul(&estimate.rotation, &truth_transpose));
    let rotation = rotation_error
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let normalize = |translation: &[f64; 3]| {
        let norm = translation
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        translation.map(|value| value / norm)
    };
    let left = normalize(&estimate.translation);
    let right = normalize(&truth.translation);
    let translation = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .clamp(-1.0, 1.0)
        .acos();
    rotation.sqrt() + translation
}

#[test]
fn pixel_two_view_ba_refines_pose_and_points() {
    let (truth, initial, truth_points, initial_points) = fixture();
    let camera1 = PinholeCamera::new(500.0, 510.0, 320.0, 240.0);
    let camera2 = PinholeCamera::new(520.0, 515.0, 315.0, 238.0).with_skew(0.2);
    let observations1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera1, point))
        .collect();
    let observations2: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera2, &transform(&truth, point)))
        .collect();
    let baseline = initial
        .translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let measurements = TwoViewMeasurements::Pixel {
        observations1: &observations1,
        observations2: &observations2,
        cameras: [camera1, camera2],
    };
    let mut points = initial_points.clone();
    let report = openmvg_ba::kernel::two_view::refine_ba_2view(
        &initial,
        &mut points,
        measurements,
        Options::l2(40),
        NoiseModel::Unit,
    )
    .unwrap();
    assert_report_improves(&report, baseline);
}

#[test]
fn angular_two_view_ba_refines_pose_and_points() {
    let (truth, initial, truth_points, mut points) = fixture();
    let bearings1: Vec<_> = truth_points.iter().map(bearing).collect();
    let bearings2: Vec<_> = truth_points
        .iter()
        .map(|point| bearing(&transform(&truth, point)))
        .collect();
    let baseline = initial
        .translation
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let report = openmvg_ba::kernel::two_view::refine_ba_2view(
        &initial,
        &mut points,
        TwoViewMeasurements::Angular {
            bearings1: &bearings1,
            bearings2: &bearings2,
        },
        Options::l2(40),
        NoiseModel::Unit,
    )
    .unwrap();
    assert_report_improves(&report, baseline);
}

#[test]
fn point_only_error_cannot_trigger_reduced_gradient_convergence() {
    let (truth, _, truth_points, _) = fixture();
    let camera = PinholeCamera::new(500.0, 510.0, 320.0, 240.0);
    let observations1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, point))
        .collect();
    let observations2: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, &transform(&truth, point)))
        .collect();
    let mut points: Vec<_> = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 1e-3 * phase.sin(),
                point[1] + 8e-4 * (0.7 * phase).cos(),
                point[2] + 1.2e-3 * (0.3 * phase).sin(),
            ]
        })
        .collect();
    let options = Options::l2(10).with_convergence(ConvergenceCriteria::new(1e-2, 1e-14, 1e-14));
    let report = openmvg_ba::kernel::two_view::refine_ba_2view(
        &truth,
        &mut points,
        TwoViewMeasurements::Pixel {
            observations1: &observations1,
            observations2: &observations2,
            cameras: [camera; 2],
        },
        options,
        NoiseModel::Unit,
    )
    .expect("point-only perturbation is a valid BA problem");

    assert!(report.accepted_step_count > 0, "{report:#?}");
    assert!(report.final_cost < report.initial_cost, "{report:#?}");
}

#[test]
fn robust_pixel_ba_protects_the_pose_from_outliers() {
    let (truth, initial, truth_points, initial_points) = fixture();
    let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    let observations1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, point))
        .collect();
    let mut observations2: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, &transform(&truth, point)))
        .collect();
    for observation in observations2.iter_mut().step_by(5) {
        observation[0] += 120.0;
        observation[1] -= 80.0;
    }
    let measurements = TwoViewMeasurements::Pixel {
        observations1: &observations1,
        observations2: &observations2,
        cameras: [camera; 2],
    };
    let mut l2_points = initial_points.clone();
    let l2 = openmvg_ba::kernel::two_view::refine_ba_2view(
        &initial,
        &mut l2_points,
        measurements,
        Options::l2(35),
        NoiseModel::Unit,
    )
    .unwrap();
    let mut robust_points = initial_points;
    let robust = openmvg_ba::kernel::two_view::refine_ba_2view(
        &initial,
        &mut robust_points,
        measurements,
        Options::cauchy(2.0, 35),
        NoiseModel::Unit,
    )
    .unwrap();
    assert!(pose_error(&robust.pose, &truth) < pose_error(&l2.pose, &truth));
}

#[test]
fn two_view_covariance_is_whitened_and_loss_normalized() {
    let (truth, _, points, _) = fixture();
    let camera = PinholeCamera::new(500.0, 510.0, 320.0, 240.0);
    let observations1: Vec<_> = points.iter().map(|point| pixel(camera, point)).collect();
    let observations2: Vec<_> = points
        .iter()
        .map(|point| pixel(camera, &transform(&truth, point)))
        .collect();
    let measurements = TwoViewMeasurements::Pixel {
        observations1: &observations1,
        observations2: &observations2,
        cameras: [camera; 2],
    };
    let unit = covariance_ba_2view(&truth, &points, measurements, Loss::L2, NoiseModel::Unit)
        .expect("well-constrained two-view geometry has covariance");
    let doubled = covariance_ba_2view(
        &truth,
        &points,
        measurements,
        Loss::L2,
        NoiseModel::isotropic(2.0),
    )
    .expect("noise scaling preserves rank");
    let cauchy = covariance_ba_2view(
        &truth,
        &points,
        measurements,
        Loss::Cauchy { scale: 2.0 },
        NoiseModel::Unit,
    )
    .expect("Cauchy information is normalized at zero residual");
    assert_eq!(
        cauchy.uncertainty_model,
        refine::UncertaintyModel::RobustFixedWeightLinearized
    );
    assert!(unit.diagnostics.is_well_constrained());
    assert_eq!(unit.point_covariances.len(), points.len());
    for row in 0..5 {
        for column in 0..5 {
            let base = unit.pose_covariance[row][column];
            assert!((doubled.pose_covariance[row][column] / base - 4.0).abs() < 2e-8);
            assert!((cauchy.pose_covariance[row][column] / base - 1.0).abs() < 2e-8);
        }
    }
    assert_eq!(unit.pose_nees(&[0.0; 5]), 0.0);
    assert_eq!(unit.point_nees(0, &[0.0; 3]), Some(0.0));
    assert_eq!(unit.point_nees(points.len(), &[0.0; 3]), None);
}

#[test]
fn two_view_covariance_has_calibrated_nonlinear_nees() {
    const TRIALS: usize = 80;
    let (truth, _, truth_points, _) = fixture();
    let camera = PinholeCamera::new(500.0, 510.0, 320.0, 240.0);
    let exact1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, point))
        .collect();
    let exact2: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, &transform(&truth, point)))
        .collect();
    let sigma = 0.02;
    let mut rng = StdRng::seed_from_u64(0x2ba_cafe);
    let mut pose_nees = 0.0;
    let mut point_nees = 0.0;
    for _ in 0..TRIALS {
        let mut observations1 = exact1.clone();
        let mut observations2 = exact2.clone();
        for observation in observations1.iter_mut().chain(&mut observations2) {
            let x: f64 = StandardNormal.sample(&mut rng);
            let y: f64 = StandardNormal.sample(&mut rng);
            observation[0] += sigma * x;
            observation[1] += sigma * y;
        }
        let measurements = TwoViewMeasurements::Pixel {
            observations1: &observations1,
            observations2: &observations2,
            cameras: [camera; 2],
        };
        let mut estimated_points = truth_points.clone();
        let report = openmvg_ba::kernel::two_view::refine_ba_2view(
            &truth,
            &mut estimated_points,
            measurements,
            Options::l2(30),
            NoiseModel::isotropic(sigma),
        )
        .expect("noisy synthetic problem remains valid");
        assert!(report.diagnostics.is_well_constrained(), "{report:#?}");
        let covariance = covariance_ba_2view(
            &report.pose,
            &estimated_points,
            measurements,
            Loss::L2,
            NoiseModel::isotropic(sigma),
        )
        .expect("converged two-view geometry has covariance");
        let pose_error = two_view_pose_local_error(&report.pose, &truth);
        pose_nees += covariance.pose_nees(&pose_error);
        for (point, (estimate, reference)) in estimated_points.iter().zip(&truth_points).enumerate()
        {
            let error = std::array::from_fn(|coordinate| {
                (reference[coordinate] - estimate[coordinate]) / covariance.normalization_scale()
            });
            point_nees += covariance
                .point_nees(point, &error)
                .expect("each point has a covariance marginal");
        }
    }
    let mean_pose_nees = pose_nees / TRIALS as f64;
    let mean_point_nees = point_nees / (TRIALS * truth_points.len()) as f64;
    println!(
        "TWO_VIEW_BA_NEES,trials={TRIALS},pose={mean_pose_nees:.6},pose_dof=5,point={mean_point_nees:.6},point_dof=3"
    );
    assert!(
        (mean_pose_nees - 5.0).abs() < 1.2,
        "pose mean NEES {mean_pose_nees} should be close to 5"
    );
    assert!(
        (mean_point_nees - 3.0).abs() < 0.45,
        "point mean NEES {mean_point_nees} should be close to 3"
    );
}

fn two_view_pose_local_error(estimate: &RelativePose, reference: &RelativePose) -> [f64; 5] {
    let estimate_transpose =
        std::array::from_fn(|row| std::array::from_fn(|column| estimate.rotation[column][row]));
    let relative_rotation = manifold::mat_mul(&reference.rotation, &estimate_transpose);
    let rotation = manifold::log_so3(&relative_rotation);
    let normalize = |translation: &[f64; 3]| {
        let length = translation
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        translation.map(|value| value / length)
    };
    let estimate_direction = normalize(&estimate.translation);
    let reference_direction = normalize(&reference.translation);
    let cosine = estimate_direction
        .iter()
        .zip(reference_direction)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let scale = if angle < 1e-8 {
        1.0 + angle * angle / 6.0
    } else {
        angle / angle.sin()
    };
    let tangent: [f64; 3] = std::array::from_fn(|coordinate| {
        scale * (reference_direction[coordinate] - cosine * estimate_direction[coordinate])
    });
    let basis = manifold::s2_tangent_basis(&estimate_direction);
    [
        rotation[0],
        rotation[1],
        rotation[2],
        basis[0]
            .iter()
            .zip(tangent)
            .map(|(left, right)| left * right)
            .sum(),
        basis[1]
            .iter()
            .zip(tangent)
            .map(|(left, right)| left * right)
            .sum(),
    ]
}

#[test]
fn two_view_ba_rejects_gauge_and_measurement_errors() {
    let pose = RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let mut points = [[0.0, 0.0, 3.0]; 5];
    let observations = [[320.0, 240.0]; 5];
    let error = openmvg_ba::kernel::two_view::refine_ba_2view(
        &pose,
        &mut points,
        TwoViewMeasurements::Pixel {
            observations1: &observations,
            observations2: &observations[..4],
            cameras: [PinholeCamera::new(500.0, 500.0, 320.0, 240.0); 2],
        },
        Options::l2(1),
        NoiseModel::Unit,
    )
    .unwrap_err();
    assert_eq!(error, BundleAdjustmentError::InvalidTranslation);

    let valid_pose = RelativePose::new(pose.rotation, [1.0, 0.0, 0.0]);
    let error = openmvg_ba::kernel::two_view::refine_ba_2view(
        &valid_pose,
        &mut points,
        TwoViewMeasurements::Pixel {
            observations1: &observations,
            observations2: &observations[..4],
            cameras: [PinholeCamera::new(500.0, 500.0, 320.0, 240.0); 2],
        },
        Options::l2(1),
        NoiseModel::Unit,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        BundleAdjustmentError::MeasurementCountMismatch { .. }
    ));
}

#[test]
fn repeated_geometry_is_reported_as_degenerate() {
    let pose = RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0, 0.0],
    );
    let mut points = [[0.0, 0.0, 4.0]; 6];
    let observations1 = [[0.0, 0.0]; 6];
    let observations2 = [[0.25, 0.0]; 6];
    let report = openmvg_ba::kernel::two_view::refine_ba_2view(
        &pose,
        &mut points,
        TwoViewMeasurements::Pixel {
            observations1: &observations1,
            observations2: &observations2,
            cameras: [PinholeCamera::new(1.0, 1.0, 0.0, 0.0); 2],
        },
        Options::l2(0),
        NoiseModel::Unit,
    )
    .unwrap();
    assert_eq!(report.status, RefinementStatus::Degenerate);
    assert_eq!(report.final_damping, 1e-4);
    assert_eq!(report.last_gain_ratio, None);
    assert!(!report.diagnostics.is_well_constrained());
    assert!(report.diagnostics.pose_rank < 5);
    assert_eq!(
        report.diagnostics.pose_hessian_reciprocal_condition_number,
        0.0
    );
}

#[test]
fn low_parallax_exposes_weak_point_blocks() {
    const POINTS: usize = 20;
    let pose = RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0, 0.0],
    );
    let camera = PinholeCamera::new(1.0, 1.0, 0.0, 0.0);
    let truth_points: Vec<_> = (0..POINTS)
        .map(|index| {
            let depth = 1.0e7 + index as f64 * 1.0e4;
            let x = ((index % 5) as f64 - 2.0) * 0.1 * depth;
            let y = ((index / 5) as f64 - 1.5) * 0.1 * depth;
            [x, y, depth]
        })
        .collect();
    let observations1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, point))
        .collect();
    let observations2: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, &transform(&pose, point)))
        .collect();
    let mut points = truth_points;
    let report = openmvg_ba::kernel::two_view::refine_ba_2view(
        &pose,
        &mut points,
        TwoViewMeasurements::Pixel {
            observations1: &observations1,
            observations2: &observations2,
            cameras: [camera; 2],
        },
        Options::l2(0),
        NoiseModel::Unit,
    )
    .unwrap();

    assert_eq!(report.status, RefinementStatus::Degenerate);
    assert!(report.diagnostics.weak_point_count > 0, "{report:#?}");
    assert!(
        report
            .diagnostics
            .minimum_point_hessian_reciprocal_condition_number
            <= 64.0 * f64::EPSILON * 3.0
    );
}

#[test]
fn realistic_low_parallax_degrades_ba_conditioning_before_rank_is_lost() {
    const POINTS: usize = 36;
    let camera = PinholeCamera::new(1000.0, 1000.0, 640.0, 480.0);
    let rotation = manifold::exp_so3(&[0.002, -0.004, 0.001]);
    let truth_points: Vec<_> = (0..POINTS)
        .map(|index| {
            let depth = 20.0 + 1.1 * (index % 9) as f64;
            let x = ((index % 6) as f64 - 2.5) * 0.035 * depth;
            let y = ((index / 6) as f64 - 2.5) * 0.03 * depth;
            [x, y, depth]
        })
        .collect();
    let observations1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, point))
        .collect();
    let solve = |baseline: f64| {
        let pose = RelativePose::new(rotation, [baseline, 0.02 * baseline, 0.0]);
        let observations2: Vec<_> = truth_points
            .iter()
            .map(|point| pixel(camera, &transform(&pose, point)))
            .collect();
        let mut estimated_points = truth_points.clone();
        openmvg_ba::kernel::two_view::refine_ba_2view(
            &pose,
            &mut estimated_points,
            TwoViewMeasurements::Pixel {
                observations1: &observations1,
                observations2: &observations2,
                cameras: [camera; 2],
            },
            Options::l2(0),
            NoiseModel::isotropic(0.5),
        )
        .expect("both baselines define valid two-view geometry")
    };
    let ordinary = solve(0.5);
    let low = solve(0.01);
    println!(
        "BA_PARALLAX,ordinary={:?},low={:?}",
        ordinary.diagnostics, low.diagnostics
    );
    assert!(ordinary.diagnostics.is_well_constrained());
    assert!(low.diagnostics.pose_rank == 5);
    assert!(
        low.diagnostics.pose_hessian_reciprocal_condition_number
            < 1e-3
                * ordinary
                    .diagnostics
                    .pose_hessian_reciprocal_condition_number
    );
    assert!(
        low.diagnostics
            .minimum_point_hessian_reciprocal_condition_number
            < 1e-3
                * ordinary
                    .diagnostics
                    .minimum_point_hessian_reciprocal_condition_number
    );
}

#[test]
fn large_problem_uses_the_same_streaming_api() {
    const POINTS: usize = 2048;
    let truth = RelativePose::new(manifold::exp_so3(&[0.02, -0.04, 0.01]), [0.9, 0.1, 0.05]);
    let initial = RelativePose::new(
        manifold::mat_mul(&manifold::exp_so3(&[0.008, -0.006, 0.004]), &truth.rotation),
        [0.895, 0.12, 0.04],
    );
    let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    let truth_points: Vec<_> = (0..POINTS)
        .map(|index| {
            let x = (index % 64) as f64 * 0.025 - 0.8;
            let y = ((index / 64) % 32) as f64 * 0.03 - 0.45;
            let z = 3.0 + 0.002 * index as f64;
            [x, y, z]
        })
        .collect();
    let observations1: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, point))
        .collect();
    let observations2: Vec<_> = truth_points
        .iter()
        .map(|point| pixel(camera, &transform(&truth, point)))
        .collect();
    let mut points: Vec<_> = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as f64;
            [
                point[0] + 0.005 * phase.sin(),
                point[1] + 0.005 * phase.cos(),
                point[2],
            ]
        })
        .collect();
    let report = openmvg_ba::kernel::two_view::refine_ba_2view(
        &initial,
        &mut points,
        TwoViewMeasurements::Pixel {
            observations1: &observations1,
            observations2: &observations2,
            cameras: [camera; 2],
        },
        Options::l2(6),
        NoiseModel::Unit,
    )
    .unwrap();
    assert!(report.final_cost < report.initial_cost);
    assert!(report.final_rmse < 0.2 * report.initial_rmse);
}
