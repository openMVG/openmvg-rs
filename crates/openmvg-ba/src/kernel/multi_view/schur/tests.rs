// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::{DMatrix, DVector};

use super::assembly::{DenseHessian, ReducedBuffers, prepare_damping_diagonal};
use super::point::{
    linearize_point_with_hessian, multiply_2x3_3x3, multiply_pose, pose_motion, scale_2x3,
};
use super::priors::{pose_prior_jacobian, pose_prior_residual};
use super::*;
#[cfg(feature = "faer")]
use crate::kernel::multi_view::CameraPoseVariables;
use crate::kernel::multi_view::dense::{self, DenseBundleAdjustmentWorkspace};
use crate::kernel::multi_view::model::{ObservationModel, rotate, transform};
use crate::kernel::multi_view::{
    AngularObservation, BundleAdjustmentPriors, FixedVariableBlocks, MultiViewMeasurements,
    PixelObservation, PosePrior, ScaleGauge,
};
use crate::kernel::{PinholeCamera, RelativePose};
use crate::numerics::jet::Jet;
use crate::numerics::manifold;
use crate::numerics::refine::{Loss, NoiseModel};

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[test]
fn block_local_damping_floor_does_not_cross_contaminate_parameter_blocks() {
    let mut diagonal = [1.0e24, 1.0e-4, 4.0, 1.0];
    let layout = VariableLayout::calibrated(&[0, 2], &[2, 2]);

    let clamped = prepare_damping_diagonal(&mut diagonal, layout);

    let first_block_floor = 64.0 * f64::EPSILON * 1.0e24;
    assert_eq!(clamped, 1);
    assert_eq!(diagonal, [1.0e24, first_block_floor, 4.0, 1.0]);
}

#[test]
fn block_without_curvature_keeps_a_damping_floor_that_survives_small_lambda() {
    let mut diagonal = [0.0, 0.0, 4.0, 1.0];
    let layout = VariableLayout::calibrated(&[0, 2], &[2, 2]);

    let clamped = prepare_damping_diagonal(&mut diagonal, layout);

    assert_eq!(clamped, 2);
    assert_eq!(diagonal, [1.0, 1.0, 4.0, 1.0]);
    // A MIN_POSITIVE-relative floor would vanish for lambda < 7.8e-3.
    assert!(1.0e-12 * diagonal[0] > 0.0);
}

#[test]
fn disabled_dense_rank_diagnostics_are_reported_as_unassessed() {
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(2, 1)
        .expect("small workspace")
        .with_exact_camera_rank_diagnostics(false);
    let assembly = Assembly {
        valid: true,
        weak_point_count: 0,
        minimum_point_rcond: 1.0,
        scaled_gradient_inf_norm: 0.0,
        point_model_reduction: 0.0,
        cost: 0.0,
        squared_residual: 0.0,
        minimum_reduced_diagonal: 0.0,
        maximum_reduced_diagonal: 0.0,
        damping_floor_clamp_count: 0,
    };

    let diagnostics = diagnostics_with_expected_nullity(&mut workspace, 6, 1, &assembly);

    assert!(!diagnostics.pose_rank_assessed);
    assert_eq!(diagnostics.pose_rank, 0);
    assert_eq!(
        diagnostics.observable_camera_hessian_reciprocal_condition_number,
        0.0
    );
    assert!(diagnostics.point_rank_assessed);
    assert!(diagnostics.is_well_constrained());
}

#[test]
fn skipped_final_diagnostics_are_explicitly_unassessed() {
    let diagnostics = unassessed_diagnostics(12, 1);

    assert!(!diagnostics.pose_rank_assessed);
    assert!(!diagnostics.point_rank_assessed);
    assert_eq!(diagnostics.weak_point_count, 0);
    assert!(diagnostics.is_well_constrained());
}

#[test]
fn candidate_point_holds_a_point_its_own_step_makes_degenerate() {
    let camera = PinholeCamera::new(430.0, 415.0, 0.0, 0.0);
    let observations = [
        PixelObservation::new(0, [10.0, 5.0]),
        PixelObservation::new(1, [-10.0, 5.0]),
    ];
    let offsets = [0, 2];
    let cameras = [camera, camera];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &offsets,
        cameras: &cameras,
    });
    let at = |center: [f64; 3]| RelativePose {
        rotation: IDENTITY,
        translation: center.map(|value| -value),
    };
    // Camera 0 sits at the origin and camera 1 at (1, 0, 0).
    let poses = [at([0.0; 3]), at([1.0, 0.0, 0.0])];
    let previous = [0.5, 0.0, 5.0];
    let candidate = |candidate_poses: &[RelativePose], stepped: [f64; 3]| {
        candidate_point(
            model,
            &poses,
            candidate_poses,
            &previous,
            stepped,
            0,
            Loss::L2,
            NoiseModel::Unit,
            5.0,
        )
    };

    let moved = candidate(&poses, [0.6, 0.0, 5.0]).expect("admissible step");
    assert!(moved.moved);
    assert_eq!(moved.position, [0.6, 0.0, 5.0]);
    // In front of both cameras, but one ten-millionth of the viewing distance from camera 1, and
    // seen by both cameras along the same ray: each is held at its previous position.
    for degenerate in [[1.0 + 1.0e-7, 0.0, 1.0e-7], [0.0, 0.0, 1.0e6]] {
        let held = candidate(&poses, degenerate).expect("the point step alone is degenerate");
        assert!(!held.moved);
        assert_eq!(held.position, previous);
    }
    // A camera step onto the previous position rejects the whole candidate.
    let collapsed = [at([0.0; 3]), at([0.5, 0.0, 5.0 - 1.0e-6])];
    assert!(candidate(&collapsed, [0.0, 0.0, 1.0e6]).is_none());
}

#[test]
fn fixed_baseline_pixel_jacobian_matches_the_retraction() {
    let camera = PinholeCamera::new(430.0, 415.0, 13.0, -8.0).with_skew(0.7);
    let observation = [PixelObservation::new(1, [91.3, -37.1])];
    let offsets = [0, 1];
    let cameras = [camera, camera];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observation,
        point_offsets: &offsets,
        cameras: &cameras,
    });
    check_retraction_jacobian(model, ScaleGauge::FixedBaseline { camera: 1 });
}

#[test]
fn free_scale_angular_jacobian_matches_the_retraction() {
    let observation = [AngularObservation::new(1, normalize([0.18, -0.09, 1.0]))];
    let offsets = [0, 1];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Angular {
        observations: &observation,
        point_offsets: &offsets,
    });
    check_retraction_jacobian(model, ScaleGauge::Free);
}

#[test]
fn fixed_translation_component_jacobian_matches_the_retraction() {
    let camera = PinholeCamera::new(430.0, 415.0, 13.0, -8.0);
    let observation = [PixelObservation::new(1, [91.3, -37.1])];
    let offsets = [0, 1];
    let cameras = [camera, camera];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observation,
        point_offsets: &offsets,
        cameras: &cameras,
    });
    check_retraction_jacobian(
        model,
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: super::super::TranslationComponent::X,
        },
    );
}

#[test]
fn local_pose_and_point_charts_match_dual_oracle() {
    let camera = PinholeCamera::new(430.0, 415.0, 13.0, -8.0).with_skew(0.7);
    let observation = [PixelObservation::new(1, [91.3, -37.1])];
    let offsets = [0, 1];
    let cameras = [camera, camera];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observation,
        point_offsets: &offsets,
        cameras: &cameras,
    });
    for gauge in [
        ScaleGauge::Free,
        ScaleGauge::FixedBaseline { camera: 1 },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: super::super::TranslationComponent::X,
        },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: super::super::TranslationComponent::Y,
        },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: super::super::TranslationComponent::Z,
        },
    ] {
        check_dual_retraction_jacobian(model, camera, observation[0].point, gauge);
    }
}

#[test]
fn tiny_global_normal_equations_match_dual_oracle() {
    let camera = PinholeCamera::new(430.0, 415.0, 13.0, -8.0).with_skew(0.7);
    let pose = RelativePose::new(manifold::exp_so3(&[0.09, -0.06, 0.04]), [0.72, -0.13, 0.22]);
    let poses = [RelativePose::new(IDENTITY, [0.0; 3]), pose];
    let point = [0.34, -0.21, 3.7];
    let observations = [
        PixelObservation::new(0, [51.3, -27.1]),
        PixelObservation::new(1, [91.3, -37.1]),
    ];
    let point_offsets = [0, 2];
    let cameras = [camera, camera];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    });
    let gauge_kind = ScaleGauge::FixedBaseline { camera: 1 };
    let gauge = GaugeState::new(gauge_kind, &poses, 0.83);
    let mut workspace = DenseBundleAdjustmentWorkspace::try_new(2, 1).unwrap();
    let camera_dimension = configure_layout(&mut workspace, 2, gauge_kind, 0);
    assert_eq!(camera_dimension, 5);
    let mut camera_hessian = vec![0.0; camera_dimension * camera_dimension];
    let mut camera_gradient = vec![0.0; camera_dimension];
    let mut camera_diagonal = vec![0.0; camera_dimension];
    let mut ignored_gradient = vec![0.0; camera_dimension];
    let mut cross = vec![0.0; camera_dimension * 3];
    let mut reduced_hessian = DenseHessian {
        values: &mut camera_hessian,
        dimension: camera_dimension,
    };
    let point_linearization = linearize_point_with_hessian::<true, false, _>(
        model,
        &poses,
        &point,
        0,
        gauge,
        Loss::L2,
        NoiseModel::Unit,
        VariableLayout {
            offsets: &workspace.camera_offsets,
            pose_dimensions: &workspace.camera_pose_dimensions,
            intrinsic_offsets: &workspace.intrinsic_offsets,
            intrinsic_dimensions: &workspace.intrinsic_dimensions,
            pose_masks: &[],
            #[cfg(feature = "faer")]
            pose_variables: CameraPoseVariables::RotationAndTranslation,
        },
        &mut cross,
        Some(ReducedBuffers {
            hessian: &mut reduced_hessian,
            gradient: &mut camera_gradient,
            full_gradient: &mut ignored_gradient,
            diagonal: &mut camera_diagonal,
        }),
    )
    .unwrap();

    let fixed_camera_point =
        oracle_fixed_camera_point(poses[0], point, camera_dimension, gauge.scene_scale);
    let moving_camera_point = oracle_camera_point(pose, point, gauge, camera_dimension);
    let dual_residuals = [
        oracle_pixel_residual(camera, observations[0].point, &fixed_camera_point),
        oracle_pixel_residual(camera, observations[1].point, &moving_camera_point),
    ];
    let mut dual_gradient = [0.0; 8];
    let mut dual_hessian = [[0.0; 8]; 8];
    for residual in dual_residuals.iter().flatten() {
        let value = residual.value;
        for (row, (gradient, hessian_row)) in dual_gradient
            .iter_mut()
            .zip(dual_hessian.iter_mut())
            .enumerate()
        {
            let row_derivative = residual.deriv[row];
            *gradient += row_derivative * value;
            for (column, hessian) in hessian_row.iter_mut().enumerate() {
                *hessian += row_derivative * residual.deriv[column];
            }
        }
    }

    let mut analytic_gradient = [0.0; 8];
    let mut analytic_hessian = [[0.0; 8]; 8];
    analytic_gradient[..camera_dimension].copy_from_slice(&camera_gradient);
    analytic_gradient[camera_dimension..].copy_from_slice(&point_linearization.gradient);
    for row in 0..camera_dimension {
        for column in 0..camera_dimension {
            analytic_hessian[row][column] = camera_hessian[row * camera_dimension + column];
        }
        for coordinate in 0..3 {
            let value = cross[row * 3 + coordinate];
            analytic_hessian[row][camera_dimension + coordinate] = value;
            analytic_hessian[camera_dimension + coordinate][row] = value;
        }
    }
    for row in 0..3 {
        for column in 0..3 {
            analytic_hessian[camera_dimension + row][camera_dimension + column] =
                point_linearization.hessian[row][column];
        }
    }
    for ((dual_gradient, analytic_gradient), (dual_row, analytic_row)) in dual_gradient
        .iter()
        .zip(&analytic_gradient)
        .zip(dual_hessian.iter().zip(&analytic_hessian))
    {
        assert_close(*dual_gradient, *analytic_gradient, 1e-11);
        for (dual, analytic) in dual_row.iter().zip(analytic_row) {
            assert_close(*dual, *analytic, 1e-11);
        }
    }
}

#[test]
fn expected_scale_nullity_respects_partial_prior_information() {
    let poses = [
        RelativePose::new(IDENTITY, [0.0; 3]),
        RelativePose::new(IDENTITY, [0.7, -0.2, 0.1]),
    ];
    let points = [[0.0; 3], [0.4, -0.3, 3.0]];
    let rotation_information = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column && row < 3 { 1.0 } else { 0.0 })
    });
    let rotation_prior = [super::super::PosePrior {
        camera: 1,
        mean: poses[1],
        sqrt_information: rotation_information,
    }];
    assert_eq!(
        expected_gauge_nullity(
            ScaleGauge::Free,
            &poses,
            &points,
            FixedVariableBlocks::NONE,
            BundleAdjustmentPriors {
                poses: &rotation_prior,
                points: &[],
                intrinsics: &[],
            },
        ),
        1
    );

    let translation_information = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column && row >= 3 { 1.0 } else { 0.0 })
    });
    let translation_prior = [super::super::PosePrior {
        camera: 1,
        mean: poses[1],
        sqrt_information: translation_information,
    }];
    assert_eq!(
        expected_gauge_nullity(
            ScaleGauge::Free,
            &poses,
            &points,
            FixedVariableBlocks::NONE,
            BundleAdjustmentPriors {
                poses: &translation_prior,
                points: &[],
                intrinsics: &[],
            },
        ),
        0
    );

    let fixed_origin = [true, false];
    let fixed_nonzero = [false, true];
    assert_eq!(
        expected_gauge_nullity(
            ScaleGauge::Free,
            &poses,
            &points,
            FixedVariableBlocks {
                camera_poses: &[],
                points: &fixed_origin,
                intrinsics: &[],
                intrinsic_dimensions: &[],
            },
            BundleAdjustmentPriors::NONE,
        ),
        1
    );
    assert_eq!(
        expected_gauge_nullity(
            ScaleGauge::Free,
            &poses,
            &points,
            FixedVariableBlocks {
                camera_poses: &[],
                points: &fixed_nonzero,
                intrinsics: &[],
                intrinsic_dimensions: &[],
            },
            BundleAdjustmentPriors::NONE,
        ),
        0
    );
}

#[test]
fn pose_prior_jacobian_matches_each_translation_chart() {
    let pose = RelativePose::new(manifold::exp_so3(&[0.2, -0.1, 0.05]), [0.7, -0.2, 0.1]);
    let mean = RelativePose::new(manifold::exp_so3(&[-0.08, 0.04, 0.03]), [0.6, -0.1, 0.15]);
    let poses = [RelativePose::new(IDENTITY, [0.0; 3]), pose];
    for gauge_kind in [
        ScaleGauge::Free,
        ScaleGauge::FixedBaseline { camera: 1 },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: super::super::TranslationComponent::X,
        },
    ] {
        let gauge = GaugeState::new(gauge_kind, &poses, 0.9);
        let dimension = if gauge.fixed_camera.is_some() { 5 } else { 6 };
        let residual = pose_prior_residual(&pose, &mean);
        let analytic = pose_prior_jacobian(&residual, 1, gauge, dimension);
        let epsilon = 2e-7;
        for parameter in 0..dimension {
            let mut positive = [0.0; 6];
            positive[parameter] = epsilon;
            let mut negative = [0.0; 6];
            negative[parameter] = -epsilon;
            let plus =
                pose_prior_residual(&retract_test_pose(pose, gauge, &positive, dimension), &mean);
            let minus =
                pose_prior_residual(&retract_test_pose(pose, gauge, &negative, dimension), &mean);
            for row in 0..6 {
                let numerical = (plus[row] - minus[row]) / (2.0 * epsilon);
                assert_close(numerical, analytic[row][parameter], 3e-7);
            }
        }
    }
}

#[test]
fn dense_normal_equations_and_schur_give_the_same_step() {
    let (poses, points, observations, point_offsets, cameras) = pixel_fixture(8, true);
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    });
    let gauge_kind = ScaleGauge::FixedBaseline { camera: 1 };
    let gauge = GaugeState::new(gauge_kind, &poses, 1.0);
    let mut workspace =
        DenseBundleAdjustmentWorkspace::try_new(3, points.len()).expect("small fixed workspace");
    let dimension = configure_layout(
        &mut workspace,
        poses.len(),
        gauge_kind,
        model.intrinsic_dimension(),
    );
    let damping = 1e-3;
    let full_dimension = dimension + 3 * points.len();
    let mut full_hessian = DMatrix::zeros(full_dimension, full_dimension);
    let mut full_gradient = DVector::zeros(full_dimension);
    let mut camera_hessian = vec![0.0; dimension * dimension];
    let mut camera_gradient = vec![0.0; dimension];
    let mut camera_diagonal = vec![0.0; dimension];
    let mut ignored_gradient = vec![0.0; dimension];
    let mut cross = vec![0.0; dimension * 3];

    for (point_index, point) in points.iter().enumerate() {
        cross.fill(0.0);
        let mut reduced_hessian = DenseHessian {
            values: &mut camera_hessian,
            dimension,
        };
        let local = linearize_point_with_hessian::<true, false, _>(
            model,
            &poses,
            point,
            point_index,
            gauge,
            Loss::L2,
            NoiseModel::Unit,
            VariableLayout {
                offsets: &workspace.camera_offsets,
                pose_dimensions: &workspace.camera_pose_dimensions,
                intrinsic_offsets: &workspace.intrinsic_offsets,
                intrinsic_dimensions: &workspace.intrinsic_dimensions,
                pose_masks: &[],
                #[cfg(feature = "faer")]
                pose_variables: CameraPoseVariables::RotationAndTranslation,
            },
            &mut cross,
            Some(ReducedBuffers {
                hessian: &mut reduced_hessian,
                gradient: &mut camera_gradient,
                full_gradient: &mut ignored_gradient,
                diagonal: &mut camera_diagonal,
            }),
        )
        .expect("fixture is projectable");
        let point_offset = dimension + 3 * point_index;
        for row in 0..3 {
            full_gradient[point_offset + row] = local.gradient[row];
            for column in 0..3 {
                full_hessian[(point_offset + row, point_offset + column)] =
                    local.hessian[row][column];
            }
        }
        for camera_parameter in 0..dimension {
            for coordinate in 0..3 {
                let value = cross[camera_parameter * 3 + coordinate];
                full_hessian[(camera_parameter, point_offset + coordinate)] = value;
                full_hessian[(point_offset + coordinate, camera_parameter)] = value;
            }
        }
        let point_scale = (0..3)
            .map(|index| local.hessian[index][index])
            .fold(0.0, f64::max);
        let point_floor = 64.0 * f64::EPSILON * point_scale.max(f64::MIN_POSITIVE);
        for coordinate in 0..3 {
            full_hessian[(point_offset + coordinate, point_offset + coordinate)] +=
                damping * local.hessian[coordinate][coordinate].max(point_floor);
        }
    }
    for row in 0..dimension {
        full_gradient[row] = camera_gradient[row];
        for column in 0..dimension {
            full_hessian[(row, column)] = camera_hessian[row * dimension + column];
        }
    }
    let camera_scale = camera_diagonal.iter().copied().fold(0.0, f64::max);
    let camera_floor = 64.0 * f64::EPSILON * camera_scale.max(f64::MIN_POSITIVE);
    for parameter in 0..dimension {
        full_hessian[(parameter, parameter)] +=
            damping * camera_diagonal[parameter].max(camera_floor);
    }
    let full_step = full_hessian
        .lu()
        .solve(&(-full_gradient))
        .expect("damped full system is nonsingular");

    let assembly = assemble_problem(
        BaProblemView {
            model,
            poses: &poses,
            points: &points,
            fixed: FixedVariableBlocks::NONE,
            priors: BundleAdjustmentPriors::NONE,
        },
        LinearizationPolicy {
            gauge,
            loss: Loss::L2,
            noise: NoiseModel::Unit,
            damping,
            prior_sqrt_scale: 1.0,
        },
        dimension,
        &mut workspace,
    );
    assert!(assembly.valid);
    assert!(dense::solve_camera_step(&mut workspace, dimension));
    for parameter in 0..dimension {
        assert_close(workspace.step[parameter], full_step[parameter], 2e-9);
    }
    let layout = VariableLayout {
        offsets: &workspace.camera_offsets,
        pose_dimensions: &workspace.camera_pose_dimensions,
        intrinsic_offsets: &workspace.intrinsic_offsets,
        intrinsic_dimensions: &workspace.intrinsic_dimensions,
        pose_masks: &[],
        #[cfg(feature = "faer")]
        pose_variables: CameraPoseVariables::RotationAndTranslation,
    };
    for (point_index, point) in points.iter().enumerate() {
        let (step, _) = point_step_with_prior_and_damping_norm(
            model,
            &poses,
            point,
            point_index,
            gauge,
            Loss::L2,
            NoiseModel::Unit,
            damping,
            &workspace.step[..dimension],
            layout,
            &mut workspace.cross,
            &mut workspace.active_parameters,
            &mut workspace.parameter_marks,
            None,
        )
        .expect("damped point block is nonsingular");
        for coordinate in 0..3 {
            assert_close(
                step[coordinate],
                full_step[dimension + 3 * point_index + coordinate],
                2e-9,
            );
        }
    }
}

#[test]
fn free_scale_direction_is_a_schur_null_vector() {
    let (poses, points, observations, point_offsets, cameras) = pixel_fixture(12, false);
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    });
    let gauge_kind = ScaleGauge::Free;
    let scene_scale = 0.8;
    let gauge = GaugeState::new(gauge_kind, &poses, scene_scale);
    let mut workspace =
        DenseBundleAdjustmentWorkspace::try_new(3, points.len()).expect("small fixed workspace");
    let dimension = configure_layout(
        &mut workspace,
        poses.len(),
        gauge_kind,
        model.intrinsic_dimension(),
    );
    let assembly = assemble_problem(
        BaProblemView {
            model,
            poses: &poses,
            points: &points,
            fixed: FixedVariableBlocks::NONE,
            priors: BundleAdjustmentPriors::NONE,
        },
        LinearizationPolicy {
            gauge,
            loss: Loss::L2,
            noise: NoiseModel::Unit,
            damping: 0.0,
            prior_sqrt_scale: 1.0,
        },
        dimension,
        &mut workspace,
    );
    assert!(assembly.valid);
    let mut scale_direction = vec![0.0; dimension];
    for (camera, pose) in poses.iter().enumerate().skip(1) {
        let offset = workspace.camera_offsets[camera];
        for coordinate in 0..3 {
            scale_direction[offset + 3 + coordinate] = pose.translation[coordinate] / scene_scale;
        }
    }
    let product: Vec<f64> = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| workspace.hessian[row * dimension + column] * scale_direction[column])
                .sum()
        })
        .collect();
    let matrix_norm = workspace.hessian[..dimension * dimension]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let direction_norm = norm_slice(&scale_direction);
    assert!(norm_slice(&product) <= 2e-11 * matrix_norm * direction_norm);
}

fn check_retraction_jacobian(model: ObservationModel<'_>, gauge_kind: ScaleGauge) {
    let pose = RelativePose::new(manifold::exp_so3(&[0.09, -0.06, 0.04]), [0.72, -0.13, 0.22]);
    let poses = [RelativePose::new(IDENTITY, [0.0; 3]), pose];
    let point = [0.34, -0.21, 3.7];
    let scene_scale = 0.83;
    let gauge = GaugeState::new(gauge_kind, &poses, scene_scale);
    let rotated = rotate(&pose, &point);
    let projection = model
        .project(0, &transform(&pose, &point))
        .expect("fixture is projectable");
    let camera_dimension = if gauge.fixed_camera == Some(1) { 5 } else { 6 };
    let analytic_pose = multiply_pose(&projection.jacobian, &pose_motion(&rotated, 1, gauge));
    let analytic_point = scale_2x3(
        multiply_2x3_3x3(&projection.jacobian, &pose.rotation),
        scene_scale,
    );
    let epsilon = 2e-7;
    for parameter in 0..camera_dimension {
        let mut positive = [0.0; 6];
        positive[parameter] = epsilon;
        let mut negative = [0.0; 6];
        negative[parameter] = -epsilon;
        let plus = retract_test_pose(pose, gauge, &positive, camera_dimension);
        let minus = retract_test_pose(pose, gauge, &negative, camera_dimension);
        let plus_residual = model
            .project(0, &transform(&plus, &point))
            .expect("positive perturbation is projectable")
            .residual;
        let minus_residual = model
            .project(0, &transform(&minus, &point))
            .expect("negative perturbation is projectable")
            .residual;
        for row in 0..2 {
            let numerical = (plus_residual[row] - minus_residual[row]) / (2.0 * epsilon);
            assert_close(numerical, analytic_pose[row][parameter], 3e-6);
        }
    }
    for coordinate in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[coordinate] += scene_scale * epsilon;
        minus[coordinate] -= scene_scale * epsilon;
        let plus_residual = model
            .project(0, &transform(&pose, &plus))
            .expect("positive perturbation is projectable")
            .residual;
        let minus_residual = model
            .project(0, &transform(&pose, &minus))
            .expect("negative perturbation is projectable")
            .residual;
        for row in 0..2 {
            let numerical = (plus_residual[row] - minus_residual[row]) / (2.0 * epsilon);
            assert_close(numerical, analytic_point[row][coordinate], 3e-6);
        }
    }
}

type OracleJet = Jet<9>;

fn oracle_exp_so3(delta: &[OracleJet; 3]) -> [[OracleJet; 3]; 3] {
    let zero = OracleJet::constant(0.0);
    let one = OracleJet::constant(1.0);
    let skew = [
        [zero, -delta[2], delta[1]],
        [delta[2], zero, -delta[0]],
        [-delta[1], delta[0], zero],
    ];
    let skew_squared: [[OracleJet; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| skew[row][inner] * skew[inner][column])
                .fold(zero, |sum, value| sum + value)
        })
    });
    let theta_squared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    // Zero-safe Taylor forms avoid differentiating sqrt(theta²) at zero.
    let sine_over_theta = one - theta_squared * OracleJet::constant(1.0 / 6.0)
        + theta_squared * theta_squared * OracleJet::constant(1.0 / 120.0);
    let one_minus_cosine_over_theta_squared = OracleJet::constant(0.5)
        - theta_squared * OracleJet::constant(1.0 / 24.0)
        + theta_squared * theta_squared * OracleJet::constant(1.0 / 720.0);
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            OracleJet::constant((row == column) as u8 as f64)
                + sine_over_theta * skew[row][column]
                + one_minus_cosine_over_theta_squared * skew_squared[row][column]
        })
    })
}

fn oracle_mat_vec(matrix: &[[OracleJet; 3]; 3], vector: &[OracleJet; 3]) -> [OracleJet; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| matrix[row][column] * vector[column])
            .fold(OracleJet::constant(0.0), |sum, value| sum + value)
    })
}

fn oracle_pixel_residual(
    camera: PinholeCamera,
    observation: [f64; 2],
    point: &[OracleJet; 3],
) -> [OracleJet; 2] {
    let inverse_depth = point[2].recip();
    let x_numerator =
        OracleJet::constant(camera.fx) * point[0] + OracleJet::constant(camera.skew) * point[1];
    [
        x_numerator * inverse_depth + OracleJet::constant(camera.cx - observation[0]),
        OracleJet::constant(camera.fy) * point[1] * inverse_depth
            + OracleJet::constant(camera.cy - observation[1]),
    ]
}

fn oracle_camera_point(
    pose: RelativePose,
    point: [f64; 3],
    gauge: GaugeState,
    camera_dimension: usize,
) -> [OracleJet; 3] {
    let rotation_delta: [OracleJet; 3] =
        std::array::from_fn(|parameter| OracleJet::variable(0.0, parameter));
    let point_delta: [OracleJet; 3] = std::array::from_fn(|coordinate| {
        OracleJet::constant(point[coordinate])
            + OracleJet::variable(0.0, camera_dimension + coordinate)
                * OracleJet::constant(gauge.scene_scale)
    });
    let base_rotation = pose.rotation.map(|row| row.map(OracleJet::constant));
    let rotated = oracle_mat_vec(&base_rotation, &point_delta);
    let rotation_update = oracle_exp_so3(&rotation_delta);
    let rotated = oracle_mat_vec(&rotation_update, &rotated);
    let translation = if camera_dimension == 5 && gauge.fixed_component.is_none() {
        let tangent = [OracleJet::variable(0.0, 3), OracleJet::variable(0.0, 4)];
        let unnormalized: [OracleJet; 3] = std::array::from_fn(|coordinate| {
            OracleJet::constant(gauge.direction[coordinate])
                + OracleJet::constant(gauge.basis[0][coordinate]) * tangent[0]
                + OracleJet::constant(gauge.basis[1][coordinate]) * tangent[1]
        });
        let norm = (unnormalized[0] * unnormalized[0]
            + unnormalized[1] * unnormalized[1]
            + unnormalized[2] * unnormalized[2])
            .sqrt();
        unnormalized.map(|value| OracleJet::constant(gauge.baseline) * value / norm)
    } else if let Some(fixed) = gauge.fixed_component {
        let mut local = 3;
        std::array::from_fn(|coordinate| {
            if coordinate == fixed {
                OracleJet::constant(gauge.fixed_component_value)
            } else {
                let value = OracleJet::constant(pose.translation[coordinate])
                    + OracleJet::variable(0.0, local) * OracleJet::constant(gauge.scene_scale);
                local += 1;
                value
            }
        })
    } else {
        std::array::from_fn(|coordinate| {
            OracleJet::constant(pose.translation[coordinate])
                + OracleJet::variable(0.0, 3 + coordinate) * OracleJet::constant(gauge.scene_scale)
        })
    };
    std::array::from_fn(|coordinate| rotated[coordinate] + translation[coordinate])
}

fn oracle_fixed_camera_point(
    pose: RelativePose,
    point: [f64; 3],
    point_offset: usize,
    scene_scale: f64,
) -> [OracleJet; 3] {
    let point_delta = std::array::from_fn(|coordinate| {
        OracleJet::constant(point[coordinate])
            + OracleJet::variable(0.0, point_offset + coordinate) * OracleJet::constant(scene_scale)
    });
    let rotation = pose.rotation.map(|row| row.map(OracleJet::constant));
    let rotated = oracle_mat_vec(&rotation, &point_delta);
    std::array::from_fn(|coordinate| {
        rotated[coordinate] + OracleJet::constant(pose.translation[coordinate])
    })
}

fn check_dual_retraction_jacobian(
    model: ObservationModel<'_>,
    camera: PinholeCamera,
    observation: [f64; 2],
    gauge_kind: ScaleGauge,
) {
    let pose = RelativePose::new(manifold::exp_so3(&[0.09, -0.06, 0.04]), [0.72, -0.13, 0.22]);
    let poses = [RelativePose::new(IDENTITY, [0.0; 3]), pose];
    let point = [0.34, -0.21, 3.7];
    let gauge = GaugeState::new(gauge_kind, &poses, 0.83);
    let camera_dimension = if gauge.fixed_camera == Some(1) { 5 } else { 6 };
    let rotated = rotate(&pose, &point);
    let projection = model
        .project(0, &transform(&pose, &point))
        .expect("fixture is projectable");
    let analytic_pose = multiply_pose(&projection.jacobian, &pose_motion(&rotated, 1, gauge));
    let analytic_point = scale_2x3(
        multiply_2x3_3x3(&projection.jacobian, &pose.rotation),
        gauge.scene_scale,
    );
    let dual_point = oracle_camera_point(pose, point, gauge, camera_dimension);
    let dual = oracle_pixel_residual(camera, observation, &dual_point);
    for row in 0..2 {
        assert_close(dual[row].value, projection.residual[row], 5e-12);
        for (parameter, expected) in analytic_pose[row].iter().take(camera_dimension).enumerate() {
            assert_close(dual[row].deriv[parameter], *expected, 5e-12);
        }
        for (coordinate, expected) in analytic_point[row].iter().enumerate() {
            assert_close(
                dual[row].deriv[camera_dimension + coordinate],
                *expected,
                5e-12,
            );
        }
    }
}

fn retract_test_pose(
    pose: RelativePose,
    gauge: GaugeState,
    step: &[f64; 6],
    camera_dimension: usize,
) -> RelativePose {
    let rotation = manifold::mat_mul(
        &manifold::exp_so3(&[step[0], step[1], step[2]]),
        &pose.rotation,
    );
    let translation = if camera_dimension == 5 && gauge.fixed_component.is_none() {
        manifold::retract_s2(&gauge.direction, &gauge.basis, &[step[3], step[4]])
            .map(|value| gauge.baseline * value)
    } else if let Some(fixed) = gauge.fixed_component {
        let mut translation = pose.translation;
        let mut local = 3;
        for (coordinate, value) in translation.iter_mut().enumerate() {
            if coordinate != fixed {
                *value += gauge.scene_scale * step[local];
                local += 1;
            }
        }
        translation
    } else {
        std::array::from_fn(|coordinate| {
            pose.translation[coordinate] + gauge.scene_scale * step[3 + coordinate]
        })
    };
    RelativePose::new(rotation, translation)
}

type PixelFixture = (
    [RelativePose; 3],
    Vec<[f64; 3]>,
    Vec<PixelObservation>,
    Vec<usize>,
    [PinholeCamera; 3],
);

fn pixel_fixture(point_count: usize, perturb_observations: bool) -> PixelFixture {
    let poses = [
        RelativePose::new(IDENTITY, [0.0; 3]),
        RelativePose::new(manifold::exp_so3(&[0.01, -0.04, 0.02]), [1.0, 0.08, 0.03]),
        RelativePose::new(manifold::exp_so3(&[-0.03, 0.02, 0.01]), [0.22, 0.91, 0.11]),
    ];
    let cameras = [PinholeCamera::new(470.0, 450.0, 8.0, -5.0); 3];
    let points: Vec<_> = (0..point_count)
        .map(|index| {
            let x = index as f64;
            [
                0.17 * (x - 3.0),
                0.23 * ((index % 4) as f64 - 1.5),
                3.2 + 0.19 * x,
            ]
        })
        .collect();
    let mut observations = Vec::with_capacity(3 * point_count);
    let mut point_offsets = Vec::with_capacity(point_count + 1);
    point_offsets.push(0);
    for (point_index, point) in points.iter().enumerate() {
        for camera in 0..3 {
            let camera_point = transform(&poses[camera], point);
            let intrinsics = cameras[camera];
            let inverse_z = camera_point[2].recip();
            let mut pixel = [
                intrinsics.fx * camera_point[0] * inverse_z
                    + intrinsics.skew * camera_point[1] * inverse_z
                    + intrinsics.cx,
                intrinsics.fy * camera_point[1] * inverse_z + intrinsics.cy,
            ];
            if perturb_observations {
                let sign = if (point_index + camera) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                pixel[0] += sign * 0.11;
                pixel[1] -= sign * 0.07;
            }
            observations.push(PixelObservation::new(camera, pixel));
        }
        point_offsets.push(observations.len());
    }
    (poses, points, observations, point_offsets, cameras)
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let length = norm(&vector);
    vector.map(|value| value / length)
}

fn norm_slice(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        error <= tolerance * scale,
        "expected {expected:.16e}, got {actual:.16e}, relative error {:.3e}",
        error / scale
    );
}

#[cfg(feature = "faer")]
#[test]
fn rotation_only_camera_zero_normal_terms_match_center_preserving_differences() {
    let rotation = manifold::exp_so3(&[0.21, -0.10, 0.07]);
    let center = [2.0, -0.4, 0.7];
    let translation = std::array::from_fn(|row| {
        -(0..3)
            .map(|column| rotation[row][column] * center[column])
            .sum::<f64>()
    });
    let poses = [
        RelativePose::new(rotation, translation),
        RelativePose::new(IDENTITY, [0.0; 3]),
    ];
    let point = [0.3, 0.2, 6.0];
    let measurement = [20.0, -10.0];
    let camera = PinholeCamera::new(430.0, 415.0, 13.0, -8.0);
    let observations = [PixelObservation::new(0, measurement)];
    let cameras = [camera, camera];
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &[0, 1],
        cameras: &cameras,
    });
    let masks = [
        crate::PoseVariables::FREE.with_translation(false),
        crate::PoseVariables::FIXED,
    ];
    let layout = VariableLayout::calibrated(&[0, 3], &[3, 0]).with_pose_masks(&masks);
    let mut values = [0.0; 9];
    let mut gradient = [0.0; 3];
    let mut full_gradient = [0.0; 3];
    let mut diagonal = [0.0; 3];
    let mut cross = [0.0; 9];
    let mut hessian = DenseHessian {
        values: &mut values,
        dimension: 3,
    };
    linearize_point_with_hessian::<true, false, _>(
        model,
        &poses,
        &point,
        0,
        GaugeState::new(ScaleGauge::Free, &poses, 1.0),
        Loss::L2,
        NoiseModel::Unit,
        layout,
        &mut cross,
        Some(ReducedBuffers {
            hessian: &mut hessian,
            gradient: &mut gradient,
            full_gradient: &mut full_gradient,
            diagonal: &mut diagonal,
        }),
    )
    .expect("finite one-observation linearization");

    // Independent pinhole formula and a rotation of X-C, with no projection-t update helper.
    let residual = |r: [[f64; 3]; 3], x: [f64; 3]| {
        let camera_point: [f64; 3] = std::array::from_fn(|row| {
            (0..3)
                .map(|column| r[row][column] * (x[column] - center[column]))
                .sum()
        });
        [
            430.0 * camera_point[0] / camera_point[2] + 13.0 - measurement[0],
            415.0 * camera_point[1] / camera_point[2] - 8.0 - measurement[1],
        ]
    };
    let h = 1.0e-6;
    let mut jc = [[0.0; 3]; 2];
    let mut jp = [[0.0; 3]; 2];
    for axis in 0..3 {
        let mut plus = [0.0; 3];
        plus[axis] = h;
        let mut minus = [0.0; 3];
        minus[axis] = -h;
        let plus_r = manifold::mat_mul(&manifold::exp_so3(&plus), &rotation);
        let minus_r = manifold::mat_mul(&manifold::exp_so3(&minus), &rotation);
        let plus_residual = residual(plus_r, point);
        let minus_residual = residual(minus_r, point);
        let mut plus_x = point;
        plus_x[axis] += h;
        let mut minus_x = point;
        minus_x[axis] -= h;
        let plus_point = residual(rotation, plus_x);
        let minus_point = residual(rotation, minus_x);
        for row in 0..2 {
            jc[row][axis] = (plus_residual[row] - minus_residual[row]) / (2.0 * h);
            jp[row][axis] = (plus_point[row] - minus_point[row]) / (2.0 * h);
        }
    }
    let r = residual(rotation, point);
    for axis in 0..3 {
        assert_close(
            gradient[axis],
            jc[0][axis] * r[0] + jc[1][axis] * r[1],
            2.0e-6,
        );
        for column in 0..=axis {
            assert_close(
                values[axis * 3 + column],
                jc[0][axis] * jc[0][column] + jc[1][axis] * jc[1][column],
                2.0e-6,
            );
        }
        for column in 0..3 {
            assert_close(
                cross[axis * 3 + column],
                jc[0][axis] * jp[0][column] + jc[1][axis] * jp[1][column],
                2.0e-6,
            );
        }
    }
}

/// The cost and gradient the optimizer actually uses, against central differences of an
/// independently evaluated cost.
///
/// The per-observation Jacobian tests above check one residual at a time. This one checks the
/// assembly built on them: robust-loss weights, the gauge chart, and camera priors, for the
/// reduced camera gradient (points held fixed) and for every point block's gradient.
fn check_assembled_gradient(
    model: ObservationModel<'_>,
    poses: [RelativePose; 3],
    points: &[[f64; 3]],
    gauge_kind: ScaleGauge,
    loss: Loss,
    priors: BundleAdjustmentPriors<'_>,
) {
    let scene_scale = 1.0;
    let gauge = GaugeState::new(gauge_kind, &poses, scene_scale);
    let mut workspace =
        DenseBundleAdjustmentWorkspace::try_new(3, points.len()).expect("small fixed workspace");
    let dimension = configure_layout(
        &mut workspace,
        poses.len(),
        gauge_kind,
        model.intrinsic_dimension(),
    );
    let total_cost = |poses: &[RelativePose], points: &[[f64; 3]]| -> f64 {
        let (observation_cost, _) = metrics(model, poses, points, loss, NoiseModel::Unit);
        observation_cost + prior_cost(model, poses, points, priors).expect("prior camera exists")
    };

    let assembly = assemble_problem(
        BaProblemView {
            model,
            poses: &poses,
            points,
            fixed: FixedVariableBlocks::NONE,
            priors,
        },
        LinearizationPolicy {
            gauge,
            loss,
            noise: NoiseModel::Unit,
            damping: 0.0,
            prior_sqrt_scale: 1.0,
        },
        dimension,
        &mut workspace,
    );
    assert!(assembly.valid);
    assert_close(assembly.cost, total_cost(&poses, points), 1e-12);
    let camera_gradient = workspace.full_gradient[..dimension].to_vec();
    let offsets = workspace.camera_offsets.clone();
    let pose_dimensions = workspace.camera_pose_dimensions.clone();

    let epsilon = 1e-6;
    let tolerance = 5e-6;
    let mut checked = 0;
    for camera in 1..poses.len() {
        for parameter in 0..pose_dimensions[camera] {
            let mut positive = [0.0; 6];
            positive[parameter] = epsilon;
            let mut negative = [0.0; 6];
            negative[parameter] = -epsilon;
            let mut plus = poses;
            let mut minus = poses;
            plus[camera] =
                retract_test_pose(poses[camera], gauge, &positive, pose_dimensions[camera]);
            minus[camera] =
                retract_test_pose(poses[camera], gauge, &negative, pose_dimensions[camera]);
            let numerical =
                (total_cost(&plus, points) - total_cost(&minus, points)) / (2.0 * epsilon);
            assert_close(
                camera_gradient[offsets[camera] + parameter],
                numerical,
                tolerance,
            );
            checked += 1;
        }
    }
    assert_eq!(checked, dimension);

    let mut camera_hessian = vec![0.0; dimension * dimension];
    let mut reduced_gradient = vec![0.0; dimension];
    let mut ignored_gradient = vec![0.0; dimension];
    let mut camera_diagonal = vec![0.0; dimension];
    let mut cross = vec![0.0; dimension * 3];
    for (point_index, point) in points.iter().enumerate() {
        cross.fill(0.0);
        let mut reduced_hessian = DenseHessian {
            values: &mut camera_hessian,
            dimension,
        };
        let local = linearize_point_with_hessian::<true, false, _>(
            model,
            &poses,
            point,
            point_index,
            gauge,
            loss,
            NoiseModel::Unit,
            VariableLayout {
                offsets: &workspace.camera_offsets,
                pose_dimensions: &workspace.camera_pose_dimensions,
                intrinsic_offsets: &workspace.intrinsic_offsets,
                intrinsic_dimensions: &workspace.intrinsic_dimensions,
                pose_masks: &[],
                #[cfg(feature = "faer")]
                pose_variables: CameraPoseVariables::RotationAndTranslation,
            },
            &mut cross,
            Some(ReducedBuffers {
                hessian: &mut reduced_hessian,
                gradient: &mut reduced_gradient,
                full_gradient: &mut ignored_gradient,
                diagonal: &mut camera_diagonal,
            }),
        )
        .expect("fixture is projectable");
        for coordinate in 0..3 {
            let mut plus = points.to_vec();
            let mut minus = points.to_vec();
            plus[point_index][coordinate] += scene_scale * epsilon;
            minus[point_index][coordinate] -= scene_scale * epsilon;
            let numerical =
                (total_cost(&poses, &plus) - total_cost(&poses, &minus)) / (2.0 * epsilon);
            assert_close(local.gradient[coordinate], numerical, tolerance);
        }
    }
}

fn check_pixel_gradient(loss: Loss, with_pose_prior: bool) {
    let (poses, mut points, observations, point_offsets, cameras) = pixel_fixture(12, true);
    for (index, point) in points.iter_mut().enumerate() {
        for (coordinate, value) in point.iter_mut().enumerate() {
            *value += 0.004 * ((3 * index + coordinate) as f64).sin();
        }
    }
    let model = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    });
    let prior = [PosePrior {
        camera: 2,
        mean: RelativePose::new(
            manifold::mat_mul(&manifold::exp_so3(&[0.02, -0.01, 0.03]), &poses[2].rotation),
            [0.27, 0.89, 0.14],
        ),
        sqrt_information: std::array::from_fn(|row| {
            std::array::from_fn(|column| if row == column { 3.0 + row as f64 } else { 0.0 })
        }),
    }];
    let priors = if with_pose_prior {
        BundleAdjustmentPriors {
            poses: &prior,
            points: &[],
            intrinsics: &[],
        }
    } else {
        BundleAdjustmentPriors::NONE
    };
    check_assembled_gradient(
        model,
        poses,
        &points,
        ScaleGauge::FixedBaseline { camera: 1 },
        loss,
        priors,
    );
}

fn check_angular_gradient(loss: Loss, gauge_kind: ScaleGauge) {
    let (poses, points, _, _, _) = pixel_fixture(12, false);
    let mut observations = Vec::new();
    let mut offsets = vec![0];
    for (index, point) in points.iter().enumerate() {
        for (camera, pose) in poses.iter().enumerate() {
            let mut bearing = transform(pose, point);
            bearing[0] += 0.004 * ((index + 2 * camera) as f64).sin();
            bearing[1] -= 0.003 * ((index + camera) as f64).cos();
            observations.push(AngularObservation::new(camera, normalize(bearing)));
        }
        offsets.push(observations.len());
    }
    let model = ObservationModel::calibrated(MultiViewMeasurements::Angular {
        observations: &observations,
        point_offsets: &offsets,
    });
    check_assembled_gradient(
        model,
        poses,
        &points,
        gauge_kind,
        loss,
        BundleAdjustmentPriors::NONE,
    );
}

#[test]
fn assembled_gradient_matches_finite_differences_of_the_l2_cost() {
    check_pixel_gradient(Loss::L2, false);
}

#[test]
fn assembled_gradient_matches_finite_differences_of_the_huber_cost() {
    check_pixel_gradient(Loss::Huber { delta: 0.2 }, false);
}

#[test]
fn assembled_gradient_matches_finite_differences_of_the_cauchy_cost() {
    check_pixel_gradient(Loss::Cauchy { scale: 0.3 }, false);
}

#[test]
fn assembled_gradient_includes_a_pose_prior_exactly() {
    check_pixel_gradient(Loss::Huber { delta: 0.2 }, true);
}

#[test]
fn assembled_angular_gradient_matches_finite_differences_for_both_gauges() {
    for gauge in [ScaleGauge::Free, ScaleGauge::FixedBaseline { camera: 1 }] {
        for loss in [
            Loss::L2,
            Loss::Huber { delta: 2e-4 },
            Loss::Cauchy { scale: 3e-4 },
        ] {
            check_angular_gradient(loss, gauge);
        }
    }
}
