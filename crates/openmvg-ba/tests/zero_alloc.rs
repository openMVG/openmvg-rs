// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::BundleAdjustmentPriors,
    multi_view::CalibratedBundleAdjustmentProblem, multi_view::FixedVariableBlocks,
    multi_view::MultiViewMeasurements, multi_view::MultiViewObservationResidual,
    multi_view::MultiViewTrackClassification, multi_view::MultiViewTrackDiagnostics,
    multi_view::MultiViewTriangulationReport, multi_view::PixelObservation, multi_view::PointPrior,
    multi_view::PosePrior, multi_view::ScaleGauge, multi_view::TranslationComponent,
    multi_view::dense::DenseBundleAdjustmentWorkspace, multi_view::self_calibration::RadialCamera,
    multi_view::self_calibration::SelfCalibratingMeasurements,
    multi_view::self_calibration::SelfCalibrationWorkspace, two_view::TwoViewMeasurements,
};
#[cfg(all(feature = "faer", feature = "parallel"))]
use openmvg_ba::kernel::{
    multi_view::sparse::DirectSchurAssembly, multi_view::sparse::SparseWorkspaceOptions,
};
#[cfg(feature = "faer")]
use openmvg_ba::kernel::{
    multi_view::sparse::IterativeSchurOptions, multi_view::sparse::RetainedCrossPrecision,
    multi_view::sparse::RetainedSchurMatvec, multi_view::sparse::SparseBundleAdjustmentWorkspace,
    multi_view::sparse::SparseCameraSolver, multi_view::sparse::SparseSelfCalibrationWorkspace,
};
use openmvg_ba::numerics::manifold;
use openmvg_ba::numerics::refine::{self, NoiseModel, Options};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    #[cfg(feature = "parallel")]
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global();
    let ba_truth = RelativePose::new(manifold::exp_so3(&[0.03, -0.06, 0.02]), [0.8, 0.1, 0.1]);
    let ba_initial = RelativePose::new(
        manifold::mat_mul(
            &manifold::exp_so3(&[0.01, -0.008, 0.005]),
            &ba_truth.rotation,
        ),
        [0.79, 0.12, 0.09],
    );
    let ba_truth_points: [[f64; 3]; 8] = std::array::from_fn(|index| {
        [
            (index % 4) as f64 * 0.2 - 0.3,
            (index / 4) as f64 * 0.25 - 0.125,
            3.0 + 0.1 * index as f64,
        ]
    });
    let normalize = |point: [f64; 3]| {
        let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
        point.map(|value| value / norm)
    };
    let ba_bearings1 = ba_truth_points.map(normalize);
    let ba_bearings2 = ba_truth_points.map(|point| {
        normalize([
            ba_truth.rotation[0][0] * point[0]
                + ba_truth.rotation[0][1] * point[1]
                + ba_truth.rotation[0][2] * point[2]
                + ba_truth.translation[0],
            ba_truth.rotation[1][0] * point[0]
                + ba_truth.rotation[1][1] * point[1]
                + ba_truth.rotation[1][2] * point[2]
                + ba_truth.translation[1],
            ba_truth.rotation[2][0] * point[0]
                + ba_truth.rotation[2][1] * point[1]
                + ba_truth.rotation[2][2] * point[2]
                + ba_truth.translation[2],
        ])
    });
    let mut ba_points = ba_truth_points.map(|point| [point[0] + 0.01, point[1] - 0.01, point[2]]);
    let ba_camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    let project = |point: [f64; 3]| {
        [
            ba_camera.fx * point[0] / point[2] + ba_camera.cx,
            ba_camera.fy * point[1] / point[2] + ba_camera.cy,
        ]
    };
    let ba_pixels1 = ba_truth_points.map(project);
    let ba_pixels2 = ba_truth_points.map(|point| {
        project([
            ba_truth.rotation[0][0] * point[0]
                + ba_truth.rotation[0][1] * point[1]
                + ba_truth.rotation[0][2] * point[2]
                + ba_truth.translation[0],
            ba_truth.rotation[1][0] * point[0]
                + ba_truth.rotation[1][1] * point[1]
                + ba_truth.rotation[1][2] * point[2]
                + ba_truth.translation[1],
            ba_truth.rotation[2][0] * point[0]
                + ba_truth.rotation[2][1] * point[1]
                + ba_truth.rotation[2][2] * point[2]
                + ba_truth.translation[2],
        ])
    });
    let mut ba_pixel_points = ba_points;
    let identity_pose = RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let third_truth = RelativePose::new(manifold::exp_so3(&[-0.02, 0.04, 0.01]), [-0.6, 0.1, 0.15]);
    let mut multi_poses = [
        identity_pose,
        ba_initial,
        RelativePose::new(
            manifold::mat_mul(
                &manifold::exp_so3(&[0.008, -0.006, 0.004]),
                &third_truth.rotation,
            ),
            [-0.58, 0.09, 0.17],
        ),
    ];
    let mut multi_points = ba_points;
    let multi_truth_poses = [identity_pose, ba_truth, third_truth];
    let multi_observations: [PixelObservation; 24] = std::array::from_fn(|observation| {
        let point = observation / 3;
        let camera = observation % 3;
        let value = if camera == 0 {
            ba_truth_points[point]
        } else {
            let pose = multi_truth_poses[camera];
            [
                pose.rotation[0][0] * ba_truth_points[point][0]
                    + pose.rotation[0][1] * ba_truth_points[point][1]
                    + pose.rotation[0][2] * ba_truth_points[point][2]
                    + pose.translation[0],
                pose.rotation[1][0] * ba_truth_points[point][0]
                    + pose.rotation[1][1] * ba_truth_points[point][1]
                    + pose.rotation[1][2] * ba_truth_points[point][2]
                    + pose.translation[1],
                pose.rotation[2][0] * ba_truth_points[point][0]
                    + pose.rotation[2][1] * ba_truth_points[point][1]
                    + pose.rotation[2][2] * ba_truth_points[point][2]
                    + pose.translation[2],
            ]
        };
        PixelObservation::new(camera, project(value))
    });
    let multi_offsets: [usize; 9] = std::array::from_fn(|point| point * 3);
    let multi_cameras = [ba_camera; 3];
    let mut multi_workspace = DenseBundleAdjustmentWorkspace::try_new(3, 8).unwrap();
    #[cfg(feature = "faer")]
    let mut sparse_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut sparse_points = multi_points;
    #[cfg(feature = "faer")]
    let mut sparse_workspace = SparseBundleAdjustmentWorkspace::try_new(
        3,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
    )
    .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_poses = multi_poses;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_points = multi_points;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_workspace = SparseBundleAdjustmentWorkspace::try_new_with_options(
        3,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        SparseWorkspaceOptions::direct_parallel(2),
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let mut iterative_sparse_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut iterative_sparse_points = multi_points;
    #[cfg(feature = "faer")]
    let mut iterative_sparse_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_camera_solver(
            3,
            MultiViewMeasurements::Pixel {
                observations: &multi_observations,
                point_offsets: &multi_offsets,
                cameras: &multi_cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            SparseCameraSolver::IterativeSchur,
        )
        .unwrap();
    #[cfg(feature = "faer")]
    iterative_sparse_workspace
        .try_reserve_direct_storage()
        .unwrap();
    #[cfg(feature = "faer")]
    let mut retained_only_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut retained_only_points = multi_points;
    #[cfg(feature = "faer")]
    let mut retained_only_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            3,
            MultiViewMeasurements::Pixel {
                observations: &multi_observations,
                point_offsets: &multi_offsets,
                cameras: &multi_cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            IterativeSchurOptions::new().with_exact_camera_rank_diagnostics(false),
        )
        .unwrap();
    #[cfg(feature = "faer")]
    let mut f32_retained_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut f32_retained_points = multi_points;
    #[cfg(feature = "faer")]
    let mut f32_retained_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            3,
            MultiViewMeasurements::Pixel {
                observations: &multi_observations,
                point_offsets: &multi_offsets,
                cameras: &multi_cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            IterativeSchurOptions::new()
                .with_cross_precision(RetainedCrossPrecision::F32)
                .with_exact_camera_rank_diagnostics(false),
        )
        .unwrap();
    #[cfg(feature = "faer")]
    let mut recomputed_schur_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut recomputed_schur_points = multi_points;
    #[cfg(feature = "faer")]
    let mut recomputed_schur_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            3,
            MultiViewMeasurements::Pixel {
                observations: &multi_observations,
                point_offsets: &multi_offsets,
                cameras: &multi_cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::Recomputed)
                .with_exact_camera_rank_diagnostics(false),
        )
        .unwrap();
    #[cfg(feature = "faer")]
    recomputed_schur_workspace
        .try_reserve_direct_storage()
        .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_recomputed_poses = multi_poses;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_recomputed_points = multi_points;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_recomputed_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            3,
            MultiViewMeasurements::Pixel {
                observations: &multi_observations,
                point_offsets: &multi_offsets,
                cameras: &multi_cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::RecomputedParallel { threads: 2 })
                .with_exact_camera_rank_diagnostics(false),
        )
        .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    parallel_recomputed_workspace
        .try_reserve_direct_storage()
        .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut point_major_poses = multi_poses;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut point_major_points = multi_points;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut point_major_workspace =
        SparseBundleAdjustmentWorkspace::try_new_with_iterative_schur_options(
            3,
            MultiViewMeasurements::Pixel {
                observations: &multi_observations,
                point_offsets: &multi_offsets,
                cameras: &multi_cameras,
            },
            ScaleGauge::FixedBaseline { camera: 1 },
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::PointMajorParallel { threads: 2 })
                .with_exact_camera_rank_diagnostics(false),
        )
        .unwrap();
    let self_truth_cameras = [
        RadialCamera::new(500.0, 0.02, -0.005),
        RadialCamera::new(520.0, -0.015, 0.004),
        RadialCamera::new(480.0, 0.01, -0.003),
    ];
    let self_observations: [PixelObservation; 24] = std::array::from_fn(|observation| {
        let point = observation / 3;
        let camera = observation % 3;
        let value = if camera == 0 {
            ba_truth_points[point]
        } else {
            let pose = multi_truth_poses[camera];
            [
                pose.rotation[0][0] * ba_truth_points[point][0]
                    + pose.rotation[0][1] * ba_truth_points[point][1]
                    + pose.rotation[0][2] * ba_truth_points[point][2]
                    + pose.translation[0],
                pose.rotation[1][0] * ba_truth_points[point][0]
                    + pose.rotation[1][1] * ba_truth_points[point][1]
                    + pose.rotation[1][2] * ba_truth_points[point][2]
                    + pose.translation[1],
                pose.rotation[2][0] * ba_truth_points[point][0]
                    + pose.rotation[2][1] * ba_truth_points[point][1]
                    + pose.rotation[2][2] * ba_truth_points[point][2]
                    + pose.translation[2],
            ]
        };
        let normalized = [value[0] / value[2], value[1] / value[2]];
        let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
        let camera_model = self_truth_cameras[camera];
        let distortion = 1.0
            + camera_model.k1 * radius_squared
            + camera_model.k2 * radius_squared * radius_squared;
        PixelObservation::new(
            camera,
            normalized.map(|coordinate| camera_model.focal * distortion * coordinate),
        )
    });
    let mut self_poses = multi_poses;
    let mut self_points = multi_points;
    let mut self_cameras = [
        RadialCamera::new(510.0, 0.018, -0.004),
        RadialCamera::new(510.0, -0.012, 0.003),
        RadialCamera::new(490.0, 0.012, -0.004),
    ];
    let mut self_workspace = SelfCalibrationWorkspace::try_new(3, 8).unwrap();
    #[cfg(feature = "faer")]
    let mut sparse_self_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut sparse_self_points = multi_points;
    #[cfg(feature = "faer")]
    let mut sparse_self_cameras = self_cameras;
    #[cfg(feature = "faer")]
    let mut sparse_self_workspace = SparseSelfCalibrationWorkspace::try_new(
        3,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        FixedVariableBlocks::NONE,
    )
    .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_self_poses = multi_poses;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_self_points = multi_points;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_self_cameras = self_cameras;
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let mut parallel_direct_self_workspace =
        SparseSelfCalibrationWorkspace::try_new_with_direct_assembly(
            3,
            SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
            ScaleGauge::FixedBaseline { camera: 1 },
            FixedVariableBlocks::NONE,
            DirectSchurAssembly::Parallel { threads: 2 },
        )
        .unwrap();
    #[cfg(feature = "faer")]
    let mut retained_self_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut retained_self_points = multi_points;
    #[cfg(feature = "faer")]
    let mut retained_self_cameras = self_cameras;
    #[cfg(feature = "faer")]
    let mut retained_self_workspace =
        SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
            3,
            SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
            ScaleGauge::FixedBaseline { camera: 1 },
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new().with_matvec(if cfg!(feature = "parallel") {
                RetainedSchurMatvec::PointMajorParallel { threads: 2 }
            } else {
                RetainedSchurMatvec::Fused
            }),
        )
        .unwrap();
    #[cfg(feature = "faer")]
    let mut recomputed_self_poses = multi_poses;
    #[cfg(feature = "faer")]
    let mut recomputed_self_points = multi_points;
    #[cfg(feature = "faer")]
    let mut recomputed_self_cameras = self_cameras;
    #[cfg(feature = "faer")]
    let mut recomputed_self_workspace =
        SparseSelfCalibrationWorkspace::try_new_with_iterative_schur_options(
            3,
            SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
            ScaleGauge::FixedBaseline { camera: 1 },
            FixedVariableBlocks::NONE,
            IterativeSchurOptions::new().with_matvec(RetainedSchurMatvec::Recomputed),
        )
        .unwrap();
    let mut grouped_poses = multi_poses;
    let mut grouped_points = multi_points;
    let mut grouped_cameras = self_cameras;
    let grouped_assignment = [0, 1, 2];
    let mut grouped_workspace = SelfCalibrationWorkspace::try_new(3, 8).unwrap();
    let fixed_poses = [false, false, true];
    let fixed_points = [true, false, false, false, false, false, false, false];
    let identity6 = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { 1.0 } else { 0.0 })
    });
    let identity3 = std::array::from_fn(|row| {
        std::array::from_fn(|column| if row == column { 1.0 } else { 0.0 })
    });
    let pose_priors = [PosePrior {
        camera: 1,
        mean: ba_truth,
        sqrt_information: identity6,
    }];
    let point_priors = [PointPrior {
        point: 1,
        mean: ba_truth_points[1],
        sqrt_information: identity3,
    }];
    let multi_problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
    );
    let factor_problem = CalibratedBundleAdjustmentProblem::new(
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedTranslationComponent {
            camera: 1,
            component: TranslationComponent::X,
        },
    );
    let mut observation_residuals = [MultiViewObservationResidual::default(); 24];
    let mut track_diagnostics = [MultiViewTrackDiagnostics::default(); 8];
    let mut observation_inliers = [false; 24];
    let mut track_classifications = [MultiViewTrackClassification::Stable; 8];
    let mut triangulation_reports = [MultiViewTriangulationReport::default(); 8];
    let track_selection = [true, false, true, false, true, false, true, false];
    ALLOCATIONS.store(0, Ordering::SeqCst);
    let _ = multi_problem
        .evaluate(
            &multi_poses,
            &multi_points,
            refine::Loss::L2,
            &mut observation_residuals,
            &mut track_diagnostics,
        )
        .unwrap();
    multi_problem
        .triangulate_tracks(
            &multi_poses,
            &[],
            &mut triangulation_reports,
            Options::l2(2),
        )
        .unwrap();
    let _ = multi_problem
        .evaluate_track_lifecycle(
            &multi_poses,
            &multi_points,
            refine::Loss::L2,
            3.0,
            &mut observation_residuals,
            &mut track_diagnostics,
            &mut observation_inliers,
            &mut track_classifications,
        )
        .unwrap();
    multi_problem
        .triangulate_selected_tracks(
            &multi_poses,
            &observation_inliers,
            &track_selection,
            &mut triangulation_reports,
            Options::l2(2),
        )
        .unwrap();
    let _ = openmvg_ba::kernel::multi_view::dense::refine_ba_multi_view_dense_with_factors(
        &mut multi_poses,
        &mut multi_points,
        factor_problem.measurements(),
        factor_problem.gauge(),
        FixedVariableBlocks {
            camera_poses: &fixed_poses,
            points: &fixed_points,
            intrinsics: &[],
            intrinsic_dimensions: &[],
        },
        BundleAdjustmentPriors {
            poses: &pose_priors,
            points: &point_priors,
            intrinsics: &[],
        },
        &mut multi_workspace,
        Options::l2(2),
        factor_problem.noise(),
    )
    .unwrap();
    let _ = openmvg_ba::kernel::two_view::refine_ba_2view(
        &ba_initial,
        &mut ba_points,
        TwoViewMeasurements::Angular {
            bearings1: &ba_bearings1,
            bearings2: &ba_bearings2,
        },
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    let _ = openmvg_ba::kernel::two_view::refine_ba_2view(
        &ba_initial,
        &mut ba_pixel_points,
        TwoViewMeasurements::Pixel {
            observations1: &ba_pixels1,
            observations2: &ba_pixels2,
            cameras: [ba_camera; 2],
        },
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    let _ = openmvg_ba::kernel::multi_view::dense::refine_ba_multi_view_dense(
        &mut multi_poses,
        &mut multi_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut multi_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut sparse_poses,
        &mut sparse_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut sparse_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut parallel_direct_poses,
        &mut parallel_direct_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut parallel_direct_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut point_major_poses,
        &mut point_major_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut point_major_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut iterative_sparse_poses,
        &mut iterative_sparse_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut iterative_sparse_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut retained_only_poses,
        &mut retained_only_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut retained_only_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut f32_retained_poses,
        &mut f32_retained_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut f32_retained_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut recomputed_schur_poses,
        &mut recomputed_schur_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut recomputed_schur_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_sparse(
        &mut parallel_recomputed_poses,
        &mut parallel_recomputed_points,
        MultiViewMeasurements::Pixel {
            observations: &multi_observations,
            point_offsets: &multi_offsets,
            cameras: &multi_cameras,
        },
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut parallel_recomputed_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    let _ = openmvg_ba::kernel::multi_view::self_calibration::refine_ba_multi_view_self_calibrating_dense(
        &mut self_poses,
        &mut self_points,
        &mut self_cameras,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut self_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_self_calibrating_sparse(
        &mut sparse_self_poses,
        &mut sparse_self_points,
        &mut sparse_self_cameras,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut sparse_self_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(all(feature = "faer", feature = "parallel"))]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_self_calibrating_sparse(
        &mut parallel_direct_self_poses,
        &mut parallel_direct_self_points,
        &mut parallel_direct_self_cameras,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut parallel_direct_self_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_self_calibrating_sparse(
        &mut retained_self_poses,
        &mut retained_self_points,
        &mut retained_self_cameras,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut retained_self_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    #[cfg(feature = "faer")]
    let _ = openmvg_ba::kernel::multi_view::sparse::refine_ba_multi_view_self_calibrating_sparse(
        &mut recomputed_self_poses,
        &mut recomputed_self_points,
        &mut recomputed_self_cameras,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut recomputed_self_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();
    let _ = openmvg_ba::kernel::multi_view::self_calibration::refine_ba_multi_view_self_calibrating_grouped_dense(
        &mut grouped_poses,
        &mut grouped_points,
        &mut grouped_cameras,
        &grouped_assignment,
        SelfCalibratingMeasurements::new(&self_observations, &multi_offsets),
        ScaleGauge::FixedBaseline { camera: 1 },
        &mut grouped_workspace,
        Options::l2(2),
        NoiseModel::Unit,
    )
    .unwrap();

    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
