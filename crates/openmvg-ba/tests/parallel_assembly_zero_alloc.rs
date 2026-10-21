// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "faer", feature = "parallel"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use openmvg_ba::kernel::{
    PinholeCamera, RelativePose, multi_view::MultiViewMeasurements, multi_view::PixelObservation,
    multi_view::ScaleGauge, multi_view::sparse::SparseBundleAdjustmentReport,
    multi_view::sparse::SparseBundleAdjustmentWorkspace,
    multi_view::sparse::SparseCameraRankAssessment, multi_view::sparse::SparseWorkspaceOptions,
    multi_view::sparse::refine_ba_multi_view_sparse,
};
use openmvg_ba::numerics::refine::{NoiseModel, Options};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const GAUGE: ScaleGauge = ScaleGauge::FixedBaseline { camera: 1 };

fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}

/// Exact observations of every point by every camera, with point offsets.
fn observe(
    camera: PinholeCamera,
    poses: &[RelativePose],
    points: &[[f64; 3]],
) -> (Vec<PixelObservation>, Vec<usize>) {
    let mut observations = Vec::with_capacity(points.len() * poses.len());
    let mut point_offsets = Vec::with_capacity(points.len() + 1);
    point_offsets.push(0);
    for point in points {
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
        point_offsets.push(observations.len());
    }
    (observations, point_offsets)
}

/// Refine once to reserve lazy storage, then refine again from the same start
/// while counting every allocation. Returns the counted solve's results.
fn warm_solve(
    initial_poses: &[RelativePose],
    initial_points: &[[f64; 3]],
    measurements: MultiViewMeasurements<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> (
    SparseBundleAdjustmentReport,
    Vec<RelativePose>,
    Vec<[f64; 3]>,
    usize,
) {
    let mut poses = initial_poses.to_vec();
    let mut points = initial_points.to_vec();
    let mut refine = |poses: &mut [RelativePose], points: &mut [[f64; 3]]| {
        refine_ba_multi_view_sparse(
            poses,
            points,
            measurements,
            GAUGE,
            workspace,
            Options::l2(8),
            NoiseModel::Unit,
        )
    };
    let warmup = refine(&mut poses, &mut points).unwrap();
    assert!(warmup.final_cost.is_finite());

    poses.copy_from_slice(initial_poses);
    points.copy_from_slice(initial_points);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let report = refine(&mut poses, &mut points);
    COUNTING.store(false, Ordering::SeqCst);
    let allocations = ALLOCATIONS.load(Ordering::SeqCst);
    (report.unwrap(), poses, points, allocations)
}

/// A reduced camera system above 128 coordinates lends the direct assembly
/// pool to its dense rank certificate. The warm solve still allocates nothing,
/// and every pool size returns the same bits.
fn pooled_rank_certificate_is_allocation_free_and_pool_invariant() {
    const CAMERAS: usize = 32;
    let truth_poses: Vec<_> = (0..CAMERAS)
        .map(|camera| {
            let phase = 0.4 * camera as f64;
            let translation = if camera == 0 {
                [0.0; 3]
            } else {
                [0.6 * phase.cos(), 0.6 * phase.sin(), 0.01 * camera as f64]
            };
            RelativePose::new(IDENTITY, translation)
        })
        .collect();
    let camera = PinholeCamera::new(700.0, 690.0, 320.0, 240.0);
    let cameras = [camera; CAMERAS];
    let truth_points: Vec<_> = (0..48)
        .map(|index| {
            [
                -1.0 + 0.28 * (index % 8) as f64,
                -0.75 + 0.3 * (index / 8) as f64,
                4.0 + 0.15 * (index % 5) as f64,
            ]
        })
        .collect();
    let (observations, point_offsets) = observe(camera, &truth_poses, &truth_points);
    let measurements = MultiViewMeasurements::Pixel {
        observations: &observations,
        point_offsets: &point_offsets,
        cameras: &cameras,
    };
    let initial_poses: Vec<_> = truth_poses
        .iter()
        .enumerate()
        .map(|(camera, pose)| {
            let mut pose = *pose;
            if camera > 0 {
                pose.translation[camera % 3] += 0.03;
            }
            pose
        })
        .collect();
    let initial_points: Vec<_> = truth_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let offset = 0.004 * (index % 7) as f64;
            [point[0] + offset, point[1] - offset, point[2] + 0.012]
        })
        .collect();
    let solve = |threads| {
        let mut workspace = SparseBundleAdjustmentWorkspace::try_new_with_options(
            CAMERAS,
            measurements,
            GAUGE,
            SparseWorkspaceOptions::direct_parallel(threads),
        )
        .unwrap();
        let (report, poses, points, allocations) = warm_solve(
            &initial_poses,
            &initial_points,
            measurements,
            &mut workspace,
        );
        assert_eq!(allocations, 0);
        let bits: Vec<u64> = poses
            .iter()
            .flat_map(|pose| pose.rotation.iter().flatten().chain(&pose.translation))
            .chain(points.iter().flatten())
            .map(|value| value.to_bits())
            .collect();
        (report, bits)
    };

    let (one, one_bits) = solve(1);
    assert!(one.diagnostics.pose_dimension > 128);
    assert_eq!(
        one.diagnostics.pose_rank,
        SparseCameraRankAssessment::FullRank
    );
    let (three, three_bits) = solve(3);
    assert_eq!(three.diagnostics, one.diagnostics);
    assert_eq!(three.final_cost.to_bits(), one.final_cost.to_bits());
    assert_eq!(three_bits, one_bits);
}

fn main() {
    pooled_rank_certificate_is_allocation_free_and_pool_invariant();
}
