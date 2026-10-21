// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation regression for the dominant two-view robust triangulation path.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nalgebra::{Point2, Point3, Rotation3};
use openmvg_core::camera::{Camera, PinholeCamera};
use openmvg_core::model::{ViewId, WorldToCameraPose};
use openmvg_geometry::triangulation::{ResolvedTrackObservation, TrackTriangulator};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: allocation is delegated unchanged to the system allocator; the
// counter records successful allocation attempts only during the measured scope.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let result = operation();
    COUNTING.store(false, Ordering::SeqCst);
    (result, ALLOCATIONS.load(Ordering::SeqCst))
}

fn robust_two_view_path_has_two_bounded_allocations() {
    let estimator = TrackTriangulator::robust();
    let camera = Camera::Pinhole(PinholeCamera::try_with_focal(500.0, 320.0, 240.0).unwrap());
    let poses = [
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(-0.5, 0.0, 0.0))
            .unwrap(),
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(0.5, 0.0, 0.0))
            .unwrap(),
    ];
    let observations = [
        ResolvedTrackObservation::new(
            ViewId::new(0),
            Point2::new(370.0, 240.0),
            &poses[0],
            &camera,
        ),
        ResolvedTrackObservation::new(
            ViewId::new(1),
            Point2::new(270.0, 240.0),
            &poses[1],
            &camera,
        ),
    ];

    black_box(estimator.triangulate_resolved(&observations).unwrap());
    let (result, allocations) = count_allocations(|| estimator.triangulate_resolved(&observations));
    black_box(result.unwrap());

    assert_eq!(
        allocations, 2,
        "two-view estimation should allocate only prepared observations and returned support"
    );
}

// A plain binary (`harness = false` in Cargo.toml): the allocator counts every thread, and the
// libtest harness allocates on its own threads while a test runs.
fn main() {
    robust_two_view_path_has_two_bounded_allocations();
}
