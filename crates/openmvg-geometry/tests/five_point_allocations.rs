// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation contract for the fixed-storage five-point hot path.
//! Custom harness: libtest shares the global allocator, so the measurement
//! runs as a plain binary instead.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Rotation3, Vector3};
use openmvg_geometry::correspondence::BearingCorrespondence;
use openmvg_geometry::essential::test_support::solve_five_point_minimal;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: Delegates the unchanged layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout came from the delegated allocation.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: Delegates the unchanged pointer, layout and size to the system allocator.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn sample() -> [BearingCorrespondence; 5] {
    let rotation = Rotation3::from_euler_angles(0.18, -0.23, 0.11);
    let translation = Vector3::new(0.35, -0.14, 0.27);
    [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.4, 0.2, 0.89).normalize(),
        Vector3::new(-0.6, 0.3, -0.74).normalize(),
        Vector3::new(0.2, -0.8, -0.56).normalize(),
    ]
    .map(|bearing1| {
        let point = bearing1 * 3.5;
        BearingCorrespondence::try_new(bearing1, rotation * point + translation).unwrap()
    })
}

fn main() {
    let sample = sample();
    assert!(!solve_five_point_minimal(&sample).is_empty());

    ALLOCATIONS.store(0, Ordering::SeqCst);
    for _ in 0..100 {
        let _ = std::hint::black_box(solve_five_point_minimal(std::hint::black_box(&sample)));
    }
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0, "five-point backend");
}
