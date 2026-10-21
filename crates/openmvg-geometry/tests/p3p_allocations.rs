// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation contract for the P3P hot paths.
//! Custom harness: libtest shares the global allocator, so the measurement
//! runs as a plain binary instead.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Point3, UnitVector3};
use openmvg_geometry::pose::p3p::{P3pMethod, solve};

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

fn main() {
    let world = [
        Point3::new(-1.2, -0.5, 4.0),
        Point3::new(1.1, -0.8, 5.5),
        Point3::new(0.2, 1.3, 6.5),
    ];
    let bearings = world.map(|point| UnitVector3::new_normalize(point.coords));

    // Ding/Fused exercise the cubic kernels; Ke/Kneip/Wu/Nakano exercise the
    // quartic kernel, so this also guards allocation-free root solving.
    // Warm up any test/runtime one-time initialization before measuring.
    let _ = solve(world, bearings, P3pMethod::Fused);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    for _ in 0..100 {
        for method in [
            P3pMethod::Fused,
            P3pMethod::Ding,
            P3pMethod::KeOrtho,
            P3pMethod::Kneip,
            P3pMethod::LambdaTwist,
            P3pMethod::Wu,
            P3pMethod::Nakano,
        ] {
            let _ = std::hint::black_box(solve(world, bearings, method));
        }
    }
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
