// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::Vector3;
use openmvg_geometry::averaging::global_positioning::test_support::{
    BataOptions, GlobalPositioningPlan, GlobalPositioningProblem, PositioningObservation,
};

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
    let cameras = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.1, 0.0),
        Vector3::new(0.2, 1.0, 0.1),
    ];
    let points = [
        Vector3::new(0.2, 0.1, 3.0),
        Vector3::new(1.1, 0.8, 4.0),
        Vector3::new(-0.4, 1.2, 3.5),
        Vector3::new(0.7, -0.4, 2.8),
    ];
    let observations: Vec<_> = points
        .iter()
        .enumerate()
        .flat_map(|(point, position)| {
            cameras.iter().enumerate().map(move |(camera, center)| {
                PositioningObservation::bearing(
                    camera,
                    point,
                    (*position - *center).normalize(),
                    1.0,
                )
            })
        })
        .collect();
    let plan = GlobalPositioningPlan::try_new(
        GlobalPositioningProblem {
            camera_count: cameras.len(),
            point_count: points.len(),
            observations: &observations,
        },
        BataOptions::default(),
    )
    .expect("valid positioning problem");
    let mut workspace = plan.try_new_workspace().expect("valid workspace");
    let mut camera_output = [Vector3::zeros(); 3];
    let mut point_output = [Vector3::zeros(); 4];

    plan.solve_into(&mut camera_output, &mut point_output, &mut workspace)
        .expect("warm solve");
    ALLOCATIONS.store(0, Ordering::SeqCst);
    plan.solve_into(&mut camera_output, &mut point_output, &mut workspace)
        .expect("allocation-free solve");
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
