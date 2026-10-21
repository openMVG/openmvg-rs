// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion comparison of the retained rotation-averaging solvers.

use std::f64::consts::PI;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::Rotation3;
use openmvg_core::model::{RelativeRotation, ViewId};
use openmvg_geometry::averaging::{RotationConfig, average_rotations, average_rotations_dense};

fn ring_problem(camera_count: usize, neighbor_count: usize) -> Vec<RelativeRotation> {
    let rotations: Vec<_> = (0..camera_count)
        .map(|index| {
            Rotation3::from_euler_angles(
                0.1 * (index as f64 * 0.3).sin(),
                0.1 * (index as f64 * 0.2).cos(),
                2.0 * PI * index as f64 / camera_count as f64,
            )
        })
        .collect();

    let mut edges = Vec::with_capacity(camera_count * neighbor_count);
    for source in 0..camera_count {
        for offset in 1..=neighbor_count {
            let target = source + offset;
            if target >= camera_count {
                break;
            }
            edges.push(
                RelativeRotation::try_new(
                    ViewId::new(source),
                    ViewId::new(target),
                    rotations[target] * rotations[source].inverse(),
                    1.0,
                )
                .expect("generated rotations and weights are valid"),
            );
        }
    }
    edges
}

fn benchmark_rotation_averaging(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("rotation_averaging");
    let config = RotationConfig::default();

    for camera_count in [10, 50, 200] {
        let edges = ring_problem(camera_count, 5);

        group.bench_with_input(
            BenchmarkId::new("default_robust_sparse", camera_count),
            &edges,
            |b, e| b.iter(|| average_rotations(e, &config)),
        );

        group.bench_with_input(BenchmarkId::new("dense", camera_count), &edges, |b, e| {
            b.iter(|| average_rotations_dense(e, &config))
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_rotation_averaging);
criterion_main!(benches);
