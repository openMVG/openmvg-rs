// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion coverage for the linear-time view-graph topology audit.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use nalgebra::Vector3;
use openmvg_core::graph::{complete_connectivity, component_partition, maximum_spanning_tree};
use openmvg_geometry::averaging::filtering::{
    DirectionConstraint, ParallelRigidityConfig, audit_graph_topology, audit_parallel_rigidity,
};

fn chain(num_views: usize) -> Vec<(usize, usize)> {
    (1..num_views).map(|view| (view - 1, view)).collect()
}

fn sparse_ring(num_views: usize, forward_degree: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(num_views * forward_degree);
    for view_i in 0..num_views {
        for offset in 1..=forward_degree {
            let view_j = (view_i + offset) % num_views;
            edges.push((view_i.min(view_j), view_i.max(view_j)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges
}

fn large_rigid_directions(num_views: usize) -> Vec<DirectionConstraint> {
    let points: Vec<_> = (0..num_views)
        .map(|view| {
            let t = view as f64 + 1.0;
            Vector3::new((0.31 * t).sin(), (0.47 * t).cos(), 0.001 * t * t)
        })
        .collect();
    let mut edges = Vec::with_capacity(3 * num_views);
    for i in 0..4 {
        for j in (i + 1)..4 {
            edges.push(DirectionConstraint::new(i, j, points[j] - points[i]));
        }
    }
    for view in 4..num_views {
        for anchor in 0..3 {
            edges.push(DirectionConstraint::new(
                anchor,
                view,
                points[view] - points[anchor],
            ));
        }
    }
    edges
}

fn bench_graph_topology(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("graph_topology");
    for num_views in [1_000, 10_000, 100_000] {
        let edges = chain(num_views);
        group.throughput(Throughput::Elements((num_views + edges.len()) as u64));
        group.bench_with_input(
            BenchmarkId::new("chain", format!("{num_views}v_{}e", edges.len())),
            &edges,
            |bencher, edges| {
                bencher
                    .iter(|| audit_graph_topology(black_box(num_views), black_box(edges)).unwrap());
            },
        );
    }
    for num_views in [1_000, 10_000, 25_000] {
        let edges = sparse_ring(num_views, 4);
        group.throughput(Throughput::Elements((num_views + edges.len()) as u64));
        group.bench_with_input(
            BenchmarkId::new("sparse_ring", format!("{num_views}v_{}e", edges.len())),
            &edges,
            |bencher, edges| {
                bencher
                    .iter(|| audit_graph_topology(black_box(num_views), black_box(edges)).unwrap());
            },
        );
    }
    group.finish();

    let edges = sparse_ring(25_000, 4);
    let weights: Vec<_> = (0..edges.len())
        .map(|edge| 1.0 + (edge % 997) as f64)
        .collect();
    let mut preference_order: Vec<_> = (0..edges.len()).collect();
    preference_order.sort_unstable_by(|&left, &right| {
        weights[right]
            .total_cmp(&weights[left])
            .then_with(|| edges[left].cmp(&edges[right]))
            .then_with(|| left.cmp(&right))
    });
    let initially_kept: Vec<_> = (0..edges.len()).map(|edge| edge % 5 == 0).collect();
    let connectivity_candidates: Vec<_> = preference_order
        .iter()
        .copied()
        .filter(|&edge| !initially_kept[edge])
        .collect();
    let mut primitives = criterion.benchmark_group("graph_primitives");
    primitives.throughput(Throughput::Elements((25_000 + edges.len()) as u64));
    primitives.bench_function("component_partition/25000v_100000e", |bencher| {
        bencher.iter(|| component_partition(black_box(25_000), black_box(&edges)).unwrap());
    });
    primitives.bench_function("complete_connectivity/25000v_100000e", |bencher| {
        bencher.iter(|| {
            complete_connectivity(
                black_box(25_000),
                black_box(&edges),
                black_box(&initially_kept),
                black_box(&connectivity_candidates),
            )
            .unwrap()
        });
    });
    primitives.bench_function("maximum_spanning_tree/25000v_100000e", |bencher| {
        bencher.iter(|| {
            maximum_spanning_tree(
                black_box(25_000),
                black_box(&edges),
                black_box(&weights),
                black_box(0),
            )
            .unwrap()
        });
    });
    primitives.finish();

    let mut rigidity = criterion.benchmark_group("parallel_rigidity");
    for num_views in [520, 1_000] {
        let edges = large_rigid_directions(num_views);
        rigidity.throughput(Throughput::Elements(edges.len() as u64));
        rigidity.bench_with_input(
            BenchmarkId::new("auto_sparse_structural", num_views),
            &edges,
            |bencher, edges| {
                bencher.iter(|| {
                    audit_parallel_rigidity(
                        black_box(num_views),
                        black_box(edges),
                        ParallelRigidityConfig::default(),
                    )
                    .unwrap()
                });
            },
        );
    }
    rigidity.finish();
}

criterion_group!(benches, bench_graph_topology);
criterion_main!(benches);
