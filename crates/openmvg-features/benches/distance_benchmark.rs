// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark for L2 distance computation methods
//! Run with: `pixi run cargo bench -p openmvg-features --bench distance_benchmark`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use openmvg_features::matching::{
    BruteForceMatcher, DistanceMetric, L2Distance, SquaredL2Distance, SquaredL2DistanceU8, l2_f32,
    squared_l2_f32, squared_l2_u8,
};

fn create_random_descriptors(n_descriptors: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..n_descriptors)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 255.0).collect())
        .collect()
}

fn create_random_u8_descriptors(n_descriptors: usize, dim: usize, seed: u64) -> Vec<Vec<u8>> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..n_descriptors)
        .map(|_| (0..dim).map(|_| rng.r#gen()).collect())
        .collect()
}

fn bench_u8_distance_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2_u8_distance_methods");
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA128_0001);
    let a: Vec<u8> = (0..128).map(|_| rng.r#gen()).collect();
    let b: Vec<u8> = (0..128).map(|_| rng.r#gen()).collect();

    group.bench_function("auto_dispatch", |bench| {
        bench.iter(|| squared_l2_u8(black_box(&a), black_box(&b)))
    });
    group.finish();
}

fn bench_distance_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2_distance_methods");

    // SIFT descriptor size = 128
    let dim = 128;

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF128_0001);
    let a: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 255.0).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 255.0).collect();

    // Primary API.
    group.bench_function("auto_dispatch", |bench| {
        bench.iter(|| squared_l2_f32(black_box(&a), black_box(&b)))
    });

    // L2 with sqrt
    group.bench_function("l2", |bench| {
        bench.iter(|| l2_f32(black_box(&a), black_box(&b)))
    });

    // Test with the actual DistanceMetric implementations
    let l2 = L2Distance::new();
    let squared_l2 = SquaredL2Distance::new();

    group.bench_function("L2Distance_trait", |bench| {
        bench.iter(|| l2.distance(black_box(&a), black_box(&b)))
    });

    group.bench_function("SquaredL2Distance_trait", |bench| {
        bench.iter(|| squared_l2.distance(black_box(&a), black_box(&b)))
    });

    group.finish();
}

fn bench_brute_force_matching_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("BruteForceMatcher_u8_128d");
    let queries = create_random_u8_descriptors(100, 1 << 7, 0x008D_157A);
    let database = create_random_u8_descriptors(1_000, 1 << 7, 0x008D_157B);
    let queries: Vec<&[u8]> = queries.iter().map(Vec::as_slice).collect();
    let database: Vec<&[u8]> = database.iter().map(Vec::as_slice).collect();

    group.bench_function("runtime_dispatched", |bench| {
        bench.iter(|| {
            BruteForceMatcher::match_direct_slices(
                SquaredL2DistanceU8::new(),
                black_box(&queries),
                black_box(&database),
                2,
            )
        })
    });
    group.finish();
}

fn bench_brute_force_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("BruteForceMatcher");

    // Test with different sizes
    for (n_queries, n_database) in [(100, 1000), (500, 2000)] {
        let dim = 128; // SIFT descriptor dimension

        let queries = create_random_descriptors(n_queries, dim, 0xF32D_157A);
        let database = create_random_descriptors(n_database, dim, 0xF32D_157B);
        let queries: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
        let database: Vec<&[f32]> = database.iter().map(Vec::as_slice).collect();

        let label = format!("{}q_x_{}db", n_queries, n_database);

        // Test with L2Distance
        group.bench_function(format!("{}_L2", label), |bench| {
            bench.iter(|| {
                BruteForceMatcher::match_direct_slices(
                    L2Distance::new(),
                    black_box(&queries),
                    black_box(&database),
                    2,
                )
            })
        });

        // Test with SquaredL2Distance (should be faster - no sqrt)
        group.bench_function(format!("{}_SquaredL2", label), |bench| {
            bench.iter(|| {
                BruteForceMatcher::match_direct_slices(
                    SquaredL2Distance::new(),
                    black_box(&queries),
                    black_box(&database),
                    2,
                )
            })
        });
    }

    group.finish();
}

fn bench_large_scale_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("LargeScale_Throughput");

    // Larger workload; this measures execution time, not peak memory.
    let n_queries = 1000;
    let n_database = 5000;
    let dim = 128;

    let queries = create_random_descriptors(n_queries, dim, 0xF32D_157A);
    let database = create_random_descriptors(n_database, dim, 0xF32D_157B);
    let queries: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
    let database: Vec<&[f32]> = database.iter().map(Vec::as_slice).collect();

    group.bench_function("large_scale_1000x5000", |bench| {
        bench.iter(|| {
            BruteForceMatcher::match_direct_slices(
                SquaredL2Distance::new(),
                black_box(&queries),
                black_box(&database),
                2,
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_methods,
    bench_u8_distance_methods,
    bench_brute_force_matching,
    bench_brute_force_matching_u8,
    bench_large_scale_throughput
);
criterion_main!(benches);
