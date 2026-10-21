// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact and HNSW matchers on RootSIFT-like `u8` descriptors.
//!
//! Each database descriptor is a RootSIFT-like vector (square root of an
//! L1-normalized, sparse, non-negative histogram) quantized with
//! [`U8Descriptor::try_from_finalized`]. Queries are noisy copies of database
//! entries, so every query has a well-defined nearest neighbour. Timed loops
//! measure index build plus a two-nearest-neighbour batch query. Recall@1
//! against the exact matcher is computed once, outside the timed loops, and
//! printed next to the timings. All data comes from one fixed seed.
//!
//! Run with:
//!   pixi run cargo bench -p openmvg-features --bench matcher_benchmark

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use openmvg_features::extraction::{DescriptorStorage, U8Descriptor};
use openmvg_features::matching::{BruteForceMatcher, HnswConfig, HnswMatcher};
use openmvg_features::matching::{Match, SquaredL2DistanceU8};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Seed for every generated descriptor and for the approximate indexes.
const SEED: u64 = 42;
const DIM: usize = 128;
/// SIFT's float-to-integer descriptor scale: unit-norm values map onto `[0, 255]`.
const FINALIZED_SCALE: f32 = 512.0;
/// `(queries, database)` sizes: one small and one image-pair-sized workload.
const SIZES: [(usize, usize); 2] = [(1_000, 2_000), (4_000, 8_000)];
/// Neighbours requested per query, as the ratio test needs.
const K: usize = 2;

type Descriptors = Vec<Vec<u8>>;

/// A sparse non-negative histogram mapped through RootSIFT: L1-normalize, then
/// take the square root, so the result has unit L2 norm.
fn rootsift_like(rng: &mut StdRng) -> [f32; DIM] {
    let mut values = [0.0f32; DIM];
    for value in &mut values {
        let sample: f32 = rng.r#gen();
        *value = sample * sample * sample;
    }
    rootsift(&mut values);
    values
}

fn rootsift(values: &mut [f32; DIM]) {
    let l1: f32 = values.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    for value in values.iter_mut() {
        *value = (*value / l1).sqrt();
    }
}

fn quantize(values: &[f32; DIM]) -> Vec<u8> {
    let scaled = values.map(|value| value * FINALIZED_SCALE);
    U8Descriptor::try_from_finalized(&scaled)
        .expect("generated RootSIFT benchmark bins are finite")
        .as_slice()
        .to_vec()
}

/// Database descriptors and queries that perturb a random database entry.
fn workload(queries: usize, database: usize) -> (Descriptors, Descriptors) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let base: Vec<[f32; DIM]> = (0..database).map(|_| rootsift_like(&mut rng)).collect();
    let query_set = (0..queries)
        .map(|_| {
            let source = &base[rng.gen_range(0..database)];
            let mut squared = source.map(|value| value * value);
            for value in &mut squared {
                *value = (*value * rng.gen_range(0.6..1.4f32)).max(0.0);
            }
            rootsift(&mut squared);
            quantize(&squared)
        })
        .collect();
    (base.iter().map(quantize).collect(), query_set)
}

fn exact(queries: &[&[u8]], database: &[&[u8]]) -> Vec<Vec<Match>> {
    BruteForceMatcher::match_direct_slices(SquaredL2DistanceU8::new(), queries, database, K)
        .expect("valid exact matching input")
}

fn hnsw_index(config: HnswConfig, database: &[&[u8]]) -> HnswMatcher<u8, SquaredL2DistanceU8> {
    let mut matcher = HnswMatcher::with_config(config.with_seed(SEED.into()));
    matcher.build(database).expect("valid HNSW database");
    matcher
}

fn hnsw(config: HnswConfig, queries: &[&[u8]], database: &[&[u8]]) -> Vec<Vec<Match>> {
    hnsw_index(config, database)
        .query_batch(queries, K)
        .expect("valid HNSW query")
}

fn slices(descriptors: &[Vec<u8>]) -> Vec<&[u8]> {
    descriptors.iter().map(Vec::as_slice).collect()
}

/// Fraction of queries whose approximate nearest neighbour is the exact one.
fn recall_at_1(reference: &[Vec<Match>], candidate: &[Vec<Match>]) -> f64 {
    let hits = reference
        .iter()
        .zip(candidate)
        .filter(|(reference, candidate)| {
            matches!(
                (reference.first(), candidate.first()),
                (Some(expected), Some(found)) if expected.index() == found.index()
            )
        })
        .count();
    hits as f64 / reference.len().max(1) as f64
}

fn bench_matchers(c: &mut Criterion) {
    let mut group = c.benchmark_group("matchers_u8");
    group.sample_size(10);
    for (n_queries, n_database) in SIZES {
        let (database, queries) = workload(n_queries, n_database);
        let (database, queries) = (slices(&database), slices(&queries));
        let label = format!("{n_queries}q_x_{n_database}db");

        let reference = exact(&queries, &database);
        println!(
            "{label}: recall@1 vs exact: hnsw {:.3}",
            recall_at_1(
                &reference,
                &hnsw(HnswConfig::balanced(), &queries, &database)
            ),
        );

        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(BenchmarkId::new("exact", &label), &(), |b, ()| {
            b.iter(|| exact(black_box(&queries), black_box(&database)))
        });
        group.bench_with_input(BenchmarkId::new("hnsw", &label), &(), |b, ()| {
            b.iter(|| {
                hnsw(
                    HnswConfig::balanced(),
                    black_box(&queries),
                    black_box(&database),
                )
            })
        });
    }
    group.finish();
}

fn bench_hnsw_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_presets_u8");
    group.sample_size(10);
    let (n_queries, n_database) = SIZES[SIZES.len() - 1];
    let (database, queries) = workload(n_queries, n_database);
    let (database, queries) = (slices(&database), slices(&queries));
    let reference = exact(&queries, &database);
    group.throughput(Throughput::Elements(n_queries as u64));

    for (name, config) in [
        ("balanced", HnswConfig::balanced()),
        (
            "fast_query",
            HnswConfig::default()
                .with_ef_construction(48)
                .with_ef_search(16),
        ),
        ("high_recall", HnswConfig::high_recall()),
    ] {
        let index = hnsw_index(config.clone(), &database);
        let found = index.query_batch(&queries, K).expect("valid HNSW query");
        println!(
            "hnsw {name}: recall@1 vs exact {:.3}",
            recall_at_1(&reference, &found)
        );
        group.bench_function(BenchmarkId::new("build", name), |b| {
            b.iter(|| hnsw_index(config.clone(), black_box(&database)))
        });
        group.bench_function(BenchmarkId::new("query", name), |b| {
            b.iter(|| index.query_batch(black_box(&queries), K))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matchers, bench_hnsw_presets);
criterion_main!(benches);
