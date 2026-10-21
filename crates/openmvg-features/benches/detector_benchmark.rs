// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ALP and SIFT detection on a deterministic synthetic texture.
//! Run with: pixi run cargo bench -p openmvg-features --bench detector_benchmark

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra as na;

type GrayImage = na::DMatrix<f32>;

fn create_test_image(size: usize) -> GrayImage {
    let mut img = GrayImage::zeros(size, size);

    // Create a textured image with gradients and blobs
    for i in 0..size {
        for j in 0..size {
            let x = j as f32 / size as f32;
            let y = i as f32 / size as f32;
            img[(i, j)] =
                128.0 + 100.0 * (x * 10.0).sin() * (y * 10.0).cos() + 50.0 * ((x + y) * 20.0).sin();
        }
    }
    img
}

fn bench_alp_detect(c: &mut Criterion) {
    use openmvg_features::extraction::AlpExtractor;

    let mut group = c.benchmark_group("ALP_detect");

    for size in [256, 512, 1024].iter() {
        let img = create_test_image(*size);

        group.bench_with_input(BenchmarkId::new("size", size), &img, |b, img| {
            let detector = AlpExtractor::new(10.8, 4.8, 24.0);
            b.iter(|| detector.extract_f32(img).unwrap());
        });
    }

    group.finish();
}

fn bench_sift_detect(c: &mut Criterion) {
    use openmvg_features::extraction::SiftExtractor;

    let mut group = c.benchmark_group("SIFT_detect");

    for size in [256, 512, 1024].iter() {
        let img = create_test_image(*size);

        group.bench_with_input(BenchmarkId::new("size", size), &img, |b, img| {
            let detector = SiftExtractor::default();
            b.iter(|| detector.extract_f32(img).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_alp_detect, bench_sift_detect);
criterion_main!(benches);
