// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RANSAC Throughput Benchmarks
//!
//! Benchmarks for measuring RANSAC performance with different solvers
//! and configurations.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{Matrix3, Point2, Vector3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use openmvg_core::{NormalizedImage, NormalizedImageThreshold};
use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::homography::HomographyConfig;
use openmvg_geometry::robust::classic::ClassicConfig;

/// Generate synthetic homography correspondences with outliers.
fn generate_homography_data(
    n_inliers: usize,
    n_outliers: usize,
    seed: u64,
) -> Vec<PixelCorrespondence> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Ground truth homography
    let angle = 0.1_f64;
    let h = Matrix3::new(
        angle.cos(),
        -angle.sin(),
        0.05,
        angle.sin(),
        angle.cos(),
        0.02,
        0.001,
        -0.001,
        1.0,
    );

    let mut correspondences = Vec::with_capacity(n_inliers + n_outliers);

    // Generate inliers
    for _ in 0..n_inliers {
        let p1 = Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));

        let p1_h = Vector3::new(p1.x, p1.y, 1.0);
        let p2_h = h * p1_h;
        let p2 = Point2::new(p2_h.x / p2_h.z, p2_h.y / p2_h.z);

        // Add small noise
        let noise = Point2::new(rng.r#gen::<f64>() * 0.001, rng.r#gen::<f64>() * 0.001);

        correspondences.push(PixelCorrespondence {
            point1: p1 + noise.coords,
            point2: p2 + noise.coords,
        });
    }

    // Generate outliers
    for _ in 0..n_outliers {
        correspondences.push(PixelCorrespondence {
            point1: Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
            point2: Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
        });
    }

    // Shuffle
    correspondences.shuffle(&mut rng);

    correspondences
}

/// Generate synthetic fundamental matrix correspondences with outliers.
fn generate_fundamental_data(
    n_inliers: usize,
    n_outliers: usize,
    seed: u64,
) -> Vec<PixelCorrespondence> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Simple fundamental matrix (rank-deficient)
    let f: Matrix3<f64> = Matrix3::new(0.0, -0.001, 0.1, 0.001, 0.0, -0.2, -0.1, 0.2, 0.0);

    let mut correspondences = Vec::with_capacity(n_inliers + n_outliers);

    // Generate inliers on epipolar lines
    for _ in 0..n_inliers {
        let p1 = Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));

        // Compute epipolar line in second image
        let x1_h = Vector3::new(p1.x, p1.y, 1.0);
        let line = f * x1_h;
        let (a, b, c) = (line.x, line.y, line.z);

        // Find point on line
        let p2 = if b.abs() > 1e-6 {
            let x2 = p1.x + rng.gen_range(-0.1..0.1);
            let y2 = -(a * x2 + c) / b;
            Point2::new(x2, y2)
        } else if a.abs() > 1e-6 {
            let y2 = p1.y + rng.gen_range(-0.1..0.1);
            let x2 = -(b * y2 + c) / a;
            Point2::new(x2, y2)
        } else {
            continue;
        };

        correspondences.push(PixelCorrespondence {
            point1: p1,
            point2: p2,
        });
    }

    // Generate outliers
    for _ in 0..n_outliers {
        correspondences.push(PixelCorrespondence {
            point1: Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
            point2: Point2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
        });
    }

    correspondences.shuffle(&mut rng);

    correspondences
}

fn make_ransac_config(max_iterations: usize, threshold: f64) -> ClassicConfig<NormalizedImage> {
    ClassicConfig::ransac(NormalizedImageThreshold::try_new(threshold).unwrap())
        .with_max_iterations(max_iterations)
        .build()
}

fn ransac_homography_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RANSAC Homography");

    // Test different data sizes and outlier ratios
    let configs = [
        (50, 10),   // 50 inliers, 10 outliers (17% outliers)
        (100, 30),  // 100 inliers, 30 outliers (23% outliers)
        (200, 100), // 200 inliers, 100 outliers (33% outliers)
        (500, 250), // 500 inliers, 250 outliers (33% outliers)
    ];

    for (n_inliers, n_outliers) in configs.iter() {
        let data = generate_homography_data(*n_inliers, *n_outliers, 42);
        let total = n_inliers + n_outliers;
        let outlier_pct = (*n_outliers as f64 / total as f64) * 100.0;

        let ransac_config = make_ransac_config(1000, 0.01);
        ransac_config
            .estimate_homography(&data, &HomographyConfig::default())
            .expect("homography benchmark fixture must produce a model");

        group.bench_with_input(
            BenchmarkId::new("DLT", format!("{}pts_{}%out", total, outlier_pct as u32)),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = black_box(&ransac_config).estimate_homography(
                        black_box(data),
                        black_box(&HomographyConfig::default()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn ransac_fundamental_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RANSAC Fundamental");

    let configs = [
        (50, 10),   // 17% outliers
        (100, 30),  // 23% outliers
        (200, 100), // 33% outliers
    ];

    for (n_inliers, n_outliers) in configs.iter() {
        let data = generate_fundamental_data(*n_inliers, *n_outliers, 42);
        let total = n_inliers + n_outliers;
        let outlier_pct = (*n_outliers as f64 / total as f64) * 100.0;

        let ransac_config = make_ransac_config(2000, 0.01);
        ransac_config
            .estimate_fundamental(&data)
            .expect("fundamental benchmark fixture must produce a model");

        group.bench_with_input(
            BenchmarkId::new("8pt", format!("{}pts_{}%out", total, outlier_pct as u32)),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = black_box(&ransac_config).estimate_fundamental(black_box(data));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn ransac_iteration_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("RANSAC Iteration Scaling");

    // Fixed data size, varying max iterations
    let data = generate_homography_data(100, 50, 42);

    for max_iter in [100, 500, 1000, 2000, 5000] {
        let ransac_config = make_ransac_config(max_iter, 0.01);
        ransac_config
            .estimate_homography(&data, &HomographyConfig::default())
            .expect("iteration benchmark fixture must produce a model");

        group.bench_with_input(BenchmarkId::new("max_iter", max_iter), &data, |b, data| {
            b.iter(|| {
                let result = black_box(&ransac_config)
                    .estimate_homography(black_box(data), black_box(&HomographyConfig::default()));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    ransac_homography_benchmark,
    ransac_fundamental_benchmark,
    ransac_iteration_scaling,
);

criterion_main!(benches);
