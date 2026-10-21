// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fundamental-matrix benchmarks: the minimal seven-point solver and the robust estimator.
//!
//! Both run on one fixed-seed synthetic two-view scene that is validated against the
//! ground-truth fundamental matrix before anything is timed.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{Matrix3, Rotation3, Vector3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::fundamental::{SevenPointSolver, estimate_fundamental};
use openmvg_geometry::normalization::NormalizedCorrespondences;
use openmvg_geometry::robust::RobustOptions;
use openmvg_geometry::robust::threshold::PixelThreshold;

const INLIERS: usize = 160;
const OUTLIERS: usize = 40;

struct Fixture {
    /// Seven noise-free correspondences.
    minimal: NormalizedCorrespondences,
    /// Noisy inliers (the first `INLIERS`) followed by random outliers.
    data: Vec<PixelCorrespondence>,
    /// Ground-truth fundamental matrix.
    expected: Matrix3<f64>,
}

fn fixture() -> Fixture {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let intrinsic = Matrix3::new(900.0, 0.0, 620.0, 0.0, 900.0, 470.0, 0.0, 0.0, 1.0);
    let rotation = Rotation3::from_euler_angles(0.05, -0.12, 0.07);
    let translation = Vector3::new(0.48, -0.14, 0.08);
    let project = |point: Vector3<f64>| {
        let pixel = intrinsic * point;
        (pixel.x / pixel.z, pixel.y / pixel.z)
    };
    let exact = (0..INLIERS)
        .map(|_| {
            let point = Vector3::new(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-1.5..1.5),
                rng.gen_range(4.0..8.0),
            );
            (project(point), project(rotation * point + translation))
        })
        .collect::<Vec<_>>();
    let mut data = exact
        .iter()
        .map(|&((x1, y1), (x2, y2))| {
            let mut noise = || rng.gen_range(-0.3..0.3);
            PixelCorrespondence::from_coords(x1 + noise(), y1 + noise(), x2 + noise(), y2 + noise())
        })
        .collect::<Vec<_>>();
    data.extend((0..OUTLIERS).map(|_| {
        PixelCorrespondence::from_coords(
            rng.gen_range(0.0..1240.0),
            rng.gen_range(0.0..940.0),
            rng.gen_range(0.0..1240.0),
            rng.gen_range(0.0..940.0),
        )
    }));
    let minimal = exact[..7]
        .iter()
        .map(|&((x1, y1), (x2, y2))| PixelCorrespondence::from_coords(x1, y1, x2, y2))
        .collect::<Vec<_>>();
    let inverse = intrinsic
        .try_inverse()
        .expect("intrinsic matrix is invertible");
    let expected = inverse.transpose() * translation.cross_matrix() * rotation.matrix() * inverse;
    Fixture {
        minimal: NormalizedCorrespondences::from_pixels(&minimal),
        data,
        expected,
    }
}

/// Distance between two fundamental matrices, up to scale and sign.
fn projective_error(left: &Matrix3<f64>, right: &Matrix3<f64>) -> f64 {
    let (left, right) = (left / left.norm(), right / right.norm());
    (left - right).norm().min((left + right).norm())
}

fn bench_fundamental(c: &mut Criterion) {
    let fixture = fixture();
    let threshold = PixelThreshold::try_new(1.0).expect("positive threshold");
    let options = RobustOptions::default().with_seed(31).with_max_draws(512);

    let models = SevenPointSolver::solve(&fixture.minimal).expect("seven-point solve");
    assert!(
        models
            .iter()
            .any(|model| projective_error(model.matrix(), &fixture.expected) < 1e-6),
        "no seven-point root matches the ground truth"
    );
    let robust = estimate_fundamental(&fixture.data, threshold, &options).expect("robust estimate");
    assert!(projective_error(robust.model().matrix(), &fixture.expected) < 0.03);
    let recovered = robust
        .inliers()
        .iter()
        .filter(|&&index| index < INLIERS)
        .count();
    assert!(
        recovered >= INLIERS * 9 / 10,
        "only {recovered} inliers recovered"
    );

    let mut group = c.benchmark_group("fundamental");
    group.bench_function("seven_point_solver", |b| {
        b.iter(|| SevenPointSolver::solve(black_box(&fixture.minimal)))
    });
    group.bench_function("estimate_fundamental_200_points_20pct_outliers", |b| {
        b.iter(|| estimate_fundamental(black_box(&fixture.data), threshold, &options))
    });
    group.finish();
}

criterion_group!(benches, bench_fundamental);
criterion_main!(benches);
