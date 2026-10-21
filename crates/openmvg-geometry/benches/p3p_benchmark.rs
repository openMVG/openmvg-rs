// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comparative P3P hot-path benchmark.
//!
//! Every lane measures the public P3P contract through the
//! `pose::p3p::solve` dispatcher: validated input, finite `SO(3)`
//! `WorldToCameraPose` output, and cheirality.
//! Raw matrix-output paper kernels are a different, cheaper contract.
//!
//! Run with `pixi run cargo bench -p openmvg-geometry --bench p3p_benchmark`.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use nalgebra::{Isometry3, Point3, UnitQuaternion, UnitVector3, Vector3};
use openmvg_geometry::pose::p3p::{P3pMethod, solve};
use rand::{Rng, SeedableRng, rngs::StdRng};

type Case = ([Point3<f64>; 3], [UnitVector3<f64>; 3]);

fn corpus(count: usize) -> Vec<Case> {
    let mut rng = StdRng::seed_from_u64(0x5033_5eed);
    (0..count)
        .map(|_| {
            let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ));
            let pose = Isometry3::from_parts(
                Vector3::new(
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-2.0..2.0),
                )
                .into(),
                rotation,
            );
            let camera = [
                Point3::new(
                    rng.gen_range(-2.0..-0.3),
                    rng.gen_range(-1.5..0.4),
                    rng.gen_range(3.0..6.0),
                ),
                Point3::new(
                    rng.gen_range(0.3..2.0),
                    rng.gen_range(-1.5..0.4),
                    rng.gen_range(4.0..8.0),
                ),
                Point3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(0.5..2.0),
                    rng.gen_range(5.0..9.0),
                ),
            ];
            (
                camera.map(|point| pose.inverse_transform_point(&point)),
                camera.map(|point| UnitVector3::new_normalize(point.coords)),
            )
        })
        .collect()
}

fn benchmark_p3p(c: &mut Criterion) {
    let cases = corpus(256);
    let mut group = c.benchmark_group("p3p_checked_pose");
    group.throughput(Throughput::Elements(1));

    for (name, method) in [
        ("Fused", P3pMethod::Fused),
        ("Ding", P3pMethod::Ding),
        ("Ke_orthonormalized", P3pMethod::KeOrtho),
        ("Kneip", P3pMethod::Kneip),
        ("LambdaTwist", P3pMethod::LambdaTwist),
        ("Wu", P3pMethod::Wu),
        ("Nakano", P3pMethod::Nakano),
    ] {
        let mut index = 0;
        group.bench_function(name, |benchmark| {
            benchmark.iter(|| {
                let (world, bearings) = cases[index & 255];
                index = index.wrapping_add(1);
                black_box(solve(black_box(world), black_box(bearings), method))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_p3p);
criterion_main!(benches);
