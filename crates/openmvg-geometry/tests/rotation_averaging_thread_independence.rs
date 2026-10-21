// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rotation averaging must give the same bits for any thread count.
//!
//! Faer's high-level sparse solvers follow its process-global parallelism, and its Rayon
//! LLT can change the result bits with the worker count. This binary disables that global
//! setting, so any solver path that still reads it panics.
#![cfg(feature = "faer")]

use nalgebra::{Rotation3, Vector3};
use openmvg_core::model::{RelativeRotation, ViewId};
use openmvg_geometry::averaging::{RotationConfig, RotationEstimate, average_rotations};

const VIEWS: usize = 120;

/// A ring plus skip-7 chords, with small deterministic edge noise and weights.
fn noisy_ring() -> Vec<RelativeRotation> {
    let truth: Vec<Rotation3<f64>> = (0..VIEWS)
        .map(|i| {
            let a = i as f64;
            Rotation3::new(Vector3::new(0.031 * a, 0.017 * a, 0.011 * a))
        })
        .collect();
    let mut edges = Vec::new();
    for i in 0..VIEWS {
        for step in [1, 7] {
            let j = (i + step) % VIEWS;
            let k = edges.len() as f64;
            let noise = Rotation3::new(Vector3::new(
                1e-3 * (k % 7.0),
                -1e-3 * (k % 5.0),
                5e-4 * (k % 3.0),
            ));
            edges.push(
                RelativeRotation::try_new(
                    ViewId::new(i),
                    ViewId::new(j),
                    noise * truth[j] * truth[i].inverse(),
                    1.0 + 0.25 * (k % 4.0),
                )
                .unwrap(),
            );
        }
    }
    edges
}

fn bits(result: &RotationEstimate) -> Vec<u64> {
    result
        .rotations
        .values()
        .flat_map(|rotation| rotation.matrix().iter().map(|value| value.to_bits()))
        .collect()
}

#[test]
fn average_rotations_ignores_faer_global_parallelism() {
    faer::disable_global_parallelism();
    let edges = noisy_ring();
    let config = RotationConfig::default();
    let solve = || bits(&average_rotations(&edges, &config).unwrap());
    let reference = solve();
    assert_eq!(reference.len(), 9 * VIEWS);

    #[cfg(feature = "parallel")]
    for threads in [1, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(solve), reference, "{threads} threads");
    }
}
