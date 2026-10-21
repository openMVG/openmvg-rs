// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accuracy of global positioning under bearing noise and gross outliers.
//!
//! Camera rotations are known (identity), so the result is determined up to a translation and a
//! positive scale. The error is the aligned camera-center RMS over the RMS spread of the true
//! centers, which is unitless and comparable across scenes.

use std::collections::BTreeMap;

use nalgebra::{Rotation3, Vector3};
use openmvg_core::ViewId;
use openmvg_geometry::averaging::global_positioning::{GlobalPositioningConfig, position_globally};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const CAMERA_COUNT: usize = 24;
const POINT_COUNT: usize = 160;
const VIEWS_PER_POINT: usize = 6;

struct Scene {
    centers: Vec<Vector3<f64>>,
    tracks: BTreeMap<usize, Vec<(ViewId, Vector3<f64>)>>,
}

fn unit(rng: &mut ChaCha8Rng) -> Vector3<f64> {
    loop {
        let value = Vector3::new(
            StandardNormal.sample(rng),
            StandardNormal.sample(rng),
            StandardNormal.sample(rng),
        );
        if value.norm() > 1e-6 {
            return value.normalize();
        }
    }
}

/// Cameras in a cube looking down +z at points in front of them. `noise_rad` perturbs every
/// bearing; each observation is replaced with probability `outlier_fraction` by a random
/// direction (even draws) or by its opposite (odd draws).
fn scene(seed: u64, noise_rad: f64, outlier_fraction: f64) -> Scene {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centers: Vec<Vector3<f64>> = (0..CAMERA_COUNT)
        .map(|_| {
            Vector3::new(
                rng.gen_range(0.0..6.0),
                rng.gen_range(0.0..6.0),
                rng.gen_range(0.0..6.0),
            )
        })
        .collect();
    let mut tracks = BTreeMap::new();
    let mut outliers = 0usize;
    for point in 0..POINT_COUNT {
        let world = Vector3::new(
            rng.gen_range(-4.0..10.0),
            rng.gen_range(-4.0..10.0),
            rng.gen_range(10.0..20.0),
        );
        let mut cameras: Vec<usize> = (0..CAMERA_COUNT).collect();
        let mut observations = Vec::new();
        for _ in 0..VIEWS_PER_POINT {
            let camera = cameras.swap_remove(rng.gen_range(0..cameras.len()));
            let mut ray = (world - centers[camera]).normalize();
            ray = (ray + unit(&mut rng) * noise_rad).normalize();
            if rng.gen_range(0.0..1.0) < outlier_fraction {
                ray = if outliers.is_multiple_of(2) {
                    unit(&mut rng)
                } else {
                    -ray
                };
                outliers += 1;
            }
            observations.push((ViewId::new(camera), ray));
        }
        tracks.insert(point, observations);
    }
    Scene { centers, tracks }
}

/// Aligned RMS camera error over the RMS spread of the true centers.
fn aligned_error(truth: &[Vector3<f64>], estimate: &[Vector3<f64>]) -> f64 {
    let n = truth.len() as f64;
    let truth_mean = truth.iter().sum::<Vector3<f64>>() / n;
    let estimate_mean = estimate.iter().sum::<Vector3<f64>>() / n;
    let (mut cross, mut spread) = (0.0, 0.0);
    for (truth, estimate) in truth.iter().zip(estimate) {
        cross += (estimate - estimate_mean).dot(&(truth - truth_mean));
        spread += (estimate - estimate_mean).norm_squared();
    }
    let scale = cross / spread;
    let squared: f64 = truth
        .iter()
        .zip(estimate)
        .map(|(truth, estimate)| {
            ((truth - truth_mean) - scale * (estimate - estimate_mean)).norm_squared()
        })
        .sum();
    let truth_spread: f64 = truth.iter().map(|t| (t - truth_mean).norm_squared()).sum();
    (squared / truth_spread).sqrt()
}

fn solve(scene: &Scene) -> Option<f64> {
    let rotations: BTreeMap<ViewId, Rotation3<f64>> = (0..CAMERA_COUNT)
        .map(|camera| (ViewId::new(camera), Rotation3::identity()))
        .collect();
    let estimate = position_globally(
        &rotations,
        &scene.tracks,
        &[],
        &GlobalPositioningConfig::default(),
    )
    .ok()?;
    let centers: Vec<Vector3<f64>> = (0..CAMERA_COUNT)
        .map(|camera| estimate.translations[&ViewId::new(camera)])
        .collect();
    Some(aligned_error(&scene.centers, &centers))
}

/// Mean and worst aligned error over eight seeds; every seed must solve.
fn sweep(noise_deg: f64, outlier_fraction: f64) -> (f64, f64) {
    let errors: Vec<f64> = (0..8)
        .map(|seed| {
            solve(&scene(seed, f64::to_radians(noise_deg), outlier_fraction))
                .expect("a well-posed scene must solve")
        })
        .collect();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    (mean, errors.iter().copied().fold(0.0, f64::max))
}

// The limits sit 15-25% above the errors measured when they were set (aligned error over the
// camera spread): exact scene 0.0000, 0.0043 with 0.2 degrees of noise, and 0.0065 / 0.0127 /
// 0.074 mean (worst 0.017 / 0.037 / 0.27) at 10 / 20 / 30% outliers. The projected Gauss-Newton
// polish run to convergence is what reaches them: eight steps leave 0.027 / 0.133 mean at 20 / 30%
// outliers, and the alternation alone stops at 0.036 clean and 0.156 / 0.30. Without the polish's
// robust-objective check, one 30% scene collapses to 0.999.

#[test]
fn clean_and_noisy_scenes_stay_accurate() {
    for (noise_deg, mean_limit, worst_limit) in [(0.0, 1e-3, 1e-3), (0.2, 0.0055, 0.0065)] {
        let (mean, worst) = sweep(noise_deg, 0.0);
        assert!(
            mean < mean_limit && worst < worst_limit,
            "noise {noise_deg}: {mean} / {worst}"
        );
    }
}

#[test]
fn gross_outliers_degrade_accuracy_gracefully() {
    for (outlier_fraction, mean_limit, worst_limit) in
        [(0.1, 0.008, 0.021), (0.2, 0.015, 0.045), (0.3, 0.09, 0.33)]
    {
        let (mean, worst) = sweep(0.2, outlier_fraction);
        assert!(
            mean < mean_limit && worst < worst_limit,
            "outliers {outlier_fraction}: mean {mean} worst {worst}"
        );
    }
}
