// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Match keypoints between two views of a scene and keep only the unambiguous matches.
//!
//! You will learn: how to detect and describe two images with ALP, find each descriptor's two
//! nearest neighbors in the other image with an HNSW index, keep the matches that pass Lowe's
//! ratio test in both directions, as `openmvg-sfm` does, and check them against a known motion.
//! Data: a textured synthetic scene of Gaussian blobs, and the same scene scaled by 1.3 and
//! rotated by 20°, both rendered exactly, with noise of one intensity level; seed 3, no files.
//! The scene comes from `openmvg-core`'s seeded generator, so a copy of this file also depends
//! on `openmvg-core`.
//! Run: cargo run --release -p openmvg-features --example match_and_filter
//! Expected: more than 95% of the matches that pass the ratio test both ways agree with the
//! motion within 2 px, about twice the share among the raw nearest neighbors.
//! Next: the `robust_homography` and `relative_pose` examples of `openmvg-geometry` estimate a
//! model from such matches.

use openmvg_core::{AlgorithmRng, RunSeed};
use openmvg_features::extraction::{AlpExtractor, AlpPreset, GrayImage, U8Descriptor};
use openmvg_features::matching::{HnswMatcher, MatchAcceptance, SquaredL2DistanceU8};

const WIDTH: usize = 512;
const HEIGHT: usize = 384;

/// The motion between the views: scale by 1.3 and rotate by 20° about the image center.
fn warp(x: f64, y: f64) -> (f64, f64) {
    let (cx, cy, scale, angle) = (256.0, 192.0, 1.3, 20_f64.to_radians());
    let (dx, dy) = (x - cx, y - cy);
    let (c, s) = (angle.cos(), angle.sin());
    (
        cx + scale * (c * dx - s * dy),
        cy + scale * (s * dx + c * dy),
    )
}

/// Render blobs `(x, y, width, amplitude)` after `motion`, on a gray background with noise.
fn render(
    blobs: &[(f64, f64, f64, f64)],
    motion: impl Fn(f64, f64) -> (f64, f64),
    scale: f64,
    rng: &mut AlgorithmRng,
) -> GrayImage {
    let mut pixels = vec![128.0; WIDTH * HEIGHT];
    for &(x, y, width, amplitude) in blobs {
        let ((bx, by), width) = (motion(x, y), width * scale);
        // Draw each blob within five widths of its center, where it is visible.
        for row in within(by, 5.0 * width, HEIGHT) {
            for column in within(bx, 5.0 * width, WIDTH) {
                let (dx, dy) = (column as f64 - bx, row as f64 - by);
                let gaussian = (-(dx * dx + dy * dy) / (2.0 * width * width)).exp();
                pixels[row * WIDTH + column] += amplitude * gaussian;
            }
        }
    }
    // Add Gaussian noise with a standard deviation of one intensity level.
    for value in &mut pixels {
        *value = (*value + f64::from(rng.standard_normal_f32())).clamp(0.0, 255.0);
    }
    GrayImage::from_fn(HEIGHT, WIDTH, |row, column| {
        pixels[row * WIDTH + column] as f32
    })
}

/// The descriptors of a feature set, as the slices a matcher reads.
fn slices(descriptors: &[U8Descriptor]) -> Vec<&[u8]> {
    descriptors
        .iter()
        .map(|d| d.as_array().as_slice())
        .collect()
}

/// The pixel indices within `reach` of `center`, inside an image dimension of `size`.
fn within(center: f64, reach: f64, size: usize) -> std::ops::Range<usize> {
    let high = (((center + reach).max(0.0) as usize) + 1).min(size);
    (((center - reach).max(0.0) as usize).min(high))..high
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = AlgorithmRng::new(RunSeed::from_u64(3));
    // Blobs over a margin around the frame, so that the warped view is textured everywhere.
    let blobs: Vec<_> = (0..1500)
        .map(|_| {
            let x = WIDTH as f64 * (2.0 * rng.unit_f64() - 0.5);
            let y = HEIGHT as f64 * (2.0 * rng.unit_f64() - 0.5);
            let width = 1.5 * (8.0_f64 / 1.5).powf(rng.unit_f64());
            let amplitude = (20.0 + 60.0 * rng.unit_f64()) * rng.symmetric_unit_f64().signum();
            (x, y, width, amplitude)
        })
        .collect();
    let first = render(&blobs, |x, y| (x, y), 1.0, &mut rng);
    let second = render(&blobs, warp, 1.3, &mut rng);

    // Detect and describe both views; u8 descriptors are what openmvg-sfm stores and matches.
    let extractor = AlpExtractor::with_preset(AlpPreset::Normal);
    let (a, b) = (
        extractor.extract_u8(&first)?,
        extractor.extract_u8(&second)?,
    );
    let (queries, database) = (slices(a.descriptors()), slices(b.descriptors()));

    // One index per image. The forward queries find the two nearest neighbors of each
    // descriptor of the first image among those of the second, in squared L2 distance.
    let mut index_a: HnswMatcher<u8, SquaredL2DistanceU8> = HnswMatcher::new();
    let mut index_b: HnswMatcher<u8, SquaredL2DistanceU8> = HnswMatcher::new();
    index_a.build(&queries)?;
    index_b.build(&database)?;
    let neighbors = index_b.query_batch(&queries, 2)?;

    // A match is kept only when its nearest neighbor is clearly closer than the second, and the
    // target, searched for in the first image, chooses the query back. The 0.8 ratio is stated
    // on plain distances; `filter_squared` squares it to compare squared ones.
    let kept = MatchAcceptance::Reciprocal.filter_squared(&neighbors, 0.8, |targets| {
        let rows: Vec<_> = targets.iter().map(|&target| database[target]).collect();
        index_a.query_batch(&rows, 2)
    })?;

    // A match is correct when the motion sends its first keypoint to its second, within 2 px.
    let correct = |query: usize, target: usize| {
        let (p, q) = (&a.keypoints()[query], &b.keypoints()[target]);
        let (x, y) = warp(f64::from(p.x), f64::from(p.y));
        (x - f64::from(q.x)).hypot(y - f64::from(q.y)) < 2.0
    };
    let nearest = neighbors
        .iter()
        .enumerate()
        .filter_map(|(i, n)| Some((i, n.first()?.index())));
    let raw_correct = nearest
        .filter(|&(query, target)| correct(query, target))
        .count();
    let kept_correct = kept
        .iter()
        .filter(|m| correct(m.query_index(), m.target_index()))
        .count();
    let raw = raw_correct as f64 / neighbors.len() as f64;
    let filtered = kept_correct as f64 / kept.len() as f64;

    println!(
        "{} and {} keypoints",
        a.keypoints().len(),
        b.keypoints().len()
    );
    println!(
        "nearest neighbors: {raw_correct} of {} correct",
        neighbors.len()
    );
    println!(
        "after the ratio test both ways: {kept_correct} of {} correct",
        kept.len()
    );
    assert!(
        filtered > 0.95 && filtered > raw,
        "the ratio test removes wrong matches"
    );
    assert!(
        kept_correct >= 50,
        "enough correct matches remain to estimate a motion"
    );
    Ok(())
}
