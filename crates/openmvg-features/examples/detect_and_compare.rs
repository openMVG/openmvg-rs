// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Detect keypoints with ALP and SIFT on the same image, measure how precisely each locates the
//! blob centers, and check the scale ALP reports.
//!
//! You will learn: how to fill a `GrayImage`, run the two detectors with their aligned `Normal`
//! presets, and compare them against a known truth: where each blob is and its standard
//! deviation.
//! On these clean, isolated blobs ALP locates the centers more precisely, and the printed
//! timings show how fast each detector runs on your machine. This example does not test scale
//! and rotation changes of textured scenes, where SIFT keeps the scale more consistent and both
//! repeat and match about equally; measure on your own images before you choose.
//! Data: a 480 × 384 image with 20 bright and dark Gaussian blobs, of standard deviation 2.75
//! to 7.5 px, at sub-pixel positions; no noise, no files.
//! Run: cargo run --release -p openmvg-features --example detect_and_compare
//! Expected: both detectors find all 20 blobs; ALP's mean center error is less than half of
//! SIFT's, and its scale matches each blob's standard deviation within 5%.
//! Next: the `match_and_filter` example matches keypoints between two views.

use std::time::Instant;

use openmvg_features::extraction::{
    AlpExtractor, AlpPreset, GrayImage, ScaleSpaceKeypoint, SiftExtractor, SiftPreset,
};

const COLUMNS: usize = 5;
const ROWS: usize = 4;
const CELL: usize = 96;

/// Center and standard deviation of blob `i`: one blob per cell, at a sub-pixel offset.
fn blob(i: usize) -> (f64, f64, f64) {
    let offset = |k: f64| (k * 0.618_033_988_75 + 0.13).fract() - 0.5;
    let x = (i % COLUMNS * CELL + CELL / 2) as f64 + offset(i as f64);
    let y = (i / COLUMNS * CELL + CELL / 2) as f64 + offset(i as f64 + 7.0);
    (x, y, 2.75 + 0.25 * i as f64)
}

/// For every blob, the center error and the scale ratio of the nearest keypoint within 1 px.
fn measure(keypoints: &[ScaleSpaceKeypoint]) -> Option<Vec<(f64, f64)>> {
    (0..COLUMNS * ROWS)
        .map(|i| {
            let (x, y, sigma) = blob(i);
            keypoints
                .iter()
                .map(|k| (f64::from(k.x) - x).hypot(f64::from(k.y) - y))
                .zip(keypoints.iter().map(|k| f64::from(k.sigma) / sigma))
                .filter(|&(error, _)| error < 1.0)
                .min_by(|a, b| a.0.total_cmp(&b.0))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rows are image rows; intensities run from 0 to 255 around a gray background.
    let image = GrayImage::from_fn(ROWS * CELL, COLUMNS * CELL, |row, column| {
        let blobs = (0..COLUMNS * ROWS).map(|i| {
            let (x, y, sigma) = blob(i);
            let squared = (column as f64 - x).powi(2) + (row as f64 - y).powi(2);
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * 100.0 * (-squared / (2.0 * sigma * sigma)).exp()
        });
        (128.0 + blobs.sum::<f64>()) as f32
    });

    let start = Instant::now();
    let alp = AlpExtractor::with_preset(AlpPreset::Normal).extract_u8(&image)?;
    let alp_time = start.elapsed();
    let start = Instant::now();
    let sift = SiftExtractor::with_preset(SiftPreset::Normal).extract_u8(&image)?;
    let sift_time = start.elapsed();

    let alp_blobs = measure(alp.keypoints()).ok_or("ALP missed a blob")?;
    let sift_blobs = measure(sift.keypoints()).ok_or("SIFT missed a blob")?;
    let mean_error =
        |blobs: &[(f64, f64)]| blobs.iter().map(|b| b.0).sum::<f64>() / blobs.len() as f64;
    let (alp_error, sift_error) = (mean_error(&alp_blobs), mean_error(&sift_blobs));
    let worst_scale = alp_blobs
        .iter()
        .map(|b| (b.1 - 1.0).abs())
        .fold(0.0, f64::max);

    // Timings vary with the machine and the build; they are printed, not checked.
    println!("ALP  found all 20 blobs: mean center error {alp_error:.4} px, {alp_time:.1?}");
    println!("SIFT found all 20 blobs: mean center error {sift_error:.4} px, {sift_time:.1?}");
    println!(
        "ALP scale matches each blob's standard deviation within {:.1}%",
        100.0 * worst_scale
    );
    assert!(
        alp_error < 0.5 * sift_error,
        "ALP locates clean blob centers more precisely"
    );
    assert!(
        worst_scale < 0.05,
        "ALP's scale is the blob's standard deviation"
    );
    Ok(())
}
