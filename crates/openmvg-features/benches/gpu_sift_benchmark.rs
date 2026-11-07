// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU against GPU SIFT on a folder of real images.
//!
//! The crate has no image decoder, so the benchmark reads 8-bit binary PGM
//! (`P5`) files. Convert a photo set once, for example with
//! `python -c "from PIL import Image; Image.open('a.jpg').convert('L').save('a.pgm')"`, then run:
//!
//! ```text
//! OMVG_SIFT_BENCH_DIR=path/to/pgm cargo bench -p openmvg-features --features gpu \
//!     --bench gpu_sift_benchmark
//! ```
//!
//! Optional variables: `OMVG_SIFT_BENCH_PRESET` (`normal`, `high` or `ultra`),
//! `OMVG_SIFT_BENCH_ROUNDS` (timed GPU rounds, default 5), `OMVG_SIFT_BENCH_CPU_ROUNDS`
//! (default 1) and `OMVG_SIFT_BENCH_SKIP_CPU=1`. `OMVG_SIFT_BENCH_MEMORY=1` instead
//! resizes the first image to several sizes and logs the GPU memory each size needs: the
//! extractor's resident buffers and, where `nvidia-smi` is available, the device memory a
//! fresh extractor adds, driver overhead included. Setting `OMVG_SIFT_BENCH_DUMP` to a
//! directory writes the GPU features there; `OMVG_SIFT_BENCH_REF` compares the GPU features
//! with such a dump, so a change can be checked for identical output.
//!
//! The report has three parts: per-image latency of the CPU extractor (which
//! already parallelizes inside one image), CPU throughput with one image per
//! Rayon worker, and GPU latency with one reused extractor. The parity section
//! matches each GPU keypoint to the nearest CPU keypoint.

use std::path::{Path, PathBuf};
use std::time::Instant;

use nalgebra as na;
use openmvg_features::extraction::{
    GrayImage, ScaleSpaceFeatures, SiftExtractor, SiftGpuExtractor, SiftPreset, U8Descriptor,
};
use rayon::prelude::*;

fn read_pgm(path: &Path) -> GrayImage {
    let bytes = std::fs::read(path).expect("readable PGM");
    let mut fields = Vec::new();
    let mut cursor = 0;
    while fields.len() < 4 {
        while bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if bytes[cursor] == b'#' {
            while bytes[cursor] != b'\n' {
                cursor += 1;
            }
            continue;
        }
        let start = cursor;
        while !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..cursor]).into_owned());
    }
    cursor += 1;
    assert_eq!(fields[0], "P5", "{} is not a binary PGM", path.display());
    let width: usize = fields[1].parse().expect("PGM width");
    let height: usize = fields[2].parse().expect("PGM height");
    assert_eq!(fields[3], "255", "only 8-bit PGM is supported");
    let pixels = &bytes[cursor..cursor + width * height];
    na::DMatrix::from_fn(height, width, |row, col| {
        f32::from(pixels[row * width + col])
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return f64::NAN;
    }
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn serialize(features: &ScaleSpaceFeatures<U8Descriptor>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (keypoint, descriptor) in features.keypoints().iter().zip(features.descriptors()) {
        for value in [
            keypoint.x,
            keypoint.y,
            keypoint.sigma,
            keypoint.orientation,
            keypoint.peak,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&(keypoint.octave as u32).to_le_bytes());
        bytes.extend_from_slice(&(keypoint.scale_index as u32).to_le_bytes());
        bytes.extend_from_slice(descriptor.as_array());
    }
    bytes
}

const RECORD: usize = 7 * 4 + 128;

/// Differences between the GPU features and a reference dump.
#[derive(Default)]
struct Diff {
    count_mismatch: usize,
    records: usize,
    geometry: usize,
    descriptors: usize,
    max_position: f32,
    max_orientation: f32,
    max_byte: u8,
}

impl Diff {
    fn add(&mut self, bytes: &[u8], expected: &[u8]) {
        if bytes.len() != expected.len() {
            self.count_mismatch += 1;
            return;
        }
        let field = |record: &[u8], index: usize| {
            f32::from_le_bytes(
                record[index * 4..index * 4 + 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            )
        };
        for (a, b) in bytes.chunks(RECORD).zip(expected.chunks(RECORD)) {
            self.records += 1;
            if a[..28] != b[..28] {
                self.geometry += 1;
                let position = (field(a, 0) - field(b, 0))
                    .abs()
                    .max((field(a, 1) - field(b, 1)).abs());
                self.max_position = self.max_position.max(position);
                let mut angle = (field(a, 3) - field(b, 3)).abs();
                angle = angle.min(std::f32::consts::TAU - angle);
                self.max_orientation = self.max_orientation.max(angle);
            }
            if a[28..] != b[28..] {
                self.descriptors += 1;
                for (x, y) in a[28..].iter().zip(&b[28..]) {
                    self.max_byte = self.max_byte.max(x.abs_diff(*y));
                }
            }
        }
    }

    fn report(&self) {
        println!(
            "  differences: {} images with another feature count; in the others {} of {} \
             features differ in geometry (max position {:.2e} px, max orientation {:.2e} rad) \
             and {} in descriptor bytes (max byte difference {})",
            self.count_mismatch,
            self.geometry,
            self.records,
            self.max_position,
            self.max_orientation,
            self.descriptors,
            self.max_byte
        );
    }
}

/// Bilinear resize to `scale` times the size, keeping exact 8-bit values.
fn resize(image: &GrayImage, scale: f32) -> GrayImage {
    let rows = ((image.nrows() as f32 * scale).round() as usize).max(8);
    let cols = ((image.ncols() as f32 * scale).round() as usize).max(8);
    let sample = |row: usize, col: usize| {
        let y = (row as f32 + 0.5) / scale - 0.5;
        let x = (col as f32 + 0.5) / scale - 0.5;
        let (y0, x0) = (y.floor().max(0.0) as usize, x.floor().max(0.0) as usize);
        let (y1, x1) = (
            (y0 + 1).min(image.nrows() - 1),
            (x0 + 1).min(image.ncols() - 1),
        );
        let (y0, x0) = (y0.min(image.nrows() - 1), x0.min(image.ncols() - 1));
        let (fy, fx) = (
            (y - y0 as f32).clamp(0.0, 1.0),
            (x - x0 as f32).clamp(0.0, 1.0),
        );
        let top = image[(y0, x0)] * (1.0 - fx) + image[(y0, x1)] * fx;
        let bottom = image[(y1, x0)] * (1.0 - fx) + image[(y1, x1)] * fx;
        (top * (1.0 - fy) + bottom * fy).round()
    };
    na::DMatrix::from_fn(rows, cols, sample)
}

/// Device memory in use according to `nvidia-smi`, in MiB, if it is available.
fn device_memory_mib() -> Option<f64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

fn memory_sweep(image: &GrayImage, preset: SiftPreset) {
    const MIB: f64 = 1024.0 * 1024.0;
    println!(
        "| Size | Megapixels | Features | Resident buffers (MiB) | Bytes per input pixel | Device memory added (MiB) | First call (ms) |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for scale in [0.25f32, 0.5, 0.75, 1.0, 1.5, 2.0] {
        let image = resize(image, scale);
        let pixels = image.len() as f64;
        let before = device_memory_mib();
        let gpu = match SiftGpuExtractor::with_preset(preset) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("GPU extractor unavailable: {error}; skipping");
                return;
            }
        };
        let start = Instant::now();
        let features = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => {
                println!(
                    "| {}x{} | {:.1} | {error} | | | | |",
                    image.ncols(),
                    image.nrows(),
                    pixels / 1e6
                );
                continue;
            }
        };
        let elapsed = ms(start);
        let resident = gpu.resident_bytes() as f64;
        let added = before
            .zip(device_memory_mib())
            .map(|(before, after)| after - before);
        println!(
            "| {}x{} | {:.1} | {} | {:.0} | {:.0} | {} | {elapsed:.1} |",
            image.ncols(),
            image.nrows(),
            pixels / 1e6,
            features.keypoints().len(),
            resident / MIB,
            resident / pixels,
            added.map_or_else(|| "n/a".to_string(), |added| format!("{added:.0}")),
        );
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// How closely the GPU features follow the CPU ones.
#[derive(Default)]
struct Parity {
    gpu: usize,
    matched: usize,
    exact_orientation: usize,
    exact_descriptor: usize,
    distance: f64,
    orientation_errors: Vec<f32>,
}

/// Match each GPU keypoint to a CPU keypoint at the same octave within
/// 0.5 px, 5% in scale and 0.05 rad in orientation; count the pairs whose
/// orientation bits and descriptor bytes are identical and sum the u8
/// descriptor L2 distance.
fn parity(
    gpu: &ScaleSpaceFeatures<U8Descriptor>,
    cpu: &ScaleSpaceFeatures<U8Descriptor>,
    out: &mut Parity,
) {
    let cell = 4.0f32;
    let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
    for (index, keypoint) in cpu.keypoints().iter().enumerate() {
        let key = ((keypoint.x / cell) as i32, (keypoint.y / cell) as i32);
        grid.entry(key).or_default().push(index);
    }
    out.gpu += gpu.keypoints().len();
    for (keypoint, descriptor) in gpu.keypoints().iter().zip(gpu.descriptors()) {
        let (cx, cy) = ((keypoint.x / cell) as i32, (keypoint.y / cell) as i32);
        let mut best: Option<(f32, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &index in grid.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    let other = &cpu.keypoints()[index];
                    let d2 = (other.x - keypoint.x).powi(2) + (other.y - keypoint.y).powi(2);
                    let mut angle = (other.orientation - keypoint.orientation).abs();
                    angle = angle.min(std::f32::consts::TAU - angle);
                    if other.octave == keypoint.octave
                        && d2 < 0.25
                        && (other.sigma / keypoint.sigma - 1.0).abs() < 0.05
                        && angle < 0.05
                        && best.is_none_or(|(best_d2, _)| d2 < best_d2)
                    {
                        best = Some((d2, index));
                    }
                }
            }
        }
        if let Some((_, index)) = best {
            out.matched += 1;
            let mut angle = (cpu.keypoints()[index].orientation - keypoint.orientation).abs();
            angle = angle.min(std::f32::consts::TAU - angle);
            out.orientation_errors.push(angle);
            if cpu.keypoints()[index].orientation.to_bits() == keypoint.orientation.to_bits() {
                out.exact_orientation += 1;
            }
            if cpu.descriptors()[index].as_array() == descriptor.as_array() {
                out.exact_descriptor += 1;
            }
            let other = cpu.descriptors()[index].as_array();
            let d: f64 = descriptor
                .as_array()
                .iter()
                .zip(other)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            out.distance += d.sqrt();
        }
    }
}

fn main() {
    let Ok(directory) = std::env::var("OMVG_SIFT_BENCH_DIR") else {
        eprintln!("set OMVG_SIFT_BENCH_DIR to a folder of 8-bit PGM images; skipping");
        return;
    };
    let preset = match std::env::var("OMVG_SIFT_BENCH_PRESET").as_deref() {
        Ok("high") => SiftPreset::High,
        Ok("ultra") => SiftPreset::Ultra,
        _ => SiftPreset::Normal,
    };
    let rounds = env_usize("OMVG_SIFT_BENCH_ROUNDS", 5);
    let cpu_rounds = env_usize("OMVG_SIFT_BENCH_CPU_ROUNDS", 1);
    let skip_cpu = std::env::var("OMVG_SIFT_BENCH_SKIP_CPU").is_ok_and(|value| value == "1");
    let dump = std::env::var("OMVG_SIFT_BENCH_DUMP")
        .ok()
        .map(PathBuf::from);
    let reference = std::env::var("OMVG_SIFT_BENCH_REF").ok().map(PathBuf::from);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)
        .expect("readable image folder")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "pgm"))
        .collect();
    paths.sort();
    let images: Vec<GrayImage> = paths.iter().map(|path| read_pgm(path)).collect();
    let (rows, cols) = images.first().map_or((0, 0), |image| image.shape());
    println!(
        "{} images from {directory}, first {cols}x{rows}, preset {preset:?}, {} Rayon threads",
        images.len(),
        rayon::current_num_threads()
    );

    if std::env::var("OMVG_SIFT_BENCH_MEMORY").is_ok_and(|value| value == "1") {
        if let Some(image) = images.first() {
            memory_sweep(image, preset);
        }
        return;
    }
    let gpu = match SiftGpuExtractor::with_preset(preset) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("GPU extractor unavailable: {error}; skipping");
            return;
        }
    };
    // Warm-up: pipelines, scratch growth and driver clocks.
    for image in images.iter().take(3) {
        gpu.extract_u8(image).expect("GPU extraction");
    }
    let mut gpu_times = Vec::new();
    let mut gpu_features = Vec::new();
    for round in 0..rounds {
        for image in &images {
            let start = Instant::now();
            let features = gpu.extract_u8(image).expect("GPU extraction");
            gpu_times.push(ms(start));
            if round == 0 {
                gpu_features.push(features);
            }
        }
    }
    // The stage profiler averages over `extract_u8` calls; keep detection out of it.
    let profiling = std::env::var_os("OMVG_SIFT_GPU_PROFILE").is_some();
    let gpu_detect_start = Instant::now();
    for image in images.iter().filter(|_| !profiling) {
        gpu.detect_keypoints(image).expect("GPU detection");
    }
    let gpu_detect = ms(gpu_detect_start) / images.len() as f64;

    let mut digest_bytes = Vec::new();
    let mut identical = 0usize;
    let mut diff = Diff::default();
    for (index, features) in gpu_features.iter().enumerate() {
        let bytes = serialize(features);
        if let Some(dump) = &dump {
            std::fs::create_dir_all(dump).expect("dump folder");
            std::fs::write(dump.join(format!("{index:04}.bin")), &bytes).expect("dump write");
        }
        if let Some(reference) = &reference {
            let expected =
                std::fs::read(reference.join(format!("{index:04}.bin"))).expect("reference dump");
            if expected == bytes {
                identical += 1;
            } else {
                diff.add(&bytes, &expected);
            }
        }
        digest_bytes.extend_from_slice(&fnv1a(&bytes).to_le_bytes());
    }
    let features_per_image = gpu_features
        .iter()
        .map(|features| features.keypoints().len())
        .sum::<usize>() as f64
        / gpu_features.len().max(1) as f64;
    let gpu_mean = mean(&gpu_times);
    let gpu_median = median(&mut gpu_times.clone());
    println!(
        "GPU extract_u8:      median {gpu_median:8.2} ms  mean {gpu_mean:8.2} ms  ({:.1} img/s)",
        1e3 / gpu_mean
    );
    println!("GPU detect_keypoints: mean {gpu_detect:8.2} ms");
    println!(
        "GPU output: {features_per_image:.1} features/image, digest {:016x}",
        fnv1a(&digest_bytes)
    );
    if reference.is_some() {
        println!(
            "GPU output identical to reference on {identical}/{} images",
            gpu_features.len()
        );
        if identical != gpu_features.len() {
            diff.report();
        }
    }
    if skip_cpu {
        return;
    }

    let cpu = SiftExtractor::with_preset(preset);
    let mut cpu_times = Vec::new();
    let mut cpu_features = Vec::new();
    for round in 0..cpu_rounds {
        for image in &images {
            let start = Instant::now();
            let features = cpu.extract_u8(image).expect("CPU extraction");
            cpu_times.push(ms(start));
            if round == 0 {
                cpu_features.push(features);
            }
        }
    }
    let batch_start = Instant::now();
    let batch: Vec<usize> = images
        .par_iter()
        .map(|image| {
            cpu.extract_u8(image)
                .expect("CPU extraction")
                .keypoints()
                .len()
        })
        .collect();
    let cpu_batch = ms(batch_start) / images.len() as f64;
    let cpu_mean = mean(&cpu_times);
    let cpu_median = median(&mut cpu_times.clone());
    println!(
        "CPU extract_u8:      median {cpu_median:8.2} ms  mean {cpu_mean:8.2} ms  ({:.1} img/s)",
        1e3 / cpu_mean
    );
    println!(
        "CPU batch (1 image per worker): {cpu_batch:8.2} ms/image ({:.1} img/s), {} images",
        1e3 / cpu_batch,
        batch.len()
    );
    println!(
        "Speed-up: {:.1}x over CPU latency, {:.1}x over CPU batch throughput",
        cpu_mean / gpu_mean,
        cpu_batch / gpu_mean
    );

    let cpu_per_image = cpu_features
        .iter()
        .map(|features| features.keypoints().len())
        .sum::<usize>() as f64
        / cpu_features.len().max(1) as f64;
    let mut stats = Parity::default();
    for (gpu, cpu) in gpu_features.iter().zip(&cpu_features) {
        parity(gpu, cpu, &mut stats);
    }
    let matched = stats.matched.max(1) as f64;
    println!(
        "Parity: CPU {cpu_per_image:.1} vs GPU {features_per_image:.1} features/image; \
         {:.3}% of GPU keypoints match a CPU keypoint; of those {:.2}% have bit-identical \
         orientation and {:.2}% identical descriptor bytes; mean u8 descriptor L2 {:.3} (of 512)",
        100.0 * stats.matched as f64 / stats.gpu.max(1) as f64,
        100.0 * stats.exact_orientation as f64 / matched,
        100.0 * stats.exact_descriptor as f64 / matched,
        stats.distance / matched
    );
    let mut errors = stats.orientation_errors;
    errors.sort_by(f32::total_cmp);
    let quantile = |q: f64| {
        errors
            .get(((errors.len() as f64 - 1.0) * q).round() as usize)
            .copied()
            .unwrap_or(f32::NAN)
    };
    println!(
        "Orientation error vs CPU (rad): mean {:.2e}, median {:.2e}, p99 {:.2e}, p99.9 {:.2e}, max {:.2e}",
        errors.iter().map(|&e| f64::from(e)).sum::<f64>() / errors.len().max(1) as f64,
        quantile(0.5),
        quantile(0.99),
        quantile(0.999),
        quantile(1.0)
    );
}
