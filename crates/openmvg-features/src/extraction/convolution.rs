// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared separable Gaussian convolution for the ALP and SIFT scale spaces.
//!
//! Images are column-major, so the first pass runs along each contiguous column and the second
//! combines corresponding samples of neighbouring columns; both stay cache-friendly without a
//! transpose. Samples outside the image repeat the nearest edge sample.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use nalgebra::DMatrix;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
use crate::math_compat::expf;

/// Shared Gaussian wrapper used by ALP and classic SIFT scale spaces.
pub(crate) struct GaussianFilter {
    kernel_half: Vec<f32>,
}

impl GaussianFilter {
    pub(crate) fn new(sigma: f32) -> Self {
        if sigma < f32::EPSILON {
            return Self {
                kernel_half: vec![1.0],
            };
        }

        let radius = (3.0 * sigma).ceil() as usize;
        let mut full = vec![0.0f32; 2 * radius + 1];
        let mut sum = 0.0f32;
        for (index, value) in full.iter_mut().enumerate() {
            let distance = (index as f32 - radius as f32) / sigma;
            *value = expf(-0.5 * distance * distance);
            sum += *value;
        }
        let kernel_half = (0..=radius)
            .map(|offset| full[radius + offset] / sum)
            .collect();
        Self { kernel_half }
    }

    pub(crate) fn convolve_separable(&self, src: &DMatrix<f32>) -> DMatrix<f32> {
        let (rows, cols) = src.shape();
        let mut temp = DMatrix::zeros(rows, cols);
        let mut dst = DMatrix::zeros(rows, cols);
        convolve_separable_into(
            src.as_slice(),
            rows,
            cols,
            &self.kernel_half,
            temp.as_mut_slice(),
            dst.as_mut_slice(),
        );
        dst
    }

    #[cfg(test)]
    pub(crate) fn kernel_half(&self) -> &[f32] {
        &self.kernel_half
    }
}

/// Convolve a column-major image of `rows` x `cols` with the symmetric kernel `kernel_half` (the
/// center tap followed by the positive half) into caller-owned scratch and output buffers.
pub(crate) fn convolve_separable_into(
    src: &[f32],
    rows: usize,
    cols: usize,
    kernel_half: &[f32],
    temp: &mut [f32],
    dst: &mut [f32],
) {
    assert!(!kernel_half.is_empty());
    let len = rows.saturating_mul(cols);
    assert_eq!(src.len(), len);
    assert_eq!(temp.len(), len);
    assert_eq!(dst.len(), len);
    if len == 0 {
        return;
    }

    // Columns are the contiguous lines.
    let (line_len, line_count) = (rows, cols);
    let radius = kernel_half.len() - 1;

    #[cfg(feature = "parallel")]
    let use_parallel = len >= 32 * 1024 && rayon::current_num_threads() > 1;
    #[cfg(not(feature = "parallel"))]
    let use_parallel = false;

    #[cfg(feature = "parallel")]
    if use_parallel {
        use rayon::prelude::*;
        temp.par_chunks_mut(line_len)
            .with_min_len(8)
            .enumerate()
            .for_each_init(
                || vec![0.0f32; line_len + 2 * radius],
                |padded, (line, output)| {
                    let source = &src[line * line_len..(line + 1) * line_len];
                    convolve_contiguous_line(source, output, padded, kernel_half);
                },
            );
    }

    if !use_parallel {
        let mut padded = vec![0.0f32; line_len + 2 * radius];
        for (line, output) in temp.chunks_mut(line_len).enumerate() {
            let source = &src[line * line_len..(line + 1) * line_len];
            convolve_contiguous_line(source, output, &mut padded, kernel_half);
        }
    }

    #[cfg(feature = "parallel")]
    if use_parallel {
        use rayon::prelude::*;
        dst.par_chunks_mut(line_len)
            .with_min_len(8)
            .enumerate()
            .for_each(|(line, output)| {
                convolve_across_lines(temp, line_len, line_count, line, output, kernel_half);
            });
    }

    if !use_parallel {
        for (line, output) in dst.chunks_mut(line_len).enumerate() {
            convolve_across_lines(temp, line_len, line_count, line, output, kernel_half);
        }
    }
}

#[inline]
fn convolve_contiguous_line(
    source: &[f32],
    output: &mut [f32],
    padded: &mut [f32],
    weights: &[f32],
) {
    let radius = weights.len() - 1;
    let line_len = source.len();
    for (index, value) in padded[..radius].iter_mut().enumerate() {
        *value = source[clamped_index(index as isize - radius as isize, line_len)];
    }
    padded[radius..radius + line_len].copy_from_slice(source);
    for (offset, value) in padded[radius + line_len..].iter_mut().enumerate() {
        *value = source[clamped_index((line_len + offset) as isize, line_len)];
    }

    output.copy_from_slice(&padded[radius..radius + line_len]);
    for value in output.iter_mut() {
        *value *= weights[0];
    }
    for (offset, &weight) in weights.iter().enumerate().skip(1) {
        let lower = &padded[radius - offset..radius - offset + line_len];
        let upper = &padded[radius + offset..radius + offset + line_len];
        add_symmetric_scaled(output, lower, upper, weight);
    }
}

#[inline]
fn convolve_across_lines(
    source: &[f32],
    line_len: usize,
    line_count: usize,
    line: usize,
    output: &mut [f32],
    weights: &[f32],
) {
    let center = &source[line * line_len..(line + 1) * line_len];
    for (value, &sample) in output.iter_mut().zip(center) {
        *value = sample * weights[0];
    }
    for (offset, &weight) in weights.iter().enumerate().skip(1) {
        let lower = clamped_index(line as isize - offset as isize, line_count);
        let upper = clamped_index(line as isize + offset as isize, line_count);
        let lower = &source[lower * line_len..(lower + 1) * line_len];
        let upper = &source[upper * line_len..(upper + 1) * line_len];
        for ((value, &lower_sample), &upper_sample) in output.iter_mut().zip(lower).zip(upper) {
            *value += (lower_sample + upper_sample) * weight;
        }
    }
}

#[inline]
fn add_symmetric_scaled(output: &mut [f32], lower: &[f32], upper: &[f32], weight: f32) {
    for ((value, &lower_sample), &upper_sample) in output.iter_mut().zip(lower).zip(upper) {
        *value += (lower_sample + upper_sample) * weight;
    }
}

/// The nearest index of `0..len` to `index`.
#[inline]
fn clamped_index(index: isize, len: usize) -> usize {
    index.clamp(0, len as isize - 1) as usize
}
