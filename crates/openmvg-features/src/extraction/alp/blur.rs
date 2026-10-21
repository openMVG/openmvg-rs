// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streaming four-level Gaussian blur for the ALP scale space.
//!
//! ALP images are column-major `nalgebra` matrices, so one image column is a
//! contiguous "line". The detector streams over those lines: every call to
//! [`ColumnBlur::blur_column`] produces one output column of all four Gaussian
//! levels from the octave base image, without materializing any full-size
//! level. This is the column-major analogue of a row-streaming detector.
//!
//! Each output column is computed in two separable passes:
//!
//! 1. **Across lines** (horizontal taps). All four levels read the same base
//!    columns, so the symmetric pair `base[j - t] + base[j + t]` is formed once
//!    and shared by every level whose radius reaches `t`. Rows are processed
//!    in register blocks of [`ACROSS_BLOCK`] samples with one accumulator array
//!    per level, and the taps are split into four phases at the (monotone)
//!    level radii so every phase is straight-line code.
//! 2. **Along the line** (vertical taps) per level, from a clamp-padded copy of
//!    the horizontal result, again register-blocked.
//!
//! The kernels and the clamped boundary are those of the shared
//! `GaussianFilter`; only the order of the two passes differs from a
//! full-image separable convolution, which changes results by rounding only.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
use crate::math_compat::expf;

use super::tables::SIGMAS;

/// Number of Gaussian levels per octave.
pub(super) const LEVELS: usize = 4;

/// Largest half-kernel radius used by any ALP level (`ceil(3 * 4.498)`).
pub(super) const MAX_RADIUS: usize = 14;

/// Rows per register block in the horizontal pass (four levels share it).
const ACROSS_BLOCK: usize = 8;

/// Rows per register block in the vertical pass (one level at a time).
const ALONG_BLOCK: usize = 16;

/// Symmetric half kernel: `taps[0]` is the center weight and `taps[t]` the
/// weight of both samples at distance `t`, for `t <= radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HalfKernel {
    pub(super) radius: usize,
    pub(super) taps: [f32; MAX_RADIUS + 1],
}

impl HalfKernel {
    /// Sampled Gaussian with radius `ceil(3 sigma)`, normalized by the sum of
    /// the full kernel in index order. This is bit-for-bit the half kernel of
    /// the shared `GaussianFilter`.
    pub(super) fn gaussian(sigma: f32) -> Self {
        let mut taps = [0.0f32; MAX_RADIUS + 1];
        if sigma < f32::EPSILON {
            taps[0] = 1.0;
            return Self { radius: 0, taps };
        }
        // ALP's largest incremental sigma (4.498) needs radius 14; the bound
        // only matters for out-of-contract inputs and keeps indexing in range.
        let radius = ((3.0 * sigma).ceil() as usize).min(MAX_RADIUS);
        let mut full = [0.0f32; 2 * MAX_RADIUS + 1];
        let mut sum = 0.0f32;
        for (index, value) in full.iter_mut().enumerate().take(2 * radius + 1) {
            let distance = (index as f32 - radius as f32) / sigma;
            *value = expf(-0.5 * distance * distance);
            sum += *value;
        }
        for (offset, tap) in taps.iter_mut().enumerate().take(radius + 1) {
            *tap = full[radius + offset] / sum;
        }
        Self { radius, taps }
    }
}

/// Incremental kernels that take an octave base to the four [`SIGMAS`] levels.
///
/// Octave 0 assumes a base blur of 0.5 px; later octaves start from the 2x
/// decimated `SIGMAS[2]` level, i.e. a base blur of `SIGMAS[2] / 2`.
pub(super) fn octave_kernels(octave: usize) -> [HalfKernel; LEVELS] {
    let base_sigma = if octave == 0 { 0.5 } else { SIGMAS[2] / 2.0 };
    core::array::from_fn(|level| {
        let target_sigma = SIGMAS[level];
        let sigma_sq_diff = (target_sigma * target_sigma - base_sigma * base_sigma).max(0.0);
        HalfKernel::gaussian(sigma_sq_diff.sqrt().max(0.1))
    })
}

/// Borrowed column-major octave base image.
#[derive(Clone, Copy)]
pub(super) struct BaseImage<'a> {
    pub(super) pixels: &'a [f32],
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl BaseImage<'_> {
    #[inline]
    fn column(&self, column: usize) -> &[f32] {
        &self.pixels[column * self.rows..(column + 1) * self.rows]
    }
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    static COLUMN_BLUR_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "std"))]
pub(super) fn reset_column_blur_calls() {
    COLUMN_BLUR_CALLS.with(|calls| calls.set(0));
}

#[cfg(all(test, feature = "std"))]
pub(super) fn column_blur_calls() -> usize {
    COLUMN_BLUR_CALLS.with(std::cell::Cell::get)
}

/// Reusable scratch for streaming column blurs; sized once per detection.
pub(super) struct ColumnBlur {
    /// Horizontal-pass result per level, with `MAX_RADIUS` clamp padding on
    /// both ends so the vertical pass needs no boundary branches.
    across: [Vec<f32>; LEVELS],
}

impl ColumnBlur {
    /// Scratch for images with at most `max_rows` rows.
    pub(super) fn new(max_rows: usize) -> Self {
        Self {
            across: core::array::from_fn(|_| vec![0.0; max_rows + 2 * MAX_RADIUS]),
        }
    }

    /// Blur column `column` of `base` at every level selected by `levels`.
    ///
    /// Level `l` is written to `out[l][offset..offset + base.rows]`; levels
    /// whose mask bit is clear are left untouched. The horizontal pass shares
    /// tap pairs among the selected levels.
    pub(super) fn blur_column(
        &mut self,
        base: BaseImage<'_>,
        column: usize,
        kernels: &[HalfKernel; LEVELS],
        levels: [bool; LEVELS],
        out: &mut [Vec<f32>; LEVELS],
        offset: usize,
    ) {
        let rows = base.rows;
        let mut used = levels.into_iter();
        let out = out.each_mut().map(|line| {
            used.next()
                .unwrap_or(false)
                .then(|| &mut line[offset..offset + rows])
        });
        self.blur_column_into(base, column, kernels, out);
    }

    /// Blur column `column` of `base` into `out[l]` (`base.rows` samples) at
    /// every level `l` with an output line; the other levels are skipped.
    pub(super) fn blur_column_into(
        &mut self,
        base: BaseImage<'_>,
        column: usize,
        kernels: &[HalfKernel; LEVELS],
        out: [Option<&mut [f32]>; LEVELS],
    ) {
        #[cfg(all(test, feature = "std"))]
        COLUMN_BLUR_CALLS.with(|calls| calls.set(calls.get() + 1));
        let rows = base.rows;
        if rows == 0 || base.cols == 0 {
            return;
        }
        if out.iter().all(Option::is_some) {
            self.blur_across(base, column, kernels);
        } else {
            let mut active = [0usize; LEVELS];
            let mut count = 0;
            for (level, line) in out.iter().enumerate() {
                if line.is_some() {
                    active[count] = level;
                    count += 1;
                }
            }
            match count {
                0 => return,
                1 => self.blur_across_selected(base, column, kernels, [active[0]]),
                2 => self.blur_across_selected(base, column, kernels, [active[0], active[1]]),
                3 => self.blur_across_selected(
                    base,
                    column,
                    kernels,
                    [active[0], active[1], active[2]],
                ),
                _ => self.blur_across(base, column, kernels),
            }
        }
        for ((kernel, across), line) in kernels.iter().zip(&mut self.across).zip(out) {
            let Some(line) = line else {
                continue;
            };
            let padded = &mut across[..rows + 2 * MAX_RADIUS];
            let first = padded[MAX_RADIUS];
            let last = padded[MAX_RADIUS + rows - 1];
            padded[..MAX_RADIUS].fill(first);
            padded[MAX_RADIUS + rows..].fill(last);
            blur_along(padded, rows, kernel, line);
        }
    }

    /// Fused horizontal taps for one, two, or three ascending active levels.
    fn blur_across_selected<const COUNT: usize>(
        &mut self,
        base: BaseImage<'_>,
        column: usize,
        kernels: &[HalfKernel; LEVELS],
        active: [usize; COUNT],
    ) {
        let rows = base.rows;
        let last_column = base.cols - 1;
        let center = base.column(column);
        let lower: [&[f32]; MAX_RADIUS + 1] =
            core::array::from_fn(|tap| base.column(column.saturating_sub(tap)));
        let upper: [&[f32]; MAX_RADIUS + 1] =
            core::array::from_fn(|tap| base.column((column + tap).min(last_column)));
        let selected = active.map(|level| kernels[level]);
        let radii = selected.map(|kernel| kernel.radius);

        // Preserve the original small-line scalar arithmetic.
        if rows < ACROSS_BLOCK {
            for row in 0..rows {
                for (&level, kernel) in active.iter().zip(&selected) {
                    let mut value = center[row] * kernel.taps[0];
                    for tap in 1..=kernel.radius {
                        value += (lower[tap][row] + upper[tap][row]) * kernel.taps[tap];
                    }
                    self.across[level][MAX_RADIUS + row] = value;
                }
            }
            return;
        }

        let mut start = 0;
        loop {
            let row = start.min(rows - ACROSS_BLOCK);
            let c = block(center, row);
            let mut acc: [[f32; ACROSS_BLOCK]; COUNT] =
                core::array::from_fn(|slot| scaled(&c, selected[slot].taps[0]));
            for tap in 1..=radii[0] {
                let pair = tap_pair(lower[tap], upper[tap], row);
                for (acc, kernel) in acc.iter_mut().zip(&selected) {
                    accumulate(acc, &pair, kernel.taps[tap]);
                }
            }
            if COUNT > 1 {
                for tap in radii[0] + 1..=radii[1] {
                    let pair = tap_pair(lower[tap], upper[tap], row);
                    for slot in 1..COUNT {
                        accumulate(&mut acc[slot], &pair, selected[slot].taps[tap]);
                    }
                }
            }
            if COUNT > 2 {
                for tap in radii[1] + 1..=radii[2] {
                    let pair = tap_pair(lower[tap], upper[tap], row);
                    accumulate(&mut acc[2], &pair, selected[2].taps[tap]);
                }
            }
            let at = MAX_RADIUS + row;
            for (&level, acc) in active.iter().zip(&acc) {
                self.across[level][at..at + ACROSS_BLOCK].copy_from_slice(acc);
            }
            start += ACROSS_BLOCK;
            if start >= rows {
                break;
            }
        }
    }

    /// Horizontal taps for all four levels into the padded scratch lines.
    fn blur_across(&mut self, base: BaseImage<'_>, column: usize, kernels: &[HalfKernel; LEVELS]) {
        let rows = base.rows;
        let last_column = base.cols - 1;
        let center = base.column(column);
        // Clamp-to-edge column indices for every tap distance.
        let lower: [&[f32]; MAX_RADIUS + 1] =
            core::array::from_fn(|tap| base.column(column.saturating_sub(tap)));
        let upper: [&[f32]; MAX_RADIUS + 1] =
            core::array::from_fn(|tap| base.column((column + tap).min(last_column)));
        let radii = kernels.map(|kernel| kernel.radius);
        debug_assert!(radii.windows(2).all(|pair| pair[0] <= pair[1]));
        let [k0, k1, k2, k3] = kernels.map(|kernel| kernel.taps);
        let [out0, out1, out2, out3] = &mut self.across;

        if rows < ACROSS_BLOCK {
            for row in 0..rows {
                for (level, out) in [&mut *out0, &mut *out1, &mut *out2, &mut *out3]
                    .into_iter()
                    .enumerate()
                {
                    let kernel = &kernels[level];
                    let mut value = center[row] * kernel.taps[0];
                    for tap in 1..=kernel.radius {
                        value += (lower[tap][row] + upper[tap][row]) * kernel.taps[tap];
                    }
                    out[MAX_RADIUS + row] = value;
                }
            }
            return;
        }

        let mut start = 0;
        loop {
            // The final block overlaps its predecessor when `rows` is not a
            // multiple of the block; overlapping rows are recomputed exactly.
            let row = start.min(rows - ACROSS_BLOCK);
            let c = block(center, row);
            let mut acc0 = scaled(&c, k0[0]);
            let mut acc1 = scaled(&c, k1[0]);
            let mut acc2 = scaled(&c, k2[0]);
            let mut acc3 = scaled(&c, k3[0]);
            for tap in 1..=radii[0] {
                let pair = tap_pair(lower[tap], upper[tap], row);
                accumulate(&mut acc0, &pair, k0[tap]);
                accumulate(&mut acc1, &pair, k1[tap]);
                accumulate(&mut acc2, &pair, k2[tap]);
                accumulate(&mut acc3, &pair, k3[tap]);
            }
            for tap in radii[0] + 1..=radii[1] {
                let pair = tap_pair(lower[tap], upper[tap], row);
                accumulate(&mut acc1, &pair, k1[tap]);
                accumulate(&mut acc2, &pair, k2[tap]);
                accumulate(&mut acc3, &pair, k3[tap]);
            }
            for tap in radii[1] + 1..=radii[2] {
                let pair = tap_pair(lower[tap], upper[tap], row);
                accumulate(&mut acc2, &pair, k2[tap]);
                accumulate(&mut acc3, &pair, k3[tap]);
            }
            for tap in radii[2] + 1..=radii[3] {
                let pair = tap_pair(lower[tap], upper[tap], row);
                accumulate(&mut acc3, &pair, k3[tap]);
            }
            let at = MAX_RADIUS + row;
            out0[at..at + ACROSS_BLOCK].copy_from_slice(&acc0);
            out1[at..at + ACROSS_BLOCK].copy_from_slice(&acc1);
            out2[at..at + ACROSS_BLOCK].copy_from_slice(&acc2);
            out3[at..at + ACROSS_BLOCK].copy_from_slice(&acc3);
            start += ACROSS_BLOCK;
            if start >= rows {
                break;
            }
        }
    }
}

#[inline(always)]
fn block(line: &[f32], row: usize) -> [f32; ACROSS_BLOCK] {
    let mut values = [0.0f32; ACROSS_BLOCK];
    values.copy_from_slice(&line[row..row + ACROSS_BLOCK]);
    values
}

#[inline(always)]
fn tap_pair(lower: &[f32], upper: &[f32], row: usize) -> [f32; ACROSS_BLOCK] {
    let lower = block(lower, row);
    let upper = block(upper, row);
    core::array::from_fn(|index| lower[index] + upper[index])
}

#[inline(always)]
fn scaled(values: &[f32; ACROSS_BLOCK], weight: f32) -> [f32; ACROSS_BLOCK] {
    core::array::from_fn(|index| values[index] * weight)
}

#[inline(always)]
fn accumulate(acc: &mut [f32; ACROSS_BLOCK], pair: &[f32; ACROSS_BLOCK], weight: f32) {
    for (value, &sample) in acc.iter_mut().zip(pair) {
        *value += sample * weight;
    }
}

/// Vertical symmetric taps of one level from a clamp-padded line.
///
/// `padded[MAX_RADIUS + row]` holds input row `row`; both pads replicate the
/// edge samples, which is exactly the clamp boundary.
// Indexed loops over fixed-size register blocks are the form LLVM vectorizes
// reliably here; iterator chains over shifted windows defeat it.
#[allow(clippy::needless_range_loop)]
fn blur_along(padded: &[f32], rows: usize, kernel: &HalfKernel, out: &mut [f32]) {
    let radius = kernel.radius;
    let taps = &kernel.taps;
    if rows < ALONG_BLOCK {
        for row in 0..rows {
            let at = MAX_RADIUS + row;
            let mut value = padded[at] * taps[0];
            for tap in 1..=radius {
                value += (padded[at - tap] + padded[at + tap]) * taps[tap];
            }
            out[row] = value;
        }
        return;
    }
    let mut start = 0;
    loop {
        let row = start.min(rows - ALONG_BLOCK);
        let at = MAX_RADIUS + row;
        let center = &padded[at..at + ALONG_BLOCK];
        let mut acc = [0.0f32; ALONG_BLOCK];
        for index in 0..ALONG_BLOCK {
            acc[index] = center[index] * taps[0];
        }
        for tap in 1..=radius {
            let lower = &padded[at - tap..at - tap + ALONG_BLOCK];
            let upper = &padded[at + tap..at + tap + ALONG_BLOCK];
            let weight = taps[tap];
            for index in 0..ALONG_BLOCK {
                acc[index] += (lower[index] + upper[index]) * weight;
            }
        }
        out[row..row + ALONG_BLOCK].copy_from_slice(&acc);
        start += ALONG_BLOCK;
        if start >= rows {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::convolution::GaussianFilter;
    use nalgebra::DMatrix;

    #[test]
    fn half_kernel_matches_shared_gaussian_filter_bits() {
        for octave in 0..3 {
            let base_sigma: f32 = if octave == 0 { 0.5 } else { SIGMAS[2] / 2.0 };
            for (level, kernel) in octave_kernels(octave).iter().enumerate() {
                let sigma = (SIGMAS[level] * SIGMAS[level] - base_sigma * base_sigma)
                    .max(0.0)
                    .sqrt()
                    .max(0.1);
                let reference = GaussianFilter::new(sigma);
                let reference = reference.kernel_half();
                assert_eq!(kernel.radius + 1, reference.len());
                for (actual, expected) in kernel.taps.iter().zip(reference) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn kernel_radii_are_monotone_for_every_octave_kind() {
        for octave in [0, 1] {
            let radii = octave_kernels(octave).map(|kernel| kernel.radius);
            assert!(radii.windows(2).all(|pair| pair[0] <= pair[1]), "{radii:?}");
            assert!(radii[3] <= MAX_RADIUS);
        }
        assert_eq!(
            octave_kernels(0).map(|kernel| kernel.radius),
            [5, 7, 10, 14]
        );
        assert_eq!(octave_kernels(1).map(|kernel| kernel.radius), [1, 5, 9, 13]);
    }

    fn streamed_levels(image: &DMatrix<f32>, octave: usize) -> [DMatrix<f32>; LEVELS] {
        let (rows, cols) = image.shape();
        let kernels = octave_kernels(octave);
        let base = BaseImage {
            pixels: image.as_slice(),
            rows,
            cols,
        };
        let mut blur = ColumnBlur::new(rows);
        let mut out: [Vec<f32>; LEVELS] = core::array::from_fn(|_| vec![0.0; rows * cols]);
        for column in 0..cols {
            blur.blur_column(
                base,
                column,
                &kernels,
                [true; LEVELS],
                &mut out,
                column * rows,
            );
        }
        out.map(|pixels| DMatrix::from_vec(rows, cols, pixels))
    }

    #[test]
    fn streamed_columns_match_full_image_separable_blur() {
        for (rows, cols) in [(37, 29), (5, 9), (64, 3), (16, 16)] {
            let image = DMatrix::from_fn(rows, cols, |row, col| {
                ((row * 37 + col * 17 + (row * col) % 23) % 256) as f32
            });
            for octave in [0, 1] {
                let streamed = streamed_levels(&image, octave);
                let base_sigma: f32 = if octave == 0 { 0.5 } else { SIGMAS[2] / 2.0 };
                for (level, actual) in streamed.iter().enumerate() {
                    let sigma = (SIGMAS[level] * SIGMAS[level] - base_sigma * base_sigma)
                        .max(0.0)
                        .sqrt()
                        .max(0.1);
                    let expected = GaussianFilter::new(sigma).convolve_separable(&image);
                    for (a, e) in actual.iter().zip(expected.iter()) {
                        assert!(
                            (a - e).abs() <= 1e-4 * e.abs().max(1.0),
                            "{rows}x{cols} octave {octave} level {level}: {a} vs {e}"
                        );
                    }
                }
            }
        }
    }

    fn check_selected_bits(
        blur: &mut ColumnBlur,
        rows: usize,
        cols: usize,
        column: usize,
        kernels: &[HalfKernel; LEVELS],
        mask: u8,
        seed: usize,
    ) {
        let image = DMatrix::from_fn(rows, cols, |row, col| {
            if (row + col + seed).is_multiple_of(7) {
                0.0
            } else {
                ((row * 37 + col * 17 + row * col + seed * 11) % 251) as f32 - 117.0
            }
        });
        let base = BaseImage {
            pixels: image.as_slice(),
            rows,
            cols,
        };
        let offset = 5;
        let sentinel = -12345.25f32;
        let mut oracle: [Vec<f32>; LEVELS] =
            core::array::from_fn(|_| vec![sentinel; offset + rows + 3]);
        ColumnBlur::new(rows).blur_column(
            base,
            column,
            kernels,
            [true; LEVELS],
            &mut oracle,
            offset,
        );
        let mut actual: [Vec<f32>; LEVELS] =
            core::array::from_fn(|_| vec![sentinel; offset + rows + 3]);
        let levels = core::array::from_fn(|level| mask & (1 << level) != 0);
        blur.blur_column(base, column, kernels, levels, &mut actual, offset);
        for level in 0..LEVELS {
            for (index, value) in actual[level].iter().enumerate() {
                let expected = if levels[level] && (offset..offset + rows).contains(&index) {
                    oracle[level][index]
                } else {
                    sentinel
                };
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "rows={rows} cols={cols} column={column} mask={mask:04b} level={level} index={index}"
                );
            }
        }
    }

    #[test]
    fn selected_horizontal_levels_match_all_four_bits_for_every_mask() {
        for octave in [0, 1] {
            let kernels = octave_kernels(octave);
            for rows in [0, 1, 7, 8, 9, 15, 16, 17, 37] {
                for cols in [1, 3, 9] {
                    for column in [0, cols - 1] {
                        for mask in 0..16 {
                            check_selected_bits(
                                &mut ColumnBlur::new(rows),
                                rows,
                                cols,
                                column,
                                &kernels,
                                mask,
                                0,
                            );
                        }
                    }
                }
            }
            check_selected_bits(&mut ColumnBlur::new(37), 37, 0, 0, &kernels, 15, 0);
        }
    }

    #[test]
    fn selected_horizontal_reuse_replaces_stale_interiors_and_padding() {
        let masks = [0, 4, 15, 10, 7];
        let row_counts = [37, 1, 17, 7, 16, 9, 0, 37, 15, 8];
        for octave in [0, 1] {
            let kernels = octave_kernels(octave);
            let mut blur = ColumnBlur::new(37);
            for (step, rows) in row_counts.into_iter().enumerate() {
                check_selected_bits(
                    &mut blur,
                    rows,
                    9,
                    if step % 2 == 0 { 0 } else { 8 },
                    &kernels,
                    masks[step % masks.len()],
                    step + 1,
                );
            }
        }
    }

    #[test]
    fn selected_horizontal_handles_equal_and_zero_radii() {
        for sigmas in [[0.0, 0.0, 0.1, 0.1], [0.0, 0.1, 0.1, 1.0]] {
            let kernels = sigmas.map(HalfKernel::gaussian);
            for rows in [1, 7, 8, 9, 17] {
                for mask in 0..16 {
                    check_selected_bits(&mut ColumnBlur::new(rows), rows, 3, 0, &kernels, mask, 0);
                }
            }
        }
    }

    #[test]
    fn unselected_levels_are_left_untouched() {
        let image = DMatrix::from_fn(20, 12, |row, col| (row * 3 + col) as f32);
        let kernels = octave_kernels(0);
        let base = BaseImage {
            pixels: image.as_slice(),
            rows: 20,
            cols: 12,
        };
        let mut blur = ColumnBlur::new(20);
        let mut out: [Vec<f32>; LEVELS] = core::array::from_fn(|_| vec![-7.0; 20]);
        blur.blur_column(base, 4, &kernels, [false, true, false, true], &mut out, 0);
        assert!(out[0].iter().all(|&value| value == -7.0));
        assert!(out[2].iter().all(|&value| value == -7.0));
        assert!(out[1].iter().all(|&value| value != -7.0));
        assert!(out[3].iter().all(|&value| value != -7.0));
    }
}
