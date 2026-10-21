// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared SIFT constants and core algorithms used by both the ALP and classic
//! SIFT detectors.
//!
//! All compute-intensive functions are generic over a `get_gradient` closure so
//! callers can supply whatever gradient source they have (precomputed image
//! arrays, continuous-sigma interpolation, etc.) without duplicating the core
//! math.
//!
//! Implementation choices:
//! - Orientation histogram: 36 bins, 6-pass [¼ ½ ¼] circular smoothing, as
//!   an in-place rolling scan (avoids the per-pass array copy).
//! - Descriptor trilinear interpolation: histogram scatters use validity
//!   masks that skip neighboring cells outside the descriptor grid.
//! - Angles: accurate `atan2` for orientation assignment and a bounded
//!   polynomial approximation for the coarser eight-bin descriptor histogram.
//! - RootSIFT normalization is applied before uint8 quantization.

use core::f32::consts::PI;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
use crate::math_compat::{cosf, expf, sinf};

// ============================================================================
// Shared constants
// ============================================================================

/// Number of spatial histogram bins per dimension (4×4 spatial grid).
pub const SIFT_DESCR_WIDTH: usize = 4;

/// Number of orientation histogram bins per spatial cell (8 × 45°).
pub const SIFT_DESCR_HIST_BINS: usize = 8;

/// Total descriptor length (4×4 spatial grid × 8 orientation bins).
pub const SIFT_DESCR_SIZE: usize = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;

/// Gradient magnitude threshold after first L2 normalization (Lowe §6.1).
pub const SIFT_DESCR_MAG_THR: f32 = 0.2;

/// Scale factor for the final float→integer conversion.
pub const SIFT_INT_DESCR_FCTR: f32 = 512.0;

/// Controls the descriptor patch radius: `patch_sigma = DESCRIPTOR_SCALE * σ`.
/// With 4 spatial bins this makes each bin 3σ wide, matching OpenCV / COLMAP.
pub const DESCRIPTOR_SCALE: f32 = 6.0;

/// Number of bins in the orientation histogram (36 × 10° = 360°).
pub const ORI_HIST_BINS: usize = 36;

/// Gaussian window multiplier for orientation: window σ = `ORI_SCALE * σ`.
pub const ORI_SCALE: f32 = 1.5;

/// Number of [¼ ½ ¼] smoothing passes applied to the orientation histogram.
pub const ORI_SMOOTH_PASSES: usize = 6;

/// Secondary-peak ratio threshold: emit a keypoint for every peak ≥ 80 % of
/// the dominant peak.
pub const ORI_PEAK_RATIO: f32 = 0.8;

pub(crate) const TWO_PI: f32 = 2.0 * PI;
pub(crate) const SQRT2: f32 = core::f32::consts::SQRT_2;

// Private short names for the descriptor grid width and histogram bins
const NB_SPLIT2D: usize = SIFT_DESCR_WIDTH;
const NB_SPLIT_ANGLE: usize = SIFT_DESCR_HIST_BINS;

// ============================================================================
// Fast math helpers (vectorization-friendly approximations)
// ============================================================================

/// Fast `atan2(y, x)` polynomial approximation.
///
/// Max error < 0.004 rad (~0.23°).  SIFT's bin width is 45°, so this is more
/// than accurate enough for trilinear interpolation into 8 orientation bins.
///
/// Algorithm: Rajan minimax polynomial on `[0, 1]` with quadrant correction.
#[inline(always)]
pub(crate) fn fast_atan2(y: f32, x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, PI};
    const FRAC_PI_4: f32 = core::f32::consts::FRAC_PI_4;
    let ax = x.abs();
    let ay = y.abs();
    let (mn, mx) = if ax < ay { (ax, ay) } else { (ay, ax) };
    // r ∈ [0, 1]; if both zero, phi = 0 (avoids division by zero).
    let r = if mx > 0.0 { mn / mx } else { 0.0 };
    // Minimax polynomial: atan(r) ≈ r * (π/4 - (r-1)(0.2447 + 0.0663·r))
    let phi = r * (FRAC_PI_4 - (r - 1.0) * (0.2447 + 0.0663 * r));
    // Octant correction: swap π/2 boundary, negate for negative x, negate for negative y.
    let phi = if ay > ax { FRAC_PI_2 - phi } else { phi };
    let phi = if x < 0.0 { PI - phi } else { phi };
    if y < 0.0 { -phi } else { phi }
}

// ============================================================================
// Orientation histogram
// ============================================================================

/// Build a 36-bin gradient-orientation histogram centered at `(cx, cy)` with
/// characteristic scale `sigma`.
///
/// `cx`, `cy` and `sigma` are in the coordinate space used by `get_gradient`
/// (octave space or original image space). `get_gradient` returns
/// `(magnitude, angle_radians)` for an integer pixel at `(row, col)`, or
/// `None` if the position is out of bounds.
///
/// The Gaussian window has σ_w = `ORI_SCALE * sigma` (= 1.5 σ) and radius
/// `3 σ_w`, matching Lowe (IJCV 2004 §5).
pub(crate) fn build_orientation_histogram<F>(
    cx: f32,
    cy: f32,
    sigma: f32,
    get_gradient: F,
) -> [f32; ORI_HIST_BINS]
where
    F: Fn(i32, i32) -> Option<(f32, f32)>,
{
    // Reject degenerate zero-scale keypoints (sigma underflow → inf weight).
    if sigma < 1e-6 {
        return [0.0; ORI_HIST_BINS];
    }
    let window_sigma = ORI_SCALE * sigma;
    let radius = (3.0 * window_sigma).ceil() as i32;
    let inv_2sigma_sq = 1.0 / (2.0 * window_sigma * window_sigma);

    let mut hist = [0.0f32; ORI_HIST_BINS];

    let center_i = cy.round() as i32;
    let center_j = cx.round() as i32;

    for di in -radius..=radius {
        for dj in -radius..=radius {
            let sample_row = center_i + di;
            let sample_col = center_j + dj;
            if let Some((mag, angle_rad)) = get_gradient(sample_row, sample_col) {
                let dy = sample_row as f32 - cy;
                let dx = sample_col as f32 - cx;
                let dist_sq = dx * dx + dy * dy;
                let weight = expf(-dist_sq * inv_2sigma_sq);

                // Map atan2 range [-π, π] → [0, 2π] → bin, with +0.5 center offset.
                let angle_pos = angle_rad.rem_euclid(TWO_PI);
                let bin =
                    ((angle_pos / TWO_PI * ORI_HIST_BINS as f32 + 0.5) as usize) % ORI_HIST_BINS;
                hist[bin] += mag * weight;
            }
        }
    }

    hist
}

/// Apply the [¼ ½ ¼] circular smoothing filter `n_passes` times **in-place**.
///
/// Uses a forward scan with saved first/last values to avoid a per-pass array
/// copy while producing the same result as the copy-based version.
pub(crate) fn smooth_orientation_histogram(hist: &mut [f32; ORI_HIST_BINS], n_passes: usize) {
    for _ in 0..n_passes {
        let first = hist[0];
        let last = hist[ORI_HIST_BINS - 1];
        let mut prev = last;
        for b in 0..ORI_HIST_BINS {
            let next = if b < ORI_HIST_BINS - 1 {
                hist[b + 1]
            } else {
                first
            };
            let smoothed = prev * 0.25 + hist[b] * 0.5 + next * 0.25;
            prev = hist[b];
            hist[b] = smoothed;
        }
    }
}

/// Extract dominant orientation angles (in radians, range `[−π, π]`) from a
/// smoothed orientation histogram.
///
/// Returns one angle per local maximum whose value is ≥ `ORI_PEAK_RATIO`
/// (80 %) of the global maximum.  Sub-bin accuracy is achieved via parabolic
/// interpolation.
pub(crate) fn extract_orientation_peaks(hist: &[f32; ORI_HIST_BINS]) -> Vec<f32> {
    orientation_peaks(hist)
        .map(|(_, orientation)| orientation)
        .collect()
}

/// The angle of the highest peak of [`extract_orientation_peaks`], the first
/// in bin order on a tie; `None` when the histogram has no finite peak.
pub(crate) fn dominant_orientation(hist: &[f32; ORI_HIST_BINS]) -> Option<f32> {
    orientation_peaks(hist)
        .filter(|&(value, orientation)| value.is_finite() && orientation.is_finite())
        .reduce(|best, peak| if peak.0 > best.0 { peak } else { best })
        .map(|(_, orientation)| orientation)
}

/// `(value, angle)` of each peak of [`extract_orientation_peaks`], in bin
/// order.
fn orientation_peaks(hist: &[f32; ORI_HIST_BINS]) -> impl Iterator<Item = (f32, f32)> + '_ {
    let max_mag = hist.iter().copied().fold(0.0f32, f32::max);
    let threshold = ORI_PEAK_RATIO * max_mag;
    // An empty histogram has no peaks.
    let bins = if max_mag > 0.0 {
        0..ORI_HIST_BINS
    } else {
        0..0
    };

    bins.filter_map(move |b| {
        let val = hist[b];
        let prev = hist[(b + ORI_HIST_BINS - 1) % ORI_HIST_BINS];
        let next = hist[(b + 1) % ORI_HIST_BINS];

        if val < threshold || val <= prev || val <= next {
            return None;
        }

        // Parabolic interpolation: δ = (prev − next) / (2(prev − 2·val + next)).
        // Clamp to [-1, 1] to avoid inf/NaN when the denominator is near zero
        // after smoothing (broad flat peak). The strict val > prev && val > next
        // guard makes denom < 0, but float rounding after 6 passes can make it
        // very small. Clamping ensures the interpolated bin stays within its cell.
        let denom = prev - 2.0 * val + next;
        let delta = if denom.abs() < 1e-6 {
            0.0
        } else {
            (0.5 * (prev - next) / denom).clamp(-1.0, 1.0)
        };
        let refined_bin = b as f32 + delta;
        let ori = refined_bin * TWO_PI / ORI_HIST_BINS as f32;
        let orientation = if ori > PI { ori - TWO_PI } else { ori };
        Some((val, orientation))
    })
}

// ============================================================================
// SIFT descriptor
// ============================================================================

/// Accumulate the 128-bin SIFT descriptor histogram for a keypoint.
///
/// Uses branchless trilinear interpolation (validity masks instead of
/// conditional branches) so the compiler can auto-vectorize the accumulation
/// loop.
///
/// `x`, `y` and `sigma` are in the coordinate space of `get_gradient`;
/// `orientation` is the dominant orientation in radians. `get_gradient`
/// returns `(magnitude, angle_radians)` at integer `(row, col)`, or `None` if
/// out of bounds.
///
/// Returns `None` if the resulting histogram has a near-zero L2 norm
/// (degenerate patch, e.g. uniform region).
#[cfg(test)]
fn compute_sift_descriptor_histogram<F>(
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    root_sift: bool,
    get_gradient: F,
) -> Option<[f32; SIFT_DESCR_SIZE]>
where
    F: Fn(i32, i32) -> Option<(f32, f32)>,
{
    let cos_t = cosf(-orientation);
    let sin_t = sinf(-orientation);

    let sigma_scale = DESCRIPTOR_SCALE * sigma;
    let r = (1.0 + 1.0 / NB_SPLIT2D as f32) * sigma_scale;
    let rp = r * SQRT2;
    let bin_scale = 2.0 * sigma_scale / NB_SPLIT2D as f32;
    let inv_bin_scale = 1.0 / bin_scale;
    let half_bins = (NB_SPLIT2D as f32 - 1.0) * 0.5;
    let inv_sigma_sq = 1.0 / (2.0 * sigma_scale * sigma_scale);
    let angle_to_bin = NB_SPLIT_ANGLE as f32 / TWO_PI;

    let x_min = ((x - rp + 0.5).max(1.0)) as i32;
    let y_min = ((y - rp + 0.5).max(1.0)) as i32;
    let x_max = (x + rp + 0.5) as i32;
    let y_max = (y + rp + 0.5) as i32;

    let mut hist = [0.0f32; SIFT_DESCR_SIZE];

    for si in x_min..x_max {
        let x_ref = si as f32 - x;

        for sj in y_min..y_max {
            let y_ref = sj as f32 - y;

            // Rotate to keypoint-aligned frame.
            let xr = cos_t * x_ref - sin_t * y_ref;
            let yr = sin_t * x_ref + cos_t * y_ref;

            if xr.abs().max(yr.abs()) >= r {
                continue;
            }

            let (mag, theta) = match get_gradient(sj, si) {
                Some(g) => g,
                None => continue,
            };

            // Orientation relative to keypoint orientation, mapped to [0, 2π).
            let ori = (theta - orientation).rem_euclid(TWO_PI);

            // Gaussian-weighted magnitude.
            let dist_sq = xr * xr + yr * yr;
            let m = mag * expf(-dist_sq * inv_sigma_sq);

            // Trilinear bin coordinates.
            let alpha = xr * inv_bin_scale + half_bins;
            let beta = yr * inv_bin_scale + half_bins;
            let gamma = ori * angle_to_bin;

            // Skip NaN/inf gamma (can arise from NaN theta on
            // extreme pixel values or if get_gradient produces non-finite angles).
            if !gamma.is_finite() {
                continue;
            }

            let i0 = alpha.floor() as i32;
            let j0 = beta.floor() as i32;

            let alpha_frac = alpha - alpha.floor();
            let beta_frac = beta - beta.floor();
            let gamma_frac = gamma - gamma.floor();

            let i0_valid = i0 >= 0 && i0 < NB_SPLIT2D as i32;
            let i1_valid = (i0 + 1) >= 0 && (i0 + 1) < NB_SPLIT2D as i32;
            let j0_valid = j0 >= 0 && j0 < NB_SPLIT2D as i32;
            let j1_valid = (j0 + 1) >= 0 && (j0 + 1) < NB_SPLIT2D as i32;

            let w_i0 = 1.0 - alpha_frac;
            let w_i1 = alpha_frac;
            let w_j0 = 1.0 - beta_frac;
            let w_j1 = beta_frac;
            let w_k0 = 1.0 - gamma_frac;
            let w_k1 = gamma_frac;

            let k_left = ((gamma as i32 % NB_SPLIT_ANGLE as i32) + NB_SPLIT_ANGLE as i32) as usize
                % NB_SPLIT_ANGLE;
            let k_right = (k_left + 1) % NB_SPLIT_ANGLE;

            // Branchless accumulation: validity masks (0.0 or 1.0) replace
            // conditional branches so the compiler can auto-vectorize.
            let m00 = (i0_valid && j0_valid) as u8 as f32;
            let m01 = (i0_valid && j1_valid) as u8 as f32;
            let m10 = (i1_valid && j0_valid) as u8 as f32;
            let m11 = (i1_valid && j1_valid) as u8 as f32;

            let i0c = i0.clamp(0, NB_SPLIT2D as i32 - 1) as usize;
            let i1c = (i0 + 1).clamp(0, NB_SPLIT2D as i32 - 1) as usize;
            let j0c = j0.clamp(0, NB_SPLIT2D as i32 - 1) as usize;
            let j1c = (j0 + 1).clamp(0, NB_SPLIT2D as i32 - 1) as usize;

            let base00 = i0c * NB_SPLIT2D * NB_SPLIT_ANGLE + j0c * NB_SPLIT_ANGLE;
            let base01 = i0c * NB_SPLIT2D * NB_SPLIT_ANGLE + j1c * NB_SPLIT_ANGLE;
            let base10 = i1c * NB_SPLIT2D * NB_SPLIT_ANGLE + j0c * NB_SPLIT_ANGLE;
            let base11 = i1c * NB_SPLIT2D * NB_SPLIT_ANGLE + j1c * NB_SPLIT_ANGLE;

            let s00 = w_i0 * w_j0 * m * m00;
            let s01 = w_i0 * w_j1 * m * m01;
            let s10 = w_i1 * w_j0 * m * m10;
            let s11 = w_i1 * w_j1 * m * m11;

            hist[base00 + k_left] += w_k0 * s00;
            hist[base00 + k_right] += w_k1 * s00;
            hist[base01 + k_left] += w_k0 * s01;
            hist[base01 + k_right] += w_k1 * s01;
            hist[base10 + k_left] += w_k0 * s10;
            hist[base10 + k_right] += w_k1 * s10;
            hist[base11 + k_left] += w_k0 * s11;
            hist[base11 + k_right] += w_k1 * s11;
        }
    }

    normalize_and_finalize_descriptor(&mut hist, root_sift)
}

/// Reusable column terms for the row-major SIFT descriptor kernel.
#[derive(Default)]
pub(crate) struct SiftDescriptorScratch {
    col_gauss: Vec<f32>,
    col_xr: Vec<f32>,
    col_yr: Vec<f32>,
}

/// Scratch-buffer variant used by batch descriptor computation to avoid three
/// allocations per keypoint and to support arbitrarily large valid patches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_sift_descriptor_from_gradients_with_scratch(
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    root_sift: bool,
    ix: &[f32],
    iy: &[f32],
    rows: usize,
    cols: usize,
    scratch: &mut SiftDescriptorScratch,
) -> Option<[f32; SIFT_DESCR_SIZE]> {
    let expected_len = rows.checked_mul(cols)?;
    if rows == 0 || cols == 0 || ix.len() != expected_len || iy.len() != expected_len {
        return None;
    }

    compute_sift_descriptor_from_gradient_sampler_with_scratch(
        x,
        y,
        sigma,
        orientation,
        root_sift,
        rows,
        cols,
        scratch,
        |row, col| {
            let index = row * cols + col;
            Some((ix[index], iy[index]))
        },
    )
}

/// Histogram core over gradients read through `gradient_at`, which returns
/// `None` for a sample outside the gradient domain.
#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_sift_descriptor_from_gradient_sampler_with_scratch<F>(
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    root_sift: bool,
    rows: usize,
    cols: usize,
    scratch: &mut SiftDescriptorScratch,
    gradient_at: F,
) -> Option<[f32; SIFT_DESCR_SIZE]>
where
    F: Fn(usize, usize) -> Option<(f32, f32)>,
{
    if sigma < 1e-6 || rows == 0 || cols == 0 {
        return None;
    }

    let cos_t = cosf(-orientation);
    let sin_t = sinf(-orientation);

    let sigma_scale = DESCRIPTOR_SCALE * sigma;
    let r = (1.0 + 1.0 / NB_SPLIT2D as f32) * sigma_scale;
    let rp = r * SQRT2;
    let inv_bin_scale = NB_SPLIT2D as f32 / (2.0 * sigma_scale);
    let half_bins = (NB_SPLIT2D as f32 - 1.0) * 0.5;
    let inv_sigma_sq = 1.0 / (2.0 * sigma_scale * sigma_scale);
    let angle_to_bin = NB_SPLIT_ANGLE as f32 / TWO_PI;

    // Bounding box clamped to valid image coordinates.
    let row_min = ((y - rp).max(0.0)) as usize;
    let row_max = ((y + rp + 1.0) as usize).min(rows);
    let col_min = ((x - rp).max(0.0)) as usize;
    let col_max = ((x + rp + 1.0) as usize).min(cols);

    if row_min >= row_max || col_min >= col_max {
        return None;
    }

    // Pre-compute per-column terms (one exp call per column, not per pixel).
    // Rotation: xr = cos_t·x_ref − sin_t·y_ref,  yr = sin_t·x_ref + cos_t·y_ref
    // → col contribution: xr_col = cos_t·x_ref,  yr_col = sin_t·x_ref
    // → row contribution: xr_row = −sin_t·y_ref, yr_row =  cos_t·y_ref
    // Gaussian is rotation-invariant: x_ref²+y_ref² = xr²+yr², so it factors.
    let n_cols = col_max - col_min;
    scratch.col_gauss.resize(n_cols, 0.0);
    scratch.col_xr.resize(n_cols, 0.0);
    scratch.col_yr.resize(n_cols, 0.0);

    for j in 0..n_cols {
        let x_ref = (col_min + j) as f32 - x;
        scratch.col_gauss[j] = expf(-x_ref * x_ref * inv_sigma_sq);
        scratch.col_xr[j] = cos_t * x_ref;
        scratch.col_yr[j] = sin_t * x_ref;
    }

    let mut hist = [0.0f32; SIFT_DESCR_SIZE];

    for row in row_min..row_max {
        let y_ref = row as f32 - y;
        let row_gauss = expf(-y_ref * y_ref * inv_sigma_sq);

        // Row contributions to rotated coordinates.
        let xr_row = -sin_t * y_ref;
        let yr_row = cos_t * y_ref;

        for j in 0..n_cols {
            let xr = scratch.col_xr[j] + xr_row;
            let yr = scratch.col_yr[j] + yr_row;

            // Circular patch boundary.
            if xr.abs().max(yr.abs()) >= r {
                continue;
            }

            let (gx, gy) = match gradient_at(row, col_min + j) {
                Some(gradient) => gradient,
                None => continue,
            };
            let mag_sq = gx * gx + gy * gy;
            if mag_sq < 1e-12 {
                continue;
            }
            let mag = mag_sq.sqrt();
            let theta = fast_atan2(gy, gx);

            // Gaussian-weighted magnitude.
            let m = mag * row_gauss * scratch.col_gauss[j];

            // `theta` and `orientation` are both in [-pi, pi], so one
            // correction maps their difference into [0, 2pi) without the
            // division used by `rem_euclid`.
            let mut ori = theta - orientation;
            if ori < 0.0 {
                ori += TWO_PI;
            } else if ori >= TWO_PI {
                ori -= TWO_PI;
            }

            // Trilinear bin coordinates.
            let alpha = xr * inv_bin_scale + half_bins;
            let beta = yr * inv_bin_scale + half_bins;
            let gamma = ori * angle_to_bin;

            if !gamma.is_finite() {
                continue;
            }

            let alpha_floor = alpha.floor();
            let beta_floor = beta.floor();
            let gamma_floor = gamma.floor();
            let i0 = alpha_floor as i32;
            let j0 = beta_floor as i32;
            let alpha_frac = alpha - alpha_floor;
            let beta_frac = beta - beta_floor;
            let gamma_frac = gamma - gamma_floor;

            let i0v = i0 >= 0 && i0 < NB_SPLIT2D as i32;
            let i1v = (i0 + 1) >= 0 && (i0 + 1) < NB_SPLIT2D as i32;
            let j0v = j0 >= 0 && j0 < NB_SPLIT2D as i32;
            let j1v = (j0 + 1) >= 0 && (j0 + 1) < NB_SPLIT2D as i32;

            let k_left = gamma_floor as usize % NB_SPLIT_ANGLE;
            let k_right = (k_left + 1) % NB_SPLIT_ANGLE;

            let alpha_inv = 1.0 - alpha_frac;
            let beta_inv = 1.0 - beta_frac;
            let gamma_inv = 1.0 - gamma_frac;

            // Histogram updates are scatters, so validity-mask arithmetic does
            // not vectorize this loop. Skip invalid neighboring cells instead;
            // pixels near the descriptor boundary commonly have only one or
            // two valid spatial destinations.
            if i0v && j0v {
                let base = i0 as usize * NB_SPLIT2D * NB_SPLIT_ANGLE + j0 as usize * NB_SPLIT_ANGLE;
                let spatial = alpha_inv * beta_inv * m;
                hist[base + k_left] += gamma_inv * spatial;
                hist[base + k_right] += gamma_frac * spatial;
            }
            if i0v && j1v {
                let base =
                    i0 as usize * NB_SPLIT2D * NB_SPLIT_ANGLE + (j0 + 1) as usize * NB_SPLIT_ANGLE;
                let spatial = alpha_inv * beta_frac * m;
                hist[base + k_left] += gamma_inv * spatial;
                hist[base + k_right] += gamma_frac * spatial;
            }
            if i1v && j0v {
                let base =
                    (i0 + 1) as usize * NB_SPLIT2D * NB_SPLIT_ANGLE + j0 as usize * NB_SPLIT_ANGLE;
                let spatial = alpha_frac * beta_inv * m;
                hist[base + k_left] += gamma_inv * spatial;
                hist[base + k_right] += gamma_frac * spatial;
            }
            if i1v && j1v {
                let base = (i0 + 1) as usize * NB_SPLIT2D * NB_SPLIT_ANGLE
                    + (j0 + 1) as usize * NB_SPLIT_ANGLE;
                let spatial = alpha_frac * beta_frac * m;
                hist[base + k_left] += gamma_inv * spatial;
                hist[base + k_right] += gamma_frac * spatial;
            }
        }
    }

    normalize_and_finalize_descriptor(&mut hist, root_sift)
}

/// L2-normalize → apply RootSIFT, or else clip at `SIFT_DESCR_MAG_THR` →
/// scale to `[0, 255]`.
///
/// Returns `None` if the initial L2 norm is below the numerical floor (~1e-7),
/// which indicates a degenerate (uniform) patch, or is not finite.
pub(crate) fn normalize_and_finalize_descriptor(
    hist: &mut [f32; SIFT_DESCR_SIZE],
    root_sift: bool,
) -> Option<[f32; SIFT_DESCR_SIZE]> {
    let norm = hist.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 1e-7 {
        return None;
    }
    let inv_norm = 1.0 / norm;
    for val in hist.iter_mut() {
        *val *= inv_norm;
    }

    if root_sift {
        // RootSIFT (Arandjelović & Zisserman, CVPR 2012): L1-normalize → sqrt.
        // This variant roots the unclipped histogram. Clipping first would
        // change the distribution represented by its Hellinger embedding.
        let l1_norm: f32 = hist.iter().map(|v| v.abs()).sum::<f32>().max(1e-7);
        for val in hist.iter_mut() {
            *val = ((*val / l1_norm).max(0.0)).sqrt();
        }
    } else {
        // Classic SIFT: clip at 0.2 to reduce illumination sensitivity.
        for val in hist.iter_mut() {
            *val = val.min(SIFT_DESCR_MAG_THR);
        }
    }

    let norm2 = hist.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-7);
    let scale = SIFT_INT_DESCR_FCTR / norm2;
    let mut result = [0.0f32; SIFT_DESCR_SIZE];
    for (i, &v) in hist.iter().enumerate() {
        result[i] = (v * scale).min(255.0);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_compat::atan2f;

    #[test]
    fn horizontal_gradient_has_zero_orientation() {
        let mut hist = build_orientation_histogram(20.0, 20.0, 2.0, |_row, _col| Some((1.0, 0.0)));
        smooth_orientation_histogram(&mut hist, ORI_SMOOTH_PASSES);
        let peaks = extract_orientation_peaks(&hist);
        assert_eq!(peaks.len(), 1);
        assert!(peaks[0].abs() < 1e-4, "orientation={}", peaks[0]);
    }

    #[test]
    fn vertical_gradient_has_half_pi_orientation() {
        let mut hist = build_orientation_histogram(20.0, 20.0, 2.0, |_row, _col| {
            Some((1.0, core::f32::consts::FRAC_PI_2))
        });
        smooth_orientation_histogram(&mut hist, ORI_SMOOTH_PASSES);
        let peaks = extract_orientation_peaks(&hist);
        assert_eq!(peaks.len(), 1);
        assert!((peaks[0] - core::f32::consts::FRAC_PI_2).abs() < 1e-4);
    }

    #[test]
    fn dominant_orientation_is_the_highest_peak() {
        let mut hist = [0.0f32; ORI_HIST_BINS];
        hist[4] = 0.9;
        hist[20] = 1.0;
        let peaks = extract_orientation_peaks(&hist);
        assert_eq!(peaks.len(), 2);
        assert_eq!(dominant_orientation(&hist), Some(peaks[1]));
        hist[4] = 1.0;
        assert_eq!(dominant_orientation(&hist), Some(peaks[0]));
        assert_eq!(dominant_orientation(&[0.0; ORI_HIST_BINS]), None);
    }

    #[test]
    fn non_finite_histogram_has_no_descriptor() {
        for poison in [f32::NAN, f32::INFINITY] {
            let mut hist = [1.0f32; SIFT_DESCR_SIZE];
            hist[7] = poison;
            for root_sift in [true, false] {
                let mut copy = hist;
                assert!(normalize_and_finalize_descriptor(&mut copy, root_sift).is_none());
            }
        }
    }

    #[test]
    fn orientation_weight_uses_subpixel_center() {
        let sigma = 2.0;
        let cx = 20.25;
        let cy = 20.5;
        let hist = build_orientation_histogram(cx, cy, sigma, |row, col| {
            (row == 21 && col == 20).then_some((1.0, 0.0))
        });
        let distance_squared = (20.0 - cx).powi(2) + (21.0 - cy).powi(2);
        let window_sigma = ORI_SCALE * sigma;
        let expected = expf(-distance_squared / (2.0 * window_sigma * window_sigma));
        let actual: f32 = hist.iter().sum();
        assert!((actual - expected).abs() < 1e-6, "actual={actual}");
        assert!(actual < 1.0);
    }

    #[test]
    fn descriptor_support_larger_than_160_columns_is_not_truncated() {
        let rows = 401;
        let cols = 401;
        let ix = vec![1.0f32; rows * cols];
        let iy = vec![0.0f32; rows * cols];
        let mut scratch = SiftDescriptorScratch::default();
        let descriptor = compute_sift_descriptor_from_gradients_with_scratch(
            200.0,
            200.0,
            20.0,
            0.0,
            true,
            &ix,
            &iy,
            rows,
            cols,
            &mut scratch,
        )
        .expect("a large valid gradient patch must produce a descriptor");

        assert!(scratch.col_gauss.len() > 160);
        assert!(descriptor.iter().any(|&value| value > 0.0));
    }

    #[test]
    fn optimized_sampler_tracks_exact_histogram() {
        let rows = 81;
        let cols = 79;
        let ix: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.02)
            .collect();
        let iy: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 29 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let exact = compute_sift_descriptor_histogram(39.35, 40.20, 3.1, 0.37, true, |row, col| {
            if row < 0 || col < 0 || row >= rows as i32 || col >= cols as i32 {
                return None;
            }
            let index = row as usize * cols + col as usize;
            let gx = ix[index];
            let gy = iy[index];
            Some(((gx * gx + gy * gy).sqrt(), atan2f(gy, gx)))
        })
        .unwrap();
        let mut scratch = SiftDescriptorScratch::default();
        let optimized = compute_sift_descriptor_from_gradients_with_scratch(
            39.35,
            40.20,
            3.1,
            0.37,
            true,
            &ix,
            &iy,
            rows,
            cols,
            &mut scratch,
        )
        .unwrap();

        let dot: f32 = exact.iter().zip(&optimized).map(|(a, b)| a * b).sum();
        let exact_norm = exact.iter().map(|value| value * value).sum::<f32>().sqrt();
        let optimized_norm = optimized
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let cosine = dot / (exact_norm * optimized_norm);
        assert!(cosine > 0.99999, "cosine={cosine}");
    }
}
