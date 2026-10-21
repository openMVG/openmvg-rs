// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full-image ALP reference implementation (test-only).
//!
//! This is the straightforward formulation of the ALP detector: every octave
//! materializes its four Gaussian levels as whole images, the scale fit runs
//! on a rolling LoG column buffer, and orientation and descriptors sample the
//! full images. The production engine in [`super::stream`] streams the
//! same computation with bounded memory; the tests here pin the arithmetic
//! that both formulations share and compare the two.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;

use super::GrayImage;
use super::blur::{BaseImage, ColumnBlur, octave_kernels};
use super::scale_fit::{
    CubicCoefficients, ORIENTATION_SCALE_SNAP, SCALE_WINDOW, band_extreme, compute_poly_coeffs,
    neighbor_scale_band, orientation_histogram_bin, owns_scale_extremum, refine_keypoint_2d,
};
use super::stream::{Candidate, DetectionParams, scale_envelope};
use super::tables::SIGMAS;
use crate::extraction::ScaleSpaceKeypoint;
use crate::extraction::convolution::GaussianFilter;
use crate::extraction::sift::common::{
    ORI_HIST_BINS, ORI_SCALE, ORI_SMOOTH_PASSES, dominant_orientation, smooth_orientation_histogram,
};
use crate::math_compat::expf;

/// Upper bound of detected octave-local σ, which sizes the orientation window.
const HIGH_SIGMA: f32 = SCALE_WINDOW.1;
const DETECTION_BORDER: usize = 4;

/// Four Gaussian-blurred images for one pyramid octave, sampled at [`SIGMAS`] levels.
pub(crate) struct ScaleSpace {
    gaussians: [GrayImage; 4],
}

impl ScaleSpace {
    /// The octave's levels materialized with the streaming engine's column
    /// blur, so a comparison isolates the ring and scan logic from the
    /// (separately tested) blur summation order.
    pub(crate) fn from_column_blur(base: &GrayImage, octave: usize) -> Self {
        let (rows, cols) = (base.nrows(), base.ncols());
        let kernels = octave_kernels(octave);
        let source = BaseImage {
            pixels: base.as_slice(),
            rows,
            cols,
        };
        let mut blur = ColumnBlur::new(rows);
        let mut levels: [Vec<f32>; 4] = core::array::from_fn(|_| vec![0.0; rows * cols]);
        for col in 0..cols {
            blur.blur_column(source, col, &kernels, [true; 4], &mut levels, col * rows);
        }
        Self {
            gaussians: levels.map(|level| GrayImage::from_vec(rows, cols, level)),
        }
    }

    /// Scale space for one octave of the ALP pyramid.
    pub(crate) fn build_octave(base: &GrayImage, octave: usize) -> Self {
        // Direct blur from base image — each scale computed independently.
        // This avoids cumulative floating-point error from cascaded blurring.
        //
        // Octave 0: base image has σ_base = 0.5 (typical camera blur)
        // Octave N>0: base from 2× downsampling of gaussians[2] (σ=3.2 → 1.6)
        let base_sigma = if octave == 0 { 0.5 } else { SIGMAS[2] / 2.0 };

        let filters: [GaussianFilter; 4] = core::array::from_fn(|i| {
            let target_sigma = SIGMAS[i];
            let sigma_sq_diff = (target_sigma * target_sigma - base_sigma * base_sigma).max(0.0);
            let incremental_sigma = sigma_sq_diff.sqrt().max(0.1);
            GaussianFilter::new(incremental_sigma)
        });

        // Build levels sequentially. Pipelines parallelize across images; running
        // four full-image convolutions concurrently only raises transient memory.
        let gaussians: [GrayImage; 4] =
            core::array::from_fn(|i| filters[i].convolve_separable(base));

        Self { gaussians }
    }

    #[inline]
    fn log_response_at(&self, scale_index: usize, row: usize, column: usize) -> f32 {
        let image = &self.gaussians[scale_index];
        let center = image[(row, column)];
        (image[(row, column - 1)]
            + image[(row, column + 1)]
            + image[(row - 1, column)]
            + image[(row + 1, column)]
            - 4.0 * center)
            * (SIGMAS[scale_index] * SIGMAS[scale_index])
    }

    #[inline]
    pub(super) fn eval_poly([a, b, c, d]: CubicCoefficients, s: f32) -> f32 {
        ((a * s + b) * s + c) * s + d
    }

    pub(super) fn polynomial_extrema(
        coeffs: CubicCoefficients,
        l0: f32,
        l3: f32,
    ) -> (f32, f32, f32, f32) {
        let [a, b, c, _] = coeffs;
        let mut min_resp = l0.min(l3);
        let mut max_resp = l0.max(l3);
        let mut min_scale = 0.0;
        let mut max_scale = 0.0;
        if a.abs() > 1e-6 {
            let ap = 6.0 * a;
            let bp = 2.0 * b;
            let discriminant = bp * bp - 2.0 * ap * c;
            if discriminant > 0.0 {
                let sqrt_d = discriminant.sqrt();
                let minimum = (-bp + sqrt_d) / ap;
                let min_curvature = ap * minimum + bp;
                let value = Self::eval_poly(coeffs, minimum);
                if (SIGMAS[0]..=SIGMAS[3]).contains(&minimum)
                    && min_curvature.is_finite()
                    && min_curvature > 0.0
                    && value.is_finite()
                    && value < min_resp
                {
                    min_resp = value;
                    min_scale = minimum;
                }
                let maximum = (-bp - sqrt_d) / ap;
                let max_curvature = ap * maximum + bp;
                let value = Self::eval_poly(coeffs, maximum);
                if (SIGMAS[0]..=SIGMAS[3]).contains(&maximum)
                    && max_curvature.is_finite()
                    && max_curvature < 0.0
                    && value.is_finite()
                    && value > max_resp
                {
                    max_resp = value;
                    max_scale = maximum;
                }
            }
        } else if b.abs() > 1e-6 {
            let stationary = -c / (2.0 * b);
            let curvature = 6.0 * a * stationary + 2.0 * b;
            let value = Self::eval_poly(coeffs, stationary);
            if (SIGMAS[0]..=SIGMAS[3]).contains(&stationary)
                && curvature.is_finite()
                && value.is_finite()
            {
                if b > 0.0 && curvature > 0.0 && value < min_resp {
                    min_resp = value;
                    min_scale = stationary;
                }
                if b < 0.0 && curvature < 0.0 && value > max_resp {
                    max_resp = value;
                    max_scale = stationary;
                }
            }
        }
        (min_resp, max_resp, min_scale, max_scale)
    }

    #[inline]
    fn is_spatial_extremum_buffered(
        val: f32,
        is_min: bool,
        sigma: f32,
        coeffs_cache: &[CubicCoefficients; 9],
    ) -> bool {
        let (low, high) = neighbor_scale_band(sigma);
        for (idx, &coeffs) in coeffs_cache.iter().enumerate() {
            if idx == 4 {
                continue;
            }

            let neighbor = band_extreme(coeffs, low, high, !is_min);

            let beats_neighbor = if is_min {
                val < neighbor
            } else {
                val > neighbor
            };
            if !beats_neighbor {
                return false;
            }
        }
        true
    }

    /// Pass-1 reference over the full octave images: candidates in row-major
    /// anchor order (minimum before maximum) with their original LoG samples.
    pub(super) fn detect_candidates(
        &self,
        octave: usize,
        params: &DetectionParams,
    ) -> Vec<Candidate> {
        let (rows, cols) = self.dimensions();
        let border = DETECTION_BORDER;
        let mut out = Vec::new();
        if rows <= 2 * border || cols <= 2 * border {
            return out;
        }
        let samples = |row: usize, col: usize| -> [f32; 4] {
            core::array::from_fn(|level| self.log_response_at(level, row, col))
        };
        let (low, high) = params.sigma_window;
        for row in border..rows - border {
            for col in border..cols - border {
                let center = samples(row, col);
                let (min_r, max_r, min_s, max_s) = scale_envelope(center);
                let [a, b, _, _] = compute_poly_coeffs(center[0], center[1], center[2], center[3]);
                for (sigma, peak, is_max) in [(min_s, min_r, false), (max_s, max_r, true)] {
                    let strong = if is_max {
                        peak > params.peak_threshold
                    } else {
                        peak < -params.peak_threshold
                    };
                    let curvature = 6.0 * a * sigma + 2.0 * b;
                    let polarity_matches = if is_max {
                        curvature < 0.0
                    } else {
                        curvature > 0.0
                    };
                    let curved = curvature.is_finite()
                        && polarity_matches
                        && curvature.abs().partial_cmp(&params.sigma_threshold)
                            == Some(core::cmp::Ordering::Greater);
                    let owned = owns_scale_extremum(&center, is_max);
                    if !((low..=high).contains(&sigma) && strong && curved && owned) {
                        continue;
                    }
                    let mut cache = [[0.0f32; 4]; 9];
                    for (index, slot) in cache.iter_mut().enumerate() {
                        let l = samples(row + index / 3 - 1, col + index % 3 - 1);
                        *slot = compute_poly_coeffs(l[0], l[1], l[2], l[3]);
                    }
                    let extremum = Self::is_spatial_extremum_buffered(peak, !is_max, sigma, &cache);
                    let mut geometry = ScaleSpaceKeypoint {
                        x: col as f32,
                        y: row as f32,
                        sigma,
                        orientation: 0.0,
                        peak,
                        octave,
                        scale_index: 0,
                        is_max,
                    };
                    if extremum
                        && refine_keypoint_2d(&mut geometry, &cache, params.curv_ratio_threshold)
                    {
                        out.push(Candidate {
                            geometry,
                            order_key: row * cols + col,
                            scale_samples: center,
                        });
                    }
                }
            }
        }
        out
    }

    /// Preserve the calibrated orientation path's near-node snap.
    #[inline]
    fn orientation_gradient_sampler(&self, sigma: f32) -> InterpolatedGradient<'_> {
        InterpolatedGradient::new(self, sigma, ORIENTATION_SCALE_SNAP)
    }

    /// Interpolate descriptor gradients continuously, snapping only exact nodes.
    #[inline]
    pub(crate) fn descriptor_gradient_sampler(&self, sigma: f32) -> InterpolatedGradient<'_> {
        InterpolatedGradient::new(self, sigma, 0.0)
    }

    pub(crate) fn dimensions(&self) -> (usize, usize) {
        (self.gaussians[0].nrows(), self.gaussians[0].ncols())
    }

    /// Dominant SIFT-style orientation of one keypoint (Lowe, IJCV 2004 §5):
    /// the highest peak of the smoothed histogram, `None` without a peak.
    fn assign_orientation(&self, kp: ScaleSpaceKeypoint) -> Option<f32> {
        let mut hist = self.orientation_histogram(kp);
        smooth_orientation_histogram(&mut hist, ORI_SMOOTH_PASSES);
        dominant_orientation(&hist)
    }

    fn orientation_histogram(&self, kp: ScaleSpaceKeypoint) -> [f32; ORI_HIST_BINS] {
        let mut histogram = [0.0; ORI_HIST_BINS];
        if kp.sigma < 1e-6 {
            return histogram;
        }

        let gradient = self.orientation_gradient_sampler(kp.sigma);
        let rows = gradient.low.nrows() as i32;
        let cols = gradient.low.ncols() as i32;
        let window_sigma = ORI_SCALE * kp.sigma;
        let radius = (3.0 * window_sigma).ceil() as i32;
        let inv_2sigma_sq = 1.0 / (2.0 * window_sigma * window_sigma);
        let center_row = kp.y.round() as i32;
        let center_col = kp.x.round() as i32;
        let row_min = (center_row - radius).max(1);
        let row_max = (center_row + radius).min(rows - 2);
        let col_min = (center_col - radius).max(1);
        let col_max = (center_col + radius).min(cols - 2);
        if row_min > row_max || col_min > col_max {
            return histogram;
        }

        // Detected ALP scales are bounded by HIGH_SIGMA (3.81), whose orientation
        // window has radius at most 18 pixels. Keeping these factors on the stack avoids
        // one exponential per image sample without adding per-keypoint storage.
        const MAX_ORIENTATION_RADIUS: usize = (3.0 * ORI_SCALE * HIGH_SIGMA) as usize + 1;
        const MAX_ORIENTATION_DIAMETER: usize = 2 * MAX_ORIENTATION_RADIUS + 1;
        let col_count = (col_max - col_min + 1) as usize;
        debug_assert!(col_count <= MAX_ORIENTATION_DIAMETER);
        let mut col_weights = [0.0f32; MAX_ORIENTATION_DIAMETER];
        for (offset, sample_col) in (col_min..=col_max).enumerate() {
            let dx = sample_col as f32 - kp.x;
            col_weights[offset] = expf(-(dx * dx) * inv_2sigma_sq);
        }

        for sample_row in row_min..=row_max {
            let dy = sample_row as f32 - kp.y;
            let row_weight = expf(-(dy * dy) * inv_2sigma_sq);
            for (offset, sample_col) in (col_min..=col_max).enumerate() {
                let (gx, gy) =
                    gradient.components_interior(sample_row as usize, sample_col as usize);
                let magnitude = (gx * gx + gy * gy).sqrt();
                let angle = crate::extraction::sift::common::fast_atan2(gy, gx);
                let weight = row_weight * col_weights[offset];
                histogram[orientation_histogram_bin(angle)] += magnitude * weight;
            }
        }
        histogram
    }
}

pub(crate) struct InterpolatedGradient<'a> {
    low: &'a GrayImage,
    high: &'a GrayImage,
    t: f32,
}

impl<'a> InterpolatedGradient<'a> {
    #[inline]
    fn new(scale_space: &'a ScaleSpace, sigma: f32, endpoint_snap: f32) -> Self {
        debug_assert!((0.0..=0.5).contains(&endpoint_snap));
        let low_idx = if sigma <= SIGMAS[1] {
            0
        } else if sigma <= SIGMAS[2] {
            1
        } else {
            2
        };
        let high_idx = low_idx + 1;
        let sigma_low = SIGMAS[low_idx];
        let sigma_high = SIGMAS[high_idx];
        let t =
            ((sigma.ln() - sigma_low.ln()) / (sigma_high.ln() - sigma_low.ln())).clamp(0.0, 1.0);

        if t == 0.0 || t < endpoint_snap {
            Self {
                low: &scale_space.gaussians[low_idx],
                high: &scale_space.gaussians[low_idx],
                t: 0.0,
            }
        } else if t == 1.0 || t > 1.0 - endpoint_snap {
            Self {
                low: &scale_space.gaussians[high_idx],
                high: &scale_space.gaussians[high_idx],
                t: 0.0,
            }
        } else {
            Self {
                low: &scale_space.gaussians[low_idx],
                high: &scale_space.gaussians[high_idx],
                t,
            }
        }
    }

    /// Return the interpolated central-difference gradient at an interior pixel.
    ///
    /// `row` and `col` must exclude the one-pixel image border.
    #[inline]
    pub(crate) fn components_interior(&self, row: usize, col: usize) -> (f32, f32) {
        let (gx_low, gy_low) = gradient_components_at(self.low, row, col);
        let (gx, gy) = if self.t == 0.0 {
            (gx_low, gy_low)
        } else {
            let (gx_high, gy_high) = gradient_components_at(self.high, row, col);
            let gx = (1.0 - self.t) * gx_low + self.t * gx_high;
            let gy = (1.0 - self.t) * gy_low + self.t * gy_high;
            (gx, gy)
        };
        (gx, gy)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn sample(&self, row: i32, col: i32) -> Option<(f32, f32)> {
        let (row, col) = interior_pixel(self.low, row, col)?;
        let (gx, gy) = self.components_interior(row, col);
        let mag = (gx * gx + gy * gy).sqrt();
        Some((mag, crate::extraction::sift::common::fast_atan2(gy, gx)))
    }
}

#[cfg(test)]
#[inline]
fn interior_pixel(img: &GrayImage, row: i32, col: i32) -> Option<(usize, usize)> {
    if row < 1 || col < 1 || row + 1 >= img.nrows() as i32 || col + 1 >= img.ncols() as i32 {
        return None;
    }
    Some((row as usize, col as usize))
}

#[inline]
fn gradient_components_at(img: &GrayImage, row: usize, col: usize) -> (f32, f32) {
    let rows = img.nrows();
    let center = col * rows + row;
    let pixels = img.as_slice();
    debug_assert!(row > 0 && row + 1 < rows);
    debug_assert!(col > 0 && col + 1 < img.ncols());
    // nalgebra stores `DMatrix` data in column-major order. Callers restrict
    // samples to the one-pixel interior, and ordinary indexing keeps that
    // invariant fail-safe if a future caller is incorrect.
    (
        0.5 * (pixels[center + rows] - pixels[center - rows]),
        0.5 * (pixels[center + 1] - pixels[center - 1]),
    )
}

// Helper functions
/// 2× decimation of an already Gaussian-smoothed octave image.
///
/// Sampling `(2r, 2c)` preserves the reference ALP coordinate phase. Averaging
/// a 2×2 block would add unmodelled blur and shift the sample center by 0.5 px.
pub(crate) fn downsample(img: &GrayImage) -> GrayImage {
    if img.nrows() < 2 || img.ncols() < 2 {
        return GrayImage::zeros(0, 0);
    }
    let h = img.nrows() / 2;
    let w = img.ncols() / 2;
    GrayImage::from_fn(h, w, |row, column| img[(2 * row, 2 * column)])
}

#[cfg(test)]
mod tests {
    use super::super::scale_fit::get_nearest_scale_index;
    use super::super::stream::{DescribeRequest, describe_octave, detect_octave};
    use super::*;
    use crate::extraction::sift::canonical_descriptor::{CanonicalKernel, Frame, LevelWindow};
    use crate::extraction::sift::common::extract_orientation_peaks;
    use core::f32::consts::PI;

    fn create_constant_image(size: usize, value: f32) -> GrayImage {
        GrayImage::from_element(size, size, value)
    }

    fn create_textured_image(size: usize) -> GrayImage {
        GrayImage::from_fn(size, size, |row, col| {
            ((row * 37 + col * 17 + (row * col) % 251) % 256) as f32
        })
    }

    fn create_gaussian_blob_image(
        size: usize,
        center: (f32, f32),
        sigma: f32,
        amplitude: f32,
    ) -> GrayImage {
        let (cx, cy) = center;
        GrayImage::from_fn(size, size, |i, j| {
            let (dx, dy) = (j as f32 - cx, i as f32 - cy);
            amplitude * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
        })
    }

    #[test]
    fn spatial_nms_compares_typed_neighbors_within_one_level_of_the_candidate_scale() {
        let sigma = 2.0;
        let mut neighbors = [[0.0; 4]; 9];
        neighbors[4] = [0.0, 1.0, -4.0, -6.0]; // Stationary minimum -10 at sigma=2.
        // This neighbor reaches -11 at sigma=4, outside the band [1.6, 2.83],
        // and stays above -10 inside it.
        neighbors[0] = [0.0, 1.0, -8.0, 5.0];
        assert!(ScaleSpace::eval_poly(neighbors[0], 4.0) < -10.0);
        assert!(ScaleSpace::eval_poly(neighbors[0], sigma * core::f32::consts::SQRT_2) > -10.0);
        assert!(ScaleSpace::is_spatial_extremum_buffered(
            -10.0, true, sigma, &neighbors
        ));
        assert!(!ScaleSpace::is_spatial_extremum_buffered(
            -10.0, false, sigma, &neighbors
        ));
        let maximum = neighbors.map(|coefficients| coefficients.map(|value| -value));
        assert!(ScaleSpace::is_spatial_extremum_buffered(
            10.0, false, sigma, &maximum
        ));
        for competitor in [-10.0, -10.01, f32::NAN] {
            let mut tied = neighbors;
            tied[1] = [0.0, 0.0, 0.0, competitor];
            assert!(!ScaleSpace::is_spatial_extremum_buffered(
                -10.0, true, sigma, &tied
            ));
        }
        // Weaker at the candidate scale but stronger half a level above it:
        // the same-sigma comparison alone would accept this candidate.
        neighbors[2] = [0.0, 4.0, -20.0, 14.5];
        assert!(ScaleSpace::eval_poly(neighbors[2], sigma) > -10.0);
        assert!(ScaleSpace::eval_poly(neighbors[2], 2.5) < -10.0);
        assert!(!ScaleSpace::is_spatial_extremum_buffered(
            -10.0, true, sigma, &neighbors
        ));
    }

    #[test]
    fn continuous_gradient_is_node_exact_and_uses_ordinary_linear_blending() {
        let image = create_textured_image(96);
        let scale_space = ScaleSpace::build_octave(&image, 0);
        let row = 48;
        let col = 47;

        for (index, &sigma) in SIGMAS.iter().enumerate() {
            let actual = scale_space
                .descriptor_gradient_sampler(sigma)
                .components_interior(row, col);
            let expected = gradient_components_at(&scale_space.gaussians[index], row, col);
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }

        let sigma = (SIGMAS[0] * SIGMAS[1]).sqrt();
        let sampler = scale_space.descriptor_gradient_sampler(sigma);
        let low = gradient_components_at(&scale_space.gaussians[0], row, col);
        let high = gradient_components_at(&scale_space.gaussians[1], row, col);
        let expected = (
            (1.0 - sampler.t) * low.0 + sampler.t * high.0,
            (1.0 - sampler.t) * low.1 + sampler.t * high.1,
        );
        let actual = sampler.components_interior(row, col);
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    #[test]
    fn polynomial_extrema_handles_quadratic_limit() {
        let vertex = 2.5;
        let coeffs = [0.0, 2.0, -4.0 * vertex, 12.0];
        let l0 = ScaleSpace::eval_poly(coeffs, SIGMAS[0]);
        let l3 = ScaleSpace::eval_poly(coeffs, SIGMAS[3]);

        let (minimum, _, minimum_scale, _) = ScaleSpace::polynomial_extrema(coeffs, l0, l3);

        assert!((minimum_scale - vertex).abs() < 1e-6);
        assert!((minimum - ScaleSpace::eval_poly([0.0, 2.0, -10.0, 12.0], vertex)).abs() < 1e-6);
    }

    #[test]
    fn test_scale_space_build_octave_dimensions() {
        let img = GrayImage::from_element(64, 64, 100.0);
        let ss = ScaleSpace::build_octave(&img, 0);

        for i in 0..4 {
            assert_eq!(ss.gaussians[i].nrows(), 64);
            assert_eq!(ss.gaussians[i].ncols(), 64);
        }
    }

    #[test]
    fn test_scale_space_gaussians_are_smooth() {
        let img = create_gaussian_blob_image(64, (32.0, 32.0), 3.0, 200.0);
        let ss = ScaleSpace::build_octave(&img, 0);

        let variance_0 = compute_gradient_variance(&ss.gaussians[0]);
        let variance_3 = compute_gradient_variance(&ss.gaussians[3]);

        assert!(
            variance_3 <= variance_0,
            "Higher scale should be smoother: var0={}, var3={}",
            variance_0,
            variance_3
        );
    }

    fn compute_gradient_variance(img: &GrayImage) -> f32 {
        let (h, w) = (img.nrows(), img.ncols());
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;

        for i in 1..h - 1 {
            for j in 1..w - 1 {
                let gx = img[(i, j + 1)] - img[(i, j - 1)];
                let gy = img[(i + 1, j)] - img[(i - 1, j)];
                let mag = (gx * gx + gy * gy).sqrt();
                sum += mag;
                sum_sq += mag * mag;
                count += 1;
            }
        }

        let mean = sum / count as f32;
        sum_sq / count as f32 - mean * mean
    }

    #[test]
    fn test_scale_space_constant_image() {
        let img = create_constant_image(64, 128.0);
        let ss = ScaleSpace::build_octave(&img, 0);

        for scale_index in 0..4 {
            let g = &ss.gaussians[scale_index];
            let center_val = g[(32, 32)];

            for i in 5..59 {
                for j in 5..59 {
                    let diff = (g[(i, j)] - center_val).abs();
                    assert!(
                        diff < 1.0,
                        "Constant image blur should stay constant, diff={}",
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_orientation_assignment_horizontal_gradient() {
        let mut img = GrayImage::zeros(64, 64);
        for i in 0..64 {
            for j in 0..64 {
                img[(i, j)] = j as f32;
            }
        }
        let ss = ScaleSpace::build_octave(&img, 0);
        let orientation = ss
            .assign_orientation(ScaleSpaceKeypoint {
                x: 32.0,
                y: 32.0,
                sigma: 2.0,
                orientation: 0.0,
                peak: 1.0,
                octave: 0,
                scale_index: 1,
                is_max: true,
            })
            .unwrap();

        assert!(
            orientation.abs() < 1e-4,
            "horizontal gradient orientation should be 0, got {orientation}"
        );
    }

    #[test]
    fn test_orientation_assignment_vertical_gradient() {
        let mut img = GrayImage::zeros(64, 64);
        for i in 0..64 {
            for j in 0..64 {
                img[(i, j)] = i as f32;
            }
        }
        let ss = ScaleSpace::build_octave(&img, 0);
        let orientation = ss
            .assign_orientation(ScaleSpaceKeypoint {
                x: 32.0,
                y: 32.0,
                sigma: 2.0,
                orientation: 0.0,
                peak: 1.0,
                octave: 0,
                scale_index: 1,
                is_max: true,
            })
            .unwrap();

        assert!(
            (orientation - PI / 2.0).abs() < 1e-4,
            "vertical gradient orientation should be pi/2, got {orientation}"
        );
    }

    #[test]
    fn optimized_orientation_histogram_tracks_reference() {
        let image = GrayImage::from_fn(64, 80, |row, col| {
            ((row * 17 + col * 29 + row * col * 3) % 251) as f32
        });
        let scale_space = ScaleSpace::build_octave(&image, 0);
        for (x, y, sigma) in [(32.25, 27.75, 1.6), (3.4, 4.2, 2.7), (70.1, 58.3, 4.2)] {
            let keypoint = ScaleSpaceKeypoint {
                x,
                y,
                sigma,
                orientation: 0.0,
                peak: 1.0,
                octave: 0,
                scale_index: get_nearest_scale_index(sigma),
                is_max: true,
            };
            let gradient = scale_space.orientation_gradient_sampler(sigma);
            let mut reference = crate::extraction::sift::common::build_orientation_histogram(
                x,
                y,
                sigma,
                |row, col| gradient.sample(row, col),
            );
            let mut optimized = scale_space.orientation_histogram(keypoint);
            for (&actual, &expected) in optimized.iter().zip(&reference) {
                let tolerance = 2e-5 * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance);
            }
            smooth_orientation_histogram(&mut reference, ORI_SMOOTH_PASSES);
            smooth_orientation_histogram(&mut optimized, ORI_SMOOTH_PASSES);
            let reference_peaks = extract_orientation_peaks(&reference);
            let optimized_peaks = extract_orientation_peaks(&optimized);
            assert_eq!(optimized_peaks.len(), reference_peaks.len());
            for (actual, expected) in optimized_peaks.into_iter().zip(reference_peaks) {
                assert!((actual - expected).abs() <= 1e-5);
            }
        }
    }

    #[test]
    fn test_downsample_dimensions() {
        let img = GrayImage::from_element(64, 64, 100.0);
        let downsampled = downsample(&img);

        assert_eq!(downsampled.nrows(), 32);
        assert_eq!(downsampled.ncols(), 32);
    }

    #[test]
    fn test_downsample_preserves_constant() {
        let img = create_constant_image(64, 100.0);
        let downsampled = downsample(&img);

        for i in 0..32 {
            for j in 0..32 {
                let diff = (downsampled[(i, j)] - 100.0).abs();
                assert!(diff < 0.01, "Constant image should downsample to constant");
            }
        }
    }

    #[test]
    fn test_downsample_preserves_reference_phase() {
        let mut img = GrayImage::zeros(4, 4);
        // Set a 2x2 block to known values
        img[(0, 0)] = 1.0;
        img[(0, 1)] = 2.0;
        img[(1, 0)] = 3.0;
        img[(1, 1)] = 4.0;

        let downsampled = downsample(&img);

        assert_eq!(downsampled[(0, 0)], 1.0);
    }

    fn noise_image(rows: usize, cols: usize) -> GrayImage {
        GrayImage::from_fn(rows, cols, |row, col| {
            let mut state = (row as u32)
                .wrapping_mul(747_796_405)
                .wrapping_add((col as u32).wrapping_mul(2_891_336_453))
                .wrapping_add(0x1234_5678);
            state ^= state >> 16;
            state = state.wrapping_mul(2_246_822_519);
            state ^= state >> 13;
            (state >> 24) as f32
        })
    }

    fn smooth_texture(rows: usize, cols: usize) -> GrayImage {
        GrayImage::from_fn(rows, cols, |row, col| {
            let (y, x) = (row as f32, col as f32);
            128.0
                + 60.0 * (0.21 * x + 0.05 * y).sin() * (0.17 * y - 0.03 * x).cos()
                + 40.0 * (0.043 * x * y / 9.0).sin()
        })
    }

    fn params() -> DetectionParams {
        DetectionParams {
            peak_threshold: 5.0,
            sigma_threshold: 0.0,
            curv_ratio_threshold: 12.1,
            sigma_window: SCALE_WINDOW,
        }
    }

    fn base_of(image: &GrayImage) -> BaseImage<'_> {
        BaseImage {
            pixels: image.as_slice(),
            rows: image.nrows(),
            cols: image.ncols(),
        }
    }

    fn assert_same_geometry(actual: &ScaleSpaceKeypoint, expected: &ScaleSpaceKeypoint) {
        assert_eq!(actual.x.to_bits(), expected.x.to_bits());
        assert_eq!(actual.y.to_bits(), expected.y.to_bits());
        assert_eq!(actual.sigma.to_bits(), expected.sigma.to_bits());
        assert_eq!(actual.peak.to_bits(), expected.peak.to_bits());
        assert_eq!(actual.octave, expected.octave);
        assert_eq!(actual.scale_index, expected.scale_index);
        assert_eq!(actual.is_max, expected.is_max);
    }

    /// Pass 1 streams columns through rings; it must reproduce the full-image
    /// formulation bit for bit, including the decimated next-octave base.
    #[test]
    fn streamed_detection_matches_full_image_reference() {
        for image in [noise_image(181, 247), smooth_texture(200, 160)] {
            let mut base = image;
            let mut total = 0;
            for octave in 0..3 {
                let reference = ScaleSpace::from_column_blur(&base, octave);
                let expected = reference.detect_candidates(octave, &params());
                let streamed = detect_octave(base_of(&base), octave, Some(&params()), true);
                assert_eq!(streamed.candidates.len(), expected.len(), "octave {octave}");
                for (actual, expected) in streamed.candidates.iter().zip(&expected) {
                    assert_eq!(actual.order_key, expected.order_key);
                    assert_same_geometry(&actual.geometry, &expected.geometry);
                    assert_eq!(
                        actual.scale_samples.map(f32::to_bits),
                        expected.scale_samples.map(f32::to_bits)
                    );
                }
                total += expected.len();
                let next = downsample(&reference.gaussians[2]);
                assert_eq!(
                    (streamed.next_rows, streamed.next_cols),
                    (next.nrows(), next.ncols())
                );
                assert!(
                    streamed
                        .next_base
                        .iter()
                        .zip(next.as_slice())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                base = next;
            }
            assert!(total > 20, "too few candidates to be meaningful: {total}");
        }
    }

    /// The vectorizable envelope agrees with the branchy extremum solver.
    #[test]
    fn scale_envelope_tracks_polynomial_extrema() {
        let image = noise_image(64, 64);
        let reference = ScaleSpace::build_octave(&image, 0);
        for row in 2..62 {
            for col in 2..62 {
                let l: [f32; 4] = core::array::from_fn(|s| reference.log_response_at(s, row, col));
                let coeffs = compute_poly_coeffs(l[0], l[1], l[2], l[3]);
                let (emin, emax, smin, smax) = scale_envelope(l);
                let (rmin, rmax, qmin, qmax) = ScaleSpace::polynomial_extrema(coeffs, l[0], l[3]);
                let tolerance = 1e-3 * (1.0 + l.iter().fold(0.0f32, |m, v| m.max(v.abs())));
                assert!((emin - rmin).abs() <= tolerance && (emax - rmax).abs() <= tolerance);
                if smin > 0.0 && qmin > 0.0 {
                    assert!((smin - qmin).abs() <= 1e-3);
                }
                if smax > 0.0 && qmax > 0.0 {
                    assert!(
                        (smax - qmax).abs() <= 1e-3,
                        "max-root mismatch at ({row},{col}): samples={l:?}, coefficients={coeffs:?}, streamed={smax}, reference={qmax}, responses=({emax},{rmax})"
                    );
                }
            }
        }
    }

    /// The level a descriptor at `sigma` samples: the blend of the two
    /// bracketing levels, column-major.
    fn descriptor_level(reference: &ScaleSpace, sigma: f32) -> Vec<f32> {
        let gradient = reference.descriptor_gradient_sampler(sigma);
        let (keep, t) = (1.0 - gradient.t, gradient.t);
        gradient
            .low
            .as_slice()
            .iter()
            .zip(gradient.high.as_slice())
            .map(|(&low, &high)| if t == 0.0 { low } else { keep * low + t * high })
            .collect()
    }

    /// Pass 2 re-streams only the covered columns. Orientations match
    /// full-image sampling (histogram sums run in a different order, hence
    /// the tolerance), and the descriptor at the streamed orientation matches
    /// the kernel on the whole blended level bit for bit.
    #[test]
    fn streamed_description_matches_full_image_reference() {
        let image = smooth_texture(160, 200);
        let reference = ScaleSpace::from_column_blur(&image, 0);
        let (rows, cols) = reference.dimensions();
        let candidates = reference.detect_candidates(0, &params());
        assert!(candidates.len() > 10);
        let requests: Vec<DescribeRequest> = candidates
            .iter()
            .map(|c| DescribeRequest {
                x: c.geometry.x,
                y: c.geometry.y,
                sigma: c.geometry.sigma,
                orientation: None,
            })
            .collect();
        let streamed = describe_octave(base_of(&image), 0, &requests, true);
        let expected: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let orientation = reference.assign_orientation(candidate.geometry)?;
                Some((index, orientation))
            })
            .collect();
        assert_eq!(streamed.len(), expected.len());
        let kernel = CanonicalKernel::default();
        let mut described = 0;
        for (actual, &(index, orientation)) in streamed.iter().zip(&expected) {
            assert_eq!(actual.request, index);
            assert!(
                (actual.orientation - orientation).abs() < 1e-4,
                "{} vs {orientation}",
                actual.orientation
            );
            let request = &requests[index];
            let level = descriptor_level(&reference, request.sigma);
            let frame = Frame::new(request.x, request.y, request.sigma, actual.orientation);
            let descriptor = frame
                .and_then(|frame| kernel.describe(&LevelWindow::full(&level, rows, cols), &frame));
            assert_eq!(
                actual.descriptor.map(|values| values.map(f32::to_bits)),
                descriptor.map(|values| values.map(f32::to_bits)),
                "request {index}"
            );
            described += usize::from(descriptor.is_some());
        }
        assert!(described > 10, "too few descriptors: {described}");
    }
}

#[cfg(test)]
mod stationary_type_tests {
    use super::*;

    #[test]
    fn zero_discriminant_and_invalid_quadratic_type_are_not_extrema() {
        let (_, _, minimum, maximum) =
            ScaleSpace::polynomial_extrema([1.0, -7.5, 18.75, 0.0], 0.0, 0.0);
        assert_eq!((minimum, maximum), (0.0, 0.0));
        for coefficients in [
            [-1e-6, 2e-6, -9.6e-6, 0.0],
            [1e-6, -2e-6, 9.6e-6, 0.0],
            [0.0, 1.0, -4.8, f32::NAN],
            [f32::NAN, 1.0, -4.8, 0.0],
        ] {
            let (_, _, minimum, maximum) = ScaleSpace::polynomial_extrema(coefficients, 0.0, 0.0);
            assert_eq!((minimum, maximum), (0.0, 0.0));
        }
    }

    #[test]
    fn quadratic_minimum_and_maximum_keep_their_type() {
        let (_, _, minimum, maximum) =
            ScaleSpace::polynomial_extrema([0.0, 1.0, -4.8, 0.0], 1.0, 1.0);
        assert!(minimum > 0.0);
        assert_eq!(maximum, 0.0);
        let (_, _, minimum, maximum) =
            ScaleSpace::polynomial_extrema([0.0, -1.0, 4.8, 0.0], -1.0, -1.0);
        assert_eq!(minimum, 0.0);
        assert!(maximum > 0.0);
    }
}
