// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hierarchical Gaussian scale space, Difference of Gaussians, and the
//! per-octave gradient images read by orientation and description.

use super::super::alp::GrayImage;
use super::super::convolution::GaussianFilter;
use crate::math_compat::{log2f, powf};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub(super) const SUPPLEMENTARY_LEVELS: usize = 3;
const MIN_OCTAVE_DIMENSION: usize = 32;

/// One octave of the Gaussian scale space.
pub(super) struct GaussianOctave {
    pub(super) octave_level: usize,
    /// Inter-pixel spacing at this octave: 1.0 for octave 0, 0.5 for -1, 2.0 for 1.
    pub(super) delta: f32,
    /// Sigma values for each slice (in absolute image-space coordinates).
    pub(super) sigmas: Vec<f32>,
    /// Gaussian-blurred images (num_scales + 3 slices).
    pub(super) slices: Vec<GrayImage>,
}

/// One octave of Difference-of-Gaussians images.
pub(super) struct DogOctave {
    pub(super) octave_level: usize,
    pub(super) delta: f32,
    pub(super) sigmas: Vec<f32>,
    pub(super) slices: Vec<GrayImage>,
}

/// X and Y gradient images for one octave.
pub(super) struct GradientOctave {
    pub(super) slices: Vec<Option<GradientSlice>>,
    pub(super) delta: f32,
}

pub(super) struct GradientSlice {
    pub(super) x: Vec<f32>,
    pub(super) y: Vec<f32>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

// ============================================================================
// Hierarchical Gaussian Scale Space
// ============================================================================

pub(super) struct HierarchicalGaussianScaleSpace {
    pub(super) num_octaves: usize,
    pub(super) num_scales: usize,
    pub(super) sigma_min: f32,
    pub(super) delta_min: f32,
    pub(super) sigma_in: f32,
    pub(super) current_octave_id: usize,
    pub(super) current_base_image: Option<GrayImage>,
}

impl HierarchicalGaussianScaleSpace {
    pub(super) fn new(
        num_octaves: usize,
        num_scales: usize,
        sigma_min: f32,
        delta_min: f32,
        sigma_in: f32,
    ) -> Self {
        Self {
            num_octaves,
            num_scales,
            sigma_min,
            delta_min,
            sigma_in,
            current_octave_id: 0,
            current_base_image: None,
        }
    }

    /// Initialize with input image, applying initial blur and optional upsampling.
    pub(super) fn set_image(&mut self, img: &GrayImage) {
        let sigma_extra =
            ((self.sigma_min * self.sigma_min - self.sigma_in * self.sigma_in).max(0.0)).sqrt()
                / self.delta_min;

        let base = if (self.delta_min - 0.5).abs() < 0.01 {
            // first_octave = -1: upsample 2× then blur
            let upsampled = upsample_2x(img);
            if sigma_extra > 0.01 {
                GaussianFilter::new(sigma_extra).convolve_separable(&upsampled)
            } else {
                upsampled
            }
        } else {
            // first_octave = 0: just blur
            if sigma_extra > 0.01 {
                GaussianFilter::new(sigma_extra).convolve_separable(img)
            } else {
                img.clone()
            }
        };

        // Limit octaves so the smallest dimension is large enough for detection.
        let max_octaves =
            (log2f(base.nrows().min(base.ncols()) as f32 / MIN_OCTAVE_DIMENSION as f32).floor()
                as usize
                + 1)
            .max(1);
        self.num_octaves = self.num_octaves.min(max_octaves);

        self.current_base_image = Some(base);
        self.current_octave_id = 0;
    }

    /// Build the next octave. Returns `None` when all octaves are done.
    pub(super) fn next_octave(&mut self) -> Option<GaussianOctave> {
        if self.current_octave_id >= self.num_octaves {
            return None;
        }

        let base = self.current_base_image.as_ref()?;
        let total_slices = self.num_scales + SUPPLEMENTARY_LEVELS;

        let delta = if self.current_octave_id == 0 {
            self.delta_min
        } else {
            self.delta_min * (1 << self.current_octave_id) as f32
        };

        // Compute sigma values for each slice
        let sigmas: Vec<f32> = (0..total_slices)
            .map(|s| {
                delta / self.delta_min
                    * self.sigma_min
                    * powf(2.0f32, s as f32 / self.num_scales as f32)
            })
            .collect();

        // Build slices with cascaded Gaussian blur
        let mut slices = Vec::with_capacity(total_slices);
        slices.push(base.clone());

        for s in 1..total_slices {
            let sig_prev = sigmas[s - 1];
            let sig_next = sigmas[s];
            let sigma_extra = ((sig_next * sig_next - sig_prev * sig_prev).max(0.0)).sqrt() / delta;
            let sigma_extra = sigma_extra.max(0.01);
            let blurred = GaussianFilter::new(sigma_extra).convolve_separable(&slices[s - 1]);
            slices.push(blurred);
        }

        let octave = GaussianOctave {
            octave_level: self.current_octave_id,
            delta,
            sigmas,
            slices,
        };

        // The `num_scales` slice is one octave above the current base.
        self.current_octave_id += 1;
        if self.current_octave_id < self.num_octaves {
            let decimated = decimate_2x(&octave.slices[self.num_scales]);
            self.current_base_image = Some(decimated);
        }

        Some(octave)
    }
}

// ============================================================================
// Image resampling
// ============================================================================

/// Bilinear 2× upsampling.
pub(super) fn upsample_2x(img: &GrayImage) -> GrayImage {
    let (h, w) = (img.nrows(), img.ncols());
    let (nh, nw) = (h * 2, w * 2);
    let mut out = GrayImage::zeros(nh, nw);

    for i in 0..nh {
        let src_y = i as f32 * 0.5;
        let iy = (src_y.floor() as usize).min(h - 1);
        let iy1 = (iy + 1).min(h - 1);
        let fy = src_y - iy as f32;

        for j in 0..nw {
            let src_x = j as f32 * 0.5;
            let ix = (src_x.floor() as usize).min(w - 1);
            let ix1 = (ix + 1).min(w - 1);
            let fx = src_x - ix as f32;

            out[(i, j)] = (1.0 - fy) * ((1.0 - fx) * img[(iy, ix)] + fx * img[(iy, ix1)])
                + fy * ((1.0 - fx) * img[(iy1, ix)] + fx * img[(iy1, ix1)]);
        }
    }
    out
}

/// 2× decimation: keep every second sample.
///
/// Keypoint coordinates place octave sample `i` at `i · delta`, so it must be
/// input sample `2i`. Averaging 2×2 blocks would center it at `2i + 0.5` and
/// shift coarse-octave keypoints by `(2^o − 1) / 2` input pixels.
pub(super) fn decimate_2x(img: &GrayImage) -> GrayImage {
    let (h, w) = (img.nrows() / 2, img.ncols() / 2);
    GrayImage::from_fn(h, w, |i, j| img[(2 * i, 2 * j)])
}

// ============================================================================
// Difference of Gaussians
// ============================================================================

pub(super) fn compute_dogs(octave: &GaussianOctave) -> DogOctave {
    let n = octave.slices.len();
    assert!(
        n >= 2,
        "need at least 2 Gaussian slices to compute DoG, got {n}"
    );

    let slices: Vec<GrayImage> = (0..n - 1)
        .map(|s| &octave.slices[s + 1] - &octave.slices[s])
        .collect();

    DogOctave {
        octave_level: octave.octave_level,
        delta: octave.delta,
        sigmas: octave.sigmas.clone(),
        slices,
    }
}

// ============================================================================
// Gradient Computation
// ============================================================================

pub(super) fn compute_gradients(octave: &GaussianOctave) -> GradientOctave {
    let n = octave.slices.len();
    let mut slices: Vec<Option<GradientSlice>> = (0..n).map(|_| None).collect();

    // Only compute gradients for slices [1, n-2] (used by orientation + descriptor)
    for (slot, im) in slices.iter_mut().zip(&octave.slices).take(n - 1).skip(1) {
        let (h, w) = (im.nrows(), im.ncols());

        let mut gx = vec![0.0f32; h * w];
        let mut gy = vec![0.0f32; h * w];

        for i in 1..h - 1 {
            for j in 1..w - 1 {
                gx[i * w + j] = (im[(i, j + 1)] - im[(i, j - 1)]) * 0.5;
                gy[i * w + j] = (im[(i + 1, j)] - im[(i - 1, j)]) * 0.5;
            }
        }

        *slot = Some(GradientSlice {
            x: gx,
            y: gy,
            rows: h,
            cols: w,
        });
    }

    GradientOctave {
        slices,
        delta: octave.delta,
    }
}
