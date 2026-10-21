// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical-grid RootSIFT descriptor kernel.
//!
//! The classic kernel in `sift_common` visits every pixel of the rotated
//! support square, rotates the pixel's gradient into the keypoint frame, and
//! spreads it over the 4×4×8 histogram with weights that depend on where the
//! pixel falls. This kernel resamples the Gaussian level on a fixed grid
//! aligned with the keypoint frame instead:
//!
//! 1. The support square of half-width 7.5 σ (four 3σ cells plus half a cell
//!    on each side for the bilinear cell weights) is covered by 24 × 24
//!    samples spaced 0.625 σ, plus a one-sample ring for central differences.
//!    Sample `(u, v)` is a bilinear read of the level at
//!    `center + σ (u e1 + v e2)`, with `e1 = (cos θ, sin θ)` and
//!    `e2 = (−sin θ, cos θ)`.
//! 2. Central differences on the grid give the gradient directly in the
//!    keypoint frame, so no per-sample rotation is needed.
//! 3. The spatial pooling weight of a sample (Gaussian window times bilinear
//!    cell weights) depends only on its grid position and separates into one
//!    factor per axis, precomputed once in [`CanonicalKernel`]. Only the
//!    orientation bin depends on the image, and the cost per keypoint does not
//!    grow with σ.
//!
//! The output has the classic layout and finalization (4×4 cells of 8
//! orientation bins, RootSIFT, scaled by 512 and capped at 255), so it feeds
//! the descriptor storage and the matchers unchanged.
//!
//! The kernel reads the level through a [`LevelWindow`] that must contain the
//! keypoint's [`Frame::footprint`]. A streaming scale space only materializes
//! that rectangle; [`support_columns`] bounds it for every orientation.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
use crate::math_compat::{cosf, expf, sinf};

use super::common::{
    SIFT_DESCR_SIZE, SQRT2, TWO_PI, fast_atan2, normalize_and_finalize_descriptor,
};

/// Half-width of the support square in units of σ.
const HALF_EXTENT: f32 = 7.5;
/// Interior (gradient) samples per side of the grid.
const INNER: usize = 24;
/// Samples per side including the one-sample ring for central differences.
const SIDE: usize = INNER + 2;
/// Grid spacing in units of σ.
const STEP: f32 = 2.0 * HALF_EXTENT / INNER as f32;
/// Canonical coordinate of interior sample 0, in units of σ.
const FIRST: f32 = -0.5 * (INNER - 1) as f32 * STEP;
/// Canonical coordinate of ring sample 0, in units of σ.
const START: f32 = FIRST - STEP;
/// Width of one spatial cell in units of σ.
const CELL_WIDTH: f32 = 3.0;
/// Standard deviation of the Gaussian window in units of σ (half the width
/// of the 4×4 cell grid).
const WINDOW_SIGMA: f32 = 6.0;
/// Spatial cells per side.
const CELLS: usize = 4;
/// Orientation bins per cell.
const ORIENTATION_BINS: usize = 8;
/// Orientation bins per radian.
const ANGLE_TO_BIN: f32 = ORIENTATION_BINS as f32 / TWO_PI;

/// Keypoint geometry in the pixels of the level being read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    /// Column coordinate.
    x: f32,
    /// Row coordinate.
    y: f32,
    /// Scale (σ).
    sigma: f32,
    cos: f32,
    sin: f32,
}

impl Frame {
    /// Frame at column `x`, row `y`, scale `sigma` and `orientation` in
    /// radians; `None` for a non-finite value or `sigma <= 1e-6`.
    pub(crate) fn new(x: f32, y: f32, sigma: f32, orientation: f32) -> Option<Self> {
        let valid = x.is_finite()
            && y.is_finite()
            && sigma.is_finite()
            && orientation.is_finite()
            && sigma > 1e-6;
        valid.then(|| Self {
            x,
            y,
            sigma,
            cos: cosf(orientation),
            sin: sinf(orientation),
        })
    }

    /// Half-width of the axis-aligned box around the center that holds every
    /// grid sample, ring included.
    fn reach(&self) -> f32 {
        -START * self.sigma * (self.cos.abs() + self.sin.abs())
    }

    /// Inclusive rectangle of a `rows × cols` level read for this frame, or
    /// `None` when no descriptor exists: a level smaller than 3×3, or a frame
    /// whose samples all miss the one-pixel interior, where every gradient
    /// would be skipped.
    ///
    /// The rectangle is the sample box widened by one pixel, which holds both
    /// bilinear neighbors of every sample; the early `None` also keeps the
    /// clamped reads of a frame crossing the border (columns 0 and 1, rows 0
    /// and 1) inside it. The spread `|cos θ| + |sin θ|` is capped at √2 so the
    /// rectangle never exceeds the orientation-free bound of
    /// [`support_columns`].
    pub(crate) fn footprint(&self, rows: usize, cols: usize) -> Option<Footprint> {
        if rows < 3 || cols < 3 {
            return None;
        }
        let reach = -START * self.sigma * (self.cos.abs() + self.sin.abs()).min(SQRT2);
        let (max_col, max_row) = ((cols - 1) as f32, (rows - 1) as f32);
        let misses = self.x + reach < 0.0
            || self.y + reach < 0.0
            || self.x - reach > max_col
            || self.y - reach > max_row;
        if !reach.is_finite() || misses {
            return None;
        }
        let radius = reach + 1.0;
        Some(Footprint {
            rows: span(self.y, radius, rows),
            cols: span(self.x, radius, cols),
        })
    }
}

/// Inclusive row and column ranges of a level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Footprint {
    /// First and last row.
    pub(crate) rows: (usize, usize),
    /// First and last column.
    pub(crate) cols: (usize, usize),
}

impl Footprint {
    fn height(&self) -> usize {
        self.rows.1 - self.rows.0 + 1
    }

    fn width(&self) -> usize {
        self.cols.1 - self.cols.0 + 1
    }

    fn covers(&self, other: &Self) -> bool {
        self.rows.0 <= other.rows.0
            && other.rows.1 <= self.rows.1
            && self.cols.0 <= other.cols.0
            && other.cols.1 <= self.cols.1
    }
}

/// Inclusive range `[center − radius, center + radius]` of a dimension of
/// `len` samples, rounded outward and clamped to it.
fn span(center: f32, radius: f32, len: usize) -> (usize, usize) {
    let last = (len - 1) as f32;
    (
        (center - radius).floor().clamp(0.0, last) as usize,
        (center + radius).ceil().clamp(0.0, last) as usize,
    )
}

/// Inclusive columns of a level with `cols` columns that a keypoint at column
/// `x` with scale `sigma` can read at any orientation; `None` when it has no
/// descriptor at any orientation.
pub(crate) fn support_columns(x: f32, sigma: f32, cols: usize) -> Option<(usize, usize)> {
    if !x.is_finite() || !sigma.is_finite() || sigma <= 1e-6 || cols < 3 {
        return None;
    }
    let reach = -START * sigma * SQRT2;
    if x + reach < 0.0 || x - reach > (cols - 1) as f32 {
        return None;
    }
    Some(span(x, reach + 1.0, cols))
}

/// A rectangle of a column-major Gaussian level.
///
/// Column `c` of the rectangle starts at `(c − first column) · stride`; the
/// kernel reads only inside it, and clamps and validates positions against
/// the whole level (`rows × cols`).
pub(crate) struct LevelWindow<'a> {
    data: &'a [f32],
    rect: Footprint,
    stride: usize,
    /// Rows of the whole level.
    rows: usize,
    /// Columns of the whole level.
    cols: usize,
}

impl<'a> LevelWindow<'a> {
    /// A whole `rows × cols` level stored column-major.
    #[cfg(test)]
    pub(crate) fn full(data: &'a [f32], rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0 && cols > 0 && data.len() >= rows * cols);
        Self {
            data,
            rect: Footprint {
                rows: (0, rows - 1),
                cols: (0, cols - 1),
            },
            stride: rows,
            rows,
            cols,
        }
    }

    /// The rectangle `rect` of a `rows × cols` level, stored column-major
    /// with one column of `rect` rows after another.
    pub(crate) fn new(data: &'a [f32], rect: Footprint, rows: usize, cols: usize) -> Self {
        debug_assert!(rect.rows.1 < rows && rect.cols.1 < cols);
        debug_assert!(data.len() >= rect.height() * rect.width());
        Self {
            data,
            rect,
            stride: rect.height(),
            rows,
            cols,
        }
    }

    /// Bilinear read at a position inside `[0, cols − 1] × [0, rows − 1]`.
    #[inline(always)]
    fn bilinear(&self, px: f32, py: f32) -> f32 {
        let col = (px as usize).min(self.cols - 1);
        let row = (py as usize).min(self.rows - 1);
        let fx = px - col as f32;
        let fy = py - row as f32;
        let right = if col + 1 < self.cols { self.stride } else { 0 };
        let below = usize::from(row + 1 < self.rows);
        let at = (col - self.rect.cols.0) * self.stride + (row - self.rect.rows.0);
        let (top_left, bottom_left) = (self.data[at], self.data[at + below]);
        let (top_right, bottom_right) = (self.data[at + right], self.data[at + right + below]);
        let top = top_left + fx * (top_right - top_left);
        let bottom = bottom_left + fx * (bottom_right - bottom_left);
        top + fy * (bottom - top)
    }
}

/// Pooling weights of one grid coordinate along one axis: the Gaussian
/// window factor times the bilinear weights of cells `cell` and `cell + 1`.
/// A cell outside the 4×4 grid gets weight zero on a valid neighbor index, so
/// every sample spreads over a fixed 2×2 block of cells.
#[derive(Clone, Copy, Debug)]
struct AxisWeights {
    cell: usize,
    weight: [f32; 2],
}

impl AxisWeights {
    fn new(index: usize) -> Self {
        let coordinate = FIRST + index as f32 * STEP;
        let window = expf(-(coordinate * coordinate) / (2.0 * WINDOW_SIGMA * WINDOW_SIGMA));
        let alpha = coordinate / CELL_WIDTH + 0.5 * (CELLS as f32 - 1.0);
        let lower = alpha.floor();
        let fraction = alpha - lower;
        let (low, high) = (window * (1.0 - fraction), window * fraction);
        match lower as i32 {
            cell if cell < 0 => Self {
                cell: 0,
                weight: [high, 0.0],
            },
            cell if cell >= CELLS as i32 - 1 => Self {
                cell: CELLS - 2,
                weight: [0.0, low],
            },
            cell => Self {
                cell: cell as usize,
                weight: [low, high],
            },
        }
    }
}

/// Canonical-grid RootSIFT kernel: the per-axis pooling weights of the grid.
#[derive(Clone, Debug)]
pub(crate) struct CanonicalKernel {
    axis: [AxisWeights; INNER],
}

impl Default for CanonicalKernel {
    fn default() -> Self {
        Self {
            axis: core::array::from_fn(AxisWeights::new),
        }
    }
}

impl CanonicalKernel {
    /// RootSIFT descriptor of `frame` read from `window`, which must contain
    /// the frame's footprint.
    ///
    /// Returns `None` for a frame without a footprint and for a degenerate or
    /// non-finite patch.
    pub(crate) fn describe(
        &self,
        window: &LevelWindow<'_>,
        frame: &Frame,
    ) -> Option<[f32; SIFT_DESCR_SIZE]> {
        let mut histogram = self.accumulate(window, frame, true)?;
        normalize_and_finalize_descriptor(&mut histogram, true)
    }

    /// Raw 4×4×8 histogram of `frame`. `allow_interior == false` forces the
    /// clamped, per-sample validated path that frames near the border take.
    fn accumulate(
        &self,
        window: &LevelWindow<'_>,
        frame: &Frame,
        allow_interior: bool,
    ) -> Option<[f32; SIFT_DESCR_SIZE]> {
        let footprint = frame.footprint(window.rows, window.cols)?;
        debug_assert!(window.rect.covers(&footprint));
        let Frame {
            x,
            y,
            sigma,
            cos,
            sin,
        } = *frame;
        let max_col = (window.cols - 1) as f32;
        let max_row = (window.rows - 1) as f32;
        // Image displacement of one grid step along e1 (u) and e2 (v).
        let (du_x, du_y) = (sigma * STEP * cos, sigma * STEP * sin);
        let (dv_x, dv_y) = (-sigma * STEP * sin, sigma * STEP * cos);
        let origin_x = x + sigma * START * (cos - sin);
        let origin_y = y + sigma * START * (sin + cos);
        // Every sample, ring included, lies in the level interior: skip all
        // per-sample clamping and validity tests.
        let reach = frame.reach();
        let interior = allow_interior
            && x - reach >= 1.0
            && y - reach >= 1.0
            && x + reach <= max_col - 1.0
            && y + reach <= max_row - 1.0;

        let mut values = [0.0f32; SIDE * SIDE];
        for (kv, line) in values.as_chunks_mut::<SIDE>().0.iter_mut().enumerate() {
            let line_x = origin_x + kv as f32 * dv_x;
            let line_y = origin_y + kv as f32 * dv_y;
            for (ku, value) in line.iter_mut().enumerate() {
                let mut px = line_x + ku as f32 * du_x;
                let mut py = line_y + ku as f32 * du_y;
                if !interior {
                    px = px.clamp(0.0, max_col);
                    py = py.clamp(0.0, max_row);
                }
                *value = window.bilinear(px, py);
            }
        }

        let mut histogram = [0.0f32; SIFT_DESCR_SIZE];
        for (kv, v) in self.axis.iter().enumerate() {
            for (ku, u) in self.axis.iter().enumerate() {
                if !interior {
                    // Like the classic kernel, a sample outside the one-pixel
                    // interior has no gradient.
                    let px = origin_x + (ku + 1) as f32 * du_x + (kv + 1) as f32 * dv_x;
                    let py = origin_y + (ku + 1) as f32 * du_y + (kv + 1) as f32 * dv_y;
                    let inside =
                        px >= 1.0 && py >= 1.0 && px <= max_col - 1.0 && py <= max_row - 1.0;
                    if !inside {
                        continue;
                    }
                }
                let center = (kv + 1) * SIDE + ku + 1;
                let gu = 0.5 * (values[center + 1] - values[center - 1]);
                let gv = 0.5 * (values[center + SIDE] - values[center - SIDE]);
                let magnitude_sq = gu * gu + gv * gv;
                if magnitude_sq < 1e-12 {
                    continue;
                }
                let magnitude = magnitude_sq.sqrt();
                let mut angle = fast_atan2(gv, gu);
                if angle < 0.0 {
                    angle += TWO_PI;
                }
                let gamma = angle * ANGLE_TO_BIN;
                if !gamma.is_finite() {
                    continue;
                }
                // `gamma` lies in [0, 8], where truncation is the floor (a
                // conversion instead of a `floor` call on targets without
                // SSE4.1).
                let bin = gamma as usize;
                let fraction = gamma - bin as f32;
                let k_left = bin % ORIENTATION_BINS;
                let k_right = (k_left + 1) % ORIENTATION_BINS;
                let left = magnitude * (1.0 - fraction);
                let right = magnitude * fraction;
                for (di, &weight_u) in u.weight.iter().enumerate() {
                    for (dj, &weight_v) in v.weight.iter().enumerate() {
                        let base = ((u.cell + di) * CELLS + v.cell + dj) * ORIENTATION_BINS;
                        let weight = weight_u * weight_v;
                        histogram[base + k_left] += weight * left;
                        histogram[base + k_right] += weight * right;
                    }
                }
            }
        }
        Some(histogram)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::sift::common::{
        SiftDescriptorScratch, compute_sift_descriptor_from_gradients_with_scratch,
    };
    use core::f32::consts::{FRAC_PI_2, PI};

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    /// Column-major test level.
    struct Level {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    }

    impl Level {
        fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Self {
            let mut data = Vec::with_capacity(rows * cols);
            for col in 0..cols {
                for row in 0..rows {
                    data.push(f(row, col));
                }
            }
            Self { data, rows, cols }
        }

        fn at(&self, row: usize, col: usize) -> f32 {
            self.data[col * self.rows + row]
        }

        fn window(&self) -> LevelWindow<'_> {
            LevelWindow::full(&self.data, self.rows, self.cols)
        }

        /// `(1 − t) low + t high`, the blend a streaming scale space gathers
        /// (a copy of `low` for `t == 0`).
        fn blend(&self, high: &Self, t: f32) -> Self {
            let keep = 1.0 - t;
            Self {
                data: self
                    .data
                    .iter()
                    .zip(&high.data)
                    .map(|(&low, &high)| if t == 0.0 { low } else { keep * low + t * high })
                    .collect(),
                rows: self.rows,
                cols: self.cols,
            }
        }

        /// The footprint rectangle, stored column after column.
        fn crop(&self, rect: Footprint) -> Vec<f32> {
            let mut out = Vec::with_capacity(rect.height() * rect.width());
            for col in rect.cols.0..=rect.cols.1 {
                out.extend_from_slice(
                    &self.data[col * self.rows + rect.rows.0..=col * self.rows + rect.rows.1],
                );
            }
            out
        }
    }

    fn textured(rows: usize, cols: usize, phase: f32) -> Level {
        Level::from_fn(rows, cols, |row, col| {
            let (r, c) = (row as f32, col as f32);
            128.0
                + 60.0 * (0.21 * c + 0.05 * r + phase).sin() * (0.17 * r - 0.03 * c).cos()
                + 40.0 * (0.0047 * c * r + phase).sin()
                + 0.3 * c
        })
    }

    /// Keypoint as `(x, y, sigma, orientation)`.
    type Key = (f32, f32, f32, f32);

    fn frame((x, y, sigma, orientation): Key) -> Frame {
        Frame::new(x, y, sigma, orientation).unwrap()
    }

    /// Keypoints well inside a 120×140 level, over the σ range of a describe
    /// pass and assorted orientations.
    fn interior_keys() -> Vec<Key> {
        let mut keys = Vec::new();
        for (k, sigma) in [1.6f32, 2.1, 2.9, 3.7, 4.5].into_iter().enumerate() {
            for (j, orientation) in [0.0f32, 0.37, FRAC_PI_2, -2.4, PI, 3.0]
                .into_iter()
                .enumerate()
            {
                let x = 66.3 + 1.7 * k as f32 - 0.9 * j as f32;
                let y = 58.8 - 1.1 * k as f32 + 0.6 * j as f32;
                keys.push((x, y, sigma, orientation));
            }
        }
        keys
    }

    /// Keypoints whose samples cross the border of a 120×140 level, some of
    /// them partly outside it.
    fn border_keys() -> Vec<Key> {
        let mut keys = Vec::new();
        for (x, y) in [
            (2.0, 60.0),
            (137.5, 61.2),
            (70.0, 1.3),
            (69.4, 118.9),
            (0.0, 0.0),
            (139.0, 119.0),
            (-6.0, 50.0),
            (75.0, 124.0),
            (10.5, 9.7),
        ] {
            for (sigma, orientation) in [(1.6, 0.0), (2.6, 0.9), (4.4, -2.2)] {
                keys.push((x, y, sigma, orientation));
            }
        }
        keys
    }

    fn all_keys() -> Vec<Key> {
        interior_keys().into_iter().chain(border_keys()).collect()
    }

    fn bits(descriptor: Option<[f32; SIFT_DESCR_SIZE]>) -> Option<[u32; SIFT_DESCR_SIZE]> {
        descriptor.map(|values| values.map(f32::to_bits))
    }

    /// Straight transcription of the canonical-grid kernel as first written:
    /// per-corner blend of two levels, two-dimensional Gaussian window, and
    /// explicit pooling entries per sample.
    fn reference_histogram(
        low: &Level,
        high: &Level,
        t: f32,
        (x, y, sigma, orientation): Key,
    ) -> [f32; SIFT_DESCR_SIZE] {
        let (rows, cols) = (low.rows, low.cols);
        let (sin_t, cos_t) = (sinf(orientation), cosf(orientation));
        let max_col = (cols - 1) as f32;
        let max_row = (rows - 1) as f32;
        let (du_x, du_y) = (sigma * STEP * cos_t, sigma * STEP * sin_t);
        let (dv_x, dv_y) = (-sigma * STEP * sin_t, sigma * STEP * cos_t);
        let origin_x = x + sigma * START * (cos_t - sin_t);
        let origin_y = y + sigma * START * (sin_t + cos_t);
        let reach = -START * sigma * (cos_t.abs() + sin_t.abs());
        let interior = x - reach >= 1.0
            && y - reach >= 1.0
            && x + reach <= max_col - 1.0
            && y + reach <= max_row - 1.0;
        let read = |px: f32, py: f32| {
            let col0 = (px as usize).min(cols - 1);
            let row0 = (py as usize).min(rows - 1);
            let col1 = (col0 + 1).min(cols - 1);
            let row1 = (row0 + 1).min(rows - 1);
            let fx = px - col0 as f32;
            let fy = py - row0 as f32;
            let corner = |row, col| {
                if t == 0.0 {
                    low.at(row, col)
                } else {
                    (1.0 - t) * low.at(row, col) + t * high.at(row, col)
                }
            };
            let (top_left, top_right) = (corner(row0, col0), corner(row0, col1));
            let (bottom_left, bottom_right) = (corner(row1, col0), corner(row1, col1));
            let top = top_left + fx * (top_right - top_left);
            let bottom = bottom_left + fx * (bottom_right - bottom_left);
            top + fy * (bottom - top)
        };
        let mut values = vec![0.0f32; SIDE * SIDE];
        for kv in 0..SIDE {
            let line_x = origin_x + kv as f32 * dv_x;
            let line_y = origin_y + kv as f32 * dv_y;
            for ku in 0..SIDE {
                let mut px = line_x + ku as f32 * du_x;
                let mut py = line_y + ku as f32 * du_y;
                if !interior {
                    px = px.clamp(0.0, max_col);
                    py = py.clamp(0.0, max_row);
                }
                values[kv * SIDE + ku] = read(px, py);
            }
        }
        let mut histogram = [0.0f32; SIFT_DESCR_SIZE];
        for kv in 0..INNER {
            for ku in 0..INNER {
                if !interior {
                    let px = origin_x + (ku + 1) as f32 * du_x + (kv + 1) as f32 * dv_x;
                    let py = origin_y + (ku + 1) as f32 * du_y + (kv + 1) as f32 * dv_y;
                    let inside =
                        px >= 1.0 && py >= 1.0 && px <= max_col - 1.0 && py <= max_row - 1.0;
                    if !inside {
                        continue;
                    }
                }
                let center = (kv + 1) * SIDE + ku + 1;
                let gu = 0.5 * (values[center + 1] - values[center - 1]);
                let gv = 0.5 * (values[center + SIDE] - values[center - SIDE]);
                let magnitude_sq = gu * gu + gv * gv;
                if magnitude_sq < 1e-12 {
                    continue;
                }
                let magnitude = magnitude_sq.sqrt();
                let mut angle = fast_atan2(gv, gu);
                if angle < 0.0 {
                    angle += TWO_PI;
                }
                let gamma = angle * ANGLE_TO_BIN;
                if !gamma.is_finite() {
                    continue;
                }
                let gamma_floor = gamma.floor();
                let fraction = gamma - gamma_floor;
                let k_left = gamma_floor as usize % ORIENTATION_BINS;
                let k_right = (k_left + 1) % ORIENTATION_BINS;
                let (left, right) = (magnitude * (1.0 - fraction), magnitude * fraction);

                let u = FIRST + ku as f32 * STEP;
                let v = FIRST + kv as f32 * STEP;
                let gauss = expf(-(u * u + v * v) / (2.0 * WINDOW_SIGMA * WINDOW_SIGMA));
                let alpha = u / CELL_WIDTH + 1.5;
                let beta = v / CELL_WIDTH + 1.5;
                let (alpha_floor, beta_floor) = (alpha.floor(), beta.floor());
                let (fa, fb) = (alpha - alpha_floor, beta - beta_floor);
                let (i0, j0) = (alpha_floor as i32, beta_floor as i32);
                for (di, wi) in [(0, 1.0 - fa), (1, fa)] {
                    for (dj, wj) in [(0, 1.0 - fb), (1, fb)] {
                        let (i, j) = (i0 + di, j0 + dj);
                        let in_grid =
                            (0..CELLS as i32).contains(&i) && (0..CELLS as i32).contains(&j);
                        if in_grid && wi * wj > 0.0 {
                            let weight = gauss * (wi * wj);
                            let base = (i as usize * CELLS + j as usize) * ORIENTATION_BINS;
                            histogram[base + k_left] += weight * left;
                            histogram[base + k_right] += weight * right;
                        }
                    }
                }
            }
        }
        histogram
    }

    fn assert_histograms_close(actual: &[f32], expected: &[f32]) {
        for (bin, (a, b)) in actual.iter().zip(expected).enumerate() {
            let scale = a.abs().max(b.abs()).max(1e-6);
            assert!((a - b).abs() <= 1e-5 * scale, "bin {bin}: {a} vs {b}");
        }
    }

    fn distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn grid_constants_are_exact() {
        assert_eq!(STEP, 0.625);
        assert_eq!(FIRST, -7.1875);
        assert_eq!(START, -7.8125);
    }

    #[test]
    fn separable_pooling_tracks_the_reference_kernel() {
        let low = textured(120, 140, 0.0);
        let high = textured(120, 140, 0.8);
        let kernel = CanonicalKernel::default();
        let mut compared = 0;
        for t in [0.0f32, 0.35] {
            let blended = low.blend(&high, t);
            for key in all_keys() {
                let expected = reference_histogram(&low, &high, t, key);
                match kernel.accumulate(&blended.window(), &frame(key), true) {
                    Some(actual) => {
                        assert_histograms_close(&actual, &expected);
                        compared += 1;
                    }
                    None => assert!(expected.iter().all(|&bin| bin == 0.0), "{key:?}"),
                }
            }
        }
        assert!(compared > 100, "too few histograms: {compared}");
    }

    #[test]
    fn cropped_window_matches_the_full_level_bit_for_bit() {
        let level = textured(120, 140, 0.4);
        let kernel = CanonicalKernel::default();
        let mut described = 0;
        for key in all_keys() {
            let f = frame(key);
            let full = bits(kernel.describe(&level.window(), &f));
            let Some(rect) = f.footprint(level.rows, level.cols) else {
                assert!(full.is_none());
                continue;
            };
            // Poison everything outside the footprint: a read there would
            // turn the patch non-finite and the kernel would reject it.
            let guarded = Level::from_fn(level.rows, level.cols, |row, col| {
                let inside = (rect.rows.0..=rect.rows.1).contains(&row)
                    && (rect.cols.0..=rect.cols.1).contains(&col);
                if inside { level.at(row, col) } else { f32::NAN }
            });
            let cropped = level.crop(rect);
            let window = LevelWindow::new(&cropped, rect, level.rows, level.cols);
            assert_eq!(bits(kernel.describe(&window, &f)), full, "{key:?}");
            assert_eq!(
                bits(kernel.describe(&guarded.window(), &f)),
                full,
                "{key:?}"
            );
            described += usize::from(full.is_some());
        }
        // More than the interior keys alone: border keys are described too.
        assert!(
            described > interior_keys().len(),
            "too few descriptors: {described}"
        );
    }

    #[test]
    fn interior_frames_skip_the_border_path_without_changing_bits() {
        let level = textured(120, 140, 1.1);
        let (max_col, max_row) = ((level.cols - 1) as f32, (level.rows - 1) as f32);
        let kernel = CanonicalKernel::default();
        for key in interior_keys() {
            let f = frame(key);
            let reach = f.reach();
            assert!(
                f.x - reach >= 1.0 && f.x + reach <= max_col - 1.0,
                "{key:?}"
            );
            assert!(
                f.y - reach >= 1.0 && f.y + reach <= max_row - 1.0,
                "{key:?}"
            );
            let fast = kernel.accumulate(&level.window(), &f, true).unwrap();
            let checked = kernel.accumulate(&level.window(), &f, false).unwrap();
            assert_eq!(fast.map(f32::to_bits), checked.map(f32::to_bits), "{key:?}");
        }
    }

    #[test]
    fn rotating_the_image_rotates_the_frame() {
        // Rotating the image by 90° and the orientation by the same angle
        // leaves the canonical patch, and hence the descriptor, nearly
        // unchanged (bilinear reads coincide on the rotated lattice).
        let image = textured(81, 81, 0.2);
        let rotated = Level::from_fn(81, 81, |row, col| image.at(col, 80 - row));
        let kernel = CanonicalKernel::default();
        let a = kernel
            .describe(&image.window(), &frame((40.0, 40.0, 2.0, 0.3)))
            .unwrap();
        let b = kernel
            .describe(
                &rotated.window(),
                &frame((40.0, 40.0, 2.0, 0.3 - FRAC_PI_2)),
            )
            .unwrap();
        let d = distance(&a, &b);
        assert!(d < 0.05 * 512.0, "distance {d}");
    }

    #[test]
    fn descriptor_is_unit_rootsift_scaled_to_512() {
        let level = textured(120, 140, 0.0);
        let kernel = CanonicalKernel::default();
        for key in interior_keys() {
            let descriptor = kernel.describe(&level.window(), &frame(key)).unwrap();
            let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 512.0).abs() < 1.0 || descriptor.contains(&255.0));
            assert!(descriptor.iter().all(|&v| (0.0..=255.0).contains(&v)));
        }
    }

    #[test]
    fn layout_matches_the_classic_kernel() {
        // Same cells, bins and orientation convention: the two kernels sample
        // the patch differently but must describe it alike.
        let level = textured(120, 140, 0.6);
        let (rows, cols) = (level.rows, level.cols);
        let mut ix = vec![0.0f32; rows * cols];
        let mut iy = vec![0.0f32; rows * cols];
        for row in 1..rows - 1 {
            for col in 1..cols - 1 {
                ix[row * cols + col] = 0.5 * (level.at(row, col + 1) - level.at(row, col - 1));
                iy[row * cols + col] = 0.5 * (level.at(row + 1, col) - level.at(row - 1, col));
            }
        }
        let kernel = CanonicalKernel::default();
        let mut scratch = SiftDescriptorScratch::default();
        for key in interior_keys() {
            let (x, y, sigma, orientation) = key;
            let canonical = kernel.describe(&level.window(), &frame(key)).unwrap();
            let classic = compute_sift_descriptor_from_gradients_with_scratch(
                x,
                y,
                sigma,
                orientation,
                true,
                &ix,
                &iy,
                rows,
                cols,
                &mut scratch,
            )
            .unwrap();
            let dot: f32 = canonical.iter().zip(&classic).map(|(a, b)| a * b).sum();
            let cosine = dot
                / (canonical.iter().map(|v| v * v).sum::<f32>().sqrt()
                    * classic.iter().map(|v| v * v).sum::<f32>().sqrt());
            assert!(cosine > 0.95, "{key:?}: cosine {cosine}");
        }
    }

    #[test]
    fn constant_level_has_no_descriptor() {
        let level = Level::from_fn(64, 64, |_, _| 0.5);
        let kernel = CanonicalKernel::default();
        assert!(
            kernel
                .describe(&level.window(), &frame((32.0, 32.0, 1.6, 0.0)))
                .is_none()
        );
    }

    #[test]
    fn non_finite_samples_have_no_descriptor() {
        let kernel = CanonicalKernel::default();
        let f = frame((60.0, 55.0, 2.0, 0.4));
        for poison in [f32::INFINITY, f32::NAN] {
            let level = textured(120, 140, 0.0);
            let mut data = level.data;
            // Column 60, row 55: the keypoint center.
            data[60 * 120 + 55] = poison;
            let window = LevelWindow::full(&data, 120, 140);
            assert!(kernel.describe(&window, &f).is_none(), "{poison}");
        }
    }

    #[test]
    fn invalid_frames_and_tiny_levels_have_no_descriptor() {
        for (x, y, sigma, orientation) in [
            (f32::NAN, 1.0, 2.0, 0.0),
            (1.0, f32::INFINITY, 2.0, 0.0),
            (1.0, 1.0, 0.0, 0.0),
            (1.0, 1.0, -2.0, 0.0),
            (1.0, 1.0, 2.0, f32::NAN),
        ] {
            assert!(Frame::new(x, y, sigma, orientation).is_none());
        }
        let f = frame((1.0, 1.0, 1.6, 0.0));
        assert!(f.footprint(2, 50).is_none());
        assert!(f.footprint(50, 2).is_none());
        let level = textured(120, 140, 0.0);
        let kernel = CanonicalKernel::default();
        for far in [
            frame((-200.0, -200.0, 1.6, 0.0)),
            frame((400.0, 60.0, 2.0, 0.5)),
            frame((70.0, -40.0, 3.0, 1.0)),
        ] {
            assert!(far.footprint(120, 140).is_none());
            assert!(kernel.describe(&level.window(), &far).is_none());
        }
        assert!(support_columns(f32::NAN, 2.0, 100).is_none());
        assert!(support_columns(10.0, 0.0, 100).is_none());
        assert!(support_columns(10.0, 2.0, 2).is_none());
        assert!(support_columns(-50.0, 2.0, 100).is_none());
        assert!(support_columns(150.0, 2.0, 100).is_none());
    }

    #[test]
    fn support_columns_cover_the_footprint_at_every_orientation() {
        let cols = 140;
        for step in 0..400 {
            let x = -30.0 + 0.53 * step as f32;
            let sigma = 1.6 + 0.0073 * step as f32;
            let span = support_columns(x, sigma, cols);
            for k in 0..64 {
                let orientation = -PI + TWO_PI * k as f32 / 64.0;
                if let Some(rect) = frame((x, 60.0, sigma, orientation)).footprint(120, cols) {
                    let (low, high) = span.expect("a footprint implies support columns");
                    assert!(low <= rect.cols.0 && rect.cols.1 <= high, "x {x} σ {sigma}");
                }
            }
        }
    }
}
