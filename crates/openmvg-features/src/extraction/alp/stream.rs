// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streaming ALP engine: bounded-memory detection and description passes.
//!
//! ALP images are column-major, so one image column is a contiguous line and
//! the engine streams over columns (the column-major analogue of a
//! row-streaming detector). Filtering retains decimated octave bases and
//! column rings rather than full Gaussian/LoG planes. Coverage arrays scale
//! with width; candidate, request, ordering and output storage scale with
//! their counts. Parallel workers retain their own filter scratch.
//!
//! **Pass 1 (detection)** keeps three-line rings per level: Gaussian lines
//! (for the five-point Laplacian), σ²·LoG lines (for the spatial fit), and
//! scale-envelope lines. For every pixel, raw end samples and interior
//! stationary points of the cubic fit are compared once (`emax`, `emin` and their locations).
//! The center envelope combines raw endpoint samples with typed critical values;
//! the octave owns an extremum only when an interior level carries its strongest
//! sample. A candidate must beat its eight neighbor cubics over one level step
//! (σ/√2 to σ·√2) around its scale, the continuous counterpart of SIFT's
//! 26-neighbor test. Scale and spatial curvature checks are applied separately.
//!
//! **Pass 2 (description)** runs only for the keypoints kept by the feature
//! budget. It re-streams the octave over the columns those keypoints cover,
//! in steps of [`DESCRIBE_STEP_COLUMNS`] columns, retaining a ring one step
//! wider than the largest descriptor support, and gives each keypoint its
//! dominant orientation and a canonical-grid descriptor in the step that
//! completes its support. With the `parallel` feature, the column blurs and
//! the keypoints of a step run on the Rayon pool; the output is identical for
//! every thread count.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;

use super::GrayImage;
use super::blur::{BaseImage, ColumnBlur, HalfKernel, LEVELS, octave_kernels};
use super::octave_dedup::{AdjacentOctavePreference, remove_adjacent_octave_duplicates_by};
use super::scale_fit::{
    CubicCoefficients, ORIENTATION_SCALE_SNAP, band_extreme, compute_poly_coeffs,
    corrected_sigma_from_anchor, horner, neighbor_scale_band, octave_count, octave_scale,
    orientation_histogram_bin, owns_scale_extremum, refine_keypoint_2d,
};
use super::tables::SIGMAS;
use crate::extraction::ScaleSpaceKeypoint;
use crate::extraction::feature_budget::{FeatureBudget, FeatureCandidate, selection_mask};
use crate::extraction::sift::canonical_descriptor::{
    self, CanonicalKernel, Footprint, Frame, LevelWindow,
};
use crate::extraction::sift::common::{
    ORI_HIST_BINS, ORI_SCALE, ORI_SMOOTH_PASSES, SIFT_DESCR_SIZE, dominant_orientation, fast_atan2,
    smooth_orientation_histogram,
};
use crate::math_compat::expf;

/// Spatial border (octave pixels) excluded from detection.
const DETECTION_BORDER: usize = 4;

/// Thresholds and scale window for pass 1, all in octave-local units.
#[derive(Clone, Copy, Debug)]
pub(super) struct DetectionParams {
    pub(super) peak_threshold: f32,
    pub(super) sigma_threshold: f32,
    pub(super) curv_ratio_threshold: f32,
    /// Accepted fitted scale `[low, high]` (octave-local σ).
    pub(super) sigma_window: (f32, f32),
}

/// One detected extremum before orientation (octave-local coordinates).
#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    /// Refined position, anchor σ, signed peak, octave, Q-fit bucket, polarity.
    pub(super) geometry: ScaleSpaceKeypoint,
    /// Row-major integer anchor order, independent of image storage order.
    pub(super) order_key: usize,
    /// Exact four LoG responses at the integer anchor, before spatial refinement.
    pub(super) scale_samples: [f32; LEVELS],
}

/// Compare raw end samples with interior stationary points of the cubic scale fit.
///
/// Returns `(emin, emax, smin, smax)`: the extreme candidate responses and the
/// interior stationary scale reaching them, or `0.0` when the extreme lies on
/// an end sample. The fitted curve can differ from the samples at the endpoints.
/// Written with selects so a column loop vectorizes.
#[inline(always)]
pub(super) fn scale_envelope(l: [f32; LEVELS]) -> (f32, f32, f32, f32) {
    let coeffs = compute_poly_coeffs(l[0], l[1], l[2], l[3]);
    let [a, b, c, _] = coeffs;
    let mut emin = l[0].min(l[3]);
    let mut emax = l[0].max(l[3]);
    let mut smin = 0.0f32;
    let mut smax = 0.0f32;

    let cubic = a.abs() > 1e-6;
    let quadratic = !cubic && b.abs() > 1e-6;
    let ap = 6.0 * a;
    let bp = 2.0 * b;
    let discriminant = bp * bp - 2.0 * ap * c;
    let root = discriminant.max(0.0).sqrt();
    let has_roots = cubic && discriminant > 0.0;
    let in_range = |s: f32| (SIGMAS[0]..=SIGMAS[3]).contains(&s);

    let first = if quadratic {
        -c / (2.0 * b)
    } else {
        (-bp + root) / ap
    };
    let first_ok = (has_roots || quadratic) && in_range(first);
    let second = (-bp - root) / ap;
    let second_ok = has_roots && in_range(second);

    let first_is_minimum = !quadratic || b > 0.0;
    for (s, ok, is_minimum) in [
        (first, first_ok, first_is_minimum),
        (second, second_ok, false),
    ] {
        let value = horner(coeffs, s);
        let curvature = ap * s + bp;
        let polarity_matches = if is_minimum {
            curvature > 0.0
        } else {
            curvature < 0.0
        };
        let valid = ok && value.is_finite() && curvature.is_finite() && polarity_matches;
        if valid && is_minimum && value < emin {
            emin = value;
            smin = s;
        }
        if valid && !is_minimum && value > emax {
            emax = value;
            smax = s;
        }
    }
    (emin, emax, smin, smax)
}

/// Three-slot ring of per-level lines; slot of line `j` is `j % 3`.
struct LineRing3 {
    lines: [Vec<f32>; LEVELS],
    rows: usize,
}

impl LineRing3 {
    fn new(rows: usize) -> Self {
        Self {
            lines: core::array::from_fn(|_| vec![0.0; 3 * rows]),
            rows,
        }
    }

    #[inline]
    fn line(&self, level: usize, j: usize) -> &[f32] {
        let at = (j % 3) * self.rows;
        &self.lines[level][at..at + self.rows]
    }
}

/// Pass-1 output for one octave.
pub(super) struct OctaveDetection {
    /// Candidates in row-major anchor order (minimum before maximum).
    pub(super) candidates: Vec<Candidate>,
    /// Column-major decimated `SIGMAS[2]` level, the next octave base.
    pub(super) next_base: Vec<f32>,
    /// Rows of `next_base`.
    pub(super) next_rows: usize,
    /// Columns of `next_base`.
    pub(super) next_cols: usize,
}

/// Pass 1 over one octave: stream all columns of `base`, detect, and decimate.
///
/// Filtering scratch is `O(rows)` apart from the requested next-octave base;
/// candidate records grow with the number of accepted extrema.
///
/// With `params == None` only the next-octave base is produced.
pub(super) fn detect_octave(
    base: BaseImage<'_>,
    octave: usize,
    params: Option<&DetectionParams>,
    want_next: bool,
) -> OctaveDetection {
    let (rows, cols) = (base.rows, base.cols);
    let (next_rows, next_cols) = if want_next {
        (rows / 2, cols / 2)
    } else {
        (0, 0)
    };
    let mut next_base = vec![0.0f32; next_rows * next_cols];
    let border = DETECTION_BORDER;
    let detect = params.is_some() && rows > 2 * border && cols > 2 * border;
    let mut candidates = Vec::new();
    if rows == 0 || cols == 0 || (!detect && next_base.is_empty()) {
        return OctaveDetection {
            candidates,
            next_base,
            next_rows,
            next_cols,
        };
    }

    let kernels = octave_kernels(octave);
    if !detect {
        let mut blur = ColumnBlur::new(rows);
        let mut level2 = vec![0.0f32; rows];
        for next_column in 0..next_cols {
            // Decimation selects outputs, not inputs: keep the full base image.
            let column = 2 * next_column;
            blur.blur_column_into(
                base,
                column,
                &kernels,
                [None, None, Some(level2.as_mut_slice()), None],
            );
            let start = next_column * next_rows;
            for (row, value) in next_base[start..start + next_rows].iter_mut().enumerate() {
                *value = level2[2 * row];
            }
        }
        return OctaveDetection {
            candidates,
            next_base,
            next_rows,
            next_cols,
        };
    }

    let mut blur = ColumnBlur::new(rows);
    let mut gauss = LineRing3::new(rows);
    let mut log = LineRing3::new(rows);
    // Envelope levels: 0 = emin, 1 = emax, 2 = smin, 3 = smax.
    let mut envelope = LineRing3::new(rows);
    let levels = if detect {
        [true; LEVELS]
    } else {
        [false, false, true, false]
    };

    for column in 0..cols {
        blur.blur_column(
            base,
            column,
            &kernels,
            levels,
            &mut gauss.lines,
            (column % 3) * rows,
        );
        if column % 2 == 0 && column / 2 < next_cols {
            let level2 = gauss.line(2, column);
            let start = (column / 2) * next_rows;
            for (row, value) in next_base[start..start + next_rows].iter_mut().enumerate() {
                *value = level2[2 * row];
            }
        }
        if !detect || column < 2 {
            continue;
        }
        let center = column - 1;
        laplacian_column(&gauss, center, &mut log);
        envelope_column(&log, center, &mut envelope);
        if center < 2 {
            continue;
        }
        let scan = center - 1;
        if let Some(params) = params.filter(|_| (border..cols - border).contains(&scan)) {
            scan_column(&log, &envelope, scan, cols, octave, params, &mut candidates);
        }
    }
    // Stable: minimum-before-maximum at one pixel is preserved.
    candidates.sort_by_key(|candidate| candidate.order_key);
    OctaveDetection {
        candidates,
        next_base,
        next_rows,
        next_cols,
    }
}

/// σ²-normalized five-point Laplacian of Gaussian column `center` (all levels).
fn laplacian_column(gauss: &LineRing3, center: usize, log: &mut LineRing3) {
    let rows = gauss.rows;
    let slot = (center % 3) * rows;
    for (level, &sigma) in SIGMAS.iter().enumerate().take(LEVELS) {
        let left = gauss.line(level, center - 1);
        let middle = gauss.line(level, center);
        let right = gauss.line(level, center + 1);
        let sigma_sq = sigma * sigma;
        let out = &mut log.lines[level][slot..slot + rows];
        out[0] = 0.0;
        out[rows - 1] = 0.0;
        for row in 1..rows - 1 {
            out[row] = (left[row] + right[row] + middle[row - 1] + middle[row + 1]
                - 4.0 * middle[row])
                * sigma_sq;
        }
    }
}

/// Scale envelope of LoG column `center` into the envelope ring.
fn envelope_column(log: &LineRing3, center: usize, envelope: &mut LineRing3) {
    let rows = log.rows;
    let slot = (center % 3) * rows;
    let l0 = log.line(0, center);
    let l1 = log.line(1, center);
    let l2 = log.line(2, center);
    let l3 = log.line(3, center);
    let [emin, emax, smin, smax] = &mut envelope.lines;
    let emin = &mut emin[slot..slot + rows];
    let emax = &mut emax[slot..slot + rows];
    let smin = &mut smin[slot..slot + rows];
    let smax = &mut smax[slot..slot + rows];
    for row in 0..rows {
        let (lo, hi, s_lo, s_hi) = scale_envelope([l0[row], l1[row], l2[row], l3[row]]);
        emin[row] = lo;
        emax[row] = hi;
        smin[row] = s_lo;
        smax[row] = s_hi;
    }
}

/// Candidate scan of column `scan`; LoG and envelope rings hold `scan ± 1`.
fn scan_column(
    log: &LineRing3,
    envelope: &LineRing3,
    scan: usize,
    cols: usize,
    octave: usize,
    params: &DetectionParams,
    out: &mut Vec<Candidate>,
) {
    let rows = log.rows;
    let border = DETECTION_BORDER;
    let (low, high) = params.sigma_window;
    let peak = params.peak_threshold;
    let emin = envelope.line(0, scan);
    let emax = envelope.line(1, scan);
    let smin = envelope.line(2, scan);
    let smax = envelope.line(3, scan);
    for row in border..rows - border {
        let min_ok = smin[row] >= low && smin[row] <= high && emin[row] < -peak;
        let max_ok = smax[row] >= low && smax[row] <= high && emax[row] > peak;
        if !(min_ok || max_ok) {
            continue;
        }
        let mut found = [(0.0f32, 0.0f32, false); 2];
        let mut count = 0;
        if min_ok {
            found[count] = (smin[row], emin[row], false);
            count += 1;
        }
        if max_ok {
            found[count] = (smax[row], emax[row], true);
            count += 1;
        }
        refine_candidates(log, row, scan, cols, octave, params, &found[..count], out);
    }
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    static SPATIAL_NEIGHBOR_CUBICS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Scale ownership, stability guards, spatial competition within one level
/// step of the anchor scale, and refinement.
#[allow(clippy::too_many_arguments)]
fn refine_candidates(
    log: &LineRing3,
    row: usize,
    col: usize,
    cols: usize,
    octave: usize,
    params: &DetectionParams,
    found: &[(f32, f32, bool)],
    out: &mut Vec<Candidate>,
) {
    let samples: [f32; LEVELS] = core::array::from_fn(|level| log.line(level, col)[row]);
    let center = compute_poly_coeffs(samples[0], samples[1], samples[2], samples[3]);
    let [a, b, _, _] = center;
    let mut coeffs_cache: [CubicCoefficients; 9] = [[0.0f32; 4]; 9];
    coeffs_cache[4] = center;
    let mut ready_mask = 1u16 << 4;
    for &(sigma, peak, is_max) in found {
        if !owns_scale_extremum(&samples, is_max) {
            continue;
        }
        let curvature = 6.0 * a * sigma + 2.0 * b;
        let polarity_matches = if is_max {
            curvature < 0.0
        } else {
            curvature > 0.0
        };
        if !curvature.is_finite()
            || !polarity_matches
            || curvature.abs().partial_cmp(&params.sigma_threshold)
                != Some(core::cmp::Ordering::Greater)
        {
            continue;
        }
        // The center stationary response must beat every neighbor cubic over
        // one level step around its scale.
        let (low, high) = neighbor_scale_band(sigma);
        let is_extremum = (0..9).filter(|&index| index != 4).all(|index| {
            let bit = 1u16 << index;
            if ready_mask & bit == 0 {
                let (r, c) = (row + index / 3 - 1, col + index % 3 - 1);
                let samples: [f32; LEVELS] = core::array::from_fn(|level| log.line(level, c)[r]);
                coeffs_cache[index] =
                    compute_poly_coeffs(samples[0], samples[1], samples[2], samples[3]);
                ready_mask |= bit;
                #[cfg(all(test, feature = "std"))]
                SPATIAL_NEIGHBOR_CUBICS.with(|count| count.set(count.get() + 1));
            }
            let neighbor = band_extreme(coeffs_cache[index], low, high, is_max);
            if is_max {
                peak > neighbor
            } else {
                peak < neighbor
            }
        });
        if !is_extremum {
            continue;
        }
        // Passing visited all eight neighbors, so Q reads a complete real cache.
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
        if refine_keypoint_2d(&mut geometry, &coeffs_cache, params.curv_ratio_threshold) {
            out.push(Candidate {
                geometry,
                order_key: row * cols + col,
                scale_samples: samples,
            });
        }
    }
}

// ============================================================================
// Pass 2: orientation and description of kept keypoints
// ============================================================================

/// Linear blend between two Gaussian levels for a continuous σ.
#[derive(Clone, Copy, Debug)]
struct LevelBlend {
    low: usize,
    high: usize,
    t: f32,
}

impl LevelBlend {
    /// Bracketing levels in log σ; `t` within `snap` of a node snaps to it.
    fn new(sigma: f32, snap: f32) -> Self {
        let low = if sigma <= SIGMAS[1] {
            0
        } else if sigma <= SIGMAS[2] {
            1
        } else {
            2
        };
        let high = low + 1;
        let (sigma_low, sigma_high) = (SIGMAS[low], SIGMAS[high]);
        let t =
            ((sigma.ln() - sigma_low.ln()) / (sigma_high.ln() - sigma_low.ln())).clamp(0.0, 1.0);
        if t == 0.0 || t < snap {
            Self {
                low,
                high: low,
                t: 0.0,
            }
        } else if t == 1.0 || t > 1.0 - snap {
            Self {
                low: high,
                high,
                t: 0.0,
            }
        } else {
            Self { low, high, t }
        }
    }
}

/// Ring of Gaussian columns per level; column `c` lives in slot `c % capacity`.
struct ColumnRing {
    lines: [Vec<f32>; LEVELS],
    rows: usize,
    capacity: usize,
}

impl ColumnRing {
    #[inline]
    fn line(&self, level: usize, col: usize) -> &[f32] {
        let at = (col % self.capacity) * self.rows;
        &self.lines[level][at..at + self.rows]
    }

    /// Interpolated central-difference gradients of interior column `col`,
    /// rows `row_start..row_end`, passed to `visit(row, gx, gy)` in row order.
    #[inline]
    fn for_each_gradient(
        &self,
        blend: LevelBlend,
        col: usize,
        row_start: usize,
        row_end: usize,
        mut visit: impl FnMut(usize, f32, f32),
    ) {
        let (left, mid, right) = (
            self.line(blend.low, col - 1),
            self.line(blend.low, col),
            self.line(blend.low, col + 1),
        );
        if blend.t == 0.0 {
            for row in row_start..row_end {
                visit(
                    row,
                    0.5 * (right[row] - left[row]),
                    0.5 * (mid[row + 1] - mid[row - 1]),
                );
            }
            return;
        }
        let (left_h, mid_h, right_h) = (
            self.line(blend.high, col - 1),
            self.line(blend.high, col),
            self.line(blend.high, col + 1),
        );
        let t = blend.t;
        for row in row_start..row_end {
            let gx_low = 0.5 * (right[row] - left[row]);
            let gy_low = 0.5 * (mid[row + 1] - mid[row - 1]);
            let gx_high = 0.5 * (right_h[row] - left_h[row]);
            let gy_high = 0.5 * (mid_h[row + 1] - mid_h[row - 1]);
            visit(
                row,
                (1.0 - t) * gx_low + t * gx_high,
                (1.0 - t) * gy_low + t * gy_high,
            );
        }
    }

    /// The `blend` of two levels over `rect`, written to `out` one column of
    /// `rect` rows after another.
    fn gather(&self, blend: LevelBlend, rect: Footprint, out: &mut Vec<f32>) {
        let rows = rect.rows.0..=rect.rows.1;
        out.clear();
        for col in rect.cols.0..=rect.cols.1 {
            let low = &self.line(blend.low, col)[rows.clone()];
            if blend.t == 0.0 {
                out.extend_from_slice(low);
            } else {
                let high = &self.line(blend.high, col)[rows.clone()];
                let (keep, t) = (1.0 - blend.t, blend.t);
                out.extend(low.iter().zip(high).map(|(&l, &h)| keep * l + t * h));
            }
        }
    }
}

/// Largest orientation window diameter: radius `ceil(3 · ORI_SCALE · σ)` for
/// σ up to `SIGMAS[3]`, with margin.
const MAX_ORIENTATION_DIAMETER: usize = 2 * 24 + 1;

/// Gaussian-weighted 36-bin gradient-orientation histogram (Lowe 2004 §5).
fn orientation_histogram(
    ring: &ColumnRing,
    x: f32,
    y: f32,
    sigma: f32,
    cols: usize,
) -> [f32; ORI_HIST_BINS] {
    let mut histogram = [0.0f32; ORI_HIST_BINS];
    let rows = ring.rows;
    if sigma < 1e-6 || rows < 3 || cols < 3 {
        return histogram;
    }
    let blend = LevelBlend::new(sigma, ORIENTATION_SCALE_SNAP);
    let window_sigma = ORI_SCALE * sigma;
    let radius =
        ((3.0 * window_sigma).ceil() as i32).min((MAX_ORIENTATION_DIAMETER as i32 - 1) / 2);
    let inv_2sigma_sq = 1.0 / (2.0 * window_sigma * window_sigma);
    let (center_row, center_col) = (y.round() as i32, x.round() as i32);
    let row_min = (center_row - radius).max(1);
    let row_max = (center_row + radius).min(rows as i32 - 2);
    let col_min = (center_col - radius).max(1);
    let col_max = (center_col + radius).min(cols as i32 - 2);
    if row_min > row_max || col_min > col_max {
        return histogram;
    }
    let mut row_weights = [0.0f32; MAX_ORIENTATION_DIAMETER];
    for (offset, row) in (row_min..=row_max).enumerate() {
        let dy = row as f32 - y;
        row_weights[offset] = expf(-(dy * dy) * inv_2sigma_sq);
    }
    let row_start = row_min as usize;
    for col in col_min as usize..=col_max as usize {
        let dx = col as f32 - x;
        let col_weight = expf(-(dx * dx) * inv_2sigma_sq);
        ring.for_each_gradient(
            blend,
            col,
            row_start,
            row_max as usize + 1,
            |row, gx, gy| {
                let magnitude = (gx * gx + gy * gy).sqrt();
                let weight = row_weights[row - row_start] * col_weight;
                histogram[orientation_histogram_bin(fast_atan2(gy, gx))] += magnitude * weight;
            },
        );
    }
    histogram
}

/// Descriptor of `request` at `orientation` from the ring.
///
/// The level blend over the frame's footprint is gathered column by column
/// (contiguous ring reads) into `window`, which the kernel samples.
fn describe_from_ring(
    kernel: &CanonicalKernel,
    window: &mut Vec<f32>,
    ring: &ColumnRing,
    request: &DescribeRequest,
    orientation: f32,
    cols: usize,
) -> Option<[f32; SIFT_DESCR_SIZE]> {
    let frame = Frame::new(request.x, request.y, request.sigma, orientation)?;
    let footprint = frame.footprint(ring.rows, cols)?;
    ring.gather(LevelBlend::new(request.sigma, 0.0), footprint, window);
    let window = LevelWindow::new(window, footprint, ring.rows, cols);
    kernel.describe(&window, &frame)
}

/// Pass-2 request in octave-local coordinates.
#[derive(Clone, Copy, Debug)]
pub(super) struct DescribeRequest {
    pub(super) x: f32,
    pub(super) y: f32,
    /// Support σ for orientation and description.
    pub(super) sigma: f32,
    /// `Some` describes at this orientation; `None` assigns orientations.
    pub(super) orientation: Option<f32>,
}

/// Pass-2 output: at most one per request, ordered by `request`.
pub(super) struct Described<D = [f32; SIFT_DESCR_SIZE]> {
    pub(super) request: usize,
    pub(super) orientation: f32,
    /// `None` when descriptors were not requested or the support is invalid.
    pub(super) descriptor: Option<D>,
}

/// Inclusive ring columns read for `request`, or `None` when nothing is read.
fn support_columns(
    request: &DescribeRequest,
    rows: usize,
    cols: usize,
    describe: bool,
) -> (Option<(usize, usize)>, [bool; LEVELS]) {
    let mut span: Option<(usize, usize)> = None;
    let mut levels = [false; LEVELS];
    let mut extend = |lo: usize, hi: usize, blend: LevelBlend, levels: &mut [bool; LEVELS]| {
        span = Some(span.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
        levels[blend.low] = true;
        levels[blend.high] = true;
    };
    if request.orientation.is_none() && request.sigma >= 1e-6 {
        let radius = ((3.0 * ORI_SCALE * request.sigma).ceil() as i32)
            .min((MAX_ORIENTATION_DIAMETER as i32 - 1) / 2);
        let center = request.x.round() as i32;
        let (lo, hi) = (
            (center - radius).max(1),
            (center + radius).min(cols as i32 - 2),
        );
        let rows_ok = (request.y.round() as i32 - radius).max(1)
            <= (request.y.round() as i32 + radius).min(rows as i32 - 2);
        if lo <= hi && rows_ok {
            let blend = LevelBlend::new(request.sigma, ORIENTATION_SCALE_SNAP);
            extend(lo as usize - 1, hi as usize + 1, blend, &mut levels);
        }
    }
    if describe
        && let Some((lo, hi)) =
            canonical_descriptor::support_columns(request.x, request.sigma, cols)
    {
        extend(lo, hi, LevelBlend::new(request.sigma, 0.0), &mut levels);
    }
    (span, levels)
}

/// Columns per pass-2 step: each step blurs this many ring columns, then
/// describes every request whose support ends among them.
///
/// The ring retains this many columns beyond the widest single support, so a
/// step's requests find all their columns resident. Within a step, the column
/// blurs and the requests are independent and run in parallel.
const DESCRIBE_STEP_COLUMNS: usize = 64;

/// Pass 2 over one octave: orientation and/or descriptors for `requests`.
///
/// Only columns covered by some request are blurred, and only at the levels
/// their gradient blends read. Every column and every request is computed
/// from the base image alone, so the output does not depend on the step
/// width or on the thread count.
pub(super) fn describe_octave(
    base: BaseImage<'_>,
    octave: usize,
    requests: &[DescribeRequest],
    describe: bool,
) -> Vec<Described> {
    describe_octave_stored(base, octave, requests, describe, Some)
}

/// Finalize built-in descriptor storage before retaining pass-2 results.
fn describe_octave_stored<D: Send>(
    base: BaseImage<'_>,
    octave: usize,
    requests: &[DescribeRequest],
    describe: bool,
    store: fn([f32; SIFT_DESCR_SIZE]) -> Option<D>,
) -> Vec<Described<D>> {
    let (rows, cols) = (base.rows, base.cols);
    let mut results = Vec::with_capacity(requests.len());
    if requests.is_empty() || rows < 3 || cols < 3 {
        return results;
    }
    let mut spans = Vec::with_capacity(requests.len());
    let mut levels = [false; LEVELS];
    let mut coverage = vec![0i32; cols + 1];
    let mut capacity = 1;
    for request in requests {
        let (span, request_levels) = support_columns(request, rows, cols, describe);
        for (level, used) in levels.iter_mut().zip(request_levels) {
            *level |= used;
        }
        if let Some((lo, hi)) = span {
            coverage[lo] += 1;
            coverage[hi + 1] -= 1;
            capacity = capacity.max(hi - lo + 1);
        }
        spans.push(span);
    }
    let capacity = (capacity + DESCRIBE_STEP_COLUMNS).min(cols);
    let mut ring = ColumnRing {
        lines: core::array::from_fn(|level| {
            vec![0.0; if levels[level] { capacity * rows } else { 0 }]
        }),
        rows,
        capacity,
    };
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by_key(|&index| spans[index].map_or(0, |(_, hi)| hi + 1));
    let kernels = octave_kernels(octave);
    let kernel = describe.then(CanonicalKernel::default);
    let mut scratch = Pass2Scratch::new(rows);
    // Requests without support produce their (empty) results first.
    let mut next = order
        .iter()
        .take_while(|&&index| spans[index].is_none())
        .count();
    describe_batch(
        &ring,
        requests,
        &order[..next],
        cols,
        kernel.as_ref(),
        &mut scratch,
        &mut results,
        store,
    );
    let mut covered = 0i32;
    let mut columns = Vec::with_capacity(DESCRIBE_STEP_COLUMNS);
    for start in (0..cols).step_by(DESCRIBE_STEP_COLUMNS) {
        if next == order.len() {
            break;
        }
        let end = (start + DESCRIBE_STEP_COLUMNS).min(cols);
        columns.clear();
        for (col, &delta) in coverage.iter().enumerate().take(end).skip(start) {
            covered += delta;
            if covered > 0 {
                columns.push(col);
            }
        }
        blur_columns(base, &kernels, &mut ring, &columns, &mut scratch);
        let ready = order[next..]
            .iter()
            .take_while(|&&index| spans[index].is_some_and(|(_, hi)| hi < end))
            .count();
        describe_batch(
            &ring,
            requests,
            &order[next..next + ready],
            cols,
            kernel.as_ref(),
            &mut scratch,
            &mut results,
            store,
        );
        next += ready;
    }
    results.sort_by_key(|described| described.request);
    results
}

/// Scratch reused across steps for serial and one-worker execution.
/// Multiworker batches own their task-local scratch.
struct Pass2Scratch {
    blur: Option<ColumnBlur>,
    window: Vec<f32>,
    #[cfg(feature = "parallel")]
    rows: usize,
}

impl Pass2Scratch {
    fn new(rows: usize) -> Self {
        #[cfg(feature = "parallel")]
        let serial = rayon::current_num_threads() == 1;
        #[cfg(not(feature = "parallel"))]
        let serial = true;
        Self {
            blur: serial.then(|| ColumnBlur::new(rows)),
            window: Vec::new(),
            #[cfg(feature = "parallel")]
            rows,
        }
    }
}

/// Blur `columns` (distinct ring slots, ascending) into the ring at the
/// levels it stores.
// Without `parallel` the serial path's early return ends the function.
#[cfg_attr(not(feature = "parallel"), allow(clippy::needless_return))]
fn blur_columns(
    base: BaseImage<'_>,
    kernels: &[HalfKernel; LEVELS],
    ring: &mut ColumnRing,
    columns: &[usize],
    scratch: &mut Pass2Scratch,
) {
    if columns.is_empty() {
        return;
    }
    let (rows, capacity) = (ring.rows, ring.capacity);
    if let Some(blur) = scratch.blur.as_mut() {
        for &col in columns {
            let start = (col % capacity) * rows;
            let out = ring
                .lines
                .each_mut()
                .map(|line| line.get_mut(start..start + rows));
            blur.blur_column_into(base, col, kernels, out);
        }
        return;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let mut slots = ring
            .lines
            .each_mut()
            .map(|line| line.chunks_exact_mut(rows).map(Some).collect::<Vec<_>>());
        let jobs = columns.iter().map(|&col| {
            let slot = col % capacity;
            let out: [Option<&mut [f32]>; LEVELS] =
                core::array::from_fn(|level| slots[level].get_mut(slot).and_then(Option::take));
            (col, out)
        });
        jobs.collect::<Vec<_>>().into_par_iter().for_each_init(
            || ColumnBlur::new(scratch.rows),
            |blur, (col, out)| blur.blur_column_into(base, col, kernels, out),
        );
    }
}

/// Results of the requests in `batch`, whose supports are resident in the
/// ring, appended to `results` (in no particular order).
#[allow(clippy::too_many_arguments)]
// Without `parallel` the serial path's early return ends the function.
#[cfg_attr(not(feature = "parallel"), allow(clippy::needless_return))]
fn describe_batch<D: Send>(
    ring: &ColumnRing,
    requests: &[DescribeRequest],
    batch: &[usize],
    cols: usize,
    kernel: Option<&CanonicalKernel>,
    scratch: &mut Pass2Scratch,
    results: &mut Vec<Described<D>>,
    store: fn([f32; SIFT_DESCR_SIZE]) -> Option<D>,
) {
    if scratch.blur.is_some() {
        results.extend(batch.iter().filter_map(|&index| {
            process_request(
                ring,
                requests,
                index,
                cols,
                kernel,
                &mut scratch.window,
                store,
            )
        }));
        return;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        results.par_extend(
            batch
                .par_iter()
                .map_init(Vec::new, |window, &index| {
                    process_request(ring, requests, index, cols, kernel, window, store)
                })
                .flatten(),
        );
    }
}

/// Result of request `index`: its orientation (the requested one, or else the
/// dominant peak of its histogram) and, with a `kernel`, its descriptor.
/// A request without an orientation peak has no result.
fn process_request<D>(
    ring: &ColumnRing,
    requests: &[DescribeRequest],
    index: usize,
    cols: usize,
    kernel: Option<&CanonicalKernel>,
    window: &mut Vec<f32>,
    store: fn([f32; SIFT_DESCR_SIZE]) -> Option<D>,
) -> Option<Described<D>> {
    let request = requests[index];
    let orientation = request.orientation.or_else(|| {
        let mut histogram = orientation_histogram(ring, request.x, request.y, request.sigma, cols);
        smooth_orientation_histogram(&mut histogram, ORI_SMOOTH_PASSES);
        dominant_orientation(&histogram)
    })?;
    let descriptor = kernel
        .and_then(|kernel| describe_from_ring(kernel, window, ring, &request, orientation, cols))
        .and_then(store);
    Some(Described {
        request: index,
        orientation,
        descriptor,
    })
}

// ============================================================================
// Pipeline
// ============================================================================

/// One output feature (octave-local anchor geometry).
pub(super) struct StreamFeature<D = [f32; SIFT_DESCR_SIZE]> {
    /// Anchor geometry carrying the assigned orientation.
    pub(super) anchor: ScaleSpaceKeypoint,
    /// Corrected public σ (octave-local).
    pub(super) public_sigma: f32,
    /// RootSIFT descriptor when description was requested.
    pub(super) descriptor: Option<D>,
}

/// Decimated octave bases `1..`, column-major with their dimensions.
type OctaveBases = Vec<(Vec<f32>, usize, usize)>;

fn base_image<'a>(image: &'a GrayImage, bases: &'a OctaveBases, octave: usize) -> BaseImage<'a> {
    if octave == 0 {
        BaseImage {
            pixels: image.as_slice(),
            rows: image.nrows(),
            cols: image.ncols(),
        }
    } else {
        let (pixels, rows, cols) = &bases[octave - 1];
        BaseImage {
            pixels,
            rows: *rows,
            cols: *cols,
        }
    }
}

fn budget_candidate(geometry: &ScaleSpaceKeypoint) -> FeatureCandidate {
    let scale = octave_scale(geometry.octave);
    FeatureCandidate::new(geometry.x * scale, geometry.y * scale, geometry.peak.abs())
}

/// Full pipeline: pass 1 per octave, adjacent-octave deduplication, budget on
/// anchors, pass 2 on kept anchors (one dominant orientation each), and
/// descriptor-failure removal.
///
/// Every kept anchor yields at most one feature, so the output never exceeds
/// the budget; an anchor without an orientation peak or a descriptor is
/// dropped, not replaced.
pub(super) fn run<B>(
    image: &GrayImage,
    params: &DetectionParams,
    budget: &B,
    describe: bool,
) -> Vec<StreamFeature>
where
    B: FeatureBudget + ?Sized,
{
    run_stored(image, params, budget, describe, Some)
}

/// Stored descriptors are built-in, thread-safe values; public custom storage
/// conversion remains in final anchor order after the float pipeline.
pub(super) fn run_stored<B, D: Send>(
    image: &GrayImage,
    params: &DetectionParams,
    budget: &B,
    describe: bool,
    store: fn([f32; SIFT_DESCR_SIZE]) -> Option<D>,
) -> Vec<StreamFeature<D>>
where
    B: FeatureBudget + ?Sized,
{
    let octaves = octave_count(image);
    let mut bases: OctaveBases = Vec::with_capacity(octaves);
    let mut candidates: Vec<Vec<Candidate>> = Vec::with_capacity(octaves);
    for octave in 0..octaves {
        let detection = detect_octave(
            base_image(image, &bases, octave),
            octave,
            Some(params),
            octave + 1 < octaves,
        );
        candidates.push(detection.candidates);
        if octave + 1 < octaves {
            bases.push((
                detection.next_base,
                detection.next_rows,
                detection.next_cols,
            ));
        }
    }
    for octave in 1..octaves {
        let (previous, current) = candidates.split_at_mut(octave);
        remove_adjacent_octave_duplicates_by(
            &mut previous[octave - 1],
            &mut current[0],
            AdjacentOctavePreference::StrongestSignedResponse,
            |candidate| &candidate.geometry,
        );
    }
    let limited = !budget.is_unlimited();
    if limited {
        let pool: Vec<FeatureCandidate> = candidates
            .iter()
            .flatten()
            .map(|candidate| budget_candidate(&candidate.geometry))
            .collect();
        let mut keep = selection_mask(&pool, image.ncols(), image.nrows(), budget).into_iter();
        for list in &mut candidates {
            list.retain(|_| keep.next().unwrap_or(false));
        }
    }

    // Supports at σ ≥ SIGMAS[2] are read from the next octave (half the
    // samples, the same σ range per octave as `describe_at_support_scale`).
    let mut requests: Vec<Vec<DescribeRequest>> = (0..octaves).map(|_| Vec::new()).collect();
    let mut origins: Vec<Vec<(usize, usize)>> = (0..octaves).map(|_| Vec::new()).collect();
    for (octave, list) in candidates.iter().enumerate() {
        for (index, candidate) in list.iter().enumerate() {
            let geometry = &candidate.geometry;
            let up = octave + 1 < octaves && geometry.sigma >= SIGMAS[2];
            let (target, factor) = if up { (octave + 1, 0.5) } else { (octave, 1.0) };
            requests[target].push(DescribeRequest {
                x: geometry.x * factor,
                y: geometry.y * factor,
                sigma: geometry.sigma * factor,
                orientation: None,
            });
            origins[target].push((octave, index));
        }
    }
    let mut features = Vec::new();
    let mut order = Vec::new();
    for (octave, (requests, origins)) in requests.iter().zip(&origins).enumerate() {
        if requests.is_empty() {
            continue;
        }
        let described = describe_octave_stored(
            base_image(image, &bases, octave),
            octave,
            requests,
            describe,
            store,
        );
        for described in described {
            if describe && described.descriptor.is_none() {
                continue;
            }
            let (source, index) = origins[described.request];
            let candidate = &candidates[source][index];
            let public_sigma = corrected_sigma_from_anchor(
                &candidate.scale_samples,
                candidate.geometry.sigma,
                candidate.geometry.is_max,
            )
            .unwrap_or(candidate.geometry.sigma);
            order.push((source, index, features.len()));
            features.push(StreamFeature {
                anchor: ScaleSpaceKeypoint {
                    orientation: described.orientation,
                    ..candidate.geometry
                },
                public_sigma,
                descriptor: described.descriptor,
            });
        }
    }
    // Release octave scratch before allocating the ordered output.
    drop(bases);
    drop(candidates);
    drop(requests);
    drop(origins);
    // Restore anchor order (octave, then anchor).
    order.sort_unstable();
    let mut slots: Vec<Option<StreamFeature<D>>> = features.into_iter().map(Some).collect();
    let features: Vec<StreamFeature<D>> = order
        .iter()
        .filter_map(|&(_, _, slot)| slots[slot].take())
        .collect();
    features
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    static OCTAVE_BASE_REBUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "std"))]
pub(super) fn reset_octave_base_rebuilds() {
    OCTAVE_BASE_REBUILDS.with(|calls| calls.set(0));
}

#[cfg(all(test, feature = "std"))]
pub(super) fn octave_base_rebuilds() -> usize {
    OCTAVE_BASE_REBUILDS.with(std::cell::Cell::get)
}

/// Decimated bases of octaves `1..octaves` without detection.
pub(super) fn octave_bases(image: &GrayImage, octaves: usize) -> OctaveBases {
    let mut bases: OctaveBases = Vec::with_capacity(octaves);
    for octave in 0..octaves.saturating_sub(1) {
        #[cfg(all(test, feature = "std"))]
        OCTAVE_BASE_REBUILDS.with(|calls| calls.set(calls.get() + 1));
        let detection = detect_octave(base_image(image, &bases, octave), octave, None, true);
        bases.push((
            detection.next_base,
            detection.next_rows,
            detection.next_cols,
        ));
    }
    bases
}

/// Descriptors at caller-supplied octave-local requests, grouped by octave.
pub(super) fn describe_requests(
    image: &GrayImage,
    by_octave: &[Vec<DescribeRequest>],
) -> Vec<Vec<Described>> {
    let bases = octave_bases(image, by_octave.len());
    by_octave
        .iter()
        .enumerate()
        .map(|(octave, requests)| {
            describe_octave(base_image(image, &bases, octave), octave, requests, true)
        })
        .collect()
}

#[cfg(all(test, feature = "std"))]
mod spatial_cache_tests {
    use super::*;

    fn flat_spatial_profiles(samples: [f32; LEVELS]) -> LineRing3 {
        let mut log = LineRing3::new(5);
        for (line, sample) in log.lines.iter_mut().zip(samples) {
            line.fill(sample);
        }
        log
    }

    fn refinement_work(log: &LineRing3, found: &[(f32, f32, bool)]) -> (usize, usize) {
        let params = DetectionParams {
            peak_threshold: 0.0,
            sigma_threshold: 0.0,
            curv_ratio_threshold: 12.1,
            sigma_window: (1.9027314, 4.1498866),
        };
        let mut out = Vec::new();
        SPATIAL_NEIGHBOR_CUBICS.with(|count| count.set(0));
        refine_candidates(log, 2, 1, 5, 0, &params, found, &mut out);
        (
            SPATIAL_NEIGHBOR_CUBICS.with(std::cell::Cell::get),
            out.len(),
        )
    }

    #[test]
    fn spatial_neighbor_cubics_short_circuit_and_are_shared_by_both_extrema() {
        let flat = flat_spatial_profiles([1.0; LEVELS]);
        assert_eq!(refinement_work(&flat, &[(2.4, -10.0, false)]), (0, 0));

        // A signed four-response profile has two typed stationary extrema.
        let samples = [0.0, -10.0, 10.0, 0.0];
        let (minimum, maximum, sigma_min, sigma_max) = scale_envelope(samples);
        assert!(minimum < 0.0 && maximum > 0.0);
        let found = [(sigma_min, minimum, false), (sigma_max, maximum, true)];
        let equal = flat_spatial_profiles(samples);
        // The first neighbor equals the center: only its cubic is needed,
        // and both polarities reuse it before failing the strict comparison.
        assert_eq!(refinement_work(&equal, &found), (1, 0));

        // A linear response perturbation raises neighbors at the minimum and
        // lowers them at the maximum. Both spatial extrema survive FD and Q.
        let midpoint = (sigma_min + sigma_max) * 0.5;
        let neighbors = core::array::from_fn(|level| samples[level] + midpoint - SIGMAS[level]);
        let mut log = flat_spatial_profiles(neighbors);
        for (line, sample) in log.lines.iter_mut().zip(samples) {
            line[7] = sample; // Integer center: col=1, row=2 in the three-column ring.
        }
        assert_eq!(refinement_work(&log, &found), (8, 2));
    }
}

#[cfg(test)]
mod base_only_tests {
    use super::*;
    use nalgebra::DMatrix;

    fn image(rows: usize, cols: usize) -> DMatrix<f32> {
        DMatrix::from_fn(rows, cols, |row, col| {
            ((row * 37 + col * 17 + row * col) % 251) as f32 - 117.0
        })
    }

    fn base(image: &DMatrix<f32>) -> BaseImage<'_> {
        BaseImage {
            pixels: image.as_slice(),
            rows: image.nrows(),
            cols: image.ncols(),
        }
    }

    // Independent eager streaming materialization, followed by exact even sampling.
    fn materialized_next(
        base: BaseImage<'_>,
        octave: usize,
        want_next: bool,
    ) -> (Vec<f32>, usize, usize) {
        let (next_rows, next_cols) = if want_next {
            (base.rows / 2, base.cols / 2)
        } else {
            (0, 0)
        };
        let mut next = vec![0.0; next_rows * next_cols];
        if next.is_empty() {
            return (next, next_rows, next_cols);
        }
        let kernels = octave_kernels(octave);
        let mut blur = ColumnBlur::new(base.rows);
        let mut levels: [Vec<f32>; LEVELS] =
            core::array::from_fn(|_| vec![0.0; base.rows * base.cols]);
        for column in 0..base.cols {
            blur.blur_column(
                base,
                column,
                &kernels,
                [true; LEVELS],
                &mut levels,
                column * base.rows,
            );
        }
        for column in 0..next_cols {
            for row in 0..next_rows {
                next[column * next_rows + row] = levels[2][2 * column * base.rows + 2 * row];
            }
        }
        (next, next_rows, next_cols)
    }

    #[test]
    fn base_only_rebuild_matches_materialized_even_sampling_bits_and_shapes() {
        let tiny_params = DetectionParams {
            peak_threshold: 0.0,
            sigma_threshold: 0.0,
            curv_ratio_threshold: 12.1,
            sigma_window: (1.9027314, 4.1498866),
        };
        for rows in [0, 1, 2, 3, 7, 8, 9, 16, 17, 37] {
            for cols in [0, 1, 2, 3, 4, 7, 8, 9, 16, 17, 37] {
                let image = image(rows, cols);
                for octave in [0, 1] {
                    for want_next in [false, true] {
                        let (expected, next_rows, next_cols) =
                            materialized_next(base(&image), octave, want_next);
                        for params in [None, Some(&tiny_params)] {
                            if params.is_some()
                                && rows > 2 * DETECTION_BORDER
                                && cols > 2 * DETECTION_BORDER
                            {
                                continue;
                            }
                            let actual = detect_octave(base(&image), octave, params, want_next);
                            assert!(actual.candidates.is_empty());
                            assert_eq!(
                                (actual.next_rows, actual.next_cols),
                                (next_rows, next_cols)
                            );
                            assert_eq!(
                                actual
                                    .next_base
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                "rows={rows} cols={cols} octave={octave} want_next={want_next}"
                            );
                        }
                    }
                }
            }
        }
        // Odd input pixels must still influence selected even outputs.
        let mut odd_impulse = DMatrix::zeros(9, 9);
        odd_impulse[(1, 1)] = 255.0;
        let (expected, _, _) = materialized_next(base(&odd_impulse), 0, true);
        assert!(expected[0] > 0.0);
        let actual = detect_octave(base(&odd_impulse), 0, None, true);
        assert_eq!(
            actual
                .next_base
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn base_only_rebuild_blurs_only_readable_even_output_columns() {
        use super::super::blur::{column_blur_calls, reset_column_blur_calls};
        for (rows, cols) in [(0, 9), (9, 0), (1, 9), (9, 1), (2, 2), (9, 9), (16, 17)] {
            let image = image(rows, cols);
            for want_next in [false, true] {
                reset_column_blur_calls();
                let actual = detect_octave(base(&image), 0, None, want_next);
                let expected = if want_next && rows / 2 > 0 && cols / 2 > 0 {
                    cols / 2
                } else {
                    0
                };
                assert_eq!(column_blur_calls(), expected);
                assert_eq!(actual.next_base.len(), actual.next_rows * actual.next_cols);
            }
        }
    }
}

#[cfg(test)]
mod extremum_type_tests {
    use super::*;

    fn finite_response_rings(center: [f32; 4], offsets: [f32; 9]) -> LineRing3 {
        let mut log = LineRing3::new(9);
        for (index, offset) in offsets.into_iter().enumerate() {
            let row = 4 + index / 3 - 1;
            let col = 4 + index % 3 - 1;
            let at = (col % 3) * 9 + row;
            let samples = center.map(|value| value + offset);
            for (line, sample) in log.lines.iter_mut().zip(samples) {
                line[at] = sample;
            }
        }
        log
    }

    fn params() -> DetectionParams {
        DetectionParams {
            peak_threshold: 30.0,
            sigma_threshold: 0.0,
            curv_ratio_threshold: 12.1,
            sigma_window: (1.9027314, 4.1498866),
        }
    }

    #[test]
    fn tied_scale_minimum_cannot_emit_as_maximum_after_q() {
        // A plateau that ends on the top level belongs to the next octave.
        let plateau = [0x4247fff9, 0x4247fffe, 0x42480000, 0x42480000].map(f32::from_bits);
        assert!(!owns_scale_extremum(&plateau, true));
        // One ULP lower at the top level, the interior level owns the maximum.
        let center = [0x4247fff9, 0x4247fffe, 0x42480000, 0x4247ffff].map(f32::from_bits);
        let offsets = core::array::from_fn(|index| {
            let x = (index % 3) as f32 - 1.0;
            let y = (index / 3) as f32 - 1.0;
            -(x * x + y * y)
        });
        let log = finite_response_rings(center, offsets);
        let [a, b, c, _] = compute_poly_coeffs(center[0], center[1], center[2], center[3]);
        let ap = 6.0 * a;
        let bp = 2.0 * b;
        let wrong_minimum = (-bp + (bp * bp - 2.0 * ap * c).sqrt()) / ap;
        let (_, peak, _, maximum) = scale_envelope(center);
        assert!(6.0 * a * wrong_minimum + 2.0 * b > 0.0);
        assert!(6.0 * a * maximum + 2.0 * b < 0.0);
        let mut out = Vec::new();
        refine_candidates(
            &log,
            4,
            4,
            9,
            0,
            &params(),
            &[(wrong_minimum, peak, true)],
            &mut out,
        );
        assert!(
            out.is_empty(),
            "cheap guard must reject the mislabeled scale minimum"
        );
        refine_candidates(
            &log,
            4,
            4,
            9,
            0,
            &params(),
            &[(maximum, peak, true)],
            &mut out,
        );
        assert_eq!(
            out.len(),
            1,
            "correctly classified maximum must remain accepted"
        );
    }

    #[test]
    fn finite_scale_minimum_cannot_emit_a_saddle_q_fit() {
        let center = [0xc23e2c10, 0xc2445844, 0xc24826fd, 0xc2418be2].map(f32::from_bits);
        let offsets = [0.01, 2.0, 0.01, 1.0, 0.0, 1.0, 0.01, 2.0, 0.01];
        let log = finite_response_rings(center, offsets);
        let (peak, _, minimum, _) = scale_envelope(center);
        let [a, b, _, _] = compute_poly_coeffs(center[0], center[1], center[2], center[3]);
        assert!(minimum > 0.0 && 6.0 * a * minimum + 2.0 * b > 0.0);
        assert!(peak < -params().peak_threshold);
        let mut out = Vec::new();
        refine_candidates(
            &log,
            4,
            4,
            9,
            0,
            &params(),
            &[(minimum, peak, false)],
            &mut out,
        );
        assert!(
            out.is_empty(),
            "Q's saddle determinant must reject after valid scale/FD guards"
        );
    }
}
