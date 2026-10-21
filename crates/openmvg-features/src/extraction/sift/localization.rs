// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Keypoint localization: 3D discrete DoG extrema, sub-pixel refinement with
//! contrast, edge and border rejection, and orientation assignment.

use super::super::ScaleSpaceKeypoint;
use super::super::alp::GrayImage;
use super::super::feature_budget::{FeatureBudgetKeypoint, FeatureCandidate};
use super::common::{
    ORI_SMOOTH_PASSES, build_orientation_histogram, extract_orientation_peaks,
    smooth_orientation_histogram,
};
use super::scale_space::{DogOctave, GradientOctave};
use crate::math_compat::{atan2f, powf};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeSet, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeSet;

const MAX_REFINEMENT_STEPS: usize = 5;
const OFFSET_MAX: f32 = 0.6;

// ============================================================================
// Data Structures
// ============================================================================

/// Minimal state needed by the discrete-extrema and refinement stages.
#[derive(Clone, Copy, Debug)]
pub(super) struct SiftCandidate {
    pub(super) scale_index: usize,
    pub(super) i: i32,
    pub(super) j: i32,
    pub(super) val: f32,
}

/// Descriptor-free keypoint state used throughout detection and orientation.
///
/// Keeping the 512-byte descriptor out of this record makes candidate buffers
/// and orientation clones small. Public keypoint records are assembled
/// only when the caller requests them.
#[derive(Clone, Copy, Debug)]
pub(super) struct SiftFeature {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) sigma: f32,
    pub(super) orientation: f32,
    pub(super) octave: usize,
    pub(super) scale_index: usize,
    pub(super) i: i32,
    pub(super) j: i32,
    pub(super) val: f32,
}

impl SiftFeature {
    pub(super) fn geometry(self) -> ScaleSpaceKeypoint {
        ScaleSpaceKeypoint {
            x: self.x,
            y: self.y,
            sigma: self.sigma,
            orientation: self.orientation,
            peak: self.val,
            octave: self.octave,
            scale_index: self.scale_index,
            is_max: self.val > 0.0,
        }
    }
}

impl FeatureBudgetKeypoint for SiftFeature {
    fn feature_candidate(&self) -> FeatureCandidate {
        FeatureCandidate::new(self.x, self.y, self.val.abs())
    }
}

// ============================================================================
// 3D Discrete Extrema Detection
// ============================================================================

/// Check if `(id_slice, id_row, id_col)` is a local min or max across 26 neighbors.
///
/// # Panics (debug builds only)
/// Panics if any index is 0 or at the upper boundary — callers must ensure
/// the point is interior to the 3D neighborhood.
#[inline]
pub(super) fn is_local_min_max(
    slices: &[GrayImage],
    id_slice: usize,
    id_row: usize,
    id_col: usize,
) -> bool {
    debug_assert!(id_slice >= 1 && id_slice + 1 < slices.len());
    debug_assert!(id_row >= 1 && id_row + 1 < slices[id_slice].nrows());
    debug_assert!(id_col >= 1 && id_col + 1 < slices[id_slice].ncols());
    let rows = slices[id_slice].nrows();
    let center = slices[id_slice].as_slice()[id_col * rows + id_row];
    let is_maximum = center > 0.0;

    // Check all 26 neighbors (8 in current + 9 above + 9 below)
    for ds in [id_slice.wrapping_sub(1), id_slice, id_slice + 1] {
        let slice = slices[ds].as_slice();
        for di in [id_row.wrapping_sub(1), id_row, id_row + 1] {
            for dj in [id_col.wrapping_sub(1), id_col, id_col + 1] {
                if ds == id_slice && di == id_row && dj == id_col {
                    continue; // skip center
                }
                let neighbor = slice[dj * rows + di];
                if (is_maximum && center <= neighbor) || (!is_maximum && center >= neighbor) {
                    return false;
                }
            }
        }
    }
    true
}

pub(super) fn find_3d_discrete_extrema(
    dogs: &DogOctave,
    peak_threshold: f32,
) -> Vec<SiftCandidate> {
    let ns = dogs.slices.len();
    let h = dogs.slices[0].nrows();
    let w = dogs.slices[0].ncols();
    let threshold = peak_threshold * 0.8; // Pre-filter at 80% as in SiftAnatomy

    let mut keypoints = Vec::new();

    for s in 1..ns - 1 {
        let slice = dogs.slices[s].as_slice();
        for col in 1..w - 1 {
            let column = &slice[col * h..(col + 1) * h];
            for (row, &val) in column.iter().enumerate().take(h - 1).skip(1) {
                if val.abs() <= threshold {
                    continue;
                }
                if is_local_min_max(&dogs.slices, s, row, col) {
                    keypoints.push(SiftCandidate {
                        scale_index: s,
                        i: col as i32,
                        j: row as i32,
                        val,
                    });
                }
            }
        }
    }
    keypoints.sort_unstable_by_key(|kp| (kp.scale_index, kp.j, kp.i));
    keypoints
}

// ============================================================================
// Sub-pixel Refinement (3D Taylor Expansion)
// ============================================================================

/// Refine keypoint position via 3D Hessian inversion.
///
/// Returns `(offset_x, offset_y, offset_s, interpolated_value)` when the
/// Hessian is non-singular, regardless of offset magnitude. The caller is
/// responsible for checking whether `|offset| < OFFSET_MAX` (convergence).
/// Returns `None` only when the LU decomposition fails (singular Hessian).
pub(super) fn inverse_3d_taylor(
    dogs: &DogOctave,
    col: i32,
    row: i32,
    s: usize,
) -> Option<(f32, f32, f32, f32)> {
    let slice = &dogs.slices[s];
    let slice_up = &dogs.slices[s + 1];
    let slice_dn = &dogs.slices[s - 1];

    let r = row as usize;
    let c = col as usize;

    let center = slice[(r, c)];

    // 3D Hessian via finite differences
    let h_xx = slice[(r, c - 1)] + slice[(r, c + 1)] - 2.0 * center;
    let h_yy = slice[(r + 1, c)] + slice[(r - 1, c)] - 2.0 * center;
    let h_ss = slice_up[(r, c)] + slice_dn[(r, c)] - 2.0 * center;
    let h_xy = (slice[(r + 1, c + 1)] - slice[(r - 1, c + 1)] - slice[(r + 1, c - 1)]
        + slice[(r - 1, c - 1)])
        / 4.0;
    let h_xs = (slice_up[(r, c + 1)] - slice_up[(r, c - 1)] - slice_dn[(r, c + 1)]
        + slice_dn[(r, c - 1)])
        / 4.0;
    let h_ys = (slice_up[(r + 1, c)] - slice_up[(r - 1, c)] - slice_dn[(r + 1, c)]
        + slice_dn[(r - 1, c)])
        / 4.0;

    // 3D gradient
    let g_x = (slice[(r, c + 1)] - slice[(r, c - 1)]) / 2.0;
    let g_y = (slice[(r + 1, c)] - slice[(r - 1, c)]) / 2.0;
    let g_s = (slice_up[(r, c)] - slice_dn[(r, c)]) / 2.0;

    // Solve H * offset = -g using nalgebra
    let hessian = nalgebra::Matrix3::new(h_xx, h_xy, h_xs, h_xy, h_yy, h_ys, h_xs, h_ys, h_ss);

    let gradient = nalgebra::Vector3::new(-g_x, -g_y, -g_s);

    let decomp = hessian.lu();
    let offset = decomp.solve(&gradient)?;

    let ofst_x = offset[0];
    let ofst_y = offset[1];
    let ofst_s = offset[2];
    let ofst_val = (g_x * ofst_x + g_y * ofst_y + g_s * ofst_s) / 2.0;

    // Always return offsets: the convergence check belongs at the call site so
    // the refinement loop can shift position using the actual computed offsets.
    Some((ofst_x, ofst_y, ofst_s, center + ofst_val))
}

/// Compute the Harris & Stephens edge response on the DoG at keypoint position.
pub(super) fn compute_edge_response(dogs: &DogOctave, kp: &SiftFeature) -> f32 {
    let s = kp.scale_index;
    let r = kp.j as usize;
    let c = kp.i as usize;
    let im = &dogs.slices[s];

    let h_xx = im[(r, c - 1)] + im[(r, c + 1)] - 2.0 * im[(r, c)];
    let h_yy = im[(r + 1, c)] + im[(r - 1, c)] - 2.0 * im[(r, c)];
    let h_xy =
        (im[(r + 1, c + 1)] - im[(r - 1, c + 1)] - im[(r + 1, c - 1)] + im[(r - 1, c - 1)]) / 4.0;

    // Saddle points (det < 0) must always be rejected per Lowe §4.1.
    // Test det <= 0 instead of |det| < epsilon so that negative det
    // (both eigenvalues with opposite signs) returns MAX rather than a
    // negative ratio that could slip through the abs() check in the caller.
    #[allow(clippy::suspicious_operation_groupings)] // det(H) = Hxx·Hyy − Hxy² is correct
    let det = h_xx * h_yy - h_xy * h_xy;
    if det <= 0.0 {
        return f32::MAX;
    }
    (h_xx + h_yy).powi(2) / det
}

/// Check if keypoint is not too close to image border (accounting for its scale).
pub(super) fn border_check(kp: &SiftFeature, w: usize, h: usize, delta: f32, lambda: f32) -> bool {
    let sigma = kp.sigma;
    kp.x - lambda * sigma > 0.0
        && kp.x + lambda * sigma < w as f32 * delta
        && kp.y - lambda * sigma > 0.0
        && kp.y + lambda * sigma < h as f32 * delta
}

/// Refine keypoint positions: iterative sub-pixel refinement, peak check, edge rejection, border check.
pub(super) fn refine_keypoints(
    dogs: &DogOctave,
    keypoints: Vec<SiftCandidate>,
    peak_threshold: f32,
    edge_threshold: f32,
) -> Vec<SiftFeature> {
    let w = dogs.slices[0].ncols();
    let h = dogs.slices[0].nrows();
    let delta = dogs.delta;

    let edge_thresh_sq = (edge_threshold + 1.0).powi(2) / edge_threshold;
    let sigma_ratio = dogs.sigmas[1] / dogs.sigmas[0];

    let mut refined = Vec::with_capacity(keypoints.len());
    // Distinct extrema can shift onto the same sample while refining. The
    // sample alone determines the refined keypoint, so keep its first visit.
    let mut refined_samples = BTreeSet::new();

    for key in &keypoints {
        let mut ic = key.i;
        let mut jc = key.j;
        let mut sc = key.scale_index;

        let mut ofst_x = 0.0f32;
        let mut ofst_y = 0.0f32;
        let mut ofst_s = 0.0f32;
        let mut val = key.val;

        let mut converged = false;

        for _iter in 0..MAX_REFINEMENT_STEPS {
            if ic <= 0 || ic >= (w as i32 - 1) || jc <= 0 || jc >= (h as i32 - 1) {
                break; // Shifted outside safe interior — give up
            }
            // inverse_3d_taylor returns offsets for any non-singular matrix.
            // Convergence is checked here so the shift direction uses the
            // computed offsets.
            let Some((dx, dy, ds, v)) = inverse_3d_taylor(dogs, ic, jc, sc) else {
                break; // Singular Hessian — give up
            };
            ofst_x = dx;
            ofst_y = dy;
            ofst_s = ds;
            val = v;

            if ofst_x.abs() < OFFSET_MAX && ofst_y.abs() < OFFSET_MAX && ofst_s.abs() < OFFSET_MAX {
                converged = true;
                break;
            }

            // Shift discrete position toward the refined location.
            if ofst_x > OFFSET_MAX && (ic + 1) < (w as i32 - 1) {
                ic += 1;
            }
            if ofst_x < -OFFSET_MAX && (ic - 1) > 0 {
                ic -= 1;
            }
            if ofst_y > OFFSET_MAX && (jc + 1) < (h as i32 - 1) {
                jc += 1;
            }
            if ofst_y < -OFFSET_MAX && (jc - 1) > 0 {
                jc -= 1;
            }
            if ofst_s > OFFSET_MAX && sc + 1 < dogs.slices.len() - 1 {
                sc += 1;
            }
            if ofst_s < -OFFSET_MAX && sc > 1 {
                sc -= 1;
            }
        }

        if !converged || !refined_samples.insert((sc, jc, ic)) {
            continue;
        }

        // Peak threshold check on interpolated value
        if val.abs() <= peak_threshold {
            continue;
        }

        let kp = SiftFeature {
            x: (ic as f32 + ofst_x) * delta,
            y: (jc as f32 + ofst_y) * delta,
            sigma: dogs.sigmas[sc] * powf(sigma_ratio, ofst_s),
            orientation: 0.0,
            octave: dogs.octave_level,
            scale_index: sc,
            i: ic,
            j: jc,
            val,
        };

        // Edge response check
        let edge_resp = compute_edge_response(dogs, &kp);
        if edge_resp < 0.0 || edge_resp.abs() > edge_thresh_sq {
            continue;
        }

        // Border check
        if !border_check(&kp, w, h, delta, 1.0) {
            continue;
        }

        refined.push(kp);
    }

    refined
}

// ============================================================================
// Orientation Assignment
// ============================================================================

/// Compute orientations for a single keypoint; returns one clone per histogram peak.
/// Pure function over `&GradientOctave` — safe to call from Rayon threads.
pub(super) fn orientations_for_one(key: SiftFeature, grads: &GradientOctave) -> Vec<SiftFeature> {
    let s = key.scale_index;
    let gradient = match &grads.slices[s] {
        Some(gradient) => gradient,
        None => return Vec::new(),
    };

    let delta = grads.delta;
    // Convert from absolute image space to octave space.
    let x = key.x / delta;
    let y = key.y / delta;
    let sigma = key.sigma / delta;
    let (h, w) = (gradient.rows, gradient.cols);

    let mut hist = build_orientation_histogram(x, y, sigma, |row, col| {
        if col <= 0 || row <= 0 || col >= w as i32 || row >= h as i32 {
            return None;
        }
        let su = col as usize;
        let sv = row as usize;
        let dx = gradient.x[sv * w + su];
        let dy = gradient.y[sv * w + su];
        let mag = (dx * dx + dy * dy).sqrt();
        Some((mag, atan2f(dy, dx)))
    });

    smooth_orientation_histogram(&mut hist, ORI_SMOOTH_PASSES);

    extract_orientation_peaks(&hist)
        .into_iter()
        .map(|orientation| {
            let mut kp = key;
            kp.orientation = orientation;
            kp
        })
        .collect()
}

/// Assign orientations to keypoints using the shared orientation histogram builder.
/// Coordinates are converted to octave space (÷ `grads.delta`) before histogram construction.
pub(super) fn assign_orientations(
    keypoints: Vec<SiftFeature>,
    grads: &GradientOctave,
) -> Vec<SiftFeature> {
    // Each keypoint is processed independently from read-only gradient slices.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        keypoints
            .into_par_iter()
            .flat_map_iter(|key| orientations_for_one(key, grads))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    keypoints
        .into_iter()
        .flat_map(|key| orientations_for_one(key, grads))
        .collect()
}
