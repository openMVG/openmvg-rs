// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-pixel fits shared by both ALP passes and the reference detector: the
//! scale cubic from the four level responses, the corrected public σ, the
//! sub-pixel refinement with edge rejection, and the octave geometry.
//!
//! Base-scale weights minimize equal-weight, unit-energy Gaussian and wave
//! response error. The model uses 32 log-scale quadrature points in the accepted
//! window, three points per phase coordinate, eight wave directions, and twelve
//! profile points per finite Gaussian radius interval. Orthonormal contrasts
//! impose `T·1=[0,0,0,1]ᵀ` before a Float64 QR solve and Float32 conversion.
//! Centered evaluation uses three quantized weights per row and a constant
//! reference; its effective fourth column follows the constraint. The fitting
//! model convolves vertically then horizontally, while runtime filtering uses
//! horizontal then vertical passes, so their rounding can differ.
//!
//! The public-scale cubic targets normalized Gaussian response and stationarity
//! over 567 native/transition cases. A common finite-kernel variance preference
//! regularizes a separate 12-variable centered QR solve. Centered response
//! differences preserve flat-input fallback. A four-point line residual of
//! `sqrt(σ²/|r|)` tests Gaussian profile compatibility, with its envelope generated
//! from the same cases. Compatibility blends a `1/40`-octave bounded Newton
//! fallback with the signed stationary root. The model root must preserve
//! curvature, stay in its domain, and improve response and stationarity relative
//! to the fallback. Compatibility is not a Gaussian probability or noise model.
//!
//! ```text
//! Four level responses -> centered differences -> cubic P(σ)
//!                                                |
//!                    owned center extrema, neighbors within one level and Q -> anchor
//!                                                |
//!                                  orientation and descriptor support
//!                                                |
//!                                    separate public-scale correction
//! ```

use super::GrayImage;
use super::geometry_tables::{
    ALP_GEOMETRY_ACCEPT_SIGMA_MAX, ALP_GEOMETRY_ACCEPT_SIGMA_MIN, ALP_GEOMETRY_BLEND_BETA,
    ALP_GEOMETRY_BLEND_BOOST, ALP_GEOMETRY_COMPATIBILITY_LOWER, ALP_GEOMETRY_COMPATIBILITY_UPPER,
    ALP_GEOMETRY_CURVATURE_EPSILON, ALP_GEOMETRY_FINAL_MOVE_CAP_LOG2, ALP_GEOMETRY_FIT_T_MAX,
    ALP_GEOMETRY_FIT_T_MIN, ALP_GEOMETRY_LOG_SIGMA_ORIGIN, ALP_GEOMETRY_SCALE_CUBIC_LOG2,
};
use super::tables::{ALP_SCALE_CUBIC_BASE, ALP_SPATIAL_QUADRATIC_BASE, Q_SAMPLE_ORDER, SIGMAS};
use crate::extraction::ScaleSpaceKeypoint;
use crate::extraction::sift::common::{ORI_HIST_BINS, TWO_PI};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // Shadowed when std is linked; see `math_compat::FloatExt`.
use crate::math_compat::FloatExt as _;
use crate::math_compat::{expf, log2f};

/// Accepted fitted scale per octave: `1.6 · 2^t` for `t ∈ [0.25, 1.375]`.
///
/// The window bounds the fitted root. Octave ownership comes from the level
/// samples ([`owns_scale_extremum`]); the window's upper eighth of an octave
/// keeps fitted roots that an interior level still owns.
pub(super) const SCALE_WINDOW: (f32, f32) = (1.902_731_4, 4.149_886_6);
/// Half-width of the scale band in which a candidate competes with its spatial
/// neighbors, as a factor on σ: one ALP level step (half an octave), the
/// counterpart of SIFT's comparison with the adjacent levels.
pub(super) const NEIGHBOR_SCALE_BAND: f32 = core::f32::consts::SQRT_2;
const MAX_DISPLACEMENT: f32 = 0.6;
const MAX_OCTAVES: usize = 6;
const MIN_IMAGE_DIMENSION: usize = 10;
const OCTAVE_REFERENCE_SIZE: f32 = 32.0;
pub(super) const ORIENTATION_SCALE_SNAP: f32 = 0.05;

pub(super) fn fit_spatial_quadratic(scale_index: usize, samples: &[f32; 9]) -> [f32; 5] {
    let coefficients = &ALP_SPATIAL_QUADRATIC_BASE[scale_index];
    core::array::from_fn(|row_idx| {
        let offset = row_idx * 9;
        Q_SAMPLE_ORDER
            .iter()
            .enumerate()
            .map(|(column, &sample_index)| coefficients[offset + column] * samples[sample_index])
            .sum()
    })
}

pub(super) type CubicCoefficients = [f32; 4];

#[inline]
fn apply_cubic_transform(transform: &[[f32; 4]; 4], samples: &[f32; 4]) -> CubicCoefficients {
    let differences = [
        samples[0] - samples[3],
        samples[1] - samples[3],
        samples[2] - samples[3],
    ];
    let mut coefficients = core::array::from_fn(|row_index| {
        let row = &transform[row_index];
        let mut value = row[0] * differences[0];
        value += row[1] * differences[1];
        value += row[2] * differences[2];
        value
    });
    coefficients[3] += samples[3];
    coefficients
}

#[inline]
pub(super) fn octave_scale(octave: usize) -> f32 {
    (1usize << octave) as f32
}

pub(super) fn octave_count(image: &GrayImage) -> usize {
    let smallest_dimension = image.nrows().min(image.ncols());
    if smallest_dimension < MIN_IMAGE_DIMENSION {
        return 0;
    }
    (log2f(smallest_dimension as f32 / OCTAVE_REFERENCE_SIZE).ceil() as usize).clamp(1, MAX_OCTAVES)
}

/// Returns the index of the closest entry in [`SIGMAS`] to `sigma`.
pub(super) fn get_nearest_scale_index(sigma: f32) -> usize {
    let mut min_diff = f32::MAX;
    let mut best_idx = 0;

    for (i, &s) in SIGMAS.iter().enumerate() {
        let diff = (sigma - s).abs();
        if diff < min_diff {
            min_diff = diff;
            best_idx = i;
        }
    }
    best_idx
}

/// Route the Q fit to one of the two central ALP levels.
#[inline]
pub(super) fn get_q_scale_index(sigma: f32) -> usize {
    if sigma <= SIGMAS[1] { 1 } else { 2 }
}

/// Whether this octave owns a scale extremum of the given polarity: an interior
/// level carries the strongest response of that sign.
///
/// For a profile symmetric in log σ this places the extremum between the
/// geometric means of the outer level pairs, `√(σ₀σ₁) ≈ 1.90` and
/// `√(σ₂σ₃) ≈ 3.81`, exactly one octave. The adjacent octave shares those level
/// scales, so a site is not claimed by both.
#[inline]
pub(super) fn owns_scale_extremum(samples: &[f32; 4], is_max: bool) -> bool {
    if is_max {
        samples[1].max(samples[2]) > samples[0].max(samples[3])
    } else {
        samples[1].min(samples[2]) < samples[0].min(samples[3])
    }
}

/// Cubic evaluation without `mul_add`, which is a library call on targets
/// without hardware FMA (for example `x86-64-v2`).
#[inline(always)]
pub(super) fn horner([a, b, c, d]: CubicCoefficients, s: f32) -> f32 {
    ((a * s + b) * s + c) * s + d
}

/// Scale band `[σ / band, σ · band]` clipped to the sampled levels.
#[inline]
pub(super) fn neighbor_scale_band(sigma: f32) -> (f32, f32) {
    (
        (sigma / NEIGHBOR_SCALE_BAND).max(SIGMAS[0]),
        (sigma * NEIGHBOR_SCALE_BAND).min(SIGMAS[3]),
    )
}

/// Strongest response of a cubic over `[low, high]` with the given polarity:
/// the largest value for a maximum, the smallest for a minimum.
///
/// It lies at a band end or at the cubic's interior stationary point of that
/// type. Non-finite coefficients give NaN, so a strict comparison fails.
#[inline]
pub(super) fn band_extreme(coeffs: CubicCoefficients, low: f32, high: f32, is_max: bool) -> f32 {
    if !coeffs.iter().all(|value| value.is_finite()) {
        return f32::NAN;
    }
    let [a, b, c, _] = coeffs;
    let sign = if is_max { 1.0 } else { -1.0 };
    let mut best = (sign * horner(coeffs, low)).max(sign * horner(coeffs, high));
    // P'' = 6aσ + 2b is negative at a maximum and positive at a minimum.
    let stationary = if a.abs() > 1e-6 {
        let (ap, bp) = (6.0 * a, 2.0 * b);
        let discriminant = bp * bp - 2.0 * ap * c;
        (discriminant > 0.0).then(|| (-bp - sign * discriminant.sqrt()) / ap)
    } else if b.abs() > 1e-6 && (b < 0.0) == is_max {
        Some(-c / (2.0 * b))
    } else {
        None
    };
    if let Some(s) = stationary
        && s > low
        && s < high
    {
        best = best.max(sign * horner(coeffs, s));
    }
    sign * best
}

/// Cubic approximation `P(σ) = aσ³ + bσ² + cσ + d` from four level responses.
#[inline]
pub(super) fn compute_poly_coeffs(l0: f32, l1: f32, l2: f32, l3: f32) -> CubicCoefficients {
    let differences = [l0 - l3, l1 - l3, l2 - l3];
    let mut coefficients = core::array::from_fn(|row_index| {
        let row = &ALP_SCALE_CUBIC_BASE[row_index];
        let mut value = row[0] * differences[0];
        value += row[1] * differences[1];
        value += row[2] * differences[2];
        value
    });
    coefficients[3] += l3;
    coefficients
}

/// Cubic in `log2 σ` used only for the corrected public scale.
#[inline]
fn compute_geometry_poly_coeffs(samples: &[f32; 4]) -> CubicCoefficients {
    apply_cubic_transform(&ALP_GEOMETRY_SCALE_CUBIC_LOG2, samples)
}

const fn profile_sum4(values: [f32; 4]) -> f32 {
    let mut value = values[0] + values[1];
    value += values[2];
    value += values[3];
    value
}

const PROFILE_X: [f32; 4] = [
    SIGMAS[0] * SIGMAS[0],
    SIGMAS[1] * SIGMAS[1],
    SIGMAS[2] * SIGMAS[2],
    SIGMAS[3] * SIGMAS[3],
];
const PROFILE_MEAN_X: f32 = profile_sum4(PROFILE_X) * 0.25;
const PROFILE_DX: [f32; 4] = [
    PROFILE_X[0] - PROFILE_MEAN_X,
    PROFILE_X[1] - PROFILE_MEAN_X,
    PROFILE_X[2] - PROFILE_MEAN_X,
    PROFILE_X[3] - PROFILE_MEAN_X,
];
const PROFILE_X_ENERGY: f32 = profile_sum4([
    PROFILE_DX[0] * PROFILE_DX[0],
    PROFILE_DX[1] * PROFILE_DX[1],
    PROFILE_DX[2] * PROFILE_DX[2],
    PROFILE_DX[3] * PROFILE_DX[3],
]);

#[inline]
pub(super) fn profile_compatibility_statistic(samples: &[f32; 4]) -> Option<f32> {
    if !samples.iter().all(|value| value.is_finite())
        || !(samples.iter().all(|&value| value > 0.0) || samples.iter().all(|&value| value < 0.0))
    {
        return None;
    }
    let quotient: [f32; 4] = core::array::from_fn(|index| PROFILE_X[index] / samples[index].abs());
    if !quotient
        .iter()
        .all(|&value| value.is_finite() && value > 0.0)
    {
        return None;
    }
    let root = quotient.map(|value| value.sqrt());
    if !root.iter().all(|&value| value.is_finite() && value > 0.0) {
        return None;
    }
    let maximum_root = root[0].max(root[1]).max(root[2].max(root[3]));
    if !maximum_root.is_finite() || maximum_root <= 0.0 {
        return None;
    }
    let z = root.map(|value| value / maximum_root);
    if !z.iter().all(|&value| value.is_finite() && value > 0.0) {
        return None;
    }
    let mean_z = profile_sum4(z) * 0.25;
    let dy = z.map(|value| value - mean_z);
    let numerator = profile_sum4(core::array::from_fn(|index| PROFILE_DX[index] * dy[index]));
    let slope = numerator / PROFILE_X_ENERGY;
    let intercept = mean_z - slope * PROFILE_MEAN_X;
    if ![mean_z, numerator, slope, intercept]
        .iter()
        .all(|value| value.is_finite())
        || slope <= 0.0
        || intercept <= 0.0
    {
        return None;
    }
    let residual: [f32; 4] = core::array::from_fn(|index| dy[index] - slope * PROFILE_DX[index]);
    let residual_energy = profile_sum4(residual.map(|value| value * value));
    let z_energy = profile_sum4(z.map(|value| value * value));
    if !residual_energy.is_finite() || !z_energy.is_finite() || z_energy <= 0.0 {
        return None;
    }
    let statistic = residual_energy / z_energy;
    if statistic.is_finite() && statistic >= 0.0 {
        Some(statistic)
    } else {
        None
    }
}

#[inline]
pub(super) fn profile_beta_from_statistic(statistic: Option<f32>) -> f32 {
    let Some(statistic) = statistic else {
        return ALP_GEOMETRY_BLEND_BETA;
    };
    let lower = ALP_GEOMETRY_COMPATIBILITY_LOWER;
    let upper = ALP_GEOMETRY_COMPATIBILITY_UPPER;
    if !lower.is_finite() || lower <= 0.0 || !upper.is_finite() || upper <= lower {
        return ALP_GEOMETRY_BLEND_BETA;
    }
    if statistic <= lower {
        return ALP_GEOMETRY_BLEND_BOOST;
    }
    if statistic >= upper {
        return ALP_GEOMETRY_BLEND_BETA;
    }
    let fraction = (statistic - lower) / (upper - lower);
    ALP_GEOMETRY_BLEND_BOOST - (ALP_GEOMETRY_BLEND_BOOST - ALP_GEOMETRY_BLEND_BETA) * fraction
}

#[inline]
pub(super) fn profile_blend_beta(samples: &[f32; 4]) -> f32 {
    profile_beta_from_statistic(profile_compatibility_statistic(samples))
}

#[inline]
pub(super) fn profile_bounded_move(beta: f32, raw_step: f32) -> f32 {
    // Clip +/-Inf products to the finite move bound.
    // The caller checks this final move and rejects NaN through that check.
    (beta * raw_step).clamp(
        -ALP_GEOMETRY_FINAL_MOVE_CAP_LOG2,
        ALP_GEOMETRY_FINAL_MOVE_CAP_LOG2,
    )
}

/// Blend the capped fallback with the Gaussian model's signed stationary root.
#[inline]
fn profile_model_move(beta: f32, a: f32, derivative: f32, curvature: f32, bounded: f32) -> f32 {
    let weight = ((beta - ALP_GEOMETRY_BLEND_BETA)
        / (ALP_GEOMETRY_BLEND_BOOST - ALP_GEOMETRY_BLEND_BETA))
        .clamp(0.0, 1.0);
    if weight == 0.0 {
        return bounded;
    }
    let discriminant = curvature * curvature - 12.0 * a * derivative;
    if !discriminant.is_finite() || discriminant <= 0.0 {
        return bounded;
    }
    let root_curvature = if curvature < 0.0 {
        -discriminant.sqrt()
    } else {
        discriminant.sqrt()
    };
    // The matching curvature sign makes this denominator non-cancelling.
    let model = -2.0 * derivative / (curvature + root_curvature);
    let proposal = bounded + weight * (model - bounded);
    if proposal.is_finite() {
        proposal
    } else {
        bounded
    }
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    static PUBLIC_CORRECTION_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "std"))]
pub(super) fn reset_public_correction_calls() {
    PUBLIC_CORRECTION_CALLS.with(|calls| calls.set(0));
}

#[cfg(all(test, feature = "std"))]
pub(super) fn public_correction_calls() -> usize {
    PUBLIC_CORRECTION_CALLS.with(std::cell::Cell::get)
}

/// Public σ from the checked log-scale stationary root or its bounded fallback,
/// or `None` when the step is untrusted (the anchor σ is then reported).
#[inline]
pub(super) fn corrected_sigma_from_anchor(
    samples: &[f32; 4],
    anchor_sigma: f32,
    is_maximum: bool,
) -> Option<f32> {
    #[cfg(all(test, feature = "std"))]
    PUBLIC_CORRECTION_CALLS.with(|calls| calls.set(calls.get() + 1));
    let [a, b, c, _] = compute_geometry_poly_coeffs(samples);
    let t = log2f(anchor_sigma / ALP_GEOMETRY_LOG_SIGMA_ORIGIN);
    let derivative = (3.0 * a * t + 2.0 * b) * t + c;
    let curvature = 6.0 * a * t + 2.0 * b;
    let polarity_is_valid = if is_maximum {
        curvature < -ALP_GEOMETRY_CURVATURE_EPSILON
    } else {
        curvature > ALP_GEOMETRY_CURVATURE_EPSILON
    };
    if !t.is_finite() || !derivative.is_finite() || !curvature.is_finite() || !polarity_is_valid {
        return None;
    }

    let raw_step = -derivative / curvature;
    let beta = profile_blend_beta(samples);
    let move_log2 = profile_bounded_move(beta, raw_step);
    let corrected_t = t + move_log2;
    let corrected_sigma =
        ALP_GEOMETRY_LOG_SIGMA_ORIGIN * expf(corrected_t * core::f32::consts::LN_2);
    if !move_log2.is_finite()
        || !corrected_t.is_finite()
        || !corrected_sigma.is_finite()
        || !(ALP_GEOMETRY_FIT_T_MIN..=ALP_GEOMETRY_FIT_T_MAX).contains(&corrected_t)
        || !(ALP_GEOMETRY_ACCEPT_SIGMA_MIN..=ALP_GEOMETRY_ACCEPT_SIGMA_MAX)
            .contains(&corrected_sigma)
    {
        return None;
    }
    let model_move = profile_model_move(beta, a, derivative, curvature, move_log2);
    if model_move != move_log2 {
        let bounded_t = t + move_log2;
        let model_t = t + model_move;
        let bounded_derivative = (3.0 * a * bounded_t + 2.0 * b) * bounded_t + c;
        let bounded_curvature = 6.0 * a * bounded_t + 2.0 * b;
        let model_derivative = (3.0 * a * model_t + 2.0 * b) * model_t + c;
        let model_curvature = 6.0 * a * model_t + 2.0 * b;
        let delta = model_t - bounded_t;
        let gain = delta * (bounded_derivative + delta * (bounded_curvature * 0.5 + a * delta));
        let model_sigma = ALP_GEOMETRY_LOG_SIGMA_ORIGIN * expf(model_t * core::f32::consts::LN_2);
        let model_polarity_matches = if is_maximum {
            model_curvature < -ALP_GEOMETRY_CURVATURE_EPSILON && gain >= 0.0
        } else {
            model_curvature > ALP_GEOMETRY_CURVATURE_EPSILON && gain <= 0.0
        };
        if model_t.is_finite()
            && model_sigma.is_finite()
            && model_derivative.is_finite()
            && bounded_derivative.is_finite()
            && gain.is_finite()
            && model_polarity_matches
            && model_derivative.abs() <= bounded_derivative.abs()
            && (ALP_GEOMETRY_FIT_T_MIN..=ALP_GEOMETRY_FIT_T_MAX).contains(&model_t)
            && (ALP_GEOMETRY_ACCEPT_SIGMA_MIN..=ALP_GEOMETRY_ACCEPT_SIGMA_MAX)
                .contains(&model_sigma)
        {
            return Some(model_sigma);
        }
    }
    Some(corrected_sigma)
}

/// Edge rejection and sub-pixel refinement of an anchor from the nine
/// neighbor scale cubics (row-major 3×3, center at index 4).
///
/// Returns `false` when the point is rejected; on success `kp` holds the
/// refined position and its Q-fit bucket.
pub(super) fn refine_keypoint_2d(
    kp: &mut ScaleSpaceKeypoint,
    coeffs_cache: &[CubicCoefficients; 9],
    curv_ratio_thresh: f32,
) -> bool {
    // The envelope supplies the anchor sigma, with a quadratic fallback for
    // nearly quadratic fits. Spatial refinement keeps that sigma fixed.
    let sigma = kp.sigma;
    let ss = sigma * sigma;
    let sss = ss * sigma;

    let mut p = [0.0f32; 9];
    for idx in 0..9 {
        let [a, b, c, d] = coeffs_cache[idx];
        p[idx] = a * sss + b * ss + c * sigma + d;
    }

    // Edge rejection (Hessian ratio test)
    let p22 = p[4];
    let gxx = p[3] - 2.0 * p22 + p[5];
    let gyy = p[1] - 2.0 * p22 + p[7];
    let gxy = 0.25 * (p[0] + p[8] - p[6] - p[2]);

    let tr_h = gxx + gyy;
    #[allow(clippy::suspicious_operation_groupings)] // det(H) = Hxx·Hyy − Hxy² is correct
    let det_h = gxx * gyy - gxy * gxy;

    // Reject non-positive determinants (saddle points and singular Hessians).
    // det_h < 0 means a saddle point (one eigenvalue +, one −), which is
    // not a blob extremum. Testing .abs() here would let a large negative
    // det (a genuine saddle) pass this check.
    let polarity_matches = if kp.is_max { tr_h < 0.0 } else { tr_h > 0.0 };
    if !gxx.is_finite()
        || !gyy.is_finite()
        || !gxy.is_finite()
        || !tr_h.is_finite()
        || !det_h.is_finite()
        || det_h < 1e-10
        || !polarity_matches
    {
        return false;
    }

    let ratio = tr_h * tr_h / det_h;
    if !ratio.is_finite() || ratio > curv_ratio_thresh {
        return false;
    }

    // Sub-pixel spatial refinement using the generated ALP base table.
    let scale_index = get_q_scale_index(sigma);
    let [w1, w2, w3, w4, w5] = fit_spatial_quadratic(scale_index, &p);

    let denominator = 4.0 * w1 * w2 - w3 * w3;
    if !w1.is_finite()
        || !w2.is_finite()
        || !w3.is_finite()
        || !denominator.is_finite()
        || denominator < 1e-10
    {
        return false;
    }

    // Check w1 independently: a non-zero denominator does not imply a
    // non-zero w1. When w1 ≈ 0, delta_x = ... / (2*w1) produces NaN, which
    // passes the MAX_DISPLACEMENT check silently (NaN > 0.6 == false).
    let polarity_matches = if kp.is_max { w1 < 0.0 } else { w1 > 0.0 };
    if w1.abs() < 1e-10 || !polarity_matches {
        return false;
    }

    let delta_y = (w3 * w4 - 2.0 * w1 * w5) / denominator;
    let delta_x = (-w4 - w3 * delta_y) / (2.0 * w1);

    // Reject any keypoint whose refined position is non-finite.
    if !delta_x.is_finite() || !delta_y.is_finite() {
        return false;
    }

    if delta_x.abs() > MAX_DISPLACEMENT || delta_y.abs() > MAX_DISPLACEMENT {
        return false;
    }

    kp.x += delta_x;
    kp.y += delta_y;
    kp.sigma = sigma;
    kp.scale_index = scale_index;

    true
}

#[inline(always)]
pub(super) fn orientation_histogram_bin(angle: f32) -> usize {
    let angle = if angle < 0.0 { angle + TWO_PI } else { angle };
    let raw_bin = (angle * (ORI_HIST_BINS as f32 / TWO_PI) + 0.5) as usize;
    if raw_bin < ORI_HIST_BINS {
        raw_bin
    } else {
        raw_bin % ORI_HIST_BINS
    }
}
