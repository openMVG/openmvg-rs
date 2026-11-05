// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// 4x4x8 SIFT descriptor, computed from central-difference gradients of one
// Gaussian level. One workgroup owns one keypoint. Each of the 16 threads owns
// one spatial cell and its 8 orientation bins, and scans only the samples that
// can contribute to that cell, so every sample is evaluated at most four times
// instead of once per histogram bin. Each cell histogram is accumulated in a
// fixed row-major order, and the 128 bins are reduced only at the end, keeping
// the result deterministic. The final L2/RootSIFT normalization matches
// `normalize_and_finalize_descriptor`.

const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;
const DESCRIPTOR_SCALE: f32 = 6.0;
const SQRT2: f32 = 1.4142135623730951;
const BINS: u32 = 128u;
const CELLS_PER_SIDE: u32 = 4u;
const ORIENTATION_BINS: u32 = 8u;
const THREADS_PER_KEYPOINT: u32 = 16u;

struct DescParams {
    rows_cols: vec2<u32>,
    level: u32,
    root_sift: u32,
    key_count: u32,
    out_off: u32,
    gauss_off: u32,
    _pad: u32,
};

struct DescKey {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    level: u32,
};

@group(0) @binding(0) var<uniform> p: DescParams;
@group(0) @binding(1) var<storage, read> gauss: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<DescKey>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> valid: array<u32>;

var<workgroup> hist: array<f32, 128>;

fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let mn = min(ax, ay);
    let mx = max(ax, ay);
    var r = 0.0;
    if (mx > 0.0) {
        r = mn / mx;
    }
    var phi = r * (0.7853981633974483 - (r - 1.0) * (0.2447 + 0.0663 * r));
    if (ay > ax) {
        phi = 1.5707963267948966 - phi;
    }
    if (x < 0.0) {
        phi = PI - phi;
    }
    if (y < 0.0) {
        phi = -phi;
    }
    return phi;
}

fn gradient_at(col: i32, row: i32, level: u32) -> vec2<f32> {
    let cols = i32(p.rows_cols.y);
    let rows = i32(p.rows_cols.x);
    if (col <= 0 || row <= 0 || col >= cols - 1 || row >= rows - 1) {
        return vec2(0.0, 0.0);
    }
    let plane = p.rows_cols.x * p.rows_cols.y;
    let base = p.gauss_off + level * plane;
    let stride = p.rows_cols.x;
    let gx = (gauss[base + u32(col + 1) * stride + u32(row)]
        - gauss[base + u32(col - 1) * stride + u32(row)]) * 0.5;
    let gy = (gauss[base + u32(col) * stride + u32(row + 1)]
        - gauss[base + u32(col) * stride + u32(row - 1)]) * 0.5;
    return vec2(gx, gy);
}

@compute @workgroup_size(16, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let id = wid.x;
    if (id >= p.key_count) {
        return;
    }
    let tid = lid.x;
    let cell_i = tid / CELLS_PER_SIDE;
    let cell_j = tid % CELLS_PER_SIDE;
    var hist_cell = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    let key = keys[id];
    let sigma = key.sigma;

    if (sigma >= 1e-6) {
        let cos_t = cos(-key.orientation);
        let sin_t = sin(-key.orientation);
        let sigma_scale = DESCRIPTOR_SCALE * sigma;
        let r = (1.0 + 0.25) * sigma_scale;
        let rp = r * SQRT2;
        let inv_bin_scale = 4.0 / (2.0 * sigma_scale);
        let bin_scale = 1.0 / inv_bin_scale;
        let half_bins = 1.5;
        let inv_sigma_sq = 1.0 / (2.0 * sigma_scale * sigma_scale);
        let angle_to_bin = 8.0 / TWO_PI;

        // A sample contributes to spatial bin i when floor(alpha) is i
        // (weight 1 - frac) or i - 1 (weight frac). Thus cell (i, j) only
        // needs alpha in [i - 1, i + 1] and beta in [j - 1, j + 1]. Transform
        // that rotated rectangle back to image coordinates and scan its
        // axis-aligned bounding box.
        let alpha_min = f32(cell_i) - 1.0;
        let alpha_max = f32(cell_i) + 1.0;
        let beta_min = f32(cell_j) - 1.0;
        let beta_max = f32(cell_j) + 1.0;
        let rot_tol = bin_scale * 1.0e-4;
        let xr_min = (alpha_min - half_bins) * bin_scale - rot_tol;
        let xr_max = (alpha_max - half_bins) * bin_scale + rot_tol;
        let yr_min = (beta_min - half_bins) * bin_scale - rot_tol;
        let yr_max = (beta_max - half_bins) * bin_scale + rot_tol;

        let x0 = key.x + cos_t * xr_min + sin_t * yr_min;
        let y0 = key.y - sin_t * xr_min + cos_t * yr_min;
        let x1 = key.x + cos_t * xr_min + sin_t * yr_max;
        let y1 = key.y - sin_t * xr_min + cos_t * yr_max;
        let x2 = key.x + cos_t * xr_max + sin_t * yr_min;
        let y2 = key.y - sin_t * xr_max + cos_t * yr_min;
        let x3 = key.x + cos_t * xr_max + sin_t * yr_max;
        let y3 = key.y - sin_t * xr_max + cos_t * yr_max;

        let pixel_tol = 1.0e-3;
        let min_x = min(min(x0, x1), min(x2, x3)) - pixel_tol;
        let max_x = max(max(x0, x1), max(x2, x3)) + pixel_tol;
        let min_y = min(min(y0, y1), min(y2, y3)) - pixel_tol;
        let max_y = max(max(y0, y1), max(y2, y3)) + pixel_tol;

        let rows = f32(p.rows_cols.x);
        let cols = f32(p.rows_cols.y);
        let row_min = u32(clamp(floor(min_y), 0.0, rows));
        let row_max = u32(clamp(ceil(max_y), 0.0, rows));
        let col_min = u32(clamp(floor(min_x), 0.0, cols));
        let col_max = u32(clamp(ceil(max_x), 0.0, cols));

        for (var row = row_min; row < row_max; row += 1u) {
            let y_ref = f32(row) - key.y;
            let row_gauss = exp(-y_ref * y_ref * inv_sigma_sq);
            let xr_row = -sin_t * y_ref;
            let yr_row = cos_t * y_ref;
            for (var col = col_min; col < col_max; col += 1u) {
                let x_ref = f32(col) - key.x;
                let col_gauss = exp(-x_ref * x_ref * inv_sigma_sq);
                let xr = cos_t * x_ref + xr_row;
                let yr = sin_t * x_ref + yr_row;
                if (max(abs(xr), abs(yr)) >= r) {
                    continue;
                }

                let gradient = gradient_at(i32(col), i32(row), key.level);
                let mag_sq = dot(gradient, gradient);
                if (mag_sq < 1e-12) {
                    continue;
                }
                let mag = sqrt(mag_sq);
                let theta = fast_atan2(gradient.y, gradient.x);
                let m = mag * row_gauss * col_gauss;

                var ori = theta - key.orientation;
                if (ori < 0.0) {
                    ori += TWO_PI;
                } else if (ori >= TWO_PI) {
                    ori -= TWO_PI;
                }

                let alpha = xr * inv_bin_scale + half_bins;
                let beta = yr * inv_bin_scale + half_bins;
                let gamma = ori * angle_to_bin;
                if (!(gamma < 3.0e38) || !(gamma > -3.0e38)) {
                    continue;
                }

                let alpha_floor = floor(alpha);
                let beta_floor = floor(beta);
                let gamma_floor = floor(gamma);
                let i0 = i32(alpha_floor);
                let j0 = i32(beta_floor);
                let alpha_frac = alpha - alpha_floor;
                let beta_frac = beta - beta_floor;
                let gamma_frac = gamma - gamma_floor;

                let i0_match = i0 == i32(cell_i);
                let i1_match = i0 + 1 == i32(cell_i);
                let j0_match = j0 == i32(cell_j);
                let j1_match = j0 + 1 == i32(cell_j);
                if (!(i0_match || i1_match) || !(j0_match || j1_match)) {
                    continue;
                }

                let wi = select(0.0, 1.0 - alpha_frac, i0_match)
                    + select(0.0, alpha_frac, i1_match);
                let wj = select(0.0, 1.0 - beta_frac, j0_match)
                    + select(0.0, beta_frac, j1_match);
                let spatial = wi * wj * m;
                if (spatial == 0.0) {
                    continue;
                }

                let k_left = u32(gamma_floor) % ORIENTATION_BINS;
                let k_right = (k_left + 1u) % ORIENTATION_BINS;
                let gamma_inv = 1.0 - gamma_frac;
                hist_cell[k_left] += gamma_inv * spatial;
                hist_cell[k_right] += gamma_frac * spatial;
            }
        }
    }

    for (var k = 0u; k < ORIENTATION_BINS; k += 1u) {
        hist[cell_i * 32u + cell_j * 8u + k] = hist_cell[k];
    }
    workgroupBarrier();

    if (tid == 0u) {
        var norm_sq = 0.0;
        for (var i = 0u; i < BINS; i += 1u) {
            norm_sq += hist[i] * hist[i];
        }
        let norm = sqrt(norm_sq);
        let out_base = p.out_off + id * BINS;
        if (!(norm > 1e-7) || !(norm < 3.0e38)) {
            valid[id] = 0u;
            for (var i = 0u; i < BINS; i += 1u) {
                out[out_base + i] = 0.0;
            }
            return;
        }

        let inv_norm = 1.0 / norm;
        var norm2_sq = 0.0;
        if (p.root_sift == 1u) {
            var l1 = 0.0;
            for (var i = 0u; i < BINS; i += 1u) {
                l1 += abs(hist[i] * inv_norm);
            }
            l1 = max(l1, 1e-7);
            for (var i = 0u; i < BINS; i += 1u) {
                let v = sqrt(max(hist[i] * inv_norm / l1, 0.0));
                norm2_sq += v * v;
            }
            let scale = 512.0 / max(sqrt(norm2_sq), 1e-7);
            for (var i = 0u; i < BINS; i += 1u) {
                let v = sqrt(max(hist[i] * inv_norm / l1, 0.0));
                out[out_base + i] = min(v * scale, 255.0);
            }
        } else {
            for (var i = 0u; i < BINS; i += 1u) {
                let v = min(hist[i] * inv_norm, 0.2);
                norm2_sq += v * v;
            }
            let scale = 512.0 / max(sqrt(norm2_sq), 1e-7);
            for (var i = 0u; i < BINS; i += 1u) {
                let v = min(hist[i] * inv_norm, 0.2);
                out[out_base + i] = min(v * scale, 255.0);
            }
        }
        valid[id] = 1u;
    }
}
