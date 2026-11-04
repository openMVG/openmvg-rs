// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// 36-bin orientation histogram, six [1/4, 1/2, 1/4] smoothing passes and
// parabolic peak interpolation. One invocation per keypoint; peaks are written
// in bin order so the host can preserve detector order.

const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;
const ORI_BINS: u32 = 36u;
const MAX_PEAKS: u32 = 36u;

struct OriParams {
    rows_cols: vec2<u32>,
    level: u32,
    key_count: u32,
    gauss_off: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct OriKey {
    x: f32,
    y: f32,
    sigma: f32,
    level: u32,
};

@group(0) @binding(0) var<uniform> p: OriParams;
@group(0) @binding(1) var<storage, read> gauss: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<OriKey>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> orientations: array<f32>;

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

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= p.key_count) {
        return;
    }
    let key = keys[id];
    let sigma = key.sigma;
    var hist: array<f32, 36>;
    for (var b = 0u; b < ORI_BINS; b += 1u) {
        hist[b] = 0.0;
    }

    if (sigma >= 1e-6) {
        let window_sigma = 1.5 * sigma;
        let radius = i32(ceil(3.0 * window_sigma));
        let inv_2sigma_sq = 1.0 / (2.0 * window_sigma * window_sigma);
        let center_i = i32(floor(key.y + 0.5));
        let center_j = i32(floor(key.x + 0.5));
        for (var di = -radius; di <= radius; di += 1) {
            for (var dj = -radius; dj <= radius; dj += 1) {
                let sample_row = center_i + di;
                let sample_col = center_j + dj;
                let gradient = gradient_at(sample_col, sample_row, key.level);
                let mag = length(gradient);
                if (mag == 0.0) {
                    continue;
                }
                let dx = f32(sample_col) - key.x;
                let dy = f32(sample_row) - key.y;
                let weight = exp(-(dx * dx + dy * dy) * inv_2sigma_sq);
                let angle = atan2(gradient.y, gradient.x);
                let angle_pos = angle - TWO_PI * floor(angle / TWO_PI);
                let bin = u32(angle_pos / TWO_PI * f32(ORI_BINS) + 0.5) % ORI_BINS;
                hist[bin] += mag * weight;
            }
        }
    }

    for (var smoothing = 0u; smoothing < 6u; smoothing += 1u) {
        let first = hist[0];
        let last = hist[ORI_BINS - 1u];
        var prev = last;
        for (var b = 0u; b < ORI_BINS; b += 1u) {
            var next = first;
            if (b + 1u < ORI_BINS) {
                next = hist[b + 1u];
            }
            let smoothed = prev * 0.25 + hist[b] * 0.5 + next * 0.25;
            prev = hist[b];
            hist[b] = smoothed;
        }
    }

    var max_mag = 0.0;
    for (var b = 0u; b < ORI_BINS; b += 1u) {
        max_mag = max(max_mag, hist[b]);
    }
    if (max_mag <= 0.0) {
        counts[id] = 0u;
        return;
    }

    let threshold = 0.8 * max_mag;
    var slot = 0u;
    for (var b = 0u; b < ORI_BINS; b += 1u) {
        let val = hist[b];
        let prev = hist[(b + ORI_BINS - 1u) % ORI_BINS];
        let next = hist[(b + 1u) % ORI_BINS];
        if (val < threshold || val <= prev || val <= next) {
            continue;
        }
        let denom = prev - 2.0 * val + next;
        var delta = 0.0;
        if (abs(denom) >= 1e-6) {
            delta = clamp(0.5 * (prev - next) / denom, -1.0, 1.0);
        }
        var orientation = (f32(b) + delta) * TWO_PI / f32(ORI_BINS);
        if (orientation > PI) {
            orientation -= TWO_PI;
        }
        if (slot < MAX_PEAKS) {
            orientations[id * MAX_PEAKS + slot] = orientation;
            slot += 1u;
        }
    }
    counts[id] = slot;
}
