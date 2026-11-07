// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// 36-bin orientation histogram, six [1/4, 1/2, 1/4] smoothing passes and
// parabolic peak interpolation. One invocation per keypoint; peaks are written
// in bin order so the host can preserve detector order.

const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;
const ORI_BINS: u32 = 36u;
const MAX_PEAKS: u32 = 36u;
const PREFIX_PEAKS: u32 = 8u;

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
@group(0) @binding(5) var<storage, read_write> prefix: array<f32>;
// `work[0]` of `order.wgsl`: the number of keys actually written; the grid is
// sized for the refined count, which bounds it.
@group(0) @binding(6) var<storage, read> kept: array<u32>;

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

// One workgroup owns one keypoint. The window is processed in chunks of
// `CHUNK` samples: every invocation evaluates one sample's gradient, weight
// and bin, then the invocation that owns a bin adds that chunk's contributions
// for its bin in sample order. Each bin therefore sums exactly the same terms
// in exactly the same row-major order as the serial loop of
// `build_orientation_histogram`, while the expensive per-sample work runs in
// parallel. Smoothing and peak extraction then run on one invocation.

const CHUNK: u32 = 64u;
const NO_BIN: u32 = 0xffffffffu;

var<workgroup> chunk_bin: array<u32, 64>;
var<workgroup> chunk_val: array<f32, 64>;
var<workgroup> hist_shared: array<f32, 36>;
var<workgroup> sample_count: u32;
var<workgroup> key_limit: u32;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let id = wid.x + wid.y * 65535u;
    let tid = lid.x;
    // A storage load is not uniform for WGSL's analysis; share it through
    // workgroup memory so the early return keeps the barriers below uniform.
    if (tid == 0u) {
        key_limit = min(p.key_count, kept[0]);
    }
    if (id >= workgroupUniformLoad(&key_limit)) {
        return;
    }
    let key = keys[id];
    let sigma = key.sigma;
    let window_sigma = 1.5 * sigma;
    let radius = i32(ceil(3.0 * window_sigma));
    let side = u32(2 * radius + 1);
    let inv_2sigma_sq = 1.0 / (2.0 * window_sigma * window_sigma);
    let center_i = i32(floor(key.y + 0.5));
    let center_j = i32(floor(key.x + 0.5));
    if (tid == 0u) {
        sample_count = select(0u, side * side, sigma >= 1e-6);
    }
    let total = workgroupUniformLoad(&sample_count);

    var bin_sum = 0.0;
    for (var base = 0u; base < total; base += CHUNK) {
        let sample = base + tid;
        var bin = NO_BIN;
        var value = 0.0;
        if (sample < total) {
            let sample_row = center_i + i32(sample / side) - radius;
            let sample_col = center_j + i32(sample % side) - radius;
            let gradient = gradient_at(sample_col, sample_row, key.level);
            let mag = length(gradient);
            if (mag != 0.0) {
                let dx = f32(sample_col) - key.x;
                let dy = f32(sample_row) - key.y;
                let weight = exp(-(dx * dx + dy * dy) * inv_2sigma_sq);
                let angle = atan2(gradient.y, gradient.x);
                let angle_pos = angle - TWO_PI * floor(angle / TWO_PI);
                bin = u32(angle_pos / TWO_PI * f32(ORI_BINS) + 0.5) % ORI_BINS;
                value = mag * weight;
            }
        }
        chunk_bin[tid] = bin;
        chunk_val[tid] = value;
        workgroupBarrier();
        if (tid < ORI_BINS) {
            let count = min(CHUNK, total - base);
            for (var k = 0u; k < count; k += 1u) {
                if (chunk_bin[k] == tid) {
                    bin_sum += chunk_val[k];
                }
            }
        }
        workgroupBarrier();
    }
    if (tid < ORI_BINS) {
        hist_shared[tid] = bin_sum;
    }
    workgroupBarrier();
    if (tid != 0u) {
        return;
    }

    var hist: array<f32, 36>;
    for (var b = 0u; b < ORI_BINS; b += 1u) {
        hist[b] = hist_shared[b];
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
            if (slot < PREFIX_PEAKS) {
                prefix[id * PREFIX_PEAKS + slot] = orientation;
            }
            slot += 1u;
        }
    }
    counts[id] = slot;
}
