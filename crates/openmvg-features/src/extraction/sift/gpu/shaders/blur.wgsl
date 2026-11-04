// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// One separable Gaussian convolution pass. `vertical == 1` convolves along
// rows (the contiguous columns of the column-major image), `vertical == 0`
// convolves across columns. The accumulation order and edge clamping match
// `openmvg_features::extraction::convolution`.

struct BlurParams {
    rows_cols: vec2<u32>,
    src_off: u32,
    dst_off: u32,
    radius: u32,
    wt_off: u32,
    vertical: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: BlurParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;

fn sample_at(col: i32, row: i32) -> f32 {
    let c = clamp(col, 0, i32(p.rows_cols.y) - 1);
    let r = clamp(row, 0, i32(p.rows_cols.x) - 1);
    return src[p.src_off + u32(c) * p.rows_cols.x + u32(r)];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= p.rows_cols.y || row >= p.rows_cols.x) {
        return;
    }

    let ci = i32(col);
    let ri = i32(row);
    var acc = sample_at(ci, ri) * weights[p.wt_off];
    for (var k = 1u; k <= p.radius; k += 1u) {
        let offset = i32(k);
        var lower = 0.0;
        var upper = 0.0;
        if (p.vertical == 1u) {
            lower = sample_at(ci, ri - offset);
            upper = sample_at(ci, ri + offset);
        } else {
            lower = sample_at(ci - offset, ri);
            upper = sample_at(ci + offset, ri);
        }
        acc += (lower + upper) * weights[p.wt_off + k];
    }
    dst[p.dst_off + col * p.rows_cols.x + row] = acc;
}
