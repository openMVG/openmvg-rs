// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Difference of consecutive Gaussian levels.

struct DogParams {
    rows_cols: vec2<u32>,
    src_off: u32,
    src_next_off: u32,
    dst_off: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> p: DogParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (col >= p.rows_cols.y || row >= p.rows_cols.x) {
        return;
    }
    let index = col * p.rows_cols.x + row;
    dst[p.dst_off + index] = src[p.src_next_off + index] - src[p.src_off + index];
}
