// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Keep every second sample, matching `scale_space::decimate_2x`.

struct DecimateParams {
    src_rows_cols: vec2<u32>,
    dst_rows_cols: vec2<u32>,
    src_off: u32,
    dst_off: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: DecimateParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (col >= p.dst_rows_cols.y || row >= p.dst_rows_cols.x) {
        return;
    }
    let source = p.src_off + (2u * col) * p.src_rows_cols.x + 2u * row;
    let destination_index = p.dst_off + col * p.dst_rows_cols.x + row;
    dst[destination_index] = src[source];
}
