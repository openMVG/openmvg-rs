// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Initial Gaussian level: bilinear 2x upsampling for `first_octave = -1`,
// or a plain copy for `first_octave = 0`. The source is already normalized to
// [0, 1] by the host. Images are column-major, matching `openmvg-features`.

struct InitParams {
    src_rows_cols: vec2<u32>,
    dst_rows_cols: vec2<u32>,
    src_off: u32,
    dst_off: u32,
    upscale: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: InitParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= p.dst_rows_cols.y || row >= p.dst_rows_cols.x) {
        return;
    }

    let dst_index = p.dst_off + col * p.dst_rows_cols.x + row;
    if (p.upscale == 0u) {
        dst[dst_index] = src[p.src_off + col * p.src_rows_cols.x + row];
        return;
    }

    let src_x = f32(col) * 0.5;
    let src_y = f32(row) * 0.5;
    let ix = min(u32(floor(src_x)), p.src_rows_cols.y - 1u);
    let iy = min(u32(floor(src_y)), p.src_rows_cols.x - 1u);
    let ix1 = min(ix + 1u, p.src_rows_cols.y - 1u);
    let iy1 = min(iy + 1u, p.src_rows_cols.x - 1u);
    let fx = src_x - f32(ix);
    let fy = src_y - f32(iy);

    let stride = p.src_rows_cols.x;
    let a = src[p.src_off + ix * stride + iy];
    let b = src[p.src_off + ix1 * stride + iy];
    let c = src[p.src_off + ix * stride + iy1];
    let d = src[p.src_off + ix1 * stride + iy1];

    dst[dst_index] = (1.0 - fy) * ((1.0 - fx) * a + fx * b)
        + fy * ((1.0 - fx) * c + fx * d);
}
