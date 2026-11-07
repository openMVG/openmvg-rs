// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Both separable Gaussian passes of `blur.wgsl` in one dispatch, for kernel
// radii up to `MAX_RADIUS`.
//
//   source level ──(along rows)──> tile[cols + 2R][rows]  (workgroup memory)
//                                    │
//                                    └──(across columns)──> destination level
//                                                           and DoG plane
//
// One workgroup owns a TILE x BLOCK_COLS block of outputs. It first runs the
// along-rows pass for the block's rows and for its columns plus an R-column
// halo on each side, keeping the f32 results in workgroup memory, then runs
// the across-columns pass from there. Every tap, accumulation order and edge
// clamp is the one `blur.wgsl` uses (halo columns are evaluated at the clamped
// column, exactly what the second pass of `blur.wgsl` reads), so the output is
// the same as the two-pass path without writing and re-reading the
// intermediate plane. With `dog == 1` the DoG plane is the new value minus the
// source sample, which is the previous Gaussian level.
//
// With `in_place == 1` the source plane lives in the destination buffer (one
// Gaussian level read to write the next); it is read through `dst`, because
// one buffer cannot be bound read-only and read-write in the same dispatch.

const TILE: u32 = 32u; // rows per block, one per invocation along x
const BLOCK_COLS: u32 = 64u; // columns per block
const MAX_RADIUS: u32 = 16u;
const TILE_COLS: u32 = 96u; // BLOCK_COLS + 2 * MAX_RADIUS

struct BlurParams {
    rows_cols: vec2<u32>,
    src_off: u32,
    dst_off: u32,
    radius: u32,
    wt_off: u32,
    vertical: u32,
    dog: u32,
    dog_off: u32,
    in_place: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> p: BlurParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dog: array<f32>;

var<workgroup> tile: array<f32, 3072>; // TILE_COLS * TILE, indexed [col][row]

fn source_index(index: u32) -> f32 {
    if (p.in_place == 1u) {
        return dst[p.src_off + index];
    }
    return src[p.src_off + index];
}

fn source_at(col: u32, row: i32) -> f32 {
    let r = clamp(row, 0, i32(p.rows_cols.x) - 1);
    return source_index(col * p.rows_cols.x + u32(r));
}

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = p.rows_cols.x;
    let cols = p.rows_cols.y;
    let radius = p.radius;
    let row0 = wid.x * TILE;
    let col0 = wid.y * BLOCK_COLS;
    let row = row0 + lid.x;
    let ri = i32(row);

    // Along-rows pass for TILE + 2R columns of this block's rows.
    let span = BLOCK_COLS + 2u * radius;
    for (var c = lid.y; c < span; c += 8u) {
        let col = u32(clamp(i32(col0 + c) - i32(radius), 0, i32(cols) - 1));
        var acc = source_at(col, ri) * weights[p.wt_off];
        for (var k = 1u; k <= radius; k += 1u) {
            let offset = i32(k);
            let lower = source_at(col, ri - offset);
            let upper = source_at(col, ri + offset);
            acc += (lower + upper) * weights[p.wt_off + k];
        }
        tile[c * TILE + lid.x] = acc;
    }
    workgroupBarrier();

    if (row >= rows) {
        return;
    }
    // Across-columns pass from the tile.
    for (var c = lid.y; c < BLOCK_COLS; c += 8u) {
        let col = col0 + c;
        if (col >= cols) {
            break;
        }
        let center = c + radius;
        var acc = tile[center * TILE + lid.x] * weights[p.wt_off];
        for (var k = 1u; k <= radius; k += 1u) {
            let lower = tile[(center - k) * TILE + lid.x];
            let upper = tile[(center + k) * TILE + lid.x];
            acc += (lower + upper) * weights[p.wt_off + k];
        }
        let index = col * rows + row;
        dst[p.dst_off + index] = acc;
        if (p.dog == 1u) {
            dog[p.dog_off + index] = acc - source_index(index);
        }
    }
}
