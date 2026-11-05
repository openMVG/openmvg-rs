// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3x3x3 DoG extrema, the discrete candidate stage of
// `find_3d_discrete_extrema`. One invocation per interior voxel.

struct ExtremaParams {
    rows_cols: vec2<u32>,
    levels: u32,
    threshold: f32,
    octave: u32,
    cap: u32,
    _pad0: u32,
    _pad1: u32,
};

struct Candidate {
    level: u32,
    i: i32,
    j: i32,
    value: f32,
    octave: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: ExtremaParams;
@group(0) @binding(1) var<storage, read> dog: array<f32>;
@group(0) @binding(2) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(3) var<storage, read_write> count: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let level = gid.z + 1u;
    if (col == 0u || col + 1u >= p.rows_cols.y
        || row == 0u || row + 1u >= p.rows_cols.x
        || level + 1u >= p.levels) {
        return;
    }

    let rows = p.rows_cols.x;
    let cols = p.rows_cols.y;
    let plane = rows * cols;
    let center = dog[level * plane + col * rows + row];
    if (abs(center) <= p.threshold) {
        return;
    }
    let is_max = center > 0.0;

    for (var ds = 0u; ds < 3u; ds += 1u) {
        let s = i32(level) + i32(ds) - 1;
        for (var di = 0u; di < 3u; di += 1u) {
            let c = i32(col) + i32(di) - 1;
            for (var dj = 0u; dj < 3u; dj += 1u) {
                if (ds == 1u && di == 1u && dj == 1u) {
                    continue;
                }
                let r = i32(row) + i32(dj) - 1;
                let neighbor = dog[u32(s) * plane + u32(c) * rows + u32(r)];
                if ((is_max && center <= neighbor) || (!is_max && center >= neighbor)) {
                    return;
                }
            }
        }
    }

    let slot = atomicAdd(&count, 1u);
    if (slot < p.cap) {
        candidates[slot] = Candidate(level, i32(col), i32(row), center, p.octave, 0u);
    }
}
