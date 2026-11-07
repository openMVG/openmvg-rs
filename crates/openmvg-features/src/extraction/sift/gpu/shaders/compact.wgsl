// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Compacts the fixed-stride orientation peaks of one octave into a dense
// descriptor-key buffer. One invocation owns one keypoint and preserves
// keypoint order and peak (bin) order, so the result matches detector order.
// The orientation values are copied unchanged.

struct CompactParams {
    key_count: u32,
    max_orientations: u32,
    _pad0: u32,
    _pad1: u32,
};

struct OriKey {
    x: f32,
    y: f32,
    sigma: f32,
    level: u32,
};

struct DescKey {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    level: u32,
};

@group(0) @binding(0) var<uniform> p: CompactParams;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> orientations: array<f32>;
@group(0) @binding(4) var<storage, read> keys: array<OriKey>;
@group(0) @binding(5) var<storage, read_write> descriptor_keys: array<DescKey>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= p.key_count) {
        return;
    }
    let count = min(counts[id], p.max_orientations);
    let base = offsets[id];
    let key = keys[id];
    for (var slot = 0u; slot < count; slot += 1u) {
        let orientation = orientations[id * p.max_orientations + slot];
        descriptor_keys[base + slot] = DescKey(
            key.x,
            key.y,
            key.sigma,
            orientation,
            key.level,
        );
    }
}
