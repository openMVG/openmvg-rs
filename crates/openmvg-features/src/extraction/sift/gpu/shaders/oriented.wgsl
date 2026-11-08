// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Turns one octave's orientation peaks into dense oriented features, on the
// device, so the host does not need the peak counts before describing them.
//
//   counts[kept] ──scan──> offsets[kept], header = (total, overflow)
//   features + orientations ──compact──> records[total] and keys[total]
//
// `scan` is the exclusive prefix sum of the peak counts of the kept features
// (`work[0]` of `order.wgsl`), in feature order. `compact` writes peak `s` of
// feature `i` at `offsets[i] + s`, so the output keeps detector order and bin
// order, exactly the order the host built before. `records` are in image
// units for the host; `keys` are in octave units for `describe.wgsl`. The
// octave delta is a power of two, so `x * inv_delta` equals the host's
// division bit for bit. `header` holds the written count, an overflow flag
// and the full count: outputs beyond the host's bound `k` are dropped, so the
// host can grow the buffers and run again.

const MAX_PEAKS: u32 = 36u;
const SCAN_THREADS: u32 = 256u;
const SCAN_TILE: u32 = 1024u; // SCAN_THREADS * 4

struct OrientedParams {
    n: u32,
    k: u32,
    inv_delta: f32,
    _pad0: u32,
};

struct Feature {
    x: f32,
    y: f32,
    sigma: f32,
    peak: f32,
    level: u32,
};

struct Record {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    peak: f32,
    level: u32,
};

struct DescKey {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    level: u32,
};

@group(0) @binding(0) var<uniform> p: OrientedParams;
@group(0) @binding(1) var<storage, read> work: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> orientations: array<f32>;
@group(0) @binding(4) var<storage, read> features: array<Feature>;
@group(0) @binding(5) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> keys: array<DescKey>;
@group(0) @binding(7) var<storage, read_write> records: array<Record>;
@group(0) @binding(8) var<storage, read_write> header: array<u32>;

var<workgroup> partial: array<u32, 256>;
var<workgroup> kept_count: u32;

fn kept() -> u32 {
    return min(work[0], p.n);
}

fn peaks(index: u32) -> u32 {
    return min(counts[index], MAX_PEAKS);
}

@compute @workgroup_size(256, 1, 1)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    if (tid == 0u) {
        kept_count = kept();
    }
    let total_len = workgroupUniformLoad(&kept_count);
    var carry = 0u;
    for (var base = 0u; base < total_len; base += SCAN_TILE) {
        var values = vec4<u32>(0u);
        for (var j = 0u; j < 4u; j += 1u) {
            let index = base + tid * 4u + j;
            if (index < total_len) {
                values[j] = peaks(index);
            }
        }
        let sum = values.x + values.y + values.z + values.w;
        partial[tid] = sum;
        workgroupBarrier();
        for (var stride = 1u; stride < SCAN_THREADS; stride *= 2u) {
            var add = 0u;
            if (tid >= stride) {
                add = partial[tid - stride];
            }
            workgroupBarrier();
            partial[tid] += add;
            workgroupBarrier();
        }
        var running = carry + partial[tid] - sum;
        for (var j = 0u; j < 4u; j += 1u) {
            let index = base + tid * 4u + j;
            if (index < total_len) {
                offsets[index] = running;
            }
            running += values[j];
        }
        carry += partial[SCAN_THREADS - 1u];
        workgroupBarrier();
    }
    if (tid == 0u) {
        header[0] = min(carry, p.k);
        header[1] = select(0u, 1u, carry > p.k);
        header[2] = carry;
    }
}

@compute @workgroup_size(64, 1, 1)
fn compact(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let id = gid.x + gid.y * nwg.x * 64u;
    if (id >= kept()) {
        return;
    }
    let feature = features[id];
    let base = offsets[id];
    for (var slot = 0u; slot < peaks(id); slot += 1u) {
        let q = base + slot;
        if (q >= p.k) {
            return;
        }
        let orientation = orientations[id * MAX_PEAKS + slot];
        records[q] = Record(
            feature.x,
            feature.y,
            feature.sigma,
            orientation,
            feature.peak,
            feature.level,
        );
        keys[q] = DescKey(
            feature.x * p.inv_delta,
            feature.y * p.inv_delta,
            feature.sigma * p.inv_delta,
            orientation,
            feature.level,
        );
    }
}
