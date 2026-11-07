// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Puts one octave's refined samples in detector order, removes duplicates and
// writes the surviving features and their orientation keys, all on the GPU.
//
//   refined (atomic order)
//     │ hist, scan_buckets, scatter, bucket_sort      counting sort by the
//     ▼                                                candidate (scale, row, col)
//   sorted ranks ─ hash_insert, flag ─> kept flags     first rank per refined
//     │ scan (kept)                                    sample wins, as in
//     ▼                                                `refine_keypoints`
//   emit ─> features[q], keys[q]                       q = rank among kept
//
// The CPU scans candidates in (scale, row, column) order, refines each one and
// keeps a refined sample only the first time it is reached, whether or not
// that first refinement passes the later checks. Here, a counting sort over
// (scale, row) buckets followed by an insertion sort by column inside each
// bucket reproduces the candidate order (every candidate is a distinct
// sample, so the order is total). An open-addressing hash table keyed by the
// refined sample keeps, through `atomicMax(!rank)`, the smallest rank that
// reached it. Every decision is a pure function of the data, so the output is
// deterministic.
//
// All scratch arrays share one `work` buffer at the offsets in `OrderParams`;
// `work[0]` receives the number of kept features.

struct OrderParams {
    n: u32,
    rows: u32,
    cols: u32,
    buckets: u32,
    _pad0: u32,
    hash_mask: u32,
    inv_delta: f32,
    scan_len: u32,
    scan_src: u32,
    scan_dst: u32,
    scan_copy: u32,
    off_count: u32,
    off_start: u32,
    off_cursor: u32,
    off_sorted: u32,
    off_hash_keys: u32,
    off_hash_vals: u32,
    off_slot: u32,
    off_kept: u32,
    off_pos: u32,
};

struct Refined {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    value: f32,
    octave: u32,
    level: u32,
    i: i32,
    j: i32,
    orig_level: u32,
    orig_i: i32,
    orig_j: i32,
    accepted: u32,
    _pad1: u32,
};

struct Feature {
    x: f32,
    y: f32,
    sigma: f32,
    peak: f32,
    level: u32,
};

struct OriKey {
    x: f32,
    y: f32,
    sigma: f32,
    level: u32,
};

const NO_COPY: u32 = 0xffffffffu;
const SCAN_THREADS: u32 = 256u;
const SCAN_TILE: u32 = 1024u; // SCAN_THREADS * 4

@group(0) @binding(0) var<uniform> p: OrderParams;
@group(0) @binding(1) var<storage, read> refined: array<Refined>;
@group(0) @binding(2) var<storage, read_write> work: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> features: array<Feature>;
@group(0) @binding(4) var<storage, read_write> keys: array<OriKey>;

var<workgroup> partial: array<u32, 256>;

fn bucket_of(r: Refined) -> u32 {
    return (r.orig_level - 1u) * p.rows + u32(r.orig_j);
}

// Candidate order within a bucket: scale and row are equal, so the column.
fn candidate_col(index: u32) -> i32 {
    return refined[index].orig_i;
}

fn item(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

@compute @workgroup_size(64, 1, 1)
fn hist(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let id = item(gid, nwg);
    if (id >= p.n) {
        return;
    }
    atomicAdd(&work[p.off_count + bucket_of(refined[id])], 1u);
}

// Exclusive prefix sum of `scan_len` words at `scan_src` into `scan_dst`
// (and `scan_copy` unless NO_COPY), with the total in `work[0]` when
// `scan_copy == NO_COPY`. One workgroup walks the array in tiles of
// SCAN_TILE words; each invocation owns four consecutive words of a tile.
// Integer sums do not depend on the order, so the result is exact.
@compute @workgroup_size(256, 1, 1)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var carry = 0u;
    for (var base = 0u; base < p.scan_len; base += SCAN_TILE) {
        var values = vec4<u32>(0u);
        for (var k = 0u; k < 4u; k += 1u) {
            let index = base + tid * 4u + k;
            if (index < p.scan_len) {
                values[k] = atomicLoad(&work[p.scan_src + index]);
            }
        }
        let sum = values.x + values.y + values.z + values.w;
        partial[tid] = sum;
        workgroupBarrier();
        // Hillis-Steele inclusive scan of the 256 partial sums.
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
        for (var k = 0u; k < 4u; k += 1u) {
            let index = base + tid * 4u + k;
            if (index < p.scan_len) {
                atomicStore(&work[p.scan_dst + index], running);
                if (p.scan_copy != NO_COPY) {
                    atomicStore(&work[p.scan_copy + index], running);
                }
            }
            running += values[k];
        }
        carry += partial[SCAN_THREADS - 1u];
        workgroupBarrier();
    }
    if (tid == 0u && p.scan_copy == NO_COPY) {
        atomicStore(&work[0], carry);
    }
}

@compute @workgroup_size(64, 1, 1)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let id = item(gid, nwg);
    if (id >= p.n) {
        return;
    }
    let slot = atomicAdd(&work[p.off_cursor + bucket_of(refined[id])], 1u);
    atomicStore(&work[p.off_sorted + slot], id);
}

@compute @workgroup_size(64, 1, 1)
fn bucket_sort(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let bucket = item(gid, nwg);
    if (bucket >= p.buckets) {
        return;
    }
    let count = atomicLoad(&work[p.off_count + bucket]);
    if (count < 2u) {
        return;
    }
    let start = p.off_sorted + atomicLoad(&work[p.off_start + bucket]);
    for (var a = 1u; a < count; a += 1u) {
        let moving = atomicLoad(&work[start + a]);
        let col = candidate_col(moving);
        var b = a;
        loop {
            if (b == 0u) {
                break;
            }
            let other = atomicLoad(&work[start + b - 1u]);
            if (candidate_col(other) <= col) {
                break;
            }
            atomicStore(&work[start + b], other);
            b -= 1u;
        }
        atomicStore(&work[start + b], moving);
    }
}

fn target_of(r: Refined) -> u32 {
    let plane = p.rows * p.cols;
    return r.level * plane + u32(r.i) * p.rows + u32(r.j);
}

@compute @workgroup_size(64, 1, 1)
fn hash_insert(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let rank = item(gid, nwg);
    if (rank >= p.n) {
        return;
    }
    let sample = target_of(refined[atomicLoad(&work[p.off_sorted + rank])]);
    let key = sample + 1u;
    var slot = (sample * 2654435761u) & p.hash_mask;
    loop {
        let result = atomicCompareExchangeWeak(&work[p.off_hash_keys + slot], 0u, key);
        if (result.old_value == key || result.exchanged) {
            break;
        }
        if (result.old_value == 0u) {
            continue; // spurious failure; retry the same slot
        }
        slot = (slot + 1u) & p.hash_mask;
    }
    atomicMax(&work[p.off_hash_vals + slot], ~rank);
    atomicStore(&work[p.off_slot + rank], slot);
}

@compute @workgroup_size(64, 1, 1)
fn flag(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let rank = item(gid, nwg);
    if (rank >= p.n) {
        return;
    }
    let record = refined[atomicLoad(&work[p.off_sorted + rank])];
    let slot = atomicLoad(&work[p.off_slot + rank]);
    let first = atomicLoad(&work[p.off_hash_vals + slot]) == ~rank;
    atomicStore(&work[p.off_kept + rank], select(0u, 1u, first && record.accepted != 0u));
}

@compute @workgroup_size(64, 1, 1)
fn emit(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let rank = item(gid, nwg);
    if (rank >= p.n || atomicLoad(&work[p.off_kept + rank]) == 0u) {
        return;
    }
    let record = refined[atomicLoad(&work[p.off_sorted + rank])];
    let q = atomicLoad(&work[p.off_pos + rank]);
    features[q] = Feature(record.x, record.y, record.sigma, record.value, record.level);
    // `inv_delta` is a power of two, so these products equal the host's
    // division by the octave delta bit for bit.
    keys[q] = OriKey(
        record.x * p.inv_delta,
        record.y * p.inv_delta,
        record.sigma * p.inv_delta,
        record.level,
    );
}
