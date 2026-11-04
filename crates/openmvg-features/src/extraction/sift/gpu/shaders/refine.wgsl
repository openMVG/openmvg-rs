// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Five-step 3D Taylor refinement, contrast and edge rejection, and scale-aware
// border rejection. This mirrors `refine_keypoints` in
// `extraction/sift/localization.rs`.

struct RefineParams {
    rows_cols: vec2<u32>,
    levels: u32,
    num_scales: u32,
    delta: f32,
    sigma_min: f32,
    delta_min: f32,
    edge_threshold: f32,
    peak_threshold: f32,
    cap: u32,
    octave: u32,
    _pad0: u32,
    _pad1: vec4<u32>,
};

struct Candidate {
    level: u32,
    i: i32,
    j: i32,
    value: f32,
    octave: u32,
    _pad: u32,
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

struct Taylor {
    dx: f32,
    dy: f32,
    ds: f32,
    val: f32,
    ok: u32,
};

@group(0) @binding(0) var<uniform> p: RefineParams;
@group(0) @binding(1) var<storage, read> dog: array<f32>;
@group(0) @binding(2) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(3) var<storage, read> count: array<u32>;
@group(0) @binding(4) var<storage, read_write> refined: array<Refined>;
@group(0) @binding(5) var<storage, read_write> out_count: atomic<u32>;

fn det3(
    a: f32, b: f32, c: f32,
    d: f32, e: f32, f: f32,
    g: f32, h: f32, i: f32,
) -> f32 {
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

fn dog_at(level: i32, col: i32, row: i32) -> f32 {
    return dog[u32(level) * p.rows_cols.x * p.rows_cols.y
        + u32(col) * p.rows_cols.x + u32(row)];
}

fn taylor(level: i32, col: i32, row: i32) -> Taylor {
    let center = dog_at(level, col, row);
    let h_xx = dog_at(level, col - 1, row) + dog_at(level, col + 1, row) - 2.0 * center;
    let h_yy = dog_at(level, col, row + 1) + dog_at(level, col, row - 1) - 2.0 * center;
    let h_ss = dog_at(level + 1, col, row) + dog_at(level - 1, col, row) - 2.0 * center;
    let h_xy = (dog_at(level, col + 1, row + 1) - dog_at(level, col + 1, row - 1)
        - dog_at(level, col - 1, row + 1) + dog_at(level, col - 1, row - 1)) / 4.0;
    let h_xs = (dog_at(level + 1, col + 1, row) - dog_at(level + 1, col - 1, row)
        - dog_at(level - 1, col + 1, row) + dog_at(level - 1, col - 1, row)) / 4.0;
    let h_ys = (dog_at(level + 1, col, row + 1) - dog_at(level + 1, col, row - 1)
        - dog_at(level - 1, col, row + 1) + dog_at(level - 1, col, row - 1)) / 4.0;

    let g_x = (dog_at(level, col + 1, row) - dog_at(level, col - 1, row)) / 2.0;
    let g_y = (dog_at(level, col, row + 1) - dog_at(level, col, row - 1)) / 2.0;
    let g_s = (dog_at(level + 1, col, row) - dog_at(level - 1, col, row)) / 2.0;

    let det = det3(
        h_xx, h_xy, h_xs,
        h_xy, h_yy, h_ys,
        h_xs, h_ys, h_ss,
    );
    if (abs(det) < 1e-12) {
        return Taylor(0.0, 0.0, 0.0, 0.0, 0u);
    }

    let dx = det3(
        -g_x, h_xy, h_xs,
        -g_y, h_yy, h_ys,
        -g_s, h_ys, h_ss,
    ) / det;
    let dy = det3(
        h_xx, -g_x, h_xs,
        h_xy, -g_y, h_ys,
        h_xs, -g_s, h_ss,
    ) / det;
    let ds = det3(
        h_xx, h_xy, -g_x,
        h_xy, h_yy, -g_y,
        h_xs, h_ys, -g_s,
    ) / det;
    let val = center + (g_x * dx + g_y * dy + g_s * ds) / 2.0;
    return Taylor(dx, dy, ds, val, 1u);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= count[0]) {
        return;
    }
    let candidate = candidates[id];
    let w = i32(p.rows_cols.y);
    let h = i32(p.rows_cols.x);
    let levels = i32(p.levels);

    var ic = candidate.i;
    var jc = candidate.j;
    var sc = i32(candidate.level);
    var dx = 0.0;
    var dy = 0.0;
    var ds = 0.0;
    var val = candidate.value;
    var converged = false;

    for (var iter = 0u; iter < 5u; iter += 1u) {
        if (ic <= 0 || ic >= w - 1 || jc <= 0 || jc >= h - 1) {
            break;
        }
        let t = taylor(sc, ic, jc);
        if (t.ok == 0u) {
            break;
        }
        dx = t.dx;
        dy = t.dy;
        ds = t.ds;
        val = t.val;
        if (abs(dx) < 0.6 && abs(dy) < 0.6 && abs(ds) < 0.6) {
            converged = true;
            break;
        }
        if (dx > 0.6 && ic + 1 < w - 1) {
            ic += 1;
        }
        if (dx < -0.6 && ic - 1 > 0) {
            ic -= 1;
        }
        if (dy > 0.6 && jc + 1 < h - 1) {
            jc += 1;
        }
        if (dy < -0.6 && jc - 1 > 0) {
            jc -= 1;
        }
        if (ds > 0.6 && sc + 1 < levels - 2) {
            sc += 1;
        }
        if (ds < -0.6 && sc > 1) {
            sc -= 1;
        }
    }
    if (!converged) {
        return;
    }

    var accepted = abs(val) > p.peak_threshold;
    if (accepted) {
        let center = dog_at(sc, ic, jc);
        let h_xx = dog_at(sc, ic - 1, jc) + dog_at(sc, ic + 1, jc) - 2.0 * center;
        let h_yy = dog_at(sc, ic, jc + 1) + dog_at(sc, ic, jc - 1) - 2.0 * center;
        let h_xy = (dog_at(sc, ic + 1, jc + 1) - dog_at(sc, ic + 1, jc - 1)
            - dog_at(sc, ic - 1, jc + 1) + dog_at(sc, ic - 1, jc - 1)) / 4.0;
        let det = h_xx * h_yy - h_xy * h_xy;
        let edge_threshold = (p.edge_threshold + 1.0) * (p.edge_threshold + 1.0)
            / p.edge_threshold;
        if (det <= 0.0 || (h_xx + h_yy) * (h_xx + h_yy) / det > edge_threshold) {
            accepted = false;
        }
    }

    let delta_ratio = p.delta / p.delta_min;
    let sigma = delta_ratio * p.sigma_min
        * exp2((f32(sc) + ds) / f32(p.num_scales));
    let x = (f32(ic) + dx) * p.delta;
    let y = (f32(jc) + dy) * p.delta;
    let width = f32(p.rows_cols.y) * p.delta;
    let height = f32(p.rows_cols.x) * p.delta;
    if (!(x - sigma > 0.0 && x + sigma < width
        && y - sigma > 0.0 && y + sigma < height)) {
        accepted = false;
    }

    let slot = atomicAdd(&out_count, 1u);
    if (slot < p.cap) {
        refined[slot] = Refined(
            x, y, sigma, 0.0, val,
            p.octave, u32(sc), ic, jc,
            candidate.level, candidate.i, candidate.j,
            select(0u, 1u, accepted), 0u,
        );
    }
}
