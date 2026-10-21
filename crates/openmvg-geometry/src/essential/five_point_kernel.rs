// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated five-point relative pose kernel behind `FivePointSolver`.
//!
//! Pipeline: Householder null space of the 5x9 epipolar system, the 10x20 cubic constraint matrix,
//! Nistér elimination to a degree-10 polynomial in `z` (with a fixed "hole frame" retry when a root
//! sits near the chart hole `z = inf`), Sturm isolation + safeguarded Newton for its real roots (when
//! the Sturm chain signals a near-multiple root, the roots of a compensated double-double determinant
//! are added), a scalar cheirality vote per root, and a Gauss-Newton pose polish on (R, t) started
//! from a quaternion readout (R is a rotation by construction). Near pure rotation (small parallax) a
//! gated cold path adds the compensated-determinant roots and a Procrustes-start candidate.
//!
//! The arithmetic is bit-exact with the project's Julia reference implementation:
//! `tests/five_point_reference.rs` checks every code path (see [`path`]) against exported reference
//! outputs. One addition: the small-parallax cold path is also entered when the bearings themselves
//! look like a pure rotation (`angles_preserved`), not only on root-side evidence. It only appends
//! candidates, after the reference ones, and changes no output outside near-pure rotation.
//!
//! # Contract
//!
//! * **Input**: `b1[i]`, `b2[i]` are the two observations of point `i` as nonzero, finite 3-vectors.
//!   The norm is irrelevant (vectors are rescaled when their squared norm would over/underflow), but
//!   the **direction (sign) matters**: a point is in front of a camera when its depth along the given
//!   vector is positive, including rays in the rear hemisphere. Other inputs never panic but give
//!   unspecified poses.
//! * **Geometry**: `X2 = R X1 + t` with `X1 ~ b1`, `X2 ~ b2`; the epipolar constraint is
//!   `b2^T E b1 = 0`.
//! * **Output normalization**: `R` is a rotation, `|t| = 1` and `E = [t]_x R / sqrt(2)`, i.e.
//!   `|E|_F = 1`. All matrices are row-major `[f64; 9]`.
//! * **Cheirality rule**: for every real root, the twisted pair is chosen by alignment and the sign of
//!   `t` by a parallax-weighted depth vote (the sign with strictly more points in front wins); the
//!   solution is kept when at least `MIN_FRONT` (3) of the 5 points have positive depth in both
//!   cameras.
//! * **Count**: at most [`MAX_SOLUTIONS`] poses (10 polynomial roots in the generic case, plus
//!   deduplicated small-parallax extras), in a deterministic order.
//! * **Threads / allocation**: no heap allocation and no global state; the workspace lives on the
//!   stack, so concurrent calls are independent.
//!
//! Floating point: the code never relies on FMA contraction (Rust does not contract `a * b + c`); the
//! reference's `muladd` and `fma` sites are `f64::mul_add`, which is correctly rounded with or without
//! hardware FMA. Every other operation is a correctly rounded IEEE 754 operation (`+ - * /`, `sqrt`,
//! `abs`), so results are bit-identical across targets.

// Index loops and literal constants mirror the reference one to one (bit-exactness); the negated
// float comparisons fix the branch a NaN takes.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::comparison_chain
)]

use arrayvec::ArrayVec;

/// Upper bound on the number of returned poses: one per polynomial root (at most 32 root slots; at
/// most 10 Float64 roots plus 10 double-double roots for a monotone Sturm sequence), up to 10
/// compensated-determinant roots of the small-parallax path and one Procrustes candidate. Further
/// candidates, reachable only when a Sturm count breaks down numerically, are dropped.
pub const MAX_SOLUTIONS: usize = 43;
/// Root slots of the main root list and of the double-double root list.
const MAX_ROOTS: usize = 32;
const MAX_ROOTS_DD: usize = 16;

/// Gauss-Newton pose iterations per root (stops early below `POSE_STEP_TOL`).
const POSE_ITERS: u32 = 8;
/// Pose Newton stopping threshold on the L1 step norm.
const POSE_STEP_TOL: f64 = 1e-7;
/// Relative tolerance of the polynomial root polish.
const ROOT_RTOL: f64 = 1e-10;
/// Minimum number of the 5 points that must be in front of both cameras.
const MIN_FRONT: i32 = 3;

/// One relative pose hypothesis. Row-major 3x3 matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Essential matrix `[t]_x R / sqrt(2)` (`|E|_F = 1`).
    pub e: [f64; 9],
    /// Rotation, `X2 = R X1 + t`.
    pub r: [f64; 9],
    /// Unit translation.
    pub t: [f64; 3],
}

/// Fixed-capacity pose buffer (no heap); reuse it across calls.
pub type Poses = ArrayVec<Pose, MAX_SOLUTIONS>;

/// Code-path flags returned by the kernel `solve` (testing and fixture selection).
pub mod path {
    /// The degree-10 polynomial had a root near the chart hole; recomputed in the fixed hole frame.
    pub const FRAME_RETRY: u32 = 1;
    /// A cheirality-rejected root was re-polished by the small-parallax rescue.
    pub const RESCUE: u32 = 2;
    /// The small-parallax cold path ran (Procrustes check).
    pub const SMALL_EXTRA: u32 = 4;
    /// The small-parallax gate passed: compensated-determinant roots + Procrustes candidate.
    pub const SMALL_DD: u32 = 8;
    /// A root needed the extra small-parallax Newton polish.
    pub const GATE_NEWTON: u32 = 16;
    /// The frame retry was triggered by a non-finite coefficient (zero pivot in the elimination).
    pub const NONFINITE_RETRY: u32 = 32;
    /// The Sturm-chain trigger fired; the double-double determinant roots were unioned in.
    pub const DD_UNION: u32 = 64;
    /// The double-double union added at least one root.
    pub const DD_ADDED: u32 = 128;
    /// A pose that did not converge got the `EXTRA_IT` extra Newton steps.
    pub const EXTRA_NEWTON: u32 = 256;
    /// The post-polish sign re-vote flipped `t`.
    pub const RESIGN_FLIP: u32 = 512;
    /// A pose was dropped (non-finite, or not converged and above the residual gate).
    pub const DROPPED: u32 = 1024;
    /// An input vector (or a point pair) needed the over/underflow-safe rescale.
    pub const RESCALE: u32 = 2048;
}

// ====================================================================== scalar helpers

type V3 = [f64; 3];
type V9 = [f64; 9];
/// The 5 unit bearings of one view, one point per row (same memory as the Julia column-major 3x5 P, Q).
type Pts = [V3; 5];

#[inline(always)]
fn cross(a: &V3, b: &V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline(always)]
fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline(always)]
fn unit(a: &V3) -> V3 {
    vscale(a, 1.0 / dot(a, a).sqrt())
}
#[inline(always)]
fn vadd<const M: usize>(a: &[f64; M], b: &[f64; M]) -> [f64; M] {
    std::array::from_fn(|i| a[i] + b[i])
}
#[inline(always)]
fn vsub<const M: usize>(a: &[f64; M], b: &[f64; M]) -> [f64; M] {
    std::array::from_fn(|i| a[i] - b[i])
}
#[inline(always)]
fn vscale<const M: usize>(a: &[f64; M], h: f64) -> [f64; M] {
    std::array::from_fn(|i| h * a[i])
}
/// Row-major 3x3 times vector.
#[inline(always)]
fn mv(m: &V9, p: &V3) -> V3 {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
        m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
        m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
    ]
}
/// Row-major 3x3 product.
#[inline(always)]
fn mm(a: &V9, b: &V9) -> V9 {
    std::array::from_fn(|k| {
        let (i, j) = (k - k % 3, k % 3);
        a[i] * b[j] + a[i + 1] * b[3 + j] + a[i + 2] * b[6 + j]
    })
}
/// Columns of a row-major 3x3.
#[inline(always)]
fn cols(m: &V9) -> [V3; 3] {
    [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]]
}
/// Cofactor matrix (row-major): row i = row (i + 1) x row (i + 2).
#[inline(always)]
fn cof3(x: &V9) -> V9 {
    let r = |i: usize| [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    let [a, b, c] = [
        cross(&r(1), &r(2)),
        cross(&r(2), &r(0)),
        cross(&r(0), &r(1)),
    ];
    [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]
}
/// Julia `muladd` (fused on FMA targets); `mul_add` is correctly rounded on every target.
#[inline(always)]
fn ma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
/// Julia `max` for Float64 (`max_float` = IEEE 754-2019 maximum: NaN-propagating, -0.0 < 0.0), branch-free.
#[inline(always)]
fn jmax(x: f64, y: f64) -> f64 {
    let diff = x - y;
    let am = if diff.is_sign_negative() { y } else { x };
    if x.is_nan() | y.is_nan() { diff } else { am }
}
/// Julia `min` for Float64 (`min_float`), branch-free.
#[inline(always)]
fn jmin(x: f64, y: f64) -> f64 {
    let diff = x - y;
    let am = if diff.is_sign_negative() { x } else { y };
    if x.is_nan() | y.is_nan() { diff } else { am }
}

// ====================================================================== workspace
// Column-major with 1-based index macros like the reference workspace, except C (10x20, one array
// per row, so elimination rows are contiguous).

macro_rules! am {
    ($r:expr, $c:expr) => {
        ($r - 1) + 9 * ($c - 1)
    };
}
macro_rules! bm {
    ($r:expr, $c:expr) => {
        ($r - 1) + 6 * ($c - 1)
    };
}
macro_rules! jm {
    ($r:expr, $c:expr) => {
        ($r - 1) + 5 * ($c - 1)
    };
}

/// Read-only state shared by the per-root post-processing.
struct Ctx {
    n: [f64; 36],
    p: Pts,
    q: Pts,
}

impl Ctx {
    #[inline(always)]
    fn newton(&self, r: &mut V9, t: &mut V3, iters: u32) -> f64 {
        pose_newton(r, t, &self.p, &self.q, iters)
    }
    /// `EXTRA_IT` more Newton steps when the last step norm `st` is not below `POSE_STEP_TOL`; returns
    /// convergence.
    #[inline(always)]
    fn converge(&self, r: &mut V9, t: &mut V3, st: f64, fl: &mut u32) -> bool {
        if st < POSE_STEP_TOL {
            return true;
        }
        *fl |= path::EXTRA_NEWTON;
        self.newton(r, t, EXTRA_IT) < POSE_STEP_TOL
    }
}

// ====================================================================== null space, coefficients, elimination

/// Constraint rows q p^T of the unit bearings (row i: the row-major entries of q_i p_i^T).
#[inline(always)]
fn fill_at(a: &mut [f64; 45], p: &Pts, q: &Pts) {
    for i in 0..5 {
        for r in 0..3 {
            for c in 0..3 {
                a[9 * i + 3 * r + c] = q[i][r] * p[i][c];
            }
        }
    }
}

#[inline(always)]
fn hh_step<const K: usize>(a: &mut [f64; 45], vk: &mut [f64; 5], tau: &mut [f64; 5]) {
    let k = K;
    let mut best = 0.0;
    let mut p = k;
    for j in k..6 {
        let mut s = 0.0;
        for r in k..10 {
            s += a[am!(r, j)] * a[am!(r, j)];
        }
        if s > best {
            best = s;
            p = j;
        }
    }
    if p != k {
        for r in 1..10 {
            a.swap(am!(r, k), am!(r, p));
        }
    }
    let nrm = best.sqrt();
    let x1 = a[am!(k, k)];
    let alpha = if x1 >= 0.0 { -nrm } else { nrm };
    let v1 = x1 - alpha;
    let vv = v1 * v1 + (best - x1 * x1);
    let t = if vv > 0.0 { 2.0 / vv } else { 0.0 };
    vk[k - 1] = v1;
    tau[k - 1] = t;
    for j in k + 1..6 {
        let mut s = v1 * a[am!(k, j)];
        for r in k + 1..10 {
            s += a[am!(r, k)] * a[am!(r, j)];
        }
        s *= t;
        a[am!(k, j)] -= s * v1;
        for r in k + 1..10 {
            a[am!(r, j)] -= s * a[am!(r, k)];
        }
    }
}
#[inline(always)]
fn hh_apply<const K: usize>(n: &mut [f64; 36], a: &[f64; 45], vk: &[f64; 5], tau: &[f64; 5]) {
    let k = K;
    let v1 = vk[k - 1];
    let t = tau[k - 1];
    for j in 1..5 {
        let mut s = v1 * n[am!(k, j)];
        for r in k + 1..10 {
            s += a[am!(r, k)] * n[am!(r, j)];
        }
        s *= t;
        n[am!(k, j)] -= s * v1;
        for r in k + 1..10 {
            n[am!(r, j)] -= s * a[am!(r, k)];
        }
    }
}
/// Null space of the 5x9 system by Householder QR with column pivoting (steps const-generic, see solve5x5).
#[inline(always)]
fn nullspace_hh(n: &mut [f64; 36], a: &mut [f64; 45]) {
    let mut vk = [0.0; 5];
    let mut tau = [0.0; 5];
    hh_step::<1>(a, &mut vk, &mut tau);
    hh_step::<2>(a, &mut vk, &mut tau);
    hh_step::<3>(a, &mut vk, &mut tau);
    hh_step::<4>(a, &mut vk, &mut tau);
    hh_step::<5>(a, &mut vk, &mut tau);
    for j in 1..5usize {
        for r in 1..10usize {
            n[am!(r, j)] = if r == 5 + j { 1.0 } else { 0.0 };
        }
    }
    hh_apply::<5>(n, a, &vk, &tau);
    hh_apply::<4>(n, a, &vk, &tau);
    hh_apply::<3>(n, a, &vk, &tau);
    hh_apply::<2>(n, a, &vk, &tau);
    hh_apply::<1>(n, a, &vk, &tau);
}

#[inline(always)]
fn elim_step<const K: usize>(c: &mut [[f64; 20]; 10]) {
    let k = K;
    let mut p = k;
    let mut best = c[k][k].abs();
    for r in k + 1..10 {
        let a = c[r][k].abs();
        if a > best {
            best = a;
            p = r;
        }
    }
    if p != k {
        let (lo, hi) = c.split_at_mut(p);
        lo[k][k..].swap_with_slice(&mut hi[0][k..]);
    }
    let (top, bot) = c.split_at_mut(k + 1);
    let rowk = &top[k];
    let inv = 1.0 / rowk[k];
    for rr in bot {
        let f = rr[k] * inv;
        for j in k + 1..20 {
            rr[j] -= f * rowk[j];
        }
    }
}
#[inline(always)]
fn elim_back<const I: usize>(b: &mut [f64; 60], c: &[[f64; 20]; 10]) {
    let i = I;
    let row = &c[i - 1];
    let inv = 1.0 / row[i - 1];
    for j in 1..11usize {
        let mut s = row[9 + j];
        for l in i + 1..11 {
            s -= row[l - 1] * b[bm!(l - 4, j)];
        }
        b[bm!(i - 4, j)] = s * inv;
    }
}
#[inline(always)]
fn acc<const ADD: bool>(s: f64, t: f64) -> f64 {
    if ADD { s + t } else { t }
}
/// `q (= | +=) a b`: linear x linear -> quadratic. A linear form is its 4 coefficients on (x, y, z, 1), a
/// quadratic its 10 on x2 xy xz x y2 yz y z2 z 1; every sum runs left to right (the Julia `coeffs!` order).
#[inline(always)]
fn lin_lin<const ADD: bool>(q: &mut [f64; 10], a: &[f64; 4], b: &[f64; 4]) {
    q[0] = acc::<ADD>(q[0], a[0] * b[0]);
    q[1] = acc::<ADD>(q[1], a[0] * b[1]) + a[1] * b[0];
    q[2] = acc::<ADD>(q[2], a[0] * b[2]) + a[2] * b[0];
    q[3] = acc::<ADD>(q[3], a[0] * b[3]) + a[3] * b[0];
    q[4] = acc::<ADD>(q[4], a[1] * b[1]);
    q[5] = acc::<ADD>(q[5], a[1] * b[2]) + a[2] * b[1];
    q[6] = acc::<ADD>(q[6], a[1] * b[3]) + a[3] * b[1];
    q[7] = acc::<ADD>(q[7], a[2] * b[2]);
    q[8] = acc::<ADD>(q[8], a[2] * b[3]) + a[3] * b[2];
    q[9] = acc::<ADD>(q[9], a[3] * b[3]);
}
/// `c (= | +=) q e`: quadratic x linear -> cubic, one row of C in the column order
/// x3 y3 x2y xy2 x2z x2 y2z y2 xyz xy xz2 xz x yz2 yz y z3 z2 z 1.
#[inline(always)]
fn quad_lin<const ADD: bool>(c: &mut [f64; 20], q: &[f64; 10], e: &[f64; 4]) {
    c[0] = acc::<ADD>(c[0], q[0] * e[0]);
    c[1] = acc::<ADD>(c[1], q[4] * e[1]);
    c[2] = acc::<ADD>(c[2], q[0] * e[1]) + q[1] * e[0];
    c[3] = acc::<ADD>(c[3], q[1] * e[1]) + q[4] * e[0];
    c[4] = acc::<ADD>(c[4], q[0] * e[2]) + q[2] * e[0];
    c[5] = acc::<ADD>(c[5], q[0] * e[3]) + q[3] * e[0];
    c[6] = acc::<ADD>(c[6], q[4] * e[2]) + q[5] * e[1];
    c[7] = acc::<ADD>(c[7], q[4] * e[3]) + q[6] * e[1];
    c[8] = acc::<ADD>(c[8], q[1] * e[2]) + q[2] * e[1] + q[5] * e[0];
    c[9] = acc::<ADD>(c[9], q[1] * e[3]) + q[3] * e[1] + q[6] * e[0];
    c[10] = acc::<ADD>(c[10], q[2] * e[2]) + q[7] * e[0];
    c[11] = acc::<ADD>(c[11], q[2] * e[3]) + q[3] * e[2] + q[8] * e[0];
    c[12] = acc::<ADD>(c[12], q[3] * e[3]) + q[9] * e[0];
    c[13] = acc::<ADD>(c[13], q[5] * e[2]) + q[7] * e[1];
    c[14] = acc::<ADD>(c[14], q[5] * e[3]) + q[6] * e[2] + q[8] * e[1];
    c[15] = acc::<ADD>(c[15], q[6] * e[3]) + q[9] * e[1];
    c[16] = acc::<ADD>(c[16], q[7] * e[2]);
    c[17] = acc::<ADD>(c[17], q[7] * e[3]) + q[8] * e[2];
    c[18] = acc::<ADD>(c[18], q[8] * e[3]) + q[9] * e[2];
    c[19] = acc::<ADD>(c[19], q[9] * e[3]);
}
/// The 10x20 C (one array per row) of the cubics det(E) = 0 and 2 E E' E - tr(E E') E = 0 for
/// E = x E1 + y E2 + z E3 + E4 (E_k = column k of the 9x4 null space `n`, column-major).
#[inline(always)]
fn coeffs(n: &[f64; 36]) -> [[f64; 20]; 10] {
    let mut e = [[0.0; 4]; 9]; // E[a][b] = e[3a + b] . (x, y, z, 1)
    for r in 0..9 {
        for k in 0..4 {
            e[r][k] = n[r + 9 * k];
        }
    }
    let mut q = [[[0.0; 10]; 3]; 3]; // E E' - tr(E E') / 2 I (upper triangle, then mirrored)
    for i in 0..3 {
        for j in i..3 {
            lin_lin::<false>(&mut q[i][j], &e[3 * i], &e[3 * j]);
            lin_lin::<true>(&mut q[i][j], &e[3 * i + 1], &e[3 * j + 1]);
            lin_lin::<true>(&mut q[i][j], &e[3 * i + 2], &e[3 * j + 2]);
        }
    }
    for k in 0..10 {
        let tr = 0.5 * (q[0][0][k] + q[1][1][k] + q[2][2][k]);
        q[0][0][k] -= tr;
        q[1][1][k] -= tr;
        q[2][2][k] -= tr;
    }
    for i in 0..3 {
        for j in 0..i {
            q[i][j] = q[j][i];
        }
    }
    let mut c = [[0.0; 20]; 10];
    for i in 0..3 {
        for j in 0..3 {
            let row = &mut c[3 * i + j]; // rows 1-9: (q E)[i, j]
            quad_lin::<false>(row, &q[i][0], &e[j]);
            quad_lin::<true>(row, &q[i][1], &e[3 + j]);
            quad_lin::<true>(row, &q[i][2], &e[6 + j]);
        }
    }
    let mut d = [[0.0; 10]; 3]; // row 10: det(E) along row 3, d[l] = cofactor of E[3, l]
    let mut m = [0.0; 10];
    for l in 0..3 {
        let (l1, l2) = ((l + 1) % 3, (l + 2) % 3);
        lin_lin::<false>(&mut d[l], &e[l1], &e[3 + l2]);
        lin_lin::<false>(&mut m, &e[l2], &e[3 + l1]);
        for k in 0..10 {
            d[l][k] -= m[k];
        }
    }
    let row = &mut c[9];
    quad_lin::<false>(row, &d[0], &e[6]);
    quad_lin::<true>(row, &d[1], &e[7]);
    quad_lin::<true>(row, &d[2], &e[8]);
    c
}
/// Gauss elimination with partial pivoting of the 10x20 C, then back-substitution of the last 6 rows
/// into B (6x10). Steps are const-generic (see solve5x5); every row update is elementwise
/// independent, so vectorization leaves the results unchanged.
#[inline(always)]
fn eliminate(b: &mut [f64; 60], c: &mut [[f64; 20]; 10]) {
    elim_step::<0>(c);
    elim_step::<1>(c);
    elim_step::<2>(c);
    elim_step::<3>(c);
    elim_step::<4>(c);
    elim_step::<5>(c);
    elim_step::<6>(c);
    elim_step::<7>(c);
    elim_step::<8>(c);
    elim_step::<9>(c);
    elim_back::<10>(b, c);
    elim_back::<9>(b, c);
    elim_back::<8>(b, c);
    elim_back::<7>(b, c);
    elim_back::<6>(b, c);
    elim_back::<5>(b, c);
}

// ascending-coefficient polynomials
#[inline(always)]
fn pmul<const M: usize, const K: usize, const R: usize>(a: &[f64; M], b: &[f64; K]) -> [f64; R] {
    const { assert!(R == M + K - 1) };
    let mut r = [0.0; R];
    for n in 1..R + 1 {
        let mut s = 0.0;
        let lo = if n + 1 > K { n + 1 - K } else { 1 };
        let hi = if M < n { M } else { n };
        for i in lo..hi + 1 {
            s += a[i - 1] * b[n - i];
        }
        r[n - 1] = s;
    }
    r
}

#[derive(Clone, Copy)]
struct Row {
    px: [f64; 4],
    py: [f64; 4],
    pc: [f64; 5],
}
/// The Nister rows [x, y, 1] of B as polynomials in z (ascending coefficients).
type Rows = [Row; 3];

#[inline(always)]
fn nister_row(b: &[f64; 60], t: usize) -> Row {
    let h = 2 * t - 1;
    let l = 2 * t;
    let g = |r: usize, c: usize| b[bm!(r, c)];
    Row {
        px: [-g(h, 3), g(l, 3) - g(h, 2), g(l, 2) - g(h, 1), g(l, 1)],
        py: [-g(h, 6), g(l, 6) - g(h, 5), g(l, 5) - g(h, 4), g(l, 4)],
        pc: [
            -g(h, 10),
            g(l, 10) - g(h, 9),
            g(l, 9) - g(h, 8),
            g(l, 8) - g(h, 7),
            g(l, 7),
        ],
    }
}

#[inline(always)]
fn nister_rows(b: &[f64; 60]) -> Rows {
    [nister_row(b, 1), nister_row(b, 2), nister_row(b, 3)]
}

#[inline(always)]
fn det_poly(r: &Rows) -> [f64; 11] {
    let [r1, r2, r3] = r;
    let m1: [f64; 8] = vsub(&pmul(&r2.py, &r3.pc), &pmul(&r3.py, &r2.pc));
    let m2: [f64; 8] = vsub(&pmul(&r2.px, &r3.pc), &pmul(&r3.px, &r2.pc));
    let m3: [f64; 7] = vsub(&pmul(&r2.px, &r3.py), &pmul(&r3.px, &r2.py));
    let a: [f64; 11] = pmul(&r1.px, &m1);
    let b: [f64; 11] = pmul(&r1.pc, &m3);
    let c: [f64; 11] = pmul(&r1.py, &m2);
    vsub(&vadd(&a, &b), &c)
}

/// Coefficients, elimination, Nister rows and their degree-10 determinant for the null space `n`.
#[inline(always)]
fn reduce(n: &[f64; 36]) -> (Rows, [f64; 11]) {
    let mut b = [0.0; 60];
    let mut c = coeffs(n);
    eliminate(&mut b, &mut c);
    let rows = nister_rows(&b);
    let dp = det_poly(&rows);
    (rows, dp)
}

// ---------------------------------------------------------------------- compensated det polynomial
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// a*b - c*d (polynomial products), each coefficient as an unnormalized (hi, lo) pair (FMA two-product,
/// `f64::mul_add` = Julia `fma`, exact on every target).
#[inline(always)]
fn dd_pmul_sub<const M: usize, const K: usize, const R: usize>(
    a: &[f64; M],
    b: &[f64; K],
    c: &[f64; M],
    d: &[f64; K],
) -> [(f64, f64); R] {
    const { assert!(R == M + K - 1) };
    let mut r = [(0.0, 0.0); R];
    for n in 1..R + 1 {
        let (mut s, mut e) = (0.0f64, 0.0f64);
        let lo = if n + 1 > K { n + 1 - K } else { 1 };
        let hi = if M < n { M } else { n };
        for i in lo..hi + 1 {
            let (x, y) = (a[i - 1], b[n - i]);
            let p = x * y;
            let ep = x.mul_add(y, -p);
            let es;
            (s, es) = two_sum(s, p);
            e += ep + es;
            let (x, y) = (-c[i - 1], d[n - i]);
            let p = x * y;
            let ep = x.mul_add(y, -p);
            let es;
            (s, es) = two_sum(s, p);
            e += ep + es;
        }
        r[n - 1] = (s, e);
    }
    r
}

/// Accumulates sg * a (f64 poly) * m (dd poly) into coefficient n (1-based).
#[inline(always)]
fn dd_acc<const M: usize, const K: usize>(
    mut s: f64,
    mut e: f64,
    a: &[f64; M],
    m: &[(f64, f64); K],
    n: usize,
    sg: f64,
) -> (f64, f64) {
    let lo = if n + 1 > K { n + 1 - K } else { 1 };
    let hi = if M < n { M } else { n };
    for i in lo..hi + 1 {
        let x = sg * a[i - 1];
        let (h, l) = m[n - i];
        let p = x * h;
        let ep = x.mul_add(h, -p);
        let es;
        (s, es) = two_sum(s, p);
        e += ep + es + x * l;
    }
    (s, e)
}

/// `det_poly` with compensated sums; coefficients are rounded back to f64.
/// Root isolation then uses the same f64 Sturm arithmetic as the main polynomial.
#[inline(never)]
#[cold]
fn det_poly_dd(r: &Rows) -> [f64; 11] {
    let [r1, r2, r3] = r;
    let m1: [(f64, f64); 8] = dd_pmul_sub(&r2.py, &r3.pc, &r3.py, &r2.pc);
    let m2: [(f64, f64); 8] = dd_pmul_sub(&r2.px, &r3.pc, &r3.px, &r2.pc);
    let m3: [(f64, f64); 7] = dd_pmul_sub(&r2.px, &r3.py, &r3.px, &r2.py);
    let mut out = [0.0; 11];
    for n in 1..12usize {
        let (s, e) = dd_acc(0.0, 0.0, &r1.px, &m1, n, 1.0);
        let (s, e) = dd_acc(s, e, &r1.pc, &m3, n, 1.0);
        let (s, e) = dd_acc(s, e, &r1.py, &m2, n, -1.0);
        out[n - 1] = s + e;
    }
    out
}

/// Trigger: min_i |lead_i| < 1e-6 or prod_i |lead_i| < 10^-11.5 over the 9 Sturm remainder leads.
const DD_MIN: f64 = 1e-6;
const DD_PROD: f64 = 3.1622776601683794e-12; // Julia 10.0^-11.5

#[inline(always)]
fn sturm_fire(sv: &[f64; 31]) -> bool {
    let (mut m, mut p) = (f64::INFINITY, 1.0f64);
    for i in 1..10usize {
        let a = sv[3 * i].abs();
        m = jmin(m, a);
        p *= a;
    }
    (m < DD_MIN) | (p < DD_PROD)
}

/// Real roots of the compensated det polynomial, unioned into `roots[..nr]` (dedup 1e-7 relative).
#[inline(never)]
#[cold]
fn dd_union(roots: &mut [f64; MAX_ROOTS], mut nr: usize, rows: &Rows, fl: &mut u32) -> usize {
    let mut roots2 = [0.0; MAX_ROOTS_DD];
    let (n2, _) = real_roots10(&mut roots2, &det_poly_dd(rows));
    for &z in &roots2[..n2] {
        let new = !roots[..nr]
            .iter()
            .any(|&w| (z - w).abs() <= 1e-7 * jmax(z.abs(), 1.0));
        if new && nr < MAX_ROOTS {
            roots[nr] = z;
            nr += 1;
            *fl |= path::DD_ADDED;
        }
    }
    nr
}

#[inline(always)]
fn hval<const M: usize>(p: &[f64; M], z: f64) -> f64 {
    let mut s = p[M - 1];
    for i in (0..M - 1).rev() {
        s = ma(s, z, p[i]);
    }
    s
}

// ---------------------------------------------------------------------- Sturm (1-based arrays)
const ND: usize = 10;

/// One pseudo-division step of the Sturm chain; the rotating row offsets (O1, O2, O3) and the step I are
/// compile-time constants, so the loops unroll without bounds checks (same arithmetic as the loop form).
#[inline(always)]
fn sturm_step<const I: usize, const O1: usize, const O2: usize, const O3: usize>(
    svec: &mut [f64; 31],
    f: &mut [f64; 31],
) {
    let (i, o1, o2, o3) = (I, O1, O2, O3);
    let q1 = f[o1 + ND - i] * f[o2 + ND - 1 - i];
    let q0 = f[o1 + ND - 1 - i] * f[o2 + ND - 1 - i] - f[o1 + ND - i] * f[o2 + ND - 2 - i];
    f[o3] = f[o1] - q0 * f[o2];
    for j in 1..ND - 1 - i {
        f[o3 + j] = f[o1 + j] - q1 * f[o2 + j - 1] - q0 * f[o2 + j];
    }
    let c = -f[o3 + ND - 2 - i].abs();
    let ci = 1.0 / c;
    for j in 0..ND - 1 - i {
        f[o3 + j] *= ci;
    }
    svec[3 * i + 1] = q0;
    svec[3 * i + 2] = q1;
    svec[3 * i + 3] = c;
}

#[inline(always)]
fn build_sturm_seq(svec: &mut [f64; 31], f: &mut [f64; 31], fvec: &[f64; 22]) {
    f[1..=21].copy_from_slice(&fvec[1..=21]);
    // rows start at 1, 1 + ND + 1, 1 + ND + 1 + ND and rotate (o1, o2, o3) <- (o2, o3, o1) after each step
    const A: usize = 1;
    const B: usize = 1 + ND + 1;
    const C: usize = 1 + ND + 1 + ND;
    sturm_step::<0, A, B, C>(svec, f);
    sturm_step::<1, B, C, A>(svec, f);
    sturm_step::<2, C, A, B>(svec, f);
    sturm_step::<3, A, B, C>(svec, f);
    sturm_step::<4, B, C, A>(svec, f);
    sturm_step::<5, C, A, B>(svec, f);
    sturm_step::<6, A, B, C>(svec, f);
    sturm_step::<7, B, C, A>(svec, f);
    sturm_step::<8, C, A, B>(svec, f);
    // ND - 1 = 9 steps: the offsets are back to (A, B, C)
    svec[3 * ND - 2] = f[A];
    svec[3 * ND - 1] = f[A + 1];
    svec[3 * ND] = f[B];
}

#[inline(always)]
fn signchanges(svec: &[f64; 31], x: f64) -> i32 {
    let mut fi2 = svec[3 * ND];
    let mut fi1 = svec[3 * ND - 2] + x * svec[3 * ND - 1];
    let mut cnt = ((fi1 < 0.0) != (fi2 < 0.0)) as i32;
    for i in (0..ND - 1).rev() {
        let fi = (svec[3 * i + 1] + x * svec[3 * i + 2]) * fi1 + svec[3 * i + 3] * fi2;
        cnt += ((fi < 0.0) != (fi1 < 0.0)) as i32;
        fi2 = fi1;
        fi1 = fi;
    }
    cnt
}

// ---------------------------------------------------------------------- real roots
#[inline(always)]
fn pd_monic(a: &[f64; 10], x: f64) -> (f64, f64) {
    let mut p = x + a[9];
    let mut d = 1.0;
    for i in (0..9).rev() {
        d = ma(d, x, p);
        p = ma(p, x, a[i]);
    }
    (p, d)
}
#[inline(always)]
fn p_monic(a: &[f64; 10], x: f64) -> f64 {
    let mut p = x + a[9];
    for i in (0..9).rev() {
        p = ma(p, x, a[i]);
    }
    p
}
#[inline(always)]
fn split_point(a: f64, b: f64) -> f64 {
    if a < 0.0 && 0.0 < b {
        return 0.5 * (a + b);
    }
    if a >= 0.0 {
        let lo = jmax(a, 1.0);
        return if b > 8.0 * lo {
            (lo * b).sqrt()
        } else {
            0.5 * (a + b)
        };
    }
    let lo = jmax(-b, 1.0);
    if -a > 8.0 * lo {
        -(lo * -a).sqrt()
    } else {
        0.5 * (a + b)
    }
}

fn rtsafe(a: &[f64; 10], mut lo: f64, mut hi: f64, neglo: bool) -> f64 {
    if !neglo {
        std::mem::swap(&mut lo, &mut hi);
    }
    let mut x = split_point(jmin(lo, hi), jmax(lo, hi));
    let mut dxold = (hi - lo).abs();
    let mut dx = dxold;
    let (mut f, mut d) = pd_monic(a, x);
    if f == 0.0 {
        return x;
    }
    if f < 0.0 {
        lo = x;
    } else {
        hi = x;
    }
    for _ in 0..100 {
        if ((x - hi) * d - f) * ((x - lo) * d - f) > 0.0 || (2.0 * f).abs() > (dxold * d).abs() {
            dxold = dx;
            let xn = split_point(jmin(lo, hi), jmax(lo, hi));
            dx = (xn - x).abs();
            x = xn;
        } else {
            dxold = dx;
            let st = f / d;
            dx = st.abs();
            x -= st;
        }
        if dx <= ROOT_RTOL * x.abs() + 1e-300 {
            break;
        }
        let fd = pd_monic(a, x);
        f = fd.0;
        d = fd.1;
        if f == 0.0 {
            break;
        }
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
    }
    x
}

/// Real roots of c[0] + ... + c[10] z^10 into `roots` (CAP slots, like the reference root vector; none
/// for a zero leading or a non-finite coefficient); also returns the `sturm_fire` trigger of the Sturm
/// chain it built. Kept out of line: inlining it into `solve` increased the instruction count.
#[inline(never)]
fn real_roots10<const CAP: usize>(roots: &mut [f64; CAP], c: &[f64; 11]) -> (usize, bool) {
    if c[10] == 0.0 || !c.iter().all(|x| x.is_finite()) {
        return (0, false);
    }
    let mut fvec = [0.0; 22];
    let mut svec = [0.0; 31];
    let mut fw = [0.0; 31];
    let ci = 1.0 / c[10];
    let mut a = [0.0; 10];
    for i in 0..10 {
        a[i] = c[i] * ci;
    }
    fvec[1..=10].copy_from_slice(&a);
    fvec[11] = 1.0;
    for i in 0..9usize {
        fvec[12 + i] = a[i + 1] * ((i + 1) as f64 / 10.0);
    }
    fvec[21] = 1.0;
    build_sturm_seq(&mut svec, &mut fw, &fvec);
    let fire = sturm_fire(&svec);
    let mut r0 = 0.0;
    for &ai in a.iter() {
        r0 = jmax(r0, ai.abs());
    }
    r0 += 1.0;
    let sv = &svec;
    let s_lo = signchanges(sv, -r0);
    let s_hi = signchanges(sv, r0);
    if s_lo == s_hi {
        return (0, fire);
    }
    let (mut sa, mut sb, mut na, mut nb) = ([0.0f64; 65], [0.0f64; 65], [0i32; 65], [0i32; 65]);
    let (mut nr, mut top) = (0usize, 1usize);
    (sa[1], sb[1], na[1], nb[1]) = (-r0, r0, s_lo, s_hi);
    while top > 0 {
        let (lo, hi, nlo, nhi) = (sa[top], sb[top], na[top], nb[top]);
        top -= 1;
        let n = nlo - nhi;
        if n < 1 {
            continue;
        }
        // one root: polish a sign-changing bracket, else bisect; several: split. A tiny interval is a root.
        let (flo, fhi) = if n == 1 {
            (p_monic(&a, lo), p_monic(&a, hi))
        } else {
            (0.0, 0.0)
        };
        if (flo < 0.0) != (fhi < 0.0) {
            roots[nr] = rtsafe(&a, lo, hi, flo < 0.0);
            nr += 1;
        } else if hi - lo < 1e-12 * jmax(lo.abs(), 1.0) {
            roots[nr] = 0.5 * (lo + hi);
            nr += 1;
        } else {
            let m = if n == 1 {
                0.5 * (lo + hi)
            } else {
                split_point(lo, hi)
            };
            let nm = signchanges(sv, m);
            (sa[top + 1], sb[top + 1], na[top + 1], nb[top + 1]) = (lo, m, nlo, nm);
            (sa[top + 2], sb[top + 2], na[top + 2], nb[top + 2]) = (m, hi, nm, nhi);
            top += 2;
        }
        // nr == CAP is only reachable with a non-monotone Sturm count (the reference would overflow)
        if nr == CAP || top >= 62 {
            break;
        }
    }
    (nr, fire)
}

// ====================================================================== pose polish

// The pivot steps are const-generic so every loop bound is a compile-time constant (full unrolling,
// no bounds checks); the arithmetic and its order are those of the plain loops.
#[inline(always)]
fn solve5x5_step<const K: usize>(j: &mut [f64; 30]) {
    let k = K;
    let mut p = k;
    let mut best = j[jm!(k, k)].abs();
    for r in k + 1..6 {
        let a = j[jm!(r, k)].abs();
        if a > best {
            best = a;
            p = r;
        }
    }
    if p != k {
        for c in k..7 {
            j.swap(jm!(k, c), jm!(p, c));
        }
    }
    let inv = 1.0 / j[jm!(k, k)];
    for r in k + 1..6 {
        let f = j[jm!(r, k)] * inv;
        for c in k + 1..7 {
            j[jm!(r, c)] -= f * j[jm!(k, c)];
        }
    }
}
#[inline(always)]
fn solve5x5_back<const K: usize>(j: &mut [f64; 30]) {
    let k = K;
    let mut s = j[jm!(k, 6)];
    for c in k + 1..6 {
        s -= j[jm!(k, c)] * j[jm!(c, 6)];
    }
    j[jm!(k, 6)] = s / j[jm!(k, k)];
}
#[inline(always)]
fn solve5x5(j: &mut [f64; 30]) {
    solve5x5_step::<1>(j);
    solve5x5_step::<2>(j);
    solve5x5_step::<3>(j);
    solve5x5_step::<4>(j);
    solve5x5_step::<5>(j);
    solve5x5_back::<5>(j);
    solve5x5_back::<4>(j);
    solve5x5_back::<3>(j);
    solve5x5_back::<2>(j);
    solve5x5_back::<1>(j);
}

/// Gauss-Newton on (R, t) over the 5 epipolar residuals; Cayley rotation update, t on the sphere.
fn pose_newton(r: &mut V9, t: &mut V3, p: &Pts, q: &Pts, iters: u32) -> f64 {
    let mut stepn = f64::INFINITY;
    let mut j = [0.0; 30];
    for _ in 0..iters {
        let [t1, t2, t3] = *t;
        let u = unit(&if t1.abs() <= t2.abs() && t1.abs() <= t3.abs() {
            [0.0, t3, -t2]
        } else if t2.abs() <= t3.abs() {
            [-t3, 0.0, t1]
        } else {
            [t2, -t1, 0.0]
        });
        let v = cross(t, &u);
        for i in 0..5 {
            let y = mv(r, &p[i]);
            let m = cross(&y, &q[i]);
            let (yt, yq) = (dot(&y, t), dot(&y, &q[i]));
            for c in 0..3 {
                j[jm!(i + 1, c + 1)] = q[i][c] * yt - t[c] * yq;
            }
            j[jm!(i + 1, 4)] = dot(&u, &m);
            j[jm!(i + 1, 5)] = dot(&v, &m);
            j[jm!(i + 1, 6)] = dot(t, &m);
        }
        solve5x5(&mut j);
        let (w1, w2, w3) = (-j[jm!(1, 6)], -j[jm!(2, 6)], -j[jm!(3, 6)]);
        let (da, db) = (-j[jm!(4, 6)], -j[jm!(5, 6)]);
        stepn = w1.abs() + w2.abs() + w3.abs() + da.abs() + db.abs();
        if !stepn.is_finite() {
            break;
        }
        let (h1, h2, h3) = (0.5 * w1, 0.5 * w2, 0.5 * w3);
        let s = 2.0 / (1.0 + h1 * h1 + h2 * h2 + h3 * h3);
        let cayley = [
            1.0 - s * (h2 * h2 + h3 * h3),
            s * (h1 * h2 - h3),
            s * (h1 * h3 + h2),
            s * (h1 * h2 + h3),
            1.0 - s * (h1 * h1 + h3 * h3),
            s * (h2 * h3 - h1),
            s * (h1 * h3 - h2),
            s * (h2 * h3 + h1),
            1.0 - s * (h1 * h1 + h2 * h2),
        ];
        *r = mm(&cayley, r);
        *t = unit(&[
            t1 + da * u[0] + db * v[0],
            t2 + da * u[1] + db * v[1],
            t3 + da * u[2] + db * v[2],
        ]);
        if stepn < POSE_STEP_TOL {
            break;
        }
        *r = rot_start(r); // re-project after every non-converged step (polar if far, then quaternion)
    }
    stepn
}

/// Re-projection that also handles R far from orthogonal: max |R^T R - I| < 0.2 -> one Newton-Schulz step
/// R (3I - R^T R) / 2, else the scaled polar Newton (`polar3`; also taken for NaN).
#[inline(always)]
fn orth_fix(r: &V9) -> V9 {
    let [c1, c2, c3] = cols(r);
    let (s11, s22, s33) = (dot(&c1, &c1), dot(&c2, &c2), dot(&c3, &c3));
    let (s12, s13, s23) = (dot(&c1, &c2), dot(&c1, &c3), dot(&c2, &c3));
    // Julia max(a, b, c, d, e, f) folds left to right
    let d = [
        (s22 - 1.0).abs(),
        (s33 - 1.0).abs(),
        s12.abs(),
        s13.abs(),
        s23.abs(),
    ]
    .iter()
    .fold((s11 - 1.0).abs(), |m, &x| jmax(m, x));
    if !(d < 0.2) {
        return polar3(r);
    }
    let (a12, a13, a23) = (-0.5 * s12, -0.5 * s13, -0.5 * s23);
    mm(
        r,
        &[
            1.5 - 0.5 * s11,
            a12,
            a13,
            a12,
            1.5 - 0.5 * s22,
            a23,
            a13,
            a23,
            1.5 - 0.5 * s33,
        ],
    )
}

/// Structural rotation readout. Quaternion by Shepperd's rule (largest of w, x, y, z from the
/// diagonal), then R(q / |q|): orthogonal with det +1 to rounding for any input.
#[inline(always)]
fn rot_quat(r: &V9) -> V9 {
    let [r11, r12, r13, r21, r22, r23, r31, r32, r33] = *r;
    let tr = r11 + r22 + r33;
    let (w, x, y, z) = if tr >= r11 && tr >= r22 && tr >= r33 {
        (1.0 + tr, r32 - r23, r13 - r31, r21 - r12)
    } else if r11 >= r22 && r11 >= r33 {
        (r32 - r23, 1.0 + r11 - r22 - r33, r12 + r21, r13 + r31)
    } else if r22 >= r33 {
        (r13 - r31, r12 + r21, 1.0 - r11 + r22 - r33, r23 + r32)
    } else {
        (r21 - r12, r13 + r31, r23 + r32, 1.0 - r11 - r22 + r33)
    };
    let s = 2.0 / (w * w + x * x + y * y + z * z);
    [
        1.0 - s * (y * y + z * z),
        s * (x * y - w * z),
        s * (x * z + w * y),
        s * (x * y + w * z),
        1.0 - s * (x * x + z * z),
        s * (y * z - w * x),
        s * (x * z - w * y),
        s * (y * z + w * x),
        1.0 - s * (x * x + y * y),
    ]
}

/// Rotation start for the pose Newton from any 3x3: `orth_fix`, then the quaternion readout.
#[inline(always)]
fn rot_start(r: &V9) -> V9 {
    rot_quat(&orth_fix(r))
}

// ====================================================================== small-parallax gate
const PAR_TAU2: f64 = 1e-4;

/// True when every point has |R p x q|^2 < PAR_TAU2 (near pure rotation; NaN -> not small).
#[inline(always)]
fn small_par(r: &V9, p: &Pts, q: &Pts) -> bool {
    (0..5).all(|i| {
        let m = cross(&mv(r, &p[i]), &q[i]);
        dot(&m, &m) < PAR_TAU2
    })
}

/// Bound on `||p_i.p_j| - |q_i.q_j||` implied by [`small_par`] for some rotation: a rotation
/// preserves the angles between bearings and each `q_i` is within `asin(0.01)` of `R p_i`, so the
/// difference is at most `2 asin(0.01) + asin(0.01)^2 < 0.0202`.
const ANGLE_TOL: f64 = 0.021;

/// Cheap necessary condition for [`small_par`] under the Procrustes rotation, read from the data
/// rather than from the found roots: the elimination can turn the true root of a near-pure rotation
/// into a complex pair while spurious roots still pass cheirality, leaving no root-side evidence
/// (NaN -> false).
#[inline(always)]
fn angles_preserved(p: &Pts, q: &Pts) -> bool {
    (0..5).all(|i| {
        (i + 1..5).all(|j| (dot(&p[i], &p[j]).abs() - dot(&q[i], &q[j]).abs()).abs() <= ANGLE_TOL)
    })
}

/// Scaled Newton polar iteration X <- (g X + X^-T / g) / 2 with the Frobenius scaling
/// g = sqrt(|X^-1|_F / |X|_F); the nearest rotation when det X > 0.
fn polar3(h: &V9) -> V9 {
    let mut x = *h;
    for _ in 1..31 {
        let c = cof3(&x);
        let d = x[0] * c[0] + x[1] * c[1] + x[2] * c[2];
        if d == 0.0 {
            break;
        }
        let (mut nx, mut nc) = (0.0, 0.0);
        for i in 0..9 {
            nx += x[i] * x[i];
            nc += c[i] * c[i];
        }
        let g = ((nc / nx).sqrt() / d.abs()).sqrt(); // |X^-1|_F = |cof X|_F / |det X|
        let (a, b) = (0.5 * g, 0.5 / (g * d));
        let xn: V9 = std::array::from_fn(|i| a * x[i] + b * c[i]);
        let dn = (0..9).fold(0.0, |s, i| s + (xn[i] - x[i]).abs());
        x = xn;
        if dn < 1e-15 {
            break;
        }
    }
    x
}

/// Pure-rotation start: the polar factor of sum_i q_i p_i^T.
fn procrustes(p: &Pts, q: &Pts) -> V9 {
    polar3(&std::array::from_fn(|k| {
        (0..5).fold(0.0, |s, i| s + q[i][k / 3] * p[i][k % 3])
    }))
}

/// Translation for a fixed R: the null direction of S = sum_i m_i m_i^T (m_i = R p_i x q_i), by power
/// iteration on adj(S) from its largest row.
fn t_ls(r: &V9, p: &Pts, q: &Pts) -> V3 {
    let (mut s11, mut s12, mut s13, mut s22, mut s23, mut s33) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..5 {
        let [m1, m2, m3] = cross(&mv(r, &p[i]), &q[i]);
        s11 += m1 * m1;
        s12 += m1 * m2;
        s13 += m1 * m3;
        s22 += m2 * m2;
        s23 += m2 * m3;
        s33 += m3 * m3;
    }
    let a11 = s22 * s33 - s23 * s23;
    let a12 = s13 * s23 - s12 * s33;
    let a13 = s12 * s23 - s13 * s22;
    let a22 = s11 * s33 - s13 * s13;
    let a23 = s12 * s13 - s11 * s23;
    let a33 = s11 * s22 - s12 * s12;
    let a = [a11, a12, a13, a12, a22, a23, a13, a23, a33];
    let [r1, r2, r3] = cols(&a); // = rows (symmetric)
    let (n1, n2, n3) = (dot(&r1, &r1), dot(&r2, &r2), dot(&r3, &r3));
    let mut x = if n1 >= n2 && n1 >= n3 {
        r1
    } else if n2 >= n3 {
        r2
    } else {
        r3
    };
    for _ in 0..3 {
        x = unit(&mv(&a, &x));
    }
    x
}

// ====================================================================== per-root pose

/// Fixed generic orthogonal 4x4 "hole frame" (column-major): moves the z = inf chart hole off
/// structured positions.
const G: [f64; 16] = [
    -0.5338327499918851,
    -0.747589708681249,
    -0.31964841125092575,
    -0.23228671011219285,
    0.34526269891610195,
    0.10396513179150346,
    -0.15087986016207586,
    -0.920445646361705,
    0.12643656666007302,
    0.238920159894008,
    -0.9354464358548387,
    0.22775187694926788,
    0.7614657302758171,
    -0.6109157356310954,
    -0.0003558492816089498,
    0.21668359177640306,
];

#[inline(always)]
fn frame(n: &mut [f64; 36]) {
    for i in 1..10usize {
        let (a, b, c, d) = (n[am!(i, 1)], n[am!(i, 2)], n[am!(i, 3)], n[am!(i, 4)]);
        n[am!(i, 1)] = a * G[0] + b * G[1] + c * G[2] + d * G[3];
        n[am!(i, 2)] = a * G[4] + b * G[5] + c * G[6] + d * G[7];
        n[am!(i, 3)] = a * G[8] + b * G[9] + c * G[10] + d * G[11];
        n[am!(i, 4)] = a * G[12] + b * G[13] + c * G[14] + d * G[15];
    }
}

#[inline(always)]
fn backsub(r: &Rows, z: f64) -> (f64, f64) {
    let ev = |w: &Row| [hval(&w.px, z), hval(&w.py, z), hval(&w.pc, z)];
    let (a, b, d) = (ev(&r[0]), ev(&r[1]), ev(&r[2]));
    let mut w = cross(&a, &b);
    let w2 = cross(&a, &d);
    if w2[2].abs() > w[2].abs() {
        w = w2;
    }
    let w3 = cross(&b, &d);
    if w3[2].abs() > w[2].abs() {
        w = w3;
    }
    (w[0] / w[2], w[1] / w[2])
}

#[inline(always)]
fn e_from(n: &[f64; 36], x: f64, y: f64, z: f64) -> V9 {
    let s = (2.0 / (x * x + y * y + z * z + 1.0)).sqrt();
    let mut e = [0.0; 9];
    for i in 1..10usize {
        e[i - 1] = s * (x * n[am!(i, 1)] + y * n[am!(i, 2)] + z * n[am!(i, 3)] + n[am!(i, 4)]);
    }
    e
}

/// (t, cof(E), [t]x E) of an essential matrix with |E|_F^2 = 2: R = cof(E) -+ [t]x E.
#[inline(always)]
fn pose_parts(e: &V9) -> (V3, V9, V9) {
    let [c1, c2, c3] = cols(e);
    let (mut u, v, w) = (cross(&c1, &c2), cross(&c1, &c3), cross(&c2, &c3));
    let (mut nu, nv, nw) = (dot(&u, &u), dot(&v, &v), dot(&w, &w));
    if nv > nu {
        u = v;
        nu = nv;
    }
    if nw > nu {
        u = w;
        nu = nw;
    }
    let t = vscale(&u, 1.0 / nu.sqrt());
    (t, cof3(e), ess(e, &t))
}

/// (t, cof(E), [t]x E) of the root z.
#[inline(always)]
fn root_parts(cx: &Ctx, rows: &Rows, z: f64) -> (V3, V9, V9) {
    let (x, y) = backsub(rows, z);
    pose_parts(&e_from(&cx.n, x, y, z))
}

/// Scalar cheirality of (t, cof(E), [t]x E): the twisted pair R0 = cof(E) -+ [t]x E by alignment argmax, the t
/// sign by the parallax-weighted depth vote (the other sign if it has strictly more points in front). Returns
/// (R0 before the rotation re-projection, signed t, #points in front).
#[inline(always)]
fn cheir_pose(t: &V3, cf: &V9, k: &V9, p: &Pts, q: &Pts) -> (V9, V3, i32) {
    let ra = vsub(cf, k);
    let (mut ca, mut ty, mut tq) = ([0.0; 5], [0.0; 5], [0.0; 5]);
    for i in 0..5 {
        let y = mv(&ra, &p[i]);
        ca[i] = dot(&y, &q[i]);
        ty[i] = dot(t, &y);
        tq[i] = dot(t, &q[i]);
    }
    let (mut sa, mut sb, mut s_a, mut s_b) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..5 {
        let cb = 2.0 * ty[i] * tq[i] - ca[i];
        sa += ca[i];
        sb += cb;
        s_a += ca[i] * tq[i] - ty[i];
        s_b += cb * tq[i] - ty[i];
    }
    let tw = sb > sa;
    let sg = if (if tw { s_b } else { s_a }) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let (mut np, mut nn) = (0i32, 0i32);
    for i in 0..5 {
        let yq = if tw {
            2.0 * ty[i] * tq[i] - ca[i]
        } else {
            ca[i]
        };
        let l1 = sg * (yq * tq[i] - ty[i]);
        let l2 = sg * (tq[i] - yq * ty[i]);
        np += ((l1 > 0.0) & (l2 > 0.0)) as i32;
        nn += ((l1 < 0.0) & (l2 < 0.0)) as i32;
    }
    let (sg, nf) = if nn > np { (-sg, nn) } else { (sg, np) };
    (if tw { vadd(cf, k) } else { ra }, vscale(t, sg), nf)
}

#[inline(always)]
fn ess(r: &V9, t: &V3) -> V9 {
    let (t1, t2, t3) = (t[0], t[1], t[2]);
    [
        t2 * r[6] - t3 * r[3],
        t2 * r[7] - t3 * r[4],
        t2 * r[8] - t3 * r[5],
        t3 * r[0] - t1 * r[6],
        t3 * r[1] - t1 * r[7],
        t3 * r[2] - t1 * r[8],
        t1 * r[3] - t2 * r[0],
        t1 * r[4] - t2 * r[1],
        t1 * r[5] - t2 * r[2],
    ]
}

/// Small parallax: re-vote the t sign on the polished (R, t), flipping t if strictly more points are in front
/// with -t.
#[inline(never)]
fn resign(r: &V9, t: &mut V3, p: &Pts, q: &Pts, fl: &mut u32) {
    let (mut np, mut nn) = (0i32, 0i32);
    for i in 0..5 {
        let y = mv(r, &p[i]);
        let (yq, ty, tq) = (dot(&y, &q[i]), dot(t, &y), dot(t, &q[i]));
        let l1 = yq * tq - ty;
        let l2 = tq - yq * ty;
        np += ((l1 > 0.0) & (l2 > 0.0)) as i32;
        nn += ((l1 < 0.0) & (l2 < 0.0)) as i32;
    }
    if nn > np {
        *t = [-t[0], -t[1], -t[2]];
        *fl |= path::RESIGN_FLIP;
    }
}

/// Residual gate for poses whose Newton did not converge (max |t . (R p x q)| on unit p, q, |t| = 1).
const RES_TOL: f64 = 1e-8;
/// Newton steps granted to a non-converged pose before the residual gate.
const EXTRA_IT: u32 = 16;

#[inline(never)]
fn resid5(r: &V9, t: &V3, p: &Pts, q: &Pts) -> f64 {
    (0..5).fold(0.0, |res, i| {
        jmax(res, dot(t, &cross(&mv(r, &p[i]), &q[i])).abs())
    })
}

/// Appends (E, R, t); drops non-finite poses, and non-converged ones above the residual gate.
#[inline(always)]
fn push_pose(out: &mut Poses, cx: &Ctx, r: &V9, t: &V3, conv: bool, fl: &mut u32) {
    let sum = r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7] + r[8] + t[0] + t[1] + t[2];
    if !sum.is_finite() || !(conv || resid5(r, t, &cx.p, &cx.q) <= RES_TOL) {
        *fl |= path::DROPPED;
        return;
    }
    // A full buffer drops the pose (see MAX_SOLUTIONS).
    let _ = out.try_push(Pose {
        e: vscale(&ess(r, t), 0.5f64.sqrt()),
        r: *r,
        t: *t,
    });
}

/// Unit vector for any nonzero finite 3-vector (|b|^2 outside (1e-290, 1e290): divide by max |b_i| first).
#[inline(always)]
fn unit3(b: &V3, fl: &mut u32) -> V3 {
    let (mut x1, mut x2, mut x3) = (b[0], b[1], b[2]);
    let mut ss = x1 * x1 + x2 * x2 + x3 * x3;
    if !(1e-290 < ss && ss < 1e290) {
        *fl |= path::RESCALE;
        (x1, x2, x3, ss) = unit3_rescale(x1, x2, x3);
    }
    let s = 1.0 / ss.sqrt();
    [x1 * s, x2 * s, x3 * s]
}

#[inline(never)]
#[cold]
fn unit3_rescale(x1: f64, x2: f64, x3: f64) -> (f64, f64, f64, f64) {
    let m = jmax(jmax(x1.abs(), x2.abs()), x3.abs());
    let (x1, x2, x3) = (x1 / m, x2 / m, x3 / m);
    (x1, x2, x3, x1 * x1 + x2 * x2 + x3 * x3)
}

const TAU_POW: [f64; 10] = [1e8, 1e16, 1e24, 1e32, 1e40, 1e48, 1e56, 1e64, 1e72, 1e80];

/// Scaled root bound test: a root near z = inf (the GT may sit in the hole of the affine chart).
#[inline(always)]
fn near_inf(c: &[f64; 11]) -> bool {
    if c[10] == 0.0 {
        return true;
    }
    let ci = 1.0 / c[10].abs();
    for i in 1..11usize {
        if !(c[i - 1].abs() * ci <= TAU_POW[11 - i - 1]) {
            return true; // NaN-true
        }
    }
    !(ci < f64::INFINITY)
}

#[inline(always)]
fn all_finite(r: &V9, t: &V3) -> bool {
    r.iter().all(|x| x.is_finite()) && t.iter().all(|x| x.is_finite())
}

/// Cheirality on a polished (R, t) (E = [t]x R); keeps the selected pose if new.
fn push_checked(out: &mut Poses, cx: &Ctx, r: &V9, t: &V3, fl: &mut u32) {
    if !all_finite(r, t) {
        return;
    }
    let e = ess(r, t);
    let ss = e.iter().fold(0.0, |s, x| s + x * x);
    let (tp, cf, k) = pose_parts(&vscale(&e, (2.0 / ss).sqrt()));
    let (r0, mut tc, np) = cheir_pose(&tp, &cf, &k, &cx.p, &cx.q);
    if np < MIN_FRONT {
        return;
    }
    let mut rc = rot_start(&r0);
    let st = cx.newton(&mut rc, &mut tc, 2);
    let conv = cx.converge(&mut rc, &mut tc, st, fl);
    resign(&rc, &mut tc, &cx.p, &cx.q, fl);
    // L1 distance in the same row-major order as the reference loop
    if out
        .iter()
        .any(|o| o.r.iter().zip(&rc).fold(0.0, |d, (a, b)| d + (a - b).abs()) < 1e-9)
    {
        return;
    }
    push_pose(out, cx, &rc, &tc, conv, fl);
}

/// A cheirality-rejected root near pure rotation: polish the R0 = cof(E) - [t]x E start and re-check it.
#[inline(never)]
fn rescue_small(out: &mut Poses, cx: &Ctx, t: &V3, cf: &V9, k: &V9, fl: &mut u32) -> bool {
    let ra = vsub(cf, k);
    if !(small_par(&ra, &cx.p, &cx.q) || small_par(&vadd(cf, k), &cx.p, &cx.q)) {
        return false;
    }
    *fl |= path::RESCUE;
    let (mut r, mut tt) = (rot_start(&ra), *t);
    cx.newton(&mut r, &mut tt, 8);
    push_checked(out, cx, &r, &tt, fl);
    true
}

/// Small-parallax gate: the compensated-determinant roots of the same rows (Newton x8 each) and the
/// Procrustes-start candidate (pure-rotation model), each through cheirality, kept if new.
#[inline(never)]
fn small_extra(out: &mut Poses, cx: &Ctx, rows: &Rows, fl: &mut u32) {
    let r0 = procrustes(&cx.p, &cx.q);
    if !small_par(&r0, &cx.p, &cx.q) {
        return;
    }
    *fl |= path::SMALL_DD;
    let mut roots2 = [0.0; MAX_ROOTS_DD];
    let (n2, _) = real_roots10(&mut roots2, &det_poly_dd(rows));
    for &z in &roots2[..n2] {
        let (t, cf, k) = root_parts(cx, rows, z);
        let (ra, mut tt, _) = cheir_pose(&t, &cf, &k, &cx.p, &cx.q);
        let mut r = rot_start(&ra);
        cx.newton(&mut r, &mut tt, 8);
        push_checked(out, cx, &r, &tt, fl);
    }
    let (mut r, mut t) = (rot_start(&r0), t_ls(&r0, &cx.p, &cx.q));
    cx.newton(&mut r, &mut t, 8);
    push_checked(out, cx, &r, &t, fl);
}

/// Solves the calibrated five-point relative pose problem (see the module contract).
///
/// Clears `out`, fills it with the pose hypotheses in deterministic order and returns the [`path`]
/// flags of the code paths taken.
pub fn solve(b1: &[[f64; 3]; 5], b2: &[[f64; 3]; 5], out: &mut Poses) -> u32 {
    out.clear();
    let mut flags = 0;
    let p: Pts = std::array::from_fn(|i| unit3(&b1[i], &mut flags));
    let q: Pts = std::array::from_fn(|i| unit3(&b2[i], &mut flags));
    let (mut a, mut n) = ([0.0; 45], [0.0; 36]);
    fill_at(&mut a, &p, &q);
    nullspace_hh(&mut n, &mut a);
    let (mut rows, mut dp) = reduce(&n);
    if near_inf(&dp) {
        flags |= path::FRAME_RETRY;
        if !dp.iter().all(|x| x.is_finite()) {
            flags |= path::NONFINITE_RETRY;
        }
        frame(&mut n);
        (rows, dp) = reduce(&n);
    }
    let mut roots = [0.0; MAX_ROOTS];
    let (mut nr, fire) = real_roots10(&mut roots, &dp);
    if fire {
        flags |= path::DD_UNION;
        nr = dd_union(&mut roots, nr, &rows, &mut flags);
    }
    let cx = Ctx { n, p, q };
    let mut flagged = false;
    for &z in &roots[..nr] {
        let (t, cf, k) = root_parts(&cx, &rows, z);
        let (r0, mut tt, np) = cheir_pose(&t, &cf, &k, &cx.p, &cx.q);
        if np < MIN_FRONT {
            flagged |= rescue_small(out, &cx, &t, &cf, &k, &mut flags);
            continue;
        }
        let mut r = rot_quat(&r0); // structural rotation start
        let mut st = cx.newton(&mut r, &mut tt, POSE_ITERS);
        if small_par(&r, &cx.p, &cx.q) {
            flagged = true;
            flags |= path::GATE_NEWTON;
            st = cx.newton(&mut r, &mut tt, 8);
            resign(&r, &mut tt, &cx.p, &cx.q, &mut flags);
        }
        let conv = cx.converge(&mut r, &mut tt, st, &mut flags);
        push_pose(out, &cx, &r, &tt, conv, &mut flags);
    }
    // `small_extra` re-checks the full Procrustes gate, so the data trigger only appends candidates.
    if flagged || out.is_empty() || angles_preserved(&cx.p, &cx.q) {
        flags |= path::SMALL_EXTRA;
        small_extra(out, &cx, &rows, &mut flags);
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_min_follow_julia_float_semantics() {
        assert_eq!(jmax(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(jmax(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(jmin(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert!(jmax(f64::NAN, 1.0).is_nan() && jmin(1.0, f64::NAN).is_nan());
        assert_eq!(jmax(2.0, 3.0), 3.0);
    }
}
