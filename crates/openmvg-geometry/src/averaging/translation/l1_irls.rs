// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # L1-IRLS Translation Averaging
//!
//! Its teaching docs are on [`average_translations`], where docs.rs shows them.

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use openmvg_core::model::{EstimationError, EstimationResult, RelativeTranslation, ViewId};
use std::collections::BTreeMap;

use crate::averaging::{TranslationConfig, TranslationEstimate, TranslationSolver};

// ============================================================================
// Named constants (replacing scattered magic numbers)
// ============================================================================

/// Minimum absolute value for a scale or dot product to be considered non-degenerate.
const SCALE_EPSILON: f64 = 1e-10;

/// Minimum allowed scale value (prevents astronomically large scales).
const SCALE_MIN: f64 = 1e-6;

/// Maximum allowed scale value (prevents astronomically large scales).
const SCALE_MAX: f64 = 1e6;

/// Tikhonov regularization parameter λ for the normal equations.
const TIKHONOV_LAMBDA: f64 = 1e-10;

/// PCG convergence tolerance on ‖r‖ relative to ‖b‖ (plus an absolute floor), matching the
/// global-positioning kernel. Edge weights are inlier counts, so an absolute-only tolerance is
/// never reached and the solve silently stops at its iteration cap.
const CG_RELATIVE_TOL: f64 = 1e-10;
const CG_ABSOLUTE_TOL: f64 = 1e-12;

/// Switch from dense Cholesky to matrix-free CG at this many free DOFs (3 × views, so 100
/// views).
const MATRIX_FREE_THRESHOLD: usize = 300;

/// Pre-computed edge data for the inner loop.
struct Edge {
    i: usize,
    j: usize,
    direction: Vector3<f64>,
    /// Square root of the input edge weight, multiplied into every IRLS weight.
    weight: f64,
}

/// Configuration for the L1-IRLS translation solver.
///
/// `huber_delta` is solver-specific, while the shared `TranslationConfig` carries
/// only the iteration and tolerance controls common to translation solvers.
/// Crate-private: the public entry point [`super::average_translations`]
/// runs the default.
#[derive(Debug, Clone)]
pub(crate) struct L1IrlsConfig {
    /// Huber loss delta, annealed from `3 × huber_delta` (loose, more L2-like —
    /// better for initialization) down to `huber_delta` (tight, more L1-like —
    /// better outlier rejection) over the solve's outer iterations.
    pub huber_delta: f64,
}

impl Default for L1IrlsConfig {
    fn default() -> Self {
        Self { huber_delta: 0.05 }
    }
}

/// L1-IRLS Translation Solver.
///
/// Uses Iteratively Reweighted Least Squares with per-edge scale variables
/// to solve for camera centers robustly.
///
/// For large scenes (n > 100 views), uses matrix-free Conjugate Gradient
/// which is O(m) per iteration instead of O(n²).
#[derive(Debug, Clone)]
pub struct L1IrlsTranslationSolver {
    config: L1IrlsConfig,
}

impl Default for L1IrlsTranslationSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl L1IrlsTranslationSolver {
    /// Create a new L1-IRLS translation solver (default `L1IrlsConfig`: `huber_delta = 0.05`).
    pub fn new() -> Self {
        Self {
            config: L1IrlsConfig::default(),
        }
    }

    /// Create with an explicit `L1IrlsConfig`.
    #[cfg(test)]
    pub(crate) fn with_config(config: L1IrlsConfig) -> Self {
        Self { config }
    }

    /// Read a `Vector3<f64>` from a flat position buffer at the given 3-element offset.
    #[inline]
    fn read_vec3(positions: &[f64], offset: usize) -> Vector3<f64> {
        debug_assert!(
            offset + 2 < positions.len(),
            "read_vec3: offset {} out of bounds (len {})",
            offset,
            positions.len()
        );
        Vector3::new(
            positions[offset],
            positions[offset + 1],
            positions[offset + 2],
        )
    }

    /// Compute Δt = t_j - t_i from the flat position buffer.
    /// Camera 0 at origin (excluded), so camera `i` maps to offset `3*(i-1)`.
    #[inline]
    fn delta_t(positions: &[f64], e: &Edge) -> Vector3<f64> {
        let ti = if e.i == 0 {
            Vector3::zeros()
        } else {
            Self::read_vec3(positions, 3 * (e.i - 1))
        };
        let tj = if e.j == 0 {
            Vector3::zeros()
        } else {
            Self::read_vec3(positions, 3 * (e.j - 1))
        };
        tj - ti
    }

    /// Matrix-free computation of (AᵀWA + λI) * x.
    ///
    /// Computes the matrix-vector product without forming the matrix.
    /// Complexity: O(m) where m = number of edges, instead of O(n²).
    ///
    /// For each edge (i,j) with weight w_k:
    ///   - Contribution to result[i] += w_k * (x[i] - x[j])
    ///   - Contribution to result[j] += w_k * (x[j] - x[i])
    #[inline]
    fn apply_ata(x: &[f64], weighted_edges: &[(usize, usize, f64)], reg: f64, result: &mut [f64]) {
        for (i, r) in result.iter_mut().enumerate() {
            *r = reg * x[i];
        }

        // Camera `i` (> 0) owns block `i - 1`; camera 0 is the fixed origin.
        let (x, _) = x.as_chunks::<3>();
        let (result, _) = result.as_chunks_mut::<3>();
        for &(ei, ej, w) in weighted_edges {
            if ei > 0 && ej > 0 {
                let (xi, xj) = (&x[ei - 1], &x[ej - 1]);
                let step: [f64; 3] = std::array::from_fn(|c| w * (xj[c] - xi[c]));
                for (r, s) in result[ei - 1].iter_mut().zip(step) {
                    *r -= s;
                }
                for (r, s) in result[ej - 1].iter_mut().zip(step) {
                    *r += s;
                }
            } else if ei > 0 {
                for (r, v) in result[ei - 1].iter_mut().zip(x[ei - 1]) {
                    *r += w * v;
                }
            } else if ej > 0 {
                for (r, v) in result[ej - 1].iter_mut().zip(x[ej - 1]) {
                    *r += w * v;
                }
            }
        }
    }

    /// Compute AᵀWb (the right-hand side) for the matrix-free CG path.
    ///
    /// For each edge k with direction d_k and weight w_k/s_k:
    ///   - b[i] -= w_k/s_k * d_k
    ///   - b[j] += w_k/s_k * d_k
    fn compute_atb(edges: &[Edge], weights: &[f64], scales: &[f64], n_free: usize) -> Vec<f64> {
        let mut atb = vec![0.0; n_free];
        // Camera `i` (> 0) owns block `i - 1`; camera 0 is the fixed origin.
        let (blocks, _) = atb.as_chunks_mut::<3>();

        for (k, e) in edges.iter().enumerate() {
            let sk = scales[k];
            if !sk.is_finite() || sk.abs() < SCALE_EPSILON {
                continue;
            }
            let ws_dir = weights[k] / sk;

            let step: [f64; 3] = std::array::from_fn(|c| ws_dir * e.direction[c]);
            if e.i > 0 && e.j > 0 {
                for (b, s) in blocks[e.j - 1].iter_mut().zip(step) {
                    *b += s;
                }
                for (b, s) in blocks[e.i - 1].iter_mut().zip(step) {
                    *b -= s;
                }
            } else if e.i > 0 {
                for (b, s) in blocks[e.i - 1].iter_mut().zip(step) {
                    *b -= s;
                }
            } else if e.j > 0 {
                for (b, s) in blocks[e.j - 1].iter_mut().zip(step) {
                    *b += s;
                }
            }
        }

        atb
    }

    /// Build weighted edges as `(i, j, w_k / s_k²)` tuples for the matrix-free CG,
    /// filtering out degenerate edges.
    fn build_weighted_edges(
        edges: &[Edge],
        weights: &[f64],
        scales: &[f64],
    ) -> Vec<(usize, usize, f64)> {
        edges
            .iter()
            .enumerate()
            .filter_map(|(k, e)| {
                let sk = scales[k];
                if !sk.is_finite() || sk.abs() < SCALE_EPSILON {
                    return None;
                }
                Some((e.i, e.j, weights[k] / (sk * sk)))
            })
            .collect()
    }

    /// Matrix-free Jacobi-preconditioned Conjugate Gradient solver.
    ///
    /// Solves (AᵀWA + λI) x = b without forming the matrix, starting from `initial`. The graph
    /// Laplacian of a weakly linked, chain-like view graph is badly conditioned: truncated plain
    /// CG leaves its slow low-frequency modes (a whole camera block's offset) near the start, so
    /// the solve must be preconditioned and certified rather than capped. Starting from the
    /// previous IRLS iterate leaves only that iterate's correction to resolve. Returns `None`
    /// when the residual is not certified within `min(8n/3, 4000)` iterations, `n = n_free`.
    fn solve_cg_matrix_free(
        edges: &[Edge],
        weights: &[f64],
        scales: &[f64],
        reg: f64,
        initial: &[f64],
    ) -> Option<Vec<f64>> {
        let n_free = initial.len();
        let max_iter = (8 * n_free / 3).clamp(64, 4000);

        let weighted_edges = Self::build_weighted_edges(edges, weights, scales);
        let b = Self::compute_atb(edges, weights, scales, n_free);
        let mut diagonal = vec![reg; n_free];
        let (diagonal_blocks, _) = diagonal.as_chunks_mut::<3>();
        for &(ei, ej, w) in &weighted_edges {
            for camera in [ei, ej] {
                if camera > 0 {
                    for d in &mut diagonal_blocks[camera - 1] {
                        *d += w;
                    }
                }
            }
        }

        let target =
            CG_ABSOLUTE_TOL.max(CG_RELATIVE_TOL * b.iter().map(|v| v * v).sum::<f64>().sqrt());
        let mut x = initial.to_vec();
        let mut ap = vec![0.0; n_free];
        Self::apply_ata(&x, &weighted_edges, reg, &mut ap);
        let mut r: Vec<f64> = b.iter().zip(&ap).map(|(bi, axi)| bi - axi).collect();
        let mut z: Vec<f64> = r.iter().zip(&diagonal).map(|(ri, di)| ri / di).collect();
        let mut p = z.clone();
        let mut rz_old: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();

        for _iter in 0..max_iter {
            if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= target {
                return Some(x);
            }
            Self::apply_ata(&p, &weighted_edges, reg, &mut ap);

            let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
            if !pap.is_finite() || pap <= f64::EPSILON * rz_old.abs() {
                return None;
            }
            let alpha = rz_old / pap;

            for (xi, pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }

            for (ri, api) in r.iter_mut().zip(ap.iter()) {
                *ri -= alpha * api;
            }
            for ((zi, ri), di) in z.iter_mut().zip(&r).zip(&diagonal) {
                *zi = ri / di;
            }

            let rz_new: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
            let beta = rz_new / rz_old;
            for (pi, zi) in p.iter_mut().zip(z.iter()) {
                *pi = zi + beta * *pi;
            }
            rz_old = rz_new;
        }

        (r.iter().map(|v| v * v).sum::<f64>().sqrt() <= target).then_some(x)
    }

    /// Accumulate weighted normal equations into pre-allocated buffers and solve
    /// via Cholesky (with LU fallback).
    ///
    /// System: (A^T diag(w/s²) A + λI) t = A^T diag(w/s) d
    ///
    /// The caller routes large systems to `solve_cg_matrix_free`, so this
    /// method only handles the dense Cholesky path.
    fn solve_normal_equations(
        ata: &mut DMatrix<f64>,
        atb: &mut DVector<f64>,
        edges: &[Edge],
        weights: &[f64],
        scales: &[f64],
        reg: f64,
    ) -> Option<Vec<f64>> {
        let n_free = ata.nrows();

        ata.fill(0.0);
        atb.fill(0.0);

        for (k, e) in edges.iter().enumerate() {
            let sk = scales[k];
            if !sk.is_finite() || sk.abs() < SCALE_EPSILON {
                continue;
            }
            let wk = weights[k];
            let ws = wk / (sk * sk);

            if e.i > 0 && e.j > 0 {
                let ii = 3 * (e.i - 1);
                let jj = 3 * (e.j - 1);

                for c in 0..3_usize {
                    ata[(ii + c, ii + c)] += ws;
                    ata[(jj + c, jj + c)] += ws;
                    ata[(ii + c, jj + c)] -= ws;
                    ata[(jj + c, ii + c)] -= ws;

                    let ab = wk * e.direction[c] / sk;
                    atb[jj + c] += ab;
                    atb[ii + c] -= ab;
                }
            } else if e.i > 0 {
                let ii = 3 * (e.i - 1);
                for c in 0..3_usize {
                    ata[(ii + c, ii + c)] += ws;
                    atb[ii + c] -= wk * e.direction[c] / sk;
                }
            } else if e.j > 0 {
                let jj = 3 * (e.j - 1);
                for c in 0..3_usize {
                    ata[(jj + c, jj + c)] += ws;
                    atb[jj + c] += wk * e.direction[c] / sk;
                }
            }
        }

        // Tikhonov regularization
        for i in 0..n_free {
            ata[(i, i)] += reg;
        }

        if let Some(chol) = ata.clone().cholesky() {
            let sol = chol.solve(atb);
            Some(sol.iter().copied().collect())
        } else {
            ata.clone()
                .lu()
                .solve(atb)
                .map(|s| s.iter().copied().collect())
        }
    }
}

impl TranslationSolver for L1IrlsTranslationSolver {
    fn solve(
        &self,
        relative_translations: &[RelativeTranslation],
        absolute_rotations: &BTreeMap<ViewId, Rotation3<f64>>,
        config: &TranslationConfig,
    ) -> EstimationResult<TranslationEstimate> {
        config.validate()?;
        if !self.config.huber_delta.is_finite() || self.config.huber_delta <= 0.0 {
            return Err(EstimationError::InvalidInput(
                "L1-IRLS huber_delta must be finite and positive".to_string(),
            ));
        }
        if relative_translations.is_empty() {
            return Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }
        for relative in relative_translations {
            relative.validate()?;
            for view in [relative.view_i(), relative.view_j()] {
                let rotation = absolute_rotations.get(&view).ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "missing absolute rotation for translation view {view}"
                    ))
                })?;
                if rotation.matrix().iter().any(|value| !value.is_finite()) {
                    return Err(EstimationError::InvalidInput(format!(
                        "absolute rotation for view {view} must be finite"
                    )));
                }
            }
        }

        // Build view index mapping
        let mut view_to_idx: BTreeMap<ViewId, usize> = BTreeMap::new();
        let mut idx_to_view: Vec<ViewId> = Vec::new();

        for rel in relative_translations {
            for v in [rel.view_i(), rel.view_j()] {
                if let std::collections::btree_map::Entry::Vacant(entry) = view_to_idx.entry(v) {
                    entry.insert(idx_to_view.len());
                    idx_to_view.push(v);
                }
            }
        }

        let n = idx_to_view.len();
        if n < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: n,
            });
        }
        let n_free = 3 * (n - 1);

        // Pre-compute world-frame edges
        let edges: Vec<Edge> = relative_translations
            .iter()
            .map(|rel| {
                let r_i = &absolute_rotations[&rel.view_i()];
                let i = view_to_idx[&rel.view_i()];
                let j = view_to_idx[&rel.view_j()];
                let dir = (r_i.transpose() * rel.direction()).normalize();
                Edge {
                    i,
                    j,
                    direction: dir,
                    // Temper count-like confidences while preserving fractional
                    // caller weights instead of silently promoting them to one.
                    weight: rel.weight().sqrt(),
                }
            })
            .collect();

        if edges.is_empty() {
            return Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }

        let mut adjacency = vec![Vec::new(); n];
        for edge in &edges {
            adjacency[edge.i].push(edge.j);
            adjacency[edge.j].push(edge.i);
        }
        let mut visited = vec![false; n];
        visited[0] = true;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            for &j in &adjacency[i] {
                if !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        if visited.iter().any(|is_visited| !is_visited) {
            return Err(EstimationError::InvalidInput(
                "relative-translation graph must be connected".to_string(),
            ));
        }

        let m = edges.len();

        // Pre-allocate dense buffers only when needed (small systems).
        let use_matrix_free = n_free > MATRIX_FREE_THRESHOLD;
        let mut ata = if use_matrix_free {
            DMatrix::zeros(0, 0)
        } else {
            DMatrix::zeros(n_free, n_free)
        };
        let mut atb = if use_matrix_free {
            DVector::zeros(0)
        } else {
            DVector::zeros(n_free)
        };

        // Working buffers
        let mut s_vec = vec![1.0_f64; m];
        // Initialize the IRLS weights from the input edge weights.
        let mut w_vec: Vec<f64> = edges.iter().map(|e| e.weight).collect();
        let mut t = vec![0.0_f64; n_free];

        // Adaptive Huber annealing.
        // Start with larger delta (more L2-like, better for initialization)
        // Anneal to target delta (more L1-like, better outlier rejection)
        let huber_start = self.config.huber_delta * 3.0; // Start more tolerant
        let huber_end = self.config.huber_delta;
        let mut prev_residual = f64::MAX;

        for iter in 0..config.max_iterations() {
            // Update scales: s_k = clamp(||Δt||² / (Δt · d))
            for (k, e) in edges.iter().enumerate() {
                let dt = Self::delta_t(&t, e);
                let dot = dt.dot(&e.direction);
                s_vec[k] = if dot > SCALE_EPSILON {
                    (dt.norm_squared() / dot).clamp(SCALE_MIN, SCALE_MAX)
                } else {
                    1.0
                };
            }

            // Compute the adaptive Huber delta (anneal from loose to tight).
            let progress = iter as f64 / config.max_iterations() as f64;
            let current_huber = huber_start + (huber_end - huber_start) * progress;

            // Update IRLS weights: Huber loss × input weight
            for (k, e) in edges.iter().enumerate() {
                let sk = s_vec[k];
                if sk.abs() < SCALE_EPSILON {
                    w_vec[k] = e.weight;
                    continue;
                }
                let r_norm = (Self::delta_t(&t, e) / sk - e.direction).norm();
                // Multiply the Huber weight by the input edge weight.
                let huber_weight = if r_norm <= current_huber {
                    1.0
                } else {
                    current_huber / r_norm
                };
                w_vec[k] = huber_weight * e.weight;
            }

            // Solve weighted normal equations
            let solution = if use_matrix_free {
                Self::solve_cg_matrix_free(&edges, &w_vec, &s_vec, TIKHONOV_LAMBDA, &t)
            } else {
                Self::solve_normal_equations(
                    &mut ata,
                    &mut atb,
                    &edges,
                    &w_vec,
                    &s_vec,
                    TIKHONOV_LAMBDA,
                )
            };

            match solution {
                Some(sol) if sol.iter().all(|v| v.is_finite()) => {
                    t = sol;
                }
                Some(_) => {
                    return Err(EstimationError::Numerical(
                        "NaN/Inf in solution vector".to_string(),
                    ));
                }
                None => {
                    return Err(EstimationError::Numerical(
                        "failed to solve the linear system".to_string(),
                    ));
                }
            }

            // Check convergence via mean angular residual
            let mut total_residual = 0.0;
            let mut count = 0usize;
            for e in &edges {
                let dt = Self::delta_t(&t, e);
                if dt.norm() < SCALE_EPSILON {
                    continue;
                }
                let cos_a = dt.normalize().dot(&e.direction).clamp(-1.0, 1.0);
                total_residual += cos_a.acos();
                count += 1;
            }

            if count == 0 {
                return Err(EstimationError::DegenerateConfiguration {
                    reason: "all edges have near-zero translation difference".to_string(),
                });
            }

            let mean_residual = total_residual / count as f64;

            if (prev_residual - mean_residual).abs() < config.tolerance().value() {
                let candidate = TranslationEstimate {
                    translations: build_output(&t, &idx_to_view),
                    iteration_count: iter + 1,
                    residual: crate::averaging::checked_angular_residual(mean_residual)?,
                };
                return Ok(candidate);
            }
            prev_residual = mean_residual;
        }

        let candidate = TranslationEstimate {
            translations: build_output(&t, &idx_to_view),
            iteration_count: config.max_iterations(),
            residual: crate::averaging::checked_angular_residual(prev_residual)?,
        };
        Ok(candidate)
    }
}

/// Convert a flat position buffer to deterministic `ViewId` order.
fn build_output(t: &[f64], idx_to_view: &[ViewId]) -> BTreeMap<ViewId, Vector3<f64>> {
    let mut translations = BTreeMap::new();
    translations.insert(idx_to_view[0], Vector3::zeros());
    for (i, &view_id) in idx_to_view.iter().enumerate().skip(1) {
        let b = 3 * (i - 1);
        translations.insert(view_id, Vector3::new(t[b], t[b + 1], t[b + 2]));
    }
    translations
}

#[cfg(test)]
mod tests;
