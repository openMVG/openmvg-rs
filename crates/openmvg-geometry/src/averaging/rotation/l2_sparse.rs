// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # L2 sparse rotation averaging
//!
//! Its teaching docs are on [`average_rotations`], where docs.rs shows them.

use nalgebra::{Matrix3, Rotation3};
use openmvg_ba::numerics::sparse::{SparseCholesky, SymmetricPattern};
use openmvg_core::model::{EstimationError, EstimationResult, RelativeRotation, ViewId};
use std::collections::BTreeMap;

use crate::averaging::{RotationConfig, RotationEstimate, RotationSolver};

/// L2 sparse rotation averaging solver.
///
/// Uses sparse matrices for memory-efficient rotation averaging at any size: the
/// rotations are recovered by a single direct sparse-Cholesky solve of the grounded normal
/// equations (the best-connected camera fixed to I). A rank-deficient / disconnected graph
/// the sparse Cholesky cannot factor falls back to the dense reference solver. Repeated
/// solves through one [`L2SparseWorkspace`], as in the IRLS loop, reuse its structure work.
#[derive(Debug, Clone, Default)]
pub struct L2SparseRotationSolver;

impl L2SparseRotationSolver {
    pub fn new() -> Self {
        Self
    }

    /// Build the view index mapping (view_id -> contiguous 0-based index).
    fn build_view_indices(
        relative_rotations: &[RelativeRotation],
    ) -> (BTreeMap<ViewId, usize>, Vec<ViewId>) {
        let mut view_set = std::collections::BTreeSet::new();
        for rel in relative_rotations {
            view_set.insert(rel.view_i());
            view_set.insert(rel.view_j());
        }
        let views: Vec<ViewId> = view_set.into_iter().collect();
        let view_to_idx: BTreeMap<ViewId, usize> =
            views.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        (view_to_idx, views)
    }

    /// Compute the mean angular edge residual in radians.
    fn compute_residual(
        rotations: &BTreeMap<ViewId, Rotation3<f64>>,
        relative_rotations: &[RelativeRotation],
    ) -> f64 {
        super::mean_angular_residual(rotations, relative_rotations)
    }
}

/// State reused by [`L2SparseRotationSolver`] across solves over one edge list.
///
/// IRLS changes only the edge weights between solves. The block graph is rebuilt only when
/// the edge list changes, and the symbolic Cholesky factorization only when the pattern of
/// `H_rr` changes (another anchor camera, or a block entry that cancels to exactly zero).
/// The rest is buffers, so a reused solve gives the same bits as a fresh one.
#[derive(Default)]
pub(crate) struct L2SparseWorkspace {
    graph: BlockGraph,
    /// Weighted camera degrees: diagonal block `c` of `H` is `diag[c]·I`.
    diag: Vec<f64>,
    /// Off-diagonal blocks of `H`, indexed by the slots of `graph`.
    blocks: Vec<Matrix3<f64>>,
    /// Lower triangle of `H_rr` in compressed sparse columns.
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
    /// Analysis of the last `H_rr` pattern, with storage for three right-hand sides.
    cholesky: Option<SparseCholesky>,
}

/// Block structure of `H` for one edge list.
#[derive(Default)]
struct BlockGraph {
    /// The `(view_i, view_j)` sequence the graph was built from.
    edges: Vec<(ViewId, ViewId)>,
    /// Views in `ViewId` order: camera `c` is `views[c]`.
    views: Vec<ViewId>,
    /// Per edge, its cameras and block slots; `None` for a self-loop.
    edge_slots: Vec<Option<EdgeSlots>>,
    /// The off-diagonal blocks of column camera `c` are the slots
    /// `col_start[c]..col_start[c + 1]`, by ascending row camera `block_row[slot]`.
    col_start: Vec<usize>,
    block_row: Vec<usize>,
}

/// The cameras `i`, `j` of an edge and the slots of its blocks `H_ij` and `H_ji`.
#[derive(Clone, Copy)]
struct EdgeSlots {
    i: usize,
    j: usize,
    ij: usize,
    ji: usize,
}

impl BlockGraph {
    fn new(relative_rotations: &[RelativeRotation]) -> Self {
        let (view_to_idx, views) = L2SparseRotationSolver::build_view_indices(relative_rotations);
        let edge_cameras: Vec<Option<(usize, usize)>> = relative_rotations
            .iter()
            .map(|rel| {
                match (
                    view_to_idx.get(&rel.view_i()),
                    view_to_idx.get(&rel.view_j()),
                ) {
                    (Some(&i), Some(&j)) if i != j => Some((i, j)),
                    _ => None,
                }
            })
            .collect();

        // Edge e adds to block (i, j), stored in column j, and to block (j, i), in column i.
        // Sorting these references by (column, row) numbers the blocks column by column.
        let mut references = Vec::with_capacity(2 * edge_cameras.len());
        for (e, pair) in edge_cameras.iter().enumerate() {
            if let Some((i, j)) = *pair {
                references.push((j, i, 2 * e));
                references.push((i, j, 2 * e + 1));
            }
        }
        references.sort_unstable();
        let mut slot_of = vec![0; 2 * edge_cameras.len()];
        let mut col_start = vec![0; views.len() + 1];
        let mut block_row = Vec::new();
        let mut previous = None;
        for (col, row, reference) in references {
            if previous != Some((col, row)) {
                previous = Some((col, row));
                block_row.push(row);
                col_start[col + 1] += 1;
            }
            slot_of[reference] = block_row.len() - 1;
        }
        let mut total = 0;
        for start in &mut col_start {
            total += *start;
            *start = total;
        }

        let edge_slots = edge_cameras
            .iter()
            .enumerate()
            .map(|(e, pair)| {
                pair.map(|(i, j)| EdgeSlots {
                    i,
                    j,
                    ij: slot_of[2 * e],
                    ji: slot_of[2 * e + 1],
                })
            })
            .collect();
        Self {
            edges: relative_rotations
                .iter()
                .map(|rel| (rel.view_i(), rel.view_j()))
                .collect(),
            views,
            edge_slots,
            col_start,
            block_row,
        }
    }

    fn matches(&self, relative_rotations: &[RelativeRotation]) -> bool {
        self.edges.len() == relative_rotations.len()
            && self
                .edges
                .iter()
                .zip(relative_rotations)
                .all(|(&edge, rel)| edge == (rel.view_i(), rel.view_j()))
    }
}

/// Order and analyze the pattern of the `dim × dim` matrix `H_rr` and reserve storage for its
/// three right-hand sides, the columns of −H_r0.
fn analyze_grounded(
    dim: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
) -> EstimationResult<SparseCholesky> {
    let pattern = SymmetricPattern::new(dim, col_ptr, row_idx)
        .ok_or_else(|| cholesky_failed("malformed pattern"))?;
    let mut cholesky = SparseCholesky::analyze(pattern)
        .ok_or_else(|| cholesky_failed("symbolic analysis failed"))?;
    if !cholesky.reserve(3) {
        return Err(cholesky_failed("workspace allocation failed"));
    }
    Ok(cholesky)
}

/// Index of `camera` in the grounded system, which drops the anchor camera.
fn grounded_index(camera: usize, anchor: usize) -> usize {
    if camera < anchor { camera } else { camera - 1 }
}

impl L2SparseWorkspace {
    /// Rebuild the block graph if the edge list changed; returns the number of cameras.
    fn update_graph(&mut self, relative_rotations: &[RelativeRotation]) -> usize {
        if !self.graph.matches(relative_rotations) {
            self.graph = BlockGraph::new(relative_rotations);
        }
        self.graph.views.len()
    }

    /// Average the rotations with a single direct sparse linear solve (faer sparse Cholesky),
    /// grounding the gauge instead of an eigen/nullspace search.
    ///
    /// faer provides sparse Cholesky/LU/QR *linsolve* but no sparse *eigensolver*, and the L2
    /// rotation solution is the gauge nullspace of H = AᵀA. Rather than emulate an eigensolver,
    /// it fixes the best-connected camera's rotation to I (grounding removes the 3-dim gauge) and
    /// solve the resulting SPD grounded normal equations `H_rr · X_r = −H_r0` directly, then
    /// project each 3×3 block to SO(3). One factorization + one triangular solve, no iteration
    /// (a tiny Tikhonov shift keeps H_rr factorable on weakly-connected/disconnected graphs).
    /// `H` is the 3×3-block weighted rotation Laplacian: diagonal block i = (Σ w)·I,
    /// off-diagonal block (i,j) = −w·R_ijᵀ. Deterministic. `pub(crate)` so a test can exercise
    /// this path directly (the public `solve` masks it behind a dense fallback).
    pub(crate) fn solve_grounded(
        &mut self,
        relative_rotations: &[RelativeRotation],
    ) -> EstimationResult<BTreeMap<ViewId, Rotation3<f64>>> {
        let num_cameras = self.update_graph(relative_rotations);
        self.assemble(relative_rotations);

        // Ground the gauge at the most-connected camera (best-conditioned reference).
        let diag = &self.diag;
        let anchor = (0..num_cameras)
            .max_by(|&a, &b| {
                diag[a]
                    .partial_cmp(&diag[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let empty = || EstimationError::Numerical("empty rotation graph".to_string());
        let Some(&anchor_degree) = diag.get(anchor) else {
            return Err(empty());
        };
        if anchor_degree <= 0.0 {
            return Err(empty());
        }
        // Tiny Tikhonov shift, relative to the largest camera degree. H_rr is already SPD for a
        // connected graph, but a weakly-connected or disconnected subgraph makes it singular;
        // adding σI keeps it factorable so the solve never degrades to the O((3n)³) dense
        // fallback on every IRLS iteration, and bounds otherwise-huge near-singular blocks. σ is
        // ~1e-9 of the diagonal scale, negligible for well-connected cameras (verified against
        // the dense solver).
        let sigma = anchor_degree * 1e-9;
        let rows = 3 * (num_cameras - 1);
        let mut x_rest = self.lay_out_grounded(anchor, sigma); // solved in place: rows × 3

        // Analyze the pattern only when it differs from the cached one. The shared sparse
        // Cholesky runs on one thread, so the result does not depend on the worker count.
        let cholesky = match &mut self.cholesky {
            Some(cached)
                if cached.pattern().column_pointers() == self.col_ptr
                    && cached.pattern().row_indices() == self.row_idx =>
            {
                cached
            }
            slot => slot.insert(analyze_grounded(
                rows,
                std::mem::take(&mut self.col_ptr),
                std::mem::take(&mut self.row_idx),
            )?),
        };
        if !cholesky.factor(&self.values) {
            return Err(cholesky_failed("H_rr is not positive definite"));
        }
        if !cholesky.solve_in_place(&mut x_rest) {
            return Err(cholesky_failed("the solution is not finite"));
        }

        // Anchor = I; every other camera's 3×3 block from the solve, projected to SO(3).
        let mut rotations = BTreeMap::new();
        for (cam, &view_id) in self.graph.views.iter().enumerate() {
            let block = if cam == anchor {
                Matrix3::identity()
            } else {
                let rc = grounded_index(cam, anchor);
                Matrix3::from_fn(|r, c| x_rest[c * rows + 3 * rc + r])
            };
            // Reuse the dense solver's public SO(3) projection (no local duplicate).
            let rot = super::L2DenseRotationSolver::closest_rotation(&block)
                .unwrap_or_else(|_| Rotation3::identity());
            rotations.insert(view_id, rot);
        }
        Ok(rotations)
    }

    /// Sum the weighted edges into the camera degrees and the off-diagonal blocks of `H`,
    /// in edge order (parallel edges share a block).
    fn assemble(&mut self, relative_rotations: &[RelativeRotation]) {
        self.diag.clear();
        self.diag.resize(self.graph.views.len(), 0.0);
        self.blocks.clear();
        self.blocks
            .resize(self.graph.block_row.len(), Matrix3::zeros());
        for (rel, slots) in relative_rotations.iter().zip(&self.graph.edge_slots) {
            let Some(slots) = slots else {
                continue;
            };
            let weight = rel.weight();
            self.diag[slots.i] += weight;
            self.diag[slots.j] += weight;
            // H_ij = −w·R_ijᵀ, H_ji = −w·R_ij; scale once and reuse the transpose.
            let wr = weight * rel.rotation().matrix();
            self.blocks[slots.ij] -= wr.transpose();
            self.blocks[slots.ji] -= wr;
        }
    }

    /// Lay out the lower triangle of `H_rr` (the anchor's row and column removed) in compressed
    /// columns with ascending rows, and return the RHS `−H_r0` as three column-major columns.
    /// The factorization reads only that triangle, and the ordering is the same as for both
    /// triangles. Exact zeros stay out of the pattern.
    fn lay_out_grounded(&mut self, anchor: usize, sigma: f64) -> Vec<f64> {
        let graph = &self.graph;
        let num_cameras = graph.views.len();
        self.col_ptr.clear();
        self.row_idx.clear();
        self.values.clear();
        self.col_ptr.push(0);
        let rows = 3 * (num_cameras - 1);
        let mut rhs = vec![0.0; 3 * rows];
        for col_cam in 0..num_cameras {
            let slots = graph.col_start[col_cam]..graph.col_start[col_cam + 1];
            if col_cam == anchor {
                // H[i, anchor] moves to the RHS: H_rr·X_r = −H_r0.
                for slot in slots {
                    let ri = grounded_index(graph.block_row[slot], anchor);
                    let block = &self.blocks[slot];
                    for r in 0..3 {
                        for c in 0..3 {
                            rhs[c * rows + 3 * ri + r] -= block[(r, c)];
                        }
                    }
                }
                continue;
            }
            let rc = grounded_index(col_cam, anchor);
            let diagonal = self.diag[col_cam] + sigma;
            for c in 0..3 {
                // The diagonal entry, then the blocks below it.
                self.row_idx.push(3 * rc + c);
                self.values.push(diagonal);
                for slot in slots.clone() {
                    let row_cam = graph.block_row[slot];
                    // Blocks above the diagonal are not stored; the anchor's rows are dropped
                    // from the grounded system.
                    if row_cam < col_cam || row_cam == anchor {
                        continue;
                    }
                    let rr = grounded_index(row_cam, anchor);
                    for r in 0..3 {
                        let v = self.blocks[slot][(r, c)];
                        if v != 0.0 {
                            self.row_idx.push(3 * rr + r);
                            self.values.push(v);
                        }
                    }
                }
                self.col_ptr.push(self.row_idx.len());
            }
        }
        rhs
    }
}

fn cholesky_failed(reason: &str) -> EstimationError {
    EstimationError::Numerical(format!("grounded sparse Cholesky failed: {reason}"))
}

impl RotationSolver for L2SparseRotationSolver {
    type Workspace = L2SparseWorkspace;

    fn solve(
        &self,
        relative_rotations: &[RelativeRotation],
        config: &RotationConfig,
    ) -> EstimationResult<RotationEstimate> {
        self.solve_in(
            relative_rotations,
            config,
            &mut L2SparseWorkspace::default(),
        )
    }

    fn solve_in(
        &self,
        relative_rotations: &[RelativeRotation],
        config: &RotationConfig,
        workspace: &mut L2SparseWorkspace,
    ) -> EstimationResult<RotationEstimate> {
        if relative_rotations.is_empty() {
            return Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }
        for relative_rotation in relative_rotations {
            relative_rotation.validate()?;
        }

        let num_cameras = workspace.update_graph(relative_rotations);

        if num_cameras < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: num_cameras,
            });
        }

        // One direct sparse linear solve at every size: ground the gauge (fix the
        // best-connected camera = I) and solve the non-singular grounded normal equations with
        // faer's sparse Cholesky — the sparse tool for the job, since faer has no sparse
        // eigensolver. If the factorization degenerates (e.g. a rank-deficient / disconnected
        // graph), fall back to the dense reference solver [`L2DenseRotationSolver`] rather than
        // a private eigendecomposition clone. (That fallback is O(n³); it should only fire on
        // pathological graphs the sparse solve can't factor.)
        let mut global_rotations = match workspace.solve_grounded(relative_rotations) {
            Ok(rotations) => rotations,
            Err(_) => {
                return super::L2DenseRotationSolver::new().solve(relative_rotations, config);
            }
        };

        // Force R0 to be Identity (gauge fix)
        let r0 = global_rotations[&workspace.graph.views[0]];
        let r0_t = r0.transpose();

        if !r0_t.matrix().iter().all(|x| x.is_finite()) {
            return Err(EstimationError::Numerical(
                "R0 transpose contains NaN".to_string(),
            ));
        }

        for rotation in global_rotations.values_mut() {
            *rotation *= r0_t;
        }

        // Validate all rotations
        for (&view_id, rotation) in &global_rotations {
            let m = rotation.matrix();
            let det = m.determinant();
            let rtr = m.transpose() * m;
            let ortho_error = (rtr - Matrix3::identity()).norm();

            if (det - 1.0).abs() > 1e-6 || ortho_error > 1e-6 {
                return Err(EstimationError::Numerical(format!(
                    "invalid rotation for view {}: det={}, ortho_error={}",
                    view_id, det, ortho_error
                )));
            }
        }

        let residual = Self::compute_residual(&global_rotations, relative_rotations);

        Ok(RotationEstimate {
            rotations: global_rotations,
            iteration_count: 1,
            residual: crate::averaging::checked_angular_residual(residual)?,
        })
    }
}

// The shared solver tests are in mod.rs; only workspace reuse is tested here.
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn bits(rotations: &BTreeMap<ViewId, Rotation3<f64>>) -> Vec<u64> {
        rotations
            .values()
            .flat_map(|rotation| rotation.matrix().iter().map(|v| v.to_bits()))
            .collect()
    }

    /// A reused workspace gives a fresh solve's bits, whether its pattern is kept, changed by
    /// an exact cancellation or a new anchor, or its edge list changes.
    #[test]
    fn reused_workspace_matches_fresh_solves() {
        let about_z = |angle: f64| Rotation3::from_axis_angle(&Vector3::z_axis(), angle);
        let edges = [
            (2, 3, about_z(0.4)),
            (2, 3, about_z(-0.4)),
            (0, 1, Rotation3::new(Vector3::new(0.1, 0.2, -0.1))),
            (1, 2, Rotation3::new(Vector3::new(-0.2, 0.1, 0.3))),
            (3, 0, Rotation3::new(Vector3::new(0.05, -0.3, 0.2))),
            (0, 2, Rotation3::new(Vector3::new(0.3, 0.0, -0.2))),
        ];
        let mut workspace = L2SparseWorkspace::default();
        // Solves with these edge weights, fresh and through `workspace`, and returns where
        // the cached pattern lives and its size.
        let mut solve = |weights: &[f64]| {
            let rels: Vec<RelativeRotation> = edges
                .iter()
                .zip(weights)
                .map(|(&(i, j, rotation), &weight)| {
                    RelativeRotation::try_new(ViewId::new(i), ViewId::new(j), rotation, weight)
                        .unwrap()
                })
                .collect();
            let fresh = L2SparseWorkspace::default().solve_grounded(&rels).unwrap();
            let reused = workspace.solve_grounded(&rels).unwrap();
            assert_eq!(bits(&reused), bits(&fresh), "weights {weights:?}");
            let pattern = workspace.cholesky.as_ref().unwrap().pattern();
            (pattern.row_indices().as_ptr(), pattern.row_indices().len())
        };

        // Camera 0 anchors; the two (2, 3) edges' z-axis entries cancel to exactly zero.
        let (first, nnz) = solve(&[0.25, 0.25, 1.0, 1.0, 1.0, 1.0]);
        // Same anchor and zeros: the cached analysis is reused.
        assert_eq!(solve(&[0.25, 0.25, 0.9, 1.1, 1.0, 1.0]), (first, nnz));
        // Unequal parallel weights: the two canceled entries of the stored block enter the
        // pattern.
        let (second, more) = solve(&[0.25, 0.3, 0.9, 1.1, 1.0, 1.0]);
        assert_ne!(second, first);
        assert_eq!(more, nnz + 2);
        // Camera 2 becomes the best-connected camera and the anchor.
        let (third, _) = solve(&[0.25, 0.3, 0.5, 1.0, 0.5, 0.5]);
        assert_ne!(third, second);
        // Dropping an edge rebuilds the block graph.
        solve(&[0.25, 0.3, 1.0, 1.0, 1.0]);
        assert_eq!(workspace.graph.edges.len(), 5);
    }
}
