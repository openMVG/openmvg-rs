// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural parallel-rigidity preflight for direction graphs.
//!
//! Its teaching docs are on [`audit_parallel_rigidity`], where docs.rs shows them.

#[cfg(feature = "faer")]
use nalgebra::Matrix3;
use nalgebra::{DMatrix, Vector3};

use super::{GraphTopologyAudit, GraphTopologyError, audit_graph_topology};

/// One world-frame camera-to-camera direction constraint.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DirectionConstraint {
    /// Dense index of the first view, in `0..num_views`.
    pub view_i: usize,
    /// Dense index of the second view, in `0..num_views`; must differ from `view_i`.
    pub view_j: usize,
    /// World-frame direction from the center of `view_i` toward the center of
    /// `view_j`. Any finite non-zero length is accepted, and the sign does not
    /// matter: the constraint is the line through both centers.
    pub direction: Vector3<f64>,
}

impl DirectionConstraint {
    /// Create a direction constraint between dense views `view_i` and `view_j`.
    #[must_use]
    pub fn new(view_i: usize, view_j: usize, direction: Vector3<f64>) -> Self {
        Self {
            view_i,
            view_j,
            direction,
        }
    }
}

/// Numerical backend actually used by an audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParallelRigidityMethod {
    /// Dense SVD of the full rigidity matrix.
    DenseSvd,
    /// Sparse Cholesky factorization of the normal matrix, then inverse iteration for its
    /// smallest eigenvalue; needs the `faer` feature.
    SparseSubspaceIteration,
}

/// Generic-position rigidity of the candidate graph.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralRigidityStatus {
    /// The candidate block has fewer than two views, or its edges give fewer
    /// than the `3 * views - 4` rows (two per edge) a rigid graph needs.
    InsufficientConstraints,
    /// Every generic-position trial found the candidate graph flexible.
    NoRigidWitness {
        /// Backend used for the rank tests.
        method: ParallelRigidityMethod,
        /// Number of generic-position trials run (all flexible).
        trials: usize,
    },
    /// The generic-position trials disagreed or could not be decided within
    /// tolerance. Without the `faer` feature, every candidate beyond the dense bounds ends here.
    NumericallyIndeterminate {
        /// Backend used for the rank tests.
        method: ParallelRigidityMethod,
    },
    /// Every generic-position trial found the candidate graph parallel rigid.
    Rigid {
        /// Backend used for the rank tests.
        method: ParallelRigidityMethod,
    },
}

/// Bounded modes of a dense rigidity matrix built from generic directions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParallelRigiditySpectrum {
    /// Largest singular value.
    largest: f64,
    /// Lower bound on the expected global-scale mode after removing global translation.
    scale_mode_lower_bound: f64,
    /// Upper bound on the global-scale mode.
    scale_mode_upper_bound: f64,
    /// Lower bound on the smallest mode a parallel-rigid graph constrains.
    weakest_constrained_lower_bound: f64,
    /// Upper bound on the weakest constrained mode.
    weakest_constrained_upper_bound: f64,
}

/// Controls the bounded dense/sparse crossover without changing graph policy.
///
/// Candidates within both dense bounds are audited with dense SVD; larger ones
/// factor the sparse normal matrix, which needs the `faer` feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParallelRigidityConfig {
    anchor_view: usize,
    dense_max_views: usize,
    dense_max_coefficients: u128,
}

config_accessors!(ParallelRigidityConfig {
    /// Preferred translation gauge. If it is outside the selected block, the
    /// smallest candidate view is used without changing the returned IDs.
    copy anchor_view, with_anchor_view: usize;
    /// Largest candidate view count audited with dense SVD; must be positive.
    copy dense_max_views, with_dense_max_views: usize;
    /// Largest rank-matrix coefficient count audited with dense SVD; must be positive.
    copy dense_max_coefficients, with_dense_max_coefficients: u128;
});

impl Default for ParallelRigidityConfig {
    fn default() -> Self {
        Self {
            anchor_view: 0,
            dense_max_views: 256,
            dense_max_coefficients: 20_000_000,
        }
    }
}

/// Typed, non-mutating preflight. Candidate IDs and masks are expressed in the
/// original graph index spaces; only the internal rank matrix is compacted.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ParallelRigidityAudit {
    /// Topology audit of the full input graph, including the selected
    /// candidate block.
    pub topology: GraphTopologyAudit,
    /// Generic rigidity of the candidate block, independent of the measured
    /// directions.
    pub structural: StructuralRigidityStatus,
    /// Rows of the candidate rigidity matrix: two per candidate edge.
    pub constraint_row_count: usize,
    /// Rank a parallel-rigid candidate needs: `3 * views - 4` (translation and
    /// global scale removed).
    pub required_rank: usize,
    /// Original view indices of the candidate block, ascending.
    pub candidate_view_ids: Vec<usize>,
    /// Original edge indices of the candidate block, ascending.
    pub candidate_edge_indices: Vec<usize>,
    /// Original index of the gauge view used for the rank audit; `None` when
    /// the candidate block is empty.
    pub anchor_view: Option<usize>,
}

/// Error returned by [`audit_parallel_rigidity`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParallelRigidityError {
    /// The constraint endpoints do not form a valid dense graph.
    Topology(GraphTopologyError),
    /// A constraint direction is zero or non-finite.
    InvalidDirection {
        /// Index of the offending constraint in the input slice.
        edge: usize,
    },
    /// The configuration violates the stated rule.
    InvalidConfig(&'static str),
}

impl std::fmt::Display for ParallelRigidityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(error) => write!(formatter, "invalid direction graph: {error}"),
            Self::InvalidDirection { edge } => {
                write!(formatter, "edge {edge} has a zero or non-finite direction")
            }
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid parallel-rigidity config: {reason}")
            }
        }
    }
}

impl std::error::Error for ParallelRigidityError {}

impl From<GraphTopologyError> for ParallelRigidityError {
    fn from(value: GraphTopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Audits topology, then the generic structural rank of the candidate block.
///
/// The structural rank depends on the graph alone: the measured directions are
/// validated but do not enter the rank tests, which run on generic positions.
/// The operation is deterministic and never removes or reorders caller data.
/// Dense SVD audits small candidates; large ones factor the sparse normal matrix
/// once per trial, which needs the `faer` feature.
///
/// # How it works
///
/// Topology is audited first, so numerical work is restricted to
/// the largest deterministic vertex-biconnected candidate, while the returned
/// [`GraphTopologyAudit`] retains masks in the caller's original view and edge
/// index spaces. The rank tests run on generic positions, so a rigid result
/// concerns the graph alone. It does not prove positive baseline scales,
/// cheirality, or that measurements are outlier-free.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                  Parallel-rigidity preflight                         │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: num_views, edges [(view_i, view_j, direction)], config
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Validate and audit topology                                  │
/// └──────────────────────────────────────────────────────────────────────┘
/// Reject bad config and zero or non-finite directions. Run
/// `audit_graph_topology` and take its candidate core (V views, E edges).
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Count constraints                                            │
/// └──────────────────────────────────────────────────────────────────────┘
/// Two rows per edge. A rigid candidate needs rank 3V − 4. If V < 2 or
/// 2E < 3V − 4, stop with `InsufficientConstraints`.
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Rank test on generic positions, two seeds                    │
/// └──────────────────────────────────────────────────────────────────────┘
/// Draw positions cᵥ from a seed, set the directions to cⱼ − cᵢ, remove the
/// anchor view's translation, and test that only the global scale is free.
/// Dense SVD, or a sparse Cholesky test for large candidates.
///   │
///   ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 4: Combine the trials                                           │
/// └──────────────────────────────────────────────────────────────────────┘
/// Output: `ParallelRigidityAudit` with the topology, the status (`Rigid`,
/// `NoRigidWitness`, `NumericallyIndeterminate`, or `InsufficientConstraints`),
/// and the candidate's view and edge IDs in the input index spaces
/// ```
///
/// # Problem and method
///
/// Each edge says that the centers `cᵢ` and `cⱼ` lie on a line with a given
/// direction `d`. That is two linear equations, `P(d) (cⱼ − cᵢ) = 0` with
/// `P(d) = I − d dᵀ`, written in an orthonormal basis of the plane orthogonal to `d`.
/// Stacking the edges of the candidate gives a `2E × 3(V − 1)` matrix once the anchor view
/// (`ParallelRigidityConfig::anchor_view`, or the smallest candidate view when it is not in
/// the candidate) is held fixed. The global translation is then removed, and a
/// parallel-rigid graph leaves one free mode, the global scale, so the matrix
/// has rank `3V − 4`.
///
/// The measured directions are not used for the rank. The matrix is built from
/// random positions that a seeded generator draws for each view (`RunSeed` 1 and
/// 2) and treats as generic positions, so the result is a property of the graph. Each trial
/// passes when the smallest singular value is zero within
/// tolerance (the scale mode, `1e-8` of the largest singular value) and the next
/// one is not (`1e-10` of the largest, with an error bound on both). A trial that is neither a
/// clear pass nor a clear failure makes the result `NumericallyIndeterminate`.
/// `Rigid` needs both trials to pass; `NoRigidWitness` needs both to fail.
///
/// The dense path is used when the candidate has at most `dense_max_views`
/// views (default 256) and `6 E (V − 1)` is at most `dense_max_coefficients`
/// (default 20 000 000). Larger candidates form the normal matrix, which is sparse, and also fix
/// one coordinate of the view where the scale mode is largest, which removes the scale. The rest
/// is positive definite exactly when the trial is rigid, so a sparse Cholesky factorization that
/// fails shows a flexing. Otherwise inverse iteration on the factor finds the smallest
/// eigenvalue: it is rigid when that value settles clearly above `1e-10` of the matrix norm, and
/// flexible when it is at roundoff level. This path needs the `faer` feature; without it, those
/// candidates are `NumericallyIndeterminate`. The audit reports the backend as
/// `ParallelRigidityMethod`.
///
/// # Failure modes
///
/// - `ParallelRigidityError::InvalidDirection` for a zero or non-finite direction.
/// - `ParallelRigidityError::InvalidConfig` when a dense bound is zero.
/// - `ParallelRigidityError::Topology` for out-of-range ends, a self-loop, or an
///   unrepresentable size.
/// - `StructuralRigidityStatus::NoRigidWitness`: the candidate is flexible.
///   `NumericallyIndeterminate`: the trials disagree or are within tolerance, or a large
///   candidate was audited without the `faer` feature.
///
/// The audit removes and reorders nothing. A `Rigid` status says nothing about
/// baseline signs or outliers in the measured directions.
///
/// # Alternatives
///
/// [`audit_graph_topology`] alone checks only connectivity: a graph can be
/// biconnected and still flexible, which this audit detects.
pub fn audit_parallel_rigidity(
    num_views: usize,
    edges: &[DirectionConstraint],
    config: ParallelRigidityConfig,
) -> Result<ParallelRigidityAudit, ParallelRigidityError> {
    validate_config(config)?;
    for (index, edge) in edges.iter().enumerate() {
        let norm_squared = edge.direction.norm_squared();
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Err(ParallelRigidityError::InvalidDirection { edge: index });
        }
    }
    let endpoints: Vec<_> = edges
        .iter()
        .map(|edge| (edge.view_i, edge.view_j))
        .collect();
    let topology = audit_graph_topology(num_views, &endpoints)?;
    let candidate_view_ids: Vec<_> = topology
        .candidate_core_view_mask()
        .iter()
        .enumerate()
        .filter_map(|(view, &keep)| keep.then_some(view))
        .collect();
    let candidate_edge_indices: Vec<_> = topology
        .candidate_core_edge_mask()
        .iter()
        .enumerate()
        .filter_map(|(edge, &keep)| keep.then_some(edge))
        .collect();
    let mut remap = vec![usize::MAX; num_views];
    for (dense, &original) in candidate_view_ids.iter().enumerate() {
        remap[original] = dense;
    }
    let compact_edges: Vec<_> = candidate_edge_indices
        .iter()
        .map(|&index| DirectionConstraint {
            view_i: remap[edges[index].view_i],
            view_j: remap[edges[index].view_j],
            direction: edges[index].direction,
        })
        .collect();
    let constraint_rows = 2usize.saturating_mul(compact_edges.len());
    let required_rank = 3usize
        .saturating_mul(candidate_view_ids.len())
        .saturating_sub(4);
    let anchor_view = if candidate_view_ids.contains(&config.anchor_view) {
        Some(config.anchor_view)
    } else {
        candidate_view_ids.first().copied()
    };
    let compact_anchor = anchor_view.map(|view| remap[view]).unwrap_or(0);
    let method = select_method(compact_edges.len(), candidate_view_ids.len(), config);
    let structural = if candidate_view_ids.len() < 2
        || !has_minimum_constraints(compact_edges.len(), candidate_view_ids.len())
    {
        StructuralRigidityStatus::InsufficientConstraints
    } else {
        match structural_parallel_rigidity(
            &compact_edges,
            candidate_view_ids.len(),
            compact_anchor,
            method,
        ) {
            StructuralRank::NoWitness { trials } => {
                StructuralRigidityStatus::NoRigidWitness { method, trials }
            }
            StructuralRank::Indeterminate => {
                StructuralRigidityStatus::NumericallyIndeterminate { method }
            }
            StructuralRank::Rigid => StructuralRigidityStatus::Rigid { method },
        }
    };
    Ok(ParallelRigidityAudit {
        topology,
        structural,
        constraint_row_count: constraint_rows,
        required_rank,
        candidate_view_ids,
        candidate_edge_indices,
        anchor_view,
    })
}

fn validate_config(config: ParallelRigidityConfig) -> Result<(), ParallelRigidityError> {
    if config.dense_max_views == 0 || config.dense_max_coefficients == 0 {
        return Err(ParallelRigidityError::InvalidConfig(
            "dense crossover bounds must be positive",
        ));
    }
    Ok(())
}

fn select_method(
    edge_count: usize,
    num_views: usize,
    config: ParallelRigidityConfig,
) -> ParallelRigidityMethod {
    let coefficients = 6u128 * edge_count as u128 * num_views.saturating_sub(1) as u128;
    if num_views <= config.dense_max_views && coefficients <= config.dense_max_coefficients {
        ParallelRigidityMethod::DenseSvd
    } else {
        ParallelRigidityMethod::SparseSubspaceIteration
    }
}

#[inline]
fn has_minimum_constraints(edge_count: usize, num_views: usize) -> bool {
    num_views >= 2 && 2u128 * edge_count as u128 >= 3u128 * num_views as u128 - 4
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuralRank {
    Rigid,
    NoWitness { trials: usize },
    Indeterminate,
}

fn structural_parallel_rigidity(
    edges: &[DirectionConstraint],
    num_views: usize,
    anchor: usize,
    method: ParallelRigidityMethod,
) -> StructuralRank {
    const SEEDS: [openmvg_core::RunSeed; 2] = [
        openmvg_core::RunSeed::from_u64(1),
        openmvg_core::RunSeed::from_u64(2),
    ];
    if matches!(method, ParallelRigidityMethod::SparseSubspaceIteration) {
        return sparse_structural_rank(edges, num_views, anchor, SEEDS);
    }

    let mut passes = 0;
    let mut definite_failures = 0;
    for seed in SEEDS {
        let rank = dense_spectrum(edges, num_views, anchor, |_, edge| {
            (generic_position(edge.view_j, seed) - generic_position(edge.view_i, seed)).normalize()
        });
        match rank {
            Some(spectrum) => {
                let scale_tolerance = 1e-8 * spectrum.largest.max(f64::MIN_POSITIVE);
                let constrained_tolerance = 1e-10 * spectrum.largest.max(f64::MIN_POSITIVE);
                if spectrum.scale_mode_upper_bound <= scale_tolerance
                    && spectrum.weakest_constrained_lower_bound > constrained_tolerance
                {
                    passes += 1;
                } else if spectrum.scale_mode_lower_bound > scale_tolerance
                    || spectrum.weakest_constrained_upper_bound <= constrained_tolerance
                {
                    definite_failures += 1;
                } else {
                    return StructuralRank::Indeterminate;
                }
            }
            None => return StructuralRank::Indeterminate,
        }
    }
    match passes {
        0 if definite_failures == SEEDS.len() => StructuralRank::NoWitness {
            trials: SEEDS.len(),
        },
        0 => StructuralRank::Indeterminate,
        1 => StructuralRank::Indeterminate,
        _ => StructuralRank::Rigid,
    }
}

/// Combine the sparse trials of every seed: `Rigid` needs them all rigid, `NoWitness` all
/// flexible.
#[cfg(feature = "faer")]
fn sparse_structural_rank(
    edges: &[DirectionConstraint],
    num_views: usize,
    anchor: usize,
    seeds: [openmvg_core::RunSeed; 2],
) -> StructuralRank {
    let mut rigid = 0usize;
    let mut flexible = 0usize;
    for seed in seeds {
        match sparse_structural_trial(edges, num_views, anchor, seed) {
            SparseStructuralTrial::Rigid => rigid += 1,
            SparseStructuralTrial::Flexible => flexible += 1,
            SparseStructuralTrial::Indeterminate => return StructuralRank::Indeterminate,
        }
    }
    if rigid == seeds.len() {
        StructuralRank::Rigid
    } else if flexible == seeds.len() {
        StructuralRank::NoWitness {
            trials: seeds.len(),
        }
    } else {
        StructuralRank::Indeterminate
    }
}

/// Without `faer` there is no sparse factorization, so a large candidate stays undecided.
#[cfg(not(feature = "faer"))]
fn sparse_structural_rank(
    _edges: &[DirectionConstraint],
    _num_views: usize,
    _anchor: usize,
    _seeds: [openmvg_core::RunSeed; 2],
) -> StructuralRank {
    StructuralRank::Indeterminate
}

#[cfg(feature = "faer")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SparseStructuralTrial {
    Rigid,
    Flexible,
    Indeterminate,
}

/// Most steps of inverse iteration. Each step divides the error by the ratio of the two smallest
/// eigenvalues, so a few dozen settle unless the weakest modes nearly coincide.
#[cfg(feature = "faer")]
const INVERSE_ITERATIONS: usize = 64;

/// Structural sparse certificate for one generic trial.
///
/// The anchor view fixes the translation, and the coordinate where the known scale mode is largest
/// fixes the scale. The remaining normal matrix, `Σ (eⱼ − eᵢ)(eⱼ − eᵢ)ᵀ ⊗ (I − d dᵀ)` over the
/// edges, is then positive definite exactly when the trial is rigid. A sparse Cholesky
/// factorization that fails shows a null mode; otherwise inverse iteration on the factor finds
/// the smallest eigenvalue, whose Rayleigh quotient and residual decide.
#[cfg(feature = "faer")]
fn sparse_structural_trial(
    edges: &[DirectionConstraint],
    num_views: usize,
    anchor: usize,
    seed: openmvg_core::RunSeed,
) -> SparseStructuralTrial {
    use openmvg_ba::numerics::sparse::{SparseCholesky, SymmetricPattern};
    use std::collections::BTreeMap;

    let free_block = |view: usize| (view != anchor).then(|| view - usize::from(view > anchor));
    let anchor_position = generic_position(anchor, seed);
    let mut scale_mode = Vec::with_capacity(3 * num_views.saturating_sub(1));
    for view in (0..num_views).filter(|&view| view != anchor) {
        scale_mode.extend_from_slice((generic_position(view, seed) - anchor_position).as_slice());
    }
    let Some(pinned) = (0..scale_mode.len())
        .max_by(|&left, &right| scale_mode[left].abs().total_cmp(&scale_mode[right].abs()))
    else {
        return SparseStructuralTrial::Indeterminate;
    };
    let reduced = |coordinate: usize| {
        (coordinate != pinned).then(|| coordinate - usize::from(coordinate > pinned))
    };
    let dimension = scale_mode.len() - 1;

    // Lower triangle of the pinned normal matrix, column by column.
    let mut columns = vec![BTreeMap::<usize, f64>::new(); dimension];
    let mut degrees = vec![0usize; num_views];
    for edge in edges {
        let difference = generic_position(edge.view_j, seed) - generic_position(edge.view_i, seed);
        let Some(direction) = difference.try_normalize(1e-15) else {
            return SparseStructuralTrial::Indeterminate;
        };
        let projector = Matrix3::identity() - direction * direction.transpose();
        degrees[edge.view_i] += 1;
        degrees[edge.view_j] += 1;
        let (i, j) = (edge.view_i, edge.view_j);
        for (row_view, column_view, sign) in [(i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)]
        {
            let (Some(row_block), Some(column_block)) =
                (free_block(row_view), free_block(column_view))
            else {
                continue;
            };
            for row_axis in 0..3 {
                for column_axis in 0..3 {
                    let (Some(row), Some(column)) = (
                        reduced(3 * row_block + row_axis),
                        reduced(3 * column_block + column_axis),
                    ) else {
                        continue;
                    };
                    if row >= column {
                        *columns[column].entry(row).or_insert(0.0) +=
                            sign * projector[(row_axis, column_axis)];
                    }
                }
            }
        }
    }
    let mut column_pointers = Vec::with_capacity(dimension + 1);
    let (mut rows, mut values) = (Vec::new(), Vec::new());
    column_pointers.push(0);
    for column in &columns {
        for (&row, &value) in column {
            rows.push(row);
            values.push(value);
        }
        column_pointers.push(rows.len());
    }
    let Some(pattern) = SymmetricPattern::new(dimension, column_pointers, rows) else {
        return SparseStructuralTrial::Indeterminate;
    };
    // Supernodal, as rotation averaging factors a matrix of the same graph: on large view graphs
    // with long-range pairs the fill makes a simplicial factor about ten times slower.
    let Some(mut cholesky) = SparseCholesky::analyze(pattern) else {
        return SparseStructuralTrial::Indeterminate;
    };
    if !cholesky.reserve(1) {
        return SparseStructuralTrial::Indeterminate;
    }
    // Not numerically positive definite: the pinned matrix has a null mode, a flexing.
    if !cholesky.factor(&values) {
        return SparseStructuralTrial::Flexible;
    }

    // Each projector has norm 1, so the normal matrix has norm at most twice the largest degree.
    let spectral_upper_bound = 2.0 * degrees.into_iter().max().unwrap_or(0) as f64 + 1.0;
    let tolerance = 1e-10 * spectral_upper_bound;
    let roundoff = 32.0 * f64::EPSILON * dimension as f64 * spectral_upper_bound;
    let mut rng = openmvg_core::determinism::AlgorithmRng::new(
        seed.derive("geometry/parallel-rigidity/inverse-iteration/v1", &[]),
    );
    let mut vector: Vec<f64> = (0..dimension).map(|_| rng.symmetric_unit_f64()).collect();
    let mut applied = vec![0.0; dimension];
    let (mut eigenvalue, mut converged) = (f64::INFINITY, false);
    for _ in 0..INVERSE_ITERATIONS {
        if !cholesky.solve_in_place(&mut vector) || !normalize(&mut vector) {
            return SparseStructuralTrial::Indeterminate;
        }
        let pattern = cholesky.pattern();
        symmetric_lower_product(
            pattern.column_pointers(),
            pattern.row_indices(),
            &values,
            &vector,
            &mut applied,
        );
        let previous = std::mem::replace(&mut eigenvalue, dot(&vector, &applied));
        if (previous - eigenvalue).abs() <= 1e-12 * eigenvalue.abs() {
            converged = true;
            break;
        }
    }
    let residual = vector
        .iter()
        .zip(&applied)
        .map(|(value, applied)| (applied - eigenvalue * value).powi(2))
        .sum::<f64>()
        .sqrt();
    if !(eigenvalue.is_finite() && residual.is_finite()) {
        SparseStructuralTrial::Indeterminate
    } else if eigenvalue <= roundoff {
        // A Rayleigh quotient bounds the smallest eigenvalue from above: a direction the graph
        // does not resist, within roundoff.
        SparseStructuralTrial::Flexible
    } else if converged && eigenvalue - residual > tolerance {
        SparseStructuralTrial::Rigid
    } else {
        SparseStructuralTrial::Indeterminate
    }
}

/// `output = A input` for the symmetric matrix `A` whose lower triangle is stored by columns.
#[cfg(feature = "faer")]
fn symmetric_lower_product(
    column_pointers: &[usize],
    rows: &[usize],
    values: &[f64],
    input: &[f64],
    output: &mut [f64],
) {
    output.fill(0.0);
    for column in 0..input.len() {
        for entry in column_pointers[column]..column_pointers[column + 1] {
            let (row, value) = (rows[entry], values[entry]);
            output[row] += value * input[column];
            if row != column {
                output[column] += value * input[row];
            }
        }
    }
}

fn generic_position(view: usize, seed: openmvg_core::RunSeed) -> Vector3<f64> {
    let mut rng = openmvg_core::determinism::AlgorithmRng::new(seed.derive(
        "geometry/parallel-rigidity/generic-position/v1",
        &[view as u64],
    ));
    Vector3::new(
        rng.symmetric_unit_f64(),
        rng.symmetric_unit_f64(),
        rng.symmetric_unit_f64(),
    )
}

fn dense_spectrum<F>(
    edges: &[DirectionConstraint],
    num_views: usize,
    anchor: usize,
    direction: F,
) -> Option<ParallelRigiditySpectrum>
where
    F: Fn(usize, &DirectionConstraint) -> Vector3<f64>,
{
    let columns = 3 * (num_views - 1);
    let mut rigidity = DMatrix::<f64>::zeros(2 * edges.len(), columns);
    for (edge_index, edge) in edges.iter().enumerate() {
        for (basis_index, row) in orthogonal_basis(direction(edge_index, edge))
            .into_iter()
            .enumerate()
        {
            for (view, sign) in [(edge.view_i, -1.0), (edge.view_j, 1.0)] {
                if view == anchor {
                    continue;
                }
                let dense = if view < anchor { view } else { view - 1 };
                for axis in 0..3 {
                    rigidity[(2 * edge_index + basis_index, 3 * dense + axis)] += sign * row[axis];
                }
            }
        }
    }
    let matrix_dimension = rigidity.nrows().max(rigidity.ncols());
    let singular_values = rigidity.svd(false, false).singular_values;
    if singular_values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let largest = singular_values.iter().next().copied().unwrap_or(0.0);
    let (scale_mode, weakest_constrained_mode) = if singular_values.len() >= columns {
        (
            singular_values[columns - 1],
            singular_values[columns.saturating_sub(2)],
        )
    } else {
        (
            0.0,
            singular_values.iter().next_back().copied().unwrap_or(0.0),
        )
    };
    let uncertainty = 32.0 * f64::EPSILON * matrix_dimension as f64 * largest;
    Some(ParallelRigiditySpectrum {
        largest,
        scale_mode_lower_bound: (scale_mode - uncertainty).max(0.0),
        scale_mode_upper_bound: scale_mode + uncertainty,
        weakest_constrained_lower_bound: (weakest_constrained_mode - uncertainty).max(0.0),
        weakest_constrained_upper_bound: weakest_constrained_mode + uncertainty,
    })
}

fn orthogonal_basis(direction: Vector3<f64>) -> [Vector3<f64>; 2] {
    let direction = direction.normalize();
    let axis = if direction.x.abs() < 0.8 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let first = direction.cross(&axis).normalize();
    [first, direction.cross(&first).normalize()]
}

#[cfg(feature = "faer")]
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(feature = "faer")]
fn normalize(vector: &mut [f64]) -> bool {
    let norm = dot(vector, vector).sqrt();
    if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
        return false;
    }
    vector.iter_mut().for_each(|value| *value /= norm);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph(points: &[Vector3<f64>]) -> Vec<DirectionConstraint> {
        let mut edges = Vec::new();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                edges.push(DirectionConstraint::new(i, j, points[j] - points[i]));
            }
        }
        edges
    }

    #[test]
    fn exact_tetrahedron_is_certified() {
        let points = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            Vector3::new(0.0, 0.0, 4.0),
        ];
        let audit = audit_parallel_rigidity(
            points.len(),
            &complete_graph(&points),
            ParallelRigidityConfig::default(),
        )
        .unwrap();
        assert!(matches!(
            audit.structural,
            StructuralRigidityStatus::Rigid {
                method: ParallelRigidityMethod::DenseSvd
            }
        ));
        assert!(audit.topology.candidate_covers_all_views());
        assert_eq!(audit.candidate_view_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn zero_dense_bound_is_rejected() {
        let error = audit_parallel_rigidity(
            0,
            &[],
            ParallelRigidityConfig::default().with_dense_max_views(0),
        )
        .unwrap_err();
        assert!(matches!(error, ParallelRigidityError::InvalidConfig(_)));
    }

    #[test]
    fn structural_certificate_ignores_measured_collinearity() {
        let topology_points = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];
        let audit = audit_parallel_rigidity(
            topology_points.len(),
            &complete_graph(&topology_points),
            ParallelRigidityConfig::default(),
        )
        .unwrap();
        assert!(matches!(
            audit.structural,
            StructuralRigidityStatus::Rigid { .. }
        ));
    }

    #[test]
    fn articulation_lobe_is_masked_without_losing_original_ids() {
        let points = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            Vector3::new(0.0, 0.0, 4.0),
            Vector3::new(5.0, 1.0, 0.0),
            Vector3::new(4.0, 3.0, 1.0),
            Vector3::new(4.0, 1.0, 3.0),
        ];
        let mut edges = complete_graph(&points[..4]);
        for i in 3..points.len() {
            for j in (i + 1)..points.len() {
                edges.push(DirectionConstraint::new(i, j, points[j] - points[i]));
            }
        }
        let audit =
            audit_parallel_rigidity(points.len(), &edges, ParallelRigidityConfig::default())
                .unwrap();
        assert!(matches!(
            audit.structural,
            StructuralRigidityStatus::Rigid { .. }
        ));
        assert_eq!(audit.candidate_view_ids, vec![0, 1, 2, 3]);
        assert_eq!(audit.topology.articulation_views(), &[3]);
        assert_eq!(audit.topology.candidate_tail_views(), &[4, 5, 6]);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn dense_and_sparse_structural_certificates_agree() {
        let points: Vec<_> = (0..12)
            .map(|i| {
                let t = i as f64;
                Vector3::new(t.cos() * (1.0 + 0.1 * t), t.sin(), 0.07 * t * t)
            })
            .collect();
        let edges = complete_graph(&points);
        let dense =
            audit_parallel_rigidity(points.len(), &edges, ParallelRigidityConfig::default())
                .unwrap();
        let sparse = audit_parallel_rigidity(
            points.len(),
            &edges,
            ParallelRigidityConfig::default().with_dense_max_views(1),
        )
        .unwrap();
        assert!(matches!(
            dense.structural,
            StructuralRigidityStatus::Rigid {
                method: ParallelRigidityMethod::DenseSvd
            }
        ));
        assert!(matches!(
            sparse.structural,
            StructuralRigidityStatus::Rigid {
                method: ParallelRigidityMethod::SparseSubspaceIteration
            }
        ));
    }
    #[cfg(feature = "faer")]
    #[test]
    fn auto_sparse_certifies_structural_rigidity_above_five_hundred_views() {
        const NUM_VIEWS: usize = 520;
        let points: Vec<_> = (0..NUM_VIEWS)
            .map(|view| generic_position(view, 0xDEAD_BEEF_CAFE_BABEu64.into()))
            .collect();
        let mut edges = complete_graph(&points[..4]);
        for view in 4..NUM_VIEWS {
            for anchor in 0..3 {
                edges.push(DirectionConstraint::new(
                    anchor,
                    view,
                    points[view] - points[anchor],
                ));
            }
        }
        let audit =
            audit_parallel_rigidity(NUM_VIEWS, &edges, ParallelRigidityConfig::default()).unwrap();
        assert!(matches!(
            audit.structural,
            StructuralRigidityStatus::Rigid {
                method: ParallelRigidityMethod::SparseSubspaceIteration
            }
        ));
    }

    /// A strip of triangles, `view` to `view + 1` and `view + 2`; `braced` closes it into a ring.
    fn triangle_strip(num_views: usize, braced: bool) -> Vec<DirectionConstraint> {
        let mut edges = Vec::new();
        for view in 0..num_views {
            for step in [1, 2] {
                if braced || view + step < num_views {
                    let other = (view + step) % num_views;
                    edges.push(DirectionConstraint::new(view, other, Vector3::x()));
                }
            }
        }
        edges
    }

    #[cfg(feature = "faer")]
    fn dense_config() -> ParallelRigidityConfig {
        ParallelRigidityConfig::default()
            .with_dense_max_views(usize::MAX)
            .with_dense_max_coefficients(u128::MAX)
    }

    #[cfg(feature = "faer")]
    #[test]
    fn long_rigid_strip_is_certified_on_the_sparse_path() {
        // The weakest mode of a long strip is very soft, which an iteration without a
        // factorization could not settle; the sparse Cholesky test certifies it.
        const NUM_VIEWS: usize = 300;
        let edges = triangle_strip(NUM_VIEWS, false);
        let sparse =
            audit_parallel_rigidity(NUM_VIEWS, &edges, ParallelRigidityConfig::default()).unwrap();
        assert!(
            matches!(
                sparse.structural,
                StructuralRigidityStatus::Rigid {
                    method: ParallelRigidityMethod::SparseSubspaceIteration
                }
            ),
            "{:?}",
            sparse.structural
        );
        let dense = audit_parallel_rigidity(NUM_VIEWS, &edges, dense_config()).unwrap();
        assert!(matches!(
            dense.structural,
            StructuralRigidityStatus::Rigid { .. }
        ));
    }

    /// A clique of `clique` views with a path of `path` more views from view 0 back to view 1:
    /// enough constraints in total, but the path flexes.
    #[cfg(feature = "faer")]
    fn long_ear(clique: usize, path: usize) -> Vec<DirectionConstraint> {
        let mut edges = Vec::new();
        for i in 0..clique {
            for j in (i + 1)..clique {
                edges.push(DirectionConstraint::new(i, j, Vector3::x()));
            }
        }
        let ear: Vec<_> = std::iter::once(0)
            .chain(clique..clique + path)
            .chain(std::iter::once(1))
            .collect();
        for pair in ear.windows(2) {
            edges.push(DirectionConstraint::new(pair[0], pair[1], Vector3::x()));
        }
        edges
    }

    #[cfg(feature = "faer")]
    #[test]
    fn dense_and_sparse_paths_agree_on_rigid_and_flexible_graphs() {
        let graphs = [
            (12, triangle_strip(12, false), true),
            (60, triangle_strip(60, true), true),
            (18, long_ear(8, 10), false),
            (46, long_ear(16, 30), false),
        ];
        let sparse_config = ParallelRigidityConfig::default().with_dense_max_views(1);
        for (num_views, edges, rigid) in graphs {
            for config in [dense_config(), sparse_config] {
                let audit = audit_parallel_rigidity(num_views, &edges, config).unwrap();
                assert!(audit.topology.candidate_covers_all_views());
                let verdict_is_right = if rigid {
                    matches!(audit.structural, StructuralRigidityStatus::Rigid { .. })
                } else {
                    matches!(
                        audit.structural,
                        StructuralRigidityStatus::NoRigidWitness { .. }
                    )
                };
                assert!(
                    verdict_is_right,
                    "{num_views} views: {:?}",
                    audit.structural
                );
            }
        }
    }

    #[cfg(not(feature = "faer"))]
    #[test]
    fn without_faer_a_large_candidate_stays_undecided() {
        let edges = triangle_strip(12, false);
        let config = ParallelRigidityConfig::default().with_dense_max_views(1);
        let audit = audit_parallel_rigidity(12, &edges, config).unwrap();
        assert!(matches!(
            audit.structural,
            StructuralRigidityStatus::NumericallyIndeterminate { .. }
        ));
    }

    #[cfg(feature = "faer")]
    #[test]
    fn auto_sparse_rejects_flexible_long_ear_above_five_hundred_views() {
        const NUM_VIEWS: usize = 520;
        const CLIQUE_VIEWS: usize = 25;
        let points: Vec<_> = (0..NUM_VIEWS)
            .map(|view| generic_position(view, 0x1234_5678_9ABC_DEF0u64.into()))
            .collect();
        let mut edges = complete_graph(&points[..CLIQUE_VIEWS]);
        let mut path = Vec::with_capacity(NUM_VIEWS - CLIQUE_VIEWS + 2);
        path.push(0);
        path.extend(CLIQUE_VIEWS..NUM_VIEWS);
        path.push(1);
        for pair in path.windows(2) {
            edges.push(DirectionConstraint::new(
                pair[0],
                pair[1],
                points[pair[1]] - points[pair[0]],
            ));
        }
        let audit =
            audit_parallel_rigidity(NUM_VIEWS, &edges, ParallelRigidityConfig::default()).unwrap();
        assert!(audit.topology.candidate_covers_all_views());
        assert!(
            matches!(
                audit.structural,
                StructuralRigidityStatus::NoRigidWitness {
                    method: ParallelRigidityMethod::SparseSubspaceIteration,
                    ..
                }
            ),
            "{audit:?}"
        );
    }
}
