// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology-specific sparse storage and symbolic analysis.

mod calibrated;
mod pattern;
mod self_calibration;
#[cfg(test)]
mod tests;
mod topology;

use topology::{ReducedLayout, SelfCalibrationSignature, SparseProblemSignature};

use super::super::super::RelativePose;
use super::super::super::point_block::PackedPointBlock;
use super::super::schur::PointCrossCacheMut;
use super::super::self_calibration::RadialCamera;
use super::api::{
    RetainedCrossPrecision, SparseBundleAdjustmentError, SparseCameraSolver,
    SparseExecutionBackend, SparseExecutionDecision, SparsePointBackSubstitution,
    SparseWorkspaceMemoryBudget,
};
use super::direct_assembly::DirectSchurAssemblyWorkspace;
use super::execution_policy::direct_fits_budget;
use super::retained::RetainedSchurWorkspace;
use super::self_calibration_retained::SelfCalibrationRetainedWorkspace;
use crate::numerics::sparse::SparseCholesky;

pub(super) enum DirectAssemblyState {
    Serial,
    ColumnOwned {
        executor: Box<DirectSchurAssemblyWorkspace>,
        point_hessians: Vec<PackedPointBlock>,
    },
}

impl DirectAssemblyState {
    fn new(
        executor: Option<DirectSchurAssemblyWorkspace>,
        point_hessians: Vec<PackedPointBlock>,
    ) -> Self {
        match executor {
            Some(executor) => Self::ColumnOwned {
                executor: Box::new(executor),
                point_hessians,
            },
            None => Self::Serial,
        }
    }

    fn parallel_threads(&self) -> Option<usize> {
        match self {
            Self::Serial => None,
            Self::ColumnOwned { executor, .. } => executor.parallel_threads(),
        }
    }

    fn auxiliary_bytes(&self) -> usize {
        match self {
            Self::Serial => 0,
            Self::ColumnOwned {
                executor,
                point_hessians,
            } => {
                executor.auxiliary_bytes()
                    + point_hessians.capacity() * std::mem::size_of::<PackedPointBlock>()
            }
        }
    }
}

pub(super) enum CalibratedBackend {
    Direct(DirectAssemblyState),
    Retained(Box<RetainedSchurWorkspace>),
}

impl CalibratedBackend {
    const fn camera_solver(&self) -> SparseCameraSolver {
        match self {
            Self::Direct(_) => SparseCameraSolver::DirectLlt,
            Self::Retained(retained) => retained.camera_solver(),
        }
    }

    /// Column-owned executor whose worker pool the dense rank certificate borrows.
    pub(super) fn direct_executor(&self) -> Option<&DirectSchurAssemblyWorkspace> {
        match self {
            Self::Direct(DirectAssemblyState::ColumnOwned { executor, .. }) => Some(executor),
            Self::Direct(DirectAssemblyState::Serial) | Self::Retained(_) => None,
        }
    }
}

pub(super) enum SelfCalibrationBackend {
    Direct(DirectAssemblyState),
    Iterative(Box<SelfCalibrationRetainedWorkspace>),
}

impl SelfCalibrationBackend {
    const fn camera_solver(&self) -> SparseCameraSolver {
        match self {
            Self::Direct(_) => SparseCameraSolver::DirectLlt,
            Self::Iterative(_) => SparseCameraSolver::IterativeSchur,
        }
    }

    #[cfg(feature = "parallel")]
    pub(super) fn install_parallel<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> Option<R> {
        match self {
            Self::Direct(DirectAssemblyState::ColumnOwned { executor, .. }) => {
                Some(executor.install_parallel(operation))
            }
            Self::Iterative(retained) => retained.install_parallel(operation),
            Self::Direct(DirectAssemblyState::Serial) => None,
        }
    }
}

/// Dense, diagonally equilibrated spectral diagnostics are both more accurate
/// and more tolerant of roundoff than a bare sparse LLT probe at small scale.
/// Larger systems retain the O(nnz) factorization-based assessment.
pub(super) const DENSE_RANK_DIAGNOSTIC_MAX_DIMENSION: usize = 2048;

/// Dense rank-certificate storage: the `dimension x dimension` matrix followed
/// by two `dimension`-long Householder product buffers.
pub(super) const fn dense_rank_diagnostic_len(dimension: usize) -> usize {
    dimension.saturating_add(2).saturating_mul(dimension)
}

/// Reusable sparse storage for one exact visibility graph and gauge.
///
/// Direct workspaces and retained workspaces with exact rank diagnostics build
/// the reduced sparsity pattern and faer's symbolic AMD/LLT analysis eagerly.
/// Retained-only iterative workspaces defer the pattern, symbolic factor, and
/// numerical storage until an exact fallback, unless
/// [`Self::try_reserve_direct_storage`] is called first.
pub(super) struct SparseWorkspaceCore {
    pub(super) execution_decision: SparseExecutionDecision,
    pub(super) signature: SparseProblemSignature,
    pub(super) point_back_substitution: SparsePointBackSubstitution,
    pub(super) layout: ReducedLayout,
    pub(super) anchored_six_camera_layout: bool,
    pub(super) candidate_poses: Vec<RelativePose>,
    pub(super) candidate_points: Vec<[f64; 3]>,
    pub(super) candidate_point_metrics: Vec<[f64; 3]>,
    /// The reduced camera pattern and its symbolic Cholesky analysis, present once the direct
    /// pattern is built; `values` follows its entries.
    pub(super) cholesky: Option<SparseCholesky>,
    reduced_nonzero_count: usize,
    pub(super) values: Vec<f64>,
    pub(super) scatter: SparseScatter,
    pub(super) exact_camera_rank_diagnostics: bool,
    pub(super) iterative_direct_fallback: bool,
    pub(super) initial_damping: f64,
    pub(super) dense_rank_diagnostic: Vec<f64>,
    #[cfg(test)]
    pub(super) fail_next_direct_storage_for_test: bool,
    pub(super) gradient: Vec<f64>,
    pub(super) full_gradient: Vec<f64>,
    pub(super) step: Vec<f64>,
    pub(super) diagonal: Vec<f64>,
    pub(super) cross: Vec<f64>,
    pub(super) active_parameters: Vec<usize>,
    pub(super) parameter_marks: Vec<bool>,
    pub(super) point_cache_valid: Vec<bool>,
    pub(super) point_inverse_gradients: Vec<[f64; 3]>,
    pub(super) point_damping_diagonals: Vec<[f64; 3]>,
    pub(super) point_cross_offsets: Vec<usize>,
    pub(super) point_parameters: Vec<u32>,
    pub(super) point_inverse_cross_values: PointCrossCache,
}

impl SparseWorkspaceCore {
    #[cfg(feature = "unstable-internals")]
    pub(super) const fn camera_count(&self) -> usize {
        self.signature.camera_count()
    }

    #[cfg(feature = "unstable-internals")]
    pub(super) const fn point_count(&self) -> usize {
        self.signature.point_count()
    }

    pub(super) const fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    pub(super) fn observation_cameras(&self) -> &[u32] {
        self.signature.observation_cameras()
    }

    pub(super) fn point_offsets(&self) -> &[usize] {
        self.signature.point_offsets()
    }

    pub(super) fn fixed_points(&self) -> &[bool] {
        self.signature.fixed_points()
    }

    pub(super) fn camera_offsets(&self) -> &[usize] {
        self.layout.camera_offsets()
    }

    pub(super) fn camera_dimensions(&self) -> &[usize] {
        self.layout.camera_dimensions()
    }

    pub(super) fn intrinsic_offsets(&self) -> &[usize] {
        self.layout.intrinsic_offsets()
    }

    pub(super) fn intrinsic_dimensions(&self) -> &[usize] {
        self.layout.intrinsic_dimensions()
    }

    pub(super) fn point_linearization_cache_bytes(&self) -> usize {
        self.point_cache_valid.capacity() * std::mem::size_of::<bool>()
            + self.point_inverse_gradients.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.point_damping_diagonals.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.point_cross_offsets.capacity() * std::mem::size_of::<usize>()
            + self.point_parameters.capacity() * std::mem::size_of::<u32>()
            + self.point_inverse_cross_values.capacity_bytes()
    }

    pub(super) fn direct_numeric_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<f64>()
            + self.dense_rank_diagnostic.capacity() * std::mem::size_of::<f64>()
            + self
                .cholesky
                .as_ref()
                .map_or(0, SparseCholesky::reserved_bytes)
    }

    #[cfg(feature = "unstable-internals")]
    pub(super) fn direct_pattern_bytes(&self) -> usize {
        self.cholesky.as_ref().map_or(0, |cholesky| {
            std::mem::size_of_val(cholesky.pattern().column_pointers())
                + std::mem::size_of_val(cholesky.pattern().row_indices())
        }) + self.scatter.capacity_bytes()
    }

    /// Refuse to build direct storage the automatic plan excluded under an explicit byte budget.
    ///
    /// Materializing it later for an exact fallback, rank certificate, or covariance would
    /// allocate exactly what that budget ruled out. The automatic budget only guides the choice
    /// of strategy, so it refuses nothing here.
    pub(super) fn check_lazy_direct_storage(&self) -> Result<(), SparseBundleAdjustmentError> {
        let decision = &self.execution_decision;
        let excluded = decision.automatic
            && decision.selected_backend != SparseExecutionBackend::DirectLlt
            && match (decision.memory_budget, decision.estimated_direct_bytes) {
                (SparseWorkspaceMemoryBudget::Bytes(limit), Some(estimate)) => {
                    !direct_fits_budget(estimate, limit)
                }
                _ => false,
            };
        if excluded {
            return Err(SparseBundleAdjustmentError::DirectStorageExceedsMemoryBudget);
        }
        Ok(())
    }

    /// Reserve the factor, its scratch, and the reduced values of the analyzed pattern. The
    /// factor storage comes first, so a failed allocation leaves the values untouched.
    pub(super) fn ensure_numeric_storage(&mut self) -> Result<(), SparseBundleAdjustmentError> {
        let value_count = reserve_direct_numeric(&mut self.cholesky)?;
        if self.values.len() != value_count {
            self.values = vec![0.0; value_count];
        }
        Ok(())
    }
}

/// Dimension and entry count of `cholesky`'s pattern, zero without one.
pub(super) fn reduced_pattern_size(cholesky: Option<&SparseCholesky>) -> (usize, usize) {
    cholesky.map_or((0, 0), |cholesky| {
        (
            cholesky.pattern().dimension(),
            cholesky.pattern().value_count(),
        )
    })
}

/// Reserve `cholesky`'s factor and scratch for one right-hand side; returns the entry count of
/// its pattern, the length of the reduced values.
pub(super) fn reserve_direct_numeric(
    cholesky: &mut Option<SparseCholesky>,
) -> Result<usize, SparseBundleAdjustmentError> {
    let cholesky = cholesky
        .as_mut()
        .ok_or(SparseBundleAdjustmentError::SymbolicFactorization)?;
    if !cholesky.reserve(1) {
        return Err(SparseBundleAdjustmentError::NumericalWorkspaceAllocation);
    }
    Ok(cholesky.pattern().value_count())
}

/// Reusable calibrated sparse storage for one exact visibility graph and gauge.
pub struct SparseBundleAdjustmentWorkspace {
    pub(super) core: SparseWorkspaceCore,
    pub(super) backend: CalibratedBackend,
}

pub(super) enum PointCrossCache {
    F64(Vec<[f64; 3]>),
    F32(Vec<[f32; 3]>),
    Recomputed { len: usize },
}

impl PointCrossCache {
    fn new(count: usize, precision: RetainedCrossPrecision, retain_rows: bool) -> Self {
        if !retain_rows {
            return Self::Recomputed { len: count };
        }
        match precision {
            RetainedCrossPrecision::F64 => Self::F64(vec![[0.0; 3]; count]),
            RetainedCrossPrecision::F32 => Self::F32(vec![[0.0; 3]; count]),
        }
    }

    #[inline(always)]
    pub(super) fn get(&self, index: usize) -> [f64; 3] {
        match self {
            Self::F64(values) => values[index],
            Self::F32(values) => values[index].map(f64::from),
            // Recomputed storage is consumed by a separate analytic operator.
            #[allow(clippy::unreachable)]
            Self::Recomputed { .. } => {
                unreachable!("recomputed Schur does not retain transformed cross rows")
            }
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Recomputed { len } => *len,
        }
    }

    pub(super) fn capacity_bytes(&self) -> usize {
        match self {
            Self::F64(values) => values.capacity() * std::mem::size_of::<[f64; 3]>(),
            Self::F32(values) => values.capacity() * std::mem::size_of::<[f32; 3]>(),
            Self::Recomputed { .. } => 0,
        }
    }

    pub(super) fn as_mut(&mut self) -> PointCrossCacheMut<'_> {
        match self {
            Self::F64(values) => PointCrossCacheMut::F64(values),
            Self::F32(values) => PointCrossCacheMut::F32(values),
            Self::Recomputed { len } => PointCrossCacheMut::Recomputed { len: *len },
        }
    }

    pub(super) fn f64_rows(&self) -> Option<&[[f64; 3]]> {
        match self {
            Self::F64(values) => Some(values),
            Self::F32(_) | Self::Recomputed { .. } => None,
        }
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(super) fn as_f64(&self) -> &[[f64; 3]] {
        match self {
            Self::F64(values) => values,
            Self::F32(_) => unreachable!("an f64 cache view was requested from f32 storage"),
            Self::Recomputed { .. } => {
                unreachable!("recomputed Schur does not retain transformed cross rows")
            }
        }
    }

    #[cfg(feature = "unstable-internals")]
    fn precision(&self) -> Option<RetainedCrossPrecision> {
        match self {
            Self::F64(values) if !values.is_empty() => Some(RetainedCrossPrecision::F64),
            Self::F32(values) if !values.is_empty() => Some(RetainedCrossPrecision::F32),
            Self::F64(_) | Self::F32(_) | Self::Recomputed { .. } => None,
        }
    }

    #[cfg(any(test, feature = "unstable-internals"))]
    fn retains_rows(&self) -> bool {
        match self {
            Self::F64(values) => !values.is_empty(),
            Self::F32(values) => !values.is_empty(),
            Self::Recomputed { .. } => false,
        }
    }
}

/// Exact-topology sparse workspace for radial-camera self-calibration.
///
/// The default camera solve is sparse direct LLT. Opt-in retained Schur-PCG
/// stores the exact pose-to-intrinsic base edges and either retains or
/// analytically recomputes point-elimination rows, so per-view and
/// shared-intrinsic systems remain matrix-free without changing their model.
///
/// The workspace owns no caller state, but it is specific to the observation
/// camera ordering, point offsets, gauge, intrinsic grouping, and fixed-block
/// masks supplied at construction. Measurement coordinates, parameter values,
/// and prior values may change between solves. Point/operator scratch is
/// allocated eagerly. Exact LLT storage is eager for direct or exact-rank
/// workspaces and otherwise lazy; call [`Self::try_reserve_direct_storage`]
/// when the direct fallback must remain allocation-free.
pub struct SparseSelfCalibrationWorkspace {
    pub(super) sparse: SparseWorkspaceCore,
    signature: SelfCalibrationSignature,
    pub(super) backend: SelfCalibrationBackend,
    pub(super) candidate_cameras: Vec<RadialCamera>,
}

/// Block-aware numeric scatter metadata for a scalar CSC pattern.
///
/// One start position is retained per block edge and scalar column. Numeric
/// assembly therefore searches only the much smaller block adjacency list;
/// the row-local position is an addition rather than a second scalar search.
pub(super) use pattern::SparseScatter;
