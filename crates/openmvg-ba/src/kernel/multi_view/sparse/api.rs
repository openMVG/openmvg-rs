// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public sparse Schur policies, results, and error types.

#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

use crate::numerics::refine::nonlinear::DEFAULT_INITIAL_DAMPING;
use crate::numerics::refine::{Loss, RefinementStatus, StepRejections};

use super::super::{CameraPoseVariables, MultiViewBundleAdjustmentError, ScaleGauge};

/// Soft budget used when selecting a sparse BA execution strategy.
///
/// The budget covers solver-specific workspace owned and reported by OpenMVG BA.
/// It is not a hard process-memory limit: allocator metadata, thread stacks,
/// symbolic-factor internals, and caller-owned problem storage are excluded.
/// When the selected iterative strategy was chosen because the direct estimate
/// does not fit an explicit [`Self::Bytes`] budget, the workspace also refuses to
/// build that direct storage later for an exact fallback, rank certificate, or
/// covariance, and reports
/// [`SparseBundleAdjustmentError::DirectStorageExceedsMemoryBudget`]. [`Self::Auto`]
/// only guides the choice of strategy: it keeps the rank certificate and refuses
/// no later storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparseWorkspaceMemoryBudget {
    /// Use OpenMVG BA's conservative, platform-independent default budget.
    #[default]
    Auto,
    /// Prefer strategies whose estimated solver-specific workspace fits this
    /// positive byte count.
    Bytes(usize),
    /// Do not constrain automatic selection by workspace memory.
    Unbounded,
}

impl SparseWorkspaceMemoryBudget {
    /// Conservative solver-specific workspace budget used by [`Self::Auto`].
    pub const AUTO_BYTES: usize = 1024 * 1024 * 1024;

    /// Effective byte limit, or `None` for [`Self::Unbounded`].
    pub const fn effective_bytes(self) -> Option<usize> {
        match self {
            Self::Auto => Some(Self::AUTO_BYTES),
            Self::Bytes(bytes) => Some(bytes),
            Self::Unbounded => None,
        }
    }
}

/// Concrete sparse camera-system strategy selected at workspace construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparseExecutionBackend {
    /// Materialized sparse Schur complement solved by exact LLT.
    DirectLlt,
    /// Matrix-free retained Schur-PCG with cached transformed cross rows.
    CachedIterativeSchur,
    /// Matrix-free Schur-PCG that recomputes transformed cross rows.
    RecomputedIterativeSchur,
}

/// Stable explanation for a construction-time sparse execution choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparseExecutionReason {
    /// The caller explicitly selected the backend.
    ExplicitBackend,
    /// Exact LLT was preferred and its estimate fits within four fifths of
    /// the effective budget.
    DirectFitsBudget,
    /// The caller explicitly removed the Auto memory constraint, so exact LLT
    /// remains the preferred primary backend.
    UnboundedPrefersDirect,
    /// Direct storage exceeded the budget, while cached retained Schur fit.
    DirectExceedsBudget,
    /// Full-precision cached rows exceeded the budget, while compact `f32`
    /// rows fit and avoid analytic recomputation during every PCG iteration.
    CompressedCachedRetainedFitsBudget,
    /// The calibrated workload crossed the measured retained-Schur speed
    /// frontier and its cached workspace still fits the soft budget.
    WorkloadPrefersCachedRetained,
    /// Cached retained storage also exceeded the budget, so calibrated BA
    /// selected analytic recomputation.
    CachedRetainedExceedsBudget,
    /// The lowest-memory backend supported by this problem was selected even
    /// though its estimate exceeds the soft budget.
    LowestMemoryBackend,
}

impl SparseExecutionReason {
    /// Short human-readable explanation suitable for reports and logs.
    #[cfg(feature = "unstable-internals")]
    pub const fn description(self) -> &'static str {
        match self {
            Self::ExplicitBackend => "the caller explicitly selected this backend",
            Self::DirectFitsBudget => "the exact direct workspace estimate fits the memory budget",
            Self::UnboundedPrefersDirect => {
                "the automatic memory policy is unbounded, so exact direct LLT is preferred"
            }
            Self::DirectExceedsBudget => {
                "the direct workspace estimate exceeds the budget and cached retained Schur fits"
            }
            Self::CompressedCachedRetainedFitsBudget => {
                "compact cached retained Schur fits the budget and avoids cross-row recomputation"
            }
            Self::WorkloadPrefersCachedRetained => {
                "the calibrated workload crosses the measured retained-Schur speed frontier"
            }
            Self::CachedRetainedExceedsBudget => {
                "cached retained Schur also exceeds the budget, so cross rows are recomputed"
            }
            Self::LowestMemoryBackend => {
                "the lowest-memory supported backend was selected despite exceeding the soft budget"
            }
        }
    }
}

/// Immutable receipt describing how one sparse workspace will execute.
///
/// Automatic selection is resolved exactly once during workspace construction.
/// Nonlinear iterations never oscillate between the listed primary backends;
/// the exact LLT fallback, taken when a PCG step fails its residual or curvature
/// check, remains a correctness mechanism of iterative Schur.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseExecutionDecision {
    pub(super) automatic: bool,
    pub(super) memory_budget: SparseWorkspaceMemoryBudget,
    pub(super) selected_backend: SparseExecutionBackend,
    pub(super) reason: SparseExecutionReason,
    pub(super) estimated_direct_bytes: Option<usize>,
    pub(super) estimated_cached_retained_bytes: Option<usize>,
    pub(super) estimated_recomputed_retained_bytes: Option<usize>,
    pub(super) selected_workspace_bytes: usize,
    pub(super) direct_assembly_threads: Option<usize>,
    pub(super) retained_threads: Option<usize>,
}

impl SparseExecutionDecision {
    /// Whether the primary backend was selected automatically.
    #[cfg(feature = "unstable-internals")]
    pub const fn is_automatic(self) -> bool {
        self.automatic
    }

    /// Requested soft workspace-memory policy.
    #[cfg(feature = "unstable-internals")]
    pub const fn memory_budget(self) -> SparseWorkspaceMemoryBudget {
        self.memory_budget
    }

    /// Selected primary sparse backend.
    pub const fn selected_backend(self) -> SparseExecutionBackend {
        self.selected_backend
    }

    /// Stable reason for the selection.
    #[cfg(feature = "unstable-internals")]
    pub const fn reason(self) -> SparseExecutionReason {
        self.reason
    }

    /// Conservative direct-workspace estimate used by Auto, when available.
    #[cfg(feature = "unstable-internals")]
    pub const fn estimated_direct_bytes(self) -> Option<usize> {
        self.estimated_direct_bytes
    }

    /// Cached retained-Schur estimate used by Auto, when available.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn estimated_cached_retained_bytes(self) -> Option<usize> {
        self.estimated_cached_retained_bytes
    }

    /// Recomputed retained-Schur estimate used by calibrated Auto, when
    /// available. Grouped radial self-calibration does not currently support
    /// this backend.
    #[cfg(feature = "unstable-internals")]
    pub const fn estimated_recomputed_retained_bytes(self) -> Option<usize> {
        self.estimated_recomputed_retained_bytes
    }

    /// Solver-specific workspace bytes actually reserved after construction.
    pub const fn selected_workspace_bytes(self) -> usize {
        self.selected_workspace_bytes
    }

    /// Resolved deterministic direct-assembly worker count.
    #[cfg(feature = "unstable-internals")]
    pub const fn direct_assembly_threads(self) -> Option<usize> {
        self.direct_assembly_threads
    }

    /// Resolved retained-Schur worker count.
    #[cfg(any(all(test, feature = "parallel"), feature = "unstable-internals"))]
    pub const fn retained_threads(self) -> Option<usize> {
        self.retained_threads
    }
}

/// Memory/speed policy for recovering eliminated point steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SparsePointBackSubstitution {
    /// Cache `V^-1 g` and `V^-1 W'` during Schur assembly. This avoids a
    /// second point linearization and is the default speed-oriented policy.
    #[default]
    Cached,
    /// Recompute point Jacobians after the camera solve. This uses less
    /// workspace memory at the cost of an additional linearization pass.
    Recompute,
}

/// Execution policy for forming an explicit direct Schur complement.
///
/// This policy affects numeric assembly only. The reduced camera matrix is
/// always factorized and solved by faer on the calling thread, because faer's
/// parallel factorization can change its bits with the worker count and
/// allocates during warm solves. A parallel assembly pool also runs the
/// Householder passes of the exact dense camera-rank certificate, whose result
/// is bitwise the same on any pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirectSchurAssembly {
    /// Assemble serially in point-major order.
    #[default]
    Serial,
    /// Use a fixed positive worker count for deterministic parallel assembly.
    ///
    /// Workers own disjoint reduced-matrix block columns. Point contributions
    /// within each column are accumulated in point order, so scheduling does
    /// not affect the assembled matrix.
    ///
    /// Builds without the `parallel` feature preserve the indexed arithmetic
    /// and execute it on the calling thread.
    #[cfg_attr(
        not(any(test, feature = "unstable-internals")),
        allow(dead_code, reason = "selected by the oracle and parallel benches")
    )]
    Parallel {
        /// Fixed positive worker count.
        threads: usize,
    },
    /// Resolve serial versus parallel assembly once from topology, available
    /// parallelism, and a bounded auxiliary-memory policy.
    ///
    /// The recommended cap is 16 workers for ordinary problems and 32 for
    /// large coupling or observation workloads. The active Rayon pool is a
    /// hard upper bound, and Auto will reduce the worker count or select the
    /// serial path rather than exceed 256 MiB of accounted direct-assembly
    /// buffers. Thread stacks and Rayon runtime state are excluded.
    ///
    /// Column-owned assembly is faster than serial even on one worker, so
    /// builds without the `parallel` feature apply the same size and memory
    /// rules with a single worker. Column-owned results do not depend on the
    /// worker count.
    Auto,
}

/// Camera-system solver used by the sparse bundle-adjustment workspace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SparseCameraSolver {
    /// Exact sparse Cholesky through faer. This remains the default.
    #[default]
    DirectLlt,
    /// Residual-bounded retained Schur-PCG with an exact LLT fallback.
    ///
    /// The solver retains compact point and camera blocks and avoids forming
    /// the Schur matrix on successful attempts. Schur-Jacobi-preconditioned
    /// PCG evaluates at most 600 iterations by default and accepts a true
    /// relative linear residual of at most 1% by default. Both controls are
    /// configurable through [`IterativeSchurOptions`]. A nonlinear rejection
    /// is retried by LM at higher damping; only a failed linear
    /// residual/curvature guard uses exact LLT. Direct numerical storage is
    /// allocated lazily for that fallback and final rank diagnostics, then reused. Call
    /// `SparseBundleAdjustmentWorkspace::try_reserve_direct_storage` before
    /// solving when allocation-free fallback and diagnostics are required. If
    /// unreserved lazy fallback storage cannot be prepared after an earlier
    /// accepted step, that linear attempt is rejected instead of returning a
    /// non-atomic error after caller state has changed, and
    /// [`SparseBundleAdjustmentReport::unavailable_direct_fallback_steps`]
    /// records it.
    IterativeSchur,
}

impl SparseCameraSolver {
    /// Whether this solver uses the retained matrix-free Schur operator.
    pub const fn uses_retained_schur(self) -> bool {
        !matches!(self, Self::DirectLlt)
    }

    /// Initial dimensionless LM damping selected for this solver family.
    ///
    /// Exact direct solves start from the low damping `1e-4`. Iterative
    /// Schur starts more conservatively so its finite inner-solve budget is
    /// not spent recovering from oversized first steps.
    pub const fn default_initial_damping(self) -> f64 {
        match self {
            Self::DirectLlt => DEFAULT_INITIAL_DAMPING,
            Self::IterativeSchur => 1e-1,
        }
    }
}

/// Retained Schur-operator policy used by
/// [`SparseCameraSolver::IterativeSchur`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RetainedSchurMatvec {
    /// Fused point-major operator. This is the compatibility default and is
    /// fastest for small visibility graphs.
    #[default]
    Fused,
    /// Recompute weighted analytic point/camera cross rows on every operator
    /// application instead of retaining `V^-1 W'` rows.
    ///
    /// This serial policy minimizes point-cache memory at the cost of repeated
    /// projection and Jacobian evaluation. Retained-cross precision is ignored,
    /// and self-calibration supports it through the same analytic radial
    /// linearization. [`Self::Auto`] never resolves to it, but the automatic
    /// calibrated execution plan selects it (or [`Self::RecomputedParallel`])
    /// when cached iterative storage exceeds the workspace memory budget.
    Recomputed,
    /// Parallel recomputation with one preallocated camera accumulator and
    /// analytic-Jacobian scratch arena per worker.
    ///
    /// Point ranges are observation-balanced and reduced in worker order. With
    /// the `parallel` feature, a persistent fixed-size Rayon pool executes the
    /// operator; otherwise threads are scoped per application. [`Self::Auto`]
    /// never resolves to it; the automatic calibrated execution plan selects it
    /// under the memory budget when parallel workers fit. `threads` must be
    /// positive. Parallel recomputation is currently calibrated-BA only.
    RecomputedParallel {
        /// Fixed positive worker count.
        threads: usize,
    },
    /// Lock-free point-major operator with reusable camera accumulators.
    ///
    /// Self-calibration uses topology-defined logical shards and reduces them
    /// in a fixed order, independent of the executor width. With the
    /// `parallel` feature, a fixed-size pool executes those shards; the same
    /// pool and preallocated worker-local blocks also parallelize calibrated
    /// retained assembly. It does not construct a second visibility index.
    /// `threads` must be positive.
    #[cfg_attr(
        not(any(test, feature = "unstable-internals")),
        allow(dead_code, reason = "selected by the oracle and parallel benches")
    )]
    PointMajorParallel {
        /// Fixed positive worker count.
        threads: usize,
    },
    /// Resolve once from topology and available parallelism at workspace
    /// construction. The gate keeps the fused path below 32,768
    /// observations and otherwise selects point-major execution. Eligible
    /// self-calibration problems retain the same numerical shards at one
    /// thread so their arithmetic is stable across executor widths. With the
    /// `parallel` feature, the active Rayon pool size and a conservative
    /// parallel-scratch budget bound the executor to at most 32 workers and
    /// 128 MiB. Without that feature, Auto remains fused; explicit parallel
    /// policies still use scoped threads.
    Auto,
}

/// Storage precision for retained `V^-1 W'` point-to-camera rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RetainedCrossPrecision {
    /// Preserve the full `f64` linearization. This is the default.
    #[default]
    F64,
    /// Store transformed cross rows as `f32` and widen them in operator use.
    ///
    /// Gradients, point/camera Hessians, Krylov vectors, reductions, damping,
    /// and nonlinear state remain `f64`. This option reduces the dominant
    /// retained-row cache but can change PCG convergence and the nonlinear
    /// trajectory slightly.
    F32,
}

/// Construction-time controls for residual-bounded iterative Schur-PCG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterativeSchurOptions {
    maximum_iterations: usize,
    relative_tolerance_bits: u64,
    direct_fallback: bool,
    matvec: RetainedSchurMatvec,
    cross_precision: RetainedCrossPrecision,
    exact_camera_rank_diagnostics: bool,
}

impl IterativeSchurOptions {
    /// Default finite guard used by the iterative camera solve.
    pub const DEFAULT_MAXIMUM_ITERATIONS: usize = 600;
    /// Default accepted true relative residual for the PCG camera solve.
    pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-2;

    /// Default options: 600 PCG iterations, the fused retained operator,
    /// full-precision cross rows, and exact final camera-rank diagnostics.
    pub const fn new() -> Self {
        Self {
            maximum_iterations: Self::DEFAULT_MAXIMUM_ITERATIONS,
            relative_tolerance_bits: Self::DEFAULT_RELATIVE_TOLERANCE.to_bits(),
            direct_fallback: true,
            matvec: RetainedSchurMatvec::Fused,
            cross_precision: RetainedCrossPrecision::F64,
            exact_camera_rank_diagnostics: true,
        }
    }

    /// Set the finite PCG iteration guard. Zero is rejected at construction.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn with_maximum_iterations(mut self, maximum_iterations: usize) -> Self {
        self.maximum_iterations = maximum_iterations;
        self
    }

    /// Set the accepted true relative PCG residual for ordinary LM steps.
    ///
    /// Values must be finite and strictly between zero and one. Invalid
    /// values are reported when the workspace is constructed. If a step first
    /// meets the outer step or cost tolerance with a looser residual, the
    /// solver confirms convergence with a true residual of at most `1e-6`,
    /// tightened to the square root of the outer relative-cost tolerance (or
    /// this configured tolerance when it is stricter).
    pub const fn with_relative_tolerance(mut self, relative_tolerance: f64) -> Self {
        self.relative_tolerance_bits = relative_tolerance.to_bits();
        self
    }

    /// Select whether a failed PCG guard materializes and solves the exact
    /// reduced system at the same damping.
    ///
    /// Disabling fallback turns that event into a rejected LM attempt, which
    /// increases damping before retrying PCG. This avoids unexpectedly large
    /// sparse-matrix allocations and factorizations on memory-constrained
    /// problems while preserving the last accepted nonlinear state.
    pub const fn with_direct_fallback(mut self, enabled: bool) -> Self {
        self.direct_fallback = enabled;
        self
    }

    /// Set the retained matrix-vector execution policy.
    #[cfg(any(test, feature = "unstable-internals", feature = "parallel"))]
    pub const fn with_matvec(mut self, matvec: RetainedSchurMatvec) -> Self {
        self.matvec = matvec;
        self
    }

    /// Select storage precision for retained transformed cross rows.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn with_cross_precision(mut self, cross_precision: RetainedCrossPrecision) -> Self {
        self.cross_precision = cross_precision;
        self
    }

    /// Select exact final camera-rank diagnostics.
    ///
    /// Small reduced systems use a dense, diagonally equilibrated Jacobi
    /// eigenspectrum; larger systems use sparse LLT probes. Disabling this
    /// avoids the associated diagnostic storage and work when every PCG
    /// attempt succeeds. Exact LLT is still prepared lazily if fallback is
    /// required, and point-block diagnostics remain available.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn with_exact_camera_rank_diagnostics(mut self, enabled: bool) -> Self {
        self.exact_camera_rank_diagnostics = enabled;
        self
    }

    /// Configured finite PCG iteration guard.
    pub const fn maximum_iterations(self) -> usize {
        self.maximum_iterations
    }

    /// Configured accepted true relative PCG residual.
    pub const fn relative_tolerance(self) -> f64 {
        f64::from_bits(self.relative_tolerance_bits)
    }

    /// Whether failed PCG guards use exact direct fallback.
    pub const fn direct_fallback(self) -> bool {
        self.direct_fallback
    }

    /// Configured retained matrix-vector execution policy.
    pub const fn matvec(self) -> RetainedSchurMatvec {
        self.matvec
    }

    /// Configured retained transformed-cross storage precision.
    pub const fn cross_precision(self) -> RetainedCrossPrecision {
        self.cross_precision
    }

    /// Whether the final reduced camera system receives an exact rank assessment.
    pub const fn exact_camera_rank_diagnostics(self) -> bool {
        self.exact_camera_rank_diagnostics
    }
}

impl Default for IterativeSchurOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Valid construction policies for a calibrated sparse workspace.
///
/// This is the compact alternative to backend-specific constructors. Each
/// variant contains only controls that the selected camera solver uses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseWorkspaceOptions {
    /// Select exact LLT, cached retained Schur, or calibrated cross-row
    /// recomputation once from topology and a soft workspace-memory budget.
    Auto {
        /// Solver-specific workspace-memory policy.
        memory_budget: SparseWorkspaceMemoryBudget,
        /// Relative PCG residual used when automatic planning selects Schur-PCG.
        iterative_relative_tolerance_bits: u64,
    },
    /// Exact sparse Cholesky with an explicit point recovery policy.
    Direct {
        /// Memory/speed policy used after the camera solve.
        point_back_substitution: SparsePointBackSubstitution,
    },
    /// Exact sparse Cholesky with an explicit direct-assembly policy.
    #[cfg_attr(
        not(any(test, feature = "unstable-internals")),
        allow(dead_code, reason = "selected by the oracle and parallel benches")
    )]
    DirectWithAssembly {
        /// Memory/speed policy used after the camera solve.
        point_back_substitution: SparsePointBackSubstitution,
        /// Serial, fixed-worker, or topology-selected Schur assembly.
        assembly: DirectSchurAssembly,
    },
    /// Residual-bounded retained Schur-PCG.
    IterativeSchur(IterativeSchurOptions),
}

impl SparseWorkspaceOptions {
    /// Automatically select a primary backend with the conservative default
    /// workspace-memory policy.
    #[cfg(feature = "unstable-internals")]
    pub const fn auto() -> Self {
        Self::Auto {
            memory_budget: SparseWorkspaceMemoryBudget::Auto,
            iterative_relative_tolerance_bits: IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE
                .to_bits(),
        }
    }

    /// Automatically select a primary backend with an explicit soft
    /// workspace-memory policy.
    #[cfg(feature = "unstable-internals")]
    pub const fn auto_with_memory_budget(memory_budget: SparseWorkspaceMemoryBudget) -> Self {
        Self::Auto {
            memory_budget,
            iterative_relative_tolerance_bits: IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE
                .to_bits(),
        }
    }

    /// Automatically select a backend with an explicit Schur-PCG residual bound.
    pub const fn auto_with_iterative_relative_tolerance(
        memory_budget: SparseWorkspaceMemoryBudget,
        relative_tolerance: f64,
    ) -> Self {
        Self::Auto {
            memory_budget,
            iterative_relative_tolerance_bits: relative_tolerance.to_bits(),
        }
    }

    /// Exact LLT with the speed-oriented cached point recovery policy.
    pub const fn direct() -> Self {
        Self::Direct {
            point_back_substitution: SparsePointBackSubstitution::Cached,
        }
    }

    /// Exact LLT with an explicit point recovery policy.
    #[cfg(feature = "unstable-internals")]
    pub const fn direct_with_point_back_substitution(
        point_back_substitution: SparsePointBackSubstitution,
    ) -> Self {
        Self::Direct {
            point_back_substitution,
        }
    }

    /// Exact LLT with an explicit direct-Schur assembly policy.
    #[cfg(any(test, feature = "unstable-internals", feature = "parallel"))]
    pub const fn direct_with_assembly(
        point_back_substitution: SparsePointBackSubstitution,
        assembly: DirectSchurAssembly,
    ) -> Self {
        Self::DirectWithAssembly {
            point_back_substitution,
            assembly,
        }
    }

    /// Exact LLT with cached point recovery and a fixed parallel assembler.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn direct_parallel(threads: usize) -> Self {
        Self::direct_with_assembly(
            SparsePointBackSubstitution::Cached,
            DirectSchurAssembly::Parallel { threads },
        )
    }

    /// Exact LLT with cached point recovery and topology-selected assembly.
    #[cfg(any(feature = "unstable-internals", feature = "parallel"))]
    pub const fn direct_auto() -> Self {
        Self::direct_with_assembly(
            SparsePointBackSubstitution::Cached,
            DirectSchurAssembly::Auto,
        )
    }

    /// Residual-bounded retained Schur-PCG with explicit controls.
    pub const fn iterative_schur(options: IterativeSchurOptions) -> Self {
        Self::IterativeSchur(options)
    }

    /// Low-memory calibrated Schur-PCG with explicit parallelism.
    ///
    /// Analytic point/camera cross rows and point back-substitution are
    /// recomputed instead of cached, and exact final camera-rank diagnostics
    /// are disabled. Exact LLT storage may still be allocated lazily if a PCG
    /// residual or curvature guard requires fallback. More workers use more
    /// preallocated camera-side scratch; `threads` must be positive.
    /// Successful purely iterative solves report
    /// [`SparseCameraRankAssessment::NotAssessed`]; commit-gated SfM workflows
    /// requiring affirmative rank evidence should configure
    /// [`IterativeSchurOptions`] explicitly with exact diagnostics enabled.
    ///
    /// This is equivalent to [`Self::iterative_schur`] with
    /// [`RetainedSchurMatvec::RecomputedParallel`] and exact camera-rank
    /// diagnostics disabled. Construct [`IterativeSchurOptions`] directly
    /// when the PCG iteration or residual controls also need tuning.
    #[cfg(feature = "unstable-internals")]
    pub const fn low_memory(threads: usize) -> Self {
        Self::IterativeSchur(
            IterativeSchurOptions::new()
                .with_matvec(RetainedSchurMatvec::RecomputedParallel { threads })
                .with_exact_camera_rank_diagnostics(false),
        )
    }

    /// Explicit camera-system solver requested by this construction policy.
    ///
    /// Returns `None` for [`Self::Auto`] because its concrete backend is
    /// topology-dependent; inspect the constructed workspace's execution
    /// decision instead.
    #[cfg(feature = "unstable-internals")]
    pub const fn camera_solver(self) -> Option<SparseCameraSolver> {
        match self {
            Self::Auto { .. } => None,
            Self::Direct { .. } | Self::DirectWithAssembly { .. } => {
                Some(SparseCameraSolver::DirectLlt)
            }
            Self::IterativeSchur(_) => Some(SparseCameraSolver::IterativeSchur),
        }
    }
}

impl Default for SparseWorkspaceOptions {
    fn default() -> Self {
        Self::direct()
    }
}

/// Linear camera-step evidence accumulated by one sparse BA solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseCameraSolveQuality {
    /// Every camera step passed to nonlinear acceptance used exact LLT, or no
    /// camera step was produced.
    Direct,
    /// Every used iterative Schur step met the internal residual bound.
    ResidualBoundedIterativeSchur,
    /// At least one iterative Schur attempt was replaced by exact LLT.
    IterativeSchurWithDirectFallback,
}

/// Coarse final rank evidence from dense spectral diagnostics at small scale
/// or sparse factorization at larger scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseCameraRankAssessment {
    /// Exact camera-rank diagnostics were explicitly disabled.
    NotAssessed,
    /// The fixed-gauge reduced camera system is numerically full rank.
    FullRank,
    /// A free-gauge system has only the expected independent scale null.
    OnlyExpectedScaleGauge,
    /// The reduced system is singular or indefinite after accounting for the
    /// selected gauge.
    RankDeficientOrIndefinite,
}

/// Final structural and numerical evidence from sparse BA.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseBundleAdjustmentDiagnostics {
    /// Dimension of the Schur-reduced camera-side system.
    ///
    /// For self-calibration this includes free pose and intrinsic parameters.
    /// [`Self::reduced_dimension`] returns the same value.
    pub pose_dimension: usize,
    /// Stored scalar entries in the lower triangle of that system.
    pub reduced_nonzero_count: usize,
    /// Exact reduced-system rank assessment.
    ///
    /// [`Self::reduced_rank_assessment`] returns the same value. It is a rank
    /// assessment, not a spectral condition number.
    pub pose_rank: SparseCameraRankAssessment,
    /// Number of eliminated point blocks that were numerically singular.
    pub weak_point_count: usize,
    /// Worst reciprocal condition estimate among eliminated point blocks.
    pub minimum_point_hessian_reciprocal_condition_number: f64,
    /// Smallest final undamped camera curvature used to construct the LM
    /// damping diagonal.
    ///
    /// Despite the field name, this is measured before point Schur
    /// elimination.
    pub minimum_reduced_diagonal: f64,
    /// Largest final undamped camera curvature used to construct the LM
    /// damping diagonal.
    pub maximum_reduced_diagonal: f64,
    /// Number of camera damping-diagonal entries raised by the selected
    /// damping-floor policy before point Schur elimination.
    pub damping_floor_clamp_count: usize,
}

impl SparseBundleAdjustmentDiagnostics {
    /// Dimension of the camera-side reduced system.
    ///
    /// This includes variable radial-intrinsic groups for self-calibration.
    #[cfg(feature = "unstable-internals")]
    pub const fn reduced_dimension(self) -> usize {
        self.pose_dimension
    }

    /// Exact rank assessment of the reduced system.
    #[cfg(feature = "unstable-internals")]
    pub const fn reduced_rank_assessment(self) -> SparseCameraRankAssessment {
        self.pose_rank
    }

    /// Whether no camera or point degeneracy was detected.
    ///
    /// This returns true for [`SparseCameraRankAssessment::NotAssessed`] when
    /// all eliminated point blocks are valid; inspect `camera_rank` when an
    /// affirmative exact camera-rank result is required.
    pub const fn is_well_constrained(self) -> bool {
        !matches!(
            self.pose_rank,
            SparseCameraRankAssessment::RankDeficientOrIndefinite
        ) && self.weak_point_count == 0
    }
}

/// Allocation-free aggregate evidence about nonlinear LM attempts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SparseNonlinearDiagnostics {
    /// Attempts rejected because Schur assembly was not finite or invertible.
    pub linearization_failures: usize,
    /// Attempts rejected because the camera-system solve failed its numerical guard.
    pub linear_solve_failures: usize,
    /// Attempts rejected because point back-substitution or candidate evaluation failed.
    pub candidate_failures: usize,
    /// Finite candidate steps rejected by the LM gain-ratio test.
    pub gain_rejections: usize,
    /// Largest Euclidean norm of any pose-local step proposed during the solve.
    pub maximum_pose_step_norm: f64,
    /// Largest Euclidean norm of any intrinsic-local step proposed during the solve.
    pub maximum_intrinsic_step_norm: f64,
    /// Largest root-mean-square point-local step proposed during the solve.
    pub maximum_point_step_rms: f64,
    /// Actual objective reduction from the last finite candidate trial.
    pub last_actual_reduction: Option<f64>,
    /// Predicted quadratic-model reduction from the last finite candidate trial.
    pub last_predicted_reduction: Option<f64>,
    /// Smallest finite gain ratio observed during candidate testing.
    pub minimum_gain_ratio: Option<f64>,
    /// Largest finite gain ratio observed during candidate testing.
    pub maximum_gain_ratio: Option<f64>,
}

impl SparseNonlinearDiagnostics {
    pub(super) fn record_reduced_step(
        &mut self,
        step: &[f64],
        pose_offsets: &[usize],
        pose_dimensions: &[usize],
        intrinsic_offsets: &[usize],
        intrinsic_dimensions: &[usize],
    ) {
        self.maximum_pose_step_norm = self.maximum_pose_step_norm.max(maximum_block_norm(
            step,
            pose_offsets,
            pose_dimensions,
        ));
        self.maximum_intrinsic_step_norm = self.maximum_intrinsic_step_norm.max(
            maximum_block_norm(step, intrinsic_offsets, intrinsic_dimensions),
        );
    }

    pub(super) fn record_candidate(
        &mut self,
        point_step_rms: f64,
        actual_reduction: f64,
        predicted_reduction: f64,
    ) {
        self.maximum_point_step_rms = self.maximum_point_step_rms.max(point_step_rms);
        self.last_actual_reduction = Some(actual_reduction);
        self.last_predicted_reduction = Some(predicted_reduction);
        let gain = actual_reduction / predicted_reduction;
        if gain.is_finite() {
            self.minimum_gain_ratio = Some(
                self.minimum_gain_ratio
                    .map_or(gain, |minimum| minimum.min(gain)),
            );
            self.maximum_gain_ratio = Some(
                self.maximum_gain_ratio
                    .map_or(gain, |maximum| maximum.max(gain)),
            );
        }
    }
}

fn maximum_block_norm(step: &[f64], offsets: &[usize], dimensions: &[usize]) -> f64 {
    offsets
        .iter()
        .copied()
        .zip(dimensions.iter().copied())
        .filter(|(_, dimension)| *dimension > 0)
        .map(|(offset, dimension)| {
            step[offset..offset + dimension]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max)
}

/// Outcome of an in-place sparse multi-view solve.
#[must_use = "inspect status and diagnostics before accepting the BA solution"]
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct SparseBundleAdjustmentReport {
    /// Loss used by the solve.
    pub loss: Loss,
    /// Explicit numerical solution status.
    pub status: RefinementStatus,
    /// Scale-gauge treatment used by the solve.
    pub scale_gauge: ScaleGauge,
    /// Finite positive world-coordinate scale used by the local parameterization.
    pub normalization_scale: f64,
    /// Camera-pose coordinates active in the solve.
    pub camera_pose_variables: CameraPoseVariables,
    /// Preserved baseline norm, or `None` for another gauge choice.
    pub fixed_baseline: Option<f64>,
    /// Preserved translation component, or `None` for another gauge choice.
    pub fixed_translation_component: Option<f64>,
    /// Number of camera poses.
    pub pose_count: usize,
    /// Number of refined 3D points.
    pub point_count: usize,
    /// Number of camera observations.
    pub observation_count: usize,
    /// Final measurements outside the camera model's unambiguous inverse
    /// domain.
    ///
    /// A nonzero count does not invalidate the forward BA objective: robust
    /// outliers may legitimately remain. Downstream triangulation should use
    /// lifecycle decisions or otherwise exclude these measurements.
    pub noninvertible_measurement_count: usize,
    /// LM iterations attempted.
    pub iteration_count: usize,
    /// Trial steps accepted.
    pub accepted_step_count: usize,
    /// Trial steps rejected.
    pub rejected_step_count: usize,
    /// The rejected steps by reason; their total is `rejected_step_count`.
    pub step_rejections: StepRejections,
    /// Initial robust objective.
    pub initial_cost: f64,
    /// Final robust objective.
    pub final_cost: f64,
    /// Initial unweighted residual root-mean-square.
    pub initial_rmse: f64,
    /// Final unweighted residual root-mean-square.
    pub final_rmse: f64,
    /// Maximum diagonal-curvature-scaled camera-or-point gradient from a fresh
    /// final-state linearization with zero LM damping.
    pub gradient_inf_norm: f64,
    /// Root-sum-square of camera and dimensionless point-step RMS values.
    pub step_norm: f64,
    /// Initial dimensionless Nielsen damping value used by this solve.
    pub initial_damping: f64,
    /// Final dimensionless Nielsen damping value.
    pub final_damping: f64,
    /// Gain ratio from the last evaluated finite trial step, if available.
    pub last_gain_ratio: Option<f64>,
    /// Aggregate causal evidence for nonlinear acceptance and rejection.
    pub nonlinear_diagnostics: SparseNonlinearDiagnostics,
    /// How camera-system steps passed to nonlinear acceptance were computed.
    /// Rejected iterative attempts remain visible through the iteration,
    /// residual, and unavailable-fallback evidence below.
    pub camera_solve_quality: SparseCameraSolveQuality,
    /// Retained-Schur camera steps passed to the nonlinear acceptance test.
    pub retained_camera_steps: usize,
    /// Iterative attempts replaced by exact LLT at the same damping.
    pub direct_fallback_steps: usize,
    /// Failed iterative attempts rejected so LM could increase damping instead
    /// of materializing an exact direct fallback.
    pub iterative_retry_steps: usize,
    /// Iterative attempts that failed the linear residual/curvature guard and
    /// were replaced by exact LLT.
    pub linear_guard_fallback_steps: usize,
    /// Iterative attempts whose exact LLT fallback storage was unavailable.
    ///
    /// These attempts are rejected and retried by LM without propagating an
    /// error after earlier accepted steps have mutated caller-owned state.
    /// They are not included in [`Self::direct_fallback_steps`]. Call
    /// `try_reserve_direct_storage` before solving to make storage failure an
    /// early, explicit error instead.
    pub unavailable_direct_fallback_steps: usize,
    /// Total retained inner iterations evaluated, including rejected attempts.
    pub retained_inner_iterations: usize,
    /// Largest finite true relative camera residual among completed attempts.
    ///
    pub maximum_retained_relative_residual: Option<f64>,
    /// True relative camera residual that confirmed outer step or cost
    /// convergence, if iterative Schur supplied the terminating step.
    pub convergence_confirmation_relative_residual: Option<f64>,
    /// Largest finite residual reported by a linear-guard rejection.
    pub maximum_rejected_iterative_relative_residual: Option<f64>,
    /// Final sparse rank and point-block evidence.
    pub diagnostics: SparseBundleAdjustmentDiagnostics,
    /// Immutable backend, memory, and worker decision captured by the
    /// topology-specific workspace.
    pub execution_decision: SparseExecutionDecision,
    /// Solver-specific storage reserved after the solve completes.
    ///
    /// Unlike the construction-time execution estimate, this includes storage
    /// allocated lazily for the direct fallback or exact diagnostics. Common
    /// pose, point, and symbolic-topology storage is excluded.
    pub solver_working_set_bytes: usize,
}

impl PartialEq for SparseBundleAdjustmentReport {
    fn eq(&self, other: &Self) -> bool {
        self.loss == other.loss
            && self.status == other.status
            && self.scale_gauge == other.scale_gauge
            && self.normalization_scale == other.normalization_scale
            && self.camera_pose_variables == other.camera_pose_variables
            && self.fixed_baseline == other.fixed_baseline
            && self.fixed_translation_component == other.fixed_translation_component
            && self.pose_count == other.pose_count
            && self.point_count == other.point_count
            && self.observation_count == other.observation_count
            && self.noninvertible_measurement_count == other.noninvertible_measurement_count
            && self.iteration_count == other.iteration_count
            && self.accepted_step_count == other.accepted_step_count
            && self.rejected_step_count == other.rejected_step_count
            && self.step_rejections == other.step_rejections
            && self.initial_cost == other.initial_cost
            && self.final_cost == other.final_cost
            && self.initial_rmse == other.initial_rmse
            && self.final_rmse == other.final_rmse
            && self.gradient_inf_norm == other.gradient_inf_norm
            && self.step_norm == other.step_norm
            && self.initial_damping == other.initial_damping
            && self.final_damping == other.final_damping
            && self.last_gain_ratio == other.last_gain_ratio
            && self.nonlinear_diagnostics == other.nonlinear_diagnostics
            && self.camera_solve_quality == other.camera_solve_quality
            && self.retained_camera_steps == other.retained_camera_steps
            && self.direct_fallback_steps == other.direct_fallback_steps
            && self.iterative_retry_steps == other.iterative_retry_steps
            && self.linear_guard_fallback_steps == other.linear_guard_fallback_steps
            && self.unavailable_direct_fallback_steps == other.unavailable_direct_fallback_steps
            && self.retained_inner_iterations == other.retained_inner_iterations
            && self.maximum_retained_relative_residual == other.maximum_retained_relative_residual
            && self.convergence_confirmation_relative_residual
                == other.convergence_confirmation_relative_residual
            && self.maximum_rejected_iterative_relative_residual
                == other.maximum_rejected_iterative_relative_residual
            && self.diagnostics == other.diagnostics
            && self.execution_decision == other.execution_decision
            && self.solver_working_set_bytes == other.solver_working_set_bytes
    }
}

impl SparseBundleAdjustmentReport {
    /// Retained iterative camera steps passed to nonlinear acceptance testing.
    #[cfg(feature = "unstable-internals")]
    pub const fn retained_camera_steps(&self) -> usize {
        self.retained_camera_steps
    }

    /// Total inner iterations for any retained Schur solver.
    #[cfg(feature = "unstable-internals")]
    pub const fn retained_inner_iterations(&self) -> usize {
        self.retained_inner_iterations
    }

    /// Largest finite residual reported by any retained Schur solver.
    #[cfg(feature = "unstable-internals")]
    pub const fn maximum_retained_relative_residual(&self) -> Option<f64> {
        self.maximum_retained_relative_residual
    }

    /// Iterative Schur camera steps passed to the nonlinear acceptance test.
    #[cfg(feature = "unstable-internals")]
    pub const fn iterative_camera_steps(&self) -> usize {
        match self.camera_solve_quality {
            SparseCameraSolveQuality::ResidualBoundedIterativeSchur
            | SparseCameraSolveQuality::IterativeSchurWithDirectFallback => {
                self.retained_camera_steps
            }
            SparseCameraSolveQuality::Direct => 0,
        }
    }

    /// Total retained inner iterations, including rejected attempts.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn iterative_inner_iterations(&self) -> usize {
        match self.camera_solve_quality {
            SparseCameraSolveQuality::ResidualBoundedIterativeSchur
            | SparseCameraSolveQuality::IterativeSchurWithDirectFallback => {
                self.retained_inner_iterations
            }
            SparseCameraSolveQuality::Direct if self.unavailable_direct_fallback_steps > 0 => {
                self.retained_inner_iterations
            }
            SparseCameraSolveQuality::Direct => 0,
        }
    }

    /// Largest finite true relative residual among iterative Schur attempts.
    #[cfg(feature = "unstable-internals")]
    pub const fn maximum_iterative_relative_residual(&self) -> Option<f64> {
        match self.camera_solve_quality {
            SparseCameraSolveQuality::ResidualBoundedIterativeSchur
            | SparseCameraSolveQuality::IterativeSchurWithDirectFallback => {
                self.maximum_retained_relative_residual
            }
            SparseCameraSolveQuality::Direct if self.unavailable_direct_fallback_steps > 0 => {
                self.maximum_retained_relative_residual
            }
            SparseCameraSolveQuality::Direct => None,
        }
    }

    /// True relative residual of the iterative step that confirmed outer
    /// convergence, if any.
    #[cfg(feature = "unstable-internals")]
    pub const fn iterative_convergence_confirmation_relative_residual(&self) -> Option<f64> {
        self.convergence_confirmation_relative_residual
    }
}

/// Failure to construct or use a sparse backend.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseBundleAdjustmentError {
    /// The ordinary multi-view problem is invalid.
    Problem(MultiViewBundleAdjustmentError),
    /// The solve topology differs from the graph used to construct the
    /// workspace.
    ///
    /// Self-calibration topology also includes intrinsic grouping and the
    /// fixed-block layout.
    TopologyMismatch,
    /// A translation-norm gauge requires the reference camera center to be held.
    GaugeReferenceCenterIsFree,
    /// A rotation-only pose cannot preserve a projection-translation component
    /// with its ordinary fixed-center rotation chart.
    RotationOnlyComponentGaugeUnsupported,
    /// Faer could not analyze the reduced sparsity pattern.
    SymbolicFactorization,
    /// Faer's reusable numerical factor or solve buffer could not be
    /// allocated.
    NumericalWorkspaceAllocation,
    /// An automatic workspace policy requested a zero-byte budget.
    WorkspaceMemoryBudgetMustBePositive,
    /// An exact fallback, rank certificate, or covariance needs the direct storage that the
    /// automatic plan excluded because its estimate exceeds the memory budget.
    DirectStorageExceedsMemoryBudget,
    /// A fixed parallel direct-Schur policy requested zero workers.
    DirectSchurThreadsMustBePositive,
    /// Parallel direct assembly requires cached point back-substitution rows.
    DirectSchurParallelRequiresCachedPoints,
    /// The compact direct-assembly index exceeded its 32-bit fields.
    DirectSchurIndexCapacityOverflow,
    /// The persistent direct-Schur worker pool could not be constructed.
    #[cfg_attr(
        not(any(feature = "parallel", feature = "unstable-internals")),
        allow(dead_code, reason = "reachable only through a parallel policy")
    )]
    DirectSchurThreadPoolBuild,
    /// A parallel retained Schur operator was requested with zero workers.
    RetainedSchurThreadsMustBePositive,
    /// The compact retained index exceeded its 32-bit point or cross-row fields.
    RetainedSchurIndexCapacityOverflow,
    /// The persistent retained-Schur worker pool could not be constructed.
    #[cfg_attr(
        not(any(feature = "parallel", feature = "unstable-internals")),
        allow(dead_code, reason = "reachable only through a parallel policy")
    )]
    RetainedSchurThreadPoolBuild,
    /// Recomputed cross rows were requested for a non-PCG camera solver.
    RecomputedSchurRequiresIterativeSolver,
    /// Parallel recomputed cross rows are not implemented for self-calibration.
    RecomputedSchurUnsupportedForSelfCalibration,
    /// A zero PCG iteration guard was requested.
    IterativeSchurIterationsMustBePositive,
    /// The requested PCG relative residual was not finite and in `(0, 1)`.
    IterativeSchurToleranceInvalid,
    /// The configured initial LM damping was not finite and positive.
    InitialDampingInvalid,
}

impl SparseBundleAdjustmentError {
    /// Whether retrying after reducing memory pressure or concurrency may
    /// succeed without changing the numerical problem.
    pub const fn is_resource_exhausted(self) -> bool {
        matches!(
            self,
            Self::NumericalWorkspaceAllocation
                | Self::DirectStorageExceedsMemoryBudget
                | Self::DirectSchurIndexCapacityOverflow
                | Self::DirectSchurThreadPoolBuild
                | Self::RetainedSchurIndexCapacityOverflow
                | Self::RetainedSchurThreadPoolBuild
        )
    }
}

impl fmt::Display for SparseBundleAdjustmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Problem(error) => error.fmt(formatter),
            Self::TopologyMismatch => formatter.write_str(
                "measurements, counts, gauge, intrinsic grouping, or fixed blocks do not match the sparse workspace topology",
            ),
            Self::GaugeReferenceCenterIsFree => formatter.write_str(
                "a translation-norm gauge requires the center of camera zero to be fixed",
            ),
            Self::RotationOnlyComponentGaugeUnsupported => formatter.write_str(
                "a Cartesian projection-translation gauge is unsupported for a rotation-only pose",
            ),
            Self::SymbolicFactorization => {
                formatter.write_str("faer could not analyze the reduced camera sparsity pattern")
            }
            Self::NumericalWorkspaceAllocation => {
                formatter.write_str("faer numerical workspace allocation failed")
            }
            Self::WorkspaceMemoryBudgetMustBePositive => {
                formatter.write_str("the sparse workspace memory budget must be positive")
            }
            Self::DirectStorageExceedsMemoryBudget => formatter.write_str(
                "exact direct storage exceeds the sparse workspace memory budget",
            ),
            Self::DirectSchurThreadsMustBePositive => {
                formatter.write_str("parallel direct Schur requires at least one worker thread")
            }
            Self::DirectSchurParallelRequiresCachedPoints => formatter.write_str(
                "parallel direct Schur requires cached point back-substitution rows",
            ),
            Self::DirectSchurIndexCapacityOverflow => formatter
                .write_str("the compact direct Schur index exceeds 32-bit capacity"),
            Self::DirectSchurThreadPoolBuild => {
                formatter.write_str("the direct Schur worker pool could not be constructed")
            }
            Self::RetainedSchurThreadsMustBePositive => {
                formatter.write_str("parallel retained Schur requires at least one worker thread")
            }
            Self::RetainedSchurIndexCapacityOverflow => {
                formatter.write_str("the compact retained Schur index exceeds 32-bit capacity")
            }
            Self::RetainedSchurThreadPoolBuild => {
                formatter.write_str("the retained Schur worker pool could not be constructed")
            }
            Self::RecomputedSchurRequiresIterativeSolver => formatter
                .write_str("recomputed Schur cross rows require the iterative PCG camera solver"),
            Self::RecomputedSchurUnsupportedForSelfCalibration => formatter.write_str(
                "parallel recomputed Schur cross rows are not implemented for self-calibration",
            ),
            Self::IterativeSchurIterationsMustBePositive => {
                formatter.write_str("iterative Schur requires at least one PCG iteration")
            }
            Self::IterativeSchurToleranceInvalid => formatter
                .write_str("iterative Schur relative tolerance must be finite and in (0, 1)"),
            Self::InitialDampingInvalid => {
                formatter.write_str("initial LM damping must be finite and positive")
            }
        }
    }
}

impl Error for SparseBundleAdjustmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Problem(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MultiViewBundleAdjustmentError> for SparseBundleAdjustmentError {
    fn from(error: MultiViewBundleAdjustmentError) -> Self {
        Self::Problem(error)
    }
}
