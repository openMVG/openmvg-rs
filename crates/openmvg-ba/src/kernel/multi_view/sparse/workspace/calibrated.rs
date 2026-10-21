// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated sparse-workspace construction and direct-storage lifecycle.

use super::pattern::{
    analyze_reduced_pattern, point_cross_offsets, reduced_block_columns, reduced_nonzero_count,
    reduced_pattern,
};
use super::topology::{ReducedLayout, SparseProblemSignature};
use super::{
    CalibratedBackend, DENSE_RANK_DIAGNOSTIC_MAX_DIMENSION, DirectAssemblyState, PointCrossCache,
    SparseBundleAdjustmentWorkspace, SparseScatter, SparseWorkspaceCore, dense_rank_diagnostic_len,
    reduced_pattern_size, reserve_direct_numeric,
};
use crate::kernel::RelativePose;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::sparse::api::{
    DirectSchurAssembly, IterativeSchurOptions, RetainedCrossPrecision, RetainedSchurMatvec,
    SparseBundleAdjustmentError, SparseCameraSolver, SparseExecutionBackend,
    SparsePointBackSubstitution, SparseWorkspaceMemoryBudget, SparseWorkspaceOptions,
};
use crate::kernel::multi_view::sparse::direct_assembly::{
    DirectSchurAssemblyWorkspace, auto_direct_assembly_threads,
    estimated_calibrated_auxiliary_bytes,
};
use crate::kernel::multi_view::sparse::execution_policy::{
    AutomaticExecutionPlan, ExecutionEstimates, estimated_direct_bytes,
    estimated_point_cache_bytes, explicit_decision, finish_automatic_decision, select_calibrated,
};
use crate::kernel::multi_view::sparse::retained::{
    RetainedSchurWorkspace, automatic_recomputed_matvec, estimated_calibrated_retained_bytes,
};
use crate::kernel::multi_view::{
    CameraPoseVariables, FixedVariableBlocks, MultiViewBundleAdjustmentError,
    MultiViewMeasurements, ScaleGauge, validation,
};

#[derive(Clone, Copy)]
struct CalibratedWorkspacePolicy {
    auto_memory_budget: Option<SparseWorkspaceMemoryBudget>,
    point_back_substitution: SparsePointBackSubstitution,
    camera_solver: SparseCameraSolver,
    retained_matvec: RetainedSchurMatvec,
    retained_cross_precision: RetainedCrossPrecision,
    iterative_maximum_iterations: usize,
    iterative_relative_tolerance: f64,
    iterative_direct_fallback: bool,
    exact_camera_rank_diagnostics: bool,
    direct_assembly: DirectSchurAssembly,
}

impl CalibratedWorkspacePolicy {
    fn from_options(options: SparseWorkspaceOptions) -> Self {
        let mut policy = Self {
            auto_memory_budget: None,
            point_back_substitution: SparsePointBackSubstitution::Cached,
            camera_solver: SparseCameraSolver::DirectLlt,
            retained_matvec: RetainedSchurMatvec::Fused,
            retained_cross_precision: RetainedCrossPrecision::F64,
            iterative_maximum_iterations: IterativeSchurOptions::DEFAULT_MAXIMUM_ITERATIONS,
            iterative_relative_tolerance: IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE,
            iterative_direct_fallback: true,
            exact_camera_rank_diagnostics: true,
            direct_assembly: DirectSchurAssembly::Serial,
        };
        match options {
            SparseWorkspaceOptions::Auto {
                memory_budget,
                iterative_relative_tolerance_bits,
            } => {
                policy.auto_memory_budget = Some(memory_budget);
                policy.iterative_relative_tolerance =
                    f64::from_bits(iterative_relative_tolerance_bits);
                policy.direct_assembly = DirectSchurAssembly::Auto;
            }
            SparseWorkspaceOptions::Direct {
                point_back_substitution,
            } => policy.point_back_substitution = point_back_substitution,
            SparseWorkspaceOptions::DirectWithAssembly {
                point_back_substitution,
                assembly,
            } => {
                policy.point_back_substitution = point_back_substitution;
                policy.direct_assembly = assembly;
            }
            SparseWorkspaceOptions::IterativeSchur(iterative) => {
                policy.camera_solver = SparseCameraSolver::IterativeSchur;
                policy.retained_matvec = iterative.matvec();
                policy.retained_cross_precision = iterative.cross_precision();
                policy.iterative_maximum_iterations = iterative.maximum_iterations();
                policy.iterative_relative_tolerance = iterative.relative_tolerance();
                policy.iterative_direct_fallback = iterative.direct_fallback();
                policy.exact_camera_rank_diagnostics = iterative.exact_camera_rank_diagnostics();
            }
        }
        policy
    }
}

impl SparseBundleAdjustmentWorkspace {
    /// Set the positive finite initial dimensionless LM damping value.
    /// Invalid values are rejected when refinement begins.
    #[cfg(feature = "unstable-internals")]
    pub fn with_initial_damping(mut self, initial_damping: f64) -> Self {
        self.core.initial_damping = initial_damping;
        self
    }

    /// Configured initial dimensionless LM damping value.
    #[cfg(feature = "unstable-internals")]
    pub const fn initial_damping(&self) -> f64 {
        self.core.initial_damping
    }

    #[cfg(test)]
    pub(in crate::kernel::multi_view::sparse) fn take_retained_for_test(
        &mut self,
    ) -> Option<RetainedSchurWorkspace> {
        let backend = std::mem::replace(
            &mut self.backend,
            CalibratedBackend::Direct(DirectAssemblyState::Serial),
        );
        match backend {
            CalibratedBackend::Direct(assembly) => {
                self.backend = CalibratedBackend::Direct(assembly);
                None
            }
            CalibratedBackend::Retained(retained) => Some(*retained),
        }
    }

    /// Analyze and allocate storage for one calibrated BA visibility graph.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_options(
            camera_count,
            measurements,
            gauge,
            SparseWorkspaceOptions::default(),
        )
    }

    /// Analyze one calibrated visibility graph with a compact backend policy.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_with_options(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        options: SparseWorkspaceOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_fixed_blocks(
            camera_count,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            options,
        )
    }

    /// Analyze one calibrated graph with an explicit camera-pose variable set.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_with_pose_variables(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        pose_variables: CameraPoseVariables,
        options: SparseWorkspaceOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_fixed_blocks_and_pose_variables(
            camera_count,
            measurements,
            gauge,
            FixedVariableBlocks::NONE,
            pose_variables,
            options,
        )
    }

    /// Analyze one calibrated graph with exact fixed pose and point blocks.
    ///
    /// Fixed-block masks are part of the workspace topology. Intrinsic masks
    /// must be empty because calibrated cameras have no optimized intrinsic
    /// block.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_with_fixed_blocks(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        options: SparseWorkspaceOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_fixed_blocks_and_pose_variables(
            camera_count,
            measurements,
            gauge,
            fixed,
            CameraPoseVariables::RotationAndTranslation,
            options,
        )
    }

    /// Analyze one calibrated graph with fixed blocks and an explicit
    /// camera-pose variable set.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_with_fixed_blocks_and_pose_variables(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        pose_variables: CameraPoseVariables,
        options: SparseWorkspaceOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_policy(
            camera_count,
            measurements,
            gauge,
            fixed,
            pose_variables,
            None,
            CalibratedWorkspacePolicy::from_options(options),
        )
    }

    /// Analyze fixed-intrinsic BA with the exact per-camera rotation/center masks.
    ///
    /// At least one camera and one point are required; track information and
    /// final rank decide whether the requested coordinates are observable.
    /// One mask is required per camera. Unlike the uniform constructors, this
    /// constructor does not implicitly fix camera zero. Whole poses selected by
    /// `fixed.camera_poses` are held regardless of their coordinate mask. The
    /// masks are immutable workspace topology and apply to every refinement.
    /// A baseline chart requires a center-zero reference frame; its observability
    /// is still assessed on the requested active coordinates. Cartesian
    /// projection-translation gauges are rejected for rotation-only poses;
    /// their fixed-center chart preserves translation norm, not components.
    /// Selected covariance queries currently reject explicit-mask workspaces;
    /// they support only the legacy uniform constructor layouts.
    pub fn try_new_with_fixed_blocks_and_pose_masks(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        pose_masks: &[crate::PoseVariables],
        options: SparseWorkspaceOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        if !matches!(gauge, ScaleGauge::Free)
            && pose_masks.first().is_some_and(|mask| mask.translation())
            && !fixed.camera_poses.first().copied().unwrap_or(false)
        {
            return Err(SparseBundleAdjustmentError::GaugeReferenceCenterIsFree);
        }
        if let ScaleGauge::FixedTranslationComponent { camera, .. } = gauge
            && pose_masks
                .get(camera)
                .is_some_and(|mask| mask.rotation() && !mask.translation())
            && !fixed.camera_poses.get(camera).copied().unwrap_or(false)
        {
            return Err(SparseBundleAdjustmentError::RotationOnlyComponentGaugeUnsupported);
        }
        let variables = if pose_masks.iter().all(|mask| !mask.rotation()) {
            CameraPoseVariables::TranslationOnly
        } else {
            CameraPoseVariables::RotationAndTranslation
        };
        Self::try_new_with_policy(
            camera_count,
            measurements,
            gauge,
            fixed,
            variables,
            Some(pose_masks),
            CalibratedWorkspacePolicy::from_options(options),
        )
    }

    /// Analyze one visibility graph with an explicit point back-substitution
    /// memory/speed policy.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_with_point_back_substitution(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        point_back_substitution: SparsePointBackSubstitution,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_options(
            camera_count,
            measurements,
            gauge,
            SparseWorkspaceOptions::direct_with_point_back_substitution(point_back_substitution),
        )
    }

    /// Analyze one visibility graph with an explicit camera-system solver.
    ///
    /// The iterative Schur policy uses cached point
    /// back-substitution because the same retained factors drive its camera
    /// operator. Exact LLT numerical storage is allocated on first fallback.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_with_camera_solver(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        camera_solver: SparseCameraSolver,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let options = match camera_solver {
            SparseCameraSolver::DirectLlt => SparseWorkspaceOptions::direct(),
            SparseCameraSolver::IterativeSchur => {
                SparseWorkspaceOptions::iterative_schur(IterativeSchurOptions::new())
            }
        };
        Self::try_new_with_options(camera_count, measurements, gauge, options)
    }

    /// Analyze one visibility graph for iterative Schur-PCG with an explicit
    /// matrix-vector policy.
    ///
    /// [`RetainedSchurMatvec::Fused`] preserves the small-problem default.
    /// [`RetainedSchurMatvec::PointMajorParallel`] is the explicit parallel
    /// policy, while [`RetainedSchurMatvec::Auto`] resolves fused versus
    /// point-major parallel execution once from this topology and available
    /// hardware. It does not select between PCG and LLT.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_with_iterative_schur_matvec(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        matvec: RetainedSchurMatvec,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_options(
            camera_count,
            measurements,
            gauge,
            SparseWorkspaceOptions::iterative_schur(
                IterativeSchurOptions::new().with_matvec(matvec),
            ),
        )
    }

    /// Analyze one visibility graph for iterative Schur-PCG with explicit
    /// finite iteration and matrix-vector policies.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_with_iterative_schur_options(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        options: IterativeSchurOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_options(
            camera_count,
            measurements,
            gauge,
            SparseWorkspaceOptions::iterative_schur(options),
        )
    }

    fn try_new_with_policy(
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        pose_variables: CameraPoseVariables,
        pose_masks: Option<&[crate::PoseVariables]>,
        policy: CalibratedWorkspacePolicy,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let CalibratedWorkspacePolicy {
            auto_memory_budget,
            mut point_back_substitution,
            mut camera_solver,
            mut retained_matvec,
            mut retained_cross_precision,
            iterative_maximum_iterations,
            iterative_relative_tolerance,
            iterative_direct_fallback,
            mut exact_camera_rank_diagnostics,
            mut direct_assembly,
        } = policy;
        if matches!(
            direct_assembly,
            DirectSchurAssembly::Parallel { threads: 0 }
        ) {
            return Err(SparseBundleAdjustmentError::DirectSchurThreadsMustBePositive);
        }
        if matches!(direct_assembly, DirectSchurAssembly::Parallel { .. })
            && point_back_substitution != SparsePointBackSubstitution::Cached
        {
            return Err(SparseBundleAdjustmentError::DirectSchurParallelRequiresCachedPoints);
        }
        if camera_solver == SparseCameraSolver::IterativeSchur && iterative_maximum_iterations == 0
        {
            return Err(SparseBundleAdjustmentError::IterativeSchurIterationsMustBePositive);
        }
        if camera_solver == SparseCameraSolver::IterativeSchur
            && !(iterative_relative_tolerance.is_finite()
                && iterative_relative_tolerance > 0.0
                && iterative_relative_tolerance < 1.0)
        {
            return Err(SparseBundleAdjustmentError::IterativeSchurToleranceInvalid);
        }
        let model = ObservationModel::from(measurements);
        let point_count = model.offsets().len().saturating_sub(1);
        validation::topology_with_fixed_points(
            camera_count,
            point_count,
            model,
            gauge,
            fixed.points,
            pose_masks.is_none(),
        )?;
        validation::fixed_topology(camera_count, point_count, 0, gauge, fixed)?;
        let fixed = if pose_masks.is_some() {
            fixed.canonical_with_camera_zero()
        } else {
            fixed.canonical()
        };
        let layout = match pose_masks {
            Some(masks) => ReducedLayout::try_calibrated_with_pose_masks(
                camera_count,
                gauge,
                masks,
                fixed.camera_poses,
            )?,
            None => ReducedLayout::try_calibrated(
                camera_count,
                gauge,
                pose_variables,
                fixed.camera_poses,
            )?,
        };
        let camera_offsets = layout.camera_offsets();
        let camera_dimensions = layout.camera_dimensions();
        let dimension = layout.dimension();
        let cross_len = dimension
            .checked_mul(3)
            .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        let point_cross_offsets = if point_back_substitution == SparsePointBackSubstitution::Cached
            || auto_memory_budget.is_some()
            || matches!(
                retained_matvec,
                RetainedSchurMatvec::Recomputed | RetainedSchurMatvec::RecomputedParallel { .. }
            ) {
            point_cross_offsets(model, point_count, camera_dimensions, fixed.points)?
        } else {
            Vec::new()
        };
        let point_cross_count = point_cross_offsets.last().copied().unwrap_or(0);
        let signature = SparseProblemSignature::try_new(
            camera_count,
            model.offsets(),
            (0..model.observation_count()).map(|observation| model.camera(observation)),
            gauge,
            pose_variables,
            fixed,
        )?;
        let signature = match pose_masks {
            Some(masks) => signature.with_pose_masks(masks, fixed),
            None => signature,
        };
        let signature = signature
            .with_model_identity(model.identity())
            .with_observation_intrinsics(model.observation_intrinsics());
        let camera_columns = reduced_block_columns(
            signature.point_pose_offsets(),
            signature.point_poses(),
            camera_dimensions,
            fixed.points,
        );
        let reduced_nonzero_count = reduced_nonzero_count(&camera_columns, camera_dimensions);
        let direct_assembly_plan = resolve_direct_assembly_plan(
            direct_assembly,
            SparsePointBackSubstitution::Cached,
            model,
            camera_dimensions,
            &point_cross_offsets,
        );
        let cached_point_bytes = estimated_point_cache_bytes(
            point_count,
            point_cross_count,
            true,
            Some(RetainedCrossPrecision::F64),
        );
        let cached_point_bytes_f32 = estimated_point_cache_bytes(
            point_count,
            point_cross_count,
            true,
            Some(RetainedCrossPrecision::F32),
        );
        let recomputed_point_bytes =
            estimated_point_cache_bytes(point_count, point_cross_count, false, None);
        let execution_estimates = ExecutionEstimates {
            direct: estimated_direct_bytes(
                reduced_nonzero_count,
                dimension,
                cached_point_bytes,
                direct_assembly_plan.estimated_bytes,
                exact_camera_rank_diagnostics && dimension <= DENSE_RANK_DIAGNOSTIC_MAX_DIMENSION,
            ),
            cached_retained: estimated_calibrated_retained_bytes(
                camera_count,
                point_count,
                dimension,
                model.observation_count(),
                cached_point_bytes,
                false,
            ),
            cached_retained_f32: Some(estimated_calibrated_retained_bytes(
                camera_count,
                point_count,
                dimension,
                model.observation_count(),
                cached_point_bytes_f32,
                false,
            )),
            recomputed_retained: Some(estimated_calibrated_retained_bytes(
                camera_count,
                point_count,
                dimension,
                model.observation_count(),
                recomputed_point_bytes,
                true,
            )),
        };
        let automatic_plan: Option<AutomaticExecutionPlan> = auto_memory_budget
            .map(|budget| {
                select_calibrated(
                    budget,
                    execution_estimates,
                    model.observation_count(),
                    dimension,
                )
            })
            .transpose()?;
        let mut retained_logical_shards = None;
        if let Some(plan) = automatic_plan {
            retained_cross_precision = plan.retained_cross_precision;
            match plan.selected {
                SparseExecutionBackend::DirectLlt => {
                    point_back_substitution = SparsePointBackSubstitution::Cached;
                    camera_solver = SparseCameraSolver::DirectLlt;
                    direct_assembly = DirectSchurAssembly::Auto;
                    retained_matvec = RetainedSchurMatvec::Fused;
                    exact_camera_rank_diagnostics = true;
                }
                SparseExecutionBackend::CachedIterativeSchur => {
                    point_back_substitution = SparsePointBackSubstitution::Cached;
                    camera_solver = SparseCameraSolver::IterativeSchur;
                    direct_assembly = DirectSchurAssembly::Serial;
                    retained_matvec = RetainedSchurMatvec::Auto;
                    // Speed selects the iterative steps; a commit gate still needs the rank
                    // certificate, so keep its one final factorization whenever it fits.
                    exact_camera_rank_diagnostics &= plan.rank_certificate_fits(
                        estimated_direct_bytes(reduced_nonzero_count, dimension, 0, 0, false),
                    );
                }
                SparseExecutionBackend::RecomputedIterativeSchur => {
                    point_back_substitution = SparsePointBackSubstitution::Recompute;
                    camera_solver = SparseCameraSolver::IterativeSchur;
                    direct_assembly = DirectSchurAssembly::Serial;
                    (retained_matvec, retained_logical_shards) = automatic_recomputed_matvec(
                        camera_count,
                        dimension,
                        model.observation_count(),
                    );
                    exact_camera_rank_diagnostics &= plan.rank_certificate_fits(
                        estimated_direct_bytes(reduced_nonzero_count, dimension, 0, 0, false),
                    );
                }
            }
        }
        let recomputed_crosses = matches!(
            retained_matvec,
            RetainedSchurMatvec::Recomputed | RetainedSchurMatvec::RecomputedParallel { .. }
        );
        if recomputed_crosses && camera_solver != SparseCameraSolver::IterativeSchur {
            return Err(SparseBundleAdjustmentError::RecomputedSchurRequiresIterativeSolver);
        }
        let point_step_cache_count = if recomputed_crosses {
            0
        } else {
            point_cross_offsets.len().saturating_sub(1)
        };
        let direct_pattern_ready =
            camera_solver == SparseCameraSolver::DirectLlt || exact_camera_rank_diagnostics;
        let (mut cholesky, scatter) = if direct_pattern_ready {
            let (cholesky, scatter) = analyze_reduced_pattern(
                dimension,
                reduced_pattern(
                    &camera_columns,
                    dimension,
                    camera_offsets,
                    camera_dimensions,
                ),
                camera_offsets,
                camera_dimensions,
            )?;
            (Some(cholesky), scatter)
        } else {
            (None, SparseScatter::empty())
        };
        let (reduced_dimension, reduced_value_count) = reduced_pattern_size(cholesky.as_ref());
        let direct_assembly_threads = resolve_direct_assembly_plan(
            direct_assembly,
            point_back_substitution,
            model,
            camera_dimensions,
            &point_cross_offsets,
        )
        .threads;
        let direct_assembly_workspace = direct_assembly_threads
            .map(|threads| {
                DirectSchurAssemblyWorkspace::try_new(
                    threads,
                    signature.point_pose_offsets(),
                    signature.point_poses(),
                    camera_dimensions,
                    &point_cross_offsets,
                    reduced_dimension,
                    reduced_value_count,
                    &scatter,
                )
            })
            .transpose()?;
        let direct_point_hessians = if direct_assembly_workspace.is_some() {
            vec![Default::default(); point_count]
        } else {
            Vec::new()
        };
        let values = if camera_solver == SparseCameraSolver::DirectLlt {
            vec![0.0; reserve_direct_numeric(&mut cholesky)?]
        } else {
            Vec::new()
        };
        let placeholder = RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let retained = match camera_solver {
            SparseCameraSolver::DirectLlt => None,
            SparseCameraSolver::IterativeSchur => Some(RetainedSchurWorkspace::new_iterative(
                camera_count,
                point_count,
                dimension,
                camera_offsets,
                camera_dimensions,
                model,
                retained_matvec,
                retained_logical_shards,
                iterative_maximum_iterations,
                iterative_relative_tolerance,
            )?),
        };
        let candidate_point_metrics = if retained
            .as_ref()
            .and_then(RetainedSchurWorkspace::parallel_threads)
            .is_some()
            && (!recomputed_crosses || cfg!(feature = "parallel"))
        {
            vec![[0.0; 3]; point_count]
        } else {
            Vec::new()
        };
        let anchored_six_camera_layout = camera_dimensions.first() == Some(&0)
            && camera_dimensions[1..]
                .iter()
                .all(|&camera_dimension| camera_dimension == 6);
        let placeholder_decision = explicit_decision(
            if recomputed_crosses {
                SparseExecutionBackend::RecomputedIterativeSchur
            } else if camera_solver.uses_retained_schur() {
                SparseExecutionBackend::CachedIterativeSchur
            } else {
                SparseExecutionBackend::DirectLlt
            },
            0,
            direct_assembly_threads,
            retained
                .as_ref()
                .and_then(RetainedSchurWorkspace::parallel_threads),
        );
        let core = SparseWorkspaceCore {
            execution_decision: placeholder_decision,
            signature,
            point_back_substitution: if recomputed_crosses {
                SparsePointBackSubstitution::Recompute
            } else {
                point_back_substitution
            },
            layout,
            anchored_six_camera_layout,
            candidate_poses: vec![placeholder; camera_count],
            candidate_points: vec![[0.0; 3]; point_count],
            candidate_point_metrics,
            cholesky,
            reduced_nonzero_count,
            values,
            scatter,
            exact_camera_rank_diagnostics,
            iterative_direct_fallback,
            initial_damping: camera_solver.default_initial_damping(),
            dense_rank_diagnostic: if camera_solver == SparseCameraSolver::DirectLlt
                && exact_camera_rank_diagnostics
                && dimension <= DENSE_RANK_DIAGNOSTIC_MAX_DIMENSION
            {
                vec![0.0; dense_rank_diagnostic_len(dimension)]
            } else {
                Vec::new()
            },
            #[cfg(test)]
            fail_next_direct_storage_for_test: false,
            gradient: vec![0.0; dimension],
            full_gradient: vec![0.0; dimension],
            step: vec![0.0; dimension],
            diagonal: vec![0.0; dimension],
            cross: vec![0.0; cross_len],
            active_parameters: vec![0; dimension],
            parameter_marks: vec![false; dimension],
            point_cache_valid: vec![false; point_step_cache_count],
            point_inverse_gradients: vec![[0.0; 3]; point_step_cache_count],
            point_damping_diagonals: vec![[0.0; 3]; point_step_cache_count],
            point_cross_offsets,
            point_parameters: Vec::new(),
            point_inverse_cross_values: PointCrossCache::new(
                point_cross_count,
                retained_cross_precision,
                !recomputed_crosses,
            ),
        };
        let backend = match retained {
            Some(retained) => CalibratedBackend::Retained(Box::new(retained)),
            None => CalibratedBackend::Direct(DirectAssemblyState::new(
                direct_assembly_workspace,
                direct_point_hessians,
            )),
        };
        let mut workspace = Self { core, backend };
        workspace.core.execution_decision = match automatic_plan {
            Some(plan) => finish_automatic_decision(
                plan,
                workspace.solver_specific_working_set_bytes(),
                workspace.direct_schur_parallel_threads(),
                workspace.retained_schur_parallel_threads(),
            ),
            None => explicit_decision(
                placeholder_decision.selected_backend(),
                workspace.solver_specific_working_set_bytes(),
                workspace.direct_schur_parallel_threads(),
                workspace.retained_schur_parallel_threads(),
            ),
        };
        Ok(workspace)
    }

    /// Exact camera count accepted by this topology-specific workspace.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn camera_count(&self) -> usize {
        self.core.signature.camera_count()
    }

    /// Exact point count accepted by this topology-specific workspace.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn point_count(&self) -> usize {
        self.core.signature.point_count()
    }

    /// Dimension of the reduced camera system.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub const fn camera_dimension(&self) -> usize {
        self.core.layout.dimension()
    }

    /// Uniform Jacobian dispatch used by this workspace. Per-camera masks may
    /// remove coordinates from this chart; the `camera_pose_masks` getter under
    /// `unstable-internals` exposes the exact masks.
    pub const fn camera_pose_variables(&self) -> CameraPoseVariables {
        self.core.signature.pose_variables()
    }

    /// Exact rotation/center masks, including explicit quotient anchors.
    #[cfg(any(feature = "unstable-internals", all(test, feature = "parallel")))]
    pub fn camera_pose_masks(&self) -> &[crate::PoseVariables] {
        self.core.signature.pose_masks()
    }

    pub(in crate::kernel::multi_view::sparse) const fn dimension(&self) -> usize {
        self.core.layout.dimension()
    }

    pub(in crate::kernel::multi_view::sparse) fn fixed_points(&self) -> &[bool] {
        self.core.signature.fixed_points()
    }

    pub(in crate::kernel::multi_view::sparse) fn camera_offsets(&self) -> &[usize] {
        self.core.layout.camera_offsets()
    }

    pub(in crate::kernel::multi_view::sparse) fn camera_dimensions(&self) -> &[usize] {
        self.core.layout.camera_dimensions()
    }

    /// Stored lower-triangular scalar entries in the reduced camera system.
    pub fn reduced_nonzero_count(&self) -> usize {
        self.core.reduced_nonzero_count
    }

    /// Bytes reserved for cached inverse point gradients, inverse cross rows,
    /// damping diagonals, validity bits, and point offsets used by Schur
    /// back-substitution and LM gain evaluation.
    pub fn point_linearization_cache_bytes(&self) -> usize {
        self.core.point_cache_valid.capacity() * std::mem::size_of::<bool>()
            + self.core.point_inverse_gradients.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.core.point_damping_diagonals.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.core.point_cross_offsets.capacity() * std::mem::size_of::<usize>()
            + self.core.point_inverse_cross_values.capacity_bytes()
    }

    /// Point back-substitution policy selected at construction.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_back_substitution(&self) -> SparsePointBackSubstitution {
        self.core.point_back_substitution
    }

    /// Camera-system solver selected at construction.
    pub const fn camera_solver(&self) -> SparseCameraSolver {
        self.backend.camera_solver()
    }

    /// Immutable construction-time backend and memory-planning receipt.
    pub const fn execution_decision(
        &self,
    ) -> crate::kernel::multi_view::sparse::SparseExecutionDecision {
        self.core.execution_decision
    }

    /// Resolved direct-Schur assembly worker count.
    ///
    /// `None` means no parallel assembly workers are active.
    pub fn direct_schur_parallel_threads(&self) -> Option<usize> {
        match &self.backend {
            CalibratedBackend::Direct(assembly) => assembly.parallel_threads(),
            CalibratedBackend::Retained(_) => None,
        }
    }

    /// Additional bytes reserved for deterministic direct-Schur assembly.
    ///
    /// Cached transformed cross rows are reported separately by
    /// [`Self::point_linearization_cache_bytes`]. Executor-internal storage is
    /// not included.
    pub fn direct_schur_assembly_bytes(&self) -> usize {
        match &self.backend {
            CalibratedBackend::Direct(assembly) => assembly.auxiliary_bytes(),
            CalibratedBackend::Retained(_) => 0,
        }
    }

    /// Configured retained Schur matvec policy, or `None` for direct LLT.
    #[cfg(feature = "unstable-internals")]
    pub fn retained_schur_matvec(&self) -> Option<RetainedSchurMatvec> {
        match &self.backend {
            CalibratedBackend::Direct(_) => None,
            CalibratedBackend::Retained(retained) => Some(retained.matvec_policy()),
        }
    }

    /// Storage precision of cached retained transformed cross rows.
    ///
    /// Returns `None` when no cross-row cache is allocated, including the
    /// recomputed policy and direct point recomputation.
    #[cfg(feature = "unstable-internals")]
    pub fn retained_cross_precision(&self) -> Option<RetainedCrossPrecision> {
        self.core.point_inverse_cross_values.precision()
    }

    /// Allocated bytes used only by cached retained `V^-1 W'` cross rows.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn retained_cross_row_bytes(&self) -> usize {
        self.core.point_inverse_cross_values.capacity_bytes()
    }

    /// Whether transformed point/camera cross rows are retained for Schur
    /// operator use or point back-substitution.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn retains_cross_rows(&self) -> bool {
        self.core.point_inverse_cross_values.retains_rows()
    }

    /// Resolved retained-Schur worker count. `None` means the fused operator
    /// is active, including when [`RetainedSchurMatvec::Auto`] selects it.
    pub fn retained_schur_parallel_threads(&self) -> Option<usize> {
        match &self.backend {
            CalibratedBackend::Direct(_) => None,
            CalibratedBackend::Retained(retained) => retained.parallel_threads(),
        }
    }

    /// Configured finite PCG iteration guard, or `None` for a non-iterative
    /// camera solver.
    #[cfg(feature = "unstable-internals")]
    pub fn iterative_schur_maximum_iterations(&self) -> Option<usize> {
        match &self.backend {
            CalibratedBackend::Direct(_) => None,
            CalibratedBackend::Retained(retained) => retained.iterative_maximum_iterations(),
        }
    }

    /// Accepted true relative PCG residual, or `None` for a non-iterative
    /// camera solver.
    pub fn iterative_schur_relative_tolerance(&self) -> Option<f64> {
        match &self.backend {
            CalibratedBackend::Direct(_) => None,
            CalibratedBackend::Retained(retained) => retained.iterative_relative_tolerance(),
        }
    }

    /// Whether the final reduced camera system receives an exact rank assessment.
    #[cfg(feature = "unstable-internals")]
    pub const fn exact_camera_rank_diagnostics(&self) -> bool {
        self.core.exact_camera_rank_diagnostics
    }

    /// Additional explicit bytes reserved by the retained Schur accelerator.
    /// Executor-internal allocations are not included.
    pub fn retained_schur_bytes(&self) -> usize {
        let retained = match &self.backend {
            CalibratedBackend::Direct(_) => 0,
            CalibratedBackend::Retained(retained) => retained.retained_bytes(),
        };
        retained + self.core.candidate_point_metrics.capacity() * std::mem::size_of::<[f64; 3]>()
    }

    /// Combined retained-operator auxiliary and point-linearization cache
    /// bytes.
    ///
    /// This is the relevant retained working-set quantity when comparing the
    /// matrix-free Schur path with direct numeric storage. It does not include
    /// common pose/point candidate buffers or symbolic sparsity metadata.
    pub fn retained_working_set_bytes(&self) -> usize {
        self.retained_schur_bytes()
            .saturating_add(self.point_linearization_cache_bytes())
    }

    /// Combined point cache, retained/direct-assembly auxiliary, and direct
    /// numeric bytes currently reserved by the selected solver.
    ///
    /// Common pose/point candidate buffers and symbolic sparsity metadata are
    /// excluded. Unlike [`Self::retained_working_set_bytes`], this quantity
    /// grows if lazy LLT fallback or exact rank diagnostics allocate direct
    /// numerical storage.
    pub fn solver_specific_working_set_bytes(&self) -> usize {
        self.retained_working_set_bytes()
            .saturating_add(self.direct_schur_assembly_bytes())
            .saturating_add(self.direct_numeric_bytes())
    }

    /// Bytes currently reserved for explicit Schur values, LLT values, and
    /// faer's numerical scratch buffer.
    pub fn direct_numeric_bytes(&self) -> usize {
        self.core.direct_numeric_bytes()
    }

    /// Bytes retained by the scalar reduced pattern and assembly scatter.
    ///
    /// Retained-only workspaces keep this near zero until exact fallback,
    /// explicit reservation, or rank diagnostics require the direct system.
    /// Faer's opaque symbolic-factor ordering is not included.
    #[cfg(feature = "unstable-internals")]
    pub fn direct_pattern_bytes(&self) -> usize {
        self.core.direct_pattern_bytes()
    }

    /// Reserve exact-LLT and final-rank diagnostic storage before entering an
    /// allocation-free solve region.
    #[cfg(any(all(test, feature = "parallel"), feature = "unstable-internals"))]
    pub fn try_reserve_direct_storage(&mut self) -> Result<(), SparseBundleAdjustmentError> {
        self.ensure_direct_storage()?;
        let Self { core, backend } = self;
        if let CalibratedBackend::Retained(retained) = backend {
            retained.ensure_fallback_materializer(core)?;
        }
        Ok(())
    }

    pub(in crate::kernel::multi_view::sparse) fn ensure_direct_storage(
        &mut self,
    ) -> Result<(), SparseBundleAdjustmentError> {
        #[cfg(test)]
        if std::mem::take(&mut self.core.fail_next_direct_storage_for_test) {
            return Err(SparseBundleAdjustmentError::NumericalWorkspaceAllocation);
        }
        if self.core.cholesky.is_none() {
            self.core.check_lazy_direct_storage()?;
            let camera_columns = reduced_block_columns(
                self.core.signature.point_pose_offsets(),
                self.core.signature.point_poses(),
                self.camera_dimensions(),
                self.fixed_points(),
            );
            let (cholesky, scatter) = analyze_reduced_pattern(
                self.dimension(),
                reduced_pattern(
                    &camera_columns,
                    self.dimension(),
                    self.camera_offsets(),
                    self.camera_dimensions(),
                ),
                self.camera_offsets(),
                self.camera_dimensions(),
            )?;
            self.core.cholesky = Some(cholesky);
            self.core.scatter = scatter;
        }
        self.core.ensure_numeric_storage()
    }

    pub(in crate::kernel::multi_view::sparse) fn matches_with_fixed(
        &self,
        camera_count: usize,
        measurements: MultiViewMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
    ) -> bool {
        self.core.signature.matches(
            camera_count,
            measurements.point_offsets(),
            (0..measurements.observation_count())
                .map(|observation| measurements.observation_camera(observation)),
            gauge,
            self.core.signature.pose_variables(),
            fixed,
        ) && self
            .core
            .signature
            .model_identity_matches(ObservationModel::from(measurements).identity())
            && self
                .core
                .signature
                .observation_intrinsics_match(measurements.observation_intrinsics())
            && fixed.intrinsics.is_empty()
    }
}

#[derive(Clone, Copy)]
struct DirectAssemblyPlan {
    threads: Option<usize>,
    estimated_bytes: usize,
}

fn resolve_direct_assembly_plan(
    policy: DirectSchurAssembly,
    point_back_substitution: SparsePointBackSubstitution,
    model: ObservationModel<'_>,
    camera_dimensions: &[usize],
    point_cross_offsets: &[usize],
) -> DirectAssemblyPlan {
    match policy {
        DirectSchurAssembly::Serial => DirectAssemblyPlan {
            threads: None,
            estimated_bytes: 0,
        },
        DirectSchurAssembly::Parallel { threads } => {
            let point_count = point_cross_offsets.len().saturating_sub(1);
            let camera_count = camera_dimensions.len();
            let dimension = camera_dimensions.iter().copied().sum::<usize>();
            let active_block_references =
                active_block_reference_count(model, camera_dimensions, point_cross_offsets);
            DirectAssemblyPlan {
                threads: Some(threads),
                estimated_bytes: estimated_calibrated_auxiliary_bytes(
                    threads,
                    point_count,
                    model.observation_count(),
                    active_block_references,
                    camera_count,
                    dimension,
                ),
            }
        }
        DirectSchurAssembly::Auto => {
            if point_back_substitution != SparsePointBackSubstitution::Cached {
                return DirectAssemblyPlan {
                    threads: None,
                    estimated_bytes: 0,
                };
            }
            let mut coupling_terms = 0usize;
            let mut active_block_references = 0usize;
            for point in 0..point_cross_offsets.len().saturating_sub(1) {
                if point_cross_offsets[point] == point_cross_offsets[point + 1] {
                    continue;
                }
                let mut active_dimension = 0usize;
                for observation in model.offsets()[point]..model.offsets()[point + 1] {
                    let dimension = camera_dimensions[model.camera(observation)];
                    if dimension > 0 {
                        active_block_references = active_block_references.saturating_add(1);
                        active_dimension = active_dimension.saturating_add(dimension);
                    }
                }
                coupling_terms = coupling_terms.saturating_add(
                    active_dimension.saturating_mul(active_dimension.saturating_add(1)) / 2,
                );
            }
            let estimate = |threads| {
                let point_count = point_cross_offsets.len().saturating_sub(1);
                let camera_count = camera_dimensions.len();
                let dimension = camera_dimensions.iter().copied().sum::<usize>();
                estimated_calibrated_auxiliary_bytes(
                    threads,
                    point_count,
                    model.observation_count(),
                    active_block_references,
                    camera_count,
                    dimension,
                )
            };
            let threads =
                auto_direct_assembly_threads(coupling_terms, model.observation_count(), estimate);
            DirectAssemblyPlan {
                threads,
                estimated_bytes: threads.map_or(0, estimate),
            }
        }
    }
}

fn active_block_reference_count(
    model: ObservationModel<'_>,
    camera_dimensions: &[usize],
    point_cross_offsets: &[usize],
) -> usize {
    let mut references = 0usize;
    for point in 0..point_cross_offsets.len().saturating_sub(1) {
        if point_cross_offsets[point] == point_cross_offsets[point + 1] {
            continue;
        }
        references = references.saturating_add(
            (model.offsets()[point]..model.offsets()[point + 1])
                .filter(|&observation| camera_dimensions[model.camera(observation)] > 0)
                .count(),
        );
    }
    references
}
