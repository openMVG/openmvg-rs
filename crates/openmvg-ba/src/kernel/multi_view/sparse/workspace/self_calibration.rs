// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grouped-radial sparse-workspace construction and storage lifecycle.

use super::pattern::{
    analyze_reduced_pattern, radial_point_parameters, radial_reduced_block_columns,
    radial_reduced_nonzero_count, radial_reduced_pattern,
};
use super::topology::{ReducedLayout, SelfCalibrationSignature, SparseProblemSignature};
use super::{
    DirectAssemblyState, PointCrossCache, SelfCalibrationBackend, SparseScatter,
    SparseSelfCalibrationWorkspace, SparseWorkspaceCore, reduced_pattern_size,
    reserve_direct_numeric,
};
use crate::kernel::RelativePose;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::self_calibration::{RadialCamera, SelfCalibratingMeasurements};
use crate::kernel::multi_view::sparse::api::{
    DirectSchurAssembly, IterativeSchurOptions, RetainedCrossPrecision, RetainedSchurMatvec,
    SparseBundleAdjustmentError, SparseCameraSolver, SparseExecutionBackend,
    SparsePointBackSubstitution, SparseWorkspaceMemoryBudget,
};
use crate::kernel::multi_view::sparse::direct_assembly::{
    DirectSchurAssemblyWorkspace, auto_direct_assembly_threads, estimated_indexed_auxiliary_bytes,
};
use crate::kernel::multi_view::sparse::execution_policy::{
    AutomaticExecutionPlan, ExecutionEstimates, estimated_direct_bytes,
    estimated_point_cache_bytes, explicit_decision, finish_automatic_decision,
    select_self_calibrating,
};
use crate::kernel::multi_view::sparse::self_calibration_retained::{
    SelfCalibrationRetainedWorkspace, estimated_self_calibrating_retained_bytes,
};
use crate::kernel::multi_view::{
    CameraPoseVariables, FixedVariableBlocks, MultiViewBundleAdjustmentError, ScaleGauge,
    validation,
};

#[derive(Clone, Copy)]
struct SelfCalibrationWorkspacePolicy {
    auto_memory_budget: Option<SparseWorkspaceMemoryBudget>,
    direct_assembly: DirectSchurAssembly,
    camera_solver: SparseCameraSolver,
    retained_matvec: RetainedSchurMatvec,
    retained_cross_precision: RetainedCrossPrecision,
    iterative_maximum_iterations: usize,
    iterative_relative_tolerance: f64,
    iterative_direct_fallback: bool,
    exact_camera_rank_diagnostics: bool,
}

impl SelfCalibrationWorkspacePolicy {
    const fn direct(assembly: DirectSchurAssembly) -> Self {
        Self {
            auto_memory_budget: None,
            direct_assembly: assembly,
            camera_solver: SparseCameraSolver::DirectLlt,
            retained_matvec: RetainedSchurMatvec::Fused,
            retained_cross_precision: RetainedCrossPrecision::F64,
            iterative_maximum_iterations: IterativeSchurOptions::DEFAULT_MAXIMUM_ITERATIONS,
            iterative_relative_tolerance: IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE,
            iterative_direct_fallback: true,
            exact_camera_rank_diagnostics: true,
        }
    }

    const fn iterative(options: IterativeSchurOptions) -> Self {
        Self {
            auto_memory_budget: None,
            direct_assembly: DirectSchurAssembly::Serial,
            camera_solver: SparseCameraSolver::IterativeSchur,
            retained_matvec: options.matvec(),
            retained_cross_precision: options.cross_precision(),
            iterative_maximum_iterations: options.maximum_iterations(),
            iterative_relative_tolerance: options.relative_tolerance(),
            iterative_direct_fallback: options.direct_fallback(),
            exact_camera_rank_diagnostics: options.exact_camera_rank_diagnostics(),
        }
    }

    const fn automatic(memory_budget: SparseWorkspaceMemoryBudget) -> Self {
        Self {
            auto_memory_budget: Some(memory_budget),
            direct_assembly: DirectSchurAssembly::Auto,
            camera_solver: SparseCameraSolver::DirectLlt,
            retained_matvec: RetainedSchurMatvec::Auto,
            retained_cross_precision: RetainedCrossPrecision::F64,
            iterative_maximum_iterations: IterativeSchurOptions::DEFAULT_MAXIMUM_ITERATIONS,
            iterative_relative_tolerance: IterativeSchurOptions::DEFAULT_RELATIVE_TOLERANCE,
            iterative_direct_fallback: true,
            exact_camera_rank_diagnostics: true,
        }
    }
}

impl SparseSelfCalibrationWorkspace {
    /// Set the positive finite initial dimensionless LM damping value.
    /// Invalid values are rejected when refinement begins.
    #[cfg(feature = "unstable-internals")]
    pub fn with_initial_damping(mut self, initial_damping: f64) -> Self {
        self.sparse.initial_damping = initial_damping;
        self
    }

    /// Configured initial dimensionless LM damping value.
    #[cfg(feature = "unstable-internals")]
    pub const fn initial_damping(&self) -> f64 {
        self.sparse.initial_damping
    }

    /// Analyze a per-view radial self-calibration graph for sparse direct LLT.
    ///
    /// `fixed` is part of the reusable workspace topology.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new(
        camera_count: usize,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            camera_count,
            None,
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::direct(DirectSchurAssembly::Serial),
        )
    }

    /// Analyze a per-view radial graph for exact LLT with an explicit
    /// direct-Schur assembly policy.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_with_direct_assembly(
        camera_count: usize,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        assembly: DirectSchurAssembly,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            camera_count,
            None,
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::direct(assembly),
        )
    }

    /// Analyze a per-view radial graph with construction-time backend
    /// selection under a soft solver-specific workspace-memory budget.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_auto(
        camera_count: usize,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        memory_budget: SparseWorkspaceMemoryBudget,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            camera_count,
            None,
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::automatic(memory_budget),
        )
    }

    /// Analyze a per-view radial graph for exact retained Schur-PCG.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_with_iterative_schur_options(
        camera_count: usize,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        options: IterativeSchurOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            camera_count,
            None,
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::iterative(options),
        )
    }

    /// Analyze a per-view radial graph for retained Schur-PCG with an
    /// explicit matrix-vector execution policy.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_with_iterative_schur_matvec(
        camera_count: usize,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        matvec: RetainedSchurMatvec,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_with_iterative_schur_options(
            camera_count,
            measurements,
            gauge,
            fixed,
            IterativeSchurOptions::new().with_matvec(matvec),
        )
    }

    /// Analyze a radial self-calibration graph with shared intrinsic groups.
    ///
    /// The group assignment and `fixed` masks are part of the reusable
    /// workspace topology. Every group must be referenced by a camera.
    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn try_new_grouped(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: &[usize],
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            intrinsic_group_count,
            Some(camera_to_group),
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::direct(DirectSchurAssembly::Serial),
        )
    }

    /// Analyze a shared-intrinsic radial graph for exact LLT with an explicit
    /// direct-Schur assembly policy.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_grouped_with_direct_assembly(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: &[usize],
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        assembly: DirectSchurAssembly,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            intrinsic_group_count,
            Some(camera_to_group),
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::direct(assembly),
        )
    }

    /// Analyze a grouped radial graph with construction-time backend
    /// selection under a soft solver-specific workspace-memory budget.
    pub fn try_new_grouped_auto(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: &[usize],
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        memory_budget: SparseWorkspaceMemoryBudget,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            intrinsic_group_count,
            Some(camera_to_group),
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::automatic(memory_budget),
        )
    }

    /// Analyze a shared-intrinsic radial graph for exact retained Schur-PCG.
    pub fn try_new_grouped_with_iterative_schur_options(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: &[usize],
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        options: IterativeSchurOptions,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_problem(
            camera_count,
            intrinsic_group_count,
            Some(camera_to_group),
            measurements,
            gauge,
            fixed,
            SelfCalibrationWorkspacePolicy::iterative(options),
        )
    }

    /// Analyze a shared-intrinsic radial graph for retained Schur-PCG with
    /// an explicit matrix-vector execution policy.
    #[cfg(feature = "unstable-internals")]
    pub fn try_new_grouped_with_iterative_schur_matvec(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: &[usize],
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        matvec: RetainedSchurMatvec,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::try_new_grouped_with_iterative_schur_options(
            camera_count,
            intrinsic_group_count,
            camera_to_group,
            measurements,
            gauge,
            fixed,
            IterativeSchurOptions::new().with_matvec(matvec),
        )
    }

    fn try_new_problem(
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: Option<&[usize]>,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
        policy: SelfCalibrationWorkspacePolicy,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let SelfCalibrationWorkspacePolicy {
            auto_memory_budget,
            mut direct_assembly,
            mut camera_solver,
            mut retained_matvec,
            retained_cross_precision,
            iterative_maximum_iterations,
            iterative_relative_tolerance,
            iterative_direct_fallback,
            mut exact_camera_rank_diagnostics,
        } = policy;
        if matches!(
            direct_assembly,
            DirectSchurAssembly::Parallel { threads: 0 }
        ) {
            return Err(SparseBundleAdjustmentError::DirectSchurThreadsMustBePositive);
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
        let self_calibration_signature = SelfCalibrationSignature::try_new(
            camera_count,
            intrinsic_group_count,
            camera_to_group,
            fixed,
        )?;
        let placeholder_camera = RadialCamera::new(1.0, 0.0, 0.0);
        let candidate_cameras = vec![placeholder_camera; intrinsic_group_count];
        let model = match camera_to_group {
            Some(camera_to_group) => ObservationModel::radial_grouped(
                measurements.observations(),
                measurements.point_offsets(),
                &candidate_cameras,
                camera_to_group,
            ),
            None => ObservationModel::radial(
                measurements.observations(),
                measurements.point_offsets(),
                &candidate_cameras,
            ),
        };
        let point_count = model.offsets().len().saturating_sub(1);
        validation::topology(camera_count, point_count, model, gauge)?;
        validation::fixed_topology(
            camera_count,
            point_count,
            intrinsic_group_count,
            gauge,
            fixed,
        )?;
        let fixed = fixed.canonical();
        let layout = ReducedLayout::try_radial(camera_count, intrinsic_group_count, gauge, fixed)?;
        let camera_offsets = layout.camera_offsets();
        let camera_dimensions = layout.camera_dimensions();
        let intrinsic_offsets = layout.intrinsic_offsets();
        let intrinsic_dimensions = layout.intrinsic_dimensions();
        let dimension = layout.dimension();
        let cross_len = dimension
            .checked_mul(3)
            .ok_or(MultiViewBundleAdjustmentError::CameraCapacityOverflow)?;
        let effective_camera_to_group: Vec<usize> = camera_to_group
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| (0..camera_count).collect());
        let problem_signature = SparseProblemSignature::try_new(
            camera_count,
            measurements.point_offsets(),
            measurements
                .observations()
                .iter()
                .map(|observation| observation.camera),
            gauge,
            CameraPoseVariables::RotationAndTranslation,
            fixed,
        )?;
        let direct_index_requested = auto_memory_budget.is_some()
            || (camera_solver == SparseCameraSolver::DirectLlt
                && !matches!(direct_assembly, DirectSchurAssembly::Serial));
        let (mut point_cross_offsets, point_parameters, active_block_references) =
            if camera_solver == SparseCameraSolver::IterativeSchur || direct_index_requested {
                radial_point_parameters(
                    model,
                    point_count,
                    dimension,
                    camera_offsets,
                    camera_dimensions,
                    intrinsic_offsets,
                    intrinsic_dimensions,
                    fixed.points,
                )?
            } else {
                (Vec::new(), Vec::new(), 0)
            };
        let build_direct_visibility = auto_memory_budget.is_some()
            || camera_solver == SparseCameraSolver::DirectLlt
            || exact_camera_rank_diagnostics;
        // A retained-only solve does not consume the explicit reduced graph.
        // Defer its potentially quadratic-in-track-length construction until a
        // direct fallback is actually requested.
        let block_columns = build_direct_visibility.then(|| {
            radial_reduced_block_columns(
                measurements.point_offsets(),
                problem_signature.observation_cameras(),
                camera_to_group,
                camera_dimensions,
                intrinsic_dimensions,
                fixed.points,
            )
        });
        let mut block_offsets = Vec::with_capacity(camera_count + intrinsic_group_count);
        block_offsets.extend_from_slice(camera_offsets);
        block_offsets.extend_from_slice(intrinsic_offsets);
        let mut block_dimensions = Vec::with_capacity(camera_count + intrinsic_group_count);
        block_dimensions.extend_from_slice(camera_dimensions);
        block_dimensions.extend_from_slice(intrinsic_dimensions);
        let base_block_edges = effective_camera_to_group
            .iter()
            .copied()
            .enumerate()
            .filter(|&(camera, group)| {
                camera_dimensions[camera] > 0 && intrinsic_dimensions[group] > 0
            })
            .map(|(camera, group)| (camera_count + group, camera))
            .collect::<Vec<_>>();
        let reduced_nonzero_count = block_columns.as_ref().map_or(0, |block_columns| {
            radial_reduced_nonzero_count(block_columns, &block_dimensions)
        });
        let direct_assembly_plan = resolve_direct_assembly_plan(
            DirectSchurAssembly::Auto,
            SparseCameraSolver::DirectLlt,
            &point_cross_offsets,
            DirectAssemblyTopologySize {
                point_observations: measurements.observation_count(),
                active_block_references,
                block_count: block_dimensions.len(),
                base_edge_count: base_block_edges.len(),
                dimension,
            },
        );
        let direct_point_bytes = estimated_point_cache_bytes(
            point_count,
            point_cross_offsets.last().copied().unwrap_or(0),
            direct_assembly_plan.threads.is_some() && fixed.points.is_empty(),
            Some(RetainedCrossPrecision::F64),
        );
        let retained_point_bytes = estimated_point_cache_bytes(
            point_count,
            point_cross_offsets.last().copied().unwrap_or(0),
            true,
            Some(RetainedCrossPrecision::F64),
        );
        let execution_estimates = ExecutionEstimates {
            direct: estimated_direct_bytes(
                reduced_nonzero_count,
                dimension,
                direct_point_bytes,
                direct_assembly_plan.estimated_bytes,
                false,
            ),
            cached_retained: estimated_self_calibrating_retained_bytes(
                block_dimensions.len(),
                base_block_edges.len(),
                point_count,
                point_parameters.len(),
                dimension,
                measurements.observation_count(),
                retained_point_bytes,
                fixed.points.is_empty(),
            ),
            cached_retained_f32: None,
            recomputed_retained: None,
        };
        let automatic_plan: Option<AutomaticExecutionPlan> = auto_memory_budget
            .map(|budget| select_self_calibrating(budget, execution_estimates))
            .transpose()?;
        if let Some(plan) = automatic_plan {
            match plan.selected {
                SparseExecutionBackend::DirectLlt => {
                    camera_solver = SparseCameraSolver::DirectLlt;
                    direct_assembly = DirectSchurAssembly::Auto;
                    retained_matvec = RetainedSchurMatvec::Fused;
                    exact_camera_rank_diagnostics = true;
                }
                SparseExecutionBackend::CachedIterativeSchur => {
                    camera_solver = SparseCameraSolver::IterativeSchur;
                    direct_assembly = DirectSchurAssembly::Serial;
                    retained_matvec = RetainedSchurMatvec::Auto;
                    exact_camera_rank_diagnostics &= plan.rank_certificate_fits(
                        estimated_direct_bytes(reduced_nonzero_count, dimension, 0, 0, false),
                    );
                }
                SparseExecutionBackend::RecomputedIterativeSchur => {
                    return Err(
                        SparseBundleAdjustmentError::RecomputedSchurUnsupportedForSelfCalibration,
                    );
                }
            }
        }
        let direct_pattern_ready =
            camera_solver == SparseCameraSolver::DirectLlt || exact_camera_rank_diagnostics;
        let (mut cholesky, scatter) = if direct_pattern_ready {
            let block_columns = block_columns
                .as_ref()
                .ok_or(SparseBundleAdjustmentError::SymbolicFactorization)?;
            let (cholesky, scatter) = analyze_reduced_pattern(
                dimension,
                radial_reduced_pattern(block_columns, dimension, &block_offsets, &block_dimensions),
                &block_offsets,
                &block_dimensions,
            )?;
            (Some(cholesky), scatter)
        } else {
            (None, SparseScatter::empty())
        };
        let (reduced_dimension, reduced_value_count) = reduced_pattern_size(cholesky.as_ref());
        let direct_assembly_threads = resolve_direct_assembly_plan(
            direct_assembly,
            camera_solver,
            &point_cross_offsets,
            DirectAssemblyTopologySize {
                point_observations: measurements.observation_count(),
                active_block_references,
                block_count: block_dimensions.len(),
                base_edge_count: base_block_edges.len(),
                dimension,
            },
        )
        .threads;
        let direct_assembly_workspace = direct_assembly_threads
            .map(|threads| {
                DirectSchurAssemblyWorkspace::try_new_indexed(
                    threads,
                    measurements.point_offsets(),
                    &point_cross_offsets,
                    &point_parameters,
                    &block_dimensions,
                    &base_block_edges,
                    reduced_dimension,
                    reduced_value_count,
                    &scatter,
                )
            })
            .transpose()?;
        if camera_solver == SparseCameraSolver::DirectLlt && direct_assembly_workspace.is_none() {
            // Serial assembly never reads the index; release it, not just its length.
            point_cross_offsets = Vec::new();
        }
        let point_cross_count = point_cross_offsets.last().copied().unwrap_or(0);
        let direct_point_hessians = if direct_assembly_workspace.is_some() {
            vec![Default::default(); point_count]
        } else {
            Vec::new()
        };
        let values =
            if camera_solver == SparseCameraSolver::DirectLlt || exact_camera_rank_diagnostics {
                vec![0.0; reserve_direct_numeric(&mut cholesky)?]
            } else {
                Vec::new()
            };
        let placeholder_pose = RelativePose::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let direct_point_parameters = if camera_solver == SparseCameraSolver::DirectLlt
            && direct_assembly_workspace.is_some()
        {
            point_parameters.clone()
        } else {
            Vec::new()
        };
        let retained = if camera_solver == SparseCameraSolver::IterativeSchur {
            Some(SelfCalibrationRetainedWorkspace::try_new(
                camera_count,
                &effective_camera_to_group,
                camera_offsets,
                camera_dimensions,
                intrinsic_offsets,
                intrinsic_dimensions,
                &point_cross_offsets,
                point_parameters,
                model.observation_count(),
                retained_matvec,
                iterative_maximum_iterations,
                iterative_relative_tolerance,
            )?)
        } else {
            None
        };
        let recomputed_crosses = matches!(
            retained_matvec,
            RetainedSchurMatvec::Recomputed | RetainedSchurMatvec::RecomputedParallel { .. }
        );
        let parallel_candidate_evaluation = direct_assembly_workspace.is_some()
            || retained
                .as_ref()
                .and_then(|workspace| workspace.parallel_threads())
                .is_some();
        let cache_point_steps = parallel_candidate_evaluation && !recomputed_crosses;
        let point_step_cache_count = if cache_point_steps { point_count } else { 0 };
        let candidate_point_metrics =
            if parallel_candidate_evaluation && cache_point_steps && fixed.points.is_empty() {
                vec![[0.0; 3]; point_count]
            } else {
                Vec::new()
            };
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
                .and_then(|workspace| workspace.parallel_threads()),
        );
        let sparse = SparseWorkspaceCore {
            execution_decision: placeholder_decision,
            signature: problem_signature,
            point_back_substitution: if cache_point_steps {
                SparsePointBackSubstitution::Cached
            } else {
                SparsePointBackSubstitution::Recompute
            },
            layout,
            anchored_six_camera_layout: false,
            candidate_poses: vec![placeholder_pose; camera_count],
            candidate_points: vec![[0.0; 3]; point_count],
            candidate_point_metrics,
            cholesky,
            reduced_nonzero_count,
            values,
            scatter,
            exact_camera_rank_diagnostics,
            iterative_direct_fallback,
            initial_damping: camera_solver.default_initial_damping(),
            // Joint pose/point/intrinsic systems can carry model-specific
            // critical-motion ambiguities that need their native sparse gauge
            // accounting rather than calibrated-camera spectral thresholds.
            dense_rank_diagnostic: Vec::new(),
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
            point_parameters: direct_point_parameters,
            point_inverse_cross_values: PointCrossCache::new(
                point_cross_count,
                retained_cross_precision,
                !recomputed_crosses,
            ),
        };
        let backend = match retained {
            Some(retained) => SelfCalibrationBackend::Iterative(Box::new(retained)),
            None => SelfCalibrationBackend::Direct(DirectAssemblyState::new(
                direct_assembly_workspace,
                direct_point_hessians,
            )),
        };
        let mut workspace = Self {
            sparse,
            signature: self_calibration_signature,
            backend,
            candidate_cameras,
        };
        workspace.sparse.execution_decision = match automatic_plan {
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
    #[cfg(feature = "unstable-internals")]
    pub const fn camera_count(&self) -> usize {
        self.sparse.camera_count()
    }

    /// Exact point count accepted by this topology-specific workspace.
    #[cfg(feature = "unstable-internals")]
    pub const fn point_count(&self) -> usize {
        self.sparse.point_count()
    }

    /// Total number of intrinsic groups, including exactly fixed groups.
    #[cfg(feature = "unstable-internals")]
    pub fn intrinsic_group_count(&self) -> usize {
        self.signature.intrinsic_group_count()
    }

    /// Dimension of the Schur-reduced pose and intrinsic system.
    #[cfg(feature = "unstable-internals")]
    pub const fn reduced_dimension(&self) -> usize {
        self.sparse.dimension()
    }

    /// Stored scalar entries in the lower triangle of the reduced system.
    pub fn reduced_nonzero_count(&self) -> usize {
        self.sparse.reduced_nonzero_count
    }

    /// Camera-side linear solver selected for this self-calibration graph.
    pub const fn camera_solver(&self) -> SparseCameraSolver {
        self.backend.camera_solver()
    }

    /// Immutable construction-time backend and memory-planning receipt.
    pub const fn execution_decision(
        &self,
    ) -> crate::kernel::multi_view::sparse::SparseExecutionDecision {
        self.sparse.execution_decision
    }

    /// Requested retained matrix-vector policy, or `None` for direct LLT.
    #[cfg(feature = "unstable-internals")]
    pub fn retained_schur_matvec(&self) -> Option<RetainedSchurMatvec> {
        match &self.backend {
            SelfCalibrationBackend::Direct(_) => None,
            SelfCalibrationBackend::Iterative(retained) => Some(retained.requested_matvec()),
        }
    }

    /// Resolved retained worker count. `None` denotes the fused operator.
    pub fn retained_schur_parallel_threads(&self) -> Option<usize> {
        match &self.backend {
            SelfCalibrationBackend::Direct(_) => None,
            SelfCalibrationBackend::Iterative(retained) => retained.parallel_threads(),
        }
    }

    /// Explicit memory retained by the self-calibration Schur accelerator.
    pub fn retained_schur_bytes(&self) -> usize {
        let retained = match &self.backend {
            SelfCalibrationBackend::Direct(_) => 0,
            SelfCalibrationBackend::Iterative(retained) => retained.retained_bytes(),
        };
        retained.saturating_add(
            self.sparse
                .candidate_point_metrics
                .capacity()
                .saturating_mul(std::mem::size_of::<[f64; 3]>()),
        )
    }

    /// Resolved direct-Schur materialization worker count.
    pub fn direct_schur_parallel_threads(&self) -> Option<usize> {
        match &self.backend {
            SelfCalibrationBackend::Direct(assembly) => assembly.parallel_threads(),
            SelfCalibrationBackend::Iterative(_) => None,
        }
    }

    /// Additional bytes reserved for deterministic direct-Schur assembly.
    pub fn direct_schur_assembly_bytes(&self) -> usize {
        match &self.backend {
            SelfCalibrationBackend::Direct(assembly) => assembly.auxiliary_bytes(),
            SelfCalibrationBackend::Iterative(_) => 0,
        }
    }

    /// Storage precision of retained transformed point/camera cross rows.
    ///
    /// Returns `None` when no cross-row cache is allocated, including serial
    /// direct and recomputed self-calibration workspaces.
    #[cfg(feature = "unstable-internals")]
    pub fn retained_cross_precision(&self) -> Option<RetainedCrossPrecision> {
        self.sparse.point_inverse_cross_values.precision()
    }

    /// Allocated bytes used by retained transformed cross rows.
    #[cfg(feature = "unstable-internals")]
    pub fn retained_cross_row_bytes(&self) -> usize {
        self.sparse.point_inverse_cross_values.capacity_bytes()
    }

    /// Whether transformed point/parameter cross rows are retained for Schur
    /// operator use or point back-substitution.
    #[cfg(feature = "unstable-internals")]
    pub fn retains_cross_rows(&self) -> bool {
        self.sparse.point_inverse_cross_values.retains_rows()
    }

    /// Configured finite PCG iteration guard, or `None` for direct LLT.
    #[cfg(feature = "unstable-internals")]
    pub fn iterative_schur_maximum_iterations(&self) -> Option<usize> {
        match &self.backend {
            SelfCalibrationBackend::Direct(_) => None,
            SelfCalibrationBackend::Iterative(retained) => Some(retained.maximum_iterations()),
        }
    }

    /// Accepted true relative PCG residual, or `None` for direct LLT.
    pub fn iterative_schur_relative_tolerance(&self) -> Option<f64> {
        match &self.backend {
            SelfCalibrationBackend::Direct(_) => None,
            SelfCalibrationBackend::Iterative(retained) => Some(retained.relative_tolerance()),
        }
    }

    /// Whether the final reduced system receives an exact rank assessment.
    #[cfg(feature = "unstable-internals")]
    pub const fn exact_camera_rank_diagnostics(&self) -> bool {
        self.sparse.exact_camera_rank_diagnostics
    }

    /// Bytes reserved by cached point elimination factors.
    pub fn point_linearization_cache_bytes(&self) -> usize {
        self.sparse.point_linearization_cache_bytes()
    }

    /// Retained operator plus cached point-factor working set.
    pub fn retained_working_set_bytes(&self) -> usize {
        self.retained_schur_bytes() + self.point_linearization_cache_bytes()
    }

    /// Numerical sparse-LLT storage retained for direct solve or fallback.
    pub fn direct_numeric_bytes(&self) -> usize {
        self.sparse.direct_numeric_bytes()
    }

    /// Scalar reduced-pattern and numeric-scatter storage.
    #[cfg(feature = "unstable-internals")]
    pub fn direct_pattern_bytes(&self) -> usize {
        self.sparse.direct_pattern_bytes()
    }

    /// Solver-specific retained/direct-assembly and direct numerical working set.
    pub fn solver_specific_working_set_bytes(&self) -> usize {
        self.retained_working_set_bytes()
            .saturating_add(self.direct_schur_assembly_bytes())
            .saturating_add(self.direct_numeric_bytes())
    }

    /// Reserve exact-LLT storage before a retained solve may need fallback.
    ///
    /// This is only necessary for allocation-free fallback when iterative
    /// options disable exact final rank diagnostics. The default exact-rank
    /// policy allocates this storage at construction.
    #[cfg(feature = "unstable-internals")]
    pub fn try_reserve_direct_storage(&mut self) -> Result<(), SparseBundleAdjustmentError> {
        self.ensure_direct_storage()
    }

    pub(in crate::kernel::multi_view::sparse) fn ensure_direct_storage(
        &mut self,
    ) -> Result<(), SparseBundleAdjustmentError> {
        #[cfg(test)]
        if std::mem::take(&mut self.sparse.fail_next_direct_storage_for_test) {
            return Err(SparseBundleAdjustmentError::NumericalWorkspaceAllocation);
        }
        if self.sparse.cholesky.is_none() {
            self.sparse.check_lazy_direct_storage()?;
            let block_columns = radial_reduced_block_columns(
                self.sparse.point_offsets(),
                self.sparse.observation_cameras(),
                self.signature.camera_to_group(),
                self.sparse.camera_dimensions(),
                self.sparse.intrinsic_dimensions(),
                self.sparse.fixed_points(),
            );
            let mut block_offsets = Vec::with_capacity(
                self.sparse.camera_offsets().len() + self.sparse.intrinsic_offsets().len(),
            );
            block_offsets.extend_from_slice(self.sparse.camera_offsets());
            block_offsets.extend_from_slice(self.sparse.intrinsic_offsets());
            let mut block_dimensions = Vec::with_capacity(
                self.sparse.camera_dimensions().len() + self.sparse.intrinsic_dimensions().len(),
            );
            block_dimensions.extend_from_slice(self.sparse.camera_dimensions());
            block_dimensions.extend_from_slice(self.sparse.intrinsic_dimensions());
            let pattern = radial_reduced_pattern(
                &block_columns,
                self.sparse.dimension(),
                &block_offsets,
                &block_dimensions,
            );
            let reduced_nonzero_count = pattern.1.len();
            let (cholesky, scatter) = analyze_reduced_pattern(
                self.sparse.dimension(),
                pattern,
                &block_offsets,
                &block_dimensions,
            )?;
            self.sparse.reduced_nonzero_count = reduced_nonzero_count;
            self.sparse.cholesky = Some(cholesky);
            self.sparse.scatter = scatter;
        }
        self.sparse.ensure_numeric_storage()
    }

    pub(in crate::kernel::multi_view::sparse) fn matches(
        &self,
        camera_count: usize,
        intrinsic_group_count: usize,
        camera_to_group: Option<&[usize]>,
        measurements: SelfCalibratingMeasurements<'_>,
        gauge: ScaleGauge,
        fixed: FixedVariableBlocks<'_>,
    ) -> bool {
        self.sparse.signature.matches(
            camera_count,
            measurements.point_offsets(),
            measurements
                .observations()
                .iter()
                .map(|observation| observation.camera),
            gauge,
            CameraPoseVariables::RotationAndTranslation,
            fixed,
        ) && self
            .signature
            .matches(intrinsic_group_count, camera_to_group, fixed)
    }
}

#[derive(Clone, Copy)]
struct DirectAssemblyTopologySize {
    point_observations: usize,
    active_block_references: usize,
    block_count: usize,
    base_edge_count: usize,
    dimension: usize,
}

#[derive(Clone, Copy)]
struct DirectAssemblyPlan {
    threads: Option<usize>,
    estimated_bytes: usize,
}

fn resolve_direct_assembly_plan(
    policy: DirectSchurAssembly,
    camera_solver: SparseCameraSolver,
    point_cross_offsets: &[usize],
    topology: DirectAssemblyTopologySize,
) -> DirectAssemblyPlan {
    if camera_solver != SparseCameraSolver::DirectLlt {
        return DirectAssemblyPlan {
            threads: None,
            estimated_bytes: 0,
        };
    }
    let estimate = |threads| {
        let point_count = point_cross_offsets.len().saturating_sub(1);
        estimated_indexed_auxiliary_bytes(
            threads,
            point_count,
            topology.point_observations,
            topology.active_block_references,
            topology.block_count,
            topology.base_edge_count,
            topology.dimension,
        )
    };
    match policy {
        DirectSchurAssembly::Serial => DirectAssemblyPlan {
            threads: None,
            estimated_bytes: 0,
        },
        DirectSchurAssembly::Parallel { threads } => DirectAssemblyPlan {
            threads: Some(threads),
            estimated_bytes: estimate(threads),
        },
        DirectSchurAssembly::Auto => {
            let coupling_terms = point_cross_offsets
                .windows(2)
                .map(|range| {
                    let dimension = range[1].saturating_sub(range[0]);
                    dimension.saturating_mul(dimension.saturating_add(1)) / 2
                })
                .fold(0usize, usize::saturating_add);
            let threads =
                auto_direct_assembly_threads(coupling_terms, topology.point_observations, estimate);
            DirectAssemblyPlan {
                threads,
                estimated_bytes: threads.map_or(0, estimate),
            }
        }
    }
}
