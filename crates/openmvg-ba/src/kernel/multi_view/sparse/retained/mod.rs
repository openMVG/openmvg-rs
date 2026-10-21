// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained Schur camera solve by block-Jacobi preconditioned conjugate gradients. The step is
//! rejected when a camera block fails to factor or the recomputed relative residual is non-finite
//! or above the tolerance.

mod block;
mod common;
mod matvec;
mod operator;
mod parallel;
mod plan;
mod point_major;
mod recomputed;

pub(super) use block::{PackedCameraBlockFactor, PackedSymmetricCameraBlock};
pub(super) use common::{RejectedSchurSolve, SchurAttempt, SchurSolve};
#[cfg(all(test, feature = "parallel"))]
use parallel::AUTO_MAXIMUM_PARALLEL_AUXILIARY_BYTES;
#[cfg(feature = "parallel")]
pub(super) use parallel::try_build_retained_schur_pool;
#[cfg(all(test, feature = "parallel"))]
use parallel::{AUTO_MAXIMUM_THREADS, auto_available_threads};
pub(super) use parallel::{
    auto_parallel_threads, auto_self_calibration_threads, balanced_contiguous_ranges,
    deterministic_point_major_shards,
};
pub(super) use plan::{RetainedMatvecPlan, RetainedProblemKind};

#[cfg(test)]
use common::norm;
use common::{dot, dot3};
use matvec::SchurMatvecWorkspace;
use operator::{CameraBlockFactor, rejected, solve_camera_blocks};

use crate::kernel::RelativePose;
use crate::kernel::multi_view::CameraPoseVariables;
use crate::kernel::point_block::PackedPointBlock;
use crate::numerics::refine::{Loss, NoiseModel};

use super::super::model::ObservationModel;
use super::super::schur::{self, HessianAccumulator};
#[cfg(feature = "parallel")]
use super::super::schur::{
    GaugeState, PointCacheBuffers, PointCrossCacheMut, PointStepCacheBuffers, VariableLayout,
};
#[cfg(test)]
use super::api::IterativeSchurOptions;
use super::api::{RetainedSchurMatvec, SparseBundleAdjustmentError, SparseCameraSolver};
use super::direct::add_sparse_symmetric;
#[cfg(feature = "parallel")]
use super::direct_assembly::DirectSchurAssemblyWorkspace;
use super::workspace::{PointCrossCache, SparseWorkspaceCore};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const ANCHORED_CAMERA_BLOCK_DIMENSION: usize = 6;
const ANCHORED_CAMERA_BLOCK_STORAGE: usize =
    ANCHORED_CAMERA_BLOCK_DIMENSION * (ANCHORED_CAMERA_BLOCK_DIMENSION + 1) / 2;

#[derive(Clone, Copy)]
pub(super) struct RecomputedSchurContext<'a> {
    pub model: ObservationModel<'a>,
    pub poses: &'a [RelativePose],
    pub points: &'a [[f64; 3]],
    pub gauge: schur::GaugeState,
    pub loss: Loss,
    pub noise: NoiseModel<'a>,
}

#[derive(Clone, Copy)]
struct SchurOperatorContext<'a> {
    model: ObservationModel<'a>,
    recomputed: Option<RecomputedSchurContext<'a>>,
}

impl<'a> SchurOperatorContext<'a> {
    const fn retained(model: ObservationModel<'a>) -> Self {
        Self {
            model,
            recomputed: None,
        }
    }

    const fn recomputed(context: RecomputedSchurContext<'a>) -> Self {
        Self {
            model: context.model,
            recomputed: Some(context),
        }
    }
}

pub(super) trait PointCrossStorage: Sync {
    type Rows<'a>: PointCrossRows
    where
        Self: 'a;

    fn rows(&self, range: std::ops::Range<usize>) -> Self::Rows<'_>;
}

pub(super) trait PointCrossRows: Copy + Send + Sync {
    fn row(self, index: usize) -> [f64; 3];
    fn len(self) -> usize;
}

impl PointCrossStorage for [[f64; 3]] {
    type Rows<'a> = &'a [[f64; 3]];

    #[inline(always)]
    fn rows(&self, range: std::ops::Range<usize>) -> Self::Rows<'_> {
        &self[range]
    }
}

impl PointCrossRows for &[[f64; 3]] {
    #[inline(always)]
    fn row(self, index: usize) -> [f64; 3] {
        self[index]
    }

    #[inline(always)]
    fn len(self) -> usize {
        self.len()
    }
}

impl PointCrossStorage for [[f32; 3]] {
    type Rows<'a> = &'a [[f32; 3]];

    #[inline(always)]
    fn rows(&self, range: std::ops::Range<usize>) -> Self::Rows<'_> {
        &self[range]
    }
}

impl PointCrossRows for &[[f32; 3]] {
    #[inline(always)]
    fn row(self, index: usize) -> [f64; 3] {
        self[index].map(f64::from)
    }

    #[inline(always)]
    fn len(self) -> usize {
        self.len()
    }
}

fn pcg_iteration_limit(camera_dimension: usize, configured_maximum: usize) -> usize {
    configured_maximum.min(camera_dimension)
}

type PackedCameraBlock =
    PackedSymmetricCameraBlock<ANCHORED_CAMERA_BLOCK_DIMENSION, ANCHORED_CAMERA_BLOCK_STORAGE>;

pub(super) fn estimated_calibrated_retained_bytes(
    camera_count: usize,
    point_count: usize,
    dimension: usize,
    observation_count: usize,
    point_cache_bytes: usize,
    recomputed: bool,
) -> usize {
    let base = camera_count
        .saturating_mul(
            2 * std::mem::size_of::<PackedCameraBlock>() + std::mem::size_of::<CameraBlockFactor>(),
        )
        .saturating_add(point_count.saturating_mul(std::mem::size_of::<PackedPointBlock>()))
        .saturating_add(dimension.saturating_mul(6 * std::mem::size_of::<f64>()))
        .saturating_add(point_cache_bytes);
    let worker_bytes = matvec::calibrated_point_major_worker_bytes(camera_count, dimension);
    let parallel_threads = auto_parallel_threads(observation_count, worker_bytes);
    let parallel_shards = parallel_threads
        .and_then(|_| deterministic_point_major_shards(observation_count, worker_bytes));
    let parallel = parallel_shards.map_or(0, |shards| {
        shards
            .saturating_mul(
                worker_bytes.saturating_add(dimension.saturating_mul(std::mem::size_of::<f64>())),
            )
            .saturating_add((shards + 1) * std::mem::size_of::<usize>())
    });
    if recomputed && parallel_threads.is_none() {
        base.saturating_add(dimension.saturating_mul(
            3 * std::mem::size_of::<f64>()
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<bool>(),
        ))
    } else {
        base.saturating_add(parallel)
    }
}

pub(super) fn automatic_recomputed_matvec(
    camera_count: usize,
    dimension: usize,
    observation_count: usize,
) -> (RetainedSchurMatvec, Option<usize>) {
    let worker_bytes = matvec::calibrated_point_major_worker_bytes(camera_count, dimension);
    auto_parallel_threads(observation_count, worker_bytes).map_or(
        (RetainedSchurMatvec::Recomputed, None),
        |threads| {
            (
                RetainedSchurMatvec::RecomputedParallel { threads },
                deterministic_point_major_shards(observation_count, worker_bytes),
            )
        },
    )
}

pub(super) struct RetainedBlockHessian<'a> {
    camera_blocks: &'a mut [PackedCameraBlock],
    parameter_cameras: &'a [usize],
    camera_offsets: &'a [usize],
    valid: bool,
}

impl HessianAccumulator for RetainedBlockHessian<'_> {
    fn clear(&mut self) {
        for block in &mut *self.camera_blocks {
            block.values.fill(0.0);
        }
        self.valid = true;
    }

    #[inline]
    fn add_symmetric(&mut self, row: usize, column: usize, value: f64) {
        let Some((&row_camera, &column_camera)) = self
            .parameter_cameras
            .get(row)
            .zip(self.parameter_cameras.get(column))
        else {
            self.valid = false;
            return;
        };
        if row_camera != column_camera {
            self.valid = false;
            return;
        }
        let offset = self.camera_offsets[row_camera];
        self.camera_blocks[row_camera].add(row - offset, column - offset, value);
    }

    #[inline]
    fn add_gram(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        lower: bool,
        left: [&[f64]; 2],
        right: [&[f64]; 2],
    ) {
        // Runs are consecutive, so their end points decide the owning camera.
        let camera = |parameters: &[usize]| {
            let first = *parameters.first()?;
            let camera = *self.parameter_cameras.get(first)?;
            (*self.parameter_cameras.get(*parameters.last()?)? == camera).then_some((camera, first))
        };
        // A lower run is also its own column run.
        let row_run = camera(rows);
        let column_run = if lower { row_run } else { camera(columns) };
        let (Some((row_camera, row)), Some((column_camera, column))) = (row_run, column_run) else {
            return schur::add_gram_scalar(self, rows, columns, lower, left, right);
        };
        if row_camera != column_camera {
            self.valid = false;
            return;
        }
        let offset = self.camera_offsets[row_camera];
        self.camera_blocks[row_camera].add_gram(row - offset, column - offset, lower, left, right);
    }

    fn is_finite(&self) -> bool {
        self.valid
            && self
                .camera_blocks
                .iter()
                .all(|block| block.values.iter().all(|value| value.is_finite()))
    }
}

pub(super) struct RetainedSchurWorkspace {
    camera_blocks: Vec<PackedCameraBlock>,
    diagonal_point_coupling: Vec<PackedCameraBlock>,
    camera_factors: Vec<CameraBlockFactor>,
    point_hessians: Vec<PackedPointBlock>,
    parameter_cameras: Vec<usize>,
    solution: Vec<f64>,
    term_right: Vec<f64>,
    term: Vec<f64>,
    next_right: Vec<f64>,
    product: Vec<f64>,
    matvec_policy: RetainedSchurMatvec,
    matvec: SchurMatvecWorkspace,
    #[cfg(feature = "parallel")]
    fallback_materializer: Option<Box<DirectSchurAssemblyWorkspace>>,
    maximum_iterations: usize,
    relative_tolerance_bits: u64,
}

impl RetainedSchurWorkspace {
    pub(super) const fn camera_solver(&self) -> SparseCameraSolver {
        SparseCameraSolver::IterativeSchur
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_iterative(
        camera_count: usize,
        point_count: usize,
        dimension: usize,
        camera_offsets: &[usize],
        camera_dimensions: &[usize],
        model: ObservationModel<'_>,
        matvec_policy: RetainedSchurMatvec,
        logical_shards_override: Option<usize>,
        maximum_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Self::new(
            camera_count,
            point_count,
            dimension,
            camera_offsets,
            camera_dimensions,
            model,
            matvec_policy,
            logical_shards_override,
            maximum_iterations,
            relative_tolerance.to_bits(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        camera_count: usize,
        point_count: usize,
        dimension: usize,
        camera_offsets: &[usize],
        camera_dimensions: &[usize],
        model: ObservationModel<'_>,
        matvec_policy: RetainedSchurMatvec,
        logical_shards_override: Option<usize>,
        maximum_iterations: usize,
        relative_tolerance_bits: u64,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let mut parameter_cameras = vec![0; dimension];
        for camera in 0..camera_count {
            let offset = camera_offsets[camera];
            let end = offset + camera_dimensions[camera];
            parameter_cameras[offset..end].fill(camera);
        }
        let matvec = SchurMatvecWorkspace::try_new(
            matvec_policy,
            logical_shards_override,
            model,
            camera_count,
            dimension,
        )?;
        Ok(Self {
            camera_blocks: vec![PackedCameraBlock::default(); camera_count],
            diagonal_point_coupling: vec![PackedCameraBlock::default(); camera_count],
            camera_factors: vec![CameraBlockFactor::default(); camera_count],
            point_hessians: vec![PackedPointBlock::default(); point_count],
            parameter_cameras,
            solution: vec![0.0; dimension],
            term_right: vec![0.0; dimension],
            term: vec![0.0; dimension],
            next_right: vec![0.0; dimension],
            product: vec![0.0; dimension],
            matvec_policy,
            matvec,
            #[cfg(feature = "parallel")]
            fallback_materializer: None,
            maximum_iterations,
            relative_tolerance_bits,
        })
    }

    pub fn assembly_buffers<'a>(
        &'a mut self,
        camera_offsets: &'a [usize],
    ) -> (RetainedBlockHessian<'a>, &'a mut [PackedPointBlock]) {
        let Self {
            camera_blocks,
            point_hessians,
            parameter_cameras,
            ..
        } = self;
        (
            RetainedBlockHessian {
                camera_blocks,
                parameter_cameras,
                camera_offsets,
                valid: true,
            },
            point_hessians,
        )
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_retained_parallel(
        &mut self,
        model: ObservationModel<'_>,
        poses: &[RelativePose],
        points: &[[f64; 3]],
        gauge: GaugeState,
        loss: Loss,
        noise: NoiseModel<'_>,
        damping: f64,
        sparse: &mut SparseWorkspaceCore,
    ) -> Option<schur::Assembly> {
        let Self {
            camera_blocks,
            point_hessians,
            parameter_cameras,
            matvec,
            ..
        } = self;
        let workspace = match matvec {
            SchurMatvecWorkspace::PointMajorParallel(workspace) => workspace,
            SchurMatvecWorkspace::RecomputedParallel(workspace) => &mut workspace.shared,
            SchurMatvecWorkspace::Fused | SchurMatvecWorkspace::Recomputed(_) => return None,
        };
        let pose_variables = sparse.signature.pose_variables();
        Some(workspace.assemble_retained(
            model,
            poses,
            points,
            gauge,
            loss,
            noise,
            damping,
            parameter_cameras,
            sparse.layout.camera_offsets(),
            sparse.layout.camera_dimensions(),
            &sparse.point_cross_offsets,
            camera_blocks,
            point_hessians,
            &mut sparse.gradient,
            &mut sparse.full_gradient,
            &mut sparse.diagonal,
            &mut sparse.point_cache_valid,
            &mut sparse.point_inverse_gradients,
            &mut sparse.point_damping_diagonals,
            sparse.point_inverse_cross_values.as_mut(),
            pose_variables,
            sparse.signature.pose_masks(),
        ))
    }

    pub fn retained_bytes(&self) -> usize {
        self.camera_blocks.capacity() * std::mem::size_of::<PackedCameraBlock>()
            + self.diagonal_point_coupling.capacity() * std::mem::size_of::<PackedCameraBlock>()
            + self.camera_factors.capacity() * std::mem::size_of::<CameraBlockFactor>()
            + self.point_hessians.capacity() * std::mem::size_of::<PackedPointBlock>()
            + self.parameter_cameras.capacity() * std::mem::size_of::<usize>()
            + (self.solution.capacity()
                + self.term_right.capacity()
                + self.term.capacity()
                + self.next_right.capacity()
                + self.product.capacity())
                * std::mem::size_of::<f64>()
            + self.matvec.retained_bytes()
            + {
                #[cfg(feature = "parallel")]
                {
                    self.fallback_materializer
                        .as_ref()
                        .map_or(0, |materializer| materializer.auxiliary_bytes())
                }
                #[cfg(not(feature = "parallel"))]
                {
                    0
                }
            }
    }

    #[cfg(feature = "unstable-internals")]
    pub const fn matvec_policy(&self) -> RetainedSchurMatvec {
        self.matvec_policy
    }

    pub const fn uses_recomputed_cross_rows(&self) -> bool {
        matches!(
            self.matvec_policy,
            RetainedSchurMatvec::Recomputed | RetainedSchurMatvec::RecomputedParallel { .. }
        )
    }

    pub fn prepare_recomputed_point_blocks(&mut self, fixed_points: &[bool]) -> bool {
        if !self.uses_recomputed_cross_rows() {
            return true;
        }
        #[cfg(feature = "parallel")]
        if let SchurMatvecWorkspace::RecomputedParallel(workspace) = &self.matvec {
            return workspace.shared.pool.install(|| {
                self.point_hessians
                    .par_iter_mut()
                    .enumerate()
                    .all(|(point, hessian)| {
                        (!fixed_points.is_empty() && fixed_points[point])
                            || hessian.invert_in_place()
                    })
            });
        }
        self.point_hessians
            .iter_mut()
            .enumerate()
            .all(|(point, hessian)| {
                (!fixed_points.is_empty() && fixed_points[point]) || hessian.invert_in_place()
            })
    }

    pub const fn parallel_threads(&self) -> Option<usize> {
        self.matvec.parallel_threads()
    }

    #[cfg(feature = "parallel")]
    pub(super) fn ensure_fallback_materializer(
        &mut self,
        sparse: &SparseWorkspaceCore,
    ) -> Result<(), SparseBundleAdjustmentError> {
        let Some(threads) = self.parallel_threads() else {
            return Ok(());
        };
        if self.fallback_materializer.is_none() {
            let (reduced_dimension, reduced_value_count) =
                super::workspace::reduced_pattern_size(sparse.cholesky.as_ref());
            self.fallback_materializer = Some(Box::new(DirectSchurAssemblyWorkspace::try_new(
                threads,
                sparse.signature.point_pose_offsets(),
                sparse.signature.point_poses(),
                sparse.layout.camera_dimensions(),
                &sparse.point_cross_offsets,
                reduced_dimension,
                reduced_value_count,
                &sparse.scatter,
            )?));
        }
        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    #[cfg(feature = "unstable-internals")]
    pub(super) const fn ensure_fallback_materializer(
        &mut self,
        _sparse: &SparseWorkspaceCore,
    ) -> Result<(), SparseBundleAdjustmentError> {
        Ok(())
    }

    #[cfg(feature = "parallel")]
    pub(super) fn assemble_direct_parallel(
        &mut self,
        problem: schur::BaProblemView<'_>,
        policy: schur::LinearizationPolicy<'_>,
        sparse: &mut SparseWorkspaceCore,
    ) -> Option<schur::Assembly> {
        self.ensure_fallback_materializer(sparse).ok()?;
        let Self {
            point_hessians,
            fallback_materializer,
            ..
        } = self;
        let direct = fallback_materializer.as_deref_mut()?;
        Some(super::calibrated::assemble_sparse_direct(
            problem,
            policy,
            sparse,
            direct,
            point_hessians,
        ))
    }

    #[cfg(feature = "unstable-internals")]
    pub const fn iterative_maximum_iterations(&self) -> Option<usize> {
        Some(self.maximum_iterations)
    }

    pub const fn iterative_relative_tolerance(&self) -> Option<f64> {
        Some(f64::from_bits(self.relative_tolerance_bits))
    }

    #[cfg(feature = "parallel")]
    pub fn install_parallel<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> Option<R> {
        self.matvec.install_parallel(operation)
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_recomputed_candidates(
        &mut self,
        context: RecomputedSchurContext<'_>,
        damping: f64,
        camera_offsets: &[usize],
        camera_dimensions: &[usize],
        camera_step: &[f64],
        candidate_poses: &[RelativePose],
        candidate_points: &mut [[f64; 3]],
        point_metrics: &mut [[f64; 3]],
        pose_variables: CameraPoseVariables,
        pose_masks: &[crate::PoseVariables],
    ) -> Option<bool> {
        let SchurMatvecWorkspace::RecomputedParallel(workspace) = &mut self.matvec else {
            return None;
        };
        Some(workspace.evaluate_candidates(
            context,
            damping,
            camera_offsets,
            camera_dimensions,
            camera_step,
            candidate_poses,
            candidate_points,
            point_metrics,
            pose_variables,
            pose_masks,
        ))
    }

    pub fn solve(
        &mut self,
        model: ObservationModel<'_>,
        sparse: &SparseWorkspaceCore,
        relative_tolerance: Option<f64>,
    ) -> SchurAttempt {
        self.solve_with_context(
            SchurOperatorContext::retained(model),
            sparse,
            relative_tolerance,
        )
    }

    pub fn solve_recomputed(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        relative_tolerance: Option<f64>,
    ) -> SchurAttempt {
        self.solve_with_context(
            SchurOperatorContext::recomputed(context),
            sparse,
            relative_tolerance,
        )
    }

    fn solve_with_context(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
        relative_tolerance: Option<f64>,
    ) -> SchurAttempt {
        let is_confirmation = relative_tolerance.is_some();
        self.solve_pcg(
            context,
            sparse,
            self.maximum_iterations,
            relative_tolerance.unwrap_or_else(|| f64::from_bits(self.relative_tolerance_bits)),
            is_confirmation,
        )
    }

    fn solve_pcg(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
        maximum_iterations: usize,
        relative_tolerance: f64,
        allow_finite_precision_iterations: bool,
    ) -> SchurAttempt {
        // Exact-arithmetic CG terminates within the camera dimension. A strict
        // confirmation may need additional iterations after finite-precision
        // loss of conjugacy, still bounded by the configured finite guard.
        let iteration_limit = if allow_finite_precision_iterations {
            maximum_iterations
        } else {
            pcg_iteration_limit(self.solution.len(), maximum_iterations)
        };
        self.solve_pcg_iterations(context, sparse, iteration_limit, relative_tolerance)
    }

    fn solve_pcg_iterations(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
        maximum_iterations: usize,
        relative_tolerance: f64,
    ) -> SchurAttempt {
        if !self.factor_camera_blocks(context, sparse) {
            return rejected(0, None);
        }
        let matvec = &mut self.matvec;
        let camera_blocks = &self.camera_blocks;
        let point_hessians = &self.point_hessians;
        let camera_factors = &self.camera_factors;
        let camera_offsets = sparse.layout.camera_offsets();
        let camera_dimensions = sparse.layout.camera_dimensions();
        super::pcg::solve(
            &sparse.gradient,
            super::pcg::PcgBuffers {
                solution: &mut self.solution,
                residual: &mut self.term_right,
                preconditioned: &mut self.term,
                direction: &mut self.next_right,
                product: &mut self.product,
            },
            super::pcg::PcgOptions {
                maximum_iterations,
                relative_tolerance,
                require_bounded_residual: true,
            },
            |input, output| {
                matvec.apply_reduced_system(
                    context,
                    sparse,
                    camera_blocks,
                    point_hessians,
                    input,
                    output,
                );
            },
            |residual, preconditioned| {
                solve_camera_blocks(
                    camera_factors,
                    residual,
                    preconditioned,
                    camera_offsets,
                    camera_dimensions,
                );
            },
        )
    }

    pub fn copy_step_to(&self, step: &mut [f64]) {
        step.copy_from_slice(&self.solution);
    }

    /// Predicted reduction of the damped quadratic after point elimination.
    ///
    /// This includes the point-only constant removed by the Schur complement,
    /// so it can be compared with the complete nonlinear objective reduction.
    pub fn predicted_reduction(
        &mut self,
        model: ObservationModel<'_>,
        sparse: &SparseWorkspaceCore,
        camera_step: &[f64],
    ) -> Option<f64> {
        let point_reduction = sparse
            .point_cache_valid
            .iter()
            .zip(&sparse.point_inverse_gradients)
            .zip(&self.point_hessians)
            .filter(|((valid, _), _)| **valid)
            .map(|((_, inverse_gradient), hessian)| {
                0.5 * hessian.bilinear(inverse_gradient, inverse_gradient)
            })
            .sum::<f64>();
        self.predicted_reduction_with_context(
            SchurOperatorContext::retained(model),
            sparse,
            camera_step,
            point_reduction,
        )
    }

    pub fn predicted_reduction_recomputed(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        camera_step: &[f64],
        point_reduction: f64,
    ) -> Option<f64> {
        self.predicted_reduction_with_context(
            SchurOperatorContext::recomputed(context),
            sparse,
            camera_step,
            point_reduction,
        )
    }

    fn predicted_reduction_with_context(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
        camera_step: &[f64],
        point_reduction: f64,
    ) -> Option<f64> {
        self.matvec.apply_reduced_system(
            context,
            sparse,
            &self.camera_blocks,
            &self.point_hessians,
            camera_step,
            &mut self.product,
        );
        let camera_reduction =
            -dot(&sparse.gradient, camera_step) - 0.5 * dot(camera_step, &self.product);
        let predicted = point_reduction + camera_reduction;
        (predicted.is_finite() && predicted > 0.0).then_some(predicted)
    }

    /// Form the exact retained Schur matrix only when LLT or final rank
    /// diagnostics require it.
    pub fn materialize_schur(
        &mut self,
        model: ObservationModel<'_>,
        sparse: &mut SparseWorkspaceCore,
    ) -> bool {
        if !self.seed_materialized_camera_blocks(sparse) {
            return false;
        }

        #[cfg(feature = "parallel")]
        if self.parallel_threads().is_some() {
            if self.ensure_fallback_materializer(sparse).is_err() {
                return false;
            }
            let Some(materializer) = self.fallback_materializer.as_deref_mut() else {
                return false;
            };
            return materializer.materialize_cached(
                &sparse.scatter,
                &mut sparse.values,
                &self.point_hessians,
                &sparse.point_inverse_cross_values,
            );
        }

        self.materialize_point_coupling_serial(model, sparse)
    }

    fn seed_materialized_camera_blocks(&self, sparse: &mut SparseWorkspaceCore) -> bool {
        sparse.values.fill(0.0);
        for camera in 0..sparse.signature.camera_count() {
            let dimension = sparse.layout.camera_dimensions()[camera];
            let offset = sparse.layout.camera_offsets()[camera];
            for row in 0..dimension {
                for column in 0..=row {
                    if !add_sparse_symmetric(
                        sparse,
                        offset + row,
                        offset + column,
                        self.camera_blocks[camera].get(row, column),
                    ) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn materialize_point_coupling_serial(
        &self,
        _model: ObservationModel<'_>,
        sparse: &mut SparseWorkspaceCore,
    ) -> bool {
        for point in 0..sparse.signature.point_count() {
            if sparse.point_cross_offsets[point] == sparse.point_cross_offsets[point + 1] {
                continue;
            }
            let mut active_count = 0;
            for &camera in &sparse.signature.point_poses()[sparse.signature.point_pose_offsets()
                [point]
                ..sparse.signature.point_pose_offsets()[point + 1]]
            {
                let camera = camera as usize;
                let offset = sparse.layout.camera_offsets()[camera];
                for local in 0..sparse.layout.camera_dimensions()[camera] {
                    sparse.active_parameters[active_count] = offset + local;
                    active_count += 1;
                }
            }
            let range = sparse.point_cross_offsets[point]..sparse.point_cross_offsets[point + 1];
            if range.len() != active_count {
                return false;
            }
            for row in 0..active_count {
                for column in 0..=row {
                    let row_cross = sparse.point_inverse_cross_values.get(range.start + row);
                    let column_cross = sparse.point_inverse_cross_values.get(range.start + column);
                    let value = -self.point_hessians[point].bilinear(&row_cross, &column_cross);
                    let row_parameter = sparse.active_parameters[row];
                    let column_parameter = sparse.active_parameters[column];
                    if !add_sparse_symmetric(sparse, row_parameter, column_parameter, value) {
                        return false;
                    }
                }
            }
        }
        sparse.values.iter().all(|value| value.is_finite())
    }

    #[cfg(all(test, feature = "parallel"))]
    fn materialize_schur_serial_for_test(
        &self,
        model: ObservationModel<'_>,
        sparse: &mut SparseWorkspaceCore,
    ) -> bool {
        self.seed_materialized_camera_blocks(sparse)
            && self.materialize_point_coupling_serial(model, sparse)
    }

    fn factor_camera_blocks(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
    ) -> bool {
        if !self.build_diagonal_point_coupling(context, sparse) {
            return false;
        }
        for (camera, factor) in self
            .camera_factors
            .iter_mut()
            .enumerate()
            .take(sparse.signature.camera_count())
        {
            let dimension = sparse.layout.camera_dimensions()[camera];
            if dimension > 0 {
                let mut preconditioner = self.camera_blocks[camera];
                for (value, diagonal_coupling) in preconditioner
                    .values
                    .iter_mut()
                    .zip(&self.diagonal_point_coupling[camera].values)
                {
                    *value -= diagonal_coupling;
                }
                if !factor.factor(&preconditioner, dimension) {
                    return false;
                }
            }
        }
        true
    }

    fn build_diagonal_point_coupling(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
    ) -> bool {
        if let SchurMatvecWorkspace::Recomputed(workspace) = &mut self.matvec {
            let Some(recomputed) = context.recomputed else {
                return false;
            };
            return workspace.build_diagonal_point_coupling(
                recomputed,
                sparse,
                &self.point_hessians,
                &mut self.diagonal_point_coupling,
            );
        }
        if let SchurMatvecWorkspace::RecomputedParallel(workspace) = &mut self.matvec {
            let Some(recomputed) = context.recomputed else {
                return false;
            };
            return workspace.build_diagonal_point_coupling(
                recomputed,
                sparse,
                &self.point_hessians,
                &mut self.diagonal_point_coupling,
            );
        }
        #[cfg(feature = "parallel")]
        if let SchurMatvecWorkspace::PointMajorParallel(workspace) = &mut self.matvec {
            return workspace.build_diagonal_point_coupling(
                sparse,
                &self.point_hessians,
                &mut self.diagonal_point_coupling,
            );
        }
        for block in &mut self.diagonal_point_coupling {
            block.values.fill(0.0);
        }
        for point in 0..sparse.signature.point_count() {
            if sparse.point_cross_offsets[point] == sparse.point_cross_offsets[point + 1] {
                continue;
            }
            let mut cross_offset = sparse.point_cross_offsets[point];
            for &camera in &sparse.signature.point_poses()[sparse.signature.point_pose_offsets()
                [point]
                ..sparse.signature.point_pose_offsets()[point + 1]]
            {
                let camera = camera as usize;
                let dimension = sparse.layout.camera_dimensions()[camera];
                for row in 0..dimension {
                    for column in 0..=row {
                        let left = sparse.point_inverse_cross_values.get(cross_offset + row);
                        let right = sparse.point_inverse_cross_values.get(cross_offset + column);
                        let value = self.point_hessians[point].bilinear(&left, &right);
                        self.diagonal_point_coupling[camera].add(row, column, value);
                    }
                }
                cross_offset += dimension;
            }
            if cross_offset != sparse.point_cross_offsets[point + 1] {
                return false;
            }
        }
        self.diagonal_point_coupling
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
    }

    #[cfg(test)]
    fn true_residual_ratio(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
        initial_norm: f64,
    ) -> f64 {
        self.matvec.apply_reduced_system(
            context,
            sparse,
            &self.camera_blocks,
            &self.point_hessians,
            &self.solution,
            &mut self.next_right,
        );
        for (residual, gradient) in self.next_right.iter_mut().zip(&sparse.gradient) {
            *residual = -*gradient - *residual;
        }
        norm(&self.next_right) / initial_norm
    }
}

#[cfg(test)]
mod tests;
