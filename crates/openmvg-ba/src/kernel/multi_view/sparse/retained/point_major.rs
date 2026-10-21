// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lock-free point-major retained operator and worker reduction.

use super::operator::*;
#[cfg(feature = "parallel")]
use super::recomputed::RecomputedMatvecWorkspace;
use super::*;

pub(super) struct PointMajorParallelWorkspace {
    pub(super) threads: usize,
    pub(super) dimension: usize,
    pub(super) point_ranges: Vec<usize>,
    pub(super) partial_outputs: Vec<f64>,
    #[cfg(feature = "parallel")]
    pub(super) assembly_scratch: Vec<RetainedAssemblyScratch>,
    #[cfg(feature = "parallel")]
    pub(super) pool: rayon::ThreadPool,
}

#[cfg(feature = "parallel")]
pub(super) struct RetainedAssemblyScratch {
    pub(super) camera_blocks: Vec<PackedCameraBlock>,
    pub(super) gradient: Vec<f64>,
    pub(super) full_gradient: Vec<f64>,
    pub(super) diagonal: Vec<f64>,
    pub(super) recomputed: RecomputedMatvecWorkspace,
}

#[cfg(feature = "parallel")]
impl RetainedAssemblyScratch {
    pub(super) fn try_new(
        camera_count: usize,
        dimension: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        Ok(Self {
            camera_blocks: vec![PackedCameraBlock::default(); camera_count],
            gradient: vec![0.0; dimension],
            full_gradient: vec![0.0; dimension],
            diagonal: vec![0.0; dimension],
            recomputed: RecomputedMatvecWorkspace::new(dimension)?,
        })
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.camera_blocks.capacity() * std::mem::size_of::<PackedCameraBlock>()
            + (self.gradient.capacity() + self.full_gradient.capacity() + self.diagonal.capacity())
                * std::mem::size_of::<f64>()
            + self.recomputed.retained_bytes()
    }
}

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
pub(super) struct RetainedAssemblyContext<'a> {
    pub(super) model: ObservationModel<'a>,
    pub(super) poses: &'a [RelativePose],
    pub(super) gauge: GaugeState,
    pub(super) loss: Loss,
    pub(super) noise: NoiseModel<'a>,
    pub(super) damping: f64,
    pub(super) dimension: usize,
    pub(super) parameter_cameras: &'a [usize],
    pub(super) camera_offsets: &'a [usize],
    pub(super) camera_dimensions: &'a [usize],
    pub(super) pose_masks: &'a [crate::PoseVariables],
    pub(super) point_cross_offsets: &'a [usize],
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn assemble_retained_partitions<const TRANSLATION_ONLY: bool>(
    context: RetainedAssemblyContext<'_>,
    point_ranges: &[usize],
    scratch: &mut [RetainedAssemblyScratch],
    points: &[[f64; 3]],
    point_start: usize,
    point_hessians: &mut [PackedPointBlock],
    point_cache_valid: &mut [bool],
    point_inverse_gradients: &mut [[f64; 3]],
    point_damping_diagonals: &mut [[f64; 3]],
    point_inverse_cross_values: PointCrossCacheMut<'_>,
    cross_start: usize,
) -> schur::Assembly {
    if scratch.len() == 1 {
        let worker = &mut scratch[0];
        let mut hessian = RetainedBlockHessian {
            camera_blocks: &mut worker.camera_blocks,
            parameter_cameras: context.parameter_cameras,
            camera_offsets: context.camera_offsets,
            valid: true,
        };
        let point_cache = PointCacheBuffers {
            point_steps: (!point_cache_valid.is_empty()).then_some(PointStepCacheBuffers {
                valid: point_cache_valid,
                inverse_gradients: point_inverse_gradients,
                damping_diagonals: point_damping_diagonals,
            }),
            cross_offsets: context.point_cross_offsets,
            inverse_cross_values: point_inverse_cross_values,
            point_offset: point_start,
            cross_offset: cross_start,
            damped_hessians: Some(point_hessians),
        };
        return if TRANSLATION_ONLY {
            schur::assemble_retained_range_translation_only(
                context.model,
                context.poses,
                points,
                point_start,
                &[],
                &[],
                1.0,
                context.gauge,
                context.loss,
                context.noise,
                context.damping,
                context.dimension,
                &mut hessian,
                &mut worker.gradient,
                &mut worker.full_gradient,
                &mut worker.diagonal,
                &mut worker.recomputed.cross,
                &mut worker.recomputed.active_parameters,
                &mut worker.recomputed.parameter_marks,
                VariableLayout::calibrated_translation_only(
                    context.camera_offsets,
                    context.camera_dimensions,
                )
                .with_pose_masks(context.pose_masks),
                point_cache,
            )
        } else {
            schur::assemble_retained_range(
                context.model,
                context.poses,
                points,
                point_start,
                &[],
                &[],
                1.0,
                context.gauge,
                context.loss,
                context.noise,
                context.damping,
                context.dimension,
                &mut hessian,
                &mut worker.gradient,
                &mut worker.full_gradient,
                &mut worker.diagonal,
                &mut worker.recomputed.cross,
                &mut worker.recomputed.active_parameters,
                &mut worker.recomputed.parameter_marks,
                VariableLayout::calibrated(context.camera_offsets, context.camera_dimensions)
                    .with_pose_masks(context.pose_masks),
                point_cache,
            )
        };
    }

    let middle = scratch.len() / 2;
    let point_split = point_ranges[middle];
    let local_point_split = point_split - point_start;
    let cross_split = context.point_cross_offsets[point_split];
    let local_cross_split = cross_split - cross_start;
    let (left_scratch, right_scratch) = scratch.split_at_mut(middle);
    let (left_points, right_points) = points.split_at(local_point_split);
    let (left_hessians, right_hessians) = point_hessians.split_at_mut(local_point_split);
    let cache_split = if point_cache_valid.is_empty() {
        0
    } else {
        local_point_split
    };
    let (left_valid, right_valid) = point_cache_valid.split_at_mut(cache_split);
    let (left_gradients, right_gradients) = point_inverse_gradients.split_at_mut(cache_split);
    let (left_diagonals, right_diagonals) = point_damping_diagonals.split_at_mut(cache_split);
    let (left_crosses, right_crosses) = point_inverse_cross_values.split_at_mut(local_cross_split);
    let (left, right) = rayon::join(
        || {
            assemble_retained_partitions::<TRANSLATION_ONLY>(
                context,
                &point_ranges[..=middle],
                left_scratch,
                left_points,
                point_start,
                left_hessians,
                left_valid,
                left_gradients,
                left_diagonals,
                left_crosses,
                cross_start,
            )
        },
        || {
            assemble_retained_partitions::<TRANSLATION_ONLY>(
                context,
                &point_ranges[middle..],
                right_scratch,
                right_points,
                point_split,
                right_hessians,
                right_valid,
                right_gradients,
                right_diagonals,
                right_crosses,
                cross_split,
            )
        },
    );
    merge_assemblies(left, right)
}

#[cfg(feature = "parallel")]
fn merge_assemblies(mut left: schur::Assembly, right: schur::Assembly) -> schur::Assembly {
    left.valid &= right.valid;
    left.weak_point_count += right.weak_point_count;
    left.minimum_point_rcond = left.minimum_point_rcond.min(right.minimum_point_rcond);
    left.scaled_gradient_inf_norm = left
        .scaled_gradient_inf_norm
        .max(right.scaled_gradient_inf_norm);
    left.point_model_reduction += right.point_model_reduction;
    left.cost += right.cost;
    left.squared_residual += right.squared_residual;
    left
}

impl PointMajorParallelWorkspace {
    pub(super) fn try_new(
        threads: usize,
        logical_shards: usize,
        _camera_count: usize,
        dimension: usize,
        point_offsets: &[usize],
    ) -> Result<Self, SparseBundleAdjustmentError> {
        #[cfg(feature = "parallel")]
        let pool = try_build_retained_schur_pool(threads)?;
        let point_ranges = balanced_contiguous_ranges(point_offsets, logical_shards);
        let worker_count = point_ranges.len().saturating_sub(1);
        let partial_len = worker_count
            .checked_mul(dimension)
            .ok_or(SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
        Ok(Self {
            threads,
            dimension,
            point_ranges,
            partial_outputs: vec![0.0; partial_len],
            #[cfg(feature = "parallel")]
            assembly_scratch: (0..worker_count)
                .map(|_| RetainedAssemblyScratch::try_new(_camera_count, dimension))
                .collect::<Result<_, _>>()?,
            #[cfg(feature = "parallel")]
            pool,
        })
    }

    pub(super) fn retained_bytes(&self) -> usize {
        let bytes = self.partial_outputs.capacity() * std::mem::size_of::<f64>()
            + self.point_ranges.capacity() * std::mem::size_of::<usize>();
        #[cfg(feature = "parallel")]
        let bytes = bytes
            + self
                .assembly_scratch
                .iter()
                .map(RetainedAssemblyScratch::retained_bytes)
                .sum::<usize>();
        bytes
    }

    #[cfg(feature = "parallel")]
    pub(super) fn build_diagonal_point_coupling(
        &mut self,
        sparse: &SparseWorkspaceCore,
        point_hessians: &[PackedPointBlock],
        output: &mut [PackedCameraBlock],
    ) -> bool {
        let valid = self.pool.install(|| {
            self.point_ranges
                .par_windows(2)
                .zip(self.assembly_scratch.par_iter_mut())
                .map(|(range, scratch)| {
                    for block in &mut scratch.camera_blocks {
                        block.values.fill(0.0);
                    }
                    for (point, point_hessian) in point_hessians
                        .iter()
                        .enumerate()
                        .take(range[1])
                        .skip(range[0])
                    {
                        if sparse.point_cross_offsets[point]
                            == sparse.point_cross_offsets[point + 1]
                        {
                            continue;
                        }
                        let mut cross_offset = sparse.point_cross_offsets[point];
                        for observation in sparse.signature.point_pose_offsets()[point]
                            ..sparse.signature.point_pose_offsets()[point + 1]
                        {
                            let camera = sparse.signature.point_poses()[observation] as usize;
                            let dimension = sparse.layout.camera_dimensions()[camera];
                            for row in 0..dimension {
                                for column in 0..=row {
                                    let left =
                                        sparse.point_inverse_cross_values.get(cross_offset + row);
                                    let right = sparse
                                        .point_inverse_cross_values
                                        .get(cross_offset + column);
                                    let value = point_hessian.bilinear(&left, &right);
                                    scratch.camera_blocks[camera].add(row, column, value);
                                }
                            }
                            cross_offset += dimension;
                        }
                        if cross_offset != sparse.point_cross_offsets[point + 1] {
                            return false;
                        }
                    }
                    true
                })
                .all(|valid| valid)
        });
        if !valid {
            return false;
        }
        for block in &mut *output {
            block.values.fill(0.0);
        }
        for scratch in &self.assembly_scratch {
            for (target, source) in output.iter_mut().zip(&scratch.camera_blocks) {
                for (target, source) in target.values.iter_mut().zip(source.values) {
                    *target += source;
                }
            }
        }
        output
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn assemble_retained(
        &mut self,
        model: ObservationModel<'_>,
        poses: &[RelativePose],
        points: &[[f64; 3]],
        gauge: GaugeState,
        loss: Loss,
        noise: NoiseModel<'_>,
        damping: f64,
        parameter_cameras: &[usize],
        camera_offsets: &[usize],
        camera_dimensions: &[usize],
        point_cross_offsets: &[usize],
        camera_blocks: &mut [PackedCameraBlock],
        point_hessians: &mut [PackedPointBlock],
        gradient: &mut [f64],
        full_gradient: &mut [f64],
        diagonal: &mut [f64],
        point_cache_valid: &mut [bool],
        point_inverse_gradients: &mut [[f64; 3]],
        point_damping_diagonals: &mut [[f64; 3]],
        point_inverse_cross_values: PointCrossCacheMut<'_>,
        pose_variables: CameraPoseVariables,
        pose_masks: &[crate::PoseVariables],
    ) -> schur::Assembly {
        match pose_variables {
            CameraPoseVariables::RotationAndTranslation => assemble_retained_with_workers::<false>(
                &self.pool,
                &self.point_ranges,
                &mut self.assembly_scratch,
                self.dimension,
                model,
                poses,
                points,
                gauge,
                loss,
                noise,
                damping,
                parameter_cameras,
                camera_offsets,
                camera_dimensions,
                point_cross_offsets,
                camera_blocks,
                point_hessians,
                gradient,
                full_gradient,
                diagonal,
                point_cache_valid,
                point_inverse_gradients,
                point_damping_diagonals,
                point_inverse_cross_values,
                pose_masks,
            ),
            CameraPoseVariables::TranslationOnly => assemble_retained_with_workers::<true>(
                &self.pool,
                &self.point_ranges,
                &mut self.assembly_scratch,
                self.dimension,
                model,
                poses,
                points,
                gauge,
                loss,
                noise,
                damping,
                parameter_cameras,
                camera_offsets,
                camera_dimensions,
                point_cross_offsets,
                camera_blocks,
                point_hessians,
                gradient,
                full_gradient,
                diagonal,
                point_cache_valid,
                point_inverse_gradients,
                point_damping_diagonals,
                point_inverse_cross_values,
                pose_masks,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply(
        &mut self,
        _model: ObservationModel<'_>,
        sparse: &SparseWorkspaceCore,
        camera_blocks: Option<&[PackedCameraBlock]>,
        point_hessians: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
        coupling_scale: f64,
    ) {
        let dimension = self.dimension;
        #[cfg(feature = "parallel")]
        self.pool.install(|| {
            self.point_ranges
                .par_windows(2)
                .zip(self.partial_outputs.par_chunks_mut(dimension))
                .for_each(|(range, partial)| {
                    let point_start = range[0];
                    let point_end = range[1];
                    partial.fill(0.0);
                    accumulate_point_coupling_range::<true>(
                        sparse,
                        point_hessians,
                        input,
                        partial,
                        point_start,
                        point_end,
                        1.0,
                    );
                });
        });
        #[cfg(not(feature = "parallel"))]
        std::thread::scope(|scope| {
            for (range, partial) in self
                .point_ranges
                .windows(2)
                .zip(self.partial_outputs.chunks_mut(dimension))
            {
                let point_start = range[0];
                let point_end = range[1];
                scope.spawn(move || {
                    partial.fill(0.0);
                    accumulate_point_coupling_range::<true>(
                        sparse,
                        point_hessians,
                        input,
                        partial,
                        point_start,
                        point_end,
                        1.0,
                    );
                });
            }
        });

        output.fill(0.0);
        reduce_point_major_outputs(
            sparse,
            camera_blocks,
            &self.partial_outputs,
            dimension,
            input,
            0,
            sparse.signature.camera_count(),
            0,
            output,
            coupling_scale,
        );
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn assemble_retained_with_workers<const TRANSLATION_ONLY: bool>(
    pool: &rayon::ThreadPool,
    point_ranges: &[usize],
    assembly_scratch: &mut [RetainedAssemblyScratch],
    dimension: usize,
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    parameter_cameras: &[usize],
    camera_offsets: &[usize],
    camera_dimensions: &[usize],
    point_cross_offsets: &[usize],
    camera_blocks: &mut [PackedCameraBlock],
    point_hessians: &mut [PackedPointBlock],
    gradient: &mut [f64],
    full_gradient: &mut [f64],
    diagonal: &mut [f64],
    point_cache_valid: &mut [bool],
    point_inverse_gradients: &mut [[f64; 3]],
    point_damping_diagonals: &mut [[f64; 3]],
    point_inverse_cross_values: PointCrossCacheMut<'_>,
    pose_masks: &[crate::PoseVariables],
) -> schur::Assembly {
    let context = RetainedAssemblyContext {
        model,
        poses,
        gauge,
        loss,
        noise,
        damping,
        dimension,
        parameter_cameras,
        camera_offsets,
        camera_dimensions,
        pose_masks,
        point_cross_offsets,
    };
    let mut assembly = pool.install(|| {
        assemble_retained_partitions::<TRANSLATION_ONLY>(
            context,
            point_ranges,
            assembly_scratch,
            points,
            0,
            point_hessians,
            point_cache_valid,
            point_inverse_gradients,
            point_damping_diagonals,
            point_inverse_cross_values,
            0,
        )
    });

    for block in &mut *camera_blocks {
        block.values.fill(0.0);
    }
    gradient.fill(0.0);
    full_gradient.fill(0.0);
    diagonal.fill(0.0);
    for scratch in &*assembly_scratch {
        for (target, source) in camera_blocks.iter_mut().zip(&scratch.camera_blocks) {
            for (target, source) in target.values.iter_mut().zip(source.values) {
                *target += source;
            }
        }
        for (target, source) in gradient.iter_mut().zip(&scratch.gradient) {
            *target += source;
        }
        for (target, source) in full_gradient.iter_mut().zip(&scratch.full_gradient) {
            *target += source;
        }
        for (target, source) in diagonal.iter_mut().zip(&scratch.diagonal) {
            *target += source;
        }
    }
    let mut hessian = RetainedBlockHessian {
        camera_blocks,
        parameter_cameras,
        camera_offsets,
        valid: true,
    };
    let layout = if TRANSLATION_ONLY {
        schur::VariableLayout::calibrated_translation_only(camera_offsets, camera_dimensions)
    } else {
        schur::VariableLayout::calibrated(camera_offsets, camera_dimensions)
    };
    let layout = layout.with_pose_masks(pose_masks);
    assembly = schur::finalize_assembly(
        &mut hessian,
        gradient,
        full_gradient,
        diagonal,
        damping,
        dimension,
        layout,
        assembly,
    );
    assembly
}
