// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Analytic cross-row recomputation operators and scratch.

use super::operator::*;
use super::point_major::PointMajorParallelWorkspace;
#[cfg(feature = "parallel")]
use super::point_major::RetainedAssemblyScratch;
use super::*;

fn calibrated_layout<const TRANSLATION_ONLY: bool>(
    sparse: &SparseWorkspaceCore,
) -> schur::VariableLayout<'_> {
    let layout = if TRANSLATION_ONLY {
        schur::VariableLayout::calibrated_translation_only(
            sparse.layout.camera_offsets(),
            sparse.layout.camera_dimensions(),
        )
    } else {
        schur::VariableLayout::calibrated(
            sparse.layout.camera_offsets(),
            sparse.layout.camera_dimensions(),
        )
    };
    layout.with_pose_masks(sparse.signature.pose_masks())
}

#[allow(clippy::too_many_arguments)]
fn recompute_point_cross_rows<const TRANSLATION_ONLY: bool>(
    context: RecomputedSchurContext<'_>,
    point: usize,
    layout: schur::VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
) -> Option<usize> {
    if TRANSLATION_ONLY {
        schur::recompute_point_cross_rows_translation_only(
            context.model,
            context.poses,
            &context.points[point],
            point,
            context.gauge,
            context.loss,
            context.noise,
            layout,
            cross,
            active_parameters,
            parameter_marks,
        )
    } else {
        schur::recompute_point_cross_rows(
            context.model,
            context.poses,
            &context.points[point],
            point,
            context.gauge,
            context.loss,
            context.noise,
            layout,
            cross,
            active_parameters,
            parameter_marks,
        )
    }
}

pub(super) struct RecomputedMatvecWorkspace {
    pub(super) cross: Vec<f64>,
    pub(super) active_parameters: Vec<usize>,
    pub(super) parameter_marks: Vec<bool>,
}

impl RecomputedMatvecWorkspace {
    pub(super) fn new(dimension: usize) -> Result<Self, SparseBundleAdjustmentError> {
        let cross_len = dimension
            .checked_mul(3)
            .ok_or(SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
        Ok(Self {
            cross: vec![0.0; cross_len],
            active_parameters: vec![0; dimension],
            parameter_marks: vec![false; dimension],
        })
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.cross.capacity() * std::mem::size_of::<f64>()
            + self.active_parameters.capacity() * std::mem::size_of::<usize>()
            + self.parameter_marks.capacity() * std::mem::size_of::<bool>()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        camera_blocks: Option<&[PackedCameraBlock]>,
        point_inverses: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
        coupling_scale: f64,
    ) {
        output.fill(0.0);
        if let Some(camera_blocks) = camera_blocks {
            for (camera, block) in camera_blocks
                .iter()
                .enumerate()
                .take(sparse.signature.camera_count())
            {
                let dimension = sparse.layout.camera_dimensions()[camera];
                let offset = sparse.layout.camera_offsets()[camera];
                block.multiply_add(
                    &input[offset..offset + dimension],
                    &mut output[offset..offset + dimension],
                    dimension,
                );
            }
        }
        if !self.apply_point_range(
            context,
            sparse,
            point_inverses,
            input,
            output,
            0,
            sparse.signature.point_count(),
            coupling_scale,
        ) {
            output.fill(f64::NAN);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_point_range(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
        point_start: usize,
        point_end: usize,
        coupling_scale: f64,
    ) -> bool {
        match sparse.signature.pose_variables() {
            CameraPoseVariables::RotationAndTranslation => self.apply_point_range_impl::<false>(
                context,
                sparse,
                point_inverses,
                input,
                output,
                point_start,
                point_end,
                coupling_scale,
            ),
            CameraPoseVariables::TranslationOnly => self.apply_point_range_impl::<true>(
                context,
                sparse,
                point_inverses,
                input,
                output,
                point_start,
                point_end,
                coupling_scale,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_point_range_impl<const TRANSLATION_ONLY: bool>(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
        point_start: usize,
        point_end: usize,
        coupling_scale: f64,
    ) -> bool {
        let layout = calibrated_layout::<TRANSLATION_ONLY>(sparse);
        for (point, point_inverse) in point_inverses
            .iter()
            .enumerate()
            .take(point_end)
            .skip(point_start)
        {
            let Some(active_count) = recompute_point_cross_rows::<TRANSLATION_ONLY>(
                context,
                point,
                layout,
                &mut self.cross,
                &mut self.active_parameters,
                &mut self.parameter_marks,
            ) else {
                return false;
            };
            let mut cross_sum = [0.0; 3];
            for &parameter in &self.active_parameters[..active_count] {
                let row = schur::cross_row(&self.cross, parameter);
                for coordinate in 0..3 {
                    cross_sum[coordinate] += row[coordinate] * input[parameter];
                }
            }
            let inverse_sum = point_inverse.multiply(&cross_sum);
            for &parameter in &self.active_parameters[..active_count] {
                let row = schur::cross_row(&self.cross, parameter);
                output[parameter] += coupling_scale * dot3(&row, &inverse_sum);
            }
        }
        true
    }

    pub(super) fn build_diagonal_point_coupling(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
        output: &mut [PackedCameraBlock],
    ) -> bool {
        for block in &mut *output {
            block.values.fill(0.0);
        }
        self.accumulate_diagonal_range(
            context,
            sparse,
            point_inverses,
            output,
            0,
            sparse.signature.point_count(),
        ) && output
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
    }

    pub(super) fn accumulate_diagonal_range(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
        output: &mut [PackedCameraBlock],
        point_start: usize,
        point_end: usize,
    ) -> bool {
        match sparse.signature.pose_variables() {
            CameraPoseVariables::RotationAndTranslation => self
                .accumulate_diagonal_range_impl::<false>(
                    context,
                    sparse,
                    point_inverses,
                    output,
                    point_start,
                    point_end,
                ),
            CameraPoseVariables::TranslationOnly => self.accumulate_diagonal_range_impl::<true>(
                context,
                sparse,
                point_inverses,
                output,
                point_start,
                point_end,
            ),
        }
    }

    fn accumulate_diagonal_range_impl<const TRANSLATION_ONLY: bool>(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
        output: &mut [PackedCameraBlock],
        point_start: usize,
        point_end: usize,
    ) -> bool {
        let layout = calibrated_layout::<TRANSLATION_ONLY>(sparse);
        for (point, point_inverse) in point_inverses
            .iter()
            .enumerate()
            .take(point_end)
            .skip(point_start)
        {
            if recompute_point_cross_rows::<TRANSLATION_ONLY>(
                context,
                point,
                layout,
                &mut self.cross,
                &mut self.active_parameters,
                &mut self.parameter_marks,
            )
            .is_none()
            {
                return false;
            }
            for &camera in &sparse.signature.point_poses()[sparse.signature.point_pose_offsets()
                [point]
                ..sparse.signature.point_pose_offsets()[point + 1]]
            {
                let camera = camera as usize;
                let offset = sparse.layout.camera_offsets()[camera];
                let dimension = sparse.layout.camera_dimensions()[camera];
                for row in 0..dimension {
                    let left = schur::cross_row(&self.cross, offset + row);
                    for column in 0..=row {
                        let right = schur::cross_row(&self.cross, offset + column);
                        output[camera].add(row, column, point_inverse.bilinear(&left, &right));
                    }
                }
            }
        }
        true
    }
}

pub(super) struct RecomputedParallelWorkspace {
    pub(super) shared: PointMajorParallelWorkspace,
    #[cfg(not(feature = "parallel"))]
    partial_diagonal_blocks: Vec<PackedCameraBlock>,
    #[cfg(not(feature = "parallel"))]
    scratch: Vec<RecomputedMatvecWorkspace>,
}

impl RecomputedParallelWorkspace {
    pub(super) fn try_new(
        threads: usize,
        logical_shards: usize,
        camera_count: usize,
        dimension: usize,
        point_offsets: &[usize],
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let shared = PointMajorParallelWorkspace::try_new(
            threads,
            logical_shards,
            camera_count,
            dimension,
            point_offsets,
        )?;
        #[cfg(not(feature = "parallel"))]
        let worker_count = shared.point_ranges.len().saturating_sub(1);
        #[cfg(not(feature = "parallel"))]
        let partial_block_len = worker_count
            .checked_mul(camera_count)
            .ok_or(SparseBundleAdjustmentError::RetainedSchurIndexCapacityOverflow)?;
        Ok(Self {
            shared,
            #[cfg(not(feature = "parallel"))]
            partial_diagonal_blocks: vec![PackedCameraBlock::default(); partial_block_len],
            #[cfg(not(feature = "parallel"))]
            scratch: (0..worker_count)
                .map(|_| RecomputedMatvecWorkspace::new(dimension))
                .collect::<Result<_, _>>()?,
        })
    }

    pub(super) fn retained_bytes(&self) -> usize {
        #[cfg(feature = "parallel")]
        {
            self.shared.retained_bytes()
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.shared.retained_bytes()
                + self.partial_diagonal_blocks.capacity() * std::mem::size_of::<PackedCameraBlock>()
                + self
                    .scratch
                    .iter()
                    .map(RecomputedMatvecWorkspace::retained_bytes)
                    .sum::<usize>()
        }
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_candidates(
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
    ) -> bool {
        match pose_variables {
            CameraPoseVariables::RotationAndTranslation => {
                let layout = schur::VariableLayout::calibrated(camera_offsets, camera_dimensions)
                    .with_pose_masks(pose_masks);
                self.shared.pool.install(|| {
                    evaluate_recomputed_candidate_partitions::<false>(
                        context,
                        damping,
                        camera_step,
                        candidate_poses,
                        layout,
                        &self.shared.point_ranges,
                        &mut self.shared.assembly_scratch,
                        candidate_points,
                        point_metrics,
                        0,
                    )
                })
            }
            CameraPoseVariables::TranslationOnly => {
                let layout = schur::VariableLayout::calibrated_translation_only(
                    camera_offsets,
                    camera_dimensions,
                )
                .with_pose_masks(pose_masks);
                self.shared.pool.install(|| {
                    evaluate_recomputed_candidate_partitions::<true>(
                        context,
                        damping,
                        camera_step,
                        candidate_poses,
                        layout,
                        &self.shared.point_ranges,
                        &mut self.shared.assembly_scratch,
                        candidate_points,
                        point_metrics,
                        0,
                    )
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        camera_blocks: Option<&[PackedCameraBlock]>,
        point_inverses: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
        coupling_scale: f64,
    ) {
        let dimension = self.shared.dimension;
        #[cfg(feature = "parallel")]
        self.shared.pool.install(|| {
            self.shared
                .point_ranges
                .par_windows(2)
                .zip(self.shared.assembly_scratch.par_iter_mut())
                .zip(self.shared.partial_outputs.par_chunks_mut(dimension))
                .for_each(|((range, scratch), partial)| {
                    partial.fill(0.0);
                    if !scratch.recomputed.apply_point_range(
                        context,
                        sparse,
                        point_inverses,
                        input,
                        partial,
                        range[0],
                        range[1],
                        1.0,
                    ) {
                        partial.fill(f64::NAN);
                    }
                });
        });
        #[cfg(not(feature = "parallel"))]
        std::thread::scope(|scope| {
            for ((range, scratch), partial) in self
                .shared
                .point_ranges
                .windows(2)
                .zip(&mut self.scratch)
                .zip(self.shared.partial_outputs.chunks_mut(dimension))
            {
                scope.spawn(move || {
                    partial.fill(0.0);
                    if !scratch.apply_point_range(
                        context,
                        sparse,
                        point_inverses,
                        input,
                        partial,
                        range[0],
                        range[1],
                        1.0,
                    ) {
                        partial.fill(f64::NAN);
                    }
                });
            }
        });

        output.fill(0.0);
        reduce_point_major_outputs(
            sparse,
            camera_blocks,
            &self.shared.partial_outputs,
            dimension,
            input,
            0,
            sparse.signature.camera_count(),
            0,
            output,
            coupling_scale,
        );
    }

    pub(super) fn build_diagonal_point_coupling(
        &mut self,
        context: RecomputedSchurContext<'_>,
        sparse: &SparseWorkspaceCore,
        point_inverses: &[PackedPointBlock],
        output: &mut [PackedCameraBlock],
    ) -> bool {
        #[cfg(not(feature = "parallel"))]
        let camera_count = sparse.signature.camera_count();
        #[cfg(feature = "parallel")]
        self.shared.pool.install(|| {
            self.shared
                .point_ranges
                .par_windows(2)
                .zip(self.shared.assembly_scratch.par_iter_mut())
                .for_each(|(range, scratch)| {
                    for block in &mut scratch.camera_blocks {
                        block.values.fill(0.0);
                    }
                    if !scratch.recomputed.accumulate_diagonal_range(
                        context,
                        sparse,
                        point_inverses,
                        &mut scratch.camera_blocks,
                        range[0],
                        range[1],
                    ) {
                        scratch.camera_blocks[0].values[0] = f64::NAN;
                    }
                });
        });
        #[cfg(not(feature = "parallel"))]
        std::thread::scope(|scope| {
            for ((range, scratch), blocks) in self
                .shared
                .point_ranges
                .windows(2)
                .zip(&mut self.scratch)
                .zip(self.partial_diagonal_blocks.chunks_mut(camera_count))
            {
                scope.spawn(move || {
                    for block in &mut *blocks {
                        block.values.fill(0.0);
                    }
                    if !scratch.accumulate_diagonal_range(
                        context,
                        sparse,
                        point_inverses,
                        blocks,
                        range[0],
                        range[1],
                    ) {
                        blocks[0].values[0] = f64::NAN;
                    }
                });
            }
        });
        #[cfg(feature = "parallel")]
        let valid = self.shared.assembly_scratch.iter().all(|scratch| {
            scratch
                .camera_blocks
                .iter()
                .all(|block| block.values.iter().all(|value| value.is_finite()))
        });
        #[cfg(not(feature = "parallel"))]
        let valid = self
            .partial_diagonal_blocks
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()));
        if !valid {
            return false;
        }
        for block in &mut *output {
            block.values.fill(0.0);
        }
        #[cfg(feature = "parallel")]
        for scratch in &self.shared.assembly_scratch {
            for (target, source) in output.iter_mut().zip(&scratch.camera_blocks) {
                for (target, source) in target.values.iter_mut().zip(source.values) {
                    *target += source;
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        for partial in self.partial_diagonal_blocks.chunks(camera_count) {
            for (target, source) in output.iter_mut().zip(partial) {
                for (target, source) in target.values.iter_mut().zip(source.values) {
                    *target += source;
                }
            }
        }
        output
            .iter()
            .all(|block| block.values.iter().all(|value| value.is_finite()))
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn evaluate_recomputed_candidate_partitions<const TRANSLATION_ONLY: bool>(
    context: RecomputedSchurContext<'_>,
    damping: f64,
    camera_step: &[f64],
    candidate_poses: &[RelativePose],
    layout: schur::VariableLayout<'_>,
    point_ranges: &[usize],
    scratch: &mut [RetainedAssemblyScratch],
    candidate_points: &mut [[f64; 3]],
    point_metrics: &mut [[f64; 3]],
    point_start: usize,
) -> bool {
    if scratch.len() == 1 {
        let worker = &mut scratch[0].recomputed;
        for (local_point, (candidate, metrics)) in
            candidate_points.iter_mut().zip(point_metrics).enumerate()
        {
            let point = point_start + local_point;
            let result = if TRANSLATION_ONLY {
                schur::point_step_with_prior_and_damping_norm_translation_only(
                    context.model,
                    context.poses,
                    &context.points[point],
                    point,
                    context.gauge,
                    context.loss,
                    context.noise,
                    damping,
                    camera_step,
                    layout,
                    &mut worker.cross,
                    &mut worker.active_parameters,
                    &mut worker.parameter_marks,
                    None,
                )
            } else {
                schur::point_step_with_prior_and_damping_norm(
                    context.model,
                    context.poses,
                    &context.points[point],
                    point,
                    context.gauge,
                    context.loss,
                    context.noise,
                    damping,
                    camera_step,
                    layout,
                    &mut worker.cross,
                    &mut worker.active_parameters,
                    &mut worker.parameter_marks,
                    None,
                )
            };
            let Some((step, damping_norm)) = result else {
                *metrics = [f64::NAN; 3];
                return false;
            };
            let previous = &context.points[point];
            let Some(evaluated) = schur::candidate_point(
                context.model,
                context.poses,
                candidate_poses,
                previous,
                std::array::from_fn(|coordinate| {
                    previous[coordinate] + context.gauge.scene_scale * step[coordinate]
                }),
                point,
                context.loss,
                context.noise,
                context.gauge.scene_scale,
            ) else {
                *metrics = [f64::NAN; 3];
                return false;
            };
            *candidate = evaluated.position;
            *metrics = if evaluated.moved {
                [evaluated.cost, dot3(&step, &step), damping_norm]
            } else {
                [evaluated.cost, 0.0, 0.0]
            };
        }
        return true;
    }

    let middle = scratch.len() / 2;
    let point_split = point_ranges[middle];
    let local_split = point_split - point_start;
    let (left_scratch, right_scratch) = scratch.split_at_mut(middle);
    let (left_points, right_points) = candidate_points.split_at_mut(local_split);
    let (left_metrics, right_metrics) = point_metrics.split_at_mut(local_split);
    let (left, right) = rayon::join(
        || {
            evaluate_recomputed_candidate_partitions::<TRANSLATION_ONLY>(
                context,
                damping,
                camera_step,
                candidate_poses,
                layout,
                &point_ranges[..=middle],
                left_scratch,
                left_points,
                left_metrics,
                point_start,
            )
        },
        || {
            evaluate_recomputed_candidate_partitions::<TRANSLATION_ONLY>(
                context,
                damping,
                camera_step,
                candidate_poses,
                layout,
                &point_ranges[middle..],
                right_scratch,
                right_points,
                right_metrics,
                point_split,
            )
        },
    );
    left && right
}
