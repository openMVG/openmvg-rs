// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained block algebra and fused point-coupling kernels.

use super::*;

pub(super) fn rejected(terms: usize, relative_residual: Option<f64>) -> SchurAttempt {
    SchurAttempt::Rejected(RejectedSchurSolve {
        terms,
        relative_residual,
    })
}

pub(super) type CameraBlockFactor =
    PackedCameraBlockFactor<ANCHORED_CAMERA_BLOCK_DIMENSION, ANCHORED_CAMERA_BLOCK_STORAGE>;

pub(super) fn solve_camera_blocks(
    factors: &[CameraBlockFactor],
    right: &[f64],
    solution: &mut [f64],
    offsets: &[usize],
    dimensions: &[usize],
) {
    solution.fill(0.0);
    for camera in 0..dimensions.len() {
        let dimension = dimensions[camera];
        if dimension == 0 {
            continue;
        }
        let offset = offsets[camera];
        factors[camera].solve(
            &right[offset..offset + dimension],
            &mut solution[offset..offset + dimension],
            dimension,
        );
    }
}

#[cfg(test)]
pub(super) fn accumulate_camera_blocks(
    blocks: &[PackedCameraBlock],
    sparse: &SparseWorkspaceCore,
    input: &[f64],
    output: &mut [f64],
    scale: f64,
) {
    for (camera, block) in blocks
        .iter()
        .enumerate()
        .take(sparse.signature.camera_count())
    {
        let dimension = sparse.layout.camera_dimensions()[camera];
        let offset = sparse.layout.camera_offsets()[camera];
        block.multiply_add_scaled(
            &input[offset..offset + dimension],
            &mut output[offset..offset + dimension],
            dimension,
            scale,
        );
    }
}

#[cfg(test)]
pub(super) fn apply_point_coupling(
    model: ObservationModel<'_>,
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    input: &[f64],
    output: &mut [f64],
) {
    output.fill(0.0);
    accumulate_point_coupling::<true>(model, sparse, point_hessians, input, output, 1.0);
}

pub(super) fn accumulate_point_coupling<const UNIT_SCALE: bool>(
    _model: ObservationModel<'_>,
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    input: &[f64],
    output: &mut [f64],
    scale: f64,
) {
    accumulate_point_coupling_range::<UNIT_SCALE>(
        sparse,
        point_hessians,
        input,
        output,
        0,
        sparse.signature.point_count(),
        scale,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_point_coupling_range<const UNIT_SCALE: bool>(
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    input: &[f64],
    output: &mut [f64],
    point_start: usize,
    point_end: usize,
    scale: f64,
) {
    debug_assert!(!UNIT_SCALE || scale == 1.0);
    match &sparse.point_inverse_cross_values {
        PointCrossCache::F64(crosses) => accumulate_point_coupling_range_f64::<UNIT_SCALE>(
            sparse,
            point_hessians,
            crosses.as_slice(),
            input,
            output,
            point_start,
            point_end,
            scale,
        ),
        PointCrossCache::F32(crosses) => accumulate_point_coupling_range_for(
            sparse,
            point_hessians,
            crosses.as_slice(),
            input,
            output,
            point_start,
            point_end,
            scale,
        ),
        // Dispatch selects this operator only for retained cross-row storage.
        #[allow(clippy::unreachable)]
        PointCrossCache::Recomputed { .. } => {
            unreachable!("recomputed cross rows use the recomputed operator")
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn accumulate_point_coupling_range_f64<const UNIT_SCALE: bool>(
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    crosses: &[[f64; 3]],
    input: &[f64],
    output: &mut [f64],
    point_start: usize,
    point_end: usize,
    scale: f64,
) {
    if sparse.anchored_six_camera_layout {
        accumulate_anchored_six_point_coupling_range_f64::<UNIT_SCALE>(
            sparse,
            point_hessians,
            crosses,
            input,
            output,
            point_start,
            point_end,
            scale,
        );
        return;
    }
    for (point, point_hessian) in point_hessians
        .iter()
        .enumerate()
        .take(point_end)
        .skip(point_start)
    {
        let range = sparse.point_cross_offsets[point]..sparse.point_cross_offsets[point + 1];
        if range.is_empty() {
            continue;
        }
        let inverse_crosses = &crosses[range];
        let mut cache_index = 0;
        let mut inverse_cross_sum = [0.0; 3];
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            let offset = sparse.layout.camera_offsets()[camera];
            let dimension = sparse.layout.camera_dimensions()[camera];
            if dimension == ANCHORED_CAMERA_BLOCK_DIMENSION {
                accumulate_six_cross_products_f64(
                    &inverse_crosses[cache_index..cache_index + ANCHORED_CAMERA_BLOCK_DIMENSION],
                    &input[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                    &mut inverse_cross_sum,
                );
                cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
                continue;
            }
            for local in 0..dimension {
                let inverse_cross = inverse_crosses[cache_index + local];
                for coordinate in 0..3 {
                    inverse_cross_sum[coordinate] +=
                        inverse_cross[coordinate] * input[offset + local];
                }
            }
            cache_index += dimension;
        }
        let weighted_sum = point_hessian.multiply(&inverse_cross_sum);
        cache_index = 0;
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            let offset = sparse.layout.camera_offsets()[camera];
            let dimension = sparse.layout.camera_dimensions()[camera];
            if dimension == ANCHORED_CAMERA_BLOCK_DIMENSION {
                scatter_six_cross_products_f64::<UNIT_SCALE>(
                    &inverse_crosses[cache_index..cache_index + ANCHORED_CAMERA_BLOCK_DIMENSION],
                    &weighted_sum,
                    &mut output[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                    scale,
                );
                cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
                continue;
            }
            for local in 0..dimension {
                let coupling = dot3(&inverse_crosses[cache_index + local], &weighted_sum);
                if UNIT_SCALE {
                    output[offset + local] += coupling;
                } else {
                    output[offset + local] += scale * coupling;
                }
            }
            cache_index += dimension;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_anchored_six_point_coupling_range_f64<const UNIT_SCALE: bool>(
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    crosses: &[[f64; 3]],
    input: &[f64],
    output: &mut [f64],
    point_start: usize,
    point_end: usize,
    scale: f64,
) {
    for (point, point_hessian) in point_hessians
        .iter()
        .enumerate()
        .take(point_end)
        .skip(point_start)
    {
        let range = sparse.point_cross_offsets[point]..sparse.point_cross_offsets[point + 1];
        if range.is_empty() {
            continue;
        }
        let inverse_crosses = &crosses[range];
        let mut cache_index = 0;
        let mut inverse_cross_sum = [0.0; 3];
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            if camera == 0 {
                continue;
            }
            let offset = (camera - 1) * ANCHORED_CAMERA_BLOCK_DIMENSION;
            accumulate_six_cross_products_f64(
                &inverse_crosses[cache_index..cache_index + ANCHORED_CAMERA_BLOCK_DIMENSION],
                &input[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                &mut inverse_cross_sum,
            );
            cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
        }
        let weighted_sum = point_hessian.multiply(&inverse_cross_sum);
        cache_index = 0;
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            if camera == 0 {
                continue;
            }
            let offset = (camera - 1) * ANCHORED_CAMERA_BLOCK_DIMENSION;
            scatter_six_cross_products_f64::<UNIT_SCALE>(
                &inverse_crosses[cache_index..cache_index + ANCHORED_CAMERA_BLOCK_DIMENSION],
                &weighted_sum,
                &mut output[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                scale,
            );
            cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
        }
        debug_assert_eq!(cache_index, inverse_crosses.len());
    }
}

#[inline(always)]
pub(super) fn accumulate_six_cross_products_f64(
    crosses: &[[f64; 3]],
    input: &[f64],
    output: &mut [f64; 3],
) {
    for local in 0..ANCHORED_CAMERA_BLOCK_DIMENSION {
        let cross = crosses[local];
        let value = input[local];
        output[0] += cross[0] * value;
        output[1] += cross[1] * value;
        output[2] += cross[2] * value;
    }
}

#[inline(always)]
fn scatter_six_cross_products_f64<const UNIT_SCALE: bool>(
    crosses: &[[f64; 3]],
    input: &[f64; 3],
    output: &mut [f64],
    scale: f64,
) {
    for local in 0..ANCHORED_CAMERA_BLOCK_DIMENSION {
        let coupling = dot3(&crosses[local], input);
        if UNIT_SCALE {
            output[local] += coupling;
        } else {
            output[local] += scale * coupling;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn accumulate_point_coupling_range_for<C: PointCrossStorage + ?Sized>(
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    crosses: &C,
    input: &[f64],
    output: &mut [f64],
    point_start: usize,
    point_end: usize,
    scale: f64,
) {
    if sparse.anchored_six_camera_layout {
        accumulate_anchored_six_point_coupling_range(
            sparse,
            point_hessians,
            crosses,
            input,
            output,
            point_start,
            point_end,
            scale,
        );
        return;
    }
    for (point, point_hessian) in point_hessians
        .iter()
        .enumerate()
        .take(point_end)
        .skip(point_start)
    {
        let range = sparse.point_cross_offsets[point]..sparse.point_cross_offsets[point + 1];
        if range.is_empty() {
            continue;
        }
        let inverse_crosses = crosses.rows(range);
        let mut cache_index = 0;
        let mut inverse_cross_sum = [0.0; 3];
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            let offset = sparse.layout.camera_offsets()[camera];
            let dimension = sparse.layout.camera_dimensions()[camera];
            if dimension == ANCHORED_CAMERA_BLOCK_DIMENSION {
                accumulate_six_cross_products(
                    inverse_crosses,
                    cache_index,
                    &input[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                    &mut inverse_cross_sum,
                );
                cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
                continue;
            }
            for local in 0..dimension {
                let inverse_cross = inverse_crosses.row(cache_index + local);
                for coordinate in 0..3 {
                    inverse_cross_sum[coordinate] +=
                        inverse_cross[coordinate] * input[offset + local];
                }
            }
            cache_index += dimension;
        }
        let weighted_sum = point_hessian.multiply(&inverse_cross_sum);
        cache_index = 0;
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            let offset = sparse.layout.camera_offsets()[camera];
            let dimension = sparse.layout.camera_dimensions()[camera];
            if dimension == ANCHORED_CAMERA_BLOCK_DIMENSION {
                scatter_six_cross_products(
                    inverse_crosses,
                    cache_index,
                    &weighted_sum,
                    &mut output[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                    scale,
                );
                cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
                continue;
            }
            for local in 0..dimension {
                let inverse_cross = inverse_crosses.row(cache_index + local);
                output[offset + local] += scale * dot3(&inverse_cross, &weighted_sum);
            }
            cache_index += dimension;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_anchored_six_point_coupling_range<C: PointCrossStorage + ?Sized>(
    sparse: &SparseWorkspaceCore,
    point_hessians: &[PackedPointBlock],
    crosses: &C,
    input: &[f64],
    output: &mut [f64],
    point_start: usize,
    point_end: usize,
    scale: f64,
) {
    for (point, point_hessian) in point_hessians
        .iter()
        .enumerate()
        .take(point_end)
        .skip(point_start)
    {
        let range = sparse.point_cross_offsets[point]..sparse.point_cross_offsets[point + 1];
        if range.is_empty() {
            continue;
        }
        let inverse_crosses = crosses.rows(range);
        let mut cache_index = 0;
        let mut inverse_cross_sum = [0.0; 3];
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            if camera == 0 {
                continue;
            }
            let offset = (camera - 1) * ANCHORED_CAMERA_BLOCK_DIMENSION;
            accumulate_six_cross_products(
                inverse_crosses,
                cache_index,
                &input[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                &mut inverse_cross_sum,
            );
            cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
        }
        let weighted_sum = point_hessian.multiply(&inverse_cross_sum);
        cache_index = 0;
        for observation in sparse.signature.point_pose_offsets()[point]
            ..sparse.signature.point_pose_offsets()[point + 1]
        {
            let camera = sparse.signature.point_poses()[observation] as usize;
            if camera == 0 {
                continue;
            }
            let offset = (camera - 1) * ANCHORED_CAMERA_BLOCK_DIMENSION;
            scatter_six_cross_products(
                inverse_crosses,
                cache_index,
                &weighted_sum,
                &mut output[offset..offset + ANCHORED_CAMERA_BLOCK_DIMENSION],
                scale,
            );
            cache_index += ANCHORED_CAMERA_BLOCK_DIMENSION;
        }
        debug_assert_eq!(cache_index, inverse_crosses.len());
    }
}

#[inline(always)]
pub(super) fn accumulate_six_cross_products<C: PointCrossRows>(
    crosses: C,
    start: usize,
    input: &[f64],
    output: &mut [f64; 3],
) {
    for (local, &value) in input
        .iter()
        .take(ANCHORED_CAMERA_BLOCK_DIMENSION)
        .enumerate()
    {
        let cross = crosses.row(start + local);
        output[0] += cross[0] * value;
        output[1] += cross[1] * value;
        output[2] += cross[2] * value;
    }
}

#[inline(always)]
pub(super) fn scatter_six_cross_products<C: PointCrossRows>(
    crosses: C,
    start: usize,
    input: &[f64; 3],
    output: &mut [f64],
    scale: f64,
) {
    for (local, value) in output
        .iter_mut()
        .take(ANCHORED_CAMERA_BLOCK_DIMENSION)
        .enumerate()
    {
        *value += scale * dot3(&crosses.row(start + local), input);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reduce_point_major_outputs(
    sparse: &SparseWorkspaceCore,
    camera_blocks: Option<&[PackedCameraBlock]>,
    partial_outputs: &[f64],
    dimension: usize,
    input: &[f64],
    camera_start: usize,
    camera_end: usize,
    parameter_start: usize,
    output: &mut [f64],
    coupling_scale: f64,
) {
    for camera in camera_start..camera_end {
        let global_offset = sparse.layout.camera_offsets()[camera];
        let local_offset = global_offset - parameter_start;
        let camera_dimension = sparse.layout.camera_dimensions()[camera];
        if let Some(camera_blocks) = camera_blocks {
            camera_blocks[camera].multiply_add(
                &input[global_offset..global_offset + camera_dimension],
                &mut output[local_offset..local_offset + camera_dimension],
                camera_dimension,
            );
        }
        for local in 0..camera_dimension {
            let parameter = global_offset + local;
            let coupling: f64 = partial_outputs
                .chunks(dimension)
                .map(|partial| partial[parameter])
                .sum();
            output[local_offset + local] += coupling_scale * coupling;
        }
    }
}

pub(super) fn apply_reduced_system(
    model: ObservationModel<'_>,
    sparse: &SparseWorkspaceCore,
    camera_blocks: &[PackedCameraBlock],
    point_hessians: &[PackedPointBlock],
    input: &[f64],
    output: &mut [f64],
) {
    output.fill(0.0);
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
    accumulate_point_coupling::<false>(model, sparse, point_hessians, input, output, -1.0);
}
