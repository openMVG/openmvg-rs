// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-point linearization, cross rows, and point steps.

use super::assembly::{HessianAccumulator, IgnoredHessian, PointLinearization, ReducedBuffers};
use super::layout::{ActiveParameterTracker, GaugeState, VariableLayout};
use super::priors::accumulate_point_prior;
use super::{MAX_CAMERA_BLOCK_DIMENSION, MAX_POSE_DIMENSION, POINT_DIMENSION, norm, scaled};
use crate::kernel::RelativePose;
use crate::kernel::multi_view::PointPrior;
use crate::kernel::multi_view::model::{ObservationModel, rotate, transform};
use crate::kernel::point_block::PointBlockFactor;
use crate::kernel::point_geometry::{
    Admission, Iterate, PointCandidate, TrackGeometry, candidate_rule,
};
use crate::kernel::projection::{Projection, ReprojectionLinearization};
use crate::numerics::refine::{Loss, NoiseModel};

#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "unstable-internals"))]
pub(in crate::kernel::multi_view) fn linearize_point(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
) -> Option<PointLinearization> {
    linearize_point_with_hessian::<true, false, IgnoredHessian>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        layout,
        cross,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn recompute_point_cross_rows(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
) -> Option<usize> {
    recompute_point_cross_rows_impl::<false>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        layout,
        cross,
        active_parameters,
        parameter_marks,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn recompute_point_cross_rows_translation_only(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
) -> Option<usize> {
    recompute_point_cross_rows_impl::<true>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        layout,
        cross,
        active_parameters,
        parameter_marks,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "faer")]
fn recompute_point_cross_rows_impl<const TRANSLATION_ONLY: bool>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
) -> Option<usize> {
    linearize_point_with_hessian_and_active::<false, TRANSLATION_ONLY, IgnoredHessian>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        layout,
        cross,
        None,
        Some(ActiveParameterTracker::new(
            active_parameters,
            parameter_marks,
        )),
    )
    .map(|(_, active_count)| active_count)
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "unstable-internals"))]
pub(super) fn linearize_point_with_hessian<
    const POINT_TERMS: bool,
    const TRANSLATION_ONLY: bool,
    H: HessianAccumulator,
>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    reduced: Option<ReducedBuffers<'_, H>>,
) -> Option<PointLinearization> {
    linearize_point_with_hessian_and_active::<POINT_TERMS, TRANSLATION_ONLY, H>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        layout,
        cross,
        reduced,
        None,
    )
    .map(|(local, _)| local)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn linearize_point_with_hessian_and_active<
    const POINT_TERMS: bool,
    const TRANSLATION_ONLY: bool,
    H: HessianAccumulator,
>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    mut reduced: Option<ReducedBuffers<'_, H>>,
    mut active: Option<ActiveParameterTracker<'_>>,
) -> Option<(PointLinearization, usize)> {
    let mut local = PointLinearization {
        hessian: [[0.0; POINT_DIMENSION]; POINT_DIMENSION],
        gradient: [0.0; POINT_DIMENSION],
        cost: 0.0,
        squared_residual: 0.0,
    };
    let track = model.offsets()[point_index]..model.offsets()[point_index + 1];
    for observation in track {
        let camera = model.camera(observation);
        let rotated = rotate(&poses[camera], point);
        let camera_point = [
            rotated[0] + poses[camera].translation[0],
            rotated[1] + poses[camera].translation[1],
            rotated[2] + poses[camera].translation[2],
        ];
        let (projection, intrinsic_override) = if model.intrinsic_dimension() != 0 {
            let (projection, intrinsics) =
                model.radial_analytic_linearization(observation, camera_point)?;
            (projection, Some(intrinsics))
        } else {
            (model.project(observation, &camera_point)?, None)
        };
        accumulate_observation::<POINT_TERMS, TRANSLATION_ONLY, H>(
            model,
            poses,
            observation,
            camera,
            rotated,
            camera_point,
            projection,
            intrinsic_override,
            gauge,
            loss,
            noise,
            layout,
            cross,
            &mut local,
            reduced.as_mut(),
            active.as_mut(),
        )?;
    }
    let active_count = active.as_ref().map_or(0, |active| active.count);
    Some((local, active_count))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_observation<
    const POINT_TERMS: bool,
    const TRANSLATION_ONLY: bool,
    H: HessianAccumulator,
>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    observation: usize,
    camera: usize,
    rotated: [f64; 3],
    camera_point: [f64; 3],
    projection: Projection,
    intrinsic_override: Option<[[f64; 3]; 2]>,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    local: &mut PointLinearization,
    reduced: Option<&mut ReducedBuffers<'_, H>>,
    active: Option<&mut ActiveParameterTracker<'_>>,
) -> Option<()> {
    let inverse_sigma = noise.inverse_standard_deviation(observation);
    let residual_norm = norm(&projection.residual);
    let sqrt_weight = inverse_sigma * loss.sqrt_weight(residual_norm * inverse_sigma);
    let point_jacobian = scale_2x3(
        multiply_2x3_3x3(&projection.jacobian, &poses[camera].rotation),
        gauge.scene_scale,
    );
    let mut factor = ReprojectionLinearization::new(
        projection,
        [[0.0; MAX_CAMERA_BLOCK_DIMENSION]; 2],
        point_jacobian,
    );
    if POINT_TERMS {
        local.cost += loss.cost(residual_norm * inverse_sigma);
        local.squared_residual += residual_norm * residual_norm;
        accumulate_point(local, &factor, sqrt_weight);
    }

    let pose_dimension = layout.pose_dimensions[camera];
    let intrinsic = model
        .intrinsic_group(camera)
        .and_then(|group| layout.intrinsic(group));
    if pose_dimension == 0 && intrinsic.is_none() {
        return Some(());
    }
    let camera_jacobian = &mut factor.camera_jacobian;
    let mut parameter_indices = [0usize; MAX_CAMERA_BLOCK_DIMENSION];
    if pose_dimension > 0 {
        let mask = layout.pose_mask(camera);
        if TRANSLATION_ONLY || !mask.rotation() {
            let translation_motion = translation_motion(camera, gauge);
            let translation_jacobian =
                multiply_translation(&projection.jacobian, &translation_motion);
            for row in 0..2 {
                camera_jacobian[row][..pose_dimension]
                    .copy_from_slice(&translation_jacobian[row][..pose_dimension]);
            }
        } else {
            // The ordinary product chart changes R while holding projection t.
            // Rotation-only instead holds C=-R^T t, so the entire camera point
            // rotates and its derivative is -[RX+t]_x.
            let rotation_point = if mask.translation() {
                &rotated
            } else {
                &camera_point
            };
            let pose_motion = pose_motion(rotation_point, camera, gauge);
            let pose_jacobian = multiply_pose(&projection.jacobian, &pose_motion);
            for row in 0..2 {
                camera_jacobian[row][..pose_dimension]
                    .copy_from_slice(&pose_jacobian[row][..pose_dimension]);
            }
        }
        for (local_index, parameter) in parameter_indices[..pose_dimension].iter_mut().enumerate() {
            *parameter = layout.offsets[camera] + local_index;
        }
    }
    let free_intrinsic_dimension = intrinsic.map_or(0, |(_, dimension)| dimension);
    if let Some((intrinsic_offset, free_intrinsic_dimension)) = intrinsic {
        let intrinsic_jacobian = match intrinsic_override {
            Some(jacobian) => jacobian,
            None => model.intrinsic_jacobian(observation, &camera_point)?,
        };
        for row in 0..2 {
            camera_jacobian[row][pose_dimension..pose_dimension + free_intrinsic_dimension]
                .copy_from_slice(&intrinsic_jacobian[row][..free_intrinsic_dimension]);
        }
        for (local_index, parameter) in parameter_indices
            [pose_dimension..pose_dimension + free_intrinsic_dimension]
            .iter_mut()
            .enumerate()
        {
            *parameter = intrinsic_offset + local_index;
        }
    }
    let camera_dimension = pose_dimension + free_intrinsic_dimension;
    if let Some(active) = active {
        let pose_start = layout.offsets[camera];
        let intrinsic_run =
            intrinsic.map_or(0..0, |(offset, dimension)| offset..offset + dimension);
        active.register(
            [pose_start..pose_start + pose_dimension, intrinsic_run],
            cross,
        );
    }
    accumulate_camera(
        reduced,
        cross,
        &parameter_indices[..camera_dimension],
        pose_dimension,
        &factor,
        sqrt_weight,
    );
    Some(())
}

fn accumulate_point(
    local: &mut PointLinearization,
    factor: &ReprojectionLinearization<MAX_CAMERA_BLOCK_DIMENSION, POINT_DIMENSION>,
    sqrt_weight: f64,
) {
    for (residual, jacobian_row) in factor.point_jacobian.iter().enumerate() {
        let weighted_residual = sqrt_weight * factor.residual[residual];
        let weighted_point = jacobian_row.map(|value| sqrt_weight * value);
        for row in 0..POINT_DIMENSION {
            local.gradient[row] += weighted_point[row] * weighted_residual;
            for column in 0..POINT_DIMENSION {
                local.hessian[row][column] += weighted_point[row] * weighted_point[column];
            }
        }
    }
}

/// Accumulate one observation's camera terms.
///
/// `parameters` holds the pose run followed by the intrinsic run, which
/// starts at local index `intrinsic_start`. Each run is consecutive and is
/// one variable block, so every buffer is updated through contiguous slices
/// and the Hessian through one dense block per run pair.
fn accumulate_camera<H: HessianAccumulator>(
    reduced: Option<&mut ReducedBuffers<'_, H>>,
    cross: &mut [f64],
    parameters: &[usize],
    intrinsic_start: usize,
    factor: &ReprojectionLinearization<MAX_CAMERA_BLOCK_DIMENSION, POINT_DIMENSION>,
    sqrt_weight: f64,
) {
    // Both residual rows are applied in one pass over distinct parameters.
    // Every entry still adds the first row's term before the second's, so
    // the sums match two sequential rank-one passes bit for bit.
    let [first, second] = factor
        .camera_jacobian
        .each_ref()
        .map(|row| scaled(row, sqrt_weight));
    let [point_first, point_second] = factor
        .point_jacobian
        .each_ref()
        .map(|row| scaled(row, sqrt_weight));
    // Slicing every buffer by the same range lets the compiler prove the
    // per-entry indices in bounds.
    let runs = [0..intrinsic_start, intrinsic_start..parameters.len()];
    let (cross_rows, _) = cross.as_chunks_mut::<POINT_DIMENSION>();
    for run in runs.iter().filter(|run| !run.is_empty()) {
        let start = parameters[run.start];
        for ((target, &first), &second) in cross_rows[start..start + run.len()]
            .iter_mut()
            .zip(&first[run.clone()])
            .zip(&second[run.clone()])
        {
            add_scaled_pair(target, &point_first, first, &point_second, second);
        }
    }
    let Some(buffers) = reduced else {
        return;
    };
    let residual = factor.residual.map(|value| sqrt_weight * value);
    for run in runs.iter().filter(|run| !run.is_empty()) {
        let start = parameters[run.start];
        let terms = [&first[run.clone()], &second[run.clone()]];
        // A fixed run length unrolls the updates into straight-line code.
        match run.len() {
            1 => buffers.add_run::<1>(start, terms, residual),
            2 => buffers.add_run::<2>(start, terms, residual),
            3 => buffers.add_run::<3>(start, terms, residual),
            4 => buffers.add_run::<4>(start, terms, residual),
            5 => buffers.add_run::<5>(start, terms, residual),
            6 => buffers.add_run::<6>(start, terms, residual),
            len => buffers.add_run_terms(start..start + len, terms, residual),
        }
    }
    for (index, row_run) in runs.iter().enumerate() {
        if row_run.is_empty() {
            continue;
        }
        let rows = &parameters[row_run.clone()];
        let row_terms = [&first[row_run.clone()], &second[row_run.clone()]];
        for column_run in runs[..index].iter().filter(|run| !run.is_empty()) {
            let columns = &parameters[column_run.clone()];
            let column_terms = [&first[column_run.clone()], &second[column_run.clone()]];
            buffers
                .hessian
                .add_gram(rows, columns, false, row_terms, column_terms);
        }
        buffers
            .hessian
            .add_gram(rows, rows, true, row_terms, row_terms);
    }
}

/// Add `left[k] * left_scale`, then `right[k] * right_scale`, to `target[k]`.
#[inline(always)]
pub(in crate::kernel::multi_view) fn add_scaled_pair(
    target: &mut [f64],
    left: &[f64],
    left_scale: f64,
    right: &[f64],
    right_scale: f64,
) {
    for ((target, left), right) in target.iter_mut().zip(left).zip(right) {
        *target += left * left_scale;
        *target += right * right_scale;
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::kernel::multi_view) fn point_step_with_prior_and_damping_norm(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    camera_step: &[f64],
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
    prior: Option<&PointPrior>,
) -> Option<([f64; 3], f64)> {
    point_step_with_prior_and_damping_norm_impl::<false>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        damping,
        camera_step,
        layout,
        cross,
        active_parameters,
        parameter_marks,
        prior,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "faer")]
pub(in crate::kernel::multi_view) fn point_step_with_prior_and_damping_norm_translation_only(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    camera_step: &[f64],
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
    prior: Option<&PointPrior>,
) -> Option<([f64; 3], f64)> {
    point_step_with_prior_and_damping_norm_impl::<true>(
        model,
        poses,
        point,
        point_index,
        gauge,
        loss,
        noise,
        damping,
        camera_step,
        layout,
        cross,
        active_parameters,
        parameter_marks,
        prior,
    )
}

#[allow(clippy::too_many_arguments)]
fn point_step_with_prior_and_damping_norm_impl<const TRANSLATION_ONLY: bool>(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    camera_step: &[f64],
    layout: VariableLayout<'_>,
    cross: &mut [f64],
    active_parameters: &mut [usize],
    parameter_marks: &mut [bool],
    prior: Option<&PointPrior>,
) -> Option<([f64; 3], f64)> {
    let (mut local, active_count) =
        linearize_point_with_hessian_and_active::<true, TRANSLATION_ONLY, IgnoredHessian>(
            model,
            poses,
            point,
            point_index,
            gauge,
            loss,
            noise,
            layout,
            cross,
            None,
            Some(ActiveParameterTracker::new(
                active_parameters,
                parameter_marks,
            )),
        )?;
    if let Some(prior) = prior {
        accumulate_point_prior(&mut local, point, prior, gauge.scene_scale, 1.0);
    }
    let factor = PointBlockFactor::new(&local.hessian, damping)?;
    let mut right = local.gradient;
    for &parameter in &active_parameters[..active_count] {
        for coordinate in 0..POINT_DIMENSION {
            right[coordinate] +=
                cross[parameter * POINT_DIMENSION + coordinate] * camera_step[parameter];
        }
    }
    let step = factor.solve(&right).map(|value| -value);
    let diagonal_scale = local
        .hessian
        .iter()
        .enumerate()
        .map(|(coordinate, row)| row[coordinate])
        .fold(0.0, f64::max);
    let floor = 64.0 * f64::EPSILON * diagonal_scale;
    let damping_norm = local
        .hessian
        .iter()
        .enumerate()
        .map(|(coordinate, row)| row[coordinate].max(floor) * step[coordinate] * step[coordinate])
        .sum();
    Some((step, damping_norm))
}

/// Robust cost of `point`, or `None` when it projects behind a camera or is degenerate against
/// its cameras and `scene_length` (see [`TrackGeometry`]).
#[cfg(feature = "unstable-internals")]
pub(in crate::kernel::multi_view) fn point_cost(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    loss: Loss,
    noise: NoiseModel<'_>,
    scene_length: f64,
) -> Option<f64> {
    admitted_point_cost(
        model,
        poses,
        point,
        point_index,
        loss,
        noise,
        scene_length,
        Admission::Geometry,
    )
}

/// A held control has no estimated depth, so parallax is not an acceptance gate.
pub(in crate::kernel::multi_view) fn fixed_point_cost(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Option<f64> {
    admitted_point_cost(
        model,
        poses,
        point,
        point_index,
        loss,
        noise,
        0.0,
        Admission::Projection,
    )
}

/// One free point of a Levenberg-Marquardt candidate.
#[derive(Clone, Copy, Debug)]
pub(in crate::kernel::multi_view) struct CandidatePoint {
    /// The stepped point, or the previous point when the step was not taken.
    pub(in crate::kernel::multi_view) position: [f64; 3],
    pub(in crate::kernel::multi_view) cost: f64,
    pub(in crate::kernel::multi_view) moved: bool,
}

/// Evaluate one free point of a candidate step.
///
/// A point whose step would leave it behind a camera or degenerate keeps its previous position,
/// so one point running to infinity no longer vetoes the step of every camera and point. The step
/// is rejected only when the camera step puts the previous position behind a camera or collapses a
/// camera onto it against the scene; a previous position that already was so is held as is.
#[allow(clippy::too_many_arguments)]
pub(in crate::kernel::multi_view) fn candidate_point(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    candidate_poses: &[RelativePose],
    previous: &[f64; 3],
    stepped: [f64; 3],
    point_index: usize,
    loss: Loss,
    noise: NoiseModel<'_>,
    scene_length: f64,
) -> Option<CandidatePoint> {
    let decision = candidate_rule(|cameras, point, admission| {
        let poses = match cameras {
            Iterate::Current => poses,
            Iterate::Candidate => candidate_poses,
        };
        let point = match point {
            Iterate::Current => previous,
            Iterate::Candidate => &stepped,
        };
        admitted_point_cost(
            model,
            poses,
            point,
            point_index,
            loss,
            noise,
            scene_length,
            admission,
        )
    })?;
    Some(match decision {
        PointCandidate::Moved(cost) => CandidatePoint {
            position: stepped,
            cost,
            moved: true,
        },
        PointCandidate::Held(cost) => CandidatePoint {
            position: *previous,
            cost,
            moved: false,
        },
    })
}

/// Robust cost of a point's track, or `None` when it fails `admission` against `scene_length`.
#[allow(clippy::too_many_arguments)]
fn admitted_point_cost(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    point: &[f64; 3],
    point_index: usize,
    loss: Loss,
    noise: NoiseModel<'_>,
    scene_length: f64,
    admission: Admission,
) -> Option<f64> {
    let mut total = 0.0;
    let mut geometry = TrackGeometry::with_scene_length(scene_length);
    let track = model.offsets()[point_index]..model.offsets()[point_index + 1];
    for observation in track {
        let camera = model.camera(observation);
        let camera_point = transform(&poses[camera], point);
        let projection = model.project(observation, &camera_point)?;
        geometry.push(rotate_transposed(&poses[camera].rotation, &camera_point));
        total +=
            loss.cost(norm(&projection.residual) * noise.inverse_standard_deviation(observation));
    }
    admission.admits(&geometry).then_some(total)
}

pub(super) fn pose_motion(
    rotated: &[f64; 3],
    camera: usize,
    gauge: GaugeState,
) -> [[f64; MAX_POSE_DIMENSION]; 3] {
    let mut motion = [
        [0.0, rotated[2], -rotated[1], 0.0, 0.0, 0.0],
        [-rotated[2], 0.0, rotated[0], 0.0, 0.0, 0.0],
        [rotated[1], -rotated[0], 0.0, 0.0, 0.0, 0.0],
    ];
    if gauge.fixed_camera == Some(camera) && gauge.fixed_component.is_none() {
        for (coordinate, row) in motion.iter_mut().enumerate() {
            row[3] = gauge.baseline * gauge.basis[0][coordinate];
            row[4] = gauge.baseline * gauge.basis[1][coordinate];
        }
    } else if let Some(fixed) = gauge
        .fixed_component
        .filter(|_| gauge.fixed_camera == Some(camera))
    {
        let mut local = 3;
        for (coordinate, row) in motion.iter_mut().enumerate() {
            if coordinate != fixed {
                row[local] = gauge.scene_scale;
                local += 1;
            }
        }
    } else {
        for (coordinate, row) in motion.iter_mut().enumerate() {
            row[3 + coordinate] = gauge.scene_scale;
        }
    }
    motion
}

fn translation_motion(camera: usize, gauge: GaugeState) -> [[f64; 3]; 3] {
    let mut motion = [[0.0; 3]; 3];
    if gauge.fixed_camera == Some(camera) && gauge.fixed_component.is_none() {
        for (coordinate, row) in motion.iter_mut().enumerate() {
            row[0] = gauge.baseline * gauge.basis[0][coordinate];
            row[1] = gauge.baseline * gauge.basis[1][coordinate];
        }
    } else if let Some(fixed) = gauge
        .fixed_component
        .filter(|_| gauge.fixed_camera == Some(camera))
    {
        let mut local = 0;
        for (coordinate, row) in motion.iter_mut().enumerate() {
            if coordinate != fixed {
                row[local] = gauge.scene_scale;
                local += 1;
            }
        }
    } else {
        for (coordinate, row) in motion.iter_mut().enumerate() {
            row[coordinate] = gauge.scene_scale;
        }
    }
    motion
}

pub(super) fn multiply_pose(
    left: &[[f64; 3]; 2],
    right: &[[f64; MAX_POSE_DIMENSION]; 3],
) -> [[f64; MAX_POSE_DIMENSION]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn multiply_translation(left: &[[f64; 3]; 2], right: &[[f64; 3]; 3]) -> [[f64; 3]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

pub(super) fn multiply_2x3_3x3(left: &[[f64; 3]; 2], right: &[[f64; 3]; 3]) -> [[f64; 3]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

pub(super) fn scale_2x3(mut matrix: [[f64; 3]; 2], scale: f64) -> [[f64; 3]; 2] {
    for row in &mut matrix {
        for value in row {
            *value *= scale;
        }
    }
    matrix
}

#[inline(always)]
pub(in crate::kernel::multi_view) fn cross_row(cross: &[f64], row: usize) -> [f64; 3] {
    cross.as_chunks::<POINT_DIMENSION>().0[row]
}

/// `rotation^T * vector`: a camera-frame vector expressed in world axes.
#[inline]
fn rotate_transposed(rotation: &[[f64; 3]; 3], vector: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        rotation[0][axis] * vector[0]
            + rotation[1][axis] * vector[1]
            + rotation[2][axis] * vector[2]
    })
}
