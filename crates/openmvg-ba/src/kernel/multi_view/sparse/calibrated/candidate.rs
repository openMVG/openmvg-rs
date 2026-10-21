// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candidate point back-substitution and evaluation.

use crate::kernel::RelativePose;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::optimizer;
use crate::kernel::multi_view::schur::{self, GaugeState};
use crate::kernel::multi_view::sparse::api::SparsePointBackSubstitution;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::retained::RecomputedSchurContext;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::workspace::CalibratedBackend;
use crate::kernel::multi_view::sparse::workspace::{
    PointCrossCache, SparseBundleAdjustmentWorkspace,
};
use crate::kernel::multi_view::{BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks};
use crate::numerics::refine::{Loss, NoiseModel};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_candidate(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> Option<(f64, f64, f64)> {
    {
        let core = &mut workspace.core;
        match core.signature.pose_variables() {
            CameraPoseVariables::RotationAndTranslation => optimizer::limit_camera_step(
                &mut core.step,
                core.layout.camera_offsets(),
                core.layout.camera_dimensions(),
                poses.len(),
                0.5,
            ),
            CameraPoseVariables::TranslationOnly => optimizer::limit_translation_step(
                &mut core.step,
                core.layout.camera_offsets(),
                core.layout.camera_dimensions(),
                poses.len(),
                0.5,
            ),
        }
        if core.signature.explicit_pose_masks() {
            optimizer::retract_poses_with_masks(
                poses,
                &mut core.candidate_poses,
                gauge,
                core.layout.camera_offsets(),
                core.layout.camera_dimensions(),
                core.signature.pose_masks(),
                &core.step,
            );
        } else {
            match core.signature.pose_variables() {
                CameraPoseVariables::RotationAndTranslation => optimizer::retract_poses(
                    poses,
                    &mut core.candidate_poses,
                    gauge,
                    core.layout.camera_offsets(),
                    core.layout.camera_dimensions(),
                    &core.step,
                    0,
                ),
                CameraPoseVariables::TranslationOnly => optimizer::retract_translations(
                    poses,
                    &mut core.candidate_poses,
                    gauge,
                    core.layout.camera_offsets(),
                    core.layout.camera_dimensions(),
                    &core.step,
                ),
            }
        }
    }
    let (point_step_rms, point_damping_norm, observation_cost) = candidate_points(
        model, poses, points, gauge, loss, noise, damping, fixed, priors, workspace,
    )?;
    let prior_cost = schur::prior_cost(
        model,
        &workspace.core.candidate_poses,
        &workspace.core.candidate_points,
        priors,
    )?;
    Some((
        point_step_rms,
        point_damping_norm,
        observation_cost + prior_cost,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn candidate_points(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> Option<(f64, f64, f64)> {
    #[cfg(feature = "parallel")]
    if workspace.core.point_back_substitution == SparsePointBackSubstitution::Recompute
        && fixed.points.is_empty()
        && priors.points.is_empty()
        && workspace.core.candidate_point_metrics.len() == points.len()
    {
        let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
        let CalibratedBackend::Retained(retained) = backend else {
            return None;
        };
        let pose_variables = core.signature.pose_variables();
        let evaluated = retained.evaluate_recomputed_candidates(
            RecomputedSchurContext {
                model,
                poses,
                points,
                gauge,
                loss,
                noise,
            },
            damping,
            core.layout.camera_offsets(),
            core.layout.camera_dimensions(),
            &core.step,
            &core.candidate_poses,
            &mut core.candidate_points,
            &mut core.candidate_point_metrics,
            pose_variables,
            core.signature.pose_masks(),
        );
        let evaluated = evaluated?;
        if !evaluated {
            return None;
        }
        return candidate_metrics_summary(&core.candidate_point_metrics, points.len());
    }

    #[cfg(feature = "parallel")]
    if workspace.core.point_back_substitution == SparsePointBackSubstitution::Cached
        && fixed.points.is_empty()
        && workspace.core.candidate_point_metrics.len() == points.len()
    {
        let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
        let cache = CachedPointStepData {
            valid: &core.point_cache_valid,
            inverse_gradients: &core.point_inverse_gradients,
            damping_diagonals: &core.point_damping_diagonals,
            cross_offsets: &core.point_cross_offsets,
            inverse_cross_values: &core.point_inverse_cross_values,
            point_pose_offsets: core.signature.point_pose_offsets(),
            point_poses: core.signature.point_poses(),
            camera_offsets: core.layout.camera_offsets(),
            camera_dimensions: core.layout.camera_dimensions(),
            camera_step: &core.step,
        };
        let candidate_poses = &core.candidate_poses;
        let candidate_points = &mut core.candidate_points;
        let point_metrics = &mut core.candidate_point_metrics;
        let CalibratedBackend::Retained(retained) = backend else {
            return None;
        };
        retained.install_parallel(|| {
            candidate_points
                .par_iter_mut()
                .zip(point_metrics.par_iter_mut())
                .enumerate()
                .for_each(|(point_index, (candidate, metrics))| {
                    let Some(step) = cached_point_step(model, point_index, cache) else {
                        *metrics = [f64::NAN; 3];
                        return;
                    };
                    let point = &points[point_index];
                    let Some(evaluated) = schur::candidate_point(
                        model,
                        poses,
                        candidate_poses,
                        point,
                        std::array::from_fn(|coordinate| {
                            point[coordinate] + gauge.scene_scale * step[coordinate]
                        }),
                        point_index,
                        loss,
                        noise,
                        gauge.scene_scale,
                    ) else {
                        *metrics = [f64::NAN; 3];
                        return;
                    };
                    *candidate = evaluated.position;
                    *metrics = if evaluated.moved {
                        [
                            evaluated.cost,
                            optimizer::squared_norm(&step),
                            point_damping_norm(cache.damping_diagonals[point_index], &step),
                        ]
                    } else {
                        [evaluated.cost, 0.0, 0.0]
                    };
                });
        })?;
        return candidate_metrics_summary(point_metrics, points.len());
    }

    let core = &mut workspace.core;
    let mut squared = 0.0;
    let mut point_damping = 0.0;
    let mut candidate_cost = 0.0;
    let mut point_prior_index = 0;
    for (point_index, point) in points.iter().enumerate() {
        let point_prior = priors
            .points
            .get(point_prior_index)
            .filter(|prior| prior.point == point_index);
        if point_prior.is_some() {
            point_prior_index += 1;
        }
        if !fixed.points.is_empty() && fixed.points[point_index] {
            core.candidate_points[point_index] = *point;
            candidate_cost += schur::fixed_point_cost(
                model,
                &core.candidate_poses,
                point,
                point_index,
                loss,
                noise,
            )?;
            continue;
        }
        let (step, damping_norm) = match core.point_back_substitution {
            SparsePointBackSubstitution::Cached => {
                let cache = CachedPointStepData {
                    valid: &core.point_cache_valid,
                    inverse_gradients: &core.point_inverse_gradients,
                    damping_diagonals: &core.point_damping_diagonals,
                    cross_offsets: &core.point_cross_offsets,
                    inverse_cross_values: &core.point_inverse_cross_values,
                    point_pose_offsets: core.signature.point_pose_offsets(),
                    point_poses: core.signature.point_poses(),
                    camera_offsets: core.layout.camera_offsets(),
                    camera_dimensions: core.layout.camera_dimensions(),
                    camera_step: &core.step,
                };
                let step = cached_point_step(model, point_index, cache)?;
                (
                    step,
                    point_damping_norm(cache.damping_diagonals[point_index], &step),
                )
            }
            SparsePointBackSubstitution::Recompute => {
                let layout = match core.signature.pose_variables() {
                    CameraPoseVariables::RotationAndTranslation => {
                        schur::VariableLayout::calibrated(
                            core.layout.camera_offsets(),
                            core.layout.camera_dimensions(),
                        )
                    }
                    CameraPoseVariables::TranslationOnly => {
                        schur::VariableLayout::calibrated_translation_only(
                            core.layout.camera_offsets(),
                            core.layout.camera_dimensions(),
                        )
                    }
                };
                let layout = layout.with_pose_masks(core.signature.pose_masks());
                let function = match core.signature.pose_variables() {
                    CameraPoseVariables::RotationAndTranslation => {
                        schur::point_step_with_prior_and_damping_norm
                    }
                    CameraPoseVariables::TranslationOnly => {
                        schur::point_step_with_prior_and_damping_norm_translation_only
                    }
                };
                function(
                    model,
                    poses,
                    point,
                    point_index,
                    gauge,
                    loss,
                    noise,
                    damping,
                    &core.step,
                    layout,
                    &mut core.cross,
                    &mut core.active_parameters,
                    &mut core.parameter_marks,
                    point_prior,
                )?
            }
        };
        let evaluated = schur::candidate_point(
            model,
            poses,
            &core.candidate_poses,
            point,
            std::array::from_fn(|coordinate| {
                point[coordinate] + gauge.scene_scale * step[coordinate]
            }),
            point_index,
            loss,
            noise,
            gauge.scene_scale,
        )?;
        core.candidate_points[point_index] = evaluated.position;
        candidate_cost += evaluated.cost;
        if evaluated.moved {
            squared += optimizer::squared_norm(&step);
            point_damping += damping_norm;
        }
    }
    Some((
        (squared
            / (0..points.len())
                .filter(|&point| fixed.points.is_empty() || !fixed.points[point])
                .count()
                .max(1) as f64)
            .sqrt(),
        point_damping,
        candidate_cost,
    ))
}

#[cfg(feature = "parallel")]
fn candidate_metrics_summary(
    point_metrics: &[[f64; 3]],
    point_count: usize,
) -> Option<(f64, f64, f64)> {
    let mut squared = 0.0;
    let mut damping_norm = 0.0;
    let mut candidate_cost = 0.0;
    for &[point_cost, step_squared, point_damping] in point_metrics {
        if !(point_cost.is_finite() && step_squared.is_finite() && point_damping.is_finite()) {
            return None;
        }
        candidate_cost += point_cost;
        squared += step_squared;
        damping_norm += point_damping;
    }
    Some((
        (squared / point_count as f64).sqrt(),
        damping_norm,
        candidate_cost,
    ))
}

#[derive(Clone, Copy)]
struct CachedPointStepData<'a> {
    valid: &'a [bool],
    inverse_gradients: &'a [[f64; 3]],
    damping_diagonals: &'a [[f64; 3]],
    cross_offsets: &'a [usize],
    inverse_cross_values: &'a PointCrossCache,
    point_pose_offsets: &'a [usize],
    point_poses: &'a [u32],
    camera_offsets: &'a [usize],
    camera_dimensions: &'a [usize],
    camera_step: &'a [f64],
}

pub(super) fn point_damping_norm(diagonal: [f64; 3], step: &[f64; 3]) -> f64 {
    diagonal
        .iter()
        .zip(step)
        .map(|(diagonal, step)| diagonal * step * step)
        .sum()
}

fn cached_point_step(
    _model: ObservationModel<'_>,
    point: usize,
    cache: CachedPointStepData<'_>,
) -> Option<[f64; 3]> {
    if !cache.valid[point] {
        return None;
    }
    let mut step = cache.inverse_gradients[point];
    let range = cache.cross_offsets[point]..cache.cross_offsets[point + 1];
    if range.end > cache.inverse_cross_values.len() {
        return None;
    }
    let mut cross_index = range.start;
    for &camera in
        &cache.point_poses[cache.point_pose_offsets[point]..cache.point_pose_offsets[point + 1]]
    {
        let camera = camera as usize;
        let offset = cache.camera_offsets[camera];
        for local in 0..cache.camera_dimensions[camera] {
            let inverse_cross = cache.inverse_cross_values.get(cross_index);
            cross_index += 1;
            for coordinate in 0..3 {
                step[coordinate] += inverse_cross[coordinate] * cache.camera_step[offset + local];
            }
        }
    }
    if cross_index != range.end {
        return None;
    }
    Some(step.map(|value| -value))
}
