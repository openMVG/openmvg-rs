// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candidate point back-substitution and evaluation.

use crate::kernel::RelativePose;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::optimizer::squared_norm;
use crate::kernel::multi_view::schur::{self, GaugeState};
use crate::kernel::multi_view::self_calibration::{
    RadialCamera, SelfCalibratingMeasurements, radial_model,
};
use crate::kernel::multi_view::sparse::api::SparsePointBackSubstitution;
use crate::kernel::multi_view::sparse::retained::{PointCrossRows, PointCrossStorage};
use crate::kernel::multi_view::sparse::workspace::{
    DirectAssemblyState, PointCrossCache, SelfCalibrationBackend, SparseSelfCalibrationWorkspace,
};
use crate::kernel::multi_view::{BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks};
use crate::numerics::refine::NoiseModel;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn candidate_points(
    measurements: SelfCalibratingMeasurements<'_>,
    cameras: &[RadialCamera],
    camera_to_group: Option<&[usize]>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: crate::numerics::refine::Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseSelfCalibrationWorkspace,
) -> Option<(f64, f64, f64)> {
    #[cfg(feature = "parallel")]
    if workspace.sparse.point_back_substitution == SparsePointBackSubstitution::Cached
        && fixed.points.is_empty()
        && priors.points.is_empty()
        && workspace.sparse.candidate_point_metrics.len() == points.len()
    {
        let SparseSelfCalibrationWorkspace {
            sparse,
            backend,
            candidate_cameras,
            ..
        } = workspace;
        let candidate_model = radial_model(measurements, candidate_cameras, camera_to_group);
        let candidate_poses = &sparse.candidate_poses;
        let candidate_points = &mut sparse.candidate_points;
        let point_metrics = &mut sparse.candidate_point_metrics;
        let point_parameters = match &*backend {
            SelfCalibrationBackend::Iterative(retained) => retained.point_parameters(),
            SelfCalibrationBackend::Direct(DirectAssemblyState::ColumnOwned { .. }) => {
                &sparse.point_parameters
            }
            SelfCalibrationBackend::Direct(DirectAssemblyState::Serial) => return None,
        };
        match &sparse.point_inverse_cross_values {
            PointCrossCache::F64(inverse_cross_values) => backend.install_parallel(|| {
                evaluate_cached_candidate_points(
                    points,
                    poses,
                    gauge,
                    candidate_model,
                    candidate_poses,
                    loss,
                    noise,
                    &sparse.point_damping_diagonals,
                    CachedSelfCalibrationPointStepData {
                        valid: &sparse.point_cache_valid,
                        inverse_gradients: &sparse.point_inverse_gradients,
                        cross_offsets: &sparse.point_cross_offsets,
                        inverse_cross_values: inverse_cross_values.as_slice(),
                        point_parameters,
                        camera_step: &sparse.step,
                    },
                    candidate_points,
                    point_metrics,
                );
            })?,
            PointCrossCache::F32(inverse_cross_values) => backend.install_parallel(|| {
                evaluate_cached_candidate_points(
                    points,
                    poses,
                    gauge,
                    candidate_model,
                    candidate_poses,
                    loss,
                    noise,
                    &sparse.point_damping_diagonals,
                    CachedSelfCalibrationPointStepData {
                        valid: &sparse.point_cache_valid,
                        inverse_gradients: &sparse.point_inverse_gradients,
                        cross_offsets: &sparse.point_cross_offsets,
                        inverse_cross_values: inverse_cross_values.as_slice(),
                        point_parameters,
                        camera_step: &sparse.step,
                    },
                    candidate_points,
                    point_metrics,
                );
            })?,
            PointCrossCache::Recomputed { .. } => return None,
        }
        let (point_step_rms, point_damping_norm, observation_cost) =
            candidate_metrics_summary(point_metrics, points.len())?;
        let prior_cost =
            schur::prior_cost(candidate_model, candidate_poses, candidate_points, priors)?;
        return Some((
            point_step_rms,
            point_damping_norm,
            observation_cost + prior_cost,
        ));
    }

    let current_model = radial_model(measurements, cameras, camera_to_group);
    let SparseSelfCalibrationWorkspace {
        sparse,
        backend,
        candidate_cameras,
        ..
    } = workspace;
    let candidate_model = radial_model(measurements, candidate_cameras, camera_to_group);
    let cached_point_parameters = match backend {
        SelfCalibrationBackend::Iterative(retained) => Some(retained.point_parameters()),
        SelfCalibrationBackend::Direct(DirectAssemblyState::ColumnOwned { .. }) => {
            Some(sparse.point_parameters.as_slice())
        }
        SelfCalibrationBackend::Direct(DirectAssemblyState::Serial) => None,
    };
    let layout = schur::VariableLayout {
        offsets: sparse.layout.camera_offsets(),
        pose_dimensions: sparse.layout.camera_dimensions(),
        intrinsic_offsets: sparse.layout.intrinsic_offsets(),
        intrinsic_dimensions: sparse.layout.intrinsic_dimensions(),
        pose_masks: &[],
        pose_variables: CameraPoseVariables::RotationAndTranslation,
    };
    let mut squared = 0.0;
    let mut point_damping_norm = 0.0;
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
            sparse.candidate_points[point_index] = *point;
            candidate_cost += schur::fixed_point_cost(
                candidate_model,
                &sparse.candidate_poses,
                point,
                point_index,
                loss,
                noise,
            )?;
            continue;
        }
        let (step, damping_norm) = match sparse.point_back_substitution {
            SparsePointBackSubstitution::Cached => {
                let step = cached_self_calibration_point_step_from_cache(
                    point_index,
                    &sparse.point_cache_valid,
                    &sparse.point_inverse_gradients,
                    &sparse.point_cross_offsets,
                    &sparse.point_inverse_cross_values,
                    cached_point_parameters?,
                    &sparse.step,
                )?;
                (
                    step,
                    cached_point_damping_norm(sparse.point_damping_diagonals[point_index], &step),
                )
            }
            SparsePointBackSubstitution::Recompute => {
                schur::point_step_with_prior_and_damping_norm(
                    current_model,
                    poses,
                    point,
                    point_index,
                    gauge,
                    loss,
                    noise,
                    damping,
                    &sparse.step,
                    layout,
                    &mut sparse.cross,
                    &mut sparse.active_parameters,
                    &mut sparse.parameter_marks,
                    point_prior,
                )?
            }
        };
        let evaluated = schur::candidate_point(
            candidate_model,
            poses,
            &sparse.candidate_poses,
            point,
            std::array::from_fn(|coordinate| {
                point[coordinate] + gauge.scene_scale * step[coordinate]
            }),
            point_index,
            loss,
            noise,
            gauge.scene_scale,
        )?;
        sparse.candidate_points[point_index] = evaluated.position;
        candidate_cost += evaluated.cost;
        if evaluated.moved {
            point_damping_norm += damping_norm;
            squared += squared_norm(&step);
        }
    }
    candidate_cost += schur::prior_cost(
        candidate_model,
        &sparse.candidate_poses,
        &sparse.candidate_points,
        priors,
    )?;
    Some((
        (squared / points.len() as f64).sqrt(),
        point_damping_norm,
        candidate_cost,
    ))
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn evaluate_cached_candidate_points<C: PointCrossStorage + ?Sized>(
    points: &[[f64; 3]],
    poses: &[RelativePose],
    gauge: GaugeState,
    candidate_model: ObservationModel<'_>,
    candidate_poses: &[RelativePose],
    loss: crate::numerics::refine::Loss,
    noise: NoiseModel<'_>,
    damping_diagonals: &[[f64; 3]],
    cache: CachedSelfCalibrationPointStepData<'_, C>,
    candidate_points: &mut [[f64; 3]],
    point_metrics: &mut [[f64; 3]],
) {
    candidate_points
        .par_iter_mut()
        .zip(point_metrics.par_iter_mut())
        .enumerate()
        .for_each(|(point_index, (candidate, metrics))| {
            let Some(step) = cached_self_calibration_point_step(point_index, &cache) else {
                *metrics = [f64::NAN; 3];
                return;
            };
            let point = &points[point_index];
            let Some(evaluated) = schur::candidate_point(
                candidate_model,
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
                    squared_norm(&step),
                    cached_point_damping_norm(damping_diagonals[point_index], &step),
                ]
            } else {
                [evaluated.cost, 0.0, 0.0]
            };
        });
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

struct CachedSelfCalibrationPointStepData<'a, C: PointCrossStorage + ?Sized> {
    valid: &'a [bool],
    inverse_gradients: &'a [[f64; 3]],
    cross_offsets: &'a [usize],
    inverse_cross_values: &'a C,
    point_parameters: &'a [u32],
    camera_step: &'a [f64],
}

#[allow(clippy::too_many_arguments)]
fn cached_self_calibration_point_step_from_cache(
    point: usize,
    valid: &[bool],
    inverse_gradients: &[[f64; 3]],
    cross_offsets: &[usize],
    inverse_cross_values: &PointCrossCache,
    point_parameters: &[u32],
    camera_step: &[f64],
) -> Option<[f64; 3]> {
    match inverse_cross_values {
        PointCrossCache::F64(values) => cached_self_calibration_point_step(
            point,
            &CachedSelfCalibrationPointStepData {
                valid,
                inverse_gradients,
                cross_offsets,
                inverse_cross_values: values.as_slice(),
                point_parameters,
                camera_step,
            },
        ),
        PointCrossCache::F32(values) => cached_self_calibration_point_step(
            point,
            &CachedSelfCalibrationPointStepData {
                valid,
                inverse_gradients,
                cross_offsets,
                inverse_cross_values: values.as_slice(),
                point_parameters,
                camera_step,
            },
        ),
        PointCrossCache::Recomputed { .. } => None,
    }
}

fn cached_self_calibration_point_step<C: PointCrossStorage + ?Sized>(
    point: usize,
    cache: &CachedSelfCalibrationPointStepData<'_, C>,
) -> Option<[f64; 3]> {
    if !cache.valid[point] {
        return None;
    }
    let mut step = cache.inverse_gradients[point];
    let range = cache.cross_offsets[point]..cache.cross_offsets[point + 1];
    if range.end > cache.point_parameters.len() {
        return None;
    }
    let inverse_crosses = cache.inverse_cross_values.rows(range.clone());
    if inverse_crosses.len() != range.len() {
        return None;
    }
    for (local, cross_index) in range.enumerate() {
        let parameter = cache.point_parameters[cross_index] as usize;
        let value = *cache.camera_step.get(parameter)?;
        let inverse_cross = inverse_crosses.row(local);
        for coordinate in 0..3 {
            step[coordinate] += inverse_cross[coordinate] * value;
        }
    }
    Some(step.map(|value| -value))
}

fn cached_point_damping_norm(diagonal: [f64; 3], step: &[f64; 3]) -> f64 {
    diagonal
        .iter()
        .zip(step)
        .map(|(diagonal, step)| diagonal * step * step)
        .sum()
}
