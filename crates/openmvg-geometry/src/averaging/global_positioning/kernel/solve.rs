// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free BATA execution with deterministic large-problem parallel passes.

use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::api::{
    BataOptions, GlobalPositioningError, GlobalPositioningReport, ReducedSystemBackend,
};
use super::dense::solve_reduced_dense;
#[cfg(feature = "faer")]
use super::direct::{ScalarCholesky, ScalarSystem};
use super::pcg::{solve_direction_projector_pcg, solve_reduced_pcg};
use super::plan::GlobalPositioningPlan;
use super::robust::{
    RobustIteration, chordal_huber_width, cosine_cutoff, direction_cosine, update_robust_weights,
};
use super::workspace::{CameraTerm, DENSE_FALLBACK_MAX_DIMENSION, GlobalPositioningWorkspace};
use super::{DENSE_DIRECT_MAX_CAMERAS, RIDGE};

const DIRECTION_SCALE_MINIMUM: f64 = 1e-5;
/// Camera displacement, in units of the median baseline depth, below which an iteration has
/// settled.
const CONVERGENCE_DISPLACEMENT: f64 = 1e-4;
const MIXED_DIRECTION_GATE_FRACTION: f64 = 0.25;
#[cfg(feature = "parallel")]
const PARALLEL_MIN_CONSTRAINTS: usize = 250_000;
/// Memory the alternation's sparse factorization may hold (its factor, scratch, value array and
/// observation-pair table): the default bundle-adjustment workspace budget. Camera graphs that
/// fill in beyond it keep conjugate gradients.
#[cfg(feature = "faer")]
const ALTERNATION_DIRECT_BUDGET_BYTES: usize =
    openmvg_ba::NativeBundleAdjuster::AUTO_WORKSPACE_BYTES;

pub(super) fn solve_into(
    plan: &GlobalPositioningPlan,
    camera_centers: &mut [Vector3<f64>],
    points: &mut [Vector3<f64>],
    workspace: &mut GlobalPositioningWorkspace,
) -> Result<GlobalPositioningReport, GlobalPositioningError> {
    if camera_centers.len() != plan.camera_count {
        return Err(GlobalPositioningError::CameraOutputCountMismatch {
            expected: plan.camera_count,
            actual: camera_centers.len(),
        });
    }
    if points.len() != plan.point_count {
        return Err(GlobalPositioningError::PointOutputCountMismatch {
            expected: plan.point_count,
            actual: points.len(),
        });
    }
    if !workspace.is_compatible(plan) {
        return Err(GlobalPositioningError::WorkspaceMismatch);
    }

    initialize(plan, workspace)?;
    let parallel_execution = uses_parallel_execution(plan);
    let options = plan.options;
    let cosine_hard_cut = cosine_cutoff(options.hard_cut_deg);
    let cosine_mixed_hard_cut = cosine_cutoff(MIXED_DIRECTION_GATE_FRACTION * options.hard_cut_deg);
    // The graduated loss reaches its final width when the hard cut starts, and no iteration
    // before that point can certify convergence.
    let final_stage_start = options
        .hard_cut_start_iteration
        .max(options.warmup_iterations)
        .min(options.max_outer_iterations - 1);
    let mut completed_iterations = 0usize;
    let mut connectivity_fallback_count = 0usize;
    let mut last_used_connectivity_fallback = false;
    let mut last_anchor = 0usize;
    let mut last_backend = ReducedSystemBackend::DenseCholesky;
    let mut total_pcg_iteration_count = 0usize;

    for outer in 0..options.max_outer_iterations {
        let robust = RobustIteration {
            outer,
            delta: chordal_huber_width(huber_delta(&options, outer, final_stage_start)),
            cosine_hard_cut,
            cosine_mixed_hard_cut,
            options,
            cut: true,
        };
        update_robust_weights(plan, workspace, robust, parallel_execution);
        // A cut can remove the only observations that tie a camera to the rest. The ridge would
        // still factor such a system and place the stranded camera arbitrarily, so keep the graph
        // connected with Huber weights for this iteration, or fail.
        let mut used_fallback = false;
        if has_zero_weight(workspace) && !active_graph_is_connected(plan, workspace) {
            update_robust_weights(plan, workspace, robust.huber_only(), parallel_execution);
            if !active_graph_is_connected(plan, workspace) {
                return Err(GlobalPositioningError::Disconnected);
            }
            used_fallback = true;
            connectivity_fallback_count += 1;
        }
        last_used_connectivity_fallback = used_fallback;
        update_point_statistics(plan, workspace, parallel_execution);
        assemble_camera_system(plan, workspace);

        let anchor = pick_anchor(&workspace.camera_terms);
        let linear_result = solve_reduced(plan, workspace, anchor);
        // A failed reduced solve is a failure at any iteration: the caller's outputs are
        // transactional, so an abandoned run must not look like a converged one.
        let (backend, pcg_iteration_count) =
            linear_result.map_err(|_| GlobalPositioningError::LinearSolveFailed)?;
        last_anchor = anchor;
        last_backend = backend;
        total_pcg_iteration_count += pcg_iteration_count;

        recover_trial_points(plan, workspace, parallel_execution);
        update_trial_dots(plan, workspace, parallel_execution);
        normalize_trial_gauge(plan, workspace, parallel_execution);
        update_trial_scales(plan, workspace, parallel_execution);

        let maximum_camera_displacement =
            gauge_aligned_displacement(&workspace.cameras, &workspace.trial_cameras);
        std::mem::swap(&mut workspace.cameras, &mut workspace.trial_cameras);
        std::mem::swap(&mut workspace.points, &mut workspace.trial_points);
        std::mem::swap(
            &mut workspace.bearing_scale,
            &mut workspace.trial_bearing_scale,
        );
        std::mem::swap(
            &mut workspace.direction_scale,
            &mut workspace.trial_direction_scale,
        );

        completed_iterations = outer + 1;
        // Both iterates have unit median depth, so the displacement needs no further scale.
        if outer >= final_stage_start
            && !used_fallback
            && maximum_camera_displacement < CONVERGENCE_DISPLACEMENT
        {
            break;
        }
    }

    let polish_step_count = if options.polish_steps > 0 {
        super::polish::polish(
            plan,
            workspace,
            &options,
            options.polish_steps,
            !last_used_connectivity_fallback,
        )
        .accepted_steps
    } else {
        0
    };
    if workspace
        .cameras
        .iter()
        .map(Vector3::norm)
        .fold(0.0, f64::max)
        < 1e-6
    {
        return Err(GlobalPositioningError::DegenerateSolution);
    }
    if !workspace
        .cameras
        .iter()
        .chain(&workspace.points)
        .all(|value| value.iter().all(|entry| entry.is_finite()))
    {
        return Err(GlobalPositioningError::NonFiniteSolution);
    }

    let (combined, bearing, direction) = rms_angular_residuals(plan, workspace);
    camera_centers.copy_from_slice(&workspace.cameras);
    points.copy_from_slice(&workspace.points);
    Ok(GlobalPositioningReport {
        iteration_count: completed_iterations,
        polish_step_count,
        connectivity_fallback_count,
        anchor_camera: last_anchor,
        reduced_backend: last_backend,
        pcg_iteration_count: total_pcg_iteration_count,
        rms_angular_error: combined,
        bearing_rms_angular_error: bearing,
        direction_rms_angular_error: direction,
        parallel_execution,
    })
}

/// Chordal Huber width of iteration `outer`: geometric from the initial to the final width, which
/// it reaches at `final_stage_start`.
fn huber_delta(options: &BataOptions, outer: usize, final_stage_start: usize) -> f64 {
    if outer >= final_stage_start {
        return options.final_huber_delta_radians;
    }
    let progress = outer as f64 / final_stage_start as f64;
    options.initial_huber_delta_radians
        * (options.final_huber_delta_radians / options.initial_huber_delta_radians).powf(progress)
}

/// Largest camera displacement between two iterates after removing each one's centroid, which
/// makes the measure independent of the translation gauge (the anchor can change between
/// iterations).
fn gauge_aligned_displacement(current: &[Vector3<f64>], trial: &[Vector3<f64>]) -> f64 {
    let centroid = |centers: &[Vector3<f64>]| {
        centers.iter().sum::<Vector3<f64>>() / centers.len().max(1) as f64
    };
    let (current_centroid, trial_centroid) = (centroid(current), centroid(trial));
    current
        .iter()
        .zip(trial)
        .map(|(current, trial)| ((trial - trial_centroid) - (current - current_centroid)).norm())
        .fold(0.0, f64::max)
}

fn has_zero_weight(workspace: &GlobalPositioningWorkspace) -> bool {
    workspace.bearing_weight.contains(&0.0) || workspace.direction_weight.contains(&0.0)
}

/// Whether the observations with positive weight connect every camera: bearings join a camera to
/// a point, directions join two cameras.
pub(super) fn active_graph_is_connected(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
) -> bool {
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    fn join(parent: &mut [usize], first: usize, second: usize) {
        let (first, second) = (root(parent, first), root(parent, second));
        parent[first.max(second)] = first.min(second);
    }
    let parent = &mut workspace.connectivity_parent;
    for (node, entry) in parent.iter_mut().enumerate() {
        *entry = node;
    }
    for (index, observation) in plan.bearings.iter().enumerate() {
        if workspace.bearing_weight[index] > 0.0 {
            join(
                parent,
                observation.camera,
                plan.camera_count + observation.point,
            );
        }
    }
    for (index, edge) in plan.directions.iter().enumerate() {
        if workspace.direction_weight[index] > 0.0 {
            join(parent, edge.source, edge.target);
        }
    }
    let first = root(parent, 0);
    (1..plan.camera_count).all(|camera| root(parent, camera) == first)
}

fn uses_parallel_execution(plan: &GlobalPositioningPlan) -> bool {
    #[cfg(feature = "parallel")]
    {
        plan.bearings.len().saturating_add(plan.directions.len()) >= PARALLEL_MIN_CONSTRAINTS
            && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = plan;
        false
    }
}

fn update_point_statistics(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    _parallel_execution: bool,
) {
    #[cfg(feature = "parallel")]
    if _parallel_execution {
        let effective_weights = &workspace.bearing_effective_weight;
        let weights = &workspace.bearing_weight;
        let scales = &workspace.bearing_scale;
        workspace
            .point_weight
            .par_iter_mut()
            .zip(workspace.point_mean_offset.par_iter_mut())
            .enumerate()
            .for_each(|(point, (weight_out, offset_out))| {
                let mut weight_sum = 0.0;
                let mut offset_sum = Vector3::zeros();
                for &index in plan.point_bearing_range(point) {
                    weight_sum += effective_weights[index];
                    offset_sum += plan.bearings[index].ray * (weights[index] * scales[index]);
                }
                *weight_out = weight_sum;
                *offset_out = if weight_sum > 1e-12 {
                    offset_sum / weight_sum
                } else {
                    Vector3::zeros()
                };
            });
        return;
    }
    for point in 0..plan.point_count {
        let mut weight_sum = 0.0;
        let mut offset_sum = Vector3::zeros();
        for &index in plan.point_bearing_range(point) {
            weight_sum += workspace.bearing_effective_weight[index];
            offset_sum += plan.bearings[index].ray
                * (workspace.bearing_weight[index] * workspace.bearing_scale[index]);
        }
        workspace.point_weight[point] = weight_sum;
        workspace.point_mean_offset[point] = if weight_sum > 1e-12 {
            offset_sum / weight_sum
        } else {
            Vector3::zeros()
        };
    }
}

pub(super) fn assemble_camera_system(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
) {
    workspace.camera_terms.fill(CameraTerm::default());
    for (index, observation) in plan.bearings.iter().enumerate() {
        let effective_weight = workspace.bearing_effective_weight[index];
        let point_weight = workspace.point_weight[observation.point];
        let term = &mut workspace.camera_terms[observation.camera];
        term.weighted_degree += effective_weight;
        if point_weight < 1e-12 {
            continue;
        }
        term.diagonal += effective_weight * (1.0 - effective_weight / point_weight);
        term.right_hand_side += workspace.point_mean_offset[observation.point] * effective_weight
            - observation.ray * (workspace.bearing_weight[index] * workspace.bearing_scale[index]);
    }
    for (index, edge) in plan.directions.iter().enumerate() {
        let effective_weight = workspace.direction_effective_weight[index];
        let weighted_offset =
            edge.direction * (workspace.direction_weight[index] * workspace.direction_scale[index]);
        let source = &mut workspace.camera_terms[edge.source];
        source.diagonal += effective_weight;
        source.weighted_degree += effective_weight;
        source.right_hand_side -= weighted_offset;
        let target = &mut workspace.camera_terms[edge.target];
        target.diagonal += effective_weight;
        target.weighted_degree += effective_weight;
        target.right_hand_side += weighted_offset;
    }
    if plan.materialized_couplings {
        assemble_couplings(plan, workspace);
    }
}

fn recover_trial_points(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    _parallel_execution: bool,
) {
    #[cfg(feature = "parallel")]
    if _parallel_execution {
        let point_weights = &workspace.point_weight;
        let trial_cameras = &workspace.trial_cameras;
        let effective_weights = &workspace.bearing_effective_weight;
        let weights = &workspace.bearing_weight;
        let scales = &workspace.bearing_scale;
        let current_points = &workspace.points;
        workspace
            .trial_points
            .par_iter_mut()
            .enumerate()
            .for_each(|(point, output)| {
                let point_weight = point_weights[point];
                if point_weight < 1e-12 {
                    *output = current_points[point];
                    return;
                }
                let mut sum = Vector3::zeros();
                for &index in plan.point_bearing_range(point) {
                    let observation = plan.bearings[index];
                    sum += trial_cameras[observation.camera] * effective_weights[index]
                        + observation.ray * (weights[index] * scales[index]);
                }
                *output = sum / point_weight;
            });
        return;
    }
    for point in 0..plan.point_count {
        let point_weight = workspace.point_weight[point];
        if point_weight < 1e-12 {
            workspace.trial_points[point] = workspace.points[point];
            continue;
        }
        let mut sum = Vector3::zeros();
        for &index in plan.point_bearing_range(point) {
            let observation = plan.bearings[index];
            sum += workspace.trial_cameras[observation.camera]
                * workspace.bearing_effective_weight[index]
                + observation.ray
                    * (workspace.bearing_weight[index] * workspace.bearing_scale[index]);
        }
        workspace.trial_points[point] = sum / point_weight;
    }
}

fn update_trial_dots(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    _parallel_execution: bool,
) {
    #[cfg(feature = "parallel")]
    if _parallel_execution {
        let trial_points = &workspace.trial_points;
        let trial_cameras = &workspace.trial_cameras;
        workspace
            .bearing_dot
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let observation = plan.bearings[index];
                *output = observation
                    .ray
                    .dot(&(trial_points[observation.point] - trial_cameras[observation.camera]));
            });
        workspace
            .direction_dot
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let edge = plan.directions[index];
                *output = edge
                    .direction
                    .dot(&(trial_cameras[edge.target] - trial_cameras[edge.source]));
            });
        return;
    }
    for (index, observation) in plan.bearings.iter().enumerate() {
        workspace.bearing_dot[index] = observation.ray.dot(
            &(workspace.trial_points[observation.point]
                - workspace.trial_cameras[observation.camera]),
        );
    }
    for (index, edge) in plan.directions.iter().enumerate() {
        workspace.direction_dot[index] = edge
            .direction
            .dot(&(workspace.trial_cameras[edge.target] - workspace.trial_cameras[edge.source]));
    }
}

fn normalize_trial_gauge(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    parallel_execution: bool,
) -> f64 {
    workspace.depths.clear();
    for (index, observation) in plan.bearings.iter().enumerate() {
        if workspace.bearing_weight[index] > 0.5 * observation.base_weight
            && workspace.bearing_dot[index] > 0.0
        {
            workspace.depths.push(
                (workspace.trial_points[observation.point]
                    - workspace.trial_cameras[observation.camera])
                    .norm(),
            );
        }
    }
    if workspace.depths.is_empty() {
        for (index, edge) in plan.directions.iter().enumerate() {
            if workspace.direction_weight[index] > 0.5 * edge.base_weight
                && workspace.direction_dot[index] > 0.0
            {
                workspace.depths.push(
                    (workspace.trial_cameras[edge.target] - workspace.trial_cameras[edge.source])
                        .norm(),
                );
            }
        }
    }
    let depth = if workspace.depths.is_empty() {
        1.0
    } else {
        workspace.depths.sort_unstable_by(f64::total_cmp);
        workspace.depths[workspace.depths.len() / 2]
    };
    if depth > 1e-12 {
        normalize_trial_solution(workspace, 1.0 / depth, parallel_execution);
    }
    depth
}

fn normalize_trial_solution(
    workspace: &mut GlobalPositioningWorkspace,
    inverse: f64,
    _parallel_execution: bool,
) {
    #[cfg(feature = "parallel")]
    if _parallel_execution {
        workspace
            .trial_cameras
            .par_iter_mut()
            .for_each(|camera| *camera *= inverse);
        workspace
            .trial_points
            .par_iter_mut()
            .for_each(|point| *point *= inverse);
        return;
    }
    for camera in &mut workspace.trial_cameras {
        *camera *= inverse;
    }
    for point in &mut workspace.trial_points {
        *point *= inverse;
    }
}

fn update_trial_scales(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    _parallel_execution: bool,
) {
    #[cfg(feature = "parallel")]
    if _parallel_execution {
        let point_weights = &workspace.point_weight;
        let current_scales = &workspace.bearing_scale;
        let trial_points = &workspace.trial_points;
        let trial_cameras = &workspace.trial_cameras;
        workspace
            .trial_bearing_scale
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let observation = plan.bearings[index];
                *output = if point_weights[observation.point] < 1e-12 {
                    current_scales[index]
                } else {
                    let baseline =
                        trial_points[observation.point] - trial_cameras[observation.camera];
                    let squared_norm = baseline.norm_squared();
                    if squared_norm < 1e-12 {
                        current_scales[index]
                    } else {
                        (observation.ray.dot(&baseline) / squared_norm).max(DIRECTION_SCALE_MINIMUM)
                    }
                };
            });
        let current_scales = &workspace.direction_scale;
        let direction_weights = &workspace.direction_weight;
        workspace
            .trial_direction_scale
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let edge = plan.directions[index];
                *output = if plan.bearings.is_empty() && direction_weights[index] < 1e-12 {
                    current_scales[index]
                } else {
                    let baseline = trial_cameras[edge.target] - trial_cameras[edge.source];
                    let squared_norm = baseline.norm_squared();
                    if squared_norm < 1e-12 {
                        current_scales[index]
                    } else {
                        (edge.direction.dot(&baseline) / squared_norm).max(DIRECTION_SCALE_MINIMUM)
                    }
                };
            });
        return;
    }
    for (index, observation) in plan.bearings.iter().enumerate() {
        workspace.trial_bearing_scale[index] = if workspace.point_weight[observation.point] < 1e-12
        {
            workspace.bearing_scale[index]
        } else {
            let baseline = workspace.trial_points[observation.point]
                - workspace.trial_cameras[observation.camera];
            let squared_norm = baseline.norm_squared();
            if squared_norm < 1e-12 {
                workspace.bearing_scale[index]
            } else {
                (observation.ray.dot(&baseline) / squared_norm).max(DIRECTION_SCALE_MINIMUM)
            }
        };
    }
    for (index, edge) in plan.directions.iter().enumerate() {
        workspace.trial_direction_scale[index] =
            if plan.bearings.is_empty() && workspace.direction_weight[index] < 1e-12 {
                workspace.direction_scale[index]
            } else {
                let baseline =
                    workspace.trial_cameras[edge.target] - workspace.trial_cameras[edge.source];
                let squared_norm = baseline.norm_squared();
                if squared_norm < 1e-12 {
                    workspace.direction_scale[index]
                } else {
                    (edge.direction.dot(&baseline) / squared_norm).max(DIRECTION_SCALE_MINIMUM)
                }
            };
    }
}

fn initialize(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
) -> Result<(), GlobalPositioningError> {
    let mut random = openmvg_core::determinism::AlgorithmRng::new(
        plan.options
            .seed
            .derive("geometry/global-positioning/initialization/v1", &[]),
    );
    if plan.bearings.is_empty() {
        initialize_from_directions(plan, workspace)?;
    } else {
        for camera in &mut workspace.cameras {
            *camera = Vector3::new(
                random.symmetric_unit_f64(),
                random.symmetric_unit_f64(),
                random.symmetric_unit_f64(),
            );
        }
    }
    for point in &mut workspace.points {
        *point = Vector3::new(
            random.symmetric_unit_f64(),
            random.symmetric_unit_f64(),
            random.symmetric_unit_f64(),
        );
    }
    workspace.trial_cameras.fill(Vector3::zeros());
    workspace.trial_points.fill(Vector3::zeros());
    workspace.bearing_scale.fill(1.0);
    workspace.trial_bearing_scale.fill(1.0);
    workspace.bearing_weight.fill(1.0);
    workspace.bearing_effective_weight.fill(0.0);
    workspace.bearing_dot.fill(0.0);
    workspace.direction_scale.fill(1.0);
    if plan.bearings.is_empty() {
        for (index, edge) in plan.directions.iter().enumerate() {
            let baseline = workspace.cameras[edge.target] - workspace.cameras[edge.source];
            let squared_norm = baseline.norm_squared();
            workspace.direction_scale[index] = if squared_norm > 1e-12 {
                (edge.direction.dot(&baseline) / squared_norm).max(DIRECTION_SCALE_MINIMUM)
            } else {
                1.0
            };
        }
    }
    workspace.trial_direction_scale.fill(1.0);
    workspace.direction_weight.fill(1.0);
    workspace.direction_effective_weight.fill(1.0);
    workspace.direction_dot.fill(0.0);
    Ok(())
}

fn initialize_from_directions(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
) -> Result<(), GlobalPositioningError> {
    const ITERATIONS: usize = 8;
    const INITIAL_CAUCHY_WIDTH: f64 = 0.5;
    const FINAL_CAUCHY_WIDTH: f64 = 0.05;

    workspace.pcg_rhs.fill(Vector3::zeros());
    for (weight, edge) in workspace
        .direction_init_weight
        .iter_mut()
        .zip(&plan.directions)
    {
        *weight = edge.base_weight;
    }
    let mut anchor = 0usize;
    let mut best_degree = -1.0;
    workspace.direction_init_degree.fill(0.0);
    for edge in &plan.directions {
        workspace.direction_init_degree[edge.source] += edge.base_weight;
        workspace.direction_init_degree[edge.target] += edge.base_weight;
    }
    for (camera, degree) in workspace.direction_init_degree.iter().enumerate() {
        if *degree > best_degree {
            best_degree = *degree;
            anchor = camera;
        }
    }

    for iteration in 0..ITERATIONS {
        workspace.pcg_rhs.fill(Vector3::zeros());
        workspace
            .direction_init_diagonal
            .fill(Vector3::repeat(RIDGE));
        for (edge, weight) in plan.directions.iter().zip(&workspace.direction_init_weight) {
            if edge.source != anchor {
                workspace.pcg_rhs[edge.source] -= edge.direction * *weight;
                for component in 0..3 {
                    workspace.direction_init_diagonal[edge.source][component] +=
                        *weight * (1.0 - edge.direction[component] * edge.direction[component]);
                }
            }
            if edge.target != anchor {
                workspace.pcg_rhs[edge.target] += edge.direction * *weight;
                for component in 0..3 {
                    workspace.direction_init_diagonal[edge.target][component] +=
                        *weight * (1.0 - edge.direction[component] * edge.direction[component]);
                }
            }
        }
        workspace.pcg_rhs[anchor] = Vector3::zeros();
        if !solve_direction_projector_pcg(plan, workspace, anchor) {
            return Err(GlobalPositioningError::DirectionInitializationFailed);
        }

        let mut signed_gauge = 0.0;
        for (edge, weight) in plan.directions.iter().zip(&workspace.direction_init_weight) {
            signed_gauge += *weight
                * edge
                    .direction
                    .dot(&(workspace.cameras[edge.target] - workspace.cameras[edge.source]));
        }
        if !signed_gauge.is_finite() || signed_gauge.abs() <= 1e-12 {
            return Err(GlobalPositioningError::DirectionInitializationFailed);
        }
        if signed_gauge < 0.0 {
            for camera in &mut workspace.cameras {
                *camera = -*camera;
            }
        }
        workspace.depths.clear();
        for edge in &plan.directions {
            let length = (workspace.cameras[edge.target] - workspace.cameras[edge.source]).norm();
            if length.is_finite() && length > 1e-12 {
                workspace.depths.push(length);
            }
        }
        if workspace.depths.is_empty() {
            return Err(GlobalPositioningError::DirectionInitializationFailed);
        }
        workspace.depths.sort_unstable_by(f64::total_cmp);
        let median = workspace.depths[workspace.depths.len() / 2];
        for camera in &mut workspace.cameras {
            *camera /= median;
        }

        let progress = iteration as f64 / (ITERATIONS - 1) as f64;
        let width =
            INITIAL_CAUCHY_WIDTH * (FINAL_CAUCHY_WIDTH / INITIAL_CAUCHY_WIDTH).powf(progress);
        let squared_width = width * width;
        for (index, edge) in plan.directions.iter().enumerate() {
            let baseline = workspace.cameras[edge.target] - workspace.cameras[edge.source];
            let norm = baseline.norm();
            let squared_line_error = if norm > 1e-12 {
                let cosine = edge.direction.dot(&baseline) / norm;
                (1.0 - cosine * cosine).clamp(0.0, 1.0)
            } else {
                1.0
            };
            workspace.direction_init_weight[index] =
                edge.base_weight / (1.0 + squared_line_error / squared_width);
        }
    }
    workspace.cameras[anchor] = Vector3::zeros();
    if workspace
        .cameras
        .iter()
        .all(|camera| camera.iter().all(|value| value.is_finite()))
    {
        Ok(())
    } else {
        Err(GlobalPositioningError::DirectionInitializationFailed)
    }
}

fn pick_anchor(camera_terms: &[CameraTerm]) -> usize {
    let mut anchor = 0usize;
    let mut best_degree = -1.0;
    for (index, term) in camera_terms.iter().enumerate() {
        if term.weighted_degree > best_degree {
            best_degree = term.weighted_degree;
            anchor = index;
        }
    }
    anchor
}

fn solve_reduced(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    anchor: usize,
) -> Result<(ReducedSystemBackend, usize), GlobalPositioningError> {
    #[cfg(feature = "faer")]
    if solve_reduced_direct(plan, workspace, anchor) == Some(true) {
        return Ok((ReducedSystemBackend::SparseCholesky, 0));
    }
    let reduced_dimension = plan.camera_count - 1;
    if plan.camera_count <= DENSE_DIRECT_MAX_CAMERAS {
        solve_reduced_dense(plan, workspace, anchor)?;
        return Ok((ReducedSystemBackend::DenseCholesky, 0));
    }
    match solve_reduced_pcg(plan, workspace, anchor) {
        Ok(iterations) => Ok((
            if plan.materialized_couplings {
                ReducedSystemBackend::SparseCouplingPcg
            } else {
                ReducedSystemBackend::MatrixFreePcg
            },
            iterations,
        )),
        Err(_) if reduced_dimension <= DENSE_FALLBACK_MAX_DIMENSION => {
            solve_reduced_dense(plan, workspace, anchor)?;
            Ok((ReducedSystemBackend::DenseFallbackAfterPcg, 0))
        }
        Err(error) => Err(error),
    }
}

/// Solve the scalar camera system with the sparse Cholesky of [`ScalarCholesky`], analyzed at the
/// first alternation iteration of the workspace. `None` when the factorization is unavailable for
/// this plan (its fill exceeds [`ALTERNATION_DIRECT_BUDGET_BYTES`]); otherwise whether the solve
/// succeeded. Either way the caller keeps the conjugate-gradient and dense paths as fallbacks.
#[cfg(feature = "faer")]
fn solve_reduced_direct(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    anchor: usize,
) -> Option<bool> {
    if workspace
        .alternation_direct
        .as_ref()
        .is_some_and(|cholesky| !cholesky.matches(plan))
    {
        workspace.alternation_direct = None;
    }
    if workspace.alternation_direct.is_none() {
        if workspace.alternation_direct_unavailable {
            return None;
        }
        match ScalarCholesky::analyze(plan, ALTERNATION_DIRECT_BUDGET_BYTES) {
            Some(cholesky) => workspace.alternation_direct = Some(Box::new(cholesky)),
            None => {
                workspace.alternation_direct_unavailable = true;
                return None;
            }
        }
    }
    let mut direct = workspace.alternation_direct.take()?;
    let solved = direct.solve(
        plan,
        &ScalarSystem {
            camera_terms: &workspace.camera_terms,
            point_weight: &workspace.point_weight,
            bearing_effective_weight: &workspace.bearing_effective_weight,
            direction_effective_weight: &workspace.direction_effective_weight,
            anchor,
        },
        &mut workspace.trial_cameras,
    );
    workspace.alternation_direct = Some(direct);
    Some(solved)
}

fn assemble_couplings(plan: &GlobalPositioningPlan, workspace: &mut GlobalPositioningWorkspace) {
    workspace.coupling_values.fill(0.0);
    for point in 0..plan.point_count {
        let indices = plan.point_bearing_range(point);
        let point_weight = workspace.point_weight[point];
        if point_weight < 1e-12 {
            continue;
        }
        let mut pair = plan.point_pair_offsets[point];
        for left in 0..indices.len() {
            let left_index = indices[left];
            for &right_index in &indices[left + 1..] {
                let slot = plan.point_pair_slots[pair];
                workspace.coupling_values[slot] -= workspace.bearing_effective_weight[left_index]
                    * workspace.bearing_effective_weight[right_index]
                    / point_weight;
                pair += 1;
            }
        }
        debug_assert_eq!(pair, plan.point_pair_offsets[point + 1]);
    }
    for (index, &slot) in plan.direction_coupling_slots.iter().enumerate() {
        workspace.coupling_values[slot] -= workspace.direction_effective_weight[index];
    }
}

fn rms_angular_residuals(
    plan: &GlobalPositioningPlan,
    workspace: &GlobalPositioningWorkspace,
) -> (f64, Option<f64>, Option<f64>) {
    let mut bearing_sum = 0.0;
    for observation in &plan.bearings {
        let baseline = workspace.points[observation.point] - workspace.cameras[observation.camera];
        let cosine = direction_cosine(&observation.ray, baseline).unwrap_or(0.0);
        let angle = cosine.acos();
        bearing_sum += angle * angle;
    }
    let mut direction_sum = 0.0;
    for edge in &plan.directions {
        let baseline = workspace.cameras[edge.target] - workspace.cameras[edge.source];
        let cosine = direction_cosine(&edge.direction, baseline).unwrap_or(0.0);
        let angle = cosine.acos();
        direction_sum += angle * angle;
    }
    let count = plan.bearings.len() + plan.directions.len();
    let combined = ((bearing_sum + direction_sum) / count as f64).sqrt();
    let bearing =
        (!plan.bearings.is_empty()).then(|| (bearing_sum / plan.bearings.len() as f64).sqrt());
    let direction = (!plan.directions.is_empty())
        .then(|| (direction_sum / plan.directions.len() as f64).sqrt());
    (combined, bearing, direction)
}

#[cfg(test)]
mod operator_tests {
    use super::super::api::{BataOptions, GlobalPositioningProblem, PositioningObservation};
    use super::super::pcg::ReducedOperator;
    use super::*;

    fn prepared() -> (GlobalPositioningPlan, GlobalPositioningWorkspace) {
        let cameras = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.1, 0.0),
            Vector3::new(0.2, 1.0, 0.1),
        ];
        let points = [
            Vector3::new(0.2, 0.1, 3.0),
            Vector3::new(1.1, 0.8, 4.0),
            Vector3::new(-0.4, 1.2, 3.5),
            Vector3::new(0.7, -0.4, 2.8),
        ];
        let mut observations = Vec::new();
        for (point, position) in points.iter().enumerate() {
            for (camera, center) in cameras.iter().enumerate() {
                observations.push(PositioningObservation::bearing(
                    camera,
                    point,
                    (*position - *center).normalize(),
                    1.0,
                ));
            }
        }
        let mut plan = GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: cameras.len(),
                point_count: points.len(),
                observations: &observations,
            },
            BataOptions::default(),
        )
        .unwrap();
        plan.materialized_couplings = true;
        plan.coupling_endpoints = vec![(0, 1), (0, 2), (1, 2)];
        plan.point_pair_offsets = vec![0, 3, 6, 9, 12];
        plan.point_pair_slots = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let mut workspace = plan.try_new_workspace().unwrap();
        for (index, weight) in workspace.bearing_effective_weight.iter_mut().enumerate() {
            *weight = 0.5 + 0.03 * index as f64;
        }
        workspace.camera_terms.fill(CameraTerm::default());
        for point in 0..plan.point_count {
            let indices = plan.point_bearing_range(point);
            let point_weight: f64 = indices
                .iter()
                .map(|index| workspace.bearing_effective_weight[*index])
                .sum();
            workspace.point_weight[point] = point_weight;
            for &index in indices {
                let observation = plan.bearings[index];
                let weight = workspace.bearing_effective_weight[index];
                workspace.camera_terms[observation.camera].diagonal +=
                    weight * (1.0 - weight / point_weight);
            }
        }
        for (index, term) in workspace.camera_terms.iter_mut().enumerate() {
            term.right_hand_side = Vector3::new(
                0.1 + index as f64,
                -0.2 * index as f64,
                0.3 - 0.1 * index as f64,
            );
        }
        assemble_couplings(&plan, &mut workspace);
        (plan, workspace)
    }

    #[test]
    fn materialized_and_matrix_free_operators_match() {
        let (mut plan, workspace) = prepared();
        let input = [
            Vector3::zeros(),
            Vector3::new(0.4, -0.2, 0.7),
            Vector3::new(-0.3, 0.6, 0.1),
        ];
        let mut materialized = [Vector3::zeros(); 3];
        ReducedOperator {
            plan: &plan,
            bearing_weights: &workspace.bearing_effective_weight,
            direction_weights: &workspace.direction_effective_weight,
            coupling_values: &workspace.coupling_values,
            camera_terms: &workspace.camera_terms,
        }
        .apply(0, &input, &mut materialized);
        plan.materialized_couplings = false;
        let mut matrix_free = [Vector3::zeros(); 3];
        ReducedOperator {
            plan: &plan,
            bearing_weights: &workspace.bearing_effective_weight,
            direction_weights: &workspace.direction_effective_weight,
            coupling_values: &workspace.coupling_values,
            camera_terms: &workspace.camera_terms,
        }
        .apply(0, &input, &mut matrix_free);
        for (left, right) in materialized.iter().zip(matrix_free) {
            assert!((*left - right).norm() < 1e-12);
        }
    }

    #[test]
    fn nonmaterialized_dense_fallback_matches_compiled_couplings() {
        let (mut plan, mut workspace) = prepared();
        solve_reduced_dense(&plan, &mut workspace, 0).unwrap();
        let materialized = workspace.trial_cameras.clone();
        plan.materialized_couplings = false;
        solve_reduced_dense(&plan, &mut workspace, 0).unwrap();
        for (left, right) in materialized.iter().zip(&workspace.trial_cameras) {
            assert!((*left - *right).norm() < 1e-12);
        }
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    #[test]
    fn huber_width_is_geometric_and_final_at_the_hard_cut() {
        let options = BataOptions::default();
        let start = options.hard_cut_start_iteration;
        assert_eq!(
            huber_delta(&options, 0, start),
            options.initial_huber_delta_radians
        );
        assert_eq!(
            huber_delta(&options, start, start),
            options.final_huber_delta_radians
        );
        assert_eq!(
            huber_delta(&options, start + 9, start),
            options.final_huber_delta_radians
        );
        let widths: Vec<f64> = (0..=start)
            .map(|outer| huber_delta(&options, outer, start))
            .collect();
        assert!(widths.windows(2).all(|pair| pair[1] < pair[0]));
        let ratio = widths[1] / widths[0];
        assert!(
            widths
                .windows(2)
                .all(|pair| (pair[1] / pair[0] - ratio).abs() < 1e-12)
        );
    }

    #[test]
    fn displacement_ignores_the_translation_gauge() {
        let current = [Vector3::zeros(), Vector3::x(), Vector3::y()];
        let shifted = current.map(|center| center + Vector3::new(3.0, -2.0, 5.0));
        assert!(gauge_aligned_displacement(&current, &shifted) < 1e-12);
        let mut moved = shifted;
        moved[2] += Vector3::new(0.0, 0.5, 0.0);
        assert!(gauge_aligned_displacement(&current, &moved) > 0.1);
    }

    #[test]
    fn positioning_short_aligned_baseline_has_true_angle_and_unit_weight() {
        use super::super::api::{GlobalPositioningProblem, PositioningObservation};
        let options = BataOptions {
            warmup_iterations: 0,
            hard_cut_start_iteration: 0,
            ..BataOptions::default()
        };
        let observations = [PositioningObservation::relative_direction(
            0,
            1,
            Vector3::x(),
            1.0,
        )];
        let plan = GlobalPositioningPlan::try_new(
            GlobalPositioningProblem {
                camera_count: 2,
                point_count: 0,
                observations: &observations,
            },
            options,
        )
        .expect("two-camera direction plan");
        let mut workspace = plan.try_new_workspace().expect("tiny workspace");
        workspace.cameras[0] = Vector3::zeros();
        workspace.cameras[1] = 1.0e-12 * Vector3::x();
        workspace.direction_scale.fill(1.0);
        let robust = RobustIteration {
            outer: 0,
            delta: chordal_huber_width(options.final_huber_delta_radians),
            cosine_hard_cut: cosine_cutoff(options.hard_cut_deg),
            cosine_mixed_hard_cut: cosine_cutoff(0.25 * options.hard_cut_deg),
            options,
            cut: true,
        };
        update_robust_weights(&plan, &mut workspace, robust, false);
        assert_eq!(workspace.direction_weight[0], 1.0);
        assert_eq!(
            rms_angular_residuals(&plan, &workspace),
            (0.0, None, Some(0.0))
        );
        assert_eq!(
            direction_cosine(&Vector3::x(), 1.0e-12 * Vector3::x()),
            Some(1.0)
        );
        assert_eq!(direction_cosine(&Vector3::x(), Vector3::zeros()), None);
        assert_eq!(
            direction_cosine(&Vector3::x(), Vector3::new(f64::INFINITY, 0.0, 0.0)),
            None
        );
    }
}
