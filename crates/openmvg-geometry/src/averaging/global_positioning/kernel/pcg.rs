// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Preconditioned conjugate-gradient solves: the fused three-right-hand-side reduced
//! camera system and the projector system of direction-only initialization.

use nalgebra::Vector3;

use super::RIDGE;
use super::api::GlobalPositioningError;
use super::plan::{Direction, GlobalPositioningPlan};
use super::workspace::{CameraTerm, GlobalPositioningWorkspace};

const PCG_RELATIVE_TOLERANCE: f64 = 1e-10;
const PCG_ABSOLUTE_TOLERANCE: f64 = 1e-12;

pub(super) fn solve_reduced_pcg(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    anchor: usize,
) -> Result<usize, GlobalPositioningError> {
    workspace.trial_cameras.fill(Vector3::zeros());
    for (index, (right, term)) in workspace
        .pcg_rhs
        .iter_mut()
        .zip(&workspace.camera_terms)
        .enumerate()
    {
        *right = if index == anchor {
            Vector3::zeros()
        } else {
            term.right_hand_side
        };
    }
    workspace.pcg_residual.copy_from_slice(&workspace.pcg_rhs);
    for index in 0..plan.camera_count {
        if index == anchor {
            workspace.pcg_preconditioned[index] = Vector3::zeros();
            continue;
        }
        let diagonal = workspace.camera_terms[index].diagonal + RIDGE;
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(GlobalPositioningError::LinearSolveFailed);
        }
        workspace.pcg_preconditioned[index] = workspace.pcg_residual[index] / diagonal;
    }
    workspace
        .pcg_search
        .copy_from_slice(&workspace.pcg_preconditioned);
    let right_norm_squared = component_dot(&workspace.pcg_rhs, &workspace.pcg_rhs, anchor);
    let mut old_dot = component_dot(
        &workspace.pcg_residual,
        &workspace.pcg_preconditioned,
        anchor,
    );
    let absolute_squared = PCG_ABSOLUTE_TOLERANCE * PCG_ABSOLUTE_TOLERANCE;
    let relative_squared = PCG_RELATIVE_TOLERANCE * PCG_RELATIVE_TOLERANCE;
    let target = Vector3::new(
        absolute_squared.max(relative_squared * right_norm_squared.x),
        absolute_squared.max(relative_squared * right_norm_squared.y),
        absolute_squared.max(relative_squared * right_norm_squared.z),
    );
    let mut active = [
        right_norm_squared.x > target.x,
        right_norm_squared.y > target.y,
        right_norm_squared.z > target.z,
    ];
    let maximum_iterations = (8 * (plan.camera_count - 1)).clamp(64, 4000);
    let operator = ReducedOperator {
        plan,
        bearing_weights: &workspace.bearing_effective_weight,
        direction_weights: &workspace.direction_effective_weight,
        coupling_values: &workspace.coupling_values,
        camera_terms: &workspace.camera_terms,
    };
    let mut iterations = 0usize;
    for iteration in 0..maximum_iterations {
        if !active.iter().any(|value| *value) {
            break;
        }
        iterations = iteration + 1;
        operator.apply(anchor, &workspace.pcg_search, &mut workspace.pcg_applied);
        let denominator = component_dot(&workspace.pcg_search, &workspace.pcg_applied, anchor);
        let mut alpha = [0.0; 3];
        for component in 0..3 {
            if !active[component] {
                continue;
            }
            if !denominator[component].is_finite()
                || denominator[component] <= f64::EPSILON * old_dot[component].abs()
            {
                return Err(GlobalPositioningError::LinearSolveFailed);
            }
            alpha[component] = old_dot[component] / denominator[component];
        }
        for index in 0..plan.camera_count {
            if index == anchor {
                continue;
            }
            for component in 0..3 {
                if active[component] {
                    workspace.trial_cameras[index][component] +=
                        alpha[component] * workspace.pcg_search[index][component];
                    workspace.pcg_residual[index][component] -=
                        alpha[component] * workspace.pcg_applied[index][component];
                }
            }
        }
        let residual_norm_squared =
            component_dot(&workspace.pcg_residual, &workspace.pcg_residual, anchor);
        for component in 0..3 {
            if active[component] && residual_norm_squared[component] <= target[component] {
                active[component] = false;
            }
        }
        if !active.iter().any(|value| *value) {
            break;
        }
        for index in 0..plan.camera_count {
            if index != anchor {
                workspace.pcg_preconditioned[index] = workspace.pcg_residual[index]
                    / (workspace.camera_terms[index].diagonal + RIDGE);
            }
        }
        let new_dot = component_dot(
            &workspace.pcg_residual,
            &workspace.pcg_preconditioned,
            anchor,
        );
        for component in 0..3 {
            if !active[component] {
                for entry in &mut workspace.pcg_search {
                    entry[component] = 0.0;
                }
                continue;
            }
            if !new_dot[component].is_finite() || old_dot[component] <= 0.0 {
                return Err(GlobalPositioningError::LinearSolveFailed);
            }
            let beta = new_dot[component] / old_dot[component];
            for index in 0..plan.camera_count {
                workspace.pcg_search[index][component] = workspace.pcg_preconditioned[index]
                    [component]
                    + beta * workspace.pcg_search[index][component];
            }
        }
        old_dot = new_dot;
    }
    operator.apply(anchor, &workspace.trial_cameras, &mut workspace.pcg_applied);
    for index in 0..plan.camera_count {
        workspace.pcg_applied[index] -= workspace.pcg_rhs[index];
    }
    let final_residual = component_dot(&workspace.pcg_applied, &workspace.pcg_applied, anchor);
    for component in 0..3 {
        let allowed = (PCG_RELATIVE_TOLERANCE * right_norm_squared[component].sqrt()
            + PCG_ABSOLUTE_TOLERANCE)
            .powi(2);
        if !final_residual[component].is_finite() || final_residual[component] > allowed {
            return Err(GlobalPositioningError::LinearSolveFailed);
        }
    }
    Ok(iterations)
}

pub(super) struct ReducedOperator<'a> {
    pub(super) plan: &'a GlobalPositioningPlan,
    pub(super) bearing_weights: &'a [f64],
    pub(super) direction_weights: &'a [f64],
    pub(super) coupling_values: &'a [f64],
    pub(super) camera_terms: &'a [CameraTerm],
}

impl ReducedOperator<'_> {
    pub(super) fn apply(&self, anchor: usize, input: &[Vector3<f64>], output: &mut [Vector3<f64>]) {
        if self.plan.materialized_couplings {
            apply_materialized_reduced(
                self.plan,
                self.camera_terms,
                anchor,
                input,
                output,
                self.coupling_values,
            );
            return;
        }
        for (index, value) in output.iter_mut().enumerate() {
            *value = if index == anchor {
                Vector3::zeros()
            } else {
                input[index] * RIDGE
            };
        }
        for point in 0..self.plan.point_count {
            let indices = self.plan.point_bearing_range(point);
            let mut weight_sum = 0.0;
            let mut weighted_camera = Vector3::zeros();
            for &index in indices {
                let observation = self.plan.bearings[index];
                let weight = self.bearing_weights[index];
                weight_sum += weight;
                weighted_camera += input[observation.camera] * weight;
            }
            if weight_sum < 1e-12 {
                continue;
            }
            let mean = weighted_camera / weight_sum;
            for &index in indices {
                let camera = self.plan.bearings[index].camera;
                if camera != anchor {
                    output[camera] += (input[camera] - mean) * self.bearing_weights[index];
                }
            }
        }
        for (edge, weight) in self.plan.directions.iter().zip(self.direction_weights) {
            let difference = input[edge.target] - input[edge.source];
            if edge.source != anchor {
                output[edge.source] -= difference * *weight;
            }
            if edge.target != anchor {
                output[edge.target] += difference * *weight;
            }
        }
        output[anchor] = Vector3::zeros();
    }
}

fn apply_materialized_reduced(
    plan: &GlobalPositioningPlan,
    camera_terms: &[CameraTerm],
    anchor: usize,
    input: &[Vector3<f64>],
    output: &mut [Vector3<f64>],
    coupling_values: &[f64],
) {
    for (index, value) in output.iter_mut().enumerate() {
        *value = if index == anchor {
            Vector3::zeros()
        } else {
            input[index] * (camera_terms[index].diagonal + RIDGE)
        };
    }
    for (&(source, target), value) in plan.coupling_endpoints.iter().zip(coupling_values) {
        if source != anchor {
            output[source] += input[target] * *value;
        }
        if target != anchor {
            output[target] += input[source] * *value;
        }
    }
    output[anchor] = Vector3::zeros();
}

fn component_dot(left: &[Vector3<f64>], right: &[Vector3<f64>], anchor: usize) -> Vector3<f64> {
    let mut sum = Vector3::zeros();
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if index != anchor {
            sum.x += left.x * right.x;
            sum.y += left.y * right.y;
            sum.z += left.z * right.z;
        }
    }
    sum
}

pub(super) fn solve_direction_projector_pcg(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    anchor: usize,
) -> bool {
    let GlobalPositioningWorkspace {
        cameras: solution,
        pcg_rhs: right_hand_side,
        pcg_residual: residual,
        pcg_preconditioned: preconditioned,
        pcg_search: search,
        pcg_applied: applied,
        direction_init_diagonal: diagonal,
        direction_init_weight: weights,
        ..
    } = workspace;
    solution.fill(Vector3::zeros());
    residual.copy_from_slice(right_hand_side);
    for index in 0..solution.len() {
        preconditioned[index] = if index == anchor
            || diagonal[index]
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            Vector3::zeros()
        } else {
            residual[index].component_div(&diagonal[index])
        };
    }
    search.copy_from_slice(preconditioned);
    let mut old_dot = vector_dot(residual, preconditioned, anchor);
    let right_norm_squared = vector_dot(right_hand_side, right_hand_side, anchor);
    if !old_dot.is_finite() || old_dot <= 0.0 || !right_norm_squared.is_finite() {
        return false;
    }
    let target = (1e-8 * right_norm_squared.sqrt() + 1e-12).powi(2);
    let maximum_iterations = (12 * solution.len()).clamp(128, 4000);
    for _ in 0..maximum_iterations {
        apply_direction_projector(&plan.directions, weights, anchor, search, applied);
        let denominator = vector_dot(search, applied, anchor);
        if !denominator.is_finite() || denominator <= f64::EPSILON * old_dot.abs() {
            return false;
        }
        let alpha = old_dot / denominator;
        for index in 0..solution.len() {
            if index != anchor {
                solution[index] += search[index] * alpha;
                residual[index] -= applied[index] * alpha;
            }
        }
        if vector_dot(residual, residual, anchor) <= target {
            break;
        }
        for index in 0..solution.len() {
            preconditioned[index] = if index == anchor {
                Vector3::zeros()
            } else {
                residual[index].component_div(&diagonal[index])
            };
        }
        let new_dot = vector_dot(residual, preconditioned, anchor);
        if !new_dot.is_finite() || new_dot < 0.0 || old_dot <= 0.0 {
            return false;
        }
        let beta = new_dot / old_dot;
        for index in 0..solution.len() {
            search[index] = preconditioned[index] + search[index] * beta;
        }
        old_dot = new_dot;
    }
    apply_direction_projector(&plan.directions, weights, anchor, solution, applied);
    for (value, right) in applied.iter_mut().zip(right_hand_side) {
        *value -= *right;
    }
    let final_residual = vector_dot(applied, applied, anchor);
    final_residual.is_finite() && final_residual <= 100.0 * target
}

fn apply_direction_projector(
    directions: &[Direction],
    weights: &[f64],
    anchor: usize,
    input: &[Vector3<f64>],
    output: &mut [Vector3<f64>],
) {
    for (index, value) in output.iter_mut().enumerate() {
        *value = if index == anchor {
            Vector3::zeros()
        } else {
            input[index] * RIDGE
        };
    }
    for (edge, weight) in directions.iter().zip(weights) {
        let delta = input[edge.target] - input[edge.source];
        let projected = delta - edge.direction * edge.direction.dot(&delta);
        if edge.source != anchor {
            output[edge.source] -= projected * *weight;
        }
        if edge.target != anchor {
            output[edge.target] += projected * *weight;
        }
    }
    output[anchor] = Vector3::zeros();
}

fn vector_dot(left: &[Vector3<f64>], right: &[Vector3<f64>], anchor: usize) -> f64 {
    left.iter()
        .zip(right)
        .enumerate()
        .filter(|(index, _)| *index != anchor)
        .map(|(_, (left, right))| left.dot(right))
        .sum()
}
