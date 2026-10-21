// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded dense Cholesky solve of the reduced camera system.

use nalgebra::Vector3;

use super::RIDGE;
use super::api::GlobalPositioningError;
use super::plan::GlobalPositioningPlan;
use super::workspace::{DENSE_FALLBACK_MAX_DIMENSION, GlobalPositioningWorkspace};

pub(super) fn solve_reduced_dense(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    anchor: usize,
) -> Result<(), GlobalPositioningError> {
    factor_reduced_dense(plan, workspace, anchor, 0.0)?;
    let dimension = plan.camera_count - 1;
    let rhs = &mut workspace.dense_rhs[..dimension * 3];
    for camera in 0..plan.camera_count {
        let reduced = workspace.reduced_index[camera];
        if reduced != usize::MAX {
            let term = workspace.camera_terms[camera].right_hand_side;
            rhs[reduced * 3..reduced * 3 + 3].copy_from_slice(term.as_slice());
        }
    }
    cholesky_solve_in_place(
        &workspace.dense_matrix[..dimension * dimension],
        rhs,
        dimension,
    )?;
    workspace.trial_cameras.fill(Vector3::zeros());
    for camera in 0..plan.camera_count {
        let reduced = workspace.reduced_index[camera];
        if reduced != usize::MAX {
            workspace.trial_cameras[camera] =
                Vector3::new(rhs[reduced * 3], rhs[reduced * 3 + 1], rhs[reduced * 3 + 2]);
        }
    }
    Ok(())
}

/// Whether the dense reduced system fits the workspace for this plan.
pub(super) fn dense_fits(
    plan: &GlobalPositioningPlan,
    workspace: &GlobalPositioningWorkspace,
) -> bool {
    let dimension = plan.camera_count - 1;
    dimension <= DENSE_FALLBACK_MAX_DIMENSION
        && workspace.dense_matrix.len() >= dimension * dimension
        && workspace.dense_rhs.len() >= dimension * 3
}

/// Assemble the scalar reduced camera matrix from the current camera terms and effective
/// weights, with its diagonal scaled by `1 + diagonal_scale`, and factor it in place. The
/// factor stays in the workspace for [`solve_with_factor`].
pub(super) fn factor_reduced_dense(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    anchor: usize,
    diagonal_scale: f64,
) -> Result<(), GlobalPositioningError> {
    let dimension = plan.camera_count - 1;
    if !dense_fits(plan, workspace) {
        return Err(GlobalPositioningError::LinearSolveFailed);
    }
    let matrix = &mut workspace.dense_matrix[..dimension * dimension];
    matrix.fill(0.0);
    workspace.reduced_index.fill(usize::MAX);
    let mut row = 0usize;
    for camera in 0..plan.camera_count {
        if camera == anchor {
            continue;
        }
        workspace.reduced_index[camera] = row;
        matrix[row * dimension + row] =
            workspace.camera_terms[camera].diagonal * (1.0 + diagonal_scale) + RIDGE;
        row += 1;
    }
    if plan.materialized_couplings {
        for (&(source, target), value) in plan
            .coupling_endpoints
            .iter()
            .zip(&workspace.coupling_values)
        {
            add_dense_coupling(
                matrix,
                dimension,
                &workspace.reduced_index,
                source,
                target,
                *value,
            );
        }
    } else {
        for point in 0..plan.point_count {
            let indices = plan.point_bearing_range(point);
            let point_weight = workspace.point_weight[point];
            if point_weight < 1e-12 {
                continue;
            }
            for left in 0..indices.len() {
                let left_index = indices[left];
                for &right_index in &indices[left + 1..] {
                    add_dense_coupling(
                        matrix,
                        dimension,
                        &workspace.reduced_index,
                        plan.bearings[left_index].camera,
                        plan.bearings[right_index].camera,
                        -workspace.bearing_effective_weight[left_index]
                            * workspace.bearing_effective_weight[right_index]
                            / point_weight,
                    );
                }
            }
        }
        for (index, edge) in plan.directions.iter().enumerate() {
            add_dense_coupling(
                matrix,
                dimension,
                &workspace.reduced_index,
                edge.source,
                edge.target,
                -workspace.direction_effective_weight[index],
            );
        }
    }
    cholesky_factor_in_place(matrix, dimension)
}

/// Solve the factored scalar system for the three coordinates of `input` (the anchor's entry is
/// ignored and its output is zero).
pub(super) fn solve_with_factor(
    plan: &GlobalPositioningPlan,
    workspace: &mut GlobalPositioningWorkspace,
    input: &[Vector3<f64>],
    output: &mut [Vector3<f64>],
) -> Result<(), GlobalPositioningError> {
    let dimension = plan.camera_count - 1;
    let rhs = &mut workspace.dense_rhs[..dimension * 3];
    for (&reduced, vector) in workspace.reduced_index.iter().zip(input) {
        if reduced != usize::MAX {
            rhs[reduced * 3..reduced * 3 + 3].copy_from_slice(vector.as_slice());
        }
    }
    cholesky_solve_in_place(
        &workspace.dense_matrix[..dimension * dimension],
        rhs,
        dimension,
    )?;
    for (&reduced, vector) in workspace.reduced_index.iter().zip(output.iter_mut()) {
        *vector = if reduced == usize::MAX {
            Vector3::zeros()
        } else {
            Vector3::new(rhs[reduced * 3], rhs[reduced * 3 + 1], rhs[reduced * 3 + 2])
        };
    }
    Ok(())
}

fn add_dense_coupling(
    matrix: &mut [f64],
    dimension: usize,
    reduced_index: &[usize],
    source: usize,
    target: usize,
    value: f64,
) {
    let source_row = reduced_index[source];
    let target_row = reduced_index[target];
    if source_row == usize::MAX || target_row == usize::MAX {
        return;
    }
    let (lower, upper) = if source_row >= target_row {
        (source_row, target_row)
    } else {
        (target_row, source_row)
    };
    matrix[lower * dimension + upper] += value;
}

fn cholesky_factor_in_place(
    matrix: &mut [f64],
    dimension: usize,
) -> Result<(), GlobalPositioningError> {
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row * dimension + column];
            for inner in 0..column {
                value -= matrix[row * dimension + inner] * matrix[column * dimension + inner];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(GlobalPositioningError::LinearSolveFailed);
                }
                matrix[row * dimension + column] = value.sqrt();
            } else {
                let diagonal = matrix[column * dimension + column];
                let factor = value / diagonal;
                if !factor.is_finite() {
                    return Err(GlobalPositioningError::LinearSolveFailed);
                }
                matrix[row * dimension + column] = factor;
            }
        }
    }
    Ok(())
}

fn cholesky_solve_in_place(
    factor: &[f64],
    right_hand_side: &mut [f64],
    dimension: usize,
) -> Result<(), GlobalPositioningError> {
    for row in 0..dimension {
        for component in 0..3 {
            let mut value = right_hand_side[row * 3 + component];
            for column in 0..row {
                value -= factor[row * dimension + column] * right_hand_side[column * 3 + component];
            }
            right_hand_side[row * 3 + component] = value / factor[row * dimension + row];
        }
    }
    for row in (0..dimension).rev() {
        for component in 0..3 {
            let mut value = right_hand_side[row * 3 + component];
            for lower_row in row + 1..dimension {
                value -= factor[lower_row * dimension + row]
                    * right_hand_side[lower_row * 3 + component];
            }
            let solution = value / factor[row * dimension + row];
            if !solution.is_finite() {
                return Err(GlobalPositioningError::LinearSolveFailed);
            }
            right_hand_side[row * 3 + component] = solution;
        }
    }
    Ok(())
}
