// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::point_block::{POINT_BLOCK_RCOND_THRESHOLD, PointBlockFactor};
use super::residual::{MeasurementModel, transform};
use super::{BundleAdjustmentDiagnostics, RelativePose};
use crate::kernel::projection::ReprojectionLinearization;
use crate::numerics::manifold::Mat3;
use crate::numerics::refine::{Loss, NoiseModel};
use nalgebra::SMatrix;

pub(super) const POSE_DIMENSION: usize = 5;
const POINT_DIMENSION: usize = 3;

#[derive(Clone, Copy)]
pub(super) struct LocalLinearization {
    pub pose_hessian: [[f64; POSE_DIMENSION]; POSE_DIMENSION],
    pub cross_hessian: [[f64; POINT_DIMENSION]; POSE_DIMENSION],
    pub point_hessian: [[f64; POINT_DIMENSION]; POINT_DIMENSION],
    pose_gradient: [f64; POSE_DIMENSION],
    point_gradient: [f64; POINT_DIMENSION],
    cost: f64,
    squared_residual: f64,
}

impl LocalLinearization {
    fn new() -> Self {
        Self {
            pose_hessian: [[0.0; POSE_DIMENSION]; POSE_DIMENSION],
            cross_hessian: [[0.0; POINT_DIMENSION]; POSE_DIMENSION],
            point_hessian: [[0.0; POINT_DIMENSION]; POINT_DIMENSION],
            pose_gradient: [0.0; POSE_DIMENSION],
            point_gradient: [0.0; POINT_DIMENSION],
            cost: 0.0,
            squared_residual: 0.0,
        }
    }

    fn accumulate(
        &mut self,
        factor: &ReprojectionLinearization<POSE_DIMENSION, POINT_DIMENSION>,
        sqrt_weight: f64,
    ) {
        for residual in 0..2 {
            let weighted_residual = sqrt_weight * factor.residual[residual];
            let weighted_pose: [f64; POSE_DIMENSION] =
                factor.camera_jacobian[residual].map(|value| sqrt_weight * value);
            let weighted_point: [f64; POINT_DIMENSION] =
                factor.point_jacobian[residual].map(|value| sqrt_weight * value);
            for row in 0..POSE_DIMENSION {
                self.pose_gradient[row] += weighted_pose[row] * weighted_residual;
                for column in 0..POSE_DIMENSION {
                    self.pose_hessian[row][column] += weighted_pose[row] * weighted_pose[column];
                }
                for (column, &point_value) in weighted_point.iter().enumerate() {
                    self.cross_hessian[row][column] += weighted_pose[row] * point_value;
                }
            }
            for row in 0..POINT_DIMENSION {
                self.point_gradient[row] += weighted_point[row] * weighted_residual;
                for column in 0..POINT_DIMENSION {
                    self.point_hessian[row][column] += weighted_point[row] * weighted_point[column];
                }
            }
        }
    }
}

pub(super) struct ReducedSystem {
    pub hessian: [[f64; POSE_DIMENSION]; POSE_DIMENSION],
    pub gradient: [f64; POSE_DIMENSION],
    pub cost: f64,
    pub squared_residual: f64,
    full_pose_gradient: [f64; POSE_DIMENSION],
    pose_diagonal: [f64; POSE_DIMENSION],
    point_model_reduction: f64,
    scaled_point_gradient_inf_norm: f64,
    weak_point_count: usize,
    minimum_point_rcond: f64,
    valid: bool,
}

impl ReducedSystem {
    pub fn new() -> Self {
        Self {
            hessian: [[0.0; POSE_DIMENSION]; POSE_DIMENSION],
            gradient: [0.0; POSE_DIMENSION],
            cost: 0.0,
            squared_residual: 0.0,
            full_pose_gradient: [0.0; POSE_DIMENSION],
            pose_diagonal: [0.0; POSE_DIMENSION],
            point_model_reduction: 0.0,
            scaled_point_gradient_inf_norm: 0.0,
            weak_point_count: 0,
            minimum_point_rcond: 1.0,
            valid: true,
        }
    }

    pub fn add(&mut self, local: &LocalLinearization, damping: f64) {
        self.cost += local.cost;
        self.squared_residual += local.squared_residual;
        for row in 0..POSE_DIMENSION {
            self.full_pose_gradient[row] += local.pose_gradient[row];
            self.pose_diagonal[row] += local.pose_hessian[row][row];
        }
        self.scaled_point_gradient_inf_norm =
            self.scaled_point_gradient_inf_norm
                .max(scaled_gradient_inf_norm(
                    &local.point_gradient,
                    &std::array::from_fn(|index| local.point_hessian[index][index]),
                ));
        let Some(factor) = PointBlockFactor::new(&local.point_hessian, damping) else {
            self.valid = false;
            if damping == 0.0 {
                self.weak_point_count += 1;
                self.minimum_point_rcond = 0.0;
            }
            return;
        };
        if damping == 0.0 {
            let rcond = reciprocal_condition_number_3(&local.point_hessian, &factor);
            self.minimum_point_rcond = self.minimum_point_rcond.min(rcond);
            if rcond <= POINT_BLOCK_RCOND_THRESHOLD {
                self.weak_point_count += 1;
            }
        }
        let inverse_gradient = factor.solve(&local.point_gradient);
        self.point_model_reduction += 0.5 * dot3(&local.point_gradient, &inverse_gradient);
        let inverse_cross: [[f64; POINT_DIMENSION]; POSE_DIMENSION] =
            std::array::from_fn(|pose| factor.solve(&local.cross_hessian[pose]));
        for row in 0..POSE_DIMENSION {
            self.gradient[row] +=
                local.pose_gradient[row] - dot3(&local.cross_hessian[row], &inverse_gradient);
            for (column, inverse_cross_column) in inverse_cross.iter().enumerate() {
                self.hessian[row][column] += local.pose_hessian[row][column]
                    - dot3(&local.cross_hessian[row], inverse_cross_column);
            }
        }
    }

    pub fn finish_damping(&mut self, damping: f64) {
        let scale = self.pose_diagonal.iter().copied().fold(0.0, f64::max);
        let floor = 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
        for parameter in 0..POSE_DIMENSION {
            self.hessian[parameter][parameter] +=
                damping * self.pose_diagonal[parameter].max(floor);
        }
    }

    pub fn is_valid(&self) -> bool {
        self.valid
            && self.gradient.iter().all(|value| value.is_finite())
            && self.hessian.iter().flatten().all(|value| value.is_finite())
    }

    pub fn scaled_gradient_inf_norm(&self) -> f64 {
        self.scaled_point_gradient_inf_norm
            .max(scaled_gradient_inf_norm(
                &self.full_pose_gradient,
                &self.pose_diagonal,
            ))
    }

    pub fn predicted_reduction(
        &self,
        step: &[f64; POSE_DIMENSION],
        damping: f64,
        point_damping_norm: f64,
    ) -> f64 {
        let quadratic = (0..POSE_DIMENSION)
            .map(|row| {
                step[row]
                    * self.hessian[row]
                        .iter()
                        .zip(step)
                        .map(|(value, step)| value * step)
                        .sum::<f64>()
            })
            .sum::<f64>();
        let damped_reduction = self.point_model_reduction
            - self
                .gradient
                .iter()
                .zip(step)
                .map(|(gradient, step)| gradient * step)
                .sum::<f64>()
            - 0.5 * quadratic;
        let scale = self.pose_diagonal.iter().copied().fold(0.0, f64::max);
        let floor = 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
        let pose_damping_norm = self
            .pose_diagonal
            .iter()
            .zip(step)
            .map(|(curvature, step)| curvature.max(floor) * step * step)
            .sum::<f64>();
        damped_reduction + 0.5 * damping * (pose_damping_norm + point_damping_norm)
    }

    pub fn diagnostics(&self) -> BundleAdjustmentDiagnostics {
        let (camera_rank, camera_rcond) = if self.is_valid() {
            camera_hessian_diagnostics(&self.hessian)
        } else {
            (0, 0.0)
        };
        BundleAdjustmentDiagnostics {
            pose_dimension: POSE_DIMENSION,
            pose_rank: camera_rank,
            pose_hessian_reciprocal_condition_number: camera_rcond,
            weak_point_count: self.weak_point_count,
            minimum_point_hessian_reciprocal_condition_number: self.minimum_point_rcond,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn linearize_point<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    translation_basis: &[[f64; 3]; 2],
    baseline: f64,
    point: &[f64; 3],
    index: usize,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Option<LocalLinearization> {
    let mut local = LocalLinearization::new();
    let inverse_standard_deviation = noise.inverse_standard_deviation(index);

    let first = model.project(0, index, point)?;
    let first_norm = first
        .residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let first_weight =
        inverse_standard_deviation * loss.sqrt_weight(first_norm * inverse_standard_deviation);
    local.cost += loss.cost(first_norm * inverse_standard_deviation);
    local.squared_residual += first_norm * first_norm;
    let first_factor = ReprojectionLinearization::new(
        first,
        [[0.0; POSE_DIMENSION]; 2],
        scale_2x3(first.jacobian, baseline),
    );
    local.accumulate(&first_factor, first_weight);

    let rotated = rotate(&pose.rotation, point);
    let camera_point = [
        rotated[0] + pose.translation[0],
        rotated[1] + pose.translation[1],
        rotated[2] + pose.translation[2],
    ];
    let second = model.project(1, index, &camera_point)?;
    let second_norm = second
        .residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let second_weight =
        inverse_standard_deviation * loss.sqrt_weight(second_norm * inverse_standard_deviation);
    local.cost += loss.cost(second_norm * inverse_standard_deviation);
    local.squared_residual += second_norm * second_norm;
    let point_jacobian = scale_2x3(multiply_2x3_3x3(&second.jacobian, &pose.rotation), baseline);
    let pose_motion = pose_motion_jacobian(&rotated, translation_basis, baseline);
    let pose_jacobian = multiply_2x3_3x5(&second.jacobian, &pose_motion);
    let second_factor = ReprojectionLinearization::new(second, pose_jacobian, point_jacobian);
    local.accumulate(&second_factor, second_weight);
    Some(local)
}

pub(super) fn point_step(
    local: &LocalLinearization,
    damping: f64,
    pose_step: &[f64; POSE_DIMENSION],
) -> Option<[f64; POINT_DIMENSION]> {
    point_step_with_damping_norm(local, damping, pose_step).map(|(step, _)| step)
}

pub(super) fn point_step_with_damping_norm(
    local: &LocalLinearization,
    damping: f64,
    pose_step: &[f64; POSE_DIMENSION],
) -> Option<([f64; POINT_DIMENSION], f64)> {
    let factor = PointBlockFactor::new(&local.point_hessian, damping)?;
    let right_hand_side: [f64; POINT_DIMENSION] = std::array::from_fn(|coordinate| {
        local.point_gradient[coordinate]
            + (0..POSE_DIMENSION)
                .map(|pose| local.cross_hessian[pose][coordinate] * pose_step[pose])
                .sum::<f64>()
    });
    let step = factor.solve(&right_hand_side).map(|value| -value);
    let diagonal = PointBlockFactor::damping_diagonal(&local.point_hessian)?;
    let damping_norm = diagonal
        .iter()
        .zip(&step)
        .map(|(diagonal, step)| diagonal * step * step)
        .sum();
    Some((step, damping_norm))
}

pub(super) fn metrics<M: MeasurementModel>(
    model: M,
    pose: &RelativePose,
    points: &[[f64; 3]],
    loss: Loss,
    noise: NoiseModel<'_>,
) -> (f64, f64) {
    let mut total = 0.0;
    let mut squared = 0.0;
    for (index, point) in points.iter().enumerate() {
        let transformed = transform(&pose.rotation, &pose.translation, point);
        for (view, camera_point) in [(0, point), (1, &transformed)] {
            let Some(projection) = model.project(view, index, camera_point) else {
                return (f64::INFINITY, f64::INFINITY);
            };
            let norm = projection
                .residual
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            total += loss.cost(norm * noise.inverse_standard_deviation(index));
            squared += norm * norm;
        }
    }
    (total, (squared / (4 * points.len()) as f64).sqrt())
}

fn camera_hessian_diagnostics(hessian: &[[f64; POSE_DIMENSION]; POSE_DIMENSION]) -> (usize, f64) {
    let matrix = SMatrix::<f64, POSE_DIMENSION, POSE_DIMENSION>::from_fn(|row, column| {
        0.5 * (hessian[row][column] + hessian[column][row])
    });
    spectral_diagnostics(matrix.symmetric_eigen().eigenvalues.as_slice())
}

fn reciprocal_condition_number_3(matrix: &Mat3, factor: &PointBlockFactor) -> f64 {
    factor.reciprocal_condition_number(matrix)
}

fn spectral_diagnostics(eigenvalues: &[f64]) -> (usize, f64) {
    let largest = eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return (0, 0.0);
    }
    let tolerance = 64.0 * f64::EPSILON * eigenvalues.len() as f64 * largest;
    let rank = eigenvalues
        .iter()
        .filter(|&&value| value.is_finite() && value > tolerance)
        .count();
    let reciprocal_condition_number = if rank == eigenvalues.len() {
        eigenvalues.iter().copied().fold(f64::INFINITY, f64::min) / largest
    } else {
        0.0
    };
    (rank, reciprocal_condition_number)
}

fn pose_motion_jacobian(
    rotated: &[f64; 3],
    translation_basis: &[[f64; 3]; 2],
    baseline: f64,
) -> [[f64; POSE_DIMENSION]; 3] {
    [
        [
            0.0,
            rotated[2],
            -rotated[1],
            baseline * translation_basis[0][0],
            baseline * translation_basis[1][0],
        ],
        [
            -rotated[2],
            0.0,
            rotated[0],
            baseline * translation_basis[0][1],
            baseline * translation_basis[1][1],
        ],
        [
            rotated[1],
            -rotated[0],
            0.0,
            baseline * translation_basis[0][2],
            baseline * translation_basis[1][2],
        ],
    ]
}

fn multiply_2x3_3x3(left: &[[f64; 3]; 2], right: &Mat3) -> [[f64; 3]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
}

fn scale_2x3(mut matrix: [[f64; 3]; 2], scale: f64) -> [[f64; 3]; 2] {
    for row in &mut matrix {
        for value in row {
            *value *= scale;
        }
    }
    matrix
}

fn scaled_gradient_inf_norm<const N: usize>(gradient: &[f64; N], diagonal: &[f64; N]) -> f64 {
    let scale = diagonal.iter().copied().fold(0.0, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return if gradient.iter().all(|value| *value == 0.0) {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let floor = 64.0 * f64::EPSILON * scale;
    gradient
        .iter()
        .zip(diagonal)
        .map(|(&value, &curvature)| value.abs() / curvature.max(floor).sqrt())
        .fold(0.0, f64::max)
}

fn multiply_2x3_3x5(
    left: &[[f64; 3]; 2],
    right: &[[f64; POSE_DIMENSION]; 3],
) -> [[f64; POSE_DIMENSION]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
}

fn rotate(rotation: &Mat3, point: &[f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2],
    ]
}

fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
