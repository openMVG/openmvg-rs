// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use nalgebra::Vector3;
use openmvg_core::camera::{CameraModel, PinholeCamera};

#[derive(Clone, Copy)]
pub(super) struct Projection {
    pub(super) residual: [f64; 2],
    pub(super) jacobian: [[f64; 3]; 2],
}

#[derive(Clone, Copy)]
pub(super) struct ReprojectionLinearization<const CAMERA_DIMENSION: usize> {
    pub(super) residual: [f64; 2],
    pub(super) camera_jacobian: [[f64; CAMERA_DIMENSION]; 2],
}

impl<const CAMERA_DIMENSION: usize> ReprojectionLinearization<CAMERA_DIMENSION> {
    #[inline(always)]
    pub(super) const fn new(
        projection: Projection,
        camera_jacobian: [[f64; CAMERA_DIMENSION]; 2],
    ) -> Self {
        Self {
            residual: projection.residual,
            camera_jacobian,
        }
    }
}

pub(super) fn pixel_projection(
    camera: &PinholeCamera,
    observation: &[f64; 2],
    point: &[f64; 3],
) -> Option<Projection> {
    let scale = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if point.iter().any(|value| !value.is_finite()) || point[2] <= 64.0 * f64::EPSILON * scale {
        return None;
    }
    let point = Vector3::from_row_slice(point);
    let (prediction, jacobian) = camera.project_with_jacobian(&point).ok()?;
    let projection = Projection {
        residual: [prediction.x - observation[0], prediction.y - observation[1]],
        jacobian: [
            [jacobian[(0, 0)], jacobian[(0, 1)], jacobian[(0, 2)]],
            [jacobian[(1, 0)], jacobian[(1, 1)], jacobian[(1, 2)]],
        ],
    };
    projection
        .residual
        .iter()
        .chain(projection.jacobian.iter().flatten())
        .all(|value| value.is_finite())
        .then_some(projection)
}

pub(super) fn angular_projection(
    raw_observation: &[f64; 3],
    point: &[f64; 3],
) -> Option<Projection> {
    let observation = normalize(raw_observation)?;
    let prediction = normalize(point)?;
    let basis = tangent_basis(&observation);
    let a = dot(&basis[0], &prediction);
    let b = dot(&basis[1], &prediction);
    let c = dot(&observation, &prediction);
    let sine_squared = a * a + b * b;
    let sine = sine_squared.sqrt();
    if c < 0.0 && sine <= 64.0 * f64::EPSILON {
        return None;
    }

    let mut derivative_prediction = [[0.0; 3]; 2];
    let (factor, factor_derivative_scale) = if sine < 1e-6 {
        let sine_fourth = sine_squared * sine_squared;
        (
            1.0 + sine_squared / 6.0 + 3.0 * sine_fourth / 40.0,
            1.0 / 3.0 + 3.0 * sine_squared / 10.0,
        )
    } else {
        let angle = sine.atan2(c);
        (angle / sine, f64::NAN)
    };

    for (coordinate, &dc) in observation.iter().enumerate() {
        let da = basis[0][coordinate];
        let db = basis[1][coordinate];
        let factor_derivative = if sine < 1e-6 {
            factor_derivative_scale * (a * da + b * db)
        } else {
            let ds = (a * da + b * db) / sine;
            let angle = sine.atan2(c);
            let angle_derivative = (c * ds - sine * dc) / (sine_squared + c * c);
            (angle_derivative * sine - angle * ds) / sine_squared
        };
        derivative_prediction[0][coordinate] = factor * da + a * factor_derivative;
        derivative_prediction[1][coordinate] = factor * db + b * factor_derivative;
    }

    let point_norm = dot(point, point).sqrt();
    let mut jacobian = [[0.0; 3]; 2];
    for row in 0..2 {
        for column in 0..3 {
            for coordinate in 0..3 {
                let normalization = ((column == coordinate) as u8 as f64
                    - prediction[column] * prediction[coordinate])
                    / point_norm;
                jacobian[row][column] += derivative_prediction[row][coordinate] * normalization;
            }
        }
    }

    Some(Projection {
        residual: [factor * a, factor * b],
        jacobian,
    })
}

#[inline(always)]
fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[inline]
pub(super) fn normalize(vector: &[f64; 3]) -> Option<[f64; 3]> {
    let norm_squared = dot(vector, vector);
    if vector.iter().any(|value| !value.is_finite())
        || !(norm_squared.is_finite() && norm_squared > f64::MIN_POSITIVE)
    {
        return None;
    }
    if (norm_squared - 1.0).abs() <= 8.0 * f64::EPSILON {
        return Some(*vector);
    }
    let inverse_norm = norm_squared.sqrt().recip();
    Some(vector.map(|value| value * inverse_norm))
}

#[inline]
pub(super) fn tangent_basis(unit: &[f64; 3]) -> [[f64; 3]; 2] {
    let reference = if unit[0].abs() <= unit[1].abs() && unit[0].abs() <= unit[2].abs() {
        [1.0, 0.0, 0.0]
    } else if unit[1].abs() <= unit[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let mut first = cross(unit, &reference);
    let inverse_norm = dot(&first, &first).sqrt().recip();
    for value in &mut first {
        *value *= inverse_norm;
    }
    [first, cross(unit, &first)]
}

#[inline(always)]
fn cross(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
