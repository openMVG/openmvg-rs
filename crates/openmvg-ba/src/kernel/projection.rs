// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Projection residuals and analytic point Jacobians shared by BA kernels.

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
use super::camera::PinholeCamera;

/// Reprojection residual and derivative with respect to a camera-frame point.
#[derive(Clone, Copy)]
pub struct Projection {
    pub residual: [f64; 2],
    pub jacobian: [[f64; 3]; 2],
}

/// Ephemeral fixed-size output of one reprojection-factor linearization.
///
/// The packet is consumed immediately by an accumulator, so it owns no
/// topology, measurement, or heap storage.
#[cfg(any(feature = "faer", feature = "unstable-internals"))]
#[derive(Clone, Copy)]
pub struct ReprojectionLinearization<const CAMERA_DIMENSION: usize, const POINT_DIMENSION: usize> {
    pub residual: [f64; 2],
    pub camera_jacobian: [[f64; CAMERA_DIMENSION]; 2],
    pub point_jacobian: [[f64; POINT_DIMENSION]; 2],
}

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
impl<const CAMERA_DIMENSION: usize, const POINT_DIMENSION: usize>
    ReprojectionLinearization<CAMERA_DIMENSION, POINT_DIMENSION>
{
    #[inline(always)]
    pub const fn new(
        projection: Projection,
        camera_jacobian: [[f64; CAMERA_DIMENSION]; 2],
        point_jacobian: [[f64; POINT_DIMENSION]; 2],
    ) -> Self {
        Self {
            residual: projection.residual,
            camera_jacobian,
            point_jacobian,
        }
    }
}

#[cfg(any(feature = "faer", feature = "unstable-internals"))]
pub fn pixel_projection(
    camera: PinholeCamera,
    observation: &[f64; 2],
    point: &[f64; 3],
) -> Option<Projection> {
    let scale = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if point.iter().any(|value| !value.is_finite()) || point[2] <= 64.0 * f64::EPSILON * scale {
        return None;
    }
    let inverse_depth = point[2].recip();
    let inverse_depth_squared = inverse_depth * inverse_depth;
    let x_numerator = camera.fx * point[0] + camera.skew * point[1];
    let prediction = [
        x_numerator * inverse_depth + camera.cx,
        camera.fy * point[1] * inverse_depth + camera.cy,
    ];
    let projection = Projection {
        residual: [
            prediction[0] - observation[0],
            prediction[1] - observation[1],
        ],
        jacobian: [
            [
                camera.fx * inverse_depth,
                camera.skew * inverse_depth,
                -x_numerator * inverse_depth_squared,
            ],
            [
                0.0,
                camera.fy * inverse_depth,
                -camera.fy * point[1] * inverse_depth_squared,
            ],
        ],
    };
    projection
        .residual
        .iter()
        .chain(projection.jacobian.as_flattened())
        .all(|value| value.is_finite())
        .then_some(projection)
}

pub fn angular_projection(raw_observation: &[f64; 3], point: &[f64; 3]) -> Option<Projection> {
    let observation = normalize(raw_observation)?;
    let prediction = normalize(point)?;
    let basis = tangent_basis(&observation);
    let a = dot(&basis[0], &prediction);
    let b = dot(&basis[1], &prediction);
    let c = dot(&observation, &prediction);
    let sine_squared = a * a + b * b;
    let sine = sine_squared.sqrt();
    // Near the antipode the angle is close to pi while the small-angle series
    // below assumes it is close to zero, and the exact factor `angle / sine`
    // diverges; treat such a point as unprojectable.
    if c < 0.0 && sine < 1e-6 {
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
pub fn normalize(vector: &[f64; 3]) -> Option<[f64; 3]> {
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
pub fn tangent_basis(unit: &[f64; 3]) -> [[f64; 3]; 2] {
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
