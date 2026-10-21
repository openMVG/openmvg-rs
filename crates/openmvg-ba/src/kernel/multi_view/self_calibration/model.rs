// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Centered radial projection and analytic local Jacobians.
//!
//! Physical forward and inverse measurements use the principal increasing radial
//! branch, ending at the first positive zero of `1 + 3*k1*r² + 5*k2*r⁴`.
//! The explicit signed-depth mode retains its algebraic BAL projection domain.

use super::RadialCamera;
use crate::kernel::multi_view::camera_kernel::CameraKernel;
use crate::kernel::projection::Projection;

const INVERSE_RELATIVE_TOLERANCE: f64 = 1.0e-14;

#[derive(Clone, Copy)]
struct RadialTerms {
    normalized: [f64; 2],
    radius_squared: f64,
    radius_fourth: f64,
    distortion: f64,
    prediction: [f64; 2],
    inverse_depth: f64,
}

fn radial_terms_with_depth(
    camera: RadialCamera,
    point: &[f64; 3],
    signed_depth: bool,
) -> Option<RadialTerms> {
    let scale = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let depth_floor = 64.0 * f64::EPSILON * scale;
    let valid_depth = if signed_depth {
        point[2].abs() > depth_floor
    } else {
        point[2] > depth_floor
    };
    if point.iter().any(|value| !value.is_finite()) || !valid_depth {
        return None;
    }
    let inverse_depth = point[2].recip();
    let normalized = [point[0] * inverse_depth, point[1] * inverse_depth];
    let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
    if !signed_depth
        && first_positive_radial_turn(camera).is_some_and(|bound| radius_squared >= bound)
    {
        return None;
    }
    let radius_fourth = radius_squared * radius_squared;
    let distortion = 1.0 + camera.k1 * radius_squared + camera.k2 * radius_fourth;
    if !signed_depth {
        let radial_derivative = camera.k1 + 2.0 * (camera.k2 * radius_squared);
        let radial_slope = distortion + 2.0 * radius_squared * radial_derivative;
        let determinant = distortion * radial_slope;
        if !determinant.is_finite() || determinant <= 1e-12 {
            return None;
        }
    }
    let prediction = [
        camera.focal * distortion * normalized[0],
        camera.focal * distortion * normalized[1],
    ];
    prediction
        .iter()
        .chain([radius_squared, radius_fourth, distortion].iter())
        .all(|value| value.is_finite())
        .then_some(RadialTerms {
            normalized,
            radius_squared,
            radius_fourth,
            distortion,
            prediction,
            inverse_depth,
        })
}

fn radial_terms(camera: RadialCamera, point: &[f64; 3]) -> Option<RadialTerms> {
    radial_terms_with_depth(camera, point, false)
}

pub(in crate::kernel::multi_view) fn radial_projection(
    camera: RadialCamera,
    observation: &[f64; 2],
    point: &[f64; 3],
) -> Option<Projection> {
    radial_projection_from_terms(camera, observation, radial_terms(camera, point)?)
}

fn radial_projection_from_terms(
    camera: RadialCamera,
    observation: &[f64; 2],
    terms: RadialTerms,
) -> Option<Projection> {
    let radial_derivative = camera.k1 + 2.0 * (camera.k2 * terms.radius_squared);
    let distortion_x = 2.0 * terms.normalized[0] * radial_derivative;
    let distortion_y = 2.0 * terms.normalized[1] * radial_derivative;
    let normalized_jacobian = [
        [
            camera.focal * (terms.distortion + terms.normalized[0] * distortion_x),
            camera.focal * terms.normalized[0] * distortion_y,
        ],
        [
            camera.focal * terms.normalized[1] * distortion_x,
            camera.focal * (terms.distortion + terms.normalized[1] * distortion_y),
        ],
    ];
    let jacobian = std::array::from_fn(|row| {
        [
            normalized_jacobian[row][0] * terms.inverse_depth,
            normalized_jacobian[row][1] * terms.inverse_depth,
            -(normalized_jacobian[row][0] * terms.normalized[0]
                + normalized_jacobian[row][1] * terms.normalized[1])
                * terms.inverse_depth,
        ]
    });
    let projection = Projection {
        residual: [
            terms.prediction[0] - observation[0],
            terms.prediction[1] - observation[1],
        ],
        jacobian,
    };
    projection
        .residual
        .iter()
        .chain(projection.jacobian.as_flattened())
        .all(|value| value.is_finite())
        .then_some(projection)
}

pub(in crate::kernel::multi_view) fn radial_projection_signed_depth(
    camera: RadialCamera,
    observation: &[f64; 2],
    point: &[f64; 3],
) -> Option<Projection> {
    radial_projection_from_terms(
        camera,
        observation,
        radial_terms_with_depth(camera, point, true)?,
    )
}

pub(in crate::kernel::multi_view) fn radial_linearization(
    camera: RadialCamera,
    observation: &[f64; 2],
    point: &[f64; 3],
) -> Option<(Projection, [[f64; 3]; 2])> {
    radial_linearization_from_terms(camera, observation, radial_terms(camera, point)?)
}

pub(in crate::kernel::multi_view) fn radial_linearization_signed_depth(
    camera: RadialCamera,
    observation: &[f64; 2],
    point: &[f64; 3],
) -> Option<(Projection, [[f64; 3]; 2])> {
    radial_linearization_from_terms(
        camera,
        observation,
        radial_terms_with_depth(camera, point, true)?,
    )
}

fn radial_linearization_from_terms(
    camera: RadialCamera,
    observation: &[f64; 2],
    terms: RadialTerms,
) -> Option<(Projection, [[f64; 3]; 2])> {
    Some((
        radial_projection_from_terms(camera, observation, terms)?,
        radial_intrinsic_jacobian_from_terms(camera, terms)?,
    ))
}

/// Invert a finite centered radial measurement into a unit camera bearing.
///
/// The radial model preserves azimuth, so only the scalar distorted radius is
/// inverted. A reconstruction check rejects non-convergent or locally
/// singular inversions instead of returning an ambiguous ray.
pub(in crate::kernel::multi_view) fn radial_bearing(
    camera: RadialCamera,
    measurement: &[f64; 2],
) -> Option<[f64; 3]> {
    if !camera.is_valid() || measurement.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let distorted = [measurement[0] / camera.focal, measurement[1] / camera.focal];
    let distorted_radius = distorted[0].hypot(distorted[1]);
    if !distorted_radius.is_finite() {
        return None;
    }
    if distorted_radius == 0.0 {
        return Some([0.0, 0.0, 1.0]);
    }

    let turning_radius = first_positive_radial_turn(camera).map(f64::sqrt);
    let mut upper = turning_radius.unwrap_or_else(|| distorted_radius.max(1.0));
    if turning_radius.is_some() {
        let maximum = radial_radius(camera, upper);
        let tolerance = 64.0 * f64::EPSILON * (1.0 + maximum.abs());
        if !maximum.is_finite() || distorted_radius >= maximum - tolerance {
            return None;
        }
    } else {
        loop {
            let projected = radial_radius(camera, upper);
            if !projected.is_finite() {
                return None;
            }
            if projected >= distorted_radius {
                break;
            }
            upper *= 2.0;
            if !upper.is_finite() {
                return None;
            }
        }
    }

    let tolerance = INVERSE_RELATIVE_TOLERANCE * (1.0 + distorted_radius);
    let mut lower = 0.0;
    let mut radius = distorted_radius.min(0.5 * upper);
    for _ in 0..48 {
        let value = radial_radius(camera, radius) - distorted_radius;
        if value.abs() <= tolerance {
            return bearing_from_radius(distorted, distorted_radius, radius);
        }
        if value < 0.0 {
            lower = radius;
        } else {
            upper = radius;
        }
        let radius_squared = radius * radius;
        let derivative = 1.0
            + 3.0 * camera.k1 * radius_squared
            + 5.0 * camera.k2 * radius_squared * radius_squared;
        let newton = radius - value / derivative;
        radius = if derivative.is_finite() && derivative > 0.0 && newton > lower && newton < upper {
            newton
        } else {
            0.5 * (lower + upper)
        };
    }
    None
}

fn bearing_from_radius(
    distorted: [f64; 2],
    distorted_radius: f64,
    radius: f64,
) -> Option<[f64; 3]> {
    let scale = radius / distorted_radius;
    let bearing = [distorted[0] * scale, distorted[1] * scale, 1.0];
    let norm = bearing
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    (norm.is_finite() && norm > f64::MIN_POSITIVE).then(|| bearing.map(|value| value / norm))
}

fn radial_radius(camera: RadialCamera, radius: f64) -> f64 {
    let radius_squared = radius * radius;
    radius * (1.0 + camera.k1 * radius_squared + camera.k2 * radius_squared * radius_squared)
}

fn first_positive_radial_turn(camera: RadialCamera) -> Option<f64> {
    let scale = camera.k1.abs().max(camera.k2.abs()).max(1.0);
    let (a, b, c) = (
        5.0 * (camera.k2 / scale),
        3.0 * (camera.k1 / scale),
        scale.recip(),
    );
    if a == 0.0 {
        let root = -c / b;
        return (b < 0.0 && root.is_finite() && root > 0.0).then_some(root);
    }
    let discriminant = b.mul_add(b, -4.0 * a * c);
    if discriminant < 0.0 || !discriminant.is_finite() {
        return None;
    }
    let square_root = discriminant.sqrt();
    let q = -0.5 * (b + square_root.copysign(b));
    let roots = if q == 0.0 {
        [-b / (2.0 * a), f64::NAN]
    } else {
        [q / a, c / q]
    };
    roots
        .into_iter()
        .filter(|value| value.is_finite() && *value > 0.0)
        .min_by(f64::total_cmp)
}

pub(in crate::kernel::multi_view) fn radial_intrinsic_jacobian(
    camera: RadialCamera,
    point: &[f64; 3],
) -> Option<[[f64; 3]; 2]> {
    radial_intrinsic_jacobian_from_terms(camera, radial_terms(camera, point)?)
}

pub(in crate::kernel::multi_view) fn radial_intrinsic_jacobian_signed_depth(
    camera: RadialCamera,
    point: &[f64; 3],
) -> Option<[[f64; 3]; 2]> {
    radial_intrinsic_jacobian_from_terms(camera, radial_terms_with_depth(camera, point, true)?)
}

fn radial_intrinsic_jacobian_from_terms(
    camera: RadialCamera,
    terms: RadialTerms,
) -> Option<[[f64; 3]; 2]> {
    let jacobian: [[f64; 3]; 2] = std::array::from_fn(|row| {
        [
            terms.prediction[row],
            camera.focal * terms.radius_squared * terms.normalized[row],
            camera.focal * terms.radius_fourth * terms.normalized[row],
        ]
    });
    jacobian
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(jacobian)
}

impl CameraKernel<3> for RadialCamera {
    fn is_valid(self) -> bool {
        RadialCamera::is_valid(self)
    }

    fn bearing(self, measurement: &[f64; 2]) -> Option<[f64; 3]> {
        radial_bearing(self, measurement)
    }

    fn project(self, observation: &[f64; 2], camera_point: &[f64; 3]) -> Option<Projection> {
        radial_projection(self, observation, camera_point)
    }

    fn intrinsic_jacobian(self, camera_point: &[f64; 3]) -> Option<[[f64; 3]; 2]> {
        radial_intrinsic_jacobian(self, camera_point)
    }

    fn plus(self, delta: &[f64; 3]) -> Self {
        Self::new(
            self.focal * delta[0].exp(),
            self.k1 + delta[1],
            self.k2 + delta[2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::jet::Jet;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "actual={actual:e}, expected={expected:e}"
        );
    }

    #[test]
    fn radial_projection_uses_inverse_principal_branch() {
        let camera = RadialCamera::new(500.0, -0.1, 0.0);
        let folded = [10.0_f64.sqrt(), 0.0, 1.0];
        assert!(radial_projection(camera, &[0.0, 0.0], &folded).is_none());
        assert!(radial_linearization(camera, &[0.0, 0.0], &folded).is_none());
        assert!(radial_projection_signed_depth(camera, &[0.0, 0.0], &folded).is_some());
        let point = [0.5, 0.2, 1.0];
        let prediction = radial_terms(camera, &point).unwrap().prediction;
        let bearing = radial_bearing(camera, &prediction).unwrap();
        let norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
        for index in 0..3 {
            assert!((bearing[index] - point[index] / norm).abs() < 1e-12);
        }
        let extreme = RadialCamera::new(1.0, -1e308, 0.0);
        let bound = first_positive_radial_turn(extreme).unwrap();
        assert!(bound.is_finite() && bound > 0.0);
        assert!(radial_terms(extreme, &[(bound * 0.25).sqrt(), 0.0, 1.0]).is_some());
        assert!(radial_terms(extreme, &[(bound * 2.0).sqrt(), 0.0, 1.0]).is_none());
        let extreme_k2 = RadialCamera::new(1.0, 0.0, -1e308);
        let bound_k2 = first_positive_radial_turn(extreme_k2).unwrap();
        assert!(bound_k2.is_finite() && bound_k2 > 0.0);
        assert!(
            radial_projection(
                extreme_k2,
                &[0.0, 0.0],
                &[(bound_k2 * 0.25).sqrt(), 0.0, 1.0]
            )
            .is_some()
        );
    }

    #[test]
    fn point_jacobian_matches_central_differences() {
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let observation = [17.0, -23.0];
        let point = [0.71, -0.43, 2.4];
        let analytic = radial_projection(camera, &observation, &point)
            .expect("positive-depth point must project")
            .jacobian;
        let epsilon: f64 = 1e-6;
        for coordinate in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[coordinate] += epsilon;
            minus[coordinate] -= epsilon;
            let plus = radial_projection(camera, &observation, &plus)
                .expect("perturbed point must project")
                .residual;
            let minus = radial_projection(camera, &observation, &minus)
                .expect("perturbed point must project")
                .residual;
            for row in 0..2 {
                assert_close(
                    analytic[row][coordinate],
                    (plus[row] - minus[row]) / (2.0 * epsilon),
                    2e-8,
                );
            }
        }
    }

    #[test]
    fn intrinsic_jacobian_matches_log_focal_retraction() {
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let point = [0.71, -0.43, 2.4];
        let analytic =
            radial_intrinsic_jacobian(camera, &point).expect("positive-depth point must project");
        let analytic_columns: [[f64; 2]; 3] =
            std::array::from_fn(|column| [analytic[0][column], analytic[1][column]]);
        let epsilon: f64 = 1e-6;
        for (parameter, analytic_column) in analytic_columns.iter().enumerate() {
            let mut plus = camera;
            let mut minus = camera;
            match parameter {
                0 => {
                    plus.focal *= epsilon.exp();
                    minus.focal *= (-epsilon).exp();
                }
                1 => {
                    plus.k1 += epsilon;
                    minus.k1 -= epsilon;
                }
                2 => {
                    plus.k2 += epsilon;
                    minus.k2 -= epsilon;
                }
                _ => unreachable!(),
            }
            let plus = radial_terms(plus, &point)
                .expect("perturbed camera must project")
                .prediction;
            let minus = radial_terms(minus, &point)
                .expect("perturbed camera must project")
                .prediction;
            for row in 0..2 {
                assert_close(
                    analytic_column[row],
                    (plus[row] - minus[row]) / (2.0 * epsilon),
                    2e-8,
                );
            }
        }
    }

    #[test]
    fn radial_bearing_inverts_projection_direction() {
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let point = [0.71, -0.43, 2.4];
        let measurement = radial_projection(camera, &[0.0; 2], &point)
            .expect("positive-depth point must project")
            .residual;
        let actual = camera
            .bearing(measurement)
            .expect("radial inverse must converge");
        let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
        let expected = point.map(|value| value / norm);
        for coordinate in 0..3 {
            assert_close(actual[coordinate], expected[coordinate], 2.0e-13);
        }
    }

    #[test]
    fn radial_bearing_rejects_a_folded_inverse() {
        let camera = RadialCamera::new(1.0, -1.0, 0.0);
        assert_eq!(camera.bearing([0.5, 0.0]), None);
    }

    #[test]
    fn radial_forward_inverse_property_sweep_covers_valid_domain_boundary() {
        let bounded = [
            RadialCamera::new(730.0, -1.0, 0.0),
            RadialCamera::new(730.0, 0.08, -0.025),
        ];
        let fractions = [0.0, 1.0e-6, 0.05, 0.25, 0.75, 0.95, 0.99, 0.999];
        let azimuths = [
            0.0,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_2,
            2.3,
        ];
        let depths = [0.2, 1.0, 17.0];
        for camera in bounded {
            let boundary = first_orientation_fold(camera)
                .expect("bounded test camera must have an orientation fold");
            for fraction in fractions {
                let radius = fraction * boundary;
                for azimuth in azimuths {
                    for depth in depths {
                        assert_round_trip(camera, radius, azimuth, depth, 2.0e-10);
                    }
                }
            }
        }

        for camera in [
            RadialCamera::new(500.0, 0.0, 0.0),
            RadialCamera::new(700.0, 0.12, 0.02),
        ] {
            for radius in [0.0, 1.0e-8, 0.1, 0.5, 1.0, 2.0, 4.0] {
                for azimuth in azimuths {
                    for depth in depths {
                        assert_round_trip(camera, radius, azimuth, depth, 2.0e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn radial_inverse_accepts_just_inside_and_rejects_beyond_the_first_fold() {
        let camera = RadialCamera::new(1.0, -1.0, 0.0);
        let fold = first_orientation_fold(camera).unwrap();
        let inside = fold * (1.0 - 1.0e-6);
        let inside_measurement = inside * (1.0 + camera.k1 * inside * inside);
        assert!(radial_bearing(camera, &[inside_measurement, 0.0]).is_some());

        let maximum = fold * (1.0 + camera.k1 * fold * fold);
        assert_eq!(
            radial_bearing(camera, &[maximum * (1.0 + 1.0e-10), 0.0]),
            None
        );
    }

    fn assert_round_trip(
        camera: RadialCamera,
        radius: f64,
        azimuth: f64,
        depth: f64,
        tolerance: f64,
    ) {
        let point = [
            radius * azimuth.cos() * depth,
            radius * azimuth.sin() * depth,
            depth,
        ];
        let measurement = radial_projection(camera, &[0.0; 2], &point)
            .expect("valid property sample must project")
            .residual;
        let actual = radial_bearing(camera, &measurement)
            .expect("orientation-preserving property sample must invert");
        let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
        let expected = point.map(|value| value / norm);
        for coordinate in 0..3 {
            assert_close(actual[coordinate], expected[coordinate], tolerance);
        }
    }

    fn first_orientation_fold(camera: RadialCamera) -> Option<f64> {
        let derivative = |radius: f64| {
            let radius_squared = radius * radius;
            1.0 + 3.0 * camera.k1 * radius_squared
                + 5.0 * camera.k2 * radius_squared * radius_squared
        };
        let mut lower = 0.0;
        let mut upper = 1.0;
        while derivative(upper) > 0.0 && upper < 1.0e6 {
            lower = upper;
            upper *= 2.0;
        }
        if derivative(upper) > 0.0 {
            return None;
        }
        for _ in 0..80 {
            let middle = 0.5 * (lower + upper);
            if derivative(middle) > 0.0 {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        Some(0.5 * (lower + upper))
    }

    #[test]
    fn point_and_log_intrinsic_jacobians_match_dual_oracle() {
        const PARAMETERS: usize = 6;
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let observation = [17.0, -23.0];
        let point = [0.71, -0.43, 2.4];
        let point_dual: [Jet<PARAMETERS>; 3] =
            std::array::from_fn(|coordinate| Jet::variable(point[coordinate], coordinate));
        let log_focal = Jet::variable(camera.focal.ln(), 3);
        let k1 = Jet::variable(camera.k1, 4);
        let k2 = Jet::variable(camera.k2, 5);
        let inverse_depth = point_dual[2].recip();
        let normalized = [point_dual[0] * inverse_depth, point_dual[1] * inverse_depth];
        let radius_squared = normalized[0] * normalized[0] + normalized[1] * normalized[1];
        let distortion =
            Jet::constant(1.0) + k1 * radius_squared + k2 * radius_squared * radius_squared;
        let focal = log_focal.exp();
        let residual = [
            focal * distortion * normalized[0] - Jet::constant(observation[0]),
            focal * distortion * normalized[1] - Jet::constant(observation[1]),
        ];
        let projection = radial_projection(camera, &observation, &point).unwrap();
        let intrinsic = radial_intrinsic_jacobian(camera, &point).unwrap();
        for row in 0..2 {
            assert_close(residual[row].value, projection.residual[row], 5e-12);
            for coordinate in 0..3 {
                assert_close(
                    residual[row].deriv[coordinate],
                    projection.jacobian[row][coordinate],
                    5e-12,
                );
            }
            for (parameter, expected) in intrinsic[row].iter().enumerate() {
                assert_close(residual[row].deriv[3 + parameter], *expected, 5e-12);
            }
        }
    }

    #[test]
    fn radial_derivative_overflow_fails_closed() {
        let overflowing_point_jacobian = RadialCamera::new(f64::MAX / 2.0, 100.0, 0.0);
        assert!(
            radial_projection(overflowing_point_jacobian, &[0.0; 2], &[0.1, 0.0, 1.0]).is_none()
        );

        let overflowing_intrinsic_jacobian = RadialCamera::new(f64::MAX / 8.0, 0.1, 0.0);
        assert!(
            radial_intrinsic_jacobian(overflowing_intrinsic_jacobian, &[2.0, 0.0, 1.0]).is_none()
        );
    }

    #[test]
    fn fused_analytic_linearization_preserves_projection_and_intrinsic_bits() {
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let observation = [17.0, -23.0];
        let point = [0.71, -0.43, 2.4];
        let (fused_projection, fused_intrinsics) =
            radial_linearization(camera, &observation, &point).unwrap();
        let separate_projection = radial_projection(camera, &observation, &point).unwrap();
        let separate_intrinsics = radial_intrinsic_jacobian(camera, &point).unwrap();

        for (fused, separate) in fused_projection
            .residual
            .iter()
            .chain(fused_projection.jacobian.iter().flatten())
            .zip(
                separate_projection
                    .residual
                    .iter()
                    .chain(separate_projection.jacobian.iter().flatten()),
            )
        {
            assert_eq!(fused.to_bits(), separate.to_bits());
        }
        for (fused, separate) in fused_intrinsics
            .iter()
            .flatten()
            .zip(separate_intrinsics.iter().flatten())
        {
            assert_eq!(fused.to_bits(), separate.to_bits());
        }
    }

    #[test]
    fn camera_kernel_plus_preserves_retraction_bits() {
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let delta = [0.013, -0.004, 0.0007];
        let actual = <RadialCamera as CameraKernel<3>>::plus(camera, &delta);
        let expected = RadialCamera::new(
            camera.focal * delta[0].exp(),
            camera.k1 + delta[1],
            camera.k2 + delta[2],
        );

        assert_eq!(actual.focal.to_bits(), expected.focal.to_bits());
        assert_eq!(actual.k1.to_bits(), expected.k1.to_bits());
        assert_eq!(actual.k2.to_bits(), expected.k2.to_bits());
    }

    #[test]
    fn camera_kernel_delegates_projection_and_jacobian_bitwise() {
        let camera = RadialCamera::new(730.0, 0.08, -0.025);
        let observation = [17.0, -23.0];
        let point = [0.71, -0.43, 2.4];
        let actual_projection =
            <RadialCamera as CameraKernel<3>>::project(camera, &observation, &point).unwrap();
        let expected_projection = radial_projection(camera, &observation, &point).unwrap();
        for row in 0..2 {
            assert_eq!(
                actual_projection.residual[row].to_bits(),
                expected_projection.residual[row].to_bits()
            );
            for column in 0..3 {
                assert_eq!(
                    actual_projection.jacobian[row][column].to_bits(),
                    expected_projection.jacobian[row][column].to_bits()
                );
            }
        }

        let actual_intrinsic =
            <RadialCamera as CameraKernel<3>>::intrinsic_jacobian(camera, &point).unwrap();
        let expected_intrinsic = radial_intrinsic_jacobian(camera, &point).unwrap();
        for row in 0..2 {
            for column in 0..3 {
                assert_eq!(
                    actual_intrinsic[row][column].to_bits(),
                    expected_intrinsic[row][column].to_bits()
                );
            }
        }
    }
}
