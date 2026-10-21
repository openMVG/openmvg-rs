// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::numerics::manifold::Mat3;

use super::{BundleAdjustmentError, PinholeCamera};
use crate::kernel::projection::{Projection, angular_projection, pixel_projection};

pub(super) trait MeasurementModel: Copy {
    fn validate(&self, points: usize) -> Result<(), BundleAdjustmentError>;
    fn project(&self, view: usize, index: usize, camera_point: &[f64; 3]) -> Option<Projection>;
}

#[derive(Clone, Copy)]
pub(super) struct PixelModel<'a> {
    pub observations1: &'a [[f64; 2]],
    pub observations2: &'a [[f64; 2]],
    pub cameras: [PinholeCamera; 2],
}

impl MeasurementModel for PixelModel<'_> {
    fn validate(&self, points: usize) -> Result<(), BundleAdjustmentError> {
        validate_lengths(points, self.observations1.len(), self.observations2.len())?;
        for (view, observations) in [self.observations1, self.observations2]
            .into_iter()
            .enumerate()
        {
            for (index, observation) in observations.iter().enumerate() {
                if observation.iter().any(|value| !value.is_finite()) {
                    return Err(BundleAdjustmentError::NonFiniteMeasurement {
                        view: view + 1,
                        index,
                    });
                }
            }
        }
        for (view, camera) in self.cameras.iter().enumerate() {
            if !camera.is_valid() {
                return Err(BundleAdjustmentError::InvalidIntrinsics { view: view + 1 });
            }
        }
        Ok(())
    }

    #[inline]
    fn project(&self, view: usize, index: usize, point: &[f64; 3]) -> Option<Projection> {
        let camera = self.cameras[view];
        let observation = [self.observations1, self.observations2][view][index];
        pixel_projection(camera, &observation, point)
    }
}

#[derive(Clone, Copy)]
pub(super) struct AngularModel<'a> {
    pub bearings1: &'a [[f64; 3]],
    pub bearings2: &'a [[f64; 3]],
}

impl MeasurementModel for AngularModel<'_> {
    fn validate(&self, points: usize) -> Result<(), BundleAdjustmentError> {
        validate_lengths(points, self.bearings1.len(), self.bearings2.len())?;
        for (view, bearings) in [self.bearings1, self.bearings2].into_iter().enumerate() {
            for (index, bearing) in bearings.iter().enumerate() {
                let norm_squared = dot(bearing, bearing);
                if bearing.iter().any(|value| !value.is_finite())
                    || !(norm_squared.is_finite() && norm_squared > f64::MIN_POSITIVE)
                {
                    return Err(BundleAdjustmentError::InvalidBearing {
                        view: view + 1,
                        index,
                    });
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn project(&self, view: usize, index: usize, point: &[f64; 3]) -> Option<Projection> {
        let raw_observation = [self.bearings1, self.bearings2][view][index];
        angular_projection(&raw_observation, point)
    }
}

fn validate_lengths(
    points: usize,
    view1: usize,
    view2: usize,
) -> Result<(), BundleAdjustmentError> {
    if points == view1 && points == view2 {
        Ok(())
    } else {
        Err(BundleAdjustmentError::MeasurementCountMismatch {
            points,
            view1,
            view2,
        })
    }
}

#[inline(always)]
fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn transform(rotation: &Mat3, translation: &[f64; 3], point: &[f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] * point[0]
            + rotation[0][1] * point[1]
            + rotation[0][2] * point[2]
            + translation[0],
        rotation[1][0] * point[0]
            + rotation[1][1] * point[1]
            + rotation[1][2] * point[2]
            + translation[1],
        rotation[2][0] * point[0]
            + rotation[2][1] * point[1]
            + rotation[2][2] * point[2]
            + translation[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::projection::{normalize, tangent_basis};
    use crate::numerics::jet::Jet;

    fn dual_pixel_projection(
        camera: PinholeCamera,
        observation: &[f64; 2],
        point: [Jet<3>; 3],
    ) -> [Jet<3>; 2] {
        let inverse_depth = point[2].recip();
        let x_numerator =
            Jet::constant(camera.fx) * point[0] + Jet::constant(camera.skew) * point[1];
        [
            x_numerator * inverse_depth + Jet::constant(camera.cx - observation[0]),
            Jet::constant(camera.fy) * point[1] * inverse_depth
                + Jet::constant(camera.cy - observation[1]),
        ]
    }

    fn dual_angular_projection(raw_observation: &[f64; 3], point: [Jet<3>; 3]) -> [Jet<3>; 2] {
        let observation = normalize(raw_observation).expect("test observation is valid");
        let basis = tangent_basis(&observation);
        let point_norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
        let prediction = point.map(|coordinate| coordinate / point_norm);
        let dual_dot = |left: &[f64; 3], right: &[Jet<3>; 3]| {
            Jet::constant(left[0]) * right[0]
                + Jet::constant(left[1]) * right[1]
                + Jet::constant(left[2]) * right[2]
        };
        let a = dual_dot(&basis[0], &prediction);
        let b = dual_dot(&basis[1], &prediction);
        let c = dual_dot(&observation, &prediction);
        let sine_squared = a * a + b * b;
        let factor = if sine_squared.value.sqrt() < 1e-6 {
            Jet::constant(1.0)
                + sine_squared * Jet::constant(1.0 / 6.0)
                + sine_squared * sine_squared * Jet::constant(3.0 / 40.0)
        } else {
            let sine = sine_squared.sqrt();
            sine.atan2(c) / sine
        };
        [factor * a, factor * b]
    }

    fn seeded_point(point: [f64; 3]) -> [Jet<3>; 3] {
        std::array::from_fn(|coordinate| Jet::variable(point[coordinate], coordinate))
    }

    fn assert_dual_projection(analytic: Projection, dual: [Jet<3>; 2], tolerance: f64) {
        for (row, dual_residual) in dual.iter().enumerate() {
            assert!((analytic.residual[row] - dual_residual.value).abs() <= tolerance);
            for coordinate in 0..3 {
                let expected = analytic.jacobian[row][coordinate];
                let actual = dual_residual.deriv[coordinate];
                let scale = 1.0_f64.max(expected.abs()).max(actual.abs());
                assert!(
                    (expected - actual).abs() <= tolerance * scale,
                    "row={row}, coordinate={coordinate}, analytic={expected}, dual={actual}"
                );
            }
        }
    }

    fn check_projection_jacobian<M: MeasurementModel>(model: M, point: [f64; 3]) {
        let analytic = model
            .project(0, 0, &point)
            .expect("test projection is valid");
        let step = 1e-7;
        for coordinate in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let plus = model
                .project(0, 0, &plus)
                .expect("test projection is valid");
            let minus = model
                .project(0, 0, &minus)
                .expect("test projection is valid");
            for row in 0..2 {
                let numerical = (plus.residual[row] - minus.residual[row]) / (2.0 * step);
                assert!(
                    (analytic.jacobian[row][coordinate] - numerical).abs()
                        < 2e-7 * numerical.abs().max(1.0),
                    "row={row}, coordinate={coordinate}, analytic={}, numerical={numerical}",
                    analytic.jacobian[row][coordinate]
                );
            }
        }
    }

    #[test]
    fn pixel_and_angular_projection_jacobians_match_finite_differences() {
        check_projection_jacobian(
            PixelModel {
                observations1: &[[330.0, 220.0]],
                observations2: &[[330.0, 220.0]],
                cameras: [PinholeCamera::new(500.0, 510.0, 320.0, 240.0); 2],
            },
            [0.2, -0.15, 3.0],
        );
        check_projection_jacobian(
            AngularModel {
                bearings1: &[[0.05, -0.03, 1.0]],
                bearings2: &[[0.05, -0.03, 1.0]],
            },
            [0.2, -0.15, 3.0],
        );
    }

    #[test]
    fn pixel_and_angular_projection_jacobians_match_dual_oracle() {
        let camera = PinholeCamera::new(500.0, 510.0, 320.0, 240.0).with_skew(0.7);
        let pixel_observation = [330.0, 220.0];
        let regular_point = [0.2, -0.15, 3.0];
        assert_dual_projection(
            pixel_projection(camera, &pixel_observation, &regular_point).unwrap(),
            dual_pixel_projection(camera, &pixel_observation, seeded_point(regular_point)),
            5e-12,
        );

        let angular_observation = normalize(&[0.05, -0.03, 1.0]).unwrap();
        assert_dual_projection(
            angular_projection(&angular_observation, &regular_point).unwrap(),
            dual_angular_projection(&angular_observation, seeded_point(regular_point)),
            5e-12,
        );
        let near_axis = [
            angular_observation[0] + 2e-8,
            angular_observation[1] - 3e-8,
            angular_observation[2] + 1e-8,
        ];
        assert_dual_projection(
            angular_projection(&angular_observation, &near_axis).unwrap(),
            dual_angular_projection(&angular_observation, seeded_point(near_axis)),
            1e-10,
        );
    }

    #[test]
    fn projection_overflow_fails_closed() {
        let camera = PinholeCamera::new(f64::MAX, 1.0, 0.0, 0.0);
        assert!(pixel_projection(camera, &[0.0; 2], &[2.0, 0.0, 1.0]).is_none());
    }
}
