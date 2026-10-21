// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private camera-model seam for multi-view optimization kernels.

use super::PixelObservation;
use crate::kernel::projection::Projection;
use nalgebra::{Point2, Vector3};
use openmvg_core::camera::{Camera as CoreCamera, CameraModel};

pub(super) trait CameraKernel<const INTRINSIC_DIMENSION: usize>: Copy {
    fn is_valid(self) -> bool;

    fn bearing(self, measurement: &[f64; 2]) -> Option<[f64; 3]>;

    fn project(self, observation: &[f64; 2], camera_point: &[f64; 3]) -> Option<Projection>;

    fn intrinsic_jacobian(self, camera_point: &[f64; 3])
    -> Option<[[f64; INTRINSIC_DIMENSION]; 2]>;

    fn plus(self, delta: &[f64; INTRINSIC_DIMENSION]) -> Self;
}

/// Borrowed fixed perspective camera using Core's exact pixel model and domain.
#[derive(Clone, Copy)]
pub(super) struct FixedPerspectiveCamera<'a> {
    camera: &'a CoreCamera,
}

impl<'a> FixedPerspectiveCamera<'a> {
    pub const fn new(camera: &'a CoreCamera) -> Self {
        Self { camera }
    }
}

impl CameraKernel<0> for FixedPerspectiveCamera<'_> {
    fn is_valid(self) -> bool {
        let (fx, fy, cx, cy) = match self.camera {
            CoreCamera::Pinhole(camera) => (
                camera.focal_x(),
                camera.focal_y(),
                camera.principal_x(),
                camera.principal_y(),
            ),
            CoreCamera::BrownConrady(camera) => {
                if camera.radial_distortion().validate().is_err()
                    || camera.tangential_distortion().validate().is_err()
                {
                    return false;
                }
                (
                    camera.focal_x(),
                    camera.focal_y(),
                    camera.principal_x(),
                    camera.principal_y(),
                )
            }
            CoreCamera::Fisheye(_) | CoreCamera::Spherical(_) => return false,
        };
        [fx, fy, cx, cy].iter().all(|value| value.is_finite()) && fx > 0.0 && fy > 0.0
    }

    fn bearing(self, measurement: &[f64; 2]) -> Option<[f64; 3]> {
        if !self.is_valid() {
            return None;
        }
        let bearing = self
            .camera
            .bearing(&Point2::new(measurement[0], measurement[1]))
            .ok()?;
        Some([bearing.x, bearing.y, bearing.z])
    }

    fn project(self, observation: &[f64; 2], camera_point: &[f64; 3]) -> Option<Projection> {
        if !matches!(
            self.camera,
            CoreCamera::Pinhole(_) | CoreCamera::BrownConrady(_)
        ) {
            return None;
        }
        let (pixel, jacobian) = self
            .camera
            .project_with_jacobian(&Vector3::from(*camera_point))
            .ok()?;
        let projection = Projection {
            residual: [pixel.x - observation[0], pixel.y - observation[1]],
            jacobian: std::array::from_fn(|row| {
                std::array::from_fn(|column| jacobian[(row, column)])
            }),
        };
        projection
            .residual
            .iter()
            .chain(projection.jacobian.as_flattened())
            .all(|value| value.is_finite())
            .then_some(projection)
    }

    fn intrinsic_jacobian(self, _camera_point: &[f64; 3]) -> Option<[[f64; 0]; 2]> {
        Some([[]; 2])
    }

    fn plus(self, _delta: &[f64; 0]) -> Self {
        self
    }
}

#[derive(Clone, Copy)]
pub(super) struct CameraTracks<
    'a,
    Camera: CameraKernel<INTRINSIC_DIMENSION>,
    const INTRINSIC_DIMENSION: usize,
> {
    observations: &'a [PixelObservation],
    point_offsets: &'a [usize],
    intrinsics: &'a [Camera],
    camera_to_group: Option<&'a [usize]>,
}

impl<'a, Camera: CameraKernel<INTRINSIC_DIMENSION>, const INTRINSIC_DIMENSION: usize>
    CameraTracks<'a, Camera, INTRINSIC_DIMENSION>
{
    pub const fn per_camera(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        cameras: &'a [Camera],
    ) -> Self {
        Self {
            observations,
            point_offsets,
            intrinsics: cameras,
            camera_to_group: None,
        }
    }

    pub const fn grouped(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        intrinsic_groups: &'a [Camera],
        camera_to_group: &'a [usize],
    ) -> Self {
        Self {
            observations,
            point_offsets,
            intrinsics: intrinsic_groups,
            camera_to_group: Some(camera_to_group),
        }
    }

    pub const fn observation_count(self) -> usize {
        self.observations.len()
    }

    pub const fn observations(self) -> &'a [PixelObservation] {
        self.observations
    }

    pub const fn offsets(self) -> &'a [usize] {
        self.point_offsets
    }

    pub const fn camera(self, observation: usize) -> usize {
        self.observations[observation].camera
    }

    pub const fn intrinsic_group(self, camera: usize) -> usize {
        match self.camera_to_group {
            Some(camera_to_group) => camera_to_group[camera],
            None => camera,
        }
    }

    pub const fn intrinsic_group_count(self) -> usize {
        self.intrinsics.len()
    }

    pub fn intrinsic(self, group: usize) -> Option<Camera> {
        self.intrinsics.get(group).copied()
    }

    pub const fn intrinsics(self) -> &'a [Camera] {
        self.intrinsics
    }

    pub const fn camera_to_group(self) -> Option<&'a [usize]> {
        self.camera_to_group
    }

    #[inline]
    pub fn project(self, observation: usize, camera_point: &[f64; 3]) -> Option<Projection> {
        let measurement = self.observations[observation];
        self.intrinsic_for_camera(measurement.camera)
            .project(&measurement.point, camera_point)
    }

    #[inline]
    pub fn bearing(self, observation: usize) -> Option<[f64; 3]> {
        let measurement = self.observations[observation];
        self.intrinsic_for_camera(measurement.camera)
            .bearing(&measurement.point)
    }

    #[inline]
    pub fn intrinsic_jacobian(
        self,
        observation: usize,
        camera_point: &[f64; 3],
    ) -> Option<[[f64; INTRINSIC_DIMENSION]; 2]> {
        let camera = self.observations[observation].camera;
        self.intrinsic_for_camera(camera)
            .intrinsic_jacobian(camera_point)
    }

    fn intrinsic_for_camera(self, camera: usize) -> Camera {
        self.intrinsics[self.intrinsic_group(camera)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::multi_view::self_calibration::RadialCamera;

    fn assert_projection_bits_equal(actual: Projection, expected: Projection) {
        for row in 0..2 {
            assert_eq!(
                actual.residual[row].to_bits(),
                expected.residual[row].to_bits()
            );
            for column in 0..3 {
                assert_eq!(
                    actual.jacobian[row][column].to_bits(),
                    expected.jacobian[row][column].to_bits()
                );
            }
        }
    }

    #[test]
    fn fixed_perspective_preserves_core_projection_and_zero_intrinsic_dimension() {
        use openmvg_core::camera::{RadialDistortion, TangentialDistortion};
        let camera = CoreCamera::try_brown_conrady(
            800.0,
            600.0,
            320.0,
            240.0,
            RadialDistortion::K1K2K3(-0.08, 0.012, -0.001),
            TangentialDistortion::P1P2(0.0007, -0.0004),
        )
        .unwrap();
        let adapter = FixedPerspectiveCamera::new(&camera);
        assert!(adapter.is_valid());
        let point = [0.71, -0.43, 2.4];
        let observation = [17.0, -23.0];
        let (pixel, jacobian) = camera.project_with_jacobian(&Vector3::from(point)).unwrap();
        let expected = Projection {
            residual: [pixel.x - observation[0], pixel.y - observation[1]],
            jacobian: std::array::from_fn(|row| {
                std::array::from_fn(|column| jacobian[(row, column)])
            }),
        };
        assert_projection_bits_equal(adapter.project(&observation, &point).unwrap(), expected);
        let bearing = adapter.bearing(&[pixel.x, pixel.y]).unwrap();
        let direction = Vector3::from(point).normalize();
        assert!((Vector3::from(bearing) - direction).norm() < 1e-12);
        assert_eq!(adapter.intrinsic_jacobian(&point), Some([[]; 2]));
        assert_projection_bits_equal(
            adapter.plus(&[]).project(&observation, &point).unwrap(),
            expected,
        );
        assert!(adapter.project(&observation, &[0.1, 0.2, -1.0]).is_none());
        assert!(adapter.project(&[f64::NAN, 0.0], &point).is_none());

        // An independent one-axis derivative checks the k3 and unequal-focal contract.
        let radial = CoreCamera::try_brown_conrady(
            800.0,
            600.0,
            0.0,
            0.0,
            RadialDistortion::K1K2K3(-0.08, 0.012, -0.001),
            TangentialDistortion::None,
        )
        .unwrap();
        let x = 0.4_f64;
        let q = x * x;
        let projection = FixedPerspectiveCamera::new(&radial)
            .project(&[0.0, 0.0], &[x, 0.0, 1.0])
            .unwrap();
        let factor = 1.0 - 0.08 * q + 0.012 * q * q - 0.001 * q * q * q;
        let slope = 1.0 - 3.0 * 0.08 * q + 5.0 * 0.012 * q * q - 7.0 * 0.001 * q * q * q;
        assert!((projection.residual[0] - 800.0 * x * factor).abs() < 1e-12);
        assert!((projection.jacobian[0][0] - 800.0 * slope).abs() < 1e-12);
        assert!((projection.jacobian[1][1] - 600.0 * factor).abs() < 1e-12);
    }

    #[test]
    fn radial_tracks_preserve_projection_and_jacobian_bits() {
        let cameras = [
            RadialCamera::new(410.0, -0.03, 0.004),
            RadialCamera::new(730.0, 0.08, -0.025),
        ];
        let observations = [PixelObservation::new(1, [17.0, -23.0])];
        let offsets = [0, 1];
        let point = [0.71, -0.43, 2.4];
        let tracks: CameraTracks<'_, RadialCamera, 3> =
            CameraTracks::per_camera(&observations, &offsets, &cameras);

        assert_projection_bits_equal(
            tracks.project(0, &point).unwrap(),
            <RadialCamera as CameraKernel<3>>::project(cameras[1], &observations[0].point, &point)
                .unwrap(),
        );
        let actual = tracks.intrinsic_jacobian(0, &point).unwrap();
        let expected =
            <RadialCamera as CameraKernel<3>>::intrinsic_jacobian(cameras[1], &point).unwrap();
        for row in 0..2 {
            for column in 0..3 {
                assert_eq!(
                    actual[row][column].to_bits(),
                    expected[row][column].to_bits()
                );
            }
        }
    }

    #[test]
    fn grouped_tracks_select_the_existing_intrinsic_block() {
        let groups = [
            RadialCamera::new(410.0, -0.03, 0.004),
            RadialCamera::new(730.0, 0.08, -0.025),
        ];
        let observations = [PixelObservation::new(2, [17.0, -23.0])];
        let offsets = [0, 1];
        let camera_to_group = [0, 0, 1];
        let tracks: CameraTracks<'_, RadialCamera, 3> =
            CameraTracks::grouped(&observations, &offsets, &groups, &camera_to_group);

        assert_eq!(tracks.intrinsic_group(2), 1);
        assert_eq!(tracks.intrinsic(1), Some(groups[1]));
        assert_eq!(tracks.intrinsic_group_count(), groups.len());
    }
}
