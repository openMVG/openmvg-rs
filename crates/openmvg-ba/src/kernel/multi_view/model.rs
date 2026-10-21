// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{PinholeCamera, RelativePose};
use super::camera_kernel::{CameraKernel, CameraTracks, FixedPerspectiveCamera};
use super::self_calibration::model::{
    radial_intrinsic_jacobian_signed_depth, radial_linearization,
    radial_linearization_signed_depth, radial_projection_signed_depth,
};
use super::{
    AngularObservation, MultiViewBundleAdjustmentError, MultiViewMeasurements, PixelObservation,
    self_calibration::RadialCamera,
};
use crate::kernel::projection::{Projection, angular_projection, pixel_projection};
#[cfg(all(feature = "parallel", feature = "faer"))]
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RadialDepthDomain {
    Positive,
    SignedNonzero,
}

/// Residual model and physical domain bound into a reusable workspace signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ObservationModelIdentity {
    Angular,
    Pinhole,
    FixedPerspective,
    FixedPerspectiveIndexed,
    RadialPositive,
    RadialSigned,
}

#[derive(Clone, Copy)]
pub(super) struct ObservationModel<'a> {
    kind: ObservationKind<'a>,
    radial_depth: RadialDepthDomain,
}

#[derive(Clone, Copy)]
enum ObservationKind<'a> {
    Angular {
        observations: &'a [AngularObservation],
        point_offsets: &'a [usize],
    },
    Pixel {
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        cameras: &'a [PinholeCamera],
    },
    FixedPerspective {
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        cameras: &'a [openmvg_core::camera::Camera],
        observation_intrinsics: Option<&'a [usize]>,
    },
    Radial(CameraTracks<'a, RadialCamera, 3>),
}

impl<'a> From<MultiViewMeasurements<'a>> for ObservationModel<'a> {
    fn from(measurements: MultiViewMeasurements<'a>) -> Self {
        Self::calibrated(measurements)
    }
}

impl<'a> ObservationModel<'a> {
    pub const fn calibrated(measurements: MultiViewMeasurements<'a>) -> Self {
        match measurements {
            MultiViewMeasurements::Angular {
                observations,
                point_offsets,
            } => Self {
                kind: ObservationKind::Angular {
                    observations,
                    point_offsets,
                },
                radial_depth: RadialDepthDomain::Positive,
            },
            MultiViewMeasurements::Pixel {
                observations,
                point_offsets,
                cameras,
            } => Self {
                kind: ObservationKind::Pixel {
                    observations,
                    point_offsets,
                    cameras,
                },
                radial_depth: RadialDepthDomain::Positive,
            },
            MultiViewMeasurements::FixedPerspective {
                observations,
                point_offsets,
                cameras,
            } => Self {
                kind: ObservationKind::FixedPerspective {
                    observations,
                    point_offsets,
                    cameras,
                    observation_intrinsics: None,
                },
                radial_depth: RadialDepthDomain::Positive,
            },
            MultiViewMeasurements::FixedPerspectiveIndexed {
                observations,
                point_offsets,
                cameras,
                observation_intrinsics,
            } => Self {
                kind: ObservationKind::FixedPerspective {
                    observations,
                    point_offsets,
                    cameras,
                    observation_intrinsics: Some(observation_intrinsics),
                },
                radial_depth: RadialDepthDomain::Positive,
            },
        }
    }

    pub const fn identity(self) -> ObservationModelIdentity {
        match self.kind {
            ObservationKind::Angular { .. } => ObservationModelIdentity::Angular,
            ObservationKind::Pixel { .. } => ObservationModelIdentity::Pinhole,
            ObservationKind::FixedPerspective {
                observation_intrinsics,
                ..
            } => {
                if observation_intrinsics.is_some() {
                    ObservationModelIdentity::FixedPerspectiveIndexed
                } else {
                    ObservationModelIdentity::FixedPerspective
                }
            }
            ObservationKind::Radial(_) => match self.radial_depth {
                RadialDepthDomain::Positive => ObservationModelIdentity::RadialPositive,
                RadialDepthDomain::SignedNonzero => ObservationModelIdentity::RadialSigned,
            },
        }
    }

    pub const fn observation_intrinsics(self) -> Option<&'a [usize]> {
        match self.kind {
            ObservationKind::FixedPerspective {
                observation_intrinsics,
                ..
            } => observation_intrinsics,
            _ => None,
        }
    }

    fn fixed_perspective_camera(self, observation: usize) -> Option<FixedPerspectiveCamera<'a>> {
        let ObservationKind::FixedPerspective {
            observations,
            cameras,
            observation_intrinsics,
            ..
        } = self.kind
        else {
            return None;
        };
        let camera = match observation_intrinsics {
            Some(selectors) => *selectors.get(observation)?,
            None => observations.get(observation)?.camera,
        };
        cameras.get(camera).map(FixedPerspectiveCamera::new)
    }

    pub const fn radial(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        cameras: &'a [RadialCamera],
    ) -> Self {
        Self {
            kind: ObservationKind::Radial(CameraTracks::per_camera(
                observations,
                point_offsets,
                cameras,
            )),
            radial_depth: RadialDepthDomain::Positive,
        }
    }

    pub const fn radial_signed_depth(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        cameras: &'a [RadialCamera],
    ) -> Self {
        Self {
            kind: ObservationKind::Radial(CameraTracks::per_camera(
                observations,
                point_offsets,
                cameras,
            )),
            radial_depth: RadialDepthDomain::SignedNonzero,
        }
    }

    pub const fn radial_grouped(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        intrinsic_groups: &'a [RadialCamera],
        camera_to_group: &'a [usize],
    ) -> Self {
        Self {
            kind: ObservationKind::Radial(CameraTracks::grouped(
                observations,
                point_offsets,
                intrinsic_groups,
                camera_to_group,
            )),
            radial_depth: RadialDepthDomain::Positive,
        }
    }

    pub const fn radial_grouped_signed_depth(
        observations: &'a [PixelObservation],
        point_offsets: &'a [usize],
        intrinsic_groups: &'a [RadialCamera],
        camera_to_group: &'a [usize],
    ) -> Self {
        Self {
            kind: ObservationKind::Radial(CameraTracks::grouped(
                observations,
                point_offsets,
                intrinsic_groups,
                camera_to_group,
            )),
            radial_depth: RadialDepthDomain::SignedNonzero,
        }
    }

    pub fn radial_analytic_linearization(
        self,
        observation: usize,
        camera_point: [f64; 3],
    ) -> Option<(Projection, [[f64; 3]; 2])> {
        let ObservationKind::Radial(tracks) = self.kind else {
            return None;
        };
        let measurement = tracks.observations()[observation];
        let camera = tracks.intrinsic(tracks.intrinsic_group(measurement.camera))?;
        if matches!(self.radial_depth, RadialDepthDomain::SignedNonzero) {
            radial_linearization_signed_depth(camera, &measurement.point, &camera_point)
        } else {
            radial_linearization(camera, &measurement.point, &camera_point)
        }
    }

    pub fn observation_count(self) -> usize {
        match self.kind {
            ObservationKind::Angular { observations, .. } => observations.len(),
            ObservationKind::Pixel { observations, .. }
            | ObservationKind::FixedPerspective { observations, .. } => observations.len(),
            ObservationKind::Radial(tracks) => tracks.observation_count(),
        }
    }

    #[cfg(feature = "unstable-internals")]
    pub fn point_count(self) -> usize {
        self.offsets().len().saturating_sub(1)
    }

    pub fn offsets(self) -> &'a [usize] {
        match self.kind {
            ObservationKind::Angular { point_offsets, .. }
            | ObservationKind::Pixel { point_offsets, .. }
            | ObservationKind::FixedPerspective { point_offsets, .. } => point_offsets,
            ObservationKind::Radial(tracks) => tracks.offsets(),
        }
    }

    #[inline(always)]
    pub fn camera(self, observation: usize) -> usize {
        match self.kind {
            ObservationKind::Angular { observations, .. } => observations[observation].camera,
            ObservationKind::Pixel { observations, .. }
            | ObservationKind::FixedPerspective { observations, .. } => {
                observations[observation].camera
            }
            ObservationKind::Radial(tracks) => tracks.camera(observation),
        }
    }

    #[inline]
    pub fn project(self, observation: usize, camera_point: &[f64; 3]) -> Option<Projection> {
        match self.kind {
            ObservationKind::Angular { observations, .. } => {
                angular_projection(&observations[observation].bearing, camera_point)
            }
            ObservationKind::Pixel {
                observations,
                cameras,
                ..
            } => {
                let observation = observations[observation];
                pixel_projection(
                    cameras[observation.camera],
                    &observation.point,
                    camera_point,
                )
            }
            ObservationKind::FixedPerspective { observations, .. } => self
                .fixed_perspective_camera(observation)?
                .project(&observations[observation].point, camera_point),
            ObservationKind::Radial(tracks) => {
                if matches!(self.radial_depth, RadialDepthDomain::SignedNonzero) {
                    let measurement = tracks.observations()[observation];
                    let camera = tracks.intrinsic(tracks.intrinsic_group(measurement.camera))?;
                    radial_projection_signed_depth(camera, &measurement.point, camera_point)
                } else {
                    tracks.project(observation, camera_point)
                }
            }
        }
    }

    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn observation_bearing(self, observation: usize) -> Option<[f64; 3]> {
        match self.kind {
            ObservationKind::Angular { observations, .. } => {
                normalize(&observations[observation].bearing)
            }
            ObservationKind::Pixel {
                observations,
                cameras,
                ..
            } => {
                let observation = observations[observation];
                let camera = cameras[observation.camera];
                let y = (observation.point[1] - camera.cy) / camera.fy;
                let x = (observation.point[0] - camera.cx - camera.skew * y) / camera.fx;
                normalize(&[x, y, 1.0])
            }
            ObservationKind::FixedPerspective { observations, .. } => self
                .fixed_perspective_camera(observation)?
                .bearing(&observations[observation].point),
            ObservationKind::Radial(tracks) => tracks.bearing(observation),
        }
    }

    /// Number of measurements that cannot be mapped to an unambiguous camera
    /// bearing by the final camera model.
    ///
    /// Calibrated angular and pinhole observations report zero after ordinary
    /// input validation. Centered radial observations outside the central
    /// orientation-preserving branch are counted instead of being hidden by a
    /// finite forward projection.
    pub fn noninvertible_measurement_count(self) -> usize {
        match self.kind {
            ObservationKind::Radial(tracks) => (0..tracks.observation_count())
                .filter(|&observation| tracks.bearing(observation).is_none())
                .count(),
            ObservationKind::FixedPerspective { observations, .. } => (0..observations.len())
                .filter(|&observation| {
                    self.fixed_perspective_camera(observation)
                        .and_then(|camera| camera.bearing(&observations[observation].point))
                        .is_none()
                })
                .count(),
            ObservationKind::Angular { .. } | ObservationKind::Pixel { .. } => 0,
        }
    }

    #[cfg(all(feature = "parallel", feature = "faer"))]
    pub fn noninvertible_measurement_count_parallel(self) -> usize {
        match self.kind {
            ObservationKind::Radial(tracks) => (0..tracks.observation_count())
                .into_par_iter()
                .filter(|&observation| tracks.bearing(observation).is_none())
                .count(),
            ObservationKind::FixedPerspective { observations, .. } => (0..observations.len())
                .into_par_iter()
                .filter(|&observation| {
                    self.fixed_perspective_camera(observation)
                        .and_then(|camera| camera.bearing(&observations[observation].point))
                        .is_none()
                })
                .count(),
            ObservationKind::Angular { .. } | ObservationKind::Pixel { .. } => 0,
        }
    }

    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn positive_depth(self, observation: usize, camera_point: &[f64; 3]) -> Option<bool> {
        match self.kind {
            ObservationKind::Angular { observations, .. } => {
                Some(dot(&observations[observation].bearing, camera_point) > 0.0)
            }
            ObservationKind::Pixel { .. } | ObservationKind::FixedPerspective { .. } => {
                Some(camera_point[2] > 0.0)
            }
            ObservationKind::Radial(_) => {
                let scale = camera_point
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f64::max);
                let floor = 64.0 * f64::EPSILON * scale;
                Some(
                    if matches!(self.radial_depth, RadialDepthDomain::SignedNonzero) {
                        camera_point[2].abs() > floor
                    } else {
                        camera_point[2] > floor
                    },
                )
            }
        }
    }

    pub const fn intrinsic_dimension(self) -> usize {
        if matches!(self.kind, ObservationKind::Radial(_)) {
            3
        } else {
            0
        }
    }

    /// Index of the intrinsic block used by `camera`.
    pub const fn intrinsic_group(self, camera: usize) -> Option<usize> {
        match self.kind {
            ObservationKind::Radial(tracks) => Some(tracks.intrinsic_group(camera)),
            ObservationKind::Angular { .. }
            | ObservationKind::Pixel { .. }
            | ObservationKind::FixedPerspective { .. } => None,
        }
    }

    pub fn intrinsic_group_count(self) -> usize {
        match self.kind {
            ObservationKind::Radial(tracks) => tracks.intrinsic_group_count(),
            ObservationKind::Angular { .. }
            | ObservationKind::Pixel { .. }
            | ObservationKind::FixedPerspective { .. } => 0,
        }
    }

    pub fn intrinsic_value(self, group: usize) -> Option<RadialCamera> {
        match self.kind {
            ObservationKind::Radial(tracks) => tracks.intrinsic(group),
            ObservationKind::Angular { .. }
            | ObservationKind::Pixel { .. }
            | ObservationKind::FixedPerspective { .. } => None,
        }
    }

    pub fn intrinsic_jacobian(
        self,
        observation: usize,
        camera_point: &[f64; 3],
    ) -> Option<[[f64; 3]; 2]> {
        match self.kind {
            ObservationKind::Radial(tracks) => {
                if matches!(self.radial_depth, RadialDepthDomain::SignedNonzero) {
                    let measurement = tracks.observations()[observation];
                    let camera = tracks.intrinsic(tracks.intrinsic_group(measurement.camera))?;
                    radial_intrinsic_jacobian_signed_depth(camera, camera_point)
                } else {
                    tracks.intrinsic_jacobian(observation, camera_point)
                }
            }
            ObservationKind::Angular { .. }
            | ObservationKind::Pixel { .. }
            | ObservationKind::FixedPerspective { .. } => None,
        }
    }

    #[cfg(any(test, feature = "unstable-internals"))]
    pub fn validate(
        self,
        camera_count: usize,
        point_count: usize,
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        self.validate_with_fixed_points(camera_count, point_count, &[])
    }

    pub fn validate_with_fixed_points(
        self,
        camera_count: usize,
        point_count: usize,
        fixed_points: &[bool],
    ) -> Result<(), MultiViewBundleAdjustmentError> {
        if !fixed_points.is_empty() && fixed_points.len() != point_count {
            return Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
                block: super::BundleAdjustmentBlock::Point,
                expected: point_count,
                actual: fixed_points.len(),
            });
        }
        let offsets = self.offsets();
        if offsets.len() != point_count + 1
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(self.observation_count())
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(MultiViewBundleAdjustmentError::InvalidTrackOffsets);
        }
        match self.kind {
            ObservationKind::Pixel { cameras, .. } => {
                validate_cameras(camera_count, cameras.len(), |camera| {
                    cameras[camera].is_valid()
                })?;
            }
            ObservationKind::FixedPerspective {
                cameras,
                observation_intrinsics,
                ..
            } => {
                if let Some(selectors) = observation_intrinsics {
                    if selectors.len() != self.observation_count() {
                        return Err(
                            MultiViewBundleAdjustmentError::ObservationIntrinsicCountMismatch {
                                expected: self.observation_count(),
                                actual: selectors.len(),
                            },
                        );
                    }
                } else if cameras.len() != camera_count {
                    return Err(MultiViewBundleAdjustmentError::IntrinsicsCountMismatch {
                        expected: camera_count,
                        actual: cameras.len(),
                    });
                }
            }
            ObservationKind::Radial(tracks) => match tracks.camera_to_group() {
                Some(camera_to_group) => {
                    validate_intrinsic_groups(camera_count, tracks.intrinsics(), camera_to_group)?
                }
                None => validate_cameras(camera_count, tracks.intrinsic_group_count(), |camera| {
                    tracks
                        .intrinsic(camera)
                        .is_some_and(<RadialCamera as CameraKernel<3>>::is_valid)
                })?,
            },
            ObservationKind::Angular { .. } => {}
        }
        for observation in 0..self.observation_count() {
            let camera = self.camera(observation);
            if camera >= camera_count {
                return Err(MultiViewBundleAdjustmentError::InvalidObservationCamera {
                    observation,
                    camera,
                });
            }
            match self.kind {
                ObservationKind::Angular { observations, .. } => {
                    let bearing = observations[observation].bearing;
                    let norm_squared = dot(&bearing, &bearing);
                    if bearing.iter().any(|value| !value.is_finite())
                        || !(norm_squared.is_finite() && norm_squared > f64::MIN_POSITIVE)
                    {
                        return Err(MultiViewBundleAdjustmentError::InvalidBearing { observation });
                    }
                }
                ObservationKind::Pixel { observations, .. }
                | ObservationKind::FixedPerspective { observations, .. } => {
                    if observations[observation]
                        .point
                        .iter()
                        .any(|value| !value.is_finite())
                    {
                        return Err(MultiViewBundleAdjustmentError::NonFiniteMeasurement {
                            observation,
                        });
                    }
                }
                ObservationKind::Radial(tracks) => {
                    if tracks.observations()[observation]
                        .point
                        .iter()
                        .any(|value| !value.is_finite())
                    {
                        return Err(MultiViewBundleAdjustmentError::NonFiniteMeasurement {
                            observation,
                        });
                    }
                }
            }
            if let ObservationKind::FixedPerspective {
                cameras,
                observation_intrinsics,
                ..
            } = self.kind
            {
                let intrinsic =
                    observation_intrinsics.map_or(camera, |selectors| selectors[observation]);
                let Some(model) = self.fixed_perspective_camera(observation) else {
                    return Err(
                        MultiViewBundleAdjustmentError::InvalidObservationIntrinsic {
                            observation,
                            intrinsic,
                            intrinsic_count: cameras.len(),
                        },
                    );
                };
                if !model.is_valid() {
                    return Err(MultiViewBundleAdjustmentError::InvalidIntrinsics {
                        camera: intrinsic,
                    });
                }
            }
        }
        for point in 0..point_count {
            let range = offsets[point]..offsets[point + 1];
            if range.len() < 2 && !fixed_points.get(point).copied().unwrap_or(false) {
                return Err(
                    MultiViewBundleAdjustmentError::InsufficientTrackObservations {
                        point,
                        actual: range.len(),
                    },
                );
            }
            // Rows are independent measurements. Repeated physical poses are
            // retained; free-point depth is assessed from the summed information.
        }
        Ok(())
    }
}

fn validate_intrinsic_groups(
    camera_count: usize,
    intrinsic_groups: &[RadialCamera],
    camera_to_group: &[usize],
) -> Result<(), MultiViewBundleAdjustmentError> {
    if camera_to_group.len() != camera_count {
        return Err(
            MultiViewBundleAdjustmentError::IntrinsicGroupMapCountMismatch {
                expected: camera_count,
                actual: camera_to_group.len(),
            },
        );
    }
    if intrinsic_groups.is_empty() || intrinsic_groups.len() > camera_count {
        return Err(MultiViewBundleAdjustmentError::InvalidIntrinsicGroupCount {
            cameras: camera_count,
            groups: intrinsic_groups.len(),
        });
    }
    for (group, camera) in intrinsic_groups.iter().enumerate() {
        if !<RadialCamera as CameraKernel<3>>::is_valid(*camera) {
            return Err(MultiViewBundleAdjustmentError::InvalidIntrinsics { camera: group });
        }
    }
    for (camera, &group) in camera_to_group.iter().enumerate() {
        if group >= intrinsic_groups.len() {
            return Err(MultiViewBundleAdjustmentError::InvalidIntrinsicGroup {
                camera,
                group,
                groups: intrinsic_groups.len(),
            });
        }
    }
    for group in 0..intrinsic_groups.len() {
        if !camera_to_group.contains(&group) {
            return Err(MultiViewBundleAdjustmentError::UnusedIntrinsicGroup { group });
        }
    }
    Ok(())
}

fn validate_cameras(
    expected: usize,
    actual: usize,
    is_valid: impl Fn(usize) -> bool,
) -> Result<(), MultiViewBundleAdjustmentError> {
    if actual != expected {
        return Err(MultiViewBundleAdjustmentError::IntrinsicsCountMismatch { expected, actual });
    }
    for camera in 0..actual {
        if !is_valid(camera) {
            return Err(MultiViewBundleAdjustmentError::InvalidIntrinsics { camera });
        }
    }
    Ok(())
}

#[inline(always)]
fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[cfg(any(test, feature = "unstable-internals"))]
fn normalize(vector: &[f64; 3]) -> Option<[f64; 3]> {
    let magnitude = vector
        .iter()
        .zip(vector)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .sqrt();
    (magnitude.is_finite() && magnitude > f64::MIN_POSITIVE)
        .then(|| vector.map(|value| value / magnitude))
}

#[inline]
pub(super) fn rotate(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    let rotation = &pose.rotation;
    [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2],
    ]
}

#[inline]
pub(super) fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    let rotated = rotate(pose, point);
    [
        rotated[0] + pose.translation[0],
        rotated[1] + pose.translation[1],
        rotated[2] + pose.translation[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_perspective_validation_preserves_short_fixed_tracks_and_model_identity() {
        use openmvg_core::camera::{Camera, SphericalCamera};
        let cameras = [
            Camera::try_pinhole(800.0, 600.0, 320.0, 240.0).unwrap(),
            Camera::Spherical(SphericalCamera::try_new(1024).unwrap()),
        ];
        let observations = [PixelObservation::new(0, [320.0, 240.0])];
        let point_offsets = [0, 0, 1];
        let model = ObservationModel::from(MultiViewMeasurements::FixedPerspective {
            observations: &observations,
            point_offsets: &point_offsets,
            cameras: &cameras,
        });
        assert_eq!(model.identity(), ObservationModelIdentity::FixedPerspective);
        assert_eq!(model.intrinsic_dimension(), 0);
        assert_eq!(model.intrinsic_group_count(), 0);
        assert!(model.intrinsic_jacobian(0, &[0.0, 0.0, 1.0]).is_none());
        assert!(
            model
                .validate_with_fixed_points(2, 2, &[true, true])
                .is_ok()
        );
        assert!(matches!(
            model.validate(2, 2),
            Err(
                MultiViewBundleAdjustmentError::InsufficientTrackObservations {
                    point: 0,
                    actual: 0
                }
            )
        ));
        assert!(matches!(
            model.validate_with_fixed_points(2, 2, &[true, false]),
            Err(
                MultiViewBundleAdjustmentError::InsufficientTrackObservations {
                    point: 1,
                    actual: 1
                }
            )
        ));
        assert!(matches!(
            model.validate_with_fixed_points(2, 2, &[true]),
            Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch { .. })
        ));
        let wide_observation = [PixelObservation::new(1, [0.0, 0.0])];
        let observed_wide = ObservationModel::from(MultiViewMeasurements::FixedPerspective {
            observations: &wide_observation,
            point_offsets: &point_offsets,
            cameras: &cameras,
        });
        assert!(matches!(
            observed_wide.validate_with_fixed_points(2, 2, &[true, true]),
            Err(MultiViewBundleAdjustmentError::InvalidIntrinsics { camera: 1 })
        ));
        let duplicates = [observations[0], observations[0]];
        let duplicate_offsets = [0, 2];
        let duplicate = ObservationModel::from(MultiViewMeasurements::FixedPerspective {
            observations: &duplicates,
            point_offsets: &duplicate_offsets,
            cameras: &cameras,
        });
        assert!(duplicate.validate_with_fixed_points(2, 1, &[true]).is_ok());
        assert_eq!(
            duplicate.observation_count(),
            2,
            "independent repeated rows remain represented"
        );
    }

    #[test]
    fn calibrated_models_preserve_measurement_topology() {
        let angular_observations = [
            AngularObservation::new(2, [0.0, 0.0, 1.0]),
            AngularObservation::new(0, [0.0, 1.0, 1.0]),
        ];
        let pixel_observations = [
            PixelObservation::new(1, [3.0, 4.0]),
            PixelObservation::new(2, [5.0, 6.0]),
        ];
        let point_offsets = [0, 2];
        let cameras = [PinholeCamera::new(1.0, 1.0, 0.0, 0.0); 3];
        let angular_measurements = MultiViewMeasurements::Angular {
            observations: &angular_observations,
            point_offsets: &point_offsets,
        };
        let pixel_measurements = MultiViewMeasurements::Pixel {
            observations: &pixel_observations,
            point_offsets: &point_offsets,
            cameras: &cameras,
        };

        for (measurements, expected_cameras) in
            [(angular_measurements, [2, 0]), (pixel_measurements, [1, 2])]
        {
            let model = ObservationModel::calibrated(measurements);
            assert_eq!(model.observation_count(), 2);
            assert_eq!(model.offsets(), point_offsets);
            assert_eq!(measurements.point_offsets(), point_offsets);
            for (observation, expected_camera) in expected_cameras.into_iter().enumerate() {
                assert_eq!(model.camera(observation), expected_camera);
                assert_eq!(
                    measurements.observation_camera(observation),
                    expected_camera
                );
            }
        }
    }

    #[test]
    fn bearing_and_depth_rules_are_owned_by_the_model() {
        let angular_observations = [AngularObservation::new(0, [0.0, 3.0, 4.0])];
        let pixel_observations = [PixelObservation::new(0, [17.0, 28.0])];
        let point_offsets = [0, 1];
        let cameras = [PinholeCamera::new(2.0, 4.0, 10.0, 20.0).with_skew(0.5)];
        let angular = ObservationModel::calibrated(MultiViewMeasurements::Angular {
            observations: &angular_observations,
            point_offsets: &point_offsets,
        });
        let pixel = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
            observations: &pixel_observations,
            point_offsets: &point_offsets,
            cameras: &cameras,
        });

        assert_eq!(angular.observation_bearing(0), Some([0.0, 0.6, 0.8]));
        let pixel_bearing = pixel.observation_bearing(0).unwrap();
        let magnitude = 14.0_f64.sqrt();
        assert_eq!(
            pixel_bearing,
            [3.0 / magnitude, 2.0 / magnitude, 1.0 / magnitude]
        );
        assert_eq!(angular.positive_depth(0, &[0.0, 3.0, 4.0]), Some(true));
        assert_eq!(angular.positive_depth(0, &[0.0, -3.0, -4.0]), Some(false));
        assert_eq!(pixel.positive_depth(0, &[0.0, 0.0, 1.0]), Some(true));
        assert_eq!(pixel.positive_depth(0, &[0.0, 0.0, 0.0]), Some(false));
    }

    #[test]
    fn signed_radial_depth_is_explicit_and_keeps_the_algebraic_projection() {
        let observations = [PixelObservation::new(0, [17.0, -23.0])];
        let offsets = [0, 1];
        let cameras = [RadialCamera::new(730.0, 0.08, -0.025)];
        let positive = ObservationModel::radial(&observations, &offsets, &cameras);
        let signed = ObservationModel::radial_signed_depth(&observations, &offsets, &cameras);
        let front = [0.71, -0.43, 2.4];
        let behind = front.map(|value| -value);

        assert!(positive.project(0, &behind).is_none());
        let front_projection = signed.project(0, &front).unwrap();
        let behind_projection = signed.project(0, &behind).unwrap();
        assert_eq!(front_projection.residual, behind_projection.residual);
        assert_eq!(signed.positive_depth(0, &behind), Some(true));
        assert_eq!(signed.positive_depth(0, &[0.0; 3]), Some(false));
    }

    #[test]
    fn radial_constructors_keep_intrinsic_topology_private() {
        let observations = [PixelObservation::new(1, [0.0; 2])];
        let point_offsets = [0, 1];
        let cameras = [
            RadialCamera::new(400.0, 0.01, -0.001),
            RadialCamera::new(500.0, 0.02, -0.002),
        ];
        let camera_to_group = [1, 0, 1];
        let radial = ObservationModel::radial(&observations, &point_offsets, &cameras);
        let grouped = ObservationModel::radial_grouped(
            &observations,
            &point_offsets,
            &cameras,
            &camera_to_group,
        );

        assert_eq!(radial.intrinsic_group(1), Some(1));
        assert_eq!(grouped.intrinsic_group(1), Some(0));
        assert_eq!(grouped.intrinsic_value(0), Some(cameras[0]));
        assert_eq!(grouped.intrinsic_group_count(), 2);
        assert_eq!(grouped.observation_bearing(0), Some([0.0, 0.0, 1.0]));
        assert_eq!(grouped.positive_depth(0, &[0.0, 0.0, 1.0]), Some(true));
    }

    #[test]
    fn radial_model_counts_measurements_outside_the_central_inverse_branch() {
        let observations = [
            PixelObservation::new(0, [0.1, 0.0]),
            PixelObservation::new(0, [0.5, 0.0]),
        ];
        let offsets = [0, 2];
        let cameras = [RadialCamera::new(1.0, -1.0, 0.0)];
        let radial = ObservationModel::radial(&observations, &offsets, &cameras);

        assert!(radial.observation_bearing(0).is_some());
        assert!(radial.observation_bearing(1).is_none());
        assert_eq!(radial.noninvertible_measurement_count(), 1);
    }

    #[test]
    fn calibrated_models_do_not_run_inverse_domain_diagnostics() {
        let observations = [PixelObservation::new(0, [1.0, 0.0])];
        let offsets = [0, 1];
        let cameras = [PinholeCamera::new(f64::MIN_POSITIVE, 1.0, 0.0, 0.0)];
        let pixel = ObservationModel::calibrated(MultiViewMeasurements::Pixel {
            observations: &observations,
            point_offsets: &offsets,
            cameras: &cameras,
        });

        assert!(pixel.observation_bearing(0).is_none());
        assert_eq!(pixel.noninvertible_measurement_count(), 0);
    }
    #[test]
    fn indexed_fixed_models_select_calibration_without_duplicating_pose() {
        let cameras = [
            openmvg_core::camera::Camera::try_pinhole(500.0, 510.0, 10.0, 20.0).unwrap(),
            openmvg_core::camera::Camera::try_pinhole(730.0, 680.0, 30.0, 40.0).unwrap(),
        ];
        let observations = [PixelObservation::new(0, [0.0; 2]); 2];
        let selectors = [0, 1];
        let model = ObservationModel::from(MultiViewMeasurements::FixedPerspectiveIndexed {
            observations: &observations,
            point_offsets: &[0, 2],
            cameras: &cameras,
            observation_intrinsics: &selectors,
        });
        assert_eq!(
            model.identity(),
            ObservationModelIdentity::FixedPerspectiveIndexed
        );
        assert!(model.validate_with_fixed_points(1, 1, &[true]).is_ok());
        let point = [0.2, 0.1, 1.0];
        let first = model.project(0, &point).unwrap().residual;
        let second = model.project(1, &point).unwrap().residual;
        assert!((first[0] - 110.0).abs() < 1.0e-12 && (first[1] - 71.0).abs() < 1.0e-12);
        assert!((second[0] - 176.0).abs() < 1.0e-12 && (second[1] - 108.0).abs() < 1.0e-12);
        let missing = ObservationModel::from(MultiViewMeasurements::FixedPerspectiveIndexed {
            observations: &observations,
            point_offsets: &[0, 2],
            cameras: &cameras,
            observation_intrinsics: &[0],
        });
        assert!(matches!(
            missing.validate_with_fixed_points(1, 1, &[true]),
            Err(MultiViewBundleAdjustmentError::ObservationIntrinsicCountMismatch { .. })
        ));
        let invalid = ObservationModel::from(MultiViewMeasurements::FixedPerspectiveIndexed {
            observations: &observations,
            point_offsets: &[0, 2],
            cameras: &cameras,
            observation_intrinsics: &[0, 2],
        });
        assert!(matches!(
            invalid.validate_with_fixed_points(1, 1, &[true]),
            Err(
                MultiViewBundleAdjustmentError::InvalidObservationIntrinsic {
                    observation: 1,
                    intrinsic: 2,
                    ..
                }
            )
        ));
    }
}
