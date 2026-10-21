// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in covariance for the centered radial self-calibration model.

use super::{RadialCamera, SelfCalibratingMeasurements, SelfCalibrationWorkspace};
use crate::kernel::multi_view::covariance::{covariance_for_grouped_model, covariance_for_model};
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::{
    RelativePose, multi_view::ScaleGauge, multi_view::dense::MultiViewBundleAdjustmentCovariance,
    multi_view::dense::MultiViewBundleAdjustmentCovarianceError,
};
use crate::numerics::refine::{Loss, NoiseModel};

/// Compute camera-intrinsic, pose, and point marginals after fixed-baseline
/// self-calibrating bundle adjustment.
///
/// This opt-in pass allocates its returned matrices. Camera-local coordinates
/// are documented by [`MultiViewBundleAdjustmentCovariance`]. A free scale
/// gauge is rejected because its ordinary covariance is not unique.
#[allow(clippy::too_many_arguments)]
pub fn covariance_ba_multi_view_self_calibrating_dense(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    cameras: &[RadialCamera],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut SelfCalibrationWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    let model = ObservationModel::radial(
        measurements.observations,
        measurements.point_offsets,
        cameras,
    );
    covariance_for_model(
        poses,
        points,
        model,
        gauge,
        &mut workspace.dense,
        loss,
        noise,
        3,
    )
}

/// Compute pose, shared-intrinsic-group, and point marginals after grouped
/// fixed-baseline self-calibration.
///
/// Group ranges are available through
/// [`MultiViewBundleAdjustmentCovariance::intrinsic_group_parameter_range`].
/// This opt-in pass allocates its returned matrices; the corresponding solve
/// remains allocation-free.
#[allow(clippy::too_many_arguments)]
pub fn covariance_ba_multi_view_self_calibrating_grouped_dense(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    intrinsic_groups: &[RadialCamera],
    camera_to_group: &[usize],
    measurements: SelfCalibratingMeasurements<'_>,
    gauge: ScaleGauge,
    workspace: &mut SelfCalibrationWorkspace,
    loss: Loss,
    noise: NoiseModel<'_>,
) -> Result<MultiViewBundleAdjustmentCovariance, MultiViewBundleAdjustmentCovarianceError> {
    covariance_for_grouped_model(
        poses,
        points,
        measurements.observations,
        measurements.point_offsets,
        intrinsic_groups,
        camera_to_group,
        gauge,
        &mut workspace.dense,
        loss,
        noise,
    )
}
