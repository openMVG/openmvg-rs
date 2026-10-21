// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural and numerical input checks kept outside LM control flow.

use crate::numerics::manifold::Rotation3;

use super::super::RelativePose;
#[cfg(any(test, feature = "unstable-internals"))]
use super::dense::DenseBundleAdjustmentWorkspace;
use super::model::{ObservationModel, transform};
use super::{
    BundleAdjustmentBlock, BundleAdjustmentPriors, FixedVariableBlocks,
    MultiViewBundleAdjustmentError, ScaleGauge,
};

/// Two views are a complete bundle once the gauge anchors one and fixes the baseline: an initial
/// pair whose camera model the pinhole two-view lowering cannot represent is solved here.
const MINIMUM_CAMERAS: usize = 2;
const MINIMUM_POINTS: usize = 4;

#[cfg(any(test, feature = "unstable-internals"))]
pub(super) fn problem(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    workspace: &DenseBundleAdjustmentWorkspace,
) -> Result<(), MultiViewBundleAdjustmentError> {
    problem_with_capacity(
        poses,
        points,
        model,
        gauge,
        workspace.camera_capacity,
        workspace.point_capacity,
    )
}

pub(super) fn problem_with_capacity(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    camera_capacity: usize,
    point_capacity: usize,
) -> Result<(), MultiViewBundleAdjustmentError> {
    camera_problem(poses, points.len(), model, gauge)?;
    if poses.len() > camera_capacity {
        return Err(MultiViewBundleAdjustmentError::WorkspaceCameraCapacity {
            required: poses.len(),
            capacity: camera_capacity,
        });
    }
    if points.len() > point_capacity {
        return Err(MultiViewBundleAdjustmentError::WorkspacePointCapacity {
            required: points.len(),
            capacity: point_capacity,
        });
    }
    for (point, value) in points.iter().enumerate() {
        if let Some(coordinate) = value.iter().position(|coordinate| !coordinate.is_finite()) {
            return Err(MultiViewBundleAdjustmentError::NonFinitePoint { point, coordinate });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn problem_with_capacity_and_fixed(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    camera_capacity: usize,
    point_capacity: usize,
    fixed_points: &[bool],
    legacy_anchor: bool,
) -> Result<(), MultiViewBundleAdjustmentError> {
    camera_problem_with_fixed(
        poses,
        points.len(),
        model,
        gauge,
        fixed_points,
        legacy_anchor,
    )?;
    if poses.len() > camera_capacity {
        return Err(MultiViewBundleAdjustmentError::WorkspaceCameraCapacity {
            required: poses.len(),
            capacity: camera_capacity,
        });
    }
    if points.len() > point_capacity {
        return Err(MultiViewBundleAdjustmentError::WorkspacePointCapacity {
            required: points.len(),
            capacity: point_capacity,
        });
    }
    for (point, value) in points.iter().enumerate() {
        if let Some(coordinate) = value.iter().position(|coordinate| !coordinate.is_finite()) {
            return Err(MultiViewBundleAdjustmentError::NonFinitePoint { point, coordinate });
        }
    }
    Ok(())
}

pub(super) fn camera_problem(
    poses: &[RelativePose],
    point_count: usize,
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
) -> Result<(), MultiViewBundleAdjustmentError> {
    camera_problem_with_fixed(poses, point_count, model, gauge, &[], true)
}

fn camera_problem_with_fixed(
    poses: &[RelativePose],
    point_count: usize,
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    fixed_points: &[bool],
    legacy_anchor: bool,
) -> Result<(), MultiViewBundleAdjustmentError> {
    topology_with_fixed_points(
        poses.len(),
        point_count,
        model,
        gauge,
        fixed_points,
        legacy_anchor,
    )?;
    validate_poses(poses)?;
    if legacy_anchor && !anchor_is_identity(&poses[0]) {
        return Err(MultiViewBundleAdjustmentError::AnchorCameraNotIdentity);
    }
    if !legacy_anchor
        && !matches!(gauge, ScaleGauge::Free)
        && poses[0]
            .translation
            .iter()
            .any(|&coordinate| coordinate != 0.0)
    {
        return Err(MultiViewBundleAdjustmentError::AnchorCameraNotIdentity);
    }
    match gauge {
        ScaleGauge::Free => {}
        ScaleGauge::FixedBaseline { camera } => {
            let baseline = norm(&poses[camera].translation);
            if !(baseline.is_finite() && baseline > 64.0 * f64::EPSILON) {
                return Err(MultiViewBundleAdjustmentError::InvalidGaugeBaseline { camera });
            }
        }
        ScaleGauge::FixedTranslationComponent { camera, component } => {
            let value = poses[camera].translation[component.index()];
            if !(value.is_finite() && value.abs() > 64.0 * f64::EPSILON) {
                return Err(
                    MultiViewBundleAdjustmentError::InvalidGaugeTranslationComponent {
                        camera,
                        component,
                    },
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "unstable-internals")]
pub(super) fn triangulation_problem(
    poses: &[RelativePose],
    point_count: usize,
    model: ObservationModel<'_>,
) -> Result<(), MultiViewBundleAdjustmentError> {
    const MINIMUM_TRIANGULATION_CAMERAS: usize = 2;
    const MINIMUM_TRIANGULATION_POINTS: usize = 1;

    if poses.len() < MINIMUM_TRIANGULATION_CAMERAS {
        return Err(MultiViewBundleAdjustmentError::InsufficientCameras {
            required: MINIMUM_TRIANGULATION_CAMERAS,
            actual: poses.len(),
        });
    }
    if point_count < MINIMUM_TRIANGULATION_POINTS {
        return Err(MultiViewBundleAdjustmentError::InsufficientPoints {
            required: MINIMUM_TRIANGULATION_POINTS,
            actual: point_count,
        });
    }
    model.validate(poses.len(), point_count)?;
    validate_poses(poses)
}

fn validate_poses(poses: &[RelativePose]) -> Result<(), MultiViewBundleAdjustmentError> {
    for (camera, pose) in poses.iter().enumerate() {
        if Rotation3::try_from_matrix(pose.rotation).is_err() {
            return Err(MultiViewBundleAdjustmentError::InvalidRotation { camera });
        }
        if pose.translation.iter().any(|value| !value.is_finite()) {
            return Err(MultiViewBundleAdjustmentError::NonFiniteTranslation { camera });
        }
    }
    Ok(())
}

pub(super) fn topology(
    camera_count: usize,
    point_count: usize,
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
) -> Result<(), MultiViewBundleAdjustmentError> {
    topology_with_fixed_points(camera_count, point_count, model, gauge, &[], true)
}

pub(super) fn topology_with_fixed_points(
    camera_count: usize,
    point_count: usize,
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    fixed_points: &[bool],
    legacy_minima: bool,
) -> Result<(), MultiViewBundleAdjustmentError> {
    // Explicit masks can describe ordinary control-point resection or a single
    // triangulated free point. Their track information and final rank decide
    // observability; only nonempty pose/point collections are structural limits.
    let minimum_cameras = if legacy_minima { MINIMUM_CAMERAS } else { 1 };
    let minimum_points = if legacy_minima { MINIMUM_POINTS } else { 1 };
    if camera_count < minimum_cameras {
        return Err(MultiViewBundleAdjustmentError::InsufficientCameras {
            required: minimum_cameras,
            actual: camera_count,
        });
    }
    if point_count < minimum_points {
        return Err(MultiViewBundleAdjustmentError::InsufficientPoints {
            required: minimum_points,
            actual: point_count,
        });
    }
    match gauge {
        ScaleGauge::Free => {}
        ScaleGauge::FixedBaseline { camera }
        | ScaleGauge::FixedTranslationComponent { camera, .. } => {
            if camera == 0 || camera >= camera_count {
                return Err(MultiViewBundleAdjustmentError::InvalidGaugeCamera { camera });
            }
        }
    }
    model.validate_with_fixed_points(camera_count, point_count, fixed_points)
}

pub(super) fn projection_domain(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
) -> Result<(), MultiViewBundleAdjustmentError> {
    for (point_index, point) in points.iter().enumerate() {
        let track = model.offsets()[point_index]..model.offsets()[point_index + 1];
        for observation in track {
            let camera = model.camera(observation);
            let camera_point = transform(&poses[camera], point);
            if model.project(observation, &camera_point).is_none() {
                return Err(MultiViewBundleAdjustmentError::PointNotProjectable {
                    point: point_index,
                    camera,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn factors(
    poses: &[RelativePose],
    points: &[[f64; 3]],
    model: ObservationModel<'_>,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
) -> Result<(), MultiViewBundleAdjustmentError> {
    fixed_topology(
        poses.len(),
        points.len(),
        model.intrinsic_group_count(),
        gauge,
        fixed,
    )?;

    for (position, prior) in priors.poses.iter().enumerate() {
        if prior.camera == 0
            || prior.camera >= poses.len()
            || is_fixed(fixed.camera_poses, prior.camera)
        {
            return Err(MultiViewBundleAdjustmentError::InvalidPriorTarget {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                index: prior.camera,
            });
        }
        if position > 0 && priors.poses[position - 1].camera >= prior.camera {
            return Err(MultiViewBundleAdjustmentError::PriorOrder {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                previous: priors.poses[position - 1].camera,
                current: prior.camera,
            });
        }
        if Rotation3::try_from_matrix(prior.mean.rotation).is_err()
            || prior
                .mean
                .translation
                .iter()
                .any(|value| !value.is_finite())
            || !finite_matrix(&prior.sqrt_information)
        {
            return Err(MultiViewBundleAdjustmentError::InvalidPrior {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                index: prior.camera,
            });
        }
    }
    for (position, prior) in priors.points.iter().enumerate() {
        if prior.point >= points.len() || is_fixed(fixed.points, prior.point) {
            return Err(MultiViewBundleAdjustmentError::InvalidPriorTarget {
                block: BundleAdjustmentBlock::Point,
                index: prior.point,
            });
        }
        if position > 0 && priors.points[position - 1].point >= prior.point {
            return Err(MultiViewBundleAdjustmentError::PriorOrder {
                block: BundleAdjustmentBlock::Point,
                previous: priors.points[position - 1].point,
                current: prior.point,
            });
        }
        if prior.mean.iter().any(|value| !value.is_finite())
            || !finite_matrix(&prior.sqrt_information)
        {
            return Err(MultiViewBundleAdjustmentError::InvalidPrior {
                block: BundleAdjustmentBlock::Point,
                index: prior.point,
            });
        }
    }
    for (position, prior) in priors.intrinsics.iter().enumerate() {
        if prior.intrinsic >= model.intrinsic_group_count()
            || is_fixed(fixed.intrinsics, prior.intrinsic)
        {
            return Err(MultiViewBundleAdjustmentError::InvalidPriorTarget {
                block: BundleAdjustmentBlock::RadialIntrinsics,
                index: prior.intrinsic,
            });
        }
        if position > 0 && priors.intrinsics[position - 1].intrinsic >= prior.intrinsic {
            return Err(MultiViewBundleAdjustmentError::PriorOrder {
                block: BundleAdjustmentBlock::RadialIntrinsics,
                previous: priors.intrinsics[position - 1].intrinsic,
                current: prior.intrinsic,
            });
        }
        let valid = prior.focal.is_finite()
            && prior.focal > 0.0
            && prior.k1.is_finite()
            && prior.k2.is_finite()
            && finite_matrix(&prior.sqrt_information);
        if !valid {
            return Err(MultiViewBundleAdjustmentError::InvalidPrior {
                block: BundleAdjustmentBlock::RadialIntrinsics,
                index: prior.intrinsic,
            });
        }
    }
    Ok(())
}

pub(super) fn fixed_topology(
    camera_count: usize,
    point_count: usize,
    intrinsic_count: usize,
    gauge: ScaleGauge,
    fixed: FixedVariableBlocks<'_>,
) -> Result<(), MultiViewBundleAdjustmentError> {
    validate_mask(
        BundleAdjustmentBlock::WorldToCameraPose,
        fixed.camera_poses,
        camera_count,
    )?;
    validate_mask(BundleAdjustmentBlock::Point, fixed.points, point_count)?;
    validate_mask(
        BundleAdjustmentBlock::RadialIntrinsics,
        fixed.intrinsics,
        intrinsic_count,
    )?;
    if !fixed.intrinsic_dimensions.is_empty() && fixed.intrinsic_dimensions.len() != intrinsic_count
    {
        return Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
            block: BundleAdjustmentBlock::RadialIntrinsics,
            expected: intrinsic_count,
            actual: fixed.intrinsic_dimensions.len(),
        });
    }
    if let Some((block, &dimension)) = fixed
        .intrinsic_dimensions
        .iter()
        .enumerate()
        .find(|(_, dimension)| !(1..=3).contains(*dimension))
    {
        return Err(MultiViewBundleAdjustmentError::InvalidIntrinsicDimension { block, dimension });
    }
    let gauge_camera = match gauge {
        ScaleGauge::Free => None,
        ScaleGauge::FixedBaseline { camera }
        | ScaleGauge::FixedTranslationComponent { camera, .. } => Some(camera),
    };
    if let Some(camera) = gauge_camera
        && is_fixed(fixed.camera_poses, camera)
    {
        return Err(MultiViewBundleAdjustmentError::GaugeCameraIsFixed { camera });
    }
    Ok(())
}

fn validate_mask(
    block: BundleAdjustmentBlock,
    mask: &[bool],
    expected: usize,
) -> Result<(), MultiViewBundleAdjustmentError> {
    if !mask.is_empty() && mask.len() != expected {
        return Err(MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
            block,
            expected,
            actual: mask.len(),
        });
    }
    Ok(())
}

fn finite_matrix<const N: usize>(matrix: &[[f64; N]; N]) -> bool {
    matrix.iter().flatten().all(|value| value.is_finite())
}

fn is_fixed(mask: &[bool], index: usize) -> bool {
    !mask.is_empty() && mask[index]
}

fn anchor_is_identity(pose: &RelativePose) -> bool {
    let mut error = pose
        .translation
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    for row in 0..3 {
        for column in 0..3 {
            error = error.max((pose.rotation[row][column] - (row == column) as u8 as f64).abs());
        }
    }
    error <= 64.0 * f64::EPSILON.sqrt()
}

fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::multi_view::{MultiViewMeasurements, PointPrior, PosePrior};

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn factor_validation_reports_typed_masks_targets_and_ordering() {
        let poses = [
            RelativePose::new(IDENTITY, [0.0; 3]),
            RelativePose::new(IDENTITY, [0.7, 0.0, 0.1]),
            RelativePose::new(IDENTITY, [-0.5, 0.1, 0.2]),
        ];
        let points = [[0.0, 0.0, 3.0]; 4];
        let observations = [];
        let point_offsets = [];
        let model = ObservationModel::calibrated(MultiViewMeasurements::Angular {
            observations: &observations,
            point_offsets: &point_offsets,
        });
        let error = factors(
            &poses,
            &points,
            model,
            ScaleGauge::Free,
            FixedVariableBlocks {
                camera_poses: &[false; 2],
                points: &[],
                intrinsics: &[],
                intrinsic_dimensions: &[],
            },
            BundleAdjustmentPriors::NONE,
        )
        .expect_err("short pose mask must be rejected");
        assert_eq!(
            error,
            MultiViewBundleAdjustmentError::FixedBlockCountMismatch {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                expected: 3,
                actual: 2,
            }
        );

        let fixed_poses = [false, true, false];
        let pose_priors = [PosePrior {
            camera: 1,
            mean: poses[1],
            sqrt_information: [[0.0; 6]; 6],
        }];
        let error = factors(
            &poses,
            &points,
            model,
            ScaleGauge::Free,
            FixedVariableBlocks {
                camera_poses: &fixed_poses,
                points: &[],
                intrinsics: &[],
                intrinsic_dimensions: &[],
            },
            BundleAdjustmentPriors {
                poses: &pose_priors,
                points: &[],
                intrinsics: &[],
            },
        )
        .expect_err("a fixed block cannot also receive a prior");
        assert_eq!(
            error,
            MultiViewBundleAdjustmentError::InvalidPriorTarget {
                block: BundleAdjustmentBlock::WorldToCameraPose,
                index: 1,
            }
        );

        let point_priors = [
            PointPrior {
                point: 2,
                mean: points[2],
                sqrt_information: [[0.0; 3]; 3],
            },
            PointPrior {
                point: 1,
                mean: points[1],
                sqrt_information: [[0.0; 3]; 3],
            },
        ];
        let error = factors(
            &poses,
            &points,
            model,
            ScaleGauge::Free,
            FixedVariableBlocks::NONE,
            BundleAdjustmentPriors {
                poses: &[],
                points: &point_priors,
                intrinsics: &[],
            },
        )
        .expect_err("unsorted priors must be rejected");
        assert_eq!(
            error,
            MultiViewBundleAdjustmentError::PriorOrder {
                block: BundleAdjustmentBlock::Point,
                previous: 2,
                current: 1,
            }
        );
    }
}
