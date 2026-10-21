// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observability preflight of translation-only problems on the sparse path's prepared frame.

use super::calibrated::PreparedCalibratedProblem;
use super::{NativePreflight, NativePreflightStatus, native_loss};
use crate::kernel::multi_view::ScaleGauge;
use crate::kernel::{
    FixedRotationGauge, FixedRotationIntrinsics, FixedRotationObservabilityProbeOptions,
    FixedRotationObservation, FixedRotationProbeReason, FixedRotationProbeStatus,
    FixedRotationProblem, probe_fixed_rotation_observability,
};
use crate::{BundleProblem, Gauge, SolveError};
use nalgebra::{Matrix3, Vector3};
use openmvg_core::camera::Camera;

fn fixed_rotation_intrinsics(camera: &Camera) -> Result<FixedRotationIntrinsics, SolveError> {
    match camera {
        Camera::Pinhole(camera) => Ok(FixedRotationIntrinsics {
            focal_x: camera.focal_x(),
            focal_y: camera.focal_y(),
            principal_x: camera.principal_x(),
            principal_y: camera.principal_y(),
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }),
        Camera::BrownConrady(camera) => Ok(FixedRotationIntrinsics {
            focal_x: camera.focal_x(),
            focal_y: camera.focal_y(),
            principal_x: camera.principal_x(),
            principal_y: camera.principal_y(),
            k1: camera.radial_distortion().k1(),
            k2: camera.radial_distortion().k2(),
            k3: camera.radial_distortion().k3(),
            p1: camera.tangential_distortion().p1(),
            p2: camera.tangential_distortion().p2(),
        }),
        Camera::Fisheye(_) | Camera::Spherical(_) => Err(SolveError::Unsupported(
            "fixed-rotation pixel BA supports perspective cameras only".to_owned(),
        )),
    }
}

pub(super) fn probe_prepared(
    problem: &BundleProblem,
    prepared: &PreparedCalibratedProblem,
) -> Result<NativePreflight, SolveError> {
    if prepared.fixed_points.iter().any(|fixed| *fixed) {
        return Err(SolveError::Unsupported(
            "fixed-rotation observability with fixed point blocks is not implemented".to_owned(),
        ));
    }
    let rotations = prepared
        .poses
        .iter()
        .map(|pose| pose.rotation)
        .collect::<Vec<_>>();
    let centers = prepared
        .poses
        .iter()
        .map(|pose| {
            let rotation = Matrix3::from_row_slice(&[
                pose.rotation[0][0],
                pose.rotation[0][1],
                pose.rotation[0][2],
                pose.rotation[1][0],
                pose.rotation[1][1],
                pose.rotation[1][2],
                pose.rotation[2][0],
                pose.rotation[2][1],
                pose.rotation[2][2],
            ]);
            let translation = Vector3::from(pose.translation);
            (-rotation.transpose() * translation).into()
        })
        .collect::<Vec<[f64; 3]>>();
    if !prepared.observation_intrinsics.is_empty() {
        return Err(SolveError::Unsupported(
            "fixed-rotation observability needs one camera model per pose".to_owned(),
        ));
    }
    // A distorted model comes as one full camera per pose; a pinhole model as its matrix.
    let intrinsics = if prepared.fixed_perspective_cameras.is_empty() {
        prepared
            .cameras
            .iter()
            .map(|camera| FixedRotationIntrinsics {
                focal_x: camera.fx,
                focal_y: camera.fy,
                principal_x: camera.cx,
                principal_y: camera.cy,
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
            })
            .collect::<Vec<_>>()
    } else {
        prepared
            .fixed_perspective_cameras
            .iter()
            .map(fixed_rotation_intrinsics)
            .collect::<Result<Vec<_>, _>>()?
    };
    let observations = prepared
        .point_offsets
        .windows(2)
        .map(|range| {
            prepared.observations[range[0]..range[1]]
                .iter()
                .map(|observation| FixedRotationObservation {
                    camera: observation.camera,
                    xy: observation.point,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scale_camera = match prepared.gauge {
        ScaleGauge::FixedBaseline { camera }
        | ScaleGauge::FixedTranslationComponent { camera, .. } => camera,
        ScaleGauge::Free => {
            return Err(SolveError::Unsupported(
                "fixed-rotation observability requires an explicit scale coordinate".to_owned(),
            ));
        }
    };
    let gauge = match problem.gauge() {
        Gauge::FixedBaseline { .. } => FixedRotationGauge::FixedBaseline {
            anchor: 0,
            scale_camera,
            distance: (Vector3::from(centers[scale_camera]) - Vector3::from(centers[0])).norm(),
        },
        Gauge::Free => FixedRotationGauge::Free {
            anchor: 0,
            scale_camera,
        },
    };
    let probe = probe_fixed_rotation_observability(
        FixedRotationProblem {
            rotations: &rotations,
            centers: &centers,
            points: &prepared.points,
            intrinsics: &intrinsics,
            observations: &observations,
            fixed_cameras: &prepared.fixed_poses,
            gauge,
        },
        native_loss(problem)?,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let required_covered_free_cameras = prepared
        .fixed_poses
        .iter()
        .enumerate()
        .filter(|(camera, fixed)| *camera != 0 && !**fixed)
        .count();
    native_preflight_from_probe(probe, required_covered_free_cameras)
}

fn native_preflight_from_probe(
    probe: crate::kernel::FixedRotationObservabilityProbe,
    required_covered_free_cameras: usize,
) -> Result<NativePreflight, SolveError> {
    let (status, dense_observability_escalation) = match probe.status {
        FixedRotationProbeStatus::FullRankCertified => (NativePreflightStatus::Proceed, false),
        FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::SparseFactorizationFailed,
        ) if probe.weak_point_count == 0
            && probe.covered_free_cameras >= required_covered_free_cameras =>
        {
            (NativePreflightStatus::Proceed, true)
        }
        FixedRotationProbeStatus::NotCertified(reason) => (
            match reason {
                FixedRotationProbeReason::InsufficientCameraCoverage => {
                    NativePreflightStatus::InsufficientCameraCoverage
                }
                FixedRotationProbeReason::ResourceLimit => NativePreflightStatus::ResourceLimit,
                FixedRotationProbeReason::WeakPointGeometry => {
                    NativePreflightStatus::WeakPointGeometry
                }
                FixedRotationProbeReason::SparseFactorizationFailed => {
                    NativePreflightStatus::ObservabilityNotCertified
                }
            },
            false,
        ),
    };
    Ok(NativePreflight {
        status,
        pose_dimension: Some(probe.pose_dimension),
        covered_free_pose_count: Some(probe.covered_free_cameras),
        weak_point_count: Some(probe.weak_point_count),
        dense_observability_escalation,
    })
}
