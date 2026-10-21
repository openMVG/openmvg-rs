// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Input checks for fixed-rotation problems and probe options.

use super::api::FixedRotationObservabilityProbeOptions;
use super::api::{FixedRotationError, FixedRotationGauge, FixedRotationProblem};

pub(super) fn validate_probe_options(
    options: FixedRotationObservabilityProbeOptions,
) -> Result<(), FixedRotationError> {
    if options.max_tracks == 0
        || options.max_observations < 2
        || options.max_observations_per_track < 2
        || options.max_coverage_observations_per_track < options.max_observations_per_track
        || options.max_reduced_nonzeros == 0
        || options.max_factor_nonzeros == 0
        || options.max_symbolic_dense_nonzeros == 0
        || !(options.minimum_normalized_eigenvalue.is_finite()
            && options.minimum_normalized_eigenvalue > 0.0
            && options.minimum_normalized_eigenvalue < 1.0)
    {
        return Err(FixedRotationError::InvalidOptions(
            "invalid fixed-rotation observability probe limits",
        ));
    }
    Ok(())
}

pub(super) fn validate_problem(
    problem: FixedRotationProblem<'_>,
) -> Result<(), FixedRotationError> {
    let FixedRotationProblem {
        rotations,
        centers,
        points,
        intrinsics,
        observations,
        fixed_cameras,
        gauge,
    } = problem;
    if rotations.len() < 2
        || rotations.len() != centers.len()
        || rotations.len() != intrinsics.len()
        || fixed_cameras.len() != centers.len()
        || points.is_empty()
        || points.len() != observations.len()
    {
        return Err(FixedRotationError::InvalidProblem(
            "fixed-rotation BA dimensions are inconsistent",
        ));
    }
    if rotations
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
        || centers.iter().flatten().any(|value| !value.is_finite())
        || points.iter().flatten().any(|value| !value.is_finite())
        || observations
            .iter()
            .flatten()
            .flat_map(|observation| observation.xy)
            .any(|value| !value.is_finite())
        || intrinsics.iter().any(|camera| {
            [
                camera.focal_x,
                camera.focal_y,
                camera.principal_x,
                camera.principal_y,
                camera.k1,
                camera.k2,
                camera.k3,
                camera.p1,
                camera.p2,
            ]
            .into_iter()
            .any(|value| !value.is_finite())
        })
    {
        return Err(FixedRotationError::InvalidProblem(
            "fixed-rotation BA inputs must be finite",
        ));
    }
    if observations.iter().any(|track| track.len() < 2)
        || observations
            .iter()
            .flatten()
            .any(|observation| observation.camera >= centers.len())
    {
        return Err(FixedRotationError::InvalidProblem(
            "fixed-rotation BA requires valid two-view point tracks",
        ));
    }
    let (anchor, scale_camera, distance) = match gauge {
        FixedRotationGauge::Free {
            anchor,
            scale_camera,
        } => (anchor, scale_camera, None),
        FixedRotationGauge::FixedBaseline {
            anchor,
            scale_camera,
            distance,
        } => (anchor, scale_camera, Some(distance)),
    };
    if anchor == scale_camera || anchor >= centers.len() || scale_camera >= centers.len() {
        return Err(FixedRotationError::InvalidProblem(
            "fixed-rotation gauge endpoints are invalid",
        ));
    }
    if distance.is_some_and(|distance| !(distance.is_finite() && distance > 0.0)) {
        return Err(FixedRotationError::InvalidProblem(
            "fixed baseline must be positive and finite",
        ));
    }
    Ok(())
}
