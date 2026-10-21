// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pose-variable masks and the gauge anchor and scale choice.

use crate::kernel::multi_view::{CameraPoseVariables, ScaleGauge};
use crate::{BundleProblem, Gauge, PoseVariables, SolveError};

pub(super) fn common_pose_variables(
    problem: &BundleProblem,
) -> Result<CameraPoseVariables, SolveError> {
    let mut selected = None;
    let observed = observed_cameras(problem);
    for variables in problem
        .poses()
        .iter()
        .zip(&observed)
        .filter(|(camera, observed)| **observed && !camera.variables().is_fixed())
        .map(|(camera, _)| camera.variables())
    {
        if !variables.translation() {
            return Err(SolveError::Unsupported(
                "direct native lowering cannot optimize rotation without translation".to_owned(),
            ));
        }
        let candidate = if variables.rotation() {
            CameraPoseVariables::RotationAndTranslation
        } else {
            CameraPoseVariables::TranslationOnly
        };
        if selected.is_some_and(|current| current != candidate) {
            return Err(SolveError::Unsupported(
                "direct native lowering requires one pose-variable mask for active cameras"
                    .to_owned(),
            ));
        }
        selected = Some(candidate);
    }
    Ok(selected.unwrap_or(CameraPoseVariables::RotationAndTranslation))
}

/// Cameras seen by at least one observation.
///
/// An unobserved camera does not enter the objective, so its pose says
/// nothing about the scene: it must never anchor the gauge or set its scale.
/// A camera whose observations were all rejected can sit arbitrarily far away,
/// and as the scale camera its distance would set the translation scaling.
pub(super) fn observed_cameras(problem: &BundleProblem) -> Vec<bool> {
    let mut observed = vec![false; problem.poses().len()];
    for observation in problem.observations() {
        observed[observation.pose()] = true;
    }
    observed
}

/// The first fixed observed camera, else the first observed camera.
pub(super) fn free_scale_anchor(fixed: impl Iterator<Item = bool>, observed: &[bool]) -> usize {
    let mut first_observed = None;
    for (camera, (fixed, &observed)) in fixed.zip(observed).enumerate() {
        if observed && fixed {
            return camera;
        }
        if observed && first_observed.is_none() {
            first_observed = Some(camera);
        }
    }
    first_observed.unwrap_or(0)
}

/// The free-scale gauge camera: the candidate farthest from the anchor,
/// ties broken towards the lower index.
pub(super) fn farthest_camera(candidates: impl Iterator<Item = (usize, f64)>) -> Option<usize> {
    candidates
        .filter(|(_, squared)| squared.is_finite())
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .filter(|(_, squared)| *squared > 0.0)
        .map(|(camera, _)| camera)
}

/// Reject a fixed-world frame that the camera-zero kernel cannot optimize without an explicit anchor.
pub(super) fn validate_anchor_policy(problem: &BundleProblem) -> Result<(), SolveError> {
    if !matches!(problem.gauge(), Gauge::Free)
        || !problem
            .observations()
            .iter()
            .any(|observation| problem.points()[observation.point()].is_fixed())
    {
        return Ok(());
    }
    let observed = observed_cameras(problem);
    if problem
        .poses()
        .iter()
        .enumerate()
        .any(|(index, pose)| observed[index] && pose.variables().is_fixed())
    {
        return Ok(());
    }
    Err(SolveError::Unsupported(
        "native BA with fixed world points requires an explicitly fixed observed pose or fixed-baseline anchor"
            .to_owned(),
    ))
}

pub(super) struct GaugePlan {
    pub(super) anchor: usize,
    pub(super) scale: f64,
    pub(super) native: ScaleGauge,
}

/// A fixed reference frame and the coordinates left after justified gauge choices.
pub(super) struct CalibratedGaugePlan {
    pub(super) anchor: usize,
    pub(super) scale: f64,
    pub(super) native: ScaleGauge,
    pub(super) pose_masks: Vec<PoseVariables>,
}

pub(super) fn calibrated_gauge_plan(
    problem: &BundleProblem,
) -> Result<CalibratedGaugePlan, SolveError> {
    let observed = observed_cameras(problem);
    let mut pose_masks = problem
        .poses()
        .iter()
        .map(|pose| pose.variables())
        .collect::<Vec<_>>();
    if matches!(problem.gauge(), Gauge::FixedBaseline { .. }) {
        let plan = gauge_plan(problem)?;
        pose_masks[plan.anchor] = PoseVariables::FIXED;
        return Ok(CalibratedGaugePlan {
            anchor: plan.anchor,
            scale: plan.scale,
            native: plan.native,
            pose_masks,
        });
    }

    let first_observed = observed
        .iter()
        .position(|&value| value)
        .ok_or_else(|| SolveError::Internal("checked bundle has no observed pose".to_owned()))?;
    let fixed_anchor = pose_masks
        .iter()
        .enumerate()
        .find_map(|(camera, mask)| (observed[camera] && mask.is_fixed()).then_some(camera));
    let has_controls = problem
        .observations()
        .iter()
        .any(|observation| problem.points()[observation.point()].is_fixed());

    // Survey controls establish world coordinates through their measurements. A frame
    // conversion remains useful, but it must not additionally constrain camera zero.
    if has_controls && fixed_anchor.is_none() {
        return Ok(CalibratedGaugePlan {
            anchor: first_observed,
            scale: 1.0,
            native: ScaleGauge::Free,
            pose_masks,
        });
    }

    let held_centers = pose_masks
        .iter()
        .enumerate()
        .filter_map(|(camera, mask)| (observed[camera] && !mask.translation()).then_some(camera))
        .collect::<Vec<_>>();
    let known_rotation = pose_masks
        .iter()
        .enumerate()
        .find_map(|(camera, mask)| (observed[camera] && !mask.rotation()).then_some(camera));
    let anchor = fixed_anchor
        .or_else(|| held_centers.first().copied())
        .or(known_rotation)
        .unwrap_or(first_observed);
    let anchor_center = problem.poses()[anchor].pose().center();
    let distinct_held_center = held_centers.iter().any(|&camera| {
        (problem.poses()[camera].pose().center() - anchor_center).norm_squared() > 0.0
    });

    if !has_controls && held_centers.is_empty() {
        // Without held centers, translation is a symmetry. If a rotation is known,
        // use that pose as the translation reference without locking another rotation.
        // Otherwise the full reference pose removes the six rigid-frame freedoms.
        pose_masks[anchor] = PoseVariables::FIXED;
    } else if !has_controls && known_rotation.is_none() && !distinct_held_center {
        // All held centers coincide, so global rotation about that center is still
        // a symmetry. Distinct centers or known orientations invalidate this choice.
        pose_masks[anchor] = PoseVariables::FIXED;
    }

    let control_sets_scale = problem.observations().iter().any(|observation| {
        let point = &problem.points()[observation.point()];
        point.is_fixed() && (point.position() - anchor_center).norm_squared() > 0.0
    });
    let native = if distinct_held_center || control_sets_scale {
        ScaleGauge::Free
    } else {
        let camera = farthest_camera(
            problem
                .poses()
                .iter()
                .enumerate()
                .filter(|(camera, _)| {
                    *camera != anchor && observed[*camera] && pose_masks[*camera].translation()
                })
                .map(|(camera, pose)| {
                    (
                        camera,
                        (pose.pose().center() - anchor_center).norm_squared(),
                    )
                }),
        )
        .ok_or_else(|| {
            SolveError::Unsupported("free-scale BA has no nonzero active baseline".to_owned())
        })?;
        ScaleGauge::FixedBaseline { camera }
    };
    Ok(CalibratedGaugePlan {
        anchor,
        scale: 1.0,
        native,
        pose_masks,
    })
}

pub(super) fn gauge_plan(problem: &BundleProblem) -> Result<GaugePlan, SolveError> {
    validate_anchor_policy(problem)?;
    match problem.gauge() {
        Gauge::Free => {
            let observed = observed_cameras(problem);
            let anchor = free_scale_anchor(
                problem
                    .poses()
                    .iter()
                    .map(|camera| camera.variables().is_fixed()),
                &observed,
            );
            let anchor_center = problem.poses()[anchor].pose().center();
            let fixed_pose_sets_scale =
                problem.poses().iter().enumerate().any(|(index, camera)| {
                    index != anchor
                        && observed[index]
                        && camera.variables().is_fixed()
                        && (camera.pose().center() - anchor_center).norm_squared() > 0.0
                });
            let fixed_point_sets_scale = problem.observations().iter().any(|observation| {
                let point = &problem.points()[observation.point()];
                point.is_fixed() && (point.position() - anchor_center).norm_squared() > 0.0
            });
            let native = if fixed_pose_sets_scale || fixed_point_sets_scale {
                ScaleGauge::Free
            } else {
                let camera = farthest_camera(
                    problem
                        .poses()
                        .iter()
                        .enumerate()
                        .filter(|(index, camera)| {
                            *index != anchor && observed[*index] && !camera.variables().is_fixed()
                        })
                        .map(|(index, camera)| {
                            (
                                index,
                                (camera.pose().center() - anchor_center).norm_squared(),
                            )
                        }),
                )
                .ok_or_else(|| {
                    SolveError::Unsupported(
                        "free-scale BA has no nonzero active baseline".to_owned(),
                    )
                })?;
                ScaleGauge::FixedBaseline { camera }
            };
            Ok(GaugePlan {
                anchor,
                scale: 1.0,
                native,
            })
        }
        Gauge::FixedBaseline {
            first,
            second,
            distance,
        } => {
            let current = (problem.poses()[second].pose().center()
                - problem.poses()[first].pose().center())
            .norm();
            if !current.is_finite() || current <= 0.0 {
                return Err(SolveError::Unsupported(
                    "fixed-baseline cameras have no usable separation".to_owned(),
                ));
            }
            let raw_scale = distance / current;
            let tolerance = 256.0 * f64::EPSILON * current.abs().max(distance.abs()).max(1.0);
            let scale = if (current - distance).abs() <= tolerance {
                1.0
            } else {
                raw_scale
            };
            if scale != 1.0
                && (problem
                    .poses()
                    .iter()
                    .enumerate()
                    .any(|(index, camera)| index != first && !camera.variables().translation())
                    || problem.points().iter().any(|point| point.is_fixed()))
            {
                return Err(SolveError::Unsupported(
                    "fixed-baseline scaling would move fixed state".to_owned(),
                ));
            }
            let native = if !problem.poses()[second].variables().translation() {
                ScaleGauge::Free
            } else {
                ScaleGauge::FixedBaseline { camera: second }
            };
            Ok(GaugePlan {
                anchor: first,
                scale,
                native,
            })
        }
    }
}
