// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rotation and verified-pose consistency diagnostics.

use super::*;
use openmvg_core::model::rotation_angle_deg;
/// Summary of a non-empty set of angular residuals, in degrees (quantiles interpolate linearly).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AngularResidualDistribution {
    /// Number of finite residuals summarized (at least 1).
    pub count: usize,
    /// Smallest residual, in degrees.
    pub minimum_deg: f64,
    /// Median residual, in degrees.
    pub median_deg: f64,
    /// 90th-percentile residual, in degrees.
    pub p90_deg: f64,
    /// Largest residual, in degrees.
    pub maximum_deg: f64,
}

/// Which verified pair-graph edges to evaluate against the current metric poses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifiedPairSelection {
    /// Only edges that are active in the pair graph.
    Active,
    /// Only edges deactivated by view-graph conditioning (held-out evidence).
    Inactive,
    /// Every verified edge, active or not.
    All,
}

/// Solver-independent agreement of current poses with a selected verified-edge set.
///
/// `Inactive` is useful as held-out evidence after view-graph conditioning. It is not fully
/// independent—the edges came from the same frontend—but it prevents a conditioned subgraph from
/// validating only against the measurements it retained.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerifiedPoseGraphConsistencyReport {
    /// Edge set the residuals were computed on.
    pub selection: VerifiedPairSelection,
    /// Pair-graph edges in the selected set.
    pub selected_pair_count: usize,
    /// Selected edges whose verified model carries a relative pose.
    pub relative_pose_pair_count: usize,
    /// Selected relative-pose edges whose two cameras are currently posed.
    pub posed_pair_count: usize,
    /// Posed edges whose solved baseline points along the measured direction (positive cosine).
    pub positive_scale_count: usize,
    /// `positive_scale_count` over the posed edges with a non-degenerate baseline; `None` when
    /// there are none.
    pub positive_scale_fraction: Option<f64>,
    /// Rotation disagreement between the measured relative rotation and the posed cameras; `None`
    /// when no posed edge.
    pub rotation_residual_deg: Option<AngularResidualDistribution>,
    /// Angle between the solved baseline and the measured baseline direction, in world frame;
    /// `None` when no posed edge has a usable baseline.
    pub translation_residual_deg: Option<AngularResidualDistribution>,
    /// Camera-balanced incident evidence, sorted by stable `ViewId`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<VerifiedPoseGraphCameraDiagnostic>,
}

/// Incident verified-edge evidence for one currently posed camera.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerifiedPoseGraphCameraDiagnostic {
    /// Posed camera the evidence belongs to.
    pub view: ViewId,
    /// Selected relative-pose edges joining this camera to another posed camera.
    pub posed_relative_pose_degree: usize,
    /// Finite rotation residuals incident to this camera.
    pub rotation_residual_support: usize,
    /// Finite translation-direction residuals incident to this camera.
    pub translation_residual_support: usize,
    /// Median incident rotation residual, in degrees; `None` without support.
    pub median_rotation_residual_deg: Option<f64>,
    /// Median incident translation-direction residual, in degrees; `None` without support.
    pub median_translation_residual_deg: Option<f64>,
}

/// Rotation-only consistency of one preliminary global rotation solution against a selected
/// verified-edge set. This is available before translation averaging or triangulation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RotationAveragingConsistencyReport {
    /// Edge set the residuals were computed on.
    pub selection: VerifiedPairSelection,
    /// Pair-graph edges in the selected set.
    pub selected_pair_count: usize,
    /// Selected edges whose verified model carries a relative rotation.
    pub relative_pose_pair_count: usize,
    /// Selected relative-rotation edges whose two cameras have a solved rotation.
    pub solved_pair_count: usize,
    /// Disagreement between measured and solved relative rotations; `None` when no edge is solved.
    pub rotation_residual_deg: Option<AngularResidualDistribution>,
    /// Per-camera incident evidence for every camera with a solved rotation, sorted by `ViewId`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<RotationAveragingCameraDiagnostic>,
}

/// Incident relative-rotation evidence for one camera with a solved global rotation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RotationAveragingCameraDiagnostic {
    /// Camera the evidence belongs to.
    pub view: ViewId,
    /// Finite rotation residuals incident to this camera.
    pub rotation_residual_support: usize,
    /// Median incident rotation residual, in degrees; `None` without support.
    pub median_rotation_residual_deg: Option<f64>,
}

/// Evaluate a previously solved global rotation field without invoking the solver again.
pub(super) fn rotation_averaging_consistency_from_estimates(
    workspace: &SfmWorkspace,
    estimates: &RotationEstimates,
    selection: VerifiedPairSelection,
) -> Result<RotationAveragingConsistencyReport> {
    let mut rotations = vec![None; workspace.pair_graph().view_count()];
    for (view, rotation) in estimates.rotations() {
        if let Some(slot) = rotations.get_mut(view.index()) {
            *slot = Some(*rotation);
        }
    }

    let mut selected_pair_count = 0usize;
    let mut relative_pose_pair_count = 0usize;
    let mut solved_pair_count = 0usize;
    let mut residuals = Vec::new();
    let mut incident = vec![Vec::new(); rotations.len()];
    for (index, pair) in workspace.pair_graph().pairs().iter().enumerate() {
        let pair_id = crate::PairId::new(index);
        let active = workspace.pair_graph().is_active(pair_id);
        let selected = match selection {
            VerifiedPairSelection::Active => active,
            VerifiedPairSelection::Inactive => !active,
            VerifiedPairSelection::All => true,
        };
        if !selected {
            continue;
        }
        selected_pair_count += 1;
        let Some(relative_rotation) = workspace
            .geometric_model_store()
            .get(index)
            .and_then(|model| model.relative_rotation())
        else {
            continue;
        };
        relative_pose_pair_count += 1;
        let (Some(rotation_i), Some(rotation_j)) = (
            rotations.get(pair.first().index()).and_then(Option::as_ref),
            rotations
                .get(pair.second().index())
                .and_then(Option::as_ref),
        ) else {
            continue;
        };
        solved_pair_count += 1;
        let predicted = rotation_j * rotation_i.inverse();
        let error = rotation_angle_deg(&(relative_rotation.inverse() * predicted));
        if error.is_finite() {
            residuals.push(error);
            incident[pair.first().index()].push(error);
            incident[pair.second().index()].push(error);
        }
    }
    let cameras = incident
        .into_iter()
        .enumerate()
        .filter_map(|(view, mut errors)| {
            rotations[view].as_ref().map(|_| {
                let rotation_residual_support = errors.len();
                RotationAveragingCameraDiagnostic {
                    view: ViewId::new(view),
                    rotation_residual_support,
                    median_rotation_residual_deg: angular_distribution(&mut errors)
                        .map(|distribution| distribution.median_deg),
                }
            })
        })
        .collect();
    Ok(RotationAveragingConsistencyReport {
        selection,
        selected_pair_count,
        relative_pose_pair_count,
        solved_pair_count,
        rotation_residual_deg: angular_distribution(&mut residuals),
        cameras,
    })
}

/// Evaluate current metric poses against active, inactive, or all verified relative poses.
pub fn verified_pose_graph_consistency(
    workspace: &SfmWorkspace,
    selection: VerifiedPairSelection,
) -> VerifiedPoseGraphConsistencyReport {
    let mut selected_pair_count = 0usize;
    let mut relative_pose_pair_count = 0usize;
    let mut posed_pair_count = 0usize;
    let mut positive_scale_count = 0usize;
    let mut oriented_baseline_count = 0usize;
    let mut rotation_errors = Vec::new();
    let mut translation_errors = Vec::new();
    let view_count = workspace.pair_graph().view_count();
    let mut posed_relative_pose_degree = vec![0usize; view_count];
    let mut incident_rotation = vec![Vec::new(); view_count];
    let mut incident_translation = vec![Vec::new(); view_count];

    for (index, pair) in workspace.pair_graph().pairs().iter().enumerate() {
        let pair_id = crate::PairId::new(index);
        let active = workspace.pair_graph().is_active(pair_id);
        let selected = match selection {
            VerifiedPairSelection::Active => active,
            VerifiedPairSelection::Inactive => !active,
            VerifiedPairSelection::All => true,
        };
        if !selected {
            continue;
        }
        selected_pair_count += 1;
        let Some(relative_pose) = workspace
            .geometric_model_store()
            .get(index)
            .and_then(|model| model.relative_pose())
        else {
            continue;
        };
        relative_pose_pair_count += 1;
        let (Some(pose_i), Some(pose_j)) =
            (workspace.pose(pair.first()), workspace.pose(pair.second()))
        else {
            continue;
        };
        posed_pair_count += 1;
        posed_relative_pose_degree[pair.first().index()] += 1;
        posed_relative_pose_degree[pair.second().index()] += 1;

        let predicted_rotation = pose_j.pose.rotation() * pose_i.pose.rotation().inverse();
        let rotation_error = rotation_angle_deg(
            &(relative_pose.rotation_source_to_target().inverse() * predicted_rotation),
        );
        if rotation_error.is_finite() {
            rotation_errors.push(rotation_error);
            incident_rotation[pair.first().index()].push(rotation_error);
            incident_rotation[pair.second().index()].push(rotation_error);
        }

        let predicted = pose_j.pose.center() - pose_i.pose.center();
        let measured_world = pose_i.pose.rotation().inverse()
            * relative_pose.baseline_direction_in_source().as_ref();
        if predicted.norm_squared() <= 1e-20 || measured_world.norm_squared() <= 1e-20 {
            continue;
        }
        let cosine = predicted
            .normalize()
            .dot(&measured_world.normalize())
            .clamp(-1.0, 1.0);
        oriented_baseline_count += 1;
        positive_scale_count += usize::from(cosine > 0.0);
        let translation_error = cosine.acos().to_degrees();
        if translation_error.is_finite() {
            translation_errors.push(translation_error);
            incident_translation[pair.first().index()].push(translation_error);
            incident_translation[pair.second().index()].push(translation_error);
        }
    }

    let cameras = incident_rotation
        .into_iter()
        .zip(incident_translation)
        .enumerate()
        .filter_map(|(view, (mut rotations, mut translations))| {
            workspace.has_pose(ViewId::new(view)).then(|| {
                let rotation_residual_support = rotations.len();
                let translation_residual_support = translations.len();
                VerifiedPoseGraphCameraDiagnostic {
                    view: ViewId::new(view),
                    posed_relative_pose_degree: posed_relative_pose_degree[view],
                    rotation_residual_support,
                    translation_residual_support,
                    median_rotation_residual_deg: angular_distribution(&mut rotations)
                        .map(|distribution| distribution.median_deg),
                    median_translation_residual_deg: angular_distribution(&mut translations)
                        .map(|distribution| distribution.median_deg),
                }
            })
        })
        .collect();

    VerifiedPoseGraphConsistencyReport {
        selection,
        selected_pair_count,
        relative_pose_pair_count,
        posed_pair_count,
        positive_scale_count,
        positive_scale_fraction: (oriented_baseline_count > 0)
            .then_some(positive_scale_count as f64 / oriented_baseline_count as f64),
        rotation_residual_deg: angular_distribution(&mut rotation_errors),
        translation_residual_deg: angular_distribution(&mut translation_errors),
        cameras,
    }
}

pub(crate) fn angular_distribution(values: &mut [f64]) -> Option<AngularResidualDistribution> {
    values.sort_by(f64::total_cmp);
    (!values.is_empty()).then(|| AngularResidualDistribution {
        count: values.len(),
        minimum_deg: values[0],
        median_deg: sorted_quantile(values, 0.5),
        p90_deg: sorted_quantile(values, 0.9),
        maximum_deg: values[values.len() - 1],
    })
}

fn sorted_quantile(values: &[f64], quantile: f64) -> f64 {
    let position = quantile.clamp(0.0, 1.0) * values.len().saturating_sub(1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    values[lower] * (1.0 - fraction) + values[upper] * fraction
}
