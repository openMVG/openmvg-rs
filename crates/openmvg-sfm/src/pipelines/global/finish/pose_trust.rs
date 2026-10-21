// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pose trust from the verified relative-pose graph and the acceptance gate built on it.

use crate::{SfmWorkspace, ViewId};
use openmvg_core::model::rotation_angle_deg;

use super::report::GlobalTrappedViewDiagnostic;

/// Floor of the translation acceptance limit, in degrees. A healthy reconstruction's translation
/// p90 against the verified baselines sits around 3 degrees (median 3.1 over the accepted finishes
/// of 286 Global reports), so a smaller floor turns a tightly conditioned starting graph into a
/// false "degradation": Avignon's identical post-fit state (3.0 degrees) was rejected once the
/// conditioning had lowered the starting residual to 0.86. It equals the absolute rotation-graph
/// limit of the post-fit trust policy.
const TRANSLATION_ACCEPTANCE_FLOOR_DEG: f64 = 5.0;
/// Floor of the rotation acceptance limit, in degrees.
const ROTATION_ACCEPTANCE_FLOOR_DEG: f64 = 2.0;

/// Whether a bundle-adjustment result keeps the bulk agreement with the verified pair graph.
///
/// Only the graph quantiles decide. The per-camera trapped count does not: rotation averaging
/// fits the pair edges, wrong edges included, so a camera that bundle adjustment moves to its true
/// pose can newly disagree with a majority of its edges. In a 12-scene MegaDepth study, each of the
/// 11 rollbacks that count caused discarded a result closer to ground truth.
pub(crate) fn graph_candidate_is_acceptable(
    before: &GraphConsistencyStats,
    after: &GraphConsistencyStats,
) -> bool {
    within_limit(
        after.translation_p90_deg,
        acceptance_limit(before.translation_p90_deg, TRANSLATION_ACCEPTANCE_FLOOR_DEG),
    ) && within_limit(
        after.rotation_p90_deg,
        acceptance_limit(before.rotation_p90_deg, ROTATION_ACCEPTANCE_FLOOR_DEG),
    )
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GraphConsistencyStats {
    pub(crate) translation_p90_deg: Option<f64>,
    pub(crate) positive_scale_fraction: Option<f64>,
    pub(crate) rotation_p90_deg: Option<f64>,
    pub(crate) locally_trapped_cameras: usize,
    pub(crate) trapped_views: Vec<GlobalTrappedViewDiagnostic>,
}

/// Median pair parallax below which a two-view baseline direction is not evidence: a small
/// calibration bias trades rotation for baseline direction, so near pure-rotation edges (a zoom
/// burst from one spot) disagree by tens of degrees with a correct reconstruction. Such edges
/// stay out of the translation quantile and the per-camera majorities, and only count toward
/// the reported sign fraction.
const TRANSLATION_EVIDENCE_MIN_PARALLAX_DEG: f64 = 5.0;

/// Independent angular agreement of the current poses with the active verified graph.
/// Quantiles protect the bulk while per-camera incident majorities catch a single leaf camera
/// falling into an otherwise low-reprojection repeated-structure minimum. There is no maximum,
/// because one two-view measurement (for example an outlier edge kept only as a
/// connectivity bridge) must not veto a multi-view reprojection refinement, globally or per
/// camera.
pub(crate) fn graph_consistency_stats(workspace: &SfmWorkspace) -> GraphConsistencyStats {
    let mut translation_errors = Vec::new();
    let mut rotation_errors = Vec::new();
    let mut positive_scale_count = 0usize;
    let mut oriented_baseline_count = 0usize;
    let mut active_pair_degree = vec![0usize; workspace.pair_graph().view_count()];
    let mut incident_translation = vec![Vec::new(); workspace.pair_graph().view_count()];
    let mut incident_rotation = vec![Vec::new(); workspace.pair_graph().view_count()];
    for pair_id in workspace.pair_graph().active_pair_ids() {
        let Some(pair) = workspace.pair_graph().pair(pair_id) else {
            continue;
        };
        active_pair_degree[pair.first().index()] += 1;
        active_pair_degree[pair.second().index()] += 1;
        let Some(model) = workspace.geometric_model_store().get(pair_id.index()) else {
            continue;
        };
        let (Some(pose_i), Some(pose_j)) =
            (workspace.pose(pair.first()), workspace.pose(pair.second()))
        else {
            continue;
        };

        let predicted_rotation = pose_j.pose.rotation() * pose_i.pose.rotation().inverse();
        if let Some(relative_rotation) = model.relative_rotation() {
            let rotation_error =
                rotation_angle_deg(&(relative_rotation.inverse() * predicted_rotation));
            if rotation_error.is_finite() {
                rotation_errors.push(rotation_error);
                incident_rotation[pair.first().index()].push(rotation_error);
                incident_rotation[pair.second().index()].push(rotation_error);
            }
        }

        let Some(relative_pose) = model.relative_pose() else {
            continue;
        };
        let low_parallax =
            crate::ops::averaging::pair_median_parallax_deg(workspace, pair_id, model)
                .is_some_and(|parallax| parallax < TRANSLATION_EVIDENCE_MIN_PARALLAX_DEG);
        let predicted = pose_j.pose.center() - pose_i.pose.center();
        if predicted.norm_squared() <= 1e-20 {
            continue;
        }
        let measured_world = pose_i.pose.rotation().inverse()
            * relative_pose.baseline_direction_in_source().as_ref();
        if measured_world.norm_squared() <= 1e-20 {
            continue;
        }
        let cosine = predicted
            .normalize()
            .dot(&measured_world.normalize())
            .clamp(-1.0, 1.0);
        oriented_baseline_count += 1;
        if cosine > 0.0 {
            positive_scale_count += 1;
        }
        let translation_error = cosine.acos().to_degrees();
        if translation_error.is_finite() && !low_parallax {
            translation_errors.push(translation_error);
            incident_translation[pair.first().index()].push(translation_error);
            incident_translation[pair.second().index()].push(translation_error);
        }
    }
    let trapped_views = incident_rotation
        .into_iter()
        .zip(incident_translation)
        .enumerate()
        .filter_map(|(view, (rotations, translations))| {
            trapped_view_diagnostic(
                ViewId::new(view),
                active_pair_degree[view],
                rotations,
                translations,
            )
        })
        .collect::<Vec<_>>();
    GraphConsistencyStats {
        translation_p90_deg: percentile(&mut translation_errors, 0.9),
        positive_scale_fraction: (oriented_baseline_count > 0)
            .then_some(positive_scale_count as f64 / oriented_baseline_count as f64),
        rotation_p90_deg: percentile(&mut rotation_errors, 0.9),
        locally_trapped_cameras: trapped_views.len(),
        trapped_views,
    }
}

pub(super) fn trapped_view_diagnostic(
    view: ViewId,
    active_pair_degree: usize,
    mut rotations: Vec<f64>,
    mut translations: Vec<f64>,
) -> Option<GlobalTrappedViewDiagnostic> {
    let rotation_residual_support = rotations.len();
    let translation_residual_support = translations.len();
    let median_rotation_residual_deg = majority_residual(&mut rotations);
    let median_translation_residual_deg = majority_residual(&mut translations);
    (median_rotation_residual_deg.is_some_and(|error| error > 5.0)
        || median_translation_residual_deg.is_some_and(|error| error > 10.0))
    .then_some(GlobalTrappedViewDiagnostic {
        view,
        active_pair_degree,
        rotation_residual_support,
        translation_residual_support,
        median_rotation_residual_deg,
        median_translation_residual_deg,
    })
}

/// Residual exceeded by a strict majority of a camera's incident edges (the lower median). One
/// edge is not redundant evidence about the camera, so it yields none.
fn majority_residual(values: &mut [f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[(values.len() - 1) / 2])
}

fn percentile(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values.get(index).copied()
}

fn acceptance_limit(before: Option<f64>, floor_deg: f64) -> Option<f64> {
    before.map(|error| (3.0 * error).max(floor_deg))
}

fn within_limit(after: Option<f64>, limit: Option<f64>) -> bool {
    match (after, limit) {
        (Some(after), Some(limit)) => after <= limit,
        (None, None) => true,
        // Losing previously available graph evidence usually means baselines collapsed or
        // poses disappeared. It must not silently turn the independent quality gate off.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GraphConsistencyStats, graph_candidate_is_acceptable, trapped_view_diagnostic, within_limit,
    };
    use crate::ViewId;

    #[test]
    fn disappearing_graph_evidence_fails_closed() {
        assert!(!within_limit(None, Some(10.0)));
        assert!(!within_limit(Some(1.0), None));
        assert!(within_limit(None, None));
        assert!(within_limit(Some(1.0), Some(2.0)));
        assert!(!within_limit(Some(3.0), Some(2.0)));
    }

    #[test]
    fn a_healthy_result_is_not_rejected_because_the_starting_graph_was_tighter() {
        // Avignon: the same 3 degree translation p90 after the fit passes whether the starting
        // residual was 1.36 or 0.86 degrees.
        let stats = |translation: f64, rotation: f64| GraphConsistencyStats {
            translation_p90_deg: Some(translation),
            rotation_p90_deg: Some(rotation),
            ..GraphConsistencyStats::default()
        };
        assert!(graph_candidate_is_acceptable(
            &stats(1.36, 1.1),
            &stats(3.1, 1.61)
        ));
        assert!(graph_candidate_is_acceptable(
            &stats(0.86, 0.75),
            &stats(3.01, 1.59)
        ));
        // A real degradation is still caught: far above both the floor and three times before.
        assert!(!graph_candidate_is_acceptable(
            &stats(1.0, 1.0),
            &stats(12.0, 1.0)
        ));
        assert!(!graph_candidate_is_acceptable(
            &stats(1.0, 0.5),
            &stats(1.0, 2.5)
        ));
    }

    #[test]
    fn a_camera_newly_disagreeing_with_its_edges_does_not_reject_a_healthy_bulk() {
        let before = GraphConsistencyStats {
            translation_p90_deg: Some(1.0),
            rotation_p90_deg: Some(1.0),
            locally_trapped_cameras: 0,
            ..GraphConsistencyStats::default()
        };
        let after = GraphConsistencyStats {
            locally_trapped_cameras: 3,
            ..before.clone()
        };

        assert!(graph_candidate_is_acceptable(&before, &after));
    }

    #[test]
    fn a_camera_is_trapped_only_by_a_strict_majority_of_its_edges() {
        let view = ViewId::new(4);
        // One edge is not redundant evidence about the camera, whatever its residual.
        assert!(trapped_view_diagnostic(view, 1, vec![90.0], vec![90.0]).is_none());
        // Half of the incident edges is not a majority.
        assert!(trapped_view_diagnostic(view, 2, vec![1.0, 6.0], vec![2.0, 60.0]).is_none());
        assert!(trapped_view_diagnostic(view, 4, vec![1.0, 2.0, 6.0, 7.0], vec![1.0; 4]).is_none());
        // The thresholds are strict.
        assert!(trapped_view_diagnostic(view, 2, vec![5.0; 2], vec![10.0; 2]).is_none());

        let trapped = trapped_view_diagnostic(view, 5, vec![1.0, 6.0, 7.0], vec![2.0])
            .expect("two of three incident rotations disagree");
        assert_eq!(trapped.view, view);
        assert_eq!(trapped.active_pair_degree, 5);
        assert_eq!(trapped.rotation_residual_support, 3);
        assert_eq!(trapped.translation_residual_support, 1);
        assert_eq!(trapped.median_rotation_residual_deg, Some(6.0));
        assert_eq!(trapped.median_translation_residual_deg, None);
    }
}
