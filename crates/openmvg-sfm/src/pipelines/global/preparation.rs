// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Global graph preparation: keep the largest connected component, condition the pairs against
//! the rotations they imply, peel a small tail of cameras that disagree with them, and solve the
//! rotations the candidate cascade reuses.
//!
//! Global initialization currently requires one PoseId per view. Shared-pose reconstructions
//! remain valid for storage and backend bundle adjustment, but are rejected here before work.

use super::config::GlobalSfmConfig;
use super::report::GlobalPreparationReport;
use crate::backends::RotationAveragingFn;
use crate::ops::averaging::{
    VerifiedPairSelection, rotation_averaging_consistency_from_preparation,
};
use crate::ops::view_graph::{
    condition_view_graph, deactivate_view_pairs, retain_largest_view_graph_component,
};
use crate::{Result, SfmError, SfmWorkspace, ViewId};

/// Largest share of the cameras a rotation-tail peel may remove: past it the rotation graph is
/// wrong as a whole, not just in a tail, and the consensus rejection stands.
const MAX_ROTATION_TAIL_PEEL_FRACTION: f64 = 0.25;
/// Peel rounds: removing a tail can expose a second, smaller one.
const MAX_ROTATION_TAIL_PEEL_ROUNDS: usize = 2;

pub(crate) struct PreparedGlobal {
    pub(crate) report: GlobalPreparationReport,
    pub(crate) pair_state: Vec<PairState>,
    pub(crate) rotations: crate::ops::averaging::GlobalRotationPreparation,
}

#[derive(Clone, Copy)]
pub(crate) struct PairState {
    active: bool,
    confidence: f64,
}

pub(crate) fn prepare_global(
    workspace: &mut SfmWorkspace,
    rotation: RotationAveragingFn,
    config: &GlobalSfmConfig,
) -> Result<PreparedGlobal> {
    let mut poses = std::collections::BTreeSet::new();
    for view in workspace.views() {
        if !poses.insert(view.pose_id) {
            return Err(SfmError::UnsupportedSharedPoseInitialization { pose: view.pose_id });
        }
    }
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "precomputed tracks",
            method: "build_tracks",
        });
    }
    let initial_largest_component = retain_largest_view_graph_component(workspace)?;
    let mut rotation_conditioning = condition_view_graph(workspace, rotation)?;
    let mut final_largest_component = retain_largest_view_graph_component(workspace)?;

    let mut peeled = 0;
    let mut rotations =
        crate::ops::averaging::prepare_global_rotation_solution(workspace, rotation)?;
    for _ in 0..MAX_ROTATION_TAIL_PEEL_ROUNDS {
        let consistency = rotation_averaging_consistency_from_preparation(
            workspace,
            &rotations,
            VerifiedPairSelection::All,
        )?;
        let Some(tail) = rotation_tail_to_peel(&consistency, config) else {
            break;
        };
        deactivate_view_pairs(workspace, &tail)?;
        final_largest_component = retain_largest_view_graph_component(workspace)?;
        peeled += tail.len();
        rotations = crate::ops::averaging::prepare_global_rotation_solution(workspace, rotation)?;
    }
    if peeled > 0 {
        rotation_conditioning
            .method
            .push_str(&format!("_peeled{peeled}"));
    }

    let candidate_domain_camera_ids = candidate_domain_camera_ids(workspace);
    Ok(PreparedGlobal {
        report: GlobalPreparationReport {
            initial_largest_component,
            rotation_conditioning,
            final_largest_component,
            candidate_domain_camera_count: candidate_domain_camera_ids.len(),
            candidate_domain_camera_ids,
        },
        pair_state: pair_conditioning_state(workspace)?,
        rotations,
    })
}

/// The cameras whose verified pairs disagree with the solved rotations, when they are a tail the
/// consensus gate would reject but small enough to peel instead of discarding the whole graph.
fn rotation_tail_to_peel(
    consistency: &crate::ops::averaging::RotationAveragingConsistencyReport,
    config: &GlobalSfmConfig,
) -> Option<Vec<ViewId>> {
    let policy = config.policy.rotation_consensus();
    let mut supported = 0usize;
    let mut tail = Vec::new();
    for camera in &consistency.cameras {
        let Some(median) = camera.median_rotation_residual_deg else {
            continue;
        };
        supported += 1;
        if median > policy.max_all_verified_camera_median_deg() {
            tail.push(camera.view);
        }
    }
    let fraction = tail.len() as f64 / supported.max(1) as f64;
    (supported > 0
        && fraction > policy.max_all_verified_inconsistent_camera_fraction()
        && fraction <= MAX_ROTATION_TAIL_PEEL_FRACTION)
        .then_some(tail)
}

fn candidate_domain_camera_ids(workspace: &SfmWorkspace) -> Vec<ViewId> {
    workspace
        .pair_graph()
        .active_pair_ids()
        .filter_map(|pair| workspace.pair_graph().pair(pair))
        .flat_map(|pair| [pair.first(), pair.second()])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn pair_conditioning_state(workspace: &SfmWorkspace) -> Result<Vec<PairState>> {
    (0..workspace.pair_graph().len())
        .map(|index| {
            let pair = crate::PairId::new(index);
            Ok(PairState {
                active: workspace.pair_graph().is_active(pair),
                confidence: workspace
                    .pair_graph()
                    .confidence(pair)
                    .ok_or(SfmError::InvalidId {
                        kind: crate::PairId::kind(),
                        index,
                    })?,
            })
        })
        .collect()
}

pub(crate) fn apply_pair_conditioning_state(
    workspace: &mut SfmWorkspace,
    pair_state: &[PairState],
) -> Result<()> {
    if workspace.pair_graph().len() != pair_state.len() {
        return Err(SfmError::InvalidInput(
            "conditioning plan does not match the prepared pair domain".to_owned(),
        ));
    }
    for (index, state) in pair_state.iter().enumerate() {
        let pair = crate::PairId::new(index);
        if state.active {
            workspace.pair_graph_mut().activate_pair_and(pair)?;
        } else {
            workspace.pair_graph_mut().deactivate_pair_and(pair)?;
        }
        workspace
            .pair_graph_mut()
            .set_confidence(pair, state.confidence)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::averaging::{
        RotationAveragingCameraDiagnostic, RotationAveragingConsistencyReport,
    };

    fn consistency(medians: &[Option<f64>]) -> RotationAveragingConsistencyReport {
        RotationAveragingConsistencyReport {
            selection: VerifiedPairSelection::All,
            selected_pair_count: 0,
            relative_pose_pair_count: 0,
            solved_pair_count: 0,
            rotation_residual_deg: None,
            cameras: medians
                .iter()
                .enumerate()
                .map(|(index, median)| RotationAveragingCameraDiagnostic {
                    view: ViewId::new(index),
                    rotation_residual_support: usize::from(median.is_some()),
                    median_rotation_residual_deg: *median,
                })
                .collect(),
        }
    }

    #[test]
    fn a_tail_between_the_gate_and_the_peel_bound_is_peeled() {
        let config = GlobalSfmConfig::default();
        // 20 of 100 cameras above the 5 degree limit: past the 10% gate, within the 25% bound.
        let mut medians = vec![Some(0.5); 100];
        for median in medians.iter_mut().take(20) {
            *median = Some(9.0);
        }
        let tail = rotation_tail_to_peel(&consistency(&medians), &config).expect("peelable tail");
        assert_eq!(tail, (0..20).map(ViewId::new).collect::<Vec<_>>());
    }

    #[test]
    fn a_passing_or_wrong_as_a_whole_graph_is_never_peeled() {
        let config = GlobalSfmConfig::default();
        let mut passing = vec![Some(0.5); 100];
        for median in passing.iter_mut().take(8) {
            *median = Some(9.0);
        }
        assert!(rotation_tail_to_peel(&consistency(&passing), &config).is_none());
        let mut whole = vec![Some(0.5); 100];
        for median in whole.iter_mut().take(40) {
            *median = Some(9.0);
        }
        assert!(rotation_tail_to_peel(&consistency(&whole), &config).is_none());
        assert!(rotation_tail_to_peel(&consistency(&[None, None]), &config).is_none());
    }
}
