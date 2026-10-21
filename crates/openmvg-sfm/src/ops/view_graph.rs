// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared view-graph conditioning before numerical SfM solvers.
//!
//! Graph inference belongs here, not inside a rotation or translation solver:
//! this stage computes one connectivity-safe edge mask and one confidence per
//! verified pair. Every downstream solver then consumes the same graph.

use openmvg_geometry::averaging::view_graph::cut_edges_protecting_bridges;

use crate::backends::RotationAveragingFn;
use crate::{PairId, Result, SfmError, SfmWorkspace};

/// Outcome of one view-graph conditioning pass over the active verified pairs. Removed pairs are
/// deactivated and get zero confidence; kept pairs keep theirs.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ViewGraphConditioningReport {
    /// Method label, `adaptive_cycle_gnc_<scale>deg` with the final scale of the graduated solve,
    /// in degrees; a `_peeled<N>` suffix records N cameras removed as a rotation tail afterwards.
    pub method: String,
    /// Active pairs entering the pass.
    pub input_edges: usize,
    /// Pairs kept active after the pass.
    pub kept_edges: usize,
    /// Pairs deactivated as redundant outliers.
    pub hard_removed: usize,
    /// Outlier candidates kept because removing them would disconnect the graph.
    pub bridges_protected: usize,
    /// Median confidence of the kept pairs after conditioning; `0` when none is kept.
    pub median_confidence: f64,
    /// Rotation-averaging solves the conditioning performed.
    pub rotation_solves: usize,
}

/// Coverage decision made before numerical averaging. Views outside the retained component
/// remain addressable in the workspace, but all of their pairs are inactive and therefore
/// cannot leak into matching, tracks, or either averaging solver.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ViewGraphComponentReport {
    /// Views in the pair graph before the pass.
    pub input_views: usize,
    /// Views in the retained largest connected component.
    pub retained_views: usize,
    /// Views outside the retained component, whose pairs were deactivated.
    pub excluded_views: usize,
    /// Active pairs before the pass.
    pub input_edges: usize,
    /// Active pairs after the pass, all inside the retained component.
    pub retained_edges: usize,
}

/// Retain the largest active connected component, with deterministic lowest-image-ID
/// tie-breaking. This is an explicit graph policy rather than behavior hidden inside an
/// averaging solver; numerical solvers receive only the retained constraints.
pub(crate) fn retain_largest_view_graph_component(
    workspace: &mut SfmWorkspace,
) -> Result<ViewGraphComponentReport> {
    let input_views = workspace.pair_graph().view_count();
    let input_edges = workspace.pair_graph().active_count();
    if input_edges == 0 {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "verified active pairs",
            method: "verify_pairs",
        });
    }

    let components = super::probes::active_connected_components(workspace)?;
    let largest = components
        .into_iter()
        .max_by(|left, right| {
            left.len()
                .cmp(&right.len())
                .then_with(|| right.first().copied().cmp(&left.first().copied()))
        })
        .unwrap_or_default();
    let retained: std::collections::BTreeSet<_> = largest.iter().copied().collect();
    let pairs_to_deactivate: Vec<_> = workspace
        .pair_graph()
        .active_pair_ids()
        .filter(|&pair_id| {
            workspace
                .pair_graph()
                .pair(pair_id)
                .map(|pair| !retained.contains(&pair.first()) || !retained.contains(&pair.second()))
                .unwrap_or(false)
        })
        .collect();
    for pair_id in pairs_to_deactivate {
        workspace.pair_graph_mut().deactivate_pair_and(pair_id)?;
    }

    Ok(ViewGraphComponentReport {
        input_views,
        retained_views: largest.len(),
        excluded_views: input_views.saturating_sub(largest.len()),
        input_edges,
        retained_edges: workspace.pair_graph().active_count(),
    })
}

/// Deactivate every pair incident to one of `views`; returns how many pairs were deactivated.
pub(crate) fn deactivate_view_pairs(
    workspace: &mut SfmWorkspace,
    views: &[crate::ViewId],
) -> Result<usize> {
    let mut pair_ids = Vec::new();
    for &view in views {
        for neighbor in workspace.pair_graph().neighbors(view)? {
            if workspace.pair_graph().is_active(neighbor.pair_id) {
                pair_ids.push(neighbor.pair_id);
            }
        }
    }
    pair_ids.sort_unstable();
    pair_ids.dedup();
    for &pair_id in &pair_ids {
        workspace.pair_graph_mut().deactivate_pair_and(pair_id)?;
    }
    Ok(pair_ids.len())
}

/// A pair whose confidence is below this fraction of the median confidence is a hard-cut
/// candidate; a bridge is never cut, so the graph stays connected.
const ADAPTIVE_CUT_BELOW_MEDIAN: f64 = 0.05;
/// A pair that disagrees with the conditioned rotations by more than this is a cut candidate too.
/// It matches the residual limit of the post-fit rotation-graph gate, which reads every active
/// pair as trusted.
const ADAPTIVE_CUT_ABOVE_RESIDUAL_DEG: f64 = 5.0;

/// Condition the active pairs with the adaptive cycle-reliability and graduated-solve method.
///
/// The method's weights decide which pairs stay: pairs whose confidence is negligible against the
/// median, or that disagree with the conditioned rotations by more than the residual limit, are
/// deactivated unless they are bridges, because later stages read "active" as "trusted" (the
/// post-fit rotation-graph residual, for one).
///
/// The cut is the weakest part of the method. Outcomes flip on about 1 degree changes of either
/// limit, and the confidence rule removes mostly well-fitting pairs on tightly fitted scenes.
/// Evaluating residual-only cuts, a minimum kept degree per view, and a tighter retry ladder on
/// 106 hard scenes found no setting that beats this one on average, so it stays as is.
pub(crate) fn condition_view_graph(
    workspace: &mut SfmWorkspace,
    rotation_initializer: RotationAveragingFn,
) -> Result<ViewGraphConditioningReport> {
    let (edges, models) = super::averaging::active_geometric_models(workspace)?;
    let conditioning =
        super::averaging::condition_rotations_adaptive(rotation_initializer, &edges, &models)?;
    let pair_ids = pair_ids_for_edges(workspace, edges.iter().map(|edge| edge.pair))?;
    let median_confidence = median(&conditioning.confidence).max(f64::MIN_POSITIVE);
    let scores: Vec<f64> = conditioning
        .confidence
        .iter()
        .zip(&conditioning.residuals_deg)
        .map(|(c, residual)| {
            if *residual > ADAPTIVE_CUT_ABOVE_RESIDUAL_DEG {
                0.0
            } else {
                (c / median_confidence).clamp(0.0, 1.0)
            }
        })
        .collect();
    let endpoints: Vec<_> = edges
        .iter()
        .map(|edge| (edge.view_i, edge.view_j))
        .collect();
    let conditioned = cut_edges_protecting_bridges(&scores, &endpoints, ADAPTIVE_CUT_BELOW_MEDIAN)
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    // Kept pairs keep their confidence: the soft confidences select the pairs, but feeding them to
    // translation averaging and positioning lost cores that the selection alone keeps (pair
    // A/B on the hard scenes: Rainbow Bridge 115 -> 0 cameras with them, 115 -> 115 without).
    apply_cut(workspace, &pair_ids, &conditioned.keep)?;
    Ok(ViewGraphConditioningReport {
        method: format!("adaptive_cycle_gnc_{:.3}deg", conditioning.final_scale_deg),
        input_edges: edges.len(),
        kept_edges: conditioned.kept_count(),
        hard_removed: conditioned.hard_removed_count,
        bridges_protected: conditioned.protected_bridge_count,
        median_confidence: active_median_confidence(workspace, &pair_ids),
        rotation_solves: conditioning.solve_count,
    })
}

fn pair_ids_for_edges(
    workspace: &SfmWorkspace,
    pairs: impl IntoIterator<Item = crate::ViewPair>,
) -> Result<Vec<PairId>> {
    pairs
        .into_iter()
        .map(|pair| {
            workspace
                .pair_graph()
                .pair_id(pair.first(), pair.second())
                .ok_or_else(|| {
                    SfmError::InvariantViolation(format!(
                        "active edge {pair} is absent from the pair graph"
                    ))
                })
        })
        .collect()
}

/// Zero the confidence of every pair with `keep == false` and deactivate it.
fn apply_cut(workspace: &mut SfmWorkspace, pair_ids: &[PairId], keep: &[bool]) -> Result<()> {
    if pair_ids.len() != keep.len() {
        return Err(SfmError::InvalidInput(
            "view-graph conditioning result has the wrong edge count".to_owned(),
        ));
    }
    for (&pair_id, _) in pair_ids.iter().zip(keep).filter(|(_, keep)| !**keep) {
        workspace.pair_graph_mut().set_confidence(pair_id, 0.0)?;
        workspace.pair_graph_mut().deactivate_pair_and(pair_id)?;
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut finite: Vec<_> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(f64::total_cmp);
    finite[finite.len() / 2]
}

fn active_median_confidence(workspace: &SfmWorkspace, pair_ids: &[PairId]) -> f64 {
    let confidences: Vec<_> = pair_ids
        .iter()
        .filter(|&&pair_id| workspace.pair_graph().is_active(pair_id))
        .filter_map(|&pair_id| workspace.pair_graph().confidence(pair_id))
        .collect();
    median(&confidences)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageMeta;
    use crate::{Camera, IntrinsicId, ViewId};

    #[test]
    fn the_cut_zeroes_and_deactivates_removed_edges_and_leaves_kept_ones_alone() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
                ImageMeta::external_id("c", 64, 48),
            ])
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs([
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(1), ViewId::new(2)),
            ])
            .build()
            .unwrap();
        let pairs = [PairId::new(0), PairId::new(1)];

        let before = workspace.pair_graph().confidence(pairs[0]);
        apply_cut(&mut workspace, &pairs, &[true, false]).unwrap();
        assert!(apply_cut(&mut workspace, &pairs, &[true]).is_err());

        assert_eq!(workspace.pair_graph().confidence(pairs[0]), before);
        assert!(workspace.pair_graph().is_active(pairs[0]));
        assert_eq!(workspace.pair_graph().confidence(pairs[1]), Some(0.0));
        assert!(!workspace.pair_graph().is_active(pairs[1]));
    }

    #[test]
    fn largest_component_policy_excludes_smaller_components_and_isolated_views() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..6).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs([
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(1), ViewId::new(2)),
                (ViewId::new(3), ViewId::new(4)),
            ])
            .build()
            .unwrap();

        let report = retain_largest_view_graph_component(&mut workspace).unwrap();

        assert_eq!(report.input_views, 6);
        assert_eq!(report.retained_views, 3);
        assert_eq!(report.excluded_views, 3);
        assert_eq!(report.input_edges, 3);
        assert_eq!(report.retained_edges, 2);
        let active_endpoints: Vec<_> = workspace
            .pair_graph()
            .active_pair_ids()
            .map(|pair_id| {
                let pair = workspace.pair_graph().pair(pair_id).unwrap();
                (pair.first(), pair.second())
            })
            .collect();
        assert_eq!(
            active_endpoints,
            vec![
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(1), ViewId::new(2))
            ]
        );
    }
}
