// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The standard front-end sequence: extract, match, verify, optionally pre-calibrate radial
//! distortion, and build tracks.

use crate::backends::{FeatureExtractor, GeometricVerifier, PairMatcher};
use crate::ops::radial_calibration::{
    calibrate_radial_distortion_with_progress, has_unknown_radial_distortion,
};
use crate::ops::tracks::build_tracks_with_progress;
use crate::progress::SfmProgressSink;
use crate::reports::FrontendReport;
use crate::workspace::selector::{PairSelector, ViewSelector};
use crate::{Result, SfmError, SfmWorkspace};
use openmvg_geometry::averaging::view_graph::{StrengthEdge, keep_strongest_edges_per_view};

/// Extract features for images that have none, match the active pairs without matches, verify
/// the active pairs without a model, and build tracks.
///
/// `radial_calibration` estimates `k1` before tracking for camera groups that declare a radial
/// model with unknown (all-zero) distortion. The stage is skipped when no group does, so
/// calibrated and pinhole cameras never enter it.
///
/// `max_pairs_per_view` keeps only the verified pairs that are among each view's strongest
/// (by inliers) at either end and deactivates the rest before tracking; `None` keeps every
/// verified pair.
pub fn run_frontend(
    workspace: &mut SfmWorkspace,
    extractor: &dyn FeatureExtractor,
    matcher: &dyn PairMatcher,
    verifier: &dyn GeometricVerifier,
    radial_calibration: bool,
    max_pairs_per_view: Option<usize>,
    progress: &dyn SfmProgressSink,
) -> Result<FrontendReport> {
    workspace.extract_features_with_progress(extractor, ViewSelector::Missing, progress)?;
    let extracted_views = workspace.feature_store().populated_count();
    let keypoints = workspace
        .feature_store()
        .iter_populated()
        .map(|(_, features)| features.len())
        .sum();

    workspace.match_pairs_with_progress(matcher, PairSelector::Unmatched, progress)?;
    let match_records = workspace.match_store().populated_count();
    let pairs_with_raw_matches = workspace
        .match_store()
        .iter_populated()
        .filter(|(_, matches)| !matches.is_empty())
        .count();
    let raw_matches = workspace
        .match_store()
        .iter_populated()
        .map(|(_, matches)| matches.len())
        .sum();

    workspace.verify_pairs_with_progress(verifier, PairSelector::Unverified, progress)?;
    let radial_calibration = (radial_calibration && has_unknown_radial_distortion(workspace))
        .then(|| calibrate_radial_distortion_with_progress(workspace, verifier, progress))
        .transpose()?;
    let verified_pairs = workspace.geometric_model_store().populated_count();
    let verified_inliers = workspace
        .geometric_model_store()
        .iter_populated()
        .map(|(_, model)| model.evidence().inlier_count)
        .sum();

    let filtered_pairs = match max_pairs_per_view {
        Some(budget) => deactivate_weak_pairs(workspace, budget)?,
        None => 0,
    };

    let tracks = build_tracks_with_progress(workspace, progress)?;

    Ok(FrontendReport {
        view_count: workspace.views().len(),
        extracted_view_count: extracted_views,
        keypoint_count: keypoints,
        match_record_count: match_records,
        raw_match_pair_count: pairs_with_raw_matches,
        raw_match_count: raw_matches,
        verified_pair_count: verified_pairs,
        verified_inlier_count: verified_inliers,
        filtered_pair_count: filtered_pairs,
        track_count: tracks.track_count,
        extraction_method: None,
        radial_calibration,
    })
}

/// Reject verification evidence for pairs outside either view's strongest
/// `budget` pairs. Putative matches remain available for storage and inspection.
fn deactivate_weak_pairs(workspace: &mut SfmWorkspace, budget: usize) -> Result<usize> {
    let (pairs, edges): (Vec<_>, Vec<_>) = workspace
        .geometric_models_view()
        .iter()
        .filter(|(pair, _)| workspace.pair_graph().is_active_pair(*pair))
        .map(|(pair, model)| {
            let edge = StrengthEdge {
                view_i: pair.first(),
                view_j: pair.second(),
                inliers: model.evidence().inlier_count,
            };
            (pair, edge)
        })
        .unzip();
    let keep = keep_strongest_edges_per_view(&edges, budget)
        .map_err(|error| SfmError::InvalidInput(format!("strongest-pairs filter: {error}")))?;
    let mut removed = 0;
    for (pair, keep) in pairs.into_iter().zip(&keep) {
        if !keep {
            let pair_id = workspace
                .pair_graph()
                .pair_id(pair.first(), pair.second())
                .ok_or_else(|| {
                    SfmError::InvariantViolation(format!(
                        "verified pair {pair} is absent from the pair graph"
                    ))
                })?;
            // Prepared graphs represent policy rejection just like failed
            // verification: no accepted model/inliers and an inactive pair.
            workspace.geometric_models.clear(pair_id.index())?;
            workspace.pair_graph_mut().deactivate_pair_and(pair_id)?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests;
