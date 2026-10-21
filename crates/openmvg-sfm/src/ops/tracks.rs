// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Track building stage: chain the inlier matches of verified pairs into multi-view tracks.

use std::collections::BTreeMap;

use nalgebra::Point2;
use openmvg_features::model::Keypoint;
use openmvg_features::tracking::{
    ScaleDispersionConfig, ScoredMatch, build_tracks_scored_owned, evaluate_scales,
};

use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::{
    FeatureId, Result, SfmError, SfmWorkspace, Track, TrackId, TrackObservation, TrackSet,
};

/// Outcome of one track-building stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackBuildReport {
    /// Tracks built.
    pub track_count: usize,
    /// Observations across all built tracks.
    pub observation_count: usize,
    /// Candidate tracks dropped because they had fewer than two observations.
    pub rejected_singleton_count: usize,
    /// FeatureMatch edges rejected because merging their components would introduce
    /// two observations from the same image.
    pub rejected_conflict_count: usize,
    /// Tracks rejected by the scale-dispersion filter (likely feature-merge
    /// errors with inconsistent SIFT scales).
    pub rejected_scale_dispersion_count: usize,
}

pub fn build_tracks(workspace: &mut SfmWorkspace) -> Result<TrackBuildReport> {
    build_tracks_with_progress(workspace, &())
}

/// [`build_tracks`], reporting [`SfmProgressStage::TrackBuilding`] in active image pairs.
pub(crate) fn build_tracks_with_progress(
    workspace: &mut SfmWorkspace,
    progress: &dyn SfmProgressSink,
) -> Result<TrackBuildReport> {
    if workspace.is_matching_data_released() {
        return Err(SfmError::MatchingDataReleased);
    }
    let pair_count = workspace.pair_graph().active_pair_ids().count();
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::TrackBuilding, pair_count);
    let mut scored_matches = Vec::new();
    let mut canonical_features = BTreeMap::new();

    for pair_id in workspace.pair_graph().active_pair_ids() {
        cancellation_point(progress)?;
        let pair = workspace
            .pair_graph()
            .pair(pair_id)
            .ok_or(SfmError::InvalidId {
                kind: crate::PairId::kind(),
                index: pair_id.index(),
            })?;
        let keypoints_i = workspace.keypoint_store().get(pair.first().index()).ok_or(
            SfmError::MissingPrerequisite {
                prerequisite: "keypoints",
                method: "extract_features",
            },
        )?;
        let keypoints_j = workspace
            .keypoint_store()
            .get(pair.second().index())
            .ok_or(SfmError::MissingPrerequisite {
                prerequisite: "keypoints",
                method: "extract_features",
            })?;
        let matches =
            workspace
                .match_store()
                .get(pair_id.index())
                .ok_or(SfmError::MissingPrerequisite {
                    prerequisite: "matches",
                    method: "match_pairs",
                })?;
        let model = workspace
            .geometric_model_store()
            .get(pair_id.index())
            .ok_or(SfmError::MissingPrerequisite {
                prerequisite: "geometric models",
                method: "verify_pairs",
            })?;
        model.validate_for_matches(matches)?;
        for (view, keypoints) in [(pair.first(), keypoints_i), (pair.second(), keypoints_j)] {
            canonical_features
                .entry(view)
                .or_insert_with(|| canonical_feature_ids(keypoints));
        }
        let canonical_i = &canonical_features[&pair.first()];
        let canonical_j = &canonical_features[&pair.second()];

        for (match_index, matched) in matches.as_slice().iter().enumerate() {
            if !model.evidence().inliers.as_slice()[match_index] {
                continue;
            }
            feature_xy(keypoints_i, matched.first())?;
            feature_xy(keypoints_j, matched.second())?;
            scored_matches.push(
                ScoredMatch::try_new(
                    TrackObservation::new(pair.first(), canonical_i[matched.first().index()]),
                    TrackObservation::new(pair.second(), canonical_j[matched.second().index()]),
                    match_quality(matched.distance()),
                )
                .map_err(|error| SfmError::InvalidInput(error.to_string()))?,
            );
        }
        progress.advance(SfmProgressStage::TrackBuilding, 1);
    }

    let outcome = build_tracks_scored_owned(scored_matches);

    let mut raw_tracks = Vec::new();
    let mut rejected_singletons = 0;
    let mut rejected_scale_dispersion = 0;
    let scale_dispersion_config = ScaleDispersionConfig::default();
    for group in outcome.tracks {
        if group.len() < 2 {
            rejected_singletons += 1;
            continue;
        }
        let scales: Vec<f64> = group
            .iter()
            .map(|observation| {
                let keypoints = workspace
                    .keypoint_store()
                    .get(observation.view.index())
                    .ok_or(SfmError::MissingPrerequisite {
                        prerequisite: "keypoints",
                        method: "extract_features",
                    })?;
                feature_scale(keypoints, observation.feature)
            })
            .collect::<Result<_>>()?;
        if evaluate_scales(&scales, &scale_dispersion_config).reject {
            rejected_scale_dispersion += 1;
            continue;
        }
        raw_tracks.push(group);
    }

    let mut tracks = Vec::with_capacity(raw_tracks.len());
    let mut observation_count = 0;
    for (track_index, group) in raw_tracks.into_iter().enumerate() {
        observation_count += group.len();
        tracks.push(Track::try_new(TrackId::new(track_index), group)?);
    }

    let report = TrackBuildReport {
        track_count: tracks.len(),
        observation_count,
        rejected_singleton_count: rejected_singletons,
        rejected_conflict_count: outcome.rejected_conflict_count,
        rejected_scale_dispersion_count: rejected_scale_dispersion,
    };
    workspace.tracks = TrackSet::try_new(tracks)?;
    progress.finish(SfmProgressStage::TrackBuilding);
    Ok(report)
}

/// Address every feature by the lowest feature id at its bit-identical pixel position.
///
/// Multi-orientation detectors emit one keypoint per dominant orientation at the same pixel, and
/// their descriptors are matched separately. One pixel in one view is one measurement: without
/// this mapping it would become several tracks of the same scene point, which double-weights its
/// residuals and duplicates the constraints that decide whether a camera is observable.
fn canonical_feature_ids(keypoints: &[Keypoint]) -> Vec<FeatureId> {
    let mut first_at_pixel = BTreeMap::new();
    keypoints
        .iter()
        .enumerate()
        .map(|(index, keypoint)| {
            *first_at_pixel
                .entry((keypoint.x().to_bits(), keypoint.y().to_bits()))
                .or_insert(FeatureId::new(index))
        })
        .collect()
}

fn match_quality(distance: Option<f32>) -> f32 {
    distance.map_or(f32::MIN, |value| -value)
}

fn feature_xy(keypoints: &[Keypoint], feature_id: FeatureId) -> Result<Point2<f64>> {
    let keypoint = keypoints
        .get(feature_id.index())
        .ok_or(SfmError::InvalidId {
            kind: FeatureId::kind(),
            index: feature_id.index(),
        })?;
    Ok(Point2::new(keypoint.x() as f64, keypoint.y() as f64))
}

fn feature_scale(keypoints: &[Keypoint], feature_id: FeatureId) -> Result<f64> {
    keypoints
        .get(feature_id.index())
        .map(|keypoint| keypoint.scale() as f64)
        .ok_or(SfmError::InvalidId {
            kind: FeatureId::kind(),
            index: feature_id.index(),
        })
}

#[cfg(test)]
mod tests {
    use super::match_quality;

    #[test]
    fn descriptor_distance_quality_orders_smaller_first() {
        assert!(match_quality(Some(1.0)) > match_quality(Some(2.0)));
        assert!(match_quality(Some(2.0)) > match_quality(None));
    }
}
