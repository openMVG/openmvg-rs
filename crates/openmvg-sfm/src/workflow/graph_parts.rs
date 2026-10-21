// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Assembly of a [`ViewGraph`] from stored parts.
//!
//! ```text
//! ViewGraphParts (cameras, views, keypoints, pairs)
//!        |  check view IDs, pair order, match order, mask lengths
//!        v
//! workspace: feature geometry from keypoints, matches, accepted models
//!        |  rejected pairs inactive; missing essential poses by cheirality
//!        v
//! tracks from the accepted inliers  ->  ViewGraph (validated)
//! ```

use nalgebra::{Matrix3, Point2};
use openmvg_core::RunSeed;
use openmvg_core::model::RelativePose;
use openmvg_features::model::{FeatureMatch, Keypoint};
use openmvg_geometry::correspondence::BearingCorrespondence;
use openmvg_geometry::essential::EssentialMatrix;

use super::ViewGraph;
use crate::adapters::verification::unit_bearing;
use crate::model::{PairGeometry, ViewRecord};
use crate::two_view::{InlierMask, ModelEvidence, PairMatches};
use crate::workspace::builder::SfmWorkspaceBuilder;
use crate::{Camera, FeatureSet, PairId, Result, SfmError, SfmWorkspace, ViewPair};

/// The stored content of a [`ViewGraph`], from which [`ViewGraph::try_from_parts`] rebuilds it.
///
/// It holds what a graph file records: camera models, views, the keypoints of every view, and
/// the putative matches of every pair with the verified geometry of the accepted ones. Feature
/// geometry and tracks are derived on assembly.
#[derive(Debug, Clone)]
#[must_use = "the parts have no effect until ViewGraph::try_from_parts is called"]
pub struct ViewGraphParts {
    root_seed: RunSeed,
    cameras: Vec<Camera>,
    views: Vec<ViewRecord>,
    keypoints: Vec<Vec<Keypoint>>,
    pairs: Vec<PartPair>,
    calibrated_intrinsics: bool,
}

#[derive(Debug, Clone)]
struct PartPair {
    views: ViewPair,
    matches: Vec<FeatureMatch>,
    verified: Option<(PairGeometry, Vec<bool>)>,
}

impl ViewGraphParts {
    /// Parts with no pairs yet: camera models indexed by
    /// [`IntrinsicId`](crate::IntrinsicId), views indexed by [`ViewId`](crate::ViewId), and the
    /// keypoints of each view in view order.
    pub fn new(
        root_seed: impl Into<RunSeed>,
        cameras: Vec<Camera>,
        views: Vec<ViewRecord>,
        keypoints: Vec<Vec<Keypoint>>,
    ) -> Self {
        Self {
            root_seed: root_seed.into(),
            cameras,
            views,
            keypoints,
            pairs: Vec::new(),
            calibrated_intrinsics: false,
        }
    }

    /// Add one pair: its putative matches, sorted by `(first, second)` feature and unique, and
    /// for an accepted pair its geometry with one inlier flag per match.
    ///
    /// Pairs must be added in increasing order.
    pub fn with_pair(
        mut self,
        views: ViewPair,
        matches: Vec<FeatureMatch>,
        verified: Option<(PairGeometry, Vec<bool>)>,
    ) -> Self {
        self.pairs.push(PartPair {
            views,
            matches,
            verified,
        });
        self
    }

    /// Mark the camera models as a known calibration, which engines keep fixed, rather than
    /// initial estimates. Off by default.
    pub const fn with_calibrated_intrinsics(mut self, calibrated: bool) -> Self {
        self.calibrated_intrinsics = calibrated;
        self
    }
}

impl ViewGraph {
    /// Rebuild a graph from its stored parts, deriving feature geometry and tracks.
    ///
    /// Rejected pairs stay in the graph inactive. An essential pair stored without its relative
    /// pose gets the decomposition of its matrix that places the most inliers in front of both
    /// cameras.
    ///
    /// # Errors
    ///
    /// Fails like [`crate::DatasetBuilder::build`] if the camera models or their assignment do
    /// not match the views. Otherwise fails with [`SfmError::InvalidInput`] if a view ID differs
    /// from its position or the keypoint lists do not match the views; and, naming the pair, if
    /// pairs are out of order, repeated, or reference unknown views, matches are unsorted,
    /// repeated, or reference unknown features, an inlier mask does not have one flag per
    /// match, a geometry is invalid, or a missing essential pose cannot be recovered.
    pub fn try_from_parts(parts: ViewGraphParts) -> Result<Self> {
        let ViewGraphParts {
            root_seed,
            cameras,
            views,
            keypoints,
            pairs,
            calibrated_intrinsics,
        } = parts;
        check_parts(&views, &keypoints, &pairs)?;

        let mut workspace = SfmWorkspaceBuilder::new()
            .root_seed(root_seed)
            .image_meta(views.iter().map(|view| view.meta.clone()))
            .camera_models(cameras)
            .view_intrinsics(views.iter().map(|view| view.intrinsic_id))
            .view_poses(views.iter().map(|view| view.pose_id))
            .imported_pairs(pairs.iter().map(|pair| pair.views))
            .build()?;
        for (index, view_keypoints) in keypoints.into_iter().enumerate() {
            let pixels = view_keypoints
                .iter()
                .map(|keypoint| Point2::new(f64::from(keypoint.x()), f64::from(keypoint.y())))
                .collect();
            workspace
                .feature_sets
                .set(index, FeatureSet::try_new(pixels)?)?;
            workspace.keypoints.set(index, view_keypoints)?;
        }
        // The pairs are strictly increasing, so the pair graph kept their order.
        for (index, pair) in pairs.into_iter().enumerate() {
            add_pair(&mut workspace, PairId::new(index), pair)?;
        }
        crate::ops::tracks::build_tracks_with_progress(&mut workspace, &())?;
        Ok(Self::try_from_workspace(workspace)?.with_calibrated_intrinsics(calibrated_intrinsics))
    }
}

fn check_parts(
    views: &[ViewRecord],
    keypoints: &[Vec<Keypoint>],
    pairs: &[PartPair],
) -> Result<()> {
    if let Some((index, view)) = views
        .iter()
        .enumerate()
        .find(|(index, view)| view.id.index() != *index)
    {
        return Err(SfmError::InvalidInput(format!(
            "view {} is stored at position {index}; view IDs must be dense and in order",
            view.id
        )));
    }
    if keypoints.len() != views.len() {
        return Err(SfmError::InvalidInput(format!(
            "got {} keypoint lists for {} views",
            keypoints.len(),
            views.len()
        )));
    }
    let mut previous: Option<ViewPair> = None;
    for pair in pairs {
        let views_in_pair = pair.views;
        if views_in_pair.second().index() >= views.len() {
            return Err(pair_error(views_in_pair, "references an unknown view"));
        }
        if views_in_pair.first() == views_in_pair.second() {
            return Err(pair_error(views_in_pair, "pairs a view with itself"));
        }
        if previous.is_some_and(|previous| previous >= views_in_pair) {
            return Err(pair_error(
                views_in_pair,
                "is repeated or out of order; pairs must be strictly increasing",
            ));
        }
        previous = Some(views_in_pair);
        if pair.matches.windows(2).any(|window| {
            (window[0].first(), window[0].second()) >= (window[1].first(), window[1].second())
        }) {
            return Err(pair_error(
                views_in_pair,
                "has matches that are not strictly increasing by feature pair",
            ));
        }
        if let Some((_, inliers)) = &pair.verified
            && inliers.len() != pair.matches.len()
        {
            return Err(pair_error(
                views_in_pair,
                &format!(
                    "has {} inlier flags for {} matches",
                    inliers.len(),
                    pair.matches.len()
                ),
            ));
        }
    }
    Ok(())
}

fn add_pair(workspace: &mut SfmWorkspace, pair_id: PairId, pair: PartPair) -> Result<()> {
    let PartPair {
        views,
        matches,
        verified,
    } = pair;
    let named = |error| in_pair(views, error);
    let matches = PairMatches::try_new(views, matches).map_err(named)?;
    let feature_count = |view: crate::ViewId| {
        workspace
            .keypoint_store()
            .get(view.index())
            .map_or(0, Vec::len)
    };
    matches
        .validate_for_pair(
            views,
            feature_count(views.first()),
            feature_count(views.second()),
        )
        .map_err(named)?;
    match verified {
        Some((geometry, inliers)) => {
            let model = geometry
                .into_model(
                    ModelEvidence::from_inliers(InlierMask::new(inliers)),
                    |matrix, inliers| {
                        essential_pose(workspace, views, matches.as_slice(), inliers, matrix)
                    },
                )
                .and_then(|model| model.validate_for_matches(&matches).map(|()| model))
                .map_err(named)?;
            workspace.geometric_models.set(pair_id.index(), model)?;
        }
        None => workspace.pair_graph_mut().deactivate_pair_and(pair_id)?,
    }
    workspace.matches.set(pair_id.index(), matches)?;
    Ok(())
}

/// The decomposition of `matrix` that places the most inlier bearings in front of both cameras.
fn essential_pose(
    workspace: &SfmWorkspace,
    views: ViewPair,
    matches: &[FeatureMatch],
    inliers: &[bool],
    matrix: &Matrix3<f64>,
) -> Result<RelativePose> {
    let essential = EssentialMatrix::try_new(*matrix).map_err(|error| {
        SfmError::InvalidInput(format!("the essential matrix is invalid: {error}"))
    })?;
    let bearing = |view: crate::ViewId, feature: crate::FeatureId| -> Result<_> {
        let record = workspace.view(view).ok_or(SfmError::InvalidId {
            kind: crate::ViewId::kind(),
            index: view.index(),
        })?;
        let camera = workspace
            .camera_model(record.intrinsic_id)
            .ok_or(SfmError::InvalidId {
                kind: crate::IntrinsicId::kind(),
                index: record.intrinsic_id.index(),
            })?;
        let keypoint = workspace
            .keypoint_store()
            .get(view.index())
            .and_then(|keypoints| keypoints.get(feature.index()))
            .ok_or(SfmError::InvalidId {
                kind: crate::FeatureId::kind(),
                index: feature.index(),
            })?;
        unit_bearing(
            camera,
            Point2::new(f64::from(keypoint.x()), f64::from(keypoint.y())),
        )
    };
    let bearings = matches
        .iter()
        .zip(inliers)
        .filter(|(_, inlier)| **inlier)
        .map(|(matched, _)| {
            BearingCorrespondence::try_new(
                bearing(views.first(), matched.first())?,
                bearing(views.second(), matched.second())?,
            )
            .map_err(|error| {
                SfmError::InvalidInput(format!("failed to form bearing correspondence: {error}"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    essential
        .decompose_with_cheirality(&bearings)
        .ok_or_else(|| {
            SfmError::InvalidInput(
            "no decomposition of the essential matrix places the inliers in front of both cameras"
                .to_owned(),
        )
        })
}

fn pair_error(pair: ViewPair, message: &str) -> SfmError {
    SfmError::InvalidInput(format!("pair {pair} {message}"))
}

/// `error`, reported as invalid input of `pair`.
fn in_pair(pair: ViewPair, error: SfmError) -> SfmError {
    let message = match error {
        SfmError::InvalidInput(message) => message,
        SfmError::InvalidFeatureData(error) => error.to_string(),
        SfmError::InvalidEstimationInput(error) => error.to_string(),
        other => other.to_string(),
    };
    SfmError::InvalidInput(format!("pair {pair}: {message}"))
}

#[cfg(test)]
mod tests;
