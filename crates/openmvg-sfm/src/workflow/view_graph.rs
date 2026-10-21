// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Immutable verified pre-reconstruction state.

#[cfg(feature = "image-io")]
use super::{Dataset, Frontend};
use crate::TrackSet;
use crate::model::{GraphPair, ViewRecord};
use crate::two_view::PairMatches;
use crate::{Camera, ExtractionMethod, FeatureSet, Keypoint, Result, SfmWorkspace, ViewId};

/// Immutable, verified pre-reconstruction state.
///
/// Built by `ViewGraph::build(&dataset, &frontend)` (with `image-io`), read back with
/// `checkpoint::read_view_graph`, or
/// rebuilt from stored parts with [`Self::try_from_parts`]. This type exposes no mutation and
/// no backend-specific storage.
#[derive(Debug, Clone)]
pub struct ViewGraph {
    pub(crate) workspace: SfmWorkspace,
    calibrated_intrinsics: bool,
    extraction_method: Option<ExtractionMethod>,
}

impl ViewGraph {
    /// Run `frontend` on an image-backed dataset and return the reusable verified graph.
    ///
    /// The same as `frontend.build(dataset)`. Save the result with
    /// [`checkpoint::write_view_graph`](crate::checkpoint::write_view_graph) to run the engines
    /// on it again without repeating the front end.
    #[cfg(feature = "image-io")]
    pub fn build(dataset: &Dataset, frontend: &Frontend) -> Result<Self> {
        frontend.build(dataset)
    }

    /// A graph whose camera models are initial estimates.
    pub(crate) fn try_from_workspace(workspace: SfmWorkspace) -> Result<Self> {
        validate_view_graph_workspace(&workspace)?;
        Ok(Self {
            workspace,
            calibrated_intrinsics: false,
            extraction_method: None,
        })
    }

    pub(crate) const fn with_calibrated_intrinsics(mut self, calibrated: bool) -> Self {
        self.calibrated_intrinsics = calibrated;
        self
    }

    pub(crate) const fn with_extraction_method(
        mut self,
        extraction_method: Option<ExtractionMethod>,
    ) -> Self {
        self.extraction_method = extraction_method;
        self
    }

    pub(crate) fn workspace(&self) -> &SfmWorkspace {
        &self.workspace
    }

    /// Whether the camera models are a known calibration, which engines keep fixed, rather than
    /// initial estimates. Inherited from the [`Dataset`](crate::Dataset).
    #[must_use]
    pub const fn has_calibrated_intrinsics(&self) -> bool {
        self.calibrated_intrinsics
    }

    /// Extraction method the front end extracted the features with; `None` when another
    /// extractor supplied them.
    #[must_use]
    pub const fn extraction_method(&self) -> Option<ExtractionMethod> {
        self.extraction_method
    }

    /// Validate the complete prepared graph and its referenced measurements.
    pub fn validate(&self) -> Result<()> {
        validate_view_graph_workspace(&self.workspace)
    }

    /// Root of the deterministic seed tree inherited from the [`Dataset`](crate::Dataset).
    #[must_use]
    pub const fn root_seed(&self) -> openmvg_core::RunSeed {
        self.workspace.root_seed()
    }

    /// Every view, indexed by [`ViewId`].
    #[must_use]
    pub fn views(&self) -> &[ViewRecord] {
        self.workspace.views()
    }

    /// Camera models, indexed by [`IntrinsicId`](crate::IntrinsicId); radial models may carry calibrated distortion.
    #[must_use]
    pub fn camera_models(&self) -> &[Camera] {
        self.workspace.camera_models()
    }

    /// Retained feature geometry for one view.
    ///
    /// Pixels may retain `f64` precision; each coordinate rounds to its corresponding
    /// `f32` keypoint coordinate.
    #[must_use]
    pub fn features(&self, view: ViewId) -> Option<&FeatureSet> {
        self.workspace.feature_set_store().get(view.index())
    }

    /// Keypoints of one view, which match feature IDs index.
    #[must_use]
    pub fn keypoints(&self, view: ViewId) -> Option<&[Keypoint]> {
        self.workspace
            .keypoint_store()
            .get(view.index())
            .map(Vec::as_slice)
    }

    /// Every selected view pair in increasing order, with its matches and, if accepted, its
    /// verified geometry.
    pub fn pairs(&self) -> impl ExactSizeIterator<Item = GraphPair<'_>> + '_ {
        let workspace = &self.workspace;
        workspace
            .pair_graph()
            .pairs()
            .iter()
            .enumerate()
            .map(move |(index, &views)| GraphPair {
                views,
                matches: workspace
                    .match_store()
                    .get(index)
                    .map_or(&[], PairMatches::as_slice),
                model: workspace.geometric_model_store().get(index),
            })
    }

    /// Number of selected view pairs, including rejected or inactive pairs.
    #[must_use]
    pub fn pair_count(&self) -> usize {
        self.workspace.pair_graph().len()
    }

    /// Number of pairs with accepted geometric evidence.
    #[must_use]
    pub fn verified_pair_count(&self) -> usize {
        self.workspace.geometric_model_store().populated_count()
    }

    /// Deterministically derived feature tracks.
    #[must_use]
    pub fn tracks(&self) -> &TrackSet {
        self.workspace.tracks()
    }
}

fn validate_view_graph_workspace(workspace: &SfmWorkspace) -> Result<()> {
    workspace.validate()?;
    if workspace.feature_store().populated_count() != 0 {
        return Err(crate::SfmError::InvalidInput(
            "ViewGraph keeps keypoints only; descriptors end at the front end".to_owned(),
        ));
    }
    if workspace.is_matching_data_released()
        || workspace.rotation_estimates().is_some()
        || workspace.camera_centers().is_some()
        || workspace.pose_store().populated_count() != 0
        || !workspace.landmarks().is_empty()
        || !workspace.components().is_empty()
        || !workspace.registration_failures().is_empty()
    {
        return Err(crate::SfmError::InvalidInput(
            "ViewGraph accepts verified pre-reconstruction state only".to_owned(),
        ));
    }
    // Every selected pair was matched, and verification either accepted it or deactivated it.
    let pair_graph = workspace.pair_graph();
    for (index, pair) in pair_graph.pairs().iter().enumerate() {
        if workspace.match_store().get(index).is_none() {
            return Err(crate::SfmError::InvalidInput(format!(
                "pair {pair} has no match record"
            )));
        }
        let verified = workspace.geometric_model_store().get(index).is_some();
        if pair_graph.is_active(crate::PairId::new(index)) != verified {
            return Err(crate::SfmError::InvalidInput(if verified {
                format!("pair {pair} keeps a verified model but is inactive")
            } else {
                format!("pair {pair} is active without a verified model")
            }));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageMeta, PoseRecord};
    use crate::{Camera, IntrinsicId, PoseId, WorldToCameraPose};

    #[test]
    fn reconstruction_state_cannot_become_a_view_graph() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([ImageMeta::external_id("a", 64, 48)])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        workspace
            .set_pose(PoseRecord {
                id: PoseId::new(0),
                pose: WorldToCameraPose::identity(),
                component: None,
            })
            .unwrap();

        assert!(ViewGraph::try_from_workspace(workspace).is_err());
    }
}
