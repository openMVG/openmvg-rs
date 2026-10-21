// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serialized forms of the prepared `ViewGraph` and the `Reconstruction` written to and read
//! from checkpoint files.

use crate::backends::{CameraCenters, RotationEstimates};
use crate::model::ViewRecord;
use crate::model::{Landmarks, PoseRecord, ReconstructionComponent};
use crate::pipelines::incremental::RegistrationFailure;
use crate::two_view::{GeometricModel, PairMatches};
use crate::workspace::store::SparseStore;
use crate::{FeatureSet, Reconstruction, SfmWorkspace, TrackSet, ViewGraph};
use openmvg_core::camera::Camera;
use openmvg_features::model::Keypoint;

/// Pre-reconstruction workspace state shared by both persisted products.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedDto {
    pub root_seed: openmvg_core::RunSeed,
    pub views: Vec<ViewRecord>,
    pub camera_models: Vec<Camera>,
    pub intrinsic_priors: Vec<crate::model::IntrinsicPrior>,
    pub pairs: Vec<PairDto>,
    pub keypoints: Vec<IndexedDto<Vec<Keypoint>>>,
    pub feature_sets: Vec<IndexedDto<FeatureSet>>,
    pub matches: Vec<IndexedDto<PairMatches>>,
    pub geometric_models: Vec<IndexedDto<GeometricModel>>,
    pub tracks: TrackSet,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ViewGraphDto {
    pub prepared: PreparedDto,
    pub calibrated_intrinsics: bool,
    pub extraction_method: Option<crate::ExtractionMethod>,
}

/// Everything a [`Reconstruction`] holds except run telemetry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReconstructionDto {
    pub prepared: PreparedDto,
    pub engine: crate::SfmEngine,
    pub calibrated_intrinsics: bool,
    pub matching_data_released: bool,
    pub rotation_estimates: Option<RotationEstimates>,
    pub camera_centers: Option<CameraCenters>,
    pub poses: Vec<IndexedDto<PoseRecord>>,
    pub landmarks: Landmarks,
    pub components: Vec<ReconstructionComponent>,
    pub registration_failures: Vec<RegistrationFailure>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PairDto {
    pub first: crate::ViewId,
    pub second: crate::ViewId,
    pub active: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct IndexedDto<T> {
    pub index: usize,
    pub value: T,
}

impl PreparedDto {
    fn from_workspace(workspace: &SfmWorkspace) -> Self {
        Self {
            root_seed: workspace.root_seed(),
            views: workspace.views().to_vec(),
            camera_models: workspace.camera_models().to_vec(),
            intrinsic_priors: workspace.intrinsic_priors().to_vec(),
            pairs: workspace
                .pair_graph()
                .pairs()
                .iter()
                .enumerate()
                .map(|(index, pair)| PairDto {
                    first: pair.first(),
                    second: pair.second(),
                    active: workspace.pair_graph().is_active(crate::PairId::new(index)),
                    confidence: workspace
                        .pair_graph()
                        .confidence(crate::PairId::new(index))
                        .unwrap_or(1.0),
                })
                .collect(),
            keypoints: indexed_store(workspace.keypoint_store()),
            feature_sets: indexed_store(workspace.feature_set_store()),
            matches: indexed_store(workspace.match_store()),
            geometric_models: indexed_store(workspace.geometric_model_store()),
            tracks: workspace.tracks().clone(),
        }
    }

    fn into_workspace(self) -> crate::Result<SfmWorkspace> {
        let views = self.views;
        let pairs = self.pairs;
        let mut workspace = SfmWorkspace::builder()
            .root_seed(self.root_seed)
            .image_meta(views.iter().map(|view| view.meta.clone()))
            .camera_models(self.camera_models)
            .intrinsic_priors(self.intrinsic_priors)
            .view_intrinsics(views.iter().map(|view| view.intrinsic_id))
            .view_poses(views.iter().map(|view| view.pose_id))
            .imported_pairs(pairs.iter().map(|pair| (pair.first, pair.second)))
            .build()?;
        // The builder sorts and deduplicates pairs, so only a canonical list keeps each stored
        // state on its own pair.
        let built = workspace.pair_graph().pairs().iter();
        if !pairs
            .iter()
            .map(|pair| (pair.first, pair.second))
            .eq(built.map(|pair| (pair.first(), pair.second())))
        {
            return Err(crate::SfmError::InvalidInput(
                "stored pairs are not sorted and unique".to_owned(),
            ));
        }

        for (index, pair) in pairs.iter().enumerate() {
            let pair_id = crate::PairId::new(index);
            workspace
                .pair_graph_mut()
                .set_confidence(pair_id, pair.confidence)?;
            if !pair.active {
                workspace.pair_graph_mut().deactivate_pair_and(pair_id)?;
            }
        }
        restore_store(&mut workspace.keypoints, self.keypoints)?;
        restore_store(&mut workspace.feature_sets, self.feature_sets)?;
        restore_store(&mut workspace.matches, self.matches)?;
        restore_store(&mut workspace.geometric_models, self.geometric_models)?;
        workspace.tracks = self.tracks;
        Ok(workspace)
    }
}

impl ViewGraphDto {
    pub(crate) fn from_view_graph(view_graph: &ViewGraph) -> Self {
        Self {
            prepared: PreparedDto::from_workspace(view_graph.workspace()),
            calibrated_intrinsics: view_graph.has_calibrated_intrinsics(),
            extraction_method: view_graph.extraction_method(),
        }
    }

    pub(crate) fn into_view_graph(self) -> crate::Result<ViewGraph> {
        ViewGraph::try_from_workspace(self.prepared.into_workspace()?).map(|graph| {
            graph
                .with_calibrated_intrinsics(self.calibrated_intrinsics)
                .with_extraction_method(self.extraction_method)
        })
    }
}

impl ReconstructionDto {
    pub(crate) fn from_reconstruction(reconstruction: &Reconstruction) -> Self {
        let workspace = &reconstruction.workspace;
        Self {
            prepared: PreparedDto::from_workspace(workspace),
            engine: reconstruction.engine(),
            calibrated_intrinsics: reconstruction.has_calibrated_intrinsics(),
            matching_data_released: workspace.is_matching_data_released(),
            rotation_estimates: workspace.rotation_estimates().cloned(),
            camera_centers: workspace.camera_centers().cloned(),
            poses: indexed_store(workspace.pose_store()),
            landmarks: workspace.landmarks().clone(),
            components: workspace.components().to_vec(),
            registration_failures: workspace.registration_failures().to_vec(),
        }
    }

    pub(crate) fn into_reconstruction(self) -> crate::Result<Reconstruction> {
        let mut workspace = self.prepared.into_workspace()?;
        workspace.matching_data_released = self.matching_data_released;
        workspace.rotation_estimates = self.rotation_estimates;
        workspace.camera_centers = self.camera_centers;
        restore_store(&mut workspace.poses, self.poses)?;
        workspace.landmarks = self.landmarks;
        workspace.components = self.components;
        workspace.registration_failures = self.registration_failures;
        workspace.validate()?;
        Ok(Reconstruction::new(
            workspace,
            self.engine,
            self.calibrated_intrinsics,
        ))
    }
}

fn indexed_store<T: Clone>(store: &SparseStore<T>) -> Vec<IndexedDto<T>> {
    store
        .iter_populated()
        .map(|(index, value)| IndexedDto {
            index,
            value: value.clone(),
        })
        .collect()
}

fn restore_store<T>(store: &mut SparseStore<T>, values: Vec<IndexedDto<T>>) -> crate::Result<()> {
    for indexed in values {
        if store.set(indexed.index, indexed.value)?.is_some() {
            return Err(crate::SfmError::InvalidInput(format!(
                "checkpoint contains duplicate sparse-store index {}",
                indexed.index
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageMeta;
    use crate::{Camera, IntrinsicId, ViewId};

    #[test]
    fn round_trip_preserves_shared_intrinsic_and_pose_identity() {
        let workspace = SfmWorkspace::builder()
            .root_seed(7)
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .shared_pose()
            .build()
            .unwrap();
        let graph = ViewGraph::try_from_workspace(workspace)
            .unwrap()
            .with_calibrated_intrinsics(true)
            .with_extraction_method(Some(crate::ExtractionMethod::Alp));

        let restored = ViewGraphDto::from_view_graph(&graph)
            .into_view_graph()
            .unwrap();

        assert_eq!(restored.root_seed(), 7.into());
        assert_eq!(restored.views()[0].pose_id, restored.views()[1].pose_id);
        assert_eq!(restored.camera_models(), graph.camera_models());
        assert!(restored.has_calibrated_intrinsics());
        assert_eq!(
            restored.extraction_method(),
            Some(crate::ExtractionMethod::Alp)
        );
    }

    #[test]
    fn duplicate_sparse_indices_are_rejected() {
        let mut store = SparseStore::new(2);
        let values = vec![
            IndexedDto { index: 0, value: 1 },
            IndexedDto { index: 0, value: 2 },
        ];

        assert!(restore_store(&mut store, values).is_err());
    }

    #[test]
    fn non_canonical_pairs_are_rejected() {
        let workspace = SfmWorkspace::builder()
            .image_meta((0..3).map(|view| ImageMeta::external_id(format!("{view}"), 64, 48)))
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs([
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(1), ViewId::new(2)),
            ])
            .build()
            .unwrap();
        let dto = PreparedDto::from_workspace(&workspace);
        assert!(dto.clone().into_workspace().is_ok());

        let mut reversed = dto.clone();
        reversed.pairs.reverse();
        let mut repeated = dto.clone();
        repeated.pairs[1] = repeated.pairs[0].clone();
        let mut flipped = dto;
        let first = flipped.pairs[0].first;
        flipped.pairs[0].first = flipped.pairs[0].second;
        flipped.pairs[0].second = first;
        for broken in [reversed, repeated, flipped] {
            let error = broken.into_workspace().unwrap_err();
            assert!(
                error.to_string().contains("not sorted and unique"),
                "{error}"
            );
        }
    }
}
