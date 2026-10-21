// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mutable SfM state behind the workflow values.
//!
//! [`SfmWorkspace`] stores every stage's results in one place: views, features, matches, the
//! pair graph, tracks, poses, and landmarks. The stages in `ops` and the engines in `pipelines`
//! read and update it. The public API wraps it in the immutable `workflow` values, so a caller
//! sees only the state of the current phase.

pub(crate) mod builder;
pub(crate) mod pair_graph;
pub(crate) mod quality;
pub(crate) mod reports;
pub(crate) mod selector;
pub(crate) mod store;
pub(crate) mod validate;
pub(crate) mod views;

use crate::SfmError;
use crate::TrackSet;
use crate::ba::backend::BundleAdjustmentBackend;
use crate::ba::config::BundleAdjustConfig;
use crate::backends::{
    CameraCenters, FeatureExtractor, GeometricVerifier, PairMatcher, PairProposer,
    RotationEstimates, Triangulator,
};
use crate::config::GaugeScalePolicy;
use crate::model::{Landmarks, PoseRecord, ReconstructionComponent, ViewRecord};
use crate::pipelines::incremental::RegistrationFailure;
use crate::progress::SfmProgressSink;
use crate::two_view::{GeometricModel, PairMatches};
use crate::workspace::builder::SfmWorkspaceBuilder;
use crate::workspace::pair_graph::PairGraph;
use crate::workspace::reports::WorkspaceReports;
use crate::workspace::selector::{PairSelector, ViewSelector};
use crate::workspace::store::SparseStore;
use crate::workspace::views::{FeaturesView, LandmarkSet, MatchesView, PairModelsView, PoseSet};
use crate::{FeatureId, FeatureSet, IntrinsicId, PoseId, Result, ViewId, ops};
use openmvg_core::RunSeed;
use openmvg_core::camera::Camera;
use openmvg_features::model::{ExtractedFeatures, Keypoint};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

/// Mutable state for feature extraction, matching, verification, and reconstruction.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # fn main() -> Result<(), openmvg_sfm::SfmError> {
/// use openmvg_sfm::model::ImageMeta;
/// use openmvg_sfm::{Camera, IntrinsicId, unstable::SfmWorkspace};
///
/// let workspace = SfmWorkspace::builder()
///     .image_meta(vec![ImageMeta::external_id("a", 640, 480)])
///     .camera_models([Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?])
///     .shared_intrinsic(IntrinsicId::new(0))
///     .build()?;
/// assert_eq!(workspace.views().len(), 1);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "unstable-internals"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct SfmWorkspace {
    pub(crate) run_seed: RunSeed,
    pub(crate) max_parallel_workers: Option<NonZeroUsize>,
    views: Vec<ViewRecord>,
    camera_models: Vec<Camera>,
    intrinsic_priors: Vec<crate::model::IntrinsicPrior>,
    pair_graph: PairGraph,
    pub(crate) features: SparseStore<ExtractedFeatures>,
    pub(crate) keypoints: SparseStore<Vec<Keypoint>>,
    pub(crate) feature_sets: SparseStore<FeatureSet>,
    pub(crate) matches: SparseStore<PairMatches>,
    pub(crate) geometric_models: SparseStore<GeometricModel>,
    pub(crate) matching_data_released: bool,
    pub(crate) rotation_estimates: Option<RotationEstimates>,
    pub(crate) camera_centers: Option<CameraCenters>,
    pub(crate) poses: SparseStore<PoseRecord>,
    pub(crate) tracks: TrackSet,
    pub(crate) landmarks: Landmarks,
    pub(crate) components: Vec<ReconstructionComponent>,
    pub(crate) registration_failures: Vec<RegistrationFailure>,
    pub(crate) reports: WorkspaceReports,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl SfmWorkspace {
    /// Start building a workspace from image metadata, camera models, and optional pairs.
    pub fn builder() -> SfmWorkspaceBuilder {
        SfmWorkspaceBuilder::new()
    }

    pub(crate) fn from_parts(
        root_seed: RunSeed,
        max_parallel_workers: Option<NonZeroUsize>,
        views: Vec<ViewRecord>,
        camera_models: Vec<Camera>,
        pair_graph: PairGraph,
    ) -> Self {
        let view_count = views.len();
        let pose_count = views
            .iter()
            .map(|view| view.pose_id)
            .collect::<BTreeSet<_>>()
            .len();
        let pair_count = pair_graph.len();
        Self {
            run_seed: root_seed,
            max_parallel_workers,
            features: SparseStore::new(view_count),
            keypoints: SparseStore::new(view_count),
            feature_sets: SparseStore::new(view_count),
            matches: SparseStore::new(pair_count),
            geometric_models: SparseStore::new(pair_count),
            matching_data_released: false,
            rotation_estimates: None,
            camera_centers: None,
            poses: SparseStore::new(pose_count),
            tracks: TrackSet::default(),
            landmarks: Landmarks::default(),
            components: Vec::new(),
            registration_failures: Vec::new(),
            reports: WorkspaceReports::default(),
            views,
            intrinsic_priors: vec![crate::model::IntrinsicPrior::default(); camera_models.len()],
            camera_models,
            pair_graph,
        }
    }

    /// View table. Every view references one intrinsic and one pose.
    pub fn views(&self) -> &[crate::model::ViewRecord] {
        &self.views
    }

    /// Root of the deterministic seed tree used by randomized pipeline stages.
    pub const fn root_seed(&self) -> RunSeed {
        self.run_seed
    }

    /// Caller-selected ceiling for memory-sensitive parallel stages.
    pub const fn max_parallel_workers(&self) -> Option<NonZeroUsize> {
        self.max_parallel_workers
    }

    pub(crate) fn derive_seed(&self, domain: &str, ids: &[u64]) -> RunSeed {
        self.run_seed.derive(domain, ids)
    }

    /// Camera models, indexed by [`IntrinsicId`].
    pub fn camera_models(&self) -> &[Camera] {
        &self.camera_models
    }

    /// Camera model for one intrinsic group.
    pub fn camera_model(&self, intrinsic_id: IntrinsicId) -> Option<&Camera> {
        self.camera_models.get(intrinsic_id.index())
    }

    /// Focal-length prior of each intrinsic group, indexed by [`IntrinsicId`]. Groups are
    /// [`Trusted`](crate::model::IntrinsicPrior::Trusted) unless the builder says otherwise.
    pub fn intrinsic_priors(&self) -> &[crate::model::IntrinsicPrior] {
        &self.intrinsic_priors
    }

    /// Focal-length prior of one intrinsic group.
    pub fn intrinsic_prior(
        &self,
        intrinsic_id: IntrinsicId,
    ) -> Option<crate::model::IntrinsicPrior> {
        self.intrinsic_priors.get(intrinsic_id.index()).copied()
    }

    pub(crate) fn set_intrinsic_priors(&mut self, priors: Vec<crate::model::IntrinsicPrior>) {
        debug_assert_eq!(priors.len(), self.camera_models.len());
        self.intrinsic_priors = priors;
    }

    /// Replace one camera model after validating it against every assigned view.
    pub fn replace_camera_model(
        &mut self,
        intrinsic_id: IntrinsicId,
        camera: Camera,
    ) -> Result<()> {
        self.replace_camera_model_groups(vec![(intrinsic_id, camera)])
    }

    /// Transactionally replace the complete camera-model table.
    pub fn replace_camera_models(&mut self, camera_models: Vec<Camera>) -> Result<()> {
        if camera_models.len() != self.camera_models.len() {
            return Err(crate::SfmError::InvalidInput(format!(
                "expected {} camera models, got {}",
                self.camera_models.len(),
                camera_models.len()
            )));
        }
        self.replace_camera_model_groups(
            camera_models
                .into_iter()
                .enumerate()
                .map(|(index, camera)| (IntrinsicId::new(index), camera))
                .collect(),
        )
    }

    /// Transactionally replace a subset of camera-model groups.
    ///
    /// Every identifier and camera domain is checked before any slot changes. This is the
    /// commit primitive for scoped bundle adjustment, whose solution may contain only the
    /// intrinsic groups touched by its optimization problem.
    pub fn replace_camera_model_groups(
        &mut self,
        replacements: Vec<(IntrinsicId, Camera)>,
    ) -> Result<()> {
        let mut seen = BTreeSet::new();
        for (intrinsic_id, camera) in &replacements {
            if !seen.insert(*intrinsic_id) {
                return Err(crate::SfmError::InvalidInput(format!(
                    "duplicate replacement for camera-model group {intrinsic_id}"
                )));
            }
            if intrinsic_id.index() >= self.camera_models.len() {
                return Err(crate::SfmError::InvalidId {
                    kind: IntrinsicId::kind(),
                    index: intrinsic_id.index(),
                });
            }
            validate_camera_for_assigned_views(&self.views, *intrinsic_id, camera)?;
        }
        for (intrinsic_id, camera) in replacements {
            self.camera_models[intrinsic_id.index()] = camera;
        }
        Ok(())
    }

    pub(crate) fn restore_camera_models(&mut self, camera_models: Vec<Camera>) {
        debug_assert_eq!(camera_models.len(), self.camera_models.len());
        self.camera_models = camera_models;
    }

    /// Candidate view pairs with their activation state and confidence.
    pub fn pair_graph(&self) -> &PairGraph {
        &self.pair_graph
    }

    /// Mutable pair graph, for adjusting pair confidence or deactivating pairs.
    pub fn pair_graph_mut(&mut self) -> &mut PairGraph {
        &mut self.pair_graph
    }

    /// Extracted keypoints and descriptors for matching and verification, one slot per view
    /// index; dropped when the front end builds a `ViewGraph` and emptied by
    /// [`Self::release_matching_data`].
    pub fn feature_store(&self) -> &SparseStore<ExtractedFeatures> {
        &self.features
    }

    /// Extracted keypoints with their scale and orientation, one slot per view index; kept
    /// without the descriptors in a `ViewGraph` and emptied by [`Self::release_matching_data`].
    pub fn keypoint_store(&self) -> &SparseStore<Vec<Keypoint>> {
        &self.keypoints
    }

    /// Store one view's extracted features and their keypoints.
    pub(crate) fn set_features(&mut self, index: usize, features: ExtractedFeatures) -> Result<()> {
        self.keypoints.set(index, features.keypoints().to_vec())?;
        self.features.set(index, features)?;
        Ok(())
    }

    /// Drop every descriptor, keeping the keypoints that tracks and engines read.
    pub(crate) fn release_descriptors(&mut self) -> Result<()> {
        for index in 0..self.features.slot_count() {
            self.features.clear(index)?;
        }
        Ok(())
    }

    /// Persistent feature geometry retained for track and landmark observations.
    pub fn feature_set_store(&self) -> &SparseStore<FeatureSet> {
        &self.feature_sets
    }

    /// Resolve a track observation through the per-view feature set.
    pub fn track_observation_pixel(
        &self,
        observation: &crate::TrackObservation,
    ) -> Result<nalgebra::Point2<f64>> {
        self.feature_sets
            .get(observation.view.index())
            .ok_or(SfmError::MissingPrerequisite {
                prerequisite: "feature geometry",
                method: "extract_features or import tracks with feature sets",
            })?
            .pixel(observation.feature)
            .map_err(|error| match error {
                openmvg_features::model::FeatureDataError::FeatureOutOfRange {
                    feature, ..
                } => SfmError::InvalidId {
                    kind: FeatureId::kind(),
                    index: feature.index(),
                },
                other => other.into(),
            })
    }

    /// Read-only, view-ID-keyed access to the extracted features.
    pub fn features_view(&self) -> FeaturesView<'_> {
        FeaturesView::new(&self.features)
    }

    /// Putative feature matches, one slot per pair index; emptied by
    /// [`Self::release_matching_data`].
    pub fn match_store(&self) -> &SparseStore<PairMatches> {
        &self.matches
    }

    /// Read-only, view-pair-keyed access to the putative matches.
    pub fn matches_view(&self) -> MatchesView<'_> {
        MatchesView::new(&self.matches, &self.pair_graph)
    }

    /// Accepted two-view geometric models, one slot per pair index.
    pub fn geometric_model_store(&self) -> &SparseStore<GeometricModel> {
        &self.geometric_models
    }

    /// Read-only, view-pair-keyed access to the accepted two-view models.
    pub fn geometric_models_view(&self) -> PairModelsView<'_> {
        PairModelsView::new(&self.geometric_models, &self.pair_graph)
    }

    /// Whether [`Self::release_matching_data`] has released features and matches; the
    /// front-end stages then fail with `SfmError::MatchingDataReleased`.
    pub const fn is_matching_data_released(&self) -> bool {
        self.matching_data_released
    }

    /// Absolute rotations from global rotation averaging, or `None` before the Global engine
    /// has averaged them.
    pub fn rotation_estimates(&self) -> Option<&RotationEstimates> {
        self.rotation_estimates.as_ref()
    }

    /// Camera centers from global translation averaging, or `None` before the Global engine
    /// has averaged them.
    pub fn camera_centers(&self) -> Option<&CameraCenters> {
        self.camera_centers.as_ref()
    }

    /// Registered poses, one slot per [`PoseId`] index; an empty slot is an unregistered pose.
    pub fn pose_store(&self) -> &SparseStore<PoseRecord> {
        &self.poses
    }

    /// Pose identity assigned to a view.
    pub fn pose_id(&self, view: crate::ViewId) -> Option<PoseId> {
        self.view(view).map(|record| record.pose_id)
    }

    /// Registered pose for a view, resolved through its independent [`PoseId`].
    pub fn pose(&self, view: crate::ViewId) -> Option<&PoseRecord> {
        let pose = self.pose_id(view)?;
        self.poses.get(pose.index())
    }

    /// Registered pose by pose identity.
    pub fn pose_by_id(&self, pose: PoseId) -> Option<&PoseRecord> {
        self.poses.get(pose.index())
    }

    /// Whether a view's assigned pose has been registered.
    pub fn has_pose(&self, view: crate::ViewId) -> bool {
        self.pose(view).is_some()
    }

    /// Number of views whose assigned pose has been registered.
    pub fn registered_view_count(&self) -> usize {
        self.views
            .iter()
            .filter(|view| self.poses.has(view.pose_id.index()))
            .count()
    }

    /// Registered views in deterministic view-ID order.
    pub fn registered_views(&self) -> impl Iterator<Item = (crate::ViewId, &PoseRecord)> {
        self.views.iter().filter_map(|view| {
            self.poses
                .get(view.pose_id.index())
                .map(|pose| (view.id, pose))
        })
    }

    /// Read-only pose table keyed by [`PoseId`].
    pub fn poses(&self) -> PoseSet<'_> {
        PoseSet::new(&self.poses)
    }

    /// Multi-view feature tracks; empty until [`Self::build_tracks`] runs or the builder imports
    /// tracks.
    pub fn tracks(&self) -> &TrackSet {
        &self.tracks
    }

    /// Every landmark, including rejected ones kept so that landmark IDs stay stable.
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    /// Read-only, landmark-ID-keyed access to the landmarks.
    pub fn landmarks_view(&self) -> LandmarkSet<'_> {
        LandmarkSet::new(&self.landmarks)
    }

    /// Connected, separately gauged parts of the current reconstruction.
    pub fn components(&self) -> &[ReconstructionComponent] {
        &self.components
    }

    /// Registration attempts that did not produce a pose, in recording order.
    pub fn registration_failures(&self) -> &[RegistrationFailure] {
        &self.registration_failures
    }

    /// Stage and quality reports recorded in this workspace.
    pub fn reports(&self) -> &WorkspaceReports {
        &self.reports
    }

    pub(crate) fn record_registration_failure(&mut self, failure: RegistrationFailure) {
        self.registration_failures.push(failure);
    }

    pub(crate) fn set_global_averaging(
        &mut self,
        rotations: RotationEstimates,
        centers: CameraCenters,
    ) {
        self.rotation_estimates = Some(rotations);
        self.camera_centers = Some(centers);
    }

    pub(crate) fn set_pose(&mut self, pose: PoseRecord) -> Result<()> {
        self.poses.set(pose.id.index(), pose)?;
        Ok(())
    }

    pub(crate) fn set_pose_for_view(
        &mut self,
        view: crate::ViewId,
        pose: openmvg_core::model::WorldToCameraPose,
        component: Option<crate::ComponentId>,
    ) -> Result<()> {
        let pose_id = self.pose_id(view).ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: view.index(),
        })?;
        self.set_pose(PoseRecord {
            id: pose_id,
            pose,
            component,
        })
    }

    pub(crate) fn clear_poses_for_views(&mut self, views: &[ViewId]) -> Result<()> {
        let requested = views.iter().copied().collect::<BTreeSet<_>>();
        let pose_ids = views
            .iter()
            .map(|view| {
                self.pose_id(*view).ok_or(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: view.index(),
                })
            })
            .collect::<Result<BTreeSet<_>>>()?;
        if let Some(view) = self
            .views
            .iter()
            .find(|view| pose_ids.contains(&view.pose_id) && !requested.contains(&view.id))
        {
            return Err(SfmError::InvalidInput(format!(
                "cannot clear only part of shared {}: view {} was not selected",
                view.pose_id, view.id
            )));
        }
        for pose in pose_ids {
            self.poses.take(pose.index())?;
        }
        Ok(())
    }

    /// Look up a view record by its stable identity.
    pub fn view(&self, view_id: crate::ViewId) -> Option<&crate::model::ViewRecord> {
        self.views.get(view_id.index())
    }

    /// Check the workspace invariants (view and pose IDs, intrinsics, pair graph, and the
    /// per-view, per-pair, track, and landmark stores); fails on the first violation.
    pub fn validate(&self) -> Result<()> {
        validate::validate_workspace(self)
    }

    /// Extract and store features for the selected views with `extractor`.
    ///
    /// Fails with `SfmError::MatchingDataReleased` after [`Self::release_matching_data`].
    pub fn extract_features(
        &mut self,
        extractor: &dyn FeatureExtractor,
        selector: ViewSelector,
    ) -> Result<&mut Self> {
        ops::features::extract_features(self, extractor, selector)?;
        Ok(self)
    }

    /// [`Self::extract_features`], reporting `FeatureExtraction` progress to `progress`.
    pub fn extract_features_with_progress(
        &mut self,
        extractor: &dyn FeatureExtractor,
        selector: ViewSelector,
        progress: &dyn SfmProgressSink,
    ) -> Result<&mut Self> {
        ops::features::extract_features_with_progress(self, extractor, selector, progress)?;
        Ok(self)
    }

    /// Install the pairs returned by `proposer` as the pair graph; see
    /// [`Self::propose_pairs_report`].
    pub fn propose_pairs(&mut self, proposer: &dyn PairProposer) -> Result<&mut Self> {
        let _ = ops::pairs::propose_pairs(self, proposer)?;
        Ok(self)
    }

    /// Install the pairs returned by `proposer` as the pair graph and report the proposal.
    ///
    /// Requires features for every view and an empty pair graph with no pair-dependent or
    /// reconstruction state; fails if the proposer returns no usable pair.
    pub fn propose_pairs_report(
        &mut self,
        proposer: &dyn PairProposer,
    ) -> Result<crate::ops::pairs::PairProposalReport> {
        ops::pairs::propose_pairs(self, proposer)
    }

    /// Match features for the selected pairs with `matcher` and store the putative matches.
    pub fn match_pairs(
        &mut self,
        matcher: &dyn PairMatcher,
        selector: PairSelector,
    ) -> Result<&mut Self> {
        ops::matching::match_pairs(self, matcher, selector)?;
        Ok(self)
    }

    /// [`Self::match_pairs`], reporting `Matching` progress, preceded by `MatcherPreparation`
    /// for matchers that build per-image state.
    pub fn match_pairs_with_progress(
        &mut self,
        matcher: &dyn PairMatcher,
        selector: PairSelector,
        progress: &dyn SfmProgressSink,
    ) -> Result<&mut Self> {
        ops::matching::match_pairs_with_progress(self, matcher, selector, progress)?;
        Ok(self)
    }

    pub(crate) fn ensure_pair_graph_uninitialized(&self) -> Result<()> {
        let downstream_state_exists = !self.pair_graph.is_empty()
            || self.matches.populated_count() != 0
            || self.geometric_models.populated_count() != 0
            || self.matches.slot_count() != 0
            || self.geometric_models.slot_count() != 0
            || self.matching_data_released
            || self.rotation_estimates.is_some()
            || self.camera_centers.is_some()
            || self.poses.populated_count() != 0
            || !self.tracks.is_empty()
            || !self.landmarks.is_empty()
            || !self.components.is_empty()
            || !self.registration_failures.is_empty();
        if downstream_state_exists {
            return Err(crate::SfmError::InvalidInput(
                "pair proposal requires an empty pair graph and no pair-dependent or reconstruction state"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn install_pair_graph_before_matching(&mut self, graph: PairGraph) -> Result<()> {
        self.ensure_pair_graph_uninitialized()?;
        if graph.view_count() != self.views.len() {
            return Err(crate::SfmError::InvalidInput(format!(
                "pair graph image count {} does not match workspace image count {}",
                graph.view_count(),
                self.views.len()
            )));
        }
        graph.validate()?;

        let matches = SparseStore::new(graph.len());
        let geometric_models = SparseStore::new(graph.len());
        self.pair_graph = graph;
        self.matches = matches;
        self.geometric_models = geometric_models;
        Ok(())
    }

    /// Geometrically verify the selected pairs; see [`Self::verify_pairs_report`].
    pub fn verify_pairs(
        &mut self,
        verifier: &dyn GeometricVerifier,
        selector: PairSelector,
    ) -> Result<&mut Self> {
        let _ = ops::verification::verify_pairs(self, verifier, selector)?;
        Ok(self)
    }

    /// Geometrically verify the selected pairs with `verifier` and report every outcome.
    ///
    /// An accepted pair stores its two-view model; a rejected pair has its model cleared and is
    /// deactivated.
    pub fn verify_pairs_report(
        &mut self,
        verifier: &dyn GeometricVerifier,
        selector: PairSelector,
    ) -> Result<crate::two_view::VerificationBatchReport> {
        ops::verification::verify_pairs(self, verifier, selector)
    }

    /// [`Self::verify_pairs_report`], reporting `Verification` progress to `progress`.
    pub fn verify_pairs_with_progress(
        &mut self,
        verifier: &dyn GeometricVerifier,
        selector: PairSelector,
        progress: &dyn SfmProgressSink,
    ) -> Result<crate::two_view::VerificationBatchReport> {
        ops::verification::verify_pairs_with_progress(self, verifier, selector, progress)
    }

    /// Replace the tracks with those built from the verified inlier matches of the active pairs.
    ///
    /// Fails with `SfmError::MatchingDataReleased` after [`Self::release_matching_data`].
    pub fn build_tracks(&mut self) -> Result<ops::tracks::TrackBuildReport> {
        ops::tracks::build_tracks(self)
    }

    /// Release keypoints, descriptors, and putative matches, keeping only the feature geometry
    /// that tracks reference.
    ///
    /// Idempotent. Fails if no tracks have been built or a track observation does not resolve to
    /// a stored feature position.
    pub fn release_matching_data(&mut self) -> Result<&mut Self> {
        ops::reconstruction::release_matching_data(self)?;
        Ok(self)
    }

    /// Triangulate every track with at least two posed observations, replacing all
    /// landmarks, and rebuild the reconstruction component with a free gauge.
    ///
    /// Fails with `SfmError::MissingPrerequisite` when no tracks have been built.
    pub fn triangulate(
        &mut self,
        triangulator: &dyn Triangulator,
    ) -> Result<ops::reconstruction::TriangulationReport> {
        ops::reconstruction::triangulate(self, triangulator, GaugeScalePolicy::Free)
    }

    pub(crate) fn bundle_adjust(
        &mut self,
        ba: &dyn BundleAdjustmentBackend,
        cfg: &BundleAdjustConfig,
    ) -> Result<ops::reconstruction::BundleReport> {
        ops::reconstruction::bundle_adjust(self, ba, cfg)
    }

    /// Compute a coverage and accuracy summary of the current state (not stored).
    pub fn quality_report(&self) -> crate::workspace::quality::QualityReport {
        crate::workspace::quality::quality_report(self)
    }
}

fn validate_camera_for_assigned_views(
    views: &[ViewRecord],
    intrinsic_id: IntrinsicId,
    camera: &Camera,
) -> Result<()> {
    let mut dimensions = None;
    for view in views
        .iter()
        .filter(|view| view.intrinsic_id == intrinsic_id)
    {
        let current = (view.meta.width, view.meta.height);
        if let Some(expected) = dimensions {
            if current != expected {
                return Err(crate::SfmError::InvalidInput(format!(
                    "intrinsic group {intrinsic_id} is shared by images with dimensions {}x{} and {}x{}",
                    expected.0, expected.1, current.0, current.1
                )));
            }
        } else {
            dimensions = Some(current);
        }
    }
    if let Some((width, height)) = dimensions {
        openmvg_core::model::CameraIntrinsics::try_new(camera.clone(), width, height).map_err(
            |error| {
                crate::SfmError::InvalidInput(format!(
                    "camera model {intrinsic_id} is invalid for {width}x{height} images: {error}"
                ))
            },
        )?;
    }
    Ok(())
}

pub(crate) fn validate_camera_models_for_views(
    views: &[ViewRecord],
    camera_models: &[Camera],
) -> Result<()> {
    let mut dimensions_by_group = BTreeMap::new();
    for view in views {
        if view.intrinsic_id.index() >= camera_models.len() {
            return Err(crate::SfmError::InvalidId {
                kind: IntrinsicId::kind(),
                index: view.intrinsic_id.index(),
            });
        }
        if view.meta.width == 0 || view.meta.height == 0 {
            return Err(crate::SfmError::InvalidInput(format!(
                "image {} has invalid dimensions {}x{}",
                view.id, view.meta.width, view.meta.height
            )));
        }
        let dimensions = (view.meta.width, view.meta.height);
        if let Some(previous) = dimensions_by_group.insert(view.intrinsic_id, dimensions)
            && previous != dimensions
        {
            return Err(crate::SfmError::InvalidInput(format!(
                "intrinsic group {} is shared by images with dimensions {}x{} and {}x{}",
                view.intrinsic_id, previous.0, previous.1, dimensions.0, dimensions.1
            )));
        }
    }
    if camera_models.len() != dimensions_by_group.len() {
        return Err(crate::SfmError::InvalidInput(
            "every camera-model group must be assigned to at least one image".to_owned(),
        ));
    }
    for (intrinsic_id, camera) in camera_models.iter().enumerate() {
        validate_camera_for_assigned_views(views, IntrinsicId::new(intrinsic_id), camera)?;
    }
    Ok(())
}

#[cfg(test)]
mod camera_model_tests {
    use super::*;
    use crate::model::ImageMeta;

    #[test]
    fn subset_replacement_is_atomic_when_any_id_is_invalid() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([
                Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
                Camera::try_pinhole(110.0, 110.0, 32.0, 24.0).unwrap(),
            ])
            .view_intrinsics([IntrinsicId::new(0), IntrinsicId::new(1)])
            .build()
            .unwrap();
        let before = workspace.camera_models().to_vec();
        let valid = Camera::try_pinhole(150.0, 150.0, 32.0, 24.0).unwrap();

        let error = workspace
            .replace_camera_model_groups(vec![
                (IntrinsicId::new(0), valid.clone()),
                (IntrinsicId::new(2), valid),
            ])
            .unwrap_err();

        assert!(matches!(error, crate::SfmError::InvalidId { .. }));
        assert_eq!(workspace.camera_models(), before);
    }

    #[test]
    fn subset_replacement_rejects_duplicates_without_mutation() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([ImageMeta::external_id("a", 64, 48)])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .build()
            .unwrap();
        let before = workspace.camera_models().to_vec();
        let first = Camera::try_pinhole(120.0, 120.0, 32.0, 24.0).unwrap();
        let second = Camera::try_pinhole(140.0, 140.0, 32.0, 24.0).unwrap();

        assert!(
            workspace
                .replace_camera_model_groups(vec![
                    (IntrinsicId::new(0), first),
                    (IntrinsicId::new(0), second),
                ])
                .is_err()
        );
        assert_eq!(workspace.camera_models(), before);
    }
}
