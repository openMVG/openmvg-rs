// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked construction of an [`SfmWorkspace`] from views, cameras, and
//! optional features, matches, and poses.

use std::collections::BTreeSet;

use crate::TrackSet;
use crate::config::WorkspaceBuildConfig;
use crate::model::{ImageMeta, ViewRecord};
use crate::workspace::pair_graph::PairGraph;
use crate::{FeatureSet, IntrinsicId, PoseId, Result, SfmError, SfmWorkspace, ViewId, ViewPair};
use openmvg_core::camera::Camera;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Default)]
enum PairInit {
    #[default]
    Empty,
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    Exhaustive,
    Imported(Vec<ViewPair>),
}

#[derive(Debug, Clone, Default)]
enum IntrinsicAssignment {
    #[default]
    Auto,
    Shared(IntrinsicId),
    PerView(Vec<IntrinsicId>),
}

#[derive(Debug, Clone, Default)]
enum PoseAssignment {
    #[default]
    PerView,
    Shared,
    Explicit(Vec<PoseId>),
}

impl PoseAssignment {
    fn resolve(self, view_count: usize) -> Result<Vec<PoseId>> {
        let assignments = match self {
            Self::PerView => (0..view_count).map(PoseId::new).collect(),
            Self::Shared => vec![PoseId::new(0); view_count],
            Self::Explicit(assignments) => {
                if assignments.len() != view_count {
                    return Err(SfmError::InvalidInput(format!(
                        "expected {view_count} view pose assignments, got {}",
                        assignments.len()
                    )));
                }
                assignments
            }
        };
        validate_dense_pose_assignments(&assignments)?;
        Ok(assignments)
    }
}

fn validate_dense_pose_assignments(assignments: &[PoseId]) -> Result<()> {
    let assigned = assignments.iter().copied().collect::<BTreeSet<_>>();
    if let Some(missing) = assigned
        .iter()
        .enumerate()
        .find_map(|(expected, actual)| (actual.index() != expected).then_some(expected))
    {
        return Err(SfmError::InvalidId {
            kind: PoseId::kind(),
            index: missing,
        });
    }
    Ok(())
}

impl IntrinsicAssignment {
    fn resolve(self, image_count: usize, intrinsic_count: usize) -> Result<Vec<IntrinsicId>> {
        match self {
            Self::Auto if image_count == 0 => Ok(Vec::new()),
            Self::Auto if intrinsic_count == 1 => Ok(vec![IntrinsicId::new(0); image_count]),
            Self::Auto if intrinsic_count == 0 => Err(SfmError::InvalidConfig(format!(
                "cannot assign intrinsics to {image_count} images: no intrinsic group was provided"
            ))),
            Self::Auto => Err(SfmError::InvalidConfig(format!(
                "{intrinsic_count} intrinsic groups require an explicit shared or per-image assignment"
            ))),
            Self::Shared(intrinsic_id) => {
                validate_intrinsic_id(intrinsic_id, intrinsic_count)?;
                Ok(vec![intrinsic_id; image_count])
            }
            Self::PerView(assignments) => {
                if assignments.len() != image_count {
                    return Err(SfmError::InvalidInput(format!(
                        "expected {image_count} image intrinsic assignments, got {}",
                        assignments.len()
                    )));
                }
                for &intrinsic_id in &assignments {
                    validate_intrinsic_id(intrinsic_id, intrinsic_count)?;
                }
                Ok(assignments)
            }
        }
    }
}

fn validate_intrinsic_id(intrinsic_id: IntrinsicId, intrinsic_count: usize) -> Result<()> {
    if intrinsic_id.index() >= intrinsic_count {
        return Err(SfmError::InvalidId {
            kind: IntrinsicId::kind(),
            index: intrinsic_id.index(),
        });
    }
    Ok(())
}

/// Builder for a checked mutable `SfmWorkspace`.
///
/// Collects image metadata, camera models, intrinsic and pose assignments, the initial pair
/// set, and optional imported tracks; [`Self::build`] validates them together.
#[derive(Debug, Clone, Default)]
#[must_use = "the workspace builder has no effect until build() is called"]
pub struct SfmWorkspaceBuilder {
    config: WorkspaceBuildConfig,
    image_meta: Vec<ImageMeta>,
    camera_models: Vec<Camera>,
    intrinsic_priors: Option<Vec<crate::model::IntrinsicPrior>>,
    intrinsic_assignment: IntrinsicAssignment,
    pose_assignment: PoseAssignment,
    pair_init: PairInit,
    tracks: TrackSet,
    feature_sets: Vec<FeatureSet>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl SfmWorkspaceBuilder {
    /// Create a workspace builder.
    ///
    /// ```
    /// # #[cfg(feature = "unstable-internals")]
    /// # fn main() -> Result<(), openmvg_sfm::SfmError> {
    /// use openmvg_sfm::model::ImageMeta;
    /// use openmvg_sfm::{Camera, IntrinsicId, unstable::SfmWorkspaceBuilder};
    ///
    /// let workspace = SfmWorkspaceBuilder::new()
    ///     .image_meta(vec![ImageMeta::external_id("image-0", 800, 600)])
    ///     .camera_models([Camera::try_pinhole(700.0, 700.0, 400.0, 300.0)?])
    ///     .shared_intrinsic(IntrinsicId::new(0))
    ///     .build()?;
    /// assert_eq!(workspace.views().len(), 1);
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "unstable-internals"))]
    /// # fn main() {}
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the whole workspace build configuration (seed, worker cap, and the rest).
    pub fn config(mut self, config: WorkspaceBuildConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the root of the deterministic seed tree for all randomized stages.
    pub fn root_seed(mut self, root_seed: impl Into<openmvg_core::RunSeed>) -> Self {
        self.config = self.config.with_root_seed(root_seed.into());
        self
    }

    /// Bound memory-sensitive parallel stages to at most `workers` threads.
    pub fn max_parallel_workers(mut self, workers: NonZeroUsize) -> Self {
        self.config = self.config.with_max_parallel_workers(Some(workers));
        self
    }

    /// Set the per-image metadata; the iteration order defines the dense view IDs.
    pub fn image_meta(mut self, image_meta: impl IntoIterator<Item = ImageMeta>) -> Self {
        self.image_meta = image_meta.into_iter().collect();
        self
    }

    /// Set the camera models for the workspace.
    pub fn camera_models(mut self, cameras: impl IntoIterator<Item = Camera>) -> Self {
        self.camera_models = cameras.into_iter().collect();
        self
    }

    /// Set the focal-length prior of each camera model, in camera-model order.
    ///
    /// Without this call every group is [`IntrinsicPrior::Trusted`](crate::model::IntrinsicPrior::Trusted). [`Self::build`] rejects a
    /// count that differs from the number of camera models.
    pub fn intrinsic_priors(
        mut self,
        priors: impl IntoIterator<Item = crate::model::IntrinsicPrior>,
    ) -> Self {
        self.intrinsic_priors = Some(priors.into_iter().collect());
        self
    }

    /// Assign one intrinsic group to every view.
    ///
    /// Assignment is resolved by [`Self::build`], so this method may be called
    /// before or after [`Self::image_meta`] without changing the result. Calling
    /// it replaces any earlier [`Self::view_intrinsics`] assignment.
    pub fn shared_intrinsic(mut self, intrinsic_id: IntrinsicId) -> Self {
        self.intrinsic_assignment = IntrinsicAssignment::Shared(intrinsic_id);
        self
    }

    /// Assign one intrinsic group per view, in view order.
    ///
    /// The assignment count and IDs are checked by [`Self::build`]. Calling
    /// this method replaces any earlier [`Self::shared_intrinsic`] assignment.
    pub fn view_intrinsics(
        mut self,
        view_intrinsics: impl IntoIterator<Item = IntrinsicId>,
    ) -> Self {
        self.intrinsic_assignment =
            IntrinsicAssignment::PerView(view_intrinsics.into_iter().collect());
        self
    }

    /// Assign one independently stored pose block to every view.
    ///
    /// This is the default. Pose identity remains distinct from view identity
    /// even when their dense indices happen to match.
    pub fn one_pose_per_view(mut self) -> Self {
        self.pose_assignment = PoseAssignment::PerView;
        self
    }

    /// Assign every view to one identical world-to-camera pose block.
    ///
    /// This models repeated observations from the same camera transform. It is
    /// not a camera-rig shortcut: distinct rig sensors require their fixed
    /// sensor extrinsics to be composed with a shared rig pose.
    pub fn shared_pose(mut self) -> Self {
        self.pose_assignment = PoseAssignment::Shared;
        self
    }

    /// Assign one pose group per view, in view order.
    ///
    /// Pose IDs must form a dense domain beginning at zero. Multiple views may
    /// reference the same pose ID when they have the exact same camera pose.
    pub fn view_poses(mut self, view_poses: impl IntoIterator<Item = PoseId>) -> Self {
        self.pose_assignment = PoseAssignment::Explicit(view_poses.into_iter().collect());
        self
    }

    /// Start with every unordered view pair active.
    pub fn exhaustive_pairs(mut self) -> Self {
        self.pair_init = PairInit::Exhaustive;
        self
    }

    /// Start with exactly the given view pairs active, checked by [`Self::build`].
    pub fn imported_pairs<P>(mut self, pairs: impl IntoIterator<Item = P>) -> Self
    where
        P: Into<ViewPair>,
    {
        self.pair_init = PairInit::Imported(pairs.into_iter().map(Into::into).collect());
        self
    }

    /// Import tracks together with their per-view feature geometry.
    ///
    /// This is the lossless entry point for trusted correspondence sources such as a dataset's
    /// reference reconstruction. Feature extraction and matching remain optional; observation
    /// image IDs are checked against the completed workspace by [`Self::build`].
    pub fn tracks(
        mut self,
        tracks: TrackSet,
        feature_sets: impl IntoIterator<Item = FeatureSet>,
    ) -> Self {
        self.tracks = tracks;
        self.feature_sets = feature_sets.into_iter().collect();
        self
    }

    /// Build and validate the workspace.
    ///
    /// A single intrinsic group is assigned automatically. With multiple
    /// groups, callers must choose [`Self::shared_intrinsic`] or
    /// [`Self::view_intrinsics`] explicitly.
    pub fn build(self) -> Result<SfmWorkspace> {
        let Self {
            config,
            image_meta,
            camera_models,
            intrinsic_priors,
            intrinsic_assignment,
            pose_assignment,
            pair_init,
            tracks,
            feature_sets,
        } = self;

        let camera_count = camera_models.len();
        if let Some(priors) = &intrinsic_priors
            && priors.len() != camera_count
        {
            return Err(SfmError::InvalidInput(format!(
                "expected {camera_count} intrinsic priors, got {}",
                priors.len()
            )));
        }
        let view_intrinsics = intrinsic_assignment.resolve(image_meta.len(), camera_count)?;
        let view_poses = pose_assignment.resolve(image_meta.len())?;

        let images: Vec<ViewRecord> = image_meta
            .into_iter()
            .zip(view_intrinsics)
            .zip(view_poses)
            .enumerate()
            .map(|(index, ((meta, intrinsic_id), pose_id))| ViewRecord {
                id: ViewId::new(index),
                meta,
                intrinsic_id,
                pose_id,
            })
            .collect();
        crate::workspace::validate_camera_models_for_views(&images, &camera_models)?;

        let pair_graph = match pair_init {
            PairInit::Empty => PairGraph::empty(images.len()),
            PairInit::Exhaustive => PairGraph::exhaustive(images.len())?,
            PairInit::Imported(pairs) => PairGraph::try_new(images.len(), pairs)?,
        };

        let mut workspace = SfmWorkspace::from_parts(
            config.root_seed(),
            config.max_parallel_workers(),
            images,
            camera_models,
            pair_graph,
        );
        if let Some(priors) = intrinsic_priors {
            workspace.set_intrinsic_priors(priors);
        }
        if (!tracks.is_empty() || !feature_sets.is_empty())
            && feature_sets.len() != workspace.views().len()
        {
            return Err(SfmError::InvalidInput(format!(
                "imported track data requires one feature set per view; got {} for {} views",
                feature_sets.len(),
                workspace.views().len()
            )));
        }
        for (index, feature_set) in feature_sets.into_iter().enumerate() {
            feature_set.validate()?;
            workspace.feature_sets.set(index, feature_set)?;
        }
        workspace.tracks = tracks;
        if config.validate_on_build() {
            workspace.validate()?;
        }
        Ok(workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureId, Track, TrackId, TrackObservation};
    use nalgebra::{Point2, Point3, Rotation3};
    use openmvg_core::camera::{RadialDistortion, TangentialDistortion};
    use openmvg_core::model::WorldToCameraPose;

    fn images(count: usize) -> Vec<ImageMeta> {
        (0..count)
            .map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48))
            .collect()
    }

    fn two_intrinsics() -> Vec<Camera> {
        vec![
            Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
            Camera::try_pinhole(200.0, 200.0, 32.0, 24.0).unwrap(),
        ]
    }

    #[test]
    fn shared_assignment_is_independent_of_builder_call_order() {
        let workspace = SfmWorkspaceBuilder::new()
            .shared_intrinsic(IntrinsicId::new(0))
            .camera_models([Camera::try_pinhole(200.0, 200.0, 32.0, 24.0).unwrap()])
            .image_meta(images(3))
            .build()
            .unwrap();

        assert!(
            workspace
                .views()
                .iter()
                .all(|image| image.intrinsic_id == IntrinsicId::new(0))
        );
    }

    #[test]
    fn views_can_share_one_independent_pose_identity() {
        let mut workspace = SfmWorkspaceBuilder::new()
            .image_meta(images(3))
            .camera_models([Camera::try_pinhole(200.0, 200.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .shared_pose()
            .build()
            .unwrap();

        assert!(
            workspace
                .views()
                .iter()
                .all(|view| view.pose_id == PoseId::new(0))
        );
        assert_eq!(workspace.pose_store().slot_count(), 1);
        workspace
            .set_pose_for_view(
                crate::ViewId::new(1),
                WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())
                    .unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(workspace.pose_store().populated_count(), 1);
        assert_eq!(workspace.registered_view_count(), 3);
        assert!(
            workspace
                .views()
                .iter()
                .all(|view| workspace.has_pose(view.id))
        );
        workspace.validate().unwrap();

        assert!(
            workspace
                .clear_poses_for_views(&[crate::ViewId::new(0)])
                .is_err()
        );
        assert_eq!(workspace.registered_view_count(), 3);
        workspace
            .clear_poses_for_views(&[
                crate::ViewId::new(0),
                crate::ViewId::new(1),
                crate::ViewId::new(2),
            ])
            .unwrap();
        assert_eq!(workspace.registered_view_count(), 0);
    }

    #[test]
    fn explicit_pose_assignments_must_form_a_dense_domain() {
        let error = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models([Camera::try_pinhole(200.0, 200.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .view_poses([PoseId::new(0), PoseId::new(2)])
            .build()
            .unwrap_err();

        assert!(matches!(
            error,
            SfmError::InvalidId {
                kind: "pose",
                index: 1
            }
        ));

        let oversized = SfmWorkspaceBuilder::new()
            .image_meta(images(1))
            .camera_models([Camera::try_pinhole(200.0, 200.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .view_poses([PoseId::new(usize::MAX)])
            .build()
            .unwrap_err();
        assert!(matches!(
            oversized,
            SfmError::InvalidId {
                kind: "pose",
                index: 0
            }
        ));
    }

    #[test]
    fn per_image_assignment_is_independent_of_builder_call_order() {
        let assignments = [
            IntrinsicId::new(1),
            IntrinsicId::new(0),
            IntrinsicId::new(1),
        ];
        let workspace = SfmWorkspaceBuilder::new()
            .view_intrinsics(assignments)
            .image_meta(images(3))
            .camera_models(two_intrinsics())
            .build()
            .unwrap();

        assert_eq!(
            workspace
                .views()
                .iter()
                .map(|image| image.intrinsic_id)
                .collect::<Vec<_>>(),
            assignments
        );
    }

    #[test]
    fn single_intrinsic_is_the_only_automatic_assignment() {
        let workspace = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .build()
            .unwrap();
        assert!(
            workspace
                .views()
                .iter()
                .all(|image| image.intrinsic_id == IntrinsicId::new(0))
        );

        let error = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models(two_intrinsics())
            .build()
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("require an explicit shared or per-image assignment")
        );
    }

    #[test]
    fn explicit_empty_assignment_is_not_treated_as_auto() {
        let error = SfmWorkspaceBuilder::new()
            .image_meta(images(1))
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .view_intrinsics([])
            .build()
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid input: expected 1 image intrinsic assignments, got 0"
        );
    }

    #[test]
    fn latest_assignment_method_wins() {
        let workspace = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models(two_intrinsics())
            .shared_intrinsic(IntrinsicId::new(1))
            .view_intrinsics([IntrinsicId::new(0), IntrinsicId::new(1)])
            .build()
            .unwrap();

        assert_eq!(workspace.views()[0].intrinsic_id, IntrinsicId::new(0));
        assert_eq!(workspace.views()[1].intrinsic_id, IntrinsicId::new(1));
    }

    #[test]
    fn build_rejects_unassigned_camera_groups() {
        let error = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models(two_intrinsics())
            .shared_intrinsic(IntrinsicId::new(1))
            .build()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("every camera-model group must be assigned")
        );
    }

    #[test]
    fn camera_domain_validation_cannot_be_disabled() {
        let error = SfmWorkspaceBuilder::new()
            .config(WorkspaceBuildConfig::default().with_validate_on_build(false))
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 80, 60),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .build()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("shared by images with dimensions")
        );
    }

    #[test]
    fn root_seed_is_owned_by_the_workspace() {
        let workspace = SfmWorkspaceBuilder::new().root_seed(42).build().unwrap();

        assert_eq!(workspace.root_seed(), 42.into());
    }

    #[test]
    fn parallel_worker_limit_is_owned_by_the_workspace() {
        let workers = NonZeroUsize::new(3).unwrap();
        let workspace = SfmWorkspaceBuilder::new()
            .max_parallel_workers(workers)
            .build()
            .unwrap();

        assert_eq!(workspace.max_parallel_workers(), Some(workers));
    }

    #[test]
    fn build_rejects_brown_calibration_that_folds_inside_the_sensor() {
        let folded = Camera::try_brown_conrady(
            100.0,
            100.0,
            32.0,
            24.0,
            RadialDistortion::K1K2K3(-1.0, 0.0, 0.0),
            TangentialDistortion::P1P2(0.0, 0.0),
        )
        .unwrap();
        let error = SfmWorkspaceBuilder::new()
            .image_meta(images(1))
            .camera_models([folded])
            .build()
            .unwrap_err();

        assert!(error.to_string().contains("camera model 0 is invalid"));
    }

    #[test]
    fn imported_tracks_preserve_exact_observations_without_features() {
        let xy = Point2::new(12.345_678_901_234, 45.678_901_234_567);
        let tracks = TrackSet::try_new(vec![
            Track::try_new(
                TrackId::new(0),
                vec![
                    TrackObservation {
                        view: ViewId::new(0),
                        feature: FeatureId::new(17),
                    },
                    TrackObservation {
                        view: ViewId::new(1),
                        feature: FeatureId::new(91),
                    },
                ],
            )
            .unwrap(),
        ])
        .unwrap();

        let workspace = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .tracks(
                tracks,
                [
                    FeatureSet::try_new({
                        let mut pixels = vec![Point2::origin(); 18];
                        pixels[17] = xy;
                        pixels
                    })
                    .unwrap(),
                    FeatureSet::try_new({
                        let mut pixels = vec![Point2::origin(); 92];
                        pixels[91] = Point2::new(20.0, 30.0);
                        pixels
                    })
                    .unwrap(),
                ],
            )
            .build()
            .unwrap();

        assert_eq!(workspace.feature_store().populated_count(), 0);
        assert_eq!(
            workspace
                .track_observation_pixel(&workspace.tracks().as_slice()[0].observations()[0])
                .unwrap(),
            xy
        );
    }

    #[test]
    fn imported_tracks_reject_observations_outside_the_workspace() {
        let tracks = TrackSet::try_new(vec![
            Track::try_new(
                TrackId::new(0),
                vec![
                    TrackObservation {
                        view: ViewId::new(0),
                        feature: FeatureId::new(0),
                    },
                    TrackObservation {
                        view: ViewId::new(2),
                        feature: FeatureId::new(0),
                    },
                ],
            )
            .unwrap(),
        ])
        .unwrap();

        let error = SfmWorkspaceBuilder::new()
            .image_meta(images(2))
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .tracks(
                tracks,
                [
                    FeatureSet::try_new(vec![Point2::origin()]).unwrap(),
                    FeatureSet::try_new(vec![Point2::origin()]).unwrap(),
                ],
            )
            .build()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("track 0 contains an invalid observation")
        );
    }

    #[test]
    fn intrinsic_priors_default_to_trusted_and_must_match_the_camera_count() {
        use crate::model::IntrinsicPrior;
        let camera = Camera::try_pinhole(200.0, 200.0, 32.0, 24.0).unwrap();
        let build = |priors: Option<Vec<IntrinsicPrior>>| {
            let builder = SfmWorkspaceBuilder::new()
                .image_meta([
                    ImageMeta::external_id("a", 64, 48),
                    ImageMeta::external_id("b", 64, 48),
                ])
                .camera_models([camera.clone(), camera.clone()])
                .view_intrinsics([IntrinsicId::new(0), IntrinsicId::new(1)]);
            match priors {
                Some(priors) => builder.intrinsic_priors(priors).build(),
                None => builder.build(),
            }
        };

        let default = build(None).unwrap();
        assert_eq!(default.intrinsic_priors(), &[IntrinsicPrior::Trusted; 2]);

        let marked = build(Some(vec![IntrinsicPrior::Trusted, IntrinsicPrior::Guess])).unwrap();
        assert_eq!(
            marked.intrinsic_prior(IntrinsicId::new(1)),
            Some(IntrinsicPrior::Guess)
        );
        assert_eq!(marked.intrinsic_prior(IntrinsicId::new(2)), None);

        assert!(matches!(
            build(Some(vec![IntrinsicPrior::Guess])),
            Err(SfmError::InvalidInput(_))
        ));
    }
}
