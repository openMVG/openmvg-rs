// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Immutable SfM input dataset and its builder.

use crate::model::{ImageMeta, ViewRecord};
use crate::workspace::builder::SfmWorkspaceBuilder;
use crate::{Camera, IntrinsicId, PoseId, Result, SfmWorkspace, ViewId};

/// Immutable input images, camera models, and stable view assignments.
///
/// ```
/// use openmvg_sfm::model::ImageMeta;
/// use openmvg_sfm::{Camera, Dataset, IntrinsicId};
///
/// let dataset = Dataset::builder()
///     .image_meta([ImageMeta::external_id("view-0", 640, 480)])
///     .camera_models([Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?])
///     .shared_intrinsic(IntrinsicId::new(0))
///     .build()?;
/// assert_eq!(dataset.views().len(), 1);
/// # Ok::<(), openmvg_sfm::SfmError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Dataset {
    pub(super) workspace: SfmWorkspace,
    pub(super) calibrated_intrinsics: bool,
}

impl Dataset {
    /// Root seed of `Dataset::from_dir` (feature `image-io`) and [`Dataset::builder`] unless
    /// [`DatasetBuilder::root_seed`] replaces it.
    pub const DEFAULT_SEED: u64 = 42;

    /// Start building a dataset from image metadata and camera models. The camera models are
    /// initial estimates that self-calibrating engines refine.
    pub fn builder() -> DatasetBuilder {
        DatasetBuilder::new()
    }

    /// Discover an image directory, or a dataset root containing `images/`, and initialize its
    /// camera models.
    ///
    /// A `K.txt` calibration matrix in `images/`, in `root`, or beside the first image (the nine
    /// values of `K`, row by row) becomes one shared pinhole camera that every engine keeps
    /// fixed. Otherwise images are grouped by EXIF camera, lens, and focal length into
    /// one-coefficient radial cameras whose focal length and distortion are estimated. An image
    /// without an EXIF focal length starts from a guess based on its size, and an image without
    /// EXIF camera make and model gets a camera of its own. The root seed is
    /// [`Self::DEFAULT_SEED`]; use [`Self::from_dir_with`] to choose the seed or the calibration
    /// file.
    #[cfg(feature = "image-io")]
    pub fn from_dir(root: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_dir_with(root, &DatasetConfig::default())
    }

    /// [`Self::from_dir`] with an explicit root seed and, optionally, an explicit calibration
    /// file.
    ///
    /// An explicit [`DatasetConfig::intrinsics`] file replaces `K.txt` discovery. If it cannot
    /// be read the error is [`crate::SfmError::ReadFile`], and if it is not a valid pinhole
    /// matrix the error is [`crate::SfmError::InvalidInput`]; there is no fallback to discovery
    /// or EXIF grouping.
    #[cfg(feature = "image-io")]
    pub fn from_dir_with(
        root: impl AsRef<std::path::Path>,
        options: &DatasetConfig,
    ) -> Result<Self> {
        use crate::image::prepare::{
            ImageDatasetConfig, ImageIntrinsicPolicy, find_calibration_file,
            load_pinhole_intrinsics_matrix, prepare_image_paths,
        };
        let root = root.as_ref();
        let paths = crate::image::decode::discover_dataset_images(root)?;
        let calibration = match options.intrinsics() {
            Some(path) => Some(path.to_path_buf()),
            None => paths
                .first()
                .and_then(|first| find_calibration_file(root, first)),
        };
        let intrinsic_policy = match calibration {
            Some(path) => ImageIntrinsicPolicy::Shared {
                camera: load_pinhole_intrinsics_matrix(&path)?,
            },
            None => ImageIntrinsicPolicy::ExifGrouped,
        };
        let config = ImageDatasetConfig::default().with_intrinsic_policy(intrinsic_policy);
        prepare_image_paths(&paths, &config)?.build_dataset(options.seed())
    }

    /// Mark the camera models as a known calibration.
    #[cfg(feature = "image-io")]
    pub(crate) const fn with_calibrated_intrinsics(mut self, calibrated: bool) -> Self {
        self.calibrated_intrinsics = calibrated;
        self
    }

    /// Whether the camera models are a known calibration, which engines keep fixed, rather than
    /// initial estimates.
    #[must_use]
    pub const fn has_calibrated_intrinsics(&self) -> bool {
        self.calibrated_intrinsics
    }

    /// Root of the deterministic seed tree used by later stages.
    #[must_use]
    pub const fn root_seed(&self) -> openmvg_core::RunSeed {
        self.workspace.root_seed()
    }

    /// Views in increasing [`ViewId`] order.
    #[must_use]
    pub fn views(&self) -> &[ViewRecord] {
        self.workspace.views()
    }

    /// Camera models in increasing [`IntrinsicId`] order.
    #[must_use]
    pub fn camera_models(&self) -> &[Camera] {
        self.workspace.camera_models()
    }

    /// Resolve one view by its stable identity.
    #[must_use]
    pub fn view(&self, view: ViewId) -> Option<&ViewRecord> {
        self.workspace.view(view)
    }

    /// Focal-length prior of one intrinsic group.
    #[must_use]
    pub fn intrinsic_prior(&self, intrinsic: IntrinsicId) -> Option<crate::model::IntrinsicPrior> {
        self.workspace.intrinsic_prior(intrinsic)
    }

    /// Resolve one shared intrinsic group.
    #[must_use]
    pub fn camera_model(&self, intrinsic: IntrinsicId) -> Option<&Camera> {
        self.workspace.camera_model(intrinsic)
    }
}

/// Explicit inputs of [`Dataset::from_dir_with`].
///
/// The default reproduces [`Dataset::from_dir`]: root seed [`Dataset::DEFAULT_SEED`] and
/// `K.txt` discovery.
///
/// ```no_run
/// use openmvg_sfm::{Dataset, DatasetConfig};
///
/// let options = DatasetConfig::default()
///     .with_seed(7.into())
///     .with_intrinsics("calibration/K.txt");
/// let dataset = Dataset::from_dir_with("scene", &options)?;
/// # Ok::<(), openmvg_sfm::SfmError>(())
/// ```
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetConfig {
    seed: openmvg_core::RunSeed,
    intrinsics: Option<std::path::PathBuf>,
}

#[cfg(feature = "image-io")]
impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            seed: Dataset::DEFAULT_SEED.into(),
            intrinsics: None,
        }
    }
}

#[cfg(feature = "image-io")]
config_accessors!(DatasetConfig {
    /// Root of the dataset's deterministic seed tree.
    copy seed, with_seed: openmvg_core::RunSeed;
});

#[cfg(feature = "image-io")]
impl DatasetConfig {
    /// Use the 3x3 pinhole matrix in `path` as the shared calibration instead of discovering a
    /// `K.txt`; engines keep it fixed.
    #[must_use]
    pub fn with_intrinsics(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.intrinsics = Some(path.into());
        self
    }

    /// Explicit calibration file; `None` discovers `K.txt` as [`Dataset::from_dir`] does.
    #[must_use]
    pub fn intrinsics(&self) -> Option<&std::path::Path> {
        self.intrinsics.as_deref()
    }
}

/// Builder for an immutable [`Dataset`].
///
/// Pair selection, features, matches, and tracks do not appear here; they belong to the
/// prepared [`ViewGraph`](crate::ViewGraph) phase.
#[derive(Debug, Clone)]
#[must_use = "the dataset builder has no effect until build() is called"]
pub struct DatasetBuilder {
    workspace: SfmWorkspaceBuilder,
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self {
            workspace: SfmWorkspaceBuilder::default().root_seed(Dataset::DEFAULT_SEED),
        }
    }
}

impl DatasetBuilder {
    /// An empty builder: no views, no camera models, one pose per view, and the root seed
    /// [`Dataset::DEFAULT_SEED`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the root of the deterministic seed tree used by every randomized stage, from the
    /// front end's matching and pair verification to the engines.
    pub fn root_seed(mut self, root_seed: impl Into<openmvg_core::RunSeed>) -> Self {
        self.workspace = self.workspace.root_seed(root_seed);
        self
    }

    /// Bound memory-sensitive frontend stages to at most `workers` threads.
    pub fn max_parallel_workers(mut self, workers: std::num::NonZeroUsize) -> Self {
        self.workspace = self.workspace.max_parallel_workers(workers);
        self
    }

    /// Set the per-image metadata; the iteration order defines the dense [`ViewId`]s.
    pub fn image_meta(mut self, images: impl IntoIterator<Item = ImageMeta>) -> Self {
        self.workspace = self.workspace.image_meta(images);
        self
    }

    /// Set the camera models; the iteration order defines the dense [`IntrinsicId`]s.
    pub fn camera_models(mut self, cameras: impl IntoIterator<Item = Camera>) -> Self {
        self.workspace = self.workspace.camera_models(cameras);
        self
    }

    /// Set the focal-length prior of each camera model, in camera-model order.
    ///
    /// Without this call every group is
    /// [`Trusted`](crate::model::IntrinsicPrior::Trusted); mark a group
    /// [`Guess`](crate::model::IntrinsicPrior::Guess) when its focal length is a placeholder,
    /// which registration may then estimate with the first pose that uses it.
    pub fn intrinsic_priors(
        mut self,
        priors: impl IntoIterator<Item = crate::model::IntrinsicPrior>,
    ) -> Self {
        self.workspace = self.workspace.intrinsic_priors(priors);
        self
    }

    /// Assign one intrinsic group to every view.
    pub fn shared_intrinsic(mut self, intrinsic: IntrinsicId) -> Self {
        self.workspace = self.workspace.shared_intrinsic(intrinsic);
        self
    }

    /// Assign one intrinsic group per view, in view order.
    pub fn view_intrinsics(mut self, assignments: impl IntoIterator<Item = IntrinsicId>) -> Self {
        self.workspace = self.workspace.view_intrinsics(assignments);
        self
    }

    /// Give every view an independent pose identity. This is the default.
    pub fn one_pose_per_view(mut self) -> Self {
        self.workspace = self.workspace.one_pose_per_view();
        self
    }

    /// Assign every view to the same pose identity.
    pub fn shared_pose(mut self) -> Self {
        self.workspace = self.workspace.shared_pose();
        self
    }

    /// Assign one pose identity per view, in view order.
    pub fn view_poses(mut self, assignments: impl IntoIterator<Item = PoseId>) -> Self {
        self.workspace = self.workspace.view_poses(assignments);
        self
    }

    /// Build and validate the dataset.
    ///
    /// A single camera model is assigned to every view automatically; with several, choose
    /// [`Self::shared_intrinsic`] or [`Self::view_intrinsics`].
    ///
    /// # Errors
    ///
    /// [`crate::SfmError::InvalidConfig`] if no camera model is given or several are given
    /// without an assignment; [`crate::SfmError::InvalidInput`] or
    /// [`crate::SfmError::InvalidId`] if an assignment does not match the views or camera
    /// models, an image has invalid dimensions, or one intrinsic group spans images of
    /// different sizes.
    pub fn build(self) -> Result<Dataset> {
        self.workspace.build().map(|workspace| Dataset {
            workspace,
            calibrated_intrinsics: false,
        })
    }
}
