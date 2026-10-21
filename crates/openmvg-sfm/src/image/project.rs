// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit image projects: the images of a folder assigned to intrinsic groups by hand or by a
//! tool.
//!
//! ```text
//! discover_dataset_images ─► prepare_image_paths ─► PreparedImageDataset
//!   (folder → paths)         (EXIF groups, auto)            │ to_project
//!                                                           ▼
//!                                  ImageProject ◄─ load / edit / save (JSON, draft schema 1)
//!                                        │
//!                                        ▼  prepare_image_project(paths, project)
//!                                  PreparedImageDataset
//! ```
//!
//! A project partitions the images it names into camera groups: one camera model and focal
//! length per group, the image center as principal point. An image left out of every group is
//! excluded from the dataset. The groups, not EXIF, decide which images share an intrinsic, so a
//! project written from the automatic grouping gives the same views, cameras, groups, and
//! focal-length trust. The per-image report differs: it records the project as the focal source
//! and keeps no EXIF camera names.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use openmvg_core::camera::{CameraModelType, RadialDistortion};

use crate::image::prepare::{
    ImageIntrinsicReport, ImageIntrinsicSource, PreparedImageDataset, camera_focal_px,
    equirectangular_camera, image_size, initial_camera,
};
use crate::model::ImageMeta;
use crate::{Camera, IntrinsicId, Result, SfmError};

/// The images of a folder assigned to intrinsic groups, with the calibration policy of the run.
///
/// Image names are file names, as discovered by
/// [`discover_dataset_images`](crate::image::discover_dataset_images). The JSON form is a draft:
/// its schema may change incompatibly before 1.0.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct ImageProject {
    /// Schema version, [`ImageProject::SCHEMA_VERSION`].
    schema_version: u32,
    /// Whether bundle adjustment should refine the focal length and distortion of every group.
    /// The project only records the choice: apply it with `with_self_calibrate_intrinsics` on
    /// the engine configuration ([`IncrementalSfmConfig`](crate::config::IncrementalSfmConfig)
    /// or [`GlobalSfmConfig`](crate::config::GlobalSfmConfig)), as the `openmvg-sfm` command
    /// does.
    pub self_calibrate: bool,
    /// The camera groups; the position of a group is its [`IntrinsicId`].
    pub groups: Vec<ImageProjectGroup>,
}

/// Images that share one camera.
///
/// Every image of a group has the same pixel size, and the group's principal point is the image
/// center.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct ImageProjectGroup {
    /// Declared camera model: `pinhole`, `pinhole_radial1`, `pinhole_radial2`, or `spherical`.
    /// Radial coefficients start at zero, meaning unknown.
    pub camera_model: CameraModelType,
    /// Initial focal length in pixels; required for a pinhole model, absent for `spherical`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focal_px: Option<f64>,
    /// The focal length is a guess, not a measurement, so bundle adjustment trusts it less.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub focal_is_guess: bool,
    /// File names of the images of the group.
    pub images: Vec<String>,
}

impl ImageProject {
    /// Version of the project JSON schema.
    pub const SCHEMA_VERSION: u32 = 1;

    /// A project with the given groups.
    pub fn new(self_calibrate: bool, groups: Vec<ImageProjectGroup>) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            self_calibrate,
            groups,
        }
    }

    /// Checks everything that does not need the image files: the schema version, the camera
    /// models and focal lengths, and that every image is named exactly once. [`load`](Self::load),
    /// [`save`](Self::save), and [`prepare_image_project`] run it.
    fn validate(&self) -> Result<()> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(invalid(format!(
                "unsupported schema version {}; expected {}",
                self.schema_version,
                Self::SCHEMA_VERSION
            )));
        }
        if self.groups.is_empty() {
            return Err(invalid("a project needs at least one group".to_owned()));
        }
        let mut named = HashSet::new();
        for (index, group) in self.groups.iter().enumerate() {
            group.validate(index)?;
            for name in &group.images {
                if !named.insert(name.as_str()) {
                    return Err(invalid(format!("image {name} is in more than one group")));
                }
            }
        }
        Ok(())
    }

    /// Reads and validates a project from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an `SfmError` when the file cannot be read, is not a valid project, names an
    /// image twice, or gives a group a model or focal length the project cannot hold.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read(path).map_err(|source| SfmError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;
        // Some Windows editors start a UTF-8 file with a byte order mark, which JSON forbids.
        let content = content.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&content);
        let project: Self = serde_json::from_slice(content)
            .map_err(|error| invalid(format!("{}: {error}", path.display())))?;
        project.validate()?;
        Ok(project)
    }

    /// Validates the project and writes it as pretty JSON.
    ///
    /// The file is written beside `path` and renamed over it, so a crash or a full disk leaves the
    /// previous project in place instead of a truncated one.
    ///
    /// # Errors
    ///
    /// Returns an `SfmError` when the project is invalid (see [`load`](Self::load)) or the file
    /// cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        crate::atomic::atomic_write_with(path, |writer| {
            serde_json::to_writer_pretty(&mut *writer, self)
                .map_err(|error| invalid(format!("cannot serialize the project: {error}")))?;
            writer.write_all(b"\n")?;
            Ok(())
        })
    }
}

impl ImageProjectGroup {
    /// A group of `images` sharing a camera with a measured focal length.
    pub fn new(camera_model: CameraModelType, focal_px: Option<f64>, images: Vec<String>) -> Self {
        Self {
            camera_model,
            focal_px,
            focal_is_guess: false,
            images,
        }
    }

    /// Marks the focal length as a guess, so bundle adjustment trusts it less.
    #[must_use]
    pub const fn with_focal_is_guess(mut self, guess: bool) -> Self {
        self.focal_is_guess = guess;
        self
    }

    fn validate(&self, index: usize) -> Result<()> {
        let group = |detail: &str| invalid(format!("group {index}: {detail}"));
        match self.camera_model {
            CameraModelType::Spherical => {
                if self.focal_px.is_some() || self.focal_is_guess {
                    return Err(group("a spherical camera has no focal length"));
                }
            }
            CameraModelType::PinholeSingle
            | CameraModelType::PinholeRadial1
            | CameraModelType::PinholeRadial2 => {
                if !self.focal_px.is_some_and(|f| f.is_finite() && f > 0.0) {
                    return Err(group("the focal length must be finite and positive"));
                }
            }
            model => {
                return Err(SfmError::UnsupportedCameraModel {
                    context: "image project groups (use pinhole, pinhole_radial1, pinhole_radial2, or spherical)",
                    camera_model: model.as_str(),
                });
            }
        }
        if self.images.is_empty() {
            return Err(group("no images"));
        }
        for name in &self.images {
            if Path::new(name).file_name().and_then(|file| file.to_str()) != Some(name) {
                return Err(group(&format!("{name:?} is not a file name")));
            }
        }
        Ok(())
    }

    /// The initial camera of the group, for images of `width` x `height` pixels.
    fn camera(&self, example: &Path, width: u32, height: u32) -> Result<Camera> {
        match (self.camera_model, self.focal_px) {
            (CameraModelType::Spherical, _) => equirectangular_camera(example, width, height),
            (model, Some(focal)) => initial_camera(
                model,
                focal,
                f64::from(width) / 2.0,
                f64::from(height) / 2.0,
            ),
            (_, None) => Err(invalid("a pinhole group needs a focal length".to_owned())),
        }
    }
}

fn invalid(detail: String) -> SfmError {
    SfmError::InvalidInput(format!("image project: {detail}"))
}

/// Reads the sizes of the images a project names and builds its cameras.
///
/// `paths` are the discovered images of the folder, in the order that decides the view order;
/// the images the project does not name are left out. Group `g` of the project becomes
/// [`IntrinsicId`] `g`.
///
/// # Errors
///
/// Returns an `SfmError` when the project is invalid (see [`ImageProject::load`]), names an image
/// that is not in `paths`, leaves fewer than two images, has a group whose images differ in size,
/// or an image cannot be read.
pub fn prepare_image_project(
    paths: &[PathBuf],
    project: &ImageProject,
) -> Result<PreparedImageDataset> {
    project.validate()?;
    let group_of: HashMap<&str, usize> = project
        .groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| group.images.iter().map(move |name| (name.as_str(), index)))
        .collect();

    let mut images = Vec::new();
    let mut image_intrinsics = Vec::new();
    // The path and size of the first image of each group, which fix the group's camera.
    let mut examples: Vec<Option<(&Path, u32, u32)>> = vec![None; project.groups.len()];
    for path in paths {
        let Some(&group) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| group_of.get(name))
        else {
            continue;
        };
        let (width, height) = image_size(path)?;
        match examples[group] {
            None => examples[group] = Some((path, width, height)),
            Some((first, first_width, first_height))
                if (first_width, first_height) != (width, height) =>
            {
                return Err(invalid(format!(
                    "group {group} mixes image sizes: {} is {first_width}x{first_height}, {} is \
                     {width}x{height}",
                    first.display(),
                    path.display()
                )));
            }
            Some(_) => {}
        }
        images.push(ImageMeta::path(path, width, height));
        image_intrinsics.push(IntrinsicId::new(group));
    }
    let found: HashSet<&str> = images.iter().map(|image| image.name.as_str()).collect();
    let mut missing: Vec<&str> = group_of
        .keys()
        .copied()
        .filter(|name| !found.contains(name))
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        return Err(invalid(format!(
            "{} named image(s) not found in the folder, first: {}",
            missing.len(),
            missing[0]
        )));
    }
    if images.len() < 2 {
        return Err(invalid("at least two images are required".to_owned()));
    }

    let mut camera_models = Vec::with_capacity(project.groups.len());
    for (group, example) in project.groups.iter().zip(&examples) {
        let Some((path, width, height)) = *example else {
            return Err(invalid("a group has no image in the folder".to_owned()));
        };
        camera_models.push(group.camera(path, width, height)?);
    }

    let intrinsic_report = images
        .iter()
        .zip(&image_intrinsics)
        .map(|(image, &intrinsic_id)| {
            let group = &project.groups[intrinsic_id.index()];
            let source = match (group.camera_model, group.focal_is_guess) {
                (CameraModelType::Spherical, _) => ImageIntrinsicSource::Equirectangular,
                (_, true) => ImageIntrinsicSource::ImageSizeFallback,
                (_, false) => ImageIntrinsicSource::Project,
            };
            Ok(ImageIntrinsicReport {
                image: image.name.clone(),
                width: image.width,
                height: image.height,
                focal_px: camera_focal_px(&camera_models[intrinsic_id.index()])?,
                source,
                camera_make: None,
                camera_model: None,
                body_serial_number: None,
                lens_model: None,
                lens_serial_number: None,
                intrinsic_id,
                warning: None,
            })
        })
        .collect::<Result<_>>()?;

    Ok(PreparedImageDataset {
        images,
        camera_models,
        image_intrinsics,
        intrinsic_report,
    })
}

impl PreparedImageDataset {
    /// The project that reproduces this dataset's grouping: one group per intrinsic, in
    /// [`IntrinsicId`] order, with the dataset's own camera models and focal lengths.
    ///
    /// # Errors
    ///
    /// Returns an `SfmError` when the dataset's vectors disagree in length or name a camera it
    /// lacks, or when a camera cannot be written as a project group: anything but a square-pixel
    /// pinhole or one-or-two-coefficient radial camera with its principal point at the image
    /// center, or a spherical camera.
    pub fn to_project(&self, self_calibrate: bool) -> Result<ImageProject> {
        let consistent = self.image_intrinsics.len() == self.images.len()
            && self.intrinsic_report.len() == self.images.len()
            && self
                .image_intrinsics
                .iter()
                .all(|id| id.index() < self.camera_models.len());
        if !consistent {
            return Err(invalid(
                "the images, intrinsic ids, and intrinsic report do not match the cameras"
                    .to_owned(),
            ));
        }
        let mut members = vec![Vec::new(); self.camera_models.len()];
        for (image, id) in self.image_intrinsics.iter().enumerate() {
            members[id.index()].push(image);
        }
        let mut groups = Vec::with_capacity(self.camera_models.len());
        for (index, (camera, members)) in self.camera_models.iter().zip(&members).enumerate() {
            let Some(&first) = members.first() else {
                return Err(invalid(format!("camera {index} has no image")));
            };
            let (model, focal_px) = project_model(camera).ok_or_else(|| {
                invalid(format!(
                    "camera {index} cannot be written as a project group"
                ))
            })?;
            let images = members
                .iter()
                .map(|&image| self.images[image].name.clone())
                .collect();
            let guess = self.intrinsic_report[first].source.is_guess();
            let group = ImageProjectGroup::new(model, focal_px, images).with_focal_is_guess(guess);
            let image = &self.images[first];
            let rebuilt = group.camera(Path::new(&image.name), image.width, image.height);
            if rebuilt.as_ref().ok() != Some(camera) {
                return Err(invalid(format!(
                    "camera {index} cannot be written as a project group"
                )));
            }
            groups.push(group);
        }
        let project = ImageProject::new(self_calibrate, groups);
        project.validate()?;
        Ok(project)
    }
}

/// The project model and focal length of a camera, when it has one.
fn project_model(camera: &Camera) -> Option<(CameraModelType, Option<f64>)> {
    match camera {
        Camera::Pinhole(camera) => Some((CameraModelType::PinholeSingle, Some(camera.focal_x()))),
        Camera::BrownConrady(camera) => {
            let model = match camera.radial_distortion() {
                RadialDistortion::K1(_) => CameraModelType::PinholeRadial1,
                RadialDistortion::K1K2(..) => CameraModelType::PinholeRadial2,
                _ => return None,
            };
            Some((model, Some(camera.focal_x())))
        }
        Camera::Spherical(_) => Some((CameraModelType::Spherical, None)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageDatasetConfig, ImageIntrinsicPolicy, prepare_image_paths};

    /// Blank PNGs of the given sizes, named `view-<index>.png` and sorted as discovered.
    fn images(root: &Path, sizes: &[(u32, u32)]) -> Vec<PathBuf> {
        sizes
            .iter()
            .enumerate()
            .map(|(index, &(width, height))| {
                let path = root.join(format!("view-{index}.png"));
                image::GrayImage::new(width, height).save(&path).unwrap();
                path
            })
            .collect()
    }

    fn names(indices: &[usize]) -> Vec<String> {
        indices.iter().map(|i| format!("view-{i}.png")).collect()
    }

    fn pinhole(focal: f64, images: &[usize]) -> ImageProjectGroup {
        ImageProjectGroup::new(CameraModelType::PinholeRadial1, Some(focal), names(images))
    }

    fn project(groups: Vec<ImageProjectGroup>) -> ImageProject {
        ImageProject::new(true, groups)
    }

    #[test]
    fn a_project_written_from_the_automatic_grouping_reproduces_it() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48), (64, 48), (32, 32), (64, 48)]);
        for model in [
            CameraModelType::PinholeSingle,
            CameraModelType::PinholeRadial2,
        ] {
            let config = ImageDatasetConfig::default()
                .with_share_unknown_cameras(true)
                .with_camera_model(model);
            let automatic = prepare_image_paths(&paths, &config).unwrap();
            assert_eq!(automatic.intrinsic_group_count(), 2);

            let written = automatic.to_project(true).unwrap();
            let rebuilt = prepare_image_project(&paths, &written).unwrap();

            // Same views, cameras, groups, and report, including the focal-guess prior.
            assert_eq!(rebuilt, automatic, "{model}");
            let file = root.path().join("project.json");
            written.save(&file).unwrap();
            assert_eq!(ImageProject::load(&file).unwrap(), written);
        }
    }

    #[test]
    fn equirectangular_groups_round_trip_without_a_focal_length() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 32), (64, 32), (96, 48)]);
        let config = ImageDatasetConfig::default().with_camera_model(CameraModelType::Spherical);
        let automatic = prepare_image_paths(&paths, &config).unwrap();

        let written = automatic.to_project(false).unwrap();

        assert!(written.groups.iter().all(|group| group.focal_px.is_none()));
        assert_eq!(prepare_image_project(&paths, &written).unwrap(), automatic);
    }

    #[test]
    fn images_outside_every_group_are_excluded_and_group_order_is_the_intrinsic_id() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48); 4]);
        let project = project(vec![pinhole(900.0, &[3]), pinhole(700.0, &[2, 0])]);

        let prepared = prepare_image_project(&paths, &project).unwrap();

        let included: Vec<_> = prepared
            .images
            .iter()
            .map(|image| image.name.as_str())
            .collect();
        assert_eq!(included, ["view-0.png", "view-2.png", "view-3.png"]);
        let groups: Vec<_> = prepared
            .image_intrinsics
            .iter()
            .map(|id| id.index())
            .collect();
        assert_eq!(groups, [1, 1, 0]);
        assert_eq!(prepared.camera_models.len(), 2);
    }

    #[test]
    fn a_group_sets_its_model_focal_length_and_trust() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48), (64, 48), (64, 48)]);
        let mut guessed = pinhole(800.0, &[2]);
        guessed.focal_is_guess = true;
        let mut single = pinhole(1000.0, &[0, 1]);
        single.camera_model = CameraModelType::PinholeSingle;

        let prepared = prepare_image_project(&paths, &project(vec![single, guessed])).unwrap();

        assert_eq!(
            prepared.camera_models[0],
            Camera::try_pinhole(1000.0, 1000.0, 32.0, 24.0).unwrap()
        );
        let sources: Vec<_> = prepared.intrinsic_report.iter().map(|r| r.source).collect();
        assert_eq!(
            sources,
            [
                ImageIntrinsicSource::Project,
                ImageIntrinsicSource::Project,
                ImageIntrinsicSource::ImageSizeFallback
            ]
        );
        let dataset = prepared.build_dataset(1).unwrap();
        assert_eq!(
            dataset.intrinsic_prior(IntrinsicId::new(1)),
            Some(crate::model::IntrinsicPrior::Guess)
        );
    }

    #[test]
    fn a_project_that_disagrees_with_the_folder_is_rejected() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48), (64, 48), (32, 32)]);
        let reject = |project: ImageProject| {
            prepare_image_project(&paths, &project)
                .unwrap_err()
                .to_string()
        };

        assert!(reject(project(vec![pinhole(900.0, &[0, 1, 7])])).contains("not found"));
        assert!(reject(project(vec![pinhole(900.0, &[0, 1, 2])])).contains("mixes image sizes"));
        assert!(reject(project(vec![pinhole(900.0, &[0])])).contains("at least two"));
    }

    #[test]
    fn invalid_projects_are_rejected_before_any_image_is_read() {
        let missing = [
            PathBuf::from("/missing/a.png"),
            PathBuf::from("/missing/b.png"),
        ];
        let spherical =
            |focal| ImageProjectGroup::new(CameraModelType::Spherical, focal, names(&[0, 1]));
        let mut other_schema = project(vec![pinhole(900.0, &[0, 1])]);
        other_schema.schema_version = 2;
        let mut fisheye = pinhole(900.0, &[0, 1]);
        fisheye.camera_model = CameraModelType::Fisheye;
        let mut nested = pinhole(900.0, &[0]);
        nested.images = vec!["../view-0.png".to_owned()];

        let cases = [
            (project(vec![]), "at least one group"),
            (other_schema, "schema version"),
            (
                project(vec![pinhole(900.0, &[]), pinhole(900.0, &[0])]),
                "no images",
            ),
            (
                project(vec![pinhole(900.0, &[0]), pinhole(900.0, &[0])]),
                "more than one group",
            ),
            (project(vec![pinhole(0.0, &[0, 1])]), "finite and positive"),
            (
                project(vec![pinhole(f64::NAN, &[0, 1])]),
                "finite and positive",
            ),
            (project(vec![spherical(Some(500.0))]), "no focal length"),
            (project(vec![fisheye]), "does not support"),
            (project(vec![nested]), "not a file name"),
        ];
        for (project, expected) in cases {
            let error = prepare_image_project(&missing, &project)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "expected {expected:?} in {error}");
        }
    }

    #[test]
    fn a_camera_a_project_cannot_express_is_not_written_silently() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48), (64, 48)]);
        for camera in [
            Camera::try_pinhole(900.0, 950.0, 32.0, 24.0).unwrap(),
            Camera::try_pinhole(900.0, 900.0, 30.0, 24.0).unwrap(),
        ] {
            let config = ImageDatasetConfig::default()
                .with_intrinsic_policy(ImageIntrinsicPolicy::Shared { camera });
            let prepared = prepare_image_paths(&paths, &config).unwrap();

            let error = prepared.to_project(false).unwrap_err();

            assert!(error.to_string().contains("cannot be written"), "{error}");
        }
    }

    #[test]
    fn the_json_file_is_strict_and_versioned() {
        let root = tempfile::TempDir::new().unwrap();
        let file = root.path().join("project.json");
        project(vec![pinhole(900.0, &[0, 1])]).save(&file).unwrap();
        let json = std::fs::read_to_string(&file).unwrap();
        assert!(json.contains("\"schema_version\": 1"), "{json}");
        assert!(
            !json.contains("focal_is_guess"),
            "default fields stay out of the file"
        );

        let load_error = |edited: String| {
            std::fs::write(&file, edited).unwrap();
            ImageProject::load(&file).unwrap_err().to_string()
        };
        // A valid project with one extra field: only `deny_unknown_fields` rejects it.
        let stray = json.replacen("\"groups\"", "\"extra\": 1, \"groups\"", 1);
        assert!(load_error(stray).contains("unknown field"));
        let newer = json.replace("\"schema_version\": 1", "\"schema_version\": 2");
        assert!(load_error(newer).contains("unsupported schema version 2"));

        // A byte order mark, which some Windows editors write, is skipped.
        std::fs::write(&file, format!("\u{feff}{json}")).unwrap();
        assert!(ImageProject::load(&file).is_ok());

        let absent = root.path().join("absent.json");
        let message = ImageProject::load(&absent).unwrap_err().to_string();
        assert!(message.contains("absent.json"), "{message}");
    }

    #[test]
    fn a_project_keeps_which_focal_lengths_are_measured_for_exif_images_too() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48), (64, 48), (32, 32)]);
        let config = ImageDatasetConfig::default().with_share_unknown_cameras(true);
        let mut prepared = prepare_image_paths(&paths, &config).unwrap();
        // Pretend the first group's focal length came from EXIF; the other is a guess.
        for report in &mut prepared.intrinsic_report[..2] {
            report.source = ImageIntrinsicSource::Exif35mmEquivalent;
        }

        let project = prepared.to_project(true).unwrap();
        let rebuilt = prepare_image_project(&paths, &project).unwrap();

        assert_eq!(
            [
                project.groups[0].focal_is_guess,
                project.groups[1].focal_is_guess
            ],
            [false, true]
        );
        assert_eq!(rebuilt.images, prepared.images);
        assert_eq!(rebuilt.camera_models, prepared.camera_models);
        assert_eq!(rebuilt.image_intrinsics, prepared.image_intrinsics);
        let trust = |dataset: PreparedImageDataset| {
            let dataset = dataset.build_dataset(1).unwrap();
            [0, 1].map(|group| dataset.intrinsic_prior(IntrinsicId::new(group)))
        };
        assert_eq!(trust(rebuilt), trust(prepared));
    }

    #[test]
    fn a_dataset_whose_vectors_disagree_is_an_error_not_a_panic() {
        let root = tempfile::TempDir::new().unwrap();
        let paths = images(root.path(), &[(64, 48), (64, 48)]);
        let good = prepare_image_paths(&paths, &ImageDatasetConfig::default()).unwrap();

        let mut short_ids = good.clone();
        short_ids.image_intrinsics.pop();
        let mut short_report = good.clone();
        short_report.intrinsic_report.pop();
        let mut unknown_camera = good;
        unknown_camera.image_intrinsics[0] = IntrinsicId::new(9);

        for broken in [short_ids, short_report, unknown_camera] {
            let error = broken.to_project(true).unwrap_err();
            assert!(error.to_string().contains("do not match"), "{error}");
        }
    }

    #[test]
    fn saving_replaces_the_file_whole_and_a_failed_save_leaves_no_partial_file() {
        let root = tempfile::TempDir::new().unwrap();
        let file = root.path().join("project.json");
        project(vec![pinhole(900.0, &[0, 1])]).save(&file).unwrap();
        project(vec![pinhole(700.0, &[0, 1])]).save(&file).unwrap();
        assert_eq!(
            ImageProject::load(&file).unwrap().groups[0].focal_px,
            Some(700.0)
        );

        // The write succeeds beside the target and the rename onto a directory fails: the
        // partial file must not stay behind.
        let directory = root.path().join("occupied");
        std::fs::create_dir(&directory).unwrap();
        assert!(project(vec![pinhole(1.0, &[0])]).save(&directory).is_err());

        let mut names: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["occupied", "project.json"]);
    }

    #[test]
    fn an_image_the_project_leaves_out_is_never_read_so_a_broken_one_does_not_matter() {
        let root = tempfile::TempDir::new().unwrap();
        let mut paths = images(root.path(), &[(64, 48), (64, 48)]);
        let empty = root.path().join("a-empty.png");
        std::fs::write(&empty, b"").unwrap();
        paths.insert(0, empty.clone());
        assert!(image_size(&empty).is_err());

        let prepared =
            prepare_image_project(&paths, &project(vec![pinhole(900.0, &[0, 1])])).unwrap();

        assert_eq!(prepared.image_count(), 2);
        // Naming it is an error that names the file, which is what a project must never do.
        let named = ImageProjectGroup::new(
            CameraModelType::PinholeRadial1,
            Some(900.0),
            vec!["a-empty.png".to_owned(), "view-0.png".to_owned()],
        );
        let error = prepare_image_project(&paths, &project(vec![named])).unwrap_err();
        assert!(error.to_string().contains("a-empty.png"), "{error}");
    }
}
