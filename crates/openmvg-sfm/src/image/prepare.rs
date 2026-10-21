// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared image-directory ingestion for every SfM engine and CLI.
//!
//! This module owns image ordering, focal initialization, EXIF camera grouping, and workspace
//! construction. Feature extraction and engine scheduling are separate stages.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use openmvg_core::camera::{
    BrownConradyCamera, CameraModelType, RadialDistortion, SphericalCamera, TangentialDistortion,
};

use crate::image::exif::{
    ExifCameraIdentity, ExifFocalSource, ExifImageMetadata, exif_image_metadata, plausible_focal,
};
use crate::model::ImageMeta;
use crate::{Camera, IntrinsicId, Result, SfmError};

/// How [`prepare_image_paths`] initializes and groups image intrinsics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageIntrinsicPolicy {
    /// Estimate focal length per image, then share exact camera/lens/focal regimes.
    ExifGrouped,
    /// Use one caller-provided perspective calibration for every image.
    Shared {
        /// Calibration assigned to every image as intrinsic 0.
        camera: Camera,
    },
}

/// Image-directory ingestion settings for [`prepare_image_paths`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImageDatasetConfig {
    intrinsic_policy: ImageIntrinsicPolicy,
    focal_scale: f64,
    fallback_focal_multiplier: f64,
    #[serde(default = "default_camera_model")]
    camera_model: CameraModelType,
    #[serde(default)]
    share_unknown_cameras: bool,
}

config_accessors!(ImageDatasetConfig {
    /// How image intrinsics are initialized.
    ref intrinsic_policy, with_intrinsic_policy: ImageIntrinsicPolicy;
    /// Multiplier applied to every estimated focal length.
    copy focal_scale, with_focal_scale: f64;
    /// Focal length, as a multiple of the longest image edge, guessed when EXIF does not give one
    /// in pixels. When EXIF records the focal length in millimetres but nothing converts it, the
    /// guess applies at the camera's median focal length and scales with each image's.
    copy fallback_focal_multiplier, with_fallback_focal_multiplier: f64;
    /// Camera model declared for EXIF-grouped cameras: `PinholeSingle`, `PinholeRadial1`,
    /// `PinholeRadial2`, or `Spherical`. Radial coefficients start at zero, meaning unknown, for
    /// radial pre-calibration and self-calibrating bundle adjustment to estimate; a pinhole camera
    /// is never given distortion. `Spherical` expects 2:1 equirectangular images and gives one
    /// camera to each image width, whatever the EXIF says. An explicit shared calibration keeps
    /// its own model.
    copy camera_model, with_camera_model: CameraModelType;
    /// Give every image with no camera make and model, and the same dimensions, one shared
    /// intrinsic group instead of one group per image (default: off). The images are then
    /// asserted to come from one camera at one zoom setting, so a single focal length and
    /// distortion are estimated from all of them. Images whose EXIF names a camera keep their
    /// grouping by camera, lens, and focal length.
    copy share_unknown_cameras, with_share_unknown_cameras: bool;
});

impl Default for ImageDatasetConfig {
    fn default() -> Self {
        Self {
            intrinsic_policy: ImageIntrinsicPolicy::ExifGrouped,
            focal_scale: 1.0,
            fallback_focal_multiplier: 1.2,
            camera_model: default_camera_model(),
            share_unknown_cameras: false,
        }
    }
}

const fn default_camera_model() -> CameraModelType {
    CameraModelType::PinholeRadial1
}

/// Where an image's initial focal length came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ImageIntrinsicSource {
    /// EXIF focal length (mm) converted with the focal-plane pixel density.
    ExifFocalPlane,
    /// EXIF focal length (mm) converted with a known make/model sensor width.
    ExifSensorWidth,
    /// EXIF 35 mm-equivalent focal length, converted against the full-frame diagonal for the
    /// camera's native sensor format (3:2, or 4:3 for Four Thirds and small sensors).
    #[serde(rename = "exif_35mm_equivalent")]
    Exif35mmEquivalent,
    /// EXIF focal length (mm) converted with the pixels per millimetre of the same camera (make,
    /// model, and image size), learned from its images whose EXIF converts on its own.
    ExifSameCamera,
    /// EXIF focal length (mm) that nothing converts: the image-size guess at the camera's median
    /// focal length, scaled by this image's focal length, so zoom settings keep their ratios.
    ExifFocalGuess,
    /// No usable EXIF focal; the configured multiple of the longest image edge.
    ImageSizeFallback,
    /// The caller-provided shared calibration.
    ExplicitShared,
    /// The focal length of an [`ImageProject`](crate::image::ImageProject) group.
    Project,
    /// An equirectangular image: the spherical camera has no focal length to estimate, and the
    /// reported focal is the pixels per radian of its longitude axis.
    Equirectangular,
}

impl ImageIntrinsicSource {
    /// Name, the same spelling as its JSON form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExifFocalPlane => "exif_focal_plane",
            Self::ExifSensorWidth => "exif_sensor_width",
            Self::Exif35mmEquivalent => "exif_35mm_equivalent",
            Self::ExifSameCamera => "exif_same_camera",
            Self::ExifFocalGuess => "exif_focal_guess",
            Self::ImageSizeFallback => "image_size_fallback",
            Self::ExplicitShared => "explicit_shared",
            Self::Project => "project",
            Self::Equirectangular => "equirectangular",
        }
    }

    /// Whether the focal length is a guess for bundle adjustment to estimate rather than a value
    /// read from the image or its project.
    pub const fn is_guess(self) -> bool {
        matches!(self, Self::ExifFocalGuess | Self::ImageSizeFallback)
    }
}

/// Per-image record of how its initial intrinsics were chosen.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ImageIntrinsicReport {
    /// File name of the image, without its directory.
    pub image: String,
    /// Image width, in pixels.
    pub width: u32,
    /// Image height, in pixels.
    pub height: u32,
    /// Initial focal length in pixels, after the configured focal scale.
    pub focal_px: f64,
    /// Where the focal length came from.
    pub source: ImageIntrinsicSource,
    /// EXIF camera make; `None` when absent or under a shared calibration or a project.
    pub camera_make: Option<String>,
    /// EXIF camera model; `None` when absent or under a shared calibration or a project.
    pub camera_model: Option<String>,
    /// EXIF body serial number, if present.
    pub body_serial_number: Option<String>,
    /// EXIF lens model, if present.
    pub lens_model: Option<String>,
    /// EXIF lens serial number, if present.
    pub lens_serial_number: Option<String>,
    /// Intrinsic group the image was assigned to.
    pub intrinsic_id: IntrinsicId,
    /// Why EXIF metadata could not be read; `None` when it was read or not needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// Images and initial intrinsics ready to become a dataset.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PreparedImageDataset {
    /// Path-backed image metadata, in input order.
    pub images: Vec<ImageMeta>,
    /// One initial camera per intrinsic group, indexed by [`IntrinsicId`].
    pub camera_models: Vec<Camera>,
    /// Intrinsic group of each image, parallel to `images`.
    pub image_intrinsics: Vec<IntrinsicId>,
    /// How each image's intrinsics were chosen, parallel to `images`.
    pub intrinsic_report: Vec<ImageIntrinsicReport>,
}

impl PreparedImageDataset {
    /// Build the immutable dataset consumed by the supported SfM frontend.
    ///
    /// The dataset [has calibrated intrinsics](crate::Dataset::has_calibrated_intrinsics) when
    /// every image uses the explicit shared camera. Groups whose focal is a guess
    /// ([`ImageIntrinsicSource::is_guess`]) are
    /// [`IntrinsicPrior::Guess`](crate::model::IntrinsicPrior::Guess); every other group is
    /// trusted.
    pub fn build_dataset(
        self,
        root_seed: impl Into<openmvg_core::RunSeed>,
    ) -> Result<crate::Dataset> {
        let calibrated = self
            .intrinsic_report
            .iter()
            .all(|report| report.source == ImageIntrinsicSource::ExplicitShared);
        let priors = self.intrinsic_priors();
        crate::Dataset::builder()
            .root_seed(root_seed)
            .image_meta(self.images)
            .camera_models(self.camera_models)
            .intrinsic_priors(priors)
            .view_intrinsics(self.image_intrinsics)
            .build()
            .map(|dataset| dataset.with_calibrated_intrinsics(calibrated))
    }

    /// Builds a mutable workspace with exhaustive image pairs.
    ///
    /// # Errors
    ///
    /// Returns an `SfmError` when the images, cameras, or intrinsic ids are inconsistent.
    #[cfg(feature = "unstable-internals")]
    pub fn build_workspace(
        self,
        root_seed: impl Into<openmvg_core::RunSeed>,
    ) -> Result<crate::SfmWorkspace> {
        let priors = self.intrinsic_priors();
        crate::SfmWorkspace::builder()
            .root_seed(root_seed.into())
            .image_meta(self.images)
            .camera_models(self.camera_models)
            .intrinsic_priors(priors)
            .view_intrinsics(self.image_intrinsics)
            .exhaustive_pairs()
            .build()
    }

    /// The focal-length prior of each intrinsic group: a guess when its focal is one, trusted
    /// otherwise.
    fn intrinsic_priors(&self) -> Vec<crate::model::IntrinsicPrior> {
        let mut priors = vec![crate::model::IntrinsicPrior::Trusted; self.camera_models.len()];
        for report in &self.intrinsic_report {
            if report.source.is_guess()
                && let Some(prior) = priors.get_mut(report.intrinsic_id.index())
            {
                *prior = crate::model::IntrinsicPrior::Guess;
            }
        }
        priors
    }

    /// Number of images.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Number of distinct intrinsic groups (cameras).
    pub fn intrinsic_group_count(&self) -> usize {
        self.camera_models.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CameraIdentityKey {
    Known {
        make: String,
        model: String,
        body_serial_number: Option<String>,
        lens_model: Option<String>,
        lens_serial_number: Option<String>,
    },
    Unknown {
        image_index: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct IntrinsicGroupKey {
    camera: CameraIdentityKey,
    parameter_bits: [u64; 9],
}

fn camera_grouping_key(camera: &ExifCameraIdentity, image_index: usize) -> CameraIdentityKey {
    match (&camera.make, &camera.model) {
        (Some(make), Some(model)) => CameraIdentityKey::Known {
            make: make.to_ascii_lowercase(),
            model: model.to_ascii_lowercase(),
            body_serial_number: camera
                .body_serial_number
                .as_ref()
                .map(|serial| serial.to_ascii_lowercase()),
            lens_model: camera
                .lens_model
                .as_ref()
                .map(|lens| lens.to_ascii_lowercase()),
            lens_serial_number: camera
                .lens_serial_number
                .as_ref()
                .map(|serial| serial.to_ascii_lowercase()),
        },
        _ => CameraIdentityKey::Unknown { image_index },
    }
}

/// Group key of an EXIF-grouped image. The lens focal length in millimetres takes part even when
/// it does not convert to pixels, so a zoom lens at two settings is never one camera.
fn exif_group_key(
    camera: CameraIdentityKey,
    focal_px: f64,
    focal_mm: Option<f64>,
    principal_x: f64,
    principal_y: f64,
) -> IntrinsicGroupKey {
    IntrinsicGroupKey {
        camera,
        parameter_bits: [
            focal_px.to_bits(),
            focal_px.to_bits(),
            principal_x.to_bits(),
            principal_y.to_bits(),
            focal_mm.map_or(0, f64::to_bits),
            0,
            0,
            0,
            0,
        ],
    }
}

/// The initial focal length of one EXIF-grouped image and what it was read from.
#[derive(Debug, Clone, PartialEq)]
struct ResolvedFocal {
    camera: ExifCameraIdentity,
    focal_mm: Option<f64>,
    focal_px: f64,
    source: ImageIntrinsicSource,
    warning: Option<String>,
}

/// Reads the EXIF of every image and resolves their focal lengths together. An unreadable EXIF
/// block is a warning, the image keeping the image-size guess, and so is an EXIF source that was
/// overruled or ignored.
fn read_exif_focals(
    paths: &[PathBuf],
    sizes: &[(u32, u32)],
    fallback_multiplier: f64,
) -> Vec<ResolvedFocal> {
    let (metadata, warnings): (Vec<_>, Vec<_>) = paths
        .iter()
        .zip(sizes)
        .map(
            |(path, &(width, height))| match exif_image_metadata(path, width, height) {
                Ok(metadata) => {
                    let warning = metadata
                        .as_ref()
                        .and_then(|metadata| metadata.warning.clone());
                    (metadata, warning)
                }
                Err(error) => (None, Some(error.to_string())),
            },
        )
        .unzip();
    resolve_exif_focals(sizes, &metadata, fallback_multiplier)
        .into_iter()
        .zip(warnings)
        .map(|(resolved, warning)| ResolvedFocal {
            warning,
            ..resolved
        })
        .collect()
}

/// A camera as far as its pixel density goes: make, model, and image size in either
/// orientation, so a portrait image shares the density of its camera's landscape images.
type DensityKey = (String, String, u32, u32);

fn density_key(camera: &ExifCameraIdentity, width: u32, height: u32) -> Option<DensityKey> {
    let make = camera.make.as_deref()?.trim().to_ascii_lowercase();
    let model = camera.model.as_deref()?.trim().to_ascii_lowercase();
    Some((make, model, width.min(height), width.max(height)))
}

/// Median of `values`, the mean of the middle two for an even count; `None` when empty.
fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    match values.len() {
        0 => None,
        count if count % 2 == 1 => Some(values[middle]),
        _ => Some(0.5 * (values[middle - 1] + values[middle])),
    }
}

/// Initial focal length of each image from its EXIF, in three steps:
///
/// 1. An image whose EXIF converts on its own (focal-plane density, known sensor width, or 35 mm
///    equivalent) keeps that value.
/// 2. Its pixels per millimetre, `focal_px / focal_mm`, belong to the camera, not the lens. Their
///    median over a camera's converting images converts the focal length of its other images.
/// 3. A focal length that still does not convert scales the image-size guess by its ratio to the
///    camera's median focal length; an image without one gets the plain guess.
///
/// The result does not depend on the order of the images.
fn resolve_exif_focals(
    sizes: &[(u32, u32)],
    metadata: &[Option<ExifImageMetadata>],
    fallback_multiplier: f64,
) -> Vec<ResolvedFocal> {
    let mut densities: BTreeMap<DensityKey, Vec<f64>> = BTreeMap::new();
    let mut focals_mm: BTreeMap<DensityKey, Vec<f64>> = BTreeMap::new();
    for (&(width, height), metadata) in sizes.iter().zip(metadata) {
        let Some(metadata) = metadata else { continue };
        let (Some(key), Some(focal_mm)) = (
            density_key(&metadata.camera, width, height),
            metadata.focal_mm,
        ) else {
            continue;
        };
        if let Some(estimate) = metadata.focal {
            densities
                .entry(key.clone())
                .or_default()
                .push(estimate.focal_px / focal_mm);
        }
        focals_mm.entry(key).or_default().push(focal_mm);
    }
    let densities: BTreeMap<_, _> = densities
        .into_iter()
        .filter_map(|(key, values)| Some((key, median(values)?)))
        .collect();
    let median_focals_mm: BTreeMap<_, _> = focals_mm
        .into_iter()
        .filter_map(|(key, values)| Some((key, median(values)?)))
        .collect();

    sizes
        .iter()
        .zip(metadata)
        .map(|(&(width, height), metadata)| {
            let camera = metadata
                .as_ref()
                .map(|metadata| metadata.camera.clone())
                .unwrap_or_default();
            let focal_mm = metadata.as_ref().and_then(|metadata| metadata.focal_mm);
            let guess_px = f64::from(width.max(height)) * fallback_multiplier;
            let (focal_px, source) = match metadata.as_ref().and_then(|metadata| metadata.focal) {
                Some(estimate) => (
                    estimate.focal_px,
                    match estimate.source {
                        ExifFocalSource::FocalPlaneResolution => {
                            ImageIntrinsicSource::ExifFocalPlane
                        }
                        ExifFocalSource::KnownSensorWidth => ImageIntrinsicSource::ExifSensorWidth,
                        ExifFocalSource::Equivalent35mm => ImageIntrinsicSource::Exif35mmEquivalent,
                    },
                ),
                None => {
                    let key = density_key(&camera, width, height);
                    let plausible = |focal_px: &f64| plausible_focal(*focal_px, width, height);
                    let same_camera = focal_mm
                        .zip(key.as_ref().and_then(|key| densities.get(key)))
                        .map(|(focal_mm, density)| focal_mm * density)
                        .filter(plausible)
                        .map(|focal_px| (focal_px, ImageIntrinsicSource::ExifSameCamera));
                    let scaled_guess = focal_mm
                        .zip(key.as_ref().and_then(|key| median_focals_mm.get(key)))
                        .map(|(focal_mm, median_mm)| guess_px * focal_mm / median_mm)
                        .filter(plausible)
                        .map(|focal_px| (focal_px, ImageIntrinsicSource::ExifFocalGuess));
                    same_camera
                        .or(scaled_guess)
                        .unwrap_or((guess_px, ImageIntrinsicSource::ImageSizeFallback))
                }
            };
            ResolvedFocal {
                camera,
                focal_mm,
                focal_px,
                source,
                warning: None,
            }
        })
        .collect()
}

/// How much of a file [`image_size`] reads first. `image::image_dimensions` reads a whole JPEG
/// before it parses the header, which costs milliseconds per photo; the headers of a camera JPEG
/// fit in far less.
const HEADER_BYTES: u64 = 256 * 1024;

/// Pixel size of an image, read from the start of the file when that is enough and from the whole
/// file otherwise: a TIFF whose directory is at the end, or a JPEG with large metadata, takes the
/// slow path and gives the same size.
///
/// # Errors
///
/// Returns an `SfmError` that names the file when it cannot be read or is not a supported image,
/// as for an empty file left by a failed copy.
pub fn image_size(path: &Path) -> Result<(u32, u32)> {
    let from_start = || {
        let mut head = Vec::new();
        std::fs::File::open(path)
            .ok()?
            .take(HEADER_BYTES)
            .read_to_end(&mut head)
            .ok()?;
        image::ImageReader::new(std::io::Cursor::new(head))
            .with_guessed_format()
            .ok()?
            .into_dimensions()
            .ok()
    };
    if let Some(size) = from_start() {
        return Ok(size);
    }
    image::image_dimensions(path).map_err(|error| match error {
        image::ImageError::IoError(source) => SfmError::ReadFile {
            path: path.to_path_buf(),
            source,
        },
        error => SfmError::InvalidInput(format!("failed to read {}: {error}", path.display())),
    })
}

/// The spherical camera of a 2:1 equirectangular image.
pub(super) fn equirectangular_camera(path: &Path, width: u32, height: u32) -> Result<Camera> {
    let not_equirectangular = |detail: String| {
        SfmError::InvalidInput(format!(
            "{} is not a 2:1 equirectangular image: {detail}",
            path.display()
        ))
    };
    let sphere =
        SphericalCamera::try_new(width).map_err(|error| not_equirectangular(error.to_string()))?;
    if sphere.height() != height {
        return Err(not_equirectangular(format!("{width}x{height} pixels")));
    }
    Ok(Camera::Spherical(sphere))
}

/// Initial camera of an EXIF group: the declared model with zero, i.e. unknown, distortion.
/// Only models whose every parameter self-calibrating bundle adjustment refines are accepted.
pub(super) fn initial_camera(
    model: CameraModelType,
    focal_px: f64,
    principal_x: f64,
    principal_y: f64,
) -> Result<Camera> {
    let radial = match model {
        CameraModelType::PinholeSingle => {
            return Ok(Camera::try_pinhole(
                focal_px,
                focal_px,
                principal_x,
                principal_y,
            )?);
        }
        CameraModelType::PinholeRadial1 => RadialDistortion::K1(0.0),
        CameraModelType::PinholeRadial2 => RadialDistortion::K1K2(0.0, 0.0),
        CameraModelType::PinholeDouble
        | CameraModelType::PinholeRadial3
        | CameraModelType::PinholeBrown
        | CameraModelType::Fisheye
        | CameraModelType::Spherical => {
            return Err(SfmError::UnsupportedCameraModel {
                context: "EXIF-grouped image ingestion (use pinhole, pinhole_radial1, or pinhole_radial2)",
                camera_model: model.as_str(),
            });
        }
    };
    Ok(Camera::BrownConrady(BrownConradyCamera::try_with_focal(
        focal_px,
        principal_x,
        principal_y,
        radial,
        TangentialDistortion::None,
    )?))
}

fn assign_intrinsic_group(
    groups: &mut BTreeMap<IntrinsicGroupKey, IntrinsicId>,
    camera_models: &mut Vec<Camera>,
    key: IntrinsicGroupKey,
    value: Camera,
) -> IntrinsicId {
    if let Some(&id) = groups.get(&key) {
        return id;
    }
    let id = IntrinsicId::new(camera_models.len());
    camera_models.push(value);
    groups.insert(key, id);
    id
}

pub(super) fn camera_focal_px(camera: &Camera) -> Result<f64> {
    match camera {
        Camera::Pinhole(camera) => Ok(camera.focal_x()),
        Camera::BrownConrady(camera) => Ok(camera.focal_x()),
        Camera::Fisheye(camera) => Ok(camera.focal_x()),
        Camera::Spherical(camera) => Ok(f64::from(camera.width()) / std::f64::consts::TAU),
    }
}

/// Reads image dimensions and EXIF metadata and assigns initial intrinsics to `paths`.
///
/// Under [`ImageIntrinsicPolicy::ExifGrouped`], images share an intrinsic only when their
/// camera identity, focal length (in millimetres when EXIF records it, and in pixels), and
/// principal point (the image center) are identical. An image whose EXIF focal length does not
/// convert to pixels on its own borrows the pixels per millimetre of its camera's other images;
/// see [`ImageIntrinsicSource`] for every source.
///
/// # Errors
///
/// Returns an `SfmError` for fewer than two images, a non-positive focal scale or multiplier,
/// an unsupported camera model, or an image whose dimensions cannot be read.
pub fn prepare_image_paths(
    paths: &[PathBuf],
    config: &ImageDatasetConfig,
) -> Result<PreparedImageDataset> {
    if paths.len() < 2 {
        return Err(SfmError::InvalidInput(
            "SfM image ingestion requires at least two images".to_owned(),
        ));
    }
    if !config.focal_scale.is_finite() || config.focal_scale <= 0.0 {
        return Err(SfmError::InvalidInput(
            "image focal scale must be finite and positive".to_owned(),
        ));
    }
    if !config.fallback_focal_multiplier.is_finite() || config.fallback_focal_multiplier <= 0.0 {
        return Err(SfmError::InvalidInput(
            "image-size focal multiplier must be finite and positive".to_owned(),
        ));
    }
    let spherical = config.camera_model == CameraModelType::Spherical;
    if matches!(config.intrinsic_policy, ImageIntrinsicPolicy::ExifGrouped) && !spherical {
        initial_camera(config.camera_model, 1.0, 0.5, 0.5)?;
    }

    let mut images = Vec::with_capacity(paths.len());
    let mut camera_models = Vec::new();
    let mut image_intrinsics = Vec::with_capacity(paths.len());
    let mut intrinsic_report = Vec::with_capacity(paths.len());
    let mut groups = BTreeMap::new();
    let mut spherical_groups = BTreeMap::new();

    let sizes = paths
        .iter()
        .map(|path| image_size(path))
        .collect::<Result<Vec<_>>>()?;
    // EXIF focals are resolved for all images at once: an image whose focal length does not
    // convert on its own learns its camera's pixel density from the others.
    let exif_focals = (matches!(config.intrinsic_policy, ImageIntrinsicPolicy::ExifGrouped)
        && !spherical)
        .then(|| read_exif_focals(paths, &sizes, config.fallback_focal_multiplier));

    for (index, (path, &(width, height))) in paths.iter().zip(&sizes).enumerate() {
        images.push(ImageMeta::path(path, width, height));

        let (focal_px, source, camera, warning, intrinsic_id) = match &config.intrinsic_policy {
            ImageIntrinsicPolicy::Shared { camera: shared } => {
                let id = if camera_models.is_empty() {
                    camera_models.push(shared.clone());
                    IntrinsicId::new(0)
                } else {
                    IntrinsicId::new(0)
                };
                (
                    camera_focal_px(shared)?,
                    ImageIntrinsicSource::ExplicitShared,
                    ExifCameraIdentity::default(),
                    None,
                    id,
                )
            }
            ImageIntrinsicPolicy::ExifGrouped if spherical => {
                // An equirectangular image has no focal length: one camera per image width,
                // whatever its EXIF says.
                let camera = equirectangular_camera(path, width, height)?;
                let id = *spherical_groups.entry(width).or_insert_with(|| {
                    camera_models.push(camera);
                    IntrinsicId::new(camera_models.len() - 1)
                });
                (
                    f64::from(width) / std::f64::consts::TAU,
                    ImageIntrinsicSource::Equirectangular,
                    ExifCameraIdentity::default(),
                    None,
                    id,
                )
            }
            ImageIntrinsicPolicy::ExifGrouped => {
                let resolved = exif_focals
                    .as_ref()
                    .and_then(|focals| focals.get(index))
                    .cloned()
                    .ok_or_else(|| {
                        SfmError::InvalidInput("EXIF focal lengths were not resolved".to_owned())
                    })?;
                let camera = resolved.camera;
                let warning = resolved.warning;
                let source = resolved.source;
                let focal_mm = resolved.focal_mm;
                let focal_px = resolved.focal_px * config.focal_scale;
                let principal_x = width as f64 / 2.0;
                let principal_y = height as f64 / 2.0;
                let initialized =
                    initial_camera(config.camera_model, focal_px, principal_x, principal_y)?;
                let id = assign_intrinsic_group(
                    &mut groups,
                    &mut camera_models,
                    exif_group_key(
                        camera_grouping_key(
                            &camera,
                            if config.share_unknown_cameras {
                                0
                            } else {
                                index
                            },
                        ),
                        focal_px,
                        focal_mm,
                        principal_x,
                        principal_y,
                    ),
                    initialized,
                );
                (focal_px, source, camera, warning, id)
            }
        };
        image_intrinsics.push(intrinsic_id);
        intrinsic_report.push(ImageIntrinsicReport {
            image: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned(),
            width,
            height,
            focal_px,
            source,
            camera_make: camera.make,
            camera_model: camera.model,
            body_serial_number: camera.body_serial_number,
            lens_model: camera.lens_model,
            lens_serial_number: camera.lens_serial_number,
            intrinsic_id,
            warning,
        });
    }

    Ok(PreparedImageDataset {
        images,
        camera_models,
        image_intrinsics,
        intrinsic_report,
    })
}

pub fn parse_pinhole_intrinsics_matrix(content: &str, source: &Path) -> Result<Camera> {
    let values = content
        .split_whitespace()
        .map(|token| {
            token.parse::<f64>().map_err(|error| {
                SfmError::InvalidInput(format!(
                    "cannot parse '{token}' as a number in {}: {error}",
                    source.display()
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != 9 || values.iter().any(|value| !value.is_finite()) {
        return Err(SfmError::InvalidInput(format!(
            "{} must contain exactly nine finite matrix values",
            source.display()
        )));
    }
    let matrix_scale = values.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let homogeneous_scale = values[8];
    if matrix_scale == 0.0 || homogeneous_scale.abs() <= f64::EPSILON * matrix_scale {
        return Err(SfmError::InvalidInput(format!(
            "{} has an invalid homogeneous scale K[2,2]",
            source.display()
        )));
    }
    let normalized = values
        .into_iter()
        .map(|value| value / homogeneous_scale)
        .collect::<Vec<_>>();
    let scale = normalized.iter().copied().map(f64::abs).fold(1.0, f64::max);
    let tolerance = 1e-12 * scale;
    if normalized[1].abs() > tolerance
        || normalized[3].abs() > tolerance
        || normalized[6].abs() > tolerance
        || normalized[7].abs() > tolerance
        || (normalized[8] - 1.0).abs() > tolerance
        || normalized[0] <= 0.0
        || normalized[4] <= 0.0
    {
        return Err(SfmError::InvalidInput(format!(
            "{} is not a valid zero-skew pinhole calibration matrix",
            source.display()
        )));
    }
    Ok(Camera::try_pinhole(
        normalized[0],
        normalized[4],
        normalized[2],
        normalized[5],
    )?)
}

/// The `K.txt` calibration matrix of a dataset: in `root/images`, in `root`, or beside
/// `first_image`, in that order.
pub fn find_calibration_file(root: &Path, first_image: &Path) -> Option<PathBuf> {
    [
        root.join("images").join("K.txt"),
        root.join("K.txt"),
        first_image.parent().unwrap_or(root).join("K.txt"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

/// Reads a zero-skew pinhole camera from a text file holding the nine values of `K`, row major.
///
/// The matrix is normalized so that `K[2,2] = 1`.
///
/// # Errors
///
/// Returns an `SfmError` when the file cannot be read or is not a valid zero-skew pinhole
/// matrix.
pub fn load_pinhole_intrinsics_matrix(path: &Path) -> Result<Camera> {
    let content = std::fs::read_to_string(path).map_err(|source| SfmError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    // Some Windows editors start a UTF-8 file with a byte order mark.
    let content = content.strip_prefix('\u{feff}').unwrap_or(&content);
    parse_pinhole_intrinsics_matrix(content, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(model: &str) -> ExifCameraIdentity {
        ExifCameraIdentity {
            make: Some("Example".to_owned()),
            model: Some(model.to_owned()),
            body_serial_number: None,
            lens_model: Some("Prime".to_owned()),
            lens_serial_number: None,
        }
    }

    #[test]
    fn only_groups_with_a_guessed_focal_are_guesses() {
        use crate::model::IntrinsicPrior;
        let report = |image: &str, source, intrinsic| ImageIntrinsicReport {
            image: image.to_owned(),
            width: 640,
            height: 480,
            focal_px: 768.0,
            source,
            camera_make: None,
            camera_model: None,
            body_serial_number: None,
            lens_model: None,
            lens_serial_number: None,
            intrinsic_id: IntrinsicId::new(intrinsic),
            warning: None,
        };
        let prepared = PreparedImageDataset {
            images: vec![
                ImageMeta::path("a.jpg", 640, 480),
                ImageMeta::path("b.jpg", 640, 480),
                ImageMeta::path("c.jpg", 640, 480),
                ImageMeta::path("d.jpg", 640, 480),
                ImageMeta::path("e.jpg", 640, 480),
            ],
            camera_models: vec![
                Camera::try_pinhole(768.0, 768.0, 320.0, 240.0).unwrap(),
                Camera::try_pinhole(700.0, 700.0, 320.0, 240.0).unwrap(),
                Camera::try_pinhole(600.0, 600.0, 320.0, 240.0).unwrap(),
                Camera::try_pinhole(900.0, 900.0, 320.0, 240.0).unwrap(),
            ],
            image_intrinsics: vec![
                IntrinsicId::new(0),
                IntrinsicId::new(1),
                IntrinsicId::new(1),
                IntrinsicId::new(2),
                IntrinsicId::new(3),
            ],
            intrinsic_report: vec![
                report("a.jpg", ImageIntrinsicSource::ImageSizeFallback, 0),
                report("b.jpg", ImageIntrinsicSource::Exif35mmEquivalent, 1),
                report("c.jpg", ImageIntrinsicSource::Exif35mmEquivalent, 1),
                report("d.jpg", ImageIntrinsicSource::ExifFocalGuess, 2),
                report("e.jpg", ImageIntrinsicSource::ExifSameCamera, 3),
            ],
        };

        let dataset = prepared.build_dataset(7).unwrap();

        assert_eq!(
            dataset.intrinsic_prior(IntrinsicId::new(0)),
            Some(IntrinsicPrior::Guess)
        );
        assert_eq!(
            dataset.intrinsic_prior(IntrinsicId::new(1)),
            Some(IntrinsicPrior::Trusted)
        );
        assert_eq!(
            dataset.intrinsic_prior(IntrinsicId::new(2)),
            Some(IntrinsicPrior::Guess)
        );
        assert_eq!(
            dataset.intrinsic_prior(IntrinsicId::new(3)),
            Some(IntrinsicPrior::Trusted)
        );
    }

    /// EXIF of an image from `model` at focal length `focal_mm`, converted to `focal_px` when given.
    fn exif(model: &str, focal_mm: Option<f64>, focal_px: Option<f64>) -> ExifImageMetadata {
        ExifImageMetadata {
            camera: camera(model),
            focal_mm,
            focal: focal_px.map(|focal_px| crate::image::exif::ExifFocalEstimate {
                focal_px,
                source: ExifFocalSource::FocalPlaneResolution,
            }),
            warning: None,
        }
    }

    fn focal_and_source(resolved: &ResolvedFocal) -> (f64, ImageIntrinsicSource) {
        (resolved.focal_px, resolved.source)
    }

    #[test]
    fn a_converting_image_teaches_its_camera_pixel_density() {
        // A landscape image converts on its own; a portrait image of the same camera at another
        // zoom setting borrows its 151.5 px/mm. The same model at another image size does not.
        let sizes = [(5616, 3744), (3744, 5616), (2808, 1872)];
        let metadata = [
            Some(exif("Body", Some(24.0), Some(24.0 * 151.5))),
            Some(exif("Body", Some(135.0), None)),
            Some(exif("Body", Some(50.0), None)),
        ];

        let resolved = resolve_exif_focals(&sizes, &metadata, 1.2);

        assert_eq!(
            focal_and_source(&resolved[0]),
            (24.0 * 151.5, ImageIntrinsicSource::ExifFocalPlane)
        );
        assert_eq!(
            focal_and_source(&resolved[1]),
            (135.0 * 151.5, ImageIntrinsicSource::ExifSameCamera)
        );
        assert_eq!(resolved[2].source, ImageIntrinsicSource::ExifFocalGuess);
        assert_eq!(resolved[1].focal_mm, Some(135.0));
    }

    #[test]
    fn unconverted_zoom_settings_keep_their_focal_ratio_as_guesses() {
        // No image converts: the 1.2 x long-side guess sits at the median focal length (50 mm)
        // and the others keep their ratio to it.
        let sizes = [(6000, 4000); 3];
        let metadata = [24.0, 50.0, 96.0].map(|focal_mm| Some(exif("Zoom", Some(focal_mm), None)));

        let resolved = resolve_exif_focals(&sizes, &metadata, 1.2);

        let guess = 1.2 * 6000.0;
        for (resolved, focal_mm) in resolved.iter().zip([24.0, 50.0, 96.0]) {
            assert_eq!(resolved.source, ImageIntrinsicSource::ExifFocalGuess);
            assert!(resolved.source.is_guess());
            assert!((resolved.focal_px - guess * focal_mm / 50.0).abs() < 1e-9);
        }
    }

    #[test]
    fn an_image_without_a_focal_length_gets_the_plain_guess() {
        let sizes = [(4000, 3000), (4000, 3000)];
        let metadata = [None, Some(exif("NoFocal", None, None))];

        let resolved = resolve_exif_focals(&sizes, &metadata, 1.2);

        for resolved in &resolved {
            assert_eq!(
                focal_and_source(resolved),
                (4800.0, ImageIntrinsicSource::ImageSizeFallback)
            );
        }
    }

    #[test]
    fn an_implausible_borrowed_or_scaled_focal_falls_back_to_the_guess() {
        // 0.5 mm on a camera learned at 151.5 px/mm is under a 3 mm full-frame equivalent.
        let sizes = [(5616, 3744), (5616, 3744)];
        let metadata = [
            Some(exif("Body", Some(24.0), Some(24.0 * 151.5))),
            Some(exif("Body", Some(0.5), None)),
        ];

        let resolved = resolve_exif_focals(&sizes, &metadata, 1.2);

        assert_eq!(
            focal_and_source(&resolved[1]),
            (1.2 * 5616.0, ImageIntrinsicSource::ImageSizeFallback)
        );
    }

    #[test]
    fn focal_resolution_does_not_depend_on_image_order() {
        let sizes = [
            (5616, 3744),
            (3744, 5616),
            (5616, 3744),
            (6000, 4000),
            (6000, 4000),
        ];
        let metadata = [
            Some(exif("Body", Some(24.0), Some(24.0 * 151.0))),
            Some(exif("Body", Some(135.0), None)),
            Some(exif("Body", Some(70.0), Some(70.0 * 152.0))),
            Some(exif("Zoom", Some(18.0), None)),
            Some(exif("Zoom", Some(55.0), None)),
        ];
        let forward = resolve_exif_focals(&sizes, &metadata, 1.2);

        let order = [4, 2, 0, 3, 1];
        let reordered_sizes: Vec<_> = order.iter().map(|&index| sizes[index]).collect();
        let reordered_metadata: Vec<_> =
            order.iter().map(|&index| metadata[index].clone()).collect();
        let reordered = resolve_exif_focals(&reordered_sizes, &reordered_metadata, 1.2);

        for (position, &index) in order.iter().enumerate() {
            assert_eq!(reordered[position], forward[index]);
        }
    }

    #[test]
    fn zoom_settings_with_equal_pixel_focals_are_separate_groups() {
        let identity = || camera_grouping_key(&camera("Zoom"), 0);
        let at = |focal_mm| exif_group_key(identity(), 7200.0, Some(focal_mm), 3000.0, 2000.0);

        assert_ne!(at(24.0), at(50.0));
        assert_eq!(at(24.0), at(24.0));
    }

    #[test]
    fn source_labels_match_their_json_names() {
        use ImageIntrinsicSource::*;
        for source in [
            ExifFocalPlane,
            ExifSensorWidth,
            Exif35mmEquivalent,
            ExifSameCamera,
            ExifFocalGuess,
            ImageSizeFallback,
            ExplicitShared,
            Project,
            Equirectangular,
        ] {
            assert_eq!(
                serde_json::to_string(&source).unwrap(),
                format!("\"{}\"", source.as_str())
            );
        }
    }

    #[test]
    fn groups_equal_camera_and_intrinsics() {
        let value = Camera::try_pinhole(1000.0, 1000.0, 500.0, 400.0).unwrap();
        let mut groups = BTreeMap::new();
        let mut cameras = Vec::new();
        let first = assign_intrinsic_group(
            &mut groups,
            &mut cameras,
            exif_group_key(
                camera_grouping_key(&camera("A"), 0),
                1000.0,
                None,
                500.0,
                400.0,
            ),
            value.clone(),
        );
        let second = assign_intrinsic_group(
            &mut groups,
            &mut cameras,
            exif_group_key(
                camera_grouping_key(&camera("A"), 1),
                1000.0,
                None,
                500.0,
                400.0,
            ),
            value,
        );
        assert_eq!(first, second);
        assert_eq!(cameras.len(), 1);
    }

    #[test]
    fn separates_focal_regimes_unknown_cameras_and_models() {
        assert_ne!(
            exif_group_key(
                camera_grouping_key(&camera("A"), 0),
                1000.0,
                None,
                500.0,
                400.0
            ),
            exif_group_key(
                camera_grouping_key(&camera("A"), 1),
                1200.0,
                None,
                500.0,
                400.0
            ),
        );
        assert_ne!(
            exif_group_key(
                camera_grouping_key(&camera("A"), 0),
                1000.0,
                None,
                500.0,
                400.0
            ),
            exif_group_key(
                camera_grouping_key(&camera("B"), 1),
                1000.0,
                None,
                500.0,
                400.0
            ),
        );
        assert_ne!(
            exif_group_key(
                camera_grouping_key(&ExifCameraIdentity::default(), 0),
                1000.0,
                None,
                500.0,
                400.0
            ),
            exif_group_key(
                camera_grouping_key(&ExifCameraIdentity::default(), 1),
                1000.0,
                None,
                500.0,
                400.0
            ),
        );
    }

    #[test]
    fn exif_groups_declare_the_configured_model_with_unknown_distortion() {
        let pinhole = initial_camera(CameraModelType::PinholeSingle, 900.0, 320.0, 240.0).unwrap();
        assert_eq!(
            pinhole,
            Camera::try_pinhole(900.0, 900.0, 320.0, 240.0).unwrap()
        );
        for (model, radial) in [
            (CameraModelType::PinholeRadial1, RadialDistortion::K1(0.0)),
            (
                CameraModelType::PinholeRadial2,
                RadialDistortion::K1K2(0.0, 0.0),
            ),
        ] {
            let Camera::BrownConrady(camera) = initial_camera(model, 900.0, 320.0, 240.0).unwrap()
            else {
                panic!("{model} must initialize a Brown-Conrady camera");
            };
            assert_eq!(*camera.radial_distortion(), radial);
            assert_eq!(*camera.tangential_distortion(), TangentialDistortion::None);
            assert_eq!(camera.focal_x(), 900.0);
        }
        assert_eq!(
            ImageDatasetConfig::default().camera_model,
            CameraModelType::PinholeRadial1
        );
    }

    #[test]
    fn rejects_models_self_calibration_cannot_refine_before_reading_images() {
        for model in [
            CameraModelType::PinholeDouble,
            CameraModelType::PinholeRadial3,
            CameraModelType::PinholeBrown,
            CameraModelType::Fisheye,
        ] {
            let config = ImageDatasetConfig {
                camera_model: model,
                ..ImageDatasetConfig::default()
            };
            let error = prepare_image_paths(
                &[
                    PathBuf::from("/missing/a.jpg"),
                    PathBuf::from("/missing/b.jpg"),
                ],
                &config,
            )
            .unwrap_err();
            assert!(
                matches!(error, SfmError::UnsupportedCameraModel { .. }),
                "{model}: {error}"
            );
        }
    }

    #[test]
    fn parses_scaled_non_square_pinhole_matrix() {
        let parsed =
            parse_pinhole_intrinsics_matrix("1400 0 800 0 1420 600 0 0 2", Path::new("K.txt"))
                .unwrap();
        assert_eq!(
            parsed,
            Camera::try_pinhole(700.0, 710.0, 400.0, 300.0).unwrap()
        );
    }

    #[test]
    fn loads_a_k_file_that_starts_with_a_byte_order_mark() {
        let root = tempfile::TempDir::new().unwrap();
        let file = root.path().join("K.txt");
        std::fs::write(&file, "\u{feff}1400 0 800\n0 1420 600\n0 0 2\n").unwrap();
        assert_eq!(
            load_pinhole_intrinsics_matrix(&file).unwrap(),
            Camera::try_pinhole(700.0, 710.0, 400.0, 300.0).unwrap()
        );
    }

    /// A JPEG whose metadata, ahead of the picture, is larger than [`HEADER_BYTES`].
    fn jpeg_with_large_metadata(path: &Path, width: u32, height: u32) {
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(width, height)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let jpeg = jpeg.into_inner();
        let mut file = jpeg[..2].to_vec();
        // Six APP2 segments of about 60 KB each, then the original segments.
        for _ in 0..6 {
            file.extend([0xFF, 0xE2, 0xEA, 0x60]);
            file.extend(vec![0_u8; 0xEA60 - 2]);
        }
        file.extend(&jpeg[2..]);
        // The picture header starts beyond what `image_size` reads first.
        assert!(file.len() - jpeg.len() > usize::try_from(HEADER_BYTES).unwrap());
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn the_image_size_is_the_same_whether_the_header_is_near_the_start_or_far_from_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let png = dir.path().join("a.png");
        image::DynamicImage::new_rgb8(64, 48).save(&png).unwrap();
        let large = dir.path().join("large.jpg");
        jpeg_with_large_metadata(&large, 40, 30);
        let small = dir.path().join("small.jpg");
        image::DynamicImage::new_rgb8(40, 30).save(&small).unwrap();

        for (path, size) in [(&png, (64, 48)), (&large, (40, 30)), (&small, (40, 30))] {
            assert_eq!(image_size(path).unwrap(), size, "{}", path.display());
            assert_eq!(image::image_dimensions(path).unwrap(), size);
        }
        assert!(matches!(
            image_size(&dir.path().join("absent.png")),
            Err(SfmError::ReadFile { .. })
        ));
    }
}
