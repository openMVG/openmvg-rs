// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! EXIF-based per-image focal-length detection.
//!
//! The API distinguishes missing EXIF data from I/O and parse errors, so the dataset
//! loader can fall back to a focal length derived from the image size when EXIF gives none.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

const FULL_FRAME_WIDTH_MM: f64 = 36.0;
/// Diagonal of the 36 x 24 mm frame, which defines the 35 mm-equivalent focal length.
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615_305_567_87;
const MIN_PLAUSIBLE_EQUIVALENT_MM: f64 = 3.0;
const MAX_PLAUSIBLE_EQUIVALENT_MM: f64 = 3_000.0;
const MAX_FOCAL_PLANE_AXIS_DISAGREEMENT: f64 = 0.02;
/// Long side of the smallest and largest sensors a focal-plane resolution may imply: below a
/// phone's, or above medium format's, the resolution is the wrong one.
const MIN_SENSOR_LONG_SIDE_MM: f64 = 2.0;
const MAX_SENSOR_LONG_SIDE_MM: f64 = 70.0;
/// How far a focal-plane or known-sensor-width focal may differ from the 35 mm-equivalent one
/// before the 35 mm equivalent wins: past rounding and the cameras that round it to a marketing
/// value, a density the image no longer matches or a width for another crop mode.
const MAX_FOCAL_PLANE_EQUIVALENT_DISAGREEMENT: f64 = 0.08;
const MAX_SENSOR_WIDTH_EQUIVALENT_DISAGREEMENT: f64 = 0.10;
/// Relative tolerance on an image aspect ratio matching the sensor's.
const ASPECT_TOLERANCE: f64 = 0.02;

#[derive(Debug, Clone, Copy)]
struct SensorWidthProfile {
    make: &'static str,
    model: &'static str,
    width_mm: f64,
}

/// Exact camera profiles used only when EXIF provides the physical focal length
/// but no focal-plane pixel density. Exact make+model matching avoids silently
/// assigning the wrong sensor to similarly named cameras.
const SENSOR_WIDTH_PROFILES: &[SensorWidthProfile] = &[
    SensorWidthProfile {
        make: "Canon",
        model: "Canon EOS 5D Mark II",
        width_mm: 36.0,
    },
    SensorWidthProfile {
        make: "Canon",
        model: "Canon EOS 70D",
        width_mm: 22.5,
    },
    // Values are independently recorded camera specifications, cross-checked
    // against the established openMVG camera-width database. Only the small
    // set needed by this workspace is retained here rather than copying that
    // separately licensed database wholesale.
    // The camera writes a rounded 24 mm equivalent even for 16:9 crops; the
    // physical 4.49 mm focal and sensor width are substantially more accurate.
    SensorWidthProfile {
        make: "DJI",
        model: "FC7303",
        width_mm: 6.16,
    },
    SensorWidthProfile {
        make: "EASTMAN KODAK COMPANY",
        model: "KODAK Z612 ZOOM DIGITAL CAMERA",
        width_mm: 5.744,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExifFocalSource {
    FocalPlaneResolution,
    KnownSensorWidth,
    Equivalent35mm,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExifFocalEstimate {
    pub focal_px: f64,
    pub source: ExifFocalSource,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExifCameraIdentity {
    pub make: Option<String>,
    pub model: Option<String>,
    pub body_serial_number: Option<String>,
    pub lens_model: Option<String>,
    pub lens_serial_number: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExifImageMetadata {
    pub camera: ExifCameraIdentity,
    /// The lens focal length in millimetres (`FocalLength`), whether or not it converts to pixels.
    pub focal_mm: Option<f64>,
    pub focal: Option<ExifFocalEstimate>,
    /// Why a source was overruled or ignored, when one was.
    pub warning: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ExifFocalError {
    #[error("invalid image dimensions {width}x{height} for `{path}`")]
    InvalidDimensions {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    #[error("failed to open image `{path}` for EXIF: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse EXIF from `{path}`: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: exif::Error,
    },
}

/// Parse camera identity and focal metadata in one pass.
///
/// Focal sources are tried in descending reliability:
///
/// 1. Physical `FocalLength` with focal-plane X/Y pixel density, when the density implies a real
///    sensor size (2 to 70 mm on the long side).
/// 2. Physical `FocalLength` with an exact make+model sensor-width profile.
/// 3. Rounded `FocalLengthIn35mmFilm`, converted against the full-frame diagonal for the
///    camera's native sensor format. It comes last because it is integer-valued and some cameras
///    round it to a marketing value.
///
/// When the 35 mm equivalent is present, it also checks the first two: a focal-plane focal more
/// than 8% away, or a sensor-width focal more than 10% away, gives way to it with a warning.
pub fn exif_image_metadata(
    path: &Path,
    width: u32,
    height: u32,
) -> Result<Option<ExifImageMetadata>, ExifFocalError> {
    if width == 0 || height == 0 {
        return Err(ExifFocalError::InvalidDimensions {
            path: path.to_path_buf(),
            width,
            height,
        });
    }

    let file = File::open(path).map_err(|source| ExifFocalError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let exif = match exif::Reader::new().read_from_container(&mut reader) {
        Ok(exif) => exif,
        Err(exif::Error::NotFound(_)) | Err(exif::Error::BlankValue(_)) => return Ok(None),
        Err(source) => {
            return Err(ExifFocalError::Parse {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    let camera = ExifCameraIdentity {
        make: exif_ascii(&exif, exif::Tag::Make),
        model: exif_ascii(&exif, exif::Tag::Model),
        body_serial_number: exif_ascii(&exif, exif::Tag::BodySerialNumber),
        lens_model: exif_ascii(&exif, exif::Tag::LensModel),
        lens_serial_number: exif_ascii(&exif, exif::Tag::LensSerialNumber),
    };

    let fields = FocalFields::read(&exif, &camera);
    let (focal, warning) = estimate_focal(&fields, width, height);
    Ok(Some(ExifImageMetadata {
        camera,
        focal_mm: fields.focal_mm,
        focal,
        warning,
    }))
}

/// The EXIF values the focal estimate reads.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct FocalFields {
    focal_mm: Option<f64>,
    focal_35mm: Option<f64>,
    /// `FocalPlaneXResolution` and `FocalPlaneYResolution`.
    plane_resolution: (Option<f64>, Option<f64>),
    plane_unit_mm: Option<f64>,
    /// `PixelXDimension` and `PixelYDimension`, the image size at capture.
    pixel_dimensions: (Option<u32>, Option<u32>),
    sensor_width_mm: Option<f64>,
}

impl FocalFields {
    fn read(exif: &exif::Exif, camera: &ExifCameraIdentity) -> Self {
        let field = |tag| exif.get_field(tag, exif::In::PRIMARY);
        let positive_rational = |tag| {
            field(tag)
                .and_then(|field| rational_value(&field.value))
                .filter(|value| *value > 0.0)
        };
        let positive_unsigned = |tag| {
            field(tag)
                .and_then(|field| unsigned_value(&field.value))
                .filter(|value| *value > 0)
        };
        Self {
            focal_mm: positive_rational(exif::Tag::FocalLength),
            focal_35mm: positive_unsigned(exif::Tag::FocalLengthIn35mmFilm).map(f64::from),
            plane_resolution: (
                positive_rational(exif::Tag::FocalPlaneXResolution),
                positive_rational(exif::Tag::FocalPlaneYResolution),
            ),
            plane_unit_mm: field(exif::Tag::FocalPlaneResolutionUnit)
                .map(|field| resolution_unit_mm(&field.value))
                .unwrap_or(Some(25.4)),
            pixel_dimensions: (
                positive_unsigned(exif::Tag::PixelXDimension),
                positive_unsigned(exif::Tag::PixelYDimension),
            ),
            sensor_width_mm: sensor_width_for(camera),
        }
    }
}

/// The focal estimate of a `width` x `height` image from `fields`, and why a source was
/// overruled or ignored, when one was. See [`exif_image_metadata`] for the order and the checks.
fn estimate_focal(
    fields: &FocalFields,
    width: u32,
    height: u32,
) -> (Option<ExifFocalEstimate>, Option<String>) {
    let estimate = |focal_px: f64, source| {
        plausible_focal(focal_px, width, height).then_some(ExifFocalEstimate { focal_px, source })
    };
    let equivalent = fields.focal_35mm.and_then(|focal_35mm| {
        estimate(
            focal_from_35mm(focal_35mm, fields.focal_mm, width, height),
            ExifFocalSource::Equivalent35mm,
        )
    });
    // A candidate the 35 mm equivalent contradicts gives way to it.
    let checked = |candidate: ExifFocalEstimate, tolerance: f64, name: &str| match equivalent {
        Some(equivalent)
            if relative_difference(candidate.focal_px, equivalent.focal_px) > tolerance =>
        {
            let warning = format!(
                "the {name} gives a focal length of {:.0} px and the 35 mm equivalent {:.0} px; \
                 using the 35 mm equivalent",
                candidate.focal_px, equivalent.focal_px
            );
            (Some(equivalent), Some(warning))
        }
        _ => (Some(candidate), None),
    };

    let mut warning = None;
    if let Some(focal_mm) = fields.focal_mm {
        let plane = fields
            .plane_unit_mm
            .and_then(|unit_mm| {
                focal_from_plane_resolution(fields, focal_mm, unit_mm, width, height)
            })
            .and_then(|focal_px| estimate(focal_px, ExifFocalSource::FocalPlaneResolution));
        if let Some(plane) = plane {
            let sensor_mm = f64::from(width.max(height)) * focal_mm / plane.focal_px;
            if (MIN_SENSOR_LONG_SIDE_MM..=MAX_SENSOR_LONG_SIDE_MM).contains(&sensor_mm) {
                return checked(
                    plane,
                    MAX_FOCAL_PLANE_EQUIVALENT_DISAGREEMENT,
                    "focal-plane resolution",
                );
            }
            warning = Some(format!(
                "the focal-plane resolution implies a {sensor_mm:.1} mm sensor; it is ignored"
            ));
        }
        let known_width = fields.sensor_width_mm.and_then(|sensor_width_mm| {
            estimate(
                focal_from_sensor_width(focal_mm, sensor_width_mm, width.max(height)),
                ExifFocalSource::KnownSensorWidth,
            )
        });
        if let Some(known_width) = known_width {
            let (focal, overruled) = checked(
                known_width,
                MAX_SENSOR_WIDTH_EQUIVALENT_DISAGREEMENT,
                "known sensor width",
            );
            return (focal, overruled.or(warning));
        }
    }
    (equivalent, warning)
}

/// `|a - b|` relative to the larger of the two.
fn relative_difference(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.max(b)
}

/// Aspect ratio of a camera's native sensor format, which its 35 mm equivalent refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeFormat {
    ThreeByTwo,
    FourByThree,
}

impl NativeFormat {
    /// The format of a camera whose 35 mm equivalent is `focal_35mm` at `focal_mm`. Full frame
    /// and APS-C (crop factor up to 1.6) and 1-inch sensors (2.7) are 3:2; Four Thirds (2.0) and
    /// the small sensors of phones, compacts, drones, and action cameras (above 3.2) are 4:3.
    /// Without the physical focal length, a 4:3 image is taken for its sensor's format.
    fn of(focal_35mm: f64, focal_mm: Option<f64>, image_aspect: f64) -> Self {
        match focal_mm.map(|focal_mm| focal_35mm / focal_mm) {
            Some(crop) if crop < 1.8 || (2.3..3.2).contains(&crop) => Self::ThreeByTwo,
            Some(_) => Self::FourByThree,
            None if (image_aspect / (4.0 / 3.0) - 1.0).abs() < ASPECT_TOLERANCE => {
                Self::FourByThree
            }
            None => Self::ThreeByTwo,
        }
    }

    /// Long-side over short-side ratio.
    fn aspect(self) -> f64 {
        match self {
            Self::ThreeByTwo => 1.5,
            Self::FourByThree => 4.0 / 3.0,
        }
    }

    /// Long side, in millimetres, of the full-frame frame with the same diagonal and this aspect.
    fn equivalent_long_side_mm(self) -> f64 {
        match self {
            Self::ThreeByTwo => FULL_FRAME_WIDTH_MM,
            Self::FourByThree => FULL_FRAME_DIAGONAL_MM * 0.8,
        }
    }
}

/// Read a focal estimate while preserving I/O and EXIF parse failures.
#[cfg(test)]
fn exif_focal_estimate(
    path: &Path,
    width: u32,
    height: u32,
) -> Result<Option<ExifFocalEstimate>, ExifFocalError> {
    Ok(exif_image_metadata(path, width, height)?.and_then(|metadata| metadata.focal))
}

/// Focal length in pixels from a 35 mm equivalent, which makers define by the diagonal of the
/// camera's native sensor. An image as wide as that sensor, or wider (16:9), spans its long side;
/// a narrower one (1:1, or 4:3 from a 3:2 sensor) spans its short side.
fn focal_from_35mm(focal_35mm: f64, focal_mm: Option<f64>, width: u32, height: u32) -> f64 {
    let long_side = f64::from(width.max(height));
    let image_aspect = long_side / f64::from(width.min(height));
    let format = NativeFormat::of(focal_35mm, focal_mm, image_aspect);
    let format_long_side_mm = format.equivalent_long_side_mm();
    let image_long_side_mm = if image_aspect >= format.aspect() * (1.0 - ASPECT_TOLERANCE) {
        format_long_side_mm
    } else {
        format_long_side_mm * image_aspect / format.aspect()
    };
    focal_35mm / image_long_side_mm * long_side
}

fn focal_from_sensor_width(focal_mm: f64, sensor_width_mm: f64, long_image_side: u32) -> f64 {
    focal_mm / sensor_width_mm * f64::from(long_image_side)
}

/// Whether `focal_px` is a 3 mm to 3000 mm full-frame equivalent for a `width` x `height` image.
pub(super) fn plausible_focal(focal_px: f64, width: u32, height: u32) -> bool {
    if !focal_px.is_finite() || focal_px <= 0.0 || width == 0 || height == 0 {
        return false;
    }
    let equivalent_mm = focal_px * FULL_FRAME_WIDTH_MM / f64::from(width.max(height));
    (MIN_PLAUSIBLE_EQUIVALENT_MM..=MAX_PLAUSIBLE_EQUIVALENT_MM).contains(&equivalent_mm)
}

fn focal_from_plane_resolution(
    fields: &FocalFields,
    focal_mm: f64,
    unit_mm: f64,
    width: u32,
    height: u32,
) -> Option<f64> {
    let (pixel_x, pixel_y) = fields.pixel_dimensions;
    // Pixels rotated after capture, with the capture size left in EXIF: the sensor's X and Y run
    // along the stored height and width.
    let rotated = matches!(
        (pixel_x, pixel_y),
        (Some(x), Some(y)) if x != y && (width, height) == (y, x)
    );
    let (along_x, along_y) = if rotated {
        (height, width)
    } else {
        (width, height)
    };
    let estimate_axis = |resolution: Option<f64>, recorded_dimension: Option<u32>, current: u32| {
        let resolution = resolution?;
        let scale = recorded_dimension
            .map(|dimension| f64::from(current) / f64::from(dimension))
            .unwrap_or(1.0);
        let focal_px = focal_mm * resolution / unit_mm * scale;
        (focal_px.is_finite() && focal_px > 0.0).then_some(focal_px)
    };

    let focal_x = estimate_axis(fields.plane_resolution.0, pixel_x, along_x);
    let focal_y = estimate_axis(fields.plane_resolution.1, pixel_y, along_y);
    match (focal_x, focal_y) {
        (Some(x), Some(y)) => select_consistent_focal_plane_axes(x, y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn select_consistent_focal_plane_axes(focal_x: f64, focal_y: f64) -> Option<f64> {
    let disagreement = (focal_x - focal_y).abs() / focal_x.max(focal_y);
    (disagreement <= MAX_FOCAL_PLANE_AXIS_DISAGREEMENT).then_some((focal_x + focal_y) * 0.5)
}

fn sensor_width_for(camera: &ExifCameraIdentity) -> Option<f64> {
    let make = camera.make.as_deref()?.trim();
    let model = camera.model.as_deref()?.trim();
    SENSOR_WIDTH_PROFILES
        .iter()
        .find(|profile| {
            make.eq_ignore_ascii_case(profile.make) && model.eq_ignore_ascii_case(profile.model)
        })
        .map(|profile| profile.width_mm)
}

fn rational_value(value: &exif::Value) -> Option<f64> {
    let value = match value {
        exif::Value::Rational(values) => values.first().map(|value| value.to_f64()),
        exif::Value::SRational(values) => values.first().map(|value| value.to_f64()),
        _ => None,
    }?;
    value.is_finite().then_some(value)
}

fn unsigned_value(value: &exif::Value) -> Option<u32> {
    match value {
        exif::Value::Byte(values) => values.first().copied().map(u32::from),
        exif::Value::Short(values) => values.first().copied().map(u32::from),
        exif::Value::Long(values) => values.first().copied(),
        _ => None,
    }
}

fn resolution_unit_mm(value: &exif::Value) -> Option<f64> {
    match unsigned_value(value)? {
        2 => Some(25.4),
        3 => Some(10.0),
        4 => Some(1.0),
        5 => Some(0.001),
        _ => None,
    }
}

fn ascii_value(value: &exif::Value) -> Option<String> {
    match value {
        exif::Value::Ascii(values) => values.first().map(|bytes| {
            String::from_utf8_lossy(bytes)
                .trim_matches(|character: char| character == '\0' || character.is_whitespace())
                .to_owned()
        }),
        _ => None,
    }
}

fn exif_ascii(exif: &exif::Exif, tag: exif::Tag) -> Option<String> {
    exif.get_field(tag, exif::In::PRIMARY)
        .and_then(|field| ascii_value(&field.value))
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use tempfile::TempDir;

    #[test]
    fn missing_file_is_reported_by_checked_api() {
        let path = Path::new("/nonexistent/path.jpg");
        assert!(matches!(
            exif_focal_estimate(path, 100, 100),
            Err(ExifFocalError::Open { .. })
        ));
    }

    #[test]
    fn image_without_exif_is_not_an_error() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("plain.jpg");
        GrayImage::from_pixel(32, 24, Luma([127]))
            .save(&path)
            .unwrap();

        assert_eq!(exif_focal_estimate(&path, 32, 24).unwrap(), None);
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        assert!(matches!(
            exif_focal_estimate(Path::new("unused.jpg"), 0, 10),
            Err(ExifFocalError::InvalidDimensions { .. })
        ));
    }

    fn close(actual: f64, expected: f64, relative: f64) -> bool {
        relative_difference(actual, expected) <= relative
    }

    #[test]
    fn equivalent_35mm_follows_the_diagonal_of_the_native_format() {
        // A full 3:2 frame spans the 36 mm side, exactly as before.
        assert_eq!(
            focal_from_35mm(50.0, Some(50.0), 6000, 4000),
            50.0 / 36.0 * 6000.0
        );
        // A full 4:3 frame (crop factor 6, a small sensor) spans 0.8 x 43.27 mm.
        let four_by_three = focal_from_35mm(35.0, Some(5.85), 2304, 1728);
        assert!(close(four_by_three, 35.0 / 34.613_292 * 2304.0, 1e-6));
        // A 16:9 image of a 4:3 sensor keeps its full width.
        assert!(close(
            focal_from_35mm(24.0, Some(4.49), 4000, 2250),
            24.0 / 34.613_292 * 4000.0,
            1e-6
        ));
        // Cropping changes no focal length in pixels: a 1:1 crop of the 4:3 frame, and a 4:3
        // crop of a 3:2 frame, give the focal of the full frame.
        assert!(close(
            focal_from_35mm(35.0, Some(5.85), 1728, 1728),
            four_by_three,
            1e-9
        ));
        assert!(close(
            focal_from_35mm(50.0, Some(50.0), 5333, 4000),
            50.0 / 36.0 * 6000.0,
            1e-3
        ));
        // Without the physical focal length, a 4:3 image is its sensor's format.
        assert!(close(
            focal_from_35mm(28.0, None, 4000, 3000),
            28.0 / 34.613_292 * 4000.0,
            1e-6
        ));
    }

    #[test]
    fn equivalent_35mm_of_a_real_compact_matches_its_sensor_width() {
        // KODAK Z612: 5.85 mm on a 5.744 mm wide 1/2.5" sensor, written as a 35 mm equivalent.
        let sensor_width = focal_from_sensor_width(5.85, 5.744, 2304);
        let equivalent = focal_from_35mm(35.0, Some(5.85), 2304, 1728);
        assert!(close(equivalent, sensor_width, 0.01));
        // The width-based reading of the same tag is 4.5% short.
        assert!(!close(35.0 / 36.0 * 2304.0, sensor_width, 0.04));
    }

    /// A Canon EOS 70D image: 22.5 mm wide sensor, 5472 x 3648 pixels, density in pixels/inch.
    fn canon_like(focal_mm: f64) -> FocalFields {
        let pixels_per_inch = 5472.0 / 22.5 * 25.4;
        FocalFields {
            focal_mm: Some(focal_mm),
            plane_resolution: (Some(pixels_per_inch), Some(pixels_per_inch)),
            plane_unit_mm: Some(25.4),
            pixel_dimensions: (Some(5472), Some(3648)),
            ..FocalFields::default()
        }
    }

    #[test]
    fn a_consistent_focal_plane_resolution_is_used() {
        let (focal, warning) = estimate_focal(&canon_like(24.0), 5472, 3648);
        let focal = focal.unwrap();
        assert_eq!(focal.source, ExifFocalSource::FocalPlaneResolution);
        assert!(close(focal.focal_px, 24.0 / 22.5 * 5472.0, 1e-9));
        assert_eq!(warning, None);
    }

    #[test]
    fn rotated_pixels_with_the_capture_size_keep_the_focal_plane_resolution() {
        // Stored as a portrait while EXIF still says 5472 x 3648: each axis would otherwise be
        // rescaled by 2/3 and 3/2 and the two would disagree.
        let (focal, _) = estimate_focal(&canon_like(24.0), 3648, 5472);
        let focal = focal.unwrap();
        assert_eq!(focal.source, ExifFocalSource::FocalPlaneResolution);
        assert!(close(focal.focal_px, 24.0 / 22.5 * 5472.0, 1e-9));
    }

    #[test]
    fn a_focal_plane_resolution_far_from_the_35mm_equivalent_gives_way() {
        // The density says 22.5 mm wide; the 35 mm equivalent says APS-C at 1.6, 36/1.6 = 22.5 mm
        // too, so it agrees. Halving the density (a resize the tags did not follow) disagrees.
        let mut fields = canon_like(24.0);
        fields.focal_35mm = Some(38.0);
        let (focal, warning) = estimate_focal(&fields, 5472, 3648);
        assert_eq!(focal.unwrap().source, ExifFocalSource::FocalPlaneResolution);
        assert_eq!(warning, None);

        fields.plane_resolution = (
            fields.plane_resolution.0.map(|value| value * 2.0),
            fields.plane_resolution.1.map(|value| value * 2.0),
        );
        let (focal, warning) = estimate_focal(&fields, 5472, 3648);
        let focal = focal.unwrap();
        assert_eq!(focal.source, ExifFocalSource::Equivalent35mm);
        assert!(close(focal.focal_px, 38.0 / 36.0 * 5472.0, 1e-9));
        assert!(warning.unwrap().contains("focal-plane resolution"));
    }

    #[test]
    fn a_focal_plane_resolution_implying_no_real_sensor_is_ignored() {
        // A density 20x too high implies a 1.1 mm sensor, though its 768 mm equivalent alone
        // would pass; the known sensor width takes over.
        let mut fields = canon_like(24.0);
        fields.plane_resolution = (
            fields.plane_resolution.0.map(|value| value * 20.0),
            fields.plane_resolution.1.map(|value| value * 20.0),
        );
        fields.sensor_width_mm = Some(22.5);
        let (focal, warning) = estimate_focal(&fields, 5472, 3648);
        assert_eq!(focal.unwrap().source, ExifFocalSource::KnownSensorWidth);
        assert!(warning.unwrap().contains("mm sensor"));
    }

    #[test]
    fn a_sensor_width_for_another_crop_mode_gives_way_to_the_35mm_equivalent() {
        // A full-frame width on an image shot in a 1.5x crop mode, which its 35 mm equivalent
        // records.
        let fields = FocalFields {
            focal_mm: Some(50.0),
            focal_35mm: Some(75.0),
            sensor_width_mm: Some(36.0),
            ..FocalFields::default()
        };
        let (focal, warning) = estimate_focal(&fields, 6000, 4000);
        let focal = focal.unwrap();
        assert_eq!(focal.source, ExifFocalSource::Equivalent35mm);
        assert!(close(focal.focal_px, 75.0 / 36.0 * 6000.0, 1e-9));
        assert!(warning.unwrap().contains("known sensor width"));
    }

    #[test]
    fn a_sensor_width_near_a_marketing_35mm_equivalent_is_kept() {
        // DJI FC7303 writes 24 mm for a 26 mm field of view; its sensor width is the better source.
        let fields = FocalFields {
            focal_mm: Some(4.49),
            focal_35mm: Some(24.0),
            sensor_width_mm: Some(6.16),
            ..FocalFields::default()
        };
        let (focal, warning) = estimate_focal(&fields, 4000, 2250);
        let focal = focal.unwrap();
        assert_eq!(focal.source, ExifFocalSource::KnownSensorWidth);
        assert!(close(focal.focal_px, 4.49 / 6.16 * 4000.0, 1e-9));
        assert_eq!(warning, None);
    }

    #[test]
    fn physical_focal_and_sensor_width_handle_cropped_aspect_ratios() {
        let focal_px = focal_from_sensor_width(4.49, 6.16, 4000);
        assert!((focal_px - 2_915.584_415_584_415_6).abs() < 1e-9);

        let camera = ExifCameraIdentity {
            make: Some("dji".to_owned()),
            model: Some("fc7303".to_owned()),
            body_serial_number: None,
            lens_model: None,
            lens_serial_number: None,
        };
        assert_eq!(sensor_width_for(&camera), Some(6.16));
    }

    #[test]
    fn observed_camera_profiles_are_exact() {
        let camera = |make: &str, model: &str| ExifCameraIdentity {
            make: Some(make.to_owned()),
            model: Some(model.to_owned()),
            ..ExifCameraIdentity::default()
        };
        assert_eq!(
            sensor_width_for(&camera("Canon", "Canon EOS 70D")),
            Some(22.5)
        );
        assert_eq!(
            sensor_width_for(&camera(
                "EASTMAN KODAK COMPANY",
                "KODAK Z612 ZOOM DIGITAL CAMERA"
            )),
            Some(5.744)
        );
    }

    #[test]
    fn sensor_profiles_require_exact_make_and_model() {
        let wrong_make = ExifCameraIdentity {
            make: Some("Not DJI".to_owned()),
            model: Some("FC7303".to_owned()),
            body_serial_number: None,
            lens_model: None,
            lens_serial_number: None,
        };
        let partial_model = ExifCameraIdentity {
            make: Some("DJI".to_owned()),
            model: Some("FC7303 copy".to_owned()),
            body_serial_number: None,
            lens_model: None,
            lens_serial_number: None,
        };
        assert_eq!(sensor_width_for(&wrong_make), None);
        assert_eq!(sensor_width_for(&partial_model), None);
    }

    #[test]
    fn focal_plane_axes_must_agree() {
        assert_eq!(
            select_consistent_focal_plane_axes(1000.0, 1005.0),
            Some(1002.5)
        );
        assert_eq!(select_consistent_focal_plane_axes(1000.0, 1050.0), None);
    }

    #[test]
    fn focal_plane_units_cover_standard_and_extended_encodings() {
        assert_eq!(resolution_unit_mm(&exif::Value::Short(vec![2])), Some(25.4));
        assert_eq!(resolution_unit_mm(&exif::Value::Short(vec![3])), Some(10.0));
        assert_eq!(resolution_unit_mm(&exif::Value::Short(vec![4])), Some(1.0));
        assert_eq!(
            resolution_unit_mm(&exif::Value::Short(vec![5])),
            Some(0.001)
        );
        assert_eq!(resolution_unit_mm(&exif::Value::Short(vec![1])), None);
    }

    #[test]
    fn legitimate_wide_and_telephoto_lenses_are_plausible() {
        assert!(plausible_focal(
            focal_from_35mm(8.0, None, 6000, 4000),
            6000,
            4000
        ));
        assert!(plausible_focal(
            focal_from_35mm(135.0, None, 6000, 4000),
            6000,
            4000
        ));
        assert!(!plausible_focal(
            focal_from_35mm(10_000.0, None, 6000, 4000),
            6000,
            4000
        ));
    }

    #[test]
    fn invalid_rationals_are_rejected() {
        let invalid = exif::Value::Rational(vec![exif::Rational { num: 1, denom: 0 }]);
        assert_eq!(rational_value(&invalid), None);
    }
}
