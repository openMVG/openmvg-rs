// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pictures for the preview: the thumbnail a camera embedded, or a downscaled decode.
//!
//! Both are turned upright by the EXIF orientation, so the preview shows the scene the way the
//! photographer saw it. The sizes the interface lists are the stored pixel sizes, which `openmvg-sfm` reads.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use exif::{In, Tag};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};

/// Longest edge of a decoded preview, in pixels.
pub const PREVIEW_EDGE: u32 = 1024;

/// The upright thumbnail embedded in a JPEG's EXIF, if there is one, as the camera made it.
///
/// Cameras do not agree on its shape: a Canon thumbnail is a 4:3 canvas with black bars around a
/// 3:2 picture, which is drawn as it is. The full decode replaces it within a moment.
pub fn exif_thumbnail(path: &Path) -> Option<DynamicImage> {
    let mut reader = BufReader::new(File::open(path).ok()?);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
    let thumbnail = |tag| exif.get_field(tag, In::THUMBNAIL)?.value.get_uint(0);
    let offset = usize::try_from(thumbnail(Tag::JPEGInterchangeFormat)?).ok()?;
    let length = usize::try_from(thumbnail(Tag::JPEGInterchangeFormatLength)?).ok()?;
    let bytes = exif.buf().get(offset..offset.checked_add(length)?)?;
    let mut thumbnail = image::load_from_memory_with_format(bytes, ImageFormat::Jpeg).ok()?;

    let orientation = exif
        .get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .and_then(|value| u8::try_from(value).ok())
        .and_then(Orientation::from_exif)
        .unwrap_or(Orientation::NoTransforms);
    thumbnail.apply_orientation(orientation);
    Some(thumbnail)
}

/// Decodes an image, turns it upright, and shrinks it to at most [`PREVIEW_EDGE`] on a side.
///
/// # Errors
///
/// Returns a message when the file cannot be opened or decoded.
pub fn decode_preview(path: &Path) -> Result<DynamicImage, String> {
    let mut decoder = ImageReader::open(path)
        .and_then(ImageReader::with_guessed_format)
        .map_err(|error| error.to_string())?
        .into_decoder()
        .map_err(|error| error.to_string())?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = DynamicImage::from_decoder(decoder).map_err(|error| error.to_string())?;
    image.apply_orientation(orientation);
    Ok(if image.width().max(image.height()) > PREVIEW_EDGE {
        image.thumbnail(PREVIEW_EDGE, PREVIEW_EDGE)
    } else {
        image
    })
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use exif::experimental::Writer;
    use exif::{Field, Value};
    use std::io::Cursor;

    fn jpeg(image: &DynamicImage) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        image
            .to_rgb8()
            .write_to(&mut bytes, ImageFormat::Jpeg)
            .unwrap();
        bytes.into_inner()
    }

    /// A JPEG file of `size` pixels whose EXIF embeds a `thumbnail` of its own and the EXIF
    /// orientation `orientation` (1 is upright).
    pub fn jpeg_with_thumbnail(
        path: &Path,
        size: (u32, u32),
        thumbnail: (u32, u32),
        orientation: u16,
    ) {
        let main = jpeg(&DynamicImage::new_rgb8(size.0, size.1));
        let thumbnail = jpeg(&DynamicImage::new_rgb8(thumbnail.0, thumbnail.1));
        let field = Field {
            tag: Tag::Orientation,
            ifd_num: In::PRIMARY,
            value: Value::Short(vec![orientation]),
        };
        let mut writer = Writer::new();
        writer.push_field(&field);
        writer.set_jpeg(&thumbnail, In::THUMBNAIL);
        let mut tiff = Cursor::new(Vec::new());
        writer.write(&mut tiff, false).unwrap();
        let tiff = tiff.into_inner();

        // SOI, then an APP1 segment holding the Exif header and the TIFF data, then the picture.
        let segment_length = u16::try_from(2 + 6 + tiff.len()).unwrap();
        let mut file = vec![0xFF, 0xD8, 0xFF, 0xE1];
        file.extend(segment_length.to_be_bytes());
        file.extend(b"Exif\0\0");
        file.extend(&tiff);
        file.extend(&main[2..]);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn an_embedded_thumbnail_is_returned_as_the_camera_made_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("a.jpg");
        // A Canon-shaped file: a 4:3 thumbnail for a 3:2 photo. It must not be rejected.
        jpeg_with_thumbnail(&path, (600, 400), (40, 30), 1);

        let thumbnail = exif_thumbnail(&path).unwrap();

        assert_eq!((thumbnail.width(), thumbnail.height()), (40, 30));
    }

    #[test]
    fn the_orientation_turns_the_thumbnail_upright() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("portrait.jpg");
        // Stored landscape, tagged "rotate 90": upright it is a 300x400 portrait.
        jpeg_with_thumbnail(&path, (400, 300), (40, 30), 6);

        let thumbnail = exif_thumbnail(&path).unwrap();

        assert_eq!((thumbnail.width(), thumbnail.height()), (30, 40));
    }

    #[test]
    fn files_without_a_usable_thumbnail_give_none_not_an_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let png = dir.path().join("a.png");
        DynamicImage::new_rgb8(8, 8).save(&png).unwrap();
        let plain = dir.path().join("plain.jpg");
        std::fs::write(&plain, jpeg(&DynamicImage::new_rgb8(8, 8))).unwrap();

        assert!(exif_thumbnail(&png).is_none());
        assert!(exif_thumbnail(&plain).is_none());
        assert!(exif_thumbnail(&dir.path().join("absent.jpg")).is_none());
    }

    #[test]
    fn a_large_image_is_shrunk_to_the_preview_edge_keeping_its_aspect_and_orientation() {
        let dir = tempfile::TempDir::new().unwrap();
        let big = dir.path().join("big.png");
        DynamicImage::new_rgb8(2048, 1024).save(&big).unwrap();
        let small = dir.path().join("small.jpg");
        jpeg_with_thumbnail(&small, (400, 300), (40, 30), 6);

        let big = decode_preview(&big).unwrap();
        let small = decode_preview(&small).unwrap();

        assert_eq!((big.width(), big.height()), (1024, 512));
        assert_eq!(
            (small.width(), small.height()),
            (300, 400),
            "not enlarged, and turned upright"
        );
        assert!(decode_preview(&dir.path().join("absent.png")).is_err());
    }
}
