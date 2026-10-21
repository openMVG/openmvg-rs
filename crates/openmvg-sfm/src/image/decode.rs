// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Image discovery in the supported dataset layouts, grayscale decoding, Lanczos3 resizing, and
//! the working-to-source pixel geometry of a resized image.

use std::path::{Path, PathBuf};

use openmvg_features::extraction::GrayImage;

use crate::{Result, SfmError};

/// Map one integer-centered pixel coordinate through an image resize.
///
/// The half-pixel translations preserve pixel-cell centers instead of treating
/// coordinate zero as an invariant image corner.
#[inline]
fn rescale_pixel_center(coordinate: f32, scale: f32) -> f32 {
    (coordinate + 0.5) * scale - 0.5
}

/// Geometry of an image after an optional resize.
///
/// Keeping both sizes avoids assuming that an integer downsample factor maps
/// odd source dimensions to an exactly isotropic working image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGeometry {
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
}

impl ImageGeometry {
    /// Construct one checked mapping between source and working image grids.
    pub fn try_new(
        source_width: usize,
        source_height: usize,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        if source_width == 0 || source_height == 0 || width == 0 || height == 0 {
            return Err(SfmError::InvalidInput(format!(
                "image geometry dimensions must be non-zero, got source={source_width}x{source_height}, working={width}x{height}"
            )));
        }
        Ok(Self {
            source_width,
            source_height,
            width,
            height,
        })
    }

    /// Width of the decoded source image, in pixels.
    pub const fn source_width(self) -> usize {
        self.source_width
    }

    /// Height of the decoded source image, in pixels.
    pub const fn source_height(self) -> usize {
        self.source_height
    }

    /// Width of the working (possibly downsampled) image, in pixels.
    pub const fn width(self) -> usize {
        self.width
    }

    /// Height of the working (possibly downsampled) image, in pixels.
    pub const fn height(self) -> usize {
        self.height
    }

    /// Per-axis scale from source-image coordinates to working coordinates.
    pub fn source_to_working_scale(self) -> (f32, f32) {
        (
            self.width as f32 / self.source_width as f32,
            self.height as f32 / self.source_height as f32,
        )
    }

    /// Per-axis scale from working coordinates back to source coordinates.
    pub fn working_to_source_scale(self) -> (f32, f32) {
        let (x, y) = self.source_to_working_scale();
        (x.recip(), y.recip())
    }

    /// Maps a source-image pixel position to working-image pixels, preserving pixel centers.
    pub fn source_to_working(self, x: f32, y: f32) -> (f32, f32) {
        let (scale_x, scale_y) = self.source_to_working_scale();
        (
            rescale_pixel_center(x, scale_x),
            rescale_pixel_center(y, scale_y),
        )
    }

    /// Maps a working-image pixel position back to source-image pixels, preserving pixel centers.
    pub fn working_to_source(self, x: f32, y: f32) -> (f32, f32) {
        let (scale_x, scale_y) = self.working_to_source_scale();
        (
            rescale_pixel_center(x, scale_x),
            rescale_pixel_center(y, scale_y),
        )
    }

    /// Area-preserving isotropic scale for keypoint radii and sigma values.
    pub fn working_length_to_source(self, length: f32) -> f32 {
        let (scale_x, scale_y) = self.working_to_source_scale();
        length * (scale_x * scale_y).sqrt()
    }
}

/// Discover the images of a dataset folder: the folder itself, its `images/` subfolder, or the
/// `dslr_images_undistorted` layout of the ETH3D benchmark.
///
/// Candidate directories are checked from most specific to least specific:
/// `<dir>/images/dslr_images_undistorted`, `<dir>/images`,
/// `<dir>/dslr_images_undistorted`, then `<dir>` itself.
///
/// Symbolic links to image files are followed and keep their link name; a
/// dangling link with an image extension is an [`SfmError::ReadFile`] error.
pub fn discover_dataset_images(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(SfmError::DatasetNotFound {
            path: dir.to_path_buf(),
        });
    }

    let candidates = [
        dir.join("images").join("dslr_images_undistorted"),
        dir.join("images"),
        dir.join("dslr_images_undistorted"),
        dir.to_path_buf(),
    ];

    for candidate in candidates {
        if !candidate.is_dir() {
            continue;
        }
        let images = list_images_sorted(&candidate)?;
        if !images.is_empty() {
            return Ok(images);
        }
    }

    Err(SfmError::NoImagesFound {
        path: dir.to_path_buf(),
    })
}

fn list_images_sorted(search_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    for entry in std::fs::read_dir(search_dir)? {
        let path = entry?.path();
        if !is_supported_image(&path) {
            continue;
        }
        // `metadata` follows symbolic links: a linked image keeps its link name,
        // and a dangling link is reported instead of silently dropping a view.
        let metadata = std::fs::metadata(&path).map_err(|source| SfmError::ReadFile {
            path: path.clone(),
            source,
        })?;
        if metadata.is_file() {
            images.push(path);
        }
    }
    images.sort_unstable();
    Ok(images)
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "tif" | "tiff"
            )
        })
}

/// Load a grayscale image, optionally limiting its longest edge.
///
/// The image is never enlarged. It comes back in the detectors' [`GrayImage`] layout, with the
/// [`ImageGeometry`] that maps its keypoints back to source-image coordinates.
pub fn load_gray_image_max_dimension(
    path: &Path,
    max_dimension: Option<u32>,
) -> Result<(GrayImage, ImageGeometry)> {
    if max_dimension == Some(0) {
        return Err(SfmError::InvalidConfig(
            "maximum image dimension must be greater than zero".to_owned(),
        ));
    }
    let gray = decode_gray_image(path)?;
    let source_width = gray.width();
    let source_height = gray.height();
    let (width, height) = max_dimension.map_or((source_width, source_height), |maximum| {
        fit_within(source_width, source_height, maximum)
    });
    prepare_gray_image(gray, width, height)
}

/// Load and resize a grayscale image by an integer reduction factor.
///
/// Resizing is performed once with an anti-aliased Lanczos3 filter. The
/// returned geometry contains the exact realized scale for odd dimensions.
pub fn load_gray_image_downsampled(
    path: &Path,
    downsample_factor: u32,
) -> Result<(GrayImage, ImageGeometry)> {
    if downsample_factor == 0 {
        return Err(SfmError::InvalidConfig(
            "image downsample factor must be at least one".to_owned(),
        ));
    }

    let gray = decode_gray_image(path)?;
    let width = (gray.width() / downsample_factor).max(1);
    let height = (gray.height() / downsample_factor).max(1);
    prepare_gray_image(gray, width, height)
}

fn decode_gray_image(path: &Path) -> Result<image::GrayImage> {
    image::open(path)
        .map_err(|error| match error {
            image::ImageError::IoError(source) => SfmError::ImageRead {
                path: path.to_path_buf(),
                source,
            },
            error => SfmError::Image(format!("{}: {error}", path.display())),
        })
        .map(image::DynamicImage::into_luma8)
}

fn fit_within(width: u32, height: u32, maximum: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= maximum {
        return (width, height);
    }
    let scale = f64::from(maximum) / f64::from(longest);
    (
        (f64::from(width) * scale).round().max(1.0) as u32,
        (f64::from(height) * scale).round().max(1.0) as u32,
    )
}

fn prepare_gray_image(
    gray: image::GrayImage,
    width: u32,
    height: u32,
) -> Result<(GrayImage, ImageGeometry)> {
    let source_width = gray.width() as usize;
    let source_height = gray.height() as usize;
    let (width, height) = (width as usize, height as usize);
    let geometry = ImageGeometry::try_new(source_width, source_height, width, height)?;
    let pixels = if (width, height) == (source_width, source_height) {
        gray.into_raw()
    } else {
        resize_lanczos3(gray.as_raw(), source_width, source_height, width, height)
    };
    // One pass from the decoder's row-major bytes to the detectors' column-major layout.
    let image = GrayImage::from_fn(height, width, |row, column| {
        f32::from(pixels[row * width + column])
    });
    Ok((image, geometry))
}

/// Lanczos3 window: `sinc(x) sinc(x / 3)` for `|x| < 3`.
fn lanczos3(x: f32) -> f32 {
    fn sinc(t: f32) -> f32 {
        if t == 0.0 {
            1.0
        } else {
            let a = t * core::f32::consts::PI;
            a.sin() / a
        }
    }
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Lanczos3 resampling of one axis from `input` to `output` samples: per
/// output sample, the first input index and the normalized weights of the
/// consecutive inputs it reads.
///
/// The support (three output-sample widths when shrinking), the kernel and
/// the normalization are those of `image::imageops::resize`.
fn lanczos3_axis(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let ratio = input as f32 / output as f32;
    let scale = ratio.max(1.0);
    let support = 3.0 * scale;
    (0..output)
        .map(|sample| {
            let center = (sample as f32 + 0.5) * ratio;
            let last = input as i64;
            let left = ((center - support).floor() as i64).clamp(0, last - 1);
            let right = ((center + support).ceil() as i64).clamp(left + 1, last);
            let center = center - 0.5;
            let mut weights: Vec<f32> = (left..right)
                .map(|index| lanczos3((index as f32 - center) / scale))
                .collect();
            let mut sum = 0.0;
            for &weight in &weights {
                sum += weight;
            }
            for weight in &mut weights {
                *weight /= sum;
            }
            (left as usize, weights)
        })
        .collect()
}

/// Separable Lanczos3 resize of a row-major 8-bit gray image.
///
/// Bit-identical to `image::imageops::resize` with `FilterType::Lanczos3`
/// (same weights, same vertical-then-horizontal order, same accumulation
/// order and rounding), on one channel instead of four and with contiguous
/// row reads, which makes it several times faster.
fn resize_lanczos3(
    pixels: &[u8],
    width: usize,
    height: usize,
    new_width: usize,
    new_height: usize,
) -> Vec<u8> {
    // Vertical pass: every output row is a weighted sum of whole input rows.
    let mut rows = vec![0.0f32; width * new_height];
    for ((first, weights), row) in lanczos3_axis(height, new_height)
        .iter()
        .zip(rows.chunks_exact_mut(width))
    {
        for (offset, &weight) in weights.iter().enumerate() {
            let source = &pixels[(first + offset) * width..][..width];
            for (sum, &pixel) in row.iter_mut().zip(source) {
                *sum += f32::from(pixel) * weight;
            }
        }
    }
    // Horizontal pass, rounded to the nearest 8-bit value.
    let columns = lanczos3_axis(width, new_width);
    let mut resized = Vec::with_capacity(new_width * new_height);
    for row in rows.chunks_exact(width) {
        resized.extend(columns.iter().map(|(first, weights)| {
            let mut sum = 0.0f32;
            for (&weight, &value) in weights.iter().zip(&row[*first..]) {
                sum += value * weight;
            }
            sum.clamp(0.0, 255.0).round() as u8
        }));
    }
    resized
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage as ImageBuffer, Luma};
    use tempfile::TempDir;

    fn write_image(path: &Path, width: u32, height: u32) {
        let image = ImageBuffer::from_fn(width, height, |x, y| Luma([(y * width + x) as u8]));
        image.save(path).unwrap();
    }

    #[test]
    fn discovery_supports_eth3d_layout_and_sorts_paths() {
        let root = TempDir::new().unwrap();
        let images = root.path().join("images").join("dslr_images_undistorted");
        std::fs::create_dir_all(&images).unwrap();
        write_image(&images.join("b.PNG"), 2, 2);
        write_image(&images.join("a.jpg"), 2, 2);
        std::fs::write(images.join("ignored.txt"), b"not an image").unwrap();

        let discovered = discover_dataset_images(root.path()).unwrap();
        let names: Vec<_> = discovered
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap())
            .collect();

        assert_eq!(names, ["a.jpg", "b.PNG"]);
    }

    #[test]
    fn discovery_falls_back_to_flat_layout() {
        let root = TempDir::new().unwrap();
        write_image(&root.path().join("frame.tiff"), 2, 2);

        assert_eq!(discover_dataset_images(root.path()).unwrap().len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn discovery_follows_image_symlinks_under_their_link_name() {
        let source = TempDir::new().unwrap();
        let root = TempDir::new().unwrap();
        write_image(&source.path().join("original.png"), 2, 2);
        write_image(&root.path().join("c.jpg"), 2, 2);
        std::os::unix::fs::symlink(
            source.path().join("original.png"),
            root.path().join("a.png"),
        )
        .unwrap();
        std::fs::create_dir(source.path().join("folder.jpg")).unwrap();
        std::os::unix::fs::symlink(source.path().join("folder.jpg"), root.path().join("b.jpg"))
            .unwrap();

        let discovered = discover_dataset_images(root.path()).unwrap();

        assert_eq!(
            discovered,
            [root.path().join("a.png"), root.path().join("c.jpg")]
        );
    }

    #[cfg(unix)]
    #[test]
    fn discovery_rejects_dangling_image_symlink() {
        let root = TempDir::new().unwrap();
        write_image(&root.path().join("a.png"), 2, 2);
        let link = root.path().join("missing.jpg");
        std::os::unix::fs::symlink(root.path().join("gone.jpg"), &link).unwrap();

        let error = discover_dataset_images(root.path()).unwrap_err();

        assert!(matches!(error, SfmError::ReadFile { ref path, .. } if *path == link));
    }

    #[test]
    fn full_resolution_pixels_keep_their_row_and_column() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("pixels.png");
        write_image(&path, 3, 2);

        let (image, _) = load_gray_image_downsampled(&path, 1).unwrap();

        assert_eq!((image.ncols(), image.nrows()), (3, 2));
        assert_eq!(image[(0, 1)], 1.0);
        assert_eq!(image[(1, 2)], 5.0);
    }

    #[test]
    fn resize_mapping_preserves_integer_centered_pixel_cells() {
        assert_eq!(rescale_pixel_center(7.25, 1.0), 7.25);
        assert_eq!(rescale_pixel_center(0.0, 4.0), 1.5);
        assert_eq!(rescale_pixel_center(3.0, 2.0), 6.5);
    }

    #[test]
    fn image_geometry_rejects_empty_domains() {
        assert!(ImageGeometry::try_new(0, 10, 5, 5).is_err());
        assert!(ImageGeometry::try_new(10, 10, 0, 5).is_err());
    }

    #[test]
    fn downsampling_reports_exact_odd_dimension_geometry() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("odd.png");
        write_image(&path, 7, 5);

        let (image, geometry) = load_gray_image_downsampled(&path, 2).unwrap();

        assert_eq!((image.ncols(), image.nrows()), (3, 2));
        assert_eq!(geometry.source_to_working_scale(), (3.0 / 7.0, 2.0 / 5.0));
        let source = geometry.working_to_source(1.5, 1.0);
        assert!((source.0 - (2.0 * 7.0 / 3.0 - 0.5)).abs() < 1e-6);
        assert!((source.1 - 3.25).abs() < 1e-6);
        let working = geometry.source_to_working(source.0, source.1);
        assert!((working.0 - 1.5).abs() < 1e-6);
        assert!((working.1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_downsample_factor_is_rejected() {
        let error = load_gray_image_downsampled(Path::new("unused.png"), 0).unwrap_err();
        assert!(matches!(error, SfmError::InvalidConfig(_)));
    }

    #[test]
    fn maximum_dimension_loader_never_enlarges_and_preserves_geometry() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("wide.png");
        write_image(&path, 12, 8);

        let (bounded, geometry) = load_gray_image_max_dimension(&path, Some(6)).unwrap();
        assert_eq!((bounded.ncols(), bounded.nrows()), (6, 4));
        assert_eq!(geometry.working_to_source(0.0, 0.0), (0.5, 0.5));

        let (native, geometry) = load_gray_image_max_dimension(&path, Some(20)).unwrap();
        assert_eq!((native.ncols(), native.nrows()), (12, 8));
        assert_eq!(geometry.working_to_source(3.0, 2.0), (3.0, 2.0));

        assert!(load_gray_image_max_dimension(&path, Some(0)).is_err());
    }

    #[test]
    fn lanczos3_resize_matches_the_image_crate_bit_for_bit() {
        let (width, height) = (311, 207);
        let (fit_width, fit_height) = fit_within(width, height, 160);
        let cases = [
            (width, height, fit_width, fit_height),
            (width, height, width.div_ceil(2), height.div_ceil(2)),
            (width, height, 37, 23),
            (width, height, width, 101),
            (width, height, 150, height),
            (13, 9, 40, 31),
            (1, 9, 1, 4),
        ];
        for (width, height, new_width, new_height) in cases {
            // Pseudo-random texture with saturated pixels, so the ringing
            // overshoots and the clamp is exercised.
            let source = ImageBuffer::from_fn(width, height, |x, y| {
                let hash = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 509;
                Luma([if hash < 64 { 255 } else { (hash % 251) as u8 }])
            });
            let expected = image::imageops::resize(
                &source,
                new_width,
                new_height,
                image::imageops::FilterType::Lanczos3,
            );

            let resized = resize_lanczos3(
                source.as_raw(),
                width as usize,
                height as usize,
                new_width as usize,
                new_height as usize,
            );

            assert_eq!(
                resized,
                expected.into_raw(),
                "{width}x{height} -> {new_width}x{new_height}"
            );
        }
    }
}
