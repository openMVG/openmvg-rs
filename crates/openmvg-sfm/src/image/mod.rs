// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) mod decode;
pub(crate) mod exif;
pub(crate) mod prepare;
pub(crate) mod project;

pub use decode::{
    ImageGeometry, discover_dataset_images, load_gray_image_downsampled,
    load_gray_image_max_dimension,
};
pub use prepare::{
    ImageDatasetConfig, ImageIntrinsicPolicy, ImageIntrinsicReport, ImageIntrinsicSource,
    PreparedImageDataset, find_calibration_file, image_size, load_pinhole_intrinsics_matrix,
    prepare_image_paths,
};
pub use project::{ImageProject, ImageProjectGroup, prepare_image_project};
