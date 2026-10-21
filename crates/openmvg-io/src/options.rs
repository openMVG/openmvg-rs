// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Options of the typed reader and writer.

use std::path::{Path, PathBuf};

use openmvg_sfm::{Dataset, RunSeed, SfmEngine};

/// The unit of world lengths: pose centers and point coordinates. It labels a dataset's
/// numbers; it is unrelated to `openmvg_sfm::config::LengthUnit`, the unit of an engine's
/// baseline gauge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum LengthUnit {
    /// An arbitrary scale, as a reconstruction from images alone has.
    #[default]
    Unknown,
    /// Meters, for a reconstruction anchored to a metric reference.
    Meter,
}

impl LengthUnit {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            LengthUnit::Unknown => "unknown",
            LengthUnit::Meter => "meter",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "unknown" => Some(LengthUnit::Unknown),
            "meter" => Some(LengthUnit::Meter),
            _ => None,
        }
    }
}

/// How [`write_view_graph`](crate::write_view_graph),
/// [`write_reconstruction`](crate::write_reconstruction), and
/// [`write_correspondences`](crate::write_correspondences) write a dataset.
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct WriteOptions {
    pub(crate) length_unit: Option<LengthUnit>,
    pub(crate) image_root: Option<PathBuf>,
}

impl WriteOptions {
    /// Lengths labeled with the unit the written value carries, and image paths given by the
    /// image names.
    pub fn new() -> Self {
        Self::default()
    }

    /// Label the written pose centers and point coordinates with `length_unit`, in place of the
    /// unit the written value carries. The coordinates are written as they are, never rescaled:
    /// [`LengthUnit::Meter`] asserts that they already are meters.
    ///
    /// Without it, a `ViewGraph` or `Reconstruction`, which holds no unit, is labeled
    /// [`LengthUnit::Unknown`], and [`Correspondences`](crate::Correspondences) keep their own.
    pub const fn with_length_unit(mut self, length_unit: LengthUnit) -> Self {
        self.length_unit = Some(length_unit);
        self
    }

    /// The label of the written lengths: the relabel, if any, else `carried`.
    pub(crate) fn length_unit_or(&self, carried: LengthUnit) -> LengthUnit {
        self.length_unit.unwrap_or(carried)
    }

    /// Store each image file as its path relative to `image_root`, the folder that holds the
    /// images, subfolders included. The write fails for an image file outside `image_root`; the
    /// check compares paths as written, so spell the root the way the image paths start (for
    /// example `castle/images`, not `./castle/images` or its absolute path).
    ///
    /// Without an image root, each view stores its image name: the file name of an image file,
    /// or the identifier of another image source. Read the dataset back with the same root,
    /// through [`ReadOptions::with_image_root`], so that each view points to its file again.
    pub fn with_image_root(mut self, image_root: impl AsRef<Path>) -> Self {
        self.image_root = Some(image_root.as_ref().to_owned());
        self
    }
}

/// How [`read_view_graph`](crate::read_view_graph) builds a `ViewGraph` and
/// [`read_reconstruction`](crate::read_reconstruction) a `Reconstruction`.
#[derive(Debug, Clone)]
#[must_use]
pub struct ReadOptions {
    pub(crate) root_seed: RunSeed,
    pub(crate) engine: SfmEngine,
    pub(crate) image_root: Option<PathBuf>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            root_seed: Dataset::DEFAULT_SEED.into(),
            engine: SfmEngine::default(),
            image_root: None,
        }
    }
}

impl ReadOptions {
    /// The default root seed and engine, and images resolved against the dataset directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// The root seed of the graph or reconstruction read, which the format does not store;
    /// defaults to [`Dataset::DEFAULT_SEED`], the seed of a dataset built without one. To replay
    /// a stored view graph, give the seed of the graph that was written, its
    /// [`root_seed`](openmvg_sfm::ViewGraph::root_seed).
    pub fn with_root_seed(mut self, root_seed: impl Into<RunSeed>) -> Self {
        self.root_seed = root_seed.into();
        self
    }

    /// The engine a reconstruction reports as its producer, which the format does not store;
    /// defaults to [`SfmEngine::Incremental`].
    pub const fn with_engine(mut self, engine: SfmEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Resolve each view's image path against `image_root` instead of the dataset directory:
    /// pass the folder given to [`WriteOptions::with_image_root`]. An image file that exists
    /// must lie inside `image_root` once symbolic links are followed; a missing one is accepted.
    pub fn with_image_root(mut self, image_root: impl AsRef<Path>) -> Self {
        self.image_root = Some(image_root.as_ref().to_owned());
        self
    }
}
