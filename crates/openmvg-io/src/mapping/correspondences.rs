// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Putative correspondences, with optional ground truth, as a `dataset`.
//!
//! ```text
//! Correspondences ──> cameras, views, keypoints, matches (null inlier), gt_poses? ──> publish
//!                                                         profiles: dataset [+ benchmark]
//! dataset ──> cameras, views, keypoints, every match, gt_poses? ──> Correspondences
//! ```
//!
//! A `ViewGraph` holds only verified pairs, so a corpus of raw matches, which an engine is to
//! verify itself, goes through this type instead.

use std::path::Path;

use openmvg_sfm::model::ViewRecord;
use openmvg_sfm::{Camera, FeatureMatch, Keypoint, ViewId, ViewPair, WorldToCameraPose};

use super::columns::{Rows, TableBuilder, id};
use super::export::{
    cameras, insert, keypoints_table, matches_table, pose_columns, publish, views_table,
};
use super::import::{ViewSet, keypoints, load_profile, match_groups, pose, table, view};
use crate::error::{Error, Result};
use crate::format::checks::Tables;
use crate::format::table::Table;
use crate::options::{LengthUnit, ReadOptions, WriteOptions};

/// The views, keypoints, and putative matches of a dataset, with the ground-truth poses of a
/// `benchmark`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Correspondences {
    /// The camera models, indexed by the views' `IntrinsicId`.
    pub camera_models: Vec<Camera>,
    /// Whether the camera models are known calibrations rather than initial guesses.
    pub calibrated_intrinsics: bool,
    /// The views, in ascending ID order.
    pub views: Vec<ViewRecord>,
    /// The keypoints of each view, in view order and feature order.
    pub keypoints: Vec<Vec<Keypoint>>,
    /// The matches of each pair, in ascending pair order.
    pub pairs: Vec<(ViewPair, Vec<FeatureMatch>)>,
    /// The ground-truth world-to-camera pose of each view that has one, in ascending view order.
    pub ground_truth: Vec<(ViewId, WorldToCameraPose)>,
    /// The unit of the ground-truth pose centers.
    pub length_unit: LengthUnit,
}

impl Correspondences {
    /// Views with their keypoints, no matches, uncalibrated cameras, and no ground truth, in an
    /// unknown length unit.
    #[must_use]
    pub fn new(
        camera_models: Vec<Camera>,
        views: Vec<ViewRecord>,
        keypoints: Vec<Vec<Keypoint>>,
    ) -> Self {
        Self {
            camera_models,
            calibrated_intrinsics: false,
            views,
            keypoints,
            pairs: Vec::new(),
            ground_truth: Vec::new(),
            length_unit: LengthUnit::Unknown,
        }
    }

    /// Mark the camera models as known calibrations, or as initial guesses.
    #[must_use]
    pub fn with_calibrated_intrinsics(mut self, calibrated: bool) -> Self {
        self.calibrated_intrinsics = calibrated;
        self
    }

    /// Add the putative matches of `pair`.
    #[must_use]
    pub fn with_pair(mut self, pair: ViewPair, matches: Vec<FeatureMatch>) -> Self {
        self.pairs.push((pair, matches));
        self
    }

    /// Add the ground-truth pose of `view`.
    #[must_use]
    pub fn with_ground_truth(mut self, view: ViewId, pose: WorldToCameraPose) -> Self {
        self.ground_truth.push((view, pose));
        self
    }

    /// Set the unit of the ground-truth pose centers, which are not rescaled.
    #[must_use]
    pub const fn with_length_unit(mut self, length_unit: LengthUnit) -> Self {
        self.length_unit = length_unit;
        self
    }
}

/// Read the dataset at `path` as putative correspondences.
///
/// The dataset must conform and declare the `dataset` profile. Camera models, views, pose
/// blocks, and keypoints are read as [`read_view_graph`](crate::read_view_graph) reads them.
/// Every match is kept with its distance, whatever its inlier flag; the flags, the `pairs`
/// table, and pairs without matches are not read. The `gt_poses` rows, if any, become the
/// ground truth, in the length unit of the manifest.
///
/// # Errors
///
/// - [`Error::Invalid`] with every rule broken if the dataset does not conform, which covers a
///   required extension.
/// - [`Error::Unsupported`] if the dataset lacks the `dataset` profile, mixes calibrated and uncalibrated cameras, or holds a keypoint, match, or pose the SfM types
///   reject.
/// - [`Error::Sfm`] if the SfM types reject a camera model.
pub fn read_correspondences(
    path: impl AsRef<Path>,
    options: &ReadOptions,
) -> Result<Correspondences> {
    let root = path.as_ref();
    let (length_unit, tables) = load_profile(root, "dataset")?;
    let views = ViewSet::read(&tables, root, options)?;
    let keypoints = keypoints(
        table(&tables, "keypoints")?,
        &views.view_index,
        views.records.len(),
    )?;
    let pairs = match_groups(&tables)?
        .into_iter()
        .map(|((first, second), matches)| {
            let pair = ViewPair::new(
                ViewId::new(view(&views.view_index, first)?),
                ViewId::new(view(&views.view_index, second)?),
            );
            Ok((
                pair,
                matches.into_iter().map(|(matched, _)| matched).collect(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let ground_truth = match tables.get("gt_poses") {
        None => Vec::new(),
        Some(gt) => {
            let rows = Rows::new(gt);
            rows.indices("view_id")?
                .into_iter()
                .enumerate()
                .map(|(row, view_id)| {
                    let dense = ViewId::new(view(&views.view_index, view_id)?);
                    Ok((dense, pose(&rows, row)?))
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    Ok(Correspondences {
        camera_models: views.models,
        calibrated_intrinsics: views.calibrated,
        views: views.records,
        keypoints,
        pairs,
        ground_truth,
        length_unit,
    })
}

/// Write `correspondences` as a new dataset at `path`.
///
/// The dataset holds the camera models, the views, the keypoints of every view, and every match
/// with a null inlier flag, so a reader verifies the pairs itself; it has no `pairs` table. With
/// ground truth it also holds `gt_poses` and declares the `benchmark` profile besides `dataset`,
/// which then needs a distinct image for each view with a ground-truth pose. Lengths are
/// labeled with `length_unit`, unless `options` relabels them.
///
/// # Errors
///
/// - [`Error::Exists`] if `path` exists.
/// - [`Error::Unsupported`] if the keypoints are not one list per view, or fails like
///   [`write_view_graph`](crate::write_view_graph).
/// - [`Error::Invalid`] if the tables would break a rule, such as a match of an unknown feature.
/// - [`Error::Io`] or [`Error::Encoding`] if the files cannot be written; nothing is published.
/// - [`Error::Unsynced`] if the dataset was published but its directory could not be flushed.
pub fn write_correspondences(
    correspondences: &Correspondences,
    path: impl AsRef<Path>,
    options: &WriteOptions,
) -> Result<()> {
    let views = &correspondences.views;
    if correspondences.keypoints.len() != views.len() {
        return Err(Error::Unsupported(format!(
            "{} keypoint lists for {} views",
            correspondences.keypoints.len(),
            views.len()
        )));
    }
    let mut tables = Tables::new();
    insert(
        &mut tables,
        cameras(
            &correspondences.camera_models,
            views,
            correspondences.calibrated_intrinsics,
        )?,
    );
    insert(&mut tables, views_table(views, options)?);
    insert(
        &mut tables,
        keypoints_table(
            views
                .iter()
                .zip(&correspondences.keypoints)
                .map(|(view, keypoints)| (view.id, keypoints.as_slice())),
        )?,
    );
    insert(
        &mut tables,
        matches_table(
            correspondences
                .pairs
                .iter()
                .map(|(pair, matches)| (*pair, matches.as_slice(), None)),
        )?,
    );
    let length_unit = options.length_unit_or(correspondences.length_unit);
    if correspondences.ground_truth.is_empty() {
        return publish(path.as_ref(), tables, &["dataset"], length_unit);
    }
    insert(&mut tables, gt_poses_table(&correspondences.ground_truth)?);
    publish(
        path.as_ref(),
        tables,
        &["benchmark", "dataset"],
        length_unit,
    )
}

/// One row per ground-truth pose.
fn gt_poses_table(ground_truth: &[(ViewId, WorldToCameraPose)]) -> Result<Table> {
    let ids = ground_truth
        .iter()
        .map(|(view, _)| id(view.index()))
        .collect::<Result<_>>()?;
    pose_columns(
        TableBuilder::new("gt_poses", ground_truth.len()).int64("view_id", ids),
        ground_truth.iter().map(|(_, pose)| pose),
    )
    .build()
}
