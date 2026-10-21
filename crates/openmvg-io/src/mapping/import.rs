// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `dataset` as a `ViewGraph`, a `reconstruction` as a `Reconstruction`, and a `benchmark`
//! as ground truth.
//!
//! ```text
//! load + check every rule ──> violations? ──> Error::Invalid
//!        │
//!        v
//! cameras used by views ──> dense IntrinsicId      views ──> dense ViewId, dense PoseId
//!        │
//!        ├─ dataset:        keypoints per view, matches ∪ pairs per pair
//!        │                  ──> ViewGraphParts ──> ViewGraph::try_from_parts
//!        │
//!        └─ reconstruction: poses, points with their observations
//!                           ──> ReconstructionParts ──> Reconstruction::try_from_parts
//! ```
//!
//! IDs are renumbered densely in ascending order (§4), which keeps the IDs of a dataset this
//! crate wrote.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::Path;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitVector3, Vector3};
use openmvg_sfm::model::{ImageMeta, PairGeometry, ViewRecord};
use openmvg_sfm::{
    Camera, FeatureId, FeatureMatch, IntrinsicId, Keypoint, PoseId, Reconstruction,
    ReconstructionParts, RelativePose, ViewGraph, ViewGraphParts, ViewId, ViewPair,
    WorldToCameraPose,
};

use super::cameras::camera_from_row;
use super::columns::Rows;
use crate::error::{Error, Result};
use crate::format::checks::Tables;
use crate::format::table::Table;
use crate::format::{Rule, Violation, load};
use crate::options::{LengthUnit, ReadOptions};

/// Read the dataset at `path` as a `ViewGraph`.
///
/// The dataset must conform and declare the `dataset` profile. Camera models, views, and pose
/// blocks are renumbered densely in ascending ID order, and only the camera models some view
/// uses are kept. Each view's image is the file `image` below the image root, named by
/// `image`, or an external image named by the view's ID when `image` is null; its size is its
/// camera's. An image file that exists must resolve inside the image root once symbolic links
/// are followed; a missing one is accepted. The check runs once, at read time, so a file
/// replaced afterwards is not covered.
///
/// Keypoint coordinates are rounded to `f32`; a null `scale` or `orientation` reads as zero.
/// A pair is accepted when its `pairs` row names a model, which then needs an inlier flag on
/// every match, and rejected when its model is null or, without a `pairs` row, when every match
/// is an outlier. The inlier flags of a rejected pair are not kept. Every pair gets a confidence
/// weight of one, and the graph gets the root seed of `options`.
///
/// # Errors
///
/// - [`Error::Invalid`] with every rule broken if the dataset does not conform (a required
///   extension is one such rule), or if an image resolves outside the image root.
/// - [`Error::Unsupported`] if the dataset lacks the `dataset` profile, cannot resolve an image path, mixes calibrated and uncalibrated cameras, holds a pair whose
///   matches are unverified, a
///   model without inlier flags, or inliers without a model, or holds a keypoint, match, or
///   relative pose the SfM types reject, such as a keypoint coordinate outside `f32`.
/// - [`Error::Sfm`] if the SfM types reject a camera model or the assembled graph.
pub fn read_view_graph(path: impl AsRef<Path>, options: &ReadOptions) -> Result<ViewGraph> {
    let root = path.as_ref();
    let (_, tables) = load_profile(root, "dataset")?;
    let views = ViewSet::read(&tables, root, options)?;
    let keypoints = keypoints(
        table(&tables, "keypoints")?,
        &views.view_index,
        views.records.len(),
    )?;
    let mut parts = ViewGraphParts::new(options.root_seed, views.models, views.records, keypoints)
        .with_calibrated_intrinsics(views.calibrated);
    for (pair, matches, verified) in pairs(&tables, &views.view_index)? {
        parts = parts.with_pair(pair, matches, verified);
    }
    Ok(ViewGraph::try_from_parts(parts)?)
}

/// A reconstruction read from a dataset, with the metadata a `Reconstruction` cannot hold.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StoredReconstruction {
    /// The cameras, poses, points, and observations.
    pub reconstruction: Reconstruction,
    /// The unit of the pose centers and point coordinates.
    pub length_unit: LengthUnit,
}

/// Read the `reconstruction` dataset at `path` as a `Reconstruction`, with its length unit.
///
/// The dataset must conform and declare the `reconstruction` profile. Camera models, views, and
/// poses are read as [`read_view_graph`] reads them, points are renumbered densely in
/// ascending ID order, and each point's observations keep their row order. The stored `feature`
/// indices refer to the producer's features, which are not read: each observation becomes a
/// feature of its view, numbered per view in point order. Reprojection errors are recomputed from
/// the observations, so `error_px` is not read, and the reconstruction has no components, which
/// the format does not record: every `PoseRecord::component` is `None`. It gets the root seed and engine of `options`.
///
/// # Errors
///
/// - [`Error::Invalid`] with every rule broken if the dataset does not conform, which covers a
///   required extension, a pose without a view, and an observation of an unknown point or of an
///   unregistered view.
/// - [`Error::Unsupported`] if the dataset lacks the `reconstruction` profile, mixes calibrated
///   and uncalibrated cameras, or holds a pose the SfM types reject.
/// - [`Error::Sfm`] if the SfM types reject a camera model or the assembled reconstruction, for
///   example a point with fewer than two observations.
pub fn read_reconstruction(
    path: impl AsRef<Path>,
    options: &ReadOptions,
) -> Result<StoredReconstruction> {
    let root = path.as_ref();
    let (length_unit, tables) = load_profile(root, "reconstruction")?;
    let views = ViewSet::read(&tables, root, options)?;
    let mut parts = ReconstructionParts::new(
        options.root_seed,
        options.engine,
        views.models,
        views.records,
    )
    .with_calibrated_intrinsics(views.calibrated);

    let poses = Rows::new(table(&tables, "poses")?);
    for (row, pose_id) in poses.indices("id")?.into_iter().enumerate() {
        let dense = views
            .pose_index
            .get(&pose_id)
            .ok_or_else(|| Error::Unsupported(format!("pose {pose_id} is the pose of no view")))?;
        parts = parts.with_pose(PoseId::new(*dense), pose(&poses, row)?);
    }

    let points = Rows::new(table(&tables, "points")?);
    let point_rows = row_index(points.indices("id")?);
    let mut observed = vec![Vec::new(); points.len()];
    let observations = Rows::new(table(&tables, "observations")?);
    let view_ids = observations.indices("view_id")?;
    for (row, point) in observations.indices("point_id")?.into_iter().enumerate() {
        let point_row = *point_rows
            .get(&point)
            .ok_or_else(|| Error::Unsupported(format!("point {point} does not exist")))?;
        let pixel = |column| observations.double(column, row).unwrap_or(f64::NAN);
        observed[point_row].push((
            ViewId::new(view(&views.view_index, view_ids[row])?),
            Point2::new(pixel("x"), pixel("y")),
        ));
    }
    for (row, observations) in observed.into_iter().enumerate() {
        let position = points
            .double_list::<3>("xyz", row)
            .map_or_else(|| Point3::new(f64::NAN, f64::NAN, f64::NAN), Point3::from);
        parts = parts.with_point(position, observations);
    }
    Ok(StoredReconstruction {
        reconstruction: Reconstruction::try_from_parts(parts)?,
        length_unit,
    })
}

/// The camera models and views of a dataset, with the dense IDs of its views and pose blocks.
pub(super) struct ViewSet {
    pub(super) models: Vec<Camera>,
    pub(super) records: Vec<ViewRecord>,
    pub(super) calibrated: bool,
    pub(super) view_index: BTreeMap<usize, usize>,
    pose_index: BTreeMap<usize, usize>,
}

impl ViewSet {
    /// The camera models some view uses, renumbered densely, and every view, with its image
    /// resolved against the image root of `options` or else the dataset directory `root`.
    pub(super) fn read(tables: &Tables, root: &Path, options: &ReadOptions) -> Result<Self> {
        let cameras = Rows::new(table(tables, "cameras")?);
        let views = Rows::new(table(tables, "views")?);

        // Rows are sorted by ID (§5), so a view's dense ID is its row.
        let view_ids = views.indices("id")?;
        let view_index = row_index(view_ids.iter().copied());
        let camera_rows = row_index(cameras.indices("id")?);
        let view_cameras = views.indices("camera_id")?;
        let used: BTreeSet<usize> = view_cameras.iter().copied().collect();
        let camera_index = row_index(used.iter().copied());
        let camera_row = |camera: usize| {
            camera_rows
                .get(&camera)
                .copied()
                .ok_or_else(|| Error::Unsupported(format!("camera {camera} does not exist")))
        };
        let models = used
            .iter()
            .map(|&camera| camera_from_row(&cameras, camera_row(camera)?))
            .collect::<Result<Vec<Camera>>>()?;
        let calibrated = calibration(&cameras, used.iter().map(|&camera| camera_row(camera)))?;

        let view_poses = views.indices("pose_id")?;
        let pose_index = row_index(view_poses.iter().copied().collect::<BTreeSet<_>>());
        let image_root = options.image_root.as_deref().unwrap_or(root);
        // A missing root holds no image, so an image that resolves anyway lies outside it.
        let resolved_root = image_root.canonicalize().ok();
        let records = (0..views.len())
            .map(|row| {
                let camera = camera_row(view_cameras[row])?;
                let size = |column| {
                    cameras
                        .int(column, camera)
                        .and_then(|value| u32::try_from(value).ok())
                        .unwrap_or(0)
                };
                let (width, height) = (size("width"), size("height"));
                let meta = match views.string("image", row) {
                    Some(image) => {
                        let file = image_root.join(image);
                        confine(&file, resolved_root.as_deref(), view_ids[row])?;
                        // The dataset path, unlike the file name, tells apart images in different
                        // folders.
                        let mut meta = ImageMeta::path(file, width, height);
                        meta.name = image.to_owned();
                        meta
                    }
                    None => ImageMeta::external_id(view_ids[row].to_string(), width, height),
                };
                Ok(ViewRecord::new(
                    ViewId::new(row),
                    meta,
                    IntrinsicId::new(*camera_index.get(&view_cameras[row]).ok_or_else(|| {
                        Error::Unsupported(format!("camera {} does not exist", view_cameras[row]))
                    })?),
                    PoseId::new(*pose_index.get(&view_poses[row]).ok_or_else(|| {
                        Error::Unsupported(format!("pose {} does not exist", view_poses[row]))
                    })?),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            models,
            records,
            calibrated,
            view_index,
            pose_index,
        })
    }
}

/// Fail for an image file of view `view` that exists but resolves outside `resolved_root`, the
/// image root with its symbolic links followed (§10); a missing file is accepted.
fn confine(file: &Path, resolved_root: Option<&Path>, view: usize) -> Result<()> {
    let resolved = match file.canonicalize() {
        Ok(resolved) => resolved,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(Error::Unsupported(format!(
                "cannot resolve the image {}: {error}",
                file.display()
            )));
        }
    };
    if resolved_root.is_some_and(|root| resolved.starts_with(root)) {
        return Ok(());
    }
    Err(Error::Invalid(vec![Violation::in_table(
        Rule::ViewsImagePath,
        "views",
        format!("the image of view {view} resolves outside the image root"),
    )]))
}

/// The ground truth of a `benchmark` dataset.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GroundTruth {
    /// The unit of the pose centers.
    pub length_unit: LengthUnit,
    /// One entry per view with a ground-truth pose, in ascending view ID order.
    pub views: Vec<GroundTruthView>,
}

/// The ground truth of one view.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GroundTruthView {
    /// The view's image path relative to the image root, if it has one.
    pub image: Option<String>,
    /// The view's camera model.
    pub camera: Camera,
    /// The view's ground-truth world-to-camera pose.
    pub pose: WorldToCameraPose,
}

/// Read the ground-truth poses of the `benchmark` dataset at `path`.
///
/// # Errors
///
/// - [`Error::Invalid`] with every rule broken if the dataset does not conform, which covers a
///   required extension.
/// - [`Error::Unsupported`] if the dataset lacks the `benchmark` profile, or holds a pose the SfM
///   types reject.
/// - [`Error::Sfm`] if the SfM types reject a camera model.
pub fn read_ground_truth(path: impl AsRef<Path>) -> Result<GroundTruth> {
    let (length_unit, tables) = load_profile(path.as_ref(), "benchmark")?;
    let cameras = Rows::new(table(&tables, "cameras")?);
    let views = Rows::new(table(&tables, "views")?);
    let gt = Rows::new(table(&tables, "gt_poses")?);
    let camera_rows = row_index(cameras.indices("id")?);
    let view_rows = row_index(views.indices("id")?);
    let view_cameras = views.indices("camera_id")?;
    let missing = |what: &str, id: usize| Error::Unsupported(format!("{what} {id} does not exist"));
    let views = gt
        .indices("view_id")?
        .into_iter()
        .enumerate()
        .map(|(row, view)| {
            let view_row = *view_rows.get(&view).ok_or_else(|| missing("view", view))?;
            let camera = view_cameras[view_row];
            let camera_row = *camera_rows
                .get(&camera)
                .ok_or_else(|| missing("camera", camera))?;
            Ok(GroundTruthView {
                image: views.string("image", view_row).map(str::to_owned),
                camera: camera_from_row(&cameras, camera_row)?,
                pose: pose(&gt, row)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GroundTruth { length_unit, views })
}

/// Load and check the dataset at `root`, which must declare `profile` and require no extension.
pub(super) fn load_profile(root: &Path, profile: &str) -> Result<(LengthUnit, Tables)> {
    let loaded = load(root);
    let Some(manifest) = loaded.manifest.filter(|_| loaded.violations.is_empty()) else {
        return Err(Error::Invalid(loaded.violations));
    };
    if !manifest.profiles.iter().any(|declared| declared == profile) {
        return Err(Error::Unsupported(format!(
            "the dataset does not declare the {profile} profile"
        )));
    }
    Ok((manifest.length_unit, loaded.tables))
}

/// Table `name`, which the declared profile requires.
pub(super) fn table<'a>(tables: &'a Tables, name: &str) -> Result<&'a Table> {
    tables
        .get(name)
        .ok_or_else(|| Error::Unsupported(format!("the dataset has no {name} table")))
}

/// Whether the camera rows are a known calibration: all calibrated, or none.
fn calibration(cameras: &Rows<'_>, rows: impl Iterator<Item = Result<usize>>) -> Result<bool> {
    let mut flags = BTreeSet::new();
    for row in rows {
        flags.insert(cameras.boolean("calibrated", row?));
    }
    match (flags.contains(&Some(true)), flags.len()) {
        (false, _) => Ok(false),
        (true, 1) => Ok(true),
        (true, _) => Err(Error::Unsupported(
            "the views mix calibrated cameras with uncalibrated ones".to_owned(),
        )),
    }
}

/// Map source IDs to their row or dense index, preserving iterator order.
fn row_index(ids: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    ids.into_iter()
        .enumerate()
        .map(|(row, id)| (id, row))
        .collect()
}

/// A view-index lookup that names the missing ID.
pub(super) fn view(view_index: &BTreeMap<usize, usize>, view: usize) -> Result<usize> {
    view_index
        .get(&view)
        .copied()
        .ok_or_else(|| Error::Unsupported(format!("view {view} does not exist")))
}

/// The keypoints of each view, in view order and feature order.
pub(super) fn keypoints(
    table: &Table,
    view_index: &BTreeMap<usize, usize>,
    view_count: usize,
) -> Result<Vec<Vec<Keypoint>>> {
    let keypoints = Rows::new(table);
    let view_ids = keypoints.indices("view_id")?;
    let mut per_view = vec![Vec::new(); view_count];
    // Rows are sorted by `(view_id, feature)` and features are dense (§5.6), so pushing in row
    // order places each keypoint at its feature index.
    for (row, &view_id) in view_ids.iter().enumerate() {
        let value = |column| keypoints.double(column, row).unwrap_or(f64::NAN) as f32;
        let keypoint = Keypoint::try_new(
            value("x"),
            value("y"),
            keypoints.float("scale", row).unwrap_or(0.0),
            keypoints.float("orientation", row).unwrap_or(0.0),
        )
        .map_err(|error| {
            Error::Unsupported(format!("a keypoint of view {view_id} is rejected: {error}"))
        })?;
        per_view[view(view_index, view_id)?].push(keypoint);
    }
    Ok(per_view)
}

/// One pair as [`ViewGraphParts::with_pair`] takes it.
type PartPair = (
    ViewPair,
    Vec<FeatureMatch>,
    Option<(PairGeometry, Vec<bool>)>,
);

/// The matches of each pair with their inlier flags, keyed by the stored view IDs.
pub(super) type MatchGroups = BTreeMap<(usize, usize), Vec<(FeatureMatch, Option<bool>)>>;

/// Every row of the `matches` table, if any, grouped by pair in ascending order.
pub(super) fn match_groups(tables: &Tables) -> Result<MatchGroups> {
    let mut groups = MatchGroups::new();
    let Some(table) = tables.get("matches") else {
        return Ok(groups);
    };
    let matches = Rows::new(table);
    let firsts = matches.indices("view_first")?;
    let seconds = matches.indices("view_second")?;
    let feature_firsts = matches.indices("feature_first")?;
    let feature_seconds = matches.indices("feature_second")?;
    for row in 0..matches.len() {
        let matched = FeatureMatch::try_new(
            FeatureId::new(feature_firsts[row]),
            FeatureId::new(feature_seconds[row]),
            matches.float("distance", row),
        )
        .map_err(|error| {
            Error::Unsupported(format!(
                "a match of pair ({}, {}) is rejected: {error}",
                firsts[row], seconds[row]
            ))
        })?;
        groups
            .entry((firsts[row], seconds[row]))
            .or_default()
            .push((matched, matches.boolean("inlier", row)));
    }
    Ok(groups)
}

/// The matches of one pair with their inlier flags, and its `pairs` row.
#[derive(Default)]
struct PairRecord {
    matches: Vec<(FeatureMatch, Option<bool>)>,
    row: Option<usize>,
}

/// Every pair of `matches` and `pairs`, in ascending order, accepted or rejected.
fn pairs(tables: &Tables, view_index: &BTreeMap<usize, usize>) -> Result<Vec<PartPair>> {
    let mut records: BTreeMap<(usize, usize), PairRecord> = match_groups(tables)?
        .into_iter()
        .map(|(key, matches)| (key, PairRecord { matches, row: None }))
        .collect();
    let pair_rows = tables.get("pairs").map(Rows::new);
    if let Some(pair_rows) = &pair_rows {
        let firsts = pair_rows.indices("view_first")?;
        let seconds = pair_rows.indices("view_second")?;
        for row in 0..pair_rows.len() {
            records.entry((firsts[row], seconds[row])).or_default().row = Some(row);
        }
    }

    records
        .into_iter()
        .map(|((first, second), record)| {
            let name = || format!("pair ({first}, {second})");
            let views = ViewPair::new(
                ViewId::new(view(view_index, first)?),
                ViewId::new(view(view_index, second)?),
            );
            let (matches, flags): (Vec<_>, Vec<_>) = record.matches.into_iter().unzip();
            let model = record
                .row
                .zip(pair_rows.as_ref())
                .and_then(|(row, rows)| rows.string("model", row).map(|model| (row, rows, model)));
            let verified = match model {
                Some((row, rows, model)) => {
                    let mask = flags
                        .into_iter()
                        .collect::<Option<Vec<bool>>>()
                        .ok_or_else(|| {
                            Error::Unsupported(format!(
                                "{} has a model but no inlier flags",
                                name()
                            ))
                        })?;
                    Some((geometry(rows, row, model, &name)?, mask))
                }
                None if record.row.is_some() || flags.iter().all(|flag| *flag == Some(false)) => {
                    None
                }
                None if flags.iter().any(Option::is_none) => {
                    return Err(Error::Unsupported(format!(
                        "{} has unverified matches",
                        name()
                    )));
                }
                None => {
                    return Err(Error::Unsupported(format!(
                        "{} has inliers but no model",
                        name()
                    )));
                }
            };
            Ok((views, matches, verified))
        })
        .collect()
}

/// The two-view geometry of `pairs` row `row`, whose model is `model`.
fn geometry(
    pairs: &Rows<'_>,
    row: usize,
    model: &str,
    name: &dyn Fn() -> String,
) -> Result<PairGeometry> {
    let matrix = pairs
        .double_list::<9>("matrix", row)
        .map(|values| Matrix3::from_row_slice(&values));
    let rotation = pairs
        .double_list::<9>("rotation", row)
        .map(|values| Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&values)));
    let translation = pairs
        .double_list::<3>("translation", row)
        .map(|values| UnitVector3::new_unchecked(Vector3::from(values)));
    let relative_pose = match (rotation, translation) {
        (Some(rotation), Some(translation)) => Some(
            RelativePose::from_projection_translation(rotation, translation).map_err(|error| {
                Error::Unsupported(format!("{} has an invalid pose: {error}", name()))
            })?,
        ),
        _ => None,
    };
    let missing = |what: &str| Error::Unsupported(format!("{} has no {what}", name()));
    Ok(match model {
        "essential" => PairGeometry::Essential {
            matrix: matrix.ok_or_else(|| missing("matrix"))?,
            relative_pose,
        },
        "fundamental" => PairGeometry::Fundamental {
            matrix: matrix.ok_or_else(|| missing("matrix"))?,
            relative_pose,
        },
        "homography" => PairGeometry::Homography {
            matrix: matrix.ok_or_else(|| missing("matrix"))?,
            relative_pose,
        },
        "rotation" => PairGeometry::RotationOnly {
            rotation: rotation.ok_or_else(|| missing("rotation"))?,
        },
        other => {
            return Err(Error::Unsupported(format!(
                "{} has the unknown model {other:?}",
                name()
            )));
        }
    })
}

/// The world-to-camera pose of a `poses` or `gt_poses` row.
pub(super) fn pose(rows: &Rows<'_>, row: usize) -> Result<WorldToCameraPose> {
    let rotation = rows.double_list::<9>("rotation", row).map_or_else(
        || Matrix3::from_element(f64::NAN),
        |v| Matrix3::from_row_slice(&v),
    );
    let center = rows
        .double_list::<3>("center", row)
        .map_or_else(|| Point3::new(f64::NAN, f64::NAN, f64::NAN), Point3::from);
    WorldToCameraPose::from_matrix(rotation, center)
        .map_err(|error| Error::Unsupported(format!("pose row {row} is rejected: {error}")))
}
