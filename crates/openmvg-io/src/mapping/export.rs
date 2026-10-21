// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `ViewGraph` as a `dataset` and a `Reconstruction` as a `reconstruction`.
//!
//! ```text
//! ViewGraph       ──> cameras, views, keypoints, matches, pairs        ──┐
//!                                                                         ├─> validate, sort,
//! Reconstruction  ──> cameras, views, poses, points, observations      ──┘   hash, publish
//! ```

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::path::{Component, Path};

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use openmvg_core::camera::CameraModel;
use openmvg_sfm::model::{ImageMeta, PairGeometry, ViewRecord};
use openmvg_sfm::{
    Camera, FeatureMatch, Keypoint, Reconstruction, RelativePose, ViewGraph, ViewId, ViewPair,
    WorldToCameraPose,
};

use super::cameras::{CameraRow, cameras_table};
use super::columns::{TableBuilder, id, int32};
use crate::error::{Error, Result};
use crate::format::checks::Tables;
use crate::format::table::Table;
use crate::format::write::{ManifestFields, write};
use crate::options::{LengthUnit, WriteOptions};

/// The `generator` written to every manifest.
const GENERATOR: &str = concat!("openmvg-io ", env!("CARGO_PKG_VERSION"));

/// Write `graph` as a new dataset of the `dataset` profile at `path`.
///
/// The dataset holds the camera models, the views, the keypoints of every view, the putative
/// matches of every pair with their inlier flags, and one `pairs` row per pair: the verified
/// two-view model of an accepted pair, and a null model for a rejected one. Every match of a
/// rejected pair is written as an outlier.
///
/// The extraction method and the root seed are not part of the format and are not written.
///
/// # Errors
///
/// - [`Error::Exists`] if `path` exists.
/// - [`Error::Unsupported`] if views sharing a camera model have different image sizes, an image
///   file lies outside the image root, an ID exceeds `2^63 − 1`, or a feature index, image size
///   or count exceeds `2^31 − 1`.
/// - [`Error::Invalid`] if the tables would break a rule, such as two views with the same image.
/// - [`Error::Io`] or [`Error::Encoding`] if the files cannot be written; nothing is published.
/// - [`Error::Unsynced`] if the dataset was published but its directory could not be flushed.
pub fn write_view_graph(
    graph: &ViewGraph,
    path: impl AsRef<Path>,
    options: &WriteOptions,
) -> Result<()> {
    let views = graph.views();
    let mut tables = Tables::new();
    insert(
        &mut tables,
        cameras(
            graph.camera_models(),
            views,
            graph.has_calibrated_intrinsics(),
        )?,
    );
    insert(&mut tables, views_table(views, options)?);
    insert(
        &mut tables,
        keypoints_table(
            views
                .iter()
                .map(|view| (view.id, graph.keypoints(view.id).unwrap_or(&[]))),
        )?,
    );
    let (matches, pairs) = pair_tables(graph)?;
    insert(&mut tables, matches);
    insert(&mut tables, pairs);
    let length_unit = options.length_unit_or(LengthUnit::Unknown);
    publish(path.as_ref(), tables, &["dataset"], length_unit)
}

/// Write `reconstruction` as a new dataset of the `reconstruction` profile at `path`.
///
/// The dataset holds the camera models, every view, one row per registered pose (views can
/// share a pose), the points of the reconstruction without its rejected landmarks, and their
/// supporting observations. Each observation stores the
/// index of its feature in the reconstruction, since the dataset holds no keypoints. Each point
/// stores its mean reprojection error, recomputed from the written observations, or none when
/// one of them does not project.
///
/// The dataset is published as a new directory in one step, so a reader never sees a partial
/// dataset, and the write fails if `path` exists.
///
/// A `Reconstruction` holds no length unit, so lengths are labeled `unknown` unless `options`
/// relabels them; pass the [`StoredReconstruction::length_unit`](crate::StoredReconstruction)
/// of a reconstruction read back to keep its unit.
///
/// # Errors
///
/// - [`Error::Exists`] if `path` exists: remove the old dataset first, or choose a new path.
/// - Otherwise fails like [`write_view_graph`].
pub fn write_reconstruction(
    reconstruction: &Reconstruction,
    path: impl AsRef<Path>,
    options: &WriteOptions,
) -> Result<()> {
    let views = reconstruction.views();
    let mut tables = Tables::new();
    insert(
        &mut tables,
        cameras(
            reconstruction.camera_models(),
            views,
            reconstruction.has_calibrated_intrinsics(),
        )?,
    );
    insert(&mut tables, views_table(views, options)?);
    insert(&mut tables, poses_table(reconstruction)?);
    let (points, observations) = point_tables(reconstruction)?;
    insert(&mut tables, points);
    insert(&mut tables, observations);
    let length_unit = options.length_unit_or(LengthUnit::Unknown);
    publish(path.as_ref(), tables, &["reconstruction"], length_unit)
}

pub(super) fn insert(tables: &mut Tables, table: Table) {
    tables.insert(table.name.clone(), table);
}

/// Publish `tables` as a new dataset declaring `profiles`, given in ascending order, with its
/// lengths labeled `length_unit`.
pub(super) fn publish(
    path: &Path,
    tables: Tables,
    profiles: &[&str],
    length_unit: LengthUnit,
) -> Result<()> {
    let fields = ManifestFields {
        profiles: profiles.iter().map(|&profile| profile.to_owned()).collect(),
        length_unit,
        generator: Some(GENERATOR.to_owned()),
        extensions_used: Vec::new(),
    };
    write(path, tables, &fields)?;
    Ok(())
}

/// Matrix element `(i, j)` at index `3i + j` (§4).
pub(super) fn row_major(matrix: &Matrix3<f64>) -> [f64; 9] {
    std::array::from_fn(|index| matrix[(index / 3, index % 3)])
}

pub(super) fn vector(value: &Vector3<f64>) -> [f64; 3] {
    [value.x, value.y, value.z]
}

/// One row per camera model, sized by the images of the views that use it.
pub(super) fn cameras(cameras: &[Camera], views: &[ViewRecord], calibrated: bool) -> Result<Table> {
    let mut sizes = BTreeMap::new();
    for view in views {
        let size = (view.meta.width, view.meta.height);
        match sizes.entry(view.intrinsic_id.index()) {
            Entry::Vacant(entry) => {
                entry.insert(size);
            }
            Entry::Occupied(entry) if *entry.get() != size => {
                return Err(Error::Unsupported(format!(
                    "camera model {} is used by images of {}x{} and {}x{} pixels",
                    view.intrinsic_id.index(),
                    entry.get().0,
                    entry.get().1,
                    size.0,
                    size.1
                )));
            }
            Entry::Occupied(_) => {}
        }
    }
    let rows = sizes
        .into_iter()
        .map(|(intrinsic, (width, height))| {
            let camera = cameras.get(intrinsic).ok_or_else(|| {
                Error::Unsupported(format!("camera model {intrinsic} does not exist"))
            })?;
            let row = CameraRow::new(camera, int32(width)?, int32(height)?, calibrated)?;
            Ok((id(intrinsic)?, row))
        })
        .collect::<Result<Vec<_>>>()?;
    cameras_table(&rows)
}

/// The `image` of a view: its file path relative to the image root, or its image name.
fn image_path(meta: &ImageMeta, image_root: Option<&Path>) -> Result<String> {
    let (Some(root), Some(file)) = (image_root, meta.file_path()) else {
        return Ok(meta.name.clone());
    };
    let outside = || {
        Error::Unsupported(format!(
            "image {} is not a file below the image root {}",
            file.display(),
            root.display()
        ))
    };
    let relative = file.strip_prefix(root).map_err(|_| outside())?;
    let parts = relative
        .components()
        .map(|component| match component {
            Component::Normal(part) => part.to_str().ok_or_else(|| {
                Error::Unsupported(format!("image path {} is not UTF-8", file.display()))
            }),
            _ => Err(outside()),
        })
        .collect::<Result<Vec<_>>>()?;
    if parts.is_empty() {
        return Err(outside());
    }
    Ok(parts.join("/"))
}

pub(super) fn views_table(views: &[ViewRecord], options: &WriteOptions) -> Result<Table> {
    let ids = |of: &dyn Fn(&ViewRecord) -> usize| -> Result<Vec<i64>> {
        views.iter().map(|view| id(of(view))).collect()
    };
    let images = views
        .iter()
        .map(|view| image_path(&view.meta, options.image_root.as_deref()).map(Some))
        .collect::<Result<Vec<_>>>()?;
    TableBuilder::new("views", views.len())
        .int64("id", ids(&|view| view.id.index())?)
        .int64("camera_id", ids(&|view| view.intrinsic_id.index())?)
        .int64("pose_id", ids(&|view| view.pose_id.index())?)
        .nullable_string("image", &images)
        .build()
}

/// Every keypoint of every view, with a null `scale` where the detector recorded none, and a
/// null `orientation` where it recorded neither a scale nor an orientation. Both read back as
/// zero, so the keypoints read back equal.
pub(super) fn keypoints_table<'a>(
    per_view: impl Iterator<Item = (ViewId, &'a [Keypoint])>,
) -> Result<Table> {
    let mut view_ids = Vec::new();
    let mut features = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut scales = Vec::new();
    let mut orientations = Vec::new();
    for (view, keypoints) in per_view {
        let view_id = id(view.index())?;
        for (feature, keypoint) in keypoints.iter().enumerate() {
            let has_scale = keypoint.scale() > 0.0;
            view_ids.push(view_id);
            features.push(int32(feature)?);
            xs.push(f64::from(keypoint.x()));
            ys.push(f64::from(keypoint.y()));
            scales.push(has_scale.then_some(keypoint.scale()));
            // Bitwise, so a negative zero stays recorded.
            let has_orientation = has_scale || keypoint.orientation().to_bits() != 0;
            orientations.push(has_orientation.then_some(keypoint.orientation()));
        }
    }
    TableBuilder::new("keypoints", view_ids.len())
        .int64("view_id", view_ids)
        .int32("feature", features)
        .double("x", xs)
        .double("y", ys)
        .nullable_float("scale", scales)
        .nullable_float("orientation", orientations)
        .build()
}

/// The `pairs` columns of one pair.
#[derive(Default)]
struct PairRow {
    model: Option<String>,
    matrix: Option<[f64; 9]>,
    rotation: Option<[f64; 9]>,
    translation: Option<[f64; 3]>,
}

impl PairRow {
    fn new(geometry: Option<PairGeometry>) -> Result<Self> {
        let model = |name: &str, matrix: &Matrix3<f64>, pose: Option<RelativePose>| PairRow {
            model: Some(name.to_owned()),
            matrix: Some(row_major(matrix)),
            rotation: pose.map(|pose| row_major(pose.rotation_source_to_target().matrix())),
            translation: pose.map(|pose| vector(pose.translation_source_to_target())),
        };
        Ok(match geometry {
            None => PairRow::default(),
            Some(PairGeometry::Essential {
                matrix,
                relative_pose,
            }) => model("essential", &matrix, relative_pose),
            Some(PairGeometry::Fundamental {
                matrix,
                relative_pose,
            }) => model("fundamental", &matrix, relative_pose),
            Some(PairGeometry::Homography {
                matrix,
                relative_pose,
            }) => model("homography", &matrix, relative_pose),
            Some(PairGeometry::RotationOnly { rotation }) => PairRow {
                model: Some("rotation".to_owned()),
                rotation: Some(row_major(rotation.matrix())),
                ..PairRow::default()
            },
            Some(_) => {
                return Err(Error::Unsupported(
                    "a pair geometry has no OMVG model".to_owned(),
                ));
            }
        })
    }
}

/// The matches of one pair, with the inlier flags of a verified pair or none.
pub(super) type MatchRows<'a> = (ViewPair, &'a [FeatureMatch], Option<&'a [bool]>);

/// The `matches` table: every match of every pair, with a null inlier flag where the pair has no
/// flags, and `false` where its flags end before the match.
pub(super) fn matches_table<'a>(pairs: impl Iterator<Item = MatchRows<'a>>) -> Result<Table> {
    let mut views = (Vec::new(), Vec::new());
    let mut features = (Vec::new(), Vec::new());
    let mut distances = Vec::new();
    let mut inliers = Vec::new();
    for (pair, matches, flags) in pairs {
        let first = id(pair.first().index())?;
        let second = id(pair.second().index())?;
        for (index, matched) in matches.iter().enumerate() {
            views.0.push(first);
            views.1.push(second);
            features.0.push(int32(matched.first().index())?);
            features.1.push(int32(matched.second().index())?);
            distances.push(matched.distance());
            inliers.push(flags.map(|flags| flags.get(index).copied().unwrap_or(false)));
        }
    }
    TableBuilder::new("matches", distances.len())
        .int64("view_first", views.0)
        .int64("view_second", views.1)
        .int32("feature_first", features.0)
        .int32("feature_second", features.1)
        .nullable_float("distance", distances)
        .nullable_boolean("inlier", inliers)
        .build()
}

/// The `matches` and `pairs` tables: every putative match with its inlier flag, and one row per
/// pair, so that rejected pairs and pairs without matches survive a round trip.
fn pair_tables(graph: &ViewGraph) -> Result<(Table, Table)> {
    // A rejected pair has no flags; all its matches are outliers.
    let matches = matches_table(graph.pairs().map(|pair| {
        let flags = pair.verified().map_or(&[][..], |(_, flags)| flags);
        (pair.views(), pair.matches(), Some(flags))
    }))?;
    let mut pair_views = (Vec::new(), Vec::new());
    let mut rows = Vec::new();
    for pair in graph.pairs() {
        pair_views.0.push(id(pair.views().first().index())?);
        pair_views.1.push(id(pair.views().second().index())?);
        rows.push(PairRow::new(pair.verified().map(|(geometry, _)| geometry))?);
    }
    let column = |of: &dyn Fn(&PairRow) -> Option<String>| -> Vec<Option<String>> {
        rows.iter().map(of).collect()
    };
    let lists = |of: &dyn Fn(&PairRow) -> Option<[f64; 9]>| -> Vec<Option<[f64; 9]>> {
        rows.iter().map(of).collect()
    };
    let translations: Vec<_> = rows.iter().map(|row| row.translation).collect();
    let pairs = TableBuilder::new("pairs", rows.len())
        .int64("view_first", pair_views.0)
        .int64("view_second", pair_views.1)
        .nullable_string("model", &column(&|row| row.model.clone()))
        .double_list("matrix", &lists(&|row| row.matrix))
        .double_list("rotation", &lists(&|row| row.rotation))
        .double_list("translation", &translations)
        .build()?;
    Ok((matches, pairs))
}

/// Add the row-major rotation and center columns of a pose table.
pub(super) fn pose_columns<'a>(
    table: TableBuilder,
    poses: impl Iterator<Item = &'a WorldToCameraPose>,
) -> TableBuilder {
    let (rotations, centers): (Vec<_>, Vec<_>) = poses
        .map(|pose| {
            (
                Some(row_major(pose.rotation_matrix())),
                Some(vector(&pose.center().coords)),
            )
        })
        .unzip();
    table
        .double_list("rotation", &rotations)
        .double_list("center", &centers)
}

/// One row per registered pose block.
fn poses_table(reconstruction: &Reconstruction) -> Result<Table> {
    let poses: BTreeMap<usize, &WorldToCameraPose> = reconstruction
        .registered_views()
        .map(|(_, record)| (record.id.index(), &record.pose))
        .collect();
    let ids = poses.keys().map(|&pose| id(pose)).collect::<Result<_>>()?;
    pose_columns(
        TableBuilder::new("poses", poses.len()).int64("id", ids),
        poses.values().copied(),
    )
    .build()
}

/// The reprojection distance of `point` measured at `pixel` in `view`, if it projects.
fn reprojection_error(
    reconstruction: &Reconstruction,
    view: ViewId,
    point: &Point3<f64>,
    pixel: &Point2<f64>,
) -> Option<f64> {
    let record = reconstruction.views().get(view.index())?;
    let camera = reconstruction
        .camera_models()
        .get(record.intrinsic_id.index())?;
    let pose = &reconstruction.pose(view)?.pose;
    let projected = camera.project(&pose.project_to_camera(point).coords).ok()?;
    Some((projected - pixel).norm())
}

/// The active landmarks as `points` and their supporting observations as `observations`.
fn point_tables(reconstruction: &Reconstruction) -> Result<(Table, Table)> {
    let landmarks: Vec<_> = reconstruction.landmarks().iter_active().collect();
    let points: BTreeMap<usize, &Point3<f64>> = landmarks
        .iter()
        .map(|landmark| (landmark.id.index(), &landmark.point))
        .collect();
    // Per point: the sum of the reprojection distances, the observation count, and whether
    // every observation projected.
    let mut errors: BTreeMap<usize, (f64, usize, bool)> = BTreeMap::new();
    let mut point_ids = Vec::new();
    let mut view_ids = Vec::new();
    let mut features = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for observation in reconstruction.observations() {
        let landmark = observation.landmark.index();
        let error = points.get(&landmark).and_then(|point| {
            reprojection_error(reconstruction, observation.view, point, &observation.pixel)
        });
        let entry = errors.entry(landmark).or_insert((0.0, 0, true));
        entry.0 += error.unwrap_or(0.0);
        entry.1 += 1;
        entry.2 &= error.is_some();
        point_ids.push(id(landmark)?);
        view_ids.push(id(observation.view.index())?);
        features.push(Some(int32(observation.feature.index())?));
        xs.push(observation.pixel.x);
        ys.push(observation.pixel.y);
    }
    let mean_error = |landmark: usize| {
        errors
            .get(&landmark)
            .filter(|(_, count, projected)| *projected && *count > 0)
            .map(|(sum, count, _)| sum / *count as f64)
            .filter(|error| error.is_finite())
    };
    let ids = points
        .keys()
        .map(|&point| id(point))
        .collect::<Result<_>>()?;
    let xyz: Vec<_> = points
        .values()
        .map(|point| Some(vector(&point.coords)))
        .collect();
    let error_px = points.keys().map(|&point| mean_error(point)).collect();
    let points = TableBuilder::new("points", xyz.len())
        .int64("id", ids)
        .double_list("xyz", &xyz)
        .nullable_double("error_px", error_px)
        .build()?;
    let observations = TableBuilder::new("observations", xs.len())
        .int64("point_id", point_ids)
        .int64("view_id", view_ids)
        .nullable_int32("feature", features)
        .double("x", xs)
        .double("y", ys)
        .build()?;
    Ok((points, observations))
}
