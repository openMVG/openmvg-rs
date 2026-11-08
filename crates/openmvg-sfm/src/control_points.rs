// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Control points: scale and register a reconstruction with points of known geometry.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Control-point registration (register_control_points)                 │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: a reconstruction; groups of points with known coordinates, each
//!        group in a rigid frame of its own (a marker, a scale bar, a
//!        surveyed set); pixel observations of the points by image name
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Triangulate every control point                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Robust track triangulation over its observations in registered views,
//! with the reconstruction's poses and cameras held fixed.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: One rotation per group, one scale for all                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Per group with three or more points: centre both point sets and fit
//! the Kabsch rotation Rg. The shared scale minimizing
//! Σg Σi ‖s·Rg·(xi − x̄g) − (Xi − X̄g)‖² is
//! s = Σg Σi (Xi − X̄g)ᵀ Rg (xi − x̄g) / Σg Σi ‖xi − x̄g‖².
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Transform                                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! With a reference group, the reconstruction moves into its frame:
//! X = s·Rg·(x − x̄g) + X̄g. Otherwise it is only scaled: X = s·x.
//!   │
//!   ▼
//! Output: the similarity, the transformed reconstruction, and residuals
//!         per point and per group in the control unit
//! ```
//!
//! A reconstruction from images alone has no scale and an arbitrary frame. Points whose
//! geometry is known fix both. The geometry need not be one frame for the whole scene: a printed
//! marker knows only its own corners, a scale bar only its two ends. Each such group says how
//! far apart its points are, and therefore the scale, but not where it sits in the scene; a
//! group that does cover the scene, such as surveyed ground control points, also fixes the frame.
//! Fitting one scale to all groups at once weighs every group by the spread of its points, so a
//! large marker counts more than a small one, as it should.
//!
//! The per-group fit is the Umeyama similarity with the scale shared across groups; with one
//! group it is exactly Umeyama's: S. Umeyama, "Least-Squares Estimation of Transformation
//! Parameters Between Two Point Patterns", IEEE TPAMI 13(4), 1991.
//!
//! Pixel coordinates follow this library's convention: the centre of the top-left pixel is
//! `(0, 0)`. Coordinates exported by tools that put it at `(0.5, 0.5)`, such as COLMAP, must be
//! shifted by half a pixel.
//!
//! The poses stay fixed, so the triangulated control points carry the reconstruction's own
//! errors; the residuals measure how well the reconstruction agrees with the known geometry.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, Similarity3, Translation3, UnitQuaternion};
use openmvg_core::model::WorldToCameraPose;
use openmvg_geometry::alignment::{AlignmentScale, fit_alignment};
use openmvg_geometry::correspondence::Point3Correspondence;
use openmvg_geometry::triangulation::{
    TrackTriangulator, TriangulationContext, TriangulationObservation,
};

use crate::{Camera, Reconstruction, Result, SfmError, ViewId};

/// Points with known coordinates in one rigid frame, such as the corners of a printed marker.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ControlGroup {
    /// Name used in the report.
    pub name: String,
    /// Known coordinates of each point, by point name, in the group's frame and the control unit.
    pub points: BTreeMap<String, Point3<f64>>,
}

impl ControlGroup {
    /// A group of named points with known coordinates.
    #[must_use]
    pub fn new(name: impl Into<String>, points: BTreeMap<String, Point3<f64>>) -> Self {
        Self {
            name: name.into(),
            points,
        }
    }
}

/// Control-point groups and their pixel observations.
///
/// Point names are unique across groups. Observations are keyed by image name (the file name of
/// a path image) and then by point name.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ControlPoints {
    /// Name of the control unit, for the report only (for example `"mm"`).
    #[serde(default)]
    pub unit: Option<String>,
    /// The groups whose coordinates are known.
    pub groups: Vec<ControlGroup>,
    /// The group whose frame becomes the reconstruction's frame. `None` keeps the reconstruction's
    /// frame and only scales it; a single group is not used as the frame unless named here.
    #[serde(default)]
    pub reference_group: Option<String>,
    /// Pixel observations: image name, then point name, then the pixel.
    pub observations: BTreeMap<String, BTreeMap<String, Point2<f64>>>,
}

/// One control point after triangulation and registration.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ControlPointReport {
    /// Group of the point.
    pub group: String,
    /// Name of the point.
    pub point: String,
    /// Observations in registered views.
    pub observation_count: usize,
    /// Observations the robust triangulation kept.
    pub inlier_count: usize,
    /// Mean reprojection error of the kept observations, in pixels; `None` when the point could
    /// not be triangulated.
    pub mean_reprojection_px: Option<f64>,
    /// Triangulated position in the registered reconstruction's frame and the control unit;
    /// `None` when the point could not be triangulated.
    pub position: Option<Point3<f64>>,
    /// Distance between the registered point and its known position, in the control unit;
    /// `None` when its group took no part in the fit.
    pub residual: Option<f64>,
}

/// One group's agreement with the shared fit.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ControlGroupReport {
    /// Name of the group.
    pub name: String,
    /// Points of the group that were triangulated.
    pub triangulated_point_count: usize,
    /// The group's own Umeyama scale, in control units per reconstruction unit; `None` when it
    /// has fewer than three triangulated points or they are degenerate.
    pub scale: Option<f64>,
    /// Root mean square residual of its points under the shared scale, in the control unit.
    pub rms_residual: Option<f64>,
}

/// What a control-point registration found.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ControlRegistrationReport {
    /// Name of the control unit, copied from the input.
    pub unit: Option<String>,
    /// Shared scale, in control units per reconstruction unit.
    pub scale: f64,
    /// Relative standard deviation of the per-group scales around the shared scale; `None` with
    /// fewer than two groups in the fit.
    pub group_scale_spread: Option<f64>,
    /// The group whose frame the reconstruction took, if any.
    pub reference_group: Option<String>,
    /// Root mean square residual over every fitted point, in the control unit.
    pub rms_residual: f64,
    /// Per-group results, in input order.
    pub groups: Vec<ControlGroupReport>,
    /// Per-point results, by group in input order, then point name.
    pub points: Vec<ControlPointReport>,
}

/// A similarity and the report of the registration that produced it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ControlRegistration {
    /// Maps reconstruction coordinates to control coordinates: `X = s·R·x + t`.
    pub similarity: Similarity3<f64>,
    /// Residuals and scales of the fit.
    pub report: ControlRegistrationReport,
}

struct FittedGroup {
    index: usize,
    names: Vec<String>,
    model: Vec<Point3<f64>>,
    known: Vec<Point3<f64>>,
    rotation: Rotation3<f64>,
    model_centroid: Point3<f64>,
    known_centroid: Point3<f64>,
    own_scale: Option<f64>,
}

/// Triangulate the control points in `reconstruction` and fit the similarity that maps it onto
/// their known geometry.
///
/// # Errors
///
/// Returns [`SfmError::InvalidInput`] when a point name repeats across groups, an observation
/// names an unknown point or the reference group does not exist, and when no group has three
/// triangulated, non-degenerate points.
pub fn register_control_points(
    reconstruction: &Reconstruction,
    control: &ControlPoints,
) -> Result<ControlRegistration> {
    let mut group_of_point = BTreeMap::new();
    for (index, group) in control.groups.iter().enumerate() {
        for name in group.points.keys() {
            if group_of_point.insert(name.as_str(), index).is_some() {
                return Err(SfmError::InvalidInput(format!(
                    "control point {name} belongs to more than one group"
                )));
            }
        }
    }
    let reference = match &control.reference_group {
        Some(name) => Some(
            control
                .groups
                .iter()
                .position(|group| &group.name == name)
                .ok_or_else(|| {
                    SfmError::InvalidInput(format!("reference group {name} does not exist"))
                })?,
        ),
        None => None,
    };

    // Registered views by image name, with their poses and cameras.
    let mut view_of_image = BTreeMap::new();
    let mut poses = BTreeMap::<ViewId, WorldToCameraPose>::new();
    let mut cameras = BTreeMap::<ViewId, Camera>::new();
    for (index, view) in reconstruction.views().iter().enumerate() {
        let view_id = ViewId::new(index);
        let (Some(pose), Some(camera)) = (
            reconstruction.pose(view_id),
            reconstruction
                .camera_models()
                .get(view.intrinsic_id.index()),
        ) else {
            continue;
        };
        view_of_image.insert(view.meta.name.as_str(), view_id);
        poses.insert(view_id, pose.pose.clone());
        cameras.insert(view_id, camera.clone());
    }

    // Observations per point, in registered views only.
    let mut observations = BTreeMap::<&str, Vec<TriangulationObservation>>::new();
    for (image, points) in &control.observations {
        for (name, pixel) in points {
            if !group_of_point.contains_key(name.as_str()) {
                return Err(SfmError::InvalidInput(format!(
                    "image {image} observes unknown control point {name}"
                )));
            }
            if let Some(view_id) = view_of_image.get(image.as_str()) {
                observations
                    .entry(name.as_str())
                    .or_default()
                    .push(TriangulationObservation::new(*view_id, *pixel));
            }
        }
    }

    // Step 1: triangulate.
    let triangulator = TrackTriangulator::robust();
    let context = TriangulationContext::new(&poses, &cameras);
    let mut point_reports = Vec::new();
    let mut triangulated = BTreeMap::<&str, Point3<f64>>::new();
    for group in &control.groups {
        for name in group.points.keys() {
            let seen = observations
                .get(name.as_str())
                .map_or(&[][..], Vec::as_slice);
            let track = (seen.len() >= 2)
                .then(|| triangulator.triangulate(seen, context).ok())
                .flatten();
            if let Some(track) = &track {
                triangulated.insert(name.as_str(), track.point);
            }
            point_reports.push(ControlPointReport {
                group: group.name.clone(),
                point: name.clone(),
                observation_count: seen.len(),
                inlier_count: track.as_ref().map_or(0, |track| track.inlier_indices.len()),
                mean_reprojection_px: track
                    .as_ref()
                    .map(|track| track.quality.mean_pixel_residual.value()),
                position: None,
                residual: None,
            });
        }
    }

    // Step 2: one rotation per group, one scale for all.
    let mut fitted = Vec::new();
    for (index, group) in control.groups.iter().enumerate() {
        let (names, (model, known)): (Vec<String>, (Vec<_>, Vec<_>)) = group
            .points
            .iter()
            .filter_map(|(name, known)| {
                triangulated
                    .get(name.as_str())
                    .map(|model| (name.clone(), (*model, *known)))
            })
            .unzip();
        if model.len() < 3 {
            continue;
        }
        let correspondences: Vec<_> = model
            .iter()
            .zip(&known)
            .map(|(model, known)| Point3Correspondence::new(*model, *known))
            .collect();
        let Ok(similarity) = fit_alignment(&correspondences, None, AlignmentScale::Estimated)
        else {
            continue;
        };
        fitted.push(FittedGroup {
            index,
            names,
            model_centroid: centroid(&model),
            known_centroid: centroid(&known),
            model,
            known,
            rotation: similarity.isometry.rotation.to_rotation_matrix(),
            own_scale: Some(similarity.scaling()),
        });
    }
    let (numerator, denominator) = fitted.iter().fold((0.0, 0.0), |(num, den), group| {
        group
            .model
            .iter()
            .zip(&group.known)
            .fold((num, den), |(num, den), (model, known)| {
                let x = group.rotation * (model - group.model_centroid);
                let y = known - group.known_centroid;
                (num + y.dot(&x), den + x.norm_squared())
            })
    });
    let positive = |value: f64| value.is_finite() && value > 0.0;
    if fitted.is_empty() || !positive(denominator) || !positive(numerator) {
        return Err(SfmError::InvalidInput(
            "no control group has three triangulated, non-degenerate points".to_owned(),
        ));
    }
    let scale = numerator / denominator;

    // Residuals under the shared scale.
    let mut group_reports: Vec<ControlGroupReport> = control
        .groups
        .iter()
        .map(|group| ControlGroupReport {
            name: group.name.clone(),
            triangulated_point_count: group
                .points
                .keys()
                .filter(|name| triangulated.contains_key(name.as_str()))
                .count(),
            scale: None,
            rms_residual: None,
        })
        .collect();
    let mut squared_sum = 0.0;
    let mut residual_count = 0usize;
    for group in &fitted {
        let mut group_sum = 0.0;
        for ((name, model), known) in group.names.iter().zip(&group.model).zip(&group.known) {
            let registered =
                group.known_centroid + scale * (group.rotation * (model - group.model_centroid));
            let residual = (registered - known).norm();
            group_sum += residual * residual;
            if let Some(report) = point_reports
                .iter_mut()
                .find(|report| &report.point == name)
            {
                report.residual = Some(residual);
            }
        }
        squared_sum += group_sum;
        residual_count += group.model.len();
        let report = &mut group_reports[group.index];
        report.scale = group.own_scale;
        report.rms_residual = Some((group_sum / group.model.len() as f64).sqrt());
    }
    let group_scales: Vec<f64> = fitted.iter().filter_map(|group| group.own_scale).collect();
    let group_scale_spread = (group_scales.len() >= 2).then(|| {
        let n = group_scales.len() as f64;
        let variance = group_scales
            .iter()
            .map(|own| (own - scale).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt() / scale
    });

    // Step 3: the similarity.
    let similarity = match reference {
        Some(index) => {
            let group = fitted
                .iter()
                .find(|group| group.index == index)
                .ok_or_else(|| {
                    SfmError::InvalidInput(format!(
                        "reference group {} has fewer than three triangulated points",
                        control.groups[index].name
                    ))
                })?;
            let rotation = UnitQuaternion::from_rotation_matrix(&group.rotation);
            let translation = group.known_centroid.coords
                - scale * (group.rotation * group.model_centroid.coords);
            Similarity3::from_parts(Translation3::from(translation), rotation, scale)
        }
        None => Similarity3::from_scaling(scale),
    };

    for report in &mut point_reports {
        report.position = triangulated
            .get(report.point.as_str())
            .map(|point| similarity.transform_point(point));
    }

    Ok(ControlRegistration {
        similarity,
        report: ControlRegistrationReport {
            unit: control.unit.clone(),
            scale,
            group_scale_spread,
            reference_group: reference.map(|index| control.groups[index].name.clone()),
            rms_residual: (squared_sum / residual_count as f64).sqrt(),
            groups: group_reports,
            points: point_reports,
        },
    })
}

fn centroid(points: &[Point3<f64>]) -> Point3<f64> {
    let sum = points
        .iter()
        .fold(nalgebra::Vector3::zeros(), |sum, point| sum + point.coords);
    Point3::from(sum / points.len() as f64)
}

/// The world-to-camera pose of a camera after the world moves by `similarity`.
pub(crate) fn transform_pose(
    pose: &WorldToCameraPose,
    similarity: &Similarity3<f64>,
) -> Result<WorldToCameraPose> {
    let world_rotation: Matrix3<f64> = similarity
        .isometry
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rotation = Rotation3::from_matrix(&(pose.rotation_matrix() * world_rotation.transpose()));
    WorldToCameraPose::from_rotation_center(rotation, similarity.transform_point(pose.center()))
        .map_err(|error| SfmError::InvalidInput(format!("transformed pose is invalid: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageMeta, Landmark, Landmarks, ObservationSupport};
    use crate::workflow::run::SfmEngine;
    use crate::{IntrinsicId, LandmarkId, SfmWorkspace, TrackId};
    use nalgebra::Vector3;
    use openmvg_core::camera::CameraModel;

    const VIEWS: usize = 6;

    /// The model frame of the test reconstruction: world (mm) = `WORLD_FROM_MODEL` · model.
    fn world_from_model() -> Similarity3<f64> {
        Similarity3::new(
            Vector3::new(30.0, -10.0, 5.0),
            Vector3::new(0.2, -0.4, 0.1),
            50.0,
        )
    }

    fn camera() -> Camera {
        Camera::try_pinhole(1000.0, 1000.0, 640.0, 480.0).unwrap()
    }

    /// Cameras 600 mm from the origin on an arc, aiming at it, in world millimetres.
    fn world_poses() -> Vec<WorldToCameraPose> {
        (0..VIEWS)
            .map(|view| {
                let angle = -0.5 + view as f64 * 0.2;
                let centre = Point3::new(600.0 * angle.sin(), 40.0, -600.0 * angle.cos());
                let rotation = Rotation3::look_at_lh(&(Point3::origin() - centre), &Vector3::y());
                WorldToCameraPose::from_rotation_center(rotation, centre).unwrap()
            })
            .collect()
    }

    /// The reconstruction in model units, with one landmark at the world origin.
    fn reconstruction() -> Reconstruction {
        let model_from_world = world_from_model().inverse();
        let mut workspace = SfmWorkspace::builder()
            .image_meta((0..VIEWS).map(|v| ImageMeta::external_id(format!("v{v}.jpg"), 1280, 960)))
            .camera_models([camera()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        for (view, pose) in world_poses().iter().enumerate() {
            let pose = transform_pose(pose, &model_from_world).unwrap();
            workspace
                .set_pose_for_view(ViewId::new(view), pose, None)
                .unwrap();
        }
        workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
            LandmarkId::new(0),
            TrackId::new(0),
            model_from_world.transform_point(&Point3::origin()),
            openmvg_core::PixelResidual::try_new(0.5).unwrap(),
            ObservationSupport::default(),
        )])
        .unwrap();
        Reconstruction::new(workspace, SfmEngine::Incremental, false)
    }

    /// A 10 mm square marker: its corners in its own frame, and where it sits in the world.
    fn marker(
        name: &str,
        world_from_marker: &Similarity3<f64>,
    ) -> (ControlGroup, Vec<(String, Point3<f64>)>) {
        let corners = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
        let mut points = BTreeMap::new();
        let mut world = Vec::new();
        for (k, (x, y)) in corners.into_iter().enumerate() {
            let local = Point3::new(x, y, 0.0);
            points.insert(format!("{name}:{k}"), local);
            world.push((
                format!("{name}:{k}"),
                world_from_marker.transform_point(&local),
            ));
        }
        (ControlGroup::new(name, points), world)
    }

    /// Observations of world points in every view, all exact.
    fn observe(
        world_points: &[(String, Point3<f64>)],
    ) -> BTreeMap<String, BTreeMap<String, Point2<f64>>> {
        let camera = camera();
        let mut observations = BTreeMap::new();
        for (view, pose) in world_poses().iter().enumerate() {
            let seen: BTreeMap<_, _> = world_points
                .iter()
                .map(|(name, point)| {
                    let pixel = camera
                        .project(&pose.project_to_camera(point).coords)
                        .unwrap();
                    (name.clone(), pixel)
                })
                .collect();
            observations.insert(format!("v{view}.jpg"), seen);
        }
        observations
    }

    fn two_markers() -> ControlPoints {
        let (a, mut world) = marker(
            "a",
            &Similarity3::new(
                Vector3::new(-20.0, 0.0, 0.0),
                Vector3::new(0.0, 0.3, 0.0),
                1.0,
            ),
        );
        let (b, world_b) = marker(
            "b",
            &Similarity3::new(
                Vector3::new(25.0, 10.0, -5.0),
                Vector3::new(0.1, -0.2, 0.0),
                1.0,
            ),
        );
        world.extend(world_b);
        ControlPoints {
            unit: Some("mm".to_owned()),
            groups: vec![a, b],
            reference_group: None,
            observations: observe(&world),
        }
    }

    #[test]
    fn markers_of_their_own_frames_fix_only_the_scale() {
        let registration = register_control_points(&reconstruction(), &two_markers()).unwrap();

        let report = &registration.report;
        assert!((report.scale - 50.0).abs() < 1e-6, "{report:?}");
        assert!(report.rms_residual < 1e-6, "{report:?}");
        assert!(report.group_scale_spread.unwrap() < 1e-8, "{report:?}");
        assert_eq!(
            registration.similarity,
            Similarity3::from_scaling(report.scale)
        );
        assert!(
            report
                .points
                .iter()
                .all(|point| point.inlier_count == VIEWS)
        );
    }

    #[test]
    fn a_reference_group_moves_the_reconstruction_into_its_frame() {
        let survey: Vec<(String, Point3<f64>)> = [
            (0.0, 0.0, 0.0),
            (40.0, 0.0, 0.0),
            (0.0, 30.0, 0.0),
            (0.0, 0.0, 25.0),
        ]
        .into_iter()
        .enumerate()
        .map(|(k, (x, y, z))| (format!("s{k}"), Point3::new(x, y, z)))
        .collect();
        let control = ControlPoints {
            unit: Some("mm".to_owned()),
            groups: vec![ControlGroup::new(
                "survey",
                survey.iter().cloned().collect(),
            )],
            reference_group: Some("survey".to_owned()),
            observations: observe(&survey),
        };
        let reconstruction = reconstruction();

        let registration = register_control_points(&reconstruction, &control).unwrap();
        let registered = reconstruction
            .transformed(&registration.similarity)
            .unwrap();

        for (view, pose) in world_poses().iter().enumerate() {
            let moved = registered.pose(ViewId::new(view)).unwrap().pose.clone();
            assert!((moved.center() - pose.center()).norm() < 1e-6);
            assert!((moved.rotation_matrix() - pose.rotation_matrix()).norm() < 1e-9);
        }
        let origin = registered
            .landmarks()
            .get(LandmarkId::new(0))
            .unwrap()
            .point;
        assert!(origin.coords.norm() < 1e-6, "{origin}");
        let position = registration.report.points[0].position.unwrap();
        assert!((position - survey[0].1).norm() < 1e-6);
    }

    #[test]
    fn a_wrong_observation_is_left_out_of_its_point() {
        let mut control = two_markers();
        let wrong = control
            .observations
            .get_mut("v2.jpg")
            .unwrap()
            .get_mut("a:0")
            .unwrap();
        *wrong += nalgebra::Vector2::new(40.0, -25.0);

        let registration = register_control_points(&reconstruction(), &control).unwrap();

        let report = &registration.report;
        let a0 = report
            .points
            .iter()
            .find(|point| point.point == "a:0")
            .unwrap();
        assert_eq!((a0.observation_count, a0.inlier_count), (VIEWS, VIEWS - 1));
        assert!((report.scale - 50.0).abs() < 1e-6, "{report:?}");
    }

    #[test]
    fn inconsistent_input_is_rejected() {
        let mut duplicate = two_markers();
        let point = duplicate.groups[0]
            .points
            .first_key_value()
            .unwrap()
            .0
            .clone();
        duplicate.groups[1].points.insert(point, Point3::origin());
        let mut unknown = two_markers();
        unknown
            .observations
            .get_mut("v0.jpg")
            .unwrap()
            .insert("nowhere".to_owned(), Point2::origin());
        let mut missing_reference = two_markers();
        missing_reference.reference_group = Some("survey".to_owned());
        let mut too_few = two_markers();
        for group in &mut too_few.groups {
            group
                .points
                .retain(|name, _| name.ends_with(":0") || name.ends_with(":1"));
        }

        for control in [duplicate, unknown, missing_reference, too_few] {
            assert!(register_control_points(&reconstruction(), &control).is_err());
        }
    }

    #[test]
    fn images_outside_the_reconstruction_are_ignored() {
        let mut control = two_markers();
        let seen = control.observations["v0.jpg"].clone();
        control
            .observations
            .insert("not-registered.jpg".to_owned(), seen);

        let report = register_control_points(&reconstruction(), &control)
            .unwrap()
            .report;

        assert!(
            report
                .points
                .iter()
                .all(|point| point.observation_count == VIEWS)
        );
    }
}
