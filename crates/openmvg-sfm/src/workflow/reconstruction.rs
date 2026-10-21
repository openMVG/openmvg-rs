// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Immutable reconstruction and engine output values.

use std::io::Write;
use std::path::Path;

use crate::SfmError;
use crate::model::PoseRecord;
use crate::model::{ImageMeta, Landmarks, Observation, ReconstructionComponent, ViewRecord};
use crate::workflow::run::{RunReport, SfmEngine};
use crate::{Camera, SfmWorkspace, ViewId};

/// Encoding of a PLY point-cloud export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlyFormat {
    /// Text vertices with `float` coordinates and RGB colors.
    Ascii,
    /// Little-endian binary vertices: three `f32` coordinates followed by three RGB bytes. The
    /// default.
    #[default]
    Binary,
}

/// Immutable accepted cameras, poses, landmarks, and reconstruction components.
///
/// Saved with [`checkpoint::write_reconstruction`](crate::checkpoint::write_reconstruction) or
/// [`Self::to_bytes`] and read back with
/// [`checkpoint::read_reconstruction`](crate::checkpoint::read_reconstruction) or
/// [`Self::from_bytes`].
#[derive(Debug, Clone)]
pub struct Reconstruction {
    pub(crate) workspace: SfmWorkspace,
    engine: SfmEngine,
    calibrated_intrinsics: bool,
}

impl Reconstruction {
    pub(crate) const fn new(
        workspace: SfmWorkspace,
        engine: SfmEngine,
        calibrated_intrinsics: bool,
    ) -> Self {
        Self {
            workspace,
            engine,
            calibrated_intrinsics,
        }
    }

    /// Wrap a workspace reconstructed by [`Runner::run_workspace`](crate::Runner::run_workspace),
    /// so tooling on the unstable workspace API can use the reconstruction writers.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn from_workspace(
        workspace: SfmWorkspace,
        engine: SfmEngine,
        calibrated_intrinsics: bool,
    ) -> Self {
        Self::new(workspace, engine, calibrated_intrinsics)
    }

    /// The mutable workspace behind this reconstruction, for the unstable workspace reports.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn workspace(&self) -> &SfmWorkspace {
        &self.workspace
    }

    /// The engine that produced this reconstruction; persisted in its file.
    #[must_use]
    pub const fn engine(&self) -> SfmEngine {
        self.engine
    }

    /// Whether the camera models were a known calibration that the engine kept fixed, rather
    /// than estimates it refined. Inherited from the [`ViewGraph`](crate::ViewGraph) and persisted in the file.
    #[must_use]
    pub const fn has_calibrated_intrinsics(&self) -> bool {
        self.calibrated_intrinsics
    }

    /// Root seed of the run that produced this reconstruction; persisted in its file.
    #[must_use]
    pub const fn root_seed(&self) -> openmvg_core::RunSeed {
        self.workspace.root_seed()
    }

    /// Every input view, registered or not, indexed by [`ViewId`].
    #[must_use]
    pub fn views(&self) -> &[ViewRecord] {
        self.workspace.views()
    }

    /// Number of input views, registered or not.
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.workspace.views().len()
    }

    /// Final camera models, indexed by [`IntrinsicId`](crate::IntrinsicId) (refined unless calibrated).
    #[must_use]
    pub fn camera_models(&self) -> &[Camera] {
        self.workspace.camera_models()
    }

    /// The accepted pose of `view`; `None` if the view is unregistered or unknown.
    #[must_use]
    pub fn pose(&self, view: ViewId) -> Option<&PoseRecord> {
        self.workspace.pose(view)
    }

    /// Number of views with an accepted pose.
    #[must_use]
    pub fn registered_view_count(&self) -> usize {
        self.workspace.registered_view_count()
    }

    /// Registered views in deterministic [`ViewId`] order.
    pub fn registered_views(&self) -> impl Iterator<Item = (ViewId, &PoseRecord)> {
        self.workspace.registered_views()
    }

    /// Images of the registered views in [`ViewId`] order; [`ImageMeta::file_path`] gives the
    /// file of an image-backed view.
    pub fn registered_images(&self) -> impl Iterator<Item = &ImageMeta> {
        self.images_by_registration(true)
    }

    /// Images of the views without an accepted pose in [`ViewId`] order.
    pub fn unregistered_images(&self) -> impl Iterator<Item = &ImageMeta> {
        self.images_by_registration(false)
    }

    fn images_by_registration(&self, registered: bool) -> impl Iterator<Item = &ImageMeta> {
        self.workspace
            .views()
            .iter()
            .filter(move |view| self.workspace.has_pose(view.id) == registered)
            .map(|view| &view.meta)
    }

    /// Every landmark, including rejected ones kept so that landmark IDs stay stable.
    #[must_use]
    pub fn landmarks(&self) -> &Landmarks {
        self.workspace.landmarks()
    }

    /// Number of reconstructed 3D points: the active landmarks, which are the points written
    /// by the scene exports.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.workspace.landmarks().active_count()
    }

    /// Connected, separately gauged parts of the reconstruction.
    #[must_use]
    pub fn components(&self) -> &[ReconstructionComponent] {
        self.workspace.components()
    }

    /// The observations that support each active landmark, ordered by landmark and then by
    /// position in the landmark's track.
    ///
    /// Rejected landmarks yield nothing. Every observation's view is registered.
    pub fn observations(&self) -> impl Iterator<Item = Observation> + '_ {
        let workspace = &self.workspace;
        workspace
            .landmarks()
            .iter_active()
            .flat_map(move |landmark| {
                // Validation guarantees the source track and every observed pixel exist.
                workspace
                    .tracks()
                    .get(landmark.source_track)
                    .into_iter()
                    .flat_map(move |track| landmark.supporting_observations(track))
                    .filter_map(move |(_, observation)| {
                        let pixel = workspace.track_observation_pixel(observation).ok()?;
                        Some(Observation {
                            landmark: landmark.id,
                            view: observation.view,
                            feature: observation.feature,
                            pixel,
                        })
                    })
            })
    }

    /// Write active landmarks and registered camera centers as a binary little-endian PLY.
    ///
    /// Use [`Self::write_ply_with_format`] to select ASCII. Colors, precision, ordering and
    /// failure behavior follow that method.
    pub fn write_ply(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        self.write_ply_with_format(path, PlyFormat::default())
    }

    /// Write active landmarks and registered camera centers in the selected PLY encoding.
    ///
    /// Landmarks are white and camera centers are green. Each registered pose block contributes
    /// one center, so views sharing a pose contribute one vertex. Coordinates are world-frame
    /// positions rounded only to PLY's `float` precision.
    ///
    /// Missing parent directories are created, and `path` is replaced atomically: a reader sees
    /// either the previous file or the complete new one. A reconstruction without landmarks or
    /// registered poses writes a valid file with zero vertices. Fails with [`SfmError::InvalidInput`]
    /// if a coordinate is non-finite or exceeds the `f32` range, before anything is written, and
    /// with [`SfmError::Io`] if a file operation fails.
    pub fn write_ply_with_format(
        &self,
        path: impl AsRef<Path>,
        format: PlyFormat,
    ) -> crate::Result<()> {
        let landmarks = self.workspace.landmarks();
        let poses = self.workspace.pose_store();
        if landmarks
            .iter_active()
            .map(|landmark| &landmark.point)
            .chain(
                poses
                    .iter_populated()
                    .map(|(_, record)| record.pose.center()),
            )
            .flat_map(|point| point.iter())
            .any(|value| !value.is_finite() || value.abs() > f64::from(f32::MAX))
        {
            return Err(SfmError::InvalidInput(
                "PLY coordinates must be finite and representable as f32".to_owned(),
            ));
        }
        let vertex_count = landmarks
            .active_count()
            .checked_add(poses.populated_count())
            .ok_or_else(|| SfmError::InvalidInput("PLY vertex count exceeds usize".to_owned()))?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::atomic::atomic_write_with(path, |writer| -> crate::Result<()> {
            let encoding = match format {
                PlyFormat::Ascii => "ascii",
                PlyFormat::Binary => "binary_little_endian",
            };
            write!(
                writer,
                "ply\nformat {} 1.0\ncomment Generated by OpenMVG-rs v{}\n\
                 element vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
                 property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
                encoding,
                env!("CARGO_PKG_VERSION"),
                vertex_count
            )?;
            let vertices = landmarks
                .iter_active()
                .map(|landmark| (landmark.point, [255_u8; 3]))
                .chain(
                    poses
                        .iter_populated()
                        .map(|(_, record)| (*record.pose.center(), [0, 255, 0])),
                );
            for (point, color) in vertices {
                match format {
                    PlyFormat::Ascii => writeln!(
                        writer,
                        "{} {} {} {} {} {}",
                        point.x as f32,
                        point.y as f32,
                        point.z as f32,
                        color[0],
                        color[1],
                        color[2],
                    )?,
                    PlyFormat::Binary => {
                        let mut vertex = [0_u8; 15];
                        vertex[0..4].copy_from_slice(&(point.x as f32).to_le_bytes());
                        vertex[4..8].copy_from_slice(&(point.y as f32).to_le_bytes());
                        vertex[8..12].copy_from_slice(&(point.z as f32).to_le_bytes());
                        vertex[12..15].copy_from_slice(&color);
                        writer.write_all(&vertex)?;
                    }
                }
            }
            writer.flush()?;
            Ok(())
        })
    }
}

/// Immutable output of one SfM engine execution.
#[derive(Debug, Clone)]
pub struct SfmOutput {
    pub(crate) reconstruction: Reconstruction,
    pub(crate) report: RunReport,
}

impl SfmOutput {
    pub(crate) const fn new(reconstruction: Reconstruction, report: RunReport) -> Self {
        Self {
            reconstruction,
            report,
        }
    }
    /// The accepted reconstruction; empty of poses when the run produced none.
    #[must_use]
    pub const fn reconstruction(&self) -> &Reconstruction {
        &self.reconstruction
    }

    /// The versioned run report, including one verdict per input view.
    #[must_use]
    pub const fn report(&self) -> &RunReport {
        &self.report
    }

    /// Split the output into its reconstruction and report.
    #[must_use]
    pub fn into_parts(self) -> (Reconstruction, RunReport) {
        (self.reconstruction, self.report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ObservationSupport;
    use crate::model::{Landmark, LandmarkStatus};
    use crate::{
        FeatureId, FeatureSet, IntrinsicId, LandmarkId, Track, TrackId, TrackObservation, TrackSet,
    };
    use nalgebra::{Point2, Point3, Rotation3};
    use openmvg_core::model::WorldToCameraPose;

    fn names<'a>(images: impl Iterator<Item = &'a ImageMeta>) -> Vec<&'a str> {
        images.map(|image| image.name.as_str()).collect()
    }

    #[test]
    fn reconstruction_projects_counts_and_registered_images_in_view_order() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([
                ImageMeta::path("scene/a.jpg", 64, 48),
                ImageMeta::path("scene/b.jpg", 64, 48),
                ImageMeta::external_id("c", 64, 48),
                ImageMeta::path("scene/d.jpg", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(50.0, 50.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        for view in [3, 0, 2] {
            let center = Point3::new(view as f64, 0.0, 0.0);
            let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center);
            workspace
                .set_pose_for_view(ViewId::new(view), pose.unwrap(), None)
                .unwrap();
        }
        let landmark = |id: usize, z: f64| {
            Landmark::active(
                LandmarkId::new(id),
                TrackId::new(id),
                Point3::new(0.0, 0.0, z),
                openmvg_core::PixelResidual::try_new(0.5).unwrap(),
                ObservationSupport::default(),
            )
        };
        let mut rejected = landmark(1, 6.0);
        rejected.status = LandmarkStatus::Rejected;
        workspace.landmarks =
            Landmarks::try_new(vec![landmark(0, 5.0), rejected, landmark(2, 7.0)]).unwrap();

        let reconstruction = Reconstruction::new(workspace, SfmEngine::Incremental, false);

        assert_eq!(reconstruction.view_count(), 4);
        assert_eq!(reconstruction.registered_view_count(), 3);
        assert_eq!(reconstruction.point_count(), 2);
        assert_eq!(reconstruction.landmarks().len(), 3);
        assert_eq!(
            names(reconstruction.registered_images()),
            ["a.jpg", "c", "d.jpg"]
        );
        assert_eq!(
            reconstruction
                .registered_images()
                .map(ImageMeta::file_path)
                .collect::<Vec<_>>(),
            [
                Some(std::path::Path::new("scene/a.jpg")),
                None,
                Some(std::path::Path::new("scene/d.jpg")),
            ]
        );
        assert_eq!(names(reconstruction.unregistered_images()), ["b.jpg"]);
    }

    #[test]
    fn observations_list_the_support_of_active_landmarks() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta((0..3).map(|view| ImageMeta::external_id(format!("{view}"), 64, 48)))
            .camera_models([Camera::try_pinhole(50.0, 50.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        for view in [0, 2] {
            let center = Point3::new(view as f64, 0.0, 0.0);
            let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center);
            workspace
                .set_pose_for_view(ViewId::new(view), pose.unwrap(), None)
                .unwrap();
        }
        let pixel = |view: usize, feature: usize| Point2::new(view as f64, feature as f64);
        for view in 0..3 {
            let pixels = (0..2).map(|feature| pixel(view, feature)).collect();
            workspace
                .feature_sets
                .set(view, FeatureSet::try_new(pixels).unwrap())
                .unwrap();
        }
        let track = |id: usize| {
            let observations = (0..3)
                .map(|view| TrackObservation {
                    view: ViewId::new(view),
                    feature: FeatureId::new(id),
                })
                .collect();
            Track::try_new(TrackId::new(id), observations).unwrap()
        };
        workspace.tracks = TrackSet::try_new(vec![track(0), track(1)]).unwrap();
        let landmark = |id: usize| {
            Landmark::active(
                LandmarkId::new(id),
                TrackId::new(id),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.5).unwrap(),
                // The unregistered view 1 does not support the landmark.
                ObservationSupport::try_from_indices([0, 2], 3).unwrap(),
            )
        };
        let mut rejected = landmark(1);
        rejected.status = LandmarkStatus::Rejected;
        workspace.landmarks = Landmarks::try_new(vec![landmark(0), rejected]).unwrap();

        let reconstruction = Reconstruction::new(workspace, SfmEngine::Incremental, false);
        let observation = |view: usize| Observation {
            landmark: LandmarkId::new(0),
            view: ViewId::new(view),
            feature: FeatureId::new(0),
            pixel: pixel(view, 0),
        };
        assert_eq!(
            reconstruction.observations().collect::<Vec<_>>(),
            [observation(0), observation(2)]
        );
    }

    /// One registered view and an active landmark at each `x`, with a rejected one in between.
    fn point_cloud(xs: &[f64]) -> Reconstruction {
        let mut workspace = SfmWorkspace::builder()
            .image_meta([ImageMeta::external_id("0", 64, 48)])
            .camera_models([Camera::try_pinhole(50.0, 50.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        workspace
            .set_pose_for_view(ViewId::new(0), WorldToCameraPose::identity(), None)
            .unwrap();
        let landmark = |id: usize, x: f64, status: LandmarkStatus| {
            let mut landmark = Landmark::active(
                LandmarkId::new(id),
                TrackId::new(id),
                Point3::new(x, -0.25, 5.0),
                openmvg_core::PixelResidual::try_new(0.5).unwrap(),
                ObservationSupport::default(),
            );
            landmark.status = status;
            landmark
        };
        let mut landmarks = vec![landmark(0, 9.0, LandmarkStatus::Rejected)];
        for (index, &x) in xs.iter().enumerate() {
            landmarks.push(landmark(index + 1, x, LandmarkStatus::Active));
        }
        workspace.landmarks = Landmarks::try_new(landmarks).unwrap();
        Reconstruction::new(workspace, SfmEngine::Incremental, false)
    }

    type PlyVertex = ([f32; 3], [u8; 3]);

    /// Decode the file format independently of the writer's vertex code.
    fn read_ply(path: &Path) -> (PlyFormat, Vec<PlyVertex>) {
        let bytes = std::fs::read(path).unwrap();
        let marker = b"end_header\n";
        let boundary = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap()
            + marker.len();
        let header = std::str::from_utf8(&bytes[..boundary]).unwrap();
        let mut lines = header.lines();
        assert_eq!(lines.next(), Some("ply"));
        let format = match lines.next().unwrap() {
            "format ascii 1.0" => PlyFormat::Ascii,
            "format binary_little_endian 1.0" => PlyFormat::Binary,
            other => panic!("unexpected PLY format: {other}"),
        };
        assert_eq!(
            lines.next().unwrap(),
            format!(
                "comment Generated by OpenMVG-rs v{}",
                env!("CARGO_PKG_VERSION")
            )
        );
        let count: usize = header
            .lines()
            .find_map(|line| line.strip_prefix("element vertex "))
            .unwrap()
            .parse()
            .unwrap();
        let properties: Vec<_> = header
            .lines()
            .filter(|line| line.starts_with("property "))
            .collect();
        assert_eq!(
            properties,
            [
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue"
            ]
        );
        let body = &bytes[boundary..];
        let vertices: Vec<PlyVertex> = match format {
            PlyFormat::Ascii => std::str::from_utf8(body)
                .unwrap()
                .lines()
                .map(|line| {
                    let fields: Vec<_> = line.split_whitespace().collect();
                    assert_eq!(fields.len(), 6);
                    (
                        std::array::from_fn(|axis| fields[axis].parse().unwrap()),
                        std::array::from_fn(|channel| fields[channel + 3].parse().unwrap()),
                    )
                })
                .collect(),
            PlyFormat::Binary => {
                assert_eq!(body.len(), count * 15);
                body.as_chunks::<15>()
                    .0
                    .iter()
                    .map(|vertex| {
                        let position = std::array::from_fn(|axis| {
                            let start = axis * 4;
                            f32::from_le_bytes([
                                vertex[start],
                                vertex[start + 1],
                                vertex[start + 2],
                                vertex[start + 3],
                            ])
                        });
                        (position, [vertex[12], vertex[13], vertex[14]])
                    })
                    .collect()
            }
        };
        assert_eq!(vertices.len(), count);
        (format, vertices)
    }

    #[test]
    fn ply_lists_active_landmarks_and_green_camera_centers() {
        let mut reconstruction = point_cloud(&[1.0, 1.0 / 3.0]);
        let center = Point3::new(1.25, -2.5, 0.75);
        let pose = WorldToCameraPose::from_rotation_center(
            Rotation3::from_euler_angles(0.2, -0.3, 0.4),
            center,
        )
        .unwrap();
        reconstruction
            .workspace
            .set_pose_for_view(ViewId::new(0), pose, None)
            .unwrap();
        for format in [PlyFormat::Ascii, PlyFormat::Binary] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("nested/points.ply");
            reconstruction.write_ply_with_format(&path, format).unwrap();
            let (decoded_format, vertices) = read_ply(&path);
            assert_eq!(decoded_format, format);
            assert_eq!(
                vertices,
                [
                    ([1.0, -0.25, 5.0], [255; 3]),
                    ([(1.0 / 3.0) as f32, -0.25, 5.0], [255; 3]),
                    ([1.25, -2.5, 0.75], [0, 255, 0]),
                ]
            );
            assert_eq!(
                vertices
                    .iter()
                    .filter(|(_, color)| *color == [0, 255, 0])
                    .count(),
                1
            );
            point_cloud(&[])
                .write_ply_with_format(&path, format)
                .unwrap();
            assert_eq!(read_ply(&path), (format, vec![([0.0; 3], [0, 255, 0])]));
            let empty = SfmWorkspace::builder()
                .image_meta([ImageMeta::external_id("0", 64, 48)])
                .camera_models([Camera::try_pinhole(50.0, 50.0, 32.0, 24.0).unwrap()])
                .shared_intrinsic(IntrinsicId::new(0))
                .build()
                .unwrap();
            Reconstruction::new(empty, SfmEngine::Incremental, false)
                .write_ply_with_format(&path, format)
                .unwrap();
            assert_eq!(read_ply(&path), (format, Vec::new()));
            let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            assert_eq!(names, ["points.ply"]);
        }
        let dir = tempfile::tempdir().unwrap();
        let default_path = dir.path().join("default.ply");
        reconstruction.write_ply(&default_path).unwrap();
        assert_eq!(PlyFormat::default(), PlyFormat::Binary);
        assert_eq!(read_ply(&default_path).0, PlyFormat::Binary);
    }

    #[test]
    fn ply_rejects_a_coordinate_outside_f32_before_writing() {
        let mut reconstruction = point_cloud(&[]);
        let pose = WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(f64::from(f32::MAX) * 2.0, 0.0, 0.0),
        )
        .unwrap();
        reconstruction
            .workspace
            .set_pose_for_view(ViewId::new(0), pose, None)
            .unwrap();
        for format in [PlyFormat::Ascii, PlyFormat::Binary] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("nested/points.ply");
            let error = point_cloud(&[f64::from(f32::MAX) * 2.0])
                .write_ply_with_format(&path, format)
                .unwrap_err();
            assert!(matches!(error, crate::SfmError::InvalidInput(_)));
            assert!(!path.parent().unwrap().exists());
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"retained file").unwrap();
            assert!(matches!(
                reconstruction.write_ply_with_format(&path, format),
                Err(crate::SfmError::InvalidInput(_))
            ));
            assert_eq!(std::fs::read(&path).unwrap(), b"retained file");
        }
    }

    #[test]
    fn ply_io_refusal_keeps_destination_and_removes_temporary_file() {
        for format in [PlyFormat::Ascii, PlyFormat::Binary] {
            let dir = tempfile::tempdir().unwrap();
            let target = dir.path().join("destination");
            std::fs::create_dir(&target).unwrap();
            std::fs::write(target.join("retained"), b"old data").unwrap();
            assert!(matches!(
                point_cloud(&[1.0]).write_ply_with_format(&target, format),
                Err(crate::SfmError::Io(_))
            ));
            assert_eq!(std::fs::read(target.join("retained")).unwrap(), b"old data");
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }
}
