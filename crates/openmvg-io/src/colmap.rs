// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! COLMAP text export of a reconstruction. Its conventions are documented on [`write_colmap`],
//! where docs.rs shows them.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, ErrorKind, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{Point2, Point3, UnitQuaternion};
use openmvg_sfm::{Camera, IntrinsicId, Reconstruction, ViewId};

use crate::error::{Error, Result};

/// Offset from this library's pixel coordinates to COLMAP's, in pixels.
const PIXEL_OFFSET: f64 = 0.5;

/// A COLMAP camera model and its parameters, in COLMAP's order.
#[derive(Debug, PartialEq)]
struct ColmapCamera {
    model: &'static str,
    parameters: Vec<f64>,
}

/// One camera model of the reconstruction as a COLMAP camera.
struct CameraLine {
    /// The first registered view that uses it, for error messages.
    first_view: ViewId,
    width: u32,
    height: u32,
    camera: ColmapCamera,
}

/// One registered view as a COLMAP image.
struct Image<'a> {
    view: ViewId,
    id: u32,
    camera_id: u32,
    name: &'a str,
    /// `qw`, `qx`, `qy`, `qz` of the world-to-camera rotation.
    rotation: [f64; 4],
    /// `t = -R C`.
    translation: [f64; 3],
    /// The observed pixels and their point IDs; `POINT2D_IDX` indexes this list.
    observations: Vec<(Point2<f64>, u64)>,
}

/// One active landmark as a COLMAP point.
struct Point {
    id: u64,
    position: Point3<f64>,
    error: f64,
    /// `IMAGE_ID` and `POINT2D_IDX` of each observation.
    track: Vec<(u32, usize)>,
}

/// Write `reconstruction` as a COLMAP text model in `path/sparse/0`.
///
/// ```text
/// Reconstruction ──> map every camera and ID ──> <path>/sparse/0/
///             (a mapping error creates nothing)    cameras.txt    CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
///                                                  images.txt     IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
///                                                                 POINTS2D[] as (X, Y, POINT3D_ID)
///                                                  points3D.txt   POINT3D_ID X Y Z R G B ERROR
///                                                                 TRACK[] as (IMAGE_ID, POINT2D_IDX)
/// ```
///
/// # Conventions
///
/// - Only registered views are written, each as one COLMAP image numbered by its view index plus
///   one. An image's `NAME` is the view's image name: the file name of an image file, or the
///   path below the image root for a reconstruction read back with one.
/// - Each camera model that a registered view uses becomes one COLMAP camera, numbered by its
///   [`IntrinsicId`] plus one, so views that share a camera model share a COLMAP camera, as
///   COLMAP's own reconstructions do. The views of one camera model have one image size, which
///   `openmvg_sfm::Dataset` checks, and that size is the camera's `WIDTH` and `HEIGHT`.
/// - Poses are world to camera, as COLMAP stores them: `QW QX QY QZ` is the rotation `R` and
///   `TX TY TZ` is `t = −R·C`, for the camera center `C`.
/// - Each active point is numbered by its landmark index plus one, so rejected points leave gaps
///   in the numbering. The reconstruction holds no color, so points are written gray,
///   `128 128 128`, and `ERROR` is the point's mean reprojection error in pixels, over its
///   observations.
/// - An image lists the points it observes in track order, and `POINT2D_IDX` indexes that list.
/// - COLMAP puts the center of the top-left pixel at `(0.5, 0.5)`, and this library at
///   `(0, 0)`, so 0.5 is added to every principal point and observation coordinate.
/// - Points are numbered from their landmark indices, so a reconstruction read back from an
///   OMVG dataset, whose points are renumbered without gaps, gets other point IDs than the
///   original.
/// - Values are written in Rust's shortest round-trip form, so they parse back to the same
///   `f64`.
///
/// Each camera takes the smallest COLMAP model that holds it without loss:
///
/// ```text
/// Pinhole                                 SIMPLE_PINHOLE (fx = fy) or PINHOLE
/// Brown-Conrady, no distortion            SIMPLE_PINHOLE or PINHOLE
///   k1 only, fx = fy                      SIMPLE_RADIAL
///   k1 k2 only, fx = fy                   RADIAL
///   no k3                                 OPENCV
///   k3                                    FULL_OPENCV (k4..k6 = 0)
/// Fisheye                                 OPENCV_FISHEYE
/// Spherical                               unsupported
/// ```
///
/// # Files
///
/// Missing directories are created, and `cameras.txt`, `images.txt`, and `points3D.txt` are
/// each replaced atomically: a reader sees either the previous file or the complete new one.
/// The three files are replaced one after the other, not as one snapshot, so an error after
/// the first replacement leaves a model that mixes new and previous files. Other files in the
/// directory are left alone.
///
/// # Errors
///
/// - [`Error::Unsupported`] if a registered view uses a spherical camera, which COLMAP cannot
///   model, an ID exceeds COLMAP's range, an image name contains a line break or NUL, or the
///   views of one camera model have different image sizes; nothing is created then.
/// - [`Error::Io`] if a file operation fails; the file being written keeps its previous content.
/// - [`Error::Unsynced`] if a file was replaced but its directory could not be flushed.
pub fn write_colmap(reconstruction: &Reconstruction, path: impl AsRef<Path>) -> Result<()> {
    let (cameras, images, points) = colmap_model(reconstruction)?;
    let dir = path.as_ref().join("sparse").join("0");
    fs::create_dir_all(&dir).map_err(Error::Io)?;
    replace(&dir, "cameras.txt", |writer| {
        write_cameras(writer, &cameras)
    })?;
    replace(&dir, "images.txt", |writer| write_images(writer, &images))?;
    replace(&dir, "points3D.txt", |writer| write_points(writer, &points))
}

/// The cameras the registered views use, keyed by COLMAP camera ID; the registered views as
/// images; and the active landmarks as points; each in the order written.
type ColmapModel<'a> = (BTreeMap<u32, CameraLine>, Vec<Image<'a>>, Vec<Point>);

/// The camera models, registered views and active landmarks as COLMAP cameras, images and
/// points.
fn colmap_model(reconstruction: &Reconstruction) -> Result<ColmapModel<'_>> {
    let mut cameras = BTreeMap::new();
    let mut images = reconstruction
        .registered_views()
        .map(|(view, record)| {
            let meta = reconstruction
                .views()
                .get(view.index())
                .ok_or_else(|| Error::Unsupported(format!("view {view} has no view record")))?;
            let camera = reconstruction
                .camera_models()
                .get(meta.intrinsic_id.index())
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "view {view} uses camera {}, which does not exist",
                        meta.intrinsic_id
                    ))
                })?;
            if meta.meta.name.contains(['\n', '\r', '\0']) {
                return Err(Error::Unsupported(format!(
                    "view {view} has an image name containing a line break or NUL, which COLMAP cannot represent"
                )));
            }
            let camera_id = camera_id(meta.intrinsic_id)?;
            let (width, height) = (meta.meta.width, meta.meta.height);
            match cameras.entry(camera_id) {
                Entry::Vacant(entry) => {
                    entry.insert(CameraLine {
                        first_view: view,
                        width,
                        height,
                        camera: colmap_camera(camera).map_err(|reason| {
                            Error::Unsupported(format!("view {view} cannot be exported: {reason}"))
                        })?,
                    });
                }
                Entry::Occupied(entry) => {
                    // The dataset already gives the views of one camera model one size.
                    let line = entry.get();
                    if (line.width, line.height) != (width, height) {
                        return Err(Error::Unsupported(format!(
                            "views {} and {view} share camera {} but are {}x{} and {width}x{height} \
                             pixels, and a COLMAP camera has one image size",
                            line.first_view, meta.intrinsic_id, line.width, line.height
                        )));
                    }
                }
            }
            let rotation = UnitQuaternion::from_rotation_matrix(record.pose.rotation());
            Ok(Image {
                view,
                id: image_id(view)?,
                camera_id,
                name: &meta.meta.name,
                rotation: [rotation.w, rotation.i, rotation.j, rotation.k],
                translation: record.pose.translation().into(),
                observations: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let slots: BTreeMap<ViewId, usize> = images
        .iter()
        .enumerate()
        .map(|(slot, image)| (image.view, slot))
        .collect();
    let mut observations = reconstruction.observations().peekable();
    let mut points = Vec::with_capacity(reconstruction.point_count());
    for landmark in reconstruction.landmarks().iter_active() {
        let id = point_id(landmark.id.index())?;
        let mut track = Vec::new();
        // Observations come grouped by landmark, in the order of the active landmarks.
        while let Some(observation) =
            observations.next_if(|observation| observation.landmark == landmark.id)
        {
            let image = slots
                .get(&observation.view)
                .and_then(|&slot| images.get_mut(slot))
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "landmark {} is observed in view {}, which is not registered",
                        landmark.id, observation.view
                    ))
                })?;
            track.push((image.id, image.observations.len()));
            image.observations.push((observation.pixel, id));
        }
        points.push(Point {
            id,
            position: landmark.point,
            error: landmark.reprojection_error.value(),
            track,
        });
    }
    Ok((cameras, images, points))
}

/// The COLMAP image ID of `view`.
fn image_id(view: ViewId) -> Result<u32> {
    view.index()
        .checked_add(1)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| Error::Unsupported(format!("view {view} exceeds COLMAP's 32-bit image IDs")))
}

/// The COLMAP camera ID of the camera model `intrinsic`.
fn camera_id(intrinsic: IntrinsicId) -> Result<u32> {
    intrinsic
        .index()
        .checked_add(1)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "camera {intrinsic} exceeds COLMAP's 32-bit camera IDs"
            ))
        })
}

/// The COLMAP point ID of the landmark at `index`.
fn point_id(index: usize) -> Result<u64> {
    u64::try_from(index)
        .ok()
        .and_then(|id| id.checked_add(1))
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "landmark {index} exceeds COLMAP's 64-bit point IDs"
            ))
        })
}

/// The smallest COLMAP model holding `camera` without loss, or why there is none.
fn colmap_camera(camera: &Camera) -> std::result::Result<ColmapCamera, &'static str> {
    let pinhole = |fx: f64, fy: f64, cx: f64, cy: f64| {
        let (cx, cy) = (cx + PIXEL_OFFSET, cy + PIXEL_OFFSET);
        if fx == fy {
            ColmapCamera {
                model: "SIMPLE_PINHOLE",
                parameters: vec![fx, cx, cy],
            }
        } else {
            ColmapCamera {
                model: "PINHOLE",
                parameters: vec![fx, fy, cx, cy],
            }
        }
    };
    let camera = match camera {
        Camera::Pinhole(camera) => pinhole(
            camera.focal_x(),
            camera.focal_y(),
            camera.principal_x(),
            camera.principal_y(),
        ),
        Camera::BrownConrady(camera) => {
            let (fx, fy) = (camera.focal_x(), camera.focal_y());
            let (cx, cy) = (camera.principal_x(), camera.principal_y());
            let [k1, k2, k3, p1, p2] = if camera.has_distortion() {
                let (radial, tangential) =
                    (camera.radial_distortion(), camera.tangential_distortion());
                [
                    radial.k1(),
                    radial.k2(),
                    radial.k3(),
                    tangential.p1(),
                    tangential.p2(),
                ]
            } else {
                [0.0; 5]
            };
            let zero = |values: &[f64]| values.iter().all(|value| *value == 0.0);
            if zero(&[k1, k2, k3, p1, p2]) {
                return Ok(pinhole(fx, fy, cx, cy));
            }
            let (cx, cy) = (cx + PIXEL_OFFSET, cy + PIXEL_OFFSET);
            if fx == fy && zero(&[k2, k3, p1, p2]) {
                ColmapCamera {
                    model: "SIMPLE_RADIAL",
                    parameters: vec![fx, cx, cy, k1],
                }
            } else if fx == fy && zero(&[k3, p1, p2]) {
                ColmapCamera {
                    model: "RADIAL",
                    parameters: vec![fx, cx, cy, k1, k2],
                }
            } else if k3 == 0.0 {
                ColmapCamera {
                    model: "OPENCV",
                    parameters: vec![fx, fy, cx, cy, k1, k2, p1, p2],
                }
            } else {
                ColmapCamera {
                    model: "FULL_OPENCV",
                    parameters: vec![fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0],
                }
            }
        }
        Camera::Fisheye(camera) => {
            let distortion = camera.fisheye_distortion();
            let [k1, k2, k3, k4] = if camera.has_distortion() {
                [
                    distortion.k1(),
                    distortion.k2(),
                    distortion.k3(),
                    distortion.k4(),
                ]
            } else {
                [0.0; 4]
            };
            ColmapCamera {
                model: "OPENCV_FISHEYE",
                parameters: vec![
                    camera.focal_x(),
                    camera.focal_y(),
                    camera.principal_x() + PIXEL_OFFSET,
                    camera.principal_y() + PIXEL_OFFSET,
                    k1,
                    k2,
                    k3,
                    k4,
                ],
            }
        }
        Camera::Spherical(_) => {
            return Err("COLMAP has no equirectangular spherical camera model");
        }
    };
    Ok(camera)
}

fn write_cameras(writer: &mut impl Write, cameras: &BTreeMap<u32, CameraLine>) -> io::Result<()> {
    writeln!(writer, "# Camera list with one line of data per camera:")?;
    writeln!(writer, "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")?;
    writeln!(writer, "# Number of cameras: {}", cameras.len())?;
    for (id, line) in cameras {
        let ColmapCamera { model, parameters } = &line.camera;
        write!(writer, "{id} {model} {} {}", line.width, line.height)?;
        for parameter in parameters {
            write!(writer, " {parameter}")?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

fn write_images(writer: &mut impl Write, images: &[Image<'_>]) -> io::Result<()> {
    writeln!(writer, "# Image list with two lines of data per image:")?;
    writeln!(
        writer,
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"
    )?;
    writeln!(writer, "#   POINTS2D[] as (X, Y, POINT3D_ID)")?;
    writeln!(writer, "# Number of images: {}", images.len())?;
    for image in images {
        let [qw, qx, qy, qz] = image.rotation;
        let [tx, ty, tz] = image.translation;
        let (id, camera_id) = (image.id, image.camera_id);
        writeln!(
            writer,
            "{id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {}",
            image.name
        )?;
        for (index, (pixel, point)) in image.observations.iter().enumerate() {
            let separator = if index == 0 { "" } else { " " };
            let (x, y) = (pixel.x + PIXEL_OFFSET, pixel.y + PIXEL_OFFSET);
            write!(writer, "{separator}{x} {y} {point}")?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

fn write_points(writer: &mut impl Write, points: &[Point]) -> io::Result<()> {
    writeln!(writer, "# 3D point list with one line of data per point:")?;
    writeln!(
        writer,
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
    )?;
    writeln!(writer, "# Number of points: {}", points.len())?;
    for point in points {
        let (id, error) = (point.id, point.error);
        let (x, y, z) = (point.position.x, point.position.y, point.position.z);
        write!(writer, "{id} {x} {y} {z} 128 128 128 {error}")?;
        // The track follows one separator, even when it is empty.
        write!(writer, " ")?;
        for (index, (image, observation)) in point.track.iter().enumerate() {
            let separator = if index == 0 { "" } else { " " };
            write!(writer, "{separator}{image} {observation}")?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

/// Replace `dir/name` atomically with the bytes `write` produces: they go to a uniquely named
/// sibling, which is synced and renamed over the target. A failure before the rename removes the
/// sibling and leaves the target untouched; a failure to flush the directory after it is
/// [`Error::Unsynced`].
fn replace(
    dir: &Path,
    name: &str,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (temporary, file) = loop {
        let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
        let candidate = dir.join(format!(".{name}.{}.{nonce}.tmp", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Error::Io(error)),
        }
    };
    let result = (|| -> io::Result<()> {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer
            .into_inner()
            .map_err(|error| error.into_error())?
            .sync_all()?;
        fs::rename(&temporary, dir.join(name))
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(Error::Io(error));
    }
    #[cfg(unix)]
    File::open(dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| Error::Unsynced {
            path: dir.join(name),
            source,
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nalgebra::{Quaternion, Rotation3, Vector3};
    use openmvg_core::camera::{
        CameraModel, FisheyeCamera, FisheyeDistortion, PinholeCamera, RadialDistortion,
        SphericalCamera, TangentialDistortion,
    };

    use super::*;
    use crate::fixtures::{engine_graph, reconstruct};

    /// The data lines of `file` in `root/sparse/0`, split into fields; an empty line stays.
    fn data_lines(root: &Path, file: &str) -> Vec<Vec<f64>> {
        fs::read_to_string(root.join("sparse/0").join(file))
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                line.split_whitespace()
                    .map(|field| field.parse().unwrap_or(f64::NAN))
                    .collect()
            })
            .collect()
    }

    fn brown(fx: f64, radial: RadialDistortion, tangential: TangentialDistortion) -> Camera {
        Camera::try_brown_conrady(fx, 500.0, 320.0, 240.0, radial, tangential).unwrap()
    }

    fn colmap(camera: &Camera) -> (&'static str, Vec<f64>) {
        let ColmapCamera { model, parameters } = colmap_camera(camera).unwrap();
        (model, parameters)
    }

    #[test]
    fn cameras_take_the_smallest_colmap_model_that_holds_them() {
        use RadialDistortion::{K1, K1K2, K1K2K3};
        use TangentialDistortion::P1P2;
        let none = TangentialDistortion::None;
        let pinhole = PinholeCamera::try_new(500.0, 510.0, 320.0, 240.0).unwrap();
        let cases = [
            (
                Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap(),
                "SIMPLE_PINHOLE",
                vec![500.0, 320.5, 240.5],
            ),
            (
                Camera::Pinhole(pinhole.clone()),
                "PINHOLE",
                vec![500.0, 510.0, 320.5, 240.5],
            ),
            (
                brown(500.0, RadialDistortion::None, none),
                "SIMPLE_PINHOLE",
                vec![500.0, 320.5, 240.5],
            ),
            (
                brown(500.0, K1(0.1), none),
                "SIMPLE_RADIAL",
                vec![500.0, 320.5, 240.5, 0.1],
            ),
            (
                brown(500.0, K1K2(0.1, -0.01), none),
                "RADIAL",
                vec![500.0, 320.5, 240.5, 0.1, -0.01],
            ),
            (
                brown(490.0, K1(0.1), none),
                "OPENCV",
                vec![490.0, 500.0, 320.5, 240.5, 0.1, 0.0, 0.0, 0.0],
            ),
            (
                brown(500.0, K1K2(0.1, -0.01), P1P2(0.002, -0.003)),
                "OPENCV",
                vec![500.0, 500.0, 320.5, 240.5, 0.1, -0.01, 0.002, -0.003],
            ),
            (
                brown(500.0, K1K2K3(0.1, -0.01, 0.001), P1P2(0.002, -0.003)),
                "FULL_OPENCV",
                vec![
                    500.0, 500.0, 320.5, 240.5, 0.1, -0.01, 0.002, -0.003, 0.001, 0.0, 0.0, 0.0,
                ],
            ),
            (
                Camera::Fisheye(
                    FisheyeCamera::try_new(
                        pinhole.clone(),
                        FisheyeDistortion::K1K2K3K4(0.1, -0.01, 0.001, -0.0001),
                    )
                    .unwrap(),
                ),
                "OPENCV_FISHEYE",
                vec![500.0, 510.0, 320.5, 240.5, 0.1, -0.01, 0.001, -0.0001],
            ),
            (
                Camera::Fisheye(
                    FisheyeCamera::try_new(pinhole, FisheyeDistortion::K1(0.1)).unwrap(),
                ),
                "OPENCV_FISHEYE",
                vec![500.0, 510.0, 320.5, 240.5, 0.1, 0.0, 0.0, 0.0],
            ),
        ];
        for (camera, model, parameters) in cases {
            assert_eq!(colmap(&camera), (model, parameters), "{camera:?}");
        }
        let spherical = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
        assert!(colmap_camera(&spherical).is_err());
    }

    #[test]
    fn unrepresentable_image_names_fail_before_creating_output() {
        use openmvg_sfm::model::{ImageMeta, ViewRecord};
        use openmvg_sfm::{
            IntrinsicId, PoseId, ReconstructionParts, RunSeed, SfmEngine, WorldToCameraPose,
        };
        for name in [
            "first\nsecond.jpg",
            "first\rsecond.jpg",
            "first\0second.jpg",
        ] {
            let views = vec![ViewRecord::new(
                ViewId::new(0),
                ImageMeta::external_id(name, 64, 48),
                IntrinsicId::new(0),
                PoseId::new(0),
            )];
            let reconstruction = Reconstruction::try_from_parts(
                ReconstructionParts::new(
                    RunSeed::from_u64(7),
                    SfmEngine::Incremental,
                    vec![Camera::try_pinhole(500.0, 500.0, 32.0, 24.0).unwrap()],
                    views,
                )
                .with_pose(PoseId::new(0), WorldToCameraPose::identity()),
            )
            .unwrap();
            let scratch = tempfile::tempdir().unwrap();
            assert!(matches!(
                write_colmap(&reconstruction, scratch.path()),
                Err(Error::Unsupported(_))
            ));
            assert!(!scratch.path().join("sparse").exists());
        }
    }

    #[test]
    fn ids_beyond_the_colmap_range_are_unsupported() {
        let last = u32::MAX as usize - 1;
        assert_eq!(image_id(ViewId::new(last)).unwrap(), u32::MAX);
        assert!(matches!(
            image_id(ViewId::new(last + 1)),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(point_id(usize::MAX - 1).unwrap(), usize::MAX as u64);
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(point_id(usize::MAX), Err(Error::Unsupported(_))));
    }

    /// Every written value parses back to the reconstruction's own, and the written model
    /// projects each point where the reconstruction does, half a pixel further.
    #[test]
    fn the_written_model_is_the_reconstruction_in_colmap_conventions() {
        let reconstruction = reconstruct(&engine_graph(), "incremental");
        let scratch = tempfile::tempdir().unwrap();
        write_colmap(&reconstruction, scratch.path()).unwrap();

        let cameras = data_lines(scratch.path(), "cameras.txt");
        let images = data_lines(scratch.path(), "images.txt");
        let points = data_lines(scratch.path(), "points3D.txt");
        let registered: Vec<_> = reconstruction.registered_views().collect();
        // One COLMAP camera per camera model that a registered view uses.
        let groups: std::collections::BTreeSet<_> = registered
            .iter()
            .map(|(view, _)| reconstruction.views()[view.index()].intrinsic_id)
            .collect();
        assert_eq!(cameras.len(), groups.len());
        let cameras: BTreeMap<u32, &Vec<f64>> =
            cameras.iter().map(|line| (line[0] as u32, line)).collect();
        assert_eq!(images.len(), 2 * registered.len());
        assert_eq!(points.len(), reconstruction.point_count());

        // Image ID -> (rotation, translation, f, cx, cy, POINTS2D[]).
        let mut models = BTreeMap::new();
        for ((view, record), image) in registered.iter().zip(images.chunks(2)) {
            let id = (view.index() + 1) as f64;
            let intrinsic = reconstruction.views()[view.index()].intrinsic_id;
            let camera = cameras[&(intrinsic.index() as u32 + 1)];
            assert_eq!(
                (image[0][0], image[0][8]),
                (id, (intrinsic.index() + 1) as f64)
            );
            let rotation = UnitQuaternion::from_rotation_matrix(record.pose.rotation());
            let written = Quaternion::new(image[0][1], image[0][2], image[0][3], image[0][4]);
            assert_eq!(written, *rotation.quaternion());
            let translation = Vector3::new(image[0][5], image[0][6], image[0][7]);
            assert_eq!(translation, record.pose.translation());
            // The fixture's pinhole camera has equal focal lengths: SIMPLE_PINHOLE f cx cy.
            let observations: Vec<_> = image[1].chunks(3).map(<[f64]>::to_vec).collect();
            models.insert(
                image[0][0] as u32,
                (
                    rotation,
                    translation,
                    camera[4],
                    camera[5],
                    camera[6],
                    observations,
                ),
            );
        }

        let pixels: BTreeMap<_, _> = reconstruction
            .observations()
            .map(|observation| {
                let key = (
                    observation.landmark.index() + 1,
                    observation.view.index() + 1,
                );
                (key, observation.pixel)
            })
            .collect();
        let mut observed = 0;
        for (landmark, point) in reconstruction.landmarks().iter_active().zip(&points) {
            let id = landmark.id.index() + 1;
            assert_eq!(point[0], id as f64);
            assert_eq!(&point[1..4], landmark.point.coords.as_slice());
            assert_eq!(&point[4..7], [128.0; 3]);
            assert_eq!(point[7], landmark.reprojection_error.value());
            for entry in point[8..].chunks(2) {
                let (image, index) = (entry[0] as u32, entry[1] as usize);
                let (rotation, translation, f, cx, cy, observations) = &models[&image];
                let written = &observations[index];
                assert_eq!(
                    written[2], id as f64,
                    "POINT2D_IDX addresses the image's list"
                );
                let pixel = pixels[&(id, image as usize)];
                assert_eq!(written[..2], [pixel.x + 0.5, pixel.y + 0.5]);

                let view = ViewId::new(image as usize - 1);
                let camera = &reconstruction.camera_models()[0];
                let pose = &reconstruction.pose(view).unwrap().pose;
                let ours = camera
                    .project(&pose.project_to_camera(&landmark.point).coords)
                    .unwrap();
                let local = rotation * landmark.point.coords + translation;
                let theirs = [f * local.x / local.z + cx, f * local.y / local.z + cy];
                assert!(
                    (theirs[0] - (ours.x + 0.5)).abs() < 1e-9,
                    "{theirs:?} vs {ours}"
                );
                assert!(
                    (theirs[1] - (ours.y + 0.5)).abs() < 1e-9,
                    "{theirs:?} vs {ours}"
                );
                observed += 1;
            }
        }
        assert_eq!(observed, reconstruction.observations().count());
        let listed: usize = models.values().map(|model| model.5.len()).sum();
        assert_eq!(listed, observed);
    }

    /// Three views, of which the first two share `cameras[0]`, each at its size in `sizes`,
    /// seeing one point. The dataset requires the views of one camera model to share a size.
    fn three_views(
        sizes: [(u32, u32); 3],
    ) -> std::result::Result<Reconstruction, Box<dyn std::error::Error>> {
        use openmvg_sfm::model::ImageMeta;
        use openmvg_sfm::{Dataset, PoseId, ReconstructionParts, SfmEngine, WorldToCameraPose};

        let cameras = [
            Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?,
            Camera::try_pinhole(700.0, 700.0, 400.0, 300.0)?,
        ];
        let dataset = Dataset::builder()
            .image_meta(sizes.iter().enumerate().map(|(view, &(width, height))| {
                ImageMeta::external_id(format!("view-{view}"), width, height)
            }))
            .camera_models(cameras.clone())
            .view_intrinsics([0, 0, 1].map(IntrinsicId::new))
            .build()?;
        let point = Point3::new(0.5, 0.25, 6.0);
        let mut parts = ReconstructionParts::new(
            7,
            SfmEngine::Incremental,
            dataset.camera_models().to_vec(),
            dataset.views().to_vec(),
        );
        let mut observations = Vec::new();
        for view in 0..3 {
            let center = Point3::new(view as f64 * 0.5, 0.0, 0.0);
            let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)?;
            let camera = &cameras[[0, 0, 1][view]];
            let pixel = camera.project(&pose.project_to_camera(&point).coords)?;
            parts = parts.with_pose(PoseId::new(view), pose);
            observations.push((ViewId::new(view), pixel));
        }
        Ok(Reconstruction::try_from_parts(
            parts.with_point(point, observations),
        )?)
    }

    #[test]
    fn views_that_share_a_camera_model_share_one_colmap_camera() {
        let reconstruction = three_views([(640, 480), (640, 480), (800, 600)]).unwrap();
        let scratch = tempfile::tempdir().unwrap();
        write_colmap(&reconstruction, scratch.path()).unwrap();

        let cameras = data_lines(scratch.path(), "cameras.txt");
        let ids_and_sizes: Vec<_> = cameras.iter().map(|line| line[..4].to_vec()).collect();
        // CAMERA_ID MODEL WIDTH HEIGHT: the model name does not parse as a number.
        assert_eq!(ids_and_sizes.len(), 2);
        assert_eq!(
            (
                ids_and_sizes[0][0],
                ids_and_sizes[0][2],
                ids_and_sizes[0][3]
            ),
            (1.0, 640.0, 480.0)
        );
        assert_eq!(
            (
                ids_and_sizes[1][0],
                ids_and_sizes[1][2],
                ids_and_sizes[1][3]
            ),
            (2.0, 800.0, 600.0)
        );
        let images = data_lines(scratch.path(), "images.txt");
        let camera_of_image: Vec<_> = images
            .chunks(2)
            .map(|image| (image[0][0], image[0][8]))
            .collect();
        assert_eq!(camera_of_image, [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0)]);
    }

    #[test]
    fn exporting_again_replaces_the_model_and_keeps_other_files() {
        let reconstruction = reconstruct(&engine_graph(), "incremental");
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path().join("sparse/0");
        write_colmap(&reconstruction, scratch.path()).unwrap();
        let first: Vec<_> = ["cameras.txt", "images.txt", "points3D.txt"]
            .map(|file| fs::read(dir.join(file)).unwrap())
            .into();
        fs::write(dir.join("project.ini"), "kept").unwrap();
        fs::write(dir.join("points3D.txt"), "stale").unwrap();

        write_colmap(&reconstruction, scratch.path()).unwrap();
        let mut names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["cameras.txt", "images.txt", "points3D.txt", "project.ini"]
        );
        let second: Vec<_> = ["cameras.txt", "images.txt", "points3D.txt"]
            .map(|file| fs::read(dir.join(file)).unwrap())
            .into();
        assert_eq!(first, second);
        assert_eq!(fs::read_to_string(dir.join("project.ini")).unwrap(), "kept");
    }
}
