// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Point2, Point3};
use openmvg_core::camera::{
    FisheyeCamera, FisheyeDistortion, PinholeCamera, RadialDistortion, SphericalCamera,
    TangentialDistortion,
};
use openmvg_sfm::model::{ImageMeta, PairGeometry, ViewRecord};
use openmvg_sfm::{
    Camera, FeatureId, FeatureMatch, IntrinsicId, Keypoint, PoseId, Reconstruction, SfmEngine,
    ViewGraph, ViewGraphParts, ViewId, ViewPair, WorldToCameraPose,
};

use super::cameras::{CameraRow, cameras_table};
use super::*;
use crate::fixtures::*;
use crate::format::load;
use crate::format::write::ManifestFields;
use crate::{Error, LengthUnit, ReadOptions, Rule, Violation, WriteOptions, check};

const POINT_COUNT: usize = 24;

/// A valid conformance vector; `None` when the source tree lacks the vectors.
fn testdata(vector: &str) -> Option<PathBuf> {
    crate::format::conformance::testdata().map(|root| root.join("valid").join(vector))
}

fn cameras() -> Vec<Camera> {
    let pinhole = PinholeCamera::try_new(FOCAL, 505.0, 320.5, 239.5).unwrap();
    vec![
        Camera::Pinhole(pinhole.clone()),
        Camera::try_brown_conrady(
            FOCAL,
            505.0,
            320.5,
            239.5,
            RadialDistortion::K1K2(-0.125, 0.0),
            TangentialDistortion::P1P2(0.0009765625, -0.001953125),
        )
        .unwrap(),
        Camera::Fisheye(
            FisheyeCamera::try_new(
                pinhole,
                FisheyeDistortion::K1K2K3K4(0.0078125, 0.0, -0.001, 0.0002),
            )
            .unwrap(),
        ),
        Camera::Spherical(SphericalCamera::try_new(640).unwrap()),
    ]
}

/// Four views, one per camera model, and one pair per pair kind: essential with a partial
/// inlier mask, fundamental without a pose, homography with one, rotation only, rejected with
/// matches, and rejected without.
fn mixed_graph() -> ViewGraph {
    let views = (0..4)
        .map(|view| {
            let height = if view == 3 { 320 } else { 480 };
            ViewRecord::new(
                ViewId::new(view),
                image(view, 640, height),
                IntrinsicId::new(view),
                PoseId::new(view),
            )
        })
        .collect();
    let mask: Vec<bool> = (0..POINT_COUNT).map(|index| index % 5 != 0).collect();
    let pose = |first, second| relative_pose(first, second);
    let parts = ViewGraphParts::new(
        11,
        cameras(),
        views,
        (0..4).map(|view| keypoints(view, POINT_COUNT)).collect(),
    )
    .with_pair(
        pair(0, 1),
        matches(POINT_COUNT),
        Some((
            PairGeometry::Essential {
                matrix: essential(&pose(0, 1)),
                relative_pose: Some(pose(0, 1)),
            },
            mask.clone(),
        )),
    )
    .with_pair(
        pair(0, 2),
        matches(POINT_COUNT),
        Some((
            PairGeometry::Fundamental {
                matrix: essential(&pose(0, 2)) * 1e-3,
                relative_pose: None,
            },
            vec![true; POINT_COUNT],
        )),
    )
    .with_pair(
        pair(0, 3),
        matches(POINT_COUNT),
        Some((
            PairGeometry::Homography {
                matrix: *pose(0, 3).rotation_source_to_target().matrix(),
                relative_pose: Some(pose(0, 3)),
            },
            mask,
        )),
    )
    .with_pair(
        pair(1, 2),
        matches(POINT_COUNT),
        Some((
            PairGeometry::RotationOnly {
                rotation: *pose(1, 2).rotation_source_to_target(),
            },
            vec![true; POINT_COUNT],
        )),
    )
    .with_pair(pair(1, 3), matches(POINT_COUNT), None)
    .with_pair(pair(2, 3), Vec::new(), None)
    .with_calibrated_intrinsics(true);
    ViewGraph::try_from_parts(parts).unwrap()
}

type StoredPair = (
    ViewPair,
    Vec<FeatureMatch>,
    Option<(PairGeometry, Vec<bool>)>,
);

fn graph_pairs(graph: &ViewGraph) -> Vec<StoredPair> {
    graph
        .pairs()
        .map(|pair| {
            (
                pair.views(),
                pair.matches().to_vec(),
                pair.verified()
                    .map(|(geometry, inliers)| (geometry, inliers.to_vec())),
            )
        })
        .collect()
}

fn all_keypoints(graph: &ViewGraph) -> Vec<Vec<Keypoint>> {
    graph
        .views()
        .iter()
        .map(|view| graph.keypoints(view.id).unwrap().to_vec())
        .collect()
}

fn options() -> (WriteOptions, ReadOptions) {
    (
        WriteOptions::new().with_image_root(IMAGE_ROOT),
        ReadOptions::new()
            .with_image_root(IMAGE_ROOT)
            .with_root_seed(11),
    )
}

#[test]
fn view_graph_round_trips_every_camera_model_and_pair_kind() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let graph = mixed_graph();
    let path = scratch.path().join("mixed.omvg");
    write_view_graph(&graph, &path, &write_options).unwrap();
    assert!(check(&path).is_empty());

    let read = read_view_graph(&path, &read_options).unwrap();
    assert_eq!(read.views(), graph.views());
    assert_eq!(read.camera_models(), graph.camera_models());
    assert_eq!(all_keypoints(&read), all_keypoints(&graph));
    assert_eq!(graph_pairs(&read), graph_pairs(&graph));
    assert_eq!(
        read.has_calibrated_intrinsics(),
        graph.has_calibrated_intrinsics()
    );
    assert_eq!(read.root_seed(), graph.root_seed());

    // Writing what was read reproduces the dataset, manifest hashes included.
    let again = scratch.path().join("again.omvg");
    write_view_graph(&read, &again, &write_options).unwrap();
    assert_eq!(
        fs::read(again.join("manifest.json")).unwrap(),
        fs::read(path.join("manifest.json")).unwrap()
    );
}

#[test]
fn view_graph_is_written_as_the_specification_lays_it_out() {
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("mixed.omvg");
    write_view_graph(&mixed_graph(), &path, &options().0).unwrap();
    let loaded = load(&path);
    let manifest = loaded.manifest.unwrap();
    assert_eq!(manifest.profiles, ["dataset"]);
    assert_eq!(manifest.length_unit, LengthUnit::Unknown);
    let tables = &loaded.tables;
    let views = super::columns::Rows::new(&tables["views"]);
    assert_eq!(views.string("image", 2), Some("rig/view-2.jpg"));
    let keypoints = super::columns::Rows::new(&tables["keypoints"]);
    assert_eq!(
        keypoints.float("scale", 0),
        None,
        "a zero scale is not recorded"
    );
    assert_eq!(keypoints.float("scale", 1), Some(2.5));
    let pairs = super::columns::Rows::new(&tables["pairs"]);
    let models: Vec<_> = (0..pairs.len())
        .map(|row| pairs.string("model", row))
        .collect();
    assert_eq!(
        models,
        [
            Some("essential"),
            Some("fundamental"),
            Some("homography"),
            Some("rotation"),
            None,
            None
        ]
    );
    let expected = essential(&relative_pose(0, 1));
    assert_eq!(
        pairs.double_list::<9>("matrix", 0).unwrap(),
        std::array::from_fn(|index| expected[(index / 3, index % 3)]),
        "matrices are row-major"
    );
    assert_eq!(pairs.double_list::<9>("matrix", 3), None);
    assert_eq!(pairs.double_list::<3>("translation", 1), None);
    let matches = super::columns::Rows::new(&tables["matches"]);
    let rejected =
        (0..matches.len()).filter(|&row| matches.indices("view_first").unwrap()[row] == 1);
    let rejected: Vec<_> = rejected
        .filter(|&row| matches.indices("view_second").unwrap()[row] == 3)
        .map(|row| matches.boolean("inlier", row))
        .collect();
    assert_eq!(rejected, vec![Some(false); POINT_COUNT]);
}

#[test]
fn reloaded_graph_reproduces_the_engine_reconstruction() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let graph = engine_graph();
    let path = scratch.path().join("graph.omvg");
    write_view_graph(&graph, &path, &write_options).unwrap();
    let read = read_view_graph(&path, &read_options.with_root_seed(graph.root_seed())).unwrap();

    let engine = "incremental";
    let original = reconstruct(&graph, engine);
    assert_eq!(original.registered_view_count(), ENGINE_VIEWS, "{engine}");
    let written = |reconstruction: &Reconstruction, name: &str| {
        let path = scratch.path().join(format!("{engine}-{name}.omvg"));
        write_reconstruction(reconstruction, &path, &write_options).unwrap();
        assert!(check(&path).is_empty(), "{engine}");
        path
    };
    let original_path = written(&original, "original");
    let reloaded_path = written(&reconstruct(&read, engine), "reloaded");
    assert_eq!(
        fs::read(original_path.join("manifest.json")).unwrap(),
        fs::read(reloaded_path.join("manifest.json")).unwrap(),
        "{engine}: the reloaded graph reconstructs the same scene bit for bit"
    );
}

/// The hash of each table of the dataset at `path`, by table name.
fn table_hashes(path: &Path) -> BTreeMap<String, String> {
    let manifest = load(path).manifest.unwrap();
    manifest
        .tables
        .into_iter()
        .map(|table| (table.name, format!("{:?}", table.hash)))
        .collect()
}

/// A point and its observed `(view, pixel)` measurements.
type ScenePoint = (Point3<f64>, Vec<(ViewId, Point2<f64>)>);

/// Per active landmark: its position and its observed `(view, pixel)` measurements.
fn scene(reconstruction: &Reconstruction) -> Vec<ScenePoint> {
    let mut points: Vec<_> = reconstruction
        .landmarks()
        .iter_active()
        .map(|landmark| (landmark.point, Vec::new()))
        .collect();
    let rows: BTreeMap<_, _> = reconstruction
        .landmarks()
        .iter_active()
        .enumerate()
        .map(|(row, landmark)| (landmark.id, row))
        .collect();
    for observation in reconstruction.observations() {
        points[rows[&observation.landmark]]
            .1
            .push((observation.view, observation.pixel));
    }
    points
}

#[test]
fn reconstruction_round_trips_both_engine_scenes() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let graph = engine_graph();
    for (name, engine) in [
        ("incremental", SfmEngine::Incremental),
        ("global", SfmEngine::Global),
    ] {
        let original = reconstruct(&graph, name);
        let path = scratch.path().join(format!("{name}.omvg"));
        write_reconstruction(&original, &path, &write_options).unwrap();
        let read_options = read_options
            .clone()
            .with_root_seed(original.root_seed())
            .with_engine(engine);
        let read = read_reconstruction(&path, &read_options)
            .unwrap()
            .reconstruction;

        assert_eq!(read.engine(), original.engine(), "{name}");
        assert_eq!(read.root_seed(), original.root_seed(), "{name}");
        assert_eq!(
            read.has_calibrated_intrinsics(),
            original.has_calibrated_intrinsics(),
            "{name}"
        );
        assert_eq!(read.views(), original.views(), "{name}");
        assert_eq!(read.camera_models(), original.camera_models(), "{name}");
        let poses = |reconstruction: &Reconstruction| {
            reconstruction
                .registered_views()
                .map(|(view, record)| (view, record.pose.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(poses(&read), poses(&original), "{name}");
        assert_eq!(scene(&read), scene(&original), "{name}");
        // The recomputed error is the `error_px` the writer derived from the same observations.
        let loaded = load(&path);
        let points = super::columns::Rows::new(&loaded.tables["points"]);
        for (row, landmark) in original.landmarks().iter_active().enumerate() {
            let stored = points.double("error_px", row).unwrap();
            assert!((stored - landmark.reprojection_error.value()).abs() <= 1e-9);
        }
        for (row, landmark) in read.landmarks().iter_active().enumerate() {
            let stored = points.double("error_px", row).unwrap();
            let error = landmark.reprojection_error.value();
            assert!(
                (error - stored).abs() <= 1e-12 * stored.max(1.0),
                "{name}: {error} vs {stored}"
            );
        }

        // Only the feature indices are renumbered: every other table is reproduced bit for bit,
        // and a second round trip is the identity.
        let again = scratch.path().join(format!("{name}-again.omvg"));
        write_reconstruction(&read, &again, &write_options).unwrap();
        let mut first = table_hashes(&path);
        let mut second = table_hashes(&again);
        first.remove("observations");
        let renumbered = second.remove("observations");
        assert_eq!(first, second, "{name}");
        let third = scratch.path().join(format!("{name}-third.omvg"));
        let reread = read_reconstruction(&again, &read_options)
            .unwrap()
            .reconstruction;
        write_reconstruction(&reread, &third, &write_options).unwrap();
        assert_eq!(
            table_hashes(&third).remove("observations"),
            renumbered,
            "{name}"
        );
    }
}

#[test]
fn reconstruction_reads_the_conformance_vector_once_uniformly_calibrated() {
    let Some(vector) = testdata("reconstruction") else {
        return;
    };
    // The vector mixes calibrated and uncalibrated cameras, which the SfM types cannot hold.
    let error = read_reconstruction(&vector, &ReadOptions::new()).unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("calibrated")),
        "{error}"
    );

    // With every camera uncalibrated it reads: a shared pose, an unregistered view, null
    // features, and a non-ASCII path.
    let mut tables = load(&vector).tables;
    let rows = {
        let cameras = super::columns::Rows::new(&tables["cameras"]);
        cameras
            .indices("id")
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(row, id)| {
                let size = |column| cameras.int(column, row).unwrap() as i32;
                let camera = super::cameras::camera_from_row(&cameras, row).unwrap();
                let row = CameraRow::new(&camera, size("width"), size("height"), false).unwrap();
                (id as i64, row)
            })
            .collect::<Vec<_>>()
    };
    tables.insert("cameras".to_owned(), cameras_table(&rows).unwrap());
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("uncalibrated.omvg");
    let fields = ManifestFields {
        profiles: vec!["reconstruction".to_owned()],
        length_unit: LengthUnit::Meter,
        generator: None,
        extensions_used: Vec::new(),
    };
    crate::format::write::write(&path, tables.clone(), &fields).unwrap();

    let stored = read_reconstruction(&path, &ReadOptions::new()).unwrap();
    assert_eq!(stored.length_unit, LengthUnit::Meter);
    let read = &stored.reconstruction;
    assert_eq!(read.view_count(), tables["views"].rows);
    assert_eq!(read.registered_view_count(), tables["views"].rows - 1);
    assert_eq!(read.point_count(), tables["points"].rows);
    assert_eq!(read.observations().count(), tables["observations"].rows);
    assert_eq!(read.engine(), SfmEngine::default());
    assert!(!read.has_calibrated_intrinsics());

    // A Reconstruction holds no unit: the unit read back is kept only when passed on, and the
    // coordinates are written unchanged either way.
    let kept = scratch.path().join("kept.omvg");
    let relabel = WriteOptions::new().with_length_unit(stored.length_unit);
    write_reconstruction(read, &kept, &relabel).unwrap();
    assert_eq!(
        read_reconstruction(&kept, &ReadOptions::new())
            .unwrap()
            .length_unit,
        LengthUnit::Meter
    );
    let unlabeled = scratch.path().join("unlabeled.omvg");
    write_reconstruction(read, &unlabeled, &WriteOptions::new()).unwrap();
    assert_eq!(
        read_reconstruction(&unlabeled, &ReadOptions::new())
            .unwrap()
            .length_unit,
        LengthUnit::Unknown
    );
    assert_eq!(table_hashes(&kept), table_hashes(&unlabeled));
}

#[test]
fn ground_truth_reads_the_benchmark_poses() {
    let Some(vector) = testdata("full") else {
        return;
    };
    let truth = read_ground_truth(&vector).unwrap();
    let loaded = load(&vector);
    assert_eq!(truth.views.len(), loaded.tables["gt_poses"].rows);
    assert_eq!(loaded.manifest.unwrap().length_unit, truth.length_unit);
    for view in &truth.views {
        let rotation = view.pose.rotation_matrix();
        assert!((rotation.transpose() * rotation - Matrix3::identity()).norm() <= 1e-8);
    }
}

#[test]
fn mixed_calibration_is_unsupported() {
    let scratch = tempfile::tempdir().unwrap();
    let graph = mixed_graph();
    let path = scratch.path().join("graph.omvg");
    write_view_graph(&graph, &path, &WriteOptions::new()).unwrap();
    let mut tables = load(&path).tables;
    for flags in [
        [Some(true), Some(false), None, None],
        [Some(true), None, None, None],
    ] {
        let rows = graph
            .camera_models()
            .iter()
            .zip(graph.views())
            .zip(flags)
            .map(|((camera, view), flag)| {
                let size = (view.meta.width as i32, view.meta.height as i32);
                let mut row = CameraRow::new(camera, size.0, size.1, false).unwrap();
                row.calibrated = flag;
                (view.intrinsic_id.index() as i64, row)
            })
            .collect::<Vec<_>>();
        tables.insert("cameras".to_owned(), cameras_table(&rows).unwrap());
        let mixed = scratch
            .path()
            .join(format!("mixed-{}.omvg", flags[1].is_some()));
        let fields = ManifestFields {
            profiles: vec!["dataset".to_owned()],
            length_unit: LengthUnit::Unknown,
            generator: None,
            extensions_used: Vec::new(),
        };
        crate::format::write::write(&mixed, tables.clone(), &fields).unwrap();
        let error = read_view_graph(&mixed, &ReadOptions::new()).unwrap_err();
        assert!(
            matches!(&error, Error::Unsupported(message) if message.contains("calibrated")),
            "{error}"
        );
    }
}

#[test]
fn images_outside_the_image_root_are_unsupported() {
    let scratch = tempfile::tempdir().unwrap();
    let options = WriteOptions::new().with_image_root("/elsewhere");
    let error =
        write_view_graph(&mixed_graph(), scratch.path().join("g.omvg"), &options).unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("image root")),
        "{error}"
    );
}

#[test]
fn without_an_image_root_the_image_is_its_name() {
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("names.omvg");
    write_view_graph(&mixed_graph(), &path, &WriteOptions::new()).unwrap();
    let read = read_view_graph(&path, &ReadOptions::new()).unwrap();
    assert_eq!(
        read.views()[1].meta.file_path(),
        Some(path.join("rig/view-1.jpg").as_path())
    );
}

#[test]
fn images_read_back_named_by_their_dataset_path() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let mut written = putative();
    for (view, folder) in written.views.iter_mut().zip(["a", "b"]) {
        let file = Path::new(IMAGE_ROOT).join(folder).join("frame.jpg");
        view.meta = ImageMeta::path(file, view.meta.width, view.meta.height);
    }
    assert_eq!(written.views[0].meta.name, written.views[1].meta.name);
    let path = scratch.path().join("folders.omvg");
    write_correspondences(&written, &path, &write_options).unwrap();
    let read = read_correspondences(&path, &read_options).unwrap();
    assert_eq!(read.views[0].meta.name, "a/frame.jpg");
    assert_eq!(read.views[1].meta.name, "b/frame.jpg");
    assert_eq!(
        read.views[0].meta.file_path(),
        written.views[0].meta.file_path()
    );

    // Without an image root the name is the image, so the two files stay apart.
    let again = scratch.path().join("again.omvg");
    write_correspondences(&read, &again, &WriteOptions::new()).unwrap();
    assert_eq!(read_correspondences(&again, &read_options).unwrap(), read);
}

#[cfg(unix)]
#[test]
fn an_image_must_resolve_inside_the_image_root() {
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("names.omvg");
    write_view_graph(&mixed_graph(), &path, &WriteOptions::new()).unwrap();
    let images = scratch.path().join("images");
    fs::create_dir_all(images.join("rig")).unwrap();
    let read_options = ReadOptions::new().with_image_root(&images);

    // Missing images and files inside the root read.
    read_view_graph(&path, &read_options).unwrap();
    fs::write(images.join("rig/view-0.jpg"), b"inside").unwrap();
    std::os::unix::fs::symlink(images.join("rig/view-0.jpg"), images.join("rig/view-2.jpg"))
        .unwrap();
    read_view_graph(&path, &read_options).unwrap();

    // A symbolic link out of the root does not.
    let outside = scratch.path().join("secret.jpg");
    fs::write(&outside, b"outside").unwrap();
    std::os::unix::fs::symlink(&outside, images.join("rig/view-1.jpg")).unwrap();
    let error = read_view_graph(&path, &read_options).unwrap_err();
    assert!(
        matches!(&error, Error::Invalid(violations)
            if violations.iter().map(Violation::rule).eq([Rule::ViewsImagePath])),
        "{error}"
    );
}

/// The views of `mixed_graph` with putative matches on three pairs; the last keypoint of view 0
/// records neither a scale nor an orientation.
fn putative() -> Correspondences {
    let graph = mixed_graph();
    let mut keypoints = all_keypoints(&graph);
    keypoints[0].push(Keypoint::try_new(12.5, 7.25, 0.0, 0.0).unwrap());
    let mut extra = matches(POINT_COUNT);
    extra
        .push(FeatureMatch::try_new(FeatureId::new(POINT_COUNT), FeatureId::new(0), None).unwrap());
    Correspondences::new(
        graph.camera_models().to_vec(),
        graph.views().to_vec(),
        keypoints,
    )
    .with_calibrated_intrinsics(true)
    .with_pair(pair(0, 1), extra)
    .with_pair(pair(0, 3), matches(POINT_COUNT))
    .with_pair(pair(2, 3), matches(5))
}

fn with_ground_truth(correspondences: Correspondences) -> Correspondences {
    (0..3).fold(correspondences, |correspondences, view| {
        let (rotation, center) = view_pose(view);
        let pose = WorldToCameraPose::from_rotation_center(rotation, Point3::from(center)).unwrap();
        correspondences.with_ground_truth(ViewId::new(view), pose)
    })
}

#[test]
fn correspondences_round_trip_with_null_inlier_flags_and_ground_truth() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let written = with_ground_truth(putative());
    let path = scratch.path().join("putative.omvg");
    write_correspondences(&written, &path, &write_options).unwrap();
    assert!(check(&path).is_empty());

    let loaded = load(&path);
    assert_eq!(loaded.manifest.unwrap().profiles, ["benchmark", "dataset"]);
    assert!(!loaded.tables.contains_key("pairs"));
    let matches = super::columns::Rows::new(&loaded.tables["matches"]);
    assert!((0..matches.len()).all(|row| matches.boolean("inlier", row).is_none()));
    let keypoints = super::columns::Rows::new(&loaded.tables["keypoints"]);
    let last = POINT_COUNT;
    assert_eq!(keypoints.float("scale", last), None);
    assert_eq!(keypoints.float("orientation", last), None);
    assert_eq!(
        keypoints.float("orientation", 0),
        Some(-1.0),
        "an orientation without a scale is recorded"
    );

    assert_eq!(read_correspondences(&path, &read_options).unwrap(), written);
    assert_eq!(read_ground_truth(&path).unwrap().views.len(), 3);
}

#[test]
fn correspondences_keep_their_length_unit_unless_relabeled() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let metric = with_ground_truth(putative()).with_length_unit(LengthUnit::Meter);
    let path = scratch.path().join("metric.omvg");
    write_correspondences(&metric, &path, &write_options).unwrap();
    assert_eq!(load(&path).manifest.unwrap().length_unit, LengthUnit::Meter);
    assert_eq!(read_correspondences(&path, &read_options).unwrap(), metric);
    assert_eq!(
        read_ground_truth(&path).unwrap().length_unit,
        LengthUnit::Meter
    );

    // A relabel changes the label only: the poses are not rescaled.
    let relabeled = scratch.path().join("relabeled.omvg");
    let relabel = write_options.clone().with_length_unit(LengthUnit::Unknown);
    write_correspondences(&metric, &relabeled, &relabel).unwrap();
    let read = read_correspondences(&relabeled, &read_options).unwrap();
    assert_eq!(read.length_unit, LengthUnit::Unknown);
    assert_eq!(read.ground_truth, metric.ground_truth);
}

#[test]
fn correspondences_without_ground_truth_are_a_plain_dataset() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let path = scratch.path().join("putative.omvg");
    write_correspondences(&putative(), &path, &write_options).unwrap();
    assert_eq!(load(&path).manifest.unwrap().profiles, ["dataset"]);
    let read = read_correspondences(&path, &read_options).unwrap();
    assert!(read.ground_truth.is_empty());
    assert_eq!(read, putative());

    // A ViewGraph holds verified pairs only.
    let error = read_view_graph(&path, &read_options).unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("unverified")),
        "{error}"
    );
}

#[test]
fn a_verified_dataset_reads_as_its_putative_matches() {
    let scratch = tempfile::tempdir().unwrap();
    let (write_options, read_options) = options();
    let graph = mixed_graph();
    let path = scratch.path().join("mixed.omvg");
    write_view_graph(&graph, &path, &write_options).unwrap();
    let read = read_correspondences(&path, &read_options).unwrap();
    assert_eq!(read.views, graph.views());
    assert_eq!(read.keypoints, all_keypoints(&graph));
    // Inlier flags, models, and the pair without matches are dropped.
    let expected: Vec<_> = graph
        .pairs()
        .filter(|pair| !pair.matches().is_empty())
        .map(|pair| (pair.views(), pair.matches().to_vec()))
        .collect();
    assert_eq!(read.pairs, expected);
}

#[test]
fn correspondences_need_one_keypoint_list_per_view() {
    let scratch = tempfile::tempdir().unwrap();
    let mut correspondences = putative();
    correspondences.keypoints.pop();
    let error = write_correspondences(
        &correspondences,
        scratch.path().join("short.omvg"),
        &options().0,
    )
    .unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("keypoint lists")),
        "{error}"
    );
}

#[test]
fn typed_readers_require_their_profile_and_valid_tables() {
    let Some(root) = crate::format::conformance::testdata() else {
        return;
    };
    type Reader = fn(&Path) -> crate::Result<()>;
    let cases: [(Reader, &str, Option<&str>); 4] = [
        (
            |path| read_reconstruction(path, &ReadOptions::new()).map(|_| ()),
            "valid/empty",
            Some("reconstruction"),
        ),
        (
            |path| read_ground_truth(path).map(|_| ()),
            "valid/reconstruction",
            Some("benchmark"),
        ),
        (
            |path| read_view_graph(path, &ReadOptions::new()).map(|_| ()),
            "valid/reconstruction",
            Some("dataset"),
        ),
        (
            |path| read_view_graph(path, &ReadOptions::new()).map(|_| ()),
            "invalid/camera_model",
            None,
        ),
    ];
    for (read, vector, profile) in cases {
        let error = read(&root.join(vector)).unwrap_err();
        match profile {
            Some(profile) => assert!(
                matches!(&error, Error::Unsupported(message) if message.contains(profile)),
                "{vector}: {error}"
            ),
            None => assert!(
                matches!(&error, Error::Invalid(violations) if !violations.is_empty()),
                "{vector}: {error}"
            ),
        }
    }
}

mod strongest_pairs;

mod shared_pose_probes;
