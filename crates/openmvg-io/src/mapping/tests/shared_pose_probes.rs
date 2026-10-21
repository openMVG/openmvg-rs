// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::mapping::columns::Rows;
use openmvg_sfm::ReconstructionParts;

#[test]
fn shared_pose_probe_omvg_preserves_two_intrinsics_on_one_pose() {
    let centers = [Point3::origin(), Point3::new(0.6, 0.05, 0.0)];
    let calibration = [[500.0, 510.0, 320.0, 240.0], [730.0, 680.0, 310.0, 250.0]];
    let pose_ids = [0, 1, 1];
    let intrinsic_ids = [0, 0, 1];
    let views = (0..3)
        .map(|view| {
            ViewRecord::new(
                ViewId::new(view),
                ImageMeta::external_id(view.to_string(), 640, 480),
                IntrinsicId::new(intrinsic_ids[view]),
                PoseId::new(pose_ids[view]),
            )
        })
        .collect();
    let cameras = calibration
        .iter()
        .map(|k| Camera::try_pinhole(k[0], k[1], k[2], k[3]).unwrap())
        .collect();
    let mut parts = ReconstructionParts::new(2026_u64, SfmEngine::Global, cameras, views)
        .with_calibrated_intrinsics(true);
    for (pose, center) in centers.iter().enumerate() {
        parts = parts.with_pose(
            PoseId::new(pose),
            WorldToCameraPose::from_rotation_center(nalgebra::Rotation3::identity(), *center)
                .unwrap(),
        );
    }
    for point in [
        Point3::new(-0.3, 0.1, 4.0),
        Point3::new(0.4, -0.2, 5.0),
        Point3::new(0.1, 0.3, 6.0),
    ] {
        let observations = (0..3)
            .map(|view| {
                let q = point - centers[pose_ids[view]];
                let k = calibration[intrinsic_ids[view]];
                (
                    ViewId::new(view),
                    Point2::new(k[0] * q.x / q.z + k[2], k[1] * q.y / q.z + k[3]),
                )
            })
            .collect();
        parts = parts.with_point(point, observations);
    }
    let original = Reconstruction::try_from_parts(parts).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("shared.omvg");
    let options = WriteOptions::new().with_length_unit(LengthUnit::Meter);
    write_reconstruction(&original, &path, &options).unwrap();
    assert!(check(&path).is_empty());
    let loaded = load(&path);
    let pose_rows = Rows::new(&loaded.tables["poses"]);
    let view_rows = Rows::new(&loaded.tables["views"]);
    assert_eq!(
        pose_rows.len(),
        2,
        "shared physical poses must have one stored row"
    );
    assert_eq!(view_rows.len(), 3);
    assert_eq!(view_rows.int("pose_id", 1), view_rows.int("pose_id", 2));
    assert_ne!(view_rows.int("camera_id", 1), view_rows.int("camera_id", 2));
    let read = read_reconstruction(
        &path,
        &ReadOptions::new()
            .with_root_seed(2026_u64)
            .with_engine(SfmEngine::Global),
    )
    .unwrap();
    assert_eq!(read.length_unit, LengthUnit::Meter);
    let restored = read.reconstruction;
    let mapped_poses: Vec<_> = restored.views().iter().map(|view| view.pose_id).collect();
    assert_eq!(mapped_poses, pose_ids.map(PoseId::new));
    assert_eq!(restored.camera_models(), original.camera_models());
    assert_eq!(
        restored
            .views()
            .iter()
            .map(|view| view.intrinsic_id)
            .collect::<Vec<_>>(),
        intrinsic_ids.map(IntrinsicId::new),
    );
    for view in 0..3 {
        assert_eq!(
            restored.pose(ViewId::new(view)),
            original.pose(ViewId::new(view))
        );
    }
    let measurements = |scene: &Reconstruction| {
        scene
            .observations()
            .map(|observation| {
                (
                    observation.landmark,
                    observation.view,
                    observation.feature,
                    observation.pixel,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(measurements(&restored), measurements(&original));
    assert_eq!(restored.observations().count(), 9);
    let again = scratch.path().join("again.omvg");
    write_reconstruction(&restored, &again, &options).unwrap();
    assert_eq!(
        fs::read(path.join("manifest.json")).unwrap(),
        fs::read(again.join("manifest.json")).unwrap(),
    );
    eprintln!(
        "SHARED_POSE_PROBE {}",
        serde_json::json!({
            "case": "omvg_round_trip",
            "views": 3,
            "physical_poses": 2,
            "intrinsics": 2,
            "observations": 9,
            "same_pose_different_intrinsics": true,
            "manifest_round_trip_equal": true,
        })
    );
}
