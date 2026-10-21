// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reconstruction operation tests, one file per operation module, sharing the fixtures and test
//! triangulators below.

use super::triangulation::{
    PosedObservationPolicy, gather_posed, triangulate_posed, triangulate_posed_with_limit,
};
use super::*;
use crate::ba::backend::BaBackendResult;
use crate::model::{ImageMeta, PoseRecord};
use crate::{FeatureId, FeatureSet, LandmarkId, Track, TrackObservation, TrackSet};
use nalgebra::{Point2, Point3, Rotation3, Vector3};
use openmvg_core::camera::{
    Camera, CameraModel, FisheyeCamera, FisheyeDistortion, RadialDistortion, SphericalCamera,
    TangentialDistortion,
};

mod bundle;
mod cleanup;
mod residual;
mod shared_pose_probes;
mod track_growth;
mod triangulation;

fn set_feature_pixels(workspace: &mut SfmWorkspace, pixels_by_view: Vec<Vec<Point2<f64>>>) {
    for (view, pixels) in pixels_by_view.into_iter().enumerate() {
        workspace
            .feature_sets
            .set(view, FeatureSet::try_new(pixels).unwrap())
            .unwrap();
    }
}

fn supported_workspace() -> SfmWorkspace {
    supported_workspace_for(
        Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
        64,
        48,
    )
}

/// The supported three-view workspace under one shared `camera` and image size.
fn supported_workspace_for(camera: Camera, width: u32, height: u32) -> SfmWorkspace {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("a", width, height),
            ImageMeta::external_id("b", width, height),
            ImageMeta::external_id("c", width, height),
        ])
        .camera_models(vec![camera])
        .shared_intrinsic(IntrinsicId::new(0))
        .build()
        .expect("valid workspace");
    for (image, center_x) in [0.0, 0.5, 1.0].into_iter().enumerate() {
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(image),
                pose: openmvg_core::model::WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(center_x, 0.0, 0.0),
                )
                .unwrap(),
                component: Some(ComponentId::new(0)),
            })
            .expect("pose index is within the workspace image domain");
    }
    workspace.tracks = TrackSet::try_new(vec![
        Track::try_new(
            TrackId::new(0),
            vec![
                TrackObservation {
                    view: ViewId::new(0),
                    feature: FeatureId::new(0),
                },
                TrackObservation {
                    view: ViewId::new(1),
                    feature: FeatureId::new(0),
                },
                TrackObservation {
                    view: ViewId::new(2),
                    feature: FeatureId::new(0),
                },
            ],
        )
        .expect("valid track"),
    ])
    .expect("valid tracks");
    set_feature_pixels(
        &mut workspace,
        vec![
            vec![Point2::new(32.0, 24.0)],
            vec![Point2::new(500.0, -300.0)],
            vec![Point2::new(12.0, 24.0)],
        ],
    );
    workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
        LandmarkId::new(0),
        TrackId::new(0),
        Point3::new(0.0, 0.0, 5.0),
        openmvg_core::PixelResidual::ZERO,
        ObservationSupport::try_from_indices([0, 2], 3).expect("valid support"),
    )])
    .expect("valid landmarks");
    workspace
}

struct AcceptingTriangulator;

impl Triangulator for AcceptingTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Ok(TriangulatedLandmark {
            point: Point3::new(0.0, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::ZERO,
            supporting_observation_indices: (0..observations.len())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
        .into())
    }
}

struct FirstTwoObservationsTriangulator;

impl Triangulator for FirstTwoObservationsTriangulator {
    fn triangulate(
        &self,
        _observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Ok(TriangulatedLandmark {
            point: Point3::new(0.0, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::ZERO,
            supporting_observation_indices: vec![0, 1].into_boxed_slice(),
        }
        .into())
    }
}

fn two_missing_track_workspace() -> SfmWorkspace {
    let mut workspace = supported_workspace();
    workspace.tracks = TrackSet::try_new(vec![
        Track::try_new(
            TrackId::new(0),
            vec![
                TrackObservation {
                    view: ViewId::new(0),
                    feature: FeatureId::new(0),
                },
                TrackObservation {
                    view: ViewId::new(1),
                    feature: FeatureId::new(0),
                },
            ],
        )
        .expect("valid track"),
        Track::try_new(
            TrackId::new(1),
            vec![
                TrackObservation {
                    view: ViewId::new(0),
                    feature: FeatureId::new(1),
                },
                TrackObservation {
                    view: ViewId::new(1),
                    feature: FeatureId::new(1),
                },
            ],
        )
        .expect("valid track"),
    ])
    .expect("valid tracks");
    set_feature_pixels(
        &mut workspace,
        vec![
            vec![Point2::new(32.0, 24.0), Point2::new(33.0, 24.0)],
            vec![Point2::new(22.0, 24.0), Point2::new(23.0, 24.0)],
            vec![Point2::new(12.0, 24.0)],
        ],
    );
    workspace.landmarks = Landmarks::default();
    workspace
}
