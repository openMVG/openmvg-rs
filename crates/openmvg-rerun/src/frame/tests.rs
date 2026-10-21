// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::GraphLayers;
use nalgebra::{Matrix3, Point2, Point3, Rotation3, Vector3};
use openmvg_features::tracking::{Track, TrackSet as Tracks};
use openmvg_sfm::model::{ImageMeta, Landmark, PoseRecord};
use openmvg_sfm::testing::{WorkspaceFixtureState, apply_workspace_fixture_state};
use openmvg_sfm::unstable::ops::averaging::{
    VerifiedPairSelection, verified_pose_graph_consistency,
};
use openmvg_sfm::unstable::reports::ErrorStats;
use openmvg_sfm::unstable::types::{
    GeometricModel, InlierMask, ModelEvidence, ObservationSupport, TerminationReason,
};
use openmvg_sfm::{
    Camera, FeatureId, FeatureSet, IntrinsicId, LandmarkId, PoseId, RelativePose, TrackId,
    TrackObservation, ViewPair, WorldToCameraPose, unstable::SfmWorkspace,
};
use std::f64::consts::FRAC_PI_2;
use std::num::NonZeroU64;

fn model_evidence() -> ModelEvidence {
    ModelEvidence::try_new(
        InlierMask::new(Vec::new()),
        ErrorStats::from_values(&[], &[]),
        openmvg_core::AngularThreshold::try_new(1.0).unwrap().into(),
        "test",
        1,
        TerminationReason::Converged,
        Vec::new(),
    )
    .unwrap()
}

fn verified_model() -> GeometricModel {
    GeometricModel::Essential {
        matrix: Matrix3::identity(),
        relative_pose: RelativePose::from_baseline_direction(
            Rotation3::identity(),
            Vector3::x_axis(),
        )
        .unwrap(),
        evidence: model_evidence(),
    }
}

fn fixture() -> SfmWorkspace {
    let track = Track::try_new(
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
    .unwrap();
    let workspace = SfmWorkspace::builder()
        .image_meta((0..4).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)))
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs([
            (ViewId::new(0), ViewId::new(1)),
            (ViewId::new(0), ViewId::new(3)),
            (ViewId::new(1), ViewId::new(2)),
            (ViewId::new(2), ViewId::new(3)),
        ])
        .tracks(
            Tracks::try_new(vec![track]).unwrap(),
            [
                FeatureSet::try_new(vec![Point2::new(10.0, 10.0)]).unwrap(),
                FeatureSet::try_new(vec![Point2::new(11.0, 10.0)]).unwrap(),
                FeatureSet::default(),
                FeatureSet::default(),
            ],
        )
        .build()
        .unwrap();
    apply_workspace_fixture_state(
        workspace,
        WorkspaceFixtureState {
            inactive_pairs: vec![ViewPair::new(ViewId::new(2), ViewId::new(3))],
            geometric_models: [
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(0), ViewId::new(3)),
                (ViewId::new(1), ViewId::new(2)),
                (ViewId::new(2), ViewId::new(3)),
            ]
            .map(|(first, second)| (ViewPair::new(first, second), verified_model()))
            .into_iter()
            .collect(),
            poses: vec![
                PoseRecord::new(PoseId::new(0), WorldToCameraPose::identity(), None),
                PoseRecord::new(
                    PoseId::new(1),
                    WorldToCameraPose::from_rotation_center(
                        Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2),
                        Point3::new(2.0, 3.0, 4.0),
                    )
                    .unwrap(),
                    None,
                ),
            ],
            landmarks: vec![Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.25).unwrap(),
                ObservationSupport::default(),
            )],
        },
    )
    .unwrap()
}

fn quality_fixture(point: Point3<f64>, second_observation: Point2<f64>) -> SfmWorkspace {
    let track = Track::try_new(
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
    .unwrap();
    let workspace = SfmWorkspace::builder()
        .image_meta((0..2).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)))
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .tracks(
            Tracks::try_new(vec![track]).unwrap(),
            [
                FeatureSet::try_new(vec![Point2::new(32.0, 24.0)]).unwrap(),
                FeatureSet::try_new(vec![second_observation]).unwrap(),
            ],
        )
        .build()
        .unwrap();
    let poses = (0..2)
        .map(|pose| PoseRecord::new(PoseId::new(pose), WorldToCameraPose::identity(), None))
        .collect();
    apply_workspace_fixture_state(
        workspace,
        WorkspaceFixtureState {
            poses,
            landmarks: vec![Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                point,
                openmvg_core::PixelResidual::ZERO,
                ObservationSupport::default(),
            )],
            ..WorkspaceFixtureState::default()
        },
    )
    .unwrap()
}

fn track(id: usize, images: &[usize]) -> Track {
    Track::try_new(
        TrackId::new(id),
        images
            .iter()
            .enumerate()
            .map(|(feature, image)| TrackObservation {
                view: ViewId::new(*image),
                feature: FeatureId::new(feature),
            })
            .collect(),
    )
    .unwrap()
}

fn event(workspace: &SfmWorkspace, sequence: u64, transition: SfmEventTransition) -> SfmEvent<'_> {
    SfmEvent::new(
        sequence,
        SfmEventBranch::Main,
        SfmEventPhase::IncrementalRegistration { index: 0 },
        transition,
        SfmSnapshot::from_workspace(workspace),
    )
}

fn extract_frame(
    event: SfmEvent<'_>,
    config: &RerunRecorderConfig,
    world_origin: [f64; 3],
) -> VisualFrame {
    let ceilings = SourceTrackIncidenceCeilings::extract(event.snapshot);
    VisualFrame::extract(event, config, world_origin, ceilings)
}

#[test]
fn extraction_is_deterministic_and_layers_have_exact_semantics() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        graph_layers: GraphLayers {
            verified: true,
            active: true,
            registered_induced: true,
            frontier: true,
        },
        ..RerunRecorderConfig::default()
    };
    let first = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &config,
        [0.0; 3],
    );
    let second = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &config,
        [0.0; 3],
    );
    assert_eq!(first, second);
    assert_eq!(first.metrics.verified_edge_count, 4);
    assert_eq!(first.metrics.active_edge_count, 3);
    assert_eq!(first.metrics.registered_induced_edge_count, 1);
    assert_eq!(first.metrics.frontier_edge_count, 2);
    let graph = first.graph.as_ref().unwrap();
    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.max_tracks_per_image, 1);
    assert_eq!(graph.nodes[0].active_landmark_count, 1);
    assert_eq!(graph.nodes[1].active_landmark_count, 1);
    assert_eq!(graph.nodes[2].active_landmark_count, 0);
    assert_eq!(graph.nodes[3].active_landmark_count, 0);
    assert_eq!(graph.layers[0].edges.len(), 4);
    assert_eq!(graph.layers[1].edges.len(), 3);
    assert_eq!(graph.layers[2].edges.len(), 1);
    assert_eq!(graph.layers[3].edges.len(), 2);
    let registered_edges = &first.geometry.as_ref().unwrap().registered_edges;
    assert_eq!(
        first
            .geometry
            .as_ref()
            .unwrap()
            .max_shared_tracks_per_verified_pair,
        1
    );
    assert_eq!(registered_edges.len(), 1);
    assert_eq!(registered_edges[0].first, ViewId::new(0));
    assert_eq!(registered_edges[0].second, ViewId::new(1));
    assert_eq!(registered_edges[0].shared_active_landmark_count, 1);
}

#[test]
fn incidence_ceilings_use_actual_image_and_verified_pair_counts() {
    let workspace = SfmWorkspace::builder()
        .image_meta((0..3).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)))
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs([
            (ViewId::new(0), ViewId::new(1)),
            (ViewId::new(0), ViewId::new(2)),
            (ViewId::new(1), ViewId::new(2)),
        ])
        .tracks(
            Tracks::try_new(vec![
                track(0, &[0, 1]),
                track(1, &[0, 1]),
                track(2, &[0, 2]),
                track(3, &[1, 2]),
            ])
            .unwrap(),
            (0..3).map(|_| {
                FeatureSet::try_new(vec![Point2::origin(), Point2::new(1.0, 0.0)]).unwrap()
            }),
        )
        .build()
        .unwrap();
    assert_eq!(
        SourceTrackIncidenceCeilings::extract(SfmSnapshot::from_workspace(&workspace))
            .max_shared_tracks_per_verified_pair,
        0
    );
    let workspace = apply_workspace_fixture_state(
        workspace,
        WorkspaceFixtureState {
            geometric_models: [
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(0), ViewId::new(2)),
                (ViewId::new(1), ViewId::new(2)),
            ]
            .map(|(first, second)| (ViewPair::new(first, second), verified_model()))
            .into_iter()
            .collect(),
            ..WorkspaceFixtureState::default()
        },
    )
    .unwrap();
    let ceilings = SourceTrackIncidenceCeilings::extract(SfmSnapshot::from_workspace(&workspace));
    assert_eq!(ceilings.max_tracks_per_image, 3);
    assert_eq!(ceilings.max_shared_tracks_per_verified_pair, 2);
    assert!(ceilings.max_shared_tracks_per_verified_pair < workspace.tracks().len());
    assert!(ceilings.matches_shape(SfmSnapshot::from_workspace(&workspace)));
    let empty = SfmWorkspace::builder().build().unwrap();
    assert!(!ceilings.matches_shape(SfmSnapshot::from_workspace(&empty)));
}

#[test]
fn sampled_quality_uses_exact_observation_bearing_residuals() {
    let workspace = quality_fixture(Point3::new(0.0, 0.0, 5.0), Point2::new(132.0, 24.0));
    let quality = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &RerunRecorderConfig::default(),
        [0.0; 3],
    )
    .quality
    .unwrap();
    let shifted_residual = std::f64::consts::FRAC_PI_4;
    let expected_rmse = shifted_residual / 2.0_f64.sqrt();
    let expected_p90 = shifted_residual;
    assert_eq!(quality.valid_observation_count, 2);
    assert_eq!(quality.invalid_observation_count, 0);
    assert_eq!(quality.unevaluable_landmark_count, 0);
    assert!((quality.bearing_rmse_rad - expected_rmse).abs() < 1e-14);
    assert!((quality.bearing_p90_rad - expected_p90).abs() < 1e-14);
    assert!((quality.reprojection_rmse_px - 100.0 / 2.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn selected_landmark_support_excludes_rejected_observations() {
    let source_track = Track::try_new(
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
    .unwrap();
    let workspace = SfmWorkspace::builder()
        .image_meta((0..3).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)))
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .tracks(
            Tracks::try_new(vec![source_track]).unwrap(),
            [
                FeatureSet::try_new(vec![Point2::new(32.0, 24.0)]).unwrap(),
                FeatureSet::try_new(vec![Point2::new(32.0, 24.0)]).unwrap(),
                FeatureSet::try_new(vec![Point2::new(1_000.0, 1_000.0)]).unwrap(),
            ],
        )
        .build()
        .unwrap();
    let workspace = apply_workspace_fixture_state(
        workspace,
        WorkspaceFixtureState {
            poses: (0..3)
                .map(|index| {
                    PoseRecord::new(PoseId::new(index), WorldToCameraPose::identity(), None)
                })
                .collect(),
            landmarks: vec![Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::ZERO,
                ObservationSupport::try_from_indices([0, 1], 3).unwrap(),
            )],
            ..WorkspaceFixtureState::default()
        },
    )
    .unwrap();

    let quality = extract_quality(SfmSnapshot::from_workspace(&workspace));
    assert_eq!(quality.valid_observation_count, 2);
    assert_eq!(quality.invalid_observation_count, 0);
    assert_eq!(quality.unevaluable_landmark_count, 0);
    assert_eq!(quality.bearing_rmse_rad, 0.0);
    assert_eq!(quality.reprojection_rmse_px, 0.0);

    let incidence = extract_active_landmark_incidence(SfmSnapshot::from_workspace(&workspace));
    assert_eq!(incidence.images, vec![1, 1, 0]);
}

#[test]
fn sampled_pose_graph_quality_is_empty_without_poses() {
    let workspace = SfmWorkspace::builder().build().unwrap();
    let quality = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &RerunRecorderConfig::default(),
        [0.0; 3],
    )
    .pose_graph_quality
    .unwrap();
    assert_eq!(quality.active_rotation_p90_deg, None);
    assert_eq!(quality.active_translation_p90_deg, None);
    assert_eq!(quality.all_verified_rotation_p90_deg, None);
}

#[test]
fn sampled_pose_graph_quality_uses_verified_relative_poses() {
    let workspace = fixture();
    let quality = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &RerunRecorderConfig::default(),
        [0.0; 3],
    )
    .pose_graph_quality
    .unwrap();
    assert!(quality.active_rotation_p90_deg.is_some_and(f64::is_finite));
    assert!(
        quality
            .active_translation_p90_deg
            .is_some_and(f64::is_finite)
    );
    assert!(
        quality
            .all_verified_rotation_p90_deg
            .is_some_and(f64::is_finite)
    );
    let active = verified_pose_graph_consistency(&workspace, VerifiedPairSelection::Active);
    let all = verified_pose_graph_consistency(&workspace, VerifiedPairSelection::All);
    assert_eq!(
        quality.active_rotation_p90_deg,
        active.rotation_residual_deg.map(|values| values.p90_deg)
    );
    assert_eq!(
        quality.active_translation_p90_deg,
        active.translation_residual_deg.map(|values| values.p90_deg)
    );
    assert_eq!(
        quality.all_verified_rotation_p90_deg,
        all.rotation_residual_deg.map(|values| values.p90_deg)
    );
}

#[test]
fn pose_graph_rotation_quality_includes_rotation_only_edges() {
    let workspace = SfmWorkspace::builder()
        .image_meta((0..2).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)))
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs([(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();
    let workspace = apply_workspace_fixture_state(
        workspace,
        WorkspaceFixtureState {
            geometric_models: vec![(
                ViewPair::new(ViewId::new(0), ViewId::new(1)),
                GeometricModel::RotationOnly {
                    rotation: Rotation3::identity(),
                    evidence: model_evidence(),
                },
            )],
            poses: (0..2)
                .map(|index| {
                    PoseRecord::new(PoseId::new(index), WorldToCameraPose::identity(), None)
                })
                .collect(),
            ..WorkspaceFixtureState::default()
        },
    )
    .unwrap();
    let quality = extract_pose_graph_quality(SfmSnapshot::from_workspace(&workspace));
    assert_eq!(quality.active_rotation_p90_deg, Some(0.0));
    assert_eq!(quality.active_translation_p90_deg, None);
    assert_eq!(quality.all_verified_rotation_p90_deg, Some(0.0));
}

#[test]
fn quality_counts_behind_camera_observations_as_invalid() {
    // An active landmark seen by an unregistered view is not a valid workspace, so a landmark
    // behind the cameras is the case that leaves observations unevaluable.
    let behind = quality_fixture(Point3::new(0.0, 0.0, -5.0), Point2::new(32.0, 24.0));
    let quality = extract_quality(SfmSnapshot::from_workspace(&behind));
    assert_eq!(quality.valid_observation_count, 0);
    assert_eq!(quality.invalid_observation_count, 2);
    assert_eq!(quality.unevaluable_landmark_count, 0);
    assert_eq!(quality.bearing_rmse_rad, 0.0);
    assert_eq!(quality.bearing_p90_rad, 0.0);
    assert_eq!(quality.reprojection_rmse_px, 0.0);
}

#[test]
fn spherical_pixel_residual_wraps_across_the_longitude_seam() {
    let camera = Camera::Spherical(openmvg_core::camera::SphericalCamera::try_new(1024).unwrap());
    let residual = camera_pixel_residual(&camera, [1023.75, 200.0], [0.25, 200.0]);
    assert!((residual - 0.5).abs() < 1e-12);
}

#[test]
fn bearing_p90_uses_the_upper_order_statistic() {
    assert_eq!(sorted_upper_quantile(&[], 0.9), 0.0);
    assert_eq!(sorted_upper_quantile(&[2.0], 0.9), 2.0);
    assert_eq!(sorted_upper_quantile(&[0.0, 10.0], 0.9), 10.0);
    assert_eq!(sorted_upper_quantile(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.9), 4.0);
}

#[test]
fn registered_topology_grows_with_reconstructed_cameras() {
    assert!(!GraphLayer::RegisteredInduced.includes_node(false));
    assert!(GraphLayer::RegisteredInduced.includes_node(true));
    assert!(GraphLayer::Verified.includes_node(false));
    assert!(GraphLayer::Active.includes_node(false));
    assert!(GraphLayer::Frontier.includes_node(false));
}

#[test]
fn camera_axes_apply_camera_to_world_rotation() {
    let workspace = fixture();
    let frame = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &RerunRecorderConfig {
            camera_axis_length: 1.0,
            ..RerunRecorderConfig::default()
        },
        [0.0; 3],
    );
    let camera = frame
        .geometry
        .unwrap()
        .cameras
        .into_iter()
        .find(|camera| camera.view == ViewId::new(1))
        .unwrap();
    assert_eq!(camera.kind, VisualCameraKind::Pinhole);
    assert_eq!(camera.center, [2.0, 3.0, 4.0]);
    assert!((camera.axis_endpoints[0][0] - 2.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[0][1] - 2.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[0][2] - 4.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[1][0] - 3.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[1][1] - 3.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[1][2] - 4.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[2][0] - 2.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[2][1] - 3.0).abs() < 1e-6);
    assert!((camera.axis_endpoints[2][2] - 5.0).abs() < 1e-6);
}

#[test]
fn graph_and_geometry_limits_are_explicit() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        graph_layers: GraphLayers {
            verified: false,
            active: false,
            registered_induced: true,
            frontier: false,
        },
        max_landmarks: 0,
        max_edges_per_layer: 0,
        ..RerunRecorderConfig::default()
    };
    let frame = extract_frame(
        event(&workspace, 0, SfmEventTransition::Checkpoint),
        &config,
        [0.0; 3],
    );
    assert_eq!(frame.metrics.registered_induced_edge_count, 1);
    let graph = frame.graph.unwrap();
    assert_eq!(graph.layers.len(), 1);
    assert_eq!(graph.layers[0].layer, GraphLayer::RegisteredInduced);
    assert!(graph.layers[0].edges.is_empty());
    let geometry = frame.geometry.unwrap();
    assert_eq!(geometry.landmarks, Some(Vec::new()));
    assert!(geometry.registered_edges.is_empty());
}

#[test]
fn stride_filters_intermediate_geometry_but_keeps_terminal_states() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        geometry: GeometryCapture::Every(NonZeroU64::new(10).unwrap()),
        ..RerunRecorderConfig::default()
    };
    let intermediate = SfmEvent::new(
        1,
        SfmEventBranch::Main,
        SfmEventPhase::IncrementalFinalBundleAdjustment,
        SfmEventTransition::Checkpoint,
        SfmSnapshot::from_workspace(&workspace),
    );
    let intermediate = extract_frame(intermediate, &config, [0.0; 3]);
    assert!(!intermediate.has_geometry());
    assert!(!intermediate.has_detail());
    assert!(intermediate.quality.is_none());
    assert!(intermediate.pose_graph_quality.is_none());
    assert!(
        extract_frame(
            event(&workspace, 1, SfmEventTransition::RolledBack),
            &config,
            [0.0; 3],
        )
        .has_detail()
    );
    assert!(
        extract_frame(
            event(&workspace, 1, SfmEventTransition::Discarded),
            &config,
            [0.0; 3],
        )
        .has_detail()
    );
    assert!(
        extract_frame(
            event(&workspace, 1, SfmEventTransition::Completed),
            &config,
            [0.0; 3],
        )
        .has_detail()
    );
    let global = SfmEvent::new(
        1,
        SfmEventBranch::GlobalCandidate {
            attempt: 0,
            candidate: openmvg_sfm::events::GlobalCandidate::GlobalPositioning,
        },
        SfmEventPhase::GlobalTranslationAveraging,
        SfmEventTransition::Checkpoint,
        SfmSnapshot::from_workspace(&workspace),
    );
    assert!(extract_frame(global, &config, [0.0; 3]).has_detail());
}

#[test]
fn seed_and_registrations_always_capture_growth() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        geometry: GeometryCapture::Every(NonZeroU64::new(100).unwrap()),
        ..RerunRecorderConfig::default()
    };
    let seed = SfmEvent::new(
        1,
        SfmEventBranch::Main,
        SfmEventPhase::IncrementalInitialPair,
        SfmEventTransition::Committed,
        SfmSnapshot::from_workspace(&workspace),
    );
    let seed = extract_frame(seed, &config, [0.0; 3]);
    assert!(seed.has_geometry());
    assert!(seed.has_detail());

    let registration = extract_frame(
        event(&workspace, 2, SfmEventTransition::Committed),
        &config,
        [0.0; 3],
    );
    assert!(registration.has_geometry());
    assert!(!registration.has_detail());
    assert_eq!(registration.geometry.unwrap().landmarks, None);
}

#[test]
fn stride_one_keeps_full_detail_for_registrations() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        geometry: GeometryCapture::Every(NonZeroU64::MIN),
        ..RerunRecorderConfig::default()
    };
    let registration = extract_frame(
        event(&workspace, 3, SfmEventTransition::Committed),
        &config,
        [0.0; 3],
    );
    assert!(registration.has_geometry());
    assert!(registration.has_detail());
}

#[test]
fn tentative_geometry_can_be_disabled() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        include_tentative_geometry: false,
        ..RerunRecorderConfig::default()
    };
    assert!(
        !extract_frame(
            event(&workspace, 0, SfmEventTransition::Tentative),
            &config,
            [0.0; 3],
        )
        .has_geometry()
    );
}

#[test]
fn disabled_geometry_capture_keeps_events_metrics_only() {
    let workspace = fixture();
    let config = RerunRecorderConfig {
        geometry: GeometryCapture::Disabled,
        ..RerunRecorderConfig::default()
    };
    let frame = extract_frame(
        event(&workspace, 0, SfmEventTransition::Committed),
        &config,
        [0.0; 3],
    );
    assert!(!frame.has_geometry());
    assert!(!frame.has_detail());
}

#[test]
fn finite_conversion_rejects_nan_infinity_and_f32_overflow() {
    assert_eq!(finite_point3(1.0, 2.0, 3.0), Some([1.0, 2.0, 3.0]));
    assert!(finite_point3(f64::NAN, 0.0, 0.0).is_none());
    assert!(finite_point3(f64::INFINITY, 0.0, 0.0).is_none());
    assert!(finite_point3(f64::MAX, 0.0, 0.0).is_none());
}

#[test]
fn recentering_preserves_local_detail_at_large_world_coordinates() {
    let origin = [1.0e12, -2.0e12, 3.0e12];
    assert_eq!(
        finite_recentered_point3(1.0e12 + 1.0, -2.0e12 + 2.0, 3.0e12 + 3.0, origin),
        Some([1.0, 2.0, 3.0])
    );
    let workspace = fixture();
    assert_eq!(
        first_finite_camera_center(SfmSnapshot::from_workspace(&workspace)),
        Some([0.0; 3])
    );
}

#[test]
fn deterministic_sampling_is_bounded_and_spans_the_input() {
    let selected = (0..10)
        .filter(|&ordinal| selected_sample(ordinal, 10, 3))
        .collect::<Vec<_>>();
    assert_eq!(selected, vec![3, 6, 9]);
    assert!((0..10).all(|ordinal| !selected_sample(ordinal, 10, 0)));
}
