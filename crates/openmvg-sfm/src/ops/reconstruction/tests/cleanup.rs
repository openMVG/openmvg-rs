// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Landmark, pose-support, underdetermined-pose, and bundle-component cleanup.

use super::*;

fn disconnected_bundle_support_workspace() -> SfmWorkspace {
    let centers = [0.0, 0.5, 1.0, 4.0, 4.5];
    // The two dense lobes share pose 2, so the incidence graph is connected but not
    // vertex-biconnected. Each lobe has its own otherwise-free scale.
    let track_views = [vec![0, 1, 2], vec![0, 1, 2], vec![2, 3, 4], vec![2, 3, 4]];
    let points = [
        Point3::new(-0.2, 0.0, 5.0),
        Point3::new(0.3, 0.2, 6.0),
        Point3::new(4.1, -0.1, 5.5),
        Point3::new(4.4, 0.2, 6.5),
    ];
    let tracks = TrackSet::try_new(
        track_views
            .iter()
            .enumerate()
            .map(|(track, views)| {
                Track::try_new(
                    TrackId::new(track),
                    views
                        .iter()
                        .map(|&view| TrackObservation {
                            view: ViewId::new(view),
                            feature: FeatureId::new(track),
                        })
                        .collect(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
    )
    .unwrap();
    let features: Vec<_> = centers
        .iter()
        .map(|&center_x| {
            FeatureSet::try_new(
                points
                    .iter()
                    .map(|point| {
                        Point2::new(
                            100.0 * (point.x - center_x) / point.z + 32.0,
                            100.0 * point.y / point.z + 24.0,
                        )
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..centers.len())
                .map(|image| ImageMeta::external_id(format!("image-{image}"), 64, 48))
                .collect::<Vec<_>>(),
        )
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .tracks(tracks, features)
        .build()
        .unwrap();
    for (image, center_x) in centers.into_iter().enumerate() {
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
            .unwrap();
    }
    workspace.landmarks = Landmarks::try_new(
        points
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                Landmark::active(
                    LandmarkId::new(index),
                    TrackId::new(index),
                    point,
                    openmvg_core::PixelResidual::ZERO,
                    ObservationSupport::default(),
                )
            })
            .collect(),
    )
    .unwrap();
    workspace.components = vec![ReconstructionComponent::new(
        ComponentId::new(0),
        (0..centers.len()).map(ViewId::new).collect(),
        (0..points.len()).map(LandmarkId::new).collect(),
        GaugeScalePolicy::Free,
    )];
    workspace
}

#[test]
fn bundle_component_cleanup_removes_an_articulation_joined_scale_lobe() {
    let mut workspace = disconnected_bundle_support_workspace();

    let report = retain_primary_bundle_component(&mut workspace)
        .expect("free-scale preparation should retain one deterministic BA component");

    assert_eq!(report.removed_views, vec![ViewId::new(3), ViewId::new(4)]);
    assert_eq!(report.rejected_landmarks, 2);
    assert_eq!(workspace.registered_view_count(), 3);
    assert_eq!(workspace.landmarks().active_count(), 2);
    assert_eq!(
        workspace.components()[0].views,
        vec![ViewId::new(0), ViewId::new(1), ViewId::new(2)]
    );
}

#[test]
fn posed_support_refresh_drops_unposed_views_from_surviving_landmarks() {
    // Peeling a view must not leave it as support of a landmark that keeps two posed
    // observations: such a scene fails validation when it is exported.
    let mut workspace = disconnected_bundle_support_workspace();
    workspace.poses.take(4).unwrap();

    let rejected = retain_posed_landmark_support(&mut workspace).unwrap();

    assert_eq!(rejected, 0, "tracks 2 and 3 keep views 2 and 3");
    assert_eq!(workspace.landmarks().active_count(), 4);
    for landmark in workspace.landmarks().iter_active() {
        let track = workspace.tracks().get(landmark.source_track).unwrap();
        assert!(
            landmark
                .supporting_observations(track)
                .all(|(_, observation)| workspace.has_pose(observation.view)),
            "landmark {} lists an unposed view as support",
            landmark.id
        );
    }
}

#[test]
fn bundle_component_cleanup_breaks_equal_size_ties_by_stable_view_id() {
    let mut workspace = disconnected_bundle_support_workspace();
    workspace.poses.take(2).unwrap();

    let report = retain_primary_bundle_component(&mut workspace)
        .expect("equal-size components should have a deterministic winner");

    assert_eq!(report.removed_views, vec![ViewId::new(3), ViewId::new(4)]);
    assert!(workspace.has_pose(ViewId::new(0)));
    assert!(workspace.has_pose(ViewId::new(1)));
}

fn shared_pose_bridge_workspace() -> SfmWorkspace {
    // Views 1 and 2 share pose 1. Both landmarks therefore span all three effective pose
    // variables even though they use different views of the shared pose.
    let track_views = [vec![0, 1, 3], vec![0, 2, 3]];
    let points = [Point3::new(0.0, 0.0, 5.0), Point3::new(1.0, 0.0, 5.0)];
    let tracks = TrackSet::try_new(
        track_views
            .iter()
            .enumerate()
            .map(|(track, views)| {
                Track::try_new(
                    TrackId::new(track),
                    views
                        .iter()
                        .map(|&view| TrackObservation {
                            view: ViewId::new(view),
                            feature: FeatureId::new(track),
                        })
                        .collect(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
    )
    .unwrap();
    let features = (0..4)
        .map(|_| FeatureSet::try_new(vec![Point2::new(32.0, 24.0); 2]).unwrap())
        .collect::<Vec<_>>();
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..4)
                .map(|image| ImageMeta::external_id(format!("image-{image}"), 64, 48))
                .collect::<Vec<_>>(),
        )
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .view_poses([
            crate::PoseId::new(0),
            crate::PoseId::new(1),
            crate::PoseId::new(1),
            crate::PoseId::new(2),
        ])
        .tracks(tracks, features)
        .build()
        .unwrap();
    for pose in 0..3 {
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(pose),
                pose: openmvg_core::model::WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(pose as f64, 0.0, 0.0),
                )
                .unwrap(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
    }
    workspace.landmarks = Landmarks::try_new(
        points
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                Landmark::active(
                    LandmarkId::new(index),
                    TrackId::new(index),
                    point,
                    openmvg_core::PixelResidual::ZERO,
                    ObservationSupport::default(),
                )
            })
            .collect(),
    )
    .unwrap();
    workspace.components = vec![ReconstructionComponent::new(
        ComponentId::new(0),
        (0..4).map(ViewId::new).collect(),
        (0..2).map(LandmarkId::new).collect(),
        GaugeScalePolicy::Free,
    )];
    workspace
}

#[test]
fn bundle_component_cleanup_treats_shared_pose_views_as_one_variable() {
    let mut workspace = shared_pose_bridge_workspace();

    let report = retain_primary_bundle_component(&mut workspace)
        .expect("shared-pose views should bridge their landmark islands");

    assert!(report.is_empty());
    assert_eq!(workspace.registered_view_count(), 4);
    assert_eq!(workspace.landmarks().active_count(), 2);
}

#[test]
fn pose_support_cleanup_aggregates_observations_across_shared_views() {
    let mut workspace = shared_pose_bridge_workspace();

    let report = remove_unsupported_poses(
        &mut workspace,
        &[ViewId::new(0), ViewId::new(3)],
        2,
        f64::INFINITY,
    )
    .expect("two observations on views sharing a pose constrain that one pose variable");

    assert!(report.is_empty());
    assert_eq!(workspace.registered_view_count(), 4);
}

#[test]
fn pose_support_cleanup_counts_a_shared_landmark_once_per_pose() {
    let mut workspace = shared_pose_bridge_workspace();
    workspace.tracks = TrackSet::try_new(vec![
        Track::try_new(
            TrackId::new(0),
            [0, 1, 2]
                .into_iter()
                .map(|view| TrackObservation {
                    view: ViewId::new(view),
                    feature: FeatureId::new(0),
                })
                .collect(),
        )
        .unwrap(),
        Track::try_new(
            TrackId::new(1),
            [0, 3]
                .into_iter()
                .map(|view| TrackObservation {
                    view: ViewId::new(view),
                    feature: FeatureId::new(1),
                })
                .collect(),
        )
        .unwrap(),
    ])
    .unwrap();

    let report = remove_unsupported_poses(
        &mut workspace,
        &[ViewId::new(0), ViewId::new(3)],
        2,
        f64::INFINITY,
    )
    .unwrap();

    assert_eq!(report.removed_views, vec![ViewId::new(1), ViewId::new(2)]);
    assert_eq!(report.rejected_landmarks, 1);
    assert!(!workspace.has_pose(ViewId::new(1)));
    assert!(!workspace.has_pose(ViewId::new(2)));
}

#[test]
fn bundle_component_cleanup_rejects_a_split_fixed_baseline() {
    let mut workspace = disconnected_bundle_support_workspace();
    workspace.components = vec![ReconstructionComponent::new(
        ComponentId::new(0),
        (0..5).map(ViewId::new).collect(),
        (0..4).map(LandmarkId::new).collect(),
        GaugeScalePolicy::FixedBaseline {
            first: ViewId::new(0),
            second: ViewId::new(4),
            distance: 4.5,
            unit: crate::config::LengthUnit::Arbitrary,
        },
    )];

    let result = retain_primary_bundle_component(&mut workspace);

    assert!(matches!(result, Err(SfmError::InvalidInput(message)) if message.contains("anchors")));
    assert_eq!(workspace.registered_view_count(), 5);
    assert_eq!(workspace.landmarks().active_count(), 4);
}

#[test]
fn landmark_cleaning_does_not_reintroduce_rejected_observations() {
    let mut workspace = supported_workspace();
    let cleanup = clean_outlier_landmark_support(&mut workspace, 1.0, 0.0)
        .expect("supported geometry should validate");

    assert!(cleanup.is_empty());
    assert_eq!(workspace.landmarks().active_count(), 1);
    let support = &workspace.landmarks().as_slice()[0].observation_support;
    assert!(support.contains(0));
    assert!(!support.contains(1));
    assert!(support.contains(2));
}

#[test]
fn landmark_cleaning_prunes_one_bad_observation_without_discarding_the_track() {
    let mut workspace = supported_workspace();
    workspace.landmarks.as_mut_slice()[0].observation_support = ObservationSupport::default();

    let cleanup = clean_outlier_landmark_support(&mut workspace, 4.0, 0.0)
        .expect("two geometrically valid observations should preserve the landmark");

    assert_eq!(cleanup.rejected_landmarks, 0);
    assert_eq!(cleanup.removed_observations, 1);
    let landmark = &workspace.landmarks().as_slice()[0];
    assert_eq!(landmark.status, LandmarkStatus::Active);
    assert!(landmark.observation_support.contains(0));
    assert!(!landmark.observation_support.contains(1));
    assert!(landmark.observation_support.contains(2));

    let support = landmark.observation_support.clone();
    let second = clean_outlier_landmark_support(&mut workspace, 4.0, 0.0)
        .expect("already-clean support should remain stable");
    assert!(second.is_empty());
    assert_eq!(
        workspace.landmarks().as_slice()[0].observation_support,
        support
    );
}

fn spherical_outlier_workspace() -> SfmWorkspace {
    let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let point = Point3::new(0.0, 0.0, 5.0);
    let exact = camera.project(&point.coords).unwrap();
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..3).map(|index| ImageMeta::external_id(format!("spherical-{index}"), 1024, 512)),
        )
        .camera_models([camera])
        .shared_intrinsic(IntrinsicId::new(0))
        .tracks(
            TrackSet::try_new(vec![
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
                .unwrap(),
            ])
            .unwrap(),
            [
                FeatureSet::try_new(vec![exact]).unwrap(),
                FeatureSet::try_new(vec![exact]).unwrap(),
                FeatureSet::try_new(vec![Point2::new(exact.x + 100.0, exact.y)]).unwrap(),
            ],
        )
        .build()
        .unwrap();
    for image in 0..3 {
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(image),
                pose: openmvg_core::model::WorldToCameraPose::default(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
    }
    workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
        LandmarkId::new(0),
        TrackId::new(0),
        point,
        openmvg_core::PixelResidual::ZERO,
        ObservationSupport::default(),
    )])
    .unwrap();
    workspace
}

#[test]
fn spherical_landmark_cleanup_prunes_finite_angular_outlier() {
    let mut workspace = spherical_outlier_workspace();

    let cleanup = clean_outlier_landmark_support(&mut workspace, 4.0, 0.0).unwrap();

    assert_eq!(cleanup.rejected_landmarks, 0);
    assert_eq!(cleanup.removed_observations, 1);
    let support = &workspace.landmarks().as_slice()[0].observation_support;
    assert!(support.contains(0));
    assert!(support.contains(1));
    assert!(!support.contains(2));
}

#[test]
fn spherical_pose_cleanup_does_not_count_finite_angular_outlier() {
    let mut workspace = spherical_outlier_workspace();

    let cleanup =
        remove_unsupported_poses(&mut workspace, &[ViewId::new(0), ViewId::new(1)], 1, 4.0)
            .unwrap();

    assert_eq!(cleanup.removed_views, vec![ViewId::new(2)]);
    assert!(!workspace.pose_store().has(2));
}

#[test]
fn spherical_near_pole_error_is_reported_in_nominal_pixels() {
    let spherical = SphericalCamera::try_new(1024).unwrap();
    let camera = Camera::Spherical(spherical.clone());
    let latitude = 89.9_f64.to_radians();
    // The camera's convention: longitude about the Y axis, positive latitude pointing down.
    let direction = |longitude: f64| {
        let (sin, cos) = latitude.sin_cos();
        nalgebra::Vector3::new(cos * longitude.sin(), sin, cos * longitude.cos())
    };
    let point = Point3::from(direction(0.0) * 5.0);
    let exact = camera.project(&point.coords).unwrap();
    let shifted_direction = direction(0.6);
    let shifted = camera.project(&shifted_direction).unwrap();
    assert!(camera_pixel_residual(&camera, exact, shifted) > 90.0);

    let policy = CameraResidualPolicy::from_pixel_limit(&camera, 4.0).unwrap();
    let evaluation = policy.evaluate(&camera, &point.coords, shifted).unwrap();
    assert!(evaluation.accepted);
    assert!(evaluation.pixel_equivalent < 1.0);

    let mut workspace = spherical_outlier_workspace();
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
        .unwrap(),
    ])
    .unwrap();
    set_feature_pixels(
        &mut workspace,
        vec![vec![exact], vec![exact], vec![shifted]],
    );
    workspace.landmarks.as_mut_slice()[0].point = point;
    workspace.landmarks.as_mut_slice()[0].observation_support = ObservationSupport::default();

    let cleanup = clean_outlier_landmark_support(&mut workspace, 4.0, 0.0).unwrap();
    assert!(cleanup.is_empty());
    assert!(
        workspace.landmarks().as_slice()[0]
            .reprojection_error
            .value()
            < 1.0
    );

    workspace.landmarks.as_mut_slice()[0].reprojection_error =
        openmvg_core::PixelResidual::try_new(100.0).unwrap();
    let refresh = crate::ops::reconstruction::bundle::refresh_landmark_reprojection_errors_for(
        &mut workspace,
        &[LandmarkId::new(0)],
    )
    .unwrap();
    assert!(refresh.is_empty());
    assert!(
        workspace.landmarks().as_slice()[0]
            .reprojection_error
            .value()
            < 1.0
    );
}

#[test]
fn landmark_cleanup_checks_apex_angle_after_pruning() {
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
                TrackObservation {
                    view: ViewId::new(2),
                    feature: FeatureId::new(0),
                },
            ],
        )
        .expect("valid test track"),
    ])
    .expect("valid track collection");
    set_feature_pixels(
        &mut workspace,
        vec![
            vec![Point2::new(32.0, 24.0)],
            vec![Point2::new(22.0, 24.0)],
            vec![Point2::new(500.0, -300.0)],
        ],
    );
    workspace.landmarks.as_mut_slice()[0].observation_support = ObservationSupport::default();

    let cleanup = clean_outlier_landmark_support(&mut workspace, 4.0, 8.0)
        .expect("cleanup should evaluate angle after removing the wide-baseline outlier");

    assert_eq!(cleanup.rejected_landmarks, 1);
    assert_eq!(workspace.landmarks().active_count(), 0);
}

#[test]
fn pose_support_cleanup_removes_only_unprotected_unsupported_pose() {
    let mut workspace = supported_workspace();
    let report = remove_unsupported_poses(&mut workspace, &[ViewId::new(0)], 1, 4.0)
        .expect("pose support audit should succeed");

    assert_eq!(report.removed_views, vec![ViewId::new(1)]);
    assert_eq!(report.rejected_landmarks, 0);
    assert!(workspace.pose_store().has(0));
    assert!(!workspace.pose_store().has(1));
    assert!(workspace.pose_store().has(2));
    assert_eq!(workspace.landmarks().active_count(), 1);
}

#[test]
fn pose_support_cleanup_rejects_landmarks_left_with_one_pose() {
    let mut workspace = supported_workspace();
    let report = remove_unsupported_poses(&mut workspace, &[ViewId::new(0)], 2, 4.0)
        .expect("pose support audit should reach a stable fixed point");

    assert_eq!(report.removed_views, vec![ViewId::new(1), ViewId::new(2)]);
    assert_eq!(report.rejected_landmarks, 1);
    assert_eq!(workspace.pose_store().populated_count(), 1);
    assert_eq!(workspace.landmarks().active_count(), 0);
}

/// Views on a small baseline observe one landmark per track, with every view's pose posed.
fn bearing_workspace(track_views: &[&[usize]]) -> SfmWorkspace {
    let centers = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.5, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 0.5, 0.0),
    ];
    let view_count = track_views
        .iter()
        .flat_map(|views| views.iter())
        .max()
        .unwrap()
        + 1;
    let points = (0..track_views.len())
        .map(|index| {
            let index = index as f64;
            Point3::new(-0.2 + 0.15 * index, 0.1 * (index % 3.0), 5.0 + 0.3 * index)
        })
        .collect::<Vec<_>>();
    let tracks = TrackSet::try_new(
        track_views
            .iter()
            .enumerate()
            .map(|(track, views)| {
                Track::try_new(
                    TrackId::new(track),
                    views
                        .iter()
                        .map(|&view| TrackObservation {
                            view: ViewId::new(view),
                            feature: FeatureId::new(track),
                        })
                        .collect(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
    )
    .unwrap();
    let features = centers[..view_count]
        .iter()
        .map(|center| {
            FeatureSet::try_new(
                points
                    .iter()
                    .map(|point| {
                        let camera_point = point - center;
                        Point2::new(
                            100.0 * camera_point.x / camera_point.z + 32.0,
                            100.0 * camera_point.y / camera_point.z + 24.0,
                        )
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..view_count)
                .map(|image| ImageMeta::external_id(format!("image-{image}"), 64, 48))
                .collect::<Vec<_>>(),
        )
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .tracks(tracks, features)
        .build()
        .unwrap();
    for (image, center) in centers[..view_count].iter().enumerate() {
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(image),
                pose: openmvg_core::model::WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    *center,
                )
                .unwrap(),
                component: Some(ComponentId::new(0)),
            })
            .unwrap();
    }
    workspace.landmarks = Landmarks::try_new(
        points
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                Landmark::active(
                    LandmarkId::new(index),
                    TrackId::new(index),
                    point,
                    openmvg_core::PixelResidual::ZERO,
                    ObservationSupport::default(),
                )
            })
            .collect(),
    )
    .unwrap();
    refresh_primary_component(&mut workspace);
    workspace
}

#[test]
fn underdetermined_pose_cleanup_bounds_what_one_neighbour_can_fix() {
    const CORE: [&[usize]; 3] = [&[0, 1, 2], &[0, 1, 2], &[0, 1, 2]];

    // Landmarks shared only with view 0 fix at most the direction from it.
    let mut workspace = bearing_workspace(&[CORE[0], CORE[1], CORE[2], &[0, 3], &[0, 3], &[0, 3]]);
    let report = remove_underdetermined_poses(&mut workspace, 3).unwrap();
    assert_eq!(report.removed_views, vec![ViewId::new(3)]);
    assert_eq!(report.rejected_landmarks, 3);
    assert_eq!(workspace.registered_view_count(), 3);

    // One such landmark per neighbor fixes a translation, not a released pose.
    let mut workspace = bearing_workspace(&[CORE[0], CORE[1], CORE[2], &[0, 3], &[1, 3], &[2, 3]]);
    assert!(
        remove_underdetermined_poses(&mut workspace, 3)
            .unwrap()
            .is_empty()
    );
    let report = remove_underdetermined_poses(&mut workspace, 6).unwrap();
    assert_eq!(report.removed_views, vec![ViewId::new(3)]);
    assert_eq!(workspace.landmarks().active_count(), 3);

    // Landmarks the other poses determine give two equations each.
    let mut workspace = bearing_workspace(&[CORE[0], CORE[1], CORE[2], &[0, 1, 3], &[0, 1, 3]]);
    assert!(
        remove_underdetermined_poses(&mut workspace, 3)
            .unwrap()
            .is_empty()
    );
    let report = remove_underdetermined_poses(&mut workspace, 6).unwrap();
    assert_eq!(report.removed_views, vec![ViewId::new(3)]);
    assert_eq!(workspace.landmarks().active_count(), 5);
}

#[test]
fn underdetermined_pose_cleanup_never_leaves_fewer_than_three_poses() {
    // With views 0 and 2 removed, view 1 alone would remain; two views would own the scale.
    let mut workspace = bearing_workspace(&[&[0, 1], &[0, 1], &[1, 2]]);

    let report = remove_underdetermined_poses(&mut workspace, 3).unwrap();

    assert!(report.is_empty());
    assert_eq!(workspace.registered_view_count(), 3);
}
