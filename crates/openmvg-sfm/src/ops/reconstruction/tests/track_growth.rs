// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental retriangulation, refits, revivals, and missing-track recovery.

use super::*;

fn support_extension_workspace() -> SfmWorkspace {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("a", 64, 48),
            ImageMeta::external_id("b", 64, 48),
            ImageMeta::external_id("c", 64, 48),
            ImageMeta::external_id("d", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .build()
        .expect("valid workspace");
    for (image, center_x) in [0.0, 0.5, 1.0, 1.5].into_iter().enumerate() {
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
                TrackObservation {
                    view: ViewId::new(3),
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
            vec![Point2::new(2.0, 24.0)],
        ],
    );
    workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
        LandmarkId::new(0),
        TrackId::new(0),
        Point3::new(0.0, 0.0, 5.0),
        openmvg_core::PixelResidual::ZERO,
        ObservationSupport::try_from_indices([0, 2], 4).expect("valid selected support"),
    )])
    .expect("valid landmarks");
    workspace
}

struct SelectiveFailingTriangulator;

impl Triangulator for SelectiveFailingTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        if observations
            .first()
            .is_some_and(|observation| observation.xy.x == 33.0)
        {
            return Err(SfmError::InvalidConfig(
                "injected missing-track recovery failure".to_owned(),
            ));
        }
        AcceptingTriangulator.triangulate(observations)
    }
}

#[test]
fn rejected_track_retry_revives_one_canonical_landmark() {
    let mut workspace = supported_workspace();
    workspace
        .landmarks
        .get_mut(LandmarkId::new(0))
        .expect("landmark exists")
        .status = LandmarkStatus::Rejected;

    let growth = triangulate_incremental(&mut workspace, &AcceptingTriangulator, 2, f64::INFINITY)
        .expect("rejected track should be retried");
    assert_eq!(growth.new_landmarks, 0);
    assert_eq!(growth.revived_landmarks, 1);
    assert_eq!(growth.total_landmarks, 1);
    assert_eq!(workspace.landmarks().len(), 1);
    assert_eq!(workspace.landmarks().active_count(), 1);

    let refit = retriangulate_incremental(
        &mut workspace,
        &AcceptingTriangulator,
        2,
        f64::INFINITY,
        None,
    )
    .expect("canonical landmark should refit");
    assert_eq!(refit.refined_landmarks, 1);
    assert_eq!(refit.revived_landmarks, 0);
    assert_eq!(workspace.landmarks().len(), 1);
    assert_eq!(workspace.landmarks().active_count(), 1);
}

#[test]
fn final_missing_recovery_preserves_rejected_landmark_tombstones() {
    let mut workspace = supported_workspace();
    let original = workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark exists")
        .clone();
    workspace
        .landmarks
        .get_mut(LandmarkId::new(0))
        .expect("landmark exists")
        .status = LandmarkStatus::Rejected;

    let report = recover_missing_tracks(&mut workspace, &AcceptingTriangulator, 2, f64::INFINITY)
        .expect("missing-only recovery should succeed");

    assert_eq!(report.attempted_refits, 0);
    assert_eq!(report.attempted_new_tracks, 0);
    assert_eq!(report.revived_landmarks, 0);
    let landmark = workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark remains canonical");
    assert_eq!(landmark.status, LandmarkStatus::Rejected);
    assert_eq!(landmark.point, original.point);
    assert_eq!(landmark.observation_support, original.observation_support);
}

#[test]
fn final_missing_recovery_adds_only_unmaterialized_tracks() {
    let mut workspace = supported_workspace();
    workspace.landmarks = Landmarks::default();

    let report = recover_missing_tracks(&mut workspace, &AcceptingTriangulator, 2, f64::INFINITY)
        .expect("missing track should be recovered");

    assert_eq!(report.attempted_refits, 0);
    assert_eq!(report.attempted_new_tracks, 1);
    assert_eq!(report.new_landmarks, 1);
    assert_eq!(report.revived_landmarks, 0);
    assert_eq!(workspace.landmarks().active_count(), 1);
}

#[test]
fn final_missing_recovery_does_not_refit_active_points() {
    let mut workspace = supported_workspace();
    let preserved_point = Point3::new(1.0, 2.0, 7.0);
    workspace
        .landmarks
        .get_mut(LandmarkId::new(0))
        .expect("landmark exists")
        .point = preserved_point;

    let report = recover_missing_tracks(&mut workspace, &AcceptingTriangulator, 2, f64::INFINITY)
        .expect("missing-only recovery should succeed");

    assert_eq!(report.attempted_refits, 0);
    assert_eq!(report.attempted_new_tracks, 0);
    assert_eq!(
        workspace
            .landmarks()
            .get(LandmarkId::new(0))
            .expect("landmark exists")
            .point,
        preserved_point
    );
}

#[test]
fn missing_recovery_is_atomic_when_any_candidate_errors() {
    let mut workspace = two_missing_track_workspace();

    let error = recover_missing_tracks(
        &mut workspace,
        &SelectiveFailingTriangulator,
        2,
        f64::INFINITY,
    )
    .expect_err("one fatal candidate must reject the complete recovery pass");

    assert!(error.to_string().contains("missing-track recovery failure"));
    assert!(workspace.landmarks().is_empty());
}

#[test]
fn new_pose_updates_refit_existing_landmarks() {
    let mut workspace = supported_workspace();
    workspace
        .landmarks
        .get_mut(LandmarkId::new(0))
        .expect("landmark exists")
        .point = Point3::new(1.0, 2.0, 7.0);

    let report = retriangulate_incremental(
        &mut workspace,
        &AcceptingTriangulator,
        2,
        f64::INFINITY,
        Some(ViewId::new(2)),
    )
    .expect("new-pose update should refit touched geometry");

    assert_eq!(report.attempted_refits, 1);
    assert_eq!(report.refined_landmarks, 1);
    assert_eq!(report.attempted_new_tracks, 0);
    assert_eq!(
        workspace
            .landmarks()
            .get(LandmarkId::new(0))
            .expect("landmark exists")
            .point,
        Point3::new(0.0, 0.0, 5.0)
    );
    let support = &workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark exists")
        .observation_support;
    assert!(support.contains(0));
    assert!(!support.contains(1));
    assert!(support.contains(2));
    assert!(
        workspace
            .components()
            .first()
            .is_some_and(|component| component.views.contains(&ViewId::new(2))),
        "new-pose retriangulation must publish the pose even when landmark topology is unchanged"
    );
}

#[test]
fn new_pose_support_can_revive_without_restoring_old_outliers() {
    let mut workspace = support_extension_workspace();
    workspace
        .landmarks
        .get_mut(LandmarkId::new(0))
        .expect("landmark exists")
        .status = LandmarkStatus::Rejected;

    let report = retriangulate_incremental(
        &mut workspace,
        &AcceptingTriangulator,
        2,
        f64::INFINITY,
        Some(ViewId::new(3)),
    )
    .expect("new support should revive the rejected landmark");

    assert_eq!(report.attempted_refits, 1);
    assert_eq!(report.revived_landmarks, 1);
    let landmark = workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark exists");
    assert_eq!(landmark.status, LandmarkStatus::Active);
    assert!(landmark.observation_support.contains(0));
    assert!(!landmark.observation_support.contains(1));
    assert!(landmark.observation_support.contains(2));
    assert!(landmark.observation_support.contains(3));
}

#[test]
fn post_optimization_refit_does_not_readmit_rejected_new_observations() {
    let mut workspace = support_extension_workspace();

    retriangulate_incremental(
        &mut workspace,
        &FirstTwoObservationsTriangulator,
        2,
        f64::INFINITY,
        Some(ViewId::new(3)),
    )
    .expect("the first refit may evaluate the newly posed image");
    let support_after_robust_refit = workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark exists")
        .observation_support
        .clone();
    assert!(support_after_robust_refit.contains(0));
    assert!(!support_after_robust_refit.contains(1));
    assert!(support_after_robust_refit.contains(2));
    assert!(!support_after_robust_refit.contains(3));

    refit_incremental_for_views(
        &mut workspace,
        &AcceptingTriangulator,
        2,
        f64::INFINITY,
        &[ViewId::new(3)],
    )
    .expect("post-optimization refit should preserve robust support");

    assert_eq!(
        workspace
            .landmarks()
            .get(LandmarkId::new(0))
            .expect("landmark exists")
            .observation_support,
        support_after_robust_refit
    );
}

#[test]
fn full_refit_preserves_existing_observation_support() {
    let mut workspace = support_extension_workspace();

    retriangulate_incremental(
        &mut workspace,
        &AcceptingTriangulator,
        2,
        f64::INFINITY,
        None,
    )
    .expect("full refit should preserve robust support");

    let support = &workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark exists")
        .observation_support;
    assert!(support.contains(0));
    assert!(!support.contains(1));
    assert!(support.contains(2));
    assert!(!support.contains(3));
}

#[test]
fn growth_only_revival_does_not_readmit_old_rejected_observations() {
    let mut workspace = support_extension_workspace();
    workspace
        .landmarks
        .get_mut(LandmarkId::new(0))
        .expect("landmark exists")
        .status = LandmarkStatus::Rejected;

    let report = triangulate_incremental_for_views(
        &mut workspace,
        &FirstTwoObservationsTriangulator,
        2,
        f64::INFINITY,
        &[ViewId::new(3)],
    )
    .expect("growth-only revival should respect retained support");

    assert_eq!(report.revived_landmarks, 1);
    let support = &workspace
        .landmarks()
        .get(LandmarkId::new(0))
        .expect("landmark exists")
        .observation_support;
    assert!(support.contains(0));
    assert!(!support.contains(1));
    assert!(support.contains(2));
    assert!(!support.contains(3));
}

#[test]
fn growth_only_new_pose_refreshes_component_without_new_landmarks() {
    let mut workspace = supported_workspace();

    let report = triangulate_incremental_for_views(
        &mut workspace,
        &AcceptingTriangulator,
        2,
        f64::INFINITY,
        &[ViewId::new(2)],
    )
    .expect("growth-only new-pose update should refresh the component");

    assert_eq!(report.attempted_tracks, 0);
    assert_eq!(report.new_landmarks, 0);
    assert_eq!(report.revived_landmarks, 0);
    assert!(
        workspace
            .components()
            .first()
            .is_some_and(|component| component.views.contains(&ViewId::new(2)))
    );
}
