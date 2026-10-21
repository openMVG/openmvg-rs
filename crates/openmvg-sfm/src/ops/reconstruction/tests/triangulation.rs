// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Track triangulation, rejection reports, and the triangulator trait boundary.

use super::*;

struct MalformedSupportTriangulator;

impl Triangulator for MalformedSupportTriangulator {
    fn triangulate(
        &self,
        _observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Ok(TriangulatedLandmark {
            point: Point3::new(0.0, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::ZERO,
            supporting_observation_indices: vec![0, 3].into_boxed_slice(),
        }
        .into())
    }
}

struct RejectingTriangulator(TriangulationRejectReason);

impl Triangulator for RejectingTriangulator {
    fn triangulate(
        &self,
        _observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Ok(TriangulationVerdict::Rejected(self.0))
    }
}

#[test]
fn global_translation_keeps_three_view_policy_explicit() {
    let mut workspace = two_missing_track_workspace();

    let report = triangulate_global_translation(
        &mut workspace,
        &AcceptingTriangulator,
        GaugeScalePolicy::Free,
    )
    .expect("global bootstrap should apply the three-view policy");

    assert_eq!(report.min_observations, 3);
    assert_eq!(report.triangulated_landmarks, 0);
    assert_eq!(workspace.landmarks().active_count(), 0);
}

struct HighPixelErrorTriangulator;

impl Triangulator for HighPixelErrorTriangulator {
    fn triangulate(
        &self,
        observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Ok(TriangulatedLandmark {
            point: Point3::new(0.0, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::try_new(100.0).unwrap(),
            supporting_observation_indices: (0..observations.len())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
        .into())
    }
}

#[test]
fn generalized_tracks_validate_geometry_instead_of_reported_pixel_summary() {
    let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let point = Point3::new(0.0, 0.0, 5.0);
    let poses = [
        openmvg_core::model::WorldToCameraPose::default(),
        openmvg_core::model::WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
    ];
    let posed = PosedTrack {
        observations: poses
            .iter()
            .enumerate()
            .map(|(index, pose)| PosedObservation {
                view: ViewId::new(index),
                xy: camera
                    .project(&pose.project_to_camera(&point).coords)
                    .unwrap(),
                pose,
                camera: &camera,
            })
            .collect(),
        source_indices: vec![0, 1],
    };

    assert!(matches!(
        triangulate_posed_with_limit(&HighPixelErrorTriangulator, &posed, 2, 1.0).unwrap(),
        PosedTriangulationOutcome::Accepted(_)
    ));
}

#[test]
fn generalized_tracks_reject_supported_angular_outliers() {
    let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let point = Point3::new(0.0, 0.0, 5.0);
    let poses = [
        openmvg_core::model::WorldToCameraPose::default(),
        openmvg_core::model::WorldToCameraPose::from_rotation_center(
            Rotation3::identity(),
            Point3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
    ];
    let first = camera
        .project(&poses[0].project_to_camera(&point).coords)
        .unwrap();
    let mut second = camera
        .project(&poses[1].project_to_camera(&point).coords)
        .unwrap();
    second.x += 20.0;
    let posed = PosedTrack {
        observations: vec![
            PosedObservation {
                view: ViewId::new(0),
                xy: first,
                pose: &poses[0],
                camera: &camera,
            },
            PosedObservation {
                view: ViewId::new(1),
                xy: second,
                pose: &poses[1],
                camera: &camera,
            },
        ],
        source_indices: vec![0, 1],
    };

    assert!(matches!(
        triangulate_posed_with_limit(&AcceptingTriangulator, &posed, 2, 1.0).unwrap(),
        PosedTriangulationOutcome::Rejected(TriangulationRejectReason::ReprojectionLimit)
    ));
}

#[test]
fn mixed_camera_tracks_validate_each_supporting_camera() {
    let perspective = Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap();
    let spherical = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let pose = openmvg_core::model::WorldToCameraPose::default();
    let point = Point3::new(0.0, 0.0, 5.0);
    let spherical_observation = spherical.project(&point.coords).unwrap();
    let posed = PosedTrack {
        observations: vec![
            PosedObservation {
                view: ViewId::new(0),
                xy: Point2::new(40.0, 24.0),
                pose: &pose,
                camera: &perspective,
            },
            PosedObservation {
                view: ViewId::new(1),
                xy: spherical_observation,
                pose: &pose,
                camera: &spherical,
            },
        ],
        source_indices: vec![0, 1],
    };

    assert!(matches!(
        triangulate_posed_with_limit(&AcceptingTriangulator, &posed, 2, 4.0).unwrap(),
        PosedTriangulationOutcome::Rejected(TriangulationRejectReason::ReprojectionLimit)
    ));
}

#[test]
fn mixed_camera_limit_ignores_observations_outside_robust_support() {
    let perspective = Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap();
    let spherical = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
    let pose = openmvg_core::model::WorldToCameraPose::default();
    let point = Point3::new(0.0, 0.0, 5.0);
    let posed = PosedTrack {
        observations: vec![
            PosedObservation {
                view: ViewId::new(0),
                xy: perspective.project(&point.coords).unwrap(),
                pose: &pose,
                camera: &perspective,
            },
            PosedObservation {
                view: ViewId::new(1),
                xy: spherical.project(&point.coords).unwrap(),
                pose: &pose,
                camera: &spherical,
            },
            PosedObservation {
                view: ViewId::new(2),
                xy: Point2::new(100.0, 100.0),
                pose: &pose,
                camera: &perspective,
            },
        ],
        source_indices: vec![0, 1, 2],
    };

    assert!(matches!(
        triangulate_posed_with_limit(&FirstTwoObservationsTriangulator, &posed, 3, 4.0).unwrap(),
        PosedTriangulationOutcome::Accepted(_)
    ));
}

struct FailingTriangulator;

impl Triangulator for FailingTriangulator {
    fn triangulate(
        &self,
        _observations: &[PosedPixelObservation<'_>],
    ) -> Result<TriangulationVerdict> {
        Err(SfmError::InvalidConfig("broken fixture".to_owned()))
    }
}

#[test]
fn global_report_attributes_expected_rejections() {
    let mut workspace = supported_workspace();
    let report = triangulate(
        &mut workspace,
        &RejectingTriangulator(TriangulationRejectReason::InsufficientParallax),
        GaugeScalePolicy::Free,
    )
    .expect("expected rejection is not a pipeline error");

    assert_eq!(report.attempted_tracks, 1);
    assert_eq!(report.triangulated_landmarks, 0);
    assert_eq!(report.rejections.total(), 1);
    assert_eq!(
        report
            .rejections
            .count(TriangulationRejectReason::InsufficientParallax),
        1
    );
    assert_eq!(
        report.attempted_tracks,
        report.triangulated_landmarks + report.rejections.total()
    );
    let json = serde_json::to_value(&report.rejections).expect("summary serializes");
    assert_eq!(json["counts"]["insufficient_parallax"], 1);
}

#[test]
fn triangulator_contract_errors_are_not_reported_as_rejections() {
    let mut workspace = supported_workspace();
    assert!(matches!(
        triangulate(&mut workspace, &FailingTriangulator, GaugeScalePolicy::Free),
        Err(SfmError::InvalidConfig(_))
    ));
}

#[test]
fn incremental_reports_separate_growth_and_refit_rejections() {
    let mut growing = supported_workspace();
    growing.landmarks = Landmarks::default();
    let growth = triangulate_incremental(
        &mut growing,
        &RejectingTriangulator(TriangulationRejectReason::NoValidHypothesis),
        2,
        f64::INFINITY,
    )
    .expect("expected rejection is not a pipeline error");
    assert_eq!(growth.attempted_tracks, 1);
    assert_eq!(growth.new_landmarks, 0);
    assert_eq!(
        growth
            .rejections
            .count(TriangulationRejectReason::NoValidHypothesis),
        1
    );

    let mut refitting = supported_workspace();
    let refit = retriangulate_incremental(
        &mut refitting,
        &RejectingTriangulator(TriangulationRejectReason::NumericalFailure),
        2,
        f64::INFINITY,
        None,
    )
    .expect("expected rejection is not a pipeline error");
    assert_eq!(refit.attempted_refits, 1);
    assert_eq!(refit.dropped_landmarks, 1);
    assert_eq!(
        refit
            .refit_rejections
            .count(TriangulationRejectReason::NumericalFailure),
        1
    );
    assert_eq!(refit.attempted_new_tracks, 0);
    assert!(refit.growth_rejections.is_empty());
}

#[test]
fn malformed_triangulator_support_is_not_silently_skipped() {
    let workspace = supported_workspace();
    let track = workspace
        .tracks()
        .get(TrackId::new(0))
        .expect("track exists");
    let posed = gather_posed(&workspace, track, PosedObservationPolicy::All)
        .expect("gather succeeds")
        .expect("track is posed");

    assert!(matches!(
        triangulate_posed(
            &MalformedSupportTriangulator,
            &posed,
            track.observations().len()
        ),
        Err(SfmError::InvalidInput(_))
    ));
}

#[test]
fn malformed_triangulator_geometry_is_rejected_at_the_trait_boundary() {
    for point in [
        Point3::new(f64::NAN, 0.0, 5.0),
        Point3::new(0.0, f64::INFINITY, 5.0),
    ] {
        let triangulated = TriangulatedLandmark {
            point,
            reprojection_error: openmvg_core::PixelResidual::ZERO,
            supporting_observation_indices: vec![0, 1].into_boxed_slice(),
        };
        assert!(matches!(
            triangulated.validate(2),
            Err(SfmError::InvalidInput(_))
        ));
    }
    for reprojection_error in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(openmvg_core::PixelResidual::try_new(reprojection_error).is_err());
    }
    for support in [vec![0], vec![1, 0], vec![0, 0], vec![0, 2]] {
        let triangulated = TriangulatedLandmark {
            point: Point3::new(0.0, 0.0, 5.0),
            reprojection_error: openmvg_core::PixelResidual::ZERO,
            supporting_observation_indices: support.into_boxed_slice(),
        };
        assert!(matches!(
            triangulated.validate(2),
            Err(SfmError::InvalidInput(_))
        ));
    }
}

#[test]
fn invalid_reprojection_limit_is_a_config_error() {
    for limit in [f64::NAN, -1.0] {
        let mut workspace = supported_workspace();
        let before = workspace.landmarks().clone();

        assert!(matches!(
            triangulate_incremental(&mut workspace, &AcceptingTriangulator, 2, limit),
            Err(SfmError::InvalidConfig(_))
        ));
        assert_eq!(workspace.landmarks(), &before);
    }
}
