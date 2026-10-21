// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Mutex;

use nalgebra::{Point2, Point3, Rotation3, Vector3};
use openmvg_core::camera::CameraModel;
use openmvg_core::model::WorldToCameraPose;
use openmvg_sfm::testing::oracle::initialize_incremental_seed_from_oracle;
use openmvg_sfm::{
    config::*,
    model::*,
    reports::RegistrationFailureReason,
    testing::seams::*,
    unstable::*,
    unstable::{backends::*, incremental::*, types::*},
    *,
};

#[test]
fn genransac_registration_preserves_kernel_pose_convention() {
    let gt_pose = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.08, -0.04, 0.03),
        Point3::new(0.25, -0.15, -0.8),
    )
    .unwrap();
    let mut workspace = resection_workspace(gt_pose.clone());
    workspace.build_tracks().unwrap();

    initialize_incremental_seed_from_oracle(
        &mut workspace,
        &openmvg_sfm::testing::minimal_incremental_config().with_initial_pair(
            InitialPairPolicy::BestPair(InitialPairConfig::default().with_min_inliers(8)),
        ),
        &openmvg_sfm::unstable::evaluation::GroundTruthReconstruction {
            poses: vec![
                (ViewId::new(0), WorldToCameraPose::identity()),
                (
                    ViewId::new(1),
                    WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::new(1.0, 0.0, 0.0),
                    )
                    .unwrap(),
                ),
                (ViewId::new(2), gt_pose.clone()),
            ],
            points: world_points()
                .into_iter()
                .enumerate()
                .map(|(index, point)| (openmvg_sfm::LandmarkId::new(index), point))
                .collect(),
        },
    )
    .unwrap();

    let registrar = GenRansacP3pRegistrar::default();
    let outcome = register_view(
        &mut workspace,
        ViewId::new(2),
        &registrar,
        &RegistrationConfig::default(),
    )
    .unwrap();

    let RegistrationVerdict::Registered(report) = outcome else {
        panic!("expected image 2 to register, got {outcome:?}");
    };
    assert_eq!(report.inliers, 8);
    assert_eq!(report.focal_px, None);
    assert!(
        report.residual_median.value() <= registrar.inlier_angle_rad(),
        "median angular residual {} exceeded {} radians",
        report.residual_median.value(),
        registrar.inlier_angle_rad()
    );
    assert!(
        workspace
            .components()
            .first()
            .is_some_and(|component| component.views.contains(&ViewId::new(2))),
        "standalone registration must refresh the public reconstruction component"
    );

    let registered = &workspace.pose_store().get(2).unwrap().pose;
    let center_error = (registered.center() - gt_pose.center()).norm();
    assert!(
        center_error < 1e-5,
        "center error {center_error}, registered {:?}, gt {:?}",
        registered.center(),
        gt_pose.center()
    );
    let rotation_error = (registered.rotation().inverse() * gt_pose.rotation()).angle();
    assert!(
        rotation_error < 1e-6,
        "rotation error {rotation_error}, registered {:?}, gt {:?}",
        registered.rotation(),
        gt_pose.rotation()
    );
}

#[test]
fn registration_failures_are_recorded() {
    let gt_pose = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.08, -0.04, 0.03),
        Point3::new(0.25, -0.15, -0.8),
    )
    .unwrap();
    let mut workspace = seeded_registration_workspace(gt_pose);

    let outcome = register_view(
        &mut workspace,
        ViewId::new(2),
        &NoPoseRegistrar,
        &RegistrationConfig::default(),
    )
    .unwrap();

    assert!(matches!(
        outcome,
        RegistrationVerdict::Rejected(openmvg_sfm::unstable::incremental::RegistrationFailure {
            reason: RegistrationFailureReason::PoseEstimationFailed,
            ..
        })
    ));
    assert_eq!(workspace.registration_failures().len(), 1);
    assert!(!workspace.pose_store().has(2));
}

#[test]
fn malformed_registrar_output_is_rejected_before_commit() {
    let gt_pose = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.08, -0.04, 0.03),
        Point3::new(0.25, -0.15, -0.8),
    )
    .unwrap();
    let mut workspace = seeded_registration_workspace(gt_pose);
    let poses_before = workspace.pose_store().clone();
    let components_before = workspace.components().to_vec();

    let error = register_view(
        &mut workspace,
        ViewId::new(2),
        &ImpossibleInlierRegistrar,
        &RegistrationConfig::default(),
    )
    .unwrap_err();

    assert!(matches!(error, SfmError::InvalidInput(_)));
    assert_eq!(workspace.pose_store(), &poses_before);
    assert_eq!(workspace.components(), components_before);
    assert!(workspace.registration_failures().is_empty());
}

fn gt_pose() -> WorldToCameraPose {
    WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.08, -0.04, 0.03),
        Point3::new(0.25, -0.15, -0.8),
    )
    .unwrap()
}

#[test]
fn an_estimated_focal_replaces_the_focal_of_the_views_own_intrinsic_group() {
    let mut workspace = seeded_registration_workspace_with(gt_pose(), true, IntrinsicPrior::Guess);
    let registrar = FocalReportingRegistrar::new(Some(520.0));

    let outcome = register_view(
        &mut workspace,
        ViewId::new(2),
        &registrar,
        &RegistrationConfig::default(),
    )
    .unwrap();

    let RegistrationVerdict::Registered(report) = outcome else {
        panic!("expected image 2 to register, got {outcome:?}");
    };
    assert_eq!(*registrar.focal_estimable.lock().unwrap(), Some(true));
    assert_eq!(report.focal_px, Some(520.0));
    let focals = |intrinsic: usize| {
        workspace
            .camera_model(IntrinsicId::new(intrinsic))
            .and_then(Camera::focal_lengths)
    };
    assert_eq!(focals(2), Some((520.0, 520.0)));
    assert_eq!(focals(0), Some((500.0, 500.0)));
    assert_eq!(focals(1), Some((500.0, 500.0)));
    let principal = workspace
        .camera_model(IntrinsicId::new(2))
        .unwrap()
        .principal_point();
    assert_eq!((principal.x, principal.y), (400.0, 300.0));
}

#[test]
fn a_focal_for_a_group_shared_with_a_registered_view_is_rejected_before_commit() {
    let mut workspace = seeded_registration_workspace(gt_pose());
    let registrar = FocalReportingRegistrar::new(Some(520.0));
    let cameras_before = workspace.camera_models().to_vec();
    let poses_before = workspace.pose_store().clone();

    let error = register_view(
        &mut workspace,
        ViewId::new(2),
        &registrar,
        &RegistrationConfig::default(),
    )
    .unwrap_err();

    assert_eq!(*registrar.focal_estimable.lock().unwrap(), Some(false));
    assert!(matches!(error, SfmError::InvalidInput(_)), "{error}");
    assert_eq!(workspace.camera_models(), cameras_before);
    assert_eq!(workspace.pose_store(), &poses_before);
}

#[test]
fn the_p4pf_registrar_recovers_the_focal_of_a_guessed_group() {
    let mut workspace = seeded_registration_workspace_with(gt_pose(), true, IntrinsicPrior::Guess);
    // The observations come from a focal of 500; the group starts from a wrong guess.
    workspace
        .replace_camera_model(
            IntrinsicId::new(2),
            Camera::try_pinhole(320.0, 320.0, 400.0, 300.0).unwrap(),
        )
        .unwrap();

    let outcome = register_view(
        &mut workspace,
        ViewId::new(2),
        &GenRansacP4pfRegistrar::default(),
        &RegistrationConfig::default(),
    )
    .unwrap();

    let RegistrationVerdict::Registered(report) = outcome else {
        panic!("expected image 2 to register, got {outcome:?}");
    };
    let focal = report
        .focal_px
        .expect("a guessed group gets an estimated focal");
    assert!((focal - 500.0).abs() < 2.5, "estimated focal {focal}");
    assert_eq!(report.method, "genransac-p4pf-resection");
    assert!(report.inliers >= 8);
    let (applied, _) = workspace
        .camera_model(IntrinsicId::new(2))
        .and_then(Camera::focal_lengths)
        .unwrap();
    assert_eq!(applied, focal);
    let registered = &workspace.pose_store().get(2).unwrap().pose;
    assert!((registered.center() - gt_pose().center()).norm() < 1e-2);
}

#[test]
fn the_p4pf_registrar_leaves_a_trusted_group_to_p3p() {
    let mut workspace =
        seeded_registration_workspace_with(gt_pose(), true, IntrinsicPrior::Trusted);

    let outcome = register_view(
        &mut workspace,
        ViewId::new(2),
        &GenRansacP4pfRegistrar::default(),
        &RegistrationConfig::default(),
    )
    .unwrap();

    let RegistrationVerdict::Registered(report) = outcome else {
        panic!("expected image 2 to register, got {outcome:?}");
    };
    assert_eq!(report.focal_px, None);
    assert_eq!(report.method, "genransac-fused-p3p-resection");
    assert_eq!(
        workspace
            .camera_model(IntrinsicId::new(2))
            .and_then(Camera::focal_lengths),
        Some((500.0, 500.0))
    );
}

#[test]
fn a_trusted_focal_is_never_estimable_even_for_the_first_view_of_its_group() {
    let mut workspace =
        seeded_registration_workspace_with(gt_pose(), true, IntrinsicPrior::Trusted);
    let registrar = FocalReportingRegistrar::new(Some(520.0));
    let cameras_before = workspace.camera_models().to_vec();

    let error = register_view(
        &mut workspace,
        ViewId::new(2),
        &registrar,
        &RegistrationConfig::default(),
    )
    .unwrap_err();

    assert_eq!(*registrar.focal_estimable.lock().unwrap(), Some(false));
    assert!(matches!(error, SfmError::InvalidInput(_)), "{error}");
    assert_eq!(workspace.camera_models(), cameras_before);
}

#[test]
fn a_registered_view_fixes_a_guessed_focal_for_the_views_that_share_its_group() {
    let mut workspace = seeded_registration_workspace_with(gt_pose(), false, IntrinsicPrior::Guess);
    let registrar = FocalReportingRegistrar::new(None);

    register_view(
        &mut workspace,
        ViewId::new(2),
        &registrar,
        &RegistrationConfig::default(),
    )
    .unwrap();

    // The seed views 0 and 1 are registered and share the group, so the guess is already used.
    assert_eq!(*registrar.focal_estimable.lock().unwrap(), Some(false));
}

#[test]
fn a_non_positive_or_non_finite_focal_is_rejected() {
    for focal in [0.0, -300.0, f64::NAN, f64::INFINITY] {
        let mut workspace =
            seeded_registration_workspace_with(gt_pose(), true, IntrinsicPrior::Guess);
        let cameras_before = workspace.camera_models().to_vec();

        let error = register_view(
            &mut workspace,
            ViewId::new(2),
            &FocalReportingRegistrar::new(Some(focal)),
            &RegistrationConfig::default(),
        )
        .unwrap_err();

        assert!(
            matches!(error, SfmError::InvalidInput(_)),
            "{focal}: {error}"
        );
        assert_eq!(workspace.camera_models(), cameras_before);
        assert!(!workspace.pose_store().has(2));
    }
}

#[test]
fn registration_receives_the_workspace_derived_seed() {
    let gt_pose = WorldToCameraPose::from_rotation_center(
        Rotation3::from_euler_angles(0.08, -0.04, 0.03),
        Point3::new(0.25, -0.15, -0.8),
    )
    .unwrap();
    let mut workspace = seeded_registration_workspace(gt_pose);
    let registrar = SeedCapturingRegistrar::default();

    register_view(
        &mut workspace,
        ViewId::new(2),
        &registrar,
        &RegistrationConfig::default(),
    )
    .unwrap();

    let captured = registrar.seed.lock().unwrap().unwrap();
    let image = ViewId::new(2);
    assert_eq!(
        captured,
        workspace
            .root_seed()
            .derive("pipeline/pose-registration/v1", &[image.value()])
    );
    assert_ne!(
        captured,
        RunSeed::from_u64(1).derive("pipeline/pose-registration/v1", &[image.value()])
    );
}

fn resection_workspace(gt_pose: WorldToCameraPose) -> SfmWorkspace {
    resection_workspace_with(gt_pose, false, IntrinsicPrior::Trusted)
}

/// The three-view resection scene; with `independent_intrinsics` every view owns an intrinsic
/// group instead of sharing one. Every group has the focal-length prior `prior`.
fn resection_workspace_with(
    gt_pose: WorldToCameraPose,
    independent_intrinsics: bool,
    prior: IntrinsicPrior,
) -> SfmWorkspace {
    struct ResectionExtractor {
        pose: WorldToCameraPose,
    }
    struct ResectionMatcher;
    struct ResectionVerifier;

    impl FeatureExtractor for ResectionExtractor {
        fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
            let pose = match ctx.view.index() {
                0 => WorldToCameraPose::identity(),
                1 => WorldToCameraPose::from_rotation_center(
                    Rotation3::identity(),
                    Point3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
                2 => self.pose.clone(),
                _ => unreachable!(),
            };
            let keypoints = world_points()
                .into_iter()
                .map(|point| project(&pose, point))
                .map(|xy| Keypoint::try_new(xy.x as f32, xy.y as f32, 1.0, 0.0))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(ExtractedFeatures::try_new(
                keypoints,
                DescriptorSet::try_f32(vec![0.0; 8], 1)?,
            )?)
        }
    }

    impl PairMatcher for ResectionMatcher {
        fn match_features(
            &self,
            ctx: &PairContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
        ) -> Result<PairMatches> {
            PairMatches::try_new(
                ctx.pair,
                (0..8)
                    .map(|index| {
                        FeatureMatch::try_new(
                            FeatureId::new(index),
                            FeatureId::new(index),
                            Some(0.0),
                        )
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )
        }
    }

    impl GeometricVerifier for ResectionVerifier {
        fn verify_pair(
            &self,
            ctx: &VerificationContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
            matches: &PairMatches,
            _ki: &openmvg_sfm::Camera,
            _kj: &openmvg_sfm::Camera,
        ) -> Result<VerificationVerdict> {
            let evidence = ModelEvidence::try_new(
                InlierMask::all_inliers(matches.len()),
                openmvg_sfm::unstable::reports::ErrorStats::from_values(
                    &vec![0.0; matches.len()],
                    &[1.0],
                ),
                openmvg_core::PixelThreshold::ONE_PIXEL.into(),
                "registration-test",
                1,
                TerminationReason::Converged,
                Vec::new(),
            )?;
            Ok(VerificationVerdict::Accepted {
                model: GeometricModel::Essential {
                    matrix: nalgebra::Matrix3::identity(),
                    relative_pose: RelativePose::from_baseline_direction(
                        Rotation3::identity(),
                        Vector3::x_axis(),
                    )?,
                    evidence,
                },
                report: PairVerificationReport {
                    pair: ctx.pair,
                    input_match_count: matches.len(),
                    inlier_count: matches.len(),
                    message: None,
                },
            })
        }
    }

    let camera = Camera::try_pinhole(500.0, 500.0, 400.0, 300.0).unwrap();
    let builder = SfmWorkspace::builder().image_meta(vec![
        ImageMeta::external_id("seed-a", 800, 600),
        ImageMeta::external_id("seed-b", 800, 600),
        ImageMeta::external_id("register-c", 800, 600),
    ]);
    let builder = if independent_intrinsics {
        builder
            .camera_models(vec![camera; 3])
            .intrinsic_priors(vec![prior; 3])
            .view_intrinsics((0..3).map(IntrinsicId::new))
    } else {
        builder
            .camera_models(vec![camera])
            .intrinsic_priors(vec![prior])
            .shared_intrinsic(IntrinsicId::new(0))
    };
    let mut workspace = builder
        .imported_pairs(vec![
            (ViewId::new(0), ViewId::new(1)),
            (ViewId::new(0), ViewId::new(2)),
            (ViewId::new(1), ViewId::new(2)),
        ])
        .build()
        .unwrap();
    workspace
        .extract_features(&ResectionExtractor { pose: gt_pose }, ViewSelector::All)
        .unwrap()
        .match_pairs(&ResectionMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&ResectionVerifier, PairSelector::Active)
        .unwrap();
    workspace
}

fn seeded_registration_workspace(gt_pose: WorldToCameraPose) -> SfmWorkspace {
    seeded_registration_workspace_with(gt_pose, false, IntrinsicPrior::Trusted)
}

fn seeded_registration_workspace_with(
    gt_pose: WorldToCameraPose,
    independent_intrinsics: bool,
    prior: IntrinsicPrior,
) -> SfmWorkspace {
    let mut workspace = resection_workspace_with(gt_pose.clone(), independent_intrinsics, prior);
    workspace.build_tracks().unwrap();
    initialize_incremental_seed_from_oracle(
        &mut workspace,
        &openmvg_sfm::testing::minimal_incremental_config().with_initial_pair(
            InitialPairPolicy::BestPair(InitialPairConfig::default().with_min_inliers(8)),
        ),
        &openmvg_sfm::unstable::evaluation::GroundTruthReconstruction {
            poses: vec![
                (ViewId::new(0), WorldToCameraPose::identity()),
                (
                    ViewId::new(1),
                    WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::new(1.0, 0.0, 0.0),
                    )
                    .unwrap(),
                ),
                (ViewId::new(2), gt_pose),
            ],
            points: world_points()
                .into_iter()
                .enumerate()
                .map(|(index, point)| (openmvg_sfm::LandmarkId::new(index), point))
                .collect(),
        },
    )
    .unwrap();
    workspace
}

struct NoPoseRegistrar;

impl PoseRegistrar for NoPoseRegistrar {
    fn estimate_pose(
        &self,
        _ctx: &PoseRegistrationContext,
        _observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        Ok(None)
    }
}

struct ImpossibleInlierRegistrar;

impl PoseRegistrar for ImpossibleInlierRegistrar {
    fn estimate_pose(
        &self,
        _ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        Ok(Some(PoseRegistrationEstimate {
            pose: WorldToCameraPose::identity(),
            inliers: observations.len() + 1,
            residual_median: openmvg_core::AngularResidual::ZERO,
            method: "malformed-test".to_owned(),
            focal_px: None,
        }))
    }
}

/// Registers with the P3P registrar and reports `focal_px` as if it had estimated the focal,
/// recording whether the pipeline allowed it.
struct FocalReportingRegistrar {
    focal_px: Option<f64>,
    focal_estimable: Mutex<Option<bool>>,
}

impl FocalReportingRegistrar {
    fn new(focal_px: Option<f64>) -> Self {
        Self {
            focal_px,
            focal_estimable: Mutex::new(None),
        }
    }
}

impl PoseRegistrar for FocalReportingRegistrar {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        *self.focal_estimable.lock().unwrap() = Some(ctx.focal_estimable);
        Ok(GenRansacP3pRegistrar::default()
            .estimate_pose(ctx, observations)?
            .map(|estimate| PoseRegistrationEstimate {
                focal_px: self.focal_px,
                ..estimate
            }))
    }
}

#[derive(Default)]
struct SeedCapturingRegistrar {
    seed: Mutex<Option<RunSeed>>,
}

impl PoseRegistrar for SeedCapturingRegistrar {
    fn estimate_pose(
        &self,
        ctx: &PoseRegistrationContext,
        _observations: &[PoseRegistrationObservation],
    ) -> Result<Option<PoseRegistrationEstimate>> {
        *self.seed.lock().unwrap() = Some(ctx.seed);
        Ok(None)
    }
}

fn world_points() -> Vec<Point3<f64>> {
    vec![
        Point3::new(-0.5, -0.3, 4.0),
        Point3::new(0.4, -0.2, 4.4),
        Point3::new(-0.2, 0.5, 4.8),
        Point3::new(0.6, 0.4, 5.2),
        Point3::new(-0.7, 0.2, 5.5),
        Point3::new(0.1, -0.6, 5.8),
        Point3::new(0.8, -0.1, 6.1),
        Point3::new(-0.4, 0.7, 6.4),
    ]
}

fn project(pose: &WorldToCameraPose, point: Point3<f64>) -> Point2<f64> {
    let camera_point = pose.project_to_camera(&point);
    Point2::new(
        500.0 * camera_point.x / camera_point.z + 400.0,
        500.0 * camera_point.y / camera_point.z + 300.0,
    )
}
