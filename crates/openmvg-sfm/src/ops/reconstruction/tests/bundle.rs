// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bundle adjustment problems, commits, adjust-and-clean rounds, and loss policy.

use super::*;

struct IdentityBackend;

#[test]
fn fixed_view_expands_to_the_complete_shared_pose_group() {
    let workspace = SfmWorkspace::builder()
        .image_meta([
            ImageMeta::external_id("a", 64, 48),
            ImageMeta::external_id("b", 64, 48),
        ])
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .shared_pose()
        .build()
        .unwrap();
    let config =
        BundleAdjustConfig::default().with_fixed_views([ViewId::new(1)].into_iter().collect());

    let expanded = crate::ops::reconstruction::bundle::config_with_complete_shared_pose_fixes(
        &workspace, &config,
    )
    .unwrap();

    assert_eq!(
        expanded.fixed_views(),
        &[ViewId::new(0), ViewId::new(1)].into_iter().collect()
    );
}

impl BundleAdjustmentBackend for IdentityBackend {
    fn adjust(
        &self,
        problem: &openmvg_ba::BundleProblem,
        _config: &BundleAdjustConfig,
    ) -> BaBackendResult {
        Ok(BaBackendStatus::Proposed(Box::new(
            crate::testing::identity_bundle_solution(problem),
        )))
    }
}

#[test]
fn angular_regression_policy_has_absolute_slack() {
    assert_eq!(angular_regression_limit(0.0), 0.005);
}

fn mixed_camera_workspace() -> SfmWorkspace {
    SfmWorkspace::builder()
        .image_meta([
            ImageMeta::external_id("perspective", 64, 48),
            ImageMeta::external_id("spherical", 1024, 512),
        ])
        .camera_models([
            Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
            Camera::Spherical(SphericalCamera::try_new(1024).unwrap()),
        ])
        .view_intrinsics([IntrinsicId::new(0), IntrinsicId::new(1)])
        .build()
        .unwrap()
}

#[test]
fn workspace_loss_policy_is_conservative_only_for_posed_mixed_camera_kinds() {
    let pixel = BundleAdjustConfig::default().with_robust_loss(openmvg_ba::RobustLoss::Cauchy(
        openmvg_ba::ResidualScale::Pixels(openmvg_core::PixelThreshold::FOUR_PIXELS),
    ));
    let mut workspace = mixed_camera_workspace();
    workspace
        .set_pose(PoseRecord {
            id: crate::PoseId::new(0),
            pose: openmvg_core::model::WorldToCameraPose::default(),
            component: None,
        })
        .unwrap();
    assert_eq!(
        pipeline_ba_config_for_workspace(&workspace, pixel.clone()).robust_loss(),
        pixel.robust_loss()
    );

    workspace
        .set_pose(PoseRecord {
            id: crate::PoseId::new(1),
            pose: openmvg_core::model::WorldToCameraPose::default(),
            component: None,
        })
        .unwrap();
    assert_eq!(
        pipeline_ba_config_for_workspace(&workspace, pixel.clone()).robust_loss(),
        openmvg_ba::RobustLoss::Squared
    );

    let angular = BundleAdjustConfig::default().with_robust_loss(openmvg_ba::RobustLoss::Cauchy(
        openmvg_ba::ResidualScale::Angular(openmvg_core::AngularThreshold::try_new(0.01).unwrap()),
    ));
    assert_eq!(
        pipeline_ba_config_for_workspace(&workspace, angular.clone()).robust_loss(),
        angular.robust_loss()
    );

    let self_calibration = pixel.with_scope(OptimizationScope::SelfCalibration);
    assert_eq!(
        pipeline_ba_config_for_workspace(&workspace, self_calibration.clone()).robust_loss(),
        self_calibration.robust_loss()
    );
}

#[test]
fn a_spherical_only_workspace_keeps_its_pixel_loss_for_conversion_to_radians() {
    let pixel = BundleAdjustConfig::default().with_robust_loss(openmvg_ba::RobustLoss::Huber(
        openmvg_ba::ResidualScale::Pixels(openmvg_core::PixelThreshold::ONE_PIXEL),
    ));
    let mut workspace = supported_workspace_for(
        Camera::Spherical(SphericalCamera::try_new(64).unwrap()),
        64,
        32,
    );
    for index in 0..3 {
        workspace
            .set_pose(PoseRecord {
                id: crate::PoseId::new(index),
                pose: openmvg_core::model::WorldToCameraPose::from_rotation_center(
                    nalgebra::Rotation3::identity(),
                    nalgebra::Point3::new(0.5 * index as f64, 0.0, 0.0),
                )
                .unwrap(),
                component: Some(crate::ComponentId::new(0)),
            })
            .unwrap();
    }

    let config = pipeline_ba_config_for_workspace(&workspace, pixel.clone());
    let problem = bundle_adjustment_problem(&workspace, &config).unwrap();

    assert_eq!(config.robust_loss(), pixel.robust_loss());
    let Some(openmvg_ba::ResidualScale::Angular(scale)) = problem.robust_loss().scale() else {
        panic!("expected an angular scale, got {:?}", problem.robust_loss());
    };
    assert!((scale.value() - std::f64::consts::TAU / 64.0).abs() < 1e-12);
}

#[test]
fn workspace_rejects_mixed_dimensions_within_one_intrinsic_group() {
    let error = SfmWorkspace::builder()
        .image_meta([
            ImageMeta::external_id("a", 64, 48),
            ImageMeta::external_id("b", 80, 60),
        ])
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .build()
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("shared by images with dimensions")
    );
}

#[test]
fn bundle_problem_uses_only_triangulation_support() {
    let workspace = supported_workspace();
    let problem = bundle_adjustment_problem(&workspace, &BundleAdjustConfig::default())
        .expect("BA problem should build");
    let pose_indices: Vec<_> = problem
        .observations()
        .iter()
        .map(|observation| observation.pose())
        .collect();

    assert_eq!(pose_indices, vec![0, 2]);
}

#[test]
fn self_calibration_never_frees_the_intrinsics_of_a_spherical_camera() {
    let config = BundleAdjustConfig::default().with_scope(OptimizationScope::SelfCalibration);
    let spherical = supported_workspace_for(
        Camera::Spherical(SphericalCamera::try_new(64).unwrap()),
        64,
        32,
    );
    let problem = bundle_adjustment_problem(&spherical, &config).unwrap();

    assert!(problem.intrinsics()[0].variables().is_fixed());
    assert_eq!(problem.residual_space(), openmvg_ba::ResidualSpace::Angular);
    let pinhole = bundle_adjustment_problem(&supported_workspace(), &config).unwrap();
    assert!(!pinhole.intrinsics()[0].variables().is_fixed());
}

#[test]
fn only_a_guessed_intrinsic_group_is_waived_from_certification() {
    let config = BundleAdjustConfig::default().with_scope(OptimizationScope::SelfCalibration);
    let mut workspace = supported_workspace();

    let trusted = bundle_adjustment_problem(&workspace, &config).unwrap();
    assert_eq!(
        trusted.intrinsics()[0].certification(),
        openmvg_ba::IntrinsicCertification::Required
    );

    workspace.set_intrinsic_priors(vec![crate::model::IntrinsicPrior::Guess]);
    let guessed = bundle_adjustment_problem(&workspace, &config).unwrap();
    assert_eq!(
        guessed.intrinsics()[0].certification(),
        openmvg_ba::IntrinsicCertification::Waived
    );
}

#[test]
fn bundle_commit_preserves_nonzero_component_identity() {
    let mut workspace = supported_workspace();
    let component = ComponentId::new(7);
    for image in 0..3 {
        workspace
            .poses
            .get_mut(image)
            .expect("supported workspace has every pose")
            .component = Some(component);
    }
    workspace.components = vec![ReconstructionComponent::new(
        component,
        (0..3).map(ViewId::new).collect(),
        vec![LandmarkId::new(0)],
        GaugeScalePolicy::Free,
    )];

    bundle_adjust(
        &mut workspace,
        &IdentityBackend,
        &BundleAdjustConfig::default(),
    )
    .expect("identity proposal should commit");

    assert_eq!(workspace.components()[0].id, component);
    assert!(
        workspace
            .pose_store()
            .iter_populated()
            .all(|(_, record)| record.component == Some(component))
    );
}

#[test]
fn observation_only_cleanup_runs_one_verification_ba_round() {
    let mut workspace = supported_workspace();
    workspace.landmarks.as_mut_slice()[0].observation_support = ObservationSupport::default();

    let report = bundle_adjust_and_clean(
        &mut workspace,
        &IdentityBackend,
        &BundleAdjustConfig::default(),
        4.0,
        0.0,
        2,
    )
    .expect("observation pruning should be verified by a second BA");

    assert_eq!(report.rounds.len(), 2);
    assert_eq!(report.rounds[0].removed_after_adjustment, 0);
    assert_eq!(report.rounds[0].removed_observations_after_adjustment, 1);
    assert_eq!(report.rounds[1].removed_observations_after_adjustment, 0);
    assert_eq!(report.total_removed_landmarks, 0);
    assert_eq!(report.total_removed_observations, 1);
    assert_eq!(report.termination, BundleAdjustAndCleanTermination::Stable);
}

#[test]
fn adjust_and_clean_validates_cleanup_policy_before_calling_backend() {
    struct MustNotRun;

    impl BundleAdjustmentBackend for MustNotRun {
        fn adjust(
            &self,
            _problem: &openmvg_ba::BundleProblem,
            _config: &BundleAdjustConfig,
        ) -> BaBackendResult {
            panic!("invalid cleanup configuration must be rejected before BA")
        }
    }

    let mut workspace = supported_workspace();
    let landmarks_before = workspace.landmarks().clone();
    let result = bundle_adjust_and_clean(
        &mut workspace,
        &MustNotRun,
        &BundleAdjustConfig::default(),
        f64::NAN,
        2.0,
        1,
    );

    assert!(matches!(result, Err(SfmError::InvalidConfig(_))));
    assert_eq!(workspace.landmarks(), &landmarks_before);
}

#[test]
fn bundle_problem_lists_only_intrinsic_groups_of_posed_views() {
    // The unposed view's group has no residual: as a parameter block it would be a free
    // self-calibration direction, and the commit would write it back unconstrained.
    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..4)
                .map(|image| ImageMeta::external_id(format!("image-{image}"), 64, 48))
                .collect::<Vec<_>>(),
        )
        .camera_models([
            Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap(),
            Camera::try_pinhole(80.0, 80.0, 32.0, 24.0).unwrap(),
        ])
        .view_intrinsics([0, 0, 0, 1].map(IntrinsicId::new))
        .build()
        .unwrap();
    let centers = [0.0, 0.5, 1.0].map(|x| Point3::new(x, 0.0, 0.0));
    let point = Point3::new(0.1, 0.1, 5.0);
    for (image, center) in centers.iter().enumerate() {
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
    workspace.tracks = TrackSet::try_new(vec![
        Track::try_new(
            TrackId::new(0),
            (0..3)
                .map(|view| TrackObservation {
                    view: ViewId::new(view),
                    feature: FeatureId::new(0),
                })
                .collect(),
        )
        .unwrap(),
    ])
    .unwrap();
    set_feature_pixels(
        &mut workspace,
        centers
            .iter()
            .map(|center| {
                let camera_point = point - center;
                vec![Point2::new(
                    100.0 * camera_point.x / camera_point.z + 32.0,
                    100.0 * camera_point.y / camera_point.z + 24.0,
                )]
            })
            .collect(),
    );
    workspace.landmarks = Landmarks::try_new(vec![Landmark::active(
        LandmarkId::new(0),
        TrackId::new(0),
        point,
        openmvg_core::PixelResidual::ZERO,
        ObservationSupport::default(),
    )])
    .unwrap();

    let problem = bundle_adjustment_problem(
        &workspace,
        &BundleAdjustConfig::default().with_scope(OptimizationScope::SelfCalibration),
    )
    .unwrap();

    assert_eq!(problem.intrinsics().len(), 1);
    assert_eq!(problem.poses().len(), 3);
}

#[test]
fn negligible_cleanup_scales_with_the_active_problem() {
    assert!(cleanup_removal_is_negligible(21, 1_280_000, 1e-4));
    assert!(cleanup_removal_is_negligible(124, 1_280_000, 1e-4));
    assert!(!cleanup_removal_is_negligible(129, 1_280_000, 1e-4));
    assert!(!cleanup_removal_is_negligible(1, 100, 1e-4));
}
