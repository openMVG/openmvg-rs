// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use openmvg_sfm::testing::determinism::stable_json_hash;
use openmvg_sfm::testing::oracle::{
    OracleSfmFixture, incremental_sfm_oracle, incremental_sfm_oracle_with,
    initialize_incremental_seed_from_oracle,
};
use openmvg_sfm::{config::*, unstable::incremental::*, unstable::*, *};

#[test]
fn incremental_oracle_fixture_can_drive_seed_selection_without_real_registration() {
    let fixture = common::oracle_fixture::fixture();
    let workspace = oracle_workspace(&fixture);

    let seed = select_initial_pair(
        &workspace,
        &InitialPairPolicy::BestPair(InitialPairConfig::default().with_min_inliers(8)),
    )
    .unwrap();

    assert_eq!(seed.views.len(), 2);
    assert!(seed.breakdown.inliers >= 8);
}

#[test]
fn oracle_seed_ba_runs_and_preserves_the_ground_truth_baseline_gauge() {
    let fixture = common::oracle_fixture::fixture();
    let mut workspace = oracle_workspace(&fixture);
    workspace.build_tracks().unwrap();
    let mut adjuster = OracleIdentityBa;
    let report = incremental_sfm_oracle_with(
        &mut workspace,
        &openmvg_sfm::testing::minimal_incremental_config()
            .with_initial_pair(InitialPairPolicy::ExplicitPair(ViewPair::new(
                ViewId::new(0),
                ViewId::new(1),
            )))
            .with_bundle_adjust_initial_pair(true)
            .with_registration_max_rotation_edge_error(None),
        &fixture.scene,
        &openmvg_sfm::unstable::incremental::GenRansacP3pRegistrar::default(),
        &openmvg_sfm::unstable::adapters::triangulation::TrackTriangulatorAdapter::default(),
        Some(&mut adjuster),
    )
    .unwrap();

    assert!(
        report
            .reconstruction
            .initialization
            .seed_bundle_adjustment
            .is_some()
    );
    let expected = (fixture.scene.poses[1].1.center() - fixture.scene.poses[0].1.center()).norm();
    assert!(matches!(
        workspace.components()[0].gauge_scale_policy,
        GaugeScalePolicy::FixedBaseline { distance, .. }
            if (distance - expected).abs() <= 256.0 * f64::EPSILON * expected.max(1.0)
    ));
}

#[test]
fn invalid_oracle_ba_policy_is_rejected_before_seed_mutation() {
    let fixture = common::oracle_fixture::fixture();
    let mut workspace = oracle_workspace(&fixture);
    workspace.build_tracks().unwrap();
    let error = incremental_sfm_oracle_with(
        &mut workspace,
        &openmvg_sfm::testing::minimal_incremental_config().with_bundle_adjust_growth_ratio(0.1),
        &fixture.scene,
        &openmvg_sfm::unstable::incremental::GenRansacP3pRegistrar::default(),
        &openmvg_sfm::unstable::adapters::triangulation::TrackTriangulatorAdapter::default(),
        None,
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("requires a bundle-adjustment backend")
    );
    assert_eq!(workspace.pose_store().populated_count(), 0);
    assert!(workspace.landmarks().is_empty());
    assert!(workspace.components().is_empty());
}

struct OracleIdentityBa;

impl openmvg_sfm::ba::BundleAdjuster for OracleIdentityBa {
    fn name(&self) -> &'static str {
        "oracle-identity"
    }

    fn solve(
        &mut self,
        problem: &openmvg_sfm::ba::BundleProblem,
        _config: openmvg_sfm::ba::SolveConfig,
    ) -> std::result::Result<openmvg_sfm::ba::Solution, openmvg_sfm::ba::SolveError> {
        Ok(openmvg_sfm::testing::identity_bundle_solution(problem))
    }
}

#[test]
fn incremental_oracle_grows_every_camera_from_a_fixed_gt_seed() {
    let fixture = common::oracle_fixture::fixture();
    let mut workspace = oracle_workspace(&fixture);
    workspace.build_tracks().unwrap();

    let report = incremental_sfm_oracle(
        &mut workspace,
        &openmvg_sfm::testing::minimal_incremental_config()
            .with_initial_pair(InitialPairPolicy::ExplicitPair(ViewPair::new(
                ViewId::new(0),
                ViewId::new(1),
            )))
            // The injected oracle verifier intentionally stores identity relative
            // rotations. They are sufficient to build the graph, but must not be
            // treated as real pose evidence by this P3P isolation test.
            .with_registration_max_rotation_edge_error(None),
        &fixture.scene,
    )
    .unwrap();

    assert_eq!(report.reconstruction.initialization.seed.views.len(), 2);
    assert_eq!(report.reconstruction.registered_view_count, 5);
    assert_eq!(report.reconstruction.growth.grown_view_count, 3);
    assert_eq!(report.reconstruction.growth.growth_registrations.len(), 3);
    assert!(
        report
            .reconstruction
            .growth
            .triangulation_rejections
            .is_empty()
    );
    assert!(report.reconstruction.landmark_count > 0);
    assert_eq!(report.evaluation.pose_count, 5);
    assert!(report.evaluation.ate.max < 1e-5);
    assert!(report.evaluation.absolute_rotation_are.max < 1e-5);
    assert!(report.growth_candidates.is_empty());
}

#[test]
fn growth_neighbor_gate_rejects_track_only_unanchored_candidate() {
    let fixture = common::oracle_fixture::fixture();
    let mut workspace = oracle_workspace(&fixture);
    workspace.build_tracks().unwrap();
    initialize_incremental_seed_from_oracle(
        &mut workspace,
        &openmvg_sfm::testing::minimal_incremental_config(),
        &fixture.reconstruction_gt,
    )
    .unwrap();

    let loose = GrowthConfig::default()
        .with_min_visible_landmarks(1)
        .with_min_verified_registered_neighbors(0);
    let target = registration_candidates(&workspace, &loose)[0].view;
    let incident_to_registered: Vec<_> = workspace
        .pair_graph()
        .neighbors(target)
        .unwrap()
        .iter()
        .filter(|neighbor| workspace.has_pose(neighbor.view))
        .map(|neighbor| ViewPair::new(target, neighbor.view))
        .collect();
    assert!(!incident_to_registered.is_empty());
    for pair in incident_to_registered {
        workspace.pair_graph_mut().deactivate_pair(pair).unwrap();
    }

    let track_only = registration_candidates(&workspace, &loose)
        .into_iter()
        .find(|candidate| candidate.view == target)
        .expect("track intersections remain after graph-edge deactivation");
    assert_eq!(track_only.verified_registered_neighbors, 0);
    let anchored = loose.with_min_verified_registered_neighbors(1);
    assert!(
        registration_candidates(&workspace, &anchored)
            .into_iter()
            .all(|candidate| candidate.view != target)
    );
}

#[test]
fn synthetic_oracle_fixture_preserves_point_ids() {
    let fixture = common::oracle_fixture::fixture();

    assert_eq!(fixture.points_gt.len(), fixture.tracks.len());
    assert!(fixture.tracks.iter().all(|track| {
        fixture
            .scene
            .points
            .iter()
            .any(|(landmark_id, _)| *landmark_id == track.landmark_id)
    }));

    let mut original_point_ids: Vec<_> = fixture
        .tracks
        .iter()
        .map(|track| track.original_point_id)
        .collect();
    original_point_ids.sort_unstable();
    original_point_ids.dedup();
    assert_eq!(original_point_ids.len(), fixture.tracks.len());
}

#[test]
fn oracle_fixture_hash_is_stable_across_two_runs() {
    let fixture_a = common::oracle_fixture::fixture();
    let fixture_b = common::oracle_fixture::fixture();

    assert_eq!(
        stable_json_hash(&fixture_a).unwrap(),
        stable_json_hash(&fixture_b).unwrap()
    );
}

#[test]
fn oracle_match_corruption_is_deterministic_and_exactly_labeled() {
    use openmvg_sfm::testing::oracle::OracleMatchCorruptionConfig;

    let config =
        OracleMatchCorruptionConfig::with_poisoned_pair(0.1, ViewId::new(0), ViewId::new(1));
    let mut fixture_a = common::oracle_fixture::fixture();
    let mut fixture_b = common::oracle_fixture::fixture();
    let report_a = fixture_a.inject_match_outliers(&config).unwrap();
    let report_b = fixture_b.inject_match_outliers(&config).unwrap();

    assert_eq!(fixture_a.matches, fixture_b.matches);
    assert_eq!(report_a, report_b);
    assert_eq!(
        report_a.total_outliers,
        fixture_a
            .matches
            .iter()
            .flat_map(|pair| &pair.inlier_mask)
            .filter(|is_inlier| !**is_inlier)
            .count()
    );
    let poisoned = report_a
        .pairs
        .iter()
        .find(|pair| pair.pair == ViewPair::new(ViewId::new(0), ViewId::new(1)))
        .unwrap();
    assert!(poisoned.fully_poisoned);
    assert_eq!(poisoned.outlier_count, poisoned.match_count);
    for pair in &fixture_a.matches {
        let mut unique = pair.matches.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), pair.matches.len());
    }

    let once_corrupted = fixture_a.clone();
    assert!(fixture_a.inject_match_outliers(&config).is_err());
    assert_eq!(
        fixture_a, once_corrupted,
        "failed injection is transactional"
    );
}

#[test]
fn oracle_match_corruption_rejects_invalid_fraction() {
    use openmvg_sfm::testing::oracle::OracleMatchCorruptionConfig;

    let mut fixture = common::oracle_fixture::fixture();
    assert!(
        fixture
            .inject_match_outliers(&OracleMatchCorruptionConfig::diffuse(1.01))
            .is_err()
    );
}

fn oracle_workspace(fixture: &OracleSfmFixture) -> SfmWorkspace {
    struct OracleExtractor<'a>(&'a OracleSfmFixture);
    struct OracleMatcher<'a>(&'a OracleSfmFixture);
    struct OracleVerifier;

    impl openmvg_sfm::unstable::backends::FeatureExtractor for OracleExtractor<'_> {
        fn extract(
            &self,
            ctx: &openmvg_sfm::unstable::backends::FeatureContext,
            _meta: &openmvg_sfm::model::ImageMeta,
        ) -> openmvg_sfm::Result<openmvg_sfm::unstable::ExtractedFeatures> {
            let mut keypoints = self
                .0
                .observations
                .iter()
                .filter(|observation| observation.view == ctx.view)
                .map(|observation| {
                    openmvg_sfm::Keypoint::try_new(
                        observation.xy.x as f32,
                        observation.xy.y as f32,
                        1.0,
                        0.0,
                    )
                    .map(|keypoint| (observation.feature_id, keypoint))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            keypoints.sort_by_key(|(feature_id, _)| *feature_id);
            let keypoints = keypoints
                .into_iter()
                .map(|(_, keypoint)| keypoint)
                .collect::<Vec<_>>();
            let descriptors =
                openmvg_sfm::unstable::DescriptorSet::try_f32(vec![0.0; keypoints.len()], 1)?;
            Ok(openmvg_sfm::unstable::ExtractedFeatures::try_new(
                keypoints,
                descriptors,
            )?)
        }
    }

    impl openmvg_sfm::unstable::backends::PairMatcher for OracleMatcher<'_> {
        fn match_features(
            &self,
            ctx: &openmvg_sfm::unstable::backends::PairContext,
            _fi: &openmvg_sfm::unstable::ExtractedFeatures,
            _fj: &openmvg_sfm::unstable::ExtractedFeatures,
        ) -> openmvg_sfm::Result<openmvg_sfm::unstable::types::PairMatches> {
            let pair = self
                .0
                .matches
                .iter()
                .find(|pair| pair.pair == ViewPair::new(ctx.view_i, ctx.view_j))
                .ok_or_else(|| {
                    openmvg_sfm::SfmError::InvalidInput("missing oracle pair".to_owned())
                })?;
            openmvg_sfm::unstable::types::PairMatches::try_new(
                ctx.pair,
                pair.matches
                    .iter()
                    .map(|(a, b)| openmvg_sfm::FeatureMatch::try_new(*a, *b, Some(0.0)))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )
        }
    }

    impl openmvg_sfm::unstable::backends::GeometricVerifier for OracleVerifier {
        fn verify_pair(
            &self,
            ctx: &openmvg_sfm::unstable::backends::VerificationContext,
            _fi: &openmvg_sfm::unstable::ExtractedFeatures,
            _fj: &openmvg_sfm::unstable::ExtractedFeatures,
            matches: &openmvg_sfm::unstable::types::PairMatches,
            _ki: &openmvg_sfm::Camera,
            _kj: &openmvg_sfm::Camera,
        ) -> openmvg_sfm::Result<openmvg_sfm::unstable::types::VerificationVerdict> {
            let evidence = openmvg_sfm::unstable::types::ModelEvidence::try_new(
                openmvg_sfm::unstable::types::InlierMask::all_inliers(matches.len()),
                openmvg_sfm::unstable::reports::ErrorStats::from_values(
                    &vec![0.0; matches.len()],
                    &[1.0],
                ),
                openmvg_core::PixelThreshold::ONE_PIXEL.into(),
                "oracle-incremental",
                1,
                openmvg_sfm::unstable::types::TerminationReason::Converged,
                Vec::new(),
            )?;
            Ok(
                openmvg_sfm::unstable::types::VerificationVerdict::Accepted {
                    model: openmvg_sfm::unstable::types::GeometricModel::Essential {
                        matrix: nalgebra::Matrix3::identity(),
                        relative_pose: openmvg_sfm::RelativePose::from_baseline_direction(
                            nalgebra::Rotation3::identity(),
                            nalgebra::Vector3::x_axis(),
                        )?,
                        evidence,
                    },
                    report: openmvg_sfm::unstable::types::PairVerificationReport {
                        pair: ctx.pair,
                        input_match_count: matches.len(),
                        inlier_count: matches.len(),
                        message: None,
                    },
                },
            )
        }
    }

    let mut workspace = SfmWorkspace::builder()
        .image_meta(fixture.image_meta.clone())
        .camera_models(fixture.camera_models.clone())
        .shared_intrinsic(openmvg_sfm::IntrinsicId::new(0))
        .imported_pairs(fixture.pairs.clone())
        .build()
        .unwrap();
    workspace
        .extract_features(
            &OracleExtractor(fixture),
            openmvg_sfm::unstable::ViewSelector::All,
        )
        .unwrap()
        .match_pairs(
            &OracleMatcher(fixture),
            openmvg_sfm::unstable::PairSelector::Active,
        )
        .unwrap()
        .verify_pairs(&OracleVerifier, openmvg_sfm::unstable::PairSelector::Active)
        .unwrap();
    workspace
}
