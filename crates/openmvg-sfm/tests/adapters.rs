// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::collections::HashMap;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, Vector3};
use openmvg_core::{
    camera::{CameraModel, RadialDistortion, TangentialDistortion},
    model::WorldToCameraPose,
};
use openmvg_sfm::{
    model::*,
    unstable::*,
    unstable::{
        adapters::{matching::*, triangulation::*, verification::*},
        backends::*,
        reports::ErrorStats,
        types::*,
    },
    *,
};

fn two_features() -> ExtractedFeatures {
    let keypoints = vec![
        Keypoint::try_new(10.0, 12.0, 1.0, 0.0).unwrap(),
        Keypoint::try_new(20.0, 24.0, 1.5, 0.2).unwrap(),
    ];
    let descriptors = DescriptorSet::try_f32(vec![0.0, 1.0, 1.0, 0.0], 2).unwrap();
    ExtractedFeatures::try_new(keypoints, descriptors).unwrap()
}

fn oracle_pipeline_features(
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
) -> Vec<ExtractedFeatures> {
    let mut per_image_keypoints: Vec<Vec<Option<Keypoint>>> =
        vec![Vec::new(); fixture.image_meta.len()];
    for observation in &fixture.observations {
        let image_keypoints = &mut per_image_keypoints[observation.view.index()];
        if image_keypoints.len() <= observation.feature_id.index() {
            image_keypoints.resize(observation.feature_id.index() + 1, None);
        }
        image_keypoints[observation.feature_id.index()] = Some(
            Keypoint::try_new(observation.xy.x as f32, observation.xy.y as f32, 1.0, 0.0).unwrap(),
        );
    }

    per_image_keypoints
        .into_iter()
        .map(|keypoints| {
            let keypoints: Vec<_> = keypoints
                .into_iter()
                .map(|keypoint| keypoint.expect("oracle feature ids are contiguous per image"))
                .collect();
            let descriptors = DescriptorSet::try_f32(vec![0.0; keypoints.len()], 1).unwrap();
            ExtractedFeatures::try_new(keypoints, descriptors).unwrap()
        })
        .collect()
}

struct StaticExtractor;

impl FeatureExtractor for StaticExtractor {
    fn extract(&self, _ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        Ok(two_features())
    }
}

struct ReversedMatcher;

impl PairMatcher for ReversedMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        PairMatches::try_new(
            ctx.pair,
            vec![
                FeatureMatch::try_new(FeatureId::new(1), FeatureId::new(0), Some(0.2))?,
                FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(1), Some(0.1))?,
            ],
        )
    }
}

struct EssentialVerifier;

impl GeometricVerifier for EssentialVerifier {
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
            ErrorStats::from_values(&vec![0.0; matches.len()], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "test-essential",
            3,
            TerminationReason::Converged,
            Vec::new(),
        )?;
        let relative_pose =
            RelativePose::from_baseline_direction(Rotation3::identity(), Vector3::x_axis())?;
        let model = GeometricModel::Essential {
            matrix: Matrix3::identity(),
            relative_pose,
            evidence,
        };
        model.validate_for_matches(matches)?;
        Ok(VerificationVerdict::Accepted {
            model,
            report: PairVerificationReport {
                pair: ctx.pair,
                input_match_count: matches.len(),
                inlier_count: matches.len(),
                message: None,
            },
        })
    }
}

#[test]
fn adapters_traits_round_trip_features_and_sorted_matches() {
    let extractor = StaticExtractor;
    let matcher = ReversedMatcher;
    let meta = ImageMeta::external_id("image-a", 64, 48);
    let features = extractor
        .extract(
            &FeatureContext {
                view: ViewId::new(0),
                seed: 11.into(),
            },
            &meta,
        )
        .unwrap();

    assert_eq!(features.len(), 2);
    assert_eq!(features.keypoints()[0].x(), 10.0);

    let matches = matcher
        .match_features(
            &PairContext {
                pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
                view_i: ViewId::new(0),
                view_j: ViewId::new(1),
                root_seed: 12.into(),
                seed: 12.into(),
            },
            &features,
            &features,
        )
        .unwrap();

    let ordered: Vec<_> = matches
        .as_slice()
        .iter()
        .map(|matched| (matched.first().index(), matched.second().index()))
        .collect();
    assert_eq!(ordered, vec![(0, 1), (1, 0)]);
}

#[test]
fn adapters_pair_matcher_wraps_brute_force_with_typed_matches() {
    let matcher =
        PairMatcherAdapter::new(MatcherType::BruteForce, 0.8, MatchAcceptance::Reciprocal);
    let features = two_features();

    let matches = matcher
        .match_features(
            &PairContext {
                pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
                view_i: ViewId::new(0),
                view_j: ViewId::new(1),
                root_seed: 99.into(),
                seed: 99.into(),
            },
            &features,
            &features,
        )
        .unwrap();

    let typed: Vec<_> = matches
        .as_slice()
        .iter()
        .map(|matched| (matched.first().index(), matched.second().index()))
        .collect();
    assert_eq!(
        matches.pair(),
        ViewPair::new(ViewId::new(0), ViewId::new(1))
    );
    assert_eq!(typed, vec![(0, 0), (1, 1)]);
}

#[test]
fn adapters_workspace_ops_populate_feature_match_and_verification_stores() {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();

    workspace
        .extract_features(&StaticExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&ReversedMatcher, PairSelector::Active)
        .unwrap()
        .verify_pairs(&EssentialVerifier, PairSelector::Active)
        .unwrap();

    workspace.validate().unwrap();
    assert_eq!(workspace.feature_store().populated_count(), 2);
    assert_eq!(workspace.match_store().populated_count(), 1);
    assert_eq!(workspace.geometric_model_store().populated_count(), 1);
    assert_eq!(workspace.pair_graph().active_count(), 1);
}

/// External sink built only from the public observer surface.
#[derive(Default)]
struct StageLog(std::sync::Mutex<Vec<(progress::SfmProgressStage, &'static str, usize)>>);

impl progress::SfmProgressSink for StageLog {
    fn begin(&self, stage: progress::SfmProgressStage, total: usize) {
        self.0.lock().unwrap().push((stage, "begin", total));
    }

    fn advance(&self, stage: progress::SfmProgressStage, completed: usize) {
        self.0.lock().unwrap().push((stage, "advance", completed));
    }

    fn finish(&self, stage: progress::SfmProgressStage) {
        self.0.lock().unwrap().push((stage, "finish", 0));
    }
}

#[test]
fn workspace_ops_report_front_end_progress_through_a_public_sink() {
    use progress::SfmProgressStage::{FeatureExtraction, Matching, Verification};

    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();
    let progress = StageLog::default();

    workspace
        .extract_features_with_progress(&StaticExtractor, ViewSelector::All, &progress)
        .unwrap()
        .match_pairs_with_progress(&ReversedMatcher, PairSelector::Active, &progress)
        .unwrap();
    let report = workspace
        .verify_pairs_with_progress(&EssentialVerifier, PairSelector::Active, &progress)
        .unwrap();

    assert_eq!(report.accepted.len(), 1);
    assert_eq!(workspace.geometric_model_store().populated_count(), 1);
    assert_eq!(
        *progress.0.lock().unwrap(),
        [
            (FeatureExtraction, "begin", 2),
            (FeatureExtraction, "advance", 1),
            (FeatureExtraction, "advance", 1),
            (FeatureExtraction, "finish", 0),
            (Matching, "begin", 1),
            (Matching, "advance", 1),
            (Matching, "finish", 0),
            (Verification, "begin", 1),
            (Verification, "advance", 1),
            (Verification, "finish", 0),
        ]
    );
}

#[test]
fn verification_batch_report_preserves_rejection_reason() {
    let mut workspace = SfmWorkspace::builder()
        .image_meta(vec![
            ImageMeta::external_id("image-a", 64, 48),
            ImageMeta::external_id("image-b", 64, 48),
        ])
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![(ViewId::new(0), ViewId::new(1))])
        .build()
        .unwrap();
    workspace
        .extract_features(&StaticExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&ReversedMatcher, PairSelector::All)
        .unwrap();

    let report = workspace
        .verify_pairs_report(&CalibratedVerifier::default(), PairSelector::All)
        .unwrap();
    assert_eq!(report.attempted_pairs(), 1);
    assert_eq!(report.accepted_pairs(), 0);
    assert_eq!(report.rejected_pairs(), 1);
    assert!(matches!(
        report.rejected[0].reason,
        VerificationRejectReason::LowMatchCount {
            required: 8,
            actual: 2
        }
    ));
}

#[cfg(feature = "parallel")]
#[test]
fn verification_uses_the_callers_rayon_pool() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct PoolObservingVerifier<'a>(&'a AtomicUsize);

    impl GeometricVerifier for PoolObservingVerifier<'_> {
        fn verify_pair(
            &self,
            ctx: &VerificationContext,
            fi: &ExtractedFeatures,
            fj: &ExtractedFeatures,
            matches: &PairMatches,
            ki: &openmvg_sfm::Camera,
            kj: &openmvg_sfm::Camera,
        ) -> Result<VerificationVerdict> {
            self.0
                .fetch_max(rayon::current_num_threads(), Ordering::Relaxed);
            EssentialVerifier.verify_pair(ctx, fi, fj, matches, ki, kj)
        }
    }

    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..4)
                .map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48))
                .collect::<Vec<_>>(),
        )
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(
            (0..4)
                .flat_map(|first| {
                    ((first + 1)..4).map(move |second| (ViewId::new(first), ViewId::new(second)))
                })
                .collect::<Vec<_>>(),
        )
        .build()
        .unwrap();
    workspace
        .extract_features(&StaticExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&ReversedMatcher, PairSelector::All)
        .unwrap();

    let observed_threads = AtomicUsize::new(0);
    let verifier = PoolObservingVerifier(&observed_threads);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let report = pool
        .install(|| workspace.verify_pairs_report(&verifier, PairSelector::All))
        .unwrap();

    assert_eq!(report.accepted_pairs(), 6);
    assert_eq!(observed_threads.load(Ordering::Relaxed), 2);
}

#[test]
fn verification_failure_is_deterministic_and_transactional() {
    struct FailingVerifier;

    impl GeometricVerifier for FailingVerifier {
        fn verify_pair(
            &self,
            ctx: &VerificationContext,
            fi: &ExtractedFeatures,
            fj: &ExtractedFeatures,
            matches: &PairMatches,
            ki: &openmvg_sfm::Camera,
            kj: &openmvg_sfm::Camera,
        ) -> Result<VerificationVerdict> {
            if ctx.pair == ViewPair::new(ViewId::new(0), ViewId::new(1)) {
                return Err(openmvg_sfm::SfmError::InvalidInput(
                    "first verification failure".to_owned(),
                ));
            }
            if ctx.pair == ViewPair::new(ViewId::new(1), ViewId::new(2)) {
                return Err(openmvg_sfm::SfmError::InvalidInput(
                    "later verification failure".to_owned(),
                ));
            }
            EssentialVerifier.verify_pair(ctx, fi, fj, matches, ki, kj)
        }
    }

    let mut workspace = SfmWorkspace::builder()
        .image_meta(
            (0..3)
                .map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48))
                .collect::<Vec<_>>(),
        )
        .camera_models(vec![Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![
            (ViewId::new(0), ViewId::new(1)),
            (ViewId::new(0), ViewId::new(2)),
            (ViewId::new(1), ViewId::new(2)),
        ])
        .build()
        .unwrap();
    workspace
        .extract_features(&StaticExtractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&ReversedMatcher, PairSelector::All)
        .unwrap();

    let error = workspace
        .verify_pairs_report(&FailingVerifier, PairSelector::All)
        .unwrap_err();
    assert!(error.to_string().contains("first verification failure"));
    assert_eq!(workspace.geometric_model_store().populated_count(), 0);
    assert_eq!(workspace.pair_graph().active_count(), 3);
}

#[test]
fn verification_adapter_accepts_model_with_matching_inlier_mask() {
    let verifier = EssentialVerifier;
    let features = two_features();
    let matches = PairMatches::try_new(
        ViewPair::new(ViewId::new(0), ViewId::new(1)),
        vec![
            FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(0), None).unwrap(),
            FeatureMatch::try_new(FeatureId::new(1), FeatureId::new(1), None).unwrap(),
        ],
    )
    .unwrap();
    let intrinsics = Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap();

    let outcome = verifier
        .verify_pair(
            &VerificationContext {
                pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
                view_i: ViewId::new(0),
                view_j: ViewId::new(1),
                seed: 13.into(),
            },
            &features,
            &features,
            &matches,
            &intrinsics,
            &intrinsics,
        )
        .unwrap();

    match outcome {
        VerificationVerdict::Accepted { model, report } => {
            assert_eq!(report.inlier_count, 2);
            assert_eq!(model.evidence().inlier_ratio, 1.0);
            assert!(model.relative_pose().is_some());
        }
        VerificationVerdict::Rejected { .. } => panic!("expected accepted verification outcome"),
    }
}

#[test]
fn verification_reject_reason_is_structured() {
    let report = PairVerificationReport {
        pair: ViewPair::new(ViewId::new(1), ViewId::new(2)),
        input_match_count: 4,
        inlier_count: 0,
        message: Some("too few matches".to_owned()),
    };
    let outcome = VerificationVerdict::Rejected {
        reason: VerificationRejectReason::LowMatchCount {
            required: 5,
            actual: 4,
        },
        report,
    };

    match outcome {
        VerificationVerdict::Rejected {
            reason: VerificationRejectReason::LowMatchCount { required, actual },
            report,
        } => {
            assert_eq!(required, 5);
            assert_eq!(actual, 4);
            assert_eq!(report.pair, ViewPair::new(ViewId::new(1), ViewId::new(2)));
        }
        VerificationVerdict::Accepted { .. } => panic!("expected rejected verification outcome"),
        _ => panic!("expected low-match-count rejection"),
    }
}

#[test]
fn calibrated_verifier_rejects_low_match_count() {
    let verifier = CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 30).unwrap();
    let features = two_features();
    let matches = PairMatches::try_new(
        ViewPair::new(ViewId::new(0), ViewId::new(1)),
        vec![FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(0), None).unwrap()],
    )
    .unwrap();
    let intrinsics = Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap();

    let outcome = verifier
        .verify_pair(
            &VerificationContext {
                pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
                view_i: ViewId::new(0),
                view_j: ViewId::new(1),
                seed: 13.into(),
            },
            &features,
            &features,
            &matches,
            &intrinsics,
            &intrinsics,
        )
        .unwrap();

    match outcome {
        VerificationVerdict::Rejected {
            reason: VerificationRejectReason::LowMatchCount { required, actual },
            report,
        } => {
            assert_eq!(required, 8);
            assert_eq!(actual, 1);
            assert_eq!(report.input_match_count, 1);
        }
        _ => panic!("expected low-match-count rejection"),
    }
}

#[test]
fn calibrated_verifier_accepts_oracle_correspondences() {
    let fixture = common::oracle_fixture::fixture();
    let features = oracle_pipeline_features(&fixture);
    let oracle_matches = fixture
        .matches
        .iter()
        .find(|pair_matches| pair_matches.matches.len() >= 12)
        .expect("oracle fixture should contain a well-constrained pair");
    assert!(fixture.pairs.contains(&oracle_matches.pair));
    let pair = oracle_matches.pair;
    let matches = PairMatches::try_new(
        pair,
        oracle_matches
            .matches
            .iter()
            .map(|(feature_i, feature_j)| {
                FeatureMatch::try_new(*feature_i, *feature_j, None).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let verifier = CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 8).unwrap();

    let outcome = verifier
        .verify_pair(
            &VerificationContext {
                pair,
                view_i: oracle_matches.pair.first(),
                view_j: oracle_matches.pair.second(),
                seed: 123.into(),
            },
            &features[oracle_matches.pair.first().index()],
            &features[oracle_matches.pair.second().index()],
            &matches,
            &fixture.camera_models[0],
            &fixture.camera_models[0],
        )
        .unwrap();

    match outcome {
        VerificationVerdict::Accepted { model, report } => {
            assert_eq!(report.pair, pair);
            assert_eq!(report.input_match_count, matches.len());
            assert!(report.inlier_count >= 8);
            assert_eq!(model.evidence().inliers.len(), matches.len());
            assert!(model.evidence().inlier_count >= 8);
            assert!(model.evidence().inlier_ratio > 0.0);
            assert!(model.relative_pose().is_some());
        }
        VerificationVerdict::Rejected { reason, report } => {
            panic!(
                "expected oracle correspondences to verify, got {:?}: {:?}",
                reason, report
            )
        }
    }
}

#[test]
fn calibrated_verifier_recovers_clean_oracle_pair_deterministically() {
    let fixture = common::oracle_fixture::fixture();
    let features = oracle_pipeline_features(&fixture);
    let oracle_matches = fixture
        .matches
        .iter()
        .find(|pair_matches| pair_matches.matches.len() >= 30)
        .expect("oracle fixture should contain a well-constrained pair");
    assert!(fixture.pairs.contains(&oracle_matches.pair));
    let pair = oracle_matches.pair;
    let matches = PairMatches::try_new(
        pair,
        oracle_matches
            .matches
            .iter()
            .enumerate()
            .map(|(index, (feature_i, feature_j))| {
                FeatureMatch::try_new(*feature_i, *feature_j, Some(index as f32 / 100.0)).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let verifier = CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 16).unwrap();
    let context = VerificationContext {
        pair,
        view_i: oracle_matches.pair.first(),
        view_j: oracle_matches.pair.second(),
        seed: 123.into(),
    };
    let verify = || {
        verifier
            .verify_pair(
                &context,
                &features[context.view_i.index()],
                &features[context.view_j.index()],
                &matches,
                &fixture.camera_models[0],
                &fixture.camera_models[0],
            )
            .unwrap()
    };
    let first = verify();
    let second = verify();
    assert_eq!(first, second, "fixed pair seed must reproduce exactly");

    let VerificationVerdict::Accepted { model, report } = first else {
        panic!("expected clean oracle correspondences to verify");
    };
    assert_eq!(report.input_match_count, matches.len());
    assert!(report.inlier_count >= 30);
    let diagnostics = model.evidence().diagnostics.as_ref().unwrap();
    assert_eq!(diagnostics.solver_name, CALIBRATED_VERIFIER_POLICY_ID);
    assert_eq!(
        diagnostics.inlier_threshold.units(),
        ResidualUnit::NormalizedImage
    );

    let estimated = model.relative_pose().unwrap();
    let pose_i = fixture
        .poses_gt
        .iter()
        .find(|(image, _)| *image == context.view_i)
        .unwrap()
        .1
        .clone();
    let pose_j = fixture
        .poses_gt
        .iter()
        .find(|(image, _)| *image == context.view_j)
        .unwrap()
        .1
        .clone();
    let expected_rotation = pose_j.rotation() * pose_i.rotation().inverse();
    let rotation_error_deg = (estimated.rotation_source_to_target().inverse() * expected_rotation)
        .angle()
        .to_degrees();
    let expected_center_direction =
        (pose_i.rotation() * (pose_j.center() - pose_i.center())).normalize();
    let translation_error_deg = estimated
        .baseline_direction_in_source()
        .as_ref()
        .dot(&expected_center_direction)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    assert!(
        rotation_error_deg < 0.25,
        "rotation error={rotation_error_deg}"
    );
    assert!(
        translation_error_deg < 0.5,
        "translation-direction error={translation_error_deg}"
    );
}

#[test]
fn calibrated_verifier_restores_mask_after_corrupted_quality_order() {
    use openmvg_sfm::testing::oracle::OracleMatchCorruptionConfig;

    let mut fixture = common::oracle_fixture::fixture();
    fixture
        .inject_match_outliers(&OracleMatchCorruptionConfig::diffuse(0.30))
        .unwrap();
    let features = oracle_pipeline_features(&fixture);
    let oracle_matches = fixture
        .matches
        .iter()
        .find(|pair_matches| pair_matches.matches.len() >= 30)
        .expect("oracle fixture should contain a well-constrained pair");
    assert!(fixture.pairs.contains(&oracle_matches.pair));
    let pair = oracle_matches.pair;
    let match_count = oracle_matches.matches.len();
    let matches = PairMatches::try_new(
        pair,
        oracle_matches
            .matches
            .iter()
            .enumerate()
            .map(|(index, (feature_i, feature_j))| {
                // Reverse the geometric fixture's order so the verifier must map the
                // estimator's quality-ordered inlier indices back to PairMatches order.
                FeatureMatch::try_new(*feature_i, *feature_j, Some((match_count - index) as f32))
                    .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let expected_by_match = oracle_matches
        .matches
        .iter()
        .copied()
        .zip(oracle_matches.inlier_mask.iter().copied())
        .collect::<HashMap<_, _>>();
    let expected_mask = matches
        .as_slice()
        .iter()
        .map(|matched| expected_by_match[&(matched.first(), matched.second())])
        .collect::<Vec<_>>();
    let verifier = CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 16).unwrap();
    let context = VerificationContext {
        pair,
        view_i: oracle_matches.pair.first(),
        view_j: oracle_matches.pair.second(),
        seed: 9876.into(),
    };
    let verify = || {
        verifier
            .verify_pair(
                &context,
                &features[context.view_i.index()],
                &features[context.view_j.index()],
                &matches,
                &fixture.camera_models[0],
                &fixture.camera_models[0],
            )
            .unwrap()
    };
    let first = verify();
    assert_eq!(first, verify(), "fixed pair seed must reproduce exactly");

    let VerificationVerdict::Accepted { model, .. } = first else {
        panic!("expected mixed oracle correspondences to verify");
    };
    let predicted_mask = model.evidence().inliers.as_slice();
    let true_inliers = expected_mask.iter().filter(|is_inlier| **is_inlier).count();
    let true_outliers = expected_mask.len() - true_inliers;
    let retained_inliers = predicted_mask
        .iter()
        .zip(&expected_mask)
        .filter(|(predicted, expected)| **predicted && **expected)
        .count();
    let rejected_outliers = predicted_mask
        .iter()
        .zip(&expected_mask)
        .filter(|(predicted, expected)| !**predicted && !**expected)
        .count();
    let false_inliers = predicted_mask
        .iter()
        .zip(&expected_mask)
        .filter(|(predicted, expected)| **predicted && !**expected)
        .count();
    let inlier_retention = retained_inliers as f64 / true_inliers as f64;
    let outlier_rejection = rejected_outliers as f64 / true_outliers as f64;
    let inlier_precision = retained_inliers as f64 / (retained_inliers + false_inliers) as f64;
    assert!(
        inlier_retention >= 0.95,
        "clean inlier retention={inlier_retention}"
    );
    assert!(
        outlier_rejection >= 0.80,
        "outlier rejection={outlier_rejection}"
    );
    assert!(
        inlier_precision >= 0.90,
        "inlier precision={inlier_precision}"
    );
}

#[test]
fn verification_model_rejects_inlier_mask_with_wrong_length() {
    let matches = PairMatches::try_new(
        ViewPair::new(ViewId::new(0), ViewId::new(1)),
        vec![FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(0), None).unwrap()],
    )
    .unwrap();
    let evidence = ModelEvidence::try_new(
        InlierMask::all_inliers(2),
        ErrorStats::from_values(&[0.0, 0.1], &[1.0]),
        openmvg_core::PixelThreshold::ONE_PIXEL.into(),
        "test-essential",
        3,
        TerminationReason::Converged,
        Vec::new(),
    )
    .unwrap();
    let model = GeometricModel::Essential {
        matrix: Matrix3::identity(),
        relative_pose: RelativePose::from_baseline_direction(
            Rotation3::identity(),
            Vector3::x_axis(),
        )
        .unwrap(),
        evidence,
    };

    let err = model.validate_for_matches(&matches).unwrap_err();
    assert_eq!(
        err.to_string(),
        "invalid input: inlier mask length 2 does not match match count 1"
    );
}

#[test]
fn triangulation_dlt_wraps_core_solver() {
    let camera = Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
    let point = Point3::new(0.2, 0.1, 4.0);
    let poses = [
        (
            ViewId::new(0),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0.0, 0.0, 0.0),
            )
            .unwrap(),
        ),
        (
            ViewId::new(1),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
        ),
    ];
    let observations: Vec<_> = poses
        .iter()
        .map(|(view_id, pose)| {
            let camera_point = pose.project_to_camera(&point);
            let xy = camera.project(&camera_point.coords).unwrap();
            PosedPixelObservation {
                view: *view_id,
                xy,
                pose,
                camera: &camera,
            }
        })
        .collect();

    let TriangulationVerdict::Accepted(triangulated) = DltTriangulator::default()
        .triangulate(&observations)
        .unwrap()
    else {
        panic!("clean DLT fixture should be accepted");
    };

    assert_eq!(triangulated.supporting_observation_count(), 2);
    assert!((triangulated.point - point).norm() <= 1e-8);
    assert!(triangulated.reprojection_error.value() <= 1e-8);
}

#[test]
fn triangulation_dlt_rejects_spherical_cameras_explicitly() {
    let camera = openmvg_sfm::Camera::Spherical(
        openmvg_core::camera::SphericalCamera::try_new(1024).unwrap(),
    );
    let poses = [
        WorldToCameraPose::identity(),
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(1.0, 0.0, 0.0))
            .unwrap(),
    ];
    let observations = [
        PosedPixelObservation {
            view: ViewId::new(0),
            xy: Point2::new(512.0, 256.0),
            pose: &poses[0],
            camera: &camera,
        },
        PosedPixelObservation {
            view: ViewId::new(1),
            xy: Point2::new(500.0, 256.0),
            pose: &poses[1],
            camera: &camera,
        },
    ];

    assert!(matches!(
        DltTriangulator::default().triangulate(&observations),
        Err(SfmError::UnsupportedCameraModel {
            context: "DLT triangulation",
            camera_model: "spherical",
        })
    ));
}

#[test]
fn triangulation_dlt_reports_original_distorted_pixel_residual() {
    let camera = Camera::try_brown_conrady(
        500.0,
        500.0,
        320.0,
        240.0,
        RadialDistortion::K1K2K3(-0.25, 0.08, 0.0),
        TangentialDistortion::P1P2(0.002, -0.001),
    )
    .unwrap();
    let point = Point3::new(0.8, 0.4, 3.0);
    let poses = [
        (ViewId::new(0), WorldToCameraPose::identity()),
        (
            ViewId::new(1),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
        ),
    ];
    let mut pixels = poses
        .iter()
        .map(|(_, pose)| {
            let camera_point = pose.project_to_camera(&point);
            camera.project(&camera_point.coords).unwrap()
        })
        .collect::<Vec<_>>();
    pixels[1].x += 0.5;
    let observations = poses
        .iter()
        .zip(&pixels)
        .map(|((image, pose), xy)| PosedPixelObservation {
            view: *image,
            xy: *xy,
            pose,
            camera: &camera,
        })
        .collect::<Vec<_>>();

    let TriangulationVerdict::Accepted(triangulated) = DltTriangulator::default()
        .triangulate(&observations)
        .unwrap()
    else {
        panic!("distorted DLT fixture should be accepted");
    };
    let expected = observations
        .iter()
        .map(|observation| {
            let camera_point = observation.pose.project_to_camera(&triangulated.point);
            (camera.project(&camera_point.coords).unwrap() - observation.xy).norm()
        })
        .sum::<f64>()
        / observations.len() as f64;

    assert!((triangulated.reprojection_error.value() - expected).abs() < 1e-10);
}

#[test]
fn robust_triangulation_adapter_returns_filtered_input_indices() {
    let point = Point3::new(0.2, -0.1, 6.0);
    let camera = Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
    let poses = [
        (ViewId::new(0), WorldToCameraPose::identity()),
        (
            ViewId::new(1),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
        ),
        (
            ViewId::new(2),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0.0, 1.0, 0.0),
            )
            .unwrap(),
        ),
    ];
    let mut pixels: Vec<_> = poses
        .iter()
        .map(|(image, pose)| {
            let camera_point = pose.project_to_camera(&point);
            (*image, camera.project(&camera_point.coords).unwrap())
        })
        .collect();
    pixels[2].1 += nalgebra::Vector2::new(80.0, -50.0);
    let observations = pixels
        .iter()
        .zip(&poses)
        .map(|((image, xy), (_, pose))| PosedPixelObservation {
            view: *image,
            xy: *xy,
            pose,
            camera: &camera,
        })
        .collect::<Vec<_>>();

    let TriangulationVerdict::Accepted(triangulated) = TrackTriangulatorAdapter::default()
        .triangulate(&observations)
        .expect("valid robust request")
    else {
        panic!("clean pair should survive one outlier");
    };

    assert_eq!(&*triangulated.supporting_observation_indices, &[0, 1]);
    assert!((triangulated.point - point).norm() < 1e-8);
}

#[test]
fn averaged_pose_triangulation_tolerates_degree_level_rotation_error() {
    let point = Point3::new(0.2, -0.1, 6.0);
    let camera = Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
    let exact = [
        WorldToCameraPose::identity(),
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(1.0, 0.0, 0.0))
            .unwrap(),
    ];
    // A typical rotation-averaging error: tilting the second camera out of the epipolar plane
    // makes the two rays skew instead of intersecting.
    let averaged = [
        exact[0].clone(),
        WorldToCameraPose::from_rotation_center(
            Rotation3::from_axis_angle(&Vector3::x_axis(), 1.5_f64.to_radians()),
            Point3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
    ];
    let observations = exact
        .iter()
        .zip(&averaged)
        .enumerate()
        .map(|(index, (truth, pose))| {
            let camera_point = truth.project_to_camera(&point);
            PosedPixelObservation {
                view: ViewId::new(index),
                xy: camera.project(&camera_point.coords).unwrap(),
                pose,
                camera: &camera,
            }
        })
        .collect::<Vec<_>>();

    let refined_pose_outcome = TrackTriangulatorAdapter::default()
        .triangulate(&observations)
        .expect("valid robust request");
    assert!(matches!(
        refined_pose_outcome,
        TriangulationVerdict::Rejected(_)
    ));

    let TriangulationVerdict::Accepted(triangulated) =
        TrackTriangulatorAdapter::for_averaged_poses()
            .expect("valid averaged-pose policy")
            .triangulate(&observations)
            .expect("valid averaged-pose request")
    else {
        panic!("degree-level pose error should not reject a clean track");
    };
    assert_eq!(&*triangulated.supporting_observation_indices, &[0, 1]);
}

#[test]
fn robust_triangulation_adapter_preserves_typed_rejection_reason() {
    let point = Point3::new(0.0, 0.0, 100.0);
    let camera = Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
    let poses = [
        (ViewId::new(0), WorldToCameraPose::identity()),
        (
            ViewId::new(1),
            WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(0.1, 0.0, 0.0),
            )
            .unwrap(),
        ),
    ];
    let observations = poses
        .iter()
        .map(|(image, pose)| {
            let camera_point = pose.project_to_camera(&point);
            PosedPixelObservation {
                view: *image,
                xy: camera.project(&camera_point.coords).unwrap(),
                pose,
                camera: &camera,
            }
        })
        .collect::<Vec<_>>();

    let outcome = TrackTriangulatorAdapter::default()
        .triangulate(&observations)
        .expect("valid request should return a typed outcome");

    assert_eq!(
        outcome,
        TriangulationVerdict::Rejected(TriangulationRejectReason::InsufficientParallax)
    );
}
