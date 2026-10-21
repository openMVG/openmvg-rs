// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::collections::BTreeMap;

use nalgebra::Vector3;
use openmvg_geometry::averaging::{average_rotations, average_translations};
use openmvg_sfm::{
    config::*,
    model::*,
    testing::seams::*,
    unstable::reports::*,
    unstable::*,
    unstable::{
        adapters::{triangulation::*, verification::*},
        backends::*,
        evaluation::*,
        incremental::*,
        types::*,
    },
    *,
};

const SYNTHETIC_ROOT_SEED: u64 = 7;

#[test]
fn algorithm_synthetic_acceptance_runs_real_geometry_stages() {
    let fixture = common::oracle_fixture::fixture();
    let mut workspace = workspace_from_fixture(&fixture);

    let config = GlobalAveragingConfig::default();
    workspace
        .extract_features(&OracleFeatureExtractor::new(&fixture), ViewSelector::All)
        .unwrap();
    workspace
        .match_pairs(&OracleMatcher::new(&fixture), PairSelector::Active)
        .unwrap();
    workspace
        .verify_pairs(
            &CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 30).unwrap(),
            PairSelector::Active,
        )
        .unwrap();
    openmvg_sfm::unstable::ops::averaging::run_global_averaging_scoped(
        &mut workspace,
        average_rotations,
        average_translations,
        config
            .min_translation_parallax()
            .map(|threshold| threshold.value().to_degrees()),
        config.min_edge_isotropy(),
    )
    .unwrap();
    workspace.build_tracks().unwrap();
    workspace.release_matching_data().unwrap();
    openmvg_sfm::unstable::ops::reconstruction::triangulate(
        &mut workspace,
        &DltTriangulator::default(),
        config.gauge_scale_policy(),
    )
    .unwrap();

    let quality = workspace.quality_report();
    assert_acceptance_quality(&quality);

    let reconstruction = reconstruction_from_workspace(&workspace);
    let evaluation = evaluate_reconstruction_against_gt(
        &reconstruction,
        &fixture.scene,
        &EvaluationConfig::default(),
    )
    .unwrap();
    let relative_rotation_are = evaluation.relative_rotation_are.clone().unwrap();
    let relative_translation_are = evaluation.relative_translation_are.clone().unwrap();
    assert!(relative_rotation_are.median <= 1.0);
    assert!(relative_translation_are.median <= 5.0);
    assert!(evaluation.ate.rmse <= 5.0);
    // The orientation-derived gauge resolves the absolute-rotation ambiguity even
    // on this near-coplanar camera arrangement, so absolute rotation error is
    // meaningful (and small) rather than dominated by the gauge artifact.
    assert!(
        evaluation.absolute_rotation_are.median <= 1.0,
        "absolute rotation median={}",
        evaluation.absolute_rotation_are.median
    );

    let median_angle = median(triangulation_angles_deg(&workspace));
    assert!(
        median_angle >= 1.0,
        "median triangulation angle={median_angle}"
    );
}

#[test]
fn incremental_circle_cube_recovers_every_camera() {
    let fixture = common::oracle_fixture::fixture();
    let result = run_incremental_fixture(&fixture, OracleMatcher::new(&fixture));

    assert_eq!(result.verification.rejected_pairs(), 0);
    assert_eq!(result.report.initialization.seed.views.len(), 2);
    assert_eq!(result.report.initialization.initialized_views.len(), 2);
    assert_eq!(
        result.report.registered_view_count,
        fixture.image_meta.len()
    );
    assert_eq!(
        result.report.growth.grown_view_count,
        fixture.image_meta.len() - 2
    );
    assert_eq!(
        result.report.growth.growth_registrations.len(),
        result.report.growth.grown_view_count
    );
    assert!(
        result
            .report
            .growth
            .growth_registrations
            .iter()
            .all(|registration| {
                registration.verified_registered_neighbor_count > 0
                    && registration.inlier_count <= registration.attempted_observation_count
            })
    );
    assert!(result.report.landmark_count >= 30);
    assert_incremental_accuracy(&result.workspace, &fixture);
}

#[test]
fn incremental_circle_cube_detects_and_survives_adversarial_matches() {
    use openmvg_sfm::testing::oracle::OracleMatchCorruptionConfig;

    let mut fixture = common::oracle_fixture::fixture();
    let poisoned_pair = (ViewId::new(0), ViewId::new(1));
    let corruption = fixture
        .inject_match_outliers(&OracleMatchCorruptionConfig::with_poisoned_pair(
            0.1,
            poisoned_pair.0,
            poisoned_pair.1,
        ))
        .unwrap();
    let result = run_incremental_fixture(&fixture, OracleMatcher::new(&fixture));
    let detection = score_match_outlier_detection(&fixture, &result.workspace);
    let poisoned_pair = ViewPair::new(poisoned_pair.0, poisoned_pair.1);

    assert!(
        result
            .verification
            .rejected
            .iter()
            .any(|rejected| rejected.report.pair == poisoned_pair),
        "the fully deranged pair should be rejected"
    );
    assert!(
        detection.outlier_recall() >= 0.9,
        "outlier recall={} ({detection:?})",
        detection.outlier_recall()
    );
    assert!(
        detection.outlier_precision() >= 0.85,
        "outlier precision={} ({detection:?})",
        detection.outlier_precision()
    );
    assert!(
        detection.inlier_retention() >= 0.95,
        "inlier retention={} ({detection:?})",
        detection.inlier_retention()
    );
    assert_eq!(
        detection.true_outliers(),
        corruption.total_outliers,
        "corruption labels and detector accounting must cover the same matches"
    );
    let mixed_tracks = mixed_track_count(&result.workspace, &fixture);
    let mixed_landmark_supports = mixed_landmark_support_count(&result.workspace, &fixture);
    eprintln!(
        "adversarial correspondence detection: recall={:.1}%, precision={:.1}%, clean retention={:.1}%, mixed tracks={mixed_tracks}, mixed active landmark supports={mixed_landmark_supports}",
        detection.outlier_recall() * 100.0,
        detection.outlier_precision() * 100.0,
        detection.inlier_retention() * 100.0,
    );
    assert!(
        mixed_landmark_supports <= 2,
        "at most two mixed landmark supports may survive this adversarial fixture; got {mixed_landmark_supports}"
    );
    assert_eq!(result.report.initialization.seed.views.len(), 2);
    assert_eq!(
        result.report.registered_view_count,
        fixture.image_meta.len()
    );
    assert_eq!(
        result.report.growth.grown_view_count,
        fixture.image_meta.len() - 2
    );
    assert_eq!(
        result.report.growth.growth_registrations.len(),
        result.report.growth.grown_view_count
    );
    assert!(
        result
            .report
            .growth
            .growth_registrations
            .iter()
            .all(|registration| registration.verified_registered_neighbor_count > 0)
    );
    assert!(result.report.landmark_count >= 25);
    assert_incremental_accuracy(&result.workspace, &fixture);
}

#[derive(Debug, Default)]
struct MatchOutlierDetection {
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
    true_negative: usize,
}

impl MatchOutlierDetection {
    fn true_outliers(&self) -> usize {
        self.true_positive + self.false_negative
    }

    fn outlier_recall(&self) -> f64 {
        ratio(self.true_positive, self.true_outliers())
    }

    fn outlier_precision(&self) -> f64 {
        ratio(self.true_positive, self.true_positive + self.false_positive)
    }

    fn inlier_retention(&self) -> f64 {
        ratio(self.true_negative, self.true_negative + self.false_positive)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn score_match_outlier_detection(
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
    workspace: &SfmWorkspace,
) -> MatchOutlierDetection {
    let mut detection = MatchOutlierDetection::default();
    for pair_matches in &fixture.matches {
        let pair = pair_matches.pair;
        let predicted_inliers = workspace
            .geometric_models_view()
            .get(pair)
            .map(|model| model.evidence().inliers.as_slice());
        for (index, &true_inlier) in pair_matches.inlier_mask.iter().enumerate() {
            let predicted_inlier = predicted_inliers
                .and_then(|mask| mask.get(index))
                .copied()
                .unwrap_or(false);
            match (true_inlier, predicted_inlier) {
                (false, false) => detection.true_positive += 1,
                (true, false) => detection.false_positive += 1,
                (false, true) => detection.false_negative += 1,
                (true, true) => detection.true_negative += 1,
            }
        }
    }
    detection
}

fn mixed_track_count(
    workspace: &SfmWorkspace,
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
) -> usize {
    let point_by_observation: BTreeMap<_, _> = fixture
        .observations
        .iter()
        .map(|observation| {
            (
                (observation.view, observation.feature_id),
                observation.original_point_id,
            )
        })
        .collect();
    workspace
        .tracks()
        .as_slice()
        .iter()
        .filter(|track| {
            let mut point_ids = track.observations().iter().filter_map(|observation| {
                point_by_observation.get(&(observation.view, observation.feature))
            });
            let Some(first) = point_ids.next() else {
                return false;
            };
            point_ids.any(|point_id| point_id != first)
        })
        .count()
}

fn mixed_landmark_support_count(
    workspace: &SfmWorkspace,
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
) -> usize {
    let point_by_observation: BTreeMap<_, _> = fixture
        .observations
        .iter()
        .map(|observation| {
            (
                (observation.view, observation.feature_id),
                observation.original_point_id,
            )
        })
        .collect();
    workspace
        .landmarks()
        .iter_active()
        .filter(|landmark| {
            let track = workspace
                .tracks()
                .get(landmark.source_track)
                .expect("active landmark source track must exist");
            let mut point_ids = landmark
                .supporting_observations(track)
                .map(|(_, observation)| {
                    point_by_observation[&(observation.view, observation.feature)]
                });
            let Some(first) = point_ids.next() else {
                return false;
            };
            point_ids.any(|point_id| point_id != first)
        })
        .count()
}

struct IncrementalFixtureResult {
    workspace: SfmWorkspace,
    report: openmvg_sfm::unstable::incremental::IncrementalSfmReport,
    verification: VerificationBatchReport,
}

fn run_incremental_fixture(
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
    matcher: OracleMatcher,
) -> IncrementalFixtureResult {
    run_incremental_fixture_with(
        fixture,
        matcher,
        openmvg_sfm::testing::minimal_incremental_config(),
        None,
    )
}

fn run_incremental_fixture_with(
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
    matcher: OracleMatcher,
    config: IncrementalSfmConfig,
    ba_backend: Option<&dyn BundleAdjustmentBackend>,
) -> IncrementalFixtureResult {
    let mut workspace = workspace_from_fixture(fixture);
    let extractor = OracleFeatureExtractor::new(fixture);
    let verifier = CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 16).unwrap();

    workspace
        .extract_features(&extractor, ViewSelector::All)
        .unwrap()
        .match_pairs(&matcher, PairSelector::Active)
        .unwrap();
    let verification = workspace
        .verify_pairs_report(&verifier, PairSelector::Active)
        .unwrap();

    workspace
        .extract_features(&extractor, ViewSelector::Missing)
        .unwrap()
        .match_pairs(&matcher, PairSelector::Unmatched)
        .unwrap()
        .verify_pairs(&verifier, PairSelector::Unverified)
        .unwrap();
    workspace.build_tracks().unwrap();
    let report = incremental_sfm_with(
        &mut workspace,
        &config,
        &GenRansacP3pRegistrar::default(),
        &TrackTriangulatorAdapter::default(),
        ba_backend,
        None,
    )
    .unwrap();

    IncrementalFixtureResult {
        workspace,
        report,
        verification,
    }
}

fn assert_incremental_accuracy(
    workspace: &SfmWorkspace,
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
) {
    let reconstruction = reconstruction_from_workspace(workspace);
    let evaluation = evaluate_reconstruction_against_gt(
        &reconstruction,
        &fixture.scene,
        &EvaluationConfig::default(),
    )
    .unwrap();
    assert!(
        evaluation.ate.rmse <= 0.05,
        "ATE RMSE={} m",
        evaluation.ate.rmse
    );
    assert!(
        evaluation.absolute_rotation_are.max <= 0.25,
        "absolute rotation max={}°",
        evaluation.absolute_rotation_are.max
    );
    let relative_rotation = evaluation.relative_rotation_are.unwrap();
    assert!(
        relative_rotation.max <= 0.25,
        "relative rotation max={}°",
        relative_rotation.max
    );
    assert!(
        workspace
            .quality_report()
            .observation_reprojection_error_stats
            .median
            <= 0.5
    );
}

fn workspace_from_fixture(
    fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture,
) -> SfmWorkspace {
    SfmWorkspace::builder()
        .root_seed(SYNTHETIC_ROOT_SEED)
        .image_meta(fixture.image_meta.clone())
        .camera_models(fixture.camera_models.clone())
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(fixture.pairs.clone())
        .build()
        .unwrap()
}

fn assert_acceptance_quality(quality: &QualityReport) {
    assert_eq!(quality.view_count, 5);
    assert!(quality.registered_view_count >= 3);
    assert!(quality.registered_fraction >= 0.6);
    assert!(quality.verified_pair_count >= 4);
    assert!(quality.track_count >= 30);
    assert!(quality.landmark_count >= 30);
    assert!(quality.observation_reprojection_error_stats.median <= 2.0);
    assert!(quality.landmark_reprojection_error_stats.max <= 4.0);
}

fn reconstruction_from_workspace(workspace: &SfmWorkspace) -> GroundTruthReconstruction {
    let mut poses: Vec<_> = workspace
        .registered_views()
        .map(|(view, record)| (view, record.pose.clone()))
        .collect();
    poses.sort_by_key(|(view_id, _)| *view_id);

    let points = workspace
        .landmarks()
        .as_slice()
        .iter()
        .map(|landmark| (landmark.id, landmark.point))
        .collect();

    GroundTruthReconstruction { poses, points }
}

fn triangulation_angles_deg(workspace: &SfmWorkspace) -> Vec<f64> {
    workspace
        .landmarks()
        .as_slice()
        .iter()
        .filter_map(|landmark| {
            let track = workspace.tracks().get(landmark.source_track)?;
            let centers: Vec<_> = track
                .observations()
                .iter()
                .filter_map(|observation| {
                    workspace
                        .pose_store()
                        .get(observation.view.index())
                        .map(|record| *record.pose.center())
                })
                .collect();
            largest_angle_deg(landmark.point.coords, &centers)
        })
        .collect()
}

fn largest_angle_deg(point: Vector3<f64>, centers: &[nalgebra::Point3<f64>]) -> Option<f64> {
    let mut best = None;
    for i in 0..centers.len() {
        for j in (i + 1)..centers.len() {
            let ray_i = point - centers[i].coords;
            let ray_j = point - centers[j].coords;
            let denom = ray_i.norm() * ray_j.norm();
            if denom <= 1e-12 {
                continue;
            }
            let angle = (ray_i.dot(&ray_j) / denom)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            best = Some(best.map_or(angle, |current: f64| current.max(angle)));
        }
    }
    best
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

struct OracleFeatureExtractor {
    features: Vec<ExtractedFeatures>,
}

impl OracleFeatureExtractor {
    fn new(fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture) -> Self {
        let mut per_image_keypoints: Vec<Vec<Option<Keypoint>>> =
            vec![Vec::new(); fixture.image_meta.len()];
        for observation in &fixture.observations {
            let keypoints = &mut per_image_keypoints[observation.view.index()];
            if keypoints.len() <= observation.feature_id.index() {
                keypoints.resize(observation.feature_id.index() + 1, None);
            }
            keypoints[observation.feature_id.index()] = Some(
                Keypoint::try_new(observation.xy.x as f32, observation.xy.y as f32, 1.0, 0.0)
                    .unwrap(),
            );
        }

        let features = per_image_keypoints
            .into_iter()
            .map(|keypoints| {
                let keypoints: Vec<_> = keypoints
                    .into_iter()
                    .map(|keypoint| keypoint.unwrap())
                    .collect();
                let descriptors = DescriptorSet::try_f32(vec![0.0; keypoints.len()], 1).unwrap();
                ExtractedFeatures::try_new(keypoints, descriptors).unwrap()
            })
            .collect();
        Self { features }
    }
}

impl FeatureExtractor for OracleFeatureExtractor {
    fn extract(&self, ctx: &FeatureContext, _meta: &ImageMeta) -> Result<ExtractedFeatures> {
        Ok(self.features[ctx.view.index()].clone())
    }
}

struct OracleMatcher {
    matches_by_pair: BTreeMap<ViewPair, Vec<(FeatureId, FeatureId)>>,
}

impl OracleMatcher {
    fn new(fixture: &openmvg_sfm::testing::oracle::OracleSfmFixture) -> Self {
        let matches_by_pair = fixture
            .matches
            .iter()
            .map(|oracle_matches| (oracle_matches.pair, oracle_matches.matches.clone()))
            .collect();
        Self { matches_by_pair }
    }
}

impl PairMatcher for OracleMatcher {
    fn match_features(
        &self,
        ctx: &PairContext,
        _fi: &ExtractedFeatures,
        _fj: &ExtractedFeatures,
    ) -> Result<PairMatches> {
        let matches = self.matches_by_pair[&ctx.pair]
            .iter()
            .map(|(feat_i, feat_j)| FeatureMatch::try_new(*feat_i, *feat_j, Some(0.0)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        PairMatches::try_new(ctx.pair, matches)
    }
}
