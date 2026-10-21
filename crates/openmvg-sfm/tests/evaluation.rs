// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use openmvg_sfm::testing::determinism::stable_json_hash;
use openmvg_sfm::unstable::evaluation::*;
use openmvg_sfm::unstable::reports::ErrorStats;

#[test]
fn evaluation_counts_matched_alignment_and_skipped_pairs() {
    let fixture = common::oracle_fixture::fixture();
    let mut reconstruction = fixture.reconstruction_gt.clone();
    reconstruction.poses.pop();

    let report = evaluate_reconstruction_against_gt(
        &reconstruction,
        &fixture.scene,
        &EvaluationConfig::default(),
    )
    .unwrap();

    assert_eq!(report.matched_gt_pose_count, reconstruction.poses.len());
    assert_eq!(report.used_for_alignment_count, reconstruction.poses.len());
    assert!(report.relative_pair_skip_counts.missing_pose > 0);
}

#[test]
fn evaluation_empty_stats_are_json_and_hash_safe() {
    let stats = ErrorStats::from_values(&[], &[1.0, 2.0]);
    assert_eq!(stats.count, 0);
    assert!(stats.min.is_finite());
    assert!(stats.max.is_finite());
    assert!(stats.mean.is_finite());
    assert_eq!(stats.threshold_pass_rates.len(), 2);

    let hash = stable_json_hash(&stats).unwrap();
    assert_eq!(hash.len(), "sha256:".len() + 64);
    assert!(hash.starts_with("sha256:"));
}

#[test]
fn oracle_gt_against_itself_has_zero_ate_are() {
    let fixture = common::oracle_fixture::fixture();
    assert_eq!(fixture.image_meta.len(), 5);
    assert_eq!(fixture.poses_gt.len(), 5);
    assert!(!fixture.observations.is_empty());
    assert!(!fixture.pairs.is_empty());
    assert!(!fixture.matches.is_empty());
    assert!(!fixture.tracks.is_empty());

    let report = evaluate_reconstruction_against_gt(
        &fixture.reconstruction_gt,
        &fixture.scene,
        &EvaluationConfig::default(),
    )
    .unwrap();

    assert_eq!(report.pose_count, fixture.poses_gt.len());
    assert_eq!(report.matched_gt_pose_count, fixture.poses_gt.len());
    assert!(report.ate.rmse <= 1e-10);
    assert!(report.absolute_rotation_are.max <= 1e-10);
    assert!(report.relative_rotation_are.unwrap().max <= 1e-10);
    let relative_translation_are = report.relative_translation_are.unwrap();
    assert!(
        relative_translation_are.max <= 1e-5,
        "relative translation max={}",
        relative_translation_are.max
    );
    assert_eq!(report.relative_pair_skip_counts.missing_pose, 0);
    assert_eq!(report.relative_pair_skip_counts.degenerate_baseline, 0);
    assert_eq!(report.relative_pair_skip_counts.non_finite, 0);

    let reconstruction = GroundTruthReconstruction {
        poses: fixture.poses_gt,
        points: fixture.points_gt,
    };
    assert_eq!(
        stable_json_hash(&reconstruction).unwrap(),
        stable_json_hash(&fixture.reconstruction_gt).unwrap()
    );
}
