// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Verification stage: estimate a two-view model for each selected pair with a
//! [`GeometricVerifier`].

use crate::backends::{GeometricVerifier, VerificationContext};
use crate::determinism::GEOMETRIC_VERIFICATION_SEED_DOMAIN;
use crate::ops::resolve_pairs;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::two_view::{RejectedPairVerification, VerificationBatchReport, VerificationVerdict};
use crate::workspace::selector::PairSelector;
use crate::{Result, SfmError, SfmWorkspace};

/// Geometrically verify selected pairs with the configured robust estimator.
///
/// Each pair is verified independently against read-only workspace state, so under the
/// `parallel` feature the pairs are fanned out across the caller's current Rayon pool
/// (verification is the third front-end stage after detection and matching). The workspace mutations —
/// storing the accepted model, or clearing + deactivating a rejected pair — touch
/// per-pair slots only, so they are applied sequentially afterwards in pair order,
/// which is exactly equivalent to the original interleaved loop (the verify result of
/// one pair never depends on another pair's activation state) and keeps the output
/// deterministic.
pub fn verify_pairs(
    workspace: &mut SfmWorkspace,
    verifier: &dyn GeometricVerifier,
    selector: PairSelector,
) -> Result<VerificationBatchReport> {
    verify_pairs_with_progress(workspace, verifier, selector, &())
}

/// [`verify_pairs`], reporting [`SfmProgressStage::Verification`] in pairs.
pub fn verify_pairs_with_progress(
    workspace: &mut SfmWorkspace,
    verifier: &dyn GeometricVerifier,
    selector: PairSelector,
    progress: &dyn SfmProgressSink,
) -> Result<VerificationBatchReport> {
    if workspace.is_matching_data_released() {
        return Err(SfmError::MatchingDataReleased);
    }
    let pair_ids = resolve_pairs(workspace, selector)?;
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::Verification, pair_ids.len());

    let verify_one =
        |ws: &SfmWorkspace, pair_id| -> Result<(usize, VerificationVerdict)> {
            cancellation_point(progress)?;
            let pair = ws.pair_graph().pair(pair_id).ok_or(SfmError::InvalidId {
                kind: crate::PairId::kind(),
                index: pair_id.index(),
            })?;
            let view_i = ws.view(pair.first()).ok_or(SfmError::InvalidId {
                kind: crate::ViewId::kind(),
                index: pair.first().index(),
            })?;
            let view_j = ws.view(pair.second()).ok_or(SfmError::InvalidId {
                kind: crate::ViewId::kind(),
                index: pair.second().index(),
            })?;
            let camera_i = ws
                .camera_model(view_i.intrinsic_id)
                .ok_or(SfmError::InvalidId {
                    kind: crate::IntrinsicId::kind(),
                    index: view_i.intrinsic_id.index(),
                })?;
            let camera_j = ws
                .camera_model(view_j.intrinsic_id)
                .ok_or(SfmError::InvalidId {
                    kind: crate::IntrinsicId::kind(),
                    index: view_j.intrinsic_id.index(),
                })?;
            let features_i = ws.feature_store().get(pair.first().index()).ok_or(
                SfmError::MissingPrerequisite {
                    prerequisite: "features",
                    method: "extract_features",
                },
            )?;
            let features_j = ws.feature_store().get(pair.second().index()).ok_or(
                SfmError::MissingPrerequisite {
                    prerequisite: "features",
                    method: "extract_features",
                },
            )?;
            let matches =
                ws.match_store()
                    .get(pair_id.index())
                    .ok_or(SfmError::MissingPrerequisite {
                        prerequisite: "matches",
                        method: "match_pairs",
                    })?;
            matches.validate_for_pair(pair, features_i.len(), features_j.len())?;
            let outcome = verifier.verify_pair(
                &VerificationContext {
                    pair,
                    view_i: pair.first(),
                    view_j: pair.second(),
                    seed: ws.derive_seed(
                        GEOMETRIC_VERIFICATION_SEED_DOMAIN,
                        &[pair.first().value(), pair.second().value()],
                    ),
                },
                features_i,
                features_j,
                matches,
                camera_i,
                camera_j,
            )?;
            validate_verification_outcome(pair, matches, &outcome)?;
            progress.advance(SfmProgressStage::Verification, 1);
            Ok((pair_id.index(), outcome))
        };

    #[cfg(feature = "parallel")]
    let results: Vec<(usize, VerificationVerdict)> = {
        use rayon::prelude::*;
        let ws: &SfmWorkspace = workspace;
        // Use the caller's current Rayon pool. Building a private pool here can
        // oversubscribe applications that already parallelize over images or
        // scenes. Indexed collection preserves pair order deterministically.
        let outcomes: Vec<Result<(usize, VerificationVerdict)>> = pair_ids
            .par_iter()
            .map(|&pair_id| verify_one(ws, pair_id))
            .collect();
        outcomes.into_iter().collect::<Result<Vec<_>>>()?
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(usize, VerificationVerdict)> = {
        let ws: &SfmWorkspace = workspace;
        pair_ids
            .iter()
            .map(|&pair_id| verify_one(ws, pair_id))
            .collect::<Result<Vec<_>>>()?
    };

    let mut report = VerificationBatchReport::default();
    for (index, outcome) in results {
        match outcome {
            VerificationVerdict::Accepted {
                model,
                report: pair_report,
            } => {
                workspace.geometric_models.set(index, model)?;
                report.accepted.push(pair_report);
            }
            VerificationVerdict::Rejected {
                reason,
                report: pair_report,
            } => {
                workspace.geometric_models.clear(index)?;
                workspace
                    .pair_graph_mut()
                    .deactivate_pair_and(crate::PairId::new(index))?;
                report.rejected.push(RejectedPairVerification {
                    reason,
                    report: pair_report,
                });
            }
        }
    }
    progress.finish(SfmProgressStage::Verification);
    Ok(report)
}

fn validate_verification_outcome(
    pair: crate::ViewPair,
    matches: &crate::two_view::PairMatches,
    outcome: &VerificationVerdict,
) -> Result<()> {
    let report = match outcome {
        VerificationVerdict::Accepted { model, report } => {
            model.validate_for_matches(matches)?;
            if report.inlier_count != model.evidence().inlier_count {
                return Err(SfmError::InvalidInput(format!(
                    "verification report for {pair} contains {} inliers, but its model evidence contains {}",
                    report.inlier_count,
                    model.evidence().inlier_count
                )));
            }
            report
        }
        VerificationVerdict::Rejected { report, .. } => report,
    };
    if report.pair != pair {
        return Err(SfmError::InvalidInput(format!(
            "verifier returned report for {} while processing {pair}",
            report.pair
        )));
    }
    if report.input_match_count != matches.len() {
        return Err(SfmError::InvalidInput(format!(
            "verification report for {pair} declares {} input matches, but {} were supplied",
            report.input_match_count,
            matches.len()
        )));
    }
    if report.inlier_count > report.input_match_count {
        return Err(SfmError::InvalidInput(format!(
            "verification report for {pair} declares more inliers than input matches"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageMeta;
    use crate::progress::testing::{ProgressRecord, RecordingProgress};
    use crate::two_view::{
        InlierMask, ModelEvidence, PairMatches, PairVerificationReport, TerminationReason,
        VerificationRejectReason,
    };
    use crate::workspace::quality::ErrorStats;
    use crate::{Camera, FeatureId, IntrinsicId};
    use nalgebra::{Matrix3, Rotation3};
    use openmvg_features::model::{DescriptorSet, ExtractedFeatures, FeatureMatch, Keypoint};

    const STAGE: SfmProgressStage = SfmProgressStage::Verification;

    struct RejectingVerifier;

    struct WrongReportPairVerifier;

    struct InvalidRotationVerifier;

    impl GeometricVerifier for RejectingVerifier {
        fn verify_pair(
            &self,
            ctx: &VerificationContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
            matches: &PairMatches,
            _camera_i: &openmvg_core::camera::Camera,
            _camera_j: &openmvg_core::camera::Camera,
        ) -> Result<VerificationVerdict> {
            Ok(VerificationVerdict::Rejected {
                reason: VerificationRejectReason::InsufficientInliers {
                    required: 2,
                    actual: 0,
                },
                report: PairVerificationReport {
                    pair: ctx.pair,
                    input_match_count: matches.len(),
                    inlier_count: 0,
                    message: None,
                },
            })
        }
    }

    impl GeometricVerifier for WrongReportPairVerifier {
        fn verify_pair(
            &self,
            ctx: &VerificationContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
            matches: &PairMatches,
            _camera_i: &openmvg_core::camera::Camera,
            _camera_j: &openmvg_core::camera::Camera,
        ) -> Result<VerificationVerdict> {
            Ok(VerificationVerdict::Rejected {
                reason: VerificationRejectReason::InsufficientInliers {
                    required: 2,
                    actual: 1,
                },
                report: PairVerificationReport {
                    pair: crate::ViewPair::new(ctx.view_i, crate::ViewId::new(2)),
                    input_match_count: matches.len(),
                    inlier_count: 1,
                    message: None,
                },
            })
        }
    }

    impl GeometricVerifier for InvalidRotationVerifier {
        fn verify_pair(
            &self,
            ctx: &VerificationContext,
            _fi: &ExtractedFeatures,
            _fj: &ExtractedFeatures,
            matches: &PairMatches,
            _camera_i: &openmvg_core::camera::Camera,
            _camera_j: &openmvg_core::camera::Camera,
        ) -> Result<VerificationVerdict> {
            let evidence = ModelEvidence::try_new(
                InlierMask::all_inliers(matches.len()),
                ErrorStats::from_values(&vec![0.0; matches.len()], &[1.0]),
                openmvg_core::PixelThreshold::ONE_PIXEL.into(),
                "malformed-test",
                1,
                TerminationReason::Converged,
                Vec::new(),
            )?;
            Ok(VerificationVerdict::Accepted {
                model: crate::two_view::GeometricModel::RotationOnly {
                    rotation: Rotation3::from_matrix_unchecked(Matrix3::zeros()),
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

    fn workspace() -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(vec![
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap();
        for index in 0..2 {
            workspace
                .set_features(
                    index,
                    ExtractedFeatures::try_new(
                        vec![Keypoint::try_new(index as f32, 0.0, 1.0, 0.0).unwrap()],
                        DescriptorSet::try_u8(vec![index as u8], 1).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        workspace
            .matches
            .set(
                0,
                PairMatches::try_new(
                    crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1)),
                    vec![
                        FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(0), None).unwrap(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        workspace
    }

    #[test]
    fn verification_reports_every_pair_and_empty_reruns() {
        let mut workspace = workspace();
        let progress = RecordingProgress::default();
        let report = verify_pairs_with_progress(
            &mut workspace,
            &RejectingVerifier,
            PairSelector::All,
            &progress,
        )
        .unwrap();
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(
            progress.records(),
            [
                ProgressRecord::Begin(STAGE, 1),
                ProgressRecord::Advanced(STAGE, 1),
                ProgressRecord::Finish(STAGE),
            ]
        );

        // The rejected pair was deactivated, so no active pair remains to verify.
        let progress = RecordingProgress::default();
        verify_pairs_with_progress(
            &mut workspace,
            &RejectingVerifier,
            PairSelector::Active,
            &progress,
        )
        .unwrap();
        assert_eq!(
            progress.records(),
            [
                ProgressRecord::Begin(STAGE, 0),
                ProgressRecord::Finish(STAGE)
            ]
        );
    }

    #[test]
    fn verifier_contract_violations_do_not_mutate_workspace() {
        for verifier in [
            &WrongReportPairVerifier as &dyn GeometricVerifier,
            &InvalidRotationVerifier as &dyn GeometricVerifier,
        ] {
            let mut workspace = workspace();
            let progress = RecordingProgress::default();
            let error =
                verify_pairs_with_progress(&mut workspace, verifier, PairSelector::All, &progress)
                    .unwrap_err();

            assert!(matches!(error, SfmError::InvalidInput(_)));
            assert_eq!(workspace.geometric_model_store().populated_count(), 0);
            assert!(workspace.pair_graph().is_active(crate::PairId::new(0)));
            assert_eq!(progress.records(), [ProgressRecord::Begin(STAGE, 1)]);
        }
    }
}
