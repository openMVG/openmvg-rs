// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Robust essential matrix estimation with GenRANSAC.
//!
//! Its teaching docs are on [`estimate_essential`], where docs.rs shows them.

use crate::SolverResult;
use crate::correspondence::BearingCorrespondence;
use crate::essential::{
    EightPointSolver as AngularEssentialSolver, EssentialMatrix, FivePointSolver,
    PinholeEssentialSolver,
};
use crate::residuals::sampson_error_essential_pinhole_squared;

use crate::robust::error::RobustError;
use crate::robust::result::RobustEstimate;
use crate::solver_traits::{MultiSolutionSolver, Solver};

use crate::robust::threshold::{AngularThreshold, NormalizedImageThreshold};

use crate::robust::genransac::config::{GenRansacConfig, RobustOptions};
use crate::robust::genransac::estimator::{GenRansacEstimator, RobustGuidance};
use crate::robust::genransac::refiner::SolverRefiner;

/// Perspective scoring policy over the five-point kernel.
#[derive(Debug, Clone, Copy, Default)]
struct PinholeScoredFivePointSolver;

impl Solver for PinholeScoredFivePointSolver {
    type Datum = BearingCorrespondence;
    type Model = EssentialMatrix;
    type Config = ();
    type Unit = openmvg_core::NormalizedImage;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        Self::squared_residual(model, datum).sqrt()
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_essential_pinhole_squared(model.matrix(), datum)
    }

    fn min_sample_size() -> usize {
        FivePointSolver::min_sample_size()
    }

    fn is_valid(model: &Self::Model) -> bool {
        FivePointSolver::is_valid(model)
    }
}

impl MultiSolutionSolver for PinholeScoredFivePointSolver {
    fn solve(data: &[Self::Datum], config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        <FivePointSolver as MultiSolutionSolver>::solve(data, config)
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
        emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        <FivePointSolver as MultiSolutionSolver>::for_each_solution_from_indices(
            indices, data, config, emit,
        )
    }
}

/// Estimate an essential matrix for a calibrated perspective-camera pair.
///
/// The threshold is in normalized image units: a pixel tolerance divided by the focal length.
/// Set it to a few times the pixel noise; with 0.5 px of noise and an 800 px focal length,
/// `2.0 / 800.0` keeps nearly every good match. Decompose the result with
/// [`EssentialMatrix::decompose_with_cheirality`] to get the relative pose; the `relative_pose`
/// example of this crate runs both steps and checks them against a known motion.
///
/// The threshold type keeps the units apart: an angular threshold does not compile here.
///
/// ```compile_fail
/// use openmvg_geometry::essential::estimate_essential;
/// use openmvg_geometry::robust::{RobustOptions, threshold::AngularThreshold};
/// let options = RobustOptions::default();
/// let radians = AngularThreshold::try_new(0.01).unwrap();
/// let _ = estimate_essential(&[], radians, &options);
/// ```
///
/// # How it works
///
/// RANSAC draws five-match samples, solves each for its essential matrices, usually up to ten,
/// and scores every candidate on all matches with the Sampson error, the first-order distance of
/// a match to its epipolar constraint, in normalized image units.
/// [`estimate_essential_angular`] scores an angular error instead, for spherical and other
/// wide-angle cameras. The best model is then refit with the eight-point solver on its inliers,
/// and the refit is kept only when it raises the consensus score; with noisy matches the best
/// five-point model often comes back unchanged. In a reconstruction, bundle adjustment refines
/// the pose further.
///
/// ```text
/// bearings -> E5 hypotheses -> Sampson or angular score -> E8 refit, kept if better -> result
/// ```
pub fn estimate_essential(
    data: &[BearingCorrespondence],
    threshold: NormalizedImageThreshold,
    options: &RobustOptions,
) -> Result<RobustEstimate<EssentialMatrix>, RobustError> {
    estimate_essential_guided(data, threshold, options, RobustGuidance::default())
}

/// Estimate a perspective-camera essential matrix with sampling guidance.
pub fn estimate_essential_guided(
    data: &[BearingCorrespondence],
    threshold: NormalizedImageThreshold,
    options: &RobustOptions,
    guidance: RobustGuidance<'_>,
) -> Result<RobustEstimate<EssentialMatrix>, RobustError> {
    let search = GenRansacConfig::from_options(GenRansacConfig::essential(), options);
    let refiner = SolverRefiner::<PinholeEssentialSolver>::new(());
    GenRansacEstimator::estimate_guided::<PinholeScoredFivePointSolver, _>(
        data,
        threshold.into(),
        &search,
        &(),
        guidance,
        &refiner,
    )
}

/// Estimate an essential matrix using camera-frame-invariant angular scoring.
///
/// Use this for spherical cameras or whenever valid rays may approach the
/// camera equator. The threshold is a first-order angular threshold in radians.
pub fn estimate_essential_angular(
    data: &[BearingCorrespondence],
    threshold: AngularThreshold,
    options: &RobustOptions,
) -> Result<RobustEstimate<EssentialMatrix>, RobustError> {
    estimate_essential_angular_guided(data, threshold, options, RobustGuidance::default())
}

/// Estimate an angular essential matrix with sampling guidance.
pub fn estimate_essential_angular_guided(
    data: &[BearingCorrespondence],
    threshold: AngularThreshold,
    options: &RobustOptions,
    guidance: RobustGuidance<'_>,
) -> Result<RobustEstimate<EssentialMatrix>, RobustError> {
    let search = GenRansacConfig::from_options(GenRansacConfig::essential(), options);
    let refiner = SolverRefiner::<AngularEssentialSolver>::new(());
    GenRansacEstimator::estimate_guided::<FivePointSolver, _>(
        data,
        threshold.into(),
        &search,
        &(),
        guidance,
        &refiner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robust::error::RobustError;
    use nalgebra::{Rotation3, Vector3};

    fn perspective_correspondences() -> (Vec<BearingCorrespondence>, usize) {
        let rotation = Rotation3::from_euler_angles(0.04, -0.11, 0.07);
        let translation = Vector3::new(0.42, -0.13, 0.09);
        let mut data = (0..24)
            .map(|i| {
                let u = i as f64;
                let point = Vector3::new(
                    1.7 * (0.71 * u).sin(),
                    1.4 * (1.13 * u + 0.2).cos(),
                    4.0 + 0.11 * u + 0.8 * (0.37 * u).sin().abs(),
                );
                BearingCorrespondence::try_new(
                    point.normalize(),
                    (rotation * point + translation).normalize(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let inlier_count = data.len();
        for i in 0..6 {
            data.push(
                BearingCorrespondence::try_new(
                    *data[i].bearing1(),
                    *data[(i + 9) % inlier_count].bearing2(),
                )
                .unwrap(),
            );
        }
        (data, inlier_count)
    }

    fn equatorial_correspondences() -> (Vec<BearingCorrespondence>, usize, Vec<usize>) {
        let rotation = Rotation3::from_euler_angles(-0.08, 0.14, 0.05);
        let translation = Vector3::new(-0.24, 0.37, 0.16);
        let mut equatorial = Vec::new();
        let mut data = (0..24)
            .map(|i| {
                let u = i as f64;
                let direction = if i < 5 {
                    equatorial.push(i);
                    Vector3::new((0.83 * u).cos(), (0.83 * u).sin(), 0.0)
                } else {
                    Vector3::new(
                        (0.61 * u).cos(),
                        (1.07 * u + 0.3).sin(),
                        0.85 * (0.47 * u - 0.2).cos(),
                    )
                    .normalize()
                };
                let point = direction * (4.0 + 0.17 * u);
                BearingCorrespondence::try_new(direction, rotation * point + translation).unwrap()
            })
            .collect::<Vec<_>>();
        let inlier_count = data.len();
        for i in 0..6 {
            data.push(
                BearingCorrespondence::try_new(
                    *data[i + 5].bearing1(),
                    *data[(i + 13) % inlier_count].bearing2(),
                )
                .unwrap(),
            );
        }
        (data, inlier_count, equatorial)
    }

    fn test_options() -> RobustOptions {
        RobustOptions::default().with_max_draws(144).with_seed(19)
    }

    #[test]
    fn both_camera_policies_preserve_the_five_point_contract() {
        let options = RobustOptions::default();
        let pinhole = estimate_essential(
            &[],
            NormalizedImageThreshold::try_new(0.01).unwrap(),
            &options,
        );
        let angular =
            estimate_essential_angular(&[], AngularThreshold::try_new(0.01).unwrap(), &options);
        for result in [pinhole, angular] {
            assert!(matches!(
                result,
                Err(RobustError::InsufficientData {
                    required: 5,
                    provided: 0
                })
            ));
        }
    }

    /// Robust calibrated Essential estimation: the typed normalized-image facade recovers a noisy
    /// calibrated consensus; five-point fitting and eight-point consensus refitting preserve
    /// threshold units.
    #[test]
    fn perspective_facade_recovers_a_noisy_consensus_end_to_end() {
        let (data, inlier_count) = perspective_correspondences();
        let order = (0..data.len()).collect::<Vec<_>>();
        let threshold = NormalizedImageThreshold::try_new(1e-4).unwrap();
        let result = estimate_essential_guided(
            &data,
            threshold,
            &test_options(),
            RobustGuidance::ranked(&order),
        )
        .expect("the perspective facade should recover the calibrated pair");

        assert!(result.model.is_valid());
        assert!(
            result
                .inliers
                .iter()
                .filter(|&&index| index < inlier_count)
                .count()
                >= inlier_count - 1
        );
        let mut true_residuals = data[..inlier_count]
            .iter()
            .map(|datum| PinholeEssentialSolver::squared_residual(&result.model, datum))
            .collect::<Vec<_>>();
        true_residuals.sort_by(f64::total_cmp);
        assert!(true_residuals[true_residuals.len() / 2] < threshold.squared());
    }

    /// Robust angular Essential estimation: the angular facade accepts valid equatorial rays;
    /// angular scoring and refinement remain camera-frame invariant.
    #[test]
    fn angular_facade_keeps_equatorial_rays_end_to_end() {
        let (data, inlier_count, equatorial) = equatorial_correspondences();
        let order = (0..data.len()).collect::<Vec<_>>();
        let threshold = AngularThreshold::try_new(1e-4).unwrap();
        let result = estimate_essential_angular_guided(
            &data,
            threshold,
            &test_options(),
            RobustGuidance::ranked(&order),
        )
        .expect("the angular facade should support rays on the camera equator");

        assert!(result.model.is_valid());
        assert!(
            equatorial
                .iter()
                .all(|index| result.inliers.binary_search(index).is_ok())
        );
        let max_equatorial_residual = equatorial
            .iter()
            .map(|&index| AngularEssentialSolver::squared_residual(&result.model, &data[index]))
            .fold(0.0_f64, f64::max);
        assert!(max_equatorial_residual.is_finite());
        assert!(max_equatorial_residual < threshold.squared());
        assert!(
            result
                .inliers
                .iter()
                .filter(|&&index| index < inlier_count)
                .count()
                >= inlier_count - 1
        );
    }
}
