// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Robust homography estimation in pixel space.
//!
//! Its teaching docs are on [`estimate_homography`], where docs.rs shows them.

use std::sync::Arc;

use crate::correspondence::PixelCorrespondence;
use crate::homography::{DltSolver, Homography, HomographyConfig, HomographyMethod};
use crate::normalization::{NormalizedCorrespondence, NormalizedCorrespondences};
use crate::residuals::symmetric_transfer_error_h_with_inv;
use crate::{SolverError, SolverResult};
use nalgebra::Matrix3;
use openmvg_ba::numerics::manifold::{from_na, to_na};
use openmvg_ba::numerics::refine::{ConvergenceCriteria, Options};

use super::refinement;

use crate::robust::error::RobustError;
use crate::robust::result::RobustEstimate;
use crate::robust::threshold::PixelThreshold;
use crate::solver_traits::{SingleSolutionSolver, Solver};

use crate::robust::genransac::config::{GenRansacConfig, RobustOptions};
use crate::robust::genransac::estimator::{GenRansacEstimator, RobustGuidance};
use crate::robust::genransac::refiner::GenRansacRefiner;

#[derive(Debug)]
struct PreparedHomographyData {
    normalized: NormalizedCorrespondences,
    points1: Vec<[f64; 3]>,
    points2: Vec<[f64; 3]>,
}

#[derive(Debug, Clone)]
struct PreparedHomographyConfig(Arc<PreparedHomographyData>);

#[derive(Debug, Clone)]
struct ScoredHomography {
    homography: Homography,
    inverse: Matrix3<f64>,
}

impl ScoredHomography {
    fn new(homography: Homography) -> Option<Self> {
        let inverse = homography.matrix().try_inverse()?;
        inverse
            .iter()
            .all(|value| value.is_finite())
            .then_some(Self {
                homography,
                inverse,
            })
    }
}

impl PreparedHomographyConfig {
    fn new(data: &[PixelCorrespondence]) -> Self {
        let normalized = NormalizedCorrespondences::from_pixels(data);
        let points1 = data
            .iter()
            .map(|correspondence| [correspondence.point1.x, correspondence.point1.y, 1.0])
            .collect();
        let points2 = data
            .iter()
            .map(|correspondence| [correspondence.point2.x, correspondence.point2.y, 1.0])
            .collect();
        Self(Arc::new(PreparedHomographyData {
            normalized,
            points1,
            points2,
        }))
    }

    fn fit_and_denormalize(
        &self,
        normalized: &[NormalizedCorrespondence],
    ) -> SolverResult<Option<ScoredHomography>> {
        let solver_config = if normalized.len() == 4 {
            HomographyConfig::default().with_method(HomographyMethod::Aca)
        } else {
            HomographyConfig::default()
        };
        DltSolver::solve_single(normalized, &solver_config).map(|model| {
            model.and_then(|model| {
                let homography = Homography::new(
                    self.0
                        .normalized
                        .normalization()
                        .denormalize_homography(*model.matrix()),
                )
                .normalized();
                ScoredHomography::new(homography)
            })
        })
    }
}

#[derive(Debug, Clone)]
struct NonlinearHomographyRefiner {
    prepared: PreparedHomographyConfig,
}

impl NonlinearHomographyRefiner {
    fn new(prepared: PreparedHomographyConfig) -> Self {
        Self { prepared }
    }
}

impl GenRansacRefiner<PreparedHomographySolver> for NonlinearHomographyRefiner {
    fn refit(
        &self,
        seed: &ScoredHomography,
        inliers: &[usize],
        data: &[PixelCorrespondence],
        _threshold: f64,
    ) -> Result<Option<ScoredHomography>, RobustError> {
        if inliers.len() < PreparedHomographySolver::min_sample_size()
            || data.len() != self.prepared.0.points1.len()
        {
            return Ok(None);
        }
        let options = Options::l2(14).with_convergence(ConvergenceCriteria::new(1e-7, 1e-7, 1e-9));
        let report = match refinement::try_optimize_report_indexed(
            &from_na(seed.homography.matrix()),
            &self.prepared.0.points1,
            &self.prepared.0.points2,
            inliers,
            options,
        ) {
            Some(report) => report,
            None => return Ok(None),
        };
        if !report.is_numerically_usable() {
            return Ok(None);
        }
        let homography = Homography::new(to_na(&report.state)).normalized();
        Ok(ScoredHomography::new(homography))
    }
}

/// Internal adapter joining normalized fitting to exact pixel-space scoring.
#[derive(Debug, Clone, Copy)]
struct PreparedHomographySolver;

impl Solver for PreparedHomographySolver {
    type Datum = PixelCorrespondence;
    type Model = ScoredHomography;
    type Config = PreparedHomographyConfig;
    type Unit = openmvg_core::Pixels;

    fn min_sample_size() -> usize {
        4
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        Self::squared_residual(model, datum).sqrt()
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        symmetric_transfer_error_h_with_inv(model.homography.matrix(), &model.inverse, datum)
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.homography.is_valid() && model.inverse.iter().all(|value| value.is_finite())
    }
}

impl SingleSolutionSolver for PreparedHomographySolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        let prepared = PreparedHomographyConfig::new(data);
        prepared.fit_and_denormalize(prepared.0.normalized.as_slice())
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        if indices.len() < Self::min_sample_size() {
            return Err(SolverError::InsufficientData {
                required: Self::min_sample_size(),
                provided: indices.len(),
            });
        }
        if config.0.normalized.len() != data.len() {
            return Err(SolverError::DegenerateConfiguration {
                reason: format!(
                    "prepared homography data has {} entries, the input has {}",
                    config.0.normalized.len(),
                    data.len()
                ),
            });
        }

        let get = |position: usize| {
            let index = indices[position];
            config.0.normalized.as_slice().get(index).copied().ok_or(
                SolverError::InvalidDataIndex {
                    index,
                    data_len: data.len(),
                },
            )
        };
        if indices.len() == 4 {
            let sample = [get(0)?, get(1)?, get(2)?, get(3)?];
            return config.fit_and_denormalize(&sample);
        }

        let solver_config = HomographyConfig::default();
        DltSolver::solve_single_from_indices(
            indices,
            config.0.normalized.as_slice(),
            &solver_config,
        )
        .map(|model| {
            model.and_then(|model| {
                let homography = Homography::new(
                    config
                        .0
                        .normalized
                        .normalization()
                        .denormalize_homography(*model.matrix()),
                )
                .normalized();
                ScoredHomography::new(homography)
            })
        })
    }
}

/// Estimate a homography with symmetric transfer error measured in pixels.
///
/// # How it works
///
/// Correspondences are Hartley-normalized once for fitting. Every candidate is
/// immediately denormalized, so consensus scoring and the public threshold stay
/// in exact pixel units.
///
/// ```text
/// pixels -> Hartley normalize -> ACA hypotheses -> denormalize + score
///        -> nonlinear consensus refit -> pixel-space result
/// ```
pub fn estimate_homography(
    data: &[PixelCorrespondence],
    threshold: PixelThreshold,
    options: &RobustOptions,
) -> Result<RobustEstimate<Homography>, RobustError> {
    estimate_homography_guided(data, threshold, options, RobustGuidance::default())
}

/// Estimate a pixel-space homography with optional sampling guidance.
pub fn estimate_homography_guided(
    data: &[PixelCorrespondence],
    threshold: PixelThreshold,
    options: &RobustOptions,
    guidance: RobustGuidance<'_>,
) -> Result<RobustEstimate<Homography>, RobustError> {
    let solver_config = PreparedHomographyConfig::new(data);
    let refiner = NonlinearHomographyRefiner::new(solver_config.clone());
    let search = GenRansacConfig::from_options(GenRansacConfig::builder(), options);
    GenRansacEstimator::estimate_guided::<PreparedHomographySolver, _>(
        data,
        threshold.into(),
        &search,
        &solver_config,
        guidance,
        &refiner,
    )
    .map(|result| RobustEstimate {
        model: result.model.homography,
        inliers: result.inliers,
        stats: result.stats,
        diagnostics: result.diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residuals::symmetric_transfer_distance_homography;
    use crate::robust::error::RobustError;
    use nalgebra::{Matrix3, Point2, Vector3};

    fn project(homography: &Matrix3<f64>, point: Point2<f64>) -> Point2<f64> {
        let projected = homography * Vector3::new(point.x, point.y, 1.0);
        Point2::new(projected.x / projected.z, projected.y / projected.z)
    }

    fn synthetic_homography_problem() -> (Homography, Vec<PixelCorrespondence>, usize) {
        let expected = Homography::new(Matrix3::new(
            1.04, 0.025, 43.0, -0.018, 0.98, 27.0, 1.4e-4, -9.0e-5, 1.0,
        ));
        let mut data = (0..36)
            .map(|index| {
                let u = index as f64;
                let source = Point2::new(
                    80.0 + 67.0 * (index % 6) as f64 + 3.0 * (0.7 * u).sin(),
                    60.0 + 59.0 * (index / 6) as f64 + 2.0 * (1.1 * u).cos(),
                );
                let mut target = project(expected.matrix(), source);
                target.x += 0.025 * (1.31 * u).sin();
                target.y += 0.025 * (0.83 * u).cos();
                PixelCorrespondence::new(source, target)
            })
            .collect::<Vec<_>>();
        let inlier_count = data.len();

        for index in 0..8 {
            let source = data[index + 3].point1;
            let mut target = data[(index + 17) % inlier_count].point2;
            target.x += 90.0;
            target.y -= 70.0;
            data.push(PixelCorrespondence::new(source, target));
        }
        (expected, data, inlier_count)
    }

    fn test_options() -> RobustOptions {
        RobustOptions::default().with_max_draws(320).with_seed(41)
    }

    fn assert_recovery(
        expected: &Homography,
        data: &[PixelCorrespondence],
        inlier_count: usize,
        result: &RobustEstimate<Homography>,
        threshold: PixelThreshold,
    ) {
        assert!(result.model.is_valid());
        assert!(
            result
                .inliers
                .iter()
                .filter(|&&index| index < inlier_count)
                .count()
                >= inlier_count - 1
        );
        let maximum_true_residual = data[..inlier_count]
            .iter()
            .map(|datum| symmetric_transfer_distance_homography(result.model.matrix(), datum))
            .fold(0.0_f64, f64::max);
        assert!(maximum_true_residual < threshold.value());
        let expected = expected.matrix() / expected.matrix().norm();
        let recovered = result.model.matrix() / result.model.matrix().norm();
        let projective_error = (expected - recovered)
            .norm()
            .min((expected + recovered).norm());
        assert!(
            projective_error < 1.0e-3,
            "projective error: {projective_error}"
        );
    }

    /// Robust Homography estimation: the typed pixel facade recovers a projective transform with
    /// outliers; normalization, symmetric scoring, consensus refit, and denormalization remain one
    /// policy.
    #[test]
    fn ordinary_facade_recovers_pixel_homography_end_to_end() {
        let (expected, data, inlier_count) = synthetic_homography_problem();
        let threshold = PixelThreshold::try_new(1.0).unwrap();
        let result = estimate_homography(&data, threshold, &test_options())
            .expect("ordinary homography policy should recover the dominant plane");

        assert_recovery(&expected, &data, inlier_count, &result, threshold);
    }

    #[test]
    fn guided_facade_is_deterministic_end_to_end() {
        let (expected, data, inlier_count) = synthetic_homography_problem();
        let threshold = PixelThreshold::try_new(1.0).unwrap();
        let order = (0..data.len()).collect::<Vec<_>>();
        let solve = || {
            estimate_homography_guided(
                &data,
                threshold,
                &test_options(),
                RobustGuidance::ranked(&order),
            )
            .expect("guided homography policy should recover the dominant plane")
        };
        let first = solve();
        let second = solve();

        assert_eq!(first.inliers, second.inliers);
        assert_eq!(first.model, second.model);
        assert_recovery(&expected, &data, inlier_count, &first, threshold);
    }

    #[test]
    fn guided_facade_without_guidance_matches_the_ordinary_facade() {
        let (_, data, _) = synthetic_homography_problem();
        let threshold = PixelThreshold::try_new(1.0).unwrap();
        let ordinary = estimate_homography(&data, threshold, &test_options())
            .expect("ordinary homography policy should recover the dominant plane");
        let guided = estimate_homography_guided(
            &data,
            threshold,
            &test_options(),
            RobustGuidance::default(),
        )
        .expect("unguided homography policy should recover the dominant plane");

        assert_eq!(guided.inliers, ordinary.inliers);
        assert_eq!(guided.model, ordinary.model);
    }

    #[test]
    fn guided_facade_survives_an_outlier_first_ranking_with_priors() {
        let (expected, data, inlier_count) = synthetic_homography_problem();
        let threshold = PixelThreshold::try_new(1.0).unwrap();
        let order = (inlier_count..data.len())
            .chain(0..inlier_count)
            .collect::<Vec<_>>();
        let priors = (0..data.len())
            .map(|index| if index < inlier_count { 0.9 } else { 0.1 })
            .collect::<Vec<_>>();
        let result = estimate_homography_guided(
            &data,
            threshold,
            &test_options(),
            RobustGuidance::ranked(&order).with_priors(&priors),
        )
        .expect("guided homography policy should recover the dominant plane");

        assert_recovery(&expected, &data, inlier_count, &result, threshold);
        assert!(result.inliers.iter().all(|&index| index < inlier_count));
    }

    #[test]
    fn guided_facade_rejects_malformed_priors() {
        let (_, data, _) = synthetic_homography_problem();
        let priors = vec![0.5; data.len() - 1];
        let result = estimate_homography_guided(
            &data,
            PixelThreshold::try_new(1.0).unwrap(),
            &test_options(),
            RobustGuidance::default().with_priors(&priors),
        );

        assert!(matches!(result, Err(RobustError::InvalidConfig { .. })));
    }

    #[test]
    fn facade_preserves_the_four_point_contract() {
        let result =
            estimate_homography(&[], PixelThreshold::try_new(1.0).unwrap(), &test_options());
        assert!(matches!(
            result,
            Err(RobustError::InsufficientData {
                required: 4,
                provided: 0
            })
        ));
    }
}
