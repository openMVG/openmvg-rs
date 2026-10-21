// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic robust-estimation orchestration shared by geometry families.

use crate::SolverError;

use super::config::{
    ClassicConfig, ClassicMethod, ImagePlaneUnit, SamplingStrategy, invalid_config,
};
#[cfg(test)]
use super::sampler::{ExhaustiveSampler, checked_combination_count};
use super::sampler::{ProsacSampler, RandomSampler, Sampler};
use super::scoring::{BinaryScorer, MsacScorer, Scorer};
use super::termination::{ExactTerminator, FixedTerminator, Terminator};
use crate::correspondence::PixelCorrespondence;
use crate::robust::error::RobustError;
use crate::robust::result::{Diagnostics, EstimateStats, RobustEstimate};
use crate::solver_traits::MultiSolutionSolver;

/// Classic robust estimator with validated configuration and deterministic defaults.
pub(crate) struct ClassicEstimator;

struct LoopOutcome<M> {
    model: M,
    inliers: Vec<usize>,
    stats: EstimateStats,
    score_history: Vec<f64>,
}

impl ClassicEstimator {
    /// Estimate a model from potentially contaminated data.
    ///
    /// Configuration is validated before the solver or sampler is invoked.
    /// The returned model always satisfies the configured minimum consensus.
    pub(crate) fn estimate<S, Q>(
        config: &ClassicConfig<Q>,
        data: &[S::Datum],
        solver_config: &S::Config,
    ) -> Result<RobustEstimate<S::Model>, RobustError>
    where
        S: MultiSolutionSolver,
        Q: Copy,
    {
        config.validate()?;

        let sample_size = S::min_sample_size();
        if sample_size == 0 {
            return invalid_config("solver minimum sample size must be greater than zero");
        }

        let required_inliers = config.min_inliers().unwrap_or(sample_size);
        if required_inliers < sample_size {
            return invalid_config(format!(
                "min_inliers ({required_inliers}) cannot be smaller than the solver sample size ({sample_size})"
            ));
        }
        if data.len() < required_inliers {
            return Err(RobustError::InsufficientData {
                required: required_inliers,
                provided: data.len(),
            });
        }

        let LoopOutcome {
            model,
            inliers,
            mut stats,
            score_history,
        } = match config.method() {
            ClassicMethod::Ransac(params) => {
                let scorer = BinaryScorer::from_validated_threshold(params.threshold.value());
                Self::run_randomized::<S, _, Q>(
                    data,
                    solver_config,
                    config,
                    &scorer,
                    sample_size,
                    params.confidence,
                    params.max_iterations,
                )?
            }
            ClassicMethod::Msac(params) => {
                let scorer = MsacScorer::from_validated_threshold(params.threshold.value());
                Self::run_randomized::<S, _, Q>(
                    data,
                    solver_config,
                    config,
                    &scorer,
                    sample_size,
                    params.confidence,
                    params.max_iterations,
                )?
            }
            #[cfg(test)]
            ClassicMethod::MaxConsensus(params) => {
                let total_combinations = checked_combination_count(data.len(), sample_size)
                    .ok_or_else(|| RobustError::InvalidConfig {
                        reason: "exhaustive combination count exceeds usize".to_owned(),
                    })?;
                if total_combinations > params.combination_limit {
                    return invalid_config(format!(
                        "exhaustive search needs {total_combinations} combinations, above the configured limit {}",
                        params.combination_limit
                    ));
                }
                let scorer = BinaryScorer::from_validated_threshold(params.threshold.value());
                let terminator = FixedTerminator::new(total_combinations);
                let mut sampler = ExhaustiveSampler::new(data.len(), sample_size);
                Self::run_loop::<S, _, _, _>(
                    data,
                    solver_config,
                    &mut sampler,
                    &scorer,
                    &terminator,
                    sample_size,
                    config.diagnostics(),
                )?
            }
        };

        if inliers.len() < required_inliers {
            return Err(RobustError::InsufficientConsensus {
                required: required_inliers,
                found: inliers.len(),
            });
        }

        stats.inlier_ratio = inliers.len() as f64 / data.len() as f64;

        let diagnostics = if config.diagnostics() {
            Some(Diagnostics {
                residuals: data
                    .iter()
                    .map(|datum| S::residual(&model, datum))
                    .collect(),
                score_history,
                ..Diagnostics::default()
            })
        } else {
            None
        };

        let mut result = RobustEstimate::new(model, inliers, stats);
        if let Some(diagnostics) = diagnostics {
            result = result.with_diagnostics(diagnostics);
        }
        Ok(result)
    }

    fn run_randomized<S, Sc, Q>(
        data: &[S::Datum],
        solver_config: &S::Config,
        config: &ClassicConfig<Q>,
        scorer: &Sc,
        sample_size: usize,
        confidence: f64,
        max_iterations: usize,
    ) -> Result<LoopOutcome<S::Model>, RobustError>
    where
        S: MultiSolutionSolver,
        Sc: Scorer,
        Q: Copy,
    {
        match config.sampling() {
            SamplingStrategy::Uniform => {
                let mut sampler = RandomSampler::new(config.seed());
                let terminator = ExactTerminator::new(confidence, max_iterations, data.len());
                Self::run_loop::<S, _, _, _>(
                    data,
                    solver_config,
                    &mut sampler,
                    scorer,
                    &terminator,
                    sample_size,
                    config.diagnostics(),
                )
            }
            SamplingStrategy::Prosac => {
                // The uniform-RANSAC confidence bound is not valid while
                // PROSAC draws from ranked prefixes. Use the complete configured
                // draw budget and make that same budget define T'_N.
                let mut sampler = ProsacSampler::with_draw_budget(
                    data.len(),
                    sample_size,
                    config.seed(),
                    max_iterations,
                );
                let terminator = FixedTerminator::new(max_iterations);
                Self::run_loop::<S, _, _, _>(
                    data,
                    solver_config,
                    &mut sampler,
                    scorer,
                    &terminator,
                    sample_size,
                    config.diagnostics(),
                )
            }
        }
    }

    fn run_loop<S, Samp, Sc, Term>(
        data: &[S::Datum],
        solver_config: &S::Config,
        sampler: &mut Samp,
        scorer: &Sc,
        terminator: &Term,
        sample_size: usize,
        track_scores: bool,
    ) -> Result<LoopOutcome<S::Model>, RobustError>
    where
        S: MultiSolutionSolver,
        Samp: Sampler,
        Sc: Scorer,
        Term: Terminator,
    {
        let mut best_model = None;
        let mut best_inliers = Vec::with_capacity(data.len());
        let mut sample_best_inliers = Vec::with_capacity(data.len());
        let mut candidate_inliers = Vec::with_capacity(data.len());
        let mut sample_indices = Vec::with_capacity(sample_size);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_inlier_ratio = 0.0;
        let mut score_history = if track_scores {
            Vec::with_capacity(1_024)
        } else {
            Vec::new()
        };
        let mut stats = EstimateStats::new();
        let mut iteration = 0;

        while !terminator.should_stop(iteration, best_inlier_ratio, sample_size) {
            sampler.sample_into(data.len(), sample_size, &mut sample_indices);
            if sample_indices.len() != sample_size {
                break;
            }

            let mut sample_best_model = None;
            let mut sample_best_score = best_score;
            let mut sample_models_evaluated = 0;
            sample_best_inliers.clear();

            let solve_result = S::for_each_solution_from_indices(
                &sample_indices,
                data,
                solver_config,
                |candidate| {
                    if !S::is_valid(&candidate) {
                        return;
                    }
                    sample_models_evaluated += 1;
                    let Some(score) = scorer.score_and_collect_inliers_with_cutoff(
                        &candidate,
                        data,
                        &mut candidate_inliers,
                        S::squared_residual,
                        sample_best_score,
                    ) else {
                        return;
                    };
                    sample_best_score = score;
                    sample_best_model = Some(candidate);
                    std::mem::swap(&mut sample_best_inliers, &mut candidate_inliers);
                },
            );

            match solve_result {
                Ok(0) => stats.degenerate_sample_count += 1,
                Ok(_) if sample_models_evaluated == 0 => stats.degenerate_sample_count += 1,
                Ok(_) => {
                    stats.evaluated_model_count += sample_models_evaluated;
                    if let Some(candidate) = sample_best_model {
                        best_score = sample_best_score;
                        best_inlier_ratio = sample_best_inliers.len() as f64 / data.len() as f64;
                        best_model = Some(candidate);
                        std::mem::swap(&mut best_inliers, &mut sample_best_inliers);
                    }
                }
                Err(
                    error @ (SolverError::InsufficientData { .. }
                    | SolverError::InvalidSampleSize { .. }
                    | SolverError::InvalidDataIndex { .. }
                    | SolverError::InvalidInput { .. }),
                ) => return Err(error.into()),
                Err(_) => stats.degenerate_sample_count += 1,
            }

            if track_scores {
                score_history.push(best_score);
            }
            iteration += 1;
        }

        stats.iteration_count = iteration;
        let Some(model) = best_model else {
            if iteration > 0 && stats.degenerate_sample_count == iteration {
                return Err(RobustError::AllSamplesDegenerate {
                    total_samples: iteration,
                });
            }
            return Err(RobustError::NoValidModel {
                iterations: iteration,
                reason: "the solver emitted no valid model".to_owned(),
            });
        };

        stats.score = best_score;
        stats.inlier_ratio = best_inlier_ratio;
        Ok(LoopOutcome {
            model,
            inliers: best_inliers,
            stats,
            score_history,
        })
    }
}

impl<Q: Copy> ClassicConfig<Q> {
    /// Estimate a model of any [`MultiSolutionSolver`] with the classic robust loop.
    ///
    /// This is the extension point for solvers defined outside this crate. The
    /// threshold unit `Q` must be the solver's [`Solver::Unit`], so a pixel
    /// threshold cannot reach an angular residual. Configuration is validated
    /// before the solver is invoked.
    ///
    /// [`Solver::Unit`]: crate::robust::Solver::Unit
    pub fn estimate<S>(
        &self,
        data: &[S::Datum],
        solver_config: &S::Config,
    ) -> Result<RobustEstimate<S::Model>, RobustError>
    where
        S: MultiSolutionSolver<Unit = Q>,
    {
        ClassicEstimator::estimate::<S, Q>(self, data, solver_config)
    }
}

// The unnormalized kernels measure residuals in whatever frame their input is in, so these
// image-plane entry points bypass the unit check of `estimate`.
impl<Q: ImagePlaneUnit> ClassicConfig<Q> {
    /// Estimate a homography with the classic robust loop.
    pub fn estimate_homography(
        &self,
        data: &[PixelCorrespondence],
        solver_config: &crate::homography::HomographyConfig,
    ) -> Result<RobustEstimate<crate::homography::Homography>, RobustError> {
        ClassicEstimator::estimate::<crate::homography::UnnormalizedHomographySolver, Q>(
            self,
            data,
            solver_config,
        )
    }

    /// Estimate a fundamental matrix with the classic robust loop.
    pub fn estimate_fundamental(
        &self,
        data: &[PixelCorrespondence],
    ) -> Result<RobustEstimate<crate::fundamental::FundamentalMatrix>, RobustError> {
        ClassicEstimator::estimate::<crate::fundamental::UnnormalizedFundamentalSolver, Q>(
            self,
            data,
            &(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SolverResult;
    use openmvg_core::ScalarThreshold;

    use crate::solver_traits::Solver;

    fn threshold(value: f64) -> ScalarThreshold {
        ScalarThreshold::try_new(value).unwrap()
    }

    #[derive(Clone, Debug)]
    struct TestModel(f64);

    #[derive(Clone)]
    struct TestSolver;

    impl Solver for TestSolver {
        type Datum = (f64, f64);
        type Model = TestModel;
        type Config = ();
        type Unit = openmvg_core::Unitless;

        fn residual(model: &Self::Model, &(x, y): &Self::Datum) -> f64 {
            (y - model.0 * x).abs()
        }

        fn min_sample_size() -> usize {
            2
        }
    }

    impl MultiSolutionSolver for TestSolver {
        fn solve(data: &[Self::Datum], _config: &()) -> SolverResult<Vec<Self::Model>> {
            let (x1, y1) = data[0];
            let (x2, y2) = data[1];
            if (x2 - x1).abs() < 1e-12 {
                return Ok(Vec::new());
            }
            Ok(vec![TestModel((y2 - y1) / (x2 - x1))])
        }
    }

    fn line_data() -> Vec<(f64, f64)> {
        vec![
            (1.0, 2.0),
            (2.0, 4.0),
            (3.0, 6.0),
            (4.0, 8.0),
            (5.0, 100.0),
            (6.0, 12.0),
            (7.0, 14.0),
            (8.0, 200.0),
        ]
    }

    #[test]
    fn ransac_finds_consensus() {
        let result = ClassicConfig::ransac(threshold(1.0))
            .with_confidence(0.99)
            .with_seed(42)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap();

        assert!((result.model.0 - 2.0).abs() < 0.1);
        assert_eq!(result.inliers.len(), 6);
    }

    #[test]
    fn msac_finds_consensus() {
        let result = ClassicConfig::msac(threshold(1.0))
            .with_confidence(0.99)
            .with_seed(42)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap();

        assert!((result.model.0 - 2.0).abs() < 0.1);
        assert_eq!(result.inliers.len(), 6);
        assert!(result.stats.score.is_finite());
    }

    #[test]
    fn prosac_is_deterministic_on_best_first_data() {
        let config = ClassicConfig::ransac(threshold(1.0))
            .prosac()
            .with_seed(17)
            .build();
        let data = line_data();
        let first = config.estimate::<TestSolver>(&data, &()).unwrap();
        let second = config.estimate::<TestSolver>(&data, &()).unwrap();

        assert_eq!(first.inliers, second.inliers);
        assert_eq!(first.stats.iteration_count, second.stats.iteration_count);
        assert!((first.model.0 - second.model.0).abs() < f64::EPSILON);
        assert!((first.model.0 - 2.0).abs() < 0.1);
        assert_eq!(first.inliers.len(), 6);
    }

    #[test]
    fn estimate_validates_configuration_before_sampling() {
        let error = ClassicConfig::ransac(threshold(1.0))
            .with_confidence(f64::NAN)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap_err();
        assert!(matches!(error, RobustError::InvalidConfig { .. }));
    }

    #[test]
    fn minimum_consensus_is_enforced() {
        let error = ClassicConfig::ransac(threshold(1.0))
            .with_min_inliers(7)
            .with_seed(42)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap_err();
        assert!(matches!(
            error,
            RobustError::InsufficientConsensus {
                required: 7,
                found: 6
            }
        ));
    }

    #[test]
    fn minimum_consensus_cannot_be_below_sample_size() {
        let error = ClassicConfig::ransac(threshold(1.0))
            .with_min_inliers(1)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap_err();
        assert!(matches!(error, RobustError::InvalidConfig { .. }));
    }

    #[test]
    fn impossible_consensus_reports_insufficient_data() {
        let error = ClassicConfig::ransac(threshold(1.0))
            .with_min_inliers(9)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap_err();
        assert!(matches!(
            error,
            RobustError::InsufficientData {
                required: 9,
                provided: 8
            }
        ));
    }

    #[test]
    fn all_degenerate_samples_have_a_precise_error() {
        let data = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)];
        let error = ClassicConfig::max_consensus(threshold(1.0))
            .build()
            .estimate::<TestSolver>(&data, &())
            .unwrap_err();
        assert!(matches!(
            error,
            RobustError::AllSamplesDegenerate { total_samples: 3 }
        ));
    }

    #[test]
    fn exhaustive_search_never_silently_truncates() {
        let error = ClassicConfig::max_consensus(threshold(1.0))
            .combination_limit(27)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap_err();
        assert!(matches!(error, RobustError::InvalidConfig { .. }));
        assert!(error.to_string().contains("needs 28 combinations"));
    }

    #[test]
    fn exhaustive_search_evaluates_the_final_combination() {
        let data = [(1.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let result = ClassicConfig::max_consensus(threshold(0.1))
            .build()
            .estimate::<TestSolver>(&data, &())
            .unwrap();

        assert!((result.model.0 - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.inliers, vec![1, 2]);
        assert_eq!(result.stats.iteration_count, 3);
    }

    #[test]
    fn diagnostics_describe_the_final_model_without_wall_clock_state() {
        let result = ClassicConfig::ransac(threshold(1.0))
            .with_diagnostics(true)
            .with_seed(42)
            .build()
            .estimate::<TestSolver>(&line_data(), &())
            .unwrap();
        let diagnostics = result.diagnostics.unwrap();
        assert_eq!(diagnostics.residuals.len(), 8);
        assert_eq!(
            diagnostics.score_history.len(),
            result.stats.iteration_count
        );
    }
}
