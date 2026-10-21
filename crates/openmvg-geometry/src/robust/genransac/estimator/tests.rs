// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::SolverResult;

use crate::solver_traits::Solver;

#[derive(Debug, Clone)]
struct LineSolver;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Solver for LineSolver {
    type Datum = Point;
    type Model = (f64, f64);
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn min_sample_size() -> usize {
        2
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        (datum.y - (model.0 * datum.x + model.1)).abs()
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.0.is_finite() && model.1.is_finite()
    }
}

impl MultiSolutionSolver for LineSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        if data.len() != 2 {
            return Ok(Vec::new());
        }
        let dx = data[1].x - data[0].x;
        if dx.abs() < 1e-12 {
            return Ok(Vec::new());
        }
        let slope = (data[1].y - data[0].y) / dx;
        Ok(vec![(slope, data[0].y - slope * data[0].x)])
    }
}

#[derive(Debug, Clone)]
struct TiedRootSolver;

impl Solver for TiedRootSolver {
    type Datum = Point;
    type Model = (f64, f64);
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn min_sample_size() -> usize {
        2
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        (datum.y - (model.0 * datum.x + model.1)).abs()
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.0.is_finite() && model.1.is_finite()
    }
}

impl MultiSolutionSolver for TiedRootSolver {
    fn solve(_data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        // Both horizontal lines have exactly the same residual on y = 1.
        Ok(vec![(0.0, 0.0), (0.0, 2.0)])
    }
}

#[derive(Debug, Clone)]
struct ContractBrokenSolver;

impl Solver for ContractBrokenSolver {
    type Datum = Point;
    type Model = (f64, f64);
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn min_sample_size() -> usize {
        2
    }

    fn residual(_model: &Self::Model, _datum: &Self::Datum) -> f64 {
        0.0
    }
}

impl MultiSolutionSolver for ContractBrokenSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        Err(SolverError::InvalidSampleSize {
            expected: 3,
            provided: data.len(),
        })
    }
}

#[derive(Debug, Clone)]
struct ZeroSampleSolver;

impl Solver for ZeroSampleSolver {
    type Datum = Point;
    type Model = (f64, f64);
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn min_sample_size() -> usize {
        0
    }

    fn residual(_model: &Self::Model, _datum: &Self::Datum) -> f64 {
        0.0
    }
}

impl MultiSolutionSolver for ZeroSampleSolver {
    fn solve(_data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        Ok(vec![(0.0, 0.0)])
    }
}

#[derive(Debug, Clone)]
struct AlwaysDegenerateSolver;

impl Solver for AlwaysDegenerateSolver {
    type Datum = Point;
    type Model = (f64, f64);
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn min_sample_size() -> usize {
        2
    }

    fn residual(_model: &Self::Model, _datum: &Self::Datum) -> f64 {
        0.0
    }
}

impl MultiSolutionSolver for AlwaysDegenerateSolver {
    fn solve(_data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
struct WeakConsensusSolver;

impl Solver for WeakConsensusSolver {
    type Datum = Point;
    type Model = (f64, f64);
    type Config = ();
    type Unit = openmvg_core::Unitless;

    fn min_sample_size() -> usize {
        2
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        (datum.y - model.1).abs()
    }
}

impl MultiSolutionSolver for WeakConsensusSolver {
    fn solve(_data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        Ok(vec![(0.0, 0.0)])
    }
}

struct CollapsingRefiner;

impl super::super::refiner::GenRansacRefiner<WeakConsensusSolver> for CollapsingRefiner {
    fn refit(
        &self,
        _seed: &(f64, f64),
        _inliers: &[usize],
        _data: &[Point],
        _threshold: f64,
    ) -> Result<Option<(f64, f64)>, RobustError> {
        Ok(Some((0.0, 10.0)))
    }
}

fn line_data() -> Vec<Point> {
    let mut data: Vec<Point> = (0..30)
        .map(|i| Point {
            x: i as f64,
            y: 2.0 * i as f64 + 1.0,
        })
        .collect();
    data.extend((0..20).map(|i| Point {
        x: i as f64 + 0.5,
        y: 80.0 - 3.0 * i as f64,
    }));
    data
}

#[test]
fn deterministic_and_returns_original_indices() {
    let data = line_data();
    let config = GenRansacConfig::builder()
        .rounds(5)
        .max_hypotheses(32)
        .prescreen_survivors(8)
        .seed(7)
        .build();
    let first = GenRansacEstimator::estimate::<LineSolver, _>(
        &data,
        ResidualThreshold::new(0.01).unwrap(),
        &config,
        &(),
        &super::super::refiner::NoRefiner,
    )
    .unwrap();
    let second = GenRansacEstimator::estimate::<LineSolver, _>(
        &data,
        ResidualThreshold::new(0.01).unwrap(),
        &config,
        &(),
        &super::super::refiner::NoRefiner,
    )
    .unwrap();
    assert_eq!(first.model, second.model);
    assert_eq!(first.inliers, second.inliers);
    assert_eq!(first.inliers, (0..30).collect::<Vec<_>>());
}

#[test]
fn hard_draw_budget_is_enforced_across_rounds() {
    let data = (0..8)
        .map(|x| Point {
            x: x as f64,
            y: 1.0,
        })
        .collect::<Vec<_>>();
    let config = GenRansacConfig::builder()
        .rounds(5)
        .max_hypotheses(32)
        .max_draws(3)
        .confidence(1.0)
        .build();
    let result = GenRansacEstimator::estimate::<TiedRootSolver, _>(
        &data,
        ResidualThreshold::new(1.5).unwrap(),
        &config,
        &(),
        &super::super::refiner::NoRefiner,
    )
    .unwrap();

    assert_eq!(result.stats.iteration_count, 3);
    assert_eq!(result.stats.evaluated_model_count, 6);
}

#[test]
fn malformed_guidance_is_rejected() {
    let data = line_data();
    let config = GenRansacConfig::builder().build();
    let result = GenRansacEstimator::estimate_guided::<LineSolver, _>(
        &data,
        ResidualThreshold::new(0.1).unwrap(),
        &config,
        &(),
        RobustGuidance::ranked(&[0, 0]),
        &super::super::refiner::NoRefiner,
    );
    assert!(matches!(result, Err(RobustError::InvalidConfig { .. })));
}

#[test]
fn structural_solver_errors_are_not_misreported_as_degeneracy() {
    let data = line_data();
    let config = GenRansacConfig::builder()
        .rounds(1)
        .max_hypotheses(1)
        .build();
    let result = GenRansacEstimator::estimate::<ContractBrokenSolver, _>(
        &data,
        ResidualThreshold::new(0.1).unwrap(),
        &config,
        &(),
        &super::super::refiner::NoRefiner,
    );

    assert!(matches!(result, Err(RobustError::SolverFailed { .. })));
}

#[test]
fn zero_sample_solver_is_rejected_before_sampling() {
    let result = GenRansacEstimator::estimate::<ZeroSampleSolver, _>(
        &line_data(),
        ResidualThreshold::new(0.1).unwrap(),
        &GenRansacConfig::builder().build(),
        &(),
        &super::super::refiner::NoRefiner,
    );

    assert!(matches!(result, Err(RobustError::InvalidConfig { .. })));
}

#[test]
fn all_degenerate_draws_have_a_precise_error() {
    let result = GenRansacEstimator::estimate::<AlwaysDegenerateSolver, _>(
        &line_data(),
        ResidualThreshold::new(0.1).unwrap(),
        &GenRansacConfig::builder()
            .rounds(1)
            .max_hypotheses(4)
            .build(),
        &(),
        &super::super::refiner::NoRefiner,
    );

    assert!(matches!(
        result,
        Err(RobustError::AllSamplesDegenerate { total_samples: 4 })
    ));
}

#[test]
fn refinement_cannot_replace_a_valid_consensus_with_too_few_inliers() {
    let data = [
        Point { x: 0.0, y: 0.9 },
        Point { x: 1.0, y: 0.9 },
        Point { x: 2.0, y: 0.9 },
        Point { x: 3.0, y: 10.0 },
    ];
    let result = GenRansacEstimator::estimate::<WeakConsensusSolver, _>(
        &data,
        ResidualThreshold::new(1.0).unwrap(),
        &GenRansacConfig::builder()
            .rounds(1)
            .max_hypotheses(1)
            .confidence(1.0)
            .build(),
        &(),
        &CollapsingRefiner,
    )
    .unwrap();

    assert_eq!(result.model, (0.0, 0.0));
    assert_eq!(result.inliers, vec![0, 1, 2]);
}

#[test]
fn constant_prior_is_neutral() {
    let priors = vec![0.5; 4];
    assert!(validate_priors(Some(&priors), 4).unwrap().is_none());
}

#[test]
fn best_root_ties_keep_solver_order() {
    let data = (0..8)
        .map(|x| Point {
            x: x as f64,
            y: 1.0,
        })
        .collect::<Vec<_>>();
    let config = GenRansacConfig::builder()
        .rounds(1)
        .max_hypotheses(8)
        .prescreen_survivors(8)
        .seed(11)
        .build();
    let result = GenRansacEstimator::estimate::<TiedRootSolver, _>(
        &data,
        ResidualThreshold::new(1.5).unwrap(),
        &config,
        &(),
        &super::super::refiner::NoRefiner,
    )
    .unwrap();

    assert_eq!(result.model, (0.0, 0.0));
    assert_eq!(result.stats.evaluated_model_count, 16);
}

#[test]
fn hostile_capacity_options_are_clamped_to_reachable_draws() {
    let data = (0..8)
        .map(|x| Point {
            x: x as f64,
            y: 1.0,
        })
        .collect::<Vec<_>>();
    let config = GenRansacConfig::builder()
        .rounds(1)
        .max_hypotheses(usize::MAX)
        .max_draws(1)
        .prescreen_survivors(usize::MAX)
        .confidence(1.0)
        .build();

    let result = GenRansacEstimator::estimate::<TiedRootSolver, _>(
        &data,
        ResidualThreshold::new(1.5).unwrap(),
        &config,
        &(),
        &super::super::refiner::NoRefiner,
    )
    .unwrap();
    assert_eq!(result.stats.iteration_count, 1);
}

#[test]
fn exact_threshold_residual_is_an_inlier() {
    let data = [Point { x: 0.0, y: 1.0 }];
    let mut inliers = Vec::new();
    let score = score_and_inliers_into::<LineSolver>(&(0.0, 0.0), &data, 1.0, &mut inliers);

    assert_eq!(score, 0.0);
    assert_eq!(inliers, vec![0]);
}

#[test]
fn confidence_formula_uses_finite_population() {
    let exact = confidence_draws_needed(25, 50, 4, 0.99);
    let approximate = ((1.0f64 - 0.99).ln() / (1.0f64 - 0.5f64.powi(4)).ln()).ceil() as usize;
    assert!(exact > approximate);
}
