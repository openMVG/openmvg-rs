// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-round consensus-guided robust estimation shared by geometry families.

use std::borrow::Cow;

use crate::SolverError;
use openmvg_core::determinism::AlgorithmRng;

use crate::robust::error::RobustError;
use crate::robust::result::{Diagnostics, EstimateStats, RobustEstimate};
use crate::solver_traits::MultiSolutionSolver;

use super::config::{GenRansacConfig, ResidualThreshold};
use super::refiner::GenRansacRefiner;

const EXPLORATION_FLOOR: f64 = 0.05;
const MARGINAL_SIGMA: f64 = 4.0;

/// Optional borrowed information used only for sampling.
///
/// `order` is best-first and may be a unique subset of the data. Without an
/// order, exploratory draws are genuinely uniform over every datum. `priors`
/// must contain calibrated inlier probabilities for every datum; constant
/// priors are treated as uninformative.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct RobustGuidance<'a> {
    pub(crate) order: Option<&'a [usize]>,
    pub(crate) priors: Option<&'a [f64]>,
}

impl<'a> RobustGuidance<'a> {
    /// Use a best-first ordering for guided sampling.
    #[must_use]
    pub fn ranked(order: &'a [usize]) -> Self {
        Self {
            order: Some(order),
            priors: None,
        }
    }

    /// Attach calibrated per-datum inlier probabilities.
    #[must_use]
    pub fn with_priors(mut self, priors: &'a [f64]) -> Self {
        self.priors = Some(priors);
        self
    }
}

struct RawDraw<M> {
    id: usize,
    model: M,
    prescore: f64,
}

struct Hypothesis<M> {
    id: usize,
    model: M,
    score: f64,
    inlier_count: usize,
    mask_offset: usize,
}

/// Internal GenRANSAC estimator core.
///
/// Its distinguishing mechanism is the population consensus marginal: after
/// each round, strong hypotheses vote for correspondences, and the next round
/// mixes that learned ranking with independent exploration.
pub(crate) struct GenRansacEstimator;

impl GenRansacEstimator {
    /// Estimate without external sampling guidance.
    #[cfg(test)]
    pub fn estimate<S, R>(
        data: &[S::Datum],
        threshold: ResidualThreshold,
        config: &GenRansacConfig,
        solver_config: &S::Config,
        refiner: &R,
    ) -> Result<RobustEstimate<S::Model>, RobustError>
    where
        S: MultiSolutionSolver,
        R: GenRansacRefiner<S>,
    {
        Self::estimate_guided::<S, R>(
            data,
            threshold,
            config,
            solver_config,
            RobustGuidance::default(),
            refiner,
        )
    }

    /// Estimate with a quality order and/or calibrated inlier priors.
    pub fn estimate_guided<S, R>(
        data: &[S::Datum],
        threshold: ResidualThreshold,
        config: &GenRansacConfig,
        solver_config: &S::Config,
        guidance: RobustGuidance<'_>,
        refiner: &R,
    ) -> Result<RobustEstimate<S::Model>, RobustError>
    where
        S: MultiSolutionSolver,
        R: GenRansacRefiner<S>,
    {
        Self::estimate_guided_impl::<S, R>(
            data,
            threshold,
            config,
            solver_config,
            guidance,
            refiner,
        )
    }

    fn estimate_guided_impl<S, R>(
        data: &[S::Datum],
        threshold: ResidualThreshold,
        config: &GenRansacConfig,
        solver_config: &S::Config,
        guidance: RobustGuidance<'_>,
        refiner: &R,
    ) -> Result<RobustEstimate<S::Model>, RobustError>
    where
        S: MultiSolutionSolver,
        R: GenRansacRefiner<S>,
    {
        config.validate()?;
        let n = data.len();
        let sample_size = S::min_sample_size();
        if sample_size == 0 {
            return Err(RobustError::InvalidConfig {
                reason: "solver minimum sample size must be greater than zero".to_owned(),
            });
        }
        if n < sample_size {
            return Err(RobustError::InsufficientData {
                required: sample_size,
                provided: n,
            });
        }

        let base_order = validate_order(guidance.order, n, sample_size)?;
        let has_quality_order = guidance.order.is_some();
        let initial_prior = validate_priors(guidance.priors, n)?;
        let has_prior = initial_prior.is_some();
        let mut marginals = initial_prior.unwrap_or_else(|| vec![0.0; n]);
        let mut posterior_ready = false;
        let mut informative_marginal_len = 0usize;

        let threshold = threshold.get();
        let threshold_squared = threshold * threshold;
        let score_gain = |squared_residual: f64| {
            let gain = threshold_squared - squared_residual;
            if gain.is_finite() && gain > 0.0 {
                gain
            } else {
                0.0
            }
        };

        let mut rng = AlgorithmRng::new(config.seed);
        let mut scratch_indices: Vec<usize> = (0..n).collect();
        let mut subsample = Vec::with_capacity(config.subsample_size.min(n));
        let mut sample = Vec::with_capacity(sample_size);
        let mut marginal_order = Vec::with_capacity(n);
        let reachable_draws = config.max_hypotheses.min(config.max_draws);
        let survivor_capacity = config.prescreen_survivors.min(reachable_draws);
        let mut raw_draws: Vec<RawDraw<S::Model>> = Vec::with_capacity(reachable_draws);
        let mut hypotheses: Vec<Hypothesis<S::Model>> = Vec::with_capacity(survivor_capacity);
        let words_per_mask = n.div_ceil(64);
        let mut masks = Vec::<u64>::new();
        let mut candidate_inliers = Vec::with_capacity(n);
        let mut current_inliers = Vec::with_capacity(n);
        let mut scored_inliers = Vec::with_capacity(n);
        let mut round_inliers = Vec::with_capacity(n);
        let mut hypothesis_order = Vec::with_capacity(survivor_capacity);

        let mut best_model: Option<S::Model> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_inliers = Vec::new();
        let mut stale_rounds = 0usize;
        let mut attempted_draws = 0usize;
        let mut valid_draws = 0usize;
        let mut evaluated_model_count = 0usize;
        let mut degenerate_sample_count = 0usize;
        let mut lo_calls = 0usize;
        let mut score_history = Vec::new();
        let mut residual_evaluation_count = 0usize;
        let mut final_sampling_pool = None;

        for round in 0..config.rounds {
            if attempted_draws >= config.max_draws {
                break;
            }
            let needed =
                confidence_draws_needed(best_inliers.len(), n, sample_size, config.confidence);
            if valid_draws >= needed {
                break;
            }
            let round_budget = if needed == usize::MAX {
                config.max_hypotheses
            } else {
                let minimum = sample_size.max(8).min(config.max_hypotheses);
                needed
                    .saturating_sub(valid_draws)
                    .clamp(minimum, config.max_hypotheses)
            }
            .min(config.max_draws - attempted_draws);

            marginal_order.clear();
            if round > 0 {
                marginal_order.extend(0..n);
                marginal_order.sort_unstable_by(|&a, &b| {
                    marginals[b]
                        .total_cmp(&marginals[a])
                        .then_with(|| a.cmp(&b))
                });
            }

            subsample.clear();
            if config.subsample_size > 0 && config.subsample_size < n {
                rng.shuffle(&mut scratch_indices);
                subsample.extend_from_slice(&scratch_indices[..config.subsample_size]);
            }

            raw_draws.clear();
            let mut pool = config.pool_init.max(sample_size).min(n);
            for draw_id in 0..round_budget {
                attempted_draws += 1;
                if pool < n {
                    pool = (pool + (pool / config.pool_growth_denom).max(1)).min(n);
                }

                let scheduled_exploration = if config.rounds <= 8 {
                    (0.6 - 0.3 * (round as f64 - 1.0)).max(0.0)
                } else {
                    let progress = round as f64 / (config.rounds - 1) as f64;
                    (0.6 * (1.0 - progress)).max(EXPLORATION_FLOOR)
                };
                let exploration = if has_prior {
                    0.40
                } else {
                    scheduled_exploration
                };
                let guidance_ready = !has_prior || posterior_ready;
                let guided = round > 0
                    && guidance_ready
                    && !marginal_order.is_empty()
                    && rng.unit_f64() >= exploration;

                let (order, effective_pool) = if guided {
                    let effective =
                        if config.clamp_guided_pool && informative_marginal_len >= sample_size {
                            pool.min(informative_marginal_len)
                        } else {
                            pool
                        };
                    (marginal_order.as_slice(), effective)
                } else if has_quality_order {
                    (base_order.as_ref(), pool.min(base_order.len()))
                } else {
                    // Important correction to the archived implementation: an
                    // unranked caller explores the whole population uniformly.
                    (base_order.as_ref(), base_order.len())
                };
                final_sampling_pool = Some(effective_pool);

                let drew_sample =
                    draw_unique(order, effective_pool, sample_size, &mut sample, &mut rng);
                if !drew_sample {
                    degenerate_sample_count += 1;
                    continue;
                }

                let mut best_root = None;
                let mut prescore = f64::NEG_INFINITY;
                let mut valid_root_count = 0usize;
                let solve_result =
                    S::for_each_solution_from_indices(&sample, data, solver_config, |model| {
                        if !S::is_valid(&model) {
                            return;
                        }
                        let score = if subsample.is_empty() {
                            data.iter()
                                .map(|datum| score_gain(S::squared_residual(&model, datum)))
                                .sum::<f64>()
                        } else {
                            subsample
                                .iter()
                                .map(|&i| score_gain(S::squared_residual(&model, &data[i])))
                                .sum::<f64>()
                        };
                        valid_root_count += 1;
                        if score > prescore {
                            best_root = Some(model);
                            prescore = score;
                        }
                    });
                match solve_result {
                    Ok(_) if valid_root_count > 0 => {}
                    Ok(_) => {
                        degenerate_sample_count += 1;
                        continue;
                    }
                    Err(
                        error @ (SolverError::InsufficientData { .. }
                        | SolverError::InvalidSampleSize { .. }
                        | SolverError::InvalidDataIndex { .. }
                        | SolverError::InvalidInput { .. }),
                    ) => return Err(error.into()),
                    Err(_) => {
                        degenerate_sample_count += 1;
                        continue;
                    }
                }

                // Root multiplicity is a solver detail, not an extra RANSAC
                // vote. Retain the first prescreen-best root from each draw;
                // strict comparison preserves deterministic solver order.
                if config.compute_diagnostics {
                    residual_evaluation_count += valid_root_count
                        * if subsample.is_empty() {
                            data.len()
                        } else {
                            subsample.len()
                        };
                }
                evaluated_model_count += valid_root_count;
                valid_draws += 1;
                let Some(model) = best_root else {
                    degenerate_sample_count += 1;
                    continue;
                };
                raw_draws.push(RawDraw {
                    id: draw_id,
                    model,
                    prescore,
                });
            }

            if raw_draws.is_empty() {
                continue;
            }
            raw_draws.sort_unstable_by(|a, b| {
                b.prescore
                    .total_cmp(&a.prescore)
                    .then_with(|| a.id.cmp(&b.id))
            });
            raw_draws.truncate(config.prescreen_survivors.min(raw_draws.len()));

            hypotheses.clear();
            masks.clear();
            for raw in raw_draws.drain(..) {
                let mask_offset = masks.len();
                masks.resize(mask_offset + words_per_mask, 0);
                let mut score = 0.0;
                let mut inlier_count = 0usize;
                for (index, datum) in data.iter().enumerate() {
                    let squared_residual = S::squared_residual(&raw.model, datum);
                    let gain = threshold_squared - squared_residual;
                    if gain.is_finite() && gain >= 0.0 {
                        score += gain;
                        inlier_count += 1;
                        masks[mask_offset + index / 64] |= 1u64 << (index % 64);
                    }
                }
                if config.compute_diagnostics {
                    residual_evaluation_count += data.len();
                }
                let id = hypotheses.len();
                hypotheses.push(Hypothesis {
                    id,
                    model: raw.model,
                    score,
                    inlier_count,
                    mask_offset,
                });
            }
            if hypotheses.is_empty() {
                continue;
            }

            hypothesis_order.clear();
            hypothesis_order.extend(0..hypotheses.len());
            hypothesis_order.sort_unstable_by(|&a, &b| {
                hypotheses[b]
                    .score
                    .total_cmp(&hypotheses[a].score)
                    .then_with(|| hypotheses[a].id.cmp(&hypotheses[b].id))
            });
            hypothesis_order.truncate(config.lo_candidates.min(hypothesis_order.len()));

            let mut round_model = None;
            let mut round_score = f64::NEG_INFINITY;
            round_inliers.clear();
            for &hypothesis_index in &hypothesis_order {
                let hypothesis = &hypotheses[hypothesis_index];
                mask_to_indices(
                    &masks[hypothesis.mask_offset..hypothesis.mask_offset + words_per_mask],
                    n,
                    &mut candidate_inliers,
                );
                let mut current = hypothesis.model.clone();
                let mut current_score = hypothesis.score;
                current_inliers.clone_from(&candidate_inliers);

                for _ in 0..config.lo_iterations {
                    if current_inliers.len() < sample_size {
                        break;
                    }
                    lo_calls += 1;
                    let Some(refined) =
                        refiner.refit(&current, &current_inliers, data, threshold)?
                    else {
                        break;
                    };
                    if !S::is_valid(&refined) {
                        break;
                    }
                    let score = score_and_inliers_into::<S>(
                        &refined,
                        data,
                        threshold_squared,
                        &mut scored_inliers,
                    );
                    if config.compute_diagnostics {
                        residual_evaluation_count += data.len();
                    }
                    if scored_inliers.len() >= sample_size && score > current_score {
                        current = refined;
                        current_score = score;
                        std::mem::swap(&mut current_inliers, &mut scored_inliers);
                    } else {
                        break;
                    }
                }

                if current_score > round_score {
                    round_model = Some(current);
                    round_score = current_score;
                    round_inliers.clone_from(&current_inliers);
                }
            }
            let Some(round_winner) = round_model else {
                continue;
            };
            if round_inliers.len() < sample_size {
                continue;
            }

            if !has_prior {
                population_marginal(
                    &hypotheses,
                    &masks,
                    words_per_mask,
                    &round_inliers,
                    &mut marginals,
                );
                informative_marginal_len = marginals.iter().filter(|&&v| v > 0.0).count();
            }

            if round_score > best_score {
                best_score = round_score;
                best_model = Some(round_winner);
                best_inliers.clone_from(&round_inliers);
                stale_rounds = 0;
                if has_prior {
                    guarded_bayes_update(
                        &mut marginals,
                        best_inliers.len() as f64 / n as f64,
                        |i| best_inliers.binary_search(&i).is_ok(),
                    );
                    informative_marginal_len = marginals
                        .iter()
                        .filter(|&&probability| probability > 0.5)
                        .count();
                    posterior_ready = true;
                }
            } else {
                stale_rounds += 1;
            }

            if config.compute_diagnostics {
                score_history.push(best_score);
            }
            if config.patience > 0
                && stale_rounds >= config.patience
                && best_inliers.len() as f64 / n as f64 >= config.patience_min_inlier_ratio
            {
                break;
            }
        }

        let Some(mut model) = best_model else {
            if valid_draws == 0 {
                return Err(RobustError::AllSamplesDegenerate {
                    total_samples: attempted_draws,
                });
            }
            return Err(RobustError::NoValidModel {
                iterations: attempted_draws,
                reason: "GenRANSAC produced no consensus with enough inliers".to_string(),
            });
        };

        // Fixed-point, score-gated final refinement.
        let mut final_score = best_score;
        for _ in 0..3 {
            if best_inliers.len() < sample_size {
                break;
            }
            lo_calls += 1;
            let Some(refined) = refiner.refit(&model, &best_inliers, data, threshold)? else {
                break;
            };
            if !S::is_valid(&refined) {
                break;
            }
            let score =
                score_and_inliers_into::<S>(&refined, data, threshold_squared, &mut scored_inliers);
            if config.compute_diagnostics {
                residual_evaluation_count += data.len();
            }
            if scored_inliers.len() >= sample_size && score > final_score {
                model = refined;
                final_score = score;
                std::mem::swap(&mut best_inliers, &mut scored_inliers);
            } else {
                break;
            }
        }

        let stats = EstimateStats {
            iteration_count: attempted_draws,
            score: final_score,
            inlier_ratio: best_inliers.len() as f64 / n as f64,
            evaluated_model_count,
            degenerate_sample_count,
        };
        let mut result = RobustEstimate::new(model, best_inliers, stats);
        if config.compute_diagnostics {
            let residuals = data
                .iter()
                .map(|datum| S::residual(&result.model, datum))
                .collect();
            result.diagnostics = Some(Diagnostics {
                residuals,
                score_history,
                final_pool_size: final_sampling_pool,
                refinement_iteration_count: lo_calls,
                final_adaptive_cap: confidence_draws_needed(
                    result.inliers.len(),
                    n,
                    sample_size,
                    config.confidence,
                )
                .min(config.max_scheduled_draws()),
                residual_evaluation_count,
                ..Diagnostics::default()
            });
        }

        Ok(result)
    }
}

fn validate_order<'a>(
    order: Option<&'a [usize]>,
    n: usize,
    sample_size: usize,
) -> Result<Cow<'a, [usize]>, RobustError> {
    let Some(order) = order else {
        return Ok(Cow::Owned((0..n).collect()));
    };
    if order.len() < sample_size {
        return Err(RobustError::InvalidConfig {
            reason: "GenRANSAC quality order is shorter than the minimal sample".to_string(),
        });
    }
    let mut seen = vec![false; n];
    for &index in order {
        if index >= n || seen[index] {
            return Err(RobustError::InvalidConfig {
                reason: "GenRANSAC quality order must contain unique, in-range indices".to_string(),
            });
        }
        seen[index] = true;
    }
    Ok(Cow::Borrowed(order))
}

fn validate_priors(priors: Option<&[f64]>, n: usize) -> Result<Option<Vec<f64>>, RobustError> {
    let Some(priors) = priors else {
        return Ok(None);
    };
    if priors.len() != n
        || priors
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(RobustError::InvalidConfig {
            reason: "GenRANSAC priors must be one finite probability per datum".to_string(),
        });
    }
    let minimum = priors.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = priors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum - minimum <= 1e-6 {
        return Ok(None);
    }
    Ok(Some(
        priors
            .iter()
            .map(|probability| probability.clamp(1e-12, 1.0 - 1e-12))
            .collect(),
    ))
}

fn draw_unique(
    order: &[usize],
    pool: usize,
    sample_size: usize,
    output: &mut Vec<usize>,
    rng: &mut AlgorithmRng,
) -> bool {
    if sample_size == 0 || order.len() < sample_size {
        return false;
    }
    let pool = pool.clamp(sample_size, order.len());
    output.clear();
    let mut attempts = 0usize;
    while output.len() < sample_size {
        let Some(position) = rng.index(pool) else {
            return false;
        };
        let index = order[position];
        if !output.contains(&index) {
            output.push(index);
        }
        attempts += 1;
        if attempts > 40 * sample_size {
            return false;
        }
    }
    true
}

fn score_and_inliers_into<S: MultiSolutionSolver>(
    model: &S::Model,
    data: &[S::Datum],
    threshold_squared: f64,
    inliers: &mut Vec<usize>,
) -> f64 {
    let mut score = 0.0;
    inliers.clear();
    for (index, datum) in data.iter().enumerate() {
        let squared_residual = S::squared_residual(model, datum);
        let gain = threshold_squared - squared_residual;
        if gain.is_finite() && gain >= 0.0 {
            score += gain;
            inliers.push(index);
        }
    }
    score
}

fn mask_to_indices(mask: &[u64], n: usize, output: &mut Vec<usize>) {
    output.clear();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let index = word_index * 64 + bit;
            if index < n {
                output.push(index);
            }
            remaining &= remaining - 1;
        }
    }
}

fn population_marginal<M>(
    hypotheses: &[Hypothesis<M>],
    masks: &[u64],
    words_per_mask: usize,
    fallback_inliers: &[usize],
    marginal: &mut [f64],
) {
    marginal.fill(0.0);
    let mut counts: Vec<usize> = hypotheses.iter().map(|hyp| hyp.inlier_count).collect();
    let middle = counts.len() / 2;
    let (_, median, _) = counts.select_nth_unstable(middle);
    let lambda = *median as f64;
    let floor = lambda + MARGINAL_SIGMA * lambda.max(1.0).sqrt();

    let mut score_low = f64::INFINITY;
    let mut score_high = f64::NEG_INFINITY;
    for hypothesis in hypotheses
        .iter()
        .filter(|hypothesis| hypothesis.inlier_count as f64 > floor)
    {
        score_low = score_low.min(hypothesis.score);
        score_high = score_high.max(hypothesis.score);
    }
    if !score_low.is_finite() {
        for &index in fallback_inliers {
            marginal[index] = 1.0;
        }
        return;
    }

    let span = (score_high - score_low).max(1e-12);
    for hypothesis in hypotheses
        .iter()
        .filter(|hypothesis| hypothesis.inlier_count as f64 > floor)
    {
        let weight = ((hypothesis.score - score_low) / span) + 1e-3;
        let mask = &masks[hypothesis.mask_offset..hypothesis.mask_offset + words_per_mask];
        for (word_index, &word) in mask.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                if index < marginal.len() {
                    marginal[index] += weight;
                }
                remaining &= remaining - 1;
            }
        }
    }
    let maximum = marginal.iter().copied().fold(0.0, f64::max);
    if maximum > 0.0 {
        for value in marginal.iter_mut() {
            *value /= maximum;
        }
    }
}

fn confidence_draws_needed(
    inlier_count: usize,
    n: usize,
    sample_size: usize,
    confidence: f64,
) -> usize {
    // This finite-population calculation is exact only for uniform draws.
    // Under ranked/marginal sampling it is a budget heuristic, not a coverage
    // guarantee, so the API does not call it an exact terminator.
    if confidence >= 1.0 || inlier_count < sample_size || n < sample_size {
        return usize::MAX;
    }
    let mut all_inlier_probability = 1.0;
    for offset in 0..sample_size {
        all_inlier_probability *= (inlier_count - offset) as f64 / (n - offset) as f64;
    }
    if all_inlier_probability <= 1e-12 {
        return usize::MAX;
    }
    if all_inlier_probability >= 1.0 - 1e-12 {
        return 1;
    }
    ((1.0 - confidence).ln() / (-all_inlier_probability).ln_1p()).ceil() as usize
}

fn guarded_bayes_update(
    posterior: &mut [f64],
    inlier_ratio: f64,
    mut observed_inlier: impl FnMut(usize) -> bool,
) {
    let gamma = if inlier_ratio < 0.7143 {
        0.62 * inlier_ratio + 0.5
    } else {
        0.2 * inlier_ratio + 0.8
    };
    for (index, probability) in posterior.iter_mut().enumerate() {
        let old_inlier = probability.clamp(1e-12, 1.0 - 1e-12);
        let old_outlier = 1.0 - old_inlier;
        let (next_inlier, next_outlier) = if observed_inlier(index) {
            (
                gamma * old_inlier + 0.2 * (1.0 - gamma) * old_outlier,
                0.8 * (1.0 - gamma) * old_outlier,
            )
        } else {
            ((1.0 - gamma) * old_inlier, gamma * old_outlier)
        };
        let normalizer = next_inlier + next_outlier;
        *probability = if normalizer.is_finite() && normalizer > 0.0 {
            (next_inlier / normalizer).clamp(0.0, 1.0)
        } else {
            0.5
        };
    }
}

#[cfg(test)]
mod tests;
