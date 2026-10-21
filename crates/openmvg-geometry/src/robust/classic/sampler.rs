// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling strategies for classic robust estimation.
//!
//! Provides different ways to select minimal samples from data:
//!
//! - [`RandomSampler`] - Standard uniform random sampling (RANSAC)
//! - [`ProsacSampler`] - Progressive sampling exploiting quality ordering (PROSAC)
//! - `ExhaustiveSampler` - All n-choose-k combinations (test-only MaxConsensus oracle)
//!
//! # PROSAC (Progressive Sample Consensus)
//!
//! PROSAC exploits prior information about point quality (e.g., matching score,
//! descriptor distance) to sample from better points first. This typically leads
//! to finding a good model in far fewer iterations than standard RANSAC.
//!
//! Reference: Chum, O., & Matas, J. (2005). Matching with PROSAC - Progressive
//! Sample Consensus. CVPR 2005.

use openmvg_core::{RunSeed, determinism::AlgorithmRng};

const DEFAULT_SAMPLER_SEED: RunSeed = RunSeed::from_u64(0x4f50_454e_4d56_4702);

/// Trait for sampling strategies.
///
/// Each randomized estimator run creates a sampler from its configured seed. A fixed seed
/// chooses the initial RNG state; continuing to sample from the same instance
/// advances that state.
pub trait Sampler: Clone + Send + Sync {
    /// Sample `sample_size` indices from `data_len` total points.
    fn sample(&mut self, data_len: usize, sample_size: usize) -> Vec<usize>;

    /// Fill a reusable buffer with one sample.
    ///
    /// Implementations should override this method when they can avoid the
    /// allocation performed by the default compatibility implementation.
    fn sample_into(&mut self, data_len: usize, sample_size: usize, output: &mut Vec<usize>) {
        *output = self.sample(data_len, sample_size);
    }
}

/// Random uniform sampler using the shared Floyd sampling protocol.
///
/// The standard RANSAC sampling strategy: pick random minimal samples.
#[derive(Clone)]
pub struct RandomSampler {
    rng: AlgorithmRng,
}

impl RandomSampler {
    /// Create a sampler. `None` selects the stable library seed.
    pub fn new(seed: Option<RunSeed>) -> Self {
        let rng = AlgorithmRng::new(seed.unwrap_or(DEFAULT_SAMPLER_SEED));
        Self { rng }
    }

    /// Create a sampler with a specific seed for reproducibility.
    #[cfg(test)]
    fn with_seed(seed: u64) -> Self {
        Self::new(Some(RunSeed::from_u64(seed)))
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self, data_len: usize, sample_size: usize) -> Vec<usize> {
        let mut output = Vec::with_capacity(sample_size.min(data_len));
        self.sample_into(data_len, sample_size, &mut output);
        output
    }

    fn sample_into(&mut self, data_len: usize, sample_size: usize, output: &mut Vec<usize>) {
        output.clear();
        if sample_size >= data_len {
            output.extend(0..data_len);
            return;
        }

        output.resize(sample_size, 0);
        if !draw_k_unique(&mut self.rng, data_len, sample_size, output) {
            output.clear();
        }
    }
}

/// Exhaustive sampler for the test-only MaxConsensus oracle.
///
/// Iterates through all n-choose-k combinations.
#[cfg(test)]
#[derive(Clone, Debug)]
pub struct ExhaustiveSampler {
    current: Vec<usize>,
    data_len: usize,
    sample_size: usize,
    exhausted: bool,
}

#[cfg(test)]
impl ExhaustiveSampler {
    /// Create a new exhaustive sampler.
    pub fn new(data_len: usize, sample_size: usize) -> Self {
        let current = if sample_size <= data_len {
            (0..sample_size).collect()
        } else {
            vec![]
        };

        Self {
            current,
            data_len,
            sample_size,
            exhausted: sample_size > data_len,
        }
    }

    /// Check if all combinations have been generated.
    fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Get the next combination (returns None when exhausted).
    pub fn next_combination(&mut self) -> Option<Vec<usize>> {
        if self.exhausted {
            return None;
        }

        let result = self.current.clone();

        // Generate next combination (algorithm L from Knuth)
        self.advance();
        Some(result)
    }

    fn advance(&mut self) {
        let k = self.sample_size;
        let n = self.data_len;

        // Find rightmost element that can be incremented
        let mut i = k;
        while i > 0 {
            i -= 1;
            if self.current[i] < n - k + i {
                // Increment this element and reset all elements to its right
                self.current[i] += 1;
                for j in (i + 1)..k {
                    self.current[j] = self.current[j - 1] + 1;
                }
                return;
            }
        }

        // No more combinations
        self.exhausted = true;
    }

    /// Total number of combinations (n choose k).
    fn total_combinations(&self) -> usize {
        binomial(self.data_len, self.sample_size)
    }
}

#[cfg(test)]
impl Sampler for ExhaustiveSampler {
    fn sample(&mut self, _data_len: usize, _sample_size: usize) -> Vec<usize> {
        self.next_combination().unwrap_or_default()
    }

    fn sample_into(&mut self, _data_len: usize, _sample_size: usize, output: &mut Vec<usize>) {
        output.clear();
        if self.exhausted {
            return;
        }
        output.extend_from_slice(&self.current);
        self.advance();
    }
}

/// Compute binomial coefficient (n choose k).
///
/// Returns `None` if the result would overflow `usize`.
#[cfg(test)]
pub(crate) fn checked_combination_count(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    if k == 0 || k == n {
        return Some(1);
    }

    let k = k.min(n - k);
    let mut result = 1u128;

    for i in 0..k {
        result = result.checked_mul((n - i) as u128)? / (i + 1) as u128;
    }

    usize::try_from(result).ok()
}

/// Compute binomial coefficient (n choose k).
///
/// Returns `usize::MAX` on overflow.
#[cfg(test)]
fn binomial(n: usize, k: usize) -> usize {
    checked_combination_count(n, k).unwrap_or(usize::MAX)
}

// ============================================================================
// PROSAC Sampler
// ============================================================================

/// PROSAC (Progressive Sample Consensus) sampler.
///
/// PROSAC exploits the ordering of points by quality (e.g., matching score,
/// descriptor distance) to find good samples faster than uniform random sampling.
///
/// # How it works
///
/// 1. Points are assumed to be sorted by quality (best first, index 0 = highest quality)
/// 2. Initially, samples are drawn from the top-n points where n = sample_size
/// 3. As iterations progress, the sampling pool grows to include more points
/// 4. Eventually, PROSAC becomes equivalent to standard RANSAC
///
/// # Theory
///
/// Let T_n be the iteration at which point n is added to the sampling pool.
/// The growth function is computed such that the probability of selecting
/// a specific n-tuple equals the probability in standard RANSAC after T_N iterations.
///
/// # Examples
///
/// ```text
/// // Correspondences sorted by match quality (best first)
/// let sorted_correspondences = sort_by_match_score(&correspondences);
/// let mut prosac = ProsacSampler::new(correspondences.len(), 5, Some(42.into()));
///
/// for iter in 0..max_iterations {
///     let sample = prosac.sample(correspondences.len(), 5);
///     // Use sample indices into sorted_correspondences
/// }
/// ```
///
/// Reference: Chum, O., & Matas, J. (2005). Matching with PROSAC - Progressive
/// Sample Consensus. CVPR 2005.
#[derive(Clone)]
pub struct ProsacSampler {
    /// Number of data points.
    n_points: usize,
    /// Sample size (minimal set).
    sample_size: usize,
    /// Current iteration (1-indexed in paper, 0-indexed here).
    iteration: usize,
    /// Current pool size (n in paper, starts at sample_size).
    pool_size: usize,
    /// Precomputed T_n values: iteration at which point n enters the pool.
    growth_function: Vec<usize>,
    /// Random number generator.
    rng: AlgorithmRng,
    /// Maximum iterations before becoming pure RANSAC.
    max_prosac_iterations: usize,
}

impl ProsacSampler {
    /// Create a PROSAC sampler over `n_points` data points sorted by quality,
    /// drawing minimal samples of `sample_size` (for example 5 for the
    /// essential 5-point solver); a `None` seed selects the stable library seed.
    pub fn new(n_points: usize, sample_size: usize, seed: Option<RunSeed>) -> Self {
        Self::with_draw_budget(n_points, sample_size, seed, 200_000)
    }

    /// Create a sampler whose progressive schedule spans `maximum_draws`.
    pub(crate) fn with_draw_budget(
        n_points: usize,
        sample_size: usize,
        seed: Option<RunSeed>,
        maximum_draws: usize,
    ) -> Self {
        let rng = AlgorithmRng::new(seed.unwrap_or(DEFAULT_SAMPLER_SEED));

        // Compute growth function T_n
        // T_n is the iteration at which point n is added to sampling pool
        let growth_function =
            Self::compute_growth_function_with_budget(n_points, sample_size, maximum_draws as f64);

        // After this many iterations, pool includes all points (pure RANSAC)
        let max_prosac_iterations = growth_function.last().copied().unwrap_or(1);

        Self {
            n_points,
            sample_size,
            iteration: 0,
            pool_size: sample_size.min(n_points),
            growth_function,
            rng,
            max_prosac_iterations,
        }
    }

    /// Create with a specific seed for reproducibility.
    #[cfg(test)]
    fn with_seed(n_points: usize, sample_size: usize, seed: u64) -> Self {
        Self::new(n_points, sample_size, Some(RunSeed::from_u64(seed)))
    }

    /// Compute the growth function T_n.
    ///
    /// T_n represents the iteration at which point n is added to the sampling pool.
    /// Uses an iterative recurrence relation for numerical stability:
    ///
    ///   t_next = t_prev × (i+1) / (i+1-m)
    ///   increment = ceil(t_next - t_prev)
    ///   T[i] = T[i-1] + max(increment, 1)
    ///
    /// Starting from t_0 = T_N / C(N, m), where T_N = 200,000.
    ///
    /// Reference: Chum, O., & Matas, J. (2005). PROSAC. CVPR 2005.
    #[cfg(test)]
    fn compute_growth_function(n_points: usize, sample_size: usize) -> Vec<usize> {
        Self::compute_growth_function_with_budget(n_points, sample_size, 200_000.0)
    }

    fn compute_growth_function_with_budget(
        n_points: usize,
        sample_size: usize,
        maximum_draws: f64,
    ) -> Vec<usize> {
        if n_points < sample_size || sample_size == 0 {
            return vec![1; n_points];
        }

        let n = n_points;
        let m = sample_size;
        let t_n = maximum_draws;

        // C(N, m) via iterative multiplication (more stable than integer binomial)
        let mut c_nm: f64 = 1.0;
        for j in 1..=m {
            c_nm *= (n - j + 1) as f64 / j as f64;
        }

        let mut t_prev = t_n / (c_nm + 1e-300);

        let mut growth = vec![0usize; n_points];

        // T_m = 1 (first m points available from iteration 1)
        for item in growth.iter_mut().take(m.min(n)) {
            *item = 1;
        }

        // Iterative recurrence for T_n, n > m
        for i in m..n {
            let t_next = t_prev * ((i + 1) as f64 / (i + 1 - m) as f64);
            let increment = (t_next - t_prev).ceil() as usize;
            growth[i] = growth[i - 1].saturating_add(increment.max(1));
            t_prev = t_next;
        }

        growth
    }

    /// Get current pool size.
    #[cfg(test)]
    fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Check whether the progressive schedule has finished and sampling is uniform.
    pub fn is_ransac_mode(&self) -> bool {
        self.pool_size >= self.n_points && self.iteration > self.max_prosac_iterations
    }

    /// Sample according to PROSAC strategy.
    ///
    /// During the progressive schedule, the newest point in the current pool
    /// is forced into each sample and the remaining `m - 1` points are drawn
    /// from the preceding prefix. After the full schedule, sampling is uniform
    /// over all points.
    fn prosac_sample_into(&mut self, output: &mut Vec<usize>) {
        // T'_n is stored at growth_function[n - 1]. Move to the next prefix
        // once the current progressive stage has consumed its allotted draws.
        while self.pool_size < self.n_points
            && self.iteration > self.growth_function[self.pool_size - 1]
        {
            self.pool_size += 1;
        }

        let n = self.pool_size;
        let m = self.sample_size;

        if m == 0 || n <= m {
            // Pool too small, return all points
            output.extend(0..n);
            return;
        }

        output.resize(m, 0);
        output[0] = n - 1;
        if !draw_k_unique(&mut self.rng, n - 1, m - 1, &mut output[1..m]) {
            output.clear();
        }
    }
}

impl Sampler for ProsacSampler {
    fn sample(&mut self, data_len: usize, sample_size: usize) -> Vec<usize> {
        let mut output = Vec::with_capacity(sample_size.min(data_len));
        self.sample_into(data_len, sample_size, &mut output);
        output
    }

    fn sample_into(&mut self, data_len: usize, sample_size: usize, output: &mut Vec<usize>) {
        output.clear();
        // Update n_points if data_len changed (shouldn't happen in normal use)
        if data_len != self.n_points {
            *self = Self::new(data_len, sample_size, None);
        }

        // Increment iteration
        self.iteration += 1;

        // After T'_N draws, PROSAC is exactly uniform RANSAC.
        if self.is_ransac_mode() {
            // Pure RANSAC mode — use same draw_k_unique for consistent RNG behavior
            let effective_size = sample_size.min(data_len);
            output.resize(effective_size, 0);
            if !draw_k_unique(&mut self.rng, data_len, effective_size, output) {
                output.clear();
            }
            return;
        }

        self.prosac_sample_into(output);
    }
}

/// Draw k unique indices from `[0, n)` with the versioned project sampler.
///
/// Returns `false` when the requested sample cannot be represented by `out`.
#[inline]
#[must_use]
pub(crate) fn draw_k_unique(rng: &mut AlgorithmRng, n: usize, k: usize, out: &mut [usize]) -> bool {
    rng.sample_indices(n, k, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_random_sampler_size() {
        let mut sampler = RandomSampler::new(Some(42.into()));

        let sample = sampler.sample(100, 8);
        assert_eq!(sample.len(), 8);

        // All indices should be unique
        let unique: HashSet<_> = sample.iter().collect();
        assert_eq!(unique.len(), 8);

        // All indices should be in range
        assert!(sample.iter().all(|&i| i < 100));
    }

    #[test]
    fn random_sampler_seed_contract_is_reproducible() {
        let mut sampler1 = RandomSampler::with_seed(42);
        let mut sampler2 = RandomSampler::with_seed(42);
        assert_eq!(sampler1.sample(100, 8), sampler2.sample(100, 8));

        let mut default1 = RandomSampler::default();
        let mut default2 = RandomSampler::default();
        assert_eq!(default1.sample(100, 8), default2.sample(100, 8));
    }

    #[test]
    fn test_random_sampler_oversized() {
        let mut sampler = RandomSampler::new(Some(42.into()));

        let sample = sampler.sample(5, 10);
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_exhaustive_sampler() {
        let mut sampler = ExhaustiveSampler::new(5, 3);

        // 5 choose 3 = 10 combinations
        assert_eq!(sampler.total_combinations(), 10);

        let mut combinations = Vec::new();
        while let Some(combo) = sampler.next_combination() {
            combinations.push(combo);
        }

        assert_eq!(combinations.len(), 10);
        assert!(sampler.is_exhausted());

        // Check first and last
        assert_eq!(combinations[0], vec![0, 1, 2]);
        assert_eq!(combinations[9], vec![2, 3, 4]);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(10, 5), 252);
        assert_eq!(binomial(4, 4), 1);
        assert_eq!(binomial(4, 0), 1);
        assert_eq!(binomial(3, 5), 0);
        assert_eq!(
            checked_combination_count(67, 33),
            Some(14_226_520_737_620_288_370)
        );
        assert_eq!(checked_combination_count(68, 34), None);
    }

    #[test]
    fn test_random_sampler_large_sample() {
        // Test the HashSet path (sample_size > 8)
        let mut sampler = RandomSampler::new(Some(42.into()));

        let sample = sampler.sample(1000, 20);
        assert_eq!(sample.len(), 20);

        // All indices should be unique
        let unique: HashSet<_> = sample.iter().collect();
        assert_eq!(unique.len(), 20);

        // All indices should be in range
        assert!(sample.iter().all(|&i| i < 1000));
    }

    // ============================================================================
    // PROSAC Sampler Tests
    // ============================================================================

    #[test]
    fn prosac_seed_contract_is_reproducible() {
        let mut sampler1 = ProsacSampler::with_seed(100, 5, 42);
        let mut sampler2 = ProsacSampler::with_seed(100, 5, 42);

        for _ in 0..10 {
            let sample1 = sampler1.sample(100, 5);
            let sample2 = sampler2.sample(100, 5);
            assert_eq!(sample1, sample2);
        }

        let mut sampler1 = ProsacSampler::new(100, 5, None);
        let mut sampler2 = ProsacSampler::new(100, 5, None);
        for _ in 0..10 {
            assert_eq!(sampler1.sample(100, 5), sampler2.sample(100, 5));
        }
    }

    #[test]
    fn test_prosac_forces_the_newest_prefix_point() {
        let mut sampler = ProsacSampler::new(20, 4, Some(42.into()));

        for _ in 0..200 {
            let sample = sampler.sample(20, 4);
            if sampler.is_ransac_mode() {
                break;
            }
            assert!(
                sample.contains(&(sampler.pool_size() - 1)),
                "progressive draws must include the newest ranked point"
            );
            assert!(sample.iter().all(|&index| index < sampler.pool_size()));
        }
    }

    #[test]
    fn test_prosac_growth_function_properties() {
        let n_points = 100;
        let sample_size = 5;
        let growth = ProsacSampler::compute_growth_function(n_points, sample_size);

        // First sample_size points should be available from iteration 1
        for item in growth.iter().take(sample_size) {
            assert_eq!(
                *item, 1,
                "First {} points should be available at iteration 1",
                sample_size
            );
        }

        // Growth function should be monotonically increasing
        for i in 1..n_points {
            assert!(
                growth[i] >= growth[i - 1],
                "Growth function should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_prosac_reference_schedule() {
        let growth = ProsacSampler::compute_growth_function_with_budget(5, 2, 20.0);
        assert_eq!(growth, vec![1, 1, 5, 11, 19]);
    }

    #[test]
    fn test_prosac_reference_draw_phases() {
        let mut sampler = ProsacSampler::new(5, 2, Some(42.into()));
        sampler.growth_function = ProsacSampler::compute_growth_function_with_budget(5, 2, 20.0);
        sampler.max_prosac_iterations = 19;

        for draw in 1..=19 {
            let sample = sampler.sample(5, 2);
            let expected_pool = match draw {
                1 => 2,
                2..=5 => 3,
                6..=11 => 4,
                _ => 5,
            };
            assert_eq!(sampler.pool_size(), expected_pool);
            assert_eq!(sample.len(), 2);
            assert_ne!(sample[0], sample[1]);
            assert!(sample.iter().all(|&index| index < expected_pool));
            assert!(sample.contains(&(expected_pool - 1)));
            assert!(!sampler.is_ransac_mode());
            if draw == 1 {
                assert_eq!(sample, vec![0, 1]);
            }
        }

        let sample = sampler.sample(5, 2);
        assert_eq!(sample.len(), 2);
        assert!(sampler.is_ransac_mode());
    }

    #[test]
    fn test_prosac_handles_sample_at_or_above_population() {
        let mut sampler = ProsacSampler::new(3, 5, Some(42.into()));
        assert_eq!(sampler.sample(3, 5), vec![0, 1, 2]);

        let mut sampler = ProsacSampler::new(5, 5, Some(42.into()));
        assert_eq!(sampler.sample(5, 5), vec![0, 1, 2, 3, 4]);
    }
}
