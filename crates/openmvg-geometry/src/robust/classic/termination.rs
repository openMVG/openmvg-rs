// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Termination criteria for classic robust estimation.
//!
//! Provides different strategies for deciding when to stop iterating.

/// Trait for termination strategies.
pub trait Terminator: Clone + Send + Sync {
    /// Check whether estimation should stop at the 0-based `iteration`, given
    /// the best inlier ratio found so far and the minimal sample size.
    fn should_stop(&self, iteration: usize, best_inlier_ratio: f64, sample_size: usize) -> bool;
}

/// Fixed iteration terminator.
///
/// Runs for a fixed number of iterations.
#[derive(Debug, Clone)]
pub struct FixedTerminator {
    max_iterations: usize,
}

impl FixedTerminator {
    /// Create a new fixed terminator.
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl Terminator for FixedTerminator {
    fn should_stop(&self, iteration: usize, _best_inlier_ratio: f64, _sample_size: usize) -> bool {
        iteration >= self.max_iterations
    }
}

/// Exact terminator using hypergeometric probability.
///
/// The standard RANSAC stopping criterion uses `w^n` which assumes sampling
/// with replacement from an infinite population. RANSAC samples without
/// replacement from a finite population.
///
/// This terminator uses the exact hypergeometric probability:
/// ```text
/// P(k inliers) = Π_{i=0}^{k-1} (num_inliers - i) / (n - i)
/// ```
///
/// On finite datasets the `w^n` approximation overestimates the probability
/// of an all-inlier sample, so it can stop early; the exact form does not.
///
/// Reference: J. L. Schönberger, V. Larsson, M. Pollefeys, "Fixing the RANSAC
/// Stopping Criterion", arXiv:2503.07829, 2025.
#[derive(Debug, Clone)]
pub struct ExactTerminator {
    /// Desired confidence level for finding a good model (0 to 1).
    confidence: f64,

    /// Maximum iterations cap.
    max_iterations: usize,

    /// Minimum iterations before early termination.
    min_iterations: usize,

    /// Total number of data points (needed for exact calculation).
    total_data_points: usize,
}

impl ExactTerminator {
    /// Create an exact terminator for a desired success probability
    /// `confidence` (for example 0.99), an iteration cap, and the total
    /// number of data points.
    pub fn new(confidence: f64, max_iterations: usize, data_size: usize) -> Self {
        Self {
            confidence,
            max_iterations,
            min_iterations: 10,
            total_data_points: data_size,
        }
    }

    /// Compute the exact probability of drawing k inliers from n samples.
    ///
    /// Uses hypergeometric distribution (sampling without replacement):
    /// P = Π_{i=0}^{k-1} (num_inliers - i) / (total - i)
    ///
    /// This is the probability that all k samples in a minimal set are inliers.
    fn exact_inlier_probability(num_inliers: usize, total: usize, sample_size: usize) -> f64 {
        if num_inliers < sample_size || total < sample_size {
            return 0.0;
        }

        let mut prob = 1.0;
        for i in 0..sample_size {
            prob *= (num_inliers - i) as f64 / (total - i) as f64;
        }
        prob
    }

    /// Compute required iterations using exact hypergeometric probability.
    ///
    /// k = ceil(log(1 - confidence) / log(1 - p_exact))
    fn compute_iterations_exact(&self, num_inliers: usize, sample_size: usize) -> usize {
        let p_exact =
            Self::exact_inlier_probability(num_inliers, self.total_data_points, sample_size);

        if p_exact <= 0.0 {
            return self.max_iterations;
        }

        if p_exact >= 1.0 {
            return 1;
        }

        // k = log(1 - confidence) / log(1 - p_exact)
        let log_denom = (1.0 - p_exact).ln();

        if log_denom.abs() < 1e-15 {
            return self.max_iterations;
        }

        let log_num = (1.0 - self.confidence).ln();
        let k = (log_num / log_denom).ceil() as usize;

        k.min(self.max_iterations).max(self.min_iterations)
    }

    /// Compare exact vs approximate iteration counts for analysis.
    #[cfg(test)]
    fn compare_with_approximate(&self, inlier_ratio: f64, sample_size: usize) -> (usize, usize) {
        let num_inliers = (inlier_ratio * self.total_data_points as f64).round() as usize;
        let exact = self.compute_iterations_exact(num_inliers, sample_size);

        // Approximate (standard RANSAC)
        let w_n = inlier_ratio.powi(sample_size as i32);
        let approx = if w_n > 0.0 && w_n < 1.0 {
            let k = ((1.0 - self.confidence).ln() / (1.0 - w_n).ln()).ceil() as usize;
            k.min(self.max_iterations).max(self.min_iterations)
        } else if w_n >= 1.0 {
            1
        } else {
            self.max_iterations
        };

        (exact, approx)
    }
}

impl Terminator for ExactTerminator {
    fn should_stop(&self, iteration: usize, best_inlier_ratio: f64, sample_size: usize) -> bool {
        if iteration >= self.max_iterations {
            return true;
        }

        if iteration < self.min_iterations {
            return false;
        }

        let num_inliers = (best_inlier_ratio * self.total_data_points as f64).round() as usize;
        let required = self.compute_iterations_exact(num_inliers, sample_size);
        iteration >= required
    }
}

impl Default for ExactTerminator {
    fn default() -> Self {
        // Default with 1000 data points - should be overridden with actual data size
        Self::new(0.99, 10000, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_terminator() {
        let term = FixedTerminator::new(100);

        assert!(!term.should_stop(50, 0.5, 8));
        assert!(!term.should_stop(99, 0.5, 8));
        assert!(term.should_stop(100, 0.5, 8));
        assert!(term.should_stop(150, 0.5, 8));
    }

    // Tests for ExactTerminator
    #[test]
    fn test_exact_inlier_probability() {
        // 50 inliers out of 100, sample size 4
        // P = (50/100) * (49/99) * (48/98) * (47/97)
        let prob = ExactTerminator::exact_inlier_probability(50, 100, 4);
        let expected = (50.0 / 100.0) * (49.0 / 99.0) * (48.0 / 98.0) * (47.0 / 97.0);
        assert!((prob - expected).abs() < 1e-10);
    }

    #[test]
    fn test_exact_vs_approximate_difference() {
        // With finite population, exact probability differs from w^n
        let term = ExactTerminator::new(0.99, 10000, 100);
        let (exact, approx) = term.compare_with_approximate(0.5, 4);

        // Exact should require MORE iterations than approximate
        // because exact probability is lower (sampling without replacement)
        assert!(
            exact >= approx,
            "Exact {} should be >= Approx {}",
            exact,
            approx
        );
    }

    #[test]
    fn test_exact_terminator_high_inlier_ratio() {
        let term = ExactTerminator::new(0.99, 10000, 1000);

        // High inlier ratio should still terminate reasonably
        let iterations = term.compute_iterations_exact(900, 4);
        assert!(iterations < 100, "Got {} iterations", iterations);
    }

    #[test]
    fn test_exact_terminator_low_inlier_ratio() {
        let term = ExactTerminator::new(0.99, 10000, 1000);

        // Low inlier ratio needs more iterations
        let iterations = term.compute_iterations_exact(200, 4);
        assert!(iterations > 100, "Got {} iterations", iterations);
    }
}
