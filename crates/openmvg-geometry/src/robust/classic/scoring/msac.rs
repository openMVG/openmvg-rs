// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MSAC (M-estimator Sample Consensus) scoring.
//!
//! Uses a truncated quadratic loss function for smoother scoring.

use super::Scorer;
#[cfg(test)]
use crate::robust::error::RobustError;

/// MSAC scorer with truncated quadratic loss.
///
/// Unlike binary scoring, MSAC uses a smooth cost function:
/// - Inliers: cost = residual²
/// - Outliers: cost = threshold² (capped)
///
/// This gives more weight to inliers with small residuals.
#[derive(Debug, Clone)]
pub struct MsacScorer {
    /// Threshold for inlier classification.
    threshold: f64,
    /// Threshold squared (cached).
    threshold_squared: f64,
}

impl MsacScorer {
    /// Create an MSAC scorer whose `threshold` is both the inlier threshold
    /// and the truncated-loss cutoff.
    #[cfg(test)]
    pub fn try_new(threshold: f64) -> Result<Self, RobustError> {
        if !threshold.is_finite() || threshold <= 0.0 || !(threshold * threshold).is_finite() {
            return Err(RobustError::InvalidConfig {
                reason: "MSAC threshold must be finite, positive, and safely squareable".to_owned(),
            });
        }
        Ok(Self::from_validated_threshold(threshold))
    }

    pub(crate) fn from_validated_threshold(threshold: f64) -> Self {
        Self {
            threshold,
            threshold_squared: threshold * threshold,
        }
    }

    /// Compute the MSAC cost for a single squared residual.
    fn cost(&self, squared_residual: f64) -> f64 {
        squared_residual.min(self.threshold_squared)
    }

    /// MSAC score with early exit — bails when running cost exceeds the cutoff.
    ///
    /// Returns `None` if the model is worse than the cutoff (i.e., -cost < cutoff_score),
    /// saving the remaining squared-residual computations. Returns `Some(score)` otherwise.
    ///
    /// `cutoff_score` is the current best score (negative cost, higher = better).
    #[inline]
    pub fn score_with_early_exit<M, D, F>(
        &self,
        model: &M,
        data: &[D],
        squared_residual_fn: F,
        cutoff_score: f64,
    ) -> Option<f64>
    where
        F: Fn(&M, &D) -> f64,
    {
        let cutoff_cost = -cutoff_score; // Convert score (negative) to cost (positive)
        let mut running_cost = 0.0;
        for d in data {
            running_cost += squared_residual_fn(model, d).min(self.threshold_squared);
            if running_cost >= cutoff_cost {
                return None; // This model is already worse than the best
            }
        }
        let score = -running_cost;
        (score > cutoff_score).then_some(score)
    }
}

impl Scorer for MsacScorer {
    fn score<M, D, F>(&self, model: &M, data: &[D], squared_residual_fn: F) -> f64
    where
        F: Fn(&M, &D) -> f64,
    {
        // MSAC score: negative total cost (higher = better)
        let total_cost: f64 = data
            .iter()
            .map(|datum| self.cost(squared_residual_fn(model, datum)))
            .sum();

        // Invert so that lower cost = higher score
        -total_cost
    }

    /// Early-exit MSAC scoring — bails as soon as running cost exceeds the best.
    fn score_with_cutoff<M, D, F>(
        &self,
        model: &M,
        data: &[D],
        squared_residual_fn: F,
        cutoff_score: f64,
    ) -> Option<f64>
    where
        F: Fn(&M, &D) -> f64,
    {
        self.score_with_early_exit(model, data, squared_residual_fn, cutoff_score)
    }

    #[cfg(test)]
    fn find_inliers<M, D, F>(&self, model: &M, data: &[D], squared_residual_fn: F) -> Vec<usize>
    where
        F: Fn(&M, &D) -> f64,
    {
        // Use the allocation-free method internally
        let mut out = Vec::with_capacity(data.len());
        self.collect_inliers_into(model, data, &mut out, squared_residual_fn);
        out
    }

    /// Optimized implementation using pre-cached threshold_squared.
    fn collect_inliers_into<M, D, F>(
        &self,
        model: &M,
        data: &[D],
        out: &mut Vec<usize>,
        squared_residual_fn: F,
    ) where
        F: Fn(&M, &D) -> f64,
    {
        out.clear();
        for (i, d) in data.iter().enumerate() {
            if squared_residual_fn(model, d) <= self.threshold_squared {
                out.push(i);
            }
        }
    }

    fn score_and_collect_inliers_with_cutoff<M, D, F>(
        &self,
        model: &M,
        data: &[D],
        out: &mut Vec<usize>,
        squared_residual_fn: F,
        cutoff_score: f64,
    ) -> Option<f64>
    where
        F: Fn(&M, &D) -> f64,
    {
        out.clear();
        let cutoff_cost = -cutoff_score;
        let mut running_cost = 0.0;
        for (index, datum) in data.iter().enumerate() {
            let squared_residual = squared_residual_fn(model, datum);
            running_cost += squared_residual.min(self.threshold_squared);
            if squared_residual <= self.threshold_squared {
                out.push(index);
            }
            if running_cost >= cutoff_cost {
                return None;
            }
        }
        let score = -running_cost;
        (score > cutoff_score).then_some(score)
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msac_cost() {
        let scorer = MsacScorer::try_new(1.0).unwrap();

        // Residual 0.5: cost = 0.25
        assert!((scorer.cost(0.25) - 0.25).abs() < 1e-10);

        // Residual 1.0: cost = 1.0 (at threshold)
        assert!((scorer.cost(1.0) - 1.0).abs() < 1e-10);

        // Residual 3.0: cost = 1.0 (capped at threshold²)
        assert!((scorer.cost(9.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_msac_scoring() {
        let scorer = MsacScorer::try_new(1.0).unwrap();

        let model = 0.0;
        let data = vec![0.5, 0.8, 1.5, 2.0, 0.3];
        let residual_fn = |_m: &f64, d: &f64| *d * *d;

        // Score is negative total cost (higher = better)
        let score = scorer.score(&model, &data, residual_fn);

        // Costs: 0.25 + 0.64 + 1.0 + 1.0 + 0.09 = 2.98
        assert!(score < 0.0); // Negative cost
        assert!((score + 2.98).abs() < 0.01);
    }

    #[test]
    fn test_msac_inliers() {
        let scorer = MsacScorer::try_new(1.0).unwrap();

        let model = 0.0;
        let data = vec![0.5, 0.8, 1.5, 2.0, 0.3];
        let residual_fn = |_m: &f64, d: &f64| *d * *d;

        // Same as binary for inlier classification
        let inliers = scorer.find_inliers(&model, &data, residual_fn);
        assert_eq!(inliers, vec![0, 1, 4]);
    }

    #[test]
    fn test_msac_collect_inliers_into() {
        let scorer = MsacScorer::try_new(1.0).unwrap();

        let model = 0.0;
        let data = vec![0.5, 0.8, 1.5, 2.0, 0.3];
        let residual_fn = |_m: &f64, d: &f64| *d * *d;

        // Test with pre-allocated buffer
        let mut inliers = Vec::with_capacity(data.len());
        scorer.collect_inliers_into(&model, &data, &mut inliers, residual_fn);
        assert_eq!(inliers, vec![0, 1, 4]);

        // Verify buffer is cleared and reused correctly
        let data2 = vec![0.2, 0.3];
        scorer.collect_inliers_into(&model, &data2, &mut inliers, residual_fn);
        assert_eq!(inliers, vec![0, 1]);
    }

    #[test]
    fn early_exit_matches_full_msac_score() {
        let scorer = MsacScorer::try_new(1.0).unwrap();
        let data = [0.0, 1.0, 1.000_001, f64::INFINITY, f64::NAN];
        let full = scorer.score(&(), &data, |_, value| *value * *value);

        for cutoff in [f64::NEG_INFINITY, full - 1.0, full, full + 1.0] {
            assert_eq!(
                scorer.score_with_cutoff(&(), &data, |_, value| *value * *value, cutoff),
                (full > cutoff).then_some(full)
            );
        }
    }
}
