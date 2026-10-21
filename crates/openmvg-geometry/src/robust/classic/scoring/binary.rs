// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Binary inlier scoring (classic RANSAC).
//!
//! Points are classified as inliers or outliers based on a threshold.

use super::Scorer;

/// Binary scorer for classic RANSAC.
///
/// Points with residual <= threshold are inliers, others are outliers.
/// Score is the count of inliers.
#[derive(Debug, Clone)]
pub struct BinaryScorer {
    /// Threshold for inlier classification (squared for efficiency).
    threshold_squared: f64,
}

impl BinaryScorer {
    pub(crate) fn from_validated_threshold(threshold: f64) -> Self {
        Self {
            threshold_squared: threshold * threshold,
        }
    }
}

impl Scorer for BinaryScorer {
    fn score<M, D, F>(&self, model: &M, data: &[D], squared_residual_fn: F) -> f64
    where
        F: Fn(&M, &D) -> f64,
    {
        // Score is the count of inliers (higher is better)
        data.iter()
            .filter(|datum| squared_residual_fn(model, datum) <= self.threshold_squared)
            .count() as f64
    }

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
        let mut inliers = 0usize;
        for (index, datum) in data.iter().enumerate() {
            if squared_residual_fn(model, datum) <= self.threshold_squared {
                inliers += 1;
            }
            let remaining = data.len() - index - 1;
            if (inliers + remaining) as f64 <= cutoff_score {
                return None;
            }
        }
        Some(inliers as f64)
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
        for (index, datum) in data.iter().enumerate() {
            if squared_residual_fn(model, datum) <= self.threshold_squared {
                out.push(index);
            }
            let remaining = data.len() - index - 1;
            if (out.len() + remaining) as f64 <= cutoff_score {
                return None;
            }
        }
        Some(out.len() as f64)
    }

    fn threshold(&self) -> f64 {
        self.threshold_squared.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correspondence::PixelCorrespondence;
    use crate::homography::{Homography, UnnormalizedHomographySolver};
    use crate::solver_traits::Solver;
    use nalgebra::Matrix3;

    #[test]
    fn test_binary_scorer_count() {
        let scorer = BinaryScorer::from_validated_threshold(1.0);

        let model = 0.0; // Dummy model
        let data = vec![0.5, 0.8, 1.5, 2.0, 0.3];

        // Residual function: return the data value
        let residual_fn = |_m: &f64, d: &f64| *d * *d;

        let score = scorer.score(&model, &data, residual_fn);
        // 0.5, 0.8, 0.3 are <= 1.0
        assert_eq!(score, 3.0);
    }

    #[test]
    fn test_binary_scorer_inliers() {
        let scorer = BinaryScorer::from_validated_threshold(1.0);

        let model = 0.0;
        let data = vec![0.5, 0.8, 1.5, 2.0, 0.3];
        let residual_fn = |_m: &f64, d: &f64| *d * *d;

        let inliers = scorer.find_inliers(&model, &data, residual_fn);
        assert_eq!(inliers, vec![0, 1, 4]);
    }

    #[test]
    fn test_binary_scorer_collect_inliers_into() {
        let scorer = BinaryScorer::from_validated_threshold(1.0);

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
    fn test_binary_scorer_threshold() {
        let scorer = BinaryScorer::from_validated_threshold(2.5);
        assert!((scorer.threshold() - 2.5).abs() < 1e-10);
    }

    /// Unsquared robust threshold contract: the public threshold is interpreted as a distance while
    /// the solver residual is squared; a squared-versus-unsquared threshold mutation changes the
    /// boundary classification and fails.
    #[test]
    fn test_homography_solver_residual_matches_unsquared_scorer_threshold() {
        let model = Homography::new(Matrix3::identity());
        let datum = PixelCorrespondence::from_coords(0.0, 0.0, 3.0, 4.0);
        let data = [datum];
        let distance = 50.0_f64.sqrt();

        let below = BinaryScorer::from_validated_threshold(distance - 1e-6);
        assert!(
            below
                .find_inliers(
                    &model,
                    &data,
                    <UnnormalizedHomographySolver as Solver>::squared_residual
                )
                .is_empty()
        );

        let above = BinaryScorer::from_validated_threshold(distance + 1e-6);
        assert_eq!(
            above.find_inliers(
                &model,
                &data,
                <UnnormalizedHomographySolver as Solver>::squared_residual
            ),
            vec![0]
        );
    }

    #[test]
    fn early_exit_matches_full_binary_score() {
        let scorer = BinaryScorer::from_validated_threshold(1.0);
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
