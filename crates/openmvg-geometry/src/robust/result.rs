// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output types of robust geometry estimation (RANSAC and related algorithms).

/// Model returned by a robust estimator, with its inlier indices and statistics.
///
/// Read it through its methods, as [`AbsolutePoseEstimate`](crate::pose::AbsolutePoseEstimate)
/// is read: [`Self::model`], [`Self::inliers`], [`Self::stats`] and [`Self::diagnostics`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RobustEstimate<M> {
    pub(crate) model: M,
    pub(crate) inliers: Vec<usize>,
    pub(crate) stats: EstimateStats,
    pub(crate) diagnostics: Option<Diagnostics>,
}

impl<M> RobustEstimate<M> {
    /// Create an estimate without diagnostics, for a solver of your own.
    pub fn new(model: M, inliers: Vec<usize>, stats: EstimateStats) -> Self {
        Self {
            model,
            inliers,
            stats,
            diagnostics: None,
        }
    }

    /// Attach diagnostics to the estimate.
    #[must_use]
    pub fn with_diagnostics(mut self, diagnostics: Diagnostics) -> Self {
        self.diagnostics = Some(diagnostics);
        self
    }

    /// The estimated model.
    #[must_use]
    pub const fn model(&self) -> &M {
        &self.model
    }

    /// Indices of the inlier data points.
    #[must_use]
    pub fn inliers(&self) -> &[usize] {
        &self.inliers
    }

    /// Number of inliers.
    #[must_use]
    pub fn inlier_count(&self) -> usize {
        self.inliers.len()
    }

    /// Statistics of the robust search, always computed.
    #[must_use]
    pub const fn stats(&self) -> &EstimateStats {
        &self.stats
    }

    /// Diagnostics of the robust search, when requested.
    #[must_use]
    pub const fn diagnostics(&self) -> Option<&Diagnostics> {
        self.diagnostics.as_ref()
    }

    /// Consume the estimate and return its model.
    #[must_use]
    pub fn into_model(self) -> M {
        self.model
    }

    /// Consume the estimate and return its model and inlier indices.
    #[must_use]
    pub fn into_parts(self) -> (M, Vec<usize>) {
        (self.model, self.inliers)
    }
}

/// Statistics about the robust estimation procedure.
///
/// These are always computed and have minimal overhead.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EstimateStats {
    /// Number of minimal samples attempted, or exhaustive combinations visited.
    pub iteration_count: usize,

    /// Best estimator-specific score; values are not comparable across methods.
    pub score: f64,

    /// Ratio of inliers to total data points.
    pub inlier_ratio: f64,

    /// Number of valid models evaluated.
    pub evaluated_model_count: usize,

    /// Number of samples that produced degenerate configurations.
    pub degenerate_sample_count: usize,
}

impl EstimateStats {
    /// Create new statistics with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create statistics with the given iteration count.
    #[must_use]
    pub fn with_iteration_count(mut self, iteration_count: usize) -> Self {
        self.iteration_count = iteration_count;
        self
    }

    /// Set the score.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Set the inlier ratio.
    #[must_use]
    pub fn with_inlier_ratio(mut self, ratio: f64) -> Self {
        self.inlier_ratio = ratio;
        self
    }
}

/// Detailed diagnostics for debugging and analysis.
///
/// These are opt-in and may have computational overhead.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Residuals for all data points against the final model.
    pub residuals: Vec<f64>,

    /// History of best scores over draws for classic estimation or rounds for GenRANSAC.
    pub score_history: Vec<f64>,

    /// Sampling or guidance pool size at termination, when applicable.
    pub final_pool_size: Option<usize>,

    /// Number of local optimization refinement iterations performed.
    pub refinement_iteration_count: usize,

    /// Effective adaptive draw cap at termination.
    pub final_adaptive_cap: usize,

    /// Number of datum/model residuals evaluated by the estimator.
    pub residual_evaluation_count: usize,
}

impl Diagnostics {
    /// Create new empty diagnostics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a score in the score history.
    pub fn record_score(&mut self, score: f64) {
        self.score_history.push(score);
    }

    /// Set the residuals.
    #[must_use]
    pub fn with_residuals(mut self, residuals: Vec<f64>) -> Self {
        self.residuals = residuals;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robust_result_creation() {
        let result = RobustEstimate::new(42.0, vec![0, 2, 5], EstimateStats::new());

        assert_eq!(result.model, 42.0);
        assert_eq!(result.inlier_count(), 3);
    }
}
