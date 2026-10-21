// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scoring strategies for classic robust estimation.
//!
//! Provides different ways to evaluate model quality.
//!
//! # Available scorers
//!
//! | Scorer | Description | Use Case |
//! |--------|-------------|----------|
//! | [`BinaryScorer`] | Binary inlier/outlier classification | Standard RANSAC |
//! | [`MsacScorer`] | Soft threshold (capped residuals) | Better near threshold |
//!
//! # Early exit
//!
//! The [`Scorer::score_with_cutoff`] method enables early termination when
//! the running score cannot exceed a cutoff threshold. This is critical for
//! performance in RANSAC where most models are bad.
//!
//! **Semantics**: Returns `Some(score)` if score > cutoff, `None` otherwise.
//! The scorer may return `None` early without computing the full score if
//! it determines the cutoff cannot be exceeded.
//!
pub mod binary;
pub mod msac;

pub use binary::BinaryScorer;
pub use msac::MsacScorer;

/// Trait for scoring strategies.
///
/// Scorers evaluate squared residuals and identify inliers.
///
/// # Implementation notes
///
/// - **Thread safety**: Scorers must be `Clone + Send + Sync` for parallel RANSAC.
/// - **Determinism**: Scorers should produce identical results for identical inputs.
/// - **Early exit**: Override [`score_with_cutoff`] for better RANSAC performance.
pub trait Scorer: Clone + Send + Sync {
    /// Score a model against all data points.
    ///
    /// Higher scores indicate better models.
    fn score<M, D, F>(&self, model: &M, data: &[D], squared_residual_fn: F) -> f64
    where
        F: Fn(&M, &D) -> f64;

    /// Score with early exit — returns `None` if running score cannot exceed `cutoff_score`.
    ///
    /// # Early exit
    ///
    /// This method enables O(1) bail-out on bad models:
    /// - Returns `Some(score)` if the final score exceeds `cutoff_score`
    /// - Returns `None` if the score is ≤ cutoff (may exit early without full computation)
    ///
    /// The default implementation computes the full score and compares. Concrete
    /// scorers (e.g., [`MsacScorer`]) override this for true early termination.
    ///
    /// # Examples
    ///
    /// ```text
    /// let best_score = 50.0;
    /// if let Some(score) = scorer.score_with_cutoff(&model, &data, residual_fn, best_score) {
    ///     // This model beat the best — update
    ///     best_score = score;
    /// }
    /// // If None, the model was worse — skip without full computation
    /// ```
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
        let s = self.score(model, data, squared_residual_fn);
        if s > cutoff_score { Some(s) } else { None }
    }

    /// Find inlier indices (allocates a new Vec).
    ///
    /// For allocation-free variant, use [`collect_inliers_into`].
    #[cfg(test)]
    fn find_inliers<M, D, F>(&self, model: &M, data: &[D], squared_residual_fn: F) -> Vec<usize>
    where
        F: Fn(&M, &D) -> f64;

    /// Collect inlier indices into a pre-allocated buffer (allocation-free).
    ///
    /// Clears `out` and fills it with the indices of the `data` points that
    /// are inliers to `model` under `squared_residual_fn`. Use this in hot
    /// loops (for example RANSAC) to avoid repeated allocations.
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut inliers = Vec::with_capacity(data.len());
    /// for _ in 0..1000 {
    ///     scorer.collect_inliers_into(&model, &data, &mut inliers, |m, d| squared_residual(m, d));
    ///     // inliers is reused each iteration - no allocation!
    /// }
    /// ```
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
        let threshold_sq = self.threshold() * self.threshold();
        for (i, d) in data.iter().enumerate() {
            if squared_residual_fn(model, d) <= threshold_sq {
                out.push(i);
            }
        }
    }

    /// Score a model and retain its inliers in one pass.
    ///
    /// `out` may contain a partial set when the candidate cannot beat the
    /// cutoff and `None` is returned.
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
        let score = self.score_with_cutoff(model, data, &squared_residual_fn, cutoff_score)?;
        self.collect_inliers_into(model, data, out, squared_residual_fn);
        Some(score)
    }

    /// Get the threshold used for inlier determination.
    fn threshold(&self) -> f64;
}
