// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit non-minimal refinement policies shared by geometry families.
//!
//! Minimal solvers such as F7 and E5 reject oversized inlier sets. Keeping the
//! refiner separate makes that invalid combination impossible to hide inside
//! the estimator: use F8/E8 (or a nonlinear optimizer) for their refit step.
//!
//! ```text
//! inlier set -> non-minimal solver -> validity check
//!           -> replacement model (or retain minimal model)
//! ```

use std::marker::PhantomData;

use crate::SolverError;

use crate::robust::error::RobustError;
use crate::solver_traits::{SingleSolutionSolver, Solver};

/// Model-specific refinement hooks used by GenRANSAC.
pub(crate) trait GenRansacRefiner<S: Solver> {
    /// Re-estimate a model from the current consensus.
    fn refit(
        &self,
        seed: &S::Model,
        inliers: &[usize],
        data: &[S::Datum],
        threshold: f64,
    ) -> Result<Option<S::Model>, RobustError>;
}

/// Policy for tests or models that do not yet have a valid non-minimal solver.
#[derive(Debug, Clone, Copy, Default)]
#[cfg(test)]
pub(crate) struct NoRefiner;

#[cfg(test)]
impl<S: Solver> GenRansacRefiner<S> for NoRefiner {
    fn refit(
        &self,
        _seed: &S::Model,
        _inliers: &[usize],
        _data: &[S::Datum],
        _threshold: f64,
    ) -> Result<Option<S::Model>, RobustError> {
        Ok(None)
    }
}

/// Refit policy backed by a single-solution, normally overdetermined solver.
///
/// For example, use `SolverRefiner<EightPointSolver>` with minimal
/// `SevenPointSolver` hypotheses. Keeping this policy single-solution avoids a
/// one-element `Vec` in every local-optimization step and matches the E8/F8
/// refiners of the library facades. A multi-root refinement policy can
/// be added separately if a concrete model needs one.
#[derive(Debug, Clone)]
pub(crate) struct SolverRefiner<R: Solver> {
    config: R::Config,
    marker: PhantomData<R>,
}

impl<R: Solver> SolverRefiner<R> {
    /// Create a refiner from the non-minimal solver's configuration.
    pub(crate) fn new(config: R::Config) -> Self {
        Self {
            config,
            marker: PhantomData,
        }
    }
}

impl<S, R> GenRansacRefiner<S> for SolverRefiner<R>
where
    S: Solver,
    R: SingleSolutionSolver<Datum = S::Datum, Model = S::Model>,
{
    fn refit(
        &self,
        _seed: &S::Model,
        inliers: &[usize],
        data: &[S::Datum],
        _threshold: f64,
    ) -> Result<Option<S::Model>, RobustError> {
        if inliers.len() < R::min_sample_size() {
            return Ok(None);
        }

        match R::solve_single_from_indices(inliers, data, &self.config) {
            Ok(model) => Ok(model.filter(|model| R::is_valid(model) && S::is_valid(model))),
            Err(
                error @ (SolverError::InsufficientData { .. }
                | SolverError::InvalidSampleSize { .. }
                | SolverError::InvalidDataIndex { .. }
                | SolverError::InvalidInput { .. }),
            ) => Err(error.into()),
            Err(_) => Ok(None),
        }
    }
}
