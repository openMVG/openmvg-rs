// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-fitting contracts shared by solver families and robust estimation.
//!
//! Re-exported from [`crate::robust`] so a solver defined outside this crate can
//! run under [`crate::robust::classic::ClassicConfig::estimate`].

use crate::{SolverError, SolverResult};

/// A minimal-sample model fitter: data type, model type, sample size and residual.
pub trait Solver {
    /// One observation, for example a pixel correspondence.
    type Datum: Clone;
    /// The fitted model, for example a fundamental matrix.
    type Model: Clone;
    /// Per-call solver configuration; `()` when the solver has none.
    type Config;
    /// Unit of [`Solver::residual`]: one of the `openmvg_core` unit markers
    /// ([`Pixels`], [`NormalizedImage`], [`Radians`], [`WorldUnits`], [`Meters`], [`Unitless`]).
    /// [`ClassicConfig::estimate`] only accepts a threshold in this unit.
    ///
    /// [`Pixels`]: openmvg_core::Pixels
    /// [`NormalizedImage`]: openmvg_core::NormalizedImage
    /// [`Radians`]: openmvg_core::Radians
    /// [`WorldUnits`]: openmvg_core::WorldUnits
    /// [`Meters`]: openmvg_core::Meters
    /// [`Unitless`]: openmvg_core::Unitless
    /// [`ClassicConfig::estimate`]: crate::robust::classic::ClassicConfig::estimate
    type Unit;

    /// Number of data the solver needs for one minimal solve.
    fn min_sample_size() -> usize;

    /// Unsquared residual of one datum against a model, in [`Solver::Unit`].
    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64;

    /// Squared residual used for scoring; override when it is cheaper than
    /// squaring [`Solver::residual`].
    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        let residual = Self::residual(model, datum);
        residual * residual
    }

    /// Whether a model satisfies the solver's geometric invariants; invalid
    /// models are discarded before scoring. Accepts every model by default.
    fn is_valid(model: &Self::Model) -> bool {
        let _ = model;
        true
    }
}

/// A solver that yields at most one model per sample.
///
/// Every such solver is a [`MultiSolutionSolver`] through a blanket impl.
pub trait SingleSolutionSolver: Solver {
    /// Fit at most one model to a minimal sample.
    fn solve_single(
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>>;

    /// Fit the sample `indices` selects from `data`; override to avoid the copy.
    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        let sample = clone_indexed_sample(indices, data)?;
        Self::solve_single(&sample, config)
    }
}

/// A solver that may yield several candidate models per sample, the contract
/// robust estimation drives.
pub trait MultiSolutionSolver: Solver {
    /// Fit every candidate model to a minimal sample.
    fn solve(data: &[Self::Datum], config: &Self::Config) -> SolverResult<Vec<Self::Model>>;

    /// Fit the sample `indices` selects from `data`; override to avoid the copy.
    fn solve_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Vec<Self::Model>> {
        let sample = clone_indexed_sample(indices, data)?;
        Self::solve(&sample, config)
    }

    /// Pass each candidate for the sample `indices` selects to `emit` and return
    /// their count; override to avoid collecting the candidates.
    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
        mut emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        let models = Self::solve_from_indices(indices, data, config)?;
        let count = models.len();
        for model in models {
            emit(model);
        }
        Ok(count)
    }
}

fn clone_indexed_sample<T: Clone>(indices: &[usize], data: &[T]) -> SolverResult<Vec<T>> {
    indices
        .iter()
        .map(|&index| {
            data.get(index)
                .cloned()
                .ok_or(SolverError::InvalidDataIndex {
                    index,
                    data_len: data.len(),
                })
        })
        .collect()
}

impl<S: SingleSolutionSolver> MultiSolutionSolver for S {
    fn solve(data: &[Self::Datum], config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        S::solve_single(data, config).map(|model| model.into_iter().collect())
    }

    fn solve_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
    ) -> SolverResult<Vec<Self::Model>> {
        S::solve_single_from_indices(indices, data, config).map(|model| model.into_iter().collect())
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        config: &Self::Config,
        mut emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        match S::solve_single_from_indices(indices, data, config)? {
            Some(model) => {
                emit(model);
                Ok(1)
            }
            None => Ok(0),
        }
    }
}

/// Generate the public inherent solver surface: `min_sample_size`, `residual`,
/// `is_valid` and `solve`. Indexed, single-solution and squared-residual
/// variants live on the [`Solver`], [`SingleSolutionSolver`] and
/// [`MultiSolutionSolver`] traits that robust estimation drives.
macro_rules! expose_solver_api {
    ($solver:ty, $datum:ty, $model:ty) => {
        $crate::solver_traits::expose_solver_api!(@common $solver, $datum, $model);

        impl $solver {
            /// Solve from a checked correspondence slice.
            pub fn solve(data: &[$datum]) -> crate::SolverResult<Vec<$model>> {
                <Self as $crate::solver_traits::MultiSolutionSolver>::solve(data, &())
            }
        }
    };
    ($solver:ty, $datum:ty, $model:ty, $config:ty) => {
        $crate::solver_traits::expose_solver_api!(@common $solver, $datum, $model);

        impl $solver {
            /// Solve from a checked correspondence slice.
            pub fn solve(data: &[$datum], config: &$config) -> crate::SolverResult<Vec<$model>> {
                <Self as $crate::solver_traits::MultiSolutionSolver>::solve(data, config)
            }
        }
    };
    (@common $solver:ty, $datum:ty, $model:ty) => {
        impl $solver {
            /// Return the solver's minimum correspondence count.
            #[must_use]
            pub fn min_sample_size() -> usize {
                <Self as $crate::solver_traits::Solver>::min_sample_size()
            }

            /// Evaluate the model's unsquared residual for one correspondence.
            pub fn residual(model: &$model, datum: &$datum) -> f64 {
                <Self as $crate::solver_traits::Solver>::residual(model, datum)
            }

            /// Validate the model's geometric invariants.
            #[must_use]
            pub fn is_valid(model: &$model) -> bool {
                <Self as $crate::solver_traits::Solver>::is_valid(model)
            }
        }
    };
}

/// Public surface of a solver whose crate-private model is a solver pose:
/// the same four methods as [`expose_solver_api`], converted to the canonical
/// [`openmvg_core::model::WorldToCameraPose`].
macro_rules! expose_pose_solver_api {
    ($solver:ty, $datum:ty, $config:ty) => {
        impl $solver {
            /// Return the solver's minimum correspondence count.
            #[must_use]
            pub fn min_sample_size() -> usize {
                <Self as $crate::solver_traits::Solver>::min_sample_size()
            }

            /// Evaluate the model's unsquared residual for one correspondence.
            pub fn residual(model: &openmvg_core::model::WorldToCameraPose, datum: &$datum) -> f64 {
                let model = $crate::pose::world_to_camera_from_camera_pose(model);
                <Self as $crate::solver_traits::Solver>::residual(&model, datum)
            }

            /// Validate the model's geometric invariants.
            #[must_use]
            pub fn is_valid(model: &openmvg_core::model::WorldToCameraPose) -> bool {
                let model = $crate::pose::world_to_camera_from_camera_pose(model);
                <Self as $crate::solver_traits::Solver>::is_valid(&model)
            }

            /// Solve from a checked correspondence slice.
            pub fn solve(
                data: &[$datum],
                config: &$config,
            ) -> crate::SolverResult<Vec<openmvg_core::model::WorldToCameraPose>> {
                <Self as $crate::solver_traits::MultiSolutionSolver>::solve(data, config)?
                    .into_iter()
                    .map(|pose| {
                        $crate::pose::camera_pose_from_world_to_camera(&pose).map_err(|error| {
                            crate::SolverError::Numerical {
                                reason: error.to_string(),
                            }
                        })
                    })
                    .collect()
            }
        }
    };
}

pub(crate) use {expose_pose_solver_api, expose_solver_api};
