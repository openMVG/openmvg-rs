// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Construction-time retained Schur execution planning.

use super::super::api::{RetainedSchurMatvec, SparseBundleAdjustmentError};
use super::parallel::{auto_parallel_threads, auto_self_calibration_threads};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::kernel::multi_view::sparse) enum RetainedProblemKind {
    Calibrated,
    SelfCalibrating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::kernel::multi_view::sparse) enum RetainedMatvecPlan {
    Fused,
    Recomputed,
    RecomputedParallel {
        threads: usize,
        logical_shards: usize,
    },
    PointMajorParallel {
        threads: usize,
        logical_shards: usize,
    },
}

impl RetainedMatvecPlan {
    /// Resolve validation and topology-dependent policy once, before any
    /// policy-specific index or scratch allocation.
    pub(in crate::kernel::multi_view::sparse) fn compile(
        requested: RetainedSchurMatvec,
        problem: RetainedProblemKind,
        observation_count: usize,
        point_major_worker_bytes: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        match requested {
            RetainedSchurMatvec::Fused => Ok(Self::Fused),
            RetainedSchurMatvec::Recomputed => Ok(Self::Recomputed),
            RetainedSchurMatvec::RecomputedParallel { threads: 0 }
            | RetainedSchurMatvec::PointMajorParallel { threads: 0 } => {
                Err(SparseBundleAdjustmentError::RetainedSchurThreadsMustBePositive)
            }
            RetainedSchurMatvec::RecomputedParallel { threads } => match problem {
                RetainedProblemKind::Calibrated => Ok(Self::RecomputedParallel {
                    threads,
                    logical_shards: threads,
                }),
                RetainedProblemKind::SelfCalibrating => {
                    Err(SparseBundleAdjustmentError::RecomputedSchurUnsupportedForSelfCalibration)
                }
            },
            RetainedSchurMatvec::PointMajorParallel { threads } => Ok(Self::PointMajorParallel {
                threads,
                logical_shards: match problem {
                    RetainedProblemKind::Calibrated => threads,
                    RetainedProblemKind::SelfCalibrating => {
                        super::parallel::deterministic_point_major_shards(
                            observation_count,
                            point_major_worker_bytes,
                        )
                        .unwrap_or(1)
                    }
                },
            }),
            RetainedSchurMatvec::Auto => {
                let threads = match problem {
                    RetainedProblemKind::Calibrated => {
                        auto_parallel_threads(observation_count, point_major_worker_bytes)
                    }
                    RetainedProblemKind::SelfCalibrating => {
                        auto_self_calibration_threads(observation_count, point_major_worker_bytes)
                    }
                };
                Ok(
                    threads.map_or(Self::Fused, |threads| Self::PointMajorParallel {
                        threads,
                        logical_shards: super::parallel::deterministic_point_major_shards(
                            observation_count,
                            point_major_worker_bytes,
                        )
                        .unwrap_or(1),
                    }),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_policy_compilation_is_validated_once() {
        let compile = |requested, problem| RetainedMatvecPlan::compile(requested, problem, 0, 0);
        assert_eq!(
            compile(RetainedSchurMatvec::Fused, RetainedProblemKind::Calibrated),
            Ok(RetainedMatvecPlan::Fused)
        );
        assert_eq!(
            compile(
                RetainedSchurMatvec::PointMajorParallel { threads: 4 },
                RetainedProblemKind::SelfCalibrating,
            ),
            Ok(RetainedMatvecPlan::PointMajorParallel {
                threads: 4,
                logical_shards: 1,
            })
        );
        assert_eq!(
            compile(
                RetainedSchurMatvec::PointMajorParallel { threads: 0 },
                RetainedProblemKind::Calibrated,
            ),
            Err(SparseBundleAdjustmentError::RetainedSchurThreadsMustBePositive)
        );
        assert_eq!(
            compile(
                RetainedSchurMatvec::RecomputedParallel { threads: 2 },
                RetainedProblemKind::SelfCalibrating,
            ),
            Err(SparseBundleAdjustmentError::RecomputedSchurUnsupportedForSelfCalibration)
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn self_calibration_auto_keeps_point_major_at_every_executor_width() {
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test pool is constructible");
            let plan = pool
                .install(|| {
                    RetainedMatvecPlan::compile(
                        RetainedSchurMatvec::Auto,
                        RetainedProblemKind::SelfCalibrating,
                        65_536,
                        std::mem::size_of::<f64>(),
                    )
                })
                .expect("self-calibration Auto policy is valid");
            assert_eq!(
                plan,
                RetainedMatvecPlan::PointMajorParallel {
                    threads,
                    logical_shards: 8,
                }
            );
        }
    }
}
