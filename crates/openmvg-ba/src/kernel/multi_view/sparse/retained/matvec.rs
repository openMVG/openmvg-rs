// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained Schur operator dispatch and policy-specific scratch selection.

use super::operator::apply_reduced_system;
use super::point_major::PointMajorParallelWorkspace;
use super::recomputed::{RecomputedMatvecWorkspace, RecomputedParallelWorkspace};
use super::*;

pub(super) enum SchurMatvecWorkspace {
    Fused,
    Recomputed(RecomputedMatvecWorkspace),
    RecomputedParallel(RecomputedParallelWorkspace),
    PointMajorParallel(PointMajorParallelWorkspace),
}

impl SchurMatvecWorkspace {
    pub(super) fn try_new(
        policy: RetainedSchurMatvec,
        logical_shards_override: Option<usize>,
        model: ObservationModel<'_>,
        camera_count: usize,
        dimension: usize,
    ) -> Result<Self, SparseBundleAdjustmentError> {
        let worker_bytes = calibrated_point_major_worker_bytes(camera_count, dimension);
        match RetainedMatvecPlan::compile(
            policy,
            RetainedProblemKind::Calibrated,
            model.observation_count(),
            worker_bytes,
        )? {
            RetainedMatvecPlan::Fused => Ok(Self::Fused),
            RetainedMatvecPlan::Recomputed => {
                Ok(Self::Recomputed(RecomputedMatvecWorkspace::new(dimension)?))
            }
            RetainedMatvecPlan::RecomputedParallel {
                threads,
                logical_shards,
            } => Ok(Self::RecomputedParallel(
                RecomputedParallelWorkspace::try_new(
                    threads,
                    logical_shards_override.unwrap_or(logical_shards),
                    camera_count,
                    dimension,
                    model.offsets(),
                )?,
            )),
            RetainedMatvecPlan::PointMajorParallel {
                threads,
                logical_shards,
            } => Ok(Self::PointMajorParallel(
                PointMajorParallelWorkspace::try_new(
                    threads,
                    logical_shards,
                    camera_count,
                    dimension,
                    model.offsets(),
                )?,
            )),
        }
    }

    pub(super) const fn parallel_threads(&self) -> Option<usize> {
        match self {
            Self::Fused | Self::Recomputed(_) => None,
            Self::RecomputedParallel(workspace) => Some(workspace.shared.threads),
            Self::PointMajorParallel(workspace) => Some(workspace.threads),
        }
    }

    pub(super) fn retained_bytes(&self) -> usize {
        match self {
            Self::Fused => 0,
            Self::Recomputed(workspace) => workspace.retained_bytes(),
            Self::RecomputedParallel(workspace) => workspace.retained_bytes(),
            Self::PointMajorParallel(workspace) => workspace.retained_bytes(),
        }
    }

    #[cfg(feature = "parallel")]
    pub(super) fn install_parallel<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> Option<R> {
        match self {
            Self::Fused | Self::Recomputed(_) => None,
            Self::RecomputedParallel(workspace) => Some(workspace.shared.pool.install(operation)),
            Self::PointMajorParallel(workspace) => Some(workspace.pool.install(operation)),
        }
    }

    pub(super) fn apply_reduced_system(
        &mut self,
        context: SchurOperatorContext<'_>,
        sparse: &SparseWorkspaceCore,
        camera_blocks: &[PackedCameraBlock],
        point_hessians: &[PackedPointBlock],
        input: &[f64],
        output: &mut [f64],
    ) {
        if input.is_empty() {
            output.fill(0.0);
            return;
        }
        match self {
            Self::Fused => apply_reduced_system(
                context.model,
                sparse,
                camera_blocks,
                point_hessians,
                input,
                output,
            ),
            Self::Recomputed(workspace) => match context.recomputed {
                Some(context) => workspace.apply(
                    context,
                    sparse,
                    Some(camera_blocks),
                    point_hessians,
                    input,
                    output,
                    -1.0,
                ),
                None => output.fill(f64::NAN),
            },
            Self::RecomputedParallel(workspace) => match context.recomputed {
                Some(context) => workspace.apply(
                    context,
                    sparse,
                    Some(camera_blocks),
                    point_hessians,
                    input,
                    output,
                    -1.0,
                ),
                None => output.fill(f64::NAN),
            },
            Self::PointMajorParallel(workspace) => workspace.apply(
                context.model,
                sparse,
                Some(camera_blocks),
                point_hessians,
                input,
                output,
                -1.0,
            ),
        }
    }
}

pub(super) fn calibrated_point_major_worker_bytes(camera_count: usize, dimension: usize) -> usize {
    let vector_bytes = dimension.saturating_mul(
        7 * std::mem::size_of::<f64>() + std::mem::size_of::<usize>() + std::mem::size_of::<bool>(),
    );
    vector_bytes
        .saturating_add(camera_count.saturating_mul(std::mem::size_of::<PackedCameraBlock>()))
}
