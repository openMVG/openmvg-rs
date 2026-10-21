// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Worker-pool construction and topology-based parallel scheduling.

#[cfg(feature = "parallel")]
use super::super::api::SparseBundleAdjustmentError;

#[cfg(feature = "parallel")]
const AUTO_PARALLEL_MINIMUM_OBSERVATIONS: usize = 32_768;
const AUTO_OBSERVATIONS_PER_THREAD: usize = 8_192;
pub(crate) const AUTO_MAXIMUM_THREADS: usize = 32;
pub(crate) const AUTO_MAXIMUM_PARALLEL_AUXILIARY_BYTES: usize = 128 * 1024 * 1024;

/// Topology-only shard count for deterministic point-major reductions.
///
/// This excludes the active executor width, because changing the number
/// of numerical shards changes the grouping of floating-point additions.
pub(crate) fn deterministic_point_major_shards(
    observation_count: usize,
    shard_bytes: usize,
) -> Option<usize> {
    let memory_limited = AUTO_MAXIMUM_PARALLEL_AUXILIARY_BYTES
        .checked_div(shard_bytes)
        .unwrap_or(AUTO_MAXIMUM_THREADS);
    let useful = observation_count
        .div_ceil(AUTO_OBSERVATIONS_PER_THREAD)
        .clamp(1, AUTO_MAXIMUM_THREADS)
        .min(memory_limited);
    (useful >= 1).then_some(useful)
}

#[cfg(feature = "parallel")]
pub(crate) fn try_build_retained_schur_pool(
    threads: usize,
) -> Result<rayon::ThreadPool, SparseBundleAdjustmentError> {
    super::super::parallel::try_build_pool(threads, "openmvg-ba-schur")
        .map_err(|_| SparseBundleAdjustmentError::RetainedSchurThreadPoolBuild)
}

pub(crate) fn auto_parallel_threads(
    observation_count: usize,
    worker_bytes: usize,
) -> Option<usize> {
    #[cfg(feature = "parallel")]
    {
        if observation_count < AUTO_PARALLEL_MINIMUM_OBSERVATIONS {
            return None;
        }
        let shards = deterministic_point_major_shards(observation_count, worker_bytes)?;
        if shards < 2 {
            return None;
        }
        Some(shards.min(auto_available_threads()).max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (observation_count, worker_bytes);
        None
    }
}

/// Resolve self-calibrating Auto without changing its arithmetic tree at 1T.
///
/// Eligible problems use the point-major operator even with a single
/// execution thread; its numerical shards remain topology-defined.
pub(crate) fn auto_self_calibration_threads(
    observation_count: usize,
    worker_bytes: usize,
) -> Option<usize> {
    #[cfg(feature = "parallel")]
    {
        if observation_count < AUTO_PARALLEL_MINIMUM_OBSERVATIONS {
            return None;
        }
        deterministic_point_major_shards(observation_count, worker_bytes)
            .map(|shards| shards.min(auto_available_threads()).max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (observation_count, worker_bytes);
        None
    }
}

#[cfg(feature = "parallel")]
pub(crate) fn auto_available_threads() -> usize {
    rayon::current_num_threads()
}

/// Partition prefix-summed work into nonempty contiguous entity ranges.
/// Contiguity keeps output slices disjoint while balancing highly uneven BAL
/// track and per-camera observation counts.
pub(crate) fn balanced_contiguous_ranges(
    offsets: &[usize],
    requested_threads: usize,
) -> Vec<usize> {
    let entity_count = offsets.len().saturating_sub(1);
    if entity_count == 0 {
        return vec![0];
    }
    let workers = requested_threads.max(1).min(entity_count);
    let total = offsets.last().copied().unwrap_or(0);
    let mut ranges = Vec::with_capacity(workers + 1);
    ranges.push(0);
    let mut previous = 0;
    for partition in 1..workers {
        let minimum = previous + 1;
        let maximum = entity_count - (workers - partition);
        let boundary = if total == 0 {
            entity_count * partition / workers
        } else {
            let target = ((total as u128 * partition as u128) / workers as u128) as usize;
            offsets.partition_point(|&offset| offset < target)
        }
        .clamp(minimum, maximum);
        ranges.push(boundary);
        previous = boundary;
    }
    ranges.push(entity_count);
    ranges
}
