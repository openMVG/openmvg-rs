// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Construction-time work and memory policy for deterministic direct Schur.

use crate::kernel::point_block::PackedPointBlock;

pub(super) const MAX_REDUCED_BLOCK_DIMENSION: usize = 6;
const PACKED_REDUCED_BLOCK_F64S: usize =
    MAX_REDUCED_BLOCK_DIMENSION * (MAX_REDUCED_BLOCK_DIMENSION + 1) / 2;
const PACKED_REDUCED_EDGE_F64S: usize = MAX_REDUCED_BLOCK_DIMENSION * MAX_REDUCED_BLOCK_DIMENSION;
const POINT_ASSEMBLY_F64S_PER_PARAMETER: usize = 6;
const COMPACT_REFERENCE_BYTES: usize = 2 * std::mem::size_of::<u32>();

const AUTO_MINIMUM_COUPLING_TERMS: usize = 1 << 13;
const AUTO_MINIMUM_POINT_OBSERVATIONS: usize = 1 << 9;
#[cfg(feature = "parallel")]
const AUTO_COUPLING_TERMS_PER_THREAD: usize = 1 << 16;
#[cfg(feature = "parallel")]
const AUTO_SMALL_WORK_MAXIMUM_THREADS: usize = 16;
#[cfg(feature = "parallel")]
const AUTO_LARGE_WORK_MAXIMUM_THREADS: usize = 32;
#[cfg(feature = "parallel")]
const AUTO_LARGE_WORK_COUPLING_TERMS: usize = 1 << 25;
#[cfg(feature = "parallel")]
const AUTO_LARGE_WORK_POINT_OBSERVATIONS: usize = 1 << 19;
const AUTO_MAXIMUM_AUXILIARY_BYTES: usize = 256 * 1024 * 1024;
pub(super) const LOGICAL_POINT_SHARD_OBSERVATIONS: usize = 1 << 12;
pub(super) const MAXIMUM_LOGICAL_POINT_SHARDS: usize = 16;

/// Column-owned worker count for `Auto`, or `None` for serial assembly.
///
/// Column-owned assembly is faster than serial even on one worker, so it is
/// selected for all but tiny problems whenever its buffers fit the memory cap.
pub(super) fn auto_direct_assembly_threads(
    coupling_terms: usize,
    point_observations: usize,
    mut auxiliary_bytes: impl FnMut(usize) -> usize,
) -> Option<usize> {
    if coupling_terms < AUTO_MINIMUM_COUPLING_TERMS
        && point_observations < AUTO_MINIMUM_POINT_OBSERVATIONS
    {
        return None;
    }
    let threads = auto_maximum_threads(coupling_terms, point_observations);
    (1..=threads)
        .rev()
        .find(|&candidate| auxiliary_bytes(candidate) <= AUTO_MAXIMUM_AUXILIARY_BYTES)
}

#[cfg(not(feature = "parallel"))]
fn auto_maximum_threads(_coupling_terms: usize, _point_observations: usize) -> usize {
    1
}

#[cfg(feature = "parallel")]
fn auto_maximum_threads(coupling_terms: usize, point_observations: usize) -> usize {
    let coupling_threads = coupling_terms
        .div_ceil(AUTO_COUPLING_TERMS_PER_THREAD)
        .max(1);
    let point_threads = point_observations
        .div_ceil(LOGICAL_POINT_SHARD_OBSERVATIONS)
        .max(1);
    let recommended = if coupling_terms >= AUTO_LARGE_WORK_COUPLING_TERMS
        || point_observations >= AUTO_LARGE_WORK_POINT_OBSERVATIONS
    {
        AUTO_LARGE_WORK_MAXIMUM_THREADS
    } else {
        AUTO_SMALL_WORK_MAXIMUM_THREADS
    };
    coupling_threads
        .max(point_threads)
        .min(recommended)
        .min(rayon::current_num_threads())
}

pub(super) fn estimated_calibrated_auxiliary_bytes(
    threads: usize,
    point_count: usize,
    point_observations: usize,
    active_block_references: usize,
    block_count: usize,
    dimension: usize,
) -> usize {
    estimated_auxiliary_bytes(
        threads,
        point_count,
        point_observations,
        active_block_references,
        block_count,
        0,
        dimension,
    )
}

pub(super) fn estimated_indexed_auxiliary_bytes(
    threads: usize,
    point_count: usize,
    point_observations: usize,
    active_block_references: usize,
    block_count: usize,
    base_edge_count: usize,
    dimension: usize,
) -> usize {
    estimated_auxiliary_bytes(
        threads,
        point_count,
        point_observations,
        active_block_references,
        block_count,
        base_edge_count,
        dimension,
    )
}

fn estimated_auxiliary_bytes(
    threads: usize,
    point_count: usize,
    point_observations: usize,
    active_block_references: usize,
    block_count: usize,
    base_edge_count: usize,
    dimension: usize,
) -> usize {
    let packed_block_bytes = PACKED_REDUCED_BLOCK_F64S * std::mem::size_of::<f64>();
    let packed_edge_bytes = PACKED_REDUCED_EDGE_F64S * std::mem::size_of::<f64>();
    let point_block_capacity = if active_block_references == 0 {
        0
    } else {
        active_block_references
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
    };
    let index_bytes = point_count
        .saturating_mul(std::mem::size_of::<PackedPointBlock>() + std::mem::size_of::<usize>())
        .saturating_add(
            active_block_references
                .saturating_add(point_block_capacity)
                .saturating_mul(COMPACT_REFERENCE_BYTES),
        )
        .saturating_add(
            block_count
                .saturating_add(1)
                .saturating_mul(2 * std::mem::size_of::<usize>()),
        );
    let global_base_bytes = block_count
        .saturating_mul(packed_block_bytes)
        .saturating_add(
            base_edge_count
                .saturating_mul(packed_edge_bytes + std::mem::size_of::<(usize, usize)>()),
        );
    let per_worker_bytes = block_count
        .saturating_mul(
            packed_block_bytes + std::mem::size_of::<bool>() + std::mem::size_of::<usize>(),
        )
        .saturating_add(base_edge_count.saturating_mul(
            packed_edge_bytes + std::mem::size_of::<bool>() + std::mem::size_of::<usize>(),
        ))
        .saturating_add(dimension.saturating_mul(
            POINT_ASSEMBLY_F64S_PER_PARAMETER * std::mem::size_of::<f64>()
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<bool>(),
        ));
    let logical_point_shards = if point_count == 0 {
        0
    } else {
        point_observations
            .div_ceil(LOGICAL_POINT_SHARD_OBSERVATIONS)
            .clamp(1, MAXIMUM_LOGICAL_POINT_SHARDS)
            .min(point_count)
    };
    let active_point_shards = logical_point_shards.min(threads);
    // Every materialization partition also owns one edge slot per block.
    let partition_bytes = threads
        .saturating_add(logical_point_shards)
        .saturating_add(2)
        .saturating_mul(std::mem::size_of::<usize>())
        .saturating_add(
            threads
                .saturating_mul(block_count)
                .saturating_mul(std::mem::size_of::<u32>()),
        );
    index_bytes
        .saturating_add(global_base_bytes)
        .saturating_add(partition_bytes)
        .saturating_add(per_worker_bytes.saturating_mul(active_point_shards))
}

#[cfg(all(test, not(feature = "parallel")))]
mod serial_build_tests {
    use super::*;

    #[test]
    fn auto_policy_uses_one_column_owned_worker_without_parallel_feature() {
        assert_eq!(auto_direct_assembly_threads(2_670, 256, |_| 0), None);
        assert_eq!(auto_direct_assembly_threads(0, 16_384, |_| 0), Some(1));
        assert_eq!(
            auto_direct_assembly_threads(1 << 26, 1 << 20, |_| 0),
            Some(1)
        );
        assert_eq!(
            auto_direct_assembly_threads(1 << 20, 1 << 16, |_| {
                AUTO_MAXIMUM_AUXILIARY_BYTES + 1
            }),
            None
        );
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    #[test]
    fn auto_policy_separates_tiny_indexed_coupling_and_point_work() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .expect("policy test pool is valid");
        pool.install(|| {
            assert_eq!(auto_direct_assembly_threads(2_670, 256, |_| 0), None);
            assert_eq!(auto_direct_assembly_threads(14_406, 128, |_| 0), Some(1));
            assert_eq!(auto_direct_assembly_threads(114_954, 1_024, |_| 0), Some(2));
            assert_eq!(auto_direct_assembly_threads(0, 16_384, |_| 0), Some(4));
            assert_eq!(
                auto_direct_assembly_threads(1 << 20, 1 << 16, |_| {
                    AUTO_MAXIMUM_AUXILIARY_BYTES + 1
                }),
                None
            );
            assert_eq!(
                auto_direct_assembly_threads(1 << 20, 1 << 16, |threads| {
                    if threads > 4 {
                        AUTO_MAXIMUM_AUXILIARY_BYTES + 1
                    } else {
                        0
                    }
                }),
                Some(4)
            );
        });
    }

    #[test]
    fn auto_policy_caps_small_and_large_work_by_recommendation() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(64)
            .build()
            .expect("policy test pool is valid");
        pool.install(|| {
            assert_eq!(
                auto_direct_assembly_threads(1 << 24, 1 << 18, |_| 0),
                Some(AUTO_SMALL_WORK_MAXIMUM_THREADS)
            );
            assert_eq!(
                auto_direct_assembly_threads(1 << 26, 1 << 20, |_| 0),
                Some(AUTO_LARGE_WORK_MAXIMUM_THREADS)
            );
        });
    }

    #[test]
    fn auxiliary_estimate_counts_only_active_point_shards() {
        let one_shard = estimated_calibrated_auxiliary_bytes(16, 1_024, 4_096, 4_096, 256, 1_536);
        let four_shards =
            estimated_calibrated_auxiliary_bytes(16, 1_024, 16_384, 4_096, 256, 1_536);
        let capped_shards =
            estimated_calibrated_auxiliary_bytes(16, 1_024, usize::MAX, 4_096, 256, 1_536);
        assert!(one_shard < four_shards);
        assert!(four_shards < capped_shards);
        assert_eq!(
            capped_shards,
            estimated_calibrated_auxiliary_bytes(
                16,
                1_024,
                MAXIMUM_LOGICAL_POINT_SHARDS * LOGICAL_POINT_SHARD_OBSERVATIONS,
                4_096,
                256,
                1_536,
            )
        );
    }
}
