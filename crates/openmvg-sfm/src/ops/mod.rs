// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable operations on an [`SfmWorkspace`], shared by both engines.
//!
//! ```text
//! Front end:     features → pairs → matching → verification → (radial_calibration) → tracks
//!                frontend runs the sequence above
//! Graph:         view_graph conditioning, probes
//! Averaging:     averaging (rotations, translations)
//! Reconstruction: reconstruction (triangulation, bundle adjustment, cleanup), track_scale
//! ```
//!
//! The engines in `pipelines` decide when each operation runs; `rollback` holds the snapshots
//! they restore when they reject a step.

use crate::workspace::selector::{PairSelector, ViewSelector};
use crate::{PairId, Result, SfmError, SfmWorkspace, ViewId};

pub mod averaging;
// Front-end stages run only through `image-io` preparation and the unstable API.
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub(crate) mod features;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub mod frontend;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub(crate) mod matching;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub(crate) mod pairs;
pub mod probes;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub mod radial_calibration;
pub mod reconstruction;
pub(crate) mod rollback;
pub mod track_scale;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub(crate) mod tracks;
#[cfg_attr(
    not(any(feature = "image-io", feature = "unstable-internals")),
    allow(dead_code)
)]
pub(crate) mod verification;
pub mod view_graph;

pub(crate) fn resolve_views(
    workspace: &SfmWorkspace,
    selector: ViewSelector,
) -> Result<Vec<ViewId>> {
    match selector {
        ViewSelector::All => Ok((0..workspace.views().len()).map(ViewId::new).collect()),
        ViewSelector::Missing => Ok(workspace
            .feature_store()
            .iter_missing()
            .filter(|index| *index < workspace.views().len())
            .map(ViewId::new)
            .collect()),
        ViewSelector::Ids(ids) => {
            canonicalize_ids(ids, workspace.views().len(), ViewId::kind(), ViewId::index)
        }
    }
}

/// Available system memory in GiB, read from `/proc/meminfo` (`MemAvailable`).
/// `None` on platforms without it (e.g. macOS) — callers then fall back to a
/// core-count-only bound.
#[cfg(feature = "parallel")]
fn available_memory_gib() -> Option<f64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: f64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kib / 1024.0 / 1024.0);
        }
    }
    None
}

/// Number of worker threads to use for a memory-sensitive parallel stage.
///
/// Bounded by three limits so a wide fan-out cannot OOM the machine:
/// 1. the active Rayon pool, then `hard_cap` (a stage-specific ceiling);
/// 2. available RAM ÷ `per_worker_gib` (peak memory each concurrent task may hold);
/// 3. the number of tasks (never spawn idle threads).
///
/// An explicit caller-owned worker ceiling overrides limits 1–2 while remaining
/// clamped to the task count.
#[cfg(feature = "parallel")]
pub(crate) fn bounded_workers(
    num_tasks: usize,
    per_worker_gib: f64,
    hard_cap: usize,
    explicit_workers: Option<usize>,
) -> usize {
    bounded_workers_from_limits(
        num_tasks,
        per_worker_gib,
        hard_cap,
        rayon::current_num_threads(),
        available_memory_gib(),
        explicit_workers,
    )
}

#[cfg(feature = "parallel")]
fn bounded_workers_from_limits(
    num_tasks: usize,
    per_worker_gib: f64,
    hard_cap: usize,
    rayon_workers: usize,
    available_memory_gib: Option<f64>,
    explicit_workers: Option<usize>,
) -> usize {
    let tasks = num_tasks.max(1);
    if let Some(workers) = explicit_workers {
        return workers.min(tasks);
    }
    let mut workers = rayon_workers.max(1).min(hard_cap);
    if per_worker_gib > 0.0
        && let Some(gib) = available_memory_gib
    {
        let mem_workers = ((gib / per_worker_gib).floor() as usize).max(1);
        workers = workers.min(mem_workers);
    }
    workers.min(tasks).max(1)
}

pub(crate) fn resolve_pairs(
    workspace: &SfmWorkspace,
    selector: PairSelector,
) -> Result<Vec<PairId>> {
    match selector {
        PairSelector::All => Ok((0..workspace.pair_graph().len()).map(PairId::new).collect()),
        PairSelector::Active => Ok(workspace.pair_graph().active_pair_ids().collect()),
        PairSelector::Pairs(mut pairs) => {
            pairs.sort_unstable();
            pairs.dedup();
            pairs
                .into_iter()
                .map(|pair| {
                    workspace
                        .pair_graph()
                        .pair_id(pair.first(), pair.second())
                        .ok_or_else(|| {
                            SfmError::InvalidInput(format!(
                                "pair {pair} is not present in this workspace"
                            ))
                        })
                })
                .collect()
        }
        PairSelector::Unmatched => Ok(workspace
            .pair_graph()
            .active_pair_ids()
            .filter(|&pair_id| workspace.matches_view().get_by_id(pair_id).is_none())
            .collect()),
        PairSelector::Unverified => Ok(workspace
            .pair_graph()
            .active_pair_ids()
            .filter(|&pair_id| {
                workspace
                    .geometric_models_view()
                    .get_by_id(pair_id)
                    .is_none()
            })
            .collect()),
    }
}

fn canonicalize_ids<T: Copy + Ord>(
    mut ids: Vec<T>,
    domain_len: usize,
    kind: &'static str,
    index: impl Fn(T) -> usize,
) -> Result<Vec<T>> {
    ids.sort_unstable();
    ids.dedup();
    if let Some(invalid) = ids.iter().copied().find(|&id| index(id) >= domain_len) {
        return Err(SfmError::InvalidId {
            kind,
            index: index(invalid),
        });
    }
    Ok(ids)
}

#[cfg(test)]
mod selector_tests {
    use super::*;
    use crate::model::ImageMeta;
    use crate::{Camera, IntrinsicId, ViewPair};

    fn workspace() -> SfmWorkspace {
        SfmWorkspace::builder()
            .image_meta([
                ImageMeta::external_id("a", 64, 48),
                ImageMeta::external_id("b", 64, 48),
                ImageMeta::external_id("c", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap()
    }

    #[test]
    fn explicit_selectors_are_canonical_sets() {
        let workspace = workspace();

        assert_eq!(
            resolve_views(
                &workspace,
                ViewSelector::Ids(vec![ViewId::new(2), ViewId::new(0), ViewId::new(2)])
            )
            .unwrap(),
            [ViewId::new(0), ViewId::new(2)]
        );
        assert_eq!(
            resolve_pairs(
                &workspace,
                PairSelector::Pairs(vec![
                    ViewPair::new(ViewId::new(1), ViewId::new(2)),
                    ViewPair::new(ViewId::new(0), ViewId::new(1)),
                    ViewPair::new(ViewId::new(1), ViewId::new(2)),
                ])
            )
            .unwrap(),
            [PairId::new(0), PairId::new(2)]
        );
    }

    #[test]
    fn explicit_selectors_reject_ids_outside_the_workspace_domain() {
        let workspace = workspace();

        assert!(matches!(
            resolve_views(&workspace, ViewSelector::Ids(vec![ViewId::new(usize::MAX)])),
            Err(SfmError::InvalidId { kind: "view", .. })
        ));
        assert!(matches!(
            resolve_pairs(
                &workspace,
                PairSelector::Pairs(vec![ViewPair::new(ViewId::new(0), ViewId::new(4))])
            ),
            Err(SfmError::InvalidInput(_))
        ));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn worker_bound_combines_rayon_memory_stage_and_task_limits() {
        assert_eq!(bounded_workers_from_limits(100, 0.0, 16, 6, None, None), 6);
        assert_eq!(bounded_workers_from_limits(100, 0.0, 4, 6, None, None), 4);
        assert_eq!(
            bounded_workers_from_limits(100, 2.0, 16, 6, Some(5.0), None),
            2
        );
        assert_eq!(bounded_workers_from_limits(3, 0.0, 16, 6, None, None), 3);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn explicit_worker_override_retains_task_clamp_semantics() {
        assert_eq!(
            bounded_workers_from_limits(7, 4.0, 2, 1, Some(1.0), Some(5)),
            5
        );
        assert_eq!(
            bounded_workers_from_limits(3, 4.0, 2, 1, Some(1.0), Some(5)),
            3
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn worker_bound_honors_the_active_rayon_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test pool builds");

        assert_eq!(pool.install(|| bounded_workers(100, 0.0, 16, None)), 2);
    }
}
