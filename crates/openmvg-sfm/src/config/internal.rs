// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configs reachable only from the unstable API.
//!
//! `unstable::config` re-exports them publicly under `unstable-internals`; stable runs use their
//! defaults.

use openmvg_core::RunSeed;
use std::num::NonZeroUsize;

/// Policy applied while assembling a mutable SfM workspace: validation, root seed, parallel width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceBuildConfig {
    validate_on_build: bool,
    #[serde(default)]
    root_seed: RunSeed,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_parallel_workers: Option<NonZeroUsize>,
}

config_accessors!(WorkspaceBuildConfig {
    /// Validate the workspace invariants when it is built.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy validate_on_build, with_validate_on_build: bool;
    /// Root of the deterministic seed tree used by every randomized stage.
    copy root_seed, with_root_seed: RunSeed;
    /// Optional caller-owned ceiling for memory-sensitive parallel stages.
    ///
    /// When absent, stages derive their width from the active Rayon pool,
    /// available memory, their own safety ceiling, and the amount of work.
    copy max_parallel_workers, with_max_parallel_workers: Option<NonZeroUsize>;
});

impl Default for WorkspaceBuildConfig {
    fn default() -> Self {
        Self {
            validate_on_build: true,
            root_seed: crate::determinism::DEFAULT_ROOT_SEED,
            max_parallel_workers: None,
        }
    }
}

/// Translation graph used after rotation averaging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalTranslationScope {
    /// Solve camera centers on the complete cleaned connected view graph.
    AllActiveEdges,
    /// Solve only the largest generically parallel-rigid vertex-biconnected translation block.
    /// Articulation/leaf lobes remain unposed and can be handled by an extension engine.
    LargestRigidBiconnectedCore,
}
