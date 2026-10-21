// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Selectors for the views or pairs a stage processes.

use crate::{ViewId, ViewPair};

/// Selects views for stage execution.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # {
/// use openmvg_sfm::unstable::ViewSelector;
/// use openmvg_sfm::ViewId;
///
/// let selector = ViewSelector::Ids(vec![ViewId::new(0)]);
/// assert!(matches!(selector, ViewSelector::Ids(_)));
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewSelector {
    /// Every view in the workspace.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    All,
    /// Explicit view IDs. Resolution validates, sorts, and deduplicates them.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    Ids(Vec<ViewId>),
    /// Views that have no stored features yet.
    Missing,
}

/// Selects view pairs for matching or verification.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # {
/// use openmvg_sfm::unstable::PairSelector;
///
/// let selector = PairSelector::Active;
/// assert_eq!(selector, PairSelector::Active);
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairSelector {
    /// Every pair in the pair graph, active or deactivated.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    All,
    /// Pairs currently active in the pair graph.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    Active,
    /// Explicit stable view pairs. Resolution validates, sorts, and deduplicates them.
    Pairs(Vec<ViewPair>),
    /// Active pairs that have **no matches yet**.
    ///
    /// The idempotent counterpart of [`ViewSelector::Missing`]: on a fresh workspace this
    /// is exactly [`PairSelector::Active`], but on a workspace whose front-end has already
    /// run it selects nothing, so the expensive matching stage is not repeated. This lets
    /// one front-end result be reused by several reconstruction engines.
    Unmatched,
    /// Active pairs that have **no geometric model yet** (i.e. have not been verified).
    ///
    /// Same idempotence rationale as [`PairSelector::Unmatched`]. Pairs that verification
    /// rejects are deactivated, so they are not re-selected either.
    Unverified,
}
