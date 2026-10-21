// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pair-proposal and matching contracts: [`PairProposer`], [`PairMatcher`], and the
//! [`PreparedPairMatcher`] state built once per matching stage.

use openmvg_core::RunSeed;

use crate::progress::SfmProgressSink;
use crate::two_view::PairMatches;
use crate::workspace::views::FeaturesView;
use crate::{Result, ViewId, ViewPair};
use openmvg_features::model::ExtractedFeatures;

/// Inputs passed to a [`PairMatcher`] or [`PreparedPairMatcher`] for one pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PairContext {
    /// Canonical pair being matched.
    pub pair: ViewPair,
    /// First endpoint (`pair.first()`, the smaller view ID); owns the first feature set.
    pub view_i: ViewId,
    /// Second endpoint (`pair.second()`); owns the second feature set.
    pub view_j: ViewId,
    /// Stage root used only to key shared per-image matcher state.
    /// Pair-local sampling must use [`Self::seed`].
    pub root_seed: RunSeed,
    /// Pair-local stream seed derived by the orchestration layer.
    pub seed: RunSeed,
}

/// Immutable inputs available while preparing a matcher for one matching stage.
///
/// The view and pair list are borrowed from the active [`crate::SfmWorkspace`],
/// so any prepared state borrowing them cannot outlive that workspace snapshot.
/// This gives indexed matchers a safe place to build one index per participating
/// image without maintaining a persistent cache that must be invalidated by hand.
#[derive(Clone, Copy)]
pub struct PairMatcherPreparation<'a> {
    features: FeaturesView<'a>,
    pairs: &'a [ViewPair],
    root_seed: RunSeed,
    progress: &'a dyn SfmProgressSink,
}

impl<'a> PairMatcherPreparation<'a> {
    pub(crate) const fn new(
        features: FeaturesView<'a>,
        pairs: &'a [ViewPair],
        root_seed: RunSeed,
        progress: &'a dyn SfmProgressSink,
    ) -> Self {
        Self {
            features,
            pairs,
            root_seed,
            progress,
        }
    }

    /// Read-only feature snapshot for the matching stage.
    pub const fn features(self) -> FeaturesView<'a> {
        self.features
    }

    /// Canonical endpoints for the selected pairs, in deterministic pair order.
    pub const fn pairs(self) -> &'a [ViewPair] {
        self.pairs
    }

    /// Root seed from which shared per-image matcher state must be derived.
    pub const fn root_seed(self) -> RunSeed {
        self.root_seed
    }

    /// Sink for the `SfmProgressStage::MatcherPreparation` stage, counted in prepared images.
    /// Report it only when preparation builds per-image state.
    pub const fn progress(self) -> &'a dyn SfmProgressSink {
        self.progress
    }
}

impl std::fmt::Debug for PairMatcherPreparation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairMatcherPreparation")
            .field("features", &self.features)
            .field("pairs", &self.pairs)
            .field("root_seed", &self.root_seed)
            .finish_non_exhaustive()
    }
}

/// Propose a sparse image-pair graph from already extracted features.
pub trait PairProposer: Send + Sync {
    /// Candidate pairs to match; the pipeline deduplicates them into the pair graph.
    fn propose_pairs(&self, features: FeaturesView<'_>) -> Result<Vec<ViewPair>>;
}

/// Matcher state prepared once for all selected pairs in one matching stage.
///
/// Implementations may borrow the immutable feature snapshot supplied during
/// preparation. The pipeline shares this object across pair workers, so query
/// operations must be thread-safe and must not depend on execution order. It
/// calls [`Self::match_pair`] only for pairs listed in the preparation context.
pub trait PreparedPairMatcher: Send + Sync {
    /// Match the features of `ctx.view_i` against those of `ctx.view_j`.
    fn match_pair(&self, ctx: &PairContext) -> Result<PairMatches>;
}

/// Matches descriptors between the two images of a pair.
pub trait PairMatcher: Send + Sync {
    /// Optionally prepare shared state before matching the selected pairs.
    ///
    /// The default is a no-op, preserving the direct per-pair path for brute
    /// force matchers and existing external implementations. Indexed backends
    /// can return an invocation-scoped cache that is built once per image. If
    /// this returns `Some`, the pipeline uses that prepared matcher for every
    /// selected pair and does not call [`Self::match_features`] in that stage.
    fn prepare<'a>(
        &'a self,
        _ctx: PairMatcherPreparation<'a>,
    ) -> Result<Option<Box<dyn PreparedPairMatcher + 'a>>> {
        Ok(None)
    }

    /// Match `fi` (features of `ctx.view_i`) against `fj` (features of `ctx.view_j`).
    fn match_features(
        &self,
        ctx: &PairContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
    ) -> Result<PairMatches>;
}
