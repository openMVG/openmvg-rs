// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-driven image-pair graph initialization.

use crate::backends::PairProposer;
use crate::workspace::pair_graph::PairGraph;
use crate::{Result, SfmError, SfmWorkspace};

/// Outcome of one transactional pair-proposal stage.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PairProposalReport {
    /// Pairs returned by the proposer, duplicates included.
    pub raw_proposals: usize,
    /// Distinct canonical pairs installed in the pair graph (at least 1).
    pub unique_pairs: usize,
}

/// Build and install a canonical pair graph without overwriting downstream state.
pub fn propose_pairs(
    workspace: &mut SfmWorkspace,
    proposer: &dyn PairProposer,
) -> Result<PairProposalReport> {
    workspace.ensure_pair_graph_uninitialized()?;
    if workspace.features_view().populated_count() != workspace.views().len() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "features for every image",
            method: "extract_features",
        });
    }

    // Proposal and graph validation happen before any workspace mutation.
    let proposed = proposer.propose_pairs(workspace.features_view())?;
    let raw_proposals = proposed.len();
    let graph = PairGraph::try_new(workspace.views().len(), proposed)?;
    graph.validate()?;
    let unique_pairs = graph.len();
    if unique_pairs == 0 {
        return Err(SfmError::InvalidInput(
            "pair proposer returned no usable image pairs".to_owned(),
        ));
    }
    workspace.install_pair_graph_before_matching(graph)?;

    Ok(PairProposalReport {
        raw_proposals,
        unique_pairs,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::model::ImageMeta;
    use crate::{Camera, IntrinsicId, ViewId, ViewPair};
    use openmvg_features::model::{DescriptorSet, ExtractedFeatures};

    struct StaticProposer {
        calls: AtomicUsize,
        pairs: Vec<ViewPair>,
    }

    impl PairProposer for StaticProposer {
        fn propose_pairs(
            &self,
            _features: crate::workspace::views::FeaturesView<'_>,
        ) -> Result<Vec<ViewPair>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.pairs.clone())
        }
    }

    fn workspace(view_count: usize) -> SfmWorkspace {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..view_count)
                    .map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        for image_index in 0..view_count {
            let features = ExtractedFeatures::try_new(
                Vec::new(),
                DescriptorSet::try_binary(Vec::new(), 256, 0).unwrap(),
            )
            .unwrap();
            workspace
                .feature_sets
                .set(image_index, features.feature_set())
                .unwrap();
            workspace.set_features(image_index, features).unwrap();
        }
        workspace
    }

    #[test]
    fn proposal_is_canonical_transactional_and_resizes_pair_stores() {
        let mut workspace = workspace(3);
        let proposer = StaticProposer {
            calls: AtomicUsize::new(0),
            pairs: vec![
                ViewPair::new(ViewId::new(2), ViewId::new(0)),
                ViewPair::new(ViewId::new(0), ViewId::new(2)),
                ViewPair::new(ViewId::new(2), ViewId::new(0)),
            ],
        };
        let report = propose_pairs(&mut workspace, &proposer).unwrap();
        assert_eq!(report.raw_proposals, 3);
        assert_eq!(report.unique_pairs, 1);
        assert_eq!(workspace.pair_graph().len(), 1);
        assert_eq!(workspace.match_store().slot_count(), 1);
        assert_eq!(workspace.geometric_model_store().slot_count(), 1);
        workspace.validate().unwrap();

        assert!(propose_pairs(&mut workspace, &proposer).is_err());
        assert_eq!(proposer.calls.load(Ordering::Relaxed), 1);
        assert_eq!(workspace.pair_graph().len(), 1);
    }

    #[test]
    fn invalid_proposal_does_not_mutate_the_workspace() {
        let mut workspace = workspace(2);
        let proposer = StaticProposer {
            calls: AtomicUsize::new(0),
            pairs: vec![ViewPair::new(ViewId::new(0), ViewId::new(9))],
        };
        assert!(propose_pairs(&mut workspace, &proposer).is_err());
        assert!(workspace.pair_graph().is_empty());
        assert_eq!(workspace.match_store().slot_count(), 0);
        assert_eq!(workspace.geometric_model_store().slot_count(), 0);
        assert_eq!(workspace.features_view().populated_count(), 2);
        workspace.validate().unwrap();
    }
}
