// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Certified rigid, biconnected translation-core extraction.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Translation core (largest_rigid_biconnected_translation_core)        │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: verified pair edges with relative poses, averaged rotations R,
//!        minimum edge-loss isotropy τ ∈ [0, 1/3]
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Biconnected blocks                                           │
//! └──────────────────────────────────────────────────────────────────────┘
//! Split the view graph at its articulation views; keep blocks of ≥ 3 edges.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Parallel rigidity per block                                  │
//! └──────────────────────────────────────────────────────────────────────┘
//! World baseline directions Rᵢᵀ tᵢⱼ must make the block structurally
//! parallel-rigid over all its views, so the directions fix every camera
//! position up to one translation and scale.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Edge-loss stability (τ > 0)                                  │
//! └──────────────────────────────────────────────────────────────────────┘
//! Per view, direction information Σ w (I − d dᵀ); its score is the least
//! λ_min / trace left after losing any one incident edge. While the weakest
//! view scores below τ, drop it and repeat Steps 1–3 on what remains.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Preferred block                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! Most views, then most edges, then the smallest view IDs.
//!   │
//!   ▼
//! Output: the core's edges and models, or an error when no block qualifies
//! ```

use super::*;
pub(super) fn largest_rigid_biconnected_translation_core(
    edges: &[PairEdge],
    models: &[GeometricModel],
    rotations: &RotationEstimates,
    min_edge_loss_isotropy: f64,
) -> Result<(Vec<PairEdge>, Vec<GeometricModel>)> {
    use openmvg_geometry::averaging::filtering::{
        DirectionConstraint, ParallelRigidityConfig, StructuralRigidityStatus,
        audit_graph_topology, audit_parallel_rigidity,
    };
    use std::collections::{BTreeMap, BTreeSet};

    if edges.len() != models.len() || edges.is_empty() {
        return Err(SfmError::InvalidInput(
            "translation-core selection requires aligned non-empty edges and models".to_owned(),
        ));
    }
    if !min_edge_loss_isotropy.is_finite() || !(0.0..=1.0 / 3.0).contains(&min_edge_loss_isotropy) {
        return Err(SfmError::InvalidInput(
            "translation-core edge-loss isotropy must be finite and in [0, 1/3]".to_owned(),
        ));
    }

    let initial_blocks = biconnected_blocks(edges, models)?;
    let mut stable_blocks = Vec::new();
    for (block_edges, block_models) in initial_blocks {
        if let Some(block) = stabilize_biconnected_block(
            block_edges,
            block_models,
            rotations,
            min_edge_loss_isotropy,
        )? {
            stable_blocks.push(block);
        }
    }
    let Some((block_edges, block_models)) = preferred_block(stable_blocks) else {
        return Err(SfmError::InvalidInput(if min_edge_loss_isotropy > 0.0 {
            "no translation core satisfies the requested edge-loss stability".to_owned()
        } else {
            "no generically parallel-rigid biconnected translation core exists".to_owned()
        }));
    };
    let selected_block = (block_edges, block_models);

    fn stabilize_biconnected_block(
        block_edges: Vec<PairEdge>,
        block_models: Vec<GeometricModel>,
        rotations: &RotationEstimates,
        min_edge_loss_isotropy: f64,
    ) -> Result<Option<(Vec<PairEdge>, Vec<GeometricModel>)>> {
        if !block_is_admissible(&block_edges, &block_models, rotations)? {
            return Ok(None);
        }
        if min_edge_loss_isotropy <= 0.0 {
            return Ok(Some((block_edges, block_models)));
        }
        let Some((weak_view, weak_score)) =
            weakest_translation_direction_support(&block_edges, &block_models, rotations)?
        else {
            return Ok(None);
        };
        if weak_score >= min_edge_loss_isotropy {
            return Ok(Some((block_edges, block_models)));
        }

        let mut next_edges = Vec::new();
        let mut next_models = Vec::new();
        for (edge, model) in block_edges.into_iter().zip(block_models) {
            if edge.view_i != weak_view && edge.view_j != weak_view {
                next_edges.push(edge);
                next_models.push(model);
            }
        }
        if next_edges.len() < 3 {
            return Ok(None);
        }

        let mut stable_children = Vec::new();
        for (child_edges, child_models) in biconnected_blocks(&next_edges, &next_models)? {
            if let Some(child) = stabilize_biconnected_block(
                child_edges,
                child_models,
                rotations,
                min_edge_loss_isotropy,
            )? {
                stable_children.push(child);
            }
        }
        Ok(preferred_block(stable_children))
    }

    fn biconnected_blocks(
        edges: &[PairEdge],
        models: &[GeometricModel],
    ) -> Result<Vec<(Vec<PairEdge>, Vec<GeometricModel>)>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let dense = DenseImageMap::from_edges(edges)?;
        let dense_edges = dense.remap_edges(edges)?;
        let endpoints = dense_edges
            .iter()
            .map(|edge| (edge.view_i.index(), edge.view_j.index()))
            .collect::<Vec<_>>();
        let topology = audit_graph_topology(dense.len(), &endpoints)
            .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
        let mut by_block = BTreeMap::<usize, Vec<usize>>::new();
        for (edge, &block) in topology.biconnected_block_by_edge().iter().enumerate() {
            by_block.entry(block).or_default().push(edge);
        }
        let candidates = by_block
            .into_values()
            .filter(|indices| indices.len() >= 3)
            .map(|indices| {
                (
                    indices.iter().map(|&index| edges[index]).collect(),
                    indices.iter().map(|&index| models[index].clone()).collect(),
                )
            })
            .collect::<Vec<_>>();
        Ok(candidates)
    }

    fn block_is_admissible(
        edges: &[PairEdge],
        models: &[GeometricModel],
        rotations: &RotationEstimates,
    ) -> Result<bool> {
        let dense = DenseImageMap::from_edges(edges)?;
        let remapped = dense.remap_edges(edges)?;
        let direction_edges = edges
            .iter()
            .zip(&remapped)
            .zip(models)
            .map(|((edge, remapped_edge), model)| {
                let relative = model.relative_pose().ok_or_else(|| {
                    SfmError::InvalidInput(format!(
                        "translation-core pair {} has no relative pose",
                        edge.pair
                    ))
                })?;
                let rotation = rotations.get(edge.view_i).ok_or_else(|| {
                    SfmError::InvalidInput(format!(
                        "translation-core view {} has no averaged rotation",
                        edge.view_i.index()
                    ))
                })?;
                Ok(DirectionConstraint::new(
                    remapped_edge.view_i.index(),
                    remapped_edge.view_j.index(),
                    rotation.inverse() * relative.baseline_direction_in_source().as_ref(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let audit = audit_parallel_rigidity(
            dense.len(),
            &direction_edges,
            ParallelRigidityConfig::default(),
        )
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
        Ok(audit.topology.candidate_covers_all_views()
            && matches!(audit.structural, StructuralRigidityStatus::Rigid { .. }))
    }

    fn preferred_block(
        mut blocks: Vec<(Vec<PairEdge>, Vec<GeometricModel>)>,
    ) -> Option<(Vec<PairEdge>, Vec<GeometricModel>)> {
        blocks.sort_by(|(left_edges, _), (right_edges, _)| {
            let left_views = block_views(left_edges);
            let right_views = block_views(right_edges);
            right_views
                .len()
                .cmp(&left_views.len())
                .then_with(|| right_edges.len().cmp(&left_edges.len()))
                .then_with(|| left_views.cmp(&right_views))
        });
        blocks.into_iter().next()
    }

    fn block_views(edges: &[PairEdge]) -> Vec<ViewId> {
        edges
            .iter()
            .flat_map(|edge| [edge.view_i, edge.view_j])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn weakest_translation_direction_support(
        edges: &[PairEdge],
        models: &[GeometricModel],
        rotations: &RotationEstimates,
    ) -> Result<Option<(ViewId, f64)>> {
        let mut contributions = BTreeMap::<ViewId, Vec<nalgebra::Matrix3<f64>>>::new();
        for (edge, model) in edges.iter().zip(models) {
            let relative = model.relative_pose().ok_or_else(|| {
                SfmError::InvalidInput(format!(
                    "translation-core pair {} has no relative pose",
                    edge.pair
                ))
            })?;
            let rotation = rotations.get(edge.view_i).ok_or_else(|| {
                SfmError::InvalidInput(format!(
                    "translation-core view {} has no averaged rotation",
                    edge.view_i.index()
                ))
            })?;
            let world_direction =
                rotation.inverse() * relative.baseline_direction_in_source().as_ref();
            let Some(direction) = world_direction.try_normalize(1e-12) else {
                continue;
            };
            let weight = edge.confidence.max(1e-6);
            let contribution =
                weight * (nalgebra::Matrix3::identity() - direction * direction.transpose());
            contributions
                .entry(edge.view_i)
                .or_default()
                .push(contribution);
            contributions
                .entry(edge.view_j)
                .or_default()
                .push(contribution);
        }
        Ok(contributions
            .into_iter()
            .map(|(view, contributions)| {
                let information = contributions
                    .iter()
                    .copied()
                    .sum::<nalgebra::Matrix3<f64>>();
                let score = contributions
                    .iter()
                    .map(|contribution| direction_information_isotropy(information - contribution))
                    .fold(f64::INFINITY, f64::min);
                (view, score)
            })
            .min_by(|(left_image, left_score), (right_image, right_score)| {
                left_score
                    .total_cmp(right_score)
                    .then_with(|| left_image.cmp(right_image))
            }))
    }

    fn direction_information_isotropy(information: nalgebra::Matrix3<f64>) -> f64 {
        let eigenvalues = information.symmetric_eigen().eigenvalues;
        let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let trace = eigenvalues.iter().sum::<f64>();
        if trace > 0.0 && minimum.is_finite() {
            (minimum / trace).max(0.0)
        } else {
            0.0
        }
    }

    Ok(selected_block)
}
