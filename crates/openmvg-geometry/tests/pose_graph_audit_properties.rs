// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_geometry::averaging::filtering::audit_graph_topology;
use proptest::prelude::*;

fn simple_edges(num_views: usize, bits: &[bool]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let mut bit = 0usize;
    for view_i in 0..num_views {
        for view_j in (view_i + 1)..num_views {
            if bits.get(bit).copied().unwrap_or(false) {
                edges.push((view_i, view_j));
            }
            bit += 1;
        }
    }
    edges
}

fn component_count(
    num_views: usize,
    edges: &[(usize, usize)],
    removed_edge: Option<usize>,
    removed_view: Option<usize>,
) -> usize {
    let mut adjacency = vec![Vec::new(); num_views];
    for (index, &(view_i, view_j)) in edges.iter().enumerate() {
        if removed_edge == Some(index)
            || removed_view == Some(view_i)
            || removed_view == Some(view_j)
        {
            continue;
        }
        adjacency[view_i].push(view_j);
        adjacency[view_j].push(view_i);
    }
    let mut visited = vec![false; num_views];
    if let Some(view) = removed_view {
        visited[view] = true;
    }
    let mut components = 0usize;
    for start in 0..num_views {
        if visited[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut stack = vec![start];
        while let Some(view) = stack.pop() {
            for &neighbor in &adjacency[view] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    components
}

fn canonical_bridge_endpoints(
    edges: &[(usize, usize)],
    bridge_edges: &[usize],
) -> Vec<(usize, usize)> {
    let mut endpoints: Vec<_> = bridge_edges
        .iter()
        .map(|&edge| {
            let (view_i, view_j) = edges[edge];
            (view_i.min(view_j), view_i.max(view_j))
        })
        .collect();
    endpoints.sort_unstable();
    endpoints
}

fn canonical_blocks(
    edges: &[(usize, usize)],
    block_by_edge: &[usize],
    block_count: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut blocks = vec![Vec::new(); block_count];
    for (&(view_i, view_j), &block) in edges.iter().zip(block_by_edge) {
        blocks[block].push((view_i.min(view_j), view_i.max(view_j)));
    }
    for block in &mut blocks {
        block.sort_unstable();
    }
    blocks.sort_unstable();
    blocks
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn topology_audit_matches_removal_oracles_and_is_edge_order_invariant(
        num_views in 1usize..=8,
        bits in proptest::collection::vec(any::<bool>(), 0..=28),
        order_keys in proptest::collection::vec(any::<u16>(), 0..=28),
        endpoint_flips in proptest::collection::vec(any::<bool>(), 0..=28),
    ) {
        let edges = simple_edges(num_views, &bits);
        let audit = audit_graph_topology(num_views, &edges).unwrap();
        let base_components = component_count(num_views, &edges, None, None);
        prop_assert_eq!(audit.connected_component_count(), base_components);
        prop_assert_eq!(audit.connected_component_by_view().len(), num_views);
        prop_assert!(audit.connected_component_by_view()
            .iter()
            .all(|&component| component < audit.connected_component_count()));
        let mut observed_components = audit.connected_component_by_view().to_vec();
        observed_components.sort_unstable();
        observed_components.dedup();
        prop_assert_eq!(observed_components.len(), audit.connected_component_count());
        for &(view_i, view_j) in &edges {
            prop_assert_eq!(
                audit.connected_component_by_view()[view_i],
                audit.connected_component_by_view()[view_j]
            );
        }
        prop_assert_eq!(audit.biconnected_block_by_edge().len(), edges.len());
        prop_assert!(audit.biconnected_block_by_edge()
            .iter()
            .all(|&block| block < audit.biconnected_block_count()));
        prop_assert_eq!(
            audit.candidate_core_view_count(),
            audit.candidate_core_view_mask().iter().filter(|&&keep| keep).count()
        );
        prop_assert_eq!(
            audit.candidate_core_edge_count(),
            audit.candidate_core_edge_mask().iter().filter(|&&keep| keep).count()
        );

        let expected_bridges: Vec<_> = (0..edges.len())
            .filter(|&edge| component_count(num_views, &edges, Some(edge), None) > base_components)
            .collect();
        prop_assert_eq!(audit.bridge_edges(), &expected_bridges);
        let expected_articulations: Vec<_> = (0..num_views)
            .filter(|&view| component_count(num_views, &edges, None, Some(view)) > base_components)
            .collect();
        prop_assert_eq!(audit.articulation_views(), &expected_articulations);

        let core_views: Vec<_> = audit.candidate_core_view_mask()
            .iter()
            .enumerate()
            .filter_map(|(view, &keep)| keep.then_some(view))
            .collect();
        let core_edges: Vec<_> = edges
            .iter()
            .zip(audit.candidate_core_edge_mask())
            .filter_map(|(&edge, &keep)| keep.then_some(edge))
            .collect();
        if core_views.len() >= 3 {
            for &removed in &core_views {
                let remaining: Vec<_> = core_views.iter().copied().filter(|&view| view != removed).collect();
                let relabel: std::collections::HashMap<_, _> = remaining
                    .iter()
                    .enumerate()
                    .map(|(dense, &view)| (view, dense))
                    .collect();
                let reduced_edges: Vec<_> = core_edges
                    .iter()
                    .filter(|&&(view_i, view_j)| view_i != removed && view_j != removed)
                    .map(|&(view_i, view_j)| (relabel[&view_i], relabel[&view_j]))
                    .collect();
                prop_assert_eq!(component_count(remaining.len(), &reduced_edges, None, None), 1);
            }
        }

        let mut keyed_edges: Vec<_> = edges
            .iter()
            .copied()
            .enumerate()
            .map(|(index, (view_i, view_j))| {
                let edge = if endpoint_flips.get(index).copied().unwrap_or(false) {
                    (view_j, view_i)
                } else {
                    (view_i, view_j)
                };
                (order_keys.get(index).copied().unwrap_or(index as u16), index, edge)
            })
            .collect();
        keyed_edges.sort_unstable_by_key(|&(key, index, _)| (key, index));
        let shuffled: Vec<_> = keyed_edges.into_iter().map(|(_, _, edge)| edge).collect();
        let shuffled_audit = audit_graph_topology(num_views, &shuffled).unwrap();
        prop_assert_eq!(
            audit.candidate_core_view_mask(),
            shuffled_audit.candidate_core_view_mask()
        );
        prop_assert_eq!(audit.articulation_views(), shuffled_audit.articulation_views());
        prop_assert_eq!(audit.isolated_views(), shuffled_audit.isolated_views());
        prop_assert_eq!(
            audit.candidate_tail_views(),
            shuffled_audit.candidate_tail_views()
        );
        prop_assert_eq!(audit.disconnected_views(), shuffled_audit.disconnected_views());
        prop_assert_eq!(
            audit.connected_component_by_view(),
            shuffled_audit.connected_component_by_view()
        );
        prop_assert_eq!(
            canonical_bridge_endpoints(&edges, audit.bridge_edges()),
            canonical_bridge_endpoints(&shuffled, shuffled_audit.bridge_edges())
        );
        prop_assert_eq!(
            canonical_blocks(
                &edges,
                audit.biconnected_block_by_edge(),
                audit.biconnected_block_count(),
            ),
            canonical_blocks(
                &shuffled,
                shuffled_audit.biconnected_block_by_edge(),
                shuffled_audit.biconnected_block_count(),
            )
        );
    }

    #[test]
    fn multigraph_blocks_match_oracles_and_block_cut_invariants(
        num_views in 2usize..=8,
        raw_edges in proptest::collection::vec((any::<u16>(), any::<u16>(), any::<bool>(), any::<u16>()), 0..=24),
    ) {
        let mut keyed_edges: Vec<_> = raw_edges
            .iter()
            .enumerate()
            .map(|(index, &(raw_i, raw_j, flip, key))| {
                let view_i = raw_i as usize % num_views;
                let mut view_j = raw_j as usize % (num_views - 1);
                if view_j >= view_i {
                    view_j += 1;
                }
                let edge = if flip { (view_j, view_i) } else { (view_i, view_j) };
                (key, index, edge)
            })
            .collect();
        let edges: Vec<_> = keyed_edges.iter().map(|&(_, _, edge)| edge).collect();
        let audit = audit_graph_topology(num_views, &edges).unwrap();
        let base_components = component_count(num_views, &edges, None, None);

        let expected_bridges: Vec<_> = (0..edges.len())
            .filter(|&edge| component_count(num_views, &edges, Some(edge), None) > base_components)
            .collect();
        prop_assert_eq!(audit.bridge_edges(), &expected_bridges);
        let expected_articulations: Vec<_> = (0..num_views)
            .filter(|&view| component_count(num_views, &edges, None, Some(view)) > base_components)
            .collect();
        prop_assert_eq!(audit.articulation_views(), &expected_articulations);

        let mut block_edges = vec![Vec::new(); audit.biconnected_block_count()];
        for (edge, &block) in audit.biconnected_block_by_edge().iter().enumerate() {
            block_edges[block].push(edge);
        }
        prop_assert!(block_edges.iter().all(|block| !block.is_empty()));
        for block in &block_edges {
            if block.len() == 1 {
                prop_assert!(audit.bridge_edges().contains(&block[0]));
            }
        }
        for &bridge in audit.bridge_edges() {
            prop_assert_eq!(block_edges[audit.biconnected_block_by_edge()[bridge]].len(), 1);
        }

        let block_views: Vec<Vec<usize>> = block_edges
            .iter()
            .map(|block| {
                let mut views: Vec<_> = block
                    .iter()
                    .flat_map(|&edge| [edges[edge].0, edges[edge].1])
                    .collect();
                views.sort_unstable();
                views.dedup();
                views
            })
            .collect();
        for left in 0..block_views.len() {
            for right in (left + 1)..block_views.len() {
                let shared = block_views[left]
                    .iter()
                    .filter(|view| block_views[right].contains(view))
                    .count();
                prop_assert!(shared <= 1);
            }
        }
        for (block, views) in block_edges.iter().zip(&block_views) {
            if views.len() < 3 {
                continue;
            }
            for &removed in views {
                let remaining: Vec<_> = views.iter().copied().filter(|&view| view != removed).collect();
                let relabel: std::collections::HashMap<_, _> = remaining
                    .iter()
                    .enumerate()
                    .map(|(dense, &view)| (view, dense))
                    .collect();
                let reduced_edges: Vec<_> = block
                    .iter()
                    .map(|&edge| edges[edge])
                    .filter(|&(view_i, view_j)| view_i != removed && view_j != removed)
                    .map(|(view_i, view_j)| (relabel[&view_i], relabel[&view_j]))
                    .collect();
                prop_assert_eq!(component_count(remaining.len(), &reduced_edges, None, None), 1);
            }
        }

        if let Some(candidate_block) = audit.candidate_core_block() {
            let mut expected = 0usize;
            for block in 1..block_edges.len() {
                let score = (block_views[block].len(), block_edges[block].len());
                let expected_score = (block_views[expected].len(), block_edges[expected].len());
                if score > expected_score
                    || (score == expected_score && block_views[block] < block_views[expected])
                {
                    expected = block;
                }
            }
            prop_assert_eq!(candidate_block, expected);
        } else {
            prop_assert!(edges.is_empty());
        }

        keyed_edges.sort_unstable_by_key(|&(key, index, _)| (key, index));
        let permuted: Vec<_> = keyed_edges.into_iter().map(|(_, _, edge)| edge).collect();
        let permuted_audit = audit_graph_topology(num_views, &permuted).unwrap();
        prop_assert_eq!(
            canonical_blocks(
                &edges,
                audit.biconnected_block_by_edge(),
                audit.biconnected_block_count(),
            ),
            canonical_blocks(
                &permuted,
                permuted_audit.biconnected_block_by_edge(),
                permuted_audit.biconnected_block_count(),
            )
        );
        prop_assert_eq!(
            audit.candidate_core_view_mask(),
            permuted_audit.candidate_core_view_mask()
        );
    }

    #[test]
    fn every_generated_tree_edge_is_a_bridge(
        parents in proptest::collection::vec(0usize..64, 1..=31),
    ) {
        let num_views = parents.len() + 1;
        let edges: Vec<_> = parents
            .iter()
            .enumerate()
            .map(|(child_offset, &raw_parent)| {
                let child = child_offset + 1;
                (raw_parent % child, child)
            })
            .collect();
        let audit = audit_graph_topology(num_views, &edges).unwrap();
        prop_assert_eq!(audit.bridge_edges(), (0..edges.len()).collect::<Vec<_>>());
        prop_assert_eq!(audit.biconnected_block_count(), edges.len());
    }
}
