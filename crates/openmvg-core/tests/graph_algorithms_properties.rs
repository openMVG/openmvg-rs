// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_core::graph::{
    GraphError, analyze_undirected_topology, complete_connectivity, component_partition,
    maximum_spanning_tree,
};
use proptest::prelude::*;

fn simple_edges(node_count: usize, selected: &[bool]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let mut index = 0usize;
    for first in 0..node_count {
        for second in (first + 1)..node_count {
            if selected.get(index).copied().unwrap_or(false) {
                edges.push((first, second));
            }
            index += 1;
        }
    }
    edges
}

fn canonical_bridge_endpoints(
    edges: &[(usize, usize)],
    bridge_edges: &[usize],
) -> Vec<(usize, usize)> {
    let mut endpoints: Vec<_> = bridge_edges
        .iter()
        .map(|&edge| {
            let (first, second) = edges[edge];
            (first.min(second), first.max(second))
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
    for (&(first, second), &block) in edges.iter().zip(block_by_edge) {
        blocks[block].push((first.min(second), first.max(second)));
    }
    for block in &mut blocks {
        block.sort_unstable();
    }
    blocks.sort_unstable();
    blocks
}

fn component_count_after_removal(
    node_count: usize,
    edges: &[(usize, usize)],
    removed_edge: Option<usize>,
    removed_node: Option<usize>,
) -> usize {
    let mut adjacency = vec![Vec::new(); node_count];
    for (edge, &(first, second)) in edges.iter().enumerate() {
        if removed_edge == Some(edge) || removed_node == Some(first) || removed_node == Some(second)
        {
            continue;
        }
        adjacency[first].push(second);
        adjacency[second].push(first);
    }
    let mut visited = vec![false; node_count];
    if let Some(node) = removed_node {
        visited[node] = true;
    }
    let mut component_count = 0usize;
    for root in 0..node_count {
        if visited[root] {
            continue;
        }
        component_count += 1;
        visited[root] = true;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            for &neighbor in &adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    component_count
}

fn brute_force_maximum_forest_weight(
    node_count: usize,
    edges: &[(usize, usize)],
    weights: &[f64],
) -> f64 {
    let full = component_partition(node_count, edges).unwrap();
    let required_edges = node_count.saturating_sub(full.component_count());
    let mut best = f64::NEG_INFINITY;
    for mask in 0usize..(1usize << edges.len()) {
        if mask.count_ones() as usize != required_edges {
            continue;
        }
        let selected: Vec<_> = edges
            .iter()
            .enumerate()
            .filter_map(|(edge, &endpoints)| ((mask >> edge) & 1 == 1).then_some(endpoints))
            .collect();
        if component_partition(node_count, &selected).unwrap().labels() != full.labels() {
            continue;
        }
        let weight = weights
            .iter()
            .enumerate()
            .filter_map(|(edge, &weight)| ((mask >> edge) & 1 == 1).then_some(weight))
            .sum::<f64>();
        best = best.max(weight);
    }
    best
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    #[test]
    fn topology_matches_component_bridge_and_articulation_oracles(
        node_count in 0usize..=8,
        selected in proptest::collection::vec(any::<bool>(), 0..=28),
    ) {
        let edges = simple_edges(node_count, &selected);
        let topology = analyze_undirected_topology(node_count, &edges).unwrap();
        let partition = component_partition(node_count, &edges).unwrap();
        let base_components = component_count_after_removal(node_count, &edges, None, None);

        prop_assert_eq!(topology.component_count(), base_components);
        prop_assert_eq!(topology.component_by_node(), partition.labels());
        let reversed: Vec<_> = edges
            .iter()
            .rev()
            .map(|&(first, second)| (second, first))
            .collect();
        let reversed_partition = component_partition(node_count, &reversed).unwrap();
        prop_assert_eq!(reversed_partition.labels(), partition.labels());
        let expected_bridges: Vec<_> = (0..edges.len())
            .filter(|&edge| {
                component_count_after_removal(node_count, &edges, Some(edge), None)
                    > base_components
            })
            .collect();
        prop_assert_eq!(topology.bridge_edges(), expected_bridges);
        let expected_articulations: Vec<_> = (0..node_count)
            .filter(|&node| {
                component_count_after_removal(node_count, &edges, None, Some(node))
                    > base_components
            })
            .collect();
        prop_assert_eq!(topology.articulation_nodes(), expected_articulations);
    }

    #[test]
    fn maximum_tree_matches_exhaustive_weight_oracle_and_orientation(
        node_count in 1usize..=5,
        selected in proptest::collection::vec(any::<bool>(), 0..=10),
        raw_weights in proptest::collection::vec(any::<i8>(), 0..=10),
        raw_root in any::<usize>(),
    ) {
        let edges = simple_edges(node_count, &selected);
        let weights: Vec<_> = (0..edges.len())
            .map(|edge| f64::from(raw_weights.get(edge).copied().unwrap_or(0)))
            .collect();
        let root = raw_root % node_count;
        let result = maximum_spanning_tree(node_count, &edges, &weights, root);
        let partition = component_partition(node_count, &edges).unwrap();
        if partition.component_count() != 1 {
            let is_disconnected = matches!(result, Err(GraphError::Disconnected { .. }));
            prop_assert!(is_disconnected);
            return Ok(());
        }

        let tree = result.unwrap();
        let selected_weight: f64 = tree.iter().map(|branch| weights[branch.edge_index()]).sum();
        prop_assert_eq!(
            selected_weight,
            brute_force_maximum_forest_weight(node_count, &edges, &weights)
        );
        let mut reached = vec![false; node_count];
        reached[root] = true;
        for branch in tree {
            prop_assert!(reached[branch.parent()]);
            prop_assert!(!reached[branch.child()]);
            reached[branch.child()] = true;
        }
        prop_assert!(reached.into_iter().all(|node| node));
    }

    #[test]
    fn multigraph_topology_matches_removal_oracles_and_block_cut_invariants(
        node_count in 2usize..=8,
        raw_edges in proptest::collection::vec(
            (any::<u16>(), any::<u16>(), any::<bool>(), any::<u16>()),
            0..=24,
        ),
    ) {
        let mut keyed_edges: Vec<_> = raw_edges
            .iter()
            .enumerate()
            .map(|(index, &(raw_first, raw_second, flip, key))| {
                let first = raw_first as usize % node_count;
                let mut second = raw_second as usize % (node_count - 1);
                if second >= first {
                    second += 1;
                }
                let edge = if flip { (second, first) } else { (first, second) };
                (key, index, edge)
            })
            .collect();
        let edges: Vec<_> = keyed_edges.iter().map(|&(_, _, edge)| edge).collect();
        let topology = analyze_undirected_topology(node_count, &edges).unwrap();
        let base_components = component_count_after_removal(node_count, &edges, None, None);

        let expected_bridges: Vec<_> = (0..edges.len())
            .filter(|&edge| {
                component_count_after_removal(node_count, &edges, Some(edge), None)
                    > base_components
            })
            .collect();
        prop_assert_eq!(topology.bridge_edges(), &expected_bridges);
        let expected_articulations: Vec<_> = (0..node_count)
            .filter(|&node| {
                component_count_after_removal(node_count, &edges, None, Some(node))
                    > base_components
            })
            .collect();
        prop_assert_eq!(topology.articulation_nodes(), &expected_articulations);

        let mut block_edges = vec![Vec::new(); topology.biconnected_block_count()];
        for (edge, &block) in topology.biconnected_block_by_edge().iter().enumerate() {
            block_edges[block].push(edge);
        }
        prop_assert!(block_edges.iter().all(|block| !block.is_empty()));
        for block in &block_edges {
            if block.len() == 1 {
                prop_assert!(topology.bridge_edges().contains(&block[0]));
            }
        }
        for &bridge in topology.bridge_edges() {
            let block = topology.biconnected_block_by_edge()[bridge];
            prop_assert_eq!(block_edges[block].len(), 1);
        }

        let block_nodes: Vec<Vec<usize>> = block_edges
            .iter()
            .map(|block| {
                let mut nodes: Vec<_> = block
                    .iter()
                    .flat_map(|&edge| [edges[edge].0, edges[edge].1])
                    .collect();
                nodes.sort_unstable();
                nodes.dedup();
                nodes
            })
            .collect();
        for left in 0..block_nodes.len() {
            for right in (left + 1)..block_nodes.len() {
                let shared = block_nodes[left]
                    .iter()
                    .filter(|node| block_nodes[right].contains(node))
                    .count();
                prop_assert!(shared <= 1);
            }
        }
        for (block, nodes) in block_edges.iter().zip(&block_nodes) {
            if nodes.len() < 3 {
                continue;
            }
            for &removed in nodes {
                let remaining: Vec<_> = nodes
                    .iter()
                    .copied()
                    .filter(|&node| node != removed)
                    .collect();
                let relabel: std::collections::HashMap<_, _> = remaining
                    .iter()
                    .enumerate()
                    .map(|(dense, &node)| (node, dense))
                    .collect();
                let reduced_edges: Vec<_> = block
                    .iter()
                    .map(|&edge| edges[edge])
                    .filter(|&(first, second)| first != removed && second != removed)
                    .map(|(first, second)| (relabel[&first], relabel[&second]))
                    .collect();
                prop_assert_eq!(
                    component_count_after_removal(
                        remaining.len(),
                        &reduced_edges,
                        None,
                        None,
                    ),
                    1,
                );
            }
        }

        keyed_edges.sort_unstable_by_key(|&(key, index, _)| (key, index));
        let permuted: Vec<_> = keyed_edges.into_iter().map(|(_, _, edge)| edge).collect();
        let permuted_topology = analyze_undirected_topology(node_count, &permuted).unwrap();
        prop_assert_eq!(
            topology.component_by_node(),
            permuted_topology.component_by_node(),
        );
        prop_assert_eq!(
            topology.articulation_nodes(),
            permuted_topology.articulation_nodes(),
        );
        prop_assert_eq!(
            canonical_bridge_endpoints(&edges, topology.bridge_edges()),
            canonical_bridge_endpoints(&permuted, permuted_topology.bridge_edges()),
        );
        prop_assert_eq!(
            canonical_blocks(
                &edges,
                topology.biconnected_block_by_edge(),
                topology.biconnected_block_count(),
            ),
            canonical_blocks(
                &permuted,
                permuted_topology.biconnected_block_by_edge(),
                permuted_topology.biconnected_block_count(),
            ),
        );
    }

    #[test]
    fn ordered_completion_restores_connectivity_with_the_minimum_edge_count(
        node_count in 0usize..=10,
        selected in proptest::collection::vec(any::<bool>(), 0..=45),
        initially_kept_bits in proptest::collection::vec(any::<bool>(), 0..=45),
        order_keys in proptest::collection::vec(any::<u16>(), 0..=45),
    ) {
        let edges = simple_edges(node_count, &selected);
        let initially_kept: Vec<_> = (0..edges.len())
            .map(|edge| initially_kept_bits.get(edge).copied().unwrap_or(false))
            .collect();
        let mut candidate_order: Vec<_> = (0..edges.len())
            .filter(|&edge| !initially_kept[edge])
            .collect();
        candidate_order.sort_unstable_by_key(|&edge| {
            (order_keys.get(edge).copied().unwrap_or(0), edge)
        });

        let completion = complete_connectivity(
            node_count,
            &edges,
            &initially_kept,
            &candidate_order,
        )
        .unwrap();
        let full = component_partition(node_count, &edges).unwrap();
        let initial_edges: Vec<_> = edges
            .iter()
            .zip(&initially_kept)
            .filter_map(|(&edge, &keep)| keep.then_some(edge))
            .collect();
        let initial = component_partition(node_count, &initial_edges).unwrap();
        let completed_edges: Vec<_> = edges
            .iter()
            .zip(completion.keep())
            .filter_map(|(&edge, &keep)| keep.then_some(edge))
            .collect();
        let completed = component_partition(node_count, &completed_edges).unwrap();

        prop_assert_eq!(completed.labels(), full.labels());
        prop_assert_eq!(completion.component_count(), full.component_count());
        prop_assert_eq!(
            completion.added_edges().len(),
            initial.component_count() - full.component_count()
        );
    }
}
