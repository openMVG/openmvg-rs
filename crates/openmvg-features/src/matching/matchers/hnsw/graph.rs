// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HNSW graph: contiguous descriptor storage, packed `u32` links,
//! deterministic level assignment, insertion, and beam search.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;

use openmvg_core::{RunSeed, determinism::AlgorithmRng};

use crate::matching::traits::DistanceMetric;

const M: usize = 12;
const M0: usize = 2 * M;
const MAX_LEVEL: usize = 32;

pub(super) const MIN_EF_CONSTRUCTION: usize = M;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Candidate {
    pub(super) distance: u32,
    pub(super) index: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct LinkBlock<const N: usize> {
    neighbors: [u32; N],
    len: u8,
}

impl<const N: usize> Default for LinkBlock<N> {
    fn default() -> Self {
        Self {
            neighbors: [u32::MAX; N],
            len: 0,
        }
    }
}

impl<const N: usize> LinkBlock<N> {
    fn as_slice(&self) -> &[u32] {
        &self.neighbors[..usize::from(self.len)]
    }

    fn set(&mut self, links: &[usize]) {
        debug_assert!(links.len() <= N);
        self.len = links.len() as u8;
        for (slot, &index) in self.neighbors.iter_mut().zip(links) {
            *slot = index as u32;
        }
    }

    fn push(&mut self, index: usize) {
        let len = usize::from(self.len);
        debug_assert!(len < N);
        self.neighbors[len] = index as u32;
        self.len += 1;
    }
}

#[derive(Debug)]
struct Node {
    base: LinkBlock<M0>,
    upper_offset: u32,
    level: u8,
}

impl Node {
    fn level(&self) -> usize {
        usize::from(self.level)
    }
}

#[derive(Debug, Default)]
pub(super) struct SearchScratch {
    marks: Vec<u32>,
    generation: u32,
    candidates: BinaryHeap<Reverse<Candidate>>,
    nearest: BinaryHeap<Candidate>,
    output: Vec<Candidate>,
}

impl SearchScratch {
    fn begin(&mut self, len: usize) {
        if self.marks.len() != len {
            self.marks.resize(len, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
        self.candidates.clear();
        self.nearest.clear();
        self.output.clear();
    }

    fn mark(&mut self, index: usize) -> bool {
        if self.marks[index] == self.generation {
            false
        } else {
            self.marks[index] = self.generation;
            true
        }
    }
}

/// Compact deterministic HNSW with a contiguous descriptor store.
pub(super) struct HnswGraph<T, D> {
    descriptors: Vec<T>,
    dimension: usize,
    nodes: Vec<Node>,
    upper_links: Vec<LinkBlock<M>>,
    entry_point: Option<usize>,
    max_level: usize,
    ef_construction: usize,
    metric: D,
    rng: AlgorithmRng,
}

impl<T, D> HnswGraph<T, D>
where
    T: Copy,
    D: DistanceMetric<T>,
{
    pub(super) fn new(dimension: usize, ef_construction: usize, seed: RunSeed, metric: D) -> Self {
        Self {
            descriptors: Vec::new(),
            dimension,
            nodes: Vec::new(),
            upper_links: Vec::new(),
            entry_point: None,
            max_level: 0,
            ef_construction: ef_construction.max(MIN_EF_CONSTRUCTION),
            metric,
            rng: AlgorithmRng::new(seed),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(super) fn reserve(&mut self, features: usize) {
        self.descriptors.reserve(features * self.dimension);
        self.nodes.reserve(features);
        self.upper_links.reserve(features.div_ceil(M - 1));
    }

    pub(super) fn finish_build(&mut self) {
        self.descriptors.shrink_to_fit();
        self.nodes.shrink_to_fit();
        self.upper_links.shrink_to_fit();
    }

    pub(super) fn memory_bytes(&self) -> usize {
        self.descriptors.capacity() * size_of::<T>()
            + self.nodes.capacity() * size_of::<Node>()
            + self.upper_links.capacity() * size_of::<LinkBlock<M>>()
    }

    pub(super) fn insert(&mut self, descriptor: &[T], scratch: &mut SearchScratch) {
        assert_eq!(descriptor.len(), self.dimension);
        let index = self.len();
        assert!(index < u32::MAX as usize, "HNSW index exceeds u32 capacity");
        let level = self.random_level();
        self.descriptors.extend_from_slice(descriptor);
        let upper_offset = self.upper_links.len();
        assert!(upper_offset < u32::MAX as usize);
        self.upper_links
            .resize_with(upper_offset + level, LinkBlock::default);
        self.nodes.push(Node {
            base: LinkBlock::default(),
            upper_offset: upper_offset as u32,
            level: level as u8,
        });

        let Some(entry_point) = self.entry_point else {
            self.entry_point = Some(index);
            self.max_level = level;
            return;
        };

        let mut entry_points = vec![entry_point];
        if self.max_level > level {
            for current_level in ((level + 1)..=self.max_level).rev() {
                self.search_layer(index, &entry_points, current_level, 1, scratch);
                entry_points.clear();
                entry_points.push(scratch.output[0].index);
            }
        }

        for current_level in (0..=level.min(self.max_level)).rev() {
            self.search_layer(
                index,
                &entry_points,
                current_level,
                self.ef_construction,
                scratch,
            );
            // As in hnswlib, a new node keeps M links on every layer; M0 only
            // caps the layer-0 lists that later insertions extend.
            let limit = if current_level == 0 { M0 } else { M };
            let selected = self.select_neighbors(&scratch.output, M);
            self.set_links(index, current_level, &selected);
            for neighbor in selected {
                self.add_reciprocal_link(neighbor, index, current_level, limit);
            }
            entry_points.clear();
            entry_points.extend(scratch.output.iter().map(|candidate| candidate.index));
        }

        if level > self.max_level {
            self.entry_point = Some(index);
            self.max_level = level;
        }
    }

    pub(super) fn search<'a>(
        &self,
        query: &[T],
        k: usize,
        ef_search: usize,
        scratch: &'a mut SearchScratch,
    ) -> &'a [Candidate] {
        debug_assert_eq!(query.len(), self.dimension);
        let Some(entry_point) = self.entry_point else {
            return &[];
        };
        let mut current = entry_point;
        for level in (1..=self.max_level).rev() {
            self.search_layer_descriptor(query, &[current], level, 1, scratch);
            current = scratch.output[0].index;
        }
        self.search_layer_descriptor(
            query,
            &[current],
            0,
            ef_search.max(k).min(self.len()),
            scratch,
        );
        let result_len = k.min(scratch.output.len());
        &scratch.output[..result_len]
    }

    fn random_level(&mut self) -> usize {
        let uniform = self.rng.open_unit_f64();
        ((-uniform.ln() / (M as f64).ln()) as usize).min(MAX_LEVEL)
    }

    fn search_layer(
        &self,
        query_index: usize,
        entry_points: &[usize],
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        self.search_layer_impl(
            entry_points,
            level,
            ef,
            |candidate| self.distance(query_index, candidate),
            scratch,
        );
    }

    fn search_layer_descriptor(
        &self,
        query: &[T],
        entry_points: &[usize],
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        self.search_layer_impl(
            entry_points,
            level,
            ef,
            |candidate| graph_distance(&self.metric, query, self.descriptor(candidate)),
            scratch,
        );
    }

    fn search_layer_impl<F>(
        &self,
        entry_points: &[usize],
        level: usize,
        ef: usize,
        mut distance_to_query: F,
        scratch: &mut SearchScratch,
    ) where
        F: FnMut(usize) -> u32,
    {
        scratch.begin(self.len());
        for &entry in entry_points {
            if entry >= self.len() || self.nodes[entry].level() < level || !scratch.mark(entry) {
                continue;
            }
            let candidate = Candidate {
                distance: distance_to_query(entry),
                index: entry,
            };
            scratch.candidates.push(Reverse(candidate));
            scratch.nearest.push(candidate);
            if scratch.nearest.len() > ef {
                scratch.nearest.pop();
            }
        }

        while let Some(Reverse(current)) = scratch.candidates.pop() {
            if scratch.nearest.len() >= ef
                && scratch.nearest.peek().is_some_and(|worst| current > *worst)
            {
                break;
            }
            for &neighbor in self.links(current.index, level) {
                let neighbor = neighbor as usize;
                if !scratch.mark(neighbor) {
                    continue;
                }
                let candidate = Candidate {
                    distance: distance_to_query(neighbor),
                    index: neighbor,
                };
                let should_add = scratch.nearest.len() < ef
                    || scratch
                        .nearest
                        .peek()
                        .is_some_and(|worst| candidate < *worst);
                if should_add {
                    scratch.candidates.push(Reverse(candidate));
                    scratch.nearest.push(candidate);
                    if scratch.nearest.len() > ef {
                        scratch.nearest.pop();
                    }
                }
            }
        }

        scratch.output.extend(scratch.nearest.iter().copied());
        scratch.output.sort_unstable();
    }

    /// Algorithm 4 with `extendCandidates=false`, `keepPrunedConnections=true`.
    ///
    /// `ranked` holds unique candidates sorted by distance to the reference.
    /// Link order does not affect search results, so the selection is stored
    /// unsorted.
    fn select_neighbors(&self, ranked: &[Candidate], limit: usize) -> Vec<usize> {
        let mut selected = Vec::with_capacity(limit);
        let mut pruned = Vec::new();
        for candidate in ranked {
            if selected.len() == limit {
                break;
            }
            let diverse = selected
                .iter()
                .all(|&neighbor| self.distance(candidate.index, neighbor) >= candidate.distance);
            if diverse {
                selected.push(candidate.index);
            } else {
                pruned.push(candidate.index);
            }
        }
        let room = limit - selected.len();
        selected.extend(pruned.into_iter().take(room));
        selected
    }

    fn add_reciprocal_link(
        &mut self,
        target: usize,
        new_neighbor: usize,
        level: usize,
        limit: usize,
    ) {
        let links = self.links(target, level);
        if links.contains(&(new_neighbor as u32)) {
            return;
        }
        if links.len() < limit {
            self.push_link(target, level, new_neighbor);
            return;
        }
        let mut ranked: Vec<Candidate> = links
            .iter()
            .map(|&index| index as usize)
            .chain(std::iter::once(new_neighbor))
            .map(|candidate| Candidate {
                distance: self.distance(target, candidate),
                index: candidate,
            })
            .collect();
        ranked.sort_unstable();
        let selected = self.select_neighbors(&ranked, limit);
        self.set_links(target, level, &selected);
    }

    pub(super) fn descriptor(&self, index: usize) -> &[T] {
        let start = index * self.dimension;
        &self.descriptors[start..start + self.dimension]
    }

    fn links(&self, index: usize, level: usize) -> &[u32] {
        if level == 0 {
            self.nodes[index].base.as_slice()
        } else {
            let offset = self.nodes[index].upper_offset as usize + level - 1;
            self.upper_links[offset].as_slice()
        }
    }

    fn set_links(&mut self, index: usize, level: usize, links: &[usize]) {
        if level == 0 {
            self.nodes[index].base.set(links);
        } else {
            let offset = self.nodes[index].upper_offset as usize + level - 1;
            self.upper_links[offset].set(links);
        }
    }

    fn push_link(&mut self, index: usize, level: usize, neighbor: usize) {
        if level == 0 {
            self.nodes[index].base.push(neighbor);
        } else {
            let offset = self.nodes[index].upper_offset as usize + level - 1;
            self.upper_links[offset].push(neighbor);
        }
    }

    #[inline]
    fn distance(&self, left: usize, right: usize) -> u32 {
        graph_distance(&self.metric, self.descriptor(left), self.descriptor(right))
    }

    #[cfg(test)]
    pub(super) fn validate(&self) -> bool {
        self.nodes.iter().enumerate().all(|(index, node)| {
            (0..=node.level()).all(|level| {
                let links = self.links(index, level);
                links.len() <= if level == 0 { M0 } else { M }
                    && links.iter().all(|&neighbor| {
                        let neighbor = neighbor as usize;
                        neighbor < self.len()
                            && neighbor != index
                            && self.nodes[neighbor].level() >= level
                    })
            })
        })
    }
}

/// Map a metric distance onto the graph's integer ordering key.
///
/// [`DistanceMetric`] distances are finite and non-negative, and for such
/// `f32` values the IEEE-754 bit pattern is monotonic after zero is canonicalized
/// to `+0.0`. Both signs of zero have the same numeric distance; the sign bit of
/// `-0.0` would otherwise rank it after positive distances. [`f32::from_bits`]
/// recovers the distance.
#[inline]
pub(super) fn graph_distance<T, D: DistanceMetric<T>>(metric: &D, left: &[T], right: &[T]) -> u32 {
    let distance = metric.distance(left, right);
    if distance == 0.0 {
        0
    } else {
        distance.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::super::HnswMatcher;
    use super::*;

    #[derive(Default)]
    struct SignedZeroMetric;

    impl DistanceMetric<u8> for SignedZeroMetric {
        fn distance(&self, left: &[u8], right: &[u8]) -> f32 {
            if left == right { -0.0 } else { 1.0 }
        }
    }

    #[test]
    fn signed_zero_distances_sort_before_positive_distances() {
        let mut matcher = HnswMatcher::<u8, SignedZeroMetric>::new();
        matcher.build(&[&[0_u8], &[1_u8]]).unwrap();
        let matches = matcher.query_single(&[0], 2).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index(), 0);
        assert_eq!(matches[0].distance(), 0.0);
        assert_eq!(matches[1].index(), 1);
        assert_eq!(matches[1].distance(), 1.0);
    }
}
