// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Triangle enumeration: the closure evidence behind the cycle prior (step 1 of the conditioner).
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         Triangle enumeration                         │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: edges (i<j) with rotations and optional baselines
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Per view, sorted list of (higher neighbor k, edge index)     │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: For each edge (i,j) in input order: merge-intersect          │
//! │         higher(i) restricted to k>j with higher(j), ascending k      │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Per triangle (ij, jk, ik): rotation closure                  │
//! │         R_ik^-1 R_jk R_ij and translation-triplet residual           │
//! └──────────────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   ▼
//! Output: triangles in input-edge then ascending-k order
//! ```
//!
//! Every triangle `i < j < k` is produced once. The visiting order (input edge order, then
//! ascending `k`) is part of the contract: the reliability rounds sum in this order, so the order
//! keeps the result bit-reproducible.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector3;
use openmvg_core::model::{ViewId, rotation_angle_deg};

use super::{AdaptiveConditioningError, AdaptiveEdge};

/// Triangles whose three baseline directions are nearly collinear carry no translation evidence.
const MIN_TRANSLATION_SINE: f64 = 0.2;

/// One triangle `i < j < k` of verified pairs, packed in 32 bytes: a dense graph has millions of
/// them, so edge indices are `u32` and an unavailable translation residual is a NaN, not an
/// `Option`.
pub(super) struct Triangle {
    edges: [u32; 3],
    rotation_error_deg: f64,
    translation_error_deg: f64,
}

const _: () = assert!(std::mem::size_of::<Triangle>() == 32);

impl Triangle {
    pub(super) fn new(
        edges: [u32; 3],
        rotation_error_deg: f64,
        translation_error_deg: Option<f64>,
    ) -> Self {
        Self {
            edges,
            rotation_error_deg,
            translation_error_deg: translation_error_deg.unwrap_or(f64::NAN),
        }
    }

    /// Indices of the edges `(i, j)`, `(j, k)` and `(i, k)`.
    pub(super) fn edges(&self) -> [usize; 3] {
        self.edges.map(|e| e as usize)
    }

    /// Angle of the rotation closure `R_ik^-1 R_jk R_ij`, degrees.
    pub(super) fn rotation_error_deg(&self) -> f64 {
        self.rotation_error_deg
    }

    /// Angle between the third baseline and the plane of the other two, degrees. `None` when a
    /// baseline is unknown or the three directions are nearly collinear.
    pub(super) fn translation_error_deg(&self) -> Option<f64> {
        (!self.translation_error_deg.is_nan()).then_some(self.translation_error_deg)
    }
}

/// Validate `edges` and enumerate every triangle of the pair graph.
pub(super) fn enumerate_triangles(
    edges: &[AdaptiveEdge],
) -> Result<Vec<Triangle>, AdaptiveConditioningError> {
    if u32::try_from(edges.len()).is_err() {
        return Err(AdaptiveConditioningError::InvalidEdge(format!(
            "{} edges exceed the supported {}",
            edges.len(),
            u32::MAX
        )));
    }
    let higher = higher_neighbours(edges)?;
    let mut triangles = Vec::new();
    for (ij, e) in edges.iter().enumerate() {
        let (Some(from_i), Some(from_j)) = (higher.get(&e.view_i), higher.get(&e.view_j)) else {
            continue;
        };
        // `from_j` holds only views above `view_j`; trim `from_i` to the same range.
        let start = from_i.partition_point(|&(view, _)| view <= e.view_j);
        let mut a = from_i.get(start..).unwrap_or_default();
        let mut b = from_j.as_slice();
        while let (Some((&(view_a, ik), rest_a)), Some((&(view_b, jk), rest_b))) =
            (a.split_first(), b.split_first())
        {
            match view_a.cmp(&view_b) {
                std::cmp::Ordering::Less => a = rest_a,
                std::cmp::Ordering::Greater => b = rest_b,
                std::cmp::Ordering::Equal => {
                    triangles.push(triangle(edges, ij, jk, ik));
                    a = rest_a;
                    b = rest_b;
                }
            }
        }
    }
    Ok(triangles)
}

/// Per view, its neighbors above it with the connecting edge, sorted by neighbor.
fn higher_neighbours(
    edges: &[AdaptiveEdge],
) -> Result<BTreeMap<ViewId, Vec<(ViewId, usize)>>, AdaptiveConditioningError> {
    let mut seen = BTreeSet::new();
    let mut higher: BTreeMap<ViewId, Vec<(ViewId, usize)>> = BTreeMap::new();
    for (k, e) in edges.iter().enumerate() {
        if e.view_i >= e.view_j {
            return Err(AdaptiveConditioningError::InvalidEdge(format!(
                "edge {k} must have view_i < view_j, got {} and {}",
                e.view_i, e.view_j
            )));
        }
        if !e.rotation.matrix().iter().all(|x| x.is_finite()) {
            return Err(AdaptiveConditioningError::InvalidEdge(format!(
                "edge {k} has a non-finite rotation"
            )));
        }
        if !(e.evidence.is_finite() && e.evidence > 0.0) {
            return Err(AdaptiveConditioningError::InvalidEdge(format!(
                "edge {k} has non-positive or non-finite evidence"
            )));
        }
        if !seen.insert((e.view_i, e.view_j)) {
            return Err(AdaptiveConditioningError::InvalidEdge(format!(
                "pair ({}, {}) appears twice",
                e.view_i, e.view_j
            )));
        }
        higher.entry(e.view_i).or_default().push((e.view_j, k));
    }
    for list in higher.values_mut() {
        list.sort_unstable();
    }
    Ok(higher)
}

fn triangle(edges: &[AdaptiveEdge], ij: usize, jk: usize, ik: usize) -> Triangle {
    let (e_ij, e_jk, e_ik) = (&edges[ij], &edges[jk], &edges[ik]);
    // R_j = R_ij R_i, R_k = R_jk R_j = R_ik R_i  =>  closure = R_ik^-1 R_jk R_ij.
    let closure = e_ik.rotation.inverse() * e_jk.rotation * e_ij.rotation;
    let translation_error_deg = match (unit(e_ij), unit(e_jk), unit(e_ik)) {
        (Some(d_ij), Some(d_jk), Some(d_ik)) => {
            let d_jk_in_i = e_ij.rotation.inverse() * d_jk;
            let normal = d_ij.cross(&d_jk_in_i);
            (normal.norm() >= MIN_TRANSLATION_SINE).then(|| {
                normal
                    .normalize()
                    .dot(&d_ik)
                    .abs()
                    .clamp(0.0, 1.0)
                    .asin()
                    .to_degrees()
            })
        }
        _ => None,
    };
    // `enumerate_triangles` checked that every edge index fits.
    Triangle::new(
        [ij as u32, jk as u32, ik as u32],
        rotation_angle_deg(&closure),
        translation_error_deg,
    )
}

/// Unit baseline direction, `None` when unknown, non-finite or zero. Any positive magnitude works:
/// the vector is rescaled by its largest component first, so neither tiny nor huge baselines
/// underflow or overflow in the norm.
fn unit(edge: &AdaptiveEdge) -> Option<Vector3<f64>> {
    let b = edge.baseline.filter(|b| b.iter().all(|x| x.is_finite()))?;
    let largest = b.amax();
    (largest > 0.0).then(|| (b / largest).normalize())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{edge, rotation, truth};
    use super::*;

    /// The map-based enumeration this module replaced, kept as the order and content oracle.
    fn reference(edges: &[AdaptiveEdge]) -> Vec<[usize; 3]> {
        let index: BTreeMap<(ViewId, ViewId), usize> = edges
            .iter()
            .enumerate()
            .map(|(k, e)| ((e.view_i, e.view_j), k))
            .collect();
        let mut higher: BTreeMap<ViewId, Vec<ViewId>> = BTreeMap::new();
        for e in edges {
            higher.entry(e.view_i).or_default().push(e.view_j);
        }
        for list in higher.values_mut() {
            list.sort_unstable();
        }
        let mut out = Vec::new();
        for (ij, e) in edges.iter().enumerate() {
            let Some(candidates) = higher.get(&e.view_i) else {
                continue;
            };
            for &k in candidates.iter().filter(|&&k| k > e.view_j) {
                if let (Some(&jk), Some(&ik)) =
                    (index.get(&(e.view_j, k)), index.get(&(e.view_i, k)))
                {
                    out.push([ij, jk, ik]);
                }
            }
        }
        out
    }

    /// A sparse graph with irregular degrees, gaps in the view ids and a shuffled edge order.
    fn irregular_graph() -> Vec<AdaptiveEdge> {
        let t = truth(30);
        let mut edges = Vec::new();
        for i in 0..30 {
            for j in (i + 1)..30 {
                if (i * 7 + j * 13) % 5 < 2 || j == i + 1 {
                    edges.push(edge(&t, i, j, 0.1, 20.0));
                }
            }
        }
        // Deterministic shuffle: stride through the list.
        let n = edges.len();
        (0..n).map(|k| edges[(k * 37) % n]).collect()
    }

    #[test]
    fn matches_the_reference_enumeration_including_order() {
        let edges = irregular_graph();
        let n = edges.len();
        assert_eq!(
            (0..n).map(|k| (k * 37) % n).collect::<BTreeSet<_>>().len(),
            n,
            "the stride shuffle must be a permutation"
        );
        let got: Vec<[usize; 3]> = enumerate_triangles(&edges)
            .unwrap()
            .iter()
            .map(|t| t.edges())
            .collect();
        let want = reference(&edges);
        assert!(!want.is_empty());
        assert_eq!(got, want);
    }

    #[test]
    fn a_clique_has_n_choose_3_triangles_each_once() {
        let n = 9;
        let t = truth(n);
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(edge(&t, i, j, 0.0, 10.0));
            }
        }
        let triangles = enumerate_triangles(&edges).unwrap();
        assert_eq!(triangles.len(), n * (n - 1) * (n - 2) / 6);
        let distinct: BTreeSet<[usize; 3]> = triangles
            .iter()
            .map(|t| {
                let mut sorted = t.edges();
                sorted.sort_unstable();
                sorted
            })
            .collect();
        assert_eq!(distinct.len(), triangles.len());
    }

    #[test]
    fn the_closure_error_is_the_planted_rotation_angle() {
        let t = truth(3);
        let mut edges = vec![
            edge(&t, 0, 1, 0.0, 10.0),
            edge(&t, 1, 2, 0.0, 10.0),
            edge(&t, 0, 2, 0.0, 10.0),
        ];
        let exact = enumerate_triangles(&edges).unwrap();
        assert!(exact[0].rotation_error_deg() < 1e-6);
        edges[2].rotation = rotation([0.0, 0.0, 1.0], 12.0) * edges[2].rotation;
        let bent = enumerate_triangles(&edges).unwrap();
        assert!((bent[0].rotation_error_deg() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn translation_residual_needs_non_collinear_known_baselines() {
        let t = truth(3);
        let mut edges = vec![
            edge(&t, 0, 1, 0.0, 10.0),
            edge(&t, 1, 2, 0.0, 10.0),
            edge(&t, 0, 2, 0.0, 10.0),
        ];
        assert!(
            enumerate_triangles(&edges).unwrap()[0]
                .translation_error_deg()
                .is_none()
        );
        // Centres on a plane: the three baselines are coplanar, so the residual is zero.
        let centres = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.3, 1.0, 0.0),
        ];
        let baseline = |i: usize, j: usize| {
            // Direction of centre j in camera i's frame; the test rotations are the identity.
            Some(centres[j] - centres[i])
        };
        for e in &mut edges {
            e.rotation = nalgebra::Rotation3::identity();
            e.baseline = baseline(e.view_i.index(), e.view_j.index());
        }
        let residual = enumerate_triangles(&edges).unwrap()[0].translation_error_deg();
        assert!(residual.is_some_and(|r| r < 1e-6), "{residual:?}");
        // Collinear centres carry no translation evidence.
        for e in &mut edges {
            e.baseline = Some(Vector3::new(1.0, 0.0, 0.0));
        }
        assert!(
            enumerate_triangles(&edges).unwrap()[0]
                .translation_error_deg()
                .is_none()
        );
    }

    #[test]
    fn baseline_magnitude_does_not_change_the_translation_evidence() {
        let t = truth(3);
        let centres = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.3, 1.0, 0.4),
        ];
        let residual = |scale: f64| {
            let mut edges = vec![
                edge(&t, 0, 1, 0.0, 10.0),
                edge(&t, 1, 2, 0.0, 10.0),
                edge(&t, 0, 2, 0.0, 10.0),
            ];
            for e in &mut edges {
                e.rotation = nalgebra::Rotation3::identity();
                e.baseline = Some((centres[e.view_j.index()] - centres[e.view_i.index()]) * scale);
            }
            enumerate_triangles(&edges).unwrap()[0].translation_error_deg()
        };
        // The centres are coplanar, so the residual is zero up to rounding at every magnitude; the
        // point is that it stays known instead of falling back to `None`.
        for scale in [1.0, 1e-300, 1e-13, 1e-3, 1e3, 1e200, 1e300] {
            let r = residual(scale);
            assert!(r.is_some_and(|x| x < 1e-6), "scale {scale}: {r:?}");
        }
    }

    #[test]
    fn malformed_edges_are_rejected() {
        let t = truth(3);
        let good = edge(&t, 0, 1, 0.0, 10.0);
        let mut reversed = good;
        reversed.view_i = ViewId::new(1);
        reversed.view_j = ViewId::new(0);
        let mut no_evidence = good;
        no_evidence.evidence = 0.0;
        let mut nan_evidence = good;
        nan_evidence.evidence = f64::NAN;
        let mut nan_rotation = good;
        nan_rotation.rotation =
            nalgebra::Rotation3::from_matrix_unchecked(nalgebra::Matrix3::from_element(f64::NAN));
        for bad in [
            &[reversed][..],
            &[no_evidence],
            &[nan_evidence],
            &[nan_rotation],
            &[good, good],
        ] {
            assert!(matches!(
                enumerate_triangles(bad),
                Err(AdaptiveConditioningError::InvalidEdge(_))
            ));
        }
    }
}
