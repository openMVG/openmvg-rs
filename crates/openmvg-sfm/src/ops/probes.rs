// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read-only probes of the active pair graph: degrees, connected components, and rotation
//! residuals.

use crate::{PairId, Result, SfmError, SfmWorkspace, ViewId, ViewPair};
use openmvg_core::graph::{ComponentPartition, component_partition};
use openmvg_core::model::rotation_angle_deg;

/// Snapshot of the active pair graph: degrees, connectivity, and rotation residuals.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphProbes {
    /// Active pair edges.
    pub active_pair_count: usize,
    /// Active-edge degree of every image, in `ViewId` order.
    pub degrees: Vec<(ViewId, usize)>,
    /// Connected components of the active pair graph; isolated images appear as singletons.
    pub connected_components: Vec<Vec<ViewId>>,
    /// Disagreement between each active edge's measured relative rotation and the averaged global
    /// rotations, in degrees; empty before rotation averaging, and edges without a relative
    /// rotation or a solved endpoint are skipped.
    pub angular_residuals_deg: Vec<(ViewPair, f64)>,
}

pub(crate) fn graph_probes(workspace: &SfmWorkspace) -> Result<GraphProbes> {
    Ok(GraphProbes {
        active_pair_count: workspace.pair_graph().active_count(),
        degrees: active_degrees(workspace)?,
        connected_components: active_connected_components(workspace)?,
        angular_residuals_deg: angular_residuals_deg(workspace)?,
    })
}

fn active_degrees(workspace: &SfmWorkspace) -> Result<Vec<(ViewId, usize)>> {
    (0..workspace.pair_graph().view_count())
        .map(|index| {
            let view_id = ViewId::new(index);
            let degree = workspace
                .pair_graph()
                .neighbors(view_id)?
                .iter()
                .filter(|neighbor| workspace.pair_graph().is_active(neighbor.pair_id))
                .count();
            Ok((view_id, degree))
        })
        .collect()
}

fn active_endpoints(workspace: &SfmWorkspace) -> Result<Vec<(usize, usize)>> {
    let mut endpoints = Vec::with_capacity(workspace.pair_graph().active_count());
    for pair_id in workspace.pair_graph().active_pair_ids() {
        let pair = workspace
            .pair_graph()
            .pair(pair_id)
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
        endpoints.push((pair.first().index(), pair.second().index()));
    }
    Ok(endpoints)
}

fn active_component_partition(workspace: &SfmWorkspace) -> Result<ComponentPartition> {
    let endpoints = active_endpoints(workspace)?;
    component_partition(workspace.pair_graph().view_count(), &endpoints)
        .map_err(|error| SfmError::InvalidInput(error.to_string()))
}

/// Count connected components induced by active pair edges, excluding isolated images.
pub(crate) fn active_edge_component_count(workspace: &SfmWorkspace) -> Result<usize> {
    Ok(active_component_partition(workspace)?
        .sizes()
        .iter()
        .filter(|&&size| size > 1)
        .count())
}

pub(crate) fn active_connected_components(workspace: &SfmWorkspace) -> Result<Vec<Vec<ViewId>>> {
    let partition = active_component_partition(workspace)?;
    let mut components = partition
        .sizes()
        .iter()
        .map(|&size| Vec::with_capacity(size))
        .collect::<Vec<_>>();
    for (image, &component) in partition.labels().iter().enumerate() {
        components[component].push(ViewId::new(image));
    }
    Ok(components)
}

fn angular_residuals_deg(workspace: &SfmWorkspace) -> Result<Vec<(ViewPair, f64)>> {
    let Some(rotations) = workspace.rotation_estimates() else {
        return Ok(Vec::new());
    };
    let mut residuals = Vec::new();
    for pair_id in workspace.pair_graph().active_pair_ids() {
        let pair = workspace
            .pair_graph()
            .pair(pair_id)
            .ok_or(SfmError::InvalidId {
                kind: PairId::kind(),
                index: pair_id.index(),
            })?;
        let Some(model) = workspace.geometric_model_store().get(pair_id.index()) else {
            continue;
        };
        let Some(relative_rotation) = model.relative_rotation() else {
            continue;
        };
        let Some(rotation_i) = rotations.get(pair.first()) else {
            continue;
        };
        let Some(rotation_j) = rotations.get(pair.second()) else {
            continue;
        };
        let predicted = *rotation_j * rotation_i.inverse();
        let error = relative_rotation.inverse() * predicted;
        residuals.push((pair, rotation_angle_deg(&error)));
    }
    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageMeta;
    use crate::{Camera, IntrinsicId};

    #[test]
    fn active_edge_components_ignore_isolated_noncontiguous_image_ids() {
        let mut workspace = SfmWorkspace::builder()
            .image_meta(
                (0..6).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs([
                (ViewId::new(0), ViewId::new(2)),
                (ViewId::new(2), ViewId::new(4)),
                (ViewId::new(1), ViewId::new(3)),
            ])
            .build()
            .unwrap();

        assert_eq!(active_edge_component_count(&workspace).unwrap(), 2);

        let separate_pair = workspace
            .pair_graph()
            .pair_id(ViewId::new(1), ViewId::new(3))
            .unwrap();
        workspace
            .pair_graph_mut()
            .deactivate_pair_and(separate_pair)
            .unwrap();
        assert_eq!(active_edge_component_count(&workspace).unwrap(), 1);
    }
}
