// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Global averaging dispatch, solver adapters, and pose commit.

use super::rigid_core::largest_rigid_biconnected_translation_core;
use super::*;
use openmvg_core::graph::component_partition;
/// Immutable global rotation solution shared by isolated candidate workspaces.
///
/// Only a compact averaging-relevant graph signature is retained. Full geometric models and
/// their inlier payloads are temporary solve scratch and are released before triangulation or BA.
/// Exact relative-pose equality is an identity guard between cloned candidate domains, not a
/// tolerance-based numerical quality test.
#[derive(Debug)]
pub(crate) struct GlobalRotationPreparation {
    active_edges: Vec<GlobalRotationEdgeSignature>,
    rotations: RotationEstimates,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GlobalRotationEdgeSignature {
    edge: PairEdge,
    relative_rotation: nalgebra::Rotation3<f64>,
    inlier_count: usize,
}

impl GlobalRotationPreparation {
    /// Validate the exact averaging-relevant active graph before exposing the rotation field.
    pub(crate) fn validate_for<'a>(
        &'a self,
        workspace: &SfmWorkspace,
    ) -> Result<ValidatedGlobalRotations<'a>> {
        self.validate_workspace(workspace)?;
        Ok(ValidatedGlobalRotations {
            rotations: &self.rotations,
        })
    }

    fn validate_workspace(&self, workspace: &SfmWorkspace) -> Result<()> {
        if workspace.pair_graph().active_count() != self.active_edges.len() {
            return Err(SfmError::InvalidInput(
                "prepared global rotations do not match the active pair count".to_owned(),
            ));
        }
        for (expected, pair_id) in self
            .active_edges
            .iter()
            .zip(workspace.pair_graph().active_pair_ids())
        {
            let pair = workspace
                .pair_graph()
                .pair(pair_id)
                .ok_or(SfmError::InvalidId {
                    kind: crate::PairId::kind(),
                    index: pair_id.index(),
                })?;
            let confidence =
                workspace
                    .pair_graph()
                    .confidence(pair_id)
                    .ok_or(SfmError::InvalidId {
                        kind: crate::PairId::kind(),
                        index: pair_id.index(),
                    })?;
            let model = workspace
                .geometric_model_store()
                .get(pair_id.index())
                .ok_or(SfmError::MissingPrerequisite {
                    prerequisite: "geometric models",
                    method: "verify_pairs",
                })?;
            let relative_rotation = model.relative_rotation().ok_or_else(|| {
                SfmError::InvalidInput(format!("pair {pair_id} has no relative rotation"))
            })?;
            if expected.edge.pair != pair
                || expected.edge.view_i != pair.first()
                || expected.edge.view_j != pair.second()
                || expected.edge.confidence.to_bits() != confidence.to_bits()
                || expected.relative_rotation != *relative_rotation
                || expected.inlier_count != model.evidence().inlier_count
            {
                return Err(SfmError::InvalidInput(
                    "prepared global rotations do not match the active pose graph".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Proof that a shared rotation solution matches one candidate workspace's active graph.
#[derive(Clone, Copy)]
pub(crate) struct ValidatedGlobalRotations<'a> {
    rotations: &'a RotationEstimates,
}

impl<'a> ValidatedGlobalRotations<'a> {
    pub(crate) const fn estimates(self) -> &'a RotationEstimates {
        self.rotations
    }
}

pub(crate) fn prepare_global_rotation_solution(
    workspace: &SfmWorkspace,
    rotation_solver: RotationAveragingFn,
) -> Result<GlobalRotationPreparation> {
    let (edges, models) = active_geometric_models(workspace)?;
    if edges.is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "verified active pairs",
            method: "verify_pairs",
        });
    }
    let rotations = solve_rotation_on_active_graph(rotation_solver, &edges, &models)?;
    for edge in &edges {
        if rotations.get(edge.view_i).is_none() || rotations.get(edge.view_j).is_none() {
            return Err(SfmError::InvalidInput(format!(
                "rotation averaging did not cover both endpoints of active pair {}",
                edge.pair
            )));
        }
    }
    let active_edges = edges
        .iter()
        .zip(&models)
        .map(|(edge, model)| {
            let relative_rotation = model.relative_rotation().copied().ok_or_else(|| {
                SfmError::InvalidInput(format!("pair {} has no relative rotation", edge.pair))
            })?;
            Ok(GlobalRotationEdgeSignature {
                edge: *edge,
                relative_rotation,
                inlier_count: model.evidence().inlier_count,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GlobalRotationPreparation {
        active_edges,
        rotations,
    })
}

/// Run rotation then translation averaging over every active verified edge.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn run_global_averaging_scoped(
    workspace: &mut SfmWorkspace,
    rotation_solver: RotationAveragingFn,
    translation_solver: TranslationAveragingFn,
    min_translation_parallax_deg: Option<f64>,
    min_edge_isotropy: f64,
) -> Result<GlobalAveragingReport> {
    let (edges, models) = active_geometric_models(workspace)?;
    if edges.is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "verified active pairs",
            method: "verify_pairs",
        });
    }

    // Rotations always use the FULL edge set — planar / near-pure-rotation (config-6)
    // pairs still give good relative rotations, so excluding them here would only lose
    // constraints. The retained component may use non-contiguous workspace image IDs, so
    // compact them only at the numerical-solver boundary and map the estimates back.
    let rotations = solve_rotation_on_active_graph(rotation_solver, &edges, &models)?;

    run_global_translation_from_domain(
        workspace,
        translation_solver,
        &rotations,
        edges,
        models,
        min_translation_parallax_deg,
        crate::config::GlobalTranslationScope::AllActiveEdges,
        min_edge_isotropy,
    )
}

/// Complete the separate global pose solve from an immutable, already estimated rotation field.
///
/// This is the candidate seam: direction translation consumes the shared rotations,
/// while other candidate families can bypass this function entirely.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_global_translation_from_validated_rotations(
    workspace: &mut SfmWorkspace,
    translation_solver: TranslationAveragingFn,
    rotations: ValidatedGlobalRotations<'_>,
    min_translation_parallax_deg: Option<f64>,
    translation_scope: crate::config::GlobalTranslationScope,
    min_edge_isotropy: f64,
) -> Result<GlobalAveragingReport> {
    let (edges, models) = active_geometric_models(workspace)?;
    run_global_translation_from_domain(
        workspace,
        translation_solver,
        rotations.estimates(),
        edges,
        models,
        min_translation_parallax_deg,
        translation_scope,
        min_edge_isotropy,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_global_translation_from_domain(
    workspace: &mut SfmWorkspace,
    translation_solver: TranslationAveragingFn,
    rotations: &RotationEstimates,
    edges: Vec<PairEdge>,
    models: Vec<GeometricModel>,
    min_translation_parallax_deg: Option<f64>,
    translation_scope: crate::config::GlobalTranslationScope,
    min_edge_isotropy: f64,
) -> Result<GlobalAveragingReport> {
    if edges.is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "verified active pairs",
            method: "verify_pairs",
        });
    }
    let active_pair_count = edges.len();

    // Rotation-only constraints end here. Translation, triangulation, and all
    // downstream metric stages receive only models carrying a full relative pose.
    let (edges, models) = translation_geometric_models(edges, models)?;
    if edges.is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "translation-capable relative poses",
            method: "global_translation",
        });
    }

    // Translation is corrupted by low-parallax edges: their relative-translation
    // *direction* is degenerate (no metric baseline), and the direction-only L1-IRLS solve
    // collapses scale if it trusts them. When the parallax gate is enabled, solve the
    // translation over a parallax-filtered edge subset only (connectivity-preserving).
    let (translation_edges, translation_models) = match min_translation_parallax_deg {
        Some(min_deg) if min_deg > 0.0 => parallax_filtered_translation_edges(
            workspace,
            &edges,
            &models,
            min_deg,
            workspace.pair_graph().view_count(),
        )?,
        _ => (edges, models),
    };
    let (translation_edges, translation_models) = match translation_scope {
        crate::config::GlobalTranslationScope::AllActiveEdges => {
            (translation_edges, translation_models)
        }
        crate::config::GlobalTranslationScope::LargestRigidBiconnectedCore => {
            largest_rigid_biconnected_translation_core(
                &translation_edges,
                &translation_models,
                rotations,
                min_edge_isotropy,
            )?
        }
    };
    // Rotation-only evidence may connect several otherwise independent metric islands.
    // Keep that evidence in the full rotation field, but give translation averaging one
    // deterministic connected domain. Cameras outside it remain available for later growth.
    let (translation_edges, translation_models) =
        largest_translation_connected_domain(translation_edges, translation_models)?;
    let centers = solve_translation_on_active_graph(
        translation_solver,
        &translation_edges,
        &translation_models,
        rotations,
    )?;

    populate_pose_records(workspace, rotations, &centers)?;
    let report = report_from_estimates(active_pair_count, rotations, &centers);
    workspace.set_global_averaging(rotations.clone(), centers);
    Ok(report)
}

/// Adaptive soft conditioning of the active rotation graph, in edge order, from
/// [`openmvg_geometry::averaging::view_graph::condition_edges_adaptive`].
pub(crate) fn condition_rotations_adaptive(
    rotation_solver: RotationAveragingFn,
    edges: &[PairEdge],
    models: &[GeometricModel],
) -> Result<openmvg_geometry::averaging::view_graph::AdaptiveConditioning> {
    use openmvg_geometry::averaging::view_graph::{AdaptiveEdge, condition_edges_adaptive};
    ensure_parallel_inputs(edges, models, "rotation")?;
    let dense = DenseImageMap::from_edges(edges)?;
    let dense_edges = dense.remap_edges(edges)?;
    let adaptive = dense_edges
        .iter()
        .zip(models)
        .map(|(edge, model)| {
            Ok(AdaptiveEdge {
                view_i: edge.view_i,
                view_j: edge.view_j,
                rotation: *model_rotation(edge, model)?,
                evidence: model.evidence().inlier_count.max(1) as f64,
                baseline: model
                    .relative_pose()
                    .map(|pose| *pose.baseline_direction_in_source().as_ref()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    condition_edges_adaptive(&adaptive, &|relatives| {
        rotation_solver(relatives, &RotationConfig::default())
    })
    .map_err(|error| SfmError::InvalidInput(format!("adaptive rotation conditioning: {error}")))
}

/// Densely reindex the images incident to `edges` for a numerical rotation solve, then
/// restore the workspace image IDs in the returned estimates. Component selection remains
/// a view-graph operation; this helper does only ID bookkeeping.
pub(crate) fn solve_rotation_on_active_graph(
    rotation_solver: RotationAveragingFn,
    edges: &[PairEdge],
    models: &[GeometricModel],
) -> Result<RotationEstimates> {
    let dense = DenseImageMap::from_edges(edges)?;
    let dense_edges = dense.remap_edges(edges)?;
    let relative_rotations = relative_rotations(&dense_edges, models)?;
    let result = rotation_solver(&relative_rotations, &RotationConfig::default())
        .map_err(|error| SfmError::InvalidInput(format!("rotation averaging failed: {error}")))?;
    let estimates = RotationEstimates::try_new(
        result.rotations.into_iter().collect(),
        result.iteration_count,
        result.residual,
    )?;
    estimates.validate_dense_domain(dense.len())?;
    let rotations = estimates
        .rotations()
        .iter()
        .map(|(view, rotation)| Ok((dense.restore(*view)?, *rotation)))
        .collect::<Result<Vec<_>>>()?;
    RotationEstimates::try_new(rotations, estimates.iterations, estimates.residual)
}

/// Translation counterpart of [`solve_rotation_on_active_graph`]. Both the pair endpoints
/// and the already-estimated rotations are mapped into the same compact image-ID space.
pub(super) fn solve_translation_on_active_graph(
    translation_solver: TranslationAveragingFn,
    edges: &[PairEdge],
    models: &[GeometricModel],
    rotations: &RotationEstimates,
) -> Result<CameraCenters> {
    let dense = DenseImageMap::from_edges(edges)?;
    let dense_edges = dense.remap_edges(edges)?;
    let absolute_rotations = dense
        .originals
        .iter()
        .enumerate()
        .map(|(dense_index, original)| {
            rotations
                .get(*original)
                .copied()
                .map(|rotation| (ViewId::new(dense_index), rotation))
                .ok_or_else(|| {
                    SfmError::InvalidInput(format!(
                        "rotation averaging did not estimate active view {original}"
                    ))
                })
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let relative_translations = relative_translations(&dense_edges, models)?;
    let result = translation_solver(
        &relative_translations,
        &absolute_rotations,
        &TranslationConfig::default(),
    )
    .map_err(|error| SfmError::InvalidInput(format!("translation averaging failed: {error}")))?;
    let estimates = CameraCenters::try_new(
        result.translations.into_iter().collect(),
        result.iteration_count,
        result.residual,
    )?;
    estimates.validate_dense_domain(dense.len())?;
    let centers = estimates
        .centers()
        .iter()
        .map(|(view, center)| Ok((dense.restore(*view)?, *center)))
        .collect::<Result<Vec<_>>>()?;
    CameraCenters::try_new(centers, estimates.iterations, estimates.residual)
}

fn model_rotation<'a>(
    edge: &PairEdge,
    model: &'a GeometricModel,
) -> Result<&'a nalgebra::Rotation3<f64>> {
    model.relative_rotation().ok_or_else(|| {
        SfmError::InvalidInput(format!(
            "geometric model for {} has no relative rotation",
            edge.pair
        ))
    })
}

fn relative_rotations(
    edges: &[PairEdge],
    models: &[GeometricModel],
) -> Result<Vec<RelativeRotation>> {
    ensure_parallel_inputs(edges, models, "rotation")?;
    edges
        .iter()
        .zip(models)
        .map(|(edge, model)| {
            RelativeRotation::try_new(
                edge.view_i,
                edge.view_j,
                *model_rotation(edge, model)?,
                observation_weight(edge, model),
            )
            .map_err(|error| SfmError::InvalidInput(error.to_string()))
        })
        .collect()
}

fn relative_translations(
    edges: &[PairEdge],
    models: &[GeometricModel],
) -> Result<Vec<RelativeTranslation>> {
    ensure_parallel_inputs(edges, models, "translation")?;
    edges
        .iter()
        .zip(models)
        .map(|(edge, model)| {
            let pose = model.relative_pose().ok_or_else(|| {
                SfmError::InvalidInput(format!(
                    "geometric model for {} has no relative pose",
                    edge.pair
                ))
            })?;
            RelativeTranslation::try_new(
                edge.view_i,
                edge.view_j,
                pose.baseline_direction_in_source().into_inner(),
                observation_weight(edge, model),
            )
            .map_err(|error| SfmError::InvalidInput(error.to_string()))
        })
        .collect()
}

fn ensure_parallel_inputs(
    edges: &[PairEdge],
    models: &[GeometricModel],
    family: &str,
) -> Result<()> {
    if edges.len() != models.len() {
        return Err(SfmError::InvalidInput(format!(
            "expected one geometric model per {family} edge, got {} edges and {} models",
            edges.len(),
            models.len()
        )));
    }
    Ok(())
}

fn observation_weight(edge: &PairEdge, model: &GeometricModel) -> f64 {
    (model.evidence().inlier_count.max(1) as f64 * edge.confidence).max(1e-6)
}

pub(super) fn largest_translation_connected_domain(
    edges: Vec<PairEdge>,
    models: Vec<GeometricModel>,
) -> Result<(Vec<PairEdge>, Vec<GeometricModel>)> {
    if edges.len() != models.len() {
        return Err(SfmError::InvalidInput(format!(
            "expected one geometric model per translation edge, got {} pairs and {} models",
            edges.len(),
            models.len()
        )));
    }
    let dense = DenseImageMap::from_edges(&edges)?;
    let endpoints = edges
        .iter()
        .map(|edge| {
            Ok((
                dense.compact(edge.view_i)?.index(),
                dense.compact(edge.view_j)?.index(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let partition = component_partition(dense.len(), &endpoints)
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    let retained = partition.largest_component().ok_or_else(|| {
        SfmError::InvalidInput("translation graph has no connected component".to_owned())
    })?;

    let (edges, models): (Vec<_>, Vec<_>) = edges
        .into_iter()
        .zip(models)
        .zip(endpoints)
        .filter_map(|((edge, model), (view_i, _))| {
            (partition.label(view_i) == Some(retained)).then_some((edge, model))
        })
        .unzip();
    Ok((edges, models))
}

/// Median per-edge parallax angle (degrees): over the edge's inlier correspondences, the
/// angle between bearing `b_i` and `R_ijᵀ·b_j`, where each `b` is the model-specific
/// unit ray returned by that camera's pixel unprojection and `R_ij` is the pair's verified
/// relative rotation (world/cam-i → cam-j).
///
/// This angle is ~0 for pure-rotation / planar pairs (`b_i ≈ R_ijᵀ·b_j`) and grows with
/// real baseline/parallax, so it is a clean, dataset-agnostic proxy for whether an edge
/// carries a metric translation constraint. Returns `None` when it cannot be computed
/// (no relative pose / intrinsics / inliers) — the caller treats such edges as informative
/// (kept) so a missing proxy never silently drops constraints.
fn edge_median_parallax_deg(
    workspace: &SfmWorkspace,
    edge: &PairEdge,
    model: &GeometricModel,
) -> Option<f64> {
    let pair_id = workspace
        .pair_graph()
        .pair_id(edge.pair.first(), edge.pair.second())?;
    pair_median_parallax_deg(workspace, pair_id, model)
}

/// [`edge_median_parallax_deg`] for one verified pair of the workspace pair graph.
pub(crate) fn pair_median_parallax_deg(
    workspace: &SfmWorkspace,
    pair_id: crate::PairId,
    model: &GeometricModel,
) -> Option<f64> {
    let rel = model.relative_pose()?;
    let r_ij = *rel.rotation_source_to_target();

    let pair = workspace.pair_graph().pair(pair_id)?;
    let (view_i, view_j) = (pair.first(), pair.second());
    let camera_i = workspace.camera_model(workspace.view(view_i)?.intrinsic_id)?;
    let camera_j = workspace.camera_model(workspace.view(view_j)?.intrinsic_id)?;
    let keypoints_i = workspace.keypoint_store().get(view_i.index())?;
    let keypoints_j = workspace.keypoint_store().get(view_j.index())?;
    let matches = workspace.match_store().get(pair_id.index())?;
    let inliers = model.evidence().inliers.as_slice();

    let mut angles = Vec::new();
    for (index, matched) in matches.as_slice().iter().enumerate() {
        if inliers.get(index).copied() != Some(true) {
            continue;
        }
        let (Some(kp_i), Some(kp_j)) = (
            keypoints_i.get(matched.first().index()),
            keypoints_j.get(matched.second().index()),
        ) else {
            continue;
        };
        let (Some(b_i), Some(b_j)) = (
            camera_i
                .unproject(&nalgebra::Point2::new(kp_i.x() as f64, kp_i.y() as f64))
                .ok()
                .and_then(|bearing| bearing.try_normalize(1e-12)),
            camera_j
                .unproject(&nalgebra::Point2::new(kp_j.x() as f64, kp_j.y() as f64))
                .ok()
                .and_then(|bearing| bearing.try_normalize(1e-12)),
        ) else {
            continue;
        };
        // R_ijᵀ·b_j: bring cam-j's bearing into cam-i's frame (inverse == transpose for SO(3)).
        let b_j = r_ij.inverse() * b_j;
        let cos = b_i.dot(&b_j).clamp(-1.0, 1.0);
        angles.push(cos.acos().to_degrees());
    }
    if angles.is_empty() {
        return None;
    }
    angles.sort_by(f64::total_cmp);
    Some(angles[angles.len() / 2])
}

/// The parallax-gated **translation** edge subset. Keeps only edges whose median per-edge
/// parallax angle ≥ `min_deg`, then restores connectivity: any edge the gate dropped is
/// re-added (highest-parallax first, deterministic tiebreak) whenever its endpoints would
/// otherwise be disconnected. Because kept ∪ dropped = the full edge set, the result has
/// exactly the full graph's connectivity — no camera the full graph placed is ever
/// orphaned, so coverage cannot collapse. Rotation averaging is unaffected (it uses the
/// full set); this subset feeds the translation solve only.
fn parallax_filtered_translation_edges(
    workspace: &SfmWorkspace,
    edges: &[PairEdge],
    models: &[GeometricModel],
    min_deg: f64,
    view_count: usize,
) -> Result<(Vec<PairEdge>, Vec<GeometricModel>)> {
    // 1. Score every edge; classify by the gate. Edges whose proxy can't be computed are
    //    treated as informative (kept) so a missing proxy never drops a constraint.
    let mut kept = vec![false; edges.len()];
    let mut dropped: Vec<(f64, usize)> = Vec::new(); // (parallax_deg, edge_index)
    for (index, (edge, model)) in edges.iter().zip(models).enumerate() {
        let angle = edge_median_parallax_deg(workspace, edge, model);
        match angle {
            Some(angle) if angle < min_deg => dropped.push((angle, index)),
            _ => kept[index] = true,
        }
    }
    // 2. Re-add the highest-parallax dropped edges needed to match the full
    //    graph's connected components.
    // Deterministic: highest parallax first, tiebreak by ascending edge index.
    dropped.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let candidate_order: Vec<_> = dropped.into_iter().map(|(_, index)| index).collect();
    let endpoints: Vec<_> = edges
        .iter()
        .map(|edge| (edge.view_i.index(), edge.view_j.index()))
        .collect();
    let completion = complete_connectivity(view_count, &endpoints, &kept, &candidate_order)
        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    kept = completion.into_keep();

    let trans_edges: Vec<PairEdge> = edges
        .iter()
        .zip(&kept)
        .filter(|(_, keep)| **keep)
        .map(|(edge, _)| *edge)
        .collect();
    let trans_models: Vec<GeometricModel> = models
        .iter()
        .zip(&kept)
        .filter(|(_, keep)| **keep)
        .map(|(model, _)| model.clone())
        .collect();

    Ok((trans_edges, trans_models))
}

pub(crate) fn translation_geometric_models(
    edges: Vec<PairEdge>,
    models: Vec<GeometricModel>,
) -> Result<(Vec<PairEdge>, Vec<GeometricModel>)> {
    if edges.len() != models.len() {
        return Err(SfmError::InvalidInput(format!(
            "expected one geometric model per pair edge, got {} pairs and {} models",
            edges.len(),
            models.len()
        )));
    }
    let (translation_edges, translation_models): (Vec<_>, Vec<_>) = edges
        .into_iter()
        .zip(models)
        .filter(|(_, model)| model.relative_pose().is_some())
        .unzip();
    Ok((translation_edges, translation_models))
}

pub(crate) fn active_geometric_models(
    workspace: &SfmWorkspace,
) -> Result<(Vec<PairEdge>, Vec<GeometricModel>)> {
    let mut edges = Vec::new();
    let mut models = Vec::new();

    for pair_id in workspace.pair_graph().active_pair_ids() {
        let pair = workspace
            .pair_graph()
            .pair(pair_id)
            .ok_or(SfmError::InvalidId {
                kind: crate::PairId::kind(),
                index: pair_id.index(),
            })?;
        let model = workspace
            .geometric_model_store()
            .get(pair_id.index())
            .ok_or(SfmError::MissingPrerequisite {
                prerequisite: "geometric models",
                method: "verify_pairs",
            })?;
        edges.push(PairEdge {
            pair,
            view_i: pair.first(),
            view_j: pair.second(),
            confidence: workspace.pair_graph().confidence(pair_id).unwrap_or(1.0),
        });
        models.push(model.clone());
    }

    Ok((edges, models))
}

pub(super) fn report_from_estimates(
    active_pair_count: usize,
    rotations: &RotationEstimates,
    centers: &CameraCenters,
) -> GlobalAveragingReport {
    GlobalAveragingReport {
        method: GlobalAveragingMethod::SeparateRotationTranslation,
        active_pair_count,
        rotation_count: rotations.rotations().len(),
        center_count: centers.centers().len(),
        rotation_iterations: rotations.iterations,
        translation_iterations: centers.iterations,
        rotation_residual: rotations.residual.value(),
        translation_residual: centers.residual.value(),
        positioning_track_count: None,
        positioning_observation_count: None,
    }
}

pub(crate) fn rotation_only_averaging_report(
    active_pair_count: usize,
    rotations: &RotationEstimates,
) -> GlobalAveragingReport {
    GlobalAveragingReport {
        method: GlobalAveragingMethod::SeparateRotationTranslation,
        active_pair_count,
        rotation_count: rotations.rotations().len(),
        center_count: 0,
        rotation_iterations: rotations.iterations,
        translation_iterations: 0,
        rotation_residual: rotations.residual.value(),
        translation_residual: 0.0,
        positioning_track_count: None,
        positioning_observation_count: None,
    }
}

pub(crate) fn populate_pose_records(
    workspace: &mut SfmWorkspace,
    rotations: &RotationEstimates,
    centers: &CameraCenters,
) -> Result<()> {
    let mut poses = crate::workspace::store::SparseStore::new(workspace.pose_store().slot_count());
    for (view_id, center) in centers.centers() {
        let rotation = rotations.get(*view_id).ok_or_else(|| {
            SfmError::InvalidInput(format!(
                "translation averaging estimated view {view_id} without a rotation"
            ))
        })?;
        let pose_id = workspace.pose_id(*view_id).ok_or(SfmError::InvalidId {
            kind: ViewId::kind(),
            index: view_id.index(),
        })?;
        let candidate = PoseRecord {
            id: pose_id,
            pose: WorldToCameraPose::from_rotation_center(*rotation, Point3::from(*center))
                .map_err(|error| SfmError::InvalidInput(error.to_string()))?,
            component: Some(ComponentId::new(0)),
        };
        if let Some(existing) = poses.get(pose_id.index()) {
            if existing != &candidate {
                return Err(SfmError::InvalidInput(format!(
                    "views sharing {pose_id} received inconsistent global poses"
                )));
            }
        } else {
            poses.set(pose_id.index(), candidate)?;
        }
    }
    workspace.poses = poses;
    workspace.components = vec![crate::model::ReconstructionComponent::new(
        ComponentId::new(0),
        centers
            .centers()
            .iter()
            .map(|(view_id, _)| *view_id)
            .collect::<Vec<ViewId>>(),
        Vec::new(),
        crate::config::GaugeScalePolicy::Free,
    )];
    Ok(())
}
