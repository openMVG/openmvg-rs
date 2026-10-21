// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked immutable BATA measurements and compact flat topology.

use nalgebra::{DMatrix, Vector3};
use std::collections::BTreeSet;

use super::api::{
    BataOptions, GlobalPositioningError, GlobalPositioningProblem, GlobalPositioningReport,
    PositioningObservation,
};
use super::solve::solve_into;
use super::workspace::GlobalPositioningWorkspace;
use super::{DENSE_DIRECT_MAX_CAMERAS, MATERIALIZED_PAIR_INCIDENCE_FACTOR};

#[derive(Clone, Copy, Debug)]
pub(super) struct Bearing {
    pub(super) camera: usize,
    pub(super) point: usize,
    pub(super) ray: Vector3<f64>,
    pub(super) base_weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Direction {
    pub(super) source: usize,
    pub(super) target: usize,
    pub(super) direction: Vector3<f64>,
    pub(super) base_weight: f64,
}

struct CompiledCouplings {
    endpoints: Vec<(usize, usize)>,
    point_pair_offsets: Vec<usize>,
    point_pair_slots: Vec<usize>,
    direction_slots: Vec<usize>,
}

/// Immutable normalized BATA measurements and prepared flat topology.
///
/// Preparation performs all index, weight, duplicate, coverage, and connectivity
/// checks. Relative-direction family mass balancing is also frozen here, so it
/// cannot be accidentally applied twice by the numerical loop.
pub struct GlobalPositioningPlan {
    pub(super) camera_count: usize,
    pub(super) point_count: usize,
    pub(super) bearings: Vec<Bearing>,
    pub(super) directions: Vec<Direction>,
    pub(super) point_offsets: Vec<usize>,
    pub(super) point_bearings: Vec<usize>,
    pub(super) coupling_endpoints: Vec<(usize, usize)>,
    pub(super) point_pair_offsets: Vec<usize>,
    pub(super) point_pair_slots: Vec<usize>,
    pub(super) direction_coupling_slots: Vec<usize>,
    pub(super) materialized_couplings: bool,
    pub(super) options: BataOptions,
}

impl std::fmt::Debug for GlobalPositioningPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GlobalPositioningPlan")
            .field("camera_count", &self.camera_count)
            .field("point_count", &self.point_count)
            .field("bearing_count", &self.bearings.len())
            .field("direction_count", &self.directions.len())
            .field("materialized_couplings", &self.materialized_couplings)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl GlobalPositioningPlan {
    /// Validate and normalize one problem, and compile its point-major topology.
    ///
    /// Measurement and topology errors are returned explicitly. Plan preparation
    /// uses standard Rust collection allocation, so process-level out-of-memory
    /// behavior remains allocator-defined; reusable numerical workspace allocation
    /// is separately fallible through [`Self::try_new_workspace`].
    pub fn try_new(
        problem: GlobalPositioningProblem<'_>,
        options: BataOptions,
    ) -> Result<Self, GlobalPositioningError> {
        let options = options.validate()?;
        if problem.camera_count < 2 {
            return Err(GlobalPositioningError::TooFewCameras {
                provided: problem.camera_count,
            });
        }
        problem
            .camera_count
            .checked_add(problem.point_count)
            .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?;
        if problem.observations.is_empty() {
            return Err(GlobalPositioningError::EmptyProblem);
        }

        let mut camera_counts = vec![0usize; problem.camera_count];
        let mut point_counts = vec![0usize; problem.point_count];
        let mut bearing_pairs = BTreeSet::new();
        let mut direction_pairs = BTreeSet::new();
        let mut bearings = Vec::with_capacity(problem.observations.len());
        let mut input_directions = Vec::with_capacity(problem.observations.len());
        let mut track_mass = 0.0;
        let mut direction_mass = 0.0;
        let mut bearing_index = 0usize;
        let mut direction_index = 0usize;
        for observation in problem.observations {
            match *observation {
                PositioningObservation::Bearing {
                    camera,
                    point,
                    world_bearing,
                    weight,
                } => {
                    let index = bearing_index;
                    bearing_index += 1;
                    if camera >= problem.camera_count {
                        return Err(GlobalPositioningError::BearingCameraOutOfBounds {
                            observation: index,
                        });
                    }
                    if point >= problem.point_count {
                        return Err(GlobalPositioningError::BearingPointOutOfBounds {
                            observation: index,
                        });
                    }
                    if !weight.is_finite() || weight <= 0.0 {
                        return Err(GlobalPositioningError::InvalidBearingWeight {
                            observation: index,
                        });
                    }
                    let norm = world_bearing.norm();
                    if !norm.is_finite() || norm <= 1e-12 {
                        return Err(GlobalPositioningError::InvalidBearingVector {
                            observation: index,
                        });
                    }
                    if !bearing_pairs.insert((camera, point)) {
                        return Err(GlobalPositioningError::DuplicateBearing {
                            observation: index,
                        });
                    }
                    camera_counts[camera] += 1;
                    point_counts[point] += 1;
                    track_mass += weight;
                    bearings.push(Bearing {
                        camera,
                        point,
                        ray: world_bearing / norm,
                        base_weight: weight,
                    });
                }
                PositioningObservation::RelativeDirection {
                    source,
                    target,
                    source_to_target,
                    weight,
                } => {
                    let index = direction_index;
                    direction_index += 1;
                    if source >= problem.camera_count || target >= problem.camera_count {
                        return Err(GlobalPositioningError::DirectionCameraOutOfBounds {
                            direction: index,
                        });
                    }
                    if source == target {
                        return Err(GlobalPositioningError::DirectionSelfLoop { direction: index });
                    }
                    if !weight.is_finite() || weight <= 0.0 {
                        return Err(GlobalPositioningError::InvalidDirectionWeight {
                            direction: index,
                        });
                    }
                    let norm = source_to_target.norm();
                    if !norm.is_finite() || norm <= 1e-12 {
                        return Err(GlobalPositioningError::InvalidDirectionVector {
                            direction: index,
                        });
                    }
                    let pair = (source.min(target), source.max(target));
                    if !direction_pairs.insert(pair) {
                        return Err(GlobalPositioningError::DuplicateDirection {
                            direction: index,
                        });
                    }
                    direction_mass += weight;
                    input_directions.push((source, target, source_to_target, weight));
                }
            }
        }
        if !track_mass.is_finite() || !direction_mass.is_finite() {
            return Err(GlobalPositioningError::InvalidOptions);
        }

        let direction_multiplier = if options.direction_reweight == 0.0 {
            0.0
        } else if !bearings.is_empty() && !input_directions.is_empty() {
            options.direction_reweight * track_mass / direction_mass
        } else {
            options.direction_reweight
        };
        if !direction_multiplier.is_finite() {
            return Err(GlobalPositioningError::InvalidOptions);
        }

        let mut directions = Vec::with_capacity(input_directions.len());
        if direction_multiplier > 0.0 {
            for (source, target, source_to_target, weight) in input_directions {
                let norm = source_to_target.norm();
                let base_weight = weight * direction_multiplier;
                if !base_weight.is_finite() || base_weight <= 0.0 {
                    return Err(GlobalPositioningError::InvalidOptions);
                }
                camera_counts[source] += 1;
                camera_counts[target] += 1;
                directions.push(Direction {
                    source,
                    target,
                    direction: source_to_target / norm,
                    base_weight,
                });
            }
        }

        if bearings.is_empty() && directions.is_empty() {
            return Err(GlobalPositioningError::EmptyProblem);
        }
        if let Some(camera) = camera_counts.iter().position(|count| *count == 0) {
            return Err(GlobalPositioningError::UnconstrainedCamera { camera });
        }
        if let Some(point) = point_counts.iter().position(|count| *count < 2) {
            return Err(GlobalPositioningError::UnderconstrainedPoint { point });
        }
        let constraint_count = bearings.len() as u128 + directions.len() as u128;
        let constraint_rows = 2u128 * constraint_count;
        let unknown_dofs = 3u128 * (problem.camera_count + problem.point_count) as u128;
        if constraint_rows + 4 < unknown_dofs {
            return Err(GlobalPositioningError::NecessarilyUnderdetermined);
        }
        if !is_connected(
            problem.camera_count,
            problem.point_count,
            &bearings,
            &directions,
        ) {
            return Err(GlobalPositioningError::Disconnected);
        }
        if !has_parallel_rigidity(
            problem.camera_count,
            problem.point_count,
            &bearings,
            &directions,
        ) {
            return Err(GlobalPositioningError::NotParallelRigid);
        }

        let point_offset_count = problem
            .point_count
            .checked_add(1)
            .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?;
        let mut point_offsets = vec![0usize; point_offset_count];
        for bearing in &bearings {
            point_offsets[bearing.point + 1] += 1;
        }
        for point in 0..problem.point_count {
            point_offsets[point + 1] += point_offsets[point];
        }
        let mut next = point_offsets[..problem.point_count].to_vec();
        let mut point_bearings = vec![0usize; bearings.len()];
        for (index, bearing) in bearings.iter().enumerate() {
            point_bearings[next[bearing.point]] = index;
            next[bearing.point] += 1;
        }

        let pair_incidence: usize = point_counts
            .iter()
            .try_fold(0usize, |sum, count| {
                count
                    .checked_mul(count.saturating_sub(1))
                    .and_then(|pairs| pairs.checked_div(2))
                    .and_then(|pairs| sum.checked_add(pairs))
            })
            .ok_or(GlobalPositioningError::WorkspaceAllocationFailed)?;
        let materialized_couplings = problem.camera_count > DENSE_DIRECT_MAX_CAMERAS
            && pair_incidence
                <= MATERIALIZED_PAIR_INCIDENCE_FACTOR
                    .saturating_mul(bearings.len().saturating_add(directions.len()));
        let compiled_couplings = if materialized_couplings {
            compile_couplings(
                problem.point_count,
                &bearings,
                &directions,
                &point_offsets,
                &point_bearings,
                pair_incidence,
            )?
        } else {
            CompiledCouplings {
                endpoints: Vec::new(),
                point_pair_offsets: Vec::new(),
                point_pair_slots: Vec::new(),
                direction_slots: Vec::new(),
            }
        };
        Ok(Self {
            camera_count: problem.camera_count,
            point_count: problem.point_count,
            bearings,
            directions,
            point_offsets,
            point_bearings,
            coupling_endpoints: compiled_couplings.endpoints,
            point_pair_offsets: compiled_couplings.point_pair_offsets,
            point_pair_slots: compiled_couplings.point_pair_slots,
            direction_coupling_slots: compiled_couplings.direction_slots,
            materialized_couplings,
            options,
        })
    }

    /// Whether this plan retained compact scalar camera couplings instead of
    /// recomputing point-Schur products during every PCG matrix application.
    /// Dense plans return `false` because their matrix is assembled directly.
    #[must_use]
    #[cfg(test)]
    pub const fn uses_materialized_couplings(&self) -> bool {
        self.materialized_couplings
    }

    /// Allocate reusable storage sized exactly for this problem shape.
    pub fn try_new_workspace(&self) -> Result<GlobalPositioningWorkspace, GlobalPositioningError> {
        GlobalPositioningWorkspace::try_new(self)
    }

    /// Solve transactionally into caller-owned dense camera and point arrays.
    pub fn solve_into(
        &self,
        camera_centers: &mut [Vector3<f64>],
        points: &mut [Vector3<f64>],
        workspace: &mut GlobalPositioningWorkspace,
    ) -> Result<GlobalPositioningReport, GlobalPositioningError> {
        solve_into(self, camera_centers, points, workspace)
    }

    pub(super) fn point_bearing_range(&self, point: usize) -> &[usize] {
        &self.point_bearings[self.point_offsets[point]..self.point_offsets[point + 1]]
    }
}

fn is_connected(
    camera_count: usize,
    point_count: usize,
    bearings: &[Bearing],
    directions: &[Direction],
) -> bool {
    let mut parent: Vec<_> = (0..camera_count + point_count).collect();
    let mut size = vec![1usize; parent.len()];
    for bearing in bearings {
        union(
            &mut parent,
            &mut size,
            bearing.camera,
            camera_count + bearing.point,
        );
    }
    for edge in directions {
        union(&mut parent, &mut size, edge.source, edge.target);
    }
    let root = find(&mut parent, 0);
    (1..parent.len()).all(|node| find(&mut parent, node) == root)
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn union(parent: &mut [usize], size: &mut [usize], left: usize, right: usize) {
    let mut left_root = find(parent, left);
    let mut right_root = find(parent, right);
    if left_root == right_root {
        return;
    }
    if size[left_root] < size[right_root] {
        std::mem::swap(&mut left_root, &mut right_root);
    }
    parent[right_root] = left_root;
    size[left_root] += size[right_root];
}

fn compile_couplings(
    point_count: usize,
    bearings: &[Bearing],
    directions: &[Direction],
    point_offsets: &[usize],
    point_bearings: &[usize],
    pair_incidence: usize,
) -> Result<CompiledCouplings, GlobalPositioningError> {
    let mut unique = BTreeSet::new();
    for point in 0..point_count {
        let indices = &point_bearings[point_offsets[point]..point_offsets[point + 1]];
        for left in 0..indices.len() {
            let left_camera = bearings[indices[left]].camera;
            for &right_index in &indices[left + 1..] {
                let right_camera = bearings[right_index].camera;
                unique.insert((left_camera.min(right_camera), left_camera.max(right_camera)));
            }
        }
    }
    for edge in directions {
        unique.insert((edge.source.min(edge.target), edge.source.max(edge.target)));
    }
    let mut endpoints: Vec<_> = unique.into_iter().collect();
    endpoints.sort_unstable();
    let mut point_pair_offsets = Vec::with_capacity(point_count + 1);
    let mut point_pair_slots = Vec::with_capacity(pair_incidence);
    point_pair_offsets.push(0);
    for point in 0..point_count {
        let indices = &point_bearings[point_offsets[point]..point_offsets[point + 1]];
        for left in 0..indices.len() {
            let left_camera = bearings[indices[left]].camera;
            for &right_index in &indices[left + 1..] {
                let right_camera = bearings[right_index].camera;
                let pair = (left_camera.min(right_camera), left_camera.max(right_camera));
                point_pair_slots.push(
                    endpoints
                        .binary_search(&pair)
                        .map_err(|_| GlobalPositioningError::WorkspaceAllocationFailed)?,
                );
            }
        }
        point_pair_offsets.push(point_pair_slots.len());
    }
    let mut direction_slots = Vec::with_capacity(directions.len());
    for edge in directions {
        let pair = (edge.source.min(edge.target), edge.source.max(edge.target));
        direction_slots.push(
            endpoints
                .binary_search(&pair)
                .map_err(|_| GlobalPositioningError::WorkspaceAllocationFailed)?,
        );
    }
    Ok(CompiledCouplings {
        endpoints,
        point_pair_offsets,
        point_pair_slots,
        direction_slots,
    })
}

fn has_parallel_rigidity(
    camera_count: usize,
    point_count: usize,
    bearings: &[Bearing],
    directions: &[Direction],
) -> bool {
    const EXACT_MAX_VARIABLES: usize = 64;
    let variable_count = camera_count + point_count;
    if variable_count > EXACT_MAX_VARIABLES {
        return true;
    }
    let row_count = 2 * (bearings.len() + directions.len());
    let column_count = 3 * variable_count;
    let mut rigidity = DMatrix::zeros(row_count, column_count);
    let mut row = 0usize;
    for observation in bearings {
        let (first, second) = orthogonal_basis(observation.ray);
        for basis in [first, second] {
            write_rigidity_row(
                &mut rigidity,
                row,
                observation.camera,
                camera_count + observation.point,
                basis,
            );
            row += 1;
        }
    }
    for edge in directions {
        let (first, second) = orthogonal_basis(edge.direction);
        for basis in [first, second] {
            write_rigidity_row(&mut rigidity, row, edge.source, edge.target, basis);
            row += 1;
        }
    }
    let singular_values = rigidity.svd(false, false).singular_values;
    let largest = singular_values.iter().copied().fold(0.0, f64::max);
    let threshold = largest.max(1.0) * 1e-9;
    let rank = singular_values
        .iter()
        .filter(|value| **value > threshold)
        .count();
    rank >= column_count.saturating_sub(4)
}

fn orthogonal_basis(direction: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let axis = if direction.x.abs() < 0.8 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let first = direction.cross(&axis).normalize();
    let second = direction.cross(&first).normalize();
    (first, second)
}

fn write_rigidity_row(
    matrix: &mut DMatrix<f64>,
    row: usize,
    source: usize,
    target: usize,
    basis: Vector3<f64>,
) {
    for component in 0..3 {
        matrix[(row, 3 * source + component)] = -basis[component];
        matrix[(row, 3 * target + component)] = basis[component];
    }
}
