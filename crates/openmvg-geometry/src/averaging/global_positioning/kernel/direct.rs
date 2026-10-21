// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic sparse Cholesky of the reduced camera systems: the alternation's scalar system
//! and the polish's 3x3-block system.
//!
//! Both systems eliminate every point, which leaves a camera system whose pattern is the set of
//! camera pairs that share a point or a relative direction. The alternation's system is scalar,
//! with the x, y and z coordinates as three right-hand sides of one factorization; its anchor
//! camera changes between iterations and keeps its row as an identity row, so one analysis serves
//! every anchor. It replaced conjugate gradients as the alternation's main solve: on BlendedMVS
//! scenes of 1,800-2,300 cameras and MegaDepth scenes of 430-720 cameras the global-positioning
//! solve took up to 25% less time at the same results, and the factorization has no iteration
//! cap to run out of. Camera graphs whose factor would exceed the bundle-adjustment workspace
//! budget keep conjugate gradients.
//!
//! The polish eliminates every point with its 3x3 block, which leaves a camera system whose block
//! pattern is the set of camera pairs that share a point or a relative direction. That pattern
//! does not change between Gauss-Newton steps, so it is ordered and analyzed once per workspace;
//! each damped attempt only refills the values and runs a numeric factorization and one triangular
//! solve. The position of every point-pair block is precomputed with the analysis: 12 bytes per
//! observation pair, one per Schur block product an attempt computes anyway. Finding the positions
//! by binary search instead made the solve 22-37% slower. Camera 0 fixes the translation gauge and
//! is left out of the system.
//!
//! Conjugate gradients on this system needed up to `3n + 50` matrix-free passes over every
//! observation per attempt and dominated the solve (92% of it at 289 cameras).
//!
//! ```text
//! camera pairs (once) -> lower pattern -> AMD ordering + symbolic analysis (once)
//! alternation iteration: scalar Laplacian, anchor as identity row -> LLT -> x, y, z
//! polish attempt: 3x3 Schur blocks -> LLT -> camera step
//! ```
//!
//! The factorization is the deterministic sparse Cholesky of `openmvg_ba::numerics::sparse`,
//! shared with bundle adjustment and rotation averaging, so the solve is bit-identical across
//! thread counts.

use std::collections::BTreeSet;
use std::mem::size_of;

use nalgebra::{Matrix3, Vector3};
use openmvg_ba::numerics::sparse::{SparseCholesky, SymmetricPattern};

use super::RIDGE;
use super::plan::GlobalPositioningPlan;
use super::workspace::CameraTerm;

/// Sparse Cholesky storage for the camera system of one plan, without camera 0.
pub(super) struct BlockCholesky {
    cholesky: SparseCholesky,
    values: Vec<f64>,
    /// Where each observation pair of each point lands, in [`Self::add_point_pair`] order.
    point_pair_slots: Vec<Slot>,
    /// Where each relative direction lands, in plan order.
    direction_slots: Vec<Slot>,
    topology: Topology,
    rhs: Vec<f64>,
}

/// Precomputed position of one camera block in the lower pattern.
#[derive(Clone, Copy)]
struct Slot {
    column: u32,
    /// Index among the column's strictly lower blocks, or [`DIAGONAL`] / [`SKIPPED`].
    position: u32,
    /// The block arrives as (column, row) and is stored transposed.
    transposed: bool,
}

const DIAGONAL: u32 = u32::MAX - 1;
const SKIPPED: u32 = u32::MAX;

impl BlockCholesky {
    /// Analyze the camera system of `plan`; `None` when it has no free camera, the analysis fails,
    /// or its storage cannot be allocated.
    pub(super) fn analyze(plan: &GlobalPositioningPlan) -> Option<Self> {
        let blocks = plan
            .camera_count
            .checked_sub(1)
            .filter(|&count| count > 0)?;
        // Without camera 0, block `b` is camera `b + 1`.
        let pairs = camera_pairs(plan)
            .into_iter()
            .filter(|&(column, _)| column != 0)
            .map(|(column, row)| (column - 1, row - 1))
            .collect::<Vec<_>>();
        // `pairs` holds (column block, row block) with row > column, sorted by column then row.
        let mut block_column_start = vec![0usize; blocks + 1];
        let mut block_rows = Vec::with_capacity(pairs.len());
        for &(column, row) in &pairs {
            block_column_start[column + 1] += 1;
            block_rows.push(row);
        }
        for column in 0..blocks {
            block_column_start[column + 1] += block_column_start[column];
        }
        let dimension = 3 * blocks;
        let mut column_pointer = Vec::with_capacity(dimension + 1);
        let mut row_index = Vec::new();
        column_pointer.push(0);
        for column in 0..blocks {
            let rows = &block_rows[block_column_start[column]..block_column_start[column + 1]];
            for axis in 0..3 {
                row_index.extend((axis..3).map(|row| 3 * column + row));
                for &row_block in rows {
                    row_index.extend((0..3).map(|row| 3 * row_block + row));
                }
                column_pointer.push(row_index.len());
            }
        }
        let value_count = row_index.len();
        let slot = |row_camera: usize, column_camera: usize| -> Option<Slot> {
            if row_camera == 0 || column_camera == 0 {
                return Some(Slot {
                    column: 0,
                    position: SKIPPED,
                    transposed: false,
                });
            }
            let transposed = row_camera < column_camera;
            let (row, column) = if transposed {
                (column_camera - 1, row_camera - 1)
            } else {
                (row_camera - 1, column_camera - 1)
            };
            let position = if row == column {
                DIAGONAL
            } else {
                let rows = &block_rows[block_column_start[column]..block_column_start[column + 1]];
                u32::try_from(rows.binary_search(&row).ok()?).ok()?
            };
            Some(Slot {
                column: u32::try_from(column).ok()?,
                position,
                transposed,
            })
        };
        let mut point_pair_slots = Vec::new();
        for point in 0..plan.point_count {
            let indices = plan.point_bearing_range(point);
            for (offset, &left) in indices.iter().enumerate() {
                for &right in &indices[offset..] {
                    point_pair_slots.push(slot(
                        plan.bearings[left].camera,
                        plan.bearings[right].camera,
                    )?);
                }
            }
        }
        let direction_slots = plan
            .directions
            .iter()
            .map(|edge| slot(edge.target, edge.source))
            .collect::<Option<Vec<_>>>()?;
        let mut cholesky =
            SparseCholesky::analyze(SymmetricPattern::new(dimension, column_pointer, row_index)?)?;
        if !cholesky.reserve(1) {
            return None;
        }
        Some(Self {
            cholesky,
            values: vec![0.0; value_count],
            point_pair_slots,
            direction_slots,
            topology: Topology::of(plan),
            rhs: vec![0.0; dimension],
        })
    }

    /// Whether this analysis was made for the topology of `plan`.
    pub(super) fn matches(&self, plan: &GlobalPositioningPlan) -> bool {
        self.topology.matches(plan)
    }

    /// Add the block of the `pair`-th point observation pair: pairs are numbered point by point,
    /// and within a point as `(left, right)` with `right` from `left` to the last observation.
    pub(super) fn add_point_pair(&mut self, pair: usize, block: &Matrix3<f64>) {
        let slot = self.point_pair_slots[pair];
        self.add_at(slot, block);
    }

    /// Add the (target, source) block of the `index`-th relative direction.
    pub(super) fn add_direction(&mut self, index: usize, block: &Matrix3<f64>) {
        let slot = self.direction_slots[index];
        self.add_at(slot, block);
    }

    fn add_at(&mut self, slot: Slot, block: &Matrix3<f64>) {
        if slot.position == SKIPPED {
            return;
        }
        let block = if slot.transposed {
            block.transpose()
        } else {
            *block
        };
        let column = slot.column as usize;
        let column_pointer = self.cholesky.pattern().column_pointers();
        if slot.position == DIAGONAL {
            for axis in 0..3 {
                let start = column_pointer[3 * column + axis];
                for entry in axis..3 {
                    self.values[start + entry - axis] += block[(entry, axis)];
                }
            }
            return;
        }
        let position = slot.position as usize;
        for axis in 0..3 {
            let start = column_pointer[3 * column + axis] + (3 - axis) + 3 * position;
            for entry in 0..3 {
                self.values[start + entry] += block[(entry, axis)];
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.values.fill(0.0);
    }

    /// Add `block` to the diagonal block of `camera`; camera 0 is not part of the system.
    pub(super) fn add_diagonal(&mut self, camera: usize, block: &Matrix3<f64>) {
        if camera == 0 {
            return;
        }
        self.add_at(
            Slot {
                column: u32::try_from(camera - 1).unwrap_or(u32::MAX),
                position: DIAGONAL,
                transposed: false,
            },
            block,
        );
    }

    /// Factor the assembled system and solve it for `rhs`, writing every camera's step into
    /// `step` (camera 0 receives zero). `false` when the system is not positive definite or the
    /// step is not finite.
    pub(super) fn solve(&mut self, rhs: &[Vector3<f64>], step: &mut [Vector3<f64>]) -> bool {
        if !self.cholesky.factor(&self.values) {
            return false;
        }
        for (camera, value) in rhs.iter().enumerate().skip(1) {
            self.rhs[3 * (camera - 1)..3 * camera].copy_from_slice(value.as_slice());
        }
        if !self.cholesky.solve_in_place(&mut self.rhs) {
            return false;
        }
        step[0] = Vector3::zeros();
        for (camera, output) in step.iter_mut().enumerate().skip(1) {
            *output = Vector3::from_column_slice(&self.rhs[3 * (camera - 1)..3 * camera]);
        }
        true
    }
}

/// Sparse Cholesky of the alternation's scalar camera system, over every camera.
///
/// The alternation picks a new anchor camera at every iteration and the dense path deletes its row
/// and column. Here the anchor keeps its place in the pattern with an identity row and no
/// couplings, so one analysis serves every anchor and the other cameras solve the reduced system.
/// The x, y and z coordinates share the matrix and are solved as three right-hand sides of one
/// factorization.
pub(super) struct ScalarCholesky {
    cholesky: SparseCholesky,
    values: Vec<f64>,
    /// Value index of each camera's diagonal entry.
    diagonal_slots: Vec<u32>,
    /// Value index of each observation pair of each point, in [`Self::solve`] order: point by
    /// point, and within a point as `(left, right)` with `right` after `left`.
    point_pair_slots: Vec<u32>,
    /// Value index of each relative direction, in plan order.
    direction_slots: Vec<u32>,
    topology: Topology,
    /// Three column-major right-hand sides of the pattern's dimension.
    rhs: Vec<f64>,
}

/// Inputs of one alternation iteration's scalar system.
pub(super) struct ScalarSystem<'a> {
    pub(super) camera_terms: &'a [CameraTerm],
    pub(super) point_weight: &'a [f64],
    pub(super) bearing_effective_weight: &'a [f64],
    pub(super) direction_effective_weight: &'a [f64],
    pub(super) anchor: usize,
}

impl ScalarCholesky {
    /// Analyze the scalar camera system of `plan`; `None` when it has fewer than two cameras, the
    /// analysis fails, or the factor, the pair table and the scratch together would exceed
    /// `budget_bytes`, the case of camera graphs that fill in close to dense.
    pub(super) fn analyze(plan: &GlobalPositioningPlan, budget_bytes: usize) -> Option<Self> {
        let cameras = plan.camera_count;
        if cameras < 2 {
            return None;
        }
        // The pair table is the first allocation that grows with track length; refuse it early.
        let pair_count = (0..plan.point_count)
            .map(|point| {
                let length = plan.point_bearing_range(point).len();
                length * length.saturating_sub(1) / 2
            })
            .try_fold(0usize, usize::checked_add)?;
        if pair_count.checked_mul(size_of::<u32>())? > budget_bytes {
            return None;
        }
        let pairs = camera_pairs(plan);
        let mut column_pointer = vec![0usize; cameras + 1];
        for &(column, _) in &pairs {
            column_pointer[column + 1] += 1;
        }
        for column in 0..cameras {
            // One diagonal entry per column, then its strictly lower rows.
            column_pointer[column + 1] += column_pointer[column] + 1;
        }
        let mut row_index = vec![0usize; column_pointer[cameras]];
        let mut next = column_pointer[..cameras].to_vec();
        for (column, slot) in next.iter_mut().enumerate() {
            row_index[*slot] = column;
            *slot += 1;
        }
        for &(column, row) in &pairs {
            row_index[next[column]] = row;
            next[column] += 1;
        }
        let value_count = row_index.len();
        if u32::try_from(value_count).is_err() {
            return None;
        }
        let slot = |left: usize, right: usize| -> Option<u32> {
            let (column, row) = (left.min(right), left.max(right));
            let rows = &row_index[column_pointer[column] + 1..column_pointer[column + 1]];
            let position = column_pointer[column] + 1 + rows.binary_search(&row).ok()?;
            u32::try_from(position).ok()
        };
        let mut point_pair_slots = Vec::new();
        point_pair_slots.try_reserve_exact(pair_count).ok()?;
        for point in 0..plan.point_count {
            let indices = plan.point_bearing_range(point);
            for (offset, &left) in indices.iter().enumerate() {
                for &right in &indices[offset + 1..] {
                    point_pair_slots.push(slot(
                        plan.bearings[left].camera,
                        plan.bearings[right].camera,
                    )?);
                }
            }
        }
        let direction_slots = plan
            .directions
            .iter()
            .map(|edge| slot(edge.source, edge.target))
            .collect::<Option<Vec<_>>>()?;
        let diagonal_slots = column_pointer[..cameras]
            .iter()
            .map(|&start| u32::try_from(start).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut cholesky =
            SparseCholesky::analyze(SymmetricPattern::new(cameras, column_pointer, row_index)?)?;
        let fixed_bytes =
            pair_count * size_of::<u32>() + (value_count + 3 * cameras) * size_of::<f64>();
        let factor_bytes = cholesky
            .factor_value_count()
            .checked_mul(size_of::<f64>())?;
        if fixed_bytes.checked_add(factor_bytes)? > budget_bytes
            || !cholesky.reserve(3)
            || fixed_bytes.checked_add(cholesky.reserved_bytes())? > budget_bytes
        {
            return None;
        }
        Some(Self {
            cholesky,
            values: vec![0.0; value_count],
            diagonal_slots,
            point_pair_slots,
            direction_slots,
            topology: Topology::of(plan),
            rhs: vec![0.0; 3 * cameras],
        })
    }

    /// Whether this analysis was made for the topology of `plan`.
    pub(super) fn matches(&self, plan: &GlobalPositioningPlan) -> bool {
        self.topology.matches(plan)
    }

    /// Assemble the system of `system` as the dense path does (diagonal plus [`RIDGE`], couplings
    /// of every point pair and relative direction), factor it, and write every camera's solution
    /// into `output`, zero for the anchor. `false` when the matrix is not positive definite or the
    /// solution is not finite.
    pub(super) fn solve(
        &mut self,
        plan: &GlobalPositioningPlan,
        system: &ScalarSystem<'_>,
        output: &mut [Vector3<f64>],
    ) -> bool {
        let cameras = plan.camera_count;
        let anchor = system.anchor;
        self.values.fill(0.0);
        for (camera, &slot) in self.diagonal_slots.iter().enumerate() {
            self.values[slot as usize] = if camera == anchor {
                1.0
            } else {
                system.camera_terms[camera].diagonal + RIDGE
            };
        }
        let mut pair = 0usize;
        for point in 0..plan.point_count {
            let indices = plan.point_bearing_range(point);
            let point_weight = system.point_weight[point];
            if point_weight < 1e-12 {
                pair += indices.len() * indices.len().saturating_sub(1) / 2;
                continue;
            }
            for (offset, &left) in indices.iter().enumerate() {
                for &right in &indices[offset + 1..] {
                    let (left_camera, right_camera) =
                        (plan.bearings[left].camera, plan.bearings[right].camera);
                    if left_camera != anchor && right_camera != anchor {
                        self.values[self.point_pair_slots[pair] as usize] -= system
                            .bearing_effective_weight[left]
                            * system.bearing_effective_weight[right]
                            / point_weight;
                    }
                    pair += 1;
                }
            }
        }
        for (index, edge) in plan.directions.iter().enumerate() {
            if edge.source != anchor && edge.target != anchor {
                self.values[self.direction_slots[index] as usize] -=
                    system.direction_effective_weight[index];
            }
        }
        if !self.cholesky.factor(&self.values) {
            return false;
        }
        for (camera, term) in system.camera_terms.iter().enumerate() {
            let value = if camera == anchor {
                Vector3::zeros()
            } else {
                term.right_hand_side
            };
            for axis in 0..3 {
                self.rhs[axis * cameras + camera] = value[axis];
            }
        }
        if !self.cholesky.solve_in_place(&mut self.rhs) {
            return false;
        }
        for (camera, solution) in output.iter_mut().enumerate() {
            *solution = if camera == anchor {
                Vector3::zeros()
            } else {
                Vector3::new(
                    self.rhs[camera],
                    self.rhs[cameras + camera],
                    self.rhs[2 * cameras + camera],
                )
            };
        }
        true
    }
}

/// The plan topology an analysis was computed for; a workspace reused for another plan of the same
/// shape must analyze again.
struct Topology {
    point_offsets: Vec<usize>,
    observing_cameras: Vec<usize>,
    direction_endpoints: Vec<(usize, usize)>,
}

impl Topology {
    fn of(plan: &GlobalPositioningPlan) -> Self {
        Self {
            point_offsets: plan.point_offsets.clone(),
            observing_cameras: observing_cameras(plan).collect(),
            direction_endpoints: plan
                .directions
                .iter()
                .map(|edge| (edge.source, edge.target))
                .collect(),
        }
    }

    fn matches(&self, plan: &GlobalPositioningPlan) -> bool {
        self.point_offsets == plan.point_offsets
            && self
                .observing_cameras
                .iter()
                .copied()
                .eq(observing_cameras(plan))
            && self.direction_endpoints.iter().copied().eq(plan
                .directions
                .iter()
                .map(|edge| (edge.source, edge.target)))
    }
}

/// Each point's observing cameras, point by point in plan order.
fn observing_cameras(plan: &GlobalPositioningPlan) -> impl Iterator<Item = usize> + '_ {
    plan.point_bearings
        .iter()
        .map(|&index| plan.bearings[index].camera)
}

/// Distinct camera pairs `(lower, higher)` that share a point or a relative direction, sorted.
fn camera_pairs(plan: &GlobalPositioningPlan) -> BTreeSet<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    let mut insert = |a: usize, b: usize| {
        if a != b {
            pairs.insert((a.min(b), a.max(b)));
        }
    };
    for point in 0..plan.point_count {
        let indices = plan.point_bearing_range(point);
        for (offset, &left) in indices.iter().enumerate() {
            for &right in &indices[offset + 1..] {
                insert(plan.bearings[left].camera, plan.bearings[right].camera);
            }
        }
    }
    for edge in &plan.directions {
        insert(edge.source, edge.target);
    }
    pairs
}
