// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wu, Bhayani and Heikkilä's conic-transformation P3P solver.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                            Wu P3P                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: triangle side ratios + unit bearings
//! Canonical conic transform
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Build and solve scaled quartic                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! Build Wu's canonical conic coefficients and solve the quartic with the
//! general Ferrari routine, including degree fallback and root polishing.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Depths, pose, and singular fallback                          │
//! └──────────────────────────────────────────────────────────────────────┘
//! Recover positive depths, refine them, construct R,t, and use Ke only for
//! the equilateral-chart singularity.
//! Output: up to 4 rigid poses after cheirality validation
//! ```
//!
//! Only Wu's geometric reduction is specific to this solver. Its quartic goes
//! through the crate's general quartic routine, which handles any coefficient
//! scale; Ke and Nakano use a lighter routine for their already-conditioned
//! coefficients.
//!
//! # References
//!
//! - H. Wu, S. Bhayani, J. Heikkilä, "A Conic Transformation Approach for Solving the
//!   Perspective-Three-Point Problem", WACV 2025, pp. 3237–3245. arXiv:2504.01620.

use crate::SolverResult;
use crate::solver_traits::{MultiSolutionSolver, Solver};
use nalgebra::{Point3, UnitVector3};

use super::common::{
    P3pInput, P3pSolutions, SolverP3pSolutions, angular_residual, exact_indexed_sample,
    exact_sample, refine_depths, rotation_from_edges, world_basis_inverse,
};
use super::ke::solve_validated as solve_ke_validated;
use crate::{
    pose::{BearingToPoint, SolverPose},
    shared::polynomial::quartic_real,
};

/// Wu et al.'s conic-transformation calibrated P3P solver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WuP3pSolver;

impl WuP3pSolver {
    /// Solve one exact minimal sample without heap allocation.
    pub(crate) fn solve_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
    ) -> P3pSolutions {
        P3pSolutions::from_solver(Self::solve_solver_minimal(world, bearings))
    }

    fn solve_solver_minimal(
        world: [Point3<f64>; 3],
        bearings: [UnitVector3<f64>; 3],
    ) -> SolverP3pSolutions {
        let Some(input) = P3pInput::new(world, bearings) else {
            return SolverP3pSolutions::default();
        };
        solve_validated(&input)
    }
}

impl Solver for WuP3pSolver {
    type Datum = BearingToPoint;
    type Model = SolverPose<f64>;
    type Config = ();
    type Unit = openmvg_core::Radians;

    fn min_sample_size() -> usize {
        3
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        angular_residual(model, datum)
    }
}

impl MultiSolutionSolver for WuP3pSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_sample(data)?;
        Ok(Self::solve_solver_minimal(world, bearings).into_vec())
    }

    fn solve_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Vec<Self::Model>> {
        let (world, bearings) = exact_indexed_sample(indices, data)?;
        Ok(Self::solve_solver_minimal(world, bearings).into_vec())
    }
}

fn solve_validated(input: &P3pInput) -> SolverP3pSolutions {
    let mut output = SolverP3pSolutions::default();
    let mut world = input.world;
    let mut bearings = input.bearings;
    let side01 = (world[0] - world[1]).norm_squared();
    let side02 = (world[0] - world[2]).norm_squared();
    let side12 = (world[1] - world[2]).norm_squared();
    let singular_tolerance = 64.0 * f64::EPSILON;
    if (1.0 - side01 / side12).abs() <= singular_tolerance {
        let cyclic_score = (1.0 - side12 / side02).abs();
        let reverse_cyclic_score = (1.0 - side02 / side01).abs();
        if cyclic_score.max(reverse_cyclic_score) <= singular_tolerance {
            let fallback = solve_ke_validated(input);
            return input.fitting_candidates(&fallback);
        }
        if cyclic_score >= reverse_cyclic_score {
            world = [world[1], world[2], world[0]];
            bearings = [bearings[1], bearings[2], bearings[0]];
        } else {
            world = [world[2], world[0], world[1]];
            bearings = [bearings[2], bearings[0], bearings[1]];
        }
    }

    let [x1, x2, x3] = world;
    let [y1, y2, y3] = bearings;
    let m12 = y1.dot(&y2).clamp(-1.0, 1.0);
    let m13 = y1.dot(&y3).clamp(-1.0, 1.0);
    let m23 = y2.dot(&y3).clamp(-1.0, 1.0);
    let d12 = x1 - x2;
    let d13 = x1 - x3;
    let a12 = d12.norm_squared();
    let a13 = d13.norm_squared();
    let a23 = (x2 - x3).norm_squared();
    if a23 <= 0.0 {
        return output;
    }
    let a = a12 / a23;
    let b = a13 / a23;
    let b1 = -2.0 * m12;
    let c1 = 1.0 - a;
    let e1 = 2.0 * a * m23;
    let f2 = 1.0 - b;

    let p3_0 = -a.sqrt();
    let mut p1_0 = -p3_0;
    let mut p1_1 = 0.0;
    let mut p2_0 = p1_0;
    let geometry_scale = 1.0 + a.abs() + b1.abs() + e1.abs();
    if c1.abs() <= 64.0 * f64::EPSILON * geometry_scale {
        return output;
    }
    let c1_inverse = c1.recip();
    let mut p2_1 = -(b1 * p1_0 + e1) * c1_inverse;

    if p2_1.abs() < 0.05 {
        let discriminant = b1.mul_add(b1, -4.0 * c1);
        let discriminant_tolerance =
            32.0 * f64::EPSILON * (b1 * b1 + 4.0 * c1.abs()).max(f64::MIN_POSITIVE);
        if discriminant < -discriminant_tolerance {
            let value = (4.0 * c1 * a + e1 * e1).max(0.0).sqrt();
            p2_0 = 0.0;
            p2_1 = 0.5 * c1_inverse * (-e1 - value);
        } else if discriminant > discriminant_tolerance {
            let xk = p3_0 - 1.0;
            let linear = b1 * xk + e1;
            let radicand = linear * linear - 4.0 * c1 * (xk * xk - a);
            let Some(span) =
                nonnegative_sqrt(radicand, linear * linear + (4.0 * c1 * (xk * xk - a)).abs())
            else {
                return output;
            };
            p2_0 = xk;
            p2_1 = 0.5 * c1_inverse * (-linear - span);
        } else {
            let Some(xk) = find_xk(-4.0 * c1 + b1 * b1, 2.0 * b1 * e1, 4.0 * c1 * a + e1 * e1)
            else {
                return output;
            };
            let linear = b1 * xk + e1;
            let radicand = linear * linear - 4.0 * c1 * (xk * xk - a);
            let Some(span) =
                nonnegative_sqrt(radicand, linear * linear + (4.0 * c1 * (xk * xk - a)).abs())
            else {
                return output;
            };
            p1_0 = xk;
            p1_1 = 0.5 * c1_inverse * (-linear + span);
            p2_0 = xk;
            p2_1 = 0.5 * c1_inverse * (-linear - span);
        }
    }

    let half_b1 = 0.5 * b1;
    let half_e1 = 0.5 * e1;
    let aa1 = p1_0 + half_b1 * p1_1;
    let aa2 = half_b1 * p1_0 + c1 * p1_1 + half_e1;
    let aa3 = half_e1 * p1_1 - a;
    let bb1 = p2_0 + half_b1 * p2_1;
    let bb2 = half_b1 * p2_0 + c1 * p2_1 + half_e1;
    let bb3 = half_e1 * p2_1 - a;

    let pp00 = aa2 * bb3 - aa3 * bb2;
    let pp01 = p1_0;
    let pp02 = p2_0;
    let pp10 = aa3 * bb1 - aa1 * bb3;
    let pp11 = p1_1;
    let pp12 = p2_1;
    let pp20 = aa1 * bb2 - aa2 * bb1;
    let determinant = if p1_0 == 0.0 && p2_0 == 0.0 {
        pp00 * (pp11 - pp12)
    } else {
        pp00 * (pp11 - pp12) - pp01 * (pp10 - pp12 * pp20) + pp02 * (pp10 - pp11 * pp20)
    };
    if !determinant.is_finite() || determinant.abs() <= 64.0 * f64::EPSILON * geometry_scale {
        return output;
    }
    let determinant_inverse = determinant.recip();

    let pi00 = pp11 - pp12;
    let pi02 = pp01 * pp12 - pp02 * pp11;
    let pi10 = pp12 * pp20 - pp10;
    let pi12 = pp02 * pp10 - pp00 * pp12;
    let pi20 = pp10 - pp11 * pp20;
    let pi22 = pp00 * pp11 - pp01 * pp10;
    let l0 = (pi00 * p3_0 + pi02) * determinant_inverse;
    let l1 = (pi10 * p3_0 + pi12) * determinant_inverse;
    let l2 = (pi20 * p3_0 + pi22) * determinant_inverse;

    let hh00 = l0 * pp00;
    let hh01 = l1 * pp01;
    let hh02 = l2 * pp02;
    let hh10 = l0 * pp10;
    let hh11 = l1 * pp11;
    let hh12 = l2 * pp12;
    let hh20 = l0 * pp20;
    let hh21 = l1;
    let hh22 = l2;

    let half_e2 = b * m23;
    let cc00 = hh00 - hh20 * m13;
    let cc01 = -hh10 * b + hh20 * half_e2;
    let cc02 = hh10 * half_e2 + hh20 * f2 - hh00 * m13;
    let cc10 = hh01 - hh21 * m13;
    let cc11 = -hh11 * b + hh21 * half_e2;
    let cc12 = hh11 * half_e2 + hh21 * f2 - hh01 * m13;
    let cc20 = hh02 - hh22 * m13;
    let cc21 = -hh12 * b + hh22 * half_e2;
    let cc22 = hh12 * half_e2 + hh22 * f2 - hh02 * m13;

    let coefficients = if p1_0 == 0.0 && p2_0 == 0.0 {
        [
            cc11 * hh11 + cc12 * hh21,
            cc01 * hh11 + cc02 * hh21 + cc10 * hh00 + cc11 * hh10 + cc12 * hh20,
            cc00 * hh00
                + cc01 * hh10
                + cc02 * hh20
                + cc11 * hh12
                + cc12 * hh22
                + cc21 * hh11
                + cc22 * hh21,
            cc01 * hh12 + cc02 * hh22 + cc20 * hh00 + cc21 * hh10 + cc22 * hh20,
            cc21 * hh12 + cc22 * hh22,
        ]
    } else {
        [
            cc10 * hh01 + cc11 * hh11 + cc12 * hh21,
            cc00 * hh01 + cc01 * hh11 + cc02 * hh21 + cc10 * hh00 + cc11 * hh10 + cc12 * hh20,
            cc00 * hh00
                + cc01 * hh10
                + cc02 * hh20
                + cc10 * hh02
                + cc11 * hh12
                + cc12 * hh22
                + cc20 * hh01
                + cc21 * hh11
                + cc22 * hh21,
            cc00 * hh02 + cc01 * hh12 + cc02 * hh22 + cc20 * hh00 + cc21 * hh10 + cc22 * hh20,
            cc20 * hh02 + cc21 * hh12 + cc22 * hh22,
        ]
    };
    let mut roots = [0.0; 4];
    let root_count = quartic_real(coefficients, &mut roots);
    let Some(world_inverse) = world_basis_inverse(d12, d13) else {
        return output;
    };

    for root in roots.into_iter().take(root_count) {
        let root_squared = root * root;
        let solution0 = hh00 * root + hh01 * root_squared + hh02;
        let solution1 = hh10 * root + hh11 * root_squared + hh12;
        let solution2 = hh20 * root + hh21 * root_squared + hh22;
        if solution0 * solution1 <= 0.0 || solution0 * solution2 <= 0.0 {
            continue;
        }
        let denominator =
            solution0 * solution0 + solution1 * solution1 - 2.0 * m12 * solution0 * solution1;
        if denominator <= 0.0 || !denominator.is_finite() {
            continue;
        }
        let depth_scale = (a12 / denominator).sqrt();
        let sign = if solution2 < 0.0 { -1.0 } else { 1.0 };
        let mut depths = nalgebra::Vector3::new(
            sign * solution0 * depth_scale,
            sign * solution1 * depth_scale,
            sign * solution2 * depth_scale,
        );
        refine_depths(&mut depths, [a12, a13, a23], [m12, m13, m23], 5);
        if depths
            .iter()
            .any(|depth| !depth.is_finite() || *depth <= 0.0)
        {
            continue;
        }
        let camera1 = y1 * depths[0];
        let camera2 = y2 * depths[1];
        let camera3 = y3 * depths[2];
        let rotation = rotation_from_edges(camera1 - camera2, camera1 - camera3, &world_inverse);
        let translation = camera1 - rotation * x1;
        input.push_edge_basis_pose_from_positive_depths(&mut output, rotation, translation);
    }
    output
}

fn find_xk(a: f64, b: f64, c: f64) -> Option<f64> {
    let value = if a > 0.0 {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant > 0.0 {
            ((-b + discriminant.sqrt()) / (2.0 * a)).ceil()
        } else if discriminant == 0.0 {
            (-b / (2.0 * a) + 1.0).ceil()
        } else {
            0.0
        }
    } else if a < 0.0 {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant <= 0.0 {
            return None;
        }
        -b / (2.0 * a)
    } else if b > 0.0 {
        (c / b + 1.0).ceil()
    } else if b < 0.0 {
        (c / b - 1.0).floor()
    } else if c > 0.0 {
        0.0
    } else {
        return None;
    };
    value.is_finite().then_some(value)
}

#[inline]
fn nonnegative_sqrt(value: f64, scale: f64) -> Option<f64> {
    let tolerance = 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    (value >= -tolerance && value.is_finite()).then(|| value.max(0.0).sqrt())
}
