// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced camera system assembly for the self-calibrating sparse driver.

use crate::kernel::RelativePose;
use crate::kernel::multi_view::model::ObservationModel;
use crate::kernel::multi_view::schur::{self, GaugeState};
use crate::kernel::multi_view::sparse::api::SparseCameraSolver;
use crate::kernel::multi_view::sparse::direct::SparseHessian;
use crate::kernel::multi_view::sparse::direct_assembly::DirectSchurAssemblyWorkspace;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::direct_assembly::{
    DirectPointCacheMut, DirectReducedAssemblyMut,
};
use crate::kernel::multi_view::sparse::workspace::{
    DirectAssemblyState, SelfCalibrationBackend, SparseSelfCalibrationWorkspace,
    SparseWorkspaceCore,
};
use crate::kernel::multi_view::{BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks};
use crate::numerics::refine::NoiseModel;

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_sparse(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: crate::numerics::refine::Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    retain: bool,
    workspace: &mut SparseSelfCalibrationWorkspace,
) -> schur::Assembly {
    if retain && workspace.camera_solver() == SparseCameraSolver::IterativeSchur {
        let SparseSelfCalibrationWorkspace {
            sparse, backend, ..
        } = workspace;
        let SelfCalibrationBackend::Iterative(retained) = backend else {
            return schur::Assembly::invalid();
        };
        let problem = schur::BaProblemView {
            model,
            poses,
            points,
            fixed,
            priors,
        };
        let policy = schur::LinearizationPolicy {
            gauge,
            loss,
            noise,
            damping,
            prior_sqrt_scale: 1.0,
        };
        #[cfg(feature = "parallel")]
        if let Some(assembly) = retained.assemble_parallel(problem, policy, sparse) {
            if assembly.valid
                && !retained.prepare_recomputed_point_blocks(sparse.signature.fixed_points())
            {
                return schur::Assembly::invalid();
            }
            return assembly;
        }
        let (mut hessian, point_hessians) = retained.assembly_buffers();
        let point_cache = schur::PointCacheBuffers {
            point_steps: (!sparse.point_cache_valid.is_empty()).then_some(
                schur::PointStepCacheBuffers {
                    valid: &mut sparse.point_cache_valid,
                    inverse_gradients: &mut sparse.point_inverse_gradients,
                    damping_diagonals: &mut sparse.point_damping_diagonals,
                },
            ),
            cross_offsets: &sparse.point_cross_offsets,
            inverse_cross_values: sparse.point_inverse_cross_values.as_mut(),
            point_offset: 0,
            cross_offset: 0,
            damped_hessians: Some(point_hessians),
        };
        let assembly = schur::assemble_with_hessian_problem(
            problem,
            policy,
            schur::SchurAssemblyMut {
                dimension: sparse.layout.dimension(),
                layout: schur::VariableLayout {
                    offsets: sparse.layout.camera_offsets(),
                    pose_dimensions: sparse.layout.camera_dimensions(),
                    intrinsic_offsets: sparse.layout.intrinsic_offsets(),
                    intrinsic_dimensions: sparse.layout.intrinsic_dimensions(),
                    pose_masks: &[],
                    pose_variables: CameraPoseVariables::RotationAndTranslation,
                },
                materialize_schur: false,
                finalize: true,
                hessian: &mut hessian,
                gradient: &mut sparse.gradient,
                full_gradient: &mut sparse.full_gradient,
                diagonal: &mut sparse.diagonal,
                cross: &mut sparse.cross,
                active_parameters: &mut sparse.active_parameters,
                parameter_marks: &mut sparse.parameter_marks,
                point_cache: Some(point_cache),
            },
        );
        if assembly.valid
            && !retained.prepare_recomputed_point_blocks(sparse.signature.fixed_points())
        {
            return schur::Assembly::invalid();
        }
        return assembly;
    }
    if workspace.ensure_direct_storage().is_err() {
        return schur::Assembly::invalid();
    }
    if let SelfCalibrationBackend::Direct(DirectAssemblyState::ColumnOwned {
        executor,
        point_hessians,
    }) = &mut workspace.backend
    {
        return assemble_sparse_direct(
            schur::BaProblemView {
                model,
                poses,
                points,
                fixed,
                priors,
            },
            schur::LinearizationPolicy {
                gauge,
                loss,
                noise,
                damping,
                prior_sqrt_scale: 1.0,
            },
            &mut workspace.sparse,
            executor,
            point_hessians,
        );
    }
    let sparse = &mut workspace.sparse;
    let mut hessian = SparseHessian {
        scatter: &sparse.scatter,
        values: &mut sparse.values,
        pattern_complete: true,
    };
    schur::assemble_with_hessian_problem(
        schur::BaProblemView {
            model,
            poses,
            points,
            fixed,
            priors,
        },
        schur::LinearizationPolicy {
            gauge,
            loss,
            noise,
            damping,
            prior_sqrt_scale: 1.0,
        },
        schur::SchurAssemblyMut {
            dimension: sparse.layout.dimension(),
            layout: schur::VariableLayout {
                offsets: sparse.layout.camera_offsets(),
                pose_dimensions: sparse.layout.camera_dimensions(),
                intrinsic_offsets: sparse.layout.intrinsic_offsets(),
                intrinsic_dimensions: sparse.layout.intrinsic_dimensions(),
                pose_masks: &[],
                pose_variables: CameraPoseVariables::RotationAndTranslation,
            },
            materialize_schur: true,
            finalize: true,
            hessian: &mut hessian,
            gradient: &mut sparse.gradient,
            full_gradient: &mut sparse.full_gradient,
            diagonal: &mut sparse.diagonal,
            cross: &mut sparse.cross,
            active_parameters: &mut sparse.active_parameters,
            parameter_marks: &mut sparse.parameter_marks,
            point_cache: None,
        },
    )
}

fn assemble_sparse_direct(
    problem: schur::BaProblemView<'_>,
    policy: schur::LinearizationPolicy<'_>,
    sparse: &mut SparseWorkspaceCore,
    direct: &mut DirectSchurAssemblyWorkspace,
    point_hessians: &mut [crate::kernel::point_block::PackedPointBlock],
) -> schur::Assembly {
    let schur::BaProblemView {
        model,
        poses,
        points,
        fixed,
        priors,
    } = problem;
    let schur::LinearizationPolicy {
        gauge,
        loss,
        noise,
        damping,
        prior_sqrt_scale,
    } = policy;
    (|| {
        let dimension = sparse.layout.dimension();
        let layout = schur::VariableLayout {
            offsets: sparse.layout.camera_offsets(),
            pose_dimensions: sparse.layout.camera_dimensions(),
            intrinsic_offsets: sparse.layout.intrinsic_offsets(),
            intrinsic_dimensions: sparse.layout.intrinsic_dimensions(),
            pose_masks: &[],
            pose_variables: CameraPoseVariables::RotationAndTranslation,
        };
        #[cfg(feature = "parallel")]
        let mut assembly = {
            direct.assemble_points(
                schur::BaProblemView {
                    model,
                    poses,
                    points,
                    fixed,
                    priors,
                },
                schur::LinearizationPolicy {
                    gauge,
                    loss,
                    noise,
                    damping,
                    prior_sqrt_scale,
                },
                layout,
                &sparse.scatter,
                DirectPointCacheMut {
                    point_steps: (!sparse.point_cache_valid.is_empty()).then_some(
                        schur::PointStepCacheBuffers {
                            valid: &mut sparse.point_cache_valid,
                            inverse_gradients: &mut sparse.point_inverse_gradients,
                            damping_diagonals: &mut sparse.point_damping_diagonals,
                        },
                    ),
                    cross_offsets: &sparse.point_cross_offsets,
                    inverse_crosses: sparse.point_inverse_cross_values.as_mut(),
                    point_hessians: &mut *point_hessians,
                },
                DirectReducedAssemblyMut {
                    gradient: &mut sparse.gradient,
                    full_gradient: &mut sparse.full_gradient,
                    diagonal: &mut sparse.diagonal,
                },
            )
        };
        #[cfg(feature = "parallel")]
        if assembly.valid && !direct.seed_base_values(&sparse.scatter, &mut sparse.values) {
            assembly.valid = false;
        }
        #[cfg(not(feature = "parallel"))]
        let mut assembly = {
            let point_cache = schur::PointCacheBuffers {
                point_steps: (!sparse.point_cache_valid.is_empty()).then_some(
                    schur::PointStepCacheBuffers {
                        valid: &mut sparse.point_cache_valid,
                        inverse_gradients: &mut sparse.point_inverse_gradients,
                        damping_diagonals: &mut sparse.point_damping_diagonals,
                    },
                ),
                cross_offsets: &sparse.point_cross_offsets,
                inverse_cross_values: sparse.point_inverse_cross_values.as_mut(),
                point_offset: 0,
                cross_offset: 0,
                damped_hessians: Some(&mut *point_hessians),
            };
            let mut hessian = SparseHessian {
                scatter: &sparse.scatter,
                values: &mut sparse.values,
                pattern_complete: true,
            };
            schur::assemble_with_hessian_problem(
                schur::BaProblemView {
                    model,
                    poses,
                    points,
                    fixed,
                    priors,
                },
                schur::LinearizationPolicy {
                    gauge,
                    loss,
                    noise,
                    damping,
                    prior_sqrt_scale,
                },
                schur::SchurAssemblyMut {
                    dimension,
                    layout,
                    materialize_schur: false,
                    finalize: false,
                    hessian: &mut hessian,
                    gradient: &mut sparse.gradient,
                    full_gradient: &mut sparse.full_gradient,
                    diagonal: &mut sparse.diagonal,
                    cross: &mut sparse.cross,
                    active_parameters: &mut sparse.active_parameters,
                    parameter_marks: &mut sparse.parameter_marks,
                    point_cache: Some(point_cache),
                },
            )
        };
        if !assembly.valid {
            return assembly;
        }
        let Some(inverse_crosses) = sparse.point_inverse_cross_values.f64_rows() else {
            assembly.valid = false;
            return assembly;
        };
        if !direct.materialize(
            &sparse.scatter,
            &mut sparse.values,
            point_hessians,
            inverse_crosses,
        ) {
            assembly.valid = false;
            return assembly;
        }
        let mut hessian = SparseHessian {
            scatter: &sparse.scatter,
            values: &mut sparse.values,
            pattern_complete: true,
        };
        schur::accumulate_camera_priors(
            model,
            poses,
            gauge,
            layout,
            priors,
            prior_sqrt_scale,
            &mut hessian,
            &mut sparse.gradient,
            &mut sparse.full_gradient,
            &mut sparse.diagonal,
            &mut assembly,
        );
        schur::finalize_assembly(
            &mut hessian,
            &sparse.gradient,
            &sparse.full_gradient,
            &mut sparse.diagonal,
            damping,
            dimension,
            layout,
            assembly,
        )
    })()
}
