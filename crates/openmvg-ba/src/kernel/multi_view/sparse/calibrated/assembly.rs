// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced camera system assembly for the calibrated sparse driver.

use crate::kernel::RelativePose;
use crate::kernel::multi_view::model::ObservationModel;
#[cfg(not(feature = "parallel"))]
use crate::kernel::multi_view::schur::HessianAccumulator;
use crate::kernel::multi_view::schur::{self, GaugeState};
use crate::kernel::multi_view::sparse::api::SparsePointBackSubstitution;
use crate::kernel::multi_view::sparse::direct::SparseHessian;
use crate::kernel::multi_view::sparse::direct_assembly::DirectSchurAssemblyWorkspace;
#[cfg(feature = "parallel")]
use crate::kernel::multi_view::sparse::direct_assembly::{
    DirectPointCacheMut, DirectReducedAssemblyMut,
};
use crate::kernel::multi_view::sparse::workspace::{
    CalibratedBackend, DirectAssemblyState, SparseBundleAdjustmentWorkspace, SparseWorkspaceCore,
};
use crate::kernel::multi_view::{BundleAdjustmentPriors, CameraPoseVariables, FixedVariableBlocks};
use crate::numerics::refine::{Loss, NoiseModel};

pub(super) fn calibrated_layout<'a>(
    offsets: &'a [usize],
    dimensions: &'a [usize],
    pose_variables: CameraPoseVariables,
    pose_masks: &'a [crate::PoseVariables],
) -> schur::VariableLayout<'a> {
    let layout = match pose_variables {
        CameraPoseVariables::RotationAndTranslation => {
            schur::VariableLayout::calibrated(offsets, dimensions)
        }
        CameraPoseVariables::TranslationOnly => {
            schur::VariableLayout::calibrated_translation_only(offsets, dimensions)
        }
    };
    layout.with_pose_masks(pose_masks)
}

fn assemble_with_pose_variables<H: schur::HessianAccumulator>(
    problem: schur::BaProblemView<'_>,
    policy: schur::LinearizationPolicy<'_>,
    target: schur::SchurAssemblyMut<'_, H>,
) -> schur::Assembly {
    match target.layout.pose_variables {
        CameraPoseVariables::RotationAndTranslation => {
            schur::assemble_with_hessian_problem(problem, policy, target)
        }
        CameraPoseVariables::TranslationOnly => {
            schur::assemble_with_hessian_problem_translation_only(problem, policy, target)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::kernel::multi_view::sparse) fn assemble_sparse(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    retain_schur: bool,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> schur::Assembly {
    assemble_sparse_with_prior_scale(
        model,
        poses,
        points,
        gauge,
        loss,
        noise,
        damping,
        retain_schur,
        fixed,
        priors,
        1.0,
        workspace,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::kernel::multi_view::sparse) fn assemble_sparse_with_prior_scale(
    model: ObservationModel<'_>,
    poses: &[RelativePose],
    points: &[[f64; 3]],
    gauge: GaugeState,
    loss: Loss,
    noise: NoiseModel<'_>,
    damping: f64,
    retain_schur: bool,
    fixed: FixedVariableBlocks<'_>,
    priors: BundleAdjustmentPriors<'_>,
    prior_sqrt_scale: f64,
    workspace: &mut SparseBundleAdjustmentWorkspace,
) -> schur::Assembly {
    if retain_schur && workspace.camera_solver().uses_retained_schur() {
        let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
        let CalibratedBackend::Retained(retained) = backend else {
            return schur::Assembly::invalid();
        };
        #[cfg(feature = "parallel")]
        if fixed == FixedVariableBlocks::NONE
            && priors == BundleAdjustmentPriors::NONE
            && prior_sqrt_scale == 1.0
            && let Some(assembly) = retained
                .assemble_retained_parallel(model, poses, points, gauge, loss, noise, damping, core)
        {
            if assembly.valid
                && !retained.prepare_recomputed_point_blocks(core.signature.fixed_points())
            {
                return schur::Assembly::invalid();
            }
            return assembly;
        }
        let (mut hessian, point_hessians) = retained.assembly_buffers(core.layout.camera_offsets());
        let point_steps =
            (!core.point_cache_valid.is_empty()).then_some(schur::PointStepCacheBuffers {
                valid: &mut core.point_cache_valid,
                inverse_gradients: &mut core.point_inverse_gradients,
                damping_diagonals: &mut core.point_damping_diagonals,
            });
        let point_cache = schur::PointCacheBuffers {
            point_steps,
            cross_offsets: &core.point_cross_offsets,
            inverse_cross_values: core.point_inverse_cross_values.as_mut(),
            point_offset: 0,
            cross_offset: 0,
            damped_hessians: Some(point_hessians),
        };
        let layout = calibrated_layout(
            core.layout.camera_offsets(),
            core.layout.camera_dimensions(),
            core.signature.pose_variables(),
            core.signature.pose_masks(),
        );
        let assembly = assemble_with_pose_variables(
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
                dimension: core.layout.dimension(),
                layout,
                materialize_schur: false,
                finalize: true,
                hessian: &mut hessian,
                gradient: &mut core.gradient,
                full_gradient: &mut core.full_gradient,
                diagonal: &mut core.diagonal,
                cross: &mut core.cross,
                active_parameters: &mut core.active_parameters,
                parameter_marks: &mut core.parameter_marks,
                point_cache: Some(point_cache),
            },
        );
        if assembly.valid
            && !retained.prepare_recomputed_point_blocks(core.signature.fixed_points())
        {
            return schur::Assembly::invalid();
        }
        return assembly;
    }

    if workspace.ensure_direct_storage().is_err() {
        return schur::Assembly::invalid();
    }

    #[cfg(feature = "parallel")]
    {
        let SparseBundleAdjustmentWorkspace { core, backend } = workspace;
        if let CalibratedBackend::Retained(retained) = backend
            && retained.parallel_threads().is_some()
            // Rounded or recomputed rows do not provide exact zero-damping
            // information. The serial branch below forms fresh f64 Schur terms.
            && (damping != 0.0 || core.point_inverse_cross_values.f64_rows().is_some())
        {
            return retained
                .assemble_direct_parallel(
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
                    core,
                )
                .unwrap_or_else(schur::Assembly::invalid);
        }
    }

    if let CalibratedBackend::Direct(DirectAssemblyState::ColumnOwned {
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
                prior_sqrt_scale,
            },
            &mut workspace.core,
            executor,
            point_hessians,
        );
    }

    let core = &mut workspace.core;
    let mut hessian = SparseHessian {
        scatter: &core.scatter,
        values: &mut core.values,
        pattern_complete: true,
    };
    let point_cache = if core.point_back_substitution == SparsePointBackSubstitution::Cached {
        Some(schur::PointCacheBuffers {
            point_steps: Some(schur::PointStepCacheBuffers {
                valid: &mut core.point_cache_valid,
                inverse_gradients: &mut core.point_inverse_gradients,
                damping_diagonals: &mut core.point_damping_diagonals,
            }),
            cross_offsets: &core.point_cross_offsets,
            inverse_cross_values: core.point_inverse_cross_values.as_mut(),
            point_offset: 0,
            cross_offset: 0,
            damped_hessians: None,
        })
    } else {
        None
    };
    let layout = calibrated_layout(
        core.layout.camera_offsets(),
        core.layout.camera_dimensions(),
        core.signature.pose_variables(),
        core.signature.pose_masks(),
    );
    assemble_with_pose_variables(
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
            dimension: core.layout.dimension(),
            layout,
            materialize_schur: true,
            finalize: true,
            hessian: &mut hessian,
            gradient: &mut core.gradient,
            full_gradient: &mut core.full_gradient,
            diagonal: &mut core.diagonal,
            cross: &mut core.cross,
            active_parameters: &mut core.active_parameters,
            parameter_marks: &mut core.parameter_marks,
            point_cache,
        },
    )
}

pub(in crate::kernel::multi_view::sparse) fn assemble_sparse_direct(
    problem: schur::BaProblemView<'_>,
    policy: schur::LinearizationPolicy<'_>,
    core: &mut SparseWorkspaceCore,
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
    {
        let dimension = core.layout.dimension();
        let layout = calibrated_layout(
            core.layout.camera_offsets(),
            core.layout.camera_dimensions(),
            core.signature.pose_variables(),
            core.signature.pose_masks(),
        );
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
                &core.scatter,
                DirectPointCacheMut {
                    point_steps: Some(schur::PointStepCacheBuffers {
                        valid: &mut core.point_cache_valid,
                        inverse_gradients: &mut core.point_inverse_gradients,
                        damping_diagonals: &mut core.point_damping_diagonals,
                    }),
                    cross_offsets: &core.point_cross_offsets,
                    inverse_crosses: core.point_inverse_cross_values.as_mut(),
                    point_hessians: &mut *point_hessians,
                },
                DirectReducedAssemblyMut {
                    gradient: &mut core.gradient,
                    full_gradient: &mut core.full_gradient,
                    diagonal: &mut core.diagonal,
                },
            )
        };
        #[cfg(feature = "parallel")]
        if assembly.valid && !direct.seed_base_values(&core.scatter, &mut core.values) {
            assembly.valid = false;
        }
        #[cfg(not(feature = "parallel"))]
        let mut assembly = {
            let mut hessian = SparseHessian {
                scatter: &core.scatter,
                values: &mut core.values,
                pattern_complete: true,
            };
            let point_cache = schur::PointCacheBuffers {
                point_steps: Some(schur::PointStepCacheBuffers {
                    valid: &mut core.point_cache_valid,
                    inverse_gradients: &mut core.point_inverse_gradients,
                    damping_diagonals: &mut core.point_damping_diagonals,
                }),
                cross_offsets: &core.point_cross_offsets,
                inverse_cross_values: core.point_inverse_cross_values.as_mut(),
                point_offset: 0,
                cross_offset: 0,
                damped_hessians: Some(&mut *point_hessians),
            };
            let mut assembly = assemble_with_pose_variables(
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
                    gradient: &mut core.gradient,
                    full_gradient: &mut core.full_gradient,
                    diagonal: &mut core.diagonal,
                    cross: &mut core.cross,
                    active_parameters: &mut core.active_parameters,
                    parameter_marks: &mut core.parameter_marks,
                    point_cache: Some(point_cache),
                },
            );
            assembly.valid &= hessian.is_finite();
            assembly
        };
        if !assembly.valid {
            return assembly;
        }
        #[cfg(feature = "parallel")]
        let materialized = direct.materialize_cached(
            &core.scatter,
            &mut core.values,
            point_hessians,
            &core.point_inverse_cross_values,
        );
        #[cfg(not(feature = "parallel"))]
        let materialized =
            core.point_inverse_cross_values
                .f64_rows()
                .is_some_and(|inverse_crosses| {
                    direct.materialize(
                        &core.scatter,
                        &mut core.values,
                        point_hessians,
                        inverse_crosses,
                    )
                });
        if !materialized {
            assembly.valid = false;
            return assembly;
        }
        let mut hessian = SparseHessian {
            scatter: &core.scatter,
            values: &mut core.values,
            pattern_complete: true,
        };
        match core.signature.pose_variables() {
            CameraPoseVariables::RotationAndTranslation => schur::accumulate_camera_priors(
                model,
                poses,
                gauge,
                layout,
                priors,
                prior_sqrt_scale,
                &mut hessian,
                &mut core.gradient,
                &mut core.full_gradient,
                &mut core.diagonal,
                &mut assembly,
            ),
            CameraPoseVariables::TranslationOnly => {
                schur::accumulate_camera_priors_translation_only(
                    model,
                    poses,
                    gauge,
                    layout,
                    priors,
                    prior_sqrt_scale,
                    &mut hessian,
                    &mut core.gradient,
                    &mut core.full_gradient,
                    &mut core.diagonal,
                    &mut assembly,
                )
            }
        }
        schur::finalize_assembly(
            &mut hessian,
            &core.gradient,
            &core.full_gradient,
            &mut core.diagonal,
            damping,
            dimension,
            layout,
            assembly,
        )
    }
}
