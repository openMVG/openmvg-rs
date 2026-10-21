// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-budget robust-loss continuation.

use crate::numerics::refine::{Loss, Options};

/// One stage of the robust-loss schedule: the loss to minimize and its iteration budget.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    /// Loss minimized during this stage.
    pub loss: Loss,
    /// Iteration budget of this stage; `0` skips it.
    pub max_iterations: usize,
}

/// Index of the last stage with a nonzero budget, where convergence tests apply.
pub fn final_stage_index(stages: &[Stage; 3]) -> usize {
    stages
        .iter()
        .rposition(|stage| stage.max_iterations > 0)
        .unwrap_or(0)
}

/// Graduated non-convexity for heavy-tailed losses.
pub fn stages(options: Options) -> [Stage; 3] {
    match options.loss {
        Loss::L2 => [
            Stage {
                loss: Loss::L2,
                max_iterations: options.max_iterations,
            },
            Stage {
                loss: Loss::L2,
                max_iterations: 0,
            },
            Stage {
                loss: Loss::L2,
                max_iterations: 0,
            },
        ],
        Loss::Huber { .. } => [
            Stage {
                loss: options.loss,
                max_iterations: options.max_iterations,
            },
            Stage {
                loss: Loss::L2,
                max_iterations: 0,
            },
            Stage {
                loss: Loss::L2,
                max_iterations: 0,
            },
        ],
        Loss::Cauchy { scale } => {
            let l2 = (options.max_iterations / 4).min(5);
            [
                Stage {
                    loss: Loss::L2,
                    max_iterations: l2,
                },
                Stage {
                    loss: Loss::Cauchy { scale },
                    max_iterations: options.max_iterations - l2,
                },
                Stage {
                    loss: Loss::L2,
                    max_iterations: 0,
                },
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::refine::ConvergenceCriteria;

    #[test]
    fn schedule_preserves_budget_and_reaches_target_loss() {
        for iterations in 0..20 {
            for loss in [
                Loss::L2,
                Loss::Huber { delta: 0.01 },
                Loss::Cauchy { scale: 0.01 },
            ] {
                let schedule = stages(Options {
                    max_iterations: iterations,
                    loss,
                    convergence: ConvergenceCriteria::DEFAULT,
                });
                assert_eq!(
                    schedule
                        .iter()
                        .map(|stage| stage.max_iterations)
                        .sum::<usize>(),
                    iterations
                );
                if iterations > 0 {
                    assert!(
                        schedule
                            .iter()
                            .any(|stage| stage.max_iterations > 0 && stage.loss == loss),
                        "target {loss:?} was omitted from a {iterations}-step schedule"
                    );
                }
            }
        }
    }
}
