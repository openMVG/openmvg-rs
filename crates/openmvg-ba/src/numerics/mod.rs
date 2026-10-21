// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Numerical primitives beneath bundle adjustment and geometric refinement.
//!
//! Use it to write a small refiner of your own: it supplies forward-mode [`jet`]s, rotation and
//! pose [`manifold`] maps, and the fixed-size Levenberg-Marquardt loop of [`refine`], and you
//! supply the residual model. `openmvg-geometry` builds its homography and pose refiners on it.
//!
//! The pieces follow the same rules. A [`jet::Jet`] carries a value and its fixed-size gradient
//! (forward-mode automatic differentiation); a model can instead supply analytic derivatives
//! through a linearized residual without changing the optimizer. The loop in [`refine`] is
//! Levenberg-Marquardt with Nielsen's gain-ratio damping. Noise standard deviations whiten each
//! residual before the robust loss weighs it. Inputs are checked before any caller-owned state
//! changes, and a solve that fits its workspace does not allocate.

pub mod jet;
pub mod manifold;
pub mod refine;
pub(crate) mod robust;
#[cfg(feature = "faer")]
#[doc(hidden)]
pub mod sparse;
