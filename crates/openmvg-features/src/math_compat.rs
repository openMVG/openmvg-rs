// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compatibility shim for `f32` transcendental math functions.
//!
//! In `std` builds the standard library provides these as inherent methods on
//! `f32`. In `no_std` builds the same operations are obtained from the `libm`
//! crate. This module exposes a uniform set of free functions so callers
//! require no `#[cfg]` scaffolding at each call site.
//!
//! All functions are `#[inline(always)]`, so release builds inline the
//! indirection away.

// `core` exposes only a small subset of floating-point methods
// on targets without `std`. Importing this trait (as `_`) lets descriptor code
// keep the usual method-call syntax while dispatching the missing operations
// to `libm` in `no_std + alloc` builds.
//
// Inherent methods win over trait methods, and `std`'s inherent float methods
// are visible to this crate as soon as any crate in the build links `std`, even
// when this crate's `std` feature is off. That happens for the lib test target
// (the test harness) and for `--all-targets` builds (dev-dependencies enable
// `num-traits/std`). The trait and its imports are then unused, hence the
// `allow`s; a plain `no_std + alloc` build (e.g. wasm32) still needs them.
#[cfg(not(feature = "std"))]
#[allow(dead_code)]
pub(crate) trait FloatExt: Sized {
    fn ceil(self) -> Self;
    fn exp(self) -> Self;
    fn floor(self) -> Self;
    fn ln(self) -> Self;
    fn mul_add(self, multiplier: Self, addend: Self) -> Self;
    fn powf(self, exponent: Self) -> Self;
    fn powi(self, exponent: i32) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
    fn round(self) -> Self;
    fn sqrt(self) -> Self;
}

#[cfg(not(feature = "std"))]
macro_rules! impl_float_ext {
    (
        $float:ty,
        $ceil:path,
        $exp:path,
        $floor:path,
        $ln:path,
        $mul_add:path,
        $pow:path,
        $round:path,
        $sqrt:path
    ) => {
        impl FloatExt for $float {
            #[inline(always)]
            fn ceil(self) -> Self {
                $ceil(self)
            }

            #[inline(always)]
            fn exp(self) -> Self {
                $exp(self)
            }

            #[inline(always)]
            fn floor(self) -> Self {
                $floor(self)
            }

            #[inline(always)]
            fn ln(self) -> Self {
                $ln(self)
            }

            #[inline(always)]
            fn mul_add(self, multiplier: Self, addend: Self) -> Self {
                $mul_add(self, multiplier, addend)
            }

            #[inline(always)]
            fn powf(self, exponent: Self) -> Self {
                $pow(self, exponent)
            }

            #[inline(always)]
            fn powi(self, exponent: i32) -> Self {
                $pow(self, exponent as $float)
            }

            #[inline(always)]
            fn rem_euclid(self, rhs: Self) -> Self {
                let remainder = self % rhs;
                if remainder < 0.0 {
                    remainder + rhs.abs()
                } else {
                    remainder
                }
            }

            #[inline(always)]
            fn round(self) -> Self {
                $round(self)
            }

            #[inline(always)]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
        }
    };
}

#[cfg(not(feature = "std"))]
impl_float_ext!(
    f32,
    libm::ceilf,
    libm::expf,
    libm::floorf,
    libm::logf,
    libm::fmaf,
    libm::powf,
    libm::roundf,
    libm::sqrtf
);

#[cfg(not(feature = "std"))]
impl_float_ext!(
    f64,
    libm::ceil,
    libm::exp,
    libm::floor,
    libm::log,
    libm::fma,
    libm::pow,
    libm::round,
    libm::sqrt
);

#[cfg(feature = "std")]
mod inner {
    #[inline(always)]
    pub fn sinf(x: f32) -> f32 {
        x.sin()
    }
    #[inline(always)]
    pub fn cosf(x: f32) -> f32 {
        x.cos()
    }
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        x.exp()
    }
    #[inline(always)]
    pub fn log2f(x: f32) -> f32 {
        x.log2()
    }
    #[inline(always)]
    pub fn powf(x: f32, y: f32) -> f32 {
        x.powf(y)
    }
    #[inline(always)]
    pub fn atan2f(y: f32, x: f32) -> f32 {
        y.atan2(x)
    }
}

#[cfg(not(feature = "std"))]
mod inner {
    #[inline(always)]
    pub fn sinf(x: f32) -> f32 {
        libm::sinf(x)
    }
    #[inline(always)]
    pub fn cosf(x: f32) -> f32 {
        libm::cosf(x)
    }
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        libm::expf(x)
    }
    #[inline(always)]
    pub fn log2f(x: f32) -> f32 {
        libm::log2f(x)
    }
    #[inline(always)]
    pub fn powf(x: f32, y: f32) -> f32 {
        libm::powf(x, y)
    }
    #[inline(always)]
    pub fn atan2f(y: f32, x: f32) -> f32 {
        libm::atan2f(y, x)
    }
}

pub(crate) use inner::*;
