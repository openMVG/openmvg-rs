// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size scalar forward-mode automatic differentiation.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A scalar first-order dual number with `N` partial derivatives.
///
/// The value and derivative array are stack-owned and require no allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jet<const N: usize> {
    /// Function value.
    pub value: f64,
    /// First derivatives with respect to the active local coordinates.
    pub deriv: [f64; N],
}

impl<const N: usize> Jet<N> {
    /// Construct a constant with zero derivatives.
    #[inline(always)]
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            deriv: [0.0; N],
        }
    }

    /// Construct a variable whose derivative is one at `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index >= N`. Use [`try_variable`](Self::try_variable) for
    /// an index received from untrusted input.
    #[inline(always)]
    pub fn variable(value: f64, index: usize) -> Self {
        let mut result = Self::constant(value);
        result.deriv[index] = 1.0;
        result
    }

    /// Construct a seeded variable, or return `None` for an invalid index.
    #[inline(always)]
    pub fn try_variable(value: f64, index: usize) -> Option<Self> {
        (index < N).then(|| Self::variable(value, index))
    }

    /// Evaluate the square root and its first derivatives.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let factor = 0.5 / value;
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate the reciprocal and its first derivatives.
    #[inline(always)]
    pub fn recip(self) -> Self {
        let value = self.value.recip();
        let factor = -(value * value);
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate the exponential and its first derivatives.
    #[inline(always)]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * value),
        }
    }

    /// Evaluate the natural logarithm and its first derivatives.
    #[inline(always)]
    pub fn ln(self) -> Self {
        let factor = self.value.recip();
        Self {
            value: self.value.ln(),
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate sine and cosine together with their first derivatives.
    #[inline(always)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (sine, cosine) = self.value.sin_cos();
        (
            Self {
                value: sine,
                deriv: self.deriv.map(|derivative| derivative * cosine),
            },
            Self {
                value: cosine,
                deriv: self.deriv.map(|derivative| -derivative * sine),
            },
        )
    }

    /// Evaluate sine and its first derivatives.
    #[inline(always)]
    pub fn sin(self) -> Self {
        self.sin_cos().0
    }

    /// Evaluate cosine and its first derivatives.
    #[inline(always)]
    pub fn cos(self) -> Self {
        self.sin_cos().1
    }

    /// Evaluate tangent and its first derivatives.
    #[inline(always)]
    pub fn tan(self) -> Self {
        let value = self.value.tan();
        let factor = 1.0 + value * value;
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate inverse sine and its first derivatives.
    #[inline(always)]
    pub fn asin(self) -> Self {
        let factor = (1.0 - self.value * self.value).sqrt().recip();
        Self {
            value: self.value.asin(),
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate inverse cosine and its first derivatives.
    #[inline(always)]
    pub fn acos(self) -> Self {
        let factor = -(1.0 - self.value * self.value).sqrt().recip();
        Self {
            value: self.value.acos(),
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate inverse tangent and its first derivatives.
    #[inline(always)]
    pub fn atan(self) -> Self {
        let factor = (1.0 + self.value * self.value).recip();
        Self {
            value: self.value.atan(),
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Four-quadrant arctangent `atan2(self, x)`.
    #[inline(always)]
    pub fn atan2(self, x: Self) -> Self {
        let denominator = self.value * self.value + x.value * x.value;
        Self {
            value: self.value.atan2(x.value),
            deriv: std::array::from_fn(|index| {
                (x.value * self.deriv[index] - self.value * x.deriv[index]) / denominator
            }),
        }
    }

    /// Evaluate hyperbolic sine and its first derivatives.
    #[inline(always)]
    pub fn sinh(self) -> Self {
        let value = self.value.sinh();
        let factor = self.value.cosh();
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate hyperbolic cosine and its first derivatives.
    #[inline(always)]
    pub fn cosh(self) -> Self {
        let value = self.value.cosh();
        let factor = self.value.sinh();
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Evaluate hyperbolic tangent and its first derivatives.
    #[inline(always)]
    pub fn tanh(self) -> Self {
        let value = self.value.tanh();
        let factor = 1.0 - value * value;
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Raise the value to an integer power and propagate derivatives.
    #[inline(always)]
    pub fn powi(self, exponent: i32) -> Self {
        if exponent == 0 {
            return Self::constant(1.0);
        }
        let value = self.value.powi(exponent);
        let factor = if exponent == i32::MIN {
            f64::from(exponent) * value / self.value
        } else {
            f64::from(exponent) * self.value.powi(exponent - 1)
        };
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Raise the value to a constant real power and propagate derivatives.
    #[inline(always)]
    pub fn powf(self, exponent: f64) -> Self {
        if exponent == 0.0 {
            return Self::constant(1.0);
        }
        let value = self.value.powf(exponent);
        let factor = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            deriv: self.deriv.map(|derivative| derivative * factor),
        }
    }

    /// Raise the value to a variable dual exponent.
    #[inline(always)]
    pub fn powd(self, exponent: Self) -> Self {
        if exponent.value == 0.0 && exponent.deriv.iter().all(|value| *value == 0.0) {
            return Self::constant(1.0);
        }
        let value = self.value.powf(exponent.value);
        let base_factor = exponent.value * self.value.powf(exponent.value - 1.0);
        Self {
            value,
            deriv: std::array::from_fn(|index| {
                let exponent_term = if exponent.deriv[index] == 0.0 {
                    0.0
                } else {
                    exponent.deriv[index] * value * self.value.ln()
                };
                self.deriv[index] * base_factor + exponent_term
            }),
        }
    }

    /// Evaluate the Euclidean norm of two dual values.
    #[inline(always)]
    pub fn hypot(self, rhs: Self) -> Self {
        let value = self.value.hypot(rhs.value);
        Self {
            value,
            deriv: std::array::from_fn(|index| {
                self.value / value * self.deriv[index] + rhs.value / value * rhs.deriv[index]
            }),
        }
    }
}

impl<const N: usize> Add for Jet<N> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            deriv: std::array::from_fn(|index| self.deriv[index] + rhs.deriv[index]),
        }
    }
}

impl<const N: usize> Sub for Jet<N> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            deriv: std::array::from_fn(|index| self.deriv[index] - rhs.deriv[index]),
        }
    }
}

impl<const N: usize> Mul for Jet<N> {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            deriv: std::array::from_fn(|index| {
                self.deriv[index] * rhs.value + self.value * rhs.deriv[index]
            }),
        }
    }
}

impl<const N: usize> Div for Jet<N> {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let inverse = rhs.value.recip();
        let inverse_squared = inverse * inverse;
        Self {
            value: self.value * inverse,
            deriv: std::array::from_fn(|index| {
                (self.deriv[index] * rhs.value - self.value * rhs.deriv[index]) * inverse_squared
            }),
        }
    }
}

impl<const N: usize> Neg for Jet<N> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            deriv: self.deriv.map(|derivative| -derivative),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Jet;

    #[test]
    fn invalid_variable_index_is_rejected() {
        assert!(Jet::<2>::try_variable(1.0, 2).is_none());
    }

    #[test]
    fn composite_derivatives_match_finite_differences() {
        fn scalar(x: f64, y: f64) -> f64 {
            (x * x + y * y).sqrt() / (x + y.recip())
        }

        let x0 = 1.7;
        let y0 = 0.8;
        let x = Jet::<2>::variable(x0, 0);
        let y = Jet::<2>::variable(y0, 1);
        let value = (x * x + y * y).sqrt() / (x + y.recip());
        let epsilon = 1e-7;
        let expected = [
            (scalar(x0 + epsilon, y0) - scalar(x0 - epsilon, y0)) / (2.0 * epsilon),
            (scalar(x0, y0 + epsilon) - scalar(x0, y0 - epsilon)) / (2.0 * epsilon),
        ];

        assert!((value.value - scalar(x0, y0)).abs() < 1e-14);
        for (actual, expected) in value.deriv.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn elementary_functions_match_finite_differences() {
        type ScalarJet = Jet<1>;
        type UnaryCase = (fn(ScalarJet) -> ScalarJet, fn(f64) -> f64);
        let cases: [UnaryCase; 12] = [
            (ScalarJet::exp, f64::exp),
            (ScalarJet::ln, f64::ln),
            (ScalarJet::sin, f64::sin),
            (ScalarJet::cos, f64::cos),
            (ScalarJet::tan, f64::tan),
            (ScalarJet::asin, f64::asin),
            (ScalarJet::acos, f64::acos),
            (ScalarJet::atan, f64::atan),
            (ScalarJet::sinh, f64::sinh),
            (ScalarJet::cosh, f64::cosh),
            (ScalarJet::tanh, f64::tanh),
            (|value| value.powi(3), |value| value.powi(3)),
        ];
        let x0 = 0.4;
        let epsilon = 1e-7;
        for (jet_function, scalar_function) in cases {
            let actual = jet_function(ScalarJet::variable(x0, 0));
            let expected_value = scalar_function(x0);
            let expected_derivative =
                (scalar_function(x0 + epsilon) - scalar_function(x0 - epsilon)) / (2.0 * epsilon);
            assert!((actual.value - expected_value).abs() < 1e-14);
            assert!(
                (actual.deriv[0] - expected_derivative).abs()
                    < 2e-8 * expected_derivative.abs().max(1.0)
            );
        }
    }

    #[test]
    fn binary_functions_match_finite_differences() {
        fn check(jet_function: fn(Jet<2>, Jet<2>) -> Jet<2>, scalar_function: fn(f64, f64) -> f64) {
            let (x0, y0) = (1.3, 0.7);
            let epsilon = 1e-7;
            let actual = jet_function(Jet::variable(x0, 0), Jet::variable(y0, 1));
            let expected = [
                (scalar_function(x0 + epsilon, y0) - scalar_function(x0 - epsilon, y0))
                    / (2.0 * epsilon),
                (scalar_function(x0, y0 + epsilon) - scalar_function(x0, y0 - epsilon))
                    / (2.0 * epsilon),
            ];
            assert!((actual.value - scalar_function(x0, y0)).abs() < 1e-14);
            for (actual, expected) in actual.deriv.iter().zip(expected) {
                assert!((actual - expected).abs() < 2e-8 * expected.abs().max(1.0));
            }
        }

        check(Jet::atan2, f64::atan2);
        check(Jet::powd, f64::powf);
        check(Jet::hypot, f64::hypot);

        let negative_base = Jet::<1>::variable(-2.0, 0).powd(Jet::constant(3.0));
        assert_eq!(negative_base.value, -8.0);
        assert_eq!(negative_base.deriv[0], 12.0);
        let zero_base = Jet::<1>::variable(0.0, 0).powd(Jet::constant(3.0));
        let zero_power = Jet::<1>::variable(0.0, 0).powf(0.0);
        assert_eq!(zero_power.value, 1.0);
        assert_eq!(zero_power.deriv, [0.0]);
        let constant_zero_power = Jet::<1>::variable(0.0, 0).powd(Jet::constant(0.0));
        assert_eq!(constant_zero_power.value, 1.0);
        assert_eq!(constant_zero_power.deriv, [0.0]);
        let variable_zero_power = Jet::<1>::constant(0.0).powd(Jet::variable(0.0, 0));
        assert!(!variable_zero_power.deriv[0].is_finite());
        assert_eq!(zero_base.value, 0.0);
        assert_eq!(zero_base.deriv[0], 0.0);
    }
}
