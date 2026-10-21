// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free Nielsen gain-ratio damping.

pub const DEFAULT_INITIAL_DAMPING: f64 = 1e-4;
const MINIMUM_DAMPING: f64 = 1e-12;
const MAXIMUM_DAMPING: f64 = 1e12;

/// Nielsen gain-ratio damping shared by fixed-size and BA optimizers.
///
/// The value is dimensionless because each solver applies it to a local
/// curvature diagonal. Rejections retain history through the increasing
/// `nu` multiplier; an accepted step resets that history.
#[derive(Clone, Copy, Debug)]
pub struct NielsenController {
    value: f64,
    nu: f64,
    last_gain_ratio: Option<f64>,
}

impl Default for NielsenController {
    fn default() -> Self {
        Self::new()
    }
}

impl NielsenController {
    /// Controller starting at the default damping of `1e-4`.
    pub const fn new() -> Self {
        Self::with_initial(DEFAULT_INITIAL_DAMPING)
    }

    /// Controller starting at `value`, which must be finite and positive.
    pub const fn with_initial(value: f64) -> Self {
        debug_assert!(value.is_finite() && value > 0.0);
        Self {
            value,
            nu: 2.0,
            last_gain_ratio: None,
        }
    }

    /// Current damping value.
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Gain ratio of the last [`Self::accept`] call, or `None` after a rejection or a
    /// non-finite ratio.
    pub const fn last_gain_ratio(self) -> Option<f64> {
        self.last_gain_ratio
    }

    /// Record a failed step and increase the damping.
    pub fn reject(&mut self) {
        self.last_gain_ratio = None;
        self.increase();
    }

    /// Judge a step by the ratio of actual to predicted cost reduction.
    ///
    /// Returns `true` and decreases the damping when the ratio is positive; otherwise
    /// returns `false` and increases it.
    pub fn accept(&mut self, actual_reduction: f64, predicted_reduction: f64) -> bool {
        let gain = if actual_reduction.is_finite()
            && predicted_reduction.is_finite()
            && predicted_reduction > 0.0
        {
            actual_reduction / predicted_reduction
        } else {
            f64::NAN
        };
        self.last_gain_ratio = gain.is_finite().then_some(gain);
        if !(gain.is_finite() && gain > 0.0) {
            self.increase();
            return false;
        }

        let centered = 2.0 * gain - 1.0;
        let factor = (1.0 - centered * centered * centered).max(1.0 / 3.0);
        self.value = (self.value * factor).max(MINIMUM_DAMPING);
        self.nu = 2.0;
        true
    }

    fn increase(&mut self) {
        self.value = (self.value * self.nu).min(MAXIMUM_DAMPING);
        self.nu = (self.nu * 2.0).min(MAXIMUM_DAMPING);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nielsen_uses_gain_and_rejection_history() {
        let mut damping = NielsenController::new();
        assert!(damping.accept(2.0, 2.0));
        assert_eq!(damping.last_gain_ratio(), Some(1.0));
        assert!((damping.value() - DEFAULT_INITIAL_DAMPING / 3.0).abs() < 1e-15);

        damping.reject();
        assert_eq!(damping.last_gain_ratio(), None);
        assert!((damping.value() - 2.0 * DEFAULT_INITIAL_DAMPING / 3.0).abs() < 1e-15);
        damping.reject();
        assert!((damping.value() - 8.0 * DEFAULT_INITIAL_DAMPING / 3.0).abs() < 1e-15);

        assert!(!damping.accept(-1.0, 1.0));
        assert_eq!(damping.last_gain_ratio(), Some(-1.0));
        assert!(!damping.accept(1.0, -1.0));
        assert_eq!(damping.last_gain_ratio(), None);
    }
}
