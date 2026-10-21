// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-rotation global camera/point positioning.
//!
//! Global positioning with a BATA-style bounded, depth-independent objective
//! (Zhuang et al., "Baseline Desensitizing in Translation Averaging", CVPR 2018),
//! as used by GLOMAP-style positioning; this is an independent implementation.
//! Camera-to-point bearings and optional oriented camera-to-camera directions
//! become BATA factors; the solver eliminates the points and their positive
//! scales and solves only the reduced system over the cameras.
//!
//! ```text
//! fixed rotations + bearing/direction edges -> BATA factors
//!                                          -> eliminate points/scales
//!                                          -> reduced camera solve
//!                                          -> global camera positions
//! ```
//!
//! # References
//!
//! - B. Zhuang, L.-F. Cheong, G. H. Lee, "Baseline Desensitizing in Translation Averaging",
//!   CVPR 2018.
//! - L. Pan, D. Baráth, M. Pollefeys, J. L. Schönberger, "Global Structure-from-Motion
//!   Revisited", ECCV 2024. arXiv:2407.20219.

mod kernel;
mod solver;

pub use solver::position_globally;

/// Internal evidence surface used only by allocation-contract tests.
#[cfg(feature = "unstable-test-support")]
#[doc(hidden)]
pub mod test_support {
    pub use super::kernel::{
        BataOptions, GlobalPositioningPlan, GlobalPositioningProblem, GlobalPositioningWorkspace,
        PositioningObservation, ReducedSystemBackend,
    };
}

use openmvg_core::RunSeed;

/// Global positioning policy for the BATA-style bounded objective.
#[derive(Debug, Clone)]
pub struct GlobalPositioningConfig {
    max_outer_iterations: usize,
    warmup_iterations: usize,
    initial_huber_delta_rad: f64,
    final_huber_delta_rad: f64,
    hard_cut_start_iteration: usize,
    hard_cut_deg: f64,
    seed: RunSeed,
    direction_reweight: f64,
    min_track_views: usize,
}

config_accessors!(GlobalPositioningConfig {
    /// Maximum robust outer iterations.
    copy max_outer_iterations, with_max_outer_iterations: usize;
    /// Initial iterations run with unit robust weights. When bearings and
    /// relative directions are both given, these iterations leave the
    /// directions out.
    copy warmup_iterations, with_warmup_iterations: usize;
    /// Initial angular Huber weight transition in radians, mapped to the
    /// alternation's chordal residual width.
    copy initial_huber_delta_rad, with_initial_huber_delta_rad: f64;
    /// Final angular Huber weight transition in radians. Alternation uses
    /// chord length and polish uses the forward sine residual; the angle
    /// locates each transition while their loss curves differ.
    copy final_huber_delta_rad, with_final_huber_delta_rad: f64;
    /// First iteration at which usual angular/cheirality outliers are cut.
    /// With bearings and relative directions together, relative directions
    /// use one quarter of `hard_cut_deg` after warmup while cuts are enabled,
    /// including iterations before this ordinary cut starts.
    /// A value at or beyond the outer-iteration budget disables angular cuts,
    /// including polish. Polish still requires finite forward baselines.
    copy hard_cut_start_iteration, with_hard_cut_start_iteration: usize;
    /// Gross angular rejection threshold in degrees.
    copy hard_cut_deg, with_hard_cut_deg: f64;
    /// Deterministic initialization seed.
    ///
    /// The standalone default uses a fixed seed. SfM replaces
    /// it with a domain-derived child of the run's caller-owned root seed.
    copy seed, with_seed: RunSeed;
    /// Aggregate camera-direction mass relative to camera-point bearing mass.
    /// Zero disables relative-direction constraints.
    copy direction_reweight, with_direction_reweight: f64;
    /// Minimum rotation-available views per admitted track. Values below two
    /// are clamped to two by the view-ID adapter.
    copy min_track_views, with_min_track_views: usize;
});

impl Default for GlobalPositioningConfig {
    fn default() -> Self {
        Self {
            max_outer_iterations: 20,
            warmup_iterations: 2,
            initial_huber_delta_rad: 0.5,
            final_huber_delta_rad: 0.05,
            hard_cut_start_iteration: 15,
            hard_cut_deg: 10.0,
            seed: RunSeed::from_u64(42),
            direction_reweight: 1.0,
            min_track_views: 2,
        }
    }
}
