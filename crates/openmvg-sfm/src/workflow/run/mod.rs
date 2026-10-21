// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stable engine composition for pretracked SfM inputs.
//!
//! This module owns only engine routing; the recommended presets are the
//! `IncrementalSfmConfig` and `GlobalSfmConfig` defaults. Feature extraction, matching,
//! verification, and track building remain explicit frontend stages, while concrete numerical
//! implementations are supplied through engine-specific backend bundles.
//!
//! `config` holds the engine selection, policies, and backend bundles; `report` the versioned
//! [`RunReport`] and its per-view verdicts; `execute` the [`Runner`](crate::Runner) methods that
//! resolve backends and run an engine transactionally.

mod config;
mod execute;
mod report;

pub use config::{GlobalBackends, IncrementalBackends, SfmConfig, SfmEngine};
pub use report::{
    FinalRoute, GlobalEmptyReason, RUN_REPORT_SCHEMA_VERSION, RunReport, ViewOmissionReason,
    ViewVerdict, ViewVerdictStatus,
};
#[cfg(feature = "unstable-internals")]
pub use report::{GlobalOutcome, SfmOutcome};
