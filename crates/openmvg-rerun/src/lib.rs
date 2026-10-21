// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lightweight Rerun recording for OpenMVG pipeline events.
//!
//! This leaf crate is the only place where OpenMVG depends on the Rerun SDK.
//! Reconstruction crates emit borrowed, behavior-neutral events and remain
//! independent of visualization and recording concerns.
//!
//! The recorder uses synchronous, backpressured, lossless SDK transport to a `.rrd` file. Metrics
//! and event messages are written for every delivered event; geometry snapshots follow
//! [`GeometryCapture`]. Pass the recorder as the event sink to
//! `openmvg_sfm::Runner::observer` (from `Incremental::runner()` or `Global::runner()`), then
//! call [`RerunRecorder::finish`] to flush the recording and inspect observer-only errors.
//! Camera centers are colored and labeled by model. Every model uses camera-frame axes, avoiding
//! an implied perspective frustum for fisheye and spherical cameras.
//! Initial-graph nodes use gray, amber, and green for unused, tentative, and committed cameras.
//! Current-graph nodes and reconstructed view edges use a stable colorblind-safe Viridis scale for
//! active landmark incidence; exact counts remain available in hover labels.
//! Camera and graph growth are recorded at every registration checkpoint, while landmark clouds
//! and residual summaries follow the configured [`GeometryCapture`] cadence.
//! Geometry is recentered at the first finite camera center before conversion to `f32`; the exact
//! `f64` offset is recorded under `/sfm/metadata/recenter_origin`.
//! Recordings are written with Rerun 0.36.3 and should be opened with a compatible viewer.
//!
//! ```no_run
//! use openmvg_rerun::{RerunRecorder, RerunRecorderConfig};
//!
//! let mut recorder = RerunRecorder::save("reconstruction.rrd", RerunRecorderConfig::default())?;
//! // Pass `&mut recorder` to `openmvg_sfm::Runner::observer`.
//! let report = recorder.finish();
//! assert!(report.first_error.is_none());
//! # Ok::<(), openmvg_rerun::RerunRecorderError>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

mod blueprint;
mod config;
mod frame;
mod recorder;

pub use config::{GeometryCapture, GraphLayers, RerunRecorderConfig};
pub use recorder::{RerunFinishReport, RerunRecorder, RerunRecorderError};
