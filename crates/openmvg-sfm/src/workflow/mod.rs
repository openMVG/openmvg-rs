// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Immutable values at the public SfM workflow boundaries.
//!
//! These values wrap the internal workspace storage, so a dataset, prepared `ViewGraph`, or
//! reconstruction has one owner and one representation; callers see only the state
//! appropriate to that phase.

mod dataset;
mod engines;
mod extraction_method;
#[cfg(feature = "image-io")]
mod frontend;
mod graph_parts;
mod reconstruction;
mod reconstruction_parts;
pub(crate) mod run;
#[cfg(all(test, feature = "image-io"))]
mod tests;
mod view_graph;

#[cfg(feature = "image-io")]
pub use dataset::DatasetConfig;
pub use dataset::{Dataset, DatasetBuilder};
pub use engines::{Global, Incremental, Runner};
pub use extraction_method::ExtractionMethod;
#[cfg(feature = "image-io")]
pub use frontend::{Frontend, MatchingMethod};
pub use graph_parts::ViewGraphParts;
pub use reconstruction::{PlyFormat, Reconstruction, SfmOutput};
pub use reconstruction_parts::ReconstructionParts;
pub use view_graph::ViewGraph;

use run::RunReport;

// Bindings run the pipeline on a worker thread and hand its values back, so these values must
// stay shareable across threads.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<Dataset>();
    send_sync::<ViewGraph>();
    send_sync::<Reconstruction>();
    send_sync::<RunReport>();
    send_sync::<SfmOutput>();
};
