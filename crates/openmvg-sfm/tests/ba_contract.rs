// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_sfm::{Global, Incremental, SfmOutput, ViewGraph, ba::BundleAdjuster};

#[test]
fn stable_engines_accept_the_reexported_ba_contract() {
    fn incremental(
        graph: &ViewGraph,
        adjuster: &mut dyn BundleAdjuster,
    ) -> openmvg_sfm::Result<SfmOutput> {
        Incremental::default()
            .runner()
            .bundle_adjuster(adjuster)
            .run_prepared(graph)
    }
    fn global(
        graph: &ViewGraph,
        adjuster: &mut dyn BundleAdjuster,
    ) -> openmvg_sfm::Result<SfmOutput> {
        Global::default()
            .runner()
            .bundle_adjuster(adjuster)
            .run_prepared(graph)
    }
    let _ = (incremental, global);
}
