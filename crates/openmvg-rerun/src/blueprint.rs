// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rerun's blueprint API is intentionally isolated here because it is less stable than logging.

use re_sdk::blueprint::{
    Blueprint, BlueprintPanel, GraphView, Horizontal, SelectionPanel, Spatial3DView, TextLogView,
    TimePanel, TimeSeriesView, Vertical,
};
use re_sdk_types::blueprint::components::PanelState;

use crate::recorder::{
    BEARING_QUALITY_ROOT, COVERAGE_ROOT, CURRENT_GRAPH_ROOT, CURRENT_WORLD_ROOT, EVENT_PATH,
    INITIAL_GRAPH_ROOT, LANDMARK_GROWTH_ROOT, PIXEL_QUALITY_ROOT, POSE_GRAPH_QUALITY_ROOT,
};

pub(crate) fn default_blueprint() -> Blueprint {
    let initial_graph =
        GraphView::new("Initial verified view graph").with_origin(INITIAL_GRAPH_ROOT);
    let current_graph = GraphView::new("Current reconstruction graph")
        .with_origin(format!("{CURRENT_GRAPH_ROOT}/registered_induced"));
    Blueprint::new(Horizontal::new([
        Spatial3DView::new("Reconstruction")
            .with_origin(CURRENT_WORLD_ROOT)
            .into(),
        Vertical::new([
            Horizontal::new([initial_graph.into(), current_graph.into()]).into(),
            Horizontal::new([
                TimeSeriesView::new("Landmark growth")
                    .with_origin(LANDMARK_GROWTH_ROOT)
                    .into(),
                TimeSeriesView::new("Camera coverage")
                    .with_origin(COVERAGE_ROOT)
                    .into(),
            ])
            .into(),
            Horizontal::new([
                TimeSeriesView::new("Bearing error (deg)")
                    .with_origin(BEARING_QUALITY_ROOT)
                    .into(),
                TimeSeriesView::new("Reprojection RMSE (px)")
                    .with_origin(PIXEL_QUALITY_ROOT)
                    .into(),
            ])
            .into(),
            TimeSeriesView::new("Pose-graph p90 (deg)")
                .with_origin(POSE_GRAPH_QUALITY_ROOT)
                .into(),
            TextLogView::new("Events").with_origin(EVENT_PATH).into(),
        ])
        .with_row_shares([2.0, 1.0, 1.0, 1.0, 1.0])
        .into(),
    ]))
    .with_auto_views(false)
    .with_auto_layout(false)
    .with_blueprint_panel(BlueprintPanel::from_state(PanelState::Hidden))
    .with_selection_panel(SelectionPanel::from_state(PanelState::Hidden))
    .with_time_panel(
        TimePanel::new()
            .with_state(PanelState::Hidden)
            .with_timeline("sfm_step")
            .with_fps(3.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_without_viewer_features() {
        let _ = default_blueprint();
    }
}
