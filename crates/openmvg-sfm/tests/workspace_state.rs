// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_sfm::{model::*, unstable::*, *};

fn image_meta() -> Vec<ImageMeta> {
    vec![
        ImageMeta::external_id("image_000.jpg", 640, 480),
        ImageMeta::external_id("image_001.jpg", 640, 480),
        ImageMeta::external_id("image_002.jpg", 640, 480),
    ]
}

fn intrinsics() -> Vec<Camera> {
    vec![Camera::try_pinhole(800.0, 800.0, 320.0, 240.0).unwrap()]
}

#[test]
fn builder_creates_deterministic_exhaustive_workspace() {
    let workspace = SfmWorkspace::builder()
        .image_meta(image_meta())
        .camera_models(intrinsics())
        .shared_intrinsic(IntrinsicId::new(0))
        .exhaustive_pairs()
        .build()
        .unwrap();

    workspace.validate().unwrap();
    assert_eq!(workspace.views()[0].id, ViewId::new(0));
    assert_eq!(workspace.views()[2].meta.name, "image_002.jpg");
    assert_eq!(workspace.pair_graph().len(), 3);
    assert_eq!(workspace.pair_graph().pairs()[0].first(), ViewId::new(0));
    assert_eq!(workspace.pair_graph().pairs()[0].second(), ViewId::new(1));
    assert_eq!(workspace.pair_graph().active_count(), 3);
}

#[test]
fn imported_pairs_normalize_duplicate_undirected_pairs() {
    let workspace = SfmWorkspace::builder()
        .image_meta(image_meta())
        .camera_models(intrinsics())
        .shared_intrinsic(IntrinsicId::new(0))
        .imported_pairs(vec![
            (ViewId::new(2), ViewId::new(0)),
            (ViewId::new(0), ViewId::new(2)),
            (ViewId::new(2), ViewId::new(1)),
        ])
        .build()
        .unwrap();

    assert_eq!(workspace.pair_graph().len(), 2);
    assert!(
        workspace
            .pair_graph()
            .contains_pair(ViewPair::new(ViewId::new(0), ViewId::new(2)))
    );
}

#[test]
fn pair_graph_adjacency_and_deactivation_are_deterministic() {
    let mut graph = PairGraph::exhaustive(3).unwrap();
    let pair = ViewPair::new(ViewId::new(0), ViewId::new(2));
    assert!(graph.contains_pair(pair));

    let neighbors: Vec<_> = graph
        .neighbors(ViewId::new(0))
        .unwrap()
        .iter()
        .map(|neighbor| neighbor.view.index())
        .collect();
    assert_eq!(neighbors, vec![1, 2]);

    graph.deactivate_pair(pair).unwrap();
    graph.deactivate_pair(pair).unwrap();
    assert!(!graph.is_active_pair(pair));
    assert_eq!(graph.active_count(), 2);
}

#[test]
fn sparse_store_tracks_populated_missing_take_and_reindex() {
    let mut store = SparseStore::new(4);
    store.set(1, "one").unwrap();
    store.set(3, "three").unwrap();

    assert_eq!(store.slot_count(), 4);
    assert_eq!(store.populated_count(), 2);
    assert!(!store.is_complete());
    assert_eq!(store.iter_missing().collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(store.take(1).unwrap(), Some("one"));
    assert!(!store.has(1));

    store.set(1, "one").unwrap();
    let populated: Vec<_> = store
        .iter_populated()
        .map(|(index, value)| (index, *value))
        .collect();
    assert_eq!(populated, vec![(1, "one"), (3, "three")]);
}

#[test]
fn validation_rejects_bad_intrinsic_assignment() {
    let err = SfmWorkspace::builder()
        .image_meta(image_meta())
        .camera_models(intrinsics())
        .shared_intrinsic(IntrinsicId::new(1))
        .build()
        .unwrap_err();

    assert_eq!(err.to_string(), "invalid intrinsic id 1");
}
