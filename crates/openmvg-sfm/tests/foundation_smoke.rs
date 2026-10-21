// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_sfm::*;

#[test]
fn id_newtypes_are_explicit_and_orderable() {
    let view = ViewId::new(3);
    let pair = ViewPair::new(ViewId::new(4), ViewId::new(2));
    let feature = FeatureId::new(5);
    let intrinsic = IntrinsicId::new(6);
    let track = TrackId::new(7);
    let pose = PoseId::new(8);
    let landmark = LandmarkId::new(9);
    let component = ComponentId::new(10);

    assert_eq!(view.index(), 3);
    assert_eq!(pair.endpoints(), (ViewId::new(2), ViewId::new(4)));
    assert_eq!(FeatureId::from(5), feature);
    assert!(ViewId::new(1) < ViewId::new(2));
    assert_eq!(intrinsic.to_string(), "6");
    assert_eq!(track.index(), 7);
    assert_eq!(pose.index(), 8);
    assert_eq!(landmark.index(), 9);
    assert_eq!(component.index(), 10);
}

#[test]
fn sfm_error_messages_are_actionable() {
    let err = SfmError::MissingPrerequisite {
        prerequisite: "pair graph",
        method: "imported_pairs",
    };

    assert_eq!(
        err.to_string(),
        "missing prerequisite `pair graph`; call `imported_pairs` first"
    );
}
