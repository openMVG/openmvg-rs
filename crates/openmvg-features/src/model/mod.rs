// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The checked, serializable keypoint, descriptor, feature and match values that `openmvg-sfm`
//! stores for each image and image pair.
//!
//! Every constructor, and deserialization, checks them: finite values, matching counts, sorted
//! and duplicate-free matches. They are separate from what the detectors and matchers return
//! ([`ScaleSpaceFeatures`](crate::extraction::ScaleSpaceFeatures) and
//! [`FilteredMatch`](crate::matching::FilteredMatch)): build them from those outputs with each
//! type's constructor, such as [`Keypoint::try_new`] from a keypoint's `x`, `y`, `sigma` and
//! `orientation`, [`DescriptorSet::try_u8`] from the descriptor bytes, and
//! [`FeatureMatch::try_new`] from a match's query and target indices and its distance.

/// Declares the private wire struct behind `#[serde(try_from = "...")]` for a
/// model type: same fields and serde shape as the type itself, but the value is
/// accepted only if the type's `validate` passes. Deserialization therefore
/// enforces the same invariants as the constructors.
macro_rules! validated_wire {
    ($model:ident, $(#[$attr:meta])* struct $wire:ident { $($field:ident: $field_ty:ty),+ $(,)? }) => {
        #[derive(serde::Deserialize)]
        $(#[$attr])*
        struct $wire {
            $($field: $field_ty),+
        }

        impl TryFrom<$wire> for $model {
            type Error = FeatureDataError;

            fn try_from(wire: $wire) -> Result<Self, Self::Error> {
                let value = Self {
                    $($field: wire.$field),+
                };
                value.validate()?;
                Ok(value)
            }
        }
    };
}

mod descriptor_set;
mod error;
mod feature_set;
mod keypoint;
mod matches;

pub use descriptor_set::DescriptorSet;
pub use error::FeatureDataError;
pub use feature_set::{ExtractedFeatures, FeatureSet};
pub use keypoint::Keypoint;
pub use matches::{FeatureMatch, MatchSet};

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;
    use openmvg_core::model::FeatureId;

    #[test]
    fn keypoint_constructor_protects_geometry_invariants() {
        let keypoint = Keypoint::try_new(1.0, 2.0, 3.0, 0.5).unwrap();
        assert_eq!(
            (
                keypoint.x(),
                keypoint.y(),
                keypoint.scale(),
                keypoint.orientation()
            ),
            (1.0, 2.0, 3.0, 0.5)
        );
        assert_eq!(
            Keypoint::try_new(f32::NAN, 2.0, 3.0, 0.5),
            Err(FeatureDataError::NonFiniteKeypoint)
        );
        assert_eq!(
            Keypoint::try_new(1.0, 2.0, -1.0, 0.5),
            Err(FeatureDataError::NegativeKeypointScale)
        );
    }

    #[test]
    fn descriptor_storage_is_validated_and_borrowed_without_copying() {
        let descriptors = DescriptorSet::try_u8(vec![1, 2, 3, 4], 2).unwrap();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors.dimension(), 2);
        assert_eq!(descriptors.as_u8(), Some((&[1, 2, 3, 4][..], 2)));
        assert_eq!(descriptors.as_f32(), None);

        assert!(matches!(
            DescriptorSet::try_f32(vec![f32::NAN], 1),
            Err(FeatureDataError::NonFiniteDescriptor)
        ));
        assert!(matches!(
            DescriptorSet::try_u8(vec![1, 2, 3], 2),
            Err(FeatureDataError::InvalidDescriptorLength { .. })
        ));
    }

    #[test]
    fn detected_features_keep_keypoints_and_descriptors_aligned() {
        let keypoint = Keypoint::try_new(1.0, 2.0, 1.0, 0.0).unwrap();
        let mismatch = ExtractedFeatures::try_new(
            vec![keypoint],
            DescriptorSet::try_f32(Vec::new(), 1).unwrap(),
        );
        assert!(matches!(
            mismatch,
            Err(FeatureDataError::FeatureDescriptorCountMismatch { .. })
        ));
    }

    #[test]
    fn feature_set_checks_pixels_and_indices() {
        assert_eq!(
            FeatureSet::try_new(vec![Point2::new(f64::NAN, 0.0)]),
            Err(FeatureDataError::NonFinitePixel)
        );
        let set = FeatureSet::try_new(vec![Point2::new(3.0, 4.0)]).unwrap();
        assert_eq!(set.pixel(FeatureId::new(0)).unwrap(), Point2::new(3.0, 4.0));
        assert!(matches!(
            set.pixel(FeatureId::new(1)),
            Err(FeatureDataError::FeatureOutOfRange { .. })
        ));
    }

    #[test]
    fn match_set_is_sorted_and_duplicate_free() {
        let late = FeatureMatch::try_new(FeatureId::new(2), FeatureId::new(1), Some(0.5)).unwrap();
        let early = FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(3), None).unwrap();
        let set = MatchSet::try_new(vec![late, early]).unwrap();
        assert_eq!(set.as_slice(), &[early, late]);

        assert_eq!(
            MatchSet::try_new(vec![early, early]),
            Err(FeatureDataError::NonCanonicalMatchOrder)
        );
        assert_eq!(
            FeatureMatch::try_new(FeatureId::new(0), FeatureId::new(1), Some(-1.0)),
            Err(FeatureDataError::InvalidMatchDistance)
        );
    }

    #[test]
    fn validated_model_round_trips_through_serde() {
        let model = ExtractedFeatures::try_new(
            vec![Keypoint::try_new(1.0, 2.0, 3.0, 0.5).unwrap()],
            DescriptorSet::try_binary(vec![1, 2, 3, 4], 256, 1).unwrap(),
        )
        .unwrap();
        let encoded = serde_json::to_string(&model).unwrap();
        let decoded: ExtractedFeatures = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(
            decoded.descriptors().as_binary(),
            Some((&[1, 2, 3, 4][..], 256))
        );
    }

    #[test]
    fn deserialization_enforces_constructor_invariants() {
        let keypoint = r#"{"x":1.0,"y":2.0,"scale":-1.0,"orientation":0.0}"#;
        assert!(serde_json::from_str::<Keypoint>(keypoint).is_err());

        let descriptors = r#"{"u8":{"data":[1,2,3],"dim":2}}"#;
        assert!(serde_json::from_str::<DescriptorSet>(descriptors).is_err());
        let descriptors = r#"{"u8":{"data":[1,2,3,4],"dim":2}}"#;
        assert_eq!(
            serde_json::from_str::<DescriptorSet>(descriptors).unwrap(),
            DescriptorSet::try_u8(vec![1, 2, 3, 4], 2).unwrap()
        );

        let misaligned = r#"{"keypoints":[{"x":1.0,"y":2.0,"scale":1.0,"orientation":0.0}],
            "descriptors":{"f32":{"data":[],"dim":1}}}"#;
        assert!(matches!(
            serde_json::from_str::<ExtractedFeatures>(misaligned),
            Err(error) if error.to_string().contains("does not match descriptor count")
        ));

        assert!(
            serde_json::from_str::<FeatureMatch>(r#"{"first":0,"second":1,"distance":-1.0}"#)
                .is_err()
        );

        let unsorted = r#"{"matches":[{"first":2,"second":1,"distance":null},
            {"first":0,"second":3,"distance":null}]}"#;
        assert!(serde_json::from_str::<MatchSet>(unsorted).is_err());
    }
}
