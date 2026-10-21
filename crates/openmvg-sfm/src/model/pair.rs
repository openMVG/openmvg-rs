// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matched and verified image pairs of a [`crate::ViewGraph`].

use nalgebra::{Matrix3, Rotation3};
use openmvg_core::ViewPair;
use openmvg_core::model::RelativePose;
use openmvg_features::model::FeatureMatch;

use crate::Result;
use crate::two_view::{GeometricModel, ModelEvidence};

/// Verified two-view geometry of an image pair, without the solver diagnostics.
///
/// The relative pose maps the first view's camera frame to the second's.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PairGeometry {
    /// Essential matrix `E` with `x_second^T E x_first = 0` on bearings, and the relative pose
    /// decomposed from it with a unit translation.
    ///
    /// A [`crate::ViewGraph`] always holds the pose. As input to
    /// [`crate::ViewGraph::try_from_parts`] it may be `None`, and the pose is then decomposed
    /// from `E` by the cheirality vote of the pair's inliers.
    Essential {
        /// Essential matrix.
        matrix: Matrix3<f64>,
        /// Relative pose decomposed from `matrix`.
        relative_pose: Option<RelativePose>,
    },
    /// Fundamental matrix `F` with `x_second^T F x_first = 0` on pixels, with a relative pose
    /// when one was recovered.
    Fundamental {
        /// Fundamental matrix.
        matrix: Matrix3<f64>,
        /// Relative pose recovered from `matrix`, if any.
        relative_pose: Option<RelativePose>,
    },
    /// Homography `H` mapping first-view pixels to second-view pixels, with a relative pose
    /// when one was recovered.
    Homography {
        /// Homography matrix.
        matrix: Matrix3<f64>,
        /// Relative pose decomposed from `matrix`, if any.
        relative_pose: Option<RelativePose>,
    },
    /// Relative rotation without an observable translation.
    RotationOnly {
        /// Rotation from the first view's camera frame to the second's.
        rotation: Rotation3<f64>,
    },
}

impl PairGeometry {
    pub(crate) fn of_model(model: &GeometricModel) -> Self {
        match *model {
            GeometricModel::Essential {
                matrix,
                relative_pose,
                ..
            } => Self::Essential {
                matrix,
                relative_pose: Some(relative_pose),
            },
            GeometricModel::Fundamental {
                matrix,
                relative_pose,
                ..
            } => Self::Fundamental {
                matrix,
                relative_pose,
            },
            GeometricModel::Homography {
                matrix,
                relative_pose,
                ..
            } => Self::Homography {
                matrix,
                relative_pose,
            },
            GeometricModel::RotationOnly { rotation, .. } => Self::RotationOnly { rotation },
        }
    }

    /// The model with `evidence`; `essential_pose` supplies a missing essential pose from the
    /// matrix and the inlier flags.
    pub(crate) fn into_model(
        self,
        evidence: ModelEvidence,
        essential_pose: impl FnOnce(&Matrix3<f64>, &[bool]) -> Result<RelativePose>,
    ) -> Result<GeometricModel> {
        Ok(match self {
            Self::Essential {
                matrix,
                relative_pose,
            } => GeometricModel::Essential {
                relative_pose: match relative_pose {
                    Some(relative_pose) => relative_pose,
                    None => essential_pose(&matrix, evidence.inliers.as_slice())?,
                },
                matrix,
                evidence,
            },
            Self::Fundamental {
                matrix,
                relative_pose,
            } => GeometricModel::Fundamental {
                matrix,
                relative_pose,
                evidence,
            },
            Self::Homography {
                matrix,
                relative_pose,
            } => GeometricModel::Homography {
                matrix,
                relative_pose,
                evidence,
            },
            Self::RotationOnly { rotation } => GeometricModel::RotationOnly { rotation, evidence },
        })
    }
}

/// One image pair of a [`crate::ViewGraph`]: its putative matches and, if accepted, its
/// geometry.
#[derive(Debug, Clone, Copy)]
pub struct GraphPair<'a> {
    pub(crate) views: ViewPair,
    pub(crate) matches: &'a [FeatureMatch],
    pub(crate) model: Option<&'a GeometricModel>,
}

impl<'a> GraphPair<'a> {
    /// The two views, first < second.
    #[must_use]
    pub const fn views(&self) -> ViewPair {
        self.views
    }

    /// Putative descriptor matches, sorted by `(first, second)` feature and unique. The first
    /// feature of a match belongs to the first view.
    #[must_use]
    pub const fn matches(&self) -> &'a [FeatureMatch] {
        self.matches
    }

    /// The verified geometry and one inlier flag per match, or `None` when verification
    /// rejected the pair. Only accepted pairs take part in tracks and averaging.
    #[must_use]
    pub fn verified(&self) -> Option<(PairGeometry, &'a [bool])> {
        self.model.map(|model| {
            (
                PairGeometry::of_model(model),
                model.evidence().inliers.as_slice(),
            )
        })
    }
}
