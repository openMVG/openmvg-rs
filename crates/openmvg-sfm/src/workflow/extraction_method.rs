// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature extraction method of the standard front end.

/// Feature extraction method the `Frontend` (with `image-io`) extracts features with.
///
/// Both methods emit 128-D RootSIFT descriptors, so the front end's descriptor encoding and
/// `MatchingMethod` apply to either, and the front end's `DetectorDensity` selects the preset of
/// the same name for either; `Ultra` exists for SIFT only. A [`ViewGraph`](crate::ViewGraph) and its
/// [`FrontendReport`](crate::reports::FrontendReport) record the extraction method that built them.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMethod {
    /// SIFT: difference-of-Gaussian keypoints, with RootSIFT descriptors.
    Sift,
    /// ALP (Analytic Laplacian Polynomial): Laplacian keypoints at a continuous scale, one
    /// orientation each, with RootSIFT descriptors. The default.
    #[default]
    Alp,
}
