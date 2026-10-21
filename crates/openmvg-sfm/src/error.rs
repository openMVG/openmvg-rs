// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The crate error type [`SfmError`], its [`Result`] alias, and the bundle-adjustment failure
//! summary carried by its BA variants.

use std::fmt;
use std::path::PathBuf;

use crate::ba::backend::{BaBackendFailure, BaCommitRejection, BaFailureKind, BaNotApplicable};
use crate::{PoseId, ViewId};
use openmvg_core::camera::CameraError;
use openmvg_core::model::EstimationError;
use openmvg_features::model::FeatureDataError;
use openmvg_features::tracking::TrackSetError;

/// Errors returned by the datasets, the front end, the engines and the file helpers of this
/// crate.
///
/// Wrapped library errors keep their typed [`std::error::Error::source`]; the
/// rendered message of every variant is stable because it is persisted in
/// reports and logs.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum SfmError {
    /// A step ran before the step that produces its input.
    #[error("missing prerequisite `{prerequisite}`; call `{method}` first")]
    MissingPrerequisite {
        /// Name of the missing input.
        prerequisite: &'static str,
        /// Method that produces the missing state.
        method: &'static str,
    },
    /// An identifier does not refer to an item of the dataset or reconstruction.
    #[error("invalid {kind} id {index}")]
    InvalidId {
        /// Kind of identifier (view, intrinsic, pose, track, ...).
        kind: &'static str,
        /// Raw index that failed to resolve.
        index: usize,
    },
    /// Caller-supplied data or an intermediate result failed validation.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A solver or estimator rejected its input; renders like [`Self::InvalidInput`].
    #[error("invalid input: {0}")]
    InvalidEstimationInput(#[from] EstimationError),
    /// Feature, descriptor, or match data is malformed; renders like
    /// [`Self::InvalidInput`].
    #[error("invalid input: {0}")]
    InvalidFeatureData(#[from] FeatureDataError),
    /// Feature tracks could not be built or are inconsistent.
    #[error("invalid feature tracks: {0}")]
    InvalidTracks(#[from] TrackSetError),
    /// A camera model or its parameters are invalid.
    #[error("invalid camera: {0}")]
    InvalidCamera(#[from] CameraError),
    /// An internal consistency check failed; indicates an implementation defect.
    #[error("internal pipeline invariant violated: {0}")]
    InvariantViolation(String),
    /// No image pair satisfied the configured initial-pair strategy.
    #[error("no initial-pair candidate satisfied the configured strategy")]
    NoInitialPairCandidate,
    /// A stage does not support the camera model of one of its views.
    #[error("{context} does not support camera model {camera_model}")]
    UnsupportedCameraModel {
        /// Stage that rejected the model.
        context: &'static str,
        /// Name of the rejected camera model.
        camera_model: &'static str,
    },
    /// Incremental or Global initialization does not support several views sharing one pose.
    /// Stored reconstructions and bundle-adjustment backends preserve this supported topology.
    #[error("SfM initialization does not support views sharing pose {pose}")]
    UnsupportedSharedPoseInitialization {
        /// Physical pose assigned to more than one input view.
        pose: PoseId,
    },
    /// A filesystem operation (dataset discovery, checkpoint I/O) failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// An input file (image header, intrinsics matrix) could not be read.
    ///
    /// Its message starts with `invalid input:`, like [`Self::InvalidInput`].
    #[error("invalid input: failed to read {}: {source}", path.display())]
    ReadFile {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// An image file could not be read for pixel decoding.
    ///
    /// Renders with the `failed to load image` prefix of [`Self::Image`].
    #[error("failed to load image {}: {source}", path.display())]
    ImageRead {
        /// Image that could not be read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The dataset directory does not exist.
    #[error("dataset not found: {path}")]
    DatasetNotFound {
        /// Requested dataset directory.
        path: PathBuf,
    },
    /// The dataset directory contains no supported image files.
    #[error("no images found in dataset: {path}")]
    NoImagesFound {
        /// Searched dataset directory.
        path: PathBuf,
    },
    /// An image file was read but could not be decoded or resized.
    #[error("failed to load image: {0}")]
    Image(String),
    /// A configuration value is outside its supported range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A view has no extracted features.
    #[error("missing features for view {0}")]
    MissingFeatures(ViewId),
    /// Matching data was released before a stage that needs it. The engines release the
    /// keypoints and matches of their own working copy once they no longer need them, to save
    /// memory; a [`ViewGraph`](crate::ViewGraph) you keep is never affected.
    #[error("matching data has been released")]
    MatchingDataReleased,
    /// The bundle-adjustment backend does not support the problem.
    #[error("bundle adjustment is not applicable: {0}")]
    BundleAdjustmentNotApplicable(#[source] BundleAdjustmentFailure),
    /// The bundle-adjustment backend failed while solving.
    #[error("bundle adjustment backend failed: {0}")]
    BundleAdjustmentBackendFailed(#[source] BundleAdjustmentFailure),
    /// A bundle-adjustment proposal failed the commit checks and was discarded.
    #[error("bundle adjustment proposal from {backend_name} was not committable: {reason:?}")]
    BundleAdjustmentProposalRejected {
        /// Backend that produced the proposal.
        backend_name: String,
        /// Commit check that rejected it.
        reason: BundleAdjustmentFailure,
    },
    /// A checkpoint was written by an unsupported format version.
    #[error("checkpoint version mismatch: found {found}, supported {supported}")]
    CheckpointVersionMismatch {
        /// Version stored in the checkpoint.
        found: u32,
        /// Version this build reads.
        supported: u32,
    },
    /// A checkpoint section failed its integrity check.
    #[error("checkpoint {section} checksum mismatch: found {found}, expected {expected}")]
    CheckpointChecksumMismatch {
        /// Checkpoint section whose checksum failed.
        section: &'static str,
        /// Checksum computed from the stored bytes.
        found: String,
        /// Checksum recorded in the checkpoint.
        expected: String,
    },
    /// The progress sink asked the run to stop; see
    /// [`SfmProgressSink::is_canceled`](crate::progress::SfmProgressSink::is_canceled).
    #[error("reconstruction canceled")]
    Canceled,
}

impl SfmError {
    /// Whether a transactional BA attempt may be rolled back without aborting the pipeline.
    ///
    /// Unsupported problems, rejected proposals, numerical failures, and bounded resource
    /// failures leave the preceding reconstruction valid. Configuration, invalid-problem, and
    /// internal failures instead indicate a caller or implementation defect and must propagate.
    pub(crate) fn is_recoverable_bundle_adjustment_failure(&self) -> bool {
        match self {
            Self::BundleAdjustmentBackendFailed(failure) => matches!(
                failure.kind(),
                BundleAdjustmentFailureKind::Numerical
                    | BundleAdjustmentFailureKind::ResourceExhausted
            ),
            Self::BundleAdjustmentNotApplicable(_)
            | Self::BundleAdjustmentProposalRejected { .. } => true,
            _ => false,
        }
    }

    /// Splits out a backend failure that exhausted a bounded resource; any other error is
    /// returned unchanged.
    pub(crate) fn into_resource_exhausted_failure(
        self,
    ) -> std::result::Result<BaBackendFailure, Self> {
        match self {
            Self::BundleAdjustmentBackendFailed(BundleAdjustmentFailure {
                detail: FailureDetail::Backend(failure),
            }) if failure.kind == BaFailureKind::ResourceExhausted => Ok(failure),
            error => Err(error),
        }
    }

    /// Wraps a backend's "not applicable" report.
    pub(crate) fn bundle_adjustment_not_applicable(reason: BaNotApplicable) -> Self {
        Self::BundleAdjustmentNotApplicable(BundleAdjustmentFailure {
            detail: FailureDetail::NotApplicable(reason),
        })
    }

    /// Wraps a backend failure.
    pub(crate) fn bundle_adjustment_backend_failed(failure: BaBackendFailure) -> Self {
        Self::BundleAdjustmentBackendFailed(BundleAdjustmentFailure {
            detail: FailureDetail::Backend(failure),
        })
    }
}

// The backend report types are internal (`backends`, unstable-internals), so the
// conversions are too: a stable `From` impl would make them reachable.
#[cfg(feature = "unstable-internals")]
impl From<BaNotApplicable> for SfmError {
    fn from(reason: BaNotApplicable) -> Self {
        Self::bundle_adjustment_not_applicable(reason)
    }
}

#[cfg(feature = "unstable-internals")]
impl From<BaBackendFailure> for SfmError {
    fn from(failure: BaBackendFailure) -> Self {
        Self::bundle_adjustment_backend_failed(failure)
    }
}

/// Result type of fallible SfM operations, with [`SfmError`] as the error.
pub type Result<T> = std::result::Result<T, SfmError>;

/// Stable category of a [`BundleAdjustmentFailure`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentFailureKind {
    /// The backend cannot represent the problem (unsupported capability).
    NotApplicable,
    /// The backend rejected its configuration.
    InvalidConfig,
    /// The solve failed numerically.
    Numerical,
    /// The solve exceeded a bounded resource.
    ResourceExhausted,
    /// The backend hit an internal defect.
    Internal,
    /// A returned proposal failed the commit checks.
    ProposalRejected,
}

/// Summary of a bundle-adjustment failure carried by the `BundleAdjustment*` variants of
/// [`SfmError`].
///
/// `Display` and `Debug` render the underlying report unchanged (the backend's own report, or,
/// for a rejected proposal, the engine's commit check that rejected it), so the messages of the
/// enclosing errors are stable; [`Self::kind`] classifies the failure for control flow.
#[derive(Clone, PartialEq, Eq)]
pub struct BundleAdjustmentFailure {
    detail: FailureDetail,
}

#[derive(Clone, PartialEq, Eq)]
enum FailureDetail {
    NotApplicable(BaNotApplicable),
    Backend(BaBackendFailure),
    Rejected(BaCommitRejection),
}

impl BundleAdjustmentFailure {
    /// Wraps the commit check that rejected a returned proposal.
    pub(crate) const fn proposal_rejected(reason: BaCommitRejection) -> Self {
        Self {
            detail: FailureDetail::Rejected(reason),
        }
    }

    /// Category of the failure.
    #[must_use]
    pub const fn kind(&self) -> BundleAdjustmentFailureKind {
        match &self.detail {
            FailureDetail::NotApplicable(_) => BundleAdjustmentFailureKind::NotApplicable,
            FailureDetail::Backend(failure) => match failure.kind {
                // The engines build every problem, so a backend that rejects one as malformed
                // points to an engine defect.
                BaFailureKind::InvalidProblem | BaFailureKind::Internal => {
                    BundleAdjustmentFailureKind::Internal
                }
                BaFailureKind::InvalidConfig => BundleAdjustmentFailureKind::InvalidConfig,
                BaFailureKind::Numerical => BundleAdjustmentFailureKind::Numerical,
                BaFailureKind::ResourceExhausted => BundleAdjustmentFailureKind::ResourceExhausted,
            },
            FailureDetail::Rejected(_) => BundleAdjustmentFailureKind::ProposalRejected,
        }
    }

    /// Backend that reported the failure; `None` for a rejected proposal, whose backend is named
    /// by [`SfmError::BundleAdjustmentProposalRejected`].
    #[must_use]
    pub fn backend_name(&self) -> Option<&str> {
        match &self.detail {
            FailureDetail::NotApplicable(reason) => Some(&reason.backend_name),
            FailureDetail::Backend(failure) => Some(&failure.backend_name),
            FailureDetail::Rejected(_) => None,
        }
    }

    /// Commit check that rejected a returned proposal.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn commit_rejection(&self) -> Option<BaCommitRejection> {
        match self.detail {
            FailureDetail::Rejected(reason) => Some(reason),
            _ => None,
        }
    }
}

impl fmt::Display for BundleAdjustmentFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.detail {
            FailureDetail::NotApplicable(reason) => fmt::Display::fmt(reason, formatter),
            FailureDetail::Backend(failure) => fmt::Display::fmt(failure, formatter),
            FailureDetail::Rejected(reason) => fmt::Debug::fmt(reason, formatter),
        }
    }
}

impl fmt::Debug for BundleAdjustmentFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.detail {
            FailureDetail::NotApplicable(reason) => fmt::Debug::fmt(reason, formatter),
            FailureDetail::Backend(failure) => fmt::Debug::fmt(failure, formatter),
            FailureDetail::Rejected(reason) => fmt::Debug::fmt(reason, formatter),
        }
    }
}

impl std::error::Error for BundleAdjustmentFailure {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ba::backend::BaUnsupportedCapability;

    fn backend_failure(kind: BaFailureKind) -> SfmError {
        SfmError::bundle_adjustment_backend_failed(BaBackendFailure::new(
            "test",
            kind,
            "injected failure",
        ))
    }

    #[test]
    fn bundle_adjustment_recoverability_is_an_explicit_closed_list() {
        assert!(
            backend_failure(BaFailureKind::Numerical).is_recoverable_bundle_adjustment_failure()
        );
        assert!(
            backend_failure(BaFailureKind::ResourceExhausted)
                .is_recoverable_bundle_adjustment_failure()
        );
        assert!(
            SfmError::bundle_adjustment_not_applicable(BaNotApplicable::new(
                "test",
                BaUnsupportedCapability::Other,
                "not applicable",
            ))
            .is_recoverable_bundle_adjustment_failure()
        );
        assert!(
            SfmError::BundleAdjustmentProposalRejected {
                backend_name: "test".to_owned(),
                reason: BundleAdjustmentFailure::proposal_rejected(BaCommitRejection::CostIncrease),
            }
            .is_recoverable_bundle_adjustment_failure()
        );

        for kind in [
            BaFailureKind::InvalidProblem,
            BaFailureKind::InvalidConfig,
            BaFailureKind::Internal,
        ] {
            assert!(!backend_failure(kind).is_recoverable_bundle_adjustment_failure());
        }
        assert!(
            !SfmError::InvalidConfig("invalid policy".to_owned())
                .is_recoverable_bundle_adjustment_failure()
        );
    }

    #[test]
    fn bundle_adjustment_failures_render_the_backend_report_unchanged() {
        let not_applicable = BaNotApplicable::new("test", BaUnsupportedCapability::Other, "why");
        let error = SfmError::bundle_adjustment_not_applicable(not_applicable.clone());
        assert_eq!(
            error.to_string(),
            format!("bundle adjustment is not applicable: {not_applicable}")
        );
        assert_eq!(
            format!("{error:?}"),
            format!("BundleAdjustmentNotApplicable({not_applicable:?})")
        );
        assert_eq!(
            std::error::Error::source(&error).map(ToString::to_string),
            Some(not_applicable.to_string())
        );

        let failure = BaBackendFailure::new("test", BaFailureKind::Internal, "boom");
        let error = SfmError::bundle_adjustment_backend_failed(failure.clone());
        assert_eq!(
            error.to_string(),
            format!("bundle adjustment backend failed: {failure}")
        );
        assert_eq!(
            format!("{error:?}"),
            format!("BundleAdjustmentBackendFailed({failure:?})")
        );
        let SfmError::BundleAdjustmentBackendFailed(summary) = &error else {
            panic!("backend failures convert to BundleAdjustmentBackendFailed");
        };
        assert_eq!(summary.kind(), BundleAdjustmentFailureKind::Internal);
        assert_eq!(summary.backend_name(), Some("test"));

        let reason = BaCommitRejection::WeakPoints { count: 3 };
        let error = SfmError::BundleAdjustmentProposalRejected {
            backend_name: "test".to_owned(),
            reason: BundleAdjustmentFailure::proposal_rejected(reason),
        };
        assert_eq!(
            error.to_string(),
            "bundle adjustment proposal from test was not committable: WeakPoints { count: 3 }"
        );
        assert_eq!(
            format!("{error:?}"),
            format!(
                "BundleAdjustmentProposalRejected {{ backend_name: \"test\", reason: {reason:?} }}"
            )
        );
        assert!(std::error::Error::source(&error).is_none());
    }
}
