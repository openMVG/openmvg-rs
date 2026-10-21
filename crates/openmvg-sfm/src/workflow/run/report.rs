// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The versioned run report and its per-view verdicts.

use super::config::{SfmConfig, SfmEngine};
use crate::pipelines::global::report::GlobalNoReconstructionReason;
use crate::pipelines::global::{GlobalCompletionReport, GlobalPreparationReport, GlobalSfmReport};
use crate::pipelines::incremental::{IncrementalSfmReport, UnregisteredViewDiagnostic};
use crate::{ComponentId, PoseId, Result, SfmError, SfmWorkspace, ViewId};

/// Schema version for [`RunReport`].
pub const RUN_REPORT_SCHEMA_VERSION: u32 = 1;

/// Concrete route taken by a completed SfM run.
///
/// `Global` may return no exportable reconstruction; [`RunReport::global_empty_reason`] then says
/// why.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalRoute {
    /// The Incremental engine ran; its reconstruction is exportable.
    Incremental,
    /// The Global engine found a core, the part of the view graph that passes its consistency
    /// checks ([`Global`](crate::Global) defines it); the reconstruction of the core is
    /// exportable.
    Global,
    /// The Global engine found no core; there is no reconstruction to export.
    GlobalEmpty,
}

impl FinalRoute {
    /// Route name, the same spelling as its JSON form (`incremental`, `global`,
    /// `global_empty`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Incremental => "incremental",
            Self::Global => "global",
            Self::GlobalEmpty => "global_empty",
        }
    }
}

/// Writes [`FinalRoute::as_str`].
impl std::fmt::Display for FinalRoute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why a Global run returned no reconstruction, from
/// [`RunReport::global_empty_reason`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalEmptyReason {
    /// The averaged rotations disagreed with too many verified pairs, so the engine did not go
    /// on to estimate positions. Repeated or symmetric structures often cause it: run
    /// [`Incremental`](crate::Incremental) on the same graph, or rebuild the graph with
    /// `Frontend::with_max_pairs_per_view` to drop weak pairs.
    RotationsInconsistent,
    /// No candidate reconstruction passed the consistency checks. Run
    /// [`Incremental`](crate::Incremental) on the same graph.
    NoCandidateAdmitted,
    /// The admitted reconstruction failed the validation on held-out tracks. Run
    /// [`Incremental`](crate::Incremental) on the same graph.
    TrackValidationFailed,
    /// Bundle adjustment needed more working memory than its budget allows. Run
    /// [`Incremental`](crate::Incremental) on the same graph, or rebuild the graph with fewer
    /// features per image (`Frontend::with_feature_budget`).
    ResourceLimit,
}

impl GlobalEmptyReason {
    /// Reason name in snake case, such as `rotations_inconsistent`, as the other report reasons
    /// spell theirs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RotationsInconsistent => "rotations_inconsistent",
            Self::NoCandidateAdmitted => "no_candidate_admitted",
            Self::TrackValidationFailed => "track_validation_failed",
            Self::ResourceLimit => "resource_limit",
        }
    }

    fn from_internal(reason: &GlobalNoReconstructionReason) -> Self {
        match reason {
            GlobalNoReconstructionReason::RotationConsensusRejected { .. } => {
                Self::RotationsInconsistent
            }
            GlobalNoReconstructionReason::NoCandidateAdmitted => Self::NoCandidateAdmitted,
            GlobalNoReconstructionReason::IndependentTrackValidation { .. } => {
                Self::TrackValidationFailed
            }
            GlobalNoReconstructionReason::ResourceCapacity { .. }
            | GlobalNoReconstructionReason::FixedRotationPreflight { .. } => Self::ResourceLimit,
        }
    }
}

/// Writes [`GlobalEmptyReason::as_str`].
impl std::fmt::Display for GlobalEmptyReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Final outcome for one input view.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ViewVerdictStatus {
    /// The view has an accepted pose; read it with
    /// [`Reconstruction::pose`](crate::Reconstruction::pose) and the verdict's view.
    Registered {
        /// Reconstruction component holding the pose; `None` if it is not assigned to one.
        component: Option<ComponentId>,
    },
    /// The view has no accepted pose.
    Unregistered {
        /// Why the view was left out.
        reason: ViewOmissionReason,
    },
}

/// Typed explanation for an input view that has no accepted pose.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum ViewOmissionReason {
    /// Incremental left the view out; the diagnostic says what the view had when the run ended.
    Incremental {
        /// The view's support and last registration attempt when the run ended.
        diagnostic: UnregisteredViewDiagnostic,
    },
    /// The Global engine found no core: no part of the view graph passes its consistency
    /// checks, as [`SfmEngine::Global`](crate::SfmEngine::Global) defines them.
    NoConsistentGlobalSubgraph,
    /// The Global engine found a core, but this view is outside it.
    OutsideConsistentGlobalSubgraph,
}

impl ViewOmissionReason {
    /// Reason name, the same spelling as the `reason` field of its JSON form, such as
    /// `outside_consistent_global_subgraph`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Incremental { .. } => "incremental",
            Self::NoConsistentGlobalSubgraph => "no_consistent_global_subgraph",
            Self::OutsideConsistentGlobalSubgraph => "outside_consistent_global_subgraph",
        }
    }
}

/// Writes [`ViewOmissionReason::as_str`]; for Incremental, followed by the view's last failed
/// registration attempt, such as `incremental: not_enough_inliers`, or `not_attempted`.
impl std::fmt::Display for ViewOmissionReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incremental { diagnostic } => {
                let last = diagnostic
                    .last_failure
                    .map_or("not_attempted", |reason| reason.as_str());
                write!(formatter, "{}: {last}", self.as_str())
            }
            _ => formatter.write_str(self.as_str()),
        }
    }
}

/// Deterministic accounting for one input view.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ViewVerdict {
    view: ViewId,
    pose: PoseId,
    status: ViewVerdictStatus,
}

impl ViewVerdict {
    /// The input view. Its image is `reconstruction.views()[verdict.view().index()].meta`, whose
    /// `name` is the image's file name.
    #[must_use]
    pub const fn view(&self) -> ViewId {
        self.view
    }

    /// The pose block the view references (possibly shared with other views).
    #[must_use]
    pub const fn pose(&self) -> PoseId {
        self.pose
    }

    /// Whether the view was registered, and why not if it was not.
    #[must_use]
    pub const fn status(&self) -> &ViewVerdictStatus {
        &self.status
    }

    /// Whether the view has an accepted pose.
    #[must_use]
    pub const fn is_registered(&self) -> bool {
        matches!(self.status, ViewVerdictStatus::Registered { .. })
    }
}

/// Versioned report for one engine run.
///
/// # Compatibility
///
/// The JSON form is versioned by [`Self::schema_version`] ([`RUN_REPORT_SCHEMA_VERSION`]), not
/// by the crate's semver. Schema 1 is the v0.1 layout.
///
/// Schema 1 keeps these top-level fields stable:
///
/// - `schema_version`: the layout version;
/// - `config`: the executed engine and both engine policies;
/// - `view_verdicts`: one [`ViewVerdict`] per input view;
/// - `outcome`: the engine-specific result, whose variants determine [`Self::final_route`].
///
/// The diagnostics nested in `outcome` may change within schema 1 without a version bump: the
/// bundle-adjustment timings and solver details, and the Global engine's per-stage reports.
///
/// Deserialization rejects any other `schema_version`, an embedded `config` that fails the
/// validation the engines apply before a run, and a Global outcome that contradicts its own
/// stage reports.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "RunReportWire")]
pub struct RunReport {
    schema_version: u32,
    config: SfmConfig,
    view_verdicts: Vec<ViewVerdict>,
    outcome: SfmOutcome,
}

#[derive(serde::Deserialize)]
#[serde(rename = "RunReport")]
struct RunReportWire {
    #[serde(deserialize_with = "deserialize_run_report_schema_version")]
    schema_version: u32,
    #[serde(deserialize_with = "deserialize_run_report_config")]
    config: SfmConfig,
    view_verdicts: Vec<ViewVerdict>,
    outcome: SfmOutcome,
}

impl TryFrom<RunReportWire> for RunReport {
    type Error = SfmError;

    fn try_from(wire: RunReportWire) -> Result<Self> {
        validate_run_completion(&wire.config, &wire.outcome)?;
        Ok(Self {
            schema_version: wire.schema_version,
            config: wire.config,
            view_verdicts: wire.view_verdicts,
            outcome: wire.outcome,
        })
    }
}

fn validate_run_completion(config: &SfmConfig, outcome: &SfmOutcome) -> Result<()> {
    let executed_engine = match outcome {
        SfmOutcome::Incremental { .. } => SfmEngine::Incremental,
        SfmOutcome::Global { .. } => SfmEngine::Global,
    };
    if config.engine() != executed_engine {
        return Err(SfmError::InvalidInput(
            "run configuration engine disagrees with outcome".to_owned(),
        ));
    }
    let consistent = match outcome {
        SfmOutcome::Global {
            result: GlobalOutcome::Reconstructed { report },
            ..
        } => {
            matches!(
                report.completion(),
                GlobalCompletionReport::Reconstructed { .. }
            )
        }
        SfmOutcome::Global {
            result: GlobalOutcome::NoReconstruction { report },
            ..
        } => {
            matches!(
                report.completion(),
                GlobalCompletionReport::NoReconstruction { .. }
            )
        }
        SfmOutcome::Incremental { .. } => true,
    };
    if !consistent {
        return Err(SfmError::InvalidInput(
            "global run outcome disagrees with cascade completion".to_owned(),
        ));
    }
    Ok(())
}

impl RunReport {
    /// Layout version of the serialized report ([`RUN_REPORT_SCHEMA_VERSION`] when produced).
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// The executed engine and its policy; the other engine's policy is recorded at its defaults.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn config(&self) -> &SfmConfig {
        &self.config
    }

    /// Exactly one verdict per input view, sorted by [`ViewId`].
    #[must_use]
    pub fn view_verdicts(&self) -> &[ViewVerdict] {
        &self.view_verdicts
    }

    /// Whether the run produced a reconstruction to export: false only when Global found none
    /// ([`FinalRoute::GlobalEmpty`]).
    #[must_use]
    pub const fn has_reconstruction(&self) -> bool {
        !matches!(self.final_route(), FinalRoute::GlobalEmpty)
    }

    /// Return the committed execution route without requiring callers to reinterpret nested
    /// engine-specific report variants.
    #[must_use]
    pub const fn final_route(&self) -> FinalRoute {
        match &self.outcome {
            SfmOutcome::Incremental { .. } => FinalRoute::Incremental,
            SfmOutcome::Global { result, .. } => match result {
                GlobalOutcome::Reconstructed { .. } => FinalRoute::Global,
                GlobalOutcome::NoReconstruction { .. } => FinalRoute::GlobalEmpty,
            },
        }
    }

    /// Why a Global run returned no reconstruction; `None` when the run has one.
    ///
    /// A Global run without a reconstruction is a valid outcome, not an error: its
    /// [`final_route`](Self::final_route) is [`FinalRoute::GlobalEmpty`], and the Incremental
    /// engine can run on the same view graph instead.
    #[must_use]
    pub fn global_empty_reason(&self) -> Option<GlobalEmptyReason> {
        match &self.outcome {
            SfmOutcome::Global {
                result: GlobalOutcome::NoReconstruction { report },
                ..
            } => match report.completion() {
                GlobalCompletionReport::NoReconstruction { reason, .. } => {
                    Some(GlobalEmptyReason::from_internal(reason))
                }
                GlobalCompletionReport::Reconstructed { .. } => None,
            },
            _ => None,
        }
    }

    /// The engine-specific report.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn outcome(&self) -> &SfmOutcome {
        &self.outcome
    }

    /// Consume the report, returning the engine-specific report.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub fn into_outcome(self) -> SfmOutcome {
        self.outcome
    }

    pub(super) fn try_new(
        workspace: &SfmWorkspace,
        config: SfmConfig,
        outcome: SfmOutcome,
    ) -> Result<Self> {
        validate_run_completion(&config, &outcome)?;
        let view_verdicts = build_view_verdicts(workspace, &outcome)?;
        Ok(Self {
            schema_version: RUN_REPORT_SCHEMA_VERSION,
            config,
            view_verdicts,
            outcome,
        })
    }
}

fn deserialize_run_report_schema_version<'de, D>(
    deserializer: D,
) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = <u32 as serde::Deserialize>::deserialize(deserializer)?;
    if version == RUN_REPORT_SCHEMA_VERSION {
        Ok(version)
    } else {
        Err(serde::de::Error::custom(format_args!(
            "unsupported RunReport schema_version {version} (this build reads {RUN_REPORT_SCHEMA_VERSION})"
        )))
    }
}

fn deserialize_run_report_config<'de, D>(
    deserializer: D,
) -> std::result::Result<SfmConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let config = <SfmConfig as serde::Deserialize>::deserialize(deserializer)?;
    config.validate().map_err(|error| {
        serde::de::Error::custom(format_args!("invalid RunReport config: {error}"))
    })?;
    Ok(config)
}

fn build_view_verdicts(workspace: &SfmWorkspace, outcome: &SfmOutcome) -> Result<Vec<ViewVerdict>> {
    use std::collections::BTreeMap;

    let incremental_diagnostic_count = match outcome {
        SfmOutcome::Incremental { report } => report.unregistered_views.len(),
        SfmOutcome::Global { .. } => 0,
    };
    let mut incremental_diagnostics = match outcome {
        SfmOutcome::Incremental { report } => report
            .unregistered_views
            .iter()
            .cloned()
            .map(|diagnostic| (diagnostic.view, diagnostic))
            .collect::<BTreeMap<_, _>>(),
        SfmOutcome::Global { .. } => BTreeMap::new(),
    };
    if incremental_diagnostics.len() != incremental_diagnostic_count {
        return Err(SfmError::InvariantViolation(
            "incremental run report contains duplicate per-view diagnostics".to_owned(),
        ));
    }

    let mut verdicts = Vec::with_capacity(workspace.views().len());
    for view in workspace.views() {
        let status = if let Some(pose) = workspace.pose(view.id) {
            ViewVerdictStatus::Registered {
                component: pose.component,
            }
        } else {
            let reason = match outcome {
                SfmOutcome::Incremental { .. } => {
                    let diagnostic = incremental_diagnostics.remove(&view.id).ok_or_else(|| {
                        SfmError::InvariantViolation(format!(
                            "incremental run report has no verdict diagnostic for {}",
                            view.id
                        ))
                    })?;
                    ViewOmissionReason::Incremental { diagnostic }
                }
                SfmOutcome::Global {
                    result: GlobalOutcome::Reconstructed { .. },
                    ..
                } => ViewOmissionReason::OutsideConsistentGlobalSubgraph,
                SfmOutcome::Global {
                    result: GlobalOutcome::NoReconstruction { .. },
                    ..
                } => ViewOmissionReason::NoConsistentGlobalSubgraph,
            };
            ViewVerdictStatus::Unregistered { reason }
        };
        verdicts.push(ViewVerdict {
            view: view.id,
            pose: view.pose_id,
            status,
        });
    }
    if !incremental_diagnostics.is_empty() {
        return Err(SfmError::InvariantViolation(
            "incremental run report contains diagnostics for registered or unknown views"
                .to_owned(),
        ));
    }
    Ok(verdicts)
}

/// Engine-specific result. Global success and rejection are distinct variants, and their
/// completion reports carry the core and input view counts.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "engine", rename_all = "snake_case")]
pub enum SfmOutcome {
    /// The Incremental engine ran.
    Incremental {
        /// Incremental run diagnostics.
        report: Box<IncrementalSfmReport>,
    },
    /// The Global engine ran.
    Global {
        /// View-graph conditioning applied before the cascade.
        preparation: GlobalPreparationReport,
        /// Whether a core was admitted, with the cascade report.
        result: GlobalOutcome,
    },
}

/// Outcome of a Global run: whether a reconstruction was committed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GlobalOutcome {
    /// A candidate passed the admission checks and its reconstruction was committed.
    Reconstructed {
        /// The global cascade report.
        report: Box<GlobalSfmReport>,
    },
    /// No candidate passed the admission checks; there is no reconstruction.
    NoReconstruction {
        /// The global cascade report, with the rejected attempts.
        report: Box<GlobalSfmReport>,
    },
}

#[cfg(test)]
mod tests {
    use super::super::config::SfmEngine;
    use super::*;
    use crate::pipelines::incremental::RegistrationFailureReason as Failure;

    /// Apps print these names; they must stay the JSON spelling of the same values.
    #[test]
    fn reason_names_match_their_json_spelling() {
        for route in [
            FinalRoute::Incremental,
            FinalRoute::Global,
            FinalRoute::GlobalEmpty,
        ] {
            assert_eq!(serde_json::to_value(route).unwrap(), route.as_str());
            assert_eq!(route.to_string(), route.as_str());
        }
        let failures = [
            Failure::AlreadyRegistered,
            Failure::NotEnoughVisibleLandmarks,
            Failure::PoseEstimationFailed,
            Failure::NotEnoughInliers,
            Failure::RotationInconsistent,
            Failure::TranslationInconsistent,
            Failure::PostRefitInsufficientSupport,
            Failure::PostBundleInsufficientSupport,
        ];
        for failure in failures {
            assert_eq!(serde_json::to_value(failure).unwrap(), failure.as_str());
            assert_eq!(failure.to_string(), failure.as_str());
        }
        let diagnostic = |last_failure| UnregisteredViewDiagnostic {
            view: ViewId::new(3),
            in_initial_pair_component: true,
            verified_neighbor_count: 2,
            verified_registered_neighbor_count: 1,
            observed_track_count: 40,
            visible_landmark_count: 9,
            tracks_with_posed_observer_count: 12,
            tracks_with_two_posed_observers_count: 8,
            last_failure,
        };
        let omissions = [
            ViewOmissionReason::Incremental {
                diagnostic: diagnostic(Some(Failure::NotEnoughInliers)),
            },
            ViewOmissionReason::NoConsistentGlobalSubgraph,
            ViewOmissionReason::OutsideConsistentGlobalSubgraph,
        ];
        for omission in &omissions {
            let json = serde_json::to_value(omission).unwrap();
            assert_eq!(json["reason"], omission.as_str());
        }
        assert_eq!(omissions[0].to_string(), "incremental: not_enough_inliers");
        assert_eq!(
            omissions[2].to_string(),
            "outside_consistent_global_subgraph"
        );
        let never = ViewOmissionReason::Incremental {
            diagnostic: diagnostic(None),
        };
        assert_eq!(never.to_string(), "incremental: not_attempted");

        let empty = [
            GlobalEmptyReason::RotationsInconsistent,
            GlobalEmptyReason::NoCandidateAdmitted,
            GlobalEmptyReason::TrackValidationFailed,
            GlobalEmptyReason::ResourceLimit,
        ];
        let names: std::collections::BTreeSet<_> = empty.iter().map(|r| r.as_str()).collect();
        assert_eq!(names.len(), empty.len(), "every reason has its own name");
        for reason in empty {
            assert!(
                reason
                    .as_str()
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_')
            );
            assert_eq!(reason.to_string(), reason.as_str());
        }
    }

    #[test]
    fn global_empty_reasons_map_to_plain_causes() {
        let cases = [
            (
                GlobalNoReconstructionReason::NoCandidateAdmitted,
                GlobalEmptyReason::NoCandidateAdmitted,
            ),
            (
                GlobalNoReconstructionReason::RotationConsensusRejected {
                    reasons: Vec::new(),
                },
                GlobalEmptyReason::RotationsInconsistent,
            ),
            (
                GlobalNoReconstructionReason::IndependentTrackValidation {
                    selected_attempt: 0,
                    report: None,
                    error: Some("rejected".to_owned()),
                },
                GlobalEmptyReason::TrackValidationFailed,
            ),
        ];
        for (internal, expected) in cases {
            assert_eq!(GlobalEmptyReason::from_internal(&internal), expected);
        }
    }

    #[test]
    fn v0_1_report_schema_versions_are_one() {
        assert_eq!(RUN_REPORT_SCHEMA_VERSION, 1);
        assert_eq!(
            crate::pipelines::global::report::GLOBAL_SFM_REPORT_SCHEMA_VERSION,
            1
        );
    }

    #[test]
    fn run_report_rejects_an_unsupported_schema_version() {
        let error = serde_json::from_str::<RunReport>(r#"{"schema_version":2}"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported RunReport schema_version 2")
        );
        let error = serde_json::from_str::<RunReport>(r#"{"schema_version":1}"#).unwrap_err();
        assert!(!error.to_string().contains("unsupported"));
    }

    #[test]
    fn run_report_rejects_an_invalid_embedded_config() {
        let report = |config: &SfmConfig| {
            serde_json::json!({ "schema_version": 1, "config": config }).to_string()
        };
        let mut invalid = SfmConfig::for_engine(SfmEngine::Incremental);
        let mut global = invalid.global().clone();
        global.global_positioning = global.global_positioning.with_min_track_views(1);
        invalid.set_global(global);

        let error = serde_json::from_str::<RunReport>(&report(&invalid)).unwrap_err();
        assert!(
            error.to_string().contains("invalid RunReport config"),
            "{error}"
        );
        let error = serde_json::from_str::<RunReport>(&report(&SfmConfig::default())).unwrap_err();
        assert!(error.to_string().contains("view_verdicts"), "{error}");
    }

    #[test]
    fn serialized_global_success_must_match_the_cascade_completion() {
        let component = serde_json::json!({
            "input_views": 1, "retained_views": 1, "excluded_views": 0,
            "input_edges": 0, "retained_edges": 0,
        });
        let preparation: GlobalPreparationReport = serde_json::from_value(serde_json::json!({
            "initial_largest_component": component,
            "rotation_conditioning": {"method": "none", "input_edges": 0, "kept_edges": 0,
                "hard_removed": 0, "bridges_protected": 0, "median_confidence": 0.0, "rotation_solves": 0},
            "final_largest_component": component,
            "candidate_domain_camera_count": 0, "candidate_domain_camera_ids": [],
        })).unwrap();
        let cascade = GlobalSfmReport::try_new(
            crate::pipelines::global::GlobalSfmConfig::default(),
            None,
            Vec::new(),
            GlobalCompletionReport::NoReconstruction {
                reason: crate::pipelines::global::report::GlobalNoReconstructionReason::NoCandidateAdmitted,
                input_view_count: 1,
            },
        )
        .unwrap();
        let outcome = SfmOutcome::Global {
            preparation,
            result: GlobalOutcome::NoReconstruction {
                report: Box::new(cascade),
            },
        };
        let verdict = ViewVerdict {
            view: ViewId::new(0),
            pose: PoseId::new(0),
            status: ViewVerdictStatus::Unregistered {
                reason: ViewOmissionReason::NoConsistentGlobalSubgraph,
            },
        };
        let mut wire = serde_json::json!({
            "schema_version": 1, "config": SfmConfig::for_engine(SfmEngine::Global),
            "view_verdicts": [verdict], "outcome": outcome,
        });
        let restored: RunReport = serde_json::from_value(wire.clone()).unwrap();
        assert!(!restored.has_reconstruction());
        let mut wrong_engine = wire.clone();
        wrong_engine["config"]["engine"] = "incremental".into();
        assert!(
            serde_json::from_value::<RunReport>(wrong_engine)
                .unwrap_err()
                .to_string()
                .contains("engine disagrees")
        );
        wire["outcome"]["result"]["status"] = "reconstructed".into();
        let error = serde_json::from_value::<RunReport>(wire).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("disagrees with cascade completion"),
            "{error}"
        );
    }

    #[test]
    fn engine_names_are_stable() {
        assert_eq!(SfmEngine::Incremental.as_str(), "incremental");
        assert_eq!(SfmEngine::Global.as_str(), "global");
        assert_eq!(FinalRoute::Global.as_str(), "global");
        assert_eq!(FinalRoute::GlobalEmpty.as_str(), "global_empty");

        assert_eq!(serde_json::to_value(FinalRoute::Global).unwrap(), "global");
        assert_eq!(
            serde_json::to_value(FinalRoute::GlobalEmpty).unwrap(),
            "global_empty"
        );
    }
}
