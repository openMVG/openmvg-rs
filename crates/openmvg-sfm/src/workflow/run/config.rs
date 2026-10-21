// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Engine selection, engine policies, and the backend bundles a run executes with.

use crate::Result;
use crate::backends::{PoseRegistrar, RotationAveragingFn, TranslationAveragingFn, Triangulator};
use crate::config::IncrementalSfmConfig;
use crate::pipelines::global::GlobalSfmConfig;

/// Supported reconstruction engines.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SfmEngine {
    /// Incremental SfM, [`Incremental`](crate::Incremental): grow the reconstruction from two
    /// views, one view at a time.
    #[default]
    Incremental,
    /// Global SfM, [`Global`](crate::Global): average the rotations once, then try up to six
    /// position attempts in a fixed order: with the tracks seen by at least three views, then
    /// with two-view tracks too, then on a smaller recovery subgraph, each time global
    /// positioning from the track bearings before translation averaging from the pairwise
    /// directions.
    ///
    /// The run keeps the first attempt that passes its checks. Its views and points form the
    /// [core](crate::Global#how-it-works): only the core is reconstructed, and views outside it
    /// stay unregistered.
    Global,
}

impl SfmEngine {
    /// Lowercase engine name, the same spelling as its JSON form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Incremental => "incremental",
            Self::Global => "global",
        }
    }
}

/// Incremental registration and triangulation algorithms; bundle adjustment is chosen on the
/// [`Runner`](crate::Runner).
#[derive(Clone, Copy)]
pub struct IncrementalBackends<'a> {
    pub(super) registrar: &'a dyn PoseRegistrar,
    pub(super) triangulator: &'a dyn Triangulator,
}

impl std::fmt::Debug for IncrementalBackends<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalBackends")
            .finish_non_exhaustive()
    }
}

impl<'a> IncrementalBackends<'a> {
    /// Bundle a 2D-3D pose registrar and a track triangulator.
    #[must_use]
    pub const fn new(registrar: &'a dyn PoseRegistrar, triangulator: &'a dyn Triangulator) -> Self {
        Self {
            registrar,
            triangulator,
        }
    }
}

/// Global averaging and triangulation algorithms; bundle adjustment is chosen on the
/// [`Runner`](crate::Runner).
#[derive(Clone, Copy)]
pub struct GlobalBackends<'a> {
    pub(super) rotation: RotationAveragingFn,
    pub(super) translation: TranslationAveragingFn,
    pub(super) bootstrap_triangulator: &'a dyn Triangulator,
    pub(super) robust_triangulator: &'a dyn Triangulator,
}

impl std::fmt::Debug for GlobalBackends<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalBackends").finish_non_exhaustive()
    }
}

impl<'a> GlobalBackends<'a> {
    /// Bundle the rotation and translation averagers with two triangulators.
    ///
    /// `bootstrap_triangulator` builds the initial structure from averaged poses;
    /// `robust_triangulator` serves candidate admission, cleanup, and retriangulation.
    #[must_use]
    pub const fn new(
        rotation: RotationAveragingFn,
        translation: TranslationAveragingFn,
        bootstrap_triangulator: &'a dyn Triangulator,
        robust_triangulator: &'a dyn Triangulator,
    ) -> Self {
        Self {
            rotation,
            translation,
            bootstrap_triangulator,
            robust_triangulator,
        }
    }
}

/// Typed engine policy over an already verified and tracked workspace.
///
/// Its JSON form rejects unknown keys and is versioned by the `schema_version` of the
/// [`RunReport`](super::RunReport) that embeds it, not by the crate's semver.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SfmConfig {
    engine: SfmEngine,
    incremental: IncrementalSfmConfig,
    global: GlobalSfmConfig,
}

impl Default for SfmConfig {
    fn default() -> Self {
        Self::for_engine(SfmEngine::Incremental)
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl SfmConfig {
    /// The default policies of both engines, with `engine` selected.
    #[must_use]
    pub fn for_engine(engine: SfmEngine) -> Self {
        Self {
            engine,
            incremental: IncrementalSfmConfig::default(),
            global: GlobalSfmConfig::default(),
        }
    }

    /// The engine a run executes.
    #[must_use]
    pub const fn engine(&self) -> SfmEngine {
        self.engine
    }

    /// Select the engine a run executes; the other engine's policy is kept but unused.
    pub fn set_engine(&mut self, engine: SfmEngine) {
        self.engine = engine;
    }

    /// The incremental-engine policy.
    #[must_use]
    pub const fn incremental(&self) -> &IncrementalSfmConfig {
        &self.incremental
    }

    /// Replace the incremental-engine policy.
    pub fn set_incremental(&mut self, config: IncrementalSfmConfig) {
        self.incremental = config;
    }

    /// The global-cascade policy.
    #[must_use]
    pub const fn global(&self) -> &GlobalSfmConfig {
        &self.global
    }

    /// Replace the global-cascade policy.
    pub fn set_global(&mut self, config: GlobalSfmConfig) {
        self.global = config;
    }

    /// Refine each intrinsic group's declared camera model, or keep every intrinsic fixed, in
    /// both engines.
    pub fn set_self_calibrate_intrinsics(&mut self, enabled: bool) {
        self.incremental = self
            .incremental
            .clone()
            .with_self_calibrate_intrinsics(enabled);
        self.global = self.global.clone().with_self_calibrate_intrinsics(enabled);
    }

    /// Validate both engine policies, whichever engine is selected.
    ///
    /// # Errors
    ///
    /// The first [`SfmError::InvalidConfig`](crate::SfmError::InvalidConfig) reported by the
    /// incremental, then the global, policy.
    pub fn validate(&self) -> Result<()> {
        self.incremental.validate()?;
        self.global.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_native_incremental() {
        let config = SfmConfig::default();
        assert_eq!(config.engine(), SfmEngine::Incremental);
        assert!(config.validate().is_ok());
        assert!(config.incremental().bundle_adjust_initial_pair());
        assert_eq!(config.incremental().min_registration_support(), 12);
        assert_eq!(config.incremental().registration_min_inliers(), 8);
        assert_eq!(config.incremental().bundle_adjust_growth_ratio(), 0.1);
    }

    #[test]
    fn validation_checks_every_serialized_engine_policy() {
        let mut config = SfmConfig::for_engine(SfmEngine::Incremental);
        config.global.global_positioning = config.global.global_positioning.with_min_track_views(1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_invalid_incremental_policy_before_execution() {
        let mut config = SfmConfig::for_engine(SfmEngine::Incremental);
        let incremental = config
            .incremental()
            .clone()
            .with_registration_min_inlier_ratio(f64::NAN);
        config.set_incremental(incremental);
        assert!(config.validate().is_err());
    }

    #[test]
    fn removed_global_config_keys_fail_closed() {
        let mut config_json = serde_json::to_value(SfmConfig::default()).unwrap();
        let fields = config_json.as_object_mut().unwrap();
        let global = fields.remove("global").unwrap();
        fields.insert("adaptive_global".to_owned(), global);
        assert!(serde_json::from_value::<SfmConfig>(config_json).is_err());

        let mut global = serde_json::to_value(GlobalSfmConfig::default()).unwrap();
        global.as_object_mut().unwrap().insert(
            "track_covisibility_core".to_owned(),
            serde_json::Value::Null,
        );
        assert!(serde_json::from_value::<GlobalSfmConfig>(global).is_err());
    }
}
