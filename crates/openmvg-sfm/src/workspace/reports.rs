// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bookkeeping a workspace keeps about its run: per-stage item accounting and the
//! reproducibility manifest.

#[cfg(feature = "unstable-internals")]
use crate::Result;
#[cfg(feature = "unstable-internals")]
use crate::SfmWorkspace;
use crate::workspace::quality::QualityReport;
use openmvg_core::RunSeed;

/// Number of items a stage rejected for one reason.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct RejectionSummary {
    /// Rejection reason label.
    pub reason: String,
    /// Number of items rejected for this reason.
    pub count: usize,
}

/// Item accounting for one pipeline stage recorded in a workspace.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct StageReport {
    /// Stage name.
    pub stage: String,
    /// Number of items offered to the stage.
    pub input_count: usize,
    /// Number of items the stage processed.
    pub attempted_count: usize,
    /// Number of processed items that succeeded.
    pub success_count: usize,
    /// Number of processed items that were rejected.
    pub rejected_count: usize,
    /// Number of offered items the stage did not process.
    pub skipped_count: usize,
    /// Seed the stage derived for its randomized work, or `None` for a deterministic stage.
    pub seed: Option<RunSeed>,
    /// Digest of the stage configuration.
    pub config_digest: String,
    /// Non-fatal diagnostics raised by the stage.
    pub warnings: Vec<String>,
    /// Rejection counts broken down by reason.
    pub rejection_reasons: Vec<RejectionSummary>,
}

/// Reports accumulated by a workspace; snapshot rollbacks restore them with the geometry.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct WorkspaceReports {
    /// Stage reports in recording order.
    pub stages: Vec<StageReport>,
    /// Stored quality summary, or `None` when none is stored (cleared by a verified-prefix
    /// append, whose new domain invalidates it).
    pub quality: Option<QualityReport>,
}

/// Build, configuration, and determinism provenance of a run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub struct RunManifest {
    /// `openmvg-sfm` version.
    pub crate_version: String,
    /// Value of the `GIT_SHA` environment variable at compile time, or `None` if it was unset.
    pub git_sha: Option<String>,
    /// Enabled optional crate features among `image-io` and `parallel`.
    pub cargo_features: Vec<String>,
    /// `<arch>-<os>` of the running build (for example `x86_64-linux`).
    pub target_triple: String,
    /// `sha256:`-prefixed digest of the run configuration's canonical JSON, or `None` when no
    /// configuration was digested.
    pub config_digest: Option<String>,
    /// Versioned RNG engine and project-owned transformation protocol.
    pub rng_algorithm: String,
    /// Portable root-to-item seed derivation used before invoking an adapter.
    pub seed_derivation_algorithm: String,
    /// Workspace root seed from which every randomized stage derives its seed.
    pub root_seed: RunSeed,
    /// Rayon workers available to parallel pipeline and solver kernels.
    pub thread_count: usize,
    /// `ViewGraph` file schema version this build reads and writes, or `None` if unknown.
    pub checkpoint_schema: Option<u32>,
}

impl RunManifest {
    #[cfg(feature = "unstable-internals")]
    pub(crate) fn current(workspace: &SfmWorkspace) -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: option_env!("GIT_SHA").map(str::to_string),
            cargo_features: enabled_features(),
            target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            config_digest: None,
            rng_algorithm: openmvg_core::determinism::RNG_ALGORITHM.to_owned(),
            seed_derivation_algorithm: openmvg_core::determinism::SEED_DERIVATION_ALGORITHM
                .to_owned(),
            root_seed: workspace.root_seed(),
            thread_count: active_rayon_threads(),
            checkpoint_schema: Some(crate::checkpoint::VIEW_GRAPH_SCHEMA_VERSION),
        }
    }

    /// Describes the current build and the workspace's root seed, with the digest of `config`.
    #[cfg(feature = "unstable-internals")]
    pub fn with_config<T: serde::Serialize>(workspace: &SfmWorkspace, config: &T) -> Result<Self> {
        let mut manifest = Self::current(workspace);
        manifest.config_digest = Some(crate::determinism::stable_json_hash(config)?);
        Ok(manifest)
    }
}

/// The cargo features of this build that change its behavior.
#[cfg(feature = "unstable-internals")]
fn enabled_features() -> Vec<String> {
    let mut features = Vec::new();
    if cfg!(feature = "image-io") {
        features.push("image-io".to_owned());
    }
    if cfg!(feature = "parallel") {
        features.push("parallel".to_owned());
    }
    features
}

#[cfg(feature = "parallel")]
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn active_rayon_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "parallel"))]
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
const fn active_rayon_threads() -> usize {
    1
}

#[cfg(all(test, feature = "unstable-internals"))]
mod tests {
    use super::*;

    #[test]
    fn run_manifest_names_reproducibility_context() {
        let workspace = SfmWorkspace::builder().root_seed(42).build().unwrap();
        let manifest = RunManifest::current(&workspace);

        assert_eq!(
            manifest.rng_algorithm,
            openmvg_core::determinism::RNG_ALGORITHM
        );
        assert_eq!(manifest.root_seed, 42.into());
        assert_eq!(manifest.thread_count, active_rayon_threads());
        assert_eq!(
            manifest.seed_derivation_algorithm,
            openmvg_core::determinism::SEED_DERIVATION_ALGORITHM
        );
        assert!(manifest.config_digest.is_none());
        assert_eq!(
            manifest.checkpoint_schema,
            Some(crate::checkpoint::VIEW_GRAPH_SCHEMA_VERSION)
        );
        assert!(
            manifest
                .cargo_features
                .iter()
                .all(|feature| !feature.is_empty())
        );

        let configured =
            RunManifest::with_config(&workspace, &crate::config::GlobalAveragingConfig::default())
                .unwrap();
        assert!(
            configured
                .config_digest
                .as_deref()
                .is_some_and(|digest| digest.starts_with("sha256:"))
        );
    }
}
