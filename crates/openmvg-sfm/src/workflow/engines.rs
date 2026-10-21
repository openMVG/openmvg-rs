// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental and Global engines and their per-run runner.

#[cfg(feature = "image-io")]
use super::{Dataset, Frontend};
use super::{Reconstruction, SfmOutput, ViewGraph};
use crate::Result;
use crate::config::IncrementalSfmConfig;
use crate::pipelines::global::GlobalSfmConfig;
use crate::workflow::run::{GlobalBackends, IncrementalBackends, SfmConfig, SfmEngine};

/// Incremental SfM with the default policy and native bundle adjustment: start from two views
/// and add the others one at a time.
///
/// Camera models are refined unless the graph [has calibrated
/// intrinsics](ViewGraph::has_calibrated_intrinsics), which stay fixed.
///
/// # How it works
///
/// 1. Choose the initial pair. A verified pair with enough parallax for a well-conditioned
///    triangulation and rotations consistent with its neighbors comes first; among those, the
///    pair in the largest connected part of the graph, then the one from which the most views
///    can register next, then the one with the most support. Its first view sits at the origin
///    and its second at unit distance, which fixes the frame and the scale.
/// 2. Grow: register the next view from the 3D points it already sees, with a robust absolute
///    pose, triangulate the tracks it adds, and check the result. A view that fails a check
///    waits and is retried later.
/// 3. Bundle-adjust as the reconstruction grows, and roll back a step that lowers its quality.
/// 4. Finish with rounds of bundle adjustment and outlier removal, refining the camera models
///    when they are not calibrated.
///
/// # When to use it
///
/// Incremental tries to register every view and records, in each
/// [`ViewVerdict`](crate::reports::ViewVerdict) of the report, why it left a view out. Choose it
/// when you want every view the data supports, and the reason for each one it leaves out. It
/// bundle-adjusts again as views join, so its run time grows faster with the number of views
/// than [`Global`]'s.
#[derive(Debug, Clone)]
pub struct Incremental {
    config: IncrementalSfmConfig,
}

impl Incremental {
    /// Use an explicit incremental policy. It is checked when execution begins.
    #[must_use]
    pub const fn new(config: IncrementalSfmConfig) -> Self {
        Self { config }
    }

    /// The incremental policy a run will execute.
    #[must_use]
    pub const fn config(&self) -> &IncrementalSfmConfig {
        &self.config
    }

    /// Reconstruct a reusable prepared graph with automatic bundle adjustment.
    pub fn run_prepared(&self, graph: &ViewGraph) -> Result<SfmOutput> {
        self.runner().run_prepared(graph)
    }

    /// [`Self::run_prepared`], consuming the graph so reconstruction releases its matching data
    /// instead of working on a copy.
    pub fn run_prepared_owned(&self, graph: ViewGraph) -> Result<SfmOutput> {
        self.runner().run_prepared_owned(graph)
    }

    /// Configure progress reporting, observer events, or a custom bundle adjuster for one run.
    pub fn runner(&self) -> Runner<'_> {
        let mut config = SfmConfig::for_engine(SfmEngine::Incremental);
        config.set_incremental(self.config.clone());
        Runner::new(config)
    }

    /// Prepare and reconstruct an image-backed dataset: `ViewGraph::build(dataset,
    /// &Frontend::default())`, then [`Self::run_prepared`] on the graph. To tune the front end,
    /// save its graph, or show progress bars, take the two steps yourself, with
    /// `Frontend::build_with_report` and [`Self::runner`].
    #[cfg(feature = "image-io")]
    pub fn run(&self, dataset: &Dataset) -> Result<SfmOutput> {
        self.run_prepared_owned(ViewGraph::build(dataset, &Frontend::default())?)
    }
}

impl Default for Incremental {
    fn default() -> Self {
        Self::new(IncrementalSfmConfig::default())
    }
}

/// Global SfM with the default policy and native bundle adjustment: estimate every rotation, then
/// every position, from the whole view graph at once.
///
/// Camera models are refined unless the graph [has calibrated
/// intrinsics](ViewGraph::has_calibrated_intrinsics), which stay fixed.
///
/// # How it works
///
/// 1. Keep the largest connected part of the view graph, drop the pairs whose relative rotation
///    disagrees with the others, and average the rotations of all views together.
/// 2. Check the averaged rotations against every verified pair. When too many views disagree,
///    the run stops with no reconstruction.
/// 3. Estimate the camera positions, with global positioning from the track bearings or with
///    translation averaging from the pairwise directions, triangulate the tracks, and
///    bundle-adjust, first with the rotations held and then with every pose and point free;
///    the camera models stay as given.
/// 4. Admit the result only when it passes consistency checks (rotation residuals,
///    reprojection error, observability) and a validation on held-out tracks; otherwise try the
///    next route of a fixed list.
/// 5. When the camera models are not calibrated, refine them in a last bundle adjustment of the
///    admitted result, and keep it only if the poses still agree with the verified pairs.
///
/// The views and points that Global admits form its *core*: the part of the view graph whose
/// rotations, positions and points agree with the verified pairs. Only the core is
/// reconstructed; the reports call views outside it `OutsideConsistentGlobalSubgraph`.
///
/// # When to use it
///
/// Global bundle-adjusts a few times rather than again as each view joins, so its run time grows
/// gently with the number of views, which matters most on large datasets. It also has no initial
/// pair and no growth order to get wrong.
///
/// It returns a reconstruction only when one passes its checks: otherwise the run succeeds with
/// no reconstruction, the report's [`final_route`](crate::reports::RunReport::final_route) is
/// `GlobalEmpty`, [`global_empty_reason`](crate::reports::RunReport::global_empty_reason) says
/// why, and views outside the core stay unregistered. Try [`Incremental`] when Global returns
/// nothing or leaves views out: it tries each view on its own.
#[derive(Debug, Clone)]
pub struct Global {
    config: GlobalSfmConfig,
}

impl Global {
    /// Use an explicit global policy. It is checked when execution begins.
    #[must_use]
    pub const fn new(config: GlobalSfmConfig) -> Self {
        Self { config }
    }

    /// The global-cascade policy a run will execute.
    #[must_use]
    pub const fn config(&self) -> &GlobalSfmConfig {
        &self.config
    }

    /// Reconstruct a reusable prepared graph with automatic bundle adjustment.
    pub fn run_prepared(&self, graph: &ViewGraph) -> Result<SfmOutput> {
        self.runner().run_prepared(graph)
    }

    /// [`Self::run_prepared`], consuming the graph so reconstruction releases its matching data
    /// instead of working on a copy.
    pub fn run_prepared_owned(&self, graph: ViewGraph) -> Result<SfmOutput> {
        self.runner().run_prepared_owned(graph)
    }

    /// Configure progress reporting, observer events, or a custom bundle adjuster for one run.
    pub fn runner(&self) -> Runner<'_> {
        let mut config = SfmConfig::for_engine(SfmEngine::Global);
        config.set_global(self.config.clone());
        Runner::new(config)
    }

    /// Prepare and reconstruct an image-backed dataset: `ViewGraph::build(dataset,
    /// &Frontend::default())`, then [`Self::run_prepared`] on the graph. To tune the front end,
    /// save its graph, or show progress bars, take the two steps yourself, with
    /// `Frontend::build_with_report` and [`Self::runner`].
    #[cfg(feature = "image-io")]
    pub fn run(&self, dataset: &Dataset) -> Result<SfmOutput> {
        self.run_prepared_owned(ViewGraph::build(dataset, &Frontend::default())?)
    }
}

impl Default for Global {
    fn default() -> Self {
        Self::new(GlobalSfmConfig::default())
    }
}

/// One engine run with optional hooks, created by [`Incremental::runner`] or [`Global::runner`].
///
/// Without hooks it is the same run as the engine's `run_prepared`.
///
/// ```no_run
/// use openmvg_sfm::{Incremental, ViewGraph, progress::SfmProgressSink};
///
/// fn reconstruct(graph: &ViewGraph, progress: &dyn SfmProgressSink) -> openmvg_sfm::Result<()> {
///     let output = Incremental::default()
///         .runner()
///         .progress(progress)
///         .run_prepared(graph)?;
///     println!("{} views registered", output.reconstruction().registered_view_count());
///     Ok(())
/// }
/// ```
#[must_use = "the runner does nothing until run_prepared() or run_prepared_owned() is called"]
pub struct Runner<'a> {
    pub(crate) config: SfmConfig,
    pub(crate) progress: &'a dyn crate::progress::SfmProgressSink,
    pub(crate) observer: Option<&'a mut dyn crate::events::SfmEventSink>,
    pub(crate) bundle_adjuster: Option<&'a mut dyn crate::ba::BundleAdjuster>,
    #[cfg(feature = "unstable-test-support")]
    pub(crate) bundle_adjustment_backend:
        Option<&'a dyn crate::ba::backend::BundleAdjustmentBackend>,
    pub(crate) incremental_backends: Option<IncrementalBackends<'a>>,
    pub(crate) global_backends: Option<GlobalBackends<'a>>,
}

impl std::fmt::Debug for Runner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runner")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<'a> Runner<'a> {
    const fn new(config: SfmConfig) -> Self {
        Self {
            config,
            progress: &(),
            observer: None,
            bundle_adjuster: None,
            #[cfg(feature = "unstable-test-support")]
            bundle_adjustment_backend: None,
            incremental_backends: None,
            global_backends: None,
        }
    }

    /// Run `config` as given, for the unstable workspace entry points [`Self::run_workspace`]
    /// and [`Self::run_in_place`].
    #[cfg(feature = "unstable-internals")]
    pub const fn for_config(config: SfmConfig) -> Self {
        Self::new(config)
    }

    /// Report the progress of every reconstruction stage to `sink`.
    pub const fn progress(mut self, sink: &'a dyn crate::progress::SfmProgressSink) -> Self {
        self.progress = sink;
        self
    }

    /// Send reconstruction events to `sink`, for example a Rerun recorder; observing does not
    /// change the result.
    #[cfg(feature = "observer")]
    pub const fn observer(mut self, sink: &'a mut dyn crate::events::SfmEventSink) -> Self {
        self.observer = Some(sink);
        self
    }

    /// Replace native bundle adjustment.
    ///
    /// The adjuster receives problems in the index-based [`crate::ba`] contract; the engines
    /// translate their IDs to indices, check every returned solution, and keep it only when it
    /// passes. A failed or rejected solve is never kept: depending on where it happens, the run
    /// goes on from the state before it, Global drops that route (and may end with no
    /// reconstruction), or the run ends with the error, as for a failure of Incremental's
    /// initial pair or a configuration or internal error.
    pub const fn bundle_adjuster(
        mut self,
        adjuster: &'a mut dyn crate::ba::BundleAdjuster,
    ) -> Self {
        self.bundle_adjuster = Some(adjuster);
        self
    }

    /// Replace bundle adjustment with a scripted backend that can also defer candidates at
    /// preflight, for orchestration contract tests. It takes precedence over
    /// [`Self::bundle_adjuster`].
    #[cfg(feature = "unstable-test-support")]
    pub const fn bundle_adjustment_backend(
        mut self,
        backend: &'a dyn crate::ba::backend::BundleAdjustmentBackend,
    ) -> Self {
        self.bundle_adjustment_backend = Some(backend);
        self
    }

    /// Replace the incremental registration and triangulation algorithms, for orchestration
    /// contract tests.
    #[cfg(feature = "unstable-test-support")]
    pub const fn incremental_backends(mut self, backends: IncrementalBackends<'a>) -> Self {
        self.incremental_backends = Some(backends);
        self
    }

    /// Replace the global averaging and triangulation algorithms, for orchestration contract
    /// tests.
    #[cfg(feature = "unstable-test-support")]
    pub const fn global_backends(mut self, backends: GlobalBackends<'a>) -> Self {
        self.global_backends = Some(backends);
        self
    }

    /// Reconstruct a reusable prepared graph.
    pub fn run_prepared(self, graph: &ViewGraph) -> Result<SfmOutput> {
        self.run_prepared_owned(graph.clone())
    }

    /// [`Self::run_prepared`], consuming the graph so reconstruction releases its matching data
    /// instead of working on a copy.
    pub fn run_prepared_owned(mut self, graph: ViewGraph) -> Result<SfmOutput> {
        self.config = graph_config(self.config, &graph);
        let engine = self.config.engine();
        let calibrated_intrinsics = graph.has_calibrated_intrinsics();
        let (workspace, report) = self.run_owned(graph.workspace)?;
        let reconstruction = Reconstruction::new(workspace, engine, calibrated_intrinsics);
        Ok(SfmOutput::new(reconstruction, report))
    }
}

/// Keep the camera models of a graph with calibrated intrinsics fixed.
pub(super) fn graph_config(mut config: SfmConfig, graph: &ViewGraph) -> SfmConfig {
    if graph.has_calibrated_intrinsics() {
        config.set_self_calibrate_intrinsics(false);
    }
    config
}
