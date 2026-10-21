// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `openmvg-sfm`: reconstructs a folder of images with the stable SfM facade.
//!
//! ```text
//! image folder ──► EXIF grouping, K.txt, or --project ──► Dataset
//!                                                            │ Frontend (SIFT or ALP, HNSW)
//!                                                            ▼
//!                                                        ViewGraph
//!                                                            │ Incremental or Global
//!                                                            ▼
//!              reconstruction.ply, reconstruction.omvg ◄── Reconstruction + RunReport
//! ```
//!
//! The front end runs exactly once. Global SfM returns only a reconstruction whose rotations and
//! translations pass its consistency checks; when there is none, the command writes its reports
//! and fails.

#![forbid(unsafe_code)]
#![deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use openmvg_core::camera::CameraModelType;
use openmvg_io::WriteOptions;
use openmvg_sfm::config::{GlobalSfmConfig, IncrementalSfmConfig};
use openmvg_sfm::image::{
    ImageDatasetConfig, ImageIntrinsicPolicy, ImageProject, discover_dataset_images,
    find_calibration_file, image_size, load_pinhole_intrinsics_matrix, prepare_image_paths,
    prepare_image_project,
};
use openmvg_sfm::progress::{ConsoleProgress, SfmProgressSink};
use openmvg_sfm::reports::{FinalRoute, FrontendReport, RunReport};
use openmvg_sfm::{
    Camera, Dataset, DetectorDensity, ExtractionMethod, Frontend, Global, Incremental,
    MatchingMethod, Reconstruction, ViewGraph,
};
use serde::Serialize;

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Layout version of `sfm_report.json`.
const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Parser)]
#[command(
    name = "openmvg-sfm",
    version,
    about = "Reconstruct a folder of images with Incremental or Global SfM"
)]
struct Cli {
    /// Image directory, or a dataset root containing images/. Without --project, an image that
    /// cannot be read, such as an empty or truncated file, is skipped with a warning.
    images: PathBuf,

    /// Reconstruction strategy. Global returns only a reconstruction that passes its rotation
    /// and translation consistency checks.
    #[arg(long, value_enum, default_value_t = Engine::Incremental)]
    engine: Engine,

    /// Output directory (default: `images/sfm_output`).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Explicit K.txt path. By default it is discovered beside the images; EXIF grouping is used
    /// when no calibration file exists.
    #[arg(long)]
    intrinsics: Option<PathBuf>,

    /// Keypoint detector. Both emit RootSIFT descriptors, so matching is the same for both.
    #[arg(long, value_enum, default_value_t = DetectorArg::Alp)]
    detector: DetectorArg,

    /// Descriptor matcher: approximate HNSW search, or exact brute force.
    #[arg(long, value_enum, default_value_t = Matcher::Hnsw)]
    matcher: Matcher,

    /// Keep only each image's strongest verified pairs (by inliers, at either end) and deactivate
    /// the rest before tracking. Removes the weak pairs that chain repeated structures
    /// together; a view with at most this many pairs is untouched. Off by default.
    #[arg(long, value_name = "PAIRS")]
    max_pairs_per_view: Option<usize>,

    /// Detector density for either detector (default: high); `ultra` is SIFT only.
    #[arg(long, value_enum)]
    density: Option<DensityArg>,

    /// Maximum image edge used for feature extraction (default: 1600); zero keeps native
    /// resolution.
    #[arg(long)]
    max_dimension: Option<u32>,

    /// Multiplier applied to EXIF/image-size focal initialization when no K.txt is used.
    #[arg(long, default_value_t = 1.0)]
    focal_scale: f64,

    /// Give images with no camera make and model, and the same dimensions, one shared intrinsic
    /// group: one focal length and distortion are estimated from all of them. Use it when they
    /// come from one camera at one zoom setting.
    #[arg(long)]
    share_intrinsics: bool,

    /// Camera model declared for EXIF camera groups; distortion starts unknown and is estimated.
    /// A K.txt calibration is always a known pinhole camera.
    #[arg(long, value_enum, default_value_t = CameraModel::PinholeRadial1)]
    camera_model: CameraModel,

    /// Image project (JSON) that assigns the images to camera groups and sets each group's model
    /// and focal length: only the images it names are used, and its `self_calibrate` is the
    /// default of `--self-calibrate`. Replaces EXIF grouping, so it excludes `--intrinsics`,
    /// `--share-intrinsics`, `--camera-model`, and `--focal-scale`.
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with_all = ["intrinsics", "share_intrinsics", "camera_model", "focal_scale"]
    )]
    project: Option<PathBuf>,

    /// Deterministic workspace seed.
    #[arg(long, default_value_t = Dataset::DEFAULT_SEED)]
    seed: u64,

    /// Self-calibrate each camera group's declared model (focal, plus distortion for radial
    /// models) in bundle adjustment (`true`), or keep every intrinsic fixed (`false`). Defaults to
    /// self-calibrating EXIF camera groups; a K.txt calibration always stays fixed.
    #[arg(long)]
    self_calibrate: Option<bool>,

    /// Skip radial pre-calibration of EXIF radial camera groups (pinhole and K.txt cameras never
    /// get it).
    #[arg(long)]
    no_radial_calibration: bool,

    /// Also write the reconstruction as a COLMAP text model under `colmap/sparse/0`.
    #[arg(long)]
    colmap: bool,

    /// Hide the front-end progress bars, which are drawn on stderr only when it is a terminal.
    #[arg(long)]
    no_progress: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Engine {
    Incremental,
    Global,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CameraModel {
    /// Pinhole camera: focal and principal point, no distortion (COLMAP SIMPLE_PINHOLE).
    Pinhole,
    /// Pinhole camera plus one radial coefficient k1 (COLMAP SIMPLE_RADIAL).
    PinholeRadial1,
    /// Pinhole camera plus radial coefficients k1 and k2 (COLMAP RADIAL).
    PinholeRadial2,
    /// Equirectangular 360-degree images with a 2:1 aspect ratio; no focal length, EXIF ignored.
    Spherical,
}

impl CameraModel {
    const fn model_type(self) -> CameraModelType {
        match self {
            Self::Pinhole => CameraModelType::PinholeSingle,
            Self::PinholeRadial1 => CameraModelType::PinholeRadial1,
            Self::PinholeRadial2 => CameraModelType::PinholeRadial2,
            Self::Spherical => CameraModelType::Spherical,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DetectorArg {
    /// SIFT at the chosen density.
    Sift,
    /// Streaming ALP: one dominant orientation and canonical-grid descriptors.
    Alp,
}

impl DetectorArg {
    const fn extraction_method(self) -> ExtractionMethod {
        match self {
            Self::Sift => ExtractionMethod::Sift,
            Self::Alp => ExtractionMethod::Alp,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Matcher {
    /// Approximate nearest neighbours on per-image HNSW graphs; the fast default.
    Hnsw,
    /// Brute force over every descriptor pair; the exact reference.
    BruteForce,
}

impl Matcher {
    const fn matching_method(self) -> MatchingMethod {
        match self {
            Self::Hnsw => MatchingMethod::Hnsw,
            Self::BruteForce => MatchingMethod::BruteForce,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DensityArg {
    /// Standard contrast threshold (SIFT and ALP `Normal`).
    Normal,
    /// Lower contrast threshold for more features (SIFT and ALP `High`).
    High,
    /// SIFT `High` with a doubled first octave; SIFT only.
    Ultra,
}

impl DensityArg {
    const fn density(self) -> DetectorDensity {
        match self {
            Self::Normal => DetectorDensity::Normal,
            Self::High => DetectorDensity::High,
            Self::Ultra => DetectorDensity::Ultra,
        }
    }
}

/// How the images were grouped into cameras.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Grouping {
    Exif(ImageDatasetConfig),
    Project(ImageProject),
}

/// An image of the folder that could not be read and was left out.
#[derive(Serialize)]
struct SkippedImage {
    path: PathBuf,
    reason: String,
}

/// `sfm_report.json`: what the command ran and what it produced.
#[derive(Serialize)]
struct SfmReport<'a> {
    schema_version: u32,
    engine: Engine,
    seed: u64,
    self_calibrate: bool,
    grouping: &'a Grouping,
    skipped_images: &'a [SkippedImage],
    frontend: Frontend,
    frontend_report: &'a FrontendReport,
    final_route: FinalRoute,
    view_count: usize,
    registered_view_count: usize,
    point_count: usize,
    reconstruction: &'a RunReport,
    frontend_seconds: f64,
    reconstruction_seconds: f64,
    total_seconds: f64,
}

impl Cli {
    /// The front end the flags select, on top of the recommended [`Frontend::default`].
    fn frontend(&self) -> DynResult<Frontend> {
        let mut frontend = Frontend::default()
            .with_extraction_method(self.detector.extraction_method())
            .with_matching_method(self.matcher.matching_method())
            .with_radial_calibration(!self.no_radial_calibration)
            .with_max_pairs_per_view(self.max_pairs_per_view);
        if let Some(density) = self.density {
            if matches!(
                (self.detector, density),
                (DetectorArg::Alp, DensityArg::Ultra)
            ) {
                return Err(
                    "--density ultra needs --detector sift; raise --max-dimension for finer ALP \
                     scales"
                        .into(),
                );
            }
            frontend = frontend.with_density(density.density());
        }
        if let Some(max_dimension) = self.max_dimension {
            frontend = frontend.with_max_dimension((max_dimension != 0).then_some(max_dimension));
        }
        Ok(frontend)
    }
}

/// Focal and radial coefficients of a camera, for the per-group log.
fn describe_camera(camera: &Camera) -> String {
    if let Camera::Spherical(camera) = camera {
        return format!("spherical width={}", camera.width());
    }
    let focal = camera.focal_lengths().map_or(f64::NAN, |(focal, _)| focal);
    match camera {
        Camera::BrownConrady(camera) => format!(
            "f={focal:.1} k1={:.5} k2={:.5}",
            camera.radial_distortion().k1(),
            camera.radial_distortion().k2()
        ),
        _ => format!("f={focal:.1}"),
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> DynResult<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

/// Outputs a run writes into its output directory; finding any of them means a previous run
/// wrote there.
const RUN_OUTPUTS: [&str; 5] = [
    "sfm_report.json",
    "intrinsic_report.json",
    "reconstruction.ply",
    "reconstruction.omvg",
    "colmap",
];

/// Whether `output_dir` already holds the outputs of a previous run.
fn holds_previous_run(output_dir: &Path) -> bool {
    RUN_OUTPUTS
        .iter()
        .any(|name| output_dir.join(name).exists())
}

/// Write `reconstruction` as an OMVG dataset at `path`, replacing the OMVG dataset a previous run
/// left there; the writer refuses any other existing path.
fn replace_omvg(reconstruction: &Reconstruction, path: &Path, image_root: &Path) -> DynResult<()> {
    if path.join("manifest.json").is_file() {
        std::fs::remove_dir_all(path)?;
    }
    openmvg_io::write_reconstruction(
        reconstruction,
        path,
        &WriteOptions::new().with_image_root(image_root),
    )?;
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Splits `paths` into the images whose header can be read and the others, warning about each
/// image it leaves out.
fn readable_images(paths: Vec<PathBuf>) -> (Vec<PathBuf>, Vec<SkippedImage>) {
    let mut readable = Vec::with_capacity(paths.len());
    let mut skipped = Vec::new();
    for path in paths {
        match image_size(&path) {
            Ok(_) => readable.push(path),
            Err(error) => {
                let reason = error.to_string().trim_end().to_owned();
                eprintln!("warning: skipping an image that cannot be read: {reason}");
                skipped.push(SkippedImage { path, reason });
            }
        }
    }
    (readable, skipped)
}

fn run(cli: Cli) -> DynResult<()> {
    if !cli.focal_scale.is_finite() || cli.focal_scale <= 0.0 {
        return Err("--focal-scale must be finite and positive".into());
    }
    let frontend = cli.frontend()?;
    let total_started = Instant::now();
    let mut image_paths = discover_dataset_images(&cli.images)?;
    // A project names its images, and one it names must be readable; a plain folder leaves out
    // the files a failed copy or download left behind.
    let mut skipped_images = Vec::new();
    if cli.project.is_none() {
        (image_paths, skipped_images) = readable_images(image_paths);
    }
    if image_paths.len() < 2 {
        return Err(if skipped_images.is_empty() {
            "SfM requires at least two images".into()
        } else {
            format!(
                "SfM requires at least two images; {} could be read and {} could not",
                image_paths.len(),
                skipped_images.len()
            )
            .into()
        });
    }
    let (prepared_images, default_self_calibrate, grouping) = match &cli.project {
        Some(file) => {
            if !file.is_file() {
                return Err(format!(
                    "the project file {0} does not exist; create it with \
                     `openmvg-sfm-configure <folder> --project {0}`, pressing w to write it",
                    file.display()
                )
                .into());
            }
            let project = ImageProject::load(file)?;
            (
                prepare_image_project(&image_paths, &project)?,
                project.self_calibrate,
                Grouping::Project(project),
            )
        }
        None => {
            let intrinsics_path = cli
                .intrinsics
                .clone()
                .or_else(|| find_calibration_file(&cli.images, &image_paths[0]));
            let dataset_config = ImageDatasetConfig::default()
                .with_intrinsic_policy(match &intrinsics_path {
                    Some(path) => ImageIntrinsicPolicy::Shared {
                        camera: load_pinhole_intrinsics_matrix(path)?,
                    },
                    None => ImageIntrinsicPolicy::ExifGrouped,
                })
                .with_focal_scale(cli.focal_scale)
                .with_share_unknown_cameras(cli.share_intrinsics)
                .with_camera_model(cli.camera_model.model_type());
            (
                prepare_image_paths(&image_paths, &dataset_config)?,
                intrinsics_path.is_none(),
                Grouping::Exif(dataset_config),
            )
        }
    };
    let intrinsic_report = prepared_images.intrinsic_report.clone();
    let dataset = prepared_images.build_dataset(cli.seed)?;
    // An explicit `--self-calibrate` wins. Otherwise EXIF camera groups self-calibrate, a K.txt
    // calibration stays fixed, and a project says which. Groups the images do not constrain stay
    // fixed either way.
    if dataset.has_calibrated_intrinsics() && cli.self_calibrate == Some(true) {
        return Err(
            "a K.txt calibration stays fixed; drop --self-calibrate true, \
                    or group the images by EXIF or with --project to refine them"
                .into(),
        );
    }
    let self_calibrate = cli.self_calibrate.unwrap_or(default_self_calibrate);

    let console_progress = (!cli.no_progress).then(ConsoleProgress::stderr);
    let progress: &dyn SfmProgressSink = match &console_progress {
        Some(console_progress) => console_progress,
        None => &(),
    };
    let frontend_started = Instant::now();
    let (graph, frontend_report) = frontend.build_with_report(&dataset, progress)?;
    let frontend_seconds = frontend_started.elapsed().as_secs_f64();
    drop(dataset);
    // The cameras entering reconstruction, after radial pre-calibration.
    let input_cameras = graph.camera_models().to_vec();

    let reconstruction_started = Instant::now();
    let output = reconstruct(cli.engine, self_calibrate, graph, progress)?;
    let reconstruction_seconds = reconstruction_started.elapsed().as_secs_f64();
    let (reconstruction, run_report) = output.into_parts();
    let final_route = run_report.final_route();
    let has_reconstruction = run_report.has_reconstruction();

    let output_dir = cli
        .output
        .clone()
        .unwrap_or_else(|| cli.images.join("sfm_output"));
    if holds_previous_run(&output_dir) {
        eprintln!(
            "note: replacing the outputs of a previous run in {}",
            output_dir.display()
        );
    }
    std::fs::create_dir_all(&output_dir)?;
    if has_reconstruction {
        let image_root = image_paths[0].parent().unwrap_or(cli.images.as_path());
        reconstruction.write_ply(output_dir.join("reconstruction.ply"))?;
        replace_omvg(
            &reconstruction,
            &output_dir.join("reconstruction.omvg"),
            image_root,
        )?;
        if cli.colmap {
            openmvg_io::write_colmap(&reconstruction, output_dir.join("colmap"))?;
        }
    }
    let report = SfmReport {
        schema_version: REPORT_SCHEMA_VERSION,
        engine: cli.engine,
        seed: cli.seed,
        self_calibrate,
        grouping: &grouping,
        skipped_images: &skipped_images,
        frontend,
        frontend_report: &frontend_report,
        final_route,
        view_count: reconstruction.view_count(),
        registered_view_count: reconstruction.registered_view_count(),
        point_count: reconstruction.point_count(),
        reconstruction: &run_report,
        frontend_seconds,
        reconstruction_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    };
    write_json(&output_dir.join("sfm_report.json"), &report)?;
    write_json(&output_dir.join("intrinsic_report.json"), &intrinsic_report)?;

    if !has_reconstruction {
        return Err(format!(
            "global SfM found no consistent reconstruction; diagnostic report: {}",
            output_dir.join("sfm_report.json").display()
        )
        .into());
    }

    println!(
        "{}: {}/{} cameras, {} landmarks, {:.2}s frontend + {:.2}s reconstruction",
        final_route.as_str(),
        report.registered_view_count,
        report.view_count,
        report.point_count,
        frontend_seconds,
        reconstruction_seconds,
    );
    print_intrinsics(&reconstruction, &input_cameras, self_calibrate);
    println!("PLY: {}", output_dir.join("reconstruction.ply").display());
    println!(
        "OMVG (0.1-draft format, may change before 1.0): {}",
        output_dir.join("reconstruction.omvg").display()
    );
    Ok(())
}

/// Run the selected engine on the prepared graph, consuming it.
fn reconstruct(
    engine: Engine,
    self_calibrate: bool,
    graph: ViewGraph,
    progress: &dyn SfmProgressSink,
) -> openmvg_sfm::Result<openmvg_sfm::SfmOutput> {
    match engine {
        Engine::Incremental => Incremental::new(
            IncrementalSfmConfig::default().with_self_calibrate_intrinsics(self_calibrate),
        )
        .runner()
        .progress(progress)
        .run_prepared_owned(graph),
        Engine::Global => {
            Global::new(GlobalSfmConfig::default().with_self_calibrate_intrinsics(self_calibrate))
                .runner()
                .progress(progress)
                .run_prepared_owned(graph)
        }
    }
}

/// Log every camera group with a registered view: its camera before and after reconstruction.
fn print_intrinsics(
    reconstruction: &Reconstruction,
    input_cameras: &[Camera],
    self_calibrate: bool,
) {
    let mut registered_per_group = vec![0usize; input_cameras.len()];
    for (view, _) in reconstruction.registered_views() {
        if let Some(count) = reconstruction
            .views()
            .get(view.index())
            .and_then(|record| registered_per_group.get_mut(record.intrinsic_id.index()))
        {
            *count += 1;
        }
    }
    for (group, (input, refined)) in input_cameras
        .iter()
        .zip(reconstruction.camera_models())
        .enumerate()
        .filter(|(group, _)| registered_per_group[*group] > 0)
    {
        // Self-calibration writes a held group back bit-exactly.
        let state = if !self_calibrate {
            "fixed"
        } else if input == refined {
            "held"
        } else {
            "refined"
        };
        println!(
            "intrinsic {group} ({} views): {} -> {} ({state})",
            registered_per_group[group],
            describe_camera(input),
            describe_camera(refined),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreadable_images_are_skipped_and_named() -> DynResult<()> {
        // A 1x1 PNG, the empty file a failed copy leaves, and a JPEG cut off inside its header.
        const PNG: [u8; 67] = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        let root = tempfile::TempDir::new()?;
        let good = root.path().join("good.png");
        let empty = root.path().join("empty.jpg");
        let truncated = root.path().join("truncated.jpg");
        std::fs::write(&good, PNG)?;
        std::fs::write(&empty, b"")?;
        std::fs::write(&truncated, [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10])?;

        let (readable, skipped) =
            readable_images(vec![empty.clone(), good.clone(), truncated.clone()]);
        assert_eq!(readable, vec![good]);
        let skipped_paths: Vec<&PathBuf> = skipped.iter().map(|image| &image.path).collect();
        assert_eq!(skipped_paths, vec![&empty, &truncated]);
        for image in &skipped {
            assert!(image.reason.contains(&image.path.display().to_string()));
        }
        Ok(())
    }

    #[test]
    fn previous_run_outputs_are_detected() -> DynResult<()> {
        let root = tempfile::TempDir::new()?;
        let output = root.path().join("sfm_output");
        assert!(!holds_previous_run(&output));
        std::fs::create_dir_all(&output)?;
        assert!(!holds_previous_run(&output));
        std::fs::write(output.join("unrelated.txt"), b"")?;
        assert!(!holds_previous_run(&output));
        std::fs::write(output.join("sfm_report.json"), b"{}")?;
        assert!(holds_previous_run(&output));
        Ok(())
    }

    #[test]
    fn defaults_declare_simple_radial_and_the_recommended_frontend() -> DynResult<()> {
        let cli = Cli::try_parse_from(["openmvg-sfm", "images"])?;
        assert!(matches!(cli.engine, Engine::Incremental));
        assert!(matches!(cli.detector, DetectorArg::Alp));
        assert!(cli.density.is_none() && cli.max_dimension.is_none());
        assert!(cli.max_pairs_per_view.is_none());
        assert_eq!(cli.seed, Dataset::DEFAULT_SEED);
        assert_eq!(
            cli.camera_model.model_type(),
            CameraModelType::PinholeRadial1
        );
        assert!(cli.self_calibrate.is_none());
        assert!(!cli.no_radial_calibration && !cli.colmap && !cli.no_progress);
        // With no front-end flag the command runs the library's recommended front end.
        assert_eq!(cli.frontend()?, Frontend::default());
        Ok(())
    }

    #[test]
    fn a_project_replaces_the_grouping_flags() -> Result<(), clap::Error> {
        let with_project = |extra: &[&str]| {
            Cli::try_parse_from(
                ["openmvg-sfm", "images", "--project", "p.json"]
                    .into_iter()
                    .chain(extra.iter().copied()),
            )
        };
        // The calibration policy of the project stays overridable.
        with_project(&["--self-calibrate", "false"])?;
        for conflicting in [
            &["--intrinsics", "K.txt"][..],
            &["--camera-model", "pinhole"],
            &["--focal-scale", "2"],
            &["--share-intrinsics"],
        ] {
            assert_eq!(
                with_project(conflicting).err().map(|error| error.kind()),
                Some(clap::error::ErrorKind::ArgumentConflict),
                "{conflicting:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn frontend_flags_map_onto_the_stable_frontend() -> DynResult<()> {
        let cli = Cli::try_parse_from([
            "openmvg-sfm",
            "images",
            "--detector",
            "alp",
            "--matcher",
            "brute-force",
            "--max-dimension",
            "0",
        ])?;
        let frontend = cli.frontend()?;
        assert_eq!(frontend.extraction_method(), ExtractionMethod::Alp);
        assert_eq!(frontend.matching_method(), MatchingMethod::BruteForce);
        assert_eq!(
            frontend,
            Frontend::default()
                .with_extraction_method(ExtractionMethod::Alp)
                .with_matching_method(MatchingMethod::BruteForce)
                .with_max_dimension(None)
        );
        // The density applies to either detector; ultra is SIFT only.
        let cli = Cli::try_parse_from(["openmvg-sfm", "images", "--density", "normal"])?;
        let frontend = cli.frontend()?;
        assert_eq!(frontend.extraction_method(), ExtractionMethod::Alp);
        assert_eq!(frontend.density(), DetectorDensity::Normal);
        let cli = Cli::try_parse_from(["openmvg-sfm", "images", "--density", "ultra"])?;
        assert!(cli.frontend().is_err());
        let cli = Cli::try_parse_from([
            "openmvg-sfm",
            "images",
            "--detector",
            "sift",
            "--density",
            "ultra",
        ])?;
        assert_eq!(cli.frontend()?.density(), DetectorDensity::Ultra);
        Ok(())
    }
}
