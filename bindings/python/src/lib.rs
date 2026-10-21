// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native extension of the `openmvg-rs` Python package, imported as `openmvg_rs._native`.
//!
//! The Python layer (`python/openmvg_rs/`) validates and normalizes the arguments.
//! This module runs a reconstruction on a worker thread with the GIL released, polls for Python
//! signals while the caller waits, and turns a signal into the library's cooperative
//! cancellation. The `geometry` module holds the robust geometric estimators.

#![forbid(unsafe_code)]

mod geometry;

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use openmvg_core::camera::{
    Camera, FisheyeDistortion, PinholeCamera, RadialDistortion, TangentialDistortion,
};
use openmvg_sfm::progress::{ConsoleProgress, SfmProgressSink, SfmProgressStage};
use openmvg_sfm::{
    Dataset, DatasetConfig, ExtractionMethod, Frontend, Global, Incremental, SfmError, checkpoint,
};
use pyo3::create_exception;
use pyo3::exceptions::{
    PyException, PyFileNotFoundError, PyKeyboardInterrupt, PyOSError, PyPermissionError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyTuple};

create_exception!(
    openmvg_rs,
    OpenMVGError,
    PyException,
    "Base class of the errors raised by openmvg_rs."
);
create_exception!(
    openmvg_rs,
    DatasetError,
    OpenMVGError,
    "The image folder or its calibration could not be loaded."
);
create_exception!(
    openmvg_rs,
    ReconstructionError,
    OpenMVGError,
    "Feature matching or reconstruction failed."
);
create_exception!(
    openmvg_rs,
    NoReconstructionError,
    ReconstructionError,
    "The run completed without registering any view."
);
create_exception!(
    openmvg_rs,
    ExportError,
    OpenMVGError,
    "A reconstruction could not be written."
);

create_exception!(
    openmvg_rs,
    EstimationError,
    OpenMVGError,
    "No geometric model could be estimated from the correspondences."
);

/// How long the waiting caller sleeps between two checks for Python signals.
const SIGNAL_POLL: Duration = Duration::from_millis(50);

/// errno values of a missing file and of a denied access; the same on every supported platform.
const ENOENT: i32 = 2;
const EACCES: i32 = 13;

#[derive(Debug, Clone, Copy)]
enum Engine {
    Incremental,
    Global,
}

/// Storage encoding for PLY point clouds.
#[pyclass(eq, eq_int, frozen, from_py_object, module = "openmvg_rs")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyFormat {
    /// Human-readable floating-point coordinates and RGB bytes.
    Ascii,
    /// Little-endian f32 coordinates followed by RGB bytes.
    Binary,
}

impl Engine {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "incremental" => Ok(Self::Incremental),
            "global" => Ok(Self::Global),
            _ => Err(PyValueError::new_err(format!(
                "engine must be 'incremental' or 'global', not {name:?}"
            ))),
        }
    }
}

/// Forwards progress to the optional console bars and reports the caller's cancellation.
struct Progress<'a> {
    console: Option<ConsoleProgress>,
    canceled: &'a AtomicBool,
}

impl SfmProgressSink for Progress<'_> {
    fn begin(&self, stage: SfmProgressStage, total: usize) {
        if let Some(console) = &self.console {
            console.begin(stage, total);
        }
    }

    fn advance(&self, stage: SfmProgressStage, completed: usize) {
        if let Some(console) = &self.console {
            console.advance(stage, completed);
        }
    }

    fn finish(&self, stage: SfmProgressStage) {
        if let Some(console) = &self.console {
            console.finish(stage);
        }
    }

    fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::Relaxed)
    }
}

/// Why a reconstruction produced no result; errors are mapped by phase, not by variant.
enum Failure {
    Load(SfmError),
    Run(SfmError),
    GlobalEmpty,
    ThreadPool(rayon::ThreadPoolBuildError),
}

impl From<Failure> for PyErr {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Load(error) => load_error(error),
            Failure::Run(error) => run_error(error),
            Failure::GlobalEmpty => NoReconstructionError::new_err(
                "the Global engine found no consistent set of views to reconstruct; try engine='incremental'",
            ),
            Failure::ThreadPool(error) => {
                PyOSError::new_err(format!("cannot start the worker threads: {error}"))
            }
        }
    }
}

fn load_error(error: SfmError) -> PyErr {
    match &error {
        SfmError::DatasetNotFound { path } => missing_file(path),
        _ => read_error(&error).unwrap_or_else(|| DatasetError::new_err(error.to_string())),
    }
}

/// The error Python's own I/O raises for an input file that is missing or refused, if `error`
/// is one; other read failures stay library errors.
fn read_error(error: &SfmError) -> Option<PyErr> {
    let (SfmError::ReadFile { path, source } | SfmError::ImageRead { path, source }) = error else {
        return None;
    };
    match source.kind() {
        ErrorKind::NotFound => Some(missing_file(path)),
        ErrorKind::PermissionDenied => Some(permission_denied(path)),
        _ => None,
    }
}

/// `FileNotFoundError` with the errno, message, and `filename` that Python's own I/O sets.
fn missing_file(path: &Path) -> PyErr {
    PyFileNotFoundError::new_err((
        ENOENT,
        "No such file or directory",
        path.as_os_str().to_owned(),
    ))
}

/// `PermissionError` with the errno, message, and `filename` that Python's own I/O sets.
fn permission_denied(path: &Path) -> PyErr {
    PyPermissionError::new_err((EACCES, "Permission denied", path.as_os_str().to_owned()))
}

fn run_error(error: SfmError) -> PyErr {
    if let Some(read) = read_error(&error) {
        return read;
    }
    match error {
        SfmError::NoInitialPairCandidate => NoReconstructionError::new_err(error.to_string()),
        // Only the caller cancels, and it re-raises the signal it caught instead.
        SfmError::Canceled => PyKeyboardInterrupt::new_err(error.to_string()),
        _ => ReconstructionError::new_err(error.to_string()),
    }
}

/// A COLMAP model that cannot be written; the library's error does not name the file, so the
/// error names the requested `target`.
fn colmap_error(error: &openmvg_io::Error, target: &Path) -> PyErr {
    match error {
        openmvg_io::Error::Io(source) if source.kind() == ErrorKind::PermissionDenied => {
            permission_denied(target)
        }
        openmvg_io::Error::Io(source) if source.kind() == ErrorKind::NotFound => {
            missing_file(target)
        }
        _ => ExportError::new_err(format!("cannot write {}: {error}", target.display())),
    }
}

/// A reconstruction or PLY file that cannot be written; the library's error does not name the
/// file.
fn save_error(error: &SfmError, target: &Path) -> PyErr {
    match error {
        SfmError::Io(source) if source.kind() == ErrorKind::PermissionDenied => {
            permission_denied(target)
        }
        SfmError::Io(source) if source.kind() == ErrorKind::NotFound => missing_file(target),
        _ => ExportError::new_err(format!("cannot write {}: {error}", target.display())),
    }
}

/// A reconstruction file that cannot be read: missing or refused like Python's own I/O, any
/// other failure (wrong product, version, checksum, or contents) as `OpenMVGError`.
fn load_reconstruction_error(error: &SfmError, source_path: &Path) -> PyErr {
    match error {
        SfmError::Io(source) if source.kind() == ErrorKind::NotFound => missing_file(source_path),
        SfmError::Io(source) if source.kind() == ErrorKind::PermissionDenied => {
            permission_denied(source_path)
        }
        _ => OpenMVGError::new_err(format!("cannot load {}: {error}", source_path.display())),
    }
}

/// Run `work` on a scoped worker thread while the caller waits with the GIL released.
///
/// Between waits the caller runs the Python signal handlers. The first exception they raise
/// (`KeyboardInterrupt` for Ctrl-C) sets `canceled`, so the run stops at its next cancellation
/// point, and is re-raised once the worker has returned. A worker panic resumes on the caller,
/// where PyO3 turns it into `PanicException`.
fn run_interruptibly<T: Send>(
    py: Python<'_>,
    canceled: &AtomicBool,
    work: impl FnOnce() -> T + Send,
) -> PyResult<T> {
    std::thread::scope(|scope| {
        let (done, mut finished) = mpsc::channel::<()>();
        // The worker never sends: its sender drops when `work` returns or unwinds.
        let worker = scope.spawn(move || {
            let _done = done;
            work()
        });
        let mut interrupt = None;
        loop {
            let (state, receiver) = py.detach(move || {
                let state = finished.recv_timeout(SIGNAL_POLL);
                (state, finished)
            });
            finished = receiver;
            if !matches!(state, Err(RecvTimeoutError::Timeout)) {
                break;
            }
            if interrupt.is_none()
                && let Err(error) = py.check_signals()
            {
                canceled.store(true, Ordering::Relaxed);
                interrupt = Some(error);
            }
        }
        let value = worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        interrupt.map_or(Ok(value), Err)
    })
}

/// The reconstruction behind `openmvg_rs.reconstruct`; see its Python docstring.
///
/// `max_workers` of `None` runs on the global Rayon pool; `Some(n)` runs on a private pool of
/// `n` threads, where 0 lets Rayon size it as it would the global pool. The Python layer passes 0
/// in a forked child, which inherits a global pool whose threads did not survive the fork.
// One parameter per keyword argument of the Python `reconstruct`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (image_dir, engine, seed, intrinsics, progress, max_workers, extraction_method = "alp"))]
fn reconstruct(
    py: Python<'_>,
    image_dir: PathBuf,
    engine: &str,
    seed: u64,
    intrinsics: Option<PathBuf>,
    progress: bool,
    max_workers: Option<usize>,
    extraction_method: &str,
) -> PyResult<Reconstruction> {
    let engine = Engine::parse(engine)?;
    let extraction_method = match extraction_method {
        "sift" => ExtractionMethod::Sift,
        "alp" => ExtractionMethod::Alp,
        _ => {
            return Err(PyValueError::new_err(
                "extraction_method must be 'sift' or 'alp'",
            ));
        }
    };
    let canceled = AtomicBool::new(false);
    let sink = Progress {
        console: progress.then(ConsoleProgress::stderr),
        canceled: &canceled,
    };

    let work = || -> Result<openmvg_sfm::Reconstruction, Failure> {
        let mut options = DatasetConfig::default().with_seed(seed.into());
        if let Some(path) = &intrinsics {
            options = options.with_intrinsics(path);
        }
        let dataset = Dataset::from_dir_with(&image_dir, &options).map_err(Failure::Load)?;
        let (graph, _) = Frontend::default()
            .with_extraction_method(extraction_method)
            .build_with_report(&dataset, &sink)
            .map_err(Failure::Run)?;
        drop(dataset);
        let output = match engine {
            Engine::Incremental => Incremental::default()
                .runner()
                .progress(&sink)
                .run_prepared_owned(graph),
            Engine::Global => Global::default()
                .runner()
                .progress(&sink)
                .run_prepared_owned(graph),
        }
        .map_err(Failure::Run)?;
        let (reconstruction, report) = output.into_parts();
        if !report.has_reconstruction() {
            return Err(Failure::GlobalEmpty);
        }
        Ok(reconstruction)
    };
    // Every automatic thread count in the library derives from the current Rayon pool, so a
    // local pool bounds the whole run.
    let bounded = || match max_workers {
        None => work(),
        Some(workers) => rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(Failure::ThreadPool)?
            .install(work),
    };

    let inner = run_interruptibly(py, &canceled, bounded)??;
    Reconstruction::new(py, inner)
}

/// The reconstruction behind `openmvg_rs.load_reconstruction`; see its Python docstring.
#[pyfunction]
#[pyo3(signature = (path, /))]
fn load_reconstruction(py: Python<'_>, path: PathBuf) -> PyResult<Reconstruction> {
    let inner = py
        .detach(|| checkpoint::read_reconstruction(&path))
        .map_err(|error| load_reconstruction_error(&error, &path))?;
    Reconstruction::new(py, inner)
}

/// Unpickle a reconstruction from the bytes of its file.
#[pyfunction]
#[pyo3(signature = (data, /))]
fn _restore_reconstruction(py: Python<'_>, data: &[u8]) -> PyResult<Reconstruction> {
    let inner = py
        .detach(|| openmvg_sfm::Reconstruction::from_bytes(data))
        .map_err(|error| OpenMVGError::new_err(format!("cannot unpickle: {error}")))?;
    Reconstruction::new(py, inner)
}

/// An immutable sparse reconstruction returned by `reconstruct`.
///
/// It holds the poses of the registered views, the camera models, and the 3D points; read them
/// as `views` and `points`, write them out with `write_colmap` or `write_ply`, or save them with
/// `save`.
#[pyclass(frozen, module = "openmvg_rs", name = "Reconstruction")]
struct Reconstruction {
    inner: openmvg_sfm::Reconstruction,
    seed: Py<PyInt>,
    registered_images: Py<PyTuple>,
    unregistered_images: Py<PyTuple>,
    points: Py<PyArray2<f64>>,
    views: Py<PyTuple>,
}

impl Reconstruction {
    fn new(py: Python<'_>, inner: openmvg_sfm::Reconstruction) -> PyResult<Self> {
        let path_type = py.import("pathlib")?.getattr("Path")?;
        let paths = |images: &mut dyn Iterator<Item = &openmvg_sfm::model::ImageMeta>| {
            let paths = images
                .filter_map(|image| image.file_path())
                .map(|path| path_type.call1((path,)))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, paths).map(Bound::unbind)
        };
        let registered_images = paths(&mut inner.registered_images())?;
        let unregistered_images = paths(&mut inner.unregistered_images())?;
        Ok(Self {
            seed: seed_int(py, inner.root_seed())?,
            registered_images,
            unregistered_images,
            points: points_array(py, &inner)?.unbind(),
            views: views_tuple(py, &inner)?.unbind(),
            inner,
        })
    }
}

/// The active landmarks, the points the exports write, as a read-only `(n, 3)` array in
/// landmark order.
fn points_array<'py>(
    py: Python<'py>,
    inner: &openmvg_sfm::Reconstruction,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut flat = Vec::with_capacity(3 * inner.point_count());
    for landmark in inner.landmarks().iter_active() {
        flat.extend_from_slice(landmark.point.coords.as_slice());
    }
    let rows = flat.len() / 3;
    let points = Array2::from_shape_vec((rows, 3), flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    points
        .as_any()
        .getattr("flags")?
        .setattr("writeable", false)?;
    Ok(points)
}

/// The model name and named parameters of `camera`, with the names of the OMVG format's
/// `cameras` table; a coefficient the camera does not model is absent.
fn camera_parameters(camera: &Camera) -> (&'static str, Vec<(&'static str, f64)>) {
    let pinhole = |camera: &PinholeCamera| {
        vec![
            ("fx", camera.focal_x()),
            ("fy", camera.focal_y()),
            ("cx", camera.principal_x()),
            ("cy", camera.principal_y()),
        ]
    };
    match camera {
        Camera::Pinhole(camera) => ("PINHOLE", pinhole(camera)),
        Camera::BrownConrady(camera) => {
            let mut parameters = pinhole(camera.pinhole());
            let radial: &[f64] = match camera.radial_distortion() {
                RadialDistortion::None => &[],
                RadialDistortion::K1(k1) => &[*k1],
                RadialDistortion::K1K2(k1, k2) => &[*k1, *k2],
                RadialDistortion::K1K2K3(k1, k2, k3) => &[*k1, *k2, *k3],
            };
            let tangential: &[f64] = match camera.tangential_distortion() {
                TangentialDistortion::None => &[],
                TangentialDistortion::P1(p1) => &[*p1],
                TangentialDistortion::P1P2(p1, p2) => &[*p1, *p2],
            };
            parameters.extend(["k1", "k2", "k3"].into_iter().zip(radial.iter().copied()));
            parameters.extend(["p1", "p2"].into_iter().zip(tangential.iter().copied()));
            ("BROWN_CONRADY", parameters)
        }
        Camera::Fisheye(camera) => {
            let mut parameters = pinhole(camera.pinhole());
            let radial: &[f64] = match camera.fisheye_distortion() {
                FisheyeDistortion::None => &[],
                FisheyeDistortion::K1(k1) => &[*k1],
                FisheyeDistortion::K1K2(k1, k2) => &[*k1, *k2],
                FisheyeDistortion::K1K2K3(k1, k2, k3) => &[*k1, *k2, *k3],
                FisheyeDistortion::K1K2K3K4(k1, k2, k3, k4) => &[*k1, *k2, *k3, *k4],
            };
            parameters.extend(
                ["k1", "k2", "k3", "k4"]
                    .into_iter()
                    .zip(radial.iter().copied()),
            );
            ("FISHEYE", parameters)
        }
        Camera::Spherical(_) => ("SPHERICAL", Vec::new()),
    }
}

/// The registered views with an image file, in view order (the order of `registered_images`),
/// as `View` values; views that share a camera share one `Camera` object.
fn views_tuple<'py>(
    py: Python<'py>,
    inner: &openmvg_sfm::Reconstruction,
) -> PyResult<Bound<'py, PyTuple>> {
    let pose_type = py.import("openmvg_rs._geometry")?.getattr("Pose")?;
    let scene = py.import("openmvg_rs._scene")?;
    let (camera_type, view_type) = (scene.getattr("Camera")?, scene.getattr("View")?);
    let path_type = py.import("pathlib")?.getattr("Path")?;
    // Views of one camera model share one Camera; validation gives them one image size.
    let mut cameras = std::collections::BTreeMap::new();
    let mut views = Vec::new();
    for view in inner.views() {
        let (Some(record), Some(path)) = (inner.pose(view.id), view.meta.file_path()) else {
            continue;
        };
        let index = view.intrinsic_id.index();
        let camera = match cameras.get(&index) {
            Some(camera) => Bound::clone(camera),
            None => {
                let model = inner.camera_models().get(index).ok_or_else(|| {
                    PyValueError::new_err(format!("view {} has no camera model", view.id))
                })?;
                let (name, parameters) = camera_parameters(model);
                let camera =
                    camera_type.call1((name, view.meta.width, view.meta.height, parameters))?;
                cameras.insert(index, camera.clone());
                camera
            }
        };
        let rotation = record.pose.rotation_matrix();
        let rotation = Array2::from_shape_fn((3, 3), |(row, column)| rotation[(row, column)]);
        let translation = PyArray1::from_slice(py, record.pose.translation().as_slice());
        let pose = pose_type.call1((rotation.into_pyarray(py), translation))?;
        views.push(view_type.call1((path_type.call1((path,))?, pose, camera))?);
    }
    PyTuple::new(py, views)
}

/// The root seed as a Python integer: its 32 bytes read little-endian, so a seed given as an
/// integer below 2**64 comes back unchanged.
fn seed_int(py: Python<'_>, seed: openmvg_sfm::RunSeed) -> PyResult<Py<PyInt>> {
    // The seed's display is its bytes in hexadecimal, in order.
    let hex = seed.to_string();
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(py
        .get_type::<PyInt>()
        .call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))?
        .cast_into::<PyInt>()?
        .unbind())
}

#[pymethods]
impl Reconstruction {
    /// The reconstruction engine: ``"incremental"`` or ``"global"``.
    #[getter]
    fn engine(&self) -> &'static str {
        self.inner.engine().as_str()
    }

    /// The seed the run was started with.
    #[getter]
    fn seed(&self, py: Python<'_>) -> Py<PyInt> {
        self.seed.clone_ref(py)
    }

    /// ``"fixed"`` when a calibration file supplied the camera intrinsics, which then stay
    /// unchanged; ``"estimated"`` when they were initialized from EXIF metadata and refined.
    #[getter]
    fn intrinsics_mode(&self) -> &'static str {
        if self.inner.has_calibrated_intrinsics() {
            "fixed"
        } else {
            "estimated"
        }
    }

    /// Number of views, one per image in the input folder.
    #[getter]
    fn num_images(&self) -> usize {
        self.inner.view_count()
    }

    /// Number of registered views: input images with a reconstructed pose.
    #[getter]
    fn num_registered_images(&self) -> usize {
        self.inner.registered_view_count()
    }

    /// Number of reconstructed 3D points.
    #[getter]
    fn num_points(&self) -> usize {
        self.inner.point_count()
    }

    /// The reconstructed 3D points in world coordinates, a read-only `(n, 3)` array.
    #[getter]
    fn points(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.points.clone_ref(py)
    }

    /// The registered views as `View` values, in the order of `registered_images`.
    #[getter]
    fn views(&self, py: Python<'_>) -> Py<PyTuple> {
        self.views.clone_ref(py)
    }

    /// Image paths of the registered views, in input order.
    #[getter]
    fn registered_images(&self, py: Python<'_>) -> Py<PyTuple> {
        self.registered_images.clone_ref(py)
    }

    /// Image paths of the views that were not registered, in input order.
    #[getter]
    fn unregistered_images(&self, py: Python<'_>) -> Py<PyTuple> {
        self.unregistered_images.clone_ref(py)
    }

    /// Write a COLMAP text model to ``directory/sparse/0/``.
    ///
    /// Missing directories are created. ``cameras.txt``, ``images.txt`` and ``points3D.txt`` are
    /// replaced; other files are left alone.
    #[pyo3(signature = (directory, /))]
    fn write_colmap(&self, py: Python<'_>, directory: PathBuf) -> PyResult<()> {
        py.detach(|| openmvg_io::write_colmap(&self.inner, &directory))
            .map_err(|error| colmap_error(&error, &directory))
    }

    /// Write white 3D points and green registered camera centers as a PLY point cloud.
    ///
    /// Binary little-endian is the default; select `PlyFormat.Ascii` for text output.
    /// Missing parent directories are created and an existing file is replaced.
    #[pyo3(signature = (path, /, *, format = PlyFormat::Binary))]
    fn write_ply(&self, py: Python<'_>, path: PathBuf, format: PlyFormat) -> PyResult<()> {
        let format = match format {
            PlyFormat::Ascii => openmvg_sfm::PlyFormat::Ascii,
            PlyFormat::Binary => openmvg_sfm::PlyFormat::Binary,
        };
        py.detach(|| self.inner.write_ply_with_format(&path, format))
            .map_err(|error| save_error(&error, &path))
    }

    /// Save the reconstruction to a binary file that `load_reconstruction` reads back.
    ///
    /// The parent directory must exist; an existing file is replaced atomically.
    /// Only OpenMVG-rs reads the file back, with a release that writes the same schema version;
    /// it is not an exchange format, so share results with `write_colmap` or `write_ply`.
    /// The conventional extension is `.omvgrs`.
    #[pyo3(signature = (path, /))]
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| checkpoint::write_reconstruction(&self.inner, &path))
            .map(drop)
            .map_err(|error| save_error(&error, &path))
    }

    fn __repr__(&self) -> String {
        format!(
            "<openmvg_rs.Reconstruction engine='{}' registered={}/{} points={}>",
            self.inner.engine().as_str(),
            self.inner.registered_view_count(),
            self.inner.view_count(),
            self.inner.point_count()
        )
    }

    /// Pickle as the bytes `save` writes.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let data = self
            .inner
            .to_bytes()
            .map_err(|error| OpenMVGError::new_err(format!("cannot pickle: {error}")))?;
        let restore = py
            .import("openmvg_rs._native")?
            .getattr("_restore_reconstruction")?;
        let arguments = (PyBytes::new(py, &data),).into_pyobject(py)?;
        Ok((restore, arguments))
    }

    /// A reconstruction is immutable, so a copy is the object itself.
    const fn __copy__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// A reconstruction is immutable, so a copy is the object itself.
    #[pyo3(signature = (_memo, /))]
    fn __deepcopy__<'py>(slf: Bound<'py, Self>, _memo: Bound<'py, PyAny>) -> Bound<'py, Self> {
        slf
    }
}

#[pymodule(name = "_native", gil_used = false)]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(reconstruct, module)?)?;
    module.add_function(wrap_pyfunction!(load_reconstruction, module)?)?;
    module.add_function(wrap_pyfunction!(_restore_reconstruction, module)?)?;
    module.add_function(wrap_pyfunction!(geometry::fundamental, module)?)?;
    module.add_function(wrap_pyfunction!(geometry::homography, module)?)?;
    module.add_function(wrap_pyfunction!(geometry::essential, module)?)?;
    module.add_function(wrap_pyfunction!(geometry::absolute_pose, module)?)?;
    module.add_class::<PlyFormat>()?;
    module.add_class::<Reconstruction>()?;
    module.add("OpenMVGError", py.get_type::<OpenMVGError>())?;
    module.add("DatasetError", py.get_type::<DatasetError>())?;
    module.add("ReconstructionError", py.get_type::<ReconstructionError>())?;
    module.add(
        "NoReconstructionError",
        py.get_type::<NoReconstructionError>(),
    )?;
    module.add("ExportError", py.get_type::<ExportError>())?;
    module.add("EstimationError", py.get_type::<EstimationError>())?;
    Ok(())
}
