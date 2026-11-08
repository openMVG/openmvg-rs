// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional wgpu backend for the classic SIFT extractor.
//!
//! ```text
//!  image ──pack u8──> init ──> blur (+ DoG in its second pass) ──> extrema ──> refine
//!                                                                              │
//!  descriptor (+ u8 quantize) <── compact <── peak offsets <── orientation <── order + dedup
//!   │                                       one submit
//!   └──> host: counts, then exactly the oriented features and descriptors
//! ```
//!
//! The GPU path follows the CPU detector stage for stage: the hierarchical
//! Gaussian scale space, DoG extrema, five-step Taylor refinement, orientation
//! histograms, feature budget and the classic 4x4x8 descriptor. It is intended
//! for callers that already decode an image into a [`GrayImage`]; it does not
//! read image files and does not create a window or surface.
//!
//! The backend is available with the `gpu` feature. It preserves the CPU
//! extractor's deterministic ordering: refined samples are deduplicated in
//! candidate `(scale_index, row, column)` order, orientations stay in bin order
//! and descriptors stay in detector order. Floating-point kernels run in f32, so a descriptor or a
//! borderline keypoint can differ by a small amount from the CPU path; compare
//! repeatability and descriptor distance rather than expecting bit equality.
//!
//! One extractor is meant to be reused for a batch of images. Its scratch
//! buffers stay resident and grow only when a later image needs more space,
//! so repeated and alternating image sizes avoid per-image GPU allocation.
//! Without a feature budget, an extraction is one submit followed by two
//! readbacks: the counters, which also wait for the work, and then exactly the
//! oriented features and descriptors that exist. Device-to-host transfers are
//! slow, so copying exact sizes beats a single readback sized for the worst
//! case. The device counts are not known when the submit is recorded, so
//! kernels read them on the GPU and buffers are sized by per-octave bounds
//! learned from earlier images; an image that exceeds them grows them and runs
//! again. A feature budget selects on the host, so a budgeted extraction
//! describes in a second submit. The host never sorts or deduplicates;
//! `order.wgsl` does that on the device, and `oriented.wgsl` turns the
//! orientation peaks into dense descriptor keys. An image of exact
//! 8-bit values is uploaded as bytes, a quarter of its f32 size, and the u8
//! path quantizes on the device and reads back packed bytes, so it transfers
//! four times fewer descriptor bytes than the f32 path. Parameters are written
//! with one queue write per submit, and the per-octave stages after refinement are recorded stage by stage, one
//! command buffer each, so the octaves of a stage run concurrently.
//!
//! [`GrayImage`]: crate::extraction::GrayImage

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::extraction::feature_budget::{
    FeatureBudget, FeatureBudgetKeypoint, apply_feature_budget,
};
use crate::extraction::{
    DescriptorStorage, F32Descriptor, GrayImage, ScaleSpaceFeatures, ScaleSpaceKeypoint,
    U8Descriptor,
};

use super::common::SIFT_DESCR_SIZE;
use super::detector::SiftExtractor;
use super::{SiftError, SiftPreset};

const BASE_SIGMA: f32 = 1.6;
const ASSUMED_INPUT_SIGMA: f32 = 0.5;
const MIN_INPUT_DIMENSION: usize = 8;
/// The CPU candidate capacity heuristic. A noisy image can exceed it; the
/// caller then receives [`SiftGpuError::Capacity`] and can use the CPU path.
const CANDIDATE_DIVISOR: usize = 32;
const MIN_CANDIDATES: usize = 256;
const MAX_ORIENTATIONS: usize = 36;
const ORIENTATION_PREFIX: usize = 8;
const CANDIDATE_BYTES: usize = 24;
/// One `Refined` record of `refine.wgsl` and `order.wgsl`.
const REFINED_BYTES: usize = core::mem::size_of::<GpuRefined>();
const REFINE_WORKGROUPS: u32 = 256;
/// `NO_COPY` and `SCAN_LIVE` in `order.wgsl`.
const ORDER_NO_COPY: u32 = u32::MAX;
const ORDER_SCAN_LIVE: u32 = u32::MAX;
const ORIENTED_RECORD_BYTES: usize = core::mem::size_of::<GpuOrientedRecord>();
/// Detection passes per image before a growing image is rejected: the first
/// may overflow the refined bound, the second the oriented bound.
const MAX_PASS_ATTEMPTS: usize = 4;
const ORDERED_FEATURE_BYTES: usize = core::mem::size_of::<GpuFeature>();
const PACKED_DESCR_WORDS: usize = SIFT_DESCR_SIZE / 4;

// =============================================================================
// Public API
// =============================================================================

/// Errors from the optional GPU SIFT backend.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SiftGpuError {
    /// The detector configuration is invalid.
    #[error(transparent)]
    InvalidConfig(#[from] SiftError),
    /// No compatible GPU adapter was found.
    #[error("no compatible GPU adapter was found")]
    NoAdapter,
    /// The GPU device could not be created.
    #[error("failed to create the GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    /// A GPU buffer or the device storage limit was exceeded.
    #[error(
        "the GPU SIFT {what} stage exceeded a buffer or device storage limit; use the CPU extractor for this image"
    )]
    Capacity {
        /// Short description of the overflowing stage.
        what: &'static str,
    },
    /// Reading a result buffer back to the host failed.
    #[error("GPU SIFT readback failed: {0}")]
    Readback(String),
    /// The device ran out of memory for this image. The extractor rebuilds
    /// its scratch buffers and stays usable; a smaller image or the CPU
    /// extractor may succeed.
    #[error("the GPU ran out of memory; use the CPU extractor for this image")]
    OutOfMemory,
    /// The device rejected a command. This indicates a bug in the backend;
    /// the extractor rebuilds its scratch buffers and stays usable.
    #[error("GPU SIFT command failed validation: {0}")]
    Validation(String),
    /// The device was lost or reported an internal error. Every later call
    /// on this extractor returns this error; create a new extractor.
    #[error("the GPU device was lost: {0}; create a new extractor")]
    DeviceLost(String),
    /// The device did not finish within [`SiftGpuExtractor::TIMEOUT`]. Every
    /// later call on this extractor returns this error; create a new
    /// extractor.
    #[error("the GPU did not finish within {timeout:?}; create a new extractor")]
    Timeout {
        /// The wait that elapsed.
        timeout: std::time::Duration,
    },
}

/// GPU implementation of [`SiftExtractor`].
///
/// Create it once and reuse it for images of different sizes. The extractor
/// owns a wgpu device and reuses its compute pipelines and scratch buffers
/// across calls; buffers grow only when a later image needs more space.
/// Use [`Self::extract_u8`], [`Self::extract_f32`], [`Self::extract_typed`] or
/// [`Self::detect_keypoints`] exactly as with the CPU [`SiftExtractor`].
///
/// # Availability
///
/// Creating the extractor fails with [`SiftGpuError::NoAdapter`] when no
/// compatible wgpu adapter is available. Applications that want a fallback
/// should catch that error and use [`SiftExtractor`].
///
/// # Example
///
/// ```no_run
/// use openmvg_features::extraction::{GrayImage, SiftGpuExtractor, SiftPreset};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = GrayImage::zeros(128, 128);
/// let extractor = SiftGpuExtractor::with_preset(SiftPreset::Normal)?;
/// let features = extractor.extract_u8(&image)?;
/// assert_eq!(features.keypoints().len(), features.descriptors().len());
/// # Ok(())
/// # }
/// ```
pub struct SiftGpuExtractor {
    config: SiftExtractor,
    engine: Engine,
}

impl core::fmt::Debug for SiftGpuExtractor {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SiftGpuExtractor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SiftGpuExtractor {
    /// Longest wait for the device to finish one stage of an extraction
    /// before the call fails with [`SiftGpuError::Timeout`].
    pub const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

    /// Create a GPU extractor for `config`.
    ///
    /// # Errors
    ///
    /// Returns [`SiftGpuError::InvalidConfig`] if [`SiftExtractor::validate`]
    /// rejects `config`, [`SiftGpuError::NoAdapter`] /
    /// [`SiftGpuError::Device`] if no GPU device can be created, or a device
    /// error if the compute pipelines cannot be built.
    pub fn new(config: SiftExtractor) -> Result<Self, SiftGpuError> {
        config.validate()?;
        Ok(Self {
            config,
            engine: Engine::new()?,
        })
    }

    /// Create a GPU extractor from one of the CPU presets.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn with_preset(preset: SiftPreset) -> Result<Self, SiftGpuError> {
        Self::new(SiftExtractor::with_preset(preset))
    }

    /// Bytes of GPU memory the extractor keeps allocated between calls.
    ///
    /// Scratch buffers grow to the largest image processed so far and are
    /// released when the extractor is dropped. The device and driver add
    /// their own overhead on top.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.engine.lock_scratch().resident_bytes()
    }

    /// The detector configuration this extractor uses.
    #[must_use]
    pub fn config(&self) -> &SiftExtractor {
        &self.config
    }

    /// Detect oriented keypoint geometry without constructing descriptors.
    ///
    /// # Errors
    ///
    /// Returns [`SiftGpuError::Capacity`] or [`SiftGpuError::OutOfMemory`]
    /// when the image does not fit the device, and a device, timeout or
    /// readback error when the GPU fails. GPU failures never panic.
    pub fn detect_keypoints(
        &self,
        image: &GrayImage,
    ) -> Result<Vec<ScaleSpaceKeypoint>, SiftGpuError> {
        self.engine
            .detect_keypoints(&self.config, image)
            .map_err(Into::into)
    }

    /// Detect and describe with f32 descriptors.
    ///
    /// # Errors
    ///
    /// Returns [`SiftGpuError::Capacity`] or [`SiftGpuError::OutOfMemory`]
    /// when the image does not fit the device, and a device, timeout or
    /// readback error when the GPU fails. GPU failures never panic.
    pub fn extract_f32(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<F32Descriptor>, SiftGpuError> {
        self.extract_typed(image)
    }

    /// Detect and describe with u8 descriptors.
    ///
    /// Quantization runs on the GPU and only the packed bytes are read back.
    ///
    /// # Errors
    ///
    /// Returns [`SiftGpuError::Capacity`] or [`SiftGpuError::OutOfMemory`]
    /// when the image does not fit the device, and a device, timeout or
    /// readback error when the GPU fails. GPU failures never panic.
    pub fn extract_u8(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<U8Descriptor>, SiftGpuError> {
        self.engine
            .extract_u8(&self.config, image)
            .map_err(Into::into)
    }

    /// Detect and describe with the descriptor storage `D`.
    ///
    /// # Errors
    ///
    /// Returns [`SiftGpuError::Capacity`] or [`SiftGpuError::OutOfMemory`]
    /// when the image does not fit the device, and a device, timeout or
    /// readback error when the GPU fails. GPU failures never panic.
    pub fn extract_typed<D: DescriptorStorage>(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<D>, SiftGpuError> {
        self.engine
            .extract::<D>(&self.config, image)
            .map_err(Into::into)
    }
}

// =============================================================================
// Detector state
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct Feature {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    peak: f32,
    octave: usize,
    scale_index: usize,
}

impl Feature {
    fn geometry(self) -> ScaleSpaceKeypoint {
        ScaleSpaceKeypoint {
            x: self.x,
            y: self.y,
            sigma: self.sigma,
            orientation: self.orientation,
            peak: self.peak,
            octave: self.octave,
            scale_index: self.scale_index,
            is_max: self.peak > 0.0,
        }
    }
}

impl FeatureBudgetKeypoint for Feature {
    fn feature_candidate(&self) -> crate::extraction::FeatureCandidate {
        crate::extraction::FeatureCandidate::new(self.x, self.y, self.peak.abs())
    }
}

struct Octave {
    rows: usize,
    cols: usize,
    delta: f32,
    gauss: wgpu::Buffer,
}

struct Detection {
    features: Vec<Feature>,
    octaves: Vec<Octave>,
    compact_keys: Option<CompactKeys>,
}

/// Descriptor keys left on the GPU by the detection pass, per octave, for
/// describing a budgeted detection whose feature set did not change.
struct CompactKeys {
    totals: Vec<usize>,
    feature_starts: Vec<usize>,
    keys: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
}

/// The descriptor format and decoder for a pass that also describes.
type Describe<D> = Option<(DescriptorFormat, fn(&[u8]) -> Option<D>)>;

/// One detection pass: oriented features in detector order, the GPU buffers
/// that still hold them, and the descriptors when the pass computed them.
struct Pass<D> {
    features: Vec<Feature>,
    octaves: Vec<Octave>,
    /// Per octave: the oriented feature count, the descriptor keys and the
    /// header buffer whose first word holds that count on the GPU.
    compact: Vec<(usize, wgpu::Buffer, wgpu::Buffer)>,
    described: Option<ScaleSpaceFeatures<D>>,
}

/// GPU buffers of one octave after [`Engine::encode_order`]: `work[0]` holds
/// the number of kept features, which is at most `bound`.
struct OrderedOctave {
    bound: usize,
    work: wgpu::Buffer,
    features: wgpu::Buffer,
    counts: wgpu::Buffer,
    orientations: wgpu::Buffer,
}

/// Buffers of one octave after [`Engine::encode_oriented`].
struct OrientedOctave {
    keys: wgpu::Buffer,
    records: wgpu::Buffer,
    header: wgpu::Buffer,
}

/// Per-octave sizes for the single-submit pass. The device counts are only
/// known after the pass, so buffers are sized from earlier images: `refined`
/// bounds the refined samples and `oriented` the oriented features, each the
/// largest count seen plus a quarter. A pass that exceeds a bound grows it and
/// runs again.
#[derive(Clone, Copy, Default)]
struct OctaveBounds {
    refined: usize,
    oriented: usize,
}

impl OctaveBounds {
    fn get(&mut self, capacity: usize) -> (usize, usize) {
        if self.refined == 0 {
            // A first guess well above typical images; it adapts afterwards.
            self.refined = (capacity / 64).max(1024).min(capacity);
            self.oriented = self.refined * 3 / 2;
        }
        (self.refined.min(capacity), self.oriented)
    }

    /// Record one pass; return whether its output was truncated.
    fn update(&mut self, capacity: usize, refined: usize, oriented: usize) -> bool {
        let grow = |count: usize| count + count / 4 + 256;
        let (refined_bound, oriented_bound) = self.get(capacity);
        if refined > refined_bound {
            self.refined = grow(refined).min(capacity);
            self.oriented = self.oriented.max(self.refined * 3 / 2);
            return true;
        }
        if oriented > oriented_bound {
            self.oriented = grow(oriented);
            return true;
        }
        // A pass that fits keeps the bounds: they are a high-water mark with
        // headroom, so later images rarely run twice. They size GPU buffers
        // only; readbacks copy the exact counts.
        false
    }
}

struct DescriptorReadback {
    key_count: usize,
    valid_bytes: usize,
    value_bytes: usize,
    valid: wgpu::Buffer,
    values: wgpu::Buffer,
    feature_indices: Vec<usize>,
}

#[derive(Clone, Copy)]
enum DescriptorFormat {
    F32,
    U8,
}

impl DescriptorFormat {
    fn quantize_u8(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::U8 => 1,
        }
    }

    fn words_per_key(self) -> usize {
        match self {
            Self::F32 => SIFT_DESCR_SIZE,
            Self::U8 => PACKED_DESCR_WORDS,
        }
    }

    fn value_bytes(self, key_count: usize) -> Option<usize> {
        key_count
            .checked_mul(self.words_per_key())
            .and_then(|words| words.checked_mul(core::mem::size_of::<u32>()))
    }
}

// =============================================================================
// Persistent scratch
// =============================================================================

/// A storage or staging buffer that is kept between extractions and replaced
/// only when a later image needs more bytes. Cloning the returned handle is
/// cheap; it does not allocate device memory.
#[derive(Default)]
struct GrowBuffer {
    buffer: Option<wgpu::Buffer>,
    #[cfg(test)]
    allocations: usize,
}

impl GrowBuffer {
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        bytes: usize,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let needed = bytes.max(16).div_ceil(4) as u64 * 4;
        let grow = self
            .buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < needed);
        if grow {
            self.buffer = Some(create_buffer(device, label, needed, usage));
            #[cfg(test)]
            {
                self.allocations += 1;
            }
        }
        match &self.buffer {
            Some(buffer) => buffer.clone(),
            None => create_buffer(device, label, needed, usage),
        }
    }

    fn bytes(&self) -> u64 {
        self.buffer.as_ref().map_or(0, wgpu::Buffer::size)
    }

    fn ensure_storage(&mut self, device: &wgpu::Device, label: &str, bytes: usize) -> wgpu::Buffer {
        self.ensure(
            device,
            label,
            bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    fn ensure_readback(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        bytes: usize,
    ) -> wgpu::Buffer {
        self.ensure(
            device,
            label,
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }
}

#[derive(Default)]
struct OctaveScratch {
    gauss: GrowBuffer,
    dog: GrowBuffer,
    temp_a: GrowBuffer,
    temp_b: GrowBuffer,
    candidates: GrowBuffer,
    candidate_count: Option<wgpu::Buffer>,
    refined: GrowBuffer,
    refined_count: Option<wgpu::Buffer>,
}

#[derive(Default)]
struct Scratch {
    source: GrowBuffer,
    packed_input: Vec<u32>,
    octaves: Vec<OctaveScratch>,
    order_work: Vec<GrowBuffer>,
    ordered_features: Vec<GrowBuffer>,
    orientation_keys: Vec<GrowBuffer>,
    orientation_counts: Vec<GrowBuffer>,
    orientations: Vec<GrowBuffer>,
    orientation_offsets: Vec<GrowBuffer>,
    orientation_prefix: Vec<GrowBuffer>,
    descriptor_keys: Vec<GrowBuffer>,
    descriptors: Vec<GrowBuffer>,
    descriptor_valid: Vec<GrowBuffer>,
    oriented_records: Vec<GrowBuffer>,
    oriented_headers: Vec<GrowBuffer>,
    bounds: Vec<OctaveBounds>,
    staging: GrowBuffer,
    blur_weights: BTreeMap<u32, (wgpu::Buffer, usize)>,
    params: ParamArena,
}

impl Scratch {
    fn ensure_stage_buffers(&mut self, stages: usize) {
        if self.octaves.len() < stages {
            self.octaves.resize_with(stages, OctaveScratch::default);
        }
        if self.orientation_keys.len() < stages {
            self.order_work.resize_with(stages, GrowBuffer::default);
            self.ordered_features
                .resize_with(stages, GrowBuffer::default);
            self.orientation_keys
                .resize_with(stages, GrowBuffer::default);
            self.orientation_counts
                .resize_with(stages, GrowBuffer::default);
            self.orientations.resize_with(stages, GrowBuffer::default);
            self.orientation_offsets
                .resize_with(stages, GrowBuffer::default);
            self.orientation_prefix
                .resize_with(stages, GrowBuffer::default);
            self.descriptor_keys
                .resize_with(stages, GrowBuffer::default);
            self.descriptors.resize_with(stages, GrowBuffer::default);
            self.descriptor_valid
                .resize_with(stages, GrowBuffer::default);
            self.oriented_records
                .resize_with(stages, GrowBuffer::default);
            self.oriented_headers
                .resize_with(stages, GrowBuffer::default);
            self.bounds.resize_with(stages, OctaveBounds::default);
        }
    }

    fn counter(
        slot: &mut Option<wgpu::Buffer>,
        device: &wgpu::Device,
        label: &str,
    ) -> wgpu::Buffer {
        match slot {
            Some(buffer) => buffer.clone(),
            None => {
                let buffer = storage_buffer(device, label, 16);
                *slot = Some(buffer.clone());
                buffer
            }
        }
    }

    /// Bytes of every GPU buffer kept between extractions.
    fn resident_bytes(&self) -> u64 {
        let octaves = self.octaves.iter().map(|octave| {
            [
                &octave.gauss,
                &octave.dog,
                &octave.temp_a,
                &octave.temp_b,
                &octave.candidates,
                &octave.refined,
            ]
            .iter()
            .map(|buffer| buffer.bytes())
            .sum::<u64>()
                + [&octave.candidate_count, &octave.refined_count]
                    .iter()
                    .filter_map(|counter| counter.as_ref())
                    .map(wgpu::Buffer::size)
                    .sum::<u64>()
        });
        let stages = [
            &self.order_work,
            &self.ordered_features,
            &self.orientation_keys,
            &self.orientation_counts,
            &self.orientations,
            &self.orientation_offsets,
            &self.orientation_prefix,
            &self.descriptor_keys,
            &self.descriptors,
            &self.descriptor_valid,
            &self.oriented_records,
            &self.oriented_headers,
        ]
        .into_iter()
        .flatten()
        .map(GrowBuffer::bytes);
        let weights = self.blur_weights.values().map(|(buffer, _)| buffer.size());
        let params = self.params.pages.iter().map(wgpu::Buffer::size);
        octaves
            .chain(stages)
            .chain(weights)
            .chain(params)
            .sum::<u64>()
            + self.source.bytes()
            + self.staging.bytes()
    }

    #[cfg(test)]
    fn allocations(&self) -> usize {
        let octaves = self
            .octaves
            .iter()
            .map(|octave| {
                octave.gauss.allocations
                    + octave.dog.allocations
                    + octave.temp_a.allocations
                    + octave.temp_b.allocations
                    + octave.candidates.allocations
                    + octave.refined.allocations
            })
            .sum::<usize>();
        let stages = self
            .orientation_keys
            .iter()
            .chain(&self.order_work)
            .chain(&self.ordered_features)
            .chain(&self.oriented_records)
            .chain(&self.oriented_headers)
            .map(|buffer| buffer.allocations)
            .sum::<usize>()
            + self
                .orientation_counts
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>()
            + self
                .orientations
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>()
            + self
                .orientation_offsets
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>()
            + self
                .orientation_prefix
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>()
            + self
                .descriptor_keys
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>()
            + self
                .descriptors
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>()
            + self
                .descriptor_valid
                .iter()
                .map(|buffer| buffer.allocations)
                .sum::<usize>();
        let counters = self
            .octaves
            .iter()
            .filter(|octave| octave.candidate_count.is_some())
            .count()
            + self
                .octaves
                .iter()
                .filter(|octave| octave.refined_count.is_some())
                .count();
        self.source.allocations + octaves + stages + counters + self.staging.allocations
    }
}

// =============================================================================
// wgpu engine
// =============================================================================

struct Engine {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    layouts: Layouts,
    scratch: Mutex<Scratch>,
    faults: Arc<Faults>,
}

/// Device errors that wgpu reports through callbacks rather than return
/// values. Without a handler, wgpu panics on them.
#[derive(Default)]
struct Faults {
    /// The first error reported since the last check, or the first fatal
    /// one, which explains every error that follows it.
    pending: Mutex<Option<Fault>>,
    /// A failure after which the device cannot be trusted.
    fatal: Mutex<Option<Fault>>,
}

#[derive(Clone, Debug)]
enum Fault {
    OutOfMemory,
    Validation(String),
    Lost(String),
    Timeout,
}

impl Fault {
    fn is_fatal(&self) -> bool {
        matches!(self, Self::Lost(_) | Self::Timeout)
    }
}

impl Faults {
    fn record(&self, fault: Fault) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let replaces = match pending.as_ref() {
            None => true,
            Some(current) => fault.is_fatal() && !current.is_fatal(),
        };
        if replaces {
            *pending = Some(fault);
        }
    }

    fn take(&self) -> Option<Fault> {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    fn fatal(&self) -> Option<Fault> {
        self.fatal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn set_fatal(&self, fault: Fault) {
        let mut fatal = self
            .fatal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if fatal.is_none() {
            *fatal = Some(fault);
        }
    }
}

struct Layouts {
    init: wgpu::BindGroupLayout,
    blur: wgpu::BindGroupLayout,
    decimate: wgpu::BindGroupLayout,
    extrema: wgpu::BindGroupLayout,
    refine: wgpu::BindGroupLayout,
    orientation: wgpu::BindGroupLayout,
    describe: wgpu::BindGroupLayout,
    order: wgpu::BindGroupLayout,
    oriented: wgpu::BindGroupLayout,
}

struct Pipelines {
    init: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    decimate: wgpu::ComputePipeline,
    extrema: wgpu::ComputePipeline,
    refine: wgpu::ComputePipeline,
    orientation: wgpu::ComputePipeline,
    describe: wgpu::ComputePipeline,
    order_hist: wgpu::ComputePipeline,
    order_scan: wgpu::ComputePipeline,
    order_scatter: wgpu::ComputePipeline,
    order_bucket_sort: wgpu::ComputePipeline,
    order_hash: wgpu::ComputePipeline,
    order_flag: wgpu::ComputePipeline,
    order_emit: wgpu::ComputePipeline,
    oriented_scan: wgpu::ComputePipeline,
    oriented_compact: wgpu::ComputePipeline,
}

impl Engine {
    fn new() -> Result<Self, SiftGpuError> {
        // Creating wgpu devices on several threads at once can hang or crash
        // in the Vulkan driver (seen with wgpu 30 on an NVIDIA L40S, without
        // any SIFT work), so device creation is serialized process-wide.
        // Extraction itself still runs concurrently on separate extractors.
        static DEVICE_CREATION: Mutex<()> = Mutex::new(());
        let creation = DEVICE_CREATION
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|_| SiftGpuError::NoAdapter)?;
        let limits = adapter.limits();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("openmvg-sift-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            }))?;
        drop(creation);
        // wgpu reports validation, out-of-memory and internal errors through
        // this handler, and panics when none is installed. Record them and
        // return them from the call that caused them.
        let faults = Arc::new(Faults::default());
        let handler_faults = Arc::clone(&faults);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            handler_faults.record(match error {
                wgpu::Error::OutOfMemory { .. } => Fault::OutOfMemory,
                wgpu::Error::Validation { description, .. } => Fault::Validation(description),
                wgpu::Error::Internal { description, .. } => Fault::Lost(description),
            });
        }));
        let lost_faults = Arc::clone(&faults);
        device.set_device_lost_callback(move |reason, message| {
            lost_faults.record(Fault::Lost(format!("{reason:?}: {message}")));
        });
        let layouts = Layouts::new(&device);
        let pipelines = Pipelines::new(&device, &layouts);
        if let Some(fault) = faults.take() {
            return Err(RunError::Fault(fault).into());
        }
        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipelines,
            layouts,
            scratch: Mutex::new(Scratch::default()),
            faults,
        })
    }

    /// Run one extraction on the scratch buffers and turn device errors into
    /// `RunError::Fault`.
    ///
    /// wgpu reports an error from the call that caused it, so the first error
    /// recorded during `body` belongs to this extraction and takes precedence
    /// over its result. After any device error the scratch buffers are
    /// rebuilt, because one of them may be invalid; after a lost device or a
    /// timeout every later call fails. A scratch lock poisoned by a panic in
    /// another thread is recovered the same way.
    fn run<T>(
        &self,
        body: impl FnOnce(&mut Scratch) -> Result<T, RunError>,
    ) -> Result<T, RunError> {
        if let Some(fault) = self.faults.fatal() {
            return Err(RunError::Fault(fault));
        }
        let mut scratch = self.lock_scratch();
        let result = body(&mut scratch);
        let fault = match (self.faults.take(), &result) {
            (Some(fault), _) => Some(fault),
            (None, Err(RunError::Fault(fault))) => Some(fault.clone()),
            (None, _) => None,
        };
        let Some(fault) = fault else {
            return result;
        };
        *scratch = Scratch::default();
        if fault.is_fatal() {
            self.faults.set_fatal(fault.clone());
        }
        Err(RunError::Fault(fault))
    }

    fn lock_scratch(&self) -> MutexGuard<'_, Scratch> {
        self.scratch.lock().unwrap_or_else(|poisoned| {
            // A panic left the scratch in an unknown state; start over.
            let mut scratch = poisoned.into_inner();
            *scratch = Scratch::default();
            self.scratch.clear_poison();
            scratch
        })
    }

    fn ensure_storage_capacity(&self, bytes: usize) -> Result<(), RunError> {
        let limits = self.device.limits();
        let bytes = bytes as u64;
        if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
            return Err(RunError::Capacity {
                what: "device storage limit",
            });
        }
        Ok(())
    }

    fn detect_keypoints(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
    ) -> Result<Vec<ScaleSpaceKeypoint>, RunError> {
        self.run(|scratch| {
            let pass = self.detect_pass::<U8Descriptor>(config, image, scratch, None)?;
            let mut features = pass.features;
            apply_feature_budget(
                &mut features,
                image.ncols(),
                image.nrows(),
                &config.feature_budget,
            );
            Ok(features.into_iter().map(Feature::geometry).collect())
        })
    }

    fn extract<D: DescriptorStorage>(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<D>, RunError> {
        self.run(|scratch| {
            self.extract_with(
                config,
                image,
                scratch,
                DescriptorFormat::F32,
                decode_f32::<D>,
            )
        })
    }

    fn extract_u8(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<U8Descriptor>, RunError> {
        self.run(|scratch| {
            self.extract_with(config, image, scratch, DescriptorFormat::U8, decode_u8)
        })
    }

    /// Detect and describe. Without a feature budget, detection and
    /// description share one submit and one readback. A budget needs the
    /// oriented features on the host before it selects, so the selected
    /// features are described in a second stage.
    fn extract_with<D>(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
        scratch: &mut Scratch,
        format: DescriptorFormat,
        decode: fn(&[u8]) -> Option<D>,
    ) -> Result<ScaleSpaceFeatures<D>, RunError> {
        if config.feature_budget.is_unlimited() {
            let pass = self.detect_pass(config, image, scratch, Some((format, decode)))?;
            return Ok(pass.described.unwrap_or_default());
        }
        let pass = self.detect_pass::<D>(config, image, scratch, None)?;
        let oriented_total = pass.features.len();
        let mut features = pass.features;
        apply_feature_budget(
            &mut features,
            image.ncols(),
            image.nrows(),
            &config.feature_budget,
        );
        // When the budget keeps every feature, the GPU still holds their
        // descriptor keys in detector order.
        let compact_keys = (!features.is_empty() && features.len() == oriented_total).then(|| {
            let octaves = pass.octaves.len();
            let mut keys = CompactKeys {
                totals: vec![0; octaves],
                feature_starts: vec![0; octaves],
                keys: (0..octaves).map(|_| None).collect(),
            };
            let mut start = 0;
            for (octave, (total, key_buffer, header)) in pass.compact.into_iter().enumerate() {
                keys.totals[octave] = total;
                keys.feature_starts[octave] = start;
                keys.keys[octave] = Some((key_buffer, header));
                start += total;
            }
            keys
        });
        let detection = Detection {
            features,
            octaves: pass.octaves,
            compact_keys,
        };
        self.describe(config, &detection, scratch, format, decode)
    }

    /// Run the detection pass, growing the per-octave bounds and running
    /// again when the device counts exceed them. That happens for the first
    /// images an extractor sees and when an image has many more features
    /// than the earlier ones.
    fn detect_pass<D>(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
        scratch: &mut Scratch,
        describe: Describe<D>,
    ) -> Result<Pass<D>, RunError> {
        for _ in 0..MAX_PASS_ATTEMPTS {
            if let Some(pass) = self.detect_features(config, image, scratch, describe)? {
                return Ok(pass);
            }
        }
        Err(RunError::Capacity {
            what: "refined keypoint",
        })
    }

    /// Queue the image upload and return the source buffer and whether it
    /// holds packed bytes.
    ///
    /// Decoded 8-bit images hold exact integers in [0, 255]. Those are sent as
    /// bytes packed four per word, a quarter of the f32 upload; the init
    /// shader widens them back to the same f32 values. Any other image is sent
    /// unchanged.
    fn upload_image(
        &self,
        image: &GrayImage,
        scratch: &mut Scratch,
    ) -> Result<(wgpu::Buffer, bool), RunError> {
        let pixels = image.as_slice();
        let packed_bytes = pixels
            .len()
            .div_ceil(4)
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(RunError::Capacity { what: "input" })?;
        self.ensure_storage_capacity(packed_bytes)?;
        let source = scratch
            .source
            .ensure_storage(&self.device, "omvg-sift-input", packed_bytes);
        scratch.packed_input.resize(pixels.len().div_ceil(4), 0);
        if pack_u8_image(pixels, &mut scratch.packed_input) {
            self.queue
                .write_buffer(&source, 0, bytemuck::cast_slice(&scratch.packed_input));
            return Ok((source, true));
        }
        // Not an 8-bit image: send the f32 values unchanged.
        let float_bytes = pixels
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(RunError::Capacity { what: "input" })?;
        self.ensure_storage_capacity(float_bytes)?;
        let source = scratch
            .source
            .ensure_storage(&self.device, "omvg-sift-input", float_bytes);
        self.queue
            .write_buffer(&source, 0, bytemuck::cast_slice(pixels));
        Ok((source, false))
    }

    /// One detection pass in a single submit: scale space, extrema,
    /// refinement, ordering and deduplication, orientation, compaction and,
    /// with `describe`, the descriptors, then one readback. Returns `None`
    /// when an octave outgrew its bounds; the bounds have grown and the
    /// caller runs the pass again.
    fn detect_features<D>(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
        scratch: &mut Scratch,
        describe: Describe<D>,
    ) -> Result<Option<Pass<D>>, RunError> {
        let height = image.nrows();
        let width = image.ncols();
        if height < MIN_INPUT_DIMENSION || width < MIN_INPUT_DIMENSION {
            return Ok(Some(Pass {
                features: Vec::new(),
                octaves: Vec::new(),
                compact: Vec::new(),
                described: describe.map(|_| ScaleSpaceFeatures::default()),
            }));
        }

        let first_octave = config.first_octave;
        let delta_min = if first_octave == -1 { 0.5f32 } else { 1.0f32 };
        let sigma_min = if first_octave == -1 {
            BASE_SIGMA / 2.0
        } else {
            BASE_SIGMA
        };
        let (base_rows, base_cols) = if first_octave == -1 {
            (height * 2, width * 2)
        } else {
            (height, width)
        };
        let max_octaves = {
            let minimum = base_rows.min(base_cols) as f32;
            let auto = if minimum >= 32.0 {
                (minimum / 32.0).log2().floor() as usize + 1
            } else {
                1
            };
            auto.max(1)
        };
        let num_octaves = config.num_octaves.min(max_octaves).max(1);
        let levels = config.num_scales + 3;
        let dogs = levels - 1;
        let sigma_extra =
            ((sigma_min * sigma_min - ASSUMED_INPUT_SIGMA * ASSUMED_INPUT_SIGMA).max(0.0)).sqrt()
                / delta_min;

        scratch.ensure_stage_buffers(num_octaves);
        let mut octaves = Vec::with_capacity(num_octaves);
        let mut dog_buffers = Vec::with_capacity(num_octaves);
        let mut temp_as = Vec::with_capacity(num_octaves);
        let mut temp_bs = Vec::with_capacity(num_octaves);
        let mut candidate_buffers = Vec::with_capacity(num_octaves);
        let mut candidate_counts = Vec::with_capacity(num_octaves);
        let mut refined_buffers = Vec::with_capacity(num_octaves);
        let mut refined_counts = Vec::with_capacity(num_octaves);
        let mut capacities = Vec::with_capacity(num_octaves);

        for octave in 0..num_octaves {
            let rows = base_rows >> octave;
            let cols = base_cols >> octave;
            let delta = delta_min * (1u32 << octave) as f32;
            let plane = rows.checked_mul(cols).ok_or(RunError::Capacity {
                what: "scale space",
            })?;
            let gauss_bytes = (levels * plane)
                .checked_mul(core::mem::size_of::<f32>())
                .ok_or(RunError::Capacity {
                    what: "scale space",
                })?;
            let dog_bytes = (dogs * plane)
                .checked_mul(core::mem::size_of::<f32>())
                .ok_or(RunError::Capacity {
                    what: "scale space",
                })?;
            let temp_bytes =
                plane
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or(RunError::Capacity {
                        what: "scale space",
                    })?;
            self.ensure_storage_capacity(gauss_bytes)?;
            self.ensure_storage_capacity(dog_bytes)?;
            self.ensure_storage_capacity(temp_bytes)?;
            let slot = &mut scratch.octaves[octave];
            let gauss = slot
                .gauss
                .ensure_storage(&self.device, "omvg-sift-gauss", gauss_bytes);
            let dog = slot
                .dog
                .ensure_storage(&self.device, "omvg-sift-dog", dog_bytes);
            let temp_a = slot
                .temp_a
                .ensure_storage(&self.device, "omvg-sift-temp-a", temp_bytes);
            let temp_b = slot
                .temp_b
                .ensure_storage(&self.device, "omvg-sift-temp-b", temp_bytes);

            let voxels = (rows.saturating_sub(2))
                .saturating_mul(cols.saturating_sub(2))
                .saturating_mul(dogs);
            let requested_capacity = (voxels / CANDIDATE_DIVISOR).max(MIN_CANDIDATES);
            let candidate_bytes = requested_capacity
                .checked_mul(CANDIDATE_BYTES)
                .ok_or(RunError::Capacity { what: "candidate" })?;
            self.ensure_storage_capacity(candidate_bytes)?;
            let candidate_buffer = slot.candidates.ensure_storage(
                &self.device,
                "omvg-sift-candidates",
                candidate_bytes,
            );
            let capacity = (candidate_buffer.size() / CANDIDATE_BYTES as u64) as usize;
            if capacity > u32::MAX as usize {
                return Err(RunError::Capacity { what: "candidate" });
            }
            let refined_bytes = capacity
                .checked_mul(REFINED_BYTES)
                .ok_or(RunError::Capacity {
                    what: "refined keypoint",
                })?;
            self.ensure_storage_capacity(refined_bytes)?;
            let refined_buffer =
                slot.refined
                    .ensure_storage(&self.device, "omvg-sift-refined", refined_bytes);
            let candidate_count = Scratch::counter(
                &mut slot.candidate_count,
                &self.device,
                "omvg-sift-candidate-count",
            );
            let refined_count = Scratch::counter(
                &mut slot.refined_count,
                &self.device,
                "omvg-sift-refined-count",
            );

            octaves.push(Octave {
                rows,
                cols,
                delta,
                gauss,
            });
            dog_buffers.push(dog);
            temp_as.push(temp_a);
            temp_bs.push(temp_b);
            candidate_buffers.push(candidate_buffer);
            candidate_counts.push(candidate_count);
            refined_buffers.push(refined_buffer);
            refined_counts.push(refined_count);
            capacities.push(capacity);
        }

        let (source, packed) = self.upload_image(image, scratch)?;

        let mut encoder = Recorder::new(&self.device, "omvg-sift-scale-space");
        scratch.params.reset();

        // Initial level of the first octave.
        self.encode_init(
            &mut encoder,
            &mut scratch.params,
            &source,
            &temp_as[0],
            InitParams {
                src_rows_cols: [height as u32, width as u32],
                dst_rows_cols: [base_rows as u32, base_cols as u32],
                src_off: 0,
                dst_off: 0,
                upscale: u32::from(first_octave == -1),
                packed: u32::from(packed),
            },
        );
        if sigma_extra > 0.01 {
            let weights = cached_gaussian_weights(
                &self.device,
                &self.queue,
                &mut scratch.blur_weights,
                sigma_extra,
            );
            self.encode_blur_pair(
                &mut encoder,
                &mut scratch.params,
                &weights,
                BlurPair {
                    source: &temp_as[0],
                    temp: &temp_bs[0],
                    destination: &octaves[0].gauss,
                    dog: &dog_buffers[0],
                    rows: base_rows,
                    cols: base_cols,
                    source_offset: 0,
                    destination_offset: 0,
                    dog_offset: None,
                },
            );
        } else {
            encoder.encoder().copy_buffer_to_buffer(
                &temp_as[0],
                0,
                &octaves[0].gauss,
                0,
                (base_rows * base_cols * core::mem::size_of::<f32>()) as u64,
            );
        }

        // Remaining Gaussian levels of every octave. The second blur pass of
        // level `l` also writes DoG plane `l - 1` = level `l` - level `l - 1`,
        // the same f32 subtraction as `compute_dogs`.
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            let plane = rows * cols;
            if octave > 0 {
                let previous = &octaves[octave - 1];
                self.encode_decimate(
                    &mut encoder,
                    &mut scratch.params,
                    &previous.gauss,
                    &octaves[octave].gauss,
                    DecimateParams {
                        src_rows_cols: [previous.rows as u32, previous.cols as u32],
                        dst_rows_cols: [rows as u32, cols as u32],
                        src_off: (config.num_scales * previous.rows * previous.cols) as u32,
                        dst_off: 0,
                        pad0: 0,
                        pad1: 0,
                    },
                );
            }
            for level in 1..levels {
                let sigma_previous = sigma_min
                    * (1u32 << octave) as f32
                    * ((level - 1) as f32 / config.num_scales as f32).exp2();
                let sigma_next = sigma_min
                    * (1u32 << octave) as f32
                    * (level as f32 / config.num_scales as f32).exp2();
                let extra =
                    (((sigma_next * sigma_next - sigma_previous * sigma_previous).max(0.0)).sqrt()
                        / octaves[octave].delta)
                        .max(0.01);
                let weights = cached_gaussian_weights(
                    &self.device,
                    &self.queue,
                    &mut scratch.blur_weights,
                    extra,
                );
                self.encode_blur_pair(
                    &mut encoder,
                    &mut scratch.params,
                    &weights,
                    BlurPair {
                        source: &octaves[octave].gauss,
                        temp: &temp_as[octave],
                        destination: &octaves[octave].gauss,
                        dog: &dog_buffers[octave],
                        rows,
                        cols,
                        source_offset: (level - 1) * plane,
                        destination_offset: level * plane,
                        dog_offset: Some((level - 1) * plane),
                    },
                );
            }
        }

        // Clear detector counters, then run extrema and refinement per octave.
        for count in &candidate_counts {
            encoder.encoder().clear_buffer(count, 0, None);
        }
        for count in &refined_counts {
            encoder.encoder().clear_buffer(count, 0, None);
        }
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            let bind_group = self.bind(
                &mut scratch.params,
                &self.layouts.extrema,
                &ExtremaParams {
                    rows_cols: [rows as u32, cols as u32],
                    levels: levels as u32,
                    threshold: config.peak_threshold / config.num_scales as f32 * 0.8,
                    octave: octave as u32,
                    cap: capacities[octave] as u32,
                    pad0: 0,
                    pad1: 0,
                },
                &[
                    &dog_buffers[octave],
                    &candidate_buffers[octave],
                    &candidate_counts[octave],
                ],
            );
            self.dispatch(
                &mut encoder,
                "omvg-sift-extrema",
                &self.pipelines.extrema,
                &bind_group,
                [
                    (rows as u32).div_ceil(8),
                    (cols as u32).div_ceil(8),
                    levels.saturating_sub(3).max(1) as u32,
                ],
            );
        }
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            let refine_workgroups =
                REFINE_WORKGROUPS.min(capacities[octave].div_ceil(64).max(1) as u32);
            let bind_group = self.bind(
                &mut scratch.params,
                &self.layouts.refine,
                &RefineParams {
                    rows_cols: [rows as u32, cols as u32],
                    levels: levels as u32,
                    num_scales: config.num_scales as u32,
                    delta: octaves[octave].delta,
                    sigma_min,
                    delta_min,
                    edge_threshold: config.edge_threshold,
                    peak_threshold: config.peak_threshold / config.num_scales as f32,
                    cap: capacities[octave] as u32,
                    octave: octave as u32,
                    grid_workgroups: refine_workgroups,
                    pad1: [0; 4],
                },
                &[
                    &dog_buffers[octave],
                    &candidate_buffers[octave],
                    &candidate_counts[octave],
                    &refined_buffers[octave],
                    &refined_counts[octave],
                ],
            );
            self.dispatch(
                &mut encoder,
                "omvg-sift-refine",
                &self.pipelines.refine,
                &bind_group,
                [refine_workgroups, 1, 1],
            );
        }
        // Order, deduplicate, orient and describe every octave in the same
        // submit. Kernels read the live counts on the device and the buffers
        // are sized by the per-octave bounds.
        // Each octave contributes one chain of dependent dispatches. They are
        // recorded stage by stage across octaves, so consecutive dispatches
        // are independent and the device can overlap the octaves; recording
        // one octave's chain after another would put a barrier between every
        // pair of dispatches.
        let mut stages = Vec::with_capacity(num_octaves);
        let mut chains: Vec<Vec<Step<'_>>> = Vec::with_capacity(num_octaves);
        for (octave, octave_data) in octaves.iter().enumerate() {
            let mut steps = Vec::new();
            let (refined_bound, oriented_bound) = scratch.bounds[octave].get(capacities[octave]);
            let order = self.encode_order(
                &mut encoder,
                &mut steps,
                scratch,
                octave,
                octave_data,
                config.num_scales,
                refined_bound,
                &refined_buffers[octave],
                &refined_counts[octave],
            )?;
            let oriented = self.encode_oriented(
                &mut steps,
                scratch,
                octave,
                octave_data,
                &order,
                oriented_bound,
            )?;
            let described = match describe {
                Some((format, _)) => Some(self.encode_descriptors(
                    &mut steps,
                    scratch,
                    config,
                    octave,
                    octave_data,
                    &oriented.keys,
                    &oriented.header,
                    oriented_bound,
                    format,
                )?),
                None => None,
            };
            stages.push((order, oriented, described, oriented_bound));
            chains.push(steps);
        }
        // wgpu inserts a barrier before each dispatch whose inputs changed
        // usage, and a Vulkan barrier waits for all earlier work, so one
        // octave's stage would wait for the previous octave's. The
        // transitions a command buffer needs are hoisted to its start, so
        // each stage gets its own command buffer and its octaves overlap.
        let longest = chains.iter().map(Vec::len).max().unwrap_or(0);
        let mut command_buffers = Vec::with_capacity(longest + 1);
        command_buffers.push(encoder.finish());
        for index in 0..longest {
            let mut stage = Recorder::new(&self.device, "omvg-sift-stage");
            for step in chains.iter().filter_map(|chain| chain.get(index)) {
                self.dispatch(
                    &mut stage,
                    step.label,
                    step.pipeline,
                    &step.bind_group,
                    step.workgroups,
                );
            }
            command_buffers.push(stage.finish());
        }
        drop(chains);
        scratch.params.flush(&self.queue);
        self.queue.submit(command_buffers);

        // First read the counters, which waits for the whole submit, then
        // read exactly the oriented features and descriptors that exist:
        // device-to-host transfers are slow enough that copying up to the
        // bounds would cost more than this extra round trip.
        let mut ranges = Vec::with_capacity(num_octaves * 4);
        for (octave, (order, oriented, _, _)) in stages.iter().enumerate() {
            ranges.push((&candidate_counts[octave], 4));
            ranges.push((&refined_counts[octave], 4));
            ranges.push((&order.work, 8));
            ranges.push((&oriented.header, 12));
        }
        let counts = read_buffers_with(
            &self.device,
            &self.queue,
            &mut scratch.staging,
            &ranges,
            |data, offsets| {
                offsets
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|offsets| {
                        (
                            read_u32(data, offsets[0]) as usize,
                            read_u32(data, offsets[1]) as usize,
                            read_u32(data, offsets[3]) as usize,
                            read_u32(data, offsets[3] + 8) as usize,
                        )
                    })
                    .collect::<Vec<_>>()
            },
        )?;
        let mut truncated = false;
        for (octave, &(candidates, refined, _, oriented_total)) in counts.iter().enumerate() {
            if candidates > capacities[octave] || refined > capacities[octave] {
                return Err(RunError::Capacity { what: "candidate" });
            }
            truncated |= scratch.bounds[octave].update(capacities[octave], refined, oriented_total);
        }
        if truncated {
            return Ok(None);
        }

        let descriptor_bytes = describe.map_or(0, |(format, _)| format.words_per_key() * 4);
        let mut ranges = Vec::with_capacity(num_octaves * 3);
        for ((_, oriented, described, _), &(_, _, written, _)) in stages.iter().zip(&counts) {
            ranges.push((&oriented.records, written * ORIENTED_RECORD_BYTES));
            if let Some((valid, values)) = described {
                ranges.push((valid, written * 4));
                ranges.push((values, written * descriptor_bytes));
            }
        }
        let per_octave = if describe.is_some() { 3 } else { 1 };
        let (features, described) = read_buffers_with(
            &self.device,
            &self.queue,
            &mut scratch.staging,
            &ranges,
            |data, offsets| {
                let mut features = Vec::new();
                let mut described = describe.map(|_| ScaleSpaceFeatures::default());
                for (octave, (offsets, &(_, _, written, _))) in
                    offsets.chunks(per_octave).zip(&counts).enumerate()
                {
                    let records = &data[offsets[0]..offsets[0] + written * ORIENTED_RECORD_BYTES];
                    for (local, record) in records
                        .as_chunks::<ORIENTED_RECORD_BYTES>()
                        .0
                        .iter()
                        .enumerate()
                    {
                        let record = bytemuck::pod_read_unaligned::<GpuOrientedRecord>(record);
                        let feature = Feature {
                            x: record.x,
                            y: record.y,
                            sigma: record.sigma,
                            orientation: record.orientation,
                            peak: record.peak,
                            octave,
                            scale_index: record.level as usize,
                        };
                        features.push(feature);
                        let (Some(out), Some((_, decode))) = (described.as_mut(), describe) else {
                            continue;
                        };
                        if read_u32(data, offsets[1] + local * 4) == 0 {
                            continue;
                        }
                        let start = offsets[2] + local * descriptor_bytes;
                        if let Some(descriptor) = decode(&data[start..start + descriptor_bytes]) {
                            out.keypoints.push(feature.geometry());
                            out.descriptors.push(descriptor);
                        }
                    }
                }
                (features, described)
            },
        )?;
        let compact = stages
            .into_iter()
            .zip(&counts)
            .map(|((_, oriented, _, _), &(_, _, total, _))| (total, oriented.keys, oriented.header))
            .collect();
        Ok(Some(Pass {
            features,
            octaves,
            compact,
            described,
        }))
    }

    /// Describe a budgeted detection in a second submit and readback.
    fn describe<D>(
        &self,
        config: &SiftExtractor,
        detection: &Detection,
        scratch: &mut Scratch,
        format: DescriptorFormat,
        decode: fn(&[u8]) -> Option<D>,
    ) -> Result<ScaleSpaceFeatures<D>, RunError> {
        let readbacks = self.encode_descriptor_readbacks(config, detection, scratch, format)?;
        let bytes = format.words_per_key() * 4;
        self.read_descriptor_buffers(scratch, &readbacks, |data, offsets| {
            assemble_features(
                detection,
                &readbacks,
                offsets,
                |local, offset| {
                    let start = offset + local * bytes;
                    decode(&data[start..start + bytes])
                },
                data,
            )
        })
    }

    fn encode_descriptor_readbacks(
        &self,
        config: &SiftExtractor,
        detection: &Detection,
        scratch: &mut Scratch,
        format: DescriptorFormat,
    ) -> Result<Vec<DescriptorReadback>, RunError> {
        let mut descriptor_readbacks = Vec::new();
        let mut encoder = Recorder::new(&self.device, "omvg-sift-descriptor");
        for (octave_index, octave) in detection.octaves.iter().enumerate() {
            let compact_key = detection.compact_keys.as_ref().and_then(|compact| {
                compact.keys[octave_index]
                    .as_ref()
                    .map(|buffers| (compact, buffers.clone()))
            });
            let (key_count, key_buffer, header, feature_indices, host_keys) =
                if let Some((compact, (key_buffer, header))) = compact_key {
                    let key_count = compact.totals[octave_index];
                    if key_count == 0 {
                        continue;
                    }
                    let start = compact.feature_starts[octave_index];
                    let end = start.checked_add(key_count).ok_or(RunError::Capacity {
                        what: "compact descriptor keys",
                    })?;
                    if end > detection.features.len() {
                        return Err(RunError::Capacity {
                            what: "compact descriptor keys",
                        });
                    }
                    (key_count, key_buffer, header, (start..end).collect(), None)
                } else {
                    let mut keys = Vec::new();
                    let mut feature_indices = Vec::new();
                    for (index, feature) in detection.features.iter().enumerate() {
                        if feature.octave == octave_index {
                            keys.push(GpuDescKey {
                                x: feature.x / octave.delta,
                                y: feature.y / octave.delta,
                                sigma: feature.sigma / octave.delta,
                                orientation: feature.orientation,
                                level: feature.scale_index as u32,
                            });
                            feature_indices.push(index);
                        }
                    }
                    if keys.is_empty() {
                        continue;
                    }
                    let key_count = keys.len();
                    let key_bytes = key_count
                        .checked_mul(core::mem::size_of::<GpuDescKey>())
                        .ok_or(RunError::Capacity {
                            what: "descriptor keys",
                        })?;
                    self.ensure_storage_capacity(key_bytes)?;
                    let key_buffer = scratch.descriptor_keys[octave_index].ensure_storage(
                        &self.device,
                        "omvg-sift-descriptor-keys",
                        key_bytes,
                    );
                    let header = scratch.oriented_headers[octave_index].ensure_storage(
                        &self.device,
                        "omvg-sift-oriented-header",
                        16,
                    );
                    (key_count, key_buffer, header, feature_indices, Some(keys))
                };
            if let Some(keys) = host_keys {
                self.queue
                    .write_buffer(&key_buffer, 0, bytemuck::cast_slice(&keys));
                self.queue.write_buffer(
                    &header,
                    0,
                    bytemuck::cast_slice(&[key_count as u32, 0, 0, 0]),
                );
            }
            let value_bytes = format.value_bytes(key_count).ok_or(RunError::Capacity {
                what: "descriptors",
            })?;
            let valid_bytes =
                key_count
                    .checked_mul(core::mem::size_of::<u32>())
                    .ok_or(RunError::Capacity {
                        what: "descriptor valid flags",
                    })?;
            self.ensure_storage_capacity(value_bytes)?;
            self.ensure_storage_capacity(valid_bytes)?;
            let values = scratch.descriptors[octave_index].ensure_storage(
                &self.device,
                "omvg-sift-descriptors",
                value_bytes,
            );
            let valid = scratch.descriptor_valid[octave_index].ensure_storage(
                &self.device,
                "omvg-sift-descriptor-valid",
                valid_bytes,
            );
            let bind_group = self.bind(
                &mut scratch.params,
                &self.layouts.describe,
                &DescParams {
                    rows_cols: [octave.rows as u32, octave.cols as u32],
                    level: 0,
                    root_sift: u32::from(config.root_sift),
                    key_count: key_count as u32,
                    out_off: 0,
                    gauss_off: 0,
                    quantize_u8: format.quantize_u8(),
                },
                &[&octave.gauss, &key_buffer, &values, &valid, &header],
            );
            self.dispatch(
                &mut encoder,
                "omvg-sift-descriptor",
                &self.pipelines.describe,
                &bind_group,
                workgroup_grid(key_count),
            );
            descriptor_readbacks.push(DescriptorReadback {
                key_count,
                valid_bytes,
                value_bytes,
                valid,
                values,
                feature_indices,
            });
        }
        scratch.params.flush(&self.queue);
        self.queue.submit(Some(encoder.finish()));
        Ok(descriptor_readbacks)
    }

    fn read_descriptor_buffers<R>(
        &self,
        scratch: &mut Scratch,
        readbacks: &[DescriptorReadback],
        decode: impl FnOnce(&[u8], &[usize]) -> R,
    ) -> Result<R, RunError> {
        let mut ranges = Vec::with_capacity(readbacks.len() * 2);
        for readback in readbacks {
            ranges.push((&readback.valid, readback.valid_bytes));
            ranges.push((&readback.values, readback.value_bytes));
        }
        read_buffers_with(
            &self.device,
            &self.queue,
            &mut scratch.staging,
            &ranges,
            decode,
        )
    }

    /// Encode the GPU ordering, deduplication and orientation of one
    /// octave's `n` refined samples (see `order.wgsl`).
    #[allow(clippy::too_many_arguments)]
    fn encode_order<'a>(
        &'a self,
        encoder: &mut Recorder,
        steps: &mut Vec<Step<'a>>,
        scratch: &mut Scratch,
        octave: usize,
        octave_data: &Octave,
        num_scales: usize,
        n: usize,
        refined: &wgpu::Buffer,
        refined_count: &wgpu::Buffer,
    ) -> Result<OrderedOctave, RunError> {
        let capacity_error = RunError::Capacity {
            what: "refined keypoint",
        };
        let (rows, cols) = (octave_data.rows, octave_data.cols);
        // Refined sample indices must fit u32 for every DoG level.
        if (num_scales + 3)
            .checked_mul(rows * cols)
            .is_none_or(|samples| samples >= u32::MAX as usize)
        {
            return Err(capacity_error);
        }
        let buckets = num_scales * rows;
        let hash_slots = (2 * n).next_power_of_two();
        // `work[0]` is the kept count, `work[1]` the overflow flag.
        let mut offset = 2usize;
        let mut region = |len: usize| {
            let start = offset;
            offset += len;
            start as u32
        };
        let off_count = region(buckets);
        let off_start = region(buckets);
        let off_cursor = region(buckets);
        let off_sorted = region(n);
        let off_hash_keys = region(hash_slots);
        let off_hash_vals = region(hash_slots);
        let off_slot = region(n);
        let off_kept = region(n);
        let off_pos = region(n);
        let work_bytes =
            offset
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(RunError::Capacity {
                    what: "refined keypoint",
                })?;
        if offset > u32::MAX as usize {
            return Err(RunError::Capacity {
                what: "refined keypoint",
            });
        }
        let feature_bytes = n * ORDERED_FEATURE_BYTES;
        let key_bytes = n * core::mem::size_of::<GpuOriKey>();
        let count_bytes = n * core::mem::size_of::<u32>();
        let orientation_bytes = n * MAX_ORIENTATIONS * core::mem::size_of::<f32>();
        let prefix_bytes = n * ORIENTATION_PREFIX * core::mem::size_of::<f32>();
        for bytes in [
            work_bytes,
            feature_bytes,
            key_bytes,
            count_bytes,
            orientation_bytes,
            prefix_bytes,
        ] {
            self.ensure_storage_capacity(bytes)?;
        }
        let device = &self.device;
        let work = scratch.order_work[octave].ensure_storage(device, "omvg-sift-order", work_bytes);
        let features = scratch.ordered_features[octave].ensure_storage(
            device,
            "omvg-sift-ordered",
            feature_bytes,
        );
        let keys = scratch.orientation_keys[octave].ensure_storage(
            device,
            "omvg-sift-orientation-keys",
            key_bytes,
        );
        let counts = scratch.orientation_counts[octave].ensure_storage(
            device,
            "omvg-sift-orientation-counts",
            count_bytes,
        );
        let orientations = scratch.orientations[octave].ensure_storage(
            device,
            "omvg-sift-orientations",
            orientation_bytes,
        );
        let prefix = scratch.orientation_prefix[octave].ensure_storage(
            device,
            "omvg-sift-orientation-prefix",
            prefix_bytes,
        );

        encoder
            .encoder()
            .clear_buffer(&work, 0, Some(work_bytes as u64));
        let base = OrderParams {
            n: n as u32,
            rows: rows as u32,
            cols: cols as u32,
            buckets: buckets as u32,
            pad0: 0,
            hash_mask: (hash_slots - 1) as u32,
            inv_delta: 1.0 / octave_data.delta,
            scan_len: 0,
            scan_src: 0,
            scan_dst: 0,
            scan_copy: ORDER_NO_COPY,
            off_count,
            off_start,
            off_cursor,
            off_sorted,
            off_hash_keys,
            off_hash_vals,
            off_slot,
            off_kept,
            off_pos,
        };
        let bucket_scan = OrderParams {
            scan_len: buckets as u32,
            scan_src: off_count,
            scan_dst: off_start,
            scan_copy: off_cursor,
            ..base
        };
        let kept_scan = OrderParams {
            scan_len: ORDER_SCAN_LIVE,
            scan_src: off_kept,
            scan_dst: off_pos,
            scan_copy: ORDER_NO_COPY,
            ..base
        };
        let storage = [refined, &work, &features, &keys, refined_count];
        let items = item_grid(n);
        let order_steps: [(&'static str, &wgpu::ComputePipeline, &OrderParams, [u32; 3]); 8] = [
            (
                "omvg-sift-order-hist",
                &self.pipelines.order_hist,
                &base,
                items,
            ),
            (
                "omvg-sift-order-scan",
                &self.pipelines.order_scan,
                &bucket_scan,
                [1, 1, 1],
            ),
            (
                "omvg-sift-order-scatter",
                &self.pipelines.order_scatter,
                &base,
                items,
            ),
            (
                "omvg-sift-order-bucket-sort",
                &self.pipelines.order_bucket_sort,
                &base,
                item_grid(buckets),
            ),
            (
                "omvg-sift-order-hash",
                &self.pipelines.order_hash,
                &base,
                items,
            ),
            (
                "omvg-sift-order-flag",
                &self.pipelines.order_flag,
                &base,
                items,
            ),
            (
                "omvg-sift-order-scan",
                &self.pipelines.order_scan,
                &kept_scan,
                [1, 1, 1],
            ),
            (
                "omvg-sift-order-emit",
                &self.pipelines.order_emit,
                &base,
                items,
            ),
        ];
        for (label, pipeline, params, workgroups) in order_steps {
            let bind_group = self.bind(&mut scratch.params, &self.layouts.order, params, &storage);
            steps.push(Step {
                label,
                pipeline,
                bind_group,
                workgroups,
            });
        }

        let bind_group = self.bind(
            &mut scratch.params,
            &self.layouts.orientation,
            &OriParams {
                rows_cols: [rows as u32, cols as u32],
                level: 0,
                key_count: n as u32,
                gauss_off: 0,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
            &[
                &octave_data.gauss,
                &keys,
                &counts,
                &orientations,
                &prefix,
                &work,
            ],
        );
        steps.push(Step {
            label: "omvg-sift-orientation",
            pipeline: &self.pipelines.orientation,
            bind_group,
            workgroups: workgroup_grid(n),
        });
        Ok(OrderedOctave {
            bound: n,
            work,
            features,
            counts,
            orientations,
        })
    }

    /// Encode the prefix sum and compaction of one octave's orientation
    /// peaks into at most `bound` oriented features (see `oriented.wgsl`).
    fn encode_oriented<'a>(
        &'a self,
        steps: &mut Vec<Step<'a>>,
        scratch: &mut Scratch,
        octave: usize,
        octave_data: &Octave,
        order: &OrderedOctave,
        bound: usize,
    ) -> Result<OrientedOctave, RunError> {
        let offset_bytes = order.bound * core::mem::size_of::<u32>();
        let key_bytes = bound * core::mem::size_of::<GpuDescKey>();
        let record_bytes = bound * ORIENTED_RECORD_BYTES;
        for bytes in [offset_bytes, key_bytes, record_bytes] {
            self.ensure_storage_capacity(bytes)?;
        }
        let device = &self.device;
        let offsets = scratch.orientation_offsets[octave].ensure_storage(
            device,
            "omvg-sift-orientation-offsets",
            offset_bytes,
        );
        let keys = scratch.descriptor_keys[octave].ensure_storage(
            device,
            "omvg-sift-compact-keys",
            key_bytes,
        );
        let records = scratch.oriented_records[octave].ensure_storage(
            device,
            "omvg-sift-oriented",
            record_bytes,
        );
        let header = scratch.oriented_headers[octave].ensure_storage(
            device,
            "omvg-sift-oriented-header",
            16,
        );
        let params = OrientedParams {
            n: order.bound as u32,
            k: bound as u32,
            inv_delta: 1.0 / octave_data.delta,
            pad0: 0,
        };
        let storage = [
            &order.work,
            &order.counts,
            &order.orientations,
            &order.features,
            &offsets,
            &keys,
            &records,
            &header,
        ];
        let bind_group = self.bind(
            &mut scratch.params,
            &self.layouts.oriented,
            &params,
            &storage,
        );
        steps.push(Step {
            label: "omvg-sift-oriented-scan",
            pipeline: &self.pipelines.oriented_scan,
            bind_group,
            workgroups: [1, 1, 1],
        });
        let bind_group = self.bind(
            &mut scratch.params,
            &self.layouts.oriented,
            &params,
            &storage,
        );
        steps.push(Step {
            label: "omvg-sift-oriented-compact",
            pipeline: &self.pipelines.oriented_compact,
            bind_group,
            workgroups: item_grid(order.bound),
        });
        Ok(OrientedOctave {
            keys,
            records,
            header,
        })
    }

    /// Encode the descriptors of one octave's oriented features, at most
    /// `bound` of them; `header[0]` holds the live count on the GPU.
    #[allow(clippy::too_many_arguments)]
    fn encode_descriptors<'a>(
        &'a self,
        steps: &mut Vec<Step<'a>>,
        scratch: &mut Scratch,
        config: &SiftExtractor,
        octave: usize,
        octave_data: &Octave,
        keys: &wgpu::Buffer,
        header: &wgpu::Buffer,
        bound: usize,
        format: DescriptorFormat,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer), RunError> {
        let value_bytes = format.value_bytes(bound).ok_or(RunError::Capacity {
            what: "descriptors",
        })?;
        let valid_bytes = bound * core::mem::size_of::<u32>();
        self.ensure_storage_capacity(value_bytes)?;
        self.ensure_storage_capacity(valid_bytes)?;
        let values = scratch.descriptors[octave].ensure_storage(
            &self.device,
            "omvg-sift-descriptors",
            value_bytes,
        );
        let valid = scratch.descriptor_valid[octave].ensure_storage(
            &self.device,
            "omvg-sift-descriptor-valid",
            valid_bytes,
        );
        let bind_group = self.bind(
            &mut scratch.params,
            &self.layouts.describe,
            &DescParams {
                rows_cols: [octave_data.rows as u32, octave_data.cols as u32],
                level: 0,
                root_sift: u32::from(config.root_sift),
                key_count: bound as u32,
                out_off: 0,
                gauss_off: 0,
                quantize_u8: format.quantize_u8(),
            },
            &[&octave_data.gauss, keys, &values, &valid, header],
        );
        steps.push(Step {
            label: "omvg-sift-descriptor",
            pipeline: &self.pipelines.describe,
            bind_group,
            workgroups: workgroup_grid(bound),
        });
        Ok((valid, values))
    }

    /// Create a bind group whose binding 0 is `value` in the parameter arena
    /// and whose bindings `1..` are `buffers`, in order.
    fn bind<T: bytemuck::Pod>(
        &self,
        params: &mut ParamArena,
        layout: &wgpu::BindGroupLayout,
        value: &T,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let (buffer, offset) = params.push(&self.device, value);
        let mut entries = Vec::with_capacity(buffers.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset,
                size: wgpu::BufferSize::new(PARAM_SLOT_BYTES),
            }),
        });
        for (index, storage) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: index as u32 + 1,
                resource: storage.as_entire_binding(),
            });
        }
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    /// Record one dispatch. Consecutive dispatches share one compute pass,
    /// which wgpu splits with the barriers the data dependencies need; a pass
    /// per dispatch would make `CommandEncoder::finish` far more expensive.
    fn dispatch(
        &self,
        recorder: &mut Recorder,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        let pass = recorder.pass.get_or_insert_with(|| {
            recorder
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                })
                .forget_lifetime()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    fn encode_init(
        &self,
        encoder: &mut Recorder,

        params: &mut ParamArena,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        init: InitParams,
    ) {
        let bind_group = self.bind(params, &self.layouts.init, &init, &[source, destination]);
        self.dispatch(
            encoder,
            "omvg-sift-init",
            &self.pipelines.init,
            &bind_group,
            [
                init.dst_rows_cols[0].div_ceil(16),
                init.dst_rows_cols[1].div_ceil(16),
                1,
            ],
        );
    }

    /// Separable blur of one plane in two passes: `source` into `temp` along
    /// rows, then `temp` into `destination` across columns. The destination may be the
    /// same buffer as `source` at another plane; the passes are ordered. With
    /// `dog_offset`, the second pass also writes the DoG plane (this level
    /// minus the source level) into `dog`.
    fn encode_blur_pair(
        &self,
        encoder: &mut Recorder,
        params: &mut ParamArena,
        weights: &(wgpu::Buffer, usize),
        pair: BlurPair<'_>,
    ) {
        let (weights, radius) = weights;
        let workgroups = [
            (pair.rows as u32).div_ceil(16),
            (pair.cols as u32).div_ceil(16),
            1,
        ];
        let first = self.bind(
            params,
            &self.layouts.blur,
            &BlurParams {
                rows_cols: [pair.rows as u32, pair.cols as u32],
                src_off: pair.source_offset as u32,
                dst_off: 0,
                radius: *radius as u32,
                wt_off: 0,
                vertical: 1,
                dog: 0,
                dog_off: 0,
                pad: [0; 3],
            },
            &[pair.source, pair.temp, weights, pair.dog],
        );
        self.dispatch(
            encoder,
            "omvg-sift-blur",
            &self.pipelines.blur,
            &first,
            workgroups,
        );
        let second = self.bind(
            params,
            &self.layouts.blur,
            &BlurParams {
                rows_cols: [pair.rows as u32, pair.cols as u32],
                src_off: 0,
                dst_off: pair.destination_offset as u32,
                radius: *radius as u32,
                wt_off: 0,
                vertical: 0,
                dog: u32::from(pair.dog_offset.is_some()),
                dog_off: pair.dog_offset.unwrap_or(0) as u32,
                pad: [0; 3],
            },
            &[pair.temp, pair.destination, weights, pair.dog],
        );
        self.dispatch(
            encoder,
            "omvg-sift-blur",
            &self.pipelines.blur,
            &second,
            workgroups,
        );
    }

    fn encode_decimate(
        &self,
        encoder: &mut Recorder,
        params: &mut ParamArena,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        decimate: DecimateParams,
    ) {
        let bind_group = self.bind(
            params,
            &self.layouts.decimate,
            &decimate,
            &[source, destination],
        );
        self.dispatch(
            encoder,
            "omvg-sift-decimate",
            &self.pipelines.decimate,
            &bind_group,
            [
                decimate.dst_rows_cols[0].div_ceil(16),
                decimate.dst_rows_cols[1].div_ceil(16),
                1,
            ],
        );
    }
}

/// One recorded dispatch, kept until its octave's chain is interleaved with
/// the other octaves.
struct Step<'a> {
    label: &'static str,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroups: [u32; 3],
}

/// A command encoder that records consecutive dispatches into one open
/// compute pass. Buffer copies and clears end the pass first.
struct Recorder {
    encoder: wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl Recorder {
    fn new(device: &wgpu::Device, label: &str) -> Self {
        Self {
            encoder: device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
            pass: None,
        }
    }

    /// The encoder for non-dispatch commands, after ending any open pass.
    fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.pass = None;
        &mut self.encoder
    }

    fn finish(mut self) -> wgpu::CommandBuffer {
        self.pass = None;
        self.encoder.finish()
    }
}

/// Buffers and planes of one [`Engine::encode_blur_pair`] call.
struct BlurPair<'a> {
    source: &'a wgpu::Buffer,
    temp: &'a wgpu::Buffer,
    destination: &'a wgpu::Buffer,
    dog: &'a wgpu::Buffer,
    rows: usize,
    cols: usize,
    source_offset: usize,
    destination_offset: usize,
    dog_offset: Option<usize>,
}

/// A 2D grid of 64-invocation workgroups covering at least `count` items,
/// for the `order.wgsl` kernels.
fn item_grid(count: usize) -> [u32; 3] {
    workgroup_grid(count.div_ceil(64))
}

/// A 2D workgroup grid holding at least `count` workgroups, for kernels that
/// own one item per workgroup; dispatch dimensions are limited to 65535.
fn workgroup_grid(count: usize) -> [u32; 3] {
    const MAX: usize = 65_535;
    let count = count.max(1);
    if count <= MAX {
        [count as u32, 1, 1]
    } else {
        [MAX as u32, count.div_ceil(MAX) as u32, 1]
    }
}

/// Size of one parameter slot; every uniform struct fits and the device
/// offset alignment divides it (checked when the engine is created).
const PARAM_SLOT_BYTES: u64 = 256;
const PARAM_PAGE_SLOTS: usize = 256;

/// One extraction's uniform parameters. Each pass gets a slot in a page of a
/// persistent uniform buffer; the bytes are staged on the host and written
/// once per submit, instead of one buffer and one write per pass. Pages are
/// kept between extractions and added only when an extraction needs more.
#[derive(Default)]
struct ParamArena {
    pages: Vec<wgpu::Buffer>,
    data: Vec<u8>,
    flushed: usize,
}

impl ParamArena {
    fn reset(&mut self) {
        self.data.clear();
        self.flushed = 0;
    }

    fn push<T: bytemuck::Pod>(&mut self, device: &wgpu::Device, value: &T) -> (wgpu::Buffer, u64) {
        let slot = self.data.len() / PARAM_SLOT_BYTES as usize;
        let page = slot / PARAM_PAGE_SLOTS;
        let bytes = bytemuck::bytes_of(value);
        self.data.extend_from_slice(bytes);
        self.data.resize((slot + 1) * PARAM_SLOT_BYTES as usize, 0);
        while self.pages.len() <= page {
            self.pages
                .push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("omvg-sift-params"),
                    size: PARAM_SLOT_BYTES * PARAM_PAGE_SLOTS as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        let offset = (slot % PARAM_PAGE_SLOTS) as u64 * PARAM_SLOT_BYTES;
        (self.pages[page].clone(), offset)
    }

    /// Write the slots added since the previous flush. Call before submitting
    /// the commands that read them; queue writes land before that submit.
    fn flush(&mut self, queue: &wgpu::Queue) {
        let page_bytes = PARAM_SLOT_BYTES as usize * PARAM_PAGE_SLOTS;
        while self.flushed < self.data.len() {
            let page = self.flushed / page_bytes;
            let end = self.data.len().min((page + 1) * page_bytes);
            queue.write_buffer(
                &self.pages[page],
                (self.flushed - page * page_bytes) as u64,
                &self.data[self.flushed..end],
            );
            self.flushed = end;
        }
    }
}

impl Layouts {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            init: layout(
                device,
                "omvg-sift-init",
                &[uniform(0), storage(1, true), storage(2, false)],
            ),
            blur: layout(
                device,
                "omvg-sift-blur",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, false),
                    storage(3, true),
                    storage(4, false),
                ],
            ),
            decimate: layout(
                device,
                "omvg-sift-decimate",
                &[uniform(0), storage(1, true), storage(2, false)],
            ),
            extrema: layout(
                device,
                "omvg-sift-extrema",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, false),
                    storage(3, false),
                ],
            ),
            refine: layout(
                device,
                "omvg-sift-refine",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, true),
                    storage(3, true),
                    storage(4, false),
                    storage(5, false),
                ],
            ),
            orientation: layout(
                device,
                "omvg-sift-orientation",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, true),
                    storage(3, false),
                    storage(4, false),
                    storage(5, false),
                    storage(6, true),
                ],
            ),
            order: layout(
                device,
                "omvg-sift-order",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, false),
                    storage(3, false),
                    storage(4, false),
                    storage(5, true),
                ],
            ),
            oriented: layout(
                device,
                "omvg-sift-oriented",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, true),
                    storage(3, true),
                    storage(4, true),
                    storage(5, false),
                    storage(6, false),
                    storage(7, false),
                    storage(8, false),
                ],
            ),
            describe: layout(
                device,
                "omvg-sift-descriptor",
                &[
                    uniform(0),
                    storage(1, true),
                    storage(2, true),
                    storage(3, false),
                    storage(4, false),
                    storage(5, true),
                ],
            ),
        }
    }
}

impl Pipelines {
    fn new(device: &wgpu::Device, layouts: &Layouts) -> Self {
        let order = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("omvg-sift-order"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/order.wgsl").into()),
        });
        let oriented = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("omvg-sift-oriented"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/oriented.wgsl").into()),
        });
        Self {
            init: pipeline(
                device,
                "omvg-sift-init",
                layouts.init.clone(),
                include_str!("shaders/init.wgsl"),
            ),
            blur: pipeline(
                device,
                "omvg-sift-blur",
                layouts.blur.clone(),
                include_str!("shaders/blur.wgsl"),
            ),
            decimate: pipeline(
                device,
                "omvg-sift-decimate",
                layouts.decimate.clone(),
                include_str!("shaders/decimate.wgsl"),
            ),
            extrema: pipeline(
                device,
                "omvg-sift-extrema",
                layouts.extrema.clone(),
                include_str!("shaders/extrema.wgsl"),
            ),
            refine: pipeline(
                device,
                "omvg-sift-refine",
                layouts.refine.clone(),
                include_str!("shaders/refine.wgsl"),
            ),
            orientation: pipeline(
                device,
                "omvg-sift-orientation",
                layouts.orientation.clone(),
                include_str!("shaders/orientation.wgsl"),
            ),
            describe: pipeline(
                device,
                "omvg-sift-descriptor",
                layouts.describe.clone(),
                include_str!("shaders/describe.wgsl"),
            ),
            order_hist: order_pipeline(device, &order, &layouts.order, "hist"),
            order_scan: order_pipeline(device, &order, &layouts.order, "scan"),
            order_scatter: order_pipeline(device, &order, &layouts.order, "scatter"),
            order_bucket_sort: order_pipeline(device, &order, &layouts.order, "bucket_sort"),
            order_hash: order_pipeline(device, &order, &layouts.order, "hash_insert"),
            order_flag: order_pipeline(device, &order, &layouts.order, "flag"),
            order_emit: order_pipeline(device, &order, &layouts.order, "emit"),
            oriented_scan: order_pipeline(device, &oriented, &layouts.oriented, "scan"),
            oriented_compact: order_pipeline(device, &oriented, &layouts.oriented, "compact"),
        }
    }
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    bind_group_layout: wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn order_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("omvg-sift-order"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

const fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// =============================================================================
// Host-side layouts and helpers
// =============================================================================

fn create_buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).div_ceil(4) * 4,
        usage,
        mapped_at_creation: false,
    })
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    create_buffer(
        device,
        label,
        bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

/// Copy the leading `bytes` of each buffer into one staging buffer, wait for
/// the GPU, and run `decode` on the mapped bytes and each range's offset in
/// them, without first copying the mapping into host memory.
fn read_buffers_with<R>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    staging_buffer: &mut GrowBuffer,
    ranges: &[(&wgpu::Buffer, usize)],
    decode: impl FnOnce(&[u8], &[usize]) -> R,
) -> Result<R, RunError> {
    let mut offsets = Vec::with_capacity(ranges.len());
    let mut total = 0usize;
    for &(_, bytes) in ranges {
        let aligned = bytes
            .div_ceil(4)
            .checked_mul(4)
            .ok_or(RunError::Capacity { what: "readback" })?;
        offsets.push(total);
        total = total
            .checked_add(aligned)
            .ok_or(RunError::Capacity { what: "readback" })?;
    }
    if total == 0 {
        return Ok(decode(&[], &offsets));
    }
    let staging = staging_buffer.ensure_readback(device, "omvg-sift-staging", total);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("omvg-sift-staging"),
    });
    for ((buffer, bytes), offset) in ranges.iter().zip(offsets.iter()) {
        let size = (*bytes).div_ceil(4) as u64 * 4;
        if size > 0 {
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, *offset as u64, size);
        }
    }
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..total as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let deadline = std::time::Instant::now() + SiftGpuExtractor::TIMEOUT;
    let map_result = loop {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| RunError::Readback(error.to_string()))?;
        match receiver.try_recv() {
            Ok(result) => break result,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(RunError::Readback(
                    "mapping callback disconnected".to_string(),
                ));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        if std::time::Instant::now() >= deadline {
            return Err(RunError::Fault(Fault::Timeout));
        }
        std::thread::yield_now();
    };
    map_result.map_err(|error| RunError::Readback(format!("{error:?}")))?;
    let result = {
        let data = slice
            .get_mapped_range()
            .map_err(|error| RunError::Readback(error.to_string()))?;
        decode(&data, &offsets)
    };
    staging.unmap();
    Ok(result)
}

/// Pair each described feature with its descriptor, in detector order.
///
/// Readbacks come in octave order and each covers its octave's features in
/// order, so visiting them in turn visits `detection.features` in order.
/// `descriptor(local, offset)` decodes the descriptor of key `local` of a
/// readback whose values start at `offset`; keys flagged invalid on the GPU,
/// and descriptors the storage rejects, are dropped like on the CPU.
fn assemble_features<D>(
    detection: &Detection,
    readbacks: &[DescriptorReadback],
    offsets: &[usize],
    descriptor: impl Fn(usize, usize) -> Option<D>,
    data: &[u8],
) -> ScaleSpaceFeatures<D> {
    let mut features = ScaleSpaceFeatures::default();
    features.keypoints.reserve(detection.features.len());
    features.descriptors.reserve(detection.features.len());
    for (readback, offsets) in readbacks.iter().zip(offsets.as_chunks::<2>().0) {
        let valid = &data[offsets[0]..offsets[0] + readback.key_count * 4];
        for (local, flag) in valid.as_chunks::<4>().0.iter().enumerate() {
            if u32::from_le_bytes(*flag) == 0 {
                continue;
            }
            if let (Some(feature), Some(descriptor)) = (
                detection.features.get(readback.feature_indices[local]),
                descriptor(local, offsets[1]),
            ) {
                features.keypoints.push(feature.geometry());
                features.descriptors.push(descriptor);
            }
        }
    }
    features
}

/// Pack `pixels` four per `u32` (little endian) into `packed` when every
/// value is exactly an integer in `[0, 255]`; return whether it did.
/// `packed` holds `pixels.len().div_ceil(4)` words.
///
/// Adding 2^23 moves an integer in `[0, 255]` into the low mantissa bits, so
/// the byte is read from the sum and exactness is checked by subtracting it
/// again. The range check compares bit patterns, so `-0.0`, NaN and negative
/// values keep the f32 upload and the GPU input is bit-identical to `pixels`.
/// The loop is branch-free so it vectorizes; with `parallel` it also splits
/// the image over Rayon workers.
fn pack_u8_image(pixels: &[f32], packed: &mut [u32]) -> bool {
    const BLOCK: usize = 1 << 16;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        packed
            .par_chunks_mut(BLOCK / 4)
            .zip(pixels.par_chunks(BLOCK))
            .map(|(words, pixels)| pack_u8_block(pixels, words))
            .reduce(|| true, |a, b| a && b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        packed
            .chunks_mut(BLOCK / 4)
            .zip(pixels.chunks(BLOCK))
            .fold(true, |exact, (words, pixels)| {
                pack_u8_block(pixels, words) && exact
            })
    }
}

fn pack_u8_block(pixels: &[f32], words: &mut [u32]) -> bool {
    const MAGIC: f32 = 8_388_608.0;
    const MAX_BITS: u32 = 0x437f_0000; // 255.0
    let mut inexact = 0u32;
    let mut pack = |values: &[f32]| {
        let mut word = 0u32;
        for (lane, &value) in values.iter().enumerate() {
            let shifted = value + MAGIC;
            inexact |= u32::from(value.to_bits() > MAX_BITS) | u32::from(shifted - MAGIC != value);
            word |= (shifted.to_bits() & 0xff) << (8 * lane);
        }
        word
    };
    let (chunks, tail) = pixels.as_chunks::<4>();
    for (word, chunk) in words.iter_mut().zip(chunks) {
        *word = pack(chunk);
    }
    if let Some(word) = words.get_mut(chunks.len()) {
        *word = pack(tail);
    }
    inexact == 0
}

/// A u8 descriptor from its packed bytes. The shader packs byte `4w + l`
/// into lane `l` of word `w`, so the little-endian words are the bytes in
/// order.
fn decode_u8(bytes: &[u8]) -> Option<U8Descriptor> {
    let bytes: [u8; SIFT_DESCR_SIZE] = bytes.try_into().ok()?;
    U8Descriptor::try_new(bytes, 255.0).ok()
}

/// A descriptor from its finalized f32 values, stored as little-endian bits.
fn decode_f32<D: DescriptorStorage>(bytes: &[u8]) -> Option<D> {
    let mut values = [0.0f32; SIFT_DESCR_SIZE];
    for (value, word) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_le_bytes(*word);
    }
    D::try_from_finalized(&values).ok()
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(word)
}

fn gaussian_weights(sigma: f32) -> Vec<f32> {
    if sigma < f32::EPSILON {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let mut full = vec![0.0f32; 2 * radius + 1];
    let mut sum = 0.0f32;
    for (index, value) in full.iter_mut().enumerate() {
        let distance = (index as f32 - radius as f32) / sigma;
        *value = (-0.5 * distance * distance).exp();
        sum += *value;
    }
    (0..=radius)
        .map(|offset| full[radius + offset] / sum)
        .collect()
}

/// The blur weight buffer and kernel radius for `sigma`, uploaded once and
/// then reused for every image.
fn cached_gaussian_weights(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut BTreeMap<u32, (wgpu::Buffer, usize)>,
    sigma: f32,
) -> (wgpu::Buffer, usize) {
    let key = sigma.to_bits();
    if let Some(entry) = cache.get(&key) {
        return entry.clone();
    }
    let kernel = gaussian_weights(sigma);
    let radius = kernel.len().saturating_sub(1);
    let bytes = kernel.len().saturating_mul(core::mem::size_of::<f32>());
    let buffer = storage_buffer(device, "omvg-sift-blur-weights", bytes as u64);
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&kernel));
    cache.insert(key, (buffer.clone(), radius));
    (buffer, radius)
}

// =============================================================================
// GPU layouts
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InitParams {
    src_rows_cols: [u32; 2],
    dst_rows_cols: [u32; 2],
    src_off: u32,
    dst_off: u32,
    upscale: u32,
    packed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    rows_cols: [u32; 2],
    src_off: u32,
    dst_off: u32,
    radius: u32,
    wt_off: u32,
    vertical: u32,
    dog: u32,
    dog_off: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecimateParams {
    src_rows_cols: [u32; 2],
    dst_rows_cols: [u32; 2],
    src_off: u32,
    dst_off: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExtremaParams {
    rows_cols: [u32; 2],
    levels: u32,
    threshold: f32,
    octave: u32,
    cap: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RefineParams {
    rows_cols: [u32; 2],
    levels: u32,
    num_scales: u32,
    delta: f32,
    sigma_min: f32,
    delta_min: f32,
    edge_threshold: f32,
    peak_threshold: f32,
    cap: u32,
    octave: u32,
    grid_workgroups: u32,
    pad1: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OriParams {
    rows_cols: [u32; 2],
    level: u32,
    key_count: u32,
    gauss_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DescParams {
    rows_cols: [u32; 2],
    level: u32,
    root_sift: u32,
    key_count: u32,
    out_off: u32,
    gauss_off: u32,
    quantize_u8: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRefined {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    value: f32,
    octave: u32,
    level: u32,
    i: i32,
    j: i32,
    orig_level: u32,
    orig_i: i32,
    orig_j: i32,
    accepted: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OrderParams {
    n: u32,
    rows: u32,
    cols: u32,
    buckets: u32,
    pad0: u32,
    hash_mask: u32,
    inv_delta: f32,
    scan_len: u32,
    scan_src: u32,
    scan_dst: u32,
    scan_copy: u32,
    off_count: u32,
    off_start: u32,
    off_cursor: u32,
    off_sorted: u32,
    off_hash_keys: u32,
    off_hash_vals: u32,
    off_slot: u32,
    off_kept: u32,
    off_pos: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OrientedParams {
    n: u32,
    k: u32,
    inv_delta: f32,
    pad0: u32,
}

/// One oriented feature as `oriented.wgsl` writes it, in image units.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOrientedRecord {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    peak: f32,
    level: u32,
}

/// One kept feature as `order.wgsl` writes it.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFeature {
    x: f32,
    y: f32,
    sigma: f32,
    peak: f32,
    level: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOriKey {
    x: f32,
    y: f32,
    sigma: f32,
    level: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDescKey {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    level: u32,
}

// =============================================================================
// Internal errors
// =============================================================================

#[derive(Debug, thiserror::Error)]
enum RunError {
    #[error(
        "the GPU SIFT {what} stage exceeded a buffer or device storage limit; use the CPU extractor for this image"
    )]
    Capacity { what: &'static str },
    #[error("GPU SIFT readback failed: {0}")]
    Readback(String),
    #[error("GPU SIFT device error: {0:?}")]
    Fault(Fault),
}

impl From<RunError> for SiftGpuError {
    fn from(error: RunError) -> Self {
        match error {
            RunError::Capacity { what } => Self::Capacity { what },
            RunError::Readback(message) => Self::Readback(message),
            RunError::Fault(Fault::OutOfMemory) => Self::OutOfMemory,
            RunError::Fault(Fault::Validation(message)) => Self::Validation(message),
            RunError::Fault(Fault::Lost(message)) => Self::DeviceLost(message),
            RunError::Fault(Fault::Timeout) => Self::Timeout {
                timeout: SiftGpuExtractor::TIMEOUT,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::FeatureBudgetPolicy;

    #[test]
    fn gpu_extractor_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SiftGpuExtractor>();
    }

    fn synthetic_blobs(size: usize) -> GrayImage {
        let mut image = GrayImage::zeros(size, size);
        let blobs = [
            (size as f32 * 0.25, size as f32 * 0.25, 3.0, 220.0),
            (size as f32 * 0.7, size as f32 * 0.3, 5.0, 180.0),
            (size as f32 * 0.4, size as f32 * 0.7, 8.0, 150.0),
            (size as f32 * 0.8, size as f32 * 0.8, 12.0, 120.0),
        ];
        for row in 0..size {
            for col in 0..size {
                let mut value = 20.0f32;
                for &(x, y, sigma, amplitude) in &blobs {
                    let dx = col as f32 - x;
                    let dy = row as f32 - y;
                    value += amplitude * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                }
                image[(row, col)] = value.min(255.0);
            }
        }
        image
    }

    fn gpu_or_skip(preset: SiftPreset) -> Option<SiftGpuExtractor> {
        match SiftGpuExtractor::with_preset(preset) {
            Ok(extractor) => Some(extractor),
            Err(SiftGpuError::NoAdapter) => None,
            Err(error) => panic!("GPU extractor unavailable: {error}"),
        }
    }

    fn scratch_allocations(gpu: &SiftGpuExtractor) -> usize {
        match gpu.engine.scratch.lock() {
            Ok(scratch) => scratch.allocations(),
            Err(error) => panic!("GPU scratch lock poisoned: {error}"),
        }
    }

    fn nearest_cpu_keypoint(gpu: &ScaleSpaceKeypoint, cpu: &[ScaleSpaceKeypoint]) -> Option<usize> {
        let mut best = None;
        let mut best_distance = f32::INFINITY;
        for (index, keypoint) in cpu.iter().enumerate() {
            let dx = gpu.x - keypoint.x;
            let dy = gpu.y - keypoint.y;
            let distance = dx * dx + dy * dy;
            if distance < best_distance {
                best_distance = distance;
                best = Some(index);
            }
        }
        best
    }

    fn circular_angle(a: f32, b: f32) -> f32 {
        let mut difference = (a - b).abs();
        if difference > core::f32::consts::PI {
            difference = core::f32::consts::TAU - difference;
        }
        difference
    }

    fn nearest_cpu_orientation(
        gpu: &ScaleSpaceKeypoint,
        cpu: &[ScaleSpaceKeypoint],
    ) -> Option<usize> {
        let mut best = None;
        let mut best_difference = f32::INFINITY;
        for (index, keypoint) in cpu.iter().enumerate() {
            let dx = gpu.x - keypoint.x;
            let dy = gpu.y - keypoint.y;
            if dx * dx + dy * dy > 4.0 || (gpu.sigma - keypoint.sigma).abs() > 0.1 {
                continue;
            }
            let difference = circular_angle(gpu.orientation, keypoint.orientation);
            if difference < best_difference {
                best_difference = difference;
                best = Some(index);
            }
        }
        best
    }

    fn assert_preset_pipeline(preset: SiftPreset) {
        let Some(gpu) = gpu_or_skip(preset) else {
            return;
        };
        let image = synthetic_blobs(128);
        let gpu_features = match gpu.extract_f32(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed for {preset:?}: {error}"),
        };
        let cpu_features = match SiftExtractor::with_preset(preset).extract_f32(&image) {
            Ok(features) => features,
            Err(error) => panic!("CPU extraction failed for {preset:?}: {error}"),
        };

        assert!(
            !gpu_features.keypoints().is_empty(),
            "{preset:?}: the GPU found no keypoints"
        );
        assert_eq!(
            gpu_features.keypoints().len(),
            gpu_features.descriptors().len(),
            "{preset:?}: GPU keypoints and descriptors must stay aligned"
        );

        let mut geometry_matches = 0usize;
        let mut orientation_matches = 0usize;
        let mut close_descriptors = 0usize;
        for (gpu_index, gpu_keypoint) in gpu_features.keypoints().iter().enumerate() {
            if let Some(cpu_index) = nearest_cpu_keypoint(gpu_keypoint, cpu_features.keypoints()) {
                let cpu_keypoint = cpu_features.keypoints()[cpu_index];
                let dx = gpu_keypoint.x - cpu_keypoint.x;
                let dy = gpu_keypoint.y - cpu_keypoint.y;
                let distance = dx * dx + dy * dy;
                if distance <= 4.0 {
                    geometry_matches += 1;
                }
            }

            let Some(cpu_index) = nearest_cpu_orientation(gpu_keypoint, cpu_features.keypoints())
            else {
                continue;
            };
            let cpu_keypoint = cpu_features.keypoints()[cpu_index];
            let orientation_difference =
                circular_angle(gpu_keypoint.orientation, cpu_keypoint.orientation);
            if orientation_difference >= 0.05 {
                continue;
            }
            orientation_matches += 1;
            let gpu_descriptor = gpu_features.descriptors()[gpu_index].as_slice();
            let cpu_descriptor = cpu_features.descriptors()[cpu_index].as_slice();
            let distance = gpu_descriptor
                .iter()
                .zip(cpu_descriptor)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if distance < 0.05 {
                close_descriptors += 1;
            }
        }

        let gpu_count = gpu_features.keypoints().len();
        assert!(
            geometry_matches * 2 >= gpu_count,
            "{preset:?}: only {geometry_matches} of {gpu_count} GPU keypoints matched CPU geometry"
        );
        assert!(
            orientation_matches * 2 >= gpu_count,
            "{preset:?}: only {orientation_matches} of {gpu_count} GPU orientations matched the CPU"
        );
        assert!(
            close_descriptors * 2 >= gpu_count,
            "{preset:?}: only {close_descriptors} of {gpu_count} descriptors matched the CPU"
        );
    }

    #[test]
    fn packed_upload_accepts_only_exact_bytes() {
        let mut words = vec![0u32; 2];
        assert!(pack_u8_image(&[0.0, 1.0, 254.0, 255.0, 7.0], &mut words));
        assert_eq!(words, [u32::from_le_bytes([0, 1, 254, 255]), 7]);
        for value in [-0.0, 0.5, 255.5, 256.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut words = vec![0u32; 1];
            assert!(
                !pack_u8_image(&[3.0, value], &mut words),
                "{value:?} must keep the f32 upload"
            );
        }
    }

    /// `synthetic_blobs` rounded to whole grey levels, as a decoded 8-bit
    /// image would be, so the extractor takes the packed upload.
    fn integer_blobs(size: usize) -> GrayImage {
        synthetic_blobs(size).map(f32::round)
    }

    /// A 6x6 grid of blobs of varied size and contrast, rounded to whole
    /// grey levels. Blobs share rows, so candidates share the counting-sort
    /// buckets of `order.wgsl` and their column order matters.
    fn integer_blob_grid() -> GrayImage {
        let size = 192;
        let mut image = GrayImage::from_element(size, size, 30.0);
        for gy in 0..6 {
            for gx in 0..6 {
                let (x, y) = (16.0 + 32.0 * gx as f32, 16.0 + 32.0 * gy as f32);
                let sigma = 2.0 + ((gx + 2 * gy) % 4) as f32;
                let amplitude = if (gx + gy) % 2 == 0 { 160.0 } else { -25.0 };
                for row in 0..size {
                    for col in 0..size {
                        let (dx, dy) = (col as f32 - x, row as f32 - y);
                        image[(row, col)] +=
                            amplitude * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                    }
                }
            }
        }
        image.map(|value| value.round().clamp(0.0, 255.0))
    }

    #[test]
    fn gpu_integer_image_keeps_cpu_detector_order() {
        let Some(gpu) = gpu_or_skip(SiftPreset::High) else {
            return;
        };
        let image = integer_blob_grid();
        let gpu_features = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        let cpu_features = match SiftExtractor::with_preset(SiftPreset::High).extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("CPU extraction failed: {error}"),
        };
        // Detector order is octave, then candidate (scale, row, column) order
        // after deduplication; every orientation of one keypoint follows it.
        // Compare the sequences of distinct keypoint geometries, which pins
        // that order. Orientations are left to the parity tests: these
        // isotropic blobs have nearly flat orientation histograms.
        let geometries = |keypoints: &[ScaleSpaceKeypoint]| {
            let mut sequence: Vec<(f32, f32, f32, usize)> = Vec::new();
            for keypoint in keypoints {
                let geometry = (keypoint.x, keypoint.y, keypoint.sigma, keypoint.octave);
                if sequence.last() != Some(&geometry) {
                    sequence.push(geometry);
                }
            }
            sequence
        };
        let gpu_sequence = geometries(gpu_features.keypoints());
        let cpu_sequence = geometries(cpu_features.keypoints());
        assert!(!gpu_sequence.is_empty(), "the GPU found no keypoints");
        assert_eq!(gpu_sequence.len(), cpu_sequence.len());
        for (gpu, cpu) in gpu_sequence.iter().zip(&cpu_sequence) {
            assert_eq!(gpu.3, cpu.3, "octave order differs: {gpu:?} vs {cpu:?}");
            assert!(
                (gpu.0 - cpu.0).abs() < 1e-2
                    && (gpu.1 - cpu.1).abs() < 1e-2
                    && (gpu.2 - cpu.2).abs() < 1e-2,
                "detector order differs: GPU {gpu:?} vs CPU {cpu:?}"
            );
        }
    }

    #[test]
    fn gpu_packed_and_f32_uploads_agree() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let mut packed = integer_blobs(128);
        packed[(0, 0)] = 0.0;
        // `-0.0` is not an exact byte, so this copy takes the f32 upload; it
        // normalizes to the same scale-space values.
        let mut float = packed.clone();
        float[(0, 0)] = -0.0;
        let extract = |image: &GrayImage| match gpu.extract_u8(image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        let (packed, float) = (extract(&packed), extract(&float));
        assert!(!packed.keypoints().is_empty());
        assert_eq!(packed.keypoints(), float.keypoints());
        assert_eq!(packed.descriptors(), float.descriptors());
    }

    #[test]
    fn gpu_validation_error_is_returned_and_recovers() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let image = integer_blobs(128);
        let expected = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        // MAP_READ only combines with COPY_DST, so this buffer is a
        // validation error that wgpu reports through its error handler.
        let _invalid = gpu.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(matches!(
            gpu.extract_u8(&image),
            Err(SiftGpuError::Validation(_))
        ));
        let recovered = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("the extractor did not recover: {error}"),
        };
        assert_eq!(recovered.keypoints(), expected.keypoints());
        assert_eq!(recovered.descriptors(), expected.descriptors());
    }

    #[test]
    fn gpu_lost_device_fails_every_later_call() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let image = integer_blobs(128);
        gpu.engine.device.destroy();
        for _ in 0..2 {
            assert!(matches!(
                gpu.extract_u8(&image),
                Err(SiftGpuError::DeviceLost(_))
            ));
        }
        assert!(matches!(
            gpu.detect_keypoints(&image),
            Err(SiftGpuError::DeviceLost(_))
        ));
    }

    #[test]
    fn gpu_poisoned_scratch_is_rebuilt() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let image = integer_blobs(128);
        let expected = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        let poisoned = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _scratch = gpu.engine.scratch.lock();
                    panic!("poison the scratch lock");
                })
                .join()
        });
        assert!(poisoned.is_err() && gpu.engine.scratch.is_poisoned());
        let features = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("a poisoned scratch lock must be recovered: {error}"),
        };
        assert!(!gpu.engine.scratch.is_poisoned());
        assert_eq!(features.keypoints(), expected.keypoints());
        assert_eq!(features.descriptors(), expected.descriptors());
    }

    #[test]
    fn gpu_normal_preset_matches_cpu() {
        assert_preset_pipeline(SiftPreset::Normal);
    }

    #[test]
    fn gpu_ultra_preset_matches_cpu() {
        assert_preset_pipeline(SiftPreset::Ultra);
    }

    #[test]
    fn gpu_keypoints_budget_and_storage_agree() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let image = synthetic_blobs(128);
        let budgeted = SiftGpuExtractor::new(
            SiftExtractor::with_preset(SiftPreset::Normal)
                .with_feature_budget(FeatureBudgetPolicy::strongest(5)),
        );
        let budgeted = match budgeted {
            Ok(extractor) => extractor,
            Err(error) => panic!("GPU extractor unavailable: {error}"),
        };
        let features = match budgeted.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("budgeted GPU extraction failed: {error}"),
        };
        assert!(!features.keypoints().is_empty());
        assert!(features.keypoints().len() <= 5);
        assert_eq!(features.keypoints().len(), features.descriptors().len());

        let keypoints = match gpu.detect_keypoints(&image) {
            Ok(keypoints) => keypoints,
            Err(error) => panic!("GPU keypoint detection failed: {error}"),
        };
        let described = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        assert_eq!(keypoints.len(), described.keypoints().len());
        for (detected, described) in keypoints.iter().zip(described.keypoints()) {
            assert_eq!(detected, described);
        }
    }

    fn assert_packed_u8_matches_typed(config: SiftExtractor) {
        let gpu = match SiftGpuExtractor::new(config) {
            Ok(extractor) => extractor,
            Err(error) => panic!("GPU extractor unavailable: {error}"),
        };
        let image = synthetic_blobs(192);
        let packed = match gpu.extract_u8(&image) {
            Ok(features) => features,
            Err(error) => panic!("packed GPU extraction failed: {error}"),
        };
        let typed = match gpu.extract_typed::<U8Descriptor>(&image) {
            Ok(features) => features,
            Err(error) => panic!("typed GPU extraction failed: {error}"),
        };
        assert_eq!(packed.keypoints(), typed.keypoints());
        assert_eq!(packed.descriptors().len(), typed.descriptors().len());
        for (packed, typed) in packed.descriptors().iter().zip(typed.descriptors()) {
            assert_eq!(packed.as_slice(), typed.as_slice());
            assert_eq!(packed.scale(), typed.scale());
        }
    }

    #[test]
    fn gpu_packed_u8_matches_typed_storage() {
        let Some(_) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        assert_packed_u8_matches_typed(SiftExtractor::with_preset(SiftPreset::Normal));
        assert_packed_u8_matches_typed(
            SiftExtractor::with_preset(SiftPreset::Normal).with_root_sift(false),
        );
    }

    #[test]
    fn gpu_custom_scale_count_runs() {
        let Some(_) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let image = synthetic_blobs(128);
        let config = SiftExtractor::with_preset(SiftPreset::Normal)
            .with_num_scales(5)
            .with_num_octaves(3);
        let gpu = match SiftGpuExtractor::new(config.clone()) {
            Ok(extractor) => extractor,
            Err(error) => panic!("GPU extractor unavailable: {error}"),
        };
        let features = match gpu.extract_f32(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        let cpu = match config.extract_f32(&image) {
            Ok(features) => features,
            Err(error) => panic!("CPU extraction failed: {error}"),
        };
        assert!(!features.keypoints().is_empty());
        assert!(!cpu.keypoints().is_empty());
        assert_eq!(features.keypoints().len(), features.descriptors().len());
    }

    #[test]
    fn gpu_classic_normalization_runs() {
        let Some(_) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let image = synthetic_blobs(128);
        let config = SiftExtractor::with_preset(SiftPreset::Normal).with_root_sift(false);
        let gpu = match SiftGpuExtractor::new(config) {
            Ok(extractor) => extractor,
            Err(error) => panic!("GPU extractor unavailable: {error}"),
        };
        let features = match gpu.extract_f32(&image) {
            Ok(features) => features,
            Err(error) => panic!("GPU extraction failed: {error}"),
        };
        assert!(!features.keypoints().is_empty());
        assert_eq!(features.keypoints().len(), features.descriptors().len());
        for descriptor in features.descriptors() {
            assert!(descriptor.as_slice().iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn gpu_scratch_is_reused_across_images_and_sizes() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let small = synthetic_blobs(128);
        let large = synthetic_blobs(256);

        let first = match gpu.extract_u8(&small) {
            Ok(features) => features,
            Err(error) => panic!("first GPU extraction failed: {error}"),
        };
        let after_first = scratch_allocations(&gpu);
        assert!(
            after_first > 0,
            "the first extraction must allocate scratch"
        );
        let resident_first = gpu.resident_bytes();
        // The scale space alone holds 6 Gaussian and 5 DoG planes of the
        // 128 x 128 base octave.
        assert!(resident_first >= 11 * 128 * 128 * 4);

        let repeated = match gpu.extract_u8(&small) {
            Ok(features) => features,
            Err(error) => panic!("repeated GPU extraction failed: {error}"),
        };
        assert_eq!(
            scratch_allocations(&gpu),
            after_first,
            "a repeated image size must reuse every scratch buffer"
        );
        assert_eq!(first.keypoints(), repeated.keypoints());
        assert_eq!(first.descriptors(), repeated.descriptors());

        let larger = match gpu.extract_u8(&large) {
            Ok(features) => features,
            Err(error) => panic!("larger GPU extraction failed: {error}"),
        };
        assert!(!larger.keypoints().is_empty());
        let after_large = scratch_allocations(&gpu);
        assert!(
            after_large > after_first,
            "a larger image must grow at least one scratch buffer"
        );
        let resident_large = gpu.resident_bytes();
        assert!(resident_large > resident_first);

        let _ = match gpu.extract_u8(&small) {
            Ok(features) => features,
            Err(error) => panic!("small image after large failed: {error}"),
        };
        assert_eq!(
            scratch_allocations(&gpu),
            after_large,
            "a smaller image must reuse the larger buffers"
        );

        let _ = match gpu.extract_u8(&large) {
            Ok(features) => features,
            Err(error) => panic!("repeated large image failed: {error}"),
        };
        assert_eq!(scratch_allocations(&gpu), after_large);

        let flat = GrayImage::zeros(256, 256);
        let flat_features = match gpu.extract_u8(&flat) {
            Ok(features) => features,
            Err(error) => panic!("featureless GPU extraction failed: {error}"),
        };
        assert!(flat_features.keypoints().is_empty());
        assert!(flat_features.descriptors().is_empty());
        assert_eq!(
            scratch_allocations(&gpu),
            after_large,
            "a featureless image must not allocate result buffers"
        );
    }

    #[test]
    fn gpu_small_and_featureless_images_return_empty() {
        let Some(gpu) = gpu_or_skip(SiftPreset::Normal) else {
            return;
        };
        let small = GrayImage::zeros(4, 4);
        let small_features = match gpu.extract_u8(&small) {
            Ok(features) => features,
            Err(error) => panic!("small-image GPU extraction failed: {error}"),
        };
        assert!(small_features.keypoints().is_empty());
        assert!(small_features.descriptors().is_empty());

        let flat = GrayImage::zeros(64, 64);
        let flat_features = match gpu.extract_u8(&flat) {
            Ok(features) => features,
            Err(error) => panic!("featureless GPU extraction failed: {error}"),
        };
        assert!(flat_features.keypoints().is_empty());
        assert!(flat_features.descriptors().is_empty());
    }
}
