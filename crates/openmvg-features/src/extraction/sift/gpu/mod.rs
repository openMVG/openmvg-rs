// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional wgpu backend for the classic SIFT extractor.
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
//! [`GrayImage`]: crate::extraction::GrayImage

use std::collections::BTreeSet;

use crate::extraction::feature_budget::{FeatureBudgetKeypoint, apply_feature_budget};
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
const CANDIDATE_BYTES: usize = 24;
const REFINED_BYTES: usize = 56;
const REFINE_WORKGROUPS: u32 = 256;

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
}

/// GPU implementation of [`SiftExtractor`].
///
/// Create it once and reuse it for images of different sizes. The extractor
/// owns a wgpu device and reuses its compute pipelines across calls; storage
/// buffers are allocated per extraction.
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
    /// Create a GPU extractor for `config`.
    ///
    /// # Errors
    ///
    /// Returns [`SiftGpuError::InvalidConfig`] if [`SiftExtractor::validate`]
    /// rejects `config`, or [`SiftGpuError::NoAdapter`] /
    /// [`SiftGpuError::Device`] if no GPU device can be created.
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

    /// The detector configuration this extractor uses.
    #[must_use]
    pub fn config(&self) -> &SiftExtractor {
        &self.config
    }

    /// Detect oriented keypoint geometry without constructing descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device or a result buffer fails.
    pub fn detect_keypoints(
        &self,
        image: &GrayImage,
    ) -> Result<Vec<ScaleSpaceKeypoint>, SiftGpuError> {
        let detection = self.detect_features(image)?;
        Ok(detection
            .features
            .into_iter()
            .map(Feature::geometry)
            .collect())
    }

    /// Detect and describe with f32 descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device or a result buffer fails.
    pub fn extract_f32(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<F32Descriptor>, SiftGpuError> {
        self.extract_typed(image)
    }

    /// Detect and describe with u8 descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device or a result buffer fails.
    pub fn extract_u8(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<U8Descriptor>, SiftGpuError> {
        self.extract_typed(image)
    }

    /// Detect and describe with the descriptor storage `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device or a result buffer fails.
    pub fn extract_typed<D: DescriptorStorage>(
        &self,
        image: &GrayImage,
    ) -> Result<ScaleSpaceFeatures<D>, SiftGpuError> {
        let detection = self.detect_features(image)?;
        let descriptors = self.describe::<D>(&detection)?;
        let mut features = ScaleSpaceFeatures::default();
        features.keypoints.reserve(detection.features.len());
        features.descriptors.reserve(detection.features.len());
        for (feature, descriptor) in detection.features.into_iter().zip(descriptors) {
            if let Some(descriptor) = descriptor {
                features.keypoints.push(feature.geometry());
                features.descriptors.push(descriptor);
            }
        }
        Ok(features)
    }

    fn detect_features(&self, image: &GrayImage) -> Result<Detection, SiftGpuError> {
        self.engine
            .detect_features(&self.config, image)
            .map_err(Into::into)
    }

    fn describe<D: DescriptorStorage>(
        &self,
        detection: &Detection,
    ) -> Result<Vec<Option<D>>, SiftGpuError> {
        self.engine
            .describe::<D>(&self.config, detection)
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
}

struct OrientationReadback {
    key_count: usize,
    count_bytes: usize,
    orientation_bytes: usize,
    counts: wgpu::Buffer,
    orientations: wgpu::Buffer,
    indices: Vec<usize>,
}

struct DescriptorReadback {
    key_count: usize,
    valid_bytes: usize,
    value_bytes: usize,
    valid: wgpu::Buffer,
    values: wgpu::Buffer,
    feature_indices: Vec<usize>,
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
}

struct Layouts {
    init: wgpu::BindGroupLayout,
    blur: wgpu::BindGroupLayout,
    decimate: wgpu::BindGroupLayout,
    dog: wgpu::BindGroupLayout,
    extrema: wgpu::BindGroupLayout,
    refine: wgpu::BindGroupLayout,
    orientation: wgpu::BindGroupLayout,
    describe: wgpu::BindGroupLayout,
}

struct Pipelines {
    init: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    decimate: wgpu::ComputePipeline,
    dog: wgpu::ComputePipeline,
    extrema: wgpu::ComputePipeline,
    refine: wgpu::ComputePipeline,
    orientation: wgpu::ComputePipeline,
    describe: wgpu::ComputePipeline,
}

impl Engine {
    fn new() -> Result<Self, SiftGpuError> {
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
        let layouts = Layouts::new(&device);
        let pipelines = Pipelines::new(&device, &layouts);
        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipelines,
            layouts,
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

    fn detect_features(
        &self,
        config: &SiftExtractor,
        image: &GrayImage,
    ) -> Result<Detection, RunError> {
        let height = image.nrows();
        let width = image.ncols();
        if height < MIN_INPUT_DIMENSION || width < MIN_INPUT_DIMENSION {
            return Ok(Detection {
                features: Vec::new(),
                octaves: Vec::new(),
            });
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
            let gauss = storage_buffer(&self.device, "omvg-sift-gauss", gauss_bytes as u64);
            let dog = storage_buffer(&self.device, "omvg-sift-dog", dog_bytes as u64);
            let temp_a = storage_buffer(&self.device, "omvg-sift-temp-a", temp_bytes as u64);
            let temp_b = storage_buffer(&self.device, "omvg-sift-temp-b", temp_bytes as u64);

            let voxels = (rows.saturating_sub(2))
                .saturating_mul(cols.saturating_sub(2))
                .saturating_mul(dogs);
            let capacity = (voxels / CANDIDATE_DIVISOR).max(MIN_CANDIDATES);
            let candidate_bytes = capacity
                .checked_mul(CANDIDATE_BYTES)
                .ok_or(RunError::Capacity { what: "candidate" })?;
            let refined_bytes = capacity
                .checked_mul(REFINED_BYTES)
                .ok_or(RunError::Capacity {
                    what: "refined keypoint",
                })?;
            self.ensure_storage_capacity(candidate_bytes)?;
            self.ensure_storage_capacity(refined_bytes)?;
            let candidate_buffer =
                storage_buffer(&self.device, "omvg-sift-candidates", candidate_bytes as u64);
            let candidate_count = storage_buffer(&self.device, "omvg-sift-candidate-count", 16);
            let refined_buffer =
                storage_buffer(&self.device, "omvg-sift-refined", refined_bytes as u64);
            let refined_count = storage_buffer(&self.device, "omvg-sift-refined-count", 16);

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

        let source_bytes = height
            .checked_mul(width)
            .and_then(|pixels| pixels.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(RunError::Capacity { what: "input" })?;
        self.ensure_storage_capacity(source_bytes)?;
        let source = storage_buffer(&self.device, "omvg-sift-input", source_bytes as u64);
        self.queue
            .write_buffer(&source, 0, bytemuck::cast_slice(image.as_slice()));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("omvg-sift-scale-space"),
            });
        let mut keep_buffers: Vec<wgpu::Buffer> = Vec::new();
        let mut keep_bind_groups: Vec<wgpu::BindGroup> = Vec::new();

        // Initial level of the first octave.
        self.encode_init(
            &mut encoder,
            &source,
            &temp_as[0],
            height,
            width,
            base_rows,
            base_cols,
            first_octave == -1,
            &mut keep_buffers,
            &mut keep_bind_groups,
        );
        if sigma_extra > 0.01 {
            self.encode_blur_pair(
                &mut encoder,
                &temp_as[0],
                &temp_bs[0],
                &temp_as[0],
                &octaves[0].gauss,
                0,
                base_rows,
                base_cols,
                0,
                sigma_extra,
                &mut keep_buffers,
                &mut keep_bind_groups,
            );
        } else {
            encoder.copy_buffer_to_buffer(
                &temp_as[0],
                0,
                &octaves[0].gauss,
                0,
                (base_rows * base_cols * core::mem::size_of::<f32>()) as u64,
            );
        }

        // Remaining Gaussian levels of every octave.
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            let plane = rows * cols;
            if octave > 0 {
                let previous = &octaves[octave - 1];
                let source_offset = config.num_scales * previous.rows * previous.cols;
                self.encode_decimate(
                    &mut encoder,
                    &previous.gauss,
                    &octaves[octave].gauss,
                    previous.rows,
                    previous.cols,
                    rows,
                    cols,
                    source_offset,
                    &mut keep_buffers,
                    &mut keep_bind_groups,
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
                self.encode_blur_pair(
                    &mut encoder,
                    &octaves[octave].gauss,
                    &temp_as[octave],
                    &temp_bs[octave],
                    &octaves[octave].gauss,
                    (level - 1) * plane,
                    rows,
                    cols,
                    level * plane,
                    extra,
                    &mut keep_buffers,
                    &mut keep_bind_groups,
                );
            }
        }

        // DoG planes.
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            let plane = rows * cols;
            for level in 0..dogs {
                self.encode_dog(
                    &mut encoder,
                    &octaves[octave].gauss,
                    &dog_buffers[octave],
                    rows,
                    cols,
                    level * plane,
                    (level + 1) * plane,
                    level * plane,
                    &mut keep_buffers,
                    &mut keep_bind_groups,
                );
            }
        }

        // Clear detector counters, then run extrema and refinement per octave.
        for count in &candidate_counts {
            encoder.clear_buffer(count, 0, None);
        }
        for count in &refined_counts {
            encoder.clear_buffer(count, 0, None);
        }
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            self.encode_extrema(
                &mut encoder,
                &dog_buffers[octave],
                &candidate_buffers[octave],
                &candidate_counts[octave],
                rows,
                cols,
                levels,
                config.peak_threshold / config.num_scales as f32 * 0.8,
                octave,
                capacities[octave],
                &mut keep_buffers,
                &mut keep_bind_groups,
            );
        }
        for octave in 0..num_octaves {
            let rows = octaves[octave].rows;
            let cols = octaves[octave].cols;
            self.encode_refine(
                &mut encoder,
                &dog_buffers[octave],
                &candidate_buffers[octave],
                &candidate_counts[octave],
                &refined_buffers[octave],
                &refined_counts[octave],
                rows,
                cols,
                levels,
                config.num_scales,
                octaves[octave].delta,
                sigma_min,
                delta_min,
                config.edge_threshold,
                config.peak_threshold / config.num_scales as f32,
                octave,
                capacities[octave],
                &mut keep_buffers,
                &mut keep_bind_groups,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        drop(keep_bind_groups);
        drop(keep_buffers);

        let mut count_ranges = Vec::with_capacity(num_octaves * 2);
        for octave in 0..num_octaves {
            count_ranges.push((&candidate_counts[octave], core::mem::size_of::<u32>()));
            count_ranges.push((&refined_counts[octave], core::mem::size_of::<u32>()));
        }
        let (count_data, count_offsets) = read_buffers(&self.device, &self.queue, &count_ranges)?;
        let mut refined_totals = Vec::with_capacity(num_octaves);
        for octave in 0..num_octaves {
            let candidate_count = read_u32(&count_data, count_offsets[octave * 2]) as usize;
            let refined_count = read_u32(&count_data, count_offsets[octave * 2 + 1]) as usize;
            if candidate_count > capacities[octave] || refined_count > capacities[octave] {
                return Err(RunError::Capacity { what: "candidate" });
            }
            refined_totals.push(refined_count);
        }

        let mut refined_ranges = Vec::with_capacity(num_octaves);
        for octave in 0..num_octaves {
            refined_ranges.push((
                &refined_buffers[octave],
                refined_totals[octave] * REFINED_BYTES,
            ));
        }
        let (refined_data, refined_offsets) =
            read_buffers(&self.device, &self.queue, &refined_ranges)?;
        let mut refined = Vec::new();
        for octave in 0..num_octaves {
            let refined_count = refined_totals[octave];
            let start = refined_offsets[octave];
            let bytes = &refined_data[start..start + refined_count * REFINED_BYTES];
            let mut records = Vec::with_capacity(refined_count);
            for chunk in bytes.as_chunks::<REFINED_BYTES>().0 {
                records.push(bytemuck::pod_read_unaligned::<GpuRefined>(chunk));
            }
            records
                .sort_unstable_by_key(|record| (record.orig_level, record.orig_j, record.orig_i));
            let mut seen = BTreeSet::new();
            for record in records {
                if !seen.insert((record.level, record.j, record.i)) {
                    continue;
                }
                if record.accepted == 0 {
                    continue;
                }
                refined.push(Feature {
                    x: record.x,
                    y: record.y,
                    sigma: record.sigma,
                    orientation: 0.0,
                    peak: record.value,
                    octave: record.octave as usize,
                    scale_index: record.level as usize,
                });
            }
        }

        // Orientation is computed per octave, preserving the CPU's octave and
        // bin order. The host applies the feature budget after all octaves.
        let mut orientation_readbacks = Vec::new();
        for (octave, octave_data) in octaves.iter().enumerate() {
            let mut keys = Vec::new();
            let mut indices = Vec::new();
            for (index, feature) in refined.iter().enumerate() {
                if feature.octave == octave {
                    keys.push(GpuOriKey {
                        x: feature.x / octave_data.delta,
                        y: feature.y / octave_data.delta,
                        sigma: feature.sigma / octave_data.delta,
                        level: feature.scale_index as u32,
                    });
                    indices.push(index);
                }
            }
            if keys.is_empty() {
                continue;
            }
            let key_count = keys.len();
            let key_bytes = key_count
                .checked_mul(core::mem::size_of::<GpuOriKey>())
                .ok_or(RunError::Capacity {
                    what: "orientation keys",
                })?;
            let count_bytes =
                key_count
                    .checked_mul(core::mem::size_of::<u32>())
                    .ok_or(RunError::Capacity {
                        what: "orientation counts",
                    })?;
            let orientation_bytes = key_count
                .checked_mul(MAX_ORIENTATIONS)
                .and_then(|count| count.checked_mul(core::mem::size_of::<f32>()))
                .ok_or(RunError::Capacity {
                    what: "orientations",
                })?;
            self.ensure_storage_capacity(key_bytes)?;
            self.ensure_storage_capacity(count_bytes)?;
            self.ensure_storage_capacity(orientation_bytes)?;
            let key_buffer =
                storage_buffer(&self.device, "omvg-sift-orientation-keys", key_bytes as u64);
            self.queue
                .write_buffer(&key_buffer, 0, bytemuck::cast_slice(&keys));
            let counts = storage_buffer(
                &self.device,
                "omvg-sift-orientation-counts",
                count_bytes as u64,
            );
            let orientations = storage_buffer(
                &self.device,
                "omvg-sift-orientations",
                orientation_bytes as u64,
            );
            let uniform = uniform_buffer(
                &self.device,
                &self.queue,
                "omvg-sift-orientation-params",
                &OriParams {
                    rows_cols: [octave_data.rows as u32, octave_data.cols as u32],
                    level: 0,
                    key_count: key_count as u32,
                    gauss_off: 0,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                },
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("omvg-sift-orientation"),
                layout: &self.layouts.orientation,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: octave_data.gauss.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: key_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: orientations.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("omvg-sift-orientation"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("omvg-sift-orientation"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.orientation);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(key_count.div_ceil(32).max(1) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            orientation_readbacks.push(OrientationReadback {
                key_count,
                count_bytes,
                orientation_bytes,
                counts,
                orientations,
                indices,
            });
        }

        let mut orientation_ranges = Vec::with_capacity(orientation_readbacks.len() * 2);
        for readback in &orientation_readbacks {
            orientation_ranges.push((&readback.counts, readback.count_bytes));
            orientation_ranges.push((&readback.orientations, readback.orientation_bytes));
        }
        let (orientation_data, orientation_offsets) =
            read_buffers(&self.device, &self.queue, &orientation_ranges)?;
        let mut oriented = Vec::new();
        for (readback, offsets) in orientation_readbacks
            .iter()
            .zip(orientation_offsets.chunks_exact(2))
        {
            let count_offset = offsets[0];
            let orientation_offset = offsets[1];
            for local in 0..readback.key_count {
                let count = read_u32(&orientation_data, count_offset + local * 4) as usize;
                for slot in 0..count.min(MAX_ORIENTATIONS) {
                    let offset = orientation_offset + (local * MAX_ORIENTATIONS + slot) * 4;
                    let mut feature = refined[readback.indices[local]];
                    feature.orientation = read_f32(&orientation_data, offset);
                    oriented.push(feature);
                }
            }
        }

        let mut features = oriented;
        apply_feature_budget(&mut features, width, height, &config.feature_budget);
        Ok(Detection { features, octaves })
    }

    fn describe<D: DescriptorStorage>(
        &self,
        config: &SiftExtractor,
        detection: &Detection,
    ) -> Result<Vec<Option<D>>, RunError> {
        let mut descriptors: Vec<Option<D>> = (0..detection.features.len()).map(|_| None).collect();
        let mut descriptor_readbacks = Vec::new();
        for (octave_index, octave) in detection.octaves.iter().enumerate() {
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
            let value_bytes = key_count
                .checked_mul(SIFT_DESCR_SIZE)
                .and_then(|count| count.checked_mul(core::mem::size_of::<f32>()))
                .ok_or(RunError::Capacity {
                    what: "descriptors",
                })?;
            let valid_bytes =
                key_count
                    .checked_mul(core::mem::size_of::<u32>())
                    .ok_or(RunError::Capacity {
                        what: "descriptor valid flags",
                    })?;
            self.ensure_storage_capacity(key_bytes)?;
            self.ensure_storage_capacity(value_bytes)?;
            self.ensure_storage_capacity(valid_bytes)?;
            let key_buffer =
                storage_buffer(&self.device, "omvg-sift-descriptor-keys", key_bytes as u64);
            self.queue
                .write_buffer(&key_buffer, 0, bytemuck::cast_slice(&keys));
            let values = storage_buffer(&self.device, "omvg-sift-descriptors", value_bytes as u64);
            let valid = storage_buffer(
                &self.device,
                "omvg-sift-descriptor-valid",
                valid_bytes as u64,
            );
            let uniform = uniform_buffer(
                &self.device,
                &self.queue,
                "omvg-sift-descriptor-params",
                &DescParams {
                    rows_cols: [octave.rows as u32, octave.cols as u32],
                    level: 0,
                    root_sift: u32::from(config.root_sift),
                    key_count: key_count as u32,
                    out_off: 0,
                    gauss_off: 0,
                    pad: 0,
                },
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("omvg-sift-descriptor"),
                layout: &self.layouts.describe,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: octave.gauss.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: key_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: valid.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("omvg-sift-descriptor"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("omvg-sift-descriptor"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.describe);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(key_count as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            descriptor_readbacks.push(DescriptorReadback {
                key_count,
                valid_bytes,
                value_bytes,
                valid,
                values,
                feature_indices,
            });
        }

        let mut descriptor_ranges = Vec::with_capacity(descriptor_readbacks.len() * 2);
        for readback in &descriptor_readbacks {
            descriptor_ranges.push((&readback.valid, readback.valid_bytes));
            descriptor_ranges.push((&readback.values, readback.value_bytes));
        }
        let (descriptor_data, descriptor_offsets) =
            read_buffers(&self.device, &self.queue, &descriptor_ranges)?;
        for (readback, offsets) in descriptor_readbacks
            .iter()
            .zip(descriptor_offsets.chunks_exact(2))
        {
            let valid_offset = offsets[0];
            let values_offset = offsets[1];
            for local in 0..readback.key_count {
                if read_u32(&descriptor_data, valid_offset + local * 4) == 0 {
                    continue;
                }
                let base = values_offset + local * SIFT_DESCR_SIZE * 4;
                let mut values = [0.0f32; SIFT_DESCR_SIZE];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = read_f32(&descriptor_data, base + index * 4);
                }
                if let Ok(descriptor) = D::try_from_finalized(&values) {
                    descriptors[readback.feature_indices[local]] = Some(descriptor);
                }
            }
        }
        Ok(descriptors)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_init(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        source_rows: usize,
        source_cols: usize,
        destination_rows: usize,
        destination_cols: usize,
        upscale: bool,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let uniform = uniform_buffer(
            &self.device,
            &self.queue,
            "omvg-sift-init",
            &InitParams {
                src_rows_cols: [source_rows as u32, source_cols as u32],
                dst_rows_cols: [destination_rows as u32, destination_cols as u32],
                src_off: 0,
                dst_off: 0,
                upscale: u32::from(upscale),
                pad: 0,
            },
        );
        keep_buffers.push(uniform.clone());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("omvg-sift-init"),
            layout: &self.layouts.init,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
            ],
        });
        keep_bind_groups.push(bind_group);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("omvg-sift-init"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.init);
        pass.set_bind_group(0, &keep_bind_groups[keep_bind_groups.len() - 1], &[]);
        pass.dispatch_workgroups(
            (destination_rows as u32).div_ceil(16),
            (destination_cols as u32).div_ceil(16),
            1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_blur_pair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        temp_a: &wgpu::Buffer,
        temp_b: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        source_offset: usize,
        rows: usize,
        cols: usize,
        destination_offset: usize,
        sigma: f32,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let kernel = gaussian_weights(sigma);
        let radius = kernel.len().saturating_sub(1);
        let weight_bytes = (kernel.len() * core::mem::size_of::<f32>()) as u64;
        let weight_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("omvg-sift-blur-weights"),
            size: weight_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&weight_buffer, 0, bytemuck::cast_slice(&kernel));
        keep_buffers.push(weight_buffer.clone());

        self.encode_blur(
            encoder,
            source,
            temp_a,
            &weight_buffer,
            rows,
            cols,
            source_offset,
            0,
            radius,
            true,
            keep_buffers,
            keep_bind_groups,
        );
        self.encode_blur(
            encoder,
            temp_a,
            temp_b,
            &weight_buffer,
            rows,
            cols,
            0,
            0,
            radius,
            false,
            keep_buffers,
            keep_bind_groups,
        );
        encoder.copy_buffer_to_buffer(
            temp_b,
            0,
            destination,
            (destination_offset * core::mem::size_of::<f32>()) as u64,
            (rows * cols * core::mem::size_of::<f32>()) as u64,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        source_offset: usize,
        destination_offset: usize,
        radius: usize,
        vertical: bool,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let uniform = uniform_buffer(
            &self.device,
            &self.queue,
            "omvg-sift-blur",
            &BlurParams {
                rows_cols: [rows as u32, cols as u32],
                src_off: source_offset as u32,
                dst_off: destination_offset as u32,
                radius: radius as u32,
                wt_off: 0,
                vertical: u32::from(vertical),
                pad: 0,
            },
        );
        keep_buffers.push(uniform.clone());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("omvg-sift-blur"),
            layout: &self.layouts.blur,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weights.as_entire_binding(),
                },
            ],
        });
        keep_bind_groups.push(bind_group);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("omvg-sift-blur"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.blur);
        pass.set_bind_group(0, &keep_bind_groups[keep_bind_groups.len() - 1], &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(16), (cols as u32).div_ceil(16), 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_decimate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        source_rows: usize,
        source_cols: usize,
        destination_rows: usize,
        destination_cols: usize,
        source_offset: usize,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let uniform = uniform_buffer(
            &self.device,
            &self.queue,
            "omvg-sift-decimate",
            &DecimateParams {
                src_rows_cols: [source_rows as u32, source_cols as u32],
                dst_rows_cols: [destination_rows as u32, destination_cols as u32],
                src_off: source_offset as u32,
                dst_off: 0,
                pad0: 0,
                pad1: 0,
            },
        );
        keep_buffers.push(uniform.clone());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("omvg-sift-decimate"),
            layout: &self.layouts.decimate,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
            ],
        });
        keep_bind_groups.push(bind_group);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("omvg-sift-decimate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.decimate);
        pass.set_bind_group(0, &keep_bind_groups[keep_bind_groups.len() - 1], &[]);
        pass.dispatch_workgroups(
            (destination_rows as u32).div_ceil(16),
            (destination_cols as u32).div_ceil(16),
            1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dog(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        source_offset: usize,
        source_next_offset: usize,
        destination_offset: usize,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let uniform = uniform_buffer(
            &self.device,
            &self.queue,
            "omvg-sift-dog",
            &DogParams {
                rows_cols: [rows as u32, cols as u32],
                src_off: source_offset as u32,
                src_next_off: source_next_offset as u32,
                dst_off: destination_offset as u32,
                pad0: 0,
            },
        );
        keep_buffers.push(uniform.clone());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("omvg-sift-dog"),
            layout: &self.layouts.dog,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
            ],
        });
        keep_bind_groups.push(bind_group);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("omvg-sift-dog"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.dog);
        pass.set_bind_group(0, &keep_bind_groups[keep_bind_groups.len() - 1], &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(16), (cols as u32).div_ceil(16), 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_extrema(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dog: &wgpu::Buffer,
        candidates: &wgpu::Buffer,
        count: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        levels: usize,
        threshold: f32,
        octave: usize,
        capacity: usize,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let uniform = uniform_buffer(
            &self.device,
            &self.queue,
            "omvg-sift-extrema",
            &ExtremaParams {
                rows_cols: [rows as u32, cols as u32],
                levels: levels as u32,
                threshold,
                octave: octave as u32,
                cap: capacity as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        keep_buffers.push(uniform.clone());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("omvg-sift-extrema"),
            layout: &self.layouts.extrema,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dog.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: count.as_entire_binding(),
                },
            ],
        });
        keep_bind_groups.push(bind_group);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("omvg-sift-extrema"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.extrema);
        pass.set_bind_group(0, &keep_bind_groups[keep_bind_groups.len() - 1], &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(8),
            (cols as u32).div_ceil(8),
            levels.saturating_sub(3).max(1) as u32,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_refine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dog: &wgpu::Buffer,
        candidates: &wgpu::Buffer,
        count: &wgpu::Buffer,
        refined: &wgpu::Buffer,
        out_count: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        levels: usize,
        num_scales: usize,
        delta: f32,
        sigma_min: f32,
        delta_min: f32,
        edge_threshold: f32,
        peak_threshold: f32,
        octave: usize,
        capacity: usize,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        keep_bind_groups: &mut Vec<wgpu::BindGroup>,
    ) {
        let refine_workgroups = REFINE_WORKGROUPS.min(capacity.div_ceil(64).max(1) as u32);
        let uniform = uniform_buffer(
            &self.device,
            &self.queue,
            "omvg-sift-refine",
            &RefineParams {
                rows_cols: [rows as u32, cols as u32],
                levels: levels as u32,
                num_scales: num_scales as u32,
                delta,
                sigma_min,
                delta_min,
                edge_threshold,
                peak_threshold,
                cap: capacity as u32,
                octave: octave as u32,
                grid_workgroups: refine_workgroups,
                pad1: [0; 4],
            },
        );
        keep_buffers.push(uniform.clone());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("omvg-sift-refine"),
            layout: &self.layouts.refine,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dog.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: refined.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_count.as_entire_binding(),
                },
            ],
        });
        keep_bind_groups.push(bind_group);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("omvg-sift-refine"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.refine);
        pass.set_bind_group(0, &keep_bind_groups[keep_bind_groups.len() - 1], &[]);
        pass.dispatch_workgroups(refine_workgroups, 1, 1);
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
                ],
            ),
            decimate: layout(
                device,
                "omvg-sift-decimate",
                &[uniform(0), storage(1, true), storage(2, false)],
            ),
            dog: layout(
                device,
                "omvg-sift-dog",
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
                ],
            ),
        }
    }
}

impl Pipelines {
    fn new(device: &wgpu::Device, layouts: &Layouts) -> Self {
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
            dog: pipeline(
                device,
                "omvg-sift-dog",
                layouts.dog.clone(),
                include_str!("shaders/dog.wgsl"),
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

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).div_ceil(4) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(value);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (bytes.len().max(16).div_ceil(16) * 16) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn read_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ranges: &[(&wgpu::Buffer, usize)],
) -> Result<(Vec<u8>, Vec<usize>), RunError> {
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
        return Ok((Vec::new(), offsets));
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("omvg-sift-staging"),
        size: total as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| RunError::Readback(format!("{error:?}")))?;
    receiver
        .recv()
        .map_err(|error| RunError::Readback(error.to_string()))?
        .map_err(|error| RunError::Readback(format!("{error:?}")))?;
    let data = slice
        .get_mapped_range()
        .map_err(|error| RunError::Readback(error.to_string()))?
        .to_vec();
    staging.unmap();
    Ok((data, offsets))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(word)
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(read_u32(bytes, offset))
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
    pad: u32,
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
    pad: u32,
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
struct DogParams {
    rows_cols: [u32; 2],
    src_off: u32,
    src_next_off: u32,
    dst_off: u32,
    pad0: u32,
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
    pad: u32,
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
}

impl From<RunError> for SiftGpuError {
    fn from(error: RunError) -> Self {
        match error {
            RunError::Capacity { what } => Self::Capacity { what },
            RunError::Readback(message) => Self::Readback(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::FeatureBudgetPolicy;

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
                if dx * dx + dy * dy <= 4.0 {
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
