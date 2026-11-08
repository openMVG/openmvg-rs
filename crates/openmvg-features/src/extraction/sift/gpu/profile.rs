// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stage profiling for the GPU SIFT backend (development instrumentation).
//!
//! `OMVG_SIFT_GPU_PROFILE` selects what an extractor records; totals per label,
//! averaged over its extractions, are printed to stderr when it is dropped:
//!
//! - `host`: wall-clock time of each host phase only. Dispatches keep sharing
//!   passes, so the timings match an unprofiled run.
//! - `span`: also GPU timestamps around every compute pass as recorded, and the
//!   span from the first to the last pass of each extraction.
//! - `1`: GPU timestamps around every dispatch, each in its own pass. This
//!   gives per-kernel times but serializes kernels that would overlap.
//!
//! GPU timestamps need an adapter with `TIMESTAMP_QUERY`; without it only host
//! phases are recorded.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const MAX_QUERIES: u32 = 4096;

pub(super) struct Profiler {
    /// Time every dispatch in its own pass; otherwise time passes as recorded.
    pub(super) split: bool,
    queries: Option<Queries>,
    next: AtomicU32,
    state: Mutex<State>,
    period: f32,
}

struct Queries {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

struct State {
    labels: Vec<&'static str>,
    gpu: BTreeMap<&'static str, (f64, u64)>,
    host: BTreeMap<&'static str, (f64, u64)>,
    last: Instant,
    images: u64,
}

impl Profiler {
    /// `Some(true)` for GPU and host profiling, `Some(false)` for host only.
    pub(super) fn requested() -> Option<bool> {
        match std::env::var("OMVG_SIFT_GPU_PROFILE").as_deref() {
            Ok("1" | "span") => Some(true),
            Ok("host") => Some(false),
            _ => None,
        }
    }

    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, timestamps: bool) -> Self {
        let queries = timestamps.then(|| {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("omvg-sift-profile"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_QUERIES,
            });
            let bytes = u64::from(MAX_QUERIES) * 8;
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("omvg-sift-profile-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("omvg-sift-profile-readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Queries {
                query_set,
                resolve,
                readback,
            }
        });
        Self {
            split: std::env::var("OMVG_SIFT_GPU_PROFILE").as_deref() == Ok("1"),
            queries,
            next: AtomicU32::new(0),
            state: Mutex::new(State {
                labels: Vec::new(),
                gpu: BTreeMap::new(),
                host: BTreeMap::new(),
                last: Instant::now(),
                images: 0,
            }),
            period: queue.get_timestamp_period(),
        }
    }

    /// Timestamp writes for one compute pass labelled `label`.
    pub(super) fn pass(&self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let queries = self.queries.as_ref()?;
        let index = self.next.fetch_add(2, Ordering::Relaxed);
        if index + 2 > MAX_QUERIES {
            return None;
        }
        if let Ok(mut state) = self.state.lock() {
            state.labels.push(label);
        }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &queries.query_set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        })
    }

    /// Start of one extraction.
    pub(super) fn begin(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.last = Instant::now();
            state.images += 1;
        }
    }

    /// End of a host phase that began at the previous mark.
    pub(super) fn mark(&self, label: &'static str) {
        if let Ok(mut state) = self.state.lock() {
            let now = Instant::now();
            let span = (now - state.last).as_secs_f64() * 1e3;
            state.last = now;
            let entry = state.host.entry(label).or_insert((0.0, 0));
            entry.0 += span;
            entry.1 += 1;
        }
    }

    /// Resolve this extraction's GPU timestamps into the per-label totals.
    pub(super) fn finish(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let used = self.next.swap(0, Ordering::Relaxed).min(MAX_QUERIES);
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let labels = core::mem::take(&mut state.labels);
        let Some(queries) = self.queries.as_ref() else {
            state.last = Instant::now();
            return;
        };
        if used == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("omvg-sift-profile"),
        });
        encoder.resolve_query_set(&queries.query_set, 0..used, &queries.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &queries.resolve,
            0,
            &queries.readback,
            0,
            u64::from(used) * 8,
        );
        queue.submit(Some(encoder.finish()));
        let slice = queries.readback.slice(..u64::from(used) * 8);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(view) = slice.get_mapped_range() {
            let stamps: Vec<u64> = view
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| u64::from_le_bytes(*chunk))
                .collect();
            let pairs = stamps.as_chunks::<2>().0;
            if let (Some(first), Some(last)) = (pairs.first(), pairs.last()) {
                let ms = last[1].saturating_sub(first[0]) as f64 * f64::from(self.period) * 1e-6;
                let entry = state.gpu.entry("~span all passes").or_insert((0.0, 0));
                entry.0 += ms;
                entry.1 += 1;
            }
            for (pair, label) in pairs.iter().zip(labels) {
                let ms = pair[1].saturating_sub(pair[0]) as f64 * f64::from(self.period) * 1e-6;
                let entry = state.gpu.entry(label).or_insert((0.0, 0));
                entry.0 += ms;
                entry.1 += 1;
            }
        }
        queries.readback.unmap();
        state.last = Instant::now();
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        let Ok(state) = self.state.lock() else {
            return;
        };
        let images = state.images.max(1) as f64;
        eprintln!(
            "GPU SIFT profile over {} extractions (ms per extraction)",
            state.images
        );
        let mut gpu_total = 0.0;
        for (label, (ms, count)) in &state.gpu {
            if !label.starts_with('~') {
                gpu_total += ms;
            }
            eprintln!(
                "  gpu  {label:<28} {:8.3} ms  ({:.1} passes)",
                ms / images,
                *count as f64 / images
            );
        }
        eprintln!(
            "  gpu  {:<28} {:8.3} ms",
            "TOTAL kernels",
            gpu_total / images
        );
        let mut host_total = 0.0;
        for (label, (ms, _)) in &state.host {
            host_total += ms;
            eprintln!("  host {label:<28} {:8.3} ms", ms / images);
        }
        eprintln!("  host {:<28} {:8.3} ms", "TOTAL wall", host_total / images);
    }
}
