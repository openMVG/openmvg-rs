# openmvg-features

Local features for OpenMVG-rs: detect keypoints in an image, describe them, match them
between images, and link the matches into tracks.

```toml
[dependencies]
openmvg-features = "0.1"
```

Detect and describe the keypoints of an image with four bright blobs:

```rust
use openmvg_features::extraction::{AlpExtractor, AlpPreset, GrayImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A 128 × 128 image: rows are image rows, intensities run from 0 to 255.
    let blobs = [(32.0, 32.0, 3.0), (96.0, 40.0, 5.0), (40.0, 96.0, 4.0), (90.0, 92.0, 6.0)];
    let image = GrayImage::from_fn(128, 128, |y, x| {
        let intensity = |&(cx, cy, size): &(f32, f32, f32)| {
            let squared_distance = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            200.0 * (-squared_distance / (2.0 * size * size)).exp()
        };
        blobs.iter().map(intensity).sum::<f32>()
    });

    let features = AlpExtractor::with_preset(AlpPreset::Normal).extract_u8(&image)?;
    assert!(features.keypoints().len() >= blobs.len());
    assert_eq!(features.keypoints().len(), features.descriptors().len());
    Ok(())
}
```

The [crate documentation](https://docs.rs/openmvg-features) explains how to choose a detector,
a descriptor storage and a matcher, and how each detector works.

## Optional GPU SIFT

Enable the `gpu` feature to run the SIFT detector on a wgpu device:

```toml
[dependencies]
openmvg-features = { version = "0.1", features = ["gpu"] }
```

```rust,ignore
use openmvg_features::extraction::{GrayImage, SiftGpuError, SiftGpuExtractor, SiftPreset};

fn extract(image: &GrayImage) -> Result<(), Box<dyn std::error::Error>> {
    let extractor = match SiftGpuExtractor::with_preset(SiftPreset::Normal) {
        Ok(extractor) => extractor,
        Err(SiftGpuError::NoAdapter) => return Ok(()), // fall back to `SiftExtractor`
        Err(error) => return Err(error.into()),
    };
    let features = extractor.extract_u8(image)?;
    assert_eq!(features.keypoints().len(), features.descriptors().len());
    Ok(())
}
```

The GPU extractor accepts the same `SiftExtractor` configuration and returns the same
`ScaleSpaceFeatures`, with `detect_keypoints`, `extract_u8`, `extract_f32` and `extract_typed`.
Create one extractor per device and reuse it for a batch: its scratch buffers stay resident and
grow only when an image needs more space, so alternating image sizes do not reallocate them.
For many images, `extract_u8_batch` takes an iterator of images, for example one that decodes
the files of a folder, and overlaps the host work of each image with the GPU work of the previous
one; it returns the same features as `extract_u8`, in input order.
It is opt-in: default builds stay CPU-only and `no_std + alloc` builds are unchanged.

### GPU against CPU

Measured with `gpu_sift_benchmark` on an NVIDIA L40S (Vulkan) and a 16-thread AMD EPYC 7R13:
castle-P30 (Strecha, 30 images, 3072x2048), Sceaux Castle (11 images, 2832x2128) and South
Building (COLMAP, 128 images, 3072x2304), decoding excluded except in the folder column. Times
are medians per image.

| Data, preset | Features, CPU / GPU | CPU, one image | CPU, 16 images in parallel | GPU `extract_u8` | GPU `extract_u8_batch` | Folder, batch with parallel decoding | Batch against CPU in parallel | GPU memory, single / batch | GPU keypoints matching a CPU keypoint | Identical descriptor bytes | Orientation error p99 / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| castle-P30 Normal | 5,250 / 5,250 | 495 ms | 108 ms | 4.84 ms | 3.00 ms | 6.86 ms | 36x | 534 / 1,069 MiB | 99.994% | 97.8% | 1.4e-5 / 2.1e-2 rad |
| castle-P30 High | 16,203 / 16,198 | 614 ms | 166 ms | 7.40 ms | 5.22 ms | 8.67 ms | 32x | 545 / 1,091 MiB | 99.984% | 95.3% | 2.8e-5 / 4.7e-2 rad |
| castle-P30 Ultra | 29,350 / 29,337 | 3,089 ms | 632 ms | 18.61 ms | 16.04 ms | 19.46 ms | 39x | 2,126 / 4,252 MiB | 99.979% | 94.4% | 3.4e-5 / 4.9e-2 rad |
| Sceaux Normal | 5,872 / 5,872 | 450 ms | 112 ms | 4.61 ms | 3.05 ms | 7.03 ms | 37x | 512 / 1,024 MiB | 99.995% | 97.0% | 1.9e-5 / 1.2e-2 rad |
| Sceaux High | 14,425 / 14,421 | 581 ms | 153 ms | 6.86 ms | 5.04 ms | 8.85 ms | 30x | 514 / 1,028 MiB | 99.984% | 94.4% | 3.5e-5 / 3.0e-2 rad |
| Sceaux Ultra | 70,375 / 70,358 | 3,002 ms | 784 ms | 27.28 ms | 26.07 ms | 30.67 ms | 30x | 2,042 / 4,084 MiB | 99.983% | 93.3% | 4.3e-5 / 4.2e-2 rad |
| South Building Normal | 20,002 / 20,002 | 658 ms | 151 ms | 8.85 ms | 6.33 ms | 18.74 ms | 24x | 602 / 1,205 MiB | 99.993% | 97.6% | 1.6e-5 / 4.3e-2 rad |
| South Building High | 28,146 / 28,143 | 812 ms | 203 ms | 11.02 ms | 8.22 ms | 18.49 ms | 25x | 604 / 1,207 MiB | 99.989% | 96.7% | 2.2e-5 / 5.0e-2 rad |
| South Building Ultra | 197,681 / 197,665 | 4,351 ms | 1,258 ms | 67.43 ms | 57.54 ms | 65.98 ms | 22x | 2,455 / 4,911 MiB | 99.988% | 95.4% | 3.1e-5 / 5.0e-2 rad |

A GPU keypoint matches a CPU keypoint at the same octave within 0.5 px, 5% in scale and
0.05 rad in orientation. Descriptor bytes and orientation errors are over the matched pairs; the
mean u8 descriptor distance between them is 0.02 to 0.07 on a scale of 512. Both paths run in
f32, but the GPU's `exp`, `atan2` and blur sums round differently from the CPU's in the last bits,
so most orientations differ by a few ULP (median 5e-7 rad) and a borderline histogram peak can
move a little. GPU memory is the extractor's resident buffers after the first image of each set; the device and
driver add about 250 MiB. The folder column decodes the files: PGM for castle-P30 and Sceaux
Castle, JPEG for South Building, whose decoding bounds that column.

Results on Apple M1

| Data, preset | Features, CPU / GPU | CPU, one image | CPU, 8 images in parallel | GPU `extract_u8` | GPU `extract_u8_batch` | Folder, batch with parallel decoding | Batch against CPU in parallel | GPU keypoints matching a CPU keypoint | Identical descriptor bytes | Orientation error p99 / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| South Building Normal | 24,388.6 / 24,388.5 | 380.04 ms | 285.79 ms | 144.95 ms | 137.77 ms | 141.89 ms | 2.0x | 99.993% | 97.67% | 1.50e-5 / 2.22e-2 rad |
| South Building High | 31,620.5 / 31,618.0 | 560.09 ms | 338.83 ms | 177.11 ms | 169.66 ms | 174.71 ms | 1.9x | 99.991% | 96.92% | 2.05e-5 / 2.55e-2 rad |
| South Building Ultra* |  — / 21,683.5 | — | — | 1,058.39 ms | 1,055.41 ms | 1,039.54 ms | — | — | — | — |

* Ultra: the GPU measurements completed, but the remaining CPU/parity measurements ran out of memory (OOM).

### GPU memory

Device memory grows with the image area; the feature count adds little. Measured on an NVIDIA
L40S with `OMVG_SIFT_BENCH_MEMORY=1 cargo bench -p openmvg-features --features gpu --bench
gpu_sift_benchmark`:

| Preset | Bytes per input pixel, `extract_u8` | `extract_u8_batch` | 6.3 MP image | 25 MP image | Largest image, 2 GiB binding limit | Largest image, 128 MiB binding limit |
|---|---:|---:|---:|---:|---:|---:|
| Normal, High | about 89 | about 178 | 0.52–0.53 GiB (batch 1.04–1.07 GiB) | 2.1 GiB (batch 4.2 GiB) | 89 MP | 5.6 MP |
| Ultra | about 355 to 365 | about 710 to 730 | 2.1 GiB (batch 4.2 GiB) | over the limit | 22 MP | 1.4 MP |

Almost all of it is the scale space of the first octave: six Gaussian levels and five DoG planes
of four bytes per pixel, two blur temporaries and the candidate buffers, summed over the octave
pyramid. Ultra doubles the image before the first octave, so it needs four times as much. The
wgpu device and driver add about 250 MiB, and `SiftGpuExtractor::resident_bytes` reports the
extractor's own share.

The largest image is set by the device's largest storage binding
(`max_storage_buffer_binding_size`), not by the total memory: the six Gaussian levels of the first
octave form one buffer of `24 × pixels` bytes, or `96 × pixels` for Ultra. Native Vulkan drivers
typically allow 2 GiB; WebGPU only guarantees 128 MiB. A larger image returns
`SiftGpuError::Capacity`, and the CPU extractor handles it.

## License

Licensed under either Apache-2.0 or MIT. See the repository license files.
