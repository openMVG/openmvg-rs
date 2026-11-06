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
It is opt-in: default builds stay CPU-only and `no_std + alloc` builds are unchanged.

## License

Licensed under either Apache-2.0 or MIT. See the repository license files.
