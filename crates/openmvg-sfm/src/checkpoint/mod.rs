// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned binary files for the prepared [`ViewGraph`] and the [`Reconstruction`].
//!
//! The front end is the expensive part of a run: save its view graph once, then run the engines
//! on the file as often as you like. A checkpoint keeps the whole value, seed included, so an
//! engine gives the same reconstruction from the file as from the original. Only OpenMVG-rs reads
//! these files, with a release that writes the same schema version; to share data with other
//! tools, use the OMVG format of `openmvg-io` instead. The conventional extension is `.omvgrs`
//! (`view_graph.omvgrs`, `scene.omvgrs`), and `.omvg` names an OMVG dataset folder; reading
//! never looks at the extension.
//!
//! Both products share one file layout:
//!
//! ```text
//! magic           8 bytes   b"OMVGVG\0\0" (ViewGraph) or b"OMVGRC\0\0" (Reconstruction)
//! schema version  u32 LE    VIEW_GRAPH_SCHEMA_VERSION or RECONSTRUCTION_SCHEMA_VERSION
//! content digest  32 bytes  SHA-256 of the content bytes
//! content         postcard  the product, to the end of the file
//! ```
//!
//! Reading checks the magic, then the version, then the digest, and finally the product
//! invariants. A build reads only the schema version it writes, so a file with another schema
//! version fails with [`SfmError::CheckpointVersionMismatch`]: keep the images and
//! configuration that rebuild it.

// Postcard stores neither field names nor variant names, so any change to a stored type (a
// field, the field order, or the variant order of an enum) changes the layout and needs a new
// schema version. The golden files in `golden/` pin the version 1 layout.

use std::io::Write;
use std::path::Path;

use dto::{ReconstructionDto, ViewGraphDto};
use sha2::{Digest, Sha256};

use crate::{Reconstruction, Result, SfmError, ViewGraph};

pub(crate) mod dto;

/// `ViewGraph` file schema version written by [`write_view_graph`] and the only one
/// [`read_view_graph`] accepts.
pub const VIEW_GRAPH_SCHEMA_VERSION: u32 = 1;
/// Reconstruction file schema version written by [`write_reconstruction`] and the only one
/// [`read_reconstruction`] accepts.
pub const RECONSTRUCTION_SCHEMA_VERSION: u32 = 1;

const VIEW_GRAPH_MAGIC: [u8; 8] = *b"OMVGVG\0\0";
const RECONSTRUCTION_MAGIC: [u8; 8] = *b"OMVGRC\0\0";
const PREAMBLE_LEN: usize = 12;
const HEADER_LEN: usize = PREAMBLE_LEN + 32;

/// One persisted product: its magic, schema version, and display name for errors.
struct Product {
    magic: [u8; 8],
    version: u32,
    name: &'static str,
}

const VIEW_GRAPH: Product = Product {
    magic: VIEW_GRAPH_MAGIC,
    version: VIEW_GRAPH_SCHEMA_VERSION,
    name: "view-graph",
};
const RECONSTRUCTION: Product = Product {
    magic: RECONSTRUCTION_MAGIC,
    version: RECONSTRUCTION_SCHEMA_VERSION,
    name: "reconstruction",
};

impl ViewGraph {
    /// Encode the graph in the `ViewGraph` file layout (see [`crate::checkpoint::write_view_graph`]).
    ///
    /// Fails if validation or serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        encode(&VIEW_GRAPH, &ViewGraphDto::from_view_graph(self))
    }

    /// Decode bytes produced by [`Self::to_bytes`] or [`crate::checkpoint::write_view_graph`].
    ///
    /// Fails like [`crate::checkpoint::read_view_graph`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        decode::<ViewGraphDto>(&VIEW_GRAPH, bytes)?.into_view_graph()
    }
}

impl Reconstruction {
    /// Encode the reconstruction in the reconstruction file layout (see
    /// [`crate::checkpoint::write_reconstruction`]). The run report is not stored.
    ///
    /// Fails if validation or serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.workspace.validate()?;
        encode(
            &RECONSTRUCTION,
            &ReconstructionDto::from_reconstruction(self),
        )
    }

    /// Decode bytes produced by [`Self::to_bytes`] or [`crate::checkpoint::write_reconstruction`].
    ///
    /// Fails like [`crate::checkpoint::read_reconstruction`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        decode::<ReconstructionDto>(&RECONSTRUCTION, bytes)?.into_reconstruction()
    }
}

/// Validate `view_graph` and write it with a checksummed header to `path`.
///
/// The file is written to a temporary sibling, synced, and renamed into place, so `path` never
/// holds a partial file. Fails if validation, serialization, or any I/O step fails.
pub fn write_view_graph(view_graph: &ViewGraph, path: impl AsRef<Path>) -> Result<()> {
    write_bytes(path.as_ref(), &view_graph.to_bytes()?)
}

/// Read and validate a `ViewGraph` file written by [`write_view_graph`].
///
/// Fails with [`SfmError::Io`] when the file cannot be read, with [`SfmError::InvalidInput`]
/// when it is not a `ViewGraph` file or cannot be decoded, with
/// [`SfmError::CheckpointVersionMismatch`] for another schema version, and with
/// [`SfmError::CheckpointChecksumMismatch`] when the content digest disagrees with the header.
pub fn read_view_graph(path: impl AsRef<Path>) -> Result<ViewGraph> {
    ViewGraph::from_bytes(&std::fs::read(path)?)
}

/// Validate `reconstruction` and write it with a checksummed header to `path`.
///
/// Stores the cameras, poses, landmarks, components, tracks, and verified pair state; the run
/// report is not part of the file. The write is atomic like [`write_view_graph`].
pub fn write_reconstruction(reconstruction: &Reconstruction, path: impl AsRef<Path>) -> Result<()> {
    write_bytes(path.as_ref(), &reconstruction.to_bytes()?)
}

/// Read and validate a reconstruction file written by [`write_reconstruction`].
///
/// Fails like [`read_view_graph`], and when the stored state violates a reconstruction
/// invariant.
pub fn read_reconstruction(path: impl AsRef<Path>) -> Result<Reconstruction> {
    Reconstruction::from_bytes(&std::fs::read(path)?)
}

fn encode<T: serde::Serialize>(product: &Product, content: &T) -> Result<Vec<u8>> {
    let content = postcard::to_stdvec(content).map_err(|error| {
        SfmError::InvalidInput(format!("{} serialization failed: {error}", product.name))
    })?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + content.len());
    bytes.extend_from_slice(&product.magic);
    bytes.extend_from_slice(&product.version.to_le_bytes());
    bytes.extend_from_slice(&Sha256::digest(&content));
    bytes.extend_from_slice(&content);
    Ok(bytes)
}

fn decode<T: serde::de::DeserializeOwned>(product: &Product, bytes: &[u8]) -> Result<T> {
    if bytes.len() < PREAMBLE_LEN || bytes[..8] != product.magic {
        return Err(SfmError::InvalidInput(format!(
            "not an OpenMVG-rs {} file",
            product.name
        )));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != product.version {
        return Err(SfmError::CheckpointVersionMismatch {
            found: version,
            supported: product.version,
        });
    }
    let Some((expected, content)) = bytes[PREAMBLE_LEN..].split_first_chunk::<32>() else {
        return Err(SfmError::InvalidInput(format!(
            "{} file ends inside its header",
            product.name
        )));
    };
    let digest: [u8; 32] = Sha256::digest(content).into();
    if digest != *expected {
        return Err(SfmError::CheckpointChecksumMismatch {
            section: "content",
            found: sha256_text(&digest),
            expected: sha256_text(expected),
        });
    }
    let (value, trailing) = postcard::take_from_bytes::<T>(content).map_err(|error| {
        SfmError::InvalidInput(format!("{} parse failed: {error}", product.name))
    })?;
    if !trailing.is_empty() {
        return Err(SfmError::InvalidInput(format!(
            "{} file has {} trailing bytes",
            product.name,
            trailing.len()
        )));
    }
    Ok(value)
}

fn sha256_text(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    digest
        .iter()
        .fold(String::from("sha256:"), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        })
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::atomic::atomic_write_with(path, |writer| -> Result<()> {
        writer.write_all(bytes)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageMeta;
    use crate::two_view::{
        GeometricModel, InlierMask, ModelEvidence, PairMatches, TerminationReason,
    };
    use crate::{
        Camera, FeatureId, FeatureMatch, FeatureSet, Incremental, IntrinsicId, Keypoint, PairId,
        RelativePose, SfmOutput, SfmWorkspace, Track, TrackId, TrackObservation, TrackSet, ViewId,
        ViewPair,
    };
    use nalgebra::{Matrix3, Point2, Point3, Rotation3, Vector3};
    use openmvg_core::camera::{
        FisheyeCamera, FisheyeDistortion, PinholeCamera, RadialDistortion, SphericalCamera,
        TangentialDistortion,
    };
    use std::path::PathBuf;

    const ROUND_TRIP_FOCAL: f64 = 51.248_178_375_505_404;
    const CAMERA_COUNT: usize = 4;
    const TRACK_COUNT: usize = 16;

    fn graph(root_seed: impl Into<openmvg_core::RunSeed>) -> ViewGraph {
        let mut workspace = SfmWorkspace::builder()
            .root_seed(root_seed)
            .image_meta([
                ImageMeta::external_id("image-a", 64, 48),
                ImageMeta::external_id("image-b", 64, 48),
            ])
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs([(ViewId::new(0), ViewId::new(1))])
            .build()
            .unwrap();
        workspace
            .pair_graph_mut()
            .set_confidence(PairId::new(0), 0.375)
            .unwrap();
        // The pair was matched, found no correspondences, and failed verification.
        for view in 0..2 {
            workspace
                .feature_sets
                .set(view, FeatureSet::default())
                .unwrap();
            workspace.keypoints.set(view, Vec::new()).unwrap();
        }
        workspace
            .matches
            .set(
                0,
                PairMatches::try_new(ViewPair::new(ViewId::new(0), ViewId::new(1)), Vec::new())
                    .unwrap(),
            )
            .unwrap();
        workspace
            .pair_graph_mut()
            .deactivate_pair_and(PairId::new(0))
            .unwrap();
        ViewGraph::try_from_workspace(workspace).unwrap()
    }

    fn reconstruction_graph() -> ViewGraph {
        let points = (0..TRACK_COUNT)
            .map(|index| {
                Point3::new(
                    -0.5 + 0.08 * index as f64,
                    -0.3 + 0.2 * (index % 4) as f64,
                    4.0 + 0.25 * (index % 3) as f64,
                )
            })
            .collect::<Vec<_>>();
        let tracks = TrackSet::try_new(
            (0..TRACK_COUNT)
                .map(|track_index| {
                    Track::try_new(
                        TrackId::new(track_index),
                        (0..CAMERA_COUNT)
                            .map(|view| TrackObservation {
                                view: ViewId::new(view),
                                feature: FeatureId::new(track_index),
                            })
                            .collect(),
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        )
        .unwrap();
        let keypoints = (0..CAMERA_COUNT)
            .map(|view| {
                points
                    .iter()
                    .map(|point| {
                        let camera_point = point.coords - Vector3::new(view as f64 * 0.5, 0.0, 0.0);
                        Keypoint::try_new(
                            (ROUND_TRIP_FOCAL * camera_point.x / camera_point.z + 64.0) as f32,
                            (ROUND_TRIP_FOCAL * camera_point.y / camera_point.z + 48.0) as f32,
                            1.0,
                            0.0,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let feature_sets = keypoints
            .iter()
            .map(|view_keypoints| {
                FeatureSet::try_new(
                    view_keypoints
                        .iter()
                        .map(|keypoint| {
                            Point2::new(f64::from(keypoint.x()), f64::from(keypoint.y()))
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pairs = (0..CAMERA_COUNT - 1)
            .map(|view| (ViewId::new(view), ViewId::new(view + 1)))
            .collect::<Vec<_>>();
        let mut workspace = SfmWorkspace::builder()
            .root_seed(7)
            .image_meta(
                (0..CAMERA_COUNT)
                    .map(|view| ImageMeta::external_id(format!("roundtrip-{view}"), 128, 96)),
            )
            .camera_models([
                Camera::try_pinhole(ROUND_TRIP_FOCAL, ROUND_TRIP_FOCAL, 64.0, 48.0).unwrap(),
            ])
            .shared_intrinsic(IntrinsicId::new(0))
            .imported_pairs(pairs)
            .tracks(tracks, feature_sets)
            .build()
            .unwrap();
        for (view, view_keypoints) in keypoints.into_iter().enumerate() {
            workspace.keypoints.set(view, view_keypoints).unwrap();
        }

        for pair_index in 0..CAMERA_COUNT - 1 {
            // Feature `k` of every view observes track `k`.
            let matches = (0..TRACK_COUNT)
                .map(|feature| {
                    FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), None)
                        .unwrap()
                })
                .collect();
            workspace
                .matches
                .set(
                    pair_index,
                    PairMatches::try_new(
                        ViewPair::new(ViewId::new(pair_index), ViewId::new(pair_index + 1)),
                        matches,
                    )
                    .unwrap(),
                )
                .unwrap();
            let evidence = ModelEvidence::try_new(
                InlierMask::all_inliers(TRACK_COUNT),
                crate::workspace::quality::ErrorStats::from_values(&[0.0; TRACK_COUNT], &[1.0]),
                openmvg_core::PixelThreshold::ONE_PIXEL.into(),
                "checkpoint-roundtrip",
                1,
                TerminationReason::Converged,
                Vec::new(),
            )
            .unwrap();
            workspace
                .geometric_models
                .set(
                    pair_index,
                    GeometricModel::Essential {
                        matrix: Matrix3::identity(),
                        relative_pose: RelativePose::from_baseline_direction(
                            Rotation3::identity(),
                            Vector3::x_axis(),
                        )
                        .unwrap(),
                        evidence,
                    },
                )
                .unwrap();
        }
        ViewGraph::try_from_workspace(workspace).unwrap()
    }

    fn scientific_digest(result: &SfmOutput) -> String {
        let reconstruction = result.reconstruction();
        let registered_views = reconstruction.registered_views().collect::<Vec<_>>();
        let mut value = serde_json::json!({
            "camera_models": reconstruction.camera_models(),
            "registered_views": registered_views,
            "landmarks": reconstruction.landmarks(),
            "components": reconstruction.components(),
            "report": result.report(),
        });
        remove_telemetry(&mut value);
        crate::determinism::stable_json_hash(&value).unwrap()
    }

    fn remove_telemetry(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    remove_telemetry(value);
                }
            }
            serde_json::Value::Object(fields) => {
                fields.remove("wall_time_s");
                fields.remove("timings");
                for value in fields.values_mut() {
                    remove_telemetry(value);
                }
            }
            _ => {}
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("openmvg-sfm-{}-{name}", std::process::id()))
    }

    #[test]
    fn view_graph_round_trips_pair_state() {
        let path = temp_path("pairs.omvgrs");
        write_view_graph(&graph(42), &path).unwrap();
        let loaded = read_view_graph(&path).unwrap();

        assert_eq!(loaded.root_seed(), 42.into());
        assert_eq!(loaded.views().len(), 2);
        assert!(!loaded.workspace().pair_graph().is_active(PairId::new(0)));
        assert_eq!(
            loaded.workspace().pair_graph().confidence(PairId::new(0)),
            Some(0.375)
        );
        std::fs::remove_file(path).ok();
    }

    /// Every stored kind of ViewGraph state, built from exactly representable values so that
    /// its encoding is the same on every platform.
    fn golden_graph() -> ViewGraph {
        use openmvg_features::model::{DescriptorSet, ExtractedFeatures};

        let pinhole = PinholeCamera::try_new(510.0, 505.0, 320.5, 239.5).unwrap();
        let cameras = [
            Camera::Pinhole(pinhole.clone()),
            Camera::try_brown_conrady(
                510.0,
                505.0,
                320.5,
                239.5,
                RadialDistortion::K1K2K3(-0.125, 0.015625, 0.0),
                TangentialDistortion::P1P2(0.0009765625, -0.001953125),
            )
            .unwrap(),
            Camera::Fisheye(
                FisheyeCamera::try_new(pinhole, FisheyeDistortion::K1(0.0078125)).unwrap(),
            ),
            Camera::Spherical(SphericalCamera::try_new(640).unwrap()),
        ];
        let mut workspace = SfmWorkspace::builder()
            .root_seed(42)
            .image_meta((0..4).map(|view| ImageMeta::external_id(format!("view-{view}"), 640, 320)))
            .camera_models(cameras)
            .intrinsic_priors([
                crate::model::IntrinsicPrior::Trusted,
                crate::model::IntrinsicPrior::Guess,
                crate::model::IntrinsicPrior::Trusted,
                crate::model::IntrinsicPrior::Trusted,
            ])
            .view_intrinsics((0..4).map(IntrinsicId::new))
            .imported_pairs([
                (ViewId::new(0), ViewId::new(1)),
                (ViewId::new(2), ViewId::new(3)),
            ])
            .build()
            .unwrap();
        for view in 0..4 {
            let keypoints = (0..3)
                .map(|feature| {
                    Keypoint::try_new(10.0 * feature as f32 + 0.25, view as f32, 1.5, 0.5).unwrap()
                })
                .collect::<Vec<_>>();
            let features = ExtractedFeatures::try_new(
                keypoints,
                DescriptorSet::try_u8(vec![7; 3], 1).unwrap(),
            )
            .unwrap();
            workspace
                .feature_sets
                .set(view, features.feature_set())
                .unwrap();
            workspace.set_features(view, features).unwrap();
        }
        let matches = (0..3)
            .map(|feature| {
                FeatureMatch::try_new(FeatureId::new(feature), FeatureId::new(feature), Some(0.5))
                    .unwrap()
            })
            .collect();
        workspace
            .matches
            .set(
                0,
                PairMatches::try_new(ViewPair::new(ViewId::new(0), ViewId::new(1)), matches)
                    .unwrap(),
            )
            .unwrap();
        workspace
            .geometric_models
            .set(
                0,
                GeometricModel::Essential {
                    matrix: Matrix3::identity(),
                    relative_pose: RelativePose::from_baseline_direction(
                        Rotation3::identity(),
                        Vector3::x_axis(),
                    )
                    .unwrap(),
                    evidence: ModelEvidence::from_inliers(InlierMask::all_inliers(3)),
                },
            )
            .unwrap();
        // The second pair was matched, found no correspondences, and failed verification.
        workspace
            .matches
            .set(
                1,
                PairMatches::try_new(ViewPair::new(ViewId::new(2), ViewId::new(3)), Vec::new())
                    .unwrap(),
            )
            .unwrap();
        workspace
            .pair_graph_mut()
            .set_confidence(PairId::new(1), 0.375)
            .unwrap();
        workspace
            .pair_graph_mut()
            .deactivate_pair_and(PairId::new(1))
            .unwrap();
        assert!(ViewGraph::try_from_workspace(workspace.clone()).is_err());
        workspace.release_descriptors().unwrap();
        ViewGraph::try_from_workspace(workspace)
            .unwrap()
            .with_calibrated_intrinsics(true)
            .with_extraction_method(Some(crate::ExtractionMethod::Alp))
    }

    /// Two registered views and one active landmark seen by both.
    fn golden_reconstruction() -> Reconstruction {
        let mut workspace = SfmWorkspace::builder()
            .root_seed(42)
            .image_meta((0..2).map(|view| ImageMeta::external_id(format!("{view}.jpg"), 640, 480)))
            .camera_models([Camera::try_pinhole(500.0, 500.0, 320.0, 240.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .build()
            .unwrap();
        for view in 0..2 {
            let center = Point3::new(view as f64, 0.0, 0.0);
            let pose =
                crate::WorldToCameraPose::from_rotation_center(Rotation3::identity(), center);
            workspace
                .set_pose_for_view(ViewId::new(view), pose.unwrap(), None)
                .unwrap();
            let pixel = Point2::new(320.0 - 100.0 * view as f64, 240.0);
            workspace
                .feature_sets
                .set(view, FeatureSet::try_new(vec![pixel]).unwrap())
                .unwrap();
        }
        let observations = (0..2)
            .map(|view| TrackObservation {
                view: ViewId::new(view),
                feature: FeatureId::new(0),
            })
            .collect();
        workspace.tracks =
            TrackSet::try_new(vec![Track::try_new(TrackId::new(0), observations).unwrap()])
                .unwrap();
        workspace.landmarks =
            crate::model::Landmarks::try_new(vec![crate::model::Landmark::active(
                crate::LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 5.0),
                openmvg_core::PixelResidual::try_new(0.125).unwrap(),
                crate::model::ObservationSupport::try_from_indices([0, 1], 2).unwrap(),
            )])
            .unwrap();
        Reconstruction::new(workspace, crate::SfmEngine::Global, false)
    }

    // The committed encodings of the two fixtures. They change only with a new schema version,
    // by replacing each file with the bytes its fixture then encodes to.
    const GOLDEN_VIEW_GRAPH: &[u8] = include_bytes!("golden/view_graph_v1.omvgrs");
    const GOLDEN_RECONSTRUCTION: &[u8] = include_bytes!("golden/reconstruction_v1.omvgrs");

    #[test]
    fn view_graph_layout_matches_the_golden_file() {
        let graph = golden_graph();
        assert!(
            graph.to_bytes().unwrap() == GOLDEN_VIEW_GRAPH,
            "the stored ViewGraph layout changed; it needs a new schema version"
        );
        let loaded = ViewGraph::from_bytes(GOLDEN_VIEW_GRAPH).unwrap();
        assert_eq!(
            ViewGraphDto::from_view_graph(&loaded),
            ViewGraphDto::from_view_graph(&graph)
        );
        assert_eq!(loaded.workspace().feature_store().populated_count(), 0);
    }

    #[test]
    fn reconstruction_layout_matches_the_golden_file() {
        let reconstruction = golden_reconstruction();
        assert!(
            reconstruction.to_bytes().unwrap() == GOLDEN_RECONSTRUCTION,
            "the stored Reconstruction layout changed; it needs a new schema version"
        );
        let loaded = Reconstruction::from_bytes(GOLDEN_RECONSTRUCTION).unwrap();
        assert_eq!(
            ReconstructionDto::from_reconstruction(&loaded),
            ReconstructionDto::from_reconstruction(&reconstruction)
        );
    }

    #[test]
    fn full_width_root_seed_round_trips_without_truncation() {
        let seed = openmvg_core::RunSeed::from_bytes(std::array::from_fn(|index| index as u8));
        let loaded = ViewGraph::from_bytes(&graph(seed).to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.root_seed(), seed);
    }

    #[test]
    fn persisted_view_graph_reproduces_the_in_memory_scientific_digest() {
        let path = temp_path("scientific-roundtrip.omvgrs");
        let graph = reconstruction_graph();
        let in_memory = Incremental::default().run_prepared(&graph).unwrap();

        write_view_graph(&graph, &path).unwrap();
        let restored = read_view_graph(&path).unwrap();
        let from_checkpoint = Incremental::default().run_prepared(&restored).unwrap();
        let owned = Incremental::default()
            .run_prepared_owned(restored.clone())
            .unwrap();

        assert_eq!(
            scientific_digest(&from_checkpoint),
            scientific_digest(&in_memory)
        );
        assert_eq!(scientific_digest(&owned), scientific_digest(&in_memory));
        assert_eq!(
            ViewGraphDto::from_view_graph(&restored),
            ViewGraphDto::from_view_graph(&graph),
            "checkpointing must preserve every prepared f64 exactly"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn reconstruction_round_trips_exactly() {
        let graph = reconstruction_graph();
        for (engine, output) in [
            ("incremental", Incremental::default().run_prepared(&graph)),
            ("global", crate::Global::default().run_prepared(&graph)),
        ] {
            let path = temp_path(&format!("{engine}.omvgrs"));
            let output = output.unwrap();
            let reconstruction = output.reconstruction();
            assert!(reconstruction.registered_view_count() > 0, "{engine}");

            write_reconstruction(reconstruction, &path).unwrap();
            let restored = read_reconstruction(&path).unwrap();

            assert_eq!(
                ReconstructionDto::from_reconstruction(&restored),
                ReconstructionDto::from_reconstruction(reconstruction),
                "{engine}: persistence must preserve every reconstruction f64 exactly"
            );
            assert_eq!(
                restored.to_bytes().unwrap(),
                std::fs::read(&path).unwrap(),
                "{engine}: file and in-memory encodings are the same bytes"
            );
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn products_are_not_interchangeable() {
        let graph_bytes = graph(42).to_bytes().unwrap();
        let error = Reconstruction::from_bytes(&graph_bytes).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not an OpenMVG-rs reconstruction file")
        );
        assert!(ViewGraph::from_bytes(b"{\"header\":{}}").is_err());
        assert!(ViewGraph::from_bytes(&graph_bytes[..10]).is_err());
    }

    #[test]
    fn corrupted_content_is_rejected() {
        let mut bytes = graph(42).to_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;

        let error = ViewGraph::from_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("content checksum mismatch"));
    }

    #[test]
    fn truncated_files_are_rejected() {
        let bytes = graph(42).to_bytes().unwrap();
        for length in [
            0,
            PREAMBLE_LEN - 1,
            PREAMBLE_LEN,
            HEADER_LEN - 1,
            HEADER_LEN,
            bytes.len() - 1,
        ] {
            assert!(
                ViewGraph::from_bytes(&bytes[..length]).is_err(),
                "{length} bytes"
            );
        }
    }

    #[test]
    fn relative_poses_reload_bit_exactly() {
        let evidence = ModelEvidence::try_new(
            InlierMask::all_inliers(1),
            crate::workspace::quality::ErrorStats::from_values(&[0.0], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "relative-pose-roundtrip",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )
        .unwrap();
        for step in 0..64 {
            let angle = f64::from(step) * 0.097;
            let model = GeometricModel::Essential {
                matrix: Matrix3::identity(),
                relative_pose: RelativePose::from_baseline_direction(
                    Rotation3::from_euler_angles(angle, -0.7 * angle, 0.3 + angle),
                    nalgebra::Unit::new_normalize(Vector3::new(1.0, angle.sin(), -0.3)),
                )
                .unwrap(),
                evidence: evidence.clone(),
            };
            let bytes = postcard::to_stdvec(&model).unwrap();
            let restored = postcard::from_bytes::<GeometricModel>(&bytes).unwrap();
            assert_eq!(restored, model, "step {step}");
        }
    }

    #[test]
    fn schema_mismatch_is_typed() {
        let mut bytes = graph(42).to_bytes().unwrap();
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());

        assert!(matches!(
            ViewGraph::from_bytes(&bytes),
            Err(SfmError::CheckpointVersionMismatch {
                found: u32::MAX,
                ..
            })
        ));
    }
}
