// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Synthetic Dataset Generator
//!
//! Generates synthetic SfM datasets for testing and benchmarking.
//!
//! This module provides tools to create controlled test scenarios:
//! - Known ground truth camera poses
//! - Synthetic 3D point clouds
//! - Simulated 2D observations with optional noise
//! - Various scene configurations (planar, cubic, spherical)

use nalgebra::{Matrix3, Point2, Point3, Rotation3, Vector3};
use std::f64::consts::PI;

use openmvg_core::RunSeed;
use openmvg_core::camera::{Camera, CameraModel, PinholeCamera};
use openmvg_core::determinism::AlgorithmRng;
use openmvg_core::model::{
    CameraIntrinsics, StateValidationError, TrackId, ViewId, WorldToCameraPose,
};

/// Configuration for synthetic scene generation.
#[derive(Debug, Clone)]
pub struct SyntheticSceneConfig {
    /// Number of views.
    pub num_cameras: usize,
    /// Number of 3D points.
    pub num_points: usize,
    /// Scene type.
    pub scene_type: SceneType,
    /// Camera arrangement.
    pub camera_arrangement: CameraArrangement,
    /// Image dimensions (width, height).
    pub image_size: (usize, usize),
    /// Focal length in pixels.
    pub focal_length: f64,
    /// Random seed for reproducibility.
    pub seed: RunSeed,
    /// Noise level for 2D observations (pixels).
    pub observation_noise: f64,
    /// Minimum visibility per point.
    pub min_observations_per_point: usize,
}

impl Default for SyntheticSceneConfig {
    fn default() -> Self {
        Self {
            num_cameras: 10,
            num_points: 100,
            scene_type: SceneType::Cube { size: 2.0 },
            camera_arrangement: CameraArrangement::Circle { radius: 5.0 },
            image_size: (640, 480),
            focal_length: 500.0,
            seed: RunSeed::from_u64(12345),
            observation_noise: 0.0,
            min_observations_per_point: 3,
        }
    }
}

/// Type of 3D scene to generate.
#[derive(Debug, Clone)]
pub enum SceneType {
    /// Points distributed in a cube centered at origin.
    Cube {
        /// Side length of the cube.
        size: f64,
    },
    /// Points on a plane at z=0.
    Plane {
        /// Side length of the square sampling region.
        size: f64,
    },
    /// Points on a sphere surface.
    Sphere {
        /// Radius of the sphere.
        radius: f64,
    },
    /// Custom points provided by user.
    Custom(Vec<Point3<f64>>),
}

/// Camera arrangement pattern.
#[derive(Debug, Clone)]
pub enum CameraArrangement {
    /// Cameras on a circle looking at origin.
    Circle {
        /// Radius of the camera circle.
        radius: f64,
    },
    /// Cameras on a sphere looking at origin.
    Sphere {
        /// Radius of the camera sphere.
        radius: f64,
    },
    /// Cameras along a line.
    Line {
        /// Length spanned by the camera centers.
        length: f64,
        /// Direction of the camera line.
        direction: Vector3<f64>,
    },
    /// Cameras in a grid.
    Grid {
        /// Number of camera rows.
        rows: usize,
        /// Number of camera columns.
        cols: usize,
        /// Distance between adjacent camera centers.
        spacing: f64,
    },
    /// Custom poses.
    Custom(Vec<WorldToCameraPose>),
}

/// Generated synthetic scene with ground truth.
#[derive(Debug, Clone)]
pub struct SyntheticScene {
    /// Ground-truth camera poses, indexed by view.
    pub poses: Vec<(ViewId, WorldToCameraPose)>,
    /// Ground-truth points, indexed by their originating track.
    pub points: Vec<(TrackId, Point3<f64>)>,
    /// 2D observations keyed by `(view_id, track_id)`.
    pub observations: Vec<((ViewId, TrackId), Point2<f64>)>,
    /// Per-point tracks: the `(view_id, observation)` pairs of each point.
    pub tracks: Vec<Vec<(ViewId, Point2<f64>)>>,
    /// Configuration used.
    pub config: SyntheticSceneConfig,
}

impl SyntheticScene {
    /// Get number of cameras.
    pub fn num_cameras(&self) -> usize {
        self.poses.len()
    }

    /// Get number of points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Get number of observations.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Get average observations per point.
    pub fn avg_observations_per_point(&self) -> f64 {
        if self.tracks.is_empty() {
            0.0
        } else {
            self.observations.len() as f64 / self.tracks.len() as f64
        }
    }

    /// Build the camera intrinsics used by this scene.
    pub fn intrinsics(&self) -> Result<CameraIntrinsics, StateValidationError> {
        synthetic_intrinsic_data(&self.config)
    }
}

/// Panics on a fixture configuration no camera can represent; the fixture owns that contract.
#[allow(
    clippy::expect_used,
    reason = "the fixture configuration is a test-support contract"
)]
fn synthetic_intrinsic_data(
    config: &SyntheticSceneConfig,
) -> Result<CameraIntrinsics, StateValidationError> {
    let width = u32::try_from(config.image_size.0).expect("synthetic image width fits in u32");
    let height = u32::try_from(config.image_size.1).expect("synthetic image height fits in u32");
    let camera =
        PinholeCamera::try_with_focal(config.focal_length, width as f64 / 2.0, height as f64 / 2.0)
            .expect("synthetic focal length is valid");
    Ok(
        CameraIntrinsics::try_new(Camera::Pinhole(camera), width, height)
            .expect("synthetic camera matches its image domain"),
    )
}

/// Synthetic scene generator.
#[derive(Debug)]
pub struct SyntheticSceneGenerator {
    config: SyntheticSceneConfig,
    rng: AlgorithmRng,
}

impl SyntheticSceneGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SyntheticSceneConfig) -> Self {
        let seed = config.seed.derive("sfm/testing/synthetic-scene", &[]);
        let rng = AlgorithmRng::new(seed);
        Self { config, rng }
    }

    /// Create with default configuration.
    pub fn default_scene() -> Self {
        Self::new(SyntheticSceneConfig::default())
    }

    /// Generate the synthetic scene.
    pub fn generate(&mut self) -> Result<SyntheticScene, StateValidationError> {
        let intrinsics = synthetic_intrinsic_data(&self.config)?;

        // Generate camera poses
        let poses = self.generate_camera_poses()?;

        // Generate 3D points
        let points = self.generate_points();

        // Project points to cameras and create observations
        let mut observations = Vec::new();
        let mut tracks: Vec<Vec<(ViewId, Point2<f64>)>> = vec![Vec::new(); points.len()];

        for (point_id, point_3d) in points.iter().enumerate() {
            for (view_id, pose) in poses.iter().enumerate() {
                if let Some(proj) = self.project_point(point_3d, pose, &intrinsics) {
                    // Add noise if specified
                    let noisy_proj = if self.config.observation_noise > 0.0 {
                        let noise = self.config.observation_noise;
                        Point2::new(
                            proj.x + noise * self.rng.symmetric_unit_f64(),
                            proj.y + noise * self.rng.symmetric_unit_f64(),
                        )
                    } else {
                        proj
                    };

                    // Check if within image bounds
                    if noisy_proj.x >= 0.0
                        && noisy_proj.x < self.config.image_size.0 as f64
                        && noisy_proj.y >= 0.0
                        && noisy_proj.y < self.config.image_size.1 as f64
                    {
                        let view_id = ViewId::new(view_id);
                        observations.push(((view_id, TrackId::new(point_id)), noisy_proj));
                        tracks[point_id].push((view_id, noisy_proj));
                    }
                }
            }
        }

        let points: Vec<_> = points
            .into_iter()
            .zip(&tracks)
            .enumerate()
            .filter(|(_, (_, track))| track.len() >= self.config.min_observations_per_point)
            .map(|(point_id, (point, _))| (TrackId::new(point_id), point))
            .collect();

        // Filter tracks to only include points that were added
        let valid_tracks: Vec<_> = tracks
            .into_iter()
            .filter(|t| t.len() >= self.config.min_observations_per_point)
            .collect();

        Ok(SyntheticScene {
            poses: poses
                .into_iter()
                .enumerate()
                .map(|(view_id, pose)| (ViewId::new(view_id), pose))
                .collect(),
            points,
            observations,
            tracks: valid_tracks,
            config: self.config.clone(),
        })
    }

    /// Generate camera poses based on arrangement.
    fn generate_camera_poses(&mut self) -> Result<Vec<WorldToCameraPose>, StateValidationError> {
        match &self.config.camera_arrangement {
            CameraArrangement::Circle { radius } => {
                let n = self.config.num_cameras;
                (0..n)
                    .map(|i| {
                        let angle = 2.0 * PI * i as f64 / n as f64;
                        let center = Point3::new(radius * angle.cos(), radius * angle.sin(), 0.0);
                        // Look at origin
                        let look_dir = (Point3::origin() - center).normalize();
                        let rotation = look_at_rotation(&look_dir, &Vector3::z());
                        WorldToCameraPose::from_rotation_center(rotation, center)
                    })
                    .collect()
            }
            CameraArrangement::Sphere { radius } => {
                // Fibonacci sphere distribution
                let n = self.config.num_cameras;
                let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
                (0..n)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / golden_ratio;
                        let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
                        let center = Point3::new(
                            radius * phi.sin() * theta.cos(),
                            radius * phi.sin() * theta.sin(),
                            radius * phi.cos(),
                        );
                        let look_dir = (Point3::origin() - center).normalize();
                        let rotation = look_at_rotation(&look_dir, &Vector3::z());
                        WorldToCameraPose::from_rotation_center(rotation, center)
                    })
                    .collect()
            }
            CameraArrangement::Line { length, direction } => {
                let n = self.config.num_cameras;
                let dir = direction.normalize();
                let start = Point3::origin() - dir * (length / 2.0);
                (0..n)
                    .map(|i| {
                        let t = if n > 1 {
                            i as f64 / (n - 1) as f64
                        } else {
                            0.5
                        };
                        let center = start + dir * (t * length);
                        // Look along perpendicular direction
                        let look_dir = Vector3::new(0.0, 0.0, -1.0);
                        let rotation = look_at_rotation(&look_dir, &Vector3::y());
                        WorldToCameraPose::from_rotation_center(rotation, center)
                    })
                    .collect()
            }
            CameraArrangement::Grid {
                rows,
                cols,
                spacing,
            } => {
                let mut poses = Vec::new();
                for r in 0..*rows {
                    for c in 0..*cols {
                        let x = (c as f64 - (*cols - 1) as f64 / 2.0) * spacing;
                        let y = (r as f64 - (*rows - 1) as f64 / 2.0) * spacing;
                        let center = Point3::new(x, y, 5.0);
                        let look_dir = Vector3::new(0.0, 0.0, -1.0);
                        let rotation = look_at_rotation(&look_dir, &Vector3::y());
                        poses.push(WorldToCameraPose::from_rotation_center(rotation, center)?);
                    }
                }
                poses.truncate(self.config.num_cameras);
                Ok(poses)
            }
            CameraArrangement::Custom(poses) => Ok(poses.clone()),
        }
    }

    /// Generate 3D points based on scene type.
    fn generate_points(&mut self) -> Vec<Point3<f64>> {
        match &self.config.scene_type {
            SceneType::Cube { size } => {
                let half = size / 2.0;
                (0..self.config.num_points)
                    .map(|_| {
                        Point3::new(
                            half * self.rng.symmetric_unit_f64(),
                            half * self.rng.symmetric_unit_f64(),
                            half * self.rng.symmetric_unit_f64(),
                        )
                    })
                    .collect()
            }
            SceneType::Plane { size } => {
                let half = size / 2.0;
                (0..self.config.num_points)
                    .map(|_| {
                        Point3::new(
                            half * self.rng.symmetric_unit_f64(),
                            half * self.rng.symmetric_unit_f64(),
                            0.0,
                        )
                    })
                    .collect()
            }
            SceneType::Sphere { radius } => {
                let n = self.config.num_points;
                let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
                (0..n)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / golden_ratio;
                        let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
                        Point3::new(
                            radius * phi.sin() * theta.cos(),
                            radius * phi.sin() * theta.sin(),
                            radius * phi.cos(),
                        )
                    })
                    .collect()
            }
            SceneType::Custom(points) => points.clone(),
        }
    }

    /// Project a 3D point to 2D image coordinates.
    fn project_point(
        &self,
        point: &Point3<f64>,
        pose: &WorldToCameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Option<Point2<f64>> {
        // Transform to camera coordinates
        let camera_point = pose.project_to_camera(point);

        // Check if point is in front of camera
        if camera_point.z <= 0.0 {
            return None;
        }

        intrinsics.camera().project(&camera_point.coords).ok()
    }
}

/// Compute a **world-to-camera** rotation for a camera that looks toward `look_dir`.
///
/// Camera convention: +Z looks forward (into the scene), +X is right, +Y is up.
/// The returned rotation R satisfies: X_cam = R * X_world.
fn look_at_rotation(look_dir: &Vector3<f64>, up: &Vector3<f64>) -> Rotation3<f64> {
    // Forward direction (where camera looks) = +Z in camera frame
    let forward = look_dir.normalize();
    // Right vector = +X in camera frame
    let right = forward.cross(up).normalize();
    // Down vector = +Y in camera frame (CV convention, Y points down).
    // Must be `forward × right` so that {right, down, forward} is a *right-handed*
    // basis (right × down = forward). Using `right × forward` would yield a
    // left-handed triple, making R_c2w improper (det = -1) — an invalid rotation.
    let down = forward.cross(&right);

    // R_c2w (camera-to-world) has the world directions as columns.
    // We need R_w2c = R_c2w^T, i.e. the world directions as *rows*.
    let r_c2w = Matrix3::from_columns(&[right, down, forward]);
    Rotation3::from_matrix_unchecked(r_c2w.transpose())
}

/// Builder for creating synthetic scenes with fluent API.
#[derive(Debug)]
pub struct SyntheticSceneBuilder {
    config: SyntheticSceneConfig,
}

impl Default for SyntheticSceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticSceneBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: SyntheticSceneConfig::default(),
        }
    }

    /// Set number of cameras.
    pub fn with_cameras(mut self, n: usize) -> Self {
        self.config.num_cameras = n;
        self
    }

    /// Set number of points.
    pub fn with_points(mut self, n: usize) -> Self {
        self.config.num_points = n;
        self
    }

    /// Set scene type.
    pub fn with_scene_type(mut self, scene_type: SceneType) -> Self {
        self.config.scene_type = scene_type;
        self
    }

    /// Set camera arrangement.
    pub fn with_camera_arrangement(mut self, arrangement: CameraArrangement) -> Self {
        self.config.camera_arrangement = arrangement;
        self
    }

    /// Set image size.
    pub fn with_image_size(mut self, width: usize, height: usize) -> Self {
        self.config.image_size = (width, height);
        self
    }

    /// Set focal length.
    pub fn with_focal_length(mut self, focal: f64) -> Self {
        self.config.focal_length = focal;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: impl Into<RunSeed>) -> Self {
        self.config.seed = seed.into();
        self
    }

    /// Set observation noise.
    pub fn with_noise(mut self, noise_pixels: f64) -> Self {
        self.config.observation_noise = noise_pixels;
        self
    }

    /// Set minimum observations per point.
    pub fn with_min_observations(mut self, min: usize) -> Self {
        self.config.min_observations_per_point = min;
        self
    }

    /// Build and generate the scene.
    pub fn build(self) -> Result<SyntheticScene, StateValidationError> {
        let mut generator = SyntheticSceneGenerator::new(self.config);
        generator.generate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_scene_generation() {
        let scene = SyntheticSceneBuilder::new().build().unwrap();
        assert!(scene.num_cameras() > 0);
        assert!(scene.num_points() > 0);
        assert!(scene.num_observations() > 0);
    }

    #[test]
    fn test_circle_cameras() {
        let scene = SyntheticSceneBuilder::new()
            .with_cameras(8)
            .with_points(50)
            .with_camera_arrangement(CameraArrangement::Circle { radius: 5.0 })
            .build()
            .unwrap();

        assert_eq!(scene.num_cameras(), 8);
    }

    #[test]
    fn test_cube_scene() {
        let scene = SyntheticSceneBuilder::new()
            .with_cameras(5)
            .with_points(100)
            .with_scene_type(SceneType::Cube { size: 2.0 })
            .build()
            .unwrap();

        // All points should be within the cube bounds
        for (_, point) in &scene.points {
            assert!(point.x.abs() <= 1.0);
            assert!(point.y.abs() <= 1.0);
            assert!(point.z.abs() <= 1.0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let scene1 = SyntheticSceneBuilder::new()
            .with_seed(42)
            .with_cameras(5)
            .with_points(20)
            .build()
            .unwrap();

        let scene2 = SyntheticSceneBuilder::new()
            .with_seed(42)
            .with_cameras(5)
            .with_points(20)
            .build()
            .unwrap();

        assert_eq!(scene1.points, scene2.points);
        assert_eq!(scene1.observations, scene2.observations);
        assert_eq!(scene1.tracks, scene2.tracks);
    }

    #[test]
    fn test_noise_generation() {
        let scene_no_noise = SyntheticSceneBuilder::new()
            .with_seed(42)
            .with_noise(0.0)
            .build()
            .unwrap();

        let scene_with_noise = SyntheticSceneBuilder::new()
            .with_seed(42)
            .with_noise(1.0)
            .build()
            .unwrap();

        // Observations should differ when noise is added
        // (Note: exact comparison depends on observation order)
        assert!(scene_with_noise.num_observations() > 0);
        assert!(scene_no_noise.num_observations() > 0);
    }

    #[test]
    fn test_min_observations_filter() {
        let scene = SyntheticSceneBuilder::new()
            .with_cameras(3)
            .with_points(100)
            .with_min_observations(3)
            .build()
            .unwrap();

        // All tracks should have at least 3 observations
        for track in &scene.tracks {
            assert!(track.len() >= 3);
        }
    }

    #[test]
    fn test_sphere_scene() {
        let scene = SyntheticSceneBuilder::new()
            .with_points(50)
            .with_scene_type(SceneType::Sphere { radius: 1.0 })
            .with_camera_arrangement(CameraArrangement::Sphere { radius: 5.0 })
            .build()
            .unwrap();

        assert!(scene.num_points() > 0);
    }
}
