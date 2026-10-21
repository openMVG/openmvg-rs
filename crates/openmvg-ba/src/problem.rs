// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "faer")]
use std::collections::BTreeSet;

use nalgebra::{Point2, Point3};
use openmvg_core::{
    AngularThreshold, PixelThreshold,
    model::{CameraIntrinsics, WorldToCameraPose},
};

use crate::numerics::refine::Loss;

/// Pose coordinates exposed to optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoseVariables {
    rotation: bool,
    translation: bool,
}

config_accessors!(PoseVariables {
    /// Whether the rotation is optimized.
    copy rotation, with_rotation: bool;
    /// Whether the world camera center is optimized; a held center stays fixed when rotation changes.
    copy translation, with_translation: bool;
});

impl PoseVariables {
    /// Rotation and translation are both optimized.
    pub const FREE: Self = Self {
        rotation: true,
        translation: true,
    };
    /// Neither the rotation nor the center is optimized: the pose is fixed.
    pub const FIXED: Self = Self {
        rotation: false,
        translation: false,
    };

    /// Whether the pose is fixed.
    #[must_use]
    pub const fn is_fixed(self) -> bool {
        !self.rotation && !self.translation
    }
}

/// Intrinsic coordinates exposed to optimization; every coordinate is fixed by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntrinsicVariables {
    focal: bool,
    principal_point: bool,
    radial: bool,
    tangential: bool,
}

config_accessors!(IntrinsicVariables {
    /// Whether the focal length is optimized.
    copy focal, with_focal: bool;
    /// Whether the principal point is optimized.
    copy principal_point, with_principal_point: bool;
    /// Whether the radial distortion coefficients are optimized.
    copy radial, with_radial: bool;
    /// Whether the tangential distortion coefficients are optimized.
    copy tangential, with_tangential: bool;
});

impl IntrinsicVariables {
    /// No intrinsic coordinate is optimized: the block is fixed.
    pub const FIXED: Self = Self {
        focal: false,
        principal_point: false,
        radial: false,
        tangential: false,
    };

    /// Whether every intrinsic coordinate is fixed.
    #[must_use]
    pub const fn is_fixed(self) -> bool {
        !self.focal && !self.principal_point && !self.radial && !self.tangential
    }
}

/// One indexed pose block.
#[derive(Clone, Debug, PartialEq)]
pub struct BundlePose {
    pose: WorldToCameraPose,
    variables: PoseVariables,
}

impl BundlePose {
    /// Construct one physical pose block from its initial state and variable mask.
    /// Observations select their calibration independently.
    #[must_use]
    pub const fn new(pose: WorldToCameraPose, variables: PoseVariables) -> Self {
        Self { pose, variables }
    }

    /// Initial world-to-camera pose.
    #[must_use]
    pub const fn pose(&self) -> &WorldToCameraPose {
        &self.pose
    }

    /// Pose coordinates exposed to optimization.
    #[must_use]
    pub const fn variables(&self) -> PoseVariables {
        self.variables
    }
}

/// Whether self-calibration must show that an intrinsic block is constrained by its observations
/// before the solution may change it.
///
/// A required block whose focal or distortion remains uncertain after the solve keeps its input
/// values. A waived block moves wherever the observations take it; only a block whose uncertainty
/// cannot be evaluated at all is held. Waive it only for values that are placeholders, such as a
/// focal length guessed from the image size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntrinsicCertification {
    /// Hold the block at its input values unless the observations constrain it. This is the default.
    #[default]
    Required,
    /// Let the block move without certification.
    Waived,
}

/// One indexed camera-intrinsic block and its variable mask.
///
/// The [`IntrinsicVariables`] mask decides what a solve may change; [`IntrinsicVariables::FIXED`]
/// alone holds the whole block. The fixed flag of the [`CameraIntrinsics`] itself
/// ([`CameraIntrinsics::with_fixed`]) marks a calibration as known: intrinsics with that flag
/// must use [`IntrinsicVariables::FIXED`], and [`Self::try_new`] rejects any other mask.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleIntrinsic {
    intrinsics: CameraIntrinsics,
    variables: IntrinsicVariables,
    certification: IntrinsicCertification,
}

impl BundleIntrinsic {
    /// Construct an intrinsic block.
    ///
    /// Fails with [`ProblemError::FixedIntrinsicIsVariable`] when the intrinsics carry the fixed
    /// flag ([`CameraIntrinsics::is_fixed`]) but `variables` lets the solve change them.
    pub fn try_new(
        intrinsics: CameraIntrinsics,
        variables: IntrinsicVariables,
    ) -> Result<Self, ProblemError> {
        if intrinsics.is_fixed() && !variables.is_fixed() {
            return Err(ProblemError::FixedIntrinsicIsVariable);
        }
        Ok(Self {
            intrinsics,
            variables,
            certification: IntrinsicCertification::Required,
        })
    }

    /// The same block with `certification` as its self-calibration policy.
    #[must_use]
    pub const fn with_certification(mut self, certification: IntrinsicCertification) -> Self {
        self.certification = certification;
        self
    }

    /// Whether self-calibration must certify this block before changing it.
    #[must_use]
    pub const fn certification(&self) -> IntrinsicCertification {
        self.certification
    }

    /// Initial camera intrinsics.
    #[must_use]
    pub const fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// Intrinsic coordinates exposed to optimization.
    #[must_use]
    pub const fn variables(&self) -> IntrinsicVariables {
        self.variables
    }
}

/// One indexed Euclidean landmark block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundlePoint {
    position: Point3<f64>,
    fixed: bool,
}

impl BundlePoint {
    /// Construct a finite point block; the point is optimized unless fixed.
    pub fn try_new(position: Point3<f64>) -> Result<Self, ProblemError> {
        if !position.coords.iter().all(|value| value.is_finite()) {
            return Err(ProblemError::NonFinitePoint);
        }
        Ok(Self {
            position,
            fixed: false,
        })
    }

    /// Set whether the point is fixed, as a control point whose position is known.
    #[must_use]
    pub const fn with_fixed(mut self, fixed: bool) -> Self {
        self.fixed = fixed;
        self
    }

    /// Initial world-frame position.
    #[must_use]
    pub const fn position(&self) -> &Point3<f64> {
        &self.position
    }

    /// Whether the point is fixed.
    #[must_use]
    pub const fn is_fixed(&self) -> bool {
        self.fixed
    }
}

/// One independent pixel measurement joining pose, intrinsic and point blocks.
///
/// Multiple exposures may measure the same point at one physical pose, with
/// either the same or different calibration. Every row contributes separately
/// to the objective and information; measurement identity belongs to the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleObservation {
    pose: usize,
    intrinsic: usize,
    point: usize,
    pixel: Point2<f64>,
}

impl BundleObservation {
    /// Construct an observation with finite pixel coordinates.
    pub fn try_new(
        pose: usize,
        intrinsic: usize,
        point: usize,
        pixel: Point2<f64>,
    ) -> Result<Self, ProblemError> {
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return Err(ProblemError::NonFiniteObservation);
        }
        Ok(Self {
            pose,
            intrinsic,
            point,
            pixel,
        })
    }

    /// Index of the observing pose in [`BundleProblem::poses`].
    #[must_use]
    pub const fn pose(&self) -> usize {
        self.pose
    }

    /// Calibration of this exposure in [`BundleProblem::intrinsics`].
    #[must_use]
    pub const fn intrinsic(&self) -> usize {
        self.intrinsic
    }

    /// Index of the observed point in [`BundleProblem::points`].
    #[must_use]
    pub const fn point(&self) -> usize {
        self.point
    }

    /// Measured image position, in pixels.
    #[must_use]
    pub const fn pixel(&self) -> &Point2<f64> {
        &self.pixel
    }
}

/// Residual domain used by every observation in one problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidualSpace {
    /// Reprojection error: projected point minus measured pixel, in pixels.
    Pixels,
    /// Tangent-plane residual at the measured pixel's bearing whose norm is the angle, in
    /// radians, between that bearing and the direction to the point.
    Angular,
}

/// A positive robust-loss scale carrying its residual unit.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ResidualScale {
    /// Scale of [`ResidualSpace::Pixels`] residuals.
    Pixels(PixelThreshold),
    /// Scale of [`ResidualSpace::Angular`] residuals, in radians.
    Angular(AngularThreshold),
}

impl ResidualScale {
    /// Residual space whose unit this scale carries.
    #[must_use]
    pub const fn space(self) -> ResidualSpace {
        match self {
            Self::Pixels(_) => ResidualSpace::Pixels,
            Self::Angular(_) => ResidualSpace::Angular,
        }
    }

    /// Scale magnitude, in pixels or radians.
    #[must_use]
    pub const fn value(self) -> f64 {
        match self {
            Self::Pixels(value) => value.value(),
            Self::Angular(value) => value.value(),
        }
    }
}

/// Robust objective applied uniformly to one problem's residuals.
///
/// [`Huber`](Self::Huber) suits residuals that are noisy but rarely wrong: it keeps every
/// observation and only caps how hard a large residual pulls, since beyond the scale the pull
/// stays constant and never falls. [`Cauchy`](Self::Cauchy) suits observations that may include
/// wrong matches: the weight of a residual far beyond the scale falls toward zero, so a match
/// tens of pixels off barely moves the solution.
///
/// Set the scale to two to four times the per-axis pixel noise of good observations: the
/// `robust_loss_and_gauge` example uses 2 px for 0.5 px of noise. The scale also decides which
/// observations the report counts as inliers
/// ([`SolveReport::inlier_count`](crate::SolveReport::inlier_count)), and most good residuals
/// exceed the noise itself, so a scale equal to the noise would count many good observations as
/// outliers. For nearly exact pixels, keep about 1 px rather than a tiny scale, so that rounding
/// and small model errors still count as inliers.
///
/// The scale also decides which observations count as good at the start. Under Cauchy, the
/// native adjuster first runs a few least-squares iterations (a quarter of the iteration limit,
/// at most five) to move toward the minimum, then switches to the robust loss; Huber applies
/// from the first iteration. If good observations still sit several scales away after that, as
/// when the start is far off, the loss treats them as wrong matches: solve once with a wider
/// scale, then again from that result with the noise scale.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum RobustLoss {
    /// Least squares on each observation's residual norm `r`: `ρ(r) = r²/2`.
    #[default]
    Squared,
    /// Huber loss: `r²/2` up to the scale, linear beyond it.
    Huber(ResidualScale),
    /// Cauchy loss `ln(1 + r²/scale²)`: an observation whose residual norm equals the scale
    /// gets half the weight of a residual near zero. The Cauchy loss of
    /// [`numerics::refine`](crate::numerics::refine) writes the same curve as
    /// `ln(1 + (r/s)²/2)`: to match a scale here, give it `s = scale/√2`.
    Cauchy(ResidualScale),
}

impl RobustLoss {
    /// Loss scale; `None` for [`RobustLoss::Squared`].
    #[must_use]
    pub const fn scale(self) -> Option<ResidualScale> {
        match self {
            Self::Squared => None,
            Self::Huber(scale) | Self::Cauchy(scale) => Some(scale),
        }
    }

    /// The kernel loss applying this objective; a Cauchy scale maps to the kernel's `scale / √2`.
    pub(crate) fn kernel_loss(self) -> Loss {
        match self {
            Self::Squared => Loss::L2,
            Self::Huber(scale) => Loss::Huber {
                delta: scale.value(),
            },
            Self::Cauchy(scale) => Loss::Cauchy {
                scale: scale.value() / 2.0_f64.sqrt(),
            },
        }
    }
}

/// Gauge constraint represented by the problem.
///
/// Moving, rotating or uniformly scaling every pose and point together changes no residual, so
/// observations alone leave seven directions undetermined: three of position, three of
/// orientation and one of scale. Fixing them picks one solution out of a family of equally good
/// ones.
///
/// - With [`Gauge::Free`], fixed poses, fixed camera centers and fixed world points fix what
///   they can, and the native adjuster completes the rest from the starting values: it holds an
///   anchor pose and, when nothing fixes the scale, the anchor's distance to the free camera
///   farthest from it. The returned frame and scale then follow the start. Among the poses that
///   have observations, the anchor is the first fixed pose, else the first that holds its
///   center, else the first that holds its rotation, else the first one. When no pose holds its
///   center, the anchor is held whole, rotation and center. Observed fixed world points (control
///   points) set the frame themselves: with fixed intrinsics and no fixed pose, no pose is held;
///   with free intrinsics, the solve returns `Unsupported` unless an observed pose is fixed.
/// - [`Gauge::FixedBaseline`] holds pose `first` at its input value, whatever its
///   [`PoseVariables`] say, and sets the distance between the centers of `first` and `second` to
///   `distance`, which gives the result a known scale. When `distance` differs from the starting
///   distance, the adjuster first rescales the whole start; that is unsupported when a pose
///   other than `first` holds its center or a world point is fixed, since rescaling would move
///   them, and the solve returns [`SolveError::Unsupported`](crate::SolveError::Unsupported), as
///   it does when the two cameras share one center. Pose `first` does not count: the gauge holds
///   it anyway. The native adjuster also returns `Unsupported` for a rescale when it quarantines
///   any point, since it cannot move a point it leaves at its input position. Far from the world
///   origin, rounding can keep the solved distance from matching `distance` closely enough, and
///   the solve fails with a numerical error; express the scene near the origin.
///
/// The report's [`Observability`](crate::Observability) describes the camera system once the
/// gauge is fixed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum Gauge {
    /// Add no constraint of its own. Fixed poses, centers or world points determine what they
    /// can, and the native adjuster holds the rest at its starting value (see the type docs).
    #[default]
    Free,
    /// Pose `first` anchors the gauge and the distance between the centers of `first`
    /// and `second` is held at `distance`.
    FixedBaseline {
        /// Index of the anchor pose in [`BundleProblem::poses`].
        first: usize,
        /// Index of the pose whose center distance to `first` is held; distinct from
        /// `first`.
        second: usize,
        /// Center-to-center distance in world units; finite and positive.
        distance: f64,
    },
}

/// A fully owned, checked, index-based bundle-adjustment problem.
///
/// Construct one with [`BundleProblem::builder`]. A problem holds the initial state, which a
/// solve never changes, and the objective: the [`ResidualSpace`], the [`RobustLoss`] and the
/// [`Gauge`].
///
/// Each [`BundlePose`] is a physical world-to-camera pose, independent of lenses. Each
/// [`BundleObservation`] selects its pose, its intrinsic group and its world point by index, so
/// several views can share one pose while using different fixed lenses. Distinct pose indices
/// stay distinct even when their initial values are equal. Observations form an ordered
/// multiset: a repeated measurement contributes a repeated term to the objective.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleProblem {
    poses: Vec<BundlePose>,
    intrinsics: Vec<BundleIntrinsic>,
    points: Vec<BundlePoint>,
    observations: Vec<BundleObservation>,
    residual_space: ResidualSpace,
    robust_loss: RobustLoss,
    gauge: Gauge,
}

/// Unchecked parts of a [`BundleProblem`]; [`BundleProblemBuilder::build`] validates them.
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct BundleProblemBuilder {
    poses: Vec<BundlePose>,
    intrinsics: Vec<BundleIntrinsic>,
    points: Vec<BundlePoint>,
    observations: Vec<BundleObservation>,
    residual_space: ResidualSpace,
    robust_loss: RobustLoss,
    gauge: Gauge,
}

impl BundleProblemBuilder {
    /// Residual domain of every observation; [`ResidualSpace::Pixels`] by default.
    pub const fn with_residual_space(mut self, residual_space: ResidualSpace) -> Self {
        self.residual_space = residual_space;
        self
    }

    /// Robust objective; [`RobustLoss::Squared`] by default.
    pub const fn with_robust_loss(mut self, robust_loss: RobustLoss) -> Self {
        self.robust_loss = robust_loss;
        self
    }

    /// Gauge constraint; [`Gauge::Free`] by default.
    pub const fn with_gauge(mut self, gauge: Gauge) -> Self {
        self.gauge = gauge;
        self
    }

    /// Validate and take ownership of one complete bundle adjustment problem.
    ///
    /// Every check runs here, so the order of the `with_*` calls does not matter.
    pub fn build(self) -> Result<BundleProblem, ProblemError> {
        let Self {
            poses,
            intrinsics,
            points,
            observations,
            residual_space,
            robust_loss,
            gauge,
        } = self;
        if poses.is_empty() {
            return Err(ProblemError::EmptyPoses);
        }
        if intrinsics.is_empty() {
            return Err(ProblemError::EmptyIntrinsics);
        }
        if points.is_empty() {
            return Err(ProblemError::EmptyPoints);
        }
        if observations.is_empty() {
            return Err(ProblemError::EmptyObservations);
        }
        for (pose, block) in poses.iter().enumerate() {
            block
                .pose
                .validate()
                .map_err(|_| ProblemError::InvalidPose { pose })?;
        }
        for (observation, value) in observations.iter().enumerate() {
            if value.pose >= poses.len() {
                return Err(ProblemError::ObservationPoseOutOfBounds {
                    observation,
                    pose: value.pose,
                    pose_count: poses.len(),
                });
            }
            if value.point >= points.len() {
                return Err(ProblemError::ObservationPointOutOfBounds {
                    observation,
                    point: value.point,
                    point_count: points.len(),
                });
            }
            if value.intrinsic >= intrinsics.len() {
                return Err(ProblemError::IntrinsicIndexOutOfBounds {
                    observation,
                    intrinsic: value.intrinsic,
                    intrinsic_count: intrinsics.len(),
                });
            }
        }
        if let Some(scale) = robust_loss.scale()
            && scale.space() != residual_space
        {
            return Err(ProblemError::LossUnitMismatch);
        }
        match gauge {
            Gauge::FixedBaseline {
                first,
                second,
                distance,
            } => {
                if first >= poses.len() || second >= poses.len() {
                    return Err(ProblemError::GaugePoseOutOfBounds);
                }
                if first == second {
                    return Err(ProblemError::GaugeUsesOnePose);
                }
                if !distance.is_finite() || distance <= 0.0 {
                    return Err(ProblemError::InvalidBaselineDistance);
                }
            }
            Gauge::Free => {}
        }

        Ok(BundleProblem {
            poses,
            intrinsics,
            points,
            observations,
            residual_space,
            robust_loss,
            gauge,
        })
    }
}

impl BundleProblem {
    /// Start a problem from its required blocks.
    ///
    /// The residual space, robust loss and gauge take their documented defaults
    /// unless set on the returned builder. Nothing is checked until
    /// [`BundleProblemBuilder::build`].
    pub fn builder(
        poses: Vec<BundlePose>,
        intrinsics: Vec<BundleIntrinsic>,
        points: Vec<BundlePoint>,
        observations: Vec<BundleObservation>,
    ) -> BundleProblemBuilder {
        BundleProblemBuilder {
            poses,
            intrinsics,
            points,
            observations,
            residual_space: ResidualSpace::Pixels,
            robust_loss: RobustLoss::default(),
            gauge: Gauge::default(),
        }
    }

    /// Pose blocks, indexed by [`BundleObservation::pose`].
    #[must_use]
    pub fn poses(&self) -> &[BundlePose] {
        &self.poses
    }

    /// Intrinsic blocks, indexed by [`BundleObservation::intrinsic`].
    #[must_use]
    pub fn intrinsics(&self) -> &[BundleIntrinsic] {
        &self.intrinsics
    }

    /// Point blocks, indexed by [`BundleObservation::point`].
    #[must_use]
    pub fn points(&self) -> &[BundlePoint] {
        &self.points
    }

    /// Observations, in input order. A repeated observation is not rejected: it adds its term to
    /// the objective again, weighting that measurement twice.
    #[must_use]
    pub fn observations(&self) -> &[BundleObservation] {
        &self.observations
    }

    /// Residual domain of every observation.
    #[must_use]
    pub const fn residual_space(&self) -> ResidualSpace {
        self.residual_space
    }

    /// Robust objective applied to every residual.
    #[must_use]
    pub const fn robust_loss(&self) -> RobustLoss {
        self.robust_loss
    }

    /// Gauge constraint.
    #[must_use]
    pub const fn gauge(&self) -> Gauge {
        self.gauge
    }

    /// This problem without the `dropped` points and their observations, optionally starting from
    /// a solution's state instead of this problem's initial state.
    ///
    /// Returns the restricted problem and, for each of its points, the index of that point here.
    #[cfg(feature = "faer")]
    pub(crate) fn without_points(
        &self,
        warm_start: Option<&crate::Solution>,
        dropped: &BTreeSet<usize>,
    ) -> Result<(Self, Vec<usize>), ProblemError> {
        let poses = self
            .poses
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let pose = warm_start.map_or(&block.pose, |solution| &solution.poses()[index]);
                BundlePose::new(pose.clone(), block.variables)
            })
            .collect();
        let intrinsics = self
            .intrinsics
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let value =
                    warm_start.map_or(&block.intrinsics, |solution| &solution.intrinsics()[index]);
                BundleIntrinsic::try_new(value.clone(), block.variables)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut new_index = vec![usize::MAX; self.points.len()];
        let mut kept = Vec::with_capacity(self.points.len().saturating_sub(dropped.len()));
        let mut points = Vec::with_capacity(kept.capacity());
        for (index, block) in self.points.iter().enumerate() {
            if dropped.contains(&index) {
                continue;
            }
            new_index[index] = kept.len();
            kept.push(index);
            let position = warm_start.map_or(&block.position, |solution| &solution.points()[index]);
            points.push(BundlePoint::try_new(*position)?.with_fixed(block.fixed));
        }
        let observations = self
            .observations
            .iter()
            .filter(|observation| new_index[observation.point] != usize::MAX)
            .map(|observation| BundleObservation {
                point: new_index[observation.point],
                ..*observation
            })
            .collect();
        let restricted = Self::builder(poses, intrinsics, points, observations)
            .with_residual_space(self.residual_space)
            .with_robust_loss(self.robust_loss)
            .with_gauge(self.gauge)
            .build()?;
        Ok((restricted, kept))
    }
}

/// Invalid state at the checked bundle adjustment (BA) problem boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProblemError {
    /// The pose list is empty.
    #[error("bundle adjustment requires at least one pose")]
    EmptyPoses,
    /// The intrinsic list is empty.
    #[error("bundle adjustment requires at least one intrinsic")]
    EmptyIntrinsics,
    /// The point list is empty.
    #[error("bundle adjustment requires at least one point")]
    EmptyPoints,
    /// The observation list is empty.
    #[error("bundle adjustment requires at least one observation")]
    EmptyObservations,
    /// A pose is not a valid rigid transform.
    #[error("pose {pose} is not a valid rigid transform")]
    InvalidPose {
        /// Index of the pose.
        pose: usize,
    },
    /// A point has a non-finite coordinate.
    #[error("point coordinates must be finite")]
    NonFinitePoint,
    /// An observation has a non-finite pixel coordinate.
    #[error("observation pixels must be finite")]
    NonFiniteObservation,
    /// A fixed intrinsic was given variable coordinates.
    #[error("a fixed intrinsic cannot expose variable coordinates")]
    FixedIntrinsicIsVariable,
    /// An observation references a missing intrinsic block.
    #[error(
        "observation {observation} references intrinsic {intrinsic}, but only {intrinsic_count} exist"
    )]
    IntrinsicIndexOutOfBounds {
        /// Index of the observation.
        observation: usize,
        /// Referenced intrinsic index.
        intrinsic: usize,
        /// Number of intrinsic blocks in the problem.
        intrinsic_count: usize,
    },
    /// An observation references a missing pose.
    #[error("observation {observation} references pose {pose}, but only {pose_count} exist")]
    ObservationPoseOutOfBounds {
        /// Index of the observation.
        observation: usize,
        /// Referenced pose index.
        pose: usize,
        /// Number of poses in the problem.
        pose_count: usize,
    },
    /// An observation references a missing point.
    #[error("observation {observation} references point {point}, but only {point_count} exist")]
    ObservationPointOutOfBounds {
        /// Index of the observation.
        observation: usize,
        /// Referenced point index.
        point: usize,
        /// Number of points in the problem.
        point_count: usize,
    },
    /// The robust-loss scale unit does not match the problem's [`ResidualSpace`].
    #[error("robust-loss scale units do not match the problem residual space")]
    LossUnitMismatch,
    /// A [`Gauge::FixedBaseline`] pose index is out of range.
    #[error("fixed-baseline gauge references an out-of-range pose")]
    GaugePoseOutOfBounds,
    /// A [`Gauge::FixedBaseline`] names the same pose twice.
    #[error("fixed-baseline gauge requires two distinct poses")]
    GaugeUsesOnePose,
    /// A [`Gauge::FixedBaseline`] distance is not finite and positive.
    #[error("fixed-baseline distance must be finite and positive")]
    InvalidBaselineDistance,
}
