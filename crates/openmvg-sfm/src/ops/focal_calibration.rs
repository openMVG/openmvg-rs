// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Focal pre-calibration of camera groups whose focal length is a guess.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Focal pre-calibration (calibrate_guessed_focals_with_progress)      │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: matched pairs, before verification; camera groups with a guessed
//!        focal (IntrinsicPrior::Guess), a perspective model without known
//!        distortion, and at least two views
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Fundamental matrix per same-group pair                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! 7-point GenRANSAC on the raw matches, 4 px Sampson threshold; a pair
//! counts when F keeps at least 60 inliers. F needs no intrinsics.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Cost curve per pair over a focal grid                        │
//! └──────────────────────────────────────────────────────────────────────┘
//! 555 focal lengths from guess/4 to guess·4, 0.5 % apart, principal point
//! of the group; E(f) = K(f)ᵀ F K(f) must have two equal singular values,
//! so the cost is (σ₁ − σ₂) / (σ₁ + σ₂). A pair counts only when its curve
//! falls to 0.02 or less somewhere: otherwise F was fitted to wrong matches.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Group estimate                                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! Median curve over the group's n pairs; its minimum is accepted when it
//! is interior to the grid and lowers the cost of more than n/2 + 3·√n/2
//! pairs compared with the guess (one-sided sign test).
//!   │
//!   ▼
//! Output: the accepted focal replaces the guess in the group's camera;
//!         the prior stays a guess, and verification runs afterwards
//! ```
//!
//! A focal length guessed from the image size (1.2 times the longest edge when an image has no
//! focal metadata) can be tens of percent away from the lens. Verification fits a calibrated
//! essential matrix, so that error enters every pair: relative rotations bend, Global rotation
//! averaging finds no consistent subgraph, and radial pre-calibration absorbs the focal error into
//! `k1`. A fundamental matrix needs no intrinsics; for one camera group shared by both images,
//! the focal length that makes `Kᵀ F K` an essential matrix is the group's focal length. This is
//! the self-calibration criterion of Mendonça and Cipolla, "A simple technique for
//! self-calibration", CVPR 1999, evaluated as a median over many pairs of the same camera.
//!
//! Only pairs inside one group constrain its focal length. A camera with one view per group
//! (images without EXIF and without `share_unknown_cameras`) is left to registration and bundle
//! adjustment. Pairs whose two views fixate one point, or that only rotate, give flat or
//! misleading curves; the median and the sign test keep a group at its guess unless most pairs
//! agree.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Point2};
use openmvg_core::camera::{
    BrownConradyCamera, PinholeCamera, RadialDistortion, TangentialDistortion,
};
use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::fundamental::estimate_fundamental;
use openmvg_geometry::robust::RobustOptions;
use openmvg_geometry::robust::threshold::PixelThreshold;

use crate::determinism::FOCAL_CALIBRATION_SEED_DOMAIN;
use crate::model::IntrinsicPrior;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::reports::{FocalCalibrationReport, FocalGroupEstimate};
use crate::{Camera, IntrinsicId, PairId, Result, SfmError, SfmWorkspace};

/// Spacing of the focal grid, as a natural-log step: neighbouring candidates differ by 0.5 %.
const LOG_FOCAL_STEP: f64 = 0.005;
/// Grid steps on each side of the guess: exp(277 · 0.005) ≈ 4, so the grid spans guess/4 to
/// guess·4, from a fisheye-like short focal to a telephoto.
const HALF_GRID_STEPS: i32 = 277;
const MIN_PAIR_INLIERS: usize = 60;
/// A pair counts only when its fundamental matrix is an essential matrix, to within this
/// singular-value inequality, at some focal length of the grid. Correct pairs reach 1e-4 to 1e-2;
/// a fundamental matrix fitted to wrong matches stays above 3e-2 at every focal length.
const MAX_PAIR_COST: f64 = 0.02;
/// One-sided sign-test margin in binomial standard deviations, as in radial pre-calibration.
const SIGN_TEST_SIGMAS: f64 = 3.0;

struct GuessedFocalGroup {
    intrinsic_id: IntrinsicId,
    camera: Camera,
    focal: f64,
    principal: (f64, f64),
}

/// Estimate and adopt the focal length of every camera group whose focal is a guess, reporting
/// [`SfmProgressStage::FocalCalibration`] in camera groups.
///
/// Requires matched pairs and must run before verification: it changes the cameras that
/// verification fits essential matrices with. Groups whose evidence is not significant keep
/// their guess.
pub(crate) fn calibrate_guessed_focals_with_progress(
    workspace: &mut SfmWorkspace,
    progress: &dyn SfmProgressSink,
) -> Result<FocalCalibrationReport> {
    if workspace.geometric_model_store().populated_count() != 0 {
        return Err(SfmError::InvalidInput(
            "focal pre-calibration must run before any pair is verified".to_owned(),
        ));
    }
    let groups = guessed_focal_groups(workspace);
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::FocalCalibration, groups.len());
    let estimates = estimate_focals(workspace, &groups);
    let replacements = groups
        .iter()
        .zip(&estimates)
        .filter(|(_, estimate)| estimate.accepted)
        .map(|(group, estimate)| {
            Ok((
                group.intrinsic_id,
                with_focal(&group.camera, estimate.focal)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    workspace.replace_camera_model_groups(replacements)?;
    if !groups.is_empty() {
        progress.advance(SfmProgressStage::FocalCalibration, groups.len());
    }
    progress.finish(SfmProgressStage::FocalCalibration);
    Ok(FocalCalibrationReport { groups: estimates })
}

/// Whether any camera group shared by two or more views has a guessed focal length.
pub(crate) fn has_guessed_focal(workspace: &SfmWorkspace) -> bool {
    !guessed_focal_groups(workspace).is_empty()
}

fn guessed_focal_groups(workspace: &SfmWorkspace) -> Vec<GuessedFocalGroup> {
    let mut view_counts = BTreeMap::<IntrinsicId, usize>::new();
    for view in workspace.views() {
        *view_counts.entry(view.intrinsic_id).or_default() += 1;
    }
    workspace
        .camera_models()
        .iter()
        .enumerate()
        .filter_map(|(index, camera)| {
            let intrinsic_id = IntrinsicId::new(index);
            let shared = view_counts
                .get(&intrinsic_id)
                .is_some_and(|count| *count >= 2);
            let guessed = workspace.intrinsic_prior(intrinsic_id) == Some(IntrinsicPrior::Guess);
            let pinhole = undistorted_pinhole(camera)?;
            (shared && guessed).then(|| GuessedFocalGroup {
                intrinsic_id,
                camera: camera.clone(),
                focal: pinhole.focal_x(),
                principal: (pinhole.principal_x(), pinhole.principal_y()),
            })
        })
        .collect()
}

/// The pinhole part of a perspective camera with square pixels and no known distortion. The
/// fundamental matrix is fitted to raw pixels, so known distortion would bias it.
fn undistorted_pinhole(camera: &Camera) -> Option<&PinholeCamera> {
    let pinhole = match camera {
        Camera::Pinhole(pinhole) => pinhole,
        Camera::BrownConrady(brown) => {
            let zero_radial = match brown.radial_distortion() {
                RadialDistortion::None => true,
                RadialDistortion::K1(k1) => *k1 == 0.0,
                RadialDistortion::K1K2(k1, k2) => *k1 == 0.0 && *k2 == 0.0,
                RadialDistortion::K1K2K3(k1, k2, k3) => *k1 == 0.0 && *k2 == 0.0 && *k3 == 0.0,
            };
            if !zero_radial || *brown.tangential_distortion() != TangentialDistortion::None {
                return None;
            }
            brown.pinhole()
        }
        Camera::Fisheye(_) | Camera::Spherical(_) => return None,
    };
    (pinhole.focal_x() == pinhole.focal_y()).then_some(pinhole)
}

/// The same camera with another focal length; principal point and distortion are kept.
fn with_focal(camera: &Camera, focal: f64) -> Result<Camera> {
    Ok(match camera {
        Camera::Pinhole(pinhole) => Camera::Pinhole(PinholeCamera::try_with_focal(
            focal,
            pinhole.principal_x(),
            pinhole.principal_y(),
        )?),
        Camera::BrownConrady(brown) => Camera::BrownConrady(BrownConradyCamera::try_new(
            PinholeCamera::try_with_focal(focal, brown.principal_x(), brown.principal_y())?,
            *brown.radial_distortion(),
            *brown.tangential_distortion(),
        )?),
        Camera::Fisheye(_) | Camera::Spherical(_) => {
            return Err(SfmError::InvariantViolation(
                "focal pre-calibration selected a non-perspective camera".to_owned(),
            ));
        }
    })
}

fn grid_focal(guess: f64, step: i32) -> f64 {
    guess * (f64::from(step) * LOG_FOCAL_STEP).exp()
}

fn estimate_focals(
    workspace: &SfmWorkspace,
    groups: &[GuessedFocalGroup],
) -> Vec<FocalGroupEstimate> {
    let group_index: BTreeMap<IntrinsicId, usize> = groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.intrinsic_id, index))
        .collect();
    let group_of = |view| {
        workspace
            .view(view)
            .and_then(|view| group_index.get(&view.intrinsic_id).copied())
    };
    let graph = workspace.pair_graph();
    let candidates: Vec<(PairId, usize)> = graph
        .active_pair_ids()
        .filter_map(|pair_id| {
            let pair = graph.pair(pair_id)?;
            let group = group_of(pair.first())?;
            (group_of(pair.second()) == Some(group)).then_some((pair_id, group))
        })
        .collect();
    let profile = |&(pair_id, group): &(PairId, usize)| {
        pair_cost_curve(workspace, pair_id, &groups[group]).map(|curve| (group, curve))
    };
    #[cfg(feature = "parallel")]
    let curves: Vec<(usize, Vec<f64>)> = {
        use rayon::prelude::*;
        candidates.par_iter().filter_map(profile).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let curves: Vec<(usize, Vec<f64>)> = candidates.iter().filter_map(profile).collect();

    let guess_step = HALF_GRID_STEPS as usize;
    groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let curves: Vec<&[f64]> = curves
                .iter()
                .filter(|(owner, _)| *owner == index)
                .map(|(_, curve)| curve.as_slice())
                .collect();
            let pairs = curves.len();
            let not_accepted = FocalGroupEstimate {
                intrinsic_id: group.intrinsic_id,
                pair_count: pairs,
                improved_pair_count: 0,
                initial_focal: group.focal,
                focal: group.focal,
                cost_ratio: 1.0,
                accepted: false,
            };
            if pairs == 0 {
                return not_accepted;
            }
            let profile: Vec<f64> = (0..=2 * guess_step)
                .map(|step| median(curves.iter().map(|curve| curve[step]).collect()))
                .collect();
            let Some((best, best_cost)) = profile
                .iter()
                .copied()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
            else {
                return not_accepted;
            };
            let improved_pairs = curves
                .iter()
                .filter(|curve| curve[best] < curve[guess_step])
                .count();
            let chance = 0.5 * pairs as f64 + SIGN_TEST_SIGMAS * 0.5 * (pairs as f64).sqrt();
            let interior = best > 0 && best < 2 * guess_step;
            let guess_cost = profile[guess_step];
            FocalGroupEstimate {
                improved_pair_count: improved_pairs,
                focal: grid_focal(group.focal, best as i32 - HALF_GRID_STEPS),
                cost_ratio: if guess_cost > 0.0 {
                    best_cost / guess_cost
                } else {
                    1.0
                },
                accepted: interior && improved_pairs as f64 > chance,
                ..not_accepted
            }
        })
        .collect()
}

/// Mendonça–Cipolla cost of one same-group pair at every grid focal length, from a robust
/// fundamental matrix of its raw matches. `None` when F keeps too few inliers or is not an
/// essential matrix at any grid focal length.
///
/// A fundamental matrix constrains a match only to a line, so on raw matches it can gather many
/// wrong ones. Fitted that way, it is no essential matrix at any focal length: its curve has no
/// near-zero minimum, and it would pull the median toward an arbitrary focal length.
fn pair_cost_curve(
    workspace: &SfmWorkspace,
    pair_id: PairId,
    group: &GuessedFocalGroup,
) -> Option<Vec<f64>> {
    let pair = workspace.pair_graph().pair(pair_id)?;
    let matches = workspace.match_store().get(pair_id.index())?;
    if matches.len() < MIN_PAIR_INLIERS {
        return None;
    }
    let keypoints_i = workspace.keypoint_store().get(pair.first().index())?;
    let keypoints_j = workspace.keypoint_store().get(pair.second().index())?;
    let pixel = |keypoint: &openmvg_features::model::Keypoint| {
        Point2::new(f64::from(keypoint.x()), f64::from(keypoint.y()))
    };
    let correspondences: Vec<PixelCorrespondence> = matches
        .as_slice()
        .iter()
        .map(|matched| {
            Some(PixelCorrespondence::new(
                pixel(keypoints_i.get(matched.first().index())?),
                pixel(keypoints_j.get(matched.second().index())?),
            ))
        })
        .collect::<Option<_>>()?;
    let options = RobustOptions::default().with_seed(workspace.derive_seed(
        FOCAL_CALIBRATION_SEED_DOMAIN,
        &[pair.first().value(), pair.second().value()],
    ));
    let estimate =
        estimate_fundamental(&correspondences, PixelThreshold::FOUR_PIXELS, &options).ok()?;
    if estimate.inlier_count() < MIN_PAIR_INLIERS {
        return None;
    }
    let fundamental = estimate.model().matrix();
    let (cx, cy) = group.principal;
    let curve: Vec<f64> = (-HALF_GRID_STEPS..=HALF_GRID_STEPS)
        .map(|step| {
            let focal = grid_focal(group.focal, step);
            let k = Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);
            essential_inequality(&(k.transpose() * fundamental * k))
        })
        .collect();
    let minimum = curve.iter().copied().fold(f64::INFINITY, f64::min);
    (curve.iter().all(|cost| cost.is_finite()) && minimum <= MAX_PAIR_COST).then_some(curve)
}

/// `(σ₁ − σ₂) / (σ₁ + σ₂)` of a 3x3 matrix: zero for an essential matrix, whose two non-zero
/// singular values are equal, and up to one when the matrix is far from it.
fn essential_inequality(matrix: &Matrix3<f64>) -> f64 {
    let mut singular = matrix.singular_values();
    singular
        .as_mut_slice()
        .sort_unstable_by(|a, b| b.total_cmp(a));
    let sum = singular[0] + singular[1];
    if sum > 0.0 {
        (singular[0] - singular[1]) / sum
    } else {
        f64::INFINITY
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    let middle = values.len() / 2;
    values.select_nth_unstable_by(middle, f64::total_cmp);
    values[middle]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::verification::CalibratedVerifier;
    use crate::model::ImageMeta;
    use crate::two_view::PairMatches;
    use crate::workspace::selector::PairSelector;
    use crate::{FeatureId, ViewId, ViewPair};
    use nalgebra::{Rotation3, Vector3};
    use openmvg_core::camera::CameraModel;
    use openmvg_features::model::{DescriptorSet, ExtractedFeatures, FeatureMatch, Keypoint};

    const WIDTH: u32 = 1600;
    const HEIGHT: u32 = 1200;
    /// The lens: what the estimate must find.
    const FOCAL: f64 = 1200.0;
    /// The image-size guess, 1.2 times the longest edge: 60 % too long.
    const GUESS: f64 = 1.2 * 1600.0;
    const VIEWS: usize = 8;

    /// Deterministic value in [-1, 1).
    fn hash(index: usize, salt: usize) -> f64 {
        let mut x = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64);
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn camera(focal: f64) -> Camera {
        Camera::BrownConrady(
            BrownConradyCamera::try_with_focal(
                focal,
                800.0,
                600.0,
                RadialDistortion::K1(0.0),
                TangentialDistortion::None,
            )
            .unwrap(),
        )
    }

    /// Eight views on an arc around a box of points, imaged at `FOCAL` and declared with
    /// `cameras`, one group per entry of `view_intrinsics`; matched exhaustively, not verified.
    /// Each view aims at a slightly different point, as a handheld capture does.
    /// `wrong_matches` replaces every pair's matches with a shuffled set.
    fn matched_workspace(
        cameras: Vec<Camera>,
        priors: Vec<IntrinsicPrior>,
        view_intrinsics: Vec<IntrinsicId>,
        wrong_matches: bool,
    ) -> SfmWorkspace {
        let Camera::BrownConrady(lens) = camera(FOCAL) else {
            unreachable!();
        };
        let points: Vec<Vector3<f64>> = (0..800)
            .map(|p| 2.5 * Vector3::new(hash(p, 1), hash(p, 2), hash(p, 3)))
            .collect();
        let mut workspace = SfmWorkspace::builder()
            .image_meta((0..VIEWS).map(|v| ImageMeta::external_id(format!("v{v}"), WIDTH, HEIGHT)))
            .camera_models(cameras)
            .intrinsic_priors(priors)
            .view_intrinsics(view_intrinsics)
            .exhaustive_pairs()
            .build()
            .unwrap();
        let mut visible = Vec::new();
        for view in 0..VIEWS {
            let angle = -0.6 + 1.2 * view as f64 / (VIEWS - 1) as f64;
            let centre = Vector3::new(6.0 * angle.sin(), 0.5 * hash(view, 4), -6.0 * angle.cos());
            let target = 0.4 * Vector3::new(hash(view, 5), hash(view, 6), hash(view, 7));
            let rotation = Rotation3::look_at_lh(&(target - centre), &Vector3::y());
            let mut keypoints = Vec::new();
            let mut features = vec![None; points.len()];
            for (p, point) in points.iter().enumerate() {
                let Ok(pixel) = lens.project(&(rotation * (point - centre))) else {
                    continue;
                };
                let (x, y) = (
                    pixel.x + 0.3 * hash(p, view + 10),
                    pixel.y + 0.3 * hash(p, view + 20),
                );
                if (0.0..f64::from(WIDTH)).contains(&x) && (0.0..f64::from(HEIGHT)).contains(&y) {
                    features[p] = Some(keypoints.len());
                    keypoints.push(Keypoint::try_new(x as f32, y as f32, 1.0, 0.0).unwrap());
                }
            }
            let descriptors = DescriptorSet::try_u8(vec![0; keypoints.len()], 1).unwrap();
            workspace
                .set_features(
                    view,
                    ExtractedFeatures::try_new(keypoints, descriptors).unwrap(),
                )
                .unwrap();
            visible.push(features);
        }
        for i in 0..VIEWS {
            for j in i + 1..VIEWS {
                let pair = ViewPair::new(ViewId::new(i), ViewId::new(j));
                let mut matched: Vec<(usize, usize)> = visible[i]
                    .iter()
                    .zip(&visible[j])
                    .filter_map(|(a, b)| Some((*a.as_ref()?, *b.as_ref()?)))
                    .collect();
                if wrong_matches {
                    let seconds: Vec<usize> = matched.iter().map(|(_, b)| *b).collect();
                    let n = seconds.len();
                    for (k, (_, b)) in matched.iter_mut().enumerate() {
                        *b = seconds[(k * 7 + 3) % n];
                    }
                }
                let matches = matched
                    .into_iter()
                    .map(|(a, b)| {
                        FeatureMatch::try_new(FeatureId::new(a), FeatureId::new(b), None).unwrap()
                    })
                    .collect();
                let index = workspace
                    .pair_graph()
                    .pair_id(pair.first(), pair.second())
                    .unwrap();
                workspace
                    .matches
                    .set(index.index(), PairMatches::try_new(pair, matches).unwrap())
                    .unwrap();
            }
        }
        workspace
    }

    fn shared_guess() -> SfmWorkspace {
        matched_workspace(
            vec![camera(GUESS)],
            vec![IntrinsicPrior::Guess],
            vec![IntrinsicId::new(0); VIEWS],
            false,
        )
    }

    fn focal_of(camera: &Camera) -> f64 {
        camera.focal_lengths().unwrap().0
    }

    #[test]
    fn a_guessed_shared_focal_is_replaced_by_the_lens_focal() {
        let mut workspace = shared_guess();

        let report = calibrate_guessed_focals_with_progress(&mut workspace, &()).unwrap();

        let estimate = &report.groups[0];
        assert!(estimate.accepted, "{report:?}");
        assert_eq!(estimate.initial_focal, GUESS);
        assert!((estimate.focal / FOCAL - 1.0).abs() < 0.01, "{report:?}");
        assert_eq!(estimate.pair_count, VIEWS * (VIEWS - 1) / 2, "{report:?}");
        // Principal point and the declared distortion model are kept; the prior stays a guess.
        assert_eq!(workspace.camera_models()[0], camera(estimate.focal));
        assert_eq!(
            workspace.intrinsic_prior(IntrinsicId::new(0)),
            Some(IntrinsicPrior::Guess)
        );
    }

    #[test]
    fn the_estimate_lets_verification_keep_more_inliers() {
        let verified_inliers = |mut workspace: SfmWorkspace| {
            workspace
                .verify_pairs(&CalibratedVerifier::default(), PairSelector::All)
                .unwrap();
            workspace
                .geometric_model_store()
                .iter_populated()
                .map(|(_, model)| model.evidence().inlier_count)
                .sum::<usize>()
        };
        let mut calibrated = shared_guess();
        calibrate_guessed_focals_with_progress(&mut calibrated, &()).unwrap();

        assert!(verified_inliers(calibrated) > verified_inliers(shared_guess()));
    }

    #[test]
    fn trusted_and_unshared_focals_are_never_profiled() {
        let trusted = matched_workspace(
            vec![camera(GUESS)],
            vec![IntrinsicPrior::Trusted],
            vec![IntrinsicId::new(0); VIEWS],
            false,
        );
        let one_view_per_group = matched_workspace(
            vec![camera(GUESS); VIEWS],
            vec![IntrinsicPrior::Guess; VIEWS],
            (0..VIEWS).map(IntrinsicId::new).collect(),
            false,
        );
        for mut workspace in [trusted, one_view_per_group] {
            let cameras = workspace.camera_models().to_vec();
            assert!(!has_guessed_focal(&workspace));

            let report = calibrate_guessed_focals_with_progress(&mut workspace, &()).unwrap();

            assert!(report.groups.is_empty(), "{report:?}");
            assert_eq!(workspace.camera_models(), cameras.as_slice());
        }
    }

    #[test]
    fn pairs_with_wrong_matches_leave_the_guess_in_place() {
        let mut workspace = matched_workspace(
            vec![camera(GUESS)],
            vec![IntrinsicPrior::Guess],
            vec![IntrinsicId::new(0); VIEWS],
            true,
        );

        let report = calibrate_guessed_focals_with_progress(&mut workspace, &()).unwrap();

        let estimate = &report.groups[0];
        assert!(!estimate.accepted, "{report:?}");
        assert_eq!(estimate.pair_count, 0, "{report:?}");
        assert_eq!(focal_of(&workspace.camera_models()[0]), GUESS);
    }

    #[test]
    fn a_verified_workspace_is_rejected_unchanged() {
        let mut workspace = shared_guess();
        workspace
            .verify_pairs(&CalibratedVerifier::default(), PairSelector::All)
            .unwrap();

        assert!(calibrate_guessed_focals_with_progress(&mut workspace, &()).is_err());
        assert_eq!(focal_of(&workspace.camera_models()[0]), GUESS);
    }

    #[test]
    fn an_essential_matrix_has_two_equal_singular_values() {
        let rotation = Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let t = Vector3::new(0.3, -0.1, 1.0);
        let skew = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
        let essential = skew * rotation.matrix();

        assert!(essential_inequality(&essential) < 1e-12);
        let not_essential = Matrix3::from_diagonal(&Vector3::new(2.0, 1.0, 0.0));
        assert!(essential_inequality(&not_essential) > 0.3);
    }
}
