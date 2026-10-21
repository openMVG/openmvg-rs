// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Radial pre-calibration of camera groups whose lens distortion is unknown.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Radial pre-calibration (calibrate_radial_distortion_with_progress)   │
//! └──────────────────────────────────────────────────────────────────────┘
//! Input: verified pairs; camera groups declaring K1 or K1K2 with all-zero
//!        coefficients (pinhole groups and known distortion are skipped)
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Candidate grid per unknown group                             │
//! └──────────────────────────────────────────────────────────────────────┘
//! 100 cameras: corner distortion k1·r_max² from −0.1475 (barrel) to +0.10
//! (pincushion) in steps of 0.0025, step 0 the undistorted model.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Residual curve per same-group pair (≥ 60 verified inliers)   │
//! └──────────────────────────────────────────────────────────────────────┘
//! At most 256 sampled inliers, unprojected through every candidate; a
//! Hartley-normalized linear essential matrix (singular values 1, 1, 0)
//! gives the median Sampson error, divided by the undistorted model's.
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Group estimate                                               │
//! └──────────────────────────────────────────────────────────────────────┘
//! Median curve over the group's n pairs; its minimum is accepted when it
//! is interior to the grid and either refines an adopted model or improves
//! more than n/2 + 3·√n/2 pairs (one-sided sign test).
//!   │
//!   ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Adopt and re-verify                                          │
//! └──────────────────────────────────────────────────────────────────────┘
//! Replace each changed group's camera, then reactivate and re-verify every
//! matched pair that touches it. Steps 2–4 repeat, at most 6 rounds, until
//! the accepted groups lie within one grid step of the adopted models.
//!   │
//!   ▼
//! Output: k1 per accepted group (k2 stays 0 for bundle adjustment), a
//!         RadialCalibrationReport per round; tracks are built afterwards
//! ```
//!
//! Global SfM consumes the verified pairwise geometry before any bundle adjustment, so a lens's
//! radial distortion must already be modeled at verification time: an unmodeled barrel bends
//! every calibrated essential matrix toward a wrong relative pose. A group declares its distortion
//! unknown by carrying a radial model (`K1` or `K1K2`) whose coefficients are all zero; pinhole
//! groups declare no distortion and are never touched. For each unknown group this profiles the
//! median residual of essential matrices fitted to the group's own verified inliers over a grid of
//! `k1` values, and adopts the best model only when significantly more pairs improve than chance
//! would give; `k2` stays zero for bundle adjustment to estimate. Verification inlier sets depend
//! on the model (an undistorted verifier discards exactly the peripheral matches that carry the
//! distortion signal), so estimation and re-verification alternate until the adopted models
//! settle. Only inliers an undistorted verifier selected can test the undistorted model; inliers
//! selected by an adopted model refine it.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Matrix3, Point2, SMatrix, SVector, Vector3};
use openmvg_core::camera::{
    BrownConradyCamera, CameraModel, RadialDistortion, TangentialDistortion,
};

use crate::backends::GeometricVerifier;
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::reports::{RadialCalibrationReport, RadialGroupEstimate};
use crate::workspace::selector::PairSelector;
use crate::{Camera, IntrinsicId, PairId, Result, SfmError, SfmWorkspace};

/// Grid spacing of the distortion at the farthest image corner, `k1 * r_max^2` in normalized
/// coordinates. Searching this instead of `k1` gives every focal length the same physical range:
/// a telephoto lens's small normalized radius needs a large `k1` for the same pixel shift.
const CORNER_DISTORTION_STEP: f64 = 0.0025;
/// Barrel end of the grid, -0.1475: beyond -4/27 a one-parameter model folds before the image
/// corner, and the corner pixels could no longer be unprojected.
const BARREL_STEPS: i32 = 59;
/// Pincushion end of the grid, +0.10.
const PINCUSHION_STEPS: i32 = 40;
const MAX_POINTS_PER_PAIR: usize = 256;
const MIN_PAIR_INLIERS: usize = 60;
/// One-sided sign-test margin in binomial standard deviations: a group is radially distorted
/// only if significantly more of its pairs improve than the half expected by chance.
const SIGN_TEST_SIGMAS: f64 = 3.0;
const MAX_ROUNDS: usize = 6;

struct UnknownRadialGroup {
    intrinsic_id: IntrinsicId,
    corner_radius_squared: f64,
    /// Candidate camera per grid step, from the barrel end; the undistorted model at `BARREL_STEPS`.
    grid: Vec<Camera>,
}

#[cfg(test)]
fn calibrate_radial_distortion(
    workspace: &mut SfmWorkspace,
    verifier: &dyn GeometricVerifier,
) -> Result<RadialCalibrationReport> {
    calibrate_radial_distortion_with_progress(workspace, verifier, &())
}

/// Estimate and adopt `k1` for every camera group whose radial distortion is declared unknown,
/// reporting [`SfmProgressStage::RadialCalibration`] in camera groups.
///
/// Requires verified pairs. Each round re-models the groups whose adopted distortion changed and
/// re-verifies every matched pair incident to them, including pairs an earlier verification
/// rejected; groups whose evidence is not significant keep their declared, undistorted model.
/// Tracks are not rebuilt: build them after this operation.
pub(crate) fn calibrate_radial_distortion_with_progress(
    workspace: &mut SfmWorkspace,
    verifier: &dyn GeometricVerifier,
    progress: &dyn SfmProgressSink,
) -> Result<RadialCalibrationReport> {
    if workspace.is_matching_data_released() {
        return Err(SfmError::MatchingDataReleased);
    }
    let groups = unknown_radial_groups(workspace)?;
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::RadialCalibration, groups.len());
    let mut adopted = BTreeMap::new();
    let mut report = RadialCalibrationReport::default();
    for _ in 0..MAX_ROUNDS {
        cancellation_point(progress)?;
        let estimates = estimate_radial_distortion(workspace, &groups, &adopted);
        let accepted: BTreeMap<IntrinsicId, i32> = estimates
            .iter()
            .filter(|estimate| estimate.accepted)
            .map(|estimate| {
                let step = (estimate.corner_distortion / CORNER_DISTORTION_STEP).round() as i32;
                (estimate.intrinsic_id, step)
            })
            .collect();
        report.rounds.push(estimates);
        // Settled when the same groups are accepted within one grid step of the adopted models.
        if accepted.len() == adopted.len()
            && accepted.iter().all(|(id, step)| {
                adopted
                    .get(id)
                    .is_some_and(|old: &i32| (step - old).abs() <= 1)
            })
        {
            report.settled = true;
            break;
        }
        adopt(workspace, verifier, &groups, &adopted, &accepted)?;
        adopted = accepted;
    }
    if !groups.is_empty() {
        progress.advance(SfmProgressStage::RadialCalibration, groups.len());
    }
    progress.finish(SfmProgressStage::RadialCalibration);
    Ok(report)
}

fn unknown_radial_groups(workspace: &SfmWorkspace) -> Result<Vec<UnknownRadialGroup>> {
    let mut groups = Vec::new();
    for (index, camera) in workspace.camera_models().iter().enumerate() {
        let intrinsic_id = IntrinsicId::new(index);
        let Some(declared) = unknown_radial(camera) else {
            continue;
        };
        let pinhole = declared.pinhole();
        let Some(view) = workspace
            .views()
            .iter()
            .find(|view| view.intrinsic_id == intrinsic_id)
        else {
            continue;
        };
        let (focal, cx, cy) = (
            pinhole.focal_x(),
            pinhole.principal_x(),
            pinhole.principal_y(),
        );
        let corner_radius_squared = (cx.max(f64::from(view.meta.width) - cx) / focal).powi(2)
            + (cy.max(f64::from(view.meta.height) - cy) / focal).powi(2);
        let grid = (-BARREL_STEPS..=PINCUSHION_STEPS)
            .map(|step| {
                let k1 = f64::from(step) * CORNER_DISTORTION_STEP / corner_radius_squared;
                radial_camera(declared, k1)
            })
            .collect::<Result<_>>()?;
        groups.push(UnknownRadialGroup {
            intrinsic_id,
            corner_radius_squared,
            grid,
        });
    }
    Ok(groups)
}

/// Whether any camera group declares a radial model with unknown distortion.
pub(crate) fn has_unknown_radial_distortion(workspace: &SfmWorkspace) -> bool {
    workspace
        .camera_models()
        .iter()
        .any(|camera| unknown_radial(camera).is_some())
}

/// The camera of a group that declares a radial model with every coefficient still zero.
fn unknown_radial(camera: &Camera) -> Option<&BrownConradyCamera> {
    let Camera::BrownConrady(brown) = camera else {
        return None;
    };
    let unknown = match brown.radial_distortion() {
        RadialDistortion::K1(k1) => *k1 == 0.0,
        RadialDistortion::K1K2(k1, k2) => *k1 == 0.0 && *k2 == 0.0,
        RadialDistortion::None | RadialDistortion::K1K2K3(..) => false,
    };
    (unknown && *brown.tangential_distortion() == TangentialDistortion::None).then_some(brown)
}

/// The declared model with `k1` set; `k2` of a `K1K2` model stays zero.
fn radial_camera(declared: &BrownConradyCamera, k1: f64) -> Result<Camera> {
    let radial = match declared.radial_distortion() {
        RadialDistortion::K1K2(..) => RadialDistortion::K1K2(k1, 0.0),
        _ => RadialDistortion::K1(k1),
    };
    Ok(Camera::BrownConrady(BrownConradyCamera::try_new(
        declared.pinhole().clone(),
        radial,
        TangentialDistortion::None,
    )?))
}

fn adopt(
    workspace: &mut SfmWorkspace,
    verifier: &dyn GeometricVerifier,
    groups: &[UnknownRadialGroup],
    previous: &BTreeMap<IntrinsicId, i32>,
    accepted: &BTreeMap<IntrinsicId, i32>,
) -> Result<()> {
    let mut replacements = Vec::new();
    for group in groups {
        let step = accepted.get(&group.intrinsic_id);
        if step != previous.get(&group.intrinsic_id) {
            let camera = group.grid[(step.unwrap_or(&0) + BARREL_STEPS) as usize].clone();
            replacements.push((group.intrinsic_id, camera));
        }
    }
    let changed: BTreeSet<IntrinsicId> = replacements.iter().map(|(id, _)| *id).collect();
    workspace.replace_camera_model_groups(replacements)?;
    let group_of = |view| workspace.view(view).map(|view| view.intrinsic_id);
    let pairs: Vec<_> = workspace
        .match_store()
        .iter_populated()
        .filter(|(_, matches)| !matches.is_empty())
        .filter_map(|(index, _)| workspace.pair_graph().pair(PairId::new(index)))
        .filter(|pair| {
            [pair.first(), pair.second()]
                .into_iter()
                .any(|view| group_of(view).is_some_and(|group| changed.contains(&group)))
        })
        .collect();
    for pair in &pairs {
        workspace.pair_graph_mut().activate_pair(*pair)?;
    }
    workspace.verify_pairs(verifier, PairSelector::Pairs(pairs))?;
    Ok(())
}

/// `adopted` holds the grid steps of the models the current inliers were verified with.
fn estimate_radial_distortion(
    workspace: &SfmWorkspace,
    groups: &[UnknownRadialGroup],
    adopted: &BTreeMap<IntrinsicId, i32>,
) -> Vec<RadialGroupEstimate> {
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
        pair_residual_curve(workspace, pair_id, &groups[group]).map(|curve| (group, curve))
    };
    #[cfg(feature = "parallel")]
    let curves: Vec<(usize, Vec<f64>)> = {
        use rayon::prelude::*;
        candidates.par_iter().filter_map(profile).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let curves: Vec<(usize, Vec<f64>)> = candidates.iter().filter_map(profile).collect();

    let mut estimates = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        let curves: Vec<&[f64]> = curves
            .iter()
            .filter(|(owner, _)| *owner == index)
            .map(|(_, curve)| curve.as_slice())
            .collect();
        if curves.is_empty() {
            continue;
        }
        let profile: Vec<f64> = (0..group.grid.len())
            .map(|step| median(curves.iter().map(|curve| curve[step]).collect()))
            .collect();
        let Some((best, cost_ratio)) = profile
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        let pairs = curves.len();
        let improved_pairs = curves.iter().filter(|curve| curve[best] < 1.0).count();
        let chance = 0.5 * pairs as f64 + SIGN_TEST_SIGMAS * 0.5 * (pairs as f64).sqrt();
        let interior = best > 0 && best + 1 < group.grid.len();
        let corner_distortion = f64::from(best as i32 - BARREL_STEPS) * CORNER_DISTORTION_STEP;
        // The sign test asks whether the undistorted model is wrong; inliers an adopted model
        // selected were never undistorted inliers, so for those groups the minimum only refines.
        let refining = adopted.contains_key(&group.intrinsic_id);
        estimates.push(RadialGroupEstimate {
            intrinsic_id: group.intrinsic_id,
            pair_count: pairs,
            improved_pair_count: improved_pairs,
            corner_distortion,
            k1: corner_distortion / group.corner_radius_squared,
            cost_ratio,
            accepted: interior && (refining || improved_pairs as f64 > chance),
        });
    }
    estimates
}

/// Median essential-matrix residual of one pair's verified inliers at every grid step, relative
/// to the undistorted model. Pixels the pinhole part is normalized with never change, so every
/// round's estimate is absolute; only the verified inlier sets change between rounds. A step
/// that cannot unproject every sampled pixel is infeasible.
fn pair_residual_curve(
    workspace: &SfmWorkspace,
    pair_id: PairId,
    group: &UnknownRadialGroup,
) -> Option<Vec<f64>> {
    let pair = workspace.pair_graph().pair(pair_id)?;
    let model = workspace.geometric_model_store().get(pair_id.index())?;
    let matches = workspace.match_store().get(pair_id.index())?;
    let keypoints_i = workspace.keypoint_store().get(pair.first().index())?;
    let keypoints_j = workspace.keypoint_store().get(pair.second().index())?;
    let pixel = |keypoint: openmvg_features::model::Keypoint| {
        Point2::new(f64::from(keypoint.x()), f64::from(keypoint.y()))
    };
    let inliers: Vec<(Point2<f64>, Point2<f64>)> = matches
        .as_slice()
        .iter()
        .zip(model.evidence().inliers.as_slice())
        .filter(|(_, inlier)| **inlier)
        .map(|(matched, _)| {
            (
                pixel(keypoints_i[matched.first().index()]),
                pixel(keypoints_j[matched.second().index()]),
            )
        })
        .collect();
    if inliers.len() < MIN_PAIR_INLIERS {
        return None;
    }
    let stride = inliers.len().div_ceil(MAX_POINTS_PER_PAIR);
    let sample: Vec<_> = inliers.into_iter().step_by(stride).collect();
    let normalized = |camera: &Camera, pixel: &Point2<f64>| {
        let ray = camera.unproject(pixel).ok()?;
        Some([ray.x / ray.z, ray.y / ray.z])
    };
    let curve: Vec<f64> = group
        .grid
        .iter()
        .map(|camera| {
            let (x1, x2): (Vec<_>, Vec<_>) = sample
                .iter()
                .map(|(a, b)| Some((normalized(camera, a)?, normalized(camera, b)?)))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .unzip();
            essential_median_sampson(&x1, &x2)
        })
        .map(|cost| cost.unwrap_or(f64::INFINITY))
        .collect();
    let undistorted = curve[BARREL_STEPS as usize];
    (undistorted.is_finite() && undistorted > 0.0)
        .then(|| curve.iter().map(|cost| cost / undistorted).collect())
}

/// Isotropic (Hartley) normalization: centroid to the origin, mean distance sqrt(2).
fn hartley(points: &[[f64; 2]]) -> Matrix3<f64> {
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), p| (x + p[0] / n, y + p[1] / n));
    let mean_distance = points
        .iter()
        .map(|p| (p[0] - mx).hypot(p[1] - my))
        .sum::<f64>()
        / n;
    let s = std::f64::consts::SQRT_2 / mean_distance.max(f64::MIN_POSITIVE);
    Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0)
}

/// Median Sampson error of a linear eight-point essential matrix in camera-normalized
/// coordinates. Verification fits a calibrated essential matrix at the group's focal length; a
/// free fundamental matrix would absorb part of the radial deformation (and any focal error) into
/// its two extra degrees of freedom.
fn essential_median_sampson(x1: &[[f64; 2]], x2: &[[f64; 2]]) -> Option<f64> {
    let (t1, t2) = (hartley(x1), hartley(x2));
    let apply = |t: &Matrix3<f64>, p: &[f64; 2]| {
        [t[(0, 0)] * p[0] + t[(0, 2)], t[(1, 1)] * p[1] + t[(1, 2)]]
    };
    let mut normal = SMatrix::<f64, 9, 9>::zeros();
    for (a, b) in x1.iter().zip(x2) {
        let (a, b) = (apply(&t1, a), apply(&t2, b));
        let row = SVector::<f64, 9>::from_column_slice(&[
            b[0] * a[0],
            b[0] * a[1],
            b[0],
            b[1] * a[0],
            b[1] * a[1],
            b[1],
            a[0],
            a[1],
            1.0,
        ]);
        normal += row * row.transpose();
    }
    let eigen = nalgebra::SymmetricEigen::new(normal);
    let (smallest, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let f = eigen.eigenvectors.column(smallest);
    let f = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
    let mut svd = (t2.transpose() * f * t1).svd(true, true);
    svd.singular_values = Vector3::new(1.0, 1.0, 0.0);
    let e = svd.recompose().ok()?;
    let errors: Vec<f64> = x1
        .iter()
        .zip(x2)
        .map(|(a, b)| {
            let (h1, h2) = (Vector3::new(a[0], a[1], 1.0), Vector3::new(b[0], b[1], 1.0));
            let (ex1, etx2) = (e * h1, e.transpose() * h2);
            let numerator = h2.dot(&ex1);
            let denominator = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
            (numerator * numerator / denominator).sqrt()
        })
        .collect();
    (!errors.is_empty()).then(|| median(errors))
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
    use crate::{FeatureId, ViewId, ViewPair};
    use nalgebra::{Rotation3, Vector3};
    use openmvg_core::camera::PinholeCamera;
    use openmvg_features::model::{DescriptorSet, ExtractedFeatures, FeatureMatch, Keypoint};

    const WIDTH: u32 = 1600;
    const HEIGHT: u32 = 1200;
    const FOCAL: f64 = 1200.0;
    const VIEWS: usize = 8;

    /// Deterministic value in [-1, 1).
    fn hash(index: usize, salt: usize) -> f64 {
        let mut x = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64);
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn declared(radial: RadialDistortion) -> Camera {
        Camera::BrownConrady(
            BrownConradyCamera::try_with_focal(
                FOCAL,
                800.0,
                600.0,
                radial,
                TangentialDistortion::None,
            )
            .unwrap(),
        )
    }

    /// Eight views on an arc around a box of points, imaged through a lens with the given
    /// corner distortion and modeled with `K1` distortion declared unknown, then verified.
    fn verified_workspace(corner_distortion: f64) -> SfmWorkspace {
        verified_workspace_declaring(declared(RadialDistortion::K1(0.0)), corner_distortion)
    }

    fn verified_workspace_declaring(camera: Camera, corner_distortion: f64) -> SfmWorkspace {
        let Camera::BrownConrady(lens) = declared(RadialDistortion::K1(0.0)) else {
            unreachable!();
        };
        let corner_radius_squared = (800.0 / FOCAL).powi(2) + (600.0 / FOCAL).powi(2);
        let lens = radial_camera(&lens, corner_distortion / corner_radius_squared).unwrap();
        let points: Vec<Vector3<f64>> = (0..800)
            .map(|p| 2.5 * Vector3::new(hash(p, 1), hash(p, 2), hash(p, 3)))
            .collect();
        let mut workspace = SfmWorkspace::builder()
            .image_meta((0..VIEWS).map(|v| ImageMeta::external_id(format!("v{v}"), WIDTH, HEIGHT)))
            .camera_models([camera])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .unwrap();
        // Per view: feature index of every visible point.
        let mut visible = Vec::new();
        for view in 0..VIEWS {
            let angle = -0.4 + 0.8 * view as f64 / (VIEWS - 1) as f64;
            let centre = Vector3::new(6.0 * angle.sin(), 0.3 * hash(view, 4), -6.0 * angle.cos());
            let rotation = Rotation3::look_at_lh(&-centre, &Vector3::y());
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
                let matches = visible[i]
                    .iter()
                    .zip(&visible[j])
                    .filter_map(|(a, b)| Some((a.as_ref()?, b.as_ref()?)))
                    .map(|(a, b)| {
                        FeatureMatch::try_new(FeatureId::new(*a), FeatureId::new(*b), None).unwrap()
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
            .verify_pairs(&CalibratedVerifier::default(), PairSelector::All)
            .unwrap();
        workspace
    }

    #[test]
    fn a_barrel_lens_with_unknown_distortion_recovers_its_corner_distortion() {
        let mut workspace = verified_workspace(-0.06);

        let report =
            calibrate_radial_distortion(&mut workspace, &CalibratedVerifier::default()).unwrap();

        let last = &report.rounds.last().unwrap()[0];
        assert!(report.settled, "{report:?}");
        assert!(last.accepted, "{report:?}");
        assert!(
            (last.corner_distortion + 0.06).abs() <= 2.0 * CORNER_DISTORTION_STEP,
            "{report:?}"
        );
        let Camera::BrownConrady(camera) = &workspace.camera_models()[0] else {
            unreachable!();
        };
        assert!(matches!(camera.radial_distortion(), RadialDistortion::K1(k1) if *k1 < 0.0));
    }

    #[test]
    fn a_declared_k1k2_model_keeps_its_variant_with_k2_left_zero() {
        let mut workspace =
            verified_workspace_declaring(declared(RadialDistortion::K1K2(0.0, 0.0)), -0.06);

        let report =
            calibrate_radial_distortion(&mut workspace, &CalibratedVerifier::default()).unwrap();

        assert!(report.rounds.last().unwrap()[0].accepted, "{report:?}");
        let Camera::BrownConrady(camera) = &workspace.camera_models()[0] else {
            unreachable!();
        };
        assert!(matches!(
            camera.radial_distortion(),
            RadialDistortion::K1K2(k1, k2) if *k1 < 0.0 && *k2 == 0.0
        ));
    }

    #[test]
    fn pinhole_and_known_distortion_groups_are_never_touched() {
        for camera in [
            Camera::Pinhole(PinholeCamera::try_with_focal(FOCAL, 800.0, 600.0).unwrap()),
            declared(RadialDistortion::K1(-0.01)),
            declared(RadialDistortion::K1K2K3(0.0, 0.0, 0.0)),
        ] {
            let mut workspace = verified_workspace_declaring(camera.clone(), -0.06);
            let before = workspace.geometric_model_store().populated_count();

            let report =
                calibrate_radial_distortion(&mut workspace, &CalibratedVerifier::default())
                    .unwrap();

            assert!(report.rounds.iter().all(Vec::is_empty), "{report:?}");
            assert_eq!(workspace.camera_models()[0], camera);
            assert_eq!(workspace.geometric_model_store().populated_count(), before);
        }
    }

    #[test]
    fn an_undistorted_lens_keeps_its_declared_model() {
        let mut workspace = verified_workspace(0.0);
        let before = workspace.geometric_model_store().populated_count();

        let report =
            calibrate_radial_distortion(&mut workspace, &CalibratedVerifier::default()).unwrap();

        assert!(report.settled && report.rounds.len() == 1, "{report:?}");
        assert!(!report.rounds[0][0].accepted, "{report:?}");
        assert_eq!(
            workspace.camera_models()[0],
            declared(RadialDistortion::K1(0.0))
        );
        assert_eq!(workspace.geometric_model_store().populated_count(), before);
    }

    #[test]
    fn a_group_verified_with_an_adopted_model_is_refined_not_retested() {
        let workspace = verified_workspace(0.0);
        let groups = unknown_radial_groups(&workspace).unwrap();
        let adopted = BTreeMap::from([(IntrinsicId::new(0), 0)]);

        let tested = estimate_radial_distortion(&workspace, &groups, &BTreeMap::new());
        let refined = estimate_radial_distortion(&workspace, &groups, &adopted);

        assert!(!tested[0].accepted, "{tested:?}");
        assert!(refined[0].accepted, "{refined:?}");
        assert_eq!(refined[0].corner_distortion, tested[0].corner_distortion);
    }
}
