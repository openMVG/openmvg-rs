// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pose decomposition of an essential matrix.
//!
//! Its teaching docs are on [`decompose`], where docs.rs shows them.

use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

use crate::correspondence::BearingCorrespondence;
use crate::{SolverError, SolverResult};

use openmvg_core::model::RelativePose;

use super::EssentialMatrix;

// ============================================================================
// Essential matrix decomposition
// ============================================================================

/// W matrix for essential matrix decomposition.
/// Encodes 90-degree rotation around Z-axis.
fn w_matrix() -> Matrix3<f64> {
    Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
}

const NOT_DECOMPOSABLE: SolverError = SolverError::InvalidInput {
    reason: "essential matrix does not decompose into finite relative poses",
};

impl EssentialMatrix {
    /// Decompose the essential matrix into 4 possible [R|t] configurations.
    ///
    /// From E = U * diag(1,1,0) * V^T, we get:
    /// - R1 = U * W * V^T,  R2 = U * W^T * V^T
    /// - `t = ±U[:, 2]`
    ///
    /// Uses the **double-SVD** approach for numerical stability:
    /// 1. First SVD of scale-normalized E to get singular values
    /// 2. Normalize to equal non-zero singular values: E_c = U * diag(s,s,0) * V^T
    /// 3. Second SVD of E_c to get clean U/V basis vectors
    ///
    /// The projection enforces the equal-singular-value essential constraint.
    /// Nonzero pure translation has a valid essential matrix; zero baseline
    /// does not determine one.
    ///
    /// The stored translation is the world-to-camera `t` of `P2 = [R|t]`, not the
    /// camera center; recover the center as `C2 = -R^T * t` after cheirality selection.
    ///
    /// Returns the 4 poses in this order:
    /// `[0]`: `(R1, +t)`, `[1]`: `(R1, -t)`, `[2]`: `(R2, +t)`,
    /// `[3]`: `(R2, -t)`
    ///
    /// # How it works
    ///
    /// Decomposes an essential matrix into rotation and translation, and selects
    /// the geometrically valid pose with a cheirality test.
    ///
    /// ```text
    /// ┌──────────────────────────────────────────────────────────────────────┐
    /// │                 Essential matrix pose decomposition                  │
    /// └──────────────────────────────────────────────────────────────────────┘
    /// Input: E (3×3, any nonzero scale), n bearing correspondences (b₁, b₂)
    ///   │
    ///   ▼
    /// ┌──────────────────────────────────────────────────────────────────────┐
    /// │ Step 1: Check and normalize E                                        │
    /// └──────────────────────────────────────────────────────────────────────┘
    /// Reject non-finite, zero, or non-essential E (Demazure cubic). Divide by
    /// the largest absolute entry.
    ///   │
    ///   ▼
    /// ┌──────────────────────────────────────────────────────────────────────┐
    /// │ Step 2: Project onto the essential manifold                          │
    /// └──────────────────────────────────────────────────────────────────────┘
    /// SVD E = U diag(s₀, s₁, s₂) Vᵀ. Replace the singular values by
    /// (s, s, 0) with s = (s₀ + s₁) / 2, then take a second SVD. Flip the
    /// third column of U and the third row of Vᵀ when their determinant is −1.
    ///   │
    ///   ▼
    /// ┌──────────────────────────────────────────────────────────────────────┐
    /// │ Step 3: Four candidate poses                                         │
    /// └──────────────────────────────────────────────────────────────────────┘
    /// R₁ = U W Vᵀ, R₂ = U Wᵀ Vᵀ, t̂ = ±U[:, 2].
    /// Output of `decompose`: [(R₁, t̂), (R₁, −t̂), (R₂, t̂), (R₂, −t̂)]
    ///   │
    ///   ▼
    /// ┌──────────────────────────────────────────────────────────────────────┐
    /// │ Step 4: Cheirality selection                                         │
    /// └──────────────────────────────────────────────────────────────────────┘
    /// Per candidate and correspondence, solve the 2×2 system for the ray
    /// depths (λ₁, λ₂) and count the pairs with both depths positive. Pick
    /// the candidate with the most points in front of both cameras if it
    /// passes the 90% and margin tests; when it does not, repeat the test on the
    /// correspondences with enough parallax to carry a depth sign (97%).
    /// Output of `decompose_with_cheirality`: one `RelativePose`, or `None`
    /// ```
    ///
    /// # Problem and method
    ///
    /// A calibrated pair of views with bearings `b₁` in the first (source) camera
    /// frame and `b₂` in the second (target) camera frame satisfies `b₂ᵀ E b₁ = 0`
    /// with `E = [t̂]ₓ R`. The rotation `R` and the unit translation `t̂` are the
    /// [`RelativePose`] fields: `x₂ = R x₁ + μ t̂`, so `t̂` is the world-to-camera
    /// translation of `P₂ = [R | t̂]` when `P₁ = [I | 0]`, not the camera center.
    /// The center of the second camera is `C₂ = −Rᵀ t̂`. The baseline length `μ` is
    /// not recoverable from `E`, and `E` itself is only defined up to scale.
    ///
    /// A valid `E` has singular values `(s, s, 0)`. With `W = [[0, −1, 0], [1, 0, 0],
    /// [0, 0, 1]]`, the factorization `E = U diag(s, s, 0) Vᵀ` gives two rotations
    /// and two translation signs, so four candidates. Only the last column of `U`
    /// and the last row of `Vᵀ` are free (they pair with the zero singular value),
    /// so the sign fix-up flips only those; negating a whole matrix would flip the
    /// sign of the recovered rotation.
    ///
    /// The cheirality test needs no triangulation. Taking the dot product of
    /// `λ₂ b₂ = λ₁ R b₁ + t̂` with `R b₁` and with `b₂` gives
    /// `[1 a; a 1] [λ₁; λ₂] = [β₁; β₂]` with `a = −(R b₁)·b₂`, `β₁ = −(R b₁)·t̂`
    /// and `β₂ = b₂·t̂`. The solution has the factor `1 − a²` in the denominator,
    /// so the signs of the numerators decide.
    ///
    /// # Details of the cheirality selection
    ///
    /// `decompose_with_cheirality` scores every candidate on all correspondences
    /// and returns the best candidate only when:
    ///
    /// - at least 90% of the correspondences (rounded up) are in front of both
    ///   cameras, and
    /// - it beats the runner-up by at least 10% of the correspondences (at least
    ///   one), or its summed `min(λ₁, λ₂)` exceeds 1.3 times the runner-up's.
    ///
    /// A short baseline leaves many correspondences with a parallax below the noise
    /// of the bearings (every bearing near the direction of motion, and every distant
    /// point), so the sign of their depth is random and the 90% test rejects a pose
    /// that the rest of the correspondences settle. A spherical camera always has
    /// both directions of motion in view. When the test fails on all
    /// correspondences, it is repeated on those whose bearings, derotated by the
    /// best-scoring candidate, differ by at least 0.01 rad, if there are at least
    /// 20 of them; the same margin tests apply and at least 97% must be in front.
    ///
    /// # Failure modes
    ///
    /// - `decompose` returns [`SolverError::InvalidInput`] when `E` is not finite,
    ///   is zero, violates the essential constraint (`E.is_valid()` is false), or
    ///   yields a non-finite or improper rotation. Pure translation is valid; zero
    ///   baseline is not.
    /// - `decompose_with_cheirality` returns `None` when the matrix does not
    ///   decompose, `correspondences` is empty, or the best candidate fails both
    ///   passes: fewer than 90% of all correspondences (97% of those with 0.01 rad
    ///   of parallax, when there are at least 20) in front, or no margin or depth-score
    ///   advantage over the runner-up.
    /// - Near-parallel bearings (`|1 − a²| < 1e-12`) never count as in front during
    ///   selection, so a set of correspondences that are all near-parallel returns `None`.
    ///
    /// # Alternatives
    ///
    /// [`FivePointSolver`](super::FivePointSolver) and the essential-matrix estimators produce
    /// `E`; this method turns an `E` into a pose. Call `decompose_with_cheirality` on the
    /// inlier bearings of an estimated `E`.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidInput`] when [`is_valid`](Self::is_valid) fails, or when the
    /// matrix does not yield finite proper rotations, as happens for entries large enough to
    /// overflow the SVD.
    pub fn decompose(&self) -> SolverResult<[RelativePose; 4]> {
        if !self.is_valid() {
            return Err(NOT_DECOMPOSABLE);
        }
        // Only the projective class matters to decomposition. Normalize before
        // either SVD so finite extreme scales cannot overflow or underflow.
        let scale = self
            .0
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let normalized = self.0 / scale;
        // Step 1: First SVD to get singular values
        let svd1 = normalized.svd(true, true);
        let u1 = svd1.u.ok_or(NOT_DECOMPOSABLE)?;
        let vt1 = svd1.v_t.ok_or(NOT_DECOMPOSABLE)?;
        let s = &svd1.singular_values;

        // Step 2: Normalize E to have equal non-zero singular values
        let avg_s = (s[0] + s[1]) / 2.0;
        let e_corrected =
            u1 * Matrix3::from_diagonal(&nalgebra::Vector3::new(avg_s, avg_s, 0.0)) * vt1;

        // Step 3: Second SVD on the corrected E for clean basis vectors
        let svd2 = e_corrected.svd(true, true);
        let u = svd2.u.ok_or(NOT_DECOMPOSABLE)?;
        let vt = svd2.v_t.ok_or(NOT_DECOMPOSABLE)?;

        let w = w_matrix();
        let wt = w.transpose();

        // Fix up U and Vt so that they have det = +1.
        // Only the last column of U and last row of Vt are undetermined
        // (they correspond to the zero singular value), so flip only those.
        // Negating the whole matrix instead would flip the sign of the
        // recovered rotation.
        let mut u = u;
        let mut vt = vt;
        if u.determinant() < 0.0 {
            u.column_mut(2).scale_mut(-1.0);
        }
        if vt.determinant() < 0.0 {
            vt.row_mut(2).scale_mut(-1.0);
        }

        // Two possible rotations
        let r1 = u * w * vt;
        let r2 = u * wt * vt;

        // Translation direction from SVD (3rd column of U)
        // This is the projection `t` from P2 = [R|t]
        let t = u.column(2).into_owned();

        // Convert to Rotation3
        let rot1 = Rotation3::from_matrix_unchecked(r1);
        let rot2 = Rotation3::from_matrix_unchecked(r2);

        let pose = |rotation, translation| {
            RelativePose::from_projection_translation(rotation, Unit::new_normalize(translation))
                .map_err(|_| NOT_DECOMPOSABLE)
        };
        Ok([
            pose(rot1, t)?,
            pose(rot1, -t)?,
            pose(rot2, t)?,
            pose(rot2, -t)?,
        ])
    }

    /// The rotation and translation direction from the first camera to the second that put the
    /// correspondences in front of both cameras, chosen among the four poses that fit `E`.
    ///
    /// Call it on the inlier bearings of an estimated `E`, such as those of
    /// [`estimate_essential`](crate::essential::estimate_essential). The pose maps the first
    /// camera's frame to the second's, `X₂ = R X₁ + t`, with `|t| = 1`: the baseline length is
    /// unknown. It returns `None` when `E` does not decompose or no candidate is clearly in front
    /// of both cameras for most correspondences; [`Self::decompose`] gives the exact test.
    pub fn decompose_with_cheirality(
        &self,
        correspondences: &[BearingCorrespondence],
    ) -> Option<RelativePose> {
        let poses = self.decompose().ok()?;
        select_pose_by_cheirality(&poses, correspondences)
    }
}

// ============================================================================
// Cheirality check
// ============================================================================

/// Smallest parallax, in radians, at which a correspondence says which side of a camera pair a
/// point lies on. Below it the derotated rays differ by less than the verifier's own inlier
/// tolerance, so the sign of the algebraic depth is noise.
const INFORMATIVE_PARALLAX_RAD: f64 = 0.01;

/// Select the pose where most points are in front of both cameras.
///
/// Uses the algebraic depth test (no triangulation needed).
///
/// Selection criteria:
/// - The winner must have ≥ [`ALL_MIN_FRONT_PERCENT`]% of checked points in front.
/// - The winner must beat the runner-up by at least 10% of checked points
///   (at least one point), or else have a depth score more than 1.3 times
///   the runner-up's.
/// - If no candidate meets the criteria over every correspondence, the same criteria apply, with
///   [`INFORMATIVE_MIN_FRONT_PERCENT`]% in front, to the correspondences of at least
///   [`INFORMATIVE_PARALLAX_RAD`] parallax under the best-scoring candidate's rotation, provided
///   there are at least [`MIN_INFORMATIVE_POINTS`] of them. Frames with a short baseline put many distant points below that parallax, and
///   their depth signs are random.
/// - If no candidate meets the criteria, returns `None`.
fn select_pose_by_cheirality(
    poses: &[RelativePose; 4],
    correspondences: &[BearingCorrespondence],
) -> Option<RelativePose> {
    let all: Vec<&BearingCorrespondence> = correspondences.iter().collect();
    let (selected, best_by_count) = select_among(poses, &all, ALL_MIN_FRONT_PERCENT);
    if selected.is_some() {
        return selected;
    }
    let rotation = poses.get(best_by_count?)?.rotation_source_to_target();
    let informative: Vec<&BearingCorrespondence> = correspondences
        .iter()
        .filter(|correspondence| {
            let rotated = rotation * correspondence.bearing1();
            let parallax = rotated
                .cross(correspondence.bearing2())
                .norm()
                .atan2(rotated.dot(correspondence.bearing2()));
            parallax >= INFORMATIVE_PARALLAX_RAD
        })
        .collect();
    if informative.len() < MIN_INFORMATIVE_POINTS {
        return None;
    }
    select_among(poses, &informative, INFORMATIVE_MIN_FRONT_PERCENT).0
}

/// Percentage of all correspondences that must lie in front of both cameras.
const ALL_MIN_FRONT_PERCENT: usize = 90;
/// Percentage of the informative correspondences that must lie in front. Stricter than
/// [`ALL_MIN_FRONT_PERCENT`]: a pair rejected on all of its points is accepted only when the
/// points that carry a depth sign agree almost unanimously.
const INFORMATIVE_MIN_FRONT_PERCENT: usize = 97;

/// Fewest informative correspondences that decide the pose by themselves.
const MIN_INFORMATIVE_POINTS: usize = 20;

/// The selected pose, and the index of the candidate with the most points in front.
fn select_among(
    poses: &[RelativePose; 4],
    correspondences: &[&BearingCorrespondence],
    min_front_percent: usize,
) -> (Option<RelativePose>, Option<usize>) {
    if correspondences.is_empty() {
        return (None, None);
    }

    // Check every correspondence.
    let n_check = correspondences.len();

    let mut counts = [0usize; 4];
    let mut depth_scores = [0.0f64; 4];

    for (idx, pose) in poses.iter().enumerate() {
        for corr in correspondences.iter() {
            let (pass, depth) = cheirality_depth(pose, corr.bearing1(), corr.bearing2());
            if pass {
                counts[idx] += 1;
                depth_scores[idx] += depth;
            }
        }
    }

    // Sort candidates by count (desc), then depth score (desc)
    let mut order: [usize; 4] = [0, 1, 2, 3];
    order.sort_unstable_by(|&a, &b| {
        counts[b].cmp(&counts[a]).then_with(|| {
            depth_scores[b]
                .partial_cmp(&depth_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let best = order[0];
    let second = order[1];

    // Require the minimum share of checked points in front
    let threshold_front = (n_check * min_front_percent).div_ceil(100);
    let margin = n_check / 10; // 10% count margin

    if counts[best] < threshold_front {
        return (None, Some(best));
    }

    // Standard margin check passes → accept
    if counts[best] >= counts[second] + margin.max(1) {
        return (Some(poses[best]), Some(best));
    }

    // Counts are close: use the depth score as tiebreaker.
    // Require at least 30% depth-score advantage for a confident pick.
    if depth_scores[second] > 0.0 && depth_scores[best] > depth_scores[second] * 1.3 {
        return (Some(poses[best]), Some(best));
    }

    (None, Some(best))
}

/// Algebraic cheirality test returning (pass, min_depth).
///
/// The min_depth value is `min(λ₁, λ₂)`, the smallest ray depth, which
/// measures how confidently the point is in front of both cameras.
#[inline]
fn cheirality_depth(pose: &RelativePose, x1: &Vector3<f64>, x2: &Vector3<f64>) -> (bool, f64) {
    let rx1 = pose.rotation_source_to_target() * x1;
    let a = -rx1.dot(x2);
    let b1 = -rx1.dot(pose.translation_source_to_target().as_ref());
    let b2 = x2.dot(pose.translation_source_to_target().as_ref());

    let denom = 1.0 - a * a;
    if denom.abs() < 1e-12 {
        return (false, 0.0);
    }

    let lambda1 = (b1 - a * b2) / denom;
    let lambda2 = (-a * b1 + b2) / denom;

    if lambda1 > 0.0 && lambda2 > 0.0 {
        (true, lambda1.min(lambda2))
    } else {
        (false, 0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn essential_decomposition_preserves_extreme_projective_scale() {
        let matrix = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        for scale in [1e-150, 1e-100, 1e-12, 1.0, 1e12, 1e100, 1e150] {
            for sign in [-1.0, 1.0] {
                let poses = EssentialMatrix::try_new(matrix * (sign * scale))
                    .unwrap()
                    .decompose()
                    .unwrap();
                for pose in poses {
                    assert!(pose.validate().is_ok());
                    let t = pose.translation_source_to_target().as_ref();
                    let cross = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
                    let recovered = cross * pose.rotation_source_to_target().matrix();
                    assert!((recovered - matrix).norm().min((recovered + matrix).norm()) < 1e-12);
                }
            }
        }
    }

    use super::*;

    /// Create an essential matrix from a known rotation and camera 2 center.
    fn make_essential(rotation: &Rotation3<f64>, camera2_center: &Vector3<f64>) -> EssentialMatrix {
        // E = [t]_x * R where t = -R * C2
        let t = -(rotation * camera2_center);
        let t_normalized = t.normalize();
        let tx = Matrix3::new(
            0.0,
            -t_normalized.z,
            t_normalized.y,
            t_normalized.z,
            0.0,
            -t_normalized.x,
            -t_normalized.y,
            t_normalized.x,
            0.0,
        );
        EssentialMatrix::new(tx * rotation.matrix())
    }

    /// Generate synthetic correspondences from a known pose.
    ///
    /// Takes the camera 2 center direction (not the projection `t`)
    fn generate_correspondences(
        rotation: &Rotation3<f64>,
        camera2_center: &Vector3<f64>,
        n: usize,
    ) -> Vec<BearingCorrespondence> {
        let mut correspondences = Vec::with_capacity(n);

        // Compute the projection `t` from camera center: t = -R * C2
        let t = -(rotation * camera2_center);

        for i in 0..n {
            let offset = (i as f64 + 1.0) * 0.1;
            let point = Vector3::new(offset * 0.5 - 0.2, offset * 0.3 - 0.1, 2.0 + offset * 0.2);

            let bearing1 = point.normalize();
            // Transform to cam2: X_cam2 = R * X + t
            let point_cam2 = rotation * point + t;
            let bearing2 = point_cam2.normalize();

            correspondences.push(BearingCorrespondence::try_new(bearing1, bearing2).unwrap());
        }

        correspondences
    }

    #[test]
    fn test_decompose_produces_four_poses() {
        let rotation = Rotation3::from_euler_angles(0.1, 0.05, 0.02);
        let translation = Vector3::new(0.5, 0.1, 0.0);

        let e = make_essential(&rotation, &translation);
        let poses = e.decompose().unwrap();

        assert_eq!(poses.len(), 4);

        // All should be valid rotations (det = 1)
        for pose in &poses {
            let det = pose.rotation_source_to_target().matrix().determinant();
            assert!(
                (det - 1.0).abs() < 1e-6,
                "Rotation should have determinant 1"
            );
        }
    }

    #[test]
    fn matrices_without_finite_poses_are_rejected_instead_of_panicking() {
        // Large finite entries overflow the SVD, so finiteness alone does not guarantee poses.
        for value in [f64::NAN, f64::INFINITY, 1e300] {
            let e = EssentialMatrix::new(Matrix3::from_element(value));
            assert!(matches!(
                e.decompose(),
                Err(SolverError::InvalidInput { .. })
            ));
            assert!(e.decompose_with_cheirality(&[]).is_none());
        }
        assert!(matches!(
            EssentialMatrix::new(Matrix3::zeros()).decompose(),
            Err(SolverError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_cheirality_selects_correct_pose() {
        let rotation = Rotation3::from_euler_angles(0.1, 0.05, 0.02);
        let translation = Vector3::new(1.0, 0.2, 0.1);

        let correspondences = generate_correspondences(&rotation, &translation, 10);
        let e = make_essential(&rotation, &translation);

        let poses = e.decompose().unwrap();
        let selected = select_pose_by_cheirality(&poses, &correspondences);

        assert!(selected.is_some(), "Should select a valid pose");

        let pose = selected.unwrap();

        // The selected rotation should be close to the original
        // (up to sign ambiguity in some cases)
        let r_diff = pose.rotation_source_to_target().matrix().transpose() * rotation.matrix();
        let trace = r_diff.trace();
        let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

        // Allow for some numerical error and the sign ambiguity
        assert!(
            angle < 0.1 || (std::f64::consts::PI - angle).abs() < 0.1,
            "Rotation should match ground truth (angle diff: {:.4} rad)",
            angle
        );
    }

    #[test]
    fn test_relative_pose_projection_matrix() {
        let rotation = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let translation = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));

        let pose = RelativePose::from_projection_translation(rotation, translation).unwrap();
        let p = pose.projection_matrix();

        assert_eq!(p.nrows(), 3);
        assert_eq!(p.ncols(), 4);

        // Check that rotation part matches
        for i in 0..3 {
            for j in 0..3 {
                assert!((p[(i, j)] - rotation.matrix()[(i, j)]).abs() < 1e-10);
            }
        }

        // Check that translation part matches
        for i in 0..3 {
            assert!((p[(i, 3)] - translation[i]).abs() < 1e-10);
        }
    }

    /// Bearings of points all around camera 1, half of them so far away that a short baseline
    /// gives them less than the noise as parallax.
    fn short_baseline_sphere(
        rotation: &Rotation3<f64>,
        camera2_center: &Vector3<f64>,
    ) -> Vec<BearingCorrespondence> {
        let t = -(rotation * camera2_center);
        let mut state = 17_u64;
        let mut noise = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.008
        };
        (0..100)
            .map(|index| {
                let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
                let z = 1.0 - 2.0 * (index as f64 + 0.5) / 100.0;
                let ring = (1.0 - z * z).sqrt();
                let direction = Vector3::new(
                    ring * (golden * index as f64).cos(),
                    ring * (golden * index as f64).sin(),
                    z,
                );
                let depth = if index % 2 == 0 { 4.0 } else { 1_000.0 };
                let point = direction * depth;
                let bearing1 = point.normalize();
                let bearing2 = (rotation * point + t).normalize();
                let jitter = |bearing: Vector3<f64>, noise: &mut dyn FnMut() -> f64| {
                    (bearing + Vector3::new(noise(), noise(), noise())).normalize()
                };
                BearingCorrespondence::try_new(
                    jitter(bearing1, &mut noise),
                    jitter(bearing2, &mut noise),
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn cheirality_ignores_correspondences_without_parallax_when_all_of_them_disagree() {
        let rotation = Rotation3::from_euler_angles(0.02, -0.03, 0.01);
        let center = Vector3::new(0.3, 0.05, 0.1);
        let correspondences = short_baseline_sphere(&rotation, &center);
        let poses = make_essential(&rotation, &center).decompose().unwrap();
        let all: Vec<&BearingCorrespondence> = correspondences.iter().collect();

        // The distant half has random depth signs, so no hypothesis reaches 90% of everything.
        assert!(
            select_among(&poses, &all, ALL_MIN_FRONT_PERCENT)
                .0
                .is_none()
        );

        let selected = make_essential(&rotation, &center)
            .decompose_with_cheirality(&correspondences)
            .expect("the near half decides the pose");
        let expected = -(rotation * center).normalize();
        assert!(
            (selected.translation_source_to_target().as_ref() - expected).norm() < 0.05,
            "translation {:?}, expected {expected:?}",
            selected.translation_source_to_target()
        );
        let relative =
            selected.rotation_source_to_target().matrix().transpose() * rotation.matrix();
        let angle = ((relative.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        assert!(angle < 0.05, "rotation off by {angle}");
    }

    #[test]
    fn cheirality_still_rejects_when_too_few_correspondences_have_parallax() {
        let rotation = Rotation3::from_euler_angles(0.02, -0.03, 0.01);
        let center = Vector3::new(0.3, 0.05, 0.1);
        let few: Vec<_> = short_baseline_sphere(&rotation, &center)
            .into_iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 1 || *index < 20)
            .map(|(_, correspondence)| correspondence)
            .collect();

        assert!(
            make_essential(&rotation, &center)
                .decompose_with_cheirality(&few)
                .is_none()
        );
    }
}
