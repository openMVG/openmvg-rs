// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::normalization::PixelNormalization;
use nalgebra::{Point2, Rotation3, SymmetricEigen, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roots::{Roots, find_roots_cubic};

fn skew(vector: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}

fn calibrated_two_view_scene() -> (Vec<PixelCorrespondence>, Matrix3<f64>) {
    let rotation = Rotation3::from_euler_angles(0.07, -0.13, 0.04);
    let translation = Vector3::new(0.55, -0.16, 0.09);
    let points = [
        Vector3::new(-1.2, -0.7, 4.0),
        Vector3::new(0.3, -1.1, 5.2),
        Vector3::new(1.4, -0.2, 6.1),
        Vector3::new(-0.8, 1.3, 4.7),
        Vector3::new(0.6, 0.9, 7.0),
        Vector3::new(1.7, 1.1, 5.5),
        Vector3::new(-1.5, 0.4, 6.6),
    ];
    let correspondences = points
        .iter()
        .map(|point| {
            let second = rotation * point + translation;
            PixelCorrespondence::from_coords(
                point.x / point.z,
                point.y / point.z,
                second.x / second.z,
                second.y / second.z,
            )
        })
        .collect();
    (correspondences, skew(translation) * rotation.matrix())
}

fn projective_distance(left: &Matrix3<f64>, right: &Matrix3<f64>) -> f64 {
    let left = left / left.norm();
    let right = right / right.norm();
    (left - right).norm().min((left + right).norm())
}

fn solve_with_normalization(correspondences: &[PixelCorrespondence]) -> Vec<FundamentalMatrix> {
    SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(correspondences)).unwrap()
}

fn transform_correspondences(
    correspondences: &[PixelCorrespondence],
    first: &Matrix3<f64>,
    second: &Matrix3<f64>,
) -> Vec<PixelCorrespondence> {
    correspondences
        .iter()
        .map(|correspondence| {
            let p1 = first * Vector3::new(correspondence.point1.x, correspondence.point1.y, 1.0);
            let p2 = second * Vector3::new(correspondence.point2.x, correspondence.point2.y, 1.0);
            PixelCorrespondence::from_coords(p1.x / p1.z, p1.y / p1.z, p2.x / p2.z, p2.y / p2.z)
        })
        .collect()
}

fn legacy_ata_nullspace(a: &SMatrix<f64, 7, 9>) -> (SVector<f64, 9>, SVector<f64, 9>) {
    let eigen = SymmetricEigen::new(a.transpose() * a);
    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
    order.sort_by(|&left, &right| {
        eigen.eigenvalues[left]
            .abs()
            .total_cmp(&eigen.eigenvalues[right].abs())
    });
    (
        eigen.eigenvectors.column(order[0]).into(),
        eigen.eigenvectors.column(order[1]).into(),
    )
}

fn relative_null_residual(a: &SMatrix<f64, 7, 9>, vector: &SVector<f64, 9>) -> f64 {
    (a * vector).norm() / (a.norm() * vector.norm())
}

fn row_major_vector(matrix: &Matrix3<f64>) -> SVector<f64, 9> {
    SVector::from_row_slice(&[
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
    ])
}

fn svd_rank_two_projection(matrix: Matrix3<f64>) -> Matrix3<f64> {
    let svd = matrix.svd(true, true);
    let mut singular_values = svd.singular_values;
    singular_values[2] = 0.0;
    svd.u.unwrap() * Matrix3::from_diagonal(&singular_values) * svd.v_t.unwrap()
}

fn solve_legacy_ata(
    correspondences: &[PixelCorrespondence],
) -> SolverResult<Vec<FundamentalMatrix>> {
    let a = build_constraint_matrix(correspondences);
    if !a.iter().all(|value| value.is_finite()) {
        return Err(SolverError::DegenerateConfiguration {
            reason: "legacy ATA input is non-finite".to_string(),
        });
    }
    let (first_vector, second_vector) = legacy_ata_nullspace(&a);
    let first = vector9_to_matrix3(&first_vector);
    let second = vector9_to_matrix3(&second_vector);
    let difference = first - second;
    let coefficients = [
        second.determinant(),
        determinant_linear_coefficient(&second, &difference),
        determinant_linear_coefficient(&difference, &second),
        difference.determinant(),
    ];
    let roots = find_roots_cubic(
        coefficients[3],
        coefficients[2],
        coefficients[1],
        coefficients[0],
    );
    let mut models = Vec::with_capacity(3);
    let mut add_root = |root: f64| {
        if !root.is_finite() {
            return;
        }
        let matrix = first * root + second * (1.0 - root);
        let norm = matrix.norm();
        if !norm.is_finite() || norm <= 0.0 {
            return;
        }
        let projected = svd_rank_two_projection(matrix / norm);
        if projected.iter().all(|value| value.is_finite()) {
            models.push(FundamentalMatrix::new(projected / projected.norm()));
        }
    };
    match roots {
        Roots::No(_) => {}
        Roots::One(values) => values.into_iter().for_each(&mut add_root),
        Roots::Two(values) => values.into_iter().for_each(&mut add_root),
        Roots::Three(values) => values.into_iter().for_each(&mut add_root),
        Roots::Four(values) => values.into_iter().for_each(&mut add_root),
    }
    if models.is_empty() {
        Err(SolverError::DegenerateConfiguration {
            reason: "legacy ATA returned no real model".to_string(),
        })
    } else {
        Ok(models)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StressKind {
    Ordinary,
    Noisy,
    HighDynamicRange,
    NearPlanar,
    NearCollinear,
    NearRankDeficient,
}

impl StressKind {
    const ALL: [Self; 6] = [
        Self::Ordinary,
        Self::Noisy,
        Self::HighDynamicRange,
        Self::NearPlanar,
        Self::NearCollinear,
        Self::NearRankDeficient,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::Noisy => "noisy",
            Self::HighDynamicRange => "high_dynamic_range",
            Self::NearPlanar => "near_planar",
            Self::NearCollinear => "near_collinear",
            Self::NearRankDeficient => "near_rank_deficient",
        }
    }

    fn is_well_conditioned_family(self) -> bool {
        matches!(self, Self::Ordinary | Self::Noisy | Self::HighDynamicRange)
    }
}

struct StressCase {
    correspondences: Vec<PixelCorrespondence>,
    reference: Matrix3<f64>,
    first_similarity: Matrix3<f64>,
    second_similarity: Matrix3<f64>,
}

fn random_similarity(rng: &mut StdRng) -> Matrix3<f64> {
    let scale = 10.0_f64.powf(rng.gen_range(-6.0..6.0));
    let angle = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
    let cosine = angle.cos();
    let sine = angle.sin();
    let offset_scale = scale * 10.0_f64.powf(rng.gen_range(2.0..8.0));
    Matrix3::new(
        scale * cosine,
        -scale * sine,
        rng.gen_range(-offset_scale..offset_scale),
        scale * sine,
        scale * cosine,
        rng.gen_range(-offset_scale..offset_scale),
        0.0,
        0.0,
        1.0,
    )
}

fn generate_stress_case(kind: StressKind, index: usize, seed: u64) -> StressCase {
    let kind_offset = StressKind::ALL
        .iter()
        .position(|candidate| *candidate == kind)
        .unwrap() as u64;
    let mut rng = StdRng::seed_from_u64(
        seed ^ kind_offset.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ index as u64,
    );
    let rotation = Rotation3::from_euler_angles(
        rng.gen_range(-0.25..0.25),
        rng.gen_range(-0.25..0.25),
        rng.gen_range(-0.25..0.25),
    );
    let mut translation = Vector3::new(
        rng.gen_range(-0.8..0.8),
        rng.gen_range(-0.8..0.8),
        rng.gen_range(-0.25..0.25),
    );
    if translation.norm() < 0.2 {
        translation.x += 0.45;
    }
    let reference = skew(translation) * rotation.matrix();
    let level = 10.0_f64.powi(-2 - (index % 11) as i32);

    let mut points = Vec::with_capacity(7);
    match kind {
        StressKind::NearPlanar => {
            for _ in 0..7 {
                let x = rng.gen_range(-2.0..2.0);
                let y = rng.gen_range(-1.7..1.7);
                let z = 6.0 + 0.25 * x - 0.18 * y + level * rng.gen_range(-1.0..1.0);
                points.push(Vector3::new(x, y, z));
            }
        }
        StressKind::NearCollinear => {
            let origin = Vector3::new(-0.4, 0.2, 6.0);
            let direction = Vector3::new(0.8, 0.35, 0.45);
            for point_index in 0..7 {
                let parameter = -1.5 + 0.5 * point_index as f64;
                let perturbation = Vector3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-0.2..0.2),
                ) * level;
                points.push(origin + direction * parameter + perturbation);
            }
        }
        _ => {
            for _ in 0..7 {
                points.push(Vector3::new(
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-1.7..1.7),
                    rng.gen_range(4.0..9.0),
                ));
            }
            if kind == StressKind::NearRankDeficient {
                points[6] = points[0]
                    + Vector3::new(
                        rng.gen_range(-1.0..1.0),
                        rng.gen_range(-1.0..1.0),
                        rng.gen_range(-1.0..1.0),
                    ) * level;
            }
        }
    }

    let noise = if kind == StressKind::Noisy {
        10.0_f64.powi(-3 - (index % 6) as i32)
    } else {
        0.0
    };
    let mut correspondences: Vec<_> = points
        .iter()
        .map(|point| {
            let second = rotation * point + translation;
            PixelCorrespondence::from_coords(
                point.x / point.z + rng.gen_range(-noise..=noise),
                point.y / point.z + rng.gen_range(-noise..=noise),
                second.x / second.z + rng.gen_range(-noise..=noise),
                second.y / second.z + rng.gen_range(-noise..=noise),
            )
        })
        .collect();

    let (first_similarity, second_similarity) = if kind == StressKind::HighDynamicRange {
        let first = random_similarity(&mut rng);
        let second = random_similarity(&mut rng);
        correspondences = transform_correspondences(&correspondences, &first, &second);
        (first, second)
    } else {
        (Matrix3::identity(), Matrix3::identity())
    };

    StressCase {
        correspondences,
        reference,
        first_similarity,
        second_similarity,
    }
}

#[derive(Debug, Clone, Copy)]
struct QualityMetrics {
    algebraic: f64,
    sampson: f64,
    rank_two_error: f64,
    ground_truth_distance: f64,
}

#[derive(Debug)]
struct MethodOutcome {
    model_count: usize,
    metrics: Option<QualityMetrics>,
    error: Option<String>,
}

fn evaluate_models(
    result: SolverResult<Vec<FundamentalMatrix>>,
    normalized: &[PixelCorrespondence],
    normalization: &PixelNormalization,
    case: &StressCase,
) -> MethodOutcome {
    let models = match result {
        Ok(models) if !models.is_empty() => models,
        Ok(_) => {
            return MethodOutcome {
                model_count: 0,
                metrics: None,
                error: Some("empty model set".to_string()),
            };
        }
        Err(error) => {
            return MethodOutcome {
                model_count: 0,
                metrics: None,
                error: Some(format!("{error:?}")),
            };
        }
    };
    let constraints = build_constraint_matrix(normalized);
    let constraint_norm = constraints.norm();
    let mut best = QualityMetrics {
        algebraic: f64::INFINITY,
        sampson: f64::INFINITY,
        rank_two_error: 0.0,
        ground_truth_distance: f64::INFINITY,
    };
    for model in &models {
        let vector = row_major_vector(&model.0);
        best.algebraic = best
            .algebraic
            .min((constraints * vector).norm() / (constraint_norm * vector.norm()));
        let mean_sampson = normalized
            .iter()
            .map(|correspondence| SevenPointSolver::residual(model, correspondence))
            .sum::<f64>()
            / normalized.len() as f64;
        best.sampson = best.sampson.min(mean_sampson);
        let singular_values = model.0.svd(false, false).singular_values;
        best.rank_two_error = best
            .rank_two_error
            .max(singular_values[2] / singular_values[0]);

        let denormalized = crate::fundamental::denormalize(model, normalization);
        let mapped_back =
            case.second_similarity.transpose() * denormalized.0 * case.first_similarity;
        best.ground_truth_distance = best
            .ground_truth_distance
            .min(projective_distance(&mapped_back, &case.reference));
    }
    let finite = best.algebraic.is_finite()
        && best.sampson.is_finite()
        && best.rank_two_error.is_finite()
        && best.ground_truth_distance.is_finite();
    MethodOutcome {
        model_count: models.len(),
        metrics: finite.then_some(best),
        error: (!finite).then(|| "non-finite quality metric".to_string()),
    }
}

#[derive(Debug)]
struct CaseComparison {
    kind: StressKind,
    index: usize,
    current: MethodOutcome,
    legacy: MethodOutcome,
}

fn run_stress_corpus(per_kind: usize, seed: u64) -> Vec<CaseComparison> {
    let mut comparisons = Vec::with_capacity(per_kind * StressKind::ALL.len());
    for kind in StressKind::ALL {
        for index in 0..per_kind {
            let case = generate_stress_case(kind, index, seed);
            let set = NormalizedCorrespondences::from_pixels(&case.correspondences);
            let normalization = set.normalization();
            let normalized = normalization.normalize(&case.correspondences);
            let current = evaluate_models(
                <SevenPointSolver as MultiSolutionSolver>::solve(set.as_slice(), &()),
                &normalized,
                normalization,
                &case,
            );
            let legacy = evaluate_models(
                solve_legacy_ata(&normalized),
                &normalized,
                normalization,
                &case,
            );
            comparisons.push(CaseComparison {
                kind,
                index,
                current,
                legacy,
            });
        }
    }
    comparisons
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * probability).round() as usize;
    sorted[index]
}

fn corpus_report(comparisons: &[CaseComparison]) -> String {
    let mut report = String::new();
    for kind in StressKind::ALL {
        let cases: Vec<_> = comparisons
            .iter()
            .filter(|comparison| comparison.kind == kind)
            .collect();
        for name in ["current", "legacy"] {
            let outcomes: Vec<&MethodOutcome> = cases
                .iter()
                .map(|comparison| {
                    if name == "current" {
                        &comparison.current
                    } else {
                        &comparison.legacy
                    }
                })
                .collect();
            let metrics: Vec<_> = outcomes
                .iter()
                .filter_map(|outcome| outcome.metrics)
                .collect();
            let algebraic: Vec<_> = metrics.iter().map(|metric| metric.algebraic).collect();
            let sampson: Vec<_> = metrics.iter().map(|metric| metric.sampson).collect();
            let rank: Vec<_> = metrics.iter().map(|metric| metric.rank_two_error).collect();
            let distance: Vec<_> = metrics
                .iter()
                .map(|metric| metric.ground_truth_distance)
                .collect();
            let mean_models = outcomes
                .iter()
                .map(|outcome| outcome.model_count as f64)
                .sum::<f64>()
                / outcomes.len().max(1) as f64;
            let mut model_histogram = [0usize; 4];
            for outcome in &outcomes {
                model_histogram[outcome.model_count.min(3)] += 1;
            }
            report.push_str(&format!(
                "{} {}: success={}/{}, models_mean={:.3}, models_hist[0,1,2,3]={:?}, algebraic[p50={:.3e},p95={:.3e},max={:.3e}], sampson[p50={:.3e},p95={:.3e},max={:.3e}], rank_worst[p95={:.3e},max={:.3e}], gt[p50={:.3e},p95={:.3e},max={:.3e}]\n",
                kind.name(),
                name,
                metrics.len(),
                outcomes.len(),
                mean_models,
                model_histogram,
                percentile(&algebraic, 0.5),
                percentile(&algebraic, 0.95),
                percentile(&algebraic, 1.0),
                percentile(&sampson, 0.5),
                percentile(&sampson, 0.95),
                percentile(&sampson, 1.0),
                percentile(&rank, 0.95),
                percentile(&rank, 1.0),
                percentile(&distance, 0.5),
                percentile(&distance, 0.95),
                percentile(&distance, 1.0),
            ));
        }
        let paired: Vec<_> = cases
            .iter()
            .filter_map(|comparison| {
                Some((
                    comparison.current.metrics?,
                    comparison.legacy.metrics?,
                    comparison.current.model_count == comparison.legacy.model_count,
                ))
            })
            .collect();
        let current_algebraic_wins = paired
            .iter()
            .filter(|pair| pair.0.algebraic <= pair.1.algebraic)
            .count();
        let current_sampson_wins = paired
            .iter()
            .filter(|pair| pair.0.sampson <= pair.1.sampson)
            .count();
        let current_ground_truth_wins = paired
            .iter()
            .filter(|pair| pair.0.ground_truth_distance <= pair.1.ground_truth_distance)
            .count();
        let equal_model_counts = paired.iter().filter(|pair| pair.2).count();
        report.push_str(&format!(
            "{} paired={}: current_wins[algebraic={},sampson={},gt={}], equal_model_count={}\n",
            kind.name(),
            paired.len(),
            current_algebraic_wins,
            current_sampson_wins,
            current_ground_truth_wins,
            equal_model_counts,
        ));
    }
    let failures: Vec<_> = comparisons
        .iter()
        .filter(|comparison| {
            comparison.current.metrics.is_none() || comparison.legacy.metrics.is_none()
        })
        .take(16)
        .map(|comparison| {
            format!(
                "{}[{}]: current={:?}, legacy={:?}",
                comparison.kind.name(),
                comparison.index,
                comparison.current.error,
                comparison.legacy.error
            )
        })
        .collect();
    if !failures.is_empty() {
        report.push_str("first failures:\n");
        report.push_str(&failures.join("\n"));
        report.push('\n');
    }
    report
}

fn assert_corpus_quality(comparisons: &[CaseComparison], report: &str) {
    let well_conditioned: Vec<_> = comparisons
        .iter()
        .filter(|comparison| comparison.kind.is_well_conditioned_family())
        .collect();
    let current_successes = well_conditioned
        .iter()
        .filter(|comparison| comparison.current.metrics.is_some())
        .count();
    assert_eq!(
        current_successes,
        well_conditioned.len(),
        "current solver rejected a regular/noisy/HDR case:\n{report}"
    );

    let paired: Vec<_> = well_conditioned
        .iter()
        .filter_map(|comparison| Some((comparison.current.metrics?, comparison.legacy.metrics?)))
        .collect();
    assert!(
        paired.len() * 100 >= well_conditioned.len() * 95,
        "legacy comparison coverage fell below 95%:\n{report}"
    );
    let equal_model_counts = well_conditioned
        .iter()
        .filter(|comparison| {
            comparison.current.metrics.is_some()
                && comparison.legacy.metrics.is_some()
                && comparison.current.model_count == comparison.legacy.model_count
        })
        .count();
    assert!(
        equal_model_counts * 100 >= paired.len() * 95,
        "model-count agreement fell below 95% on regular/noisy/HDR cases:\n{report}"
    );
    let current_algebraic: Vec<_> = paired.iter().map(|pair| pair.0.algebraic).collect();
    let legacy_algebraic: Vec<_> = paired.iter().map(|pair| pair.1.algebraic).collect();
    let current_sampson: Vec<_> = paired.iter().map(|pair| pair.0.sampson).collect();
    let legacy_sampson: Vec<_> = paired.iter().map(|pair| pair.1.sampson).collect();
    let current_rank: Vec<_> = paired.iter().map(|pair| pair.0.rank_two_error).collect();
    let current_distance: Vec<_> = paired
        .iter()
        .map(|pair| pair.0.ground_truth_distance)
        .collect();
    let legacy_distance: Vec<_> = paired
        .iter()
        .map(|pair| pair.1.ground_truth_distance)
        .collect();

    assert!(
        percentile(&current_algebraic, 0.95) <= percentile(&legacy_algebraic, 0.95) * 10.0 + 1e-14,
        "current algebraic p95 regressed:\n{report}"
    );
    assert!(
        percentile(&current_sampson, 0.95) <= percentile(&legacy_sampson, 0.95) * 10.0 + 1e-12,
        "current Sampson p95 regressed:\n{report}"
    );
    assert!(
        percentile(&current_rank, 1.0) < 1e-6,
        "current rank-2 error exceeded the model contract:\n{report}"
    );
    assert!(
        percentile(&current_distance, 0.95) <= percentile(&legacy_distance, 0.95) * 1.1 + 2e-6,
        "current ground-truth projective p95 regressed:\n{report}"
    );
}

#[test]
fn deterministic_randomized_corpus_matches_or_beats_legacy_ata() {
    let comparisons = run_stress_corpus(8, 0xA11C_E5E7_2026);
    let report = corpus_report(&comparisons);
    eprintln!("seven-point default corpus:\n{report}");
    assert_corpus_quality(&comparisons, &report);
}

#[test]
fn test_seven_point_validity_uses_rank_not_matrix_scale() {
    let rank_two = FundamentalMatrix::new(Matrix3::new(
        1e-12, 0.0, 0.0, 0.0, 5e-13, 0.0, 0.0, 0.0, 0.0,
    ));
    let rank_one =
        FundamentalMatrix::new(Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(SevenPointSolver::is_valid(&rank_two));
    assert!(!SevenPointSolver::is_valid(&rank_one));
}

/// Create synthetic correspondences from a known fundamental matrix.
fn create_synthetic_correspondences(
    f: &Matrix3<f64>,
    num_points: usize,
) -> Vec<PixelCorrespondence> {
    let mut correspondences = Vec::with_capacity(num_points);

    for i in 0..num_points {
        // Generate points in image 1
        let x1 = 100.0 + (i as f64) * 50.0 + (i as f64 * 7.3).sin() * 30.0;
        let y1 = 200.0 + (i as f64) * 30.0 + (i as f64 * 5.1).cos() * 40.0;
        let p1 = Point2::new(x1, y1);

        // Compute epipolar line in image 2: l2 = F * [x1, y1, 1]^T
        let x1_h = Vector3::new(x1, y1, 1.0);
        let l2 = f * x1_h;

        // Find a point on this line
        let x2 = 150.0 + (i as f64) * 60.0;
        let y2 = if l2.y.abs() > 1e-10 {
            -(l2.x * x2 + l2.z) / l2.y
        } else if l2.x.abs() > 1e-10 {
            let y2 = 200.0 + (i as f64) * 40.0;
            // x = -(l.y * y + l.z) / l.x
            let x2_new = -(l2.y * y2 + l2.z) / l2.x;
            correspondences.push(PixelCorrespondence::new(p1, Point2::new(x2_new, y2)));
            continue;
        } else {
            200.0 + (i as f64) * 40.0
        };
        let p2 = Point2::new(x2, y2);

        correspondences.push(PixelCorrespondence::new(p1, p2));
    }

    correspondences
}

#[test]
fn test_seven_point_solver_basic() {
    // Create a fundamental matrix (rank 2)
    // Translation along x-axis
    let f_true = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);

    // Generate exactly 7 correspondences
    let correspondences = create_synthetic_correspondences(&f_true, 7);

    // Normalize data (as RANSAC would do)
    let normalized_data = NormalizedCorrespondences::from_pixels(&correspondences);

    // The public solve returns models in the caller's coordinate frame.
    let result = SevenPointSolver::solve(&normalized_data);

    assert!(result.is_ok(), "Solver failed: {:?}", result.err());
    let models = result.unwrap();

    // Should return 1-3 solutions
    assert!(!models.is_empty(), "Should return at least one model");
    assert!(models.len() <= 3, "Should return at most 3 models");

    // At least one solution should have low algebraic error
    let mut found_good_solution = false;
    for f in &models {
        let mut total_error = 0.0;
        for corr in &correspondences {
            let x1 = Vector3::new(corr.point1.x, corr.point1.y, 1.0);
            let x2 = Vector3::new(corr.point2.x, corr.point2.y, 1.0);
            let error = (x2.transpose() * f.0 * x1)[(0, 0)].abs();
            total_error += error;
        }
        let avg_error = total_error / correspondences.len() as f64;
        if avg_error < 1.0 {
            found_good_solution = true;
            break;
        }
    }

    assert!(
        found_good_solution,
        "No solution with low algebraic error found"
    );
}

#[test]
fn test_direct_nullspace_matches_independent_camera_reference() {
    let (correspondences, reference) = calibrated_two_view_scene();
    let models = solve_with_normalization(&correspondences);
    let best_distance = models
        .iter()
        .map(|model| projective_distance(&model.0, &reference))
        .fold(f64::INFINITY, f64::min);
    assert!(best_distance < 1e-8, "reference distance: {best_distance}");
}

#[test]
fn test_direct_nullspace_residual_and_orthogonality() {
    let (correspondences, _) = calibrated_two_view_scene();
    let a = build_constraint_matrix(&correspondences);
    let (first, second) = direct_nullspace(&a).unwrap();
    assert!(relative_null_residual(&a, &first) < 1e-15);
    assert!(relative_null_residual(&a, &second) < 1e-15);
    assert!(first.dot(&second).abs() < 1e-14);
    assert!((first.norm() - 1.0).abs() < 1e-14);
    assert!((second.norm() - 1.0).abs() < 1e-14);
}

#[test]
fn test_similarity_scale_invariance_over_high_dynamic_range() {
    let (correspondences, reference) = calibrated_two_view_scene();
    let first = Matrix3::new(2.0e6, 0.0, 8.0e8, 0.0, 2.0e6, -3.0e8, 0.0, 0.0, 1.0);
    let second = Matrix3::new(3.0e-4, 0.0, -2.0e4, 0.0, 3.0e-4, 7.0e4, 0.0, 0.0, 1.0);
    let transformed = transform_correspondences(&correspondences, &first, &second);
    let models = solve_with_normalization(&transformed);
    let best_distance = models
        .iter()
        .map(|model| {
            let mapped_back = second.transpose() * model.0 * first;
            projective_distance(&mapped_back, &reference)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        best_distance < 2e-6,
        "scale-invariant distance: {best_distance}"
    );
}

#[test]
fn test_near_degenerate_noisy_sample_remains_solvable() {
    let fundamental = Matrix3::new(0.0, -0.02, 0.1, 0.015, 0.0, -1.0, -0.08, 1.0, 0.02);
    let mut correspondences = Vec::with_capacity(7);
    for i in 0..7 {
        let x1 = -1.5 + 0.5 * i as f64;
        let y1 = 0.2 * x1 + 0.015 * (1.7 * i as f64).sin();
        let line = fundamental * Vector3::new(x1, y1, 1.0);
        let x2 = -1.2 + 0.43 * i as f64 + 0.08 * (0.9 * i as f64).cos();
        let mut y2 = -(line.x * x2 + line.z) / line.y;
        y2 += 1e-7 * (i as f64 + 0.3).sin();
        correspondences.push(PixelCorrespondence::from_coords(x1, y1, x2, y2));
    }
    let models = solve_with_normalization(&correspondences);
    assert!(!models.is_empty());
    let best_max_residual = models
        .iter()
        .map(|model| {
            correspondences
                .iter()
                .map(|corr| SevenPointSolver::residual(model, corr))
                .fold(0.0_f64, f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        best_max_residual < 1e-4,
        "near-degenerate residual: {best_max_residual}"
    );
}

#[test]
fn test_cpqr_beats_normal_equations_on_high_dynamic_range_constraints() {
    let (correspondences, _) = calibrated_two_view_scene();
    let first = Matrix3::new(4.0e5, 0.0, 8.0e6, 0.0, 4.0e5, -3.0e6, 0.0, 0.0, 1.0);
    let second = Matrix3::new(2.0e3, 0.0, -9.0e5, 0.0, 2.0e3, 5.0e5, 0.0, 0.0, 1.0);
    let transformed = transform_correspondences(&correspondences, &first, &second);
    let a = build_constraint_matrix(&transformed);
    let (direct_first, direct_second) = direct_nullspace(&a).unwrap();
    let (legacy_first, legacy_second) = legacy_ata_nullspace(&a);
    let direct =
        relative_null_residual(&a, &direct_first).max(relative_null_residual(&a, &direct_second));
    let legacy =
        relative_null_residual(&a, &legacy_first).max(relative_null_residual(&a, &legacy_second));
    assert!(
        direct < legacy * 0.1,
        "direct={direct:e}, legacy={legacy:e}"
    );
}

#[test]
fn test_preserving_nullspace_is_no_worse_than_svd_rank_projection() {
    let (correspondences, _) = calibrated_two_view_scene();
    let normalized = NormalizedCorrespondences::from_pixels(&correspondences);
    let a = build_constraint_matrix(normalized.as_slice());
    let models =
        <SevenPointSolver as MultiSolutionSolver>::solve(normalized.as_slice(), &()).unwrap();
    for model in models {
        let direct = (a * row_major_vector(&model.0)).norm();
        let projected_matrix = svd_rank_two_projection(model.0);
        let projected = (a * row_major_vector(&projected_matrix)).norm();
        assert!(
            direct <= projected + 32.0 * f64::EPSILON * a.norm(),
            "direct={direct:e}, projected={projected:e}"
        );
    }
}

#[test]
fn test_seven_point_rejects_repeated_sample() {
    let correspondence = PixelCorrespondence::from_coords(0.1, -0.2, 0.15, -0.1);
    let result = SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(
        &[correspondence; 7],
    ));
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_seven_point_accepts_small_measurement_noise() {
    let f_true = Matrix3::new(0.0, -0.01, 0.08, 0.012, 0.0, -1.0, -0.07, 1.0, 0.0);
    let mut correspondences = create_synthetic_correspondences(&f_true, 7);
    for (i, correspondence) in correspondences.iter_mut().enumerate() {
        correspondence.point2.x += 1e-3 * (i as f64 + 0.2).sin();
        correspondence.point2.y += 1e-3 * (i as f64 + 0.5).cos();
    }
    let models =
        SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(&correspondences)).unwrap();
    assert!(!models.is_empty());
    assert!(models.iter().all(SevenPointSolver::is_valid));
}

#[test]
fn test_seven_point_requires_exactly_seven() {
    let correspondences: Vec<PixelCorrespondence> = (0..8)
        .map(|i| {
            PixelCorrespondence::from_coords(
                100.0 + i as f64 * 10.0,
                200.0,
                150.0 + i as f64 * 10.0,
                250.0,
            )
        })
        .collect();

    let result = SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(&correspondences));

    assert!(result.is_err());
    if let Err(SolverError::InvalidSampleSize { expected, provided }) = result {
        assert_eq!(expected, 7);
        assert_eq!(provided, 8);
    } else {
        panic!("Expected InvalidSampleSize error");
    }
}

#[test]
fn test_cubic_coefficients() {
    // Test with simple matrices
    let f1 = Matrix3::<f64>::identity();
    let f2 = Matrix3::zeros();

    let coeffs = compute_det_cubic_coefficients(&f1, &f2);

    // F(α) = α*I + (1-α)*0 = α*I
    // det(α*I) = α³
    // So coefficients should be [0, 0, 0, 1]
    assert!((coeffs[0] - 0.0).abs() < 1e-10);
    assert!((coeffs[1] - 0.0).abs() < 1e-10);
    assert!((coeffs[2] - 0.0).abs() < 1e-10);
    assert!((coeffs[3] - 1.0).abs() < 1e-10);
}

#[test]
fn test_cubic_coefficients_match_direct_determinant() {
    let f1: Matrix3<f64> = Matrix3::new(0.2, -0.7, 1.1, 0.4, 0.8, -0.3, -0.9, 0.5, 0.6);
    let f2: Matrix3<f64> = Matrix3::new(-0.6, 0.3, 0.7, 1.2, -0.4, 0.2, 0.1, 0.9, -0.8);
    let [c0, c1, c2, c3] = compute_det_cubic_coefficients(&f1, &f2);
    for root in [-3.0, -0.4, 0.0, 0.7, 2.5] {
        let polynomial = ((c3 * root + c2) * root + c1) * root + c0;
        let determinant = (f1 * root + f2).determinant();
        assert!((polynomial - determinant).abs() < 1e-12);
    }
}

#[test]
fn test_projective_infinite_and_large_root_construction() {
    let f1: Matrix3<f64> = Matrix3::new(1.0, 0.2, 0.0, -0.1, 0.7, 0.0, 0.3, -0.4, 0.0);
    let f2: Matrix3<f64> = Matrix3::new(0.0, 0.0, 0.4, 0.0, 0.0, -0.2, 0.0, 0.0, 0.1);
    let at_infinity = construct_fundamental_matrix(&f1, &f2, None).unwrap();
    let large = construct_fundamental_matrix(&f1, &f2, Some(1e300)).unwrap();
    assert!(at_infinity.0.iter().all(|value| value.is_finite()));
    assert!(large.0.iter().all(|value| value.is_finite()));
    assert!(projective_distance(&large.0, &at_infinity.0) < 1e-12);
}

#[test]
fn test_constructor_conditioning_is_separate_from_pixel_model_validity() {
    let zero = Matrix3::zeros();
    let near_rank_one = Matrix3::from_diagonal(&Vector3::new(1.0, 0.5e-6, 0.0));
    assert!(construct_fundamental_matrix(&near_rank_one, &zero, None).is_none());
    assert!(FundamentalMatrix::new(near_rank_one).is_valid());

    let rank_two = Matrix3::from_diagonal(&Vector3::new(1.0, 2.0e-6, 0.0));
    let model = construct_fundamental_matrix(&rank_two, &zero, None).unwrap();
    assert!(model.is_valid());
}

#[test]
fn test_projective_chart_stabilizes_unbalanced_determinant_coefficients() {
    // det(x*F1 + F2) has roots 1e14, 7, and 0.75, so its cubic
    // coefficient is fourteen orders of magnitude below its constant.
    let f1 = Matrix3::from_diagonal(&Vector3::new(1e-14, 1.0 / 7.0, 4.0 / 3.0));
    let f2 = -Matrix3::identity();
    let coefficients = compute_det_cubic_coefficients(&f1, &f2);
    let scale = coefficients
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let mut charted = coefficients.map(|coefficient| coefficient / scale);
    assert!(charted[3].abs() < charted[0].abs() * 1e-12);

    let (direction, origin) = choose_projective_root_chart(&f1, &f2, &mut charted);
    assert!(std::ptr::eq(direction, &f2));
    assert!(std::ptr::eq(origin, &f1));
    assert!(charted[3].abs() > 0.5);

    let mut roots = [0.0; 3];
    let root_count = cubic_real([charted[3], charted[2], charted[1], charted[0]], &mut roots);
    assert_eq!(root_count, 3);
    let models: Vec<_> = roots
        .into_iter()
        .take(root_count)
        .map(|root| construct_fundamental_matrix(direction, origin, Some(root)).unwrap())
        .collect();
    assert!(models.iter().all(FundamentalMatrix::is_valid));
    assert!(
        models
            .iter()
            .all(|model| model.0.determinant().abs() < 1e-15)
    );
}

#[test]
fn test_rotated_chart_preserves_cubic_when_both_endpoints_are_tiny() {
    let a = std::f64::consts::FRAC_1_SQRT_2;
    let f1 = Matrix3::new(0.0, -a, 0.0, -2e-15, 0.0, 0.0, 0.0, 0.0, -a);
    let f1 = f1 / f1.norm();
    let seed_f2 = Matrix3::from_diagonal(&Vector3::new(-a, -a, 2e-15));
    let f2 = seed_f2 - f1 * seed_f2.dot(&f1);
    let f2 = f2 / f2.norm();

    let coefficients = normalized_det_coefficients(&f1, &f2).unwrap();
    assert!(coefficients[0].abs().max(coefficients[3].abs()) < 3e-15);
    assert!(coefficients[1].abs().max(coefficients[2].abs()) > 0.9);

    let models = solve_with_rotated_root_chart(&f1, &f2, coefficients).unwrap();
    assert_eq!(models.len(), 3);
    assert!(models.iter().all(FundamentalMatrix::is_valid));
}

#[test]
fn test_adjugate() {
    let m = Matrix3::<f64>::new(1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0);

    let adj = adjugate(&m);
    let det = m.determinant();

    // adj(M) * M = det(M) * I
    let product = adj * m;
    let expected = Matrix3::<f64>::identity() * det;

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (product[(i, j)] - expected[(i, j)]).abs() < 1e-10,
                "Adjugate property failed at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn indexed_emission_matches_solve_without_allocating_the_sample() {
    let (correspondences, _) = calibrated_two_view_scene();
    let normalized = NormalizedCorrespondences::from_pixels(&correspondences);
    let data = normalized.as_slice();
    let reference = <SevenPointSolver as MultiSolutionSolver>::solve(data, &()).unwrap();
    assert!(!reference.is_empty());

    let mut emitted = Vec::new();
    let count = <SevenPointSolver as MultiSolutionSolver>::for_each_solution_from_indices(
        &[0, 1, 2, 3, 4, 5, 6],
        data,
        &(),
        |model| emitted.push(model),
    )
    .unwrap();
    assert_eq!(count, reference.len());
    assert_eq!(emitted, reference);

    let error = <SevenPointSolver as MultiSolutionSolver>::for_each_solution_from_indices(
        &[0, 1, 2, 3, 4, 5, 99],
        data,
        &(),
        |_| {},
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SolverError::InvalidDataIndex { index: 99, .. }
    ));
}
