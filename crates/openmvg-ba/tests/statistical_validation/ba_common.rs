// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::RelativePose;

use super::common::{COVARIANCE_TRIALS, NOMINAL_COVERAGE, chi_square_quantiles};

#[derive(Default)]
pub(super) struct ClusteredCoverage {
    name: String,
    trial_mean_nees: Vec<f64>,
    trial_coverage: [Vec<f64>; 3],
}

impl ClusteredCoverage {
    pub(super) fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            trial_mean_nees: Vec::with_capacity(COVARIANCE_TRIALS),
            trial_coverage: std::array::from_fn(|_| Vec::with_capacity(COVARIANCE_TRIALS)),
        }
    }

    pub(super) fn push_trial(&mut self, values: impl Iterator<Item = f64>) {
        let values: Vec<_> = values.collect();
        assert!(!values.is_empty(), "empty point cluster {}", self.name);
        assert!(
            values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        self.trial_mean_nees
            .push(values.iter().sum::<f64>() / values.len() as f64);
        for (coverage, quantile) in self.trial_coverage.iter_mut().zip(chi_square_quantiles(3)) {
            coverage.push(
                values.iter().filter(|value| **value <= quantile).count() as f64
                    / values.len() as f64,
            );
        }
    }

    pub(super) fn report_and_check(&self) {
        assert_eq!(self.trial_mean_nees.len(), COVARIANCE_TRIALS);
        let (mean_nees, nees_half_width) = clustered_mean_interval(&self.trial_mean_nees);
        println!(
            "BA_POINT_CLUSTER,{},trials={},mean_nees={mean_nees:.6},clustered_95=[{:.6},{:.6}]",
            self.name,
            self.trial_mean_nees.len(),
            mean_nees - nees_half_width,
            mean_nees + nees_half_width,
        );
        assert!(
            (mean_nees - 3.0).abs() <= 0.70,
            "{} mean point NEES {mean_nees}",
            self.name
        );
        for ((nominal, values), tolerance) in NOMINAL_COVERAGE
            .into_iter()
            .zip(&self.trial_coverage)
            .zip([0.12, 0.08, 0.06])
        {
            let (empirical, half_width) = clustered_mean_interval(values);
            println!(
                "BA_POINT_COVERAGE,{},nominal={nominal:.2},empirical={empirical:.6},clustered_95=[{:.6},{:.6}]",
                self.name,
                empirical - half_width,
                empirical + half_width,
            );
            assert!(
                (empirical - nominal).abs() <= tolerance,
                "{} empirical point coverage at {nominal} is {empirical}",
                self.name
            );
        }
    }
}

fn clustered_mean_interval(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (
        mean,
        1.959_963_984_540_054 * (variance / values.len() as f64).sqrt(),
    )
}

pub(super) fn depth_bins(points: &[[f64; 3]]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..points.len()).collect();
    indices.sort_by(|left, right| points[*left][2].total_cmp(&points[*right][2]));
    let mut bins = vec![0; points.len()];
    for (rank, index) in indices.into_iter().enumerate() {
        bins[index] = (3 * rank / points.len()).min(2);
    }
    bins
}

pub(super) fn identity_pose() -> RelativePose {
    RelativePose::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    )
}

pub(super) fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub(super) fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let length = norm(&vector);
    vector.map(|value| value / length)
}

pub(super) fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub(super) fn transpose(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}
