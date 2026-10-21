// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_ba::kernel::RelativePose;

pub(super) const COVARIANCE_TRIALS: usize = 200;
pub(super) const NOMINAL_COVERAGE: [f64; 3] = [0.50, 0.90, 0.95];

pub(super) struct IndependentCoverage {
    name: &'static str,
    dimension: usize,
    values: Vec<f64>,
}

impl IndependentCoverage {
    pub(super) fn new(name: &'static str, dimension: usize) -> Self {
        Self {
            name,
            dimension,
            values: Vec::with_capacity(COVARIANCE_TRIALS),
        }
    }

    pub(super) fn push(&mut self, nees: f64) {
        assert!(nees.is_finite() && nees >= 0.0, "{} NEES={nees}", self.name);
        self.values.push(nees);
    }

    pub(super) fn report_and_check(&self) {
        assert_eq!(self.values.len(), COVARIANCE_TRIALS);
        let total: f64 = self.values.iter().sum();
        let total_dof = self.dimension * self.values.len();
        let (lower, upper) = aggregate_nees_interval(total_dof);
        let quantiles = chi_square_quantiles(self.dimension);
        println!(
            "COVERAGE,{},trials={},mean_nees={:.6},max_nees={:.6},aggregate_95=[{lower:.3},{upper:.3}]",
            self.name,
            self.values.len(),
            total / self.values.len() as f64,
            self.values.iter().copied().fold(0.0, f64::max),
        );
        for ((nominal, quantile), tolerance) in NOMINAL_COVERAGE
            .into_iter()
            .zip(quantiles)
            .zip([0.10, 0.065, 0.05])
        {
            let empirical = self
                .values
                .iter()
                .filter(|value| **value <= quantile)
                .count() as f64
                / self.values.len() as f64;
            println!(
                "COVERAGE_LEVEL,{},nominal={nominal:.2},empirical={empirical:.6},chi2={quantile:.6}",
                self.name
            );
            assert!(
                (empirical - nominal).abs() <= tolerance,
                "{} empirical {:.0}% coverage is {empirical:.3}",
                self.name,
                100.0 * nominal
            );
        }
        assert!(
            (lower..=upper).contains(&total),
            "{} aggregate NEES {total} outside [{lower}, {upper}]",
            self.name
        );
    }
}

fn aggregate_nees_interval(degrees_of_freedom: usize) -> (f64, f64) {
    let dof = degrees_of_freedom as f64;
    let center = 1.0 - 2.0 / (9.0 * dof);
    let radius = 1.959_963_984_540_054 * (2.0 / (9.0 * dof)).sqrt();
    (
        dof * (center - radius).powi(3),
        dof * (center + radius).powi(3),
    )
}

pub(super) fn chi_square_quantiles(dimension: usize) -> [f64; 3] {
    match dimension {
        3 => [2.365_973_884, 6.251_388_631, 7.814_727_903],
        5 => [4.351_460_191, 9.236_356_900, 11.070_497_69],
        7 => [6.345_811_196, 12.017_036_62, 14.067_140_45],
        8 => [7.344_121_498, 13.361_566_14, 15.507_313_06],
        11 => [10.340_998_07, 17.275_008_52, 19.675_137_57],
        20 => [19.337_429_23, 28.411_980_58, 31.410_432_84],
        _ => panic!("missing audited chi-square quantiles for dimension {dimension}"),
    }
}

pub(super) fn transform(pose: &RelativePose, point: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| pose.rotation[row][column] * point[column])
            .sum::<f64>()
            + pose.translation[row]
    })
}
