// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn fast_median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let middle = values.len() / 2;
    values.select_nth_unstable_by(middle, f64::total_cmp);
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        let lower = values[..middle]
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .unwrap_or(values[middle]);
        (lower + values[middle]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::fast_median;

    #[test]
    fn median_covers_odd_even_and_empty_inputs() {
        let mut odd = [5.0, 1.0, 3.0, 2.0, 4.0];
        let mut even = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(fast_median(&mut odd), 3.0);
        assert_eq!(fast_median(&mut even), 2.5);
        assert_eq!(fast_median(&mut []), 0.0);
    }
}
