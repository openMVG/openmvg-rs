// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained self-calibration operator policy equivalence tests.

use super::*;

#[test]
fn fused_and_point_major_apply_the_same_grouped_operator() {
    let mut blocks = vec![PackedBlock::default(); 2];
    blocks[0].add(0, 0, 4.0);
    blocks[0].add(1, 0, 0.3);
    blocks[0].add(1, 1, 3.0);
    blocks[1].add(0, 0, 2.5);
    blocks[1].add(1, 0, -0.2);
    blocks[1].add(1, 1, 2.0);
    let mut edges = vec![PoseIntrinsicEdge::new(0, 1)];
    edges[0].add(0, 0, 0.4);
    edges[0].add(0, 1, -0.1);
    edges[0].add(1, 0, 0.2);
    edges[0].add(1, 1, 0.5);
    let block_offsets = [0, 2];
    let block_dimensions = [2, 2];
    let point_offsets = [0, 4, 7];
    let point_parameters = [0, 1, 2, 3, 1, 2, 3];
    let inverse_crosses = [
        [0.2, -0.1, 0.05],
        [0.1, 0.3, -0.2],
        [-0.2, 0.15, 0.1],
        [0.05, -0.25, 0.2],
        [0.3, 0.1, -0.1],
        [-0.1, 0.2, 0.25],
        [0.15, -0.05, 0.3],
    ];
    let point_hessians = [
        PackedPointBlock::from_matrix(&[[2.0, 0.1, -0.2], [0.1, 1.5, 0.05], [-0.2, 0.05, 1.8]]),
        PackedPointBlock::from_matrix(&[[1.2, -0.1, 0.0], [-0.1, 1.7, 0.2], [0.0, 0.2, 1.4]]),
    ];
    let input = [0.7, -0.4, 0.2, 0.9];

    let apply = |policy| {
        let mut operator =
            SelfCalibrationMatvec::try_new(policy, 7, &point_offsets, input.len()).unwrap();
        let mut output = [0.0; 4];
        operator.apply(
            &blocks,
            &edges,
            &block_offsets,
            &block_dimensions,
            &point_hessians,
            &point_offsets,
            &point_parameters,
            &inverse_crosses[..],
            &input,
            &mut output,
        );
        output
    };
    let fused = apply(RetainedSchurMatvec::Fused);
    let point_major = apply(RetainedSchurMatvec::PointMajorParallel { threads: 2 });
    for (expected, actual) in fused.iter().zip(&point_major) {
        assert!(
            (expected - actual).abs() <= 1e-14,
            "{fused:?} {point_major:?}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn auto_point_major_apply_is_bitwise_stable_across_thread_counts() {
    const DIMENSION: usize = 4;
    const POINT_COUNT: usize = 16_384;

    let mut blocks = vec![PackedBlock::default()];
    for parameter in 0..DIMENSION {
        blocks[0].add(parameter, parameter, 2.0 + parameter as f64 * 0.25);
    }
    let point_offsets = (0..=POINT_COUNT)
        .map(|point| point * DIMENSION)
        .collect::<Vec<_>>();
    let point_parameters = (0..POINT_COUNT)
        .flat_map(|_| 0..DIMENSION as u32)
        .collect::<Vec<_>>();
    let inverse_crosses = (0..POINT_COUNT * DIMENSION)
        .map(|row| {
            let point = row / DIMENSION;
            let local = row % DIMENSION;
            [
                1.0e-3 * (1 + point % 29) as f64 + 2.0e-4 * local as f64,
                -7.0e-4 * (1 + point % 17) as f64 + 3.0e-4 * local as f64,
                5.0e-4 * (1 + point % 11) as f64 - 1.0e-4 * local as f64,
            ]
        })
        .collect::<Vec<_>>();
    let point_hessians = (0..POINT_COUNT)
        .map(|point| {
            let perturbation = 1.0e-3 * (point % 13) as f64;
            PackedPointBlock::from_matrix(&[
                [1.1 + perturbation, 0.02, -0.01],
                [0.02, 1.3 + perturbation, 0.03],
                [-0.01, 0.03, 1.5 + perturbation],
            ])
        })
        .collect::<Vec<_>>();
    let input = [0.7, -0.4, 0.2, 0.9];

    let apply = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool is constructible");
        pool.install(|| {
            let mut operator = SelfCalibrationMatvec::try_new(
                RetainedSchurMatvec::Auto,
                inverse_crosses.len(),
                &point_offsets,
                DIMENSION,
            )
            .expect("automatic point-major operator is constructible");
            assert_eq!(operator.parallel_threads(), Some(threads));
            let SelfCalibrationMatvec::PointMajor(workspace) = &operator else {
                panic!("eligible self-calibration Auto must use point-major")
            };
            assert_eq!(workspace.point_ranges.len() - 1, 8);

            let mut output = [0.0; DIMENSION];
            operator.apply(
                &blocks,
                &[],
                &[0],
                &[DIMENSION],
                &point_hessians,
                &point_offsets,
                &point_parameters,
                &inverse_crosses[..],
                &input,
                &mut output,
            );
            output.map(f64::to_bits)
        })
    };

    let expected = apply(1);
    for threads in [2, 4, 8] {
        assert_eq!(apply(threads), expected, "{threads}-thread output drifted");
    }
}
