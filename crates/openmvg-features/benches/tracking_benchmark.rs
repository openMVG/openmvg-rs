// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use openmvg_core::model::{FeatureId, ViewId};
use openmvg_features::tracking::{
    ScoredMatch, TrackMatch, TrackObservation, build_tracks, build_tracks_scored,
};

fn observation(view: usize, feature: usize) -> TrackObservation {
    TrackObservation::new(ViewId::new(view), FeatureId::new(feature))
}

fn synthetic_matches(
    track_count: usize,
    track_length: usize,
) -> (Vec<TrackMatch>, Vec<ScoredMatch>) {
    let mut pairs = Vec::with_capacity(track_count * track_length);
    let mut scored = Vec::with_capacity(track_count * track_length);

    for feature in 0..track_count {
        for image in 0..track_length - 1 {
            let pair =
                TrackMatch::new(observation(image, feature), observation(image + 1, feature));
            pairs.push(pair);
            scored.push(ScoredMatch::try_new(pair.first, pair.second, 1.0).expect("finite score"));
        }
    }
    for feature in 1..track_count {
        let pair = TrackMatch::new(
            observation(0, feature),
            observation(track_length - 1, feature - 1),
        );
        pairs.push(pair);
        scored.push(ScoredMatch::try_new(pair.first, pair.second, 0.0).expect("finite score"));
    }

    (pairs, scored)
}

fn benchmark_tracking(c: &mut Criterion) {
    let (pairs, scored) = synthetic_matches(1_000, 10);
    let mut group = c.benchmark_group("tracking_10000_observations");
    group.sample_size(20);
    group.bench_function("connected_components", |bench| {
        bench.iter(|| build_tracks(black_box(&pairs)))
    });
    group.bench_function("scored_conflict_aware", |bench| {
        bench.iter(|| build_tracks_scored(black_box(&scored)))
    });
    group.finish();

    let (_, long_track) = synthetic_matches(1, 10_000);
    let mut long_group = c.benchmark_group("tracking_single_track_10000_observations");
    long_group.sample_size(10);
    long_group.bench_function("scored_conflict_aware", |bench| {
        bench.iter(|| build_tracks_scored(black_box(&long_track)))
    });
    long_group.finish();
}

criterion_group!(benches, benchmark_tracking);
criterion_main!(benches);
