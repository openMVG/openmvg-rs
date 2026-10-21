// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HNSW backend: approximate nearest-neighbor search over a per-image graph index.

use openmvg_features::matching::{DistanceMetric, FilteredMatch};
use openmvg_features::matching::{HnswConfig, HnswMatcher, MatchAcceptance};
use openmvg_features::matching::{SquaredL2Distance, SquaredL2DistanceU8};

use super::{
    descriptor_slices_f32, descriptor_slices_u8, filtered_matches_to_pipeline, prepare_pair_views,
    prepared_view,
};
use crate::backends::PairContext;
use crate::backends::{PairMatcherPreparation, PreparedPairMatcher};
use crate::two_view::PairMatches;
use crate::{Result, SfmError};
use openmvg_features::model::ExtractedFeatures;

const HNSW_INDEX_SEED_DOMAIN: &str = "pipeline/pair-matcher-preparation/v1";

#[derive(Debug, Clone, Copy)]
pub(super) struct HnswOptions {
    pub(super) ef_construction: usize,
    pub(super) ef_search: usize,
    pub(super) ratio_threshold: f32,
    pub(super) acceptance: MatchAcceptance,
}

enum PreparedHnswImage<'a> {
    U8 {
        descriptors: Vec<&'a [u8]>,
        index: Option<HnswMatcher<u8, SquaredL2DistanceU8>>,
    },
    F32 {
        descriptors: Vec<&'a [f32]>,
        index: Option<HnswMatcher<f32, SquaredL2Distance>>,
    },
}

pub(super) struct PreparedHnswMatcher<'a> {
    images: Vec<Option<PreparedHnswImage<'a>>>,
    options: HnswOptions,
}

impl<'a> PreparedHnswMatcher<'a> {
    pub(super) fn new(ctx: PairMatcherPreparation<'a>, options: HnswOptions) -> Result<Self> {
        let root_seed = ctx.root_seed();
        // Image i queries image j's index; only the reciprocal test queries i's.
        let mut indexed = vec![false; ctx.features().slot_count()];
        for pair in ctx.pairs() {
            indexed[pair.second().index()] = true;
            if options.acceptance == MatchAcceptance::Reciprocal {
                indexed[pair.first().index()] = true;
            }
        }

        let images = prepare_pair_views(ctx, |view, view_features| {
            let is_indexed = indexed[view.index()];
            let descriptors = view_features.descriptors();
            if let Some((data, dimension)) = descriptors.as_u8() {
                let descriptors = descriptor_slices_u8(data, dimension)?;
                let index = is_indexed
                    .then(|| {
                        build_hnsw_index::<u8, SquaredL2DistanceU8>(
                            &descriptors,
                            options.ef_construction,
                            options.ef_search,
                            view,
                            root_seed,
                        )
                    })
                    .transpose()?
                    .flatten();
                Ok(PreparedHnswImage::U8 { descriptors, index })
            } else if let Some((data, dimension)) = descriptors.as_f32() {
                let descriptors = descriptor_slices_f32(data, dimension)?;
                let index = is_indexed
                    .then(|| {
                        build_hnsw_index::<f32, SquaredL2Distance>(
                            &descriptors,
                            options.ef_construction,
                            options.ef_search,
                            view,
                            root_seed,
                        )
                    })
                    .transpose()?
                    .flatten();
                Ok(PreparedHnswImage::F32 { descriptors, index })
            } else {
                Err(SfmError::InvalidInput(
                    "HNSW requires matching u8 or f32 descriptor storage".to_owned(),
                ))
            }
        })?;

        Ok(Self { images, options })
    }
}

fn build_hnsw_index<T, Met>(
    descriptors: &[&[T]],
    ef_construction: usize,
    ef_search: usize,
    view: crate::ViewId,
    root_seed: crate::RunSeed,
) -> Result<Option<HnswMatcher<T, Met>>>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    if descriptors.len() < 2 {
        return Ok(None);
    }
    let mut index = HnswMatcher::<T, Met>::with_config(hnsw_config(
        ef_construction,
        ef_search,
        view,
        root_seed,
    ));
    index.build(descriptors).map_err(|err| {
        SfmError::InvalidInput(format!(
            "HNSW build failed for view {}: {err}",
            view.index()
        ))
    })?;
    Ok(Some(index))
}

fn hnsw_config(
    ef_construction: usize,
    ef_search: usize,
    view: crate::ViewId,
    root_seed: crate::RunSeed,
) -> HnswConfig {
    HnswConfig::default()
        .with_ef_construction(ef_construction)
        .with_ef_search(ef_search)
        // Index topology is an implementation detail rather than a sampled model:
        // key it only by view ID so direct/prepared and threaded runs agree.
        .with_seed(root_seed.derive(HNSW_INDEX_SEED_DOMAIN, &[view.index() as u64]))
}

impl PreparedPairMatcher for PreparedHnswMatcher<'_> {
    fn match_pair(&self, ctx: &PairContext) -> Result<PairMatches> {
        let prepared_i = prepared_view(&self.images, ctx.view_i, "HNSW")?;
        let prepared_j = prepared_view(&self.images, ctx.view_j, "HNSW")?;
        let filtered = match (prepared_i, prepared_j) {
            (
                PreparedHnswImage::U8 {
                    descriptors: descriptors_i,
                    index: index_i,
                },
                PreparedHnswImage::U8 {
                    descriptors: descriptors_j,
                    index: index_j,
                },
            ) => match_hnsw_indexes(
                index_i.as_ref(),
                descriptors_i,
                index_j.as_ref(),
                descriptors_j,
                self.options,
            )?,
            (
                PreparedHnswImage::F32 {
                    descriptors: descriptors_i,
                    index: index_i,
                },
                PreparedHnswImage::F32 {
                    descriptors: descriptors_j,
                    index: index_j,
                },
            ) => match_hnsw_indexes(
                index_i.as_ref(),
                descriptors_i,
                index_j.as_ref(),
                descriptors_j,
                self.options,
            )?,
            _ => {
                return Err(SfmError::InvalidInput(
                    "HNSW requires matching u8 or f32 descriptor storage".to_owned(),
                ));
            }
        };
        filtered_matches_to_pipeline(ctx.pair, filtered)
    }
}

/// Match image i against image j's index; the reciprocal test queries i's.
fn match_hnsw_indexes<T, Met>(
    index_i: Option<&HnswMatcher<T, Met>>,
    slices_i: &[&[T]],
    index_j: Option<&HnswMatcher<T, Met>>,
    slices_j: &[&[T]],
    options: HnswOptions,
) -> Result<Vec<FilteredMatch>>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    if !options.acceptance.can_match(slices_i.len(), slices_j.len()) {
        return Ok(Vec::new());
    }
    let missing_index = || SfmError::InvariantViolation("HNSW pair index was not built".to_owned());
    let forward = index_j
        .ok_or_else(missing_index)?
        .query_batch(slices_i, 2)
        .map_err(|err| SfmError::InvalidInput(format!("HNSW query failed: {err}")))?;
    // HNSW queries are independent, so only accepted targets are re-queried.
    options
        .acceptance
        .filter_squared(&forward, options.ratio_threshold, |targets| {
            let queries: Vec<_> = targets.iter().map(|&target| slices_j[target]).collect();
            index_i
                .ok_or_else(missing_index)?
                .query_batch(&queries, 2)
                .map_err(|err| SfmError::InvalidInput(format!("reverse HNSW query failed: {err}")))
        })
}

fn query_hnsw<T, Met>(
    slices_i: &[&[T]],
    slices_j: &[&[T]],
    images: [crate::ViewId; 2],
    root_seed: crate::RunSeed,
    options: HnswOptions,
) -> Result<Vec<FilteredMatch>>
where
    T: Copy + Send + Sync + 'static,
    Met: DistanceMetric<T> + Default + Send + Sync,
{
    if !options.acceptance.can_match(slices_i.len(), slices_j.len()) {
        return Ok(Vec::new());
    }
    let index = |slices, view| {
        build_hnsw_index::<T, Met>(
            slices,
            options.ef_construction,
            options.ef_search,
            view,
            root_seed,
        )
    };
    let index_j = index(slices_j, images[1])?;
    let index_i = match options.acceptance {
        MatchAcceptance::Reciprocal => index(slices_i, images[0])?,
        MatchAcceptance::OneWay => None,
        _ => None,
    };
    match_hnsw_indexes(
        index_i.as_ref(),
        slices_i,
        index_j.as_ref(),
        slices_j,
        options,
    )
}

pub(super) fn match_hnsw_features(
    fi: &ExtractedFeatures,
    fj: &ExtractedFeatures,
    images: [crate::ViewId; 2],
    root_seed: crate::RunSeed,
    options: HnswOptions,
) -> Result<Vec<FilteredMatch>> {
    match (fi.descriptors().as_u8(), fj.descriptors().as_u8()) {
        (Some((data_i, dim_i)), Some((data_j, dim_j))) => query_hnsw::<u8, SquaredL2DistanceU8>(
            &descriptor_slices_u8(data_i, dim_i)?,
            &descriptor_slices_u8(data_j, dim_j)?,
            images,
            root_seed,
            options,
        ),
        _ => match (fi.descriptors().as_f32(), fj.descriptors().as_f32()) {
            (Some((data_i, dim_i)), Some((data_j, dim_j))) => query_hnsw::<f32, SquaredL2Distance>(
                &descriptor_slices_f32(data_i, dim_i)?,
                &descriptor_slices_f32(data_j, dim_j)?,
                images,
                root_seed,
                options,
            ),
            _ => Err(SfmError::InvalidInput(
                "HNSW requires matching u8 or f32 descriptor storage".to_owned(),
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_way_hnsw_accepts_a_single_query_descriptor() {
        let database = [vec![0_u8, 0], vec![10, 10]];
        let query = [vec![1_u8, 0]];
        let database = database.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let query = query.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let matches = query_hnsw::<u8, SquaredL2DistanceU8>(
            &query,
            &database,
            [crate::ViewId::new(0), crate::ViewId::new(1)],
            crate::RunSeed::ZERO,
            HnswOptions {
                ef_construction: 16,
                ef_search: 36,
                ratio_threshold: 0.8,
                acceptance: MatchAcceptance::OneWay,
            },
        )
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(
            (matches[0].query_index(), matches[0].target_index()),
            (0, 0)
        );
    }
}
