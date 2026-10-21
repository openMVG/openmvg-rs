// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared construction for persistent sparse-worker pools.

#[cfg(feature = "parallel")]
pub(super) fn try_build_pool(
    threads: usize,
    name: &'static str,
) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |worker| format!("{name}-{worker}"))
        .build()?;
    pool.broadcast(|_| {});
    Ok(pool)
}
