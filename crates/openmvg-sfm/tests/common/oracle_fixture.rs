// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_sfm::testing::oracle::{OracleSfmFixture, synthetic_oracle_fixture};

pub fn fixture() -> OracleSfmFixture {
    synthetic_oracle_fixture().expect("oracle fixture should build")
}
