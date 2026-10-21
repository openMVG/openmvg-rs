// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `omvg`: command-line tools for OMVG datasets.
//!
//! ```text
//! omvg validate <path>  ──> openmvg_io::check ──> one line per violation, then a verdict
//!                                                  exit 0: conforms   exit 1: violations
//! ```
//!
//! The exit status, the rule identifiers, and the verdict line match `omvg.py check` of the
//! Python reference; violation messages are free text. A usage error exits with status 2.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "\
usage: omvg validate <path>

Check the OMVG dataset at <path> against every rule of the specification, print each
violation with its rule identifier, and exit 0 if the dataset conforms or 1 if it does not.";

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [command, path] if command == "validate" => validate(Path::new(path)),
        [flag] if flag == "-h" || flag == "--help" || flag == "help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        [flag] if flag == "-V" || flag == "--version" => {
            println!("omvg {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn validate(path: &Path) -> ExitCode {
    let violations = openmvg_io::check(path);
    for violation in &violations {
        println!("{violation}");
    }
    if violations.is_empty() {
        println!("conforms");
        ExitCode::SUCCESS
    } else {
        println!("{} violation(s)", violations.len());
        ExitCode::FAILURE
    }
}
