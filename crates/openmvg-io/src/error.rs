// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error of the typed reader and writer and of the COLMAP export.

use std::fmt;
use std::path::PathBuf;

use openmvg_sfm::SfmError;

#[cfg(feature = "omvg")]
use crate::Violation;

/// Why a dataset was not read or written, or a reconstruction not exported.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The dataset read, or the dataset that would be written, breaks the rules listed: the
    /// list that [`check`](crate::check) reports, or, when reading with an image root, an image
    /// that resolves outside that root.
    #[cfg(feature = "omvg")]
    Invalid(Vec<Violation>),
    /// The target path exists; datasets are published to a new path.
    Exists(PathBuf),
    /// The dataset conforms, but holds content the SfM types cannot represent, or the SfM value
    /// holds content the format, or COLMAP, cannot; the message names it.
    Unsupported(String),
    /// The SfM types rejected the content read.
    Sfm(SfmError),
    /// A file could not be written. An OMVG writer then publishes nothing at the target;
    /// [`write_colmap`](crate::write_colmap) keeps the files it already replaced.
    Io(std::io::Error),
    /// The data was published at `path` and is complete, but flushing the directory that holds
    /// it failed, so a system crash may still undo the publication. For
    /// [`write_colmap`](crate::write_colmap), `path` is the file just replaced and the files
    /// after it were not written.
    Unsynced {
        /// The published dataset or file.
        path: PathBuf,
        /// Why the directory could not be flushed.
        source: std::io::Error,
    },
    /// A table could not be encoded.
    Encoding(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "omvg")]
            Error::Invalid(violations) => {
                write!(f, "the dataset breaks {} rule(s)", violations.len())?;
                violations
                    .first()
                    .map_or(Ok(()), |first| write!(f, ", first: {first}"))
            }
            Error::Exists(path) => write!(
                f,
                "{} exists; datasets are published to a new path",
                path.display()
            ),
            Error::Unsupported(message) => f.write_str(message),
            Error::Sfm(error) => write!(f, "the SfM types rejected the dataset: {error}"),
            Error::Io(error) => write!(f, "cannot write a file: {error}"),
            Error::Unsynced { path, source } => write!(
                f,
                "{} was published, but a crash may undo it: cannot flush its directory: {source}",
                path.display()
            ),
            Error::Encoding(error) => write!(f, "cannot encode a table: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Sfm(error) => Some(error),
            Error::Io(error) | Error::Unsynced { source: error, .. } => Some(error),
            Error::Encoding(error) => Some(error.as_ref()),
            #[cfg(feature = "omvg")]
            Error::Invalid(_) => None,
            Error::Exists(_) | Error::Unsupported(_) => None,
        }
    }
}

impl From<SfmError> for Error {
    fn from(error: SfmError) -> Self {
        Error::Sfm(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

#[cfg(feature = "omvg")]
impl From<parquet::errors::ParquetError> for Error {
    fn from(error: parquet::errors::ParquetError) -> Self {
        Error::Encoding(Box::new(error))
    }
}

#[cfg(feature = "omvg")]
impl From<arrow_schema::ArrowError> for Error {
    fn from(error: arrow_schema::ArrowError) -> Self {
        Error::Encoding(Box::new(error))
    }
}

/// The result of the typed reader and writer and of the COLMAP export.
pub type Result<T> = std::result::Result<T, Error>;
