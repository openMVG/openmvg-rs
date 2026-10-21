// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Atomic file replacement shared by the checkpoint and PLY writers.
//!
//! The content goes to a uniquely named sibling temporary file, which is synced and then renamed
//! over the destination, so readers see either the previous file or the complete new one.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, ErrorKind};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Replace `path` atomically with the bytes produced by `write`.
///
/// On failure the temporary file is removed and `path` is left untouched.
/// A destination that is a directory fails with [`ErrorKind::IsADirectory`]
/// on every platform (Windows reports the same rename as access-denied).
pub(crate) fn atomic_write_with<E: From<std::io::Error>>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<(), E>,
) -> Result<(), E> {
    let parent = parent_dir(path);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let (temporary_path, file) = loop {
        let nonce = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), nonce));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };

    let result = (|| -> Result<(), E> {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        // The file is closed before the rename, which Windows may otherwise refuse.
        writer
            .into_inner()
            .map_err(|error| error.into_error())?
            .sync_all()?;
        std::fs::rename(&temporary_path, path).map_err(|error| {
            if path.is_dir() {
                std::io::Error::new(ErrorKind::IsADirectory, error)
            } else {
                error
            }
        })?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

/// The directory holding `path`; a bare file name lies in the current directory.
fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_file_name_lies_in_the_current_directory() {
        assert_eq!(parent_dir(Path::new("model.omvgrs")), Path::new("."));
        assert_eq!(parent_dir(Path::new("out/model.omvgrs")), Path::new("out"));
        assert_eq!(parent_dir(Path::new("/model.omvgrs")), Path::new("/"));
    }

    #[test]
    fn renaming_onto_a_directory_reports_is_a_directory() {
        let root = std::env::temp_dir().join(format!(
            "openmvg-sfm-atomic-{}-onto-directory",
            std::process::id()
        ));
        let destination = root.join("scene.omvgrs");
        std::fs::create_dir_all(&destination).unwrap();
        let error = atomic_write_with::<std::io::Error>(&destination, |writer| {
            use std::io::Write as _;
            writer.write_all(b"replacement")
        })
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::IsADirectory);
        std::fs::remove_dir_all(&root).ok();
    }
}
