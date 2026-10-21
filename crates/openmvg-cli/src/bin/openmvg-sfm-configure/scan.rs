// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading the folder and the project file, and writing the project back.
//!
//! The scan is `prepare_image_paths` with images that name no camera shared by size: it reads each
//! image's header and EXIF once, and its grouping is the automatic grouping `a` returns to. The
//! model holds the rules a project must satisfy, so a loaded project is checked when the model is
//! built, and a written one needs no second read of the images.

use std::path::{Path, PathBuf};

use clap::ValueEnum;
use openmvg_sfm::image::{
    ImageDatasetConfig, ImageIntrinsicReport, ImageProject, ImageProjectGroup,
    discover_dataset_images, image_size, prepare_image_paths, prepare_image_project,
};

use crate::model::{Group, Model, Row, Unreadable};
use crate::stats::{Stats, name_some};

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

/// How a new project groups its images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Grouping {
    /// By EXIF camera, lens, and focal length; images with no camera share a group by size.
    Auto,
    /// One camera per image.
    Off,
}

/// The folder being configured and the file the project is written to.
pub struct Session {
    paths: Vec<PathBuf>,
    pub project_file: PathBuf,
    /// The project file did not exist, so the model starts from the scan.
    pub is_new: bool,
}

impl Session {
    /// Scans `folder` and builds the model: from `project_file` when it exists, else from the scan
    /// grouped as `grouping`. Images the project does not name start excluded.
    ///
    /// # Errors
    ///
    /// Returns an error when the folder has fewer than two readable images, or the project file
    /// is unreadable, invalid, names an image the folder lacks or one that cannot be read, or
    /// groups images that cannot share a camera. An image that cannot be read does not stop the
    /// scan: it is left out and listed by [`Model::unreadable`].
    pub fn open(
        folder: &Path,
        project_file: PathBuf,
        grouping: Grouping,
    ) -> DynResult<(Self, Model)> {
        let (paths, unreadable) = split_unreadable(discover(folder)?);
        if paths.len() < 2
            && let Some(first) = unreadable.first()
        {
            return Err(format!(
                "only {} of {} images can be read; the first that cannot is {}: {}",
                paths.len(),
                paths.len() + unreadable.len(),
                first.name,
                first.reason
            )
            .into());
        }
        let Scan { rows, auto } = scan(&paths)?;
        let is_new = !project_file.exists();
        let project = if !is_new {
            let project = ImageProject::load(&project_file)?;
            let named = |name: &str| {
                project
                    .groups
                    .iter()
                    .any(|g| g.images.iter().any(|i| i == name))
            };
            if let Some(image) = unreadable.iter().find(|image| named(&image.name)) {
                return Err(format!(
                    "the project names {}, which cannot be read: {}",
                    image.name, image.reason
                )
                .into());
            }
            project
        } else if grouping == Grouping::Auto {
            auto.clone()
        } else {
            one_camera_per_image(&rows)
        };
        let model = Model::new(rows, &project, &auto)?.with_unreadable(unreadable);
        Ok((
            Self {
                paths,
                project_file,
                is_new,
            },
            model,
        ))
    }

    /// The file of a row of the model.
    pub fn path(&self, row: usize) -> &Path {
        &self.paths[row]
    }

    /// Writes the model's project and returns how many images it names. The model keeps its
    /// groups valid, so only the number of images, which `openmvg-sfm` needs to be two or more, is
    /// checked.
    ///
    /// # Errors
    ///
    /// Returns an error when the project names fewer than two images or the file cannot be
    /// written.
    pub fn write(&self, model: &mut Model) -> DynResult<usize> {
        let project = model.project();
        let images: usize = project.groups.iter().map(|group| group.images.len()).sum();
        if images < 2 {
            return Err("a project needs at least two images".into());
        }
        project.save(&self.project_file)?;
        model.mark_saved();
        Ok(images)
    }

    /// Prepares `project` as `openmvg-sfm --project` does, reading the headers of its images.
    fn prepare(&self, project: &ImageProject) -> DynResult<()> {
        prepare_image_project(&self.paths, project)?;
        Ok(())
    }
}

/// Finds the images of `folder`, refusing a file name a JSON project cannot hold.
fn discover(folder: &Path) -> DynResult<Vec<PathBuf>> {
    let paths = discover_dataset_images(folder)?;
    if let Some(path) = paths
        .iter()
        .find(|path| path.file_name().and_then(|name| name.to_str()).is_none())
    {
        return Err(format!(
            "{} has a file name that is not valid UTF-8, which a project file cannot hold; \
             rename it",
            path.display()
        )
        .into());
    }
    Ok(paths)
}

/// Splits `paths` into the images whose size can be read and the ones that cannot, with why.
fn split_unreadable(paths: Vec<PathBuf>) -> (Vec<PathBuf>, Vec<Unreadable>) {
    let mut readable = Vec::with_capacity(paths.len());
    let mut unreadable = Vec::new();
    for path in paths {
        let Err(error) = image_size(&path) else {
            readable.push(path);
            continue;
        };
        let reason = if std::fs::metadata(&path).is_ok_and(|file| file.len() == 0) {
            "the file is empty".to_owned()
        } else {
            // The message names the file already, which the list does too.
            let message = error.to_string();
            let prefix = format!("invalid input: failed to read {}: ", path.display());
            message
                .strip_prefix(&prefix)
                .unwrap_or(&message)
                .trim()
                .to_owned()
        };
        let name = path.file_name().and_then(|name| name.to_str());
        unreadable.push(Unreadable {
            name: name.unwrap_or_default().to_owned(),
            reason,
        });
    }
    (readable, unreadable)
}

/// One line per image that could not be read, with why.
pub fn unreadable_lines(model: &Model) -> Vec<String> {
    model
        .unreadable()
        .iter()
        .map(|image| format!("cannot read {}: {}", image.name, image.reason))
        .collect()
}

/// What reading the images found.
struct Scan {
    rows: Vec<Row>,
    /// The automatic grouping.
    auto: ImageProject,
}

fn scan(paths: &[PathBuf]) -> DynResult<Scan> {
    let config = ImageDatasetConfig::default().with_share_unknown_cameras(true);
    let prepared = prepare_image_paths(paths, &config)?;
    Ok(Scan {
        rows: prepared.intrinsic_report.iter().map(row).collect(),
        auto: prepared.to_project(true)?,
    })
}

/// Checks a project file against the images of `folder` without changing anything: it must be a
/// valid project, name only images the folder has, and group only images `openmvg-sfm` can share a camera
/// among. A project that fits is reported with its cameras, statistics, and warnings, and with
/// the images it leaves out, which are noted, not counted as problems.
///
/// # Errors
///
/// Returns what is wrong with the project or the folder.
pub fn check(folder: &Path, project_file: &Path) -> Result<Vec<String>, String> {
    if !project_file.is_file() {
        return Err(format!("{} does not exist", project_file.display()));
    }
    let (session, model) = Session::open(folder, project_file.to_path_buf(), Grouping::Auto)
        .map_err(|error| error.to_string())?;
    session
        .prepare(&model.project())
        .map_err(|error| error.to_string())?;

    let stats = Stats::of(&model);
    let mut report: Vec<String> = stats.groups.iter().map(|group| group.line()).collect();
    report.extend(unreadable_lines(&model));
    report.extend(stats.summary_lines());
    let left_out: Vec<&str> = (0..model.rows().len())
        .filter(|&row| model.group_of(row).is_none())
        .map(|row| model.rows()[row].name.as_str())
        .collect();
    if !left_out.is_empty() {
        report.push(format!(
            "{} image(s) in the folder are not in the project, so openmvg-sfm leaves them out: {}",
            left_out.len(),
            name_some(&left_out)
        ));
    }
    Ok(report)
}

fn row(report: &ImageIntrinsicReport) -> Row {
    // Canon writes make "Canon" and model "Canon EOS 5D": name the camera once.
    let camera: Vec<&str> = match (&report.camera_make, &report.camera_model) {
        (Some(make), Some(model)) if model.to_lowercase().starts_with(&make.to_lowercase()) => {
            vec![model.as_str()]
        }
        (make, model) => [make, model]
            .into_iter()
            .flatten()
            .map(String::as_str)
            .collect(),
    };
    let mut camera = camera.join(" ");
    if let Some(lens) = &report.lens_model {
        camera = if camera.is_empty() {
            lens.clone()
        } else {
            format!("{camera}, {lens}")
        };
    }
    Row::new(
        report.image.clone(),
        report.width,
        report.height,
        report.focal_px,
        report.source.is_guess(),
        camera,
    )
}

fn one_camera_per_image(rows: &[Row]) -> ImageProject {
    let groups = rows
        .iter()
        .map(|row| {
            let camera = Group::proposed(row);
            ImageProjectGroup::new(camera.model, camera.focal_px, vec![row.name.clone()])
                .with_focal_is_guess(camera.focal_is_guess)
        })
        .collect();
    ImageProject::new(true, groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmvg_core::camera::CameraModelType;

    /// Blank PNGs `view-<index>.png` of the given sizes; they have no EXIF.
    fn folder(sizes: &[(u32, u32)]) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        for (index, &(width, height)) in sizes.iter().enumerate() {
            image::GrayImage::new(width, height)
                .save(dir.path().join(format!("view-{index}.png")))
                .unwrap();
        }
        dir
    }

    fn open(dir: &tempfile::TempDir, grouping: Grouping) -> DynResult<(Session, Model)> {
        Session::open(dir.path(), dir.path().join("project.json"), grouping)
    }

    #[test]
    fn auto_shares_unknown_cameras_by_size_and_off_gives_every_image_its_own() {
        let dir = folder(&[(64, 48), (64, 48), (32, 32)]);

        let (_, auto) = open(&dir, Grouping::Auto).unwrap();
        let (_, off) = open(&dir, Grouping::Off).unwrap();

        assert_eq!(auto.groups().len(), 2);
        assert_eq!(off.groups().len(), 3);
        assert!(!auto.is_modified() && !off.is_modified());
        // Either way `a` returns to the automatic grouping.
        let mut off = off;
        off.reset_to_auto().unwrap();
        assert_eq!(off.groups().len(), 2);
    }

    #[test]
    fn a_written_project_is_what_sfm_prepares_and_reopening_it_keeps_every_edit() {
        let dir = folder(&[(64, 48), (64, 48), (64, 48), (32, 32)]);
        let (session, mut model) = open(&dir, Grouping::Auto).unwrap();
        assert!(session.is_new);
        model.set_filter("view-2");
        model.exclude().unwrap();
        model.set_filter("view-1");
        model
            .set_focal(model.cursor_group().unwrap(), 777.0)
            .unwrap();
        model.toggle_self_calibrate().unwrap();

        assert_eq!(session.write(&mut model).unwrap(), 3);
        assert!(!model.is_modified());

        let paths = discover_dataset_images(dir.path()).unwrap();
        let project = ImageProject::load(&session.project_file).unwrap();
        let prepared = prepare_image_project(&paths, &project).unwrap();
        assert_eq!(prepared.image_count(), 3);
        assert!(!project.self_calibrate);

        let (session, reopened) = open(&dir, Grouping::Off).unwrap();
        assert!(!session.is_new, "the project wins over --group");
        assert_eq!(reopened.project(), project);
        assert!(
            reopened.group_of(2).is_none(),
            "an image the project leaves out stays excluded"
        );
    }

    #[test]
    fn writing_refuses_a_project_sfm_could_not_prepare() {
        let dir = folder(&[(64, 48), (64, 48)]);
        let (session, mut model) = open(&dir, Grouping::Auto).unwrap();
        model.set_filter("view-0");
        model.exclude().unwrap();

        let error = session.write(&mut model).unwrap_err().to_string();

        assert!(error.contains("at least two"), "{error}");
        assert!(!session.project_file.exists());
        assert!(model.is_modified());
    }

    fn write_project(dir: &tempfile::TempDir, groups: Vec<ImageProjectGroup>) -> PathBuf {
        let file = dir.path().join("project.json");
        ImageProject::new(true, groups).save(&file).unwrap();
        file
    }

    fn group(images: &[&str]) -> ImageProjectGroup {
        ImageProjectGroup::new(
            CameraModelType::PinholeRadial1,
            Some(500.0),
            images.iter().map(|name| (*name).to_owned()).collect(),
        )
    }

    #[test]
    fn a_project_that_fits_is_reported_with_its_cameras_and_the_images_it_leaves_out() {
        let dir = folder(&[(64, 48); 8]);
        let file = write_project(&dir, vec![group(&["view-0.png", "view-1.png"])]);

        let report = check(dir.path(), &file).unwrap().join("\n");

        assert!(
            report.contains("#0 pinhole_radial1 f=500.0  2 img"),
            "{report}"
        );
        assert!(report.contains("intrinsics   1"), "{report}");
        assert!(
            report.contains("6 image(s) in the folder are not in the project")
                && report.contains(
                    "view-2.png, view-3.png, view-4.png, view-5.png, view-6.png, and 1 more"
                ),
            "{report}"
        );
    }

    #[test]
    fn every_way_a_project_can_fail_to_fit_is_a_problem_and_nothing_is_written() {
        let dir = folder(&[(64, 48), (64, 48), (32, 32)]);
        let before = std::fs::read_dir(dir.path()).unwrap().count();
        let problem = |file: &Path| check(dir.path(), file).unwrap_err();

        assert!(problem(&dir.path().join("absent.json")).contains("does not exist"));

        let file = write_project(&dir, vec![group(&["view-0.png", "gone.png"])]);
        let message = problem(&file);
        assert!(
            message.contains("gone.png, which is not in the folder"),
            "{message}"
        );

        write_project(&dir, vec![group(&["view-0.png", "view-2.png"])]);
        assert!(problem(&file).contains("mixes image sizes"));

        // The model accepts one image; `openmvg-sfm` does not, and `--check` asks it.
        write_project(&dir, vec![group(&["view-0.png"])]);
        assert!(problem(&file).contains("at least two"));

        std::fs::write(&file, "{\"schema_version\": 1, \"unknown\": true}").unwrap();
        assert!(problem(&file).contains("unknown field"));

        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            before + 1,
            "only the project file the test wrote"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_file_name_that_is_not_utf8_is_named_in_the_error_not_reported_as_a_camera_problem() {
        use std::os::unix::ffi::OsStrExt;
        let dir = folder(&[(64, 48), (64, 48)]);
        let latin1 = dir.path().join(std::ffi::OsStr::from_bytes(b"caf\xE9.png"));
        std::fs::copy(dir.path().join("view-0.png"), latin1).unwrap();

        let error = open(&dir, Grouping::Auto).err().unwrap().to_string();
        assert!(
            error.contains("not valid UTF-8") && error.contains("caf"),
            "{error}"
        );

        let file = write_project(&dir, vec![group(&["view-0.png", "view-1.png"])]);
        assert!(
            check(dir.path(), &file)
                .unwrap_err()
                .contains("not valid UTF-8")
        );
    }

    /// A folder of three readable images plus an empty file and a file of garbage.
    fn folder_with_unreadable_images() -> tempfile::TempDir {
        let dir = folder(&[(64, 48), (64, 48), (64, 48)]);
        std::fs::write(dir.path().join("broken.png"), b"").unwrap();
        std::fs::write(dir.path().join("junk.png"), b"not an image").unwrap();
        dir
    }

    #[test]
    fn unreadable_images_are_skipped_listed_with_why_and_left_out_of_the_project() {
        let dir = folder_with_unreadable_images();

        let (session, mut model) = open(&dir, Grouping::Auto).unwrap();

        assert_eq!(
            model.rows().len(),
            3,
            "the folder opens with what can be read"
        );
        let listed: Vec<_> = model
            .unreadable()
            .iter()
            .map(|image| (image.name.as_str(), image.reason.as_str()))
            .collect();
        assert_eq!(listed[0], ("broken.png", "the file is empty"));
        assert_eq!(listed[1].0, "junk.png");
        assert!(
            !listed[1].1.is_empty() && !listed[1].1.contains("failed to read"),
            "{listed:?}"
        );
        assert!(
            unreadable_lines(&model)[0].starts_with("cannot read broken.png: the file is empty")
        );

        session.write(&mut model).unwrap();
        let written = ImageProject::load(&session.project_file).unwrap();
        let names: Vec<_> = written.groups.iter().flat_map(|g| &g.images).collect();
        assert_eq!(names.len(), 3);
        assert!(
            !names
                .iter()
                .any(|name| *name == "broken.png" || *name == "junk.png")
        );

        let report = check(dir.path(), &session.project_file).unwrap().join("\n");
        assert!(
            report.contains("unreadable   2, left out: broken.png, junk.png"),
            "{report}"
        );
    }

    #[test]
    fn a_project_that_names_an_unreadable_image_or_a_folder_with_too_few_readable_ones_is_refused()
    {
        let dir = folder_with_unreadable_images();
        let file = write_project(&dir, vec![group(&["view-0.png", "broken.png"])]);
        let error = open(&dir, Grouping::Auto).err().unwrap().to_string();
        assert!(
            error.contains("names broken.png, which cannot be read: the file is empty"),
            "{error}"
        );
        assert!(
            check(dir.path(), &file)
                .unwrap_err()
                .contains("cannot be read")
        );

        let sparse = tempfile::TempDir::new().unwrap();
        image::GrayImage::new(8, 8)
            .save(sparse.path().join("ok.png"))
            .unwrap();
        std::fs::write(sparse.path().join("bad-1.png"), b"").unwrap();
        std::fs::write(sparse.path().join("bad-2.png"), b"").unwrap();
        let error = open(&sparse, Grouping::Auto).err().unwrap().to_string();
        assert!(error.contains("only 1 of 3 images can be read"), "{error}");
        assert!(error.contains("bad-1.png: the file is empty"), "{error}");
    }
}
