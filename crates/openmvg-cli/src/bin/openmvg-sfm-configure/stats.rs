// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the project looks like, computed from the model alone: no pipeline run is needed.

use std::collections::{BTreeMap, HashMap};

use openmvg_core::camera::CameraModelType;

use crate::model::{CameraLabel, FOCAL_SPREAD, Model, Row};

/// A single-image camera whose focal length is within this ratio of an established camera (two or
/// more images) of the same make, lens, and size should probably have joined it.
const DUPLICATE_FOCAL: f64 = 1.01;

/// How many warnings the stats panel lists before it elides the rest.
const MAX_WARNINGS: usize = 4;

/// Names at most this many images in a line before eliding the rest.
const NAMED: usize = 5;

/// `a, b, c, d, e, and 3 more`: the first few names, then how many are left.
pub fn name_some(names: &[&str]) -> String {
    let more = names.len().saturating_sub(NAMED);
    let shown = names[..names.len() - more].join(", ");
    if more == 0 {
        shown
    } else {
        format!("{shown}, and {more} more")
    }
}

/// One intrinsic group, as the groups pane and the headless report list it.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupStat {
    pub id: usize,
    pub model: CameraModelType,
    pub focal_px: Option<f64>,
    pub focal_is_guess: bool,
    pub images: usize,
    pub size: (u32, u32),
    pub camera: CameraLabel,
    /// The members name no camera and were put together by size alone.
    pub assumed: bool,
    /// Why the grouping may hurt bundle adjustment; empty when nothing looks wrong.
    pub warnings: Vec<String>,
}

impl GroupStat {
    /// One line, most important first so a narrow pane cuts the camera name, not the warnings:
    /// id (with `⚠` when there are warnings), model, focal length, image count, size, camera.
    pub fn line(&self) -> String {
        let focal = self.focal_px.map_or_else(String::new, |focal| {
            let guess = if self.focal_is_guess { " (guess)" } else { "" };
            format!(" f={focal:.1}{guess}")
        });
        let assumed = if self.assumed { " (assumed)" } else { "" };
        let warning = if self.warnings.is_empty() { "" } else { " ⚠" };
        format!(
            "#{}{warning} {}{focal}  {} img{assumed}  {}x{}  {}",
            self.id,
            self.model.as_str(),
            self.images,
            self.size.0,
            self.size.1,
            self.camera
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub images: usize,
    pub included: usize,
    pub self_calibrate: bool,
    pub groups: Vec<GroupStat>,
    /// The images that could not be read, which no project names.
    pub unreadable: Vec<String>,
    /// Pixel sizes of the included images with their counts, most common first.
    pub sizes: Vec<((u32, u32), usize)>,
    /// EXIF cameras of the included images with their counts, most common first.
    pub cameras: Vec<(String, usize)>,
}

impl Stats {
    pub fn of(model: &Model) -> Self {
        let rows = model.rows();
        let mut sizes = BTreeMap::new();
        let mut cameras = BTreeMap::new();
        let mut included = 0;
        for (row, scanned) in rows.iter().enumerate() {
            if model.group_of(row).is_some() {
                included += 1;
                *sizes.entry(scanned.size()).or_insert(0) += 1;
                let camera = CameraLabel::of([scanned.camera.as_str()]).to_string();
                *cameras.entry(camera).or_insert(0) += 1;
            }
        }
        let mut groups: Vec<GroupStat> = model
            .groups()
            .iter()
            .zip(model.members_by_group())
            .enumerate()
            .map(|(id, (group, members))| {
                let members: Vec<&Row> = members.into_iter().map(|row| &rows[row]).collect();
                let camera = CameraLabel::of(members.iter().map(|row| row.camera.as_str()));
                GroupStat {
                    id,
                    model: group.model,
                    focal_px: group.focal_px,
                    focal_is_guess: group.focal_is_guess,
                    images: members.len(),
                    size: members.first().map_or((0, 0), |row| row.size()),
                    assumed: members.len() > 1 && camera == CameraLabel::Unknown,
                    camera,
                    warnings: member_warnings(&members),
                }
            })
            .collect();
        for (id, duplicate) in duplicates(&groups) {
            groups[id].warnings.push(format!(
                "one image with the camera, size, and focal length of #{duplicate}; its own \
                 intrinsics add unknowns (press g to join it)"
            ));
        }
        Self {
            images: rows.len(),
            included,
            self_calibrate: model.self_calibrate(),
            groups,
            unreadable: model
                .unreadable()
                .iter()
                .map(|image| image.name.clone())
                .collect(),
            sizes: by_count(sizes),
            cameras: by_count(cameras),
        }
    }

    /// Image counts of the groups, largest first, with `×n` when `n` groups have that count:
    /// `185, 40 ×3, 1 ×12`.
    fn images_per_group(&self) -> Vec<String> {
        let mut by_size = BTreeMap::new();
        for group in &self.groups {
            *by_size.entry(group.images).or_insert(0usize) += 1;
        }
        by_size
            .into_iter()
            .rev()
            .map(|(images, groups)| {
                if groups == 1 {
                    images.to_string()
                } else {
                    format!("{images} ×{groups}")
                }
            })
            .collect()
    }

    /// The warning count and the first few warnings, which lead the stats panel so that the
    /// panel's height limit never cuts them off.
    fn warning_lines(&self) -> Vec<String> {
        let warnings: Vec<String> = self
            .groups
            .iter()
            .flat_map(|group| {
                let id = group.id;
                group
                    .warnings
                    .iter()
                    .map(move |warning| format!("#{id} {warning}"))
            })
            .collect();
        if warnings.is_empty() {
            return Vec::new();
        }
        let mut lines = vec![format!("warnings     {}", warnings.len())];
        lines.extend(
            warnings
                .iter()
                .take(MAX_WARNINGS)
                .map(|w| format!("  ⚠ {w}")),
        );
        if warnings.len() > MAX_WARNINGS {
            lines.push("  …".to_owned());
        }
        lines
    }

    /// The summary the stats panel shows.
    pub fn summary_lines(&self) -> Vec<String> {
        let sizes: Vec<_> = self
            .sizes
            .iter()
            .map(|((w, h), count)| format!("{w}x{h} ({count})"))
            .collect();
        let cameras: Vec<_> = self
            .cameras
            .iter()
            .map(|(camera, count)| format!("{camera} ({count})"))
            .collect();
        let mut lines = self.warning_lines();
        if !self.unreadable.is_empty() {
            let names: Vec<&str> = self.unreadable.iter().map(String::as_str).collect();
            lines.push(format!(
                "unreadable   {}, left out: {}",
                names.len(),
                name_some(&names)
            ));
        }
        lines.extend([
            format!(
                "images       {} ({} included, {} excluded)",
                self.images,
                self.included,
                self.images - self.included
            ),
            format!("intrinsics   {}", self.groups.len()),
            format!("per group    {}", self.images_per_group().join(", ")),
            format!("sizes        {}", sizes.join(", ")),
            format!("cameras      {}", cameras.join(", ")),
        ]);
        let guessed: usize = self
            .groups
            .iter()
            .filter(|group| group.focal_is_guess)
            .map(|group| group.images)
            .sum();
        if guessed > 0 {
            lines.push(format!("guessed f    {guessed} (no EXIF focal length)"));
        }
        lines.push(format!(
            "self-calibrate  {}",
            if self.self_calibrate { "yes" } else { "no" }
        ));
        lines
    }
}

/// What is wrong with sharing one camera among `members`.
fn member_warnings(members: &[&Row]) -> Vec<String> {
    let mut warnings = Vec::new();
    if let CameraLabel::Mixed(count) =
        CameraLabel::of(members.iter().map(|row| row.camera.as_str()))
    {
        warnings.push(format!("mixes {count} cameras (press d to split them)"));
    }
    let focals = members.iter().map(|row| row.focal_px);
    let low = focals.clone().fold(f64::INFINITY, f64::min);
    let high = focals.fold(0.0, f64::max);
    if low > 0.0 && high / low > FOCAL_SPREAD {
        warnings.push(format!(
            "scanned focal lengths span {low:.0} to {high:.0} px; a zoom lens at several \
             settings needs one camera per setting (press d)"
        ));
    }
    warnings
}

/// The cameras of two or more images, by EXIF camera and size, with their group and focal length.
type Established<'a> = HashMap<(&'a str, (u32, u32)), Vec<(usize, f64)>>;

/// The single-image cameras that repeat an established camera, as `(camera, established camera)`.
/// Only cameras of two or more images count as established, so one camera per image is not a
/// warning for every image.
fn duplicates(groups: &[GroupStat]) -> Vec<(usize, usize)> {
    let mut established: Established = HashMap::new();
    for group in groups.iter().filter(|group| group.images >= 2) {
        if let (CameraLabel::One(name), Some(focal)) = (&group.camera, group.focal_px) {
            established
                .entry((name.as_str(), group.size))
                .or_default()
                .push((group.id, focal));
        }
    }
    groups
        .iter()
        .filter(|group| group.images == 1)
        .filter_map(|group| {
            let (CameraLabel::One(name), Some(focal)) = (&group.camera, group.focal_px) else {
                return None;
            };
            let same_camera = established.get(&(name.as_str(), group.size))?;
            same_camera
                .iter()
                .find(|(_, other)| focal.max(*other) / focal.min(*other) <= DUPLICATE_FOCAL)
                .map(|&(other, _)| (group.id, other))
        })
        .collect()
}

fn by_count<K: Ord>(counts: BTreeMap<K, usize>) -> Vec<(K, usize)> {
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{focal_row, merged, model, row};

    /// Marks the rows (all are visible, so a row is its own cursor position).
    fn mark(model: &mut Model, rows: &[usize]) {
        model.clear_marks();
        for &row in rows {
            model.cursor_to(row);
            model.toggle_mark();
        }
    }

    #[test]
    fn counts_images_groups_sizes_and_cameras_of_the_included_images() {
        let mut model = model(vec![
            row("a.jpg", 400, 300, "Canon"),
            row("b.jpg", 400, 300, "Canon"),
            row("c.jpg", 400, 300, ""),
            row("d.jpg", 400, 300, ""),
            row("e.jpg", 600, 300, ""),
        ]);
        mark(&mut model, &[0, 1]);
        model.move_to_group(0).unwrap();
        mark(&mut model, &[2, 3]);
        model.move_to_new_group().unwrap();
        mark(&mut model, &[4]);
        model.exclude().unwrap();
        model.clear_marks();

        let stats = Stats::of(&model);

        assert_eq!((stats.images, stats.included), (5, 4));
        let per_group: Vec<_> = stats.groups.iter().map(|g| g.images).collect();
        assert_eq!(per_group, [2, 2]);
        assert_eq!(stats.sizes, [((400, 300), 4)]);
        assert_eq!(
            stats.cameras,
            [("Canon".to_owned(), 2), ("no EXIF camera".to_owned(), 2)]
        );
        assert!(!stats.groups[0].assumed, "its members name a camera");
        assert!(stats.groups[1].assumed, "unknown cameras shared by size");
        let summary = stats.summary_lines().join("\n");
        assert!(summary.contains("5 (4 included, 1 excluded)"), "{summary}");
        assert!(summary.contains("guessed f    4"), "{summary}");
    }

    #[test]
    fn the_images_per_group_are_listed_largest_first_and_repeated_counts_are_folded() {
        let mut model = model(
            (0..6)
                .map(|i| row(&format!("{i}.jpg"), 400, 300, ""))
                .collect(),
        );
        mark(&mut model, &[0, 1, 2]);
        model.move_to_group(0).unwrap();
        assert_eq!(Stats::of(&model).images_per_group(), ["3", "1 ×3"]);
    }

    #[test]
    fn a_group_line_names_its_model_focal_length_size_and_tags() {
        let mut model = model(vec![row("a.jpg", 400, 300, ""), row("b.jpg", 400, 300, "")]);
        model.set_filter("jpg");
        model.move_to_group(0).unwrap();
        let line = Stats::of(&model).groups[0].line();
        assert_eq!(
            line,
            "#0 pinhole_radial1 f=1000.0 (guess)  2 img (assumed)  400x300  no EXIF camera"
        );
        model.set_focal(0, 1500.0).unwrap();
        assert!(!Stats::of(&model).groups[0].line().contains("guess"));
    }

    #[test]
    fn a_camera_that_mixes_cameras_or_zoom_settings_is_flagged_with_the_way_out() {
        let model = merged(vec![
            focal_row("a.jpg", 1000.0, "Canon", false),
            focal_row("b.jpg", 1000.0, "Nikon", false),
            focal_row("c.jpg", 1500.0, "Canon", false),
        ]);

        let stats = Stats::of(&model);

        let warnings = &stats.groups[0].warnings;
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings[0].contains("mixes 2 cameras"), "{warnings:?}");
        assert!(warnings[1].contains("span 1000 to 1500 px"), "{warnings:?}");
        assert!(stats.groups[0].line().starts_with("#0 ⚠ "));
        let summary = stats.summary_lines().join("\n");
        assert!(summary.contains("warnings     2"), "{summary}");
        assert!(summary.contains("⚠ #0 mixes 2 cameras"), "{summary}");
    }

    #[test]
    fn five_percent_of_focal_spread_is_tolerated_and_six_is_not() {
        for (high, flagged) in [(1050.0, false), (1060.0, true)] {
            let model = merged(vec![
                focal_row("a.jpg", 1000.0, "Canon", false),
                focal_row("b.jpg", high, "Canon", false),
            ]);
            assert_eq!(
                Stats::of(&model).groups[0].warnings.len(),
                usize::from(flagged),
                "{high}"
            );
        }
    }

    /// Whether a lone image is warned about repeating an established camera of two 400x300 images
    /// with focal length 1000: `lone` is its camera, size, and focal length.
    fn lone_is_flagged(established_camera: &str, lone: (&str, (u32, u32), f64)) -> bool {
        let scanned = |name: &str, camera: &str, (width, height), focal| {
            Row::new(
                name.to_owned(),
                width,
                height,
                focal,
                false,
                camera.to_owned(),
            )
        };
        let mut model = model(vec![
            scanned("a.jpg", established_camera, (400, 300), 1000.0),
            scanned("b.jpg", established_camera, (400, 300), 1000.0),
            scanned("lone.jpg", lone.0, lone.1, lone.2),
        ]);
        mark(&mut model, &[0, 1]);
        model.move_to_group(0).unwrap();

        let stats = Stats::of(&model);

        assert!(
            stats.groups[0].warnings.is_empty(),
            "the established camera is not at fault"
        );
        !stats.groups[1].warnings.is_empty()
    }

    #[test]
    fn a_lone_image_is_warned_about_only_when_it_repeats_the_camera_size_and_focal_length() {
        let size = (400, 300);
        assert!(lone_is_flagged("Canon", ("Canon", size, 1005.0)));
        assert!(
            lone_is_flagged("Canon", ("Canon", size, 1010.0)),
            "1% is the bound"
        );
        assert!(
            !lone_is_flagged("Canon", ("Canon", size, 1016.0)),
            "another focal length"
        );
        assert!(
            !lone_is_flagged("Canon", ("Nikon", size, 1000.0)),
            "another camera"
        );
        assert!(
            !lone_is_flagged("Canon", ("Canon", (600, 300), 1000.0)),
            "another size"
        );
        assert!(
            !lone_is_flagged("", ("", size, 1000.0)),
            "unknown cameras are not compared"
        );
    }

    #[test]
    fn unreadable_images_are_counted_and_named_with_the_rest_elided() {
        let unreadable = |names: &[&str]| {
            let images = names
                .iter()
                .map(|name| crate::model::Unreadable {
                    name: (*name).to_owned(),
                    reason: "the file is empty".to_owned(),
                })
                .collect();
            let model = model(vec![row("a.jpg", 400, 300, ""), row("b.jpg", 400, 300, "")]);
            Stats::of(&model.with_unreadable(images))
                .summary_lines()
                .join("\n")
        };

        assert!(!unreadable(&[]).contains("unreadable"));
        assert!(unreadable(&["x.jpg", "y.jpg"]).contains("unreadable   2, left out: x.jpg, y.jpg"));
        let many: Vec<String> = (0..8).map(|i| format!("{i}.jpg")).collect();
        let many: Vec<&str> = many.iter().map(String::as_str).collect();
        assert!(
            unreadable(&many).contains(
                "unreadable   8, left out: 0.jpg, 1.jpg, 2.jpg, 3.jpg, 4.jpg, and 3 more"
            )
        );
    }
}
