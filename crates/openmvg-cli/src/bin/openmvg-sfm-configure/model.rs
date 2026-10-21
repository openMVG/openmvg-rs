// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The editable state: scanned images, their camera groups, the filter, the marks, and undo.
//!
//! Every edit goes through [`Model::edit`], which changes a copy of the grouping and keeps it only
//! when the change succeeds, so a rejected edit leaves the model untouched. The invariants hold
//! after every edit, and for a project the model starts from: images of one group have the same
//! pixel size, a spherical group holds only 2:1 images, and no group is empty.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use openmvg_core::camera::CameraModelType;
use openmvg_sfm::image::{ImageProject, ImageProjectGroup};

/// Scanned focal lengths of one camera that differ by more than this ratio suggest a zoom lens used
/// at several settings, which one focal length cannot describe. Cameras are warned about above it,
/// and a split divides wherever consecutive focal lengths jump by more than it.
pub const FOCAL_SPREAD: f64 = 1.05;

/// The outcome of an edit: what changed, or why it was refused.
pub type Edit = Result<String, String>;

/// The models a group can take, in the order `m` cycles through them.
const MODELS: [CameraModelType; 4] = [
    CameraModelType::PinholeSingle,
    CameraModelType::PinholeRadial1,
    CameraModelType::PinholeRadial2,
    CameraModelType::Spherical,
];

const UNDO_LIMIT: usize = 100;

/// How the images of a group name their camera.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraLabel {
    /// No image names a camera.
    Unknown,
    /// Every image names this camera.
    One(String),
    /// The images name this many different cameras, or some name one and some do not.
    Mixed(usize),
}

impl CameraLabel {
    /// The label of images whose EXIF cameras are `cameras`; an empty name means none is named.
    pub fn of<'a>(cameras: impl IntoIterator<Item = &'a str>) -> Self {
        let distinct: BTreeSet<&str> = cameras.into_iter().collect();
        match (distinct.len(), distinct.first()) {
            (0, _) | (1, Some(&"")) => Self::Unknown,
            (1, Some(&name)) => Self::One(name.to_owned()),
            (count, _) => Self::Mixed(count),
        }
    }
}

impl fmt::Display for CameraLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => formatter.write_str("no EXIF camera"),
            Self::One(name) => formatter.write_str(name),
            Self::Mixed(count) => write!(formatter, "{count} cameras"),
        }
    }
}

/// An image the scan could not read, which every project leaves out.
#[derive(Debug, Clone, PartialEq)]
pub struct Unreadable {
    pub name: String,
    pub reason: String,
}

/// One scanned image.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// The focal length the scan estimated, in pixels.
    pub focal_px: f64,
    /// The scan's focal length is a guess, not read from EXIF.
    pub focal_is_guess: bool,
    /// EXIF camera and lens, empty when the image names none.
    pub camera: String,
    /// Lowercase text the filter looks in.
    search: String,
}

impl Row {
    pub fn new(
        name: String,
        width: u32,
        height: u32,
        focal_px: f64,
        focal_is_guess: bool,
        camera: String,
    ) -> Self {
        let search = format!("{name} {camera} {width}x{height}").to_lowercase();
        Self {
            name,
            width,
            height,
            focal_px,
            focal_is_guess,
            camera,
            search,
        }
    }

    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether the image can be a spherical camera: equirectangular images are 2:1.
    pub const fn is_equirectangular(&self) -> bool {
        self.width == 2 * self.height
    }
}

/// The camera of a group.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub model: CameraModelType,
    /// Focal length in pixels; `None` for a spherical camera.
    pub focal_px: Option<f64>,
    pub focal_is_guess: bool,
}

impl Group {
    /// The camera the scan proposes for an image: a radial pinhole at the scanned focal length.
    pub fn proposed(row: &Row) -> Self {
        Self {
            model: CameraModelType::PinholeRadial1,
            focal_px: Some(row.focal_px),
            focal_is_guess: row.focal_is_guess,
        }
    }
}

/// A proposal to divide one camera into several.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub group: usize,
    /// The rows of each new camera, ascending, ordered by their first row.
    pub clusters: Vec<Vec<usize>>,
}

/// The camera of a cluster of rows: the model, and the median scanned focal length.
fn cluster_camera(rows: &[Row], model: CameraModelType, cluster: &[usize]) -> Group {
    if model == CameraModelType::Spherical {
        return Group {
            model,
            focal_px: None,
            focal_is_guess: false,
        };
    }
    let mut focals: Vec<f64> = cluster.iter().map(|&row| rows[row].focal_px).collect();
    focals.sort_by(f64::total_cmp);
    Group {
        model,
        focal_px: focals.get(focals.len() / 2).copied(),
        focal_is_guess: cluster.iter().all(|&row| rows[row].focal_is_guess),
    }
}

/// The part of the model that edits change and undo restores.
#[derive(Debug, Clone, PartialEq)]
struct Snapshot {
    /// The group of each row, `None` when the image is excluded.
    group_of: Vec<Option<usize>>,
    groups: Vec<Group>,
    self_calibrate: bool,
}

impl Snapshot {
    /// The grouping of `project` over `rows`, which must hold the model's invariants.
    fn from_project(rows: &[Row], project: &ImageProject) -> Result<Self, String> {
        let index: HashMap<&str, usize> = rows
            .iter()
            .enumerate()
            .map(|(row, r)| (r.name.as_str(), row))
            .collect();
        let mut group_of = vec![None; rows.len()];
        let mut groups = Vec::with_capacity(project.groups.len());
        for (id, group) in project.groups.iter().enumerate() {
            let spherical = group.camera_model == CameraModelType::Spherical;
            let mut size = None;
            for name in &group.images {
                let row = *index.get(name.as_str()).ok_or_else(|| {
                    format!("the project names {name}, which is not in the folder")
                })?;
                let scanned = &rows[row];
                if *size.get_or_insert(scanned.size()) != scanned.size() {
                    return Err(format!("group {id} mixes image sizes ({name} differs)"));
                }
                if spherical && !scanned.is_equirectangular() {
                    return Err(format!("group {id} is spherical but {name} is not 2:1"));
                }
                group_of[row] = Some(id);
            }
            groups.push(Group {
                model: group.camera_model,
                focal_px: group.focal_px,
                focal_is_guess: group.focal_is_guess,
            });
        }
        Ok(Self {
            group_of,
            groups,
            self_calibrate: project.self_calibrate,
        })
    }

    fn members(&self, group: usize) -> impl Iterator<Item = usize> + '_ {
        self.group_of
            .iter()
            .enumerate()
            .filter(move |(_, g)| **g == Some(group))
            .map(|(row, _)| row)
    }

    /// The rows of every group, in one pass.
    fn members_by_group(&self) -> Vec<Vec<usize>> {
        let mut members = vec![Vec::new(); self.groups.len()];
        for (row, group) in self.group_of.iter().enumerate() {
            if let Some(group) = group {
                members[*group].push(row);
            }
        }
        members
    }

    /// Drops the groups no image belongs to and renumbers the others.
    fn compact(&mut self) {
        let mut used = vec![false; self.groups.len()];
        for &group in self.group_of.iter().flatten() {
            used[group] = true;
        }
        let mut next = 0;
        let renumbered: Vec<Option<usize>> = used
            .iter()
            .map(|&is_used| {
                is_used.then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let mut kept = renumbered.iter();
        self.groups
            .retain(|_| kept.next().is_some_and(Option::is_some));
        for group in &mut self.group_of {
            *group = group.and_then(|old| renumbered[old]);
        }
    }
}

/// The scanned images and everything the user edits about them.
pub struct Model {
    rows: Vec<Row>,
    state: Snapshot,
    auto: Snapshot,
    saved: Snapshot,
    undo: Vec<Snapshot>,
    filter: String,
    /// Rows the filter keeps, ascending.
    visible: Vec<usize>,
    /// Index into `visible`.
    cursor: usize,
    marked: BTreeSet<usize>,
    unreadable: Vec<Unreadable>,
}

impl Model {
    /// A model of `rows` grouped as `project`; `auto` is the grouping `reset_to_auto` returns to.
    ///
    /// # Errors
    ///
    /// Returns a message when a project names an image that is not among the rows, puts images of
    /// different sizes in one group, or makes a group spherical that holds an image that is not 2:1.
    pub fn new(
        rows: Vec<Row>,
        project: &ImageProject,
        auto: &ImageProject,
    ) -> Result<Self, String> {
        let state = Snapshot::from_project(&rows, project)?;
        let auto = Snapshot::from_project(&rows, auto)?;
        Ok(Self {
            visible: (0..rows.len()).collect(),
            saved: state.clone(),
            state,
            auto,
            undo: Vec::new(),
            filter: String::new(),
            cursor: 0,
            marked: BTreeSet::new(),
            unreadable: Vec::new(),
            rows,
        })
    }

    /// Records the images of the folder that could not be read.
    #[must_use]
    pub fn with_unreadable(mut self, unreadable: Vec<Unreadable>) -> Self {
        self.unreadable = unreadable;
        self
    }

    /// The images of the folder that could not be read, with the reason.
    pub fn unreadable(&self) -> &[Unreadable] {
        &self.unreadable
    }

    // ----- reading -----

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn groups(&self) -> &[Group] {
        &self.state.groups
    }

    pub fn group_of(&self, row: usize) -> Option<usize> {
        self.state.group_of.get(row).copied().flatten()
    }

    /// The rows of every group, indexed by group.
    pub fn members_by_group(&self) -> Vec<Vec<usize>> {
        self.state.members_by_group()
    }

    pub const fn self_calibrate(&self) -> bool {
        self.state.self_calibrate
    }

    pub fn filter(&self) -> &str {
        &self.filter
    }

    pub fn visible(&self) -> &[usize] {
        &self.visible
    }

    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// The row under the cursor, if the filter keeps any.
    pub fn cursor_row(&self) -> Option<usize> {
        self.visible.get(self.cursor).copied()
    }

    /// The group of the row under the cursor.
    pub fn cursor_group(&self) -> Option<usize> {
        self.cursor_row().and_then(|row| self.group_of(row))
    }

    pub fn is_marked(&self, row: usize) -> bool {
        self.marked.contains(&row)
    }

    /// Whether the grouping differs from what was last written.
    pub fn is_modified(&self) -> bool {
        self.state != self.saved
    }

    pub fn mark_saved(&mut self) {
        self.saved = self.state.clone();
    }

    /// The project the model describes: its groups in order, each with its images in row order.
    pub fn project(&self) -> ImageProject {
        let groups = self
            .state
            .groups
            .iter()
            .zip(self.state.members_by_group())
            .map(|(group, members)| {
                let images = members
                    .into_iter()
                    .map(|row| self.rows[row].name.clone())
                    .collect();
                ImageProjectGroup::new(group.model, group.focal_px, images)
                    .with_focal_is_guess(group.focal_is_guess)
            })
            .collect();
        ImageProject::new(self.state.self_calibrate, groups)
    }

    // ----- what an edit acts on -----

    /// The images `g`, `n`, and `x` act on, and how they were chosen: the marked ones, else the
    /// filtered ones when a filter is set, else the one under the cursor.
    fn target_set(&self) -> (&'static str, Vec<usize>) {
        if !self.marked.is_empty() {
            ("marked", self.marked.iter().copied().collect())
        } else if !self.filter.trim().is_empty() {
            ("filtered", self.visible.clone())
        } else {
            ("selected", self.cursor_row().into_iter().collect())
        }
    }

    pub fn targets(&self) -> Vec<usize> {
        self.target_set().1
    }

    /// Names what [`targets`](Self::targets) holds, for the status line.
    pub fn target_label(&self) -> String {
        let (kind, targets) = self.target_set();
        let count = targets.len();
        format!("{count} {kind} image{}", if count == 1 { "" } else { "s" })
    }

    // ----- cursor, marks, filter -----

    pub fn move_cursor(&mut self, delta: isize) {
        self.cursor_to(self.cursor.saturating_add_signed(delta));
    }

    pub fn cursor_to(&mut self, index: usize) {
        self.cursor = index.min(self.visible.len().saturating_sub(1));
    }

    pub fn toggle_mark(&mut self) {
        if let Some(row) = self.cursor_row()
            && !self.marked.remove(&row)
        {
            self.marked.insert(row);
        }
    }

    pub fn mark_visible(&mut self) {
        self.marked.extend(self.visible.iter().copied());
    }

    pub fn clear_marks(&mut self) {
        self.marked.clear();
    }

    /// Keeps the images that contain every term of `filter`, case-insensitively; a term that
    /// starts with `!` must be absent. The terms look in the file name, the camera, and the size.
    pub fn set_filter(&mut self, filter: &str) {
        let before = self.cursor_row();
        self.filter = filter.to_owned();
        let terms: Vec<(bool, String)> = filter
            .split_whitespace()
            .map(str::to_lowercase)
            .map(|term| match term.strip_prefix('!') {
                Some(rest) if !rest.is_empty() => (false, rest.to_owned()),
                _ => (true, term),
            })
            .collect();
        self.visible = (0..self.rows.len())
            .filter(|&row| {
                terms
                    .iter()
                    .all(|(wanted, term)| self.rows[row].search.contains(term.as_str()) == *wanted)
            })
            .collect();
        self.cursor = before
            .and_then(|row| self.visible.binary_search(&row).ok())
            .unwrap_or(self.cursor);
        self.cursor_to(self.cursor);
    }

    // ----- edits -----

    fn edit(&mut self, change: impl FnOnce(&mut Snapshot, &[Row]) -> Edit) -> Edit {
        let mut next = self.state.clone();
        let message = change(&mut next, &self.rows)?;
        next.compact();
        if next != self.state {
            self.undo.push(std::mem::replace(&mut self.state, next));
            if self.undo.len() > UNDO_LIMIT {
                self.undo.remove(0);
            }
        }
        Ok(message)
    }

    /// Moves the targets into an existing group.
    pub fn move_to_group(&mut self, group: usize) -> Edit {
        let targets = self.targets();
        self.edit(|state, rows| {
            if group >= state.groups.len() {
                return Err("no such group".to_owned());
            }
            let sizes: HashSet<_> = targets
                .iter()
                .copied()
                .chain(state.members(group))
                .map(|row| rows[row].size())
                .collect();
            if sizes.len() > 1 {
                return Err("images of different sizes cannot share a camera".to_owned());
            }
            for &row in &targets {
                state.group_of[row] = Some(group);
            }
            Ok(format!(
                "moved {} image(s) to group #{group}",
                targets.len()
            ))
        })
    }

    /// Moves the targets into a new group that starts with the camera of the first target.
    pub fn move_to_new_group(&mut self) -> Edit {
        let targets = self.targets();
        self.edit(|state, rows| {
            let Some(&first) = targets.first() else {
                return Err("no image to move".to_owned());
            };
            if targets
                .iter()
                .any(|&row| rows[row].size() != rows[first].size())
            {
                return Err("images of different sizes cannot share a camera".to_owned());
            }
            let camera = state.group_of[first]
                .map(|group| state.groups[group].clone())
                .unwrap_or_else(|| Group::proposed(&rows[first]));
            state.groups.push(camera);
            let new = state.groups.len() - 1;
            for &row in &targets {
                state.group_of[row] = Some(new);
            }
            Ok(format!("{} image(s) in a new group", targets.len()))
        })
    }

    /// Leaves the targets out of the project.
    pub fn exclude(&mut self) -> Edit {
        let targets = self.targets();
        self.edit(|state, _| {
            for &row in &targets {
                state.group_of[row] = None;
            }
            Ok(format!("excluded {} image(s)", targets.len()))
        })
    }

    /// Gives `group` the next model of [`MODELS`] it can take: a spherical camera needs 2:1
    /// images, and has no focal length, so leaving it proposes the scanned one again.
    pub fn cycle_model(&mut self, group: usize) -> Edit {
        self.edit(|state, rows| {
            let current = state.groups.get(group).ok_or("no such group")?.model;
            let members: Vec<usize> = state.members(group).collect();
            let spherical_ok = members.iter().all(|&row| rows[row].is_equirectangular());
            let start = MODELS.iter().position(|&m| m == current).unwrap_or(0);
            let next = (1..=MODELS.len())
                .map(|step| MODELS[(start + step) % MODELS.len()])
                .find(|&model| model != CameraModelType::Spherical || spherical_ok)
                .unwrap_or(current);
            if next == CameraModelType::Spherical {
                state.groups[group] = cluster_camera(rows, next, &members);
            } else {
                let camera = &mut state.groups[group];
                if camera.focal_px.is_none()
                    && let Some(&first) = members.first()
                {
                    let proposed = Group::proposed(&rows[first]);
                    camera.focal_px = proposed.focal_px;
                    camera.focal_is_guess = proposed.focal_is_guess;
                }
                camera.model = next;
            }
            Ok(format!("group #{group} is {}", next.as_str()))
        })
    }

    /// Sets the focal length of `group`, which makes it a measurement, not a guess.
    pub fn set_focal(&mut self, group: usize, focal_px: f64) -> Edit {
        self.edit(|state, _| {
            let Some(camera) = state.groups.get_mut(group) else {
                return Err("no such group".to_owned());
            };
            if camera.model == CameraModelType::Spherical {
                return Err("a spherical camera has no focal length".to_owned());
            }
            if !focal_px.is_finite() || focal_px <= 0.0 {
                return Err("the focal length must be a positive number of pixels".to_owned());
            }
            camera.focal_px = Some(focal_px);
            camera.focal_is_guess = false;
            Ok(format!("group #{group} focal length is {focal_px} px"))
        })
    }

    /// Proposes dividing `group` by EXIF camera and, within a camera, wherever consecutive scanned
    /// focal lengths differ by more than [`FOCAL_SPREAD`]: images of a zoom lens at one setting
    /// stay together, and a smooth zoom range is not cut. `None` when there is nothing to divide.
    pub fn propose_split(&self, group: usize) -> Option<Split> {
        let spherical = self.state.groups.get(group)?.model == CameraModelType::Spherical;
        let mut by_camera: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for row in self.state.members(group) {
            by_camera
                .entry(self.rows[row].camera.as_str())
                .or_default()
                .push(row);
        }
        let mut clusters = Vec::new();
        for mut rows in by_camera.into_values() {
            if spherical {
                clusters.push(rows);
                continue;
            }
            rows.sort_by(|&a, &b| self.rows[a].focal_px.total_cmp(&self.rows[b].focal_px));
            let mut current: Vec<usize> = Vec::new();
            for row in rows {
                let jumps = current.last().is_some_and(|&last| {
                    self.rows[row].focal_px / self.rows[last].focal_px > FOCAL_SPREAD
                });
                if jumps {
                    clusters.push(std::mem::take(&mut current));
                }
                current.push(row);
            }
            clusters.push(current);
        }
        for cluster in &mut clusters {
            cluster.sort_unstable();
        }
        clusters.sort();
        (clusters.len() > 1).then_some(Split { group, clusters })
    }

    /// One line per new camera of `split`: image count, focal length, and camera.
    pub fn describe_split(&self, split: &Split) -> Vec<String> {
        let model = self.state.groups.get(split.group).map(|group| group.model);
        split
            .clusters
            .iter()
            .map(|cluster| {
                let camera =
                    CameraLabel::of(cluster.iter().map(|&row| self.rows[row].camera.as_str()));
                match model.and_then(|model| cluster_camera(&self.rows, model, cluster).focal_px) {
                    Some(focal) => format!("{} img  f={focal:.1}  {camera}", cluster.len()),
                    None => format!("{} img  {camera}", cluster.len()),
                }
            })
            .collect()
    }

    /// Divides the camera of `split` into its clusters, each with its own median focal length.
    pub fn apply_split(&mut self, split: &Split) -> Edit {
        self.edit(|state, rows| {
            let model = state.groups.get(split.group).map(|group| group.model);
            let current = split
                .clusters
                .iter()
                .flatten()
                .all(|&row| state.group_of[row] == Some(split.group));
            let Some(model) = model.filter(|_| current) else {
                return Err("the camera changed since the split was proposed".to_owned());
            };
            for (index, cluster) in split.clusters.iter().enumerate() {
                let camera = cluster_camera(rows, model, cluster);
                let target = if index == 0 {
                    state.groups[split.group] = camera;
                    split.group
                } else {
                    state.groups.push(camera);
                    state.groups.len() - 1
                };
                for &row in cluster {
                    state.group_of[row] = Some(target);
                }
            }
            Ok(format!(
                "split group #{} into {} cameras",
                split.group,
                split.clusters.len()
            ))
        })
    }

    pub fn toggle_self_calibrate(&mut self) -> Edit {
        self.edit(|state, _| {
            state.self_calibrate = !state.self_calibrate;
            Ok(format!(
                "self-calibration {}",
                if state.self_calibrate { "on" } else { "off" }
            ))
        })
    }

    /// Returns every image to the automatic grouping; the calibration switch stays.
    pub fn reset_to_auto(&mut self) -> Edit {
        let auto = self.auto.clone();
        self.edit(|state, _| {
            state.group_of = auto.group_of;
            state.groups = auto.groups;
            Ok("grouping reset to automatic".to_owned())
        })
    }

    pub fn undo(&mut self) -> Edit {
        let previous = self.undo.pop().ok_or("nothing to undo")?;
        self.state = previous;
        Ok("undone".to_owned())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A 1000 px image with a guessed focal length.
    pub fn row(name: &str, width: u32, height: u32, camera: &str) -> Row {
        Row::new(
            name.to_owned(),
            width,
            height,
            1000.0,
            true,
            camera.to_owned(),
        )
    }

    /// A 400x300 image with a scanned focal length, measured (`guess` false) or guessed.
    pub fn focal_row(name: &str, focal: f64, camera: &str, guess: bool) -> Row {
        Row::new(name.to_owned(), 400, 300, focal, guess, camera.to_owned())
    }

    /// A model with each row in its own radial group.
    pub fn model(rows: Vec<Row>) -> Model {
        let project = ImageProject::new(
            true,
            rows.iter()
                .map(|r| {
                    ImageProjectGroup::new(
                        CameraModelType::PinholeRadial1,
                        Some(r.focal_px),
                        vec![r.name.clone()],
                    )
                    .with_focal_is_guess(true)
                })
                .collect(),
        );
        Model::new(rows, &project, &project).unwrap()
    }

    /// One camera holding every row.
    pub fn merged(rows: Vec<Row>) -> Model {
        let mut model = model(rows);
        model.mark_visible();
        model.move_to_group(0).unwrap();
        model
    }

    fn four() -> Model {
        model(vec![
            row("a-1.jpg", 400, 300, "Canon"),
            row("a-2.jpg", 400, 300, "Canon"),
            row("b-1.jpg", 400, 300, ""),
            row("c-1.jpg", 600, 300, ""),
        ])
    }

    fn groups_of(model: &Model) -> Vec<Option<usize>> {
        (0..model.rows().len())
            .map(|row| model.group_of(row))
            .collect()
    }

    #[test]
    fn the_filter_matches_every_term_in_name_camera_and_size_and_honors_negation() {
        let mut model = four();
        model.set_filter("A- canon");
        assert_eq!(model.visible(), [0, 1]);
        model.set_filter("600x300");
        assert_eq!(model.visible(), [3]);
        model.set_filter("-1 !a-");
        assert_eq!(model.visible(), [2, 3]);
        model.set_filter("!");
        assert!(model.visible().is_empty(), "a lone `!` is a literal term");
        model.set_filter("");
        assert_eq!(model.visible().len(), 4);
    }

    #[test]
    fn the_cursor_stays_on_its_image_when_the_filter_still_keeps_it() {
        let mut model = four();
        model.cursor_to(2);
        model.set_filter("-1");
        assert_eq!(model.cursor_row(), Some(2));
        model.set_filter("a-");
        assert_eq!(
            model.cursor_row(),
            Some(1),
            "clamped to the last kept image"
        );
        model.set_filter("nothing");
        assert_eq!(model.cursor_row(), None);
        assert!(model.targets().is_empty());
    }

    #[test]
    fn edits_act_on_the_marked_images_then_the_filtered_ones_then_the_cursor() {
        let mut model = four();
        model.cursor_to(1);
        assert_eq!(model.targets(), [1]);
        model.set_filter("jpg !a-1 !c-1");
        assert_eq!(model.targets(), [1, 2]);
        model.toggle_mark();
        assert_eq!(model.targets(), [1], "marks win over the filter");
        assert_eq!(model.target_label(), "1 marked image");
    }

    #[test]
    fn images_of_one_size_share_a_group_and_empty_groups_disappear() {
        let mut model = four();
        model.set_filter("a-");
        model.move_to_group(0).unwrap();
        assert_eq!(groups_of(&model), [Some(0), Some(0), Some(1), Some(2)]);
        assert_eq!(model.groups().len(), 3);
        model.set_filter("a-2");
        model.move_to_new_group().unwrap();
        assert_eq!(groups_of(&model), [Some(0), Some(3), Some(1), Some(2)]);
    }

    #[test]
    fn a_rejected_edit_changes_nothing_and_is_not_an_undo_step() {
        let mut model = four();
        model.set_filter("c-1");
        let error = model.move_to_group(0).unwrap_err();
        assert!(error.contains("different sizes"), "{error}");
        model.set_filter("1.jpg");
        assert!(model.move_to_new_group().is_err());
        assert_eq!(groups_of(&model), [Some(0), Some(1), Some(2), Some(3)]);
        assert!(!model.is_modified());
        assert!(model.undo().is_err());
    }

    #[test]
    fn undo_restores_the_grouping_and_modified_follows_the_last_write() {
        let mut model = four();
        model.set_filter("a-");
        model.move_to_group(0).unwrap();
        assert!(model.is_modified());
        model.mark_saved();
        assert!(!model.is_modified());
        model.exclude().unwrap();
        assert_eq!(model.group_of(0), None);
        assert!(model.is_modified());
        model.undo().unwrap();
        assert_eq!(model.group_of(0), Some(0));
        assert!(!model.is_modified(), "back to the saved grouping");
    }

    #[test]
    fn a_set_focal_length_is_a_measurement_and_a_bad_one_changes_nothing() {
        let mut model = four();
        assert!(model.groups()[0].focal_is_guess);
        model.set_focal(0, 1234.5).unwrap();
        assert_eq!(model.groups()[0].focal_px, Some(1234.5));
        assert!(!model.groups()[0].focal_is_guess);
        for bad in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            assert!(model.set_focal(0, bad).is_err(), "{bad}");
            assert_eq!(model.groups()[0].focal_px, Some(1234.5), "after {bad}");
        }
        assert!(model.set_focal(9, 500.0).is_err(), "no such group");
    }

    #[test]
    fn the_model_cycle_skips_spherical_unless_every_image_is_two_to_one() {
        let mut flat = four();
        let mut cycle = || {
            flat.cycle_model(0).unwrap();
            flat.groups()[0].model
        };
        assert_eq!(cycle(), CameraModelType::PinholeRadial2);
        assert_eq!(cycle(), CameraModelType::PinholeSingle);
        assert_eq!(cycle(), CameraModelType::PinholeRadial1);

        let mut sphere = model(vec![
            row("p-1.jpg", 400, 200, ""),
            row("p-2.jpg", 400, 200, ""),
        ]);
        sphere.mark_visible();
        sphere.move_to_group(0).unwrap();
        sphere.cycle_model(0).unwrap();
        sphere.cycle_model(0).unwrap();
        assert_eq!(sphere.groups()[0].model, CameraModelType::Spherical);
        assert_eq!(sphere.groups()[0].focal_px, None);
        assert!(sphere.set_focal(0, 500.0).is_err());
        sphere.cycle_model(0).unwrap();
        assert_eq!(sphere.groups()[0].model, CameraModelType::PinholeSingle);
        assert_eq!(
            sphere.groups()[0].focal_px,
            Some(1000.0),
            "a pinhole group gets the scanned focal length back"
        );
    }

    #[test]
    fn reset_to_auto_returns_the_grouping_but_keeps_the_calibration_switch() {
        let mut model = four();
        model.set_filter("a-");
        model.move_to_group(0).unwrap();
        model.toggle_self_calibrate().unwrap();
        model.reset_to_auto().unwrap();
        assert_eq!(groups_of(&model), [Some(0), Some(1), Some(2), Some(3)]);
        assert!(!model.self_calibrate());
    }

    #[test]
    fn the_project_lists_groups_in_order_with_their_images_and_omits_excluded_ones() {
        let mut model = four();
        model.set_filter("a-");
        model.move_to_group(1).unwrap();
        model.set_filter("c-1");
        model.exclude().unwrap();
        let named: Vec<Vec<String>> = model
            .project()
            .groups
            .into_iter()
            .map(|g| g.images)
            .collect();
        assert_eq!(named, [vec!["a-1.jpg", "a-2.jpg"], vec!["b-1.jpg"]]);
    }

    #[test]
    fn a_project_that_breaks_the_model_invariants_is_refused_when_loaded() {
        let rows = || {
            vec![
                row("a.jpg", 400, 300, ""),
                row("b.jpg", 400, 300, ""),
                row("c.jpg", 600, 300, ""),
            ]
        };
        let group = |model, images: &[&str]| {
            ImageProjectGroup::new(
                model,
                (model != CameraModelType::Spherical).then_some(1.0),
                images.iter().map(|name| (*name).to_owned()).collect(),
            )
        };
        let refuse = |project: ImageProject| {
            Model::new(rows(), &project, &project)
                .err()
                .expect("refused")
        };

        let radial = CameraModelType::PinholeRadial1;
        let missing = refuse(ImageProject::new(
            true,
            vec![group(radial, &["a.jpg", "gone.jpg"])],
        ));
        assert!(missing.contains("gone.jpg"), "{missing}");
        let mixed = refuse(ImageProject::new(
            true,
            vec![group(radial, &["a.jpg", "c.jpg"])],
        ));
        assert!(mixed.contains("mixes image sizes"), "{mixed}");
        let flat_sphere = refuse(ImageProject::new(
            true,
            vec![group(CameraModelType::Spherical, &["a.jpg"])],
        ));
        assert!(flat_sphere.contains("not 2:1"), "{flat_sphere}");
    }

    #[test]
    fn a_zoom_camera_splits_by_camera_and_focal_setting_with_each_median_focal_length() {
        let mut model = merged(vec![
            focal_row("a.jpg", 1000.0, "Canon", false),
            focal_row("b.jpg", 1500.0, "Canon", false),
            focal_row("c.jpg", 1010.0, "Canon", false),
            focal_row("d.jpg", 1000.0, "Nikon", false),
            focal_row("e.jpg", 1520.0, "Canon", false),
            focal_row("f.jpg", 990.0, "Canon", false),
        ]);

        let split = model.propose_split(0).unwrap();

        assert_eq!(split.clusters, [vec![0, 2, 5], vec![1, 4], vec![3]]);
        assert_eq!(
            model.describe_split(&split),
            [
                "3 img  f=1000.0  Canon",
                "2 img  f=1520.0  Canon",
                "1 img  f=1000.0  Nikon"
            ]
        );
        model.apply_split(&split).unwrap();
        let groups: Vec<_> = (0..6).map(|row| model.group_of(row)).collect();
        assert_eq!(
            groups,
            [Some(0), Some(1), Some(0), Some(2), Some(1), Some(0)]
        );
        assert_eq!(model.groups()[1].focal_px, Some(1520.0));
        assert!(!model.groups()[1].focal_is_guess);

        model.undo().unwrap();
        assert_eq!(model.groups().len(), 1, "one undo step restores the camera");
    }

    #[test]
    fn nothing_is_proposed_for_a_uniform_camera_or_a_smooth_zoom_range() {
        let uniform = merged(vec![
            focal_row("a.jpg", 1000.0, "Canon", false),
            focal_row("b.jpg", 1000.0, "Canon", false),
        ]);
        assert!(uniform.propose_split(0).is_none());
        // Each step is 4%, under the 5% gap, so the range is one camera although it spans 12%.
        let smooth = merged(
            [1000.0, 1040.0, 1080.0, 1120.0]
                .iter()
                .enumerate()
                .map(|(i, &focal)| focal_row(&format!("{i}.jpg"), focal, "Canon", false))
                .collect(),
        );
        assert!(smooth.propose_split(0).is_none());
        assert!(smooth.propose_split(7).is_none(), "no such camera");
    }

    #[test]
    fn a_jump_of_more_than_five_percent_starts_a_new_camera() {
        for (high, parts) in [(1050.0, 1), (1051.0, 2)] {
            let model = merged(vec![
                focal_row("a.jpg", 1000.0, "Canon", false),
                focal_row("b.jpg", high, "Canon", false),
            ]);
            let found = model
                .propose_split(0)
                .map_or(1, |split| split.clusters.len());
            assert_eq!(found, parts, "{high}");
        }
    }

    #[test]
    fn a_split_keeps_a_cluster_of_guessed_focal_lengths_a_guess() {
        let mut model = merged(vec![
            focal_row("a.jpg", 1000.0, "Canon", true),
            focal_row("b.jpg", 1000.0, "Nikon", true),
            focal_row("c.jpg", 1000.0, "Nikon", false),
        ]);
        let split = model.propose_split(0).unwrap();
        model.apply_split(&split).unwrap();
        let guesses: Vec<_> = model.groups().iter().map(|g| g.focal_is_guess).collect();
        assert_eq!(
            guesses,
            [true, false],
            "Nikon has an EXIF-focal image, so it is measured"
        );
    }

    #[test]
    fn a_spherical_camera_splits_by_camera_only() {
        let wide = |name: &str, focal: f64, camera: &str| {
            Row::new(name.into(), 400, 200, focal, false, camera.into())
        };
        let mut sphere = model(vec![
            wide("a.jpg", 100.0, "Canon"),
            wide("b.jpg", 900.0, "Canon"),
            wide("c.jpg", 100.0, "Nikon"),
        ]);
        sphere.mark_visible();
        sphere.move_to_group(0).unwrap();
        sphere.cycle_model(0).unwrap();
        sphere.cycle_model(0).unwrap();
        assert_eq!(sphere.groups()[0].model, CameraModelType::Spherical);

        let split = sphere.propose_split(0).unwrap();

        assert_eq!(
            split.clusters,
            [vec![0, 1], vec![2]],
            "focal lengths mean nothing here"
        );
        sphere.apply_split(&split).unwrap();
        assert!(sphere.groups().iter().all(|g| g.focal_px.is_none()));
    }

    #[test]
    fn a_proposal_made_before_the_camera_changed_is_refused_not_a_panic() {
        let build = || {
            let mut model = merged(vec![
                focal_row("a.jpg", 1000.0, "Canon", false),
                focal_row("b.jpg", 1500.0, "Canon", false),
            ]);
            let split = model.propose_split(0).unwrap();
            model.clear_marks();
            (model, split)
        };

        // An image of the proposal left the camera.
        let (mut model, split) = build();
        model.cursor_to(1);
        model.exclude().unwrap();
        let error = model.apply_split(&split).unwrap_err();
        assert!(error.contains("changed since"), "{error}");
        assert_eq!(model.groups().len(), 1);

        // The camera is gone altogether: refusing and describing must not index it.
        let (mut model, split) = build();
        model.mark_visible();
        model.exclude().unwrap();
        assert!(model.groups().is_empty());
        assert!(model.apply_split(&split).is_err());
        let _ = model.describe_split(&split);
    }
}
