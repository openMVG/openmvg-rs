// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Turns key presses and mouse events into edits of the model and changes of the interface mode.
//!
//! Nothing here draws or touches the filesystem: the event loop acts on the returned [`Effect`].

use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Position, Rect, Size};

use crate::model::{Edit, Model, Split};
use crate::preview::Preview;

/// Rows a page key and a mouse wheel notch move.
const PAGE: isize = 10;
const WHEEL: isize = 3;

/// What the interface is waiting for.
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Browse,
    /// Typing the filter; the filter applies as it is typed.
    Filter,
    /// Choosing the group the targets move to.
    PickGroup {
        cursor: usize,
    },
    /// Typing the focal length of a group.
    EditFocal {
        group: usize,
        text: String,
        /// What the field started with: confirming it unchanged edits nothing, so a guessed focal
        /// length is not quietly promoted to a measured one.
        initial: String,
    },
    /// Asking what to do with unsaved changes before quitting.
    ConfirmQuit,
    /// Showing how a camera would be divided until it is accepted.
    ConfirmSplit {
        split: Split,
    },
}

/// What the event loop does after an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    None,
    Write,
    WriteAndQuit,
    Quit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Notice {
    pub text: String,
    pub is_error: bool,
}

/// Where the image list was last drawn, for mouse hit-testing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListView {
    /// The area of the rows, inside the border.
    pub inner: Rect,
    /// Index into the visible rows of the first one drawn.
    pub offset: usize,
}

pub struct App {
    pub model: Model,
    pub mode: Mode,
    pub notice: Option<Notice>,
    pub list: ListView,
    /// Where `w` writes, for the status line.
    pub project_file: String,
    /// The project file does not exist yet. Quitting asks first even before any edit, so a
    /// session that only looked does not leave `openmvg-sfm --project` without a file.
    pub unwritten: bool,
    /// The image preview; `None` when it is switched off.
    pub preview: Option<Preview>,
    /// The size of the preview pane in cells, as last drawn.
    pub preview_area: Size,
}

impl App {
    pub fn new(model: Model, project_file: String) -> Self {
        Self {
            model,
            mode: Mode::Browse,
            notice: None,
            list: ListView::default(),
            project_file,
            unwritten: false,
            preview: None,
            preview_area: Size::ZERO,
        }
    }

    pub fn info(&mut self, text: String) {
        self.notice = Some(Notice {
            text,
            is_error: false,
        });
    }

    pub fn error(&mut self, text: String) {
        self.notice = Some(Notice {
            text,
            is_error: true,
        });
    }

    /// Runs an edit of the model and shows what it did, or why it was refused.
    fn apply(&mut self, edit: impl FnOnce(&mut Model) -> Edit) {
        match edit(&mut self.model) {
            Ok(text) => self.info(text),
            Err(text) => self.error(text),
        }
    }

    /// The group of the row under the cursor, or an error notice explaining there is none.
    fn cursor_group(&mut self) -> Option<usize> {
        let group = self.model.cursor_group();
        if group.is_none() {
            self.error("the image under the cursor is excluded; move it to a group first".into());
        }
        group
    }

    fn request_quit(&mut self) -> Effect {
        if self.model.is_modified() || self.unwritten {
            self.mode = Mode::ConfirmQuit;
            Effect::None
        } else {
            Effect::Quit
        }
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> Effect {
    app.notice = None;
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return app.request_quit();
    }
    match app.mode {
        Mode::Browse => return browse(app, key),
        Mode::Filter => filter(app, key),
        Mode::PickGroup { .. } => pick_group(app, key),
        Mode::EditFocal { .. } => edit_focal(app, key),
        Mode::ConfirmSplit { .. } => confirm_split(app, key),
        Mode::ConfirmQuit => {
            app.mode = Mode::Browse;
            return match key.code {
                KeyCode::Char('w') => Effect::WriteAndQuit,
                KeyCode::Char('y') => Effect::Quit,
                _ => Effect::None,
            };
        }
    }
    Effect::None
}

fn browse(app: &mut App, key: KeyEvent) -> Effect {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.model.move_cursor(-1),
        KeyCode::Down | KeyCode::Char('j') => app.model.move_cursor(1),
        KeyCode::PageUp => app.model.move_cursor(-PAGE),
        KeyCode::PageDown => app.model.move_cursor(PAGE),
        KeyCode::Home => app.model.cursor_to(0),
        KeyCode::End => app.model.cursor_to(usize::MAX),
        KeyCode::Char('/') => app.mode = Mode::Filter,
        KeyCode::Char(' ') => {
            app.model.toggle_mark();
            app.model.move_cursor(1);
        }
        KeyCode::Char('*') => app.model.mark_visible(),
        KeyCode::Char('c') => app.model.clear_marks(),
        KeyCode::Esc => {
            if app.model.filter().is_empty() {
                app.model.clear_marks();
            } else {
                app.model.set_filter("");
            }
        }
        KeyCode::Char('g') => {
            if app.model.groups().is_empty() {
                app.error("there is no group to move to; press n for a new one".into());
            } else {
                let cursor = app.model.cursor_group().unwrap_or(0);
                app.mode = Mode::PickGroup { cursor };
            }
        }
        KeyCode::Char('n') => app.apply(Model::move_to_new_group),
        KeyCode::Char('x') => app.apply(Model::exclude),
        KeyCode::Char('m') => {
            if let Some(group) = app.cursor_group() {
                app.apply(|model| model.cycle_model(group));
            }
        }
        KeyCode::Char('f') => {
            if let Some(group) = app.cursor_group() {
                match app.model.groups()[group].focal_px {
                    Some(focal) => {
                        let text = format!("{focal:.1}");
                        app.mode = Mode::EditFocal {
                            group,
                            initial: text.clone(),
                            text,
                        };
                    }
                    None => app.error("a spherical camera has no focal length".into()),
                }
            }
        }
        KeyCode::Char('d') => {
            if let Some(group) = app.cursor_group() {
                match app.model.propose_split(group) {
                    Some(split) => app.mode = Mode::ConfirmSplit { split },
                    None => app.info(format!(
                        "nothing to split in group #{group}: its images agree on camera and focal length"
                    )),
                }
            }
        }
        KeyCode::Char('s') => app.apply(Model::toggle_self_calibrate),
        KeyCode::Char('a') => app.apply(Model::reset_to_auto),
        KeyCode::Char('u') => app.apply(Model::undo),
        KeyCode::Char('w') => return Effect::Write,
        KeyCode::Char('q') => return app.request_quit(),
        _ => {}
    }
    Effect::None
}

/// Whether a character key types text: Ctrl and Alt combinations are commands, never text.
fn is_text(key: KeyEvent) -> bool {
    !key.modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
}

fn filter(app: &mut App, key: KeyEvent) {
    let mut text = app.model.filter().to_owned();
    match key.code {
        KeyCode::Enter => {
            app.mode = Mode::Browse;
            return;
        }
        KeyCode::Esc => {
            app.mode = Mode::Browse;
            text.clear();
        }
        KeyCode::Backspace => {
            text.pop();
        }
        KeyCode::Char(c) if is_text(key) => text.push(c),
        _ => {}
    }
    app.model.set_filter(&text);
}

fn pick_group(app: &mut App, key: KeyEvent) {
    let last = app.model.groups().len().saturating_sub(1);
    let Mode::PickGroup { cursor } = &mut app.mode else {
        return;
    };
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => *cursor = cursor.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => *cursor = (*cursor + 1).min(last),
        KeyCode::Enter => {
            let group = *cursor;
            app.mode = Mode::Browse;
            app.apply(|model| model.move_to_group(group));
        }
        KeyCode::Esc => app.mode = Mode::Browse,
        _ => {}
    }
}

fn edit_focal(app: &mut App, key: KeyEvent) {
    let Mode::EditFocal {
        group,
        text,
        initial,
    } = &mut app.mode
    else {
        return;
    };
    match key.code {
        KeyCode::Enter => {
            let (group, unchanged, text) = (*group, text == initial, text.clone());
            app.mode = Mode::Browse;
            if unchanged {
                app.info("focal length unchanged".into());
            } else if let Ok(focal) = text.trim().parse::<f64>() {
                app.apply(|model| model.set_focal(group, focal));
            } else {
                app.error(format!("{text:?} is not a number"));
            }
        }
        KeyCode::Esc => app.mode = Mode::Browse,
        KeyCode::Backspace => {
            text.pop();
        }
        KeyCode::Char(c) if is_text(key) && (c.is_ascii_digit() || c == '.') => text.push(c),
        _ => {}
    }
}

fn confirm_split(app: &mut App, key: KeyEvent) {
    let Mode::ConfirmSplit { split } = std::mem::replace(&mut app.mode, Mode::Browse) else {
        return;
    };
    if matches!(key.code, KeyCode::Enter | KeyCode::Char('y')) {
        app.apply(|model| model.apply_split(&split));
    }
}

pub fn handle_mouse(app: &mut App, mouse: MouseEvent) {
    if app.mode != Mode::Browse {
        return;
    }
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            let inner = app.list.inner;
            if inner.contains(Position::new(mouse.column, mouse.row)) {
                let row = app.list.offset + usize::from(mouse.row - inner.y);
                if row < app.model.visible().len() {
                    app.model.cursor_to(row);
                }
            }
        }
        MouseEventKind::ScrollUp => app.model.move_cursor(-WHEEL),
        MouseEventKind::ScrollDown => app.model.move_cursor(WHEEL),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{model, row};
    use ratatui::crossterm::event::KeyEventKind;

    fn app() -> App {
        App::new(
            model(vec![
                row("dji-1.jpg", 400, 300, ""),
                row("dji-2.jpg", 400, 300, ""),
                row("cam-1.jpg", 400, 300, "Canon"),
                row("wide.jpg", 600, 300, ""),
            ]),
            "project.json".to_owned(),
        )
    }

    fn press(app: &mut App, keys: &str) -> Effect {
        let mut effect = Effect::None;
        for c in keys.chars() {
            let code = if c == '\n' {
                KeyCode::Enter
            } else {
                KeyCode::Char(c)
            };
            effect = handle_key(
                app,
                KeyEvent::new_with_kind(code, KeyModifiers::NONE, KeyEventKind::Press),
            );
        }
        effect
    }

    fn named_groups(app: &App) -> Vec<Vec<String>> {
        app.model
            .project()
            .groups
            .into_iter()
            .map(|g| g.images)
            .collect()
    }

    #[test]
    fn filtering_then_naming_a_new_group_moves_every_match() {
        let mut app = app();
        press(&mut app, "/dji\n");
        assert_eq!(app.mode, Mode::Browse);
        assert_eq!(app.model.target_label(), "2 filtered images");

        press(&mut app, "n");

        assert_eq!(named_groups(&app)[2], ["dji-1.jpg", "dji-2.jpg"]);
        assert_eq!(app.notice.as_ref().map(|n| n.is_error), Some(false));
    }

    #[test]
    fn control_and_alt_keys_are_not_typed_into_the_filter_or_the_focal_length() {
        let mut app = app();
        press(&mut app, "/dj");
        for modifier in [KeyModifiers::CONTROL, KeyModifiers::ALT] {
            handle_key(&mut app, KeyEvent::new(KeyCode::Char('j'), modifier));
        }
        assert_eq!((app.mode.clone(), app.model.filter()), (Mode::Filter, "dj"));
        handle_key(&mut app, KeyEvent::from(KeyCode::Enter));

        press(&mut app, "f");
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('7'), KeyModifiers::CONTROL),
        );
        assert!(matches!(&app.mode, Mode::EditFocal { text, .. } if text == "1000.0"));
    }

    #[test]
    fn escape_leaves_the_filter_mode_and_clears_the_filter_then_the_marks() {
        let mut app = app();
        press(&mut app, "/dji");
        assert_eq!(app.mode, Mode::Filter);
        handle_key(&mut app, KeyEvent::from(KeyCode::Esc));
        assert_eq!((app.mode.clone(), app.model.filter()), (Mode::Browse, ""));

        press(&mut app, " ");
        assert!(app.model.is_marked(0));
        handle_key(&mut app, KeyEvent::from(KeyCode::Esc));
        assert!(!app.model.is_marked(0));
    }

    #[test]
    fn picking_a_group_moves_the_marked_images_into_it() {
        let mut app = app();
        press(&mut app, " ");
        press(&mut app, "g");
        assert_eq!(
            app.mode,
            Mode::PickGroup { cursor: 1 },
            "starts on the cursor's group"
        );
        press(&mut app, "kj");
        assert_eq!(app.mode, Mode::PickGroup { cursor: 1 });
        press(&mut app, "\n");
        assert_eq!(app.mode, Mode::Browse);
        assert_eq!(named_groups(&app)[0], ["dji-1.jpg", "dji-2.jpg"]);
    }

    #[test]
    fn a_size_conflict_is_an_error_notice_and_leaves_the_grouping_alone() {
        let mut app = app();
        press(&mut app, "/wide\n");
        press(&mut app, "gkkk\n");
        let notice = app.notice.clone().unwrap();
        assert!(
            notice.is_error && notice.text.contains("different sizes"),
            "{notice:?}"
        );
        assert!(!app.model.is_modified());
    }

    #[test]
    fn the_focal_length_is_typed_in_and_validated() {
        let mut app = app();
        press(&mut app, "f");
        assert!(matches!(app.mode, Mode::EditFocal { group: 0, .. }));
        for _ in 0..6 {
            handle_key(&mut app, KeyEvent::from(KeyCode::Backspace));
        }
        press(&mut app, "812.5\n");
        assert_eq!(app.model.groups()[0].focal_px, Some(812.5));

        press(&mut app, "f");
        press(&mut app, "..\n");
        assert!(app.notice.as_ref().is_some_and(|n| n.is_error));
        assert_eq!(app.model.groups()[0].focal_px, Some(812.5));
    }

    #[test]
    fn editing_an_excluded_image_asks_to_put_it_in_a_group_first() {
        let mut app = app();
        press(&mut app, "x");
        press(&mut app, "m");
        assert!(
            app.notice
                .as_ref()
                .is_some_and(|n| n.is_error && n.text.contains("excluded"))
        );
        press(&mut app, "f");
        assert_eq!(app.mode, Mode::Browse);
    }

    #[test]
    fn quitting_asks_only_when_something_is_unsaved() {
        let mut app = app();
        assert_eq!(press(&mut app, "q"), Effect::Quit);

        press(&mut app, "x");
        assert_eq!(press(&mut app, "q"), Effect::None);
        assert_eq!(app.mode, Mode::ConfirmQuit);
        assert_eq!(press(&mut app, "z"), Effect::None, "any other key cancels");
        assert_eq!(app.mode, Mode::Browse);

        press(&mut app, "q");
        assert_eq!(press(&mut app, "w"), Effect::WriteAndQuit);
        press(&mut app, "q");
        assert_eq!(press(&mut app, "y"), Effect::Quit);
        assert_eq!(press(&mut app, "w"), Effect::Write);
    }

    #[test]
    fn a_click_selects_the_row_it_hits_and_the_wheel_scrolls() {
        let mut app = app();
        app.list = ListView {
            inner: Rect::new(1, 2, 40, 10),
            offset: 1,
        };
        let click = |column, row| MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row,
            modifiers: KeyModifiers::NONE,
        };
        handle_mouse(&mut app, click(5, 4));
        assert_eq!(app.model.cursor(), 3, "offset 1 plus two rows down");
        handle_mouse(&mut app, click(0, 4));
        handle_mouse(&mut app, click(5, 9));
        assert_eq!(
            app.model.cursor(),
            3,
            "outside the rows or past the last image is ignored"
        );
        let wheel = |kind| MouseEvent {
            kind,
            column: 5,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        handle_mouse(&mut app, wheel(MouseEventKind::ScrollUp));
        assert_eq!(app.model.cursor(), 0);
        handle_mouse(&mut app, wheel(MouseEventKind::ScrollDown));
        assert_eq!(app.model.cursor(), 3);
    }

    #[test]
    fn d_proposes_a_split_that_enter_applies_and_anything_else_cancels() {
        let mut app = App::new(
            model(vec![
                crate::model::tests::row("a.jpg", 400, 300, "Canon"),
                crate::model::tests::row("b.jpg", 400, 300, "Nikon"),
                crate::model::tests::row("c.jpg", 400, 300, "Canon"),
            ]),
            "project.json".to_owned(),
        );
        press(&mut app, "d");
        assert!(
            app.notice
                .as_ref()
                .is_some_and(|n| !n.is_error && n.text.contains("nothing to split")),
            "{:?}",
            app.notice
        );

        press(&mut app, "*g");
        press(&mut app, "kk\n");
        assert_eq!(app.model.groups().len(), 1);
        press(&mut app, "d");
        let Mode::ConfirmSplit { split } = app.mode.clone() else {
            panic!("expected a split proposal, got {:?}", app.mode);
        };
        assert_eq!(split.clusters, [vec![0, 2], vec![1]]);
        assert_eq!(app.model.describe_split(&split).len(), 2);

        press(&mut app, "n");
        assert_eq!(
            (app.mode.clone(), app.model.groups().len()),
            (Mode::Browse, 1)
        );
        press(&mut app, "d");
        press(&mut app, "\n");
        assert_eq!(
            (app.mode.clone(), app.model.groups().len()),
            (Mode::Browse, 2)
        );
        assert!(
            app.notice
                .as_ref()
                .is_some_and(|n| n.text.contains("split group #0"))
        );
        press(&mut app, "u");
        assert_eq!(app.model.groups().len(), 1, "undo restores the camera");
    }

    #[test]
    fn confirming_the_focal_length_unchanged_keeps_a_guess_a_guess() {
        let mut app = app();
        assert!(app.model.groups()[0].focal_is_guess);

        press(&mut app, "f\n");

        assert_eq!(app.mode, Mode::Browse);
        assert!(app.model.groups()[0].focal_is_guess, "still a guess");
        assert!(!app.model.is_modified());
        assert!(
            app.notice
                .as_ref()
                .is_some_and(|n| n.text.contains("unchanged"))
        );

        // Typing the same number back is the same as not editing it.
        press(&mut app, "f");
        for _ in 0..6 {
            handle_key(&mut app, KeyEvent::from(KeyCode::Backspace));
        }
        press(&mut app, "1000.0\n");
        assert!(app.model.groups()[0].focal_is_guess);

        // A different number is a measurement.
        press(&mut app, "f");
        handle_key(&mut app, KeyEvent::from(KeyCode::Backspace));
        press(&mut app, "5\n");
        assert_eq!(app.model.groups()[0].focal_px, Some(1000.5));
        assert!(!app.model.groups()[0].focal_is_guess);
    }
}
