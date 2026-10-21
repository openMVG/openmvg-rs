// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Draws the interface.
//!
//! ```text
//! ┌ Images ────────────────┬ Cameras ──────────────┐
//! │ ✓ #0  IMG_0001.jpg …   │ #0 pinhole_radial1 …  │
//! │   #1  IMG_0002.jpg …   ├ Stats ────────────────┤
//! │   —   IMG_0003.jpg …   │ images 487 …          │
//! └────────────────────────┴───────────────────────┘
//!  key hints
//!  status: notice, or what an edit would act on
//! ```

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect, Size};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};

use ratatui_image::Image;

use crate::preview::{Stage, View};
use crate::stats::Stats;
use crate::update::{App, Mode};

/// Group numbers cycle through these colors in the image list and the camera list.
const PALETTE: [Color; 6] = [
    Color::Cyan,
    Color::Green,
    Color::Yellow,
    Color::Magenta,
    Color::Blue,
    Color::Red,
];

const BROWSE_HINTS: &str = "↑↓ move  / filter  space mark  g group  n new  x drop  m model  \
                            f focal  d split  s calib  a auto  u undo  w write  q quit";

pub fn draw(frame: &mut Frame, app: &mut App) {
    let [main, hints, status] = Layout::vertical([
        Constraint::Min(5),
        Constraint::Length(2),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let [files, side] =
        Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)]).areas(main);

    let stats = Stats::of(&app.model);
    let summary = stats.summary_lines();
    let previewing = app.preview.is_some();
    let stats_height = u16::try_from(summary.len() + 2)
        .unwrap_or(u16::MAX)
        .min(side.height / if previewing { 3 } else { 2 });
    let (cameras, preview, stats_area) = if previewing {
        let [cameras, preview, stats] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Percentage(45),
            Constraint::Length(stats_height),
        ])
        .areas(side);
        (cameras, Some(preview), stats)
    } else {
        let [cameras, stats] =
            Layout::vertical([Constraint::Min(3), Constraint::Length(stats_height)]).areas(side);
        (cameras, None, stats)
    };

    draw_images(frame, files, app);
    draw_cameras(frame, cameras, app, &stats);
    if let Some(area) = preview {
        draw_preview(frame, area, app);
    }
    frame.render_widget(
        Paragraph::new(summary.into_iter().map(Line::from).collect::<Vec<_>>())
            .block(Block::bordered().title(" Stats ")),
        stats_area,
    );
    frame.render_widget(
        Paragraph::new(hint(&app.mode))
            .dim()
            .wrap(Wrap { trim: true }),
        hints,
    );
    frame.render_widget(Paragraph::new(status_line(app, &stats)), status);

    match &app.mode {
        Mode::PickGroup { cursor } => draw_group_picker(frame, &stats, *cursor),
        Mode::EditFocal { group, text, .. } => draw_popup(
            frame,
            format!(" Focal length of group #{group}, in pixels "),
            vec![Line::from(format!("{text}▏"))],
        ),
        Mode::ConfirmQuit => draw_popup(
            frame,
            " Unsaved changes ".to_owned(),
            vec![
                Line::from("w  write, then quit"),
                Line::from("y  quit without saving"),
                Line::from("any other key: keep editing"),
            ],
        ),
        Mode::ConfirmSplit { split } => {
            let mut lines: Vec<Line<'static>> = app
                .model
                .describe_split(split)
                .into_iter()
                .map(Line::from)
                .collect();
            lines.push(Line::from(""));
            lines.push(Line::from("Enter  split it   any other key: cancel"));
            draw_popup(
                frame,
                format!(
                    " Split camera #{} into {} cameras? ",
                    split.group,
                    split.clusters.len()
                ),
                lines,
            );
        }
        Mode::Browse | Mode::Filter => {}
    }
}

fn hint(mode: &Mode) -> &'static str {
    match mode {
        Mode::Browse => BROWSE_HINTS,
        Mode::Filter => " type to filter  (terms are ANDed, !term excludes)  Enter keep  Esc clear",
        Mode::PickGroup { .. } => " ↑↓ choose the group  Enter move the images  Esc cancel",
        Mode::EditFocal { .. } => " type the focal length  Enter set  Esc cancel",
        Mode::ConfirmQuit => " w write and quit  y quit without saving  any other key: stay",
        Mode::ConfirmSplit { .. } => " Enter split the camera  any other key: cancel",
    }
}

fn status_line(app: &App, stats: &Stats) -> Line<'static> {
    if let Some(notice) = &app.notice {
        let color = if notice.is_error {
            Color::Red
        } else {
            Color::Green
        };
        return Line::from(Span::styled(
            format!(" {}", notice.text),
            Style::new().fg(color),
        ));
    }
    let warning = app
        .model
        .cursor_group()
        .and_then(|group| stats.groups[group].warnings.first())
        .map_or_else(String::new, |warning| format!("  ·  ⚠ {warning}"));
    let modified = if app.unwritten {
        " [not written]"
    } else if app.model.is_modified() {
        " [modified]"
    } else {
        ""
    };
    Line::from(format!(
        " g, n, x act on {}  ·  {}{modified}{warning}",
        app.model.target_label(),
        app.project_file
    ))
}

/// The first row to draw so that `selected` is inside a window of `height` rows over `len`.
fn window_start(previous: usize, selected: usize, len: usize, height: usize) -> usize {
    let start = if selected < previous {
        selected
    } else if selected >= previous + height {
        selected + 1 - height
    } else {
        previous
    };
    start.min(len.saturating_sub(height))
}

fn group_color(group: usize) -> Color {
    PALETTE[group % PALETTE.len()]
}

fn draw_images(frame: &mut Frame, area: Rect, app: &mut App) {
    let model = &app.model;
    let filtering = matches!(app.mode, Mode::Filter);
    let title = if filtering || !model.filter().is_empty() {
        format!(
            " Images {}/{}  filter: {}{} ",
            model.visible().len(),
            model.rows().len(),
            model.filter(),
            if filtering { "▏" } else { "" }
        )
    } else {
        format!(" Images {} ", model.rows().len())
    };
    let block = Block::bordered().title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let height = usize::from(inner.height);
    let start = window_start(
        app.list.offset,
        model.cursor(),
        model.visible().len(),
        height,
    );
    app.list.inner = inner;
    app.list.offset = start;

    let name_width = model
        .rows()
        .iter()
        .map(|row| row.name.chars().count())
        .max()
        .unwrap_or(0);
    let lines: Vec<Line> = model
        .visible()
        .iter()
        .enumerate()
        .skip(start)
        .take(height)
        .map(|(position, &index)| {
            let row = &model.rows()[index];
            let group = model.group_of(index);
            let (tag, tag_style) = match group {
                Some(group) => (format!("#{group}"), Style::new().fg(group_color(group))),
                None => ("—".to_owned(), Style::new().dark_gray()),
            };
            // Only a mark is yellow: the cursor row is drawn reversed, and a yellow space would
            // turn into a solid block that looks like a mark.
            let mark = if model.is_marked(index) {
                Span::styled("✓", Style::new().fg(Color::Yellow).bold())
            } else {
                Span::raw(" ")
            };
            let mut line = Line::from(vec![
                mark,
                Span::raw(" "),
                Span::styled(format!("{tag:<4}"), tag_style),
                Span::raw(format!(
                    "{:<name_width$}  {:>9}  {}",
                    row.name,
                    format!("{}x{}", row.width, row.height),
                    row.camera
                )),
            ]);
            if group.is_none() {
                line = line.dark_gray();
            }
            if position == model.cursor() {
                line = line.patch_style(Modifier::REVERSED);
            }
            line
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The lines of the camera list that fit in `height` rows around `selected`, which is drawn with
/// `highlight`.
fn group_lines(
    stats: &Stats,
    selected: Option<usize>,
    highlight: Modifier,
    height: usize,
) -> Vec<Line<'static>> {
    let start = window_start(0, selected.unwrap_or(0), stats.groups.len(), height);
    stats
        .groups
        .iter()
        .skip(start)
        .take(height)
        .map(|group| {
            let line = Line::from(Span::styled(
                group.line(),
                Style::new().fg(group_color(group.id)),
            ));
            if selected == Some(group.id) {
                line.patch_style(highlight)
            } else {
                line
            }
        })
        .collect()
}

fn draw_cameras(frame: &mut Frame, area: Rect, app: &App, stats: &Stats) {
    let block = Block::bordered().title(format!(" Cameras ({}) ", stats.groups.len()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let lines = group_lines(
        stats,
        app.model.cursor_group(),
        Modifier::BOLD | Modifier::REVERSED,
        usize::from(inner.height),
    );
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The picture of the image under the cursor, centered in its pane, or why there is none.
fn draw_preview(frame: &mut Frame, area: Rect, app: &mut App) {
    let block = Block::bordered();
    let inner = block.inner(area);
    app.preview_area = Size::new(inner.width, inner.height);
    let dialog_open = !matches!(app.mode, Mode::Browse | Mode::Filter);
    let row = app.model.cursor_row();
    let name = row.map_or("", |row| app.model.rows()[row].name.as_str());
    let titled = format!(" Preview: {name} ");

    let (title, picture, note) = match (row, app.preview.as_mut()) {
        // A terminal image can show through a dialog drawn over it, so it waits.
        (Some(_), _) if dialog_open => (titled, None, "hidden while a dialog is open".to_owned()),
        (Some(row), Some(preview)) => match preview.view(row, app.preview_area) {
            View::Ready { protocol, stage } => {
                let low = if stage == Stage::Thumbnail {
                    " (thumbnail)"
                } else {
                    ""
                };
                (
                    format!(" Preview: {name}{low} "),
                    Some(protocol),
                    String::new(),
                )
            }
            View::Loading => (titled, None, "loading…".to_owned()),
            View::Failed(reason) => (titled, None, format!("no preview: {reason}")),
        },
        _ => (" Preview ".to_owned(), None, "no image".to_owned()),
    };
    frame.render_widget(block.title(title), area);
    match picture {
        Some(protocol) => {
            let size = protocol.size();
            let shown = inner.centered(
                Constraint::Length(size.width),
                Constraint::Length(size.height),
            );
            frame.render_widget(Image::new(protocol), shown);
        }
        None => frame.render_widget(Paragraph::new(note).dim().wrap(Wrap { trim: true }), inner),
    }
}

fn draw_popup(frame: &mut Frame, title: String, lines: Vec<Line<'static>>) {
    let width = lines
        .iter()
        .map(Line::width)
        .chain([title.chars().count()])
        .max()
        .unwrap_or(0);
    let area = frame.area().centered(
        Constraint::Length(u16::try_from(width + 4).unwrap_or(u16::MAX)),
        Constraint::Length(u16::try_from(lines.len() + 2).unwrap_or(u16::MAX)),
    );
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(title)),
        area,
    );
}

fn draw_group_picker(frame: &mut Frame, stats: &Stats, cursor: usize) {
    let height = stats
        .groups
        .len()
        .min(usize::from(frame.area().height.saturating_sub(6)));
    let lines = group_lines(stats, Some(cursor), Modifier::REVERSED, height);
    draw_popup(frame, " Move to group ".to_owned(), lines);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::Split;
    use crate::model::tests::{model, row};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    pub(crate) fn demo_app() -> App {
        let mut rows: Vec<_> = (0..40)
            .map(|i| {
                row(
                    &format!("img-{i:02}.jpg"),
                    400,
                    300,
                    if i < 20 { "Canon EOS" } else { "" },
                )
            })
            .collect();
        rows.push(row("pano.jpg", 800, 400, ""));
        App::new(model(rows), "sfm_project.json".to_owned())
    }

    /// The text of the screen after drawing `app` on a terminal of the given size.
    pub(crate) fn render(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_screen_lists_the_images_and_cameras_and_says_what_an_edit_acts_on() {
        let mut app = demo_app();
        let screen = render(&mut app, 140, 30);
        for expected in [
            "img-00.jpg",
            "Cameras (41)",
            "intrinsics   41",
            "g, n, x act on 1 selected image  ·  sfm_project.json",
        ] {
            assert!(
                screen.contains(expected),
                "missing {expected:?} in\n{screen}"
            );
        }
        assert!(!screen.contains("Preview"), "no pane without a preview");
    }

    #[test]
    fn a_filter_shows_its_matches_and_what_an_edit_would_act_on() {
        let mut app = demo_app();
        app.model.set_filter("img-1 !img-15");
        app.mode = Mode::Filter;
        let screen = render(&mut app, 140, 30);
        assert!(
            screen.contains("Images 9/41  filter: img-1 !img-15▏"),
            "{screen}"
        );
        assert!(!screen.contains("img-15.jpg"), "{screen}");
        assert!(screen.contains("act on 9 filtered images"), "{screen}");
    }

    #[test]
    fn the_list_scrolls_with_the_cursor_and_records_where_the_rows_are() {
        let mut app = demo_app();
        app.model.cursor_to(40);
        let screen = render(&mut app, 100, 14);
        assert!(screen.contains("pano.jpg"), "{screen}");
        assert!(!screen.contains("img-00.jpg"), "{screen}");
        assert_eq!(app.list.inner.height, 9);
        assert_eq!(app.list.offset, 41 - 9);
    }

    #[test]
    fn every_dialog_and_notice_is_drawn_over_the_screen() {
        let split = Split {
            group: 3,
            clusters: vec![vec![0, 1], vec![2]],
        };
        let dialogs = [
            (Mode::PickGroup { cursor: 2 }, "Move to group"),
            (
                Mode::EditFocal {
                    group: 3,
                    text: "812".into(),
                    initial: "1000.0".into(),
                },
                "Focal length of group #3, in pixels",
            ),
            (Mode::ConfirmQuit, "quit without saving"),
            (
                Mode::ConfirmSplit { split },
                "Split camera #3 into 2 cameras?",
            ),
        ];
        for (mode, expected) in dialogs {
            let mut app = demo_app();
            app.mode = mode.clone();
            let screen = render(&mut app, 120, 30);
            assert!(
                screen.contains(expected),
                "{mode:?}: missing {expected:?} in\n{screen}"
            );
        }

        // The split dialog lists the new cameras, from the model, at draw time.
        let mut app = demo_app();
        app.mode = Mode::ConfirmSplit {
            split: Split {
                group: 3,
                clusters: vec![vec![0, 1], vec![2]],
            },
        };
        let screen = render(&mut app, 120, 30);
        assert!(screen.contains("2 img  f=1000.0  Canon EOS"), "{screen}");

        let mut app = demo_app();
        app.error("images of different sizes cannot share a camera".into());
        let screen = render(&mut app, 120, 30);
        assert!(screen.contains("different sizes"), "{screen}");
    }

    #[test]
    fn the_key_hints_wrap_instead_of_being_cut_in_a_narrow_terminal() {
        let mut app = demo_app();
        let screen = render(&mut app, 64, 20);
        for key in ["↑↓ move", "g group", "w write", "q quit"] {
            assert!(screen.contains(key), "missing {key:?} in\n{screen}");
        }
    }

    #[test]
    fn tiny_terminals_do_not_panic_in_any_mode() {
        let modes = [
            Mode::Browse,
            Mode::Filter,
            Mode::PickGroup { cursor: 0 },
            Mode::EditFocal {
                group: 0,
                text: "1".into(),
                initial: "1".into(),
            },
            Mode::ConfirmQuit,
            Mode::ConfirmSplit {
                split: Split {
                    group: 0,
                    clusters: vec![vec![0], vec![1]],
                },
            },
        ];
        for (width, height) in [(1, 1), (10, 3), (30, 8), (60, 5)] {
            for mode in &modes {
                let mut app = demo_app();
                app.mode = mode.clone();
                render(&mut app, width, height);
            }
        }
    }

    #[test]
    fn a_warning_marks_the_camera_and_explains_itself_in_the_status_line() {
        let mut app = demo_app();
        // Row 0 names a camera and row 25 does not: two cameras in one.
        for row in [0, 25] {
            app.model.cursor_to(row);
            app.model.toggle_mark();
        }
        app.model.move_to_group(0).unwrap();
        app.model.clear_marks();
        app.model.cursor_to(0);

        let screen = render(&mut app, 160, 34);

        assert!(screen.contains("#0 ⚠ pinhole_radial1"), "{screen}");
        assert!(screen.contains("warnings     1"), "{screen}");
        assert!(
            screen.contains("[modified]  ·  ⚠ mixes 2 cameras (press d to split them)"),
            "{screen}"
        );
    }

    #[test]
    fn only_a_marked_row_shows_a_yellow_mark_even_under_the_cursor() {
        let mut app = demo_app();
        let mark_cell = |app: &mut App| {
            let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
            terminal.draw(|frame| draw(frame, app)).unwrap();
            // The first row of the list, in its first column inside the border.
            let cell = terminal.backend().buffer()[(1, 1)].clone();
            (cell.symbol().to_owned(), cell.fg)
        };

        let (symbol, color) = mark_cell(&mut app);
        assert_eq!(symbol, " ");
        assert_ne!(color, Color::Yellow, "the cursor row is not marked");

        app.model.toggle_mark();
        app.model.cursor_to(0);
        let (symbol, color) = mark_cell(&mut app);
        assert_eq!((symbol.as_str(), color), ("✓", Color::Yellow));
    }
}
