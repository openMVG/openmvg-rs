// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `openmvg-sfm-configure`: prepares an image folder as a project file for `openmvg-sfm --project`.
//!
//! ```text
//! folder ──► Session::open ──► Model ◄── handle_key / handle_mouse ── terminal events
//!            (scan, or load           │
//!             the project file)       ├──► ui::draw ──► terminal
//!                                     │       ▲
//!                                     │    Preview (worker thread: thumbnail, decode, encode)
//!                                     └──► Session::write ──► project file ──► openmvg-sfm --project
//! ```
//!
//! The scan groups the images automatically; the interface edits that grouping, and the file it
//! writes is the grouping `openmvg-sfm` runs with.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod model;
mod preview;
mod scan;
mod stats;
mod thumbnail;
mod ui;
mod update;

use std::io::{IsTerminal, stdout};
use std::path::PathBuf;
use std::process::ExitCode;
use std::thread;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui_image::picker::cap_parser::QueryStdioOptions;
use ratatui_image::picker::{Picker, ProtocolType};

use crate::model::Model;
use crate::preview::Preview;
use crate::scan::{Grouping, Session};
use crate::stats::Stats;
use crate::update::{App, Effect};

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

/// The project file's name in the image folder, unless `--project` names another.
const DEFAULT_PROJECT: &str = "sfm_project.json";

/// How long to wait for the terminal to answer the graphics query. A terminal that never answers,
/// such as a Linux console or a serial line, costs this once at startup before the preview falls
/// back to half blocks; the library waits 2 s by default.
const QUERY_TIMEOUT: Duration = Duration::from_millis(300);

/// How the preview draws pictures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PreviewMode {
    /// Ask the terminal which graphics protocol it supports; fall back to half blocks.
    Auto,
    /// No preview pane.
    Off,
    /// Unicode half blocks, which every terminal draws.
    Halfblocks,
    /// The Kitty graphics protocol (kitty, Ghostty).
    Kitty,
    /// Sixel graphics (foot, xterm, mlterm).
    Sixel,
    /// The iTerm2 inline image protocol (iTerm2, WezTerm, Rio).
    Iterm2,
}

impl PreviewMode {
    /// The picker for this mode, or `None` when the preview is off. A terminal query needs the
    /// terminal in raw mode, so call this after the interface has started.
    fn picker(self) -> Option<Picker> {
        let forced = match self {
            Self::Off => return None,
            Self::Auto => None,
            Self::Halfblocks => return Some(Picker::halfblocks()),
            Self::Kitty => Some(ProtocolType::Kitty),
            Self::Sixel => Some(ProtocolType::Sixel),
            Self::Iterm2 => Some(ProtocolType::Iterm2),
        };
        let options = QueryStdioOptions {
            timeout: QUERY_TIMEOUT,
            ..QueryStdioOptions::default()
        };
        let mut picker =
            Picker::from_query_stdio_with_options(options).unwrap_or_else(|_| Picker::halfblocks());
        if let Some(protocol) = forced {
            picker.set_protocol_type(protocol);
        }
        Some(picker)
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "openmvg-sfm-configure",
    about = "Group the images of a folder into cameras and write the project file for `openmvg-sfm --project`"
)]
struct Cli {
    /// Image directory, or a dataset root containing images/: the argument of `openmvg-sfm`.
    folder: PathBuf,

    /// Project file to edit, or to create when it does not exist
    /// (default: `sfm_project.json` in the folder). An existing file is the starting point and
    /// `--group` does not apply to it.
    #[arg(long, value_name = "FILE")]
    project: Option<PathBuf>,

    /// How a new project groups the images: `auto` by EXIF camera, lens, and focal length (images
    /// with no camera share a group by size), or `off` for one camera per image. Press `a` in the
    /// interface to return to `auto`.
    #[arg(long, value_enum, default_value_t = Grouping::Auto)]
    group: Grouping,

    /// How the preview pane draws the image under the cursor. `auto` asks the terminal; inside
    /// tmux, graphics need `set -g allow-passthrough on`, and `halfblocks` always works.
    #[arg(long, value_enum, default_value_t = PreviewMode::Auto)]
    preview: PreviewMode,

    /// Check the project file against the folder, print its cameras, statistics, and warnings, and
    /// change nothing. Fails when the file is not a valid project, names an image the folder lacks,
    /// or groups images that cannot share a camera. Needs no terminal.
    #[arg(long, conflicts_with_all = ["no_tui", "group"])]
    check: bool,

    /// Print the statistics, and write the project when the file does not exist yet, without the
    /// terminal interface.
    #[arg(long)]
    no_tui: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> DynResult<()> {
    let cli = Cli::parse();
    let project_file = cli
        .project
        .unwrap_or_else(|| cli.folder.join(DEFAULT_PROJECT));
    if cli.check {
        return run_check(&cli.folder, &project_file);
    }
    if !cli.no_tui && !stdout().is_terminal() {
        return Err(
            "stdout is not a terminal; use --no-tui to write the project headlessly".into(),
        );
    }
    eprintln!("scanning {} ...", cli.folder.display());
    let (session, model) = Session::open(&cli.folder, project_file, cli.group)?;
    if cli.no_tui {
        headless(&session, model)
    } else {
        let app = new_app(&session, model);
        run_terminal(&session, app, cli.preview)
    }
}

/// The interface for `model`, which marks the project as unwritten when its file is new.
fn new_app(session: &Session, model: Model) -> App {
    let unreadable = model.unreadable().len();
    let first = model.unreadable().first().cloned();
    let mut app = App::new(model, session.project_file.display().to_string());
    app.unwritten = session.is_new;
    if let Some(first) = first {
        app.error(format!(
            "{unreadable} image(s) cannot be read and are left out; the first is {}: {}",
            first.name, first.reason
        ));
    }
    app
}

/// Prints what `--check` found; an error when the project does not fit the folder.
fn run_check(folder: &std::path::Path, project_file: &std::path::Path) -> DynResult<()> {
    match scan::check(folder, project_file) {
        Ok(report) => {
            for line in &report {
                println!("{line}");
            }
            println!("{} fits the folder", project_file.display());
            Ok(())
        }
        Err(problem) => {
            eprintln!("problem: {problem}");
            Err(format!("{} does not fit the folder", project_file.display()).into())
        }
    }
}

/// Prints the cameras and the statistics, and writes a project that does not exist yet.
fn headless(session: &Session, mut model: Model) -> DynResult<()> {
    let stats = Stats::of(&model);
    for group in &stats.groups {
        println!("{}", group.line());
    }
    for line in scan::unreadable_lines(&model) {
        println!("{line}");
    }
    for line in stats.summary_lines() {
        println!("{line}");
    }
    if session.is_new {
        let images = session.write(&mut model)?;
        println!(
            "wrote {images} images to {}",
            session.project_file.display()
        );
    } else {
        println!(
            "{} exists and was left as it is",
            session.project_file.display()
        );
    }
    Ok(())
}

fn run_terminal(session: &Session, mut app: App, preview: PreviewMode) -> DynResult<()> {
    let mut terminal = ratatui::init();
    app.preview = preview.picker().map(Preview::new);
    // The default panic hook restores the screen; mouse capture is ours to release. A panic in the
    // preview worker is caught and shown in the pane, so it must not restore the screen under the
    // running interface.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if thread::current().name() == Some(preview::WORKER_NAME) {
            return;
        }
        let _ = execute!(stdout(), DisableMouseCapture);
        hook(info);
    }));
    let result = execute!(stdout(), EnableMouseCapture)
        .map_err(Into::into)
        .and_then(|()| event_loop(&mut terminal, session, &mut app));
    let _ = execute!(stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

/// How long the loop waits for a key before it checks for finished pictures.
const TICK: Duration = Duration::from_millis(30);

fn event_loop(terminal: &mut DefaultTerminal, session: &Session, app: &mut App) -> DynResult<()> {
    let mut redraw = true;
    loop {
        if redraw {
            terminal.draw(|frame| ui::draw(frame, app))?;
            refresh_preview(session, app);
            redraw = false;
        }
        if event::poll(TICK)? {
            let event = event::read()?;
            // Key releases and pointer motion change nothing, and `EnableMouseCapture` reports
            // every cell the pointer crosses.
            redraw |= match &event {
                Event::Key(key) => key.kind == KeyEventKind::Press,
                Event::Mouse(mouse) => mouse.kind != MouseEventKind::Moved,
                _ => true,
            };
            let effect = match event {
                Event::Key(key) if key.kind == KeyEventKind::Press => update::handle_key(app, key),
                Event::Mouse(mouse) => {
                    update::handle_mouse(app, mouse);
                    Effect::None
                }
                _ => Effect::None,
            };
            match effect {
                Effect::None => {}
                Effect::Quit => return Ok(()),
                Effect::Write => {
                    save(session, app);
                }
                Effect::WriteAndQuit => {
                    if save(session, app) {
                        return Ok(());
                    }
                }
            }
        }
        if let Some(preview) = &mut app.preview {
            redraw |= preview.receive();
        }
    }
}

/// Asks the preview for the image under the cursor, at the size of the pane as last drawn.
fn refresh_preview(session: &Session, app: &mut App) {
    let (Some(row), Some(preview)) = (app.model.cursor_row(), app.preview.as_mut()) else {
        return;
    };
    preview.select(row, session.path(row).to_path_buf(), app.preview_area);
}

/// Writes the project and reports the outcome in the status line.
fn save(session: &Session, app: &mut App) -> bool {
    match session.write(&mut app.model) {
        Ok(images) => {
            app.unwritten = false;
            app.info(format!(
                "wrote {images} images to {}",
                session.project_file.display()
            ));
            true
        }
        Err(error) => {
            app.error(error.to_string());
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent};
    use std::time::Instant;

    #[test]
    fn the_command_line_matches_the_readme() {
        let parse = |args: &[&str]| {
            Cli::try_parse_from(
                ["openmvg-sfm-configure", "images"]
                    .into_iter()
                    .chain(args.iter().copied()),
            )
        };
        let defaults = parse(&[]).unwrap();
        assert_eq!(defaults.group, Grouping::Auto);
        assert_eq!(defaults.preview, PreviewMode::Auto);

        for (name, expected) in [
            ("off", PreviewMode::Off),
            ("halfblocks", PreviewMode::Halfblocks),
            ("kitty", PreviewMode::Kitty),
            ("sixel", PreviewMode::Sixel),
            ("iterm2", PreviewMode::Iterm2),
        ] {
            assert_eq!(parse(&["--preview", name]).unwrap().preview, expected);
        }
        assert!(PreviewMode::Off.picker().is_none());

        assert!(parse(&["--check"]).unwrap().check);
        for conflicting in [&["--no-tui"][..], &["--group", "off"]] {
            let args = [&["--check"], conflicting].concat();
            assert_eq!(
                parse(&args).err().map(|error| error.kind()),
                Some(clap::error::ErrorKind::ArgumentConflict),
                "{conflicting:?}"
            );
        }
    }

    /// Two-tone PNGs, so the half-block preview draws `▀`.
    fn folder() -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        for index in 0..2 {
            image::RgbImage::from_fn(64, 48, |_, y| {
                image::Rgb(if y < 24 { [255; 3] } else { [0; 3] })
            })
            .save(dir.path().join(format!("view-{index}.png")))
            .unwrap();
        }
        dir
    }

    /// A session over `dir` with a half-block preview.
    fn previewing(dir: &tempfile::TempDir) -> (Session, App) {
        let (session, model) =
            Session::open(dir.path(), dir.path().join("project.json"), Grouping::Auto).unwrap();
        let mut app = new_app(&session, model);
        app.preview = PreviewMode::Halfblocks.picker().map(Preview::new);
        (session, app)
    }

    /// Draws and refreshes the preview as the event loop does until the screen satisfies `done`,
    /// waiting up to ten seconds for the worker.
    fn draw_until(app: &mut App, session: &Session, done: impl Fn(&str) -> bool) -> String {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let screen = ui::tests::render(app, 120, 30);
            refresh_preview(session, app);
            if done(&screen) {
                return screen;
            }
            assert!(
                Instant::now() < deadline,
                "the screen never showed what the test waits for:\n{screen}"
            );
            thread::sleep(Duration::from_millis(10));
            if let Some(preview) = &mut app.preview {
                preview.receive();
            }
        }
    }

    #[test]
    fn the_pane_shows_the_image_under_the_cursor_and_follows_the_cursor() {
        let dir = folder();
        let (session, mut app) = previewing(&dir);

        let first = draw_until(&mut app, &session, |screen| screen.contains('▀'));
        assert!(first.contains("Preview: view-0.png"), "{first}");

        app.model.move_cursor(1);
        let second = draw_until(&mut app, &session, |screen| screen.contains('▀'));
        assert!(second.contains("Preview: view-1.png"), "{second}");
    }

    #[test]
    fn a_dialog_hides_the_picture_until_it_closes() {
        let dir = folder();
        let (session, mut app) = previewing(&dir);
        draw_until(&mut app, &session, |screen| screen.contains('▀'));

        app.mode = update::Mode::ConfirmQuit;
        let hidden = ui::tests::render(&mut app, 120, 30);
        assert!(hidden.contains("hidden while a dialog is open"), "{hidden}");
        assert!(!hidden.contains('▀'), "{hidden}");

        app.mode = update::Mode::Browse;
        draw_until(&mut app, &session, |screen| screen.contains('▀'));
    }

    #[test]
    fn a_truncated_image_says_why_there_is_no_preview() {
        let dir = folder();
        // The header survives, so the scan accepts the file; the compressed pixels do not.
        let broken = dir.path().join("view-1.png");
        let mut bytes = std::fs::read(&broken).unwrap();
        let (data_start, data_end) = (bytes.len() / 2, bytes.len() - 12);
        bytes[data_start..data_end].fill(0xFF);
        std::fs::write(&broken, bytes).unwrap();
        let (session, mut app) = previewing(&dir);
        app.model.move_cursor(1);

        let screen = draw_until(&mut app, &session, |screen| screen.contains("no preview:"));

        assert!(screen.contains("Preview: view-1.png"), "{screen}");
    }

    #[test]
    fn a_new_project_is_marked_and_quitting_asks_until_it_is_written() {
        let dir = folder();
        let (session, mut app) = previewing(&dir);
        assert!(session.is_new && app.unwritten);
        assert!(ui::tests::render(&mut app, 120, 30).contains("[not written]"));

        let quit = KeyEvent::from(KeyCode::Char('q'));
        assert_eq!(update::handle_key(&mut app, quit), Effect::None);
        assert_eq!(
            app.mode,
            update::Mode::ConfirmQuit,
            "asks before leaving no file"
        );
        app.mode = update::Mode::Browse;

        assert!(save(&session, &mut app));
        assert!(!app.unwritten);
        assert_eq!(update::handle_key(&mut app, quit), Effect::Quit);
        assert!(!ui::tests::render(&mut app, 120, 30).contains("[not written]"));
    }
}
