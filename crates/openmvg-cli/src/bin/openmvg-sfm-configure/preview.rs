// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The image preview: decoding and terminal-graphics encoding happen on a worker thread, so the
//! interface never waits for a picture.
//!
//! ```text
//! cursor row ─► Preview::select ──► Request ──► worker ─┬─ EXIF thumbnail ─► encode ─► Reply (Thumbnail)
//!                (latest wins)                          └─ full decode ────► encode ─► Reply (Full)
//! ui::draw ◄── Preview::view ◄── cache ◄── Preview::receive ◄───────────────────────────┘
//! ```
//!
//! Only the newest request is served: a request that a newer one overtakes is dropped, and the
//! full decode is skipped when a newer request is already waiting. Each reply carries a ratatui-image
//! `Protocol` already encoded for the pane's size in cells, so drawing it costs nothing.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Duration;

use image::DynamicImage;
use ratatui::layout::Size;
use ratatui_image::Resize;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::Protocol;

use crate::thumbnail::{decode_preview, exif_thumbnail};

/// How long the cursor must rest before a full decode starts.
const DEBOUNCE: Duration = Duration::from_millis(80);

/// How many pictures each side keeps.
const CACHE: usize = 16;

/// How sharp a cached picture is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The thumbnail the camera embedded: instant, low resolution.
    Thumbnail,
    /// The decoded image, shrunk to the preview edge.
    Full,
}

struct Request {
    row: usize,
    path: PathBuf,
    /// The pane, in cells.
    size: Size,
}

struct Reply {
    row: usize,
    entry: Entry,
}

/// What the pane shows for a row.
pub enum View<'a> {
    Loading,
    Ready {
        protocol: &'a Protocol,
        stage: Stage,
    },
    Failed(&'a str),
}

/// A small least-recently-used map from row to value.
struct Lru<V> {
    items: Vec<(usize, V)>,
}

impl<V> Lru<V> {
    const fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn peek(&self, key: usize) -> Option<&V> {
        self.items.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    /// Looks `key` up and makes it the most recently used.
    fn touch(&mut self, key: usize) -> Option<&V> {
        let at = self.items.iter().position(|(k, _)| *k == key)?;
        let item = self.items.remove(at);
        self.items.push(item);
        self.items.last().map(|(_, v)| v)
    }

    fn insert(&mut self, key: usize, value: V) {
        self.items.retain(|(k, _)| *k != key);
        self.items.push((key, value));
        if self.items.len() > CACHE {
            self.items.remove(0);
        }
    }
}

struct Entry {
    size: Size,
    stage: Stage,
    outcome: Result<Protocol, String>,
}

/// The preview of the image under the cursor, loaded in the background.
pub struct Preview {
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    cache: Lru<Entry>,
    /// The row and pane size most recently asked for.
    requested: Option<(usize, Size)>,
    /// The worker stopped, so nothing more will arrive.
    worker_stopped: bool,
}

impl Preview {
    /// Starts the worker; `picker` decides how pictures are encoded for the terminal.
    pub fn new(picker: Picker) -> Self {
        let (requests, inbox) = channel();
        let (outbox, replies) = channel();
        let worker = thread::Builder::new()
            .name(WORKER_NAME.to_owned())
            .spawn(move || run(&picker, &Inbox::new(inbox), &outbox));
        Self {
            requests,
            replies,
            cache: Lru::new(),
            requested: None,
            worker_stopped: worker.is_err(),
        }
    }

    /// Asks for the picture of `row` at the pane `size` unless it is loaded or already asked for.
    pub fn select(&mut self, row: usize, path: PathBuf, size: Size) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let loaded = matches!(
            self.cache.peek(row),
            Some(entry) if entry.size == size && entry.stage == Stage::Full
        );
        if loaded || self.requested == Some((row, size)) {
            return;
        }
        let request = Request { row, path, size };
        self.worker_stopped |= self.requests.send(request).is_err();
        self.requested = Some((row, size));
    }

    /// Takes in the pictures the worker has finished; returns whether there were any.
    pub fn receive(&mut self) -> bool {
        let mut any = false;
        while let Ok(reply) = self.replies.try_recv() {
            self.cache.insert(reply.row, reply.entry);
            any = true;
        }
        any
    }

    /// What to show for `row` in a pane of `size`.
    pub fn view(&mut self, row: usize, size: Size) -> View<'_> {
        if self.worker_stopped {
            return View::Failed("the preview worker stopped");
        }
        match self.cache.touch(row) {
            Some(entry) if entry.size == size => match &entry.outcome {
                Ok(protocol) => View::Ready {
                    protocol,
                    stage: entry.stage,
                },
                Err(message) => View::Failed(message),
            },
            _ => View::Loading,
        }
    }
}

/// The worker's queue: it serves the newest request and notices when a newer one is waiting.
struct Inbox {
    requests: Receiver<Request>,
    waiting: std::cell::RefCell<Option<Request>>,
}

impl Inbox {
    const fn new(requests: Receiver<Request>) -> Self {
        Self {
            requests,
            waiting: std::cell::RefCell::new(None),
        }
    }

    /// Blocks for a request, then returns the newest one; `None` when the interface is gone.
    fn next_newest(&self) -> Option<Request> {
        let mut newest = match self.waiting.borrow_mut().take() {
            Some(request) => request,
            None => self.requests.recv().ok()?,
        };
        while let Ok(newer) = self.requests.try_recv() {
            newest = newer;
        }
        Some(newest)
    }

    /// Waits up to `wait` for a newer request, which makes the one being served stale, and
    /// returns whether one is waiting. The request is kept for the next [`next_newest`].
    ///
    /// [`next_newest`]: Self::next_newest
    fn newer_within(&self, wait: Duration) -> bool {
        let mut waiting = self.waiting.borrow_mut();
        if waiting.is_none() {
            *waiting = self.requests.recv_timeout(wait).ok();
        }
        waiting.is_some()
    }
}

fn encode(picker: &Picker, image: &DynamicImage, size: Size) -> Result<Protocol, String> {
    picker
        .new_protocol(image.clone(), size, Resize::Fit(None))
        .map_err(|error| error.to_string())
}

/// Name of the worker thread; the panic hook in `main` leaves the screen alone for its panics,
/// which [`guarded`] turns into a message in the pane.
pub const WORKER_NAME: &str = "preview";

/// Runs `work`, turning a panic into a message: a decoder that panics on a malformed file must
/// not take the worker, or the screen, down with it.
fn guarded<T>(work: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).map_err(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned());
        match detail {
            Some(detail) => format!("decoding this file panicked: {detail}"),
            None => "decoding this file panicked".to_owned(),
        }
    })
}

fn run(picker: &Picker, inbox: &Inbox, outbox: &Sender<Reply>) {
    // The decoded pictures of recent rows, so a resized pane re-encodes without decoding again.
    let mut decoded: Lru<(Stage, DynamicImage)> = Lru::new();
    while let Some(request) = inbox.next_newest() {
        let alive = match guarded(|| serve(picker, inbox, &request, &mut decoded, outbox)) {
            Ok(alive) => alive,
            Err(message) => outbox
                .send(Reply {
                    row: request.row,
                    entry: Entry {
                        size: request.size,
                        stage: Stage::Full,
                        outcome: Err(message),
                    },
                })
                .is_ok(),
        };
        if !alive {
            return;
        }
    }
}

/// Answers one request; `false` when the interface has hung up.
fn serve(
    picker: &Picker,
    inbox: &Inbox,
    request: &Request,
    decoded: &mut Lru<(Stage, DynamicImage)>,
    outbox: &Sender<Reply>,
) -> bool {
    let reply = |stage, outcome| {
        outbox
            .send(Reply {
                row: request.row,
                entry: Entry {
                    size: request.size,
                    stage,
                    outcome,
                },
            })
            .is_ok()
    };
    let have = decoded.peek(request.row).map(|(stage, _)| *stage);
    if have.is_none()
        && let Some(thumbnail) = exif_thumbnail(&request.path)
    {
        let outcome = encode(picker, &thumbnail, request.size);
        decoded.insert(request.row, (Stage::Thumbnail, thumbnail));
        if !reply(Stage::Thumbnail, outcome) {
            return false;
        }
    }
    if have == Some(Stage::Full) {
        return decoded
            .peek(request.row)
            .is_none_or(|(_, image)| reply(Stage::Full, encode(picker, image, request.size)));
    }
    // A decode cannot be cancelled, so it starts only once the cursor has rested: holding an
    // arrow would otherwise queue half-second decodes of pictures nobody sees.
    if inbox.newer_within(DEBOUNCE) {
        return true;
    }
    let outcome = decode_preview(&request.path).and_then(|image| {
        let outcome = encode(picker, &image, request.size);
        decoded.insert(request.row, (Stage::Full, image));
        outcome
    });
    reply(Stage::Full, outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thumbnail::tests::jpeg_with_thumbnail;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use ratatui_image::Image;
    use ratatui_image::picker::ProtocolType;
    use std::time::{Duration, Instant};

    const PANE: Size = Size::new(20, 10);

    /// Receives until `done` holds for the cache, giving the worker up to ten seconds.
    fn wait_for(preview: &mut Preview, done: impl Fn(&Preview) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            preview.receive();
            if done(preview) {
                return;
            }
            assert!(Instant::now() < deadline, "the worker produced nothing");
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn stage_of(preview: &Preview, row: usize) -> Option<Stage> {
        preview.cache.peek(row).map(|entry| entry.stage)
    }

    /// Runs the worker on this thread over `requests`, all waiting before it starts, and lists
    /// the `(row, stage)` of every reply in order.
    fn serve_all(requests: Vec<Request>) -> Vec<(usize, Stage)> {
        let (sender, inbox) = channel();
        let (outbox, replies) = channel();
        for request in requests {
            sender.send(request).unwrap();
        }
        drop(sender);
        run(&Picker::halfblocks(), &Inbox::new(inbox), &outbox);
        drop(outbox);
        replies
            .iter()
            .map(|reply| (reply.row, reply.entry.stage))
            .collect()
    }

    fn request(row: usize, path: PathBuf) -> Request {
        Request {
            row,
            path,
            size: PANE,
        }
    }

    #[test]
    fn the_worker_replies_with_the_thumbnail_then_the_full_image() {
        let dir = tempfile::TempDir::new().unwrap();
        let jpeg = dir.path().join("with.jpg");
        jpeg_with_thumbnail(&jpeg, (400, 300), (40, 30), 1);
        let png = dir.path().join("without.png");
        image::DynamicImage::new_rgb8(64, 48).save(&png).unwrap();

        assert_eq!(
            serve_all(vec![request(0, jpeg)]),
            [(0, Stage::Thumbnail), (0, Stage::Full)]
        );
        assert_eq!(
            serve_all(vec![request(1, png)]),
            [(1, Stage::Full)],
            "a PNG has no thumbnail"
        );
    }

    #[test]
    fn only_the_newest_request_is_served_by_the_worker() {
        let dir = tempfile::TempDir::new().unwrap();
        let png = dir.path().join("a.png");
        image::DynamicImage::new_rgb8(64, 48).save(&png).unwrap();

        let served = serve_all(vec![request(0, png.clone()), request(1, png)]);

        assert_eq!(served, [(1, Stage::Full)], "row 0 was stale");
    }

    #[test]
    fn a_file_that_cannot_be_decoded_is_a_failure_the_pane_can_show() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("broken.png");
        std::fs::write(&path, b"not an image").unwrap();
        let mut preview = Preview::new(Picker::halfblocks());

        preview.select(3, path, PANE);

        wait_for(&mut preview, |p| stage_of(p, 3).is_some());
        assert!(matches!(preview.view(3, PANE), View::Failed(_)));
    }

    #[test]
    fn a_pane_of_another_size_asks_again_and_is_loading_until_it_arrives() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("a.png");
        image::DynamicImage::new_rgb8(64, 48).save(&path).unwrap();
        let mut preview = Preview::new(Picker::halfblocks());
        preview.select(0, path.clone(), PANE);
        wait_for(&mut preview, |p| stage_of(p, 0) == Some(Stage::Full));

        let larger = Size::new(40, 20);
        assert!(matches!(preview.view(0, larger), View::Loading));
        preview.select(0, path, larger);
        wait_for(
            &mut preview,
            |p| matches!(p.cache.peek(0), Some(e) if e.size == larger),
        );

        assert!(matches!(preview.view(0, larger), View::Ready { .. }));
    }

    #[test]
    fn a_request_is_sent_once_and_only_for_a_picture_that_is_not_loaded() {
        let mut preview = Preview::new(Picker::halfblocks());
        let (sender, receiver) = channel();
        preview.requests = sender;
        let entry = |stage| Entry {
            size: PANE,
            stage,
            outcome: Err("test".to_owned()),
        };
        preview.cache.insert(2, entry(Stage::Full));
        preview.cache.insert(3, entry(Stage::Thumbnail));
        let path = PathBuf::from("a.png");

        preview.select(0, path.clone(), PANE);
        preview.select(0, path.clone(), PANE);
        preview.select(1, path.clone(), PANE);
        preview.select(1, path.clone(), Size::new(0, 5));
        preview.select(2, path.clone(), PANE);
        preview.select(3, path, PANE);

        assert_eq!(
            receiver.try_iter().map(|r| r.row).collect::<Vec<_>>(),
            [0, 1, 3],
            "a repeat, an empty pane, and a loaded picture are not sent; a thumbnail is not loaded"
        );
    }

    #[test]
    fn only_the_newest_waiting_request_is_served_and_a_stale_one_is_noticed() {
        let (sender, receiver) = channel();
        let inbox = Inbox::new(receiver);
        let request = |row| Request {
            row,
            path: PathBuf::new(),
            size: PANE,
        };
        for row in 0..3 {
            sender.send(request(row)).unwrap();
        }

        assert_eq!(inbox.next_newest().map(|r| r.row), Some(2));
        let wait = Duration::from_millis(5);
        assert!(!inbox.newer_within(wait), "nothing arrived while it waited");
        sender.send(request(7)).unwrap();
        assert!(inbox.newer_within(wait));
        assert!(inbox.newer_within(wait), "noticing does not consume it");
        assert_eq!(inbox.next_newest().map(|r| r.row), Some(7));
        drop(sender);
        assert!(
            inbox.next_newest().is_none(),
            "the worker ends with the interface"
        );
    }

    #[test]
    fn the_cache_keeps_the_most_recently_used_pictures() {
        let mut lru = Lru::new();
        for key in 0..CACHE {
            lru.insert(key, key);
        }
        lru.touch(0);
        lru.insert(CACHE, CACHE);
        assert!(lru.peek(0).is_some(), "touched, so kept");
        assert!(lru.peek(1).is_none(), "least recently used, so dropped");
        assert_eq!(lru.items.len(), CACHE);
    }

    /// The text of every cell of the screen, where an encoded picture leaves its escape sequence.
    fn screen_text(protocol: &Protocol) -> String {
        let mut terminal = Terminal::new(TestBackend::new(20, 10)).unwrap();
        terminal
            .draw(|frame| frame.render_widget(Image::new(protocol), frame.area()))
            .unwrap();
        let buffer: &Buffer = terminal.backend().buffer();
        buffer
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>()
    }

    #[test]
    fn every_graphics_protocol_encodes_a_picture_for_the_pane() {
        // Two tones, so a half block's upper and lower colors differ.
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(64, 48, |_, y| {
            image::Rgb(if y < 24 { [255; 3] } else { [0; 3] })
        }));
        let expect = [
            (ProtocolType::Kitty, "\u{1b}_G"),
            (ProtocolType::Sixel, "\u{1b}P"),
            (ProtocolType::Iterm2, "\u{1b}]1337;File="),
            (ProtocolType::Halfblocks, "▀"),
        ];
        for (kind, marker) in expect {
            let mut picker = Picker::halfblocks();
            picker.set_protocol_type(kind);
            let protocol = encode(&picker, &image, PANE).unwrap();
            assert!(
                screen_text(&protocol).contains(marker),
                "{kind:?} did not emit {marker:?}"
            );
        }
    }

    #[test]
    fn a_panic_while_decoding_becomes_a_message() {
        assert_eq!(guarded(|| 7), Ok(7));
        let message = guarded(|| -> u8 { panic!("bad chunk") }).unwrap_err();
        assert_eq!(message, "decoding this file panicked: bad chunk");
        let message = guarded(|| -> u8 { std::panic::panic_any(3_u8) }).unwrap_err();
        assert_eq!(message, "decoding this file panicked");
    }
}
