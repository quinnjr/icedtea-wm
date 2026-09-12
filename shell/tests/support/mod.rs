//! Shared scaffolding for `icedtea-shell`'s integration tests: one compositor,
//! one real panel on its own thread, recording mocks behind the two command
//! traits, and a pointer to click with.
//!
//! Not every test uses every helper, hence the blanket `dead_code` allow: this
//! module is compiled once per test binary that declares it.

#![allow(
    dead_code,
    reason = "shared test-support module compiled per test binary; not every binary uses every helper"
)]

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use icedtea_contract::{
    ClipEntry, ClipKind, Rectangle, Snapshot, WindowId, WindowInfo, WorkspaceInfo,
};
use icedtea_harness::{CapturedFrame, Compositor, ScreencopyClient, VirtualPointerClient};
use icedtea_shell::clip_client::ClipCommands;
use icedtea_shell::compositor_client::CompositorCommands;
use icedtea_shell::panel::{self, Msg, PanelModel};
use icedtea_shell::style;
use icedtea_ui::gallery::Theme;
use icedtea_ui::text::FontDatabase;
use icedtea_ui::view::{App, Inbox, InboxSender, PopupEvent};

/// How long any wait in this module gives the compositor and the panel.
///
/// A generous complexity bound, never a timing pin: the panel has to connect,
/// take a configure, lay out and paint before its first report exists.
const TIMEOUT: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(50);

/// A recording `CompositorCommands`, shared with the panel's thread.
///
/// `Arc<Mutex<_>>`, unlike `panel.rs`'s unit-test mock: the panel runs on its
/// own thread here and the assertions read from the test's.
#[derive(Clone, Default)]
pub struct MockWm(pub Arc<Mutex<Vec<(String, u32)>>>);

impl CompositorCommands for MockWm {
    fn focus_window(&self, id: u32) {
        self.0.lock().expect("wm calls").push(("focus".into(), id));
    }
    fn close_window(&self, id: u32) {
        self.0.lock().expect("wm calls").push(("close".into(), id));
    }
    fn set_workspace(&self, id: u32) {
        self.0
            .lock()
            .expect("wm calls")
            .push(("workspace".into(), id));
    }
}

/// A recording `ClipCommands`, shared with the panel's thread.
///
/// Records `(op, id, on)`: the `on` flag carries `pin`'s direction so a gate
/// can assert it; non-pin calls use `false` as a sentinel.
#[derive(Clone, Default)]
pub struct MockClip(pub Arc<Mutex<Vec<(String, u64, bool)>>>);

impl ClipCommands for MockClip {
    fn activate(&self, id: u64) {
        self.0
            .lock()
            .expect("clip calls")
            .push(("activate".into(), id, false));
    }
    fn pin(&self, id: u64, on: bool) {
        self.0
            .lock()
            .expect("clip calls")
            .push(("pin".into(), id, on));
    }
    fn remove(&self, id: u64) {
        self.0
            .lock()
            .expect("clip calls")
            .push(("remove".into(), id, false));
    }
    fn clear(&self) {
        self.0
            .lock()
            .expect("clip calls")
            .push(("clear".into(), 0, false));
    }
}

/// A directory removed when the test ends, however it ends.
///
/// Hand-rolled rather than `tempfile`: contract §3.5 keeps shell's
/// dev-dependencies as they are, and this is ten lines.
pub struct TempDir(PathBuf);

impl TempDir {
    #[must_use]
    pub fn new(tag: &str) -> TempDir {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let n = COUNT.fetch_add(1, Ordering::SeqCst);
        let path =
            std::env::temp_dir().join(format!("icedtea-shell-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).expect("temp dir");
        TempDir(path)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One compositor, one running panel, one pointer and one screencopy client.
pub struct Panel {
    pub wm: MockWm,
    pub clip: MockClip,
    tx: InboxSender<Msg>,
    pointer: VirtualPointerClient,
    screencopy: ScreencopyClient,
    report: PathBuf,
    /// The open popover's own probe lines, truncate-written beside `report`
    /// (see `panel::write_popup_report` for why it is a separate, current-state
    /// file rather than an append to `report`).
    popup_report: PathBuf,
    output: (u32, u32),
    background: (u8, u8, u8),
    _dir: TempDir,
    _panel: JoinHandle<()>,
    // Dropped last: killing the compositor first would take the panel's
    // connection out from under it mid-assertion.
    _compositor: Compositor,
}

impl Panel {
    /// Boot a compositor and run the real panel against it under `theme`.
    ///
    /// # Panics
    ///
    /// If the compositor advertises no output or screencopy, or if the panel
    /// never publishes its first report.
    #[must_use]
    pub fn spawn(theme: Theme) -> Panel {
        let compositor = Compositor::spawn();
        // The panel connects by the compositor's *absolute* socket path, not
        // by a bare `$WAYLAND_DISPLAY` name resolved against the process
        // environment. That path already lives under the harness's private
        // runtime dir, so the panel joins this compositor and never the
        // developer's live session — the effect the settings harness gets by
        // setting `XDG_RUNTIME_DIR` on the child it spawns, achieved here
        // without touching the process-global environment the way an
        // `unsafe set_var` would (P5 Task 12 reconciliation).
        let socket = compositor.socket.clone();
        let socket_path = compositor.socket_path();
        let (ow, oh) = compositor.output_size();
        let output = (ow as u32, oh as u32);

        let dir = TempDir::new(theme.name());
        let report = dir.path().join("report");
        let popup_report = report.with_extension("popups");

        let wm = MockWm::default();
        let clip = MockClip::default();
        let (handshake_tx, handshake_rx) = mpsc::channel::<InboxSender<Msg>>();

        let thread_wm = wm.clone();
        let thread_clip = clip.clone();
        let thread_report = report.clone();
        let panel_thread = std::thread::spawn(move || {
            let window = match icedtea_ui::window::Window::open_at_path(
                &socket_path,
                panel::spec(),
                style::sheet_for(theme),
                FontDatabase::new(),
            ) {
                Ok(window) => window,
                Err(err) => panic!("the panel could not open its layer surface: {err}"),
            };
            let (inbox, tx) = Inbox::<Msg>::new().expect("inbox");
            let width_tx = tx.clone();
            if handshake_tx.send(tx).is_err() {
                return;
            }
            let model = PanelModel::new(
                Rc::new(thread_wm) as Rc<dyn CompositorCommands>,
                Rc::new(thread_clip) as Rc<dyn ClipCommands>,
            );
            // The exact `App::on_frame` hook the binary installs, from the one
            // shared factory (`panel::frame_hook`, M5 finding #3) so this path
            // cannot drift from `main.rs`'s. The window's own `probe`/`alloc`
            // report is written by `App::run` (P5 Task 11 moved it there from a
            // panel-owned hook), so the harness still names its path with
            // `with_probe_report`; the hook only publishes the anchor rect, the
            // popover's own probe lines, and the surface-width message.
            let clip_rect = model.clip_rect.clone();
            let open_popover = model.open_popover_cell.clone();
            let bar_width = model.bar_width;
            // The popover's own probe lines go beside the probe report (which
            // `App::run` owns, append-only, for the window's own lines).
            let popup_report = thread_report.with_extension("popups");
            let _ = App::new(model, panel::update, panel::view)
                .with_inbox(inbox)
                .on_popup(|ev| match ev {
                    PopupEvent::Opened(key) => Some(Msg::PopoverOpened(key)),
                    PopupEvent::Dismissed(key) => Some(Msg::PopoverDismissed(key)),
                    _ => None,
                })
                .on_frame(panel::frame_hook(
                    clip_rect,
                    open_popover,
                    Some(popup_report),
                    width_tx,
                    bar_width,
                ))
                .with_probe_report(thread_report)
                .run(window);
        });

        let tx = handshake_rx
            .recv_timeout(TIMEOUT)
            .expect("the panel never opened its window");
        let screencopy = ScreencopyClient::spawn(&socket);
        let pointer = VirtualPointerClient::spawn(&socket);

        let mut panel = Panel {
            wm,
            clip,
            tx,
            pointer,
            screencopy,
            report,
            popup_report,
            output,
            // Replaced below, before this constructor returns: `background`
            // is never read with this placeholder still in it.
            background: (0, 0, 0),
            _dir: dir,
            _panel: panel_thread,
            _compositor: compositor,
        };
        let _ = panel.wait_for("alloc bar ");

        // The panel's own chrome colour, sampled *inside* the layer surface —
        // never a point outside it. `output.0 - 3, output.1 - 3` (the bottom
        // right corner) used to stand in for "background", but the bar only
        // occupies the output's top `BAR_HEIGHT` pixels: that corner is
        // desktop wallpaper, not the panel's own `#bar` background-color, so
        // a widget that painted nothing but inherited the bar's own
        // background would still read as "painted something" by differing
        // from the wallpaper underneath it. This is the exact lesson
        // `settings/tests/support/displays.rs::background` already learned,
        // applied here to a bar instead of a page.
        //
        // Reconciliation (P5 Task 16): a point just inside `#windows`'s own
        // right edge (this constructor's first attempt) is not safe — live
        // capture showed `#bar`'s three children (`#workspaces`, `#windows`,
        // `#clip`) packed as one horizontally-*centred* cluster rather than
        // spread taskbar-style across the bar (`#windows`'s own
        // `.hexpand(true)` does not reach taffy here, the same per-child
        // `ChildLayout` gap `style.css`'s `#bar` rule already documents one
        // level up), so a point derived from `#windows`'s edge lands right
        // next to -- sometimes inside -- `#clip`, sampling *its* background
        // instead of the bar's plain chrome. `#bar` itself does span the
        // whole output (`the_bar_spans_the_output_and_fits_its_surface`), and
        // that centred cluster is far narrower than a real output, so a
        // point pinned to the bar's own far-left edge stays outside it
        // regardless of how many buttons the cluster ever grows to hold.
        let (bx, by, _, bh) = panel.allocation("bar");
        let frame = panel.capture();
        let sample_x = (bx + 10).max(0) as u32;
        let sample_y = (by + bh / 2) as u32;
        panel.background =
            pixel(&frame, sample_x, sample_y).expect("the background probe is inside the frame");
        panel
    }

    /// Push a message onto the panel's inbox — what a D-Bus worker does.
    pub fn send(&self, msg: Msg) {
        self.tx.send(msg).expect("the panel is still running");
    }

    #[must_use]
    pub fn wm_calls(&self) -> Vec<(String, u32)> {
        self.wm.0.lock().expect("wm calls").clone()
    }

    #[must_use]
    pub fn clip_calls(&self) -> Vec<(String, u64, bool)> {
        self.clip.0.lock().expect("clip calls").clone()
    }

    /// Every line of the panel's current report, followed by the open
    /// popover's `popup <label> <x> <y>` lines.
    ///
    /// The window's `probe`/`alloc` lines are append-only across frames (see
    /// `wait_for`); the popover's lines are the *current* state of a separate,
    /// truncate-written file (`panel::write_popup_report`), so a dismissal or
    /// a replacement shows up as those `popup ` lines simply being gone — the
    /// current-state semantics the gate reads with a whole-report scan.
    #[must_use]
    pub fn report(&self) -> Vec<String> {
        let mut lines: Vec<String> = std::fs::read_to_string(&self.report)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect();
        lines.extend(self.popup_report());
        lines
    }

    /// The open popover's current probe lines, or empty when it is closed.
    #[must_use]
    pub fn popup_report(&self) -> Vec<String> {
        std::fs::read_to_string(&self.popup_report)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// The output-space centre of a probe point inside the open popover.
    ///
    /// The popover's own lines already carry output coordinates
    /// (`panel::popup_report_lines` adds the compositor-assigned popup
    /// position), so a gate clicks a row exactly as it clicks a bar button.
    ///
    /// # Panics
    ///
    /// If the popover exposes no such point within [`TIMEOUT`].
    #[must_use]
    pub fn popup_point(&self, label: &str) -> (i32, i32) {
        let line = self.wait_for(&format!("popup {label} "));
        let fields: Vec<&str> = line.split_whitespace().collect();
        (
            fields[2].parse().expect("popup x"),
            fields[3].parse().expect("popup y"),
        )
    }

    /// Poll the report until a line starting with `prefix` appears, and return
    /// the **most recent** such line.
    ///
    /// The report is append-only across frames (`ui/src/view/app.rs` writes a
    /// fresh `frame N` block each time the geometry changes), so a coordinate
    /// evolves down the file as the surface is configured and clicked. The
    /// last matching line is the one that describes what is on screen now —
    /// the same "latest wins" `settings/tests/support/mod.rs` reads with
    /// `.rev().find`.
    ///
    /// # Panics
    ///
    /// If no such line appears within [`TIMEOUT`]; the message lists what the
    /// report does hold, which is what makes a renamed id a readable failure.
    pub fn wait_for(&self, prefix: &str) -> String {
        let deadline = Instant::now() + TIMEOUT;
        while Instant::now() < deadline {
            if let Some(line) = self
                .report()
                .into_iter()
                .rev()
                .find(|l| l.starts_with(prefix))
            {
                return line;
            }
            std::thread::sleep(POLL);
        }
        panic!(
            "no report line starting with {prefix:?} within {TIMEOUT:?}; report holds {:?}",
            self.report()
        );
    }

    /// Poll the report until `want` accepts it.
    ///
    /// # Panics
    ///
    /// If it does not within [`TIMEOUT`].
    pub fn wait_until(&self, want: impl Fn(&[String]) -> bool, what: &str) {
        let deadline = Instant::now() + TIMEOUT;
        while Instant::now() < deadline {
            if want(&self.report()) {
                return;
            }
            std::thread::sleep(POLL);
        }
        panic!(
            "{what} did not happen within {TIMEOUT:?}; report holds {:?}",
            self.report()
        );
    }

    /// Poll `current()` until `want` accepts it, panicking with the last
    /// observed state if [`TIMEOUT`] elapses first.
    ///
    /// The shared body behind [`Panel::wait_for_calls`] and
    /// [`Panel::wait_for_clip_calls`], which differ only in the state they read.
    ///
    /// # Panics
    ///
    /// If `want` never accepts within [`TIMEOUT`].
    fn wait_for_state<T: std::fmt::Debug>(
        &self,
        current: impl Fn() -> T,
        want: impl Fn(&T) -> bool,
        what: &str,
    ) {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            let state = current();
            if want(&state) {
                return;
            }
            if Instant::now() >= deadline {
                panic!("{what} did not happen within {TIMEOUT:?}; state was {state:?}");
            }
            std::thread::sleep(POLL);
        }
    }

    /// Poll the recorded compositor commands until `want` accepts them.
    ///
    /// # Panics
    ///
    /// If it does not within [`TIMEOUT`].
    pub fn wait_for_calls(&self, want: impl Fn(&[(String, u32)]) -> bool, what: &str) {
        self.wait_for_state(|| self.wm_calls(), |c| want(c.as_slice()), what);
    }

    /// The `ClipCommands` counterpart.
    ///
    /// # Panics
    ///
    /// If it does not within [`TIMEOUT`].
    pub fn wait_for_clip_calls(&self, want: impl Fn(&[(String, u64, bool)]) -> bool, what: &str) {
        self.wait_for_state(|| self.clip_calls(), |c| want(c.as_slice()), what);
    }

    /// The output-space centre of the probe point `label`.
    ///
    /// The layer surface is anchored left, right and top with zero margins, so
    /// its own (0, 0) is the output's: a window coordinate *is* an output
    /// coordinate here, and no offset arithmetic is needed.
    ///
    /// # Panics
    ///
    /// If the panel exposes no such probe point.
    #[must_use]
    pub fn point(&self, label: &str) -> (i32, i32) {
        let line = self.wait_for(&format!("probe {label} "));
        let fields: Vec<&str> = line.split_whitespace().collect();
        (
            fields[2].parse().expect("probe x"),
            fields[3].parse().expect("probe y"),
        )
    }

    /// `id`'s border box in output coordinates, as `(x, y, width, height)`.
    ///
    /// # Panics
    ///
    /// If the panel reports no allocation for `id`.
    #[must_use]
    pub fn allocation(&self, id: &str) -> (i32, i32, i32, i32) {
        let line = self.wait_for(&format!("alloc {id} "));
        let f: Vec<&str> = line.split_whitespace().collect();
        (
            parse_coord(f[2]),
            parse_coord(f[3]),
            parse_coord(f[4]),
            parse_coord(f[5]),
        )
    }

    /// The ids the report holds for direct children of `container`, in
    /// left-to-right order — the replacement for `shell_gtk.rs`'s `labels()`
    /// walk of a GTK widget tree.
    ///
    /// Membership is by prefix and geometry: `window_*`/`ws_*`/`history_*`
    /// ids are unique per entity, and an id whose box sits inside
    /// `container`'s box is one of its children. Ordered by `x` for a
    /// horizontal container and by `y` for a vertical one, which is decided by
    /// which of the container's dimensions is the larger.
    #[must_use]
    pub fn labels_under(&self, container: &str, prefix: &str) -> Vec<String> {
        let (cx, cy, cw, ch) = self.allocation(container);
        // Latest wins per id: the append-only report carries an id once per
        // frame it changed in, so a plain scan would list it as many times as
        // it moved. The last line for an id is the box it holds now.
        let mut latest: std::collections::HashMap<String, (i32, i32)> =
            std::collections::HashMap::new();
        for line in self.report() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 6 || f[0] != "alloc" || !f[1].starts_with(prefix) {
                continue;
            }
            latest.insert(f[1].to_string(), (parse_coord(f[2]), parse_coord(f[3])));
        }
        let mut found: Vec<(i32, i32, String)> = latest
            .into_iter()
            .filter(|(_, (x, y))| *x >= cx && *x < cx + cw && *y >= cy && *y < cy + ch)
            .map(|(id, (x, y))| (x, y, id))
            .collect();
        if cw >= ch {
            found.sort_by_key(|(x, _, _)| *x);
        } else {
            found.sort_by_key(|(_, y, _)| *y);
        }
        found.into_iter().map(|(_, _, id)| id).collect()
    }

    /// Left-click at `(x, y)`.
    pub fn click(&mut self, x: i32, y: i32) {
        self.click_button(x, y, icedtea_ui::window::pointer::BTN_LEFT);
    }

    /// Press and release `button` at `(x, y)`.
    ///
    /// The settle between the motion and the button repeats
    /// `ui/tests/support/mod.rs`'s `Driver::move_to` rationale verbatim: the
    /// compositor's assignment of pointer focus to the surface under the
    /// cursor is not synchronous with the `motion_absolute` that triggers it,
    /// and `wlr_seat_pointer_notify_button` drops a button with no focused
    /// surface silently.
    pub fn click_button(&mut self, x: i32, y: i32, button: u32) {
        self.pointer
            .motion_absolute(f64::from(x), f64::from(y), self.output.0, self.output.1);
        self.pointer.frame();
        self.pointer.pump();
        for _ in 0..8 {
            std::thread::sleep(Duration::from_millis(25));
            self.pointer.pump();
        }
        self.pointer.button(button, true);
        self.pointer.frame();
        self.pointer.pump();
        std::thread::sleep(Duration::from_millis(25));
        self.pointer.button(button, false);
        self.pointer.frame();
        self.pointer.pump();
        std::thread::sleep(Duration::from_millis(25));
        self.pointer.pump();
    }

    /// One screencopy frame of the whole output.
    pub fn capture(&mut self) -> CapturedFrame {
        self.screencopy.capture()
    }

    /// How many consecutive identical captures [`Panel::capture_settled`]
    /// requires before it trusts the output has actually stopped changing.
    ///
    /// Live capture during Task 16 found a genuine plateau: two consecutive
    /// captures already read back byte-identical a moment after the report
    /// converged, then the very next capture changed anyway (`#workspaces`'s
    /// and `#windows`' buttons had not painted yet). A single repeat is not
    /// proof of settling, only of two frames landing inside the same
    /// composited-output tick; `STABLE_STREAK` at [`POLL`]'s cadence held
    /// for the remainder of a multi-second observation window in every case
    /// this module hit, well past where the true plateau above broke.
    const STABLE_STREAK: u32 = 10;

    /// A screencopy frame taken only once the compositor's own output has
    /// stopped changing — [`STABLE_STREAK`] consecutive captures with
    /// identical bytes, not merely two.
    ///
    /// `wait_until`/`wait_for` confirm the panel's *report* converged (its
    /// `on_frame` hook ran and published the laid-out geometry), but
    /// publishing that report and the corresponding surface commit actually
    /// reaching a composited output frame are two different events with a
    /// gap between them (observed live: several hundred milliseconds after
    /// the report already read as converged, a screencopy capture still
    /// showed the *previous* layout's buttons in the *previous* positions —
    /// and, more surprisingly, that stale frame could itself repeat
    /// byte-for-byte across a couple of captures before the real one
    /// landed, defeating a naive "two in a row" check). A rest-state gate
    /// that samples pixels — unlike every other gate in this file, which
    /// only ever reads the report or clicks a point — has to wait out that
    /// gap itself, the same way
    /// `a_focused_window_button_looks_different_from_an_unfocused_one`
    /// already resamples in a loop rather than trusting a single capture.
    ///
    /// # Panics
    ///
    /// If the output never holds [`STABLE_STREAK`] identical captures in a
    /// row within [`TIMEOUT`].
    pub fn capture_settled(&mut self) -> CapturedFrame {
        let deadline = Instant::now() + TIMEOUT;
        let mut previous = self.capture();
        let mut streak = 1u32;
        while Instant::now() < deadline {
            std::thread::sleep(POLL);
            let next = self.capture();
            if next.bytes == previous.bytes {
                streak += 1;
                if streak >= Self::STABLE_STREAK {
                    return next;
                }
            } else {
                streak = 1;
            }
            previous = next;
        }
        panic!("the output never settled within {TIMEOUT:?}");
    }

    #[must_use]
    pub fn background(&self) -> (u8, u8, u8) {
        self.background
    }

    #[must_use]
    pub fn output(&self) -> (u32, u32) {
        self.output
    }
}

/// Parse one `alloc`/`probe` coordinate field.
///
/// The report writes each box as a laid-out `f32` (`ui/src/view/app.rs`'s
/// `probe_report_lines`), so a whole number arrives as `28` and a fractional
/// one as `27.5`; both must round to the same integer output pixel a pointer
/// can be aimed at. A bare `str::parse::<i32>` would reject the second form.
#[must_use]
fn parse_coord(field: &str) -> i32 {
    if let Ok(n) = field.parse::<i32>() {
        return n;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a laid-out coordinate is within i32 by construction"
    )]
    field
        .parse::<f32>()
        .map(|v| v.round() as i32)
        .unwrap_or_else(|_| panic!("unparseable report coordinate {field:?}"))
}

/// The RGB of `(x, y)` in a captured frame, or `None` outside it.
#[must_use]
pub fn pixel(frame: &CapturedFrame, x: u32, y: u32) -> Option<(u8, u8, u8)> {
    icedtea_ui::shm::pixel_rgb(frame.format, &frame.bytes, frame.stride, x, y)
}

/// How far apart two channel bytes may be and still count as the same colour
/// on a screencopy capture — `ui/tests/support/mod.rs`'s constant, for the
/// same format-conversion reason.
pub const SCREENCOPY_TOLERANCE: u8 = 12;

/// Whether `a` and `b` are the same colour within [`SCREENCOPY_TOLERANCE`].
#[must_use]
pub fn same(a: (u8, u8, u8), b: (u8, u8, u8)) -> bool {
    let close =
        |p: u8, q: u8| i32::from(p).abs_diff(i32::from(q)) <= u32::from(SCREENCOPY_TOLERANCE);
    close(a.0, b.0) && close(a.1, b.1) && close(a.2, b.2)
}

/// Whether anything inside `rect` differs from `background`.
#[must_use]
pub fn paints_something(
    frame: &CapturedFrame,
    rect: (i32, i32, i32, i32),
    background: (u8, u8, u8),
) -> bool {
    let (x, y, w, h) = rect;
    if w <= 0 || h <= 0 {
        return false;
    }
    (y..y + h).any(|py| {
        (x..x + w)
            .any(|px| pixel(frame, px as u32, py as u32).is_some_and(|got| !same(got, background)))
    })
}

// --- fixtures, verbatim from `tests/shell_gtk.rs` -------------------------

#[must_use]
pub fn win(id: u32, title: &str) -> WindowInfo {
    WindowInfo {
        id: WindowId(id),
        app_id: "app".into(),
        title: title.into(),
        pid: 0,
        workspace: 0,
        geometry: Rectangle {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        maximized: false,
        minimized: false,
        fullscreen: false,
        focused: false,
        attention: false,
    }
}

#[must_use]
pub fn snapshot(windows: Vec<WindowInfo>, workspaces: Vec<WorkspaceInfo>) -> Snapshot {
    Snapshot {
        seq: 1,
        windows,
        workspaces,
        active_workspace: 0,
        cursor_visible: false,
        cursor_pos: None,
        touch_active: false,
    }
}

#[must_use]
pub fn clip_entry(id: u64, preview: &str) -> ClipEntry {
    ClipEntry {
        id,
        kind: ClipKind::Text,
        preview: preview.into(),
        mime: "text/plain".into(),
        pinned: false,
        source_app: None,
    }
}
