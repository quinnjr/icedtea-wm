//! The `org.icedtea.Compositor` D-Bus service.
//!
//! Per the binding threading-model ruling from task 7: this runs on its own
//! dedicated thread and talks to the compositor's own loop only by message
//! passing -- never touching `State` directly. Two channels cross that
//! boundary:
//!
//! - `events_rx` (compositor -> this thread): every `contract::Event`
//!   produced by a `State`/`WindowManager` mutation, forwarded here as a
//!   D-Bus signal by a dedicated emitter thread spawned from
//!   [`spawn_service`]. This is the consumer the task-7 handoff comment in
//!   `main.rs` promised for the previously-undrained `_dbus_rx` channel.
//! - `cmd_tx` (this thread -> compositor): [`DbCommand`]s produced by
//!   incoming [`CompositorInterface`] method calls, applied to `State` by whatever
//!   drains this channel (see `State::handle_command`).
//!
//! The command channel is a `crossbeam_channel::Sender`, drained by the
//! compositor's own event loop once per turn. It used to be paired with a
//! dedicated event-source abstraction that woke the loop on a send; with
//! that library gone the loop polls instead, which is the same latency in
//! practice because it wakes on every input and frame event anyway.
//!
//! ## Deviations from the task-12 brief
//!
//! (Standing human ruling: the plan's stated invariants/intended semantics
//! govern over its verbatim sample code; deviations are documented here.)
//!
//! - **Bus name ownership.** The brief's Step-3 sample connects to the
//!   session bus and registers the interface object but never calls
//!   `request_name`, so nothing would actually own `org.icedtea.Compositor` --
//!   the brief's own Step-5 manual verification
//!   (`gdbus --dest org.icedtea.Compositor ...`) would fail with "name has no
//!   owner" as written. Added below, after the interface is registered
//!   (so the object already exists by the time anyone can observe the name
//!   becoming owned).
//! - **`CompositorInterface::conn` field dropped.** The brief's sample struct
//!   carries a `conn: Connection` field that no `#[interface]` method ever
//!   reads (every method just forwards onto `cmd_tx`); an unread field is a
//!   `dead_code` warning under this workspace's `-D warnings` gate. Nothing
//!   in this task's interface needs a connection handle on `self` --
//!   `emit_signal` runs from the separate emitter thread against its own
//!   connection clone.
//! - **Signal payload typing.** The brief's sample builds one `payload`
//!   local via a `match` whose arms produce differently-shaped tuples
//!   (`(WindowInfo,)`, `(u32,)`, `(u32, WindowUpdate)`, ...) bound to a
//!   single variable -- that doesn't type-check (a `match` expression must
//!   produce one concrete type). `emit_signal` is called directly inside
//!   each arm below instead, each against its own concretely-typed body.
//! - **`quit_signal`'s use, and the return type that carries it.** Present
//!   in the brief's "Produces" signature but absent from its Step-3 sample
//!   body. Used here to turn the emitter thread's blocking `recv()` into a
//!   polling `recv_timeout`, so the thread can notice the compositor
//!   shutting down and exit its loop between messages instead of being
//!   severed mid-`emit_signal` when the process exits. Setting the flag
//!   alone doesn't achieve that, though: `main.rs` sets it as its last
//!   statement, and if nothing then waits for the thread to actually act on
//!   it, `main` (and the process with it) can return before the next
//!   `recv_timeout` tick ever wakes up -- the flag would be set on a thread
//!   that's already gone. So `spawn_service` returns
//!   `(Connection, JoinHandle<()>)`, not just the brief's bare
//!   `Connection`, and `main.rs` joins that handle immediately after
//!   setting the flag; the join is bounded by the `recv_timeout` tick
//!   (200ms) it's waiting on, not by traffic on `events_rx`.

use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use icedtea_contract::{COMPOSITOR_BUS_NAME, COMPOSITOR_PATH, Event, SeqEvent, Snapshot, WindowId};
use zbus::blocking::Connection;
use zbus::interface;

/// Commands sent from the D-Bus interface thread to the compositor's main
/// loop. Applied to `State` by `State::handle_command`.
///
/// The `Test-only` variants are an internal harness channel, not wire API:
/// they are never exposed through `CompositorInterface` and remain subject
/// to additive change as tests need new oracles.
#[derive(Debug, Clone)]
pub enum DbCommand {
    Focus(WindowId),
    Close(WindowId),
    Minimize(WindowId, bool),
    Maximize(WindowId, bool),
    Fullscreen(WindowId, bool),
    SetWorkspace(u32),
    MoveToWorkspace(WindowId, u32),
    ReloadConfig,
    Quit,
    GetState(Sender<Snapshot>),
    /// Test-only: synthesize a touch-down at `(x, y)` for touch point `id`
    /// via `wlr::Runtime::inject_touch_down`, replying with the grab serial
    /// it mints (`None` if there is no seat or no surface under the point).
    /// Not reachable from `CompositorInterface` -- only the test harness sends
    /// this, directly onto `cmd_tx`, since injecting synthetic touch input
    /// makes no sense as a D-Bus-exposed production operation.
    InjectTouchDown {
        x: f64,
        y: f64,
        id: i32,
        time_msec: u32,
        reply: Sender<Option<u32>>,
    },
    /// Test-only: synthesize a touch-motion to `(x, y)` for touch point
    /// `id` via `wlr::Runtime::inject_touch_motion`. `reply` is purely a
    /// synchronization ack -- see `InjectTouchDown`'s doc -- so the harness
    /// call blocks until the injection has actually run on the compositor
    /// thread instead of racing ahead of it.
    InjectTouchMotion {
        x: f64,
        y: f64,
        id: i32,
        time_msec: u32,
        reply: Sender<()>,
    },
    /// Test-only: synthesize a touch-up for touch point `id` via
    /// `wlr::Runtime::inject_touch_up`. See `InjectTouchDown`'s doc.
    InjectTouchUp {
        id: i32,
        time_msec: u32,
        reply: Sender<()>,
    },
    /// Test-only: drive the M7 touch-cancel consumer path. Headless has no
    /// wire producer for cancels (they come from hardware), so this calls
    /// the real `SeatHandler::touch_cancelled` on the loop thread; the
    /// wire cancel to the client (the crate's token-consuming
    /// `send_cancel`) stays hardware-driven. See `InjectTouchDown`'s doc
    /// for why this is not a `CompositorInterface` operation.
    InjectTouchCancel {
        reply: Sender<()>,
    },
    /// Test-only: drive the M7 gesture consumer path. Headless has no
    /// gesture hardware, so this calls the real
    /// `SeatHandler::gesture_began`/`gesture_ended` on the loop thread with
    /// an id that names no live pointer (harmless by the handlers'
    /// contract). See `InjectTouchDown`'s doc.
    InjectGesture {
        began: bool,
        reply: Sender<()>,
    },
    /// Test-only: drive the M7 switch consumer path with a
    /// hardware-decoded `(type, on)` pair no headless device can produce.
    /// Calls the same `apply_switch_toggle` the live `switch_toggled`
    /// handler calls after resolving the pair from the runtime aggregate.
    /// See `InjectTouchDown`'s doc.
    InjectSwitchToggle {
        switch_type: wlr::SwitchType,
        on: bool,
        reply: Sender<()>,
    },
    /// Test-only: read the drag icon's current scene layout position via
    /// `wlr::Runtime::drag_icon_position`, replying with `None` if no drag
    /// with a visible icon is in progress. Not reachable from
    /// `CompositorInterface` -- only the test harness sends this, same reasoning
    /// as `InjectTouchDown`.
    DragIconPosition {
        reply: Sender<Option<(i32, i32)>>,
    },
    /// Test-only: read `State::popups_dismissed` -- how many popups this
    /// compositor has itself closed through `wlr::Runtime::dismiss_popup`
    /// since boot. Not reachable from `CompositorInterface` -- only the test
    /// harness sends this, same reasoning as `DragIconPosition`.
    ///
    /// The *count* rather than the order is what the harness needs, because
    /// the order is not observable from a client: wlroots frees a popup's
    /// children before the popup itself, so the `xdg_popup.popup_done` events
    /// reach the wire deepest-first however the caller iterated. What a
    /// shallow-first caller loses is rows -- it finds the deeper popups
    /// already swept and under-counts. See
    /// `compositor/tests/popups.rs`'s
    /// `destroying_a_parent_destroys_its_popup_chain_without_a_double_free`.
    PopupsDismissed {
        reply: Sender<usize>,
    },
    /// Test-only: read `wlr::Runtime::is_session_locked` via `wayland`'s
    /// runtime handle. Not reachable from `CompositorInterface` -- only the test
    /// harness sends this, same reasoning as `DragIconPosition`.
    SessionLocked {
        reply: Sender<bool>,
    },
    /// Test-only: read `wlr::Runtime::input_method_active` via `wayland`'s
    /// runtime handle. Not reachable from `CompositorInterface` -- only the
    /// test harness sends this, same reasoning as `SessionLocked`.
    InputMethodActive {
        reply: Sender<bool>,
    },
    /// Test-only: read `wlr::Runtime::cursor_position` via `wayland`'s
    /// runtime handle. Not reachable from `CompositorInterface` -- only the test
    /// harness sends this, same reasoning as `SessionLocked`.
    CursorPosition {
        reply: Sender<(f64, f64)>,
    },
    /// Test-only: the scene position of the currently-placed input-method
    /// candidate popup, as the compositor positioned it (`set_node_position`
    /// on the node `add_input_popup_in_band` returned). `None` when no popup is
    /// placed. Read from `State`'s own record of the placed node (the crate
    /// exposes no by-id popup-position accessor), then resolved through
    /// `wlr::Runtime::node_position`. Not reachable from `CompositorInterface`
    /// -- only the test harness sends this, same reasoning as `CursorPosition`.
    InputPopupPosition {
        reply: Sender<Option<(i32, i32)>>,
    },
    /// Test-only: the scene node of the currently-placed input-method
    /// candidate popup (the `NodeId` `add_input_popup_in_band` returned, as
    /// `State` recorded it). `None` when no popup is placed. Paired with
    /// `SceneNodePosition`: a test captures the node while placed, tears the
    /// popup down, and asserts the node itself is gone — the tripwire that the
    /// crate destroys the popup's scene node on popup destroy (if the destroy
    /// call were deleted, `InputPopupPosition` would still return to `None`
    /// via bookkeeping while the node leaked). Not reachable from
    /// `CompositorInterface` -- only the test harness sends this.
    InputPopupNode {
        reply: Sender<Option<wlr::NodeId>>,
    },
    /// Test-only: the scene position of an arbitrary node by id, resolved
    /// through `wlr::Runtime::node_position`. `None` for an unknown or stale
    /// (destroyed) id. The second half of the popup-teardown tripwire: after
    /// capturing `InputPopupNode` and destroying the IME, the node must read
    /// `None`. Not reachable from `CompositorInterface`.
    SceneNodePosition {
        node: wlr::NodeId,
        reply: Sender<Option<(i32, i32)>>,
    },
    /// Test-only: the scene position of the preedit overlay, or `None` when
    /// no composing text is shown. The position is the compositor's own
    /// placement record (the crate exposes no buffer-position accessor), so
    /// the visibility half — `Some` while composing, `None` after
    /// commit-string/deactivate/focus-away — is the load-bearing assertion;
    /// the coordinates pin the caret-anchored placement. Not reachable from
    /// `CompositorInterface` -- only the test harness sends this, same
    /// reasoning as `CursorPosition`.
    PreeditOverlay {
        reply: Sender<Option<(i32, i32)>>,
    },
    /// Test-only: read `wlr::Runtime::cursor_shape` -- the named shape
    /// currently in force as the crate itself records it, `None` rendered as
    /// `"Default"` -- as its `Debug` name. Not reachable from `CompositorInterface` --
    /// only the test harness sends this, same reasoning as `SessionLocked`.
    CursorShape {
        reply: Sender<String>,
    },
    /// Test-only: the primary output's real geometry, straight off
    /// `State::outputs`. `None` before any output exists.
    ///
    /// Review finding F11: tests that need the output's extent (to address a
    /// `zwlr_virtual_pointer_v1.motion_absolute` coordinate, or to pick a
    /// point on bare desktop) used to discover it by maximizing a window and
    /// reading its geometry back -- which is the snap-gap-inset *usable*
    /// rect, not the output, and so was systematically short by the gap on
    /// each edge. Not reachable from `CompositorInterface` -- only the test
    /// harness sends this, same reasoning as `CursorShape`.
    OutputSize {
        reply: Sender<Option<icedtea_contract::Rectangle>>,
    },
    /// Test-only: read the `DISPLAY` name (`:N`) Xwayland advertises, via
    /// `wlr::Runtime::xwayland_display_name`. `None` when no Xwayland was
    /// created (the `Xwayland` binary is absent), so the X11 end-to-end test
    /// can skip cleanly. Available as soon as the manager reserves its display
    /// socket -- before the lazy `Xwayland` start -- which is exactly what lets
    /// the test read `DISPLAY`, connect an X11 client, and *trigger* that lazy
    /// start. Not reachable from `CompositorInterface` -- only the test harness sends
    /// this, same reasoning as `SessionLocked`.
    XwaylandDisplay {
        reply: Sender<Option<String>>,
    },
    /// Test-only: report whether `xwayland_ready` has fired — i.e. the lazy
    /// `Xwayland` has actually started and the crate has wired its seat (arming
    /// the clipboard/primary/DND bridge). Distinct from `XwaylandDisplay`, which
    /// answers as soon as the display socket is *reserved* (before the lazy
    /// start): a selection/DND test connects using `XwaylandDisplay`, then waits
    /// on this for the bridge to be live. Replaces the old
    /// republish-`DISPLAY`-on-ready barrier, which relied on a `set_var` from
    /// inside `run_all` (review finding #5). Not reachable from `CompositorInterface` --
    /// only the test harness sends this, same reasoning as `SessionLocked`.
    XwaylandReady {
        reply: Sender<bool>,
    },
    /// Test-only: probe every mapped override-redirect (OR) X11 pop-up the
    /// compositor is tracking in its M3 side-table, reading each one's *real*
    /// scene state — node position, whether it is parented in the band above
    /// managed toplevels, and whether it holds the seat keyboard — via the
    /// `wlr::Runtime` xwayland scene accessors. Lets the OR end-to-end test
    /// assert placement/stacking/focus without the OR surface ever entering the
    /// `Window` model (which is the whole point of the OR path). Not reachable
    /// from `CompositorInterface` -- only the test harness sends this, same reasoning as
    /// `SessionLocked`.
    XwaylandOverrideRedirect {
        reply: Sender<Vec<OverrideRedirectProbe>>,
    },
    /// Test-only: record the primary output's scale. The reply is `true` only
    /// when an output actually existed to record it on — `spawn` returns at the
    /// boot handshake, *before* `run_all` creates the headless output, so a scale
    /// set too early would otherwise be silently dropped yet still acked
    /// "recorded" (review finding #11). The harness helper polls on this `bool`
    /// until the output exists, making the ordering deterministic instead of a
    /// flaky red.
    SetOutputScaleForTest {
        scale: f64,
        reply: Sender<bool>,
    },
}

/// One mapped override-redirect X11 pop-up, as the test-only
/// [`DbCommand::XwaylandOverrideRedirect`] probe reports it — read straight off
/// the live scene, not the compositor's own bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverrideRedirectProbe {
    /// The pop-up scene node's position, in layout coordinates — the proof it
    /// landed at its client-requested absolute coordinates.
    pub position: (i32, i32),
    /// Whether the node is parented in `Band::Top`, i.e. the band **above**
    /// every managed toplevel (`Band::Toplevel`) — the proof it stacks over
    /// managed windows.
    pub above_toplevel: bool,
    /// Whether the seat keyboard is currently pointed at this pop-up — the
    /// proof a focus-taking menu is navigable.
    pub keyboard_focused: bool,
}

/// Map a `contract::Event` to its D-Bus signal name, so the emitter thread
/// can dispatch. Pure and unit-tested.
pub fn event_signal_name(event: &Event) -> &'static str {
    match event {
        Event::WindowOpened(_) => "WindowOpened",
        Event::WindowClosed(_) => "WindowClosed",
        Event::WindowUpdated { .. } => "WindowUpdated",
        Event::WorkspaceSet { .. } => "WorkspaceSet",
        Event::WorkspaceList(_) => "WorkspaceList",
        Event::AltTabState(_) => "AltTabState",
        Event::ConfigReloaded(_) => "ConfigReloaded",
        Event::GestureBegan => "GestureBegan",
        Event::GestureEnded => "GestureEnded",
        Event::SwitchToggled { .. } => "SwitchToggled",
    }
}

/// The registered `org.icedtea.Compositor` interface object. Every method just
/// forwards a [`DbCommand`] onto the compositor's main loop; none of them
/// mutate compositor state directly (see this module's doc for why).
pub struct CompositorInterface {
    cmd_tx: crossbeam_channel::Sender<DbCommand>,
    /// Write half of the loop's D-Bus command wake pipe
    /// (`backend::wake_source`). `send` nudges it after every command so a
    /// compositor blocked in `dispatch(-1)` (idle: no damage, no input)
    /// wakes to drain `cmd_tx`'s receiver instead of waiting for whatever
    /// unrelated event happens along next.
    wake: UnixStream,
}

impl CompositorInterface {
    fn send(&self, cmd: DbCommand) {
        let _ = self.cmd_tx.send(cmd);
        crate::backend::wake(&self.wake);
    }
}

#[interface(name = "org.icedtea.Compositor")]
impl CompositorInterface {
    fn focus_window(&self, id: u32) {
        self.send(DbCommand::Focus(WindowId(id)));
    }
    fn close_window(&self, id: u32) {
        self.send(DbCommand::Close(WindowId(id)));
    }
    fn minimize_window(&self, id: u32, toggle: bool) {
        self.send(DbCommand::Minimize(WindowId(id), toggle));
    }
    fn maximize_window(&self, id: u32, toggle: bool) {
        self.send(DbCommand::Maximize(WindowId(id), toggle));
    }
    fn fullscreen_window(&self, id: u32, toggle: bool) {
        self.send(DbCommand::Fullscreen(WindowId(id), toggle));
    }
    fn set_workspace(&self, id: u32) {
        self.send(DbCommand::SetWorkspace(id));
    }
    fn move_window_to_workspace(&self, id: u32, workspace: u32) {
        self.send(DbCommand::MoveToWorkspace(WindowId(id), workspace));
    }
    fn get_state(&self) -> Snapshot {
        // Synchronous round-trip: ask the compositor for a snapshot. The
        // main loop answers this the moment it drains the `GetState`
        // command (see `State::handle_command`), so this blocks the
        // zbus dispatch for this connection only as long as one loop
        // iteration takes -- by design, per the brief.
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::GetState(reply_tx));
        reply_rx.recv().unwrap_or_else(|_| Snapshot {
            seq: 0,
            windows: vec![],
            workspaces: vec![],
            active_workspace: 0,
            ime_active: false,
            keyboard_layout: None,
            shortcuts_inhibited: false,
            cursor_visible: false,
            cursor_pos: None,
            touch_active: false,
        })
    }
    /// The wire-contract revision this compositor speaks
    /// ([`icedtea_contract::COMPOSITOR_CONTRACT_VERSION`]), so a client can
    /// name a mismatch instead of only discovering one as a
    /// `SignatureMismatch` on its first typed call (review finding F7).
    ///
    /// A read-only zbus *property*: adding it leaves every method signature
    /// on this interface exactly as it was, so it cannot itself be the
    /// incompatibility it exists to report.
    #[zbus(property)]
    fn version(&self) -> u32 {
        icedtea_contract::COMPOSITOR_CONTRACT_VERSION
    }
    fn reload_config(&self) {
        self.send(DbCommand::ReloadConfig);
    }
    fn quit(&self) {
        self.send(DbCommand::Quit);
    }
}

/// Spawn the D-Bus service: connects to the session bus, registers
/// [`CompositorInterface`] at [`COMPOSITOR_PATH`], claims [`COMPOSITOR_BUS_NAME`], and starts a
/// dedicated emitter thread that turns every `contract::Event` received on
/// `events_rx` into a D-Bus signal. Returns the shared connection (kept
/// alive by the caller for as long as the service should stay registered)
/// and the emitter thread's `JoinHandle`, so the caller can wait for it to
/// actually observe `quit_signal` on shutdown (see this module's doc for
/// why a bare `Connection` return, as the brief's "Produces" line has it,
/// isn't enough for that).
///
/// `cmd_wake` is the write half of a `backend::wake_source` registered by
/// the caller against the same `Runtime` the loop runs on -- see
/// `CompositorInterface::send`'s doc for why a command needs one at all.
pub fn spawn_service(
    events_rx: Receiver<SeqEvent>,
    cmd_tx: crossbeam_channel::Sender<DbCommand>,
    quit_signal: Arc<AtomicBool>,
    cmd_wake: UnixStream,
) -> (Connection, std::thread::JoinHandle<()>) {
    let conn = Connection::session().expect("session bus available");
    let iface = CompositorInterface {
        cmd_tx,
        wake: cmd_wake,
    };
    conn.object_server()
        .at(COMPOSITOR_PATH, iface)
        .expect("register org.icedtea.Compositor interface");
    conn.request_name(COMPOSITOR_BUS_NAME).unwrap_or_else(|err| {
        panic!(
            "failed to acquire the {COMPOSITOR_BUS_NAME} bus name -- is another icedtea-compositor \
             instance (or a stale connection holding the name) already running? ({err})"
        )
    });

    let emitter_conn = conn.clone();
    let handle = std::thread::spawn(move || {
        loop {
            if quit_signal.load(Ordering::Relaxed) {
                break;
            }
            let SeqEvent { seq, event } = match events_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(event) => event,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            let name = event_signal_name(&event);
            let dest: Option<&str> = None;
            // Review finding I2: `seq` is the first argument of every
            // signal, so a subscriber can drop signals already folded into
            // the `GetState()` snapshot it started from (`seq <=
            // snapshot.seq`) and detect a gap (`seq > last_seen + 1`) that
            // means it must re-sync. Recorded signatures (kept in step with
            // `contract`'s own `wire_signatures_are_locked` test -- the
            // `attention` bit added the fifth `b` to `WindowInfo` and a
            // SIXTH `ab` to `WindowUpdate` -- whose bools run
            // maximized/minimized/fullscreen/focused/mapped/attention --
            // which is what `COMPOSITOR_CONTRACT_VERSION` 2 names.
            // Version 3 appended `Snapshot.ime_active` to `GetState`'s
            // reply; version 4 appends `Snapshot.keyboard_layout` /
            // `Snapshot.shortcuts_inhibited` (signature `...ubasb`),
            // which no signal below carries):
            //   WindowOpened   t(ussuu(iiii)bbbbb)
            //   WindowClosed   tu
            //   WindowUpdated  tu(asa(iiii)auabababababab)
            //   WorkspaceSet   tub
            //   WorkspaceList  ta(us)
            //   AltTabState    t(baut)
            //   ConfigReloaded t(siii(sss)as)
            //   GestureBegan   t
            //   GestureEnded   t
            //   SwitchToggled  (tb)
            // (M7: the gesture signals carry no payload beyond `seq` -- the
            // phase is the member name itself, since the full-fidelity
            // gesture data already reached clients through the crate's token
            // path and never belonged on this feed. A bare `seq` rather
            // than a 1-tuple keeps the body an ordinary `u64`.)
            let result = match &event {
                Event::WindowOpened(info) => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, info.clone()),
                ),
                Event::WindowClosed(id) => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, id.0),
                ),
                Event::WindowUpdated { id, update } => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, id.0, update.clone()),
                ),
                Event::WorkspaceSet { id, active } => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, *id, *active),
                ),
                Event::WorkspaceList(ws) => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, ws.clone()),
                ),
                Event::AltTabState(s) => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, s.clone()),
                ),
                Event::ConfigReloaded(a) => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, a.clone()),
                ),
                Event::GestureBegan => {
                    emitter_conn.emit_signal(dest, COMPOSITOR_PATH, COMPOSITOR_BUS_NAME, name, &seq)
                }
                Event::GestureEnded => {
                    emitter_conn.emit_signal(dest, COMPOSITOR_PATH, COMPOSITOR_BUS_NAME, name, &seq)
                }
                Event::SwitchToggled { lid_closed } => emitter_conn.emit_signal(
                    dest,
                    COMPOSITOR_PATH,
                    COMPOSITOR_BUS_NAME,
                    name,
                    &(seq, *lid_closed),
                ),
            };
            if let Err(err) = result {
                tracing::warn!(signal = name, error = %err, "failed to emit D-Bus signal");
            }
        }
    });

    (conn, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use icedtea_contract::{
        AltTabState, Appearance, Rectangle, WindowInfo, WindowUpdate, WorkspaceInfo,
    };

    #[test]
    fn event_names_match_interface() {
        assert_eq!(
            event_signal_name(&Event::WindowOpened(sample_info())),
            "WindowOpened"
        );
        assert_eq!(
            event_signal_name(&Event::WindowClosed(WindowId(1))),
            "WindowClosed"
        );
        assert_eq!(
            event_signal_name(&Event::ConfigReloaded(default_appearance())),
            "ConfigReloaded"
        );
        // Fix-round addition: the brief's own Step-1 sample only exercised
        // 3 of the 7 `Event` variants; cover the remaining 4 so every
        // `event_signal_name` match arm has a passing assertion behind it.
        assert_eq!(
            event_signal_name(&Event::WindowUpdated {
                id: WindowId(1),
                update: WindowUpdate::default()
            }),
            "WindowUpdated"
        );
        assert_eq!(
            event_signal_name(&Event::WorkspaceSet {
                id: 0,
                active: true
            }),
            "WorkspaceSet"
        );
        assert_eq!(
            event_signal_name(&Event::WorkspaceList(vec![WorkspaceInfo {
                id: 0,
                name: "1".into()
            }])),
            "WorkspaceList"
        );
        assert_eq!(
            event_signal_name(&Event::AltTabState(AltTabState {
                active: true,
                entries: vec![],
                index: 0
            })),
            "AltTabState"
        );
        // M7: every new `Event` variant needs a name mapping too -- same
        // rule as the fix-round addition above.
        assert_eq!(event_signal_name(&Event::GestureBegan), "GestureBegan");
        assert_eq!(event_signal_name(&Event::GestureEnded), "GestureEnded");
        assert_eq!(
            event_signal_name(&Event::SwitchToggled { lid_closed: true }),
            "SwitchToggled"
        );
        assert_eq!(
            event_signal_name(&Event::SwitchToggled { lid_closed: false }),
            "SwitchToggled"
        );
    }

    fn sample_info() -> WindowInfo {
        WindowInfo {
            id: WindowId(1),
            app_id: "a".into(),
            title: "t".into(),
            pid: 1,
            workspace: 0,
            geometry: Rectangle {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
            maximized: false,
            minimized: false,
            fullscreen: false,
            focused: true,
            attention: false,
        }
    }
    fn default_appearance() -> Appearance {
        Appearance {
            bar_position: "bottom".into(),
            bar_height: 42,
            corner_radius: 8,
            snap_gap: 8,
            palette: icedtea_contract::Palette {
                background: "#000000".into(),
                foreground: "#ffffff".into(),
                accent: "#0000ff".into(),
            },
            wallpaper: None,
        }
    }
}
