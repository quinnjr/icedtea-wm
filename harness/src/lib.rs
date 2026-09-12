//! The client-driven protocol test harness.
//!
//! Two halves:
//!
//! * [`Compositor`] boots the production compositor — the same
//!   `Display`/`Runtime`/`Backend`/`State` wiring `lib.rs::run()` does, minus
//!   the pieces a test has no use for (D-Bus service, wallpaper worker,
//!   config-reload pipe, signal source) — on a thread of its
//!   own, against the headless backend, and hands back its socket name plus
//!   the two production channels a test observes and drives it through: the
//!   `SeqEvent` stream and the `DbCommand` queue.
//! * [`TestClient`] is a real `wayland-client` that binds `wl_compositor`,
//!   `wl_shm` and `xdg_wm_base`, and maps one shm-backed xdg toplevel for
//!   real: commit, wait for configure, ack, attach a buffer, commit.
//!
//! Why a thread rather than a child process: `wlr::Runtime` and
//! `wlr::Display` are `!Send`, so the compositor thread has to *create*
//! everything itself and can never hand any of it back. Only the socket name
//! and the wake pipe's write half — both `Send` — cross the
//! `crossbeam_channel::bounded(1)` boot handshake.
//!
//! Why no `WAYLAND_DISPLAY`: libtest runs these tests on parallel threads and
//! each gets its own compositor with its own `add_socket_auto` name. A
//! process-global env var would be a race between them and would not even
//! name the right compositor. The client connects to an explicit path
//! instead, via `Connection::from_socket`.

use std::io::Write as _;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use icedtea_compositor::dbus::DbCommand;
use icedtea_contract::{Event, SeqEvent, Snapshot};

use wayland_client::protocol::wl_output::{self, WlOutput};
use wayland_client::protocol::{
    wl_buffer, wl_compositor, wl_data_device, wl_data_device_manager, wl_data_offer,
    wl_data_source, wl_keyboard, wl_pointer, wl_region, wl_registry, wl_seat, wl_shm, wl_shm_pool,
    wl_surface, wl_touch,
};
use wayland_client::{
    Connection, Dispatch, EventQueue, QueueHandle, WEnum, delegate_noop, event_created_child,
};
use wayland_protocols::ext::idle_notify::v1::client::{
    ext_idle_notification_v1, ext_idle_notifier_v1,
};
use wayland_protocols::ext::session_lock::v1::client::{
    ext_session_lock_manager_v1, ext_session_lock_surface_v1, ext_session_lock_v1,
};
use wayland_protocols::wp::cursor_shape::v1::client::{
    wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1,
};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1, wp_fractional_scale_v1,
};
use wayland_protocols::wp::idle_inhibit::zv1::client::{
    zwp_idle_inhibit_manager_v1, zwp_idle_inhibitor_v1,
};
use wayland_protocols::wp::pointer_constraints::zv1::client::{
    zwp_confined_pointer_v1, zwp_locked_pointer_v1, zwp_pointer_constraints_v1,
};
use wayland_protocols::wp::pointer_gestures::zv1::client::{
    zwp_pointer_gesture_pinch_v1, zwp_pointer_gesture_swipe_v1, zwp_pointer_gestures_v1,
};
use wayland_protocols::wp::presentation_time::client::{wp_presentation, wp_presentation_feedback};
use wayland_protocols::wp::primary_selection::zv1::client::{
    zwp_primary_selection_device_manager_v1, zwp_primary_selection_device_v1,
    zwp_primary_selection_offer_v1, zwp_primary_selection_source_v1,
};
use wayland_protocols::wp::relative_pointer::zv1::client::{
    zwp_relative_pointer_manager_v1, zwp_relative_pointer_v1,
};
use wayland_protocols::wp::text_input::zv3::client::{
    zwp_text_input_manager_v3, zwp_text_input_v3,
};
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};
use wayland_protocols::xdg::activation::v1::client::{xdg_activation_token_v1, xdg_activation_v1};
use wayland_protocols::xdg::decoration::zv1::client::{
    zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1,
};
use wayland_protocols::xdg::shell::client::{
    xdg_popup, xdg_positioner, xdg_surface, xdg_toplevel, xdg_wm_base,
};
use wayland_protocols_misc::zwp_input_method_v2::client::{
    zwp_input_method_keyboard_grab_v2, zwp_input_method_manager_v2, zwp_input_method_v2,
    zwp_input_popup_surface_v2,
};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1, zwp_virtual_keyboard_v1,
};
use wayland_protocols_wlr::data_control::v1::client::{
    zwlr_data_control_device_v1, zwlr_data_control_manager_v1, zwlr_data_control_offer_v1,
    zwlr_data_control_source_v1,
};
use wayland_protocols_wlr::gamma_control::v1::client::{
    zwlr_gamma_control_manager_v1, zwlr_gamma_control_v1,
};
use wayland_protocols_wlr::layer_shell::v1::client::{zwlr_layer_shell_v1, zwlr_layer_surface_v1};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::{self, ZwlrScreencopyManagerV1},
};
use wayland_protocols_wlr::virtual_pointer::v1::client::{
    zwlr_virtual_pointer_manager_v1, zwlr_virtual_pointer_v1,
};
use xkbcommon::xkb;

/// The named cursor images `wp_cursor_shape_device_v1.set_shape` accepts,
/// re-exported so a test can name one without depending on
/// `wayland-protocols` itself.
pub use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_device_v1::Shape as CursorShape;

/// The edge set `xdg_toplevel.resize` names, re-exported for the same reason
/// as [`CursorShape`]: a test naming an edge should not have to depend on
/// `wayland-protocols` itself.
pub use wayland_protocols::xdg::shell::client::xdg_toplevel::ResizeEdge;

/// The `xdg_positioner` enums, re-exported so a test can name an anchor or a
/// gravity without depending on `wayland-protocols` itself.
pub use wayland_protocols::xdg::shell::client::xdg_positioner::{
    Anchor as PopupAnchor, ConstraintAdjustment as PopupConstraint, Gravity as PopupGravity,
};

/// How long any "wait for the compositor to do a thing" helper waits before
/// declaring the harness broken.
const TIMEOUT: Duration = Duration::from_secs(5);

/// Size the client falls back to when the compositor's configure carries no
/// size of its own (0x0 means "you pick").
const FALLBACK_SIZE: (i32, i32) = (200, 100);

/// Set the headless-backend environment exactly once, no matter which test
/// thread reaches it first.
///
/// Copied rather than shared with `tests/headless_boot.rs`: each integration
/// test file is its own binary, so there is nothing to import. Same argument
/// as there — `Once::call_once` makes it structurally true that exactly one
/// thread ever runs the write and every other blocks until it returns, so no
/// thread can observe or cause a torn environment read.
fn ensure_headless_env() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let dir = private_runtime_dir();
        // SAFETY (icedtea unsafe exception (c)): `Once::call_once` above
        // guarantees this closure runs on exactly one thread and that every
        // other thread calling `ensure_headless_env` blocks until it
        // finishes. `XDG_RUNTIME_DIR` is rewritten here, before the first
        // display exists, so `add_socket_auto` (which reads it through
        // libwayland's `getenv`) never sees the session's directory.
        unsafe {
            std::env::set_var("WLR_BACKENDS", "headless");
            std::env::set_var("WLR_HEADLESS_OUTPUTS", "1");
            std::env::set_var("XDG_RUNTIME_DIR", &dir);
        }
        RUNTIME_DIR.set(dir).expect("runtime dir set once");
    });
}

/// The private `XDG_RUNTIME_DIR` every harness compositor in this process
/// creates its socket under, set by the first [`Compositor::spawn`].
static RUNTIME_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// The harness's own `XDG_RUNTIME_DIR`: `<session runtime dir>/icedtea-harness/<pid>`,
/// mode 0700, created by the first [`Compositor::spawn`] in this process.
///
/// Why not the session's directory: `wl_display_add_socket_auto` probes
/// `wayland-0` upward in `XDG_RUNTIME_DIR`. Under a live desktop that is the
/// user's own display -- the harness used to fight its lock file
/// ("unable to lock lockfile /run/user/1000/wayland-0.lock") and litter
/// `wayland-N` sockets beside the real one, which is enough to keep new
/// windows from reaching the user's session while a gate runs. A private
/// directory keeps every harness socket, and every app the tests spawn, off
/// the desktop's runtime dir entirely. Spawned apps must be handed this path
/// explicitly (`.env("XDG_RUNTIME_DIR", icedtea_harness::runtime_dir())`);
/// the process env is rewritten too, so children inherit it either way.
///
/// The directory lives on the same tmpfs as the session's runtime dir and is
/// named by pid; libwayland unlinks each socket when its display is
/// destroyed, and the empty directory goes with the session at logout.
///
/// # Panics
///
/// If called before any compositor was spawned in this process.
#[must_use]
pub fn runtime_dir() -> &'static std::path::Path {
    ensure_headless_env();
    RUNTIME_DIR
        .get()
        .expect("runtime dir is set by ensure_headless_env")
}

/// Create the per-process private runtime directory (see [`runtime_dir`]).
fn private_runtime_dir() -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base
        .join("icedtea-harness")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("creating harness runtime dir {}: {e}", dir.display()));
    // Wayland refuses a runtime dir that is not 0700 (it checks
    // `XDG_RUNTIME_DIR` permissions on connect in libwayland ≥ 1.22).
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .unwrap_or_else(|e| panic!("chmod 0700 on {}: {e}", dir.display()));
    dir
}

/// Serializes compositor *creation* across test threads.
///
/// Not a nicety, and not about the socket: wlroots keeps a single
/// process-global, unsynchronized `wl_array` of buffer-resource interfaces
/// (`buffer_resource_interfaces` in `types/buffer/resource.c`).
/// `wlr_buffer_register_resource_interface` — reached from
/// `init_graphics`, via the shm/linux-dmabuf/wl_drm globals — grows it with
/// `wl_array_add`, which *reallocs*, while `wlr_buffer_try_from_resource`
/// walks the same array from every running compositor's
/// `wl_surface.attach` handler, with no lock on either side. Two harness
/// compositors booting on different threads while a third dispatched a
/// client's attach segfaulted this test binary roughly one run in six
/// (confirmed under gdb: three threads inside
/// `wlr_buffer_try_from_resource`, one boot in flight).
///
/// Holding this lock across the whole boot makes those writes single
/// threaded, which is enough to close the race outright rather than merely
/// narrow it: registration is deduplicated by interface pointer, and every
/// compositor in the process registers the same static interfaces, so the
/// array is only ever *written* by the first boot — and no other compositor
/// exists to be reading it at that point. Every later boot only walks the
/// array to find its interfaces already there.
///
/// This is a wlroots-side process-global, not a harness bug and not
/// something the compositor could opt out of, so serializing boot is the
/// fix rather than serializing the tests: clients, event loops and
/// assertions all still run fully in parallel.
///
/// **Constraint on future changes.** The paragraph above is only sound
/// because *every* boot in the process registers the identical set of
/// static interface pointers, which is what makes "written only by the
/// first boot" true. This lock excludes writers from each other; it does
/// **not** exclude readers, and it cannot — the readers are other
/// compositors' running event loops. So a test binary that ever boots a
/// compositor with a *different* graphics configuration (a different
/// renderer, or one that skips linux-dmabuf) re-opens the race outright:
/// that boot would take the `wl_array_add` path with other compositors
/// live and walking the array. If a later task needs that, the boots with
/// differing configurations have to be kept out of the same process, not
/// merely out of each other's way.
static BOOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A headless compositor running the production event loop on its own thread.
pub struct Compositor {
    /// The socket name `add_socket_auto` picked; a `WAYLAND_DISPLAY` value,
    /// relative to the harness's private [`runtime_dir`] (never the
    /// session's `XDG_RUNTIME_DIR`).
    pub socket: String,
    /// The production event stream (`State::new`'s `dbus_tx`), exactly what
    /// the D-Bus emitter thread would consume.
    pub events: crossbeam_channel::Receiver<SeqEvent>,
    /// The production command queue (`State::set_command_receiver`), exactly
    /// what `dbus::CompositorInterface` would push onto. Prefer [`Compositor::send`],
    /// which also nudges the wake pipe.
    pub commands: crossbeam_channel::Sender<DbCommand>,
    /// Write half of the command wake pipe, so a send reaches a loop blocked
    /// in `Until::Stop`'s `dispatch(-1)`.
    wake: UnixStream,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Compositor {
    /// Boot a headless compositor on its own thread; returns once its socket
    /// exists. Panics (test context) on any boot failure.
    pub fn spawn() -> Compositor {
        ensure_headless_env();

        let (boot_tx, boot_rx) = crossbeam_channel::bounded::<(String, UnixStream)>(1);
        let (event_tx, event_rx) = crossbeam_channel::unbounded::<SeqEvent>();
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<DbCommand>();

        let thread = std::thread::spawn(move || {
            // See `BOOT_LOCK`: everything from here to the handshake below
            // touches wlroots' process-global buffer-interface registry.
            // Poison is irrelevant — a panicking boot leaves the registry
            // no worse off than a successful one, and the next `spawn`
            // failing for an unrelated earlier panic would only obscure it.
            let boot_guard = BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

            // Same order as `lib.rs::run()`, and load-bearing for the same
            // reasons documented there.
            let display = wlr::Display::new().expect("display");
            let runtime = wlr::Runtime::new().expect("runtime");
            let backend = wlr::Backend::autocreate(&display.event_loop()).expect("backend");
            runtime.init_graphics(&display, &backend).expect("graphics");
            runtime.create_xdg_shell(&display, 6).expect("xdg_wm_base");
            // Unlike `lib.rs::run()`, which degrades, a harness that cannot
            // advertise xdg-decoration is simply broken: the SSD negotiation
            // test would then assert against a global that was never there.
            runtime
                .create_xdg_decoration_manager(&display)
                .expect("zxdg_decoration_manager_v1");
            // Same "harness cannot degrade" tone as the decoration manager
            // just above: a test that maps a layer panel needs the global
            // to actually exist, not just for `lib.rs::run()`'s production
            // boot to log and move on.
            runtime
                .create_layer_shell(&display, 4)
                .expect("zwlr_layer_shell_v1");
            // Same "harness cannot degrade" tone: the selection tests bind
            // these globals directly and would assert against ones that were
            // never advertised.
            runtime
                .create_primary_selection_manager(&display)
                .expect("zwp_primary_selection_device_manager_v1");
            runtime
                .create_data_control_manager(&display)
                .expect("zwlr_data_control_manager_v1");
            runtime
                .create_virtual_keyboard_manager(&display)
                .expect("zwp_virtual_keyboard_manager_v1");
            // Same "harness cannot degrade" tone: the DnD tests inject
            // pointer motion/buttons and would have no manager to bind.
            runtime
                .create_virtual_pointer_manager(&display)
                .expect("zwlr_virtual_pointer_manager_v1");
            // Same "harness cannot degrade" tone: the screencopy test binds
            // this global directly and would assert against one that was
            // never advertised.
            runtime
                .create_screencopy_manager(&display)
                .expect("zwlr_screencopy_manager_v1");
            // Same "harness cannot degrade" tone: the session-lock and idle
            // tests bind these globals directly and would assert against
            // ones that were never advertised.
            runtime
                .create_session_lock_manager(&display)
                .expect("ext_session_lock_manager_v1");
            runtime
                .create_idle_notifier(&display)
                .expect("ext_idle_notifier_v1");
            runtime
                .create_idle_inhibit_manager(&display)
                .expect("zwp_idle_inhibit_manager_v1");
            // Same "harness cannot degrade" tone: the pointer-constraints
            // tests bind these globals directly and would assert against
            // ones that were never advertised.
            runtime
                .create_pointer_constraints_manager(&display)
                .expect("zwp_pointer_constraints_v1");
            runtime
                .create_relative_pointer_manager(&display)
                .expect("zwp_relative_pointer_manager_v1");
            // Same "harness cannot degrade" tone: the M7 gesture double
            // binds this global directly and would assert against one that
            // was never advertised.
            runtime
                .create_pointer_gestures_manager(&display)
                .expect("zwp_pointer_gestures_v1");
            // Same "harness cannot degrade" tone: the IME relay tests bind
            // these globals directly and would assert against ones that were
            // never advertised.
            runtime
                .create_text_input_manager(&display)
                .expect("zwp_text_input_manager_v3");
            runtime
                .create_input_method_manager(&display)
                .expect("zwp_input_method_manager_v2");
            // Same "harness cannot degrade" tone: the output-management test
            // binds `zwlr_output_manager_v1` directly and would assert against
            // one that was never advertised.
            runtime
                .create_output_manager(&display)
                .expect("zwlr_output_manager_v1");
            // Finding F13: the nine A2 compat globals come from the
            // compositor crate's own `create_compat_globals`, the very
            // function `lib.rs::run()` calls -- not a hand-copied list. That
            // is what makes `a2_batch1_globals_are_advertised` /
            // `a2_batch2_globals_are_advertised` load-bearing for the real
            // boot path instead of proving only that the harness advertises
            // them. Its non-fatal tone is inherited deliberately: those two
            // tests are the assertion, so a `create_*` that fails here fails
            // them rather than aborting the test process.
            icedtea_compositor::create_compat_globals(&runtime, &display, &backend);
            runtime.create_seat(&display, "seat0").expect("seat0");
            // X11 application support. Non-fatal here, unlike the globals
            // above: a host with no `Xwayland` binary is a legitimate CI
            // configuration, and the X11 end-to-end test skips cleanly when
            // `DbCommand::XwaylandDisplay` comes back `None`. `lazy` is `true`,
            // so no `Xwayland` process is spawned for the non-X11 tests -- the
            // manager only reserves a display socket, and nothing connects to
            // it unless a test asks for `DISPLAY` and drives an X11 client.
            // Must come after `create_seat`, whose seat the clipboard/DND
            // bridge needs.
            if let Err(err) = runtime.create_xwayland(&display, true) {
                eprintln!("harness: Xwayland unavailable ({err}); X11 tests will skip");
            }
            // Test-only: makes the seat advertise the touch capability so
            // headless clients can bind `wl_touch` and injected touch
            // points are accepted. Harness-only -- the real
            // `compositor/src/lib.rs` boot must NOT call this. Must come
            // AFTER `create_seat`: that call does not itself trigger a
            // capability recompute, so calling this before the seat exists
            // sets the flag but its own immediate recompute is a no-op (no
            // seat yet) and nothing re-triggers one afterward -- the seat
            // would never actually advertise touch.
            runtime.enable_test_touch();

            // `state` is declared after `display`/`runtime`/`backend` so that
            // ordinary end-of-scope drop order drops it first: `attach` hands
            // it a `Runtime` clone, which must not outlive the `Display`.
            let mut state =
                icedtea_compositor::state::State::new(icedtea_config::default_config(), event_tx);
            state.wayland.attach(runtime.clone());

            // Same "harness cannot degrade" tone as the globals above, and
            // required for M4.3's screencopy tests specifically: without
            // this, the scene is empty and a capture reads back whatever the
            // renderer's clear color is rather than the configured wallpaper
            // color, which is exactly what `screencopy_of_empty_output_is_
            // the_wallpaper_color` asserts on. Same boot-order and sizing
            // contract as `lib.rs::run()` (sized to nothing until an output
            // with a mode arrives; `new_output` resizes it) and the same
            // "lowered now so nothing later has to remember to" reasoning.
            let background = runtime
                .add_rect(
                    1,
                    1,
                    icedtea_compositor::render::wallpaper_color(&state.config.appearance),
                )
                .expect("background rect");
            runtime.lower_rect_to_bottom(background);
            state.set_background(background);

            state.set_command_receiver(cmd_rx);

            let (cmd_wake_write, cmd_wake_id) =
                icedtea_compositor::backend::wake_source(&runtime).expect("cmd wake source");
            state.set_cmd_wake_source(cmd_wake_id);

            let socket = display.add_socket_auto().expect("wayland socket");
            // Deliberately no `set_var("WAYLAND_DISPLAY", ...)`: see the
            // module doc. The handshake is the only way out of this thread —
            // everything above is `!Send`.
            //
            // DISPLAY + the X11 cursor hints ARE published, though, and here —
            // *before* the boot handshake below (review finding #5). Publishing
            // before `boot_tx.send` means the writes happen-before the test
            // thread resumes from `Compositor::spawn`, so a test that reads the
            // process env (the DISPLAY/cursor-env test) is ordered after them
            // rather than racing the old `xwayland_ready` write on this thread.
            icedtea_compositor::state::State::publish_xwayland_env(
                runtime.xwayland_display_name().as_deref(),
            );
            boot_tx
                .send((socket, cmd_wake_write))
                .expect("boot handshake");
            drop(boot_tx);
            drop(boot_guard);

            if let Err(err) = backend.run_all(&display, &mut state, &runtime, wlr::Until::Stop) {
                panic!("run_all: {err:?}");
            }
        });

        let (socket, wake) = boot_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("compositor thread never completed its boot handshake");

        Compositor {
            socket,
            events: event_rx,
            commands: cmd_tx,
            wake,
            thread: Some(thread),
        }
    }

    /// Absolute path of this compositor's socket, under [`runtime_dir`].
    pub fn socket_path(&self) -> std::path::PathBuf {
        runtime_dir().join(&self.socket)
    }

    /// Send a command the way `dbus::CompositorInterface::send` does: onto the
    /// channel, then a nudge on the wake pipe.
    pub fn send(&self, cmd: DbCommand) {
        self.commands
            .send(cmd)
            .expect("compositor command channel closed");
        icedtea_compositor::backend::wake(&self.wake);
    }

    /// `GetState` round trip, with a `TIMEOUT` deadline.
    pub fn snapshot(&self) -> Snapshot {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::GetState(reply_tx));
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered GetState")
    }

    /// The `DISPLAY` name (`:N`) Xwayland advertises, or `None` when no
    /// Xwayland was created (the `Xwayland` binary is absent). Available with
    /// lazy start as soon as the manager reserves its display socket -- before
    /// any `Xwayland` process is spawned -- so an X11 test reads this, connects
    /// a client to it, and that connection is what triggers the lazy start.
    pub fn xwayland_display(&self) -> Option<String> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::XwaylandDisplay { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered XwaylandDisplay")
    }

    /// Whether the lazy Xwayland has started and `xwayland_ready` has fired (the
    /// seat is wired, so the clipboard/primary/DND bridge is armed). A test that
    /// needs the bridge live polls this after connecting the client that
    /// triggers the lazy start — the readiness barrier that used to be inferred
    /// from `DISPLAY` being republished on ready (review finding #5).
    pub fn xwayland_ready(&self) -> bool {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::XwaylandReady { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered XwaylandReady")
    }

    /// Override the primary output's recorded scale (and re-publish the X11
    /// `Xft.dpi` hint if Xwayland is already up), for the M4 HiDPI test. Blocks
    /// on the ack -- see [`Self::inject_touch_down`]'s doc -- so the scale has
    /// actually been recorded before this returns.
    pub fn set_output_scale_for_test(&self, scale: f64) {
        // `spawn` returns at the boot handshake, before `run_all` creates the
        // headless output, so the very first attempt can land before any output
        // exists. Poll until the compositor reports it actually recorded the
        // scale (review finding #11) rather than acking a silent no-op, so the
        // caller is guaranteed the scale is live before it proceeds.
        let deadline = std::time::Instant::now() + TIMEOUT;
        loop {
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            self.send(DbCommand::SetOutputScaleForTest {
                scale,
                reply: reply_tx,
            });
            let recorded = reply_rx
                .recv_timeout(TIMEOUT)
                .expect("compositor never answered SetOutputScaleForTest");
            if recorded {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no output ever existed to record the test scale on"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Probe every mapped override-redirect (OR) X11 pop-up the compositor is
    /// tracking, reading each one's real scene state (position, whether it is in
    /// the band above managed toplevels, and whether it holds the keyboard).
    /// Blocks on the reply -- see [`Self::inject_touch_down`]'s doc.
    pub fn xwayland_override_redirect(
        &self,
    ) -> Vec<icedtea_compositor::dbus::OverrideRedirectProbe> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::XwaylandOverrideRedirect { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered XwaylandOverrideRedirect")
    }

    /// Synthesize a touch-down at `(x, y)` for touch point `id` on the
    /// compositor thread, via `wlr::Runtime::inject_touch_down`. Returns the
    /// grab serial it minted, or `None` if there was no seat or no surface
    /// under the point. Blocks on the reply so the injection has actually
    /// run before this returns -- see `DbCommand::InjectTouchDown`'s doc.
    pub fn inject_touch_down(&self, x: f64, y: f64, id: i32, time_msec: u32) -> Option<u32> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InjectTouchDown {
            x,
            y,
            id,
            time_msec,
            reply: reply_tx,
        });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InjectTouchDown")
    }

    /// Synthesize a touch-motion to `(x, y)` for touch point `id`. Blocks on
    /// the reply -- see [`Self::inject_touch_down`]'s doc.
    pub fn inject_touch_motion(&self, x: f64, y: f64, id: i32, time_msec: u32) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InjectTouchMotion {
            x,
            y,
            id,
            time_msec,
            reply: reply_tx,
        });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InjectTouchMotion");
    }

    /// Synthesize a touch-up for touch point `id`. Blocks on the reply --
    /// see [`Self::inject_touch_down`]'s doc.
    pub fn inject_touch_up(&self, id: i32, time_msec: u32) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InjectTouchUp {
            id,
            time_msec,
            reply: reply_tx,
        });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InjectTouchUp");
    }

    /// Drive the consumer's `SeatHandler::touch_cancelled` on the
    /// compositor thread (M7). Headless has no wire producer for cancels,
    /// so this clears the consumer mirror only -- the client sees no wire
    /// cancel. Blocks on the reply -- see [`Self::inject_touch_down`]'s
    /// doc.
    pub fn inject_touch_cancel(&self) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InjectTouchCancel { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InjectTouchCancel");
    }

    /// Drive the consumer's `SeatHandler::gesture_began` (`began == true`)
    /// or `gesture_ended` on the compositor thread (M7). Headless has no
    /// gesture hardware, so the id names no live pointer -- harmless by
    /// the handlers' contract. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc.
    pub fn inject_gesture(&self, began: bool) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InjectGesture {
            began,
            reply: reply_tx,
        });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InjectGesture");
    }

    /// Drive the consumer's switch-apply path on the compositor thread
    /// (M7) with a hardware-decoded `(type, on)` pair no headless device
    /// can produce. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc.
    pub fn inject_switch_toggle(&self, switch_type: wlr::SwitchType, on: bool) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InjectSwitchToggle {
            switch_type,
            on,
            reply: reply_tx,
        });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InjectSwitchToggle");
    }

    /// The drag icon's current scene layout position, via
    /// `wlr::Runtime::drag_icon_position`. `None` if no drag with a visible
    /// icon is in progress. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc.
    pub fn drag_icon_position(&self) -> Option<(i32, i32)> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::DragIconPosition { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered DragIconPosition")
    }

    /// How many popups the compositor has itself closed through
    /// `wlr::Runtime::dismiss_popup` since boot. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc.
    pub fn popups_dismissed(&self) -> usize {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::PopupsDismissed { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered PopupsDismissed")
    }

    /// Whether the session is currently locked, via
    /// `wlr::Runtime::is_session_locked`. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc.
    pub fn session_locked(&self) -> bool {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::SessionLocked { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered SessionLocked")
    }

    /// Whether an IME is currently activated for a focused+enabled
    /// text-input, via `wlr::Runtime::input_method_active`. Blocks on the
    /// reply -- see [`Self::inject_touch_down`]'s doc.
    pub fn input_method_active(&self) -> bool {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InputMethodActive { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InputMethodActive")
    }

    /// The pointer's current position, via `wlr::Runtime::cursor_position`.
    /// Blocks on the reply -- see [`Self::inject_touch_down`]'s doc.
    pub fn cursor_position(&self) -> (f64, f64) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::CursorPosition { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered CursorPosition")
    }

    /// The scene position of the currently-placed input-method candidate
    /// popup, or `None` when none is placed. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc. The compositor half of A6.2 test 8.
    pub fn input_popup_position(&self) -> Option<(i32, i32)> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InputPopupPosition { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InputPopupPosition")
    }

    /// The scene node of the currently-placed input-method candidate popup,
    /// or `None` when none is placed. The first half of the popup-teardown
    /// tripwire (test 10b): capture while placed, tear down, then assert
    /// [`Self::scene_node_position`] reads `None` for it.
    pub fn input_popup_node(&self) -> Option<wlr::NodeId> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::InputPopupNode { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered InputPopupNode")
    }

    /// The scene position of an arbitrary node by id, or `None` for an
    /// unknown or destroyed id. The second half of the tripwire.
    pub fn scene_node_position(&self, node: wlr::NodeId) -> Option<(i32, i32)> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::SceneNodePosition {
            node,
            reply: reply_tx,
        });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered SceneNodePosition")
    }

    /// The `Debug` name of the named cursor shape currently in force
    /// (e.g. `"Default"`, `"Text"`), read straight off
    /// `wlr::Runtime::cursor_shape` -- the crate's own record of what it
    /// handed wlroots, with `None` rendered as `"Default"`. Load-bearing
    /// since `wlr` 0.20.26: deleting the compositor's
    /// `Runtime::set_cursor_shape` call now genuinely makes this read
    /// `"Default"`. Blocks on the reply -- see
    /// [`Self::inject_touch_down`]'s doc.
    pub fn cursor_shape(&self) -> String {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.send(DbCommand::CursorShape { reply: reply_tx });
        reply_rx
            .recv_timeout(TIMEOUT)
            .expect("compositor never answered CursorShape")
    }

    /// The primary output's real geometry, via `State::outputs`. Panics if
    /// no output ever appears within `TIMEOUT`.
    ///
    /// Review finding F11: the honest answer to "how big is the screen",
    /// replacing the maximize-a-window-and-read-its-geometry dance the
    /// compat tests used -- that returned the *usable* rect inset by the
    /// configured snap gap, so a test picking a point "just inside the
    /// bottom-right corner" was actually picking one a gap-width away from
    /// it. Polls because `spawn` returns at the boot handshake, before
    /// `run_all` has created the headless output (the same race
    /// [`Self::set_output_scale_for_test`] documents).
    pub fn output_size(&self) -> (i32, i32) {
        let deadline = std::time::Instant::now() + TIMEOUT;
        loop {
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            self.send(DbCommand::OutputSize { reply: reply_tx });
            let geo = reply_rx
                .recv_timeout(TIMEOUT)
                .expect("compositor never answered OutputSize");
            if let Some(geo) = geo {
                assert!(
                    geo.width > 0 && geo.height > 0,
                    "output geometry must be real, got {geo:?}"
                );
                return (geo.width, geo.height);
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no output ever appeared to size"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Give the compositor a bounded window to finish processing something
    /// this side of the socket can't directly observe -- most notably a
    /// client's disconnect, which the event loop only notices on its own
    /// next dispatch. Deliberately NOT a "poll `pred` until true" loop: that
    /// shape can pass vacuously (or hide a state flapping back and forth)
    /// when the caller's real intent is "make sure whatever the disconnect
    /// triggers has actually landed" before taking a single, final reading.
    /// Instead this forces a fixed number of full round trips through the
    /// command channel -- each `snapshot()` wakes the loop via the same wake
    /// pipe a real socket-readable event would use and blocks until it has
    /// replied, so by the last iteration the loop has been given many
    /// dispatch cycles with real pauses between them.
    pub fn settle(&self) {
        for _ in 0..20 {
            let _ = self.snapshot();
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Block until an event matching `pred` arrives; panics on timeout.
    ///
    /// The predicate sees the inner [`Event`]; the `seq` wrapper is dropped
    /// because nothing a test asserts on depends on it (the sequence
    /// contract itself is `contract`'s own unit tests' job).
    pub fn wait_event(&self, pred: impl Fn(&Event) -> bool) -> Event {
        let deadline = Instant::now() + TIMEOUT;
        let mut seen: Vec<Event> = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("timed out waiting for a matching event; saw: {seen:?}");
            }
            match self.events.recv_timeout(remaining) {
                Ok(SeqEvent { event, .. }) => {
                    if pred(&event) {
                        return event;
                    }
                    seen.push(event);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    panic!("timed out waiting for a matching event; saw: {seen:?}")
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    panic!("compositor event channel closed; saw: {seen:?}")
                }
            }
        }
    }
}

impl Drop for Compositor {
    fn drop(&mut self) {
        // The production shutdown path: `Quit` on the command channel plus a
        // wake, which `State::should_stop` honours through `quitting`.
        let _ = self.commands.send(DbCommand::Quit);
        icedtea_compositor::backend::wake(&self.wake);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
            && !std::thread::panicking()
        {
            panic!("compositor thread panicked");
        }
    }
}

/// One gesture phase a `GestureClient` was sent, in arrival order (M7).
///
/// The phase log the harness double keeps: kind (swipe vs pinch), stage
/// (begin/update/end/cancelled) and finger count. Headless produces no
/// phases (they come from hardware gesture signals), so the log stays empty
/// in tests -- it exists so the double records payloads the way every other
/// double here does, and so runs against real hardware can assert on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedGesture {
    /// Whether the phase belongs to the swipe or the pinch object.
    pub swipe: bool,
    /// The stage: begin, update, end, or cancelled (an `end` with the
    /// protocol's `cancelled` bit set).
    pub stage: GestureStage,
    /// The finger count the phase carried.
    pub fingers: u32,
}

/// The stage of a [`RecordedGesture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureStage {
    Begin,
    Update,
    End,
    Cancelled,
}

/// What the client's `Dispatch` impls accumulate.
#[derive(Default)]
struct ClientState {
    compositor: Option<wl_compositor::WlCompositor>,
    shm: Option<wl_shm::WlShm>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    decoration_manager: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    layer_shell: Option<zwlr_layer_shell_v1::ZwlrLayerShellV1>,
    /// Every `mode` this client has been sent on its decoration object, in
    /// arrival order (1 = client-side, 2 = server-side). A `Vec` rather than
    /// a latest-only field because "the compositor answered *once*, with the
    /// right mode" is a stronger and more useful claim than "it eventually
    /// said the right thing", and only the history can express it.
    decoration_modes: Vec<u32>,
    /// Most recent `xdg_toplevel.configure` size.
    configured: Option<(i32, i32)>,
    /// How many `xdg_surface.configure` events have arrived, ever.
    ///
    /// Both the gate `map_toplevel` waits on before attaching its first
    /// buffer (`> 0`) and — via [`TestClient::configure_count`] — the only
    /// reliable "a *new* configure arrived" signal a test has. A counter
    /// rather than a flag or a serial because every configure is acked in
    /// the handler itself, so no serial ever survives to the caller, while
    /// `configured`/`states` only ever report the *latest* values:
    /// [`TestClient::wait_until`] evaluates its predicate before it pumps,
    /// so a predicate phrased over those alone returns `true` instantly on
    /// stale state if it already happened to hold. Anything asserting an
    /// idempotent-looking round trip ("request maximize while already
    /// maximized", "resize to the same geometry") must latch this counter
    /// first and wait for it to advance.
    configures: u32,
    /// `xdg_toplevel` states from the most recent configure.
    states: Vec<u32>,
    closed: bool,
    /// Most recent `zwlr_layer_surface_v1.configure` size.
    layer_configured: Option<(u32, u32)>,
    /// How many `zwlr_layer_surface_v1.configure` events have arrived, ever
    /// — the layer analogue of [`ClientState::configures`], and for the same
    /// reason: `layer_configured` only ever reports the *latest* size, so a
    /// predicate over it alone cannot tell "a fresh configure arrived" from
    /// "the old one is still sitting there". The unmap/remap round trip
    /// (final review I1) is exactly that case: the placement a remapped
    /// panel is configured with is byte-identical to the one it had before
    /// it unmapped, so only the counter can see the second one.
    layer_configures: u32,
    /// Every global the compositor advertised on the registry, in
    /// `(interface, name)` arrival order. Recorded for every global, not just
    /// the bound ones, so a test can assert a global *exists* without this
    /// harness having to bind it.
    globals: Vec<(String, u32)>,

    // --- selection (M4.1) ---
    seat: Option<wl_seat::WlSeat>,
    /// This client's keyboard, created when the seat advertises the keyboard
    /// capability — the source of the input serial `set_selection` needs.
    keyboard: Option<wl_keyboard::WlKeyboard>,
    virtual_keyboard_manager: Option<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1>,
    /// Lets a test spawn a [`VirtualPointerClient`] and mint pointer motion
    /// and button events without a real input device — the M4.2 drag-and-drop
    /// grab serial's source.
    virtual_pointer_manager: Option<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1>,
    data_device_manager: Option<wl_data_device_manager::WlDataDeviceManager>,
    /// This client's data device, created from the manager + seat during
    /// connect so it is listening before the client is ever focused.
    data_device: Option<wl_data_device::WlDataDevice>,
    /// The source this client last offered, kept alive so it can answer
    /// `send` for as long as it owns the selection.
    data_source: Option<wl_data_source::WlDataSource>,
    /// The most recent selection `wl_data_offer` the compositor delivered (the
    /// clipboard this client would paste from), or `None` if the selection was
    /// cleared.
    current_offer: Option<wl_data_offer::WlDataOffer>,
    /// Mimes advertised on the in-flight offer, reset when a new offer is
    /// introduced. One selection at a time, so a single vec suffices.
    offer_mimes: Vec<String>,
    /// The last input-event serial this client saw (keyboard enter/key). `None`
    /// on a headless seat with no keyboard capability — `set_selection` then
    /// passes 0.
    last_serial: Option<u32>,
    /// How many `wl_keyboard.key` events this client has received, ever —
    /// unlike `last_serial` (which `enter` also sets), this only advances on
    /// an actual key press/release delivered to this client, which is exactly
    /// what M4.4's lock-isolation test needs to observe.
    key_events: u32,
    /// What this client's own clipboard data source answers `send` with. The
    /// `wl_data_source` and clipboard `zwlr_data_control_source_v1` handlers read
    /// these; the *primary* data-control source has its own pair below so one
    /// client can legitimately own CLIPBOARD and PRIMARY at once, each answered
    /// from its own payload (review finding #13).
    offered_mime: String,
    offered_payload: Vec<u8>,
    /// The primary (middle-click) data-control source's `send` payload, kept
    /// separate from the clipboard pair above so a client owning both selections
    /// does not cross-feed one into the other's reader.
    offered_primary_mime: String,
    offered_primary_payload: Vec<u8>,
    /// How many `wl_data_source.send` requests this client has serviced — the
    /// signal a reader uses to know the owner has written the payload. Shared
    /// by the `wl_data_source` and `zwlr_data_control_source_v1` send handlers,
    /// since a client owns the selection through one or the other, never both.
    source_sends: u32,

    // --- data-control (M4.1) ---
    data_control_manager: Option<zwlr_data_control_manager_v1::ZwlrDataControlManagerV1>,
    data_control_device: Option<zwlr_data_control_device_v1::ZwlrDataControlDeviceV1>,
    /// The source a data-control client last set, kept alive to answer `send`.
    data_control_source: Option<zwlr_data_control_source_v1::ZwlrDataControlSourceV1>,
    /// The current data-control selection offer (the clipboard a manager reads).
    data_control_offer: Option<zwlr_data_control_offer_v1::ZwlrDataControlOfferV1>,
    /// Mimes advertised on the in-flight data-control offer, reset per offer.
    data_control_mimes: Vec<String>,
    /// The current data-control *primary* selection offer (data-control v2
    /// bridges the middle-click/primary selection focus-lessly, exactly as it
    /// does the clipboard). Distinct from `data_control_offer` so a test can
    /// assert the two selections independently.
    data_control_primary_offer: Option<zwlr_data_control_offer_v1::ZwlrDataControlOfferV1>,
    /// The primary source a data-control client last set, kept alive to answer
    /// `send` -- the primary counterpart of `data_control_source`.
    data_control_primary_source: Option<zwlr_data_control_source_v1::ZwlrDataControlSourceV1>,

    // --- primary selection (M4.1) ---
    primary_manager:
        Option<zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1>,
    primary_device: Option<zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1>,
    primary_source: Option<zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1>,
    primary_offer: Option<zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1>,
    primary_mimes: Vec<String>,

    // --- pointer + drag-and-drop (M4.2) ---
    /// This client's pointer, created when the seat advertises the pointer
    /// capability -- the source of the implicit-grab serial `start_drag`
    /// needs. Mirrors `keyboard` above.
    pointer: Option<wl_pointer::WlPointer>,
    /// The last serial this client saw on `wl_pointer` (enter or button).
    /// `start_drag`'s serial must be the *button* press that grabbed the
    /// surface, and `button` always arrives after `enter`, so the latest
    /// value here is always the right one to pass.
    last_pointer_serial: Option<u32>,
    /// Every `wl_pointer.button` this client has seen, as
    /// `(button_code, pressed)` in arrival order.
    ///
    /// A `Vec` rather than a "last button" scalar: the implicit-grab tests
    /// assert on a *sequence* (press here, release there) and on the
    /// *absence* of an event on the other client, both of which a scalar
    /// that the next event overwrites cannot express.
    pointer_buttons: Vec<(u32, bool)>,
    /// Every `wl_pointer.motion` this client has seen, in surface-local
    /// coordinates. `enter` is deliberately *not* recorded here -- the
    /// grab tests need to tell "the surface was re-entered" from "the
    /// surface was sent a motion while it held the grab" apart.
    pointer_motions: Vec<(f64, f64)>,
    /// How many `wl_pointer.enter` events this client has seen. The grab
    /// tests assert this does *not* advance while another surface holds
    /// the implicit grab.
    pointer_enters: u32,
    /// Whether the popup this client opened has been dismissed with
    /// `xdg_popup.popup_done`. Set once and never cleared: a dismissed
    /// popup is gone for good.
    popup_done: bool,
    /// The depth of every popup that has received `xdg_popup.popup_done`, in
    /// the order the events arrived.
    ///
    /// `popup_done` alone cannot tell a whole-chain dismissal from a partial
    /// one, nor deepest-first order from shallowest-first: any depth's event
    /// latches the same bool. Recording the depths is what lets a nested
    /// chain's dismissal *order* be asserted -- `[1, 0]` for a two-level
    /// chain torn down deepest-first, which is what xdg-shell requires
    /// (`xdg_popup.destroy` on a popup with live children is a protocol
    /// error).
    popup_done_depths: Vec<usize>,
    /// The geometry each live popup's last `xdg_popup.configure` carried,
    /// indexed by depth -- `[0]` is the outermost popup of the chain.
    ///
    /// A `Vec` rather than a scalar because a nested menu chain has one
    /// configure per level and every level is separately assertable
    /// (`popup_configured_at`).
    popup_geometries: Vec<Option<(i32, i32, i32, i32)>>,
    /// Whether each depth's popup `xdg_surface.configure` has arrived, which
    /// is what says that popup may legally attach a buffer and map.
    popup_acked: Vec<bool>,
    /// The token echoed by the most recent `xdg_popup.repositioned`.
    popup_repositioned: Option<u32>,
    /// The surface-local coordinates the most recent `wl_pointer.enter`
    /// carried.
    ///
    /// Tracked separately from `pointer_motions` because wlroots suppresses
    /// a `motion` whose coordinates match the ones the `enter` just
    /// established -- so the enter is the only place the grab's reference
    /// point is observable from a client.
    pointer_enter_position: Option<(f64, f64)>,
    /// How many `wl_pointer.leave` events this client has seen.
    pointer_leaves: u32,
    /// The drag offer delivered on `wl_data_device.enter` while this client
    /// is a drag-and-drop destination, or `None` before a drag has entered
    /// (or after it has left).
    dnd_offer: Option<wl_data_offer::WlDataOffer>,
    /// Whether `wl_data_device.drop` has arrived for the current drag.
    dropped: bool,

    // --- touch drag-and-drop (M4.2) ---
    /// This client's touch object, created when the seat advertises the
    /// touch capability -- mirrors `pointer`/`keyboard` above. The
    /// destination side of a touch drag learns about it entirely through
    /// `wl_data_device`, never through this; the M7 `TouchClient` double
    /// reads the recorded streams below instead.
    touch: Option<wl_touch::WlTouch>,
    // --- touch recording (M7) ---
    /// Every `wl_touch.down` this client has seen, as
    /// `(touch_id, surface_x, surface_y, serial)` in arrival order. The
    /// payload a touch e2e asserts the focused surface received.
    touch_downs: Vec<(i32, f64, f64, u32)>,
    /// Every `wl_touch.motion` this client has seen, as
    /// `(touch_id, surface_x, surface_y)` in arrival order.
    touch_motions: Vec<(i32, f64, f64)>,
    /// Every `wl_touch.up` this client has seen, as `(touch_id, serial)`
    /// in arrival order.
    touch_ups: Vec<(i32, u32)>,
    /// How many `wl_touch.cancel` events this client has seen. Headless
    /// produces none (cancels come from hardware), so this only advances
    /// on real hardware -- recorded for completeness, same as the gesture
    /// phase log below.
    touch_cancels: u32,

    // --- screencopy (M4.3) ---
    output: Option<WlOutput>,
    screencopy_manager: Option<ZwlrScreencopyManagerV1>,
    /// Screencopy frame bookkeeping, filled by the frame's event dispatch.
    screencopy_frame: ScreencopyFrameState,

    // --- session lock (M4.4) ---
    session_lock_manager: Option<ext_session_lock_manager_v1::ExtSessionLockManagerV1>,
    /// Set true on the active lock's `locked` event.
    session_locked: bool,
    /// Set true on the active lock's `finished` event.
    session_finished: bool,
    /// One entry per `ext_session_lock_surface_v1` this client has created,
    /// keyed by identity so the surface's own `configure` dispatch can find
    /// its matching `wl_surface` + shm keepalives. The `configure` handler
    /// does the whole ack/attach/commit dance itself (it has both `state`
    /// and `qh` in hand), so nothing outside `Dispatch` needs to drive it.
    lock_surfaces: Vec<LockSurfaceEntry>,

    // --- idle-notify / idle-inhibit (M4.4) ---
    idle_notifier: Option<ext_idle_notifier_v1::ExtIdleNotifierV1>,
    idle_inhibit_manager: Option<zwp_idle_inhibit_manager_v1::ZwpIdleInhibitManagerV1>,
    /// Set true on the active `ext_idle_notification_v1`'s `idled` event;
    /// reset whenever [`IdleNotifyClient::notification`] requests a fresh
    /// notification object, so a stale flag from a previous request can
    /// never be mistaken for a fresh one.
    idle_idled: bool,
    /// As `idle_idled`, for `resumed`.
    idle_resumed: bool,

    // --- pointer-constraints (M4.5) ---
    pointer_constraints: Option<zwp_pointer_constraints_v1::ZwpPointerConstraintsV1>,
    relative_pointer_manager: Option<zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1>,
    /// Running sum of every `zwp_relative_pointer_v1.relative_motion` event's
    /// (accelerated) `dx`/`dy` this client has received.
    relative_delta: (f64, f64),
    /// How many `relative_motion` events have arrived, ever -- the "did any
    /// relative motion arrive at all" signal `relative_delta` alone cannot
    /// give (a delta that nets to exactly zero looks identical to "none").
    relative_motion_events: u32,

    // --- pointer-gestures (M7) ---
    /// Bound whenever advertised; used by [`GestureClient::spawn`].
    gestures_manager: Option<zwp_pointer_gestures_v1::ZwpPointerGesturesV1>,
    /// Every swipe/pinch phase this client has been sent, in arrival
    /// order. Headless produces none (phases come from hardware gesture
    /// signals), so this stays empty in tests -- recorded for completeness
    /// and for runs against real hardware, the same shape as
    /// `touch_cancels` above.
    gesture_events: Vec<RecordedGesture>,

    // --- A2 batch-1 passive protocols (Task 7-9) ---
    /// Bound only by the [`TestClient::get_viewport`] path (task 8's
    /// crop/scale test).
    viewporter: Option<wp_viewporter::WpViewporter>,
    /// Bound only by the [`TestClient::get_fractional_scale`] path (task
    /// 8's preferred-scale test).
    fractional_scale_manager: Option<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1>,
    /// The latest `preferred_scale` this client's `wp_fractional_scale_v1`
    /// object has received (the numerator of a fraction over 120), or
    /// `None` before the compositor has sent one.
    preferred_scale: Option<u32>,
    /// Bound only by the [`TestClient::request_presentation_feedback`] path
    /// (task 9's feedback test).
    presentation: Option<wp_presentation::WpPresentation>,
    /// The terminal event (`presented` or `discarded`) this client's most
    /// recent `wp_presentation_feedback` object has received, if any. See
    /// [`PresentationOutcome`]'s own doc for why both count as terminal.
    presentation_outcome: Option<PresentationOutcome>,

    // --- A2 batch-2 request-handled protocols (Task 7-10) ---
    /// Bound whenever advertised; used by [`TestClient::set_cursor_shape`].
    cursor_shape_manager: Option<wp_cursor_shape_manager_v1::WpCursorShapeManagerV1>,
    /// Bound whenever advertised; used by
    /// [`TestClient::create_activation_token`] / [`TestClient::activate_self`].
    activation: Option<xdg_activation_v1::XdgActivationV1>,
    /// The token string the most recent `xdg_activation_token_v1.done`
    /// carried, or `None` before one has arrived. Reset by
    /// [`TestClient::create_activation_token`] so a stale token from an
    /// earlier request can never be mistaken for a fresh one.
    activation_token: Option<String>,
    /// Bound whenever advertised; used by [`GammaControlClient`].
    gamma_control_manager: Option<zwlr_gamma_control_manager_v1::ZwlrGammaControlManagerV1>,
    /// The `gamma_size` the compositor reported for this client's
    /// `zwlr_gamma_control_v1`, if it sent one.
    gamma_size: Option<u32>,
    /// Set true on the gamma control's `failed` event -- what wlroots sends
    /// instead of `gamma_size` for an output whose gamma LUT size is 0.
    gamma_failed: bool,

    // --- text-input (M6.1) ---
    /// Bound whenever advertised; used by [`TextInputClient::spawn`].
    text_input_manager: Option<zwp_text_input_manager_v3::ZwpTextInputManagerV3>,
    /// This client's `zwp_text_input_v3`, created by
    /// [`TestClient::map_toplevel_with_text_input`] -- *before* the surface's
    /// first commit, so it is already registered when the compositor's
    /// keyboard-focus relay (`relay_keyboard_focus`, wlr crate) fires the
    /// auto-focus-on-map `enter`. A text-input created only after mapping
    /// (on an already-focused surface) never receives that `enter`, leaving
    /// its wlroots-side `focused_surface` null; a later `leave` (e.g. at
    /// teardown) then trips wlroots' own `wlr_text_input_v3_send_leave`
    /// assertion, since `relay_keyboard_focus` matches outgoing text-inputs
    /// by *client*, not by per-object entered state. `None` for every other
    /// [`TestClient`], which never creates this object at all.
    text_input: Option<zwp_text_input_v3::ZwpTextInputV3>,
    /// How many `zwp_text_input_v3.enter` events this client has received.
    text_input_enters: u32,
    /// How many `zwp_text_input_v3.leave` events this client has received.
    text_input_leaves: u32,
    /// Every `preedit_string` event's text, in arrival order (a `None` text
    /// -- the protocol allows a null preedit string -- is recorded as `""`).
    text_input_preedit_strings: Vec<String>,
    /// Every `commit_string` event's text, in arrival order.
    text_input_commit_strings: Vec<String>,
    /// Every `delete_surrounding_text` event, as `(before_length,
    /// after_length)`, in arrival order.
    text_input_deletes: Vec<(u32, u32)>,
    /// How many `zwp_text_input_v3.done` events this client has received.
    text_input_dones: u32,

    // --- input-method (M6.1) ---
    /// Bound whenever advertised; used by [`InputMethodClient::spawn`].
    input_method_manager: Option<zwp_input_method_manager_v2::ZwpInputMethodManagerV2>,
    /// How many `zwp_input_method_v2.activate` events this client has
    /// received.
    im_activates: u32,
    /// How many `zwp_input_method_v2.deactivate` events this client has
    /// received.
    im_deactivates: u32,
    /// Every `surrounding_text` event's `(text, cursor, anchor)`, in
    /// arrival order.
    im_surroundings: Vec<(String, u32, u32)>,
    /// Every `content_type` event's `(hint, purpose)`, in arrival order.
    im_content_types: Vec<(u32, u32)>,
    /// Every `text_change_cause` event's raw `cause`, in arrival order.
    im_text_change_causes: Vec<u32>,
    /// How many `zwp_input_method_v2.done` events this client has received
    /// -- also the serial [`InputMethodClient::send_commit`] echoes back on
    /// its `commit` request, per the protocol ("the value of the serial
    /// argument must be equal to the number of done events already issued").
    im_dones: u32,
    /// Set true on the input-method's `unavailable` event -- sent when
    /// another input method is already associated with this seat.
    im_unavailable: bool,
    /// The most recent `zwp_input_popup_surface_v2.text_input_rectangle`
    /// event's `(x, y, width, height)`, or `None` until one arrives. This is
    /// the anchor rectangle the compositor told the popup to sit against
    /// (`send_input_popup_rectangle`), the popup half of M6.1 A6.2 test 8.
    im_popup_text_input_rectangle: Option<(i32, i32, i32, i32)>,
    /// How many `zwp_input_method_keyboard_grab_v2.key` events this client has
    /// received -- the count test 9 asserts a grab intercepts.
    im_grab_key_events: u32,
    /// How many `zwp_input_method_keyboard_grab_v2.modifiers` events this
    /// client has received.
    im_grab_modifier_events: u32,
}

impl ClientState {
    /// Grow `popup_geometries`/`popup_acked` so `depth` is addressable.
    fn ensure_popup_depth(&mut self, depth: usize) {
        if self.popup_geometries.len() <= depth {
            self.popup_geometries.resize(depth + 1, None);
        }
        if self.popup_acked.len() <= depth {
            self.popup_acked.resize(depth + 1, false);
        }
    }
}

/// The two terminal `wp_presentation_feedback` events. Task 9's brief: the
/// headless backend may have no presentation clock, so a real client has to
/// accept either as proof presentation feedback actually arrived, not just
/// `presented`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationOutcome {
    Presented,
    Discarded,
}

/// One lock surface this client has created via `get_lock_surface`, plus the
/// shm buffer keepalives its `configure` handler attaches. See
/// `ClientState::lock_surfaces`'s doc.
struct LockSurfaceEntry {
    lock_surface: ext_session_lock_surface_v1::ExtSessionLockSurfaceV1,
    wl_surface: wl_surface::WlSurface,
    /// Kept alive only so the shm file/pool/buffer survive as long as the
    /// compositor may still read them -- their contents are never inspected.
    _buffer_keepalive: Option<(std::fs::File, wl_shm_pool::WlShmPool, wl_buffer::WlBuffer)>,
}

#[derive(Default)]
struct ScreencopyFrameState {
    /// `(format, width, height, stride)` from the `buffer` event.
    params: Option<(wl_shm::Format, u32, u32, u32)>,
    ready: bool,
    failed: bool,
}

impl Dispatch<wl_registry::WlRegistry, ()> for ClientState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            state.globals.push((interface.clone(), name));
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind(name, version.min(4), qh, ()));
                }
                "wl_shm" => state.shm = Some(registry.bind(name, 1, qh, ())),
                "xdg_wm_base" => {
                    state.wm_base = Some(registry.bind(name, version.min(6), qh, ()));
                }
                "zxdg_decoration_manager_v1" => {
                    state.decoration_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell = Some(registry.bind(name, version.min(4), qh, ()));
                }
                "wl_seat" => {
                    state.seat = Some(registry.bind(name, version.min(7), qh, ()));
                }
                "wl_data_device_manager" => {
                    state.data_device_manager = Some(registry.bind(name, version.min(3), qh, ()));
                }
                "zwlr_data_control_manager_v1" => {
                    state.data_control_manager = Some(registry.bind(name, version.min(2), qh, ()));
                }
                "zwp_virtual_keyboard_manager_v1" => {
                    state.virtual_keyboard_manager =
                        Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwlr_virtual_pointer_manager_v1" => {
                    state.virtual_pointer_manager =
                        Some(registry.bind(name, version.min(2), qh, ()));
                }
                "zwp_primary_selection_device_manager_v1" => {
                    state.primary_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wl_output" => {
                    state.output = Some(registry.bind(name, version.min(4), qh, ()));
                }
                "zwlr_screencopy_manager_v1" => {
                    state.screencopy_manager = Some(registry.bind(name, version.min(3), qh, ()));
                }
                "ext_session_lock_manager_v1" => {
                    state.session_lock_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "ext_idle_notifier_v1" => {
                    state.idle_notifier = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_idle_inhibit_manager_v1" => {
                    state.idle_inhibit_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_pointer_constraints_v1" => {
                    state.pointer_constraints = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_relative_pointer_manager_v1" => {
                    state.relative_pointer_manager =
                        Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_pointer_gestures_v1" => {
                    state.gestures_manager = Some(registry.bind(name, version.min(3), qh, ()));
                }
                "wp_viewporter" => {
                    state.viewporter = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wp_fractional_scale_manager_v1" => {
                    state.fractional_scale_manager =
                        Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wp_presentation" => {
                    state.presentation = Some(registry.bind(name, version.min(2), qh, ()));
                }
                "wp_cursor_shape_manager_v1" => {
                    state.cursor_shape_manager = Some(registry.bind(name, version.min(2), qh, ()));
                }
                "xdg_activation_v1" => {
                    state.activation = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwlr_gamma_control_manager_v1" => {
                    state.gamma_control_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_text_input_manager_v3" => {
                    state.text_input_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_input_method_manager_v2" => {
                    state.input_method_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for ClientState {
    fn event(
        _: &mut Self,
        base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // A client that never pongs gets killed by the compositor.
        if let xdg_wm_base::Event::Ping { serial } = event {
            base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for ClientState {
    fn event(
        state: &mut Self,
        surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Ack right here rather than in the pump: an unacked configure wedges
        // every later one the compositor would send (it will not send a new
        // one while the previous is outstanding), which silently breaks any
        // test that resizes, maximizes or fullscreens after mapping.
        if let xdg_surface::Event::Configure { serial } = event {
            surface.ack_configure(serial);
            state.configures = state.configures.saturating_add(1);
        }
    }
}

/// Marker user-data for a popup's `xdg_surface` **and** its `xdg_popup`,
/// carrying the popup's depth in its client's chain (`0` = outermost).
///
/// Without the marker both roles would share `Dispatch<XdgSurface, ()>` and a
/// popup configure would advance [`TestClient::configure_count`], quietly
/// breaking every test that waits on that counter to observe a toplevel
/// change. The depth is what lets a nested chain's levels be asserted
/// separately.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PopupRole(pub(crate) usize);

impl Dispatch<xdg_surface::XdgSurface, PopupRole> for ClientState {
    fn event(
        state: &mut Self,
        surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        role: &PopupRole,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Acked here for the same reason the toplevel's is -- see that impl.
        if let xdg_surface::Event::Configure { serial } = event {
            surface.ack_configure(serial);
            state.ensure_popup_depth(role.0);
            state.popup_acked[role.0] = true;
        }
    }
}

impl Dispatch<xdg_popup::XdgPopup, PopupRole> for ClientState {
    fn event(
        state: &mut Self,
        _: &xdg_popup::XdgPopup,
        event: xdg_popup::Event,
        role: &PopupRole,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_popup::Event::Configure {
                x,
                y,
                width,
                height,
            } => {
                state.ensure_popup_depth(role.0);
                state.popup_geometries[role.0] = Some((x, y, width, height));
            }
            // Latched, never cleared: a dismissed popup is gone for good, and
            // a chain is dismissed whole. The depth is appended in arrival
            // order so the *order* of a chain's teardown is assertable too --
            // see `popup_done_depths`.
            xdg_popup::Event::PopupDone => {
                state.popup_done = true;
                state.popup_done_depths.push(role.0);
            }
            xdg_popup::Event::Repositioned { token } => state.popup_repositioned = Some(token),
            _ => {}
        }
    }
}

delegate_noop!(ClientState: ignore xdg_positioner::XdgPositioner);

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure {
                width,
                height,
                states,
            } => {
                state.configured = Some((width, height));
                state.states = states
                    .chunks_exact(4)
                    .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
            }
            xdg_toplevel::Event::Close => state.closed = true,
            _ => {}
        }
    }
}

impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1,
        event: zxdg_toplevel_decoration_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The mode is recorded raw (`1` client-side, `2` server-side) rather
        // than as the generated enum: the assertion a test wants to make is
        // about the value that crossed the wire.
        if let zxdg_toplevel_decoration_v1::Event::Configure { mode } = event
            && let Ok(mode) = mode.into_result()
        {
            state.decoration_modes.push(mode as u32);
        }
    }
}

impl Dispatch<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        surface: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Acked right here, the same reasoning `xdg_surface`'s `Dispatch`
        // impl gives: an unacked configure wedges every later one, and
        // `map_layer_panel`'s own commit-after-ack sequencing depends on
        // this having already happened by the time it runs.
        if let zwlr_layer_surface_v1::Event::Configure {
            serial,
            width,
            height,
        } = event
        {
            surface.ack_configure(serial);
            state.layer_configured = Some((width, height));
            state.layer_configures += 1;
        }
    }
}

impl Dispatch<wl_data_device::WlDataDevice, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wl_data_device::WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            // A new offer is being introduced; its `offer(mime)` events follow
            // before the `selection`/`enter` that names it. Reset the mime
            // list so it reflects only this offer.
            wl_data_device::Event::DataOffer { .. } => state.offer_mimes.clear(),
            // The clipboard this client would paste from (or `None` if cleared).
            wl_data_device::Event::Selection { id } => state.current_offer = id,
            // A drag entered a surface owned by this client. `offer_mimes`
            // was already populated by the `data_offer`/`offer` events that
            // preceded this one on the wire (same dispatch pass, in order),
            // so accepting the single mime our test sources ever offer is
            // safe here. Real destination clients do the same accept +
            // set_actions dance before the compositor will deliver `drop`.
            wl_data_device::Event::Enter { serial, id, .. } => {
                state.dropped = false;
                if let Some(offer) = id {
                    if let Some(mime) = state.offer_mimes.first().cloned() {
                        offer.accept(serial, Some(mime));
                    }
                    offer.set_actions(
                        wl_data_device_manager::DndAction::Copy,
                        wl_data_device_manager::DndAction::Copy,
                    );
                    state.dnd_offer = Some(offer);
                }
            }
            // The drag session ended (successfully); `read_drag_offer` drives
            // the actual byte transfer from here.
            wl_data_device::Event::Drop => state.dropped = true,
            // wlroots sends `leave` immediately after a successful `drop`
            // too (not only for a drag that left without dropping), so this
            // must not blindly drop the offer: `Drop` is always dispatched
            // first (same wire order, same dispatch pass) and sets
            // `dropped`, so by the time this runs the flag already
            // distinguishes the two cases. Only a "left without dropping"
            // leave invalidates the offer here -- a successful drop's offer
            // stays live for `read_drag_offer`'s `receive`/`finish`.
            wl_data_device::Event::Leave => {
                if !state.dropped {
                    state.dnd_offer = None;
                }
            }
            wl_data_device::Event::Motion { .. } => {}
            _ => {}
        }
    }

    // `data_offer` (opcode 0) introduces a server-created wl_data_offer.
    event_created_child!(ClientState, wl_data_device::WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (wl_data_offer::WlDataOffer, ()),
    ]);
}

impl Dispatch<wl_data_offer::WlDataOffer, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wl_data_offer::WlDataOffer,
        event: wl_data_offer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_data_offer::Event::Offer { mime_type } = event {
            state.offer_mimes.push(mime_type);
        }
    }
}

impl Dispatch<wl_data_source::WlDataSource, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wl_data_source::WlDataSource,
        event: wl_data_source::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The paste side asked for the data on `fd`: write our payload and drop
        // the fd (close), so the reader sees EOF after the bytes.
        if let wl_data_source::Event::Send { mime_type, fd } = event
            && mime_type == state.offered_mime
        {
            let mut f = std::fs::File::from(fd);
            let _ = f.write_all(&state.offered_payload);
            state.source_sends = state.source_sends.saturating_add(1);
        }
    }
}

impl Dispatch<zwlr_data_control_device_v1::ZwlrDataControlDeviceV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwlr_data_control_device_v1::ZwlrDataControlDeviceV1,
        event: zwlr_data_control_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_data_control_device_v1::Event::DataOffer { .. } => {
                state.data_control_mimes.clear()
            }
            zwlr_data_control_device_v1::Event::Selection { id } => state.data_control_offer = id,
            // data-control v2's primary (middle-click) selection, delivered the
            // same focus-less way the clipboard `Selection` is. Stored so the
            // X11<->primary bridge can be read without a focused Wayland client.
            zwlr_data_control_device_v1::Event::PrimarySelection { id } => {
                state.data_control_primary_offer = id
            }
            _ => {}
        }
    }

    // `data_offer` (opcode 0) introduces a server-created data-control offer.
    event_created_child!(ClientState, zwlr_data_control_device_v1::ZwlrDataControlDeviceV1, [
        zwlr_data_control_device_v1::EVT_DATA_OFFER_OPCODE => (zwlr_data_control_offer_v1::ZwlrDataControlOfferV1, ()),
    ]);
}

impl Dispatch<zwlr_data_control_offer_v1::ZwlrDataControlOfferV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwlr_data_control_offer_v1::ZwlrDataControlOfferV1,
        event: zwlr_data_control_offer_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_data_control_offer_v1::Event::Offer { mime_type } = event {
            state.data_control_mimes.push(mime_type);
        }
    }
}

impl Dispatch<zwlr_data_control_source_v1::ZwlrDataControlSourceV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        source: &zwlr_data_control_source_v1::ZwlrDataControlSourceV1,
        event: zwlr_data_control_source_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // A client may own CLIPBOARD and PRIMARY at once via data-control, each
        // with its own source object. Route the `send` to the matching payload
        // by which source fired it (review finding #13) — the primary source
        // answers from `offered_primary_*`, everything else from the clipboard
        // pair — so neither reader ever gets the other's bytes.
        if let zwlr_data_control_source_v1::Event::Send { mime_type, fd } = event {
            let is_primary = state.data_control_primary_source.as_ref() == Some(source);
            let (want_mime, payload) = if is_primary {
                (&state.offered_primary_mime, &state.offered_primary_payload)
            } else {
                (&state.offered_mime, &state.offered_payload)
            };
            if &mime_type == want_mime {
                let mut f = std::fs::File::from(fd);
                let _ = f.write_all(payload);
                state.source_sends = state.source_sends.saturating_add(1);
            }
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for ClientState {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        // Create a keyboard the moment the seat advertises the capability, so
        // that on focus this client receives `wl_keyboard.enter` — the input
        // serial `set_selection` is validated against.
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(caps),
        } = event
        {
            if caps.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                state.keyboard = Some(seat.get_keyboard(qh, ()));
            }
            // Same reasoning as the keyboard above: created the moment the
            // capability appears, so this client is already listening for
            // `enter`/`button` by the time a virtual pointer moves over it —
            // the source of the implicit-grab serial `start_drag` needs.
            if caps.contains(wl_seat::Capability::Pointer) && state.pointer.is_none() {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
            // Gated the same way -- an unconditional `get_touch` is a fatal
            // `wl_seat.get_touch called when no touch capability` protocol
            // error on a seat that has not (yet) advertised touch.
            if caps.contains(wl_seat::Capability::Touch) && state.touch.is_none() {
                state.touch = Some(seat.get_touch(qh, ()));
            }
        }
    }
}

impl Dispatch<wl_touch::WlTouch, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wl_touch::WlTouch,
        event: wl_touch::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // M7: record every touch payload the way the pointer arm records
        // its own -- the `TouchClient` double asserts on these streams.
        // Recording only: nothing here changes protocol behavior, so the
        // M4.2 touch-drag tests (which never read these) are unaffected.
        match event {
            wl_touch::Event::Down {
                serial,
                time: _,
                surface: _,
                id,
                x,
                y,
            } => state.touch_downs.push((id, x, y, serial)),
            wl_touch::Event::Up {
                serial,
                time: _,
                id,
            } => state.touch_ups.push((id, serial)),
            wl_touch::Event::Motion { time: _, id, x, y } => {
                state.touch_motions.push((id, x, y));
            }
            wl_touch::Event::Cancel => {
                state.touch_cancels = state.touch_cancels.saturating_add(1);
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Any serial-bearing pointer event is a valid serial; `button` (the
        // implicit grab `start_drag` validates against) always arrives after
        // `enter`, so the latest one recorded is always the right one.
        match event {
            wl_pointer::Event::Enter {
                serial,
                surface_x,
                surface_y,
                ..
            } => {
                state.last_pointer_serial = Some(serial);
                state.pointer_enters = state.pointer_enters.saturating_add(1);
                state.pointer_enter_position = Some((surface_x, surface_y));
            }
            wl_pointer::Event::Leave { .. } => {
                state.pointer_leaves = state.pointer_leaves.saturating_add(1);
            }
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => state.pointer_motions.push((surface_x, surface_y)),
            wl_pointer::Event::Button {
                serial,
                button,
                state: button_state,
                ..
            } => {
                state.last_pointer_serial = Some(serial);
                let pressed = match button_state {
                    WEnum::Value(wl_pointer::ButtonState::Pressed) => true,
                    WEnum::Value(wl_pointer::ButtonState::Released) => false,
                    other => {
                        panic!("wl_pointer.button carried an unrecognized button_state: {other:?}")
                    }
                };
                state.pointer_buttons.push((button, pressed));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Any serial-bearing keyboard event is a valid serial for set_selection;
        // `enter` (on focus) is the one this harness relies on. The `keymap`
        // event's fd is dropped with the event.
        match event {
            wl_keyboard::Event::Enter { serial, .. } => state.last_serial = Some(serial),
            wl_keyboard::Event::Key { serial, .. } => {
                state.last_serial = Some(serial);
                state.key_events = state.key_events.saturating_add(1);
            }
            _ => {}
        }
    }
}

impl Dispatch<zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1,
        event: zwp_primary_selection_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_primary_selection_device_v1::Event::DataOffer { .. } => state.primary_mimes.clear(),
            zwp_primary_selection_device_v1::Event::Selection { id } => state.primary_offer = id,
            _ => {}
        }
    }

    // `data_offer` (opcode 0) introduces a server-created primary offer.
    event_created_child!(ClientState, zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1, [
        zwp_primary_selection_device_v1::EVT_DATA_OFFER_OPCODE => (zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1, ()),
    ]);
}

impl Dispatch<zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1,
        event: zwp_primary_selection_offer_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_primary_selection_offer_v1::Event::Offer { mime_type } = event {
            state.primary_mimes.push(mime_type);
        }
    }
}

impl Dispatch<zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1,
        event: zwp_primary_selection_source_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_primary_selection_source_v1::Event::Send { mime_type, fd } = event
            && mime_type == state.offered_primary_mime
        {
            let mut f = std::fs::File::from(fd);
            let _ = f.write_all(&state.offered_primary_payload);
            state.source_sends = state.source_sends.saturating_add(1);
        }
    }
}

// wayland-client requires a `Dispatch` impl per bound interface; these carry
// nothing the harness asserts on.
delegate_noop!(ClientState: ignore wl_data_device_manager::WlDataDeviceManager);
delegate_noop!(ClientState: ignore zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1);
delegate_noop!(ClientState: ignore zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1);
delegate_noop!(ClientState: ignore zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1);
delegate_noop!(ClientState: ignore zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1);
delegate_noop!(ClientState: ignore zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1);
delegate_noop!(ClientState: ignore zwlr_data_control_manager_v1::ZwlrDataControlManagerV1);
delegate_noop!(ClientState: ignore wl_compositor::WlCompositor);
delegate_noop!(ClientState: ignore wl_surface::WlSurface);
delegate_noop!(ClientState: ignore wl_shm::WlShm);
delegate_noop!(ClientState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(ClientState: ignore wl_buffer::WlBuffer);
delegate_noop!(ClientState: ignore zxdg_decoration_manager_v1::ZxdgDecorationManagerV1);
delegate_noop!(ClientState: ignore zwlr_layer_shell_v1::ZwlrLayerShellV1);
delegate_noop!(ClientState: ignore wl_output::WlOutput);
delegate_noop!(ClientState: ignore zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1);

impl Dispatch<ZwlrScreencopyFrameV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _frame: &ZwlrScreencopyFrameV1,
        event: zwlr_screencopy_frame_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            // `format` is a WEnum<wl_shm::Format>; keep the shm path only.
            zwlr_screencopy_frame_v1::Event::Buffer {
                format: wayland_client::WEnum::Value(fmt),
                width,
                height,
                stride,
            } => {
                state.screencopy_frame.params = Some((fmt, width, height, stride));
            }
            zwlr_screencopy_frame_v1::Event::Ready { .. } => state.screencopy_frame.ready = true,
            zwlr_screencopy_frame_v1::Event::Failed => state.screencopy_frame.failed = true,
            // `Flags`, `Damage`, `LinuxDmabuf`, `BufferDone` need no bookkeeping
            // here: the capture loop copies after it has seen `Buffer`.
            _ => {}
        }
    }
}

delegate_noop!(ClientState: ignore ext_session_lock_manager_v1::ExtSessionLockManagerV1);

impl Dispatch<ext_session_lock_v1::ExtSessionLockV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &ext_session_lock_v1::ExtSessionLockV1,
        event: ext_session_lock_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            ext_session_lock_v1::Event::Locked => state.session_locked = true,
            ext_session_lock_v1::Event::Finished => state.session_finished = true,
            _ => {}
        }
    }
}

impl Dispatch<ext_session_lock_surface_v1::ExtSessionLockSurfaceV1, ()> for ClientState {
    /// Per the protocol's own doc on `ext_session_lock_surface_v1`: on
    /// `configure`, `ack_configure` then attach an shm buffer of the
    /// configured size and commit -- only then can the compositor consider
    /// this output covered and send `locked`. Done right here, synchronously,
    /// since this handler already has both `state` (for `shm`) and `qh` (to
    /// create the buffer) in hand; nothing outside `Dispatch` needs to drive
    /// it.
    fn event(
        state: &mut Self,
        surface: &ext_session_lock_surface_v1::ExtSessionLockSurfaceV1,
        event: ext_session_lock_surface_v1::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let ext_session_lock_surface_v1::Event::Configure {
            serial,
            width,
            height,
        } = event
        else {
            return;
        };
        surface.ack_configure(serial);
        let Some(entry) = state
            .lock_surfaces
            .iter_mut()
            .find(|e| &e.lock_surface == surface)
        else {
            return;
        };
        let shm = state
            .shm
            .clone()
            .expect("compositor did not advertise wl_shm");
        let (w, h) = (width.max(1) as i32, height.max(1) as i32);
        let (file, pool, buffer) = create_shm_buffer(&shm, qh, w, h);
        entry.wl_surface.attach(Some(&buffer), 0, 0);
        entry.wl_surface.commit();
        entry._buffer_keepalive = Some((file, pool, buffer));
    }
}

delegate_noop!(ClientState: ignore ext_idle_notifier_v1::ExtIdleNotifierV1);

impl Dispatch<ext_idle_notification_v1::ExtIdleNotificationV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &ext_idle_notification_v1::ExtIdleNotificationV1,
        event: ext_idle_notification_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            ext_idle_notification_v1::Event::Idled => state.idle_idled = true,
            ext_idle_notification_v1::Event::Resumed => state.idle_resumed = true,
            _ => {}
        }
    }
}

delegate_noop!(ClientState: ignore zwp_idle_inhibit_manager_v1::ZwpIdleInhibitManagerV1);
delegate_noop!(ClientState: ignore zwp_idle_inhibitor_v1::ZwpIdleInhibitorV1);

// --- pointer-constraints (M4.5) ---
delegate_noop!(ClientState: ignore wl_region::WlRegion);
delegate_noop!(ClientState: ignore zwp_pointer_constraints_v1::ZwpPointerConstraintsV1);
delegate_noop!(ClientState: ignore zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1);
delegate_noop!(ClientState: ignore zwp_pointer_gestures_v1::ZwpPointerGesturesV1);

/// M7: record swipe phases into [`ClientState::gesture_events`]. The full
/// payload (deltas) is client traffic the shell never sees; the log keeps
/// kind, stage and finger count -- the phase triple the e2e vocabulary
/// needs.
impl Dispatch<zwp_pointer_gesture_swipe_v1::ZwpPointerGestureSwipeV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_pointer_gesture_swipe_v1::ZwpPointerGestureSwipeV1,
        event: zwp_pointer_gesture_swipe_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_pointer_gesture_swipe_v1::Event::Begin { fingers, .. } => {
                state.gesture_events.push(RecordedGesture {
                    swipe: true,
                    stage: GestureStage::Begin,
                    fingers,
                });
            }
            zwp_pointer_gesture_swipe_v1::Event::Update { .. } => {
                state.gesture_events.push(RecordedGesture {
                    swipe: true,
                    stage: GestureStage::Update,
                    // An update carries no finger count (only begin does),
                    // so updates record zero rather than a stale value.
                    fingers: 0,
                });
            }
            zwp_pointer_gesture_swipe_v1::Event::End { cancelled, .. } => {
                state.gesture_events.push(RecordedGesture {
                    swipe: true,
                    stage: if cancelled == 0 {
                        GestureStage::End
                    } else {
                        GestureStage::Cancelled
                    },
                    // An end carries no finger count; the count is only
                    // meaningful on begin/update, so ends record zero
                    // rather than a stale value.
                    fingers: 0,
                });
            }
            _ => {}
        }
    }
}

/// M7: record pinch phases, mirroring the swipe arm above.
impl Dispatch<zwp_pointer_gesture_pinch_v1::ZwpPointerGesturePinchV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_pointer_gesture_pinch_v1::ZwpPointerGesturePinchV1,
        event: zwp_pointer_gesture_pinch_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_pointer_gesture_pinch_v1::Event::Begin { fingers, .. } => {
                state.gesture_events.push(RecordedGesture {
                    swipe: false,
                    stage: GestureStage::Begin,
                    fingers,
                });
            }
            zwp_pointer_gesture_pinch_v1::Event::Update { .. } => {
                state.gesture_events.push(RecordedGesture {
                    swipe: false,
                    stage: GestureStage::Update,
                    // An update carries no finger count (only begin does),
                    // so updates record zero rather than a stale value.
                    fingers: 0,
                });
            }
            zwp_pointer_gesture_pinch_v1::Event::End { cancelled, .. } => {
                state.gesture_events.push(RecordedGesture {
                    swipe: false,
                    stage: if cancelled == 0 {
                        GestureStage::End
                    } else {
                        GestureStage::Cancelled
                    },
                    fingers: 0,
                });
            }
            _ => {}
        }
    }
}
// `locked`/`unlocked` and `confined`/`unconfined` carry no data this harness
// asserts on directly -- what matters for T6/T7 is the cursor position
// (`Compositor::cursor_position`) and the relative-motion deltas below, both
// observed independently of these two objects' own events.
delegate_noop!(ClientState: ignore zwp_locked_pointer_v1::ZwpLockedPointerV1);
delegate_noop!(ClientState: ignore zwp_confined_pointer_v1::ZwpConfinedPointerV1);

impl Dispatch<zwp_relative_pointer_v1::ZwpRelativePointerV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_relative_pointer_v1::ZwpRelativePointerV1,
        event: zwp_relative_pointer_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // `dx`/`dy` are the accelerated deltas (vs. `dx_unaccel`/`dy_unaccel`)
        // -- see this module's `PointerConstraintsClient::relative_delta` doc.
        if let zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } = event {
            state.relative_delta.0 += dx;
            state.relative_delta.1 += dy;
            state.relative_motion_events = state.relative_motion_events.saturating_add(1);
        }
    }
}

// --- text-input (M6.1) ---
delegate_noop!(ClientState: ignore zwp_text_input_manager_v3::ZwpTextInputManagerV3);

impl Dispatch<zwp_text_input_v3::ZwpTextInputV3, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_text_input_v3::ZwpTextInputV3,
        event: zwp_text_input_v3::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_text_input_v3::Event::Enter { .. } => {
                state.text_input_enters = state.text_input_enters.saturating_add(1);
            }
            zwp_text_input_v3::Event::Leave { .. } => {
                state.text_input_leaves = state.text_input_leaves.saturating_add(1);
            }
            zwp_text_input_v3::Event::PreeditString { text, .. } => {
                state
                    .text_input_preedit_strings
                    .push(text.unwrap_or_default());
            }
            zwp_text_input_v3::Event::CommitString { text } => {
                state
                    .text_input_commit_strings
                    .push(text.unwrap_or_default());
            }
            zwp_text_input_v3::Event::DeleteSurroundingText {
                before_length,
                after_length,
            } => {
                state.text_input_deletes.push((before_length, after_length));
            }
            zwp_text_input_v3::Event::Done { .. } => {
                state.text_input_dones = state.text_input_dones.saturating_add(1);
            }
            _ => {}
        }
    }
}

// --- input-method (M6.1) ---
delegate_noop!(ClientState: ignore zwp_input_method_manager_v2::ZwpInputMethodManagerV2);

impl Dispatch<zwp_input_method_v2::ZwpInputMethodV2, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_input_method_v2::ZwpInputMethodV2,
        event: zwp_input_method_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_input_method_v2::Event::Activate => {
                state.im_activates = state.im_activates.saturating_add(1);
            }
            zwp_input_method_v2::Event::Deactivate => {
                state.im_deactivates = state.im_deactivates.saturating_add(1);
            }
            zwp_input_method_v2::Event::SurroundingText {
                text,
                cursor,
                anchor,
            } => {
                state.im_surroundings.push((text, cursor, anchor));
            }
            zwp_input_method_v2::Event::TextChangeCause { cause } => {
                state.im_text_change_causes.push(cause.into());
            }
            zwp_input_method_v2::Event::ContentType { hint, purpose } => {
                state.im_content_types.push((hint.into(), purpose.into()));
            }
            zwp_input_method_v2::Event::Done => {
                state.im_dones = state.im_dones.saturating_add(1);
            }
            zwp_input_method_v2::Event::Unavailable => {
                state.im_unavailable = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<zwp_input_popup_surface_v2::ZwpInputPopupSurfaceV2, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_input_popup_surface_v2::ZwpInputPopupSurfaceV2,
        event: zwp_input_popup_surface_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The only event on this interface: the anchor rectangle the compositor
        // placed the popup against. Overwrite (not push): the popup is told its
        // *current* rectangle each time, and test 8 asserts the latest.
        if let zwp_input_popup_surface_v2::Event::TextInputRectangle {
            x,
            y,
            width,
            height,
        } = event
        {
            state.im_popup_text_input_rectangle = Some((x, y, width, height));
        }
    }
}

impl Dispatch<zwp_input_method_keyboard_grab_v2::ZwpInputMethodKeyboardGrabV2, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwp_input_method_keyboard_grab_v2::ZwpInputMethodKeyboardGrabV2,
        event: zwp_input_method_keyboard_grab_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Count the two events test 9 asserts a grab intercepts; keymap and
        // repeat_info carry nothing this harness needs.
        match event {
            zwp_input_method_keyboard_grab_v2::Event::Key { .. } => {
                state.im_grab_key_events = state.im_grab_key_events.saturating_add(1);
            }
            zwp_input_method_keyboard_grab_v2::Event::Modifiers { .. } => {
                state.im_grab_modifier_events = state.im_grab_modifier_events.saturating_add(1);
            }
            _ => {}
        }
    }
}

// --- A2 batch-1 passive protocols (Task 7-9) ---
delegate_noop!(ClientState: ignore wp_viewporter::WpViewporter);
delegate_noop!(ClientState: ignore wp_viewport::WpViewport);
delegate_noop!(ClientState: ignore wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1);
// `clock_id` carries no data this harness asserts on -- the presentation
// clock domain is irrelevant to "did a terminal feedback event arrive".
delegate_noop!(ClientState: ignore wp_presentation::WpPresentation);

impl Dispatch<wp_fractional_scale_v1::WpFractionalScaleV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wp_fractional_scale_v1::WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            state.preferred_scale = Some(scale);
        }
    }
}

impl Dispatch<wp_presentation_feedback::WpPresentationFeedback, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &wp_presentation_feedback::WpPresentationFeedback,
        event: wp_presentation_feedback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // `sync_output` is informational (which output presentation was
        // synced to); only the two destructor events are terminal -- see
        // [`PresentationOutcome`]'s own doc.
        match event {
            wp_presentation_feedback::Event::Presented { .. } => {
                state.presentation_outcome = Some(PresentationOutcome::Presented);
            }
            wp_presentation_feedback::Event::Discarded => {
                state.presentation_outcome = Some(PresentationOutcome::Discarded);
            }
            _ => {}
        }
    }
}

// Neither manager has any event at all, and a cursor-shape device is
// request-only too -- the compositor answers `set_shape` by repainting the
// seat cursor, which a client cannot see. `Compositor::cursor_shape` is how
// the test observes it instead.
delegate_noop!(ClientState: ignore wp_cursor_shape_manager_v1::WpCursorShapeManagerV1);
delegate_noop!(ClientState: ignore wp_cursor_shape_device_v1::WpCursorShapeDeviceV1);
delegate_noop!(ClientState: ignore xdg_activation_v1::XdgActivationV1);
delegate_noop!(ClientState: ignore zwlr_gamma_control_manager_v1::ZwlrGammaControlManagerV1);

impl Dispatch<xdg_activation_token_v1::XdgActivationTokenV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &xdg_activation_token_v1::XdgActivationTokenV1,
        event: xdg_activation_token_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // `done` is the token object's only event and its destructor: the
        // opaque string it carries is the whole point of the handshake, and
        // is what a client hands to whoever should redeem it.
        if let xdg_activation_token_v1::Event::Done { token } = event {
            state.activation_token = Some(token);
        }
    }
}

impl Dispatch<zwlr_gamma_control_v1::ZwlrGammaControlV1, ()> for ClientState {
    fn event(
        state: &mut Self,
        _: &zwlr_gamma_control_v1::ZwlrGammaControlV1,
        event: zwlr_gamma_control_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Exactly one of these two arrives per control object: `gamma_size`
        // with the output's LUT size, or `failed` (the object's destructor)
        // when the compositor cannot hand out gamma control for that output
        // at all -- which for wlroots includes an output whose LUT size is 0.
        match event {
            zwlr_gamma_control_v1::Event::GammaSize { size } => state.gamma_size = Some(size),
            zwlr_gamma_control_v1::Event::Failed => state.gamma_failed = true,
            _ => {}
        }
    }
}

/// Everything an `xdg_positioner` needs, in one value, so a test that opens a
/// popup reads as one statement rather than eight setter calls.
///
/// The fields are the protocol's own: `anchor_rect` is `(x, y, width, height)`
/// in the **parent's window-geometry** coordinates, `size` is the popup's
/// requested size, and `constraint_adjustment` is the bitmask the compositor
/// is allowed to use when the unadjusted position would fall outside the
/// constraint box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PopupSpec {
    pub anchor_rect: (i32, i32, i32, i32),
    pub size: (i32, i32),
    pub anchor: PopupAnchor,
    pub gravity: PopupGravity,
    pub constraint_adjustment: PopupConstraint,
    pub offset: (i32, i32),
    /// `Some(serial)` sends `xdg_popup.grab(seat, serial)` **before** the
    /// first commit, as xdg-shell requires. The serial must come from
    /// [`TestClient::last_pointer_serial`] after a button press.
    pub grab: Option<u32>,
    pub reactive: bool,
}

impl PopupSpec {
    /// A `w` x `h` popup anchored at `(0, 0, 1, 1)` with anchor and gravity
    /// both `BottomLeft`, no constraint adjustment, no grab and not reactive
    /// -- the shape most tests want, with every interesting knob left to the
    /// builders below.
    pub fn new(w: i32, h: i32) -> Self {
        PopupSpec {
            anchor_rect: (0, 0, 1, 1),
            size: (w, h),
            anchor: PopupAnchor::BottomLeft,
            gravity: PopupGravity::BottomLeft,
            constraint_adjustment: PopupConstraint::empty(),
            offset: (0, 0),
            grab: None,
            reactive: false,
        }
    }

    pub fn anchor_rect(self, x: i32, y: i32, w: i32, h: i32) -> Self {
        PopupSpec {
            anchor_rect: (x, y, w, h),
            ..self
        }
    }

    pub fn anchor(self, a: PopupAnchor) -> Self {
        PopupSpec { anchor: a, ..self }
    }

    pub fn gravity(self, g: PopupGravity) -> Self {
        PopupSpec { gravity: g, ..self }
    }

    pub fn constraint(self, c: PopupConstraint) -> Self {
        PopupSpec {
            constraint_adjustment: c,
            ..self
        }
    }

    pub fn grab(self, serial: u32) -> Self {
        PopupSpec {
            grab: Some(serial),
            ..self
        }
    }

    pub fn reactive(self, on: bool) -> Self {
        PopupSpec {
            reactive: on,
            ..self
        }
    }
}

/// The objects backing one `xdg_popup`, held so the popup outlives the call
/// that made it.
///
/// Dropping these fields does *not* destroy the popup: `wayland-client`
/// 0.31 proxies have no `Drop` impl that sends a protocol destroy request,
/// so simply letting a `PopupHandles` go out of scope leaks the objects
/// server-side until the connection itself closes. To retire a popup
/// deliberately, call [`PopupHandles::destroy`] (or go through
/// [`TestClient::detach`], which does so for you). `popup_done` -- what
/// [`TestClient::popup_done`] watches for -- is sent by the compositor
/// on its own initiative; nothing this struct does triggers it.
///
/// The popup is driven all the way to mapped, so it owns a real shm buffer
/// at the size the compositor configured it to.
pub(crate) struct PopupHandles {
    popup: xdg_popup::XdgPopup,
    xdg_surface: xdg_surface::XdgSurface,
    surface: wl_surface::WlSurface,
    positioner: xdg_positioner::XdgPositioner,
    buffer: wl_buffer::WlBuffer,
    pool: wl_shm_pool::WlShmPool,
    /// The shm file stays open for as long as the pool refers to it.
    _shm_file: std::fs::File,
}

impl PopupHandles {
    /// Destroy every object backing this popup, child-to-parent as
    /// xdg-shell requires: the buffer and its pool, then the `xdg_popup`
    /// role object, then its `xdg_surface`, then the `wl_surface` it was
    /// assigned to, then the `xdg_positioner` that placed it.
    pub(crate) fn destroy(self) {
        self.buffer.destroy();
        self.pool.destroy();
        self.popup.destroy();
        self.xdg_surface.destroy();
        self.surface.destroy();
        self.positioner.destroy();
    }
}

/// A real wayland client with exactly one mapped xdg toplevel.
pub struct TestClient {
    // Field order is drop order: the wayland objects go before the queue and
    // the connection that own their backing.
    decoration: Option<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1>,
    buffer: wl_buffer::WlBuffer,
    /// The size the mapped `buffer` was created at -- what
    /// [`TestClient::mapped_size`] reports.
    mapped_size: (i32, i32),
    pool: wl_shm_pool::WlShmPool,
    toplevel: xdg_toplevel::XdgToplevel,
    xdg_surface: xdg_surface::XdgSurface,
    surface: wl_surface::WlSurface,
    /// The shm file stays open for as long as the pool refers to it.
    shm_file: std::fs::File,
    /// The drag-icon surface + its shm backing, created by
    /// [`TestClient::start_drag_text_with_icon`] and held here for the rest
    /// of this client's lifetime so it isn't dropped (and its role revoked)
    /// mid-drag. `None` until that method has been called once.
    icon: Option<(
        wl_surface::WlSurface,
        wl_buffer::WlBuffer,
        wl_shm_pool::WlShmPool,
        std::fs::File,
    )>,
    /// Every live popup this client has opened, outermost first -- the chain
    /// `open_popup`/`open_popup_from_popup` build and `destroy_popup` unwinds.
    /// `TestClient::detach` destroys them topmost-first, as xdg-shell requires.
    popups: Vec<PopupHandles>,
    state: ClientState,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    conn: Connection,
}

/// Connect to `socket` and bind every global this harness knows about.
///
/// Factored out of `TestClient::map` so [`LayerPanelClient::spawn`] can
/// share the identical connect-and-bind sequence -- both clients need the
/// same registry roundtrip, and a second implementation of it would be one
/// more place for the "twice: globals, then binds settle" comment below to
/// drift out of sync with reality.
fn connect_and_bind(
    socket: &str,
) -> (
    Connection,
    EventQueue<ClientState>,
    QueueHandle<ClientState>,
    ClientState,
) {
    let path = runtime_dir().join(socket);
    let stream = UnixStream::connect(&path)
        .unwrap_or_else(|e| panic!("connecting to {}: {e}", path.display()));
    let conn = Connection::from_socket(stream).expect("wayland connection");

    let mut queue = conn.new_event_queue::<ClientState>();
    let qh = queue.handle();
    let mut state = ClientState::default();

    let display = conn.display();
    let _registry = display.get_registry(&qh, ());
    // Twice: the first hears the globals, the second lets the binds (and
    // anything they announce, e.g. `wl_shm.format`) settle.
    queue.roundtrip(&mut state).expect("registry roundtrip");
    queue.roundtrip(&mut state).expect("bind roundtrip");

    // Create this client's data device now — before it is ever focused — so it
    // is already listening when the compositor delivers the selection offer on
    // keyboard-enter. Harmless for clients that never touch the clipboard.
    if let (Some(mgr), Some(seat)) = (state.data_device_manager.as_ref(), state.seat.as_ref()) {
        state.data_device = Some(mgr.get_data_device(seat, &qh, ()));
    }
    // Likewise the data-control device: a clipboard manager observes the seat
    // selection without ever being focused, so it must be listening from
    // connect. Harmless for clients that never read the clipboard.
    if let (Some(mgr), Some(seat)) = (state.data_control_manager.as_ref(), state.seat.as_ref()) {
        state.data_control_device = Some(mgr.get_data_device(seat, &qh, ()));
    }
    if let (Some(mgr), Some(seat)) = (state.primary_manager.as_ref(), state.seat.as_ref()) {
        state.primary_device = Some(mgr.get_device(seat, &qh, ()));
    }
    queue.roundtrip(&mut state).expect("data-device roundtrip");

    (conn, queue, qh, state)
}

/// Connect, complete the registry roundtrip, and return the interface names of
/// every global the compositor advertised. Lets a test assert a global exists
/// without the harness having to bind it.
pub fn advertised_globals(socket: &str) -> Vec<String> {
    let (_conn, _queue, _qh, state) = connect_and_bind(socket);
    state
        .globals
        .into_iter()
        .map(|(interface, _)| interface)
        .collect()
}

/// Read the current clipboard selection that `reader` has been offered, in
/// `mime`, driving `owner` (whose `wl_data_source` supplies the bytes) until it
/// has answered. Returns the transferred bytes.
///
/// The owner writes its whole payload inside one `send` dispatch, before the
/// reader drains the pipe, so the payload must fit the pipe buffer (~64 KiB on
/// Linux). Every selection payload in this suite is a short string; a larger
/// one would need a concurrent read instead of this write-then-read shape.
///
/// The transfer is inherently two-sided: `reader.receive` hands the compositor
/// an fd it forwards to `owner`'s data source as a `send`; `owner` must be
/// pumped for its `Dispatch` to write the payload and close its copy, at which
/// point `reader` sees EOF. A free function rather than a method because it
/// needs both clients at once.
pub fn read_selection(reader: &mut TestClient, owner: &mut TestClient, mime: &str) -> Vec<u8> {
    let offer = reader
        .state
        .current_offer
        .clone()
        .expect("no wl_data_offer was delivered to the reader (is it focused?)");
    let (read_end, write_end) = std::io::pipe().expect("pipe");
    offer.receive(mime.to_string(), write_end.as_fd());
    reader.conn.flush().expect("flush receive");
    // The compositor dups the write end into `owner`'s `send`; drop ours so the
    // owner's copy is the only writer left and EOF is reachable.
    drop(write_end);

    // Drive the owner until its data source has serviced the send (wrote the
    // payload and dropped its fd). Deadline-guarded so a wedged transfer fails
    // rather than hangs.
    let before = owner.state.source_sends;
    let deadline = Instant::now() + TIMEOUT;
    while owner.state.source_sends == before {
        assert!(
            Instant::now() < deadline,
            "the selection owner never serviced a data_source.send within {TIMEOUT:?}"
        );
        owner.pump();
        reader.pump();
        std::thread::sleep(Duration::from_millis(5));
    }

    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut { read_end }, &mut buf).expect("read selection");
    buf
}

/// As [`read_selection`], but reads the drag offer delivered to `dst` on
/// `wl_data_device.enter` (M4.2 drag-and-drop) rather than the selection
/// offer, and `finish`es it afterward -- the same "tell the compositor the
/// transfer completed" step a real DnD destination performs once it has
/// successfully received the data. `src` is the drag's origin client, whose
/// `wl_data_source` supplies the bytes exactly as the selection owner's does.
pub fn read_drag_offer(dst: &mut TestClient, src: &mut TestClient, mime: &str) -> Vec<u8> {
    let offer = dst
        .state
        .dnd_offer
        .clone()
        .expect("no drag offer was delivered to the destination (did the drag ever enter it?)");
    let (read_end, write_end) = std::io::pipe().expect("pipe");
    offer.receive(mime.to_string(), write_end.as_fd());
    dst.conn.flush().expect("flush receive");
    drop(write_end);

    let before = src.state.source_sends;
    let deadline = Instant::now() + TIMEOUT;
    while src.state.source_sends == before {
        assert!(
            Instant::now() < deadline,
            "the drag source never serviced a data_source.send within {TIMEOUT:?}"
        );
        src.pump();
        dst.pump();
        std::thread::sleep(Duration::from_millis(5));
    }

    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut { read_end }, &mut buf).expect("read drag offer");

    offer.finish();
    dst.conn.flush().expect("flush finish");
    let _ = dst.queue.roundtrip(&mut dst.state);

    buf
}

/// As [`read_selection`], but over the primary (middle-click) selection: the
/// reader reads its primary offer, the owner's primary source supplies bytes.
pub fn read_primary(reader: &mut TestClient, owner: &mut TestClient, mime: &str) -> Vec<u8> {
    let offer = reader
        .state
        .primary_offer
        .clone()
        .expect("no primary offer delivered to the reader");
    let (read_end, write_end) = std::io::pipe().expect("pipe");
    offer.receive(mime.to_string(), write_end.as_fd());
    reader.conn.flush().expect("flush receive");
    drop(write_end);

    let before = owner.state.source_sends;
    let deadline = Instant::now() + TIMEOUT;
    while owner.state.source_sends == before {
        assert!(
            Instant::now() < deadline,
            "the primary owner never serviced a send within {TIMEOUT:?}"
        );
        owner.pump();
        reader.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut { read_end }, &mut buf).expect("read primary selection");
    buf
}

/// As [`read_selection`], but the selection owner is a data-control client
/// (whose `zwlr_data_control_source_v1` supplies the bytes). The reader is a
/// focused `wl_data_device` client.
pub fn read_selection_from_data_control(
    reader: &mut TestClient,
    owner: &mut DataControlClient,
    mime: &str,
) -> Vec<u8> {
    let offer = reader
        .state
        .current_offer
        .clone()
        .expect("no wl_data_offer was delivered to the reader");
    let (read_end, write_end) = std::io::pipe().expect("pipe");
    offer.receive(mime.to_string(), write_end.as_fd());
    reader.conn.flush().expect("flush receive");
    drop(write_end);

    let before = owner.source_sends();
    let deadline = Instant::now() + TIMEOUT;
    while owner.source_sends() == before {
        assert!(
            Instant::now() < deadline,
            "the data-control owner never serviced a send within {TIMEOUT:?}"
        );
        owner.pump();
        reader.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut { read_end }, &mut buf).expect("read selection");
    buf
}

/// Build a `w`x`h` shm-backed buffer, solid opaque grey. Shared by
/// `TestClient::map` and [`LayerPanelClient::spawn`] -- both need exactly
/// this to answer their respective compositor-chosen size with a real
/// attach.
fn create_shm_buffer(
    shm: &wl_shm::WlShm,
    qh: &QueueHandle<ClientState>,
    w: i32,
    h: i32,
) -> (std::fs::File, wl_shm_pool::WlShmPool, wl_buffer::WlBuffer) {
    let stride = w * 4;
    let len = (stride * h) as usize;

    let fd: OwnedFd =
        rustix::fs::memfd_create("icedtea-harness-shm", rustix::fs::MemfdFlags::CLOEXEC)
            .expect("memfd_create");
    rustix::fs::ftruncate(&fd, len as u64).expect("ftruncate");
    let mut shm_file = std::fs::File::from(fd);
    // Solid opaque grey. `Xrgb8888` (not `Argb8888`): it is the one format
    // every wlroots renderer is required to advertise.
    shm_file.write_all(&vec![0x80u8; len]).expect("write shm");
    shm_file.flush().expect("flush shm");

    let pool = shm.create_pool(shm_file.as_fd(), len as i32, qh, ());
    let buffer = pool.create_buffer(0, w, h, stride, wl_shm::Format::Xrgb8888, qh, ());
    (shm_file, pool, buffer)
}

/// As [`create_shm_buffer`], but honors a caller-supplied format/stride
/// rather than hardcoding `Xrgb8888` -- the screencopy path must match
/// whatever format/stride the compositor reported on the frame's `buffer`
/// event. Zero-filled rather than pre-painted: the compositor's `copy`
/// overwrites every byte the frame actually captures.
fn create_shm_buffer_format(
    shm: &wl_shm::WlShm,
    qh: &QueueHandle<ClientState>,
    w: i32,
    h: i32,
    stride: i32,
    format: wl_shm::Format,
) -> (std::fs::File, wl_shm_pool::WlShmPool, wl_buffer::WlBuffer) {
    let len = (stride * h) as usize;
    let fd: OwnedFd =
        rustix::fs::memfd_create("icedtea-harness-shm", rustix::fs::MemfdFlags::CLOEXEC)
            .expect("memfd_create");
    rustix::fs::ftruncate(&fd, len as u64).expect("ftruncate");
    let shm_file = std::fs::File::from(fd);
    let pool = shm.create_pool(shm_file.as_fd(), len as i32, qh, ());
    let buffer = pool.create_buffer(0, w, h, stride, format, qh, ());
    (shm_file, pool, buffer)
}

/// A read-back screencopy capture: raw pixel bytes plus the geometry and
/// format wlroots reported for the frame.
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: wl_shm::Format,
    pub bytes: Vec<u8>,
}

/// A `zwlr_screencopy_manager_v1` client that captures the first advertised
/// `wl_output` into a `wl_shm` buffer and reads the pixels back.
pub struct ScreencopyClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    state: ClientState,
}

impl ScreencopyClient {
    /// Connect and bind. Panics if the compositor advertised neither the
    /// screencopy manager nor a `wl_output`.
    pub fn spawn(socket: &str) -> ScreencopyClient {
        let (conn, queue, qh, state) = connect_and_bind(socket);
        assert!(
            state.screencopy_manager.is_some(),
            "compositor did not advertise zwlr_screencopy_manager_v1"
        );
        assert!(state.output.is_some(), "compositor advertised no wl_output");
        ScreencopyClient {
            conn,
            queue,
            qh,
            state,
        }
    }

    /// Capture the output into a `wl_shm` buffer and return its pixels.
    /// Panics on `failed`, on a missing `buffer` event, or on timeout.
    pub fn capture(&mut self) -> CapturedFrame {
        let manager = self.state.screencopy_manager.clone().unwrap();
        let output = self.state.output.clone().unwrap();
        self.state.screencopy_frame = ScreencopyFrameState::default();

        // overlay_cursor = 0: do not composite the cursor into the capture.
        let frame = manager.capture_output(0, &output, &self.qh, ());
        self.conn.flush().expect("flush capture_output");

        // Pump until the `buffer` event lands (geometry/format), bounded.
        for _ in 0..500 {
            self.queue
                .roundtrip(&mut self.state)
                .expect("roundtrip buffer");
            if self.state.screencopy_frame.params.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let (format, width, height, stride) = self
            .state
            .screencopy_frame
            .params
            .expect("screencopy sent no buffer event");

        // Allocate a matching shm buffer, zero-filled, and request the copy.
        let (mmap_file, pool, buffer) = create_shm_buffer_format(
            self.state.shm.as_ref().unwrap(),
            &self.qh,
            width as i32,
            height as i32,
            stride as i32,
            format,
        );
        frame.copy(&buffer);
        self.conn.flush().expect("flush copy");

        for _ in 0..500 {
            self.queue
                .roundtrip(&mut self.state)
                .expect("roundtrip ready");
            if self.state.screencopy_frame.ready || self.state.screencopy_frame.failed {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            !self.state.screencopy_frame.failed,
            "screencopy frame failed"
        );
        assert!(
            self.state.screencopy_frame.ready,
            "screencopy never became ready"
        );

        // Read the pixels back out of the shared file.
        use std::io::{Read, Seek, SeekFrom};
        let mut file = mmap_file;
        file.seek(SeekFrom::Start(0)).expect("seek shm");
        let mut bytes = vec![0u8; (stride * height) as usize];
        file.read_exact(&mut bytes).expect("read shm pixels");

        // Hand every per-capture object back. `wayland-client` proxies do not
        // send a destroy request when the Rust value drops, so without this
        // each `capture()` leaks a `zwlr_screencopy_frame_v1`, a
        // `wl_shm_pool` and a `wl_buffer` server-side -- and, because the pool
        // keeps the compositor's mapping of our memfd alive, one
        // output-sized allocation *in both processes* per call. Three or four
        // captures never showed it; `gallery_gate` polls at 40 Hz for as long
        // as a debug-build first paint takes and leaked ~85 MB/s, which is
        // enough to starve every other test binary `cargo test` runs
        // alongside it.
        buffer.destroy();
        pool.destroy();
        frame.destroy();
        self.conn.flush().expect("flush destroy");

        CapturedFrame {
            width,
            height,
            stride,
            format,
            bytes,
        }
    }
}

impl TestClient {
    /// Connect to `socket`, bind the globals, and map one shm-backed toplevel
    /// for real: commit, await configure, ack, attach, commit.
    pub fn map_toplevel(socket: &str, app_id: &str, title: &str) -> TestClient {
        Self::map(socket, app_id, title, false)
    }

    /// As [`TestClient::map_toplevel`], but the toplevel also gets a
    /// `zxdg_toplevel_decoration_v1` and states no mode preference of its
    /// own -- the "the compositor decides" path a client that defers to the
    /// server takes. Read the compositor's answer with
    /// [`TestClient::decoration_mode`].
    ///
    /// Created here rather than through a `create_decoration()` a test could
    /// call after mapping, because the protocol forbids that outright:
    /// wlroots raises `xdg_toplevel_decoration must not have a buffer at
    /// creation` (an unrecoverable protocol error that kills the
    /// connection) for a decoration created against a surface that already
    /// has one. The decoration therefore has to exist before the very first
    /// buffer attach, which is inside this constructor.
    pub fn map_decorated_toplevel(socket: &str, app_id: &str, title: &str) -> TestClient {
        Self::map(socket, app_id, title, true)
    }

    /// As [`TestClient::map_toplevel`], but also creates a `zwp_text_input_v3`
    /// on the client's seat *before* the surface's first commit -- see
    /// [`ClientState::text_input`]'s doc for why that ordering matters.
    /// [`TextInputClient::spawn`]'s own constructor.
    pub(crate) fn map_toplevel_with_text_input(
        socket: &str,
        app_id: &str,
        title: &str,
    ) -> TestClient {
        Self::map_with(socket, app_id, title, false, |state, qh| {
            if let (Some(mgr), Some(seat)) =
                (state.text_input_manager.as_ref(), state.seat.as_ref())
            {
                state.text_input = Some(mgr.get_text_input(seat, qh, ()));
            }
        })
    }

    fn map(socket: &str, app_id: &str, title: &str, decorated: bool) -> TestClient {
        Self::map_with(socket, app_id, title, decorated, |_, _| {})
    }

    fn map_with(
        socket: &str,
        app_id: &str,
        title: &str,
        decorated: bool,
        pre_map: impl FnOnce(&mut ClientState, &QueueHandle<ClientState>),
    ) -> TestClient {
        let (conn, mut queue, qh, mut state) = connect_and_bind(socket);
        pre_map(&mut state, &qh);

        let compositor = state
            .compositor
            .clone()
            .expect("compositor did not advertise wl_compositor");
        let shm = state
            .shm
            .clone()
            .expect("compositor did not advertise wl_shm");
        let wm_base = state
            .wm_base
            .clone()
            .expect("compositor did not advertise xdg_wm_base");

        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id(app_id.to_string());
        toplevel.set_title(title.to_string());
        let decoration = decorated.then(|| {
            let manager = state
                .decoration_manager
                .clone()
                .expect("compositor did not advertise zxdg_decoration_manager_v1");
            // No `set_mode` call at all: the client states no preference and
            // leaves the decision entirely to the compositor.
            manager.get_toplevel_decoration(&toplevel, &qh, ())
        });
        // The initial-commit handshake: an empty commit, then wait for the
        // compositor's first configure (acked inside the `Dispatch` impl).
        surface.commit();
        conn.flush().expect("flush");

        // `roundtrip`, not `blocking_dispatch`, for the same reason
        // `wait_until` uses it: `blocking_dispatch` returns only once *some*
        // event has been dispatched, so against a compositor that is alive
        // but never answers the initial commit it blocks forever and the
        // deadline below is never re-evaluated — a wedged CI job with no
        // output instead of a 5s assertion failure. A roundtrip's own
        // `wl_display.sync` reply is guaranteed by any live loop, so the
        // deadline stays honest.
        let deadline = Instant::now() + TIMEOUT;
        while state.configures == 0 {
            assert!(
                Instant::now() < deadline,
                "no xdg_surface.configure within {TIMEOUT:?}"
            );
            queue.roundtrip(&mut state).expect("configure roundtrip");
        }

        // A configure of 0x0 means "you choose"; the compositor's
        // `initial_commit` normally sizes us, so this is the safety net.
        let (w, h) = match state.configured {
            Some((w, h)) if w > 0 && h > 0 => (w, h),
            _ => FALLBACK_SIZE,
        };
        let (shm_file, pool, buffer) = create_shm_buffer(&shm, &qh, w, h);

        surface.attach(Some(&buffer), 0, 0);
        surface.damage_buffer(0, 0, w, h);
        surface.commit();
        conn.flush().expect("flush");
        queue.roundtrip(&mut state).expect("map roundtrip");

        TestClient {
            decoration,
            buffer,
            mapped_size: (w, h),
            pool,
            toplevel,
            xdg_surface,
            surface,
            shm_file,
            icon: None,
            popups: Vec::new(),
            state,
            queue,
            qh,
            conn,
        }
    }

    /// Pump the client queue until `pred(self)` holds or `TIMEOUT` elapses.
    /// Returns whether the predicate ever held.
    ///
    /// A `roundtrip` rather than `blocking_dispatch`: a roundtrip always
    /// returns (the compositor answers `wl_display.sync` immediately), which
    /// keeps the deadline honest — `blocking_dispatch` would sit forever on a
    /// compositor that has nothing to say, turning a failed assertion into a
    /// hung test.
    pub fn wait_until(&mut self, pred: impl Fn(&TestClient) -> bool) -> bool {
        self.wait_until_timeout(TIMEOUT, pred)
    }

    /// As [`Self::wait_until`], but with an explicit timeout rather than
    /// `TIMEOUT` -- for a negative assertion ("this must not happen")
    /// where waiting the full default timeout on every green run would slow
    /// the suite down for no benefit.
    pub fn wait_until_timeout(
        &mut self,
        timeout: Duration,
        pred: impl Fn(&TestClient) -> bool,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            self.queue.roundtrip(&mut self.state).expect("roundtrip");
            if pred(self) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Most recent `xdg_toplevel.configure` size.
    pub fn last_configure(&self) -> Option<(i32, i32)> {
        self.state.configured
    }

    /// The size of the buffer actually attached at map time -- the
    /// compositor's configure size, or `FALLBACK_SIZE` if it configured
    /// `0x0`. A caller that derives a press point from the model's snapshot
    /// geometry should check it against this rather than assume the two
    /// agree.
    pub fn mapped_size(&self) -> (i32, i32) {
        self.mapped_size
    }

    /// `xdg_toplevel` states from the most recent configure.
    pub fn states(&self) -> &[u32] {
        &self.state.states
    }

    /// How many `xdg_surface.configure` events this client has seen.
    ///
    /// Monotonic, so it is the one signal that distinguishes "a new
    /// configure arrived" from "the old one still says what I expected".
    /// [`TestClient::wait_until`] checks its predicate *before* pumping, so
    /// any assertion whose expected end state may already hold — a second
    /// maximize, a resize to the same geometry — has to latch this first:
    ///
    /// ```ignore
    /// let n = client.configure_count();
    /// comp.send(/* … */);
    /// assert!(client.wait_until(|c| c.configure_count() > n));
    /// ```
    pub fn configure_count(&self) -> u32 {
        self.state.configures
    }

    /// Whether `xdg_toplevel.close` has arrived.
    pub fn closed(&self) -> bool {
        self.state.closed
    }

    /// The mode from the most recent decoration `configure`: `1` client-side,
    /// `2` server-side, `None` if none has arrived.
    pub fn decoration_mode(&self) -> Option<u32> {
        self.state.decoration_modes.last().copied()
    }

    /// Every decoration mode this client has been sent, in order.
    pub fn decoration_modes(&self) -> &[u32] {
        &self.state.decoration_modes
    }

    pub fn set_title(&mut self, title: &str) {
        self.toplevel.set_title(title.to_string());
        self.surface.commit();
        self.conn.flush().expect("flush");
    }

    pub fn request_maximize(&mut self, on: bool) {
        if on {
            self.toplevel.set_maximized();
        } else {
            self.toplevel.unset_maximized();
        }
        self.conn.flush().expect("flush");
    }

    pub fn request_fullscreen(&mut self, on: bool) {
        if on {
            self.toplevel.set_fullscreen(None);
        } else {
            self.toplevel.unset_fullscreen();
        }
        self.conn.flush().expect("flush");
    }

    /// Ask the compositor to start an interactive move of this toplevel
    /// (`xdg_toplevel.move`), citing `serial` against this client's own seat.
    ///
    /// `serial` must be one the seat genuinely issued to *this* client for a
    /// button press -- [`TestClient::last_pointer_serial`] after a
    /// `VirtualPointerClient` button press over this window's surface. The
    /// compositor's own grab path additionally requires the pointer to still
    /// be down and over this window, so a test that wants the request
    /// honored has to leave the button pressed.
    ///
    /// Panics if the compositor never advertised a `wl_seat`.
    pub fn request_move(&mut self, serial: u32) {
        let seat = self
            .state
            .seat
            .clone()
            .expect("compositor did not advertise wl_seat");
        self.toplevel._move(&seat, serial);
        self.conn.flush().expect("flush move");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// As [`TestClient::request_move`], but `xdg_toplevel.resize` with an
    /// explicit edge set.
    pub fn request_resize(&mut self, serial: u32, edges: ResizeEdge) {
        let seat = self
            .state
            .seat
            .clone()
            .expect("compositor did not advertise wl_seat");
        self.toplevel.resize(&seat, serial, edges);
        self.conn.flush().expect("flush resize");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Unmap without destroying: attach a null buffer and commit, exactly
    /// what a real client does to hide itself while keeping its toplevel
    /// alive (xdg-shell's "you may attach `null` to unmap" — the model row
    /// survives this, unlike [`TestClient::detach`]).
    pub fn unmap(&mut self) {
        self.surface.attach(None, 0, 0);
        self.surface.commit();
        self.conn.flush().expect("flush");
    }

    /// Destroy the toplevel cleanly, in the order xdg-shell requires
    /// (any popup child first, then the toplevel, then its xdg_surface, then
    /// the wl_surface), and flush so the compositor sees it before the
    /// connection goes away.
    pub fn detach(self) {
        let mut this = self;
        // Topmost-first: xdg-shell requires a popup's children destroyed
        // before it.
        while let Some(popup) = this.popups.pop() {
            popup.destroy();
        }
        if let Some(decoration) = &this.decoration {
            decoration.destroy();
        }
        this.buffer.destroy();
        this.pool.destroy();
        this.toplevel.destroy();
        this.xdg_surface.destroy();
        this.surface.destroy();
        this.conn.flush().expect("flush");
        // `roundtrip` so the destroys have provably been processed before the
        // connection drops: a bare flush leaves it racing the socket close,
        // and the two produce different server-side teardown paths. Best
        // effort only -- the connection is going away regardless -- but a
        // failure here is still worth flagging in a debug build rather than
        // silently swallowing it.
        let TestClient { state, queue, .. } = &mut this;
        if let Err(e) = queue.roundtrip(state) {
            debug_assert!(false, "TestClient::detach roundtrip failed: {e}");
        }
    }

    /// Own the clipboard: create a `wl_data_source` advertising `mime` with
    /// `payload`, and `set_selection` it with the last input serial this client
    /// saw (0 on a headless seat with no keyboard). The client must currently
    /// hold keyboard focus for the compositor to honor it.
    pub fn set_selection_text(&mut self, mime: &str, payload: &[u8]) {
        let manager = self
            .state
            .data_device_manager
            .clone()
            .expect("compositor did not advertise wl_data_device_manager");
        let device = self.state.data_device.clone().expect("no data device");
        self.state.offered_mime = mime.to_string();
        self.state.offered_payload = payload.to_vec();
        let source = manager.create_data_source(&self.qh, ());
        source.offer(mime.to_string());
        device.set_selection(Some(&source), self.state.last_serial.unwrap_or(0));
        self.state.data_source = Some(source);
        self.conn.flush().expect("flush set_selection");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Whether this client created its `wl_data_device` (manager + seat both
    /// advertised).
    pub fn has_data_device(&self) -> bool {
        self.state.data_device.is_some()
    }

    /// Build the `xdg_positioner` `spec` describes.
    fn positioner_for(&self, spec: PopupSpec) -> xdg_positioner::XdgPositioner {
        let wm_base = self
            .state
            .wm_base
            .clone()
            .expect("compositor did not advertise xdg_wm_base");
        let positioner = wm_base.create_positioner(&self.qh, ());
        let (w, h) = spec.size;
        positioner.set_size(w, h);
        let (ax, ay, aw, ah) = spec.anchor_rect;
        positioner.set_anchor_rect(ax, ay, aw, ah);
        positioner.set_anchor(spec.anchor);
        positioner.set_gravity(spec.gravity);
        positioner.set_constraint_adjustment(spec.constraint_adjustment);
        positioner.set_offset(spec.offset.0, spec.offset.1);
        if spec.reactive {
            positioner.set_reactive();
        }
        positioner
    }

    /// Open an `xdg_popup` on this client's toplevel and drive it to mapped.
    ///
    /// Sequence, exactly as xdg-shell requires: positioner → `get_popup` →
    /// optional `grab` (before the first commit) → an empty commit → wait for
    /// `xdg_popup.configure` + `xdg_surface.configure` → ack (inside the
    /// `Dispatch` impl) → attach a real shm buffer at the configured size →
    /// commit.
    ///
    /// Panics if a popup from this client is already live (use
    /// [`TestClient::open_popup_from_popup`] for a nested one), on a
    /// non-positive size, or if the compositor never configures within
    /// `TIMEOUT`.
    pub fn open_popup(&mut self, spec: PopupSpec) {
        assert!(
            self.popups.is_empty(),
            "open_popup called with a previous popup still live -- destroy it \
             first (TestClient::destroy_popup) or use open_popup_from_popup"
        );
        let parent = self.xdg_surface.clone();
        self.push_popup(&parent, spec);
    }

    /// Open a child popup of this client's current topmost popup, so a test
    /// can build a nested menu chain.
    ///
    /// Panics if there is no popup to hang it off.
    pub fn open_popup_from_popup(&mut self, spec: PopupSpec) {
        let parent = self
            .popups
            .last()
            .expect("open_popup_from_popup with no popup open")
            .xdg_surface
            .clone();
        self.push_popup(&parent, spec);
    }

    /// The shared body of both `open_popup` entry points.
    fn push_popup(&mut self, parent: &xdg_surface::XdgSurface, spec: PopupSpec) {
        let (w, h) = spec.size;
        assert!(
            w > 0 && h > 0,
            "popup size ({w}, {h}) must be positive -- xdg-shell rejects a \
             zero-sized positioner outright"
        );
        let depth = self.popups.len();
        let compositor = self
            .state
            .compositor
            .clone()
            .expect("compositor did not advertise wl_compositor");
        let shm = self
            .state
            .shm
            .clone()
            .expect("compositor did not advertise wl_shm");
        let wm_base = self
            .state
            .wm_base
            .clone()
            .expect("compositor did not advertise xdg_wm_base");

        let positioner = self.positioner_for(spec);
        let surface = compositor.create_surface(&self.qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &self.qh, PopupRole(depth));
        let popup = xdg_surface.get_popup(Some(parent), &positioner, &self.qh, PopupRole(depth));
        if let Some(serial) = spec.grab {
            let seat = self.state.seat.clone().expect("no wl_seat");
            // Before the first commit, per xdg-shell.
            popup.grab(&seat, serial);
        }
        self.state.ensure_popup_depth(depth);
        self.state.popup_geometries[depth] = None;
        self.state.popup_acked[depth] = false;
        surface.commit();
        self.conn.flush().expect("flush popup create");

        // `roundtrip`, not `blocking_dispatch`, for the reason `map` gives:
        // a roundtrip always returns, so the deadline stays honest against a
        // compositor that has nothing to say.
        let deadline = Instant::now() + TIMEOUT;
        while !(self.state.popup_acked[depth] && self.state.popup_geometries[depth].is_some()) {
            assert!(
                Instant::now() < deadline,
                "no xdg_popup.configure + xdg_surface.configure within {TIMEOUT:?}"
            );
            self.queue
                .roundtrip(&mut self.state)
                .expect("popup configure roundtrip");
        }

        let (_, _, cw, ch) = self.state.popup_geometries[depth].expect("just checked above");
        // A configure of 0x0 would mean the compositor chose nothing; the
        // positioner's own size is the honest fallback.
        let (bw, bh) = if cw > 0 && ch > 0 { (cw, ch) } else { (w, h) };
        let (shm_file, pool, buffer) = create_shm_buffer(&shm, &self.qh, bw, bh);
        surface.attach(Some(&buffer), 0, 0);
        surface.damage_buffer(0, 0, bw, bh);
        surface.commit();
        self.conn.flush().expect("flush popup map");
        self.queue
            .roundtrip(&mut self.state)
            .expect("popup map roundtrip");

        self.popups.push(PopupHandles {
            popup,
            xdg_surface,
            surface,
            positioner,
            buffer,
            pool,
            _shm_file: shm_file,
        });
    }

    /// `(x, y, width, height)` from the outermost popup's last
    /// `xdg_popup.configure`, in the parent's window-geometry coordinates.
    /// `None` until one has arrived.
    pub fn popup_configured(&self) -> Option<(i32, i32, i32, i32)> {
        self.popup_configured_at(0)
    }

    /// Per-depth: `popup_configured_at(0)` is the outermost popup.
    pub fn popup_configured_at(&self, depth: usize) -> Option<(i32, i32, i32, i32)> {
        self.state.popup_geometries.get(depth).copied().flatten()
    }

    /// Whether *any* popup of this client has received
    /// `xdg_popup.popup_done`. Latched: a chain is dismissed whole.
    pub fn popup_done(&self) -> bool {
        self.state.popup_done
    }

    /// The depth of every popup that has received `xdg_popup.popup_done`, in
    /// arrival order -- `[1, 0]` for a two-level chain torn down deepest-first.
    ///
    /// [`popup_done`](Self::popup_done) is a single latched bool and so cannot
    /// distinguish a whole-chain dismissal from a partial one, nor
    /// deepest-first order from shallowest-first. This can.
    pub fn popup_done_depths(&self) -> &[usize] {
        &self.state.popup_done_depths
    }

    /// How many popups this client currently has open.
    pub fn popup_depth(&self) -> usize {
        self.popups.len()
    }

    /// Destroy the topmost popup -- the reverse-creation order xdg-shell
    /// requires. A no-op when there is none.
    pub fn destroy_popup(&mut self) {
        if let Some(popup) = self.popups.pop() {
            popup.destroy();
            self.conn.flush().expect("flush popup destroy");
            let _ = self.queue.roundtrip(&mut self.state);
        }
    }

    /// Send `xdg_popup.reposition(positioner, token)` for the topmost popup.
    ///
    /// The old positioner is destroyed and replaced: xdg-shell discards every
    /// parameter the previous one set, so keeping it alive would only leak an
    /// object. Panics if there is no popup open.
    ///
    /// This only *sends* the request; the answer arrives asynchronously as
    /// `xdg_popup.repositioned` + `xdg_popup.configure`. Wait for it with
    /// `wait_until(|c| c.popup_repositioned() == Some(token))`.
    pub fn reposition_popup(&mut self, spec: PopupSpec, token: u32) {
        assert!(
            !self.popups.is_empty(),
            "reposition_popup with no popup open"
        );
        let positioner = self.positioner_for(spec);
        let handles = self.popups.last_mut().expect("just checked above");
        handles.popup.reposition(&positioner, token);
        let previous = std::mem::replace(&mut handles.positioner, positioner);
        previous.destroy();
        self.conn.flush().expect("flush reposition");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// The token echoed by the most recent `xdg_popup.repositioned`, or
    /// `None` if no reposition has completed.
    pub fn popup_repositioned(&self) -> Option<u32> {
        self.state.popup_repositioned
    }

    /// Open a grabbing popup at the default positioner.
    ///
    /// Kept for `client_protocol.rs`'s implicit-vs-explicit grab regression
    /// test, which predates [`PopupSpec`]; it is exactly
    /// `open_popup(PopupSpec::new(w, h).grab(serial))`.
    ///
    /// `serial` must be one the seat issued this client's `wl_pointer` --
    /// [`TestClient::last_pointer_serial`] after a button press.
    pub fn open_grabbing_popup(&mut self, serial: u32, w: i32, h: i32) {
        self.open_popup(PopupSpec::new(w, h).grab(serial));
    }

    pub fn start_drag_text(&mut self, mime: &str, payload: &[u8], serial: u32) {
        let manager = self
            .state
            .data_device_manager
            .clone()
            .expect("compositor did not advertise wl_data_device_manager");
        let device = self.state.data_device.clone().expect("no data device");
        self.state.offered_mime = mime.to_string();
        self.state.offered_payload = payload.to_vec();
        let source = manager.create_data_source(&self.qh, ());
        source.offer(mime.to_string());
        source.set_actions(wl_data_device_manager::DndAction::Copy);
        device.start_drag(Some(&source), &self.surface, None, serial);
        self.state.data_source = Some(source);
        self.conn.flush().expect("flush start_drag");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// As [`TestClient::start_drag_text`], but with a visible drag icon: a
    /// second `wl_surface` sized `icon_w`x`icon_h`, backed by the same
    /// solid-grey shm buffer machinery `TestClient::map` uses for the
    /// toplevel.
    ///
    /// Order matters here, and was resolved empirically:
    /// `wl_data_device.start_drag` must be the request that assigns
    /// the icon surface its `drag_icon` role *first* -- the icon surface is
    /// created with no buffer attached, handed to `start_drag`, and only
    /// *after* that request is sent does this attach the buffer and commit.
    /// Attaching/committing before `start_drag` (i.e. giving the surface
    /// committed content before it has the drag-icon role) leaves the scene
    /// with no mapped drag-icon node -- `wlr::Runtime::drag_icon_position`
    /// then reads back `None` for the whole drag.
    ///
    /// The icon surface is stashed in `self.icon` so it survives for the
    /// rest of this client's lifetime -- dropping it mid-drag would destroy
    /// the wl_surface object and, with it, the drag_icon role.
    pub fn start_drag_text_with_icon(
        &mut self,
        mime: &str,
        payload: &[u8],
        serial: u32,
        icon_w: i32,
        icon_h: i32,
    ) {
        let manager = self
            .state
            .data_device_manager
            .clone()
            .expect("compositor did not advertise wl_data_device_manager");
        let device = self.state.data_device.clone().expect("no data device");
        let compositor = self.state.compositor.clone().expect("no wl_compositor");
        let shm = self.state.shm.clone().expect("no wl_shm");

        self.state.offered_mime = mime.to_string();
        self.state.offered_payload = payload.to_vec();
        let source = manager.create_data_source(&self.qh, ());
        source.offer(mime.to_string());
        source.set_actions(wl_data_device_manager::DndAction::Copy);

        let icon_surface = compositor.create_surface(&self.qh, ());

        // Assign the drag_icon role first, with no buffer on the surface
        // yet -- see this method's doc for why the order is load-bearing.
        device.start_drag(Some(&source), &self.surface, Some(&icon_surface), serial);
        self.state.data_source = Some(source);
        self.conn.flush().expect("flush start_drag");
        let _ = self.queue.roundtrip(&mut self.state);

        // Now that the surface has the drag_icon role, attach and commit a
        // real buffer -- the scene node only gets live layout coordinates
        // once the icon surface has committed content.
        let (icon_shm_file, icon_pool, icon_buffer) =
            create_shm_buffer(&shm, &self.qh, icon_w, icon_h);
        icon_surface.attach(Some(&icon_buffer), 0, 0);
        icon_surface.damage_buffer(0, 0, icon_w, icon_h);
        icon_surface.commit();
        self.conn.flush().expect("flush icon commit");
        let _ = self.queue.roundtrip(&mut self.state);

        self.icon = Some((icon_surface, icon_buffer, icon_pool, icon_shm_file));
    }

    /// Commit this client's surface -- the trailing half of the
    /// double-buffered `wp_viewport` / `wl_surface` state dance: a queued
    /// `set_source`/`set_destination` (or a fresh `attach_pattern_buffer`)
    /// only takes effect on the next commit.
    pub fn commit(&mut self) {
        self.surface.commit();
        self.conn.flush().expect("flush commit");
    }

    /// Bind `wp_fractional_scale_v1` for this client's surface -- task 8's
    /// preferred-scale test. Panics if the compositor did not advertise
    /// `wp_fractional_scale_manager_v1`.
    pub fn get_fractional_scale(&mut self) -> wp_fractional_scale_v1::WpFractionalScaleV1 {
        let mgr = self
            .state
            .fractional_scale_manager
            .clone()
            .expect("compositor did not advertise wp_fractional_scale_manager_v1");
        let obj = mgr.get_fractional_scale(&self.surface, &self.qh, ());
        self.conn.flush().expect("flush get_fractional_scale");
        obj
    }

    /// The latest `preferred_scale` this client has received (the numerator
    /// of a fraction over 120), or `None` before one has arrived.
    pub fn preferred_scale(&self) -> Option<u32> {
        self.state.preferred_scale
    }

    /// Bind `wp_viewport` for this client's surface -- task 8's crop/scale
    /// test. Panics if the compositor did not advertise `wp_viewporter`.
    pub fn get_viewport(&mut self) -> wp_viewport::WpViewport {
        let viewporter = self
            .state
            .viewporter
            .clone()
            .expect("compositor did not advertise wp_viewporter");
        let obj = viewporter.get_viewport(&self.surface, &self.qh, ());
        self.conn.flush().expect("flush get_viewport");
        obj
    }

    /// Replace this client's surface content with a fresh `w`x`h`
    /// `Xrgb8888` buffer painted per-pixel by `paint(x, y) -> 0x00RRGGBB`,
    /// attach and damage it, but do **not** commit -- a caller that also
    /// needs to queue `wp_viewport` crop/scale state before the compositor
    /// observes the new content calls [`TestClient::commit`] itself once
    /// that state is set too (both are double-buffered surface state,
    /// applied together on the next commit).
    pub fn attach_pattern_buffer(&mut self, w: i32, h: i32, paint: impl Fn(i32, i32) -> u32) {
        let shm = self
            .state
            .shm
            .clone()
            .expect("compositor did not advertise wl_shm");
        let stride = w * 4;
        let len = (stride * h) as usize;
        let fd: OwnedFd = rustix::fs::memfd_create(
            "icedtea-harness-shm-pattern",
            rustix::fs::MemfdFlags::CLOEXEC,
        )
        .expect("memfd_create");
        rustix::fs::ftruncate(&fd, len as u64).expect("ftruncate");
        let mut shm_file = std::fs::File::from(fd);
        let mut bytes = vec![0u8; len];
        for y in 0..h {
            for x in 0..w {
                let color = paint(x, y);
                let o = (y * stride + x * 4) as usize;
                // Xrgb8888 is little-endian bytes B,G,R,X (see
                // `create_shm_buffer`'s own doc).
                bytes[o] = (color & 0xFF) as u8;
                bytes[o + 1] = ((color >> 8) & 0xFF) as u8;
                bytes[o + 2] = ((color >> 16) & 0xFF) as u8;
                bytes[o + 3] = 0;
            }
        }
        shm_file.write_all(&bytes).expect("write pattern shm");
        shm_file.flush().expect("flush pattern shm");
        let pool = shm.create_pool(shm_file.as_fd(), len as i32, &self.qh, ());
        let buffer = pool.create_buffer(0, w, h, stride, wl_shm::Format::Xrgb8888, &self.qh, ());
        self.surface.attach(Some(&buffer), 0, 0);
        self.surface.damage_buffer(0, 0, w, h);
        self.conn.flush().expect("flush attach_pattern_buffer");
        // Replace the keepalives so the previous buffer's shm backing can be
        // dropped once wl_buffer.release confirms the compositor is done
        // with it -- same lifetime contract `TestClient::map` establishes.
        self.buffer = buffer;
        self.pool = pool;
        self.shm_file = shm_file;
    }

    /// Bind `wp_presentation` feedback for this client's surface's current
    /// content submission -- task 9's presentation-feedback test. Panics if
    /// the compositor did not advertise `wp_presentation`.
    pub fn request_presentation_feedback(
        &mut self,
    ) -> wp_presentation_feedback::WpPresentationFeedback {
        let presentation = self
            .state
            .presentation
            .clone()
            .expect("compositor did not advertise wp_presentation");
        let obj = presentation.feedback(&self.surface, &self.qh, ());
        self.conn.flush().expect("flush presentation feedback");
        obj
    }

    /// The terminal `wp_presentation_feedback` event this client's most
    /// recent feedback request has received (`presented` or `discarded`),
    /// or `None` before either has arrived.
    pub fn presentation_outcome(&self) -> Option<PresentationOutcome> {
        self.state.presentation_outcome
    }

    /// Name a cursor image for this client's pointer via
    /// `wp_cursor_shape_device_v1.set_shape` -- task 10's cursor-shape test.
    ///
    /// `serial` must be one the seat issued to this client's `wl_pointer`
    /// (its `enter`, or a `button`): read it off
    /// [`TestClient::last_pointer_serial`] after moving a virtual pointer
    /// over this client's surface. Panics if the compositor did not
    /// advertise `wp_cursor_shape_manager_v1`, or if the seat never gave
    /// this client a pointer (no pointer capability).
    ///
    /// The device object is created fresh per call and explicitly destroyed
    /// before this returns -- it holds no state the compositor reads back,
    /// and leaking one per call would pile up server-side objects across a
    /// test that sets a shape more than once. `set_shape` (and the
    /// `destroy` after it) have been flushed by the time this returns.
    pub fn set_cursor_shape(&mut self, serial: u32, shape: wp_cursor_shape_device_v1::Shape) {
        let mgr = self
            .state
            .cursor_shape_manager
            .clone()
            .expect("compositor did not advertise wp_cursor_shape_manager_v1");
        let pointer = self
            .state
            .pointer
            .clone()
            .expect("the seat never advertised a pointer capability to this client");
        let device = mgr.get_pointer(&pointer, &self.qh, ());
        device.set_shape(serial, shape);
        device.destroy();
        self.conn.flush().expect("flush set_shape");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Mint an `xdg_activation_v1` token from this client and return the
    /// opaque string the compositor answered `done` with -- task 10's
    /// xdg-activation test.
    ///
    /// * `serial`: `Some(s)` calls `set_serial(s, seat)`, which is what makes
    ///   [`wlr::ActivationToken::has_seat`] true on the compositor side --
    ///   the token's only evidence of a real user interaction. `None` leaves
    ///   the token seat-less, the shape a launcher uses when minting a token
    ///   for some other process to redeem later.
    /// * `own_surface`: whether to call `set_surface(this client's surface)`,
    ///   which is what fills the compositor's
    ///   `ActivationToken::requesting_toplevel`.
    ///
    /// Panics if the compositor did not advertise `xdg_activation_v1` or
    /// never answered `done`.
    pub fn create_activation_token(&mut self, serial: Option<u32>, own_surface: bool) -> String {
        let activation = self
            .state
            .activation
            .clone()
            .expect("compositor did not advertise xdg_activation_v1");
        // Cleared first so the wait below cannot succeed on a token left
        // over from an earlier call.
        self.state.activation_token = None;
        let token = activation.get_activation_token(&self.qh, ());
        if let Some(serial) = serial {
            let seat = self
                .state
                .seat
                .clone()
                .expect("compositor did not advertise wl_seat");
            token.set_serial(serial, &seat);
        }
        if own_surface {
            token.set_surface(&self.surface);
        }
        token.commit();
        self.conn.flush().expect("flush token commit");
        assert!(
            self.wait_until(|c| c.state.activation_token.is_some()),
            "no xdg_activation_token_v1.done arrived within {TIMEOUT:?}"
        );
        self.state
            .activation_token
            .clone()
            .expect("just asserted this is Some")
    }

    /// Redeem `token` against *this* client's own surface, via
    /// `xdg_activation_v1.activate`.
    ///
    /// Deliberately only ever this client's own surface: a `wl_surface` is a
    /// per-connection object, so no client can name another's. Handing the
    /// token string across is exactly how the protocol is meant to be used
    /// (the requester mints, the target redeems), and it is what makes the
    /// compositor's requester-vs-target policy testable at all.
    pub fn activate_self(&mut self, token: &str) {
        let activation = self
            .state
            .activation
            .clone()
            .expect("compositor did not advertise xdg_activation_v1");
        activation.activate(token.to_string(), &self.surface);
        self.conn.flush().expect("flush activate");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// The last serial this client saw on `wl_pointer` (enter or button) --
    /// the implicit-grab serial [`TestClient::start_drag_text`] needs. `None`
    /// if the seat has no pointer capability or the pointer never entered
    /// this client's surface.
    pub fn last_pointer_serial(&self) -> Option<u32> {
        self.state.last_pointer_serial
    }

    /// Every `wl_pointer.button` this client has seen, as
    /// `(button_code, pressed)` in arrival order.
    pub fn pointer_buttons(&self) -> &[(u32, bool)] {
        &self.state.pointer_buttons
    }

    /// Every `wl_pointer.motion` this client has seen, in surface-local
    /// coordinates and in arrival order. Excludes the coordinates carried by
    /// `wl_pointer.enter`.
    pub fn pointer_motions(&self) -> &[(f64, f64)] {
        &self.state.pointer_motions
    }

    /// The surface-local coordinates carried by the most recent
    /// `wl_pointer.enter`, or `None` if the pointer never entered.
    pub fn pointer_enter_position(&self) -> Option<(f64, f64)> {
        self.state.pointer_enter_position
    }

    /// How many `wl_pointer.enter` events this client has seen.
    pub fn pointer_enters(&self) -> u32 {
        self.state.pointer_enters
    }

    /// How many `wl_pointer.leave` events this client has seen.
    pub fn pointer_leaves(&self) -> u32 {
        self.state.pointer_leaves
    }

    /// Whether `wl_data_device.drop` has arrived for the drag currently
    /// entering this client's surface.
    pub fn got_drop(&self) -> bool {
        self.state.dropped
    }

    /// Whether a drag offer has been delivered to this client via
    /// `wl_data_device.enter`.
    pub fn has_drag_offer(&self) -> bool {
        self.state.dnd_offer.is_some()
    }

    /// Whether a selection `wl_data_offer` has been delivered to this client.
    pub fn has_selection_offer(&self) -> bool {
        self.state.current_offer.is_some()
    }

    /// Own the primary (middle-click) selection with `payload` under `mime`.
    /// Like [`set_selection_text`](Self::set_selection_text), needs an input
    /// serial and keyboard focus.
    pub fn set_primary_text(&mut self, mime: &str, payload: &[u8]) {
        let manager = self
            .state
            .primary_manager
            .clone()
            .expect("no primary manager");
        let device = self
            .state
            .primary_device
            .clone()
            .expect("no primary device");
        // The native PRIMARY source has its own offer storage, distinct from the
        // native CLIPBOARD's `offered_mime`/`offered_payload` (review finding #13,
        // applied to the native pair too): a single client that owns both with
        // different payloads must feed each reader its own bytes, not the other's.
        self.state.offered_primary_mime = mime.to_string();
        self.state.offered_primary_payload = payload.to_vec();
        let source = manager.create_source(&self.qh, ());
        source.offer(mime.to_string());
        device.set_selection(Some(&source), self.state.last_serial.unwrap_or(0));
        self.state.primary_source = Some(source);
        self.conn.flush().expect("flush set_primary");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Whether a primary-selection offer has been delivered to this client.
    pub fn has_primary_offer(&self) -> bool {
        self.state.primary_offer.is_some()
    }

    /// Whether this client has captured an input serial (via `wl_keyboard`),
    /// which `set_selection` needs. Requires a keyboard on the seat (e.g. an
    /// injected virtual keyboard) and this client to hold focus.
    pub fn has_input_serial(&self) -> bool {
        self.state.last_serial.is_some()
    }

    /// The last serial this client saw on `wl_keyboard` (`enter` or `key`) --
    /// the same value [`TestClient::has_input_serial`] reports the presence
    /// of, exposed for the callers that must actually *pass* it back
    /// (`xdg_activation_token_v1.set_serial`). `None` until the seat has
    /// given this client a keyboard and focused it.
    pub fn last_input_serial(&self) -> Option<u32> {
        self.state.last_serial
    }

    /// Forget the last `wl_keyboard` serial this client captured, so a
    /// following [`TestClient::wait_until`] on
    /// [`TestClient::has_input_serial`] can only succeed on a serial that
    /// arrives *after* this call.
    ///
    /// Review finding (low): a test that focuses this client and then waits
    /// for `has_input_serial()` passes instantly on a serial left over from
    /// an earlier focus -- it asserts nothing about the focus it just
    /// requested. Clearing first turns that wait into a real one.
    pub fn clear_input_serial(&mut self) {
        self.state.last_serial = None;
    }

    /// How many `wl_keyboard.key` events this client has received, ever. The
    /// M4.4 lock-isolation test's load-bearing observable: a key injected via
    /// the virtual keyboard advances this when (and only when) the compositor
    /// actually delivers it to this client's keyboard.
    pub fn key_events(&self) -> u32 {
        self.state.key_events
    }

    /// One `roundtrip`, exposed so a transfer helper can drive an owner client
    /// whose data source must answer `send`.
    pub fn pump(&mut self) {
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// One `roundtrip` that panics if the connection is dead, for tests whose
    /// NEGATIVE assertions ("B was not entered") would otherwise pass vacuously
    /// after a protocol error killed the client.
    pub fn pump_checked(&mut self) {
        self.queue
            .roundtrip(&mut self.state)
            .expect("roundtrip on a live connection");
    }

    /// How many `send` requests this client's source(s) have serviced.
    pub fn source_sends(&self) -> u32 {
        self.state.source_sends
    }

    /// Map a top-anchored `zwlr_layer_shell_v1` panel: `TOP | LEFT | RIGHT`
    /// anchor, `exclusive` reserved along the top edge, real shm-backed
    /// attach once the compositor answers with a size. See
    /// [`LayerPanelClient`] for the returned handle's own API --
    /// `wait_until`/`layer_configure`, mirroring `TestClient`'s own.
    pub fn map_layer_panel(socket: &str, exclusive: i32) -> LayerPanelClient {
        LayerPanelClient::spawn(socket, exclusive)
    }
}

/// A real `zwlr_layer_shell_v1` client, mapped as a top-anchored panel.
///
/// A separate type from [`TestClient`] rather than an `Option`-ified
/// generalization of it: a layer surface has no `xdg_toplevel` (no
/// maximize/fullscreen requests, no decoration, no `states()`), and
/// threading `Option`s for all of that through every one of `TestClient`'s
/// existing toplevel-only methods would make every call site (this crate's
/// whole existing suite) responsible for a case that can never apply to it.
pub struct LayerPanelClient {
    surface: wl_surface::WlSurface,
    layer_surface: zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
    /// The mapped buffer. Re-attached verbatim by [`LayerPanelClient::remap`]
    /// (the panel maps back at the same size it had), and kept alive for as
    /// long as the client is regardless: dropping the pool/buffer/file early
    /// would free the backing memory out from under a compositor that may
    /// still be reading it.
    buffer: wl_buffer::WlBuffer,
    /// The size `buffer` was created at, for `remap`'s damage rectangle.
    size: (i32, i32),
    _pool: wl_shm_pool::WlShmPool,
    _shm_file: std::fs::File,
    state: ClientState,
    queue: EventQueue<ClientState>,
    /// Kept so popups can be created after `spawn` returns -- every proxy
    /// this client makes later needs it.
    qh: QueueHandle<ClientState>,
    conn: Connection,
    /// The popup opened by [`LayerPanelClient::open_popup`], if any. Same
    /// caveat as [`PopupHandles`]: dropping it does not destroy it.
    popup: Option<PopupHandles>,
}

impl LayerPanelClient {
    /// Connect, bind `zwlr_layer_shell_v1`, and map a top-anchored panel
    /// reserving `exclusive` pixels: `set_anchor(TOP | LEFT | RIGHT)`,
    /// `set_exclusive_zone(exclusive)`, `set_size(0, exclusive as u32)`,
    /// commit, wait for `Configure`, ack (inside the `Dispatch` impl), then
    /// a real shm-backed attach at the compositor-chosen size and a second
    /// commit -- the same "commit, await configure, ack, attach, commit"
    /// shape `TestClient::map` follows for a toplevel.
    fn spawn(socket: &str, exclusive: i32) -> LayerPanelClient {
        let (conn, mut queue, qh, mut state) = connect_and_bind(socket);

        let compositor = state
            .compositor
            .clone()
            .expect("compositor did not advertise wl_compositor");
        let shm = state
            .shm
            .clone()
            .expect("compositor did not advertise wl_shm");
        let layer_shell = state
            .layer_shell
            .clone()
            .expect("compositor did not advertise zwlr_layer_shell_v1");

        let surface = compositor.create_surface(&qh, ());
        let layer_surface = layer_shell.get_layer_surface(
            &surface,
            None,
            zwlr_layer_shell_v1::Layer::Top,
            "harness-panel".to_string(),
            &qh,
            (),
        );
        layer_surface.set_anchor(
            zwlr_layer_surface_v1::Anchor::Top
                | zwlr_layer_surface_v1::Anchor::Left
                | zwlr_layer_surface_v1::Anchor::Right,
        );
        layer_surface.set_exclusive_zone(exclusive);
        layer_surface.set_size(0, exclusive as u32);
        surface.commit();
        conn.flush().expect("flush");

        let deadline = Instant::now() + TIMEOUT;
        while state.layer_configured.is_none() {
            assert!(
                Instant::now() < deadline,
                "no zwlr_layer_surface_v1.configure within {TIMEOUT:?}"
            );
            queue.roundtrip(&mut state).expect("configure roundtrip");
        }

        let (w, h) = state.layer_configured.expect("just checked above");
        let (w, h) = if w > 0 && h > 0 {
            (w as i32, h as i32)
        } else {
            FALLBACK_SIZE
        };
        let (shm_file, pool, buffer) = create_shm_buffer(&shm, &qh, w, h);

        surface.attach(Some(&buffer), 0, 0);
        surface.damage_buffer(0, 0, w, h);
        surface.commit();
        conn.flush().expect("flush");
        queue.roundtrip(&mut state).expect("map roundtrip");

        LayerPanelClient {
            surface,
            layer_surface,
            buffer,
            size: (w, h),
            _pool: pool,
            _shm_file: shm_file,
            state,
            queue,
            qh,
            conn,
            popup: None,
        }
    }

    /// Pump the client queue until `pred(self)` holds or `TIMEOUT`
    /// elapses. Mirrors [`TestClient::wait_until`] exactly.
    pub fn wait_until(&mut self, pred: impl Fn(&LayerPanelClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            self.queue.roundtrip(&mut self.state).expect("roundtrip");
            if pred(self) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Every `wl_pointer.button` this panel has seen, as
    /// `(button_code, pressed)` in arrival order. The layer-shell twin of
    /// [`TestClient::pointer_buttons`].
    pub fn pointer_buttons(&self) -> &[(u32, bool)] {
        &self.state.pointer_buttons
    }

    /// Every `wl_pointer.motion` this panel has seen, in surface-local
    /// coordinates. The layer-shell twin of [`TestClient::pointer_motions`].
    pub fn pointer_motions(&self) -> &[(f64, f64)] {
        &self.state.pointer_motions
    }

    /// How many `wl_pointer.enter` events this panel has seen.
    pub fn pointer_enters(&self) -> u32 {
        self.state.pointer_enters
    }

    /// The surface-local coordinates carried by the most recent
    /// `wl_pointer.enter`, or `None` if the pointer never entered.
    pub fn pointer_enter_position(&self) -> Option<(f64, f64)> {
        self.state.pointer_enter_position
    }

    /// How many `wl_pointer.leave` events this panel has seen. The
    /// layer-shell twin of [`TestClient::pointer_leaves`].
    pub fn pointer_leaves(&self) -> u32 {
        self.state.pointer_leaves
    }

    /// The last serial this panel saw on `wl_pointer` (enter or button).
    /// The layer-shell twin of [`TestClient::last_pointer_serial`].
    pub fn last_pointer_serial(&self) -> Option<u32> {
        self.state.last_pointer_serial
    }

    /// Most recent `zwlr_layer_surface_v1.configure` size.
    pub fn layer_configure(&self) -> Option<(i32, i32)> {
        self.state
            .layer_configured
            .map(|(w, h)| (w as i32, h as i32))
    }

    /// Unmap without destroying: attach a null buffer and commit. Mirrors
    /// [`TestClient::unmap`] -- wlr-layer-shell defines the identical
    /// "attach null to unmap, the surface returns to its
    /// right-after-`get_layer_surface` state" contract (`layer.rs`'s own
    /// module doc), and the entry survives on the compositor side so a
    /// remap would find it again.
    pub fn unmap(&mut self) {
        self.surface.attach(None, 0, 0);
        self.surface.commit();
        self.conn.flush().expect("flush");
    }

    /// How many `zwlr_layer_surface_v1.configure` events have arrived, ever.
    /// See `ClientState::layer_configures` for why a remap test cannot use
    /// [`Self::layer_configure`] instead.
    pub fn layer_configure_count(&self) -> u32 {
        self.state.layer_configures
    }

    /// Open an `xdg_popup` parented to this panel and drive it to mapped.
    ///
    /// The layer-shell dance, per `zwlr_layer_shell_v1`'s own protocol xml:
    /// the popup is created with `xdg_surface.get_popup(None, …)` -- a NULL
    /// xdg parent -- and then reparented with
    /// `zwlr_layer_surface_v1.get_popup`, before the popup's initial commit.
    ///
    /// Panics if a popup is already live, on a non-positive size, or if the
    /// compositor never configures within `TIMEOUT`.
    pub fn open_popup(&mut self, spec: PopupSpec) {
        assert!(
            self.popup.is_none(),
            "open_popup called with a previous popup still live"
        );
        let (w, h) = spec.size;
        assert!(w > 0 && h > 0, "popup size ({w}, {h}) must be positive");

        let compositor = self.state.compositor.clone().expect("no wl_compositor");
        let shm = self.state.shm.clone().expect("no wl_shm");
        let wm_base = self
            .state
            .wm_base
            .clone()
            .expect("compositor did not advertise xdg_wm_base");

        let positioner = wm_base.create_positioner(&self.qh, ());
        positioner.set_size(w, h);
        let (ax, ay, aw, ah) = spec.anchor_rect;
        positioner.set_anchor_rect(ax, ay, aw, ah);
        positioner.set_anchor(spec.anchor);
        positioner.set_gravity(spec.gravity);
        positioner.set_constraint_adjustment(spec.constraint_adjustment);
        positioner.set_offset(spec.offset.0, spec.offset.1);
        if spec.reactive {
            positioner.set_reactive();
        }

        let surface = compositor.create_surface(&self.qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &self.qh, PopupRole(0));
        // NULL xdg parent, then reparented onto the layer surface.
        let popup = xdg_surface.get_popup(None, &positioner, &self.qh, PopupRole(0));
        self.layer_surface.get_popup(&popup);
        if let Some(serial) = spec.grab {
            let seat = self.state.seat.clone().expect("no wl_seat");
            popup.grab(&seat, serial);
        }
        self.state.ensure_popup_depth(0);
        self.state.popup_geometries[0] = None;
        self.state.popup_acked[0] = false;
        surface.commit();
        self.conn.flush().expect("flush panel popup create");

        let deadline = Instant::now() + TIMEOUT;
        while !(self.state.popup_acked[0] && self.state.popup_geometries[0].is_some()) {
            assert!(
                Instant::now() < deadline,
                "no popup configure for a layer-shell popup within {TIMEOUT:?}"
            );
            self.queue
                .roundtrip(&mut self.state)
                .expect("panel popup configure roundtrip");
        }

        let (_, _, cw, ch) = self.state.popup_geometries[0].expect("just checked above");
        let (bw, bh) = if cw > 0 && ch > 0 { (cw, ch) } else { (w, h) };
        let (shm_file, pool, buffer) = create_shm_buffer(&shm, &self.qh, bw, bh);
        surface.attach(Some(&buffer), 0, 0);
        surface.damage_buffer(0, 0, bw, bh);
        surface.commit();
        self.conn.flush().expect("flush panel popup map");
        self.queue
            .roundtrip(&mut self.state)
            .expect("panel popup map roundtrip");

        self.popup = Some(PopupHandles {
            popup,
            xdg_surface,
            surface,
            positioner,
            buffer,
            pool,
            _shm_file: shm_file,
        });
    }

    /// `(x, y, width, height)` from this panel's popup's last
    /// `xdg_popup.configure`, in the **panel surface's** coordinates.
    pub fn popup_configured(&self) -> Option<(i32, i32, i32, i32)> {
        self.state.popup_geometries.first().copied().flatten()
    }

    /// Whether this panel's popup has received `xdg_popup.popup_done`.
    pub fn popup_done(&self) -> bool {
        self.state.popup_done
    }

    /// Map again after [`Self::unmap`], following the protocol's own
    /// re-initialization sequence: an empty commit (wlroots cleared the
    /// surface's `initialized` flag on the unmap, so this is a fresh
    /// *initial* commit and the compositor owes a mandatory configure),
    /// then — once that configure has arrived and been acked in the
    /// `Dispatch` impl — the buffer attach and the commit that maps it.
    ///
    /// Returns whether the mandatory configure actually arrived within
    /// `TIMEOUT`. `false` is the exact symptom of final review I1: the
    /// compositor recomputes the identical placement, its storm guard
    /// suppresses the send, and the client waits forever for a configure it
    /// can never map without.
    pub fn remap(&mut self) -> bool {
        let before = self.state.layer_configures;
        self.surface.commit();
        self.conn.flush().expect("flush");
        if !self.wait_until(|c| c.layer_configure_count() > before) {
            return false;
        }
        let (w, h) = self.size;
        self.surface.attach(Some(&self.buffer), 0, 0);
        self.surface.damage_buffer(0, 0, w, h);
        self.surface.commit();
        self.conn.flush().expect("flush");
        true
    }
}

impl Drop for LayerPanelClient {
    fn drop(&mut self) {
        self.layer_surface.destroy();
        self.surface.destroy();
        self.conn.flush().expect("flush");
        let _ = self.queue.roundtrip(&mut self.state);
    }
}

/// Injects a virtual keyboard so the seat gains keyboard capability, which is
/// what lets other clients receive `wl_keyboard.enter` and mint the input
/// serial `wl_data_device.set_selection` requires. The headless backend has no
/// physical input device, so without this the seat advertises no keyboard and
/// `set_selection` can never be validated. Kept alive for the test's duration;
/// dropping it destroys the virtual keyboard and the capability with it.
pub struct VirtualKeyboardClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    state: ClientState,
    vk: zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
    /// Monotonic millisecond timestamp handed to every `key` request; mirrors
    /// `VirtualPointerClient::time` -- the compositor only cares that it does
    /// not go backwards.
    time: u32,
    /// Client-side xkb state, fed every `key` request so the matching
    /// `modifiers` request can be derived. `zwp_virtual_keyboard_v1.key` does
    /// **not** update the compositor's xkb state (wlroots' virtual-keyboard
    /// implementation passes `update_state = false` to
    /// `wlr_keyboard_notify_key`, precisely because the protocol makes the
    /// client the owner of modifier state), so a virtual keyboard that only
    /// ever sends `key` can hold Shift down forever without any client ever
    /// being told a modifier is active.
    xkb_state: xkb::State,
    /// The last `(depressed, latched, locked, group)` sent, so `modifiers` is
    /// only re-sent when it actually changes.
    mods: (u32, u32, u32, u32),
}

/// `wl_keyboard`'s `keymap_format` enum value for `XKB_V1` -- the virtual
/// keyboard protocol's `keymap.format` arg carries this same value (its own
/// xml declares no enum of its own, but reuses `wl_keyboard`'s).
const KEYMAP_FORMAT_XKB_V1: u32 = 1;

/// `wl_keyboard`'s `key_state` enum values -- likewise reused by
/// `zwp_virtual_keyboard_v1.key`'s plain-`uint` `state` arg.
const KEY_STATE_RELEASED: u32 = 0;
const KEY_STATE_PRESSED: u32 = 1;

impl VirtualKeyboardClient {
    /// Connect, create a virtual keyboard on the seat, hand it a minimal
    /// valid "us" xkb keymap (required before any `key` request -- wlroots
    /// refuses `key` with a protocol error until a keymap has been set), and
    /// settle. Panics if the compositor did not advertise
    /// `zwp_virtual_keyboard_manager_v1`.
    pub fn spawn(socket: &str) -> VirtualKeyboardClient {
        let (conn, mut queue, qh, mut state) = connect_and_bind(socket);
        let manager = state
            .virtual_keyboard_manager
            .clone()
            .expect("compositor did not advertise zwp_virtual_keyboard_manager_v1");
        let seat = state.seat.clone().expect("no seat");
        let vk = manager.create_virtual_keyboard(&seat, &qh, ());

        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            "us",
            "",
            None,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .expect("compile a minimal us xkb keymap");
        // Null-terminated, per `wl_keyboard.keymap`'s own contract for the
        // XKB_V1 text format -- `size` below includes that terminator.
        let mut keymap_str = keymap
            .get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1)
            .into_bytes();
        keymap_str.push(0);
        let fd: OwnedFd =
            rustix::fs::memfd_create("icedtea-harness-keymap", rustix::fs::MemfdFlags::CLOEXEC)
                .expect("memfd_create");
        rustix::fs::ftruncate(&fd, keymap_str.len() as u64).expect("ftruncate keymap");
        let mut keymap_file = std::fs::File::from(fd);
        keymap_file.write_all(&keymap_str).expect("write keymap");
        keymap_file.flush().expect("flush keymap");
        vk.keymap(
            KEYMAP_FORMAT_XKB_V1,
            keymap_file.as_fd(),
            keymap_str.len() as u32,
        );
        // The fd is dup'd across the wire by the connection's own send path
        // (wayland-client dups on `flush`), so the client's copy can close
        // right after -- drop it explicitly for clarity.
        drop(keymap_file);

        conn.flush().expect("flush vk create");
        // Two roundtrips so the compositor processes new_virtual_keyboard and
        // the seat's capability change is on the wire before callers connect.
        queue.roundtrip(&mut state).expect("vk roundtrip");
        queue.roundtrip(&mut state).expect("vk settle");
        let xkb_state = xkb::State::new(&keymap);
        VirtualKeyboardClient {
            conn,
            queue,
            state,
            vk,
            time: 0,
            xkb_state,
            mods: (0, 0, 0, 0),
        }
    }

    /// Next monotonic timestamp for a request.
    fn next_time(&mut self) -> u32 {
        self.time = self.time.saturating_add(1);
        self.time
    }

    /// Send one `key` request and the `modifiers` request it implies.
    ///
    /// A real `wl_keyboard` gets its modifier state from the compositor's own
    /// xkb state machine; a virtual keyboard's does not update on `key` (see
    /// [`VirtualKeyboardClient::xkb_state`]), so this mirrors what a real
    /// on-screen keyboard does: run the press through a client-side xkb state
    /// and send `modifiers` whenever the serialised masks change. Ordering
    /// matches a real keyboard's -- the `modifiers` for a Shift press lands
    /// before the next key's `key` event, so the compositor translates that
    /// key with Shift already held.
    fn send_key(&mut self, key: u32, pressed: bool) {
        let time = self.next_time();
        self.vk.key(
            time,
            key,
            if pressed {
                KEY_STATE_PRESSED
            } else {
                KEY_STATE_RELEASED
            },
        );
        // xkb keycodes are evdev keycodes + 8.
        self.xkb_state.update_key(
            xkb::Keycode::new(key + 8),
            if pressed {
                xkb::KeyDirection::Down
            } else {
                xkb::KeyDirection::Up
            },
        );
        let next = (
            self.xkb_state.serialize_mods(xkb::STATE_MODS_DEPRESSED),
            self.xkb_state.serialize_mods(xkb::STATE_MODS_LATCHED),
            self.xkb_state.serialize_mods(xkb::STATE_MODS_LOCKED),
            self.xkb_state.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE),
        );
        if next != self.mods {
            self.mods = next;
            self.vk.modifiers(next.0, next.1, next.2, next.3);
        }
    }

    /// Press then release `key` (a Linux input-event keycode, e.g. `30` for
    /// `KEY_A`) + flush. Requires the keymap `spawn` already set.
    pub fn key_press(&mut self, key: u32) {
        self.send_key(key, true);
        self.send_key(key, false);
        self.conn.flush().expect("flush key_press");
    }

    /// Press `key` and hold it -- no matching release -- + flush. Pairs with
    /// [`Self::key_up`]; a caller wanting to exercise repeat-while-held has
    /// to hold, not press-and-release, since `key_press` never leaves a key
    /// down long enough for a client's repeat timer to fire.
    pub fn key_down(&mut self, key: u32) {
        self.send_key(key, true);
        self.conn.flush().expect("flush key_down");
    }

    /// Release a key previously held with [`Self::key_down`] + flush.
    pub fn key_up(&mut self, key: u32) {
        self.send_key(key, false);
        self.conn.flush().expect("flush key_up");
    }

    /// One roundtrip, to keep the injector responsive during a test.
    pub fn pump(&mut self) {
        let _ = self.queue.roundtrip(&mut self.state);
    }
}

/// A `zwlr_virtual_pointer_manager_v1` client — an on-screen-keyboard-style
/// pointer injector. It mints the pointer motion and button events (and thus
/// the serial) a drag-and-drop grab needs on a headless seat that has no real
/// pointer device.
pub struct VirtualPointerClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    state: ClientState,
    vp: zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
    /// Monotonic millisecond timestamp handed to every request; the
    /// compositor only cares that it does not go backwards, so a simple
    /// incrementing counter is fine for tests.
    time: u32,
}

impl VirtualPointerClient {
    /// Connect, create a virtual pointer on the seat, and settle. Panics if
    /// the compositor did not advertise `zwlr_virtual_pointer_manager_v1`.
    pub fn spawn(socket: &str) -> VirtualPointerClient {
        let (conn, mut queue, qh, mut state) = connect_and_bind(socket);
        let manager = state
            .virtual_pointer_manager
            .clone()
            .expect("compositor did not advertise zwlr_virtual_pointer_manager_v1");
        let seat = state.seat.clone();
        let vp = manager.create_virtual_pointer(seat.as_ref(), &qh, ());
        conn.flush().expect("flush vp create");
        // Two roundtrips so the compositor processes new_virtual_pointer and
        // the seat's capability change is on the wire before callers connect.
        queue.roundtrip(&mut state).expect("vp roundtrip");
        queue.roundtrip(&mut state).expect("vp settle");
        VirtualPointerClient {
            conn,
            queue,
            state,
            vp,
            time: 0,
        }
    }

    /// Next monotonic timestamp for a request.
    fn next_time(&mut self) -> u32 {
        self.time = self.time.saturating_add(1);
        self.time
    }

    /// Move the pointer to an absolute position in a `x_extent` by `y_extent`
    /// coordinate space + flush.
    pub fn motion_absolute(&mut self, x: f64, y: f64, x_extent: u32, y_extent: u32) {
        let time = self.next_time();
        self.vp
            .motion_absolute(time, x as u32, y as u32, x_extent, y_extent);
        self.conn.flush().expect("flush motion_absolute");
    }

    /// Move the pointer by a relative `(dx, dy)` amount, in the global
    /// compositor coordinate space + flush. M4.5's T6/T7 driver: constraint
    /// activation and enforcement both key off relative motion events, not
    /// `motion_absolute`.
    pub fn motion(&mut self, dx: f64, dy: f64) {
        let time = self.next_time();
        self.vp.motion(time, dx, dy);
        self.conn.flush().expect("flush motion");
    }

    /// Press or release a button (Linux input-event code, e.g. `0x110` for
    /// left) + flush.
    pub fn button(&mut self, button: u32, pressed: bool) {
        let time = self.next_time();
        let state = if pressed {
            wl_pointer::ButtonState::Pressed
        } else {
            wl_pointer::ButtonState::Released
        };
        self.vp.button(time, button, state);
        self.conn.flush().expect("flush button");
    }

    /// End the current event sequence + flush.
    pub fn frame(&mut self) {
        self.vp.frame();
        self.conn.flush().expect("flush frame");
    }

    /// Scroll by a continuous amount, in surface-local units + flush.
    ///
    /// `wl_pointer`'s own sign convention: positive is down/right. Callers
    /// send `frame()` themselves, as with every other request here.
    pub fn axis(&mut self, horizontal: f64, vertical: f64) {
        let time = self.next_time();
        if horizontal != 0.0 {
            self.vp
                .axis(time, wl_pointer::Axis::HorizontalScroll, horizontal);
        }
        if vertical != 0.0 {
            self.vp
                .axis(time, wl_pointer::Axis::VerticalScroll, vertical);
        }
        self.conn.flush().expect("flush axis");
    }

    /// Scroll by whole wheel clicks + flush: one click is 10 units, which is
    /// what a real wheel sends alongside its discrete value.
    pub fn axis_discrete(&mut self, horizontal: i32, vertical: i32) {
        let time = self.next_time();
        if horizontal != 0 {
            self.vp.axis_discrete(
                time,
                wl_pointer::Axis::HorizontalScroll,
                f64::from(horizontal) * 10.0,
                horizontal,
            );
        }
        if vertical != 0 {
            self.vp.axis_discrete(
                time,
                wl_pointer::Axis::VerticalScroll,
                f64::from(vertical) * 10.0,
                vertical,
            );
        }
        self.conn.flush().expect("flush axis_discrete");
    }

    /// One roundtrip, to keep the injector responsive during a test.
    pub fn pump(&mut self) {
        let _ = self.queue.roundtrip(&mut self.state);
    }
}

/// A `zwlr_data_control_manager_v1` client — a clipboard manager. It never maps
/// a surface and never holds focus, yet can both set the selection (no input
/// serial required, unlike `wl_data_device`) and read it. This is what makes
/// the selection stack testable on a headless seat that has no input device to
/// mint the serial `wl_data_device.set_selection` would need.
pub struct DataControlClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    state: ClientState,
}

impl DataControlClient {
    /// Connect and bind; the data-control device is created inside
    /// `connect_and_bind`. Panics if the compositor did not advertise
    /// `zwlr_data_control_manager_v1`.
    pub fn spawn(socket: &str) -> DataControlClient {
        let (conn, queue, qh, state) = connect_and_bind(socket);
        assert!(
            state.data_control_device.is_some(),
            "compositor did not advertise zwlr_data_control_manager_v1"
        );
        DataControlClient {
            conn,
            queue,
            qh,
            state,
        }
    }

    /// Own the clipboard with `payload` under `mime`. No serial: data-control
    /// is designed for focus-less clipboard managers.
    pub fn set_clipboard(&mut self, mime: &str, payload: &[u8]) {
        let manager = self
            .state
            .data_control_manager
            .clone()
            .expect("no data-control manager");
        let device = self
            .state
            .data_control_device
            .clone()
            .expect("no data-control device");
        self.state.offered_mime = mime.to_string();
        self.state.offered_payload = payload.to_vec();
        let source = manager.create_data_source(&self.qh, ());
        source.offer(mime.to_string());
        device.set_selection(Some(&source));
        self.state.data_control_source = Some(source);
        self.conn.flush().expect("flush data-control set");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Whether a data-control selection offer has been delivered.
    pub fn has_offer(&self) -> bool {
        self.state.data_control_offer.is_some()
    }

    /// Own the *primary* (middle-click) selection with `payload` under `mime`,
    /// focus-lessly, via data-control v2. The primary counterpart of
    /// [`set_clipboard`](Self::set_clipboard).
    pub fn set_primary(&mut self, mime: &str, payload: &[u8]) {
        let manager = self
            .state
            .data_control_manager
            .clone()
            .expect("no data-control manager");
        let device = self
            .state
            .data_control_device
            .clone()
            .expect("no data-control device");
        // The primary selection's own payload pair, not the clipboard's, so a
        // client that owns both feeds each reader the right bytes (finding #13).
        self.state.offered_primary_mime = mime.to_string();
        self.state.offered_primary_payload = payload.to_vec();
        let source = manager.create_data_source(&self.qh, ());
        source.offer(mime.to_string());
        device.set_primary_selection(Some(&source));
        self.state.data_control_primary_source = Some(source);
        self.conn.flush().expect("flush data-control set_primary");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Whether a data-control *primary* selection offer has been delivered.
    pub fn has_primary_offer(&self) -> bool {
        self.state.data_control_primary_offer.is_some()
    }

    /// Read the current data-control *primary* selection when the owner services
    /// `send` on its own thread (e.g. the X11 selection owner's event loop). The
    /// primary counterpart of [`read_offer_blocking`](Self::read_offer_blocking).
    pub fn read_primary_offer_blocking(&mut self, mime: &str) -> Vec<u8> {
        let offer = self
            .state
            .data_control_primary_offer
            .clone()
            .expect("no data-control primary offer delivered");
        let (read_end, write_end) = std::io::pipe().expect("pipe");
        offer.receive(mime.to_string(), write_end.as_fd());
        self.conn.flush().expect("flush primary receive");
        drop(write_end);

        let handle = std::thread::spawn(move || {
            use std::io::Read as _;
            let mut buf = Vec::new();
            let mut read_end = read_end;
            let _ = read_end.read_to_end(&mut buf);
            buf
        });
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if handle.is_finished() {
                return handle.join().expect("read thread panicked");
            }
            assert!(
                Instant::now() < deadline,
                "primary selection read timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Pump the queue until `pred` holds or `TIMEOUT` elapses.
    pub fn wait_until(&mut self, pred: impl Fn(&DataControlClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            let _ = self.queue.roundtrip(&mut self.state);
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// One roundtrip, so a transfer helper can drive this client as the owner.
    pub fn pump(&mut self) {
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// How many `send` requests this client's source has serviced — lets a
    /// transfer helper know the owner has written the payload.
    pub fn source_sends(&self) -> u32 {
        self.state.source_sends
    }

    /// Read the current data-control selection when the owner services `send`
    /// on its OWN thread (e.g. the `icedtea-clipboard` daemon re-pasting). We
    /// only need to send the receive and read; a spawned thread does the
    /// blocking read so a broken owner fails on the deadline instead of
    /// hanging the test.
    pub fn read_offer_blocking(&mut self, mime: &str) -> Vec<u8> {
        let offer = self
            .state
            .data_control_offer
            .clone()
            .expect("no data-control offer delivered");
        let (read_end, write_end) = std::io::pipe().expect("pipe");
        offer.receive(mime.to_string(), write_end.as_fd());
        self.conn.flush().expect("flush receive");
        drop(write_end);

        let handle = std::thread::spawn(move || {
            use std::io::Read as _;
            let mut buf = Vec::new();
            let mut read_end = read_end;
            let _ = read_end.read_to_end(&mut buf);
            buf
        });
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if handle.is_finished() {
                return handle.join().expect("read thread panicked");
            }
            assert!(
                Instant::now() < deadline,
                "self-served selection read timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Read the current data-control selection when the owner is a *regular*
    /// `wl_data_device` client (an ordinary app that copied). This is the
    /// direction the M4.6 clipboard daemon relies on: an app copies, the
    /// manager observes it. `owner`'s `wl_data_source` supplies the bytes.
    pub fn read_from_wl_data_device_owner(
        &mut self,
        owner: &mut TestClient,
        mime: &str,
    ) -> Vec<u8> {
        let offer = self
            .state
            .data_control_offer
            .clone()
            .expect("no data-control offer delivered");
        let (read_end, write_end) = std::io::pipe().expect("pipe");
        offer.receive(mime.to_string(), write_end.as_fd());
        self.conn.flush().expect("flush receive");
        drop(write_end);

        let before = owner.source_sends();
        let deadline = Instant::now() + TIMEOUT;
        while owner.source_sends() == before {
            assert!(
                Instant::now() < deadline,
                "the wl_data_device owner never serviced a send within {TIMEOUT:?}"
            );
            owner.pump();
            self.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut { read_end }, &mut buf)
            .expect("read data-control selection");
        buf
    }

    /// Read the current data-control selection in `mime`, driving `owner`
    /// (whose source supplies the bytes) until it answers. Mirrors
    /// [`read_selection`] but the reader side is a data-control device.
    pub fn read_selection(&mut self, owner: &mut DataControlClient, mime: &str) -> Vec<u8> {
        let offer = self
            .state
            .data_control_offer
            .clone()
            .expect("no data-control offer delivered");
        let (read_end, write_end) = std::io::pipe().expect("pipe");
        offer.receive(mime.to_string(), write_end.as_fd());
        self.conn.flush().expect("flush receive");
        drop(write_end);

        let before = owner.state.source_sends;
        let deadline = Instant::now() + TIMEOUT;
        while owner.state.source_sends == before {
            assert!(
                Instant::now() < deadline,
                "the data-control owner never serviced a send within {TIMEOUT:?}"
            );
            owner.pump();
            self.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut { read_end }, &mut buf)
            .expect("read data-control selection");
        buf
    }
}

/// An `ext_session_lock_manager_v1` client: takes a session lock, covers
/// every advertised `wl_output` with a lock surface (the protocol's own
/// precondition for the compositor to send `locked` -- see
/// `ext_session_lock_surface_v1`'s Dispatch impl, which does the per-surface
/// ack/attach/commit dance), and can unlock again.
pub struct SessionLockClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    state: ClientState,
    lock: Option<ext_session_lock_v1::ExtSessionLockV1>,
}

impl SessionLockClient {
    /// Connect and bind. Panics if the compositor did not advertise
    /// `ext_session_lock_manager_v1` or a `wl_output`.
    pub fn spawn(socket: &str) -> SessionLockClient {
        let (conn, queue, qh, state) = connect_and_bind(socket);
        assert!(
            state.session_lock_manager.is_some(),
            "compositor did not advertise ext_session_lock_manager_v1"
        );
        assert!(state.output.is_some(), "compositor advertised no wl_output");
        SessionLockClient {
            conn,
            queue,
            qh,
            state,
            lock: None,
        }
    }

    /// Take the lock (`ext_session_lock_manager_v1.lock`), then immediately
    /// create + cover a lock surface for every known `wl_output`, per the
    /// protocol's own recommendation ("Clients should immediately create
    /// lock surfaces for all outputs ... to make this possible"). Each lock
    /// surface's `configure` handler (in `Dispatch`) acks, attaches an shm
    /// buffer, and commits on its own -- this only has to pump the queue
    /// until that has happened.
    pub fn lock(&mut self) {
        let manager = self
            .state
            .session_lock_manager
            .clone()
            .expect("no ext_session_lock_manager_v1");
        let compositor = self.state.compositor.clone().expect("no wl_compositor");
        let output = self.state.output.clone().expect("no wl_output");

        let lock = manager.lock(&self.qh, ());
        let wl_surface = compositor.create_surface(&self.qh, ());
        let lock_surface = lock.get_lock_surface(&wl_surface, &output, &self.qh, ());
        self.state.lock_surfaces.push(LockSurfaceEntry {
            lock_surface,
            wl_surface,
            _buffer_keepalive: None,
        });
        self.lock = Some(lock);
        self.conn.flush().expect("flush lock");

        // Pump until the lock surface's configure has round-tripped through
        // (ack + attach + commit happen inside its own Dispatch handler).
        let deadline = Instant::now() + TIMEOUT;
        while self
            .state
            .lock_surfaces
            .iter()
            .all(|e| e._buffer_keepalive.is_none())
        {
            assert!(Instant::now() < deadline, "lock surface never configured");
            let _ = self.queue.roundtrip(&mut self.state);
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Pump until the `locked` event arrives, bounded by `TIMEOUT`. Returns
    /// whether it did.
    pub fn wait_locked(&mut self) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        while !self.state.session_locked {
            if Instant::now() >= deadline {
                return false;
            }
            let _ = self.queue.roundtrip(&mut self.state);
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    /// Whether `finished` has arrived on the active lock.
    pub fn is_finished(&self) -> bool {
        self.state.session_finished
    }

    /// Unlock: `unlock_and_destroy` on the active lock (per protocol, valid
    /// only after `locked` was received -- exactly [`Self::lock`] + the
    /// [`Self::wait_locked`] precondition every caller is expected to honor).
    /// Per the protocol's own doc on `unlock_and_destroy`, lock surfaces
    /// created through this object should now be destroyed by the client too.
    ///
    /// NOTE (M4.4 task 6): destroying an `ext_session_lock_surface_v1` --
    /// whether via this explicit request, or implicitly via a client
    /// disconnect -- currently aborts the compositor process inside
    /// wlroots' own `lock_surface_destroy` (`types/wlr_session_lock_v1.c:37`,
    /// `Assertion 'wl_list_empty(&lock_surface->events.destroy.listener_list)'
    /// failed`). Reproduces with nothing more than one `get_lock_surface` +
    /// any destruction of it, with or without a commit ever happening; see
    /// the task report for the full isolation. This looks like a listener
    /// bookkeeping bug in the vendored `wlr` crate's
    /// `on_session_lock_new_surface`/`on_session_lock_surface_destroy` (in
    /// `wlroots-sys/crates/wlr/src/backend.rs`), which this harness is not
    /// authorized to modify -- so the destroy calls below are the
    /// protocol-correct thing to do and are kept for when that crate bug is
    /// fixed, even though invoking them currently crashes the test process.
    pub fn unlock(&mut self) {
        let lock = self
            .lock
            .take()
            .expect("unlock called without an active lock");
        lock.unlock_and_destroy();
        for entry in self.state.lock_surfaces.drain(..) {
            entry.lock_surface.destroy();
            entry.wl_surface.destroy();
        }
        self.conn.flush().expect("flush unlock");
        let _ = self.queue.roundtrip(&mut self.state);
    }
}

/// An `ext_idle_notifier_v1` client -- requests idle notifications on its own
/// seat and observes `idled`/`resumed`. M4.4 criterion 4's driver.
pub struct IdleNotifyClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    state: ClientState,
    /// The notification object [`Self::notification`] last created, kept
    /// alive so it keeps delivering events and so a later call can destroy
    /// it before replacing it.
    notification: Option<ext_idle_notification_v1::ExtIdleNotificationV1>,
}

impl IdleNotifyClient {
    /// Connect and bind. Panics if the compositor did not advertise
    /// `ext_idle_notifier_v1` or a `wl_seat`.
    pub fn spawn(socket: &str) -> IdleNotifyClient {
        let (conn, queue, qh, state) = connect_and_bind(socket);
        assert!(
            state.idle_notifier.is_some(),
            "compositor did not advertise ext_idle_notifier_v1"
        );
        assert!(state.seat.is_some(), "compositor advertised no wl_seat");
        IdleNotifyClient {
            conn,
            queue,
            qh,
            state,
            notification: None,
        }
    }

    /// Request a fresh `ext_idle_notification_v1` with `timeout_ms`, on this
    /// client's seat. Destroys and replaces any previous notification object
    /// and resets the `idled`/`resumed` flags, so a stale event from an
    /// earlier notification can never be mistaken for one on this fresh
    /// request -- exactly what letting the idle-inhibit test re-request a
    /// notification after destroying its inhibitor needs.
    pub fn notification(&mut self, timeout_ms: u32) {
        let notifier = self
            .state
            .idle_notifier
            .clone()
            .expect("no ext_idle_notifier_v1");
        let seat = self.state.seat.clone().expect("no wl_seat");
        if let Some(old) = self.notification.take() {
            old.destroy();
        }
        self.state.idle_idled = false;
        self.state.idle_resumed = false;
        let notification = notifier.get_idle_notification(timeout_ms, &seat, &self.qh, ());
        self.notification = Some(notification);
        self.conn.flush().expect("flush get_idle_notification");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Pump bounded by `TIMEOUT` (5s -- generous relative to the short
    /// timeouts these tests request) until `idled` arrives. Returns whether
    /// it did.
    pub fn wait_idled(&mut self) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        while !self.state.idle_idled {
            if Instant::now() >= deadline {
                return false;
            }
            let _ = self.queue.roundtrip(&mut self.state);
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    /// As [`Self::wait_idled`], for `resumed`.
    pub fn wait_resumed(&mut self) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        while !self.state.idle_resumed {
            if Instant::now() >= deadline {
                return false;
            }
            let _ = self.queue.roundtrip(&mut self.state);
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    /// Pump for up to `ms` and report whether `idled` fired during that
    /// window -- used to assert idle did *not* fire within a bound
    /// comfortably longer than the requested timeout (an active inhibitor
    /// suppressing it), which `wait_idled`'s "wait until it happens" shape
    /// cannot express.
    pub fn idled_within(&mut self, ms: u64) -> bool {
        let deadline = Instant::now() + Duration::from_millis(ms);
        loop {
            let _ = self.queue.roundtrip(&mut self.state);
            if self.state.idle_idled {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// A `zwp_idle_inhibit_manager_v1` client -- creates an idle inhibitor on a
/// surface of its own and can destroy it again. M4.4 criterion 5's driver.
/// The inhibited-or-not state it controls is process-wide (any live
/// inhibitor gates every notifier's idle timer, per
/// `Runtime::refresh_idle_inhibited`), so the inhibiting surface need not be
/// mapped, focused, or otherwise visible -- a bare `wl_surface` suffices.
pub struct IdleInhibitClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    state: ClientState,
    surface: wl_surface::WlSurface,
    inhibitor: Option<zwp_idle_inhibitor_v1::ZwpIdleInhibitorV1>,
}

impl IdleInhibitClient {
    /// Connect, bind, and create the surface `create_inhibitor` will attach
    /// an inhibitor to. Panics if the compositor did not advertise
    /// `zwp_idle_inhibit_manager_v1`.
    pub fn spawn(socket: &str) -> IdleInhibitClient {
        let (conn, queue, qh, state) = connect_and_bind(socket);
        assert!(
            state.idle_inhibit_manager.is_some(),
            "compositor did not advertise zwp_idle_inhibit_manager_v1"
        );
        let compositor = state.compositor.clone().expect("no wl_compositor");
        let surface = compositor.create_surface(&qh, ());
        IdleInhibitClient {
            conn,
            queue,
            qh,
            state,
            surface,
            inhibitor: None,
        }
    }

    /// Create an inhibitor on this client's surface. Panics if one is
    /// already active -- callers destroy before creating another.
    pub fn create_inhibitor(&mut self) {
        assert!(self.inhibitor.is_none(), "an inhibitor is already active");
        let manager = self
            .state
            .idle_inhibit_manager
            .clone()
            .expect("no zwp_idle_inhibit_manager_v1");
        let inhibitor = manager.create_inhibitor(&self.surface, &self.qh, ());
        self.inhibitor = Some(inhibitor);
        self.conn.flush().expect("flush create_inhibitor");
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Destroy the active inhibitor. Panics if none is active.
    pub fn destroy_inhibitor(&mut self) {
        let inhibitor = self
            .inhibitor
            .take()
            .expect("no active inhibitor to destroy");
        inhibitor.destroy();
        self.conn.flush().expect("flush destroy_inhibitor");
        let _ = self.queue.roundtrip(&mut self.state);
    }
}

/// A `zwp_pointer_constraints_v1` + `zwp_relative_pointer_manager_v1` client:
/// maps one mapped toplevel (via [`TestClient`], so it can hold pointer
/// focus -- constraints only ever activate on the currently focused
/// surface), creates a `zwp_relative_pointer_v1` on the seat's `wl_pointer`,
/// and can lock or confine the pointer to its surface. M4.5's T6/T7 driver.
pub struct PointerConstraintsClient {
    client: TestClient,
    pointer_constraints: zwp_pointer_constraints_v1::ZwpPointerConstraintsV1,
    /// Kept alive for the client's whole lifetime -- dropping it would stop
    /// relative-motion delivery.
    _relative_pointer: zwp_relative_pointer_v1::ZwpRelativePointerV1,
    /// The active lock, if [`Self::lock_pointer`] has been called. Kept
    /// alive so [`Self::unlock_pointer`] can destroy it explicitly:
    /// dropping the proxy alone does NOT end the lock (wayland-client
    /// only releases the client-side id on drop; the protocol `destroy`
    /// must be sent), so this must not be dropped except through
    /// `unlock_pointer`.
    locked_pointer: Option<zwp_locked_pointer_v1::ZwpLockedPointerV1>,
    /// The active confinement, if [`Self::confine_pointer`] has been called
    /// -- kept so [`Self::set_confine_region`] can update its region.
    confined_pointer: Option<zwp_confined_pointer_v1::ZwpConfinedPointerV1>,
}

impl PointerConstraintsClient {
    /// Connect, map a toplevel, and create the relative pointer on this
    /// client's `wl_pointer`. Panics if the compositor did not advertise
    /// `zwp_pointer_constraints_v1`, `zwp_relative_pointer_manager_v1`, or
    /// the pointer capability.
    pub fn spawn(socket: &str) -> PointerConstraintsClient {
        let client = TestClient::map_toplevel(
            socket,
            "icedtea-harness-pointer-constraints",
            "pointer-constraints",
        );
        let pointer_constraints = client
            .state
            .pointer_constraints
            .clone()
            .expect("compositor did not advertise zwp_pointer_constraints_v1");
        let relative_pointer_manager = client
            .state
            .relative_pointer_manager
            .clone()
            .expect("compositor did not advertise zwp_relative_pointer_manager_v1");
        let pointer = client
            .state
            .pointer
            .clone()
            .expect("compositor advertised no pointer capability");
        let relative_pointer =
            relative_pointer_manager.get_relative_pointer(&pointer, &client.qh, ());
        client.conn.flush().expect("flush get_relative_pointer");

        let mut this = PointerConstraintsClient {
            client,
            pointer_constraints,
            _relative_pointer: relative_pointer,
            locked_pointer: None,
            confined_pointer: None,
        };
        this.pump();
        this
    }

    /// Lock the pointer to its current position on this client's surface
    /// (`Persistent` lifetime -- a lock/confinement this harness drives
    /// across several injected motions should keep working after any
    /// incidental enter/leave rather than going defunct after one). Per the
    /// protocol, a constraint created while the surface is already focused
    /// only activates on the *next* pointer motion, not at creation -- see
    /// the M4.5 design's activation-ordering note, and this module's T6 test.
    pub fn lock_pointer(&mut self) {
        let pointer = self
            .client
            .state
            .pointer
            .clone()
            .expect("compositor advertised no pointer capability");
        let locked = self.pointer_constraints.lock_pointer(
            &self.client.surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Persistent,
            &self.client.qh,
            (),
        );
        self.locked_pointer = Some(locked);
        self.client.conn.flush().expect("flush lock_pointer");
    }

    /// Confine the pointer to a single `x,y,w,h` region, in surface-local
    /// coordinates (`Persistent` lifetime -- see [`Self::lock_pointer`]'s
    /// doc). Keeps the returned `zwp_confined_pointer_v1` so
    /// [`Self::set_confine_region`] can later update its region.
    pub fn confine_pointer(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.confine_pointer_rects(&[(x, y, w, h)]);
    }

    /// As [`Self::confine_pointer`], but the region is built from one or more
    /// (possibly disjoint) rectangles -- the T7 two-rectangle regression
    /// test's shape, which pins the compositor to re-anchoring into one of
    /// the rectangles rather than into the region's bounding-box extents.
    pub fn confine_pointer_rects(&mut self, rects: &[(i32, i32, i32, i32)]) {
        let compositor = self
            .client
            .state
            .compositor
            .clone()
            .expect("no wl_compositor");
        let pointer = self
            .client
            .state
            .pointer
            .clone()
            .expect("compositor advertised no pointer capability");
        let region = compositor.create_region(&self.client.qh, ());
        for &(x, y, w, h) in rects {
            region.add(x, y, w, h);
        }
        let confined = self.pointer_constraints.confine_pointer(
            &self.client.surface,
            &pointer,
            Some(&region),
            zwp_pointer_constraints_v1::Lifetime::Persistent,
            &self.client.qh,
            (),
        );
        region.destroy();
        self.confined_pointer = Some(confined);
        self.client.conn.flush().expect("flush confine_pointer");
    }

    /// Replace the active confinement's region with a fresh single-rect one
    /// and commit the surface -- the T7 re-anchor regression test's driver
    /// (a region move that no longer contains the cursor must re-anchor it
    /// inside the new region). Panics if no confinement is active.
    pub fn set_confine_region(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.set_confine_region_rects(&[(x, y, w, h)]);
    }

    /// As [`Self::set_confine_region`], but the replacement region is built
    /// from one or more rectangles -- see [`Self::confine_pointer_rects`].
    pub fn set_confine_region_rects(&mut self, rects: &[(i32, i32, i32, i32)]) {
        let compositor = self
            .client
            .state
            .compositor
            .clone()
            .expect("no wl_compositor");
        let confined = self
            .confined_pointer
            .as_ref()
            .expect("set_confine_region called without an active confinement")
            .clone();
        let region = compositor.create_region(&self.client.qh, ());
        for &(x, y, w, h) in rects {
            region.add(x, y, w, h);
        }
        confined.set_region(Some(&region));
        region.destroy();
        self.client.surface.commit();
        self.client.conn.flush().expect("flush set_confine_region");
    }

    /// The accumulated `zwp_relative_pointer_v1.relative_motion` deltas
    /// (the accelerated `dx`/`dy`, not `dx_unaccel`/`dy_unaccel`), summed
    /// across every event this client has received so far.
    pub fn relative_delta(&self) -> (f64, f64) {
        self.client.state.relative_delta
    }

    /// How many `relative_motion` events this client has received, ever --
    /// the "did any relative motion arrive at all" signal, since a delta
    /// that nets to exactly zero is indistinguishable from "none" by
    /// [`Self::relative_delta`] alone.
    pub fn relative_motion_events(&self) -> u32 {
        self.client.state.relative_motion_events
    }

    /// One roundtrip.
    pub fn pump(&mut self) {
        let _ = self.client.queue.roundtrip(&mut self.client.state);
    }

    /// Release the active lock, if [`Self::lock_pointer`] was called.
    /// Sends the protocol `destroy` (dropping the proxy alone does NOT
    /// destroy the object -- wayland-client only releases the client-side
    /// id on drop, so a drop-only "unlock" would leave the constraint
    /// active and the cursor frozen) and drops it; the next motion after
    /// that moves the cursor again. Panics if no lock is active.
    pub fn unlock_pointer(&mut self) {
        let locked = self
            .locked_pointer
            .take()
            .expect("unlock_pointer called without an active lock");
        locked.destroy();
        self.client.conn.flush().expect("flush unlock_pointer");
    }
}

/// A `wl_touch` app double: maps a focusable toplevel (via [`TestClient`],
/// so it is the surface under injected touch points) and records every
/// touch payload with its serial (M7). The `TouchClient` half of the
/// brief's pointer/touch/gesture doubles; the pointer half is
/// [`PointerClient`] below.
pub struct TouchClient {
    client: TestClient,
}

impl TouchClient {
    /// Connect and map a toplevel. Panics if the compositor did not
    /// advertise the touch capability (the harness boot enables the
    /// test-touch stand-in, so a missing capability means a broken boot,
    /// not a client problem).
    pub fn spawn(socket: &str, app_id: &str, title: &str) -> TouchClient {
        let client = TestClient::map_toplevel(socket, app_id, title);
        assert!(
            client.state.touch.is_some(),
            "compositor advertised no touch capability"
        );
        TouchClient { client }
    }

    /// Every `wl_touch.down` seen so far, as
    /// `(touch_id, surface_x, surface_y, serial)` in arrival order.
    pub fn touch_downs(&self) -> &[(i32, f64, f64, u32)] {
        &self.client.state.touch_downs
    }

    /// Every `wl_touch.motion` seen so far, as
    /// `(touch_id, surface_x, surface_y)` in arrival order.
    pub fn touch_motions(&self) -> &[(i32, f64, f64)] {
        &self.client.state.touch_motions
    }

    /// Every `wl_touch.up` seen so far, as `(touch_id, serial)` in arrival
    /// order.
    pub fn touch_ups(&self) -> &[(i32, u32)] {
        &self.client.state.touch_ups
    }

    /// How many `wl_touch.cancel` events have arrived, ever.
    pub fn touch_cancels(&self) -> u32 {
        self.client.state.touch_cancels
    }

    /// The mapped surface's last configure size, if any.
    pub fn last_configure(&self) -> Option<(i32, i32)> {
        self.client.last_configure()
    }

    /// One roundtrip.
    pub fn pump(&mut self) {
        self.client.pump();
    }

    /// Pump until `pred` holds or `TIMEOUT` lapses (mirrors
    /// [`TestClient::wait_until`).
    pub fn wait_until(&mut self, pred: impl Fn(&TouchClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            self.client
                .queue
                .roundtrip(&mut self.client.state)
                .expect("roundtrip");
            if pred(self) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// A `zwp_pointer_gestures_v1` client double: binds the gestures manager,
/// holds one swipe and one pinch object on the seat, and records every
/// phase triple (M7). Headless produces no phases (they come from hardware
/// gesture signals), so the log stays empty in tests -- the double exists
/// so the bind path is proven and hardware runs can assert on phases, the
/// same record-everything shape as every other double here.
pub struct GestureClient {
    client: TestClient,
    /// Kept alive for the client's whole lifetime -- dropping a gesture
    /// object stops its phases.
    _swipe: zwp_pointer_gesture_swipe_v1::ZwpPointerGestureSwipeV1,
    /// As `_swipe`, for pinch.
    _pinch: zwp_pointer_gesture_pinch_v1::ZwpPointerGesturePinchV1,
}

impl GestureClient {
    /// Connect, map a toplevel, and create the swipe + pinch objects on
    /// this client's pointer. Panics if the compositor did not advertise
    /// `zwp_pointer_gestures_v1` or the pointer capability -- the latter
    /// needs a pointer device behind the seat (a
    /// [`VirtualPointerClient`] suffices), since the seat only advertises
    /// pointer once a device backs it.
    pub fn spawn(socket: &str) -> GestureClient {
        let client = TestClient::map_toplevel(socket, "icedtea-harness-gestures", "gestures");
        let manager = client
            .state
            .gestures_manager
            .clone()
            .expect("compositor did not advertise zwp_pointer_gestures_v1");
        // Gesture objects hang off the `wl_pointer`, not the seat: the
        // protocol routes phases to the pointer in whose gesture the
        // fingers are.
        let pointer = client
            .state
            .pointer
            .clone()
            .expect("compositor advertised no pointer capability");
        let swipe = manager.get_swipe_gesture(&pointer, &client.qh, ());
        let pinch = manager.get_pinch_gesture(&pointer, &client.qh, ());
        client.conn.flush().expect("flush get_gestures");
        let mut this = GestureClient {
            client,
            _swipe: swipe,
            _pinch: pinch,
        };
        this.pump();
        this
    }

    /// Every gesture phase seen so far, in arrival order.
    pub fn gesture_events(&self) -> &[RecordedGesture] {
        &self.client.state.gesture_events
    }

    /// One roundtrip.
    pub fn pump(&mut self) {
        let _ = self.client.queue.roundtrip(&mut self.client.state);
    }
}

/// A `wl_pointer` observer double: maps a focusable toplevel (via
/// [`TestClient`]) and records every pointer payload with its serial (M7).
/// The `PointerClient` half of the brief's pointer/touch/gesture doubles;
/// injection stays on [`VirtualPointerClient`], recording here.
pub struct PointerClient {
    client: TestClient,
}

impl PointerClient {
    /// Connect and map a toplevel. Panics if the compositor did not
    /// advertise the pointer capability -- which needs a pointer device
    /// behind the seat first (a [`VirtualPointerClient`] suffices), since
    /// the seat only advertises pointer once a device backs it.
    pub fn spawn(socket: &str, app_id: &str, title: &str) -> PointerClient {
        let client = TestClient::map_toplevel(socket, app_id, title);
        assert!(
            client.state.pointer.is_some(),
            "compositor advertised no pointer capability"
        );
        PointerClient { client }
    }

    /// How many `wl_pointer.enter` events have arrived, ever.
    pub fn pointer_enters(&self) -> u32 {
        self.client.pointer_enters()
    }

    /// The surface-local coordinates of the most recent enter, if any.
    pub fn pointer_enter_position(&self) -> Option<(f64, f64)> {
        self.client.pointer_enter_position()
    }

    /// Every `wl_pointer.motion` seen so far, in surface-local coordinates.
    pub fn pointer_motions(&self) -> &[(f64, f64)] {
        self.client.pointer_motions()
    }

    /// Every `wl_pointer.button` seen so far, as `(button_code, pressed)`.
    pub fn pointer_buttons(&self) -> &[(u32, bool)] {
        self.client.pointer_buttons()
    }

    /// The last input-event serial seen on the pointer, if any.
    pub fn last_pointer_serial(&self) -> Option<u32> {
        self.client.last_pointer_serial()
    }

    /// One roundtrip.
    pub fn pump(&mut self) {
        self.client.pump();
    }

    /// Pump until `pred` holds or `TIMEOUT` lapses (mirrors
    /// [`TestClient::wait_until`).
    pub fn wait_until(&mut self, pred: impl Fn(&PointerClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            self.client
                .queue
                .roundtrip(&mut self.client.state)
                .expect("roundtrip");
            if pred(self) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// A `zwp_text_input_v3` app double: maps a focusable toplevel (via
/// [`TestClient`], so it gains keyboard focus via the existing
/// auto-focus-on-map path), creates a text-input on the seat, and records
/// relay events (enter/leave/preedit/commit/delete/done). M6.1 tests 2-5's
/// app side.
pub struct TextInputClient {
    client: TestClient,
    text_input: zwp_text_input_v3::ZwpTextInputV3,
}

impl TextInputClient {
    /// Connect, create a `zwp_text_input_v3` on this client's seat, then map
    /// a toplevel -- in that order, so the text-input is already registered
    /// when the compositor's auto-focus-on-map path fires `enter` (see
    /// [`ClientState::text_input`]'s doc). Panics if the compositor did not
    /// advertise `zwp_text_input_manager_v3` or a seat.
    pub fn spawn(socket: &str) -> TextInputClient {
        let client = TestClient::map_toplevel_with_text_input(
            socket,
            "icedtea-harness-text-input",
            "text-input",
        );
        let text_input = client
            .state
            .text_input
            .clone()
            .expect("compositor did not advertise zwp_text_input_manager_v3 or wl_seat");

        let mut this = TextInputClient { client, text_input };
        this.pump();
        this
    }

    /// As [`Self::spawn`], but the `zwp_text_input_v3` is created *after* the
    /// toplevel is mapped and already keyboard-focused, rather than before its
    /// first commit. This is the exact scenario the wlr relay's
    /// enter-on-create-if-focused fix and its `leave` guard close: a
    /// text-input born onto an already-focused surface must still receive
    /// `enter` (wlroots' `relay_keyboard_focus` only sends `enter` on a
    /// focus *change*, so the crate has to synthesise it on create), and a
    /// later focus change or teardown must not trip wlroots'
    /// `wlr_text_input_v3_send_leave` assertion -- which SIGABRTs the whole
    /// compositor process. Panics if the compositor did not advertise
    /// `zwp_text_input_manager_v3` or a seat.
    pub fn spawn_on_focused_surface(socket: &str) -> TextInputClient {
        let mut client =
            TestClient::map_toplevel(socket, "icedtea-harness-text-input", "text-input");
        let manager = client
            .state
            .text_input_manager
            .clone()
            .expect("compositor did not advertise zwp_text_input_manager_v3");
        let seat = client
            .state
            .seat
            .clone()
            .expect("compositor did not advertise wl_seat");
        let text_input = manager.get_text_input(&seat, &client.qh, ());
        client.state.text_input = Some(text_input.clone());
        client.conn.flush().expect("flush get_text_input");

        let mut this = TextInputClient { client, text_input };
        this.pump();
        this
    }

    /// Enable text input on the current surface and commit.
    pub fn enable(&mut self) {
        self.text_input.enable();
        self.text_input.commit();
        self.flush();
    }

    /// Set the surrounding text, cursor rectangle, and commit -- the
    /// double-buffered state a real IME-aware app sends whenever its text
    /// state changes.
    pub fn commit_with(
        &mut self,
        surrounding: &str,
        cursor: u32,
        anchor: u32,
        cursor_rect: (i32, i32, i32, i32),
    ) {
        self.text_input
            .set_surrounding_text(surrounding.to_string(), cursor as i32, anchor as i32);
        self.text_input.set_cursor_rectangle(
            cursor_rect.0,
            cursor_rect.1,
            cursor_rect.2,
            cursor_rect.3,
        );
        self.text_input.commit();
        self.flush();
    }

    /// Disable text input on the current surface and commit.
    pub fn disable(&mut self) {
        self.text_input.disable();
        self.text_input.commit();
        self.flush();
    }

    /// Destroy ONLY the `zwp_text_input_v3` object -- the
    /// `zwp_text_input_v3.destroy` request -- while leaving the toplevel
    /// mapped and keyboard-focused and the client connection alive. This is
    /// the exact wire event that drives wlroots' `on_text_input_destroy`,
    /// deliberately kept distinct from unmapping the toplevel or dropping the
    /// whole client (either of which would move keyboard focus and take the
    /// `relay_keyboard_focus` leave path instead). `destroy` is a `&self`
    /// destructor request, so the proxy is left inert afterwards -- do not
    /// enable/commit on this client again.
    pub fn destroy_text_input(&mut self) {
        self.text_input.destroy();
        self.flush();
    }

    fn flush(&mut self) {
        self.client.conn.flush().expect("flush text-input request");
        self.pump();
    }

    /// One roundtrip.
    pub fn pump(&mut self) {
        let _ = self.client.queue.roundtrip(&mut self.client.state);
    }

    /// Pump the queue until `pred` holds or `TIMEOUT` elapses.
    pub fn wait_until(&mut self, pred: impl Fn(&TextInputClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            let _ = self.client.queue.roundtrip(&mut self.client.state);
        }
    }

    /// How many `zwp_text_input_v3.enter` events this client has received.
    pub fn entered(&self) -> u32 {
        self.client.state.text_input_enters
    }

    /// How many `zwp_text_input_v3.leave` events this client has received.
    pub fn left(&self) -> u32 {
        self.client.state.text_input_leaves
    }

    /// How many `wl_keyboard.key` events the underlying app client has
    /// received — the app's *ordinary* keyboard delivery, distinct from the
    /// text-input relay. A6.2 test 9 asserts this advances before an IME
    /// keyboard grab and then stops once the grab intercepts.
    pub fn wl_keyboard_key_events(&self) -> u32 {
        self.client.state.key_events
    }

    /// Every `preedit_string` event's text, in arrival order.
    pub fn preedits(&self) -> &[String] {
        &self.client.state.text_input_preedit_strings
    }

    /// Every `commit_string` event's text, in arrival order.
    pub fn commits(&self) -> &[String] {
        &self.client.state.text_input_commit_strings
    }

    /// Every `delete_surrounding_text` event, as `(before_length,
    /// after_length)`, in arrival order.
    pub fn deletes(&self) -> &[(u32, u32)] {
        &self.client.state.text_input_deletes
    }

    /// How many `zwp_text_input_v3.done` events this client has received.
    pub fn dones(&self) -> u32 {
        self.client.state.text_input_dones
    }
}

/// A `zwp_input_method_v2` IME/OSK double: binds the manager (surfaceless, like
/// a real IME daemon), creates an input-method, records activate/deactivate/
/// surrounding_text/content_type/done/unavailable, and can send preedit/commit/
/// delete + commit. M6.1 tests 3-6's IME side.
///
/// Its own connection rather than a [`TestClient`] method, for the same
/// reason as [`GammaControlClient`]: the real thing (an IME/OSK daemon)
/// never holds keyboard focus and binds a bare `wl_seat`, mirroring
/// `DataControlClient`'s "never holds focus" rationale. It does map one
/// surface -- the candidate popup ([`create_popup`](InputMethodClient::create_popup)) --
/// which is exactly what a real IME does; that surface never takes focus.
pub struct InputMethodClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    qh: QueueHandle<ClientState>,
    state: ClientState,
    input_method: zwp_input_method_v2::ZwpInputMethodV2,
    /// The candidate popup surface, once [`create_popup`](InputMethodClient::create_popup)
    /// has made one. Kept alive here (dropping the proxy would send the
    /// destructor and tear the popup down).
    popup: Option<zwp_input_popup_surface_v2::ZwpInputPopupSurfaceV2>,
    /// The `wl_surface` the popup wraps, kept alive for the same reason.
    popup_surface: Option<wl_surface::WlSurface>,
    /// Kept alive so the popup's shm buffer/pool file are not dropped mid-test.
    _popup_shm: Option<(std::fs::File, wl_shm_pool::WlShmPool, wl_buffer::WlBuffer)>,
    /// The active keyboard grab, once [`grab_keyboard`](InputMethodClient::grab_keyboard)
    /// has taken one. Kept alive (dropping it releases the grab).
    grab: Option<zwp_input_method_keyboard_grab_v2::ZwpInputMethodKeyboardGrabV2>,
}

impl InputMethodClient {
    /// Connect, bind `zwp_input_method_manager_v2`, and create this seat's
    /// `zwp_input_method_v2`. Panics if the compositor did not advertise the
    /// manager or a seat.
    pub fn spawn(socket: &str) -> InputMethodClient {
        let (conn, mut queue, qh, mut state) = connect_and_bind(socket);
        let manager = state
            .input_method_manager
            .clone()
            .expect("compositor did not advertise zwp_input_method_manager_v2");
        let seat = state
            .seat
            .clone()
            .expect("compositor did not advertise wl_seat");
        let input_method = manager.get_input_method(&seat, &qh, ());
        conn.flush().expect("flush get_input_method");
        queue.roundtrip(&mut state).expect("input-method roundtrip");
        InputMethodClient {
            conn,
            queue,
            qh,
            state,
            input_method,
            popup: None,
            popup_surface: None,
            _popup_shm: None,
            grab: None,
        }
    }

    /// Create the IME's candidate popup surface: a real `wl_surface` given the
    /// `input_popup` role via `get_input_popup_surface`, with a small opaque
    /// buffer attached and committed so it has a mapped size. This is what
    /// drives the compositor's `new_popup_surface` handler (placement +
    /// `send_input_popup_rectangle`). Panics if the compositor advertised
    /// neither `wl_compositor` nor `wl_shm`, or if a popup already exists.
    pub fn create_popup(&mut self) {
        assert!(
            self.popup.is_none(),
            "create_popup called with a popup already live"
        );
        let compositor = self
            .state
            .compositor
            .clone()
            .expect("compositor did not advertise wl_compositor");
        let shm = self
            .state
            .shm
            .clone()
            .expect("compositor did not advertise wl_shm");

        let surface = compositor.create_surface(&self.qh, ());
        let popup = self
            .input_method
            .get_input_popup_surface(&surface, &self.qh, ());

        // Attach a small opaque buffer and commit so the popup surface maps
        // with a real size, exactly as a real IME does before its candidate
        // list can be shown.
        let (file, pool, buffer) = create_shm_buffer(&shm, &self.qh, 120, 80);
        surface.attach(Some(&buffer), 0, 0);
        surface.damage(0, 0, 120, 80);
        surface.commit();

        self.popup = Some(popup);
        self.popup_surface = Some(surface);
        self._popup_shm = Some((file, pool, buffer));
        self.flush();
    }

    /// Take a hardware keyboard grab (`grab_keyboard`). While held, the
    /// compositor forwards seat key/modifier events to the returned grab
    /// object rather than to any client's `wl_keyboard`. Panics if a grab is
    /// already live.
    pub fn grab_keyboard(&mut self) {
        assert!(
            self.grab.is_none(),
            "grab_keyboard called with a grab already live"
        );
        let grab = self.input_method.grab_keyboard(&self.qh, ());
        self.grab = Some(grab);
        self.flush();
    }

    /// The most recent `text_input_rectangle` the popup was told, or `None`
    /// until the compositor sends one. The popup half of A6.2 test 8.
    pub fn popup_text_input_rectangle(&self) -> Option<(i32, i32, i32, i32)> {
        self.state.im_popup_text_input_rectangle
    }

    /// Destroy the `zwp_input_method_v2` object itself (its `destroy` request),
    /// leaving the client connection alive. This is the exact wire event that
    /// drives wlroots' input-method teardown -- deliberately distinct from
    /// dropping the whole client -- so a test can assert the compositor cascades
    /// the IME's still-live candidate popups away with it. `destroy` is a
    /// destructor request, so the proxy is inert afterwards; do not call popup/
    /// grab/commit on this client again.
    pub fn destroy(&mut self) {
        self.input_method.destroy();
        self.flush();
    }

    /// How many `key` events the keyboard grab has intercepted.
    pub fn grab_key_events(&self) -> u32 {
        self.state.im_grab_key_events
    }

    /// How many `modifiers` events the keyboard grab has intercepted.
    pub fn grab_modifier_events(&self) -> u32 {
        self.state.im_grab_modifier_events
    }

    /// Send any combination of preedit/commit/delete, then `commit` --
    /// mirroring how a real IME batches double-buffered state before
    /// applying it. The `commit` request's serial echoes back the number of
    /// `done` events received so far, per the protocol.
    pub fn send_commit(
        &mut self,
        preedit: Option<&str>,
        commit: Option<&str>,
        delete: Option<(u32, u32)>,
    ) {
        if let Some(p) = preedit {
            self.input_method
                .set_preedit_string(p.to_string(), 0, p.len() as i32);
        }
        if let Some(c) = commit {
            self.input_method.commit_string(c.to_string());
        }
        if let Some((before, after)) = delete {
            self.input_method.delete_surrounding_text(before, after);
        }
        self.input_method.commit(self.state.im_dones);
        self.flush();
    }

    fn flush(&mut self) {
        self.conn.flush().expect("flush input-method request");
        self.pump();
    }

    /// One roundtrip.
    pub fn pump(&mut self) {
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Pump the queue until `pred` holds or `TIMEOUT` elapses.
    pub fn wait_until(&mut self, pred: impl Fn(&InputMethodClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            let _ = self.queue.roundtrip(&mut self.state);
        }
    }

    /// How many `zwp_input_method_v2.activate` events this client has
    /// received.
    pub fn activates(&self) -> u32 {
        self.state.im_activates
    }

    /// How many `zwp_input_method_v2.deactivate` events this client has
    /// received.
    pub fn deactivates(&self) -> u32 {
        self.state.im_deactivates
    }

    /// Every `surrounding_text` event's `(text, cursor, anchor)`, in
    /// arrival order.
    pub fn surroundings(&self) -> &[(String, u32, u32)] {
        &self.state.im_surroundings
    }

    /// Every `content_type` event's `(hint, purpose)`, in arrival order.
    pub fn content_types(&self) -> &[(u32, u32)] {
        &self.state.im_content_types
    }

    /// Every `text_change_cause` event's raw `cause`, in arrival order.
    pub fn text_change_causes(&self) -> &[u32] {
        &self.state.im_text_change_causes
    }

    /// How many `zwp_input_method_v2.done` events this client has received.
    pub fn dones(&self) -> u32 {
        self.state.im_dones
    }

    /// Whether the compositor sent `unavailable` -- another input method was
    /// already associated with this seat.
    pub fn unavailable(&self) -> bool {
        self.state.im_unavailable
    }
}

/// A `zwlr_gamma_control_manager_v1` client -- the shape a night-light/
/// redshift daemon takes: it never maps a surface, it just claims the
/// output's gamma LUT.
///
/// Its own connection rather than a [`TestClient`] method because that is
/// what the real thing is (a background daemon, no surface at all), and
/// because `get_gamma_control` is exclusive per output: a second control on
/// an output that already has one makes the compositor destroy the first,
/// which would silently poison an unrelated test's client.
pub struct GammaControlClient {
    conn: Connection,
    queue: EventQueue<ClientState>,
    state: ClientState,
    control: zwlr_gamma_control_v1::ZwlrGammaControlV1,
}

impl GammaControlClient {
    /// Connect and claim gamma control of the compositor's output. Panics if
    /// the compositor did not advertise `zwlr_gamma_control_manager_v1` or
    /// `wl_output`.
    pub fn spawn(socket: &str) -> GammaControlClient {
        let (conn, mut queue, qh, mut state) = connect_and_bind(socket);
        let manager = state
            .gamma_control_manager
            .clone()
            .expect("compositor did not advertise zwlr_gamma_control_manager_v1");
        let output = state
            .output
            .clone()
            .expect("compositor did not advertise wl_output");
        let control = manager.get_gamma_control(&output, &qh, ());
        conn.flush().expect("flush get_gamma_control");
        queue.roundtrip(&mut state).expect("gamma roundtrip");
        GammaControlClient {
            conn,
            queue,
            state,
            control,
        }
    }

    /// Pump until `pred` holds or `TIMEOUT` elapses; returns whether it
    /// ever held. A bounded `roundtrip` loop for the same reason
    /// [`TestClient::wait_until`] is one -- the deadline stays honest even
    /// against a compositor that has nothing to say.
    pub fn wait_until(&mut self, pred: impl Fn(&GammaControlClient) -> bool) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            self.queue.roundtrip(&mut self.state).expect("roundtrip");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The `gamma_size` the compositor reported for the output, if it sent
    /// one at all.
    pub fn gamma_size(&self) -> Option<u32> {
        self.state.gamma_size
    }

    /// Whether the compositor sent `failed` -- the protocol's "this object
    /// is now inert" event. Not a destructor: the object stays alive on the
    /// wire and the *client* is the one that must destroy it, which is why
    /// this is a readable flag rather than a state the harness tears down.
    pub fn failed(&self) -> bool {
        self.state.gamma_failed
    }

    /// One roundtrip, so a `failed` the compositor sent after some other
    /// action lands before it is read.
    pub fn pump(&mut self) {
        let _ = self.queue.roundtrip(&mut self.state);
    }

    /// Upload an identity ramp of `size` entries per channel via
    /// `set_gamma(fd)`. The protocol's payload is three consecutive
    /// `uint16` tables (red, green, blue) of `size` entries each, so
    /// `3 * size * 2` bytes; identity means entry `i` maps to
    /// `i * 65535 / (size - 1)`.
    ///
    /// Only reachable on a backend that actually reported a `gamma_size` --
    /// see the gamma test's own doc for why the headless one does not.
    pub fn set_identity_gamma(&mut self, size: u32) {
        let entries = size as usize;
        let mut bytes = Vec::with_capacity(entries * 3 * 2);
        for _ in 0..3 {
            for i in 0..entries {
                // Widened to `u64`: `i * 65535` overflows `u32` for any
                // `size` past ~65538, which is a legal (if unusual) LUT size
                // for a client to be told, and a debug build would panic
                // rather than upload a ramp.
                let v = if entries <= 1 {
                    0u16
                } else {
                    ((i as u64 * 65535) / (entries as u64 - 1)) as u16
                };
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
        }
        let fd: OwnedFd =
            rustix::fs::memfd_create("icedtea-harness-gamma", rustix::fs::MemfdFlags::CLOEXEC)
                .expect("memfd_create");
        let mut file = std::fs::File::from(fd);
        file.write_all(&bytes).expect("write gamma ramp");
        file.flush().expect("flush gamma ramp");
        // Rewind before handing the fd over: wlroots `read(2)`s the ramp
        // from the descriptor's *current* offset, which `write_all` has
        // just left at EOF. Without this the compositor reads zero bytes
        // and answers `failed`, and the test would be asserting on an
        // error path it never meant to exercise.
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0)).expect("rewind gamma ramp");
        self.control.set_gamma(file.as_fd());
        self.conn.flush().expect("flush set_gamma");
        let _ = self.queue.roundtrip(&mut self.state);
    }
}

#[cfg(test)]
mod tests {
    use super::{Compositor, VirtualPointerClient};

    /// The harness never touches the session's runtime directory: its
    /// socket lives under a private, pid-named, 0700 `XDG_RUNTIME_DIR`, and
    /// the process env points there so spawned apps inherit it.
    ///
    /// Mutation check: drop the `set_var("XDG_RUNTIME_DIR", ..)` in
    /// `ensure_headless_env`; `add_socket_auto` puts the socket in the
    /// session dir, `socket_path` (built from `runtime_dir`) names a file
    /// that does not exist, and the `exists` assertion fails. Restore.
    #[test]
    fn the_harness_owns_a_private_runtime_dir() {
        use std::os::unix::fs::PermissionsExt as _;
        let compositor = Compositor::spawn();
        let dir = super::runtime_dir();
        assert_eq!(
            dir.file_name().and_then(|n| n.to_str()),
            Some(std::process::id().to_string().as_str()),
            "runtime dir is named by pid: {}",
            dir.display()
        );
        assert_eq!(
            dir.parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str()),
            Some("icedtea-harness"),
            "runtime dir sits under icedtea-harness/: {}",
            dir.display()
        );
        let mode = std::fs::metadata(dir)
            .expect("runtime dir exists")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700, "runtime dir is private");
        assert_eq!(
            std::env::var_os("XDG_RUNTIME_DIR").as_deref(),
            Some(dir.as_os_str()),
            "the process env points spawned apps at the private dir"
        );
        let socket = compositor.socket_path();
        assert_eq!(
            socket.parent(),
            Some(dir),
            "the socket lives in the private dir"
        );
        assert!(socket.exists(), "the socket exists at {}", socket.display());
    }

    /// The axis request the interaction gate's scroll test needs; the harness
    /// injector had motion and buttons but no axis at all.
    #[test]
    fn a_virtual_pointer_can_send_an_axis_event() {
        let compositor = Compositor::spawn();
        let socket = compositor
            .socket_path()
            .file_name()
            .expect("socket name")
            .to_string_lossy()
            .to_string();
        let mut pointer = VirtualPointerClient::spawn(&socket);
        pointer.motion_absolute(10.0, 10.0, 100, 100);
        pointer.frame();
        pointer.axis(0.0, 10.0);
        pointer.frame();
        pointer.pump();
        // No protocol error killed the client: the connection is still usable.
        pointer.motion_absolute(11.0, 11.0, 100, 100);
        pointer.frame();
        pointer.pump();
    }
}
