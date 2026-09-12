//! `icedtea-compositor` library crate: everything the binary needs lives
//! here (mirroring anvil's `main.rs`/`lib.rs` split) so that integration
//! tests -- and any future embedder -- can drive `State`, `apply_action`,
//! and `handle_command` directly without going through a real event loop,
//! DRM session, or D-Bus connection.

pub mod backend;
pub mod config_combo;
pub mod dbus;
pub mod decoration;
pub mod ime_overlay;
pub mod input;
pub mod input_method;
pub mod layout;
pub mod render;
pub mod state;
pub mod text;
pub mod wayland;
pub mod window;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use icedtea_contract::SeqEvent;

use state::State;

/// Boot and run the compositor.
///
/// Ordering is load-bearing and each step says why:
///
/// 1. `apply_backend_choice` before anything, because `autocreate` reads
///    `WLR_BACKENDS` when it runs and never again.
/// 2. `Runtime::new` before `init_graphics`, which needs the display, the
///    backend, and the runtime all to already exist. There is no ordering
///    requirement between `Runtime::new` and `Backend::autocreate`
///    themselves: `Backend::autocreate` announces nothing on its own -- the
///    backend's existing outputs are announced to `OutputHandler::new_output`
///    from inside `run_all`'s own start (`ensure_started`), well after
///    `state.wayland.attach` below has already handed the seam a runtime to
///    enable/init an output against.
/// 3. `state` is constructed after `display`/`runtime`/`backend` (and so, by
///    ordinary end-of-scope drop order, is dropped before them): `attach`
///    gives `state.wayland` a `Runtime` clone, and `wlr` documents that a
///    `Runtime` must not outlive the `Display` it was initialized against.
/// 4. `add_socket_auto` and the `WAYLAND_DISPLAY` set happen after the
///    backend, so that a nested backend opens its window against the
///    *host* session's display before that variable is clobbered for our
///    own children -- and, just as importantly, before the D-Bus service or
///    the wallpaper decode worker are spawned: the `set_var` below is
///    single-threaded-with-respect-to-the-environment only if nothing else
///    is running yet, so both of those threads start *after* it, not before.
pub fn run() {
    let choice = backend::BackendChoice::from_args(std::env::args());
    backend::apply_backend_choice(choice);

    let db_path = icedtea_config::default_db_path();
    let config = icedtea_config::load_or_default(&db_path);

    let display = wlr::Display::new().expect("failed to create the wayland display");
    let runtime = wlr::Runtime::new().expect("failed to create the scene graph");
    let backend = wlr::Backend::autocreate(&display.event_loop())
        .expect("failed to create a backend; is a session available, or WLR_BACKENDS set?");
    runtime
        .init_graphics(&display, &backend)
        .expect("failed to create the renderer and the core protocol globals");
    runtime
        .create_xdg_shell(&display, 6)
        .expect("failed to advertise xdg_wm_base");
    // Not fatal, matching the shutdown source's tone below: without the
    // manager a client simply never gets to state a decoration preference
    // and draws its own or not on its own defaults, which is a degraded
    // compositor, not a dead one.
    if let Err(err) = runtime.create_xdg_decoration_manager(&display) {
        tracing::error!(%err, "xdg-decoration negotiation is unavailable");
    }
    // Same non-fatal tone as the decoration manager just above: without
    // this global, panels/bars simply cannot bind `zwlr_layer_shell_v1` and
    // the desktop runs with no shell chrome, which is degraded, not dead.
    if let Err(err) = runtime.create_layer_shell(&display, 4) {
        tracing::error!(%err, "layer-shell is unavailable");
    }
    // Same non-fatal tone: without these, middle-click paste and
    // clipboard-manager access are simply absent, which is degraded, not
    // dead. Regular clipboard (`wl_data_device`) already came up in
    // `init_graphics`; the seat's selection request events are wired in the
    // backend regardless.
    if let Err(err) = runtime.create_primary_selection_manager(&display) {
        tracing::error!(%err, "primary selection (middle-click paste) is unavailable");
    }
    if let Err(err) = runtime.create_data_control_manager(&display) {
        tracing::error!(%err, "data-control (clipboard manager access) is unavailable");
    }
    // Lets on-screen keyboards, remote-input bridges, and IME helpers inject a
    // keyboard. Non-fatal: without it those tools simply cannot attach.
    if let Err(err) = runtime.create_virtual_keyboard_manager(&display) {
        tracing::error!(%err, "virtual-keyboard input is unavailable");
    }
    // Lets on-screen keyboards' pointer counterparts, remote-input bridges,
    // and DnD test harnesses inject pointer motion/buttons. Non-fatal: same
    // reasoning as the virtual-keyboard manager just above.
    if let Err(err) = runtime.create_virtual_pointer_manager(&display) {
        tracing::error!(%err, "virtual-pointer input is unavailable");
    }
    // Lets clients capture output contents (grim, wf-recorder, and the
    // xdg-desktop-portal-wlr screen-share path). Non-fatal: a compositor that
    // fails to create the manager simply offers no screen capture.
    if let Err(err) = runtime.create_screencopy_manager(&display) {
        tracing::error!(%err, "screen capture is unavailable");
    }
    // Secure screen locking (loginctl lock-session, swaylock, etc.).
    // Non-fatal: the crate enforces the security invariants internally, so a
    // failure here just means no client can lock this session.
    if let Err(err) = runtime.create_session_lock_manager(&display) {
        tracing::error!(%err, "session locking is unavailable");
    }
    if let Err(err) = runtime.create_idle_notifier(&display) {
        tracing::error!(%err, "idle notification is unavailable");
    }
    if let Err(err) = runtime.create_idle_inhibit_manager(&display) {
        tracing::error!(%err, "idle inhibition is unavailable");
    }
    // Lets a focused client (games, remote-desktop viewers, drawing tools)
    // ask that the compositor's own keybindings be skipped while it holds
    // focus, so every key reaches it. Non-fatal: without this global such
    // clients simply fall back to the compositor consuming its bindings.
    if let Err(err) = runtime.create_shortcuts_inhibit_manager(&display) {
        tracing::error!(%err, "shortcuts inhibition is unavailable");
    }
    // Bridges tablet tools and pads to tablet-aware clients
    // (`zwp_tablet_manager_v2`). Non-fatal: without it a tablet still moves
    // the cursor (the crate attaches every input device to it, tablet
    // included), but no client ever sees tool/pad traffic.
    if let Err(err) = runtime.create_tablet_manager(&display) {
        tracing::error!(%err, "tablet input is unavailable");
    }
    // Lets clients (games, remote-desktop viewers, drawing tools) confine or
    // lock the pointer and read unaccelerated relative motion. Non-fatal:
    // without these, pointer-constraint clients simply fall back to normal
    // absolute pointer behavior.
    if let Err(err) = runtime.create_pointer_constraints_manager(&display) {
        tracing::error!(%err, "pointer constraints are unavailable");
    }
    if let Err(err) = runtime.create_relative_pointer_manager(&display) {
        tracing::error!(%err, "relative pointer motion is unavailable");
    }
    // Lets external IMEs (fcitx5/ibus) and on-screen keyboards (squeekboard)
    // relay composition to apps' text fields. Non-fatal: without them,
    // IME/OSK clients simply cannot attach and apps fall back to raw key
    // input.
    if let Err(err) = runtime.create_text_input_manager(&display) {
        tracing::error!(%err, "text-input (IME app side) is unavailable");
    }
    if let Err(err) = runtime.create_input_method_manager(&display) {
        tracing::error!(%err, "input-method (IME/OSK side) is unavailable");
    }
    // Non-fatal, matching the other `create_*_manager` calls above: a
    // compositor that cannot advertise output-management still runs, it just
    // cannot be reconfigured by a settings client. `new_output`/`destroyed`
    // guard every `update_output_manager_state` on the runtime, so a missing
    // manager degrades to "no persisted layout applied", never a crash.
    if let Err(err) = runtime.create_output_manager(&display) {
        tracing::error!(%err, "output-management unavailable");
    }
    // Every A2 compat global (batch 1 + batch 2) in one place, shared with
    // the test harness -- see `create_compat_globals`' own doc for why the
    // extraction is the point.
    create_compat_globals(&runtime, &display, &backend);
    runtime
        .create_seat(&display, "seat0")
        .expect("failed to create the seat");
    // X11 application support via Xwayland. Non-fatal, matching every other
    // `create_*` above and the spec's boot note: a host without the `Xwayland`
    // binary simply runs Wayland-only, exactly as it did before A1. `lazy` is
    // `true`, so the `Xwayland` process is only spawned when the first X11
    // client connects to the advertised `DISPLAY`; the crate points Xwayland at
    // this runtime's seat itself on `ready` (so the clipboard/primary/DND bridge
    // comes up), and `State`'s `xwayland_ready` publishes `DISPLAY` for session
    // children. Must come after `create_seat`, whose seat the bridge needs.
    if let Err(err) = runtime.create_xwayland(&display, true) {
        tracing::error!(%err, "Xwayland (X11 application support) is unavailable");
    }

    let (dbus_tx, dbus_events_rx) = crossbeam_channel::unbounded::<SeqEvent>();
    let mut state = State::new(config, dbus_tx);
    state.wayland.attach(runtime.clone());

    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<dbus::DbCommand>();
    state.set_command_receiver(cmd_rx);

    let (config_reload_tx, config_reload_rx) =
        crossbeam_channel::unbounded::<icedtea_config::Config>();
    state.set_config_reload_sender(config_reload_tx);
    state.set_config_reload_receiver(config_reload_rx);

    // Sized to nothing until an output arrives with a mode; `new_output`
    // resizes it. Lowered now so nothing later has to remember to.
    let background = runtime
        .add_rect(1, 1, render::wallpaper_color(&state.config.appearance))
        .expect("failed to create the background rect");
    runtime.lower_rect_to_bottom(background);
    state.set_background(background);

    match backend::shutdown_source(&runtime) {
        Ok(source) => state.set_shutdown_source(source),
        // Not fatal: the compositor runs, it just cannot be stopped with a
        // signal, and saying so is better than refusing to boot.
        Err(err) => tracing::error!(%err, "SIGINT/SIGTERM will not stop the compositor"),
    }

    // Wake sources for the three `crossbeam_channel`s the loop can only
    // drain from inside a handler (see `backend::wake_source`'s doc):
    // without these, a D-Bus command, a finished config reload, or a
    // finished wallpaper decode sent while the loop is idle in
    // `Until::Stop`'s blocking `dispatch(-1)` sits unseen until some
    // unrelated event happens to wake it, or never. Registration failure
    // here is as fatal as the setup above it -- there would be no way to
    // ever apply a D-Bus command, a config reload, or a wallpaper decode,
    // which is not a compositor worth booting.
    let (cmd_wake_write, cmd_wake_id) =
        backend::wake_source(&runtime).expect("failed to register the D-Bus command wake pipe");
    state.set_cmd_wake_source(cmd_wake_id);

    let (reload_wake_write, reload_wake_id) =
        backend::wake_source(&runtime).expect("failed to register the config-reload wake pipe");
    state.set_config_reload_wake_source(reload_wake_id);
    state.set_config_reload_wake(reload_wake_write);

    let (wallpaper_wake_write, wallpaper_wake_id) =
        backend::wake_source(&runtime).expect("failed to register the wallpaper-decode wake pipe");
    state.set_wallpaper_wake_source(wallpaper_wake_id);
    // Review finding C1: `wallpaper_wake_write` is *not* handed to the
    // decode worker thread directly -- that was the bug (see
    // `State::spawn_wallpaper`'s doc). Keeping the original here and letting
    // `spawn_wallpaper` hand the worker a `try_clone`d copy is the same
    // pattern `config_reload_wake`/`spawn_config_reload` already use.
    state.set_wallpaper_wake(wallpaper_wake_write);

    let socket = display
        .add_socket_auto()
        .expect("failed to create a wayland socket; is XDG_RUNTIME_DIR set?");
    // SAFETY (icedtea unsafe exception (c)): single-threaded, full stop --
    // nothing above this point spawns a thread. The D-Bus service and the
    // wallpaper decode worker are both started below, after this write, so
    // nothing else in the process can be reading or writing the environment
    // concurrently with it.
    unsafe {
        std::env::set_var("WAYLAND_DISPLAY", &socket);
    }
    // Publish DISPLAY + the X11 cursor hints in the SAME pre-thread window
    // (review finding #5): the lazy `create_xwayland` above already reserved the
    // display socket, so the name is known now, and doing this here — rather than
    // from `xwayland_ready` inside `run_all` — keeps every process-env mutation
    // before the D-Bus and wallpaper threads spawn, where it cannot race a
    // concurrent getenv.
    State::publish_xwayland_env(runtime.xwayland_display_name().as_deref());
    tracing::info!(%socket, "listening on wayland socket");

    let dbus_quit_signal = Arc::new(AtomicBool::new(false));
    let (_dbus_conn, dbus_emitter_thread) = dbus::spawn_service(
        dbus_events_rx,
        cmd_tx,
        dbus_quit_signal.clone(),
        cmd_wake_write,
    );

    let wallpaper_path = state.config.appearance.wallpaper.clone();
    state.spawn_wallpaper(wallpaper_path);

    if let Err(err) = backend.run_all(&display, &mut state, &runtime, wlr::Until::Stop) {
        tracing::error!(?err, "event loop ended with an error");
    }

    // Live delivery is `wallpaper_wake_source`'s `fd_ready` arm above,
    // wired the same way the D-Bus command and config-reload channels are:
    // the decode worker nudges the wake pipe after it sends, so a result
    // that arrives while the loop is running is applied within that same
    // turn, not after `run_all` returns. This call is the shutdown safety
    // net for the one case that wiring can't cover -- a result that arrives
    // (or a decode that finishes) after `run_all` has already returned --
    // so the channel isn't dropped mid-send and `state.wallpaper` still
    // ends up correct even on a compositor that quit immediately after
    // boot. `drain_wallpaper` is panic-free on a disconnected sender, same
    // as the live path.
    state.drain_wallpaper();

    dbus_quit_signal.store(true, Ordering::Relaxed);
    let _ = dbus_emitter_thread.join();
}

/// Advertise every A2 compatibility global, in the one order that satisfies
/// their inter-dependencies. Called from [`run`]'s boot *and* from the test
/// harness's own boot, so the two cannot drift.
///
/// Review finding F13: the harness used to carry its own hand-copied list of
/// these `create_*` calls, which meant `a2_batch*_globals_are_advertised`
/// proved only that *the harness* advertises them -- deleting the whole block
/// from `run()` left every test green and shipped a compositor with no
/// viewporter, no cursor-shape, and no gamma control. One shared function is
/// what makes those tests load-bearing for the real boot path.
///
/// Non-fatal throughout, matching every other `create_*` in `run()`: a global
/// that fails to come up just means clients take the fallback path (unscaled
/// buffers, integer scale, no logical geometry, no named cursor, no gamma
/// ramp). The harness deliberately inherits that tone rather than panicking:
/// the advertisement tests are the assertion, and a `create_*` that silently
/// failed there now fails them instead of aborting the process.
///
/// Ordering constraints, all of them real:
///
/// * `create_xdg_output_manager` needs the scene's output layout, so it must
///   follow `init_graphics` (the caller's job -- both callers do it well
///   before this).
/// * `create_presentation` needs the backend, hence the `backend` parameter,
///   and `set_scene_presentation` must follow both `init_graphics` and
///   `create_presentation` (the crate enforces this internally).
/// * `create_gamma_control_manager` wires the manager straight into this
///   runtime's scene (`wlr_scene_set_gamma_control_manager_v1`), so it too
///   needs `init_graphics`.
///
/// `create_cursor_shape_manager` and `create_xdg_activation_manager` only let
/// clients *ask* for a named cursor or an activation -- the crate applies
/// neither itself, so `State`'s `SeatHandler::request_set_shape` /
/// `request_activate` are what make them do anything.
pub fn create_compat_globals(
    runtime: &wlr::Runtime,
    display: &wlr::Display,
    backend: &wlr::Backend,
) {
    // A2 batch-1 passive protocols: none of these change client-visible
    // behavior on their own, they just let clients discover/opt into finer
    // scaling, buffer, and geometry hints. Non-fatal, same tone as every
    // `create_*` above -- a missing global just means the fallback path
    // (unscaled buffers, integer scale, no logical geometry, etc.) stays in
    // effect.
    if let Err(err) = runtime.create_viewporter(display) {
        tracing::error!(%err, "viewporter unavailable; clients fall back to unscaled buffers");
    }
    if let Err(err) = runtime.create_fractional_scale_manager(display) {
        tracing::error!(
            %err,
            "fractional-scale unavailable; HiDPI clients render at integer scale"
        );
    }
    if let Err(err) = runtime.create_single_pixel_buffer_manager(display) {
        tracing::error!(%err, "single-pixel-buffer unavailable");
    }
    if let Err(err) = runtime.create_content_type_manager(display) {
        tracing::error!(%err, "content-type manager unavailable");
    }
    if let Err(err) = runtime.create_xdg_output_manager(display) {
        tracing::error!(
            %err,
            "xdg-output unavailable; some panels/tools lose logical geometry"
        );
    }
    // Needs the backend and the scene graph, so it can only run after
    // `init_graphics` above created both; `set_scene_presentation` wires the
    // scene side and the crate enforces that ordering internally.
    if let Err(err) = runtime.create_presentation(display, backend) {
        tracing::error!(%err, "presentation-time unavailable; clients get no presentation feedback");
    } else if let Err(err) = runtime.set_scene_presentation() {
        tracing::error!(%err, "presentation created but scene wiring failed");
    }
    // A2 batch-2 passive/request-handled protocols: same non-fatal tone as
    // the batch-1 block above. `create_cursor_shape_manager` and
    // `create_xdg_activation_manager` only let clients *ask* for a named
    // cursor or activation -- the crate does not apply either itself, so
    // `SeatHandler::request_set_shape`/`request_activate` below are what
    // make them do anything. `create_gamma_control_manager` needs
    // `init_graphics` (already run above) because it wires the manager
    // straight into this runtime's scene (`wlr_scene_set_gamma_control_manager_v1`),
    // which applies gamma ramps on its own commit path with no handler
    // involvement.
    if let Err(err) = runtime.create_cursor_shape_manager(display) {
        tracing::error!(%err, "cursor-shape unavailable; clients cannot name a cursor image");
    }
    if let Err(err) = runtime.create_xdg_activation_manager(display) {
        tracing::error!(%err, "xdg-activation unavailable; clients cannot request focus-raise");
    }
    if let Err(err) = runtime.create_gamma_control_manager(display) {
        tracing::error!(%err, "gamma-control unavailable; clients cannot set a display gamma ramp");
    }
    if let Err(err) = runtime.create_tearing_control_manager(display, 1) {
        tracing::error!(%err, "tearing control unavailable; clients get no tearing hints");
    }
    if let Err(err) = runtime.create_power_manager(display) {
        tracing::error!(%err, "output power management unavailable; clients cannot request power modes");
    }
}
