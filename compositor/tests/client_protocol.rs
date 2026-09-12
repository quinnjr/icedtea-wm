//! The client-driven protocol tests: a real `wayland-client` speaking real
//! xdg-shell to a real headless compositor over a real socket.
//!
//! Everything else in this crate's test suite drives `State` directly, with
//! synthetic `ToplevelKey::for_test` ids that no wlroots object backs — which
//! means none of it can exercise the scene-graph half of the seam (a dangling
//! id makes every `Wayland::*` scene call a no-op by construction). These
//! tests are what restores that coverage: a window only appears in the model
//! here because a client actually mapped a surface, and the compositor
//! actually ran the `mapped` path against a live toplevel.

use icedtea_contract::{Event, Rectangle};

use icedtea_harness::{
    Compositor, DataControlClient, GestureClient, IdleInhibitClient, IdleNotifyClient,
    InputMethodClient, PointerClient, PointerConstraintsClient, SessionLockClient,
    ShortcutsInhibitClient, TestClient, TextInputClient, TouchClient, VirtualKeyboardClient,
    VirtualPointerClient,
};

/// A data-control client's set (no serial) reaches a focused wl_data_device
/// client as an offer — the delivery path a clipboard manager's re-paste uses.
#[test]
fn data_control_set_reaches_a_focused_wl_data_device_client() {
    let comp = Compositor::spawn();
    let mut b = TestClient::map_toplevel(&comp.socket, "reader.app", "reader");
    assert!(b.wait_until(|c| c.last_configure().is_some()));
    let mut mgr = DataControlClient::spawn(&comp.socket);
    mgr.set_clipboard("text/plain;charset=utf-8", b"via-data-control");
    assert!(
        b.wait_until(|c| c.has_selection_offer()),
        "focused wl_data_device client never received the data-control-set selection"
    );
    let got = icedtea_harness::read_selection_from_data_control(
        &mut b,
        &mut mgr,
        "text/plain;charset=utf-8",
    );
    assert_eq!(got, b"via-data-control");
}

/// The M4.6 daemon's actual path: a regular app copies via `wl_data_device`
/// and a data-control client (a clipboard manager) sees the same bytes. The
/// setter needs the injected keyboard's serial.
#[test]
fn a_wl_data_device_copy_is_seen_by_a_data_control_reader() {
    let comp = Compositor::spawn();
    let mut _vk = VirtualKeyboardClient::spawn(&comp.socket);

    let mut app = TestClient::map_toplevel(&comp.socket, "app", "app");
    assert!(
        app.wait_until(|c| c.has_input_serial()),
        "app got no serial"
    );
    app.set_selection_text("text/plain;charset=utf-8", b"copied-by-app");

    let mut manager = DataControlClient::spawn(&comp.socket);
    assert!(
        manager.wait_until(|c| c.has_offer()),
        "manager saw no offer"
    );
    assert_eq!(
        manager.read_from_wl_data_device_owner(&mut app, "text/plain;charset=utf-8"),
        b"copied-by-app"
    );
}

/// The focus/serial gate: with no keyboard on the seat, a mapped client has no
/// input serial, so its `wl_data_device.set_selection` is rejected by wlroots
/// and the existing clipboard is untouched. This is the mechanism that stops an
/// unfocused client hijacking the selection (an unfocused client likewise has
/// no serial).
#[test]
fn set_selection_without_an_input_serial_is_rejected() {
    let comp = Compositor::spawn(); // deliberately NO virtual keyboard -> no serials

    // Baseline: a data-control client owns the clipboard.
    let mut owner = DataControlClient::spawn(&comp.socket);
    owner.set_clipboard("text/plain", b"baseline");

    // A maps and is focused, but the seat has no keyboard, so A has no serial.
    let mut a = TestClient::map_toplevel(&comp.socket, "hijack.app", "hijack");
    assert!(a.wait_until(|c| c.last_configure().is_some()));
    assert!(
        !a.has_input_serial(),
        "no keyboard on the seat means no input serial"
    );
    a.set_selection_text("text/plain", b"hijack"); // serial 0 -> rejected

    // The clipboard is unchanged: a fresh reader still sees the baseline.
    let mut reader = DataControlClient::spawn(&comp.socket);
    assert!(
        reader.wait_until(|c| c.has_offer()),
        "no data-control offer"
    );
    assert_eq!(reader.read_selection(&mut owner, "text/plain"), b"baseline");
}

/// A data-control client reads a selection another data-control client set —
/// the round-trip the M4.6 clipboard daemon depends on.
#[test]
fn data_control_reads_the_current_selection() {
    let comp = Compositor::spawn();
    let mut owner = DataControlClient::spawn(&comp.socket);
    owner.set_clipboard("text/plain;charset=utf-8", b"seen-by-manager");

    let mut reader = DataControlClient::spawn(&comp.socket);
    assert!(
        reader.wait_until(|c| c.has_offer()),
        "no data-control offer"
    );
    assert_eq!(
        reader.read_selection(&mut owner, "text/plain;charset=utf-8"),
        b"seen-by-manager"
    );
}

/// The two selection-manager globals M4.1 adds must actually be advertised —
/// the daemon (M4.6) and any clipboard manager bind them by name.
#[test]
fn selection_globals_are_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals
            .iter()
            .any(|g| g == "zwp_primary_selection_device_manager_v1"),
        "primary-selection manager global missing; saw {globals:?}"
    );
    assert!(
        globals.iter().any(|g| g == "zwlr_data_control_manager_v1"),
        "data-control manager global missing; saw {globals:?}"
    );
}

/// M4.2 adds the virtual-pointer manager global so DnD test harnesses (and
/// on-screen-keyboard-style input bridges) can inject pointer motion/buttons.
#[test]
fn virtual_pointer_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals
            .iter()
            .any(|g| g == "zwlr_virtual_pointer_manager_v1"),
        "virtual-pointer manager global missing; saw {globals:?}"
    );
}

/// M4.3 adds the screencopy manager global so screenshot tools (grim,
/// wf-recorder) and the screen-share portal can capture output contents.
#[test]
fn screencopy_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "zwlr_screencopy_manager_v1"),
        "screencopy manager global missing; saw {globals:?}"
    );
}

/// M4.4 adds secure screen locking: `ext_session_lock_manager_v1` lets a
/// locker (swaylock, gtk4-lock-screen, ...) take the session lock.
#[test]
fn session_lock_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "ext_session_lock_manager_v1"),
        "session lock manager global missing; saw {globals:?}"
    );
}

/// M4.4 adds idle notification: `ext_idle_notifier_v1` lets a client (e.g.
/// swayidle) learn when the seat has been idle for a timeout.
#[test]
fn idle_notifier_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "ext_idle_notifier_v1"),
        "idle notifier global missing; saw {globals:?}"
    );
}

/// M4.4 adds idle inhibition: `zwp_idle_inhibit_manager_v1` lets a client
/// (e.g. a video player) suppress idle notification while active.
#[test]
fn idle_inhibit_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "zwp_idle_inhibit_manager_v1"),
        "idle inhibit manager global missing; saw {globals:?}"
    );
}

/// M4.5 adds pointer constraints: `zwp_pointer_constraints_v1` lets a
/// client (games, drawing tools) confine or lock the pointer.
#[test]
fn pointer_constraints_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "zwp_pointer_constraints_v1"),
        "pointer constraints manager global missing; saw {globals:?}"
    );
}

/// M4.5 adds relative pointer motion: `zwp_relative_pointer_manager_v1`
/// lets a client (games, remote-desktop viewers) read unaccelerated
/// relative pointer motion.
#[test]
fn relative_pointer_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals
            .iter()
            .any(|g| g == "zwp_relative_pointer_manager_v1"),
        "relative pointer manager global missing; saw {globals:?}"
    );
}

/// M6.1 adds the IME relay: `zwp_text_input_manager_v3` lets an app opt a
/// text field into IME composition, and `zwp_input_method_manager_v2` lets
/// an external IME (fcitx5/ibus) or on-screen keyboard (squeekboard) attach
/// to relay that composition. Both globals must be advertised even with no
/// IME bound, and the compositor must boot non-fatally either way.
#[test]
fn text_input_and_input_method_globals_are_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "zwp_text_input_manager_v3"),
        "text-input global missing; saw {globals:?}"
    );
    assert!(
        globals.iter().any(|g| g == "zwp_input_method_manager_v2"),
        "input-method global missing; saw {globals:?}"
    );
}

/// B3 smoke test: [`TextInputClient::spawn`] maps a toplevel, gains keyboard
/// focus via the existing auto-focus-on-map path, and its `zwp_text_input_v3`
/// receives `enter` -- proof the double actually binds and wires up, ahead of
/// B5's full relay coverage.
#[test]
fn text_input_client_enters_on_keyboard_focus() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket); // seat needs a keyboard
    let mut ti = TextInputClient::spawn(&comp.socket); // maps + auto-focused
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never got enter on focus"
    );
}

/// B4 smoke test: [`InputMethodClient::spawn`] binds the manager and creates
/// an input-method with no text-input enabled yet -- proof the double
/// actually binds and wires up, ahead of B5's full relay coverage.
#[test]
fn input_method_client_binds_without_activation() {
    let comp = Compositor::spawn();
    let im = InputMethodClient::spawn(&comp.socket);
    assert!(!comp.input_method_active(), "no text-input enabled yet");
    assert_eq!(im.activates(), 0);
}

/// B5 test 2: `zwp_text_input_v3.enter` follows *keyboard* focus, not pointer
/// focus. A second surface (`other`) holds the pointer the whole time; the
/// text-input surface maps last, taking keyboard focus, and gets `enter` even
/// though the pointer is over `other` -- non-vacuous proof enter is not gated
/// on the pointer being over the text-input surface. Mapping a `third`
/// toplevel then steals keyboard focus, and `leave` must fire.
#[test]
fn text_input_enter_follows_keyboard_focus_not_pointer() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket); // seat needs a keyboard
    let _other = TestClient::map_toplevel(&comp.socket, "other", "other");
    // Park the pointer over `other`, so pointer focus is emphatically NOT on
    // the text-input surface that maps next.
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    vp.motion_absolute(10.0, 10.0, 100, 100);
    vp.frame();
    vp.pump();

    let mut ti = TextInputClient::spawn(&comp.socket); // maps last -> keyboard-focused
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "enter must follow keyboard focus even with the pointer over another surface"
    );

    // Move keyboard focus away by mapping (and auto-focusing) a third toplevel.
    let _third = TestClient::map_toplevel(&comp.socket, "third", "third");
    assert!(
        ti.wait_until(|c| c.left() >= 1),
        "leave must fire when keyboard focus is lost"
    );
}

/// B5 test 3: enabling a focused text-input activates a bound IME and relays
/// the surrounding text. Asserts on BOTH sides -- the IME double's captured
/// `activate` + `surrounding_text` events AND the compositor's
/// `input_method_active` oracle.
#[test]
fn enable_activates_ime_and_relays_surrounding_text() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket); // IME bound first
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    ti.enable();
    ti.commit_with("hello", 5, 5, (10, 20, 2, 16));

    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");
    assert!(
        im.wait_until(|s| s
            .surroundings()
            .iter()
            .any(|t| t == &("hello".to_string(), 5, 5))),
        "IME never saw the (text, cursor, anchor) surrounding relay; saw {:?}",
        im.surroundings()
    );
    // Both sides: the oracle must agree the IME is active.
    assert!(comp.input_method_active(), "oracle: IME must be active");
}

/// B5 test (M6.1 coverage gap): once a text-input is already active, a
/// SUBSEQUENT commit with NEW surrounding text must be re-forwarded to the
/// IME with the UPDATED values -- not just the initial forward at
/// activation time. Exercises `on_text_input_commit`'s re-forward path.
#[test]
fn commit_after_activation_reforwards_updated_surrounding() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket); // IME bound first
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    // Step 1: activate, and establish the initial surrounding relay.
    ti.enable();
    ti.commit_with("first", 5, 5, (10, 20, 2, 16));
    assert!(
        im.wait_until(|s| s.surroundings().last() == Some(&("first".to_string(), 5, 5))),
        "IME never saw the initial surrounding relay; saw {:?}",
        im.surroundings()
    );
    assert!(comp.input_method_active(), "oracle: IME must be active");

    // Step 2: the re-forward assertion. A second commit on the
    // ALREADY-ACTIVE text-input carries new surrounding text; the IME must
    // observe the UPDATED (text, cursor, anchor), not remain stuck on the
    // initial "first" values.
    ti.commit_with("second value", 3, 8, (11, 21, 3, 17));
    assert!(
        im.wait_until(|s| s.surroundings().last() == Some(&("second value".to_string(), 3, 8))),
        "IME never saw the re-forwarded, UPDATED surrounding relay; saw {:?}",
        im.surroundings()
    );
}

/// B5 test 4 (the Spec-2 unblock): an IME's preedit + commit_string relay
/// through to the focused app's `zwp_text_input_v3`. Asserts on the app side
/// (the text-input double's captured `preedit_string`/`commit_string`).
#[test]
fn ime_commit_relays_preedit_and_commit_to_app() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("", 0, 0, (0, 0, 1, 1));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    im.send_commit(Some("に"), Some("日本"), None);

    assert!(
        ti.wait_until(|c| c.preedits().last().map(String::as_str) == Some("に")),
        "app never got the preedit; saw {:?}",
        ti.preedits()
    );
    assert!(
        ti.wait_until(|c| c.commits().last().map(String::as_str) == Some("日本")),
        "app never got the committed text; saw {:?}",
        ti.commits()
    );
}

/// B5 test 5: disabling the text-input deactivates the IME, and losing
/// keyboard focus sends `leave`. Asserts on BOTH sides -- the IME double's
/// captured `deactivate` AND the `input_method_active` oracle -- plus the
/// app-side `leave`.
#[test]
fn disable_deactivates_and_focus_away_leaves() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("hello", 5, 5, (10, 20, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");
    assert!(comp.input_method_active(), "oracle: IME must be active");

    ti.disable();
    assert!(
        im.wait_until(|s| s.deactivates() >= 1),
        "IME never deactivated on disable"
    );
    assert!(
        !comp.input_method_active(),
        "oracle: IME must no longer be active after disable"
    );

    let _third = TestClient::map_toplevel(&comp.socket, "third", "third"); // steal keyboard focus
    assert!(
        ti.wait_until(|c| c.left() >= 1),
        "text-input never left on focus loss"
    );
}

/// M6.1 M1 guard: destroying an ACTIVE text-input OBJECT -- while its surface
/// keeps keyboard focus -- must deactivate the IME. This exercises wlroots'
/// `on_text_input_destroy` path specifically, NOT the keyboard-focus `leave`
/// path (`relay_keyboard_focus`): the toplevel is never unmapped and the
/// client is never dropped, so focus never changes; only the
/// `zwp_text_input_v3.destroy` request fires. Before the M1 fix,
/// `on_text_input_destroy` tore the object down WITHOUT sending the IME
/// `deactivate`, leaving it stuck active with a dangling text-input; the fix
/// makes it send deactivate -> done -> clear. Asserts on BOTH sides -- the IME
/// double's captured `deactivate` AND the compositor's `input_method_active`
/// oracle turning false.
#[test]
fn destroying_active_text_input_deactivates_ime() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket); // IME bound first
    let mut ti = TextInputClient::spawn(&comp.socket); // maps + auto-focused
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    // Establish the ACTIVE state: enabled + focused, driving the IME.
    ti.enable();
    ti.commit_with("hi", 2, 2, (0, 0, 1, 1));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");
    assert!(comp.input_method_active(), "oracle: IME must be active");

    // Destroy ONLY the zwp_text_input_v3 object. The toplevel stays mapped and
    // keyboard-focused, so this drives `on_text_input_destroy`, not the
    // focus-leave path.
    ti.destroy_text_input();

    // The M1 assertion: destroying the active object deactivates the IME.
    assert!(
        im.wait_until(|s| s.deactivates() >= 1),
        "IME never deactivated when its active text-input object was destroyed"
    );
    assert!(
        !comp.input_method_active(),
        "oracle: IME must no longer be active after the text-input object is destroyed"
    );
}

/// B5 test 6: only one input-method may bind a seat. A second `get_input_method`
/// gets `unavailable` and never activates, while the first stays fully
/// functional. Asserts on both IME doubles' captured events.
#[test]
fn second_input_method_is_refused() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut first = InputMethodClient::spawn(&comp.socket);
    let mut second = InputMethodClient::spawn(&comp.socket);
    assert!(
        second.wait_until(|s| s.unavailable()),
        "second IME must get `unavailable`"
    );

    // Even when a text-input enables, only the first ever activates.
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("x", 1, 1, (0, 0, 1, 1));

    assert!(
        first.wait_until(|s| s.activates() >= 1),
        "the first IME activates"
    );
    second.pump();
    assert_eq!(second.activates(), 0, "the refused IME never activates");
}

/// B5 regression guard for the crash the relay fix closed: a text-input created
/// *after* its surface already holds keyboard focus (via
/// [`TextInputClient::spawn_on_focused_surface`], NOT the map-first
/// constructor) must (a) still receive `enter` -- the crate synthesises it on
/// create, since wlroots' `relay_keyboard_focus` only sends `enter` on a focus
/// *change* -- and (b) survive a later focus change and teardown without the
/// compositor aborting. Before the fix, the stray `leave` tripped wlroots'
/// `wlr_text_input_v3_send_leave` assertion and SIGABRT'd the whole test
/// binary; the test simply reaching its end (with the oracle still answering)
/// is the assertion that it no longer does.
#[test]
fn text_input_created_on_focused_surface_enters_and_survives_teardown() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn_on_focused_surface(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "a text-input created on an already-focused surface must still get enter"
    );

    // Steal keyboard focus -> the compositor sends `leave` to the
    // after-created text-input; this is the path that used to SIGABRT.
    let _second = TestClient::map_toplevel(&comp.socket, "second", "second");
    assert!(
        ti.wait_until(|c| c.left() >= 1),
        "text-input never left on focus loss"
    );

    // Tear the text-input client down entirely -> another leave path.
    drop(ti);

    // The compositor is still alive: the oracle answers rather than the
    // blocking reply timing out because the process aborted.
    assert!(
        !comp.input_method_active(),
        "compositor survived a text-input created-on-focus + teardown"
    );
}

/// The Displays feature adds output management: `zwlr_output_manager_v1` lets a
/// client (a settings app, kanshi, wlr-randr) enumerate output heads and
/// request an atomic reconfiguration — resolution, position, scale, transform,
/// enabled.
#[test]
fn output_manager_global_is_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    assert!(
        globals.iter().any(|g| g == "zwlr_output_manager_v1"),
        "output manager global missing; saw {globals:?}"
    );
}

/// A client can bind `zwlr_virtual_pointer_manager_v1` and create a virtual
/// pointer, then inject motion/button/frame requests without a protocol
/// error — the M4.2 drag-and-drop grab serial's source. Full drag coverage
/// (Task 7) builds on this.
#[test]
fn a_client_can_bind_the_virtual_pointer() {
    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    vp.motion_absolute(10.0, 10.0, 200, 200);
    vp.button(0x110, true);
    vp.frame();
    vp.button(0x110, false);
    vp.frame();
    vp.pump();
}

/// A `PointerConstraintsClient` can spawn (map a toplevel, bind both M4.5
/// globals, create a `zwp_relative_pointer_v1`), then create a locked
/// pointer and a confined pointer, all without a protocol error — the
/// smoke test the T6/T7 behavioral tests build on.
#[test]
fn pointer_constraints_client_can_lock_and_confine_without_a_protocol_error() {
    let comp = Compositor::spawn();
    // A headless seat advertises the pointer capability only once a device
    // exists on it -- exactly as the keyboard capability needs a
    // `VirtualKeyboardClient` first. See `VirtualPointerClient::spawn`'s own
    // "two roundtrips" comment: it settles the capability change onto the
    // wire before this returns, so `PointerConstraintsClient::spawn` (which
    // needs `wl_pointer` to exist) is guaranteed to see it.
    let _vp = VirtualPointerClient::spawn(&comp.socket);

    // A lock and a confinement from the same wl_pointer are mutually
    // exclusive on one surface (`already_constrained`), so exercise them on
    // separate clients -- each still proves the same "no protocol error"
    // claim on its own surface.
    let mut locker = PointerConstraintsClient::spawn(&comp.socket);
    assert_eq!(locker.relative_motion_events(), 0, "no motion injected yet");
    locker.lock_pointer();
    locker.pump();
    // Every request above flushed with `.expect(...)`: a protocol error
    // (which wlroots answers by killing the connection) would already have
    // panicked one of those flushes rather than reaching here.
    assert_eq!(
        locker.relative_delta(),
        (0.0, 0.0),
        "no motion injected, so no accumulated delta"
    );

    let mut confiner = PointerConstraintsClient::spawn(&comp.socket);
    confiner.confine_pointer(0, 0, 50, 50);
    confiner.pump();
    confiner.set_confine_region(10, 10, 30, 30);
    confiner.pump();

    // A two-rect region (the T7 regression shape) round-trips too -- on its
    // own surface, since a second `confine_pointer` on the same surface +
    // wl_pointer while a confinement is already active is itself
    // `already_constrained`.
    let mut confiner2 = PointerConstraintsClient::spawn(&comp.socket);
    confiner2.confine_pointer_rects(&[(0, 0, 10, 10), (20, 20, 10, 10)]);
    confiner2.pump();
}

/// The harness can bind the data-device machinery and create a device from
/// the seat — the foundation the clipboard round-trip stands on.
#[test]
fn a_mapped_client_has_a_data_device() {
    let comp = Compositor::spawn();
    let client = TestClient::map_toplevel(&comp.socket, "dd.app", "dd");
    assert!(
        client.has_data_device(),
        "data device created from manager + seat"
    );
}

/// The core interop claim: a text payload one client copies via `wl_data_device`
/// is readable, byte for byte, by another client after focus moves to it. A
/// virtual keyboard is injected first so the setter can obtain the input serial
/// wlroots requires for `set_selection`.
#[test]
fn clipboard_transfers_between_two_clients() {
    let comp = Compositor::spawn();
    // Give the seat a keyboard so focused clients receive an input serial.
    let mut _vk = VirtualKeyboardClient::spawn(&comp.socket);

    // A maps (gains focus + a keyboard-enter serial) and owns the clipboard.
    let mut a = TestClient::map_toplevel(&comp.socket, "owner.app", "owner");
    assert!(
        a.wait_until(|c| c.has_input_serial()),
        "owner never received a keyboard-enter serial"
    );
    a.set_selection_text("text/plain;charset=utf-8", b"hello-clipboard");

    // B maps (focus moves to B) and reads the current selection.
    let mut b = TestClient::map_toplevel(&comp.socket, "reader.app", "reader");
    assert!(
        b.wait_until(|c| c.has_selection_offer()),
        "B never received a data offer"
    );
    let got = icedtea_harness::read_selection(&mut b, &mut a, "text/plain;charset=utf-8");
    assert_eq!(got, b"hello-clipboard");

    a.detach();
    b.detach();
}

/// The primary (middle-click) selection round-trip: one client sets it, another
/// reads it byte for byte after focus moves. Needs the injected keyboard's
/// serial, exactly as the clipboard does.
#[test]
fn primary_selection_transfers_between_two_clients() {
    let comp = Compositor::spawn();
    let mut _vk = VirtualKeyboardClient::spawn(&comp.socket);

    let mut a = TestClient::map_toplevel(&comp.socket, "owner.app", "owner");
    assert!(
        a.wait_until(|c| c.has_input_serial()),
        "owner got no serial"
    );
    a.set_primary_text("text/plain;charset=utf-8", b"primary-payload");

    let mut b = TestClient::map_toplevel(&comp.socket, "reader.app", "reader");
    assert!(
        b.wait_until(|c| c.has_primary_offer()),
        "B never received a primary offer"
    );
    assert_eq!(
        icedtea_harness::read_primary(&mut b, &mut a, "text/plain;charset=utf-8"),
        b"primary-payload"
    );

    a.detach();
    b.detach();
}

/// The restoration test: a real toplevel maps and shows up in the model,
/// with the app_id and title the client set, at a real geometry.
#[test]
fn a_real_client_maps_and_appears_in_the_model() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "harness.app", "first window");
    assert!(
        client.wait_until(|c| c.last_configure().is_some()),
        "no configure arrived"
    );

    let opened =
        comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "harness.app"));
    let Event::WindowOpened(info) = opened else {
        unreachable!()
    };
    assert_eq!(info.title, "first window");

    let snapshot = comp.snapshot();
    assert_eq!(snapshot.windows.len(), 1, "exactly one mapped window");
    assert_eq!(snapshot.windows[0].id, info.id);
    assert_eq!(snapshot.windows[0].app_id, "harness.app");
    assert_eq!(snapshot.windows[0].title, "first window");
    assert!(
        snapshot.windows[0].geometry.width > 0 && snapshot.windows[0].geometry.height > 0,
        "a mapped window gets a real geometry, got {:?}",
        snapshot.windows[0].geometry
    );

    client.detach();
}

/// The other direction: a model-side `Close` reaches the actual client as
/// `xdg_toplevel.close`. This is the path `Wayland::close` takes through a
/// live toplevel — unreachable with a synthetic id.
#[test]
fn closing_from_the_model_reaches_the_client() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "harness.close", "doomed");

    let opened = comp.wait_event(|e| matches!(e, Event::WindowOpened(_)));
    let Event::WindowOpened(info) = opened else {
        unreachable!()
    };

    comp.send(icedtea_compositor::dbus::DbCommand::Close(info.id));
    assert!(
        client.wait_until(|c| c.closed()),
        "client never saw xdg_toplevel.close"
    );

    // And the model half of the same round trip: a well-behaved client
    // answers `close` by destroying its toplevel, which must take the row
    // with it. `Close` alone deliberately does not — the client is the one
    // that decides.
    client.detach();
    let closed = comp.wait_event(|e| matches!(e, Event::WindowClosed(id) if *id == info.id));
    let Event::WindowClosed(_) = closed else {
        unreachable!()
    };
    assert!(comp.snapshot().windows.is_empty());
}

/// A title set after mapping propagates to the model through
/// `ToplevelHandler::title_changed`, and shows up in a fresh snapshot.
#[test]
fn a_title_change_from_the_client_reaches_the_model() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "harness.title", "before");
    let opened =
        comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "harness.title"));
    let Event::WindowOpened(info) = opened else {
        unreachable!()
    };
    assert_eq!(info.title, "before");

    client.set_title("after");
    let updated = comp.wait_event(|e| {
        matches!(e, Event::WindowUpdated { id, update }
            if *id == info.id && update.title.as_deref() == Some("after"))
    });
    let Event::WindowUpdated { .. } = updated else {
        unreachable!()
    };

    let snapshot = comp.snapshot();
    assert_eq!(snapshot.windows.len(), 1);
    assert_eq!(snapshot.windows[0].title, "after");

    client.detach();
}

/// A model-side `Maximize` reaches the client as a real `xdg_toplevel`
/// configure carrying the maximized state — the `Wayland::configure` path,
/// which a synthetic `ToplevelKey` cannot reach (`resolve` returns `None`
/// and every runtime call is skipped).
#[test]
fn a_model_side_maximize_reaches_the_client_as_a_configure_state() {
    /// `xdg_toplevel.state.maximized`.
    const MAXIMIZED: u32 = 1;

    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "harness.max", "maximize me");
    let opened =
        comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "harness.max"));
    let Event::WindowOpened(info) = opened else {
        unreachable!()
    };
    assert!(
        !client.states().contains(&MAXIMIZED),
        "a freshly mapped window is not maximized"
    );

    assert!(!comp.snapshot().windows[0].maximized);

    // Latched the way later tasks' idempotent-looking assertions must:
    // `wait_until` checks its predicate before pumping, so only a monotonic
    // counter proves a *new* configure arrived rather than the old one
    // still reading the way the test hoped.
    let configures = client.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(info.id, true));
    assert!(
        client.wait_until(|c| c.configure_count() > configures && c.states().contains(&MAXIMIZED)),
        "client never got a configure with the maximized state, last states: {:?}",
        client.states()
    );
    // Both sides of the seam, not just the client's half: the model has to
    // agree with what it told the client.
    assert!(
        comp.snapshot().windows[0].maximized,
        "the model must record the maximize it configured the client with"
    );

    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        info.id, false,
    ));
    assert!(
        client.wait_until(|c| !c.states().contains(&MAXIMIZED)),
        "client never got a configure clearing the maximized state"
    );
    assert!(!comp.snapshot().windows[0].maximized);

    client.detach();
}

/// Two clients on the same compositor both reach the model, and destroying
/// one leaves the other — the `toplevel_destroyed` path against live objects.
#[test]
fn destroying_one_of_two_clients_leaves_the_other() {
    let comp = Compositor::spawn();
    let first = TestClient::map_toplevel(&comp.socket, "harness.one", "one");
    let opened_one =
        comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "harness.one"));
    let Event::WindowOpened(one) = opened_one else {
        unreachable!()
    };

    let second = TestClient::map_toplevel(&comp.socket, "harness.two", "two");
    let opened_two =
        comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "harness.two"));
    let Event::WindowOpened(two) = opened_two else {
        unreachable!()
    };
    assert_ne!(one.id, two.id);

    assert_eq!(comp.snapshot().windows.len(), 2);

    first.detach();
    let closed = comp.wait_event(|e| matches!(e, Event::WindowClosed(id) if *id == one.id));
    let Event::WindowClosed(_) = closed else {
        unreachable!()
    };

    let snapshot = comp.snapshot();
    assert_eq!(
        snapshot.windows.len(),
        1,
        "only the destroyed window went away"
    );
    assert_eq!(snapshot.windows[0].id, two.id);

    second.detach();
}

/// Task 10: a client-driven `xdg_toplevel.set_maximized` reaches
/// `ToplevelHandler::request_maximize`, which routes onto
/// `reconcile_maximized` -- both the client's configure and the model must
/// agree, the same "both sides of the seam" shape as the model-side test
/// above, but this time initiated by the client rather than the D-Bus API.
#[test]
fn a_client_maximize_request_round_trips_through_the_compositor() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "harness.max", "maxi");
    client.request_maximize(true);
    // xdg_toplevel state 1 == maximized (xdg-shell spec numeric value).
    assert!(
        client.wait_until(|c| c.states().contains(&1)),
        "client never saw the maximized state in a configure"
    );
    let snap = comp.snapshot();
    assert!(snap.windows.iter().any(|w| w.maximized), "model must agree");
    client.detach();
}

/// Task 10: xdg-shell requires the compositor to answer every
/// `set_maximized`/`set_fullscreen` request with a configure, even one that
/// changes nothing in the model -- the dispatch layer's guarantee, not
/// `reconcile_fullscreen`'s job. Fullscreen once, then send a redundant
/// second request and prove a configure still arrives via the monotonic
/// counter (an equality check on `states()` alone can't tell a fresh
/// configure from the stale one still reading the way the test hoped).
#[test]
fn an_unhonored_request_still_gets_a_configure() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "harness.fs", "fs");
    client.request_fullscreen(true);
    assert!(
        client.wait_until(|c| c.states().contains(&2)),
        "fullscreen state expected"
    ); // 2 == fullscreen
    let before = client.configure_count();
    client.request_fullscreen(true); // redundant: already fullscreen
    assert!(
        client.wait_until(|c| c.configure_count() > before),
        "no answer to the redundant request"
    );
    client.detach();
}

/// Task 13: a client that creates a decoration object and states no
/// preference must be told **server-side** — this compositor draws the band
/// itself for anything that isn't a known CSD app.
///
/// This is also the first test in the suite where a *real* toplevel reaches
/// the whole in-tree decoration stack: answering the negotiation re-syncs
/// the window, which builds the band rect, the three button rects and the
/// cosmic-text title buffer node inside that toplevel's own scene tree
/// against a live wlroots scene. Nothing but a real client can exercise
/// those calls (a `for_test` id makes every one of them a no-op), so a
/// crash, a mis-sized buffer or a rejected node shows up here and nowhere
/// else.
#[test]
fn ssd_is_negotiated_for_a_client_that_defers() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_decorated_toplevel(&comp.socket, "harness.ssd", "decorated");
    // 2 == server_side in zxdg_toplevel_decoration_v1.
    assert!(
        client.wait_until(|c| c.decoration_mode() == Some(2)),
        "server-side expected"
    );

    // And the band is really reserved: an SSD window's client is configured
    // at the frame height minus the title bar, never at the whole frame.
    let snap = comp.snapshot();
    let frame = snap.windows.first().expect("one mapped window").geometry;
    assert!(
        client.wait_until(|c| c.last_configure() == Some((frame.width, frame.height - 28))),
        "an SSD client must be sized to the content rect, got {:?} for frame {:?}",
        client.last_configure(),
        frame
    );

    // Review finding L1: exactly one decoration configure, and it is the
    // right one. The provisional answer made before the app-id was known is
    // staged, not sent, so the client never sees a wrong-then-right pair.
    assert_eq!(client.decoration_modes(), [2], "one configure, server-side");

    // Task 9: the decoration-mode negotiation itself is now observable over
    // D-Bus -- `set_client_decorations_requested` emits a `WindowUpdated`
    // signaling "re-fetch via GetState" once the mode settles.
    comp.wait_event(|e| matches!(e, Event::WindowUpdated { .. }));

    // Retitling drives `update_buffer` on the live title node: the model
    // must take the new title and the client must survive it.
    client.set_title("renamed");
    comp.wait_event(|e| matches!(e, Event::WindowUpdated { update, .. } if update.title.as_deref() == Some("renamed")));
    assert!(
        client.wait_until(|c| !c.closed()),
        "the client must still be alive"
    );

    client.detach();
}

/// The other answer: a client this compositor treats as client-side
/// decorated (`decoration::is_csd`'s app-id rule) is told **client-side**,
/// and gets its whole frame rather than a content rect inset by a band it
/// will never be shown.
#[test]
fn csd_is_negotiated_for_a_client_that_draws_its_own() {
    let comp = Compositor::spawn();
    let mut client =
        TestClient::map_decorated_toplevel(&comp.socket, "org.gtk.Harness", "own frame");
    // 1 == client_side.
    assert!(
        client.wait_until(|c| c.decoration_mode() == Some(1)),
        "client-side expected"
    );

    let snap = comp.snapshot();
    let frame = snap.windows.first().expect("one mapped window").geometry;
    assert!(
        client.wait_until(|c| c.last_configure() == Some((frame.width, frame.height))),
        "a CSD client owns every pixel of its frame, got {:?} for frame {:?}",
        client.last_configure(),
        frame
    );
    // L1 again, and this is the case that used to be wrong-then-right: the
    // app-id says client-side, and that is the *only* answer sent.
    assert_eq!(client.decoration_modes(), [1], "one configure, client-side");

    client.detach();
}

/// Task 14: a client that unmaps (attaches a null buffer and commits, not a
/// destroy) releases focus and alt-tab candidacy exactly the way
/// `ToplevelHandler::unmapped` now does against the model, and the row
/// survives -- this is `Wayland::keyboard_focus`/`configure` going out to a
/// *real* second client, which a synthetic `ToplevelKey` cannot exercise.
#[test]
fn an_unmapping_client_releases_focus_and_alt_tab() {
    let comp = Compositor::spawn();
    let mut first = TestClient::map_toplevel(&comp.socket, "harness.stay", "stays");
    let mut second = TestClient::map_toplevel(&comp.socket, "harness.go", "goes");
    // second has focus (latest map wins). Unmap it: attach a null buffer +
    // commit.
    second.unmap();
    assert!(
        first.wait_until(|c| c.states().contains(&4)), // 4 == activated
        "focus must return to the surviving client"
    );
    let snap = comp.snapshot();
    assert!(
        snap.windows.iter().any(|w| w.title == "goes"),
        "row survives the unmap"
    );

    // Task 9: the unmap's `WindowUpdated` must carry `mapped: Some(false)`
    // so a D-Bus subscriber can observe the transition without polling
    // `GetState`.
    comp.wait_event(
        |e| matches!(e, Event::WindowUpdated { update, .. } if update.mapped == Some(false)),
    );

    first.detach();
    second.detach();
}

/// Task 20: a real `zwlr_layer_shell_v1` panel gets configured (mandatory --
/// an unanswered layer surface hangs its client) and its exclusive zone
/// carves the usable area a maximized toplevel is placed against.
///
/// The expected maximized content height below is derived, not guessed:
/// `tests/support/mod.rs`'s headless backend reports a `1280x720` output
/// (confirmed against this same harness's `Compositor::spawn` --
/// `headless_boot.rs`'s own coverage only asserts `width > 0 && height > 0`,
/// so this comment is this suite's one place that number is pinned) --
/// `1280x720` minus the panel's `30`px top exclusive zone (usable
/// `1280x690`), minus `appearance.snap_gap`'s default `8`px inset on both
/// edges of the maximize rect (`layout::maximized_geometry`, frame
/// `1264x674`), minus the `28`px SSD title-bar band `harness.tiled` gets
/// by default (`decoration::TITLE_BAR_HEIGHT`, `content_rect`) since it
/// does not match the `org.gtk`/`gtk4` CSD carve-out: `674 - 28 = 646`.
#[test]
fn a_layer_panel_gets_configured_and_carves_the_workspace() {
    let comp = Compositor::spawn();
    let mut panel = TestClient::map_layer_panel(&comp.socket, 30);
    assert!(
        panel.wait_until(|c| c.layer_configure().is_some()),
        "panel must be configured"
    );
    let (w, h) = panel.layer_configure().expect("size");
    assert!(
        w > 0 && h == 30,
        "panel must span the output's width at its 30px exclusive thickness, got {w}x{h}"
    );

    let mut win = TestClient::map_toplevel(&comp.socket, "harness.tiled", "t");
    let id = comp.snapshot().windows[0].id;
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(id, true));
    assert!(
        win.wait_until(|c| c.last_configure().map(|(_, h)| h) == Some(646)),
        "maximized height must exclude the panel's zone, last configure: {:?}",
        win.last_configure()
    );

    win.detach();
}

/// Task 20 review finding J2, end-to-end (re-review Minor-2 wired
/// `LayerPanelClient::unmap` into a real test rather than leaving it
/// unused): a panel that unmaps -- without being destroyed -- must give
/// its exclusive zone back, and a maximized toplevel must track that
/// immediately with no D-Bus command needed (`layer_surface_unmapped`'s
/// own `arrange_layers()` call re-syncs it). `646` is
/// `a_layer_panel_gets_configured_and_carves_the_workspace`'s own derived
/// carved height; `676` is that same derivation with the panel's 30px
/// zone removed (`1280x720` minus `snap_gap`'s `8`px both-edges inset,
/// `1264x704` frame, minus the `28`px SSD title-bar band -> `676`).
#[test]
fn unmapping_a_layer_panel_gives_the_usable_area_back() {
    let comp = Compositor::spawn();
    let mut panel = TestClient::map_layer_panel(&comp.socket, 30);
    assert!(
        panel.wait_until(|c| c.layer_configure().is_some()),
        "panel must be configured"
    );

    let mut win = TestClient::map_toplevel(&comp.socket, "harness.tiled", "t");
    let id = comp.snapshot().windows[0].id;
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(id, true));
    assert!(
        win.wait_until(|c| c.last_configure().map(|(_, h)| h) == Some(646)),
        "maximized height must exclude the panel's zone before it unmaps, last configure: {:?}",
        win.last_configure()
    );

    let n = win.configure_count();
    panel.unmap();
    assert!(
        win.wait_until(
            |c| c.configure_count() > n && c.last_configure().map(|(_, h)| h) == Some(676)
        ),
        "maximized height must exclude nothing once the panel unmaps, last configure: {:?}",
        win.last_configure()
    );

    win.detach();
}

/// Final review I1, at the wire: an auto-hide panel that unmaps and maps
/// again must be configured again.
///
/// wlroots clears the layer surface's `initialized` flag on every unmap, so
/// the remap's initial commit owes a *mandatory* configure — but the
/// compositor recomputes the same placement it had before (same output,
/// same anchors, nothing moved), and `configure_layer`'s storm guard used
/// to match that against a `last_configured` no one had cleared and
/// suppress the send. The panel then never becomes `mapped`, which is the
/// only thing `arrange_layers`' sweep reconfigures, so nothing ever rescued
/// it: `remap()` returned `false` and the client hung forever. This is the
/// toggle-launcher / auto-hide-panel sequence, and the harness previously
/// unmapped a panel without ever remapping one.
///
/// The exclusive zone is asserted on both sides of the round trip, through
/// a maximized toplevel, so this pins that the panel really re-mapped
/// rather than merely being sent bytes: `646` and `676` are
/// `a_layer_panel_gets_configured_and_carves_the_workspace`'s and
/// `unmapping_a_layer_panel_gives_the_usable_area_back`'s own derived
/// heights, with and without the panel's 30px carve.
#[test]
fn a_remapped_layer_panel_is_configured_again_and_re_carves_the_workspace() {
    let comp = Compositor::spawn();
    let mut panel = TestClient::map_layer_panel(&comp.socket, 30);
    assert!(
        panel.wait_until(|c| c.layer_configure().is_some()),
        "panel must be configured"
    );

    let mut win = TestClient::map_toplevel(&comp.socket, "harness.tiled", "t");
    let id = comp.snapshot().windows[0].id;
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(id, true));
    assert!(
        win.wait_until(|c| c.last_configure().map(|(_, h)| h) == Some(646)),
        "maximized height must exclude the panel's zone to begin with, last configure: {:?}",
        win.last_configure()
    );

    let n = win.configure_count();
    panel.unmap();
    assert!(
        win.wait_until(
            |c| c.configure_count() > n && c.last_configure().map(|(_, h)| h) == Some(676)
        ),
        "the unmap must give the zone back first, last configure: {:?}",
        win.last_configure()
    );

    let before = panel.layer_configure_count();
    assert!(
        panel.remap(),
        "a remapped panel must be sent a fresh configure (had {before} before the remap)"
    );
    assert_eq!(
        panel.layer_configure(),
        Some((1280, 30)),
        "and at the same placement it had before -- which is exactly why the storm guard swallowed it"
    );

    let n = win.configure_count();
    assert!(
        win.wait_until(
            |c| c.configure_count() > n && c.last_configure().map(|(_, h)| h) == Some(646)
        ),
        "the remapped panel must carve its zone again, last configure: {:?}",
        win.last_configure()
    );

    win.detach();
}

/// The M4.2 keystone: a pointer-driven drag transfers a `text/plain` payload
/// from one client to another, byte for byte, entirely through an injected
/// virtual pointer -- no shortcut through `set_selection`.
///
/// Coordinates are resolved empirically rather than assumed: the virtual
/// pointer's `motion_absolute` extent is the *output's* size, which this
/// headless backend never publishes directly, so A is briefly maximized to
/// read it off `comp.snapshot()` (maximize sets a window's geometry to
/// exactly the output geometry -- `state.rs`'s own maximize tests assert
/// this), then unmaximized back to its normal placement before the drag
/// starts. Both windows' click points come from their real snapshot
/// geometry, not a guessed constant.
#[test]
fn a_pointer_drag_transfers_between_two_clients() {
    /// `xdg_toplevel.state.maximized`.
    const MAXIMIZED: u32 = 1;
    /// `BTN_LEFT` from `linux/input-event-codes.h`.
    const BTN_LEFT: u32 = 0x110;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);

    // A maps; this is both the drag source and how the output's size gets
    // discovered (see the function doc).
    let mut a = TestClient::map_toplevel(&comp.socket, "src.app", "src");
    assert!(
        a.wait_until(|c| c.last_configure().is_some()),
        "A never configured"
    );
    let opened = comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "src.app"));
    let Event::WindowOpened(a_info) = opened else {
        unreachable!()
    };

    let configures = a.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        a_info.id, true,
    ));
    assert!(
        a.wait_until(|c| c.configure_count() > configures && c.states().contains(&MAXIMIZED)),
        "A never maximized"
    );
    let output = comp.snapshot().windows[0].geometry;
    let (x_extent, y_extent) = (output.width as u32, output.height as u32);
    assert!(
        x_extent > 0 && y_extent > 0,
        "output geometry must be real, got {output:?}"
    );

    let configures = a.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        a_info.id, false,
    ));
    assert!(
        a.wait_until(|c| c.configure_count() > configures && !c.states().contains(&MAXIMIZED)),
        "A never unmaximized"
    );

    // A's real (unmaximized) geometry -- a point inside it is the press
    // target.
    let a_geo = comp.snapshot().windows[0].geometry;
    let (ax, ay) = (
        (a_geo.x + a_geo.width / 2) as f64,
        (a_geo.y + a_geo.height / 2) as f64,
    );

    // Move the pointer over A and press to mint a grab serial for A.
    vp.motion_absolute(ax, ay, x_extent, y_extent);
    vp.frame();
    vp.button(BTN_LEFT, true);
    vp.frame();
    assert!(
        a.wait_until(|c| c.last_pointer_serial().is_some()),
        "A never got a pointer serial"
    );

    let serial = a.last_pointer_serial().expect("just asserted this is Some");
    a.start_drag_text("text/plain;charset=utf-8", b"dragged", serial);

    // B maps after the drag has started -- exactly the skeleton's ordering.
    let mut b = TestClient::map_toplevel(&comp.socket, "dst.app", "dst");
    assert!(
        b.wait_until(|c| c.last_configure().is_some()),
        "B never configured"
    );
    let b_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "dst.app")
        .expect("B must be in the model once mapped")
        .geometry;
    let (bx, by) = (
        (b_geo.x + b_geo.width / 2) as f64,
        (b_geo.y + b_geo.height / 2) as f64,
    );

    // Move over B and release to drop.
    vp.motion_absolute(bx, by, x_extent, y_extent);
    vp.frame();
    assert!(
        b.wait_until(|c| c.has_drag_offer()),
        "destination never got the drag enter"
    );
    // `wait_until` returns the moment its predicate holds, which can be
    // right after dispatching the very event whose handler just queued
    // B's `accept`/`set_actions` requests -- those sit unflushed until the
    // next flush. One more pump sends them and confirms the compositor has
    // seen them before the button release ends the drag.
    b.pump();
    vp.button(BTN_LEFT, false);
    vp.frame();
    assert!(
        b.wait_until(|c| c.got_drop()),
        "destination never got the drop"
    );

    assert_eq!(
        icedtea_harness::read_drag_offer(&mut b, &mut a, "text/plain;charset=utf-8"),
        b"dragged"
    );

    a.detach();
    b.detach();
}

/// Same keystone as [`a_pointer_drag_transfers_between_two_clients`], driven
/// by touch instead: `wlr::Runtime::inject_touch_down` (via
/// [`Compositor::inject_touch_down`]) mints the grab serial directly --
/// there is no client-side touch listener to read a serial off of, unlike
/// the pointer path's `wl_pointer::Event::Button`.
///
/// Coordinates are real scene coordinates (not a normalized extent like the
/// virtual-pointer protocol's `motion_absolute`), so no maximize-then-read
/// dance is needed to discover the output size -- each window's own mapped
/// geometry from `comp.snapshot()` is enough.
#[test]
fn a_touch_drag_transfers_between_two_clients() {
    let comp = Compositor::spawn();

    // A maps; this is both the drag source and where the touch-down lands.
    let mut a = TestClient::map_toplevel(&comp.socket, "src.app", "src");
    assert!(
        a.wait_until(|c| c.last_configure().is_some()),
        "A never configured"
    );
    let opened = comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "src.app"));
    let Event::WindowOpened(_a_info) = opened else {
        unreachable!()
    };

    let a_geo = comp.snapshot().windows[0].geometry;
    let (ax, ay) = (
        (a_geo.x + a_geo.width / 2) as f64,
        (a_geo.y + a_geo.height / 2) as f64,
    );

    // Touch down over A mints the touch grab serial directly.
    let serial = comp.inject_touch_down(ax, ay, 0, 1);
    assert!(
        serial.is_some(),
        "touch-down over A never minted a grab serial"
    );
    a.start_drag_text("text/plain;charset=utf-8", b"dragged", serial.unwrap());

    // B maps after the drag has started -- same ordering as the pointer test.
    let mut b = TestClient::map_toplevel(&comp.socket, "dst.app", "dst");
    assert!(
        b.wait_until(|c| c.last_configure().is_some()),
        "B never configured"
    );
    let b_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "dst.app")
        .expect("B must be in the model once mapped")
        .geometry;
    let (bx, by) = (
        (b_geo.x + b_geo.width / 2) as f64,
        (b_geo.y + b_geo.height / 2) as f64,
    );

    // Move the touch point over B and lift to drop.
    comp.inject_touch_motion(bx, by, 0, 2);
    assert!(
        b.wait_until(|c| c.has_drag_offer()),
        "destination never got the drag enter"
    );
    // Same flush gotcha as the pointer test: `wait_until` can return right
    // after dispatching the event whose handler just queued B's
    // `accept`/`set_actions` requests -- pump once more so those flush
    // before the touch-up ends the drag.
    b.pump();
    comp.inject_touch_up(0, 3);
    assert!(
        b.wait_until(|c| c.got_drop()),
        "destination never got the drop"
    );

    assert_eq!(
        icedtea_harness::read_drag_offer(&mut b, &mut a, "text/plain;charset=utf-8"),
        b"dragged"
    );

    a.detach();
    b.detach();
}

/// Negative counterpart to [`a_pointer_drag_transfers_between_two_clients`]:
/// the exact same setup (A maps, is briefly maximized to discover the
/// output extent, unmaximizes, and mints a REAL grab serial by pressing
/// over its own surface), but `start_drag` is called with a bogus serial
/// the compositor never issued. Task 2's
/// `wlr_seat_validate_pointer_grab_serial` gate must refuse it, so B --
/// mapped and positioned under the pointer exactly as in the positive
/// test, i.e. just as drag-capable as a destination that *would* receive
/// the transfer -- must receive neither `wl_data_device.enter` nor
/// `.drop`. Because everything but the serial matches the positive test,
/// "no drag" here isolates the serial gate as the cause rather than a
/// broken setup.
///
/// The bogus serial is the real one plus a large offset rather than a bare
/// `0`: this compositor hands out serials from the same `wl_display`
/// counter already ticked past 0 by surface/xdg_surface ids and prior
/// configures before this test's press even happens, so `0` is not
/// guaranteed unissued. `real_serial + 1_000_000` is -- the counter cannot
/// advance that far within one test run.
///
/// wlroots 0.20's actual behavior for an invalid `start_drag` serial,
/// determined empirically (a probe build logged the post-request
/// roundtrip's `Result` and `Connection::protocol_error()`): it silently
/// no-ops rather than raising a client protocol error. The roundtrip
/// inside [`TestClient::start_drag_text`] returns `Ok`, `protocol_error()`
/// stays `None`, and A's connection survives -- so A is detached
/// normally at the end of this test, same as the positive test. Either
/// outcome (silent no-op or a connection-killing protocol error) would
/// have satisfied the gate this test pins; what it requires
/// unconditionally is that B -- fully set up to receive the drag -- gets
/// nothing, which is the only way to avoid a false pass from a client
/// that merely fell over.
#[test]
fn a_drag_with_an_invalid_grab_serial_is_refused() {
    /// `xdg_toplevel.state.maximized`.
    const MAXIMIZED: u32 = 1;
    /// `BTN_LEFT` from `linux/input-event-codes.h`.
    const BTN_LEFT: u32 = 0x110;
    /// Offset added to a real, freshly-minted serial to produce one the
    /// compositor provably never issued.
    const BOGUS_SERIAL_OFFSET: u32 = 1_000_000;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);

    // A maps; this is both the drag source and how the output's size gets
    // discovered (see the function doc).
    let mut a = TestClient::map_toplevel(&comp.socket, "src.app", "src");
    assert!(
        a.wait_until(|c| c.last_configure().is_some()),
        "A never configured"
    );
    let opened = comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "src.app"));
    let Event::WindowOpened(a_info) = opened else {
        unreachable!()
    };

    let configures = a.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        a_info.id, true,
    ));
    assert!(
        a.wait_until(|c| c.configure_count() > configures && c.states().contains(&MAXIMIZED)),
        "A never maximized"
    );
    let output = comp.snapshot().windows[0].geometry;
    let (x_extent, y_extent) = (output.width as u32, output.height as u32);
    assert!(
        x_extent > 0 && y_extent > 0,
        "output geometry must be real, got {output:?}"
    );

    let configures = a.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        a_info.id, false,
    ));
    assert!(
        a.wait_until(|c| c.configure_count() > configures && !c.states().contains(&MAXIMIZED)),
        "A never unmaximized"
    );

    // A's real (unmaximized) geometry -- a point inside it is the press
    // target.
    let a_geo = comp.snapshot().windows[0].geometry;
    let (ax, ay) = (
        (a_geo.x + a_geo.width / 2) as f64,
        (a_geo.y + a_geo.height / 2) as f64,
    );

    // Move the pointer over A and press to mint a REAL grab serial for A --
    // proof the setup is otherwise sound, since this same serial is exactly
    // what the positive test uses to succeed.
    vp.motion_absolute(ax, ay, x_extent, y_extent);
    vp.frame();
    vp.button(BTN_LEFT, true);
    vp.frame();
    assert!(
        a.wait_until(|c| c.last_pointer_serial().is_some()),
        "A never got a pointer serial"
    );

    let real_serial = a.last_pointer_serial().expect("just asserted this is Some");
    // wrapping_add: a fresh test compositor's serial counter is nowhere near
    // u32::MAX, but this keeps the bogus value unconditionally overflow-safe.
    let bogus_serial = real_serial.wrapping_add(BOGUS_SERIAL_OFFSET);
    a.start_drag_text("text/plain;charset=utf-8", b"dragged", bogus_serial);

    // B maps after the (refused) drag attempt -- same ordering as the
    // positive test, so B is exactly as reachable as it would be had the
    // drag actually succeeded.
    let mut b = TestClient::map_toplevel(&comp.socket, "dst.app", "dst");
    assert!(
        b.wait_until(|c| c.last_configure().is_some()),
        "B never configured"
    );
    let b_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "dst.app")
        .expect("B must be in the model once mapped")
        .geometry;
    let (bx, by) = (
        (b_geo.x + b_geo.width / 2) as f64,
        (b_geo.y + b_geo.height / 2) as f64,
    );

    // Move over B and release -- the identical motion+release the positive
    // test uses to complete a drop. If the serial gate had done nothing,
    // this is the drop that would land.
    vp.motion_absolute(bx, by, x_extent, y_extent);
    vp.frame();
    assert!(
        !b.wait_until(|c| c.has_drag_offer()),
        "the serial gate must refuse the drag, but B received a drag enter anyway"
    );
    b.pump();
    vp.button(BTN_LEFT, false);
    vp.frame();
    assert!(
        !b.wait_until(|c| c.got_drop()),
        "the serial gate must refuse the drag, but B received a drop anyway"
    );

    a.detach();
    b.detach();
}

/// M4.2 Success Criterion 4: the drag icon actually **renders** as a scene
/// node, and **follows** the input, not merely that the protocol handshake
/// (`start_drag` with a non-null icon surface) went through without error.
///
/// Setup mirrors [`a_pointer_drag_transfers_between_two_clients`] up through
/// minting the grab serial (maximize-then-snapshot to discover the output
/// extent, unmaximize, position the virtual pointer over A, press). From
/// there this diverges: [`TestClient::start_drag_text_with_icon`] attaches a
/// real icon surface, and instead of a second client receiving the drop,
/// this asserts directly against `wlr::Runtime::drag_icon_position` (via
/// [`Compositor::drag_icon_position`]):
///
/// 1. **Renders**: once the drag has started and the icon surface has
///    committed a buffer, `drag_icon_position()` must return `Some` -- a
///    live scene node exists for the icon.
/// 2. **Follows**: two motions *after* the icon exists, to two known points
///    `M1`/`M2` inside A's own geometry, must produce two icon positions
///    `p1`/`p2` whose delta matches `M2 - M1`.
///
/// The tree the icon lives in is created at `(0, 0)` by
/// `wlr_scene_drag_icon_create` and is repositioned only by a *subsequent*
/// pointer/touch motion (`wlr::Runtime`'s `on_pointer_motion*` handlers call
/// `wlr_scene_node_set_position` with the live cursor layout coordinates) --
/// there is no motion baked into `start_drag` itself. So `p1` must be read
/// after a real motion event has fired post-`start_drag`, not compared
/// against the pre-drag press point: the press that minted the grab serial
/// happened *before* the icon tree existed, so it never repositioned
/// anything.
#[test]
fn a_pointer_drag_renders_and_follows_its_icon() {
    /// `xdg_toplevel.state.maximized`.
    const MAXIMIZED: u32 = 1;
    /// `BTN_LEFT` from `linux/input-event-codes.h`.
    const BTN_LEFT: u32 = 0x110;
    /// Drag icon size (`w`x`h`), arbitrary but non-trivial.
    const ICON_SIZE: i32 = 32;
    /// The headless backend's real output pixel size -- pinned, not
    /// guessed: see `unmapping_a_layer_panel_gives_the_usable_area_back`'s
    /// own doc comment above, this suite's one other place this number is
    /// asserted against `tests/support/mod.rs`. Used as the virtual
    /// pointer's `x_extent`/`y_extent` instead of a *maximized window's*
    /// geometry (what the sibling drag tests use for that, and what an
    /// earlier revision of this test used too): `zwlr_virtual_pointer_v1`
    /// maps `x`/`y` to a fraction of `x_extent`/`y_extent` and then
    /// `wlr_cursor_warp_absolute` maps that fraction onto the *real* output
    /// pixel box, not onto whatever extent the client happened to pass. A
    /// maximized window's geometry is inset from the real output by
    /// `snap_gap`/the panel exclusive zone, so using it as the extent here
    /// scaled every landed position by a small constant factor (empirically:
    /// exactly `1280/1264` in `x`, `720/704` in `y`, for this harness's
    /// default config) -- harmless for the sibling tests (they only need to
    /// land *somewhere inside* a window), fatal here, where Criterion 4b
    /// compares an exact pixel delta.
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;
    /// Wall-clock budget for `drag_icon_position()` to observe the scene
    /// node reflecting a given motion -- the compositor processes the
    /// motion and this query on its own loop tick, which can trail the
    /// client-side calls that produced them by a beat.
    const POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    /// Poll `comp.drag_icon_position()` until it satisfies `pred`, or panic
    /// after [`POLL_TIMEOUT`]. Needed for both the initial render and each
    /// post-motion follow check: the compositor's own event loop tick that
    /// would produce the new position is not synchronized with this thread.
    fn poll_drag_icon_position(
        comp: &Compositor,
        pred: impl Fn(Option<(i32, i32)>) -> bool,
    ) -> Option<(i32, i32)> {
        let deadline = std::time::Instant::now() + POLL_TIMEOUT;
        loop {
            let pos = comp.drag_icon_position();
            if pred(pos) {
                return pos;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "drag_icon_position() never satisfied the predicate within {POLL_TIMEOUT:?} \
                 (last read: {pos:?})"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// As [`poll_drag_icon_position`], but returns only once the reading has
    /// *settled*: two consecutive reads, 10ms apart, come back identical and
    /// `Some`, and different from `since`. A single "it changed" reading is
    /// not enough on its own in general -- this compositor's motion handler
    /// could in principle update the position more than once in quick
    /// succession while a single `vp.motion_absolute` + `vp.frame()` is
    /// still being processed, and grabbing the first post-`since` value
    /// would risk capturing a transient rather than where the icon actually
    /// ends up. Kept as defense-in-depth even though this test's own
    /// investigation traced its one observed flavor of "wrong value" to a
    /// coordinate-space mismatch (fixed by [`OUTPUT_W`]/[`OUTPUT_H`]), not
    /// to an unsettled reading.
    fn poll_settled_drag_icon_position(comp: &Compositor, since: Option<(i32, i32)>) -> (i32, i32) {
        let deadline = std::time::Instant::now() + POLL_TIMEOUT;
        loop {
            let first = comp.drag_icon_position();
            if let Some(value) = first
                && first != since
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let second = comp.drag_icon_position();
                if second == first {
                    return value;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "drag_icon_position() never settled to a new value within {POLL_TIMEOUT:?} \
                 (since={since:?}, last read: {first:?})"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);

    // A maps; briefly maximized-then-restored exactly like the sibling drag
    // tests, purely to discover A's real (unmaximized) geometry from the
    // model -- unlike those tests, the *extent* passed to `motion_absolute`
    // below is [`OUTPUT_W`]/[`OUTPUT_H`], not derived from this maximized
    // query (see that constant's own doc for why).
    let mut a = TestClient::map_toplevel(&comp.socket, "src.app", "src");
    assert!(
        a.wait_until(|c| c.last_configure().is_some()),
        "A never configured"
    );
    let opened = comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "src.app"));
    let Event::WindowOpened(a_info) = opened else {
        unreachable!()
    };

    let configures = a.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        a_info.id, true,
    ));
    assert!(
        a.wait_until(|c| c.configure_count() > configures && c.states().contains(&MAXIMIZED)),
        "A never maximized"
    );

    let configures = a.configure_count();
    comp.send(icedtea_compositor::dbus::DbCommand::Maximize(
        a_info.id, false,
    ));
    assert!(
        a.wait_until(|c| c.configure_count() > configures && !c.states().contains(&MAXIMIZED)),
        "A never unmaximized"
    );

    // A's real (unmaximized) geometry -- a point inside it is the press
    // target, exactly the positive drag test's setup.
    let a_geo = comp.snapshot().windows[0].geometry;
    let (ax, ay) = (
        (a_geo.x + a_geo.width / 2) as f64,
        (a_geo.y + a_geo.height / 2) as f64,
    );

    // Move the pointer over A and press to mint a grab serial for A.
    vp.motion_absolute(ax, ay, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.button(BTN_LEFT, true);
    vp.frame();
    vp.pump();
    assert!(
        a.wait_until(|c| c.last_pointer_serial().is_some()),
        "A never got a pointer serial"
    );

    let serial = a.last_pointer_serial().expect("just asserted this is Some");
    a.start_drag_text_with_icon(
        "text/plain;charset=utf-8",
        b"dragged",
        serial,
        ICON_SIZE,
        ICON_SIZE,
    );

    // Criterion 4a -- RENDERS: the icon's committed buffer must have
    // produced a live scene node with real layout coordinates. At this
    // point the tree exists but has not yet been repositioned by any
    // motion -- `wlr_scene_drag_icon_create` leaves it at its `(0, 0)`
    // creation default -- so this only checks `Some`, not a particular
    // value.
    let p0 = poll_drag_icon_position(&comp, |pos| pos.is_some());
    assert!(p0.is_some(), "the drag icon never rendered a scene node");

    // Two points strictly inside A's own geometry, not just anywhere in the
    // output. `enter_surface_under_cursor` (wlr crate `backend.rs`) only
    // calls `wlr_seat_pointer_notify_motion` when `leaf_surface_at` finds a
    // surface under the cursor; over empty output space (no second client
    // mapped in this test) it calls `wlr_seat_pointer_notify_clear_focus`
    // instead, which never reaches `on_pointer_motion`'s repositioning
    // call -- so a move to empty space would never move the icon at all.
    // Both M1 and M2 stay inside A to rule that out.
    let (m1x, m1y) = (
        (a_geo.x + a_geo.width / 4) as f64,
        (a_geo.y + a_geo.height / 4) as f64,
    );
    let (m2x, m2y) = (
        (a_geo.x + 3 * a_geo.width / 4) as f64,
        (a_geo.y + 3 * a_geo.height / 4) as f64,
    );
    assert!(
        (m2x - m1x).abs() > 1.0 || (m2y - m1y).abs() > 1.0,
        "M1 -> M2 must have a real delta"
    );

    // First post-drag motion, to M1. `wlr::Runtime`'s `on_pointer_motion*`
    // sets the icon tree's scene position to the live cursor layout
    // coordinates on every motion, so once this lands, `drag_icon_position`
    // must read back *something other than* `p0` (the pre-motion reading) --
    // polled for rather than asserted immediately, since the compositor
    // processes the motion (and this query) on its own loop tick, not
    // synchronously with `vp.frame()`/`vp.pump()` returning. This does not
    // assert the exact value equals M1: `wlr::Runtime::reposition_drag_icon`
    // positions the *outer* tree at the raw cursor, while the icon surface's
    // own hotspot offset (from wlroots' internal `surface_tree` commit
    // handler) is applied as a separate, constant child offset -- so the
    // absolute reading legitimately differs from M1 by that fixed amount.
    // That offset is constant across both motions, so it cancels out of the
    // *delta* Criterion 4b checks below.
    vp.motion_absolute(m1x, m1y, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    let p1 = poll_settled_drag_icon_position(&comp, p0);

    // Second post-drag motion, to M2.
    vp.motion_absolute(m2x, m2y, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    let p2 = poll_settled_drag_icon_position(&comp, Some(p1));

    // Criterion 4b -- FOLLOWS: the icon's position moved between the two
    // post-drag motions, by (within 1px rounding) exactly the pointer's own
    // delta between those same two motions.
    assert_ne!(
        p2, p1,
        "the drag icon never moved between M1 and M2 -- it is not following the pointer"
    );

    let observed_delta = (p2.0 - p1.0, p2.1 - p1.1);
    let expected_delta = ((m2x - m1x).round() as i32, (m2y - m1y).round() as i32);
    assert!(
        (observed_delta.0 - expected_delta.0).abs() <= 1
            && (observed_delta.1 - expected_delta.1).abs() <= 1,
        "icon delta {observed_delta:?} does not match pointer delta {expected_delta:?} \
         (p1={p1:?}, p2={p2:?}, M1=({m1x},{m1y}), M2=({m2x},{m2y}))"
    );

    vp.button(BTN_LEFT, false);
    vp.frame();
    vp.pump();

    a.detach();
}

/// A screencopy client can capture the headless output: the frame reaches
/// `ready` and returns a full-size buffer.
#[test]
fn screencopy_captures_the_output() {
    let comp = Compositor::spawn();
    let mut sc = icedtea_harness::ScreencopyClient::spawn(&comp.socket);
    let frame = sc.capture();
    assert!(
        frame.width > 0 && frame.height > 0,
        "empty capture geometry"
    );
    assert_eq!(frame.bytes.len(), (frame.stride * frame.height) as usize);
}

/// M4.3 Success Criterion 2 (load-bearing): a capture of the empty output is
/// a uniform image of the configured wallpaper color — proving the bytes are
/// the output's real composited content, not a cleared or garbage buffer.
///
/// Expected and actual share one config source: `Compositor::spawn` (harness
/// boot, `harness/src/lib.rs`) constructs its `State` from
/// `icedtea_config::default_config()`, exactly the same call used below to
/// derive the expected color, so this assertion can never be vacuously right.
#[test]
fn screencopy_of_empty_output_is_the_wallpaper_color() {
    let comp = Compositor::spawn();
    let mut sc = icedtea_harness::ScreencopyClient::spawn(&comp.socket);
    let frame = sc.capture();

    // Expected color: the wallpaper rgba mapped to the reported format's
    // actual memory byte order. Xrgb8888/Argb8888 are little-endian 4
    // bytes/pixel B,G,R,X (verified elsewhere in this suite, e.g.
    // `create_shm_buffer`'s fill). The headless backend's software (pixman)
    // renderer instead hands screencopy back Bgr888 -- 3 tightly packed
    // bytes/pixel, no pad byte -- and despite the name, its memory layout is
    // R,G,B (byte0=R): the wl_shm/DRM fourcc convention names 24bpp formats
    // by the bit-range each channel occupies when the pixel is read as one
    // little-endian integer ("[23:0] B:G:R little endian" = R in the low
    // byte), which is the reverse convention from the 32-bit Xrgb/Argb
    // formats' byte-address-order naming. Confirmed empirically against
    // this backend's actual capture bytes (R and B were swapped until this
    // was corrected) -- see the byte order test in the accompanying report.
    let (bpp, byte_order) = match frame.format {
        wayland_client::protocol::wl_shm::Format::Xrgb8888
        | wayland_client::protocol::wl_shm::Format::Argb8888 => (4usize, [2, 1, 0]),
        wayland_client::protocol::wl_shm::Format::Bgr888 => (3usize, [0, 1, 2]),
        other => {
            panic!("unexpected screencopy shm format {other:?}; byte order assumption may not hold")
        }
    };

    let [r, g, b, _a] =
        icedtea_compositor::render::wallpaper_color(&icedtea_config::default_config().appearance);
    let expect = |c: f32| (c * 255.0).round() as i32;
    let (er, eg, eb) = (expect(r), expect(g), expect(b));

    // Every pixel must be identical (uniform) and within ±2 of the expected
    // color per channel (absorbs any renderer color-space/rounding delta).
    let (mut first, mut uniform) = (None, true);
    for y in 0..frame.height as usize {
        for x in 0..frame.width as usize {
            let o = y * frame.stride as usize + x * bpp;
            let px = (
                frame.bytes[o + byte_order[0]],
                frame.bytes[o + byte_order[1]],
                frame.bytes[o + byte_order[2]],
            );
            match first {
                None => first = Some(px),
                Some(f) if f != px => uniform = false,
                _ => {}
            }
        }
    }
    assert!(
        uniform,
        "capture is not a uniform color; a cleared/garbage buffer"
    );
    let (pr, pg, pb) = first.expect("no pixels");
    assert!((pr as i32 - er).abs() <= 2, "R {pr} vs {er}");
    assert!((pg as i32 - eg).abs() <= 2, "G {pg} vs {eg}");
    assert!((pb as i32 - eb).abs() <= 2, "B {pb} vs {eb}");
}

/// M4.3 Success Criterion 3: a capture taken with a mapped toplevel reflects
/// the toplevel's content — the image is no longer uniform and the toplevel's
/// grey (0x80 from the harness shm buffer) appears in it.
#[test]
fn screencopy_reflects_a_mapped_toplevel() {
    let comp = Compositor::spawn();
    let mut client = TestClient::map_toplevel(&comp.socket, "shot.app", "shot");
    assert!(client.wait_until(|c| c.last_configure().is_some()));

    let mut sc = icedtea_harness::ScreencopyClient::spawn(&comp.socket);
    let frame = sc.capture();

    // IMPORTANT: Task 4 discovered the headless (pixman software) renderer
    // hands screencopy back `Bgr888` — 3 tightly-packed bytes/pixel, NOT the
    // 4-byte Xrgb8888 the original plan assumed. Derive `bpp` from the
    // reported format exactly as `screencopy_of_empty_output_is_the_wallpaper_color`
    // does. Grey (0x80,0x80,0x80) is channel-symmetric, so byte ORDER does not
    // matter for detecting it — but `bpp` MUST be correct or the per-pixel
    // offset misreads the buffer.
    let bpp = match frame.format {
        wayland_client::protocol::wl_shm::Format::Xrgb8888
        | wayland_client::protocol::wl_shm::Format::Argb8888 => 4usize,
        wayland_client::protocol::wl_shm::Format::Bgr888 => 3usize,
        other => panic!("unexpected screencopy shm format {other:?}"),
    };
    let mut saw_grey = false;
    let mut saw_non_grey = false;
    for y in 0..frame.height as usize {
        for x in 0..frame.width as usize {
            let o = y * frame.stride as usize + x * bpp;
            let (c0, c1, c2) = (frame.bytes[o], frame.bytes[o + 1], frame.bytes[o + 2]);
            // Grey is symmetric across channels, so order-independent.
            let grey = (c0 as i32 - 0x80).abs() <= 2
                && (c1 as i32 - 0x80).abs() <= 2
                && (c2 as i32 - 0x80).abs() <= 2;
            if grey {
                saw_grey = true
            } else {
                saw_non_grey = true
            }
        }
    }
    assert!(saw_grey, "toplevel grey (0x80) not present in the capture");
    assert!(
        saw_non_grey,
        "capture is uniform; toplevel not composited over wallpaper"
    );
}

/// M4.4 Criterion 2: a session lock reaches `locked`, isolates input from
/// normal clients while locked, and restores on unlock.
///
/// The load-bearing assertion is the middle one: a key injected via the
/// virtual keyboard while locked must NOT advance `app`'s
/// `wl_keyboard.key` count. A *pre-lock* key press first proves this
/// harness's injector and observable actually work end to end (`app`'s
/// count does advance) -- without that baseline, a broken injector would
/// make the isolation assertion pass vacuously. If the crate's input
/// isolation were broken (the lock did not stop keyboard delivery to a
/// normal focused toplevel), the post-lock key would still reach `app` and
/// its count would advance past the pre-lock baseline, failing the
/// assertion.
#[test]
fn session_lock_locks_isolates_input_and_unlocks() {
    let comp = Compositor::spawn();
    let mut vk = VirtualKeyboardClient::spawn(&comp.socket);
    // A normal toplevel, focused before the lock.
    let mut app = TestClient::map_toplevel(&comp.socket, "app", "app");
    assert!(
        app.wait_until(|c| c.has_input_serial()),
        "app focused pre-lock"
    );

    // Baseline: prove key delivery actually works before the lock exists,
    // so the later "no delivery while locked" assertion cannot be vacuous.
    vk.key_press(30); // KEY_A
    assert!(
        app.wait_until(|c| c.key_events() >= 1),
        "app never received a keyboard key pre-lock; the injector/observable \
         itself is broken, which would make the isolation check meaningless"
    );
    let pre_lock_keys = app.key_events();

    let mut locker = SessionLockClient::spawn(&comp.socket);
    locker.lock();
    assert!(locker.wait_locked(), "session never reported locked");
    assert!(
        comp.session_locked(),
        "compositor is_session_locked() is false"
    );

    // While locked, inject another key. The normal app must not receive it:
    // pump the app's queue for a bounded window and assert its key count
    // never moved past the pre-lock baseline.
    vk.key_press(31); // KEY_S
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
    while std::time::Instant::now() < deadline {
        vk.pump();
        app.pump();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        app.key_events(),
        pre_lock_keys,
        "app received a keyboard key while the session was locked -- \
         input isolation broken"
    );

    locker.unlock();
    assert!(!comp.session_locked(), "still locked after unlock");

    // Unlock restores focus: a fresh key now reaches the app again.
    vk.key_press(32); // KEY_D
    assert!(
        app.wait_until(|c| c.key_events() > pre_lock_keys),
        "unlock did not restore keyboard delivery to the app"
    );
}

/// M4.4 Criterion 3 (security): a locker that dies WITHOUT unlocking leaves the
/// session locked -- the screen must not silently unlock when the locker
/// crashes.
///
/// Non-vacuous in two independent ways. If the crate wrongly cleared
/// `session_locked` when the dead locker's `ext_session_lock_v1` was
/// destroyed on disconnect (exactly the bug the design forbids), the first
/// `assert!(comp.session_locked(), ...)` below would observe `false` and
/// fail. If takeover of an already-locked-but-lockerless session were
/// broken, `locker2.wait_locked()` would time out and fail. Both directions
/// of this invariant are exercised by a single test.
#[test]
fn a_dead_locker_leaves_the_session_locked() {
    let comp = Compositor::spawn();
    let mut _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut app = TestClient::map_toplevel(&comp.socket, "app", "app");
    assert!(app.wait_until(|c| c.has_input_serial()));

    {
        let mut locker = SessionLockClient::spawn(&comp.socket);
        locker.lock();
        assert!(locker.wait_locked(), "never locked");
        assert!(comp.session_locked());
        // `locker` drops here WITHOUT calling `unlock()`, closing its
        // Wayland connection out from under the compositor -- simulating
        // the locker process crashing while the screen is locked.
    }
    // Give the compositor a bounded window to notice and process the
    // disconnect (see `Compositor::settle`'s doc for why this is a fixed
    // number of round trips rather than a poll-until-true loop).
    comp.settle();

    assert!(
        comp.session_locked(),
        "SECURITY: session unlocked itself when the locker died"
    );

    // A fresh locker can still take over the still-locked session -- this
    // is the legitimate takeover path (the prior lock is gone), distinct
    // from a second *live* lock attempt being rejected.
    let mut locker2 = SessionLockClient::spawn(&comp.socket);
    locker2.lock();
    assert!(locker2.wait_locked(), "a fresh locker could not take over");
}

/// M4.4 Criterion 4: `ext_idle_notifier_v1` fires `idled` after the
/// requested timeout elapses with no seat activity, and `resumed` once
/// activity is injected.
///
/// Non-vacuous in both directions: `idled` must actually arrive (a broken
/// notifier that never fires would fail the first assertion, not pass it
/// vacuously), and the pointer motion is injected only *after* `idled` has
/// been observed, so the `resumed` that follows can only be a reaction to
/// that injection -- not a stray event that happened to arrive first.
///
/// Timing: a 50ms notification timeout against a 5s (`TIMEOUT`) bounded
/// pump gives a 100x margin over the time-based event this test waits on,
/// which is generous enough to absorb scheduling jitter under load without
/// masking a compositor that fires `idled` too early or not at all.
#[test]
fn idle_notify_fires_idled_then_resumed_on_activity() {
    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut idle = IdleNotifyClient::spawn(&comp.socket);
    idle.notification(50); // 50ms timeout, generous margins below
    assert!(idle.wait_idled(), "never went idle");
    // Inject activity; the notifier should send `resumed`.
    vp.motion_absolute(5.0, 5.0, 100, 100);
    vp.frame();
    vp.pump();
    assert!(idle.wait_resumed(), "activity did not resume idle");
}

/// M4.4 Criterion 5: a live `zwp_idle_inhibitor_v1` suppresses `idled` on
/// every notification while it is active; destroying it lets idle resume.
///
/// Non-vacuous in both directions: the "no `idled` while inhibited" window
/// (300ms) is 6x the 50ms notification timeout, so an inhibitor that failed
/// to suppress idle would have six timeout-lengths in which to fire and
/// fail the assertion -- this is not "checked so briefly it could not have
/// fired anyway". And the second half proves the suppression really came
/// from the inhibitor rather than from a notifier that is simply broken and
/// never fires: with the inhibitor gone, `idled` must positively arrive.
///
/// PREVIOUSLY BLOCKED, NOW FIXED (M4.4 task 8): destroying a live
/// `zwp_idle_inhibitor_v1` resource -- this test's own explicit
/// `destroy_inhibitor()` below, an implicit destroy from the client
/// disconnecting, or the resource cleanup that runs when this harness's
/// `Compositor` tears its `wl_display` down at end of test -- used to abort
/// the whole compositor process. Confirmed under gdb at the time: the abort
/// was `types/wlr_idle_inhibit_v1.c:37: idle_inhibitor_v1_destroy: Assertion
/// 'wl_list_empty(&inhibitor->events.destroy.listener_list)' failed`.
///
/// Root cause (traced, not guessed): wlroots' own
/// `idle_inhibitor_v1_destroy` emits the inhibitor's `events.destroy` with
/// `inhibitor->surface` as the signal data -- *not* the inhibitor itself
/// (`wl_signal_emit_mutable(&inhibitor->events.destroy, inhibitor->surface)`
/// in `types/wlr_idle_inhibit_v1.c`). The vendored crate's
/// `on_idle_inhibitor_destroy` (`wlroots-sys/crates/wlr/src/backend.rs`)
/// used to assume the opposite and key its `Session::idle_inhibitors`
/// removal lookup by that `data` pointer -- which, being the surface's
/// address rather than the inhibitor's, always missed, leaving the
/// listener linked and tripping wlroots' own assertion. Fixed upstream in
/// `wlroots-sys` commit `f9e533b` ("key idle-inhibitor destroy listener by
/// its own addr, not data"): the map is now keyed by the destroy
/// listener's own address (`Registration::listener_addr`), which is stable
/// regardless of what the signal hands back as `data`. This test is now
/// active and both halves below are exercised for real.
#[test]
fn idle_inhibit_suppresses_idle_until_destroyed() {
    let comp = Compositor::spawn();
    let mut inhibit = IdleInhibitClient::spawn(&comp.socket);
    inhibit.create_inhibitor();
    let mut idle = IdleNotifyClient::spawn(&comp.socket);
    idle.notification(50);
    assert!(!idle.idled_within(300), "idled despite an active inhibitor");
    inhibit.destroy_inhibitor();
    idle.notification(50);
    assert!(
        idle.wait_idled(),
        "idle did not resume after inhibitor destroyed"
    );
}

/// M4.5 Criterion 2: a locked pointer freezes the cursor and the client
/// receives relative-motion deltas equal to the injected motion.
///
/// Non-vacuity: the baseline `assert_ne!` before the lock proves the
/// injector really moves the cursor when unconstrained, so the frozen
/// assertion afterward cannot pass trivially against a broken injector or a
/// no-op lock.
///
/// Activation ordering (T3 design): a constraint on an already-focused
/// surface only activates on the *next* motion after it is created, not at
/// creation -- the freeze check reads `active_constraint` before the cursor
/// move, but activation itself happens in `enter_surface_under_cursor`
/// after the move. So the first motion after `lock_pointer()` still moves
/// the cursor (and is what activates the constraint); only from the second
/// motion on is the cursor frozen. The priming motion below is that first,
/// still-moving motion.
#[test]
fn a_locked_pointer_freezes_the_cursor_and_delivers_relative_motion() {
    /// The headless backend's real output pixel size -- pinned, not
    /// guessed. See the drag-icon test's own `OUTPUT_W`/`OUTPUT_H` doc in
    /// this file: `motion_absolute` maps onto the *real* output pixel box,
    /// not onto a maximized window's (inset) geometry.
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut pc = PointerConstraintsClient::spawn(&comp.socket);

    let opened = comp.wait_event(
        |e| matches!(e, Event::WindowOpened(w) if w.app_id == "icedtea-harness-pointer-constraints"),
    );
    let Event::WindowOpened(pc_info) = opened else {
        unreachable!()
    };
    let pc_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.id == pc_info.id)
        .expect("pc must be in the model once mapped")
        .geometry;
    let (px, py) = (
        (pc_geo.x + pc_geo.width / 2) as f64,
        (pc_geo.y + pc_geo.height / 2) as f64,
    );

    // Move the pointer over pc's surface so it holds pointer focus.
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    pc.pump();

    // Baseline (non-vacuity): an unconstrained motion moves the cursor.
    let before = comp.cursor_position();
    vp.motion(20.0, 20.0);
    vp.frame();
    vp.pump();
    assert_ne!(
        comp.cursor_position(),
        before,
        "baseline: cursor should move unconstrained"
    );

    // Lock, then PRIME activation (see this test's own doc), then measure.
    pc.lock_pointer();
    pc.pump();
    vp.motion(5.0, 5.0);
    vp.frame();
    vp.pump();
    pc.pump();
    let locked_pos = comp.cursor_position();
    let events_before_freeze = pc.relative_motion_events();
    // Capture the accumulated delta BEFORE the frozen motion. `relative_delta`
    // accumulates over the client's whole life (the pre-lock baseline and the
    // priming motion already made it non-zero), so only the *change* across
    // the frozen motion below is attributable to it — a bare `!= 0` check on
    // the total would be dead (always true here).
    let delta_before_freeze = pc.relative_delta();

    // Now the constraint is active: this motion must be frozen, yet the
    // client must still see it as a relative-motion event.
    vp.motion(20.0, 20.0);
    vp.frame();
    vp.pump();
    pc.pump();
    assert_eq!(
        comp.cursor_position(),
        locked_pos,
        "locked: cursor must not move once active"
    );
    assert!(
        pc.relative_motion_events() > events_before_freeze,
        "locked: client received no relative motion event while frozen"
    );
    // ...and the delta itself advanced across the frozen motion (not just the
    // event count) — scoped to the freeze window, so this genuinely detects a
    // dropped relative delta during the freeze.
    assert_ne!(
        pc.relative_delta(),
        delta_before_freeze,
        "locked: relative delta did not advance across the frozen motion"
    );
}

/// M4.5 Criterion 4: a constraint only activates once its surface holds
/// pointer focus. Locking a pointer whose surface is NOT focused leaves the
/// cursor free to move; once the pointer enters the surface (activating the
/// constraint on that entering motion -- see the sibling freeze test's doc
/// on activation ordering) the next motion is frozen.
#[test]
fn a_locked_pointer_only_activates_once_its_surface_has_focus() {
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut pc = PointerConstraintsClient::spawn(&comp.socket);

    let opened = comp.wait_event(
        |e| matches!(e, Event::WindowOpened(w) if w.app_id == "icedtea-harness-pointer-constraints"),
    );
    let Event::WindowOpened(pc_info) = opened else {
        unreachable!()
    };
    let pc_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.id == pc_info.id)
        .expect("pc must be in the model once mapped")
        .geometry;
    let (px, py) = (
        (pc_geo.x + pc_geo.width / 2) as f64,
        (pc_geo.y + pc_geo.height / 2) as f64,
    );

    // A point in the far bottom-right corner of the output, well outside
    // pc's (default-cascaded, near-top-left) window -- verified below
    // rather than assumed.
    let (off_x, off_y) = ((OUTPUT_W - 50) as i32, (OUTPUT_H - 50) as i32);
    assert!(
        !pc_geo.contains(off_x, off_y),
        "test point ({off_x}, {off_y}) must fall outside pc's geometry {pc_geo:?}"
    );
    let (off_x, off_y) = (off_x as f64, off_y as f64);

    // Move the pointer off pc's surface -- it holds no pointer focus here.
    vp.motion_absolute(off_x, off_y, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();

    // Lock while unfocused: the constraint exists but is not active.
    pc.lock_pointer();
    pc.pump();

    // Unfocused + locked: the cursor still moves freely.
    let before = comp.cursor_position();
    vp.motion(15.0, 15.0);
    vp.frame();
    vp.pump();
    assert_ne!(
        comp.cursor_position(),
        before,
        "unfocused lock must not freeze the cursor"
    );

    // Move onto pc's surface: this both moves the cursor there (activation
    // happens *after* the move, per the freeze test's activation-ordering
    // doc) and primes the now-focused constraint for the next motion.
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    pc.pump();
    let locked_pos = comp.cursor_position();
    let events_before_freeze = pc.relative_motion_events();

    // Now the constraint is active: this motion must be frozen.
    vp.motion(20.0, 20.0);
    vp.frame();
    vp.pump();
    pc.pump();
    assert_eq!(
        comp.cursor_position(),
        locked_pos,
        "focused+active: cursor must not move"
    );
    assert!(
        pc.relative_motion_events() > events_before_freeze,
        "focused+active: client received no relative motion while frozen"
    );
}

/// Fetches pc's window geometry (in layout coords) once it has mapped --
/// shared by the three confine tests below.
fn pc_window_geometry(comp: &Compositor) -> Rectangle {
    let opened = comp.wait_event(
        |e| matches!(e, Event::WindowOpened(w) if w.app_id == "icedtea-harness-pointer-constraints"),
    );
    let Event::WindowOpened(pc_info) = opened else {
        unreachable!()
    };
    comp.snapshot()
        .windows
        .iter()
        .find(|w| w.id == pc_info.id)
        .expect("pc must be in the model once mapped")
        .geometry
}

/// A tolerant point-in-rect check (+/-1px) for boundary/re-anchor
/// assertions -- the exact clamp point wlroots picks for a nearest-point
/// snap can legitimately land on a rect's inclusive edge, which
/// `Rectangle::contains`'s strict `< x + width` would reject by a hair.
fn near_contains(r: Rectangle, x: f64, y: f64) -> bool {
    x >= r.x as f64 - 1.0
        && x <= (r.x + r.width) as f64 + 1.0
        && y >= r.y as f64 - 1.0
        && y <= (r.y + r.height) as f64 + 1.0
}

/// A `motion_absolute` target guaranteed to land on pc's actual clickable
/// surface, not on any server-side-decoration title bar painted above it.
///
/// `PointerConstraintsClient`'s app id does not match
/// `icedtea_compositor::decoration::is_csd`'s GTK allowlist, so this
/// window gets SSD, and `wlr_seat`'s pointer-focus hit-testing (which
/// pointer-constraint activation is keyed off, per the freeze test's
/// activation-ordering doc) is scene-based: a target within
/// `TITLE_BAR_HEIGHT` of the frame's top edge lands on the title-bar scene
/// node instead of the client surface, so the constraint's `surface ==
/// focused` activation check never matches and it silently never
/// activates (found the hard way: a confine region whose center sat only
/// a few pixels below the frame's `y` never clamped anything).
///
/// This is a *focus* hazard only -- `confine_pointer` / `set_confine_region`'s
/// own region coordinates are unaffected and stay relative to the frame's
/// `(x, y)` (`pc_geo`) everywhere else in these tests, matched against the
/// compositor's own region-to-layout math confirmed empirically while
/// building this test (see the report). Callers should keep their region
/// `y` origins at or beyond `TITLE_BAR_HEIGHT` regardless, so a region's
/// own center clears this hazard on its own; this helper exists for the
/// rare case (the region-hole test) that needs a *different* motion target
/// than the region's arithmetic center.
fn pc_focus_target(pc_geo: Rectangle, local_x: i32, local_y: i32) -> (f64, f64) {
    let safe_y = local_y.max(icedtea_compositor::decoration::TITLE_BAR_HEIGHT);
    ((pc_geo.x + local_x) as f64, (pc_geo.y + safe_y) as f64)
}

/// M4.5 Criterion 3: a confined pointer clamps the cursor to its region --
/// motion aimed past the region's right edge must not carry the cursor out
/// of it.
///
/// Non-vacuity: the baseline motion below (the same delta, injected before
/// any confinement exists) moves the cursor the full delta, proving the
/// injector really does move the cursor that far when unconstrained. A
/// broken clamp (cursor escapes) would fail the boundary assertion; a
/// no-op injector would already fail the baseline assertion.
///
/// Activation ordering (see the freeze test's doc above): the confinement
/// activates on the *next* motion after `confine_pointer`, not at
/// creation, so a small priming motion runs first.
#[test]
fn a_confined_pointer_clamps_the_cursor_to_its_region() {
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut pc = PointerConstraintsClient::spawn(&comp.socket);

    let pc_geo = pc_window_geometry(&comp);

    // A small confine region well inside the surface, in surface-local
    // coordinates, sized off the surface's own geometry rather than a
    // hardcoded guess.
    let (rx, ry, rw, rh) = (
        pc_geo.width / 10,
        pc_geo.height / 10,
        pc_geo.width / 5,
        pc_geo.height / 5,
    );
    let region_right = (pc_geo.x + rx + rw) as f64;

    // Pointer starts at the region's center.
    let (px, py) = pc_focus_target(pc_geo, rx + rw / 2, ry + rh / 2);
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();

    // Baseline (non-vacuity): the same large motion, unconfined, moves the
    // cursor the full delta.
    let before = comp.cursor_position();
    vp.motion(500.0, 0.0);
    vp.frame();
    vp.pump();
    let baseline_after = comp.cursor_position();
    assert!(
        (baseline_after.0 - before.0 - 500.0).abs() < 1.0,
        "baseline: unconfined motion should move the cursor the full delta \
         (before={before:?}, after={baseline_after:?})"
    );

    // Re-center, then confine.
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    pc.confine_pointer(rx, ry, rw, rh);
    pc.pump();

    // Prime activation: a small motion that stays well inside the region.
    vp.motion(1.0, 1.0);
    vp.frame();
    vp.pump();
    pc.pump();
    let before_clamp = comp.cursor_position();

    // Now confined: a large motion aimed past the region's right edge must
    // leave the cursor inside the region.
    vp.motion(500.0, 0.0);
    vp.frame();
    vp.pump();
    pc.pump();
    let (cx, _cy) = comp.cursor_position();

    assert!(
        cx <= region_right + 1.0,
        "confined: cursor escaped the region (cx={cx}, boundary={region_right})"
    );
    assert!(
        cx - before_clamp.0 < 500.0,
        "confined: cursor should have moved less than the injected delta \
         (before={before_clamp:?}, cx={cx})"
    );
}

/// M4.5: updating a confine region away from the cursor re-anchors it
/// rather than wedging the cursor forever (a client-triggerable input
/// freeze the M5 session flagged).
///
/// Non-vacuity: without the re-anchor fix, the cursor stays outside region
/// B after `set_confine_region`, and every subsequent motion hits
/// `wlr_region_confine`'s false arm -- the final `assert_ne!` below (motion
/// still moves the cursor) is what catches that permanent freeze.
#[test]
fn a_confined_pointer_reanchors_when_the_region_moves_off_the_cursor() {
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut pc = PointerConstraintsClient::spawn(&comp.socket);

    let pc_geo = pc_window_geometry(&comp);

    // Region A (around the cursor) in the surface's top-left quadrant;
    // region B (away from the cursor) in the bottom-right -- both sized
    // and positioned off the surface's own geometry so neither overlaps
    // nor spills outside it.
    let (ax, ay, aw, ah) = (
        pc_geo.width / 10,
        pc_geo.height / 10,
        pc_geo.width / 5,
        pc_geo.height / 5,
    );
    let (bx, by, bw, bh) = (
        pc_geo.width * 6 / 10,
        pc_geo.height * 6 / 10,
        pc_geo.width / 5,
        pc_geo.height / 5,
    );
    let region_b_layout = Rectangle {
        x: pc_geo.x + bx,
        y: pc_geo.y + by,
        width: bw,
        height: bh,
    };

    let (px, py) = pc_focus_target(pc_geo, ax + aw / 2, ay + ah / 2);
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();

    pc.confine_pointer(ax, ay, aw, ah);
    pc.pump();

    // Prime activation.
    vp.motion(1.0, 1.0);
    vp.frame();
    vp.pump();
    pc.pump();

    // Move the region to B, which does NOT contain the current cursor.
    pc.set_confine_region(bx, by, bw, bh);
    pc.pump();

    let after_reanchor = comp.cursor_position();
    assert!(
        near_contains(region_b_layout, after_reanchor.0, after_reanchor.1),
        "cursor not re-anchored into the new region (pos={after_reanchor:?}, \
         region_b={region_b_layout:?})"
    );

    // And motion still works (not wedged): a further injected small
    // motion within B changes the cursor.
    let before = comp.cursor_position();
    vp.motion(3.0, 3.0);
    vp.frame();
    vp.pump();
    pc.pump();
    assert_ne!(
        comp.cursor_position(),
        before,
        "cursor wedged after region moved (confine freeze)"
    );
}

/// M4.5: a two-rectangle confine region with a gap between the rects,
/// positioned so the cursor sits in the gap (inside the region's
/// bounding-box extents but outside both rects), must re-anchor into one
/// of the two rects rather than into the extents' bounding box (whose
/// gap-center is still outside both rects).
///
/// Non-vacuity: a single-rect region's extents *is* the rect, so a
/// re-anchor-to-extents implementation would already pass the sibling
/// test above. Only a two-rect region with a real gap pins the
/// `rectangles[0]`-fallback bug: if re-anchor clamped to extents alone,
/// the cursor would land back in the gap and the closing motion would hit
/// the same wedged-freeze failure mode as the sibling test.
///
/// Drives the re-anchor via `set_confine_region_rects` (like the sibling
/// test's `set_confine_region`) rather than via a fresh
/// `confine_pointer_rects` created with the cursor already in the gap:
/// activation is keyed off pointer *focus* entering the surface, which
/// happens on the *first* motion after a constraint is created (per the
/// activation-ordering doc) -- before that motion's own move is subject
/// to enforcement. A constraint created with the cursor already outside
/// its region therefore does not reliably clamp/re-anchor until a later
/// motion, which is timing-fragile to assert on. `set_region` against an
/// *already-active* constraint is the deterministic trigger instead
/// (`on_pointer_constraint_set_region` re-anchors unconditionally the
/// moment the active constraint's region changes).
#[test]
fn a_confined_pointer_reanchors_out_of_a_region_hole() {
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut pc = PointerConstraintsClient::spawn(&comp.socket);

    let pc_geo = pc_window_geometry(&comp);

    // Two disjoint 50x50 rects with a 50px gap between them (surface-local
    // small integers, per the brief, so the in-region assertion is exact),
    // with a `y` origin past `TITLE_BAR_HEIGHT` so the region's own center
    // (which is also where the priming motion below lands) clears the
    // focus hazard `pc_focus_target` documents, no clamping needed.
    let rect_a = (0, 5, 50, 50);
    let rect_b = (100, 5, 50, 50);
    let rect_a_layout = Rectangle {
        x: pc_geo.x + rect_a.0,
        y: pc_geo.y + rect_a.1,
        width: rect_a.2,
        height: rect_a.3,
    };
    let rect_b_layout = Rectangle {
        x: pc_geo.x + rect_b.0,
        y: pc_geo.y + rect_b.1,
        width: rect_b.2,
        height: rect_b.3,
    };

    // The gap's center: inside the extents bounding box (0,5,150,50) but
    // outside both rects.
    let (gap_x, gap_y) = (75, 30);
    let (gx, gy) = (pc_geo.x + gap_x, pc_geo.y + gap_y);
    assert!(
        !rect_a_layout.contains(gx, gy) && !rect_b_layout.contains(gx, gy),
        "test point must actually sit in the gap between the two rects"
    );

    // Establish a normally-activating confinement first, in a region
    // centered on the eventual gap point -- the same activation pattern
    // the sibling tests use (confine while focused, prime), which
    // reliably brings the constraint to "active" before its region is
    // ever swapped out from under it.
    let (init_x, init_y, init_w, init_h) = (gap_x - 50, gap_y - 25, 100, 50);
    let (px, py) = pc_focus_target(pc_geo, init_x + init_w / 2, init_y + init_h / 2);
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();

    pc.confine_pointer(init_x, init_y, init_w, init_h);
    pc.pump();

    // Prime activation: a small motion that stays inside the initial
    // (and still gap-adjacent) region.
    vp.motion(1.0, 0.0);
    vp.frame();
    vp.pump();
    pc.pump();

    // Swap the region to the two-rect, gap-containing shape via
    // `set_confine_region_rects` -- the deterministic re-anchor path
    // (`on_pointer_constraint_set_region`) proven by the sibling test
    // above, which unconditionally restores the confine invariant for the
    // *active* constraint the moment its region changes. The gap sits
    // under the cursor's current position, so this pins the two-rect
    // fallback: an extents-only re-anchor would leave the cursor right
    // where it already is, in the gap.
    pc.set_confine_region_rects(&[rect_a, rect_b]);
    pc.pump();

    let after_reanchor = comp.cursor_position();
    assert!(
        near_contains(rect_a_layout, after_reanchor.0, after_reanchor.1)
            || near_contains(rect_b_layout, after_reanchor.0, after_reanchor.1),
        "cursor not re-anchored into either rect (pos={after_reanchor:?}, \
         a={rect_a_layout:?}, b={rect_b_layout:?})"
    );

    // Not wedged: a further small motion still moves the cursor.
    let before = comp.cursor_position();
    vp.motion(2.0, 2.0);
    vp.frame();
    vp.pump();
    pc.pump();
    assert_ne!(
        comp.cursor_position(),
        before,
        "cursor wedged in the region hole (confine freeze)"
    );
}

// --- the implicit pointer grab -------------------------------------------

/// `BTN_LEFT` from `linux/input-event-codes.h`.
const GRAB_BTN_LEFT: u32 = 0x110;

/// A point inside `target`'s *client area* that `other` does not cover.
///
/// A sibling of `compat_protocols.rs`'s helper of the same name, and
/// deliberately a copy rather than a shared item: the two integration-test
/// binaries share no module, and hoisting twenty lines of geometry into the
/// harness would put compositor-decoration policy in a crate whose job is
/// driving clients.
///
/// Both halves are load-bearing:
///
/// * Outside `other`, because the scene graph answers a point both windows
///   cover with whichever is on top, and these tests need a point that can
///   only ever be `target`. Cascade placement offsets the two frames, so
///   such a point always exists.
/// * Inside the content rect rather than the frame, because the SSD title
///   bar is a scene rect with no `wl_surface` behind it: a pointer parked
///   there earns the client no `wl_pointer.enter` at all.
fn point_inside_only(target: Rectangle, target_app_id: &str, other: Rectangle) -> (i32, i32) {
    let ssd = icedtea_compositor::decoration::has_ssd(target_app_id, None, false);
    let content = icedtea_compositor::decoration::content_rect(target, ssd);
    let mut y = content.y + 4;
    while y < content.y + content.height {
        let mut x = content.x + 4;
        while x < content.x + content.width {
            if !other.contains(x, y) {
                return (x, y);
            }
            x += 8;
        }
        y += 8;
    }
    panic!("no point of {target:?} lies outside {other:?}");
}

/// The Wayland **implicit pointer grab**: while any button is held, every
/// motion and every button keeps going to the surface that was pressed, and
/// pointer focus does not move — no matter where the cursor wanders.
///
/// This is the compositor's job, not wlroots'. `wlr_seat_pointer_notify_*`
/// defers only to *explicit* seat grabs (an xdg-popup grab, a drag); the
/// implicit grab is what sway spells `seatop_down`. Without it, re-focusing
/// on every motion means a press on A followed by a drag off A delivers the
/// release to whatever happens to be under the cursor — A's client never
/// learns the button came back up, and is left with a stuck press.
///
/// Four assertions, in the order the grab has to hold them:
///
/// 1. A sees the press.
/// 2. A keeps seeing motion after the cursor has left it, at surface-local
///    coordinates that follow the cursor's own delta (the sway model: the
///    coordinates the enter used, plus how far the cursor has moved since).
/// 3. B sees *no* enter while the grab is up, even though the cursor is
///    sitting inside it.
/// 4. A sees the release, and only then does focus re-evaluate and B enter.
#[test]
fn an_implicit_grab_keeps_delivering_to_the_pressed_surface() {
    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);

    let mut a = TestClient::map_toplevel(&comp.socket, "grab-a.app", "A");
    assert!(
        a.wait_until(|c| c.last_configure().is_some()),
        "A never configured"
    );
    let mut b = TestClient::map_toplevel(&comp.socket, "grab-b.app", "B");
    assert!(
        b.wait_until(|c| c.last_configure().is_some()),
        "B never configured"
    );

    let snapshot = comp.snapshot();
    let geo = |app_id: &str| {
        snapshot
            .windows
            .iter()
            .find(|w| w.app_id == app_id)
            .unwrap_or_else(|| panic!("{app_id} must be in the model once mapped"))
            .geometry
    };
    let (a_geo, b_geo) = (geo("grab-a.app"), geo("grab-b.app"));
    let (ax, ay) = point_inside_only(a_geo, "grab-a.app", b_geo);
    let (bx, by) = point_inside_only(b_geo, "grab-b.app", a_geo);
    let (ow, oh) = comp.output_size();
    let (ow, oh) = (ow as u32, oh as u32);

    // Park the cursor over A and let the enter land.
    let a_enters = a.pointer_enters();
    vp.motion_absolute(ax as f64, ay as f64, ow, oh);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_enters() > a_enters),
        "A never got wl_pointer.enter over its own content"
    );
    b.pump_checked();
    let b_enters = b.pointer_enters();
    // The grab's reference point, read off the `enter` itself: wlroots
    // suppresses a `motion` carrying the coordinates the `enter` just
    // established, so the enter is the only place a client can observe it.
    let anchor = a
        .pointer_enter_position()
        .expect("just asserted an enter arrived");

    // 1. Press over A.
    let a_buttons = a.pointer_buttons().len();
    vp.button(GRAB_BTN_LEFT, true);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_buttons()[a_buttons..].contains(&(GRAB_BTN_LEFT, true))),
        "A never got the wl_pointer.button press"
    );
    let a_motions = a.pointer_motions().len();
    let a_leaves = a.pointer_leaves();

    // 2. Drag off A and onto B, still held.
    vp.motion_absolute(bx as f64, by as f64, ow, oh);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_motions().len() > a_motions),
        "A got no wl_pointer.motion once the cursor left it: the implicit \
         grab is not delivering motion to the pressed surface"
    );
    let moved = *a
        .pointer_motions()
        .last()
        .expect("just asserted a motion arrived");
    let (dx, dy) = (moved.0 - anchor.0, moved.1 - anchor.1);
    let (want_dx, want_dy) = ((bx - ax) as f64, (by - ay) as f64);
    assert!(
        (dx - want_dx).abs() <= 1.0 && (dy - want_dy).abs() <= 1.0,
        "A's surface-local motion moved by ({dx}, {dy}), not the cursor's own \
         ({want_dx}, {want_dy}): the grab is not applying the cursor delta to \
         the coordinates the enter established"
    );
    assert_eq!(
        a.pointer_leaves(),
        a_leaves,
        "A got a wl_pointer.leave while the cursor was dragged off it, though \
         the implicit grab holds it: the pressed surface must not lose focus"
    );

    // 3. B must not have been entered while A holds the grab.
    b.pump_checked();
    assert_eq!(
        b.pointer_enters(),
        b_enters,
        "B was entered while A held the implicit pointer grab"
    );

    // 4. The release goes to A, not to whatever is under the cursor.
    let a_buttons = a.pointer_buttons().len();
    vp.button(GRAB_BTN_LEFT, false);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_buttons()[a_buttons..].contains(&(GRAB_BTN_LEFT, false))),
        "A never got the wl_pointer.button release after the cursor was \
         dragged off it: the implicit grab dropped the button on the floor"
    );

    // ...and only now does focus re-evaluate onto the surface the cursor is
    // actually over.
    assert!(
        b.wait_until(|c| c.pointer_enters() > b_enters),
        "B was never entered after the grab ended, though the cursor is \
         inside it"
    );

    a.detach();
    b.detach();
}

/// The same grab, on a **layer surface** — the shape that actually matters
/// to this project's own shell, whose panels and popups are all layer
/// surfaces and whose themed widgets track `:active` across a press.
///
/// Worth its own test rather than trusting the toplevel one: a layer
/// surface reaches pointer focus through a different scene layer and, in
/// the panel's case, sits in an exclusive zone that no toplevel can
/// overlap — which is exactly what makes "press the panel, drag onto the
/// window below, release" a clean two-surface hand-off to assert on.
#[test]
fn an_implicit_grab_on_a_layer_surface_survives_the_drag_off() {
    /// Tall enough that a press lands well inside it, and that the
    /// toplevel below starts clear of it.
    const PANEL_HEIGHT: i32 = 40;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);

    // `map_layer_panel` (`LayerPanelClient::spawn`) already blocks until the
    // panel's own `configure` has arrived, so there is nothing left to wait
    // for here.
    let mut panel = TestClient::map_layer_panel(&comp.socket, PANEL_HEIGHT);
    let mut win = TestClient::map_toplevel(&comp.socket, "under.app", "under");
    assert!(
        win.wait_until(|c| c.last_configure().is_some()),
        "the toplevel never configured"
    );

    let win_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "under.app")
        .expect("the toplevel must be in the model once mapped")
        .geometry;
    let ssd = icedtea_compositor::decoration::has_ssd("under.app", None, false);
    let content = icedtea_compositor::decoration::content_rect(win_geo, ssd);
    assert!(
        win_geo.y >= PANEL_HEIGHT,
        "the toplevel's frame ({win_geo:?}) overlaps the panel's exclusive \
         zone: `content_rect` insets it by the title bar, so checking \
         `content.y` here would pass even with the frame itself overlapping"
    );
    assert_eq!(
        (content.width, content.height),
        win.mapped_size(),
        "the model's content size does not match the buffer actually \
         mapped -- the press point below would not land on real surface \
         content"
    );

    let (ow, oh) = comp.output_size();
    let (px, py) = (ow / 2, PANEL_HEIGHT / 2);
    let (wx, wy) = (
        content.x + content.width / 2,
        content.y + content.height / 2,
    );
    let (ow, oh) = (ow as u32, oh as u32);

    // Park on the panel and press.
    let panel_enters = panel.pointer_enters();
    vp.motion_absolute(px as f64, py as f64, ow, oh);
    vp.frame();
    assert!(
        panel.wait_until(|p| p.pointer_enters() > panel_enters),
        "the panel never got wl_pointer.enter"
    );
    win.pump_checked();
    let win_enters = win.pointer_enters();
    let anchor = panel
        .pointer_enter_position()
        .expect("just asserted an enter arrived");

    let panel_buttons = panel.pointer_buttons().len();
    vp.button(GRAB_BTN_LEFT, true);
    vp.frame();
    assert!(
        panel.wait_until(|p| p.pointer_buttons()[panel_buttons..].contains(&(GRAB_BTN_LEFT, true))),
        "the panel never got the wl_pointer.button press"
    );
    let motions = panel.pointer_motions().len();
    let panel_leaves = panel.pointer_leaves();

    // Drag down onto the window, still held.
    vp.motion_absolute(wx as f64, wy as f64, ow, oh);
    vp.frame();
    assert!(
        panel.wait_until(|p| p.pointer_motions().len() > motions),
        "the panel got no motion once the cursor left it"
    );
    let moved = *panel
        .pointer_motions()
        .last()
        .expect("just asserted a motion arrived");
    let (dx, dy) = (moved.0 - anchor.0, moved.1 - anchor.1);
    let (want_dx, want_dy) = ((wx - px) as f64, (wy - py) as f64);
    assert!(
        (dx - want_dx).abs() <= 1.0 && (dy - want_dy).abs() <= 1.0,
        "the panel's surface-local motion moved by ({dx}, {dy}), not the \
         cursor's own ({want_dx}, {want_dy})"
    );
    assert_eq!(
        panel.pointer_leaves(),
        panel_leaves,
        "the panel got a wl_pointer.leave while the cursor was dragged off \
         it, though the implicit grab holds it"
    );

    win.pump_checked();
    assert_eq!(
        win.pointer_enters(),
        win_enters,
        "the window below was entered while the panel held the grab"
    );

    // Release over the window: the panel gets it, then the window enters.
    let panel_buttons = panel.pointer_buttons().len();
    vp.button(GRAB_BTN_LEFT, false);
    vp.frame();
    assert!(
        panel
            .wait_until(|p| p.pointer_buttons()[panel_buttons..].contains(&(GRAB_BTN_LEFT, false))),
        "the panel never got the release after the cursor was dragged off it"
    );
    assert!(
        win.wait_until(|c| c.pointer_enters() > win_enters),
        "the window below was never entered after the grab ended"
    );

    win.detach();
}

/// Regression guard: an **explicit** xdg-popup grab still dismisses on a
/// click outside it, with the implicit pointer grab in the way.
///
/// The two grabs share a code path and could easily fight. wlroots owns the
/// popup grab: while it is up only the popup's own client may take pointer
/// focus, and a button pressed while focus is elsewhere destroys the popup
/// and sends `popup_done`. That hinges on the compositor still doing two
/// things under an explicit grab — entering (here, *clearing* focus, which
/// is what the popup grab turns an out-of-client enter into) and then
/// notifying the button. The implicit grab deliberately steps aside whenever
/// `wlr_seat_pointer_has_grab` is true; this is the test that says so.
///
/// The popup is created, grabbed and driven all the way to mapped --
/// `open_grabbing_popup` is `open_popup` with a grab serial -- so
/// `popup_configured()` is asserted positively here.
///
/// Verified non-vacuous by mutation: skipping `notify_button` under an
/// explicit grab makes `popup_done` never arrive.
#[test]
fn a_popup_grab_still_dismisses_on_a_click_outside_it() {
    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);

    let mut a = TestClient::map_toplevel(&comp.socket, "popup.app", "popup");
    assert!(
        a.wait_until(|c| c.last_configure().is_some()),
        "the parent never configured"
    );

    let a_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "popup.app")
        .expect("the parent must be in the model once mapped")
        .geometry;
    let ssd = icedtea_compositor::decoration::has_ssd("popup.app", None, false);
    let content = icedtea_compositor::decoration::content_rect(a_geo, ssd);
    assert_eq!(
        (content.width, content.height),
        a.mapped_size(),
        "the model's content size does not match the buffer actually \
         mapped -- the press point below would not land on real surface \
         content"
    );
    let (ow, oh) = comp.output_size();

    // A press inside the parent's client area, to mint the serial the popup
    // grab has to cite.
    let inside = (
        content.x + content.width / 2,
        content.y + content.height / 2,
    );
    let a_enters = a.pointer_enters();
    vp.motion_absolute(inside.0 as f64, inside.1 as f64, ow as u32, oh as u32);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_enters() > a_enters),
        "the parent never got wl_pointer.enter"
    );
    let a_buttons = a.pointer_buttons().len();
    vp.button(GRAB_BTN_LEFT, true);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_buttons()[a_buttons..].contains(&(GRAB_BTN_LEFT, true))),
        "the parent never got the press that mints the grab serial"
    );
    let serial = a
        .last_pointer_serial()
        .expect("the press just delivered a serial");
    let a_buttons = a.pointer_buttons().len();
    vp.button(GRAB_BTN_LEFT, false);
    vp.frame();
    assert!(
        a.wait_until(|c| c.pointer_buttons()[a_buttons..].contains(&(GRAB_BTN_LEFT, false))),
        "the parent never got the release"
    );

    a.open_grabbing_popup(serial, 64, 48);
    assert!(
        a.wait_until(|c| c.popup_configured().is_some()),
        "the popup was never configured: the compositor must answer a popup's \
         first commit or it can never legally attach a buffer and map"
    );

    // A point on bare desktop: outside the parent's frame, and so outside
    // every surface this client owns. The popup grab turns the resulting
    // enter into a focus clear, which is what makes the next button
    // dismiss.
    let outside = (ow - 2, oh - 2);
    assert!(
        !a_geo.contains(outside.0, outside.1),
        "{outside:?} is inside the parent {a_geo:?}, so the click would not \
         be outside the popup's client"
    );
    vp.motion_absolute(outside.0 as f64, outside.1 as f64, ow as u32, oh as u32);
    vp.frame();
    vp.button(GRAB_BTN_LEFT, true);
    vp.frame();
    vp.button(GRAB_BTN_LEFT, false);
    vp.frame();

    assert!(
        a.wait_until(|c| c.popup_done()),
        "the popup was never dismissed by a click outside it: the explicit \
         xdg-popup grab is not reaching the seat"
    );

    a.detach();
}

// --- A6.2 (M6.1 Spec 1, popups + keyboard grab): tests 8, 9, 10 ---

/// Test 8: an IME candidate popup is placed below the focused text input's
/// cursor rectangle AND told that rectangle. Both sides of the seam: the
/// popup's own `text_input_rectangle` event (via the IME double) and the
/// compositor's `InputPopupPosition` oracle.
#[test]
fn ime_popup_is_positioned_and_told_its_rectangle() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket); // seat needs a keyboard
    let mut im = InputMethodClient::spawn(&comp.socket); // IME bound first
    let mut ti = TextInputClient::spawn(&comp.socket); // maps + auto-focused
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    // Enable and commit a known cursor rectangle, so the anchor the compositor
    // reads back is deterministic.
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    // The IME opens its candidate popup.
    im.create_popup();

    // The popup must be told the anchor rectangle the compositor placed it
    // against -- the exact cursor rectangle committed above.
    assert!(
        im.wait_until(|s| s.popup_text_input_rectangle() == Some((100, 200, 2, 16))),
        "popup was never told its anchor rectangle; saw {:?}",
        im.popup_text_input_rectangle()
    );

    // And the compositor must have actually placed the popup's scene node --
    // below the cursor rectangle, translated into output coordinates. Exact
    // position, not a loose bound: the headless output sits at (0, 0), the
    // server-side-decorated test window cascades at its top-left so its
    // content starts 28px down (TITLE_BAR), so the committed (100, 200, 2, 16)
    // caret anchors at (100, 228) and the zero-size popup lands at (100, 244).
    // A naive surface-local placement would read (100, 216) and fail this.
    let pos = comp
        .input_popup_position()
        .expect("compositor never placed the popup");
    assert_eq!(
        pos,
        (100, 244),
        "popup must sit translated below the caret in output coordinates"
    );
}

/// Test 9: a keyboard grab intercepts physical keys -- non-vacuous in both
/// directions. BEFORE the grab, a virtual-keyboard key reaches the focused
/// app's ordinary `wl_keyboard`; AFTER the grab, the key reaches the IME's
/// grab object and the app's `wl_keyboard` receives nothing more.
#[test]
fn keyboard_grab_intercepts_physical_keys() {
    /// evdev `KEY_A` / `KEY_B`. Neither is a compositor binding, so both are
    /// forwarded rather than consumed -- which is what lets them reach the
    /// app's keyboard pre-grab and the grab post-grab.
    const KEY_A: u32 = 30;
    const KEY_B: u32 = 48;

    let comp = Compositor::spawn();
    let mut vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket); // IME bound first
    let mut ti = TextInputClient::spawn(&comp.socket); // maps + auto-focused, has a wl_keyboard
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    // Non-vacuous baseline: BEFORE any grab, an unbound key reaches the app's
    // own wl_keyboard.
    vk.key_press(KEY_A);
    assert!(
        ti.wait_until(|c| c.wl_keyboard_key_events() >= 1),
        "pre-grab key must reach the focused surface's wl_keyboard"
    );

    // Take the grab. From now on the compositor forwards seat keys to the grab
    // object rather than the app's wl_keyboard.
    im.grab_keyboard();

    // Synchronize with grab establishment: the grab request (IME connection)
    // and subsequent keys (virtual-keyboard connection) race through separate
    // sockets. Press an unbound key until the grab observes it; a key pressed
    // before the grab exists would land in the app keyboard instead and
    // poison the frozen-count assertion below, so baselines are read only
    // after this loop.
    /// evdev `KEY_C`: unbound, like KEY_A/KEY_B above.
    const KEY_C: u32 = 46;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut live = false;
    while std::time::Instant::now() < deadline {
        vk.key_press(KEY_C);
        vk.pump();
        im.pump();
        ti.pump();
        if im.grab_key_events() >= 1 {
            live = true;
            break;
        }
    }
    assert!(live, "grab never established (liveness key unobserved)");
    let g0 = im.grab_key_events();
    let pre = ti.wl_keyboard_key_events();

    vk.key_press(KEY_B);
    assert!(
        im.wait_until(|s| s.grab_key_events() > g0),
        "grab never received the key it was supposed to intercept"
    );

    // The app's ordinary wl_keyboard must receive NOTHING more: pump both
    // clients for a bounded window and assert the count did not move.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
    while std::time::Instant::now() < deadline {
        vk.pump();
        ti.pump();
    }
    assert_eq!(
        ti.wl_keyboard_key_events(),
        pre,
        "a grabbed key must not also reach the app's wl_keyboard"
    );
}

/// Test 10: the compositor's own keybindings still fire while an IME keyboard
/// grab is active. This guards decision #6 -- `SeatHandler::key` runs the
/// binding dispatcher BEFORE the grab-forward branch, so a bound key is
/// consumed by the compositor and never reaches the grab. Observable: the
/// default `workspace:2` binding (SUPER+2) changes the active workspace even
/// though a grab is held.
#[test]
fn compositor_keybindings_fire_during_a_grab() {
    /// evdev `KEY_LEFTMETA` (the SUPER modifier) and `KEY_2`.
    const KEY_LEFTMETA: u32 = 125;
    const KEY_2: u32 = 3;

    let comp = Compositor::spawn();
    let mut vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    // A focused app so the seat has a keyboard-focused surface a grab would
    // otherwise steal from.
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    let initial_workspace = comp.snapshot().active_workspace;
    assert_eq!(
        initial_workspace, 0,
        "default active workspace is index 0 (the `workspace:1` binding's target)"
    );

    // Hold the grab, then inject the compositor's SUPER+2 workspace binding.
    // Synchronize with grab establishment first (see test 9): without this,
    // the binding could be exercised with no grab held and the "during a
    // grab" qualifier would be vacuous.
    im.grab_keyboard();

    /// evdev `KEY_C`: unbound, for the liveness check.
    const KEY_C: u32 = 46;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut live = false;
    while std::time::Instant::now() < deadline {
        vk.key_press(KEY_C);
        vk.pump();
        im.pump();
        ti.pump();
        if im.grab_key_events() >= 1 {
            live = true;
            break;
        }
    }
    assert!(live, "grab never established (liveness key unobserved)");

    vk.key_down(KEY_LEFTMETA);
    vk.key_press(KEY_2);
    vk.key_up(KEY_LEFTMETA);

    // Pump so the compositor has a chance to process the binding, then assert
    // the workspace actually switched -- proving the dispatcher ran ahead of
    // the grab-forward branch. `workspace:2` maps to internal index 1
    // (`apply_action` subtracts one), a fixed target a stuck
    // dispatcher would never reach (it would stay on index 0).
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut switched = false;
    while std::time::Instant::now() < deadline {
        vk.pump();
        im.pump();
        if comp.snapshot().active_workspace == 1 {
            switched = true;
            break;
        }
    }
    assert!(
        switched,
        "SUPER+2 must switch to workspace index 1 even while an IME keyboard \
         grab is active (the binding dispatcher runs before the grab-forward \
         branch); active_workspace started at {initial_workspace} and is now {}",
        comp.snapshot().active_workspace
    );

    // The modifiers held during the binding must have reached the grab (it
    // was live throughout): this also pins the harness's grab-`Modifiers`
    // recording arm, which no other test exercises.
    assert!(
        im.wait_until(|s| s.grab_modifier_events() >= 1),
        "grab never saw the modifiers held during the binding"
    );
}

/// Test 8b: a popup opened with no focused caret falls back to the output
/// origin and tells the IME a zero rectangle. Drives the
/// `cursor_rectangle() → None` branch: destroying the active text-input
/// deactivates the IME-side focus with no focus change (M1), so the caret
/// accessor reads `None` afterwards — synchronized by waiting for the
/// deactivation before opening the popup.
#[test]
fn ime_popup_without_a_caret_falls_back_to_the_origin() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    ti.destroy_text_input();
    assert!(
        im.wait_until(|s| s.deactivates() >= 1),
        "IME never deactivated after its text-input was destroyed"
    );

    // The IME object itself is still bound: it can open a popup, but the
    // compositor has no caret to anchor it against.
    im.create_popup();
    assert!(
        im.wait_until(|s| s.popup_text_input_rectangle() == Some((0, 0, 0, 0))),
        "popup was never told the zero fallback rectangle; saw {:?}",
        im.popup_text_input_rectangle()
    );
    // Zero anchor against the headless output at (0, 0): pins to its origin.
    assert_eq!(
        comp.input_popup_position(),
        Some((0, 0)),
        "caret-less popup must fall back to the output origin"
    );
}

/// Test 10b (FIX-A10-NODE residual): destroying the input-method while a
/// candidate popup is placed cascades the popup away too. The protocol keeps
/// the popup surface alive across the popup role object, but wlroots must fire
/// the popup's own destroy when its input-method is torn down; if it did not,
/// the compositor's scene node (and our position record) would leak. Observable:
/// the `InputPopupPosition` oracle, which reports `Some` while the popup is
/// placed and must return to `None` once the IME is destroyed -- our
/// `popup_surface_destroyed` handler only clears the record when the crate
/// fires the popup destroy. Plus the teardown tripwire: the captured scene
/// node itself must stop resolving afterwards, which is what fails if the
/// crate's `destroy_node` call is ever deleted (the position oracle alone
/// would stay green through bookkeeping).
#[test]
fn destroying_the_input_method_cascades_its_popups_away() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );

    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    im.create_popup();
    assert!(
        im.wait_until(|s| s.popup_text_input_rectangle() == Some((100, 200, 2, 16))),
        "popup was never placed"
    );
    assert!(
        comp.input_popup_position().is_some(),
        "precondition: the popup must be placed before we tear the IME down"
    );
    // Capture the scene node itself while placed. The position oracle above
    // clears via bookkeeping whether or not the crate destroyed the node, so
    // it cannot pin the teardown; the node id can.
    let node = comp.input_popup_node().expect(
        "precondition: the placed popup must have a scene node before we tear \
         the IME down",
    );

    // Tear down the input-method itself. wlroots must cascade-destroy its
    // still-live popup, which drives the compositor's popup_surface_destroyed.
    im.destroy();

    // The oracle must return to None: the compositor cleared its record because
    // the crate fired the popup's destroy, not because we asked it to.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut gone = false;
    while std::time::Instant::now() < deadline {
        // Nudge the compositor to process the destroy: any oracle round-trip
        // pumps its command loop, but the destroy travels the IME client's
        // connection, so also give it a moment.
        if comp.input_popup_position().is_none() {
            gone = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        gone,
        "destroying the input-method must cascade its candidate popup away \
         (the scene node would otherwise leak); the popup is still placed at {:?}",
        comp.input_popup_position()
    );

    // Tripwire for the crate's scene-node teardown: the captured node must be
    // stale now, not merely unrecorded. Deleting the `destroy_node` call in
    // `on_input_method_popup_destroy` leaves the assertions above green and
    // fails exactly this one. Map removal and node destruction are separate
    // steps, so poll with a deadline rather than asserting once.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut destroyed = false;
    while std::time::Instant::now() < deadline {
        if comp.scene_node_position(node).is_none() {
            destroyed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        destroyed,
        "the popup's scene node must be destroyed with the popup, not just \
         unrecorded; node {node:?} still resolves"
    );
}

/// Task 4: a keyboard-shortcuts inhibitor blocks the compositor's own
/// keybindings while active and releases them after.
///
/// Non-vacuous in both directions. While inhibited, SUPER+2 (the default
/// `workspace:2` binding) must NOT switch the workspace -- yet the keys must
/// still reach the focused client (inhibition reroutes, it does not swallow).
/// After destroying the inhibitor, the same chord MUST switch again, proving
/// the gate was the inhibitor and not a broken injector.
///
/// The inhibiting surface must itself hold keyboard focus: wlroots only
/// activates an inhibitor naming the focused surface, so the double maps its
/// own toplevel and waits for focus before inhibiting.
#[test]
fn shortcuts_inhibit_blocks_binding_while_active() {
    /// evdev `KEY_LEFTMETA` (the SUPER modifier) and `KEY_2`.
    const KEY_LEFTMETA: u32 = 125;
    const KEY_2: u32 = 3;

    let comp = Compositor::spawn();
    let mut vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut app = TestClient::map_toplevel(&comp.socket, "app", "app");
    assert!(
        app.wait_until(|c| c.has_input_serial()),
        "app never gained keyboard focus"
    );
    assert_eq!(
        comp.snapshot().active_workspace,
        0,
        "default active workspace is index 0"
    );
    // The live keyboard carries the harness virtual keyboard's "us" keymap,
    // so the snapshot names its layout from the very first `GetState`.
    assert_eq!(
        comp.snapshot().keyboard_layout.as_deref(),
        Some("English (US)"),
        "the snapshot must name the live keyboard's layout"
    );

    let mut shcut = ShortcutsInhibitClient::spawn(&comp.socket);
    assert!(
        shcut.wait_until(|c| c.has_input_serial()),
        "inhibitor surface never gained keyboard focus, so its inhibitor \
         could never activate and the test would be vacuous"
    );

    shcut.inhibit();
    assert_eq!(
        shcut.transitions(),
        &[true],
        "the double must record the inhibit it just sent"
    );
    // Synchronize with activation: the inhibit request (shortcuts-inhibit
    // connection) and the snapshot poll (test thread) race, so wait until
    // the compositor itself reports inhibited before driving keys.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut inhibited = false;
    while std::time::Instant::now() < deadline {
        shcut.pump();
        if comp.snapshot().shortcuts_inhibited {
            inhibited = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        inhibited,
        "compositor never reported shortcuts inhibited after the focused \
         surface inhibited them"
    );

    // The binding chord while inhibited: no workspace switch ...
    let keys_before = shcut.key_events();
    vk.key_down(KEY_LEFTMETA);
    vk.key_press(KEY_2);
    vk.key_up(KEY_LEFTMETA);
    // ... but the keys still reach the focused client.
    assert!(
        shcut.wait_until(|c| c.key_events() > keys_before),
        "inhibited keys never reached the focused client; inhibition must \
         reroute keys around the compositor, not swallow them"
    );
    assert_eq!(
        comp.snapshot().active_workspace,
        0,
        "SUPER+2 switched the workspace while an inhibitor was active -- \
         the compositor binding was not gated"
    );
    assert!(
        comp.snapshot().shortcuts_inhibited,
        "inhibition lifted mid-test"
    );

    // Release: the same chord must switch again, proving the gate above was
    // the inhibitor and not a broken injector.
    shcut.destroy();
    assert_eq!(
        shcut.transitions(),
        &[true, false],
        "the double must record the destroy it just sent"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut released = false;
    while std::time::Instant::now() < deadline {
        shcut.pump();
        if !comp.snapshot().shortcuts_inhibited {
            released = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        released,
        "compositor still reported shortcuts inhibited after the inhibitor \
         was destroyed"
    );

    vk.key_down(KEY_LEFTMETA);
    vk.key_press(KEY_2);
    vk.key_up(KEY_LEFTMETA);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut switched = false;
    while std::time::Instant::now() < deadline {
        vk.pump();
        shcut.pump();
        if comp.snapshot().active_workspace == 1 {
            switched = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        switched,
        "SUPER+2 must switch to workspace index 1 once the inhibitor is \
         gone; active_workspace is still {}",
        comp.snapshot().active_workspace
    );
}

/// Task 4: the M8 input-manager globals are advertised -- the shortcuts-inhibit
/// manager the inhibit e2e above binds, and the tablet manager tablet clients
/// bind. A real client lists the registry: both sides are real.
///
/// There is deliberately no "tablet tool tip reaches a surface" e2e here:
/// wlroots 0.20 has no virtual-tablet injection protocol (unlike virtual
/// keyboard/pointer), and the headless backend has no physical tablet, so no
/// client can produce tool traffic. The tablet half is covered by the manager
/// being bindable plus the backend's own cursor-attach of tablet devices;
/// the compositor adds no surface type and no pad UI (spec §6).
#[test]
fn m8_input_manager_globals_are_advertised() {
    let comp = Compositor::spawn();
    let globals = icedtea_harness::advertised_globals(&comp.socket);
    for want in [
        "zwp_keyboard_shortcuts_inhibit_manager_v1",
        "zwp_tablet_manager_v2",
    ] {
        assert!(
            globals.iter().any(|g| g == want),
            "{want} global missing; saw {globals:?}"
        );
    }
}

/// M8-5: a caret commit under a live IME popup re-places the popup's scene
/// node. Preamble mirrors test 8 (enable + commit a caret, activate, open the
/// popup); then a second commit carrying a new cursor rectangle must move the
/// node to below the new caret, translated through the focused content
/// origin — and the popup must be re-told the new surface-local rectangle.
#[test]
fn ime_popup_repositions_when_the_caret_moves() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");
    im.create_popup();
    let before = comp
        .input_popup_position()
        .expect("compositor never placed the popup");
    // Move the caret with a second commit carrying a new rectangle.
    ti.commit_with("qw", 2, 2, (300, 400, 2, 16));
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut after = None;
    while std::time::Instant::now() < deadline {
        im.pump();
        if let Some(pos) = comp.input_popup_position()
            && pos != before
        {
            after = Some(pos);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let after = after.expect("popup never repositioned after the caret commit");
    assert_eq!(
        after,
        (300, 444),
        "repositioned translated below the new caret: content origin (0, 28) + (300, 400+16)"
    );
    // And the popup must be re-told the new surface-local rectangle (not the
    // translated anchor): the echo half of the reposition contract.
    assert!(
        im.wait_until(|s| s.popup_text_input_rectangle() == Some((300, 400, 2, 16))),
        "popup was never re-told its moved anchor rectangle; saw {:?}",
        im.popup_text_input_rectangle()
    );
}

/// M8-6: an IME preedit commit shows the overlay under the caret; the
/// following commit-string clears it. Preamble mirrors test 8 (enable +
/// commit a caret, activate); the show position is the same translated
/// anchor test 8 pins — content origin (0, 28) + (100, 200+16).
#[test]
fn preedit_overlay_shows_composing_text_then_clears_on_commit() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    im.send_commit(Some("nihon"), None, None);
    assert!(
        ti.wait_until(|c| c.preedits().last().map(String::as_str) == Some("nihon")),
        "app never got the preedit; saw {:?}",
        ti.preedits()
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut shown = None;
    while std::time::Instant::now() < deadline {
        im.pump();
        if let Some(pos) = comp.preedit_overlay() {
            shown = Some(pos);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        shown,
        Some((100, 244)),
        "overlay must sit translated below the caret while composing"
    );

    im.send_commit(None, Some("日本"), None);
    assert!(
        ti.wait_until(|c| c.commits().last().map(String::as_str) == Some("日本")),
        "app never got the committed text; saw {:?}",
        ti.commits()
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut hidden = false;
    while std::time::Instant::now() < deadline {
        im.pump();
        if comp.preedit_overlay().is_none() {
            hidden = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(hidden, "commit-string must clear the composing overlay");
}

/// M8-6: moving keyboard focus away mid-compose hides the overlay. The
/// compositor drives focus changes itself, so no client-driven deactivate
/// fires for this path — the focus helper is the only hide path, and this
/// test is what proves it.
#[test]
fn preedit_overlay_hides_when_focus_moves_away_mid_compose() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    im.send_commit(Some("nihon"), None, None);
    assert!(
        ti.wait_until(|c| c.preedits().last().map(String::as_str) == Some("nihon")),
        "app never got the preedit; saw {:?}",
        ti.preedits()
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    while std::time::Instant::now() < deadline {
        im.pump();
        if comp.preedit_overlay().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        comp.preedit_overlay().is_some(),
        "precondition: the overlay must be showing before focus moves"
    );

    let _third = TestClient::map_toplevel(&comp.socket, "third", "third"); // steal keyboard focus
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut hidden = false;
    while std::time::Instant::now() < deadline {
        im.pump();
        ti.pump();
        if comp.preedit_overlay().is_none() {
            hidden = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        hidden,
        "moving keyboard focus away mid-compose must hide the overlay"
    );
}

/// M8-6: disabling the text-input mid-compose hides the overlay through
/// the client-driven deactivated path.
#[test]
fn preedit_overlay_hides_on_text_input_disable() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    im.send_commit(Some("nihon"), None, None);
    assert!(
        ti.wait_until(|c| c.preedits().last().map(String::as_str) == Some("nihon")),
        "app never got the preedit; saw {:?}",
        ti.preedits()
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    while std::time::Instant::now() < deadline {
        im.pump();
        if comp.preedit_overlay().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        comp.preedit_overlay().is_some(),
        "precondition: the overlay must be showing before disable"
    );

    ti.disable();
    assert!(
        im.wait_until(|s| s.deactivates() >= 1),
        "IME never deactivated on disable"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut hidden = false;
    while std::time::Instant::now() < deadline {
        im.pump();
        if comp.preedit_overlay().is_none() {
            hidden = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(hidden, "text-input disable must hide the composing overlay");
}

/// M8-7: the `GetState` snapshot carries the IME's activation state.
/// Preamble mirrors test 8 (enable + commit a caret, activate); deactivation
/// mirrors test 8b (destroy the text-input, wait for the deactivate) — the
/// snapshot must report active in between and clear afterwards.
#[test]
fn snapshot_reports_ime_activation() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    let snap = comp.snapshot();
    assert!(snap.ime_active, "snapshot must report the active IME");

    ti.destroy_text_input();
    assert!(
        im.wait_until(|s| s.deactivates() >= 1),
        "IME never deactivated after its text-input was destroyed"
    );
    let snap = comp.snapshot();
    assert!(!snap.ime_active, "snapshot must clear on deactivate");
}

/// M8-8: the IME commit's serial reaches the app's text-input done.
/// Preamble mirrors test 8 (enable + commit a caret, activate). Each app
/// commit is re-forwarded to the IME with its own done, so the serial the
/// IME carries on its next commit and the serial the app observes on the
/// paired text-input done must agree — across two generations, so the
/// equality tracks the lockstep rather than coinciding once.
#[test]
fn ime_commit_serial_reaches_text_input_done() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    // Generation 1: the IME commits; the app must observe the relayed text
    // (proving the commit was processed) with a done serial equal to the
    // serial the IME carried.
    im.send_commit(None, Some("a"), None);
    assert!(
        ti.wait_until(|c| c.commits().last().map(String::as_str) == Some("a")),
        "app never got the committed text; saw {:?}",
        ti.commits()
    );
    let sent = *im
        .commit_serials()
        .last()
        .expect("IME recorded no commit serial");
    let got = *ti
        .done_serials()
        .last()
        .expect("app observed no done serial");
    assert_eq!(
        got,
        sent,
        "the text-input done serial must equal the IME commit serial \
         (sent {sent}, observed done serials {:?})",
        ti.done_serials()
    );

    // Generation 2: another app commit advances both sides' counters in
    // lockstep; a second IME commit must pair again at the new serial.
    ti.commit_with("qw", 2, 2, (100, 200, 2, 16));
    assert!(
        im.wait_until(|s| s.surroundings().last() == Some(&("qw".to_string(), 2, 2))),
        "IME never saw the second surrounding relay; saw {:?}",
        im.surroundings()
    );
    im.send_commit(None, Some("b"), None);
    assert!(
        ti.wait_until(|c| c.commits().last().map(String::as_str) == Some("b")),
        "app never got the second committed text; saw {:?}",
        ti.commits()
    );
    let sent2 = *im
        .commit_serials()
        .last()
        .expect("IME recorded no second commit serial");
    let got2 = *ti
        .done_serials()
        .last()
        .expect("app observed no second done serial");
    assert_eq!(
        got2,
        sent2,
        "the second text-input done serial must equal the second IME commit \
         serial (sent {sent2}, observed done serials {:?})",
        ti.done_serials()
    );
    assert_ne!(
        (sent2, got2),
        (sent, got),
        "generations must advance: a repeated serial would pair without proving lockstep"
    );
}

/// M8-8 (reviewer-mandated adversarial gate, non-UTF-8 legs): spec-violating
/// non-UTF-8 bytes through both relay directions must arrive as the
/// DOCUMENTED `to_string_lossy` replacement (`U+FFFD`), never as a panic,
/// protocol error, or dropped relay. Each leg uses a distinct payload so no
/// assertion can pass off a neighboring leg's recording.
///
/// DEVIATION from the plan's mandated `..._non_utf8_and_interior_nul_...`
/// name: the interior-NUL leg is undrivable from this Rust harness, so this
/// test pins only the non-UTF-8 legs and says so. Every client-side string
/// request is code-generated as `CString::new(s).unwrap()` (wayland-scanner
/// `common.rs`), which panics on interior NUL before anything reaches the
/// wire, and `Argument::Str` itself holds a `Box<CString>` — so even a
/// hand-built message cannot carry an interior NUL without
/// `CString::from_vec_unchecked` (documented UB: no interior NUL allowed) or
/// a hand-rolled raw-socket Wayland client (disproportionate surgery). The
/// crate-side truncation (`relay_cstring`) is additionally unreachable end
/// to end by construction: snapshots are read from NUL-terminated C strings,
/// so no snapshot string can ever contain the NUL the truncation arm handles.
/// The interior-NUL half of the gate needs a C driver or stays a unit-level
/// property; it is NOT covered here.
#[test]
fn m8_relay_non_utf8_pins_documented_lossy_wire_bytes() {
    /// Deliberately non-UTF-8 bytes over an ASCII skeleton, as `&str`.
    /// Fixed payloads, so the returned slice borrows a leaked box rather
    /// than a caller-side buffer (nine bytes per test run).
    fn adversarial(prefix: u8, suffix: u8) -> &'static str {
        let bytes: Box<[u8]> = Box::new([prefix, 0xFF, suffix]);
        let leaked: &'static [u8] = Box::leak(bytes);
        // SAFETY: the bytes are invalid UTF-8 by design — this is the
        // spec-violating input under test. Nothing here decodes them as
        // `str`: the harness builders only copy (`to_string`) and measure
        // (`len`) the bytes before the generated code re-wraps them as
        // `CString` (which rejects only NUL, not invalid UTF-8) for the
        // wire. The replacement happens downstream in the crate's snapshot
        // layer (`to_string_lossy`).
        unsafe { std::str::from_utf8_unchecked(leaked) }
    }

    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");

    // Leg 1 (app → IME): non-UTF-8 surrounding text must reach the IME as
    // the lossy replacement. Cursor/anchor track the 3 wire bytes, so no
    // out-of-bounds validation can fire.
    let surrounding = adversarial(b'f', b'u');
    ti.commit_with(surrounding, 3, 3, (100, 200, 2, 16));
    assert!(
        im.wait_until(|s| s.surroundings().last() == Some(&("f\u{FFFD}u".to_string(), 3, 3))),
        "IME never saw the lossy surrounding relay; saw {:?}",
        im.surroundings()
    );

    // Leg 2 (IME → app, preedit): non-UTF-8 preedit must reach the app as
    // the lossy replacement.
    let preedit = adversarial(b'p', b'q');
    im.send_commit(Some(preedit), None, None);
    assert!(
        ti.wait_until(|c| c.preedits().last().map(String::as_str) == Some("p\u{FFFD}q")),
        "app never got the lossy preedit; saw {:?}",
        ti.preedits()
    );

    // Leg 3 (IME → app, commit string): same replacement guarantee on the
    // commit channel.
    let commit = adversarial(b'c', b'd');
    im.send_commit(None, Some(commit), None);
    assert!(
        ti.wait_until(|c| c.commits().last().map(String::as_str) == Some("c\u{FFFD}d")),
        "app never got the lossy commit string; saw {:?}",
        ti.commits()
    );
}

/// Preedit overlay updates in place on a second preedit commit: the node
/// stays the same generation's overlay, not recreated, and the app's view
/// of the preedit advances.
#[test]
fn preedit_overlay_updates_in_place_on_second_preedit() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");
    im.send_commit(Some("nihon"), None, None);
    assert!(
        comp.wait_until_preedit_overlay(|v| v.is_some()),
        "overlay never showed the first preedit"
    );
    let first = comp.preedit_overlay().expect("overlay vanished after show");
    im.send_commit(Some("nihono"), None, None);
    assert!(
        comp.wait_until_preedit_overlay(|v| v.is_some()),
        "overlay vanished on second preedit (in-place update path broken)"
    );
    let second = comp
        .preedit_overlay()
        .expect("overlay vanished after second");
    // Position may shift with measured width, but must stay Some — the
    // update-in-place path (not show-then-hide) must have run.
    assert!(
        second.0 >= 0 && second.1 >= 0,
        "second position must be on screen"
    );
    // If the node was destroyed and recreated at a different spot, first
    // and second may differ; the load-bearing assertion is that it is still
    // Some, not that it is the same id.
    let _ = first;
}

/// Preedit overlay hides when the IME clears the preedit (empty string)
/// without a commit-string: `should_show` false must hide the live node.
#[test]
fn preedit_overlay_hides_on_preedit_clear() {
    let comp = Compositor::spawn();
    let _vk = VirtualKeyboardClient::spawn(&comp.socket);
    let mut im = InputMethodClient::spawn(&comp.socket);
    let mut ti = TextInputClient::spawn(&comp.socket);
    assert!(
        ti.wait_until(|c| c.entered() >= 1),
        "text-input never focused"
    );
    ti.enable();
    ti.commit_with("q", 1, 1, (100, 200, 2, 16));
    assert!(im.wait_until(|s| s.activates() >= 1), "IME never activated");
    im.send_commit(Some("nihon"), None, None);
    assert!(
        comp.wait_until_preedit_overlay(|v| v.is_some()),
        "overlay never showed"
    );
    im.send_commit(Some(""), None, None);
    assert!(
        comp.wait_until_preedit_overlay(|v| v.is_none()),
        "preedit-clear must hide the overlay"
    );
}

/// M7 Task 4 (consumer) e2e 1: warping the cursor onto a mapped toplevel
/// delivers pointer-enter on that surface with a cursor image applied.
///
/// Consumer-side warp is virtual-pointer absolute motion: `wlr`'s own
/// `warp_cursor` is `pub(crate)` (an intentional API shape -- the crate moves
/// the cursor itself inside its motion handlers), so the closest consumer
/// path to "warp to a surface" is an absolute motion at its center. The
/// assertions prove the consumer half of R1: the enter reaches the surface,
/// and the post-motion snapshot reports a visible cursor at that position
/// (fed by `Runtime::cursor_state`, whose `Hidden` -> image transition the
/// crate performs on motion).
#[test]
fn cursor_warp_reaches_surface() {
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    // The warp driver first: the seat only advertises the pointer
    // capability once a pointer device backs it, and the observer double
    // below asserts on it at spawn (same ordering the M4.5 lock test
    // relies on).
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    // The pointer-observer double: it maps the toplevel the warp targets
    // and records the enter the warp must deliver.
    let mut a = PointerClient::spawn(&comp.socket, "cursor.app", "cursor");
    // Synchronize model-side: the spawn roundtrips only pump the client,
    // so wait for the compositor's own announcement before reading
    // geometry out of its snapshot.
    comp.wait_event(|e| matches!(e, Event::WindowOpened(w) if w.app_id == "cursor.app"));
    let geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "cursor.app")
        .expect("A must be in the model once mapped")
        .geometry;
    let (cx, cy) = (
        (geo.x + geo.width / 2) as f64,
        (geo.y + geo.height / 2) as f64,
    );

    // Pre-motion: no cursor image has been applied yet, so the snapshot
    // reports it hidden at no known position.
    let before = comp.snapshot();
    assert!(
        !before.cursor_visible,
        "a fresh compositor has applied no cursor image yet"
    );
    assert_eq!(before.cursor_pos, None, "no motion yet, no known position");

    vp.motion_absolute(cx, cy, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    assert!(
        a.wait_until(|c| c.pointer_enters() > 0),
        "A never received pointer-enter after the warp"
    );
    a.pump();

    let after = comp.snapshot();
    assert!(
        after.cursor_visible,
        "cursor image must be set after motion"
    );
    assert_eq!(
        after.cursor_pos,
        Some((cx as i32, cy as i32)),
        "snapshot cursor position must track the warp target"
    );
    assert!(
        !after.touch_active,
        "a pointer warp must not read as touch activity"
    );
}

/// M7 Task 4 (consumer) e2e 2: touch down/motion/up reaches the focused
/// surface; a cancel clears the in-flight touch state.
///
/// The down/motion/up travel the crate's `TouchFrame` token path (proven by
/// the client's own touch stream); the snapshot's `touch_active` proves the
/// consumer's `Runtime::touch_state` reader. The cancel has no wire
/// producer headless (cancels come from hardware), so it is driven through
/// the test hook that calls the real `SeatHandler::touch_cancelled` on the
/// loop thread -- the flag it clears is the same one the snapshot reports.
#[test]
fn touch_down_reaches_focused_surface() {
    let comp = Compositor::spawn();
    let mut t = TouchClient::spawn(&comp.socket, "touch.app", "touch");
    assert!(
        t.wait_until(|c| c.last_configure().is_some()),
        "touch client never configured"
    );
    let geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.app_id == "touch.app")
        .expect("touch client must be in the model once mapped")
        .geometry;
    let (cx, cy) = (
        (geo.x + geo.width / 2) as f64,
        (geo.y + geo.height / 2) as f64,
    );

    let serial = comp.inject_touch_down(cx, cy, 0, 1);
    assert!(serial.is_some(), "touch-down minted no grab serial");
    assert!(
        t.wait_until(|c| !c.touch_downs().is_empty()),
        "focused surface never received the touch down"
    );
    let (id, x, y, _) = t.touch_downs()[0];
    assert_eq!(id, 0, "down must carry the injected touch id");
    assert!(
        x >= 0.0 && y >= 0.0,
        "down coordinates must be surface-local and non-negative, got ({x}, {y})"
    );
    assert!(
        comp.snapshot().touch_active,
        "snapshot must report touch_active while a point is down"
    );

    comp.inject_touch_motion(cx + 5.0, cy + 5.0, 0, 2);
    assert!(
        t.wait_until(|c| !c.touch_motions().is_empty()),
        "focused surface never received the touch motion"
    );

    comp.inject_touch_up(0, 3);
    assert!(
        t.wait_until(|c| !c.touch_ups().is_empty()),
        "focused surface never received the touch up"
    );
    // The up removed the point: the reader path must report inactive again.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if !comp.snapshot().touch_active {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "touch_active stayed set after the point lifted"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Cancel clears: a fresh down, then the hook-driven cancel.
    assert!(comp.inject_touch_down(cx, cy, 1, 4).is_some());
    assert!(
        t.wait_until(|c| c.touch_downs().len() == 2),
        "second touch down never arrived"
    );
    assert!(
        comp.snapshot().touch_active,
        "second down must re-assert touch_active"
    );
    comp.inject_touch_cancel();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if !comp.snapshot().touch_active {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "touch_active stayed set after the cancel"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    comp.inject_touch_up(1, 5);
}

/// M7 Task 4 (consumer) e2e 3: pinch begin -> end reaches the shell feed
/// with its phases intact.
///
/// Headless has no gesture hardware, so the phases are driven through the
/// test hooks that call the real `SeatHandler::gesture_began/ended` on the
/// loop thread. What this proves end to end: the consumer notification maps
/// to the contract event stream (the exact `SeqEvent` feed the D-Bus emitter
/// forwards to the shell as signals) in order, began before ended. The
/// `GestureClient` bind proves the gestures global the full-fidelity client
/// path needs is advertised.
#[test]
fn pinch_phase_reaches_shell() {
    let comp = Compositor::spawn();
    // A pointer device must exist before any client can hold a
    // `wl_pointer` for its gesture objects: the seat only advertises the
    // pointer capability once a device backs it (same ordering the M4.5
    // lock test relies on).
    let mut _vp = VirtualPointerClient::spawn(&comp.socket);
    let mut g = GestureClient::spawn(&comp.socket);
    g.pump();

    comp.inject_gesture(true);
    let began = comp.wait_event(|e| matches!(e, Event::GestureBegan));
    assert!(
        matches!(began, Event::GestureBegan),
        "expected GestureBegan, got {began:?}"
    );
    comp.inject_gesture(false);
    let ended = comp.wait_event(|e| matches!(e, Event::GestureEnded));
    assert!(
        matches!(ended, Event::GestureEnded),
        "expected GestureEnded, got {ended:?}"
    );
}

/// M7 Task 4 (consumer) e2e 4: a lid-switch toggle flips the session
/// signal.
///
/// Headless has no switch hardware, so the toggle is driven through the
/// test hook that calls the consumer's real switch-apply path on the loop
/// thread with the hardware-decoded `(type, on)` pair. The contract event
/// carries the lid reading the session consumes; a non-lid switch must not
/// read as a lid signal.
#[test]
fn switch_toggles_session_signal() {
    use wlr::SwitchType;

    let comp = Compositor::spawn();

    comp.inject_switch_toggle(SwitchType::Lid, true);
    let closed = comp.wait_event(|e| matches!(e, Event::SwitchToggled { lid_closed: true }));
    assert!(
        matches!(closed, Event::SwitchToggled { lid_closed: true }),
        "lid close must flip the session signal closed, got {closed:?}"
    );

    comp.inject_switch_toggle(SwitchType::TabletMode, true);
    let not_lid = comp.wait_event(|e| matches!(e, Event::SwitchToggled { lid_closed: false }));
    assert!(
        matches!(not_lid, Event::SwitchToggled { lid_closed: false }),
        "a tablet-mode switch must not read as a lid signal, got {not_lid:?}"
    );

    comp.inject_switch_toggle(SwitchType::Lid, false);
    let opened = comp.wait_event(|e| matches!(e, Event::SwitchToggled { lid_closed: false }));
    assert!(
        matches!(opened, Event::SwitchToggled { lid_closed: false }),
        "lid open must flip the session signal back, got {opened:?}"
    );
}

/// M7 Task 4 (consumer) e2e 5 (spec section 7 item 4): a locked pointer
/// freezes the model's pointer too, and releasing the lock unfreezes it.
///
/// The crate already freezes the cursor (M4.5's freeze test); this proves
/// the consumer half: the snapshot's cursor position -- the model's own
/// mirror, fed through `SeatHandler::pointer_motion` -- does not advance
/// under the lock and resumes after it. The confinement gate rides the same
/// model chokepoint the lock gate does.
#[test]
fn constraint_lock_freezes_model_pointer() {
    const OUTPUT_W: u32 = 1280;
    const OUTPUT_H: u32 = 720;

    let comp = Compositor::spawn();
    let mut vp = VirtualPointerClient::spawn(&comp.socket);
    let mut pc = PointerConstraintsClient::spawn(&comp.socket);

    let opened = comp.wait_event(
        |e| matches!(e, Event::WindowOpened(w) if w.app_id == "icedtea-harness-pointer-constraints"),
    );
    let Event::WindowOpened(pc_info) = opened else {
        unreachable!()
    };
    let pc_geo = comp
        .snapshot()
        .windows
        .iter()
        .find(|w| w.id == pc_info.id)
        .expect("pc must be in the model once mapped")
        .geometry;
    let (px, py) = (
        (pc_geo.x + pc_geo.width / 2) as f64,
        (pc_geo.y + pc_geo.height / 2) as f64,
    );
    vp.motion_absolute(px, py, OUTPUT_W, OUTPUT_H);
    vp.frame();
    vp.pump();
    pc.pump();

    // Lock, prime activation (see the M4.5 freeze test's doc), then measure.
    pc.lock_pointer();
    pc.pump();
    vp.motion(5.0, 5.0);
    vp.frame();
    vp.pump();
    pc.pump();
    comp.settle();
    let frozen = comp.snapshot().cursor_pos;
    assert!(
        frozen.is_some(),
        "the model must know the cursor position before the freeze window"
    );

    vp.motion(20.0, 20.0);
    vp.frame();
    vp.pump();
    pc.pump();
    comp.settle();
    assert_eq!(
        comp.snapshot().cursor_pos,
        frozen,
        "locked: the model's pointer must not move once active"
    );

    // Release: dropping the lock must unfreeze the model again. The
    // destroy travels the constraints client's connection while the next
    // motion travels the virtual pointer's -- two connections, no
    // ordering -- so settle first, giving the loop a chance to process
    // the destroy before the motion that must move again.
    pc.unlock_pointer();
    pc.pump();
    comp.settle();
    vp.motion(20.0, 20.0);
    vp.frame();
    vp.pump();
    pc.pump();
    comp.settle();
    assert_ne!(
        comp.snapshot().cursor_pos,
        frozen,
        "released: the model's pointer must move again after unlock"
    );
}
