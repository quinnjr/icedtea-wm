//! Task 17: a second headless output must land in `state.outputs` with a
//! real, disjoint layout-box geometry -- not the (0,0)-origin fallback
//! `OutputHandler::new_output` used before `Runtime::output_layout_box`
//! existed.
//!
//! A separate integration-test binary, not an addition to `headless_boot.rs`:
//! `WLR_HEADLESS_OUTPUTS` is read once by `Backend::autocreate` and every
//! other test in that binary assumes the default single-output headless
//! backend. Each `tests/*.rs` file is its own process, so the env var set
//! here can never leak into `headless_boot.rs`'s tests (or vice versa).

use icedtea_compositor::state::State;
// `output_configuration_applied` is a `wlr::OutputHandler` method; the trait
// must be in scope to call it on `State`.
use wlr::OutputHandler;

/// Set the headless-backend environment (two outputs, this file's own
/// concern) exactly once, no matter which of this binary's `#[test]`s
/// reaches it first. See `headless_boot.rs`'s `ensure_headless_env` for the
/// torn-environment hazard this guards against -- identical reasoning, this
/// file's own copy because each integration-test binary has its own statics.
fn ensure_headless_env() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // SAFETY (icedtea unsafe exception (c)): `Once::call_once` above
        // guarantees this closure runs on exactly one thread and that every
        // other thread calling `ensure_headless_env` blocks until it
        // finishes -- so nothing can observe a torn read and nothing races
        // this write.
        unsafe {
            std::env::set_var("WLR_BACKENDS", "headless");
            std::env::set_var("WLR_HEADLESS_OUTPUTS", "2");
        }
    });
}

/// Serializes compositor *creation* across this binary's test threads. See
/// `headless_boot.rs`'s `BOOT_LOCK` for the full argument (the process-global,
/// unsynchronized `wl_array` of buffer-resource interfaces `init_graphics`
/// grows). Duplicated rather than shared because each integration-test file
/// is its own binary with its own statics.
static BOOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take [`BOOT_LOCK`] (and set the headless environment) for the duration of
/// one compositor's creation. `drop` the returned guard once the
/// display/backend/runtime triple exists.
fn boot_lock() -> std::sync::MutexGuard<'static, ()> {
    ensure_headless_env();
    BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn a_second_headless_output_is_tracked_with_a_layout_box() {
    let boot = boot_lock();
    let display = wlr::Display::new().expect("display");
    let backend = wlr::Backend::autocreate(&display.event_loop()).expect("backend");
    let runtime = wlr::Runtime::new().expect("runtime");
    runtime.init_graphics(&display, &backend).expect("graphics");
    runtime.create_xdg_shell(&display, 6).expect("xdg_wm_base");
    runtime.create_seat(&display, "seat0").expect("seat0");
    drop(boot);

    // `state` is declared (and so, by ordinary end-of-scope drop order,
    // dropped) after `display`/`backend`/`runtime`, matching every
    // `headless_boot.rs` test: `attach` below gives `state.wayland` a
    // `Runtime` clone, and `wlr` documents that a `Runtime` must not outlive
    // the `Display` it was initialized against.
    let (tx, _rx) = crossbeam_channel::unbounded();
    let mut state = State::new(icedtea_config::default_config(), tx);
    state.wayland.attach(runtime.clone());

    let background = runtime
        .add_rect(
            1,
            1,
            icedtea_compositor::render::wallpaper_color(&state.config.appearance),
        )
        .expect("background rect");
    runtime.lower_rect_to_bottom(background);
    state.set_background(background);

    // The command channel and its wake pipe, used only as this test's
    // bounded backstop -- same role as `headless_boot.rs`'s `run_all` tests:
    // it gives both headless outputs time to arrive (and so
    // `OutputHandler::new_output` time to run twice) before asking the loop
    // to stop.
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    state.set_command_receiver(cmd_rx);
    let (cmd_wake_write, cmd_wake_id) =
        icedtea_compositor::backend::wake_source(&runtime).expect("cmd wake source");
    state.set_cmd_wake_source(cmd_wake_id);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        let _ = cmd_tx.send(icedtea_compositor::dbus::DbCommand::Quit);
        icedtea_compositor::backend::wake(&cmd_wake_write);
    });

    backend
        .run_all(&display, &mut state, &runtime, wlr::Until::Stop)
        .expect("run_all");

    assert!(
        state.quitting,
        "the backstop Quit command must have stopped the loop"
    );

    assert_eq!(
        state.outputs.len(),
        2,
        "both headless outputs must have reached the model"
    );

    let geometries: Vec<_> = state.outputs.values().map(|o| o.geometry).collect();
    let a = geometries[0];
    let b = geometries[1];
    assert!(
        a.width > 0 && a.height > 0 && b.width > 0 && b.height > 0,
        "both outputs report a real size, got {a:?} and {b:?}"
    );
    // Disjoint boxes are the proof the layout-box path (not the
    // (0,0)-at-origin fallback, which would stack both outputs on top of
    // each other) produced these geometries.
    let disjoint = a.x + a.width <= b.x
        || b.x + b.width <= a.x
        || a.y + a.height <= b.y
        || b.y + b.height <= a.y;
    assert!(
        disjoint,
        "the two outputs' layout boxes must not overlap, got {a:?} and {b:?}"
    );
}

/// [HIGH H4] `sync_wallpaper_nodes`' hot-unplug cleanup branch: a wallpaper
/// buffer node for an output that is no longer in `state.outputs` (simulating
/// `OutputHandler::destroyed` having already removed it) must be torn down on
/// the next sync, not left dangling.
///
/// Needs the two-output headless boot this file already sets up
/// (`WLR_HEADLESS_OUTPUTS=2`): one wallpaper node per output is created first,
/// then one output is removed from the model directly (the same thing
/// `OutputHandler::destroyed` does before calling `sync_wallpaper_nodes`), and
/// a second sync must drop the orphaned node.
#[test]
fn sync_wallpaper_nodes_removes_a_node_for_an_output_that_is_gone() {
    let boot = boot_lock();
    let display = wlr::Display::new().expect("display");
    let backend = wlr::Backend::autocreate(&display.event_loop()).expect("backend");
    let runtime = wlr::Runtime::new().expect("runtime");
    runtime.init_graphics(&display, &backend).expect("graphics");
    runtime.create_xdg_shell(&display, 6).expect("xdg_wm_base");
    runtime.create_seat(&display, "seat0").expect("seat0");
    drop(boot);

    // `state` is declared (and so dropped) after `display`/`backend`/`runtime`
    // for the same reason as every other test in this file.
    let (tx, _rx) = crossbeam_channel::unbounded();
    let mut state = State::new(icedtea_config::default_config(), tx);
    state.wayland.attach(runtime.clone());

    let background = runtime
        .add_rect(
            1,
            1,
            icedtea_compositor::render::wallpaper_color(&state.config.appearance),
        )
        .expect("background rect");
    runtime.lower_rect_to_bottom(background);
    state.set_background(background);

    // The command channel and its wake pipe, used only as this test's
    // bounded backstop -- gives both headless outputs time to arrive before
    // asking the loop to stop.
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    state.set_command_receiver(cmd_rx);
    let (cmd_wake_write, cmd_wake_id) =
        icedtea_compositor::backend::wake_source(&runtime).expect("cmd wake source");
    state.set_cmd_wake_source(cmd_wake_id);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        let _ = cmd_tx.send(icedtea_compositor::dbus::DbCommand::Quit);
        icedtea_compositor::backend::wake(&cmd_wake_write);
    });

    backend
        .run_all(&display, &mut state, &runtime, wlr::Until::Stop)
        .expect("run_all");

    assert!(
        state.quitting,
        "the backstop Quit command must have stopped the loop"
    );
    assert_eq!(
        state.outputs.len(),
        2,
        "both headless outputs must have reached the model"
    );

    let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 8, 7, 255]));
    state.wallpaper.set_decoded(Some(image));
    state.sync_wallpaper_nodes();
    assert_eq!(
        state.wallpaper_node_count(),
        2,
        "one buffer node per output"
    );

    // Simulate `OutputHandler::destroyed`: it removes the output from the
    // model before calling `sync_wallpaper_nodes` (see `state.rs`), so drop
    // one output out from under the wallpaper node map here directly.
    let gone_index = *state.outputs.keys().min().expect("at least one output");
    state.outputs.remove(&gone_index);
    state.sync_wallpaper_nodes();

    assert_eq!(
        state.wallpaper_node_count(),
        1,
        "the node for the removed output must be torn down on the next sync"
    );
}

/// A minimal [`wlr::AppliedHead`] naming `name` with the given enabled state --
/// the fields `output_configuration_applied`'s enable/disable/rehydrate
/// branches actually read here. Width/height are populated for the enabled
/// case so the head-reported geometry fallback is available even if the layout
/// box lookup returns `None`.
fn applied_head(name: &str, enabled: bool) -> wlr::AppliedHead {
    wlr::AppliedHead {
        name: Some(name.to_string()),
        enabled,
        width: if enabled { 1920 } else { 0 },
        height: if enabled { 1080 } else { 0 },
        refresh_mhz: 0,
        x: 0,
        y: 0,
        scale: 1.0,
        transform: wlr::Transform::Normal,
    }
}

/// T5 display-config fix (defect 2): an output DISABLED via
/// `output_configuration_applied` leaves the active set (`state.outputs`) but
/// stays tracked by connector name -> `wlr::OutputId` in `disabled_outputs`, so
/// a later RE-ENABLE of the same connector rehydrates it back into the active
/// set (a fresh index + surface, its id re-mapped) instead of being left
/// enabled-but-untracked and rendering nothing until restart.
///
/// Drives the handler directly with owned `AppliedHead`s. The full
/// zwlr_output_manager_v1 client round-trip is T6/T7; this exercises exactly
/// the handler code the fix changed, with a live runtime so
/// `output_layout_box`/`create_output` run for real. Distinguishes the fix
/// from the pre-fix `continue`, which left `outputs.len()` stuck at 1 on
/// re-enable.
#[test]
fn a_disabled_output_can_be_re_enabled_within_a_session() {
    let boot = boot_lock();
    let display = wlr::Display::new().expect("display");
    let backend = wlr::Backend::autocreate(&display.event_loop()).expect("backend");
    let runtime = wlr::Runtime::new().expect("runtime");
    runtime.init_graphics(&display, &backend).expect("graphics");
    runtime.create_xdg_shell(&display, 6).expect("xdg_wm_base");
    runtime.create_seat(&display, "seat0").expect("seat0");
    drop(boot);

    // Declared (dropped) after display/backend/runtime, as every test here.
    let (tx, _rx) = crossbeam_channel::unbounded();
    let mut state = State::new(icedtea_config::default_config(), tx);
    state.wayland.attach(runtime.clone());

    // Isolate the off-loop redb persist `output_configuration_applied` kicks
    // off; without this it would write to the real XDG database path.
    let tmp = std::env::temp_dir().join(format!("icedtea-disp-{}.redb", std::process::id()));
    state.config_path = Some(tmp.clone());

    let background = runtime
        .add_rect(
            1,
            1,
            icedtea_compositor::render::wallpaper_color(&state.config.appearance),
        )
        .expect("background rect");
    runtime.lower_rect_to_bottom(background);
    state.set_background(background);

    // Bounded backstop: give both headless outputs time to arrive before the
    // loop stops -- identical to the other tests in this file.
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    state.set_command_receiver(cmd_rx);
    let (cmd_wake_write, cmd_wake_id) =
        icedtea_compositor::backend::wake_source(&runtime).expect("cmd wake source");
    state.set_cmd_wake_source(cmd_wake_id);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        let _ = cmd_tx.send(icedtea_compositor::dbus::DbCommand::Quit);
        icedtea_compositor::backend::wake(&cmd_wake_write);
    });

    backend
        .run_all(&display, &mut state, &runtime, wlr::Until::Stop)
        .expect("run_all");

    assert_eq!(
        state.outputs.len(),
        2,
        "both headless outputs must have reached the model"
    );

    // The connector name of one output -- the stable key the disable/re-enable
    // round-trip matches against.
    let victim_index = *state.outputs.keys().min().expect("an output");
    let victim_name = state
        .outputs
        .get(&victim_index)
        .expect("victim surface")
        .name
        .clone();
    assert!(!victim_name.is_empty(), "headless outputs are named");

    // DISABLE: drops it from the active set but records name -> id.
    state.output_configuration_applied(vec![applied_head(&victim_name, false)]);
    assert_eq!(
        state.outputs.len(),
        1,
        "the disabled output left the active set"
    );
    assert!(
        state.outputs.values().all(|o| o.name != victim_name),
        "no active surface still carries the disabled connector's name"
    );

    // RE-ENABLE the same connector: rehydrate must add it back under its own
    // name. Pre-fix this hit the `continue` and `outputs.len()` stayed at 1.
    state.output_configuration_applied(vec![applied_head(&victim_name, true)]);
    assert_eq!(
        state.outputs.len(),
        2,
        "the re-enabled output rejoined the active set"
    );
    assert_eq!(
        state
            .outputs
            .values()
            .filter(|o| o.name == victim_name)
            .count(),
        1,
        "exactly one active surface carries the re-enabled connector's name"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Task 3 (wlr 0.20.34 wire-up): `OutputHandler::output_committed` /
/// `output_precommitted` observe every commit into `state.last_commit`,
/// keyed by live output id, and an external MODE commit re-derives geometry
/// through the layout path.
///
/// No handler is driven directly: a `wlr::Output` handle cannot be built
/// outside the crate, so the headless loop's own commits (enable commits at
/// boot, which stage MODE, plus the frame path's scene commits) flow through
/// wlroots' precommit-then-commit emission and both handlers record. The
/// single slot is last-writer-wins in emission order, so the surviving record
/// per output is the commit's view of the staged fields; asserting its mask
/// is non-empty pins that both the staged mask and the commit timestamp made
/// it into the model. Record-only, assert-after-run (a panic inside a
/// handler body aborts through C).
#[test]
fn output_commit_and_precommit_are_observed_per_output() {
    let boot = boot_lock();
    let display = wlr::Display::new().expect("display");
    let backend = wlr::Backend::autocreate(&display.event_loop()).expect("backend");
    let runtime = wlr::Runtime::new().expect("runtime");
    runtime.init_graphics(&display, &backend).expect("graphics");
    runtime.create_xdg_shell(&display, 6).expect("xdg_wm_base");
    runtime.create_seat(&display, "seat0").expect("seat0");
    drop(boot);

    let (tx, _rx) = crossbeam_channel::unbounded();
    let mut state = State::new(icedtea_config::default_config(), tx);
    state.wayland.attach(runtime.clone());

    let background = runtime
        .add_rect(
            1,
            1,
            icedtea_compositor::render::wallpaper_color(&state.config.appearance),
        )
        .expect("background rect");
    runtime.lower_rect_to_bottom(background);
    state.set_background(background);

    // Bounded backstop: gives both headless outputs time to arrive, enable
    // (MODE-staging commits), and run at least one frame commit each.
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    state.set_command_receiver(cmd_rx);
    let (cmd_wake_write, cmd_wake_id) =
        icedtea_compositor::backend::wake_source(&runtime).expect("cmd wake source");
    state.set_cmd_wake_source(cmd_wake_id);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let _ = cmd_tx.send(icedtea_compositor::dbus::DbCommand::Quit);
        icedtea_compositor::backend::wake(&cmd_wake_write);
    });

    backend
        .run_all(&display, &mut state, &runtime, wlr::Until::Stop)
        .expect("run_all");

    assert!(
        state.quitting,
        "the backstop Quit command must have stopped the loop"
    );
    assert_eq!(
        state.outputs.len(),
        2,
        "both headless outputs must have reached the model"
    );

    // Every live output committed at least once (enable + frame path), so
    // each must have a last-commit record.
    assert_eq!(
        state.last_commit.len(),
        state.outputs.len(),
        "one last-commit record per live output"
    );
    for (id, (fields, _when)) in &state.last_commit {
        assert!(
            !fields.is_empty(),
            "the record for {id:?} must carry the staged fields, not an empty mask"
        );
    }

    // The boot enable commits stage MODE, which runs the re-derivation
    // branch: geometries must still be real and disjoint afterwards.
    let geometries: Vec<_> = state.outputs.values().map(|o| o.geometry).collect();
    let a = geometries[0];
    let b = geometries[1];
    assert!(
        a.width > 0 && a.height > 0 && b.width > 0 && b.height > 0,
        "MODE re-derivation must not collapse geometries, got {a:?} and {b:?}"
    );
    let disjoint = a.x + a.width <= b.x
        || b.x + b.width <= a.x
        || a.y + a.height <= b.y
        || b.y + b.height <= a.y;
    assert!(
        disjoint,
        "the two outputs' layout boxes must not overlap, got {a:?} and {b:?}"
    );
}
/// Review finding #5 (interactive guard): a client that disables every
/// connector must NOT be able to drive the compositor to zero active outputs.
/// An empty `state.outputs` makes `outputs.keys().min()` `None`, so window
/// placement / migration / reclaim all early-return -- a black screen with no
/// way back. `output_configuration_applied` refuses the disable that would
/// empty the active set (when the same apply enables nothing to replace it),
/// keeping at least one output live.
///
/// Drives the handler directly (like the re-enable test above) with a live
/// two-output headless runtime. Disabling the first output is honored (the
/// second survives); disabling the last remaining one is refused.
#[test]
fn disabling_every_output_keeps_at_least_one_active() {
    let boot = boot_lock();
    let display = wlr::Display::new().expect("display");
    let backend = wlr::Backend::autocreate(&display.event_loop()).expect("backend");
    let runtime = wlr::Runtime::new().expect("runtime");
    runtime.init_graphics(&display, &backend).expect("graphics");
    runtime.create_xdg_shell(&display, 6).expect("xdg_wm_base");
    runtime.create_seat(&display, "seat0").expect("seat0");
    drop(boot);

    let (tx, _rx) = crossbeam_channel::unbounded();
    let mut state = State::new(icedtea_config::default_config(), tx);
    state.wayland.attach(runtime.clone());

    // Isolate the off-loop redb persist from the real XDG database path.
    let tmp = std::env::temp_dir().join(format!("icedtea-lastout-{}.redb", std::process::id()));
    state.config_path = Some(tmp.clone());

    let background = runtime
        .add_rect(
            1,
            1,
            icedtea_compositor::render::wallpaper_color(&state.config.appearance),
        )
        .expect("background rect");
    runtime.lower_rect_to_bottom(background);
    state.set_background(background);

    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    state.set_command_receiver(cmd_rx);
    let (cmd_wake_write, cmd_wake_id) =
        icedtea_compositor::backend::wake_source(&runtime).expect("cmd wake source");
    state.set_cmd_wake_source(cmd_wake_id);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        let _ = cmd_tx.send(icedtea_compositor::dbus::DbCommand::Quit);
        icedtea_compositor::backend::wake(&cmd_wake_write);
    });

    backend
        .run_all(&display, &mut state, &runtime, wlr::Until::Stop)
        .expect("run_all");

    assert_eq!(
        state.outputs.len(),
        2,
        "both headless outputs must have reached the model"
    );

    let names: Vec<String> = state.outputs.values().map(|o| o.name.clone()).collect();
    let (name_a, name_b) = (names[0].clone(), names[1].clone());

    // Disable the first: allowed, because the second output survives it.
    state.output_configuration_applied(vec![applied_head(&name_a, false)]);
    assert_eq!(
        state.outputs.len(),
        1,
        "disabling one of two outputs is honored"
    );

    // Review finding #6: seed a SAVED config for the last output with a real
    // custom mode/scale/transform/position, so we can prove that a REFUSED
    // disable does not corrupt it. The refused head reports enabled=false with a
    // 0x0 mode; persisting that would flip the record to disabled and wipe the
    // saved mode -- which the next boot would then force-enable at preferred,
    // losing everything.
    state.config.displays.push(icedtea_config::DisplayConfig {
        name: name_b.clone(),
        enabled: true,
        width: 2560,
        height: 1440,
        refresh_mhz: 144_000,
        x: 100,
        y: 0,
        scale: 1.5,
        transform: 3,
    });

    // Disable the last remaining one, alone: the guard must refuse it so the
    // session is never left with zero active outputs.
    state.output_configuration_applied(vec![applied_head(&name_b, false)]);
    assert_eq!(
        state.outputs.len(),
        1,
        "the last active output must not be disabled -- >=1 output stays live"
    );
    assert!(
        state.outputs.keys().min().is_some(),
        "an active survivor output remains for placement"
    );

    // Review finding #6: the refused disable must NOT have corrupted name_b's
    // persisted entry. It stays enabled=true with its saved mode intact.
    let saved = state
        .config
        .displays
        .iter()
        .find(|d| d.name == name_b)
        .expect("the refused output's saved config entry must survive");
    assert!(
        saved.enabled,
        "a refused disable must keep the output enabled=true in persisted config"
    );
    assert_eq!(
        saved.width, 2560,
        "the saved mode width must not be wiped to 0 by a refused disable"
    );
    assert_eq!(
        saved.height, 1440,
        "the saved mode height must survive a refused disable"
    );
    assert_eq!(
        saved.refresh_mhz, 144_000,
        "the saved refresh must survive a refused disable"
    );
    assert_eq!(
        saved.scale, 1.5,
        "the saved scale must survive a refused disable"
    );
    assert_eq!(
        saved.transform, 3,
        "the saved transform must survive a refused disable"
    );
    assert_eq!(
        saved.x, 100,
        "the saved position must survive a refused disable"
    );

    let _ = std::fs::remove_file(&tmp);
}
