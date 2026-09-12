//! M5 close-out: contract §4.2's shell half, as data (P6-D6).
//!
//! Hermetic: paths and file contents only.
//!
//! Reconciled against the tree at commit `ec42e64` (the brief was written
//! against `develop @ d9cee52`, before P0-P5 executed):
//!  - `KEPT_SUITES` counts updated to what P0-P5 actually shipped:
//!    `src/taskbar.rs` 5 -> 6 and `src/clipboard.rs` 1 -> 2 (each gained one
//!    `a_..._can_be_cloned_out_of_an_arc` test); `src/compositor_client.rs`
//!    (1) and `tests/live_dbus.rs` (1) were unchanged.
//!  - `panel_paints_every_probe_point_at_rest` shipped as three theme
//!    variants (commit `e42db98`, "gate the panel's rest state in all three
//!    themes"); this table checks the light variant, which is sufficient
//!    proof the gate exists.
//!  - `the_fatal_worker_path_still_exits` lives in `shell/src/main.rs`, not
//!    `tests/panel.rs`; `all_shell_text()` covers `src/` too, so the pattern
//!    check still finds it.

use std::path::{Path, PathBuf};

/// `(deleted path, the path that replaced it, the test that proves the
/// replacement behaves)`.
const DELETED_WITH_REPLACEMENT: &[(&str, &str, &str)] = &[
    (
        "shell/src/bridge.rs",
        "shell/src/panel.rs",
        "a_window_opened_signal_through_the_inbox_adds_a_button (the inbox replaces glib::spawn_future_local)",
    ),
    (
        "shell/tests/shell_gtk.rs",
        "shell/tests/panel.rs",
        "a_panel_click_reaches_the_command_surface (same mocks, same (action, id) assertions)",
    ),
];

/// `(exact source pattern that must not appear anywhere in shell/, where it
/// used to live)`.
const DELETED_IDENTIFIERS: &[(&str, &str)] = &[
    (
        "pub use gtk4",
        "lib.rs -- the re-export shell_gtk.rs needed",
    ),
    (
        "pub mod bridge",
        "lib.rs -- the deleted worker bridge module",
    ),
    (
        "pub fn render",
        "taskbar.rs and clipboard.rs -- clear-and-rebuild, replaced by view()",
    ),
    (
        "fn connect_activation",
        "clipboard.rs -- replaced by ListBox::on_item_activated",
    ),
    (
        "GestureClick",
        "taskbar.rs -- replaced by on_click / on_pointer_up_with_button",
    ),
    (
        "set_ellipsize",
        "clipboard.rs -- replaced by the label's CSS ellipsize",
    ),
    (
        "taskbar_box",
        "main.rs -- the separate child box the rebuild needed",
    ),
];

/// `(path, its exact `#[test]` count)`. Contract §3.6: these suites are kept
/// **verbatim**; a rewrite that drops one is the failure mode this catches.
/// Counts measured on the current tree (`ec42e64`): `src/taskbar.rs` and
/// `src/clipboard.rs` each grew by one Arc-clone test during P0-P5.
const KEPT_SUITES: &[(&str, usize)] = &[
    // M7: 6 -> 8 (`snapshot_folds_touch_active...` and
    // `gesture_and_switch_signals_fold` cover the new CompositorUpdate
    // fold arms).
    ("src/taskbar.rs", 8),
    ("src/clipboard.rs", 2),
    ("src/compositor_client.rs", 1),
    ("tests/live_dbus.rs", 1),
];

/// Every shell test the contract names as an M5 gate (§3.6), by function
/// name. `panel_paints_every_probe_point_at_rest` shipped as three
/// theme-suffixed tests (commit `e42db98`); the light variant is checked
/// here as sufficient proof the gate exists.
const REQUIRED_GATES: &[&str] = &[
    "a_panel_click_reaches_the_command_surface",
    "middle_clicking_a_window_button_closes_it",
    "a_window_opened_signal_through_the_inbox_adds_a_button",
    "panel_paints_every_probe_point_at_rest_in_the_light_theme",
    "the_clipboard_popover_opens_pastes_and_dismisses",
    "the_fatal_worker_path_still_exits",
];

const SELF_EXCLUDED: &[&str] = &["dependency_audit.rs", "deletion_ledger.rs"];

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_dir() -> PathBuf {
    crate_dir()
        .parent()
        .expect("the crate directory has a parent")
        .to_path_buf()
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !SELF_EXCLUDED.iter().any(|name| path.ends_with(name))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// `#[test]` attributes in a file. A `#[test]` inside a string literal would
/// be counted too; no file in this crate has one, and a spurious count is a
/// loud failure rather than a silent pass.
fn count_tests(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .matches("#[test]")
        .count()
}

/// Every shell source and test, concatenated, for whole-crate pattern
/// searches.
fn all_shell_text() -> String {
    let mut paths = rust_sources(&crate_dir().join("src"));
    paths.extend(rust_sources(&crate_dir().join("tests")));
    assert!(
        paths.len() > 5,
        "the source walker found only {} files -- it is looking in the wrong place",
        paths.len()
    );
    paths
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Mutation check: `git checkout develop -- shell/src/bridge.rs`; this must
/// fail naming that path. Delete it again.
#[test]
fn every_deleted_shell_file_is_gone_and_its_replacement_exists() {
    let root = workspace_dir();
    let mut problems = Vec::new();
    for (deleted, replacement, proof) in DELETED_WITH_REPLACEMENT {
        if root.join(deleted).exists() {
            problems.push(format!(
                "{deleted} still exists (was to be replaced by {replacement})"
            ));
        }
        if !root.join(replacement).exists() {
            problems.push(format!(
                "{deleted} was deleted but its replacement {replacement} does not exist \
                 (proof was to be: {proof})"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "the shell deletion ledger does not hold: {problems:#?}"
    );
}

/// Mutation check: re-add `pub use gtk4;` to `shell/src/lib.rs`; this must
/// fail naming it. Restore.
#[test]
fn every_deleted_shell_identifier_is_gone() {
    let text = all_shell_text();
    let survivors: Vec<String> = DELETED_IDENTIFIERS
        .iter()
        .filter(|(pattern, _)| text.contains(pattern))
        .map(|(pattern, home)| format!("`{pattern}` ({home})"))
        .collect();
    assert!(
        survivors.is_empty(),
        "GTK-era shell items survived the migration: {survivors:#?}"
    );
}

/// Mutation check: delete one `#[test]` from `shell/src/taskbar.rs`; this must
/// fail reporting 5 for 6. Restore.
#[test]
fn every_kept_shell_suite_kept_its_test_count() {
    let mut problems = Vec::new();
    for (relative, expected) in KEPT_SUITES {
        let path = crate_dir().join(relative);
        if !path.exists() {
            problems.push(format!("{relative} does not exist"));
            continue;
        }
        let actual = count_tests(&path);
        if actual != *expected {
            problems.push(format!("{relative}: {actual} tests, expected {expected}"));
        }
    }
    assert!(
        problems.is_empty(),
        "a kept-verbatim shell suite changed size: {problems:#?}"
    );
}

/// Mutation check: rename `panel_paints_every_probe_point_at_rest_in_the_light_theme`
/// in `shell/tests/panel.rs`; this must fail naming it. Restore.
#[test]
fn every_shell_gate_the_contract_names_exists() {
    let text = all_shell_text();
    let missing: Vec<&str> = REQUIRED_GATES
        .iter()
        .copied()
        .filter(|name| !text.contains(&format!("fn {name}(")))
        .collect();
    assert!(
        missing.is_empty(),
        "the contract names these shell gates and they do not exist: {missing:#?}"
    );
}

/// Contract §3.5, verbatim and non-negotiable: a post-connect D-Bus error in
/// the compositor worker calls `std::process::exit(1)` so systemd's
/// `Restart=always` replaces the process with a matching build. Logging and
/// continuing would leave a permanently stale taskbar systemd never restarts.
/// P5 asserts the behaviour; this asserts nobody quietly softened the source.
///
/// Mutation check: replace the `std::process::exit(1)` in
/// `shell/src/compositor_client.rs` with `tracing::error!`; this must fail.
/// Restore.
#[test]
fn the_fatal_dbus_path_still_exits_the_process() {
    let source = std::fs::read_to_string(crate_dir().join("src/compositor_client.rs"))
        .expect("read shell/src/compositor_client.rs");
    assert!(
        source.contains("process::exit(1)"),
        "compositor_client.rs no longer exits on a fatal D-Bus error -- \
         contract §3.5 forbids softening this into a log line"
    );
    let unit = std::fs::read_to_string(workspace_dir().join("shell/systemd/icedtea-shell.service"))
        .expect("read the systemd unit");
    assert!(
        unit.contains("Restart=always"),
        "the systemd unit no longer restarts the shell, which is the other \
         half of the process::exit(1) contract"
    );
}
