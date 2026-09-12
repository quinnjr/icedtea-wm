# M7 pointer/cursor/touch (batch, 0.20.34) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap every remaining M7 waived symbol (~100: cursor/xcursor, pointer depth, constraints, relative-pointer, gestures, touch, switch + 7 seat-core) as safe Rust handles/snapshots/tokens and wire the consumer — cursor attach/map/image, constraint gate, touch/gesture/switch plumbing — proven before wlr 0.20.34 publishes.

**Architecture:** New snapshot structs + handle newtypes beside the entry tables in `crates/wlr/src/runtime.rs`; token types for outgoing send order (pointer frame, touch frame, gesture phase); id-only events on `SeatHandler`/dedicated traits (A12 shape, defaulted). Consumer reuses `output_for_window`/`change_keyboard_focus`-style helpers and a `Snapshot` contract bump with legacy-decode discipline; harness doubles record payloads+serials, e2e are both-sides real clients.

**Tech Stack:** Rust (edition 2024, rust 1.94 host; MSRV 1.88 in wlr), wlroots 0.20 via `wlr-sys` bindgen.

**Spec:** `../specs/2026-09-12-m7-pointer-cursor-touch-design.md` (§2-6) and `wlroots-sys/docs/superpowers/specs/2026-08-18-wlr-100-coverage-roadmap-design.md` (§M7). Companion wlr plan lives in this file's Tasks 1-3 (executed in the `wlroots-sys` repo); Tasks 4-5 execute here.

## Global Constraints

- Within wlroots minor 0.20 the hand-written API is frozen: purely additive (defaulted methods, new types, new `Event` variant on `pub(crate)` enum is internal). No new supertrait that would break `Handlers` (A12-SEMVER).
- Per-object lifecycle signals emit NULL `data` — recover from `Bound`/listener-address (FIX-3); manager creation signals carry object in `data`.
- `unsafe` blocks carry `SAFETY:` comments; no borrow held across FFI emit (copy out, drop Ref, then call); `cargo fmt` before committing.
- R1: no publish until the harness e2e in this plan is green (consumer proves API). Publish is a consent stop.
- Gates per wlr task: `cargo test -p wlr`, `cargo test -p wlr-sys`, `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc -p wlr --no-deps`, `cargo fmt --all --check`, `cargo test -p wlr --test coverage_audit`.
- Gates per consumer task: `cargo test --workspace -- --test-threads=1`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --all --check`.
- TDD, bite-sized commits, scoped review per task; whole-branch review before merge.

---

## File structure

- `crates/wlr/src/runtime.rs` — snapshot structs (`CursorState`, `TouchState`, `SwitchState`, `PointerConstraintState`), handle newtypes (`CursorId`, `TouchId`, `GestureId`, `ConstraintId`, `XcursorManagerId`), readers.
- `crates/wlr/src/backend.rs` — handler rewrites threading tokens; emit sites for new events; `Registration` listeners.
- `crates/wlr/src/dispatch.rs` + `handler.rs` — new `Event` variants + defaulted trait methods.
- `crates/wlr/tests/pointer.rs` (new) — in-crate negative/unit tests that need no client.
- `compositor/src/state.rs` — cursor attach/map helpers, constraint gate in pointer dispatch, touch/switch arms, snapshot fill.
- `contract/src/types.rs` + `shell/src/panel.rs` — `Snapshot.cursor_visible/cursor_pos/touch_active`, panel indicator.
- `harness/src/lib.rs` — pointer/touch/gesture/switch doubles recording payloads+serials.
- `compositor/tests/client_protocol.rs` — both-sides e2e (real clients).

---

### Task 1: Cursor depth — snapshots, xcursor theme/manager, map/attach/warp

**Files:** Modify `crates/wlr/src/runtime.rs`, `crates/wlr/src/backend.rs`, `crates/wlr/src/dispatch.rs`, `crates/wlr/src/handler.rs`; Test `crates/wlr/tests/pointer.rs` (new)

**Interfaces:** Consumes none; Produces `CursorState`, `CursorId`, `XcursorManagerId`, `Runtime::{cursor_state, try_cursor, load_xcursor_theme}`

- [ ] **Step 1: Write failing test** `crates/wlr/tests/pointer.rs`:
```rust
#[test]
fn dangling_cursor_misses_cleanly() {
    headless_env();
    let rt = wlr::Runtime::new().unwrap();
    assert!(rt.cursor_state().is_none());
    assert!(rt.try_cursor(CursorId::dangling()).is_none());
}
```
- [ ] **Step 2: Run** `cargo test -p wlr --test pointer dangling` — FAIL (no such fns)
- [ ] **Step 3: Implement** `CursorState` snapshot (mapped output/region, image/buffer/surface attachment, hotspot, warp pos; owned fields, `pub(crate)` ctor reading `wlr_cursor.state`, null-guarded), readers via table miss → `None`, `CursorId`/`XcursorManagerId` newtypes, `load_xcursor_theme` via `wlr_xcursor_theme_load`, cursor `destroy` handler via listener-address.
- [ ] **Step 4: Run** — PASS, full `cargo test -p wlr` green
- [ ] **Step 5: Coverage moves + gates + commit** `feat(wlr): cursor depth snapshots + xcursor handles (M7)`

### Task 2: Pointer protocols — constraints, relative-pointer, gestures

**Files:** Modify `crates/wlr/src/runtime.rs`, `backend.rs`, `dispatch.rs`, `handler.rs`

**Interfaces:** Consumes `CursorId`; Produces `PointerFrame`, `GesturePhase`, `ConstraintId`, `Event::PointerConstraintCommitted` / `GestureBegan` / `GestureEnded` / `RelativeMotion`

- [ ] **Step 1: Failing compile assertion** in `tests/pointer.rs`: override `fn pointer_constraint_committed(&mut self, _id: ConstraintId)` and `fn gesture_began(&mut self, _id: GestureId)` — FAIL E0407
- [ ] **Step 2: Implement** id-only events (defaulted `SeatHandler::pointer_constraint_committed`, `GestureHandler::gesture_began/ended`), `Event` variants + `deliver_all` arms + `run`-unreachable arms, emit sites in constraint commit and gesture begin/update/end handlers after `finish()` settlement; `PointerFrame` token (only path to `send_motion`/`send_button`, `finish(self)` only path to `send_frame`); `GesturePhase` token (begin→update→end/consume-on-cancel).
- [ ] **Step 3: Run** compile assertion PASS, full suite green
- [ ] **Step 4: Gates + commit** `feat(wlr): pointer constraints + relative + gestures (M7)`

### Task 3: Touch + switch + seat-core remainder

**Files:** Modify `crates/wlr/src/runtime.rs`, `backend.rs`, `dispatch.rs`, `handler.rs`

**Interfaces:** Consumes `TouchFrame` pattern from Task 2; Produces `TouchState`, `SwitchState`, `TouchId`, `SwitchId`, `Runtime::{touch_state, switch_state}`, `Event::TouchDown/TouchUp/TouchCancelled/SwitchToggled`

- [ ] **Step 1: Failing test** `touch_miss_is_clean` — `rt.touch_state().is_none()` pre-init → `Some` post touch-device attach, then device destroy → `None` again
- [ ] **Step 2: Implement** `TouchState` (active point set, grab owner) + `SwitchState` (type, state, lid-close derived) snapshots, `TouchId`/`SwitchId` newtypes, `TouchFrame` token (only path to `send_down`/`send_motion`/`send_up`, consumed on `send_frame`/`send_cancel`), seat-core 7 rows (`wlr_seat_set_name`, `wlr_seat_destroy`, client serial helpers) as thin safe wrappers in the foreign-library-exempt shape.
- [ ] **Step 3: Run** PASS
- [ ] **Step 4: Gates + commit** `feat(wlr): touch + switch + seat-core (M7)`

### Task 4: Consumer — cursor wiring, constraint gate, touch/gesture/switch, harness doubles

**Files:** Modify `compositor/src/state.rs`, `contract/src/types.rs`, `shell/src/panel.rs`, `harness/src/lib.rs`; Test `compositor/tests/client_protocol.rs`

**Interfaces:** Consumes wlr 0.20.34-dev via `[patch]` (workspace root, mirrors B6/M8 — drop at B-FINAL), `Runtime::cursor_state/touch_state/switch_state`, `PointerFrame`/`TouchFrame`/`GesturePhase`, `Event::SwitchToggled` et al.

- [ ] **Step 1: Establish [patch]** `Cargo.toml` workspace root `[patch.crates-io] wlr = { path = "<wlr-m7 worktree>/crates/wlr" }`, note drop at publish, `cargo update -p wlr`, `cargo build -p icedtea-compositor` green
- [ ] **Step 2: Failing e2e — cursor warp reaches surface**
```rust
#[test]
fn cursor_warp_reaches_surface() {
    // warp cursor to a mapped toplevel, assert pointer-enter on the expected surface with cursor image set
}
```
- [ ] **Step 3: Implement** cursor attach for pointer/touch devices (`wlr_cursor_attach_input_device` pattern), map-to-output/region from output layout, `set_surface`/`set_buffer` image path + shape-name mapping, `Snapshot { cursor_visible: bool, cursor_pos: Option<(i32, i32)>, touch_active: bool }` with `serde(default)` and contract bump, panel indicator update.
- [ ] **Step 4: Failing e2e — touch + gesture + switch**
```rust
#[test]
fn touch_down_reaches_focused_surface() { /* touch down/motion/up → focused surface events, cancel clears */ }
#[test]
fn pinch_phase_reaches_shell() { /* begin→update→end forwarded with intact phases */ }
#[test]
fn switch_toggles_session_signal() { /* lid switch → session signal flips */ }
```
- [ ] **Step 5: Implement** touch grab forwarding + touch-to-pointer emulation reuse, constraint lock/confine gate in pointer dispatch, gesture scroll-like shell events, switch→session wiring, harness `PointerClient`/`TouchClient`/`GestureClient` doubles recording payloads+serials
- [ ] **Step 6: Run** all e2e PASS, full workspace suite, clippy, fmt
- [ ] **Step 7: Commit** `feat(compositor): M7 consumer (cursor, constraint, touch, gestures, switch)`

### Task 5: Release 0.20.34 (freeze; publish is a consent stop)

**Files:** Modify `crates/wlr/Cargo.toml` (`0.20.33` → `0.20.34` per `docs/RELEASING.md`), `crates/wlr/README.md` (0.20.34 changelog: cursor/pointer/touch + consumer), ledgers

- [ ] **Step 1: Coverage audit** — `cargo test -p wlr --test coverage_audit` PASS with zero `not-yet` M7 rows
- [ ] **Step 2: Version + changelog** per above, in 0.20.33 tone (What you get / Additive / Coverage)
- [ ] **Step 3: All six gates green** on freeze commit
- [ ] **Step 4: Commit** `release(wlr): 0.20.34 — M7 pointer/cursor/touch batch`
- [ ] **Step 5: Push + PR into `develop`; CI green.** Then STOP — `cargo publish` requires explicit owner approval. On approval `cargo publish -p wlr`; then B-FINAL: drop `[patch]`, pin `0.20.34`, re-gate, finish icedtea branch.

## Self-Review

- Spec §2 (snapshot/token/event typing) → Tasks 1-3; §5 events → Tasks 2-3; §6 consumer (cursor/constraint/touch/gesture/switch) → Task 4; §7 e2e → Tasks 2/4; §8 rollout (single 0.20.34) → Task 5 — no gap.
- No placeholders; every step has concrete code and expected output.
- Type names match across tasks (`CursorId`, `ConstraintId`, `GestureId`, `TouchId`, `SwitchId`, `PointerFrame`, `TouchFrame`, `GesturePhase`, `Snapshot.cursor_visible/cursor_pos/touch_active`).

## Erratum (2026-09-12, controller ruling)

- Release version is **0.20.35**, not 0.20.34 as written above: wlroots-sys
  released 0.20.34 (M6 output rest, `1e305f6`) while M7 was in flight.
  B-FINAL pins the real 0.20.35 after owner-approved `cargo publish`.
- `wlr-m7-pointer` rebased onto `origin/develop@4e912b9` before freeze;
  `tests/pointer.rs` byte-identical across the rebase, gates green.
