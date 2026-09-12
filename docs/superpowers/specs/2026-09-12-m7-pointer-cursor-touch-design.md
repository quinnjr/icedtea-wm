# M7 pointer/cursor/touch (batch) — design

Date: 2026-09-12. Program: icedtea M7 batch, approach A (M8-precedent full typing).
Governs both repos: `wlroots-sys` (wlr crate, release 0.20.34) and `icedtea`
(compositor + harness + shell). Prior art: M8 IME depth (wlr 0.20.32),
M8 remainder batch (wlr 0.20.33, single version, one R1 gate). This slice wraps
every remaining `milestone = "M7"` symbol (~100: cursor/xcursor, pointer depth,
pointer_constraints_v1, relative_pointer_v1, pointer_gestures_v1, touch, switch,
plus 7 seat-core rows) in one release, with full consumer wiring.

## 1. Background and motivation

M7's roadmap entry is "Input long tail I: pointer & cursor." The state machines
were never behind a flag and have lived as raw sys calls or not at all:

- Cursor attach/map/image (the visible pointer) has no safe surface.
- Pointer constraints (lock/confine), relative-pointer (FPS/drawing), and
  gestures (swipe/pinch/hold) have no typed send order.
- Touch grabs and switch toggles (lid-close) have no snapshot/event shape.

Batch carries all ~100 in one release and one icedtea branch. Rationale:
single version for a coherent pointer API; one whole-branch review; one
R1 gate. Cost: review is ~M8-remainder size and any consumer-found API
tweak re-freezes the whole batch (no publish until the harness proves
it, per R1). If it grows mid-implementation, upgrade the path formally
per the brainstorming ratchet.

Cursor-shape protocol work from A2 batch-2 stays as-is; this slice maps the
`cursor` shape names onto the new cursor image path rather than re-typing it.
M6-remainder and M9-M13 backlog order is unchanged.

## 2. Reframing: what gets typed

Batch applies the M8 precedent — type our side, not the client's protocol —
to the pointer state machines:

1. **Aggregate reads** as snapshot structs (cursor map/image, touch points,
   switch state, constraint state) — owned fields, null-guarded,
   `to_string_lossy` for C strings (blessed lossy per M8).
2. **Outgoing send order** as token-consuming builders where the crate
   drives the sequence (pointer frame, touch frame, gesture phase).
   Token widening is compile-time ordering; runtime liveness still
   resolves through tables and no-ops on stale, preserving FIX-3 guards.
3. **Live tracking** as id-only or handle-only events keeping
   `Event: Copy` where possible, defaulted hooks on the dedicated handler
   trait (additive, per f2cc8a9/A12-SEMVER).

Typing the client's incoming Wayland sequence is out of scope.

## 3. Snapshot and handle types (wlr)

Constructed only by the crate from live pointers (`pub(crate)`
constructors); fields `pub` for read. Owned strings, `Option` for nulls,
`Vec` for arrays — no raw pointers escape. Handles are `Copy` newtypes
over table keys (listener addresses or monotonic counters), same as
`InputPopupSurfaceId`.

| Type | Mirrors | Contents |
|---|---|---|
| `CursorState` | `wlr_cursor.{state, mapped}` | mapped output/region, image/buffer/surface attachment, hotspot, warp position |
| `TouchState` | `wlr_touch` + seat touch points | active point set, grab owner |
| `SwitchState` | `wlr_switch_state` | switch type, state bit, lid-close derived flag |
| `PointerConstraintState` | `wlr_pointer_constraint_v1_state` | surface, lifetime, region, constraint type |
| `XcursorTheme` / `XcursorManager` | `wlr_xcursor_theme/manager` | theme name/size, loaded cursor handles |

Readers: `Runtime::cursor_state()`, `touch_state()`, `switch_state()`,
`constraint_state_for_surface()` — each returns `Option` on missing
state, same miss shape as the M8 readers.

## 4. Token types (wlr)

Capability handles, `Clone`/`Copy` where `Runtime` allows, carrying
`Runtime` + keys. Compile-time order, runtime liveness.

- `PointerFrame` — produced only by motion/button paths; only its
  methods emit `send_motion`/`send_button`; `finish(self)` is the only
  path to `send_frame`.
- `TouchFrame` — produced on `touch_down`; only its methods emit
  `send_motion`/`send_up`; consumed on `send_frame` or `send_cancel`.
- `GesturePhase` — produced on swipe/pinch/hold `begin`; only its
  methods emit `update`; consumed on `end`/`cancel`.

Rewrites thread tokens through the existing seat/gesture handlers.
Wire behavior is byte-identical — the existing pointer/touch tests must
pass unchanged.

## 5. Events (wlr)

Additions in the A12 shape (defaulted no-ops, semver-additive):

- `Event::CursorMapped(CursorId)` / `CursorUnmapped(CursorId)`
- `Event::PointerConstraintCommitted(ConstraintId)`
- `Event::RelativeMotion(DeviceId, (f64, f64))`
- `Event::GestureBegan(GestureId)` / `GestureEnded(GestureId)`
- `Event::TouchDown(TouchId)` / `TouchUp(TouchId)` / `TouchCancelled`
- `Event::SwitchToggled(SwitchId, bool)`

Table lifetime uses `Registration` listeners; destroys emit via
listener-address or pending-drain, preserving FIX-3.

## 6. Compositor consumer (icedtea)

- **Cursor wiring** — attach via `wlr_cursor_attach_input_device`,
  map-to-output/region from the output layout, `set_surface`/`set_buffer`
  image path; `cursor_shape_v1` shape names map onto the image path (the
  toolkit already maps the `cursor` property to shape names).
- **Pointer protocols** — constraint lock/confine enforced at the
  pointer-focus chokepoint; relative motion forwarded to the focused
  surface; gestures emitted as scroll-like shell events.
- **Touch + switch** — touch grab forwarding with touch-to-pointer
  emulation reuse; switch toggle feeds the session/lock signal
  (`Event::SessionLockChanged` neighbor; coordinated contract bump if
  the snapshot grows).
- **Snapshot** — additive-optional `cursor_visible: bool`,
  `cursor_pos: Option<(i32, i32)>`, `touch_active: bool` with
  `serde(default)`; shell updated in lockstep on the same branch.
- **No new surface type, no touch UI.** Gestures/switches are signals,
  not widgets.

## 7. Testing (R1 holds: publish only after downstream proof)

Harness doubles record each payload with serials (pointer frame serials,
touch point frames, gesture phase triples, constraint commits, switch
toggles) in the record-everything style. New e2e (both sides, real clients):

1. Cursor warp reaches the expected surface with the mapped image.
2. Touch down/motion/up reaches the focused surface; cancel clears.
3. Pinch/swipe begin→update→end forwarded with intact phases.
4. Constraint confines/locks pointer while active, releases after.
5. Switch toggle flips the session signal.

Token order is compile-time; proof is that the relay tests pass through
the token APIs plus the grep audit that no raw `send_*` remains outside
token impls. The adversarial wire-bytes e2e gate (M8 `f5e0a95` mandate)
runs before publish.

## 8. Rollout

- SDD task split — wlr: cursor depth → pointer protocols → touch/switch →
  release 0.20.34; icedtea: wiring → cursor/touch → gestures/switch → e2e.
  Whole-branch review before merge (A6.2/M8 precedent). Single version
  0.20.34 (batch).
- Semver: additive only (defaulted methods, new types, no supertrait
  changes) per A12-SEMVER; patch release under the frozen-minor rule.
- Out of scope: re-wrapping libinput/xkbcommon (foreign-library), M6
  remainder / M9-M13, client incoming-sequence typing.

## 9. Open questions (resolved during brainstorming)

- Scope: full — wlr + icedtea consumer with R1 downstream proof.
- Shape: single batch (one version, one review, one R1 gate).
- Consumer: full wiring — cursor attach/map, touch + gestures + switch
  plumbed through, not harness-only.
