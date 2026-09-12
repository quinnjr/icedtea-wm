# icedtea wlr 0.20.34 wire-up — design

Date: 2026-09-12. Scope decision: bump + wire + prove. M8 remainder
features are explicitly out of scope. Two PRs: wire (PR1), then prove
(PR2). Path: architectural (multi-release span), executed as
bounded-style slices with per-PR review.

## Context

icedtea pins `wlr` 0.20.31 (`compositor/Cargo.toml`, `Cargo.lock`).
Upstream shipped 0.20.32 (IME depth), 0.20.33 (M8 remainder), 0.20.34
(M6 rest: swapchain, output-signal dispatch, output protocols, plus
retro fixes). wlr's within-minor policy is additive-only, so the bump
itself should compile clean; the work is wiring what is new.

`State: wlr::OutputHandler` (`compositor/src/state.rs:5693`) currently
implements `new_output`, `frame`, `destroyed`,
`output_configuration_applied`, `gamma_control_changed` (+ others).
New since 0.20.31: `output_committed`, `output_precommitted`,
`output_damaged`, `output_bound`, `output_state_requested`,
`output_power_mode_requested`. New globals: `create_tearing_control_manager`,
`create_power_manager`. Available but unwired: `SwapchainManager`,
`send_request_state`, `primary_formats`.

Manager setup lives in `compositor/src/lib.rs` (~line 340): non-fatal
tone (`tracing::error!` + continue). `create_presentation(display,
backend)` signature is unchanged upstream — no call-site churn.

## Key findings shaping the design

1. `frame()` commits the scene **unconditionally** every frame. Gating
   commits on damage would restructure the render loop — out of scope.
   Damage handling is therefore observe-and-record, not repaint
   scheduling.
2. `destroyed()` already tracks a `disabled_outputs` set: power-Off has
   a natural home.
3. `output_state_requested` carries only the fields mask, not values —
   literal "apply" is impossible from it. Full apply lives in the
   already-wired output-management flow (`output_configuration_applied`).
4. `SwapchainManager` needs M13 `prepare` for its repaint loop;
   `send_request_state` is an emit-side test hook; `primary_formats`
   feeds future dmabuf work. None get icedtea callers in PR1.

## PR1: bump + wire

- Bump `wlr` 0.20.31 → 0.20.34 (`compositor/Cargo.toml` + lockfile).
  Existing suite green proves the jump.
- `output_committed` / `output_precommitted`: record last mask +
  timestamp per output; on a `MODE` commit icedtea did not initiate,
  re-derive that output's geometry through the existing layout path.
- `output_damaged`: record last damage per output + debug log.
- `output_bound`: info log (bind already happened; nothing to refuse).
- `output_state_requested`: info log + record (see finding 3).
- `output_power_mode_requested`: `Off` → add to `disabled_outputs` +
  commit disabled state; `On` → remove + re-enable. Disable mechanics
  to be verified against the existing set during implementation.
- Boot block: add both manager creates beside the existing managers,
  same non-fatal tone.
- Work setup: `.worktrees/wlr-034-wireup`, branch
  `feature/wlr-0.20.34-wireup` (ticketless, per repo convention).

## PR2: prove

Harness round trips (`harness/`, existing `end_to_end.rs` /
`headless_boot.rs` patterns):
- Commit/precommit order + staged masks.
- Damage delivery followed by a frame commit.
- Request-state round trip driven by `send_request_state` (no protocol
  client needed).
- Power `Off` → output disabled, `On` → re-enabled.

## Success criteria

- PR1: `cargo test` (workspace) green on 0.20.34, no behavior change
  beyond the six methods + two globals above.
- PR2: new round-trip tests green headless; they fail if any wired
  path is gutted (deletion-tested).
- No render-loop, scene-graph, or protocol-semantics changes in either PR.
