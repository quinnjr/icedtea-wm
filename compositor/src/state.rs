//! The compositor's own state: `State` owns the window model
//! (`WindowManager`) and configuration, and fans out `contract::Event`s
//! produced by model mutations onto a crossbeam channel for the D-Bus side
//! to consume.
//!
//! This is the model-only shape of `State`, mid-port (wlr-port milestone 1,
//! task 4): every push out to a client goes through `wayland: Wayland`
//! (`wayland.rs`) rather than through smithay's `Space`/`ToplevelSurface`
//! types directly, so this file has no compositor-library dependency at
//! all. `Wayland`'s methods are no-ops until task 5 backs them with `wlr`;
//! until then every mutation here behaves exactly as it does in any test
//! that never binds a toplevel.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use icedtea_config::Config;
use icedtea_contract::{AltTabState, Event, Rectangle, SeqEvent, WindowId};
// `SeatHandler` for the M7 test-hook arms in `handle_command`, which drive
// the real touch/gesture notifications (`touch_cancelled`,
// `gesture_began/ended`) on the loop thread.
use wlr::SeatHandler;

use crate::input;
use crate::layout::{self, SnapZone};
use crate::render::{self, WallpaperState};
use crate::window::WindowManager;

/// The model's placeholder toplevel size: staged as both the new window's
/// frame geometry and the client's first configure, until the client's own
/// first commit says otherwise. One constant rather than the literal
/// `(640, 400)` repeated at each site (`new_toplevel`'s cascade placement,
/// `initial_commit`'s configure, and the test that asserts on it) so the
/// three/four call sites cannot drift apart from each other.
pub const PLACEHOLDER_SIZE: (i32, i32) = (640, 400);

/// Sentinel [`LayerEntry::output`] value meaning "no output existed at
/// all when this surface was announced" (review finding M5). `u32::MAX`
/// rather than `0`: `next_output_index` starts at `0` and only
/// increments, so `0` is a real, common output index the very first
/// hotplugged output takes, while reaching `u32::MAX` real outputs is not
/// a case this process will ever see -- the same "far outside the real
/// id space" reasoning [`wlr::LayerSurfaceId::dangling_for_test`] uses
/// for its own sentinel.
const NO_OUTPUT: u32 = u32::MAX;

/// Output geometry information (simplified from smithay's `Output`).
pub struct OutputSurface {
    pub geometry: icedtea_contract::Rectangle,
    /// `geometry` shrunk by any exclusive-zone layer surfaces anchored to
    /// this output's edges. Equal to `geometry` when none reserve space.
    /// Maintained solely by [`State::arrange_layers`]; every placement
    /// consumer that means "the space windows may occupy" (tiling,
    /// maximize) reads this instead of `geometry` -- fullscreen is the one
    /// exception, since it covers panels by definition.
    pub usable: icedtea_contract::Rectangle,
    /// The output's connector name (`wlr::Output::name()`), the stable key a
    /// persisted [`icedtea_contract::DisplayConfig`] is matched against on
    /// connect/hotplug and the reverse lookup
    /// `OutputHandler::output_configuration_applied` uses to map an
    /// [`wlr::AppliedHead`]'s name back to an output index. Empty for the
    /// unit-test outputs built straight through [`Self::new`], and for the
    /// rare wlroots output not yet named.
    pub name: String,
    /// The output's committed scale (`wlr::Output::set_scale`), mirrored here
    /// because wlroots exposes no getter and the value is needed off the live
    /// handler — specifically by [`State::primary_output_scale`], which the M4
    /// HiDPI DPI-hint export reads to pick `Xft.dpi = 96 * integer_scale`.
    /// Defaults to `1.0` for the unit-test outputs built through [`Self::new`];
    /// `new_output` records the real applied scale on a live output.
    pub scale: f64,
}

impl OutputSurface {
    /// Construct with no exclusive zones yet reserved -- `usable` starts
    /// equal to `geometry`, exactly as `create_output` documents. The name is
    /// empty; `new_output` records the real connector name on the live output
    /// once one exists (see the field doc). Scale starts at the `1.0` identity.
    pub fn new(geometry: icedtea_contract::Rectangle) -> Self {
        Self {
            geometry,
            usable: geometry,
            name: String::new(),
            scale: 1.0,
        }
    }
}

/// The model's own record of a wlr-layer-shell surface: everything
/// [`State::arrange_layers`] and [`State::configure_layer`] need, kept in
/// the model rather than re-read from the library's `LayerSurface` handle on
/// every arrangement pass -- the handle only lives for the duration of one
/// handler call (see [`wlr::LayerSurface`]'s own doc), so anything an
/// unrelated later call (a different surface's commit, a hotplug) needs has
/// to be copied out while the handle is live. Updated wholesale on every
/// `layer_surface_commit`, since wlr-layer-shell clients routinely re-anchor
/// or resize their exclusive zone after mapping.
pub struct LayerEntry {
    /// The model output index (`State::outputs`' key) this surface is
    /// placed on. Resolved once, at `new_layer_surface`, from
    /// [`wlr::LayerSurface::output_id`] via `output_ids`, falling back to
    /// `output_for_pointer` -- see `new_layer_surface`'s own doc. May be
    /// `NO_OUTPUT`, a sentinel meaning "no output existed at all when
    /// this surface was announced" (review finding M5); every reader that
    /// looks it up in `self.outputs` already treats a miss as "nothing to
    /// do yet", which is exactly right for the sentinel too, and
    /// `State::resolve_orphaned_layers` re-homes it the moment any output
    /// exists.
    pub output: u32,
    /// This entry's position in a total, stable order across every layer
    /// surface this compositor has ever announced -- assigned once, at
    /// `new_layer_surface`, from `State::next_layer_sequence`.
    ///
    /// Exists only for [`State::configure_layer`]'s N6 fix: two
    /// same-edge, same-output panels must not draw on top of each other,
    /// which needs a deterministic placement order, and
    /// [`wlr::LayerSurfaceId`] has no such order exposed (`Hash`/`Eq`
    /// only, and its inner value is `pub(crate)` to the `wlr` crate, not
    /// this one) -- so this crate mints its own rather than one it cannot
    /// read.
    pub sequence: u64,
    /// Carried for the model's own record; not consumed by
    /// [`State::arrange_layers`]/[`State::configure_layer`] -- the crate
    /// itself reparents this surface's scene node into the right band
    /// whenever a commit reports a different layer
    /// (`Runtime::reparent_layer_surface_if_changed`), so nothing here has
    /// to act on a change to it.
    pub layer: wlr::Layer,
    pub anchor: wlr::Anchor,
    /// The raw value from [`wlr::LayerSurface::exclusive_zone`]: `0` or
    /// negative means "reserve nothing" (any negative value additionally
    /// asks not to be moved to avoid occlusion, which this compositor's
    /// manual placement has no use for); only a positive value reserves
    /// space in [`State::arrange_layers`].
    pub exclusive: i32,
    /// The client's last-requested size (`0` on either axis means "the
    /// compositor decides" for that axis) -- the input `configure_layer`'s
    /// placement rule reads, not the size that method actually chose.
    pub size: (u32, u32),
    /// Whether this surface currently wants keyboard focus
    /// ([`wlr::LayerSurface::keyboard_interactive`]). `false` until the
    /// surface's first commit populates it (see that accessor's own
    /// timing doc) -- `new_layer_surface` always inserts `false` here.
    pub interactive: bool,
    /// Whether this surface is currently mapped (has a buffer and is on
    /// screen). `false` at `new_layer_surface` (review finding J2: a
    /// surface that never mapped, or that unmapped and is waiting to
    /// remap, must not reserve space); flipped by
    /// `layer_surface_mapped`/`layer_surface_unmapped`.
    /// [`State::arrange_layers`]'s fold skips every `!mapped` entry.
    pub mapped: bool,
    /// The `(width, height, x, y)` [`State::configure_layer`] most
    /// recently computed for this surface, or `None` before its first
    /// call. Set just before the runtime-gated
    /// `configure_layer_surface`/`set_layer_surface_position` calls, not
    /// only after they run, so this is testable without a live
    /// `wlr::Runtime` attached (every unit test in this file) -- in
    /// production a runtime is always attached by the time any handler
    /// runs, so this never records a placement that was not actually put
    /// on the wire. Re-review finding Minor-1's storm guard: `configure_layer`
    /// compares its freshly computed placement against this before doing
    /// anything else, so calling it again with an unchanged input --
    /// which `arrange_layers` now does for every mapped panel on every
    /// pass -- costs one comparison rather than a wire round trip.
    pub last_configured: Option<(u32, u32, i32, i32)>,
    /// The client's requested margin, `(top, right, bottom, left)` --
    /// wlr-layer-shell's own field order. Always `(0, 0, 0, 0)` for now:
    /// task 7 (N8) added this field so the rest of the placement plumbing
    /// has somewhere to read a margin from, but `wlr` 0.20.12's
    /// `LayerSurface` exposes no accessor to actually capture the
    /// client's requested value (`anchor`/`exclusive_zone`/`desired_size`/
    /// `keyboard_interactive` all exist; `margin` does not) -- verified
    /// against both 0.20.11 and 0.20.12's `layer.rs`. Capturing a real
    /// value in `layer_surface_commit` and insetting placement/exclusive
    /// folds by it is deferred until a future `wlr` release adds the
    /// accessor (tracked as a 0.20.13 additive gap); nothing here may
    /// synthesize a margin from anything else in the meantime.
    pub margin: (i32, i32, i32, i32),
}

/// What a popup hangs off, in model terms.
///
/// The library's own `wlr::PopupParent` says the same thing in library terms;
/// this is its model translation, made once in `new_popup` so nothing past
/// the handler boundary has to resolve a `ToplevelId` again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PopupHost {
    Window(WindowId),
    Layer(wlr::LayerSurfaceId),
    Popup(crate::wayland::PopupKey),
}

/// The bottom of a popup chain -- never a popup.
///
/// Every placement, focus and dismissal decision is taken against this rather
/// than against the immediate host: a menu three levels deep is still
/// constrained to *its window's* output, and focus still returns to *its
/// window* when the chain ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PopupRoot {
    Window(WindowId),
    Layer(wlr::LayerSurfaceId),
}

/// The model's own record of one live xdg-popup.
///
/// Same reasoning as [`LayerEntry`]: the library's `wlr::Popup` handle only
/// lives for the duration of one handler call, so everything a later,
/// unrelated call needs (a window move re-running placement, a click looking
/// for the topmost popup, a destroy deciding whether the chain has emptied)
/// is copied out while the handle is live.
pub struct PopupEntry {
    /// The immediate parent -- a window, a layer surface, or another popup.
    pub host: PopupHost,
    /// The chain's bottom, resolved once at record time. Never re-derived:
    /// the host may die before this popup does.
    pub root: PopupRoot,
    /// The model output index the root sits on, or `NO_OUTPUT` when the
    /// root had no output at record time -- exactly [`LayerEntry::output`]'s
    /// convention, and every reader treats a miss the same way.
    pub output: u32,
    /// This popup's position in a total, stable creation order across every
    /// popup this compositor has ever announced. The z-order tiebreak among
    /// siblings, and the sort key `popup_chain` orders by.
    pub sequence: u64,
    /// The client sent `xdg_popup.grab`. Observed, never driven: wlroots owns
    /// the grab's whole lifetime (contract §1.6).
    pub grabbing: bool,
    /// Whether no other popup on this `root` was recorded yet when this one
    /// was. Captured once, at record time, for `reconcile_popup_grab` to
    /// finish the "only the chain's first popup parks a restore target"
    /// decision `record_popup` starts -- see that field's use there for why
    /// it cannot finish the decision itself.
    chain_was_empty_at_record: bool,
    /// Whether the popup currently has a buffer on screen.
    pub mapped: bool,
    /// Where the popup is, in its `root`'s surface coordinates. Committed
    /// state, refreshed on map and after every reconfigure -- see
    /// `Wayland::popup_geometry`'s doc for why that is the right reading for
    /// `popup_at_point`.
    pub geometry: Rectangle,
}

/// Shrink `rect` by one layer entry's positive exclusive zone along
/// whichever single edge it is anchored to: `top`-anchored (regardless of
/// whether `left`/`right` are also set -- "single edge or
/// edge+both-perpendicular" both carve the same way) shrinks the top,
/// `bottom` the bottom, `left` the left, `right` the right.
/// `exclusive <= 0` reserves nothing, per wlr-layer-shell's own
/// definition (see [`LayerEntry::exclusive`]'s doc), and a surface
/// anchored to both edges of an axis at once (e.g. `top` and `bottom`)
/// has no single edge to carve into and is left unchanged -- the same
/// "nothing sane to do" case [`State::configure_layer`]'s placement rule
/// falls back to centering for.
///
/// A free function rather than a method, and shared by
/// [`State::arrange_layers`] (folding every output's `usable`) and
/// [`State::usable_before`] (a single panel's own placement base): the
/// two calls need to compute an identical fold, and a second
/// hand-written copy of this arithmetic is one more place for them to
/// drift apart.
///
/// H2: every arithmetic op here is saturating. `exclusive` is captured
/// (clamped) at `new_layer_surface`/`layer_surface_commit` before it ever
/// reaches this function, but this function has no way to enforce that on
/// its own -- and it folds over every mapped layer entry on every
/// `arrange_layers`/`usable_before` pass, so a second, defense-in-depth
/// guard here costs nothing and turns "two large-exclusive panels overflow
/// `rect.x`/`rect.y`" from a debug-build panic (release: silent i32
/// wraparound corruption) into a saturated, still-sane rect.
fn fold_exclusive_zone(mut rect: Rectangle, anchor: wlr::Anchor, exclusive: i32) -> Rectangle {
    if exclusive <= 0 {
        return rect;
    }
    // Task 7 (N8/exclusive-edge), corrected post-review (2f04984's Major
    // finding): matches wlroots' own `wlr_layer_surface_v1_get_exclusive_
    // edge`, which is the protocol spec's own rule
    // (`wlr-layer-shell-unstable-v1.xml`'s `set_exclusive_zone` doc,
    // conformance-tested by WLCS's `is_positioned_to_accommodate_other_
    // surfaces_exclusive_zone`): a positive exclusive zone is only
    // meaningful -- reserves space at all -- for exactly two anchor
    // shapes per axis: anchored to **one edge alone** (e.g. `ANCHOR_TOP`
    // with no `left`/`right`), or anchored to **that edge plus both
    // perpendicular edges** (e.g. `TOP | LEFT | RIGHT`, spanning the
    // other axis). Anchored to only two perpendicular edges (a corner,
    // e.g. `TOP | LEFT`), only two parallel edges (e.g. `LEFT | RIGHT`
    // with neither `top` nor `bottom`), or all four edges: reserves
    // nothing, same as the protocol's own "treated the same as zero"
    // wording.
    //
    // `left == right` below means "both set, or both unset" -- i.e.
    // either the perpendicular axis fully spans (the second legal shape)
    // or is not anchored at all (the first legal shape); a corner like
    // `TOP | LEFT` has `left=true, right=false`, `left == right` is
    // `false`, and correctly falls through to no reservation.
    //
    // 2f04984 first tightened this from the pre-task-7 rule
    // (`anchor.top != anchor.bottom`, which correctly matched single-edge
    // but *also* wrongly matched corners) straight to requiring both
    // perpendicular edges unconditionally -- fixing the corner
    // over-reservation but silently dropping the single-edge-alone case
    // to zero reservation, a real regression for the single most common
    // panel shape (a bar anchored to one edge, spanning nothing else).
    // No test caught it: every exclusive-zone test until this fix used
    // `TOP | LEFT | RIGHT` (`top_panel_entry`). See
    // `a_single_edge_only_anchor_still_reserves_its_exclusive_zone` and
    // `a_corner_anchored_exclusive_zone_reserves_nothing`, which now both
    // pass against this rule.
    let (left, right, top, bottom) = (anchor.left, anchor.right, anchor.top, anchor.bottom);
    if top && !bottom && (left == right) {
        rect.y = rect.y.saturating_add(exclusive);
        rect.height = rect.height.saturating_sub(exclusive).max(0);
    } else if bottom && !top && (left == right) {
        rect.height = rect.height.saturating_sub(exclusive).max(0);
    } else if left && !right && (top == bottom) {
        rect.x = rect.x.saturating_add(exclusive);
        rect.width = rect.width.saturating_sub(exclusive).max(0);
    } else if right && !left && (top == bottom) {
        rect.width = rect.width.saturating_sub(exclusive).max(0);
    }
    rect
}

/// H2: bound a client-controlled exclusive zone to something sane before it
/// is stored, so two panels with pathological `exclusive_zone` requests (a
/// hostile or buggy client can request anything up to `i32::MAX`) cannot
/// together carve past an output's own extent -- the belt to
/// `fold_exclusive_zone`'s saturating-arithmetic suspenders. Clamped
/// against `max(output.width, output.height)`, since a zone only ever
/// folds a single axis and either axis's whole span is already generous
/// headroom for a real panel; `None` (no output resolved yet, e.g. the
/// [`NO_OUTPUT`] sentinel) leaves the value unclamped here and relies
/// entirely on the saturating fold. `exclusive <= 0` ("reserve nothing",
/// see [`LayerEntry::exclusive`]'s doc) passes through untouched -- only
/// the reservation itself is bounded, not the "don't reserve" sentinel
/// space.
fn clamp_exclusive_zone(exclusive: i32, output: Option<&OutputSurface>) -> i32 {
    if exclusive <= 0 {
        return exclusive;
    }
    let bound = output
        .map(|o| o.geometry.width.max(o.geometry.height))
        .unwrap_or(i32::MAX);
    exclusive.min(bound)
}

/// Whether [`State::sync_seat_focus`] must leave the seat's real keyboard
/// focus alone because an interactive layer surface still holds it --
/// review finding J3's guard, extracted as a pure function of `layer_focus`
/// and `layers` so its exact logic is unit-testable without a live
/// `wlr::Runtime` (`sync_seat_focus`'s tail, `wayland.keyboard_focus`, is a
/// no-op with none attached, which would make every branch of the guard
/// look identical to a test that could only observe side effects on
/// `wayland`).
///
/// `true` only when `layer_focus` names an entry that is still `mapped`:
/// a stale `layer_focus` (the entry unmapped or was removed without
/// going through `layer_surface_unmapped`/`destroyed` -- defensively,
/// should not happen, but this is the one guard standing between that and
/// a permanently dead keyboard) must not block the model's own focus from
/// ever being asserted again.
/// One `lower_*_to_bottom` call [`State::sync_wallpaper_nodes`] issues, in
/// the order it issues them.
///
/// Extracted (with [`wallpaper_lower_plan`]) purely so review finding C1's
/// ordering contract is *assertable*: `wlr` exposes no scene z-query, and
/// `BufferId`/`RectId` have no `dangling_for_test` constructor, so neither
/// the resulting stacking order nor the ids involved can be observed from a
/// test. What can be pinned is the sequence of lower calls the sync will
/// make, which is where the whole bug lived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LowerStep {
    /// The freshly created wallpaper node at this index in the caller's
    /// created-in-this-pass list.
    Wallpaper(usize),
    /// The boot-time full-output background rect (`lib.rs::run()`).
    Background,
}

/// Verdict of [`State::resolve_unique_disabled`] -- the name->id lookup a
/// re-enable performs against `disabled_outputs` (review finding #13).
enum DisabledLookup<K> {
    /// No disabled output carries this connector name.
    None,
    /// Exactly one does -- safe to rehydrate to this id.
    Unique(K),
    /// Two or more share the name (empty `""` names, or a genuine duplicate):
    /// rehydrating an arbitrary one could map the head to the WRONG output, so
    /// the caller refuses rather than guess.
    Ambiguous,
}

/// The order in which [`State::sync_wallpaper_nodes`] must lower the
/// `created` wallpaper nodes it just made, plus the background rect.
///
/// **The contract: the background rect is lowered LAST.** `lower_*_to_bottom`
/// moves its target to the very bottom of the root's children, so the last
/// call wins the bottom. Lowering a wallpaper node last (what the code did
/// before finding C1) buried it under the opaque background rect and the
/// wallpaper was never visible; lowering the background last leaves it at
/// the very bottom with the wallpaper immediately above it and everything
/// else above that.
///
/// Empty when nothing was created: an existing node is already correctly
/// stacked, and re-lowering the background for a pass that changed nothing
/// would be pure churn.
fn wallpaper_lower_plan(created: usize, has_background: bool) -> Vec<LowerStep> {
    if created == 0 {
        return Vec::new();
    }
    let mut plan: Vec<LowerStep> = (0..created).map(LowerStep::Wallpaper).collect();
    if has_background {
        plan.push(LowerStep::Background);
    }
    plan
}

/// One axis of a layer surface's placement: its size along that axis and
/// its position, from the two anchors on that axis, the client's
/// `desired` size (0 = "compositor decides"), and the usable box's
/// `origin`/`span` on that axis.
///
/// The wlr-layer-shell rule, per axis (review finding I3):
///
/// | anchors on the axis | size                        | position                          |
/// |---------------------|-----------------------------|-----------------------------------|
/// | both edges          | span the box                | box origin                        |
/// | exactly one edge    | `desired`, else 30px        | flush against the anchored edge   |
/// | neither edge        | `desired`, else span the box| centered (origin when it spans)   |
///
/// The three fallbacks are each the protocol's own answer to a 0 the
/// client left for the compositor to choose: an axis anchored to both
/// edges *must* span it (a 0 there is not merely allowed but expected);
/// an axis anchored to exactly one edge is a panel's thickness, where 30px
/// is this compositor's house default; an axis anchored to neither with no
/// desired size is the protocol's fill case (all-four-anchors and
/// zero-anchors lockers/launchers alike) and gets the whole usable box
/// rather than an arbitrary 200px.
///
/// Worked examples: a `TOP|LEFT|RIGHT` 800x30 panel gets `(800, box.x)`
/// horizontally (both edges) and `(30, box.y)` vertically (one edge,
/// desired honored) -- unchanged from before. A `TOP|RIGHT` 300x100
/// notification gets `(300, box.x + box.width - 300)` and `(100, box.y)`:
/// its own size, flush into the corner. A four-edge-anchored 0x0 locker
/// gets the whole box on both axes.
fn layer_axis_placement(
    start: bool,
    end: bool,
    desired: u32,
    origin: i32,
    span: i32,
) -> (i32, i32) {
    // Finding 7, security: `desired` is client-controlled and otherwise
    // casts straight to `i32` -- a value at or past `i32::MAX` would wrap
    // negative on the cast, handing a negative "size" into the arithmetic
    // below. Clamped to the output's own span first: no legitimate panel
    // needs to claim more than the box it is being placed in.
    let desired = desired.min(span.max(0) as u32);
    let size = if start && end {
        span
    } else if desired != 0 {
        desired as i32
    } else if start || end {
        30
    } else {
        span
    };
    let pos = if start || (start == end && size >= span) {
        origin
    } else if end {
        origin + span - size
    } else {
        origin + (span - size) / 2
    };
    (size, pos)
}

fn layer_holds_keyboard_focus(
    layer_focus: Option<wlr::LayerSurfaceId>,
    layers: &HashMap<wlr::LayerSurfaceId, LayerEntry>,
) -> bool {
    layer_focus.is_some_and(|id| layers.get(&id).is_some_and(|entry| entry.mapped))
}

/// Finding 6, testing: the client's `wlr::Edges` -> `input::ResizeEdges`
/// mapping [`State::begin_client_resize`] uses, extracted as a pure
/// field-by-field translation so it is unit-testable without a live
/// `wlr::Runtime` -- `begin_client_resize`'s own success path (a pointer
/// actually positioned over the window, `pointer_pressed` true) has no
/// headless-runtime way to inject a pointer position at all, so this
/// mapping is what stays testable of it.
fn resize_edges_from_wlr(edges: wlr::Edges) -> input::ResizeEdges {
    input::ResizeEdges {
        top: edges.top,
        bottom: edges.bottom,
        left: edges.left,
        right: edges.right,
    }
}

/// Translate the compositor library's modifier booleans into the model's
/// bitflags.
///
/// A free function taking four `bool`s rather than a method on
/// `wlr::Modifiers`, because that type belongs to another crate and this
/// mapping is the compositor's own decision -- and because a pure function is
/// the only part of the key path that can be unit-tested at all: a
/// `wlr::KeyEvent` cannot be constructed outside a live seat.
///
/// `logo` is the Super / Windows key, which is what this project's `SUPER`
/// binding token means (`config/src/defaults.rs`).
pub fn to_model_modifiers(logo: bool, ctrl: bool, alt: bool, shift: bool) -> input::Modifiers {
    let mut out = input::Modifiers::empty();
    if logo {
        out |= input::Modifiers::SUPER;
    }
    if ctrl {
        out |= input::Modifiers::CTRL;
    }
    if alt {
        out |= input::Modifiers::ALT;
    }
    if shift {
        out |= input::Modifiers::SHIFT;
    }
    out
}

/// Whether an in-progress alt-tab session should end on this key event.
///
/// `watched` is the modifier flags [`input::modifiers_for_tokens`] derives
/// from the configured `cycle:alt_tab` binding -- not hardcoded to SUPER, so
/// a rebind (e.g. to ALT+Tab) still ends its session on the right key.
///
/// Two independent checks, not one, because neither alone is enough:
///
/// - `keysym_is_modifier(watched, keysym)` on a *release* (`!pressed`)
///   catches the modifier's own release **on that very event**. This is the
///   one that matters: wlroots' `keyboard_key_update` emits the `key` signal
///   *before* it calls `xkb_state_update_key`/`keyboard_modifier_update`
///   (`types/wlr_keyboard.c`), so `event.modifiers()` on the modifier key's
///   own release event still reports that modifier held -- `mods` cannot
///   see its own key going up. Only the keysym can.
/// - `!mods.contains(watched)` is the fallback for every event *after*
///   that one (or for any case the keysym check doesn't cover): once the
///   library's modifier state has actually updated, a session that somehow
///   missed its modifier's release event still ends on the very next key,
///   rather than lingering until an unrelated later sync.
///
/// A free function taking plain values rather than `&wlr::KeyEvent`, for the
/// same reason `to_model_modifiers` is one: `wlr::KeyEvent::new` is
/// `pub(crate)` to the `wlr` crate, so no test outside it can construct a
/// real one, and this is the only shape of the decision a test *can* drive.
pub fn alt_tab_should_end(
    watched: input::Modifiers,
    mods: input::Modifiers,
    pressed: bool,
    keysym: u32,
) -> bool {
    let modifier_released_this_event = !pressed && input::keysym_is_modifier(watched, keysym);
    modifier_released_this_event || !mods.contains(watched)
}

/// This compositor's `xdg-activation-v1` focus-steal policy, as a decision
/// over plain values so it can be table-tested: the decision needs a live
/// `State` and its toplevel map to reach otherwise, and neither is something
/// a unit test can stand up.
///
/// An activation may take the keyboard only when *all* of these hold:
///
/// - `requester == focused`: the token was requested by the window the user
///   is *currently* working in. This is the load-bearing condition -- the
///   one actually standing between a background process and the keyboard.
///   A window still holding focus is the only "a real user is interacting
///   with this client, right now" that this compositor can vouch for by
///   itself, and it makes the honored case exactly the intended one (the
///   app you are using handing you off to another of its windows).
/// - `target_activatable`: the target is mapped, not minimized, and on the
///   *active* workspace. A client's activation request never switches
///   workspaces or unminimizes here (owner ruling): honoring one that
///   cannot actually be focused where the user is looking would be a silent
///   no-op, so those cases take the refused path and raise an attention
///   hint the shell can show instead.
/// - `has_seat`: the client called `set_serial(serial, seat)` when minting
///   the token. **This is weaker than it looks and is not a security
///   check.** wlroots does *not* validate the serial against the seat: it
///   records whatever number the client passed, verbatim, and `has_seat`
///   only means "a seat was named at all". A token minted for another
///   process to redeem later usually lacks it, which is the whole of its
///   value here -- it filters out the sloppy case, not a determined one.
///   `requester == focused` is what carries the actual weight.
///
/// Everything else is refused -- not dropped: the caller marks the target
/// with an attention hint instead, so the shell can surface it without the
/// keyboard moving out from under the user.
fn activation_may_steal_focus(
    has_seat: bool,
    target_activatable: bool,
    requester: Option<WindowId>,
    focused: Option<WindowId>,
) -> bool {
    has_seat && target_activatable && requester.is_some() && requester == focused
}

/// Input passed to `State::handle_pointer`, in output logical coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerEvent {
    /// Button pressed while the pointer is over `id`.
    Press { id: WindowId, pointer: (i32, i32) },
    /// Pointer moved during an in-progress drag.
    Motion { pointer: (i32, i32) },
    /// Button released, ending an in-progress drag (a no-op if none is
    /// active).
    Release { pointer: (i32, i32) },
}

/// One mapped override-redirect (OR) X11 surface — an unmanaged pop-up (menu,
/// tooltip, combo/dropdown list, drag icon) tracked entirely outside the
/// `WindowManager` model (XWayland design Decision 4, M3).
///
/// An OR surface is never a `Window` row: it is placed at its own
/// client-requested coordinates, wears no server-side decoration, is not an
/// alt-tab or focus-MRU candidate, and stacks in the band **above** managed
/// toplevels. All the compositor keeps for it is where it goes and whether it
/// takes the keyboard; the scene node itself is the `wlr` crate's, built on
/// `associate` and reparented above `Band::Toplevel` on map.
#[derive(Debug, Clone, Copy)]
struct OverrideRedirectSurface {
    /// The pop-up's client-requested absolute geometry, refreshed on every
    /// `request_configure`. `x`/`y` drive the scene-node placement; the size is
    /// kept so a later reposition can be re-clamped if needed.
    geometry: Rectangle,
}

pub struct State {
    pub window_manager: WindowManager,
    pub config: Config,
    /// Outbound event channel. Every message carries the `seq` its mutation
    /// produced (review finding I2) so a subscriber can order signals
    /// against a `GetState()` snapshot and detect gaps.
    pub dbus_tx: crossbeam_channel::Sender<SeqEvent>,
    /// The compositor's Wayland side -- the one seam between this model and
    /// a client. See `wayland.rs`'s module doc.
    pub wayland: crate::wayland::Wayland,
    pub start_time: Instant,
    /// Wallpaper decode state (`render::spawn_wallpaper_decode` produces it;
    /// there is no renderer in this crate to upload or draw it yet).
    pub wallpaper: WallpaperState,
    /// The drag machine's current snap-preview target, in output logical
    /// coordinates, or `None` when no drag is in a snap-preview state. Set
    /// by the drag machine (Task 9); this is the geometry hook `render.rs`
    /// consumes.
    pub snap_preview: Option<icedtea_contract::Rectangle>,
    /// The scene rect node backing `snap_preview`, or `None` when no drag is
    /// showing one. Reconciled by `sync_snap_preview`, which must run
    /// immediately after every `snap_preview` assignment -- see that
    /// method's doc.
    snap_preview_rect: Option<wlr::RectId>,
    /// Output geometries keyed by output index. Used by fullscreen toggle.
    pub outputs: HashMap<u32, OutputSurface>,
    /// The next index `new_output` hands out. Monotonic, never reused --
    /// `outputs.len()` was tried first and is wrong the moment an output is
    /// removed and a new one added: `len()` computes the same index the
    /// removed output had, colliding with whatever the model (or a
    /// client-facing consumer) still remembers about it (review finding M2).
    next_output_index: u32,
    /// Incremented once per `LoopHandler::should_stop` call, i.e. once per
    /// event-loop turn regardless of what woke it (see that method's doc).
    /// Not read anywhere in a real boot -- this exists so a test can prove a
    /// negative ("the loop did not busy-spin") that no other observable
    /// state can: a wedged wake pipe (review finding C1) doesn't fail any
    /// assertion about *what* the loop did, only about how relentlessly it
    /// did nothing, and turn count is the only signal for that.
    pub turns: u64,
    /// Incremented once per `OutputHandler::frame` call. Review finding I1's
    /// test coverage: proves `new_output`'s `schedule_frame()` call actually
    /// results in at least one `frame` callback, not just that it compiles
    /// and doesn't panic.
    pub frames: u64,
    /// Saved window geometries before fullscreen toggle, keyed by window ID.
    /// Used to restore non-fullscreen geometry when exiting fullscreen.
    ///
    /// Kept separate from `snap_saved_geometry` (Task 11 review #2): both
    /// `toggle_fullscreen` and `snap` used to share one `saved_geometry` map
    /// keyed only by `WindowId`, so snapping a window and then
    /// fullscreening it clobbered the pre-snap geometry with the snapped
    /// one, and `snap_restore` after unfullscreening silently restored
    /// nothing (it returns `Some(())` on a missing entry). Two independent
    /// save slots per window means each feature's restore point survives
    /// the other feature running in between.
    pub fullscreen_saved_geometry: HashMap<WindowId, icedtea_contract::Rectangle>,
    /// Saved window geometry before a `snap`, keyed by window ID. Used by
    /// `snap_restore`. See `fullscreen_saved_geometry`'s doc for why this is
    /// a separate map.
    pub snap_saved_geometry: HashMap<WindowId, icedtea_contract::Rectangle>,
    /// Saved window geometry before a maximize, keyed by window ID. Its own
    /// slot for the same reason `fullscreen_saved_geometry` and
    /// `snap_saved_geometry` are separate: maximizing a snapped window must
    /// not clobber snap's restore point (review finding I5).
    pub maximized_saved_geometry: HashMap<WindowId, icedtea_contract::Rectangle>,
    /// Alt-tab cycling state, driven by `apply_action("cycle:alt_tab")` and
    /// ended by `end_alt_tab` (see its doc for the chosen end condition --
    /// driving this from a real keyboard filter is task 5's job).
    pub alt_tab: input::AltTabMachine,
    /// Pointer-driven window move/snap state machine.
    pub drag: input::DragMachine,
    /// Pointer-driven interactive-resize state machine, stepped by the same
    /// pointer motion/release path the drag machine uses. A client-driven
    /// resize request will start one again once task 5 wires the inbound
    /// half of that protocol back up.
    pub resize: input::ResizeMachine,
    /// Set by `apply_action("quit")`; the event loop (task 5 reintroduces
    /// one) checks this each iteration and calls `stop()` once true.
    pub quitting: bool,
    /// Set by `SeatHandler::session_lock_changed` while an
    /// `ext-session-lock-v1` client holds the session locked. The crate
    /// already refuses normal keyboard/pointer focus in this state (see
    /// `wlr::Runtime::is_session_locked`); this flag exists only so this
    /// model stops *fighting* that refusal -- without it, every geometry or
    /// focus mutation still queued behind the lock (a workspace switch, a
    /// close, a drag settling) would call `sync_window_to_scene`/
    /// `sync_seat_focus` and try to reassert toplevel keyboard focus the
    /// crate is silently dropping, which is at best wasted work and at
    /// worst a race the moment the lock is released. `session_lock_changed`
    /// clears it and restores focus to the MRU toplevel.
    pub session_locked: bool,
    /// Last known pointer position in output logical coordinates. Task 5's
    /// input plumbing updates this on every pointer-motion event;
    /// button-only events (which carry no position of their own) read it
    /// back to build `PointerEvent::Press`/`Release`.
    pub pointer_location: (i32, i32),
    /// Whether a pointer button is currently held down. Maintained at the
    /// top of `SeatHandler::pointer_button`, before any routing, so
    /// `begin_client_move`/`begin_client_resize` can enforce the policy this
    /// crate substitutes for the seat/serial the library deliberately does
    /// not forward with `request_move`/`request_resize` (see `wlr::Toplevel
    /// Handler`'s doc on those methods): honor an interactive move/resize
    /// request only while a real button-down backs it, never on the
    /// client's claim alone.
    pub pointer_pressed: bool,
    /// M7: whether any touch point is currently down. Set by the touch
    /// notification handlers (`touch_down` sets, `touch_up` re-derives
    /// from `Runtime::touch_state`, `touch_cancelled` clears) and by the
    /// test-only `InjectTouch*` command arms the same way, and read by the
    /// `GetState` arm into `Snapshot.touch_active` (which the panel renders
    /// as its touch indicator). A model mirror rather than a live
    /// `touch_state()` read at `GetState` time so `GetState` stays a pure
    /// model read (see that arm's "no model mutation" comment).
    pub touch_active: bool,
    /// M7: whether the seat cursor currently shows an image. Refreshed on
    /// every pointer path (motion, button) from `Runtime::cursor_state`'s
    /// image when a runtime is attached (falling back to `true`: in
    /// production a motion always applies an image first), and on
    /// `new_output`. Read by the `GetState` arm into
    /// `Snapshot.cursor_visible`.
    pub cursor_visible: bool,
    /// M7: the cursor's last-known position in output-logical coordinates,
    /// or `None` before the first pointer motion. Updated wherever
    /// `pointer_location` is (every `pointer_motion`/`pointer_button`),
    /// but kept separate: `pointer_location` is the drag machine's live
    /// input (always some pair, `(0, 0)` at boot), while this is the
    /// shell-facing reading where "unknown yet" is expressible -- which is
    /// what `Snapshot.cursor_pos` carries.
    pub cursor_pos: Option<(i32, i32)>,
    /// On-disk location `reload_config_from_disk`/`handle_command`'s
    /// `ReloadConfig` worker thread reads from. `None` (the boot-time
    /// default) means "use `icedtea_config::default_db_path()`" -- this is
    /// only ever overridden by tests, which need an isolated temp DB rather
    /// than the real XDG path.
    pub config_path: Option<PathBuf>,
    /// Set by `set_config_reload_sender` once the caller has somewhere to
    /// send reload results (in production, `lib.rs`'s `run()`, which also
    /// keeps the paired receiver -- see `config_reload_rx`).
    /// `handle_command`'s `ReloadConfig` arm sends the freshly-loaded
    /// `Config` back over this from a worker thread so the redb I/O + JSON
    /// parse never blocks the caller's loop; `None` before that wiring
    /// exists is a no-op (nothing to reload into).
    config_reload_tx: Option<crossbeam_channel::Sender<Config>>,
    /// Paired receiving half of `config_reload_tx`, drained once per turn by
    /// `drain_config_reload`. `None` until `set_config_reload_receiver`
    /// wires it -- tests that only exercise the sender side (the reload
    /// tests below) never set this.
    config_reload_rx: Option<crossbeam_channel::Receiver<Config>>,
    /// Which model output index each library output id maps to.
    ///
    /// Kept because `OutputHandler::destroyed` is given only an id, and the
    /// model's geometry map is keyed by index.
    output_ids: HashMap<wlr::OutputId, u32>,
    /// A scale `DbCommand::SetOutputScaleForTest` wants pushed onto the
    /// *live* `wlr::Output` (not just this model's `OutputSurface::scale`
    /// mirror) the next time `OutputHandler::frame` hands one back for that
    /// output id -- the only place after boot this crate has a live
    /// `&wlr::Output` to call `wlr::Output::set_scale` on (there is no
    /// `Runtime` method that reaches an output by id outside a handler
    /// callback). The reply is answered from `frame`, once the live output
    /// has actually adopted the scale, not from the `DbCommand` handler
    /// itself -- anything that reads the *live* `wlr_output.scale` (the
    /// fractional-scale protocol's auto-sent `preferred_scale`, notably)
    /// needs the real thing, and this crate's own `primary_output_scale`
    /// mirror was never a substitute for it. `None` when no test scale
    /// override is outstanding, which is true almost all the time.
    pending_test_output_scale: Option<(wlr::OutputId, f32, crossbeam_channel::Sender<bool>)>,
    /// Connector name -> the live `wlr::OutputId` of an output that is
    /// currently *disabled* and therefore has NO entry in `self.outputs` /
    /// `self.output_ids`. Populated both by `new_output`'s persisted-disabled
    /// branch (an output that boots disabled is never inserted into the active
    /// set) and by `output_configuration_applied`'s disable branch. It exists
    /// solely so a later re-enable can recover the id: the applied-config
    /// handler is handed only `Vec<AppliedHead>` (owned name + geometry, no
    /// id/handle), so without this map a re-enabled head could not be mapped
    /// back to a `wlr::OutputId` and would be left enabled-but-untracked
    /// (rendering nothing until restart). `destroyed` prunes it (a disabled
    /// output can be unplugged).
    ///
    /// Review finding #15: keyed by the always-unique [`wlr::OutputId`] (the
    /// value carries the connector name for the re-enable name->id match)
    /// rather than by name. Two unnamed disabled outputs both key under `""`
    /// under the old name-keyed map, so the second insert dropped the first
    /// id and it could never be rehydrated; a unique id key keeps both.
    disabled_outputs: HashMap<wlr::OutputId, String>,
    /// Serializes every worker-thread redb OPEN (both
    /// [`Self::spawn_config_save`]'s writer and
    /// [`Self::spawn_config_reload`]'s reader) so the two never hold a handle
    /// to the same file at once. redb permits only one `Database` per file
    /// per process; without this a save racing a reload hits
    /// `DatabaseAlreadyOpen`, and `load_or_default` then falls back to
    /// `default_config()` -- silently WIPING the live appearance/keybindings/
    /// workspaces (or dropping the display write) that were on disk (review
    /// finding #3). Each worker holds this lock across its whole open + use +
    /// drop, so the next open never begins until the previous handle is gone.
    config_db_lock: Arc<Mutex<()>>,
    /// The scene rect painted behind everything, once boot has made one.
    background: Option<wlr::RectId>,
    /// The fd source SIGINT/SIGTERM write to. Compared in `fd_ready` so that
    /// a future second source cannot be mistaken for this one.
    shutdown_source: Option<wlr::SourceId>,
    /// The D-Bus service's command channel, drained once per loop turn (by
    /// `fd_ready`'s `cmd_wake_source` arm, and as a backstop by
    /// `should_stop`).
    cmd_rx: Option<crossbeam_channel::Receiver<crate::dbus::DbCommand>>,
    /// The fd source `CompositorInterface::send` (`dbus.rs`) nudges after every
    /// command it forwards onto `cmd_rx`. Compared in `fd_ready` the same
    /// way `shutdown_source` is; without it a command sent while the loop is
    /// blocked in `dispatch(-1)` would sit undrained until some unrelated
    /// event happened to wake the loop anyway.
    cmd_wake_source: Option<wlr::SourceId>,
    /// The fd source `spawn_config_reload`'s worker thread nudges after
    /// sending a freshly-loaded config over `config_reload_tx`. Same
    /// purpose as `cmd_wake_source`, for `drain_config_reload`.
    config_reload_wake_source: Option<wlr::SourceId>,
    /// Write half of the pipe registered as `config_reload_wake_source`.
    /// Kept here (not just handed to the one worker thread that exists when
    /// `set_config_reload_wake` runs) because `spawn_config_reload` can spin
    /// up a fresh worker on every reload trigger, and each one needs its own
    /// `try_clone`d handle -- see that method's doc.
    config_reload_wake: Option<std::os::unix::net::UnixStream>,
    /// The receiving half of `render::spawn_wallpaper_decode`'s channel,
    /// drained once per turn by `drain_wallpaper` (mirrors `config_reload_rx`
    /// for the wallpaper decode worker). `None` until `set_wallpaper_receiver`
    /// wires it -- boot is the only caller, since there is exactly one decode
    /// worker per process lifetime.
    wallpaper_rx: Option<crossbeam_channel::Receiver<Option<image::RgbaImage>>>,
    /// Whether `drain_wallpaper` has ever successfully received a result on
    /// `wallpaper_rx`. Distinguishes the two ways the channel can report
    /// `Disconnected`: before this is `true`, disconnect-without-a-message
    /// means the worker really did exit early (panicked) and is worth a
    /// warn; after it, the worker sent its one message and exited exactly as
    /// designed, and the sender dropping is the expected, silent end of its
    /// lifetime (review finding M4).
    wallpaper_received: bool,
    /// The fd source the wallpaper decode worker nudges after it sends its
    /// result. Compared in `fd_ready` the same way `config_reload_wake_source`
    /// is, so a result produced while the loop is idle in `Until::Stop`'s
    /// blocking `dispatch(-1)` is picked up immediately rather than sitting
    /// unseen until an unrelated event happens to wake the loop.
    wallpaper_wake_source: Option<wlr::SourceId>,
    /// Write half of the pipe registered as `wallpaper_wake_source`, kept
    /// alive here for the same reason `config_reload_wake` is (review
    /// finding C1, fixed the same way `shutdown_source`'s sibling comment
    /// and `spawn_config_reload` both already document): the wallpaper
    /// decode worker is only ever handed a `try_clone`d copy, never this
    /// original. Without an owner that outlives the worker thread, the
    /// worker's clone was the *only* live write half, and it dropped the
    /// moment the thread exited after its one send -- leaving the
    /// registered read end with a permanent `EPOLLHUP` on libwayland's
    /// level-triggered loop, so `fd_ready`'s wallpaper arm (and
    /// `drain_wallpaper`'s "likely panicked" warn, before the M4 fix above)
    /// fired every single turn forever, for the rest of the process's life.
    wallpaper_wake: Option<std::os::unix::net::UnixStream>,
    /// One buffer scene node per output currently showing the decoded
    /// wallpaper image, keyed by the same model output index `outputs` is.
    /// Empty whenever `wallpaper.decoded()` is `None` (pre-decode, or a
    /// decode that failed) -- `sync_wallpaper_nodes` tears every node down
    /// in that case, leaving the solid `background` rect (`lib.rs::run()`)
    /// as the only thing on screen. `pub(crate)` so `wallpaper_node_count`
    /// can read it for test introspection without exposing the map itself.
    wallpaper_nodes: HashMap<u32, wlr::BufferId>,
    /// Font enumeration and glyph raster caches for `text::rasterize_title`.
    ///
    /// Deviation 5: built lazily, once, and never rebuilt.
    /// `FontSystem::new` walks every font directory on the machine (tens of
    /// milliseconds), and `SwashCache` is pure memoization of glyph rasters,
    /// so both belong to the process rather than to a call. A `OnceCell`
    /// rather than an eager field because the overwhelming majority of this
    /// crate's `State`s -- every unit test -- never rasterize anything at
    /// all, and must not pay for a font scan to construct a model.
    fonts: std::cell::OnceCell<(cosmic_text::FontSystem, cosmic_text::SwashCache)>,
    /// Per-window memo of the last title raster: the inputs it was made
    /// from, the pixels that came out, and the generation that names them.
    ///
    /// `sync_window_to_scene` runs on every geometry mutation -- once per
    /// pointer motion during a drag -- while a title's *pixels* only change
    /// when the text, the width available for it, or the resolved text color
    /// changes. This memo is what keeps an unchanged title from being
    /// re-shaped on every one of those syncs; the `generation` is what keeps
    /// it from being re-*uploaded* (review finding M2), which is the more
    /// expensive half: the seam remembers which generation each title node
    /// currently holds and skips `update_buffer` -- and the scene damage
    /// that comes with it -- when it already matches. The pixels are
    /// borrowed across the seam rather than cloned, so a cache hit costs a
    /// map lookup and nothing else.
    ///
    /// `None` on the pixel side is a *cached negative*: this title
    /// rasterized to nothing (empty, or unshapeable on this machine) and
    /// must not be retried on every sync. Dropped in `forget_window` and in
    /// the two teardown paths that bypass it (`apply_config`, and
    /// `sync_window_to_scene`'s vanished-row arm), so a window's raster
    /// never outlives it.
    title_rasters: HashMap<WindowId, TitleRasterEntry>,
    /// Source of `TitleRasterEntry::generation`. Monotonic and never reused,
    /// so "the seam holds generation N" can only ever mean one exact set of
    /// pixels -- across windows as well as within one.
    next_title_generation: u64,
    /// Decoration preferences stated by clients that have no model window
    /// yet, in `Window::client_decorations_requested`'s own three-valued
    /// spelling.
    ///
    /// A client creates its `zxdg_toplevel_decoration_v1` and calls
    /// `set_mode` on it *before* its initial commit, which is well before
    /// `mapped` creates the model row -- so the preference arrives with
    /// nowhere to put it. Parking it here and applying it at `new_toplevel`
    /// is what keeps an explicit "I draw my own decorations" from being
    /// silently dropped and the window handed a band it asked not to have.
    /// Entries are removed when the toplevel is bound (`new_toplevel`) or
    /// dies unmapped (`forget_toplevel`), so this never outgrows the set of
    /// unmapped toplevels.
    pending_decorations: HashMap<crate::wayland::ToplevelKey, Option<bool>>,
    /// Every live wlr-layer-shell surface's model-side bookkeeping, keyed
    /// by the library's own id. Populated at `new_layer_surface`, kept
    /// current on every `layer_surface_commit`, and removed only at
    /// `layer_surface_destroyed` -- an unmap leaves the entry in place
    /// (see `layer_surface_unmapped`'s doc, mirroring the toplevel
    /// unmapped/mapped distinction `Window::mapped` already draws).
    pub layers: HashMap<wlr::LayerSurfaceId, LayerEntry>,
    /// The keyboard-interactive layer surface currently holding seat
    /// keyboard focus, if any. Tracked separately from the model's own
    /// window focus because a layer surface has no `WindowId` at all --
    /// this is what `layer_surface_unmapped`/`layer_surface_destroyed`
    /// check before calling `sync_seat_focus` to hand focus back to
    /// whatever the model says is focused.
    layer_focus: Option<wlr::LayerSurfaceId>,
    /// Source of [`LayerEntry::sequence`]. Monotonic and never reused, the
    /// same shape as `next_title_generation`.
    next_layer_sequence: u64,
    /// Every live popup's model-side bookkeeping, keyed by the library's own
    /// id. Populated at `new_popup`, kept current on map/unmap/reposition,
    /// removed at `popup_destroyed` -- and pruned wholesale when a chain's
    /// root dies, since a root's death takes its popups with it and the
    /// per-popup destroys may never be delivered.
    popups: HashMap<crate::wayland::PopupKey, PopupEntry>,
    /// Creation-ordered; the tail is the topmost popup.
    ///
    /// Z-order itself is the scene's (contract §2.2: a popup's scene subtree
    /// hangs off its parent's, so it stacks with its parent for free and the
    /// compositor never calls `wlr_scene_node_place_*` for a popup). This
    /// exists for the two things the scene cannot answer from the model side:
    /// hit-testing (`popup_at_point`) and dismissal order.
    ///
    popup_stack: Vec<crate::wayland::PopupKey>,
    /// How many popups this compositor has itself closed through
    /// `Wayland::dismiss_popup` since boot, summed over every call.
    ///
    /// Introspection for `compositor/tests/popups.rs`, which cannot see this
    /// number any other way: the client-visible `popup_done` order is
    /// wlroots' (it frees a popup's children before the popup), so the count
    /// is the only place a shallow-first dismissal shows up -- it finds the
    /// deeper rows already swept and under-counts. Read through
    /// `DbCommand::PopupsDismissed`.
    popups_dismissed: usize,
    /// Source of [`PopupEntry::sequence`]. Monotonic and never reused, the
    /// same shape as `next_layer_sequence`.
    ///
    next_popup_sequence: u64,
    /// Whom keyboard focus returns to when a **grabbing** chain ends.
    ///
    /// Set only when a chain's first popup grabs (contract deviation D3: a
    /// non-grabbing popup never moves focus, so restoring one would be the
    /// move rule 2 forbids). Taken -- not merely read -- by
    /// `restore_focus_after_popups`, so a second chain cannot inherit the
    /// first one's target.
    ///
    focus_before_popup: Option<PopupRoot>,
    /// Rasterized button-glyph pixels, keyed by `(button index, width,
    /// height, fg)`. Unlike `title_rasters` this needs no per-window entry
    /// or generation counter: a glyph's pixels are a pure function of that
    /// key (the text for each index is fixed, the cell size is always
    /// `BUTTON_WIDTH x TITLE_BAR_HEIGHT`, and `fg` never varies with the
    /// window), so in practice this map holds exactly three entries for the
    /// whole process's life -- one rasterization per button, ever. `None` is
    /// a cached negative, the same convention `title_rasters` uses.
    button_glyph_cache: HashMap<ButtonGlyphKey, Option<Vec<u8>>>,
    /// The title-bar button, if any, the pointer currently sits over --
    /// `(window, button index)`, `decoration::button_at`'s own index order.
    /// Recomputed on every pointer motion (`update_ssd_hover`); scene-only
    /// state, so changing it re-syncs the affected window(s)'s decoration
    /// but never emits a `contract::Event` -- there is no model mutation
    /// here, only a color the seam draws.
    ssd_hover: Option<(WindowId, usize)>,
    /// The title-bar button, if any, currently held down -- set for the
    /// span of `handle_pointer_press`'s own button-action branch, so the
    /// pressed color is what that press's `sync_window_to_scene` call sees
    /// before the action it triggers (close/maximize/minimize) runs. Same
    /// "scene-only, no `Event`" rule as `ssd_hover`.
    ssd_press: Option<(WindowId, usize)>,
    /// Which model window each mapped **managed** X11 (Xwayland) surface backs.
    ///
    /// The M1 spike's minimal parallel binding: xdg toplevels ride
    /// `wayland::Wayland`'s `ToplevelKey`↔`WindowId` maps and the
    /// `ToplevelId`-keyed scene seam, but an `XwaylandSurfaceId` has no
    /// `ToplevelId`, so a managed X11 window enters the same `WindowManager`
    /// model through this side-table instead of that seam. It proves one X11
    /// window reaches the model end-to-end; the fuller `SurfaceKey`
    /// generalization that folds both onto one focus/stacking/SSD path is a
    /// later milestone (M2). Override-redirect surfaces are never entered here.
    xwayland_windows: HashMap<wlr::XwaylandSurfaceId, WindowId>,
    /// Every mapped override-redirect (OR) X11 surface, keyed by
    /// `XwaylandSurfaceId` — the M3 unmanaged-pop-up side-table (XWayland design
    /// Decision 4). Deliberately **not** the `WindowManager` model: an OR
    /// surface is a menu/tooltip/combo popup positioned at its own coordinates,
    /// above managed toplevels, with no SSD and no place in alt-tab/MRU. Entries
    /// are added on `xwayland_surface_mapped` (and on a managed→OR runtime flip),
    /// repositioned on `request_configure`, and removed on unmap/destroy (or on
    /// an OR→managed flip). Native xdg-popups are modelled separately, by
    /// `State::popups`/`PopupEntry`: they hang off a `PopupHost`, are placed
    /// against `popup_constraint_box`, and never enter `WindowManager` either.
    override_redirect: HashMap<wlr::XwaylandSurfaceId, OverrideRedirectSurface>,
    /// The stack of focus-taking OR pop-ups that hold the seat keyboard, oldest
    /// first — the override-redirect analogue of [`layer_focus`](Self::layer_focus),
    /// but a stack rather than a single slot so nested menus nest correctly
    /// (review finding #7).
    ///
    /// A focus-taking OR pop-up (a keyboard-navigable menu) is handed the
    /// keyboard on map and pushed here; a submenu that opens over it is pushed on
    /// top. While the stack is non-empty, `sync_seat_focus` must not reassert the
    /// model's toplevel focus over the top entry (exactly the churn `layer_focus`
    /// guards against). When the top pop-up unmaps, is destroyed, or flips back
    /// to managed, it is popped and the keyboard returns to the *parent* menu
    /// still beneath it — only when the stack empties does the model's toplevel
    /// focus come back. A middle entry closing is simply spliced out, leaving the
    /// current holder untouched.
    or_keyboard_stack: Vec<wlr::XwaylandSurfaceId>,
    /// The `DISPLAY` name (`:N`) Xwayland last advertised, captured on
    /// `xwayland_ready`. `None` until Xwayland is up (or on a build/host with no
    /// Xwayland at all). Read back by the test-only `DbCommand::XwaylandDisplay`
    /// accessor, and the value exported into the environment session children
    /// inherit.
    xwayland_display: Option<String>,
    /// Scene nodes of currently-placed input-method candidate popups, keyed by
    /// the crate's [`wlr::InputPopupSurfaceId`]. Written in
    /// `SeatHandler::new_popup_surface` (the node `add_input_popup_in_band`
    /// returned) and removed in `popup_surface_destroyed`. The crate exposes no
    /// by-id popup-position accessor, so the compositor keeps this so the
    /// test-only `DbCommand::InputPopupPosition` oracle can resolve a placed
    /// popup's scene position through `wlr::Runtime::node_position`.
    input_popup_nodes: HashMap<wlr::InputPopupSurfaceId, wlr::NodeId>,
    /// The live preedit overlay, if composing text is currently shown. A
    /// dedicated field — not `input_popup_nodes`, whose lifecycle is the
    /// candidate popup's — written by the IME-commit hook and cleared by
    /// every hide path (commit-string, deactivate, keyboard-focus change).
    /// Read back by the test-only `DbCommand::PreeditOverlay` oracle.
    preedit_overlay: Option<crate::ime_overlay::PreeditOverlay>,
}

/// What a cached title raster depends on: the title text, the pixel width
/// available for it, and the *resolved* foreground color. Anything else
/// changing -- position, height, workspace -- cannot change the pixels.
///
/// The color rather than a `focused` flag (review finding M3): focus is
/// only one of the two things that pick the text color, the palette being
/// the other. Keying on the resolved bytes covers both, and covers them
/// exactly -- a reload that changed some unrelated part of the config
/// re-uses the raster, while one that changed `palette.foreground`
/// invalidates it even if the reload someday stops destroying every window.
type TitleRasterKey = (String, i32, [u8; 4]);

/// A rasterized title's pixels: `(width, height, premultiplied RGBA)`.
type TitlePixels = (i32, i32, Vec<u8>);

/// What a cached button-glyph raster depends on: which button
/// (`decoration::button_rects`' index), the cell it was shaped into, and the
/// (constant) glyph color -- see `button_glyph_cache`'s own doc for why that
/// is the whole key, with no per-window component.
type ButtonGlyphKey = (usize, i32, i32, [u8; 4]);

/// One window's memoized title raster.
struct TitleRasterEntry {
    /// The inputs these pixels were shaped from; a miss against this is the
    /// only thing that shapes.
    key: TitleRasterKey,
    /// Names this exact set of pixels for the seam's upload check. Bumped
    /// only when the pixels are re-shaped.
    generation: u64,
    /// `None` is a cached negative -- nothing shaped, show the bare band.
    pixels: Option<TitlePixels>,
}

/// Left inset of the title text inside the band, in pixels.
const TITLE_PAD_X: i32 = 8;

/// The three button glyphs, `decoration::button_rects`' own order: an
/// en-dash for minimize (the universal "make this go away downward" mark),
/// a hollow square for maximize (an unfilled window outline), and a
/// multiplication-x for close. Plain Unicode rather than an icon font or an
/// embedded bitmap -- the same reasoning `rasterize_title` already commits
/// to for text -- so `cosmic-text`'s ordinary font-fallback path draws them,
/// with no new asset for this crate to ship or a machine to be missing.
const BUTTON_GLYPHS: [&str; 3] = ["\u{2013}", "\u{25A1}", "\u{2715}"];

/// The glyph color painted into every button, regardless of which button or
/// which window: white reads against both the semi-transparent foreground
/// chip (minimize/maximize) and the opaque accent close button, for every
/// palette this config format can express -- one fewer color the config
/// would otherwise need a field for.
const BUTTON_GLYPH_FG: [u8; 4] = [255, 255, 255, 255];

/// `color` at `alpha`, premultiplied -- every channel scaled, not just the
/// alpha one, because the wlroots scene graph composites premultiplied
/// colors and a rect whose RGB outran its alpha would come out brighter than
/// the same color opaque.
fn premultiply(color: [f32; 4], alpha: f32) -> [f32; 4] {
    let a = (color[3] * alpha).clamp(0.0, 1.0);
    [color[0] * a, color[1] * a, color[2] * a, a]
}

/// Word-splits a `"spawn"` action's command string into a program plus its
/// argument vector, for a direct (no-shell) `Command::new(program).args(..)`
/// exec. This intentionally does NOT implement shell semantics: no pipes,
/// redirection, `$VAR` expansion, or globbing -- only whitespace splitting
/// with single/double-quote grouping (a backslash inside a quoted or bare
/// word escapes the following character; an unterminated quote consumes the
/// rest of the string literally, matching a simple shlex-style reader).
/// Returns `None` when the command is empty or splits to zero words (e.g.
/// all whitespace).
fn parse_spawn_argv(cmd: &str) -> Option<(String, Vec<String>)> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = cmd.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else if c == '\\' && q == '"' {
                    // Backslash escapes inside double quotes (matches
                    // common shell behavior); single quotes are literal.
                    if let Some(&next) = chars.peek() {
                        current.push(next);
                        chars.next();
                    } else {
                        current.push(c);
                    }
                } else {
                    current.push(c);
                }
            }
            None => {
                if c == '\'' || c == '"' {
                    quote = Some(c);
                    in_word = true;
                } else if c == '\\' {
                    if let Some(&next) = chars.peek() {
                        current.push(next);
                        chars.next();
                        in_word = true;
                    }
                } else if c.is_whitespace() {
                    if in_word {
                        words.push(std::mem::take(&mut current));
                        in_word = false;
                    }
                } else {
                    current.push(c);
                    in_word = true;
                }
            }
        }
    }
    if in_word || quote.is_some() {
        words.push(current);
    }
    let mut words = words.into_iter();
    // Review finding (LOW): an all-quotes input like `""` splits to one
    // *empty* word rather than zero words, so a bare `words.next()?` would
    // return `Some("")` instead of taking the "empty or unparseable
    // command" `None` path that `apply_action`'s `"spawn"` arm logs and
    // ignores. Filtering the empty program out here routes `""` / `"" ""`
    // through that same `None` path.
    let program = words.next().filter(|p| !p.is_empty())?;
    Some((program, words.collect()))
}

impl State {
    pub fn new(config: Config, dbus_tx: crossbeam_channel::Sender<SeqEvent>) -> Self {
        let workspace_names = config.workspace_names.clone();
        // Review finding I4: report unusable bindings once, here, instead of
        // from inside the per-key-press lookup.
        input::warn_about_keybindings(&config.keybindings);

        Self {
            window_manager: WindowManager::new(workspace_names),
            config,
            dbus_tx,
            wayland: crate::wayland::Wayland::new(),
            start_time: Instant::now(),
            wallpaper: WallpaperState::new(),
            snap_preview: None,
            snap_preview_rect: None,
            outputs: HashMap::new(),
            next_output_index: 0,
            turns: 0,
            frames: 0,
            fullscreen_saved_geometry: HashMap::new(),
            snap_saved_geometry: HashMap::new(),
            maximized_saved_geometry: HashMap::new(),
            alt_tab: input::AltTabMachine::new(),
            drag: input::DragMachine::new(),
            resize: input::ResizeMachine::new(),
            quitting: false,
            session_locked: false,
            pointer_location: (0, 0),
            pointer_pressed: false,
            // M7 input mirrors, feeding `Snapshot.cursor_visible` /
            // `cursor_pos` / `touch_active` (see each field's doc).
            // `cursor_visible: false` is the truthful boot state: the
            // crate applies the first cursor image on the first pointer
            // motion (`ensure_cursor_image`), so before that the cursor
            // is `Hidden`. `cursor_pos: None` for the same reason: the
            // model has observed no cursor position yet.
            touch_active: false,
            cursor_visible: false,
            cursor_pos: None,
            config_path: None,
            config_reload_tx: None,
            config_reload_rx: None,
            output_ids: HashMap::new(),
            pending_test_output_scale: None,
            disabled_outputs: HashMap::new(),
            config_db_lock: Arc::new(Mutex::new(())),
            background: None,
            shutdown_source: None,
            cmd_rx: None,
            cmd_wake_source: None,
            config_reload_wake_source: None,
            config_reload_wake: None,
            wallpaper_rx: None,
            wallpaper_received: false,
            wallpaper_wake_source: None,
            wallpaper_wake: None,
            wallpaper_nodes: HashMap::new(),
            fonts: std::cell::OnceCell::new(),
            title_rasters: HashMap::new(),
            next_title_generation: 0,
            pending_decorations: HashMap::new(),
            layers: HashMap::new(),
            layer_focus: None,
            next_layer_sequence: 0,
            popups: HashMap::new(),
            popup_stack: Vec::new(),
            popups_dismissed: 0,
            next_popup_sequence: 0,
            focus_before_popup: None,
            button_glyph_cache: HashMap::new(),
            ssd_hover: None,
            ssd_press: None,
            xwayland_windows: HashMap::new(),
            override_redirect: HashMap::new(),
            or_keyboard_stack: Vec::new(),
            xwayland_display: None,
            input_popup_nodes: HashMap::new(),
            preedit_overlay: None,
        }
    }

    /// Wire up the config-reload result channel. Both halves are kept: the
    /// sender goes to the worker thread `spawn_config_reload` spins up, and
    /// the loop drains the receiver itself (via `drain_config_reload`) now
    /// that there is no event-source abstraction to do it.
    pub fn set_config_reload_sender(&mut self, tx: crossbeam_channel::Sender<Config>) {
        self.config_reload_tx = Some(tx);
    }

    /// Keep the receiving half so [`Self::drain_config_reload`] has
    /// somewhere to read from. Separate from `set_config_reload_sender`
    /// because the tests wire only the sender and read the receiver
    /// themselves.
    pub fn set_config_reload_receiver(&mut self, rx: crossbeam_channel::Receiver<Config>) {
        self.config_reload_rx = Some(rx);
    }

    /// Apply any config a reload worker has finished loading.
    ///
    /// Called once per event-loop turn. Non-blocking: `try_recv` on an empty
    /// channel is the overwhelmingly common case and must cost nothing.
    pub fn drain_config_reload(&mut self) {
        let Some(rx) = self.config_reload_rx.as_ref() else {
            return;
        };
        let pending: Vec<Config> = rx.try_iter().collect();
        for cfg in pending {
            let _ = self.apply_reloaded_config(cfg);
        }
    }

    /// Apply the wallpaper decode worker's result, if it has sent one.
    ///
    /// Called once per event-loop turn, the same shape as
    /// `drain_config_reload`: non-blocking (`try_recv` on an empty channel
    /// is the overwhelmingly common case, both before the worker finishes
    /// and forever after, since it sends exactly one result and exits), and
    /// panic-free on a disconnected sender -- `spawn_wallpaper_decode`'s
    /// thread has either not sent yet, sent once, or panicked, and none of
    /// those is a reason for this handler to abort the process.
    pub fn drain_wallpaper(&mut self) {
        let Some(rx) = self.wallpaper_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(image) => {
                self.wallpaper.set_decoded(image);
                self.wallpaper_received = true;
                self.sync_wallpaper_nodes();
            }
            // Review finding M4: `Disconnected` means two different things
            // depending on `wallpaper_received`. Before a result has ever
            // arrived, the worker exiting without sending one really is
            // abnormal (a panic) and is worth a warn. After one has arrived,
            // the sender dropping is just the worker reaching the end of its
            // one-shot lifetime -- `spawn_wallpaper_decode` sends exactly
            // once and returns -- and this arm is reached on every
            // subsequent turn's poll, so warning here would be a false
            // "likely panicked" on every turn of an otherwise healthy,
            // long-idle compositor.
            Err(crossbeam_channel::TryRecvError::Disconnected) if !self.wallpaper_received => {
                tracing::warn!(
                    "wallpaper decode worker thread exited without producing a result \
                     (it likely panicked); wallpaper stays solid-color"
                );
                // Only warn once: this arm's condition would otherwise stay
                // true and re-fire the same warn on every future turn too.
                self.wallpaper_received = true;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {}
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }
    }

    /// How many wallpaper buffer nodes are currently tracked. Introspection
    /// for tests -- nothing in a real boot needs to count these itself
    /// (mirrors `Wayland::ssd_rect_count`).
    pub fn wallpaper_node_count(&self) -> usize {
        self.wallpaper_nodes.len()
    }

    /// Creates/updates one wallpaper buffer node per output from the
    /// decoded image, stretched to fill (`render::wallpaper_dest`); removes
    /// nodes for outputs that are gone. Idempotent -- calling this again
    /// with nothing changed just re-sets each node's position/size and
    /// re-lowers it, which are all no-ops on an already-correct scene.
    ///
    /// **Contract (task 7/14): nodes are never pixel-refreshed in place.**
    /// The existing-node branch below only repositions/resizes; it never
    /// calls `update_buffer` (or re-creates the node) to show a *different*
    /// image's pixels under an unchanged output index. That branch was
    /// unreachable until task 14 gave `apply_reloaded_config` a wallpaper
    /// swap path -- reachable now, so the contract has to be stated rather
    /// than left implicit: a wallpaper change must clear first
    /// (`wallpaper.set_decoded(None)` + a call here to tear every node
    /// down) before the fresh decode's result calls this again to rebuild
    /// them. `apply_reloaded_config` enforces exactly that ordering; nothing
    /// here checks it, so a caller that skips the clear would silently keep
    /// showing the old image at the old node's size/position rather than
    /// picking up the new one.
    ///
    /// Called whenever either side of the sync could have changed: a fresh
    /// decode landing (`drain_wallpaper`), or the output set changing
    /// (`OutputHandler::new_output`/`destroyed`). Not a model mutation --
    /// these are scene bookkeeping only, so this never touches
    /// `window_manager` or emits an event (the exactly-one-event-per-mutation
    /// invariant has nothing to say about a rect existing on screen).
    ///
    /// With no runtime attached (every unit test that never calls
    /// `wayland.attach`), every `wlr` call below is a no-op that returns
    /// `None`/does nothing, so this degrades to a harmless no-op that also
    /// never populates `wallpaper_nodes` -- there is no `BufferId` a real
    /// call could have returned to put in the map.
    pub fn sync_wallpaper_nodes(&mut self) {
        let Some(runtime) = self.wayland.runtime().cloned() else {
            return;
        };

        let Some(img) = self.wallpaper.decoded() else {
            // No decoded image (pre-decode, or a decode that failed): tear
            // every node down and fall back to the solid background rect.
            // `remove_buffer` tolerates a stale id (already gone via its own
            // toplevel teardown, though wallpaper nodes are root buffers so
            // that specific race does not apply here) -- just `None`, never
            // a panic.
            for (_, node) in self.wallpaper_nodes.drain() {
                runtime.remove_buffer(node);
            }
            return;
        };

        let width = img.width() as i32;
        let height = img.height() as i32;
        // Never panic in handler-reachable code: a zero-dimension decode
        // (which `image` should never actually produce, but nothing here
        // proves it can't) skips every output rather than handing wlroots a
        // buffer `add_buffer`/`update_buffer` would reject anyway.
        if width == 0 || height == 0 {
            return;
        }
        let rgba = img.as_raw().as_slice();

        // Freshly created nodes, in creation order. Collected rather than
        // lowered inline because the lowering *order* is the whole
        // correctness question here -- see `wallpaper_lower_plan`.
        let mut created: Vec<wlr::BufferId> = Vec::new();
        for (index, surface) in &self.outputs {
            let dest = render::wallpaper_dest(surface.geometry);
            if let Some(&node) = self.wallpaper_nodes.get(index) {
                runtime.set_buffer_position(node, dest.x, dest.y);
                runtime.set_buffer_dest_size(node, dest.width, dest.height);
            } else {
                match runtime.add_buffer(width, height, rgba) {
                    Ok(node) => {
                        runtime.set_buffer_position(node, dest.x, dest.y);
                        runtime.set_buffer_dest_size(node, dest.width, dest.height);
                        created.push(node);
                        self.wallpaper_nodes.insert(*index, node);
                    }
                    // Finding 5, errors: the failure used to be silently
                    // dropped, leaving that output with no wallpaper node
                    // and no trace of why -- named with the output index so
                    // a multi-output setup's log points at the one output
                    // actually missing its wallpaper.
                    Err(err) => {
                        tracing::warn!(
                            output = index,
                            ?err,
                            "failed to create wallpaper buffer node for output"
                        );
                    }
                }
            }
        }
        // Review finding C1: every wallpaper node used to be lowered inline
        // right here, *after* `lib.rs::run()` had already lowered the
        // opaque background rect at boot -- and `lower_*_to_bottom` moves
        // its target to the very bottom of the root's children, so the
        // most recently lowered node wins the bottom. That put every
        // wallpaper node *underneath* the full-output opaque rect and the
        // wallpaper was never visible at all. The background rect goes
        // last now, which is what actually puts it at the very bottom and
        // leaves the wallpaper directly above it -- solid color as the
        // pre-decode/decode-failure fallback, not a permanent occlusion.
        for step in wallpaper_lower_plan(created.len(), self.background.is_some()) {
            match step {
                LowerStep::Wallpaper(i) => {
                    if let Some(&node) = created.get(i) {
                        runtime.lower_buffer_to_bottom(node);
                    }
                }
                LowerStep::Background => {
                    if let Some(bg) = self.background {
                        runtime.lower_rect_to_bottom(bg);
                    }
                }
            }
        }

        // Drop nodes for outputs that are gone.
        let gone: Vec<u32> = self
            .wallpaper_nodes
            .keys()
            .filter(|index| !self.outputs.contains_key(index))
            .copied()
            .collect();
        for index in gone {
            if let Some(node) = self.wallpaper_nodes.remove(&index) {
                runtime.remove_buffer(node);
            }
        }
    }

    /// Reconcile the snap-preview scene rect with `self.snap_preview`:
    /// create it (accent color at 35% alpha, premultiplied) the moment a
    /// drag starts showing a target, reposition/resize it in place as the
    /// target changes, and remove it the moment there is no longer one to
    /// show.
    ///
    /// Call this immediately after every assignment to `snap_preview` --
    /// the two fields are meant to be read as one unit, and nothing else
    /// keeps them in sync. With no runtime attached (every unit test that
    /// never calls `wayland.attach`) this is a no-op, the same degradation
    /// `sync_wallpaper_nodes` documents for the identical reason: there is
    /// no `RectId` a real call could have returned to put in
    /// `snap_preview_rect`.
    pub fn sync_snap_preview(&mut self) {
        let Some(runtime) = self.wayland.runtime().cloned() else {
            return;
        };
        match self.snap_preview {
            Some(rect) => match self.snap_preview_rect {
                Some(id) => {
                    runtime.set_rect_position(id, rect.x, rect.y);
                    runtime.set_rect_size(id, rect.width, rect.height);
                }
                None => {
                    let color = premultiply(
                        render::hex_to_rgba(&self.config.appearance.palette.accent),
                        0.35,
                    );
                    if let Ok(id) =
                        runtime.add_rect_in_band(wlr::Band::Overlay, rect.width, rect.height, color)
                    {
                        runtime.set_rect_position(id, rect.x, rect.y);
                        self.snap_preview_rect = Some(id);
                    }
                }
            },
            None => {
                if let Some(id) = self.snap_preview_rect.take() {
                    runtime.remove_rect(id);
                }
            }
        }
    }

    /// Ask the event loop to stop.
    ///
    /// Sets the same flag `apply_action("quit")` does rather than signalling a
    /// loop handle, because the loop is now driven by `wlr` and asks the state
    /// whether to stop (`LoopHandler::should_stop`) instead of being told.
    pub fn stop(&mut self) {
        self.quitting = true;
    }

    /// Record an output's geometry. Backends call this once they know the mode.
    ///
    /// The smithay version of this also created a protocol global and mapped
    /// the output into a `Space`; both now belong to the compositor library
    /// (the global comes with the backend, the placement with the scene's
    /// output layout), so all that is left here is the geometry map that
    /// `snap`, `set_fullscreen_target`, `set_maximized_target` and
    /// `handle_pointer_motion` read.
    pub fn create_output(&mut self, index: u32, geometry: icedtea_contract::Rectangle) {
        self.outputs.insert(index, OutputSurface::new(geometry));
    }

    /// The output whose box contains the pointer; falls back to the lowest
    /// index. Placement (cascade origin, maximize/fullscreen/snap target)
    /// uses this instead of the implicit single output, so every one of
    /// those consumers moves with the pointer once a second output exists.
    ///
    /// With no attached runtime (every unit test in this file, and any
    /// caller ahead of `wayland.attach`), `pointer_position` is unavailable
    /// -- the fallback path also covers a pointer that landed outside every
    /// known output's box, which can happen for an instant right after a
    /// hotplug removes the one it was over.
    pub fn output_for_pointer(&self) -> Option<u32> {
        if let Some(runtime) = self.wayland.runtime() {
            let (px, py) = runtime.pointer_position();
            let (px, py) = (px as i32, py as i32);
            let hit = self
                .outputs
                .iter()
                .find(|(_, out)| out.geometry.contains(px, py));
            if let Some((&idx, _)) = hit {
                return Some(idx);
            }
        }
        self.outputs.keys().min().copied()
    }

    /// The output whose box contains `geometry`'s frame center, falling
    /// back to [`Self::output_for_pointer`] (whose own fallback is the
    /// lowest index) when no output's box contains it.
    ///
    /// Review finding J1: `arrange_layers` used to re-home every
    /// maximized window through `output_for_pointer` alone, which is a
    /// *pointer*-driven disambiguator every one of its other callers gets
    /// to use because they all run inside a user-initiated action
    /// (maximize, snap, drag) where the pointer genuinely names the
    /// window in question. `arrange_layers` runs from an unrelated
    /// client's layer-surface commit -- a status bar's clock tick is
    /// enough -- so the pointer carries no information about which
    /// maximized window is being re-laid-out, and using it teleported a
    /// maximized window on output 1 onto output 0's `usable` rect the
    /// moment a panel on output 1 committed while the pointer merely
    /// happened to be sitting over output 0. A window's own frame center
    /// is what actually names its output; mirrors
    /// `migrate_windows_from`'s identical containment test.
    fn output_for_window(&self, geometry: Rectangle) -> Option<u32> {
        let cx = geometry.x + geometry.width / 2;
        let cy = geometry.y + geometry.height / 2;
        let hit = self
            .outputs
            .iter()
            .find(|(_, out)| out.geometry.contains(cx, cy));
        if let Some((&idx, _)) = hit {
            return Some(idx);
        }
        self.output_for_pointer()
    }

    /// Finding 4, maintainability: the `usable` rect of whatever output
    /// [`Self::output_for_pointer`] names, extracted once rather than
    /// inlined at each pointer-correct call site (`snap`,
    /// `handle_pointer_motion`, `new_toplevel`'s cascade placement) --
    /// those genuinely run inside a user-initiated, pointer-driven action
    /// (see `output_for_window`'s own doc for the distinction from a
    /// client-request/D-Bus path, which is not one of these). `None` when
    /// no output resolves at all.
    fn usable_geo_for_pointer(&self) -> Option<Rectangle> {
        self.output_for_pointer()
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.usable)
    }

    /// Recompute every output's `usable` rect from its layer surfaces'
    /// exclusive zones, then re-sync every maximized window whose target
    /// rect actually changed so maximize immediately tracks the new
    /// usable area.
    ///
    /// Reset-then-fold: every output's `usable` starts back at `geometry`
    /// (a panel that shrank its zone, moved output, unmapped, or was
    /// destroyed must give its space back, not just never claim more of
    /// it) and each *mapped* (review finding J2 -- an unmapped surface
    /// reserves nothing, see [`LayerEntry::mapped`]'s doc) layer entry's
    /// positive exclusive zone shrinks the respective edge via
    /// `fold_exclusive_zone` -- `exclusive <= 0` reserves nothing, per
    /// wlr-layer-shell's own definition (see [`LayerEntry::exclusive`]'s
    /// doc).
    ///
    /// Fullscreen windows are deliberately never touched here at all --
    /// not filtered out after the fact, never in the affected set to
    /// begin with -- because fullscreen geometry is defined to keep
    /// `geometry`, never `usable` (`set_fullscreen_target`'s own doc), so
    /// no exclusive-zone change this method makes could ever be relevant
    /// to one; re-syncing them on every panel commit was pure waste
    /// (review finding M4).
    ///
    /// A maximized window's target is only applied -- `set_geometry` +
    /// `sync_window_to_scene` -- when it actually differs from the
    /// window's current geometry (review finding M4): `set_geometry`
    /// emits `WindowUpdated` unconditionally, so without this guard a
    /// panel redrawing its clock once a second produced one D-Bus signal
    /// and one client `configure` per maximized window, per panel frame,
    /// with byte-identical geometry.
    ///
    /// Each maximized window's output is its own frame center's, not a
    /// single pointer-derived one (review finding J1) -- see
    /// `output_for_window`'s own doc for the multi-output regression this
    /// closes.
    ///
    /// Windows are collected into a `Vec` before any is synced (review
    /// pattern this crate already follows elsewhere, e.g.
    /// `migrate_windows_from`): `sync_window_to_scene` reads `self`
    /// broadly, so mutating `window_manager` while still mid-iteration
    /// over it would not borrow-check.
    pub fn arrange_layers(&mut self) {
        for output in self.outputs.values_mut() {
            output.usable = output.geometry;
        }
        for entry in self.layers.values() {
            if !entry.mapped {
                continue;
            }
            let Some(output) = self.outputs.get_mut(&entry.output) else {
                continue;
            };
            output.usable = fold_exclusive_zone(output.usable, entry.anchor, entry.exclusive);
        }

        // Re-review finding Minor-1: the fold above can change a *later*
        // (higher-`sequence`) mapped panel's own placement input --
        // `usable_before`'s output -- without that panel ever committing
        // itself, e.g. an earlier panel growing its zone or unmapping.
        // Nothing else would ever re-offer it a fresh placement, so it
        // stayed positioned where it was last configured until its own
        // next commit happened to come along. Reconfiguring every mapped
        // panel here closes that, and does not reopen M4's storm class:
        // `configure_layer` only actually sends when the computed
        // placement differs from the one it last sent (`LayerEntry::
        // last_configured`), so an arrange pass that changed nothing for
        // a given panel costs one comparison, not a wire round trip.
        let mapped_layers: Vec<wlr::LayerSurfaceId> = self
            .layers
            .iter()
            .filter(|(_, e)| e.mapped)
            .map(|(&id, _)| id)
            .collect();
        for id in mapped_layers {
            self.configure_layer(id);
        }

        let gap = self.config.appearance.snap_gap;
        // N11: a maximized window that is minimized, or sitting on a
        // workspace that is not the active one, is not on screen -- this
        // sweep re-syncing it anyway did no visible harm by itself (nothing
        // draws it), but it still emitted a `WindowUpdated` and warped its
        // stored geometry to whatever `usable` happens to be *right now* on
        // an output the user cannot see, which is wrong the moment the
        // window is later shown again (unminimized, or its workspace
        // switched to) with a panel layout that has since changed underfoot
        // with no configure of its own. `continue` for both up front, same
        // "not actually visible" test `windows_in_workspace` already uses.
        let active_workspace = self.window_manager.active_workspace();
        let affected: Vec<WindowId> = self
            .window_manager
            .windows()
            .filter(|w| w.maximized && !w.minimized && w.workspace == active_workspace)
            .map(|w| w.id)
            .collect();
        for id in affected {
            let Some(w) = self.window_manager.get(id) else {
                continue;
            };
            let Some(output_idx) = self.output_for_window(w.geometry) else {
                continue;
            };
            let Some(output_geo) = self.outputs.get(&output_idx).map(|o| o.usable) else {
                continue;
            };
            let target = layout::maximized_geometry(output_geo, gap);
            if target == w.geometry {
                continue;
            }
            let _ = self.window_manager.set_geometry(id, target);
            self.sync_window_to_scene(id);
        }
        self.emit_pending();

        // Every output's `usable` rect was just recomputed, and that rect *is*
        // the popup constraint box -- a panel appearing, resizing or leaving
        // re-places every reactive popup on that output, whatever it hangs
        // off.
        for root in self.popup_roots() {
            self.reconstrain_popups(root);
        }
    }

    /// The `usable` rect for `output_idx` as every *other*, mapped,
    /// positive-exclusive-zone entry whose [`LayerEntry::sequence`] is
    /// strictly less than `before` has already carved it -- `None` if
    /// `output_idx` names no live output.
    ///
    /// This is [`Self::configure_layer`]'s N6 fix: `arrange_layers`'
    /// `output.usable` already folds *every* entry together (itself
    /// included), which is right for "how much space is left for
    /// windows" but wrong as a placement base for an individual panel --
    /// self-excluding shrinks a panel by its own reservation, and folding
    /// every other entry with no ordering stacks two same-edge panels on
    /// top of each other rather than one beside the other. Folding only
    /// entries ordered strictly before this one gives a deterministic
    /// stack instead: whichever panel was announced first sits flush
    /// against the edge, the next stacks outward from it.
    fn usable_before(&self, output_idx: u32, before: u64) -> Option<Rectangle> {
        let mut rect = self.outputs.get(&output_idx)?.geometry;
        for entry in self.layers.values() {
            if entry.output != output_idx || entry.sequence >= before || !entry.mapped {
                continue;
            }
            rect = fold_exclusive_zone(rect, entry.anchor, entry.exclusive);
        }
        Some(rect)
    }

    /// Pure half of `configure_layer` (task 7): choose `id`'s layer
    /// surface's `(width, height, x, y)` for its output box, the
    /// placement rule from task 20's brief, without touching
    /// `last_configured` or sending anything over the wire. Extracted so
    /// the rule itself -- particularly N7, honoring `desired_size` on
    /// whichever axis is not anchored to both of its edges -- is directly
    /// unit-testable without a live `wlr::Runtime`; `configure_layer` is
    /// now compute-then-send: call this, then decide whether to record
    /// and answer.
    ///
    /// `None`, panic-free, if the entry or its output has vanished (a
    /// commit racing a hotplug-removed output, or the `NO_OUTPUT`
    /// sentinel -- see [`LayerEntry::output`]'s doc).
    pub fn compute_layer_placement(&self, id: wlr::LayerSurfaceId) -> Option<(u32, u32, i32, i32)> {
        let entry = self.layers.get(&id)?;
        let (output_idx, sequence, a, (desired_w, desired_h)) =
            (entry.output, entry.sequence, entry.anchor, entry.size);
        // N6: placed against the space every earlier-announced panel on
        // this output has already carved, not the raw output box --
        // otherwise two same-edge exclusive panels draw on top of each
        // other. See `usable_before`'s own doc.
        let box_ = self.usable_before(output_idx, sequence)?;

        // Placement is per-axis and independent (review finding I3) -- see
        // `layer_axis_placement`. The rule task 20 shipped keyed on
        // `a.top != a.bottom` / `a.left != a.right` for the *whole*
        // placement, which spanned a corner-anchored notification across
        // the full output width (discarding its desired width outright) and
        // dropped a four-edge-anchored 0x0 locker into a centered 200x200
        // box instead of filling the output.
        let (w, x) = layer_axis_placement(a.left, a.right, desired_w, box_.x, box_.width);
        let (h, y) = layer_axis_placement(a.top, a.bottom, desired_h, box_.y, box_.height);

        let (w, h) = (w.max(0) as u32, h.max(0) as u32);
        Some((w, h, x, y))
    }

    /// Announce a popup to the model.
    ///
    /// Drops the popup outright when its host is not something this
    /// compositor models -- an unbound toplevel, a layer surface already
    /// forgotten, or a parent popup that died between the two `new_popup`
    /// signals. That is the untrusted-client posture the rest of this file
    /// takes (contract §9): a client can drive any of the three, and none of
    /// them is a reason to fabricate a root.
    ///
    /// `grabbing` is whatever `new_popup`'s caller already knows, which in
    /// production is `Popup::grab_requested()` read at `new_popup` time --
    /// and that read is unreliable there. `xdg_popup.grab` (which is what
    /// sets `wlr_xdg_popup.seat`, the field `grab_requested` reads) is a
    /// *separate*, later protocol request than the `get_popup` request whose
    /// processing fires `new_popup`, so a grabbing popup's very first
    /// `record_popup` call always sees `grabbing == false` here even though
    /// the client did send `grab` (xdg-shell only requires it to precede the
    /// first *commit*, not `get_popup`). `reconcile_popup_grab`, called from
    /// `popup_initial_commit`, is what corrects `grabbing` and finishes this
    /// method's "only the chain's first popup parks a restore target"
    /// decision once the grab request (if any) is guaranteed to have landed.
    pub(crate) fn record_popup(
        &mut self,
        popup: crate::wayland::PopupKey,
        host: PopupHost,
        grabbing: bool,
    ) {
        let Some(root) = self.popup_root_of_host(host) else {
            tracing::debug!(
                ?popup,
                ?host,
                "ignoring a popup whose host this compositor does not model"
            );
            return;
        };
        let output = self.popup_output_of_root(root);
        let sequence = self.next_popup_sequence;
        self.next_popup_sequence += 1;
        // Read before the insert: "was this chain empty" must not count the
        // popup being recorded.
        let chain_was_empty = !self.popups.values().any(|entry| entry.root == root);
        self.popups.insert(
            popup,
            PopupEntry {
                host,
                root,
                output,
                sequence,
                grabbing,
                chain_was_empty_at_record: chain_was_empty,
                // False until `popup_mapped`, exactly as `LayerEntry::mapped`
                // is: a popup with no buffer yet is not on screen and must not
                // answer `popup_at_point`.
                mapped: false,
                geometry: Rectangle {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                },
            },
        );
        self.popup_stack.push(popup);
        // Deviation D3: only a grabbing chain parks a restore target, and only
        // its first popup does -- a submenu opening over a menu must not
        // overwrite the window the whole chain came from.
        if grabbing && chain_was_empty {
            self.focus_before_popup = Some(root);
        }
    }

    /// Finish `record_popup`'s grab bookkeeping once `Popup::grab_requested`
    /// is actually trustworthy (see `record_popup`'s doc on why it cannot be
    /// read at `new_popup` time). A no-op unless the popup both requested a
    /// grab and `record_popup` has not already been corrected for it --
    /// guards re-entry, since a popup can commit more than once.
    fn reconcile_popup_grab(&mut self, popup: &wlr::Popup<'_>) {
        if !popup.grab_requested() {
            return;
        }
        let key = crate::wayland::PopupKey::new(popup.id());
        let Some(entry) = self.popups.get(&key) else {
            return;
        };
        if entry.grabbing {
            return;
        }
        let root = entry.root;
        let chain_was_empty = entry.chain_was_empty_at_record;
        if let Some(entry) = self.popups.get_mut(&key) {
            entry.grabbing = true;
        }
        // Same condition `record_popup` applies, just evaluated now that the
        // grab is known for certain.
        if chain_was_empty {
            self.focus_before_popup = Some(root);
        }
    }

    /// Drop `popup` from the model. A no-op, panic-free, on a key this
    /// compositor was never told about -- contract §1.3's caveat, which
    /// `layer_surface_destroyed` already takes the same posture towards.
    ///
    pub(crate) fn forget_popup(&mut self, popup: crate::wayland::PopupKey) {
        self.popups.remove(&popup);
        self.popup_stack.retain(|key| *key != popup);
    }

    /// The chain root `popup` belongs to, or `None` if it is not recorded.
    pub fn popup_root(&self, popup: crate::wayland::PopupKey) -> Option<PopupRoot> {
        self.popups.get(&popup).map(|entry| entry.root)
    }

    /// Every popup under `root`, deepest last.
    ///
    /// Sorted by `sequence` rather than by walking `host` links: a chain is
    /// created bottom-up, so creation order *is* depth order, and a sort
    /// cannot loop on a cycle a malicious client might otherwise build out of
    /// `PopupHost::Popup` links.
    pub fn popup_chain(&self, root: PopupRoot) -> Vec<crate::wayland::PopupKey> {
        let mut chain: Vec<(u64, crate::wayland::PopupKey)> = self
            .popups
            .iter()
            .filter(|(_, entry)| entry.root == root)
            .map(|(key, entry)| (entry.sequence, *key))
            .collect();
        chain.sort_unstable_by_key(|(sequence, _)| *sequence);
        chain.into_iter().map(|(_, key)| key).collect()
    }

    /// How many popups the model currently holds. Introspection for tests --
    /// mirrors `ssd_rect_count()`.
    pub fn popup_count(&self) -> usize {
        self.popups.len()
    }

    /// Resolve a host to its chain root, or `None` when the host is not
    /// modelled. A popup host resolves through the parent's own recorded
    /// root, so the walk is one step deep however deep the chain is.
    ///
    fn popup_root_of_host(&self, host: PopupHost) -> Option<PopupRoot> {
        match host {
            PopupHost::Window(id) => self
                .window_manager
                .get(id)
                .map(|window| PopupRoot::Window(window.id)),
            PopupHost::Layer(id) => self
                .layers
                .contains_key(&id)
                .then_some(PopupRoot::Layer(id)),
            PopupHost::Popup(parent) => self.popups.get(&parent).map(|entry| entry.root),
        }
    }

    /// The model output index `root` sits on, or [`NO_OUTPUT`].
    ///
    fn popup_output_of_root(&self, root: PopupRoot) -> u32 {
        match root {
            PopupRoot::Window(id) => self
                .window_manager
                .get(id)
                .and_then(|window| self.output_for_window(window.geometry))
                .unwrap_or(NO_OUTPUT),
            PopupRoot::Layer(id) => self.layers.get(&id).map_or(NO_OUTPUT, |entry| entry.output),
        }
    }

    /// The origin of `root`'s own `wl_surface`, in frame (output-logical)
    /// space.
    ///
    /// For a window that is the **content** rect, not the frame: a
    /// server-decorated window's client surface starts one `TITLE_BAR_HEIGHT`
    /// below the frame's top edge (`decoration::content_rect`), and the scene
    /// node the popup's subtree hangs off is placed there too. For a layer
    /// surface it is the placement `compute_layer_placement` computes.
    ///
    /// `None`, panic-free, when the root or its output has vanished.
    fn root_surface_origin(&self, root: PopupRoot) -> Option<(i32, i32)> {
        match root {
            PopupRoot::Window(id) => {
                let window = self.window_manager.get(id)?;
                let ssd = crate::decoration::has_ssd(
                    &window.app_id,
                    window.client_decorations_requested,
                    window.fullscreen,
                );
                let content = crate::decoration::content_rect(window.geometry, ssd);
                Some((content.x, content.y))
            }
            PopupRoot::Layer(id) => {
                let (_, _, x, y) = self.compute_layer_placement(id)?;
                Some((x, y))
            }
        }
    }

    /// The constraint box handed to the library's `configure_popup`, **in the
    /// root toplevel/layer surface's coordinate system** (contract ruling R7).
    ///
    /// The box itself is the root's output `usable` rect
    /// ([`OutputSurface::usable`]) -- the same working area tiling and
    /// maximize read, so a panel's menu is clamped exactly where a window
    /// would be -- translated by minus the root's surface origin.
    ///
    /// `None` if the popup is unknown, or its root or that root's output is
    /// gone (including the `NO_OUTPUT` sentinel, which never resolves).
    pub fn popup_constraint_box(&self, popup: crate::wayland::PopupKey) -> Option<Rectangle> {
        let entry = self.popups.get(&popup)?;
        let (origin_x, origin_y) = self.root_surface_origin(entry.root)?;
        let usable = self.outputs.get(&entry.output)?.usable;
        Some(Rectangle {
            x: usable.x.saturating_sub(origin_x),
            y: usable.y.saturating_sub(origin_y),
            width: usable.width,
            height: usable.height,
        })
    }

    /// `Rectangle::contains`, but every addition saturates.
    ///
    /// `icedtea_contract::Rectangle::contains` computes `self.x + self.width`
    /// (and the `y`/`height` equivalent) with plain arithmetic; a positioner
    /// geometry with an `i32::MIN` origin and an `i32::MIN` extent -- both
    /// reachable from a hostile client -- overflows there in a debug build.
    /// `popup_at_point` is the only caller, and only for a `PopupEntry`
    /// geometry, which is exactly that untrusted value.
    fn rect_contains_saturating(rect: Rectangle, x: i32, y: i32) -> bool {
        x >= rect.x
            && x < rect.x.saturating_add(rect.width)
            && y >= rect.y
            && y < rect.y.saturating_add(rect.height)
    }

    /// The topmost mapped popup whose last-known geometry contains `point`
    /// (frame space), or `None`.
    ///
    /// Searched back to front over `popup_stack`, so the most recently
    /// created popup over a point wins -- which is also its scene order, since
    /// a popup's subtree is created above its parent's existing children.
    ///
    /// `None` while the session is locked, matching every other model path
    /// that answers input: the library's own hit test is already rooted at the
    /// lock band while locked (contract §1.5), and this is the model-side
    /// half of the same rule.
    pub fn popup_at_point(&self, point: (i32, i32)) -> Option<crate::wayland::PopupKey> {
        if self.session_locked {
            return None;
        }
        self.popup_stack.iter().rev().copied().find(|key| {
            let Some(entry) = self.popups.get(key) else {
                return false;
            };
            if !entry.mapped {
                return false;
            }
            let Some((origin_x, origin_y)) = self.root_surface_origin(entry.root) else {
                return false;
            };
            // Saturating: both the point and the origin are ultimately
            // client-controlled, and `Rectangle::contains` on a saturated
            // coordinate is merely wrong, not a crash.
            //
            // Reconciliation (Task 4): `icedtea_contract::Rectangle::contains`
            // itself computes `self.x + self.width` with plain arithmetic, so
            // a hostile positioner's `i32::MIN` origin plus `i32::MIN` extent
            // overflows there regardless of how the point is translated. This
            // crate cannot edit the contract crate from this task, so
            // `rect_contains_saturating` below is the same test with every
            // addition also saturating.
            Self::rect_contains_saturating(
                entry.geometry,
                point.0.saturating_sub(origin_x),
                point.1.saturating_sub(origin_y),
            )
        })
    }

    /// Unconstrain `popup` against its current constraint box and answer it
    /// with a configure. A no-op on any miss -- no runtime, no entry, no
    /// output, or a surface that is not `initialized` yet.
    fn configure_popup_now(&mut self, popup: crate::wayland::PopupKey) {
        let Some(constraint) = self.popup_constraint_box(popup) else {
            return;
        };
        if !self.wayland.configure_popup(popup, constraint) {
            tracing::trace!(
                ?popup,
                "popup not configured yet -- not initialized, or already gone"
            );
        }
    }

    /// Re-read where `popup` actually is and store it. See
    /// `Wayland::popup_geometry` for why this is committed rather than
    /// scheduled state.
    fn refresh_popup_geometry(&mut self, popup: crate::wayland::PopupKey) {
        let Some(geometry) = self.wayland.popup_geometry(popup) else {
            return;
        };
        if let Some(entry) = self.popups.get_mut(&popup) {
            entry.geometry = geometry;
        }
    }

    /// Re-run unconstrain + configure for every **reactive** popup under
    /// `root`, parents before children.
    ///
    /// Called from the two places a root's surface can move under a live
    /// popup: `sync_window_to_scene` (every geometry push, including each
    /// frame of a drag) and `arrange_layers` (a panel appearing, resizing or
    /// leaving changes every output's usable area, which is the constraint
    /// box itself). Non-reactive popups are left alone: the client did not
    /// ask to be re-placed, and re-placing one would move a menu out from
    /// under the pointer.
    pub fn reconstrain_popups(&mut self, root: PopupRoot) {
        for popup in self.popup_chain(root) {
            if !self.wayland.popup_is_reactive(popup) {
                continue;
            }
            self.configure_popup_now(popup);
            self.refresh_popup_geometry(popup);
        }
    }

    /// Hand keyboard focus back to the chain's root once a grabbing chain has
    /// fully emptied.
    ///
    /// Only a grabbing chain ever parks a target (deviation D3), so this is a
    /// no-op for tooltips and non-modal popovers. Points focus at the
    /// **parent**, not at the pointer position: the click that dismissed a
    /// menu commonly lands on some other window, and inheriting that as the
    /// new focus is exactly the behaviour spec §2 rules out.
    ///
    /// A root that died while its chain was up (the window closed under the
    /// menu) drops the target and re-derives the seat from the model instead.
    fn restore_focus_after_popups(&mut self) {
        let Some(root) = self.focus_before_popup.take() else {
            return;
        };
        match root {
            PopupRoot::Window(id) if self.window_manager.get(id).is_some() => {
                let previous = self.focused_id();
                // An explicit toplevel-focus assertion, so `layer_focus` is
                // released first -- see `release_layer_focus`'s own doc.
                self.release_layer_focus();
                if self.window_manager.focus(id).is_some() {
                    self.sync_focus_change(previous);
                }
                self.emit_pending();
            }
            PopupRoot::Layer(id) if self.layers.contains_key(&id) => {
                // Through the focus helper: restoring layer focus can itself
                // move the seat off a composing text input.
                let took = self
                    .change_keyboard_focus(|rt| rt.focus_layer_keyboard(id))
                    .is_some_and(|took| took.is_some());
                if took {
                    self.layer_focus = Some(id);
                } else {
                    self.sync_seat_focus();
                }
            }
            _ => self.sync_seat_focus(),
        }
    }

    /// Whether wlroots currently has an explicit seat grab up -- an xdg-popup
    /// grab or a drag-and-drop grab.
    fn runtime_has_explicit_grab(&self) -> bool {
        self.wayland.has_explicit_grab()
    }

    /// Drop every popup under `root`, and any focus-restore target pointing
    /// at it.
    ///
    /// wlroots frees a dying parent's popups with it, and the per-popup
    /// `destroy` signals may or may not reach a handler first; the model has
    /// to be correct either way. Called from `forget_toplevel` and
    /// `layer_surface_destroyed`.
    fn forget_popups_of_root(&mut self, root: PopupRoot) {
        for popup in self.popup_chain(root) {
            self.forget_popup(popup);
        }
        if self.focus_before_popup == Some(root) {
            self.focus_before_popup = None;
        }
    }

    /// Dismiss every popup chain whose root window the compositor has just
    /// taken off screen -- minimized, moved to another workspace, or left
    /// behind by a workspace switch.
    ///
    /// A menu is otherwise only ever closed by its own client (Escape is
    /// client-side, contract focus rule 5), by wlroots' popup grab on a press
    /// outside a *grabbing* chain (§1.6), or by its parent dying
    /// (`forget_popups_of_root`). None of those fires when the **compositor**
    /// hides the parent: the popup's scene subtree goes invisible with it, but
    /// the popup is still alive, the client still believes its menu is open,
    /// and a grabbing chain keeps wlroots' seat grab up over a surface nobody
    /// can see or click. So the compositor closes them itself -- which is the
    /// decision `Wayland::dismiss_popup` exists for, and the only path in this
    /// compositor that reaches `Runtime::dismiss_popup`.
    ///
    /// Deepest-first is the library's job and non-negotiable:
    /// `xdg_popup.destroy` on a popup with live children is a protocol error.
    /// `compositor/tests/popups.rs`'s
    /// `destroying_a_parent_destroys_its_popup_chain_without_a_double_free`
    /// is what proves the order end to end, per contract §2.4.
    ///
    /// Idempotent, and safe to call from a handler: an already-dismissed chain
    /// is a lookup miss inside the library, and the model rows are pruned by
    /// the `popup_destroyed` events this queues (the library defers a handler
    /// callback raised from inside a handler rather than re-entering).
    fn dismiss_popups_of_hidden_roots(&mut self) {
        let active = self.window_manager.active_workspace();
        for root in self.popup_roots() {
            let PopupRoot::Window(id) = root else {
                continue;
            };
            let visible = self
                .window_manager
                .get(id)
                .is_some_and(|w| w.mapped && !w.minimized && w.workspace == active);
            if visible {
                continue;
            }
            // Shallow-first over the chain: the first call takes the whole
            // subtree under it, and the rest are misses. Iterating the chain
            // rather than dismissing only its head is what covers a root with
            // more than one chain hanging off it.
            for popup in self.popup_chain(root) {
                self.popups_dismissed += self.wayland.dismiss_popup(popup);
            }
        }
    }

    /// Every distinct chain root with at least one live popup, in a stable
    /// order (by the lowest `sequence` in each chain) so a sweep over them is
    /// deterministic.
    fn popup_roots(&self) -> Vec<PopupRoot> {
        let mut seen: Vec<(u64, PopupRoot)> = Vec::new();
        for entry in self.popups.values() {
            match seen.iter_mut().find(|(_, root)| *root == entry.root) {
                Some((sequence, _)) => *sequence = (*sequence).min(entry.sequence),
                None => seen.push((entry.sequence, entry.root)),
            }
        }
        seen.sort_unstable_by_key(|(sequence, _)| *sequence);
        seen.into_iter().map(|(_, root)| root).collect()
    }

    /// Translate the library's `PopupParent` into the model's `PopupHost`.
    /// `None` when the parent is not something this compositor models.
    fn popup_host_for(&self, parent: wlr::PopupParent) -> Option<PopupHost> {
        match parent {
            wlr::PopupParent::Toplevel(id) => self
                .wayland
                .window_for(crate::wayland::ToplevelKey::new(id))
                .map(PopupHost::Window),
            wlr::PopupParent::Layer(id) => self
                .layers
                .contains_key(&id)
                .then_some(PopupHost::Layer(id)),
            wlr::PopupParent::Popup(id) => {
                let key = crate::wayland::PopupKey::new(id);
                self.popups
                    .contains_key(&key)
                    .then_some(PopupHost::Popup(key))
            }
        }
    }

    /// Test-only shim: drive `ToplevelHandler::popup_destroyed` from a
    /// `PopupKey` without the caller needing to unwrap the library id.
    #[cfg(test)]
    fn popup_destroyed_for_test(&mut self, popup: crate::wayland::PopupKey) {
        wlr::ToplevelHandler::popup_destroyed(self, popup.0);
    }

    /// Choose `id`'s layer surface's size and position for its output box
    /// (the placement rule from task 20's brief) and answer with
    /// `configure_layer_surface` + `set_layer_surface_position`.
    ///
    /// A no-op, panic-free, if the entry or its output has vanished (a
    /// commit racing a hotplug-removed output, or the `NO_OUTPUT`
    /// sentinel -- see [`LayerEntry::output`]'s doc), no runtime is
    /// attached (every unit test in this file), or the computed placement
    /// is unchanged from [`LayerEntry::last_configured`] -- see that
    /// field's own doc (review finding Minor-1): this makes the method
    /// safe to call unconditionally, which `arrange_layers` now does for
    /// every mapped panel on every pass.
    pub fn configure_layer(&mut self, id: wlr::LayerSurfaceId) {
        let Some(placement) = self.compute_layer_placement(id) else {
            return;
        };
        let (w, h, x, y) = placement;
        // Minor-1's storm guard: unchanged since the last time this was
        // computed is a no-op. `None` (never computed before) never
        // matches, so a surface's first `configure_layer` -- from
        // `new_layer_surface` or its own first commit -- always goes out;
        // the mandatory-configure contract is unaffected.
        if self
            .layers
            .get(&id)
            .is_some_and(|e| e.last_configured == Some(placement))
        {
            return;
        }
        // Recorded before the runtime-gated send below, not after: this
        // is what `configure_layer` computed and is *treating* as current
        // regardless of whether a runtime happened to be attached to
        // actually put it on the wire (every unit test in this file has
        // none). Production always has one attached by the time any of
        // this crate's handlers run, so the gap that would open --
        // recording a placement this method never actually sent -- cannot
        // happen outside a test.
        if let Some(entry) = self.layers.get_mut(&id) {
            entry.last_configured = Some(placement);
        }
        let Some(runtime) = self.wayland.runtime() else {
            return;
        };
        runtime.configure_layer_surface(id, w, h);
        runtime.set_layer_surface_position(id, x, y);
    }

    /// Re-home every [`LayerEntry`] whose `output` no longer names a live
    /// output (the [`NO_OUTPUT`] sentinel, or an index a since-removed
    /// output left behind) onto a surviving one -- the layer analogue of
    /// [`Self::migrate_windows_from`], but pull rather than push: this
    /// scans every entry rather than only the ones a specific removed
    /// output owned, so the same sweep also recovers a surface that had
    /// no output *at all* when it was announced (review finding M5) the
    /// moment any output exists.
    ///
    /// Re-resolution is per-entry (task 8, M2): each orphaned entry's own
    /// last-configured placement's frame center is tested against every
    /// surviving output's box, falling back to the lowest surviving index
    /// only when the entry has no placement yet (`last_configured: None`)
    /// or its center hits none of them -- not a single
    /// `output_for_pointer`-derived survivor applied to the whole batch,
    /// which let an unrelated commit's pointer position decide where every
    /// orphaned surface (however placed) landed.
    ///
    /// A no-op, correctly, when no output exists at all: every entry's
    /// resolution falls through to the lowest-index fallback, which itself
    /// yields nothing, and every orphaned entry is left exactly where it
    /// was for the next call (the next `new_output`) to try again --
    /// which is what makes the "no output at all when announced" case
    /// self-heal instead of hanging its client forever (review finding
    /// M5).
    ///
    /// Called from both `OutputHandler::new_output` (a fresh or returning
    /// output may be exactly what an orphaned entry was waiting for) and
    /// `OutputHandler::destroyed` (an entry the dying output owned needs
    /// somewhere else to go right away, not just whenever the next output
    /// happens to appear).
    fn resolve_orphaned_layers(&mut self) {
        let orphaned: Vec<wlr::LayerSurfaceId> = self
            .layers
            .iter()
            .filter(|(_, entry)| !self.outputs.contains_key(&entry.output))
            .map(|(&id, _)| id)
            .collect();
        if orphaned.is_empty() {
            return;
        }
        // M2/M5: cloned once, not re-borrowed per entry -- `set_layer_surface_output`
        // below runs inside the loop, alongside a mutable borrow of
        // `self.layers`, which a live `&wlr::Runtime` borrowed from
        // `self.wayland` would conflict with. `Runtime` is cheap to clone
        // (an `Rc`-shaped handle), the same pattern `OutputHandler::new_output`
        // already uses ahead of its own per-output mutations.
        let runtime = self.wayland.runtime().cloned();
        for id in &orphaned {
            // Each entry re-homes to the output its own last-known
            // placement's frame center actually sits over (mirrors
            // `migrate_windows_from`'s per-window containment test) --
            // *not* a single pointer-derived survivor for the whole batch
            // (M2: an unrelated panel commit elsewhere no longer decides
            // where a status bar on a hot-removed display ends up). A
            // surface with no placement yet (`last_configured: None`,
            // never configured) falls back to the lowest surviving index,
            // same as `migrate_windows_from`'s own fallback.
            let geometry =
                self.layers
                    .get(id)
                    .and_then(|e| e.last_configured)
                    .map(|(w, h, x, y)| Rectangle {
                        x,
                        y,
                        width: w as i32,
                        height: h as i32,
                    });
            let hit = geometry.and_then(|g| {
                let cx = g.x + g.width / 2;
                let cy = g.y + g.height / 2;
                self.outputs
                    .iter()
                    .find(|(_, out)| out.geometry.contains(cx, cy))
                    .map(|(&idx, _)| idx)
            });
            let Some(survivor) = hit.or_else(|| self.outputs.keys().min().copied()) else {
                continue;
            };
            if let Some(entry) = self.layers.get_mut(id) {
                entry.output = survivor;
            }
            if let Some(rt) = &runtime
                && let Some(output_id) = self.wlr_output_id_for(survivor)
            {
                rt.set_layer_surface_output(*id, output_id);
            }
            self.configure_layer(*id);
        }
        self.arrange_layers();
    }

    /// The live `wlr::OutputId` behind this crate's own `u32` index, the
    /// reverse of `output_ids`' own direction (`wlr::OutputId -> u32`) --
    /// needed wherever a consumer must call back into the runtime by
    /// output rather than by this crate's index, e.g.
    /// `set_layer_surface_output`. `None` if `index` names no live output
    /// (already removed, or never inserted -- the `NO_OUTPUT` sentinel,
    /// say). A linear scan over `output_ids`, which stays tiny (one entry
    /// per live output).
    fn wlr_output_id_for(&self, index: u32) -> Option<wlr::OutputId> {
        self.output_ids
            .iter()
            .find_map(|(&oid, &idx)| (idx == index).then_some(oid))
    }

    /// N9: react to `entry.interactive` changing on an already-mapped
    /// surface -- `layer_surface_commit`'s post-map counterpart to
    /// `layer_surface_mapped`'s at-map take-focus branch, which only ever
    /// runs once (at map) and so misses a surface that starts
    /// non-interactive and flips the flag on a later commit while already
    /// on screen (an auto-hide launcher's menu, say).
    ///
    /// Flipping to `true` while mapped and nothing else already holds
    /// layer focus (`layer_holds_keyboard_focus`, mirroring the guard
    /// `sync_seat_focus` itself reads) takes it: `layer_focus` is recorded
    /// success-gated on the actual grab, exactly like
    /// `layer_surface_mapped`'s own take-focus branch -- an unconditional
    /// record ahead of the runtime call (this method's first cut) is the
    /// J3/Important-1 split-brain class reopened: a legitimate
    /// `focus_layer_keyboard` miss (no seat, a stale id, a null surface,
    /// or wlroots' own surface-mapped flag disagreeing with
    /// `LayerEntry::mapped` at commit time) would leave `layer_focus`
    /// claiming a focus the seat never actually held, and
    /// `sync_seat_focus`'s guard would then refuse every toplevel the
    /// keyboard with no self-heal until this id unmapped, was destroyed,
    /// or flipped `interactive` back off. With no `wlr::Runtime` attached
    /// (every unit test in this file), the grab is treated as taken --
    /// gating on `focus_layer_keyboard`'s own `Option` unconditionally (as
    /// if a live runtime were always present) would make this
    /// unobservable outside a running compositor.
    ///
    /// Flipping to `false` while this surface held focus releases it and
    /// hands the seat back to whatever the model says is focused
    /// (`sync_seat_focus`), the same hand-back `layer_surface_unmapped`
    /// and `layer_surface_destroyed` already perform.
    fn sync_layer_interactive_focus(
        &mut self,
        id: wlr::LayerSurfaceId,
        was_interactive: bool,
        now_interactive: bool,
    ) {
        if now_interactive && !was_interactive {
            let mapped = self.layers.get(&id).is_some_and(|e| e.mapped);
            if !mapped || layer_holds_keyboard_focus(self.layer_focus, &self.layers) {
                return;
            }
            // Review finding HIGH (task 8 re-review): `layer_focus` used to
            // be recorded unconditionally, ahead of the runtime call --
            // when `focus_layer_keyboard` legitimately misses (no seat, a
            // stale id, a null surface, or wlroots' own surface-mapped
            // flag disagreeing with `LayerEntry::mapped` at commit time),
            // the seat's *real* keyboard focus never moved, but
            // `layer_focus` claimed it had. `layer_holds_keyboard_focus`
            // then reported `true` and `sync_seat_focus`'s guard refused
            // every toplevel the keyboard, with no self-heal until this id
            // unmapped, was destroyed, or flipped `interactive` back off
            // -- the same split-brain class J3/Important-1 already closed
            // for map/unmap, reopened here. Success-gated exactly like
            // `layer_surface_mapped`'s own take-focus branch: with no
            // runtime attached (every unit test in this file), `took` is
            // `true` so the bookkeeping stays observable without a live
            // `wlr::Runtime`; with one attached, only an actual grab
            // records the claim. Through the focus helper, so taking the
            // keyboard off a composing text input hides its overlay.
            let took = match self.change_keyboard_focus(|rt| rt.focus_layer_keyboard(id)) {
                Some(took) => took.is_some(),
                None => true,
            };
            if took {
                self.layer_focus = Some(id);
            } else {
                // Finding 8, errors: mirrors `layer_surface_mapped`'s own
                // trace for the same failure -- a keyboard grab that
                // silently didn't take used to leave no clue why a
                // post-map-interactive panel never got input.
                tracing::debug!(
                    ?id,
                    "layer surface became interactive after map but did not take keyboard focus"
                );
            }
        } else if !now_interactive && was_interactive && self.layer_focus == Some(id) {
            self.layer_focus = None;
            self.sync_seat_focus();
        }
    }

    /// Hot-remove semantics: every window whose frame center sat inside the
    /// dead output's box (`dead`, captured by the caller before the entry
    /// left `self.outputs`) is moved onto the surviving output with the
    /// lowest index (clamped into its box, or centered if it does not fit
    /// -- see the centering branch's own comment for F, task 8) and
    /// re-synced. Called from `OutputHandler::destroyed`.
    ///
    /// With no surviving output, this is a deliberate no-op: there is
    /// nowhere to move a window to, and leaving geometry alone (rather than
    /// clamping into an empty rect, which would collapse every affected
    /// window to a single point) is the only choice that doesn't invent a
    /// placement nothing asked for.
    pub fn migrate_windows_from(&mut self, dead: Rectangle) {
        let Some(survivor_idx) = self.outputs.keys().min().copied() else {
            return;
        };
        let Some(survivor) = self.outputs.get(&survivor_idx).map(|o| o.geometry) else {
            return;
        };

        let affected: Vec<WindowId> = self
            .window_manager
            .windows()
            .filter(|w| {
                let cx = w.geometry.x + w.geometry.width / 2;
                let cy = w.geometry.y + w.geometry.height / 2;
                dead.contains(cx, cy)
            })
            .map(|w| w.id)
            .collect();

        for id in affected {
            let Some(w) = self.window_manager.get(id) else {
                continue;
            };
            let geometry = w.geometry;

            // Preserve the window's offset from the dead output's origin,
            // clamped so the frame fits inside the survivor's box. A frame
            // wider/taller than the survivor itself is centered instead
            // (task 8, F: a `max` bound below its `min` bound would panic
            // `clamp`, and pinning it to the survivor's own origin drew it
            // hard against one corner with all the overflow bleeding off a
            // single edge -- centering spreads the unavoidable overflow
            // symmetrically, which is what every other oversized-window
            // placement in this crate already does). The subtraction below
            // is total, not `clamp`ed -- an oversized frame legitimately
            // produces a negative offset, and there is no bound to violate.
            let new_x = if geometry.width >= survivor.width {
                survivor.x + (survivor.width - geometry.width) / 2
            } else {
                (survivor.x + (geometry.x - dead.x))
                    .clamp(survivor.x, survivor.x + survivor.width - geometry.width)
            };
            let new_y = if geometry.height >= survivor.height {
                survivor.y + (survivor.height - geometry.height) / 2
            } else {
                (survivor.y + (geometry.y - dead.y))
                    .clamp(survivor.y, survivor.y + survivor.height - geometry.height)
            };

            self.window_manager.set_geometry(
                id,
                Rectangle {
                    x: new_x,
                    y: new_y,
                    ..geometry
                },
            );
            self.sync_window_to_scene(id);
        }
        self.emit_pending();
    }

    /// Drain `window_manager.pending_events` onto `dbus_tx`. Must be called
    /// after every mutation of `window_manager` so subscribers observe it.
    ///
    /// Each event goes out as a `SeqEvent` carrying the `seq` its mutation
    /// produced (review finding I2); `apply_reloaded_config` pushes
    /// `apply_config`'s events through `WindowManager::push_event` for the
    /// same reason, so there is exactly one queue and one counter.
    pub fn emit_pending(&mut self) {
        for ev in self.window_manager.pending_events.drain(..) {
            let _ = self.dbus_tx.send(ev);
        }
    }

    /// Queue an event for the next `emit_pending()` drain without it having
    /// come from a `window_manager` mutation (e.g. `AltTabState`, which is
    /// driven by `alt_tab`, not `window_manager`).
    ///
    /// Task 11 re-review #4: this used to push straight onto
    /// `pending_events`, bypassing `WindowManager::bump()` -- every
    /// `AltTabState` went out without ever advancing `seq`, so
    /// `snapshot().seq` couldn't be used to detect that an alt-tab change
    /// had happened. Routed through `note_event()` (which bumps then
    /// returns the same `pending_events` vec) so it participates in the
    /// same sequence counter as every other event.
    pub fn emit(&mut self, ev: Event) {
        self.window_manager.push_event(ev);
    }

    // --- Model <-> Wayland reconciliation (review finding C1) ---
    //
    // Before this, `new_toplevel`'s `map_element(window, (0, 0), false)` was
    // the only call that ever positioned anything in the `Space`, and no
    // `send_configure`/`send_close` existed anywhere in the tree: every
    // client rendered at (0,0), was never told its size, and never learned
    // it had been asked to close, fullscreen, or maximize. The model, the
    // D-Bus surface, and the tests were all self-consistent -- it was the
    // model -> Wayland edge that was missing, and no single task owned it.
    //
    // The whole edge is these three functions plus their call sites: every
    // geometry/state mutation ends in `sync_window_to_scene`, every
    // workspace/visibility change ends in `sync_scene`, and every close goes
    // through `request_close`.

    /// Push model window `id`'s geometry, visibility, and xdg state out to its
    /// client: position its scene node at the model's position, hide it when
    /// the window isn't on the active workspace (review finding I1), and
    /// configure the toplevel with the model's size plus the
    /// maximized/fullscreen/activated states.
    ///
    /// Call this at the tail of *every* geometry or state mutation. A window
    /// with no backing client is a silent no-op.
    ///
    /// Coordinate spaces (re-review minor 3, carried over verbatim from the
    /// smithay implementation because the invariant is unchanged): the model
    /// (`window.rs`, `window_at`, `decoration::hit_test`, drag offsets) is in
    /// *frame* space; the scene node's position and the staged size are in
    /// *content* space -- for an SSD window they differ by `TITLE_BAR_HEIGHT`.
    /// Any consumer of scene coordinates (pointer hit-testing into client
    /// surfaces, above all) must convert with `decoration::content_rect`,
    /// never compare the two spaces directly.
    ///
    /// Note (re-review minor 4): the early return means the trailing
    /// `sync_seat_focus` only runs for windows that still have a client.
    /// Focus-clearing on removal paths must not rely on this tail --
    /// `forget_window` calls `sync_focus_change(None)` explicitly for exactly
    /// that reason.
    pub fn sync_window_to_scene(&mut self, id: WindowId) {
        // While the session is locked, the crate already refuses normal
        // toplevel focus and routes input to lock surfaces only (see
        // `session_locked`'s doc); suspend this model's own layout/focus
        // reconciliation entirely rather than race that refusal.
        if self.session_locked {
            return;
        }
        if !self.wayland.is_backed(id) {
            return;
        }
        let Some(w) = self.window_manager.get(id) else {
            // The model window is gone but the client is still here (a close
            // that raced the client's destroy): hide it and drop the binding
            // rather than showing a window nothing tracks.
            self.wayland.set_visible(id, false);
            self.wayland.forget(id);
            // The other teardown path that bypasses `forget_window`
            // (review finding M1): the row vanished without going through
            // it, so the raster it left behind has to be collected here.
            self.title_rasters.remove(&id);
            if self.ssd_hover.is_some_and(|(hid, _)| hid == id) {
                self.ssd_hover = None;
            }
            if self.ssd_press.is_some_and(|(hid, _)| hid == id) {
                self.ssd_press = None;
            }
            return;
        };
        let (geo, fullscreen, maximized, focused) =
            (w.geometry, w.fullscreen, w.maximized, w.focused);
        let minimized = w.minimized;
        let visible = self.window_manager.is_visible(w);
        // Re-review finding New-4: the model's geometry is the *frame*; an
        // SSD window's client owns only the band below the title bar.
        let ssd = crate::decoration::has_ssd(&w.app_id, w.client_decorations_requested, fullscreen);
        let content = crate::decoration::content_rect(geo, ssd);

        // Review finding I2: paint (or hide) the SSD decoration. `geo`,
        // not `content` -- `title_bar_rect` is frame-space, same as
        // `content_rect`'s input, and the two are computed from the same
        // `geo` precisely so they can never disagree about where the band
        // ends and the client's content begins.
        let bar = crate::decoration::title_bar_rect(geo);
        let bar_color = crate::render::hex_to_rgba(&self.config.appearance.palette.background);
        let button_colors = self.button_colors();
        // Only rasterize for a window that will actually show a band: a CSD
        // or fullscreen window's title is never drawn, and shaping it would
        // be pure waste on the most common client kind there is.
        let decorated = ssd && visible;
        // All three button rects share one size (`BUTTON_WIDTH x
        // TITLE_BAR_HEIGHT`); index 0 stands in for the cell every glyph is
        // rasterized into.
        let button_cell = crate::decoration::button_rects(bar)[0];
        let (gw, gh) = (button_cell.width.max(1), button_cell.height.max(1));
        if decorated {
            let title = w.title.clone();
            self.ensure_title_raster(id, title, bar, focused);
            self.ensure_button_glyphs(gw, gh);
        }
        // Whichever button this window currently shows pressed, or failing
        // that hovered -- a press implies the pointer is over that same
        // button, so it always takes precedence when both happen to be set.
        let pressed = self.ssd_press.filter(|(hid, _)| *hid == id);
        let active_button = pressed
            .or_else(|| self.ssd_hover.filter(|(hid, _)| *hid == id))
            .map(|(_, idx)| {
                let state = if pressed.is_some() {
                    crate::wayland::ButtonState::Pressed
                } else {
                    crate::wayland::ButtonState::Hover
                };
                (idx, state)
            });
        // Field-wise borrow so the seam can read the memos' pixels in place:
        // the alternative is cloning them into owned buffers on every sync
        // -- i.e. on every pointer motion of a drag -- to hand across a call
        // that, on a cache hit, will not even look at them (M2).
        let Self {
            wayland,
            title_rasters,
            button_glyph_cache,
            ..
        } = self;
        let title_px = decorated
            .then(|| title_rasters.get(&id))
            .flatten()
            .and_then(|entry| {
                entry
                    .pixels
                    .as_ref()
                    .map(|(width, height, pixels)| crate::wayland::TitleRaster {
                        width: *width,
                        height: *height,
                        generation: entry.generation,
                        pixels,
                    })
            });
        let button_glyph_px: [Option<crate::wayland::GlyphRaster<'_>>; 3] =
            std::array::from_fn(|i| {
                if !decorated {
                    return None;
                }
                let key = (i, gw, gh, BUTTON_GLYPH_FG);
                button_glyph_cache
                    .get(&key)
                    .and_then(|entry| entry.as_ref())
                    .map(|pixels| crate::wayland::GlyphRaster {
                        width: gw,
                        height: gh,
                        fg: BUTTON_GLYPH_FG,
                        pixels,
                    })
            });
        wayland.sync_ssd(
            id,
            ssd,
            visible,
            bar,
            content,
            bar_color,
            button_colors,
            title_px,
            button_glyph_px,
            active_button,
        );

        self.wayland.set_visible(id, visible);
        // Reflect the minimized flag to the client (X11 `_NET_WM_STATE_HIDDEN`;
        // a no-op for xdg, whose hide is carried entirely by `set_visible`), so
        // a WM-initiated minimize is observable to an X11 app, not just a scene
        // hide the client never learns about.
        self.wayland.set_minimized(id, minimized);
        if visible {
            self.wayland.set_position(id, content.x, content.y);
            // Ledger item 28 / recommendation 4: `behavior.raise_on_focus`
            // decides whether a focus change alone may restack. With it off a
            // focused window is still activated and configured, it just keeps
            // its place in the stack.
            //
            // LEDGER DECISION (task 8): the pre-port smithay code also raised
            // on *any* geometry change, with no `raise_on_focus` check at
            // all -- `Space::map_element` was both the move and the raise in
            // one call, so every move restacked whether or not the mover was
            // the focused window. That does not return here. It was an
            // artifact of `map_element`'s API shape, not a documented
            // behavior the seam owes: a pure move not restacking is the
            // better floating-WM behavior, and it is what `raise_on_focus`'s
            // own name promises -- a knob that says "raise on focus," not
            // "raise on focus or move." Reintroducing moved-implies-raise
            // would make the knob lie for drag-move, which is by far the most
            // common source of geometry changes.
            if focused && self.config.behavior.raise_on_focus {
                self.wayland.raise(id);
            }
        }
        self.wayland
            .configure(id, content, focused && visible, maximized, fullscreen);

        // Keyboard focus is the other half of "focus reached the client".
        self.sync_seat_focus();

        // A window that moved or resized moved its popups' constraint box
        // with it. Reactive popups asked to be re-placed when that happens;
        // the rest are left where the client put them.
        self.reconstrain_popups(PopupRoot::Window(id));
    }

    /// The three title-bar button colors, in `decoration::button_rects`'
    /// order: minimize, maximize, close.
    ///
    /// Close is the palette's accent -- it is the destructive one and the
    /// one a user aims at without looking. The other two are the foreground
    /// color at 40% alpha, which reads as a subdued chip against the band
    /// whatever the palette is, without inventing two more color names the
    /// config has no field for.
    ///
    /// Premultiplied, like every other color that reaches a scene node: the
    /// wlroots scene graph composites premultiplied alpha, so scaling only
    /// the alpha channel and leaving RGB at full strength would paint the
    /// chips brighter than opaque foreground rather than fainter.
    fn button_colors(&self) -> [[f32; 4]; 3] {
        let fg = crate::render::hex_to_rgba(&self.config.appearance.palette.foreground);
        let accent = crate::render::hex_to_rgba(&self.config.appearance.palette.accent);
        let chip = premultiply(fg, 0.4);
        [chip, chip, accent]
    }

    /// Make sure window `id`'s memoized title raster matches `title`, the
    /// band `bar` and the focus state -- shaping only when one of those
    /// actually changed.
    ///
    /// Memoized through `title_rasters`: see that field's doc for why a drag
    /// must not re-shape (or re-upload) the same string sixty times a
    /// second, and why a negative result is cached too. The caller reads the
    /// pixels back out of the map rather than taking them from here, which
    /// is what keeps a cache hit free of any copy at all.
    ///
    /// The title node spans the band minus the three buttons, so a long
    /// title runs out of room before it runs under the close button rather
    /// than being drawn beneath it.
    fn ensure_title_raster(&mut self, id: WindowId, title: String, bar: Rectangle, focused: bool) {
        // Finding 3, security: cap the title before it becomes the cache
        // key, not just before it reaches `rasterize_title` -- otherwise
        // two distinct multi-kilobyte titles that agree on their first
        // `MAX_TITLE_BYTES` bytes would shape identically but still cache
        // (and compare) as different keys, defeating the point of capping.
        // `rasterize_title` caps again internally (its own contract, kept
        // independent of this caller), so this is belt and suspenders on
        // the same bound, not two different ones.
        let title = if title.len() > crate::text::MAX_TITLE_BYTES {
            crate::text::cap_title(&title).to_owned()
        } else {
            title
        };
        let width = (bar.width - 3 * crate::decoration::BUTTON_WIDTH).max(1);
        let height = bar.height.max(1);

        // Unfocused windows get the same text at 60% strength rather than a
        // second palette color: the config has one foreground, and dimming
        // it is the least surprising way to say "this window is not the
        // active one" in a title bar that is otherwise identical.
        //
        // Dimmed via `fg`'s alpha byte, not its RGB channels: `text::
        // rasterize_title` folds `fg[3]` back into the glyph coverage itself
        // (M3), so a reduced alpha here really does draw a translucent
        // glyph instead of the RGB-blend approximation M2 needed while that
        // premultiply ignored `fg[3]`.
        let palette = crate::render::hex_to_rgba(&self.config.appearance.palette.foreground);
        let to_u8 = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        let alpha = if focused { 255 } else { (255.0 * 0.6) as u8 };
        let fg = [
            to_u8(palette[0]),
            to_u8(palette[1]),
            to_u8(palette[2]),
            alpha,
        ];

        let key: TitleRasterKey = (title, width, fg);
        if self
            .title_rasters
            .get(&id)
            .is_some_and(|entry| entry.key == key)
        {
            return;
        }

        self.fonts.get_or_init(|| {
            (
                cosmic_text::FontSystem::new(),
                cosmic_text::SwashCache::new(),
            )
        });
        // `get_mut` cannot miss after `get_or_init`; the `else` is the
        // panic-free spelling of that, not a case with any behavior of its
        // own.
        let Some((fonts, swash)) = self.fonts.get_mut() else {
            return;
        };
        let pixels =
            crate::text::rasterize_title(fonts, swash, &key.0, width, height, TITLE_PAD_X, fg)
                .map(|pixels| (width, height, pixels));
        // A fresh generation for fresh pixels: this is what tells the seam
        // its title node is out of date (M2). Bumped on a negative result
        // too -- "the title now shapes to nothing" is a change the seam has
        // to act on, by dropping the node.
        let generation = self.next_title_generation;
        self.next_title_generation += 1;
        self.title_rasters.insert(
            id,
            TitleRasterEntry {
                key,
                generation,
                pixels,
            },
        );
    }

    /// Make sure every `BUTTON_GLYPHS` entry has a rasterized `width x
    /// height` @ `BUTTON_GLYPH_FG` entry in `button_glyph_cache`, shaping
    /// only the ones that are missing.
    ///
    /// No per-window key, unlike `ensure_title_raster`: a button glyph's
    /// pixels depend only on its index, the cell size, and the (constant)
    /// glyph color, none of which vary per window -- so the cache this fills
    /// is shared by every decorated window in the compositor, and steady
    /// -state calls here are a `contains_key` check per index, not a shape.
    fn ensure_button_glyphs(&mut self, width: i32, height: i32) {
        for (i, glyph) in BUTTON_GLYPHS.iter().enumerate() {
            let key = (i, width, height, BUTTON_GLYPH_FG);
            if self.button_glyph_cache.contains_key(&key) {
                continue;
            }
            self.fonts.get_or_init(|| {
                (
                    cosmic_text::FontSystem::new(),
                    cosmic_text::SwashCache::new(),
                )
            });
            // Same panic-free spelling as `ensure_title_raster`'s: `get_mut`
            // cannot miss right after `get_or_init`.
            let Some((fonts, swash)) = self.fonts.get_mut() else {
                return;
            };
            let pixels =
                crate::text::rasterize_glyph(fonts, swash, glyph, width, height, BUTTON_GLYPH_FG);
            self.button_glyph_cache.insert(key, pixels);
        }
    }

    /// The active workspace's focused window id, if any. Callers capture this
    /// *before* a focus-changing mutation and hand it to `sync_focus_change`.
    fn focused_id(&self) -> Option<WindowId> {
        self.window_manager.focused_window().map(|w| w.id)
    }

    /// Point the seat's keyboard at whatever the model currently says is
    /// focused -- or at nothing, when it says nothing is.
    ///
    /// Re-review finding New-3: seat keyboard focus used to be set-only, so
    /// after the last window on a workspace was closed, minimized, or left
    /// behind by a workspace switch, the seat kept pointing at a surface the
    /// user could no longer see. Driving the target from the model -- not from
    /// whichever window a caller happens to be syncing -- makes this
    /// idempotent, which is why `sync_window_to_scene` can call it
    /// unconditionally. The library compares against the seat's current focus
    /// and does nothing when it already matches, so an ordinary geometry sync
    /// does not churn leave/enter pairs at the client.
    fn sync_seat_focus(&mut self) {
        // See `session_locked`'s doc: don't reassert toplevel/layer keyboard
        // focus the crate is already refusing while locked.
        if self.session_locked {
            return;
        }
        // Focus rule 1 (contract §2.2): while an explicit seat grab is up --
        // an xdg-popup grab, a drag-and-drop grab -- the seat's keyboard is
        // wlroots' to route, not the model's. wlroots gives the popup chain
        // the keyboard itself and restores the pre-grab focus when the grab
        // ends, and every `wlr_seat_keyboard_notify_enter` this method would
        // make is routed *through* that grab anyway. Reasserting the model's
        // toplevel focus here would be the same churn the `layer_focus` guard
        // below already exists to prevent, one grab further out.
        if self.runtime_has_explicit_grab() {
            return;
        }
        // Review finding J3: this used to be unconditional, which meant
        // *any* `sync_window_to_scene`-driven resync -- a drag-motion
        // frame, a title change, a workspace switch, `arrange_layers`
        // itself -- silently reasserted the model's toplevel focus over
        // an interactive layer surface's, because the crate's seat model
        // has no separate "layer focus" slot: the next call to
        // `focus_toplevel_keyboard`/`clear_keyboard_focus` (which this
        // method's tail makes, via `wayland.keyboard_focus`) replaces
        // whatever `focus_layer_keyboard` last gave a layer surface. So a
        // keyboard-interactive panel lost the keyboard within one frame
        // of being given it. While `layer_focus` names a still-mapped
        // entry, this leaves the seat's real focus alone entirely; only
        // the paths that actually drop layer focus
        // (`layer_surface_unmapped`/`destroyed`, plus
        // `release_layer_focus` -- see re-review finding Important-1)
        // clear the field first, which is what lets a later call here
        // reach the toplevel branch again. `layer_holds_keyboard_focus`'s
        // own unit tests exercise this guard's exact logic directly;
        // there is no test that observes its effect on a real seat's
        // keyboard focus end-to-end, for a harness limitation documented
        // on `arrange_layers_leaves_layer_focus_alone_and_unmap_clears_it`
        // (this file's `mod tests`).
        if layer_holds_keyboard_focus(self.layer_focus, &self.layers) {
            return;
        }
        // An override-redirect pop-up that took the keyboard (a keyboard-
        // navigable X11 menu) keeps it while mapped, for the same reason a
        // layer surface does: the crate's seat has one keyboard-focus slot, so
        // reasserting the model's toplevel focus here would yank the keyboard
        // out from under the menu within a frame. The top of the OR keyboard
        // stack is the current holder; the OR unmap/destroy/flip paths pop it
        // (handing the keyboard to the parent menu beneath, or, when the stack
        // empties, calling this so the model's focus is restored).
        if self
            .or_keyboard_stack
            .last()
            .is_some_and(|id| self.override_redirect.contains_key(id))
        {
            return;
        }
        let focused = self
            .focused_id()
            .filter(|&id| self.window_manager.is_visible_id(id))
            .filter(|&id| self.wayland.is_backed(id));
        // Through the focus helper, not `keyboard_focus` directly: a focus
        // change away from a composing text input deactivates the IME with
        // no hook fired, and the overlay must hide with it.
        let key = self.wayland.focus_key(focused);
        self.change_keyboard_focus(|rt| crate::wayland::Wayland::apply_focus_key(rt, key));
    }

    /// Drop `layer_focus`, if held, so the next `sync_seat_focus` reaches
    /// its toplevel branch instead of being blocked by
    /// `layer_holds_keyboard_focus`'s guard.
    ///
    /// Re-review finding Important-1: round 1's `sync_seat_focus` guard
    /// closed the *passive* leak (`arrange_layers` and any other
    /// `sync_window_to_scene`-driven resync must never steal focus from a
    /// mapped interactive layer surface -- that must stay closed, and
    /// still is: nothing below calls this), but implemented only the
    /// guard's first clause. Missing was its second: *something* has to
    /// clear `layer_focus` when a toplevel focus is genuinely,
    /// user-, D-Bus-, or alt-tab-intentionally being asserted over it,
    /// or no toplevel can ever take the keyboard again once any
    /// interactive layer surface has ever mapped. Called at the top of
    /// every EXPLICIT toplevel-focus assertion -- `handle_pointer_press`
    /// (click-to-focus), `DbCommand::Focus`, and the `cycle:alt_tab`
    /// action's per-step focus -- each of which calls
    /// `sync_focus_change`/`sync_seat_focus` of its own accord shortly
    /// after, which is what actually pushes the toplevel focus out; this
    /// method itself makes no seat call; it only clears the field the
    /// guard reads.
    fn release_layer_focus(&mut self) {
        self.layer_focus = None;
    }

    /// Reconcile a focus transition all the way out to the clients.
    /// `previous` is what `focused_id()` returned *before* the model
    /// mutation; pass `None` when the previously focused window is already
    /// gone from the model (a close/destroy).
    ///
    /// Re-review findings New-1 and New-2. New-1: every focus path synced
    /// only the window that *gained* focus, so the one that lost it kept its
    /// `Activated` xdg state and its client went on rendering itself as the
    /// active window -- moving focus A -> B left two windows looking focused.
    /// New-2: when the focused toplevel was destroyed the model picked a
    /// successor (`forget_window` -> `focus_mru_in_workspace`) that was never
    /// synced, so the successor's client was never activated and never
    /// received seat keyboard focus -- the keyboard was dead until the user
    /// clicked something. Both are the same missing step: a focus change has
    /// two ends, and both have to be pushed.
    ///
    /// Wayland-side only: the model mutation that moved focus already emitted
    /// its own `WindowUpdated` events (`WindowManager::focus` emits for the
    /// window losing focus as well as the one gaining it), so nothing here
    /// emits.
    fn sync_focus_change(&mut self, previous: Option<WindowId>) {
        let current = self.focused_id();
        if let Some(prev) = previous.filter(|prev| Some(*prev) != current) {
            self.sync_window_to_scene(prev);
        }
        match current {
            Some(id) => self.sync_window_to_scene(id),
            // Nothing focused: `sync_window_to_scene` isn't reached at all,
            // so clear the seat here (New-3).
            None => self.sync_seat_focus(),
        }
    }

    /// Minimize or restore window `id`, handing focus to the workspace's MRU
    /// successor when the focused window is the one being minimized, then
    /// reconcile both ends of the transition out to the clients.
    ///
    /// Re-review Important 1: `DbCommand::Minimize` mutated the model and
    /// synced only `id`, bypassing the focus-transition path entirely --
    /// minimizing the focused window from the taskbar (the common entry
    /// point) left the model's focus on a now-invisible window, activated no
    /// successor, and `sync_seat_focus`'s visibility filter then cleared the
    /// seat: a dead keyboard with windows still on screen, New-2's symptom
    /// through a different door. The title-bar button already did this
    /// correctly; this is that arm's body, extracted so "minimize a window"
    /// has exactly one implementation, matching the maximize/fullscreen
    /// handlers' pattern.
    fn set_minimized_and_reconcile(&mut self, id: WindowId, value: bool) -> Option<()> {
        let previous = self.focused_id();
        let target = self.window_manager.get(id)?;
        let workspace = target.workspace;
        let was_focused_on_own_workspace = target.focused;
        self.window_manager.set_minimized(id, value)?;
        if !value {
            // Finding F6: restoring a window is the user attending to it, so
            // it answers an attention hint even when the restore does not
            // move focus (the taskbar's un-minimize on a background window,
            // `DbCommand::Minimize(id, false)`, and an X11 client's own
            // unminimize all land here). `focus` below clears it too when the
            // restore does take the keyboard; `clear_attention` is a no-op
            // when there was no hint.
            self.window_manager.clear_attention(id);
        }
        if value && was_focused_on_own_workspace {
            // Task 6: resolve `id`'s *own* workspace, not whichever one is
            // currently active -- `DbCommand::Minimize` can target a window
            // on an inactive workspace, and `focused_id()`/`previous` above
            // only ever reflects the active one. `refocus_after_hide` clears
            // the pointer outright when nothing qualifies, so a minimized
            // window can never linger as that workspace's `focused_window`
            // (masked only at the seat by `sync_seat_focus`'s visibility
            // filter until now).
            self.window_manager.refocus_after_hide(workspace);
        }
        // `id` itself always needs a sync (its visibility just changed),
        // even when it wasn't the focused window; `sync_focus_change` then
        // handles the successor and the seat (New-1/2/3). On restore, if the
        // model's focus pointer never left `id`, the visibility filter now
        // passes again and this same pair re-activates it and returns it the
        // keyboard.
        self.sync_window_to_scene(id);
        self.sync_focus_change(previous);
        // Drain the queued `WindowUpdated { minimized }` (and any focus-change
        // events) to D-Bus here, exactly as `set_maximized_target`/
        // `set_fullscreen_target` do. `DbCommand::Minimize` also flushes via
        // `handle_command`'s tail (a harmless second drain), but the
        // `wlr::XwaylandHandler::xwayland_request_minimize` caller does *not*
        // flow through `handle_command`, so without this an X11 app's
        // self-minimize would update the model, scene and `_NET_WM_STATE_HIDDEN`
        // yet never tell a subscribed taskbar/panel the window minimized.
        self.emit_pending();
        Some(())
    }

    /// Reconcile every model window with its client at once. Used after
    /// changes that can alter many windows' visibility in one go (workspace
    /// switch, config reload).
    ///
    /// Per-window syncing covers the focus transition's two ends on its own
    /// (every window is synced, including whichever one just lost focus), but
    /// the trailing `sync_seat_focus` is still needed for New-3's
    /// "switched to an empty workspace" case: with nothing focused *and*
    /// possibly nothing to iterate, the loop body may never run.
    pub fn sync_scene(&mut self) {
        let ids: Vec<WindowId> = self.window_manager.windows().map(|w| w.id).collect();
        for id in &ids {
            self.sync_window_to_scene(*id);
        }
        // Review finding #1: a managed X11 window's SSD nodes are `Band::Toplevel`
        // siblings that `sync_ssd` rebuilds at the *top* of the band on every
        // sync (unlike an xdg toplevel, whose decoration rides its own per-window
        // scene tree). Re-syncing every window therefore leaves whichever window
        // was synced last with its title bar above the others — including above
        // the focused window, whose own sync ran earlier in MRU order. Restack
        // the band to the model's stacking order once all nodes exist: raise each
        // visible window bottom-to-top (reverse MRU), which lifts its content and
        // decoration nodes together, so the focused window (MRU top) ends up on
        // top with its decoration intact and no window's title bar floats over a
        // neighbour. Gated on `raise_on_focus`: with restack-on-focus disabled
        // the user has asked us not to reorder the stack, so the band is left as
        // is (fully decoupling X11 SSD from stacking order in that non-default
        // mode is the per-window-subtree follow-up).
        if self.config.behavior.raise_on_focus {
            for id in self.stacking_order_bottom_to_top() {
                self.wayland.raise(id);
            }
        }
        self.sync_seat_focus();
    }

    /// The visible windows in bottom-to-top stacking order — reverse focus-MRU,
    /// since [`WindowManager::windows`] yields most-recently-focused first (the
    /// top of the stack). This is the order [`sync_scene`](Self::sync_scene)
    /// re-raises them in so each window's content and SSD nodes end up stacked in
    /// MRU order, with the focused window on top (review finding #1). Minimized
    /// and off-workspace windows are skipped — raising a hidden window would
    /// churn its X restack for no visible effect.
    fn stacking_order_bottom_to_top(&self) -> Vec<WindowId> {
        // Reuse the one visibility+ordering predicate (`visible_windows`, which
        // exists so this filter lives in exactly one place — review finding I1);
        // it yields MRU order (top first), so reverse for bottom-to-top.
        let mut ids: Vec<WindowId> = self
            .window_manager
            .visible_windows()
            .iter()
            .map(|w| w.id)
            .collect();
        ids.reverse();
        ids
    }

    /// Ask window `id` to close.
    pub fn request_close(&mut self, id: WindowId) {
        if self.wayland.close(id) {
            // A real client: the model row stays until the client actually
            // destroys its toplevel, which lands in `forget_toplevel`.
            return;
        }
        self.wayland.forget(id);
        self.forget_window(id);
        self.emit_pending();
    }

    /// Drop every trace of `id` from the model and its side tables, then
    /// hand focus to whatever is left on the active workspace (review
    /// finding I6's coherence requirement, applied to closes as well as
    /// moves: an action right after a close must not no-op on a dangling
    /// focus pointer).
    fn forget_window(&mut self, id: WindowId) {
        self.wayland.forget(id);
        self.window_manager.remove_window(id);
        self.fullscreen_saved_geometry.remove(&id);
        self.snap_saved_geometry.remove(&id);
        self.maximized_saved_geometry.remove(&id);
        // The scene nodes went with `wayland.forget` above; this is their
        // CPU-side memo, and a window's title pixels must not outlive it
        // (ids are never reused, so a stale entry would simply leak).
        self.title_rasters.remove(&id);
        // Same reasoning: a hover/press pointed at a window that no longer
        // exists must not linger to be read by some later window's sync.
        if self.ssd_hover.is_some_and(|(hid, _)| hid == id) {
            self.ssd_hover = None;
        }
        if self.ssd_press.is_some_and(|(hid, _)| hid == id) {
            self.ssd_press = None;
        }
        if self.window_manager.focused_window().is_none() {
            let active = self.window_manager.active_workspace();
            self.window_manager.refocus_after_hide(active);
        }
        // Re-review finding New-2: picking a successor in the model was only
        // half the job -- it also has to reach the client (`Activated`) and
        // the seat (keyboard focus), and when the workspace has no successor
        // left the seat's focus has to be cleared rather than left pointing
        // at the destroyed surface. `previous` is `None` because `id` is
        // already out of the model by this point, so there is nothing left
        // to sync on the losing end.
        self.sync_focus_change(None);
    }

    /// Remove the managed X11 (Xwayland) window backing surface `id`, if any.
    ///
    /// The Xwayland counterpart of the xdg `forget_toplevel` path: drops the
    /// side-table binding and takes the model row down through the shared
    /// `forget_window` (which reseats focus and cleans up rasters/SSD/scene
    /// memos). Idempotent -- a miss on the side-table returns without touching
    /// the model -- so an unmap followed by a destroy (both fire for the same
    /// surface) and an id we were never told about are each harmless.
    fn remove_xwayland_window(&mut self, id: wlr::XwaylandSurfaceId) {
        let Some(window_id) = self.xwayland_windows.remove(&id) else {
            return;
        };
        self.forget_window(window_id);
        self.emit_pending();
    }

    /// Add a first-mapping managed (non-override-redirect) X11 surface to the
    /// `WindowManager` model and drive it out to the client through the shared
    /// path. Shared by the ordinary map and the OR→managed runtime flip.
    ///
    /// `add_window` autofocuses, so the outgoing focus is captured first, then
    /// `sync_window_to_scene` builds the SSD, positions the scene node,
    /// configures the client to the SSD content rect, activates it and reseats
    /// keyboard focus — exactly as it does for an xdg toplevel. Placement comes
    /// from [`place_managed_x11`](Self::place_managed_x11) (window-type / modal /
    /// transient-parent aware), not the plain cascade.
    fn add_managed_x11(&mut self, surface: &wlr::XwaylandSurface<'_>) {
        let sid = surface.id();
        let app_id = surface
            .class()
            .or_else(|| surface.instance())
            .unwrap_or_default();
        let title = surface.title().unwrap_or_default();
        let pid = surface.pid().unwrap_or(0);
        // X11 clients self-position, but a managed window is placed by the WM.
        // The client's requested size is its *content* (an X11 window has no
        // decorations), so convert it to a frame — reserving the SSD strip — via
        // the shared `frame_rect` inverse before placing, exactly as a later
        // self-configure does (review finding #2). Without this the frame was
        // set to the requested size and `sync_window_to_scene` then inset the
        // client 28px shorter than it asked, and the window jumped taller on its
        // first `ConfigureRequest`. Falls back to the placeholder content size.
        let g = surface.geometry();
        let (content_w, content_h) = if g.width > 0 && g.height > 0 {
            (g.width, g.height)
        } else {
            (PLACEHOLDER_SIZE.0, PLACEHOLDER_SIZE.1)
        };
        // Decorations are decided the same way `sync_window_to_scene` will for
        // this window (a fresh row has no client-decoration request yet, and a
        // just-mapped window is not fullscreen).
        let ssd = crate::decoration::has_ssd(&app_id, None, false);
        let content = Rectangle {
            x: 0,
            y: 0,
            width: content_w,
            height: content_h,
        };
        let frame_dims = crate::decoration::frame_rect(content, ssd);
        let geometry = self.place_managed_x11(surface, frame_dims.width, frame_dims.height);
        tracing::info!(?sid, %app_id, %title, pid, ?geometry, "managed X11 window mapped");
        let previous = self.focused_id();
        let window_id = self
            .window_manager
            .add_window(&app_id, &title, pid, geometry);
        self.xwayland_windows.insert(sid, window_id);
        self.wayland.bind_x11(window_id, sid);
        self.sync_window_to_scene(window_id);
        self.sync_focus_change(previous);
        self.emit_pending();
    }

    /// Clamp a frame's top-left so the window — its SSD title bar included —
    /// stays reachable on an output, mirroring the map-time placement clamp
    /// (review finding #9). The output is the one the frame's own center falls
    /// on, falling back to the pointer's output and finally a default, so a
    /// self-configuring X11 client cannot drive its only drag handle off-screen.
    /// A frame already fully on its output is returned unchanged.
    fn clamp_frame_onto_output(&self, frame: Rectangle) -> Rectangle {
        // Clamp to the *usable* area, not the full output geometry: map-time
        // placement (`center_in_area`) and the whole placement convention keep a
        // window's frame — its SSD title bar included — clear of panels'
        // exclusive zones, so a self-configuring X11 client must land in the same
        // area or its only drag handle ends up under a top panel (review finding
        // #9's own goal). Output chosen by the frame's center, falling back to the
        // pointer's usable area and finally a default.
        let clamp_geo = self
            .output_for_window(frame)
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.usable)
            .or_else(|| self.usable_geo_for_pointer())
            .unwrap_or(Rectangle {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            });
        let (x, y) = Self::clamp_top_left(frame.x, frame.y, frame.width, frame.height, clamp_geo);
        Rectangle { x, y, ..frame }
    }

    /// Clamp a `width`×`height` box's top-left corner so the box stays within
    /// `clamp_geo`. The `.max(clamp_geo.x/y)` guards the degenerate case of a box
    /// larger than the area, where the upper clamp bound would otherwise fall
    /// below the lower. Shared by [`clamp_frame_onto_output`](Self::clamp_frame_onto_output)
    /// and [`center_in_area`](Self::center_in_area) so the two can never disagree.
    fn clamp_top_left(x: i32, y: i32, width: i32, height: i32, clamp_geo: Rectangle) -> (i32, i32) {
        let max_x = (clamp_geo.x + clamp_geo.width - width).max(clamp_geo.x);
        let max_y = (clamp_geo.y + clamp_geo.height - height).max(clamp_geo.y);
        (x.clamp(clamp_geo.x, max_x), y.clamp(clamp_geo.y, max_y))
    }

    /// Center a `width`×`height` frame inside `area`, then clamp its top-left so
    /// it stays on the *output that `area` sits on* — not the pointer's (review
    /// finding #6). Centering a dialog over a transient parent on another monitor
    /// must keep it on the parent's monitor; the old pointer-output clamp dragged
    /// it onto whichever screen the mouse happened to rest on. `fallback` (the
    /// pointer's usable area) is used only when `area`'s output cannot be found.
    fn center_in_area(
        &self,
        area: Rectangle,
        width: i32,
        height: i32,
        fallback: Rectangle,
    ) -> (i32, i32) {
        let clamp_geo = self
            .output_for_window(area)
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.usable)
            .unwrap_or(fallback);
        let x = area.x + (area.width - width) / 2;
        let y = area.y + (area.height - height) / 2;
        Self::clamp_top_left(x, y, width, height, clamp_geo)
    }

    /// Choose a managed X11 window's initial frame placement from its
    /// `_NET_WM_WINDOW_TYPE`, modal hint and transient parent (M3, task 4).
    ///
    /// A dialog — or any window carrying the modal hint — is centered over its
    /// transient parent when it has one that is mapped on the active workspace,
    /// so a "Save changes?" dialog lands over the document it belongs to; with
    /// no such parent it centers on the output. A splash screen centers on the
    /// output. Everything else (normal, utility, toolbar, …) cascades, exactly
    /// as a native xdg toplevel does. The returned rectangle is the *frame*,
    /// clamped so its top-left keeps the window on the output.
    fn place_managed_x11(
        &self,
        surface: &wlr::XwaylandSurface<'_>,
        width: i32,
        height: i32,
    ) -> Rectangle {
        let output_geo = self.usable_geo_for_pointer().unwrap_or(Rectangle {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        });
        let centered_in = |area: Rectangle| self.center_in_area(area, width, height, output_geo);
        // The transient parent's frame, if it is a managed X11 window mapped on
        // the active, visible workspace — the anchor a dialog centers over.
        let parent_geo = self
            .wayland
            .runtime()
            .and_then(|rt| rt.xwayland_surface_parent(surface.id()))
            .and_then(|pid| self.xwayland_windows.get(&pid).copied())
            .and_then(|wid| self.window_manager.get(wid))
            .filter(|w| w.workspace == self.window_manager.active_workspace() && !w.minimized)
            .map(|w| w.geometry);
        let (x, y) = match surface.window_type() {
            wlr::XwaylandWindowType::Dialog => centered_in(parent_geo.unwrap_or(output_geo)),
            wlr::XwaylandWindowType::Splash => centered_in(output_geo),
            _ if surface.is_modal() => centered_in(parent_geo.unwrap_or(output_geo)),
            _ => {
                let occupied: Vec<Rectangle> = self
                    .window_manager
                    .windows_in_workspace(self.window_manager.active_workspace())
                    .iter()
                    .map(|w| w.geometry)
                    .collect();
                layout::cascade_point_in(&occupied, (width, height), 24, output_geo)
            }
        };
        Rectangle {
            x,
            y,
            width,
            height,
        }
    }

    /// Track and place a mapped override-redirect pop-up (M3). Shared by the
    /// ordinary OR map and the managed→OR runtime flip.
    ///
    /// The pop-up's scene node — built by the crate on `associate` in
    /// `Band::Toplevel` — is lifted into `Band::Top` (above every managed
    /// toplevel, below the lock/overlay bands), positioned at the client's own
    /// coordinates, and raised so a newer pop-up sits over an older one. No SSD
    /// is ever attached. A focus-taking pop-up (a keyboard-navigable menu) is
    /// then handed the seat keyboard.
    fn map_override_redirect(&mut self, surface: &wlr::XwaylandSurface<'_>) {
        let sid = surface.id();
        let g = surface.geometry();
        let geometry = Rectangle {
            x: g.x,
            y: g.y,
            width: g.width.max(1),
            height: g.height.max(1),
        };
        let wants_focus = surface.override_redirect_wants_focus();
        tracing::info!(
            ?sid,
            ?geometry,
            wants_focus,
            "override-redirect X11 surface mapped"
        );
        if let Some(rt) = self.wayland.runtime() {
            rt.reparent_xwayland_surface_to_band(sid, wlr::Band::Top);
            rt.set_xwayland_surface_position(sid, geometry.x, geometry.y);
            rt.raise_xwayland_surface(sid);
        }
        self.override_redirect
            .insert(sid, OverrideRedirectSurface { geometry });
        if wants_focus {
            self.focus_override_redirect(sid);
        }
    }

    /// Hand the seat keyboard to a focus-taking OR pop-up and remember it holds
    /// it, so `sync_seat_focus` leaves it alone until it goes away.
    fn focus_override_redirect(&mut self, sid: wlr::XwaylandSurfaceId) {
        // Activation and the keyboard push go through the focus helper
        // together: either can move the seat off a composing text input.
        let focused = self.change_keyboard_focus(|rt| {
            rt.activate_xwayland_surface(sid, true);
            rt.focus_xwayland_surface_keyboard(sid)
        });
        if focused.is_some_and(|focused| focused.is_some()) {
            // Push onto the keyboard stack as the new top holder; move it up if
            // it was already present (a re-focus), so the stack never grows a
            // duplicate and `last()` always names the true current holder.
            self.or_keyboard_stack.retain(|&id| id != sid);
            self.or_keyboard_stack.push(sid);
        }
    }

    /// Drop a mapped OR pop-up from the side-table, if present, and reconcile the
    /// keyboard stack (review finding #7). Idempotent against an id that was never
    /// an OR surface (a managed window, or one we were never told about), so the
    /// unmap/destroy paths can call it blindly.
    ///
    /// If the closing pop-up was the current keyboard holder (the top of the
    /// stack), the keyboard returns to the parent menu still beneath it; only
    /// when the stack empties does `sync_seat_focus` fall through to the model's
    /// focused window. A pop-up closing from the *middle* of the stack (a parent
    /// dismissed while its submenu is still open) is simply spliced out, leaving
    /// the current holder untouched.
    fn remove_override_redirect(&mut self, sid: wlr::XwaylandSurfaceId) {
        if self.override_redirect.remove(&sid).is_none() {
            return;
        }
        let was_holder = self.or_keyboard_stack.last() == Some(&sid);
        self.or_keyboard_stack.retain(|&id| id != sid);
        if was_holder {
            if let Some(&parent) = self.or_keyboard_stack.last() {
                // Hand the keyboard back to the still-open parent menu.
                self.focus_override_redirect(parent);
            } else {
                // Stack empty: the guard in `sync_seat_focus` now falls through
                // to the model's focused window (or clears the seat if none).
                self.sync_seat_focus();
            }
        }
    }

    /// Toggle fullscreen state for a window. When entering fullscreen, saves the
    /// current geometry and sets geometry to the output rect. When exiting, restores
    /// the saved geometry. Emits WindowUpdated event.
    pub fn toggle_fullscreen(&mut self, id: WindowId) -> Option<()> {
        let w = self.window_manager.get(id)?;
        let target = !w.fullscreen;
        self.set_fullscreen_target(id, target)
    }

    /// Set fullscreen to an explicit `target` value, saving/restoring
    /// geometry exactly like `toggle_fullscreen` (which is now a thin
    /// `target = !current` wrapper around this). Added for task 12's
    /// `DbCommand::Fullscreen(id, toggle)` D-Bus command, whose `toggle`
    /// argument (despite the name -- it's the interface method's parameter
    /// name from the brief) is an explicit target state, not a flip
    /// request; a `FullscreenWindow(id, true)` call on an
    /// already-fullscreen window must stay a no-op rather than treating the
    /// current geometry as a fresh "pre-fullscreen" save point and
    /// clobbering the real one.
    pub fn set_fullscreen_target(&mut self, id: WindowId, target: bool) -> Option<()> {
        let w = self.window_manager.get(id)?;
        if w.fullscreen == target {
            return Some(());
        }
        // H1: resolved via the window's own frame center
        // (`output_for_window`), not the pointer -- this is reached from
        // client requests and `DbCommand::Fullscreen`, neither of which
        // correlates the pointer with the target window, so the old
        // `output_for_pointer` call fullscreened onto whatever output the
        // mouse happened to be resting on. `w.geometry` is read here,
        // before the borrow on `w` would otherwise need to outlive the
        // mutable calls below.
        let geometry = w.geometry;
        let output_geo = self
            .output_for_window(geometry)
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.geometry)?;
        self.window_manager.set_fullscreen(id, target)?;
        if target {
            // Save current geometry before entering fullscreen
            if let Some(w) = self.window_manager.get(id) {
                self.fullscreen_saved_geometry.insert(id, w.geometry);
            }
            self.window_manager.set_geometry(id, output_geo)?;
        } else {
            // Restore saved geometry when exiting fullscreen
            let saved = self.fullscreen_saved_geometry.remove(&id)?;
            self.window_manager.set_geometry(id, saved)?;
        }
        self.sync_window_to_scene(id);
        self.emit_pending();
        Some(())
    }

    /// Toggle maximized state for `id`.
    pub fn toggle_maximized(&mut self, id: WindowId) -> Option<()> {
        let target = !self.window_manager.get(id)?.maximized;
        self.set_maximized_target(id, target)
    }

    /// Set maximized to an explicit `target`, computing and applying real
    /// geometry (the output rect inset by `snap_gap`) with its own restore
    /// slot, mirroring `set_fullscreen_target`.
    ///
    /// Review finding I5: `toggle_maximized` used to flip a flag and emit,
    /// with no geometry and no configure -- the maximize button, the
    /// `MaximizeWindow` D-Bus method, and the client's own
    /// `xdg_toplevel.set_maximized` were all visual no-ops.
    pub fn set_maximized_target(&mut self, id: WindowId, target: bool) -> Option<()> {
        let w = self.window_manager.get(id)?;
        if w.maximized == target {
            return Some(());
        }
        // Maximize honors the usable area (panels' exclusive zones carve
        // into it); fullscreen, just below, deliberately keeps the full
        // `geometry` instead.
        //
        // H1: resolved via the window's own frame center
        // (`output_for_window`), not the pointer -- see
        // `set_fullscreen_target`'s identical fix just above for why.
        let geometry = w.geometry;
        let output_geo = self
            .output_for_window(geometry)
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.usable)?;
        self.window_manager.set_maximized(id, target)?;
        if target {
            let current = self.window_manager.get(id)?.geometry;
            self.maximized_saved_geometry.insert(id, current);
            let gap = self.config.appearance.snap_gap;
            self.window_manager
                .set_geometry(id, layout::maximized_geometry(output_geo, gap))?;
        } else if let Some(saved) = self.maximized_saved_geometry.remove(&id) {
            self.window_manager.set_geometry(id, saved)?;
        }
        self.sync_window_to_scene(id);
        self.emit_pending();
        Some(())
    }

    /// Apply a client-requested maximize/unmaximize for `toplevel`, reporting
    /// whether the model actually changed -- i.e. whether
    /// `sync_window_to_scene` will have sent the client a configure of its
    /// own. `false` (unknown toplevel, state already as requested, or a
    /// failed precondition such as no known output) tells the caller to
    /// answer with a bare configure instead, so the request is never left
    /// unanswered.
    ///
    /// Wired to `ToplevelHandler::request_maximize`.
    pub fn reconcile_maximized(
        &mut self,
        toplevel: crate::wayland::ToplevelKey,
        target: bool,
    ) -> bool {
        let Some(id) = self.wayland.window_for(toplevel) else {
            return false;
        };
        let changes = self
            .window_manager
            .get(id)
            .is_some_and(|w| w.maximized != target);
        changes && self.set_maximized_target(id, target).is_some()
    }

    /// Fullscreen counterpart of [`Self::reconcile_maximized`]; same
    /// "did a configure actually go out" contract. Wired to
    /// `ToplevelHandler::request_fullscreen`.
    pub fn reconcile_fullscreen(
        &mut self,
        toplevel: crate::wayland::ToplevelKey,
        target: bool,
    ) -> bool {
        let Some(id) = self.wayland.window_for(toplevel) else {
            return false;
        };
        let changes = self
            .window_manager
            .get(id)
            .is_some_and(|w| w.fullscreen != target);
        changes && self.set_fullscreen_target(id, target).is_some()
    }

    /// Synchronously reload the config from `self.config_path` (falling
    /// back to `icedtea_config::default_db_path()` when unset) and apply it.
    /// Returns the events `apply_config` produced (which already includes a
    /// trailing `Event::ConfigReloaded` -- see that method's doc -- so
    /// nothing is pushed again here).
    ///
    /// This is the *synchronous* load+apply path -- both the load and the
    /// apply happen on whatever thread calls it, in one blocking call. It
    /// exists for tests (task-13's Step-1 test calls it against a temp DB,
    /// where the extra thread hop of the async path would just be noise)
    /// and is *not* wired to either live reload trigger: per the module's
    /// threading requirement, the redb I/O + JSON parse must never run on
    /// the render loop, so neither `handle_command`'s `ReloadConfig` arm nor
    /// `apply_action`'s `"reload"` arm calls this directly -- they both go
    /// through `spawn_config_reload`'s worker-thread + channel path instead,
    /// which lands on `apply_reloaded_config` (the same apply+emit tail this
    /// method also uses) once the load finishes off-loop.
    ///
    /// NOTE (brief deviation): the brief's Step-3 sample pushed a *second*
    /// `Event::ConfigReloaded(self.config.appearance.clone())` after calling
    /// `apply_config`. `apply_config` (task 11) already appends its own
    /// `Event::ConfigReloaded(cfg.appearance.clone())` to the vec it
    /// returns, so following the sample literally would emit the signal
    /// twice per reload -- a genuine duplicate-event bug, not just
    /// transcription noise (the standing human ruling is to fix genuine
    /// sample bugs and document them here). Dropped; `apply_config`'s event
    /// is the one and only `ConfigReloaded` this path emits.
    pub fn reload_config_from_disk(&mut self) -> Vec<Event> {
        let path = self
            .config_path
            .clone()
            .unwrap_or_else(icedtea_config::default_db_path);
        let cfg = icedtea_config::load_or_default(&path);
        self.apply_reloaded_config(cfg)
    }

    /// Apply a `Config` (already loaded, whether synchronously by
    /// `reload_config_from_disk` or on a worker thread by
    /// `handle_command`'s `ReloadConfig` async path): calls `apply_config`,
    /// pushes its events onto the ordinary seq-tagged event queue, and flushes them
    /// immediately (via `emit_pending`) so the D-Bus emitter thread
    /// observes `ConfigReloaded` (and any `WindowClosed`/`WorkspaceList`/
    /// terminal `AltTabState` it carries) right away rather than waiting for
    /// some unrelated later mutation to flush the queue. Returns the same
    /// events, for callers (like `reload_config_from_disk`'s test) that want
    /// to inspect what was produced.
    pub fn apply_reloaded_config(&mut self, cfg: Config) -> Vec<Event> {
        // Captured before `apply_config` overwrites `self.config` -- this is
        // the only way to tell whether the wallpaper *path* changed rather
        // than merely being reapplied.
        let old_wallpaper = self.config.appearance.wallpaper.clone();

        let events = self.apply_config(cfg);
        // Review finding I2: these used to be extended onto a separate
        // `pending_config_events` vec that bypassed the sequence counter
        // entirely, so the reload's events went out with no seq of their
        // own. `apply_config`'s returned `events` (the summary
        // `WorkspaceList`/`ConfigReloaded`, plus a terminal `AltTabState` if
        // a cycle was in flight) are the ones without a seq yet; pushing them
        // through the ordinary queue gives each a real, monotonically
        // increasing seq after any per-window events `apply_config` already
        // queued (e.g. a workspace migration's `WindowUpdated`), preserving
        // their order.
        for ev in &events {
            self.window_manager.push_event(ev.clone());
        }

        // Task 14 gap-close: `apply_config` swaps `self.config` in but never
        // touched anything downstream of it -- the background rect kept the
        // old color, a wallpaper path change never re-decoded, and
        // (already covered above via `apply_config`'s own
        // `input::warn_about_keybindings` call) keybinding revalidation was
        // the one piece that already worked.
        if let Some(bg) = self.background
            && let Some(runtime) = self.wayland.runtime()
        {
            runtime.set_rect_color(bg, render::wallpaper_color(&self.config.appearance));
        }
        if self.config.appearance.wallpaper != old_wallpaper {
            // Carried-forward obligation (task 7 review): clear the decoded
            // image and tear down every wallpaper node *before* the fresh
            // decode can land, so `sync_wallpaper_nodes`'s existing-node
            // branch (which never refreshes pixels in place -- see its own
            // doc) is never asked to show stale pixels under a new path.
            self.wallpaper.set_decoded(None);
            self.sync_wallpaper_nodes();
            let path = self.config.appearance.wallpaper.clone();
            self.spawn_wallpaper(path);
        }
        // `apply_config` now preserves every window row across the reload, so
        // this re-sync repaints the surviving windows against the swapped-in
        // appearance (palette/bar colors) -- exactly the case this
        // unconditional call was left in place to cover.
        self.sync_scene();

        self.emit_pending();
        events
    }

    /// Spawn the off-loop worker thread shared by both async reload
    /// triggers -- `handle_command`'s `DbCommand::ReloadConfig` arm and
    /// `apply_action`'s `"reload"` arm (bound to `SUPER+SHIFT+r` by
    /// default). Loads `self.config_path` (falling back to
    /// `icedtea_config::default_db_path()`) on the worker thread and ships
    /// the result back over `config_reload_tx`; `drain_config_reload` picks
    /// it up on the next turn once it arrives. A no-op if `config_reload_tx`
    /// hasn't been wired yet (only possible before `set_config_reload_sender`
    /// has run).
    fn spawn_config_reload(&self) {
        let Some(tx) = self.config_reload_tx.clone() else {
            return;
        };
        let path = self
            .config_path
            .clone()
            .unwrap_or_else(icedtea_config::default_db_path);
        // `try_clone` dup(2)s the wake pipe's write half so the worker
        // thread owns a handle it can move across the `'static` bound
        // rather than borrowing `self`'s. `None` (no test wires this, and a
        // failed `try_clone` degrades the same way) means no wake is sent
        // and `drain_config_reload`'s periodic poll from `should_stop`
        // remains the only way the result is ever picked up -- exactly the
        // pre-wake-pipe behavior, not a regression.
        let wake = self
            .config_reload_wake
            .as_ref()
            .and_then(|w| w.try_clone().ok());
        let lock = Arc::clone(&self.config_db_lock);
        std::thread::spawn(move || {
            // `None` == persistently locked by another handle: send nothing so
            // the render loop keeps its current `self.config` untouched (review
            // finding #2), and skip the wake (nothing to drain). Only a real
            // load result is shipped and woken on.
            if let Some(cfg) = Self::load_config_locked(&lock, &path) {
                let _ = tx.send(cfg);
                if let Some(wake) = wake {
                    crate::backend::wake(&wake);
                }
            }
        });
    }

    /// Load the config from `path`, holding `lock` across the whole
    /// open+read+drop so no concurrent [`Self::save_config_to`] can have the
    /// same redb file open at the same moment (review finding #3). The lock is
    /// the ONLY thing serializing this process's two worker threads' opens; see
    /// `config_db_lock`'s doc.
    ///
    /// Returns `None` when the file is *persistently* locked by some OTHER
    /// handle -- the settings app, in production -- so the caller must KEEP its
    /// current `self.config` rather than adopt defaults (review finding #2). The
    /// intra-process `lock` cannot serialize against a different process, so a
    /// cross-process collision surfaces here as `LoadOutcome::Locked`; we retry
    /// a few times with a brief backoff (the settings app is documented to hold
    /// the DB open only momentarily) and only give up -- returning `None` -- if
    /// it stays locked. A genuinely missing/corrupt file is `Loaded(defaults)`,
    /// never `Locked`, so this only ever declines to load on a true lock, never
    /// masking real corruption.
    fn load_config_locked(lock: &Mutex<()>, path: &std::path::Path) -> Option<Config> {
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        // 4 tries * 25ms = up to ~75ms of backoff before giving up and keeping
        // the current config. Brief enough not to stall the reload worker
        // noticeably, long enough to ride out a momentary settings-app write.
        const ATTEMPTS: u32 = 4;
        const BACKOFF: std::time::Duration = std::time::Duration::from_millis(25);
        for attempt in 0..ATTEMPTS {
            match icedtea_config::try_load(path) {
                icedtea_config::LoadOutcome::Loaded(cfg) => return Some(cfg),
                icedtea_config::LoadOutcome::Locked => {
                    if attempt + 1 < ATTEMPTS {
                        std::thread::sleep(BACKOFF);
                    }
                }
            }
        }
        tracing::warn!(
            "config db stayed locked across {ATTEMPTS} attempts; keeping current config (no default wipe)"
        );
        None
    }

    /// Persist `self.config` to the redb file off the render loop, the write
    /// sibling of [`Self::spawn_config_reload`]. Called from
    /// `OutputHandler::output_configuration_applied` after it has upserted the
    /// applied heads into `config.displays`: the handler runs on the render
    /// thread underneath a wlroots `extern "C"` frame, so it must never block
    /// on redb I/O. A cloned `Config` and the db path move onto a worker
    /// thread that opens the database and writes; failures are logged, never
    /// propagated (a persist failure must not take down the compositor).
    ///
    /// Unlike the reload path this needs no channel back: nothing on the
    /// render loop consumes the result. `config_path` `None` (the default
    /// boot) resolves to `icedtea_config::default_db_path`, exactly as
    /// `spawn_config_reload` does.
    fn spawn_config_save(&self) {
        let config = self.config.clone();
        let path = self
            .config_path
            .clone()
            .unwrap_or_else(icedtea_config::default_db_path);
        let lock = Arc::clone(&self.config_db_lock);
        std::thread::spawn(move || Self::save_config_to(&lock, &config, &path));
    }

    /// The synchronous body [`Self::spawn_config_save`] runs on its worker
    /// thread: open the redb file and write `config`. Failures are logged,
    /// never propagated -- a persist failure must not take the compositor
    /// down. Split out so it can be exercised deterministically in a test
    /// (redb permits only one `Database` handle per file per process, so a
    /// test that polled the detached thread by re-opening the file would race
    /// the writer's own open; calling this directly does not).
    fn save_config_to(lock: &Mutex<()>, config: &Config, path: &std::path::Path) {
        // Hold the shared open-lock across the whole open + write + drop so a
        // concurrent `spawn_config_reload` worker cannot have this same redb
        // file open at the same moment (review finding #3): a second open of a
        // file redb already has open is `DatabaseAlreadyOpen`, which would turn
        // one of the two paths into a no-op (a dropped write) or a
        // default-config wipe (a failed load). See `config_db_lock`'s doc.
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        match icedtea_config::open(path) {
            Ok(db) => {
                // Scoped save (review findings #2/#5): the compositor writes ONLY
                // the displays table, never the appearance/keybindings/workspaces
                // sections the settings app owns. `Config::save` would rewrite the
                // whole file and revert a concurrent settings-app edit to those
                // sections with the compositor's stale in-memory copy.
                if let Err(err) = config.save_displays(&db) {
                    tracing::error!(%err, "failed to persist display config");
                }
            }
            Err(err) => tracing::error!(%err, "could not open config db to persist displays"),
        }
    }

    /// Apply a persisted [`icedtea_contract::DisplayConfig`] to a live output,
    /// committing mode, scale, transform, and layout position INDEPENDENTLY
    /// (review finding #7).
    ///
    /// Enabling is the one hard prerequisite -- an output that cannot be enabled
    /// cannot come up at all -- so its failure is the only one that propagates
    /// (via `?`), letting `new_output`'s caller fall back rather than leave the
    /// output dark. The mode, scale, transform, and position are each applied on
    /// their own and merely logged on failure: a persisted mode wlroots can no
    /// longer honor (a panel swapped, a link that renegotiated a different set
    /// of modes) must NOT throw away the still-valid scale/transform/position by
    /// short-circuiting the whole apply. A config with a zero width or height
    /// means "no explicit mode" -- keep the preferred one but still honor
    /// scale/transform/position.
    fn apply_display_config(
        runtime: &wlr::Runtime,
        output: &wlr::Output<'_>,
        cfg: &icedtea_contract::DisplayConfig,
    ) -> wlr::Result<()> {
        // A backend's outputs arrive disabled and modeless, and wlroots
        // requires an output to be *enabled* before a custom mode can be
        // committed (wlr_output.h: "The output needs to be enabled."). Enable
        // with a valid preferred baseline first, then override with the
        // persisted custom mode below. Two commits, but the output is enabled
        // for both. This is the only setter allowed to abort the apply.
        output.enable_with_preferred_mode()?;
        let name = output.name().unwrap_or_default();
        if cfg.width > 0
            && cfg.height > 0
            && let Err(err) = output.set_mode(cfg.width, cfg.height, cfg.refresh_mhz)
        {
            tracing::warn!(?err, %name, "persisted mode rejected; keeping preferred mode, still applying scale/transform/position");
        }
        // Guard the persisted/echoed-back scale exactly as the settings
        // head_rect path does (review finding #12): a `scale <= 0` or
        // non-finite value would otherwise reach wlroots and yield degenerate
        // geometry. Fall back to 1.0 for any unusable value.
        if let Err(err) = output.set_scale(Self::guarded_scale(cfg.scale)) {
            tracing::warn!(?err, %name, "persisted scale rejected");
        }
        if let Err(err) = output.set_transform(Self::transform_from_i32(cfg.transform)) {
            tracing::warn!(?err, %name, "persisted transform rejected");
        }
        // `set_output_position` returns `None` on a stale output id; log it and
        // move on rather than discarding the mode/scale/transform we just
        // applied.
        if runtime
            .set_output_position(output.id(), cfg.x, cfg.y)
            .is_none()
        {
            tracing::warn!(%name, "persisted layout position rejected (stale output id)");
        }
        Ok(())
    }

    /// The largest output scale the compositor will honour. Real HiDPI outputs
    /// top out around 3×; 16 is a generous ceiling that still keeps every
    /// derived quantity — notably `96 * scale` in [`export_x11_dpi`] and
    /// wlroots' scale-driven buffer sizing — far from any integer overflow, so a
    /// corrupted persisted `DisplayConfig.scale` or a hostile test value cannot
    /// blow up the DPI publish (review finding #8).
    const MAX_OUTPUT_SCALE: f64 = 16.0;

    /// Clamp a persisted/echoed-back display scale to something wlroots can
    /// use (review finding #12). A `DisplayConfig.scale` that is `<= 0` or
    /// non-finite (NaN/inf) would produce degenerate output geometry or push
    /// the whole apply into its error fallback; any such value falls back to
    /// `1.0`. An absurdly large but finite value (review finding #8) is capped
    /// at [`MAX_OUTPUT_SCALE`] rather than allowed to overflow downstream integer
    /// math. Mirrors the identical guard the settings `head_rect` path uses.
    fn guarded_scale(scale: f64) -> f32 {
        if scale.is_finite() && scale > 0.0 {
            scale.min(Self::MAX_OUTPUT_SCALE) as f32
        } else {
            1.0
        }
    }

    /// Map a persisted `DisplayConfig::transform` (the `wl_output_transform`
    /// integer, 0-7) to a [`wlr::Transform`]. Any out-of-range value falls
    /// back to `Normal`, matching wlroots' own read-back behavior.
    fn transform_from_i32(value: i32) -> wlr::Transform {
        match value {
            1 => wlr::Transform::R90,
            2 => wlr::Transform::R180,
            3 => wlr::Transform::R270,
            4 => wlr::Transform::Flipped,
            5 => wlr::Transform::Flipped90,
            6 => wlr::Transform::Flipped180,
            7 => wlr::Transform::Flipped270,
            _ => wlr::Transform::Normal,
        }
    }

    /// The inverse of [`Self::transform_from_i32`]: a committed
    /// [`wlr::Transform`] back to the `wl_output_transform` integer stored in
    /// a `DisplayConfig`.
    fn transform_to_i32(transform: wlr::Transform) -> i32 {
        match transform {
            wlr::Transform::Normal => 0,
            wlr::Transform::R90 => 1,
            wlr::Transform::R180 => 2,
            wlr::Transform::R270 => 3,
            wlr::Transform::Flipped => 4,
            wlr::Transform::Flipped90 => 5,
            wlr::Transform::Flipped180 => 6,
            wlr::Transform::Flipped270 => 7,
            // `wlr::Transform` (M5's `geom::Transform`) is `#[non_exhaustive]`;
            // any future variant maps to Normal (0) rather than blocking the build.
            _ => 0,
        }
    }

    /// Whether `new_output` honoring the currently-booting output's persisted
    /// "disabled" would strand the session at zero active outputs -- keyed off
    /// LIVE outputs, never the persisted enabled bools (review findings #1/#5).
    ///
    /// The earlier version of this guard read the persisted config
    /// (`config.displays.iter().any(|d| d.enabled)`). That is finding #1's bug:
    /// a persisted-enabled connector that is physically ABSENT (undocked laptop,
    /// unplugged external) still counted as "something else is enabled", so the
    /// only PRESENT output -- persisted disabled -- got honored and disabled,
    /// leaving zero active outputs and a black screen with no recovery. `self.
    /// outputs` is the set that has actually come up active this boot; if it is
    /// non-empty, another PRESENT output is already live and honoring the
    /// disable is safe. If it is empty, disabling would strand the session, so
    /// `new_output` force-enables this output instead.
    ///
    /// RESIDUAL LIMITATION: wlroots delivers `new_output` per connector during
    /// boot with no "enumeration complete" signal, and the backend exposes no
    /// list of present-but-not-yet-arrived outputs. So if a persisted-disabled
    /// connector enumerates BEFORE its persisted-enabled peer has come up,
    /// `self.outputs` is momentarily empty and this returns `true` -- the
    /// disabled output boots active. That is self-correcting: once the peer is
    /// live, an output-management apply can disable this one via
    /// `output_configuration_applied`'s own last-output guard. The invariant
    /// held unconditionally is the one that matters -- the compositor never
    /// lands at zero active outputs.
    fn boot_disable_would_strand(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Whether disabling the active output at `index` right now would strand
    /// the session at zero active outputs (review finding #5): true when no
    /// OTHER output is currently active AND no head in the same atomic apply
    /// (`batch_enables`) is enabling to replace it. `output_configuration_applied`'s
    /// disable branch refuses the disable when this returns true, keeping at
    /// least one output live.
    fn disable_would_strand_session(&self, index: u32, batch_enables: bool) -> bool {
        !batch_enables && !self.outputs.keys().any(|i| *i != index)
    }

    /// Remove and return the [`wlr::OutputId`] of the currently-disabled
    /// output whose connector `name` matches, if the match is UNAMBIGUOUS
    /// (review findings #13/#15). `disabled_outputs` is keyed by the unique id
    /// with the name as its value, so this is the name->id reverse lookup the
    /// re-enable path in `output_configuration_applied` needs. Keying by id (not
    /// name) is what keeps two unnamed disabled outputs -- both named `""` --
    /// from colliding under a single key and dropping one id permanently
    /// (finding #15, the write side).
    ///
    /// Finding #13 is the read side of that same hazard: if two disabled
    /// outputs share a connector name (both `""`, or a genuine duplicate), a
    /// plain `find` would rehydrate an ARBITRARY one -- `HashMap` iteration
    /// order is unspecified -- and could map the re-enabled head to the WRONG
    /// `OutputId`. So we require exactly one match: on two or more we refuse,
    /// log, and return `None` rather than guess.
    ///
    /// RESIDUAL LIMITATION: while two identically-named disabled outputs
    /// coexist, neither can be re-enabled through this name-keyed path (an
    /// `AppliedHead` carries only a name, no id). This is inherent -- there is
    /// no other key to disambiguate on -- and self-clears once one of them is
    /// unplugged (`destroyed` prunes it) or renamed. The output simply stays in
    /// `disabled_outputs` until then; it is never lost.
    fn take_disabled_output(&mut self, name: &str) -> Option<wlr::OutputId> {
        match Self::resolve_unique_disabled(&self.disabled_outputs, name) {
            DisabledLookup::Unique(oid) => {
                self.disabled_outputs.remove(&oid);
                Some(oid)
            }
            DisabledLookup::Ambiguous => {
                tracing::warn!(
                    name,
                    "ambiguous re-enable: multiple disabled outputs share this connector name; refusing to rehydrate to avoid mapping the wrong output id"
                );
                None
            }
            DisabledLookup::None => None,
        }
    }

    /// Pure name->id resolution behind [`Self::take_disabled_output`] (review
    /// finding #13): return the SOLE id whose stored name equals `name`, or a
    /// non-`Unique` verdict when there is no match or the match is ambiguous
    /// (two+ ids share the name). Generic over the id type so it can be unit
    /// tested without a [`wlr::OutputId`], which cannot be constructed outside
    /// the wlr crate.
    fn resolve_unique_disabled<K: Copy>(
        disabled: &HashMap<K, String>,
        name: &str,
    ) -> DisabledLookup<K> {
        let mut matches = disabled
            .iter()
            .filter(|(_, n)| n.as_str() == name)
            .map(|(id, _)| *id);
        let Some(first) = matches.next() else {
            return DisabledLookup::None;
        };
        if matches.next().is_some() {
            DisabledLookup::Ambiguous
        } else {
            DisabledLookup::Unique(first)
        }
    }

    /// Upsert each named [`wlr::AppliedHead`] into `self.config.displays`,
    /// keyed by connector name: an existing entry with the same name is
    /// replaced in place (preserving order), an unseen name is appended. This
    /// is the in-memory half of persistence; [`Self::spawn_config_save`]
    /// writes the result to redb. Heads with no name are skipped -- there is
    /// no stable key to store them under.
    fn upsert_displays_from_heads(&mut self, heads: &[wlr::AppliedHead]) {
        for head in heads {
            let Some(name) = head.name.as_deref() else {
                continue;
            };
            let entry = icedtea_contract::DisplayConfig {
                name: name.to_string(),
                enabled: head.enabled,
                width: head.width,
                height: head.height,
                refresh_mhz: head.refresh_mhz,
                x: head.x,
                y: head.y,
                scale: head.scale as f64,
                transform: Self::transform_to_i32(head.transform),
            };
            if let Some(existing) = self.config.displays.iter_mut().find(|d| d.name == name) {
                *existing = entry;
            } else {
                self.config.displays.push(entry);
            }
        }
    }

    /// Re-home any window whose frame center now sits outside every output's
    /// box -- the aftermath of an output shrinking under an applied
    /// output-management config. Each stranded window is clamped into the
    /// lowest-index (survivor) output, mirroring
    /// [`Self::migrate_windows_from`]'s placement. A no-op with no outputs.
    fn reclaim_offscreen_windows(&mut self) {
        let Some(survivor_idx) = self.outputs.keys().min().copied() else {
            return;
        };
        let Some(survivor) = self.outputs.get(&survivor_idx).map(|o| o.geometry) else {
            return;
        };

        let stranded: Vec<WindowId> = self
            .window_manager
            .windows()
            .filter(|w| {
                let cx = w.geometry.x + w.geometry.width / 2;
                let cy = w.geometry.y + w.geometry.height / 2;
                !self.outputs.values().any(|o| o.geometry.contains(cx, cy))
            })
            .map(|w| w.id)
            .collect();

        for id in stranded {
            let Some(w) = self.window_manager.get(id) else {
                continue;
            };
            let geometry = w.geometry;
            let new_x = if geometry.width >= survivor.width {
                survivor.x + (survivor.width - geometry.width) / 2
            } else {
                geometry
                    .x
                    .clamp(survivor.x, survivor.x + survivor.width - geometry.width)
            };
            let new_y = if geometry.height >= survivor.height {
                survivor.y + (survivor.height - geometry.height) / 2
            } else {
                geometry
                    .y
                    .clamp(survivor.y, survivor.y + survivor.height - geometry.height)
            };
            self.window_manager.set_geometry(
                id,
                Rectangle {
                    x: new_x,
                    y: new_y,
                    ..geometry
                },
            );
            self.sync_window_to_scene(id);
        }
        self.emit_pending();
    }

    /// Spawn the wallpaper decode worker and wire up the receiving end of
    /// its result channel. The one and only caller in a real boot is
    /// `run()`; tests that want to exercise the exact wake-pipe path a real
    /// boot takes call this too, rather than `render::spawn_wallpaper_decode`
    /// directly.
    ///
    /// Mirrors `spawn_config_reload`'s wake-pipe handling exactly (review
    /// finding C1): the worker thread is handed a `try_clone`d copy of
    /// `self.wallpaper_wake`, never the original. Before this fix, callers
    /// passed the wake pipe's only write half straight into the worker
    /// thread by value; it dropped the moment that (one-shot, exits after
    /// its single send) thread ended, leaving the registered read end with a
    /// permanent `EPOLLHUP` on libwayland's level-triggered loop for the
    /// rest of the process's life -- `fd_ready`'s wallpaper arm firing, and
    /// `drain_wallpaper` re-entering its (now also fixed, see M4)
    /// "disconnected" arm, on every single turn forever. Keeping the
    /// original alive on `State` (`set_wallpaper_wake`, called once at boot)
    /// closes that off the same way `config_reload_wake` already does for
    /// its own channel.
    pub fn spawn_wallpaper(&mut self, path: Option<String>) {
        let wake = self
            .wallpaper_wake
            .as_ref()
            .and_then(|w| w.try_clone().ok());
        let rx = crate::render::spawn_wallpaper_decode(path, wake);
        self.wallpaper_rx = Some(rx);
    }

    /// Central dispatcher for every [`crate::dbus::DbCommand`] received from
    /// the D-Bus service thread (see `dbus.rs`'s module doc for the
    /// threading model). Mirrors `apply_action`'s shape: `?`-propagates a
    /// failed precondition as `None` (skipping the trailing
    /// `emit_pending()`, which would be a no-op anyway since nothing queued
    /// on a failed mutation), `Some(())` on success.
    ///
    /// `ReloadConfig` is the one arm that doesn't mutate anything itself:
    /// per this task's threading requirement, the redb I/O + JSON parse
    /// must never block the render loop, so this spawns a worker thread
    /// (see `reload_config_from_disk`'s doc) that loads the config off-loop
    /// and ships the result back over `config_reload_tx`; `drain_config_reload`
    /// is what actually calls `apply_reloaded_config` once the load
    /// finishes. If `config_reload_tx` hasn't been wired yet, this is a
    /// no-op.
    /// The integer scale of the primary output (the lowest model index — the
    /// same output `outputs.keys().min()` picks for single-output placement),
    /// clamped to at least `1`. This is the scale the M4 HiDPI export applies to
    /// X11 uniformly: X11 has no per-window scale, so the compositor targets the
    /// single global (primary-output) scale, exactly as the design's HiDPI
    /// decision spells out. A fractional committed scale is rounded up to the
    /// next integer, since Xwayland is single-integer-scale. `1` when no output
    /// exists yet (nothing to size against).
    pub fn primary_output_scale(&self) -> i32 {
        self.outputs
            .keys()
            .min()
            .and_then(|idx| self.outputs.get(idx))
            // Normalise through `guarded_scale` before the `as i32` cast: a
            // non-finite, non-positive, or absurdly large `surface.scale` (a
            // corrupted persisted value, or a hostile test scale) would
            // otherwise saturate the cast to `i32::MAX` and overflow
            // `96 * scale` in `export_x11_dpi` (review finding #8). The guard
            // yields a finite value in `[1, MAX_OUTPUT_SCALE]`.
            .map(|o| (Self::guarded_scale(o.scale) as f64).ceil().max(1.0) as i32)
            .unwrap_or(1)
    }

    /// Publish the X11 HiDPI hint: set `Xft.dpi = 96 * scale` in the root
    /// window's `RESOURCE_MANAGER` property on the compositor's own Xwayland, so
    /// X11 toolkits (GTK/Qt read this the way `xrdb` merges it) size fonts and
    /// UI for the output scale. wlroots' `xwm` does not set `RESOURCE_MANAGER`
    /// and there is no `wlr` API for it, so — like a real desktop session — the
    /// compositor connects to its own X server as an ordinary client and sets
    /// the property itself.
    ///
    /// Best-effort and fire-and-forget on a detached thread: a slow or refused
    /// X connection must never block the compositor loop, and a failure only
    /// costs X11 apps a correct DPI (they fall back to 96), never correctness of
    /// the Wayland session. `display` is the `:N` just made valid by `ready`.
    ///
    /// HiDPI scope (design "HiDPI is fundamentally limited for X11"): this DPI
    /// hint is how DPI-aware X11 toolkits scale under the single global integer
    /// scale. Buffer-upscaling a *DPI-unaware* X11 client's surface to the
    /// output scale would need scene-node scaling of a `wlr_scene_tree`, which
    /// wlroots' scene graph does not offer for surface trees (only
    /// `wlr_scene_buffer` dest-size), so it is a documented follow-up, not done
    /// here. Mixed-DPI multi-monitor for X11 is the same known wlroots limit.
    /// Publish `DISPLAY` and the X11 cursor hints into the process environment,
    /// for session children (real or test-spawned) that inherit it. Called once,
    /// from the *pre-thread* boot window — in production before the D-Bus and
    /// wallpaper threads spawn, and in the harness before its boot handshake —
    /// so the `set_var`s never race a concurrent `getenv` on another thread
    /// (review finding #5). The lazy Xwayland manager reserves its display socket
    /// at `create_xwayland` time, so `display_name` is already known here, well
    /// before the first client triggers the actual Xwayland start.
    ///
    /// A theme the session already exported is respected; an absent one gets a
    /// sane `default` so X11 apps never fall back to the tiny bitmap core-X
    /// cursor. `XCURSOR_SIZE` is deliberately *not* forced here: the output scale
    /// is unknown at boot, and forcing a fixed `24` would both lose HiDPI sizing
    /// and (since the env var outranks the X resource in Xcursor's lookup)
    /// override the scale-aware `Xcursor.size` that `export_x11_dpi` writes to
    /// the root `RESOURCE_MANAGER` once the scale is known. A session that
    /// explicitly exported `XCURSOR_SIZE` still wins, exactly as intended.
    pub fn publish_xwayland_env(display_name: Option<&str>) {
        let Some(name) = display_name else { return };
        // SAFETY (icedtea unsafe exception (c)): every caller runs in the
        // pre-thread boot window (production `run`) or before the harness boot
        // handshake, so nothing else in the process is reading or writing the
        // environment concurrently — the same guarantee the `WAYLAND_DISPLAY`
        // write next to the production call site relies on.
        unsafe {
            std::env::set_var("DISPLAY", name);
        }
        if std::env::var_os("XCURSOR_THEME").is_none() {
            unsafe { std::env::set_var("XCURSOR_THEME", "default") };
        }
    }

    fn export_x11_dpi(display_name: String, scale: i32) {
        // `scale` arrives already clamped from `primary_output_scale`, but keep
        // the multiply saturating as defense in depth so no future caller can
        // overflow it (review finding #8) — the dev/test profile builds with
        // overflow-checks on, where a plain `96 * scale` would panic in this
        // detached thread and silently drop the DPI publish.
        let scale = scale.max(1);
        let dpi = 96i32.saturating_mul(scale);
        // The X11 cursor size that scales with the output (review finding #4):
        // the conventional logical 24 times the integer scale, published as the
        // `Xcursor.size` resource rather than the `XCURSOR_SIZE` env var so it is
        // an X-property write with no getenv/setenv race, and so a session's own
        // `XCURSOR_SIZE` (which outranks the resource in Xcursor's lookup) still
        // wins where it was set.
        let cursor_size = 24i32.saturating_mul(scale);
        // Stamp this export with a dispatch-order generation. `export_x11_dpi` is
        // only ever called from the compositor thread (`xwayland_ready` and the
        // test-only scale command, both on the event loop), so `fetch_add` here
        // orders exports by the order their scales were set — which the detached
        // threads below then race for the lock in *some* order (review finding
        // #5). Serializing the writes is not enough: a Mutex makes each publish
        // atomic but does not stop an older scale's thread from acquiring the lock
        // last and clobbering a newer one. The generation lets the loser skip.
        static NEXT_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let generation = NEXT_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            use x11rb::connection::Connection as _;
            use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _, PropMode};
            use x11rb::wrapper::ConnectionExt as _;

            // Serialize the connect→write→sync→disconnect against any other
            // in-flight DPI export: each export runs on its own detached thread
            // with its own X connection, and without this two overlapping
            // publishes could interleave on the server. The lock keeps each
            // publish atomic; the generation below keeps the *last dispatched*
            // scale winning regardless of lock-acquisition order.
            static DPI_EXPORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            // The highest generation that has committed a write; a thread whose
            // generation is older than this arrives after a newer scale already
            // won and must not overwrite it with stale values.
            static LAST_WRITTEN_GEN: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let _serialized = DPI_EXPORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            // `generation + 1` so a genuine gen 0 can still clear the initial 0.
            if generation + 1 < LAST_WRITTEN_GEN.load(Ordering::Relaxed) {
                return;
            }

            let (conn, screen_num) = match x11rb::connect(Some(&display_name)) {
                Ok(pair) => pair,
                Err(err) => {
                    tracing::warn!(?err, name = %display_name, "could not connect to Xwayland to set Xft.dpi");
                    return;
                }
            };
            let root = conn.setup().roots[screen_num].root;
            // `RESOURCE_MANAGER` is the conventional xrdb database, a `STRING`
            // property of xrdb-style `name:\tvalue\n` lines on the root window.
            // Both hints scale with the output: `Xft.dpi` for fonts/UI, and
            // `Xcursor.size` so an Xcursor-reading X11 app draws its pointer at
            // the right device size on a HiDPI monitor (review finding #4).
            let value = format!("Xft.dpi:\t{dpi}\nXcursor.size:\t{cursor_size}\n");
            if let Err(err) = conn.change_property8(
                PropMode::REPLACE,
                root,
                AtomEnum::RESOURCE_MANAGER,
                AtomEnum::STRING,
                value.as_bytes(),
            ) {
                tracing::warn!(?err, "could not set RESOURCE_MANAGER Xft.dpi on the X root");
                return;
            }
            // The write landed: record this generation so any older in-flight
            // export skips rather than clobbering it (review finding #5).
            LAST_WRITTEN_GEN.fetch_max(generation + 1, Ordering::Relaxed);
            // A synchronous round-trip (rather than a bare `flush`) before the
            // connection is dropped: the change request must be fully processed
            // by the X server before this client disconnects, or the server can
            // discard the still-buffered request when the socket closes. Reading
            // back the input focus is a cheap request whose reply cannot arrive
            // until everything queued ahead of it — the property change — has
            // been handled.
            match conn.get_input_focus() {
                Ok(cookie) => {
                    if let Err(err) = cookie.reply() {
                        tracing::warn!(
                            ?err,
                            "RESOURCE_MANAGER change may not have reached Xwayland"
                        );
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        ?err,
                        "could not sync the RESOURCE_MANAGER change to Xwayland"
                    );
                }
            }
        });
    }

    pub fn handle_command(&mut self, cmd: crate::dbus::DbCommand) -> Option<()> {
        use crate::dbus::DbCommand;
        match cmd {
            DbCommand::Focus(id) => {
                let previous = self.focused_id();
                self.window_manager.focus(id)?;
                // Re-review finding Important-1: an explicit toplevel-focus
                // assertion -- see `release_layer_focus`'s own doc. Round-2
                // finding Important-2: after the `?`, not before -- `id`
                // can name a stale/unknown window (this arm is reachable
                // from a D-Bus caller with a wire id this compositor never
                // heard of), and an early return must leave `layer_focus`
                // exactly as it was, not silently defeat a panel's
                // keyboard grab for a focus assertion that never actually
                // happened.
                self.release_layer_focus();
                self.sync_focus_change(previous);
            }
            DbCommand::Close(id) => self.request_close(id),
            DbCommand::Minimize(id, value) => {
                self.set_minimized_and_reconcile(id, value)?;
                self.dismiss_popups_of_hidden_roots();
            }
            DbCommand::Maximize(id, value) => self.set_maximized_target(id, value)?,
            DbCommand::Fullscreen(id, value) => self.set_fullscreen_target(id, value)?,
            DbCommand::SetWorkspace(id) => {
                self.switch_workspace(id)?;
                self.dismiss_popups_of_hidden_roots();
            }
            DbCommand::MoveToWorkspace(id, workspace) => {
                self.move_to_workspace(id, workspace)?;
                self.dismiss_popups_of_hidden_roots();
            }
            DbCommand::GetState(reply_tx) => {
                let mut snapshot = self.window_manager.snapshot();
                // The model snapshot carries no input state, so fill the
                // indicator fields here from the live runtime. No runtime
                // (every model-only unit test) reads as inactive / no
                // layout / uninhibited, matching the defaults.
                snapshot.ime_active = self
                    .wayland
                    .runtime()
                    .map(|rt| rt.input_method_active())
                    .unwrap_or(false);
                snapshot.keyboard_layout = self.keyboard_layout_name();
                snapshot.shortcuts_inhibited = self.shortcuts_inhibited();
                // M7 input mirrors live on `State` (the window model has no
                // runtime to read them from -- see `snapshot()`'s own doc).
                // Pure reads: no model mutation happens here, so the early
                // return below (skipping the unconditional `emit_pending()`)
                // stays correct.
                snapshot.cursor_visible = self.cursor_visible;
                snapshot.cursor_pos = self.cursor_pos;
                snapshot.touch_active = self.touch_active;
                let _ = reply_tx.send(snapshot); // No model mutation happened; nothing new to flush. Return
                // early so the unconditional `emit_pending()` below (a
                // no-op here, but let's not rely on that) stays meaningful
                // for every other arm.
                return Some(());
            }
            DbCommand::ReloadConfig => {
                self.spawn_config_reload();
                return Some(());
            }
            DbCommand::Quit => self.quitting = true,
            DbCommand::InjectTouchDown {
                x,
                y,
                id,
                time_msec,
                reply,
            } => {
                let serial = self
                    .wayland
                    .runtime()
                    .and_then(|rt| rt.inject_touch_down(x, y, id, time_msec));
                // M7: the inject path performs no `SeatHandler::touch_down`
                // (only real hardware input emits it), so the arm refreshes
                // the mirror itself -- exactly what that handler would have
                // established (a minted down means a live point).
                self.refresh_touch_active();
                let _ = reply.send(serial);
                return Some(());
            }
            DbCommand::InjectTouchMotion {
                x,
                y,
                id,
                time_msec,
                reply,
            } => {
                if let Some(rt) = self.wayland.runtime() {
                    rt.inject_touch_motion(x, y, id, time_msec);
                }
                self.refresh_touch_active();
                let _ = reply.send(());
                return Some(());
            }
            DbCommand::InjectTouchUp {
                id,
                time_msec,
                reply,
            } => {
                if let Some(rt) = self.wayland.runtime() {
                    rt.inject_touch_up(id, time_msec);
                }
                // M7: the up removed the point; re-derive (multi-touch may
                // still hold others).
                self.refresh_touch_active();
                let _ = reply.send(());
                return Some(());
            }
            DbCommand::InjectTouchCancel { reply } => {
                // M7, test-only: no wire producer for cancels exists
                // headless (cancels come from hardware), so this calls the
                // real `SeatHandler::touch_cancelled` on the loop thread.
                // The wire cancel to the client is the crate's
                // token-consuming `send_cancel`, which only real hardware
                // input drives -- this clears the consumer mirror only.
                self.touch_cancelled();
                let _ = reply.send(());
                return Some(());
            }
            DbCommand::InjectGesture { began, reply } => {
                // M7, test-only: headless has no gesture hardware, so this
                // calls the real `SeatHandler` notification on the loop
                // thread. The id names no live pointer (nothing could),
                // which is harmless by the handlers' contract.
                if began {
                    self.gesture_began(wlr::GestureId::dangling_nth_for_test(0));
                } else {
                    self.gesture_ended(wlr::GestureId::dangling_nth_for_test(0));
                }
                let _ = reply.send(());
                return Some(());
            }
            DbCommand::InjectSwitchToggle {
                switch_type,
                on,
                reply,
            } => {
                // M7, test-only: headless has no switch hardware, so this
                // feeds the real apply path a hardware-decoded `(type, on)`
                // pair no headless device can produce. The live
                // `switch_toggled` handler resolves the same pair from the
                // runtime aggregate before calling this function, so the
                // hook-driven e2e proves the production fold.
                self.apply_switch_toggle(switch_type, on);
                let _ = reply.send(());
                return Some(());
            }
            DbCommand::DragIconPosition { reply } => {
                let pos = self
                    .wayland
                    .runtime()
                    .and_then(|rt| rt.drag_icon_position());
                let _ = reply.send(pos);
                return Some(());
            }
            DbCommand::PopupsDismissed { reply } => {
                let _ = reply.send(self.popups_dismissed);
                return Some(());
            }
            DbCommand::SessionLocked { reply } => {
                let locked = self
                    .wayland
                    .runtime()
                    .map(|rt| rt.is_session_locked())
                    .unwrap_or(false);
                let _ = reply.send(locked);
                return Some(());
            }
            DbCommand::InputMethodActive { reply } => {
                let active = self
                    .wayland
                    .runtime()
                    .map(|rt| rt.input_method_active())
                    .unwrap_or(false);
                let _ = reply.send(active);
                return Some(());
            }
            DbCommand::CursorPosition { reply } => {
                let pos = self
                    .wayland
                    .runtime()
                    .map(|rt| rt.cursor_position())
                    .unwrap_or((0.0, 0.0));
                let _ = reply.send(pos);
                return Some(());
            }
            DbCommand::InputPopupPosition { reply } => {
                // The scene position of any currently-placed candidate popup.
                // `None` when none is placed, when the node went stale, or
                // before the runtime is attached.
                let pos = self.wayland.runtime().and_then(|rt| {
                    self.first_input_popup_node()
                        .and_then(|node| rt.node_position(node))
                });
                let _ = reply.send(pos);
                return Some(());
            }
            DbCommand::InputPopupNode { reply } => {
                // The node itself (not its position) behind the
                // `InputPopupPosition` oracle: the first tracked popup's node.
                // Captured while placed, it lets a test assert the node is
                // destroyed — not merely unrecorded — after teardown.
                let node = self.first_input_popup_node();
                let _ = reply.send(node);
                return Some(());
            }
            DbCommand::SceneNodePosition { node, reply } => {
                let pos = self.wayland.runtime().and_then(|rt| rt.node_position(node));
                let _ = reply.send(pos);
                return Some(());
            }
            DbCommand::PreeditOverlay { reply } => {
                // The overlay's own placement record: `Some` while composing
                // text is shown, `None` once every hide path cleared it.
                let pos = self.preedit_overlay.as_ref().map(|o| o.position);
                let _ = reply.send(pos);
                return Some(());
            }
            DbCommand::CursorShape { reply } => {
                // Load-bearing since wlr 0.20.26: read the crate's own
                // record of what it handed wlroots (`None` = the default
                // `left_ptr`), not a compositor-side mirror. Deleting the
                // `rt.set_cursor_shape` call in `request_set_shape` now
                // genuinely makes this read back `"Default"`.
                let shape = self
                    .wayland
                    .runtime()
                    .and_then(|rt| rt.cursor_shape())
                    .map(|s| format!("{s:?}"))
                    .unwrap_or_else(|| format!("{:?}", wlr::CursorShape::Default));
                let _ = reply.send(shape);
                return Some(());
            }
            DbCommand::OutputSize { reply } => {
                // The lowest-index output, which is the only one the headless
                // harness ever has -- and `geometry`, deliberately, not
                // `usable`: this answers "how big is the screen", and a
                // caller that wanted the space windows may occupy would be
                // asking a different question (see `OutputSurface::usable`).
                let geo = self.lowest_index_output_geometry();
                let _ = reply.send(geo);
                return Some(());
            }
            DbCommand::XwaylandDisplay { reply } => {
                // Read live from the runtime rather than the `xwayland_display`
                // field: with lazy start the manager reserves its display
                // socket (and sets `display_name`) at `create_xwayland` time,
                // well before `ready` fires -- so this answers before the
                // stored-on-`ready` field would, which is what lets the test
                // read `DISPLAY` and connect the very client whose connection
                // triggers the lazy `Xwayland` start.
                let name = self
                    .wayland
                    .runtime()
                    .and_then(|rt| rt.xwayland_display_name());
                let _ = reply.send(name);
                return Some(());
            }
            DbCommand::XwaylandReady { reply } => {
                // `xwayland_display` is set only in `xwayland_ready`, so its
                // presence is the readiness signal — the crate has started the
                // lazy Xwayland and wired the seat (selection/DND bridge armed).
                let _ = reply.send(self.xwayland_display.is_some());
                return Some(());
            }
            DbCommand::XwaylandOverrideRedirect { reply } => {
                // Report each tracked OR pop-up's *real* scene state, so the
                // test asserts against the live scene rather than the
                // compositor's own bookkeeping.
                let mut out = Vec::new();
                if let Some(rt) = self.wayland.runtime() {
                    for &sid in self.override_redirect.keys() {
                        let position = rt
                            .xwayland_surface_scene_position(sid)
                            .unwrap_or((i32::MIN, i32::MIN));
                        let above_toplevel =
                            rt.xwayland_surface_scene_parent_band(sid) == Some(wlr::Band::Top);
                        let keyboard_focused =
                            rt.xwayland_surface_has_keyboard_focus(sid).unwrap_or(false);
                        out.push(crate::dbus::OverrideRedirectProbe {
                            position,
                            above_toplevel,
                            keyboard_focused,
                        });
                    }
                }
                let _ = reply.send(out);
                return Some(());
            }
            DbCommand::SetOutputScaleForTest { scale, reply } => {
                // Record the scale on the primary output (the lowest model
                // index, the one `primary_output_scale` reads). `recorded` is
                // `false` when no output exists yet (review finding #11): the
                // reply lets the harness poll until one does, rather than
                // silently dropping the scale but acking success.
                let Some(&idx) = self.outputs.keys().min() else {
                    let _ = reply.send(false);
                    return Some(());
                };
                // Guard the stored scale exactly as the production
                // config-apply path does (review finding #8): a non-finite,
                // non-positive, or absurdly large test value is normalised
                // rather than stored raw and left to overflow the DPI math.
                let guarded = Self::guarded_scale(scale);
                if let Some(surface) = self.outputs.get_mut(&idx) {
                    surface.scale = guarded as f64;
                }
                // If Xwayland is already `ready`, re-publish the DPI hint
                // immediately so a test that raises the scale after boot
                // observes the new `Xft.dpi`; a test that raises it before the
                // first X connection instead has `ready` pick it up.
                if let Some(display) = self.xwayland_display.clone() {
                    Self::export_x11_dpi(display, self.primary_output_scale());
                }
                match self.wlr_output_id_for(idx) {
                    // A live output backs this index: defer the ack to
                    // `frame` (see `pending_test_output_scale`'s doc), which
                    // pushes `guarded` onto the *real* `wlr::Output` -- the
                    // model mirror this branch just wrote is not itself what
                    // the fractional-scale protocol's auto-sent
                    // `preferred_scale` reads from.
                    Some(oid) => self.pending_test_output_scale = Some((oid, guarded, reply)),
                    // No live output backs this index -- a `State` built
                    // straight through `State::new` with no `wlr::Runtime`
                    // attached, as unit tests do. Nothing to defer onto, so
                    // ack immediately exactly as before this deferred path
                    // existed.
                    None => {
                        let _ = reply.send(true);
                    }
                }
                return Some(());
            }
        }
        self.emit_pending();
        Some(())
    }

    /// Switch the active workspace to `workspace`, give it a focused window
    /// if it has a focusable one, and reconcile the scene so only its
    /// windows are visible (review findings I1 and I6).
    pub fn switch_workspace(&mut self, workspace: u32) -> Option<()> {
        if !self.window_manager.set_active_workspace(workspace) {
            return None;
        }
        // Review finding I2, second half: `is_none()` alone let a focus
        // pointer that names an *unfocusable* row (unmapped, or minimized --
        // the same shape, which predates this branch) survive a workspace
        // switch and go on answering `close`/`maximize`/`snap` while the
        // seat correctly refused it. `release_focus` on the unmap path is
        // what stops the unmapped case from arising at all; this is the
        // belt-and-braces re-pick for anything that still slips through.
        let current = self.window_manager.focused_window().map(|w| w.id);
        if current.is_none_or(|id| !self.window_manager.is_visible_id(id)) {
            self.window_manager.refocus_after_hide(workspace);
        }
        // Finding F14: arriving on the workspace answers an attention hint on
        // whatever window is focused there. `WindowManager::focus` clears the
        // hint, but a window that *already* holds its own workspace's focus
        // pointer is never re-focused by the switch (nothing calls `focus`),
        // so exactly the case `focus`'s already-focused branch was written
        // for -- a background window flagged on an inactive workspace -- kept
        // its hint after the user came and looked straight at it.
        if let Some(id) = self.window_manager.focused_window().map(|w| w.id) {
            self.window_manager.clear_attention(id);
        }
        self.sync_scene();
        self.emit_pending();
        Some(())
    }

    /// Move `id` to `workspace`, switch to it, and focus the moved window
    /// there -- leaving the origin workspace focused on whatever it has left.
    ///
    /// Review finding I6: `set_workspace` cleared the origin's focus pointer
    /// and never set the destination's, so after `MoveToWorkspace` the
    /// active workspace had no focused window and the very next
    /// `close`/`fullscreen`/`snap` action silently no-opped.
    pub fn move_to_workspace(&mut self, id: WindowId, workspace: u32) -> Option<()> {
        let origin = self.window_manager.get(id)?.workspace;
        self.window_manager.set_workspace(id, workspace)?;
        if origin != workspace {
            self.window_manager.refocus_after_hide(origin);
        }
        if !self.window_manager.set_active_workspace(workspace) {
            return None;
        }
        self.window_manager.focus(id);
        self.sync_scene();
        Some(())
    }

    /// Get the decoration action for a given window at the specified local coordinates.
    /// Returns the action if the window is focused and not fullscreen, None otherwise.
    /// Note: This gates on focused state (click-to-focus design), not CSD state.
    pub fn decoration_action_for(
        &self,
        id: WindowId,
        local: (i32, i32),
    ) -> Option<crate::decoration::DecorationAction> {
        let w = self.window_manager.get(id)?;
        if !w.focused || w.fullscreen {
            return None;
        }
        Some(crate::decoration::hit_test(w.geometry, local))
    }

    /// Apply a decoration action to a window: Close removes it, Maximize toggles maximized state,
    /// Minimize minimizes, Move would be handled by the input layer (not here).
    pub fn apply_decoration_action(
        &mut self,
        id: WindowId,
        action: crate::decoration::DecorationAction,
    ) {
        use crate::decoration::DecorationAction;
        match action {
            DecorationAction::Close => {
                // Asks the client to close (and lets its own destroy drive
                // the model removal) rather than dropping the model row out
                // from under a still-running client -- see `request_close`.
                self.request_close(id);
            }
            DecorationAction::Maximize => {
                let _ = self.toggle_maximized(id);
            }
            DecorationAction::Minimize => {
                let _ = self.set_minimized_and_reconcile(id, true);
            }
            DecorationAction::Move => {
                // Move is handled by the input layer (pointer drag),
                // not by a discrete window-manager mutation.
            }
            DecorationAction::None => {
                // No action.
            }
        }
        self.emit_pending();
    }

    /// Dispatch a keybinding action string (from `handle_key`, i.e. a config
    /// keybinding). Today this is reachable only from config keybindings --
    /// the D-Bus interface (`dbus.rs`'s `DbCommand`) has no action
    /// passthrough. The `"spawn"` arm in particular must never be wired
    /// into a future D-Bus action passthrough: it execs a program directly
    /// (see `parse_spawn_argv`), and a config-sourced command string
    /// reaching that arm from an untrusted D-Bus caller would be a remote
    /// code-execution path. Returns `None` for unrecognized actions or
    /// when a precondition (no focused window, unknown output, etc.) isn't
    /// met; `Some(())` on success. Always drains pending events on success.
    pub fn apply_action(&mut self, action: &str) -> Option<()> {
        let mut parts = action.splitn(2, ':');
        let base = parts.next()?;
        let arg = parts.next().map(|s| s.to_string());
        match base {
            "close" => {
                let id = self.window_manager.focused_window()?.id;
                self.request_close(id);
            }
            "maximize" => {
                let id = self.window_manager.focused_window()?.id;
                self.toggle_maximized(id)?;
            }
            "fullscreen" => {
                let id = self.window_manager.focused_window()?.id;
                self.toggle_fullscreen(id)?;
            }
            "quit" => self.quitting = true,
            // Task 13 threading requirement (binding, carried over from the
            // task-7 threading-model ruling): the redb I/O + JSON parse must
            // never block the render loop, for the keybinding-triggered
            // reload just as much as the D-Bus `ReloadConfig` method --
            // `handle_command`'s `ReloadConfig` arm and this arm share the
            // exact same worker-thread dispatch for that reason.
            "reload" => self.spawn_config_reload(),
            // NOTE (brief deviation): the sample matched on the literal
            // `"cycle:alt_tab"` here, but `action.splitn(2, ':')` above
            // already split that string into `base = "cycle"`, `arg =
            // Some("alt_tab")` -- a match on `base` can never see the colon,
            // so as written this arm was as unreachable as the
            // `"snap:restore"` arm noted above. Matching `"cycle"` and
            // checking `arg` makes it reachable (and leaves room for other
            // `cycle:*` variants later without another silent dead arm).
            //
            // Task 11 review #1: a session's entry list is now captured
            // once by `AltTabMachine::start` and kept stable for the whole
            // session -- steps read it back via `self.alt_tab.entries()`
            // instead of recomputing `alt_tab_entries()` (and therefore a
            // possibly different length/order) on every keypress. The
            // session itself is ended by `end_alt_tab` (see its doc for the
            // chosen end condition), not here.
            "cycle" => {
                if arg.as_deref() != Some("alt_tab") {
                    return None;
                }
                if !self.alt_tab.is_active() {
                    let entries = self.window_manager.alt_tab_entries();
                    if entries.is_empty() {
                        return None;
                    }
                    self.alt_tab.start(entries);
                } else {
                    self.alt_tab.step(true);
                }
                let idx = self.alt_tab.index();
                let entries = self.alt_tab.entries().to_vec();
                if let Some(wid) = entries.get(idx).copied() {
                    // Re-review finding Important-1: an explicit
                    // toplevel-focus assertion -- see `release_layer_focus`'s
                    // own doc.
                    self.release_layer_focus();
                    let previous = self.focused_id();
                    self.window_manager.focus(wid);
                    self.sync_focus_change(previous);
                }
                self.emit(Event::AltTabState(AltTabState {
                    active: true,
                    entries,
                    index: idx,
                }));
            }
            // Security hardening: `cmd` comes from a config keybinding's
            // action string (see this fn's doc). It used to be handed to
            // `sh -c`, which gives full shell interpretation (pipes,
            // metacharacters, `$VAR` expansion) to that config-sourced
            // string, at the compositor's own privilege level. Instead it's
            // word-split into a direct argv and exec'd with no shell in
            // between -- `parse_spawn_argv` is the pure, unit-tested core
            // of that split. An empty or unparseable command is logged and
            // ignored rather than spawning anything.
            "spawn" => {
                let cmd = arg?;
                match parse_spawn_argv(&cmd) {
                    Some((program, args)) => {
                        std::process::Command::new(program).args(args).spawn().ok();
                    }
                    None => {
                        tracing::warn!("spawn action: empty or unparseable command: {cmd:?}");
                    }
                }
            }
            // Task 11 review #4: `n: u32` from an externally-parseable
            // action string (this dispatcher's own doc says it's also
            // reachable "from a D-Bus-triggered action") minus 1 is unsigned
            // subtraction -- `"workspace:0"` panicked the whole compositor
            // in debug builds (and silently wrapped to `u32::MAX` in
            // release). `checked_sub` turns that into a clean `None`
            // instead. Separately, `set_active_workspace` returns `bool`
            // ("does this workspace exist") and was being discarded, so an
            // out-of-range `"workspace:7"` against the 4-workspace default
            // used to report success (`Some(())`) having silently done
            // nothing; it's now propagated as a real failure.
            "workspace" => {
                let n: u32 = arg?.parse().ok()?;
                let idx = n.checked_sub(1)?;
                self.switch_workspace(idx)?;
            }
            "move_to_workspace" => {
                let id = self.window_manager.focused_window()?.id;
                let n: u32 = arg?.parse().ok()?;
                let idx = n.checked_sub(1)?;
                self.move_to_workspace(id, idx)?;
            }
            // NOTE (brief deviation): the sample dispatch code had a second,
            // unreachable `"snap:restore" => ...` match arm alongside this
            // one. `action.splitn(2, ':')` always assigns `base = "snap"`
            // and `arg = Some("restore")` for the action string
            // `"snap:restore"` -- matching on `base` can therefore never
            // observe `"snap:restore"` as a whole. The restore case is
            // folded into this arm's `match arg?.as_str()` instead, which is
            // the only way it's reachable.
            "snap" => {
                let id = self.window_manager.focused_window()?.id;
                match arg?.as_str() {
                    "left" => self.snap(id, SnapZone::Left)?,
                    "right" => self.snap(id, SnapZone::Right)?,
                    "up" => self.snap(id, SnapZone::Top)?,
                    "down" => self.snap(id, SnapZone::Bottom)?,
                    "restore" => self.snap_restore(id)?,
                    _ => return None,
                }
            }
            _ => return None,
        }
        self.emit_pending();
        Some(())
    }

    /// End an in-progress alt-tab session, if one is active: marks the
    /// machine inactive and emits the terminal `AltTabState { active: false,
    /// .. }` so the shell's overlay (the `contract::Event` consumer) can
    /// dismiss itself. A no-op (no event emitted) if no session is active,
    /// so callers can call this unconditionally.
    ///
    /// Task 11 review #1: nothing called `AltTabMachine::end()` anywhere in
    /// the compositor, so a session never terminated -- `is_active()` stayed
    /// `true` forever after the first `SUPER+Tab` and the shell's overlay
    /// had no way to ever be told to close. The brief/spec don't name an
    /// explicit end mechanism, so the one real WMs use was picked: alt-tab
    /// ends when the *configured `cycle:alt_tab` binding's* modifier is
    /// released -- SUPER for the default binding, but not hardcoded to it
    /// (see `watched_alt_tab_modifiers`, `alt_tab_should_end`). `SeatHandler::key`
    /// (`wlr::SeatHandler` impl below) calls this via `alt_tab_should_end`
    /// on every key event, which covers "released the modifier while still
    /// holding Tab" and "released Tab first, then the modifier" alike, and
    /// -- the fix-round correction -- recognizes the modifier's own release
    /// event by its keysym rather than relying solely on the (there, stale)
    /// modifier-held booleans; see `alt_tab_should_end`'s doc for why the
    /// booleans alone miss that event.
    pub fn end_alt_tab(&mut self) {
        if !self.alt_tab.is_active() {
            return;
        }
        let entries = self.alt_tab.entries().to_vec();
        let idx = self.alt_tab.index();
        self.alt_tab.end();
        self.emit(Event::AltTabState(AltTabState {
            active: false,
            entries,
            index: idx,
        }));
        self.emit_pending();
    }

    /// The modifier flags `SeatHandler::key` should watch for alt-tab's end
    /// condition: whatever the configured `cycle:alt_tab` binding names, or
    /// `SUPER` (the default binding's own modifier) if that action has no
    /// binding at all -- which cannot happen with `default_config()` but is
    /// not assumed here, since a config can in principle drop the binding
    /// entirely while alt-tab is mid-session from before the reload.
    fn watched_alt_tab_modifiers(&self) -> input::Modifiers {
        self.config
            .keybindings
            .get("cycle:alt_tab")
            .map(|combo| input::modifiers_for_tokens(&combo.modifiers))
            .unwrap_or(input::Modifiers::SUPER)
    }

    /// Whether the compositor's own keybindings are currently suppressed by a
    /// live keyboard-shortcuts inhibitor (M8).
    ///
    /// Polled straight off the runtime on every call -- never cached from
    /// `SeatHandler::shortcuts_inhibitor_toggled`. The wlr review's recorded
    /// constraint: `Runtime::set_shortcuts_inhibitor_active` announces
    /// NOTHING, so an event arm alone cannot see transitions the compositor
    /// itself drove; only lifecycle birth/destroy announce. The compositor
    /// never drives an inhibitor itself (clients do), so polling is the whole
    /// source of truth here. No runtime (every model-only unit test) reads as
    /// uninhibited.
    fn shortcuts_inhibited(&self) -> bool {
        self.wayland
            .runtime()
            .is_some_and(|rt| rt.shortcuts_inhibited())
    }

    /// Human-readable name of the live keyboard's layout (M8), or `None`
    /// when no keyboard is tracked or its keymap cannot be read.
    ///
    /// The runtime only exposes the full keymap string
    /// (`KeyboardState::keymap`), which is kilobytes of xkb source -- not an
    /// indicator. The name is resolved through this crate's own `xkbcommon`
    /// copy (already a dependency for the keybinding matcher): compile the
    /// keymap, read layout 0's name (`"us"` compiles to `"English (US)"`).
    /// An uncompilable keymap (a hostile client uploaded garbage the seat
    /// accepted) or an empty layout set reads as `None`, never a panic, and
    /// an empty layout name (an unnamed layout) reads as `None` rather than
    /// an empty badge. No runtime reads as `None`, matching the default.
    fn keyboard_layout_name(&self) -> Option<String> {
        let keymap = self
            .wayland
            .runtime()
            .and_then(|rt| rt.keyboard_state())
            .and_then(|state| state.keymap)?;
        let context = xkbcommon::xkb::Context::new(xkbcommon::xkb::CONTEXT_NO_FLAGS);
        let map = xkbcommon::xkb::Keymap::new_from_string(
            &context,
            keymap,
            xkbcommon::xkb::KEYMAP_FORMAT_TEXT_V1,
            xkbcommon::xkb::KEYMAP_COMPILE_NO_FLAGS,
        )?;
        if map.num_layouts() == 0 {
            return None;
        }
        let name = map.layout_get_name(0);
        (!name.is_empty()).then(|| name.to_string())
    }

    /// Snap `id` to `zone` on the (first) output, saving its pre-snap
    /// geometry so `snap_restore` can undo it.
    /// Snap `id` to `zone` on the (first) output, saving its pre-snap
    /// geometry so `snap_restore` can undo it. A no-op returning `None` when
    /// `behavior.snap_enabled` is off (review finding I3: the flag had no
    /// consumers, so a user who disabled snapping still got snapping from
    /// both the `snap:*` actions and the drag machine).
    pub fn snap(&mut self, id: WindowId, zone: SnapZone) -> Option<()> {
        if !self.config.behavior.snap_enabled {
            return None;
        }
        let output_geo = self.usable_geo_for_pointer()?;
        let gap = self.config.appearance.snap_gap;
        let current = self.window_manager.get(id)?.geometry;
        self.snap_saved_geometry.entry(id).or_insert(current);
        self.window_manager
            .set_geometry(id, layout::snapped_geometry(output_geo, zone, gap))?;
        self.sync_window_to_scene(id);
        Some(())
    }

    /// Restore `id`'s geometry as it was before its most recent `snap`, if
    /// any was saved. Deliberately *not* gated on `snap_enabled`: a window
    /// snapped before the flag was turned off must still be restorable.
    pub fn snap_restore(&mut self, id: WindowId) -> Option<()> {
        if let Some(orig) = self.snap_saved_geometry.remove(&id) {
            let snapped = self.window_manager.get(id)?.geometry;
            let restored = layout::restored_geometry(orig, snapped);
            self.window_manager.set_geometry(id, restored)?;
            self.sync_window_to_scene(id);
        }
        Some(())
    }

    /// Apply a new `Config` to a *live* session: swap in the new
    /// keybindings/appearance/behavior and reconcile the workspace list,
    /// **without closing any client**. Returns the events the caller should
    /// queue (via `apply_reloaded_config`).
    ///
    /// Window rows survive the reload untouched -- their `WindowId`, focus,
    /// focus MRU, workspace assignment, geometry, client bindings, and
    /// mapped/minimized/maximized/fullscreen state all persist, as do the
    /// saved restore-geometry maps and title rasters that key off those ids.
    /// The manager is never rebuilt, so `next_id`/`seq` advance monotonically
    /// on their own and no id is ever reissued. This is the whole point of
    /// the M3 rewrite: a `SUPER+SHIFT+r` / D-Bus `ReloadConfig` no longer
    /// abandons every client to a permanently-invisible, unmodelled limbo.
    ///
    /// A reload does still cancel *transient* UI state, because it can be
    /// mid-interaction: `drag`, `resize`, `snap_preview`, and any alt-tab
    /// session are reset. Resetting an active alt-tab is a state change with
    /// no natural event, so the terminal `AltTabState{active: false, ..}` is
    /// appended to `events` (rather than via `end_alt_tab()`, which would
    /// flush on its own) so a shell overlay rendered from the last
    /// `active: true` gets its dismiss signal in order with the reload's
    /// other events.
    ///
    /// The workspace list is reconciled in place by
    /// [`WindowManager::set_workspace_names`]: names are renamed/extended/
    /// truncated, and a window on a now-removed workspace index migrates to
    /// workspace 0. One `WorkspaceList` (and one `ConfigReloaded`) is emitted.
    pub fn apply_config(&mut self, cfg: Config) -> Vec<Event> {
        let mut events: Vec<Event> = Vec::new();

        // A reload can land mid-interaction: drop any in-flight drag/resize
        // and clear the snap preview. Unlike the old behavior, this does NOT
        // close the windows those interactions referenced -- the rows stay.
        self.drag = input::DragMachine::new();
        self.resize = input::ResizeMachine::new();
        // Resetting an active alt-tab session is a state change with no
        // natural event, so emit the terminal `AltTabState{active: false}`
        // here (appended to `events` so it drains in order with the reload's
        // `WorkspaceList`/`ConfigReloaded`, instead of `end_alt_tab()`
        // jumping the queue via its own flush).
        if self.alt_tab.is_active() {
            events.push(Event::AltTabState(AltTabState {
                active: false,
                entries: self.alt_tab.entries().to_vec(),
                index: self.alt_tab.index(),
            }));
        }
        self.alt_tab = input::AltTabMachine::new();
        self.snap_preview = None;
        self.sync_snap_preview();

        // Review finding I4: revalidate keybindings against the new config.
        input::warn_about_keybindings(&cfg.keybindings);
        // Reconcile the workspace list in place -- no window row is dropped;
        // a window on a removed workspace index migrates to workspace 0.
        self.window_manager
            .set_workspace_names(cfg.workspace_names.clone());

        // Mirrors review finding I6 (`forget_window`'s same guard): a
        // truncated/clamped active workspace can leave migrated windows
        // sitting visible with no `focused_window`, since
        // `set_workspace_names` clears each migrated window's own
        // `focused` flag but never re-picks a successor. Belt-and-braces --
        // a no-op when focus is already valid -- picking the MRU candidate
        // here (rather than a manual seat call) lets the existing
        // `apply_reloaded_config`/`sync_scene` path carry it to the seat.
        if self.window_manager.focused_window().is_none() {
            let active = self.window_manager.active_workspace();
            self.window_manager.refocus_after_hide(active);
        }

        events.push(Event::WorkspaceList(self.window_manager.workspace_info()));
        events.push(Event::ConfigReloaded(cfg.appearance.clone()));
        self.config = cfg;
        events
    }

    /// Resolve `mods`+`keysym` against the configured keybindings and apply
    /// the matched action, if any.
    pub fn handle_key(&mut self, mods: input::Modifiers, keysym: u32) -> Option<()> {
        let action = input::match_action(&self.config.keybindings, mods, keysym)?;
        self.apply_action(&action)
    }

    /// Dispatch a pointer event (output logical coordinates) through the
    /// drag/snap state machine. See `PointerEvent` for the three cases.
    pub fn handle_pointer(&mut self, event: PointerEvent) -> Option<()> {
        match event {
            PointerEvent::Press { id, pointer } => self.handle_pointer_press(id, pointer),
            PointerEvent::Motion { pointer } => self.handle_pointer_motion(pointer),
            PointerEvent::Release { pointer } => self.handle_pointer_release(pointer),
        }
    }

    /// Pointer button press at `pointer` (output logical coordinates) on
    /// window `id`: focuses the window (click-to-focus), then, if the press
    /// hits the title bar's move area, begins a drag; otherwise applies
    /// whatever discrete decoration action (close/maximize/minimize) was
    /// hit.
    ///
    /// Task 11 re-review #2: this used to skip straight to
    /// `decoration_action_for`, which gates on `w.focused` -- so pressing
    /// an unfocused window's title bar or buttons did nothing at all (no
    /// focus change, no drag, no click action) since the gate always failed
    /// on the first click. Focusing first (unconditionally; `focus` on an
    /// already-focused window is a no-op) makes the very click that should
    /// raise a window also be the click that acts on it, matching ordinary
    /// click-to-focus window manager behavior.
    fn handle_pointer_press(&mut self, id: WindowId, pointer: (i32, i32)) -> Option<()> {
        // Round-2 re-review, same security class as finding F1: while the
        // session is locked, a pointer press must not touch the model at
        // all. Everything below this line acts on a window the lock screen
        // is covering -- it moves model focus (which decides who holds the
        // keyboard the instant the session unlocks), clears that window's
        // attention hint via `WindowManager::focus`, and can fire a
        // decoration action, so a blind click behind the lock screen could
        // close a window outright.
        //
        // Nothing the lock surface needs is gated here. Its input never
        // reaches this method: wlroots routes pointer events to the lock
        // surface through the seat itself, and this handler only ever
        // consults `window_at_point`, which answers from the *model* --
        // where a lock surface is not a row at all. The compositor's own
        // seat reconciliation is already lock-gated the same way
        // (`sync_seat_focus`/`sync_window_to_scene`), so this closes the one
        // remaining path into the model that a locked session left open.
        if self.session_locked {
            tracing::debug!(
                ?id,
                "ignoring a pointer press on a model window while the session is locked"
            );
            return None;
        }
        let previous = self.focused_id();
        self.window_manager.focus(id)?;
        // Re-review finding Important-1: click-to-focus is an explicit
        // toplevel-focus assertion -- see `release_layer_focus`'s own doc.
        // Round-2 finding Important-2: after the `?`, not before -- an
        // early return here (an unknown `id`) must leave `layer_focus`
        // untouched, not clear it for a focus assertion that never
        // actually happened.
        self.release_layer_focus();
        // Focus changes the client's activation state, so it has to reach
        // the client too (C1) -- both ends of the transition, not just the
        // new one (New-1).
        self.sync_focus_change(previous);
        // Task 11 re-review round 3 #2: flush right after `focus()`
        // mutates, before any of the `?`-early-returns below (e.g.
        // `decoration_action_for` returning `None` for a fullscreen
        // window) can skip the trailing `emit_pending()` call at the end
        // of this function and leave the focus event sitting queued until
        // some unrelated later flush.
        self.emit_pending();
        let geo = self.window_manager.get(id)?.geometry;
        // Despite its parameter name, `decoration_action_for`/`hit_test`
        // compares against `title_bar_rect`/`button_rects`, which are
        // themselves in the window's absolute (output) geometry space (see
        // the existing `decoration_action_for_returns_close_on_button_click`
        // test, which passes `geo.x + geo.width - 5` -- an absolute
        // coordinate -- as its "local" point). So `pointer` is passed
        // through unconverted here; only the drag `grab_offset` below is a
        // genuine window-relative offset.
        let action = self.decoration_action_for(id, pointer)?;
        if action == crate::decoration::DecorationAction::Move {
            let grab_offset = (pointer.0 - geo.x, pointer.1 - geo.y);
            self.drag.begin(id, grab_offset);
        } else {
            // A button action: give it its pressed color for the span of
            // applying it. Not a model mutation -- `ssd_press` is scene-only
            // state, so this costs one extra `sync_window_to_scene` call and
            // no additional `contract::Event`.
            if let Some(idx) = crate::decoration::button_at(geo, pointer) {
                self.ssd_press = Some((id, idx));
                self.sync_window_to_scene(id);
            }
            self.apply_decoration_action(id, action);
            // The action already ran (close/maximize/minimize); nothing is
            // still "held down." Clearing before this function's own
            // `sync_window_to_scene` (below `apply_decoration_action`'s own,
            // via e.g. `toggle_maximized`) keeps a window that outlives the
            // click (maximize) from being left with a stuck pressed tint.
            self.ssd_press = None;
            self.sync_window_to_scene(id);
        }
        self.emit_pending();
        Some(())
    }

    /// Recompute which title-bar button, if any, `pointer` (output logical
    /// coordinates) sits over, and re-sync whichever window(s) that changes
    /// -- the one that lost the hover, the one that gained it, or both.
    ///
    /// Scene-only state (`ssd_hover`'s own doc): no `contract::Event` is
    /// ever queued from here, only a `sync_window_to_scene` push to the
    /// seam.
    fn update_ssd_hover(&mut self, pointer: (i32, i32), over: Option<WindowId>) {
        let hovered = over.and_then(|id| {
            let geo = self.window_manager.get(id)?.geometry;
            crate::decoration::button_at(geo, pointer).map(|idx| (id, idx))
        });
        if hovered == self.ssd_hover {
            return;
        }
        let old_id = self.ssd_hover.map(|(id, _)| id);
        let new_id = hovered.map(|(id, _)| id);
        self.ssd_hover = hovered;
        if let Some(id) = old_id {
            self.sync_window_to_scene(id);
        }
        if let Some(id) = new_id.filter(|id| Some(*id) != old_id) {
            self.sync_window_to_scene(id);
        }
    }

    /// Honors a client move request (`xdg_toplevel.move`): only when the
    /// pointer is pressed and currently over `id`'s window; begins the
    /// existing `DragMachine` grab. A no-op otherwise -- per
    /// `wlr::ToplevelHandler::request_move`'s doc, an interactive move that
    /// never starts is legal, not a protocol violation, so there is nothing
    /// to answer.
    ///
    /// Finding 6, testing: the success path (guards pass, `self.drag.begin`
    /// actually runs) is untested -- it needs a live `wlr::Runtime` to read
    /// a real `pointer_position()` from, and there is no headless-runtime
    /// way to inject one in this crate's unit tests, the same gap
    /// `arrange_layers_leaves_layer_focus_alone_and_unmap_clears_it`
    /// documents for a live keyboard device.
    fn begin_client_move(&mut self, id: WindowId) {
        if !self.pointer_pressed {
            return;
        }
        let Some(rt) = self.wayland.runtime() else {
            return;
        };
        let (px, py) = rt.pointer_position();
        let pointer = (px as i32, py as i32);
        if self.window_at_point(pointer) != Some(id) {
            return;
        }
        let Some(geo) = self.window_manager.get(id).map(|w| w.geometry) else {
            return;
        };
        // Baseline behavior: an interactive move raises and focuses, the
        // same as `handle_pointer_press`'s click-to-focus-then-drag path.
        let previous = self.focused_id();
        if self.window_manager.focus(id).is_none() {
            return;
        }
        self.sync_focus_change(previous);
        let grab_offset = (pointer.0 - geo.x, pointer.1 - geo.y);
        self.drag.begin(id, grab_offset);
        self.emit_pending();
    }

    /// Same guards as [`Self::begin_client_move`], for `xdg_toplevel.resize`:
    /// maps the client's `wlr::Edges` onto `input::ResizeEdges`
    /// ([`resize_edges_from_wlr`]) and begins the existing `ResizeMachine`
    /// grab.
    ///
    /// Finding 6, testing: same untested-success-path limitation as
    /// `begin_client_move` -- see its doc. `resize_edges_from_wlr` is
    /// exactly the part of this method that stays testable without a live
    /// pointer.
    fn begin_client_resize(&mut self, id: WindowId, edges: wlr::Edges) {
        if !self.pointer_pressed {
            return;
        }
        let Some(rt) = self.wayland.runtime() else {
            return;
        };
        let (px, py) = rt.pointer_position();
        let pointer = (px as i32, py as i32);
        if self.window_at_point(pointer) != Some(id) {
            return;
        }
        let Some(geo) = self.window_manager.get(id).map(|w| w.geometry) else {
            return;
        };
        self.resize
            .begin(id, resize_edges_from_wlr(edges), geo, pointer);
    }

    /// Pointer motion at `pointer` (output logical coordinates) during an
    /// in-progress drag: updates the snap-zone preview (rendering consumes
    /// `self.snap_preview`).
    fn handle_pointer_motion(&mut self, pointer: (i32, i32)) -> Option<()> {
        let over = self.window_at_point(pointer);
        self.handle_pointer_motion_with_hit(pointer, over)
    }

    /// [`Self::handle_pointer_motion`] with the "which window is under the
    /// pointer" hit test already computed.
    ///
    /// Finding F12: the real `SeatHandler::pointer_motion` needs that answer
    /// for its own cursor-shape tracking anyway, and `window_at_point` walks
    /// and *allocates* (`visible_windows` builds a `Vec`) -- so it is
    /// computed once there and threaded through here rather than recomputed
    /// inside `update_ssd_hover`. `handle_pointer_motion` stays as the
    /// compute-it-yourself entry point for `handle_pointer`'s public
    /// `PointerEvent` dispatch, which unit tests drive directly.
    fn handle_pointer_motion_with_hit(
        &mut self,
        pointer: (i32, i32),
        over: Option<WindowId>,
    ) -> Option<()> {
        // Hover feedback tracks the pointer independently of drag/resize --
        // it must run even on every one of the early returns below, since
        // e.g. `self.drag.window_id()?` failing (no drag in progress, the
        // ordinary case whenever the pointer merely moves over a title bar)
        // must not skip it.
        self.update_ssd_hover(pointer, over);
        // An interactive resize (started by the client's `resize_request`)
        // takes precedence: it owns the pointer until the button is
        // released.
        if let Some(id) = self.resize.window_id() {
            let geometry = self.resize.geometry_for(pointer)?;
            self.window_manager.set_geometry(id, geometry)?;
            self.sync_window_to_scene(id);
            self.emit_pending();
            return Some(());
        }
        self.drag.window_id()?;
        let output_geo = self.usable_geo_for_pointer()?;
        let threshold = self.config.appearance.snap_gap.max(1) * 4;
        // Review finding I3: with snapping disabled the drag machine is
        // never told about zones at all, so no preview is staged *and*
        // `handle_pointer_release` can only ever produce a plain move.
        if !self.config.behavior.snap_enabled {
            self.snap_preview = None;
            self.sync_snap_preview();
            return Some(());
        }
        self.drag.motion(pointer, output_geo, threshold);
        self.snap_preview = self.drag.preview_zone().map(|zone| {
            layout::snapped_geometry(output_geo, zone, self.config.appearance.snap_gap)
        });
        self.sync_snap_preview();
        Some(())
    }

    /// Pointer button release at `pointer` (output logical coordinates):
    /// ends the drag, either snapping the window to the previewed zone or
    /// moving it to `pointer - grab_offset`.
    fn handle_pointer_release(&mut self, pointer: (i32, i32)) -> Option<()> {
        if self.resize.window_id().is_some() {
            let final_geometry = self.resize.geometry_for(pointer);
            let id = self.resize.end()?;
            if let Some(geometry) = final_geometry {
                self.window_manager.set_geometry(id, geometry)?;
            }
            self.sync_window_to_scene(id);
            self.emit_pending();
            return Some(());
        }
        let id = self.drag.window_id()?;
        let grab_offset = self.drag.grab_offset();
        self.snap_preview = None;
        self.sync_snap_preview();
        match self.drag.end() {
            input::DragResult::Snapped(zone) => {
                self.snap(id, zone)?;
            }
            input::DragResult::Moved => {
                let geo = self.window_manager.get(id)?.geometry;
                let new_pos = (pointer.0 - grab_offset.0, pointer.1 - grab_offset.1);
                self.window_manager.set_geometry(
                    id,
                    Rectangle {
                        x: new_pos.0,
                        y: new_pos.1,
                        width: geo.width,
                        height: geo.height,
                    },
                )?;
                self.sync_window_to_scene(id);
            }
            input::DragResult::Restored => {}
        }
        self.emit_pending();
        Some(())
    }

    /// A client mapped a new toplevel. Creates the model window at a cascade
    /// position, binds it to `toplevel`, and reconciles focus.
    pub fn new_toplevel(
        &mut self,
        toplevel: crate::wayland::ToplevelKey,
        app_id: &str,
        title: &str,
        pid: u32,
    ) {
        // Default toplevel size: real geometry arrives from the client's
        // first commit, which isn't known yet; `PLACEHOLDER_SIZE` is the
        // model's placeholder until then. Position cascades off whatever is
        // already mapped so new windows don't stack exactly on top of each
        // other.
        let occupied: Vec<icedtea_contract::Rectangle> = self
            .window_manager
            .windows_in_workspace(self.window_manager.active_workspace())
            .iter()
            .map(|w| w.geometry)
            .collect();
        let output_geo = self
            .usable_geo_for_pointer()
            .unwrap_or(icedtea_contract::Rectangle {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            });
        // Review finding M7: cascade positions wrap inside the output (and
        // count only this workspace's windows) so the Nth window can't open
        // off-screen with no title bar to grab.
        let (x, y) = layout::cascade_point_in(&occupied, PLACEHOLDER_SIZE, 24, output_geo);
        let geometry = icedtea_contract::Rectangle {
            x,
            y,
            width: PLACEHOLDER_SIZE.0,
            height: PLACEHOLDER_SIZE.1,
        };
        // `add_window` autofocuses, so this is a focus change like any other
        // (New-1): capture the outgoing focus before it happens.
        let previous = self.focused_id();
        let id = self.window_manager.add_window(app_id, title, pid, geometry);
        self.wayland.bind(id, toplevel);
        // Any decoration preference this client stated before it had a model
        // row (the normal ordering -- see `pending_decorations`) lands now.
        // The client itself was already answered at `initial_commit`, with
        // this same preference and the app-id both in hand (L1), so there is
        // nothing to re-negotiate here -- only the model to bring in line
        // with what the client was told.
        let requested = self.pending_decorations.remove(&toplevel).flatten();
        if requested.is_some() {
            self.window_manager
                .set_client_decorations_requested(id, requested);
        }
        // Explicit sync of the new window first (re-review minor 2):
        // `sync_focus_change` also reaches it today, but only because
        // `add_window` autofocuses onto the active workspace.
        self.sync_window_to_scene(id);
        self.sync_focus_change(previous);
        self.emit_pending();
    }

    /// A client destroyed its toplevel.
    ///
    /// Named `forget_toplevel` rather than `toplevel_destroyed` (task 8):
    /// `wlr::ToplevelHandler::toplevel_destroyed` now exists as a same-named
    /// trait method on `State`, and while the two don't collide (an inherent
    /// method always wins over a trait method of the same name during
    /// resolution, so `self.toplevel_destroyed(key)` would keep working), the
    /// call site inside the trait impl would read as a same-name recursive
    /// call to a reader who can't see that resolution rule. Renaming the
    /// inherent method removes the ambiguity at the source instead of relying
    /// on it being resolved correctly.
    pub fn forget_toplevel(&mut self, toplevel: crate::wayland::ToplevelKey) {
        // Before the early return: a toplevel that dies without ever
        // mapping still had a decoration preference parked for it, and
        // nothing else would ever come to collect it.
        self.pending_decorations.remove(&toplevel);
        let Some(id) = self.wayland.window_for(toplevel) else {
            return;
        };
        // The window's popups die with it; the model must not outlive them.
        self.forget_popups_of_root(PopupRoot::Window(id));
        self.wayland.forget(id);
        self.forget_window(id);
        self.emit_pending();
    }

    /// A client changed its title.
    ///
    /// The scene sync is not optional bookkeeping: the title is *drawn* now
    /// (`sync_ssd`'s buffer node), so a retitle that only updated the model
    /// would leave the old string on screen until some unrelated geometry
    /// change happened to re-sync the window.
    pub fn toplevel_title_changed(&mut self, toplevel: crate::wayland::ToplevelKey, title: &str) {
        let Some(id) = self.wayland.window_for(toplevel) else {
            return;
        };
        if self
            .window_manager
            .set_title(id, title.to_string())
            .is_some()
        {
            self.sync_window_to_scene(id);
            self.emit_pending();
        }
    }

    /// The topmost model window at `point` (frame space), or `None`.
    ///
    /// The model is the authority on this rather than the scene: the scene
    /// knows nothing about workspaces or minimization, and `visible_windows`
    /// is already MRU-ordered, which is a correct topmost-first order for
    /// click-to-focus (review finding I1).
    pub fn window_at_point(&self, point: (i32, i32)) -> Option<WindowId> {
        self.window_manager.window_at(point).map(|w| w.id)
    }

    pub fn set_background(&mut self, rect: wlr::RectId) {
        self.background = Some(rect);
    }

    pub fn background(&self) -> Option<wlr::RectId> {
        self.background
    }

    /// The scene rect node currently backing `snap_preview`, if any.
    /// Introspection for tests -- mirrors `background()`.
    pub fn snap_preview_rect(&self) -> Option<wlr::RectId> {
        self.snap_preview_rect
    }

    pub fn set_shutdown_source(&mut self, id: wlr::SourceId) {
        self.shutdown_source = Some(id);
    }

    pub fn set_command_receiver(
        &mut self,
        rx: crossbeam_channel::Receiver<crate::dbus::DbCommand>,
    ) {
        self.cmd_rx = Some(rx);
    }

    /// The `SourceId` `fd_ready` compares against to route a wake to
    /// `drain_pending_commands`. See `cmd_wake_source`'s field doc.
    pub fn set_cmd_wake_source(&mut self, id: wlr::SourceId) {
        self.cmd_wake_source = Some(id);
    }

    /// The `SourceId` `fd_ready` compares against to route a wake to
    /// `drain_config_reload`. See `config_reload_wake_source`'s field doc.
    pub fn set_config_reload_wake_source(&mut self, id: wlr::SourceId) {
        self.config_reload_wake_source = Some(id);
    }

    /// The write half `spawn_config_reload` nudges its worker threads with.
    /// See `config_reload_wake`'s field doc for why it lives on `State`
    /// rather than being handed to one worker directly.
    pub fn set_config_reload_wake(&mut self, write: std::os::unix::net::UnixStream) {
        self.config_reload_wake = Some(write);
    }

    /// Keep the receiving half of `render::spawn_wallpaper_decode`'s channel
    /// so [`Self::drain_wallpaper`] has somewhere to read from.
    pub fn set_wallpaper_receiver(
        &mut self,
        rx: crossbeam_channel::Receiver<Option<image::RgbaImage>>,
    ) {
        self.wallpaper_rx = Some(rx);
    }

    /// The `SourceId` `fd_ready` compares against to route a wake to
    /// `drain_wallpaper`. See `wallpaper_wake_source`'s field doc.
    pub fn set_wallpaper_wake_source(&mut self, id: wlr::SourceId) {
        self.wallpaper_wake_source = Some(id);
    }

    /// Keep the wake pipe's write half alive for the process's life. See
    /// `wallpaper_wake`'s field doc (review finding C1) for why this must be
    /// called with the *original*, not a clone -- callers hand
    /// `render::spawn_wallpaper_decode` a `try_clone`d copy instead.
    pub fn set_wallpaper_wake(&mut self, write: std::os::unix::net::UnixStream) {
        self.wallpaper_wake = Some(write);
    }

    /// Apply every D-Bus command that arrived since the last turn.
    ///
    /// Collected before applying, rather than iterated lazily: `handle_command`
    /// takes `&mut self`, and the receiver lives in `self`.
    fn drain_pending_commands(&mut self) {
        let Some(rx) = self.cmd_rx.as_ref() else {
            return;
        };
        let pending: Vec<crate::dbus::DbCommand> = rx.try_iter().collect();
        for cmd in pending {
            self.handle_command(cmd);
        }
    }

    /// Whether a client-initiated interactive grab (`xdg_toplevel.move` /
    /// `xdg_toplevel.resize`) may be honored right now.
    ///
    /// Re-review: the lock. `begin_client_move`/`begin_client_resize` both
    /// run `WindowManager::focus` + `sync_focus_change` before they touch
    /// the drag/resize machines, so an unfocused background client behind
    /// the lock screen could re-pick who owns the keyboard the instant the
    /// session unlocks -- the same hole finding F1 closed for
    /// `request_activate` and click-to-focus.
    ///
    /// Split out of the two handlers, rather than inlined, so the policy has
    /// one name and one definition. It is deliberately *not* unit tested:
    /// with no `wlr::Runtime` attached both handlers bottom out in
    /// `begin_client_move`'s own `pointer_position` guard (see its doc's
    /// "finding 6, testing" note), so a unit test can only ever re-state
    /// this one-line predicate back to itself -- it stays green with both
    /// call sites' gates deleted, which is worse than no test. The real
    /// coverage is the pair of end-to-end tests named on
    /// [`Self::request_move`] and [`Self::request_resize`], which drive the
    /// handlers from a real locked session and are mutation-verified red.
    pub(crate) fn client_grab_requests_allowed(&self) -> bool {
        !self.session_locked
    }

    /// Which `cursor-shape-v1` device kinds this compositor honors.
    ///
    /// Split out of [`State::request_set_shape`] so the policy is testable:
    /// the handler's other half is `wlr::Runtime::set_cursor_shape`, which a
    /// unit test has no runtime to observe, and since `wlr` 0.20.26 the
    /// *who is asking* half is the crate's (it delivers the callback only to
    /// the pointer-focused seat client), leaving this as the only decision
    /// the compositor still makes for itself.
    pub(crate) fn honors_cursor_shape_device(device: wlr::CursorShapeDevice) -> bool {
        device == wlr::CursorShapeDevice::Pointer
    }

    /// The single call site for `wlr::Runtime::set_cursor_shape`: applies
    /// `shape` to the seat cursor, a no-op if there is no seat cursor yet
    /// (see the crate's own doc).
    ///
    /// Since `wlr` 0.20.26 the crate owns everything that used to live
    /// around this call: the shape persists across pointer events instead of
    /// being stomped back to `left_ptr` by `ensure_cursor_image`, the crate
    /// resets it on every pointer-focus change (surface leave, enter, or
    /// client death), and it short-circuits a repeat of the shape already in
    /// force. So there is no mirror field, no per-motion re-assert and no
    /// model-side "who owns the pointer" tracking left to keep in step --
    /// `wlr::Runtime::cursor_shape` is the one true reading.
    fn apply_cursor_shape(&mut self, shape: wlr::CursorShape) {
        if let Some(rt) = self.wayland.runtime() {
            rt.set_cursor_shape(shape);
        }
        // A shape request means a visible cursor: something (a client with
        // pointer focus, or the session lock path above) just named the
        // image, so the pre-first-motion `Hidden` reading no longer holds.
        self.cursor_visible = true;
    }

    /// M7: re-derive `touch_active` from the live seat touch state. The
    /// up/inject arms call this rather than clearing unconditionally: other
    /// points may still be down (multi-touch), and only the seat knows.
    /// Without a runtime there is no seat to ask, which in production is
    /// impossible on these paths (touch input implies a live seat) and in
    /// unit tests means "no points": both read `false`.
    fn refresh_touch_active(&mut self) {
        self.touch_active = self
            .wayland
            .runtime()
            .and_then(|rt| rt.touch_state())
            .is_some_and(|s| !s.points.is_empty());
    }

    /// M7: record one observed cursor position and refresh the visibility
    /// mirror alongside it. Called from every pointer path that carries a
    /// position (motion, button): the crate applies the cursor image before
    /// it emits either (`ensure_cursor_image`), so an observed position
    /// normally means a showing image -- but the live `cursor_state` read
    /// below is authoritative when a runtime exists, and without one (unit
    /// tests) `true` is the only sane reading of "the pointer moved".
    fn note_cursor_at(&mut self, pointer: (i32, i32)) {
        self.cursor_pos = Some(pointer);
        self.cursor_visible = self
            .wayland
            .runtime()
            .and_then(|rt| rt.cursor_state())
            .map(|s| s.image != wlr::CursorImage::Hidden)
            .unwrap_or(true);
    }

    /// M7: fold one decoded switch toggle into the session feed. The same
    /// function the live `switch_toggled` handler calls after resolving the
    /// pair from the runtime aggregate, and the test-only
    /// `InjectSwitchToggle` arm calls with a synthesized pair -- one path
    /// for both, so the hook-driven e2e proves the production fold.
    ///
    /// The lid reading is what the session consumes; non-lid switches ride
    /// along with `lid_closed: false` so the toggle itself stays visible on
    /// the feed. This deliberately does NOT drive `session_locked`: that
    /// flag gates focus reconciliation while a lock *surface* holds the
    /// session, and a lid position is a power signal for the session to act
    /// on, not a lock-surface assertion.
    fn apply_switch_toggle(&mut self, switch_type: wlr::SwitchType, on: bool) {
        let lid_closed = matches!(switch_type, wlr::SwitchType::Lid) && on;
        self.emit(Event::SwitchToggled { lid_closed });
        self.emit_pending();
    }

    /// M7 consumer half of the pointer-constraint gate: whether the focused
    /// surface currently carries a LOCKED constraint.
    ///
    /// The crate enforces lock/confine itself before any `pointer_motion`
    /// reaches this model (a locked cursor is frozen outright, so no motion
    /// event arrives at all; a confined one arrives already clamped), which
    /// is why this reads as redundant on the motion path -- and that is
    /// exactly the point: the model must never advance focus, hover, drag
    /// or resize on motion a lock froze, no matter which dispatch source
    /// delivered it. Confined constraints need no model gate: the clamped
    /// position they carry is already a legal model position.
    ///
    /// Anything without a runtime, without focus, or without a live
    /// constraint reads "unlocked": the gate only ever closes on a positive
    /// live read, never on a miss.
    fn pointer_locked_for_focus(&self) -> bool {
        let Some(focused) = self.focused_id() else {
            return false;
        };
        let Some(key) = self.wayland.toplevel_for(focused) else {
            return false;
        };
        let Some(rt) = self.wayland.runtime() else {
            return false;
        };
        rt.constraint_state_for_surface(wlr::ConstraintSurface::Toplevel(key.0))
            .is_some_and(|s| s.constraint_type == wlr::ConstraintType::Locked)
    }

    /// Raise `id`'s attention hint -- but only if the hint is one the user
    /// can ever answer.
    ///
    /// Finding F6: `WindowManager::focus` is what clears an attention hint,
    /// and it *refuses an unmapped window outright*. So flagging an unmapped
    /// target produced a hint that nothing could ever take back: the shell
    /// would show it until the window either mapped and was focused, or was
    /// destroyed. Both refusal paths (`request_activate` and
    /// `xwayland_request_activate`) route through here, so a request aimed at
    /// a window with no client on screen is dropped with a debug log rather
    /// than turned into permanent shell noise.
    ///
    /// Minimized and off-workspace targets are still flagged: those are
    /// answerable -- restoring or switching to them clears the hint (see
    /// `set_minimized_and_reconcile` and `switch_workspace`) -- and they are
    /// precisely the cases the hint exists for.
    fn raise_attention_if_answerable(&mut self, id: WindowId) {
        if !self.window_manager.get(id).is_some_and(|w| w.mapped) {
            tracing::debug!(
                ?id,
                "not flagging attention on an unmapped target; focus could never clear it"
            );
            return;
        }
        self.window_manager.set_attention(id, true);
    }
}

// --- Compositor library handlers ---
//
// Every method below runs underneath an `extern "C"` frame: a panic escaping
// one aborts the process rather than failing anything. So there is no
// `unwrap`, no `expect`, no `assert!`, and no indexing in any of these
// bodies — a condition that cannot be handled is recorded in `State` and
// acted on once control is back on the loop.

impl wlr::OutputHandler for State {
    fn new_output(&mut self, output: &wlr::Output<'_>) {
        let Some(runtime) = self.wayland.runtime().cloned() else {
            return;
        };
        // Renderer before enable: the DRM backend's enabling commit needs a
        // framebuffer, which wlroots can only allocate once the output has
        // its renderer/allocator. Nested backends tolerate either order,
        // which is how enable-first survived until the first real-DRM boot.
        if let Err(err) = runtime.init_output(output) {
            tracing::error!(?err, "could not give output a renderer");
            return;
        }

        // Apply any persisted `DisplayConfig` for this connector before the
        // geometry/scene sequence below reads the output's resulting box.
        // The name (`wlr::Output::name()`) is the stable key the settings
        // client and this lookup share; `None` (an unnamed output) matches
        // no persisted entry and takes the default preferred-mode path.
        let name = output.name().unwrap_or_default();
        let display_cfg = self
            .config
            .displays
            .iter()
            .find(|d| d.name == name)
            .cloned();
        // The scale actually committed on this output, mirrored into
        // `OutputSurface::scale` below so the M4 HiDPI DPI-hint export can read
        // it off the model without a live handler. Every arm that ends in the
        // preferred-mode default leaves this at the `1.0` identity; only a
        // successfully-applied persisted config raises it.
        let mut applied_scale = 1.0f64;
        match &display_cfg {
            Some(cfg) if cfg.enabled => {
                // On ANY setter error the output must not be left dark:
                // fall back to the preferred mode with wlroots' own auto
                // layout position, exactly the no-config path.
                if let Err(err) = Self::apply_display_config(&runtime, output, cfg) {
                    tracing::error!(?err, %name, "persisted display config failed; using preferred mode");
                    if let Err(err) = output.enable_with_preferred_mode() {
                        tracing::error!(?err, "could not enable output");
                        return;
                    }
                    // A partial apply may have committed a scale/transform
                    // before the failing setter; reset both to the identity so
                    // the preferred-mode fallback is a clean default state
                    // rather than a mix of persisted and preferred values.
                    if let Err(err) = output.set_scale(1.0) {
                        tracing::warn!(?err, %name, "could not reset scale on fallback");
                    }
                    if let Err(err) = output.set_transform(wlr::Transform::Normal) {
                        tracing::warn!(?err, %name, "could not reset transform on fallback");
                    }
                } else {
                    // The persisted config committed cleanly, so the scale it
                    // asked for (guarded the same way `apply_display_config`
                    // guards it) is what is now live on the output.
                    applied_scale = Self::guarded_scale(cfg.scale) as f64;
                }
            }
            Some(_) if !self.boot_disable_would_strand() => {
                // Persisted as disabled AND at least one other output is already
                // LIVE this boot (review finding #1: keyed off `self.outputs`,
                // not persisted enabled bools), so honoring this cannot strand
                // the session at zero outputs: honor it and stop.
                //
                // Crucially, a disabled output is modeless (`size()` is 0x0 and
                // `output_layout_box` is `None`), so it must NOT fall through to
                // the create_output/geometry/scene block below: inserting a 0x0
                // phantom into `self.outputs` would make it a survivor
                // candidate for `outputs.keys().min()` and collapse new-window
                // placement to the origin. Mirror `output_configuration_applied`'s
                // disabled branch, which likewise keeps disabled heads out of
                // the active set. Still re-advertise so bound managers see the
                // new (disabled) head.
                if let Err(err) = output.disable() {
                    // Review finding #12: the disable FAILED, so the output is
                    // physically still enabled. Recording it disabled and
                    // returning surfaceless would leave an enabled-but-untracked
                    // output rendering nothing. Instead, give it a valid mode and
                    // fall through to the geometry/scene block below, tracking it
                    // as the live output it actually is. Do NOT insert it into
                    // `disabled_outputs`.
                    tracing::error!(?err, %name, "could not disable output per persisted config; keeping it live");
                    if let Err(err) = output.enable_with_preferred_mode() {
                        tracing::error!(?err, "could not enable output after failed disable");
                        return;
                    }
                    // fall through (no return): treat as an enabled output.
                } else {
                    // Record the id (keyed by the unique `OutputId`, value = its
                    // connector name) so a later re-enable
                    // (`output_configuration_applied`, which is handed no id) can
                    // recover it and rehydrate the output into the active set. Do
                    // NOT create a surface / add to `self.outputs` -- it stays out
                    // of the active set exactly as the doc above requires.
                    self.disabled_outputs.insert(output.id(), name.clone());
                    runtime.update_output_manager_state();
                    return;
                }
            }
            Some(_) => {
                // Persisted as disabled, but honoring it would leave ZERO active
                // outputs (`self.outputs` is empty -> `outputs.keys().min()` is
                // `None` -> window placement/migrate/reclaim early-return -> a
                // black screen with no way to recover, review findings #1/#5).
                // This fires for an all-disabled persisted config AND for the
                // undocked-laptop case where the only present connector is the
                // persisted-disabled one and the persisted-enabled peer is
                // absent. Treat it as a config that cannot be honored and enable
                // this output with its preferred mode instead, falling through to
                // the geometry/scene block below like the `None` arm.
                // `output_configuration_applied`'s interactive path has its own
                // last-output guard for the live-reconfigure case.
                tracing::warn!(
                    %name,
                    "honoring persisted disable would leave zero active outputs; enabling this output to avoid a black screen with no recovery"
                );
                if let Err(err) = output.enable_with_preferred_mode() {
                    tracing::error!(?err, "could not enable output");
                    return;
                }
            }
            None => {
                if let Err(err) = output.enable_with_preferred_mode() {
                    tracing::error!(?err, "could not enable output");
                    return;
                }
            }
        }

        let (width, height) = output.size();
        // `next_output_index` is a monotonic counter, not `outputs.len()`
        // (review finding M2): `len()` recomputes the same index a
        // just-removed output had the moment a new one is added, colliding
        // with whatever the model (or a client-facing consumer) still
        // remembers about the old one.
        let index = self.next_output_index;
        self.next_output_index += 1;
        // The layout box is the output's real position (and, once placed,
        // its mode-derived size) in the shared multi-output coordinate
        // space; a `None` -- the output not yet in the layout, or a
        // 0x0-mode output, see `output_layout_box`'s own doc -- falls back
        // to the single-output-at-origin behavior this replaced.
        let geometry = runtime
            .output_layout_box(output.id())
            .map(|(x, y, w, h)| icedtea_contract::Rectangle {
                x,
                y,
                width: w,
                height: h,
            })
            .unwrap_or(icedtea_contract::Rectangle {
                x: 0,
                y: 0,
                width,
                height,
            });
        self.create_output(index, geometry);
        // Record the connector name so the applied-config handler can map an
        // `AppliedHead` back to this index and `destroyed`/persist can look
        // it up (see `OutputSurface::name`), and the committed scale so the M4
        // HiDPI DPI-hint export can read it (see `OutputSurface::scale`).
        if let Some(surface) = self.outputs.get_mut(&index) {
            surface.name = name;
            surface.scale = applied_scale;
        }
        self.output_ids.insert(output.id(), index);
        // A new output needs its own wallpaper node (if a decode has
        // already landed) at this output's own size -- nothing else calls
        // `sync_wallpaper_nodes` when the output set grows.
        self.sync_wallpaper_nodes();

        // The background covers the whole output. Sized here rather than at
        // creation because the mode is not known until now.
        if let Some(rect) = self.background {
            runtime.set_rect_size(rect, width, height);
            runtime.set_rect_position(rect, 0, 0);
        }

        // Existing windows (there are none at boot, but a hotplugged output
        // is the same code path) need their geometry pushed at the new size.
        self.sync_scene();
        self.emit_pending();

        // Review finding M5b: a layer surface announced before any output
        // existed (parked under the `NO_OUTPUT` sentinel), or left behind
        // by an output `destroyed` with no survivor at the time, gets its
        // first (or next) real home and configure the moment any output
        // -- this one -- exists.
        self.resolve_orphaned_layers();

        // Review finding I1: nothing else owns the first frame -- it used to
        // arrive only incidentally, whenever the background rect's own
        // damage happened to trigger one. `schedule_frame` (public since
        // 0.20.1) asks wlroots to fire `OutputHandler::frame` for this
        // output on its own, so a freshly enabled output that draws nothing
        // still gets a `frame` callback and a first commit.
        output.schedule_frame();

        // Re-advertise the manager state so every bound
        // `zwlr_output_manager_v1` client sees the new head (and any layout
        // shift the connect caused) with a bumped serial. A no-op when no
        // manager was created (`lib.rs::run` degrades gracefully).
        runtime.update_output_manager_state();

        // M7: refresh the input mirrors against the new output set. The
        // cursor<->layout attachment itself is crate-owned (`create_seat`
        // attaches the cursor to the layout; per-output pinning via
        // `map_cursor_to_output` is deliberately NOT done here -- on a
        // multi-output layout it would strand the cursor on the newest
        // output, and the whole-layout attach already covers every
        // head). What this refreshes is the model's reading: a cursor
        // image applied before this output existed, or touch points down
        // across the hotplug.
        self.cursor_visible = runtime
            .cursor_state()
            .map(|s| s.image != wlr::CursorImage::Hidden)
            .unwrap_or(self.cursor_visible);
        self.refresh_touch_active();
    }

    fn frame(&mut self, output: &wlr::Output<'_>) {
        self.frames += 1;
        // See `pending_test_output_scale`'s own doc: this is the only place
        // after boot with a live `&wlr::Output` to push a
        // `DbCommand::SetOutputScaleForTest` scale onto. `take` first so a
        // rejected `set_scale` still consumes the pending entry rather than
        // retrying forever against every future frame of every output.
        if self
            .pending_test_output_scale
            .as_ref()
            .is_some_and(|(oid, ..)| *oid == output.id())
        {
            let (_, scale, reply) = self.pending_test_output_scale.take().unwrap();
            if let Err(err) = output.set_scale(scale) {
                tracing::warn!(?err, "SetOutputScaleForTest: live scale rejected");
            }
            let _ = reply.send(true);
        }
        let Some(runtime) = self.wayland.runtime() else {
            return;
        };
        // A rejected commit is routine — wlroots rejects one when nothing
        // changed — so it is logged at debug and never escalated.
        if let Err(err) = runtime.commit_output(output) {
            tracing::debug!(?err, "scene commit rejected");
        }
    }

    fn destroyed(&mut self, id: wlr::OutputId) {
        // A disabled output (tracked only in `disabled_outputs`, never in
        // `output_ids`/`outputs`) can be unplugged too. `destroyed` gets only
        // the id; the map is keyed by that id now (review finding #15), so a
        // single `remove` drops the entry -- otherwise a stale id mapping would
        // outlive the physical output and a reconnected connector reusing the
        // name could rehydrate a dead id.
        self.disabled_outputs.remove(&id);
        // `remove` on an unknown id, not indexing: this can name an output
        // this handler was never told about (see the library's own docs), and
        // a panic here aborts.
        if let Some(index) = self.output_ids.remove(&id) {
            if let Some(dead) = self.outputs.remove(&index) {
                // Every window whose frame center sat on the dead output
                // moves onto the survivor before anything else notices the
                // output is gone.
                self.migrate_windows_from(dead.geometry);
            }
            // The gone output's wallpaper node (if any) must go with it --
            // nothing else calls `sync_wallpaper_nodes` when the output set
            // shrinks, and a stale node would otherwise sit in the scene
            // pointing at nothing.
            self.sync_wallpaper_nodes();

            // Review finding M5a: every layer surface `entry.output ==
            // index` owned is now orphaned exactly like `dead`'s windows
            // were -- re-home them onto a survivor (or leave them parked,
            // with no survivor, for `resolve_orphaned_layers`'s own doc's
            // self-heal via the next `new_output`) so a bar on an
            // undocked external display keeps being configured instead of
            // going silent forever.
            self.resolve_orphaned_layers();
        }
        // Re-advertise even when the id was unknown-to-us: the output is gone
        // from wlroots' layout regardless, and a bound manager client must
        // see the shrunk head set. A no-op when no manager exists.
        if let Some(runtime) = self.wayland.runtime() {
            runtime.update_output_manager_state();
        }
    }

    fn output_configuration_applied(&mut self, heads: Vec<wlr::AppliedHead>) {
        // The crate has already committed each head and applied its layout
        // position by the time this runs (see the trait doc); our job is to
        // re-derive geometry from the new layout, migrate any window off an
        // output that shrank or went dark, persist the result, and
        // re-advertise. Never block on redb here -- `spawn_config_save` does
        // the write off-loop.
        let Some(runtime) = self.wayland.runtime().cloned() else {
            return;
        };

        // Review finding #5 (interactive guard): whether any head in THIS
        // batch is being enabled. A disable that would empty `self.outputs` is
        // only safe to honor when some other head in the same atomic apply is
        // enabling to replace it; otherwise refusing the disable is the only
        // thing standing between the session and an unrecoverable black screen.
        let batch_enables = heads.iter().any(|h| h.enabled);

        // Review finding #6: connector names whose disable we REFUSED below
        // (kept the output live). Their `AppliedHead` reports enabled=false with
        // a 0x0 mode, so persisting it would both flip the record to disabled
        // and wipe the saved mode/scale/transform/position. We skip them in the
        // upsert instead, leaving their existing (enabled, full-mode) persisted
        // entry untouched.
        let mut refused_disables: Vec<String> = Vec::new();

        for head in &heads {
            let Some(head_name) = head.name.as_deref() else {
                continue;
            };
            // name -> our output index (via the name recorded in `new_output`).
            let index = match self
                .outputs
                .iter()
                .find(|(_, o)| o.name == head_name)
                .map(|(i, _)| *i)
            {
                Some(index) => index,
                None => {
                    // Not in the active set. If this head is being ENABLED and
                    // we still track it as a disabled output (found by name ->
                    // its unique id), rehydrate it: a fresh index + surface,
                    // its id re-mapped, and drop it from `disabled_outputs`. It
                    // then rides the geometry/arrange/scene sequence after this
                    // loop like any enabled head. If the name is in neither map
                    // there is nothing we can do without an id -- skip it.
                    if head.enabled
                        && let Some(oid) = self.take_disabled_output(head_name)
                    {
                        let index = self.next_output_index;
                        self.next_output_index += 1;
                        let geometry = runtime
                            .output_layout_box(oid)
                            .map(|(x, y, w, h)| icedtea_contract::Rectangle {
                                x,
                                y,
                                width: w,
                                height: h,
                            })
                            .or_else(|| {
                                (head.width > 0 && head.height > 0).then_some(
                                    icedtea_contract::Rectangle {
                                        x: head.x,
                                        y: head.y,
                                        width: head.width,
                                        height: head.height,
                                    },
                                )
                            })
                            .unwrap_or(icedtea_contract::Rectangle {
                                x: 0,
                                y: 0,
                                width: 1920,
                                height: 1080,
                            });
                        self.create_output(index, geometry);
                        if let Some(surface) = self.outputs.get_mut(&index) {
                            surface.name = head_name.to_string();
                        }
                        self.output_ids.insert(oid, index);
                        // First-frame parity with `new_output` (review finding
                        // #13): `new_output` calls `output.schedule_frame()`
                        // (review finding I1) so a freshly enabled output that
                        // draws nothing still gets a `frame` callback and a
                        // first commit. This rehydrate branch has only the
                        // re-mapped `oid`, so use the id-keyed
                        // `Runtime::schedule_frame` sibling to give the
                        // re-enabled output the same one-time kick rather than
                        // leaving it blank until some incidental damage repaints
                        // it.
                        let _ = runtime.schedule_frame(oid);
                    }
                    continue;
                }
            };

            let old_geometry = self.outputs.get(&index).map(|o| o.geometry);

            if head.enabled {
                // Re-derive from the layout box wlroots just committed; fall
                // back to the head's own reported mode/position, then to the
                // prior box, so the output is never collapsed to nothing.
                let oid = self
                    .output_ids
                    .iter()
                    .find(|(_, i)| **i == index)
                    .map(|(id, _)| *id);
                let geometry = oid
                    .and_then(|oid| runtime.output_layout_box(oid))
                    .map(|(x, y, w, h)| icedtea_contract::Rectangle {
                        x,
                        y,
                        width: w,
                        height: h,
                    })
                    .or_else(|| {
                        (head.width > 0 && head.height > 0).then_some(icedtea_contract::Rectangle {
                            x: head.x,
                            y: head.y,
                            width: head.width,
                            height: head.height,
                        })
                    })
                    .or(old_geometry);
                if let (Some(surface), Some(geometry)) = (self.outputs.get_mut(&index), geometry) {
                    surface.geometry = geometry;
                    // `arrange_layers`/exclusive zones recompute `usable`;
                    // reset it to the full box for now, exactly as a fresh
                    // `create_output` does.
                    surface.usable = geometry;
                }
                // A shrunken output can strand windows -- but `reclaim` judges
                // containment against `self.outputs`, whose LATER heads still
                // hold their pre-apply geometry until this loop reaches them
                // (review finding #10). So it runs exactly once, after the
                // whole batch has been re-derived, not per enabled head here.
            } else {
                // Disabled. Guard (review finding #5): never let a config
                // empty the active set. Refuse this disable when it would drop
                // `self.outputs` to zero AND no other head in this atomic
                // apply is enabling to replace it -- an empty `self.outputs`
                // collapses window placement/migrate/reclaim to an early return
                // and leaves a black screen with no way to recover. The head
                // simply stays active; the persisted (disabled) record is
                // caught again by `new_output`'s all-disabled boot guard.
                if self.disable_would_strand_session(index, batch_enables) {
                    tracing::warn!(
                        %head_name,
                        "refusing to disable the last active output; keeping >=1 enabled to avoid a black screen"
                    );
                    // Review finding #6: mark it so the persist step below does
                    // not record enabled=false (0x0 mode) for the output we just
                    // kept live -- that would lose its saved mode/scale/
                    // transform/position on the next boot.
                    refused_disables.push(head_name.to_string());
                    continue;
                }
                // Drop it from the active set (mirroring `destroyed`) and
                // migrate its windows onto a survivor. Before dropping the id,
                // record it keyed by that id (value = this head's name) in
                // `disabled_outputs` so a later re-enable (handed no id) can
                // rehydrate it -- see that map's doc and the enable branch
                // above.
                self.outputs.remove(&index);
                if let Some(oid) = self
                    .output_ids
                    .iter()
                    .find(|(_, i)| **i == index)
                    .map(|(id, _)| *id)
                {
                    self.disabled_outputs.insert(oid, head_name.to_string());
                    self.output_ids.remove(&oid);
                }
                if let Some(old) = old_geometry {
                    self.migrate_windows_from(old);
                }
            }
        }

        // Now that EVERY head has been re-derived (review finding #10), re-home
        // windows stranded off every output's box exactly once, judging
        // containment against the fully-updated `self.outputs` rather than a
        // half-updated one. Running it per enabled head inside the loop judged
        // against later heads' stale pre-apply geometry.
        self.reclaim_offscreen_windows();

        // Recompute panel exclusive zones and maximized-window rects against
        // the re-derived geometry. `resolve_orphaned_layers` below early-returns
        // when nothing was orphaned, so a pure reposition/resize (which orphans
        // no layer) would otherwise leave `usable` stuck at the full box and
        // exclusive zones / maximized windows stale. Mirrors the
        // create_output -> arrange_layers -> sync sequence `new_output` runs.
        self.arrange_layers();

        // Re-run the scene sequence a hotplug runs, so windows and layers
        // settle onto the re-derived geometry.
        self.sync_wallpaper_nodes();
        self.sync_scene();
        self.emit_pending();
        self.resolve_orphaned_layers();

        // Persist: upsert each applied head into `config.displays`, then
        // write off-loop. Review finding #6: skip any head whose disable we
        // refused -- persisting its enabled=false/0x0 record would corrupt the
        // saved config; its existing (enabled, full-mode) entry is left intact.
        let heads_to_persist: Vec<wlr::AppliedHead> = heads
            .iter()
            .filter(|h| match h.name.as_deref() {
                Some(name) => !refused_disables.iter().any(|r| r == name),
                None => true,
            })
            .cloned()
            .collect();
        self.upsert_displays_from_heads(&heads_to_persist);
        self.spawn_config_save();

        // The trait doc REQUIRES this once the layout is settled and
        // persisted, so other bound managers see the fresh state + serial.
        runtime.update_output_manager_state();
    }

    /// A `gamma-control-v1` client set (or wlroots otherwise changed) this
    /// output's gamma ramp. Notification-only (see the trait doc):
    /// `Runtime::create_gamma_control_manager` wires the manager straight
    /// into the scene, which applies the ramp (or rejects it) on its own
    /// commit path before this ever runs -- there is nothing to stash or
    /// apply here, only a trace for anyone reading logs.
    fn gamma_control_changed(&mut self, output: wlr::OutputId) {
        tracing::debug!(?output, "gamma ramp changed");
    }
}

impl wlr::FdHandler for State {
    fn fd_ready(
        &mut self,
        source: wlr::SourceId,
        fd: std::os::fd::BorrowedFd<'_>,
        _readiness: wlr::Readiness,
    ) {
        // Drain whatever byte(s) woke this source before acting on it, for
        // every arm below: libwayland's loop is level-triggered, so a
        // handler that leaves data behind is called again every turn
        // forever.
        let mut buf = [0u8; 32];
        if Some(source) == self.shutdown_source {
            let _ = rustix::io::read(fd, &mut buf);
            tracing::info!("shutdown signal received");
            self.quitting = true;
            return;
        }
        if Some(source) == self.cmd_wake_source {
            let _ = rustix::io::read(fd, &mut buf);
            self.drain_pending_commands();
            return;
        }
        if Some(source) == self.config_reload_wake_source {
            let _ = rustix::io::read(fd, &mut buf);
            self.drain_config_reload();
            return;
        }
        if Some(source) == self.wallpaper_wake_source {
            let _ = rustix::io::read(fd, &mut buf);
            self.drain_wallpaper();
        }
    }
}

impl wlr::LoopHandler for State {
    fn should_stop(&mut self) -> bool {
        self.turns += 1;
        // Not called from C — this is the one handler that may panic safely —
        // but it is still not a place to. `fd_ready`'s `cmd_wake_source` and
        // `config_reload_wake_source` arms above are what actually pull the
        // loop out of a blocked `dispatch(-1)` the instant either channel
        // has something; these two calls are a backstop that runs on every
        // turn regardless of *why* it woke, so a command or reload result
        // is never left sitting past whatever else already woke the loop
        // for an unrelated reason (a frame, input, the shutdown source).
        self.drain_pending_commands();
        self.drain_config_reload();
        // Review finding I3: the wallpaper decode channel gets the same
        // per-turn backstop as the other two, for the same reason -- a
        // result that arrived while something unrelated woke the loop must
        // not sit past this turn just because it wasn't *this* wake source.
        self.drain_wallpaper();
        self.quitting
    }
}

impl wlr::ToplevelHandler for State {
    fn new_toplevel(&mut self, toplevel: &wlr::Toplevel<'_>) {
        // Nothing is created in the model yet: at this point the client has
        // sent no buffer and no size, and it may never map at all. The model
        // row is created on `mapped`, which is the first moment a window
        // genuinely exists on screen — and the moment the smithay
        // implementation's `new_toplevel` was standing in for.
        let _ = toplevel;
    }

    fn initial_commit(&mut self, toplevel: &wlr::Toplevel<'_>) {
        // xdg-shell requires a configure here. Staging the model's
        // placeholder size means the client's very first buffer is already
        // the right size, rather than being resized one frame later --
        // except the size the client must be told is the *content* size,
        // not the model's frame size: `mapped`/`new_toplevel` hasn't run
        // yet, so there is no model window and no `client_decorations_requested`
        // to read, but a window's default SSD-or-not answer only depends on
        // `app_id` (`decoration::has_ssd`'s `requested: None` case), which
        // `Toplevel::app_id` already has at this point. Skipping
        // `content_rect` here would configure a default (SSD) window's
        // client one `TITLE_BAR_HEIGHT` too tall -- the exact one-frame-late
        // resize this comment already claims not to have.
        let key = crate::wayland::ToplevelKey::new(toplevel.id());
        let app_id = toplevel.app_id().unwrap_or_default();
        // Review finding L1: this is also the first -- and only -- moment
        // before the client's first frame at which both halves of the
        // decoration answer are known: the app-id (from `toplevel`) and any
        // preference the client stated on its decoration object (parked in
        // `pending_decorations`, since `set_mode` precedes this commit).
        // Answering here rather than from `request_decoration_mode` is what
        // makes the client's *first* decoration configure the correct one:
        // an earlier `set_decoration_mode` is staged rather than sent, and
        // staging is last-write-wins, so this overwrites the provisional
        // answer instead of adding a second configure after it.
        let requested = self.pending_decorations.get(&key).copied().flatten();
        let ssd = crate::decoration::has_ssd(&app_id, requested, false);
        self.wayland.set_decoration_mode(
            key,
            if ssd {
                wlr::DecorationMode::ServerSide
            } else {
                wlr::DecorationMode::ClientSide
            },
        );
        let Some(runtime) = self.wayland.runtime() else {
            return;
        };
        let placeholder = icedtea_contract::Rectangle {
            x: 0,
            y: 0,
            width: PLACEHOLDER_SIZE.0,
            height: PLACEHOLDER_SIZE.1,
        };
        let content = crate::decoration::content_rect(placeholder, ssd);
        runtime.set_toplevel_size(toplevel.id(), content.width, content.height);
    }

    fn mapped(&mut self, toplevel: &wlr::Toplevel<'_>) {
        let id = toplevel.id();
        let key = crate::wayland::ToplevelKey::new(id);
        if let Some(window_id) = self.wayland.window_for(key) {
            // Remapped after an unmap: the model row survived, so this is a
            // visibility change rather than a new window. Task 14: the
            // model itself now tracks that state, so flip it back before
            // syncing -- otherwise `is_visible`/`alt_tab_entries` would keep
            // treating a window the client just remapped as still absent.
            tracing::info!(?id, ?window_id, "toplevel remapped");
            self.window_manager.set_mapped(window_id, true);
            self.sync_window_to_scene(window_id);
            self.emit_pending();
            return;
        }
        let app_id = toplevel.app_id().unwrap_or_default();
        let title = toplevel.title().unwrap_or_default();
        let pid = toplevel.pid().unwrap_or(0);
        tracing::info!(?id, %app_id, %title, pid, "toplevel mapped");
        self.new_toplevel(key, &app_id, &title, pid);
    }

    fn unmapped(&mut self, id: wlr::ToplevelId) {
        // An unmap is not a destroy: the client may map again with the same
        // id. Hide it, and let the model keep the row.
        //
        // Task 14: the model now has its own "unmapped" concept
        // (`Window::mapped`, `WindowManager::set_mapped`) instead of the
        // ledgered gap task 11 documented here -- `is_visible` and
        // `alt_tab_entries` both gate on it, and `focus`/
        // `focus_mru_in_workspace` refuse an unmapped window as a candidate.
        // So an unmap that hits the focused window now really does move
        // focus to the next mapped candidate on the same workspace, the
        // same way `set_minimized_and_reconcile` already does for
        // minimizing the focused window; `window_manager.get(window)` still
        // returns the row (unchanged geometry/title/etc, per `set_mapped`'s
        // own doc), and `is_backed` stays `true` -- only mapped-ness and
        // its consequences on visibility/focus change.
        //
        // The seat trap task 11 closed still matters here too:
        // `sync_focus_change` re-derives keyboard focus from the model
        // (`sync_seat_focus`, at the tail of `sync_window_to_scene` or
        // directly when nothing is focused), so a focused window unmapping
        // with nothing to hand focus to still clears the seat rather than
        // leaving it pointed at a hidden surface.
        tracing::info!(?id, "toplevel unmapped");
        let key = crate::wayland::ToplevelKey::new(id);
        let Some(window) = self.wayland.window_for(key) else {
            return;
        };
        self.wayland.set_visible(window, false);
        let previous = self.focused_id();
        self.window_manager.set_mapped(window, false);
        // Review finding I2: this used to re-pick only when the unmapping
        // window was the *active* workspace's focus, so an unmap on an
        // inactive workspace left that workspace's `focused_window` pointing
        // at the now-unmapped row -- dead to the seat, but still live to
        // `apply_action("close"/"maximize"/"fullscreen"/"snap")` and the
        // decoration actions once the user switched back.
        // `release_focus` resolves the window's *own* workspace, so the
        // active case behaves exactly as before and the inactive case is
        // covered too; when no successor exists it clears the pointer,
        // matching `remove_window`.
        self.window_manager.release_focus(window);
        self.sync_focus_change(previous);
        self.emit_pending();
    }

    fn title_changed(&mut self, toplevel: &wlr::Toplevel<'_>) {
        let id = toplevel.id();
        let key = crate::wayland::ToplevelKey::new(id);
        let title = toplevel.title().unwrap_or_default();
        tracing::info!(?id, %title, "toplevel title changed");
        self.toplevel_title_changed(key, &title);
    }

    fn toplevel_destroyed(&mut self, id: wlr::ToplevelId) {
        // Safe against an id we were never told about (the library documents
        // that this can happen): `forget_toplevel` resolves through the map
        // and returns early on a miss, and never indexes.
        tracing::info!(?id, "toplevel destroyed");
        self.forget_toplevel(crate::wayland::ToplevelKey::new(id));
    }

    fn request_maximize(&mut self, toplevel: &wlr::Toplevel<'_>, maximize: bool) {
        let key = crate::wayland::ToplevelKey::new(toplevel.id());
        if !self.reconcile_maximized(key, maximize) {
            // The model didn't change (unknown toplevel, no output yet, or
            // already at the requested state) -- the dispatch layer answers
            // with a bare configure regardless, so xdg-shell's "every
            // request gets a configure" contract is honored either way.
            // Nothing to do here.
        }
    }

    fn request_fullscreen(&mut self, toplevel: &wlr::Toplevel<'_>, fullscreen: bool) {
        let key = crate::wayland::ToplevelKey::new(toplevel.id());
        if !self.reconcile_fullscreen(key, fullscreen) {
            // Same contract as `request_maximize` above: dispatch already
            // answers with a bare configure when the model didn't change.
        }
    }

    /// Re-review: gated on the lock, like `request_activate` and the F1
    /// click-to-focus path. `begin_client_move` runs both
    /// `WindowManager::focus` and `sync_focus_change`, so an
    /// `xdg_toplevel.move` arriving from a background client behind the lock
    /// screen would re-pick who owns the keyboard the instant the session
    /// unlocks -- and start a drag against geometry the user cannot see.
    /// Nothing a hidden client asks for may move focus.
    ///
    /// A2 batch-2 follow-up: proven end to end by
    /// `a_locked_session_refuses_a_client_move_request` (in
    /// `compositor/tests/compat_protocols.rs`), which drives this handler
    /// from a real client holding a real implicit pointer grab. Deleting the
    /// gate below turns it red. The
    /// `client_grab_requests_allowed`-only unit test could not: with no
    /// runtime attached, `begin_client_move` returns at its
    /// `pointer_pressed`/`runtime()`/`window_at_point` guards long before
    /// `focus`, so a unit test cannot tell a refusal from those.
    fn request_move(&mut self, id: wlr::ToplevelId) {
        if !self.client_grab_requests_allowed() {
            tracing::debug!(
                ?id,
                "refusing a client move request while the session is locked"
            );
            return;
        }
        let key = crate::wayland::ToplevelKey::new(id);
        let Some(window_id) = self.wayland.window_for(key) else {
            return;
        };
        self.begin_client_move(window_id);
    }

    /// Locked-session gate, for the same reason as `Self::request_move`.
    ///
    /// The reason is *nearly* the same: `begin_client_resize` does not
    /// focus, it only starts a `ResizeMachine` grab. What a hidden client
    /// gains without this gate is therefore the geometry, not the keyboard
    /// -- it resizes itself behind the lock screen off pointer motion the
    /// user believes the lock surface is consuming.
    /// `a_locked_session_refuses_a_client_resize_request` asserts exactly
    /// that geometry, since focus would say nothing here.
    fn request_resize(&mut self, id: wlr::ToplevelId, edges: wlr::Edges) {
        if !self.client_grab_requests_allowed() {
            tracing::debug!(
                ?id,
                ?edges,
                "refusing a client resize request while the session is locked"
            );
            return;
        }
        let key = crate::wayland::ToplevelKey::new(id);
        let Some(window_id) = self.wayland.window_for(key) else {
            return;
        };
        self.begin_client_resize(window_id, edges);
    }

    /// xdg-decoration negotiation, both halves in one place: record what the
    /// client asked for in the model, then answer with what this compositor
    /// is actually going to do.
    ///
    /// The answer is `decoration::has_ssd`, not an echo of the preference:
    /// that predicate is already the single definition of "do we draw a
    /// title bar for this window" (`sync_window_to_scene`, `content_rect`,
    /// the hit-test), so routing the reply through it is what keeps the
    /// client's belief and the compositor's own drawing from ever
    /// disagreeing. A client asking for client-side decorations gets them
    /// (`is_csd` honors an explicit request); a client asking for
    /// server-side, or stating no preference at all, gets our band.
    ///
    /// A toplevel with no model window yet -- the normal case, since a
    /// decoration is created before the initial commit and `mapped` has not
    /// run -- has its preference parked in `pending_decorations` and is
    /// still answered here, because a request must never be left unanswered
    /// (the dispatch layer would otherwise impose its own blanket
    /// server-side default). That answer is made from the preference alone:
    /// there is no app-id to read from a bare `ToplevelId`. It is only
    /// provisional, and it is *staged* rather than sent while the surface is
    /// uninitialized, so `initial_commit` -- which has both the app-id and
    /// the parked preference -- overwrites it before anything reaches the
    /// client (review finding L1).
    fn request_decoration_mode(
        &mut self,
        id: wlr::ToplevelId,
        preference: Option<wlr::DecorationMode>,
    ) {
        // `client_decorations_requested` is the model's own spelling of the
        // same three-valued answer: "the client wants to draw them itself",
        // "the client wants the server to", "the client did not say".
        let requested = match preference {
            Some(wlr::DecorationMode::ClientSide) => Some(true),
            Some(wlr::DecorationMode::ServerSide) => Some(false),
            None => None,
        };
        let key = crate::wayland::ToplevelKey::new(id);
        let window = self.wayland.window_for(key);
        let (app_id, fullscreen) = match window.and_then(|w| self.window_manager.get(w)) {
            Some(w) => (w.app_id.clone(), w.fullscreen),
            None => (String::new(), false),
        };
        match window {
            Some(window) => {
                self.window_manager
                    .set_client_decorations_requested(window, requested);
            }
            None => {
                self.pending_decorations.insert(key, requested);
            }
        }

        let ssd = crate::decoration::has_ssd(&app_id, requested, fullscreen);
        let mode = if ssd {
            wlr::DecorationMode::ServerSide
        } else {
            wlr::DecorationMode::ClientSide
        };
        self.wayland.set_decoration_mode(key, mode);

        // The band may have to appear or vanish, and the client's content
        // rect changes with it. A window that isn't in the model yet has
        // nothing to sync -- its first sync comes with `mapped`.
        if let Some(window) = window {
            self.sync_window_to_scene(window);
            self.emit_pending();
        }
    }

    /// A client created a wlr-layer-shell surface. Resolve its output --
    /// what it asked for (`output_id` via `output_ids`), or the output
    /// under the pointer (which itself falls back to the lowest index) --
    /// and park it under the `NO_OUTPUT` sentinel if neither resolves,
    /// i.e. no output exists yet at all (review finding M5): the surface
    /// is *not* dropped, because `State::resolve_orphaned_layers` re-homes
    /// every `NO_OUTPUT` entry (and configures it) the moment `new_output`
    /// next fires -- dropping it here left a bar launched before the first
    /// output settled waiting forever for a configure that would never
    /// come, even after one did arrive. Always record it and always answer
    /// -- `configure_layer` is a no-op, correctly, on the sentinel, and is
    /// otherwise safe even before this surface's first commit
    /// (`Runtime::configure_layer_surface` stages pre-initial-commit
    /// answers rather than sending them, see its own doc) -- it is
    /// mandatory: nothing else in this crate's dispatch layer answers a
    /// layer surface that no handler ever does.
    fn new_layer_surface(&mut self, surface: &wlr::LayerSurface<'_>) {
        let id = surface.id();
        let client_output = surface
            .output_id()
            .and_then(|oid| self.output_ids.get(&oid).copied());
        let output = client_output.or_else(|| self.output_for_pointer());
        let sequence = self.next_layer_sequence;
        self.next_layer_sequence += 1;
        // H2: bound the client-controlled exclusive zone at capture time --
        // see `clamp_exclusive_zone`'s own doc.
        let exclusive = clamp_exclusive_zone(
            surface.exclusive_zone(),
            output.and_then(|idx| self.outputs.get(&idx)),
        );
        self.layers.insert(
            id,
            LayerEntry {
                output: output.unwrap_or(NO_OUTPUT),
                sequence,
                layer: surface.layer(),
                anchor: surface.anchor(),
                exclusive,
                size: surface.desired_size(),
                // Always `false` here regardless of what the client asked
                // for -- `keyboard_interactive` reads `current`, which is
                // entirely zeroed until this surface's first commit (see
                // that accessor's own doc). `layer_surface_commit` is
                // where the real value lands.
                interactive: false,
                // False until `layer_surface_mapped` (review finding J2):
                // a surface with no buffer yet must not reserve space.
                mapped: false,
                last_configured: None,
                // N8: no accessor to capture this from -- see
                // `LayerEntry::margin`'s own doc.
                margin: (0, 0, 0, 0),
            },
        );
        // The client left the output unset: this crate chose one on its
        // behalf (`client_output.is_none()`, above), so the raw layer
        // surface's own `output` field needs to agree with the model --
        // `set_layer_surface_output` is the 0.20.12 API for that
        // assignment. A no-op, correctly, with no runtime attached (every
        // unit test in this file) or no live output to name yet (`output`
        // is `None`, parked under `NO_OUTPUT`; `resolve_orphaned_layers`
        // picks this back up the moment one exists).
        if client_output.is_none()
            && let (Some(idx), Some(runtime)) = (output, self.wayland.runtime())
            && let Some(output_id) = self.wlr_output_id_for(idx)
        {
            runtime.set_layer_surface_output(id, output_id);
        }
        self.configure_layer(id);
    }

    /// Every commit of an already-announced layer surface: refresh the
    /// entry from the surface's current request (anchors, exclusive zone
    /// and size all commonly change after mapping) and answer again --
    /// `configure_layer` reads the entry it just updated, so a client that
    /// re-anchors gets reconfigured for its new placement, not its old
    /// one. Then re-derive every output's usable area, since this
    /// surface's exclusive zone may have just changed.
    ///
    /// A miss on `self.layers` cannot happen for any id this crate ever
    /// hands a handler -- `new_layer_surface` always inserts an entry now
    /// (see its own doc on the `NO_OUTPUT` sentinel) -- but the `let …
    /// else` stays as the defensive, panic-free answer for an id this
    /// handler was never told about, the same posture every other handler
    /// in this file takes.
    fn layer_surface_commit(&mut self, surface: &wlr::LayerSurface<'_>) {
        let id = surface.id();
        let Some(output_idx) = self.layers.get(&id).map(|e| e.output) else {
            return;
        };
        // H2: bound the client-controlled exclusive zone at capture time --
        // see `clamp_exclusive_zone`'s own doc. Looked up before the
        // `entry` borrow below, since both read `self.outputs`.
        let exclusive =
            clamp_exclusive_zone(surface.exclusive_zone(), self.outputs.get(&output_idx));
        let Some(entry) = self.layers.get_mut(&id) else {
            return;
        };
        entry.layer = surface.layer();
        entry.anchor = surface.anchor();
        entry.exclusive = exclusive;
        entry.size = surface.desired_size();
        let was_interactive = entry.interactive;
        entry.interactive = surface.keyboard_interactive();
        let now_interactive = entry.interactive;
        self.configure_layer(id);
        self.arrange_layers();
        // N9: a surface that only becomes keyboard-interactive *after* it
        // mapped (a menu that opens with no interactivity, then flips it
        // on once the user drives it) never passes through
        // `layer_surface_mapped`'s own take-focus branch, since that only
        // ever runs once, at map time. See `sync_layer_interactive_focus`'s
        // own doc for the take/release rule this drives.
        self.sync_layer_interactive_focus(id, was_interactive, now_interactive);
    }

    /// The layer surface now has a buffer and is on screen: it starts
    /// reserving its exclusive zone (review finding J2 -- `mapped = true`
    /// before `arrange_layers`, which is what makes this a real fold
    /// rather than the guaranteed no-op it used to be). A
    /// keyboard-interactive surface then takes seat keyboard focus --
    /// `focus_layer_keyboard` refuses an unmapped surface (`None` while
    /// unmapped, per its own doc), which is exactly why this waits for
    /// `mapped` rather than acting from `layer_surface_commit`, where
    /// `interactive` first becomes known but the surface may still be
    /// unmapped.
    fn layer_surface_mapped(&mut self, id: wlr::LayerSurfaceId) {
        if let Some(entry) = self.layers.get_mut(&id) {
            entry.mapped = true;
        }
        self.arrange_layers();
        let Some(entry) = self.layers.get(&id) else {
            return;
        };
        if !entry.interactive {
            return;
        }
        // Through the focus helper: an interactive panel taking the keyboard
        // off a composing text input hides its overlay. No runtime (every
        // unit test) returns silently, exactly like the old early return —
        // only a live miss logs.
        let Some(took) = self.change_keyboard_focus(|rt| rt.focus_layer_keyboard(id)) else {
            return;
        };
        if took.is_some() {
            self.layer_focus = Some(id);
        } else {
            // Finding 8, errors: an interactive panel that just mapped and
            // failed to take keyboard focus used to fail silently, leaving
            // no trace of why an auto-hide launcher/locker never got
            // keyboard input.
            tracing::debug!(
                ?id,
                "interactive layer surface mapped but did not take keyboard focus"
            );
        }
    }

    /// The layer surface should no longer be displayed (not a destroy --
    /// see this method's own trait doc; the entry survives so a remap
    /// finds it again). Stops reserving its exclusive zone (review finding
    /// J2 -- `mapped = false` before `arrange_layers`, which is what makes
    /// this call a real fold instead of a no-op: an unmapped panel used to
    /// leave a permanent hole in the workspace until it was destroyed
    /// outright). If it held keyboard focus, hand focus back to whatever
    /// the model says is focused: `sync_seat_focus` re-derives the seat's
    /// keyboard target from `window_manager` rather than leaving it
    /// pointed at a surface that just stopped being shown.
    fn layer_surface_unmapped(&mut self, id: wlr::LayerSurfaceId) {
        if let Some(entry) = self.layers.get_mut(&id) {
            entry.mapped = false;
            // Review finding I1: wlroots resets the surface's `initialized`
            // flag on every unmap (documented in wlr 0.20.11's `layer.rs`;
            // the crate deliberately refuses to synthesize a fallback
            // configure), so a remap needs a *fresh* mandatory configure --
            // but the remap commit recomputes the identical placement, and
            // `configure_layer`'s storm guard would suppress the send
            // against a surviving `last_configured`. The surface would then
            // never become `mapped`, which is the only thing
            // `arrange_layers`' sweep reconfigures, and the client hangs
            // forever: the auto-hide-panel / toggle-launcher sequence.
            // Forgetting the placement here is what makes the next
            // `configure_layer` unconditionally send.
            entry.last_configured = None;
        }
        if self.layer_focus == Some(id) {
            self.layer_focus = None;
            self.sync_seat_focus();
        }
        self.arrange_layers();
    }

    /// The layer surface is gone for good. Same focus hand-back as
    /// `layer_surface_unmapped` (a destroy while mapped and focused is
    /// legal -- a client can drop its surface without ever unmapping it
    /// first), plus removing the entry, which `layer_surface_unmapped`
    /// deliberately does not. Panic-free on an id this handler was never
    /// told about (the trait's own doc: this can happen) -- both the
    /// focus check and `HashMap::remove` are no-ops on a miss.
    fn layer_surface_destroyed(&mut self, id: wlr::LayerSurfaceId) {
        self.forget_popups_of_root(PopupRoot::Layer(id));
        if self.layer_focus == Some(id) {
            self.layer_focus = None;
            self.sync_seat_focus();
        }
        self.layers.remove(&id);
        self.arrange_layers();
    }

    /// A client created a popup. Resolve its host, record it, and place it.
    ///
    /// The configure attempted here is normally a no-op: the popup's surface
    /// is not `initialized` until its first commit, and the library skips the
    /// call rather than tripping wlroots' own assert (contract §1.2). It is
    /// made anyway so that a popup which *is* already initialized -- a
    /// reposition racing a re-announce -- is placed at once rather than a
    /// round trip later. `popup_initial_commit` is what actually lands the
    /// constraint box on the client's first configure.
    fn new_popup(&mut self, popup: &wlr::Popup<'_>) {
        let key = crate::wayland::PopupKey::new(popup.id());
        let Some(host) = self.popup_host_for(popup.parent()) else {
            tracing::debug!(?key, "popup on a parent this compositor does not model");
            return;
        };
        // Contract §11 E19: the crate folds an unrecognised
        // `xdg_positioner_anchor`/`_gravity` into its `None` variant and binds
        // no logging symbol of its own, so this is the first place a malformed
        // positioner becomes an observable compositor decision -- and the only
        // place it can be logged at all. What is recoverable is the whole
        // ruleset the placement about to run is derived from; the raw wire
        // value is *not*, because `PositionerAnchor::from_raw` /
        // `PositionerGravity::from_raw` make an unknown value indistinguishable
        // from a legitimate `NONE` before it ever reaches this crate (see the
        // fix-report deviation this note is paired with). An `anchor: None` or
        // `gravity: None` in this line is therefore "the client sent NONE, or
        // sent something this protocol version does not define" -- which is
        // exactly the pair of cases a reader debugging a mis-placed menu needs
        // to see.
        let rules = popup.positioner_rules();
        tracing::debug!(
            ?key,
            ?host,
            anchor = ?rules.anchor,
            gravity = ?rules.gravity,
            size = ?rules.size,
            anchor_rect = ?rules.anchor_rect,
            reactive = rules.reactive,
            "placing a popup against its client's positioner"
        );
        self.record_popup(key, host, popup.grab_requested());
        self.configure_popup_now(key);
    }

    /// The popup's first commit: unconstrain before the library answers it.
    ///
    /// Deviation D2. xdg-shell requires the compositor to answer a popup's
    /// first commit or it never maps, and the library does that
    /// unconditionally right after this returns (contract §1.7) -- so this is
    /// the one moment at which the compositor's constraint box can reach the
    /// client's *first* configure rather than its second.
    ///
    /// Also where `reconcile_popup_grab` runs: xdg-shell requires `grab` to
    /// precede this commit, so it is the first point at which
    /// `Popup::grab_requested()` is trustworthy (see `record_popup`'s doc).
    fn popup_initial_commit(&mut self, popup: &wlr::Popup<'_>) {
        self.reconcile_popup_grab(popup);
        self.configure_popup_now(crate::wayland::PopupKey::new(popup.id()));
    }

    /// The popup now has a buffer on screen: it starts answering
    /// `popup_at_point`, and its committed geometry is finally readable.
    fn popup_mapped(&mut self, id: wlr::PopupId) {
        let key = crate::wayland::PopupKey::new(id);
        if let Some(entry) = self.popups.get_mut(&key) {
            entry.mapped = true;
        }
        self.refresh_popup_geometry(key);
    }

    /// The popup is no longer displayed. Not a destroy -- the entry survives,
    /// mirroring `layer_surface_unmapped`.
    fn popup_unmapped(&mut self, id: wlr::PopupId) {
        if let Some(entry) = self.popups.get_mut(&crate::wayland::PopupKey::new(id)) {
            entry.mapped = false;
        }
    }

    /// The client sent `xdg_popup.reposition` with a new positioner: re-run
    /// placement against the current constraint box. The library sends
    /// `xdg_popup.repositioned` with the client's token off the configure
    /// this triggers -- the compositor forges nothing.
    fn popup_reposition(&mut self, popup: &wlr::Popup<'_>) {
        let key = crate::wayland::PopupKey::new(popup.id());
        if !self.popups.contains_key(&key) {
            return;
        }
        self.configure_popup_now(key);
        self.refresh_popup_geometry(key);
    }

    /// The popup is gone for good. Panic-free on an id this handler was never
    /// told about -- contract §1.3's caveat, the same posture
    /// `layer_surface_destroyed` takes.
    ///
    /// When this empties the chain, focus goes back to the chain's root
    /// (focus rule 3, deviation D3).
    fn popup_destroyed(&mut self, id: wlr::PopupId) {
        let key = crate::wayland::PopupKey::new(id);
        let Some(root) = self.popup_root(key) else {
            return;
        };
        self.forget_popup(key);
        if self.popup_chain(root).is_empty() {
            self.restore_focus_after_popups();
        }
    }

    // --- Xwayland (X11) ------------------------------------------------------
    //
    // M2 — managed-window parity. A *managed* (non-override-redirect) X11
    // window is first-class alongside xdg toplevels: it enters the same
    // `WindowManager` model and is driven through the same
    // `sync_window_to_scene` path, so SSD, focus/activation, interactive and
    // client-initiated move/resize, maximize/fullscreen/minimize, workspaces,
    // snap, cascade, alt-tab and MRU focus all work through one code path. The
    // only Xwayland-specific bookkeeping here is the `XwaylandSurfaceId ->
    // WindowId` side-table (`xwayland_windows`), the reverse of the
    // `wayland::bind_x11` binding the outbound seam dispatches on; the WM logic
    // itself is never re-implemented per surface kind. State is pushed back to
    // the X11 surface through the `wayland` seam's `SurfaceKey::X11` arms
    // (configure to the SSD content rect, activate, set maximized/fullscreen,
    // scene position/visibility/raise). Override-redirect surfaces stay
    // unmanaged (M3); the crate renders them from their own scene node.

    fn xwayland_ready(&mut self, display_name: Option<&str>) {
        tracing::info!(?display_name, "Xwayland ready");
        // The crate has already pointed Xwayland at this runtime's seat by now
        // (so the clipboard/primary/DND bridge is live). `DISPLAY` and the X11
        // cursor hints are published at *boot* (see `publish_xwayland_env`),
        // where the lazy manager has already reserved the display socket and no
        // other thread yet exists — the only work left for `ready`, once the X
        // server is actually running, is the `Xft.dpi` hint (an X property
        // write, not a process-env mutation, so it has no getenv/setenv race).
        self.xwayland_display = display_name.map(str::to_owned);
        if let Some(name) = display_name {
            // Publish the X11 HiDPI hint (`Xft.dpi` in the root
            // `RESOURCE_MANAGER`) so X11 toolkits size for the output scale.
            Self::export_x11_dpi(name.to_owned(), self.primary_output_scale());
        }
    }

    fn xwayland_surface_mapped(&mut self, surface: &wlr::XwaylandSurface<'_>) {
        let sid = surface.id();
        if surface.override_redirect() {
            // Override-redirect pop-ups (menus, tooltips, combo/dropdown lists,
            // drag icons) are unmanaged: placed at their own client coordinates,
            // in the band above managed toplevels, with no SSD, tracked in the
            // OR side-table rather than the `Window` model (M3, Decision 4).
            self.map_override_redirect(surface);
            return;
        }
        if let Some(&window_id) = self.xwayland_windows.get(&sid) {
            // Remapped after an unmap while the row survived (an X11 window can
            // unmap and map again keeping its id). Treat it as a visibility
            // change, matching the xdg `mapped` remap arm.
            self.window_manager.set_mapped(window_id, true);
            self.sync_window_to_scene(window_id);
            self.emit_pending();
            return;
        }
        self.add_managed_x11(surface);
    }

    fn xwayland_surface_unmapped(&mut self, id: wlr::XwaylandSurfaceId) {
        // An X11 unmap removes the model row. Idempotent against an id we do
        // not know (a surface that never mapped, or a double unmap/destroy):
        // the take-or-return guard makes it a no-op. (Parity note: xdg keeps
        // the row on unmap and hides it; X11 windows are removed because a
        // re-map mints a fresh model row, which is simpler and correct for the
        // less common X11 remap case.) The id is *either* a managed window or an
        // OR pop-up, never both, so both removals are tried — each is a no-op on
        // the wrong table. An OR menu's unmap is exactly how it is dismissed.
        self.remove_xwayland_window(id);
        self.remove_override_redirect(id);
    }

    fn xwayland_surface_destroyed(&mut self, id: wlr::XwaylandSurfaceId) {
        // Same removal as unmap, and equally safe against an unknown id (the
        // trait documents that `destroyed` may name one we were never told
        // about).
        self.remove_xwayland_window(id);
        self.remove_override_redirect(id);
    }

    fn xwayland_title_changed(&mut self, surface: &wlr::XwaylandSurface<'_>) {
        let Some(&window_id) = self.xwayland_windows.get(&surface.id()) else {
            return;
        };
        let title = surface.title().unwrap_or_default();
        if self.window_manager.set_title(window_id, title).is_some() {
            self.sync_window_to_scene(window_id);
            self.emit_pending();
        }
    }

    fn xwayland_class_changed(&mut self, surface: &wlr::XwaylandSurface<'_>) {
        let Some(&window_id) = self.xwayland_windows.get(&surface.id()) else {
            return;
        };
        let app_id = surface
            .class()
            .or_else(|| surface.instance())
            .unwrap_or_default();
        // `app_id` drives `decoration::has_ssd`, so a class change can flip a
        // window between decorated and undecorated; resync when it actually
        // changed (`set_app_id` returns `None` on a no-op, and emits nothing —
        // the contract has no `app_id` update field, but a snapshot reads the
        // model fresh).
        if self.window_manager.set_app_id(window_id, app_id).is_some() {
            self.sync_window_to_scene(window_id);
            self.emit_pending();
        }
    }

    fn xwayland_request_configure(&mut self, id: wlr::XwaylandSurfaceId, geometry: wlr::Box2D) {
        // An override-redirect pop-up self-positioning: honour it verbatim —
        // an OR surface owns its own coordinates (a menu that follows its
        // anchor, a tooltip that repositions). Move the scene node to the new
        // client coordinates and keep the side-table's geometry in step, so a
        // later probe/reposition reads the truth. No SSD, no model row.
        if let Some(or) = self.override_redirect.get_mut(&id) {
            or.geometry = Rectangle {
                x: geometry.x,
                y: geometry.y,
                width: geometry.width.max(1),
                height: geometry.height.max(1),
            };
            let (x, y) = (or.geometry.x, or.geometry.y);
            if let Some(rt) = self.wayland.runtime() {
                rt.set_xwayland_surface_position(id, x, y);
            }
            return;
        }
        // A managed X11 client self-positioning/resizing. icedtea is a floating
        // WM, so honour it — but the request is in *content* terms (the client
        // knows nothing of the SSD strip), and the model's geometry is the
        // frame, so add the title bar back on for a decorated window before
        // storing it, then let the shared path re-inset and re-configure. Skip
        // a resize while an interactive grab owns the window, so a client
        // configure cannot fight a drag/resize in progress.
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        if self.drag.window_id() == Some(window_id) || self.resize.window_id() == Some(window_id) {
            return;
        }
        let Some(w) = self.window_manager.get(window_id) else {
            return;
        };
        let ssd =
            crate::decoration::has_ssd(&w.app_id, w.client_decorations_requested, w.fullscreen);
        // A maximized/fullscreen window's geometry is WM-owned; ignore a
        // client's attempt to move out of it, matching how xdg toplevels are
        // pinned in those states.
        if w.maximized || w.fullscreen {
            return;
        }
        // The client's request is in *content* terms (an X11 window knows
        // nothing of the SSD strip); convert it to a frame the same way
        // `add_managed_x11` does, through the shared `frame_rect` inverse, so
        // the two never disagree by a title bar (review finding #2).
        let content = icedtea_contract::Rectangle {
            x: geometry.x,
            y: geometry.y,
            width: geometry.width,
            height: geometry.height,
        };
        // Clamp so a self-positioning client cannot push its title bar (its only
        // drag handle and window buttons) off the top/edge of the output, the
        // same clamp map-time placement applies (review finding #9).
        let frame = self.clamp_frame_onto_output(crate::decoration::frame_rect(content, ssd));
        if self.window_manager.set_geometry(window_id, frame).is_some() {
            self.sync_window_to_scene(window_id);
            self.emit_pending();
        }
    }

    fn xwayland_request_move(&mut self, id: wlr::XwaylandSurfaceId) {
        // `_NET_WM_MOVERESIZE` move: drive the same interactive grab an xdg
        // `xdg_toplevel.move` does (`begin_client_move` is WindowId-keyed and
        // gates on the pointer being pressed), which writes geometry back
        // through the shared `sync_window_to_scene` path.
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        self.begin_client_move(window_id);
    }

    fn xwayland_request_resize(&mut self, id: wlr::XwaylandSurfaceId, edges: wlr::Edges) {
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        self.begin_client_resize(window_id, edges);
    }

    fn xwayland_request_maximize(&mut self, id: wlr::XwaylandSurfaceId, maximized: bool) {
        // The same WM state machine xdg `request_maximize` drives — geometry,
        // restore slot and the reflect-back configure all come from
        // `set_maximized_target` -> `sync_window_to_scene`, whose seam pushes
        // `set_xwayland_surface_maximized` back to the client.
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        self.set_maximized_target(window_id, maximized);
    }

    fn xwayland_request_fullscreen(&mut self, id: wlr::XwaylandSurfaceId, fullscreen: bool) {
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        self.set_fullscreen_target(window_id, fullscreen);
    }

    fn xwayland_request_minimize(&mut self, id: wlr::XwaylandSurfaceId, minimized: bool) {
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        self.set_minimized_and_reconcile(window_id, minimized);
    }

    /// An X11 client asked to be activated (`_NET_ACTIVE_WINDOW`).
    ///
    /// Finding F9: this used to run its own, far more permissive policy --
    /// *honor* the request for any mapped, non-minimized window on the
    /// active workspace -- so an X11 client could take the keyboard from
    /// under the user at will, while the very same ask over
    /// `xdg-activation-v1` was refused by `activation_may_steal_focus`.
    /// Two protocols, one behavior: an `_NET_ACTIVE_WINDOW` message carries
    /// no activation token, so it has no seat serial (`has_seat = false`)
    /// and names no requesting toplevel (`requester = None`) -- exactly the
    /// input on which that policy returns `false`. An X11 activate therefore
    /// never steals; it raises the attention hint the shell surfaces
    /// instead, the same fallback `request_activate` uses, so the shell does
    /// not have to care which protocol a window speaks.
    ///
    /// Gated on the lock for the same reason `request_activate` is (finding
    /// F1), and it returns early for a target that already holds focus:
    /// there is nothing to raise and nothing to flag.
    fn xwayland_request_activate(&mut self, id: wlr::XwaylandSurfaceId) {
        if self.session_locked {
            tracing::debug!(
                ?id,
                "ignoring X11 activate request while the session is locked"
            );
            return;
        }
        let Some(&window_id) = self.xwayland_windows.get(&id) else {
            return;
        };
        let focused = self.focused_id();
        if focused == Some(window_id) {
            return;
        }
        let active_workspace = self.window_manager.active_workspace();
        let target_activatable = self
            .window_manager
            .get(window_id)
            .is_some_and(|w| w.mapped && !w.minimized && w.workspace == active_workspace);
        // Written as the real call rather than a hardcoded `false` so the
        // two protocols cannot drift: if the policy ever grows a case a
        // token-less request satisfies, this picks it up for free.
        if activation_may_steal_focus(false, target_activatable, None, focused) {
            if self.window_manager.focus(window_id).is_some() {
                // Explicit toplevel-focus assertion must release any keyboard-
                // interactive layer surface's grab first (review finding #3),
                // exactly as `DbCommand::Focus`, alt-tab and a pointer press do.
                // After the successful `focus`, never before: a focus that did
                // not happen must leave `layer_focus` untouched.
                self.release_layer_focus();
                self.sync_focus_change(focused);
                self.emit_pending();
            }
            return;
        }
        tracing::debug!(
            ?window_id,
            target_activatable,
            "refusing X11 focus steal; flagging attention instead"
        );
        self.raise_attention_if_answerable(window_id);
        self.emit_pending();
    }

    fn xwayland_override_redirect_changed(&mut self, surface: &wlr::XwaylandSurface<'_>) {
        // A live surface can flip its override-redirect flag; the compositor
        // migrates it between the managed `Window` model and the unmanaged OR
        // side-table in both directions, with no leak and no double-track (M3,
        // task 3). The surface handle carries the new flag *and* the identity
        // needed to re-model it on the path it is moving to.
        let sid = surface.id();
        if surface.override_redirect() {
            // managed → OR. Drop the model row first (this also tears down its
            // SSD and reseats focus), then re-add it as an unmanaged pop-up.
            // `map_override_redirect` reparents the still-live scene node up into
            // `Band::Top` and places it at the client's own coordinates.
            // Idempotent if it was somehow already OR: `remove_xwayland_window`
            // no-ops and `map_override_redirect` overwrites the same entry.
            self.remove_xwayland_window(sid);
            self.map_override_redirect(surface);
        } else {
            // OR → managed. Only meaningful if it was actually an OR surface;
            // for anything else (already managed, or unknown) there is nothing
            // to migrate.
            if self.override_redirect.remove(&sid).is_some() {
                // Whether the surface flipping to managed was the OR keyboard
                // holder (top of the stack) matters below.
                let was_holder = self.or_keyboard_stack.last() == Some(&sid);
                // Drop it from the keyboard stack without restoring model focus
                // here — `add_managed_x11` below reseats focus itself, so a
                // `sync_seat_focus` now would only churn.
                self.or_keyboard_stack.retain(|&id| id != sid);
                // Reparent the scene node back down into the toplevel band, then
                // model it through the shared managed path (SSD, window-type
                // placement, focus), exactly like a fresh managed map.
                if let Some(rt) = self.wayland.runtime() {
                    rt.reparent_xwayland_surface_to_band(sid, wlr::Band::Toplevel);
                }
                self.add_managed_x11(surface);
                // If the promoted surface had held the keyboard as a pop-up, it
                // must keep it now that it is a managed toplevel. `add_managed_x11`
                // focuses it in the model, but its `sync_seat_focus` is blocked by
                // any parent menu still on the OR stack (the guard keeps the
                // keyboard on an OR holder), so the new window would be focused yet
                // keyboard-dead. Assert the seat directly. When `sid` was *not* the
                // holder, a parent menu legitimately keeps the keyboard and we
                // leave the seat alone.
                if was_holder && let Some(&wid) = self.xwayland_windows.get(&sid) {
                    self.window_manager.focus(wid);
                    // Same helper as every other focus push: keeping the seat
                    // while promoted must not strand a composing overlay.
                    let key = self.wayland.focus_key(Some(wid));
                    self.change_keyboard_focus(|rt| {
                        crate::wayland::Wayland::apply_focus_key(rt, key)
                    });
                }
            }
        }
    }
}

/// Inputs for placing IME UI: the caret anchor in output coordinates, the
/// output to clamp against, the original surface-local caret rectangle, and
/// whether a real caret existed.
struct AnchorInputs {
    anchor: icedtea_contract::Rectangle,
    output: Option<icedtea_contract::Rectangle>,
    echo: icedtea_contract::Rectangle,
    had_caret: bool,
}

impl State {
    /// Geometry of the lowest-index output — the single "which screen"
    /// fallback shared by the `DbCommand::OutputSize` arm and popup placement
    /// when focus names no window. `None` before any output exists.
    fn lowest_index_output_geometry(&self) -> Option<icedtea_contract::Rectangle> {
        self.outputs
            .keys()
            .min()
            .copied()
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.geometry)
    }

    /// The node of an arbitrary currently-placed IME candidate popup, for the
    /// test-only position/node oracles. At most one popup is placed in
    /// practice; if several ever coexist this picks an unspecified one
    /// (kept in one helper so the two oracle arms cannot disagree on what
    /// "first" means — and if multi-popup ever matters, sort here).
    fn first_input_popup_node(&self) -> Option<wlr::NodeId> {
        self.input_popup_nodes.values().next().copied()
    }

    /// The placement inputs IME UI positions against, read fresh on every
    /// call: the caret anchor translated into output coordinates, the
    /// output to clamp against, and the surface-local caret rectangle
    /// echoed back to the IME. Shared by the candidate popup arms
    /// (`new_popup_surface`, `popup_repositioned`) and the preedit overlay
    /// so the three cannot disagree on what "under the caret" means.
    ///
    /// The cursor rectangle the client commits is in its surface's local
    /// space (text-input-v3), so the anchor is translated into output
    /// coordinates through the focused window's content origin — the same
    /// translation xdg popups get via `root_surface_origin` — and clamped
    /// against the focused window's own output. Either lookup can miss
    /// (focus not on a window at all): then there is no caret to sit under
    /// and no screen to prefer, so fall back to the lowest-index output.
    ///
    /// The echo stays in the text input's surface-local space
    /// (input-method-v2: `text_input_rectangle` is "a rectangle in surface
    /// local coordinates") — NOT the translated output-space anchor, which
    /// is only for scene placement (A6.2 review lesson).
    ///
    /// `had_caret` is `false` exactly when no caret was found (no runtime,
    /// no focused input, or it went stale): the popup arm logs the popup id
    /// for that case, the overlay arm ignores it.
    fn anchor_inputs(&self) -> AnchorInputs {
        let (origin, frame) = match self.window_manager.focused_window() {
            Some(w) => {
                let ssd = crate::decoration::has_ssd(
                    &w.app_id,
                    w.client_decorations_requested,
                    w.fullscreen,
                );
                let content = crate::decoration::content_rect(w.geometry, ssd);
                ((content.x, content.y), Some(w.geometry))
            }
            None => ((0, 0), None),
        };
        let output = frame
            .and_then(|g| self.output_for_window(g))
            .and_then(|idx| self.outputs.get(&idx))
            .map(|o| o.geometry)
            .or_else(|| self.lowest_index_output_geometry());

        // Anchor against the focused text input's last-committed cursor
        // rectangle. `None` (no runtime, no focused input, or it went stale)
        // → a zero anchor, which places the popup at the output origin — the
        // best we can do without a caret to sit under.
        let caret = self
            .wayland
            .runtime()
            .and_then(|rt| rt.focused_text_input_cursor_rectangle());
        let had_caret = caret.is_some();
        let anchor = match &caret {
            Some(b) => icedtea_contract::Rectangle {
                x: origin.0.saturating_add(b.x),
                y: origin.1.saturating_add(b.y),
                width: b.width,
                height: b.height,
            },
            None => icedtea_contract::Rectangle {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
        };
        let echo = caret
            .map(|b| icedtea_contract::Rectangle {
                x: b.x,
                y: b.y,
                width: b.width,
                height: b.height,
            })
            .unwrap_or(icedtea_contract::Rectangle {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            });
        AnchorInputs {
            anchor,
            output,
            echo,
            had_caret,
        }
    }

    /// The placement inputs an IME candidate popup is positioned against:
    /// [`Self::anchor_inputs`], plus the no-caret debug log.
    ///
    /// `popup` only names the placement in that log: a zero rect echoed to
    /// the IME is otherwise indistinguishable from a real top-left caret
    /// when debugging misplaced popups.
    fn popup_anchor(&self, popup: wlr::InputPopupSurfaceId) -> AnchorInputs {
        let a = self.anchor_inputs();
        if !a.had_caret {
            tracing::debug!(
                ?popup,
                "placing input popup with no focused caret; using origin anchor"
            );
        }
        a
    }

    /// Run `f` — a seat keyboard-focus change — against the runtime, hiding
    /// the preedit overlay when the IME deactivates across the call.
    ///
    /// Client-driven deactivates arrive via
    /// `SeatHandler::input_method_deactivated`; compositor-driven
    /// keyboard-focus changes fire no such hook (the crate has no focus
    /// dispatcher), yet they still deactivate the IME inside wlroots. Every
    /// focus push in this file routes through here so a mid-compose focus
    /// change can never leave a stale overlay behind. Returns `f`'s value
    /// (`None` with no runtime attached, in which case `f` never runs).
    fn change_keyboard_focus<R>(&mut self, f: impl FnOnce(&wlr::Runtime) -> R) -> Option<R> {
        let before = self.wayland.runtime().map(|rt| rt.input_method_active());
        let rt = self.wayland.runtime()?;
        let out = f(rt);
        let after = self.wayland.runtime().map(|rt| rt.input_method_active());
        if before == Some(true) && after != Some(true) {
            self.hide_preedit_overlay();
        }
        Some(out)
    }

    /// Hide the preedit overlay, if shown: destroy its scene node and clear
    /// the record. Idempotent — every hide path calls this blindly, and a
    /// node already gone (or never created, with no runtime) clears cleanly
    /// rather than double-destroying.
    fn hide_preedit_overlay(&mut self) {
        let Some(overlay) = self.preedit_overlay.take() else {
            return;
        };
        if let Some(rt) = self.wayland.runtime() {
            rt.remove_buffer(overlay.node);
        }
    }

    /// Refresh the preedit overlay from the IME's committed generation:
    /// show or update while it carries composing text, hide otherwise.
    /// Called from `SeatHandler::input_method_committed`, which fires at the
    /// end of every IME commit.
    fn refresh_preedit_overlay(&mut self) {
        let Some(rt) = self.wayland.runtime().cloned() else {
            // No scene: nothing can be shown; drop any record rather than
            // report a node that was never created.
            self.preedit_overlay = None;
            return;
        };
        let Some(committed) = rt.committed_ime_state() else {
            self.hide_preedit_overlay();
            return;
        };
        if !crate::ime_overlay::should_show(&committed) {
            self.hide_preedit_overlay();
            return;
        }
        let Some(preedit) = committed.preedit else {
            self.hide_preedit_overlay();
            return;
        };
        self.show_preedit_overlay(&rt, &preedit.text, preedit.cursor_end);
    }

    /// Show (or update, in place) the overlay for `text` with the IME cursor
    /// at `cursor_end`. The anchor and clamp target come from
    /// [`Self::anchor_inputs`] — the same translated caret the candidate
    /// popup uses; the popup's surface-local echo is discarded here.
    ///
    /// Degrades to hidden wherever the scene refuses: no node kept without
    /// pixels (`rasterize_title` shapes to nothing), no record kept without
    /// a placed node.
    fn show_preedit_overlay(&mut self, rt: &wlr::Runtime, text: &str, cursor_end: i32) {
        let a = self.anchor_inputs();
        let anchor = a.anchor;
        let output = a.output;
        self.fonts.get_or_init(|| {
            (
                cosmic_text::FontSystem::new(),
                cosmic_text::SwashCache::new(),
            )
        });
        // Same panic-free spelling as `ensure_title_raster`'s: `get_mut`
        // cannot miss right after `get_or_init`.
        let Some((fonts, swash)) = self.fonts.get_mut() else {
            self.hide_preedit_overlay();
            return;
        };
        let measured = crate::ime_overlay::measure_preedit(fonts, text, cursor_end);
        let at = crate::ime_overlay::layout_overlay(anchor, &measured, output);
        let Some(px) = Self::rasterize_preedit(fonts, swash, text, &at) else {
            self.hide_preedit_overlay();
            return;
        };
        match self.preedit_overlay.take() {
            Some(overlay) => {
                // In place, like the title path: keeps the node's place in
                // the stacking order instead of re-adding it on top.
                if rt
                    .update_buffer(overlay.node, at.width, at.height, &px)
                    .is_none()
                {
                    rt.remove_buffer(overlay.node);
                    self.preedit_overlay = Self::add_overlay_node(rt, &at, &px);
                } else if rt.set_buffer_position(overlay.node, at.x, at.y).is_some() {
                    self.preedit_overlay = Some(crate::ime_overlay::PreeditOverlay {
                        node: overlay.node,
                        position: (at.x, at.y),
                    });
                } else {
                    rt.remove_buffer(overlay.node);
                }
            }
            None => {
                self.preedit_overlay = Self::add_overlay_node(rt, &at, &px);
            }
        }
    }

    /// Rasterize the overlay's pixels for a placed `at`: dark translucent
    /// box, white text, white caret bar at the cursor. `None` when the text
    /// shapes to nothing (whitespace-only preedit, or no font on this
    /// machine with the glyphs) — the caller hides instead of showing an
    /// empty box, the same degradation a title that didn't shape gets.
    fn rasterize_preedit(
        fonts: &mut cosmic_text::FontSystem,
        swash: &mut cosmic_text::SwashCache,
        text: &str,
        at: &crate::ime_overlay::OverlayLayout,
    ) -> Option<Vec<u8>> {
        let mut px = crate::text::rasterize_title(
            fonts,
            swash,
            text,
            at.width,
            at.height,
            crate::ime_overlay::OVERLAY_PAD_X,
            [255, 255, 255, 255],
        )?;
        // Dark translucent backdrop wherever no glyph pixel landed, so the
        // text reads over any app content. Premultiplied: each channel
        // already scaled by the alpha, which is what the scene graph
        // composites.
        const BG: [u8; 4] = [17, 17, 17, 220];
        for p in px.chunks_exact_mut(4) {
            if p[3] == 0 {
                p.copy_from_slice(&BG);
            }
        }
        // Caret bar at the cursor: two pixels wide, inset from the band's
        // top and bottom, clamped into the buffer so a measure/raster
        // disagreement can never panic.
        let caret_x = (crate::ime_overlay::OVERLAY_PAD_X + at.cursor_x)
            .max(0)
            .min(at.width.saturating_sub(1));
        debug_assert!(caret_x >= 0 && caret_x < at.width);
        for y in 4..at.height.saturating_sub(4).max(5) {
            for dx in 0..2 {
                let x = caret_x.saturating_add(dx).min(at.width.saturating_sub(1));
                debug_assert!(x >= 0 && x < at.width && y >= 0 && y < at.height);
                let i = ((y * at.width + x) * 4) as usize;
                debug_assert!(i + 4 <= px.len());
                px[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        Some(px)
    }

    /// Create the overlay's scene node in the top band and place it. `None`
    /// (leaving nothing recorded) wherever the scene refuses.
    fn add_overlay_node(
        rt: &wlr::Runtime,
        at: &crate::ime_overlay::OverlayLayout,
        px: &[u8],
    ) -> Option<crate::ime_overlay::PreeditOverlay> {
        let node = rt.add_buffer_in_band(wlr::Band::Top, at.width, at.height, px)?;
        if rt.set_buffer_position(node, at.x, at.y).is_none() {
            rt.remove_buffer(node);
            return None;
        }
        Some(crate::ime_overlay::PreeditOverlay {
            node,
            position: (at.x, at.y),
        })
    }
}

impl wlr::SeatHandler for State {
    fn key(&mut self, event: &wlr::KeyEvent<'_>) -> bool {
        let m = event.modifiers();
        let mods = to_model_modifiers(m.logo(), m.ctrl(), m.alt(), m.shift());

        // Alt-tab's chosen end condition: the session ends when the
        // configured `cycle:alt_tab` binding's modifier is no longer down.
        // See `alt_tab_should_end`'s doc for why this event's own `keysym`,
        // not just its (stale, on a modifier's own release) `mods`, has to
        // be consulted.
        if self.alt_tab.is_active() {
            let watched = self.watched_alt_tab_modifiers();
            if alt_tab_should_end(watched, mods, event.pressed(), event.keysym()) {
                // Emits its own event internally; nothing to flush here.
                self.end_alt_tab();
            }
        }

        // Releases are never consumed: a client that is sent a press but not
        // its release believes the key is still held forever.
        if !event.pressed() {
            return false;
        }

        // M8: while a shortcuts inhibitor is active the compositor's own
        // bindings are skipped entirely -- every key goes to the focused
        // client, which is what inhibition promises. Polled, not event-gated
        // (see `shortcuts_inhibited`'s doc): the `shortcuts_inhibitor_toggled`
        // arm only ever sees lifecycle birth/destroy. Returning `false`, not
        // `true`: the key is not consumed, it is rerouted -- swallowing it
        // here would starve the very client that asked for it.
        if self.shortcuts_inhibited() {
            return false;
        }

        let consumed = self.handle_key(mods, event.keysym()).is_some();
        // `handle_key` -> `apply_action` mutates the model, and every
        // mutation owes exactly one event; flush here rather than at the
        // next unrelated sync.
        self.emit_pending();
        consumed
    }

    fn pointer_motion(&mut self, x: f64, y: f64, _time_msec: u32) {
        // The model works in integer output-logical coordinates; the library
        // reports scene coordinates, which for the slice's single output at
        // the layout origin are the same space. `as i32` truncates toward
        // zero, which is what a pixel index wants.
        let pointer = (x as i32, y as i32);
        self.pointer_location = pointer;
        // M7 shell mirror: every motion is an observed cursor position (see
        // `note_cursor_at` for the visibility half). Positions arrive
        // post-constraint from the crate, so this mirror can never run
        // ahead of the real cursor.
        self.note_cursor_at(pointer);

        // M7 constraint gate (lock): a locked pointer's motion must not
        // drive focus, hover, drag or resize (see
        // `pointer_locked_for_focus`). The crate normally suppresses the
        // event before it could arrive, making this defense-in-depth; the
        // location mirror above still records it, since the crate cursor
        // genuinely is where the event says.
        if self.pointer_locked_for_focus() {
            self.emit_pending();
            return;
        }

        // Finding F12: the hit test is computed exactly once per motion
        // event and handed to `update_ssd_hover` inside
        // `handle_pointer_motion`, which used to run its own second,
        // allocating `window_at_point` on every single pointer motion.
        let over = self.window_at_point(pointer);
        self.handle_pointer_motion_with_hit(pointer, over);
        self.emit_pending();
    }

    fn pointer_button(&mut self, x: f64, y: f64, button: u32, pressed: bool, _time_msec: u32) {
        // BTN_LEFT only: every decoration interaction this compositor has is
        // a left-click one, and forwarding the rest to the client unchanged
        // is the correct behaviour for them. Hoisted above `pointer_pressed`
        // below (review: state.rs:2050-2056) -- both need it before the
        // early return this gate gives everything past it.
        const BTN_LEFT: u32 = 0x110;

        // Set before any routing (Deviation 8): `begin_client_move`/
        // `begin_client_resize` gate an interactive move/resize on this, and
        // it must already reflect *this* event by the time anything below
        // reads it -- including a request that arrives interleaved with the
        // button event itself. Gated on `BTN_LEFT` -- the same button the
        // grab/release path below acts on exclusively -- because this is a
        // single scalar, not a per-button set: an ungated assignment means a
        // right/middle release while the left button is still held wrongly
        // clears it (spuriously rejecting a legit move grab), and a lone
        // right-click press wrongly sets it. "Left button held" is the only
        // reading that matches the grab semantics throughout this file.
        if button == BTN_LEFT {
            self.pointer_pressed = pressed;
        }

        // Recorded before the `BTN_LEFT` gate below, not after: this is the
        // same `pointer_location` `pointer_motion` updates, and a
        // button-only event (which carries no position of its own once
        // `handle_pointer` reads it back) must not skip the update just
        // because the button that arrived happens to not be the left one --
        // a right-click at a new position must still leave
        // `pointer_location` correct for whatever left-click/drag comes
        // next.
        let pointer = (x as i32, y as i32);
        self.pointer_location = pointer;
        // M7 shell mirror, same as `pointer_motion`'s: a button event
        // observes the cursor too.
        self.note_cursor_at(pointer);

        if button != BTN_LEFT {
            return;
        }

        if pressed {
            // M7 constraint gate (lock): a locked pointer's clicks belong
            // to the focused client -- they still reach it through the
            // crate -- and must not start model drags or refocus windows.
            if !self.pointer_locked_for_focus() {
                // The model answers "what is under the pointer", not the scene:
                // the scene knows nothing about workspaces or minimization, and
                // `window_at_point` is already MRU-ordered, which is a correct
                // topmost-first order (review finding I1).
                let Some(id) = self.window_at_point(pointer) else {
                    return;
                };
                self.handle_pointer(PointerEvent::Press { id, pointer });
            }
        } else {
            self.handle_pointer(PointerEvent::Release { pointer });
        }
        self.emit_pending();
    }

    /// M7: a touch point went down.
    ///
    /// Notification-only, by crate design: the down already reached the
    /// touch client through the token path before this runs, and the event
    /// is id-only, so no position arrives and the model cannot focus the
    /// touched window here. All that remains for the consumer is marking
    /// the in-flight touch the snapshot (`touch_active`) and the panel
    /// report. (The test-only `InjectTouchDown` arm knows its coordinates
    /// but likewise only refreshes the flag: focusing there would give
    /// test input model side-effects production hardware can never
    /// produce. A position-carrying touch notification is a possible future
    /// wlr gap -- deliberately not worked around here.)
    fn touch_down(&mut self, _id: wlr::TouchId) {
        self.touch_active = true;
    }

    /// M7: a touch point went up. Re-derived rather than cleared: other
    /// points may still be down (multi-touch), and only the seat knows.
    /// See `refresh_touch_active` for the no-runtime reading.
    fn touch_up(&mut self, _id: wlr::TouchId) {
        self.refresh_touch_active();
    }

    /// M7: the touch sequence was cancelled wholesale. No id arrives (the
    /// cancel names no single point) and no points survive it, so this
    /// clears unconditionally rather than re-deriving.
    fn touch_cancelled(&mut self) {
        self.touch_active = false;
    }

    /// M7: a pointer gesture began. Notification-only: the full-fidelity
    /// forward (kind, deltas, finger count) already reached gesture clients
    /// through the crate's token path, so the consumer only forwards the
    /// phase to the shell feed. The id is deliberately unread -- an unknown
    /// (deferred, device-gone) id is harmless by construction.
    fn gesture_began(&mut self, _id: wlr::GestureId) {
        self.emit(Event::GestureBegan);
        self.emit_pending();
    }

    /// M7: the in-flight gesture ended (completed or cancelled -- both end
    /// the gesture as far as this notification goes). Same terms as
    /// `gesture_began`.
    fn gesture_ended(&mut self, _id: wlr::GestureId) {
        self.emit(Event::GestureEnded);
        self.emit_pending();
    }

    /// M7: a switch toggled.
    ///
    /// The hardware signal names no device and no type, so the reading
    /// comes from the runtime aggregate, which the crate recorded *before*
    /// emitting (record-then-emit). Reading the aggregate rather than the
    /// event's own `on` keeps the emitted signal self-consistent with what
    /// `switch_state()` reports right now -- at the cost that a deferred
    /// delivery of an older toggle reports the latest transition instead;
    /// rapid double-toggles converge on the true final state either way.
    /// Without a runtime -- impossible in production (a toggle implies a
    /// live seat), reachable only in unit tests -- there is nothing
    /// truthful to report, so this stays silent rather than guess a type.
    fn switch_toggled(&mut self, _id: wlr::SwitchId, _on: bool) {
        let Some(rt) = self.wayland.runtime() else {
            tracing::debug!("switch toggled with no runtime attached; ignoring");
            return;
        };
        let Some(state) = rt.switch_state() else {
            tracing::debug!("switch toggled with no switch state recorded; ignoring");
            return;
        };
        self.apply_switch_toggle(state.switch_type, state.on);
    }

    /// Track `wlr::Runtime::is_session_locked` locally so this model stops
    /// fighting the crate's own focus refusal while locked (see
    /// `session_locked`'s doc on the struct field). On `locked = true`,
    /// normal layout/focus reconciliation is suspended from here on --
    /// `sync_window_to_scene`/`sync_seat_focus` both early-return. On
    /// `locked = false` (a genuine unlock; the crate never calls this with
    /// `false` after a locker dies without unlocking -- that is the
    /// stay-locked security invariant), clear the flag and push the
    /// model's current focus (the MRU/last toplevel; nothing here can have
    /// changed it while locked, since the crate refused it) back out to
    /// the seat and scene in one go.
    fn session_lock_changed(&mut self, locked: bool) {
        self.session_locked = locked;
        // Findings F5/F15, still needed after `wlr` 0.20.26: the crate drops
        // a named shape on a pointer-*focus* change, and a lock engaging (or
        // releasing) under a stationary pointer is not one -- no pointer
        // event happens, so no `focus_change` fires. Without this a client's
        // `Text` cursor rides onto the lock screen, and the lock surface's
        // own I-beam rides back onto the desktop.
        self.apply_cursor_shape(wlr::CursorShape::Default);
        if !locked {
            match self.focused_id() {
                Some(id) => self.sync_window_to_scene(id),
                None => self.sync_seat_focus(),
            }
        }
    }

    /// A client asked, via `cursor-shape-v1`, to name the seat cursor.
    /// wlroots does not apply this itself (see the crate's own
    /// `request_set_shape` doc), so this handler is what makes the request
    /// do anything.
    ///
    /// `TabletTool` requests are ignored (logged at debug): a stray
    /// background tablet-tool client should not repaint the shared cursor
    /// image ahead of whatever the pointer is doing.
    ///
    /// `Pointer` requests are honored unconditionally. Since `wlr` 0.20.26
    /// the crate delivers this callback *only* for the seat client that
    /// currently holds pointer focus, so the surfaceless background daemon
    /// this compositor used to guess at with a model hit test cannot reach
    /// here at all. That includes the locked case: while locked the lock
    /// surface is what holds pointer focus, so its own request (an I-beam in
    /// its password field) arrives and is honored, and nothing else's is.
    ///
    /// **The SSD title bar is the non-obvious consequence of that gate.**
    /// This compositor's server-side decoration is a scene *rect* node
    /// (`render::draw_frame`), not a `wl_surface` — the client's surface
    /// begins `TITLE_BAR_HEIGHT` lower, at `decoration::content_rect`. So
    /// while the pointer sits anywhere on the band, wlroots' pointer focus
    /// is NULL rather than the window's client: the client is sent
    /// `wl_pointer.leave` on the way in, the crate's own
    /// `on_pointer_focus_change` resets the named shape to the default, and
    /// any `set_shape` that client makes from then on is dropped by the
    /// crate before it reaches this handler. That is the correct outcome and
    /// not a gap to work around — a window's client has no business naming
    /// the cursor for a strip *the compositor* draws and owns the hit
    /// testing for (`decoration::hit_test`) — but it does mean "my
    /// `set_shape` did nothing" is expected whenever the pointer is over a
    /// title bar, and it is why the cursor over the band is whatever this
    /// compositor last applied rather than whatever the window asked for.
    ///
    /// Nothing else this compositor puts on screen behaves that way: xdg
    /// popups and layer surfaces are real `wl_surface`s, take pointer focus
    /// normally, and so route their own `set_shape` here like any toplevel.
    /// See `State::popup_at_point` and the popup handlers below for the model
    /// side of that.
    /// `the_ssd_title_bar_drops_a_cursor_shape_request` (in
    /// `compositor/tests/compat_protocols.rs`) pins all three readings.
    fn request_set_shape(
        &mut self,
        device: wlr::CursorShapeDevice,
        serial: u32,
        shape: wlr::CursorShape,
    ) {
        if !Self::honors_cursor_shape_device(device) {
            tracing::debug!(
                ?device,
                serial,
                ?shape,
                "ignoring cursor-shape request from a non-pointer device"
            );
            return;
        }
        self.apply_cursor_shape(shape);
    }

    /// A client asked, via `xdg-activation-v1`, that a surface be focused.
    ///
    /// The decision itself is `activation_may_steal_focus` (see its doc for
    /// why those are the two conditions); this only maps ids and applies the
    /// outcome:
    ///
    /// - honored -> the ordinary explicit-focus path, the same one
    ///   `DbCommand::Focus`, click-to-focus and `xwayland_request_activate`
    ///   take: `WindowManager::focus` (which emits), `release_layer_focus`
    ///   after it succeeds, then `sync_focus_change`. The raise comes free
    ///   with that -- `sync_window_to_scene` raises the focused window when
    ///   `behavior.raise_on_focus` is set -- so there is no new focus or
    ///   stacking machinery here.
    /// - refused -> `set_attention(target, true)`, a shell-facing hint the
    ///   next focus clears. A target that is *already* focused gets neither:
    ///   there is nothing to raise and nothing to flag.
    ///
    /// The honored branch also requires an *activatable* target -- mapped,
    /// not minimized, on the active workspace -- because this compositor
    /// will not switch workspaces or unminimize on a client's say-so.
    /// A request aimed anywhere else is refused (and so flagged), rather
    /// than honored into a no-op.
    fn request_activate(&mut self, target: Option<wlr::ToplevelId>, token: wlr::ActivationToken) {
        // Finding F1 (security): while the session is locked this handler
        // does nothing at all -- neither branch. Honoring one would let a
        // background client move the model's focus behind the lock screen
        // and so choose who receives the keyboard the instant it is
        // released; even the refused branch would raise an attention hint
        // the user cannot see, act on, or clear until then. The same
        // early-return `sync_window_to_scene`/`sync_seat_focus` take.
        if self.session_locked {
            tracing::debug!(
                ?target,
                ?token,
                "ignoring activation request while the session is locked"
            );
            return;
        }
        let Some(target) = target else {
            tracing::debug!(
                ?token,
                "ignoring activation request for a surface with no toplevel"
            );
            return;
        };
        let Some(target_window) = self
            .wayland
            .window_for(crate::wayland::ToplevelKey::new(target))
        else {
            tracing::debug!(
                ?target,
                ?token,
                "ignoring activation request for an untracked toplevel"
            );
            return;
        };
        let focused = self.focused_id();
        if focused == Some(target_window) {
            return;
        }
        let requester = token.requesting_toplevel.and_then(|id| {
            self.wayland
                .window_for(crate::wayland::ToplevelKey::new(id))
        });
        // Owner ruling: a client's activation request never switches
        // workspaces and never unminimizes. Honoring one aimed at a target
        // the focus could not actually land on where the user is looking
        // would be a silent no-op; those fall through to the attention
        // branch, which is exactly the signal the shell wants for them.
        let active_workspace = self.window_manager.active_workspace();
        let target_activatable = self
            .window_manager
            .get(target_window)
            .is_some_and(|w| w.mapped && !w.minimized && w.workspace == active_workspace);
        if activation_may_steal_focus(token.has_seat, target_activatable, requester, focused) {
            tracing::debug!(
                ?target_window,
                ?requester,
                "honoring xdg-activation focus request"
            );
            if self.window_manager.focus(target_window).is_some() {
                // Same ordering as every other explicit focus assertion: only
                // after `focus` actually succeeded, so a refused focus leaves
                // a layer surface's keyboard grab alone.
                self.release_layer_focus();
                self.sync_focus_change(focused);
            }
        } else {
            tracing::debug!(
                ?target_window,
                ?requester,
                has_seat = token.has_seat,
                target_activatable,
                "refusing xdg-activation focus steal; flagging attention instead"
            );
            self.raise_attention_if_answerable(target_window);
        }
        self.emit_pending();
    }

    fn new_popup_surface(&mut self, popup: wlr::InputPopupSurfaceId) {
        // Placement needs the runtime (for the scene node + the anchor rect);
        // no runtime means no scene, so there is nothing to place.
        let Some(rt) = self.wayland.runtime() else {
            return;
        };
        // Create the popup's scene node in the top band. `None` if the popup is
        // already gone or has no surface yet — cases where placing nothing is
        // correct — or while a scene walk is live. The walk refusal is
        // believed unreachable from single-threaded event dispatch (this
        // signal fires from client request handling, never inside a
        // compositor scene walk), so there is no retry: a refused popup is
        // dropped, and logged.
        let Some(node) = rt.add_input_popup_in_band(popup, wlr::Band::Top) else {
            tracing::debug!(?popup, "input popup placement refused; dropping");
            return;
        };

        // Anchor + output + surface-local echo, shared with the reposition arm
        // (see `State::popup_anchor`).
        let a = self.popup_anchor(popup);
        let anchor = a.anchor;
        let output = a.output;
        let echo = a.echo;

        // The popup's own committed size is not known at creation (the IME has
        // not necessarily attached a buffer yet — `input_popup_size` reads
        // (0, 0) before the first commit). Place against a zero size: the
        // below-the-cursor anchor and origin clamp still apply; right/bottom
        // clamping only matters once a real size is known, which arrives with
        // the reposition event (`popup_repositioned`).
        let popup_size = icedtea_contract::Rectangle {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };
        let (x, y) = crate::input_method::place_below_clamped(anchor, popup_size, output);

        if rt.set_node_position(node, x, y).is_none() {
            // The node went stale between creation two lines above and this
            // call (a teardown racing placement on the same thread): its
            // position was never applied, so neither record it nor tell the
            // IME it was placed against the anchor.
            tracing::debug!(
                ?popup,
                "popup scene node went stale before positioning; dropping placement"
            );
            return;
        }
        // Record the placed node so the test-only InputPopupPosition oracle can
        // read its scene position back (the crate has no by-id accessor).
        self.input_popup_nodes.insert(popup, node);
        // Tell the popup which rectangle it was anchored against
        // (surface-local — see `State::popup_anchor` for why the echo is not
        // the translated anchor).
        let anchor_box = wlr::Box2D::new(echo.x, echo.y, echo.width, echo.height);
        rt.send_input_popup_rectangle(popup, anchor_box);
    }

    fn popup_repositioned(&mut self, popup: wlr::InputPopupSurfaceId) {
        // Re-placement needs the runtime (for the node move + the fresh
        // anchor); no runtime means no scene, so there is nothing to move.
        // Known limitation: the crate only emits this when a popup is
        // tracked, so a caret commit with no popup re-anchors nothing;
        // the overlay then stays stale until the next IME commit. A future
        // wlr version should emit ungated when the IME is active.
        let Some(rt) = self.wayland.runtime() else {
            return;
        };
        // Same translated-anchor computation as creation: the caret moved
        // under the live popup, so re-read the anchor (plus the output and
        // the surface-local echo) fresh rather than reusing the
        // creation-time position.
        let a = self.popup_anchor(popup);
        let anchor = a.anchor;
        let output = a.output;
        let echo = a.echo;
        // The popup's own extent, now that it may have committed a buffer.
        // Unknown id, destroyed popup, or no surface yet → (0, 0): the same
        // zero-size placement creation uses, harmless for a popup whose node
        // lookup below then misses anyway.
        let size = rt.input_popup_size(popup).unwrap_or((0, 0));
        let popup_rect = icedtea_contract::Rectangle {
            x: 0,
            y: 0,
            width: size.0,
            height: size.1,
        };
        let (x, y) = crate::input_method::place_below_clamped(anchor, popup_rect, output);
        // The record is untouched: the node was tracked at creation, and a
        // popup destroyed while this event was queued already cleared it via
        // `popup_surface_destroyed` — an unknown id is a silent no-op.
        let Some(node) = self.input_popup_nodes.get(&popup).copied() else {
            tracing::debug!(
                ?popup,
                "input popup reposition for untracked popup; dropping"
            );
            return;
        };
        if rt.set_node_position(node, x, y).is_none() {
            tracing::debug!(
                ?popup,
                "popup scene node went stale before repositioning; dropping"
            );
            return;
        }
        // Re-tell the popup its (possibly moved) rectangle, surface-local —
        // never the translated anchor (see `State::popup_anchor`).
        let anchor_box = wlr::Box2D::new(echo.x, echo.y, echo.width, echo.height);
        rt.send_input_popup_rectangle(popup, anchor_box);

        // Keep the preedit overlay glued to the caret: a caret commit
        // without an IME commit moves the popup via this hook but not the
        // overlay via `input_method_committed`, so refresh it here while
        // it is shown.
        if self.preedit_overlay.is_some() {
            self.refresh_preedit_overlay();
        }
    }

    fn popup_surface_destroyed(&mut self, popup: wlr::InputPopupSurfaceId) {
        // Drop the compositor's record of the placed node. The crate itself
        // tears down the popup's scene node on destroy (FIX-A10-NODE); this only
        // clears our by-id lookup so the InputPopupPosition oracle stops
        // reporting a gone popup.
        self.input_popup_nodes.remove(&popup);
    }

    /// The bound IME committed: the relay to the focused text input already
    /// went out, so this only refreshes the preedit overlay from the fresh
    /// committed generation — show/update while it carries composing text,
    /// hide on commit-string, delete, or preedit-clear.
    fn input_method_committed(&mut self) {
        self.refresh_preedit_overlay();
    }

    /// The bound IME deactivated (client-driven: text-input disable or
    /// destroy, IME destroy, session lock/unlock — never a keyboard-focus
    /// change, whose hide path is `change_keyboard_focus`): whatever was
    /// composing is gone, so the overlay goes with it, unconditionally.
    fn input_method_deactivated(&mut self) {
        self.hide_preedit_overlay();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icedtea_config::default_config;
    use icedtea_contract::Rectangle;

    const DEFAULT_GEO: Rectangle = Rectangle {
        x: 0,
        y: 0,
        width: 640,
        height: 400,
    };

    #[test]
    fn state_emits_pending_events_on_channel() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.emit_pending();
        assert!(matches!(
            rx.try_recv().map(|e| e.event),
            Ok(Event::WindowOpened(_))
        ));
    }

    #[test]
    fn decoration_action_for_returns_close_on_button_click() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state
            .window_manager
            .set_geometry(
                id,
                Rectangle {
                    x: 100,
                    y: 100,
                    width: 600,
                    height: 400,
                },
            )
            .unwrap();
        state.window_manager.focus(id).unwrap();
        let geo = Rectangle {
            x: 100,
            y: 100,
            width: 600,
            height: 400,
        };
        // Click on rightmost button (close button)
        let close_pt = (geo.x + geo.width - 5, geo.y + 5);
        assert_eq!(
            state.decoration_action_for(id, close_pt),
            Some(crate::decoration::DecorationAction::Close)
        );
    }

    #[test]
    fn decoration_action_for_returns_none_for_unfocused_window() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id1 = state
            .window_manager
            .add_window("app1", "t1", 1, DEFAULT_GEO);
        let _id2 = state
            .window_manager
            .add_window("app2", "t2", 2, DEFAULT_GEO);
        // _id2 is now focused; id1 is unfocused
        state
            .window_manager
            .set_geometry(
                id1,
                Rectangle {
                    x: 100,
                    y: 100,
                    width: 600,
                    height: 400,
                },
            )
            .unwrap();
        let geo = Rectangle {
            x: 100,
            y: 100,
            width: 600,
            height: 400,
        };
        let close_pt = (geo.x + geo.width - 5, geo.y + 5);
        // Should return None because id1 is not focused
        assert_eq!(state.decoration_action_for(id1, close_pt), None);
    }

    #[test]
    fn decoration_action_for_returns_none_for_fullscreen_window() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }),
        );
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.window_manager.focus(id).unwrap();
        state.toggle_fullscreen(id).unwrap();
        let geo = Rectangle {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        };
        let close_pt = (geo.x + 5, geo.y + 5);
        // Should return None because window is fullscreen
        assert_eq!(state.decoration_action_for(id, close_pt), None);
    }

    #[test]
    fn decoration_action_for_ignores_csd_and_still_returns_close() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state
            .window_manager
            .add_window("org.gtk.App", "t", 1, DEFAULT_GEO);
        state
            .window_manager
            .set_geometry(
                id,
                Rectangle {
                    x: 100,
                    y: 100,
                    width: 600,
                    height: 400,
                },
            )
            .unwrap();
        state.window_manager.focus(id).unwrap();
        state
            .window_manager
            .set_client_decorations_requested(id, Some(true))
            .unwrap();
        let geo = Rectangle {
            x: 100,
            y: 100,
            width: 600,
            height: 400,
        };
        let close_pt = (geo.x + geo.width - 5, geo.y + 5);
        // Should work even though client requested decorations (decoration_action_for doesn't filter by CSD)
        // CSD filtering happens at rendering time in draw_frame
        assert_eq!(
            state.decoration_action_for(id, close_pt),
            Some(crate::decoration::DecorationAction::Close)
        );
    }

    #[test]
    fn toggle_fullscreen_flips_state_and_geometry() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }),
        );
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.toggle_fullscreen(id).unwrap();
        let w = state.window_manager.get(id).unwrap();
        assert!(w.fullscreen);
        assert_eq!(
            w.geometry,
            Rectangle {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080
            }
        );
    }

    // NOTE (brief deviation): the brief's Step-1 sample for
    // `apply_action_switches_workspaces_and_snaps` calls
    // `state.window_manager.add_window("app", "t", 1)` (3 args), never
    // registers an output, and adds the window *before* switching
    // workspaces. Three real bugs, not just transcription noise:
    //   1. 3-arg `add_window` no longer compiles against the new 5-arg
    //      signature (this task's own change).
    //   2. `snap()` reads `self.outputs` (empty by default in `State::new`)
    //      to find output geometry, despite the brief's own prose saying
    //      "the tests assume ... a 1000x800 output" -- nothing in the
    //      sample ever inserts one.
    //   3. `WindowManager::focused_window()` is scoped to the *active*
    //      workspace (established well before this task -- see
    //      `window.rs`'s `move_to_workspace_keeps_focus_valid` test, which
    //      asserts exactly this). Adding the window on workspace 0 and then
    //      switching to workspace 2 (index 1) leaves workspace 1 with no
    //      focused window at all, so `apply_action("snap:left")` -- which
    //      dispatches through `focused_window()` -- would return `None`
    //      and the `.unwrap()` would panic. Switching workspace *first*,
    //      then adding the window (which auto-focuses on whatever is
    //      currently active), keeps the window's workspace and the active
    //      workspace in agreement, matching how a real "switch to an empty
    //      workspace, open something there, then snap it" sequence would
    //      actually behave.
    // All three are fixed below; the assertions are unchanged from the
    // brief.
    #[test]
    fn apply_action_switches_workspaces_and_snaps() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            }),
        );
        state.apply_action("workspace:2").unwrap();
        assert_eq!(state.window_manager.active_workspace(), 1);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.apply_action("snap:left").unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry.width,
            1000 / 2 - 16
        );
        state.apply_action("snap:restore").unwrap();
        assert_eq!(state.window_manager.get(id).unwrap().geometry.width, 640);
    }

    // These pin `parse_spawn_argv` as a direct argv builder, not a shell
    // invocation: it must never see `sh -c` semantics (no pipes, no `$VAR`
    // expansion) and must split quoted words correctly. A revert to
    // `Command::new("sh").arg("-c").arg(cmd)` -- or a naive
    // `cmd.split_whitespace()` that ignores quoting -- fails these.
    #[test]
    fn parse_spawn_argv_splits_program_and_args() {
        assert_eq!(
            parse_spawn_argv("firefox --new-window https://example.com"),
            Some((
                "firefox".to_string(),
                vec![
                    "--new-window".to_string(),
                    "https://example.com".to_string()
                ],
            ))
        );
    }

    #[test]
    fn parse_spawn_argv_handles_quoted_arguments() {
        // A double-quoted argument with an embedded space must survive as
        // one argv element, not two -- the exact case `sh -c` would have
        // handled via shell quoting, and a naive whitespace split would
        // shatter into `["My", "Documents"]`.
        assert_eq!(
            parse_spawn_argv(r#"nautilus "My Documents""#),
            Some(("nautilus".to_string(), vec!["My Documents".to_string()],))
        );
        // Single quotes too.
        assert_eq!(
            parse_spawn_argv("echo 'hello world'"),
            Some(("echo".to_string(), vec!["hello world".to_string()]))
        );
    }

    #[test]
    fn parse_spawn_argv_collapses_extra_whitespace() {
        assert_eq!(
            parse_spawn_argv("  ls   -la   /tmp  "),
            Some((
                "ls".to_string(),
                vec!["-la".to_string(), "/tmp".to_string()],
            ))
        );
    }

    #[test]
    fn parse_spawn_argv_rejects_empty_or_whitespace_only() {
        assert_eq!(parse_spawn_argv(""), None);
        assert_eq!(parse_spawn_argv("   "), None);
    }

    #[test]
    fn parse_spawn_argv_does_not_expand_shell_metacharacters() {
        // `$HOME`, `;`, `|`, `&&` etc. must come through as literal argv
        // text -- there is no shell here to interpret them. This is the
        // regression test for the `sh -c` removal: if `apply_action`
        // reverted to shelling out, this string would run two commands
        // instead of passing `rm;`, `-rf`, `/;`, `$HOME` as literal
        // (harmless, since no such program exists) arguments to a program
        // named `echo`.
        assert_eq!(
            parse_spawn_argv("echo $HOME; rm -rf /"),
            Some((
                "echo".to_string(),
                vec![
                    "$HOME;".to_string(),
                    "rm".to_string(),
                    "-rf".to_string(),
                    "/".to_string(),
                ],
            ))
        );
    }

    // Review finding (LOW): an all-quotes input like `""` used to set
    // `in_word = true` on the opening quote and push one empty-string word,
    // so `program` came back as `""` -- distinct from the intended "empty or
    // unparseable command" `None` path (which `apply_action`'s `"spawn"` arm
    // logs and ignores) -- and would instead reach
    // `Command::new("").args([]).spawn()`, which fails at spawn time with no
    // warning logged. An empty program must be treated the same as no words
    // at all.
    #[test]
    fn parse_spawn_argv_rejects_empty_program_from_all_quotes() {
        assert_eq!(parse_spawn_argv(r#""""#), None);
        assert_eq!(parse_spawn_argv("''"), None);
        assert_eq!(parse_spawn_argv(r#"""  """#), None);
    }

    // Review finding (HIGH, lex-review testing): every test above drives
    // `parse_spawn_argv` directly, so a revert of the `"spawn"` arm in
    // `apply_action` back to `Command::new("sh").arg("-c").arg(cmd)` (the
    // exact regression `parse_spawn_argv_does_not_expand_shell_metacharacters`
    // means to guard against) would leave all of them green -- none of them
    // ever calls `apply_action` at all. This test drives the real
    // `apply_action("spawn:...")` wiring end-to-end and asserts an
    // observable difference between a direct argv exec and a shell exec: the
    // spawned program (a tiny shell script acting purely as a probe -- it is
    // never invoked as a shell by `apply_action` itself) records its raw
    // `$@` to a marker file. A command string containing `$HOME` is
    // dispatched through `apply_action`; if `apply_action` execs directly
    // (the fix), the probe receives the two literal characters `$HOME` as
    // its argv, unexpanded. If `apply_action` reverted to `sh -c`, the outer
    // shell would expand `$HOME` to the real home directory *before* the
    // probe ever saw it, so the marker would contain a real path instead of
    // the literal string -- a safe (non-destructive), reliably observable
    // stand-in for the metacharacter-injection risk the fix closes.
    #[test]
    fn apply_action_spawn_execs_directly_without_shell_expansion() {
        let dir = tempfile::tempdir().unwrap();
        let probe_path = dir.path().join("probe.sh");
        let marker_path = dir.path().join("marker.txt");
        std::fs::write(
            &probe_path,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$1\".out\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&probe_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let action = format!(
            "spawn:{} {} $HOME",
            probe_path.display(),
            marker_path.display()
        );
        assert_eq!(state.apply_action(&action), Some(()));

        let out_path = format!("{}.out", marker_path.display());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut contents = None;
        while std::time::Instant::now() < deadline {
            if let Ok(s) = std::fs::read_to_string(&out_path) {
                contents = Some(s);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let contents = contents.expect("probe script must run and write its marker");
        let lines: Vec<&str> = contents.lines().collect();
        // argv[0] (the probe's own $1) is the marker path itself, echoed
        // back as the first word of "$@"; argv[1] must be the literal,
        // unexpanded `$HOME` -- a shell-exec revert would put a real
        // filesystem path (or empty string) there instead.
        assert_eq!(lines.last().copied(), Some("$HOME"));
    }

    #[test]
    fn apply_config_reconciles_workspace_names_without_dropping_windows() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // default_config has 4 workspaces; the window stays on workspace 0.
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["A".into(), "B".into()];
        let events = state.apply_config(cfg);

        // Workspace list reconciled in place to the new names/count.
        let info = state.window_manager.workspace_info();
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].name, "A");
        assert_eq!(info[1].name, "B");
        assert!(events.iter().any(|e| matches!(e, Event::WorkspaceList(_))));

        // The window on a still-existing workspace index survives with its
        // assignment intact -- and no WindowClosed was emitted for it.
        assert!(
            state.window_manager.get(a).is_some(),
            "window survives a reload"
        );
        assert_eq!(state.window_manager.get(a).unwrap().workspace, 0);
        assert!(!events.iter().any(|e| matches!(e, Event::WindowClosed(_))));
    }

    /// A reload that removes the workspace a window sits on migrates that
    /// window to workspace 0 rather than dropping it (the decided semantics
    /// of `WindowManager::set_workspace_names`). Workspace 0 is also the
    /// (untouched, default) active workspace here, so `apply_config`'s M-1
    /// belt-and-braces re-pick (mirroring review finding I6) legitimately
    /// picks the migrated window back up as `focused_window` -- it is now
    /// the sole candidate sitting on the active workspace, exactly the
    /// "migrated window must not be left with a dangling focus pointer"
    /// case the fix exists for.
    #[test]
    fn apply_config_migrates_windows_off_removed_workspaces_to_zero() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // default_config has 4 workspaces; park the window on workspace 2.
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.window_manager.set_workspace(a, 2).unwrap();
        assert_eq!(state.window_manager.get(a).unwrap().workspace, 2);

        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["only".into()]; // removes workspaces 1..4
        let events = state.apply_config(cfg);

        let w = state
            .window_manager
            .get(a)
            .expect("window survives, is not closed");
        assert_eq!(
            w.workspace, 0,
            "a window on a removed workspace migrates to 0"
        );
        assert!(
            w.focused,
            "M-1: landing on the active workspace with no focus set must re-pick it"
        );
        assert!(!events.iter().any(|e| matches!(e, Event::WindowClosed(_))));
    }

    /// M1 (review): truncating the workspace list below the active index must
    /// clamp `active_workspace` back to 0 AND emit exactly one
    /// `WorkspaceSet{active: true}` -- `WorkspaceList` does not carry the
    /// active index, so subscribers would otherwise keep showing the vanished
    /// workspace until a manual switch.
    #[test]
    fn apply_config_emits_workspace_set_when_active_workspace_is_truncated() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // default_config has 4 workspaces; make workspace 3 active.
        assert!(state.window_manager.set_active_workspace(3));
        assert_eq!(state.window_manager.active_workspace(), 3);
        // Drain setup events so we only observe the reload's.
        state.window_manager.pending_events.clear();

        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["only".into()]; // removes workspaces 1..4
        let _ = state.apply_config(cfg);

        assert_eq!(
            state.window_manager.active_workspace(),
            0,
            "active clamps into range"
        );
        assert!(
            state
                .window_manager
                .pending_events
                .iter()
                .any(|se| matches!(
                    se.event,
                    Event::WorkspaceSet {
                        id: 0,
                        active: true
                    }
                )),
            "clamping the active workspace must emit a WorkspaceSet"
        );
    }

    /// M-1 (review): truncating the workspace list out from under the
    /// *active* workspace migrates its window to workspace 0 (clearing that
    /// window's own `focused` flag, matching `set_workspace_names`'
    /// documented per-window behavior) but must not leave `focused_window()`
    /// empty -- `apply_config` has to re-pick a successor on the (now-active)
    /// workspace 0, the same belt-and-braces guard `forget_window` already
    /// applies (review finding I6) and `switch_workspace` already applies.
    /// Without that re-pick, the next close/maximize/snap hotkey silently
    /// no-ops on a dangling focus pointer until the user clicks or switches
    /// workspace.
    #[test]
    fn apply_config_refocuses_migrated_window_when_active_workspace_is_truncated() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // default_config has 4 workspaces; park the window on workspace 2
        // and make that workspace active and the window focused.
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.window_manager.set_workspace(a, 2).unwrap();
        assert!(state.window_manager.set_active_workspace(2));
        state.window_manager.focus(a).unwrap();
        assert_eq!(state.window_manager.focused_window().map(|w| w.id), Some(a));

        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["only".into()]; // removes workspaces 1..4
        let _ = state.apply_config(cfg);

        // The window migrated to workspace 0, which is now also the active
        // (clamped) workspace, and it must be the one thing `focused_window`
        // reports -- not `None`.
        assert_eq!(state.window_manager.active_workspace(), 0);
        assert_eq!(state.window_manager.get(a).unwrap().workspace, 0);
        let focused = state.window_manager.focused_window();
        assert_eq!(
            focused.map(|w| w.id),
            Some(a),
            "a migrated window must be re-focused, not left with a dangling focus pointer"
        );
        assert!(
            focused.unwrap().focused,
            "the re-picked window's own focused flag must be set"
        );
    }

    /// Review finding: the brief's own sample test for this name is vacuous
    /// in a plain unit-test harness -- `sync_window_to_scene` early-returns
    /// on `!is_backed(id)`, and nothing in the sample binds the window to a
    /// toplevel, so `apply_reloaded_config`'s recolor/resync block is a
    /// guaranteed no-op regardless of whether it exists (confirmed by
    /// deleting that block and rerunning: the sample still passed).
    ///
    /// This version closes that gap the way the harness actually allows:
    /// `wayland::ToplevelKey::for_test` + `Wayland::bind` make the window
    /// `is_backed` without needing a real `wlr::Runtime` (only the
    /// scene-graph painting inside `sync_ssd` needs one -- see its own
    /// `let Some(runtime) = ... else { return }` guard -- but
    /// `ensure_title_raster` runs *before* that guard, so the raster cache,
    /// this crate's palette-keyed memo, still gets driven for a backed
    /// window even in a runtime-less unit test). Seeding a cached raster
    /// under the *old* palette and asserting its pixels differ after a
    /// reload with a *new* palette is a genuine regression guard: deleting
    /// `apply_reloaded_config`'s `self.sync_scene()` call (which is what
    /// reaches `sync_window_to_scene` -> `ensure_title_raster` for every
    /// surviving window) leaves the pre-reload pixels in place and fails
    /// this test.
    #[test]
    fn reload_rethemes_surviving_windows_and_recolors_background() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        // "app" is not GTK-style, so `decoration::has_ssd` gives it a
        // server-side title bar -- `sync_window_to_scene` only rasterizes a
        // title for a decorated, visible window.
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        state
            .wayland
            .bind(id, crate::wayland::ToplevelKey::for_test(1));

        // Drive one sync under the *old* palette to seed the cache, then
        // capture its pixels.
        state.sync_scene();
        let before = state
            .title_rasters
            .get(&id)
            .expect("a decorated, backed window must have a cached title raster")
            .pixels
            .clone();

        let mut cfg = icedtea_config::default_config();
        cfg.appearance.palette.background = "#abcdef".into();
        cfg.appearance.palette.foreground = "#123456".into();
        state.apply_reloaded_config(cfg);

        assert!(state.window_manager.get(id).is_some(), "window survived");
        assert_eq!(state.config.appearance.palette.background, "#abcdef");
        let after = state
            .title_rasters
            .get(&id)
            .expect("the raster survives the reload (Task 4 preserve contract)")
            .pixels
            .clone();
        assert_ne!(
            before, after,
            "a reload with a new foreground must have re-driven sync_window_to_scene, \
             re-rasterizing the title against the new palette-resolved color"
        );
    }

    #[test]
    fn reload_reswaps_wallpaper_only_on_path_change() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state
            .wallpaper
            .set_decoded(Some(image::RgbaImage::from_pixel(
                1,
                1,
                image::Rgba([0, 0, 0, 255]),
            )));
        let mut cfg = icedtea_config::default_config();
        cfg.appearance.wallpaper = Some("/nonexistent/x.png".into());
        state.apply_reloaded_config(cfg);
        assert!(
            state.wallpaper.decoded().is_none(),
            "path change clears the stale wallpaper before redecode"
        );
    }

    /// The negative counterpart the review flagged as missing: reloading
    /// with the *same* wallpaper path (including the default `None`) must
    /// leave the already-decoded image in place rather than clearing it and
    /// re-spawning a decode for nothing.
    #[test]
    fn reload_leaves_wallpaper_untouched_when_the_path_is_unchanged() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([7, 7, 7, 255]));
        state.wallpaper.set_decoded(Some(img.clone()));

        // Same wallpaper path as `default_config()` (`None`), only an
        // unrelated field changes.
        let mut cfg = icedtea_config::default_config();
        cfg.appearance.palette.background = "#abcdef".into();
        assert_eq!(
            cfg.appearance.wallpaper, state.config.appearance.wallpaper,
            "path unchanged"
        );
        state.apply_reloaded_config(cfg);

        assert_eq!(
            state.wallpaper.decoded(),
            Some(&img),
            "an unchanged wallpaper path must not clear the already-decoded image"
        );
    }

    #[test]
    fn close_action_removes_focused() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.apply_action("close").unwrap();
        assert!(state.window_manager.get(id).is_none());
    }

    #[test]
    fn alt_tab_cycles_focus() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        state.apply_action("cycle:alt_tab").unwrap();
        assert!(state.window_manager.get(a).unwrap().focused);
        state.apply_action("cycle:alt_tab").unwrap();
        assert!(state.window_manager.get(b).unwrap().focused);
        state.apply_action("cycle:alt_tab").unwrap();
        assert!(state.window_manager.get(a).unwrap().focused);
    }

    #[test]
    fn handle_key_dispatches_bound_action() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        // Default config binds SUPER+q to "close" (see config/src/defaults.rs).
        state
            .handle_key(input::Modifiers::SUPER, input::key_name_to_keysym("KEY_q"))
            .unwrap();
        assert!(state.window_manager.get(id).is_none());
    }

    #[test]
    fn handle_key_ignores_unbound_combo() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert_eq!(state.handle_key(input::Modifiers::empty(), 0x12345), None);
    }

    #[test]
    fn handle_pointer_drag_moves_window_without_snap() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            }),
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 100,
                width: 640,
                height: 400,
            },
        );
        state.window_manager.focus(id).unwrap();
        // Press inside the title bar's move area (not on a button).
        state
            .handle_pointer(PointerEvent::Press {
                id,
                pointer: (120, 105),
            })
            .unwrap();
        // Drag to a point away from any snap edge.
        state
            .handle_pointer(PointerEvent::Motion {
                pointer: (400, 400),
            })
            .unwrap();
        assert_eq!(state.snap_preview, None);
        state
            .handle_pointer(PointerEvent::Release {
                pointer: (400, 400),
            })
            .unwrap();
        let geo = state.window_manager.get(id).unwrap().geometry;
        // grab_offset was (20, 5); released at (400, 400) => top-left (380, 395).
        assert_eq!((geo.x, geo.y), (380, 395));
    }

    #[test]
    fn handle_pointer_drag_to_edge_snaps() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            }),
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 100,
                width: 640,
                height: 400,
            },
        );
        state.window_manager.focus(id).unwrap();
        state
            .handle_pointer(PointerEvent::Press {
                id,
                pointer: (120, 105),
            })
            .unwrap();
        state
            .handle_pointer(PointerEvent::Motion { pointer: (2, 400) })
            .unwrap();
        assert!(state.snap_preview.is_some());
        state
            .handle_pointer(PointerEvent::Release { pointer: (2, 400) })
            .unwrap();
        let geo = state.window_manager.get(id).unwrap().geometry;
        assert_eq!(geo.width, 1000 / 2 - 16);
        assert_eq!(state.snap_preview, None);
    }

    // --- Review fix-round tests (task-11-review.md) ---

    /// Important #1: a session's entry list is frozen at `start()` and used
    /// for the whole session even if `window_manager` changes underneath it
    /// mid-cycle -- stepping must not panic or desync just because a window
    /// in the frozen list got removed.
    #[test]
    fn alt_tab_session_survives_churn_and_ends_cleanly() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        let c = state.window_manager.add_window("c", "c", 3, DEFAULT_GEO);
        // `alt_tab_entries()` is id-ordered: [a, b, c]. First cycle starts
        // the session and focuses entries[0] = a.
        state.apply_action("cycle:alt_tab").unwrap();
        assert!(state.alt_tab.is_active());
        assert!(state.window_manager.get(a).unwrap().focused);

        // Churn mid-session: a window in the frozen entry list disappears.
        state.window_manager.remove_window(b);

        // Stepping must not panic even though the stored entries still
        // reference the now-gone `b` (index 1); `WindowManager::focus`
        // quietly no-ops on a missing id instead of this call failing.
        state.apply_action("cycle:alt_tab").unwrap(); // steps to index 1 (b, gone)
        state.apply_action("cycle:alt_tab").unwrap(); // steps to index 2 (c)
        assert!(state.alt_tab.is_active());
        assert_eq!(
            state.alt_tab.entries().len(),
            3,
            "entry list must stay frozen across churn"
        );
        assert!(state.window_manager.get(c).unwrap().focused);

        // Drain events so we can inspect exactly what `end_alt_tab` sends.
        let _ = rx.try_iter().count();
        state.end_alt_tab();
        assert!(!state.alt_tab.is_active());
        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            events.iter().any(|e| matches!(
                e.event,
                Event::AltTabState(AltTabState { active: false, .. })
            )),
            "end_alt_tab must emit a terminal AltTabState so the shell overlay can dismiss"
        );

        // A no-op call afterwards must not emit a second terminal event.
        state.end_alt_tab();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn end_alt_tab_is_a_noop_when_no_session_is_active() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert!(!state.alt_tab.is_active());
        state.end_alt_tab();
        assert!(
            rx.try_recv().is_err(),
            "no session was active, nothing should be emitted"
        );
    }

    /// Important #2: `saved_geometry` used to be a single map shared by
    /// `toggle_fullscreen` and `snap`; snapping then fullscreening a window
    /// clobbered the pre-snap restore point with the snapped geometry, and
    /// `snap_restore` after exiting fullscreen silently did nothing.
    #[test]
    fn snap_then_fullscreen_then_unfullscreen_then_snap_restore_round_trips() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            }),
        );
        let original = Rectangle {
            x: 50,
            y: 60,
            width: 300,
            height: 200,
        };
        let id = state.window_manager.add_window("app", "t", 1, original);
        state.window_manager.focus(id).unwrap();

        state.snap(id, SnapZone::Left).unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry.width,
            1000 / 2 - 16
        );

        state.toggle_fullscreen(id).unwrap(); // enter fullscreen from the snapped geometry
        assert!(state.window_manager.get(id).unwrap().fullscreen);
        state.toggle_fullscreen(id).unwrap(); // exit fullscreen
        assert!(!state.window_manager.get(id).unwrap().fullscreen);
        // Exiting fullscreen must restore the *snapped* geometry, not the
        // pre-snap original -- fullscreen's own restore point is untouched
        // by snap's map.
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry.width,
            1000 / 2 - 16
        );

        state.snap_restore(id).unwrap();
        assert_eq!(state.window_manager.get(id).unwrap().geometry, original);
    }

    /// A live reload must PRESERVE every window row and its client -- no
    /// `WindowClosed` burst -- while still swapping in the new appearance.
    /// This is the M3 contract that replaced the old "rebuild the manager
    /// and close everything" behavior.
    #[test]
    fn apply_config_preserves_windows_and_does_not_close_them() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let a = state.window_manager.add_window(
            "a",
            "a",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        );
        let b = state.window_manager.add_window(
            "b",
            "b",
            1,
            Rectangle {
                x: 10,
                y: 10,
                width: 100,
                height: 100,
            },
        );
        let mut cfg = icedtea_config::default_config();
        cfg.appearance.palette.background = "#123456".into();
        let events = state.apply_config(cfg);
        assert!(
            state.window_manager.get(a).is_some() && state.window_manager.get(b).is_some(),
            "windows survive a reload"
        );
        assert!(
            !events.iter().any(|e| matches!(e, Event::WindowClosed(_))),
            "no WindowClosed burst on reload"
        );
        assert_eq!(state.config.appearance.palette.background, "#123456");
    }

    /// A post-reload `add_window` must never reuse a pre-reload id: because
    /// the manager is no longer rebuilt, `next_id` advances monotonically on
    /// its own without any explicit floor.
    #[test]
    fn apply_config_keeps_id_counter_monotonic() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        let _ = state.apply_config(icedtea_config::default_config());
        let c = state.window_manager.add_window("c", "c", 3, DEFAULT_GEO);
        assert!(c.0 > a.0 && c.0 > b.0, "ids never regress across a reload");
    }

    /// Important #4: `n - 1` on a `u32` action argument must never panic,
    /// and an out-of-range workspace index must be a reported failure, not
    /// a silent no-op success.
    #[test]
    fn workspace_zero_does_not_panic_and_fails_cleanly() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert_eq!(state.apply_action("workspace:0"), None);
        assert_eq!(
            state.window_manager.active_workspace(),
            0,
            "a failed switch must not move the active workspace"
        );
    }

    #[test]
    fn workspace_out_of_range_fails_instead_of_reporting_success() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // Default config has 4 workspaces (config/src/defaults.rs); 99 is
        // well out of range.
        assert_eq!(state.apply_action("workspace:99"), None);
        assert_eq!(state.window_manager.active_workspace(), 0);
    }

    #[test]
    fn move_to_workspace_zero_does_not_panic_and_fails_cleanly() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        assert_eq!(state.apply_action("move_to_workspace:0"), None);
    }

    // --- Re-review fix-round tests (task-11-review.md round 2) ---

    /// Important #2: pressing an unfocused window must focus it, and that
    /// focus must take effect *before* the decoration action for the same
    /// click is evaluated (so the click that raises a window can also act
    /// on it in one motion, matching ordinary click-to-focus behavior).
    #[test]
    fn press_on_unfocused_window_focuses_then_hits_its_decorations() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let a_geo = Rectangle {
            x: 0,
            y: 0,
            width: 640,
            height: 400,
        };
        let a = state.window_manager.add_window("a", "a", 1, a_geo);
        let b = state.window_manager.add_window(
            "b",
            "b",
            2,
            Rectangle {
                x: 700,
                y: 0,
                width: 640,
                height: 400,
            },
        );
        // `b` was added last, so it's focused; `a` is not.
        assert!(!state.window_manager.get(a).unwrap().focused);
        assert!(state.window_manager.get(b).unwrap().focused);

        // Press on `a`'s title bar (not a button): before the fix,
        // `decoration_action_for` would gate on `a.focused` (false) and
        // this whole call would return `None`, doing nothing.
        state
            .handle_pointer(PointerEvent::Press {
                id: a,
                pointer: (20, 5),
            })
            .unwrap();
        assert!(state.window_manager.get(a).unwrap().focused);
        assert!(!state.window_manager.get(b).unwrap().focused);

        // Now that `a` is focused, a press on its close button must close
        // it -- proving the *same* click sequence both focuses and acts.
        let close_pt = (a_geo.x + a_geo.width - 5, a_geo.y + 5);
        state
            .handle_pointer(PointerEvent::Press {
                id: a,
                pointer: close_pt,
            })
            .unwrap();
        assert!(state.window_manager.get(a).is_none());
    }

    /// Important #3 (adjacent staleness bugs on the reload seam): a reload
    /// mid-drag/mid-alt-tab must not leave `drag`/`alt_tab`/`snap_preview`
    /// referencing windows that `apply_config` is about to discard.
    #[test]
    fn apply_config_resets_drag_alt_tab_and_snap_preview() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            }),
        );
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);

        state.drag.begin(a, (5, 5));
        state.alt_tab.start(vec![a, b]);
        state.snap_preview = Some(Rectangle {
            x: 0,
            y: 0,
            width: 500,
            height: 800,
        });
        assert!(state.drag.window_id().is_some());
        assert!(state.alt_tab.is_active());
        assert!(state.snap_preview.is_some());

        let _ = state.apply_config(icedtea_config::default_config());

        assert!(
            state.drag.window_id().is_none(),
            "drag must not survive a reload mid-drag"
        );
        assert!(
            !state.alt_tab.is_active(),
            "alt-tab session must not survive a reload mid-cycle"
        );
        assert!(
            state.snap_preview.is_none(),
            "a stale snap preview must not render forever after reload"
        );
    }

    /// Important #4: `snapshot().seq` must never go backwards across a
    /// reload, and `State::emit` (used for `AltTabState`) must advance the
    /// same counter as every other event.
    #[test]
    fn apply_config_does_not_regress_seq() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        let seq_before = state.window_manager.snapshot().seq;
        assert!(seq_before > 0);

        let _ = state.apply_config(icedtea_config::default_config());

        assert!(
            state.window_manager.snapshot().seq >= seq_before,
            "a fresh WindowManager's seq must be floored at the pre-reload high-water mark"
        );
    }

    #[test]
    fn emit_advances_seq() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let seq_before = state.window_manager.snapshot().seq;
        state.emit(Event::AltTabState(AltTabState {
            active: true,
            entries: vec![],
            index: 0,
        }));
        assert!(
            state.window_manager.snapshot().seq > seq_before,
            "State::emit must bump seq like every other pending-event producer"
        );
    }

    // --- Re-review fix-round-3 tests (task-11-review.md round 3) ---

    /// Important #1: a reload mid-alt-tab-cycle must emit the terminal
    /// `AltTabState{active: false, ..}` so a shell overlay rendered from
    /// the session's last `active: true` event has a dismiss signal,
    /// instead of `apply_config` just silently resetting the machine.
    #[test]
    fn apply_config_mid_cycle_emits_alt_tab_inactive() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        state.apply_action("cycle:alt_tab").unwrap();
        assert!(state.alt_tab.is_active());

        let events = state.apply_config(icedtea_config::default_config());

        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::AltTabState(AltTabState { active: false, .. }))),
            "reload mid-cycle must emit the terminal AltTabState"
        );
        assert!(!state.alt_tab.is_active());
    }

    #[test]
    fn apply_config_when_alt_tab_inactive_emits_no_extra_alt_tab_state() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert!(!state.alt_tab.is_active());
        let events = state.apply_config(icedtea_config::default_config());
        assert!(!events.iter().any(|e| matches!(e, Event::AltTabState(_))));
    }

    /// Minor #2: `handle_pointer_press`'s `focus()` mutation must flush
    /// immediately, not get deferred by an early `?`-return further down
    /// the same function (e.g. `decoration_action_for` returning `None`
    /// for a fullscreen window).
    #[test]
    fn press_on_unfocused_fullscreen_window_flushes_focus_immediately() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            }),
        );
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let _b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        state.window_manager.set_fullscreen(a, true).unwrap();
        let _ = rx.try_iter().count(); // drain setup events
        assert!(
            !state.window_manager.get(a).unwrap().focused,
            "b was added last and is focused, not a"
        );

        // `decoration_action_for` returns `None` for a fullscreen window,
        // so this call itself returns `None` -- but the focus mutation
        // must already be visible on the channel by the time it does.
        assert_eq!(
            state.handle_pointer(PointerEvent::Press {
                id: a,
                pointer: (20, 5)
            }),
            None
        );
        assert!(state.window_manager.get(a).unwrap().focused);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(&e.event, Event::WindowUpdated { id, update } if *id == a && update.focused == Some(true))),
            "focus event must be flushed immediately, not deferred until an unrelated later flush"
        );
    }

    // --- Task 13: config hot reload over `ReloadConfig` ---

    #[test]
    fn reload_config_applies_new_workspaces_and_emits() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        // Persist a config with different workspace names, then reload from disk.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.redb");
        let db = icedtea_config::open(&path).unwrap();
        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["A".into(), "B".into(), "C".into()];
        cfg.save(&db).unwrap();
        drop(db);

        state.config_path = Some(path.clone());
        let events = state.reload_config_from_disk();
        assert_eq!(state.window_manager.workspace_info().len(), 3);
        assert!(events.iter().any(|e| matches!(e, Event::ConfigReloaded(_))));
        let emitted = rx.try_iter().collect::<Vec<_>>();
        assert!(
            emitted
                .iter()
                .any(|e| matches!(e.event, Event::ConfigReloaded(_)))
        );
    }

    /// `handle_command`'s `ReloadConfig` arm must go through the async
    /// worker path -- it only spawns a thread and returns, never blocking
    /// on the load itself -- and must be a harmless no-op when
    /// `config_reload_tx` hasn't been wired (as in every other test in this
    /// module, which never call `set_config_reload_sender`).
    #[test]
    fn handle_command_reload_without_sender_wired_is_a_harmless_noop() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert_eq!(
            state.handle_command(crate::dbus::DbCommand::ReloadConfig),
            Some(())
        );
    }

    /// The async worker path actually delivers a reloaded config back to
    /// the channel `set_config_reload_sender` was given, and
    /// `apply_reloaded_config` applies + emits it exactly like the sync
    /// path.
    #[test]
    fn handle_command_reload_spawns_worker_that_delivers_config() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.redb");
        let db = icedtea_config::open(&path).unwrap();
        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["X".into(), "Y".into()];
        cfg.save(&db).unwrap();
        drop(db);
        state.config_path = Some(path);

        let (reload_tx, reload_rx) = crossbeam_channel::unbounded::<Config>();
        state.set_config_reload_sender(reload_tx);

        assert_eq!(
            state.handle_command(crate::dbus::DbCommand::ReloadConfig),
            Some(())
        );

        // The worker thread runs off-loop; block briefly for its result
        // (mirrors how `drain_config_reload` would pick it up on the next
        // turn).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut received = None;
        while std::time::Instant::now() < deadline {
            match reload_rx.try_recv() {
                Ok(cfg) => {
                    received = Some(cfg);
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(5))
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => break,
            }
        }
        let cfg = received.expect("worker thread must deliver a Config over the reload channel");
        assert_eq!(cfg.workspace_names, vec!["X".to_string(), "Y".to_string()]);

        state.apply_reloaded_config(cfg);
        assert_eq!(state.window_manager.workspace_info().len(), 2);
        let emitted = rx.try_iter().collect::<Vec<_>>();
        assert!(
            emitted
                .iter()
                .any(|e| matches!(e.event, Event::ConfigReloaded(_)))
        );
    }

    /// The keybinding-triggered `"reload"` action (`SUPER+SHIFT+r` by
    /// default) must go through the exact same async worker path as the
    /// D-Bus `ReloadConfig` command -- per the task-13 threading
    /// requirement, neither may block the render loop with redb I/O.
    #[test]
    fn apply_action_reload_spawns_worker_that_delivers_config() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.redb");
        let db = icedtea_config::open(&path).unwrap();
        let mut cfg = icedtea_config::default_config();
        cfg.workspace_names = vec!["P".into(), "Q".into()];
        cfg.save(&db).unwrap();
        drop(db);
        state.config_path = Some(path);

        let (reload_tx, reload_rx) = crossbeam_channel::unbounded::<Config>();
        state.set_config_reload_sender(reload_tx);

        assert_eq!(state.apply_action("reload"), Some(()));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut received = None;
        while std::time::Instant::now() < deadline {
            match reload_rx.try_recv() {
                Ok(cfg) => {
                    received = Some(cfg);
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(5))
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => break,
            }
        }
        let cfg = received.expect("worker thread must deliver a Config over the reload channel");
        assert_eq!(cfg.workspace_names, vec!["P".to_string(), "Q".to_string()]);
    }

    #[test]
    fn apply_action_reload_without_sender_wired_is_a_harmless_noop() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert_eq!(state.apply_action("reload"), Some(()));
    }

    // --- Final-review fix-round tests ---

    fn state_with_output(
        width: i32,
        height: i32,
    ) -> (State, crossbeam_channel::Receiver<SeqEvent>) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.outputs.insert(
            0,
            OutputSurface::new(Rectangle {
                x: 0,
                y: 0,
                width,
                height,
            }),
        );
        (state, rx)
    }

    /// C1/I5: maximize computes and applies real geometry against its own
    /// restore slot, and toggling back returns exactly the pre-maximize
    /// geometry -- it used to flip a flag and nothing else.
    #[test]
    fn maximize_applies_output_geometry_and_restores_it() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let original = Rectangle {
            x: 40,
            y: 50,
            width: 300,
            height: 200,
        };
        let id = state.window_manager.add_window("app", "t", 1, original);

        state.toggle_maximized(id).unwrap();
        let w = state.window_manager.get(id).unwrap();
        assert!(w.maximized);
        assert_eq!(
            w.geometry,
            layout::maximized_geometry(
                Rectangle {
                    x: 0,
                    y: 0,
                    width: 1000,
                    height: 800
                },
                8
            )
        );

        state.toggle_maximized(id).unwrap();
        let w = state.window_manager.get(id).unwrap();
        assert!(!w.maximized);
        assert_eq!(w.geometry, original);
    }

    /// I5: maximize's restore slot is independent of snap's, exactly like
    /// fullscreen's -- snapping between maximize and unmaximize must not
    /// clobber either restore point.
    #[test]
    fn maximize_and_snap_keep_independent_restore_points() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let original = Rectangle {
            x: 40,
            y: 50,
            width: 300,
            height: 200,
        };
        let id = state.window_manager.add_window("app", "t", 1, original);
        state.window_manager.focus(id).unwrap();

        state.snap(id, SnapZone::Left).unwrap();
        let snapped = state.window_manager.get(id).unwrap().geometry;
        state.set_maximized_target(id, true).unwrap();
        state.set_maximized_target(id, false).unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry,
            snapped,
            "unmaximize returns to the snapped geometry"
        );
        state.snap_restore(id).unwrap();
        assert_eq!(state.window_manager.get(id).unwrap().geometry, original);
    }

    /// I5: `MaximizeWindow` over D-Bus is not a visual no-op any more, and
    /// an explicit target that already holds stays a no-op (it must not
    /// re-save the current geometry as a fresh restore point).
    #[test]
    fn maximize_command_is_idempotent_on_its_target() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let original = Rectangle {
            x: 40,
            y: 50,
            width: 300,
            height: 200,
        };
        let id = state.window_manager.add_window("app", "t", 1, original);
        state
            .handle_command(crate::dbus::DbCommand::Maximize(id, true))
            .unwrap();
        let maximized = state.window_manager.get(id).unwrap().geometry;
        state
            .handle_command(crate::dbus::DbCommand::Maximize(id, true))
            .unwrap();
        assert_eq!(state.window_manager.get(id).unwrap().geometry, maximized);
        state
            .handle_command(crate::dbus::DbCommand::Maximize(id, false))
            .unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry,
            original,
            "the restore point survived the no-op"
        );
    }

    /// H1: `set_maximized_target` resolves the output via the window's own
    /// frame center (`output_for_window`), not the pointer -- with two
    /// outputs and no runtime attached, `output_for_pointer`'s own fallback
    /// is the *lowest* index (output 0), which is exactly the wrong output
    /// for a window that lives on output 1. Reached from client requests
    /// and `DbCommand::Maximize`, neither of which carries any pointer
    /// correlation with the target window at all.
    #[test]
    fn set_maximized_target_resolves_via_the_windows_own_output_not_the_pointer() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.config.appearance.snap_gap = 0;
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 810,
                y: 10,
                width: 300,
                height: 200,
            },
        );
        state.set_maximized_target(id, true).unwrap();

        let geo = state.window_manager.get(id).unwrap().geometry;
        assert_eq!(
            geo, state.outputs[&1].usable,
            "must maximize onto output 1's usable rect, not output 0's"
        );
        assert!(
            geo.x >= 800,
            "must not have resolved onto output 0, got {geo:?}"
        );
    }

    /// H1's fullscreen counterpart: `set_fullscreen_target` resolves via the
    /// window's own frame center too, and keeps `.geometry` (not `.usable`)
    /// once it does -- fullscreen covers panels by definition.
    #[test]
    fn set_fullscreen_target_resolves_via_the_windows_own_output_not_the_pointer() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 810,
                y: 10,
                width: 300,
                height: 200,
            },
        );
        state.set_fullscreen_target(id, true).unwrap();

        let geo = state.window_manager.get(id).unwrap().geometry;
        assert_eq!(
            geo, state.outputs[&1].geometry,
            "must fullscreen onto output 1's geometry, not output 0's"
        );
        assert!(
            geo.x >= 800,
            "must not have resolved onto output 0, got {geo:?}"
        );
    }

    /// Re-review Important 1: `DbCommand::Minimize` goes through the same
    /// implementation as the title-bar button, so minimizing the focused
    /// window over D-Bus (the taskbar path) hands focus to the workspace's
    /// MRU successor instead of leaving the model's focus pointer on a
    /// now-invisible window and the keyboard dead.
    #[test]
    fn dbus_minimize_of_focused_window_hands_focus_to_successor() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let geo = Rectangle {
            x: 10,
            y: 10,
            width: 300,
            height: 200,
        };
        let a = state.window_manager.add_window("app", "a", 1, geo);
        let b = state.window_manager.add_window("app", "b", 2, geo);
        state.window_manager.focus(b).unwrap();

        state
            .handle_command(crate::dbus::DbCommand::Minimize(b, true))
            .unwrap();
        assert!(state.window_manager.get(b).unwrap().minimized);
        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(a),
            "focus handed to the MRU successor, matching the title-bar button"
        );
    }

    /// Review B#1: `set_minimized_and_reconcile` drains its queued
    /// `WindowUpdated { minimized }` to D-Bus itself, so the
    /// `xwayland_request_minimize` trait path (an X11 app self-minimizing) --
    /// which does not flow through `handle_command`'s `emit_pending` tail --
    /// still notifies a subscribed taskbar. Driving the shared reconcile
    /// method directly reproduces that path without a live Xwayland.
    #[test]
    fn self_minimize_emits_the_window_update_to_dbus() {
        let (mut state, rx) = state_with_output(1000, 800);
        let geo = Rectangle {
            x: 10,
            y: 10,
            width: 300,
            height: 200,
        };
        let id = state.window_manager.add_window("app", "a", 1, geo);
        state.window_manager.focus(id).unwrap();
        state.emit_pending();
        while rx.try_recv().is_ok() {} // drain the open/focus events

        // The trait handler calls this directly (no `handle_command` tail).
        state.set_minimized_and_reconcile(id, true).unwrap();

        let saw_minimized = std::iter::from_fn(|| rx.try_recv().ok()).any(|e| {
            matches!(
                e.event,
                Event::WindowUpdated { id: wid, update }
                    if wid == id && update.minimized == Some(true)
            )
        });
        assert!(
            saw_minimized,
            "a self-minimize must emit WindowUpdated{{minimized:true}} to D-Bus"
        );
    }

    /// Review finding #1: `sync_scene` must re-raise windows bottom-to-top in
    /// reverse focus-MRU so the focused window (and its SSD) ends up on top after
    /// every window's decoration was rebuilt at the top of the band. The order is
    /// the visible windows, least-recently-focused first; a minimized window is
    /// excluded so it is never restacked while hidden.
    #[test]
    fn stacking_order_is_visible_windows_bottom_to_top() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let geo = Rectangle {
            x: 0,
            y: 0,
            width: 200,
            height: 150,
        };
        let a = state.window_manager.add_window("app", "a", 1, geo);
        let b = state.window_manager.add_window("app", "b", 2, geo);
        let c = state.window_manager.add_window("app", "c", 3, geo);
        // Focus order a, b, c -> MRU is [c, b, a] (c on top), so bottom-to-top is
        // [a, b, c] with the focused c raised last (ending on top).
        state.window_manager.focus(a).unwrap();
        state.window_manager.focus(b).unwrap();
        state.window_manager.focus(c).unwrap();
        assert_eq!(state.stacking_order_bottom_to_top(), vec![a, b, c]);

        // A minimized window drops out (never restacked while hidden).
        state.window_manager.set_minimized(b, true).unwrap();
        assert_eq!(state.stacking_order_bottom_to_top(), vec![a, c]);
    }

    /// Review finding #7: closing a keyboard-holding OR submenu returns the
    /// keyboard to its still-open parent menu (the next entry down the stack),
    /// not the model toplevel. A single-slot design stole the keyboard from the
    /// parent; the stack keeps it.
    #[test]
    fn closing_a_submenu_returns_keyboard_to_its_parent_menu() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let geo = Rectangle {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let menu1 = wlr::XwaylandSurfaceId::dangling_nth_for_test(1);
        let menu2 = wlr::XwaylandSurfaceId::dangling_nth_for_test(2);
        state
            .override_redirect
            .insert(menu1, OverrideRedirectSurface { geometry: geo });
        state
            .override_redirect
            .insert(menu2, OverrideRedirectSurface { geometry: geo });
        // menu1 is the parent, menu2 the submenu currently holding the keyboard.
        state.or_keyboard_stack = vec![menu1, menu2];

        // Submenu closes: keyboard must return to the parent, not the model.
        state.remove_override_redirect(menu2);
        assert_eq!(
            state.or_keyboard_stack.last(),
            Some(&menu1),
            "closing the submenu must hand the keyboard back to its parent menu"
        );

        // A middle entry closing (parent dismissed while submenu still open)
        // leaves the current holder untouched.
        state
            .override_redirect
            .insert(menu2, OverrideRedirectSurface { geometry: geo });
        state.or_keyboard_stack = vec![menu1, menu2];
        state.remove_override_redirect(menu1);
        assert_eq!(
            state.or_keyboard_stack,
            vec![menu2],
            "closing the parent must splice it out and leave the submenu holding the keyboard"
        );

        // Closing the last pop-up empties the stack (model focus restored).
        state.remove_override_redirect(menu2);
        assert!(
            state.or_keyboard_stack.is_empty(),
            "closing the last OR pop-up must empty the keyboard stack"
        );
    }

    /// Review finding #6: a dialog centered over a transient parent that lives
    /// on a different monitor than the pointer must be clamped to the *parent's*
    /// output, not the pointer's — otherwise it is dragged onto the wrong
    /// screen. `center_in_area` clamps to the output the centering `area` sits on.
    #[test]
    fn center_in_area_clamps_to_the_areas_own_output_not_the_fallback() {
        let (mut state, _rx) = state_with_output(1000, 800); // output A: x in 0..1000
        state.create_output(
            1,
            Rectangle {
                x: 1000,
                y: 0,
                width: 1000,
                height: 800,
            },
        ); // output B
        // Parent frame sits on output B; the pointer's usable area (fallback) is A.
        let parent_on_b = Rectangle {
            x: 1200,
            y: 300,
            width: 400,
            height: 300,
        };
        let fallback_a = Rectangle {
            x: 0,
            y: 0,
            width: 1000,
            height: 800,
        };
        let (x, y) = state.center_in_area(parent_on_b, 200, 150, fallback_a);
        // Centered over the parent: x = 1200 + (400-200)/2 = 1300, on output B.
        assert_eq!((x, y), (1300, 375), "dialog centers over its parent on B");
        assert!(
            x >= 1000,
            "the dialog must land on the parent's output B, not the pointer's A"
        );
    }

    /// Review finding #9: a frame whose SSD title bar would sit above the top
    /// of the output (a self-positioning X11 client requesting content at y=0,
    /// so the 28px bar spans y=-28..0) is clamped back on-screen so its only
    /// drag handle and window buttons stay reachable. A frame already fully on
    /// its output is returned unchanged.
    #[test]
    fn clamp_frame_onto_output_pulls_an_offscreen_title_bar_back() {
        let (state, _rx) = state_with_output(1000, 800);
        // Title bar above the top edge -> clamped so y >= output top (0).
        let off_top = Rectangle {
            x: 100,
            y: -28,
            width: 300,
            height: 228,
        };
        let fixed = state.clamp_frame_onto_output(off_top);
        assert_eq!(fixed.y, 0, "the title bar must not sit above the output");
        assert_eq!(
            fixed.x, 100,
            "x was already on-screen and must be left alone"
        );
        assert_eq!(
            (fixed.width, fixed.height),
            (300, 228),
            "size is never changed"
        );
        // Off the right/bottom -> clamped so the frame stays fully on-output.
        let off_corner = Rectangle {
            x: 950,
            y: 790,
            width: 300,
            height: 228,
        };
        let fixed = state.clamp_frame_onto_output(off_corner);
        assert_eq!(
            fixed.x,
            1000 - 300,
            "clamped to keep the right edge on-output"
        );
        assert_eq!(
            fixed.y,
            800 - 228,
            "clamped to keep the bottom edge on-output"
        );
        // A frame already fully on-screen is untouched.
        let on = Rectangle {
            x: 200,
            y: 200,
            width: 300,
            height: 228,
        };
        assert_eq!(state.clamp_frame_onto_output(on), on);
    }

    /// I3: `behavior.snap_enabled` is actually consumed -- with snapping
    /// off, neither the `snap:*` actions nor a drag to an edge snap
    /// anything, and the drag still performs a plain move.
    #[test]
    fn snap_disabled_blocks_actions_and_drag_snapping() {
        let (mut state, _rx) = state_with_output(1000, 800);
        state.config.behavior.snap_enabled = false;
        let geo = Rectangle {
            x: 100,
            y: 100,
            width: 640,
            height: 400,
        };
        let id = state.window_manager.add_window("app", "t", 1, geo);
        state.window_manager.focus(id).unwrap();

        assert_eq!(
            state.apply_action("snap:left"),
            None,
            "the snap action must not fire when snapping is off"
        );
        assert_eq!(state.window_manager.get(id).unwrap().geometry, geo);

        // A drag to the left edge shows no preview and ends as a plain move.
        state
            .handle_pointer(PointerEvent::Press {
                id,
                pointer: (120, 105),
            })
            .unwrap();
        state
            .handle_pointer(PointerEvent::Motion { pointer: (2, 400) })
            .unwrap();
        assert_eq!(
            state.snap_preview, None,
            "no snap preview may be drawn when snapping is off"
        );
        state
            .handle_pointer(PointerEvent::Release { pointer: (2, 400) })
            .unwrap();
        let moved = state.window_manager.get(id).unwrap().geometry;
        assert_eq!(
            (moved.x, moved.y),
            (2 - 20, 400 - 5),
            "the drag still moves the window"
        );
        assert_eq!(
            (moved.width, moved.height),
            (geo.width, geo.height),
            "…without resizing it"
        );
    }

    /// I3 (the other direction): the default config leaves snapping on, so
    /// nothing about the existing behavior changes.
    #[test]
    fn snap_enabled_by_default_still_snaps() {
        let (mut state, _rx) = state_with_output(1000, 800);
        assert!(state.config.behavior.snap_enabled);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.window_manager.focus(id).unwrap();
        assert_eq!(state.apply_action("snap:left"), Some(()));
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry.width,
            1000 / 2 - 16
        );
    }

    /// I6: after `MoveToWorkspace` the destination has a focused window, so
    /// the *next* action isn't a silent no-op -- and the origin workspace is
    /// left focused on whatever it still has.
    #[test]
    fn move_to_workspace_focuses_the_moved_window_and_reseats_the_origin() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        // `b` is focused (added last) and is the one that moves.
        state
            .handle_command(crate::dbus::DbCommand::MoveToWorkspace(b, 1))
            .unwrap();

        assert_eq!(state.window_manager.active_workspace(), 1);
        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(b),
            "the moved window is focused there"
        );
        // The origin kept a coherent focus rather than a dangling pointer.
        state.switch_workspace(0).unwrap();
        assert_eq!(state.window_manager.focused_window().map(|w| w.id), Some(a));
    }

    /// I6, via the keybinding path: the action that used to leave the
    /// destination unfocused is followed by a `fullscreen` that must act on
    /// the moved window.
    #[test]
    fn action_after_move_to_workspace_is_not_a_noop() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.apply_action("move_to_workspace:2").unwrap();
        assert_eq!(
            state.apply_action("fullscreen"),
            Some(()),
            "the next action must find a focused window"
        );
        assert!(state.window_manager.get(id).unwrap().fullscreen);
    }

    /// I1/I6: switching to a workspace that has windows focuses its MRU
    /// head, so actions work there immediately; switching to an empty one
    /// leaves focus cleanly absent rather than pointing elsewhere.
    #[test]
    fn switch_workspace_focuses_that_workspaces_mru_head() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.switch_workspace(1).unwrap();
        assert!(
            state.window_manager.focused_window().is_none(),
            "workspace 2 is empty"
        );
        assert_eq!(
            state.apply_action("close"),
            None,
            "…so an action there is a clean no-op"
        );
        state.switch_workspace(0).unwrap();
        assert_eq!(state.window_manager.focused_window().map(|w| w.id), Some(a));
    }

    /// I2: every event reaching the D-Bus channel carries the seq its
    /// mutation produced, those seqs strictly increase, and the last one
    /// matches `snapshot().seq` -- which is what lets a subscriber order
    /// signals against a `GetState()` snapshot.
    #[test]
    fn emitted_events_carry_monotonic_seq_matching_the_snapshot() {
        let (mut state, rx) = state_with_output(1000, 800);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.emit_pending();
        state
            .handle_command(crate::dbus::DbCommand::Fullscreen(id, true))
            .unwrap();
        state
            .handle_command(crate::dbus::DbCommand::SetWorkspace(1))
            .unwrap();

        let events: Vec<SeqEvent> = rx.try_iter().collect();
        assert!(
            events.len() >= 3,
            "expected several events, got {}",
            events.len()
        );
        let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
        assert!(
            seqs.windows(2).all(|w| w[1] > w[0]),
            "seqs must strictly increase: {seqs:?}"
        );
        assert_eq!(*seqs.last().unwrap(), state.window_manager.snapshot().seq);
        assert!(
            seqs[0] > 0,
            "seq 0 means 'nothing has happened yet' and must never be emitted"
        );
    }

    /// I2: the reload path used to bypass the counter entirely (its events
    /// went out through a separate queue with no seq of their own).
    #[test]
    fn reloaded_config_events_carry_seq_and_never_regress() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.emit_pending();
        let before = state.window_manager.snapshot().seq;
        let drained: Vec<SeqEvent> = rx.try_iter().collect();
        assert!(drained.iter().all(|e| e.seq <= before));

        state.apply_reloaded_config(icedtea_config::default_config());
        let events: Vec<SeqEvent> = rx.try_iter().collect();
        assert!(!events.is_empty());
        assert!(
            events.iter().all(|e| e.seq > before),
            "reload events must advance past the pre-reload high-water mark"
        );
        let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
        assert!(
            seqs.windows(2).all(|w| w[1] > w[0]),
            "seqs must strictly increase: {seqs:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, Event::ConfigReloaded(_)))
        );
    }

    /// C1: closing a model window with no backing client surface still
    /// removes it synchronously (there is no client to send `close` to and
    /// no destroy will ever arrive), and focus lands somewhere sensible.
    #[test]
    fn request_close_without_a_surface_removes_and_reseats_focus() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        state.request_close(b);
        assert!(state.window_manager.get(b).is_none());
        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(a),
            "focus must not dangle after a close"
        );
    }

    /// C1: closing must not leave restore points behind for an id that can
    /// never come back.
    #[test]
    fn closing_clears_saved_geometry_slots() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.window_manager.focus(id).unwrap();
        state.snap(id, SnapZone::Left).unwrap();
        state.set_maximized_target(id, true).unwrap();
        state.set_fullscreen_target(id, true).unwrap();
        assert!(state.snap_saved_geometry.contains_key(&id));
        state.request_close(id);
        assert!(!state.snap_saved_geometry.contains_key(&id));
        assert!(!state.maximized_saved_geometry.contains_key(&id));
        assert!(!state.fullscreen_saved_geometry.contains_key(&id));
    }

    /// C1 (`resize_request`'s model half): an interactive resize driven by
    /// the pointer path updates the model geometry as it goes and commits
    /// the final one on release.
    #[test]
    fn interactive_resize_updates_geometry_and_ends_on_release() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let geo = Rectangle {
            x: 100,
            y: 100,
            width: 400,
            height: 300,
        };
        let id = state.window_manager.add_window("app", "t", 1, geo);
        state.resize.begin(
            id,
            input::ResizeEdges {
                bottom: true,
                right: true,
                ..Default::default()
            },
            geo,
            (500, 400),
        );

        state
            .handle_pointer(PointerEvent::Motion {
                pointer: (560, 430),
            })
            .unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry,
            Rectangle {
                x: 100,
                y: 100,
                width: 460,
                height: 330
            }
        );

        state
            .handle_pointer(PointerEvent::Release {
                pointer: (600, 500),
            })
            .unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry,
            Rectangle {
                x: 100,
                y: 100,
                width: 500,
                height: 400
            }
        );
        assert!(
            state.resize.window_id().is_none(),
            "the resize ended on release"
        );
        // A later motion with no resize and no drag is a clean no-op.
        assert_eq!(
            state.handle_pointer(PointerEvent::Motion {
                pointer: (700, 700)
            }),
            None
        );
    }

    /// M2: `State::new` accepts an arbitrary `Config`, including one with no
    /// workspace names -- that must not leave a reachable index panic.
    #[test]
    fn state_with_no_configured_workspaces_does_not_panic() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut cfg = default_config();
        cfg.workspace_names = vec![];
        let mut state = State::new(cfg, tx);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        assert_eq!(state.window_manager.workspace_info().len(), 1);
        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(id)
        );
    }

    // --- Re-review fix round: New-1 / New-2 / New-3 ---
    //
    // These cover the *model* half of the focus-sync work plus the fact that
    // every focus path now runs `sync_focus_change`/`sync_seat_focus` without
    // panicking. The Wayland half proper -- that the losing window's client
    // is staged `Activated = false`, that the successor's client is staged
    // `Activated = true`, and that `wayland.keyboard_focus(Some(id))` is
    // called -- is **not** observable here: no window in this module's
    // tests is ever bound to a toplevel (`state.wayland.is_backed` is false
    // for all of them), so `sync_window_to_scene` short-circuits before
    // touching client state and `wayland.keyboard_focus` (a no-op in this
    // commit regardless) never fires. See the fix report for the
    // manual-trace argument that stands in for those.

    /// New-2: the successor `forget_window` picks is reconciled, not just
    /// chosen -- and the model never ends up with a dangling focus pointer.
    #[test]
    fn closing_the_focused_window_reconciles_the_mru_successor() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        assert_eq!(state.focused_id(), Some(b));

        state.request_close(b);
        assert_eq!(
            state.focused_id(),
            Some(a),
            "the successor is focused, not left dangling"
        );
        assert!(state.window_manager.get(a).unwrap().focused);
    }

    /// New-2/New-3: with no successor left there is nothing to activate, and
    /// the model's focus pointer ends up cleared rather than pointing at the
    /// destroyed window.
    #[test]
    fn closing_the_last_window_leaves_nothing_focused_and_clears_the_seat() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let id = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.request_close(id);
        assert_eq!(state.focused_id(), None);
    }

    /// New-1: a focus change reports the window that *lost* focus as
    /// unfocused in the model, which is the state `sync_focus_change` then
    /// pushes to that window's client as `Activated = false`.
    #[test]
    fn click_to_focus_unfocuses_the_previous_window() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);
        assert!(state.window_manager.get(b).unwrap().focused);

        state.handle_pointer(PointerEvent::Press {
            id: a,
            pointer: (5, 5),
        });
        assert!(state.window_manager.get(a).unwrap().focused);
        assert!(
            !state.window_manager.get(b).unwrap().focused,
            "the old focus is dropped"
        );
    }

    /// New-3: switching to an empty workspace leaves nothing focused, and
    /// `sync_scene`'s trailing `sync_seat_focus` runs even though the
    /// per-window loop has nothing visible to say about it.
    #[test]
    fn switching_to_an_empty_workspace_clears_focus_and_the_seat() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let id = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        assert_eq!(state.focused_id(), Some(id));

        state.switch_workspace(1).unwrap();
        assert_eq!(state.focused_id(), None, "workspace 2 is empty");

        // And switching back re-focuses the workspace's MRU head.
        state.switch_workspace(0).unwrap();
        assert_eq!(state.focused_id(), Some(id));
    }

    /// New-3: minimizing the focused window hands focus on when there is a
    /// successor, and leaves nothing focused when there isn't.
    #[test]
    fn minimizing_the_focused_window_moves_focus_on() {
        let (mut state, _rx) = state_with_output(1000, 800);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 2, DEFAULT_GEO);

        state.apply_decoration_action(b, crate::decoration::DecorationAction::Minimize);
        assert!(state.window_manager.get(b).unwrap().minimized);
        assert_eq!(state.focused_id(), Some(a));

        state.apply_decoration_action(a, crate::decoration::DecorationAction::Minimize);
        // Task 6: with no successor, `refocus_after_hide` clears the focus
        // pointer outright rather than leaving it on the now-hidden `a` --
        // the model itself must never report a hidden window as focused, not
        // just have the seat mask it via `sync_seat_focus`'s `is_visible_id`
        // filter.
        assert_eq!(state.focused_id(), None);
        assert!(!state.window_manager.is_visible_id(a));
    }

    /// Task 6: `DbCommand::Minimize` (`set_minimized_and_reconcile`) must
    /// resolve the minimized window's *own* workspace, not whichever one is
    /// currently active -- `focused_id()` only ever reflects the active
    /// workspace, so minimizing a focused window on an inactive one used to
    /// skip the refocus branch entirely and could leave that workspace's
    /// pointer stale once something did populate it.
    #[test]
    fn minimizing_the_focused_window_on_an_inactive_workspace_leaves_no_stale_focus() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // Make `a` workspace 2's *genuinely* focused window via the real
        // focus path (`add_window` focuses on the currently-active
        // workspace), not a raw `set_workspace` -- that unconditionally
        // clears `.focused` and never sets the destination's pointer, so a
        // test built on it would pass even with the old, unfixed guard.
        state.window_manager.set_active_workspace(2);
        let a = state.window_manager.add_window(
            "a",
            "a",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
        );
        assert_eq!(state.window_manager.focused_window().map(|w| w.id), Some(a));
        assert!(state.window_manager.get(a).unwrap().focused);
        // Switch away to ws 1 (now inactive workspace 2 still points at `a`),
        // minimize `a`, then switch back.
        state.window_manager.set_active_workspace(1);
        state.set_minimized_and_reconcile(a, true);
        state.window_manager.set_active_workspace(2);
        assert!(
            state.window_manager.focused_window().is_none()
                || state
                    .window_manager
                    .focused_window()
                    .map(|w| !w.minimized)
                    .unwrap_or(true),
            "a minimized window must not remain the workspace's focus"
        );
        assert!(
            !state.window_manager.get(a).unwrap().focused,
            "the hidden window's own focused flag must clear too"
        );
    }

    /// New-4: the inset the sync path applies is the shared
    /// `decoration::content_rect`, keyed off the same `has_ssd` predicate the
    /// renderer uses -- so a decorated window's client is configured a title
    /// bar shorter and a title bar lower, and a CSD one is left alone.
    /// (`sync_window_to_scene` itself can't be observed without a real
    /// toplevel; this pins the geometry contract it consumes.)
    #[test]
    fn ssd_inset_applies_to_decorated_windows_only() {
        use crate::decoration::{TITLE_BAR_HEIGHT, content_rect, has_ssd};
        let (mut state, _rx) = state_with_output(1000, 800);
        let geo = Rectangle {
            x: 10,
            y: 20,
            width: 400,
            height: 300,
        };
        let ssd = state
            .window_manager
            .add_window("org.example.Ssd", "t", 1, geo);
        let csd = state.window_manager.add_window("org.gtk.Csd", "t", 2, geo);

        let w = state.window_manager.get(ssd).unwrap();
        assert!(has_ssd(
            &w.app_id,
            w.client_decorations_requested,
            w.fullscreen
        ));
        assert_eq!(
            content_rect(w.geometry, true),
            Rectangle {
                x: 10,
                y: 20 + TITLE_BAR_HEIGHT,
                width: 400,
                height: 300 - TITLE_BAR_HEIGHT
            }
        );

        let w = state.window_manager.get(csd).unwrap();
        assert!(!has_ssd(
            &w.app_id,
            w.client_decorations_requested,
            w.fullscreen
        ));
        assert_eq!(
            content_rect(w.geometry, false),
            geo,
            "CSD windows are untouched"
        );

        // Fullscreen drops the strip, so the client gets the whole output.
        state.set_fullscreen_target(ssd, true).unwrap();
        let w = state.window_manager.get(ssd).unwrap();
        assert!(!has_ssd(
            &w.app_id,
            w.client_decorations_requested,
            w.fullscreen
        ));
        assert_eq!(content_rect(w.geometry, false), w.geometry);
    }

    /// Direct `SeatHandler::key` coverage, previously impossible (the M1
    /// gap): `wlr::KeyEvent::for_test` builds a synthetic press with no live
    /// keyboard behind it.
    ///
    /// `wlr::Modifiers` has no public constructor with flags set (only the
    /// `logo()`/`ctrl()`/`alt()`/`shift()` accessors), so this can't drive
    /// the default `quit` binding (super+shift+q) directly -- the fallback
    /// the task 3 brief calls for is a modifier-free binding inserted into
    /// the config before `State::new`, exercised with `Modifiers::default()`.
    /// `apply_action` dispatches on the keybindings map's own key as the
    /// action name (see its `match base` arms), so the override keeps the
    /// name `"quit"` and only replaces the combo -- a differently-named
    /// entry would never reach the `"quit"` arm at all.
    #[test]
    fn seat_key_matches_a_binding_and_consumes_the_event() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut config = default_config();
        config.keybindings.insert(
            "quit".into(),
            icedtea_config::KeyCombo {
                modifiers: vec![],
                key: "KEY_F12".into(),
            },
        );
        let mut state = State::new(config, tx);
        let keysym = crate::input::key_name_to_keysym("KEY_F12");
        let ev = wlr::KeyEvent::for_test(keysym, wlr::Modifiers::default(), true, 1);

        let consumed = wlr::SeatHandler::key(&mut state, &ev);

        assert!(consumed, "a bound combo must be consumed, not forwarded");
        assert!(state.quitting, "the quit action must have run");
    }

    #[test]
    fn seat_key_without_a_binding_is_forwarded() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let ev = wlr::KeyEvent::for_test(0x61 /* 'a' */, wlr::Modifiers::default(), true, 1);
        assert!(!wlr::SeatHandler::key(&mut state, &ev));
    }

    #[test]
    fn wallpaper_nodes_track_outputs() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([9, 9, 9, 255]));
        state.wallpaper.set_decoded(Some(img));
        state.sync_wallpaper_nodes();
        // No runtime: node creation returns None, so the map stays empty --
        // the assertion here is that the call is a clean no-op without a
        // runtime.
        assert!(state.wallpaper_nodes.is_empty());
    }

    /// Task 10: `ToplevelHandler::request_maximize`'s model half --
    /// `reconcile_maximized` actually mutates the model, and reports whether
    /// it did so the dispatch layer knows when a bare configure is enough.
    #[test]
    fn a_client_maximize_request_reaches_the_model() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 10,
                y: 10,
                width: 200,
                height: 100,
            },
        );
        let key = crate::wayland::ToplevelKey::for_test(1);
        state.wayland.bind(id, key);
        assert!(state.reconcile_maximized(key, true), "model must change");
        assert!(state.window_manager.get(id).expect("window").maximized);
        assert!(
            !state.reconcile_maximized(key, true),
            "idempotent request must report no change"
        );
    }

    /// Task 10 (Deviation 8): a move request that arrives with no button
    /// held must not start a grab -- the crate deliberately forwards no
    /// seat/serial with `request_move`, so the compositor enforces its own
    /// pointer-pressed policy instead of trusting the client's claim.
    #[test]
    fn a_move_request_without_a_pressed_pointer_is_ignored() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        );
        state.pointer_pressed = false;
        state.begin_client_move(id);
        assert!(
            state.drag.window_id().is_none(),
            "no grab without a pressed pointer"
        );
    }

    /// Review fix (task 10): `pointer_pressed` is a single scalar tracking
    /// "the left button is held", not a per-button set -- every button's
    /// `pressed` value used to overwrite it unconditionally, so a
    /// right-button press or release interleaved with a held left button
    /// spuriously set or cleared the flag `begin_client_move`/
    /// `begin_client_resize` gate on. Gating the assignment on `BTN_LEFT`
    /// fixes it: a right press/release while the left button stays down
    /// must leave `pointer_pressed` (and so a move grab) untouched.
    #[test]
    fn a_right_click_does_not_disturb_a_held_left_button() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        const BTN_LEFT: u32 = 0x110;
        const BTN_RIGHT: u32 = 0x111;

        wlr::SeatHandler::pointer_button(&mut state, 0.0, 0.0, BTN_LEFT, true, 0);
        assert!(
            state.pointer_pressed,
            "a left press must set pointer_pressed"
        );

        wlr::SeatHandler::pointer_button(&mut state, 0.0, 0.0, BTN_RIGHT, true, 0);
        assert!(
            state.pointer_pressed,
            "a right press must not clear a held left button"
        );

        wlr::SeatHandler::pointer_button(&mut state, 0.0, 0.0, BTN_RIGHT, false, 0);
        assert!(
            state.pointer_pressed,
            "a right release must not clear a held left button"
        );
    }

    /// Counterpart: a lone right-click (no left button ever pressed) must
    /// never set `pointer_pressed`, so a `request_move` that happens to
    /// arrive during it is still ignored.
    #[test]
    fn a_lone_right_click_leaves_pointer_pressed_false() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        );
        const BTN_RIGHT: u32 = 0x111;

        wlr::SeatHandler::pointer_button(&mut state, 0.0, 0.0, BTN_RIGHT, true, 0);
        assert!(
            !state.pointer_pressed,
            "a lone right press must not set pointer_pressed"
        );

        state.begin_client_move(id);
        assert!(
            state.drag.window_id().is_none(),
            "request_move must be ignored without a held left button"
        );
    }

    // --- Task 13: full server-side decorations ---

    /// A left press on the rightmost `BUTTON_WIDTH` of a title bar is a
    /// close, all the way through the library's own seat entry point --
    /// `hit_test` maps it to `DecorationAction::Close`, which takes
    /// `request_close`'s path like every other close in this file.
    ///
    /// Deliberately *not* bound to a toplevel: `Wayland::close` reports
    /// `true` for a bound window (wait for the client's destroy) and `false`
    /// for an unbacked one (remove the row now), and only the second is
    /// observable without a live client, so this asserts the row is gone.
    #[test]
    fn a_titlebar_close_click_routes_to_request_close() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "plain.app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 100,
                width: 300,
                height: 200,
            },
        );
        let frame = state.window_manager.get(id).expect("w").geometry;
        let bar = crate::decoration::title_bar_rect(frame);
        // The close button is the rightmost of the three (`button_rects`
        // orders them minimize, maximize, close).
        let click = (bar.x + bar.width - 5, bar.y + 5);

        wlr::SeatHandler::pointer_button(
            &mut state,
            click.0 as f64,
            click.1 as f64,
            0x110,
            true,
            1,
        );

        assert!(
            state.window_manager.get(id).is_none(),
            "close button must close"
        );
        assert!(
            rx.try_iter()
                .any(|e| matches!(e.event, icedtea_contract::Event::WindowClosed { .. })),
            "the close must have been announced"
        );
    }

    /// The middle button is maximize and the leftmost is minimize -- the
    /// same three model command paths D-Bus drives, reached by pointer.
    #[test]
    fn titlebar_buttons_hit_the_models_own_command_paths() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "plain.app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 100,
                width: 300,
                height: 200,
            },
        );
        let frame = state.window_manager.get(id).expect("w").geometry;
        let rects = crate::decoration::button_rects(frame);

        let maximize = (rects[1].x + 5, rects[1].y + 5);
        wlr::SeatHandler::pointer_button(
            &mut state,
            maximize.0 as f64,
            maximize.1 as f64,
            0x110,
            true,
            1,
        );
        assert!(
            state.window_manager.get(id).expect("w").maximized,
            "middle button maximizes"
        );

        let minimize = (rects[0].x + 5, rects[0].y + 5);
        // The window is maximized now, so its frame moved; re-derive.
        let frame = state.window_manager.get(id).expect("w").geometry;
        let minimize = if crate::decoration::button_rects(frame)[0].contains(minimize.0, minimize.1)
        {
            minimize
        } else {
            let r = crate::decoration::button_rects(frame)[0];
            (r.x + 5, r.y + 5)
        };
        wlr::SeatHandler::pointer_button(
            &mut state,
            minimize.0 as f64,
            minimize.1 as f64,
            0x110,
            true,
            1,
        );
        assert!(
            state.window_manager.get(id).expect("w").minimized,
            "left button minimizes"
        );
    }

    /// The negotiation's model half: a client asking to draw its own
    /// decorations is recorded as such, which is what `has_ssd` then reads.
    #[test]
    fn decoration_mode_request_updates_the_model_preference() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        );
        let key = crate::wayland::ToplevelKey::for_test(1);
        state.wayland.bind(id, key);

        wlr::ToplevelHandler::request_decoration_mode(
            &mut state,
            key.0,
            Some(wlr::DecorationMode::ClientSide),
        );
        assert_eq!(
            state
                .window_manager
                .get(id)
                .expect("w")
                .client_decorations_requested,
            Some(true)
        );

        wlr::ToplevelHandler::request_decoration_mode(
            &mut state,
            key.0,
            Some(wlr::DecorationMode::ServerSide),
        );
        assert_eq!(
            state
                .window_manager
                .get(id)
                .expect("w")
                .client_decorations_requested,
            Some(false)
        );

        wlr::ToplevelHandler::request_decoration_mode(&mut state, key.0, None);
        assert_eq!(
            state
                .window_manager
                .get(id)
                .expect("w")
                .client_decorations_requested,
            None
        );
    }

    /// The ordering that actually happens on the wire: the client states its
    /// preference before its initial commit, so there is no model window to
    /// record it against yet. It must not be lost -- `new_toplevel` collects
    /// it when the window is finally mapped.
    #[test]
    fn a_preference_stated_before_mapping_survives_until_the_window_exists() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let key = crate::wayland::ToplevelKey::for_test(9);

        wlr::ToplevelHandler::request_decoration_mode(
            &mut state,
            key.0,
            Some(wlr::DecorationMode::ClientSide),
        );
        assert!(
            state.window_manager.windows().next().is_none(),
            "nothing is mapped yet"
        );

        state.new_toplevel(key, "plain.app", "t", 1);
        let id = state.wayland.window_for(key).expect("bound at map");
        assert_eq!(
            state
                .window_manager
                .get(id)
                .expect("w")
                .client_decorations_requested,
            Some(true),
            "the pre-map preference must reach the model"
        );
        assert!(
            !crate::decoration::has_ssd("plain.app", Some(true), false),
            "and must suppress the band"
        );
    }

    /// A toplevel that dies before mapping takes its parked preference with
    /// it, rather than leaving an entry nothing will ever collect.
    #[test]
    fn a_preference_parked_for_a_toplevel_that_never_maps_is_dropped() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let key = crate::wayland::ToplevelKey::for_test(11);

        wlr::ToplevelHandler::request_decoration_mode(
            &mut state,
            key.0,
            Some(wlr::DecorationMode::ClientSide),
        );
        assert_eq!(state.pending_decorations.len(), 1);
        state.forget_toplevel(key);
        assert!(
            state.pending_decorations.is_empty(),
            "an unmapped toplevel's preference must not leak"
        );
    }

    /// The title raster is memoized on `(title, width, resolved color)`: the
    /// same window synced twice shapes once and keeps one generation, and a
    /// retitle both re-shapes and advances the generation (which is what
    /// makes the seam re-upload).
    #[test]
    fn title_rasters_are_memoized_and_invalidated_by_a_retitle() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "plain.app",
            "first",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 400,
                height: 300,
            },
        );
        let bar =
            crate::decoration::title_bar_rect(state.window_manager.get(id).expect("w").geometry);
        let entry = |state: &State| {
            let e = state.title_rasters.get(&id).expect("cached");
            (e.generation, e.pixels.clone())
        };

        state.ensure_title_raster(id, "first".into(), bar, true);
        assert_eq!(state.title_rasters.len(), 1, "the raster must be cached");
        let first = entry(&state);

        state.ensure_title_raster(id, "first".into(), bar, true);
        assert_eq!(
            entry(&state),
            first,
            "an unchanged title must not re-shape or re-generation"
        );

        state.ensure_title_raster(id, "second".into(), bar, true);
        let renamed = entry(&state);
        assert_eq!(
            state.title_rasters.len(),
            1,
            "one entry per window, replaced not appended"
        );
        assert_ne!(
            renamed.1, first.1,
            "a retitle must produce different pixels"
        );
        assert!(renamed.0 > first.0, "a retitle must advance the generation");

        // Losing focus changes the resolved color, so it invalidates too.
        state.ensure_title_raster(id, "second".into(), bar, false);
        let unfocused = entry(&state);
        assert!(unfocused.0 > renamed.0, "a focus change must re-shape");
        assert_ne!(unfocused.1, renamed.1, "unfocused text is dimmer");

        state.forget_window(id);
        assert!(
            state.title_rasters.is_empty(),
            "a closed window's raster must not outlive it"
        );
    }

    /// M3: the cache key carries the resolved foreground color, so a palette
    /// change invalidates a raster even for an otherwise identical title on
    /// an otherwise identical window.
    #[test]
    fn a_palette_change_invalidates_a_cached_title_raster() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "plain.app",
            "same",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 400,
                height: 300,
            },
        );
        let bar =
            crate::decoration::title_bar_rect(state.window_manager.get(id).expect("w").geometry);

        state.ensure_title_raster(id, "same".into(), bar, true);
        let before = state.title_rasters.get(&id).expect("cached").pixels.clone();

        // Repaint the palette directly (rather than through `apply_config`,
        // which now preserves windows and would need a full appearance
        // resync to reach here) to isolate the raster cache's color key.
        state.config.appearance.palette.foreground = "#ff0000".into();
        state.ensure_title_raster(id, "same".into(), bar, true);
        let after = state.title_rasters.get(&id).expect("cached").pixels.clone();

        assert_ne!(
            before, after,
            "a repainted palette must not serve stale-colored text"
        );
    }

    /// M3 preserve contract: a reload keeps every window row, so it must
    /// also keep those windows' cached title rasters -- purging them would
    /// force a needless re-shape of text that has not changed. (Palette
    /// changes are still invalidated by the raster's own color-keyed cache;
    /// see `a_palette_change_invalidates_a_cached_title_raster`.)
    #[test]
    fn a_config_reload_preserves_cached_rasters_for_surviving_windows() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "plain.app",
            "survivor",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 400,
                height: 300,
            },
        );
        let bar =
            crate::decoration::title_bar_rect(state.window_manager.get(id).expect("w").geometry);
        state.ensure_title_raster(id, "survivor".into(), bar, true);
        assert_eq!(state.title_rasters.len(), 1);

        let _ = state.apply_config(default_config());
        assert!(
            state.window_manager.get(id).is_some(),
            "the window survives the reload"
        );
        assert!(
            state.title_rasters.contains_key(&id),
            "its cached title raster survives too"
        );
    }

    /// The title node is inset by the three buttons, so a long title runs
    /// out of room before it runs underneath the close button.
    #[test]
    fn the_title_raster_is_narrower_than_the_band_by_the_button_span() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let id = state.window_manager.add_window(
            "plain.app",
            "a very long window title that would otherwise run under the buttons",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 400,
                height: 300,
            },
        );
        let bar =
            crate::decoration::title_bar_rect(state.window_manager.get(id).expect("w").geometry);
        let title = state.window_manager.get(id).expect("w").title.clone();

        state.ensure_title_raster(id, title, bar, true);
        let (w, h, px) = state
            .title_rasters
            .get(&id)
            .expect("cached")
            .pixels
            .clone()
            .expect("pixels");
        assert_eq!(w, bar.width - 3 * crate::decoration::BUTTON_WIDTH);
        assert_eq!(h, bar.height);
        assert_eq!(px.len(), (w * h * 4) as usize);
    }

    /// Button colors are premultiplied (every channel no greater than the
    /// alpha), which is what the wlroots scene graph composites.
    #[test]
    fn button_colors_are_premultiplied() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let state = State::new(default_config(), tx);
        for color in state.button_colors() {
            for channel in &color[..3] {
                assert!(
                    *channel <= color[3] + f32::EPSILON,
                    "{color:?} is not premultiplied"
                );
            }
        }
    }

    // --- Task 14: snap preview rect, model-level unmapped, reload gap-close ---

    /// Without a runtime attached, `sync_snap_preview` is a seam no-op both
    /// ways -- but the `Option` field must still mirror `snap_preview`
    /// exactly (no stale id kept). The runtime-backed version of this test
    /// lives in `tests/headless_boot.rs`.
    #[test]
    fn snap_preview_rect_bookkeeping_follows_the_preview() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.snap_preview = Some(Rectangle {
            x: 0,
            y: 0,
            width: 400,
            height: 600,
        });
        state.sync_snap_preview();
        state.snap_preview = None;
        state.sync_snap_preview();
        assert!(state.snap_preview_rect.is_none());
    }

    /// Task 3 (wlr-port M3): the snap-preview rect is model-only overlay
    /// chrome, never a hit target. `window_at_point` consults only
    /// `window_manager`'s windows, so a click inside both an active preview
    /// and the window it overlaps must still resolve to the window -- this
    /// pins the model contract the `Band::Overlay` move (see
    /// `sync_snap_preview`) is meant to preserve at the wlr layer too.
    #[test]
    fn a_click_under_an_active_snap_preview_hits_the_window() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 100,
                width: 300,
                height: 200,
            },
        );
        state.snap_preview = Some(Rectangle {
            x: 0,
            y: 0,
            width: 400,
            height: 600,
        });
        state.sync_snap_preview();
        // The window at a point inside both the preview and the window must
        // still resolve to the window -- the preview rect is not a hit
        // target.
        assert_eq!(state.window_at_point((150, 150)), Some(id));
    }

    /// Task 14 Step 2: `ToplevelHandler::unmapped` now moves focus to the
    /// next mapped candidate on the same workspace when it unmaps the
    /// focused window, the same shape `set_minimized_and_reconcile` already
    /// gives minimizing the focused window.
    #[test]
    fn unmapping_the_focused_window_moves_focus_to_the_next_candidate() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "b", 1, DEFAULT_GEO);
        let key_b = crate::wayland::ToplevelKey::for_test(2);
        state.wayland.bind(b, key_b);
        assert_eq!(state.window_manager.focused_window().map(|w| w.id), Some(b));

        wlr::ToplevelHandler::unmapped(&mut state, key_b.0);

        assert_eq!(state.window_manager.get(b).map(|w| w.mapped), Some(false));
        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(a),
            "focus must fall to the next mapped candidate"
        );
    }

    /// Step 2's focus-refusal decision, exercised through the D-Bus surface:
    /// `Focus` on an unmapped window's id must refuse, not remap-safely
    /// no-op.
    #[test]
    fn handle_command_focus_refuses_an_unmapped_window() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let a = state.window_manager.add_window("a", "a", 1, DEFAULT_GEO);
        state.window_manager.set_mapped(a, false).unwrap();
        assert_eq!(
            state.handle_command(crate::dbus::DbCommand::Focus(a)),
            None,
            "an unmapped window must never be focused via D-Bus either"
        );
    }

    /// Baseline DBus surface audit (Step 3): `GetState` had no direct
    /// `handle_command` test anywhere in the workspace -- the round trip is
    /// via `reply_tx`, not the model, so nothing else in this file happened
    /// to cover it.
    #[test]
    fn handle_command_get_state_replies_with_a_snapshot() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        assert_eq!(
            state.handle_command(crate::dbus::DbCommand::GetState(reply_tx)),
            Some(())
        );
        let snapshot = reply_rx
            .try_recv()
            .expect("GetState must reply synchronously");
        assert_eq!(snapshot.windows.len(), 1);
    }

    /// Baseline DBus surface audit (Step 3): `Quit` likewise had no direct
    /// `handle_command` test -- every existing coverage went through
    /// `apply_action("quit")` or sent the command on a channel a live loop
    /// drains, never `handle_command` itself.
    #[test]
    fn handle_command_quit_sets_the_quitting_flag() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert_eq!(state.handle_command(crate::dbus::DbCommand::Quit), Some(()));
        assert!(state.quitting);
    }

    /// Reload audit (Step 3): a config with a new background color lands
    /// through the reload path and `state.config` reflects it. This baseline
    /// already worked before this task (`apply_config` swaps `self.config`
    /// wholesale) -- documented here rather than left unasserted.
    #[test]
    fn reload_applies_appearance_to_live_state() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let mut cfg = default_config();
        cfg.appearance.palette.background = "#ff0000".into();
        state.apply_reloaded_config(cfg);
        assert_eq!(state.config.appearance.palette.background, "#ff0000");
    }

    /// Reload audit (Step 3): a reloaded config with an unparseable key
    /// takes the same warn-and-skip path load-time validation uses
    /// (`input::warn_about_keybindings`, already called from `apply_config`)
    /// rather than panicking or poisoning the rest of the map.
    #[test]
    fn reload_revalidates_keybindings() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let mut cfg = default_config();
        cfg.keybindings.insert(
            "broken".into(),
            icedtea_config::KeyCombo {
                modifiers: vec![],
                key: "NoSuchKeysym".into(),
            },
        );
        state.apply_reloaded_config(cfg);
        // Surviving proof: the compositor did not panic and a real binding
        // still resolves to a real keysym.
        let combo = &state.config.keybindings["quit"];
        assert!(crate::input::key_name_to_keysym(&combo.key) != 0);
    }

    /// Reload gap-close (Step 3): an `appearance.wallpaper` path change
    /// clears the decoded image so no stale pixels survive under the new
    /// path -- the fresh decode is spawned but this test doesn't wait on it
    /// (the point under test is that the *old* image is gone immediately,
    /// not that the new one has landed yet).
    #[test]
    fn reload_swaps_the_wallpaper_when_the_path_changed() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state
            .wallpaper
            .set_decoded(Some(image::RgbaImage::from_pixel(
                1,
                1,
                image::Rgba([0, 0, 0, 255]),
            )));
        let mut cfg = default_config();
        cfg.appearance.wallpaper = Some("/nonexistent/new.png".into());
        state.apply_reloaded_config(cfg);
        assert!(
            state.wallpaper.decoded().is_none(),
            "stale wallpaper must not survive a path change"
        );
    }

    /// The wallpaper path staying the same across a reload must not tear
    /// down and respawn the decode worker for no reason.
    #[test]
    fn reload_leaves_the_wallpaper_alone_when_the_path_is_unchanged() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let image = image::RgbaImage::from_pixel(1, 1, image::Rgba([9, 9, 9, 255]));
        state.wallpaper.set_decoded(Some(image.clone()));
        let cfg = default_config(); // same (absent) wallpaper as boot
        state.apply_reloaded_config(cfg);
        assert_eq!(
            state.wallpaper.decoded(),
            Some(&image),
            "an unrelated reload must not clear the wallpaper"
        );
    }

    // --- Task 17: multi-output placement and hotplug migration ---

    #[test]
    fn placement_targets_the_output_under_the_pointer() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        // Without a runtime, pointer_position is unavailable: output_for_pointer
        // must fall back to the lowest index deterministically.
        assert_eq!(state.output_for_pointer(), Some(0));
    }

    #[test]
    fn windows_migrate_off_a_removed_output() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 900,
                y: 50,
                width: 300,
                height: 200,
            },
        );
        let dead = state.outputs.remove(&1).expect("output 1").geometry;
        state.migrate_windows_from(dead);
        let w = state.window_manager.get(id).expect("window survives");
        let survivor = state.outputs[&0].geometry;
        assert!(
            w.geometry.x >= survivor.x
                && w.geometry.x + w.geometry.width <= survivor.x + survivor.width,
            "window must land inside the surviving output, got {:?}",
            w.geometry
        );
    }

    /// Task 8, F: a window too wide for the survivor used to pin flush to
    /// the survivor's origin on that axis (`new_x = survivor.x`), bleeding
    /// all of the overflow off the right/bottom edge. Centering spreads it
    /// symmetrically instead -- negative on both edges rather than zero on
    /// one and everything on the other.
    #[test]
    fn an_oversized_migrated_window_is_centered_not_corner_pinned() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 900,
                y: 50,
                width: 1000,
                height: 800,
            },
        );
        let dead = state
            .outputs
            .remove(&1)
            .map(|o| o.geometry)
            .unwrap_or(Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            });
        state.migrate_windows_from(dead);
        let w = state.window_manager.get(id).expect("window");
        // centered: x = 0 + (800 - 1000)/2 = -100 (symmetric overflow), not pinned to 0.
        assert_eq!(w.geometry.x, (800 - 1000) / 2);
    }

    #[test]
    fn migration_with_no_surviving_output_keeps_geometry() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 10,
                y: 10,
                width: 300,
                height: 200,
            },
        );
        let dead = state.outputs.remove(&0).expect("output 0").geometry;
        state.migrate_windows_from(dead); // zero outputs left: must not panic, must not move
        assert_eq!(state.window_manager.get(id).expect("w").geometry.x, 10);
    }

    /// Review fix: `new_toplevel`'s cascade placement used to read
    /// `outputs.values().next()` -- arbitrary `HashMap` iteration order, not
    /// even deterministic -- instead of routing through `output_for_pointer`
    /// like the other three placement consumers. With no runtime attached
    /// (the fallback path), `output_for_pointer` always resolves to the
    /// lowest index, so a two-output state must place a new window inside
    /// output 0's box every time, and the cascade origin itself must come
    /// from that box's own `(x, y)` -- not `(0, 0)` -- so this also pins
    /// down that the placeholder-1920x1080 fallback rect isn't what's
    /// actually feeding `cascade_point_in` here.
    #[test]
    fn new_toplevel_cascades_inside_the_lowest_index_output_without_a_runtime() {
        use crate::wayland::ToplevelKey;

        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 1000,
                y: 2000,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 1800,
                y: 2000,
                width: 800,
                height: 600,
            },
        );

        let key = ToplevelKey::for_test(1);
        state.new_toplevel(key, "app", "t", 1);
        let id = state.wayland.window_for(key).expect("bound");
        let geo = state.window_manager.get(id).expect("model row").geometry;

        let output0 = state.outputs[&0].geometry;
        assert!(
            geo.x >= output0.x
                && geo.x + geo.width <= output0.x + output0.width
                && geo.y >= output0.y
                && geo.y + geo.height <= output0.y + output0.height,
            "the new window must cascade inside output 0's box, got {geo:?}, output 0 is {output0:?}"
        );
        // The cascade origin for the very first window in a workspace is the
        // output box's own top-left corner (`cascade_point_in` with no
        // occupied rects yet) -- pinning that it's `output0`'s `(x, y)`
        // (1000, 2000), not `(0, 0)`, is what actually distinguishes "used
        // the real box" from "used the (0,0)-origin fallback rect."
        assert_eq!(
            (geo.x, geo.y),
            (output0.x, output0.y),
            "cascade origin must be the output box's own (x, y)"
        );
    }

    // --- Task 20: layer-shell arrangement and exclusive zones ---

    #[test]
    fn an_exclusive_top_panel_shrinks_the_usable_area() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    left: true,
                    right: true,
                    bottom: false,
                },
                exclusive: 30,
                size: (800, 30),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.arrange_layers();
        let usable = state.outputs[&0].usable;
        assert_eq!(
            usable,
            Rectangle {
                x: 0,
                y: 30,
                width: 800,
                height: 570
            }
        );
    }

    /// C1/I5's maximize helper (`maximize_applies_output_geometry_and_restores_it`)
    /// proves maximize tracks `geometry` when there is no panel; this proves
    /// it tracks `usable` once one exists, and that fullscreen -- which
    /// covers panels by definition -- keeps ignoring it.
    #[test]
    fn maximize_respects_the_usable_area_but_fullscreen_ignores_it() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        // Snapping/maximize geometry math (`layout::maximized_geometry`)
        // also insets by `appearance.snap_gap`; zero it here so the
        // expected numbers below are the panel's carve alone, not a mix of
        // the carve and an unrelated gap constant.
        state.config.appearance.snap_gap = 0;
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    left: true,
                    right: true,
                    bottom: false,
                },
                exclusive: 30,
                size: (800, 30),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.arrange_layers();
        assert_eq!(
            state.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 30,
                width: 800,
                height: 570
            }
        );

        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 40,
                y: 50,
                width: 300,
                height: 200,
            },
        );

        // Maximize via the D-Bus command path (the established pattern):
        // its geometry must equal `usable`, the panel's zone excluded.
        state
            .handle_command(crate::dbus::DbCommand::Maximize(id, true))
            .unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry,
            state.outputs[&0].usable,
            "maximize must exclude the panel's exclusive zone"
        );

        // Unmaximize, then fullscreen: fullscreen covers panels by
        // definition, so its geometry must equal the *full* output
        // geometry, not `usable`.
        state
            .handle_command(crate::dbus::DbCommand::Maximize(id, false))
            .unwrap();
        state
            .handle_command(crate::dbus::DbCommand::Fullscreen(id, true))
            .unwrap();
        assert_eq!(
            state.window_manager.get(id).unwrap().geometry,
            state.outputs[&0].geometry,
            "fullscreen must ignore the panel's exclusive zone"
        );
    }

    // --- Task 20 review: J1, J2, J3, M4, M5 ---

    fn top_panel_entry(exclusive: i32, mapped: bool) -> LayerEntry {
        LayerEntry {
            output: 0,
            sequence: 0,
            layer: wlr::Layer::Top,
            anchor: wlr::Anchor {
                top: true,
                left: true,
                right: true,
                bottom: false,
            },
            exclusive,
            size: (800, exclusive as u32),
            interactive: false,
            mapped,
            last_configured: None,
            margin: (0, 0, 0, 0),
        }
    }

    /// J2: an unmapped entry must not reserve; mapping starts reserving;
    /// unmapping gives the space back. Previously `arrange_layers`' fold had
    /// no mapped check at all, so an unmapped-but-not-yet-destroyed panel
    /// left a permanent hole in the workspace.
    #[test]
    fn an_unmapped_layer_entry_does_not_reserve_until_mapped() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, false));

        state.arrange_layers();
        assert_eq!(
            state.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600
            },
            "an unmapped entry must not reserve"
        );

        state.layers.get_mut(&id).unwrap().mapped = true;
        state.arrange_layers();
        assert_eq!(
            state.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 30,
                width: 800,
                height: 570
            },
            "mapping must start reserving"
        );

        state.layers.get_mut(&id).unwrap().mapped = false;
        state.arrange_layers();
        assert_eq!(
            state.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600
            },
            "unmapping must give the space back"
        );
    }

    /// M4: a second `arrange_layers` pass with nothing changed must not
    /// re-emit for a maximized window -- `WindowManager::set_geometry`
    /// emits `WindowUpdated` unconditionally, so without the "does the
    /// target actually differ" guard, a panel redrawing at its own frame
    /// rate produced one signal (and one client configure) per maximized
    /// window, per panel frame, with byte-identical geometry.
    #[test]
    fn arrange_layers_does_not_resync_a_maximized_window_when_the_target_is_unchanged() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.config.appearance.snap_gap = 0;
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(panel_id, top_panel_entry(30, true));

        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        // Maximized before the panel's zone is folded into `usable` (that
        // only happens inside `arrange_layers`), so the first arrange
        // below genuinely changes the target and must emit.
        state
            .handle_command(crate::dbus::DbCommand::Maximize(id, true))
            .unwrap();
        while rx.try_recv().is_ok() {}

        state.arrange_layers();
        assert!(
            rx.try_recv().is_ok(),
            "the first arrange changes the target and must emit"
        );
        while rx.try_recv().is_ok() {}

        state.arrange_layers();
        assert!(
            rx.try_recv().is_err(),
            "an unchanged maximize target must not re-emit"
        );
    }

    /// J1: `arrange_layers` re-homes each maximized window through its own
    /// frame center, not a single pointer-derived output -- otherwise an
    /// unrelated panel commit on output 1 could teleport a maximized window
    /// that is actually on output 1 onto output 0's `usable` rect merely
    /// because the pointer (with no attached runtime, `output_for_pointer`'s
    /// own fallback: the lowest index) resolved to output 0.
    #[test]
    fn arrange_layers_resyncs_a_maximized_window_against_its_own_output_not_the_pointers() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.config.appearance.snap_gap = 0;
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        // Maximized on output 1.
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 810,
                y: 10,
                width: 300,
                height: 200,
            },
        );
        state
            .window_manager
            .set_geometry(
                id,
                Rectangle {
                    x: 800,
                    y: 0,
                    width: 800,
                    height: 600,
                },
            )
            .unwrap();
        state.window_manager.set_maximized(id, true).unwrap();

        // A panel maps on output 1 -- with no runtime attached,
        // `output_for_pointer`'s fallback is the *lowest* index, output 0,
        // which is exactly the wrong output for this window.
        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                output: 1,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    left: true,
                    right: true,
                    bottom: false,
                },
                exclusive: 30,
                size: (800, 30),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.arrange_layers();

        let geo = state.window_manager.get(id).unwrap().geometry;
        assert_eq!(
            geo, state.outputs[&1].usable,
            "must track output 1's usable rect, not output 0's"
        );
        assert!(
            geo.x >= 800,
            "must not have teleported onto output 0, got {geo:?}"
        );
    }

    /// M5: a layer surface with no output at all when the sweep runs is a
    /// no-op, correctly (left parked for the next call, not dropped); once
    /// an output exists, the sweep re-homes it.
    #[test]
    fn resolve_orphaned_layers_configures_a_layer_surface_parked_with_no_output() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: NO_OUTPUT,
                ..top_panel_entry(30, false)
            },
        );

        state.resolve_orphaned_layers();
        assert_eq!(
            state.layers[&id].output, NO_OUTPUT,
            "no output exists yet: must stay parked, not vanish"
        );

        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.resolve_orphaned_layers();
        assert_eq!(
            state.layers[&id].output, 0,
            "must be re-homed the moment an output exists"
        );
    }

    /// M5a: an entry orphaned by its output being removed is re-homed onto
    /// a surviving output -- the layer analogue of `migrate_windows_from`.
    #[test]
    fn resolve_orphaned_layers_rehomes_onto_a_surviving_output() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 1,
                ..top_panel_entry(30, true)
            },
        );

        // The caller (`OutputHandler::destroyed`) removes the dead output
        // from `self.outputs` before this runs.
        state.outputs.remove(&1);
        state.resolve_orphaned_layers();
        assert_eq!(
            state.layers[&id].output, 0,
            "must re-home onto the surviving output"
        );
    }

    /// Task 8, M5: each orphaned entry re-homes to whichever surviving
    /// output its own last-known placement's frame center actually sits
    /// over, not a single survivor picked once for the whole batch (the
    /// old `output_for_pointer` heuristic this replaces) -- a panel
    /// already configured onto the far side of a two-output layout must
    /// land on the output whose box it geometrically overlaps, even when
    /// the lowest surviving index is the *other* one.
    #[test]
    fn resolve_orphaned_layers_rehomes_each_entry_by_its_own_frame_center() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        // Orphaned by a third, now-dead output; its own last-configured
        // placement sits squarely inside output 1's box, not output 0's
        // (the lowest index, and what a uniform pointer-derived pick would
        // have chosen instead).
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 2,
                last_configured: Some((200, 30, 850, 0)),
                ..top_panel_entry(30, true)
            },
        );

        state.resolve_orphaned_layers();
        assert_eq!(
            state.layers[&id].output, 1,
            "must land on the output its own frame center overlaps"
        );
    }

    /// M5: with no surviving output at all, the sweep is a deliberate
    /// no-op -- there is nowhere to re-home to, and the entry is left
    /// exactly where it was for the next call (the next `new_output`) to
    /// try again, rather than panicking or being dropped.
    #[test]
    fn resolve_orphaned_layers_is_a_no_op_with_no_surviving_output() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, true));

        state.outputs.remove(&0);
        state.resolve_orphaned_layers();
        assert_eq!(
            state.layers[&id].output, 0,
            "left exactly where it was; no survivor to re-home onto"
        );
    }

    /// J3's guard, exercised directly: `true` only while `layer_focus`
    /// names an entry that is still mapped. `None`, a dangling id, and an
    /// unmapped entry must all fall through to the model's own focus.
    #[test]
    fn layer_holds_keyboard_focus_only_while_the_entry_is_mapped() {
        let mut layers: HashMap<wlr::LayerSurfaceId, LayerEntry> = HashMap::new();
        let id = wlr::LayerSurfaceId::dangling_for_test();
        assert!(
            !layer_holds_keyboard_focus(None, &layers),
            "no layer focus at all"
        );
        assert!(
            !layer_holds_keyboard_focus(Some(id), &layers),
            "layer_focus names an entry that does not exist"
        );

        layers.insert(id, top_panel_entry(30, false));
        assert!(
            !layer_holds_keyboard_focus(Some(id), &layers),
            "the entry exists but is unmapped"
        );

        layers.get_mut(&id).unwrap().mapped = true;
        assert!(
            layer_holds_keyboard_focus(Some(id), &layers),
            "a mapped entry must block the model's own focus"
        );
    }

    /// Finding 6: `resize_edges_from_wlr`'s field-by-field mapping, for
    /// every single edge and every corner (two edges at once) --
    /// `begin_client_resize`'s own success path has no headless way to
    /// exercise (see that method's doc), so this is the pure part of it
    /// that stays directly testable.
    #[test]
    fn resize_edges_from_wlr_maps_every_edge_and_corner() {
        let none = wlr::Edges::default();
        assert_eq!(
            resize_edges_from_wlr(none),
            input::ResizeEdges {
                top: false,
                bottom: false,
                left: false,
                right: false
            }
        );

        let top = wlr::Edges { top: true, ..none };
        assert_eq!(
            resize_edges_from_wlr(top),
            input::ResizeEdges {
                top: true,
                bottom: false,
                left: false,
                right: false
            }
        );

        let bottom = wlr::Edges {
            bottom: true,
            ..none
        };
        assert_eq!(
            resize_edges_from_wlr(bottom),
            input::ResizeEdges {
                top: false,
                bottom: true,
                left: false,
                right: false
            }
        );

        let left = wlr::Edges { left: true, ..none };
        assert_eq!(
            resize_edges_from_wlr(left),
            input::ResizeEdges {
                top: false,
                bottom: false,
                left: true,
                right: false
            }
        );

        let right = wlr::Edges {
            right: true,
            ..none
        };
        assert_eq!(
            resize_edges_from_wlr(right),
            input::ResizeEdges {
                top: false,
                bottom: false,
                left: false,
                right: true
            }
        );

        let top_left = wlr::Edges {
            top: true,
            left: true,
            ..none
        };
        assert_eq!(
            resize_edges_from_wlr(top_left),
            input::ResizeEdges {
                top: true,
                bottom: false,
                left: true,
                right: false
            }
        );

        let top_right = wlr::Edges {
            top: true,
            right: true,
            ..none
        };
        assert_eq!(
            resize_edges_from_wlr(top_right),
            input::ResizeEdges {
                top: true,
                bottom: false,
                left: false,
                right: true
            }
        );

        let bottom_left = wlr::Edges {
            bottom: true,
            left: true,
            ..none
        };
        assert_eq!(
            resize_edges_from_wlr(bottom_left),
            input::ResizeEdges {
                top: false,
                bottom: true,
                left: true,
                right: false
            }
        );

        let bottom_right = wlr::Edges {
            bottom: true,
            right: true,
            ..none
        };
        assert_eq!(
            resize_edges_from_wlr(bottom_right),
            input::ResizeEdges {
                top: false,
                bottom: true,
                left: false,
                right: true
            }
        );
    }

    /// J3's wiring: with an interactive panel holding `layer_focus` and a
    /// maximized window present (so `arrange_layers` has something to
    /// re-sync -- the exact path that used to reassert toplevel focus over
    /// the panel's), `arrange_layers` must leave `layer_focus` alone;
    /// unmapping the panel is the one path that legitimately clears it.
    ///
    /// This cannot observe the seat's *real* keyboard target end-to-end --
    /// there is no attached `wlr::Runtime` in a unit test, and the wlr crate
    /// exposes no way to create a virtual keyboard device for a headless
    /// test harness to bind `wl_seat.get_keyboard` against (confirmed: the
    /// harness's headless backend advertises no keyboard capability at all,
    /// so a real client's `get_keyboard` is a protocol error). This is
    /// `layer_focus`'s bookkeeping wired through the real handler methods,
    /// paired with `layer_holds_keyboard_focus`'s own direct proof of the
    /// guard's logic above.
    #[test]
    fn arrange_layers_leaves_layer_focus_alone_and_unmap_clears_it() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let win_id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        state
            .handle_command(crate::dbus::DbCommand::Maximize(win_id, true))
            .unwrap();

        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        // What `layer_surface_mapped` would have set with a live runtime
        // attached (see that handler's own doc for why it cannot here).
        state.layer_focus = Some(panel_id);

        state.arrange_layers();
        assert_eq!(
            state.layer_focus,
            Some(panel_id),
            "arrange_layers must not clear layer_focus"
        );

        wlr::ToplevelHandler::layer_surface_unmapped(&mut state, panel_id);
        assert_eq!(state.layer_focus, None, "unmapping must clear layer_focus");
        assert!(
            !state.layers[&panel_id].mapped,
            "unmapping must stop the entry from reserving"
        );
    }

    /// Task 8, N9: a layer surface that becomes keyboard-interactive
    /// *after* it already mapped -- an auto-hide launcher's menu opening,
    /// say -- must still take keyboard focus, even though
    /// `layer_surface_mapped`'s own take-focus branch already ran once (at
    /// map, while the surface was still non-interactive) and will not run
    /// again. Driven via `sync_layer_interactive_focus` directly, the
    /// internal bookkeeping `layer_surface_commit` calls after updating
    /// `entry.interactive` -- `layer_surface_commit` itself takes a live
    /// `&wlr::LayerSurface`, which nothing in this harness can fabricate
    /// (same limitation `arrange_layers_leaves_layer_focus_alone_and_unmap_clears_it`
    /// documents on its own).
    #[test]
    fn a_layer_surface_that_becomes_interactive_after_map_takes_focus() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        // Mapped, not (yet) interactive -- what `layer_surface_mapped` left
        // behind for a surface that mapped before ever asking for the
        // keyboard.
        state.layers.insert(id, top_panel_entry(30, true));
        assert_eq!(
            state.layer_focus, None,
            "must not hold focus before the flip"
        );

        // The commit that flips `keyboard_interactive()` to `true`.
        state.layers.get_mut(&id).unwrap().interactive = true;
        state.sync_layer_interactive_focus(id, false, true);

        assert_eq!(
            state.layer_focus,
            Some(id),
            "must take layer focus once it becomes interactive post-map"
        );
    }

    /// Task 8, N9: the release half -- a mapped, focused surface that
    /// flips interactive back to `false` (still mapped) must give the
    /// keyboard back up rather than hold a focus it no longer claims.
    #[test]
    fn a_layer_surface_that_stops_being_interactive_releases_focus() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(id);

        state.layers.get_mut(&id).unwrap().interactive = false;
        state.sync_layer_interactive_focus(id, true, false);

        assert_eq!(
            state.layer_focus, None,
            "must release layer focus once it stops being interactive"
        );
    }

    /// Task 8, N9: a surface that is not mapped yet must never take focus
    /// through this path, even if the flag flips -- `focus_layer_keyboard`
    /// refuses an unmapped surface for good reason (see that method's own
    /// doc), and this guard is what keeps the bookkeeping in step with it.
    #[test]
    fn an_unmapped_surface_does_not_take_focus_on_becoming_interactive() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, false));

        state.layers.get_mut(&id).unwrap().interactive = true;
        state.sync_layer_interactive_focus(id, false, true);

        assert_eq!(
            state.layer_focus, None,
            "an unmapped surface must not take layer focus"
        );
    }

    /// Task 8, N9: something else already holding layer focus must not be
    /// stolen from just because an unrelated surface also flips
    /// interactive on.
    #[test]
    fn an_already_focused_layer_is_not_stolen_from_by_another_turning_interactive() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let held = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            held,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(held);

        let other = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(other, top_panel_entry(30, true));
        state.layers.get_mut(&other).unwrap().interactive = true;
        state.sync_layer_interactive_focus(other, false, true);

        assert_eq!(
            state.layer_focus,
            Some(held),
            "must not steal focus from an already-focused layer surface"
        );
    }

    /// N6: two top-anchored exclusive panels on the same output must stack
    /// rather than both drawing at the box's own top edge -- the earlier
    /// (lower `sequence`) panel's placement base is the raw output box, the
    /// later one's is the box already shrunk by the earlier panel's own
    /// zone, so it is placed *below* the first rather than on top of it.
    #[test]
    fn usable_before_stacks_two_same_edge_panels_instead_of_overlapping() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let first = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            first,
            LayerEntry {
                sequence: 0,
                ..top_panel_entry(30, true)
            },
        );

        // `usable_before` for the first panel (nothing precedes it in
        // sequence order): the raw output box, unshrunk.
        assert_eq!(
            state.usable_before(0, 0).unwrap(),
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600
            },
            "the first panel's placement base must be the raw box"
        );

        // `usable_before` for a second panel (sequence 1, after the
        // first): shrunk by the first panel's own 30px zone -- this is
        // what makes `configure_layer` place it flush below the first
        // instead of drawing over it.
        assert_eq!(
            state.usable_before(0, 1).unwrap(),
            Rectangle {
                x: 0,
                y: 30,
                width: 800,
                height: 570
            },
            "the second panel's placement base must exclude the first panel's zone"
        );

        // An unmapped earlier entry must not shrink a later panel's base
        // either -- mirrors J2's "unmapped reserves nothing" for the
        // per-surface placement path, not just `arrange_layers`' own fold.
        state.layers.get_mut(&first).unwrap().mapped = false;
        assert_eq!(
            state.usable_before(0, 1).unwrap(),
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600
            },
            "an unmapped earlier entry must not shrink a later panel's placement base"
        );
    }

    /// H2: two panels each requesting a pathologically large
    /// `exclusive_zone` (well past the output's own extent -- what a
    /// hostile or buggy client can ask for) must not panic
    /// (`rect.y += exclusive` overflowing `i32` in a debug build) or wrap
    /// into a corrupted rect in release; `usable` must saturate at a sane,
    /// non-negative box instead.
    #[test]
    fn two_large_exclusive_top_panels_do_not_overflow_or_panic() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        let first = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            first,
            LayerEntry {
                sequence: 0,
                exclusive: i32::MAX,
                ..top_panel_entry(i32::MAX, true)
            },
        );
        let second = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            second,
            LayerEntry {
                sequence: 1,
                exclusive: i32::MAX,
                ..top_panel_entry(i32::MAX, true)
            },
        );

        state.arrange_layers();

        let usable = state.outputs[&0].usable;
        assert!(
            usable.height >= 0,
            "height must never go negative, got {usable:?}"
        );
        assert!(
            usable.width >= 0,
            "width must never go negative, got {usable:?}"
        );
        assert!(
            usable.y >= 0,
            "y must stay a sane, saturated value, got {usable:?}"
        );
    }

    /// H2: the exclusive zone is also clamped where it is captured --
    /// `new_layer_surface`/`layer_surface_commit` -- not only where it is
    /// folded, so `LayerEntry::exclusive` itself never holds a
    /// pathological value in the first place. `clamp_exclusive_zone`'s
    /// contract directly, since a client-supplied `wlr::LayerSurface` has
    /// no test constructor for `exclusive_zone()`.
    #[test]
    fn clamp_exclusive_zone_bounds_a_positive_value_to_the_outputs_extent() {
        let output = OutputSurface::new(Rectangle {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        });
        assert_eq!(
            clamp_exclusive_zone(i32::MAX, Some(&output)),
            800,
            "clamped to max(width, height)"
        );
        assert_eq!(
            clamp_exclusive_zone(30, Some(&output)),
            30,
            "a sane value passes through unchanged"
        );
        assert_eq!(
            clamp_exclusive_zone(-5, Some(&output)),
            -5,
            "a non-positive sentinel is never touched"
        );
        assert_eq!(
            clamp_exclusive_zone(i32::MAX, None),
            i32::MAX,
            "no output resolved yet: left to the saturating fold"
        );
    }

    // --- Task 20 re-review: Important-1, Minor-1 ---

    /// Important-1: click-to-focus is an explicit toplevel-focus assertion
    /// and must win over an interactive panel's held `layer_focus`, unlike
    /// the passive resyncs round 1 (correctly) left alone -- see
    /// `release_layer_focus`'s own doc for the two-clause guard this
    /// completes.
    #[test]
    fn clicking_a_window_releases_layer_focus_for_an_interactive_panel() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 100,
                width: 600,
                height: 400,
            },
        );

        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(panel_id);

        let _ = state.handle_pointer(PointerEvent::Press {
            id,
            pointer: (120, 105),
        });
        assert_eq!(
            state.layer_focus, None,
            "clicking a window must release layer_focus"
        );
        assert!(
            state.window_manager.get(id).unwrap().focused,
            "the model's own focus must have won"
        );
    }

    /// Important-1: `DbCommand::Focus` is the D-Bus-driven counterpart of a
    /// click and must release `layer_focus` the same way.
    #[test]
    fn db_command_focus_releases_layer_focus_for_an_interactive_panel() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let a = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        let _b = state.window_manager.add_window(
            "app2",
            "t2",
            2,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );

        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(panel_id);

        state
            .handle_command(crate::dbus::DbCommand::Focus(a))
            .unwrap();
        assert_eq!(
            state.layer_focus, None,
            "DbCommand::Focus must release layer_focus"
        );
        assert!(state.window_manager.get(a).unwrap().focused);
    }

    /// Round-2 re-review finding Important-2: `release_layer_focus()` ran
    /// *before* the fallible `window_manager.focus(id)?` in both
    /// `DbCommand::Focus` and `handle_pointer_press`, so a focus request
    /// naming a window that cannot actually be focused (unmapped, or --
    /// `DbCommand::Focus` is reachable straight from a D-Bus caller -- a
    /// stale wire id) still cleared `layer_focus` on its way to the early
    /// return, silently defeating an interactive panel's keyboard grab for
    /// a focus assertion that never happened. `release_layer_focus` now
    /// runs after the `?` at both sites.
    #[test]
    fn a_failing_db_command_focus_does_not_release_layer_focus() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let a = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        state.window_manager.set_mapped(a, false).unwrap();

        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(panel_id);

        assert_eq!(
            state.handle_command(crate::dbus::DbCommand::Focus(a)),
            None,
            "an unmapped window must never be focused via D-Bus"
        );
        assert_eq!(
            state.layer_focus,
            Some(panel_id),
            "a failing focus request must not release layer_focus"
        );
    }

    /// Important-1: alt-tab's per-step focus is the third explicit path
    /// named in the finding.
    #[test]
    fn alt_tab_cycling_releases_layer_focus_for_an_interactive_panel() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let _a = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        let _b = state.window_manager.add_window(
            "app2",
            "t2",
            2,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );

        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(panel_id);

        state.apply_action("cycle:alt_tab");
        assert_eq!(
            state.layer_focus, None,
            "alt-tab cycling must release layer_focus"
        );
    }

    /// Finding F9: an X11 client's `_NET_ACTIVE_WINDOW`
    /// (`xwayland_request_activate`) now runs the *same* focus-steal policy
    /// as `xdg-activation-v1`, and a token-less request never satisfies it --
    /// so an X11 activate raises an attention hint instead of taking the
    /// keyboard.
    ///
    /// This test used to assert the opposite (that the activate honored the
    /// steal, and therefore released an interactive layer surface's keyboard
    /// grab on its way through). Both halves are inverted here, and the
    /// layer-focus half is what keeps it load-bearing in the new direction:
    /// `release_layer_focus` runs only on the honored branch, so a
    /// regression back to honoring would show up as `layer_focus` being
    /// cleared -- exactly what this now forbids.
    #[test]
    fn xwayland_activate_flags_attention_and_never_steals_the_keyboard() {
        use wlr::ToplevelHandler as _;
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let other = state.window_manager.add_window(
            "other",
            "o",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        let win = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        // `add_window` autofocuses, so `win` would be the already-focused
        // early return. Hand focus back to `other` so the request has a real
        // steal to attempt.
        state
            .window_manager
            .focus(other)
            .expect("other must be focusable");
        let sid = wlr::XwaylandSurfaceId::dangling_for_test();
        state.xwayland_windows.insert(sid, win);

        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            panel_id,
            LayerEntry {
                interactive: true,
                ..top_panel_entry(30, true)
            },
        );
        state.layer_focus = Some(panel_id);

        state.xwayland_request_activate(sid);

        assert_eq!(
            state.focused_id(),
            Some(other),
            "an X11 activate must not move focus; `_NET_ACTIVE_WINDOW` carries no seat serial"
        );
        assert!(
            state
                .window_manager
                .get(win)
                .expect("still in the model")
                .attention,
            "a refused X11 activate must raise the attention hint instead"
        );
        assert_eq!(
            state.layer_focus,
            Some(panel_id),
            "a refused activate performs no focus assertion, so it must leave layer_focus alone"
        );
    }

    /// A `cursor-shape-v1` request from a tablet tool is ignored: a stray
    /// background tablet-tool client must not repaint the shared cursor
    /// image ahead of whatever the pointer is doing. The pointer device is
    /// the control.
    #[test]
    fn a_tablet_tool_may_not_name_the_seat_cursor() {
        assert!(
            State::honors_cursor_shape_device(wlr::CursorShapeDevice::Pointer),
            "control: a pointer device names the cursor"
        );
        assert!(
            !State::honors_cursor_shape_device(wlr::CursorShapeDevice::TabletTool),
            "a tablet tool must not name the shared seat cursor"
        );
    }

    /// Finding F1 (security): `request_activate` must do nothing at all
    /// while the session is locked -- not honor, and not flag. A honored
    /// request would let a background client pick who gets the keyboard the
    /// instant the session unlocks; a hint raised behind the lock screen is
    /// one the user can neither see nor clear until then.
    ///
    /// Round-2 re-review: this test used to drive the handler with
    /// `target: None`, which returns at the `let Some(target) = target else`
    /// arm with or without the lock gate -- so it asserted nothing about the
    /// gate at all. It now redeems a **fully honorable** token (`has_seat`,
    /// requester == the focused window, target mapped and on the active
    /// workspace) against a real, bound toplevel: precisely the request that
    /// DOES move focus when unlocked, which the unlocked control at the
    /// bottom proves. Everything the handler could possibly have done is
    /// asserted against: focus, the attention bit, and the event stream.
    #[test]
    fn activation_requests_are_ignored_while_the_session_is_locked() {
        use wlr::SeatHandler as _;
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let a = state.window_manager.add_window("a", "A", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "B", 2, DEFAULT_GEO);
        // Real toplevel bindings, so `request_activate`'s `window_for`
        // lookups resolve and the handler reaches its actual decision.
        let a_key = crate::wayland::ToplevelKey::for_test(1);
        let b_key = crate::wayland::ToplevelKey::for_test(2);
        state.wayland.bind(a, a_key);
        state.wayland.bind(b, b_key);
        state.window_manager.focus(a).expect("A must be focusable");

        // The token that WOULD be honored: seat-backed, minted by the
        // focused window, aimed at a mapped target on the active workspace.
        let honorable = || wlr::ActivationToken {
            has_seat: true,
            serial: 1,
            requesting_toplevel: Some(wlr::ToplevelId::dangling_nth_for_test(1)),
        };

        state.session_lock_changed(true);
        while rx.try_recv().is_ok() {}

        state.request_activate(Some(wlr::ToplevelId::dangling_nth_for_test(2)), honorable());

        assert_eq!(
            state.focused_id(),
            Some(a),
            "a locked session must not let an activation move the model's focus"
        );
        assert!(
            !state.window_manager.get(b).expect("B in model").attention,
            "the honored branch must not run while locked -- and neither may the refused one"
        );
        assert!(
            rx.try_recv().is_err(),
            "a locked activation must emit nothing at all; the shell must not see it happen"
        );

        // The refused shape too: a token with no seat and no requester would
        // take the attention branch when unlocked, and must not even do that.
        state.request_activate(
            Some(wlr::ToplevelId::dangling_nth_for_test(2)),
            wlr::ActivationToken {
                has_seat: false,
                serial: 0,
                requesting_toplevel: None,
            },
        );
        assert!(
            !state.window_manager.get(b).expect("B in model").attention,
            "a refused activation must not raise a hint the user cannot see or clear while locked"
        );
        assert!(
            rx.try_recv().is_err(),
            "the refused branch must emit nothing while locked either"
        );

        // Control: unlocked, the very same honorable token moves focus. This
        // is what makes every assertion above a claim about the LOCK rather
        // than about the token being unacceptable on its own merits.
        state.session_lock_changed(false);
        state.request_activate(Some(wlr::ToplevelId::dangling_nth_for_test(2)), honorable());
        assert_eq!(
            state.focused_id(),
            Some(b),
            "control: unlocked, this token is honored -- so the assertions above are about the lock"
        );
    }

    /// Finding F1, the Xwayland half: `_NET_ACTIVE_WINDOW` is gated on the
    /// lock for exactly the same reason, and by the same early return.
    ///
    /// The control at the bottom is the attention branch rather than a focus
    /// move, because an X11 activate never steals (finding F9) -- raising the
    /// hint is the whole of what this handler can do, so it is the whole of
    /// what the lock has to suppress.
    #[test]
    fn xwayland_activate_requests_are_ignored_while_the_session_is_locked() {
        use wlr::{SeatHandler as _, ToplevelHandler as _};
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let a = state.window_manager.add_window("a", "A", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "B", 2, DEFAULT_GEO);
        let sid = wlr::XwaylandSurfaceId::dangling_for_test();
        state.xwayland_windows.insert(sid, b);
        state.window_manager.focus(a).expect("A must be focusable");

        state.session_lock_changed(true);
        while rx.try_recv().is_ok() {}

        state.xwayland_request_activate(sid);

        assert_eq!(
            state.focused_id(),
            Some(a),
            "a locked X11 activate must not move focus"
        );
        assert!(
            !state.window_manager.get(b).expect("B in model").attention,
            "a locked X11 activate must not raise an attention hint either"
        );
        assert!(
            rx.try_recv().is_err(),
            "a locked X11 activate must emit nothing at all"
        );

        // Control: unlocked, the same request does reach the refusal branch
        // and flags B -- so the assertions above are about the lock.
        state.session_lock_changed(false);
        state.xwayland_request_activate(sid);
        assert!(
            state.window_manager.get(b).expect("B in model").attention,
            "control: unlocked, an X11 activate flags attention"
        );
    }

    /// Round-2 re-review (F1's security class): click-to-focus is a model
    /// mutation and must not run while the session is locked. Without the
    /// gate, a press routed at a window hidden behind the lock screen moved
    /// the model's focus -- choosing who receives the keyboard on unlock --
    /// and cleared that window's attention hint on the way through.
    #[test]
    fn a_pointer_press_does_not_move_focus_while_the_session_is_locked() {
        use wlr::SeatHandler as _;
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let a = state.window_manager.add_window("a", "A", 1, DEFAULT_GEO);
        let b = state.window_manager.add_window("b", "B", 2, DEFAULT_GEO);
        state.window_manager.focus(a).expect("A must be focusable");
        state.window_manager.set_attention(b, true).expect("flag B");
        // A point inside B's frame, so the press has a real target.
        let inside_b = (DEFAULT_GEO.x + 5, DEFAULT_GEO.y + DEFAULT_GEO.height / 2);

        state.session_lock_changed(true);
        while rx.try_recv().is_ok() {}

        state.handle_pointer(PointerEvent::Press {
            id: b,
            pointer: inside_b,
        });

        assert_eq!(
            state.focused_id(),
            Some(a),
            "a click behind the lock screen must not move focus"
        );
        assert!(
            state.window_manager.get(b).expect("B in model").attention,
            "a click behind the lock screen must not clear an attention hint either"
        );
        assert!(
            rx.try_recv().is_err(),
            "a locked pointer press must emit nothing at all"
        );

        // Control: unlocked, the very same press does focus B and answers
        // its hint -- so the assertions above are about the lock.
        state.session_lock_changed(false);
        state.handle_pointer(PointerEvent::Press {
            id: b,
            pointer: inside_b,
        });
        assert_eq!(
            state.focused_id(),
            Some(b),
            "control: unlocked, the press focuses B"
        );
        assert!(
            !state.window_manager.get(b).expect("B in model").attention,
            "control: unlocked, focusing B answers its attention hint"
        );
    }

    /// Finding F6: an attention hint must never be raised on a target
    /// `WindowManager::focus` can never clear -- and `focus` refuses an
    /// unmapped window outright, so such a hint would stick forever.
    #[test]
    fn attention_is_not_raised_on_an_unmapped_target() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let id = state.window_manager.add_window("app", "t", 1, DEFAULT_GEO);
        state.window_manager.set_mapped(id, false);

        state.raise_attention_if_answerable(id);
        assert!(
            !state
                .window_manager
                .get(id)
                .expect("still in the model")
                .attention,
            "an unmapped target must not be flagged: nothing could ever clear the hint"
        );

        // Control, so this is not passing because flagging is broken outright.
        state.window_manager.set_mapped(id, true);
        state.raise_attention_if_answerable(id);
        assert!(
            state
                .window_manager
                .get(id)
                .expect("still in the model")
                .attention
        );
    }

    /// Finding F6, second half: restoring a minimized window is the user
    /// attending to it, so it answers an attention hint even when the
    /// restore does not move focus. `focus` clears the hint, but a restore
    /// on a *background* window (the taskbar's ordinary un-minimize) never
    /// calls it.
    #[test]
    fn restoring_a_minimized_window_clears_its_attention_hint() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let flagged = state.window_manager.add_window("a", "A", 1, DEFAULT_GEO);
        let keeper = state.window_manager.add_window("b", "B", 2, DEFAULT_GEO);
        state
            .set_minimized_and_reconcile(flagged, true)
            .expect("minimize");
        state
            .window_manager
            .focus(keeper)
            .expect("B takes the keyboard");
        state
            .window_manager
            .set_attention(flagged, true)
            .expect("flagged");
        while rx.try_recv().is_ok() {}

        state
            .set_minimized_and_reconcile(flagged, false)
            .expect("restore");

        assert!(
            !state
                .window_manager
                .get(flagged)
                .expect("still in the model")
                .attention,
            "restoring a flagged window must clear its attention hint"
        );
        assert_eq!(
            state.focused_id(),
            Some(keeper),
            "the restore must not have moved focus"
        );
        assert!(
            rx.try_iter().any(|e| matches!(
                e.event,
                Event::WindowUpdated { id, ref update } if id == flagged && update.attention == Some(false)
            )),
            "the clear must reach the shell as an event, not just the model"
        );
    }

    /// Finding F14: switching to a workspace whose focused window is flagged
    /// clears the hint. That window already holds its workspace's focus
    /// pointer, so the switch never calls `focus` on it -- which is what let
    /// exactly the case `focus`'s already-focused branch exists for keep its
    /// hint after the user came and looked right at it.
    #[test]
    fn switching_to_a_workspace_clears_its_focused_window_s_attention() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let here = state.window_manager.add_window("here", "H", 1, DEFAULT_GEO);
        let there = state
            .window_manager
            .add_window("there", "T", 2, DEFAULT_GEO);
        state
            .window_manager
            .set_workspace(there, 1)
            .expect("move to workspace 1");
        state
            .window_manager
            .focus(there)
            .expect("focus it on its own workspace");
        state.switch_workspace(0).expect("back to workspace 0");
        state
            .window_manager
            .focus(here)
            .expect("H holds workspace 0");
        state
            .window_manager
            .set_attention(there, true)
            .expect("flag the off-workspace window");

        state.switch_workspace(1).expect("switch to workspace 1");

        assert_eq!(
            state.focused_id(),
            Some(there),
            "T must be workspace 1's focused window"
        );
        assert!(
            !state
                .window_manager
                .get(there)
                .expect("still in the model")
                .attention,
            "arriving on the workspace answers the hint on its focused window"
        );
    }

    /// Minor-1: `arrange_layers` must reconfigure every mapped panel on
    /// every pass, not only the one whose own commit triggered it --
    /// otherwise a panel whose placement input changed for a reason other
    /// than its own commit (another panel's zone growing or shrinking,
    /// unmapping, or -- as reproduced here -- the output itself resizing)
    /// keeps the stale position it was last configured with until its own
    /// next commit happens to come along.
    ///
    /// `wlr::LayerSurfaceId::dangling_for_test()` is the only synthetic id
    /// this crate exposes, so a unit test cannot hold two distinct live
    /// `LayerEntry`s at once the way a real two-panel scenario would --
    /// see `usable_before_stacks_two_same_edge_panels_instead_of_overlapping`,
    /// which works around the identical limitation the same way. This test
    /// instead reproduces the single-panel case of the same mechanism: an
    /// output resize (`create_output` again -- what `OutputHandler::
    /// new_output` does on a real mode change) moves the panel's placement
    /// input with no layer-side event, real or simulated, in between, and
    /// only `arrange_layers`'s own unconditional reconfigure loop notices.
    #[test]
    fn arrange_layers_reconfigures_every_mapped_panel_even_without_its_own_commit() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, true));

        state.arrange_layers();
        let first_placement = state.layers[&id].last_configured;
        assert_eq!(
            first_placement,
            Some((800, 30, 0, 0)),
            "the initial arrange must record a placement"
        );

        // Nothing calls `configure_layer` or `layer_surface_commit` for
        // this entry between the two arranges -- an output mode change
        // (`create_output` again, standing in for `OutputHandler::
        // new_output` re-recording a resized output's box) moves this
        // panel's placement input with no layer-side event of any kind,
        // real or simulated, in between.
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 640,
                height: 480,
            },
        );
        state.arrange_layers();
        let second_placement = state.layers[&id].last_configured;
        assert_ne!(
            second_placement, first_placement,
            "arrange_layers must reconfigure a panel with no commit of its own"
        );
        assert_eq!(second_placement, Some((640, 30, 0, 0)));
    }

    /// Minor-1's other half: an `arrange_layers` pass that changes nothing
    /// about a panel's placement must not touch `last_configured` at all
    /// -- this is the M4-storm-class guard `configure_layer` itself now
    /// provides, proven here at the `arrange_layers` call site rather than
    /// `configure_layer` directly.
    #[test]
    fn arrange_layers_does_not_reconfigure_an_unchanged_panel() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, true));

        state.arrange_layers();
        let placement = state.layers[&id].last_configured;

        state.arrange_layers();
        assert_eq!(
            state.layers[&id].last_configured, placement,
            "an unchanged panel's placement must not be re-recorded"
        );
    }

    // --- Final review C1: wallpaper / background lowering order ---

    /// C1's regression guard. The scene's actual stacking order is not
    /// observable -- `wlr` exposes no z-query, and `BufferId`/`RectId` have
    /// no synthetic constructor -- so what is pinned is the thing that was
    /// wrong: the *order* of the lower calls `sync_wallpaper_nodes` issues.
    /// `lower_*_to_bottom` gives the bottom to whichever node was lowered
    /// last, so the background rect must come last or it sits on top of the
    /// wallpaper and the wallpaper is never seen (which is exactly what
    /// shipped).
    #[test]
    fn the_background_rect_is_lowered_after_every_wallpaper_node() {
        assert_eq!(
            wallpaper_lower_plan(1, true),
            vec![LowerStep::Wallpaper(0), LowerStep::Background],
            "one output: the background must be lowered last"
        );
        assert_eq!(
            wallpaper_lower_plan(3, true),
            vec![
                LowerStep::Wallpaper(0),
                LowerStep::Wallpaper(1),
                LowerStep::Wallpaper(2),
                LowerStep::Background,
            ],
            "multi-output: still exactly one background lower, still last"
        );
        assert_eq!(
            wallpaper_lower_plan(3, true).last(),
            Some(&LowerStep::Background),
            "the contract in one line: last call wins the bottom, and it must be the background"
        );
    }

    /// The two degenerate inputs. Nothing created means nothing to restack
    /// (existing nodes are already correctly ordered, and re-lowering the
    /// background every idempotent re-sync would be pure churn); no
    /// background rect at all -- every model-only build, and the window
    /// between `State::new` and `set_background` -- still lowers the
    /// wallpaper nodes it made.
    #[test]
    fn the_lower_plan_is_empty_when_nothing_was_created() {
        assert_eq!(wallpaper_lower_plan(0, true), Vec::new());
        assert_eq!(wallpaper_lower_plan(0, false), Vec::new());
        assert_eq!(
            wallpaper_lower_plan(2, false),
            vec![LowerStep::Wallpaper(0), LowerStep::Wallpaper(1)],
            "no background rect to lower, but the new nodes still get lowered"
        );
    }

    // --- Final review I3: per-axis layer placement ---

    /// The panel case, unchanged by I3: `TOP | LEFT | RIGHT`, desired
    /// `(800, 30)` -- both horizontal edges anchored so the width spans the
    /// box, one vertical edge anchored so the height is the client's 30 and
    /// it sits flush against the top.
    #[test]
    fn a_top_anchored_panel_still_spans_the_box_at_its_desired_thickness() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, true));
        state.configure_layer(id);
        assert_eq!(state.layers[&id].last_configured, Some((800, 30, 0, 0)));
    }

    /// I3, first half: a corner-anchored notification (`TOP | RIGHT`,
    /// 300x100 -- the shape every notification daemon uses) must get *its
    /// own* width, flush into the top-right corner. The shipped rule keyed
    /// the whole placement on `a.top != a.bottom` and stretched it across
    /// the full 800px output width, discarding `desired_size` outright.
    #[test]
    fn a_corner_anchored_layer_surface_keeps_its_desired_size() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    right: true,
                    left: false,
                    bottom: false,
                },
                exclusive: 0,
                size: (300, 100),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.configure_layer(id);
        assert_eq!(
            state.layers[&id].last_configured,
            Some((300, 100, 500, 0)),
            "300x100 flush into the top-right corner (x = 800 - 300), not stretched to 800 wide"
        );
    }

    // --- Task 7: N7 (`compute_layer_placement`), N8 (margin plumbing),
    // N11 (skip-hidden maximized re-sync) ---

    /// N7, via the pure function the brief asks for directly: a corner
    /// anchored panel (`TOP | LEFT`, anchored to exactly one edge on each
    /// axis, neither one spanning) keeps its own desired width rather than
    /// spanning the full output -- `compute_layer_placement` answering the
    /// same as `configure_layer` did, just without a live runtime or a
    /// `last_configured` side effect.
    #[test]
    fn a_corner_anchored_panel_keeps_its_desired_width() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        // top+left anchored (a corner), desired 200x30: width must be 200, not 800.
        state.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    left: true,
                    right: false,
                    bottom: false,
                },
                exclusive: 0,
                size: (200, 30),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        let placed = state.compute_layer_placement(wlr::LayerSurfaceId::dangling_for_test());
        assert_eq!(placed.map(|(w, _h, _x, _y)| w), Some(200));
    }

    /// N7/exclusive-edge: tightened `fold_exclusive_zone` against wlroots'
    /// own `wlr_layer_surface_v1_get_exclusive_edge` -- a corner-anchored
    /// panel (`TOP | LEFT`, anchored to exactly one edge on *each* axis,
    /// spanning neither) has no exclusive edge at all per that rule, even
    /// with a positive `exclusive_zone`. The old `anchor.top !=
    /// anchor.bottom` check folded it as a full-width top panel regardless
    /// of `left`/`right`, reserving space the surface never actually spans
    /// edge-to-edge.
    #[test]
    fn a_corner_anchored_exclusive_zone_reserves_nothing() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        state.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    left: true,
                    right: false,
                    bottom: false,
                },
                exclusive: 30,
                size: (200, 30),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.arrange_layers();
        assert_eq!(
            state.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600
            },
            "a corner anchor spans neither axis, so wlroots' own rule assigns it no exclusive edge at all"
        );
    }

    /// Review fix (post-2f04984, Major finding): a single-edge-only anchor
    /// -- e.g. `ANCHOR_TOP` alone, no `left`/`right` at all -- is a
    /// distinct, spec-legal shape from both the corner case above and the
    /// edge-plus-both-perpendicular case `top_panel_entry` already covers.
    /// The wlr-layer-shell spec (`set_exclusive_zone`) is explicit that a
    /// positive exclusive zone is meaningful for "one edge" *or* "an edge
    /// and both perpendicular edges" -- both must reserve. WLCS's own
    /// conformance test (`is_positioned_to_accommodate_other_surfaces_
    /// exclusive_zone`) anchors `ANCHOR_TOP` alone with `exclusive_zone =
    /// 12` and asserts the reservation happens. The realignment toward
    /// wlroots' `get_exclusive_edge` in this same commit over-corrected:
    /// every arm of the rewritten `fold_exclusive_zone` required *both*
    /// perpendicular edges, so this single-edge shape fell through every
    /// `if`/`else if` and reserved nothing -- a real, silent regression
    /// (the corner test above and every pre-existing exclusive-zone test
    /// use `TOP | LEFT | RIGHT`, so none of them caught it).
    #[test]
    fn a_single_edge_only_anchor_still_reserves_its_exclusive_zone() {
        let (tx, _rx) = crossbeam_channel::unbounded();

        let mut top = State::new(icedtea_config::default_config(), tx.clone());
        top.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        top.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: true,
                    left: false,
                    right: false,
                    bottom: false,
                },
                exclusive: 12,
                size: (0, 12),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        top.arrange_layers();
        assert_eq!(
            top.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 12,
                width: 800,
                height: 588
            },
            "ANCHOR_TOP alone with a positive exclusive_zone must reserve at the top (WLCS conformance shape)"
        );

        let mut bottom = State::new(icedtea_config::default_config(), tx.clone());
        bottom.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        bottom.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: false,
                    left: false,
                    right: false,
                    bottom: true,
                },
                exclusive: 12,
                size: (0, 12),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        bottom.arrange_layers();
        assert_eq!(
            bottom.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 588
            },
            "ANCHOR_BOTTOM alone with a positive exclusive_zone must reserve at the bottom"
        );

        let mut left = State::new(icedtea_config::default_config(), tx.clone());
        left.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        left.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: false,
                    left: true,
                    right: false,
                    bottom: false,
                },
                exclusive: 12,
                size: (12, 0),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        left.arrange_layers();
        assert_eq!(
            left.outputs[&0].usable,
            Rectangle {
                x: 12,
                y: 0,
                width: 788,
                height: 600
            },
            "ANCHOR_LEFT alone with a positive exclusive_zone must reserve at the left"
        );

        let mut right = State::new(icedtea_config::default_config(), tx);
        right.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        right.layers.insert(
            wlr::LayerSurfaceId::dangling_for_test(),
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: false,
                    left: false,
                    right: true,
                    bottom: false,
                },
                exclusive: 12,
                size: (12, 0),
                interactive: false,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        right.arrange_layers();
        assert_eq!(
            right.outputs[&0].usable,
            Rectangle {
                x: 0,
                y: 0,
                width: 788,
                height: 600
            },
            "ANCHOR_RIGHT alone with a positive exclusive_zone must reserve at the right"
        );
    }

    /// N11: `arrange_layers`' maximized re-sync loop must skip a maximized
    /// window that is minimized, or sitting on a workspace that is not the
    /// active one -- neither is actually on screen, so warping its stored
    /// geometry to whatever `usable` happens to be right now (on an output
    /// the user cannot see) is wrong the moment it is shown again with a
    /// panel layout that has since changed underfoot with no configure of
    /// its own.
    #[test]
    fn arrange_layers_skips_hidden_maximized_windows_in_the_resync() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.config.appearance.snap_gap = 0;
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        // A maximized, minimized window on the active workspace (0).
        let minimized_id = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        state
            .window_manager
            .set_maximized(minimized_id, true)
            .unwrap();
        state
            .window_manager
            .set_minimized(minimized_id, true)
            .unwrap();
        let geo_before_min = state.window_manager.get(minimized_id).unwrap().geometry;

        // A maximized window on workspace 1, while workspace 0 stays active.
        let other_ws_id = state.window_manager.add_window(
            "app2",
            "t2",
            2,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        state.window_manager.set_workspace(other_ws_id, 1).unwrap();
        state
            .window_manager
            .set_maximized(other_ws_id, true)
            .unwrap();
        let geo_before_ws = state.window_manager.get(other_ws_id).unwrap().geometry;

        // A panel maps, changing `usable` -- the trigger that would have
        // re-synced every maximized window before N11.
        let panel_id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(panel_id, top_panel_entry(30, true));
        state.arrange_layers();

        assert_eq!(
            state.window_manager.get(minimized_id).unwrap().geometry,
            geo_before_min,
            "a minimized maximized window must not be re-synced while hidden"
        );
        assert_eq!(
            state.window_manager.get(other_ws_id).unwrap().geometry,
            geo_before_ws,
            "a maximized window on an inactive workspace must not be re-synced while hidden"
        );
    }

    /// I3, second half: a surface anchored to all four edges with a `0x0`
    /// desired size -- the protocol's fill-the-output case, what lockers
    /// and fullscreen launchers use -- must get the whole usable box. The
    /// shipped rule fell through to the `else` arm and handed it a centered
    /// 200x200. The offset origin proves it used the real box rather than
    /// a `(0, 0)` fallback.
    #[test]
    fn a_four_edge_anchored_layer_surface_fills_the_usable_box() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 100,
                y: 50,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Overlay,
                anchor: wlr::Anchor {
                    top: true,
                    bottom: true,
                    left: true,
                    right: true,
                },
                exclusive: 0,
                size: (0, 0),
                interactive: true,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.configure_layer(id);
        assert_eq!(
            state.layers[&id].last_configured,
            Some((800, 600, 100, 50)),
            "all four anchors with a 0x0 desired size fills the box, not a centered 200x200"
        );
    }

    /// The other fill spelling the protocol allows: *no* anchors and a
    /// `0x0` desired size. Same answer, and the same arm of
    /// `layer_axis_placement` (neither edge, no desired -> span).
    #[test]
    fn an_unanchored_zero_sized_layer_surface_also_fills_the_usable_box() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Overlay,
                anchor: wlr::Anchor {
                    top: false,
                    bottom: false,
                    left: false,
                    right: false,
                },
                exclusive: 0,
                size: (0, 0),
                interactive: true,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.configure_layer(id);
        assert_eq!(state.layers[&id].last_configured, Some((800, 600, 0, 0)));
    }

    /// And an unanchored surface that *did* name a size is still centered
    /// at exactly that size -- the one behavior of the old `else` arm that
    /// was already right, kept.
    #[test]
    fn an_unanchored_sized_layer_surface_is_centered() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Overlay,
                anchor: wlr::Anchor {
                    top: false,
                    bottom: false,
                    left: false,
                    right: false,
                },
                exclusive: 0,
                size: (400, 200),
                interactive: true,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.configure_layer(id);
        assert_eq!(
            state.layers[&id].last_configured,
            Some((400, 200, 200, 200))
        );
    }

    /// Finding 7, security: a client-supplied `desired` size that dwarfs
    /// the output (up to `u32::MAX`) must not wrap negative on the `as
    /// i32` cast -- clamped to the output's own span first, `configure_layer`
    /// must never hand wlroots a negative width/height.
    #[test]
    fn a_pathologically_large_desired_size_is_clamped_to_the_output() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            id,
            LayerEntry {
                output: 0,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: false,
                    bottom: false,
                    left: false,
                    right: false,
                },
                exclusive: 0,
                size: (u32::MAX, u32::MAX),
                interactive: true,
                mapped: true,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        state.configure_layer(id);
        let (w, h, _, _) = state.layers[&id].last_configured.unwrap();
        assert_eq!(
            w, 800,
            "width must clamp to the output's own span, not wrap negative"
        );
        assert_eq!(
            h, 600,
            "height must clamp to the output's own span, not wrap negative"
        );
    }

    /// `layer_axis_placement` directly, for the same clamp: `desired` past
    /// `span` is bounded to `span` before the cast, whichever of `start`/
    /// `end` is set.
    #[test]
    fn layer_axis_placement_clamps_desired_to_the_span() {
        assert_eq!(
            layer_axis_placement(false, false, u32::MAX, 0, 800),
            (800, 0)
        );
        assert_eq!(
            layer_axis_placement(true, false, u32::MAX, 0, 800),
            (800, 0),
            "flush start, still clamped"
        );
        assert_eq!(
            layer_axis_placement(false, true, u32::MAX, 0, 800),
            (800, 0),
            "flush end, still clamped"
        );
    }

    // --- Final review I1: a remapped layer surface must be reconfigured ---

    /// I1 at the model level (the wire-level half lives in
    /// `client_protocol::a_remapped_layer_panel_is_configured_again`):
    /// `layer_surface_unmapped` must forget `last_configured`, or
    /// `configure_layer`'s storm guard suppresses the mandatory configure
    /// the remap needs -- with an identical placement, which is the normal
    /// case for an auto-hide panel, the guard matches and the client hangs
    /// forever.
    #[test]
    fn unmapping_a_layer_surface_forgets_its_placement_so_a_remap_reconfigures() {
        use wlr::ToplevelHandler;

        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(id, top_panel_entry(30, true));

        state.configure_layer(id);
        assert_eq!(state.layers[&id].last_configured, Some((800, 30, 0, 0)));

        state.layer_surface_unmapped(id);
        assert_eq!(
            state.layers[&id].last_configured, None,
            "an unmap must forget the placement -- wlroots resets `initialized`, so the remap needs a fresh configure"
        );

        // The remap: same output, same anchors, so the recomputed placement
        // is byte-identical to the one before the unmap. That is precisely
        // the case the storm guard used to swallow.
        if let Some(entry) = state.layers.get_mut(&id) {
            entry.mapped = true;
        }
        state.configure_layer(id);
        assert_eq!(
            state.layers[&id].last_configured,
            Some((800, 30, 0, 0)),
            "the remap must re-send the identical placement, not be suppressed as unchanged"
        );
    }

    // --- Final review I2: unmapping on an inactive workspace ---

    /// I2. A window focused on workspace 2 unmaps while workspace 1 is
    /// active. Before the fix, `unmapped` re-picked only when the unmapping
    /// window was the *active* workspace's focus, so workspace 2 kept a
    /// `focused_window` pointer aimed at an unmapped row; switching back
    /// re-picked only on `is_none()`, so the pointer survived, the seat went
    /// dead (correctly refusing an invisible window), and `close`/
    /// `maximize`/`fullscreen`/`snap` all still resolved to it.
    #[test]
    fn unmapping_on_an_inactive_workspace_does_not_leave_a_stale_focus_pointer() {
        use wlr::ToplevelHandler;

        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        let key = crate::wayland::ToplevelKey::for_test(1);
        state.new_toplevel(key, "a", "A", 1);
        let id = state.wayland.window_for(key).expect("bound");

        // Focus it on workspace 1 (`move_to_workspace` switches there and
        // focuses it), then leave for workspace 0.
        state
            .move_to_workspace(id, 1)
            .expect("moved to workspace 1");
        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(id)
        );
        state.switch_workspace(0).expect("switched away");

        // The unmap happens while workspace 1 is *not* the active one.
        state.unmapped(wlr::ToplevelId::dangling_nth_for_test(1));

        state.switch_workspace(1).expect("switched back");
        assert!(
            state.window_manager.focused_window().is_none(),
            "workspace 1 must have no focused window: its only candidate is unmapped"
        );
        assert!(
            state.window_manager.get(id).is_some_and(|w| !w.focused),
            "the unmapped row must not still claim focus"
        );
        // The consequence that made the stale pointer actionable rather
        // than merely untidy: a focus-targeted command must find nothing.
        assert!(
            state.apply_action("close").is_none(),
            "a close action must not resolve to the unmapped window"
        );
        assert!(
            state.apply_action("maximize").is_none(),
            "a maximize action must not resolve to the unmapped window"
        );
    }

    /// The other half of I2's fix: `switch_workspace`'s guard widened from
    /// `is_none()` to "no focusable focus", which also covers the
    /// pre-existing minimized case -- switching to a workspace whose focus
    /// pointer names a minimized window now hands focus to a real
    /// candidate instead of leaving the seat dead.
    #[test]
    fn switching_to_a_workspace_whose_focus_is_minimized_repicks() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );

        let a = state.window_manager.add_window(
            "a",
            "A",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        );
        let b = state.window_manager.add_window(
            "b",
            "B",
            2,
            Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        );
        // `b` was added second so it is focused; minimize it and leave.
        assert_eq!(state.window_manager.focused_window().map(|w| w.id), Some(b));
        state.window_manager.set_minimized(b, true);
        state.switch_workspace(1).expect("switched away");
        state.switch_workspace(0).expect("switched back");

        assert_eq!(
            state.window_manager.focused_window().map(|w| w.id),
            Some(a),
            "a minimized focus pointer must be re-picked on switch-back, not kept"
        );
    }

    // -- T5: output-management (persisted display config) ----------------
    //
    // These cover the runtime-independent halves of Task 5: the persisted
    // config lookup/round-trip, the `AppliedHead` -> `DisplayConfig` upsert,
    // the transform integer mapping, and the offscreen-window reclaim a
    // shrinking apply triggers. The live-output halves -- `new_output`
    // committing mode/scale/transform/position on a real `wlr::Output`, and
    // `output_configuration_applied` re-deriving geometry through
    // `output_layout_box` -- need an attached `wlr::Runtime` (both handlers
    // early-return without one), so they are exercised end-to-end by the
    // protocol round-trip in T6/T7 rather than here. See the task notes.

    fn head(name: &str, enabled: bool) -> wlr::AppliedHead {
        wlr::AppliedHead {
            name: Some(name.to_string()),
            enabled,
            width: 1920,
            height: 1080,
            refresh_mhz: 60_000,
            x: 0,
            y: 0,
            scale: 1.0,
            transform: wlr::Transform::Normal,
        }
    }

    #[test]
    fn transform_integer_mapping_round_trips_every_variant() {
        for value in 0..=7 {
            let t = State::transform_from_i32(value);
            assert_eq!(
                State::transform_to_i32(t),
                value,
                "transform {value} must round-trip"
            );
        }
        // Out-of-range falls back to Normal (0), not a panic.
        assert_eq!(State::transform_from_i32(99), wlr::Transform::Normal);
    }

    #[test]
    fn upsert_displays_inserts_new_and_replaces_by_name() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert!(state.config.displays.is_empty());

        // Two distinct outputs -> two appended entries.
        state.upsert_displays_from_heads(&[head("DP-1", true), head("HDMI-A-1", true)]);
        assert_eq!(state.config.displays.len(), 2);
        assert_eq!(state.config.displays[0].name, "DP-1");
        assert_eq!(state.config.displays[1].name, "HDMI-A-1");

        // Re-applying DP-1 disabled must REPLACE in place (still 2 entries,
        // order preserved), not append a duplicate.
        let mut disabled = head("DP-1", false);
        disabled.width = 2560;
        disabled.height = 1440;
        disabled.transform = wlr::Transform::R270;
        disabled.scale = 2.0;
        state.upsert_displays_from_heads(&[disabled]);
        assert_eq!(
            state.config.displays.len(),
            2,
            "same name must not append a duplicate"
        );
        let dp1 = &state.config.displays[0];
        assert_eq!(dp1.name, "DP-1");
        assert!(!dp1.enabled);
        assert_eq!(dp1.width, 2560);
        assert_eq!(dp1.height, 1440);
        assert_eq!(dp1.transform, 3, "wlr::Transform::R270 persists as 3");
        assert_eq!(dp1.scale, 2.0);
    }

    #[test]
    fn upsert_displays_skips_unnamed_head() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        let mut anon = head("ignored", true);
        anon.name = None;
        state.upsert_displays_from_heads(&[anon]);
        assert!(
            state.config.displays.is_empty(),
            "an unnamed head has no key to store"
        );
    }

    #[test]
    fn save_config_to_persists_displays_to_redb() {
        // Exercises the exact synchronous body `spawn_config_save` runs on
        // its worker thread (see `save_config_to`'s doc for why the detached
        // thread itself is not what the test drives). The upserted displays
        // must survive a real `load_or_default` round-trip.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.redb");

        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.config_path = Some(path.clone());
        state.upsert_displays_from_heads(&[head("DP-1", true)]);

        State::save_config_to(&state.config_db_lock, &state.config, &path);

        let loaded = icedtea_config::load_or_default(&path);
        assert_eq!(
            loaded.displays.len(),
            1,
            "save_config_to must persist the displays"
        );
        assert_eq!(loaded.displays[0].name, "DP-1");
        assert_eq!(loaded.displays[0].width, 1920);
        assert_eq!(loaded.displays[0].refresh_mhz, 60_000);
    }

    #[test]
    fn racing_save_and_reload_never_wipes_live_config() {
        // Review finding #3: `spawn_config_save` and `spawn_config_reload`
        // each open a fresh redb `Database` on a detached worker thread, but
        // redb permits only ONE handle per file per process. Without the
        // shared open-lock a save racing a reload hits `DatabaseAlreadyOpen`;
        // `load_or_default` then silently returns `default_config()`, WIPING
        // the live appearance/keybindings/workspaces on disk. The lock (held
        // by each worker across its whole open+use+drop) serializes the two
        // opens so neither ever observes a file the other still has open. This
        // hammers both real code paths (`save_config_to` /
        // `load_config_locked`) through one shared lock and asserts no reload
        // ever surfaced a wipe. Remove the lock and DatabaseAlreadyOpen makes
        // this fail.
        use std::sync::atomic::{AtomicBool, Ordering};

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.redb");

        // Seed a NON-default config so a wipe is observable: a load that comes
        // back with the default workspace names is a wipe; one that sees these
        // custom names is a real read.
        let mut seeded = default_config();
        seeded.workspace_names = vec!["alpha".into(), "beta".into(), "gamma".into()];
        assert_ne!(seeded.workspace_names, default_config().workspace_names);
        {
            let db = icedtea_config::open(&path).expect("seed open");
            seeded.save(&db).expect("seed save");
        }

        let lock = Arc::new(Mutex::new(()));
        let wiped = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        // Writers: persist the seeded (non-default) config repeatedly.
        for _ in 0..8 {
            let lock = Arc::clone(&lock);
            let cfg = seeded.clone();
            let path = path.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..12 {
                    State::save_config_to(&lock, &cfg, &path);
                }
            }));
        }
        // Readers: load through the same lock and flag any default wipe.
        for _ in 0..8 {
            let lock = Arc::clone(&lock);
            let path = path.clone();
            let wiped = Arc::clone(&wiped);
            let default_names = default_config().workspace_names;
            handles.push(std::thread::spawn(move || {
                for _ in 0..12 {
                    // `None` == kept current (a persistent lock), never a wipe.
                    // A `Some` whose workspaces went default IS the wipe we guard
                    // against.
                    if let Some(cfg) = State::load_config_locked(&lock, &path)
                        && cfg.workspace_names == default_names
                    {
                        wiped.store(true, Ordering::SeqCst);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        assert!(
            !wiped.load(Ordering::SeqCst),
            "a save/reload race surfaced default_config() -- the live config was wiped"
        );
        // The file still holds the seeded config after all the racing opens.
        let final_cfg = icedtea_config::load_or_default(&path);
        assert_eq!(final_cfg.workspace_names, seeded.workspace_names);
    }

    #[test]
    fn load_config_locked_keeps_current_on_persistent_lock() {
        // Review finding #2: a cross-process open collision (the settings app
        // holding the DB) must NEVER downgrade the live config to defaults.
        // `load_config_locked` returns `None` (== keep `self.config`) when the
        // file stays locked, rather than `Some(default_config())`. Simulate the
        // collision by holding a live redb handle across the call: redb's
        // whole-file lock rejects the second opener within this process too, so
        // every retry sees `LoadOutcome::Locked` and the method gives up with
        // `None`.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.redb");

        let mut seeded = default_config();
        seeded.workspace_names = vec!["keepme".into()];
        {
            let db = icedtea_config::open(&path).expect("seed open");
            seeded.save(&db).expect("seed save");
        }

        let lock = Arc::new(Mutex::new(()));
        // Hold the DB open for the whole load attempt so it stays locked.
        let _held = icedtea_config::open(&path).expect("hold open");
        assert!(
            State::load_config_locked(&lock, &path).is_none(),
            "a persistently locked db must return None (keep current), never a default wipe"
        );
    }

    #[test]
    fn guarded_scale_rejects_nonpositive_and_nonfinite() {
        // Review finding #12: a persisted/echoed-back `DisplayConfig.scale`
        // that is <= 0 or non-finite must not reach wlroots.
        assert_eq!(State::guarded_scale(1.0), 1.0);
        assert_eq!(State::guarded_scale(2.5), 2.5);
        assert_eq!(
            State::guarded_scale(0.0),
            1.0,
            "zero scale falls back to 1.0"
        );
        assert_eq!(
            State::guarded_scale(-2.0),
            1.0,
            "negative scale falls back to 1.0"
        );
        assert_eq!(
            State::guarded_scale(f64::NAN),
            1.0,
            "NaN scale falls back to 1.0"
        );
        assert_eq!(
            State::guarded_scale(f64::INFINITY),
            1.0,
            "inf scale falls back to 1.0"
        );
        assert_eq!(
            State::guarded_scale(f64::NEG_INFINITY),
            1.0,
            "-inf scale falls back to 1.0"
        );
        // Review finding #8: an absurdly large but finite scale is capped, not
        // passed through to overflow `96 * scale` downstream.
        assert_eq!(
            State::guarded_scale(1.0e9),
            State::MAX_OUTPUT_SCALE as f32,
            "a huge finite scale is capped at MAX_OUTPUT_SCALE"
        );
    }

    /// Review finding #8: `primary_output_scale` never returns a value that
    /// overflows the `96 * scale` DPI computation, even when the stored output
    /// scale is non-finite or absurdly large (a corrupted persisted value or a
    /// hostile `SetOutputScaleForTest`). The cast is clamped to
    /// `[1, MAX_OUTPUT_SCALE]` so `export_x11_dpi` stays well within `i32`.
    #[test]
    fn primary_output_scale_is_clamped_against_overflow() {
        let (mut state, _rx) = state_with_output(1000, 800);
        for bad in [f64::INFINITY, f64::NAN, 1.0e12, -5.0, 0.0] {
            state.outputs.get_mut(&0).unwrap().scale = bad;
            let s = state.primary_output_scale();
            assert!(
                (1..=State::MAX_OUTPUT_SCALE as i32).contains(&s),
                "scale {bad} produced out-of-range {s}"
            );
            // The exact multiplication `export_x11_dpi` performs must not overflow.
            assert!(96i32.checked_mul(s.max(1)).is_some(), "96 * {s} overflowed");
        }
    }

    #[test]
    fn boot_disable_would_strand_keys_off_live_outputs_not_persisted_bools() {
        // Review finding #1 (undock scenario): the boot guard must key off LIVE
        // outputs, never the persisted enabled bools. A persisted-enabled but
        // physically-ABSENT connector must NOT make `new_output` honor the
        // disable of the only PRESENT output and go dark.
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);

        // Undock: eDP-1 (present) persisted disabled; DP-1 (persisted enabled)
        // is unplugged, so it never comes up. With NO live outputs yet,
        // honoring eDP-1's disable would strand the session at zero -> the guard
        // must report "would strand" so `new_output` force-enables it. The old
        // persisted-bool guard (`any_display_enabled`) returned "safe to
        // disable" here BECAUSE DP-1 is persisted enabled -- exactly the bug.
        state.config.displays = vec![
            icedtea_contract::DisplayConfig {
                name: "eDP-1".into(),
                enabled: false,
                width: 1920,
                height: 1080,
                refresh_mhz: 60_000,
                x: 0,
                y: 0,
                scale: 1.0,
                transform: 0,
            },
            icedtea_contract::DisplayConfig {
                name: "DP-1".into(),
                enabled: true,
                width: 2560,
                height: 1440,
                refresh_mhz: 144_000,
                x: 0,
                y: 0,
                scale: 1.0,
                transform: 0,
            },
        ];
        assert!(
            state.boot_disable_would_strand(),
            "no LIVE output yet -> honoring a disable would strand at zero, even though a persisted-enabled connector exists (it is absent)"
        );

        // Once some present output is actually live, honoring another output's
        // persisted disable is safe.
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 2560,
                height: 1440,
            },
        );
        assert!(
            !state.boot_disable_would_strand(),
            "a live output exists -> honoring a disable no longer strands the session"
        );
    }

    #[test]
    fn resolve_unique_disabled_refuses_ambiguous_names() {
        // Review finding #13: the re-enable name->id lookup must not silently
        // rehydrate an arbitrary id when two disabled outputs share a name.
        // Exercised through a plain `HashMap<u64, String>` stand-in because
        // `wlr::OutputId` cannot be constructed outside the wlr crate.
        let mut disabled: HashMap<u64, String> = HashMap::new();
        disabled.insert(1, "DP-1".into());
        disabled.insert(2, "DP-2".into());

        // Unambiguous name -> its id.
        assert!(matches!(
            State::resolve_unique_disabled(&disabled, "DP-1"),
            DisabledLookup::Unique(1)
        ));
        // Missing name -> None.
        assert!(matches!(
            State::resolve_unique_disabled(&disabled, "HDMI-A-1"),
            DisabledLookup::None
        ));

        // Two disabled outputs share the empty name (the exact #15 write-side
        // collision, now on the read side): the lookup must report Ambiguous,
        // never guess an id.
        disabled.insert(3, String::new());
        disabled.insert(4, String::new());
        assert!(matches!(
            State::resolve_unique_disabled(&disabled, ""),
            DisabledLookup::Ambiguous
        ));

        // A single empty-named output is still unambiguous.
        let mut one_empty: HashMap<u64, String> = HashMap::new();
        one_empty.insert(9, String::new());
        assert!(matches!(
            State::resolve_unique_disabled(&one_empty, ""),
            DisabledLookup::Unique(9)
        ));
    }

    #[test]
    fn disable_would_strand_session_guards_the_last_output() {
        // Review finding #5 (interactive guard): refuse to disable the last
        // active output unless the same atomic apply enables another.
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        // Only output 0 active, nothing else enabling -> disabling it strands.
        assert!(state.disable_would_strand_session(0, false));
        // A head in the same batch is enabling -> safe to disable index 0.
        assert!(!state.disable_would_strand_session(0, true));
        // A second active output survives the disable -> safe.
        state.create_output(
            1,
            Rectangle {
                x: 800,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        assert!(!state.disable_would_strand_session(0, false));
    }

    #[test]
    fn reclaim_offscreen_windows_clamps_stranded_window_onto_survivor() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        // Single 800x600 output at origin: a window centered at (1400,300)
        // sits entirely off it (simulating the output it lived on shrinking
        // away under an applied config).
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let id = state.window_manager.add_window(
            "a",
            "A",
            1,
            Rectangle {
                x: 1350,
                y: 250,
                width: 100,
                height: 100,
            },
        );

        state.reclaim_offscreen_windows();

        let geo = state.window_manager.get(id).expect("window").geometry;
        assert!(
            geo.x >= 0 && geo.x + geo.width <= 800 && geo.y >= 0 && geo.y + geo.height <= 600,
            "stranded window must be clamped inside the survivor output, got {geo:?}"
        );
    }

    #[test]
    fn reclaim_offscreen_windows_leaves_onscreen_window_untouched() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let onscreen = Rectangle {
            x: 100,
            y: 100,
            width: 200,
            height: 150,
        };
        let id = state.window_manager.add_window("a", "A", 1, onscreen);

        state.reclaim_offscreen_windows();

        assert_eq!(
            state.window_manager.get(id).expect("window").geometry,
            onscreen,
            "a window already on an output must not be moved"
        );
    }

    /// A2 task 8: the xdg-activation focus-steal policy, isolated from the
    /// handler so the table below can exercise every combination.
    #[test]
    fn activation_may_steal_focus_only_for_a_seat_backed_request_from_the_focused_window() {
        let a = WindowId(1);
        let b = WindowId(2);
        let cases = [
            // (has_seat, target_activatable, requester, focused, expected)
            (true, true, Some(a), Some(a), true),
            // No seat: the token names no seat at all, the shape a launcher
            // mints for another process to redeem later.
            (false, true, Some(a), Some(a), false),
            // Seat-backed, but the requester is not the focused window.
            (true, true, Some(b), Some(a), false),
            // Token named no (live) requesting toplevel.
            (true, true, None, Some(a), false),
            // Nothing focused at all: there is no interaction to vouch for.
            (true, true, Some(a), None, false),
            // The row the table was missing (review finding, low): a
            // seat-backed token that names no requesting toplevel, redeemed
            // with nothing focused. Both of the two conditions that could
            // still refuse it are absent, so nothing but the explicit
            // `requester.is_some()` clause stands between it and a steal.
            (true, true, None, None, false),
            (false, true, None, None, false),
            // Otherwise-perfect request, but the target is not somewhere
            // focus can land without moving the user (off the active
            // workspace, minimized, or unmapped): refused, so it is flagged
            // instead of honored into a silent no-op.
            (true, false, Some(a), Some(a), false),
            (false, false, Some(b), Some(a), false),
        ];
        for (has_seat, target_activatable, requester, focused, expected) in cases {
            assert_eq!(
                activation_may_steal_focus(has_seat, target_activatable, requester, focused),
                expected,
                "has_seat={has_seat} target_activatable={target_activatable} \
                 requester={requester:?} focused={focused:?}"
            );
        }
    }

    /// Recording a chain builds it root-first and `popup_chain` returns it
    /// deepest-last, whatever order the map iterates in.
    ///
    /// The chain order is load-bearing twice over: `reconstrain_popups`
    /// reconfigures parents before children (a child's own placement is
    /// expressed against its parent), and `dismiss` order is the protocol's
    /// reverse-creation requirement.
    ///
    /// Mutation check: drop the `sort_unstable_by_key` from `popup_chain` and
    /// this fails -- `HashMap` iteration order is not creation order.
    #[test]
    fn a_recorded_popup_chain_reports_its_root_and_orders_itself_deepest_last() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let window = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );

        let outer = crate::wayland::PopupKey::for_test(1);
        let inner = crate::wayland::PopupKey::for_test(2);
        let deepest = crate::wayland::PopupKey::for_test(3);
        state.record_popup(outer, PopupHost::Window(window), false);
        state.record_popup(inner, PopupHost::Popup(outer), false);
        state.record_popup(deepest, PopupHost::Popup(inner), false);

        let root = PopupRoot::Window(window);
        assert_eq!(state.popup_root(outer), Some(root));
        assert_eq!(
            state.popup_root(deepest),
            Some(root),
            "a nested popup's root is the chain's bottom, not its parent"
        );
        assert_eq!(state.popup_chain(root), vec![outer, inner, deepest]);
        assert_eq!(state.popup_count(), 3);

        // A key nobody recorded is a miss, never a panic.
        assert_eq!(
            state.popup_root(crate::wayland::PopupKey::for_test(99)),
            None
        );
        assert!(
            state
                .popup_chain(PopupRoot::Layer(wlr::LayerSurfaceId::dangling_for_test()))
                .is_empty(),
            "a root with no popups has an empty chain"
        );
    }

    /// Forgetting a popup drops it from both the map and the stack, and
    /// forgetting one nobody recorded is a no-op rather than a panic
    /// (contract §1.3: a destroy may name an id this handler was never told
    /// about).
    ///
    /// Mutation check: drop the `popup_stack.retain` from `forget_popup` and
    /// the stack-length assertion fails -- a stale key would keep answering
    /// `popup_at_point` after its popup was gone.
    #[test]
    fn forgetting_a_popup_clears_both_the_map_and_the_stack() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let window = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            },
        );
        let a = crate::wayland::PopupKey::for_test(1);
        let b = crate::wayland::PopupKey::for_test(2);
        state.record_popup(a, PopupHost::Window(window), false);
        state.record_popup(b, PopupHost::Popup(a), false);

        state.forget_popup(b);
        assert_eq!(state.popup_chain(PopupRoot::Window(window)), vec![a]);
        assert_eq!(state.popup_stack, vec![a]);

        state.forget_popup(crate::wayland::PopupKey::for_test(99));
        assert_eq!(state.popup_count(), 1, "an unknown key forgets nothing");

        state.forget_popup(a);
        assert_eq!(state.popup_count(), 0);
        assert!(state.popup_stack.is_empty());
    }

    /// A popup whose host this compositor does not model is dropped rather
    /// than recorded under a fabricated root -- the untrusted-client rule
    /// (contract §9: a malformed value is dropped and logged, never a panic).
    ///
    /// Mutation check: make `popup_root_of_host` return
    /// `Some(PopupRoot::Window(WindowId(0)))` on a miss and the count
    /// assertion fails.
    #[test]
    fn a_popup_on_an_unmodelled_host_is_not_recorded() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        // A parent popup that was never recorded: the client raced its own
        // destroy against the child's creation.
        state.record_popup(
            crate::wayland::PopupKey::for_test(1),
            PopupHost::Popup(crate::wayland::PopupKey::for_test(50)),
            true,
        );
        assert_eq!(state.popup_count(), 0);
        assert!(state.popup_stack.is_empty());
        assert_eq!(
            state.focus_before_popup, None,
            "a popup that was never recorded must not have parked a focus \
             restore target"
        );
    }
    /// The constraint box is the root's output `usable` rect expressed in the
    /// **root surface's own** coordinates -- contract ruling R7, and
    /// `wlr_xdg_popup_unconstrain_from_box`'s own header. For a
    /// server-decorated window that origin is the *content* rect, not the
    /// frame: the client's surface starts one title bar below the frame's top
    /// edge, so a box translated by the frame origin would let a popup ride
    /// `TITLE_BAR_HEIGHT` past the bottom of the screen.
    ///
    /// Mutation check: translate by `w.geometry` instead of
    /// `content_rect(w.geometry, ssd)` and the `y` assertion fails by exactly
    /// `TITLE_BAR_HEIGHT`.
    #[test]
    fn a_popups_constraint_box_is_the_usable_area_in_root_surface_coordinates() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        // A 30px top exclusive zone, so `usable` is genuinely smaller than
        // `geometry` and the test cannot pass by reading the wrong one.
        if let Some(output) = state.outputs.get_mut(&0) {
            output.usable = Rectangle {
                x: 0,
                y: 30,
                width: 800,
                height: 570,
            };
        }
        // "app" is not GTK-style, so `has_ssd` gives it a title bar.
        let window = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 50,
                width: 300,
                height: 200,
            },
        );
        let ssd = crate::decoration::has_ssd("app", None, false);
        assert!(ssd, "this test needs a server-decorated window");
        let content = crate::decoration::content_rect(
            Rectangle {
                x: 100,
                y: 50,
                width: 300,
                height: 200,
            },
            ssd,
        );

        let popup = crate::wayland::PopupKey::for_test(1);
        state.record_popup(popup, PopupHost::Window(window), false);
        assert_eq!(
            state.popup_constraint_box(popup),
            Some(Rectangle {
                x: 0 - content.x,
                y: 30 - content.y,
                width: 800,
                height: 570,
            })
        );

        // A nested popup is constrained against the same root, not against
        // its immediate parent.
        let nested = crate::wayland::PopupKey::for_test(2);
        state.record_popup(nested, PopupHost::Popup(popup), false);
        assert_eq!(
            state.popup_constraint_box(nested),
            state.popup_constraint_box(popup)
        );

        // An unrecorded key, and a popup whose output vanished, are misses.
        assert_eq!(
            state.popup_constraint_box(crate::wayland::PopupKey::for_test(99)),
            None
        );
        state.outputs.remove(&0);
        assert_eq!(state.popup_constraint_box(popup), None);
    }

    /// `popup_at_point` searches the stack back to front, so the newest popup
    /// over a point wins, and answers nothing at all while the session is
    /// locked.
    ///
    /// Mutation check: drop the `.rev()` and the "topmost wins" assertion
    /// fails; drop the `session_locked` gate and the locked assertion fails.
    #[test]
    fn popup_at_point_finds_the_topmost_mapped_popup_and_nothing_while_locked() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let window = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 100,
                y: 50,
                width: 300,
                height: 200,
            },
        );
        let ssd = crate::decoration::has_ssd("app", None, false);
        let content = crate::decoration::content_rect(
            Rectangle {
                x: 100,
                y: 50,
                width: 300,
                height: 200,
            },
            ssd,
        );

        let under = crate::wayland::PopupKey::for_test(1);
        let over = crate::wayland::PopupKey::for_test(2);
        let unmapped = crate::wayland::PopupKey::for_test(3);
        for key in [under, over, unmapped] {
            state.record_popup(key, PopupHost::Window(window), false);
        }
        // All three cover the same root-surface rect; only the first two are
        // mapped.
        let rect = Rectangle {
            x: 10,
            y: 10,
            width: 40,
            height: 40,
        };
        for key in [under, over] {
            if let Some(entry) = state.popups.get_mut(&key) {
                entry.mapped = true;
                entry.geometry = rect;
            }
        }
        if let Some(entry) = state.popups.get_mut(&unmapped) {
            entry.geometry = rect;
        }

        // Frame space: the root surface's origin plus the popup's own offset.
        let inside = (content.x + 20, content.y + 20);
        assert_eq!(
            state.popup_at_point(inside),
            Some(over),
            "the newest popup over the point wins"
        );
        assert_eq!(
            state.popup_at_point((content.x + 200, content.y + 200)),
            None,
            "a point outside every popup hits nothing"
        );

        // Unmapping the top one hands the point to the one below it, not to
        // the unmapped third.
        if let Some(entry) = state.popups.get_mut(&over) {
            entry.mapped = false;
        }
        assert_eq!(state.popup_at_point(inside), Some(under));

        state.session_locked = true;
        assert_eq!(
            state.popup_at_point(inside),
            None,
            "no popup answers input while the session is locked"
        );
    }

    /// Hostile positioner geometry never panics.
    ///
    /// Every number in a `PopupEntry::geometry` originates in an
    /// `xdg_positioner` the client wrote, so `i32::MIN`/`i32::MAX` extents and
    /// origins are reachable from a malicious client, and so is a root whose
    /// own geometry is degenerate. Contract §9: a malformed value is dropped,
    /// never a panic -- in particular the coordinate translation
    /// `point - origin` must not overflow.
    ///
    /// Mutation check: replace the `saturating_sub` in `popup_at_point` with
    /// `-` and this test panics with "attempt to subtract with overflow" in a
    /// debug build.
    #[test]
    fn hostile_popup_geometry_never_panics() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        state.create_output(
            0,
            Rectangle {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
        );
        let window = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: i32::MIN,
                y: i32::MIN,
                width: 1,
                height: 1,
            },
        );
        let popup = crate::wayland::PopupKey::for_test(1);
        state.record_popup(popup, PopupHost::Window(window), false);
        if let Some(entry) = state.popups.get_mut(&popup) {
            entry.mapped = true;
            entry.geometry = Rectangle {
                x: i32::MIN,
                y: i32::MAX,
                width: i32::MIN,
                height: i32::MAX,
            };
        }

        for point in [
            (0, 0),
            (i32::MIN, i32::MIN),
            (i32::MAX, i32::MAX),
            (i32::MIN, i32::MAX),
        ] {
            let _ = state.popup_at_point(point);
        }
        let _ = state.popup_constraint_box(popup);
        let _ = state.popup_chain(PopupRoot::Window(window));
    }
    /// A grabbing chain parks its root and hands focus back to it when the
    /// last popup dies -- to the **parent**, not to whatever the pointer
    /// wandered onto while the menu was up (spec §2: "on destroy, focus
    /// returns to the parent, not the pointer position").
    ///
    /// Mutation check: delete the `restore_focus_after_popups()` call from
    /// `popup_destroyed` and the final `focused_id()` assertion reports `b`.
    #[test]
    fn a_grabbing_chains_end_returns_focus_to_its_root_not_to_the_pointer() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let a = state.window_manager.add_window(
            "a.app",
            "a",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 200,
                height: 200,
            },
        );
        let b = state.window_manager.add_window(
            "b.app",
            "b",
            2,
            Rectangle {
                x: 300,
                y: 0,
                width: 200,
                height: 200,
            },
        );
        state.window_manager.focus(a);

        let outer = crate::wayland::PopupKey::for_test(1);
        let inner = crate::wayland::PopupKey::for_test(2);
        state.record_popup(outer, PopupHost::Window(a), true);
        state.record_popup(inner, PopupHost::Popup(outer), false);
        assert_eq!(
            state.focus_before_popup,
            Some(PopupRoot::Window(a)),
            "the chain's first, grabbing popup parks its root"
        );

        // While the chain is up, something else takes model focus -- the
        // click that dismisses it lands on another window.
        state.window_manager.focus(b);
        assert_eq!(state.focused_id(), Some(b));

        // The chain unwinds deepest-first, as the protocol requires.
        state.popup_destroyed_for_test(inner);
        assert_eq!(
            state.focused_id(),
            Some(b),
            "focus is restored only when the chain has fully emptied"
        );
        state.popup_destroyed_for_test(outer);
        assert_eq!(state.focused_id(), Some(a));
        assert_eq!(
            state.focus_before_popup, None,
            "the restore target is taken, so a later chain cannot inherit it"
        );
    }

    /// A non-grabbing chain never moves keyboard focus -- not when it opens,
    /// and not when it ends (contract deviation D3).
    ///
    /// Mutation check: park `focus_before_popup` unconditionally in
    /// `record_popup` (drop the `grabbing &&`) and the final assertion
    /// reports `a`.
    #[test]
    fn a_non_grabbing_chain_never_moves_focus_at_either_end() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let a = state.window_manager.add_window(
            "a.app",
            "a",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 200,
                height: 200,
            },
        );
        let b = state.window_manager.add_window(
            "b.app",
            "b",
            2,
            Rectangle {
                x: 300,
                y: 0,
                width: 200,
                height: 200,
            },
        );
        state.window_manager.focus(b);

        let popup = crate::wayland::PopupKey::for_test(1);
        state.record_popup(popup, PopupHost::Window(a), false);
        assert_eq!(state.focus_before_popup, None);
        assert_eq!(state.focused_id(), Some(b), "opening moved nothing");

        state.popup_destroyed_for_test(popup);
        assert_eq!(state.focused_id(), Some(b), "closing moved nothing");
    }

    /// A root's death takes its whole popup chain with it, however many
    /// per-popup destroys the library gets round to delivering.
    ///
    /// Mutation check: delete the `forget_popups_of_root` call from
    /// `forget_toplevel` and `popup_count()` stays at 2.
    #[test]
    fn a_dying_root_prunes_every_popup_under_it() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(icedtea_config::default_config(), tx);
        let window = state.window_manager.add_window(
            "app",
            "t",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 200,
                height: 200,
            },
        );
        let toplevel = crate::wayland::ToplevelKey::for_test(1);
        state.wayland.bind(window, toplevel);
        let outer = crate::wayland::PopupKey::for_test(1);
        let inner = crate::wayland::PopupKey::for_test(2);
        state.record_popup(outer, PopupHost::Window(window), true);
        state.record_popup(inner, PopupHost::Popup(outer), false);
        assert_eq!(state.popup_count(), 2);

        state.forget_toplevel(toplevel);
        assert_eq!(state.popup_count(), 0);
        assert!(state.popup_stack.is_empty());
        assert_eq!(
            state.focus_before_popup, None,
            "a restore target whose root is gone is dropped, not left to \
             re-focus a dead window"
        );

        // And a layer root behaves the same way.
        let layer = wlr::LayerSurfaceId::dangling_for_test();
        state.layers.insert(
            layer,
            LayerEntry {
                output: NO_OUTPUT,
                sequence: 0,
                layer: wlr::Layer::Top,
                anchor: wlr::Anchor {
                    top: false,
                    bottom: false,
                    left: false,
                    right: false,
                },
                exclusive: 0,
                size: (0, 0),
                interactive: false,
                mapped: false,
                last_configured: None,
                margin: (0, 0, 0, 0),
            },
        );
        let on_panel = crate::wayland::PopupKey::for_test(3);
        state.record_popup(on_panel, PopupHost::Layer(layer), false);
        assert_eq!(state.popup_count(), 1);
        wlr::ToplevelHandler::layer_surface_destroyed(&mut state, layer);
        assert_eq!(state.popup_count(), 0);
    }

    /// M7: the touch notification handlers drive `touch_active` -- down
    /// sets it, up re-derives it (no runtime in a unit test reads "no
    /// points"), cancel clears it.
    #[test]
    fn touch_handlers_drive_touch_active() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert!(!state.touch_active, "no touch down yet");
        wlr::SeatHandler::touch_down(&mut state, wlr::TouchId::dangling_nth_for_test(0));
        assert!(state.touch_active, "down must mark touch active");
        wlr::SeatHandler::touch_up(&mut state, wlr::TouchId::dangling_nth_for_test(0));
        assert!(
            !state.touch_active,
            "up with no live points must clear touch active"
        );
        wlr::SeatHandler::touch_down(&mut state, wlr::TouchId::dangling_nth_for_test(1));
        wlr::SeatHandler::touch_cancelled(&mut state);
        assert!(
            !state.touch_active,
            "cancel must clear touch active unconditionally"
        );
    }

    /// M7: gesture began/ended forward to the event feed in order with
    /// their phases intact -- began maps to `GestureBegan`, ended to
    /// `GestureEnded`, and an unknown (dangling) id changes nothing about
    /// that mapping.
    #[test]
    fn gesture_handlers_emit_phases_in_order() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        wlr::SeatHandler::gesture_began(&mut state, wlr::GestureId::dangling_nth_for_test(0));
        wlr::SeatHandler::gesture_ended(&mut state, wlr::GestureId::dangling_nth_for_test(0));
        let got: Vec<Event> = vec![
            rx.try_recv().expect("began").event,
            rx.try_recv().expect("ended").event,
        ];
        assert_eq!(
            got,
            vec![Event::GestureBegan, Event::GestureEnded],
            "phases must forward began-before-ended and intact"
        );
        assert!(rx.try_recv().is_err(), "nothing else may be emitted");
    }

    /// M7: the switch fold maps the hardware pair to the lid reading --
    /// lid+on is closed, lid+off is open, and any non-lid switch reads
    /// open regardless of position.
    #[test]
    fn apply_switch_toggle_maps_type_and_position_to_lid_closed() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.apply_switch_toggle(wlr::SwitchType::Lid, true);
        state.apply_switch_toggle(wlr::SwitchType::TabletMode, true);
        state.apply_switch_toggle(wlr::SwitchType::Lid, false);
        let got: Vec<Event> = (0..3)
            .map(|_| rx.try_recv().expect("switch event").event)
            .collect();
        assert_eq!(
            got,
            vec![
                Event::SwitchToggled { lid_closed: true },
                Event::SwitchToggled { lid_closed: false },
                Event::SwitchToggled { lid_closed: false },
            ]
        );
    }

    /// M7: the live switch handler without a runtime stays silent -- it
    /// cannot resolve the toggle's type, and guessing would emit a
    /// possibly-wrong lid signal. (Production always has a runtime on this
    /// path; only unit tests drive it without one.)
    #[test]
    fn switch_toggled_without_a_runtime_emits_nothing() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        wlr::SeatHandler::switch_toggled(&mut state, wlr::SwitchId::dangling_nth_for_test(0), true);
        assert!(
            rx.try_recv().is_err(),
            "an unresolvable toggle must stay silent"
        );
        assert!(
            !state.session_locked,
            "a switch must never drive the session lock flag"
        );
    }

    /// M7: pointer motion records the shell-facing cursor mirror -- a
    /// position and (with no runtime to read an image from) a visible
    /// cursor. The constraint gate stays open without a runtime: there is
    /// no live constraint to lock on.
    #[test]
    fn pointer_motion_records_the_cursor_mirror() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        assert_eq!(state.cursor_pos, None);
        assert!(!state.cursor_visible);
        state.pointer_motion(12.0, 34.0, 1);
        assert_eq!(state.cursor_pos, Some((12, 34)));
        assert!(state.cursor_visible);
        assert!(!state.pointer_locked_for_focus());
    }

    /// M7: `GetState` carries the input mirrors (cursor + touch) the shell
    /// renders and holds.
    #[test]
    fn get_state_carries_the_input_mirrors() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut state = State::new(default_config(), tx);
        state.cursor_visible = true;
        state.cursor_pos = Some((7, 9));
        state.touch_active = true;
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        state.handle_command(crate::dbus::DbCommand::GetState(reply_tx));
        let snap = reply_rx.try_recv().expect("GetState reply");
        assert!(snap.cursor_visible);
        assert_eq!(snap.cursor_pos, Some((7, 9)));
        assert!(snap.touch_active);
        // The enrichment is a pure read: nothing new was queued behind it.
        assert!(rx.try_recv().is_err());
    }
}
