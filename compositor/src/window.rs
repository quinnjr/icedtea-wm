use std::collections::BTreeMap;

use contract::{
    Event, Rectangle, SeqEvent, Snapshot, WindowId, WindowInfo, WindowUpdate, WorkspaceInfo,
};
use icedtea_contract as contract;

#[derive(Debug, Clone)]
pub struct Window {
    pub id: WindowId,
    pub app_id: String,
    pub title: String,
    pub pid: u32,
    pub workspace: u32,
    pub geometry: Rectangle,
    pub maximized: bool,
    pub minimized: bool,
    pub fullscreen: bool,
    pub focused: bool,
    /// Whether this window's client currently has a mapped buffer attached.
    /// `true` from `add_window` (the model row is only ever created on a
    /// real map) until an `unmapped` (or, for a model-only row, a caller
    /// through `set_mapped`) says otherwise. Distinct from `minimized`:
    /// minimizing is a compositor-driven, user-visible state a client never
    /// sees reflected in its own protocol state, while unmapping is the
    /// client's own act of detaching its buffer (and can reverse itself by
    /// mapping again, which is why the row survives rather than being
    /// removed). Gates visibility (`is_visible`), alt-tab candidacy
    /// (`alt_tab_entries`), and focus candidacy (`focus`,
    /// `focus_mru_in_workspace`) the same way `minimized` already does.
    pub mapped: bool,
    /// Client's negotiated xdg-decoration mode: Some(true) for ClientSide, Some(false) for ServerSide, None if unset.
    pub client_decorations_requested: Option<bool>,
    /// The window is asking to be noticed without being allowed to take the
    /// keyboard: an `xdg-activation-v1` request this compositor's focus-steal
    /// policy refused (see `State::request_activate`). Purely a shell-facing
    /// hint -- it gates nothing in the model -- and it is cleared the moment
    /// the window actually gains focus (`focus`), which is the only thing
    /// that can answer it.
    pub attention: bool,
}

#[derive(Debug, Clone)]
pub struct Workspace {
    pub id: u32,
    pub name: String,
    pub focused_window: Option<WindowId>,
}

pub struct WindowManager {
    windows: BTreeMap<WindowId, Window>,
    workspaces: Vec<Workspace>,
    active_workspace: u32,
    next_id: u32,
    seq: u64,
    /// Pending events drained by the compositor each frame, each tagged with
    /// the `seq` value its mutation advanced the counter to (review finding
    /// I2 -- see `contract::SeqEvent`).
    pub pending_events: Vec<SeqEvent>,
    /// Focus history: most-recently-focused windows (head = most recent).
    /// Invariant: every WindowId in self.windows is in focus_mru (guaranteed because
    /// add_window always calls focus, and remove_window prunes from focus_mru).
    focus_mru: Vec<WindowId>,
}

impl WindowManager {
    pub fn new(workspace_names: Vec<String>) -> Self {
        // Review finding M2: `workspace_mut` indexes `self.workspaces`
        // directly, so a manager built with an empty name list panicked on
        // the first `add_window`. Production callers are guarded
        // (`load_or_default` rejects an empty list), but `State::new` accepts
        // an arbitrary `Config`, so fall back to a single workspace here
        // rather than leaving a reachable index panic.
        let workspace_names = if workspace_names.is_empty() {
            vec!["1".to_string()]
        } else {
            workspace_names
        };
        let workspaces = workspace_names
            .iter()
            .enumerate()
            .map(|(i, name)| Workspace {
                id: i as u32,
                name: name.clone(),
                focused_window: None,
            })
            .collect();
        Self {
            windows: BTreeMap::new(),
            workspaces,
            active_workspace: 0,
            next_id: 1,
            seq: 0,
            pending_events: Vec::new(),
            focus_mru: Vec::new(),
        }
    }

    fn bump(&mut self) {
        self.seq += 1;
    }

    fn emit(&mut self, event: Event) {
        self.bump();
        let seq = self.seq;
        self.pending_events.push(SeqEvent { seq, event });
    }

    /// Queue an event that didn't come from one of this type's own mutators
    /// (e.g. `AltTabState`, driven by `State::alt_tab`), advancing the same
    /// sequence counter every other event uses.
    pub fn push_event(&mut self, event: Event) {
        self.emit(event);
    }

    pub fn add_window(
        &mut self,
        app_id: &str,
        title: &str,
        pid: u32,
        geometry: Rectangle,
    ) -> WindowId {
        let id = WindowId(self.next_id);
        self.next_id += 1;
        let window = Window {
            id,
            app_id: app_id.to_string(),
            title: title.to_string(),
            pid,
            workspace: self.active_workspace,
            geometry,
            maximized: false,
            minimized: false,
            fullscreen: false,
            focused: false,
            mapped: true,
            client_decorations_requested: None,
            attention: false,
        };
        self.windows.insert(id, window.clone());
        self.emit(Event::WindowOpened(self.to_info(&self.windows[&id])));
        self.focus(id);
        id
    }

    pub fn remove_window(&mut self, id: WindowId) -> Option<Window> {
        let window = self.windows.remove(&id)?;
        let workspace = window.workspace;
        let was_focused = self.workspace_mut(workspace).focused_window == Some(id);
        if was_focused {
            self.workspace_mut(workspace).focused_window = None;
        }
        // Prune from focus MRU to prevent unbounded growth.
        self.focus_mru.retain(|&wid| wid != id);
        self.emit(Event::WindowClosed(id));
        Some(window)
    }

    pub fn get(&self, id: WindowId) -> Option<&Window> {
        self.windows.get(&id)
    }

    pub fn set_title(&mut self, id: WindowId, title: String) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        w.title = title.clone();
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                title: Some(title),
                ..Default::default()
            },
        });
        Some(())
    }

    /// Update a window's `app_id` — the X11 `WM_CLASS` for a managed Xwayland
    /// window whose class arrives (or changes) after it maps. There is no
    /// `WindowUpdate` field for `app_id` in the contract, so this emits
    /// nothing; the change is still observable through the model (a `GetState`
    /// snapshot reads the field fresh) and, more importantly, keeps
    /// `decoration::has_ssd` — which is keyed on `app_id` — in step with the
    /// window's real class. Returns `Some(())` only when the value actually
    /// changed, so the caller can skip a redundant scene resync.
    pub fn set_app_id(&mut self, id: WindowId, app_id: String) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        if w.app_id == app_id {
            return None;
        }
        w.app_id = app_id;
        Some(())
    }

    pub fn set_geometry(&mut self, id: WindowId, geometry: Rectangle) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        w.geometry = geometry;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                geometry: Some(geometry),
                ..Default::default()
            },
        });
        Some(())
    }

    pub fn set_maximized(&mut self, id: WindowId, value: bool) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        w.maximized = value;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                maximized: Some(value),
                ..Default::default()
            },
        });
        Some(())
    }

    pub fn toggle_maximized(&mut self, id: WindowId) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        let new_value = !w.maximized;
        w.maximized = new_value;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                maximized: Some(new_value),
                ..Default::default()
            },
        });
        Some(())
    }

    pub fn set_minimized(&mut self, id: WindowId, value: bool) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        w.minimized = value;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                minimized: Some(value),
                ..Default::default()
            },
        });
        Some(())
    }

    /// Raise or drop `id`'s attention hint. `None` (no emission) on an
    /// unknown id or a value that already matches, like the setters around
    /// it. The compositor sets this when it *refuses* an activation request
    /// (`State::request_activate`); `focus` is what clears it, so a caller
    /// normally only ever passes `true` here.
    pub fn set_attention(&mut self, id: WindowId, value: bool) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        if w.attention == value {
            return None;
        }
        w.attention = value;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                attention: Some(value),
                ..Default::default()
            },
        });
        Some(())
    }

    /// Drop `id`'s attention hint, if it has one -- the named counterpart of
    /// [`Self::set_attention`], which callers otherwise only ever call with
    /// `true`.
    ///
    /// Finding F6: an attention hint is a "look at me until the user does",
    /// and until now only [`Self::focus`] could answer one. So every way a
    /// user could attend to a flagged window *without* focusing it --
    /// restoring it from the taskbar while another window keeps the keyboard,
    /// or switching to the workspace where it already holds the focus pointer
    /// -- left the hint stuck on the shell forever. Those paths call this.
    ///
    /// `None` (no emission) when the window is unknown or already clear, the
    /// same contract `set_attention` has.
    pub fn clear_attention(&mut self, id: WindowId) -> Option<()> {
        self.set_attention(id, false)
    }

    pub fn set_fullscreen(&mut self, id: WindowId, value: bool) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        w.fullscreen = value;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                fullscreen: Some(value),
                ..Default::default()
            },
        });
        Some(())
    }

    pub fn set_client_decorations_requested(
        &mut self,
        id: WindowId,
        value: Option<bool>,
    ) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        if w.client_decorations_requested == value {
            return None;
        }
        w.client_decorations_requested = value;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate::default(),
        });
        Some(())
    }

    /// Flip `id`'s mapped state. `None` (no emission) on an unknown id or a
    /// value that already matches -- the same "no-op on no change" shape
    /// every other setter here follows. `Some(())` otherwise, having emitted
    /// exactly one `WindowUpdated` (the milestone's one sanctioned addition
    /// to the exactly-one-event-per-mutation invariant).
    ///
    /// The row itself is never touched otherwise: unlike `remove_window`,
    /// this leaves geometry, title, and every other field exactly as they
    /// were, because an unmap is not a destroy (a client can map the same
    /// toplevel again). What changes is only what `is_visible`,
    /// `alt_tab_entries`, and `focus`/`focus_mru_in_workspace` are willing to
    /// do with the row while it's unmapped.
    pub fn set_mapped(&mut self, id: WindowId, mapped: bool) -> Option<()> {
        let w = self.windows.get_mut(&id)?;
        if w.mapped == mapped {
            return None;
        }
        w.mapped = mapped;
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                mapped: Some(mapped),
                ..Default::default()
            },
        });
        Some(())
    }

    pub fn set_workspace(&mut self, id: WindowId, workspace: u32) -> Option<()> {
        if !self.workspace_exists(workspace) {
            return None;
        }
        let w = self.windows.get_mut(&id)?;
        let old_workspace = w.workspace;
        let was_focused = w.focused;
        w.workspace = workspace;
        w.focused = false;
        // Clear focus pointer from origin workspace if this window was focused there.
        if was_focused && self.workspace_mut(old_workspace).focused_window == Some(id) {
            self.workspace_mut(old_workspace).focused_window = None;
        }
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                workspace: Some(workspace),
                focused: Some(false),
                ..Default::default()
            },
        });
        Some(())
    }

    pub fn focus(&mut self, id: WindowId) -> Option<()> {
        let (ws, was_focused, was_minimized, mapped) = {
            let w = self.windows.get(&id)?;
            (w.workspace, w.focused, w.minimized, w.mapped)
        };
        // An unmapped window has no client to activate and must never be
        // handed the keyboard: refuse outright rather than remap-safely
        // no-op (decided -- see the task-14 brief's Step 2).
        if !mapped {
            return None;
        }
        if was_focused {
            // Already focused: there is no focus *transition* to emit, but
            // the two side effects a focus carries still apply.
            //
            // The attention clear is not theoretical. `State::request_activate`
            // compares its target against the *active* workspace's focused
            // window (`State::focused_id`), so a window that is the focused
            // window of some INACTIVE workspace is not "already focused" by
            // that test and can be flagged; `switch_workspace` then leaves
            // that workspace's focus pointer alone, so when the user comes
            // back and clicks the window, focus lands right here. Clearing
            // only on the not-yet-focused path below left such a hint stuck
            // on forever.
            let w = self.windows.get_mut(&id)?;
            w.minimized = false;
            let cleared_attention = std::mem::take(&mut w.attention);
            // One combined event, and only when something actually changed.
            if was_minimized || cleared_attention {
                self.emit(Event::WindowUpdated {
                    id,
                    update: WindowUpdate {
                        minimized: was_minimized.then_some(false),
                        attention: cleared_attention.then_some(false),
                        ..Default::default()
                    },
                });
            }
            return Some(());
        }
        // Unminimize if needed.
        if was_minimized {
            let w = self.windows.get_mut(&id)?;
            w.minimized = false;
        }
        // Clear prior focus on the same workspace.
        let old = self.workspace_mut(ws).focused_window.replace(id);
        match old {
            Some(old_id) if old_id != id => {
                if let Some(old_w) = self.windows.get_mut(&old_id) {
                    old_w.focused = false;
                    self.emit(Event::WindowUpdated {
                        id: old_id,
                        update: WindowUpdate {
                            focused: Some(false),
                            ..Default::default()
                        },
                    });
                }
            }
            _ => {}
        }
        let w = self.windows.get_mut(&id)?;
        w.focused = true;
        // Gaining focus answers an attention hint, so it clears one: the
        // shell must not go on flagging a window the user is now looking at.
        // Folded into the focus emission below rather than emitted
        // separately -- one mutation, one event.
        let cleared_attention = std::mem::take(&mut w.attention);
        // One combined event, carrying whichever of the two side effects
        // (an unminimize, an attention clear) this focus actually performed.
        self.emit(Event::WindowUpdated {
            id,
            update: WindowUpdate {
                focused: Some(true),
                minimized: was_minimized.then_some(false),
                attention: cleared_attention.then_some(false),
                ..Default::default()
            },
        });
        // Update focus MRU.
        self.focus_mru.retain(|&wid| wid != id);
        self.focus_mru.insert(0, id);
        Some(())
    }

    /// Drop `id`'s claim on **its own** workspace's focus pointer and hand
    /// focus to that workspace's next MRU candidate, if it has one.
    /// Returns the id that took over, or `None` when `id` did not hold the
    /// pointer or nothing focusable was left behind.
    ///
    /// Review finding I2: the unmap path re-picked a successor only when
    /// the unmapping window was the *active* workspace's focus, so a
    /// focused window that unmapped while its workspace was in the
    /// background left its `focused_window` pointer intact on an unmapped
    /// row. `switch_workspace` re-picks only on `is_none()`, so switching
    /// back restored a focus pointer aimed at an invisible window: the seat
    /// correctly refused it (`sync_seat_focus`/`is_visible_id`), leaving the
    /// keyboard dead, while `apply_action("close"/"maximize"/"fullscreen"/
    /// "snap")` and the decoration actions all still resolved to it.
    ///
    /// The no-successor behavior deliberately matches [`Self::remove_window`]:
    /// the pointer is cleared either way, so "focused" is never left naming a
    /// window that cannot be focused. Workspace visibility is irrelevant here
    /// on purpose -- the workspace this acts on is whichever one `id` is on.
    pub fn release_focus(&mut self, id: WindowId) -> Option<WindowId> {
        let ws = self.windows.get(&id)?.workspace;
        // Indexed through `get_mut`, not `workspace_mut`: this is reachable
        // straight from a wlroots handler (`State::unmapped`), where the
        // panic-free policy applies, and a row naming a workspace the
        // current `workspaces` vec no longer has is exactly the kind of
        // thing a config reload could in principle leave behind.
        let slot = self.workspaces.get_mut(ws as usize)?;
        if slot.focused_window != Some(id) {
            return None;
        }
        slot.focused_window = None;
        // One event for the one mutation: the flag only flips (and only
        // emits) when it was actually set, the same "no-op on no change"
        // shape every setter in this file follows. The successor's own
        // `focus()` below emits its own `focused: true` and, finding the
        // pointer already cleared, emits nothing further for `id`.
        if self.windows.get(&id).is_some_and(|w| w.focused) {
            if let Some(w) = self.windows.get_mut(&id) {
                w.focused = false;
            }
            self.emit(Event::WindowUpdated {
                id,
                update: WindowUpdate {
                    focused: Some(false),
                    ..Default::default()
                },
            });
        }
        self.focus_mru_in_workspace(ws)
    }

    pub fn focused_window(&self) -> Option<&Window> {
        self.workspace(self.active_workspace)
            .and_then(|ws| ws.focused_window)
            .and_then(|id| self.windows.get(&id))
    }

    pub fn set_active_workspace(&mut self, id: u32) -> bool {
        if !self.workspace_exists(id) {
            return false;
        }
        self.active_workspace = id;
        self.emit(Event::WorkspaceSet { id, active: true });
        true
    }

    pub fn active_workspace(&self) -> u32 {
        self.active_workspace
    }

    pub fn windows(&self) -> impl Iterator<Item = &Window> {
        // Return windows ordered by focus MRU (most recent first).
        // Invariant: every window in self.windows is in focus_mru, so filter_map is safe.
        self.focus_mru
            .iter()
            .filter_map(move |id| self.windows.get(id))
    }

    pub fn windows_in_workspace(&self, ws: u32) -> Vec<&Window> {
        self.windows
            .values()
            .filter(|w| w.workspace == ws)
            .collect()
    }

    /// The windows that should actually be drawn and hit-tested right now:
    /// the active workspace's non-minimized windows, topmost (most recently
    /// focused) first.
    ///
    /// Review finding I1: rendering and click-to-focus both consumed the raw
    /// `windows()` MRU list -- every window on *every* workspace, minimized
    /// ones included -- so switching workspaces changed nothing on screen and
    /// a click could focus (or close) a window belonging to an inactive
    /// workspace.
    pub fn visible_windows(&self) -> Vec<&Window> {
        self.windows().filter(|w| self.is_visible(w)).collect()
    }

    /// Whether `w` belongs to the active workspace, isn't minimized, and is
    /// mapped (see `visible_windows`).
    pub fn is_visible(&self, w: &Window) -> bool {
        w.workspace == self.active_workspace && !w.minimized && w.mapped
    }

    /// Same predicate as `is_visible`, by id (`false` for an unknown id).
    pub fn is_visible_id(&self, id: WindowId) -> bool {
        self.get(id).is_some_and(|w| self.is_visible(w))
    }

    /// The topmost visible window containing `point` (output logical
    /// coordinates), i.e. what a click at that point acts on. Used by the
    /// backend's click-to-focus path (review finding I1).
    pub fn window_at(&self, point: (i32, i32)) -> Option<&Window> {
        self.visible_windows()
            .into_iter()
            .find(|w| w.geometry.contains(point.0, point.1))
    }

    /// Focus the most-recently-focused non-minimized, mapped window on `ws`,
    /// if any; when nothing qualifies, clear `ws`'s focus pointer instead of
    /// leaving it aimed at a window that just became hidden (minimized,
    /// unmapped, closed, or moved off the workspace). Mirrors
    /// `remove_window`'s no-successor clear, so "focused" never goes on
    /// naming a window nothing can act on. Returns the id that took over, or
    /// `None`.
    ///
    /// Every hide path -- minimize, toplevel unmap, close, and a workspace
    /// switch landing on a stale/invisible pointer -- routes through this
    /// rather than discarding `focus_mru_in_workspace`'s (this function's
    /// former name and still its candidate-picking half) result, which is
    /// what let a hidden window remain `focused_window()` when it was the
    /// workspace's last visible one.
    pub fn refocus_after_hide(&mut self, ws: u32) -> Option<WindowId> {
        let candidate = self.focus_mru.iter().copied().find(|id| {
            self.windows
                .get(id)
                .is_some_and(|w| w.workspace == ws && !w.minimized && w.mapped)
        });
        match candidate {
            Some(id) => {
                self.focus(id)?;
                Some(id)
            }
            None => {
                let cleared = self
                    .workspaces
                    .get_mut(ws as usize)
                    .and_then(|slot| slot.focused_window.take());
                // The pointer is only half the story: the hidden window's own
                // `focused` field (what `to_info`/`snapshot` report to IPC
                // clients, e.g. the taskbar) must drop too, with exactly one
                // `WindowUpdated{focused:false}` for it -- mirrors how the
                // successful-candidate arm above (via `focus`) and
                // `remove_window`'s no-successor clear both handle the flag
                // and its emission. Guarded on `w.focused` so this stays a
                // no-op-on-no-change like every other setter here (the
                // pointer and the flag can already disagree, e.g. a window
                // unmapped without ever having been re-focused).
                if let Some(prev_id) =
                    cleared.filter(|id| self.windows.get(id).is_some_and(|w| w.focused))
                {
                    if let Some(w) = self.windows.get_mut(&prev_id) {
                        w.focused = false;
                    }
                    self.emit(Event::WindowUpdated {
                        id: prev_id,
                        update: WindowUpdate {
                            focused: Some(false),
                            ..Default::default()
                        },
                    });
                }
                None
            }
        }
    }

    /// Focus the most-recently-focused non-minimized window on `ws`, if any.
    ///
    /// Review finding I6: `set_workspace` cleared the *origin* workspace's
    /// focus pointer but never gave the destination one, so
    /// `MoveToWorkspace` (which then switches to that workspace) left
    /// `focused_window()` as `None` and the very next `close`/`fullscreen`/
    /// `snap` action silently no-opped.
    ///
    /// Thin alias kept for call sites that only care about the picked
    /// candidate; see [`Self::refocus_after_hide`] for the no-candidate
    /// behavior (it now also clears the pointer on `None`).
    pub fn focus_mru_in_workspace(&mut self, ws: u32) -> Option<WindowId> {
        self.refocus_after_hide(ws)
    }

    pub fn to_info(&self, w: &Window) -> WindowInfo {
        WindowInfo {
            id: w.id,
            app_id: w.app_id.clone(),
            title: w.title.clone(),
            pid: w.pid,
            workspace: w.workspace,
            geometry: w.geometry,
            maximized: w.maximized,
            minimized: w.minimized,
            fullscreen: w.fullscreen,
            focused: w.focused,
            attention: w.attention,
        }
    }

    pub fn workspace_info(&self) -> Vec<WorkspaceInfo> {
        self.workspaces
            .iter()
            .map(|w| WorkspaceInfo {
                id: w.id,
                name: w.name.clone(),
            })
            .collect()
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            seq: self.seq,
            windows: self.windows.values().map(|w| self.to_info(w)).collect(),
            workspaces: self.workspace_info(),
            active_workspace: self.active_workspace,
            // The model carries no input state: `State::handle_command`'s
            // `GetState` arm fills the indicator fields from the live
            // runtime. No runtime (every model-only unit test) reads as
            // inactive / no layout / uninhibited.
            ime_active: false,
            keyboard_layout: None,
            shortcuts_inhibited: false,
            // M7 input mirrors live on `State`, not on this model (which
            // has no runtime): defaults here, enriched by
            // `State::handle_command`'s `GetState` arm before the reply.
            cursor_visible: false,
            cursor_pos: None,
            touch_active: false,
        }
    }

    pub fn alt_tab_entries(&self) -> Vec<WindowId> {
        self.windows_in_workspace(self.active_workspace)
            .into_iter()
            .filter(|w| !w.minimized && w.mapped)
            .map(|w| w.id)
            .collect()
    }

    /// The id the next `add_window` call will assign.
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Raise the id counter to at least `min_next_id`, never lowering it.
    /// Used by `State::apply_config` (Task 11 review #3): that method
    /// discards all windows and rebuilds a fresh `WindowManager`, but must
    /// not let the fresh instance start reissuing ids from 1 -- a shell that
    /// hasn't yet processed the `WindowClosed` events for the old windows
    /// could otherwise see a brand-new window claim an id it still believes
    /// is live.
    pub fn raise_id_floor(&mut self, min_next_id: u32) {
        if min_next_id > self.next_id {
            self.next_id = min_next_id;
        }
    }

    /// The current snapshot sequence number (`Snapshot::seq`).
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Raise the sequence counter to at least `min_seq`, never lowering it.
    /// Same rationale as `raise_id_floor`: a fresh `WindowManager` built by
    /// `State::apply_config` starts `seq` back at 0, which would make
    /// `snapshot().seq` go backwards across a reload -- a subscriber
    /// comparing sequence numbers to detect missed updates would wrongly
    /// conclude nothing changed (or that time ran backwards).
    pub fn raise_seq_floor(&mut self, min_seq: u64) {
        if min_seq > self.seq {
            self.seq = min_seq;
        }
    }

    /// Reconcile the workspace list against a new set of `names` *in place*,
    /// without dropping any window row. Existing workspaces are renamed
    /// (keyed by index, which is their id, so a surviving window keeps its
    /// assignment); extra names append fresh empty workspaces; trailing
    /// workspaces beyond the new list are truncated. Any window still
    /// assigned to a now-removed index migrates to workspace 0, emitting one
    /// `WindowUpdated` per migrated row.
    ///
    /// This is the config-reload path (`State::apply_config`). It replaces
    /// the old "rebuild a brand-new `WindowManager`" behavior, which closed
    /// every client on every reload: here ids, focus MRU, geometry, and all
    /// per-window state survive.
    pub fn set_workspace_names(&mut self, names: Vec<String>) {
        // Match `new`'s guard: never leave a zero-workspace manager, which
        // would panic `workspace_mut` on the next `add_window`.
        let names = if names.is_empty() {
            vec!["1".to_string()]
        } else {
            names
        };
        let new_len = names.len() as u32;

        // Migrate any window off a workspace index that is about to vanish.
        // Collect ids first so we don't borrow `windows` while mutating it.
        if new_len < self.workspaces.len() as u32 {
            let migrants: Vec<WindowId> = self
                .windows
                .values()
                .filter(|w| w.workspace >= new_len)
                .map(|w| w.id)
                .collect();
            for id in migrants {
                if let Some(w) = self.windows.get_mut(&id) {
                    w.workspace = 0;
                    w.focused = false;
                }
                self.emit(Event::WindowUpdated {
                    id,
                    update: WindowUpdate {
                        workspace: Some(0),
                        focused: Some(false),
                        ..Default::default()
                    },
                });
            }
        }

        // Rename existing, append new, truncate removed -- all keyed by index
        // (== workspace id), so surviving windows keep their assignment.
        self.workspaces.truncate(new_len as usize);
        for (i, name) in names.into_iter().enumerate() {
            match self.workspaces.get_mut(i) {
                Some(ws) => ws.name = name,
                None => self.workspaces.push(Workspace {
                    id: i as u32,
                    name,
                    focused_window: None,
                }),
            }
        }

        // The active workspace may have been truncated away; clamp it back
        // into range so `workspace_mut`/`focused_window` stay panic-free.
        // This is a real change to `active_workspace`, and `WorkspaceList`
        // does not carry the active index, so emit the same
        // `WorkspaceSet{active: true}` every other active-workspace change
        // emits -- otherwise subscribers keep showing the vanished workspace
        // until a manual switch.
        if self.active_workspace >= new_len {
            self.active_workspace = 0;
            self.emit(Event::WorkspaceSet {
                id: 0,
                active: true,
            });
        }
    }

    fn workspace(&self, id: u32) -> Option<&Workspace> {
        self.workspaces.get(id as usize)
    }

    fn workspace_mut(&mut self, id: u32) -> &mut Workspace {
        &mut self.workspaces[id as usize]
    }

    fn workspace_exists(&self, id: u32) -> bool {
        (id as usize) < self.workspaces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> WindowManager {
        WindowManager::new(vec!["1".into(), "2".into()])
    }

    const GEO: Rectangle = Rectangle {
        x: 0,
        y: 0,
        width: 640,
        height: 400,
    };

    #[test]
    fn add_focuses_window_and_emits_opened() {
        let mut m = mgr();
        let id = m.add_window("app", "title", 1, GEO);
        assert!(m.get(id).unwrap().focused);
        assert!(matches!(
            m.pending_events.first(),
            Some(SeqEvent {
                event: Event::WindowOpened(_),
                ..
            })
        ));
    }

    #[test]
    fn focus_unfocuses_previous() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        assert!(!m.get(a).unwrap().focused);
        assert!(m.get(b).unwrap().focused);
    }

    #[test]
    fn move_to_workspace_keeps_focus_valid() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        m.set_workspace(a, 1).unwrap();
        assert_eq!(m.get(a).unwrap().workspace, 1);
        assert!(!m.get(a).unwrap().focused);
        // Verify focused_window() is cleared after move (stale pointer check).
        assert!(m.focused_window().is_none());
    }

    #[test]
    fn remove_clears_focus_to_none() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        m.remove_window(a).unwrap();
        assert!(m.focused_window().is_none());
    }

    #[test]
    fn snapshot_matches_state() {
        let mut m = mgr();
        let id = m.add_window("app", "t", 7, GEO);
        let snap = m.snapshot();
        assert_eq!(snap.active_workspace, 0);
        assert_eq!(snap.windows.len(), 1);
        assert_eq!(snap.windows[0].id, id);
        assert_eq!(snap.windows[0].pid, 7);
        assert_eq!(snap.workspaces.len(), 2);
    }

    #[test]
    fn alt_tab_skips_minimized() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        m.set_minimized(a, true).unwrap();
        assert_eq!(m.alt_tab_entries(), vec![b]);
    }

    #[test]
    fn focus_minimized_already_focused_emits_event() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        // Minimize while focused.
        m.set_minimized(a, true).unwrap();
        m.pending_events.clear();
        // Focus again (already focused, but minimized).
        m.focus(a).unwrap();
        // Should emit an event for the minimized state change.
        assert!(!m.pending_events.is_empty());
        assert!(matches!(
            m.pending_events.first(),
            Some(SeqEvent {
                event: Event::WindowUpdated { .. },
                ..
            })
        ));
        assert!(!m.get(a).unwrap().minimized);
    }

    #[test]
    fn windows_ordered_by_mru() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        let c = m.add_window("c", "c", 3, GEO);
        // Current order (MRU first): c, b, a
        let ids: Vec<_> = m.windows().map(|w| w.id).collect();
        assert_eq!(ids, vec![c, b, a]);
        // Focus a, should move to front.
        m.focus(a).unwrap();
        let ids: Vec<_> = m.windows().map(|w| w.id).collect();
        assert_eq!(ids, vec![a, c, b]);
    }

    #[test]
    fn remove_window_prunes_focus_mru() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        let c = m.add_window("c", "c", 3, GEO);
        // MRU order: c, b, a
        assert_eq!(m.windows().map(|w| w.id).collect::<Vec<_>>(), vec![c, b, a]);
        // Remove b (middle of MRU).
        m.remove_window(b).unwrap();
        // Should have c, a in MRU order.
        assert_eq!(m.windows().map(|w| w.id).collect::<Vec<_>>(), vec![c, a]);
        // Add new window d, should be MRU first.
        let d = m.add_window("d", "d", 4, GEO);
        assert_eq!(m.windows().map(|w| w.id).collect::<Vec<_>>(), vec![d, c, a]);
        // Verify b is not in the state at all.
        assert!(m.get(b).is_none());
    }

    // --- Final-review fix-round tests ---

    /// M2: a manager built from a config with no workspace names must not
    /// leave a reachable index panic in `workspace_mut`.
    #[test]
    fn empty_workspace_list_falls_back_to_one_workspace() {
        let mut m = WindowManager::new(vec![]);
        assert_eq!(m.workspace_info().len(), 1);
        let id = m.add_window("a", "a", 1, GEO);
        assert!(m.get(id).unwrap().focused);
    }

    /// I1: only the active workspace's non-minimized windows are drawn and
    /// hit-tested, topmost (MRU) first.
    #[test]
    fn visible_windows_filters_by_workspace_and_minimized() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        let c = m.add_window("c", "c", 3, GEO);
        m.set_workspace(c, 1).unwrap();
        m.set_minimized(b, true).unwrap();

        assert_eq!(
            m.visible_windows().iter().map(|w| w.id).collect::<Vec<_>>(),
            vec![a]
        );
        assert!(m.is_visible_id(a));
        assert!(
            !m.is_visible_id(b),
            "minimized windows are not drawn or clickable"
        );
        assert!(
            !m.is_visible_id(c),
            "another workspace's windows are not drawn or clickable"
        );

        m.set_active_workspace(1);
        assert_eq!(
            m.visible_windows().iter().map(|w| w.id).collect::<Vec<_>>(),
            vec![c]
        );
    }

    /// I1: a click must never land on a window from an inactive workspace,
    /// even when its geometry contains the point.
    #[test]
    fn window_at_only_hits_visible_windows() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        // Both cover (10, 10); `b` is MRU head, so it wins while visible.
        assert_eq!(m.window_at((10, 10)).map(|w| w.id), Some(b));
        m.set_workspace(b, 1).unwrap();
        assert_eq!(m.window_at((10, 10)).map(|w| w.id), Some(a));
        m.set_minimized(a, true).unwrap();
        assert!(m.window_at((10, 10)).is_none());
        assert!(m.window_at((10_000, 10_000)).is_none());
    }

    /// Task 6: minimizing (or otherwise hiding) the sole window on a
    /// workspace must not leave `focused_window()` still naming it -- the
    /// model must never report a hidden window as focused.
    #[test]
    fn hiding_the_sole_window_clears_the_focus_pointer() {
        let mut m = WindowManager::new(vec!["1".into()]);
        let a = m.add_window(
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
        assert_eq!(m.focused_window().map(|w| w.id), Some(a));
        m.set_minimized(a, true);
        let seq_before = m.seq();
        assert_eq!(m.refocus_after_hide(m.active_workspace()), None);
        assert!(
            m.focused_window().is_none(),
            "no hidden window may remain focused"
        );
        // The reporting boundary must agree: `a`'s own `focused` field --
        // what `to_info`/`snapshot` hand IPC clients like the taskbar -- has
        // to drop too, with exactly one `WindowUpdated{focused:false}`
        // emitted for it, not just the workspace pointer clearing.
        assert!(
            !m.get(a).unwrap().focused,
            "the hidden window's own focused flag must clear too"
        );
        let emitted_unfocus = m
            .pending_events
            .iter()
            .filter(|e| e.seq > seq_before)
            .any(|e| matches!(&e.event, Event::WindowUpdated { id, update } if *id == a && update.focused == Some(false)));
        assert!(
            emitted_unfocus,
            "must emit exactly one WindowUpdated{{focused:false}} for the hidden window"
        );
    }

    /// I6: after a window is moved away, the workspace it lands on has a
    /// focusable head that the next action can act on.
    #[test]
    fn focus_mru_in_workspace_picks_head_and_skips_minimized() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        m.set_workspace(b, 1).unwrap();
        // Workspace 1 now holds only `b`, unfocused.
        assert_eq!(m.focus_mru_in_workspace(1), Some(b));
        assert!(m.get(b).unwrap().focused);

        // Minimized windows aren't focus candidates; an empty workspace
        // reports `None` rather than focusing something on another one.
        m.set_minimized(b, true).unwrap();
        assert_eq!(m.focus_mru_in_workspace(1), None);
        assert_eq!(m.focus_mru_in_workspace(0), Some(a));
    }

    /// I2: every queued event carries the seq its mutation produced, and
    /// those seqs are strictly increasing.
    #[test]
    fn pending_events_carry_increasing_seq() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        m.set_title(a, "new".into()).unwrap();
        let seqs: Vec<u64> = m.pending_events.iter().map(|e| e.seq).collect();
        assert!(seqs.len() >= 2);
        assert!(
            seqs.windows(2).all(|w| w[1] > w[0]),
            "seqs must strictly increase: {seqs:?}"
        );
        assert_eq!(
            *seqs.last().unwrap(),
            m.seq(),
            "the last queued event carries the current seq"
        );
    }

    // --- Task 14: the model-level "unmapped" concept ---

    /// Unmapping a window keeps its row (title, geometry, etc. all
    /// untouched) but pulls it out of visibility and alt-tab candidacy;
    /// remapping restores both. The value not changing is a no-op.
    #[test]
    fn an_unmapped_window_leaves_visibility_and_alt_tab_but_keeps_its_row() {
        let mut m = WindowManager::new(vec!["1".into()]);
        let a = m.add_window(
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
        let b = m.add_window(
            "b",
            "b",
            1,
            Rectangle {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
        );
        assert_eq!(m.set_mapped(b, false), Some(()));
        assert!(!m.get(b).expect("row kept").mapped);
        assert!(!m.alt_tab_entries().contains(&b));
        assert!(m.visible_windows().iter().all(|w| w.id != b));
        assert_eq!(m.set_mapped(b, false), None, "unchanged value is a no-op");
        assert_eq!(m.set_mapped(b, true), Some(()));
        assert!(m.alt_tab_entries().contains(&b));
        let _ = a;
    }

    /// The milestone's one sanctioned addition to the exactly-one-event-
    /// per-mutation invariant: `set_mapped` emits exactly one `WindowUpdated`.
    #[test]
    fn set_mapped_emits_exactly_one_window_updated() {
        let mut m = WindowManager::new(vec!["1".into()]);
        let a = m.add_window(
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
        let before = m.seq();
        m.set_mapped(a, false);
        assert_eq!(m.seq(), before + 1, "exactly one emission");
    }

    /// Step 2's focus-refusal decision: focusing an unmapped window is
    /// refused outright, not a remap-safe no-op.
    #[test]
    fn focus_refuses_an_unmapped_window() {
        let mut m = WindowManager::new(vec!["1".into()]);
        let a = m.add_window("a", "a", 1, GEO);
        m.set_mapped(a, false).unwrap();
        assert_eq!(m.focus(a), None, "an unmapped window must never gain focus");
    }

    // --- Task 9: D-Bus observability of mapped state + decoration mode ---

    /// A decoration-mode change is a new sanctioned `WindowUpdated` emission:
    /// exactly one on an actual value change, none on a no-op repeat.
    #[test]
    fn set_client_decorations_requested_emits_one_window_updated() {
        let mut m = WindowManager::new(vec!["1".into()]);
        let a = m.add_window(
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
        let before = m.seq();
        m.set_client_decorations_requested(a, Some(true));
        assert_eq!(m.seq(), before + 1, "exactly one emission");
        // and a no-op (unchanged value) emits nothing:
        let mid = m.seq();
        m.set_client_decorations_requested(a, Some(true));
        assert_eq!(m.seq(), mid, "unchanged value is silent");
    }

    /// `set_mapped`'s existing emission now carries the new `mapped` field.
    #[test]
    fn set_mapped_payload_carries_mapped() {
        let mut m = WindowManager::new(vec!["1".into()]);
        let a = m.add_window(
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
        m.pending_events.clear(); // clear the add's events
        m.set_mapped(a, false);
        let carried = m.pending_events.iter().any(
            |e| matches!(&e.event, Event::WindowUpdated { id, update } if *id == a && update.mapped == Some(false)),
        );
        assert!(
            carried,
            "set_mapped's emission must carry update.mapped == Some(false)"
        );
    }

    /// A2 task 8: `set_attention` emits the additive `attention` field and
    /// the snapshot reads the model's real value rather than a hard-coded
    /// `false`.
    #[test]
    fn set_attention_emits_update_and_shows_in_snapshot() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        assert!(!m.get(a).unwrap().attention, "attention defaults to false");
        m.pending_events.clear();

        assert_eq!(m.set_attention(a, true), Some(()));
        let carried = m.pending_events.iter().any(
            |e| matches!(&e.event, Event::WindowUpdated { id, update } if *id == a && update.attention == Some(true)),
        );
        assert!(
            carried,
            "set_attention's emission must carry update.attention == Some(true)"
        );
        assert!(m.get(a).unwrap().attention);
        let info = m
            .snapshot()
            .windows
            .into_iter()
            .find(|w| w.id == a)
            .unwrap();
        assert!(
            info.attention,
            "snapshot must reflect the model's attention flag"
        );

        // Setting the same value again is silent, like every other setter here.
        let before = m.seq();
        assert_eq!(m.set_attention(a, true), None);
        assert_eq!(m.seq(), before, "unchanged value is silent");
    }

    /// A2 task 8: gaining focus clears attention, folded into the very
    /// `focused: Some(true)` update rather than a second event.
    #[test]
    fn focus_clears_attention_in_the_focused_update() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let b = m.add_window("b", "b", 2, GEO);
        // `b` is focused; mark the background window `a` as needing attention.
        m.set_attention(a, true).unwrap();
        assert!(m.get(a).unwrap().attention);
        m.pending_events.clear();

        m.focus(a).unwrap();

        assert!(!m.get(a).unwrap().attention, "focus must clear attention");
        let updates: Vec<_> = m
            .pending_events
            .iter()
            .filter_map(|e| match &e.event {
                Event::WindowUpdated { id, update } if *id == a => Some(update),
                _ => None,
            })
            .collect();
        assert_eq!(
            updates.len(),
            1,
            "exactly one update for the newly focused window"
        );
        assert_eq!(updates[0].focused, Some(true));
        assert_eq!(
            updates[0].attention,
            Some(false),
            "the focused update must clear attention"
        );
        assert!(!m.get(b).unwrap().focused);
    }

    /// Final-review finding 2: attention must also clear on the
    /// *already-focused* path. A window that holds its own workspace's
    /// focus pointer can still be flagged (`State::request_activate` tests
    /// only the ACTIVE workspace's focused window), and the user's eventual
    /// click on it takes `focus`'s `was_focused` early return -- which used
    /// to leave the hint set forever.
    #[test]
    fn focus_of_an_already_focused_window_still_clears_attention() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        assert!(
            m.get(a).unwrap().focused,
            "the only window in its workspace holds focus"
        );
        m.set_attention(a, true).unwrap();
        assert!(m.get(a).unwrap().attention);
        m.pending_events.clear();

        m.focus(a).unwrap();

        assert!(
            !m.get(a).unwrap().attention,
            "re-focusing an already-focused window must clear attention"
        );
        let updates: Vec<_> = m
            .pending_events
            .iter()
            .filter_map(|e| match &e.event {
                Event::WindowUpdated { id, update } if *id == a => Some(update),
                _ => None,
            })
            .collect();
        assert_eq!(
            updates.len(),
            1,
            "the clear owes exactly one update, got {updates:?}"
        );
        assert_eq!(
            updates[0].attention,
            Some(false),
            "the update must carry the attention clear"
        );
        let info = m
            .snapshot()
            .windows
            .into_iter()
            .find(|w| w.id == a)
            .unwrap();
        assert!(!info.attention, "snapshot must show the cleared flag");
    }

    /// The already-focused path stays silent when there is nothing to
    /// clear: no attention, not minimized, no event.
    #[test]
    fn focus_of_an_already_focused_clean_window_emits_nothing() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        m.pending_events.clear();
        m.focus(a).unwrap();
        assert!(
            m.pending_events.is_empty(),
            "a no-op focus must not emit, got {:?}",
            m.pending_events
        );
    }

    /// Focusing a window that had no attention set must not advertise a
    /// clear it never needed.
    #[test]
    fn focus_without_attention_omits_the_field() {
        let mut m = mgr();
        let a = m.add_window("a", "a", 1, GEO);
        let _b = m.add_window("b", "b", 2, GEO);
        m.pending_events.clear();
        m.focus(a).unwrap();
        let update = m
            .pending_events
            .iter()
            .find_map(|e| match &e.event {
                Event::WindowUpdated { id, update } if *id == a => Some(update),
                _ => None,
            })
            .unwrap();
        assert_eq!(update.focused, Some(true));
        assert_eq!(update.attention, None);
    }
}
