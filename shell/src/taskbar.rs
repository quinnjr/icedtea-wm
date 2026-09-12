//! The pure taskbar model: the window list, workspaces, and active workspace,
//! folded from `org.icedtea.Compositor` traffic. No GTK — `panel::view`
//! renders it.

use icedtea_contract::{Snapshot, WindowInfo, WindowUpdate, WorkspaceInfo};

#[derive(Default, Debug, Clone)]
pub struct TaskbarModel {
    pub windows: Vec<WindowInfo>,
    pub workspaces: Vec<WorkspaceInfo>,
    pub active_workspace: u32,
    /// M7: whether any touch point is down, folded from the seed
    /// snapshot's `touch_active`. Rendered by the panel as its touch
    /// indicator.
    pub touch_active: bool,
    /// M7: whether a pointer gesture is in flight (began without an end).
    /// Folded from the gesture signals; held, not rendered -- gestures are
    /// signals, not widgets (spec section 6), so the state exists to be
    /// queried, not painted.
    pub gesture_active: bool,
    /// M7: whether the lid is closed, folded from the switch signal. Held,
    /// not rendered, for the same reason as `gesture_active`: the session
    /// (not the panel) decides what a closed lid means.
    pub lid_closed: bool,
}

/// The subset of `org.icedtea.Compositor` traffic the taskbar folds in. The worker
/// (`compositor_client`) translates D-Bus signals + the seed snapshot into these.
///
/// `Clone` because the update crosses a thread boundary inside an `Arc` on the
/// way to `panel::update` (contract §3.1 makes `Msg` `Send`), which unwraps it
/// with `Arc::unwrap_or_clone`.
#[derive(Debug, Clone)]
pub enum CompositorUpdate {
    Snapshot(Snapshot),
    Opened(WindowInfo),
    Closed(u32),
    Updated {
        id: u32,
        update: WindowUpdate,
    },
    WorkspaceSet {
        id: u32,
        active: bool,
    },
    WorkspaceList(Vec<WorkspaceInfo>),
    /// M7: a pointer gesture began (translated from the `GestureBegan`
    /// D-Bus signal).
    GestureBegan,
    /// M7: the in-flight gesture ended (from `GestureEnded`).
    GestureEnded,
    /// M7: a switch toggled (from `SwitchToggled`); `lid_closed` is the
    /// session reading the compositor derived from the `(type, on)` pair.
    SwitchToggled {
        lid_closed: bool,
    },
}

impl TaskbarModel {
    pub fn apply(&mut self, u: CompositorUpdate) {
        match u {
            CompositorUpdate::Snapshot(s) => {
                self.windows = s.windows;
                self.workspaces = s.workspaces;
                self.active_workspace = s.active_workspace;
                // M7 lockstep: the seed snapshot carries the touch mirror
                // the panel renders. Gesture/lid state is signal-only (no
                // snapshot field), so a seed leaves whatever the signals
                // established alone.
                self.touch_active = s.touch_active;
            }
            CompositorUpdate::Opened(w) => match self.windows.iter_mut().find(|x| x.id == w.id) {
                Some(existing) => *existing = w,
                None => self.windows.push(w),
            },
            CompositorUpdate::Closed(id) => self.windows.retain(|w| w.id.0 != id),
            CompositorUpdate::Updated { id, update } => {
                if let Some(w) = self.windows.iter_mut().find(|x| x.id.0 == id) {
                    merge(w, update);
                }
            }
            CompositorUpdate::WorkspaceSet { id, active } => {
                if active {
                    self.active_workspace = id;
                }
            }
            CompositorUpdate::WorkspaceList(ws) => self.workspaces = ws,
            CompositorUpdate::GestureBegan => self.gesture_active = true,
            CompositorUpdate::GestureEnded => self.gesture_active = false,
            CompositorUpdate::SwitchToggled { lid_closed } => self.lid_closed = lid_closed,
        }
    }
}

fn merge(w: &mut WindowInfo, u: WindowUpdate) {
    if let Some(t) = u.title {
        w.title = t;
    }
    if let Some(g) = u.geometry {
        w.geometry = g;
    }
    if let Some(ws) = u.workspace {
        w.workspace = ws;
    }
    if let Some(m) = u.maximized {
        w.maximized = m;
    }
    if let Some(m) = u.minimized {
        w.minimized = m;
    }
    if let Some(f) = u.fullscreen {
        w.fullscreen = f;
    }
    if let Some(f) = u.focused {
        w.focused = f;
    }
    if let Some(a) = u.attention {
        w.attention = a;
    }
    // `mapped` has no field on WindowInfo; the taskbar ignores it.
}

#[cfg(test)]
mod tests {
    use super::*;
    use icedtea_contract::{Rectangle, WindowId};

    fn win(id: u32, app: &str) -> WindowInfo {
        WindowInfo {
            id: WindowId(id),
            app_id: app.into(),
            title: app.into(),
            pid: 0,
            workspace: 0,
            geometry: Rectangle {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            maximized: false,
            minimized: false,
            fullscreen: false,
            focused: false,
            attention: false,
        }
    }

    fn title_update(t: &str) -> WindowUpdate {
        WindowUpdate {
            title: Some(t.into()),
            ..Default::default()
        }
    }

    #[test]
    fn opened_then_closed_tracks_the_window_set() {
        let mut m = TaskbarModel::default();
        m.apply(CompositorUpdate::Opened(win(1, "a")));
        m.apply(CompositorUpdate::Opened(win(2, "b")));
        assert_eq!(m.windows.len(), 2);
        m.apply(CompositorUpdate::Closed(1));
        assert_eq!(
            m.windows.iter().map(|w| w.id.0).collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn opened_with_a_known_id_replaces_rather_than_duplicates() {
        let mut m = TaskbarModel::default();
        m.apply(CompositorUpdate::Opened(win(1, "a")));
        m.apply(CompositorUpdate::Opened(win(1, "a-again")));
        assert_eq!(m.windows.len(), 1);
        assert_eq!(m.windows[0].app_id, "a-again");
    }

    #[test]
    fn updated_merges_title() {
        let mut m = TaskbarModel::default();
        m.apply(CompositorUpdate::Opened(win(1, "a")));
        m.apply(CompositorUpdate::Updated {
            id: 1,
            update: title_update("renamed"),
        });
        assert_eq!(m.windows[0].title, "renamed");
    }

    /// Final-review finding 3: the additive `attention` bit is part of the
    /// update stream the taskbar folds in, and it must round-trip in both
    /// directions -- the compositor clears it (on focus) as well as raising
    /// it, and a merge that ignored the field left a button flagged forever.
    #[test]
    fn updated_merges_attention_in_both_directions() {
        let mut m = TaskbarModel::default();
        m.apply(CompositorUpdate::Opened(win(1, "a")));
        assert!(!m.windows[0].attention, "attention starts clear");

        m.apply(CompositorUpdate::Updated {
            id: 1,
            update: WindowUpdate {
                attention: Some(true),
                ..Default::default()
            },
        });
        assert!(
            m.windows[0].attention,
            "a raised attention hint must reach the taskbar model"
        );

        // An unrelated update must not disturb it.
        m.apply(CompositorUpdate::Updated {
            id: 1,
            update: title_update("renamed"),
        });
        assert!(
            m.windows[0].attention,
            "an unrelated update must leave attention alone"
        );

        m.apply(CompositorUpdate::Updated {
            id: 1,
            update: WindowUpdate {
                attention: Some(false),
                ..Default::default()
            },
        });
        assert!(
            !m.windows[0].attention,
            "the compositor's clear must reach the taskbar model too"
        );
    }

    /// M5: the update stream crosses a thread boundary as `Arc<CompositorUpdate>`
    /// (contract §3.1 makes `Msg` `Send`), and `update` unwraps it with
    /// `Arc::unwrap_or_clone`. That needs `Clone`, and a payload that silently
    /// stopped deriving it would only show up in `panel.rs`.
    #[test]
    fn a_compositor_update_can_be_cloned_out_of_an_arc() {
        let update = CompositorUpdate::Opened(win(1, "a"));
        let shared = std::sync::Arc::new(update);
        let mut m = TaskbarModel::default();
        m.apply(std::sync::Arc::unwrap_or_clone(shared.clone()));
        m.apply(std::sync::Arc::unwrap_or_clone(shared));
        assert_eq!(m.windows.len(), 1, "the same update applied twice upserts");
    }

    #[test]
    fn workspace_set_tracks_active_only_when_active() {
        let mut m = TaskbarModel::default();
        m.apply(CompositorUpdate::WorkspaceSet {
            id: 3,
            active: true,
        });
        assert_eq!(m.active_workspace, 3);
        m.apply(CompositorUpdate::WorkspaceSet {
            id: 5,
            active: false,
        });
        assert_eq!(
            m.active_workspace, 3,
            "an inactive set must not move the active workspace"
        );
    }

    /// M7: the seed snapshot's `touch_active` reaches the model (the
    /// panel's indicator reads this), while signal-only state survives a
    /// re-seed untouched.
    #[test]
    fn snapshot_folds_touch_active_and_leaves_signal_state_alone() {
        use icedtea_contract::Snapshot;

        let mut m = TaskbarModel::default();
        m.apply(CompositorUpdate::GestureBegan);
        m.apply(CompositorUpdate::SwitchToggled { lid_closed: true });
        m.apply(CompositorUpdate::Snapshot(Snapshot {
            seq: 1,
            windows: vec![],
            workspaces: vec![],
            active_workspace: 0,
            ime_active: false,
            keyboard_layout: None,
            shortcuts_inhibited: false,
            cursor_visible: true,
            cursor_pos: Some((4, 4)),
            touch_active: true,
        }));
        assert!(
            m.touch_active,
            "the seed's touch_active must reach the model"
        );
        assert!(
            m.gesture_active,
            "a seed must not clear signal-established gesture state"
        );
        assert!(
            m.lid_closed,
            "a seed must not clear signal-established lid state"
        );
    }

    /// M7: gesture began/ended fold as an in-flight flag, and switch
    /// toggles fold the lid reading in both directions.
    #[test]
    fn gesture_and_switch_signals_fold() {
        let mut m = TaskbarModel::default();
        assert!(!m.gesture_active && !m.lid_closed, "both start clear");
        m.apply(CompositorUpdate::GestureBegan);
        assert!(m.gesture_active, "began must mark the gesture in flight");
        m.apply(CompositorUpdate::GestureEnded);
        assert!(!m.gesture_active, "ended must clear the in-flight gesture");
        m.apply(CompositorUpdate::SwitchToggled { lid_closed: true });
        assert!(m.lid_closed, "lid close must reach the model");
        m.apply(CompositorUpdate::SwitchToggled { lid_closed: false });
        assert!(!m.lid_closed, "lid open must clear the model");
    }
}
