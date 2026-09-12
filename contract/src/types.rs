use serde::{Deserialize, Serialize};
use zvariant::Type;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Type,
)]
#[zvariant(signature = "u")]
pub struct WindowId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[zvariant(signature = "(iiii)")]
pub struct Rectangle {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rectangle {
    pub fn contains(self, x: i32, y: i32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct WindowInfo {
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
    pub attention: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct WorkspaceInfo {
    pub id: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct Snapshot {
    pub seq: u64,
    pub windows: Vec<WindowInfo>,
    pub workspaces: Vec<WorkspaceInfo>,
    pub active_workspace: u32,
    /// Whether the seat cursor currently shows an image. M7: fed by the
    /// compositor from `wlr::Runtime::cursor_state` (an image is applied on
    /// the first pointer motion; before that the cursor is `Hidden`).
    /// `#[serde(default)]` (false) so a pre-M7 snapshot still decodes.
    #[serde(default)]
    pub cursor_visible: bool,
    /// The cursor's last-known position in output-logical coordinates, or
    /// `None` before the first pointer motion. M7: the compositor's model
    /// mirror of the crate cursor. `None` decodes from a missing field.
    #[serde(default)]
    pub cursor_pos: Option<(i32, i32)>,
    /// Whether any touch point is currently down. M7: fed by the
    /// compositor from `wlr::Runtime::touch_state`. `#[serde(default)]`
    /// (false) so a pre-M7 snapshot still decodes.
    #[serde(default)]
    pub touch_active: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
pub struct WindowUpdate {
    pub title: Option<String>,
    pub geometry: Option<Rectangle>,
    pub workspace: Option<u32>,
    pub maximized: Option<bool>,
    pub minimized: Option<bool>,
    pub fullscreen: Option<bool>,
    pub focused: Option<bool>,
    pub mapped: Option<bool>,
    pub attention: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct AltTabState {
    pub active: bool,
    pub entries: Vec<WindowId>,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct Palette {
    pub background: String,
    pub foreground: String,
    pub accent: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct Appearance {
    pub bar_position: String,
    pub bar_height: i32,
    pub corner_radius: i32,
    pub snap_gap: i32,
    pub palette: Palette,
    pub wallpaper: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisplayConfig {
    pub name: String,
    pub enabled: bool,
    pub width: i32,
    pub height: i32,
    pub refresh_mhz: i32,
    pub x: i32,
    pub y: i32,
    pub scale: f64,
    pub transform: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_window() -> WindowInfo {
        WindowInfo {
            id: WindowId(1),
            app_id: "org.gnome.Calculator".into(),
            title: "Calculator".into(),
            pid: 1234,
            workspace: 0,
            geometry: Rectangle {
                x: 100,
                y: 100,
                width: 400,
                height: 300,
            },
            maximized: false,
            minimized: false,
            fullscreen: false,
            focused: true,
            attention: false,
        }
    }

    fn sample_snapshot() -> Snapshot {
        Snapshot {
            seq: 7,
            windows: vec![sample_window()],
            workspaces: vec![
                WorkspaceInfo {
                    id: 0,
                    name: "1".into(),
                },
                WorkspaceInfo {
                    id: 1,
                    name: "2".into(),
                },
            ],
            active_workspace: 0,
            cursor_visible: true,
            cursor_pos: Some((10, 20)),
            touch_active: false,
        }
    }

    /// Review finding M4: the Plan-3 shell is not Rust and must marshal
    /// these signatures explicitly, so the wire encoding is part of the
    /// contract, not an implementation detail. Nothing used to fail if the
    /// `option-as-array` feature were dropped from `zvariant` in
    /// `contract/Cargo.toml` -- the round-trip tests below still pass while
    /// the wire format silently changes from `as`/`au`/`ab` to a
    /// variant-based `Option` encoding. These assertions lock it.
    #[test]
    fn wire_signatures_are_locked() {
        use zvariant::Type;
        assert_eq!(
            WindowUpdate::SIGNATURE.to_string(),
            "(asa(iiii)auabababababab)"
        );
        assert_eq!(Appearance::SIGNATURE.to_string(), "(siii(sss)as)");
        assert_eq!(WindowId::SIGNATURE.to_string(), "u");
        assert_eq!(Rectangle::SIGNATURE.to_string(), "(iiii)");
        assert_eq!(WindowInfo::SIGNATURE.to_string(), "(ussuu(iiii)bbbbb)");
        assert_eq!(WorkspaceInfo::SIGNATURE.to_string(), "(us)");
        assert_eq!(
            Snapshot::SIGNATURE.to_string(),
            "(ta(ussuu(iiii)bbbbb)a(us)uba(ii)b)"
        );
        // `index: usize` marshals as `t` (u64) on 64-bit targets.
        assert_eq!(AltTabState::SIGNATURE.to_string(), "(baut)");
    }

    #[test]
    fn snapshot_json_round_trip() {
        let s = sample_snapshot();
        let json = serde_json::to_string(&s).unwrap();
        let back: Snapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    /// M7: a pre-M7 snapshot (no cursor/touch fields) still decodes, with
    /// the new fields at their `#[serde(default)]` values -- the
    /// additive-only contract discipline: an older compositor's `GetState`
    /// reply never fails a newer shell's parse.
    #[test]
    fn pre_m7_snapshot_json_still_decodes() {
        let legacy = serde_json::json!({
            "seq": 7,
            "windows": [],
            "workspaces": [],
            "active_workspace": 0,
        });
        let back: Snapshot = serde_json::from_value(legacy).unwrap();
        assert!(!back.cursor_visible);
        assert_eq!(back.cursor_pos, None);
        assert!(!back.touch_active);
    }

    #[test]
    fn window_info_zvariant_round_trip() {
        let w = sample_window();
        let ctxt = zvariant::serialized::Context::new_dbus(zvariant::LE, 0);
        let encoded = zvariant::to_bytes(ctxt, &w).unwrap();
        let decoded: WindowInfo = encoded.deserialize().unwrap().0;
        assert_eq!(w, decoded);
    }

    #[test]
    fn window_update_default_is_all_none() {
        let u = WindowUpdate::default();
        assert!(u.title.is_none() && u.geometry.is_none() && u.workspace.is_none());
        assert!(u.maximized.is_none() && u.minimized.is_none() && u.fullscreen.is_none());
        assert!(u.focused.is_none());
        assert!(u.mapped.is_none());
        assert!(u.attention.is_none());
    }

    #[test]
    fn window_update_option_round_trip() {
        let ctxt = zvariant::serialized::Context::new_dbus(zvariant::LE, 0);
        let some = WindowUpdate {
            title: Some("New title".into()),
            geometry: Some(Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 80,
            }),
            workspace: Some(2),
            maximized: Some(true),
            minimized: Some(false),
            fullscreen: Some(true),
            focused: Some(false),
            mapped: Some(true),
            attention: Some(true),
        };
        let none = WindowUpdate::default();

        for update in [&some, &none] {
            let encoded = zvariant::to_bytes(ctxt, update).unwrap();
            let decoded: WindowUpdate = encoded.deserialize().unwrap().0;
            assert_eq!(update, &decoded);
        }

        // option-as-array encodes None as a 0-length array: in the all-None
        // WindowUpdate the first field (`title`, an `as`) is an empty array.
        let none_bytes = zvariant::to_bytes(ctxt, &none).unwrap();
        assert_eq!(&none_bytes[..4], &[0, 0, 0, 0]);
        // A bare Option<u32> is `au`: None is only the 4-byte length prefix.
        let empty = zvariant::to_bytes(ctxt, &None::<u32>).unwrap();
        assert_eq!(empty.len(), 4);
        assert!(empty.iter().all(|b| *b == 0));
        // Some(7) adds one u32 element: 4-byte length + 4-byte value.
        let filled = zvariant::to_bytes(ctxt, &Some(7u32)).unwrap();
        assert_eq!(filled.len(), 8);
    }
}
