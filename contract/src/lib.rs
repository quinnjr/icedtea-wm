pub mod clipboard;
pub mod event;
pub mod notifications;
pub mod types;

pub const COMPOSITOR_BUS_NAME: &str = "org.icedtea.Compositor";
pub const COMPOSITOR_PATH: &str = "/org/icedtea/Compositor";

/// The `org.icedtea.Compositor` wire-contract revision, exposed on the bus as
/// the interface's read-only `Version` property and compiled into every
/// client that links this crate.
///
/// Review finding F7: the shell and the compositor marshal these types
/// independently, so a compositor upgrade that changes a *signature* --
/// `WindowInfo` gaining the `attention` bit turned `GetState`'s reply from
/// `...bbbb` into `...bbbbb` -- makes an older shell's `GetState` fail with
/// `SignatureMismatch` and nothing else. Bumping this on every such change
/// gives the mismatch a name: the shell reads the property at connect and
/// says which side is stale, instead of only reporting a deserialization
/// error from one call.
///
/// A property, not a method argument, so adding it changes no method
/// signature and cannot itself become the next incompatibility.
///
/// * `1` -- the pre-A2 contract.
/// * `2` -- A2 batch 2: `WindowInfo.attention` / `WindowUpdate.attention`.
/// * `3` -- M8 IME depth: `Snapshot.ime_active` (the IME indicator bit).
/// * `4` -- M8 remainder: `Snapshot.keyboard_layout` +
///   `Snapshot.shortcuts_inhibited` (the keyboard-layout and
///   shortcuts-inhibit indicator fields).
/// * `5` -- M7 pointer/cursor/touch: `Snapshot.cursor_visible` /
///   `cursor_pos` / `touch_active`, plus the `GestureBegan` / `GestureEnded`
///   / `SwitchToggled` signals.
pub const COMPOSITOR_CONTRACT_VERSION: u32 = 5;

pub use clipboard::{CLIP_BUS_NAME, CLIP_PATH, ClipEntry, ClipKind};
pub use event::{Event, SeqEvent};
pub use notifications::{
    CloseReason, IconSource, NOTIF_BUS_NAME, NOTIF_PATH, Notification, NotificationAction, Urgency,
};
pub use types::*;
