//! The `org.icedtea.Compositor` client. A worker thread receives the compositor's
//! signals (seeded by a `GetState` snapshot) and pushes [`CompositorUpdate`]s onto the
//! panel's loop thread through its inbox; [`CompositorProxy`] issues commands
//! (focus/close/workspace) with a separate blocking connection.

use async_channel::Sender;
use futures_util::StreamExt as _;
use icedtea_contract::{
    COMPOSITOR_BUS_NAME, COMPOSITOR_CONTRACT_VERSION, COMPOSITOR_PATH, Snapshot, WindowInfo,
    WindowUpdate, WorkspaceInfo,
};

use crate::taskbar::CompositorUpdate;

const COMPOSITOR_IFACE: &str = "org.icedtea.Compositor";

/// Spawn the signal worker. It seeds with `GetState`, then forwards every
/// `org.icedtea.Compositor` signal as a [`CompositorUpdate`] until the bus drops.
///
/// Review finding F7 -- why this kills the process instead of returning:
/// the compositor and this shell marshal the contract types independently,
/// so a compositor upgrade that changes a signature (`GetState`'s reply
/// gaining `WindowInfo.attention`) makes an *older* shell's typed call fail
/// with `zbus::Error::SignatureMismatch`. Logging and letting this worker
/// thread end leaves the panel running with a permanently empty, permanently
/// stale taskbar -- and, because the process is still alive and healthy as
/// far as the service manager can tell, `Restart=always` never fires and the
/// upgraded shell binary is never picked up. A non-zero exit is the only
/// signal that actually gets the session restarted onto the matching build,
/// so any error out of `run` takes the process down with it.
pub fn spawn(tx: Sender<CompositorUpdate>) {
    std::thread::spawn(move || {
        zbus::block_on(async move {
            if let Err(err) = run(tx).await {
                tracing::error!(
                    %err,
                    "org.icedtea.Compositor client failed; exiting so the service manager \
                     restarts this shell against the running compositor"
                );
                std::process::exit(1);
            }
        });
    });
}

/// Compare the compositor's advertised contract revision against the one
/// this binary was compiled with, returning a description of the mismatch
/// or `None` when they agree.
///
/// `remote` is `None` when the compositor does not expose the `Version`
/// property at all -- an older build, predating it -- which is itself a
/// mismatch worth naming.
///
/// A free function purely so this half of finding F7 is unit-testable: the
/// call path around it needs a live session bus and a running compositor.
pub fn version_mismatch(remote: Option<u32>, local: u32) -> Option<String> {
    match remote {
        Some(v) if v == local => None,
        Some(v) => Some(format!(
            "compositor speaks org.icedtea.Compositor contract v{v}, this shell was built for \
             v{local}; restart whichever side is stale"
        )),
        None => Some(format!(
            "compositor exposes no org.icedtea.Compositor `Version` property (a build older than \
             contract v{local}); restart it"
        )),
    }
}

async fn run(tx: Sender<CompositorUpdate>) -> zbus::Result<()> {
    let conn = zbus::Connection::session().await?;
    let proxy = zbus::Proxy::new(
        &conn,
        COMPOSITOR_BUS_NAME,
        COMPOSITOR_PATH,
        COMPOSITOR_IFACE,
    )
    .await?;

    // Read (never enforce) the contract revision first: a mismatch is
    // reported by name here, and the typed `GetState` immediately below is
    // what actually fails -- loudly, all the way to `spawn`'s
    // `process::exit` -- when the signatures have genuinely diverged. Not
    // fatal on its own: a missing property must not stop a shell whose
    // signatures still line up from running.
    let remote_version: Option<u32> = proxy.get_property("Version").await.ok();
    if let Some(complaint) = version_mismatch(remote_version, COMPOSITOR_CONTRACT_VERSION) {
        tracing::error!("{complaint}");
    }

    // Seed from the current snapshot before watching deltas. A
    // `SignatureMismatch` here is the upgrade-skew case finding F7 is
    // about: it propagates out of `run` and takes the process with it.
    let snapshot: Snapshot = proxy.call("GetState", &()).await?;
    let _ = tx.send(CompositorUpdate::Snapshot(snapshot)).await;

    // One stream for every org.icedtea.Compositor signal, dispatched by member name.
    let rule = zbus::MatchRule::builder()
        .msg_type(zbus::message::Type::Signal)
        .interface(COMPOSITOR_IFACE)?
        .path(COMPOSITOR_PATH)?
        .build();
    let mut stream = zbus::MessageStream::for_match_rule(rule, &conn, None).await?;

    while let Some(Ok(msg)) = stream.next().await {
        let member = msg.header().member().map(|m| m.as_str().to_string());
        let body = msg.body();
        let update = match member.as_deref() {
            Some("WindowOpened") => body
                .deserialize::<(u64, WindowInfo)>()
                .ok()
                .map(|(_, w)| CompositorUpdate::Opened(w)),
            Some("WindowClosed") => body
                .deserialize::<(u64, u32)>()
                .ok()
                .map(|(_, id)| CompositorUpdate::Closed(id)),
            Some("WindowUpdated") => body
                .deserialize::<(u64, u32, WindowUpdate)>()
                .ok()
                .map(|(_, id, update)| CompositorUpdate::Updated { id, update }),
            Some("WorkspaceSet") => body
                .deserialize::<(u64, u32, bool)>()
                .ok()
                .map(|(_, id, active)| CompositorUpdate::WorkspaceSet { id, active }),
            Some("WorkspaceList") => body
                .deserialize::<(u64, Vec<WorkspaceInfo>)>()
                .ok()
                .map(|(_, ws)| CompositorUpdate::WorkspaceList(ws)),
            // M7: gesture phases carry no payload beyond `seq` (the phase
            // is the member name); the switch signal carries the lid
            // reading. Bodies that fail to deserialize are dropped, the
            // same as every arm above.
            Some("GestureBegan") => body
                .deserialize::<u64>()
                .ok()
                .map(|_| CompositorUpdate::GestureBegan),
            Some("GestureEnded") => body
                .deserialize::<u64>()
                .ok()
                .map(|_| CompositorUpdate::GestureEnded),
            Some("SwitchToggled") => body
                .deserialize::<(u64, bool)>()
                .ok()
                .map(|(_, lid_closed)| CompositorUpdate::SwitchToggled { lid_closed }),
            _ => None,
        };
        if let Some(update) = update
            && tx.send(update).await.is_err()
        {
            break; // The panel exited.
        }
    }
    Ok(())
}

/// The taskbar's command surface — abstracted so a test can inject a recording
/// mock in place of the real D-Bus proxy.
pub trait CompositorCommands {
    fn focus_window(&self, id: u32);
    fn close_window(&self, id: u32);
    fn set_workspace(&self, id: u32);
}

/// Issues `org.icedtea.Compositor` commands from the panel's loop thread.
/// Method calls are no-reply and sub-millisecond, so a blocking connection here is fine.
pub struct CompositorProxy {
    conn: zbus::blocking::Connection,
}

impl CompositorProxy {
    pub fn new() -> zbus::Result<Self> {
        Ok(CompositorProxy {
            conn: zbus::blocking::Connection::session()?,
        })
    }
}

impl CompositorCommands for CompositorProxy {
    // zbus's #[interface] exposes Rust methods in PascalCase, so the wire
    // members are FocusWindow/CloseWindow/SetWorkspace (matching GetState).
    fn focus_window(&self, id: u32) {
        let _ = self.conn.call_method(
            Some(COMPOSITOR_BUS_NAME),
            COMPOSITOR_PATH,
            Some(COMPOSITOR_IFACE),
            "FocusWindow",
            &(id,),
        );
    }
    fn close_window(&self, id: u32) {
        let _ = self.conn.call_method(
            Some(COMPOSITOR_BUS_NAME),
            COMPOSITOR_PATH,
            Some(COMPOSITOR_IFACE),
            "CloseWindow",
            &(id,),
        );
    }
    fn set_workspace(&self, id: u32) {
        let _ = self.conn.call_method(
            Some(COMPOSITOR_BUS_NAME),
            COMPOSITOR_PATH,
            Some(COMPOSITOR_IFACE),
            "SetWorkspace",
            &(id,),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finding F7: the version comparison itself, isolated from the bus.
    #[test]
    fn version_mismatch_names_both_skew_directions_and_stays_quiet_when_matched() {
        assert_eq!(
            version_mismatch(
                Some(COMPOSITOR_CONTRACT_VERSION),
                COMPOSITOR_CONTRACT_VERSION
            ),
            None
        );
        let newer = version_mismatch(Some(99), 2).expect("a newer compositor is a mismatch");
        assert!(newer.contains("v99") && newer.contains("v2"), "{newer}");
        let older = version_mismatch(Some(1), 2).expect("an older compositor is a mismatch");
        assert!(older.contains("v1") && older.contains("v2"), "{older}");
        let absent = version_mismatch(None, 2).expect("no property at all is a mismatch");
        assert!(absent.contains("Version"), "{absent}");
    }
}
