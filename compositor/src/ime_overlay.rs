//! Preedit overlay geometry.
//!
//! When an IME is composing, the uncommitted text has to be visible
//! somewhere: this module decides *where* (a small box under the caret,
//! kept on screen) and *how wide* (shaped with the same cosmic-text stack
//! `text.rs` uses). It owns no scene nodes and touches no runtime — the
//! handler in `state.rs` does the FFI plumbing and delegates every
//! coordinate decision here, the same split `input_method.rs` documents
//! for the candidate popup.
//!
//! Two pieces, split so the pure half stays font-free:
//!
//! * [`measure_preedit`] shapes the text (needs the caller's `FontSystem`)
//!   into a width plus the cursor's x offset inside it.
//! * [`layout_overlay`] places that extent below the caret, reusing
//!   [`crate::input_method::place_below_clamped`] — never reimplementing
//!   clamping — and [`should_show`] pins the single hide rule: the overlay
//!   is visible if and only if the committed generation carries non-empty
//!   preedit text.

use icedtea_contract::Rectangle;

/// Height of the overlay band: the same 28px title band `text.rs` shapes
/// into, so preedit text reads at the same size as window titles.
pub const OVERLAY_HEIGHT: i32 = 28;

/// Horizontal inset of the text inside the overlay, matching
/// `TITLE_PAD_X` in `state.rs`.
pub const OVERLAY_PAD_X: i32 = 8;

/// The single hide rule: show if and only if the committed generation
/// carries preedit text that shapes to something visible. A commit-string,
/// a delete, a preedit-clear, and deactivation all converge here. Whitespace-
/// only preedit shapes to nothing (see `rasterize_title`'s `None` case) and
/// is treated as hidden, matching the show path which hides on `None`.
pub fn should_show(committed: &wlr::CommittedImeState) -> bool {
    committed.preedit.as_ref().is_some_and(|preedit| {
        // `rasterize_title` returns `None` for empty *or* whitespace-only
        // text with no glyphs, so a spaces-only composition must not claim
        // to be visible — it would vanish mid-compose.
        !preedit.text.trim().is_empty()
    })
}

/// The shaped size of a preedit string: its pixel width and the cursor's x
/// offset from the text's left edge.
pub struct PreeditMeasure {
    /// Shaped width of the (capped) text, excluding padding.
    pub width: i32,
    /// Shaped width of the text before the clamped cursor: where the caret
    /// bar renders, in `[0, width]`.
    pub cursor_x: i32,
}

/// Shape `text` into a [`PreeditMeasure`].
///
/// `cursor_end` is the IME's `cursor_end` byte index; it is clamped into
/// `text` (and onto a char boundary) so adversarial indices can never push
/// the caret outside the text. Text is capped with `text::cap_title`
/// before shaping — a client's preedit is otherwise unbounded, and shaping
/// it at full length on every keystroke would be the same CPU burn
/// finding 3 closed for titles.
pub fn measure_preedit(
    fonts: &mut cosmic_text::FontSystem,
    text: &str,
    cursor_end: i32,
) -> PreeditMeasure {
    // Capped before shaping: see the fn doc.
    let text = crate::text::cap_title(text);
    let width = shaped_width(fonts, text);
    let cursor = clamp_cursor(text, cursor_end);
    let cursor_x = shaped_width(fonts, &text[..cursor]);
    PreeditMeasure { width, cursor_x }
}

/// `cursor` clamped into `text` and onto a char boundary, so an
/// out-of-range or mid-codepoint index from the IME can never push the
/// caret outside the shaped text.
fn clamp_cursor(text: &str, cursor: i32) -> usize {
    let mut index = cursor.max(0).min(text.len() as i32) as usize;
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Shaped advance width of `text` at the overlay's font, `0` for empty
/// text. Same shaping call `text.rs` makes (sans-serif, advanced), at the
/// overlay's own size: the measure and the later raster must agree, or the
/// node is sized for text it does not hold.
fn shaped_width(fonts: &mut cosmic_text::FontSystem, text: &str) -> i32 {
    if text.is_empty() {
        return 0;
    }
    use cosmic_text::{Attrs, Buffer, Family, Metrics, Shaping};
    let font_px = (OVERLAY_HEIGHT as f32 * 0.55).max(8.0);
    let metrics = Metrics::new(font_px, OVERLAY_HEIGHT as f32);
    let mut buffer = Buffer::new(fonts, metrics);
    let mut buffer = buffer.borrow_with(fonts);
    // Unbounded width: this measures, it does not lay out for a band, so
    // nothing may wrap or clip.
    buffer.set_size(Some(10000.0), Some(OVERLAY_HEIGHT as f32));
    buffer.set_text(
        text,
        &Attrs::new().family(Family::SansSerif),
        Shaping::Advanced,
    );
    buffer.shape_until_scroll(true);
    buffer
        .layout_runs()
        .map(|run| run.line_w.ceil() as i32)
        .max()
        .unwrap_or(0)
        .max(0)
}

/// The live preedit overlay: the scene node holding the composing text
/// plus where it was placed. Tracked in a dedicated `Option` on `State`
/// — NOT in `input_popup_nodes`, whose lifecycle is the popup's, not the
/// composition's — and torn down with `remove_buffer`.
pub struct PreeditOverlay {
    /// The node `add_buffer_in_band` returned for the overlay text.
    pub node: wlr::BufferId,
    /// Where it was placed (output-local). Cached at placement because the
    /// crate exposes no buffer-position accessor; written only after a
    /// successful `set_buffer_position`, so it names where the node is,
    /// not just where it was meant to go.
    pub position: (i32, i32),
}

/// Where a measured preedit overlay goes, and how big it is.
pub struct OverlayLayout {
    /// Horizontal scene position (output-local), below the caret when it
    /// fits and clamped to the output.
    pub x: i32,
    /// Vertical scene position (output-local), below the caret when it fits
    /// and flipped above when it would spill off the bottom.
    pub y: i32,
    /// Full overlay width including both pads.
    pub width: i32,
    /// Always [`OVERLAY_HEIGHT`].
    pub height: i32,
    /// Caret offset from the text's left edge (add [`OVERLAY_PAD_X`] for
    /// the buffer coordinate).
    pub cursor_x: i32,
}

/// Place a measured preedit under `caret` (already translated into
/// output-local coordinates by the caller), clamped onto `output`.
/// `caret`'s own width is ignored; only its left edge and bottom matter,
/// exactly like the candidate popup's anchor.
pub fn layout_overlay(
    caret: Rectangle,
    measured: &PreeditMeasure,
    output: Option<Rectangle>,
) -> OverlayLayout {
    // Saturating: the shaped width is client-influenced (long preedit),
    // so extreme values must clamp, never panic or wrap.
    let width = measured
        .width
        .saturating_add(2 * OVERLAY_PAD_X)
        .max(2 * OVERLAY_PAD_X);
    let popup = Rectangle {
        x: 0,
        y: 0,
        width,
        height: OVERLAY_HEIGHT,
    };
    let (x, y) = crate::input_method::place_below_clamped(caret, popup, output);
    OverlayLayout {
        x,
        y,
        width,
        height: OVERLAY_HEIGHT,
        cursor_x: measured.cursor_x.max(0).min(measured.width.max(0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed(
        preedit: Option<&str>,
        commit: Option<&str>,
        delete: (u32, u32),
    ) -> wlr::CommittedImeState {
        wlr::CommittedImeState {
            preedit: preedit.map(|text| wlr::ImePreedit {
                text: text.to_string(),
                cursor_begin: 0,
                cursor_end: text.len() as i32,
            }),
            commit_text: commit.map(str::to_string),
            delete_before: delete.0,
            delete_after: delete.1,
        }
    }

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle {
        Rectangle {
            x,
            y,
            width: w,
            height: h,
        }
    }

    #[test]
    fn no_preedit_never_shows() {
        assert!(
            !should_show(&committed(None, None, (0, 0))),
            "nothing committed, nothing to show"
        );
    }

    #[test]
    fn empty_preedit_never_shows() {
        assert!(
            !should_show(&committed(Some(""), None, (0, 0))),
            "an empty preedit shapes to nothing"
        );
    }

    #[test]
    fn commit_string_without_preedit_hides() {
        assert!(
            !should_show(&committed(None, Some("日本"), (0, 0))),
            "a commit-string carries no composing text"
        );
    }

    #[test]
    fn delete_without_preedit_hides() {
        assert!(
            !should_show(&committed(None, None, (2, 0))),
            "a delete carries no composing text"
        );
    }

    #[test]
    fn non_empty_preedit_shows() {
        assert!(
            should_show(&committed(Some("nihon"), None, (0, 0))),
            "composing text must be visible"
        );
    }

    #[test]
    fn short_text_sits_below_the_caret() {
        let measured = PreeditMeasure {
            width: 60,
            cursor_x: 60,
        };
        let at = layout_overlay(
            rect(100, 228, 2, 16),
            &measured,
            Some(rect(0, 0, 1920, 1080)),
        );
        assert_eq!((at.x, at.y), (100, 244), "left-aligned, just below");
        assert_eq!(at.width, 60 + 2 * OVERLAY_PAD_X);
        assert_eq!(at.height, OVERLAY_HEIGHT);
        assert_eq!(at.cursor_x, 60);
    }

    #[test]
    fn spill_flips_above_the_caret() {
        let measured = PreeditMeasure {
            width: 60,
            cursor_x: 0,
        };
        let at = layout_overlay(
            rect(100, 1040, 2, 16),
            &measured,
            Some(rect(0, 0, 1920, 1080)),
        );
        assert_eq!(
            (at.x, at.y),
            (100, 1040 - OVERLAY_HEIGHT),
            "flipped above, never off screen"
        );
    }

    #[test]
    fn slides_left_off_the_right_edge() {
        let measured = PreeditMeasure {
            width: 120,
            cursor_x: 0,
        };
        let at = layout_overlay(
            rect(1900, 200, 2, 16),
            &measured,
            Some(rect(0, 0, 1920, 1080)),
        );
        assert_eq!(at.x, 1920 - (120 + 2 * OVERLAY_PAD_X));
        assert_eq!(at.y, 216);
    }

    #[test]
    fn cursor_clamps_inside_the_text() {
        let mut fonts = cosmic_text::FontSystem::new();
        let m = measure_preedit(&mut fonts, "nihon", 99);
        assert!(m.width > 0, "latin preedit must shape to a positive width");
        assert_eq!(
            m.cursor_x, m.width,
            "a cursor past the end clamps to the text's end"
        );
        let m = measure_preedit(&mut fonts, "nihon", -3);
        assert_eq!(m.cursor_x, 0, "a negative cursor clamps to the start");
    }

    #[test]
    fn empty_text_measures_zero() {
        let mut fonts = cosmic_text::FontSystem::new();
        let m = measure_preedit(&mut fonts, "", 0);
        assert_eq!(m.width, 0);
        assert_eq!(m.cursor_x, 0);
    }

    #[test]
    fn cursor_clamps_on_mid_codepoint() {
        let mut fonts = cosmic_text::FontSystem::new();
        // "日本" is 6 bytes, 2 chars; index 1 lands inside the first char.
        let m = measure_preedit(&mut fonts, "日本", 1);
        assert!(
            m.cursor_x >= 0 && m.cursor_x <= m.width,
            "mid-codepoint cursor must clamp inside [0, width]"
        );
    }

    #[test]
    fn layout_with_no_output_falls_back() {
        let measured = PreeditMeasure {
            width: 60,
            cursor_x: 60,
        };
        let at = layout_overlay(rect(100, 228, 2, 16), &measured, None);
        assert_eq!((at.x, at.y), (100, 244));
    }

    #[test]
    fn layout_clamps_wild_cursor_x() {
        let at = layout_overlay(
            rect(100, 228, 2, 16),
            &PreeditMeasure {
                width: 60,
                cursor_x: 999,
            },
            Some(rect(0, 0, 1920, 1080)),
        );
        assert_eq!(at.cursor_x, 60);
    }
}
