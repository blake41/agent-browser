//! Click actionability transaction.
//!
//! Wraps the CDP click sequence with the same pre-flight checks Playwright performs:
//! 1. Resolve element to an objectId (existing infra).
//! 2. `DOM.scrollIntoViewIfNeeded` so the element has a chance to be visible.
//! 3. `DOM.getContentQuads` for transform-aware geometry.
//! 4. `Page.getLayoutMetrics` to clip the chosen point against the layout viewport.
//! 5. `Runtime.callFunctionOn` hit-test (`document.elementFromPoint`) with shadow-DOM
//!    awareness, accepting the target itself, descendants, ancestors, and label-for
//!    relationships.
//! 6. Only on success returns an [`ActionablePoint`] for the caller to dispatch input on.
//!
//! On failure, returns an [`ActionabilityError`] with diagnostic context. Notably,
//! callers must NOT silently fall through to a JS click — the existing `click_js` verb
//! remains the explicit unverified escape hatch.
//!
//! Modeled on browser-use's pragmatic actionability check
//! (`default_action_watchdog.py:590-680`) but kept minimal — no toolbar/header
//! special-casing, no ancestor-coverage geometry test, no retry loop. That's a
//! follow-up.

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;

use super::cdp::client::CdpClient;
use super::cdp::types::*;
use super::element::{quad_center, resolve_element_object_id, RefMap};

/// A point that has been verified actionable for a click.
#[derive(Debug, Clone)]
pub struct ActionablePoint {
    pub x: f64,
    pub y: f64,
    pub session_id: String,
    /// The objectId of the verified target element. Held so callers can re-target
    /// via JS if they choose; not currently used by the dispatch path.
    pub target_object_id: String,
}

/// Reasons why the actionability transaction can fail.
#[derive(Debug)]
pub enum ActionabilityError {
    /// `DOM.getContentQuads` returned no quads — the element has no visible area
    /// (display:none, detached, zero-size). This is distinct from "out of viewport".
    NoVisibleArea { target_desc: String },
    /// The chosen point is outside the layout viewport (offscreen, below the fold,
    /// horizontally clipped). Even after `scrollIntoViewIfNeeded`.
    NotInViewport {
        target_desc: String,
        x: f64,
        y: f64,
        viewport: (f64, f64, f64, f64), // (x, y, width, height)
    },
    /// The hit-test at the chosen point returned a different element than the target,
    /// and that element is not a descendant/ancestor/associated-label of the target.
    /// This is the case Playwright's "intercepted" error catches — modal overlays,
    /// stacking-context bugs, stale coordinates after layout shift.
    Intercepted {
        target_desc: String,
        hit_desc: String,
        x: f64,
        y: f64,
    },
    /// An underlying CDP call failed. Wraps the original string error.
    Cdp(String),
}

impl fmt::Display for ActionabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionabilityError::NoVisibleArea { target_desc } => {
                write!(
                    f,
                    "actionability: target {} has no visible area (DOM.getContentQuads returned no quads)",
                    target_desc
                )
            }
            ActionabilityError::NotInViewport {
                target_desc,
                x,
                y,
                viewport,
            } => {
                let (vx, vy, vw, vh) = *viewport;
                write!(
                    f,
                    "actionability: target {} center ({:.1}, {:.1}) is outside layout viewport ({:.1}, {:.1}, {:.1}x{:.1}) — element is offscreen even after scrollIntoViewIfNeeded",
                    target_desc, x, y, vx, vy, vw, vh
                )
            }
            ActionabilityError::Intercepted {
                target_desc,
                hit_desc,
                x,
                y,
            } => {
                write!(
                    f,
                    "actionability: target {} is covered by <{}> at ({:.1}, {:.1}) — the click was intercepted and would land on that element instead (overlay/modal/stacking issue). Dismiss or interact with the covering element first, or use click-js to bypass.",
                    target_desc, hit_desc, x, y
                )
            }
            ActionabilityError::Cdp(msg) => {
                write!(f, "actionability: CDP error: {}", msg)
            }
        }
    }
}

impl From<String> for ActionabilityError {
    fn from(s: String) -> Self {
        ActionabilityError::Cdp(s)
    }
}

impl From<ActionabilityError> for String {
    fn from(e: ActionabilityError) -> Self {
        e.to_string()
    }
}

/// JS predicate executed via `Runtime.callFunctionOn` with `this` bound to the
/// target element. Accepts the click if the hit-test returns the target itself,
/// a descendant, an ancestor, or an associated `<label for>` element.
///
/// Walks up the composed tree (shadow DOM) so a hit on a slotted element is
/// interpreted in the host's tree.
pub(crate) const HIT_TEST_JS: &str = r#"function(x, y) {
    let hit = document.elementFromPoint(x, y);
    while (hit && hit.getRootNode && hit.getRootNode() !== document && hit.getRootNode().host) {
        hit = hit.getRootNode().host;
    }
    const desc = el => {
        if (!el) return '(none)';
        const tag = el.tagName ? el.tagName.toLowerCase() : '?';
        const id = el.id ? '#' + el.id : '';
        let cls = '';
        if (el.className && typeof el.className === 'string' && el.className.trim()) {
            cls = '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.');
        }
        return tag + id + cls;
    };
    if (!hit) return { ok: false, hit_desc: '(none)', target_desc: desc(this) };
    const target_desc = desc(this);
    const hit_desc = desc(hit);
    if (hit === this || this.contains(hit) || hit.contains(this)) {
        return { ok: true, hit_desc, target_desc };
    }
    if (this.tagName === 'LABEL' && this.htmlFor) {
        const ref = document.getElementById(this.htmlFor);
        if (ref && (ref === hit || ref.contains(hit) || hit.contains(ref))) {
            return { ok: true, hit_desc, target_desc };
        }
    }
    if (hit.tagName === 'LABEL' && hit.htmlFor) {
        const ref = document.getElementById(hit.htmlFor);
        if (ref && (ref === this || ref.contains(this) || this.contains(ref))) {
            return { ok: true, hit_desc, target_desc };
        }
    }
    if (this.labels) {
        for (const l of Array.from(this.labels)) {
            if (l === hit || l.contains(hit) || hit.contains(l)) {
                return { ok: true, hit_desc, target_desc };
            }
        }
    }
    return { ok: false, hit_desc, target_desc };
}"#;

/// Resolve a click target to a fully verified actionable point.
///
/// On success returns an [`ActionablePoint`] the caller can pass straight into
/// the existing `Input.dispatchMouseEvent` sequence.
///
/// On failure returns an [`ActionabilityError`] — the caller should propagate
/// it; do NOT silently fall back to a different click strategy.
pub async fn resolve_actionable_point(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<ActionablePoint, ActionabilityError> {
    // 1. Resolve to an objectId. Existing infra handles ref-or-selector + frame routing.
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    // 2. Scroll into view. Mirror resolve_element_center_quads: ignore failures —
    //    element may already be in view, and an error here doesn't preclude success.
    let _ = client
        .send_command_typed::<_, Value>(
            "DOM.scrollIntoViewIfNeeded",
            &DomScrollIntoViewIfNeededParams {
                backend_node_id: None,
                node_id: None,
                object_id: Some(object_id.clone()),
            },
            Some(&effective_session_id),
        )
        .await;

    // 3. Geometry. getContentQuads accounts for CSS transforms (virtualized lists,
    //    rotated/scaled elements) where getBoxModel lies.
    let quads_result: DomGetContentQuadsResult = client
        .send_command_typed(
            "DOM.getContentQuads",
            &DomGetContentQuadsParams {
                backend_node_id: None,
                node_id: None,
                object_id: Some(object_id.clone()),
            },
            Some(&effective_session_id),
        )
        .await?;

    // Fetch a target descriptor up-front so error variants carry context.
    let target_desc = describe_target(client, &effective_session_id, &object_id).await;

    let quad = match quads_result.quads.first() {
        Some(q) => q,
        None => {
            return Err(ActionabilityError::NoVisibleArea { target_desc });
        }
    };

    let (cx, cy) = quad_center(quad)?;

    // 4. Viewport clip. Use cssLayoutViewport so the coordinates match the CSS
    //    pixel space content quads return.
    let metrics: Value = client
        .send_command_no_params("Page.getLayoutMetrics", Some(&effective_session_id))
        .await?;

    let viewport = metrics
        .get("cssLayoutViewport")
        .or_else(|| metrics.get("layoutViewport"))
        .ok_or_else(|| {
            ActionabilityError::Cdp(
                "Page.getLayoutMetrics: missing cssLayoutViewport/layoutViewport".to_string(),
            )
        })?;

    let vx = viewport
        .get("pageX")
        .and_then(|v| v.as_f64())
        .or_else(|| viewport.get("clientX").and_then(|v| v.as_f64()))
        .unwrap_or(0.0);
    let vy = viewport
        .get("pageY")
        .and_then(|v| v.as_f64())
        .or_else(|| viewport.get("clientY").and_then(|v| v.as_f64()))
        .unwrap_or(0.0);
    let vw = viewport
        .get("clientWidth")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| ActionabilityError::Cdp("layoutViewport missing clientWidth".to_string()))?;
    let vh = viewport
        .get("clientHeight")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            ActionabilityError::Cdp("layoutViewport missing clientHeight".to_string())
        })?;

    // Quads are in CSS pixels relative to the layout viewport's pageX/pageY origin.
    // The Page.getLayoutMetrics origin (vx, vy) is the scroll offset; clipping
    // against [vx, vx+vw] × [vy, vy+vh] is what Playwright effectively does.
    if cx < vx || cx > vx + vw || cy < vy || cy > vy + vh {
        return Err(ActionabilityError::NotInViewport {
            target_desc,
            x: cx,
            y: cy,
            viewport: (vx, vy, vw, vh),
        });
    }

    // 5. Hit-test. We pass cx,cy as JS arguments; `this` is the target element.
    //    The predicate returns { ok, hit_desc, target_desc } by value.
    let hit: EvaluateResult = client
        .send_command_typed(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: HIT_TEST_JS.to_string(),
                object_id: Some(object_id.clone()),
                arguments: Some(vec![
                    CallArgument {
                        value: Some(Value::from(cx)),
                        object_id: None,
                    },
                    CallArgument {
                        value: Some(Value::from(cy)),
                        object_id: None,
                    },
                ]),
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    if let Some(ex) = hit.exception_details {
        return Err(ActionabilityError::Cdp(format!(
            "hit-test threw: {}",
            ex.text
        )));
    }

    let hit_value = hit.result.value.unwrap_or(Value::Null);
    let ok = hit_value
        .get("ok")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let hit_desc = hit_value
        .get("hit_desc")
        .and_then(|v| v.as_str())
        .unwrap_or("(unknown)")
        .to_string();
    // Prefer the JS-side target_desc if present (it's computed at hit-test time
    // and won't have drifted since); fall back to the eagerly-computed one.
    let resolved_target_desc = hit_value
        .get("target_desc")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or(target_desc);

    if !ok {
        return Err(ActionabilityError::Intercepted {
            target_desc: resolved_target_desc,
            hit_desc,
            x: cx,
            y: cy,
        });
    }

    Ok(ActionablePoint {
        x: cx,
        y: cy,
        session_id: effective_session_id,
        target_object_id: object_id,
    })
}

/// Best-effort target descriptor for diagnostics. Errors here are swallowed —
/// the descriptor is purely cosmetic.
async fn describe_target(client: &CdpClient, session_id: &str, object_id: &str) -> String {
    let result: Result<EvaluateResult, String> = client
        .send_command_typed(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: r#"function() {
                    const tag = this.tagName ? this.tagName.toLowerCase() : '?';
                    const id = this.id ? '#' + this.id : '';
                    let cls = '';
                    if (this.className && typeof this.className === 'string' && this.className.trim()) {
                        cls = '.' + this.className.trim().split(/\s+/).slice(0, 2).join('.');
                    }
                    return tag + id + cls;
                }"#
                .to_string(),
                object_id: Some(object_id.to_string()),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(session_id),
        )
        .await;

    match result {
        Ok(r) => r
            .result
            .value
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "(unknown)".to_string()),
        Err(_) => "(unknown)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_no_visible_area() {
        let e = ActionabilityError::NoVisibleArea {
            target_desc: "button#submit.primary".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("button#submit.primary"), "got: {}", s);
        assert!(s.contains("no visible area"), "got: {}", s);
        assert!(s.contains("DOM.getContentQuads"), "got: {}", s);
    }

    #[test]
    fn display_not_in_viewport() {
        let e = ActionabilityError::NotInViewport {
            target_desc: "a#deep-link".to_string(),
            x: 1500.0,
            y: 2400.0,
            viewport: (0.0, 0.0, 1280.0, 800.0),
        };
        let s = e.to_string();
        assert!(s.contains("a#deep-link"), "got: {}", s);
        assert!(s.contains("1500.0"), "got: {}", s);
        assert!(s.contains("2400.0"), "got: {}", s);
        assert!(s.contains("1280.0"), "got: {}", s);
        assert!(s.contains("offscreen"), "got: {}", s);
    }

    #[test]
    fn display_intercepted() {
        let e = ActionabilityError::Intercepted {
            target_desc: "button#confirm".to_string(),
            hit_desc: "div.modal-overlay".to_string(),
            x: 640.0,
            y: 360.0,
        };
        let s = e.to_string();
        assert!(s.contains("button#confirm"), "got: {}", s);
        assert!(s.contains("div.modal-overlay"), "got: {}", s);
        assert!(s.contains("640.0"), "got: {}", s);
        assert!(s.contains("360.0"), "got: {}", s);
        assert!(s.contains("intercepted"), "got: {}", s);
    }

    #[test]
    fn display_cdp() {
        let e = ActionabilityError::Cdp("websocket closed".to_string());
        assert_eq!(e.to_string(), "actionability: CDP error: websocket closed");
    }

    #[test]
    fn from_string_wraps_into_cdp() {
        let e: ActionabilityError = "boom".to_string().into();
        match e {
            ActionabilityError::Cdp(s) => assert_eq!(s, "boom"),
            other => panic!("expected Cdp, got {:?}", other),
        }
    }

    #[test]
    fn into_string_uses_display() {
        let e = ActionabilityError::Cdp("boom".to_string());
        let s: String = e.into();
        assert_eq!(s, "actionability: CDP error: boom");
    }

    #[test]
    fn hit_test_js_is_static_str() {
        // Compile-time: HIT_TEST_JS is a &'static str.
        let _: &'static str = HIT_TEST_JS;
        // Sanity: it's non-empty and starts with the expected function preamble.
        assert!(HIT_TEST_JS.starts_with("function(x, y) {"));
        // Sanity: the composed-tree walk + label-for branches are present.
        assert!(HIT_TEST_JS.contains("getRootNode"));
        assert!(HIT_TEST_JS.contains("htmlFor"));
        assert!(HIT_TEST_JS.contains("this.labels"));
        assert!(HIT_TEST_JS.contains("elementFromPoint"));
    }

    #[test]
    fn call_argument_assembles_for_hit_test() {
        // Sanity: the CallArgument shape we use for cx,cy serializes the way CDP
        // expects ({ value: <number> }). If this drifts, the hit-test will pass
        // undefined into the predicate and silently break.
        let arg = CallArgument {
            value: Some(serde_json::Value::from(123.5_f64)),
            object_id: None,
        };
        let v = serde_json::to_value(&arg).unwrap();
        assert_eq!(v.get("value").and_then(|x| x.as_f64()), Some(123.5));
        assert!(v.get("objectId").is_none(), "should skip empty objectId");
    }
}
